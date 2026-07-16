"""Generate the registered non-binding diagnostics for the G5 Stage-1 screen.

This command is deliberately separate from the gate: it replays only frozen
checkpoint inference, writes descriptive fidelity/cost evidence, and never
changes the registered pass/cut rule.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import torch
from scipy.stats import pearsonr
from torch.profiler import ProfilerActivity, profile

from src.data.artifacts import load_benchmark
from src.data.grounding import build_grounding_pool
from src.eval.ego_fidelity import (
    degree_calibration_curve,
    slot_adjacency_clustering_correlation,
    slot_recall_at_k,
)
from src.eval.graph_metrics import strip_self_loops
from src.experiments.g5_stage1 import (
    assemble_matched_global_rd_graph,
    select_matched_global_rd_rows,
)
from src.model.egostitch import EgoStitchStage1
from src.model.egostitch.imagine import SlotSet
from src.model.egostitch.stitch import sinkhorn_plan
from src.score_universe import (
    ScoresArtifact,
    _autocast_context,
    _load_checkpoint,
    load_scores,
)
from src.train_egostitch import EgoConfig, assemble_egostitch_data, load_config


@dataclass(frozen=True)
class NodeCache:
    """CPU-resident frozen per-node outputs used by all diagnostics."""

    node_ids: list[str]
    h: torch.Tensor
    pi: torch.Tensor
    mult: torch.Tensor
    adj: torch.Tensor
    d_hat_raw: torch.Tensor
    proj: torch.Tensor


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _encode_nodes(
    model: EgoStitchStage1,
    matrix: torch.Tensor,
    node_index: Mapping[str, int],
    node_ids: Sequence[str],
    grounding_rows: torch.Tensor,
    *,
    device: torch.device,
    amp: str,
    batch_nodes: int,
) -> NodeCache:
    rows_all = torch.tensor([node_index[node] for node in node_ids], dtype=torch.int64)
    fields: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("h", "pi", "mult", "adj", "d_hat_raw", "proj")
    }
    for start in range(0, len(node_ids), batch_nodes):
        local = torch.arange(start, min(start + batch_nodes, len(node_ids)))
        rows = rows_all[local]
        x = matrix[rows].to(device)
        ground_x = matrix[grounding_rows[local]].to(device)
        with torch.inference_mode(), _autocast_context(device, amp):
            enc = model.encode_nodes(x, ground_x)
            projected = model.proj(x)
        values = {
            "h": enc.slots.h,
            "pi": enc.slots.pi,
            "mult": enc.slots.mult,
            "adj": enc.slots.adj,
            "d_hat_raw": enc.tok.d_hat_raw,
            "proj": projected,
        }
        for name, value in values.items():
            fields[name].append(value.detach().float().cpu())
    return NodeCache(
        node_ids=list(node_ids),
        **{name: torch.cat(parts) for name, parts in fields.items()},
    )


def _slots(cache: NodeCache, rows: torch.Tensor, device: torch.device) -> SlotSet:
    h = cache.h[rows].to(device)
    pi = cache.pi[rows].to(device)
    filler = torch.zeros_like(pi)
    return SlotSet(
        h=h,
        pi=pi,
        mult=cache.mult[rows].to(device),
        gate=filler,
        pointer=filler.unsqueeze(-1),
        adj=cache.adj[rows].to(device),
    )


def _channel_matrix(
    model: EgoStitchStage1,
    cache: NodeCache,
    pairs: Sequence[tuple[str, str]],
    s0: np.ndarray,
    *,
    device: torch.device,
    batch_pairs: int,
) -> np.ndarray:
    index = {node: row for row, node in enumerate(cache.node_ids)}
    u_rows = torch.tensor([index[u] for u, _ in pairs], dtype=torch.int64)
    v_rows = torch.tensor([index[v] for _, v in pairs], dtype=torch.int64)
    output: list[torch.Tensor] = []
    for start in range(0, len(pairs), batch_pairs):
        end = min(start + batch_pairs, len(pairs))
        u = u_rows[start:end]
        v = v_rows[start:end]
        base = torch.from_numpy(s0[start:end].astype(np.float32)).to(device)
        channels = torch.zeros((end - start, 3), dtype=torch.float32, device=device)
        self_mask = u == v
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            nonself = torch.nonzero(~self_mask, as_tuple=False).squeeze(-1)
            if nonself.numel():
                slots_u = _slots(cache, u[nonself], device)
                slots_v = _slots(cache, v[nonself], device)
                plan = sinkhorn_plan(
                    slots_u.h,
                    slots_v.h,
                    slots_u.pi,
                    slots_v.pi,
                    slots_u.mult,
                    slots_v.mult,
                    eps=model.config.sinkhorn_eps,
                    iters=model.config.sinkhorn_iters,
                    tau=model.config.sinkhorn_tau,
                )
                out = model.decision.channels(
                    slots_u,
                    slots_v,
                    plan,
                    cache.proj[u[nonself]].to(device),
                    cache.proj[v[nonself]].to(device),
                    cache.d_hat_raw[u[nonself]].to(device),
                    cache.d_hat_raw[v[nonself]].to(device),
                )
                channels[nonself] = torch.stack([out["s1"], out["s2"], out["s2_aa"]], 1)
            self_rows = torch.nonzero(self_mask, as_tuple=False).squeeze(-1)
            if self_rows.numel():
                out = model.decision.self_channels(
                    _slots(cache, u[self_rows], device),
                    cache.proj[u[self_rows]].to(device),
                    cache.d_hat_raw[u[self_rows]].to(device),
                )
                channels[self_rows] = torch.stack([out["s1"], out["s2"], out["s2_aa"]], 1)
        output.append(torch.column_stack((base, channels)).cpu())
    return torch.cat(output).numpy()


def _correlation_report(values: np.ndarray) -> dict[str, object]:
    labels = ["s0", "s1", "s2", "s2_aa"]
    matrix = np.full((values.shape[1], values.shape[1]), np.nan, dtype=np.float64)
    for i in range(values.shape[1]):
        for j in range(values.shape[1]):
            if np.std(values[:, i]) > 0 and np.std(values[:, j]) > 0:
                matrix[i, j] = float(pearsonr(values[:, i], values[:, j]).statistic)
    return {"labels": labels, "pearson": matrix.tolist(), "n": int(len(values))}


def _split_summary(artifact: ScoresArtifact, self_rows: np.ndarray) -> dict[str, object]:
    probs = artifact.probs()

    def row(mask: np.ndarray) -> dict[str, object]:
        labels = artifact.label[mask]
        known = labels >= 0
        return {
            "n": int(mask.sum()),
            "n_labeled": int(known.sum()),
            "positive_rate": float(labels[known].mean()) if known.any() else None,
            "mean_probability": float(probs[mask].mean()) if mask.any() else None,
        }

    return {"self": row(self_rows), "nonself": row(~self_rows)}


def _measure_flops_and_wall(
    action: Callable[[], None], *, device: torch.device, batch_size: int
) -> tuple[float, float]:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    _synchronize(device)
    with profile(activities=activities, with_flops=True) as prof:
        action()
        _synchronize(device)
    flops = float(sum(event.flops or 0 for event in prof.key_averages())) / batch_size
    samples: list[float] = []
    for _ in range(5):
        _synchronize(device)
        started = time.perf_counter()
        action()
        _synchronize(device)
        samples.append((time.perf_counter() - started) / batch_size)
    return flops, float(np.median(samples))


def _projection_variance(
    checkpoint: Path, matrix: torch.Tensor, rows: torch.Tensor, *, device: torch.device
) -> dict[str, object]:
    loaded, family, _ = _load_checkpoint(checkpoint)
    if family != "egostitch" or not isinstance(loaded, EgoStitchStage1):
        raise ValueError(f"{checkpoint} is not an EgoStitch checkpoint")
    payload = cast(
        dict[str, object], torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    epoch = int(cast(int, payload["epoch"]))
    loaded.to(device)
    with torch.inference_mode():
        projected = loaded.proj(matrix[rows].to(device)).float()
    return {
        "checkpoint": checkpoint.name,
        "epoch": epoch,
        "mean_feature_variance": float(projected.var(dim=0, unbiased=False).mean().cpu()),
        "n_nodes": len(rows),
    }


def _finite_json(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    return value


def generate_reports(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    """Generate fidelity and cost payloads from frozen artifacts."""
    cfg: EgoConfig = load_config(args.config)
    data = assemble_egostitch_data(cfg)
    model_raw, family, checkpoint_id = _load_checkpoint(args.checkpoint)
    if family != "egostitch" or not isinstance(model_raw, EgoStitchStage1):
        raise ValueError("diagnostics require an EgoStitch checkpoint")
    model = model_raw.to(args.device)
    model.set_density_ratio(1.0)
    candidate = load_scores(args.candidate)
    s0_candidate = load_scores(args.s0_universe)
    metadata = json.loads(args.run_metadata.read_text(encoding="utf-8"))
    if checkpoint_id != metadata.get("checkpoint_id"):
        raise ValueError("candidate diagnostics checkpoint does not match run metadata")
    if list(candidate.pairs()) != list(s0_candidate.pairs()):
        raise ValueError("candidate and fresh-s0 universes are not row-aligned")

    benchmark = load_benchmark(
        args.data_root / "benchmark_2025_neurips",
        args.strategy,
        exclude_nodes=frozenset(cfg.data.expected_missing_features),
    )
    train_nodes = data.train_nodes
    train_ground = torch.from_numpy(data.grounding_index)
    train_cache = _encode_nodes(
        model,
        data.f0,
        data.node_index,
        train_nodes,
        train_ground,
        device=args.device,
        amp=args.amp,
        batch_nodes=args.batch_nodes,
    )

    test_nodes = candidate.node_ids
    test_matrix = np.asarray(
        data.f0[[data.node_index[node] for node in test_nodes]].numpy(), dtype=np.float32
    )
    test_pool = build_grounding_pool(
        test_matrix, test_nodes, n_ground=model.config.n_ground, cache_path=None
    )
    test_ground = torch.tensor(
        [[data.node_index[neighbor] for neighbor in test_pool[node]] for node in test_nodes],
        dtype=torch.int64,
    )
    test_cache = _encode_nodes(
        model,
        data.f0,
        data.node_index,
        test_nodes,
        test_ground,
        device=args.device,
        amp=args.amp,
        batch_nodes=args.batch_nodes,
    )

    train_graph = data.target_builder.graph
    test_graph = benchmark.split.test_graph
    train_recall = slot_recall_at_k(train_cache.h, train_graph, train_nodes, train_cache.proj)
    test_recall = slot_recall_at_k(test_cache.h, test_graph, test_nodes, test_cache.proj)

    nonself_target = strip_self_loops(test_graph).number_of_edges()
    selection = select_matched_global_rd_rows(
        candidate.probs(),
        candidate.u_idx,
        candidate.v_idx,
        target_edges=nonself_target,
        reference_edges=nonself_target,
    )
    assembled = assemble_matched_global_rd_graph(
        list(candidate.pairs()),
        candidate.probs(),
        candidate.u_idx,
        candidate.v_idx,
        selection,
        test_nodes,
    )
    assembled_simple = strip_self_loops(assembled)
    density = cast(dict[str, object], candidate.meta["density_calibration"])
    rho_hat = float(cast(float, density["rho_hat_eval"]))
    rho_train = float(metadata["rho_train"])
    expected_degree = test_cache.d_hat_raw.numpy().astype(np.float64) * (rho_hat / rho_train)
    realized_degree = np.array(
        [assembled_simple.degree(node) for node in test_nodes], dtype=np.float64
    )

    val_channels = _channel_matrix(
        model,
        train_cache,
        data.val_pairs,
        data.s0.lookup(data.val_pairs),
        device=args.device,
        batch_pairs=args.batch_pairs,
    )
    self_rows = candidate.u_idx == candidate.v_idx
    projection_rows = torch.tensor(
        [data.node_index[node] for node in test_nodes[: min(256, len(test_nodes))]],
        dtype=torch.int64,
    )
    projection_trajectory = [
        _projection_variance(args.checkpoint, data.f0, projection_rows, device=args.device),
        _projection_variance(args.last_checkpoint, data.f0, projection_rows, device=args.device),
    ]
    fidelity: dict[str, object] = {
        "degree_calibration_curve": degree_calibration_curve(expected_degree, realized_degree),
        "slot_recall_at_k_train": asdict(train_recall),
        "slot_recall_at_k_test": asdict(test_recall),
        "slot_adjacency_clustering_correlation": {
            "train": slot_adjacency_clustering_correlation(
                train_cache.adj, train_cache.pi, train_nodes, nx.clustering(train_graph)
            ),
            "test": slot_adjacency_clustering_correlation(
                test_cache.adj, test_cache.pi, test_nodes, nx.clustering(test_graph)
            ),
        },
        "s_channel_correlation": _correlation_report(val_channels),
        "self_nonself": _split_summary(candidate, self_rows),
        "self_loop_rate": {
            "selected_self_loops": nx.number_of_selfloops(assembled),
            "n_nodes": len(test_nodes),
            "rate_per_node": nx.number_of_selfloops(assembled) / len(test_nodes),
        },
        "proj_variance_trajectory": projection_trajectory,
    }

    node_batch = min(args.profile_nodes, len(test_nodes))
    node_rows = torch.tensor(
        [data.node_index[node] for node in test_nodes[:node_batch]], dtype=torch.int64
    )
    node_ground = test_ground[:node_batch]

    def node_action() -> None:
        with torch.inference_mode(), _autocast_context(args.device, args.amp):
            model.encode_nodes(
                data.f0[node_rows].to(args.device), data.f0[node_ground].to(args.device)
            )

    nonself_indices = np.flatnonzero(~self_rows)[: args.profile_pairs]
    pair_u = torch.from_numpy(candidate.u_idx[nonself_indices].astype(np.int64))
    pair_v = torch.from_numpy(candidate.v_idx[nonself_indices].astype(np.int64))
    pair_slots_u = _slots(test_cache, pair_u, args.device)
    pair_slots_v = _slots(test_cache, pair_v, args.device)
    pair_s0 = torch.from_numpy(s0_candidate.logit[nonself_indices]).to(args.device)

    def pair_action() -> None:
        with torch.inference_mode(), torch.autocast(device_type=args.device.type, enabled=False):
            plan = sinkhorn_plan(
                pair_slots_u.h,
                pair_slots_v.h,
                pair_slots_u.pi,
                pair_slots_v.pi,
                pair_slots_u.mult,
                pair_slots_v.mult,
                eps=model.config.sinkhorn_eps,
                iters=model.config.sinkhorn_iters,
                tau=model.config.sinkhorn_tau,
            )
            model.decision(
                pair_s0,
                pair_slots_u,
                pair_slots_v,
                plan,
                test_cache.proj[pair_u].to(args.device),
                test_cache.proj[pair_v].to(args.device),
                test_cache.d_hat_raw[pair_u].to(args.device),
                test_cache.d_hat_raw[pair_v].to(args.device),
            )

    node_flops, node_wall = _measure_flops_and_wall(
        node_action, device=args.device, batch_size=node_batch
    )
    pair_flops, pair_wall = _measure_flops_and_wall(
        pair_action, device=args.device, batch_size=len(nonself_indices)
    )
    score_profile = candidate.meta.get("score_profile")
    if not isinstance(score_profile, dict) or float(score_profile.get("wall_seconds", 0)) <= 0:
        raise ValueError("candidate artifact lacks measured score_profile; rescore it")
    total_flops = node_flops * len(test_nodes) + pair_flops * len(candidate.logit)
    cost: dict[str, object] = {
        "harmonization_rounds": 0,
        "measurement": "torch.profiler supported-op FLOPs; median of five synchronized passes",
        "per_node_cached": {
            "flops": node_flops,
            "wall_seconds": node_wall,
            "profile_batch": node_batch,
        },
        "per_pair_marginal": {
            "flops": pair_flops,
            "wall_seconds": pair_wall,
            "profile_batch": len(nonself_indices),
        },
        "candidate_universe": {
            "flops": total_flops,
            "wall_seconds": float(score_profile["wall_seconds"]),
            "rows": len(candidate.logit),
            "unique_nodes": len(test_nodes),
            "flops_method": "per-node plus per-pair extrapolation",
            "wall_method": score_profile.get("measurement"),
        },
    }
    return cast(dict[str, object], _finite_json(fidelity)), cast(
        dict[str, object], _finite_json(cost)
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the frozen-diagnostics CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--last-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--s0-universe", type=Path, required=True)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--strategy", default="breadth_first")
    parser.add_argument("--fidelity-output", type=Path, required=True)
    parser.add_argument("--cost-output", type=Path, required=True)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    parser.add_argument("--amp", choices=("off", "bf16"), default="bf16")
    parser.add_argument("--batch-nodes", type=int, default=64)
    parser.add_argument("--batch-pairs", type=int, default=512)
    parser.add_argument("--profile-nodes", type=int, default=32)
    parser.add_argument("--profile-pairs", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Generate and atomically publish the two registered JSON reports."""
    args = build_parser().parse_args(argv)
    for path in (
        args.checkpoint,
        args.last_checkpoint,
        args.candidate,
        args.s0_universe,
        args.run_metadata,
        args.config,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    fidelity, cost = generate_reports(args)
    for path, payload in ((args.fidelity_output, fidelity), (args.cost_output, cost)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

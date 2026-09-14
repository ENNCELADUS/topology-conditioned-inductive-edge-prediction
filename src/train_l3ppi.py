"""Three-stage, resumable L3-PPI reproduction on the existing node holdout."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from src.baselines.l3ppi import L3PPI, build_l3ppi, path_number_loss
from src.baselines.l3ppi_features import encode_nodes, score_cached
from src.data.l3_patterns import L3PatternCache, batch_patterns
from src.data.packed_features import load_packed_manifest
from src.data.pairs import NegativeSampler
from src.data.training_sampler import enumerate_edge_stream
from src.data.val_region import val_ball_union_universe
from src.eval.checkpoint_selection import (
    SELECTION_RULE,
    CheckpointCandidate,
    TopologyValidationMetrics,
    select_checkpoint,
)
from src.eval.val_topology import build_val_topology_reference, val_region_topology_metrics
from src.score_universe import _checkpoint_id, _load_val_region_split

logger = logging.getLogger(__name__)


def write_json(path: Path, value: object) -> None:
    """Replace a JSON artifact atomically after fully serializing it."""
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def save_state(path: Path, state: object) -> None:
    """Publish a checkpoint atomically so interruption preserves the prior epoch."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    """Copy weights, including on CPU where detach alone would alias live tensors."""
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def converged(losses: list[float]) -> bool:
    """Paper appendix C: loss range below 0.05 over ten consecutive epochs."""
    return len(losses) >= 10 and max(losses[-10:]) - min(losses[-10:]) < 0.05


def temperature(epoch: int, epochs: int) -> float:
    """Geometrically anneal the joint-stage Concrete temperature from 1 to 0.1."""
    return float(0.1 ** ((epoch - 1) / max(1, epochs - 1)))


def gather_objects(value: object, world: int) -> list[Any]:
    """Collect small per-rank epoch artifacts, including exact RNG recovery state."""
    if world == 1:
        return [value]
    result: list[Any] = [None] * world
    dist.all_gather_object(result, value)
    return result


def rng_state(device: torch.device) -> dict[str, torch.Tensor]:
    """Capture the CPU and this rank's CUDA random streams."""
    result = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        result["cuda"] = torch.cuda.get_rng_state(device)
    return result


def restore_rng(state: dict[str, torch.Tensor], device: torch.device) -> None:
    """Restore stochastic prompt gates and dropout at an epoch boundary."""
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def load_features(
    model: L3PPI,
    config: dict[str, Any],
    nodes: list[str],
    backbone_id: str,
    rank: int,
    world: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Share an attribute-only cache for the same frozen encoder and node universe."""
    path = Path(config["feature_cache"])
    metadata = {
        "backbone_checkpoint_id": backbone_id,
        "nodes": nodes,
        "max_length": model.max_length,
    }
    if path.exists():
        stored = torch.load(path, map_location="cpu", weights_only=True)
        if stored["metadata"] != metadata:
            raise ValueError("endpoint cache belongs to a different encoder or node universe")
        matrix = stored["features"]
        if matrix.shape != (len(nodes), model.feature_dim) or not torch.isfinite(matrix).all():
            raise ValueError("invalid L3-PPI endpoint cache")
        return dict(zip(nodes, matrix.float().unbind(), strict=True))
    local = encode_nodes(
        model,
        Path(config["pack_dir"]),
        nodes[rank::world],
        device,
        int(config["encode_batch_size"]),
    )
    merged = {key: value for part in gather_objects(local, world) for key, value in part.items()}
    if rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_state(
            path, {"metadata": metadata, "features": torch.stack([merged[n] for n in nodes])}
        )
    if world > 1:
        dist.barrier()
    return merged


def evaluate(
    model: L3PPI,
    features: Mapping[str, torch.Tensor],
    pairs: list[tuple[str, str]],
    device: torch.device,
    rank: int,
    world: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Gather contiguous scoring slices back into the shared canonical row order."""
    start, stop = len(pairs) * rank // world, len(pairs) * (rank + 1) // world
    local = score_cached(model, features, pairs[start:stop], device)
    parts = gather_objects(local, world)
    return tuple(np.concatenate([part[i] for part in parts]) for i in range(3))


def train_epoch(
    model: L3PPI,
    wrapped: nn.Module,
    optimizer: torch.optim.Optimizer,
    rows: list[tuple[str, str, int]],
    features: Mapping[str, torch.Tensor],
    patterns: L3PatternCache,
    phase: str,
    config: dict[str, Any],
    tau: float,
    rank: int,
    world: int,
    device: torch.device,
) -> dict[str, float]:
    """One globally sampled epoch with correctly normalized uneven DDP tails."""
    model.train()
    global_batch = int(config["batch_size"])
    totals = torch.zeros(5, dtype=torch.float64, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    for start in range(0, len(rows), global_batch):
        stop = min(start + global_batch, len(rows))
        local = rows[start + rank : stop : world]
        count = len(local)
        batch = local or rows[:1]  # Empty ranks still participate in backward with zero weight.
        labels = torch.tensor([y for _, _, y in batch], dtype=torch.float32, device=device)
        if phase == "surrogate":
            graphs = [patterns.extract(a, b) for a, b, _ in batch]
            inputs = batch_patterns(graphs, features, device)
            logits = wrapped(*inputs, len(graphs))
            path_loss = torch.zeros_like(logits)
            soft = hard = torch.zeros_like(logits)
        else:
            a = torch.stack([features[u] for u, _, _ in batch]).to(device)
            b = torch.stack([features[v] for _, v, _ in batch]).to(device)
            logits, probabilities, activations = wrapped(
                a, b, all_open=phase == "prompt", temperature=tau
            )
            path_loss = path_number_loss(probabilities, labels, float(config["gamma"]))
            if phase == "prompt":
                path_loss = torch.zeros_like(logits)
            soft, hard = probabilities.sum(-1).mean(-1), activations.sum(-1).mean(-1)
        task = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=logits.new_tensor(5.0), reduction="none"
        )
        # Match B0 weighted_pair_bce: divide BCE by weight mass, not row count.
        weight_mass = sum(1 + 4 * y for _, _, y in rows[start:stop])
        loss = (
            world
            * bool(count)
            * (
                task.sum() / weight_mass
                + float(config["path_weight"]) * path_loss.sum() / (stop - start)
            )
        )
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite {phase} loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if count:
            totals += (
                torch.stack(
                    [(1 + 4 * labels).sum(), task.sum(), path_loss.sum(), soft.sum(), hard.sum()]
                )
                .detach()
                .double()
            )
        if rank == 0 and start % (global_batch * 250) == 0:
            logger.info("%s rows=%d/%d loss=%.6f", phase, stop, len(rows), float(loss.detach()))
    if world > 1:
        dist.all_reduce(totals)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = torch.tensor(time.perf_counter() - start_time, device=device)
    if world > 1:
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    mass, task_sum, path_sum, soft_sum, hard_sum = totals.cpu().tolist()
    result = {
        "train_task_loss": task_sum / mass,
        "train_path_loss": path_sum / len(rows),
        "train_soft_paths": soft_sum / len(rows),
        "train_hard_paths": hard_sum / len(rows),
        "train_loss": task_sum / mass + float(config["path_weight"]) * path_sum / len(rows),
    }
    result.update(train_seconds=float(elapsed), pairs_per_second=len(rows) / float(elapsed))
    peak = torch.tensor(
        torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
        device=device,
    )
    if world > 1:
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    result["peak_gpu_gib"] = float(peak)
    return result


def candidate_from_row(row: dict[str, Any]) -> CheckpointCandidate:
    """Restore a serialized five-metric candidate without changing selection semantics."""
    return CheckpointCandidate(
        row["epoch"], row["auprc"], TopologyValidationMetrics(**row["topology"])
    )


def run(
    config: dict[str, Any],
    stage: str,
    resume: bool,
    rank: int,
    world: int,
    device: torch.device,
) -> None:
    """Pretrain once or tune a trial, publishing only terminal stage artifacts."""
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    last_path = output / "last.pt"
    if (output / "complete.json").exists():
        if not resume:
            raise FileExistsError(f"completed output already exists: {output}")
        if not (output / "best.pt").exists() or (output / "failure.json").exists():
            raise ValueError("inconsistent completion artifacts")
        logger.info("stage already complete: %s", output)
        return
    if last_path.exists() and not resume:
        raise FileExistsError(f"use --resume for interrupted output: {output}")
    restored = (
        torch.load(last_path, map_location="cpu", weights_only=True) if last_path.exists() else None
    )
    if restored and (
        restored["config"] != config
        or restored["stage"] != stage
        or restored["world_size"] != world
    ):
        raise ValueError("resume requires the same config, stage and world size")
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    model = build_l3ppi(config["model"]["config"]).to(device)
    backbone = torch.load(config["backbone_checkpoint"], map_location="cpu", weights_only=True)
    if backbone["model_family"] != "v3_1":
        raise ValueError("L3-PPI requires the published B0 v3_1 encoder")
    if backbone["model_config"] != config["model"]["config"]["backbone_config"]:
        raise ValueError("L3-PPI backbone_config differs from the actual B0 checkpoint")
    model.load_backbone(backbone)
    backbone_id = _checkpoint_id(backbone["model_state"])
    provenance = {
        "backbone_checkpoint": config["backbone_checkpoint"],
        "backbone_checkpoint_id": backbone_id,
        "backbone_epoch": backbone["epoch"],
    }
    split = _load_val_region_split(Path(config["data_root"]), "breadth_first")
    position = load_packed_manifest(Path(config["pack_dir"])).node_index()
    train_nodes = sorted(split.train_nodes & position.keys())
    graph = split.build_training_graph()
    missing = split.train_nodes - position.keys()
    # The two known featureless training nodes are isolated in the active split.
    # Do not silently delete a structural intermediate if that changes.
    if any(graph.degree(node) for node in missing):
        raise ValueError("training L3 graph has connected nodes without intrinsic features")
    nodes = sorted(set(train_nodes) | split.v_val)
    features = load_features(model, config, nodes, backbone_id, rank, world, device)
    patterns = L3PatternCache(graph, set(split.train_nodes))
    positives = sorted(
        (u, v) for u, v in split.training_positives if u in position and v in position
    )
    sampler = NegativeSampler(train_nodes, dict(graph.degree()), frozenset(positives))
    if not positives:
        raise ValueError("no training positives")
    if stage == "trial":
        pretrained = torch.load(
            config["surrogate_checkpoint"], map_location="cpu", weights_only=True
        )
        if pretrained["stage"] != "surrogate" or pretrained["provenance"] != provenance:
            raise ValueError("surrogate is not pretrained on this frozen B0 encoder")
        model.surrogate.load_state_dict(
            {
                k.removeprefix("surrogate."): v
                for k, v in pretrained["model_state"].items()
                if k.startswith("surrogate.")
            }
        )
        model.initialize_prompt(torch.stack([features[n] for n in train_nodes]).to(device))
    if restored:
        model.load_state_dict(restored["model_state"])
    # Identical initialization, independent per-rank gates/dropout; restored later at resume.
    torch.manual_seed(int(config["seed"]) + rank)
    history: list[dict[str, Any]] = restored["history"] if restored else []
    candidates = [candidate_from_row(row) for row in restored["candidates"]] if restored else []
    optim_config = config["optim"]
    phases = ["surrogate"] if stage == "surrogate" else ["prompt", "joint"]
    reference = build_val_topology_reference(split) if stage == "trial" else None
    union = val_ball_union_universe(split) if stage == "trial" else None
    val_pairs, val_labels = list(split.val_cls_pairs), np.asarray(split.val_cls_labels)
    if rank == 0:
        write_json(output / "config.json", config)
        write_json(
            output / "split.json",
            {
                "train_nodes": len(train_nodes),
                "V_val_nodes": len(split.v_val),
                "training_positive_pairs": len(positives),
                "excluded_featureless_nodes": sorted(missing),
                "split_seed": split.params.split_seed,
                "root": split.region_seeds,
            },
        )
        (output / "checkpoints").mkdir(exist_ok=True)
        if resume:
            for path in output.glob("failure*.json"):
                path.unlink()
        write_json(output / "status.json", {"status": "running", "stage": stage})
        sample_rows = enumerate_edge_stream(
            positives, sampler, negative_ratio=5, seed=config["seed"], epoch=1, rank=0, world_size=1
        )[:512]
        started = time.perf_counter()
        sample = [patterns.extract(a, b) for a, b, _ in sample_rows]
        write_json(
            output / "profile.json",
            {
                "sample_pairs": len(sample),
                "extraction_seconds": time.perf_counter() - started,
                "paths_mean": float(np.mean([p.num_paths for p in sample])),
                "paths_max": max(p.num_paths for p in sample),
                "union_nodes_max": max(len(p.nodes) for p in sample),
                "union_edges_max": max(p.edge_index.shape[1] // 2 for p in sample),
                "zero_path_pairs": sum(p.num_paths == 0 for p in sample),
                "truncated_paths": False,
            },
        )
    if world > 1:
        dist.barrier()
    for phase in phases:
        if restored and phases.index(phase) < phases.index(restored["phase"]):
            continue
        model.surrogate.requires_grad_(phase == "surrogate")
        model.prompt.requires_grad_(phase != "surrogate")
        model.gate.requires_grad_(phase == "joint")
        trainable: nn.Module = model.surrogate if phase == "surrogate" else model
        wrapped = (
            DistributedDataParallel(
                trainable,
                device_ids=[device.index] if device.type == "cuda" else None,
                broadcast_buffers=False,
            )
            if world > 1
            else trainable
        )
        optimizer = torch.optim.Adam(
            [p for p in trainable.parameters() if p.requires_grad],
            lr=float(optim_config["surrogate_lr"] if phase == "surrogate" else optim_config["lr"]),
            weight_decay=0.0,
        )
        epochs = int(optim_config[f"{phase}_epochs"])
        best_loss, bad_epochs, first = math.inf, 0, 1
        best_phase_state: dict[str, torch.Tensor] | None = None
        phase_losses: list[float] = []
        if restored and phase == restored["phase"]:
            optimizer.load_state_dict(restored["optimizer"])
            restore_rng(restored["rng"][rank], device)
            first = restored["epoch"] + 1
            best_loss, bad_epochs = restored["best_loss"], restored["bad_epochs"]
            best_phase_state = restored["best_phase_state"]
            phase_losses = restored["phase_losses"]
            if restored["phase_done"]:
                first = epochs + 1
        for epoch in range(first, epochs + 1):
            rows = enumerate_edge_stream(
                positives,
                sampler,
                negative_ratio=5,
                seed=config["seed"],
                epoch=epoch,
                rank=0,
                world_size=1,
            )
            tau = temperature(epoch, epochs) if phase == "joint" else 1.0
            metrics: dict[str, Any] = train_epoch(
                model,
                wrapped,
                optimizer,
                rows,
                features,
                patterns,
                phase,
                config,
                tau,
                rank,
                world,
                device,
            )
            metrics.update(phase=phase, epoch=epoch, temperature=tau)
            phase_losses.append(metrics["train_loss"])
            if phase == "joint":
                assert union is not None and reference is not None
                validation_started = time.perf_counter()
                logits, soft, hard = evaluate(model, features, val_pairs, device, rank, world)
                metrics.update(
                    val_task_loss=float(
                        F.binary_cross_entropy_with_logits(
                            torch.from_numpy(logits), torch.tensor(val_labels, dtype=torch.float32)
                        )
                    ),
                    val_auprc=float(average_precision_score(val_labels, logits)),
                    val_auroc=float(roc_auc_score(val_labels, logits)),
                    val_soft_paths=float(soft.mean()),
                    val_hard_paths=float(hard.mean()),
                )
                topo_pairs = [
                    (reference.nodes[int(u)], reference.nodes[int(v)])
                    for u, v in zip(union.u_idx, union.v_idx, strict=True)
                ]
                topo_logits, _, _ = evaluate(model, features, topo_pairs, device, rank, world)
                topology = (
                    val_region_topology_metrics(
                        u_idx=union.u_idx,
                        v_idx=union.v_idx,
                        logits=topo_logits,
                        reference=reference,
                    )
                    if rank == 0
                    else None
                )
                if world > 1:
                    value = [topology]
                    dist.broadcast_object_list(value, src=0)
                    topology = value[0]
                assert topology is not None
                candidate = CheckpointCandidate(epoch, metrics["val_auprc"], topology.metrics)
                candidates.append(candidate)
                metrics.update(
                    topology=dataclasses.asdict(topology),
                    validation_seconds=time.perf_counter() - validation_started,
                )
                if rank == 0:
                    save_state(
                        output / "checkpoints" / f"epoch-{epoch:04d}.pt",
                        {
                            "model_state": cpu_state(model),
                            "model_family": "l3ppi",
                            "model_config": config["model"]["config"],
                            "epoch": epoch,
                            "stage": stage,
                            "phase": phase,
                            "provenance": provenance,
                            "selection_rule": SELECTION_RULE,
                            "val_threshold_transfer": {
                                "n_val": len(split.v_val),
                                "threshold": topology.threshold,
                            },
                            "topology": dataclasses.asdict(topology),
                            "val_auprc": metrics["val_auprc"],
                        },
                    )
                watched = metrics["val_task_loss"]
            else:
                watched = metrics["train_loss"]
            if not math.isfinite(watched):
                raise ValueError("non-finite watched loss")
            if watched < best_loss:
                best_loss, bad_epochs = watched, 0
                if phase != "joint":
                    best_phase_state = cpu_state(trainable)
            else:
                bad_epochs += 1
            done = (
                bad_epochs >= int(optim_config["patience"])
                if phase == "joint"
                else converged(phase_losses)
            ) or epoch == epochs
            metrics["converged"] = converged(phase_losses) if phase != "joint" else None
            history.append(metrics)
            rng = gather_objects(rng_state(device), world)
            if rank == 0:
                save_state(
                    last_path,
                    {
                        "config": config,
                        "model_state": cpu_state(model),
                        "optimizer": optimizer.state_dict(),
                        "stage": stage,
                        "phase": phase,
                        "epoch": epoch,
                        "phase_done": done,
                        "world_size": world,
                        "rng": rng,
                        "best_loss": best_loss,
                        "bad_epochs": bad_epochs,
                        "best_phase_state": best_phase_state,
                        "phase_losses": phase_losses,
                        "history": history,
                        "candidates": [dataclasses.asdict(c) for c in candidates],
                    },
                )
                # Rebuild from committed epoch history: no duplicate rows after a retry.
                (output / "metrics.jsonl").write_text(
                    "".join(json.dumps(r, allow_nan=False) + "\n" for r in history)
                )
                write_json(
                    output / "status.json",
                    {"status": "running", "stage": stage, "phase": phase, "epoch": epoch},
                )
                logger.info("epoch %s", json.dumps(metrics, allow_nan=False))
            if world > 1:
                dist.barrier()
            if done:
                break
        if phase != "joint":
            if best_phase_state is None:
                raise RuntimeError(f"{phase} produced no finite checkpoint")
            trainable.load_state_dict(best_phase_state)
        del wrapped, optimizer
        restored = None
    if rank == 0:
        if stage == "trial":
            selected = select_checkpoint(candidates)
            if selected is None:
                raise RuntimeError("no selectable L3-PPI checkpoint")
            payload = torch.load(
                output / "checkpoints" / f"epoch-{selected.epoch:04d}.pt", weights_only=True
            )
            write_json(
                output / "selection.json",
                {
                    "selected_epoch": selected.epoch,
                    "epoch": selected.epoch,
                    "val_auprc": payload["val_auprc"],
                    "topology": payload["topology"],
                    "selection_rule": SELECTION_RULE,
                    "candidates": [dataclasses.asdict(c) for c in candidates],
                },
            )
        else:
            payload = {
                "model_state": cpu_state(model),
                "model_family": "l3ppi",
                "model_config": config["model"]["config"],
                "stage": stage,
                "provenance": provenance,
                "epoch": min(history, key=lambda row: row["train_loss"])["epoch"],
                "trained_epochs": len(history),
                "converged": converged([r["train_loss"] for r in history]),
            }
        save_state(output / "best.pt", payload)
        write_json(
            output / "complete.json",
            {
                "status": "published",
                "stage": stage,
                "selected_epoch": payload["epoch"],
                "test_evaluated": False,
            },
        )
        write_json(output / "status.json", {"status": "published", "stage": stage})
    if world > 1:
        dist.barrier()


def main() -> None:
    """Enter via hpc/run.sh; CPU execution is available for meaningful smoke tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--stage", choices=("surrogate", "trial"), default="trial")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        run(config, args.stage, args.resume, rank, world, device)
    except Exception as exc:
        path = Path(config["output_dir"])
        path.mkdir(parents=True, exist_ok=True)
        # Refusing a second launch must not poison an already published run.
        if not (path / "complete.json").exists():
            write_json(path / f"failure_rank{rank}.json", {"error": str(exc)})
            if rank == 0:
                write_json(path / "failure.json", {"error": str(exc)})
        raise
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

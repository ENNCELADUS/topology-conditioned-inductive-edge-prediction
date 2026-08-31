r"""Dump Full-Ego Pooled Oracle KD teacher targets over the trainer's official rows.

CLI (full-row KD rework -- replaces the sampled anchor-context stream):

    python -m src.distill.teacher_targets \\
        --config configs/b0_v31_breadth_first.yaml \\
        --checkpoint outputs/.../best.pt \\
        --output outputs/distill/kd_row_targets_breadth_first --device cuda

Row universe: this module re-derives the trainer's own row lists
(`src.train_b0.assemble_data` + `_training_rows`/`_val_cls_rows`) from the
same training YAML rather than sampling anchor/context pairs -- the training
block covers every official training row exactly once, in the trainer's
exact row order (row_id == array position), so the artifact is directly
joinable against it by position; the validation block covers the official
V_val classification rows. See `src.distill.artifacts` for the resulting
`kd_row_targets_v1` format.

Query-edge masking (structural, not reimplemented here): the teacher is the
`full_ego_oracle` generator, and its `stitch`
(`src/model/egostitch/generator/full_oracle/generator.py`) already excludes
the queried edge from both endpoints' structural context per pair -- the
endpoints' neighbor channels drop the partner (``keep = values !=
partners[owners]``) and the adjacency skips the query edge (``keep = found &
~query_edge``). A positive training edge is therefore never visible in its
own structural context, and because this masking is applied per pair inside
`stitch`, not by mutating the underlying graph, the per-node encode-once
cache (`encode_all_nodes`) -- which carries only row identity, never
edge-masked state -- stays masking-safe: the same cached node state is
correctly reusable across every row touching that node.

Legality (never bind anything but a truth graph free of V_val-internal
edges): the truth graph is `ValRegionSplit.build_training_graph()` --
loopless training positives over every train node, so V_val nodes appear
only through their cross-boundary neighbors, never through a V_val-internal
edge, by construction. `assert_training_side_only` hard-refuses if a
benchmark test-split node reaches the truth graph or the node universe, or if
the truth graph somehow carries a V_val-internal edge.
`assert_no_val_internal_training_rows` additionally hard-refuses a *training*
row with both endpoints inside V_val (the validation block is exempt: those
rows are the held-out V_val classification rows, scored only for
validation-side diagnostics).

Architecture: this module reuses `score_universe.py`'s lower-level,
universe-agnostic building blocks directly (`_load_checkpoint`,
`_install_oracle_context`, `_file_sha256`, `_shard_range`,
`_oracle_truth_graph_sha256`) and mirrors its per-node encode-once cache +
`build_pair_context_from_states`/`score_pair_context` pattern
(``_score_egostitch_e2e``) rather than duplicating either. Every pair is
scored fp32 with no autocast on the pair pass, matching the
``egostitch_e2e_pair_fp32_v1`` contract `_score_egostitch_e2e` itself uses --
this artifact's `teacher_logit` must be numerically consistent with what a
training-time rescoring of the same checkpoint would produce, not merely
close.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from numpy.typing import NDArray

from src.data.features import FeatureStore, build_f0_matrix
from src.data.pairs import BUCKET_BOUNDARIES, probe_lengths
from src.data.val_region import Pair, ValRegionSplit
from src.distill.artifacts import write_kd_targets
from src.model.egostitch.composite import E2ENodeState, EgoStitchModel
from src.model.egostitch.generator.full_oracle import FullOracleGenerator
from src.model.egostitch.generator.imagine import SlotSet
from src.score_universe import (
    _file_sha256,
    _install_oracle_context,
    _load_checkpoint,
    _load_model_config,
    _oracle_truth_graph_sha256,
    _shard_range,
)
from src.train_b0 import _training_rows, _val_cls_rows, assemble_data, load_config

logger = logging.getLogger(__name__)

_DEFAULT_TOKEN_BUDGET = 32_768
_DEFAULT_BATCH_PAIRS = 32


# --------------------------------------------------------------------------- training-side legality


def truth_graph_for_kd(split: ValRegionSplit) -> nx.Graph:
    """Training-side truth graph: loopless training positives over every train node.

    By construction (`ValRegionSplit.training_positives` excludes any pair
    with both endpoints in `split.v_val`), this graph has zero V_val-internal
    edges: V_val nodes appear here only through their cross-boundary
    neighbors.
    """
    return split.build_training_graph()


def assert_training_side_only(
    node_ids: Sequence[str],
    truth_graph: nx.Graph,
    split: ValRegionSplit,
    test_nodes: frozenset[str],
) -> None:
    """Hard-refuse test-split leakage, an off-universe node, or a V_val-internal truth edge.

    V_val nodes are legitimately part of `split.train_nodes` and `truth_graph`
    (cross-boundary edges are not quarantined); only a V_val-*internal* edge
    is illegal here (row-level quarantine is `assert_no_val_internal_training_rows`).

    Raises:
        ValueError: If `node_ids` overlaps the benchmark test split; `node_ids`
            or `truth_graph` contains a node outside `split.train_nodes`; or
            `truth_graph` contains an edge with both endpoints inside
            `split.v_val`.
    """
    # Test-split membership is checked before the generic "outside the
    # training universe" fallback: the test split is disjoint from
    # `split.train_nodes` by construction, so a test node is always also
    # "outside the training universe" -- checking the specific partition
    # first gives the more diagnostic message instead of always being
    # shadowed by the generic one.
    test_overlap = sorted(set(node_ids) & test_nodes)
    if test_overlap:
        raise ValueError(
            f"KD node universe contains {len(test_overlap)} benchmark test-split node(s) "
            f"(first few: {test_overlap[:5]})"
        )
    foreign_universe = sorted(set(node_ids) - split.train_nodes)
    if foreign_universe:
        raise ValueError(
            f"KD node universe contains {len(foreign_universe)} node(s) outside the training "
            f"universe (first few: {foreign_universe[:5]})"
        )
    foreign_truth = sorted(set(truth_graph.nodes) - split.train_nodes)
    if foreign_truth:
        raise ValueError(
            f"KD truth graph contains {len(foreign_truth)} node(s) outside the training "
            f"universe (first few: {foreign_truth[:5]}); only training-side truth is legal here"
        )
    internal_edges = sorted(
        (u, v) for u, v in truth_graph.edges() if u in split.v_val and v in split.v_val
    )
    if internal_edges:
        raise ValueError(
            f"KD truth graph contains {len(internal_edges)} V_val-internal edge(s) "
            f"(first few: {internal_edges[:5]}); the validation region is quarantined "
            "from supervision"
        )


def require_full_ego_oracle(model: EgoStitchModel) -> FullOracleGenerator:
    """Refuse any generator but `full_ego_oracle` (the only legal KD teacher)."""
    if (
        model.cfg.generator.name != "full_ego_oracle"
        or type(model.generator) is not FullOracleGenerator
    ):
        raise ValueError(
            "KD teacher targets require a checkpoint whose generator is "
            f"'full_ego_oracle'; got {type(model.generator).__name__}. Scoring any "
            "other generator would not be the Full-Ego Pooled Oracle teacher this "
            "KD pipeline distills from."
        )
    return model.generator


# --------------------------------------------------------------------------- row universe


def assert_no_val_internal_training_rows(train_rows: Sequence[Pair], v_val: frozenset[str]) -> None:
    """Hard-refuse a training row with both endpoints inside V_val.

    The validation block is exempt: `_val_cls_rows` rows are the held-out
    V_val classification rows, scored only for validation-side diagnostics.

    Raises:
        ValueError: If any training row has both endpoints inside `v_val`.
    """
    violations = [(u, v) for u, v in train_rows if u in v_val and v in v_val]
    if violations:
        raise ValueError(
            f"{len(violations)} training row(s) have both endpoints inside V_val "
            f"(quarantined from supervision; first few: {violations[:5]})"
        )


def _row_positions(
    rows: Sequence[Pair], position: Mapping[str, int]
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Map row endpoints to `node_ids` positions, refusing an off-universe endpoint.

    Raises:
        ValueError: If any row endpoint is missing from `position`.
    """
    missing = sorted({node for pair in rows for node in pair if node not in position})
    if missing:
        raise ValueError(
            f"{len(missing)} row endpoint(s) are outside the training node universe "
            f"(first few: {missing[:5]})"
        )
    a_idx = np.array([position[u] for u, _ in rows], dtype=np.int32)
    b_idx = np.array([position[v] for _, v in rows], dtype=np.int32)
    return a_idx, b_idx


# --------------------------------------------------------------------------- node encoding


def encode_all_nodes(
    model: EgoStitchModel,
    store: FeatureStore,
    node_ids: Sequence[str],
    *,
    device: torch.device,
    token_budget: int,
    f0_cache: Path,
) -> dict[str, E2ENodeState]:
    """Encode every node in `node_ids` exactly once.

    Mirrors `score_universe._score_egostitch_e2e`'s bucketed encode-once
    node-state cache. `ground` is a dummy zero tensor:
    `FullOracleGenerator.encode_node` reads only its batch dimension (`del
    ground`), never its content, so a real grounding pool -- and the
    `build_grounding_pool` cache it would otherwise require -- is unneeded
    for this generator.
    """
    f0_cache.parent.mkdir(parents=True, exist_ok=True)
    matrix, index = build_f0_matrix(store, node_ids, cache_path=f0_cache)
    n_ground = model.generator_cfg.n_ground

    node_lengths = {
        node_id: length[0]
        for node_id, length in zip(
            node_ids, probe_lengths(store, [(n, n) for n in node_ids]), strict=True
        )
    }
    buckets: dict[int, list[str]] = {}
    for node_id in node_ids:
        length = node_lengths[node_id]
        boundary = next((value for value in BUCKET_BOUNDARIES if length <= value), length)
        buckets.setdefault(boundary, []).append(node_id)

    node_cache: dict[str, E2ENodeState] = {}
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for boundary in sorted(buckets):
            batch_size = max(1, token_budget // max(1, boundary))
            bucket_nodes = buckets[boundary]
            for start in range(0, len(bucket_nodes), batch_size):
                batch_nodes = bucket_nodes[start : start + batch_size]
                tokens = [store.load_tokens(node_id) for node_id in batch_nodes]
                lengths = torch.tensor(
                    [token.size(0) for token in tokens], dtype=torch.int64, device=device
                )
                embeddings = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True).to(device)
                rows = torch.tensor([index[node_id] for node_id in batch_nodes], dtype=torch.long)
                ground = torch.zeros(len(batch_nodes), n_ground, model.input_dim, device=device)
                state = model.encode_node_state(
                    embeddings,
                    lengths,
                    matrix.index_select(0, rows).to(device),
                    ground,
                    None,
                    rows.to(device),
                )
                if state.slots is None or state.projected_x is None:
                    raise RuntimeError(
                        "full-ego oracle generator produced no slot state -- this is a "
                        "checkpoint/generator mismatch, not a legal degraded mode"
                    )
                for position, node_id in enumerate(batch_nodes):
                    true_length = int(lengths[position].item())
                    slots = SlotSet(
                        *(
                            value[position : position + 1].detach().float().cpu()
                            for value in state.slots
                        )
                    )
                    projected_x = state.projected_x[position : position + 1].detach().float().cpu()
                    node_cache[node_id] = E2ENodeState(
                        encoded=state.encoded[position : position + 1, :true_length]
                        .detach()
                        .float()
                        .cpu(),
                        length=state.length[position : position + 1].detach().cpu(),
                        slots=slots,
                        projected_x=projected_x,
                        ground_ids=None,
                    )
    return node_cache


def _stack_states(states: Sequence[E2ENodeState]) -> E2ENodeState:
    """Pad/concatenate cached per-node states into one batch (always slot-producing)."""
    slot_sets: list[SlotSet] = []
    projected: list[torch.Tensor] = []
    for state in states:
        if state.slots is None or state.projected_x is None:
            raise ValueError("KD pair scoring requires slot-producing node states")
        slot_sets.append(state.slots)
        projected.append(state.projected_x)
    encoded = torch.nn.utils.rnn.pad_sequence(
        [state.encoded.squeeze(0) for state in states], batch_first=True
    )
    slots = SlotSet(*(torch.cat(values, dim=0) for values in zip(*slot_sets, strict=True)))
    return E2ENodeState(
        encoded=encoded,
        length=torch.cat([state.length for state in states], dim=0),
        slots=slots,
        projected_x=torch.cat(projected, dim=0),
        ground_ids=None,
    )


def _move_state(state: E2ENodeState, device: torch.device) -> E2ENodeState:
    assert state.slots is not None and state.projected_x is not None
    slots = SlotSet(*(value.to(device=device) for value in state.slots))
    return E2ENodeState(
        encoded=state.encoded.to(device=device),
        length=state.length.to(device=device),
        slots=slots,
        projected_x=state.projected_x.to(device=device),
        ground_ids=None,
    )


# --------------------------------------------------------------------------- pair scoring


@dataclass(frozen=True)
class ScoredRows:
    """One `score_rows` pass, aligned row-for-row with its `(a_idx, b_idx)` input.

    ``teacher_rep`` is the dump-side symmetrized teacher pooled pair embedding
    ``0.5 * (pooled_ab + pooled_ba)``, formed in fp32 then cast to fp16.
    """

    teacher_logit: NDArray[np.float32]
    teacher_rep: NDArray[np.float16]


def _encoder_seed_count(model: EgoStitchModel) -> int:
    seeds = getattr(model.encoder, "seeds", None)
    if not isinstance(seeds, int) or seeds <= 0:
        raise RuntimeError(
            "KD teacher targets require the grit_gmt PMA encoder: the PMA "
            "pooled embedding is the kd_gen latent target"
        )
    return seeds


def score_rows(
    model: EgoStitchModel,
    node_cache: Mapping[str, E2ENodeState],
    node_ids: Sequence[str],
    a_positions: NDArray[np.int32],
    b_positions: NDArray[np.int32],
    *,
    device: torch.device,
    batch_pairs: int = _DEFAULT_BATCH_PAIRS,
) -> ScoredRows:
    """Score every `(a_positions[i], b_positions[i])` row fp32, no autocast (see module docstring).

    This is the CPU-testable scoring core: given an already-encoded
    `node_cache` (`encode_all_nodes`) and a bound `full_ego_oracle` context
    (`_install_oracle_context`), it only stitches/scores rows -- no
    FeatureStore or checkpoint I/O.
    """
    _encoder_seed_count(model)
    n_rows = len(a_positions)
    teacher_logit = np.empty(n_rows, dtype=np.float32)
    teacher_rep: NDArray[np.float16] | None = None
    for start in range(0, n_rows, max(batch_pairs, 1)):
        end = min(start + max(batch_pairs, 1), n_rows)
        a_batch = a_positions[start:end]
        b_batch = b_positions[start:end]
        nodes_a = [node_ids[i] for i in a_batch]
        nodes_b = [node_ids[j] for j in b_batch]
        state_a = _move_state(_stack_states([node_cache[n] for n in nodes_a]), device)
        state_b = _move_state(_stack_states([node_cache[n] for n in nodes_b]), device)
        is_self = torch.from_numpy(a_batch == b_batch).to(device=device, dtype=torch.bool)
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            pair_context = model.build_pair_context_from_states(state_a, state_b, is_self)
            logits = model.score_pair_context(pair_context)
        teacher_logit[start:end] = logits.detach().to(dtype=torch.float32, device="cpu").numpy()
        if pair_context.pooled_ab is None or pair_context.pooled_ba is None:
            raise RuntimeError(
                "full-ego oracle pair context carries no pooled embedding -- "
                "this is a bug, not a legal degraded mode"
            )
        ab = pair_context.pooled_ab.detach().to(dtype=torch.float32, device="cpu")
        ba = pair_context.pooled_ba.detach().to(dtype=torch.float32, device="cpu")
        rep = (0.5 * (ab + ba)).to(dtype=torch.float16).numpy()
        if teacher_rep is None:
            teacher_rep = np.empty((n_rows, rep.shape[-1]), dtype=np.float16)
        teacher_rep[start:end] = rep
    if teacher_rep is None:
        teacher_rep = np.empty((0, 0), dtype=np.float16)
    return ScoredRows(
        teacher_logit=teacher_logit,
        teacher_rep=teacher_rep,
    )


# --------------------------------------------------------------------------- pre-check stats gate

_STATS_EPS = 1e-12


def build_stats_report(
    train_logit: NDArray[np.float32],
    train_label: NDArray[np.int8],
    val_logit: NDArray[np.float32],
    val_label: NDArray[np.int8],
) -> dict[str, object]:
    """Per-block (train, val) entropy/calibration pre-check summary stats."""
    return {
        "train": _block_stats(train_logit, train_label),
        "val": _block_stats(val_logit, val_label),
    }


def _block_stats(logit: NDArray[np.float32], label: NDArray[np.int8]) -> dict[str, object]:
    logit64 = logit.astype(np.float64)
    prob = 1.0 / (1.0 + np.exp(-logit64))
    entropy = -(prob * np.log(prob + _STATS_EPS) + (1 - prob) * np.log(1 - prob + _STATS_EPS))
    return {
        "n_rows": int(len(logit)),
        "label_prevalence": float(label.mean()) if len(label) else 0.0,
        "teacher_logit_overall": _histogram(logit64),
        "teacher_logit_positives": _histogram(logit64[label == 1]),
        "teacher_logit_negatives": _histogram(logit64[label == 0]),
        "sigmoid_entropy": _histogram(entropy),
    }


def _histogram(values: NDArray[np.float64]) -> dict[str, float]:
    if values.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def write_stats_report(output_dir: Path, report: Mapping[str, object]) -> None:
    """Write the pre-check stats report next to the artifact and log it."""
    text = json.dumps(report, indent=2, sort_keys=True)
    (output_dir / "stats_report.json").write_text(text, encoding="utf-8")
    logger.info("KD teacher-target stats report:\n%s", text)


# --------------------------------------------------------------------------- sharding


def _parse_shard(spec: str) -> tuple[int, int]:
    parts = spec.split("/")
    if len(parts) != 2:
        raise ValueError(f"--row-shard must be 'I/N', got {spec!r}")
    try:
        index, total = int(parts[0]), int(parts[1])
    except ValueError as error:
        raise ValueError(f"--row-shard must be 'I/N', got {spec!r}") from error
    if total <= 0 or not 0 <= index < total:
        raise ValueError(f"--row-shard index must satisfy 0 <= I < N, got {spec!r}")
    return index, total


def _shard_path(output: Path, shard: int, num_shards: int) -> Path:
    return output.parent / f"{output.name}.shard-{shard}-of-{num_shards}.npz"


def _write_shard(
    output: Path,
    shard: int,
    num_shards: int,
    *,
    a_idx: NDArray[np.int32],
    b_idx: NDArray[np.int32],
    scored: ScoredRows,
) -> None:
    path = _shard_path(output, shard, num_shards)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        a_idx=a_idx,
        b_idx=b_idx,
        teacher_logit=scored.teacher_logit,
        teacher_rep=scored.teacher_rep,
    )


def _all_shards_present(output: Path, num_shards: int) -> bool:
    return all(_shard_path(output, shard, num_shards).is_file() for shard in range(num_shards))


def _merge_shards(
    output: Path, a_idx: NDArray[np.int32], b_idx: NDArray[np.int32], num_shards: int
) -> ScoredRows:
    """Concatenate `--row-shard` outputs back into the full deterministic row order.

    Every shard invocation re-derives the identical combined row list (a pure
    function of `--config`), so shards need not carry offsets of their own:
    shard `i`'s row range is exactly its `_shard_range`-derived range over
    ``len(a_idx)``.
    """
    n_total = len(a_idx)
    teacher_logit = np.empty(n_total, dtype=np.float32)
    teacher_rep = np.empty((n_total, 0), dtype=np.float16)
    for shard in range(num_shards):
        start, end = _shard_range(n_total, shard, num_shards)
        path = _shard_path(output, shard, num_shards)
        if not path.is_file():
            raise ValueError(f"missing shard file for merge: {path}")
        with np.load(path) as archive:
            if not np.array_equal(archive["a_idx"], a_idx[start:end]) or not np.array_equal(
                archive["b_idx"], b_idx[start:end]
            ):
                raise ValueError(
                    f"shard {shard} row identity does not match the deterministic row list "
                    "(re-run with the same --config)"
                )
            teacher_logit[start:end] = archive["teacher_logit"]
            if teacher_rep.shape[1] == 0 and archive["teacher_rep"].shape[0]:
                rep_width = archive["teacher_rep"].shape[1]
                teacher_rep = np.empty((n_total, rep_width), dtype=np.float16)
            teacher_rep[start:end] = archive["teacher_rep"]
    return ScoredRows(
        teacher_logit=teacher_logit,
        teacher_rep=teacher_rep,
    )


# --------------------------------------------------------------------------- verification


def verify_sample(
    model: EgoStitchModel,
    node_cache: Mapping[str, E2ENodeState],
    node_ids: Sequence[str],
    a_idx: NDArray[np.int32],
    b_idx: NDArray[np.int32],
    teacher_logit: NDArray[np.float32],
    *,
    n: int,
    device: torch.device,
) -> None:
    """Recompute `n` random combined rows directly and assert they match the written logits."""
    total = len(a_idx)
    if total == 0 or n <= 0:
        return
    rng = np.random.default_rng(total)
    sample = rng.choice(total, size=min(n, total), replace=False)
    recomputed = score_rows(
        model,
        node_cache,
        node_ids,
        a_idx[sample],
        b_idx[sample],
        device=device,
        batch_pairs=max(len(sample), 1),
    ).teacher_logit
    if not np.allclose(recomputed, teacher_logit[sample], atol=1e-5, rtol=1e-5):
        raise ValueError("--verify-sample recomputation does not match the written teacher logits")
    logger.info("verify-sample: %d/%d recomputed rows match", len(sample), total)


# --------------------------------------------------------------------------- CLI


def _load_teacher_checkpoint(args: argparse.Namespace) -> tuple[torch.nn.Module, str, str]:
    """Rebuild the teacher, accepting bare eligible-epoch checkpoints via ``--model-config``."""
    if args.model_config is None:
        return _load_checkpoint(args.checkpoint)
    return _load_checkpoint(
        args.checkpoint,
        model_family="egostitch_e2e",
        model_config=_load_model_config(args.model_config, "egostitch_e2e"),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the `python -m src.distill.teacher_targets` argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="Training YAML whose rows this artifact joins"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help=(
            "Training YAML whose model.config rebuilds a bare eligible-epoch "
            "checkpoint; published best.pt/last.pt embed it and need no override"
        ),
    )
    parser.add_argument("--token-budget", type=int, default=_DEFAULT_TOKEN_BUDGET)
    parser.add_argument("--batch-pairs", type=int, default=_DEFAULT_BATCH_PAIRS)
    parser.add_argument("--f0-cache", type=Path, default=None)
    parser.add_argument(
        "--row-shard",
        type=str,
        default=None,
        help="'I/N': score only row shard I of N of the combined train+val row list "
        "(resume sharding)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge previously written --row-shard outputs into the final artifact",
    )
    parser.add_argument("--verify-sample", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the KD teacher-target dumper CLI (score, `--row-shard`, or `--merge`)."""
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)

    cfg = load_config(args.config)
    assembled = assemble_data(cfg, verify=True)
    split = assembled.val_split
    positives, negatives = _training_rows(split, assembled.exclude_nodes)
    train_rows = positives + negatives
    train_labels = [1] * len(positives) + [0] * len(negatives)
    val_rows, val_labels = _val_cls_rows(split, assembled.exclude_nodes)

    truth_graph = truth_graph_for_kd(split)
    node_ids = sorted(split.train_nodes - assembled.exclude_nodes)
    test_nodes = frozenset(assembled.benchmark.split.test_nodes)
    assert_training_side_only(node_ids, truth_graph, split, test_nodes)
    assert_no_val_internal_training_rows(train_rows, split.v_val)

    position = {node_id: i for i, node_id in enumerate(node_ids)}
    combined_rows = train_rows + val_rows
    a_idx, b_idx = _row_positions(combined_rows, position)
    labels = np.array(train_labels + val_labels, dtype=np.int8)
    n_train = len(train_rows)
    n_total = len(combined_rows)

    if args.merge:
        if args.row_shard is None:
            raise ValueError("--merge requires --row-shard I/N to know the shard count")
        _, num_shards = _parse_shard(args.row_shard)
        _finish_merge(args, node_ids, truth_graph, a_idx, b_idx, labels, n_train, num_shards)
        return

    shard_range: tuple[int, int, int] | None = None
    if args.row_shard is not None:
        shard_index, num_shards = _parse_shard(args.row_shard)
        start, end = _shard_range(n_total, shard_index, num_shards)
        shard_range = (start, end, num_shards)

    device = torch.device(args.device)
    model, model_family, checkpoint_id = _load_teacher_checkpoint(args)
    if model_family != "egostitch_e2e" or not isinstance(model, EgoStitchModel):
        raise ValueError(
            f"KD teacher targets require model_family 'egostitch_e2e', got {model_family!r}"
        )
    require_full_ego_oracle(model)
    model.to(device)
    model.eval()
    _install_oracle_context(model, node_ids, truth_graph=truth_graph)

    f0_cache = args.f0_cache if args.f0_cache is not None else args.output / "f0_cache.pt"
    node_cache = encode_all_nodes(
        model,
        assembled.store,
        node_ids,
        device=device,
        token_budget=args.token_budget,
        f0_cache=f0_cache,
    )

    if shard_range is not None:
        start, end, num_shards = shard_range
        scored = score_rows(
            model,
            node_cache,
            node_ids,
            a_idx[start:end],
            b_idx[start:end],
            device=device,
            batch_pairs=args.batch_pairs,
        )
        shard_index = _parse_shard(args.row_shard)[0]
        _write_shard(
            args.output,
            shard_index,
            num_shards,
            a_idx=a_idx[start:end],
            b_idx=b_idx[start:end],
            scored=scored,
        )
        logger.info("wrote shard %d/%d", shard_index, num_shards)
        if _all_shards_present(args.output, num_shards):
            logger.info("all %d shards present; auto-merging", num_shards)
            _finish_merge(
                args,
                node_ids,
                truth_graph,
                a_idx,
                b_idx,
                labels,
                n_train,
                num_shards,
                checkpoint_id_override=checkpoint_id,
            )
        return

    scored = score_rows(
        model, node_cache, node_ids, a_idx, b_idx, device=device, batch_pairs=args.batch_pairs
    )
    _finalize_artifact(
        args, node_ids, truth_graph, a_idx, b_idx, labels, n_train, scored, checkpoint_id
    )
    if args.verify_sample > 0:
        verify_sample(
            model,
            node_cache,
            node_ids,
            a_idx,
            b_idx,
            scored.teacher_logit,
            n=args.verify_sample,
            device=device,
        )


def _finish_merge(
    args: argparse.Namespace,
    node_ids: Sequence[str],
    truth_graph: nx.Graph,
    a_idx: NDArray[np.int32],
    b_idx: NDArray[np.int32],
    labels: NDArray[np.int8],
    n_train: int,
    num_shards: int,
    *,
    checkpoint_id_override: str | None = None,
) -> None:
    scored = _merge_shards(args.output, a_idx, b_idx, num_shards)
    if checkpoint_id_override is not None:
        checkpoint_id = checkpoint_id_override
    else:
        _, model_family, checkpoint_id = _load_teacher_checkpoint(args)
        if model_family != "egostitch_e2e":
            raise ValueError(
                f"KD teacher targets require model_family 'egostitch_e2e', got {model_family!r}"
            )
    _finalize_artifact(
        args, node_ids, truth_graph, a_idx, b_idx, labels, n_train, scored, checkpoint_id
    )


def _finalize_artifact(
    args: argparse.Namespace,
    node_ids: Sequence[str],
    truth_graph: nx.Graph,
    a_idx: NDArray[np.int32],
    b_idx: NDArray[np.int32],
    labels: NDArray[np.int8],
    n_train: int,
    scored: ScoredRows,
    checkpoint_id: str | None,
) -> None:
    checkpoint_sha256 = _file_sha256(args.checkpoint)
    write_kd_targets(
        args.output,
        node_ids=node_ids,
        pair_a_idx=a_idx[:n_train],
        pair_b_idx=b_idx[:n_train],
        pair_label=labels[:n_train],
        teacher_logit=scored.teacher_logit[:n_train],
        teacher_rep=scored.teacher_rep[:n_train],
        val_pair_a_idx=a_idx[n_train:],
        val_pair_b_idx=b_idx[n_train:],
        val_pair_label=labels[n_train:],
        val_teacher_logit=scored.teacher_logit[n_train:],
        val_teacher_rep=scored.teacher_rep[n_train:],
        truth_graph_sha256=_oracle_truth_graph_sha256(truth_graph),
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_id=checkpoint_id,
    )
    report = build_stats_report(
        scored.teacher_logit[:n_train],
        labels[:n_train],
        scored.teacher_logit[n_train:],
        labels[n_train:],
    )
    write_stats_report(args.output, report)


if __name__ == "__main__":
    main()

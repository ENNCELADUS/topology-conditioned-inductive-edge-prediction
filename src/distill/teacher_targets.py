r"""Dump Full-Ego Pooled Oracle KD teacher targets over V_fit-only anchor/context pairs.

CLI (design B1 plan, "KD sampler + teacher-target artifact"):

    python -m src.distill.teacher_targets \\
        --checkpoint outputs/.../best.pt --data-root data --strategy breadth_first \\
        --seed 0 --k-near 10 --k-rand 10 --output outputs/distill/kd_targets_breadth_first_seed0

Architecture (see the B1 plan's "Architecture decisions" -- why this is a
dedicated module, not a `score_universe.py` extension): `score_universe.py`'s
``file:``/``candidate``/``test`` pair sources are held-out-ledgered and its
grounding pools are rebuilt with ``role_universe="test"``, both wrong for a
strictly train-side (V_fit-only) teacher pass. This module instead reuses
`score_universe.py`'s lower-level, universe-agnostic building blocks directly
(`_load_checkpoint`, `_install_oracle_context`, `_file_sha256`, `_shard_range`,
`_oracle_truth_graph_sha256`) and mirrors its per-node encode-once cache +
`build_pair_context_from_states`/`score_pair_context` pattern
(``_score_egostitch_e2e``, score_universe.py:2330-2399) rather than duplicating
either.

Every pair is scored fp32 with no autocast on the pair pass, matching the
``egostitch_e2e_pair_fp32_v1`` contract `_score_egostitch_e2e` itself uses
(score_universe.py:2696) -- this artifact's `teacher_logit` must be numerically
consistent with what a training-time rescoring of the same checkpoint would
produce, not merely close.

Legality (never bind anything but a G_fit-only truth graph): the truth graph is
`InternalHoldoutPartition.build_g_fit()` plus explicit isolates for every
V_fit node; `_assert_v_fit_only` hard-refuses if any V_hold or test-split node
reaches the truth graph or the sampled node universe. No pre-existing oracle
cache is ever read -- the training-side `g_fit_plus_v_hold` diagnostic caches
would be leakage here.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import torch
from numpy.typing import NDArray

from src.data.artifacts import load_benchmark
from src.data.features import FeatureStore, build_f0_matrix
from src.data.internal_holdout import InternalHoldoutPartition, derive_internal_holdout
from src.data.pairs import BUCKET_BOUNDARIES, probe_lengths
from src.data.partition import derive_training_interactions
from src.distill.artifacts import write_kd_targets
from src.distill.context_sampler import ContextSets, sample_context_sets
from src.model.egostitch.composite import E2ENodeState, EgoStitchModel
from src.model.egostitch.generator.full_oracle import FullOracleGenerator
from src.model.egostitch.generator.imagine import SlotSet
from src.score_universe import (
    _file_sha256,
    _install_oracle_context,
    _load_checkpoint,
    _oracle_truth_graph_sha256,
    _shard_range,
)

logger = logging.getLogger(__name__)

_BENCHMARK_SUBDIR = "benchmark_2025_neurips"
_FEATURES_SUBDIR = Path("features") / "frozen_node_features_1024"

_DEFAULT_TOKEN_BUDGET = 32_768
_DEFAULT_BATCH_PAIRS = 32


# --------------------------------------------------------------------------- V_fit legality


def _load_v_fit_holdout(
    data_root: Path, strategy: str
) -> tuple[InternalHoldoutPartition, frozenset[str]]:
    """Load `InternalHoldoutPartition` exactly as the training path derives it.

    Returns:
        The holdout partition and the benchmark's held-out test-split nodes
        (a second, separate universe `_assert_v_fit_only` must also refuse).
    """
    benchmark = load_benchmark(data_root / _BENCHMARK_SUBDIR, strategy)
    positives = [
        pair
        for pair, label in zip(
            benchmark.split.train_pairs.pairs,
            benchmark.split.train_pairs.labels.tolist(),
            strict=True,
        )
        if int(label) == 1
    ]
    interactions = derive_training_interactions(positives)
    holdout = derive_internal_holdout(sorted(benchmark.split.train_nodes), interactions.positives)
    return holdout, frozenset(benchmark.split.test_nodes)


def truth_graph_for_kd(holdout: InternalHoldoutPartition) -> nx.Graph:
    """G_fit-only truth graph: loopless V_fit training topology plus explicit isolates."""
    graph = holdout.build_g_fit().copy()
    graph.add_nodes_from(sorted(holdout.v_fit))
    return graph


def assert_v_fit_only(
    node_ids: Sequence[str],
    truth_graph: nx.Graph,
    holdout: InternalHoldoutPartition,
    test_nodes: frozenset[str],
) -> None:
    """Hard-refuse any V_hold or test-split node in the truth graph or node universe.

    Raises:
        ValueError: If the truth graph or `node_ids` contains a node outside
            V_fit, or `node_ids` overlaps V_hold or the benchmark test split.
    """
    # V_hold/test-split membership is checked before the generic "outside
    # V_fit" fallback: both partitions are disjoint from V_fit by construction
    # (`derive_internal_holdout`), so a V_hold or test node is always also
    # "outside V_fit" -- checking the specific partitions first gives the more
    # diagnostic message instead of always being shadowed by the generic one.
    v_fit = holdout.v_fit
    foreign_truth = sorted(set(truth_graph.nodes) - v_fit)
    if foreign_truth:
        raise ValueError(
            f"KD truth graph contains {len(foreign_truth)} node(s) outside V_fit "
            f"(first few: {foreign_truth[:5]}); only g_fit_only truth is legal here"
        )
    hold_overlap = sorted(set(node_ids) & holdout.v_hold)
    if hold_overlap:
        raise ValueError(
            f"KD node universe contains {len(hold_overlap)} V_hold node(s) "
            f"(first few: {hold_overlap[:5]}); this is training-side leakage"
        )
    test_overlap = sorted(set(node_ids) & test_nodes)
    if test_overlap:
        raise ValueError(
            f"KD node universe contains {len(test_overlap)} benchmark test-split node(s) "
            f"(first few: {test_overlap[:5]})"
        )
    foreign_universe = sorted(set(node_ids) - v_fit)
    if foreign_universe:
        raise ValueError(
            f"KD node universe contains {len(foreign_universe)} node(s) outside V_fit "
            f"(first few: {foreign_universe[:5]})"
        )


def require_full_ego_oracle(model: EgoStitchModel) -> FullOracleGenerator:
    """Refuse any generator but `full_ego_oracle` (the B1 plan's only legal teacher)."""
    if not isinstance(model.generator, FullOracleGenerator):
        raise ValueError(
            "KD teacher targets require a checkpoint whose generator is "
            f"'full_ego_oracle'; got {type(model.generator).__name__}. Scoring any "
            "other generator would not be the Full-Ego Pooled Oracle teacher the "
            "B1 plan distills from."
        )
    return model.generator


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
    node-state cache (score_universe.py:2315-2399). `ground` is a dummy zero
    tensor: `FullOracleGenerator.encode_node` reads only its batch dimension
    (`del ground`), never its content, so a real grounding pool -- and the
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


def score_kd_context(
    model: EgoStitchModel,
    node_cache: Mapping[str, E2ENodeState],
    node_ids: Sequence[str],
    context: ContextSets,
    *,
    device: torch.device,
    batch_pairs: int = _DEFAULT_BATCH_PAIRS,
) -> tuple[NDArray[np.float32], NDArray[np.float16], NDArray[np.float16]]:
    """Score every context pair fp32, no autocast (see module docstring).

    This is the CPU-testable scoring core: given an already-encoded
    `node_cache` (`encode_all_nodes`) and a bound `full_ego_oracle` context
    (`_install_oracle_context`), it only stitches/scores pairs -- no
    FeatureStore or checkpoint I/O.

    Returns:
        ``(teacher_logit, teacher_pooled_ab, teacher_pooled_ba)``, aligned
        row-for-row with `context.anchor_idx`/`context.partner_idx`.
    """
    n_pairs = len(context.anchor_idx)
    teacher_logit = np.empty(n_pairs, dtype=np.float32)
    pooled_ab: NDArray[np.float16] | None = None
    pooled_ba: NDArray[np.float16] | None = None
    for start in range(0, n_pairs, max(batch_pairs, 1)):
        end = min(start + max(batch_pairs, 1), n_pairs)
        anchors = context.anchor_idx[start:end]
        partners = context.partner_idx[start:end]
        nodes_a = [node_ids[i] for i in anchors]
        nodes_b = [node_ids[j] for j in partners]
        state_a = _move_state(_stack_states([node_cache[n] for n in nodes_a]), device)
        state_b = _move_state(_stack_states([node_cache[n] for n in nodes_b]), device)
        is_self = torch.from_numpy(anchors == partners).to(device=device, dtype=torch.bool)
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            pair_context = model.build_pair_context_from_states(state_a, state_b, is_self)
            logits = model.score_pair_context(pair_context)
        teacher_logit[start:end] = logits.detach().to(dtype=torch.float32, device="cpu").numpy()
        if pair_context.pooled_ab is None or pair_context.pooled_ba is None:
            raise RuntimeError(
                "full-ego oracle pair context carries no pooled embedding -- "
                "this is a bug, not a legal degraded mode"
            )
        ab = pair_context.pooled_ab.detach().to(dtype=torch.float16, device="cpu").numpy()
        ba = pair_context.pooled_ba.detach().to(dtype=torch.float16, device="cpu").numpy()
        if pooled_ab is None:
            pooled_ab = np.empty((n_pairs, ab.shape[-1]), dtype=np.float16)
            pooled_ba = np.empty((n_pairs, ba.shape[-1]), dtype=np.float16)
        pooled_ab[start:end] = ab
        cast(NDArray[np.float16], pooled_ba)[start:end] = ba
    if pooled_ab is None:
        pooled_ab = np.empty((0, 0), dtype=np.float16)
        pooled_ba = np.empty((0, 0), dtype=np.float16)
    return teacher_logit, pooled_ab, cast(NDArray[np.float16], pooled_ba)


def pair_labels(node_ids: Sequence[str], context: ContextSets, g_fit: nx.Graph) -> NDArray[np.int8]:
    """Return 1 for a context pair that is a `g_fit` edge, else 0."""
    labels = np.zeros(len(context.anchor_idx), dtype=np.int8)
    for row, (anchor, partner) in enumerate(
        zip(context.anchor_idx.tolist(), context.partner_idx.tolist(), strict=True)
    ):
        labels[row] = int(g_fit.has_edge(node_ids[anchor], node_ids[partner]))
    return labels


# --------------------------------------------------------------------------- pre-check stats gate


def build_stats_report(
    teacher_logit: NDArray[np.float32], pair_label: NDArray[np.int8], is_near: NDArray[np.uint8]
) -> dict[str, object]:
    """The B1 plan's pre-check gate: entropy/calibration/hop-stratified summary stats."""
    logit64 = teacher_logit.astype(np.float64)
    prob = 1.0 / (1.0 + np.exp(-logit64))
    entropy = -(prob * np.log(prob + _STATS_EPS) + (1 - prob) * np.log(1 - prob + _STATS_EPS))

    report: dict[str, object] = {
        "n_pairs": int(len(teacher_logit)),
        "teacher_logit_overall": _histogram(logit64),
        "teacher_logit_negatives": _histogram(logit64[pair_label == 0]),
        "sigmoid_entropy": _histogram(entropy),
        "hop_stratified": {},
    }
    hop_stratified = cast(dict[str, object], report["hop_stratified"])
    for near_flag, name in ((1, "near"), (0, "random")):
        row_mask = is_near == near_flag
        hop_stratified[name] = {
            "count": int(row_mask.sum()),
            "label_prevalence": float(pair_label[row_mask].mean()) if row_mask.any() else 0.0,
            "mean_sigmoid": float(prob[row_mask].mean()) if row_mask.any() else 0.0,
            "teacher_logit": _histogram(logit64[row_mask]),
        }
    return report


_STATS_EPS = 1e-12


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
        raise ValueError(f"--anchor-shard must be 'I/N', got {spec!r}")
    try:
        index, total = int(parts[0]), int(parts[1])
    except ValueError as error:
        raise ValueError(f"--anchor-shard must be 'I/N', got {spec!r}") from error
    if total <= 0 or not 0 <= index < total:
        raise ValueError(f"--anchor-shard index must satisfy 0 <= I < N, got {spec!r}")
    return index, total


def _shard_path(output: Path, shard: int, num_shards: int) -> Path:
    return output.parent / f"{output.name}.shard-{shard}-of-{num_shards}.npz"


def _write_shard(
    output: Path,
    shard: int,
    num_shards: int,
    *,
    pair_anchor_idx: NDArray[np.int32],
    pair_partner_idx: NDArray[np.int32],
    teacher_logit: NDArray[np.float32],
    teacher_pooled_ab: NDArray[np.float16],
    teacher_pooled_ba: NDArray[np.float16],
    is_near: NDArray[np.uint8],
    pair_label: NDArray[np.int8],
) -> None:
    path = _shard_path(output, shard, num_shards)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        pair_anchor_idx=pair_anchor_idx,
        pair_partner_idx=pair_partner_idx,
        teacher_logit=teacher_logit,
        teacher_pooled_ab=teacher_pooled_ab,
        teacher_pooled_ba=teacher_pooled_ba,
        is_near=is_near,
        pair_label=pair_label,
    )


def _all_shards_present(output: Path, num_shards: int) -> bool:
    return all(_shard_path(output, shard, num_shards).is_file() for shard in range(num_shards))


def _merge_shards(
    output: Path, node_ids: Sequence[str], context: ContextSets, num_shards: int
) -> tuple[NDArray[np.float32], NDArray[np.float16], NDArray[np.float16], NDArray[np.int8]]:
    """Concatenate `--anchor-shard` outputs back into the full deterministic pair order.

    Every shard invocation resamples the identical `ContextSets` (a pure
    function of `node_ids`/seed/k values), so shards need not carry offsets
    of their own: shard `i`'s pair-row range is exactly
    ``[anchor_offsets[start_i], anchor_offsets[end_i])`` for its
    `_shard_range`-derived anchor range.
    """
    n_pairs = len(context.anchor_idx)
    teacher_logit = np.empty(n_pairs, dtype=np.float32)
    pooled_ab = np.empty((n_pairs, 0), dtype=np.float16)
    pooled_ba = np.empty((n_pairs, 0), dtype=np.float16)
    pair_label = np.empty(n_pairs, dtype=np.int8)
    for shard in range(num_shards):
        start, end = _shard_range(len(node_ids), shard, num_shards)
        pair_start = int(context.anchor_offsets[start])
        pair_end = int(context.anchor_offsets[end])
        path = _shard_path(output, shard, num_shards)
        if not path.is_file():
            raise ValueError(f"missing shard file for merge: {path}")
        with np.load(path) as archive:
            if not np.array_equal(
                archive["pair_anchor_idx"], context.anchor_idx[pair_start:pair_end]
            ) or not np.array_equal(
                archive["pair_partner_idx"], context.partner_idx[pair_start:pair_end]
            ):
                raise ValueError(
                    f"shard {shard} pair identity does not match the deterministic context "
                    "(re-run with the same --seed/--k-near/--k-rand)"
                )
            teacher_logit[pair_start:pair_end] = archive["teacher_logit"]
            if pooled_ab.shape[1] == 0 and archive["teacher_pooled_ab"].shape[0]:
                ab_width = archive["teacher_pooled_ab"].shape[1]
                ba_width = archive["teacher_pooled_ba"].shape[1]
                pooled_ab = np.empty((n_pairs, ab_width), dtype=np.float16)
                pooled_ba = np.empty((n_pairs, ba_width), dtype=np.float16)
            pooled_ab[pair_start:pair_end] = archive["teacher_pooled_ab"]
            pooled_ba[pair_start:pair_end] = archive["teacher_pooled_ba"]
            pair_label[pair_start:pair_end] = archive["pair_label"]
    return teacher_logit, pooled_ab, pooled_ba, pair_label


# --------------------------------------------------------------------------- verification


def verify_sample(
    model: EgoStitchModel,
    node_cache: Mapping[str, E2ENodeState],
    node_ids: Sequence[str],
    context: ContextSets,
    teacher_logit: NDArray[np.float32],
    *,
    n: int,
    device: torch.device,
) -> None:
    """Recompute `n` random pairs directly and assert they match the written logits."""
    total = len(context.anchor_idx)
    if total == 0 or n <= 0:
        return
    rng = np.random.default_rng(context.seed)
    sample = rng.choice(total, size=min(n, total), replace=False)
    sample_context = ContextSets(
        anchor_idx=context.anchor_idx[sample],
        partner_idx=context.partner_idx[sample],
        anchor_offsets=np.array([0, len(sample)], dtype=np.int64),
        is_near=context.is_near[sample],
        k_near=context.k_near,
        k_rand=context.k_rand,
        seed=context.seed,
    )
    recomputed, _, _ = score_kd_context(
        model, node_cache, node_ids, sample_context, device=device, batch_pairs=max(len(sample), 1)
    )
    if not np.allclose(recomputed, teacher_logit[sample], atol=1e-5, rtol=1e-5):
        raise ValueError("--verify-sample recomputation does not match the written teacher logits")
    logger.info("verify-sample: %d/%d recomputed pairs match", len(sample), total)


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    """Build the `python -m src.distill.teacher_targets` argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--strategy", type=str, default="breadth_first")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k-near", type=int, required=True)
    parser.add_argument("--k-rand", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--token-budget", type=int, default=_DEFAULT_TOKEN_BUDGET)
    parser.add_argument("--batch-pairs", type=int, default=_DEFAULT_BATCH_PAIRS)
    parser.add_argument("--f0-cache", type=Path, default=None)
    parser.add_argument(
        "--anchor-shard",
        type=str,
        default=None,
        help="'I/N': score only the pairs of node-position shard I of N (resume sharding)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge previously written --anchor-shard outputs into the final artifact",
    )
    parser.add_argument("--verify-sample", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the KD teacher-target dumper CLI (score, `--anchor-shard`, or `--merge`)."""
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)

    holdout, test_nodes = _load_v_fit_holdout(args.data_root, args.strategy)
    truth_graph = truth_graph_for_kd(holdout)
    node_ids = sorted(holdout.v_fit)
    assert_v_fit_only(node_ids, truth_graph, holdout, test_nodes)

    context = sample_context_sets(
        truth_graph, node_ids, seed=args.seed, k_near=args.k_near, k_rand=args.k_rand
    )

    if args.merge:
        if args.anchor_shard is None:
            raise ValueError("--merge requires --anchor-shard I/N to know the shard count")
        _, num_shards = _parse_shard(args.anchor_shard)
        _finish_merge(args, node_ids, context, num_shards)
        return

    shard_range: tuple[int, int, int] | None = None
    if args.anchor_shard is not None:
        shard_index, num_shards = _parse_shard(args.anchor_shard)
        start, end = _shard_range(len(node_ids), shard_index, num_shards)
        shard_range = (start, end, num_shards)

    device = torch.device(args.device)
    model, model_family, checkpoint_id = _load_checkpoint(args.checkpoint)
    if model_family != "egostitch_e2e" or not isinstance(model, EgoStitchModel):
        raise ValueError(
            f"KD teacher targets require model_family 'egostitch_e2e', got {model_family!r}"
        )
    require_full_ego_oracle(model)
    model.to(device)
    model.eval()
    _install_oracle_context(model, node_ids, truth_graph=truth_graph)

    store = FeatureStore(args.data_root / _FEATURES_SUBDIR)
    f0_cache = args.f0_cache if args.f0_cache is not None else args.output / "f0_cache.pt"
    node_cache = encode_all_nodes(
        model, store, node_ids, device=device, token_budget=args.token_budget, f0_cache=f0_cache
    )

    if shard_range is not None:
        start, end, num_shards = shard_range
        pair_start, pair_end = int(context.anchor_offsets[start]), int(context.anchor_offsets[end])
        shard_context = ContextSets(
            anchor_idx=context.anchor_idx[pair_start:pair_end],
            partner_idx=context.partner_idx[pair_start:pair_end],
            anchor_offsets=context.anchor_offsets[start : end + 1] - context.anchor_offsets[start],
            is_near=context.is_near[pair_start:pair_end],
            k_near=context.k_near,
            k_rand=context.k_rand,
            seed=context.seed,
        )
        teacher_logit, pooled_ab, pooled_ba = score_kd_context(
            model, node_cache, node_ids, shard_context, device=device, batch_pairs=args.batch_pairs
        )
        labels = pair_labels(node_ids, shard_context, truth_graph)
        shard_index = _parse_shard(args.anchor_shard)[0]
        _write_shard(
            args.output,
            shard_index,
            num_shards,
            pair_anchor_idx=shard_context.anchor_idx,
            pair_partner_idx=shard_context.partner_idx,
            teacher_logit=teacher_logit,
            teacher_pooled_ab=pooled_ab,
            teacher_pooled_ba=pooled_ba,
            is_near=shard_context.is_near,
            pair_label=labels,
        )
        logger.info("wrote shard %d/%d", shard_index, num_shards)
        if _all_shards_present(args.output, num_shards):
            logger.info("all %d shards present; auto-merging", num_shards)
            _finish_merge(
                args, node_ids, context, num_shards, checkpoint_override=(model, checkpoint_id)
            )
        return

    teacher_logit, pooled_ab, pooled_ba = score_kd_context(
        model, node_cache, node_ids, context, device=device, batch_pairs=args.batch_pairs
    )
    labels = pair_labels(node_ids, context, truth_graph)
    _finalize_artifact(
        args,
        node_ids,
        context,
        truth_graph,
        teacher_logit,
        pooled_ab,
        pooled_ba,
        labels,
        checkpoint_id,
    )
    if args.verify_sample > 0:
        verify_sample(
            model, node_cache, node_ids, context, teacher_logit, n=args.verify_sample, device=device
        )


def _finish_merge(
    args: argparse.Namespace,
    node_ids: Sequence[str],
    context: ContextSets,
    num_shards: int,
    *,
    checkpoint_override: tuple[EgoStitchModel, str] | None = None,
) -> None:
    teacher_logit, pooled_ab, pooled_ba, labels = _merge_shards(
        args.output, node_ids, context, num_shards
    )
    if checkpoint_override is not None:
        _, checkpoint_id = checkpoint_override
    else:
        _, model_family, checkpoint_id = _load_checkpoint(args.checkpoint)
        if model_family != "egostitch_e2e":
            raise ValueError(
                f"KD teacher targets require model_family 'egostitch_e2e', got {model_family!r}"
            )
    holdout, test_nodes = _load_v_fit_holdout(args.data_root, args.strategy)
    truth_graph = truth_graph_for_kd(holdout)
    assert_v_fit_only(node_ids, truth_graph, holdout, test_nodes)
    _finalize_artifact(
        args,
        node_ids,
        context,
        truth_graph,
        teacher_logit,
        pooled_ab,
        pooled_ba,
        labels,
        checkpoint_id,
    )


def _finalize_artifact(
    args: argparse.Namespace,
    node_ids: Sequence[str],
    context: ContextSets,
    truth_graph: nx.Graph,
    teacher_logit: NDArray[np.float32],
    pooled_ab: NDArray[np.float16],
    pooled_ba: NDArray[np.float16],
    labels: NDArray[np.int8],
    checkpoint_id: str | None,
) -> None:
    checkpoint_sha256 = _file_sha256(args.checkpoint)
    write_kd_targets(
        args.output,
        node_ids=node_ids,
        pair_anchor_idx=context.anchor_idx,
        pair_partner_idx=context.partner_idx,
        anchor_offsets=context.anchor_offsets,
        teacher_logit=teacher_logit,
        teacher_pooled_ab=pooled_ab,
        teacher_pooled_ba=pooled_ba,
        is_near=context.is_near,
        pair_label=labels,
        truth_graph_sha256=_oracle_truth_graph_sha256(truth_graph),
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_id=checkpoint_id,
        k_near=args.k_near,
        k_rand=args.k_rand,
        seed=args.seed,
    )
    report = build_stats_report(teacher_logit, labels, context.is_near)
    write_stats_report(args.output, report)


if __name__ == "__main__":
    main()

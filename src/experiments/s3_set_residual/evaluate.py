"""S3 evaluation: paired test-bucket readouts for the B0+set-residual arms.

Loads a trained `SetResidualModel` checkpoint (`best.pt`, written by Task 3's
`train.py`) and scores three arms over `test_node_buckets.pkl` -- B0 (frozen
base logits, delta forced to 0, always computed as the paired baseline), MODEL
(base + trained delta), and optionally SHUF (same checkpoint, within-set
feature permutation fed as `x_set`) -- reusing S2's evaluation substrate
throughout: `build_eval_context` for the test-side graph/features/buckets/
bucket-reference state, and `s2_latent_topology.evaluate`'s private per-set
readout, bootstrap, and macro-aggregation helpers (imported, not copied).

Base logits come from the B0 candidate-universe artifact's raw `logit` column,
never its sigmoid probabilities: the residual is additive in logit space
(`total_logit = base_logit + delta`), so probabilities are only formed once at
the end via `sigmoid(total_logit)`. A pair absent from the artifact is
excluded from AUPRC and assigned probability 0.0 in every assembled graph
(S2's convention for a missing B0 row), with missing counts reported.

Determinism: SHUF's permutation and each arm's per-metric bootstrap CI reuse
`seed` via the same `torch.Generator`/`np.random.default_rng` entropy-tuple
conventions `s2_latent_topology.evaluate` already uses. The paired-delta
bootstrap (MODEL-B0, SHUF-B0) is seeded independently at a fixed constant
(`_PAIRED_DELTA_SEED = 20260818`, 10k resamples), per the S3 plan's pinned
statistics protocol -- this one is not a CLI knob, so pooling reports from
different `--seed` eval runs still bootstraps from the same stream.

Checkpoint contract (agreed with Task 3, `load_residual_checkpoint`): `best.pt`
is a plain dict with keys `model_state` (a `state_dict`) and `config` (a dict
of `ResidualConfig` field values, `mode` plus any subset of the numeric dims --
omitted dims fall back to `ResidualConfig`'s own defaults).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import networkx as nx
import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score

from src.data.features import FeatureStore
from src.eval.assembly import assemble_graph, density_matched_threshold
from src.eval.graph_metrics import STATISTICS, compute_graph_similarity, compute_relative_density
from src.experiments.s2_latent_topology import evaluate as s2eval
from src.experiments.s2_latent_topology.evaluate import EvalContext, build_eval_context
from src.experiments.s3_set_residual.model import ResidualConfig, SetResidualModel
from src.score_universe import _checkpoint_id, load_scores, validate_artifact_precision

_FEATURES_SUBDIR = Path("features") / "frozen_node_features_1024"
_ARM_CODES_S3: dict[str, int] = {"b0": 1, "model": 2, "shuf": 3}
_CONTRAST_CODES: dict[str, int] = {"model_minus_b0": 101, "shuf_minus_b0": 102}
_DELTA_METRIC_CODES: dict[str, int] = {"auprc": 1, "gs": 2}
_PAIRED_DELTA_SEED = 20260818
_PAIRED_DELTA_N_BOOT = 10_000
_ECE_BINS = 15

__all__ = [
    "aggregate_reports",
    "load_residual_checkpoint",
    "register_eval_args",
    "run_eval",
    "run_s3_eval",
]


# --------------------------------------------------------------------------- checkpoint loading


def load_residual_checkpoint(
    run_dir: Path, device: torch.device
) -> tuple[SetResidualModel, ResidualConfig, str]:
    """Load `<run_dir>/best.pt` per the Task 3 checkpoint contract.

    `best.pt` must be a dict with keys `model_state` (a state dict) and
    `config` (a dict of `ResidualConfig` field values; `mode` is required,
    every other field falls back to `ResidualConfig`'s own default when
    absent).

    Args:
        run_dir: Directory containing `best.pt`.
        device: Device the returned model is moved to (`eval()` mode).

    Returns:
        `(model, cfg, checkpoint_id)`; `checkpoint_id` is a state-dict digest
        (`score_universe._checkpoint_id`), independent of whatever identity
        (if any) the checkpoint payload itself carries.

    Raises:
        ValueError: If `best.pt` is missing `model_state` or `config`.
    """
    path = run_dir / "best.pt"
    payload = cast(dict[str, object], torch.load(path, map_location="cpu", weights_only=True))
    if "model_state" not in payload or "config" not in payload:
        raise ValueError(f"{path}: expected keys 'model_state' and 'config'")
    config_payload = cast(dict[str, object], payload["config"])
    mode = cast(str, config_payload["mode"])
    if mode not in ("res", "pair", "diag"):
        raise ValueError(f"{path}: config.mode must be 'res', 'pair', or 'diag', got {mode!r}")

    defaults = ResidualConfig(mode="res")
    cfg = ResidualConfig(
        mode=cast(Literal["res", "pair", "diag"], mode),
        d_in=cast(int, config_payload.get("d_in", defaults.d_in)),
        d_model=cast(int, config_payload.get("d_model", defaults.d_model)),
        sab_layers=cast(int, config_payload.get("sab_layers", defaults.sab_layers)),
        heads=cast(int, config_payload.get("heads", defaults.heads)),
        pma_seeds=cast(int, config_payload.get("pma_seeds", defaults.pma_seeds)),
        p_dim=cast(int, config_payload.get("p_dim", defaults.p_dim)),
        head_hidden=cast(int, config_payload.get("head_hidden", defaults.head_hidden)),
    )
    model_state = cast(dict[str, torch.Tensor], payload["model_state"])
    model = SetResidualModel(cfg)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model, cfg, _checkpoint_id(model_state)


# --------------------------------------------------------------------------- base-logit gather


def _b0_logit_matrix(
    ctx: EvalContext,
    base_logit: NDArray[np.float32],
    node_index: dict[str, int],
    ordered_set: list[str],
) -> tuple[NDArray[np.float64], NDArray[np.bool_], int]:
    """Gather one node set's fp64 base-logit matrix from the B0 artifact via `ctx.b0_lookup`.

    Mirrors `s2_latent_topology.evaluate._b0_prob_matrix`'s row-lookup approach
    but reads raw logits (`base_logit`) instead of `ctx.b0_probs`, and returns
    the missing-pair mask alongside the matrix instead of pre-filling 0.0 --
    callers decide the fill (0.0 probability for assembly, exclusion for AUPRC).

    Returns:
        `(logit_matrix, missing_mask, missing_count)`: `(n, n)` fp64
        zero-diagonal symmetric logit matrix (missing entries hold an
        arbitrary 0.0 placeholder -- always masked by `missing_mask` before
        use), the matching `(n, n)` bool symmetric zero-diagonal missing mask,
        and the `i < j` missing-pair count.
    """
    assert ctx.b0_lookup is not None
    n = len(ordered_set)
    n_test = len(ctx.nodes)
    gidx = np.array([node_index[node_id] for node_id in ordered_set], dtype=np.int64)
    iu, iv = np.triu_indices(n, k=1)
    flat = gidx[iu] * n_test + gidx[iv]
    rows = ctx.b0_lookup[flat]
    missing = rows < 0
    vals = np.where(missing, 0.0, base_logit[np.clip(rows, 0, None)].astype(np.float64))

    mat = np.zeros((n, n), dtype=np.float64)
    mat[iu, iv] = vals
    mat[iv, iu] = vals
    miss_mat = np.zeros((n, n), dtype=bool)
    miss_mat[iu, iv] = missing
    miss_mat[iv, iu] = missing
    return mat, miss_mat, int(missing.sum())


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically plain sigmoid (logits here are bounded; no overflow guard needed)."""
    result: NDArray[np.float64] = 1.0 / (1.0 + np.exp(-x))
    return result


def _all_pairs_tensor(b_count: int, n: int, device: torch.device) -> torch.Tensor:
    """Build every `i < j` pair for `b_count` same-size sets as a `(P, 3)` long tensor.

    Row order is `set_row`-major, then `np.triu_indices(n, k=1)` order within
    a set -- `_delta_to_matrix` relies on this exact order to scatter the
    model's flat output back into `(B, n, n)` matrices.
    """
    iu, iv = np.triu_indices(n, k=1)
    set_row = np.repeat(np.arange(b_count, dtype=np.int64), len(iu))
    i_arr = np.tile(iu, b_count)
    j_arr = np.tile(iv, b_count)
    pairs = np.stack([set_row, i_arr, j_arr], axis=1)
    return torch.as_tensor(pairs, dtype=torch.long, device=device)


def _delta_to_matrix(delta: torch.Tensor, *, b_count: int, n: int) -> NDArray[np.float64]:
    """Scatter a flat `(B * C(n,2),)` delta tensor (from `_all_pairs_tensor` order) to `(B,n,n)`."""
    iu, iv = np.triu_indices(n, k=1)
    arr = delta.detach().cpu().numpy().astype(np.float64).reshape(b_count, len(iu))
    mat = np.zeros((b_count, n, n), dtype=np.float64)
    mat[:, iu, iv] = arr
    mat[:, iv, iu] = arr
    return mat


def _run_arm_probs(
    model: SetResidualModel,
    x: torch.Tensor,
    mask: torch.Tensor,
    pairs_tensor: torch.Tensor,
    *,
    x_set: torch.Tensor | None,
    base_logit_stack: NDArray[np.float64],
    missing_stack: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Run one arm's forward pass and return its `(B, n, n)` fp64 probability matrix.

    `total_logit = base_logit_stack + delta`; missing pairs are forced to
    probability 0.0 regardless of `delta` (a missing B0 row has no real total
    logit to speak of).
    """
    b_count, n = x.shape[0], x.shape[1]
    with torch.no_grad():
        delta = model(x, mask, pairs_tensor, x_set=x_set)
    total_logit = base_logit_stack + _delta_to_matrix(delta, b_count=b_count, n=n)
    return np.where(missing_stack, 0.0, _sigmoid(total_logit))


# --------------------------------------------------------------------------- per-set readout


def _compute_set_readout_s3(
    *,
    probs: NDArray[np.float64],
    true_adj: NDArray[np.float64],
    node_ids: list[str],
    ref_subgraph: nx.Graph,
    target_edges: int,
    missing_mask: NDArray[np.bool_],
) -> s2eval._SetReadout:
    """S2's per-set readout, with missing B0 pairs excluded from AUPRC (not S2's default).

    Assembly, GS/RD, Spearman, hub recall, and free densities all use `probs`
    as given (missing pairs already forced to 0.0 by the caller -- S2's
    "probability 0.0 in assembled graphs" convention); only AUPRC additionally
    drops missing rows entirely, per the S3 brief's explicit exclusion rule.
    """
    pairs, pair_probs = s2eval._upper_pairs(probs, node_ids)
    n = len(node_ids)
    iu, iv = np.triu_indices(n, k=1)
    pair_labels = true_adj[iu, iv].astype(np.int64)
    present = ~missing_mask[iu, iv]

    threshold = density_matched_threshold(pair_probs, target_edges)
    g_pred = assemble_graph(pairs, pair_probs, threshold=threshold, nodes=node_ids)
    gs = compute_graph_similarity(g_pred, ref_subgraph)
    rd = compute_relative_density(g_pred, ref_subgraph)

    deg_hat = probs.sum(axis=1)
    true_degree = true_adj.sum(axis=1)
    spearman = s2eval._safe_spearman(deg_hat, true_degree)
    hub_recall = s2eval._hub_recall(deg_hat, true_degree, node_ids)

    present_labels = pair_labels[present]
    present_probs = pair_probs[present]
    auprc = (
        float(average_precision_score(present_labels, present_probs))
        if present_labels.size > 0 and len(set(present_labels.tolist())) >= 2
        else None
    )

    free_density_expected = float(pair_probs.sum() / target_edges)
    free_density_hard = float((pair_probs >= 0.5).sum() / target_edges)

    return s2eval._SetReadout(
        gs=gs,
        rd=rd,
        spearman=spearman,
        hub_recall=hub_recall,
        auprc=auprc,
        free_density_expected=free_density_expected,
        free_density_hard=free_density_hard,
        g_pred=g_pred,
    )


@dataclass(frozen=True)
class _SizeResult:
    """One bucket size's per-arm aggregates, raw paired deltas, and calibration pools."""

    arm_blocks: dict[str, dict[str, Any]]
    deltas: dict[str, list[float]]
    n_skipped: int
    missing_pairs: int
    calib_probs: dict[str, NDArray[np.float64]]
    calib_labels: dict[str, NDArray[np.int64]]
    true_edges_total: int


def _process_size(
    *,
    size: int,
    seed: int,
    arm_probs: dict[str, NDArray[np.float64]],
    missing_mask: NDArray[np.bool_],
    ordered_sets: list[list[str]],
    true_adj: NDArray[np.float64],
    ref_subgraphs: list[nx.Graph],
    ref_descs: list[dict[str, NDArray[np.float64]]],
) -> _SizeResult:
    """Compute every arm's readouts, MMD bank, paired deltas, and calibration pool for one size."""
    arm_names = list(arm_probs.keys())
    n = len(ordered_sets[0])
    iu, iv = np.triu_indices(n, k=1)

    metrics_by_arm: dict[str, dict[str, list[float]]] = {
        name: {m: [] for m in s2eval._METRICS} for name in arm_names
    }
    none_counts_by_arm: dict[str, dict[str, int]] = {
        name: dict.fromkeys(s2eval._METRICS, 0) for name in arm_names
    }
    bank_by_arm: dict[str, dict[str, list[NDArray[np.float64]]]] = {
        name: {stat: [] for stat in STATISTICS} for name in arm_names
    }
    deltas: dict[str, list[float]] = defaultdict(list)
    calib_probs: dict[str, list[NDArray[np.float64]]] = {name: [] for name in arm_names}
    calib_labels: dict[str, list[NDArray[np.int64]]] = {name: [] for name in arm_names}

    n_skipped = 0
    missing_pairs = 0
    true_edges_total = 0

    for b, node_ids in enumerate(ordered_sets):
        target_edges = int(round(float(true_adj[b].sum()))) // 2
        true_edges_total += target_edges
        missing_pairs += int(missing_mask[b][iu, iv].sum())
        present = ~missing_mask[b][iu, iv]
        labels_b = true_adj[b][iu, iv].astype(np.int64)
        for name in arm_names:
            probs_b = arm_probs[name][b][iu, iv]
            calib_probs[name].append(probs_b[present])
            calib_labels[name].append(labels_b[present])

        if target_edges == 0:
            n_skipped += 1
            continue

        readouts: dict[str, s2eval._SetReadout] = {}
        for name in arm_names:
            readout = _compute_set_readout_s3(
                probs=arm_probs[name][b],
                true_adj=true_adj[b],
                node_ids=node_ids,
                ref_subgraph=ref_subgraphs[b],
                target_edges=target_edges,
                missing_mask=missing_mask[b],
            )
            readouts[name] = readout
            metrics_by_arm[name]["gs"].append(readout.gs)
            metrics_by_arm[name]["rd"].append(readout.rd)
            metrics_by_arm[name]["free_density_expected"].append(readout.free_density_expected)
            metrics_by_arm[name]["free_density_hard"].append(readout.free_density_hard)
            for metric_name, value in (
                ("spearman", readout.spearman),
                ("hub_recall", readout.hub_recall),
                ("auprc", readout.auprc),
            ):
                if value is None:
                    none_counts_by_arm[name][metric_name] += 1
                else:
                    metrics_by_arm[name][metric_name].append(value)
            bank_d = s2eval._descriptors(readout.g_pred)
            for stat in STATISTICS:
                bank_by_arm[name][stat].append(bank_d[stat])

        if "model" in readouts and "b0" in readouts:
            deltas["model_minus_b0__gs"].append(readouts["model"].gs - readouts["b0"].gs)
            model_auprc, b0_auprc = readouts["model"].auprc, readouts["b0"].auprc
            if model_auprc is not None and b0_auprc is not None:
                deltas["model_minus_b0__auprc"].append(model_auprc - b0_auprc)
        if "shuf" in readouts and "b0" in readouts:
            deltas["shuf_minus_b0__gs"].append(readouts["shuf"].gs - readouts["b0"].gs)
            shuf_auprc, b0_auprc2 = readouts["shuf"].auprc, readouts["b0"].auprc
            if shuf_auprc is not None and b0_auprc2 is not None:
                deltas["shuf_minus_b0__auprc"].append(shuf_auprc - b0_auprc2)

    arm_blocks: dict[str, dict[str, Any]] = {}
    for name in arm_names:
        per_size = s2eval._aggregate_metrics(
            metrics_by_arm[name],
            none_counts_by_arm[name],
            seed=seed,
            arm_code=_ARM_CODES_S3[name],
            size=size,
        )
        per_size["n_sets"] = len(ordered_sets)
        per_size["n_skipped"] = n_skipped
        arm_blocks[name] = {
            "per_size": per_size,
            "mmd": s2eval._aggregate_mmd(bank_by_arm[name], ref_descs),
        }

    return _SizeResult(
        arm_blocks=arm_blocks,
        deltas=dict(deltas),
        n_skipped=n_skipped,
        missing_pairs=missing_pairs,
        calib_probs={
            name: (np.concatenate(calib_probs[name]) if calib_probs[name] else np.array([]))
            for name in arm_names
        },
        calib_labels={
            name: (
                np.concatenate(calib_labels[name])
                if calib_labels[name]
                else np.array([], dtype=np.int64)
            )
            for name in arm_names
        },
        true_edges_total=true_edges_total,
    )


# --------------------------------------------------------------------------- paired deltas


def _ci_entry(values: list[float], *, arm_code: int, size: int, metric_code: int) -> dict[str, Any]:
    """One `(mean, ci_lo, ci_hi, n)` percentile-bootstrap row over raw `values`, `values` kept."""
    mean, lo, hi = s2eval._bootstrap_ci(
        values,
        seed=_PAIRED_DELTA_SEED,
        arm_code=arm_code,
        size=size,
        metric_code=metric_code,
        n_boot=_PAIRED_DELTA_N_BOOT,
    )
    return {
        "values": values,
        "mean": None if math.isnan(mean) else mean,
        "ci_lo": None if math.isnan(lo) else lo,
        "ci_hi": None if math.isnan(hi) else hi,
        "n": len(values),
    }


def _build_paired_deltas(
    delta_raw: dict[str, dict[str, list[float]]], arm_names: list[str], sizes: list[int]
) -> dict[str, Any]:
    """Bootstrap the paired MODEL-B0 (and SHUF-B0, if present) deltas per size and pooled."""
    contrasts = ["model_minus_b0"] + (["shuf_minus_b0"] if "shuf" in arm_names else [])
    out: dict[str, Any] = {}
    for contrast in contrasts:
        arm_code = _CONTRAST_CODES[contrast]
        metric_block: dict[str, Any] = {}
        for metric in ("auprc", "gs"):
            key = f"{contrast}__{metric}"
            by_size: dict[str, Any] = {}
            pooled_values: list[float] = []
            for size in sizes:
                values = delta_raw.get(key, {}).get(str(size), [])
                pooled_values.extend(values)
                by_size[str(size)] = _ci_entry(
                    values, arm_code=arm_code, size=size, metric_code=_DELTA_METRIC_CODES[metric]
                )
            metric_block[metric] = {
                "by_size": by_size,
                "pooled": _ci_entry(
                    pooled_values,
                    arm_code=arm_code,
                    size=0,
                    metric_code=_DELTA_METRIC_CODES[metric],
                ),
            }
        out[contrast] = metric_block
    return out


# --------------------------------------------------------------------------- calibration


def _ece(
    probs: NDArray[np.float64], labels: NDArray[np.int64], *, n_bins: int = _ECE_BINS
) -> float:
    """15-equal-width-bin expected calibration error of `probs` vs `labels`."""
    if probs.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bins[1:-1], right=True), 0, n_bins - 1)
    total = probs.size
    ece = 0.0
    for b in range(n_bins):
        in_bin = bin_idx == b
        count = int(in_bin.sum())
        if count == 0:
            continue
        ece += (count / total) * abs(float(probs[in_bin].mean()) - float(labels[in_bin].mean()))
    return float(ece)


def _calibration_block(
    probs: NDArray[np.float64], labels: NDArray[np.int64], true_edges_total: int
) -> dict[str, Any]:
    """ECE plus implied edge count ratio (`sum(probs) / true_edges`) over present in-set pairs."""
    ece = _ece(probs, labels)
    implied_ratio = float(probs.sum() / true_edges_total) if true_edges_total > 0 else None
    return {
        "ece": None if math.isnan(ece) else ece,
        "implied_edge_count_ratio": implied_ratio,
        "n_pairs": int(probs.size),
    }


# --------------------------------------------------------------------------- full region


def _run_full_region(
    *,
    ctx: EvalContext,
    model: SetResidualModel,
    base_logit: NDArray[np.float32],
    node_index: dict[str, int],
    device: torch.device,
    shuf: bool,
    seed: int,
) -> dict[str, Any]:
    """The optional full 2,018-node test-region secondary readout, B0/MODEL(/SHUF).

    Reuses `s2_latent_topology.evaluate._full_region_block` directly (it only
    needs a probability matrix, `target_edges`, `nodes`, and `ctx` -- none of
    it is entangled with S2's own GEN/DET/AE prior-model arms).
    """
    nodes = ctx.nodes
    n = len(nodes)
    x = ctx.features.unsqueeze(0).to(device)
    mask = torch.ones(1, n, device=device)
    target_edges = ctx.g_simple.number_of_edges()

    base_mat, missing_mat, missing_count = _b0_logit_matrix(ctx, base_logit, node_index, nodes)
    base_stack = base_mat[None, ...]
    missing_stack = missing_mat[None, ...]
    pairs_tensor = _all_pairs_tensor(1, n, device)

    b0_probs = np.where(missing_stack, 0.0, _sigmoid(base_stack))[0]
    model_probs = _run_arm_probs(
        model,
        x,
        mask,
        pairs_tensor,
        x_set=None,
        base_logit_stack=base_stack,
        missing_stack=missing_stack,
    )[0]

    result: dict[str, Any] = {
        "missing_pairs": missing_count,
        "b0": s2eval._full_region_block(b0_probs, target_edges=target_edges, nodes=nodes, ctx=ctx),
        "model": s2eval._full_region_block(
            model_probs, target_edges=target_edges, nodes=nodes, ctx=ctx
        ),
    }
    if shuf:
        x_shuf = s2eval._shuffle_features_within_sets(x, seed=seed, size=n)
        shuf_probs = _run_arm_probs(
            model,
            x,
            mask,
            pairs_tensor,
            x_set=x_shuf,
            base_logit_stack=base_stack,
            missing_stack=missing_stack,
        )[0]
        result["shuf"] = s2eval._full_region_block(
            shuf_probs, target_edges=target_edges, nodes=nodes, ctx=ctx
        )
    return result


# --------------------------------------------------------------------------- orchestration


def run_s3_eval(
    *,
    ctx: EvalContext,
    model: SetResidualModel,
    cfg: ResidualConfig,
    base_logit: NDArray[np.float32],
    seed: int,
    device: str,
    shuf: bool,
    sanity_zero_init: bool,
    full_region: bool,
) -> dict[str, Any]:
    """Run every S3 arm over every test-side bucket and return the JSON-ready report payload.

    Args:
        ctx: The S2 evaluation context, built with `b0_universe` set (required
            here -- B0 is the always-computed paired baseline).
        model: The trained (or, for a sanity probe, freshly zero-init)
            `SetResidualModel`.
        cfg: The model's `ResidualConfig` (used to build a fresh zero-init
            twin for `--sanity-zero-init`).
        base_logit: The B0 candidate-universe artifact's raw fp32 `logit` column.
        seed: Base seed for SHUF's permutation and each arm's per-metric
            bootstrap CI (paired-delta CIs use the fixed `_PAIRED_DELTA_SEED`
            regardless of this value).
        device: Torch device string.
        shuf: Whether to additionally run the SHUF eval-only arm.
        sanity_zero_init: Whether to run the M0 sanity gate: a fresh zero-init
            twin of `model` must reproduce the B0 arm's probabilities exactly.
        full_region: Whether to additionally run the full-test-region secondary readout.

    Returns:
        The JSON-serializable report payload (`meta`, `arms`, `paired_deltas`,
        `calibration`, optionally `sanity_zero_init`/`full_region`).

    Raises:
        ValueError: If `ctx` has no B0 lookup, or the zero-init sanity gate fails.
    """
    if ctx.b0_lookup is None:
        raise ValueError("run_s3_eval requires an EvalContext built with b0_universe set")
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    node_index = {node_id: i for i, node_id in enumerate(ctx.nodes)}
    sizes = sorted(ctx.buckets)
    arm_names = ["b0", "model"] + (["shuf"] if shuf else [])

    zero_model: SetResidualModel | None = None
    if sanity_zero_init:
        zero_model = SetResidualModel(cfg).to(dev)
        zero_model.eval()

    arms_by_size: dict[str, dict[int, dict[str, Any]]] = {name: {} for name in arm_names}
    delta_raw: dict[str, dict[str, list[float]]] = defaultdict(dict)
    calib_probs_all: dict[str, list[NDArray[np.float64]]] = {name: [] for name in arm_names}
    calib_labels_all: dict[str, list[NDArray[np.int64]]] = {name: [] for name in arm_names}
    missing_pairs_total = 0
    true_edges_total = 0
    sanity_ok = True
    sanity_max_diff: dict[str, float] = {}

    for size in sizes:
        x, adj, mask, ordered_sets = s2eval._build_size_batch(
            ctx, ctx.buckets[size], node_index, dev
        )
        true_adj = adj.detach().cpu().numpy().astype(np.float64)
        b_count, n = x.shape[0], x.shape[1]

        base_mats = [_b0_logit_matrix(ctx, base_logit, node_index, s) for s in ordered_sets]
        base_stack = np.stack([m[0] for m in base_mats])
        missing_stack = np.stack([m[1] for m in base_mats])
        pairs_tensor = _all_pairs_tensor(b_count, n, dev)

        arm_probs: dict[str, NDArray[np.float64]] = {
            "b0": np.where(missing_stack, 0.0, _sigmoid(base_stack))
        }
        arm_probs["model"] = _run_arm_probs(
            model,
            x,
            mask,
            pairs_tensor,
            x_set=None,
            base_logit_stack=base_stack,
            missing_stack=missing_stack,
        )
        if shuf:
            x_shuf = s2eval._shuffle_features_within_sets(x, seed=seed, size=size)
            arm_probs["shuf"] = _run_arm_probs(
                model,
                x,
                mask,
                pairs_tensor,
                x_set=x_shuf,
                base_logit_stack=base_stack,
                missing_stack=missing_stack,
            )

        if zero_model is not None:
            zero_probs = _run_arm_probs(
                zero_model,
                x,
                mask,
                pairs_tensor,
                x_set=None,
                base_logit_stack=base_stack,
                missing_stack=missing_stack,
            )
            ok = bool(np.array_equal(zero_probs, arm_probs["b0"]))
            sanity_ok = sanity_ok and ok
            sanity_max_diff[str(size)] = float(np.max(np.abs(zero_probs - arm_probs["b0"])))

        result = _process_size(
            size=size,
            seed=seed,
            arm_probs=arm_probs,
            missing_mask=missing_stack,
            ordered_sets=ordered_sets,
            true_adj=true_adj,
            ref_subgraphs=ctx.bucket_ref.ref_subgraphs[size],
            ref_descs=ctx.bucket_ref.ref_descriptors[size],
        )
        for name in arm_names:
            arms_by_size[name][size] = result.arm_blocks[name]
            calib_probs_all[name].append(result.calib_probs[name])
            calib_labels_all[name].append(result.calib_labels[name])
        for key, values in result.deltas.items():
            delta_raw[key][str(size)] = values
        missing_pairs_total += result.missing_pairs
        true_edges_total += result.true_edges_total

    arms: dict[str, Any] = {}
    for name in arm_names:
        by_size = arms_by_size[name]
        arms[name] = {
            "per_size": {str(s): by_size[s]["per_size"] for s in sizes},
            "macro": s2eval._macro_over_sizes(by_size, sizes),
            "mmd": {str(s): by_size[s]["mmd"] for s in sizes},
            "mmd_macro": s2eval._mmd_macro_over_sizes(by_size, sizes),
        }

    paired_deltas = _build_paired_deltas(delta_raw, arm_names, sizes)
    calibration = {
        name: _calibration_block(
            np.concatenate(calib_probs_all[name]) if calib_probs_all[name] else np.array([]),
            (
                np.concatenate(calib_labels_all[name])
                if calib_labels_all[name]
                else np.array([], dtype=np.int64)
            ),
            true_edges_total,
        )
        for name in arm_names
    }

    payload: dict[str, Any] = {
        "meta": {
            "sizes": sizes,
            "seed": seed,
            "device": device,
            "mode": cfg.mode,
            "arms": arm_names,
            "featureless_in_test": ctx.featureless_in_test,
            "missing_pairs_total": missing_pairs_total,
            "true_edges_total": true_edges_total,
        },
        "arms": arms,
        "paired_deltas": paired_deltas,
        "calibration": calibration,
    }

    if sanity_zero_init:
        if not sanity_ok:
            raise ValueError(
                "sanity-zero-init (M0 gate) failed: a fresh zero-init SetResidualModel does not "
                f"reproduce the B0 arm's probabilities exactly; max abs diff per size = "
                f"{sanity_max_diff}"
            )
        payload["sanity_zero_init"] = {
            "ran": True,
            "passed": True,
            "max_abs_diff_by_size": sanity_max_diff,
        }

    if full_region:
        payload["full_region"] = _run_full_region(
            ctx=ctx,
            model=model,
            base_logit=base_logit,
            node_index=node_index,
            device=dev,
            shuf=shuf,
            seed=seed,
        )

    return payload


def aggregate_reports(paths: Sequence[Path]) -> dict[str, Any]:
    """Pool several `report.json` files' raw per-set paired deltas into pooled bootstrap CIs.

    Concatenates every input report's `paired_deltas.<contrast>.<metric>.by_size.<size>.values`
    (the raw per-set deltas `run_s3_eval` persists, not just their aggregated
    mean/CI) sharing a `(contrast, metric, size)` key, then re-bootstraps at
    the pooled scale -- a true set-level pool across seeds/arms, not a pool of
    per-report means.

    Args:
        paths: `report.json` paths to pool (e.g. one per seed).

    Returns:
        `{"source_reports": [...], "paired_deltas": {contrast: {metric: {"by_size": ...,
        "pooled": ...}}}}`.
    """
    pooled_raw: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in paths:
        report = json.loads(Path(path).read_text())
        for contrast, metrics in report.get("paired_deltas", {}).items():
            for metric, block in metrics.items():
                for size_key, entry in block.get("by_size", {}).items():
                    pooled_raw[(contrast, metric)][size_key].extend(entry["values"])

    contrasts: dict[str, Any] = {}
    for (contrast, metric), by_size_values in pooled_raw.items():
        arm_code = _CONTRAST_CODES.get(contrast, 199)
        metric_code = _DELTA_METRIC_CODES.get(metric, 9)
        by_size: dict[str, Any] = {}
        pooled_values: list[float] = []
        for size_key, values in by_size_values.items():
            pooled_values.extend(values)
            by_size[size_key] = _ci_entry(
                values, arm_code=arm_code, size=int(size_key), metric_code=metric_code
            )
        contrasts.setdefault(contrast, {})[metric] = {
            "by_size": by_size,
            "pooled": _ci_entry(pooled_values, arm_code=arm_code, size=0, metric_code=metric_code),
        }

    return {"source_reports": [str(p) for p in paths], "paired_deltas": contrasts}


# --------------------------------------------------------------------------- CLI


def register_eval_args(parser: argparse.ArgumentParser) -> None:
    """Register the S3 eval stage's exclusive CLI flags.

    `run_eval` also reads `args.data_root`, `args.strategy`, `args.device`,
    and `args.seed`, which this function does *not* register -- per the S2 CLI
    convention (`s2_latent_topology.cli.build_parser`), those four are shared,
    stage-independent flags the top-level parser registers once; this function
    only adds flags exclusive to `--stage eval`.
    """
    parser.add_argument("--run-dir", type=Path, default=None, help="Trained S3 run dir (best.pt)")
    parser.add_argument(
        "--b0-universe", type=Path, default=None, help="B0 candidate-universe scores artifact"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="report.json path (default: <run-dir>/report.json)",
    )
    parser.add_argument("--shuf", action="store_true", help="Also run the SHUF eval-only arm")
    parser.add_argument(
        "--sanity-zero-init",
        action="store_true",
        help="M0 gate: assert a fresh zero-init model reproduces B0 exactly",
    )
    parser.add_argument(
        "--full-region", action="store_true", help="Also run the full test-region secondary readout"
    )
    parser.add_argument(
        "--aggregate",
        type=Path,
        nargs="+",
        default=None,
        help="Pool these report.json files into pooled_report.json instead of evaluating",
    )


def run_eval(args: argparse.Namespace) -> Path:
    """Run the S3 eval stage (or, with `--aggregate`, pool existing reports) and write JSON.

    Args:
        args: Parsed CLI namespace; see `register_eval_args` plus the shared
            `data_root`/`strategy`/`device`/`seed` flags it assumes exist.

    Returns:
        The written `report.json` (or `pooled_report.json`) path.

    Raises:
        ValueError: If `--run-dir`/`--b0-universe` are missing and
            `--aggregate` was not given, or the M0 sanity gate fails.
    """
    if args.aggregate:
        pooled = aggregate_reports(args.aggregate)
        out = (
            args.output
            if args.output is not None
            else args.aggregate[0].parent / "pooled_report.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(pooled, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return cast(Path, out)

    if args.run_dir is None or args.b0_universe is None:
        raise ValueError(
            "run_eval requires --run-dir and --b0-universe unless --aggregate is given"
        )

    device = torch.device(args.device)
    model, cfg, checkpoint_id = load_residual_checkpoint(args.run_dir, device)

    store = FeatureStore(args.data_root / _FEATURES_SUBDIR)
    cache_dir = args.run_dir / "eval_cache"
    ctx = build_eval_context(
        data_root=args.data_root,
        strategy=args.strategy,
        store=store,
        cache_dir=cache_dir,
        b0_universe=args.b0_universe,
    )
    artifact = load_scores(args.b0_universe)
    validate_artifact_precision(artifact, label=str(args.b0_universe))

    payload = run_s3_eval(
        ctx=ctx,
        model=model,
        cfg=cfg,
        base_logit=artifact.logit,
        seed=args.seed,
        device=args.device,
        shuf=args.shuf,
        sanity_zero_init=args.sanity_zero_init,
        full_region=args.full_region,
    )
    payload["provenance"] = {
        "run_dir": str(args.run_dir),
        "b0_universe": str(args.b0_universe),
        "data_root": str(args.data_root),
        "strategy": args.strategy,
        "checkpoint_id": checkpoint_id,
    }

    out = args.output if args.output is not None else args.run_dir / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return cast(Path, out)

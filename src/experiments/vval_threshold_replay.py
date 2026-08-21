r"""Offline acceptance replay for the sampled V_val topology threshold estimator.

Per-epoch V_val topology validation (`src.eval.val_topology`) scores only the
ball-union pairs U plus a pinned uniform complement sample, and estimates the
full-universe density-matched assembly threshold via
`estimated_density_matched_threshold` rather than computing it exactly. This
module replays that estimator offline against a cached FULL-universe V_val
score artifact (produced by `python -m src.score_universe score --pairs
val_region ...`) so the sampled estimator's error can be measured directly,
before any training run depends on it: it compares the exact full-universe
threshold and its resulting metrics against the sampled estimator over many
independent complement re-draws.

CLI::

    python -m src.experiments.vval_threshold_replay \
        --scores scores/full_val_region.npz \
        --data-root data --strategy breadth_first \
        [--redraws 100] [--sample-size 200000] [--base-seed 0] \
        [--output outputs/vval_threshold_replay.json]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from src.data.artifacts import load_benchmark
from src.data.val_region import (
    ValRegionSplit,
    derive_val_region_split,
    val_ball_union_universe,
    val_universe_arrays,
)
from src.eval.assembly import (
    assemble_graph,
    density_matched_threshold,
    estimated_density_matched_threshold,
)
from src.eval.graph_metrics import (
    BucketedMMDReport,
    MMDConfig,
    evaluate_assembled_graph_with_reference,
)
from src.eval.val_topology import build_val_topology_reference
from src.score_universe import load_scores, validate_artifact_precision

_BENCHMARK_SUBDIR = Path("benchmark_2025_neurips")
_METRIC_NAMES: tuple[str, ...] = ("gs", "rd", "degree_mmd", "clustering_mmd", "spectral_mmd")


def _metrics_dict(report: BucketedMMDReport) -> dict[str, float]:
    """Pull the five topology metrics out of a `BucketedMMDReport`."""
    return {
        "gs": float(report.graph_similarity),
        "rd": float(report.relative_density),
        "degree_mmd": float(report.mmd_ratio["degree"]),
        "clustering_mmd": float(report.mmd_ratio["clustering"]),
        "spectral_mmd": float(report.mmd_ratio["spectral"]),
    }


def _summarize_abs_deltas(deltas: list[float]) -> dict[str, float]:
    """Summarize signed per-redraw deltas as {p50, p95, max_abs} over absolute value."""
    if not deltas:
        return {"p50": 0.0, "p95": 0.0, "max_abs": 0.0}
    abs_arr = np.abs(np.asarray(deltas, dtype=np.float64))
    return {
        "p50": float(np.percentile(abs_arr, 50)),
        "p95": float(np.percentile(abs_arr, 95)),
        "max_abs": float(np.max(abs_arr)),
    }


def replay_report(
    probs: NDArray[np.float64],
    split: ValRegionSplit,
    *,
    redraws: int,
    sample_size: int,
    base_seed: int = 0,
) -> dict[str, object]:
    """Replay the sampled V_val threshold estimator against the exact full-universe value.

    `probs` must be aligned to `val_universe_arrays(split.v_val)` row order
    (the same complete `C(n,2)+n` universe `src.score_universe`'s
    ``val_region`` pairs source scores). The exact full-universe
    density-matched threshold and its resulting assembly are computed once;
    the sampled `estimated_density_matched_threshold` estimator is then
    replayed `redraws` times, each drawing its own uniform without-replacement
    complement sample of `sample_size` rows (capped at the true complement
    size) from a fresh `np.random.default_rng(base_seed + r)`.

    Caveat baked into this replay: the exact-threshold assembly (step 2) has
    edges outside the ball-union U that the U-only assembly (steps 4/6) never
    realizes, so the two predicted graphs differ globally. The five topology
    metrics, however, are computed from `evaluate_assembled_graph_with_reference`,
    which only ever reads bucket-induced subgraphs -- and every bucket is a
    subset of U by construction -- so an assembly's edges outside U never
    reach the metrics. This is what lets `exact_threshold_u_assembly` reproduce
    `exact`'s five metrics bit-for-bit despite assembling a strictly smaller
    graph, and what lets `redraw_deltas` be read as isolating threshold error
    alone rather than U-vs-full assembly error.

    Args:
        probs: Full-universe predicted probabilities, aligned to
            `val_universe_arrays(split.v_val)` row order.
        split: The derived V_val region split.
        redraws: Number of independent complement re-draws to replay.
        sample_size: Complement sample size per redraw (capped at the true
            complement size).
        base_seed: Redraw `r` uses `np.random.default_rng(base_seed + r)`.

    Returns:
        A JSON-serializable report; see module tests for the exact schema.

    Raises:
        ValueError: If `probs` has the wrong length for `n*(n+1)/2` (with
            `n = len(sorted(split.v_val))`), contains a non-finite value, or
            contains a value outside `[0, 1]`.
    """
    probs = np.asarray(probs, dtype=np.float64)
    reference = build_val_topology_reference(split)
    nodes = reference.nodes
    n = len(nodes)
    expected_rows = n * (n + 1) // 2
    if probs.shape != (expected_rows,):
        raise ValueError(
            f"probs must have {expected_rows} rows (n*(n+1)/2 for n={n}), got shape {probs.shape}"
        )
    if not np.all(np.isfinite(probs)):
        raise ValueError("probs must be finite")
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("probs must lie in [0, 1]")

    u_full, v_full = val_universe_arrays(split.v_val)
    non_self_mask = u_full != v_full
    full_non_self_probs = probs[non_self_mask]

    # --- exact block: the full-universe density-matched threshold and assembly ---
    tau_full = density_matched_threshold(full_non_self_probs, reference.target_edges)
    full_pairs = [
        (nodes[u], nodes[v]) for u, v in zip(u_full.tolist(), v_full.tolist(), strict=True)
    ]
    g_exact = assemble_graph(full_pairs, probs, threshold=tau_full, nodes=nodes)
    exact_report = evaluate_assembled_graph_with_reference(
        g_exact, reference.bucket_ref, MMDConfig()
    )
    exact_admitted = int(np.sum(full_non_self_probs >= tau_full))
    exact_metrics = _metrics_dict(exact_report)

    # --- map U pairs and the complement onto full-universe row positions ---
    universe = val_ball_union_universe(split)
    full_encoded = u_full.astype(np.int64) * n + v_full.astype(np.int64)
    sort_order = np.argsort(full_encoded)
    sorted_encoded = full_encoded[sort_order]

    def _row_index(encoded: NDArray[np.int64]) -> NDArray[np.int64]:
        pos = np.searchsorted(sorted_encoded, encoded)
        pos = np.clip(pos, 0, sorted_encoded.size - 1)
        if not np.array_equal(sorted_encoded[pos], encoded):
            raise ValueError("pair-encoding lookup failed to match full-universe rows")
        result: NDArray[np.int64] = sort_order[pos]
        return result

    u_encoded = universe.u_idx.astype(np.int64) * n + universe.v_idx.astype(np.int64)
    u_row_idx = _row_index(u_encoded)
    u_probs = probs[u_row_idx]
    u_pairs = [
        (nodes[u], nodes[v])
        for u, v in zip(universe.u_idx.tolist(), universe.v_idx.tolist(), strict=True)
    ]
    u_non_self_mask = universe.u_idx != universe.v_idx
    u_non_self_probs = u_probs[u_non_self_mask]
    u_non_self_encoded = u_encoded[u_non_self_mask]

    full_non_self_encoded = full_encoded[non_self_mask]
    complement_encoded = np.setdiff1d(full_non_self_encoded, u_non_self_encoded, assume_unique=True)
    if complement_encoded.size != universe.complement_total:
        raise ValueError(
            "complement size mismatch: val_ball_union_universe reports "
            f"complement_total={universe.complement_total}, but the replay's own "
            f"non-self-minus-U derivation found {complement_encoded.size}"
        )
    complement_total = universe.complement_total
    complement_pool = (
        probs[_row_index(complement_encoded)]
        if complement_encoded.size
        else np.empty(0, dtype=np.float64)
    )

    # --- exact_threshold_u_assembly: U rows only, at the exact full-universe threshold ---
    g_u_exact_threshold = assemble_graph(u_pairs, u_probs, threshold=tau_full, nodes=nodes)
    exact_threshold_u_report = evaluate_assembled_graph_with_reference(
        g_u_exact_threshold, reference.bucket_ref, MMDConfig()
    )
    exact_threshold_u_metrics = _metrics_dict(exact_threshold_u_report)

    # --- redraws: independent complement samples, each estimating its own threshold ---
    thresholds: list[float] = []
    admitted_deltas: list[float] = []
    metric_deltas: dict[str, list[float]] = {name: [] for name in _METRIC_NAMES}
    for r in range(redraws):
        rng = np.random.default_rng(base_seed + r)
        draw_size = min(sample_size, complement_total)
        if draw_size > 0:
            sampled_positions = rng.choice(complement_pool.size, size=draw_size, replace=False)
            sampled_probs = complement_pool[sampled_positions]
        else:
            sampled_probs = np.empty(0, dtype=np.float64)

        tau_hat = estimated_density_matched_threshold(
            u_non_self_probs, sampled_probs, complement_total, reference.target_edges
        )
        thresholds.append(tau_hat)

        realized_admitted = int(np.sum(full_non_self_probs >= tau_hat))
        admitted_deltas.append(float(realized_admitted - exact_admitted))

        g_redraw = assemble_graph(u_pairs, u_probs, threshold=tau_hat, nodes=nodes)
        redraw_report = evaluate_assembled_graph_with_reference(
            g_redraw, reference.bucket_ref, MMDConfig()
        )
        redraw_metrics = _metrics_dict(redraw_report)
        for name in _METRIC_NAMES:
            metric_deltas[name].append(redraw_metrics[name] - exact_metrics[name])

    redraw_deltas: dict[str, object] = {
        name: _summarize_abs_deltas(metric_deltas[name]) for name in _METRIC_NAMES
    }
    redraw_deltas["admitted_non_self"] = _summarize_abs_deltas(admitted_deltas)
    redraw_deltas["thresholds"] = thresholds

    return {
        "n_val": n,
        "target_edges": reference.target_edges,
        "sample_size": sample_size,
        "redraws": redraws,
        "base_seed": base_seed,
        "complement_total": complement_total,
        "exact": {"threshold": tau_full, "admitted_non_self": exact_admitted, **exact_metrics},
        "exact_threshold_u_assembly": exact_threshold_u_metrics,
        "redraw_deltas": redraw_deltas,
    }


def _load_val_region_split(data_root: Path, strategy: str) -> ValRegionSplit:
    """Rebuild `ValRegionSplit` exactly as `src.score_universe`'s val-universe path does.

    Mirrors `src.score_universe._load_val_region_split`'s derivation (same
    `train_graph.pkl` edges, `train_edges.txt` + `val_edges.txt` label-0 rows,
    `positive_edges.txt` global positive set, default `ValRegionParams()`) via
    the house `load_benchmark` loader.
    """
    benchmark = load_benchmark(data_root / _BENCHMARK_SUBDIR, strategy)
    truth_edges = list(benchmark.split.train_graph.edges())
    benchmark_negatives = [
        pair
        for pair, label in zip(
            benchmark.split.train_pairs.pairs,
            benchmark.split.train_pairs.labels.tolist(),
            strict=True,
        )
        if int(label) == 0
    ] + [
        pair
        for pair, label in zip(
            benchmark.split.val_pairs.pairs, benchmark.split.val_pairs.labels.tolist(), strict=True
        )
        if int(label) == 0
    ]
    return derive_val_region_split(
        benchmark.split.train_nodes, truth_edges, benchmark_negatives, benchmark.positive_edges
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.experiments.vval_threshold_replay",
        description=(
            "Offline acceptance replay: compare the exact full-universe V_val "
            "density-matched threshold against the sampled-complement estimator "
            "over many independent complement re-draws."
        ),
    )
    parser.add_argument(
        "--scores", type=Path, required=True, help="full-universe V_val scores .npz"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--redraws", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=200_000)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None, help="default: alongside --scores")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to `sys.argv[1:]`).

    Raises:
        SystemExit: On argument errors (via `argparse`).
    """
    args = _build_arg_parser().parse_args(argv)

    artifact = load_scores(args.scores)
    validate_artifact_precision(artifact, label=str(args.scores))

    split = _load_val_region_split(args.data_root, args.strategy)
    expected_u_idx, expected_v_idx = val_universe_arrays(split.v_val)
    if not (
        np.array_equal(artifact.u_idx, expected_u_idx)
        and np.array_equal(artifact.v_idx, expected_v_idx)
    ):
        raise ValueError(
            f"{args.scores}: artifact u_idx/v_idx do not match "
            "val_universe_arrays(split.v_val) row order -- was this artifact scored "
            "with --pairs val_region against this exact --data-root/--strategy?"
        )

    probs = expit(artifact.logit.astype(np.float64))
    report = replay_report(
        probs,
        split,
        redraws=args.redraws,
        sample_size=args.sample_size,
        base_seed=args.base_seed,
    )

    output_path = (
        args.output
        if args.output is not None
        else args.scores.parent / ("vval_threshold_replay.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    exact = cast(dict[str, object], report["exact"])
    redraw_deltas = cast(dict[str, object], report["redraw_deltas"])
    print(f"V_val threshold replay: {args.scores}")  # noqa: T201 -- CLI summary
    print(  # noqa: T201 -- CLI summary
        f"  n_val={report['n_val']} target_edges={report['target_edges']} "
        f"complement_total={report['complement_total']} redraws={report['redraws']} "
        f"sample_size={report['sample_size']}"
    )
    print(  # noqa: T201 -- CLI summary
        f"  exact: threshold={cast(float, exact['threshold']):.6g} "
        f"admitted_non_self={exact['admitted_non_self']}"
    )
    for name in (*_METRIC_NAMES, "admitted_non_self"):
        summary = cast(dict[str, float], redraw_deltas[name])
        print(  # noqa: T201 -- CLI summary
            f"  {name} |delta| vs exact: p50={summary['p50']:.6g} p95={summary['p95']:.6g} "
            f"max_abs={summary['max_abs']:.6g}"
        )
    print(f"  wrote {output_path}")  # noqa: T201 -- CLI summary


if __name__ == "__main__":
    main()


__all__ = ["main", "replay_report"]

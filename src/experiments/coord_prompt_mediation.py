"""Causal decomposition of the topology prompt's effect on the assembled graph.

The ``v3_1_coord_gen`` student scores a pair from coordinates it predicts,
``s_hat = g(x_u, x_v)``. Scoring-time interventions already measure *whether*
those coordinates matter (``none`` against ``mean`` and ``gates_off``); they do
not say *through what* they matter. This module answers that from artifacts
alone, with no re-scoring.

Write ``delta_i = logit_none_i - logit_mean_i`` for the per-row causal effect of
replacing the predicted coordinates' content by the training mean. On each
universe ``delta`` is split into three orthogonal parts,

    delta_i = offset + (a_u + a_v + c - offset) + residual_i,

a per-universe scalar ``offset = mean(delta)``, a symmetric node-additive term
fitted by least squares, and the pair-specific residual. Feeding
``logit_mean + <part>`` through the ordinary protocol (select one threshold on
V_val, replay it on every test subgraph) prices each part against the five
topology numbers. The parts are surrogates built from the run's own logits, so
every row here is a **diagnostic attribution, never a deployable arm**: the node
effects are fitted on the universe they are then evaluated on.

The offset row is the load-bearing control. A shift applied to V_val and test
alike is neutral under threshold transfer, but ``delta`` has a different mean on
each universe, and that difference alone moves the replayed operating point.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from sklearn.metrics import average_precision_score, roc_auc_score

from src.eval.fixed_threshold import evaluate_fixed_threshold, select_fixed_threshold
from src.eval.graph_metrics import MMDConfig
from src.experiments.g1_hardened_e2 import load_test_graph, load_test_node_buckets
from src.score_universe import (
    ScoresArtifact,
    _load_val_region_split,
    load_scores,
    validate_artifact_precision,
)

logger = logging.getLogger(__name__)

BENCHMARK_SUBDIR = "benchmark_2025_neurips"
#: Scoring subdirectory of each intervention row, relative to the run directory.
ROW_DIRS: dict[str, str] = {
    "none": "scores",
    "mean": "intervention_mean/scores",
    "gates_off": "intervention_gates_off/scores",
    "mean_relation": "intervention_mean_relation/scores",
}


@dataclass(frozen=True)
class UniverseDecomposition:
    """One universe's split of ``delta`` into offset, node and pair parts.

    Attributes:
        offset: ``mean(delta)`` on this universe.
        node: Mean-centred symmetric node-additive component, row-aligned.
        pair: The least-squares residual, row-aligned.
        additive_r2: Fraction of ``delta``'s variance the node fit explains.
        node_effect: The per-node coefficient ``a``, indexed like ``node_ids``.
        delta: The measured per-row effect itself.
    """

    offset: float
    node: NDArray[np.float64]
    pair: NDArray[np.float64]
    additive_r2: float
    node_effect: NDArray[np.float64]
    delta: NDArray[np.float64]


def load_row(run_dir: Path, row: str, universe: str) -> ScoresArtifact:
    """Load one intervention row's artifact for one universe.

    Args:
        run_dir: Published run directory holding ``scores/`` and ``intervention_*/``.
        row: Key of `ROW_DIRS`.
        universe: ``val_topology``, ``test_topology`` or ``test``.

    Returns:
        The loaded artifact, precision-validated.
    """
    artifact = load_scores(run_dir / ROW_DIRS[row] / f"{universe}.npz")
    validate_artifact_precision(artifact, label=f"{row}/{universe}")
    return artifact


def load_aligned(run_dir: Path, rows: list[str], universe: str) -> dict[str, ScoresArtifact]:
    """Load several rows of one universe and require identical row order.

    Args:
        run_dir: Published run directory.
        rows: Row keys to load.
        universe: Universe name.

    Returns:
        Row key to artifact.

    Raises:
        ValueError: If two rows disagree on the pair order or node vocabulary.
    """
    artifacts = {row: load_row(run_dir, row, universe) for row in rows}
    reference = artifacts[rows[0]]
    for row in rows[1:]:
        other = artifacts[row]
        same = (
            np.array_equal(other.u_idx, reference.u_idx)
            and np.array_equal(other.v_idx, reference.v_idx)
            and np.array_equal(other.node_ids, reference.node_ids)
        )
        if not same:
            raise ValueError(f"{universe}: row {row!r} is not aligned with {rows[0]!r}")
    return artifacts


def decompose(
    u_idx: NDArray[np.integer],
    v_idx: NDArray[np.integer],
    delta: NDArray[np.float64],
    num_nodes: int,
) -> UniverseDecomposition:
    """Split ``delta`` into a scalar offset, node-additive and pair-residual parts.

    The design matrix puts a one in the column of each endpoint (a self-pair
    therefore loads its node twice) plus an intercept column, matching the
    symmetry of the coordinates and of the reader.

    Args:
        u_idx: First endpoint index per row.
        v_idx: Second endpoint index per row.
        delta: The measured per-row effect.
        num_nodes: Size of the artifact's node vocabulary.

    Returns:
        The decomposition; ``node`` and ``pair`` sum with ``offset`` to ``delta``.
    """
    rows = np.arange(len(delta))
    design = coo_matrix(
        (
            np.ones(3 * len(delta)),
            (
                np.concatenate([rows, rows, rows]),
                np.concatenate([u_idx, v_idx, np.full(len(delta), num_nodes)]),
            ),
        ),
        shape=(len(delta), num_nodes + 1),
    ).tocsr()
    solution = cast(
        NDArray[np.float64], lsqr(design, delta, atol=1e-10, btol=1e-10, iter_lim=400)[0]
    )
    fitted = cast(NDArray[np.float64], design @ solution)
    residual = delta - fitted
    total = float(((delta - delta.mean()) ** 2).sum())
    return UniverseDecomposition(
        offset=float(delta.mean()),
        node=fitted - float(fitted.mean()),
        pair=residual,
        additive_r2=1.0 - float((residual**2).sum()) / total,
        node_effect=solution[:num_nodes],
        delta=delta,
    )


def topology_operating_point(  # noqa: PLR0913
    *,
    val_logits: NDArray[np.float64],
    test_logits: NDArray[np.float64],
    val_artifact: ScoresArtifact,
    test_artifact: ScoresArtifact,
    val_graph: nx.Graph,
    val_buckets: dict[int, list[set[str]]],
    test_graph: nx.Graph,
    test_buckets: dict[int, list[set[str]]],
    config: MMDConfig,
) -> dict[str, float]:
    """Select one threshold on V_val and replay it on every test subgraph.

    Args:
        val_logits: V_val topology-universe logits of this row.
        test_logits: Test topology-universe logits of this row.
        val_artifact: V_val artifact supplying the pair order.
        test_artifact: Test artifact supplying the pair order.
        val_graph: The V_val gold graph.
        val_buckets: V_val sampled node sets.
        test_graph: The held-out test graph.
        test_buckets: The benchmark's sampled test node sets.
        config: MMD configuration.

    Returns:
        Threshold and the five reported topology numbers.
    """
    selection = select_fixed_threshold(
        pairs=list(val_artifact.pairs()),
        logits=val_logits,
        g_ref=val_graph,
        buckets=val_buckets,
        config=config,
    )
    _, report = evaluate_fixed_threshold(
        pairs=list(test_artifact.pairs()),
        logits=test_logits,
        g_ref=test_graph,
        buckets=test_buckets,
        threshold=selection.logit_threshold,
        config=config,
    )
    ratios = cast(dict[str, float], report["mmd_ratio"])
    density = cast(dict[str, float], report["relative_density"])["bfs_macro"]
    return {
        "threshold": float(selection.logit_threshold),
        "graph_similarity": float(cast(dict[str, float], report["graph_similarity"])["bfs_macro"]),
        "relative_density": float(density),
        "degree_mmd_ratio": float(ratios["degree"]),
        "clustering_mmd_ratio": float(ratios["clustering"]),
        "spectral_mmd_ratio": float(ratios["spectral"]),
        "geometric_rd": float(np.exp(-abs(np.log(max(density, 1e-12))))),
    }


def matched_density_point(
    *,
    test_logits: NDArray[np.float64],
    test_artifact: ScoresArtifact,
    test_graph: nx.Graph,
    test_buckets: dict[int, list[set[str]]],
    config: MMDConfig,
) -> dict[str, float]:
    """Re-select the threshold on the test universe itself, removing the transfer.

    Test-informed and therefore a diagnostic only: it never selects anything and
    is never a reported operating point. Its purpose is to hold density fixed so
    the shape metrics compare like with like -- MMD ratios move strongly with
    density, so two rows at different relative densities are not comparable.

    Args:
        test_logits: Test topology-universe logits of this row.
        test_artifact: Test artifact supplying the pair order.
        test_graph: The held-out test graph.
        test_buckets: The benchmark's sampled test node sets.
        config: MMD configuration.

    Returns:
        The row's own best-density operating point.
    """
    selection = select_fixed_threshold(
        pairs=list(test_artifact.pairs()),
        logits=test_logits,
        g_ref=test_graph,
        buckets=test_buckets,
        config=config,
    )
    metrics = selection.metrics
    return {
        "threshold": float(selection.logit_threshold),
        "graph_similarity": float(metrics.graph_similarity),
        "relative_density": float(metrics.relative_density),
        "degree_mmd_ratio": float(metrics.mmd_ratio["degree"]),
        "clustering_mmd_ratio": float(metrics.mmd_ratio["clustering"]),
        "spectral_mmd_ratio": float(metrics.mmd_ratio["spectral"]),
    }


def edge_metrics(logits: NDArray[np.float64], labels: NDArray[np.integer]) -> dict[str, float]:
    """Return threshold-free edge metrics of one row.

    Args:
        logits: Row logits on the held-out 1:1 test list.
        labels: Their binary labels.

    Returns:
        AUROC and AUPRC.
    """
    return {
        "auroc": float(roc_auc_score(labels, logits)),
        "auprc": float(average_precision_score(labels, logits)),
    }


def node_effect_on_rows(
    effect: NDArray[np.float64],
    source_nodes: NDArray[np.str_],
    target: ScoresArtifact,
) -> NDArray[np.float64]:
    """Map a fitted per-node effect onto another universe's rows.

    Args:
        effect: Per-node coefficients indexed like ``source_nodes``.
        source_nodes: Node vocabulary the effect was fitted on.
        target: Artifact whose rows the effect is evaluated on.

    Returns:
        ``a_u + a_v`` per target row; nodes absent from the fit contribute zero.
    """
    lookup = {str(node): float(value) for node, value in zip(source_nodes, effect, strict=True)}
    node_ids = [str(node) for node in target.node_ids]
    per_node = np.array([lookup.get(node, 0.0) for node in node_ids], dtype=np.float64)
    return per_node[target.u_idx] + per_node[target.v_idx]


def build_rows(
    val: dict[str, ScoresArtifact],
    test: dict[str, ScoresArtifact],
    val_split: UniverseDecomposition,
    test_split: UniverseDecomposition,
    seed: int,
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Assemble every measured and surrogate row as ``(V_val, test)`` logits.

    Args:
        val: V_val artifacts by row key.
        test: Test artifacts by row key.
        val_split: V_val decomposition.
        test_split: Test decomposition.
        seed: Seed of the within-universe permutation row.

    Returns:
        Row name to its two logit vectors.
    """
    val_mean = val["mean"].logit.astype(np.float64)
    test_mean = test["mean"].logit.astype(np.float64)
    permuted_val = val_split.node + val_split.pair
    permuted_test = test_split.node + test_split.pair
    permuted_val = permuted_val[np.random.default_rng(seed).permutation(len(permuted_val))]
    permuted_test = permuted_test[np.random.default_rng(seed).permutation(len(permuted_test))]
    rows: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {
        "predicted (no intervention)": (
            val["none"].logit.astype(np.float64),
            test["none"].logit.astype(np.float64),
        ),
        "mean coordinates": (val_mean, test_mean),
        "gates off": (
            val["gates_off"].logit.astype(np.float64),
            test["gates_off"].logit.astype(np.float64),
        ),
        "mean relation field": (
            val["mean_relation"].logit.astype(np.float64),
            test["mean_relation"].logit.astype(np.float64),
        ),
        "offset only": (val_mean + val_split.offset, test_mean + test_split.offset),
        "offset + node": (
            val_mean + val_split.offset + val_split.node,
            test_mean + test_split.offset + test_split.node,
        ),
        "offset + pair": (
            val_mean + val_split.offset + val_split.pair,
            test_mean + test_split.offset + test_split.pair,
        ),
        "offset + permuted structure": (
            val_mean + val_split.offset + permuted_val,
            test_mean + test_split.offset + permuted_test,
        ),
    }
    return rows


def run(run_dir: Path, data_root: Path, strategy: str, seed: int) -> dict[str, object]:
    """Decompose one published ``coord_gen`` run and price every part.

    Args:
        run_dir: Published run directory with its intervention subdirectories.
        data_root: Data root holding the benchmark package.
        strategy: Benchmark split strategy.
        seed: Seed of the permutation row.

    Returns:
        The full record: per-universe decomposition statistics and one entry per row.
    """
    keys = list(ROW_DIRS)
    val = load_aligned(run_dir, keys, "val_topology")
    test = load_aligned(run_dir, keys, "test_topology")
    edge = load_aligned(run_dir, keys, "test")
    val_reference, test_reference, edge_reference = val["none"], test["none"], edge["none"]

    val_split = decompose(
        val_reference.u_idx,
        val_reference.v_idx,
        (val["none"].logit - val["mean"].logit).astype(np.float64),
        len(val_reference.node_ids),
    )
    test_split = decompose(
        test_reference.u_idx,
        test_reference.v_idx,
        (test["none"].logit - test["mean"].logit).astype(np.float64),
        len(test_reference.node_ids),
    )

    split = _load_val_region_split(data_root, strategy)
    benchmark_root = data_root / BENCHMARK_SUBDIR
    test_graph = load_test_graph(benchmark_root, strategy)
    test_buckets = load_test_node_buckets(benchmark_root, strategy)
    config = MMDConfig()

    edge_delta = (edge["none"].logit - edge["mean"].logit).astype(np.float64)
    edge_mean = edge["mean"].logit.astype(np.float64)
    edge_node = node_effect_on_rows(
        test_split.node_effect, cast(NDArray[np.str_], test_reference.node_ids), edge_reference
    )
    edge_node = edge_node - float(edge_node.mean())
    edge_logits: dict[str, NDArray[np.float64]] = {
        "predicted (no intervention)": edge["none"].logit.astype(np.float64),
        "mean coordinates": edge_mean,
        "gates off": edge["gates_off"].logit.astype(np.float64),
        "mean relation field": edge["mean_relation"].logit.astype(np.float64),
        "offset only": edge_mean + test_split.offset,
        "offset + node": edge_mean + test_split.offset + edge_node,
        "offset + pair": edge_mean
        + test_split.offset
        + (edge_delta - edge_delta.mean() - edge_node),
        "offset + permuted structure": edge_mean
        + test_split.offset
        + (edge_delta - edge_delta.mean())[
            np.random.default_rng(seed).permutation(len(edge_delta))
        ],
    }

    results: list[dict[str, object]] = []
    for name, (val_logits, test_logits) in build_rows(
        val, test, val_split, test_split, seed
    ).items():
        point: dict[str, object] = dict(
            topology_operating_point(
                val_logits=val_logits,
                test_logits=test_logits,
                val_artifact=val_reference,
                test_artifact=test_reference,
                val_graph=split.build_g_val(),
                val_buckets=split.buckets,
                test_graph=test_graph,
                test_buckets=test_buckets,
                config=config,
            )
        )
        point.update(edge_metrics(edge_logits[name], edge_reference.label))
        point["matched_density"] = matched_density_point(
            test_logits=test_logits,
            test_artifact=test_reference,
            test_graph=test_graph,
            test_buckets=test_buckets,
            config=config,
        )
        matched = cast(dict[str, float], point["matched_density"])
        logger.info(
            "%-30s thr %+.3f  GS %.3f  RD %.3f  MMD %.1f/%.1f/%.1f  AUROC %.3f | "
            "matched density: GS %.3f  RD %.3f  MMD %.1f/%.1f/%.1f",
            name,
            point["threshold"],
            point["graph_similarity"],
            point["relative_density"],
            point["degree_mmd_ratio"],
            point["clustering_mmd_ratio"],
            point["spectral_mmd_ratio"],
            point["auroc"],
            matched["graph_similarity"],
            matched["relative_density"],
            matched["degree_mmd_ratio"],
            matched["clustering_mmd_ratio"],
            matched["spectral_mmd_ratio"],
        )
        results.append({"row": name, **point})

    degrees = np.array(
        [
            float(test_graph.degree(str(n))) if test_graph.has_node(str(n)) else 0.0
            for n in test_reference.node_ids
        ]
    )
    present = degrees > 0
    return {
        "run_dir": str(run_dir),
        "checkpoint_id": str(test_reference.meta.get("checkpoint_id", "")),
        "seed": seed,
        "decomposition": {
            universe: {
                "rows": int(len(part.delta)),
                "nodes": int(len(part.node_effect)),
                "offset": part.offset,
                "delta_sd": float(part.delta.std()),
                "node_sd": float(part.node.std()),
                "pair_sd": float(part.pair.std()),
                "additive_r2": part.additive_r2,
            }
            for universe, part in (("val_topology", val_split), ("test_topology", test_split))
        },
        "offset_gap_val_minus_test": val_split.offset - test_split.offset,
        "test_node_effect_vs_log_degree_r": float(
            np.corrcoef(test_split.node_effect[present], np.log1p(degrees[present]))[0, 1]
        ),
        "rows": results,
    }


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point.

    Args:
        argv: Optional argument vector.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--strategy", default="breadth_first")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    record = run(args.run_dir, args.data_root, args.strategy, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()

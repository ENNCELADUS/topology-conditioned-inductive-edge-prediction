r"""Gate G1 (hardened E2) analysis pipeline: cached-score artifacts -> gate numbers.

Consumes a frozen B0 (and optional B0-alt) candidate-universe scores artifact
(``src.score_universe``) plus the benchmark package (``src.data.artifacts``) and
produces every number the G1 gate report needs: reference-normalized MMD ratios,
official BFS-macro GS/RD, the perturbation diagnostic, a threshold sweep with the
density-matched operating point, a hard/easy/degree-corrected/feature negative
regime table (edge metrics), a preferential-attachment null row (evaluator-side
diagnostic), and assembled-graph rows at the operating point. No model scoring
happens here -- the candidate universe already contains every canonical test
pair, so every negative regime is a row-selection over the cached artifact.

CLI::

    python -m src.experiments.g1_hardened_e2 \
        --universe scores/b0_v31_candidate.npz \
        --alt-universe scores/b0_alt_candidate.npz \
        --data-root data --strategy breadth_first \
        --output-dir outputs/g1 [--seed 0] [--skip-perturbation-check]

Determinism: every random draw is seeded from ``--seed`` via
:func:`numpy.random.default_rng` entropy tuples derived deterministically from
the regime name and target sample size (never from Python's built-in ``hash``,
which is salted per process). Two runs with identical inputs produce a
byte-identical ``g1_results.json`` (``json.dumps(..., sort_keys=True)``; no
wall-clock fields are written into the results payload).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from src.data.features import FeatureStore, build_f0_matrix
from src.eval.assembly import assemble_graph, density_matched_threshold, threshold_sweep
from src.eval.composite import perturbation_check
from src.eval.edge_metrics import EdgeMetrics, compute_edge_metrics
from src.eval.graph_metrics import (
    STATISTICS,
    MMDConfig,
    bootstrap_mmd,
    evaluate_assembled_graph,
    noise_floor,
    strip_self_loops,
)
from src.score_universe import ScoresArtifact, load_scores

logger = logging.getLogger(__name__)

_BENCHMARK_SUBDIR = Path("benchmark_2025_neurips")
_FEATURES_SUBDIR = Path("features") / "frozen_node_features_1024"
_PERCENTILES: tuple[float, ...] = (50, 80, 90, 95, 97.5, 99, 99.5, 99.9)
_REGIME_NAMES: tuple[str, ...] = (
    "easy_uniform",
    "degree_corrected",
    "hard_heuristic",
    "hard_feature",
)
_RATIOS: tuple[int, ...] = (1, 5)
_REGIME_SEED_CODE: dict[str, int] = {
    "easy_uniform": 1,
    "degree_corrected": 2,
    "hard_heuristic": 3,
    "hard_feature": 4,
}
_EDGE_METRIC_THRESHOLD = 0.5

_BREADTH_FIRST_ZERO_DROP_NOTE = (
    "expected missing features {node_004764, node_007050} touch zero labeled pairs "
    "under breadth_first"
)
_NEGATIVE_SAMPLER_COIN_FLIP_NOTE = (
    "training negatives: 50% uniform / 50% degree-corrected per-sample coin flip"
)


# --------------------------------------------------------------------------- benchmark I/O


def load_test_graph(benchmark_root: Path, strategy: str) -> nx.Graph:
    """Unpickle the strategy's held-out reference test graph.

    Reads ``<benchmark_root>/<strategy>/test_graph.pkl`` directly (the pinned
    ``networkx.Graph`` artifact, spec Sec 9); mirrors
    :func:`src.experiments.g2_ceiling.load_test_graph` -- this module only needs
    the test graph and node buckets, so it does not run the full
    ``load_benchmark`` package verification (which also makes the CLI directly
    testable against synthetic data).

    Args:
        benchmark_root: Benchmark package root (e.g. ``data/benchmark_2025_neurips``).
        strategy: Split strategy name.

    Returns:
        The pickled `networkx.Graph` (self-loops not stripped; callers strip via
        `strip_self_loops` as needed).

    Raises:
        TypeError: If the unpickled object is not a `networkx.Graph`.
    """
    path = benchmark_root / strategy / "test_graph.pkl"
    with path.open("rb") as f:
        graph = pickle.load(f)
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"{path}: expected a pickled networkx.Graph, got {type(graph)!r}")
    return graph


def load_test_node_buckets(benchmark_root: Path, strategy: str) -> dict[int, list[set[str]]]:
    """Unpickle the strategy's node-count-keyed evaluation sample buckets.

    Reads ``<benchmark_root>/<strategy>/test_node_buckets.pkl`` directly (spec Sec 9).

    Args:
        benchmark_root: Benchmark package root.
        strategy: Split strategy name.

    Returns:
        Bucket size -> list of node-id sets.

    Raises:
        TypeError: If the unpickled object is not a `dict`.
    """
    path = benchmark_root / strategy / "test_node_buckets.pkl"
    with path.open("rb") as f:
        buckets = pickle.load(f)
    if not isinstance(buckets, dict):
        raise TypeError(f"{path}: expected a pickled dict, got {type(buckets)!r}")
    return cast(dict[int, list[set[str]]], buckets)


# --------------------------------------------------------------------------- validation


def _expected_candidate_rows(n_test_nodes: int) -> int:
    """Return ``C(n_test_nodes, 2) + n_test_nodes``, the candidate-universe row count."""
    return math.comb(n_test_nodes, 2) + n_test_nodes


def validate_universe_artifact(
    artifact: ScoresArtifact, *, strategy: str, n_test_nodes: int, label: str = "universe"
) -> None:
    """Validate a scores artifact is the full labeled candidate universe for `strategy`.

    Mirrors :func:`src.experiments.g2_ceiling.validate_universe`.

    Args:
        artifact: The loaded scores artifact.
        strategy: Expected split strategy name.
        n_test_nodes: Number of test nodes the candidate universe is defined over
            (used to derive the expected row count ``C(n, 2) + n``).
        label: Human-readable artifact label used in error messages.

    Raises:
        ValueError: If ``meta["pairs_source"] != "candidate"``, ``meta["strategy"]``
            does not match `strategy`, the row count does not equal the expected
            candidate-universe size, or any label is outside ``{0, 1}``.
    """
    errors: list[str] = []

    pairs_source = artifact.meta.get("pairs_source")
    if pairs_source != "candidate":
        errors.append(f"{label}: pairs_source expected 'candidate', got {pairs_source!r}")

    meta_strategy = artifact.meta.get("strategy")
    if meta_strategy != strategy:
        errors.append(f"{label}: strategy expected {strategy!r}, got {meta_strategy!r}")

    expected_rows = _expected_candidate_rows(n_test_nodes)
    n_rows = len(artifact.logit)
    if n_rows != expected_rows:
        errors.append(
            f"{label}: row count expected {expected_rows} "
            f"(C({n_test_nodes},2)+{n_test_nodes}), got {n_rows}"
        )

    labels_present = set(np.unique(artifact.label).tolist())
    bad_labels = labels_present - {0, 1}
    if bad_labels:
        errors.append(f"{label}: label values outside {{0,1}} found: {sorted(bad_labels)}")

    if errors:
        raise ValueError("; ".join(errors))


def _remap_indices(
    target_node_ids: Sequence[str], source_node_ids: Sequence[str], indices: NDArray[np.int32]
) -> NDArray[np.int64]:
    """Remap `indices` (into `source_node_ids`) onto positions in `target_node_ids`."""
    position = {node_id: i for i, node_id in enumerate(target_node_ids)}
    try:
        mapping = np.array([position[node_id] for node_id in source_node_ids], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"node id {exc} not present in the primary universe's node_ids") from exc
    return mapping[indices]


def validate_alt_artifact(alt: ScoresArtifact, universe: ScoresArtifact) -> None:
    """Validate that an alt (B0-alt) artifact matches the primary universe exactly.

    Checks equal ``pairs_source``/``strategy``, equal row count, and that every
    row is the SAME canonical pair (compared through each artifact's own
    ``node_ids`` vocabulary, not by string equality of the vocab lists).

    Args:
        alt: The loaded B0-alt scores artifact.
        universe: The loaded, already-validated primary (B0) scores artifact.

    Raises:
        ValueError: On any mismatch; all detected issues are reported together.
    """
    errors: list[str] = []
    for key in ("pairs_source", "strategy"):
        if alt.meta.get(key) != universe.meta.get(key):
            errors.append(
                f"alt-universe meta[{key!r}]={alt.meta.get(key)!r} != "
                f"universe meta[{key!r}]={universe.meta.get(key)!r}"
            )
    if len(alt.logit) != len(universe.logit):
        errors.append(
            f"alt-universe has {len(alt.logit)} rows, universe has {len(universe.logit)} rows"
        )
        raise ValueError("; ".join(errors))

    mapped_u = _remap_indices(universe.node_ids, alt.node_ids, alt.u_idx)
    mapped_v = _remap_indices(universe.node_ids, alt.node_ids, alt.v_idx)
    if not np.array_equal(mapped_u, universe.u_idx.astype(np.int64)) or not np.array_equal(
        mapped_v, universe.v_idx.astype(np.int64)
    ):
        errors.append("alt-universe pair order does not match the primary universe's pair order")
    if errors:
        raise ValueError("; ".join(errors))


# --------------------------------------------------------------------------- shared helpers


def _degree_array(g_simple: nx.Graph, node_ids: Sequence[str]) -> NDArray[np.float64]:
    """Return degrees of `node_ids` in `g_simple`, in `node_ids` order (0 if absent)."""
    degree = dict(g_simple.degree())
    return np.array([float(degree.get(node_id, 0)) for node_id in node_ids], dtype=np.float64)


def _regime_seed_sequence(seed: int, regime: str, n: int) -> tuple[int, int, int]:
    """Deterministic RNG entropy tuple for one (regime, target-size) draw.

    Never uses Python's built-in ``hash`` (salted per process); regime names map
    to small pinned integer codes in `_REGIME_SEED_CODE`.
    """
    return (int(seed), _REGIME_SEED_CODE[regime], int(n))


def _weighted_sample_without_replacement(
    weights: NDArray[np.float64], n: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    """Efraimidis-Spirakis weighted sampling without replacement (vectorized).

    Draws ``key_i = u_i ** (1 / w_i)`` for ``u_i ~ Uniform(0, 1)`` and returns the
    positions of the `n` largest keys. `weights` must be strictly positive.

    Args:
        weights: Shape ``(m,)`` strictly-positive weights.
        n: Number of items to draw (clamped to `m`).
        rng: Random generator (caller controls seeding).

    Returns:
        Shape ``(min(n, m),)`` int64 positions into `weights`, unordered.
    """
    m = weights.shape[0]
    n = min(n, m)
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    u = rng.random(m)
    keys = u ** (1.0 / weights)
    if n >= m:
        return np.arange(m, dtype=np.int64)
    top = np.argpartition(-keys, n - 1)[:n]
    return top.astype(np.int64)


def _lexsort_top_k(
    primary_desc: NDArray[np.float64],
    secondary_desc: NDArray[np.float64],
    u_idx: NDArray[np.int32],
    v_idx: NDArray[np.int32],
    k: int,
) -> NDArray[np.int64]:
    """Positions of the top-`k` rows by (primary desc, secondary desc, pair order asc).

    Fully vectorized deterministic ranking (no per-row Python loop): primary and
    secondary scores break ties in that order, and any remaining tie is broken by
    ascending canonical pair order (`u_idx` then `v_idx`, which matches
    lexicographic node-id order since `node_ids` is sorted ascending).
    """
    m = primary_desc.shape[0]
    k = min(k, m)
    order = np.lexsort((v_idx, u_idx, -secondary_desc, -primary_desc))
    return cast(NDArray[np.int64], order[:k].astype(np.int64))


# --------------------------------------------------------------------------- regime selectors


def select_easy_uniform(neg_idx: NDArray[np.int64], n: int, *, seed: int) -> NDArray[np.int64]:
    """Select `n` label-0 rows uniformly at random, without replacement.

    Args:
        neg_idx: Row positions (into the full universe arrays) of label-0 rows.
        n: Target sample size (clamped to ``len(neg_idx)``).
        seed: Base seed; combined with a pinned regime code and `n` for full
            reproducibility (`_regime_seed_sequence`).

    Returns:
        Sorted row positions, a subset of `neg_idx`.
    """
    rng = np.random.default_rng(_regime_seed_sequence(seed, "easy_uniform", n))
    n = min(n, neg_idx.size)
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    chosen = rng.choice(neg_idx, size=n, replace=False)
    return np.sort(chosen)


def select_degree_corrected(
    neg_idx: NDArray[np.int64],
    u_idx: NDArray[np.int32],
    v_idx: NDArray[np.int32],
    degree_array: NDArray[np.float64],
    n: int,
    *,
    seed: int,
) -> NDArray[np.int64]:
    """Select `n` label-0 rows with probability proportional to ``(deg_u+1)(deg_v+1)``.

    Degree-corrected negative sampling per 2405.14985. Implemented via the
    Efraimidis-Spirakis weighted-sampling-without-replacement algorithm
    (`_weighted_sample_without_replacement`), which is exact and fully
    vectorized (no per-pair Python loop even at candidate-universe scale).

    Args:
        neg_idx: Row positions of label-0 rows.
        u_idx: Full-universe ``u_idx`` array.
        v_idx: Full-universe ``v_idx`` array.
        degree_array: Node degree, indexed the same way as `u_idx`/`v_idx` (i.e.
            aligned with the universe's ``node_ids``).
        n: Target sample size (clamped to ``len(neg_idx)``).
        seed: Base seed (see `_regime_seed_sequence`).

    Returns:
        Sorted row positions, a subset of `neg_idx`.
    """
    if neg_idx.size == 0 or n <= 0:
        return np.empty(0, dtype=np.int64)
    weights = (degree_array[u_idx[neg_idx]] + 1.0) * (degree_array[v_idx[neg_idx]] + 1.0)
    rng = np.random.default_rng(_regime_seed_sequence(seed, "degree_corrected", n))
    chosen_positions = _weighted_sample_without_replacement(weights, n, rng)
    return np.sort(neg_idx[chosen_positions])


def common_neighbor_and_adamic_adar(
    g_simple: nx.Graph, node_order: Sequence[str]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Dense common-neighbor-count and Adamic-Adar score matrices over `node_order`.

    Computed via sparse adjacency matrix products (``A @ A`` for common-neighbor
    counts, ``A @ diag(w) @ A`` for Adamic-Adar with ``w_i = 1/log(deg_i)`` for
    ``deg_i > 1`` else 0) so it never iterates node pairs in Python. Intended for
    graphs the size of one benchmark split's test nodes (a few thousand), where a
    dense ``(n, n)`` output is cheap.

    Args:
        g_simple: A simple graph (self-loops already stripped).
        node_order: Node ids defining row/column order of the returned matrices.
            Graph nodes not in `node_order` are ignored.

    Returns:
        ``(common_neighbors, adamic_adar)``, each shape ``(len(node_order),
        len(node_order))`` float64, symmetric.
    """
    n = len(node_order)
    position = {node_id: i for i, node_id in enumerate(node_order)}
    rows: list[int] = []
    cols: list[int] = []
    for u, v in g_simple.edges():
        if u not in position or v not in position:
            continue
        i, j = position[u], position[v]
        rows.extend((i, j))
        cols.extend((j, i))
    data = np.ones(len(rows), dtype=np.float64)
    adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    common_neighbors = (adjacency @ adjacency).toarray()
    degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    weight = np.zeros(n, dtype=np.float64)
    mask = degrees > 1
    weight[mask] = 1.0 / np.log(degrees[mask])
    weighted = sparse.diags(weight)
    adamic_adar = (adjacency @ weighted @ adjacency).toarray()
    return common_neighbors, adamic_adar


def select_hard_heuristic(
    neg_idx: NDArray[np.int64],
    u_idx: NDArray[np.int32],
    v_idx: NDArray[np.int32],
    node_ids: Sequence[str],
    g_simple: nx.Graph,
    k: int,
) -> NDArray[np.int64]:
    """Select the top-`k` label-0 rows by HeaRT-style structural proximity.

    Ranked by common-neighbor count (desc), ties broken by Adamic-Adar (desc),
    remaining ties broken by canonical pair order (asc) -- fully deterministic.

    Args:
        neg_idx: Row positions of label-0 rows.
        u_idx: Full-universe ``u_idx`` array.
        v_idx: Full-universe ``v_idx`` array.
        node_ids: The universe's node_ids (defines the index space of `u_idx`/`v_idx`).
        g_simple: The evaluator-side reference graph, self-loops already stripped.
        k: Number of rows to select (clamped to ``len(neg_idx)``).

    Returns:
        Sorted row positions, a subset of `neg_idx`.
    """
    if neg_idx.size == 0 or k <= 0:
        return np.empty(0, dtype=np.int64)
    common_neighbors, adamic_adar = common_neighbor_and_adamic_adar(g_simple, node_ids)
    u_neg = u_idx[neg_idx]
    v_neg = v_idx[neg_idx]
    cn_scores = common_neighbors[u_neg, v_neg]
    aa_scores = adamic_adar[u_neg, v_neg]
    top = _lexsort_top_k(cn_scores, aa_scores, u_neg, v_neg, k)
    return np.sort(neg_idx[top])


def select_hard_feature(
    neg_idx: NDArray[np.int64],
    u_idx: NDArray[np.int32],
    v_idx: NDArray[np.int32],
    node_ids: Sequence[str],
    store: FeatureStore,
    k: int,
    *,
    cache_path: Path | None = None,
) -> NDArray[np.int64]:
    """Select the top-`k` label-0 rows by cosine similarity of mean-pooled F0 features.

    Ties broken by canonical pair order (asc) -- fully deterministic.

    Args:
        neg_idx: Row positions of label-0 rows.
        u_idx: Full-universe ``u_idx`` array.
        v_idx: Full-universe ``v_idx`` array.
        node_ids: The universe's node_ids (defines the index space of `u_idx`/`v_idx`
            and the row order of the F0 matrix).
        store: Feature store to build the F0 matrix from.
        k: Number of rows to select (clamped to ``len(neg_idx)``).
        cache_path: Optional F0 matrix cache path (`build_f0_matrix`).

    Returns:
        Sorted row positions, a subset of `neg_idx`.
    """
    if neg_idx.size == 0 or k <= 0:
        return np.empty(0, dtype=np.int64)
    matrix, _ = build_f0_matrix(store, node_ids, cache_path=cache_path)
    mat = matrix.numpy().astype(np.float64)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normed = mat / norms
    u_neg = u_idx[neg_idx]
    v_neg = v_idx[neg_idx]
    cos = np.einsum("ij,ij->i", normed[u_neg], normed[v_neg])
    top = _lexsort_top_k(cos, cos, u_neg, v_neg, k)
    return np.sort(neg_idx[top])


def select_negative_indices(
    regime: str,
    *,
    neg_idx: NDArray[np.int64],
    u_idx: NDArray[np.int32],
    v_idx: NDArray[np.int32],
    node_ids: Sequence[str],
    degree_array: NDArray[np.float64],
    g_simple: nx.Graph,
    store: FeatureStore | None,
    f0_cache: Path | None,
    n: int,
    seed: int,
) -> NDArray[np.int64]:
    """Dispatch to the regime-specific negative selector.

    Raises:
        ValueError: If `regime` is unknown, or `regime == "hard_feature"` and
            `store` is ``None``.
    """
    if regime == "easy_uniform":
        return select_easy_uniform(neg_idx, n, seed=seed)
    if regime == "degree_corrected":
        return select_degree_corrected(neg_idx, u_idx, v_idx, degree_array, n, seed=seed)
    if regime == "hard_heuristic":
        return select_hard_heuristic(neg_idx, u_idx, v_idx, node_ids, g_simple, n)
    if regime == "hard_feature":
        if store is None:
            raise ValueError("hard_feature regime requires a FeatureStore")
        return select_hard_feature(neg_idx, u_idx, v_idx, node_ids, store, n, cache_path=f0_cache)
    raise ValueError(f"unknown regime {regime!r}; expected one of {_REGIME_NAMES}")


# --------------------------------------------------------------------------- PA-null


def compute_pa_null_scores(
    g_ref: nx.Graph,
    node_ids: Sequence[str],
    u_idx: NDArray[np.int32],
    v_idx: NDArray[np.int32],
) -> NDArray[np.float64]:
    """Compute the preferential-attachment null score ``s_ij = k_i * k_j`` per row.

    Degrees come from `strip_self_loops(g_ref)` (evaluator-side / held-out-graph
    access -- legal only as a diagnostic, per the E5 zero-edge construction
    rule). No +1 smoothing: pairs where either endpoint has degree 0 score 0.

    Args:
        g_ref: The reference graph (self-loops stripped internally).
        node_ids: Node ids defining the index space of `u_idx`/`v_idx`.
        u_idx: Full-universe ``u_idx`` array.
        v_idx: Full-universe ``v_idx`` array.

    Returns:
        Shape ``(len(u_idx),)`` float64 raw (unnormalized) PA-null scores.
    """
    g_simple = strip_self_loops(g_ref)
    degree_array = _degree_array(g_simple, node_ids)
    return degree_array[u_idx] * degree_array[v_idx]


def normalize_min_max(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    """Min-max normalize `scores` into ``[0, 1]``.

    If `scores` is constant (or empty), returns an array of 0.5s (a fully
    uninformative but valid "probability").
    """
    if scores.size == 0:
        return scores.astype(np.float64)
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo <= 0.0:
        return np.full_like(scores, 0.5, dtype=np.float64)
    return (scores - lo) / (hi - lo)


def degree_heterogeneity_sigma(g_ref: nx.Graph) -> float:
    """Degree heterogeneity sigma: ``std(log(k))`` over nodes with ``k >= 1``.

    A log-normal-fit dispersion parameter (protocol Sec 2 PA-null row).
    Self-loops are stripped before computing degrees. Returns 0.0 if no node has
    positive degree.
    """
    g_simple = strip_self_loops(g_ref)
    degrees = np.array([float(d) for _, d in g_simple.degree()], dtype=np.float64)
    degrees = degrees[degrees >= 1.0]
    if degrees.size == 0:
        return 0.0
    return float(np.std(np.log(degrees)))


# --------------------------------------------------------------------------- regime table


def evaluate_regime_table(
    *,
    labels: NDArray[np.int8],
    probs: NDArray[np.float64],
    u_idx: NDArray[np.int32],
    v_idx: NDArray[np.int32],
    node_ids: Sequence[str],
    g_ref: nx.Graph,
    store: FeatureStore | None,
    f0_cache: Path | None,
    seed: int,
    threshold: float = _EDGE_METRIC_THRESHOLD,
) -> dict[str, dict[str, EdgeMetrics]]:
    """Build the full regime x ratio edge-metric table for one probability array.

    Positives are all label-1 rows; negatives for each `_REGIME_NAMES` entry are
    selected (per `select_negative_indices`) at each ratio in `_RATIOS`, plus one
    unsampled `full_universe` row over every labeled row (the imbalance view).

    Args:
        labels: Full-universe label array (``-1`` rows, if any, are excluded).
        probs: Full-universe probability array, aligned with `labels`.
        u_idx: Full-universe ``u_idx`` array.
        v_idx: Full-universe ``v_idx`` array.
        node_ids: Node ids defining the index space of `u_idx`/`v_idx`.
        g_ref: The reference graph (self-loops stripped internally where needed).
        store: Feature store for the `hard_feature` regime (``None`` skips it).
        f0_cache: F0 matrix cache path for `hard_feature`.
        seed: Base seed for regime sampling.
        threshold: Decision threshold forwarded to `compute_edge_metrics`.

    Returns:
        Regime name -> (``"ratio_{r}"`` -> `EdgeMetrics`), plus
        ``"full_universe"`` -> ``{"all": EdgeMetrics}``.
    """
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_idx = np.flatnonzero(pos_mask)
    neg_idx = np.flatnonzero(neg_mask)
    n_pos = int(pos_idx.size)

    g_simple = strip_self_loops(g_ref)
    degree_array = _degree_array(g_simple, node_ids)

    regime_names = (
        _REGIME_NAMES
        if store is not None
        else tuple(name for name in _REGIME_NAMES if name != "hard_feature")
    )

    result: dict[str, dict[str, EdgeMetrics]] = {}
    for regime in regime_names:
        result[regime] = {}
        for ratio in _RATIOS:
            target = ratio * n_pos
            sel_neg = select_negative_indices(
                regime,
                neg_idx=neg_idx,
                u_idx=u_idx,
                v_idx=v_idx,
                node_ids=node_ids,
                degree_array=degree_array,
                g_simple=g_simple,
                store=store,
                f0_cache=f0_cache,
                n=target,
                seed=seed,
            )
            idx = np.concatenate([pos_idx, sel_neg])
            row_labels = labels[idx].astype(np.int64)
            row_probs = probs[idx]
            result[regime][f"ratio_{ratio}"] = compute_edge_metrics(
                row_labels, row_probs, threshold=threshold
            )

    full_idx = np.flatnonzero(pos_mask | neg_mask)
    result["full_universe"] = {
        "all": compute_edge_metrics(
            labels[full_idx].astype(np.int64), probs[full_idx], threshold=threshold
        )
    }
    return result


# --------------------------------------------------------------------------- self-pair edge metrics


@dataclass(frozen=True)
class SelfPairEdgeMetrics:
    """Edge metrics computed over only the self-pair (``u_idx == v_idx``) rows.

    Self-pairs are legal, meaningful universe rows -- a self-pair is positive iff
    that node carries a self-loop in the reference test graph (spec Sec 9.4) --
    but they are excluded from every regime table and from the density-matched
    assembly quota. This is the dedicated slice reporting how well a scorer
    separates them.

    Attributes:
        n: Number of labeled (``label in {0, 1}``) self-pair rows evaluated.
        n_pos: Number of positive-labeled (self-loop) self-pair rows among `n`.
        metrics: `EdgeMetrics` (as a plain dict) at threshold 0.5, or ``None`` if
            the self-pair rows are single-class (degenerate -- AUROC/AUPRC/MCC
            are undefined).
        reason: ``None`` when `metrics` is present; otherwise why it could not be
            computed.
    """

    n: int
    n_pos: int
    metrics: dict[str, float | int] | None
    reason: str | None


def compute_self_pair_edge_metrics(
    labels: NDArray[np.int8],
    probs: NDArray[np.float64],
    u_idx: NDArray[np.int32],
    v_idx: NDArray[np.int32],
    *,
    threshold: float = _EDGE_METRIC_THRESHOLD,
) -> SelfPairEdgeMetrics:
    """Compute edge metrics restricted to self-pair (``u_idx == v_idx``) rows.

    Args:
        labels: Full-universe label array (``-1`` rows are excluded, matching
            `evaluate_regime_table`'s convention).
        probs: Full-universe probability array for one scorer, aligned with `labels`.
        u_idx: Full-universe ``u_idx`` array.
        v_idx: Full-universe ``v_idx`` array.
        threshold: Decision threshold forwarded to `compute_edge_metrics`.

    Returns:
        A `SelfPairEdgeMetrics`. `compute_edge_metrics` raises on single-class
        labels rather than emit NaN; that exception is caught here and converted
        into a disclosed null (`metrics=None`, `reason=str(exc)`) instead of
        propagating.
    """
    self_mask = (u_idx == v_idx) & ((labels == 0) | (labels == 1))
    self_labels = labels[self_mask].astype(np.int64)
    self_probs = probs[self_mask]
    n = int(self_mask.sum())
    n_pos = int(np.sum(self_labels == 1))
    try:
        metrics = compute_edge_metrics(self_labels, self_probs, threshold=threshold)
    except ValueError as exc:
        return SelfPairEdgeMetrics(n=n, n_pos=n_pos, metrics=None, reason=str(exc))
    metrics_dict = cast(dict[str, float | int], dataclasses.asdict(metrics))
    return SelfPairEdgeMetrics(n=n, n_pos=n_pos, metrics=metrics_dict, reason=None)


def _self_pair_edge_metrics_to_dict(value: SelfPairEdgeMetrics) -> dict[str, object]:
    return dataclasses.asdict(value)


# --------------------------------------------------------------------------- threshold sweep


@dataclass(frozen=True)
class SweepRow:
    """One row of the threshold sweep table, with self-loop bookkeeping added.

    Attributes:
        threshold: The assembly threshold used.
        recall: Fraction of the reference graph's edges recovered.
        graph_similarity: Official per-subgraph GS macro mean.
        relative_density: Official per-subgraph RD macro mean.
        mmd_ratio: Statistic -> reference-normalized MMD ratio at this threshold.
        self_loop_count: Number of self-pairs `(u, u)` with `p >= threshold`.
        self_loop_rate: `self_loop_count` divided by the total number of
            self-pairs in the universe.
    """

    threshold: float
    recall: float
    graph_similarity: float
    relative_density: float
    mmd_ratio: dict[str, float]
    self_loop_count: int
    self_loop_rate: float


def build_threshold_grid(probs: NDArray[np.float64], operating_point: float) -> list[float]:
    """Build the sweep grid: the operating point plus pinned percentiles, deduplicated.

    Args:
        probs: Full-universe probability array.
        operating_point: The density-matched operating-point threshold.

    Returns:
        Sorted, deduplicated threshold values.
    """
    percentile_values = [float(np.percentile(probs, q)) for q in _PERCENTILES]
    return sorted({float(operating_point), *percentile_values})


def compute_self_loop_stats(
    u_idx: NDArray[np.int32], v_idx: NDArray[np.int32], probs: NDArray[np.float64], threshold: float
) -> tuple[int, float]:
    """Count self-pairs `(u, u)` with `p >= threshold` and their rate over all self-pairs."""
    self_mask = u_idx == v_idx
    total_self = int(self_mask.sum())
    hits = int(np.sum(self_mask & (probs >= threshold)))
    rate = hits / total_self if total_self > 0 else 0.0
    return hits, rate


def run_threshold_sweep(
    *,
    universe: ScoresArtifact,
    probs: NDArray[np.float64],
    g_ref: nx.Graph,
    buckets: dict[int, list[set[str]]],
    config: MMDConfig,
) -> tuple[float, list[SweepRow]]:
    """Run the density-matched threshold sweep for one scores artifact.

    Args:
        universe: The scores artifact `probs` was derived from (supplies pairs
            and the self-loop bookkeeping arrays).
        probs: Full-universe probability array.
        g_ref: The reference graph (self-loops kept; stripped internally).
        buckets: Bucket size -> list of node sets.
        config: Shared MMD/descriptor configuration.

    Returns:
        ``(operating_point_threshold, sweep_rows)``, `sweep_rows` sorted by
        ascending threshold.

    Convention:
        `operating_point` is density-matched on non-self-pair rows
        (``u_idx != v_idx``) against `target_edges` (the simple, self-loop-
        stripped reference edge count) -- self-pairs never consume quota meant
        for real edges. Self-pairs still assemble as self-loops at that same
        threshold (never dropped) and are reported via `compute_self_loop_stats`
        in each `SweepRow`.
    """
    g_ref_simple = strip_self_loops(g_ref)
    target_edges = g_ref_simple.number_of_edges()
    non_self_mask = universe.u_idx != universe.v_idx
    operating_point = density_matched_threshold(probs[non_self_mask], target_edges)
    grid = build_threshold_grid(probs, operating_point)

    pairs = list(universe.pairs())
    points = threshold_sweep(
        pairs, probs, thresholds=grid, g_ref=g_ref, buckets=buckets, config=config
    )
    rows: list[SweepRow] = []
    for t, point in zip(grid, points, strict=True):
        hits, rate = compute_self_loop_stats(universe.u_idx, universe.v_idx, probs, t)
        rows.append(
            SweepRow(
                threshold=t,
                recall=point.recall,
                graph_similarity=point.graph_similarity,
                relative_density=point.relative_density,
                mmd_ratio=dict(point.mmd_ratio),
                self_loop_count=hits,
                self_loop_rate=rate,
            )
        )
    return operating_point, rows


# --------------------------------------------------------------------------- assembled rows


@dataclass(frozen=True)
class AssembledRow:
    """One assembled-graph evaluation row at a fixed operating point.

    Attributes:
        threshold: Assembly threshold used (``None`` for the PA-null top-N row,
            which selects an exact edge count instead of thresholding).
        mmd_ratio: Statistic -> reference-normalized MMD ratio.
        raw_mmd2: Statistic -> raw MMD^2 numerator.
        reference_mmd2: Statistic -> reference-split MMD^2 denominator.
        graph_similarity: Official per-subgraph GS macro mean.
        relative_density: Official per-subgraph RD macro mean.
        per_size_graph_similarity: Per-size lists of per-subgraph GS values.
        per_size_relative_density: Per-size lists of per-subgraph RD values.
        self_loops_pred: Self-loop count in the assembled graph.
        self_loops_ref: Self-loop count in the reference graph.
        bootstrap_mean: Statistic -> mean normalized MMD ratio over bootstrap resamples
            after aggregating raw/reference MMD2 across bucket sizes.
        bootstrap_std: Statistic -> std of normalized MMD ratio over bootstrap
            resamples after aggregating raw/reference MMD2 across bucket sizes.
    """

    threshold: float | None
    mmd_ratio: dict[str, float]
    raw_mmd2: dict[str, float]
    reference_mmd2: dict[str, float]
    graph_similarity: float
    relative_density: float
    per_size_graph_similarity: dict[int, list[float]]
    per_size_relative_density: dict[int, list[float]]
    self_loops_pred: int
    self_loops_ref: int
    bootstrap_mean: dict[str, float]
    bootstrap_std: dict[str, float]


def assemble_and_evaluate(
    *,
    g_pred: nx.Graph,
    g_ref: nx.Graph,
    buckets: dict[int, list[set[str]]],
    config: MMDConfig,
    seed: int,
    threshold: float | None,
) -> AssembledRow:
    """Evaluate one already-assembled predicted graph into an `AssembledRow`.

    Args:
        g_pred: The assembled predicted graph.
        g_ref: The reference graph.
        buckets: Bucket size -> list of node sets.
        config: Shared MMD/descriptor configuration.
        seed: Seed for `bootstrap_mmd`.
        threshold: The assembly threshold used to build `g_pred` (for bookkeeping
            only; ``None`` for non-threshold assemblies like the PA-null top-N row).

    Returns:
        The `AssembledRow`.
    """
    report = evaluate_assembled_graph(g_pred, g_ref, buckets, config)
    boot = bootstrap_mmd(g_pred, g_ref, buckets, config, seed=seed)
    bootstrap_mean = {stat: boot[stat][0] for stat in STATISTICS}
    bootstrap_std = {stat: boot[stat][1] for stat in STATISTICS}
    return AssembledRow(
        threshold=threshold,
        mmd_ratio=dict(report.mmd_ratio),
        raw_mmd2=dict(report.raw_mmd2),
        reference_mmd2=dict(report.reference_mmd2),
        graph_similarity=report.graph_similarity,
        relative_density=report.relative_density,
        per_size_graph_similarity=dict(report.per_size_graph_similarity),
        per_size_relative_density=dict(report.per_size_relative_density),
        self_loops_pred=report.self_loops_pred,
        self_loops_ref=report.self_loops_ref,
        bootstrap_mean=bootstrap_mean,
        bootstrap_std=bootstrap_std,
    )


def assemble_top_n_by_score(
    pairs: Sequence[tuple[str, str]],
    u_idx: NDArray[np.int32],
    v_idx: NDArray[np.int32],
    scores: NDArray[np.float64],
    n: int,
    nodes: Sequence[str],
) -> nx.Graph:
    """Assemble a graph from the top-`n` pairs by `scores` (deterministic tie-break).

    Ties broken by ascending canonical pair order, matching the regime
    selectors' tie-break convention.

    Args:
        pairs: Node pairs, in the same row order as `scores`.
        u_idx: Row-aligned node-index array (for tie-breaking).
        v_idx: Row-aligned node-index array (for tie-breaking).
        scores: Row-aligned scores; higher is more likely to be an edge.
        n: Number of top pairs to realize as edges (clamped to ``len(pairs)``).
        nodes: Full node universe for the returned graph.

    Returns:
        The assembled `networkx.Graph`.
    """
    n = min(n, len(pairs))
    order = np.lexsort((v_idx, u_idx, -scores))
    top = order[:n]
    g = nx.Graph()
    g.add_nodes_from(nodes)
    for i in top.tolist():
        u, v = pairs[i]
        g.add_edge(u, v)
    return g


# --------------------------------------------------------------------------- results assembly


def _edge_metrics_table_to_dict(
    table: dict[str, dict[str, EdgeMetrics]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    return {
        regime: {key: dataclasses.asdict(metrics) for key, metrics in ratios.items()}
        for regime, ratios in table.items()
    }


def _assembled_row_to_dict(row: AssembledRow) -> dict[str, object]:
    return dataclasses.asdict(row)


def _sweep_rows_to_list(rows: list[SweepRow]) -> list[dict[str, object]]:
    return [dataclasses.asdict(row) for row in rows]


@dataclass(frozen=True)
class G1Result:
    """The full, JSON-ready G1 hardened-E2 result payload.

    Attributes:
        metadata: Every disclosure required by the gate (artifact provenance,
            benchmark strategy, MMD and official GS/RD config, normalization and construction
            rules, seeds, and the two orchestrator-adjudication notes).
        noise_floor: Bucket size -> statistic -> mean noise-floor MMD^2.
        threshold_sweep: Operating point plus the sweep grid rows.
        regime_table: Scorer key (``"b0"``, optionally ``"b0_alt"``, ``"pa_null"``)
            -> regime edge-metric table.
        assembled: Scorer key -> `AssembledRow`, JSON-encoded.
        self_pair_edge_metrics: Scorer key -> `SelfPairEdgeMetrics`, JSON-encoded
            (spec Sec 9.4 self/non-self split; ``"b0_alt"`` is ``None`` when no
            B0-alt artifact was supplied).
        degree_heterogeneity_sigma: `std(log(k))` over `k >= 1` on `g_ref`.
        positive_rate: Measured positive rate over the full candidate universe.
    """

    metadata: dict[str, object]
    noise_floor: dict[int, dict[str, float]]
    threshold_sweep: dict[str, object]
    regime_table: dict[str, object]
    assembled: dict[str, object]
    self_pair_edge_metrics: dict[str, object]
    degree_heterogeneity_sigma: float
    positive_rate: float

    def to_jsonable(self) -> dict[str, object]:
        """Return a plain-dict, `json.dumps`-ready representation."""
        return {
            "metadata": self.metadata,
            "noise_floor": {str(size): dict(stats) for size, stats in self.noise_floor.items()},
            "threshold_sweep": self.threshold_sweep,
            "regime_table": self.regime_table,
            "assembled": self.assembled,
            "self_pair_edge_metrics": self.self_pair_edge_metrics,
            "degree_heterogeneity_sigma": self.degree_heterogeneity_sigma,
            "positive_rate": self.positive_rate,
        }


# --------------------------------------------------------------------------- markdown tables


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6g}"


def render_tables_markdown(payload: dict[str, object]) -> str:
    """Render the GitHub-markdown tables for the G1 gate report from a results payload.

    Args:
        payload: The `G1Result.to_jsonable()` output (or an equivalent dict).

    Returns:
        Markdown text (regime table, sweep table, assembled rows, noise-floor row).
    """
    lines: list[str] = ["# G1 hardened-E2 tables", ""]

    lines.append("## Regime table")
    lines.append("")
    lines.append("| scorer | regime | key | n_pos | n_neg | AUROC | AUPRC | MCC |")
    lines.append("|---|---|---|---|---|---|---|---|")
    regime_table = cast(dict[str, object], payload["regime_table"])
    for scorer, table in regime_table.items():
        if table is None:
            continue
        table_dict = cast(dict[str, dict[str, dict[str, float | int]]], table)
        for regime, ratios in table_dict.items():
            for key, metrics in ratios.items():
                auroc = _fmt(cast(float, metrics["auroc"]))
                auprc = _fmt(cast(float, metrics["auprc"]))
                mcc = _fmt(cast(float, metrics["mcc"]))
                lines.append(
                    f"| {scorer} | {regime} | {key} | {metrics['n_pos']} | {metrics['n_neg']} | "
                    f"{auroc} | {auprc} | {mcc} |"
                )
    lines.append("")

    lines.append("## Threshold sweep")
    lines.append("")
    lines.append(
        "| threshold | recall | graph similarity | rel. density | degree MMD ratio | "
        "clustering MMD ratio | spectral MMD ratio | "
        "self-loop count | self-loop rate |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    sweep = cast(dict[str, object], payload["threshold_sweep"])
    for row in cast(list[dict[str, object]], sweep["rows"]):
        mmd_ratio = cast(dict[str, float], row["mmd_ratio"])
        lines.append(
            f"| {_fmt(cast(float, row['threshold']))} | {_fmt(cast(float, row['recall']))} | "
            f"{_fmt(cast(float, row['graph_similarity']))} | "
            f"{_fmt(cast(float, row['relative_density']))} | {_fmt(mmd_ratio.get('degree'))} | "
            f"{_fmt(mmd_ratio.get('clustering'))} | {_fmt(mmd_ratio.get('spectral'))} | "
            f"{row['self_loop_count']} | {_fmt(cast(float, row['self_loop_rate']))} |"
        )
    lines.append("")

    lines.append("## Assembled-graph rows")
    lines.append("")
    lines.append(
        "| scorer | threshold | graph similarity | rel. density | degree MMD ratio | "
        "clustering MMD ratio | spectral MMD ratio | "
        "self-loops (pred/ref) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    assembled = cast(dict[str, object], payload["assembled"])
    for scorer, assembled_row in assembled.items():
        if assembled_row is None:
            continue
        row_dict = cast(dict[str, object], assembled_row)
        mmd_ratio = cast(dict[str, float], row_dict["mmd_ratio"])
        lines.append(
            f"| {scorer} | {_fmt(cast(float | None, row_dict['threshold']))} | "
            f"{_fmt(cast(float, row_dict['graph_similarity']))} | "
            f"{_fmt(cast(float, row_dict['relative_density']))} | "
            f"{_fmt(mmd_ratio.get('degree'))} | {_fmt(mmd_ratio.get('clustering'))} | "
            f"{_fmt(mmd_ratio.get('spectral'))} | "
            f"{row_dict['self_loops_pred']}/{row_dict['self_loops_ref']} |"
        )
    lines.append("")

    lines.append("## MMD ratio components")
    lines.append("")
    lines.append(
        "| scorer | statistic | raw numerator | reference denominator | normalized ratio |"
    )
    lines.append("|---|---|---|---|---|")
    for scorer, assembled_row in assembled.items():
        if assembled_row is None:
            continue
        row_dict = cast(dict[str, object], assembled_row)
        raw = cast(dict[str, float], row_dict["raw_mmd2"])
        reference = cast(dict[str, float], row_dict["reference_mmd2"])
        ratio = cast(dict[str, float], row_dict["mmd_ratio"])
        for stat in STATISTICS:
            lines.append(
                f"| {scorer} | {stat} | {_fmt(raw[stat])} | "
                f"{_fmt(reference[stat])} | {_fmt(ratio[stat])} |"
            )
    lines.append("")

    lines.append("## Noise floor")
    lines.append("")
    lines.append(
        "| bucket size | degree reference MMD2 | clustering reference MMD2 | "
        "spectral reference MMD2 |"
    )
    lines.append("|---|---|---|---|")
    for size, stats in cast(dict[str, dict[str, float]], payload["noise_floor"]).items():
        lines.append(
            f"| {size} | {_fmt(stats.get('degree'))} | {_fmt(stats.get('clustering'))} | "
            f"{_fmt(stats.get('spectral'))} |"
        )
    lines.append("")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- orchestration


def _artifact_meta_summary(meta: dict[str, object]) -> dict[str, object]:
    return {
        "checkpoint_id": meta.get("checkpoint_id"),
        "model_family": meta.get("model_family"),
        "pairs_source": meta.get("pairs_source"),
        "strategy": meta.get("strategy"),
        "num_rows": meta.get("num_rows"),
    }


def run_g1_pipeline(
    *,
    universe_path: Path,
    alt_universe_path: Path | None,
    data_root: Path,
    strategy: str,
    output_dir: Path,
    seed: int = 0,
    skip_perturbation_check: bool = False,
) -> dict[str, object]:
    """Run the full G1 hardened-E2 pipeline and write its outputs.

    Args:
        universe_path: Path to the primary (B0) candidate-universe scores artifact.
        alt_universe_path: Optional path to the B0-alt candidate-universe scores
            artifact; when ``None``, B0-alt rows are absent from the output.
        data_root: Directory containing ``benchmark_2025_neurips/``.
        strategy: Benchmark split strategy (e.g. ``"breadth_first"``).
        output_dir: Directory to write ``g1_results.json`` and ``g1_tables.md`` into.
        seed: Base seed for every randomized step of the pipeline.
        skip_perturbation_check: Debug-only speed flag. When True, the O'Bray
            perturbation diagnostic is skipped; official GS/RD remain defined.

    Returns:
        The JSON-ready results payload (also written to
        ``output_dir/g1_results.json``).

    Raises:
        ValueError: If either scores artifact fails validation.
    """
    universe = load_scores(universe_path)
    alt: ScoresArtifact | None = load_scores(alt_universe_path) if alt_universe_path else None

    benchmark_root = data_root / _BENCHMARK_SUBDIR
    g_ref = load_test_graph(benchmark_root, strategy)
    buckets = load_test_node_buckets(benchmark_root, strategy)
    n_test_nodes = g_ref.number_of_nodes()

    validate_universe_artifact(
        universe, strategy=strategy, n_test_nodes=n_test_nodes, label="universe"
    )
    if alt is not None:
        validate_universe_artifact(
            alt, strategy=strategy, n_test_nodes=n_test_nodes, label="alt-universe"
        )
        validate_alt_artifact(alt, universe)

    output_dir.mkdir(parents=True, exist_ok=True)
    config = MMDConfig()

    logger.info("computing real-vs-real noise floor (seed=%d)", seed)
    nf = noise_floor(g_ref, buckets, config)
    perturbation_meta: dict[str, object]
    if skip_perturbation_check:
        perturbation_meta = {"skipped": True, "passed": None, "failures": []}
    else:
        logger.info("running the O'Bray perturbation check (seed=%d)", seed)
        perturbation_result = perturbation_check(g_ref, buckets, config, seed=seed)
        perturbation_meta = {
            "skipped": False,
            "passed": perturbation_result.passed,
            "failures": perturbation_result.failures,
            "similarities": perturbation_result.similarities,
            "fractions": list(perturbation_result.fractions),
        }

    store: FeatureStore | None = None
    features_root = data_root / _FEATURES_SUBDIR
    if (features_root / "index.json").exists():
        store = FeatureStore(features_root)
    f0_cache = output_dir / "f0_cache.pt"

    probs = universe.probs()
    op_threshold, sweep_rows = run_threshold_sweep(
        universe=universe, probs=probs, g_ref=g_ref, buckets=buckets, config=config
    )

    regime_table: dict[str, object] = {
        "b0": evaluate_regime_table(
            labels=universe.label,
            probs=probs,
            u_idx=universe.u_idx,
            v_idx=universe.v_idx,
            node_ids=universe.node_ids,
            g_ref=g_ref,
            store=store,
            f0_cache=f0_cache,
            seed=seed,
        )
    }

    alt_probs: NDArray[np.float64] | None = None
    if alt is not None:
        alt_probs = alt.probs()
        regime_table["b0_alt"] = evaluate_regime_table(
            labels=universe.label,
            probs=alt_probs,
            u_idx=universe.u_idx,
            v_idx=universe.v_idx,
            node_ids=universe.node_ids,
            g_ref=g_ref,
            store=store,
            f0_cache=f0_cache,
            seed=seed,
        )
    else:
        regime_table["b0_alt"] = None

    pa_null_raw = compute_pa_null_scores(g_ref, universe.node_ids, universe.u_idx, universe.v_idx)
    pa_null_probs = normalize_min_max(pa_null_raw)
    regime_table["pa_null"] = evaluate_regime_table(
        labels=universe.label,
        probs=pa_null_probs,
        u_idx=universe.u_idx,
        v_idx=universe.v_idx,
        node_ids=universe.node_ids,
        g_ref=g_ref,
        store=store,
        f0_cache=f0_cache,
        seed=seed,
    )
    sigma = degree_heterogeneity_sigma(g_ref)

    self_pair_metrics: dict[str, object] = {
        "b0": _self_pair_edge_metrics_to_dict(
            compute_self_pair_edge_metrics(universe.label, probs, universe.u_idx, universe.v_idx)
        ),
        "b0_alt": (
            _self_pair_edge_metrics_to_dict(
                compute_self_pair_edge_metrics(
                    universe.label, alt_probs, universe.u_idx, universe.v_idx
                )
            )
            if alt is not None and alt_probs is not None
            else None
        ),
        "pa_null": _self_pair_edge_metrics_to_dict(
            compute_self_pair_edge_metrics(
                universe.label, pa_null_probs, universe.u_idx, universe.v_idx
            )
        ),
    }

    nodes = list(g_ref.nodes())
    pairs = list(universe.pairs())
    target_edges = strip_self_loops(g_ref).number_of_edges()
    # Self-pairs (u_idx == v_idx) never consume the density-matched quota: the
    # quota is matched against target_edges, the SIMPLE (self-loop-stripped)
    # reference edge count, so only non-self rows may count toward it. Self-pairs
    # still assemble as self-loops at whatever threshold is chosen (see
    # `assemble_graph`, called below with the full pair/prob arrays).
    non_self_mask = universe.u_idx != universe.v_idx

    b0_threshold = op_threshold
    b0_graph = assemble_graph(pairs, probs, threshold=b0_threshold, nodes=nodes)
    assembled: dict[str, object] = {
        "b0": _assembled_row_to_dict(
            assemble_and_evaluate(
                g_pred=b0_graph,
                g_ref=g_ref,
                buckets=buckets,
                config=config,
                seed=seed,
                threshold=b0_threshold,
            )
        )
    }
    if alt is not None and alt_probs is not None:
        alt_threshold = density_matched_threshold(alt_probs[non_self_mask], target_edges)
        alt_graph = assemble_graph(pairs, alt_probs, threshold=alt_threshold, nodes=nodes)
        assembled["b0_alt"] = _assembled_row_to_dict(
            assemble_and_evaluate(
                g_pred=alt_graph,
                g_ref=g_ref,
                buckets=buckets,
                config=config,
                seed=seed,
                threshold=alt_threshold,
            )
        )
    else:
        assembled["b0_alt"] = None

    # PA-null's top-N candidate pool excludes self-pairs entirely: a self-pair's
    # score is k_i * k_i (that node's maximum possible PA-null score), so leaving
    # self-pairs in the pool would systematically pollute the top-N ranking.
    non_self_row_idx = np.flatnonzero(non_self_mask)
    pa_null_graph = assemble_top_n_by_score(
        [pairs[i] for i in non_self_row_idx.tolist()],
        universe.u_idx[non_self_row_idx],
        universe.v_idx[non_self_row_idx],
        pa_null_raw[non_self_row_idx],
        target_edges,
        nodes,
    )
    assembled["pa_null"] = _assembled_row_to_dict(
        assemble_and_evaluate(
            g_pred=pa_null_graph,
            g_ref=g_ref,
            buckets=buckets,
            config=config,
            seed=seed,
            threshold=None,
        )
    )

    n_pos = int(np.sum(universe.label == 1))
    n_neg = int(np.sum(universe.label == 0))
    positive_rate = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0.0

    metadata: dict[str, object] = {
        "artifacts": {
            "b0": _artifact_meta_summary(universe.meta),
            "b0_alt": _artifact_meta_summary(alt.meta) if alt is not None else None,
        },
        "benchmark": {"benchmark_a": strategy},
        "mmd_config": dataclasses.asdict(config),
        "metric_normalization": "ratio_of_size_mean_mmd2",
        "reference_split": "artifact_order_even_vs_odd_within_each_node_size",
        "canonical_metric": "mmd_ratio",
        "component_disclosure": ["raw_mmd2", "reference_mmd2", "mmd_ratio"],
        "graph_similarity": {
            "formula": "1 - L1(A_pred - A_ref) / (sum(A_pred) + sum(A_ref))",
            "aggregation": "unweighted macro mean over every fixed sampled subgraph",
            "self_loops": "retained",
            "empty_empty": 1.0,
        },
        "relative_density": {
            "formula": "density(G_pred) / density(G_ref)",
            "aggregation": "unweighted macro mean over every fixed sampled subgraph",
            "self_loops": "retained",
            "empty_empty": 1.0,
            "nonempty_over_empty": "inf",
        },
        "perturbation_check": perturbation_meta,
        "threshold_policy": (
            "operating point = density_matched_threshold(own probs, target_edges = "
            "|E(strip_self_loops(test_graph))|), computed independently per scorer "
            "(B0, B0-alt) and density-matched on non-self-pair rows against the "
            "simple reference edge count; self-pairs assemble at the same threshold "
            "and are reported in the self-loop row (SweepRow.self_loop_count / "
            "self_loop_rate), never counted toward the quota. PA-null instead takes "
            "the exact top-target_edges pairs by s_ij among non-self pairs only, "
            "with a deterministic tie-break (self-pairs are excluded from PA-null's "
            "top-N candidate pool entirely, since s_ij = k_i * k_i is a self-pair's "
            "own maximum possible score and would otherwise dominate the ranking; "
            "no threshold is used for PA-null). Sweep grid = operating point plus "
            f"the {_PERCENTILES} percentiles of the scorer's own probability "
            "distribution over all rows (self-pairs included), deduplicated and sorted."
        ),
        "seed": seed,
        "regime_construction": {
            "easy_uniform": "uniform random sample of label-0 rows, without replacement.",
            "degree_corrected": (
                "label-0 rows sampled without replacement with probability proportional to "
                "(deg_u + 1) * (deg_v + 1) (2405.14985), via the Efraimidis-Spirakis "
                "weighted-sampling-without-replacement algorithm; degrees from "
                "strip_self_loops(test_graph)."
            ),
            "hard_heuristic": (
                "top-K label-0 rows by common-neighbor count on strip_self_loops(test_graph), "
                "ties broken by Adamic-Adar, remaining ties by canonical pair order; "
                "K = ratio * n_positives."
            ),
            "hard_feature": (
                "top-K label-0 rows by cosine similarity of F0 mean-pooled features, ties "
                "broken by canonical pair order; K = ratio * n_positives."
            ),
            "full_universe": "all labeled rows, no sampling (the imbalance view).",
            "ratios": list(_RATIOS),
        },
        "pa_null": {
            "access": "evaluator_side_diagnostic",
            "formula": (
                "s_ij = k_i * k_j (no +1 smoothing); degrees from strip_self_loops(test_graph)."
            ),
            "normalization": "min-max over the full candidate universe.",
        },
        "notes": {
            "breadth_first_zero_drop": _BREADTH_FIRST_ZERO_DROP_NOTE,
            "negative_sampler_coin_flip": _NEGATIVE_SAMPLER_COIN_FLIP_NOTE,
        },
    }

    result = G1Result(
        metadata=metadata,
        noise_floor=nf,
        threshold_sweep={
            "operating_point": op_threshold,
            "target_edges": target_edges,
            "rows": _sweep_rows_to_list(sweep_rows),
        },
        regime_table={
            "b0": _edge_metrics_table_to_dict(
                cast(dict[str, dict[str, EdgeMetrics]], regime_table["b0"])
            ),
            "b0_alt": (
                _edge_metrics_table_to_dict(
                    cast(dict[str, dict[str, EdgeMetrics]], regime_table["b0_alt"])
                )
                if regime_table["b0_alt"] is not None
                else None
            ),
            "pa_null": _edge_metrics_table_to_dict(
                cast(dict[str, dict[str, EdgeMetrics]], regime_table["pa_null"])
            ),
        },
        assembled=assembled,
        self_pair_edge_metrics=self_pair_metrics,
        degree_heterogeneity_sigma=sigma,
        positive_rate=positive_rate,
    )
    payload = result.to_jsonable()

    (output_dir / "g1_results.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "g1_tables.md").write_text(render_tables_markdown(payload), encoding="utf-8")
    logger.info("wrote G1 results and tables to %s", output_dir)
    return payload


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    """Build the ``g1_hardened_e2`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.experiments.g1_hardened_e2",
        description="Gate G1 (hardened E2) analysis pipeline from cached scores artifacts.",
    )
    parser.add_argument("--universe", type=Path, required=True, help="B0 candidate-universe .npz")
    parser.add_argument(
        "--alt-universe", type=Path, default=None, help="B0-alt candidate-universe .npz (optional)"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--strategy", default="breadth_first")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-perturbation-check",
        action="store_true",
        help=(
            "DEBUG-ONLY speed flag: skip the perturbation diagnostic; "
            "official GS/RD remain defined."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Raises:
        SystemExit: On argument errors (via ``parser.error``, missing input files).
        ValueError: If a scores artifact fails validation (`validate_universe_artifact`,
            `validate_alt_artifact`) -- surfaced as an uncaught exception with a
            descriptive message, matching `src.experiments.g2_ceiling.main`.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.universe.exists():
        parser.error(f"--universe not found: {args.universe}")
    if args.alt_universe is not None and not args.alt_universe.exists():
        parser.error(f"--alt-universe not found: {args.alt_universe}")

    run_g1_pipeline(
        universe_path=args.universe,
        alt_universe_path=args.alt_universe,
        data_root=args.data_root,
        strategy=args.strategy,
        output_dir=args.output_dir,
        seed=args.seed,
        skip_perturbation_check=args.skip_perturbation_check,
    )


__all__ = [
    "AssembledRow",
    "G1Result",
    "SweepRow",
    "assemble_and_evaluate",
    "assemble_top_n_by_score",
    "build_parser",
    "build_threshold_grid",
    "SelfPairEdgeMetrics",
    "common_neighbor_and_adamic_adar",
    "compute_pa_null_scores",
    "compute_self_loop_stats",
    "compute_self_pair_edge_metrics",
    "degree_heterogeneity_sigma",
    "evaluate_regime_table",
    "load_test_graph",
    "load_test_node_buckets",
    "main",
    "normalize_min_max",
    "render_tables_markdown",
    "run_g1_pipeline",
    "run_threshold_sweep",
    "select_degree_corrected",
    "select_easy_uniform",
    "select_hard_feature",
    "select_hard_heuristic",
    "select_negative_indices",
    "validate_alt_artifact",
    "validate_universe_artifact",
]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()

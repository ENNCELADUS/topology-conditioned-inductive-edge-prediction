"""Assembled-graph topological realism metrics: descriptors, MMD, bucketed evaluation.

Structural computations always operate on **simple** graphs. Callers of the
descriptor functions must strip self-loops first (per the repository's spec §9.4);
the bucketed helpers in this module strip self-loops internally and report the
counts separately so callers never have to.

No dependence on `src.data` or `torch` — this module operates purely on
`networkx.Graph`, numpy arrays, and plain dicts/sets so it is testable standalone.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

STATISTICS = ("degree", "clustering", "spectral")


def degree_histogram(g: nx.Graph, *, max_degree: int) -> np.ndarray:
    """Compute the L1-normalized degree histogram of a simple graph.

    Args:
        g: A simple graph (caller must have stripped self-loops).
        max_degree: Degrees above this value are clipped into the last bin.

    Returns:
        A length-`(max_degree + 1)` array over bins `0..max_degree`, L1-normalized.
        For a graph with zero nodes, returns an all-zero vector (normalization is
        skipped since there is no mass to distribute).
    """
    degrees = [d for _, d in g.degree()]
    n = len(degrees)
    hist = np.zeros(max_degree + 1, dtype=float)
    if n == 0:
        return hist
    clipped = np.clip(np.asarray(degrees, dtype=int), 0, max_degree)
    counts = np.bincount(clipped, minlength=max_degree + 1)[: max_degree + 1]
    hist = counts.astype(float)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist


def clustering_histogram(g: nx.Graph, *, bins: int = 100) -> np.ndarray:
    """Compute the L1-normalized histogram of local clustering coefficients.

    Args:
        g: A simple graph (caller must have stripped self-loops).
        bins: Number of equal-width bins over the support `[0, 1]`.

    Returns:
        A length-`bins` array, L1-normalized. For a graph with zero nodes, returns
        an all-zero vector.
    """
    coeffs = list(nx.clustering(g).values())
    if len(coeffs) == 0:
        return np.zeros(bins, dtype=float)
    counts, _ = np.histogram(coeffs, bins=bins, range=(0.0, 1.0))
    hist = counts.astype(float)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist


def laplacian_spectrum_histogram(g: nx.Graph, *, bins: int = 200) -> np.ndarray:
    """Compute the L1-normalized histogram of normalized Laplacian eigenvalues.

    Args:
        g: A simple graph (caller must have stripped self-loops).
        bins: Number of equal-width bins over the support `[0, 2]`.

    Returns:
        A length-`bins` array, L1-normalized. A graph with zero nodes, or a graph
        with nodes but no edges, has its entire mass placed in bin 0 (the
        normalized Laplacian is either undefined or identically zero in these
        cases). Eigenvalues are clipped to `[0, 2]` before binning to guard
        against floating-point noise pushing values just outside the
        theoretical support.
    """
    n = g.number_of_nodes()
    hist = np.zeros(bins, dtype=float)
    if n == 0:
        hist[0] = 1.0
        return hist
    if g.number_of_edges() == 0:
        hist[0] = 1.0
        return hist
    eigs = np.asarray(list(nx.normalized_laplacian_spectrum(g)), dtype=float)
    eigs = np.clip(eigs, 0.0, 2.0)
    counts, _ = np.histogram(eigs, bins=bins, range=(0.0, 2.0))
    hist = counts.astype(float)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist


@dataclass(frozen=True)
class MMDConfig:
    """Disclosed kernel/binning parameters for MMD-based graph-realism metrics.

    Attributes:
        kernel: Kernel identifier (only `"gaussian_rbf"` is implemented).
        bandwidth_scales: Multipliers applied to the median-heuristic bandwidth;
            the scale-`1.0` value is the canonical one.
        degree_max: `max_degree` passed to `degree_histogram`.
        clustering_bins: `bins` passed to `clustering_histogram`.
        spectral_bins: `bins` passed to `laplacian_spectrum_histogram`.
    """

    kernel: str = "gaussian_rbf"
    bandwidth_scales: tuple[float, ...] = (0.5, 1.0, 2.0)
    degree_max: int = 64
    clustering_bins: int = 100
    spectral_bins: int = 200


@dataclass(frozen=True)
class MMDResult:
    """Result of a biased MMD^2 estimate at one or more bandwidth scales.

    Attributes:
        canonical: The biased MMD^2 estimate at bandwidth scale 1.0.
        by_scale: Biased MMD^2 estimate for each scale in `MMDConfig.bandwidth_scales`.
        median_bandwidth: The (possibly guarded) median-heuristic bandwidth used as
            the base for all scales. If the true pooled-pairwise-distance median is
            0, this is set to 1.0 (see `mmd_squared`); a warning is logged in that case.
    """

    canonical: float
    by_scale: dict[float, float]
    median_bandwidth: float


def _median_heuristic_bandwidth(pooled: np.ndarray) -> float:
    """Median of pooled pairwise L2 distances, guarded against a zero median."""
    if pooled.shape[0] < 2:
        return 1.0
    dists = cdist(pooled, pooled, metric="euclidean")
    iu = np.triu_indices(pooled.shape[0], k=1)
    pairwise = dists[iu]
    median = float(np.median(pairwise)) if pairwise.size > 0 else 0.0
    if median == 0.0:
        logger.warning(
            "median heuristic pairwise-distance is 0 (degenerate/identical inputs); "
            "guarding median_bandwidth to 1.0"
        )
        return 1.0
    return median


def _rbf_kernel_sum(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    """Sum of the Gaussian RBF kernel over all pairs between `x` and `y`."""
    sq_dists = cdist(x, y, metric="sqeuclidean")
    return float(np.exp(-sq_dists / (2.0 * sigma**2)).sum())


def _mmd2_biased(a: np.ndarray, b: np.ndarray, sigma: float) -> float:
    """Biased (V-statistic) MMD^2 estimate between samples `a` and `b`."""
    m, n = int(a.shape[0]), int(b.shape[0])
    kxx = _rbf_kernel_sum(a, a, sigma)
    kyy = _rbf_kernel_sum(b, b, sigma)
    kxy = _rbf_kernel_sum(a, b, sigma)
    return float(kxx / (m * m) + kyy / (n * n) - 2.0 * kxy / (m * n))


def mmd_squared(a: list[np.ndarray], b: list[np.ndarray], config: MMDConfig) -> MMDResult:
    """Compute biased MMD^2 between two sets of descriptor histograms.

    Each element of `a`/`b` is one graph's descriptor histogram (all must share
    the same dimensionality). The kernel is a Gaussian RBF on the L2 distance
    between histograms, with bandwidth set via the median heuristic (median of
    pooled pairwise L2 distances over `a ∪ b`), scaled by each of
    `config.bandwidth_scales`.

    Args:
        a: Non-empty list of 1-D descriptor arrays (first sample set).
        b: Non-empty list of 1-D descriptor arrays (second sample set).
        config: Kernel/bandwidth configuration.

    Returns:
        An `MMDResult` with the biased MMD^2 at scale 1.0 (`canonical`), at every
        configured scale (`by_scale`), and the (possibly guarded) median bandwidth.

    Raises:
        ValueError: If `a` or `b` is empty.
    """
    if len(a) == 0 or len(b) == 0:
        raise ValueError("mmd_squared requires non-empty sample sets for both a and b")
    mat_a = np.stack([np.asarray(v, dtype=float) for v in a])
    mat_b = np.stack([np.asarray(v, dtype=float) for v in b])
    pooled = np.vstack([mat_a, mat_b])
    median_bandwidth = _median_heuristic_bandwidth(pooled)

    by_scale = {
        scale: _mmd2_biased(mat_a, mat_b, median_bandwidth * scale)
        for scale in config.bandwidth_scales
    }
    canonical = _mmd2_biased(mat_a, mat_b, median_bandwidth * 1.0)
    return MMDResult(canonical=canonical, by_scale=by_scale, median_bandwidth=median_bandwidth)


@dataclass(frozen=True)
class BucketedMMDReport:
    """Per-bucket-size MMD report plus whole-graph density and self-loop counts.

    Attributes:
        per_size: Bucket size -> statistic -> `MMDResult` (over the 50-per-size
            predicted vs. reference subgraph descriptors).
        aggregate: Statistic -> mean canonical MMD^2 over all bucket sizes.
        relative_density: `|E_pred_simple| / |E_ref_simple|` over the whole graphs
            (self-loops stripped).
        self_loops_pred: Number of self-loops in the whole predicted graph.
        self_loops_ref: Number of self-loops in the whole reference graph.
    """

    per_size: dict[int, dict[str, MMDResult]]
    aggregate: dict[str, float]
    relative_density: float
    self_loops_pred: int
    self_loops_ref: int


def strip_self_loops(g: nx.Graph) -> nx.Graph:
    """Return a copy of `g` with self-loops removed."""
    g2 = g.copy()
    g2.remove_edges_from(list(nx.selfloop_edges(g2)))
    return g2


def _simple_subgraph(g: nx.Graph, nodes: Iterable[str]) -> nx.Graph:
    """Return the induced subgraph on `nodes`, as a simple graph (self-loops stripped).

    The subgraph is materialized with **sorted** node insertion order. Bucket node
    sets are Python sets, whose iteration order changes with per-process hash
    randomization; networkx subgraph views iterate the filter node set when it is
    much smaller than the parent graph, which would permute the Laplacian nodelist
    and shift eigenvalues by ~1e-15 — enough to flip a histogram bin when an
    eigenvalue sits exactly on a bin edge (e.g. 1.0). Canonical sorted order makes
    every downstream metric bit-reproducible across processes.
    """
    view = g.subgraph(nodes)
    out = nx.Graph()
    out.add_nodes_from(sorted(view.nodes()))
    out.add_edges_from(view.edges())
    out.remove_edges_from(list(nx.selfloop_edges(out)))
    return out


def _descriptors(g_simple: nx.Graph, config: MMDConfig) -> dict[str, np.ndarray]:
    """Compute the three descriptor histograms for an already-simple graph."""
    return {
        "degree": degree_histogram(g_simple, max_degree=config.degree_max),
        "clustering": clustering_histogram(g_simple, bins=config.clustering_bins),
        "spectral": laplacian_spectrum_histogram(g_simple, bins=config.spectral_bins),
    }


def evaluate_assembled_graph(
    g_pred: nx.Graph,
    g_ref: nx.Graph,
    buckets: dict[int, list[set[str]]],
    config: MMDConfig,
) -> BucketedMMDReport:
    """Evaluate an assembled predicted graph against a reference graph, bucketed by size.

    For every bucket size, the descriptors of each predicted subgraph
    (`g_pred.subgraph(nodes)`, self-loops stripped) are compared against the
    corresponding reference subgraph via `mmd_squared`, one `MMDResult` per
    statistic in `STATISTICS`.

    Args:
        g_pred: The assembled predicted graph.
        g_ref: The held-out reference graph.
        buckets: Bucket size -> list of node sets to evaluate at that size.
        config: Shared MMD/descriptor configuration.

    Returns:
        A `BucketedMMDReport`. If the reference graph has zero (self-loop-stripped)
        edges, `relative_density` is `0.0` when the predicted graph also has zero
        edges, else `inf` (an explicit guard rather than a `ZeroDivisionError`;
        not expected to trigger against real benchmark reference graphs).
    """
    self_loops_pred = nx.number_of_selfloops(g_pred)
    self_loops_ref = nx.number_of_selfloops(g_ref)

    pred_simple_full = strip_self_loops(g_pred)
    ref_simple_full = strip_self_loops(g_ref)
    ref_edge_count = ref_simple_full.number_of_edges()
    pred_edge_count = pred_simple_full.number_of_edges()
    if ref_edge_count > 0:
        relative_density = pred_edge_count / ref_edge_count
    else:
        relative_density = 0.0 if pred_edge_count == 0 else float("inf")

    per_size: dict[int, dict[str, MMDResult]] = {}
    for size, node_sets in buckets.items():
        pred_descs: dict[str, list[np.ndarray]] = {stat: [] for stat in STATISTICS}
        ref_descs: dict[str, list[np.ndarray]] = {stat: [] for stat in STATISTICS}
        for nodes in node_sets:
            pred_d = _descriptors(_simple_subgraph(g_pred, nodes), config)
            ref_d = _descriptors(_simple_subgraph(g_ref, nodes), config)
            for stat in STATISTICS:
                pred_descs[stat].append(pred_d[stat])
                ref_descs[stat].append(ref_d[stat])
        per_size[size] = {
            stat: mmd_squared(pred_descs[stat], ref_descs[stat], config) for stat in STATISTICS
        }

    aggregate = {
        stat: float(np.mean([per_size[size][stat].canonical for size in per_size]))
        for stat in STATISTICS
    }

    return BucketedMMDReport(
        per_size=per_size,
        aggregate=aggregate,
        relative_density=relative_density,
        self_loops_pred=self_loops_pred,
        self_loops_ref=self_loops_ref,
    )


def noise_floor(
    g_ref: nx.Graph,
    buckets: dict[int, list[set[str]]],
    config: MMDConfig,
    *,
    seed: int,
    n_splits: int = 10,
) -> dict[int, dict[str, float]]:
    """Estimate the real-vs-real MMD^2 noise floor by split-halving reference subgraphs.

    For each bucket size, the descriptors of the reference subgraphs are randomly
    split into two halves `n_splits` times; the mean canonical MMD^2 between the
    halves is the noise floor for that size/statistic.

    Args:
        g_ref: The reference graph.
        buckets: Bucket size -> list of node sets.
        config: Shared MMD/descriptor configuration.
        seed: Seed for the split RNG.
        n_splits: Number of random half/half splits to average over.

    Returns:
        Bucket size -> statistic -> mean canonical MMD^2 across splits.
    """
    rng = np.random.default_rng(seed)
    result: dict[int, dict[str, float]] = {}
    for size, node_sets in buckets.items():
        descs: dict[str, list[np.ndarray]] = {stat: [] for stat in STATISTICS}
        for nodes in node_sets:
            d = _descriptors(_simple_subgraph(g_ref, nodes), config)
            for stat in STATISTICS:
                descs[stat].append(d[stat])
        n = len(node_sets)
        half = n // 2
        split_values: dict[str, list[float]] = {stat: [] for stat in STATISTICS}
        for _ in range(n_splits):
            perm = rng.permutation(n)
            idx_a = perm[:half]
            idx_b = perm[half : 2 * half]
            for stat in STATISTICS:
                arr = descs[stat]
                a_list = [arr[i] for i in idx_a]
                b_list = [arr[i] for i in idx_b]
                split_values[stat].append(mmd_squared(a_list, b_list, config).canonical)
        result[size] = {stat: float(np.mean(split_values[stat])) for stat in STATISTICS}
    return result


def bootstrap_mmd(
    g_pred: nx.Graph,
    g_ref: nx.Graph,
    buckets: dict[int, list[set[str]]],
    config: MMDConfig,
    *,
    seed: int,
    n_boot: int = 200,
) -> dict[int, dict[str, tuple[float, float]]]:
    """Bootstrap (mean, std) of the canonical MMD^2 by resampling bucket node sets.

    For each bucket size, the (paired) predicted/reference descriptor pairs are
    resampled with replacement `n_boot` times; the mean and std of the resulting
    canonical MMD^2 values are reported.

    Args:
        g_pred: The assembled predicted graph.
        g_ref: The reference graph.
        buckets: Bucket size -> list of node sets.
        config: Shared MMD/descriptor configuration.
        seed: Seed for the resampling RNG.
        n_boot: Number of bootstrap resamples.

    Returns:
        Bucket size -> statistic -> (mean, std) over the bootstrap resamples.
    """
    rng = np.random.default_rng(seed)
    result: dict[int, dict[str, tuple[float, float]]] = {}
    for size, node_sets in buckets.items():
        n = len(node_sets)
        pred_descs: dict[str, list[np.ndarray]] = {stat: [] for stat in STATISTICS}
        ref_descs: dict[str, list[np.ndarray]] = {stat: [] for stat in STATISTICS}
        for nodes in node_sets:
            pred_d = _descriptors(_simple_subgraph(g_pred, nodes), config)
            ref_d = _descriptors(_simple_subgraph(g_ref, nodes), config)
            for stat in STATISTICS:
                pred_descs[stat].append(pred_d[stat])
                ref_descs[stat].append(ref_d[stat])
        boot_values: dict[str, list[float]] = {stat: [] for stat in STATISTICS}
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            for stat in STATISTICS:
                a_list = [pred_descs[stat][i] for i in idx]
                b_list = [ref_descs[stat][i] for i in idx]
                boot_values[stat].append(mmd_squared(a_list, b_list, config).canonical)
        result[size] = {
            stat: (float(np.mean(boot_values[stat])), float(np.std(boot_values[stat])))
            for stat in STATISTICS
        }
    return result

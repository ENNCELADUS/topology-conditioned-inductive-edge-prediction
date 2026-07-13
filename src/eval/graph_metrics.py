"""Canonical assembled-graph topology descriptors and normalized MMD ratios."""

from collections.abc import Iterable
from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.linalg import eigvalsh

STATISTICS = ("degree", "clustering", "spectral")


def degree_histogram(g: nx.Graph) -> np.ndarray:
    """Return the complete, unnormalized NetworkX degree histogram."""
    return np.asarray(nx.degree_histogram(g), dtype=float)


def clustering_histogram(g: nx.Graph) -> np.ndarray:
    """Return the official 100-bin local-clustering histogram on [0, 1]."""
    coeffs = list(nx.clustering(g).values())
    counts, _ = np.histogram(coeffs, bins=100, range=(0.0, 1.0), density=False)
    return counts.astype(float)


def laplacian_spectrum_histogram(g: nx.Graph) -> np.ndarray:
    """Return the official 200-bin normalized-Laplacian spectral PMF."""
    try:
        eigs = eigvalsh(nx.normalized_laplacian_matrix(g).todense())
    except Exception:
        eigs = np.zeros(g.number_of_nodes())
    counts, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2.0), density=False)
    hist = counts.astype(float)
    return hist / max(1.0, float(hist.sum()))


@dataclass(frozen=True)
class MMDConfig:
    """Fixed parameters for the canonical normalized MMD evaluation."""

    sigma: float = 1.0
    reference_epsilon: float = 1e-12


def _pad_histograms(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    support_size = max(len(x), len(y))
    x_values = np.pad(np.asarray(x, dtype=float), (0, support_size - len(x)))
    y_values = np.pad(np.asarray(y, dtype=float), (0, support_size - len(y)))
    return x_values, y_values


def _gaussian_tv(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    x_values, y_values = _pad_histograms(x, y)
    distance = float(np.abs(x_values - y_values).sum() / 2.0)
    return float(np.exp(-(distance * distance) / (2.0 * sigma * sigma)))


def _mean_kernel(
    samples1: list[np.ndarray],
    samples2: list[np.ndarray],
    *,
    sigma: float,
) -> float:
    total = sum(_gaussian_tv(x, y, sigma) for x in samples1 for y in samples2)
    return float(total / (len(samples1) * len(samples2)))


def mmd_squared(
    samples1: list[np.ndarray],
    samples2: list[np.ndarray],
    config: MMDConfig,
) -> float:
    """Return the biased Gaussian-TV MMD2 used by the canonical evaluator."""
    if not samples1 or not samples2:
        raise ValueError("mmd_squared requires two non-empty sample sets")
    normalized1 = [sample / (float(np.sum(sample)) + 1e-6) for sample in samples1]
    normalized2 = [sample / (float(np.sum(sample)) + 1e-6) for sample in samples2]
    return float(
        _mean_kernel(normalized1, normalized1, sigma=config.sigma)
        + _mean_kernel(normalized2, normalized2, sigma=config.sigma)
        - 2.0 * _mean_kernel(normalized1, normalized2, sigma=config.sigma)
    )


@dataclass(frozen=True)
class BucketedMMDReport:
    """Raw, reference, and normalized topology metrics across size buckets."""

    per_size_raw_mmd2: dict[int, dict[str, float]]
    per_size_reference_mmd2: dict[int, dict[str, float]]
    raw_mmd2: dict[str, float]
    reference_mmd2: dict[str, float]
    mmd_ratio: dict[str, float]
    relative_density: float
    self_loops_pred: int
    self_loops_ref: int


def strip_self_loops(g: nx.Graph) -> nx.Graph:
    """Return a copy of ``g`` with self-loops removed."""
    g2 = g.copy()
    g2.remove_edges_from(list(nx.selfloop_edges(g2)))
    return g2


def _induced_subgraph(g: nx.Graph, nodes: Iterable[str]) -> nx.Graph:
    """Return a deterministically ordered induced subgraph, preserving self-loops."""
    view = g.subgraph(nodes)
    out = nx.Graph()
    out.add_nodes_from(sorted(view.nodes()))
    out.add_edges_from(view.edges())
    return out


def _descriptors(g_subgraph: nx.Graph) -> dict[str, np.ndarray]:
    return {
        "degree": degree_histogram(g_subgraph),
        "clustering": clustering_histogram(g_subgraph),
        "spectral": laplacian_spectrum_histogram(g_subgraph),
    }


def evaluate_assembled_graph(
    g_pred: nx.Graph,
    g_ref: nx.Graph,
    buckets: dict[int, list[set[str]]],
    config: MMDConfig,
) -> BucketedMMDReport:
    """Evaluate predicted subgraphs and normalize MMD2 by reference variability."""
    self_loops_pred = nx.number_of_selfloops(g_pred)
    self_loops_ref = nx.number_of_selfloops(g_ref)
    pred_simple_full = strip_self_loops(g_pred)
    ref_simple_full = strip_self_loops(g_ref)
    ref_edge_count = ref_simple_full.number_of_edges()
    pred_edge_count = pred_simple_full.number_of_edges()
    relative_density = (
        pred_edge_count / ref_edge_count
        if ref_edge_count > 0
        else (0.0 if pred_edge_count == 0 else float("inf"))
    )

    per_size_raw: dict[int, dict[str, float]] = {}
    per_size_reference: dict[int, dict[str, float]] = {}
    for size, node_sets in buckets.items():
        if len(node_sets) < 2:
            raise ValueError(f"bucket {size} requires at least two reference samples")
        pred_descs: dict[str, list[np.ndarray]] = {stat: [] for stat in STATISTICS}
        ref_descs: dict[str, list[np.ndarray]] = {stat: [] for stat in STATISTICS}
        for nodes in node_sets:
            pred_d = _descriptors(_induced_subgraph(g_pred, nodes))
            ref_d = _descriptors(_induced_subgraph(g_ref, nodes))
            for stat in STATISTICS:
                pred_descs[stat].append(pred_d[stat])
                ref_descs[stat].append(ref_d[stat])
        per_size_raw[size] = {
            stat: mmd_squared(pred_descs[stat], ref_descs[stat], config) for stat in STATISTICS
        }
        per_size_reference[size] = {
            stat: mmd_squared(ref_descs[stat][::2], ref_descs[stat][1::2], config)
            for stat in STATISTICS
        }

    raw_mmd2 = {
        stat: float(np.mean([per_size_raw[size][stat] for size in per_size_raw]))
        for stat in STATISTICS
    }
    reference_mmd2 = {
        stat: float(np.mean([per_size_reference[size][stat] for size in per_size_reference]))
        for stat in STATISTICS
    }
    mmd_ratio = {
        stat: raw_mmd2[stat] / max(reference_mmd2[stat], config.reference_epsilon)
        for stat in STATISTICS
    }
    return BucketedMMDReport(
        per_size_raw_mmd2=per_size_raw,
        per_size_reference_mmd2=per_size_reference,
        raw_mmd2=raw_mmd2,
        reference_mmd2=reference_mmd2,
        mmd_ratio=mmd_ratio,
        relative_density=relative_density,
        self_loops_pred=self_loops_pred,
        self_loops_ref=self_loops_ref,
    )


def noise_floor(
    g_ref: nx.Graph,
    buckets: dict[int, list[set[str]]],
    config: MMDConfig,
) -> dict[int, dict[str, float]]:
    """Return the evaluator's deterministic odd/even reference denominator."""
    result: dict[int, dict[str, float]] = {}
    for size, node_sets in buckets.items():
        if len(node_sets) < 2:
            raise ValueError(f"bucket {size} requires at least two reference samples")
        descs: dict[str, list[np.ndarray]] = {stat: [] for stat in STATISTICS}
        for nodes in node_sets:
            values = _descriptors(_induced_subgraph(g_ref, nodes))
            for stat in STATISTICS:
                descs[stat].append(values[stat])
        result[size] = {
            stat: mmd_squared(descs[stat][::2], descs[stat][1::2], config) for stat in STATISTICS
        }
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
    """Bootstrap per-bucket normalized MMD ratios as ``(mean, std)`` pairs."""
    rng = np.random.default_rng(seed)
    result: dict[int, dict[str, tuple[float, float]]] = {}
    for size, node_sets in buckets.items():
        if len(node_sets) < 2:
            raise ValueError(f"bucket {size} requires at least two reference samples")
        pred = [_descriptors(_induced_subgraph(g_pred, nodes)) for nodes in node_sets]
        ref = [_descriptors(_induced_subgraph(g_ref, nodes)) for nodes in node_sets]
        values: dict[str, list[float]] = {stat: [] for stat in STATISTICS}
        for _ in range(n_boot):
            indices = rng.integers(0, len(node_sets), size=len(node_sets))
            for stat in STATISTICS:
                pred_samples = [pred[i][stat] for i in indices]
                ref_samples = [ref[i][stat] for i in indices]
                raw = mmd_squared(pred_samples, ref_samples, config)
                denominator = mmd_squared(ref_samples[::2], ref_samples[1::2], config)
                values[stat].append(raw / max(denominator, config.reference_epsilon))
        result[size] = {
            stat: (float(np.mean(values[stat])), float(np.std(values[stat]))) for stat in STATISTICS
        }
    return result

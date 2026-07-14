"""Perturbation sanity check for official graph similarity.

No dependence on `src.data` or `torch`.
"""

import logging
import math
from dataclasses import dataclass

import networkx as nx
import numpy as np

from src.eval.graph_metrics import MMDConfig, evaluate_assembled_graph

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerturbationCheckResult:
    """Result of validating that official GS decreases under graph perturbation.

    Attributes:
        similarities: Perturbation mode -> similarity at each fraction (mean over
            `n_trials`), in the same order as `fractions`.
        fractions: The perturbation fractions evaluated.
        passed: Whether every mode satisfied both pass conditions (see
            `perturbation_check`).
        failures: Human-readable description of every violated condition, empty iff
            `passed`.
    """

    similarities: dict[str, list[float]]
    fractions: tuple[float, ...]
    passed: bool
    failures: list[str]


def _degree_preserving_swap(g: nx.Graph, fraction: float, rng: np.random.Generator) -> nx.Graph:
    """Perturb `g` by degree-preserving double-edge swaps (seeded, capped tries)."""
    g2 = g.copy()
    n_edges = g2.number_of_edges()
    nswap = int(round(fraction * n_edges))
    if nswap <= 0:
        return g2
    seed_val = int(rng.integers(0, 2**31 - 1))
    max_tries = max(nswap * 20, 200)
    try:
        nx.double_edge_swap(g2, nswap=nswap, max_tries=max_tries, seed=seed_val)
    except nx.NetworkXError as exc:
        logger.warning("degree_preserving_swap could not complete all swaps: %s", exc)
    return g2


def _uniform_rewire(g: nx.Graph, fraction: float, rng: np.random.Generator) -> nx.Graph:
    """Perturb `g` by removing k random edges and adding k random non-edges."""
    g2 = g.copy()
    n_edges = g2.number_of_edges()
    k = math.ceil(fraction * n_edges)
    if k <= 0:
        return g2
    edges = list(g2.edges())
    k = min(k, len(edges))
    remove_idx = rng.choice(len(edges), size=k, replace=False)
    to_remove = [edges[i] for i in remove_idx]
    g2.remove_edges_from(to_remove)

    nodes = list(g2.nodes())
    existing = {frozenset(e) for e in g2.edges()}
    added = 0
    attempts = 0
    max_attempts = max(k * 50, 200)
    while added < k and attempts < max_attempts and len(nodes) >= 2:
        attempts += 1
        u, v = rng.choice(nodes, size=2, replace=False)
        fe = frozenset((u, v))
        if fe in existing:
            continue
        g2.add_edge(u, v)
        existing.add(fe)
        added += 1
    if added < k:
        logger.warning(
            "uniform_rewire only added %d of %d requested non-edges within attempt budget",
            added,
            k,
        )
    return g2


_PERTURBATION_MODES = {
    "degree_preserving_swap": _degree_preserving_swap,
    "uniform_rewire": _uniform_rewire,
}


def perturbation_check(
    g_ref: nx.Graph,
    buckets: dict[int, list[set[str]]],
    config: MMDConfig,
    *,
    fractions: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5),
    n_trials: int = 3,
    seed: int,
    modes: tuple[str, ...] = ("degree_preserving_swap", "uniform_rewire"),
) -> PerturbationCheckResult:
    """Validate that official GS decreases monotonically under perturbation.

    For each mode and fraction, `n_trials` independent perturbed copies of `g_ref`
    are compared back against `g_ref`; the macro-averaged per-subgraph GS is recorded.
    `g_ref` itself is never mutated.

    PASS conditions (checked independently per mode; every violation is recorded):
        - The mean similarity sequence over `fractions` is non-increasing within a
          tolerance of `1e-3` (small increases from sampling noise are tolerated).
        - `similarity(max(fractions)) < 0.8 * similarity(0.0)`.

    Args:
        g_ref: Reference graph to perturb (never mutated).
        buckets: Bucket size -> list of node sets, passed through to
            `evaluate_assembled_graph`.
        config: Shared MMD/descriptor configuration.
        fractions: Perturbation fractions to evaluate, in increasing order.
        n_trials: Number of independent perturbation trials averaged per fraction.
        seed: Seed for the perturbation RNG.
        modes: Perturbation modes to validate.

    Returns:
        A `PerturbationCheckResult`.

    Raises:
        ValueError: If an unknown perturbation mode is requested.
    """
    for mode in modes:
        if mode not in _PERTURBATION_MODES:
            raise ValueError(f"unknown perturbation mode: {mode!r}")

    rng = np.random.default_rng(seed)
    similarities: dict[str, list[float]] = {}
    failures: list[str] = []

    for mode in modes:
        perturb_fn = _PERTURBATION_MODES[mode]
        sims_per_fraction: list[float] = []
        for frac in fractions:
            trial_sims = []
            for _ in range(n_trials):
                g_pert = perturb_fn(g_ref, frac, rng)
                report = evaluate_assembled_graph(g_pert, g_ref, buckets, config)
                trial_sims.append(report.graph_similarity)
            sims_per_fraction.append(float(np.mean(trial_sims)))
        similarities[mode] = sims_per_fraction

        for i in range(1, len(sims_per_fraction)):
            if sims_per_fraction[i] > sims_per_fraction[i - 1] + 1e-3:
                failures.append(
                    f"{mode}: similarity increased from fraction {fractions[i - 1]} "
                    f"({sims_per_fraction[i - 1]:.6f}) to {fractions[i]} "
                    f"({sims_per_fraction[i]:.6f})"
                )
        if sims_per_fraction[-1] >= 0.8 * sims_per_fraction[0]:
            failures.append(
                f"{mode}: similarity at max fraction ({sims_per_fraction[-1]:.6f}) not < "
                f"0.8x baseline ({sims_per_fraction[0]:.6f})"
            )

    passed = len(failures) == 0
    return PerturbationCheckResult(
        similarities=similarities, fractions=fractions, passed=passed, failures=failures
    )

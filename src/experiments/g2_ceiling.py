r"""Gate G2 — edge-independence ceiling (Chanpuriya et al. 2111.00048).

Computes, from the cached candidate-universe soft scores of a frozen B0-family
checkpoint, how many triangles an edge-independent, calibrated assembly can
possibly express at the scorer's measured overlap, versus what the real test
graph demands. All identities are exact (no Thm 6 big-O), computed in float64
over the dense :math:`n \times n` probability matrix of the test-node universe
(``n = 2{,}018`` for ``breadth_first`` -> 32 MB, seconds of compute, no GPU).

Identities (unordered-pair convention, self-pairs excluded per the repository's
simple-graph policy, spec section 9.4)::

    V(P)  = sum_{i<j} p_ij                     (expected volume)
    Ov(P) = sum_{i<j} p_ij^2 / sum_{i<j} p_ij   (measured overlap)
    E[Delta] = tr(P^3) / 6                      (expected triangle count)

Ceiling curve at matched volume::

    T_max(omega) = (sqrt(2) / 3) * (omega * V) ** 1.5

``Ov_min`` is the overlap at which ``T_max(Ov_min) == delta_star`` (the real
test graph's triangle count); ``Ov_min = (3 * delta_star / sqrt(2)) ** (2/3) / V``.

CLI::

    python -m src.experiments.g2_ceiling \
        --universe scores/b0_v31_candidate.npz \
        --data-root data --strategy breadth_first \
        --output-dir outputs/g2 [--figure figures/g2-ceiling.html]

The pure pieces below stay importable for tests; :func:`main` wires the real
scores artifact and benchmark test graph into them.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from src.eval.graph_metrics import strip_self_loops
from src.score_universe import ScoresArtifact, load_scores

logger = logging.getLogger(__name__)

_BENCHMARK_SUBDIR = Path("benchmark_2025_neurips")
_OMEGA_GRID_POINTS = 513

#: Hard-thresholded assemblies realize Ov = 1, where the Chanpuriya bound is vacuous;
#: the ceiling curve constrains stochastic, calibrated assemblies at their measured
#: overlap instead.
CAVEAT_VACUOUS_AT_OV_1 = (
    "hard-thresholded assemblies have Ov = 1, where the bound is vacuous; the curve "
    "constrains stochastic, calibrated assemblies at their measured overlap"
)

#: We use Chanpuriya et al. 2111.00048's exact identities directly, not their
#: (looser) Theorem 6 big-O bound.
CAVEAT_EXACT_IDENTITIES = "exact identities; Thm 6 big-O deliberately not used"


@dataclass(frozen=True)
class G2Result:
    """Gate G2 edge-independence ceiling results for one candidate-universe artifact.

    Attributes:
        n_nodes: Number of test nodes in the scored candidate universe (`|node_ids|`).
        n_pairs_used: Number of unordered non-self pairs used in the identities
            (`C(n_nodes, 2)` on a complete candidate universe).
        n_self_pairs: Number of `(u, u)` self-pair rows excluded from every identity.
        self_pair_prob_mass: Sum of `sigmoid(logit)` over the excluded self-pair rows.
        volume: `V(P) = sum_{i<j} p_ij`, the expected edge volume.
        overlap: `Ov(P) = sum_{i<j} p_ij^2 / V(P)`, the scorer's measured overlap.
        expected_triangles: `E[Delta] = tr(P^3) / 6`, the expected triangle count.
        delta_star: Triangle count of the simple (self-loop-stripped) real test graph.
        simple_graph_edges: Edge count of the simple real test graph.
        ov_min: Minimum overlap at matched volume `V` for which the ceiling curve
            reaches `delta_star` (`T_max(ov_min) == delta_star`).
        headroom_triangles: `expected_triangles / delta_star`.
        headroom_overlap: `overlap / ov_min`.
        omega: Sorted, deduplicated overlap grid the ceiling curve is evaluated on
            (`linspace(0, 1, 513)` plus the exact `overlap` and `ov_min` points).
        t_max_curve: `T_max(omega)` at matched volume `V`, aligned with `omega`.
        checkpoint_id: Scores artifact's `checkpoint_id` (from `meta`).
        model_family: Scores artifact's `model_family` (from `meta`).
        strategy: Split strategy name the artifact and reference graph were checked
            against.
    """

    n_nodes: int
    n_pairs_used: int
    n_self_pairs: int
    self_pair_prob_mass: float
    volume: float
    overlap: float
    expected_triangles: float
    delta_star: int
    simple_graph_edges: int
    ov_min: float
    headroom_triangles: float
    headroom_overlap: float
    omega: NDArray[np.float64]
    t_max_curve: NDArray[np.float64]
    checkpoint_id: str
    model_family: str
    strategy: str


def _expected_candidate_rows(n_test_nodes: int) -> int:
    """Return `C(n_test_nodes, 2) + n_test_nodes`, the candidate-universe row count."""
    return math.comb(n_test_nodes, 2) + n_test_nodes


def validate_universe(artifact: ScoresArtifact, *, strategy: str, n_test_nodes: int) -> None:
    """Validate a scores artifact is the full labeled candidate universe for `strategy`.

    Args:
        artifact: The loaded scores artifact.
        strategy: Expected split strategy name.
        n_test_nodes: Number of test nodes the candidate universe is defined over
            (used to derive the expected row count `C(n, 2) + n`).

    Raises:
        ValueError: If `meta["pairs_source"] != "candidate"`, `meta["strategy"]`
            does not match `strategy`, the row count does not equal the expected
            candidate-universe size, or any label is outside `{0, 1}`.
    """
    errors: list[str] = []

    pairs_source = artifact.meta.get("pairs_source")
    if pairs_source != "candidate":
        errors.append(f"pairs_source: expected 'candidate', got {pairs_source!r}")

    meta_strategy = artifact.meta.get("strategy")
    if meta_strategy != strategy:
        errors.append(f"strategy: expected {strategy!r}, got {meta_strategy!r}")

    expected_rows = _expected_candidate_rows(n_test_nodes)
    n_rows = len(artifact.logit)
    if n_rows != expected_rows:
        errors.append(
            f"row count: expected {expected_rows} (C({n_test_nodes},2)+{n_test_nodes}), "
            f"got {n_rows}"
        )

    labels_present = set(np.unique(artifact.label).tolist())
    bad_labels = labels_present - {0, 1}
    if bad_labels:
        errors.append(f"label: values outside {{0,1}} found: {sorted(bad_labels)}")

    if errors:
        raise ValueError("; ".join(errors))


def build_dense_probs(
    artifact: ScoresArtifact,
) -> tuple[NDArray[np.float64], int, float, int]:
    """Build the dense symmetric probability matrix over the artifact's node universe.

    Self-pair rows (`u_idx == v_idx`) are excluded from `P` (zero diagonal) per the
    repository's simple-graph convention (spec section 9.4); they are counted and
    their probability mass summed separately instead of being folded in as zeros.

    Args:
        artifact: The loaded scores artifact (candidate universe).

    Returns:
        `(p_matrix, n_self_pairs, self_pair_prob_mass, n_pairs_used)`:
        `p_matrix` is `(n, n)` float64, symmetric, zero diagonal, `n = |node_ids|`;
        `n_self_pairs` and `self_pair_prob_mass` describe the excluded self-pair
        rows; `n_pairs_used` is the number of non-self rows folded into `p_matrix`.
    """
    n = len(artifact.node_ids)
    probs = artifact.probs()
    self_mask = artifact.u_idx == artifact.v_idx
    n_self_pairs = int(np.count_nonzero(self_mask))
    self_pair_prob_mass = float(probs[self_mask].sum())

    keep = ~self_mask
    n_pairs_used = int(np.count_nonzero(keep))
    rows = artifact.u_idx[keep].astype(np.int64)
    cols = artifact.v_idx[keep].astype(np.int64)
    vals = probs[keep]

    p_matrix = np.zeros((n, n), dtype=np.float64)
    p_matrix[rows, cols] = vals
    p_matrix[cols, rows] = vals
    np.fill_diagonal(p_matrix, 0.0)
    return p_matrix, n_self_pairs, self_pair_prob_mass, n_pairs_used


def compute_identities(p_matrix: NDArray[np.float64]) -> tuple[float, float, float]:
    """Compute the exact Chanpuriya identities `(V, Ov, E[Delta])` from a dense `P`.

    Args:
        p_matrix: `(n, n)` float64 symmetric probability matrix, zero diagonal.

    Returns:
        `(volume, overlap, expected_triangles)`. `overlap` is `0.0` when `volume`
        is `0.0` (degenerate all-zero-probability guard).
    """
    volume = 0.5 * float(np.sum(p_matrix))
    sum_sq = 0.5 * float(np.sum(p_matrix * p_matrix))
    overlap = sum_sq / volume if volume > 0 else 0.0
    p_cubed = p_matrix @ p_matrix @ p_matrix
    expected_triangles = float(np.trace(p_cubed)) / 6.0
    return volume, overlap, expected_triangles


def compute_delta_star(test_graph: nx.Graph) -> tuple[int, int]:
    """Compute the real test graph's triangle count and simple edge count.

    Args:
        test_graph: The held-out reference graph (may contain self-loops).

    Returns:
        `(delta_star, simple_graph_edges)`: the triangle count and edge count of
        `strip_self_loops(test_graph)`.
    """
    simple = strip_self_loops(test_graph)
    delta_star = sum(nx.triangles(simple).values()) // 3
    return delta_star, simple.number_of_edges()


def t_max_ceiling(omega: NDArray[np.float64], volume: float) -> NDArray[np.float64]:
    """Chanpuriya edge-independence ceiling `T_max(omega) = (sqrt(2)/3) * (omega*V)^1.5`.

    Args:
        omega: Overlap value(s) to evaluate the ceiling at.
        volume: Matched expected volume `V`.

    Returns:
        `T_max(omega)`, same shape as `omega`.
    """
    return (math.sqrt(2.0) / 3.0) * np.power(omega * volume, 1.5)


def compute_ov_min(delta_star: int, volume: float) -> float:
    """Compute the minimum overlap at matched volume for which `T_max == delta_star`.

    Args:
        delta_star: The real test graph's triangle count.
        volume: Matched expected volume `V`.

    Returns:
        `Ov_min = (3 * delta_star / sqrt(2)) ** (2/3) / V`. Returns `inf` if
        `volume <= 0` (degenerate guard; not expected on real data).
    """
    if volume <= 0:
        return float("inf")
    return float((3.0 * delta_star / math.sqrt(2.0)) ** (2.0 / 3.0) / volume)


def ceiling_curve(
    volume: float, overlap: float, ov_min: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate the ceiling curve on a fixed grid plus the exact `overlap`/`ov_min` points.

    Args:
        volume: Matched expected volume `V`.
        overlap: The scorer's measured overlap `Ov`.
        ov_min: The minimum overlap required to reach `delta_star`.

    Returns:
        `(omega, t_max)`: `omega` is `linspace(0, 1, 513)` unioned with `{overlap,
        ov_min}` (sorted, deduplicated by `numpy.union1d`); `t_max` is
        `T_max(omega)` at `volume`, aligned with `omega`.
    """
    grid = np.linspace(0.0, 1.0, _OMEGA_GRID_POINTS, dtype=np.float64)
    omega = np.union1d(grid, np.array([overlap, ov_min], dtype=np.float64))
    curve = t_max_ceiling(omega, volume)
    return omega, curve


def sanity_check(expected_triangles: float, overlap: float, volume: float) -> None:
    """Assert the Chanpuriya bound `E[Delta] <= T_max(Ov) * (1 + 1e-9)` holds.

    This is a theorem check on our own computation, not a research finding: the
    Chanpuriya bound is proven, so a violation means our identities/curve are
    computed incorrectly.

    Args:
        expected_triangles: `E[Delta] = tr(P^3) / 6`.
        overlap: The scorer's measured overlap `Ov`.
        volume: The expected volume `V`.

    Raises:
        AssertionError: If the bound is violated.
    """
    bound = float(t_max_ceiling(np.asarray(overlap, dtype=np.float64), volume))
    if not expected_triangles <= bound * (1.0 + 1e-9):
        raise AssertionError(
            f"Chanpuriya bound violated: E_triangles={expected_triangles!r} > "
            f"T_max(Ov)={bound!r} (Ov={overlap!r}, V={volume!r}); this indicates a bug "
            "in our own computation, not a real finding"
        )


def compute_g2_ceiling(
    artifact: ScoresArtifact, test_graph: nx.Graph, *, strategy: str
) -> G2Result:
    """Run the full G2 edge-independence ceiling computation.

    Args:
        artifact: The loaded candidate-universe scores artifact.
        test_graph: The held-out reference test graph (may contain self-loops).
        strategy: Split strategy name (validated against `artifact.meta["strategy"]`).

    Returns:
        The `G2Result`.

    Raises:
        ValueError: If `artifact` fails :func:`validate_universe`.
        AssertionError: If the Chanpuriya sanity bound is violated (a bug in this
            module's own computation).
    """
    n_test_nodes = test_graph.number_of_nodes()
    validate_universe(artifact, strategy=strategy, n_test_nodes=n_test_nodes)

    p_matrix, n_self_pairs, self_pair_prob_mass, n_pairs_used = build_dense_probs(artifact)
    volume, overlap, expected_triangles = compute_identities(p_matrix)
    delta_star, simple_graph_edges = compute_delta_star(test_graph)
    ov_min = compute_ov_min(delta_star, volume)

    sanity_check(expected_triangles, overlap, volume)

    omega, t_max_curve = ceiling_curve(volume, overlap, ov_min)
    headroom_triangles = expected_triangles / delta_star if delta_star > 0 else float("inf")
    headroom_overlap = overlap / ov_min if ov_min > 0 else float("inf")

    return G2Result(
        n_nodes=len(artifact.node_ids),
        n_pairs_used=n_pairs_used,
        n_self_pairs=n_self_pairs,
        self_pair_prob_mass=self_pair_prob_mass,
        volume=volume,
        overlap=overlap,
        expected_triangles=expected_triangles,
        delta_star=delta_star,
        simple_graph_edges=simple_graph_edges,
        ov_min=ov_min,
        headroom_triangles=headroom_triangles,
        headroom_overlap=headroom_overlap,
        omega=omega,
        t_max_curve=t_max_curve,
        checkpoint_id=str(artifact.meta.get("checkpoint_id")),
        model_family=str(artifact.meta.get("model_family")),
        strategy=strategy,
    )


def to_json_payload(result: G2Result) -> dict[str, object]:
    """Build the deterministic JSON-serializable payload for a `G2Result`.

    Args:
        result: The computed G2 ceiling result.

    Returns:
        A plain dict matching the `g2_results.json` output contract: identities,
        reference stats, headroom ratios, the curve arrays, artifact meta, and the
        two protocol-required caveat strings.
    """
    return {
        "n_nodes": result.n_nodes,
        "n_pairs_used": result.n_pairs_used,
        "n_self_pairs": result.n_self_pairs,
        "self_pair_prob_mass": result.self_pair_prob_mass,
        "volume": result.volume,
        "overlap": result.overlap,
        "expected_triangles": result.expected_triangles,
        "delta_star": result.delta_star,
        "simple_graph_edges": result.simple_graph_edges,
        "ov_min": result.ov_min,
        "headroom_triangles": result.headroom_triangles,
        "headroom_overlap": result.headroom_overlap,
        "curve": {
            "omega": result.omega.tolist(),
            "t_max": result.t_max_curve.tolist(),
        },
        "artifact_meta": {
            "checkpoint_id": result.checkpoint_id,
            "model_family": result.model_family,
            "strategy": result.strategy,
        },
        "caveats": [CAVEAT_VACUOUS_AT_OV_1, CAVEAT_EXACT_IDENTITIES],
    }


def write_json(result: G2Result, path: Path) -> None:
    """Write the deterministic `g2_results.json` payload for `result` to `path`.

    Same inputs always produce byte-identical output: `sort_keys=True`, fixed
    `indent=2` formatting, and no wall-clock-derived fields.

    Args:
        result: The computed G2 ceiling result.
        path: Destination JSON path (parent directories are created).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_json_payload(result)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def format_number(value: float, sig: int = 6) -> str:
    """Format `value` with `sig` significant digits (shared by JSON caption and figure).

    Args:
        value: The number to format.
        sig: Number of significant digits.

    Returns:
        A `%g`-style string. `inf`/`nan` are rendered via their normal `repr`.
    """
    if math.isinf(value) or math.isnan(value):
        return repr(value)
    if value == 0:
        return "0"
    return f"{value:.{sig}g}"


_FIGURE_WIDTH = 640
_FIGURE_HEIGHT = 320
_MARGIN_LEFT = 60
_MARGIN_RIGHT = 24
_MARGIN_TOP = 24
_MARGIN_BOTTOM = 44
_PLOT_WIDTH = _FIGURE_WIDTH - _MARGIN_LEFT - _MARGIN_RIGHT
_PLOT_HEIGHT = _FIGURE_HEIGHT - _MARGIN_TOP - _MARGIN_BOTTOM
_PLOT_LEFT = float(_MARGIN_LEFT)
_PLOT_RIGHT = float(_MARGIN_LEFT + _PLOT_WIDTH)
_PLOT_TOP = float(_MARGIN_TOP)
_PLOT_BOTTOM = float(_MARGIN_TOP + _PLOT_HEIGHT)

_STYLE = """<style>
  :root {
    --ground:#f5f7f9; --surface:#fff; --ink:#141a22; --ink-soft:#46515f;
    --ink-faint:#6b7686; --hair:#dde3ea; --accent:#0e7c6b; --accent-2:#c2571f;
    --grid:#e6ebf1; --mono:ui-monospace,"SF Mono","Cascadia Code",Menlo,Consolas,monospace;
    --font:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  }
  @media (prefers-color-scheme:dark) {
    :root {
      --ground:#0d1117; --surface:#161c24; --ink:#e9edf3; --ink-soft:#aeb8c6;
      --ink-faint:#7f8b9c; --hair:#2a323d; --accent:#2fd6bf; --accent-2:#e78a4e;
      --grid:#242c36;
    }
  }
  :root[data-theme="light"] {
    --ground:#f5f7f9; --surface:#fff; --ink:#141a22; --ink-soft:#46515f;
    --ink-faint:#6b7686; --hair:#dde3ea; --accent:#0e7c6b; --accent-2:#c2571f; --grid:#e6ebf1;
  }
  :root[data-theme="dark"] {
    --ground:#0d1117; --surface:#161c24; --ink:#e9edf3; --ink-soft:#aeb8c6;
    --ink-faint:#7f8b9c; --hair:#2a323d; --accent:#2fd6bf; --accent-2:#e78a4e; --grid:#242c36;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--ground); color:var(--ink); font-family:var(--font); }
  .wrap { max-width:760px; margin:0 auto; padding:clamp(1.5rem,4vw,3rem) clamp(1rem,4vw,2rem); }
  .eyebrow { font-family:var(--mono); font-size:.72rem; letter-spacing:.14em;
             text-transform:uppercase; color:var(--accent-2); margin:0 0 .5rem; }
  h1 { font-size:clamp(1.3rem,3vw,1.8rem); line-height:1.15; margin:0 0 .6rem; font-weight:660; }
  .lede { margin:0 0 1.4rem; max-width:64ch; color:var(--ink-soft); font-size:.98rem; }
  .lede code { font-family:var(--mono); color:var(--ink); }
  .panel { background:var(--surface); border:1px solid var(--hair); border-radius:14px;
           padding:1.1rem 1.2rem 1.3rem; }
  .chart { width:100%; height:auto; display:block; }
  .chart .axis { stroke:var(--ink-faint); stroke-width:1; }
  .chart .curve { fill:none; stroke:var(--accent); stroke-width:2.5; stroke-linejoin:round; }
  .chart .vline { stroke:var(--accent-2); stroke-width:1.5; stroke-dasharray:4 3; }
  .chart .hline { stroke:var(--ink-faint); stroke-width:1; stroke-dasharray:2 3; }
  .chart .mlabel { fill:var(--accent-2); font-family:var(--mono); font-size:10px; }
  .caption { margin:1rem 0 0; font-size:.85rem; color:var(--ink-soft); line-height:1.55; }
  .caption b { color:var(--ink); font-weight:620; }
</style>
"""


def render_figure_html(result: G2Result) -> str:
    """Render the standalone G2 ceiling HTML figure (inline SVG, no external resources).

    Log-y ceiling curve `T_max(omega)` for `omega` in `(0, 1]`, with vertical
    markers at the measured overlap `Ov` and the minimum required overlap
    `Ov_min`, horizontal markers at `delta_star` and `E[triangles]`, and a caption
    quoting `V`, `Ov`, `Ov_min`, `delta_star`, `E[triangles]`, and both protocol
    caveat strings. Rendered entirely from `result` (single source of truth).

    Args:
        result: The computed G2 ceiling result.

    Returns:
        A complete standalone HTML fragment (title + inline `<style>` + body),
        with zero external resources.
    """
    plot_mask = (result.omega > 0) & (result.omega <= 1.0)
    xs = result.omega[plot_mask]
    ys = result.t_max_curve[plot_mask]

    x_max = max(1.0, float(result.ov_min), float(result.overlap), 1e-9)

    positive_y = [float(v) for v in ys.tolist() if v > 0]
    positive_y += [float(v) for v in (result.delta_star, result.expected_triangles) if v > 0]
    if not positive_y:
        positive_y = [1.0]
    y_floor = max(min(positive_y) * 0.5, 1e-300)
    y_ceiling = max(positive_y) * 1.15

    log_floor = math.log10(y_floor)
    log_ceiling = math.log10(y_ceiling)
    if log_ceiling <= log_floor:
        log_ceiling = log_floor + 1.0

    def x_px(x: float) -> float:
        return _PLOT_LEFT + (x / x_max) * _PLOT_WIDTH

    def y_px(y: float) -> float:
        y_clamped = max(y, y_floor)
        frac = (math.log10(y_clamped) - log_floor) / (log_ceiling - log_floor)
        return _PLOT_TOP + (1.0 - frac) * _PLOT_HEIGHT

    points = " ".join(
        f"{x_px(x):.2f},{y_px(y):.2f}" for x, y in zip(xs.tolist(), ys.tolist(), strict=True)
    )

    ov_x = x_px(float(result.overlap))
    ov_min_x = x_px(float(result.ov_min))
    delta_star_y = y_px(float(result.delta_star)) if result.delta_star > 0 else _PLOT_BOTTOM
    e_tri_y = (
        y_px(float(result.expected_triangles)) if result.expected_triangles > 0 else _PLOT_BOTTOM
    )

    v_str = format_number(result.volume)
    ov_str = format_number(result.overlap)
    ov_min_str = format_number(result.ov_min)
    delta_star_str = str(result.delta_star)
    e_tri_str = format_number(result.expected_triangles)

    body = f"""<div class="wrap">
  <p class="eyebrow">Gate G2 &mdash; edge-independence ceiling</p>
  <h1>How many triangles can an edge-independent assembly express?</h1>
  <p class="lede">Chanpuriya et al. 2111.00048 exact identities on the cached candidate-universe
    soft scores. <code>T_max(&omega;) = (&radic;2/3)&middot;(&omega;&middot;V)^1.5</code> bounds
    the expected triangle count any edge-independent, calibrated assembly can reach at overlap
    &omega;, holding volume V fixed.</p>
  <div class="panel">
    <svg class="chart" viewBox="0 0 {_FIGURE_WIDTH} {_FIGURE_HEIGHT}" role="img"
         aria-label="Log-scale ceiling curve T_max(omega) with markers at measured overlap,
         minimum required overlap, delta_star, and E_triangles">
      <line class="axis"
            x1="{_PLOT_LEFT}" y1="{_PLOT_BOTTOM}" x2="{_PLOT_RIGHT}" y2="{_PLOT_BOTTOM}"></line>
      <line class="axis"
            x1="{_PLOT_LEFT}" y1="{_PLOT_TOP}" x2="{_PLOT_LEFT}" y2="{_PLOT_BOTTOM}"></line>
      <polyline class="curve" points="{points}"></polyline>
      <line class="hline" data-marker="delta_star"
            x1="{_PLOT_LEFT}" y1="{delta_star_y:.2f}"
            x2="{_PLOT_RIGHT}" y2="{delta_star_y:.2f}"></line>
      <line class="hline" data-marker="E_triangles"
            x1="{_PLOT_LEFT}" y1="{e_tri_y:.2f}" x2="{_PLOT_RIGHT}" y2="{e_tri_y:.2f}"></line>
      <line class="vline" data-marker="Ov"
            x1="{ov_x:.2f}" y1="{_PLOT_TOP}" x2="{ov_x:.2f}" y2="{_PLOT_BOTTOM}"></line>
      <line class="vline" data-marker="Ov_min"
            x1="{ov_min_x:.2f}" y1="{_PLOT_TOP}" x2="{ov_min_x:.2f}" y2="{_PLOT_BOTTOM}"></line>
      <text class="mlabel" x="{ov_x:.2f}" y="{_PLOT_TOP - 8}" text-anchor="middle">Ov</text>
      <text class="mlabel" x="{ov_min_x:.2f}" y="{_PLOT_TOP - 8}"
            text-anchor="middle">Ov_min</text>
      <text class="mlabel" x="{_PLOT_LEFT}" y="{_PLOT_BOTTOM + 16}">0</text>
      <text class="mlabel" x="{_PLOT_RIGHT}" y="{_PLOT_BOTTOM + 16}"
            text-anchor="end">{x_max:.3g}</text>
    </svg>
    <p class="caption">
      <b>V</b> = {v_str} &middot; <b>Ov</b> = {ov_str} &middot; <b>Ov_min</b> = {ov_min_str}
      &middot; <b>delta*</b> = {delta_star_str} &middot; <b>E[triangles]</b> = {e_tri_str}.
      {CAVEAT_VACUOUS_AT_OV_1}. {CAVEAT_EXACT_IDENTITIES}.
    </p>
  </div>
</div>
"""
    return f"<title>G2 Edge-Independence Ceiling</title>\n{_STYLE}{body}"


def load_test_graph(benchmark_root: Path, strategy: str) -> nx.Graph:
    """Unpickle the strategy's held-out reference test graph.

    Reads `<benchmark_root>/<strategy>/test_graph.pkl` directly (the pinned
    `networkx.Graph` artifact, spec section 9); this module only needs the test
    graph, so it does not run the full `load_benchmark` package verification.

    Args:
        benchmark_root: Benchmark package root (e.g. `data/benchmark_2025_neurips`).
        strategy: Split strategy name.

    Returns:
        The pickled `networkx.Graph` (self-loops not stripped; callers strip via
        :func:`src.eval.graph_metrics.strip_self_loops` as needed).

    Raises:
        TypeError: If the unpickled object is not a `networkx.Graph`.
    """
    path = benchmark_root / strategy / "test_graph.pkl"
    with path.open("rb") as f:
        graph = pickle.load(f)
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"{path}: expected a pickled networkx.Graph, got {type(graph)!r}")
    return graph


def build_parser() -> argparse.ArgumentParser:
    """Build the G2 ceiling CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.experiments.g2_ceiling",
        description=(
            "Compute the G2 edge-independence ceiling (Chanpuriya 2111.00048) from a "
            "cached candidate-universe scores artifact."
        ),
    )
    parser.add_argument("--universe", type=Path, required=True, help="candidate scores .npz")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--strategy", default="breadth_first")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--figure", type=Path, default=None, help="optional standalone HTML figure path"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point: compute the G2 ceiling and write `g2_results.json` (+ figure).

    Args:
        argv: Argument list (defaults to `sys.argv[1:]`).

    Raises:
        SystemExit: On argument errors (via `argparse`).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    artifact = load_scores(args.universe)
    benchmark_root = args.data_root / _BENCHMARK_SUBDIR
    test_graph = load_test_graph(benchmark_root, args.strategy)
    result = compute_g2_ceiling(artifact, test_graph, strategy=args.strategy)

    output_path = args.output_dir / "g2_results.json"
    write_json(result, output_path)
    logger.info("wrote G2 ceiling results to %s", output_path)

    if args.figure is not None:
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        args.figure.write_text(render_figure_html(result), encoding="utf-8")
        logger.info("wrote G2 ceiling figure to %s", args.figure)


__all__ = [
    "CAVEAT_EXACT_IDENTITIES",
    "CAVEAT_VACUOUS_AT_OV_1",
    "G2Result",
    "build_dense_probs",
    "build_parser",
    "ceiling_curve",
    "compute_delta_star",
    "compute_g2_ceiling",
    "compute_identities",
    "compute_ov_min",
    "format_number",
    "load_test_graph",
    "main",
    "render_figure_html",
    "sanity_check",
    "t_max_ceiling",
    "to_json_payload",
    "validate_universe",
    "write_json",
]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()

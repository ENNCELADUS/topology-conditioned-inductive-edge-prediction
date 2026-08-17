"""S2 evaluation: are features-only-generated latents node-aligned on held-out test nodes?

Loads the frozen Stage-A autoencoder (`ae.pt`) plus the Stage-B prior/det/marg
checkpoints (`prior.pt`/`det.pt`/`marg.pt`, `src.experiments.s2_latent_topology.train`'s
output) and scores seven arms -- GEN (features-only prior sample), UNC
(unconditional prior sample), SHUF (shuffled-feature prior sample), DET
(deterministic features-only regressor), AE (encode-then-decode latent ceiling),
B0 (a frozen classification baseline, if supplied), and MARG (a strictly
per-node degree baseline, if a B0 artifact and `marg.pt` are supplied) -- over
every `test_node_buckets.pkl` node set, plus an optional full-test-region pass.
Every readout is edge-level (AUPRC), assembled-graph (GS/RD/MMD-ratio degree,
clustering, spectral), or degree-alignment (Spearman, hub recall) -- reusing
`src.eval.graph_metrics`/`src.eval.assembly` throughout rather than
reimplementing the canonical evaluator.

Determinism: every random draw is seeded from `seed` via a `torch.Generator`
(`seed * 10000 + <arm offset>`) or `np.random.default_rng((seed, <codes...>))`
entropy tuples -- never Python's built-in `hash`. The returned payload carries
no wall-clock field, so two runs with identical inputs are byte-identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import networkx as nx
import numpy as np
import numpy.typing as npt
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from src.data.features import FeatureStore
from src.eval.assembly import assemble_degree_quota, assemble_graph, density_matched_threshold
from src.eval.graph_metrics import (
    STATISTICS,
    BucketReference,
    MMDConfig,
    clustering_histogram,
    compute_graph_similarity,
    compute_relative_density,
    degree_histogram,
    evaluate_assembled_graph_with_reference,
    laplacian_spectrum_histogram,
    mmd_squared,
    precompute_bucket_reference,
    strip_self_loops,
)
from src.experiments.g1_hardened_e2 import (
    _BENCHMARK_SUBDIR,
    load_test_graph,
    load_test_node_buckets,
)
from src.experiments.s2_latent_topology.data import build_features
from src.experiments.s2_latent_topology.model import (
    DetTwin,
    MargDegreeRegressor,
    RegionAutoencoder,
    SetDenoiser,
    sample_prior,
)
from src.score_universe import load_scores, validate_artifact_precision

_ARM_CODES: dict[str, int] = {"gen": 1, "unc": 2, "shuf": 3, "det": 4, "ae": 5, "b0": 6, "marg": 7}
_METRICS: tuple[str, ...] = (
    "gs",
    "rd",
    "spearman",
    "hub_recall",
    "auprc",
    "free_density_expected",
    "free_density_hard",
)
_METRIC_CODES: dict[str, int] = {name: code + 1 for code, name in enumerate(_METRICS)}
_STOCHASTIC_OFFSETS: dict[str, int] = {"gen": 11, "unc": 12, "shuf": 13}


@dataclass(frozen=True)
class EvalContext:
    """Once-per-run S2 evaluation state: test-side graph, features, and references.

    Attributes:
        nodes: Sorted `test_graph` node ids; a node's row in `features` is its
            position in this list.
        features: `(N_test, D)` fp32 F0 mean-pooled feature matrix in `nodes` order.
        g_ref: The held-out test graph as loaded (self-loops kept).
        g_simple: `strip_self_loops(g_ref)`.
        buckets: Bucket size -> list of node-id sets (`test_node_buckets.pkl`).
        bucket_ref: Precomputed reference-side bucket state.
        b0_probs: Per-row sigmoid probabilities of the B0 candidate-universe
            scores artifact, or `None` when no artifact was supplied.
        b0_lookup: Flat `(N_test * N_test,)` row-index array into `b0_probs`
            (`-1` where absent); both `u*N+v` and `v*N+u` point at the same row.
        featureless_in_test: Count of `nodes` absent from the feature store.
    """

    nodes: list[str]
    features: torch.Tensor
    g_ref: nx.Graph
    g_simple: nx.Graph
    buckets: dict[int, list[set[str]]]
    bucket_ref: BucketReference
    b0_probs: npt.NDArray[np.float64] | None
    b0_lookup: npt.NDArray[np.int64] | None
    featureless_in_test: int


def build_eval_context(
    *,
    data_root: Path,
    strategy: str,
    store: FeatureStore,
    cache_dir: Path,
    b0_universe: Path | None,
) -> EvalContext:
    """Build the once-per-run S2 evaluation context from the test-side benchmark artifacts.

    Args:
        data_root: Directory containing `benchmark_2025_neurips/`.
        strategy: Benchmark split strategy (e.g. `"breadth_first"`).
        store: Frozen per-node feature store to mean-pool F0 features from.
        cache_dir: Directory the F0 feature cache is written to.
        b0_universe: Optional B0 candidate-universe scores `.npz` artifact,
            validated via `validate_artifact_precision`; `None` disables the
            B0, MARG, and full-region-B0 arms.

    Returns:
        The `EvalContext`.
    """
    benchmark_root = data_root / _BENCHMARK_SUBDIR
    g_ref = load_test_graph(benchmark_root, strategy)
    buckets = load_test_node_buckets(benchmark_root, strategy)
    nodes = sorted(g_ref.nodes())
    g_simple = strip_self_loops(g_ref)

    cache_dir.mkdir(parents=True, exist_ok=True)
    features, missing = build_features(store, nodes, cache_path=cache_dir / "f0_test.pt")
    bucket_ref = precompute_bucket_reference(g_ref, buckets, MMDConfig())

    b0_probs: npt.NDArray[np.float64] | None = None
    b0_lookup: npt.NDArray[np.int64] | None = None
    if b0_universe is not None:
        artifact = load_scores(b0_universe)
        validate_artifact_precision(artifact, label=str(b0_universe))
        b0_probs = artifact.probs()
        n = len(nodes)
        node_index = {node_id: i for i, node_id in enumerate(nodes)}
        b0_lookup = np.full(n * n, -1, dtype=np.int64)
        for row, (u, v) in enumerate(artifact.pairs()):
            if u in node_index and v in node_index:
                iu, iv = node_index[u], node_index[v]
                b0_lookup[iu * n + iv] = row
                b0_lookup[iv * n + iu] = row

    return EvalContext(
        nodes=nodes,
        features=features,
        g_ref=g_ref,
        g_simple=g_simple,
        buckets=buckets,
        bucket_ref=bucket_ref,
        b0_probs=b0_probs,
        b0_lookup=b0_lookup,
        featureless_in_test=len(missing),
    )


# --------------------------------------------------------------------------- checkpoint loading


def _load_ae(
    path: Path, device: torch.device
) -> tuple[RegionAutoencoder, torch.Tensor, torch.Tensor]:
    """Rebuild `RegionAutoencoder` plus its latent mean/std from `ae.pt`."""
    payload = cast(dict[str, Any], torch.load(path, map_location="cpu"))
    cfg = cast(dict[str, Any], payload["config"])
    model = RegionAutoencoder(
        in_dim=cast(int, payload["in_dim"]),
        z_dim=cast(int, cfg["z_dim"]),
        dim=cast(int, cfg["ae_dim"]),
        layers=cast(int, cfg["ae_layers"]),
        rrwp_k=cast(int, cfg["ae_rrwp_k"]),
        n_heads=cast(int, cfg["ae_heads"]),
        seeds=cast(int, cfg["ae_seeds"]),
        d_model=cast(int, cfg["ae_d_model"]),
    )
    model.load_state_dict(cast(dict[str, torch.Tensor], payload["model"]))
    model.to(device)
    model.eval()
    latent_mean = cast(torch.Tensor, payload["latent_mean"]).to(device=device, dtype=torch.float32)
    latent_std = cast(torch.Tensor, payload["latent_std"]).to(device=device, dtype=torch.float32)
    return model, latent_mean, latent_std


def _load_prior(path: Path, device: torch.device) -> SetDenoiser:
    """Rebuild `SetDenoiser` from `prior.pt`."""
    payload = cast(dict[str, Any], torch.load(path, map_location="cpu"))
    cfg = cast(dict[str, Any], payload["config"])
    model = SetDenoiser(
        in_dim=cast(int, payload["in_dim"]),
        z_dim=cast(int, cfg["z_dim"]),
        width=cast(int, cfg["set_width"]),
        sab_layers=cast(int, cfg["set_layers"]),
        heads=cast(int, cfg["set_heads"]),
        pma_seeds=cast(int, cfg["set_pma_seeds"]),
        time_dim=cast(int, cfg["time_dim"]),
    )
    model.load_state_dict(cast(dict[str, torch.Tensor], payload["model"]))
    model.to(device)
    model.eval()
    return model


def _load_det(path: Path, device: torch.device) -> DetTwin:
    """Rebuild `DetTwin` from `det.pt`."""
    payload = cast(dict[str, Any], torch.load(path, map_location="cpu"))
    cfg = cast(dict[str, Any], payload["config"])
    model = DetTwin(
        in_dim=cast(int, payload["in_dim"]),
        z_dim=cast(int, cfg["z_dim"]),
        width=cast(int, cfg["set_width"]),
        sab_layers=cast(int, cfg["set_layers"]),
        heads=cast(int, cfg["set_heads"]),
        pma_seeds=cast(int, cfg["set_pma_seeds"]),
    )
    model.load_state_dict(cast(dict[str, torch.Tensor], payload["model"]))
    model.to(device)
    model.eval()
    return model


def _load_marg(path: Path, device: torch.device) -> MargDegreeRegressor:
    """Rebuild `MargDegreeRegressor` from `marg.pt`."""
    payload = cast(dict[str, Any], torch.load(path, map_location="cpu"))
    cfg = cast(dict[str, Any], payload["config"])
    model = MargDegreeRegressor(
        in_dim=cast(int, payload["in_dim"]), hidden=cast(int, cfg["marg_hidden"])
    )
    model.load_state_dict(cast(dict[str, torch.Tensor], payload["model"]))
    model.to(device)
    model.eval()
    return model


# --------------------------------------------------------------------------- batching helpers


def _build_size_batch(
    ctx: EvalContext, node_sets: list[set[str]], node_index: dict[str, int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[str]]]:
    """Batch one bucket size's node sets into dense `x`/`adj`/`mask` tensors.

    Sorted-node-id order within each set is the local order used everywhere
    else in this module: it equals global test-node index order, since
    `ctx.nodes` is itself sorted ascending.

    Returns:
        `(x, adj, mask, ordered_sets)`: `x` is `(B, n, D)` features, `adj` is
        `(B, n, n)` the true loopless induced adjacency, `mask` is `(B, n)`
        all-ones, and `ordered_sets` is each set's sorted node-id list.
    """
    ordered_sets = [sorted(node_set) for node_set in node_sets]
    n = len(ordered_sets[0])
    b = len(ordered_sets)
    x = torch.stack(
        [ctx.features[[node_index[node_id] for node_id in s]] for s in ordered_sets]
    ).to(device)
    adj = torch.zeros(b, n, n, dtype=torch.float32, device=device)
    for row, s in enumerate(ordered_sets):
        local = {node_id: i for i, node_id in enumerate(s)}
        for u, v in ctx.g_simple.subgraph(s).edges():
            iu, iv = local[u], local[v]
            adj[row, iu, iv] = 1.0
            adj[row, iv, iu] = 1.0
    mask = torch.ones(b, n, device=device)
    return x, adj, mask, ordered_sets


def _shuffle_features_within_sets(x: torch.Tensor, *, seed: int, size: int) -> torch.Tensor:
    """Independently permute each set's `n` feature rows (SHUF's conditioning transform)."""
    b, n, _ = x.shape
    out = x.clone()
    for set_index in range(b):
        rng = np.random.default_rng((seed, 7, size, set_index))
        perm = torch.as_tensor(rng.permutation(n), dtype=torch.long, device=x.device)
        out[set_index] = x[set_index, perm]
    return out


def _zero_diag_np(t: torch.Tensor) -> npt.NDArray[np.float64]:
    """Convert a `(..., n, n)` tensor to float64 numpy with the trailing diagonal zeroed."""
    arr = t.detach().cpu().numpy().astype(np.float64)
    idx = np.arange(arr.shape[-1])
    arr[..., idx, idx] = 0.0
    return arr


def _upper_pairs(
    mat: npt.NDArray[np.float64], node_ids: list[str]
) -> tuple[list[tuple[str, str]], npt.NDArray[np.float64]]:
    """Return the `i < j` node-id pairs and their values from a symmetric matrix."""
    n = len(node_ids)
    iu, iv = np.triu_indices(n, k=1)
    pairs = [(node_ids[i], node_ids[j]) for i, j in zip(iu.tolist(), iv.tolist(), strict=True)]
    return pairs, mat[iu, iv]


def _descriptors(g: nx.Graph) -> dict[str, npt.NDArray[np.float64]]:
    """Degree/clustering/spectral descriptor histograms, keyed like `graph_metrics.STATISTICS`."""
    return {
        "degree": degree_histogram(g),
        "clustering": clustering_histogram(g),
        "spectral": laplacian_spectrum_histogram(g),
    }


# --------------------------------------------------------------------------- arm forward passes


def _run_stochastic(
    name: str,
    prior: SetDenoiser,
    ae: RegionAutoencoder,
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    k_draws: int,
    flow_steps: int,
    seed: int,
    size: int,
    device: torch.device,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Sample the GEN/UNC/SHUF prior arm and decode every draw to edge probabilities.

    UNC zeros `x` before conditioning; SHUF permutes each set's feature rows
    independently; GEN uses `x` unchanged. All three seed `sample_prior`'s
    generator from `seed * 10000 + offset` (11/12/13 respectively).

    Returns:
        `(mean_probs, draw0_probs, all_probs, activity)`: `mean_probs`/
        `draw0_probs` are `(B, n, n)` zero-diagonal probability matrices (mean
        over draws, and the first draw alone); `all_probs` is `(K, B, n, n)`;
        `activity` is `(K, B, n)`, the de-standardized activity latent `a_i`.

    Raises:
        ValueError: If `name` is not one of `"gen"`, `"unc"`, `"shuf"`.
    """
    if name == "gen":
        x_arm = x
    elif name == "unc":
        x_arm = torch.zeros_like(x)
    elif name == "shuf":
        x_arm = _shuffle_features_within_sets(x, seed=seed, size=size)
    else:
        raise ValueError(f"unknown stochastic arm {name!r}")

    generator = torch.Generator(device=device).manual_seed(seed * 10000 + _STOCHASTIC_OFFSETS[name])
    with torch.no_grad():
        draws_std = sample_prior(
            prior,
            x_arm,
            mask,
            z_dim=ae.z_dim,
            n_draws=k_draws,
            steps=flow_steps,
            generator=generator,
        )
        latents = draws_std * latent_std + latent_mean
        k, b, n, dim = latents.shape
        flat_latents = latents.reshape(k * b, n, dim)
        flat_mask = mask.unsqueeze(0).expand(k, b, n).reshape(k * b, n)
        logits = ae.decode(flat_latents, flat_mask).reshape(k, b, n, n)
        probs = torch.sigmoid(logits)
    activity = latents[..., 0]
    mean_probs = probs.mean(dim=0)
    return (
        _zero_diag_np(mean_probs),
        _zero_diag_np(probs[0]),
        probs.detach().cpu().numpy().astype(np.float64),
        activity.detach().cpu().numpy().astype(np.float64),
    )


def _run_det(
    det: DetTwin,
    ae: RegionAutoencoder,
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
) -> npt.NDArray[np.float64]:
    """Run the deterministic DET arm: predict standardized latents, de-standardize, decode."""
    with torch.no_grad():
        pred_std = det(x, mask)
        latents = pred_std * latent_std + latent_mean
        logits = ae.decode(latents, mask)
        probs = torch.sigmoid(logits)
    return _zero_diag_np(probs)


def _run_ae(
    ae: RegionAutoencoder, x: torch.Tensor, adj_true: torch.Tensor, mask: torch.Tensor
) -> npt.NDArray[np.float64]:
    """Run the AE (latent-ceiling) arm: encode the true adjacency, then decode it back."""
    with torch.no_grad():
        latents = ae.encode(x, adj_true, mask)
        logits = ae.decode(latents, mask)
        probs = torch.sigmoid(logits)
    return _zero_diag_np(probs)


def _b0_prob_matrix(
    ctx: EvalContext, node_index: dict[str, int], ordered_set: list[str]
) -> tuple[npt.NDArray[np.float64], int]:
    """Gather the B0 candidate-universe probability matrix for one node set.

    Returns:
        `(matrix, missing)`: the `(n, n)` zero-diagonal symmetric probability
        matrix (absent pairs score 0.0) and the count of absent pairs.
    """
    assert ctx.b0_probs is not None
    assert ctx.b0_lookup is not None
    n = len(ordered_set)
    n_test = len(ctx.nodes)
    gidx = np.array([node_index[node_id] for node_id in ordered_set], dtype=np.int64)
    iu, iv = np.triu_indices(n, k=1)
    flat = gidx[iu] * n_test + gidx[iv]
    rows = ctx.b0_lookup[flat]
    missing = int((rows < 0).sum())
    pair_probs = np.where(rows >= 0, ctx.b0_probs[np.clip(rows, 0, None)], 0.0)
    mat = np.zeros((n, n), dtype=np.float64)
    mat[iu, iv] = pair_probs
    mat[iv, iu] = pair_probs
    return mat, missing


# --------------------------------------------------------------------------- per-set readouts


@dataclass(frozen=True)
class _SetReadout:
    """One node set's density-matched graph and readout metrics for one arm.

    Attributes:
        gs: Official per-subgraph graph similarity vs the reference subgraph.
        rd: Official per-subgraph relative density vs the reference subgraph.
        spearman: Spearman correlation of `deg_hat` vs true in-set degree, or
            `None` when either ranking is constant.
        hub_recall: Top-`ceil(0.1n)` hub overlap fraction, or `None` when
            `ceil(0.1n) == 0`.
        auprc: AUPRC of in-set pair probabilities vs true edge labels, or
            `None` for a single-class set.
        free_density_expected: `sum(probs) / target_edges`.
        free_density_hard: `(probs >= 0.5).sum() / target_edges`.
        g_pred: The density-matched assembled graph.
    """

    gs: float
    rd: float
    spearman: float | None
    hub_recall: float | None
    auprc: float | None
    free_density_expected: float
    free_density_hard: float
    g_pred: nx.Graph


def _safe_spearman(
    deg_hat: npt.NDArray[np.float64], true_degree: npt.NDArray[np.float64]
) -> float | None:
    """Spearman rank correlation of `deg_hat` vs `true_degree`, or `None` if undefined."""
    if float(np.std(deg_hat)) == 0.0 or float(np.std(true_degree)) == 0.0:
        return None
    rho, _ = spearmanr(deg_hat, true_degree)
    return None if math.isnan(rho) else float(rho)


def _hub_recall(
    deg_hat: npt.NDArray[np.float64], true_degree: npt.NDArray[np.float64], node_ids: list[str]
) -> float | None:
    """Top-`ceil(0.1n)` hub overlap fraction between `deg_hat` and `true_degree` rankings.

    Ties are broken by ascending node id, matching sort key `(-value, node_id)`.
    """
    n = len(node_ids)
    k = math.ceil(0.1 * n)
    if k <= 0:
        return None
    top_hat = sorted(range(n), key=lambda i: (-deg_hat[i], node_ids[i]))[:k]
    top_true = sorted(range(n), key=lambda i: (-true_degree[i], node_ids[i]))[:k]
    return len(set(top_hat) & set(top_true)) / k


def _compute_set_readout(
    *,
    probs: npt.NDArray[np.float64],
    true_adj: npt.NDArray[np.float64],
    node_ids: list[str],
    ref_subgraph: nx.Graph,
    target_edges: int,
) -> _SetReadout:
    """Density-match `probs` against `target_edges` and compute the full readout bundle."""
    pairs, pair_probs = _upper_pairs(probs, node_ids)
    n = len(node_ids)
    iu, iv = np.triu_indices(n, k=1)
    pair_labels = true_adj[iu, iv].astype(np.int64)

    threshold = density_matched_threshold(pair_probs, target_edges)
    g_pred = assemble_graph(pairs, pair_probs, threshold=threshold, nodes=node_ids)
    gs = compute_graph_similarity(g_pred, ref_subgraph)
    rd = compute_relative_density(g_pred, ref_subgraph)

    deg_hat = probs.sum(axis=1)
    true_degree = true_adj.sum(axis=1)
    spearman = _safe_spearman(deg_hat, true_degree)
    hub_recall = _hub_recall(deg_hat, true_degree, node_ids)
    auprc = (
        float(average_precision_score(pair_labels, pair_probs))
        if len(set(pair_labels.tolist())) >= 2
        else None
    )
    free_density_expected = float(pair_probs.sum() / target_edges)
    free_density_hard = float((pair_probs >= 0.5).sum() / target_edges)

    return _SetReadout(
        gs=gs,
        rd=rd,
        spearman=spearman,
        hub_recall=hub_recall,
        auprc=auprc,
        free_density_expected=free_density_expected,
        free_density_hard=free_density_hard,
        g_pred=g_pred,
    )


def _accumulate_coherence(
    coherence: dict[str, list[float]],
    draws_probs: npt.NDArray[np.float64],
    draws_activity: npt.NDArray[np.float64],
    *,
    node_ids: list[str],
    target_edges: int,
    ref_subgraph: nx.Graph,
    mean_prob_graph: nx.Graph,
) -> None:
    """Accumulate one set's coherence statistics into the running per-size lists.

    Args:
        coherence: Running per-size lists, mutated in place.
        draws_probs: `(K, n, n)` per-draw probability matrices for this set.
        draws_activity: `(K, n)` per-draw activity latent for this set.
        node_ids: This set's sorted node ids.
        target_edges: The set's true induced loopless edge count.
        ref_subgraph: The reference (ground-truth) induced subgraph.
        mean_prob_graph: The density-matched graph built from the arm's mean probs.
    """
    coherence["activity_variance"].append(float(np.var(draws_activity, axis=0).mean()))

    draw_tri: list[float] = []
    draw_clust: list[float] = []
    draw_degvar: list[float] = []
    draw_gs: list[float] = []
    for k in range(draws_probs.shape[0]):
        pairs, pair_probs = _upper_pairs(draws_probs[k], node_ids)
        threshold = density_matched_threshold(pair_probs, target_edges)
        g = assemble_graph(pairs, pair_probs, threshold=threshold, nodes=node_ids)
        draw_tri.append(sum(nx.triangles(g).values()) / 3.0)
        draw_clust.append(float(nx.average_clustering(g)))
        degrees = np.array([d for _, d in g.degree()], dtype=np.float64)
        draw_degvar.append(float(np.var(degrees)))
        draw_gs.append(compute_graph_similarity(g, ref_subgraph))
    coherence["draw_triangle"].append(float(np.mean(draw_tri)))
    coherence["draw_clustering"].append(float(np.mean(draw_clust)))
    coherence["draw_degree_var"].append(float(np.mean(draw_degvar)))
    coherence["draw_gs"].append(float(np.mean(draw_gs)))

    coherence["mean_prob_triangle"].append(sum(nx.triangles(mean_prob_graph).values()) / 3.0)
    coherence["mean_prob_clustering"].append(float(nx.average_clustering(mean_prob_graph)))
    mean_degrees = np.array([d for _, d in mean_prob_graph.degree()], dtype=np.float64)
    coherence["mean_prob_degree_var"].append(float(np.var(mean_degrees)))


# --------------------------------------------------------------------------- aggregation


def _bootstrap_ci(
    values: list[float],
    *,
    seed: int,
    arm_code: int,
    size: int,
    metric_code: int,
    n_boot: int = 1000,
) -> tuple[float, float, float]:
    """Percentile-bootstrap `(mean, ci_lo, ci_hi)` of `values` (2.5/97.5 percentiles)."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])
    rng = np.random.default_rng((seed, arm_code, size, metric_code))
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def _aggregate_metrics(
    metrics: dict[str, list[float]],
    none_counts: dict[str, int],
    *,
    seed: int,
    arm_code: int,
    size: int,
) -> dict[str, Any]:
    """Bootstrap-aggregate each per-set metric list into a mean + percentile-CI row.

    An empty metric list has no defined mean/CI; those fields are emitted as
    `None` (never `NaN`) so the payload stays finite-or-`None` throughout.
    """
    per_size: dict[str, Any] = {}
    for name in _METRICS:
        mean, lo, hi = _bootstrap_ci(
            metrics[name], seed=seed, arm_code=arm_code, size=size, metric_code=_METRIC_CODES[name]
        )
        per_size[name] = {
            "mean": None if math.isnan(mean) else mean,
            "ci_lo": None if math.isnan(lo) else lo,
            "ci_hi": None if math.isnan(hi) else hi,
            "n": len(metrics[name]),
            "n_none": none_counts.get(name, 0),
        }
    return per_size


def _aggregate_mmd(
    bank_descs: dict[str, list[npt.NDArray[np.float64]]],
    ref_descs: list[dict[str, npt.NDArray[np.float64]]],
) -> dict[str, Any]:
    """Compute raw/floor/ratio MMD^2 for a per-size bank of descriptor histograms.

    Returns an empty dict (disclosed absence, matching the "no bank data"
    convention) rather than raising when fewer than 2 reference samples are
    available for the odd/even floor split -- always true in production (buckets
    are pinned at 50 sets), but a defensive guard for a near-fully-skipped size.
    """
    if not bank_descs[STATISTICS[0]] or len(ref_descs) < 2:
        return {}
    config = MMDConfig()
    ref_by_stat = {stat: [d[stat] for d in ref_descs] for stat in STATISTICS}
    raw = {stat: mmd_squared(bank_descs[stat], ref_by_stat[stat], config) for stat in STATISTICS}
    floor = {
        stat: mmd_squared(ref_by_stat[stat][::2], ref_by_stat[stat][1::2], config)
        for stat in STATISTICS
    }
    ratio = {stat: raw[stat] / max(floor[stat], 1e-12) for stat in STATISTICS}
    return {"raw": raw, "floor": floor, "ratio": ratio}


def _macro_over_sizes(
    arm_by_size: dict[int, dict[str, Any]], sizes: list[int]
) -> dict[str, float | None]:
    """Unweighted mean across sizes of each metric's per-size mean (`None` if none defined)."""
    macro: dict[str, float | None] = {}
    for name in _METRICS:
        values = [
            arm_by_size[size]["per_size"][name]["mean"]
            for size in sizes
            if arm_by_size[size]["per_size"][name]["mean"] is not None
        ]
        macro[name] = float(np.mean(values)) if values else None
    return macro


def _mmd_macro_over_sizes(
    arm_by_size: dict[int, dict[str, Any]], sizes: list[int]
) -> dict[str, float | None]:
    """Canonical macro MMD ratio: mean raw across sizes over mean floor across sizes.

    Matches `evaluate_assembled_graph_with_reference`, which averages raw and
    reference MMD separately before dividing; averaging per-size ratios instead
    would overweight sizes with tiny reference floors.
    """
    present = [size for size in sizes if arm_by_size[size]["mmd"]]
    if not present:
        return dict.fromkeys(STATISTICS, None)
    macro: dict[str, float | None] = {}
    for stat in STATISTICS:
        raw = float(np.mean([arm_by_size[size]["mmd"]["raw"][stat] for size in present]))
        floor = float(np.mean([arm_by_size[size]["mmd"]["floor"][stat] for size in present]))
        macro[stat] = raw / max(floor, 1e-12)
    return macro


# --------------------------------------------------------------------------- per-arm-per-size


def _process_arm_size(
    *,
    arm_code: int,
    size: int,
    seed: int,
    mean_probs: npt.NDArray[np.float64],
    draw0_probs: npt.NDArray[np.float64],
    ordered_sets: list[list[str]],
    true_adj: npt.NDArray[np.float64],
    ref_subgraphs: list[nx.Graph],
    ref_descs: list[dict[str, npt.NDArray[np.float64]]],
    all_probs: npt.NDArray[np.float64] | None = None,
    activity: npt.NDArray[np.float64] | None = None,
) -> dict[str, Any]:
    """Compute one arm's readouts, MMD bank, and (stochastic arms only) coherence for one size.

    The MMD bank uses `draw0_probs` per set for a stochastic arm (`all_probs`
    given) and `mean_probs` otherwise, per the house convention. A set whose
    true induced loopless edge count is 0 is skipped entirely and counted.
    """
    metrics: dict[str, list[float]] = {name: [] for name in _METRICS}
    none_counts: dict[str, int] = dict.fromkeys(_METRICS, 0)
    n_skipped = 0
    bank_descs: dict[str, list[npt.NDArray[np.float64]]] = {stat: [] for stat in STATISTICS}
    stochastic = all_probs is not None and activity is not None
    coherence: dict[str, list[float]] | None = (
        {
            "activity_variance": [],
            "draw_triangle": [],
            "draw_clustering": [],
            "draw_degree_var": [],
            "mean_prob_triangle": [],
            "mean_prob_clustering": [],
            "mean_prob_degree_var": [],
            "draw_gs": [],
        }
        if stochastic
        else None
    )

    for b, node_ids in enumerate(ordered_sets):
        target_edges = int(round(float(true_adj[b].sum()))) // 2
        if target_edges == 0:
            n_skipped += 1
            continue
        readout = _compute_set_readout(
            probs=mean_probs[b],
            true_adj=true_adj[b],
            node_ids=node_ids,
            ref_subgraph=ref_subgraphs[b],
            target_edges=target_edges,
        )
        metrics["gs"].append(readout.gs)
        metrics["rd"].append(readout.rd)
        metrics["free_density_expected"].append(readout.free_density_expected)
        metrics["free_density_hard"].append(readout.free_density_hard)
        for name, value in (
            ("spearman", readout.spearman),
            ("hub_recall", readout.hub_recall),
            ("auprc", readout.auprc),
        ):
            if value is None:
                none_counts[name] += 1
            else:
                metrics[name].append(value)

        bank_source = draw0_probs[b] if stochastic else mean_probs[b]
        bank_pairs, bank_pair_probs = _upper_pairs(bank_source, node_ids)
        bank_threshold = density_matched_threshold(bank_pair_probs, target_edges)
        bank_graph = assemble_graph(
            bank_pairs, bank_pair_probs, threshold=bank_threshold, nodes=node_ids
        )
        bank_d = _descriptors(bank_graph)
        for stat in STATISTICS:
            bank_descs[stat].append(bank_d[stat])

        if coherence is not None and all_probs is not None and activity is not None:
            _accumulate_coherence(
                coherence,
                all_probs[:, b],
                activity[:, b],
                node_ids=node_ids,
                target_edges=target_edges,
                ref_subgraph=ref_subgraphs[b],
                mean_prob_graph=readout.g_pred,
            )

    per_size = _aggregate_metrics(metrics, none_counts, seed=seed, arm_code=arm_code, size=size)
    per_size["n_sets"] = len(ordered_sets)
    per_size["n_skipped"] = n_skipped
    result: dict[str, Any] = {"per_size": per_size, "mmd": _aggregate_mmd(bank_descs, ref_descs)}
    if coherence is not None:
        result["coherence"] = {
            name: (float(np.mean(values)) if values else None) for name, values in coherence.items()
        }
    return result


def _largest_remainder_quota(deg_hat: npt.NDArray[np.float64], total: int) -> npt.NDArray[np.int64]:
    """Round `deg_hat`, scaled to sum exactly `total`, via largest-remainder apportionment."""
    n = deg_hat.shape[0]
    if total <= 0 or n == 0:
        return np.zeros(n, dtype=np.int64)
    weight_sum = float(deg_hat.sum())
    scaled = (
        deg_hat * (total / weight_sum)
        if weight_sum > 0.0
        else np.full(n, total / n, dtype=np.float64)
    )
    floor = np.floor(scaled).astype(np.int64)
    deficit = int(total - int(floor.sum()))
    if deficit > 0:
        remainder = scaled - floor
        order = np.argsort(-remainder, kind="stable")[:deficit]
        floor[order] += 1
    return floor


def _process_marg_size(
    *,
    arm_code: int,
    size: int,
    seed: int,
    marg_deg_hat: npt.NDArray[np.float64],
    ordered_sets: list[list[str]],
    true_adj: npt.NDArray[np.float64],
    ref_subgraphs: list[nx.Graph],
    ref_descs: list[dict[str, npt.NDArray[np.float64]]],
    ctx: EvalContext,
    node_index: dict[str, int],
) -> dict[str, Any]:
    """Compute the MARG arm's readouts and MMD bank for one size.

    Reuses B0 probabilities as the quota-assembly score and as the AUPRC row
    (spec: "MARG's AUPRC row = B0's"); the degree-alignment readouts (Spearman,
    hub recall, free density) use the regressor's own `deg_hat`, not the
    assembled graph's realized degrees.
    """
    metrics: dict[str, list[float]] = {name: [] for name in _METRICS}
    none_counts: dict[str, int] = dict.fromkeys(_METRICS, 0)
    n_skipped = 0
    bank_descs: dict[str, list[npt.NDArray[np.float64]]] = {stat: [] for stat in STATISTICS}
    shortfall_total = 0
    residual_total = 0
    b0_missing_total = 0

    for b, node_ids in enumerate(ordered_sets):
        n = len(node_ids)
        target_edges = int(round(float(true_adj[b].sum()))) // 2
        if target_edges == 0:
            n_skipped += 1
            continue
        b0_mat, missing = _b0_prob_matrix(ctx, node_index, node_ids)
        b0_missing_total += missing
        iu, iv = np.triu_indices(n, k=1)
        pairs = [(node_ids[i], node_ids[j]) for i, j in zip(iu.tolist(), iv.tolist(), strict=True)]
        pair_probs = b0_mat[iu, iv]
        pair_labels = true_adj[b][iu, iv].astype(np.int64)

        deg_hat = marg_deg_hat[b, :n]
        quotas_arr = _largest_remainder_quota(deg_hat, 2 * target_edges)
        quotas = dict(zip(node_ids, quotas_arr.tolist(), strict=True))
        g_pred, stats = assemble_degree_quota(pairs, iu, iv, pair_probs, quotas, node_ids)
        shortfall_total += stats.shortfall
        residual_total += stats.residual_quota

        metrics["gs"].append(compute_graph_similarity(g_pred, ref_subgraphs[b]))
        metrics["rd"].append(compute_relative_density(g_pred, ref_subgraphs[b]))

        true_degree = true_adj[b].sum(axis=1)
        spearman = _safe_spearman(deg_hat, true_degree)
        if spearman is None:
            none_counts["spearman"] += 1
        else:
            metrics["spearman"].append(spearman)

        hub_recall = _hub_recall(deg_hat, true_degree, node_ids)
        if hub_recall is None:
            none_counts["hub_recall"] += 1
        else:
            metrics["hub_recall"].append(hub_recall)

        if len(set(pair_labels.tolist())) >= 2:
            metrics["auprc"].append(float(average_precision_score(pair_labels, pair_probs)))
        else:
            none_counts["auprc"] += 1

        metrics["free_density_expected"].append(float(deg_hat.sum() / (2 * target_edges)))
        # Unscaled regressor edge budget: quota assembly rescales to target_edges,
        # so realized_edges/target_edges would be ~1 by construction.
        metrics["free_density_hard"].append(round(float(deg_hat.sum()) / 2.0) / target_edges)

        bank_d = _descriptors(g_pred)
        for stat in STATISTICS:
            bank_descs[stat].append(bank_d[stat])

    per_size = _aggregate_metrics(metrics, none_counts, seed=seed, arm_code=arm_code, size=size)
    per_size["n_sets"] = len(ordered_sets)
    per_size["n_skipped"] = n_skipped
    per_size["quota_shortfall_total"] = shortfall_total
    per_size["quota_residual_total"] = residual_total
    per_size["b0_missing_pairs_total"] = b0_missing_total
    return {"per_size": per_size, "mmd": _aggregate_mmd(bank_descs, ref_descs)}


# --------------------------------------------------------------------------- full region


def _run_full_region(
    *,
    ctx: EvalContext,
    ae: RegionAutoencoder,
    prior: SetDenoiser,
    det: DetTwin,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    k_draws: int,
    flow_steps: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run the optional full-test-region GEN/DET/B0 pass and evaluate the house flow.

    AE is skipped (disclosed): it requires the true adjacency, which a
    features-only full-region pass does not have.
    """
    nodes = ctx.nodes
    x = ctx.features.unsqueeze(0).to(device)
    mask = torch.ones(1, len(nodes), device=device)
    target_edges = ctx.g_simple.number_of_edges()

    gen_probs, _, _, _ = _run_stochastic(
        "gen",
        prior,
        ae,
        x,
        mask,
        k_draws=k_draws,
        flow_steps=flow_steps,
        seed=seed,
        size=len(nodes),
        device=device,
        latent_mean=latent_mean,
        latent_std=latent_std,
    )
    det_probs = _run_det(det, ae, x, mask, latent_mean=latent_mean, latent_std=latent_std)

    result: dict[str, Any] = {
        "ae": {
            "skipped": True,
            "reason": "AE requires the true adjacency; undefined for the features-only full region",
        },
        "gen": _full_region_block(gen_probs[0], target_edges=target_edges, nodes=nodes, ctx=ctx),
        "det": _full_region_block(det_probs[0], target_edges=target_edges, nodes=nodes, ctx=ctx),
    }
    if ctx.b0_probs is not None and ctx.b0_lookup is not None:
        n = len(nodes)
        iu, iv = np.triu_indices(n, k=1)
        flat = iu.astype(np.int64) * n + iv.astype(np.int64)
        rows = ctx.b0_lookup[flat]
        pair_probs = np.where(rows >= 0, ctx.b0_probs[np.clip(rows, 0, None)], 0.0)
        b0_mat = np.zeros((n, n), dtype=np.float64)
        b0_mat[iu, iv] = pair_probs
        b0_mat[iv, iu] = pair_probs
        result["b0"] = _full_region_block(b0_mat, target_edges=target_edges, nodes=nodes, ctx=ctx)
    return result


def _full_region_block(
    probs: npt.NDArray[np.float64], *, target_edges: int, nodes: list[str], ctx: EvalContext
) -> dict[str, Any]:
    """Density-match, assemble, and evaluate one full-region probability matrix.

    Reports the five topology numbers named separately (never aggregated): the
    BFS-macro GS/RD plus the three MMD ratios, and the global simple-edge GS/RD
    against `ctx.g_simple`.
    """
    pairs, pair_probs = _upper_pairs(probs, nodes)
    threshold = density_matched_threshold(pair_probs, target_edges)
    g_pred = assemble_graph(pairs, pair_probs, threshold=threshold, nodes=nodes)
    report = evaluate_assembled_graph_with_reference(g_pred, ctx.bucket_ref, MMDConfig())
    g_pred_simple = strip_self_loops(g_pred)
    return {
        "threshold": float(threshold),
        "gs_bfs_macro": report.graph_similarity,
        "rd_bfs_macro": report.relative_density,
        "mmd_ratio": dict(report.mmd_ratio),
        "gs_global": compute_graph_similarity(g_pred_simple, ctx.g_simple),
        "rd_global": compute_relative_density(g_pred_simple, ctx.g_simple),
    }


# --------------------------------------------------------------------------- orchestration


def run_s2_eval(
    *,
    ctx: EvalContext,
    ae_path: Path,
    prior_path: Path,
    det_path: Path,
    marg_path: Path | None,
    k_draws: int,
    flow_steps: int,
    seed: int,
    device: str,
    full_region: bool,
) -> dict[str, object]:
    """Run every S2 arm over every test-side bucket and return the JSON-ready payload.

    Args:
        ctx: The once-per-run evaluation context (`build_eval_context`).
        ae_path: Path to the Stage-A `ae.pt` checkpoint.
        prior_path: Path to the Stage-B `prior.pt` (`SetDenoiser`) checkpoint.
        det_path: Path to the `det.pt` (`DetTwin`) checkpoint.
        marg_path: Optional `marg.pt` (`MargDegreeRegressor`) checkpoint path;
            `None`, or a missing `ctx.b0_probs`, disables the MARG arm.
        k_draws: Number of prior draws `K` for the GEN/UNC/SHUF arms.
        flow_steps: Number of Euler integration steps for `sample_prior`.
        seed: Base seed for every randomized step (generators, permutations, bootstrap).
        device: Torch device string (`"cpu"`, `"cuda"`, `"mps"`).
        full_region: Whether to additionally run the optional full-test-region pass.

    Returns:
        The JSON-serializable results payload (`meta`, `arms`, optionally `full_region`).
    """
    dev = torch.device(device)
    ae, latent_mean, latent_std = _load_ae(ae_path, dev)
    prior = _load_prior(prior_path, dev)
    det = _load_det(det_path, dev)
    marg = (
        _load_marg(marg_path, dev) if marg_path is not None and ctx.b0_probs is not None else None
    )

    node_index = {node_id: i for i, node_id in enumerate(ctx.nodes)}
    sizes = sorted(ctx.buckets)

    arm_names = ["gen", "unc", "shuf", "det", "ae"]
    if ctx.b0_probs is not None:
        arm_names.append("b0")
    if marg is not None:
        arm_names.append("marg")

    arms: dict[str, dict[int, dict[str, Any]]] = {name: {} for name in arm_names}
    skipped_by_size: dict[int, int] = {}

    for size in sizes:
        x, adj_true_t, mask, ordered_sets = _build_size_batch(
            ctx, ctx.buckets[size], node_index, dev
        )
        true_adj = adj_true_t.detach().cpu().numpy().astype(np.float64)
        ref_subgraphs = ctx.bucket_ref.ref_subgraphs[size]
        ref_descs = ctx.bucket_ref.ref_descriptors[size]
        skipped_by_size[size] = int(sum(int(round(float(row.sum()))) == 0 for row in true_adj))

        for name in ("gen", "unc", "shuf"):
            mean_p, draw0_p, all_p, activity = _run_stochastic(
                name,
                prior,
                ae,
                x,
                mask,
                k_draws=k_draws,
                flow_steps=flow_steps,
                seed=seed,
                size=size,
                device=dev,
                latent_mean=latent_mean,
                latent_std=latent_std,
            )
            arms[name][size] = _process_arm_size(
                arm_code=_ARM_CODES[name],
                size=size,
                seed=seed,
                mean_probs=mean_p,
                draw0_probs=draw0_p,
                ordered_sets=ordered_sets,
                true_adj=true_adj,
                ref_subgraphs=ref_subgraphs,
                ref_descs=ref_descs,
                all_probs=all_p,
                activity=activity,
            )

        det_p = _run_det(det, ae, x, mask, latent_mean=latent_mean, latent_std=latent_std)
        arms["det"][size] = _process_arm_size(
            arm_code=_ARM_CODES["det"],
            size=size,
            seed=seed,
            mean_probs=det_p,
            draw0_probs=det_p,
            ordered_sets=ordered_sets,
            true_adj=true_adj,
            ref_subgraphs=ref_subgraphs,
            ref_descs=ref_descs,
        )

        ae_p = _run_ae(ae, x, adj_true_t, mask)
        arms["ae"][size] = _process_arm_size(
            arm_code=_ARM_CODES["ae"],
            size=size,
            seed=seed,
            mean_probs=ae_p,
            draw0_probs=ae_p,
            ordered_sets=ordered_sets,
            true_adj=true_adj,
            ref_subgraphs=ref_subgraphs,
            ref_descs=ref_descs,
        )

        if ctx.b0_probs is not None:
            b0_p = np.stack([_b0_prob_matrix(ctx, node_index, s)[0] for s in ordered_sets])
            arms["b0"][size] = _process_arm_size(
                arm_code=_ARM_CODES["b0"],
                size=size,
                seed=seed,
                mean_probs=b0_p,
                draw0_probs=b0_p,
                ordered_sets=ordered_sets,
                true_adj=true_adj,
                ref_subgraphs=ref_subgraphs,
                ref_descs=ref_descs,
            )

        if marg is not None:
            with torch.no_grad():
                n_tensor = torch.full((len(ordered_sets),), size, dtype=torch.long, device=dev)
                deg_hat_t = marg(x, n_tensor)
            arms["marg"][size] = _process_marg_size(
                arm_code=_ARM_CODES["marg"],
                size=size,
                seed=seed,
                marg_deg_hat=deg_hat_t.detach().cpu().numpy().astype(np.float64),
                ordered_sets=ordered_sets,
                true_adj=true_adj,
                ref_subgraphs=ref_subgraphs,
                ref_descs=ref_descs,
                ctx=ctx,
                node_index=node_index,
            )

    result_arms: dict[str, Any] = {}
    for name in arm_names:
        by_size = arms[name]
        entry: dict[str, Any] = {
            "per_size": {str(size): by_size[size]["per_size"] for size in sizes},
            "macro": _macro_over_sizes(by_size, sizes),
            "mmd": {str(size): by_size[size]["mmd"] for size in sizes},
            "mmd_macro": _mmd_macro_over_sizes(by_size, sizes),
        }
        if name in ("gen", "unc", "shuf"):
            entry["coherence"] = {str(size): by_size[size]["coherence"] for size in sizes}
        result_arms[name] = entry

    payload: dict[str, Any] = {
        "meta": {
            "k_draws": k_draws,
            "flow_steps": flow_steps,
            "seed": seed,
            "device": device,
            "sizes": sizes,
            "arm_codes": dict(_ARM_CODES),
            "metric_codes": dict(_METRIC_CODES),
            "featureless_in_test": ctx.featureless_in_test,
            "has_b0": ctx.b0_probs is not None,
            "has_marg": marg is not None,
            "skipped_sets_by_size": {str(size): skipped_by_size[size] for size in sizes},
            "ref_subgraph_self_loop_asymmetry_note": (
                "ref_subgraphs (bucket_ref, built from g_ref) keep self-loops; predicted "
                "density-matched graphs are assembled from non-self in-set pairs only, so "
                "GS/RD compare a self-loop-free predicted subgraph against a self-loop-"
                "bearing reference subgraph -- a fixed asymmetry, disclosed once here."
            ),
        },
        "arms": result_arms,
    }
    if full_region:
        payload["full_region"] = _run_full_region(
            ctx=ctx,
            ae=ae,
            prior=prior,
            det=det,
            latent_mean=latent_mean,
            latent_std=latent_std,
            k_draws=k_draws,
            flow_steps=flow_steps,
            seed=seed,
            device=dev,
        )
    return payload


# --------------------------------------------------------------------------- markdown tables


def _fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.6g}"


def render_tables_markdown(payload: dict[str, object]) -> str:
    """Render every arm's GS/Spearman/hub-recall, macro, MMD-ratio, and coherence tables.

    Args:
        payload: A `run_s2_eval` result payload (or an equivalent dict).

    Returns:
        Markdown text.
    """
    data = cast(dict[str, Any], payload)
    meta = data["meta"]
    sizes: list[int] = meta["sizes"]
    arms: dict[str, Any] = data["arms"]

    lines: list[str] = ["# S2 latent-topology evaluation tables", ""]

    lines.append("## Arm x size: GS / Spearman / hub recall")
    lines.append("")
    lines.append("| arm | size | GS | GS CI | spearman | hub recall | AUPRC |")
    lines.append("|---|---|---|---|---|---|---|")
    for arm_name, entry in arms.items():
        per_size = entry["per_size"]
        for size in sizes:
            row = per_size[str(size)]
            gs, spearman, hub, auprc = row["gs"], row["spearman"], row["hub_recall"], row["auprc"]
            lines.append(
                f"| {arm_name} | {size} | {_fmt(gs['mean'])} "
                f"| [{_fmt(gs['ci_lo'])}, {_fmt(gs['ci_hi'])}] "
                f"| {_fmt(spearman['mean'])} | {_fmt(hub['mean'])} | {_fmt(auprc['mean'])} |"
            )
    lines.append("")

    lines.append("## Arm macro")
    lines.append("")
    lines.append("| arm | " + " | ".join(_METRICS) + " |")
    lines.append("|---|" + "---|" * len(_METRICS))
    for arm_name, entry in arms.items():
        macro = entry["macro"]
        values = " | ".join(_fmt(macro[m]) for m in _METRICS)
        lines.append(f"| {arm_name} | {values} |")
    lines.append("")

    lines.append("## MMD ratio (macro across sizes)")
    lines.append("")
    lines.append("| arm | degree | clustering | spectral |")
    lines.append("|---|---|---|---|")
    for arm_name, entry in arms.items():
        mmd_macro = entry["mmd_macro"]
        values = " | ".join(_fmt(mmd_macro.get(stat, float("nan"))) for stat in STATISTICS)
        lines.append(f"| {arm_name} | {values} |")
    lines.append("")

    lines.append("## Coherence (GEN / UNC / SHUF)")
    lines.append("")
    lines.append(
        "| arm | size | activity var | draw triangle | draw clustering | draw deg var | "
        "mean-prob triangle | mean-prob clustering | mean-prob deg var | mean per-draw GS |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for arm_name in ("gen", "unc", "shuf"):
        if arm_name not in arms:
            continue
        coherence = arms[arm_name].get("coherence", {})
        for size in sizes:
            row = coherence.get(str(size))
            if row is None:
                continue
            lines.append(
                f"| {arm_name} | {size} | {_fmt(row.get('activity_variance'))} "
                f"| {_fmt(row.get('draw_triangle'))} | {_fmt(row.get('draw_clustering'))} "
                f"| {_fmt(row.get('draw_degree_var'))} | {_fmt(row.get('mean_prob_triangle'))} "
                f"| {_fmt(row.get('mean_prob_clustering'))} "
                f"| {_fmt(row.get('mean_prob_degree_var'))} | {_fmt(row.get('draw_gs'))} |"
            )
    lines.append("")

    if "full_region" in data:
        lines.append("## Full region (five numbers, never aggregated)")
        lines.append("")
        full = data["full_region"]
        for arm_name, block in full.items():
            if block.get("skipped"):
                lines.append(f"- **{arm_name}**: skipped ({block.get('reason')})")
                continue
            mmd_ratio = block["mmd_ratio"]
            lines.append(
                f"- **{arm_name}**: GS(BFS-macro)={_fmt(block['gs_bfs_macro'])}, "
                f"RD(BFS-macro)={_fmt(block['rd_bfs_macro'])}, "
                f"degree MMD={_fmt(mmd_ratio.get('degree'))}, "
                f"clustering MMD={_fmt(mmd_ratio.get('clustering'))}, "
                f"spectral MMD={_fmt(mmd_ratio.get('spectral'))}, "
                f"GS(global simple)={_fmt(block['gs_global'])}, "
                f"RD(global simple)={_fmt(block['rd_global'])}"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


__all__ = [
    "EvalContext",
    "build_eval_context",
    "render_tables_markdown",
    "run_s2_eval",
]

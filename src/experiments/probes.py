"""Registered EgoStitch representation probes (spec Sec 14.3(3), Sec 14.4.7).

Diagnostics-only linear probes over frozen encoder states (STE token states,
in the registered representation-probe protocol): out-of-fold ``R^2`` from a
closed-form ridge fit, implemented in plain numpy (no sklearn dependency, per
the pinned protocol). ``degree_partialled_r2`` additionally residualizes both
``states`` and ``targets`` against node degree before probing, isolating any
predictive signal beyond what a trivial degree confound would already give
away for free (registration ``diagnostics_nonbinding``: "linear probes on
frozen STE token states: R2 to degree / ego-density / clustering (ridge
lambda 1e-3, 5-fold), plus degree-partialled variants and Pi-consistency").

The rev-3.1 artifact additionally records direct alignment, grounding-pool
recall, relational-state, and slot-dispersion measurements. Structural targets
are always explicitly scoped to train-side ``E_msg``; no default can select a
sealed universe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import networkx as nx
import numpy as np
import torch
from numpy.typing import NDArray

_RIDGE_LAMBDA = 1e-3
_N_FOLDS = 5
_MIN_VARIANCE = 1e-10
E2E_PROBE_FORMAT = "egostitch_e2e_probe_v2"
_E2E_PROBE_V1_FORMAT = "egostitch_e2e_probe_v1"
_E2E_PAIR_LIMIT = 4096
E2EProbeScope = Literal["formal_train"]
E2E_PROBE_SCOPES: tuple[E2EProbeScope, ...] = ("formal_train",)
_DISPERSION_NAMES = (
    "pi_slot_std",
    "h_pairwise_cosine_mean",
    "adj_offdiag_std",
    "plan_row_entropy",
)


def _load_train_side_probe_inputs(
    strategy_dir: Path,
    *,
    expected_missing_features: Sequence[str],
) -> tuple[list[str], list[tuple[str, str]]]:
    """Load probe identities/positives without opening any test-side artifact."""
    from src.train_egostitch import _read_labeled_pairs

    with (strategy_dir / "split.pkl").open("rb") as handle:
        split_payload = pickle.load(handle)  # noqa: S301 - repository benchmark artifact
    if not isinstance(split_payload, dict) or "train" not in split_payload:
        raise ValueError("split.pkl must contain a train node collection")
    train_nodes = sorted(
        set(cast(Sequence[str], split_payload["train"]))
        - set(expected_missing_features)
    )
    train_pairs, train_labels = _read_labeled_pairs(strategy_dir / "train_edges.txt")
    positives = [
        pair
        for pair, label in zip(train_pairs, train_labels, strict=True)
        if int(label) == 1
    ]
    return train_nodes, positives


def build_probe_scope_context(
    scope: E2EProbeScope,
    *,
    data_root: Path,
    strategy: str,
    partition_seed: int,
    msg_fraction: float,
    expected_missing_features: Sequence[str],
) -> tuple[list[str], nx.Graph]:
    """Rebuild the exact node set and message graph a probe scope authorizes.

    ``formal_train`` is the only scope: every operative train node over the full
    train-side ``E_msg``. Re-validating a written artifact must reconstruct that
    universe exactly, or :func:`evaluate_e2e_probe_artifact` would compare it
    against the wrong ``g_struct_sha256`` and the wrong stored targets.

    :func:`produce_e2e_probe_artifact` deliberately keeps its own inline
    derivation because it additionally needs the partition and holdout objects
    for its per-role grounding caches.
    """
    from src import train_egostitch as te
    from src.data.partition import build_g_struct, derive_partition

    if scope not in E2E_PROBE_SCOPES:
        raise ValueError(f"unsupported E2E probe scope {scope!r}")
    strategy_dir = data_root / te._BENCHMARK_SUBDIR / strategy
    formal_nodes, train_positives = _load_train_side_probe_inputs(
        strategy_dir,
        expected_missing_features=expected_missing_features,
    )
    partition = derive_partition(
        train_positives, seed=partition_seed, msg_fraction=msg_fraction
    )
    return formal_nodes, build_g_struct(formal_nodes, partition.e_msg)


def _validate_pi_consistency_inputs(
    plan: torch.Tensor,
    cells: torch.Tensor,
) -> None:
    """Validate the common plan/cell shape contract for both consistency probes."""
    if plan.ndim != 3 or plan.shape[-1] != plan.shape[-2]:
        raise ValueError("Pi consistency plan must have shape (B, K, K)")
    if cells.shape != plan.shape or cells.dtype != torch.bool:
        raise ValueError("Pi consistency cells must be boolean and match the plan")
    if not bool(torch.isfinite(plan).all()) or bool((plan < 0).any()):
        raise ValueError("Pi consistency plan must be finite and nonnegative")


def pi_grounding_chain_consistency_v1(
    plan: torch.Tensor,
    same_real_grounded_identity: torch.Tensor,
    *,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
) -> torch.Tensor:
    """Measure the complete pointer/gate/grounding chain, not alignment alone.

    ``same_real_grounded_identity`` is true where the two pointer argmaxes
    select the same real shared-neighbor identity. The gate threshold is then
    applied here. Consequently this continuity probe is structurally zero when
    either endpoint's gates are shut, regardless of the quality of ``Pi``.
    """
    _validate_pi_consistency_inputs(plan, same_real_grounded_identity)
    expected_gate_shape = plan.shape[:2]
    if gate_a.shape != expected_gate_shape or gate_b.shape != expected_gate_shape:
        raise ValueError("Pi-consistency v1 gates must have shape (B, K)")
    with torch.autocast(device_type=plan.device.type, enabled=False):
        plan32 = plan.float()
        grounded = (gate_a[:, :, None] > 0.5) & (gate_b[:, None, :] > 0.5)
        eligible = same_real_grounded_identity & grounded
        return (plan32 * eligible.float()).sum(dim=(1, 2)) / plan32.sum(
            dim=(1, 2)
        ).clamp_min(1e-30)


def pi_alignment_consistency_v2(
    plan: torch.Tensor,
    same_identity_teacher_cells: torch.Tensor,
) -> torch.Tensor:
    """Measure alignment directly as ``Pi`` mass on the exact ``L_align`` set ``S``."""
    _validate_pi_consistency_inputs(plan, same_identity_teacher_cells)
    with torch.autocast(device_type=plan.device.type, enabled=False):
        plan32 = plan.float()
        return (plan32 * same_identity_teacher_cells.float()).sum(
            dim=(1, 2)
        ) / plan32.sum(dim=(1, 2)).clamp_min(1e-30)


def slot_recall_at_n_ground(
    graph: nx.Graph,
    nodes: Sequence[str],
    grounding_pool: Mapping[str, Sequence[str]],
    selected_slots: Mapping[str, Sequence[str]],
) -> NDArray[np.float64]:
    """Return grounded-slot recall rows within each node's own ``n_g`` pool.

    Each row is the fraction of true scoped neighbors recovered by at least one
    gate-active pointer-selected slot. Selected identities must belong to that
    node's supplied grounding pool, making the P0.2 pool recall an explicit
    ceiling. Isolates are excluded from the mean.
    """
    missing_graph = [node for node in nodes if node not in graph]
    if missing_graph:
        raise ValueError(f"slot-recall nodes missing from scoped graph: {missing_graph[:5]}")
    missing_pool = [node for node in nodes if node not in grounding_pool]
    if missing_pool:
        raise ValueError(f"slot-recall nodes missing grounding pools: {missing_pool[:5]}")
    missing_slots = [node for node in nodes if node not in selected_slots]
    if missing_slots:
        raise ValueError(f"slot-recall nodes missing selected slots: {missing_slots[:5]}")
    rows: list[float] = []
    for node in nodes:
        neighbors = {str(neighbor) for neighbor in graph.neighbors(node) if neighbor != node}
        pool = set(grounding_pool[node])
        selected = set(selected_slots[node])
        if not selected <= pool:
            raise ValueError(f"selected slot identity falls outside {node!r}'s grounding pool")
        if neighbors:
            rows.append(len(neighbors & selected) / len(neighbors))
    return np.asarray(rows, dtype=np.float64)


def probe_targets(graph: nx.Graph, nodes: Sequence[str]) -> dict[str, NDArray[np.float64]]:
    """Per-node probe regression targets via the spec Sec 13.6 evaluator convention.

    Computes the pinned ego-stat quantities with the NetworkX implementations
    (the same evaluator convention the Stage-1 ego-stat targets bind to):
    ``degree = deg(u)``, ``clustering = nx.clustering(G, u)``,
    ``ego_edges = |E(ego(u))|`` and ``ego_density = nx.density(ego(u))`` with
    ``ego(u) = G.subgraph(N(u) | {u})`` on the simple graph. Callers pass the
    message-partition structural graph (``G_struct``, spec Sec 9.3) so probe
    targets never touch the target test graph.

    Args:
        graph: The simple structural graph (no self-loops).
        nodes: Probe node ids, in the row order of the probe states.

    Returns:
        ``{"degree", "clustering", "ego_edges", "ego_density"}`` -> shape
        ``(len(nodes),)`` float64 arrays, row-aligned with `nodes`.

    Raises:
        ValueError: If any probe node is missing from `graph`.
    """
    missing = [node for node in nodes if node not in graph]
    if missing:
        raise ValueError(f"probe nodes missing from the graph: {missing[:5]}")
    clustering = nx.clustering(graph, nodes)
    degree = np.array([float(graph.degree(node)) for node in nodes], dtype=np.float64)
    clustering_arr = np.array([float(clustering[node]) for node in nodes], dtype=np.float64)
    ego_edges = np.empty(len(nodes), dtype=np.float64)
    ego_density = np.empty(len(nodes), dtype=np.float64)
    for i, node in enumerate(nodes):
        ego = graph.subgraph(set(graph.neighbors(node)) | {node})
        ego_edges[i] = float(ego.number_of_edges())
        ego_density[i] = float(nx.density(ego))
    return {
        "degree": degree,
        "clustering": clustering_arr,
        "ego_edges": ego_edges,
        "ego_density": ego_density,
    }


def _as_2d(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Reshape a 1-D feature array to a single-column 2-D array (no-op if already 2-D)."""
    reshaped: NDArray[np.float64] = values.reshape(-1, 1) if values.ndim == 1 else values
    return reshaped


def _kfold_indices(n: int, n_folds: int, seed: int) -> list[NDArray[np.int64]]:
    """Partition ``range(n)`` into `n_folds` shuffled, near-equal folds."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return [fold.astype(np.int64) for fold in np.array_split(order, n_folds)]


def _ridge_fit_predict(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    lam: float,
) -> NDArray[np.float64]:
    """Closed-form (mean-centered) ridge regression: fit on train, predict on test.

    ``w = (Xc^T Xc + lam I)^-1 Xc^T yc`` on centered train data; the intercept
    is recovered by re-adding the train means at predict time.
    """
    x_mean = x_train.mean(axis=0)
    y_mean = y_train.mean()
    x_centered = x_train - x_mean
    y_centered = y_train - y_mean
    gram = x_centered.T @ x_centered + lam * np.eye(x_centered.shape[1])
    weights = np.linalg.solve(gram, x_centered.T @ y_centered)
    predictions: NDArray[np.float64] = (x_test - x_mean) @ weights + y_mean
    return predictions


def linear_probe_r2(
    states: NDArray[np.float64],
    targets: NDArray[np.float64],
    *,
    lam: float = _RIDGE_LAMBDA,
    n_folds: int = _N_FOLDS,
    seed: int = 0,
) -> float:
    """Cross-validated closed-form ridge probe ``R^2``.

    Args:
        states: Shape ``(n,)`` or ``(n, d)`` frozen representation features.
        targets: Shape ``(n,)`` regression targets.
        lam: Ridge penalty (spec-pinned ``1e-3``).
        n_folds: Number of cross-validation folds (spec-pinned ``5``).
        seed: Fold-assignment seed (deterministic given fixed inputs).

    Returns:
        The out-of-fold ``R^2``. Returns ``0.0`` (rather than an undefined
        ``0/0``) when the held-out target has ~zero variance.

    Raises:
        ValueError: If there are fewer samples than `n_folds`.
    """
    states64 = _as_2d(np.asarray(states, dtype=np.float64))
    targets64 = np.asarray(targets, dtype=np.float64).reshape(-1)
    n = states64.shape[0]
    if n < n_folds:
        raise ValueError(f"linear_probe_r2 requires at least {n_folds} samples, got {n}")
    folds = _kfold_indices(n, n_folds, seed)
    predictions = np.empty(n, dtype=np.float64)
    for i, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        predictions[test_idx] = _ridge_fit_predict(
            states64[train_idx], targets64[train_idx], states64[test_idx], lam
        )
    residual = targets64 - predictions
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((targets64 - targets64.mean()) ** 2))
    if ss_tot < _MIN_VARIANCE:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _partial_out_train_test(
    train_values: NDArray[np.float64],
    test_values: NDArray[np.float64],
    train_degrees: NDArray[np.float64],
    test_degrees: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Residualize train/test values with a degree fit learned on train rows only."""
    train_design = np.column_stack([np.ones_like(train_degrees), train_degrees])
    test_design = np.column_stack([np.ones_like(test_degrees), test_degrees])
    coefficients, *_ = np.linalg.lstsq(train_design, train_values, rcond=None)
    return (
        train_values - train_design @ coefficients,
        test_values - test_design @ coefficients,
    )


def degree_partialled_r2(
    states: NDArray[np.float64],
    targets: NDArray[np.float64],
    degrees: NDArray[np.float64],
    *,
    lam: float = _RIDGE_LAMBDA,
    n_folds: int = _N_FOLDS,
    seed: int = 0,
) -> float:
    """Degree-partialled probe ``R^2``: residualize `states`/`targets` against `degrees` first.

    Args:
        states: Shape ``(n,)`` or ``(n, d)`` frozen representation features.
        targets: Shape ``(n,)`` regression targets.
        degrees: Shape ``(n,)`` node degrees — the confound to partial out.
        lam: Ridge penalty (spec-pinned ``1e-3``).
        n_folds: Number of cross-validation folds (spec-pinned ``5``).
        seed: Fold-assignment seed (deterministic given fixed inputs).

    Returns:
        The `linear_probe_r2` of the degree-residualized target from the
        degree-residualized states — signal beyond a pure degree confound.
        ``0.0`` when the degree-residualized target has ~zero variance (e.g.
        the target IS degree).
    """
    degrees64 = np.asarray(degrees, dtype=np.float64).reshape(-1)
    states64 = _as_2d(np.asarray(states, dtype=np.float64))
    targets64 = np.asarray(targets, dtype=np.float64).reshape(-1)
    n = states64.shape[0]
    if n < n_folds:
        raise ValueError(f"degree_partialled_r2 requires at least {n_folds} samples, got {n}")
    folds = _kfold_indices(n, n_folds, seed)
    predictions = np.empty(n, dtype=np.float64)
    residual_targets = np.empty(n, dtype=np.float64)
    for i, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        train_states, test_states = _partial_out_train_test(
            states64[train_idx], states64[test_idx], degrees64[train_idx], degrees64[test_idx]
        )
        train_targets, test_targets = _partial_out_train_test(
            targets64[train_idx], targets64[test_idx], degrees64[train_idx], degrees64[test_idx]
        )
        predictions[test_idx] = _ridge_fit_predict(train_states, train_targets, test_states, lam)
        residual_targets[test_idx] = test_targets
    ss_res = float(np.sum((residual_targets - predictions) ** 2))
    ss_tot = float(np.sum((residual_targets - residual_targets.mean()) ** 2))
    if ss_tot < _MIN_VARIANCE:
        return 0.0
    return 1.0 - ss_res / ss_tot


def g_struct_sha256(graph: nx.Graph) -> str:
    """Stable identity of one simple structural graph."""
    rows = [f"{min(str(u), str(v))}\t{max(str(u), str(v))}\n" for u, v in graph.edges()]
    return hashlib.sha256("".join(sorted(rows)).encode()).hexdigest()


def select_probe_pairs(graph: nx.Graph, *, limit: int = _E2E_PAIR_LIMIT) -> list[tuple[str, str]]:
    """Select the registered hash-smallest non-self E_msg pairs."""
    if limit <= 0:
        raise ValueError("probe pair limit must be positive")
    pairs = {
        (min(str(node_u), str(node_v)), max(str(node_u), str(node_v)))
        for node_u, node_v in graph.edges()
        if node_u != node_v
    }
    return sorted(
        pairs,
        key=lambda pair: (
            hashlib.sha256(f"{pair[0]}|{pair[1]}".encode()).digest(),
            pair,
        ),
    )[:limit]


def write_e2e_probe_artifact(
    path: Path,
    *,
    metadata: Mapping[str, object],
    node_ids: Sequence[str],
    states: NDArray[np.float32],
    targets: Mapping[str, NDArray[np.float64]],
    pair_ids: Sequence[tuple[str, str]],
    pair_states: NDArray[np.float32],
    pi_consistency_v1: NDArray[np.float64],
    pi_consistency_v2: NDArray[np.float64],
    slot_recall: NDArray[np.float64],
    shared_neighbor_count: NDArray[np.float64],
    dispersion: Mapping[str, NDArray[np.float64]],
) -> None:
    """Write one validated provenance-bound rev-3.1 E2E probe artifact."""
    n_nodes = len(node_ids)
    n_pairs = len(pair_ids)
    required_targets = {"degree", "ego_density", "clustering"}
    if set(targets) != required_targets:
        raise ValueError(f"probe targets must be exactly {sorted(required_targets)}")
    if states.ndim != 2 or states.shape[0] != n_nodes:
        raise ValueError("probe states must have shape (n_nodes, d_state)")
    if pair_states.ndim != 2 or pair_states.shape[0] != n_pairs:
        raise ValueError("probe pair states must have shape (n_pairs, d_state)")
    if any(np.asarray(targets[name]).shape != (n_nodes,) for name in required_targets):
        raise ValueError("every probe target must have shape (n_nodes,)")
    pair_arrays: dict[str, NDArray[np.float64]] = {
        "Pi consistency v1": np.asarray(pi_consistency_v1, dtype=np.float64),
        "Pi consistency v2": np.asarray(pi_consistency_v2, dtype=np.float64),
        "shared-neighbor count": np.asarray(shared_neighbor_count, dtype=np.float64),
        **{
            name: np.asarray(values, dtype=np.float64)
            for name, values in dispersion.items()
        },
    }
    if set(dispersion) != set(_DISPERSION_NAMES):
        raise ValueError(f"probe dispersion must be exactly {sorted(_DISPERSION_NAMES)}")
    if any(values.shape != (n_pairs,) for values in pair_arrays.values()):
        raise ValueError("every pair probe must align with probe pair identities")
    if slot_recall.ndim != 1 or len(slot_recall) > n_nodes:
        raise ValueError("slot recall must be one-dimensional and cannot exceed node count")
    if (
        not np.isfinite(states).all()
        or not np.isfinite(pair_states).all()
        or not np.isfinite(slot_recall).all()
        or any(not np.isfinite(values).all() for values in pair_arrays.values())
    ):
        raise ValueError("probe artifact contains non-finite values")
    for values in (pi_consistency_v1, pi_consistency_v2, slot_recall):
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("Pi consistency and slot recall values must lie in [0, 1]")
    if np.any(shared_neighbor_count < 0.0):
        raise ValueError("shared-neighbor counts must be nonnegative")
    if metadata.get("scope") not in E2E_PROBE_SCOPES:
        raise ValueError(f"probe metadata scope must be one of {E2E_PROBE_SCOPES}")
    n_ground_value = metadata.get("n_ground")
    if not isinstance(n_ground_value, int) or n_ground_value <= 0:
        raise ValueError("probe metadata n_ground must be a positive integer")
    payload_meta = {**metadata, "format": E2E_PROBE_FORMAT}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        meta=np.array(json.dumps(payload_meta, sort_keys=True)),
        node_ids=np.asarray(node_ids, dtype=np.str_),
        states=np.asarray(states, dtype=np.float32),
        degree=np.asarray(targets["degree"], dtype=np.float64),
        ego_density=np.asarray(targets["ego_density"], dtype=np.float64),
        clustering=np.asarray(targets["clustering"], dtype=np.float64),
        pair_u=np.asarray([pair[0] for pair in pair_ids], dtype=np.str_),
        pair_v=np.asarray([pair[1] for pair in pair_ids], dtype=np.str_),
        pair_states=np.asarray(pair_states, dtype=np.float32),
        pi_shared_neighbor_consistency=np.asarray(pi_consistency_v1, dtype=np.float64),
        pi_consistency_v2=np.asarray(pi_consistency_v2, dtype=np.float64),
        slot_recall_at_n_ground=np.asarray(slot_recall, dtype=np.float64),
        shared_neighbor_count=np.asarray(shared_neighbor_count, dtype=np.float64),
        pi_slot_std=np.asarray(dispersion["pi_slot_std"], dtype=np.float64),
        h_pairwise_cosine_mean=np.asarray(
            dispersion["h_pairwise_cosine_mean"], dtype=np.float64
        ),
        adj_offdiag_std=np.asarray(dispersion["adj_offdiag_std"], dtype=np.float64),
        plan_row_entropy=np.asarray(dispersion["plan_row_entropy"], dtype=np.float64),
    )


def evaluate_e2e_probe_artifact(
    path: Path,
    *,
    graph: nx.Graph,
    train_nodes: Sequence[str],
    expected_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Validate registered identities/targets and compute all nonbinding probe rows."""
    required_arrays = {
        "meta",
        "node_ids",
        "states",
        "degree",
        "ego_density",
        "clustering",
        "pair_u",
        "pair_v",
        "pair_states",
        "pi_shared_neighbor_consistency",
        "pi_consistency_v2",
        "slot_recall_at_n_ground",
        "shared_neighbor_count",
        *_DISPERSION_NAMES,
    }
    with np.load(path, allow_pickle=False) as archive:
        if "meta" not in archive.files:
            raise ValueError("E2E probe artifact is missing metadata")
        metadata = cast(dict[str, object], json.loads(str(archive["meta"].item())))
        actual_format = metadata.get("format")
        if actual_format != E2E_PROBE_FORMAT:
            if actual_format == _E2E_PROBE_V1_FORMAT:
                raise ValueError(
                    f"E2E probe artifact format {_E2E_PROBE_V1_FORMAT!r} is not supported; "
                    f"expected {E2E_PROBE_FORMAT!r}"
                )
            raise ValueError(
                f"E2E probe artifact format {actual_format!r} is not supported; "
                f"expected {E2E_PROBE_FORMAT!r}"
            )
        if set(archive.files) != required_arrays:
            raise ValueError(
                f"E2E probe arrays must be exactly {sorted(required_arrays)}, "
                f"got {sorted(archive.files)}"
            )
        node_ids = archive["node_ids"].astype(str).tolist()
        states = np.asarray(archive["states"], dtype=np.float64)
        pair_states = np.asarray(archive["pair_states"], dtype=np.float64)
        stored_targets = {
            name: np.asarray(archive[name], dtype=np.float64)
            for name in ("degree", "ego_density", "clustering")
        }
        pair_ids = list(
            zip(
                archive["pair_u"].astype(str).tolist(),
                archive["pair_v"].astype(str).tolist(),
                strict=True,
            )
        )
        pi_v1 = np.asarray(archive["pi_shared_neighbor_consistency"], dtype=np.float64)
        pi_v2 = np.asarray(archive["pi_consistency_v2"], dtype=np.float64)
        slot_recall = np.asarray(archive["slot_recall_at_n_ground"], dtype=np.float64)
        shared_neighbor_count = np.asarray(archive["shared_neighbor_count"], dtype=np.float64)
        dispersion = {
            name: np.asarray(archive[name], dtype=np.float64) for name in _DISPERSION_NAMES
        }
    if metadata.get("scope") not in E2E_PROBE_SCOPES:
        raise ValueError("E2E probe metadata has an unsupported structural scope")
    n_ground_value = metadata.get("n_ground")
    if not isinstance(n_ground_value, int) or n_ground_value <= 0:
        raise ValueError("E2E probe metadata has an invalid n_ground")
    registered_n_ground = expected_metadata.get("n_ground")
    if not isinstance(registered_n_ground, int) or registered_n_ground <= 0:
        raise ValueError("E2E probe validation requires the registered arm's positive n_ground")
    if n_ground_value != registered_n_ground:
        raise ValueError(
            "E2E probe n_ground mismatch: "
            f"probe={n_ground_value}, registered={registered_n_ground}"
        )
    for key, expected in expected_metadata.items():
        if key == "n_ground":
            continue
        if metadata.get(key) != expected:
            raise ValueError(
                f"E2E probe metadata {key!r} mismatch: {metadata.get(key)!r} != {expected!r}"
            )
    expected_nodes = sorted(train_nodes)
    if node_ids != expected_nodes:
        raise ValueError("E2E probe node identities do not match sorted operative train nodes")
    expected_pairs = select_probe_pairs(graph)
    if pair_ids != expected_pairs:
        raise ValueError("E2E probe pair identities do not match registered E_msg selection")
    if metadata.get("g_struct_sha256") != g_struct_sha256(graph):
        raise ValueError("E2E probe G_struct identity mismatch")
    expected_targets = probe_targets(graph, expected_nodes)
    for name in stored_targets:
        if not np.array_equal(stored_targets[name], expected_targets[name]):
            raise ValueError(f"E2E probe target {name!r} does not match G_struct")
    expected_shared = np.asarray(
        [
            len(set(graph.neighbors(node_u)) & set(graph.neighbors(node_v)))
            for node_u, node_v in expected_pairs
        ],
        dtype=np.float64,
    )
    if not np.array_equal(shared_neighbor_count, expected_shared):
        raise ValueError("E2E probe shared-neighbor counts do not match G_struct")
    if states.ndim != 2 or states.shape[0] != len(expected_nodes):
        raise ValueError("E2E probe state shape does not match node identities")
    if pair_states.ndim != 2 or pair_states.shape[0] != len(expected_pairs):
        raise ValueError("E2E pair-state shape does not match pair identities")
    pair_arrays = {
        "Pi consistency v1": pi_v1,
        "Pi consistency v2": pi_v2,
        "shared-neighbor count": shared_neighbor_count,
        **dispersion,
    }
    n_recall_nodes = sum(graph.degree(node) > 0 for node in expected_nodes)
    if slot_recall.shape != (n_recall_nodes,):
        raise ValueError("E2E slot-recall shape does not match non-isolated node identities")
    if any(values.shape != (len(expected_pairs),) for values in pair_arrays.values()):
        raise ValueError("E2E pair-probe shape does not match pair identities")
    finite_arrays = [states, pair_states, slot_recall, *pair_arrays.values()]
    if any(not np.isfinite(values).all() for values in finite_arrays):
        raise ValueError("E2E probe artifact contains non-finite values")
    if any(np.any((values < 0.0) | (values > 1.0)) for values in (pi_v1, pi_v2, slot_recall)):
        raise ValueError("E2E Pi consistency and slot recall values must lie in [0, 1]")
    degree = stored_targets["degree"]

    def summary(values: NDArray[np.float64]) -> dict[str, float | int]:
        return {
            "mean": float(np.mean(values)) if len(values) else 0.0,
            "std": float(np.std(values)) if len(values) else 0.0,
            "nonzero_fraction": float(np.mean(values > 0.0)) if len(values) else 0.0,
            "n": len(values),
        }

    return {
        "metadata": metadata,
        "linear_probe_r2": {
            name: linear_probe_r2(states, stored_targets[name], seed=0)
            for name in ("degree", "ego_density", "clustering")
        },
        "degree_partialled_r2": {
            name: degree_partialled_r2(states, stored_targets[name], degree, seed=0)
            for name in ("ego_density", "clustering")
        },
        "shared_neighbor_count_r2": linear_probe_r2(
            pair_states,
            shared_neighbor_count,
            seed=0,
        ),
        "slot_recall_at_n_ground": {
            **summary(slot_recall),
            "n_ground": n_ground_value,
            "n_nodes": len(slot_recall),
        },
        "pi_shared_neighbor_consistency": {
            **summary(pi_v1),
            "n_pairs": len(pi_v1),
        },
        "pi_consistency_v2": {
            **summary(pi_v2),
            "n_pairs": len(pi_v2),
        },
        "dispersion": {name: summary(values) for name, values in dispersion.items()},
    }


@dataclass(frozen=True)
class _ProbeBundle:
    """Frozen-store view read by probe batches (duck-types ``EgoStitchData``)."""

    node_index: dict[str, int]
    f0: torch.Tensor
    grounding_index: NDArray[np.int64]
    train_pos: dict[str, int]


def _probe_batch(
    data: object,
    table: object,
    token_index: Mapping[str, int],
    endpoints_a: Sequence[str],
    endpoints_b: Sequence[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Assemble one production probe batch using the worker's frozen stores."""
    from src import train_egostitch as te
    from src.data.packed_features import PackedFeatureTable

    bundle = cast("te.EgoStitchData", data)
    packed = cast(PackedFeatureTable, table)
    batch = te._gather_token_streams(packed, token_index, endpoints_a, endpoints_b)
    rows_a = torch.tensor([bundle.node_index[node] for node in endpoints_a], dtype=torch.long)
    rows_b = torch.tensor([bundle.node_index[node] for node in endpoints_b], dtype=torch.long)
    ground_a = torch.from_numpy(
        bundle.grounding_index[[bundle.train_pos[node] for node in endpoints_a]]
    )
    ground_b = torch.from_numpy(
        bundle.grounding_index[[bundle.train_pos[node] for node in endpoints_b]]
    )
    batch.update(
        {
            "emb_a": batch["emb_a"].float(),
            "emb_b": batch["emb_b"].float(),
            "x_a": bundle.f0[rows_a],
            "x_b": bundle.f0[rows_b],
            "ground_a": bundle.f0[ground_a],
            "ground_b": bundle.f0[ground_b],
            "ground_id_a": ground_a,
            "ground_id_b": ground_b,
            "is_self": torch.tensor(
                [a == b for a, b in zip(endpoints_a, endpoints_b, strict=True)],
                dtype=torch.bool,
            ),
        }
    )
    return {name: value.to(device) for name, value in batch.items()}


def produce_e2e_probe_artifact(
    *,
    checkpoint_path: Path,
    run_metadata_path: Path,
    data_root: Path,
    strategy: str,
    output_path: Path,
    scope: E2EProbeScope,
) -> None:
    """Produce the full-checkpoint STE/Pi evidence artifact.

    ``scope`` is deliberately required and ``formal_train`` is its only value:
    the full operative train-side ``G_struct``.
    """
    from src import train_egostitch as te
    from src.data.ego_targets import EgoTargetBuilder, EgoTargets
    from src.data.packed_features import PackedFeatureTable
    from src.model.egostitch.composite import EgoStitchModel
    from src.model.egostitch.config import E2EConfig
    from src.model.egostitch.generator.assemble import match_slots
    from src.model.egostitch.generator.losses import alignment_teacher_cells
    from src.train_b0 import _state_digest

    if scope not in E2E_PROBE_SCOPES:
        raise ValueError(f"unsupported E2E probe scope {scope!r}; expected {E2E_PROBE_SCOPES}")
    run_metadata = cast(
        dict[str, object], json.loads(run_metadata_path.read_text(encoding="utf-8"))
    )
    config_path = Path(str(run_metadata.get("config_path")))
    cfg = te.load_config(config_path)
    if cfg.model.family != "egostitch_e2e":
        raise ValueError("E2E probe producer requires model family egostitch_e2e")
    formal_model_cfg = E2EConfig.from_mapping(cfg.model.config)
    if formal_model_cfg.permanent_null != "none" or formal_model_cfg.p_topo != 0.15:
        raise ValueError(
            "E2E probe producer requires full-arm permanent_null=none and p_topo=0.15"
        )
    if cfg.data.root.resolve() != data_root.resolve() or cfg.data.strategy != strategy:
        raise ValueError("probe CLI data root/strategy do not match the formal config")
    # Multi-seed runs are permitted, so the model seed is only required to be a
    # real seed and is recorded verbatim. ``data.partition_seed`` is not the
    # model seed: it selects G_struct and the whole pair universe this probe is
    # validated against, so it is pinned to the config, not relaxed.
    run_seed = run_metadata.get("seed")
    run_partition_seed = run_metadata.get("partition_seed")
    if isinstance(run_seed, bool) or not isinstance(run_seed, int) or run_seed < 0:
        raise ValueError("E2E probe producer requires a nonnegative integer run seed")
    if (
        isinstance(run_partition_seed, bool)
        or not isinstance(run_partition_seed, int)
        or run_partition_seed != cfg.data.partition_seed
    ):
        raise ValueError(
            "E2E probe run metadata partition_seed does not match the config "
            f"({cfg.data.partition_seed})"
        )
    payload = cast(dict[str, object], torch.load(checkpoint_path, map_location="cpu"))
    state = cast(dict[str, torch.Tensor], payload["model_state"])
    checkpoint_id = _state_digest(state)[:16]
    # The rev-3.1 checkpoint-config backfills were deleted with the content path
    # (design 2026-08-02 §11); the checkpoint carries its own complete config.
    checkpoint_model_cfg = cast(dict[str, object], payload["model_config"])
    model = EgoStitchModel(E2EConfig.from_mapping(checkpoint_model_cfg))
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    if cfg.data.pack_dir is None:
        raise ValueError("E2E probe producer requires data.pack_dir")
    table = PackedFeatureTable.from_pack(cfg.data.pack_dir, torch.device("cpu"))
    token_index = table.manifest.node_index()

    # The formal artifact is pinned to all operative train nodes over the full
    # train-side E_msg. Grounding pools stay universe-scoped (spec Sec 13.12):
    # V_fit and the single validation holdout V_hold are hashed separately and
    # one cache may never serve the other.
    from src.data.features import FeatureStore, build_f0_matrix
    from src.data.grounding import build_grounding_pool
    from src.data.internal_holdout import derive_internal_holdout
    from src.data.partition import build_g_struct, derive_partition

    strategy_dir = cfg.data.root / te._BENCHMARK_SUBDIR / cfg.data.strategy
    formal_nodes, train_positives = _load_train_side_probe_inputs(
        strategy_dir,
        expected_missing_features=cfg.data.expected_missing_features,
    )
    partition = derive_partition(
        train_positives, seed=cfg.data.partition_seed, msg_fraction=cfg.data.msg_fraction
    )
    holdout = derive_internal_holdout(formal_nodes, partition.e_msg, partition.e_sup)
    nodes = list(formal_nodes)
    graph = build_g_struct(nodes, partition.e_msg)
    missing_tokens = [node for node in nodes if node not in token_index]
    if missing_tokens:
        raise ValueError(f"token pack is missing {len(missing_tokens)} probe nodes")
    store = FeatureStore(cfg.data.root / te._FEATURES_SUBDIR)
    cache_dir = output_path.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix, node_index = build_f0_matrix(
        store,
        nodes,
        cache_path=cache_dir / "probe_f0.pt",
        allow_cache_subset=True,
    )
    # Sourced from the loaded checkpoint's own generator_cfg (spec Sec 14.4.4:
    # n_ground is per-arm, not the pinned EgoStitchConfig() spec default).
    n_ground = model.generator_cfg.n_ground
    matrix_np = matrix.numpy()
    grounding_rows: dict[str, list[int]] = {}
    grounding_pool: dict[str, list[str]] = {}
    role_universes: tuple[tuple[list[str], str, str], ...] = (
        (sorted(holdout.v_fit), "probe_grounding_fit.npz", "V_fit"),
        (sorted(holdout.v_hold), "probe_grounding_hold.npz", "V_hold"),
    )
    for role_nodes, cache_name, role_universe in role_universes:
        role_rows = np.asarray(
            matrix_np[[node_index[node] for node in role_nodes]], dtype=np.float32
        )
        pool = build_grounding_pool(
            role_rows,
            role_nodes,
            n_ground=n_ground,
            role_universe=role_universe,
            cache_path=cache_dir / cache_name,
        )
        for node in role_nodes:
            grounding_pool[node] = list(pool[node])
            grounding_rows[node] = [node_index[neighbor] for neighbor in pool[node]]
    data = _ProbeBundle(
        node_index=node_index,
        f0=matrix,
        grounding_index=np.asarray([grounding_rows[node] for node in nodes], dtype=np.int64),
        train_pos={node: position for position, node in enumerate(nodes)},
    )
    batch_size = max(1, cfg.data.edge_batch)
    state_rows: list[NDArray[np.float32]] = []
    selected_slots: dict[str, list[str]] = {}
    inverse_index = {row: node for node, row in node_index.items()}
    with torch.inference_mode():
        for start in range(0, len(nodes), batch_size):
            batch_nodes = nodes[start : start + batch_size]
            batch = _probe_batch(data, table, token_index, batch_nodes, batch_nodes, device)
            # `_pair_node_states` is private and does not survive the composite
            # rewrite (three-component refactor design §5). Every row here is
            # a self-pair (endpoints_a == endpoints_b), so the public
            # `encode_node_state` need only run once per batch, mirroring
            # `_pair_node_states`'s own all-self short-circuit
            # (`state_a, state_a, is_self`).
            state_a = model.encode_node_state(
                batch["emb_a"],
                batch["len_a"],
                batch["x_a"],
                batch["ground_a"],
                batch.get("ground_id_a"),
            )
            context = model.build_pair_context_from_states(
                state_a,
                state_a,
                batch["is_self"],
                need_topo=True,
            )
            assert context.topo_ab is not None
            assert state_a.ground_ids is not None
            state_rows.append(context.topo_ab.mean(dim=1).float().cpu().numpy())
            selected_ids = torch.gather(
                state_a.ground_ids,
                1,
                state_a.slots.pointer.argmax(dim=-1),
            )
            active = state_a.slots.gate > 0.5
            for row, node in enumerate(batch_nodes):
                selected_slots[node] = [
                    inverse_index[int(identity)]
                    for identity in selected_ids[row, active[row]].cpu().tolist()
                ]

    target_builder = EgoTargetBuilder(
        graph,
        matrix_np,
        node_index,
        grounding_pool,
        slots=model.generator_cfg.slots,
    )
    checkpoint_epoch = cast(int, payload["epoch"])

    def batch_targets(batch_nodes: Sequence[str]) -> EgoTargets:
        """Build the exact node/epoch-keyed matching targets used by ``L_align``."""
        rows = [
            target_builder.build(
                [node],
                te._edge_target_rng(node, seed=0, epoch=checkpoint_epoch),
            )
            for node in batch_nodes
        ]
        return EgoTargets(
            features=torch.cat([row.features for row in rows], dim=0).to(device),
            mult=torch.cat([row.mult for row in rows], dim=0).to(device),
            adj=torch.cat([row.adj for row in rows], dim=0).to(device),
            mask=torch.cat([row.mask for row in rows], dim=0).to(device),
            degree=torch.cat([row.degree for row in rows], dim=0).to(device),
            in_pool=torch.cat([row.in_pool for row in rows], dim=0).to(device),
            pool_index=torch.cat([row.pool_index for row in rows], dim=0).to(device),
            node_index=torch.cat([row.node_index for row in rows], dim=0).to(device),
            ego_stats=torch.cat([row.ego_stats for row in rows], dim=0).to(device),
        )

    pairs = select_probe_pairs(graph)
    consistency_v1: list[float] = []
    consistency_v2: list[float] = []
    pair_state_rows: list[NDArray[np.float32]] = []
    shared_neighbor_counts: list[float] = []
    dispersion_rows: dict[str, list[float]] = {name: [] for name in _DISPERSION_NAMES}
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            endpoints_a = [pair[0] for pair in batch_pairs]
            endpoints_b = [pair[1] for pair in batch_pairs]
            batch = _probe_batch(
                data,
                table,
                token_index,
                endpoints_a,
                endpoints_b,
                device,
            )
            # No row here is a self-pair (`select_probe_pairs` excludes
            # `node_u == node_v`), so this is exactly `_pair_node_states`'s
            # own all-non-self branch: independently encode both endpoints
            # via the public surface.
            state_a = model.encode_node_state(
                batch["emb_a"],
                batch["len_a"],
                batch["x_a"],
                batch["ground_a"],
                batch.get("ground_id_a"),
            )
            state_b = model.encode_node_state(
                batch["emb_b"],
                batch["len_b"],
                batch["x_b"],
                batch["ground_b"],
                batch.get("ground_id_b"),
            )
            context = model.build_pair_context_from_states(
                state_a, state_b, batch["is_self"], need_topo=True
            )
            assert context.plan is not None
            assert context.topo_ab is not None
            assert state_a.ground_ids is not None and state_b.ground_ids is not None
            pair_state_rows.append(context.topo_ab.float().mean(dim=1).cpu().numpy())
            targets_a = batch_targets(endpoints_a)
            targets_b = batch_targets(endpoints_b)
            assignment_a = match_slots(
                state_a.slots,
                target_proj=model.generator.stage1.project_features(targets_a.features).detach(),
                target_mult=targets_a.mult,
                target_adj=targets_a.adj,
                target_mask=targets_a.mask,
            )
            assignment_b = match_slots(
                state_b.slots,
                target_proj=model.generator.stage1.project_features(targets_b.features).detach(),
                target_mult=targets_b.mult,
                target_adj=targets_b.adj,
                target_mask=targets_b.mask,
            )
            teacher_cells = alignment_teacher_cells(
                assignment_a,
                assignment_b,
                target_ids_a=targets_a.node_index,
                target_ids_b=targets_b.node_index,
                slots=model.generator_cfg.slots,
            )
            batch_v2 = pi_alignment_consistency_v2(context.plan, teacher_cells)
            dispersion_a = te._e2e_dispersion_rows(
                state_a.slots.pi,
                state_a.slots.h,
                state_a.slots.adj,
                context.plan,
            )
            dispersion_b = te._e2e_dispersion_rows(
                state_b.slots.pi,
                state_b.slots.h,
                state_b.slots.adj,
                context.plan,
            )
            for name in _DISPERSION_NAMES:
                values = (
                    0.5 * (dispersion_a[name] + dispersion_b[name])
                    if name != "plan_row_entropy"
                    else dispersion_a[name]
                )
                dispersion_rows[name].extend(values.float().cpu().tolist())
            selected_a = torch.gather(state_a.ground_ids, 1, state_a.slots.pointer.argmax(dim=-1))
            selected_b = torch.gather(state_b.ground_ids, 1, state_b.slots.pointer.argmax(dim=-1))
            for row, (node_u, node_v) in enumerate(batch_pairs):
                common = set(graph.neighbors(node_u)) & set(graph.neighbors(node_v))
                common_rows = {data.node_index[node] for node in common}
                ids_a = selected_a[row]
                ids_b = selected_b[row]
                equal = ids_a[:, None] == ids_b[None, :]
                real_common = torch.zeros_like(equal)
                for slot_a in range(equal.size(0)):
                    identity = int(ids_a[slot_a].item())
                    if identity in common_rows:
                        real_common[slot_a] = equal[slot_a]
                v1 = pi_grounding_chain_consistency_v1(
                    context.plan[row : row + 1],
                    (equal & real_common)[None],
                    gate_a=state_a.slots.gate[row : row + 1],
                    gate_b=state_b.slots.gate[row : row + 1],
                )
                consistency_v1.append(float(v1[0]))
                consistency_v2.append(float(batch_v2[row]))
                shared_neighbor_counts.append(float(len(common)))
    targets = probe_targets(graph, nodes)
    write_e2e_probe_artifact(
        output_path,
        metadata={
            "checkpoint_id": checkpoint_id,
            "seed": run_seed,
            "partition_seed": run_partition_seed,
            "strategy": strategy,
            "g_struct_sha256": g_struct_sha256(graph),
            "scope": scope,
            "n_ground": n_ground,
        },
        node_ids=nodes,
        states=np.concatenate(state_rows, axis=0),
        targets={name: targets[name] for name in ("degree", "ego_density", "clustering")},
        pair_ids=pairs,
        pair_states=np.concatenate(pair_state_rows, axis=0),
        pi_consistency_v1=np.asarray(consistency_v1, dtype=np.float64),
        pi_consistency_v2=np.asarray(consistency_v2, dtype=np.float64),
        slot_recall=slot_recall_at_n_ground(
            graph,
            nodes,
            grounding_pool,
            selected_slots,
        ),
        shared_neighbor_count=np.asarray(shared_neighbor_counts, dtype=np.float64),
        dispersion={
            name: np.asarray(values, dtype=np.float64)
            for name, values in dispersion_rows.items()
        },
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the probe producer CLI."""
    parser = argparse.ArgumentParser(prog="python -m src.experiments.probes")
    subparsers = parser.add_subparsers(dest="command", required=True)
    produce = subparsers.add_parser("produce-e2e")
    produce.add_argument("--checkpoint", type=Path, required=True)
    produce.add_argument("--run-metadata", type=Path, required=True)
    produce.add_argument("--data-root", type=Path, required=True)
    produce.add_argument("--strategy", required=True)
    produce.add_argument("--output", type=Path, required=True)
    produce.add_argument("--scope", choices=E2E_PROBE_SCOPES, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    if args.command == "produce-e2e":
        produce_e2e_probe_artifact(
            checkpoint_path=args.checkpoint,
            run_metadata_path=args.run_metadata,
            data_root=args.data_root,
            strategy=args.strategy,
            output_path=args.output,
            scope=args.scope,
        )


__all__ = [
    "E2E_PROBE_FORMAT",
    "E2E_PROBE_SCOPES",
    "degree_partialled_r2",
    "evaluate_e2e_probe_artifact",
    "g_struct_sha256",
    "linear_probe_r2",
    "pi_alignment_consistency_v2",
    "pi_grounding_chain_consistency_v1",
    "probe_targets",
    "select_probe_pairs",
    "slot_recall_at_n_ground",
    "write_e2e_probe_artifact",
]


if __name__ == "__main__":
    main()

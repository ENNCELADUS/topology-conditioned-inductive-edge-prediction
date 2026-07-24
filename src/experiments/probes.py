"""Closed-form ridge-regression representation probes (spec Sec 14.3(3)).

Diagnostics-only linear probes over frozen encoder states (STE token states,
in the registered representation-probe protocol): out-of-fold ``R^2`` from a
closed-form ridge fit, implemented in plain numpy (no sklearn dependency, per
the pinned protocol). ``degree_partialled_r2`` additionally residualizes both
``states`` and ``targets`` against node degree before probing, isolating any
predictive signal beyond what a trivial degree confound would already give
away for free (registration ``diagnostics_nonbinding``: "linear probes on
frozen STE token states: R2 to degree / ego-density / clustering (ridge
lambda 1e-3, 5-fold), plus degree-partialled variants and Pi-consistency").

Both probes are diagnostics only — they never gate the Stage-1 verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import torch
from numpy.typing import NDArray

_RIDGE_LAMBDA = 1e-3
_N_FOLDS = 5
_MIN_VARIANCE = 1e-10
_E2E_PROBE_FORMAT = "egostitch_e2e_probe_v1"
_E2E_PAIR_LIMIT = 4096


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
    pi_consistency: NDArray[np.float64],
) -> None:
    """Write one validated provenance-bound E2E probe artifact."""
    n_nodes = len(node_ids)
    required_targets = {"degree", "ego_density", "clustering"}
    if set(targets) != required_targets:
        raise ValueError(f"probe targets must be exactly {sorted(required_targets)}")
    if states.ndim != 2 or states.shape[0] != n_nodes:
        raise ValueError("probe states must have shape (n_nodes, d_state)")
    if any(np.asarray(targets[name]).shape != (n_nodes,) for name in required_targets):
        raise ValueError("every probe target must have shape (n_nodes,)")
    if len(pair_ids) != len(pi_consistency):
        raise ValueError("probe pair identities and Pi consistency must align")
    if not np.isfinite(states).all() or not np.isfinite(pi_consistency).all():
        raise ValueError("probe artifact contains non-finite values")
    if np.any((pi_consistency < 0.0) | (pi_consistency > 1.0)):
        raise ValueError("Pi consistency values must lie in [0, 1]")
    payload_meta = {**metadata, "format": _E2E_PROBE_FORMAT}
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
        pi_shared_neighbor_consistency=np.asarray(pi_consistency, dtype=np.float64),
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
        "pi_shared_neighbor_consistency",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required_arrays:
            raise ValueError(
                f"E2E probe arrays must be exactly {sorted(required_arrays)}, "
                f"got {sorted(archive.files)}"
            )
        metadata = cast(dict[str, object], json.loads(str(archive["meta"].item())))
        node_ids = archive["node_ids"].astype(str).tolist()
        states = np.asarray(archive["states"], dtype=np.float64)
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
        pi_values = np.asarray(archive["pi_shared_neighbor_consistency"], dtype=np.float64)
    if metadata.get("format") != _E2E_PROBE_FORMAT:
        raise ValueError("unsupported E2E probe artifact format")
    for key, expected in expected_metadata.items():
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
    if states.ndim != 2 or states.shape[0] != len(expected_nodes):
        raise ValueError("E2E probe state shape does not match node identities")
    if pi_values.shape != (len(expected_pairs),):
        raise ValueError("E2E probe Pi consistency shape does not match pair identities")
    if not np.isfinite(states).all() or not np.isfinite(pi_values).all():
        raise ValueError("E2E probe artifact contains non-finite values")
    degree = stored_targets["degree"]
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
        "pi_shared_neighbor_consistency": {
            "mean": float(np.mean(pi_values)) if len(pi_values) else 0.0,
            "std": float(np.std(pi_values)) if len(pi_values) else 0.0,
            "nonzero_fraction": float(np.mean(pi_values > 0.0)) if len(pi_values) else 0.0,
            "n_pairs": len(pi_values),
        },
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
    preregistration_path: Path,
    data_root: Path,
    strategy: str,
    output_path: Path,
) -> None:
    """Produce the registered full-checkpoint STE/Pi evidence artifact."""
    from src import train_egostitch as te
    from src.data.packed_features import PackedFeatureTable
    from src.model.egostitch.config import E2EConfig
    from src.model.egostitch.e2e_model import EgoStitchE2E
    from src.train_b0 import _state_digest

    run_metadata = cast(
        dict[str, object], json.loads(run_metadata_path.read_text(encoding="utf-8"))
    )
    preregistration_bytes = preregistration_path.read_bytes()
    registration_sha = hashlib.sha256(preregistration_bytes).hexdigest()
    registration_payload = cast(
        dict[str, object], json.loads(preregistration_bytes.decode("utf-8"))
    )
    probe_registration = cast(
        Mapping[str, object] | None, registration_payload.get("probe_artifact")
    )
    if registration_payload.get("status") != "BINDING":
        raise ValueError("E2E probe producer requires a BINDING preregistration")
    if probe_registration is None or probe_registration.get("format") != _E2E_PROBE_FORMAT:
        raise ValueError("preregistration does not bind the E2E probe artifact format")
    if probe_registration.get("source_arm") != "full":
        raise ValueError("preregistration does not bind the E2E probe source to the full arm")
    arms = registration_payload.get("arms")
    if not isinstance(arms, Mapping):
        raise ValueError("preregistration does not define formal arms")
    full_arm = arms.get("full")
    if not isinstance(full_arm, Mapping) or not isinstance(full_arm.get("training"), str):
        raise ValueError("preregistration does not bind arms.full.training")
    registered_config = Path(cast(str, full_arm["training"]))
    if not registered_config.is_absolute():
        registered_config = preregistration_path.resolve().parents[2] / registered_config
    expected_output = Path(str(probe_registration.get("expected_path")))
    if not expected_output.is_absolute():
        expected_output = preregistration_path.resolve().parents[2] / expected_output
    if output_path.resolve() != expected_output.resolve():
        raise ValueError(
            f"probe output path does not match registration: {output_path} != {expected_output}"
        )
    if run_metadata.get("preregistration_sha256") != registration_sha:
        raise ValueError("probe run metadata does not match preregistration SHA-256")
    if (
        run_metadata.get("run_kind") != "formal"
        or run_metadata.get("status") != "complete"
        or run_metadata.get("formal_artifacts_published") is not True
        or run_metadata.get("permanent_null") != "none"
    ):
        raise ValueError("E2E probe producer requires the completed formal full arm")
    if run_metadata.get("seed") != 0 or run_metadata.get("partition_seed") != 0:
        raise ValueError("E2E probe producer requires Seed 0 and partition Seed 0")
    config_path = Path(str(run_metadata.get("config_path")))
    if config_path.resolve() != registered_config.resolve():
        raise ValueError("E2E probe producer requires the registered full-arm config path")
    cfg = te.load_config(config_path)
    if cfg.model.family != "egostitch_e2e":
        raise ValueError("E2E probe producer requires model family egostitch_e2e")
    formal_model_cfg = E2EConfig.from_mapping(cfg.model.config)
    if (
        formal_model_cfg.permanent_null != "none"
        or formal_model_cfg.p_topo != 0.15
        or formal_model_cfg.p_cont != 0.15
    ):
        raise ValueError(
            "E2E probe producer requires full-arm permanent_null=none and p_topo=p_cont=0.15"
        )
    if cfg.data.root.resolve() != data_root.resolve() or cfg.data.strategy != strategy:
        raise ValueError("probe CLI data root/strategy do not match the formal config")
    config_hash = te._config_hash(cfg)
    if run_metadata.get("config_hash") != config_hash:
        raise ValueError("probe run metadata config hash does not match the formal config")
    payload = cast(dict[str, object], torch.load(checkpoint_path, map_location="cpu"))
    state = cast(dict[str, torch.Tensor], payload["model_state"])
    checkpoint_id = _state_digest(state)[:16]
    if run_metadata.get("checkpoint_id") != checkpoint_id:
        raise ValueError("probe checkpoint does not match selected formal checkpoint_id")
    model = EgoStitchE2E(E2EConfig.from_mapping(cast(dict[str, object], payload["model_config"])))
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    if cfg.data.pack_dir is None:
        raise ValueError("E2E probe producer requires data.pack_dir")
    table = PackedFeatureTable.from_pack(cfg.data.pack_dir, torch.device("cpu"))
    token_index = table.manifest.node_index()

    # Registered probe identities (registration `probe_artifact`): all operative
    # train nodes over the full-E_msg `G_struct`, matching the gate's own
    # reconstruction — not the worker's internal-holdout training view. Each
    # node grounds in its spec §13.12 role universe (V_fit / V_qual / V_select).
    from src.data.features import FeatureStore, build_f0_matrix
    from src.data.grounding import build_grounding_pool
    from src.data.internal_holdout import derive_internal_holdout
    from src.data.partition import build_g_struct, derive_partition
    from src.model.egostitch.config import EgoStitchConfig

    benchmark = te._load_benchmark_for(cfg)
    operative = sorted(set(benchmark.graph.nodes()) - set(cfg.data.expected_missing_features))
    nodes = sorted(set(benchmark.split.train_nodes) & set(operative))
    train_positives = [
        pair
        for pair, label in zip(
            benchmark.split.train_pairs.pairs,
            benchmark.split.train_pairs.labels,
            strict=True,
        )
        if label == 1
    ]
    partition = derive_partition(
        train_positives, seed=cfg.data.partition_seed, msg_fraction=cfg.data.msg_fraction
    )
    graph = build_g_struct(nodes, partition.e_msg)
    missing_tokens = [node for node in nodes if node not in token_index]
    if missing_tokens:
        raise ValueError(f"token pack is missing {len(missing_tokens)} probe nodes")
    holdout = derive_internal_holdout(nodes, partition.e_msg, partition.e_sup)
    store = FeatureStore(cfg.data.root / te._FEATURES_SUBDIR)
    cache_dir = output_path.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix, node_index = build_f0_matrix(
        store, nodes, cache_path=cache_dir / "probe_f0.pt", allow_cache_subset=True
    )
    n_ground = EgoStitchConfig().n_ground
    matrix_np = matrix.numpy()
    grounding_rows: dict[str, list[int]] = {}
    role_universes = (
        (sorted(holdout.v_fit), "probe_grounding_fit.npz"),
        (sorted(holdout.v_qual), "probe_grounding_qual.npz"),
        (sorted(holdout.v_select), "probe_grounding_select.npz"),
    )
    for role_nodes, cache_name in role_universes:
        role_rows = np.asarray(
            matrix_np[[node_index[node] for node in role_nodes]], dtype=np.float32
        )
        pool = build_grounding_pool(
            role_rows, role_nodes, n_ground=n_ground, cache_path=cache_dir / cache_name
        )
        for node in role_nodes:
            grounding_rows[node] = [node_index[neighbor] for neighbor in pool[node]]
    data = _ProbeBundle(
        node_index=node_index,
        f0=matrix,
        grounding_index=np.asarray([grounding_rows[node] for node in nodes], dtype=np.int64),
        train_pos={node: position for position, node in enumerate(nodes)},
    )
    batch_size = max(1, cfg.data.edge_batch)
    state_rows: list[NDArray[np.float32]] = []
    with torch.inference_mode():
        for start in range(0, len(nodes), batch_size):
            batch_nodes = nodes[start : start + batch_size]
            batch = _probe_batch(data, table, token_index, batch_nodes, batch_nodes, device)
            state_rows.append(model.probe_states(batch).mean(dim=1).float().cpu().numpy())

    pairs = select_probe_pairs(graph)
    inverse_index = {row: node for node, row in data.node_index.items()}
    consistency: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            batch = _probe_batch(
                data,
                table,
                token_index,
                [pair[0] for pair in batch_pairs],
                [pair[1] for pair in batch_pairs],
                device,
            )
            state_a, state_b, is_self = model._pair_node_states(batch)
            context = model.build_pair_context_from_states(
                state_a, state_b, is_self, need_topo=True, need_cont=False
            )
            assert context.plan is not None
            assert state_a.ground_ids is not None and state_b.ground_ids is not None
            selected_a = torch.gather(state_a.ground_ids, 1, state_a.slots.pointer.argmax(dim=-1))
            selected_b = torch.gather(state_b.ground_ids, 1, state_b.slots.pointer.argmax(dim=-1))
            for row, (node_u, node_v) in enumerate(batch_pairs):
                common = set(graph.neighbors(node_u)) & set(graph.neighbors(node_v))
                common_rows = {data.node_index[node] for node in common}
                ids_a = selected_a[row]
                ids_b = selected_b[row]
                equal = ids_a[:, None] == ids_b[None, :]
                grounded = (state_a.slots.gate[row, :, None] > 0.5) & (
                    state_b.slots.gate[row, None, :] > 0.5
                )
                real_common = torch.zeros_like(equal)
                for slot_a in range(equal.size(0)):
                    identity = int(ids_a[slot_a].item())
                    if identity in common_rows and inverse_index.get(identity) in common:
                        real_common[slot_a] = equal[slot_a]
                mask = equal & grounded & real_common
                plan = context.plan[row]
                consistency.append(
                    float((plan * mask).sum().float() / plan.sum().float().clamp_min(1e-30))
                )
    targets = probe_targets(graph, nodes)
    write_e2e_probe_artifact(
        output_path,
        metadata={
            "checkpoint_id": checkpoint_id,
            "registration_sha256": registration_sha,
            "config_hash": config_hash,
            "seed": 0,
            "partition_seed": 0,
            "strategy": strategy,
            "g_struct_sha256": g_struct_sha256(graph),
        },
        node_ids=nodes,
        states=np.concatenate(state_rows, axis=0),
        targets={name: targets[name] for name in ("degree", "ego_density", "clustering")},
        pair_ids=pairs,
        pi_consistency=np.asarray(consistency, dtype=np.float64),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the probe producer CLI."""
    parser = argparse.ArgumentParser(prog="python -m src.experiments.probes")
    subparsers = parser.add_subparsers(dest="command", required=True)
    produce = subparsers.add_parser("produce-e2e")
    produce.add_argument("--checkpoint", type=Path, required=True)
    produce.add_argument("--run-metadata", type=Path, required=True)
    produce.add_argument("--preregistration", type=Path, required=True)
    produce.add_argument("--data-root", type=Path, required=True)
    produce.add_argument("--strategy", required=True)
    produce.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    if args.command == "produce-e2e":
        produce_e2e_probe_artifact(
            checkpoint_path=args.checkpoint,
            run_metadata_path=args.run_metadata,
            preregistration_path=args.preregistration,
            data_root=args.data_root,
            strategy=args.strategy,
            output_path=args.output,
        )


__all__ = [
    "degree_partialled_r2",
    "evaluate_e2e_probe_artifact",
    "g_struct_sha256",
    "linear_probe_r2",
    "probe_targets",
    "select_probe_pairs",
    "write_e2e_probe_artifact",
]


if __name__ == "__main__":
    main()

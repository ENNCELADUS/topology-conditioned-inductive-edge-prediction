"""Train and score the released CAZI-MBN teacher/student on the local benchmark."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import pickle
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import yaml
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, svds
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from src.baselines.cazi_mbn import CAZIStudent, CAZITeacher
from src.data.artifacts import Benchmark, load_benchmark
from src.data.feature_stats import FeatureStats, feature_stats_for_universe
from src.data.features import FeatureStore, build_f0_matrix
from src.data.pairs import NegativeSampler
from src.data.partition import build_g_struct
from src.data.training_sampler import enumerate_edge_stream
from src.data.val_region import derive_val_region_split, val_ball_union_universe
from src.eval.checkpoint_selection import SELECTION_RULE, CheckpointCandidate, select_checkpoint
from src.eval.val_topology import (
    ValTopologyReference,
    build_val_topology_reference,
    val_region_topology_metrics,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class CAZIConfig:
    """Validated execution settings for the CAZI-MBN baseline."""

    data_root: Path
    strategy: str
    output_dir: Path
    f0_cache: Path
    expected_missing_features: tuple[str, ...]
    seed: int
    order: int
    topology_dim: int
    latent_dim: int
    network_layers: int
    heads: int
    batch_size: int
    score_batch_size: int
    learning_rate: float
    weight_decay: float
    teacher_epochs: int
    student_epochs: int
    patience: int
    discriminator_coef: float
    regularization_coef: float
    classification_coef: float
    distillation_weight: float
    supervised_weight: float
    topology_every: int = 10


@dataclasses.dataclass(frozen=True)
class PreparedData:
    """Train-side arrays assembled without opening held-out topology."""

    benchmark: Benchmark
    train_nodes: list[str]
    train_sequence: torch.Tensor
    topology: torch.Tensor
    positive_edge_index: torch.Tensor
    negative_edge_index: torch.Tensor
    training_positives: list[tuple[str, str]]
    sampler: NegativeSampler
    topology_reference: ValTopologyReference
    topology_pairs: list[tuple[str, str]]
    student_val_nodes: list[str]
    student_val_sequence: torch.Tensor
    student_val_pairs: list[tuple[str, str]]
    student_val_labels: NDArray[np.int8]
    student_val_position: dict[str, int]
    train_node_position: dict[str, int]
    feature_stats: FeatureStats
    missing_features: frozenset[str]


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def load_config(path: Path) -> CAZIConfig:
    """Load the small isolated CAZI baseline YAML configuration."""
    raw = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "config")
    data = _mapping(raw["data"], "data")
    model = _mapping(raw["model"], "model")
    optim = _mapping(raw["optim"], "optim")
    runtime = _mapping(raw["runtime"], "runtime")
    loss = _mapping(raw["loss"], "loss")
    return CAZIConfig(
        data_root=Path(str(data["root"])),
        strategy=str(data["strategy"]),
        output_dir=Path(str(raw["output_dir"])),
        f0_cache=Path(str(data["f0_cache"])),
        expected_missing_features=tuple(str(x) for x in data["expected_missing_features"]),
        seed=int(raw["seed"]),
        order=int(model["order"]),
        topology_dim=int(model["topology_dim"]),
        latent_dim=int(model["latent_dim"]),
        network_layers=int(model["network_layers"]),
        heads=int(model["heads"]),
        batch_size=int(runtime["batch_size"]),
        score_batch_size=int(runtime["score_batch_size"]),
        learning_rate=float(optim["learning_rate"]),
        weight_decay=float(optim["weight_decay"]),
        teacher_epochs=int(optim["teacher_epochs"]),
        student_epochs=int(optim["student_epochs"]),
        patience=int(optim["patience"]),
        discriminator_coef=float(loss["discriminator_coef"]),
        regularization_coef=float(loss["regularization_coef"]),
        classification_coef=float(loss["classification_coef"]),
        distillation_weight=float(loss["distillation_weight"]),
        supervised_weight=float(loss["supervised_weight"]),
        topology_every=int(_mapping(raw.get("eval", {}), "eval").get("topology_every", 10)),
    )


def seed_everything(seed: int) -> None:
    """Seed every random source used by this runner."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(requested: str) -> torch.device:
    """Resolve auto/cpu/mps/cuda without silently changing the request."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _edge_index(pairs: Sequence[tuple[str, str]], node_position: Mapping[str, int]) -> torch.Tensor:
    rows: list[int] = []
    cols: list[int] = []
    for u, v in pairs:
        if u == v:
            continue
        ui = node_position[u]
        vi = node_position[v]
        rows.extend((ui, vi))
        cols.extend((vi, ui))
    return torch.tensor([rows, cols], dtype=torch.long)


def compute_ugt_projection(
    node_ids: Sequence[str],
    edges: Sequence[tuple[str, str]],
    *,
    order: int,
    feature_length: int,
    seed: int,
) -> NDArray[np.float32]:
    """Compute CAZI's order-k normalized-adjacency SVD without densifying it."""
    if feature_length >= len(node_ids):
        raise ValueError("UGT feature length must be smaller than the node count")
    position = {node: i for i, node in enumerate(node_ids)}
    row: list[int] = []
    col: list[int] = []
    for u, v in edges:
        if u == v:
            continue
        ui = position[u]
        vi = position[v]
        row.extend((ui, vi))
        col.extend((vi, ui))
    adjacency = sp.coo_matrix(
        (np.ones(len(row), dtype=np.float64), (row, col)),
        shape=(len(node_ids), len(node_ids)),
    ).tocsr()
    adjacency.data.fill(1.0)
    adjacency.eliminate_zeros()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree)
    nonzero = degree > 0
    inv_sqrt[nonzero] = np.power(degree[nonzero], -0.5)
    normalized = sp.diags(inv_sqrt) @ adjacency @ sp.diags(inv_sqrt)

    def apply(values: NDArray[np.float64]) -> NDArray[np.float64]:
        power = values
        total = np.zeros_like(values)
        for _ in range(order):
            power = normalized @ power
            total += power
        return total

    operator = LinearOperator(
        normalized.shape,
        matvec=apply,
        rmatvec=apply,
        matmat=apply,
        rmatmat=apply,
        dtype=np.float64,
    )
    u, singular_values, _ = svds(
        operator,
        k=feature_length,
        which="LM",
        rng=np.random.default_rng(seed),
    )
    order_idx = np.argsort(singular_values)[::-1]
    projected = u[:, order_idx] * singular_values[order_idx]
    std = projected.std(axis=0)
    std[std == 0] = 1.0
    projected = (projected - projected.mean(axis=0)) / std
    return cast(NDArray[np.float32], projected.astype(np.float32))


def load_or_build_ugt(
    path: Path,
    node_ids: Sequence[str],
    edges: Sequence[tuple[str, str]],
    *,
    order: int,
    feature_length: int,
    seed: int,
) -> torch.Tensor:
    """Load an exact-node-order UGT cache or create it."""
    if path.exists():
        with np.load(path, allow_pickle=False) as payload:
            if payload["node_ids"].astype(str).tolist() != list(node_ids):
                raise ValueError(f"{path}: cached UGT node order mismatch")
            projection = payload["projection"].astype(np.float32, copy=False)
            if projection.shape != (len(node_ids), feature_length):
                raise ValueError(f"{path}: cached UGT projection shape mismatch")
    else:
        logger.info("computing sparse UGT projection for %d nodes", len(node_ids))
        projection = compute_ugt_projection(
            node_ids,
            edges,
            order=order,
            feature_length=feature_length,
            seed=seed,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            node_ids=np.asarray(node_ids),
            projection=projection,
        )
    return torch.from_numpy(projection.copy())


def _feature_coverage(
    benchmark_root: Path,
    store: FeatureStore,
    expected_missing: Iterable[str],
) -> tuple[frozenset[str], list[str]]:
    with (benchmark_root / "graph.pkl").open("rb") as handle:
        graph = pickle.load(handle)  # noqa: S301 - pinned local artifact
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph.pkl must contain a networkx.Graph")
    graph_nodes = set(cast(Iterable[str], graph.nodes()))
    missing = frozenset(graph_nodes - store.node_ids)
    expected = frozenset(expected_missing)
    if missing != expected:
        raise ValueError(
            f"feature coverage drift: actual missing={sorted(missing)}, expected={sorted(expected)}"
        )
    return missing, sorted(graph_nodes & store.node_ids)


def _standardize_f0(rows: torch.Tensor, stats: FeatureStats) -> torch.Tensor:
    """Apply the training-universe feature statistics to fp32 F0 rows."""
    standardized = (rows.float() - torch.from_numpy(stats.mu)) / torch.from_numpy(stats.sigma)
    if not bool(torch.isfinite(standardized).all()):
        raise ValueError("standardized CAZI features are not finite")
    return standardized


def prepare_data(cfg: CAZIConfig) -> PreparedData:
    """Assemble shared training interactions and CAZI inputs."""
    benchmark_root = cfg.data_root / "benchmark_2025_neurips"
    feature_root = cfg.data_root / "features" / "frozen_node_features_1024"
    store = FeatureStore(feature_root)
    missing, operative_nodes = _feature_coverage(
        benchmark_root, store, cfg.expected_missing_features
    )
    benchmark = load_benchmark(
        benchmark_root,
        cfg.strategy,
        verify=True,
        exclude_nodes=missing,
    )
    # V_val derivation needs the raw label-0 rows; `benchmark` above already
    # dropped `missing`-touching rows, so re-read train/val_edges.txt unfiltered
    # here (train_graph.pkl and train_nodes are never exclude-node filtered).
    raw_benchmark = load_benchmark(benchmark_root, cfg.strategy, verify=False)
    raw_negatives = [
        pair
        for pair, label in zip(
            raw_benchmark.split.train_pairs.pairs,
            raw_benchmark.split.train_pairs.labels,
            strict=True,
        )
        if label == 0
    ] + [
        pair
        for pair, label in zip(
            raw_benchmark.split.val_pairs.pairs,
            raw_benchmark.split.val_pairs.labels,
            strict=True,
        )
        if label == 0
    ]
    val_split = derive_val_region_split(
        benchmark.split.train_nodes,
        benchmark.split.train_graph.edges(),
        raw_negatives,
        benchmark.positive_edges,
    )
    cfg.f0_cache.parent.mkdir(parents=True, exist_ok=True)
    f0, f0_position = build_f0_matrix(store, operative_nodes, cache_path=cfg.f0_cache)
    train_nodes = sorted(val_split.train_nodes - missing)
    student_val_nodes = sorted(val_split.v_val)
    train_node_position = {node: i for i, node in enumerate(train_nodes)}
    student_val_position = {node: i for i, node in enumerate(student_val_nodes)}
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    feature_stats = feature_stats_for_universe(
        f0.numpy(),
        f0_position,
        train_nodes,
        cache_path=cfg.output_dir / "feature_stats.npz",
    )
    train_sequence = _standardize_f0(f0[[f0_position[node] for node in train_nodes]], feature_stats)
    student_val_sequence = _standardize_f0(
        f0[[f0_position[node] for node in student_val_nodes]], feature_stats
    )

    training_positives = sorted(
        pair
        for pair in val_split.training_positives
        if pair[0] not in missing and pair[1] not in missing
    )
    g_struct = build_g_struct(train_nodes, training_positives)
    topology_edges = sorted(cast(Iterable[tuple[str, str]], g_struct.edges()))
    sampler = NegativeSampler(train_nodes, dict(g_struct.degree()), frozenset(training_positives))
    rows = enumerate_edge_stream(
        training_positives,
        sampler,
        negative_ratio=5,
        seed=cfg.seed,
        epoch=1,
        rank=0,
        world_size=1,
    )
    negative_edges = [(u, v) for u, v, label in rows if label == 0 and u != v][
        : len(topology_edges)
    ]
    topology = load_or_build_ugt(
        cfg.output_dir / "ugt_projection.npz",
        train_nodes,
        topology_edges,
        order=cfg.order,
        feature_length=cfg.topology_dim,
        seed=cfg.seed,
    )
    student_val_pairs = list(val_split.val_cls_pairs)
    student_val_labels = np.asarray(val_split.val_cls_labels, dtype=np.int8)
    if not student_val_pairs or len(set(student_val_labels.tolist())) != 2:
        raise ValueError("CAZI validation must contain both classes on V_val")
    union = val_ball_union_universe(val_split)
    topology_pairs = [
        (student_val_nodes[int(u)], student_val_nodes[int(v)])
        for u, v in zip(union.u_idx, union.v_idx, strict=True)
    ]
    logger.info(
        "split seed=%d root=%s train_nodes=%d train_positives=%d V_val_nodes=%d val_cls=%d",
        val_split.params.split_seed,
        val_split.region_seeds,
        len(train_nodes),
        len(training_positives),
        len(student_val_nodes),
        len(student_val_pairs),
    )
    return PreparedData(
        benchmark=benchmark,
        train_nodes=train_nodes,
        train_sequence=train_sequence,
        topology=topology,
        positive_edge_index=_edge_index(topology_edges, train_node_position),
        negative_edge_index=_edge_index(negative_edges, train_node_position),
        training_positives=training_positives,
        sampler=sampler,
        topology_reference=build_val_topology_reference(val_split),
        topology_pairs=topology_pairs,
        student_val_nodes=student_val_nodes,
        student_val_sequence=student_val_sequence,
        student_val_pairs=student_val_pairs,
        student_val_labels=student_val_labels,
        student_val_position=student_val_position,
        train_node_position=train_node_position,
        feature_stats=feature_stats,
        missing_features=missing,
    )


def _pair_indices(
    pairs: Sequence[tuple[str, str]], position: Mapping[str, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([position[u] for u, _ in pairs], dtype=torch.long),
        torch.tensor([position[v] for _, v in pairs], dtype=torch.long),
    )


def _batched_logits(
    model: CAZITeacher | CAZIStudent,
    sequence: torch.Tensor,
    u_idx: torch.Tensor,
    v_idx: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for start in range(0, len(u_idx), batch_size):
        stop = min(start + batch_size, len(u_idx))
        parts.append(model.pair_logits(sequence, u_idx[start:stop], v_idx[start:stop]))
    return torch.cat(parts)


def _validation_metrics(
    model: CAZITeacher | CAZIStudent,
    sequence: torch.Tensor,
    pairs: Sequence[tuple[str, str]],
    labels: NDArray[np.int8],
    position: Mapping[str, int],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float, float]:
    """Return ``(auroc, auprc, task_loss)`` over the validation pairs.

    ``task_loss`` is the unweighted val_cls BCE used for student early stopping.
    Teacher graph terms and student distillation have no V_val targets.
    """
    u_idx, v_idx = _pair_indices(pairs, position)
    model.eval()
    with torch.no_grad():
        logits = _batched_logits(
            model,
            sequence,
            u_idx.to(device),
            v_idx.to(device),
            batch_size=batch_size,
        )
        task_loss = float(
            nn.functional.binary_cross_entropy_with_logits(
                logits, torch.as_tensor(labels, dtype=logits.dtype, device=logits.device)
            )
        )
    probs = logits.cpu().numpy()
    if not np.isfinite(probs).all() or not math.isfinite(task_loss):
        raise ValueError("non-finite validation state")
    return (
        float(roc_auc_score(labels, probs)),
        float(average_precision_score(labels, probs)),
        task_loss,
    )


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _write_history(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _epoch_pairs(
    cfg: CAZIConfig,
    data: PreparedData,
    epoch: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = enumerate_edge_stream(
        data.training_positives,
        data.sampler,
        negative_ratio=5,
        seed=cfg.seed,
        epoch=epoch,
        rank=0,
        world_size=1,
    )
    u, v = _pair_indices([(a, b) for a, b, _ in rows], data.train_node_position)
    labels = torch.tensor([label for _, _, label in rows], dtype=torch.float32, device=device)
    return u.to(device), v.to(device), labels


def train_teacher(
    cfg: CAZIConfig,
    data: PreparedData,
    *,
    device: torch.device,
    epochs: int,
) -> CAZITeacher:
    """Train the topology-aware teacher with the released loss coefficients."""
    sequence = data.train_sequence.to(device)
    topology = data.topology.to(device)
    positive_edge_index = data.positive_edge_index.to(device)
    negative_edge_index = data.negative_edge_index.to(device)
    model = CAZITeacher(
        len(data.train_nodes),
        sequence.shape[1],
        topology_dim=cfg.topology_dim,
        latent_dim=cfg.latent_dim,
        network_layers=cfg.network_layers,
        heads=cfg.heads,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.75)
    best_training_loss = math.inf
    best_state = _clone_state(model)
    best_epoch = 0
    patience = 0
    history_path = cfg.output_dir / "teacher_history.jsonl"
    history_path.unlink(missing_ok=True)
    for epoch in range(1, epochs + 1):
        train_u, train_v, labels = _epoch_pairs(cfg, data, epoch, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        discriminator_loss, regularization_loss = model.graph_objective(
            topology, positive_edge_index, negative_edge_index
        )
        graph_loss = (
            cfg.discriminator_coef * discriminator_loss
            + cfg.regularization_coef * regularization_loss
        )
        graph_loss.backward()  # type: ignore[no-untyped-call]
        permutation = torch.randperm(len(train_u), device=device)
        classification_sum = 0.0
        for start in range(0, len(train_u), cfg.batch_size):
            batch = permutation[start : start + cfg.batch_size]
            logits = model.pair_logits(sequence, train_u[batch], train_v[batch])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labels[batch], pos_weight=logits.new_tensor(5.0)
            )
            weighted = cfg.classification_coef * loss * (len(batch) / len(train_u))
            weighted.backward()  # type: ignore[no-untyped-call]
            classification_sum += float(loss.detach()) * len(batch)
        nn.utils.clip_grad_norm_(model.parameters(), math.inf, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        total_loss = float(
            graph_loss.detach()
        ) + cfg.classification_coef * classification_sum / len(train_u)
        if not math.isfinite(total_loss):
            raise ValueError("non-finite CAZI teacher loss")
        _write_history(
            history_path,
            {
                "epoch": epoch,
                "training_total_loss": total_loss,
                "discriminator_loss": float(discriminator_loss.detach()),
                "regularization_loss": float(regularization_loss.detach()),
                "classification_loss": classification_sum / len(train_u),
            },
        )
        logger.info("teacher epoch=%d training_total_loss=%.6f", epoch, total_loss)
        # Consensus is a train-node lookup: V_val cannot be scored by this teacher.
        # This is training convergence, not a validation-selected deployable model.
        if total_loss < best_training_loss:
            best_training_loss = total_loss
            best_state = _clone_state(model)
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= cfg.patience:
                break
    model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": best_state,
            "epoch": best_epoch,
            "selection_rule": "teacher_training_total_loss",
            "training_total_loss": best_training_loss,
        },
        cfg.output_dir / "teacher.pt",
    )
    return model


def train_student(
    cfg: CAZIConfig,
    data: PreparedData,
    teacher: CAZITeacher,
    *,
    device: torch.device,
    epochs: int,
) -> CAZIStudent:
    """Distill the teacher node latent into the sequence-only student."""
    sequence = data.train_sequence.to(device)
    student_val_sequence = data.student_val_sequence.to(device)
    teacher.eval()
    with torch.no_grad():
        teacher_latent = teacher.distilled_latent().detach()
    model = CAZIStudent(
        sequence.shape[1],
        latent_dim=cfg.latent_dim,
        network_layers=cfg.network_layers,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
    best_val_loss = math.inf
    candidates: list[CheckpointCandidate] = []
    checkpoint_dir = cfg.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    patience = 0
    history_path = cfg.output_dir / "student_history.jsonl"
    history_path.unlink(missing_ok=True)
    for epoch in range(1, epochs + 1):
        train_u, train_v, labels = _epoch_pairs(cfg, data, epoch, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        student_latent = model.node_latent(sequence)
        distillation_loss = nn.functional.mse_loss(student_latent, teacher_latent)
        permutation = torch.randperm(len(train_u), device=device)
        classification_parts: list[torch.Tensor] = []
        classification_labels: list[torch.Tensor] = []
        for start in range(0, len(train_u), cfg.batch_size):
            batch = permutation[start : start + cfg.batch_size]
            classification_parts.append(
                model.classifier(
                    student_latent[train_u[batch]], student_latent[train_v[batch]]
                ).squeeze(1)
            )
            classification_labels.append(labels[batch])
        classification_loss = nn.functional.binary_cross_entropy_with_logits(
            torch.cat(classification_parts),
            torch.cat(classification_labels),
            pos_weight=sequence.new_tensor(5.0),
        )
        total_loss = (
            cfg.distillation_weight * distillation_loss
            + cfg.supervised_weight * classification_loss
        )
        if not bool(torch.isfinite(total_loss)):
            raise ValueError("non-finite CAZI student loss")
        total_loss.backward()  # type: ignore[no-untyped-call]
        nn.utils.clip_grad_norm_(model.parameters(), math.inf, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        val_auroc, val_auprc, val_total_loss = _validation_metrics(
            model,
            student_val_sequence,
            data.student_val_pairs,
            data.student_val_labels,
            data.student_val_position,
            batch_size=cfg.score_batch_size,
            device=device,
        )
        row = {
            "epoch": epoch,
            "distillation_loss": float(distillation_loss.detach()),
            "classification_loss": float(classification_loss.detach()),
            "val_auroc": val_auroc,
            "val_auprc": val_auprc,
            "val_total_loss": val_total_loss,
        }
        _write_history(history_path, row)
        logger.info(
            "student epoch=%d val_auroc=%.6f val_auprc=%.6f val_total=%.6f",
            epoch,
            val_auroc,
            val_auprc,
            val_total_loss,
        )
        due = epoch == 1 or epoch % cfg.topology_every == 0 or epoch == epochs
        if due:
            u, v = _pair_indices(data.topology_pairs, data.student_val_position)
            model.eval()
            with torch.no_grad():
                logits = (
                    _batched_logits(
                        model,
                        student_val_sequence,
                        u.to(device),
                        v.to(device),
                        batch_size=cfg.score_batch_size,
                    )
                    .cpu()
                    .numpy()
                )
            topology = val_region_topology_metrics(
                u_idx=u.numpy(),
                v_idx=v.numpy(),
                logits=logits,
                reference=data.topology_reference,
            )
            candidates.append(CheckpointCandidate(epoch, val_auprc, topology.metrics))
            payload = {
                "state_dict": _clone_state(model),
                "epoch": epoch,
                "selection_rule": SELECTION_RULE,
                "val_threshold_transfer": {
                    "n_val": len(data.student_val_nodes),
                    "threshold": topology.threshold,
                },
                "val_auprc": val_auprc,
                "topology": dataclasses.asdict(topology),
            }
            torch.save(payload, checkpoint_dir / f"epoch-{epoch:04d}.pt")
            _write_history(
                cfg.output_dir / "validation_topology.jsonl",
                {key: value for key, value in payload.items() if key != "state_dict"},
            )
            logger.info("student epoch=%d topology=%s", epoch, dataclasses.asdict(topology))
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            patience = 0
        else:
            patience += 1
        if patience >= cfg.patience and due:
            logger.info("student early stop at epoch %d", epoch)
            break
    selected = select_checkpoint(candidates)
    if selected is None:
        raise RuntimeError("no CAZI checkpoint received topology validation")
    selected_payload = torch.load(
        checkpoint_dir / f"epoch-{selected.epoch:04d}.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(selected_payload["state_dict"])
    torch.save(selected_payload, cfg.output_dir / "student.pt")
    (cfg.output_dir / "selection.json").write_text(
        json.dumps(
            {
                "selection_rule": SELECTION_RULE,
                "selected_epoch": selected.epoch,
                "candidates": [dataclasses.asdict(candidate) for candidate in candidates],
                "val_threshold_transfer": selected_payload["val_threshold_transfer"],
            },
            indent=2,
        )
        + "\n"
    )
    logger.info(
        "selected student epoch=%d threshold=%s",
        selected.epoch,
        selected_payload["val_threshold_transfer"],
    )
    return model


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stage", choices=("prepare", "train", "score", "all"), default="all")
    parser.add_argument("--max-teacher-epochs", type=int)
    parser.add_argument("--max-student-epochs", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.output_dir is not None:
        cfg = dataclasses.replace(cfg, output_dir=args.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.topology_every < 1:
        raise ValueError("eval.topology_every must be positive")
    started = time.monotonic()
    seed_everything(cfg.seed)
    device = select_device(args.device)
    logger.info("device=%s output_dir=%s", device, cfg.output_dir)
    try:
        if args.stage != "score":
            data = prepare_data(cfg)
        if args.stage == "prepare":
            return
        if args.stage in {"train", "all"}:
            teacher = train_teacher(
                cfg,
                data,
                device=device,
                epochs=args.max_teacher_epochs or cfg.teacher_epochs,
            )
            train_student(
                cfg,
                data,
                teacher,
                device=device,
                epochs=args.max_student_epochs or cfg.student_epochs,
            )
        if args.stage in {"train", "all"}:
            (cfg.output_dir / "failure.json").unlink(missing_ok=True)
            (cfg.output_dir / "complete.json").write_text(
                json.dumps(
                    {"status": "complete", "total_seconds": time.monotonic() - started},
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        if args.stage in {"score", "all"}:
            from src.eval.test_protocol import main as test_main

            test_main(
                [
                    "--checkpoint",
                    str(cfg.output_dir / "student.pt"),
                    "--model-family",
                    "cazi_mbn",
                    "--model-config",
                    str(args.config),
                    "--output-dir",
                    str(cfg.output_dir),
                    "--data-root",
                    str(cfg.data_root),
                    "--strategy",
                    cfg.strategy,
                    "--arm",
                    "cazi_mbn",
                    "--seed",
                    str(cfg.seed),
                ]
            )
    except Exception as error:
        (cfg.output_dir / "complete.json").unlink(missing_ok=True)
        (cfg.output_dir / "failure.json").write_text(
            json.dumps({"status": "failed", "error": str(error)}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()

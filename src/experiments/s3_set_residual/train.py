"""S3 trainer: set-size-invariant residual loop, V_val selection, and topology-ranked publish.

Trains `SetResidualModel` (`src.experiments.s3_set_residual.model`) on `S3Corpus`'s
V_val-masked training pairs (`src.experiments.s3_set_residual.data`), single device, no
DDP, no autocast, fp32 throughout. Loss is `BCEWithLogitsLoss(base_logit + delta,
label)`, averaged per set then across sets in a batch (`_batch_loss`) so a batch's
gradient does not depend on how many pairs a large set happens to contribute --
grouped by `pairs[:, 0]` (the set row), matching the zero-init contract's per-pair
residual semantics.

Selection is two-staged:

1. **Per epoch**, `_validate` runs the model over every `VvalEval` bucket's full
   `i < j` pair matrix and reports the paired macro ΔAUPRC (`AUPRC(base + delta) -
   AUPRC(base)`, meaned over sets with both classes present) plus telemetry-only
   readouts (delta variance; share of pairs where `|delta| > |base|`; the count of
   sets that contributed to ΔAUPRC vs. were skipped for being single-class) --
   logged to `metrics.jsonl`. Fail-closed: if *every* bucket set is single-class,
   ΔAUPRC is undefined and `_validate` raises `RuntimeError` rather than returning
   `NaN` into downstream top-k/early-stop/selection math. An explicit epoch-0 pass
   runs *before* the first optimizer step; at zero-init `delta` is exactly `0` for
   every pair (`model.py`'s contract), so `base + delta` is bit-identical to `base`
   and the epoch-0 ΔAUPRC is exactly `0.0`, not merely close to it. The top
   `cfg.top_k_checkpoints` training epochs (by this metric, epoch 0 excluded) are
   kept as `epoch_<E>.pt`; the rest are pruned as the frontier moves. `last.pt` is
   overwritten every epoch regardless of rank.
2. **After training**, each of the top-K checkpoints' five-number topology (BFS-macro
   GS, RD, and the degree/clustering/spectral MMD ratios -- `_topology_five_numbers`,
   built from the same public `src.eval.assembly`/`src.eval.graph_metrics` primitives
   S2's evaluate module uses for density-matched assembly and bucket-referenced MMD)
   is computed once against `VvalEval`. `_select_best` mean-ranks
   `[ΔAUPRC↑, GS↑, |RD-1|↓, deg-MMD↓, clust-MMD↓, spec-MMD↓]` across the K candidates
   (ties broken by the lower epoch) and that epoch's checkpoint is copied to `best.pt`.
   The full selection table, published epoch, and the base scorer's `checkpoint_id`
   are written to `run_metadata.json`.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score
from torch.nn.utils import clip_grad_norm_

from src.data.features import FeatureStore
from src.eval.assembly import assemble_graph, density_matched_threshold
from src.eval.graph_metrics import (
    STATISTICS,
    MMDConfig,
    clustering_histogram,
    compute_graph_similarity,
    compute_relative_density,
    degree_histogram,
    laplacian_spectrum_histogram,
    mmd_squared,
)
from src.experiments.s2_latent_topology.data import build_features, iter_size_batches
from src.experiments.s3_set_residual.data import S3Corpus, VvalEval, collate_pair_batch
from src.experiments.s3_set_residual.model import ResidualConfig, SetResidualModel

logger = logging.getLogger(__name__)

__all__ = [
    "TrainConfig",
    "VvalFeatures",
    "build_vval_features",
    "load_vval_features",
    "save_vval_features",
    "train_arm",
]


# --------------------------------------------------------------------------- V_val features


@dataclass(frozen=True)
class VvalFeatures:
    """The fp32 F0 feature matrix covering every node any `VvalEval` bucket references.

    Attributes:
        node_ids: Sorted node ids; a node's row in `features` is its position here.
        features: `(N, D)` fp32 mean-pooled F0 feature matrix in `node_ids` order.
    """

    node_ids: list[str]
    features: torch.Tensor


def build_vval_features(vval: VvalEval, store: FeatureStore, *, cache_path: Path) -> VvalFeatures:
    """Build the feature matrix for every node referenced by any `vval` bucket.

    `VvalEval` itself carries no raw features (only base logits and true
    adjacency), so the residual model's per-epoch validation pass needs this
    built once, up front, and persisted -- the `train` CLI stage loads it back
    without needing a `FeatureStore` at all.

    Args:
        vval: The `VvalEval` whose bucket node ids define the required universe.
        store: Frozen per-node feature store.
        cache_path: F0 feature cache path (passed through to `build_features`).

    Returns:
        The `VvalFeatures`.
    """
    node_ids = sorted({node_id for sets in vval.node_ids.values() for s in sets for node_id in s})
    features, _missing = build_features(store, node_ids, cache_path=cache_path)
    return VvalFeatures(node_ids=node_ids, features=features)


def save_vval_features(features: VvalFeatures, path: Path) -> None:
    """Save a `VvalFeatures` as a plain-dict torch checkpoint at `path`."""
    torch.save({"node_ids": features.node_ids, "features": features.features}, path)


def load_vval_features(path: Path) -> VvalFeatures:
    """Load a `VvalFeatures` previously written by `save_vval_features`."""
    payload = cast(dict[str, object], torch.load(path, map_location="cpu", weights_only=True))
    return VvalFeatures(
        node_ids=cast(list[str], payload["node_ids"]),
        features=cast(torch.Tensor, payload["features"]),
    )


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class TrainConfig:
    """Training-loop hyperparameters, independent of `ResidualConfig`'s model dims.

    Attributes:
        epochs: Maximum training epochs.
        batch_regions: Max regions per size-pure batch (`iter_size_batches`).
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        grad_clip: Max gradient norm (`clip_grad_norm_`).
        patience: Early-stop patience (epochs since the last ΔAUPRC improvement).
        seed: Controls torch/numpy init and batch shuffling only -- corpus/V_val
            sampling salts are fixed and shared across arms and seeds.
        device: Torch device string.
        top_k_checkpoints: Number of top-ΔAUPRC epoch checkpoints kept for the
            post-training five-number topology selection.
    """

    epochs: int = 40
    batch_regions: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    patience: int = 10
    seed: int = 0
    device: str = "cpu"
    top_k_checkpoints: int = 3


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    """Append one JSON-encoded record as a line to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------- V_val forward pass


def _vval_forward(
    model: SetResidualModel, vval: VvalEval, features: VvalFeatures, device: torch.device
) -> dict[int, torch.Tensor]:
    """Run `model` over every V_val bucket's full `i < j` pair matrix, batched by size.

    Returns:
        Size -> `(b, P)` fp32 delta tensor, `b` the draw count for that size and
        `P = size * (size - 1) // 2`; column order matches
        `np.triu_indices(size, k=1)`, the same order every other V_val-side
        upper-triangle read in this module uses.
    """
    node_index = {node_id: i for i, node_id in enumerate(features.node_ids)}
    was_training = model.training
    model.eval()
    out: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for size in vval.sizes:
            sets = vval.node_ids[size]
            b = len(sets)
            x = torch.stack(
                [features.features[[node_index[node_id] for node_id in s]] for s in sets]
            ).to(device)
            mask = torch.ones(b, size, device=device, dtype=torch.float32)
            iu, iv = np.triu_indices(size, k=1)
            p = iu.shape[0]
            local_pairs = torch.tensor(np.stack([iu, iv], axis=1), dtype=torch.long)
            pairs = torch.cat(
                [
                    torch.cat([torch.full((p, 1), row, dtype=torch.long), local_pairs], dim=1)
                    for row in range(b)
                ],
                dim=0,
            ).to(device)
            delta = model(x, mask, pairs)
            out[size] = delta.reshape(b, p)
    if was_training:
        model.train()
    return out


def _validate(
    model: SetResidualModel, vval: VvalEval, features: VvalFeatures, device: torch.device
) -> tuple[float, float, float, int, int]:
    """Paired macro ΔAUPRC over every V_val bucket, plus delta telemetry.

    Returns:
        `(macro_delta_auprc, delta_variance, margin_share, n_sets_used,
        n_sets_skipped_single_class)`: `macro_delta_auprc` is the mean of
        `AUPRC(base + delta) - AUPRC(base)` over every bucket set with both
        classes present in its `i < j` labels; `delta_variance` is the
        population variance of every pair's delta across every bucket (all
        sets, used or skipped); `margin_share` is the fraction of pairs where
        `|delta| > |base|` (all sets); `n_sets_used`/
        `n_sets_skipped_single_class` count how many bucket sets did/didn't
        contribute to `macro_delta_auprc`. The variance/margin/count fields
        are telemetry only, never gating.

    Raises:
        RuntimeError: If every V_val bucket set is single-class, so
            `macro_delta_auprc` would otherwise be an undefined mean of zero
            values -- fail-closed rather than silently returning `NaN` into
            downstream top-k/early-stop/selection math.
    """
    deltas_by_size = _vval_forward(model, vval, features, device)
    diffs: list[float] = []
    all_deltas: list[float] = []
    margin_hits = 0
    margin_total = 0
    n_skipped = 0
    for size in vval.sizes:
        iu, iv = np.triu_indices(size, k=1)
        for k in range(len(vval.node_ids[size])):
            base = vval.base_logits[size][k].numpy()[iu, iv]
            true = vval.true_adj[size][k].numpy()[iu, iv]
            delta = deltas_by_size[size][k].cpu().numpy()
            all_deltas.extend(delta.tolist())
            margin_hits += int(np.sum(np.abs(delta) > np.abs(base)))
            margin_total += delta.shape[0]
            if len(set(true.tolist())) < 2:
                n_skipped += 1
                continue
            auprc_model = average_precision_score(true, base + delta)
            auprc_base = average_precision_score(true, base)
            diffs.append(float(auprc_model - auprc_base))
    if not diffs:
        raise RuntimeError(
            "S3 _validate: every V_val bucket set is single-class (no positive or no "
            "negative true label) -- ΔAUPRC is undefined for this checkpoint; refusing "
            "to return NaN into top-k/early-stop/selection math (fail-closed)."
        )
    macro_delta_auprc = float(np.mean(diffs))
    delta_variance = float(np.var(all_deltas)) if all_deltas else 0.0
    margin_share = float(margin_hits / margin_total) if margin_total else 0.0
    return macro_delta_auprc, delta_variance, margin_share, len(diffs), n_skipped


# --------------------------------------------------------------------------- five-number topology


def _descriptors(g: nx.Graph) -> dict[str, NDArray[np.float64]]:
    """Degree/clustering/spectral descriptor histograms, keyed like `STATISTICS`."""
    return {
        "degree": degree_histogram(g),
        "clustering": clustering_histogram(g),
        "spectral": laplacian_spectrum_histogram(g),
    }


def _topology_five_numbers(
    model: SetResidualModel,
    vval: VvalEval,
    features: VvalFeatures,
    device: torch.device,
    config: MMDConfig | None = None,
) -> dict[str, float]:
    """BFS-macro GS/RD and the three MMD ratios for one checkpoint against `vval`.

    Each bucket set gets its own density-matched assembled graph (target edge
    count = its true loopless edge count; sets with zero true edges are
    skipped, mirroring the house convention). GS/RD are compared against
    `vval.bucket_ref`'s precomputed reference subgraph (self-loops kept there,
    never in the loopless prediction) and macro-averaged over every kept set
    across every size. The MMD ratio matches
    `evaluate_assembled_graph_with_reference`'s convention: mean raw MMD2
    across sizes over mean odd/even reference-floor MMD2 across sizes, not a
    mean of per-size ratios.

    Returns:
        `{"gs", "rd", "degree_mmd_ratio", "clustering_mmd_ratio",
        "spectral_mmd_ratio"}`, `nan` for any entry with no eligible set.
    """
    cfg = config or MMDConfig()
    deltas_by_size = _vval_forward(model, vval, features, device)

    gs_values: list[float] = []
    rd_values: list[float] = []
    raw_by_size: dict[int, dict[str, float]] = {}
    floor_by_size: dict[int, dict[str, float]] = {}

    for size in vval.sizes:
        iu, iv = np.triu_indices(size, k=1)
        bank_by_stat: dict[str, list[NDArray[np.float64]]] = {stat: [] for stat in STATISTICS}
        ref_by_stat: dict[str, list[NDArray[np.float64]]] = {stat: [] for stat in STATISTICS}
        for k, node_ids in enumerate(vval.node_ids[size]):
            base = vval.base_logits[size][k].numpy()[iu, iv]
            true = vval.true_adj[size][k].numpy()[iu, iv]
            target_edges = int(round(float(true.sum())))
            if target_edges == 0:
                continue
            delta = deltas_by_size[size][k].cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-(base + delta)))
            pairs = [
                (node_ids[i], node_ids[j]) for i, j in zip(iu.tolist(), iv.tolist(), strict=True)
            ]
            threshold = density_matched_threshold(probs, target_edges)
            g_pred = assemble_graph(pairs, probs, threshold=threshold, nodes=node_ids)
            ref_subgraph = vval.bucket_ref.ref_subgraphs[size][k]
            gs_values.append(compute_graph_similarity(g_pred, ref_subgraph))
            rd_values.append(compute_relative_density(g_pred, ref_subgraph))
            bank_d = _descriptors(g_pred)
            ref_d = vval.bucket_ref.ref_descriptors[size][k]
            for stat in STATISTICS:
                bank_by_stat[stat].append(bank_d[stat])
                ref_by_stat[stat].append(ref_d[stat])
        if bank_by_stat[STATISTICS[0]] and len(ref_by_stat[STATISTICS[0]]) >= 2:
            raw_by_size[size] = {
                stat: mmd_squared(bank_by_stat[stat], ref_by_stat[stat], cfg) for stat in STATISTICS
            }
            floor_by_size[size] = {
                stat: mmd_squared(ref_by_stat[stat][::2], ref_by_stat[stat][1::2], cfg)
                for stat in STATISTICS
            }

    mmd_ratio: dict[str, float] = {}
    for stat in STATISTICS:
        raws = [raw_by_size[size][stat] for size in raw_by_size]
        floors = [floor_by_size[size][stat] for size in floor_by_size]
        mmd_ratio[stat] = (
            float(np.mean(raws)) / max(float(np.mean(floors)), cfg.reference_epsilon)
            if raws and floors
            else float("nan")
        )

    return {
        "gs": float(np.mean(gs_values)) if gs_values else float("nan"),
        "rd": float(np.mean(rd_values)) if rd_values else float("nan"),
        "degree_mmd_ratio": mmd_ratio["degree"],
        "clustering_mmd_ratio": mmd_ratio["clustering"],
        "spectral_mmd_ratio": mmd_ratio["spectral"],
    }


# --------------------------------------------------------------------------- selection


def _rank(values: list[float], *, descending: bool) -> list[float]:
    """Rank `values` (1 = best); ties are impossible here (finite, distinct floats expected)."""
    order = sorted(range(len(values)), key=lambda i: -values[i] if descending else values[i])
    ranks = [0.0] * len(values)
    for position, idx in enumerate(order):
        ranks[idx] = float(position + 1)
    return ranks


def _select_best(
    candidates: list[tuple[int, float, dict[str, float]]],
) -> tuple[int, list[dict[str, Any]]]:
    """Mean-rank the top-K checkpoints over `[ΔAUPRC↑, GS↑, |RD-1|↓, 3xMMD↓]`.

    Args:
        candidates: `(epoch, delta_auprc, five_number_topology)` rows, one per
            kept checkpoint.

    Returns:
        `(best_epoch, table)`: the winning epoch (ties broken by the lower
        epoch) and every candidate's row (metrics, per-metric ranks, mean rank).
    """
    rows: list[dict[str, Any]] = []
    for epoch, delta_auprc, five in candidates:
        rows.append(
            {
                "epoch": epoch,
                "delta_auprc": delta_auprc,
                "gs": five["gs"],
                "rd": five["rd"],
                "abs_rd_minus_1": abs(five["rd"] - 1.0),
                "degree_mmd_ratio": five["degree_mmd_ratio"],
                "clustering_mmd_ratio": five["clustering_mmd_ratio"],
                "spectral_mmd_ratio": five["spectral_mmd_ratio"],
            }
        )

    metrics: tuple[tuple[str, bool], ...] = (
        ("delta_auprc", True),
        ("gs", True),
        ("abs_rd_minus_1", False),
        ("degree_mmd_ratio", False),
        ("clustering_mmd_ratio", False),
        ("spectral_mmd_ratio", False),
    )
    rank_columns = [
        _rank([row[name] for row in rows], descending=descending) for name, descending in metrics
    ]
    for i, row in enumerate(rows):
        row["mean_rank"] = float(np.mean([column[i] for column in rank_columns]))

    best = min(rows, key=lambda row: (cast(float, row["mean_rank"]), cast(int, row["epoch"])))
    return int(best["epoch"]), rows


# --------------------------------------------------------------------------- training loop


def _seed(cfg: TrainConfig) -> np.random.Generator:
    """Seed torch's global RNG for model init and return this run's shuffle rng."""
    torch.manual_seed(cfg.seed)
    return np.random.default_rng(cfg.seed)


def _batch_loss(
    delta: torch.Tensor,
    base_logits: torch.Tensor,
    labels: torch.Tensor,
    pairs: torch.Tensor,
    n_sets: int,
) -> torch.Tensor | None:
    """Per-set-mean-then-batch-mean `BCEWithLogitsLoss(base_logit + delta, label)`.

    Grouping by `pairs[:, 0]` (the set row) makes a batch's loss invariant to
    how many pairs any one set happens to contribute -- a set with 500 pairs
    counts exactly as much as a set with 5.

    Returns:
        The scalar batch loss, or `None` if no set in the batch contributed
        any pair (an all-empty batch).
    """
    per_row = F.binary_cross_entropy_with_logits(base_logits + delta, labels, reduction="none")
    set_row = pairs[:, 0]
    sums = torch.zeros(n_sets, device=delta.device, dtype=per_row.dtype).index_add_(
        0, set_row, per_row
    )
    counts = torch.zeros(n_sets, device=delta.device, dtype=per_row.dtype).index_add_(
        0, set_row, torch.ones_like(per_row)
    )
    valid = counts > 0
    if not bool(torch.any(valid)):
        return None
    return (sums[valid] / counts[valid]).mean()


def _save_checkpoint(
    path: Path,
    *,
    model: SetResidualModel,
    model_cfg: ResidualConfig,
    epoch: int,
    delta_auprc: float,
) -> None:
    """Save one checkpoint per the Task 3/4 contract: `model_state` + `config` keys.

    `evaluate.load_residual_checkpoint` hard-requires exactly these two key
    names (`model_state`: the state dict; `config`:
    `dataclasses.asdict(ResidualConfig)`) to load `best.pt`; the extra
    `epoch`/`vval_delta_auprc` fields are S3-internal bookkeeping Task 4 does
    not read.
    """
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": dataclasses.asdict(model_cfg),
            "epoch": epoch,
            "vval_delta_auprc": delta_auprc,
        },
        path,
    )


def _load_model(path: Path, model_cfg: ResidualConfig, device: torch.device) -> SetResidualModel:
    payload = cast(dict[str, object], torch.load(path, map_location="cpu", weights_only=True))
    model = SetResidualModel(model_cfg).to(device)
    model.load_state_dict(cast(dict[str, torch.Tensor], payload["model_state"]))
    model.eval()
    return model


def train_arm(
    corpus: S3Corpus,
    vval: VvalEval,
    vval_features: VvalFeatures,
    model_cfg: ResidualConfig,
    cfg: TrainConfig,
    *,
    run_dir: Path,
) -> None:
    """Train one S3 arm end to end: loop, per-epoch V_val selection, topology-ranked publish.

    Writes `run_dir/metrics.jsonl` (one record per epoch, epoch 0 = the
    pre-training sanity pass), `run_dir/last.pt`, up to `cfg.top_k_checkpoints`
    `run_dir/epoch_<E>.pt` files (pruned as the ΔAUPRC frontier moves),
    `run_dir/best.pt` (a copy of the topology-selected checkpoint), and
    `run_dir/run_metadata.json` (arm, seed, base-scorer `checkpoint_id`, and
    the full selection table).

    Args:
        corpus: The V_val-masked training corpus (`build_s3_corpus`).
        vval: The V_val selection buckets (`build_vval_eval`).
        vval_features: Feature matrix for every node `vval` references
            (`build_vval_features`).
        model_cfg: `SetResidualModel` architecture (arm and dims).
        cfg: Training-loop hyperparameters.
        run_dir: Output directory for checkpoints, metrics, and metadata.

    Raises:
        RuntimeError: If any batch's training loss is non-finite.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    rng = _seed(cfg)

    model = SetResidualModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    metrics_path = run_dir / "metrics.jsonl"
    last_path = run_dir / "last.pt"

    # Epoch 0: pre-training sanity pass. At zero-init `delta` is exactly 0 for every
    # pair, so `base + delta` is bit-identical to `base` and this ΔAUPRC is exactly 0.0.
    delta_auprc0, delta_var0, margin0, n_used0, n_skip0 = _validate(
        model, vval, vval_features, device
    )
    append_jsonl(
        metrics_path,
        {
            "epoch": 0,
            "train_loss": None,
            "vval_delta_auprc": delta_auprc0,
            "delta_variance": delta_var0,
            "margin_share": margin0,
            "n_sets_used": n_used0,
            "n_sets_skipped_single_class": n_skip0,
        },
    )

    top_k: list[tuple[int, float]] = []  # (epoch, delta_auprc), sorted desc by delta_auprc
    best_seen = -math.inf
    epochs_since_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses: list[tuple[float, int]] = []
        for region_indices in iter_size_batches(
            corpus.base, corpus.base.train_idx, batch_sets=cfg.batch_regions, rng=rng
        ):
            region_batch, pairs, labels, base_logits = collate_pair_batch(
                corpus, region_indices, device
            )
            if pairs.numel() == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            delta = model(region_batch.x, region_batch.mask, pairs)
            loss = _batch_loss(delta, base_logits, labels, pairs, len(region_indices))
            if loss is None:
                continue
            loss_value = float(loss.detach())
            if not math.isfinite(loss_value):
                raise RuntimeError(f"non-finite training loss at epoch={epoch}: {loss_value}")
            loss.backward()  # type: ignore[no-untyped-call]
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_losses.append((loss_value, len(region_indices)))

        total_weight = sum(weight for _, weight in train_losses)
        train_loss_mean = (
            sum(value * weight for value, weight in train_losses) / total_weight
            if total_weight
            else float("nan")
        )

        delta_auprc, delta_var, margin_share, n_used, n_skip = _validate(
            model, vval, vval_features, device
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_loss_mean,
                "vval_delta_auprc": delta_auprc,
                "delta_variance": delta_var,
                "margin_share": margin_share,
                "n_sets_used": n_used,
                "n_sets_skipped_single_class": n_skip,
            },
        )

        _save_checkpoint(
            last_path, model=model, model_cfg=model_cfg, epoch=epoch, delta_auprc=delta_auprc
        )

        if len(top_k) < cfg.top_k_checkpoints or delta_auprc > min(v for _, v in top_k):
            _save_checkpoint(
                run_dir / f"epoch_{epoch}.pt",
                model=model,
                model_cfg=model_cfg,
                epoch=epoch,
                delta_auprc=delta_auprc,
            )
            top_k.append((epoch, delta_auprc))
            top_k.sort(key=lambda item: -item[1])
            while len(top_k) > cfg.top_k_checkpoints:
                dropped_epoch, _ = top_k.pop()
                (run_dir / f"epoch_{dropped_epoch}.pt").unlink(missing_ok=True)

        if delta_auprc > best_seen:
            best_seen = delta_auprc
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        if epochs_since_improve >= cfg.patience:
            logger.info("early stopping at epoch=%d (patience=%d)", epoch, cfg.patience)
            break

    candidates: list[tuple[int, float, dict[str, float]]] = []
    for epoch, delta_auprc in sorted(top_k):
        checkpoint_model = _load_model(run_dir / f"epoch_{epoch}.pt", model_cfg, device)
        five = _topology_five_numbers(checkpoint_model, vval, vval_features, device)
        candidates.append((epoch, delta_auprc, five))

    best_epoch, table = _select_best(candidates)
    checkpoint_payload = cast(
        dict[str, object],
        torch.load(run_dir / f"epoch_{best_epoch}.pt", map_location="cpu", weights_only=True),
    )
    torch.save(checkpoint_payload, run_dir / "best.pt")

    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "arm": model_cfg.mode,
                "seed": cfg.seed,
                "checkpoint_id": vval.checkpoint_id,
                "published_epoch": best_epoch,
                "selection_table": table,
                "model_config": dataclasses.asdict(model_cfg),
                "train_config": dataclasses.asdict(cfg),
                "torch_version": torch.__version__,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("S3 %s: published epoch=%d -> %s", model_cfg.mode, best_epoch, run_dir / "best.pt")

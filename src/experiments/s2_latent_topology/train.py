"""S2 trainers: Stage A graph AE, Stage B flow prior, deterministic twin, degree baseline.

Four independent training entrypoints, one per `docs/tmp/s2_set_generation_experiment.md`
arm: `train_ae` (Stage A `RegionAutoencoder`, the latent ceiling), `train_prior`
(Stage B `SetDenoiser` flow-matching prior over a frozen AE's latents), `train_det`
(`DetTwin`, the deterministic twin isolating generation from set encoding), and
`train_marg` (`MargDegreeRegressor`, the strictly per-node independence baseline).
All CPU-friendly fp32, single device, no autocast, no DDP. Each trainer keeps the
best (lowest val-total-loss) `state_dict`, writes one `append_jsonl` telemetry
record per epoch, and saves a single checkpoint at the end; only non-finite
epoch losses raise. `evaluate.py` consumes the checkpoint contract documented on
each trainer and reconstructs models via `TrainConfig` + `in_dim` alone.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from src.experiments.s2_latent_topology.data import RegionCorpus, collate_regions, iter_size_batches
from src.experiments.s2_latent_topology.model import (
    DetTwin,
    MargDegreeRegressor,
    RegionAutoencoder,
    SetDenoiser,
    ae_losses,
    flow_matching_loss,
)

__all__ = [
    "TrainConfig",
    "append_jsonl",
    "load_checkpoint",
    "train_ae",
    "train_det",
    "train_marg",
    "train_prior",
]

_STAGE_AE = 1
_STAGE_PRIOR = 2
_STAGE_DET = 3
_STAGE_MARG = 4


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters shared by all four S2 trainers.

    Attributes:
        epochs: Number of training epochs.
        batch_sets: Max regions per size-pure batch (`iter_size_batches`).
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        grad_clip: Max gradient norm (`clip_grad_norm_`).
        seed: Base seed; combined with a per-stage code for reproducibility.
        device: Torch device string (e.g. `"cpu"`).
        z_dim: Latent role width shared by the AE, prior, and det twin.
        ae_dim: `RegionAutoencoder` GRIT hidden width.
        ae_layers: `RegionAutoencoder` GRIT layer count.
        ae_rrwp_k: `RegionAutoencoder` RRWP positional-encoding order.
        ae_heads: `RegionAutoencoder` GRIT attention head count.
        ae_seeds: `RegionAutoencoder` GMT pooling seed count.
        ae_d_model: `RegionAutoencoder` GRIT token width.
        lambda_tri: AE triangle-moment penalty weight.
        set_width: `_SetBackbone` hidden width (prior + det twin).
        set_layers: `_SetBackbone` SAB layer count.
        set_heads: `_SetBackbone` attention head count.
        set_pma_seeds: `_SetBackbone` PMA seed count.
        time_dim: `SetDenoiser` flow-time embedding width.
        p_drop: `SetDenoiser` classifier-free conditioning dropout probability.
        lambda_edge: Det twin's decoded-edge BCE weight.
        marg_hidden: `MargDegreeRegressor` MLP hidden width.
    """

    epochs: int = 30
    batch_sets: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cpu"
    z_dim: int = 16
    ae_dim: int = 96
    ae_layers: int = 4
    ae_rrwp_k: int = 8
    ae_heads: int = 8
    ae_seeds: int = 4
    ae_d_model: int = 512
    lambda_tri: float = 0.1
    set_width: int = 384
    set_layers: int = 4
    set_heads: int = 4
    set_pma_seeds: int = 4
    time_dim: int = 64
    p_drop: float = 0.1
    lambda_edge: float = 1.0
    marg_hidden: int = 256


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    """Append one JSON-encoded record as a line to `path`.

    Args:
        path: Target `.jsonl` file; created (with parents) if absent, else
            appended to.
        record: JSON-serializable telemetry record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def load_checkpoint(path: Path) -> dict[str, object]:
    """Load a checkpoint dict written by one of this module's trainers.

    Args:
        path: Checkpoint file path.

    Returns:
        The plain dict payload, all tensors on CPU.
    """
    return cast(dict[str, object], torch.load(path, map_location="cpu"))


def _seed(cfg: TrainConfig, stage_code: int) -> np.random.Generator:
    """Seed torch's global RNG for model init and return this run's shuffle rng."""
    torch.manual_seed(cfg.seed * 100 + stage_code)
    return np.random.default_rng((cfg.seed, stage_code))


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Deep-copy `model`'s state dict to detached CPU tensors."""
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _weighted_mean(values: list[tuple[float, int]]) -> float:
    """Weighted mean of `(value, weight)` pairs; `nan` if the total weight is zero."""
    total_weight = sum(weight for _, weight in values)
    if total_weight == 0:
        return float("nan")
    return sum(value * weight for value, weight in values) / total_weight


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over valid (`mask == 1`) entries, `mask` broadcast to `pred`."""
    valid = mask
    while valid.dim() < pred.dim():
        valid = valid.unsqueeze(-1)
    valid = valid.expand_as(pred).to(pred.dtype)
    count = valid.sum().clamp(min=1.0)
    return ((pred - target) ** 2 * valid).sum() / count


def _edge_density_ratios(
    logits: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Per-region ratio of soft predicted edge count to true edge count.

    Args:
        logits: `(B, N, N)` decoder logits.
        adj: `(B, N, N)` 0/1 true adjacency, self-loop-free.
        mask: `(B, N)` 0/1 node existence.

    Returns:
        `(B,)` ratio of summed sigmoid probability to summed true edge count
        over valid, strict-upper-triangle pairs; true-edge counts are clamped
        to avoid division by zero on edgeless regions.
    """
    n = logits.shape[-1]
    valid = mask > 0.0
    pairmask = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    triu = torch.triu(torch.ones(n, n, dtype=torch.bool, device=logits.device), diagonal=1)
    upper = (pairmask & triu).to(logits.dtype)
    true_edges = (adj * upper).sum(dim=(-2, -1))
    pred_edges = (torch.sigmoid(logits) * upper).sum(dim=(-2, -1))
    return pred_edges / true_edges.clamp(min=1e-6)


def _build_ae(cfg: TrainConfig, in_dim: int) -> RegionAutoencoder:
    """Construct a `RegionAutoencoder` sized per `cfg` for `in_dim`-wide features."""
    return RegionAutoencoder(
        in_dim=in_dim,
        z_dim=cfg.z_dim,
        dim=cfg.ae_dim,
        layers=cfg.ae_layers,
        rrwp_k=cfg.ae_rrwp_k,
        n_heads=cfg.ae_heads,
        seeds=cfg.ae_seeds,
        d_model=cfg.ae_d_model,
    )


def _build_denoiser(cfg: TrainConfig, in_dim: int) -> SetDenoiser:
    """Construct a `SetDenoiser` sized per `cfg` for `in_dim`-wide features."""
    return SetDenoiser(
        in_dim=in_dim,
        z_dim=cfg.z_dim,
        width=cfg.set_width,
        sab_layers=cfg.set_layers,
        heads=cfg.set_heads,
        pma_seeds=cfg.set_pma_seeds,
        time_dim=cfg.time_dim,
    )


def _build_det(cfg: TrainConfig, in_dim: int) -> DetTwin:
    """Construct a `DetTwin` sized per `cfg` for `in_dim`-wide features."""
    return DetTwin(
        in_dim=in_dim,
        z_dim=cfg.z_dim,
        width=cfg.set_width,
        sab_layers=cfg.set_layers,
        heads=cfg.set_heads,
        pma_seeds=cfg.set_pma_seeds,
    )


def _build_marg(cfg: TrainConfig, in_dim: int) -> MargDegreeRegressor:
    """Construct a `MargDegreeRegressor` sized per `cfg` for `in_dim`-wide features."""
    return MargDegreeRegressor(in_dim=in_dim, hidden=cfg.marg_hidden)


def _pad_latents(
    z1_by_region: list[torch.Tensor], indices: Sequence[int], max_n: int, device: torch.device
) -> torch.Tensor:
    """Pad per-region unpadded latent tensors to a dense `(B, max_n, D)` batch."""
    dim = z1_by_region[indices[0]].shape[-1]
    out = torch.zeros(len(indices), max_n, dim, dtype=torch.float32, device=device)
    for row, idx in enumerate(indices):
        z = z1_by_region[idx]
        out[row, : z.shape[0]] = z.to(device)
    return out


def _load_frozen_ae(
    ae_path: Path, device: torch.device
) -> tuple[RegionAutoencoder, torch.Tensor, torch.Tensor]:
    """Load, rebuild, and freeze a `RegionAutoencoder` from a `train_ae` checkpoint.

    Args:
        ae_path: Path to a checkpoint written by `train_ae`.
        device: Device the returned model and latent stats are placed on.

    Returns:
        `(ae, latent_mean, latent_std)`: the frozen (`eval`, no-grad) model and
        its stored latent standardization stats, both on `device`.
    """
    payload = load_checkpoint(ae_path)
    ae_cfg = TrainConfig(**cast(dict[str, Any], payload["config"]))
    in_dim = cast(int, payload["in_dim"])
    ae = _build_ae(ae_cfg, in_dim).to(device)
    ae.load_state_dict(cast(dict[str, torch.Tensor], payload["model"]))
    ae.eval()
    ae.requires_grad_(False)
    latent_mean = cast(torch.Tensor, payload["latent_mean"]).to(device)
    latent_std = cast(torch.Tensor, payload["latent_std"]).to(device)
    return ae, latent_mean, latent_std


def _precompute_standardized_z1(
    ae: RegionAutoencoder,
    corpus: RegionCorpus,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    *,
    batch_sets: int,
    rng: np.random.Generator,
    device: torch.device,
) -> list[torch.Tensor]:
    """Encode every corpus region once and standardize with the frozen AE's stats.

    Args:
        ae: Frozen `RegionAutoencoder` to encode with.
        corpus: Region corpus; every region (`train_idx` and `val_idx`) is encoded.
        latent_mean: `(1 + z_dim,)` standardization mean.
        latent_std: `(1 + z_dim,)` standardization std.
        batch_sets: Batch cap passed through to `iter_size_batches`.
        rng: Shuffle generator; consumes draws from the trainer's shared rng.
        device: Device the encode pass runs on.

    Returns:
        Per-region unpadded `(n_i, 1 + z_dim)` standardized latent tensors,
        indexed by region id.
    """
    z1_by_region: list[torch.Tensor | None] = [None] * len(corpus.regions)
    indices = list(range(len(corpus.regions)))
    with torch.no_grad():
        for batch_indices in iter_size_batches(corpus, indices, batch_sets=batch_sets, rng=rng):
            batch = collate_regions(corpus, batch_indices, device)
            latents = ae.encode(batch.x, batch.adj, batch.mask)
            standardized = (latents - latent_mean) / latent_std
            for row, idx in enumerate(batch_indices):
                n = int(batch.n[row])
                z1_by_region[idx] = standardized[row, :n].detach().clone()
    return cast(list[torch.Tensor], z1_by_region)


def _compute_latent_stats(
    model: RegionAutoencoder,
    corpus: RegionCorpus,
    *,
    batch_sets: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-dimension latent mean/std over `corpus.train_idx` node latents.

    Args:
        model: `RegionAutoencoder` with the (already-restored best) weights to
            standardize against.
        corpus: Region corpus to draw `train_idx` regions from.
        batch_sets: Batch cap passed through to `iter_size_batches`.
        rng: Shuffle generator; consumes draws from the trainer's shared rng.
        device: Device the encode pass runs on.

    Returns:
        `(mean, std)`, each shape `(1 + z_dim,)` on CPU; `std` clamped to `>= 1e-6`.
    """
    model.eval()
    latents: list[torch.Tensor] = []
    with torch.no_grad():
        for indices in iter_size_batches(corpus, corpus.train_idx, batch_sets=batch_sets, rng=rng):
            batch = collate_regions(corpus, indices, device)
            encoded = model.encode(batch.x, batch.adj, batch.mask)
            for row, n in enumerate(batch.n.tolist()):
                latents.append(encoded[row, :n])
    stacked = torch.cat(latents, dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0, unbiased=False).clamp(min=1e-6)
    return mean.cpu(), std.cpu()


def train_ae(corpus: RegionCorpus, cfg: TrainConfig, *, out_path: Path, metrics_path: Path) -> None:
    """Train Stage A's `RegionAutoencoder` and save the best checkpoint.

    Trains `ae_losses`' BCE + triangle-moment total against `corpus.train_idx`,
    validates on `corpus.val_idx` each epoch, and keeps the lowest-val-total
    state. After restoring the best weights, computes latent standardization
    statistics over `corpus.train_idx` and writes the `ae.pt` checkpoint contract.

    Args:
        corpus: Region corpus to train on.
        cfg: Training hyperparameters.
        out_path: Checkpoint output path.
        metrics_path: JSONL metrics path; one record appended per epoch.

    Raises:
        RuntimeError: If any epoch's aggregate train or val loss is non-finite.
    """
    device = torch.device(cfg.device)
    rng = _seed(cfg, _STAGE_AE)
    in_dim = corpus.features.shape[1]
    model = _build_ae(cfg, in_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state = _clone_state(model)
    best_val = math.inf
    best_epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_total: list[tuple[float, int]] = []
        train_bce: list[tuple[float, int]] = []
        train_tri: list[tuple[float, int]] = []
        for indices in iter_size_batches(
            corpus, corpus.train_idx, batch_sets=cfg.batch_sets, rng=rng
        ):
            batch = collate_regions(corpus, indices, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.decode(model.encode(batch.x, batch.adj, batch.mask), batch.mask)
            losses = ae_losses(logits, batch.adj, batch.mask, lambda_tri=cfg.lambda_tri)
            losses["total"].backward()  # type: ignore[no-untyped-call]
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            weight = len(indices)
            train_total.append((float(losses["total"].detach()), weight))
            train_bce.append((float(losses["bce"].detach()), weight))
            train_tri.append((float(losses["triangle"].detach()), weight))

        train_total_mean = _weighted_mean(train_total)
        if not math.isfinite(train_total_mean):
            raise RuntimeError(f"non-finite train loss in stage=ae epoch={epoch}")

        model.eval()
        val_total: list[tuple[float, int]] = []
        val_bce: list[tuple[float, int]] = []
        val_tri: list[tuple[float, int]] = []
        liveness: list[float] = []
        with torch.no_grad():
            for indices in iter_size_batches(
                corpus, corpus.val_idx, batch_sets=cfg.batch_sets, rng=rng
            ):
                batch = collate_regions(corpus, indices, device)
                logits = model.decode(model.encode(batch.x, batch.adj, batch.mask), batch.mask)
                losses = ae_losses(logits, batch.adj, batch.mask, lambda_tri=cfg.lambda_tri)
                weight = len(indices)
                val_total.append((float(losses["total"]), weight))
                val_bce.append((float(losses["bce"]), weight))
                val_tri.append((float(losses["triangle"]), weight))
                liveness.extend(_edge_density_ratios(logits, batch.adj, batch.mask).tolist())

        val_total_mean = _weighted_mean(val_total)
        if not math.isfinite(val_total_mean):
            raise RuntimeError(f"non-finite val loss in stage=ae epoch={epoch}")

        append_jsonl(
            metrics_path,
            {
                "stage": "ae",
                "epoch": epoch,
                "train_total": train_total_mean,
                "val_total": val_total_mean,
                "lr": cfg.lr,
                "train_bce": _weighted_mean(train_bce),
                "train_triangle": _weighted_mean(train_tri),
                "val_bce": _weighted_mean(val_bce),
                "val_triangle": _weighted_mean(val_tri),
                "edge_density_liveness": (
                    sum(liveness) / len(liveness) if liveness else float("nan")
                ),
            },
        )

        if val_total_mean < best_val:
            best_val = val_total_mean
            best_epoch = epoch
            best_state = _clone_state(model)

    model.load_state_dict(best_state)
    latent_mean, latent_std = _compute_latent_stats(
        model, corpus, batch_sets=cfg.batch_sets, rng=rng, device=device
    )

    torch.save(
        {
            "model": best_state,
            "config": dataclasses.asdict(cfg),
            "in_dim": in_dim,
            "latent_mean": latent_mean,
            "latent_std": latent_std,
            "best_epoch": best_epoch,
            "val_loss": best_val,
        },
        out_path,
    )


def train_prior(
    corpus: RegionCorpus, cfg: TrainConfig, *, ae_path: Path, out_path: Path, metrics_path: Path
) -> None:
    """Train Stage B's `SetDenoiser` flow-matching prior against a frozen AE.

    Loads and freezes the `RegionAutoencoder` at `ae_path`, precomputes every
    corpus region's standardized latent target once, then trains the denoiser
    on `corpus.train_idx` / validates on `corpus.val_idx` each epoch with
    `flow_matching_loss`, keeping the lowest-val-total state.

    Args:
        corpus: Region corpus to train on.
        cfg: Training hyperparameters.
        ae_path: Path to a checkpoint written by `train_ae`.
        out_path: Checkpoint output path.
        metrics_path: JSONL metrics path; one record appended per epoch.

    Raises:
        FileNotFoundError: If `ae_path` does not exist.
        RuntimeError: If any epoch's aggregate train or val loss is non-finite.
    """
    device = torch.device(cfg.device)
    rng = _seed(cfg, _STAGE_PRIOR)
    in_dim = corpus.features.shape[1]
    ae, latent_mean, latent_std = _load_frozen_ae(ae_path, device)
    z1_by_region = _precompute_standardized_z1(
        ae, corpus, latent_mean, latent_std, batch_sets=cfg.batch_sets, rng=rng, device=device
    )

    model = _build_denoiser(cfg, in_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state = _clone_state(model)
    best_val = math.inf
    best_epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        gen = torch.Generator(device=cfg.device).manual_seed(cfg.seed * 1000 + epoch)
        train_total: list[tuple[float, int]] = []
        for indices in iter_size_batches(
            corpus, corpus.train_idx, batch_sets=cfg.batch_sets, rng=rng
        ):
            batch = collate_regions(corpus, indices, device)
            z1_std = _pad_latents(z1_by_region, indices, batch.x.shape[1], device)
            optimizer.zero_grad(set_to_none=True)
            loss = flow_matching_loss(
                model, z1_std, batch.x, batch.mask, p_drop=cfg.p_drop, generator=gen
            )
            loss.backward()  # type: ignore[no-untyped-call]
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_total.append((float(loss.detach()), len(indices)))

        train_total_mean = _weighted_mean(train_total)
        if not math.isfinite(train_total_mean):
            raise RuntimeError(f"non-finite train loss in stage=prior epoch={epoch}")

        model.eval()
        gen_val = torch.Generator(device=cfg.device).manual_seed(cfg.seed * 1000 + epoch + 500)
        val_total: list[tuple[float, int]] = []
        with torch.no_grad():
            for indices in iter_size_batches(
                corpus, corpus.val_idx, batch_sets=cfg.batch_sets, rng=rng
            ):
                batch = collate_regions(corpus, indices, device)
                z1_std = _pad_latents(z1_by_region, indices, batch.x.shape[1], device)
                loss = flow_matching_loss(
                    model, z1_std, batch.x, batch.mask, p_drop=cfg.p_drop, generator=gen_val
                )
                val_total.append((float(loss), len(indices)))

        val_total_mean = _weighted_mean(val_total)
        if not math.isfinite(val_total_mean):
            raise RuntimeError(f"non-finite val loss in stage=prior epoch={epoch}")

        append_jsonl(
            metrics_path,
            {
                "stage": "prior",
                "epoch": epoch,
                "train_total": train_total_mean,
                "val_total": val_total_mean,
                "lr": cfg.lr,
            },
        )

        if val_total_mean < best_val:
            best_val = val_total_mean
            best_epoch = epoch
            best_state = _clone_state(model)

    model.load_state_dict(best_state)
    torch.save(
        {
            "model": best_state,
            "config": dataclasses.asdict(cfg),
            "in_dim": in_dim,
            "best_epoch": best_epoch,
            "val_loss": best_val,
        },
        out_path,
    )


def train_det(
    corpus: RegionCorpus, cfg: TrainConfig, *, ae_path: Path, out_path: Path, metrics_path: Path
) -> None:
    """Train the deterministic twin `DetTwin` against a frozen AE's latents.

    Same frozen-AE setup and precomputed standardized latents as `train_prior`;
    loss is masked latent MSE plus `cfg.lambda_edge` times the decoded-edge BCE
    against the true adjacency, keeping the lowest-val-total state.

    Args:
        corpus: Region corpus to train on.
        cfg: Training hyperparameters.
        ae_path: Path to a checkpoint written by `train_ae`.
        out_path: Checkpoint output path.
        metrics_path: JSONL metrics path; one record appended per epoch.

    Raises:
        FileNotFoundError: If `ae_path` does not exist.
        RuntimeError: If any epoch's aggregate train or val loss is non-finite.
    """
    device = torch.device(cfg.device)
    rng = _seed(cfg, _STAGE_DET)
    in_dim = corpus.features.shape[1]
    ae, latent_mean, latent_std = _load_frozen_ae(ae_path, device)
    z1_by_region = _precompute_standardized_z1(
        ae, corpus, latent_mean, latent_std, batch_sets=cfg.batch_sets, rng=rng, device=device
    )

    model = _build_det(cfg, in_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state = _clone_state(model)
    best_val = math.inf
    best_epoch = 0

    def _step(indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = collate_regions(corpus, indices, device)
        target = _pad_latents(z1_by_region, indices, batch.x.shape[1], device)
        pred = model(batch.x, batch.mask)
        mse = _masked_mse(pred, target, batch.mask)
        decoded = ae.decode(pred * latent_std + latent_mean, batch.mask)
        bce = ae_losses(decoded, batch.adj, batch.mask, lambda_tri=0.0)["bce"]
        return mse + cfg.lambda_edge * bce, mse, bce

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_total: list[tuple[float, int]] = []
        train_mse: list[tuple[float, int]] = []
        train_bce: list[tuple[float, int]] = []
        for indices in iter_size_batches(
            corpus, corpus.train_idx, batch_sets=cfg.batch_sets, rng=rng
        ):
            optimizer.zero_grad(set_to_none=True)
            total, mse, bce = _step(indices)
            total.backward()  # type: ignore[no-untyped-call]
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            weight = len(indices)
            train_total.append((float(total.detach()), weight))
            train_mse.append((float(mse.detach()), weight))
            train_bce.append((float(bce.detach()), weight))

        train_total_mean = _weighted_mean(train_total)
        if not math.isfinite(train_total_mean):
            raise RuntimeError(f"non-finite train loss in stage=det epoch={epoch}")

        model.eval()
        val_total: list[tuple[float, int]] = []
        val_mse: list[tuple[float, int]] = []
        val_bce: list[tuple[float, int]] = []
        with torch.no_grad():
            for indices in iter_size_batches(
                corpus, corpus.val_idx, batch_sets=cfg.batch_sets, rng=rng
            ):
                total, mse, bce = _step(indices)
                weight = len(indices)
                val_total.append((float(total), weight))
                val_mse.append((float(mse), weight))
                val_bce.append((float(bce), weight))

        val_total_mean = _weighted_mean(val_total)
        if not math.isfinite(val_total_mean):
            raise RuntimeError(f"non-finite val loss in stage=det epoch={epoch}")

        append_jsonl(
            metrics_path,
            {
                "stage": "det",
                "epoch": epoch,
                "train_total": train_total_mean,
                "val_total": val_total_mean,
                "lr": cfg.lr,
                "train_mse": _weighted_mean(train_mse),
                "train_bce": _weighted_mean(train_bce),
                "val_mse": _weighted_mean(val_mse),
                "val_bce": _weighted_mean(val_bce),
            },
        )

        if val_total_mean < best_val:
            best_val = val_total_mean
            best_epoch = epoch
            best_state = _clone_state(model)

    model.load_state_dict(best_state)
    torch.save(
        {
            "model": best_state,
            "config": dataclasses.asdict(cfg),
            "in_dim": in_dim,
            "best_epoch": best_epoch,
            "val_loss": best_val,
        },
        out_path,
    )


def train_marg(
    corpus: RegionCorpus, cfg: TrainConfig, *, out_path: Path, metrics_path: Path
) -> None:
    """Train `MargDegreeRegressor`, the strictly per-node independence baseline.

    Target is each valid node's true in-set loopless degree (`batch.adj.sum(-1)`);
    loss is masked MSE, keeping the lowest-val-total state.

    Args:
        corpus: Region corpus to train on.
        cfg: Training hyperparameters.
        out_path: Checkpoint output path.
        metrics_path: JSONL metrics path; one record appended per epoch.

    Raises:
        RuntimeError: If any epoch's aggregate train or val loss is non-finite.
    """
    device = torch.device(cfg.device)
    rng = _seed(cfg, _STAGE_MARG)
    in_dim = corpus.features.shape[1]
    model = _build_marg(cfg, in_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state = _clone_state(model)
    best_val = math.inf
    best_epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_total: list[tuple[float, int]] = []
        for indices in iter_size_batches(
            corpus, corpus.train_idx, batch_sets=cfg.batch_sets, rng=rng
        ):
            batch = collate_regions(corpus, indices, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch.x, batch.n)
            degree = batch.adj.sum(-1)
            loss = _masked_mse(pred, degree, batch.mask)
            loss.backward()  # type: ignore[no-untyped-call]
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_total.append((float(loss.detach()), len(indices)))

        train_total_mean = _weighted_mean(train_total)
        if not math.isfinite(train_total_mean):
            raise RuntimeError(f"non-finite train loss in stage=marg epoch={epoch}")

        model.eval()
        val_total: list[tuple[float, int]] = []
        with torch.no_grad():
            for indices in iter_size_batches(
                corpus, corpus.val_idx, batch_sets=cfg.batch_sets, rng=rng
            ):
                batch = collate_regions(corpus, indices, device)
                pred = model(batch.x, batch.n)
                degree = batch.adj.sum(-1)
                loss = _masked_mse(pred, degree, batch.mask)
                val_total.append((float(loss), len(indices)))

        val_total_mean = _weighted_mean(val_total)
        if not math.isfinite(val_total_mean):
            raise RuntimeError(f"non-finite val loss in stage=marg epoch={epoch}")

        append_jsonl(
            metrics_path,
            {
                "stage": "marg",
                "epoch": epoch,
                "train_total": train_total_mean,
                "val_total": val_total_mean,
                "lr": cfg.lr,
            },
        )

        if val_total_mean < best_val:
            best_val = val_total_mean
            best_epoch = epoch
            best_state = _clone_state(model)

    model.load_state_dict(best_state)
    torch.save(
        {
            "model": best_state,
            "config": dataclasses.asdict(cfg),
            "in_dim": in_dim,
            "best_epoch": best_epoch,
            "val_loss": best_val,
        },
        out_path,
    )

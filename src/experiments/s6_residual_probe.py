r"""S6: pair-residual identifiability probe -- is the beyond-content residual learnable?

Every pointwise residual-KD arm (D5 in particular) asks the student to predict
``Delta = teacher_logit - content_logit``: what the full-ego oracle teacher knows
about a pair beyond what a content-only model already explains. Those arms moved
GS-BFS by at most +0.0017. This module separates two explanations that the KD
runs cannot tell apart:

- **Not identifiable.** ``Delta`` is not a function of ``(x_u, x_v)`` at all, so no
  pointwise arm can ever fit it. Held-out R^2 near or below the predict-zero
  baseline, at full capacity and with a warm start, upper-bounds every pointwise
  residual-KD arm and protects the D5 claim.
- **Not optimized.** ``Delta`` is learnable and D5 simply failed to learn it.
  Held-out R^2 well above zero says the KD arm's failure was an optimization
  problem and is worth revisiting.

The probe therefore trains the *full* `V3_1` student directly on ``Delta`` with a
regression objective -- the most favourable setting a pointwise arm could ever
get -- rather than adding a new head class.

Zero-init control: the final ``output_head`` linear is zeroed so the model's
first prediction is exactly the predict-zero baseline, making "did training
help at all" answerable without a separate run. The published v3_1 checkpoints
carry ``mlp_head.spectral_norm: True``, and a spectral-normed linear cannot be
zeroed in place -- ``weight = weight_orig / sigma`` becomes ``0 / 0 = NaN``.
Stripping it with `torch.nn.utils.remove_spectral_norm` is also unsafe: it
leaves a stale ``_load_state_dict_pre_hook`` that keeps demanding
``weight_orig``/``weight_u``, so every later `load_state_dict` raises. That one
final linear is therefore *replaced* with a fresh zeroed `torch.nn.Linear`
(recorded in the report). This changes only the probe's own head, never the
shipped classifier.

Data legality: rows come from an already train-side-legal KD-targets artifact;
`assert_training_side_only` is re-run defensively against the artifact's own
node universe (the `content_logit.py` precedent), not as a new gate. The anchor
split is by *anchor*, never by row: every CSR row of a held-out anchor goes to
the eval side, so no anchor's neighbourhood is split across the boundary.
Training rows are further pruned to exclude any row whose *partner* endpoint is
a held-out anchor: the classifier's symmetric `abba_max` readout makes `(u, v)`
and `(v, u)` the same pair, so a held-out anchor appearing as a training-row
partner would leak that anchor into training via its reciprocal row. A held-out
anchor therefore never appears on the training side, as anchor or as partner;
the count of rows this drops is disclosed, never silently absorbed.

Precision: fp32 everywhere, no autocast, single GPU, no DDP.

CLI::

    python -m src.experiments.s6_residual_probe \
        --targets outputs/distill/kd_targets_breadth_first_seed0_v3 \
        --checkpoint outputs/deliverables/b0_v31_breadth_first_20260711/model/best.pt \
        --data-root data --strategy breadth_first \
        --init warm --seed 0 --output-dir outputs/s6_residual_probe/warm_s0
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import spearmanr
from torch import nn

from src.data.features import FeatureStore
from src.data.pairs import (
    LengthBucketedBatchSampler,
    TokenPairDataset,
    collate_token_pairs,
    probe_lengths,
)
from src.distill.artifacts import KDTargets, load_kd_targets
from src.distill.teacher_targets import (
    _FEATURES_SUBDIR,
    _load_val_region_split,
    assert_training_side_only,
    truth_graph_for_kd,
)
from src.model.egostitch.classifier.b0_v31 import V3_1
from src.score_universe import _load_checkpoint, build_model

logger = logging.getLogger(__name__)

S6_REPORT_FORMAT = "s6_residual_probe_v1"

_HOLDOUT_FRACTION = 0.1
_TOKEN_BUDGET = 32_768
_MAX_EPOCHS = 60
_PATIENCE = 8
_WEIGHT_DECAY = 0.01
_GRAD_CLIP = 1.0
_LR_WARM = 1e-5
_LR_SCRATCH = 1e-4


# --------------------------------------------------------------------------- targets


def residual_targets(targets: KDTargets) -> NDArray[np.float32]:
    """Compute ``Delta = teacher_logit - content_logit`` in fp32.

    Args:
        targets: A loaded KD-targets artifact.

    Returns:
        Shape ``(n_pairs,)`` float32 residual targets.

    Raises:
        ValueError: If the artifact carries no ``content_logit`` (a v2 artifact);
            the residual is undefined without it.
    """
    if targets.content_logit is None:
        raise ValueError(
            "KD targets artifact has no 'content_logit' array (v2 artifact); S6 needs a v3 "
            "artifact written by src.distill.content_logit to form the residual target"
        )
    delta = np.asarray(targets.teacher_logit, dtype=np.float32) - np.asarray(
        targets.content_logit, dtype=np.float32
    )
    return delta.astype(np.float32)


@dataclass(frozen=True)
class AnchorSplit:
    """Anchor-disjoint train/eval row split, plus the training-side leakage guard.

    A held-out anchor never appears on the training side, as anchor or as
    partner: `train_rows` already excludes rows whose partner endpoint is
    held out, and `n_dropped_touching_heldout` discloses exactly how many rows
    that exclusion removed.
    """

    train_rows: NDArray[np.int64]
    eval_rows: NDArray[np.int64]
    heldout_anchors: NDArray[np.int64]
    n_dropped_touching_heldout: int


def anchor_holdout_split(
    anchor_offsets: NDArray[np.integer],
    pair_partner_idx: NDArray[np.integer],
    *,
    seed: int,
    fraction: float = _HOLDOUT_FRACTION,
) -> AnchorSplit:
    """Split CSR rows by *anchor*, holding out the last `fraction` of a permutation.

    Only anchors that actually own rows participate. Every row belonging to a
    held-out anchor lands on the eval side, so an anchor's neighbourhood is
    never split across the boundary. Training rows are then pruned to drop any
    row whose *partner* endpoint is a held-out anchor -- the symmetric
    ``abba_max`` readout makes ``(u, v)`` and ``(v, u)`` the same pair, so
    without this a held-out anchor could still enter training via a reciprocal
    row, and the held-out anchor's R^2 would partly reflect memorization
    rather than identifiability from ``(x_u, x_v)`` alone.

    Args:
        anchor_offsets: CSR row-start array, ``len(node_ids) + 1`` entries.
        pair_partner_idx: Per-row partner node position, same indexing space
            as `anchor_offsets`.
        seed: Seed for `numpy.random.default_rng`.
        fraction: Held-out share of the live anchors.

    Returns:
        An `AnchorSplit`.

    Raises:
        ValueError: If the split would leave either anchor side empty.
    """
    offsets = np.asarray(anchor_offsets, dtype=np.int64)
    partner_idx = np.asarray(pair_partner_idx, dtype=np.int64)
    live = np.flatnonzero(np.diff(offsets) > 0).astype(np.int64)
    n_heldout = int(round(live.size * fraction))
    if n_heldout < 1 or n_heldout >= live.size:
        raise ValueError(
            f"anchor holdout needs both sides non-empty; got {live.size} live anchors, "
            f"fraction={fraction} -> n_heldout={n_heldout}"
        )
    permuted = live[np.random.default_rng(seed).permutation(live.size)]
    heldout_anchors = np.sort(permuted[-n_heldout:])
    train_anchors = np.sort(permuted[:-n_heldout])

    def rows_for(anchors: NDArray[np.int64]) -> NDArray[np.int64]:
        if anchors.size == 0:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(
            [np.arange(offsets[a], offsets[a + 1], dtype=np.int64) for a in anchors.tolist()]
        )

    eval_rows = rows_for(heldout_anchors)
    candidate_train_rows = rows_for(train_anchors)
    touches_heldout = np.isin(partner_idx[candidate_train_rows], heldout_anchors)
    train_rows = candidate_train_rows[~touches_heldout]

    return AnchorSplit(
        train_rows=train_rows,
        eval_rows=eval_rows,
        heldout_anchors=heldout_anchors,
        n_dropped_touching_heldout=int(np.count_nonzero(touches_heldout)),
    )


# --------------------------------------------------------------------------- model


def zero_output_head(model: V3_1) -> bool:
    """Zero the final ``output_head`` linear so the model's first prediction is exactly 0.

    A spectral-normed linear cannot simply be zeroed in place: ``weight_orig /
    sigma`` becomes ``0 / 0 = NaN``. Nor can the parametrization be stripped with
    `torch.nn.utils.remove_spectral_norm`, which drops the forward pre-hook and
    the ``weight_orig``/``weight_u``/``weight_v`` entries but leaves its
    ``_load_state_dict_pre_hook`` installed -- so the module still *demands*
    those keys and every later `load_state_dict` raises, including the
    best-epoch restore at the end of training. The final linear is therefore
    replaced outright with a fresh, unparametrized, zeroed `torch.nn.Linear`,
    which carries no hooks and round-trips through `state_dict` cleanly.

    Only the probe's own final layer changes; the shipped classifier and the
    earlier spectral-normed head layers are untouched.

    Args:
        model: The probe's `V3_1` instance, modified in place.

    Returns:
        Whether the replaced layer had been spectral-normed.

    Raises:
        TypeError: If the output head does not end in a `torch.nn.Linear`.
    """
    final = model.output_head.layers[-1]
    if not isinstance(final, nn.Linear):
        raise TypeError(f"output_head must end in nn.Linear, got {type(final)!r}")
    was_spectral_normed = any(
        name.startswith("weight_orig") for name, _ in final.named_parameters()
    )
    replacement = nn.Linear(
        final.in_features,
        final.out_features,
        bias=final.bias is not None,
        device=final.weight.device,
        dtype=final.weight.dtype,
    )
    with torch.no_grad():
        replacement.weight.zero_()
        if replacement.bias is not None:
            replacement.bias.zero_()
    model.output_head.layers[-1] = replacement
    return was_spectral_normed


def build_residual_model(
    checkpoint_path: Path, *, init: str, device: torch.device
) -> tuple[V3_1, str, bool]:
    """Build the probe's `V3_1`, warm-loaded or randomly initialized, with a zeroed head.

    Args:
        checkpoint_path: v3_1 checkpoint; supplies ``model_config`` for both
            variants and the weights for ``warm``.
        init: ``"warm"`` or ``"scratch"``.
        device: Compute device.

    Returns:
        ``(model, checkpoint_id, spectral_norm_removed)``.

    Raises:
        ValueError: If `init` is not a known variant.
    """
    if init not in {"warm", "scratch"}:
        raise ValueError(f"init must be 'warm' or 'scratch', got {init!r}")
    if init == "warm":
        model, _family, checkpoint_id = _load_checkpoint(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_config = cast(dict[str, object], checkpoint["model_config"])
        model = cast(V3_1, build_model("v3_1", model_config))
        checkpoint_id = "scratch"
    probe = cast(V3_1, model)
    removed = zero_output_head(probe)
    probe.to(device)
    return probe, checkpoint_id, removed


# --------------------------------------------------------------------------- metrics


def regression_readout(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
) -> dict[str, float]:
    """Spearman rho, R^2, and MAE of a prediction vector against its target.

    Args:
        y_true: Shape ``(n,)`` targets.
        y_pred: Shape ``(n,)`` predictions.

    Returns:
        ``{"spearman": ..., "r2": ..., "mae": ...}``. Spearman is ``0.0`` when
        either side is constant; R^2 is ``0.0`` when the target has zero
        variance. R^2 is measured against the target's own mean, so the
        predict-mean baseline scores exactly ``0.0`` by construction and
        predict-zero can score below it.
    """
    residual = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    rho = float(spearmanr(y_true, y_pred).statistic) if y_true.size > 1 else 0.0
    return {
        "spearman": 0.0 if not np.isfinite(rho) else rho,
        "r2": 0.0 if total == 0.0 else 1.0 - residual / total,
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }


# --------------------------------------------------------------------------- training


@dataclass(frozen=True)
class EpochRecord:
    """One epoch of the residual probe's training history."""

    epoch: int
    train_loss: float
    heldout_spearman: float
    heldout_r2: float
    heldout_mae: float


def _pairs_for_rows(targets: KDTargets, rows: NDArray[np.int64]) -> list[tuple[str, str]]:
    """Map CSR row indices to ``(anchor_id, partner_id)`` node-id pairs."""
    node_ids = targets.node_ids
    return [
        (node_ids[int(targets.pair_anchor_idx[r])], node_ids[int(targets.pair_partner_idx[r])])
        for r in rows.tolist()
    ]


def _predict(
    model: V3_1,
    pairs: Sequence[tuple[str, str]],
    store: FeatureStore,
    device: torch.device,
    token_budget: int,
) -> NDArray[np.float64]:
    """Score `pairs` in eval mode, returning predictions in **input row order**.

    `LengthBucketedBatchSampler` emits bucket-major batches, so even with
    ``shuffle=False`` the visiting order is not ``0, 1, 2, ...``. Each batch's
    logits are therefore scattered back to their own row positions (the
    `score_universe._score_v3_1` convention). A sequential write would silently
    misalign predictions with targets whenever the pairs span more than one
    length bucket, corrupting every readout without raising.
    """
    model.eval()
    lengths = probe_lengths(store, pairs)
    dataset = TokenPairDataset(pairs, None, store, lengths)
    sampler = LengthBucketedBatchSampler(
        lengths, token_budget=token_budget, shuffle=False, seed=0, epoch=0
    )
    out = np.zeros(len(pairs), dtype=np.float64)
    with torch.no_grad():
        for batch_indices in sampler:
            batch = collate_token_pairs([dataset[i] for i in batch_indices])
            moved = {k: v.to(device) for k, v in batch.items()}
            logits = model(moved)["logits"].squeeze(-1)
            out[np.asarray(batch_indices, dtype=np.int64)] = (
                logits.detach().cpu().numpy().astype(np.float64).reshape(-1)
            )
    return out


def train_residual_probe(
    model: V3_1,
    *,
    store: FeatureStore,
    train_pairs: Sequence[tuple[str, str]],
    train_delta: NDArray[np.float32],
    eval_pairs: Sequence[tuple[str, str]],
    eval_delta: NDArray[np.float32],
    device: torch.device,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    token_budget: int,
    seed: int,
) -> tuple[list[EpochRecord], dict[str, float], int, NDArray[np.float64]]:
    """Train the probe on ``Delta`` with early stopping on held-out Spearman.

    On return, ``model`` holds the best epoch's weights, restored from a CPU
    copy taken when that epoch was selected -- not whatever epoch training
    happened to stop on -- so a train-side readout computed on ``model`` after
    this call describes the same checkpoint as ``best_readout``.

    Returns:
        ``(history, best_readout, best_epoch, best_predictions)``.

    Raises:
        ValueError: If the training loss becomes non-finite.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=_WEIGHT_DECAY)

    train_lengths = probe_lengths(store, train_pairs)
    train_dataset = TokenPairDataset(train_pairs, None, store, train_lengths)
    train_sampler = LengthBucketedBatchSampler(
        train_lengths, token_budget=token_budget, shuffle=True, seed=seed, epoch=0
    )
    eval_targets = eval_delta.astype(np.float64)

    history: list[EpochRecord] = []
    best_readout = {"spearman": -np.inf, "r2": 0.0, "mae": 0.0}
    best_epoch = -1
    best_predictions = np.zeros(len(eval_pairs), dtype=np.float64)
    best_state: dict[str, torch.Tensor] = {}

    for epoch in range(max_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch_rows in train_sampler:
            items = [train_dataset[i] for i in batch_rows]
            batch = collate_token_pairs(items)
            moved = {k: v.to(device) for k, v in batch.items()}
            y = torch.tensor(train_delta[batch_rows], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(moved)["logits"].squeeze(-1)
            loss = nn.functional.smooth_l1_loss(logits, y)
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite training loss at epoch {epoch}")
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_rows)
            total_rows += len(batch_rows)

        predicted = _predict(model, eval_pairs, store, device, token_budget)
        readout = regression_readout(eval_targets, predicted)
        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=total_loss / max(total_rows, 1),
                heldout_spearman=readout["spearman"],
                heldout_r2=readout["r2"],
                heldout_mae=readout["mae"],
            )
        )
        if readout["spearman"] > best_readout["spearman"]:
            best_readout = readout
            best_epoch = epoch
            best_predictions = predicted
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            logger.info("early stop at epoch %d (best epoch %d)", epoch, best_epoch)
            break

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    return history, best_readout, best_epoch, best_predictions


def baseline_readouts(
    eval_delta: NDArray[np.float32], train_delta: NDArray[np.float32]
) -> dict[str, dict[str, float]]:
    """Predict-zero and predict-mean baselines on the eval side.

    The predict-mean baseline uses the *training* side's mean, which is the only
    mean a real method would have access to.
    """
    y = eval_delta.astype(np.float64)
    return {
        "predict_zero": regression_readout(y, np.zeros_like(y)),
        "predict_mean": regression_readout(y, np.full_like(y, float(np.mean(train_delta)))),
    }


# --------------------------------------------------------------------------- pipeline


def run_s6_pipeline(
    *,
    targets_path: Path,
    checkpoint_path: Path,
    data_root: Path,
    strategy: str,
    output_dir: Path,
    init: str,
    seed: int,
    device: str,
    max_epochs: int = _MAX_EPOCHS,
    patience: int = _PATIENCE,
    token_budget: int = _TOKEN_BUDGET,
    learning_rate: float | None = None,
) -> dict[str, object]:
    """Train one residual-probe variant and write ``report.json`` + ``predictions.npz``.

    Args:
        targets_path: A v3 KD-targets artifact directory (must carry ``content_logit``).
        checkpoint_path: v3_1 checkpoint for the architecture and warm weights.
        data_root: Root holding the benchmark and feature-store subdirectories.
        strategy: Split strategy name.
        output_dir: Destination directory for both artifacts.
        init: ``"warm"`` or ``"scratch"``.
        seed: Split/shuffle/init seed.
        device: Torch device string.
        max_epochs: Epoch ceiling.
        patience: Early-stop patience on held-out Spearman.
        token_budget: Per-batch token budget.
        learning_rate: Overrides the per-variant default when given.

    Returns:
        The written report payload.
    """
    torch.manual_seed(seed)
    resolved_device = torch.device(device)

    targets = load_kd_targets(targets_path)
    delta = residual_targets(targets)

    # Defensive re-check: these rows are already train-side legal by construction
    # (the artifact was written by the KD target pass). This is the content_logit.py
    # precedent, not a new gate.
    split, test_nodes = _load_val_region_split(data_root, strategy)
    assert_training_side_only(targets.node_ids, truth_graph_for_kd(split), split, test_nodes)

    store = FeatureStore(data_root / _FEATURES_SUBDIR)
    anchor_split = anchor_holdout_split(targets.anchor_offsets, targets.pair_partner_idx, seed=seed)
    train_rows, eval_rows, heldout_anchors = (
        anchor_split.train_rows,
        anchor_split.eval_rows,
        anchor_split.heldout_anchors,
    )
    train_pairs = _pairs_for_rows(targets, train_rows)
    eval_pairs = _pairs_for_rows(targets, eval_rows)
    train_delta = delta[train_rows]
    eval_delta = delta[eval_rows]

    model, checkpoint_id, spectral_removed = build_residual_model(
        checkpoint_path, init=init, device=resolved_device
    )
    lr = (
        learning_rate
        if learning_rate is not None
        else (_LR_WARM if init == "warm" else _LR_SCRATCH)
    )

    history, best_readout, best_epoch, best_predictions = train_residual_probe(
        model,
        store=store,
        train_pairs=train_pairs,
        train_delta=train_delta,
        eval_pairs=eval_pairs,
        eval_delta=eval_delta,
        device=resolved_device,
        learning_rate=lr,
        max_epochs=max_epochs,
        patience=patience,
        token_budget=token_budget,
        seed=seed,
    )

    train_side = regression_readout(
        train_delta.astype(np.float64),
        _predict(model, train_pairs, store, resolved_device, token_budget),
    )
    baselines = baseline_readouts(eval_delta, train_delta)

    variant = f"{init}_s{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "predictions.npz",
        eval_rows=eval_rows,
        eval_delta=eval_delta,
        eval_predicted=best_predictions.astype(np.float32),
        heldout_anchors=heldout_anchors,
    )

    report: dict[str, object] = {
        "format": S6_REPORT_FORMAT,
        "evidence_class": "diagnostic",
        "variant": variant,
        "init": init,
        "seed": seed,
        "strategy": strategy,
        "targets_path": str(targets_path),
        "checkpoint_id": checkpoint_id,
        "spectral_norm_removed_from_output_head": spectral_removed,
        "learning_rate": lr,
        "split_manifest": {
            "n_rows": int(delta.size),
            "n_train_rows": int(train_rows.size),
            "n_eval_rows": int(eval_rows.size),
            "n_heldout_anchors": int(heldout_anchors.size),
            "holdout_fraction": _HOLDOUT_FRACTION,
            "n_train_rows_dropped_touching_heldout": anchor_split.n_dropped_touching_heldout,
        },
        "delta_stats": {
            "mean": float(np.mean(delta)),
            "std": float(np.std(delta)),
            "min": float(np.min(delta)),
            "max": float(np.max(delta)),
        },
        "selection": {
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "heldout": best_readout,
            "train_side": train_side,
            "generalization_gap_r2": train_side["r2"] - best_readout["r2"],
        },
        "baselines": baselines,
        "history": [
            {
                "epoch": record.epoch,
                "train_loss": record.train_loss,
                "heldout_spearman": record.heldout_spearman,
                "heldout_r2": record.heldout_r2,
                "heldout_mae": record.heldout_mae,
            }
            for record in history
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the S6 residual-probe CLI argument parser."""
    parser = argparse.ArgumentParser(description="S6 pair-residual identifiability probe")
    parser.add_argument(
        "--targets", type=Path, required=True, help="v3 KD-targets artifact directory"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init", choices=("warm", "scratch"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device"
    )
    parser.add_argument("--max-epochs", type=int, default=_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=_PATIENCE)
    parser.add_argument("--token-budget", type=int, default=_TOKEN_BUDGET)
    parser.add_argument("--learning-rate", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the S6 residual probe from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    report = run_s6_pipeline(
        targets_path=args.targets,
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        strategy=args.strategy,
        output_dir=args.output_dir,
        init=args.init,
        seed=args.seed,
        device=args.device,
        max_epochs=args.max_epochs,
        patience=args.patience,
        token_budget=args.token_budget,
        learning_rate=args.learning_rate,
    )
    logger.info(
        "s6 %s: selection=%s baselines=%s",
        report["variant"],
        json.dumps(report["selection"], sort_keys=True),
        json.dumps(report["baselines"], sort_keys=True),
    )


if __name__ == "__main__":
    main()

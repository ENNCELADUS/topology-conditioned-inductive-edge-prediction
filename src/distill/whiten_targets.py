r"""Whitened-axis KD row targets for the ``kd_white`` arm.

Projects a dumped teacher bank's ``teacher_rep`` onto its top PCA axes (fit on
the training block), standardizes each retained axis to unit variance on the
training rows, and packs the selected axes as a ``kd_row_targets_v1`` bank the
trainer regresses through the same auxiliary head as ``kd_struct``. Dropping
axis 1 keeps the arm orthogonal to the logit KD (the top axis is the teacher
logit); the V_val block is transformed with the training statistics.

    python -m src.distill.whiten_targets --bank outputs/distill/kd_row_targets_pma1_breadth_first \\
        --axes 2-8 --output outputs/distill/kd_row_targets_pma1_white_breadth_first
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.distill.artifacts import KDRowTargets, load_kd_targets, write_kd_targets

logger = logging.getLogger(__name__)

F64 = NDArray[np.float64]
WHITE_REP_SOURCE = "whitened_axes"


@dataclass(frozen=True)
class WhitenedAxes:
    """Top-``k`` PCA coordinates, unit variance on the training block."""

    train: F64
    val: F64
    axis_std: F64
    var_share: F64
    val_shift: F64
    val_std: F64


def whiten_axes(rep_tr: F64, rep_va: F64, *, k: int) -> WhitenedAxes:
    """Project both blocks onto the training block's top-``k`` centered PCA axes.

    Raises:
        ValueError: If ``k`` exceeds the bank's dimensions or numerical rank.
    """
    if k < 1 or k > min(rep_tr.shape):
        raise ValueError(f"k must be in [1, {min(rep_tr.shape)}], got {k}")
    mu = rep_tr.mean(axis=0)
    _, sing, vt = np.linalg.svd(rep_tr - mu, full_matrices=False)
    var = sing**2 / rep_tr.shape[0]
    axis_std = np.sqrt(var[:k])
    if not np.all(axis_std > 0.0):
        raise ValueError(f"k={k} exceeds the bank's numerical rank")
    train = ((rep_tr - mu) @ vt[:k].T) / axis_std
    val = ((rep_va - mu) @ vt[:k].T) / axis_std
    return WhitenedAxes(
        train, val, axis_std, var[:k] / var.sum(), val.mean(axis=0), val.std(axis=0)
    )


def whitened_row_targets(bank: KDRowTargets, *, axes: tuple[int, int]) -> KDRowTargets:
    """Replace ``teacher_rep`` by whitened PCA axes ``first..last`` (1-based, inclusive).

    Raises:
        ValueError: If the axis range is empty, starts below 1, or exceeds the bank.
    """
    first, last = axes
    if first < 1 or last < first:
        raise ValueError(f"axes must satisfy 1 <= first <= last, got {axes}")
    white = whiten_axes(
        bank.teacher_rep.astype(np.float64), bank.val_teacher_rep.astype(np.float64), k=last
    )
    sel = slice(first - 1, last)
    names = [f"pc{i}" for i in range(first, last + 1)]
    manifest = {
        "rep_source": WHITE_REP_SOURCE,
        "source_rep_source": bank.manifest.get("rep_source"),
        "descriptors": names,
        "axes": list(range(first, last + 1)),
        "axis_std": white.axis_std[sel].tolist(),
        "axis_var_share": white.var_share[sel].tolist(),
        "val_shift_whitened": white.val_shift[sel].tolist(),
    }
    return KDRowTargets(
        node_ids=bank.node_ids,
        pair_a_idx=bank.pair_a_idx,
        pair_b_idx=bank.pair_b_idx,
        pair_label=bank.pair_label,
        teacher_logit=bank.teacher_logit,
        teacher_rep=white.train[:, sel].astype(np.float16),
        val_pair_a_idx=bank.val_pair_a_idx,
        val_pair_b_idx=bank.val_pair_b_idx,
        val_pair_label=bank.val_pair_label,
        val_teacher_logit=bank.val_teacher_logit,
        val_teacher_rep=white.val[:, sel].astype(np.float16),
        manifest=manifest,
    )


def _parse_axes(spec: str) -> tuple[int, int]:
    first, _, last = spec.partition("-")
    return int(first), int(last or first)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--axes", type=_parse_axes, default=(2, 8))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    bank = load_kd_targets(args.bank)
    out = whitened_row_targets(bank, axes=args.axes)
    write_kd_targets(
        args.output,
        node_ids=out.node_ids,
        pair_a_idx=out.pair_a_idx,
        pair_b_idx=out.pair_b_idx,
        pair_label=out.pair_label,
        teacher_logit=out.teacher_logit,
        teacher_rep=out.teacher_rep,
        val_pair_a_idx=out.val_pair_a_idx,
        val_pair_b_idx=out.val_pair_b_idx,
        val_pair_label=out.val_pair_label,
        val_teacher_logit=out.val_teacher_logit,
        val_teacher_rep=out.val_teacher_rep,
        truth_graph_sha256=str(bank.manifest.get("truth_graph_sha256", "")),
        checkpoint_path=Path(str(bank.manifest.get("checkpoint_path", ""))),
        checkpoint_sha256=str(bank.manifest.get("checkpoint_sha256", "")),
        checkpoint_id=(
            str(bank.manifest["checkpoint_id"]) if bank.manifest.get("checkpoint_id") else None
        ),
        rep_source=WHITE_REP_SOURCE,
        manifest_extra={**out.manifest, "source_bank": str(args.bank)},
    )
    logger.info("wrote %s: axes %s, manifest %s", args.output, args.axes, out.manifest)


if __name__ == "__main__":
    main()

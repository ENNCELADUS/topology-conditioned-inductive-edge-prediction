"""Total validation loss: the quantity training patience counts on.

Every training loop in this repository stops on *total validation loss*, not on
a ranking statistic. The total mirrors the training objective:

    val_total = val_task_loss + sum_i  w_i * val_<term_i>

where ``w_i`` are the training weights carried on `DistillConfig` and
``val_<term_i>`` is that KD term's existing validation counterpart, already
computed by `src.train_b0._evaluate_distributed` and
`KDContextStream.validation_diagnostics`. Checkpoint *selection* is a separate
concern and is unaffected: see `src.eval.checkpoint_selection`.

Terms with no validation counterpart are omitted rather than approximated --
today that is only ``kd_gen``, whose generator loss is never scored on the
validation rows, so its monitor degrades to the task term alone.

Two unit caveats, deliberate and inherited from the diagnostics themselves:

- ``val_kd_gram_block_loss`` is a *batch* mean while the train-side ``kd_gram``
  telemetry is a *row* mean. The two differ by a batch-composition factor, so
  the ``kd_gram`` total is comparable across epochs of one run (all that
  patience needs) but not against another arm's total.
- ``val_kd_rank_loss``/``val_kd_dist_loss`` come from scoring the whole fixed
  V_val context bank on every rank, so they carry none of the
  ``rank_scale``/``dist_scale`` DDP count scaling the training terms apply.
  Every rank already holds the identical value; never all-reduce them.
"""

from __future__ import annotations

from collections.abc import Mapping

from src.distill.config import DistillConfig

# arm -> ((training weight attribute, validation metric key), ...)
_VAL_TERMS: dict[str, tuple[tuple[str, str], ...]] = {
    "none": (),
    "kd_logit": (("w_logit", "val_kd_logit_loss"),),
    "kd_rank": (("w_rank", "val_kd_rank_loss"), ("w_dist", "val_kd_dist_loss")),
    "kd_gram": (("w_gram", "val_kd_gram_block_loss"),),
    "kd_rep": (("w_rep", "val_kd_rep_loss"),),
    "kd_gen": (),
    "kd_struct": (("w_struct", "val_kd_struct_loss"),),
    "kd_white": (("w_white", "val_kd_struct_loss"),),
}


def val_total_terms(distill: DistillConfig | None) -> list[str]:
    """Name the terms `compose_val_total` sums, for the rank-zero startup log.

    Args:
        distill: The active KD section, or ``None`` for an undistilled family.

    Returns:
        ``["val_task_loss", "<w> * <key>", ...]`` in summation order.
    """
    terms = ["val_task_loss"]
    if distill is None or not distill.active:
        return terms
    terms.extend(f"{weight} * {key}" for weight, key in _VAL_TERMS[distill.arm])
    return terms


def compose_val_total(
    task_loss: float, kd: Mapping[str, float] | None, distill: DistillConfig | None
) -> float:
    """Sum the task validation loss and the weighted KD validation terms.

    Args:
        task_loss: `ValidationOutcome.task_loss` -- BCE-with-logits over the
            fixed validation rows, smoothed exactly as the training objective is.
        kd: `ValidationOutcome.kd`, the unweighted KD validation diagnostics;
            ``None`` for an undistilled run.
        distill: The active KD section, or ``None`` for an undistilled family.

    Returns:
        The monitored total. Equals ``task_loss`` when no KD arm is active.

    Raises:
        RuntimeError: If an active arm's validation counterpart is missing --
            the KD diagnostics did not run, so patience would silently fall back
            to the task term and the monitor would jump mid-run.
    """
    total = float(task_loss)
    if distill is None or not distill.active:
        return total
    for weight, key in _VAL_TERMS[distill.arm]:
        value = None if kd is None else kd.get(key)
        if value is None:
            raise RuntimeError(
                f"arm {distill.arm} needs validation diagnostic '{key}' to compose "
                "the total validation loss, but the validation outcome has no such key"
            )
        total += float(getattr(distill, weight)) * float(value)
    return total


__all__ = ["compose_val_total", "val_total_terms"]

"""Pilot B: does the Stage II generator predict a row-dependent motif graph at all?

Spec section 0.2 and plan ``docs/tmp/2026-09-18-motif-stage2-fix-plan.md`` part C
phase 0. Wave 1 found a near-constant, slot-symmetric generator whose ``L_slot``
was worse than a fitted constant, so the wave-2 warm-start prefix is read at three
pre-registered levels before any compute goes into the joint phase:

1. *Fit*: the generator's ``L_G`` against a constant template re-fitted under the
   same loss, the corpus mean template, the positive-row wedge-mass
   reconstruction, and the predicted-vs-true mass correlations.
2. *Conditional dependence*: the rise of ``L_G`` when each row is scored against
   another row's target under seeded whole-set permutations. A generator whose
   output does not depend on the row cannot raise it.
3. *Downstream utility*: the predicted graphs fed through the formal scoring path
   to the frozen Stage I bundle and to the checkpoint's own reader/interface,
   read against the ``gates_off`` and true-template cells.

Held-out training rows are **node**-disjoint from the rows the constant is fitted
on: the training nodes are split by seed and a row is kept only when both of its
endpoints fall on one side. V_val rows are the ``val_cls`` pair list with true
templates compiled on the V_val truth graph, which makes every V_val number here
a labelled diagnostic, exactly as wave 1 recorded it.

Nothing here reimplements the trunk: the level-3 logits come from
``src.score_universe``'s packed path with an explicit ``row_templates`` bank.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import stats as scipy_stats
from sklearn.metrics import average_precision_score, roc_auc_score

import src.score_universe as su
from src.data.motif_crossfit import seeded_hash_node_folds
from src.data.motif_template import (
    EDGE_TYPES,
    N_EDGE_TYPES,
    N_EDGES,
    MotifTemplateTable,
    count_statistics,
    slot_profiles,
    template_statistics,
)
from src.distill.motif_losses import slot_loss_rows
from src.model.egostitch.classifier.motif_prompt import MotifPromptConfig, V3_1MotifPrompt

logger = logging.getLogger("motif_pilot_b")

Pair = tuple[str, str]

#: The four supervised multisets, by their `slot_profiles` key.
FAMILY_NAMES: dict[str, str] = {
    "p": "wedge_product",
    "q": "bridge_path_product",
    "a": "attach_raw",
    "b": "interior_raw",
    "c": "closure_raw",
}
#: The mass statistics `count_statistics` returns.
MASS_KEYS: tuple[str, ...] = ("wedge_mass", "bridge_mass", "deg_u", "deg_v")

#: Level 1: the positive-row wedge-mass reconstruction must fall below this.
LEVEL1_RECONSTRUCTION_MAX = 0.7
#: Level 1: the predicted mean wedge mass must sit within this factor of the truth's.
LEVEL1_MASS_FACTOR = 3.0
#: Level 2: every seeded transplant must raise ``L_G`` by at least this fraction.
LEVEL2_MIN_RELATIVE_RISE = 0.25

_GATE_EPS = 1e-6


@dataclass(frozen=True)
class GraphLossWeights:
    """The ``L_G`` term weights a checkpoint was trained under (spec section 7.3, F2)."""

    beta_p: float
    beta_q: float
    beta_a: float
    beta_i: float
    beta_c: float
    huber_delta: float

    @classmethod
    def from_config(cls, cfg: MotifPromptConfig) -> GraphLossWeights:
        """Read the betas from a checkpoint's own config rather than assuming them.

        Args:
            cfg: The checkpoint's ``model.config.motif_prompt`` block.

        Returns:
            The weights `slot_loss_rows` is called with.
        """
        return cls(
            beta_p=float(cfg.beta_p),
            beta_q=float(cfg.beta_q),
            beta_a=float(cfg.beta_a),
            beta_i=float(cfg.beta_i),
            beta_c=float(cfg.beta_c),
            huber_delta=float(cfg.huber_delta),
        )

    def wave_one(self) -> GraphLossWeights:
        """The historical wave-1 ``L_slot``: the same betas with the raw closure term off."""
        return replace(self, beta_c=0.0)

    def rows(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-row ``L_G`` of one prediction against one target table.

        Args:
            predicted: ``(B, 96)`` predicted weights.
            target: ``(B, 96)`` compiled weights.

        Returns:
            ``(B,)`` per-row losses.
        """
        return slot_loss_rows(
            predicted,
            target,
            beta_p=self.beta_p,
            beta_q=self.beta_q,
            beta_a=self.beta_a,
            beta_i=self.beta_i,
            beta_c=self.beta_c,
            huber_delta=self.huber_delta,
        )

    def mean(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """Batch-mean ``L_G``.

        Args:
            predicted: ``(B, 96)`` predicted weights.
            target: ``(B, 96)`` compiled weights.

        Returns:
            The scalar mean.
        """
        return float(self.rows(predicted, target).mean())


# ---------------------------------------------------------------------------
# Row selection: node-disjoint training sides
# ---------------------------------------------------------------------------


def split_training_nodes(
    nodes: Sequence[str], *, seed: int
) -> tuple[frozenset[str], frozenset[str]]:
    """Split the training nodes into a fit side and a held-out side.

    The constant template is fitted on one side and read on the other, so the
    comparison cannot be won by memorising a node's neighbourhood.

    Args:
        nodes: Training-universe node ids.
        seed: Partition seed.

    Returns:
        ``(fit_nodes, heldout_nodes)``, disjoint and covering ``nodes``.

    Raises:
        ValueError: If fewer than two nodes are supplied.
    """
    ordered = sorted(set(nodes))
    if len(ordered) < 2:
        raise ValueError(f"a node-disjoint split needs at least two nodes, got {len(ordered)}")
    order = np.random.default_rng(seed).permutation(len(ordered))
    half = len(ordered) // 2
    fit = frozenset(ordered[int(i)] for i in order[:half])
    return fit, frozenset(ordered[int(i)] for i in order[half:])


def rows_inside(
    positives: Sequence[Pair],
    negatives: Sequence[Pair],
    side: frozenset[str],
    *,
    limit: int,
    seed: int,
    negative_ratio: int = 5,
) -> tuple[list[Pair], NDArray[np.int8]]:
    """Draw a nonself row sample at the corpus label mix, entirely on one side.

    The generator was trained on the 1:``negative_ratio`` stream and 94.5% of
    training negatives carry an empty closure family, so a balanced draw would
    put the fitted constant at its optimum and the generator off-distribution
    (or the reverse on V_val): the level-1 stop rule is read at the training mix.

    Self pairs are dropped: their compiled template is empty by construction, so
    they carry no graph supervision and `slot_loss_rows` masks them out in
    training (spec section 3).

    Args:
        positives: Training positives.
        negatives: Training negatives.
        side: The node side a row must lie entirely inside.
        limit: Maximum rows; ``0`` or more than available takes everything.
        seed: Sampling seed.
        negative_ratio: Negatives per positive, the training stream's ratio.

    Returns:
        ``(pairs, labels)``, positives first, each half sorted by its own index.
    """
    if negative_ratio < 1:
        raise ValueError("negative_ratio must be at least 1")
    generator = np.random.default_rng(seed)
    kept: list[Pair] = []
    labels: list[int] = []
    shares = {1: 1, 0: negative_ratio}
    total = 1 + negative_ratio
    for label, source in ((1, positives), (0, negatives)):
        pool = sorted((u, v) for u, v in source if u != v and u in side and v in side)
        want = len(pool) if limit <= 0 else min(len(pool), max(limit * shares[label] // total, 0))
        if want < len(pool):
            picked = np.sort(generator.choice(len(pool), size=want, replace=False))
            pool = [pool[int(i)] for i in picked]
        kept.extend(pool)
        labels.extend([label] * len(pool))
    return kept, np.asarray(labels, dtype=np.int8)


# ---------------------------------------------------------------------------
# Level 1: fit
# ---------------------------------------------------------------------------


def fit_constant_template(
    target: torch.Tensor,
    weights: GraphLossWeights,
    *,
    steps: int = 3000,
    lr: float = 0.05,
    batch: int = 2048,
    seed: int = 7,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Fit one asymmetric 96-weight constant template by Adam under ``L_G``.

    The comparator is deliberately *asymmetric*: it may put a different weight on
    every one of the 96 edges, so beating it requires row dependence and not just
    the right per-type magnitude. The parameter is the pre-sigmoid logit vector,
    matching the generator's own output parameterisation.

    Args:
        target: ``(n, 96)`` compiled templates of the fit rows.
        weights: The loss the constant is fitted under.
        steps: Adam steps.
        lr: Adam learning rate.
        batch: Rows resampled per step.
        seed: Minibatch seed.
        device: Device to fit on; the target's device when omitted.

    Returns:
        The ``(96,)`` fitted constant template, on the CPU.

    Raises:
        ValueError: If ``target`` is not a non-empty ``(n, 96)`` table.
    """
    if target.dim() != 2 or target.size(-1) != N_EDGES or target.size(0) == 0:
        raise ValueError(f"target must be a non-empty (n, {N_EDGES}) table")
    where = device if device is not None else target.device
    rows = target.to(device=where, dtype=torch.float32)
    logits = torch.zeros(N_EDGES, device=where, requires_grad=True)
    optimiser = torch.optim.Adam([logits], lr=lr)
    sampler = torch.Generator(device="cpu").manual_seed(seed)
    size = min(batch, rows.size(0))
    for _ in range(steps):
        index = torch.randint(0, rows.size(0), (size,), generator=sampler).to(where)
        chunk = rows.index_select(0, index)
        constant = torch.sigmoid(logits).unsqueeze(0).expand(chunk.size(0), -1)
        loss = weights.rows(constant, chunk).mean()
        optimiser.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimiser.step()
    return torch.sigmoid(logits).detach().cpu()


def _describe(values: NDArray[np.float64]) -> dict[str, float]:
    """Return the standard spread summary of one array."""
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def _correlate(left: NDArray[np.float64], right: NDArray[np.float64]) -> dict[str, float | None]:
    """Pearson and Spearman correlation, or ``None`` when either side is constant."""
    if left.size < 2 or float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(np.corrcoef(left, right)[0, 1]),
        "spearman": float(scipy_stats.spearmanr(left, right).statistic),
    }


def mass_report(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, object]:
    """Compare the closed-form count statistics of a prediction and its target.

    ``s_family`` is the mean of the target's positive values, so the
    reconstruction error is reported in units of a typical true mass rather than
    in raw weight units.

    Args:
        predicted: ``(B, 96)`` predicted weights of nonself rows.
        target: ``(B, 96)`` compiled weights of the same rows.

    Returns:
        One entry per key of `MASS_KEYS`.
    """
    ours = count_statistics(predicted.float())
    theirs = count_statistics(target.float())
    report: dict[str, object] = {}
    for key in MASS_KEYS:
        mine = ours[key].numpy().astype(np.float64)
        truth = theirs[key].numpy().astype(np.float64)
        positive = truth > 0.0
        zero = ~positive
        scale = float(truth[positive].mean()) if positive.any() else 0.0
        report[key] = {
            "predicted": _describe(mine),
            "true": _describe(truth),
            "s_family": scale,
            "n_positive_target_rows": int(positive.sum()),
            "n_zero_target_rows": int(zero.sum()),
            "false_mass_on_zero_rows": (float(mine[zero].mean()) if zero.any() else None),
            "normalised_reconstruction_on_positive_rows": (
                float(np.abs(mine[positive] - truth[positive]).mean() / scale)
                if positive.any() and scale > 0.0
                else None
            ),
            "predicted_over_true_mean": (
                float(mine.mean() / truth.mean()) if float(truth.mean()) > 0.0 else None
            ),
            "correlation": _correlate(mine, truth),
        }
    return report


def sorted_profile_dispersion(
    predicted: torch.Tensor, target: torch.Tensor
) -> dict[str, dict[str, float | None]]:
    """Per-family spread of the descending-sorted profiles, in ``s_family`` units.

    ``L_G`` compares sorted profiles, so this is the dispersion the loss actually
    sees: a generator emitting one constant graph has a per-coordinate standard
    deviation of zero however large its weights are.

    Args:
        predicted: ``(B, 96)`` predicted weights of nonself rows.
        target: ``(B, 96)`` compiled weights of the same rows.

    Returns:
        One entry per supervised multiset.
    """
    ours = slot_profiles(predicted.float())
    theirs = slot_profiles(target.float())
    report: dict[str, dict[str, float | None]] = {}
    for key, name in FAMILY_NAMES.items():
        mine = ours[key].sort(dim=1, descending=True).values.numpy().astype(np.float64)
        truth = theirs[key].sort(dim=1, descending=True).values.numpy().astype(np.float64)
        positive = truth > 0.0
        scale = float(truth[positive].mean()) if positive.any() else 0.0
        report[name] = {
            "s_family": scale,
            "predicted_coord_std_over_s": (
                float(np.mean(mine.std(axis=0)) / scale) if scale > 0.0 else None
            ),
            "true_coord_std_over_s": (
                float(np.mean(truth.std(axis=0)) / scale) if scale > 0.0 else None
            ),
            "predicted_top_coord_mean": float(mine[:, 0].mean()),
            "true_top_coord_mean": float(truth[:, 0].mean()),
        }
    return report


def gate_logit_summary(predicted: torch.Tensor) -> dict[str, dict[str, float]]:
    """Per-edge-type summary of the generator's pre-sigmoid gate logits.

    The heads end in a sigmoid, so the logit of a predicted weight is the head's
    own pre-activation whenever the family gate leaves the edge alone; a gated
    (exactly zero) edge is clamped instead of diverging, which the F1 saturation
    telemetry reads as a saturated head. This is the quantity the closure-bias
    rule of F1 moves, so it is recorded here as the drift detector.

    Args:
        predicted: ``(B, 96)`` predicted weights.

    Returns:
        ``mean``, ``frac_abs_gt_3`` and the mean sigmoid derivative per edge type.
    """
    values = predicted.float().clamp(_GATE_EPS, 1.0 - _GATE_EPS)
    logits = torch.logit(values)
    names = ("closure", "attach", "interior")
    summary: dict[str, dict[str, float]] = {}
    for kind in range(N_EDGE_TYPES):
        mask = torch.as_tensor([edge == kind for edge in EDGE_TYPES])
        block = logits[:, mask]
        probability = values[:, mask]
        summary[names[kind]] = {
            "mean": float(block.mean()),
            "std": float(block.std()),
            "frac_abs_gt_3": float((block.abs() > 3.0).float().mean()),
            "mean_sigmoid_derivative": float((probability * (1.0 - probability)).mean()),
            "std_across_rows_mean": float(block.std(dim=0).mean()),
        }
    return summary


# ---------------------------------------------------------------------------
# Level 2: conditional dependence
# ---------------------------------------------------------------------------


def transplant_permutations(rows: int, *, count: int, seed: int) -> list[NDArray[np.int64]]:
    """Draw seeded whole-set permutations, never the identity.

    Each permutation re-pairs every row with another row's target, which
    preserves the universe's distribution of targets and destroys only the
    pairing -- the row-dependence null the corpus mean template is not.

    Args:
        rows: Rows to permute.
        count: Permutations to draw.
        seed: Base seed; permutation ``i`` uses ``seed + i``, redrawn while it
            comes out as the identity.

    Returns:
        ``count`` permutations of ``range(rows)``.

    Raises:
        ValueError: If ``rows`` is below two or ``count`` is negative.
    """
    if rows < 2:
        raise ValueError(f"a transplant needs at least two rows, got {rows}")
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    identity = np.arange(rows, dtype=np.int64)
    drawn: list[NDArray[np.int64]] = []
    offset = 0
    for index in range(count):
        while True:
            permutation = np.random.default_rng(seed + index + offset).permutation(rows)
            if not np.array_equal(permutation, identity):
                break
            offset += count + 1
        drawn.append(permutation.astype(np.int64))
    return drawn


def transplant_report(
    predicted: torch.Tensor,
    target: torch.Tensor,
    weights: GraphLossWeights,
    permutations: Sequence[NDArray[np.int64]],
) -> dict[str, object]:
    """Rise of ``L_G`` when each row reads another row's target.

    Args:
        predicted: ``(B, 96)`` predicted weights of nonself rows.
        target: ``(B, 96)`` compiled weights of the same rows.
        weights: The loss to measure under.
        permutations: The seeded row permutations.

    Returns:
        The base loss, each permutation's loss and relative rise, and the
        mean/min/max of those rises.
    """
    base = weights.mean(predicted, target)
    per_permutation: list[dict[str, float]] = []
    for index, permutation in enumerate(permutations):
        order = torch.from_numpy(permutation)
        value = weights.mean(predicted, target.index_select(0, order))
        per_permutation.append(
            {
                "permutation": float(index),
                "loss": value,
                "rise": value - base,
                "relative_rise": (value - base) / base if base > 0.0 else float("nan"),
            }
        )
    rises = [entry["relative_rise"] for entry in per_permutation]
    return {
        "base_loss": base,
        "permutations": len(per_permutation),
        "per_permutation": per_permutation,
        "relative_rise_mean": float(np.mean(rises)) if rises else None,
        "relative_rise_min": float(np.min(rises)) if rises else None,
        "relative_rise_max": float(np.max(rises)) if rises else None,
    }


# ---------------------------------------------------------------------------
# Level 3: downstream utility through the formal scoring path
# ---------------------------------------------------------------------------


@contextmanager
def _capture_bank() -> Iterator[dict[str, torch.Tensor]]:
    """Temporarily wrap the scorer's transplant-bank builder so the bank is returned.

    `src.score_universe` has no hook that hands a caller the generator's own
    predicted graphs, and reimplementing the packed encode/predict pass would be
    a second, unverified copy of the scoring path. Wrapping the one private
    builder for the duration of a single call keeps the formal path intact.

    Yields:
        A mapping that holds the captured ``(num_rows, 96)`` bank under ``bank``.
    """
    captured: dict[str, torch.Tensor] = {}
    original = su._motif_source_weights

    def spy(
        source_batches: Sequence[Sequence[int]],
        predict_batch: Callable[[Sequence[int]], torch.Tensor],
        *,
        num_rows: int,
    ) -> torch.Tensor:
        bank = original(source_batches, predict_batch, num_rows=num_rows)
        captured["bank"] = bank
        return bank

    su._motif_source_weights = spy
    try:
        yield captured
    finally:
        su._motif_source_weights = original


def predicted_bank(
    model: V3_1MotifPrompt,
    pairs: Sequence[Pair],
    pack_dir: Path,
    *,
    device: torch.device,
    amp: str,
    token_budget: int,
) -> tuple[torch.Tensor, NDArray[np.float32]]:
    """Return the generator's own predicted graphs and the logits they produce.

    Passing each row its *own* pair as the transplant source makes the scorer
    build its bank over the rows themselves, so the captured bank is the
    generator's own prediction and the returned logits reproduce the formal
    scoring pass exactly.

    Args:
        model: A Stage II motif-prompt checkpoint in ``eval()`` mode.
        pairs: The rows to score.
        pack_dir: The packed feature directory.
        device: Compute device.
        amp: Encoder autocast mode.
        token_budget: Scoring token budget.

    Returns:
        ``(bank, logits)`` with the ``(n, 96)`` fp32 CPU bank first.
    """
    with _capture_bank() as captured:
        logits = su._score_v3_1_packed(
            model,
            pairs,
            pack_dir,
            device=device,
            amp=amp,
            token_budget=token_budget,
            shuffle_sources=list(pairs),
        )
    return captured["bank"], logits


def score_with_graph(
    model: V3_1MotifPrompt,
    pairs: Sequence[Pair],
    pack_dir: Path,
    *,
    device: torch.device,
    amp: str,
    token_budget: int,
    weights: torch.Tensor | None = None,
    intervention: str = "none",
) -> NDArray[np.float32]:
    """Score ``pairs`` through the formal packed path, optionally on a substituted graph.

    Args:
        model: A motif-prompt checkpoint in ``eval()`` mode.
        pairs: The rows to score.
        pack_dir: The packed feature directory.
        device: Compute device.
        amp: Encoder autocast mode.
        token_budget: Scoring token budget.
        weights: An explicit ``(n, 96)`` bank the reader reads, or ``None`` to let
            the model produce its own graph.
        intervention: A model-level scoring intervention (``gates_off``/``mean``).

    Returns:
        The ``(n,)`` float32 logits.
    """
    previous = model.intervention
    model.intervention = intervention
    try:
        return su._score_v3_1_packed(
            model,
            pairs,
            pack_dir,
            device=device,
            amp=amp,
            token_budget=token_budget,
            row_templates=weights,
        )
    finally:
        model.intervention = previous


def edge_metrics(logits: NDArray[np.float32], labels: NDArray[np.int8]) -> dict[str, float | None]:
    """AUROC and AUPRC of raw logits (monotone in the probability), or ``None``."""
    truth = np.asarray(labels, dtype=np.int64)
    if truth.size == 0 or truth.min() == truth.max():
        return {"auroc": None, "auprc": None}
    scores = np.asarray(logits, dtype=np.float64)
    return {
        "auroc": float(roc_auc_score(truth, scores)),
        "auprc": float(average_precision_score(truth, scores)),
    }


def _write_cell(
    output_dir: Path,
    name: str,
    *,
    pairs: Sequence[Pair],
    labels: NDArray[np.int8],
    logits: NDArray[np.float32],
    checkpoint_id: str,
    strategy: str,
    graph_source: str,
) -> Path:
    """Write one level-3 cell as a pinned scores artifact and validate its precision.

    Args:
        output_dir: Destination directory.
        name: Cell name (``s1_pred``, ``s2_pred``, ...).
        pairs: The scored rows.
        labels: Their labels.
        logits: Their raw logits.
        checkpoint_id: The reading checkpoint's id.
        strategy: Split strategy.
        graph_source: Which 96-edge bank the reader read (``true``, ``predicted``,
            ``gates_off``), so cells sharing a reader stay distinguishable.

    Returns:
        The artifact path.
    """
    node_ids = sorted({node for pair in pairs for node in pair})
    index = {node: position for position, node in enumerate(node_ids)}
    path = output_dir / f"{name}.val_cls.npz"
    su.save_scores(
        path,
        node_ids=node_ids,
        u_idx=np.asarray([index[u] for u, _ in pairs], dtype=np.int32),
        v_idx=np.asarray([index[v] for _, v in pairs], dtype=np.int32),
        logit=np.asarray(logits, dtype=np.float32),
        label=np.asarray(labels, dtype=np.int8),
        row_start=0,
        meta={
            "checkpoint_id": checkpoint_id,
            "model_family": su.MOTIF_PROMPT_FAMILY,
            "pairs_source": "val_cls",
            "strategy": strategy,
            "num_rows": len(pairs),
            "created_utc": datetime.now(UTC).isoformat(),
            "torch_version": str(torch.__version__),
            "score_precision": {
                "contract": su._MOTIF_PROMPT_PRECISION_CONTRACT,
                "encode_autocast": "bf16",
                "pair_autocast": False,
                "pair_compute_dtype": "float32",
                "logit_storage_dtype": "float32",
            },
            "pilot_b_cell": name,
            "graph_source": graph_source,
        },
    )
    su.validate_artifact_precision(su.load_scores(path), label=str(path))
    return path


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def level1_verdict(universes: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Apply the pre-registered level-1 rule to both universes.

    Args:
        universes: One `_universe_report` payload per universe name.

    Returns:
        The per-check booleans and the overall pass.
    """
    beats: dict[str, bool] = {}
    reconstruction: dict[str, bool] = {}
    mass: dict[str, bool] = {}
    for name, payload in universes.items():
        losses = cast(Mapping[str, float], payload["L_G"])
        beats[name] = losses["generator"] < losses["fitted_constant_template"]
        masses = cast(Mapping[str, Mapping[str, object]], payload["masses"])
        wedge = masses["wedge_mass"]
        value = wedge["normalised_reconstruction_on_positive_rows"]
        reconstruction[name] = isinstance(value, float) and value < LEVEL1_RECONSTRUCTION_MAX
        ratio = wedge["predicted_over_true_mean"]
        mass[name] = (
            isinstance(ratio, float)
            and ratio > 0.0
            and 1.0 / LEVEL1_MASS_FACTOR <= ratio <= LEVEL1_MASS_FACTOR
        )
    passed = all(beats.values()) and all(reconstruction.values()) and all(mass.values())
    return {
        "rule": (
            "the generator beats the re-fitted asymmetric constant on both universes, "
            f"positive-row wedge reconstruction < {LEVEL1_RECONSTRUCTION_MAX}, and the "
            f"predicted mean wedge mass sits within {LEVEL1_MASS_FACTOR}x of the truth's"
        ),
        "beats_fitted_constant": beats,
        "wedge_reconstruction_below_max": reconstruction,
        "wedge_mass_within_factor": mass,
        "passed": passed,
    }


def level2_verdict(universes: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Apply the pre-registered level-2 rule: every permutation raises ``L_G`` enough.

    Args:
        universes: One `_universe_report` payload per universe name.

    Returns:
        The per-universe booleans and the overall pass.
    """
    per_universe: dict[str, bool] = {}
    for name, payload in universes.items():
        transplant = cast(Mapping[str, object], payload["transplant"])
        entries = cast(Sequence[Mapping[str, float]], transplant["per_permutation"])
        per_universe[name] = bool(entries) and all(
            entry["relative_rise"] >= LEVEL2_MIN_RELATIVE_RISE for entry in entries
        )
    return {
        "rule": (
            f"every seeded transplant raises L_G by at least "
            f"{LEVEL2_MIN_RELATIVE_RISE:.0%}, on every universe"
        ),
        "per_universe": per_universe,
        "passed": all(per_universe.values()),
    }


def level3_verdict(cells: Mapping[str, Mapping[str, float | None]]) -> dict[str, object]:
    """Apply the pre-registered level-3 rule: both predicted-graph cells beat ``gates_off``.

    Args:
        cells: AUROC/AUPRC per level-3 cell name.

    Returns:
        The per-cell booleans and the overall pass.
    """
    reference = cells.get("s2_gates_off", {}).get("auprc")
    beats: dict[str, bool] = {}
    for name in ("s1_pred", "s2_pred"):
        value = cells.get(name, {}).get("auprc")
        beats[name] = (
            isinstance(value, float) and isinstance(reference, float) and value > reference
        )
    return {
        "rule": "both predicted-graph cells beat the gates_off cell on val_cls AUPRC",
        "gates_off_auprc": reference,
        "beats_gates_off": beats,
        "passed": all(beats.values()),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _universe_report(
    *,
    predicted: torch.Tensor,
    target: torch.Tensor,
    weights: GraphLossWeights,
    mean_template: torch.Tensor,
    fitted_constant: torch.Tensor,
    permutations: Sequence[NDArray[np.int64]],
) -> dict[str, object]:
    """Assemble the level-1 and level-2 readings of one universe's nonself rows.

    Args:
        predicted: ``(B, 96)`` predicted weights.
        target: ``(B, 96)`` compiled weights.
        weights: The checkpoint's own ``L_G``.
        mean_template: The corpus mean adjacency the checkpoint carries.
        fitted_constant: The constant re-fitted on the node-disjoint fit rows.
        permutations: The seeded transplant permutations.

    Returns:
        The universe payload.
    """
    historical = weights.wave_one()
    constant_rows = fitted_constant.unsqueeze(0).expand_as(predicted)
    mean_rows = mean_template.unsqueeze(0).expand_as(predicted)
    return {
        "rows": int(predicted.size(0)),
        "L_G": {
            "generator": weights.mean(predicted, target),
            "fitted_constant_template": weights.mean(constant_rows, target),
            "mean_template": weights.mean(mean_rows, target),
            "generator_minus_fitted": (
                weights.mean(predicted, target) - weights.mean(constant_rows, target)
            ),
        },
        "L_slot_wave_one_reference": {
            "note": "the wave-1 loss: the same betas with beta_c = 0",
            "generator": historical.mean(predicted, target),
            "fitted_constant_template": historical.mean(constant_rows, target),
            "mean_template": historical.mean(mean_rows, target),
        },
        "masses": mass_report(predicted, target),
        "sorted_profile_dispersion": sorted_profile_dispersion(predicted, target),
        "gate_logits": gate_logit_summary(predicted),
        "transplant": transplant_report(predicted, target, weights, permutations),
    }


def run_pilot_b(
    *,
    checkpoint: Path,
    stage1_checkpoint: Path,
    pack_dir: Path,
    data_root: Path,
    strategy: str,
    output_dir: Path,
    permutations: int,
    train_rows: int,
    heldout_rows: int,
    device: torch.device,
    amp: str,
    token_budget: int,
    seed: int,
    fit_steps: int,
) -> dict[str, object]:
    """Run the three pilot-B levels over one Stage II prefix checkpoint.

    Args:
        checkpoint: The Stage II prefix checkpoint (``best.pt``/``last.pt``).
        stage1_checkpoint: The frozen Stage I bundle for the level-3 reader swap.
        pack_dir: The packed feature directory.
        data_root: Benchmark data root.
        strategy: Split strategy.
        output_dir: Destination for the report and the level-3 artifacts.
        permutations: Seeded transplant permutations per universe.
        train_rows: Rows the constant template is fitted on.
        heldout_rows: Node-disjoint held-out training rows the generator is read on.
        device: Compute device.
        amp: Encoder autocast mode; the pair pass is fp32 by construction.
        token_budget: Scoring token budget.
        seed: Master seed for the node split, the row draws and the permutations.
        fit_steps: Adam steps for the constant template.

    Returns:
        The full ``pilot_b.json`` payload.

    Raises:
        ValueError: If either checkpoint is not the motif-prompt stage it must be.
    """
    torch.manual_seed(seed)
    loaded, family, checkpoint_id = su._load_checkpoint(checkpoint)
    if family != su.MOTIF_PROMPT_FAMILY:
        raise ValueError(f"{checkpoint}: model_family {family!r}, expected a motif prompt")
    model = cast(V3_1MotifPrompt, loaded)
    if model.cfg.stage != "two":
        raise ValueError(f"{checkpoint}: pilot B reads a Stage II checkpoint")
    model = model.to(device).eval()
    stage1_loaded, stage1_family, stage1_id = su._load_checkpoint(stage1_checkpoint)
    if stage1_family != su.MOTIF_PROMPT_FAMILY:
        raise ValueError(f"{stage1_checkpoint}: expected a motif-prompt Stage I bundle")
    stage1 = cast(V3_1MotifPrompt, stage1_loaded)
    if stage1.cfg.stage != "one":
        raise ValueError(f"{stage1_checkpoint}: pilot B's reader swap needs a Stage I bundle")
    stage1 = stage1.to(device).eval()

    weights = GraphLossWeights.from_config(model.cfg)
    split = su._load_val_region_split(data_root, strategy)
    if model.cfg.crossfit_fold is None:
        fit_nodes, heldout_nodes = split_training_nodes(sorted(split.train_nodes), seed=seed)
        split_kind = "seeded_random_half"
    else:
        folds = seeded_hash_node_folds(split.train_nodes, seed=model.cfg.crossfit_seed)
        fit_nodes = frozenset(
            node for node, fold in folds.items() if fold == model.cfg.crossfit_fold
        )
        heldout_nodes = frozenset(
            node for node, fold in folds.items() if fold != model.cfg.crossfit_fold
        )
        split_kind = "crossfit_opposite_fold"
    fit_pairs, _ = rows_inside(
        sorted(split.training_positives),
        list(split.training_negatives),
        fit_nodes,
        limit=train_rows,
        seed=seed + 1,
    )
    heldout_pairs, _ = rows_inside(
        sorted(split.training_positives),
        list(split.training_negatives),
        heldout_nodes,
        limit=heldout_rows,
        seed=seed + 2,
    )
    logger.info(
        "fit rows %d on %d nodes; held-out rows %d on %d nodes (node-disjoint)",
        len(fit_pairs),
        len(fit_nodes),
        len(heldout_pairs),
        len(heldout_nodes),
    )

    training_table = MotifTemplateTable(split.build_training_graph())
    fit_compiled = training_table.weights(fit_pairs)
    fit_target = torch.from_numpy(fit_compiled)
    heldout_target = torch.from_numpy(training_table.weights(heldout_pairs))
    corpus = template_statistics(fit_compiled)

    val_pairs, val_labels = su._resolve_pairs("val_cls", data_root, strategy)
    val_table = MotifTemplateTable(
        su._oracle_truth_graph_for_scoring("val_cls", data_root, strategy)
    )
    val_target = torch.from_numpy(val_table.weights(val_pairs))

    logger.info("fitting the asymmetric constant template on %d rows", len(fit_pairs))
    fitted = fit_constant_template(fit_target, weights, steps=fit_steps, device=device)

    heldout_predicted, _ = predicted_bank(
        model, heldout_pairs, pack_dir, device=device, amp=amp, token_budget=token_budget
    )
    val_predicted, val_own_logits = predicted_bank(
        model, val_pairs, pack_dir, device=device, amp=amp, token_budget=token_budget
    )

    mean_template = model.mean_template.detach().float().cpu()
    universes: dict[str, Mapping[str, object]] = {}
    for name, pairs, predicted, target in (
        ("heldout_train", heldout_pairs, heldout_predicted, heldout_target),
        ("val_cls", val_pairs, val_predicted, val_target),
    ):
        keep = torch.as_tensor([u != v for u, v in pairs])
        rows = int(keep.sum())
        universes[name] = _universe_report(
            predicted=predicted[keep].float(),
            target=target[keep].float(),
            weights=weights,
            mean_template=mean_template,
            fitted_constant=fitted,
            permutations=transplant_permutations(rows, count=permutations, seed=seed + 100),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    cells: dict[str, dict[str, float | None]] = {}
    logits_by_cell: dict[str, NDArray[np.float32]] = {"s2_pred": val_own_logits}
    logits_by_cell["s2_true"] = score_with_graph(
        model,
        val_pairs,
        pack_dir,
        device=device,
        amp=amp,
        token_budget=token_budget,
        weights=val_target,
    )
    logits_by_cell["s2_gates_off"] = score_with_graph(
        model,
        val_pairs,
        pack_dir,
        device=device,
        amp=amp,
        token_budget=token_budget,
        intervention="gates_off",
    )
    logits_by_cell["s1_pred"] = score_with_graph(
        stage1,
        val_pairs,
        pack_dir,
        device=device,
        amp=amp,
        token_budget=token_budget,
        weights=val_predicted,
    )
    logits_by_cell["s1_true"] = score_with_graph(
        stage1,
        val_pairs,
        pack_dir,
        device=device,
        amp=amp,
        token_budget=token_budget,
        weights=val_target,
    )
    graph_sources = {"true": "true", "pred": "predicted", "gates_off": "gates_off"}
    for name, logits in logits_by_cell.items():
        reader = stage1_id if name.startswith("s1_") else checkpoint_id
        _write_cell(
            output_dir,
            name,
            pairs=val_pairs,
            labels=val_labels,
            logits=logits,
            checkpoint_id=reader,
            strategy=strategy,
            graph_source=graph_sources[name.split("_", 1)[1]],
        )
        cells[name] = edge_metrics(logits, val_labels)

    return {
        "pilot": "motif_stage2_pilot_b",
        "checkpoint": str(checkpoint),
        "checkpoint_id": checkpoint_id,
        "stage1_checkpoint": str(stage1_checkpoint),
        "stage1_checkpoint_id": stage1_id,
        "strategy": strategy,
        "seed": seed,
        "loss_weights": {
            "beta_p": weights.beta_p,
            "beta_q": weights.beta_q,
            "beta_a": weights.beta_a,
            "beta_i": weights.beta_i,
            "beta_c": weights.beta_c,
            "huber_delta": weights.huber_delta,
        },
        "rows": {
            "fit_train": len(fit_pairs),
            "heldout_train": len(heldout_pairs),
            "val_cls": len(val_pairs),
            "node_disjoint": True,
            "fit_nodes": len(fit_nodes),
            "heldout_nodes": len(heldout_nodes),
            "split_kind": split_kind,
            "crossfit_fold": model.cfg.crossfit_fold,
            "crossfit_seed": model.cfg.crossfit_seed,
        },
        "diagnostic_sample_statistics": {
            "note": "statistics of the node-disjoint fit rows, not the F1 initialisation corpus; "
            "the biases actually installed are in the checkpoint run's profile.json "
            "under motif_generator.bias_init",
            "rows": int(corpus.rows),
            "nonzero_mean_by_type": [float(value) for value in corpus.nonzero_mean_by_type],
            "positive_row_wedge_mass": float(corpus.positive_row_wedge_mass),
            "positive_row_bridge_mass": float(corpus.positive_row_bridge_mass),
            "max_abs_diff_to_checkpoint_mean_template": float(
                (torch.from_numpy(corpus.mean).float() - mean_template).abs().max()
            ),
        },
        "fitted_constant_template": {
            "steps": fit_steps,
            "fitted_on": "node-disjoint training rows",
            "weights": [float(value) for value in fitted],
        },
        "level_1_fit": universes,
        "level_3_downstream": {
            "pairs_source": "val_cls",
            "truth": "true templates compiled on the V_val truth graph (labelled diagnostic)",
            "cells": cells,
        },
        "verdicts": {
            "level_1": level1_verdict(universes),
            "level_2": level2_verdict(universes),
            "level_3": level3_verdict(cells),
        },
    }


def markdown_summary(report: Mapping[str, object]) -> str:
    """Render the short pilot-B table.

    Args:
        report: A `run_pilot_b` payload.

    Returns:
        The markdown document.
    """
    universes = cast(Mapping[str, Mapping[str, object]], report["level_1_fit"])
    lines = [
        "# Motif Stage II pilot B",
        "",
        f"Checkpoint `{report['checkpoint_id']}`; "
        f"Stage I bundle `{report['stage1_checkpoint_id']}`.",
        "",
        "## Level 1 -- fit (L_G, the checkpoint's own betas)",
        "",
        "| universe | rows | generator | fitted constant | mean template | wedge recon | "
        "pred/true wedge mass |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, payload in universes.items():
        losses = cast(Mapping[str, float], payload["L_G"])
        wedge = cast(Mapping[str, Mapping[str, object]], payload["masses"])["wedge_mass"]
        recon = wedge["normalised_reconstruction_on_positive_rows"]
        ratio = wedge["predicted_over_true_mean"]
        lines.append(
            f"| {name} | {payload['rows']} | {losses['generator']:.5f} | "
            f"{losses['fitted_constant_template']:.5f} | {losses['mean_template']:.5f} | "
            + (f"{recon:.3f}" if isinstance(recon, float) else "n/a")
            + " | "
            + (f"{ratio:.4f}" if isinstance(ratio, float) else "n/a")
            + " |"
        )
    lines += [
        "",
        "## Level 2 -- transplant rise of L_G",
        "",
        "| universe | mean | min | max |",
        "|---|---|---|---|",
    ]
    for name, payload in universes.items():
        transplant = cast(Mapping[str, object], payload["transplant"])
        values = [
            transplant["relative_rise_mean"],
            transplant["relative_rise_min"],
            transplant["relative_rise_max"],
        ]
        rendered = " | ".join(f"{v:+.2%}" if isinstance(v, float) else "n/a" for v in values)
        lines.append(f"| {name} | {rendered} |")
    cells = cast(
        Mapping[str, Mapping[str, float | None]],
        cast(Mapping[str, object], report["level_3_downstream"])["cells"],
    )
    lines += [
        "",
        "## Level 3 -- downstream utility on val_cls",
        "",
        "| cell | AUROC | AUPRC |",
        "|---|---|---|",
    ]
    for name, metrics in cells.items():
        auroc, auprc = metrics["auroc"], metrics["auprc"]
        lines.append(
            f"| {name} | "
            + (f"{auroc:.4f}" if isinstance(auroc, float) else "n/a")
            + " | "
            + (f"{auprc:.4f}" if isinstance(auprc, float) else "n/a")
            + " |"
        )
    verdicts = cast(Mapping[str, Mapping[str, object]], report["verdicts"])
    lines += ["", "## Pre-registered verdicts", ""]
    for level, verdict in verdicts.items():
        mark = "PASS" if verdict["passed"] else "FAIL"
        lines.append(f"- **{level}**: {mark} -- {verdict['rule']}")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the command line.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Pilot B: read a Stage II motif-prompt prefix checkpoint at the three "
            "pre-registered levels (fit, conditional dependence, downstream utility)."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None, help="recorded for provenance only")
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--strategy", default="breadth_first")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=10)
    parser.add_argument("--train-rows", type=int, default=20_000)
    parser.add_argument("--heldout-rows", type=int, default=5_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", default="bf16", choices=["off", "bf16"])
    parser.add_argument("--token-budget", type=int, default=131_072)
    parser.add_argument("--fit-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run pilot B and write ``pilot_b.json`` and ``pilot_b.md``.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    report = run_pilot_b(
        checkpoint=args.checkpoint,
        stage1_checkpoint=args.stage1_checkpoint,
        pack_dir=args.pack_dir,
        data_root=args.data_root,
        strategy=args.strategy,
        output_dir=args.output_dir,
        permutations=args.permutations,
        train_rows=args.train_rows,
        heldout_rows=args.heldout_rows,
        device=su._resolve_device(args.device),
        amp=args.amp,
        token_budget=args.token_budget,
        seed=args.seed,
        fit_steps=args.fit_steps,
    )
    if args.config is not None:
        report["config"] = str(args.config)
    (args.output_dir / "pilot_b.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "pilot_b.md").write_text(markdown_summary(report), encoding="utf-8")
    logger.info("wrote %s", args.output_dir / "pilot_b.json")


if __name__ == "__main__":
    main()


__all__ = [
    "LEVEL1_MASS_FACTOR",
    "LEVEL1_RECONSTRUCTION_MAX",
    "LEVEL2_MIN_RELATIVE_RISE",
    "GraphLossWeights",
    "build_parser",
    "edge_metrics",
    "fit_constant_template",
    "gate_logit_summary",
    "level1_verdict",
    "level2_verdict",
    "level3_verdict",
    "main",
    "markdown_summary",
    "mass_report",
    "predicted_bank",
    "rows_inside",
    "run_pilot_b",
    "score_with_graph",
    "sorted_profile_dispersion",
    "split_training_nodes",
    "transplant_permutations",
    "transplant_report",
]

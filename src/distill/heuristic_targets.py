r"""Dump a closed-form-heuristic-teacher KD targets artifact (EHDM, B1 plan D7a).

CLI:

    python -m src.distill.heuristic_targets \
        --targets outputs/distill/kd_targets_breadth_first_seed0_v2 \
        --data-root data --strategy breadth_first --heuristic ra \
        --output outputs/distill/kd_targets_heuristic_ra_v1

``--heuristic`` is one of ``cn`` (Common Neighbors), ``aa`` (Adamic-Adar), or
``ra`` (Resource Allocation). Row identity (`pair_anchor_idx`/
`pair_partner_idx`/`anchor_offsets`/`is_near`/`pair_label`) and `node_ids`
are copied byte-identical from an existing v2 KD-targets artifact -- no
resampling -- so a D7a run differs from a D2 (learned full-ego oracle) run
only in *teacher provenance*: `teacher_logit` here comes from a
parameter-free structural heuristic instead of a trained model.

Truth graph: the SAME training-side truth graph `src.distill.teacher_targets`
uses -- `ValRegionSplit.build_training_graph()`, loopless training positives
over every train node -- and this module reuses (imports)
`teacher_targets.truth_graph_for_kd`/`assert_training_side_only` rather than
re-deriving the legality argument, so a stale/foreign artifact fails the same
way here that it would in the oracle-teacher path.

Per real pair (u, v), the score is a leave-edge-out link-prediction heuristic
on ``G - {(u, v)}`` (removed only if present): with ``CN(u, v)`` the common
neighbors of u and v in that edge-removed graph,

    CN  = |CN(u, v)|
    AA  = sum(1 / ln(deg(w)) for w in CN(u, v))
    RA  = sum(1 / deg(w)     for w in CN(u, v))

degrees taken in the edge-removed graph; a term whose denominator would be
zero or undefined (``deg(w) <= 1`` -- RA's ``1/deg(w)`` is undefined at 0 and
AA's ``ln(deg(w)) <= 0`` at 0 or 1) contributes 0 instead of raising. A
self-pair (``u == v``, should one appear) scores 0 before the log.

``teacher_logit := log1p(score)``, fp32. Plain `score` (unbounded, heavy-
tailed for CN especially) is not on the scale the KD losses expect: the LLP
distribution loss (`kd_dist_loss`) applies a fixed-temperature softmax over
each anchor's context group, which is not scale-invariant, so an unscaled
heuristic score would over/under-sharpen that softmax relative to the
oracle's already-logit-scaled `teacher_logit`. `log1p` compresses the heavy
tail while keeping the score monotonic and zero-preserving (`log1p(0) == 0`).

`teacher_pooled_ab`/`teacher_pooled_ba` are written as zeros (float16, same
second dimension as the source v2 artifact): a heuristic teacher has no
embedding to pool, and these arrays are never read by the rank/distribution
losses (`kd_rank_loss`/`kd_dist_loss`), only by the Gram/alignment KD losses
this arm does not use.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from src.distill.artifacts import load_kd_targets, write_kd_targets
from src.distill.teacher_targets import (
    _load_val_region_split,
    assert_training_side_only,
    truth_graph_for_kd,
)
from src.score_universe import _oracle_truth_graph_sha256

logger = logging.getLogger(__name__)

HEURISTICS: tuple[str, ...] = ("cn", "aa", "ra")

_MIN_DEGREE_FOR_TERM = 1  # deg(w) <= 1 -> the AA/RA term contributes 0


@contextmanager
def graph_without_edge(graph: nx.Graph, u: str, v: str) -> Iterator[None]:
    """Temporarily remove edge ``(u, v)`` from `graph`, if present, then restore it.

    A no-op when `u == v` (no self-loop to remove) or the edge is absent
    ("remove only if present"). `graph` is mutated in place for the
    duration of the ``with`` block only -- restored in a ``finally`` so an
    exception mid-block never leaves the graph edge-short.
    """
    removed = u != v and graph.has_edge(u, v)
    if removed:
        graph.remove_edge(u, v)
    try:
        yield
    finally:
        if removed:
            graph.add_edge(u, v)


def _aa_term(degree: int) -> float:
    """One Adamic-Adar summand; 0 when `degree` <= 1 (`ln(degree) <= 0`)."""
    return 1.0 / math.log(degree) if degree > _MIN_DEGREE_FOR_TERM else 0.0


def _ra_term(degree: int) -> float:
    """One Resource-Allocation summand; 0 when `degree` <= 1 (undefined)."""
    return 1.0 / degree if degree > _MIN_DEGREE_FOR_TERM else 0.0


def heuristic_score(graph: nx.Graph, u: str, v: str, heuristic: str) -> float:
    """Leave-edge-out CN/AA/RA score for one (u, v) pair on `graph`.

    `graph` is mutated transiently (edge removed, then restored) via
    `graph_without_edge`; the caller sees no net change.

    Args:
        graph: The training-side truth graph (loopless, over all train
            nodes); must already contain both `u` and `v`.
        u: Anchor node id.
        v: Partner node id.
        heuristic: One of `HEURISTICS`.

    Returns:
        The non-negative heuristic score (0.0 for a self-pair).

    Raises:
        ValueError: If `heuristic` is not one of `HEURISTICS`.
    """
    if heuristic not in HEURISTICS:
        raise ValueError(f"unknown heuristic {heuristic!r}; expected one of {HEURISTICS}")
    if u == v:
        return 0.0
    with graph_without_edge(graph, u, v):
        common = set(nx.common_neighbors(graph, u, v))
        if heuristic == "cn":
            return float(len(common))
        if heuristic == "aa":
            return sum(_aa_term(graph.degree(w)) for w in common)
        return sum(_ra_term(graph.degree(w)) for w in common)


def compute_heuristic_logits(
    graph: nx.Graph,
    node_ids: Sequence[str],
    pair_anchor_idx: NDArray[np.integer],
    pair_partner_idx: NDArray[np.integer],
    heuristic: str,
) -> NDArray[np.float32]:
    """Return `log1p(heuristic_score(...))` fp32, row-aligned with the pair arrays.

    Raises:
        ValueError: If any produced logit is non-finite.
    """
    n_pairs = len(pair_anchor_idx)
    scores = np.empty(n_pairs, dtype=np.float64)
    for row, (a, b) in enumerate(
        zip(pair_anchor_idx.tolist(), pair_partner_idx.tolist(), strict=True)
    ):
        scores[row] = heuristic_score(graph, node_ids[int(a)], node_ids[int(b)], heuristic)
        if (row + 1) % 10_000 == 0:
            logger.info("scored %d/%d heuristic rows", row + 1, n_pairs)
    teacher_logit = np.log1p(scores).astype(np.float32)
    if not np.isfinite(teacher_logit.astype(np.float64)).all():
        raise ValueError("heuristic teacher_logit contains non-finite values")
    return teacher_logit


def build_parser() -> argparse.ArgumentParser:
    """Build the `python -m src.distill.heuristic_targets` argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True, help="Existing v2 KD-targets dir")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--strategy", type=str, default="breadth_first")
    parser.add_argument("--heuristic", type=str, required=True, choices=HEURISTICS)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the heuristic-teacher KD-targets dumper CLI."""
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)

    v2 = load_kd_targets(args.targets)

    split, test_nodes = _load_val_region_split(args.data_root, args.strategy)
    truth_graph = truth_graph_for_kd(split)
    assert_training_side_only(v2.node_ids, truth_graph, split, test_nodes)

    teacher_logit = compute_heuristic_logits(
        truth_graph, v2.node_ids, v2.pair_anchor_idx, v2.pair_partner_idx, args.heuristic
    )

    pooled_dim = v2.teacher_pooled_ab.shape[-1]
    n_pairs = len(v2.pair_anchor_idx)
    pooled_ab = np.zeros((n_pairs, pooled_dim), dtype=np.float16)
    pooled_ba = np.zeros((n_pairs, pooled_dim), dtype=np.float16)

    write_kd_targets(
        args.output,
        node_ids=v2.node_ids,
        pair_anchor_idx=v2.pair_anchor_idx,
        pair_partner_idx=v2.pair_partner_idx,
        anchor_offsets=v2.anchor_offsets,
        teacher_logit=teacher_logit,
        teacher_pooled_ab=pooled_ab,
        teacher_pooled_ba=pooled_ba,
        is_near=v2.is_near,
        pair_label=v2.pair_label,
        truth_graph_sha256=_oracle_truth_graph_sha256(truth_graph),
        # No model checkpoint produces a heuristic teacher; the checkpoint_*
        # fields carry the heuristic identity instead (informational, never
        # gated -- CLAUDE.md: provenance metadata, not a formalism gate).
        checkpoint_path=Path(f"heuristic:{args.heuristic}"),
        checkpoint_sha256="",
        checkpoint_id=None,
        k_near=int(cast(int, v2.manifest["k_near"])),
        k_rand=int(cast(int, v2.manifest["k_rand"])),
        seed=int(cast(int, v2.manifest["seed"])),
    )

    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["teacher"] = f"heuristic_{args.heuristic}"
    manifest["source_targets_dir"] = str(args.targets)
    manifest["source_targets_manifest"] = {
        "checkpoint_path": v2.manifest.get("checkpoint_path"),
        "checkpoint_id": v2.manifest.get("checkpoint_id"),
        "truth_graph_sha256": v2.manifest.get("truth_graph_sha256"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("wrote heuristic_%s targets artifact to %s", args.heuristic, args.output)


if __name__ == "__main__":
    main()


__all__ = [
    "HEURISTICS",
    "build_parser",
    "compute_heuristic_logits",
    "graph_without_edge",
    "heuristic_score",
    "main",
]

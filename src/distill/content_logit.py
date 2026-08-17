r"""Append a content-baseline logit column to a v2 KD teacher-target artifact.

CLI (B1 plan D5, "topology-residual distillation"):

    python -m src.distill.content_logit \
        --targets outputs/distill/kd_targets_breadth_first_seed0_v2 \
        --checkpoint outputs/b0_v31/best.pt --data-root data \
        --strategy breadth_first \
        --output outputs/distill/kd_targets_breadth_first_seed0_v3

Scores every (anchor, partner) row of an existing v2 KD teacher-target
artifact with a frozen `V3_1` checkpoint -- the "content baseline"
(`model.config.node_factor_dim: 0`, no KD) -- fp32 with no autocast, and
writes a NEW artifact directory whose pair/anchor/label/teacher arrays are
copied byte-identical from v2 plus the new `content_logit` array
(`src.distill.artifacts.write_kd_targets`'s `content_logit` parameter bumps
the written manifest ``"format"`` to ``"kd_targets_v3"``). No resampling: the
row order and content are otherwise untouched.

D5's residual loss needs `teacher_logit - content_logit` (`src/train_b0.py`'s
`KDStream`): the full-ego oracle teacher's signal minus what a content-only
model already explains. This module produces only the second term.

Scoring path: reuses `score_universe._score_v3_1` -- the length-bucketed
`TokenPairDataset`/`LengthBucketedBatchSampler`/`collate_token_pairs`
machinery (`src/data/pairs.py`) already wired to a `FeatureStore`
(`src/data/features.py`) -- rather than reimplementing feature loading. This
is deliberately NOT `score_universe`'s CLI pair sources (`file:`/
`candidate`/`test`): those are held-out-ledgered (see
`src.distill.teacher_targets`'s module docstring for why that is wrong for a
training-side pass). This module's pairs come only from an already-legal v2
KD-targets artifact's own rows; the training-side legality check
(`assert_training_side_only`) is re-run defensively against the v2 artifact's
own node universe before scoring.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

from src.data.features import FeatureStore
from src.distill.artifacts import load_kd_targets, write_kd_targets
from src.distill.teacher_targets import (
    _DEFAULT_TOKEN_BUDGET,
    _FEATURES_SUBDIR,
    _load_val_region_split,
    assert_training_side_only,
    truth_graph_for_kd,
)
from src.model.egostitch.classifier.b0_v31 import V3_1
from src.score_universe import _file_sha256, _load_checkpoint, _score_v3_1

logger = logging.getLogger(__name__)


def score_content_logit(
    model: V3_1,
    node_ids: Sequence[str],
    pair_anchor_idx: NDArray[np.integer],
    pair_partner_idx: NDArray[np.integer],
    store: FeatureStore,
    *,
    device: torch.device,
    token_budget: int = _DEFAULT_TOKEN_BUDGET,
) -> NDArray[np.float32]:
    """Score every v2 row with the frozen content-baseline model, fp32, no autocast.

    Args:
        model: A frozen `V3_1` model (the content baseline), already on
            `device` and in ``eval()`` mode.
        node_ids: The v2 artifact's node universe (`KDTargets.node_ids`).
        pair_anchor_idx: The v2 artifact's `pair_anchor_idx` (row-aligned).
        pair_partner_idx: The v2 artifact's `pair_partner_idx` (row-aligned).
        store: Feature store providing per-node token sequences.
        device: Compute device.
        token_budget: Approximate per-batch token budget for the bucketed
            sampler that `_score_v3_1` uses internally.

    Returns:
        Shape ``(n_pairs,)`` float32 logits, row-aligned with the input pairs.

    Raises:
        ValueError: If any produced logit is non-finite.
    """
    pairs = [
        (node_ids[int(a)], node_ids[int(b)])
        for a, b in zip(pair_anchor_idx.tolist(), pair_partner_idx.tolist(), strict=True)
    ]
    logit = _score_v3_1(model, pairs, store, device=device, amp="off", token_budget=token_budget)
    if not np.isfinite(logit.astype(np.float64)).all():
        raise ValueError("content_logit scoring produced a non-finite logit")
    return logit.astype(np.float32)


def build_parser() -> argparse.ArgumentParser:
    """Build the `python -m src.distill.content_logit` argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True, help="Existing v2 KD-targets dir")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Frozen V3_1 checkpoint")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--strategy", type=str, default="breadth_first")
    parser.add_argument("--output", type=Path, required=True, help="New v3 artifact directory")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--token-budget", type=int, default=_DEFAULT_TOKEN_BUDGET)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the content-logit append CLI."""
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)

    v2 = load_kd_targets(args.targets)

    # Defensive re-check: the v2 artifact's own node universe must still be
    # training-side-only under the current split (fail closed on drift).
    split, test_nodes = _load_val_region_split(args.data_root, args.strategy)
    truth_graph = truth_graph_for_kd(split)
    assert_training_side_only(v2.node_ids, truth_graph, split, test_nodes)

    device = torch.device(args.device)
    model, model_family, checkpoint_id = _load_checkpoint(args.checkpoint)
    if model_family != "v3_1" or not isinstance(model, V3_1):
        raise ValueError(
            f"content_logit requires a 'v3_1' checkpoint (the content baseline), "
            f"got model_family={model_family!r}"
        )
    model.to(device)
    model.eval()

    store = FeatureStore(args.data_root / _FEATURES_SUBDIR)
    content_logit = score_content_logit(
        model,
        v2.node_ids,
        v2.pair_anchor_idx,
        v2.pair_partner_idx,
        store,
        device=device,
        token_budget=args.token_budget,
    )

    content_checkpoint_sha256 = _file_sha256(args.checkpoint)
    v2_checkpoint_id = v2.manifest.get("checkpoint_id")
    write_kd_targets(
        args.output,
        node_ids=v2.node_ids,
        pair_anchor_idx=v2.pair_anchor_idx,
        pair_partner_idx=v2.pair_partner_idx,
        anchor_offsets=v2.anchor_offsets,
        teacher_logit=v2.teacher_logit,
        teacher_pooled_ab=v2.teacher_pooled_ab,
        teacher_pooled_ba=v2.teacher_pooled_ba,
        is_near=v2.is_near,
        pair_label=v2.pair_label,
        content_logit=content_logit,
        # The primary checkpoint_* fields keep referring to the v2 artifact's
        # own (oracle-teacher) checkpoint -- unchanged meaning from v2 -- so
        # they stay comparable across v2/v3; the content checkpoint gets its
        # own clearly-named fields below.
        truth_graph_sha256=str(v2.manifest.get("truth_graph_sha256", "")),
        checkpoint_path=Path(str(v2.manifest.get("checkpoint_path", ""))),
        checkpoint_sha256=str(v2.manifest.get("checkpoint_sha256", "")),
        checkpoint_id=v2_checkpoint_id if isinstance(v2_checkpoint_id, str) else None,
        k_near=int(cast(int, v2.manifest["k_near"])),
        k_rand=int(cast(int, v2.manifest["k_rand"])),
        seed=int(cast(int, v2.manifest["seed"])),
    )

    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["v2_provenance"] = {
        "source_targets_dir": str(args.targets),
        "checkpoint_path": v2.manifest.get("checkpoint_path"),
        "checkpoint_sha256": v2.manifest.get("checkpoint_sha256"),
        "checkpoint_id": v2.manifest.get("checkpoint_id"),
        "truth_graph_sha256": v2.manifest.get("truth_graph_sha256"),
    }
    manifest["content_checkpoint_path"] = str(args.checkpoint)
    manifest["content_checkpoint_sha256"] = content_checkpoint_sha256
    manifest["content_checkpoint_id"] = checkpoint_id
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("wrote content_logit artifact to %s (format=kd_targets_v3)", args.output)


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "main", "score_content_logit"]

"""Oracle validation context, row alignment and diagnostic-only evaluation."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from src.distill.artifacts import KDRowTargets, load_kd_targets, write_kd_targets
from src.distill.config import DistillConfig
from src.distill.validation import OracleValidationBank
from src.train_b0 import _evaluate_distributed

pytestmark = pytest.mark.unit

PAIRS = [("a", "b"), ("a", "c"), ("b", "c")]
LABELS = [1, 0, 1]


def targets() -> KDRowTargets:
    return KDRowTargets(
        node_ids=["a", "b", "c"],
        pair_a_idx=np.array([0, 0, 1], dtype=np.int32),
        pair_b_idx=np.array([1, 2, 2], dtype=np.int32),
        pair_label=np.array(LABELS, dtype=np.int8),
        teacher_logit=np.array([1.0, -1.0, 2.0], dtype=np.float32),
        teacher_rep=np.eye(3, dtype=np.float16),
        manifest={"truth_source": "validation_structure"},
    )


@pytest.mark.parametrize("arm", ["logit", "rep", "gram", "rank"])
def test_perfect_agreement_and_wrong_structure(arm: str) -> None:
    weights = {f"w_{arm}": 1.0}
    if arm == "rank":
        weights["w_dist"] = 1.0
    cfg = DistillConfig.from_mapping(
        {"targets_path": "train", "context_targets_path": "ctx" if arm == "rank" else "", **weights}
    )
    t = targets()
    bank = OracleValidationBank(t, cfg, PAIRS, LABELS)
    metrics = bank.metrics(t.teacher_logit, t.teacher_rep)
    assert metrics["val_prob_mae"] == 0
    assert metrics["val_logit_pearson"] == pytest.approx(1.0)
    if arm in {"rep", "gram"}:
        key = "val_kd_rep_loss" if arm == "rep" else "val_kd_gram_block_loss"
        assert metrics[key] == pytest.approx(0.0)
    if arm == "rank":
        assert metrics["val_kd_dist_loss"] == pytest.approx(0.0, abs=1e-7)
    with pytest.raises(ValueError, match="G_val"):
        OracleValidationBank(
            replace(t, manifest={"truth_source": "training_structure"}), cfg, PAIRS, LABELS
        )
    with pytest.raises(ValueError, match="join mismatch"):
        OracleValidationBank(t, cfg, PAIRS[::-1], LABELS)


def test_diagnostic_archive_cannot_load_as_training(tmp_path: Path) -> None:
    t = targets()
    write_kd_targets(
        tmp_path,
        node_ids=t.node_ids,
        pair_a_idx=t.pair_a_idx,
        pair_b_idx=t.pair_b_idx,
        pair_label=t.pair_label,
        teacher_logit=t.teacher_logit,
        teacher_rep=t.teacher_rep,
        truth_graph_sha256="record",
        checkpoint_path=Path("teacher.pt"),
        checkpoint_sha256="record",
        checkpoint_id="same-teacher",
        rep_source="topo",
        truth_source="validation_structure",
    )
    with pytest.raises(ValueError, match="truth_source"):
        load_kd_targets(tmp_path)
    loaded = load_kd_targets(tmp_path, truth_source="validation_structure")
    assert loaded.manifest["format"] == "oracle_val_diagnostics_v1"
    np.testing.assert_array_equal(loaded.teacher_logit, t.teacher_logit)


class Scorer(torch.nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"logits": batch["seed"], "kd_rep": batch["rep"]}


def test_evaluation_joins_shuffled_batches_without_changing_task_loss() -> None:
    t = targets()
    cfg = DistillConfig(targets_path="train", w_rep=1.0)
    bank = OracleValidationBank(t, cfg, PAIRS, LABELS)
    batches = []
    for ids in ([2, 0], [1]):
        batches.append(
            {
                "_row_id": torch.tensor(ids),
                "label": torch.tensor(np.array(LABELS)[ids]),
                "seed": torch.tensor(t.teacher_logit[ids]),
                "rep": torch.tensor(t.teacher_rep[ids]),
            }
        )
    accelerator = Accelerator(cpu=True)
    baseline = _evaluate_distributed(Scorer(), batches, accelerator)
    diagnosed = _evaluate_distributed(Scorer(), batches, accelerator, validation_bank=bank)
    assert diagnosed.task_loss == baseline.task_loss
    assert diagnosed.metrics == baseline.metrics
    assert baseline.diagnostics is None
    assert diagnosed.diagnostics is not None
    assert diagnosed.diagnostics["val_kd_rep_loss"] == pytest.approx(0.0)

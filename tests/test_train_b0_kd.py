"""Tests for src.train_b0's `KDRowBank`: the row-exact-join, same-batch KD rework.

Covers: (a) the row-exact join and its failure modes, (b) task/KD sharing exact
row IDs, (c) exactly one forward and one backward per batch under a real KD
bank, (d) the zero-weight/no-`distill:` matched control, (e) the sampled
anchor-context stream and its obsolete artifact fields are fully gone, (f)
epoch telemetry keys and `_pearson_from_moments`, (g) the `_evaluate_distributed`
validation-only KD diagnostics, and (h) architecture-mismatch fail-closed checks.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import src.train_b0 as train_b0
import torch
from accelerate import Accelerator
from src.distill.artifacts import KDRowTargets, load_kd_targets, write_kd_targets
from src.distill.config import DistillConfig
from src.distill.losses import kd_logit_loss
from src.model.egostitch.classifier.b0_v31 import V3_1
from src.train_b0 import (
    KDRowBank,
    KDValDiagnostics,
    ValidationOutcome,
    _evaluate_distributed,
    _pearson_from_moments,
    train_ddp_loop,
)
from torch import nn

from tests.test_train_b0 import (
    _batch_of,
    _constant_metrics,
    _make_synthetic_pair_dataset,
    _tiny_config,
    _TinyPairMLP,
)

pytestmark = pytest.mark.unit

Pair = tuple[str, str]


# --------------------------------------------------------------------------- artifact helpers


def _node_index(node_ids: list[str], pairs: list[Pair]) -> tuple[np.ndarray, np.ndarray]:
    index = {node: position for position, node in enumerate(node_ids)}
    a_idx = np.array([index[a] for a, _ in pairs], dtype=np.int32)
    b_idx = np.array([index[b] for _, b in pairs], dtype=np.int32)
    return a_idx, b_idx


def _write_targets(
    out_dir: Path,
    *,
    node_ids: list[str],
    train_pairs: list[Pair],
    train_labels: list[int],
    teacher_logit: np.ndarray,
    teacher_rep: np.ndarray | None = None,
    val_pairs: list[Pair] | None = None,
    val_labels: list[int] | None = None,
) -> None:
    """Write one minimal, internally consistent KD row-targets artifact."""
    a_idx, b_idx = _node_index(node_ids, train_pairs)
    rep_dim = teacher_rep.shape[1] if teacher_rep is not None else 4
    rep = (
        teacher_rep
        if teacher_rep is not None
        else np.zeros((len(train_pairs), rep_dim), dtype=np.float32)
    )

    v_pairs = val_pairs if val_pairs is not None else train_pairs[:1]
    v_labels = val_labels if val_labels is not None else train_labels[:1]
    v_a_idx, v_b_idx = _node_index(node_ids, v_pairs)
    v_logit = np.asarray(teacher_logit[: len(v_pairs)], dtype=np.float32)
    v_rep = rep[: len(v_pairs)]

    write_kd_targets(
        out_dir,
        node_ids=node_ids,
        pair_a_idx=a_idx,
        pair_b_idx=b_idx,
        pair_label=np.asarray(train_labels, dtype=np.int8),
        teacher_logit=np.asarray(teacher_logit, dtype=np.float32),
        teacher_rep=np.asarray(rep, dtype=np.float32),
        val_pair_a_idx=v_a_idx,
        val_pair_b_idx=v_b_idx,
        val_pair_label=np.asarray(v_labels, dtype=np.int8),
        val_teacher_logit=v_logit,
        val_teacher_rep=v_rep,
        truth_graph_sha256="0" * 64,
        checkpoint_path=out_dir / "ckpt.pt",
        checkpoint_sha256="1" * 64,
        checkpoint_id=None,
    )


def _ring_pairs(node_ids: list[str]) -> list[Pair]:
    """One row per node, paired to its ring successor -- a simple distinct-row set."""
    n = len(node_ids)
    return [(node_ids[i], node_ids[(i + 1) % n]) for i in range(n)]


# --------------------------------------------------------------------------- (a) row-exact join


class TestKDRowBankJoin:
    _NODE_IDS = [f"n{i}" for i in range(4)]
    _TRAIN_PAIRS = _ring_pairs(_NODE_IDS)
    _TRAIN_LABELS = [1, 0, 1, 0]

    def _artifact(self, tmp_path: Path) -> KDRowTargets:
        _write_targets(
            tmp_path / "targets",
            node_ids=self._NODE_IDS,
            train_pairs=self._TRAIN_PAIRS,
            train_labels=self._TRAIN_LABELS,
            teacher_logit=np.linspace(-1.0, 1.0, 4).astype(np.float32),
        )
        return load_kd_targets(tmp_path / "targets")

    def _bank(
        self,
        tmp_path: Path,
        *,
        train_pairs: list[Pair],
        train_labels: list[int],
        val_pairs: list[Pair] | None = None,
        val_labels: list[int] | None = None,
    ) -> KDRowBank:
        targets = self._artifact(tmp_path)
        distill = DistillConfig(targets_path="t", w_logit=1.0)
        return KDRowBank(
            distill,
            targets,
            train_pairs=train_pairs,
            train_labels=train_labels,
            val_pairs=val_pairs if val_pairs is not None else self._TRAIN_PAIRS[:1],
            val_labels=val_labels if val_labels is not None else self._TRAIN_LABELS[:1],
            model=nn.Linear(1, 1),
            device=torch.device("cpu"),
        )

    def test_matching_rows_succeeds(self, tmp_path: Path) -> None:
        bank = self._bank(tmp_path, train_pairs=self._TRAIN_PAIRS, train_labels=self._TRAIN_LABELS)
        assert bank.arm == "kd_logit"

    def test_wrong_row_count_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="rows, trainer has"):
            self._bank(
                tmp_path,
                train_pairs=self._TRAIN_PAIRS[:3],
                train_labels=self._TRAIN_LABELS[:3],
            )

    def test_permuted_rows_raise(self, tmp_path: Path) -> None:
        permuted_pairs = [
            self._TRAIN_PAIRS[1],
            self._TRAIN_PAIRS[0],
            self._TRAIN_PAIRS[2],
            self._TRAIN_PAIRS[3],
        ]
        permuted_labels = [
            self._TRAIN_LABELS[1],
            self._TRAIN_LABELS[0],
            self._TRAIN_LABELS[2],
            self._TRAIN_LABELS[3],
        ]
        with pytest.raises(ValueError, match="training block endpoints do not match"):
            self._bank(tmp_path, train_pairs=permuted_pairs, train_labels=permuted_labels)

    def test_mismatched_endpoint_raises(self, tmp_path: Path) -> None:
        bad_pairs = list(self._TRAIN_PAIRS)
        # Row 2's artifact endpoint is (n2, n3); swap in a different partner.
        bad_pairs[2] = (bad_pairs[2][0], "n1")
        with pytest.raises(ValueError, match="training block endpoints do not match"):
            self._bank(tmp_path, train_pairs=bad_pairs, train_labels=self._TRAIN_LABELS)

    def test_mismatched_label_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="training block labels do not match"):
            self._bank(tmp_path, train_pairs=self._TRAIN_PAIRS, train_labels=[1, 1, 1, 0])

    def test_mismatched_val_block_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="validation block labels do not match"):
            self._bank(
                tmp_path,
                train_pairs=self._TRAIN_PAIRS,
                train_labels=self._TRAIN_LABELS,
                val_pairs=self._TRAIN_PAIRS[:1],
                val_labels=[1 - self._TRAIN_LABELS[0]],
            )


# --------------------------------------------------------------------------- (b) exact row IDs


def test_kd_loss_uses_exact_row_ids(tmp_path: Path) -> None:
    node_ids = [f"n{i}" for i in range(8)]
    train_pairs = _ring_pairs(node_ids)[:6]
    train_labels = [i % 2 for i in range(6)]
    teacher_logit = np.array([0.1 * i for i in range(6)], dtype=np.float32)
    _write_targets(
        tmp_path / "targets",
        node_ids=node_ids,
        train_pairs=train_pairs,
        train_labels=train_labels,
        teacher_logit=teacher_logit,
    )
    targets = load_kd_targets(tmp_path / "targets")
    distill = DistillConfig(targets_path="t", w_logit=1.0)
    bank = KDRowBank(
        distill,
        targets,
        train_pairs=train_pairs,
        train_labels=train_labels,
        val_pairs=train_pairs[:1],
        val_labels=train_labels[:1],
        model=nn.Linear(1, 1),
        device=torch.device("cpu"),
    )

    student_logit = torch.tensor([0.5, -0.2, 0.9])
    output = {"logits": student_logit}

    rows_a = torch.tensor([4, 1, 3])
    loss_a, stats_a, _ = bank.loss({"_row_id": rows_a}, output, global_step=1)
    expected_a = kd_logit_loss(student_logit, torch.tensor(teacher_logit[[4, 1, 3]]))
    assert torch.allclose(loss_a, expected_a, atol=1e-6)
    assert stats_a["rows"] == 3.0

    rows_b = torch.tensor([0, 2, 5])
    loss_b, _, _ = bank.loss({"_row_id": rows_b}, output, global_step=1)
    expected_b = kd_logit_loss(student_logit, torch.tensor(teacher_logit[[0, 2, 5]]))
    assert torch.allclose(loss_b, expected_b, atol=1e-6)
    assert not torch.allclose(loss_a, loss_b)


# --------------------------------------------------------------------------- (c) one fwd/one bwd


class _CountingLogitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.forward_calls = 0

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.forward_calls += 1
        n = batch["label"].shape[0]
        logits = self.weight * torch.ones(n)
        loss = self.weight * 0.0 + batch["loss_value"].mean()
        return {"loss": loss, "logits": logits}


def test_ddp_loop_one_forward_one_backward_per_batch_with_real_kd_bank(tmp_path: Path) -> None:
    node_ids = [f"n{i}" for i in range(4)]
    train_pairs = _ring_pairs(node_ids)
    train_labels = [0, 1, 0, 1]
    _write_targets(
        tmp_path / "targets",
        node_ids=node_ids,
        train_pairs=train_pairs,
        train_labels=train_labels,
        teacher_logit=np.zeros(4, dtype=np.float32),
    )
    targets = load_kd_targets(tmp_path / "targets")
    distill = DistillConfig(targets_path="t", w_logit=1.0)
    model = _CountingLogitModel()
    bank = KDRowBank(
        distill,
        targets,
        train_pairs=train_pairs,
        train_labels=train_labels,
        val_pairs=train_pairs[:1],
        val_labels=train_labels[:1],
        model=model,
        device=torch.device("cpu"),
    )

    def _loss_batch(row_ids: list[int]) -> dict[str, torch.Tensor]:
        n = len(row_ids)
        return {
            "loss_value": torch.zeros(n),
            "label": torch.tensor([float(train_labels[row]) for row in row_ids]),
            "_row_id": torch.tensor(row_ids),
            "_local_pair_count": torch.tensor(n),
            "_global_pair_count": torch.tensor(n),
        }

    batches = [_loss_batch([0]), _loss_batch([1, 2, 3])]
    cfg = _tiny_config(epochs=1)
    accelerator = Accelerator(cpu=True)
    backward_calls: list[int] = []
    original_backward = accelerator.backward

    def counting_backward(loss: torch.Tensor, **kwargs: object) -> None:
        backward_calls.append(1)
        original_backward(loss, **kwargs)

    accelerator.backward = counting_backward

    train_ddp_loop(
        model,
        lambda epoch: batches,
        batches,
        cfg,
        accelerator,
        warmup_steps=1,
        artifact_dir=tmp_path / "attempt",
        evaluate_fn=lambda model, loader, accelerator: ValidationOutcome(_constant_metrics(), None),
        kd_bank=bank,
    )

    assert model.forward_calls == len(batches)
    assert len(backward_calls) == len(batches)


# --------------------------------------------------------------------------- (d) matched control


def test_distill_config_default_and_all_zero_are_inactive_none_arm() -> None:
    assert DistillConfig().arm == "none"
    assert DistillConfig().active is False
    zero = DistillConfig.from_mapping(
        {"w_logit": 0.0, "w_rep": 0.0, "w_seed": 0.0, "w_geom": 0.0, "w_kl": 0.0}
    )
    assert zero.arm == "none"
    assert zero.active is False


def test_ddp_loop_distill_none_and_all_zero_are_bit_identical(tmp_path: Path) -> None:
    def _run(distill: DistillConfig | None, subdir: str) -> dict[str, torch.Tensor]:
        cfg = replace(_tiny_config(epochs=2), distill=distill)
        torch.manual_seed(7)
        model = _TinyPairMLP(input_dim=4, hidden_dims=(8,), dropout=0.0)
        batch = _batch_of(_make_synthetic_pair_dataset(8, input_dim=4, seed=1))
        batch["_row_id"] = torch.arange(8)
        batch["_local_pair_count"] = torch.tensor(8)
        batch["_global_pair_count"] = torch.tensor(8)
        result = train_ddp_loop(
            model,
            lambda epoch: [batch],
            [batch],
            cfg,
            Accelerator(cpu=True),
            warmup_steps=1,
            artifact_dir=tmp_path / subdir,
            evaluate_fn=lambda model, loader, accelerator: ValidationOutcome(
                _constant_metrics(), None
            ),
            kd_bank=None,
        )
        return result.last_state_dict

    none_state = _run(None, "none")
    zero_state = _run(
        DistillConfig.from_mapping(
            {"w_logit": 0.0, "w_rep": 0.0, "w_seed": 0.0, "w_geom": 0.0, "w_kl": 0.0}
        ),
        "zero",
    )

    assert none_state.keys() == zero_state.keys()
    for key in none_state:
        torch.testing.assert_close(none_state[key], zero_state[key], rtol=0.0, atol=0.0)


# --------------------------------------------------------------------------- (e) obsolete paths


def test_obsolete_kd_sampler_modules_are_gone() -> None:
    for module_name in (
        "src.distill.context_sampler",
        "src.distill.content_logit",
        "src.distill.heuristic_targets",
    ):
        assert importlib.util.find_spec(module_name) is None
    assert not hasattr(train_b0, "KDStream")


def test_kd_row_targets_has_no_obsolete_fields() -> None:
    field_names = {field.name for field in dataclasses.fields(KDRowTargets)}
    assert field_names.isdisjoint(
        {"anchor_offsets", "is_near", "teacher_pooled_ab", "content_logit"}
    )


def test_distill_config_has_no_obsolete_fields() -> None:
    field_names = {field.name for field in dataclasses.fields(DistillConfig)}
    assert field_names.isdisjoint(
        {"anchors_per_step", "temperature", "margin", "arm_label", "w_rank"}
    )


# --------------------------------------------------------------------------- (f) telemetry


def _tiny_v31_kwargs() -> dict[str, object]:
    return {
        "input_dim": 4,
        "d_model": 8,
        "encoder_layers": 1,
        "cross_attn_layers": 1,
        "n_heads": 2,
        "mlp_head": {"hidden_dims": [8], "dropout": 0.0},
        "regularization": {"dropout": 0.0},
    }


def _v31_batch(row_ids: list[int], labels: list[int], *, seed: int = 44) -> dict[str, torch.Tensor]:
    n = len(row_ids)
    generator = torch.Generator().manual_seed(seed)
    return {
        "emb_a": torch.randn(n, 3, 4, generator=generator),
        "emb_b": torch.randn(n, 3, 4, generator=generator),
        "len_a": torch.full((n,), 3, dtype=torch.long),
        "len_b": torch.full((n,), 3, dtype=torch.long),
        "label": torch.tensor([float(value) for value in labels]),
        "_row_id": torch.tensor(row_ids),
        "_local_pair_count": torch.tensor(n),
        "_global_pair_count": torch.tensor(n),
    }


def test_ddp_loop_kd_logit_telemetry_keys(tmp_path: Path) -> None:
    node_ids = [f"n{i}" for i in range(4)]
    train_pairs = _ring_pairs(node_ids)
    train_labels = [1, 0, 1, 0]
    teacher_logit = np.array([0.2, -0.4, 0.6, -0.1], dtype=np.float32)
    _write_targets(
        tmp_path / "targets",
        node_ids=node_ids,
        train_pairs=train_pairs,
        train_labels=train_labels,
        teacher_logit=teacher_logit,
    )
    targets = load_kd_targets(tmp_path / "targets")
    distill = DistillConfig(targets_path="t", w_logit=1.0)
    model = V3_1(**_tiny_v31_kwargs())
    bank = KDRowBank(
        distill,
        targets,
        train_pairs=train_pairs,
        train_labels=train_labels,
        val_pairs=train_pairs[:1],
        val_labels=train_labels[:1],
        model=model,
        device=torch.device("cpu"),
    )

    batch = _v31_batch([0, 1, 2, 3], train_labels)
    cfg = replace(_tiny_config(epochs=1), distill=distill)

    result = train_ddp_loop(
        model,
        lambda epoch: [batch],
        [batch],
        cfg,
        Accelerator(cpu=True),
        warmup_steps=1,
        artifact_dir=tmp_path / "attempt",
        evaluate_fn=lambda model, loader, accelerator: ValidationOutcome(_constant_metrics(), None),
        kd_bank=bank,
    )

    entry = result.history[0]
    for key in (
        "train_kd_loss",
        "kd_logit_corr",
        "kd_prob_mae",
        "grad_norm_task",
        "grad_norm_kd",
        "val_ece",
        "val_brier",
    ):
        assert key in entry, f"missing telemetry key {key!r} in {sorted(entry)}"


def test_pearson_from_moments_perfect_positive_correlation() -> None:
    values = [1.0, 2.0, 3.0]
    n = float(len(values))
    sum_s = sum_t = sum(values)
    sum_s2 = sum_t2 = sum(value * value for value in values)
    sum_st = sum(value * value for value in values)
    assert _pearson_from_moments(sum_s, sum_t, sum_s2, sum_t2, sum_st, n) == pytest.approx(1.0)


def test_pearson_from_moments_perfect_negative_correlation() -> None:
    student = [1.0, 2.0, 3.0]
    teacher = [3.0, 2.0, 1.0]
    n = 3.0
    sum_s, sum_t = sum(student), sum(teacher)
    sum_s2 = sum(value * value for value in student)
    sum_t2 = sum(value * value for value in teacher)
    sum_st = sum(s * t for s, t in zip(student, teacher, strict=True))
    assert _pearson_from_moments(sum_s, sum_t, sum_s2, sum_t2, sum_st, n) == pytest.approx(-1.0)


def test_pearson_from_moments_zero_variance_guard() -> None:
    student = [1.0, 1.0, 1.0]
    teacher = [1.0, 2.0, 3.0]
    n = 3.0
    sum_s, sum_t = sum(student), sum(teacher)
    sum_s2 = sum(value * value for value in student)
    sum_t2 = sum(value * value for value in teacher)
    sum_st = sum(s * t for s, t in zip(student, teacher, strict=True))
    assert _pearson_from_moments(sum_s, sum_t, sum_s2, sum_t2, sum_st, n) == pytest.approx(0.0)


# --------------------------------------------------------------------------- (g) val diagnostics


class _RepDiagModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "logits": self.scale * batch["logit_seed"],
            "pair_repr": self.scale * batch["rep_seed"],
        }


def test_evaluate_distributed_kd_rep_diagnostics() -> None:
    n = 3
    rep_dim = 4
    rep_seed = torch.randn(n, rep_dim)
    batch = {
        "label": torch.tensor([1.0, 0.0, 1.0]),
        "_row_id": torch.tensor([0, 1, 2]),
        "logit_seed": torch.tensor([0.5, -0.5, 1.0]),
        "rep_seed": rep_seed,
    }
    teacher_logit = torch.tensor([0.4, -0.6, 0.9])
    kd_val = KDValDiagnostics(
        arm="kd_rep",
        teacher_logit=teacher_logit,
        teacher_logit_np=teacher_logit.double().numpy(),
        teacher_rep=rep_seed.clone().to(torch.float16),
        teacher_seeds=None,
    )
    outcome = _evaluate_distributed(
        _RepDiagModel(),
        [batch],
        Accelerator(cpu=True),
        expected_row_ids=np.arange(n),
        kd_val=kd_val,
    )
    assert outcome.kd is not None
    assert {"val_kd_rep_cos", "val_kd_logit_corr", "val_kd_prob_mae"} <= outcome.kd.keys()
    assert outcome.kd["val_kd_rep_cos"] == pytest.approx(1.0, abs=1e-3)


def test_evaluate_distributed_without_kd_val_leaves_kd_field_none() -> None:
    n = 2
    batch = {
        "label": torch.tensor([1.0, 0.0]),
        "_row_id": torch.tensor([0, 1]),
        "logit_seed": torch.tensor([0.1, -0.2]),
        "rep_seed": torch.randn(n, 4),
    }
    outcome = _evaluate_distributed(
        _RepDiagModel(),
        [batch],
        Accelerator(cpu=True),
        expected_row_ids=np.arange(n),
        kd_val=None,
    )
    assert outcome.kd is None


# --------------------------------------------------------------------------- (h) arch checks


class TestKDRowBankArchChecks:
    _NODE_IDS = [f"n{i}" for i in range(4)]
    _TRAIN_PAIRS = _ring_pairs(_NODE_IDS)
    _TRAIN_LABELS = [1, 0, 1, 0]

    def _targets(self, tmp_path: Path, *, teacher_rep_dim: int = 4) -> KDRowTargets:
        teacher_rep = np.random.default_rng(0).normal(size=(4, teacher_rep_dim)).astype(np.float32)
        _write_targets(
            tmp_path / "targets",
            node_ids=self._NODE_IDS,
            train_pairs=self._TRAIN_PAIRS,
            train_labels=self._TRAIN_LABELS,
            teacher_logit=np.zeros(4, dtype=np.float32),
            teacher_rep=teacher_rep,
        )
        return load_kd_targets(tmp_path / "targets")

    def _build(self, distill: DistillConfig, targets: KDRowTargets, model: nn.Module) -> KDRowBank:
        return KDRowBank(
            distill,
            targets,
            train_pairs=self._TRAIN_PAIRS,
            train_labels=self._TRAIN_LABELS,
            val_pairs=self._TRAIN_PAIRS[:1],
            val_labels=self._TRAIN_LABELS[:1],
            model=model,
            device=torch.device("cpu"),
        )

    def test_w_rep_mismatched_width_without_kd_rep_head_raises(self, tmp_path: Path) -> None:
        targets = self._targets(tmp_path, teacher_rep_dim=6)
        distill = DistillConfig(targets_path="t", w_rep=1.0)
        model = nn.Module()
        model.d_model = 8  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="d_model"):
            self._build(distill, targets, model)

    def test_kd_rep_head_present_with_w_rep_zero_raises(self, tmp_path: Path) -> None:
        targets = self._targets(tmp_path, teacher_rep_dim=4)
        distill = DistillConfig(targets_path="t", w_logit=1.0)  # w_rep == 0
        model = nn.Module()
        model.kd_rep_head = nn.Linear(4, 4)
        with pytest.raises(RuntimeError, match=r"distill\.w_rep > 0"):
            self._build(distill, targets, model)

    def test_kd_d9_without_pair_latent_gen_raises(self, tmp_path: Path) -> None:
        targets = self._targets(tmp_path)
        distill = DistillConfig(targets_path="t", w_seed=1.0, w_geom=1.0, w_kl=1.0)
        model = nn.Module()
        with pytest.raises(RuntimeError, match="pair_latent_gen"):
            self._build(distill, targets, model)

    def test_pair_latent_gen_present_with_w_seed_zero_raises(self, tmp_path: Path) -> None:
        targets = self._targets(tmp_path)
        distill = DistillConfig(targets_path="t", w_logit=1.0)  # w_seed == 0
        model = nn.Module()
        model.pair_latent_gen = nn.Module()
        with pytest.raises(RuntimeError, match=r"distill\.w_seed > 0"):
            self._build(distill, targets, model)

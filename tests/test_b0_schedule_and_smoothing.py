"""Tests for the B0 ``optim.scheduler`` block and V3.1 loss label smoothing.

Both features exist to reproduce the legacy V3.1 training recipe (OneCycle LR +
0.05 label smoothing). The invariants that matter here are: a config *without* a
scheduler block must keep the historical warmup-then-constant schedule bit for
bit, OneCycle must be sized from an exact step count rather than an extrapolated
one, and smoothing must change only the loss, never the logits.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
import torch
import yaml
from src.model.egostitch.classifier.b0_v31 import V3_1
from src.train_b0 import (
    Config,
    SchedulerConfig,
    _build_scheduler,
    _count_single_process_steps,
    _parse_scheduler,
    _step_scheduler,
    config_to_dict,
    load_config,
)
from torch import nn

from tests.test_train_b0 import _write_yaml_config

pytestmark = pytest.mark.unit


ONECYCLE_BLOCK: dict[str, object] = {
    "type": "onecycle",
    "max_lr": 1.0e-4,
    "pct_start": 0.1,
    "div_factor": 25.0,
    "final_div_factor": 10000.0,
    "anneal_strategy": "cos",
}


def _config_with_scheduler(tmp_path: Path, scheduler: object, epochs: int = 4) -> Config:
    """Load a `Config` whose ``optim`` carries `scheduler` (or no block)."""
    path = tmp_path / "config.yaml"
    overrides: dict[str, object] = {"optim.epochs": epochs}
    _write_yaml_config(path, overrides)
    raw = json.loads(path.read_text())
    optim = cast(dict[str, object], raw["optim"])
    if scheduler is not None:
        optim["scheduler"] = scheduler
    path.write_text(json.dumps(raw))
    return load_config(path)


def _tiny_v31(**extra: object) -> V3_1:
    """Build a minimal V3.1 whose forward is cheap enough for a unit test."""
    config: dict[str, object] = {
        "input_dim": 8,
        "d_model": 8,
        "encoder_layers": 1,
        "cross_attn_layers": 1,
        "n_heads": 2,
        "mlp_head": {"hidden_dims": [8], "dropout": 0.0},
        "regularization": {"dropout": 0.0},
    }
    config.update(extra)
    return V3_1(**config)


# --------------------------------------------------------------------------- parsing


class TestParseScheduler:
    """`_parse_scheduler` validation."""

    def test_absent_block_returns_none(self) -> None:
        """No scheduler block selects the historical schedule."""
        assert _parse_scheduler(None) is None

    def test_parses_the_legacy_onecycle_block(self) -> None:
        """A well-formed block round-trips into `SchedulerConfig`."""
        parsed = _parse_scheduler(dict(ONECYCLE_BLOCK))
        assert parsed == SchedulerConfig(
            type="onecycle",
            max_lr=1.0e-4,
            pct_start=0.1,
            div_factor=25.0,
            final_div_factor=10000.0,
            anneal_strategy="cos",
        )

    def test_rejects_unknown_scheduler_type(self) -> None:
        """An unsupported scheduler name fails loudly rather than silently."""
        block = dict(ONECYCLE_BLOCK) | {"type": "cosine"}
        with pytest.raises(ValueError, match="optim.scheduler.type"):
            _parse_scheduler(block)

    def test_rejects_unknown_anneal_strategy(self) -> None:
        """`anneal_strategy` is constrained to what OneCycleLR accepts."""
        block = dict(ONECYCLE_BLOCK) | {"anneal_strategy": "cosine"}
        with pytest.raises(ValueError, match="anneal_strategy"):
            _parse_scheduler(block)

    def test_rejects_unknown_key(self) -> None:
        """A typo'd key is a hard error, matching the rest of the schema."""
        block = dict(ONECYCLE_BLOCK) | {"three_phase": True}
        with pytest.raises(ValueError, match="unknown config keys"):
            _parse_scheduler(block)

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("max_lr", 0.0),
            ("pct_start", 0.0),
            ("pct_start", 1.0),
            ("div_factor", 0.0),
            ("final_div_factor", -1.0),
        ],
    )
    def test_rejects_out_of_range_values(self, key: str, value: float) -> None:
        """Each numeric field is range-checked."""
        block = dict(ONECYCLE_BLOCK) | {key: value}
        with pytest.raises(ValueError, match=key):
            _parse_scheduler(block)

    def test_missing_required_field(self) -> None:
        """A partially specified block names the missing key."""
        block = dict(ONECYCLE_BLOCK)
        del block["max_lr"]
        with pytest.raises(ValueError, match="max_lr"):
            _parse_scheduler(block)


class TestConfigIntegration:
    """The block flows through `load_config` and `config_to_dict`."""

    def test_config_without_block_has_none(self, tmp_path: Path) -> None:
        """Existing configs keep `scheduler=None` (backward compatible)."""
        cfg = _config_with_scheduler(tmp_path, None)
        assert cfg.optim.scheduler is None

    def test_config_to_dict_serializes_scheduler(self, tmp_path: Path) -> None:
        """The checkpoint config payload records the schedule that trained it."""
        cfg = _config_with_scheduler(tmp_path, dict(ONECYCLE_BLOCK))
        as_dict = config_to_dict(cfg)
        assert as_dict["optim"]["scheduler"] == ONECYCLE_BLOCK
        json.dumps(as_dict)  # must stay JSON-serializable for the config hash

    def test_shipped_b0_config_uses_the_legacy_recipe(self) -> None:
        """The committed B0 config carries the recipe this change exists for."""
        cfg = load_config(Path("configs/b0_v31_breadth_first.yaml"))
        assert cfg.optim.epochs == 50
        assert cfg.optim.weight_decay == 0.05
        assert cfg.eval.patience == 10
        scheduler = cfg.optim.scheduler
        assert scheduler is not None
        assert scheduler.type == "onecycle"
        assert scheduler.max_lr == 1.0e-4
        assert scheduler.pct_start == 0.1
        assert scheduler.div_factor == 25.0
        assert scheduler.final_div_factor == 10000.0
        assert scheduler.anneal_strategy == "cos"
        assert cfg.model.config["label_smoothing"] == 0.05


# --------------------------------------------------------------------------- building


class TestBuildScheduler:
    """`_build_scheduler` dispatch and LR trajectory."""

    def test_no_block_reproduces_warmup_then_constant(self, tmp_path: Path) -> None:
        """The historical schedule is preserved exactly when no block is given."""
        cfg = _config_with_scheduler(tmp_path, None)
        param = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([param], lr=1.0e-4)
        scheduler = _build_scheduler(optimizer, cfg, warmup_steps=4, total_steps=None)

        assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
        observed = [scheduler.get_last_lr()[0]]
        for _ in range(6):
            optimizer.step()
            _step_scheduler(scheduler)
            observed.append(scheduler.get_last_lr()[0])

        # Linear ramp over 4 steps, then pinned at the configured LR.
        assert observed[0] == pytest.approx(0.25e-4)
        assert observed[3] == pytest.approx(1.0e-4)
        assert observed[-1] == pytest.approx(1.0e-4)

    def test_onecycle_requires_total_steps(self, tmp_path: Path) -> None:
        """A mis-sized OneCycle is a hard error, never a silent wrong schedule."""
        cfg = _config_with_scheduler(tmp_path, dict(ONECYCLE_BLOCK))
        optimizer = torch.optim.AdamW([nn.Parameter(torch.zeros(1))], lr=1.0e-4)
        with pytest.raises(ValueError, match="total_steps"):
            _build_scheduler(optimizer, cfg, warmup_steps=4, total_steps=None)

    @pytest.mark.parametrize("total_steps", [0, -5])
    def test_onecycle_rejects_nonpositive_total_steps(
        self, tmp_path: Path, total_steps: int
    ) -> None:
        """Zero or negative step counts cannot size a schedule."""
        cfg = _config_with_scheduler(tmp_path, dict(ONECYCLE_BLOCK))
        optimizer = torch.optim.AdamW([nn.Parameter(torch.zeros(1))], lr=1.0e-4)
        with pytest.raises(ValueError, match="total_steps"):
            _build_scheduler(optimizer, cfg, warmup_steps=4, total_steps=total_steps)

    def test_onecycle_lr_trajectory(self, tmp_path: Path) -> None:
        """LR starts at max_lr/div_factor, peaks at max_lr, anneals to the floor."""
        cfg = _config_with_scheduler(tmp_path, dict(ONECYCLE_BLOCK))
        param = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([param], lr=1.0e-4)
        total_steps = 100
        scheduler = _build_scheduler(
            optimizer, cfg, warmup_steps=999, total_steps=total_steps
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)

        observed = [scheduler.get_last_lr()[0]]
        for _ in range(total_steps - 1):
            optimizer.step()
            _step_scheduler(scheduler)
            observed.append(scheduler.get_last_lr()[0])

        initial_lr = 1.0e-4 / 25.0
        final_lr = initial_lr / 10000.0
        assert observed[0] == pytest.approx(initial_lr, rel=1e-6)
        assert max(observed) == pytest.approx(1.0e-4, rel=1e-6)
        # pct_start=0.1 ends the ramp at index int(0.1*100) - 1 == 9.
        assert observed.index(max(observed)) == 9
        assert observed[-1] == pytest.approx(final_lr, rel=1e-3)
        # Monotone ramp up to the peak, monotone anneal after it.
        assert observed[:10] == sorted(observed[:10])
        assert observed[9:] == sorted(observed[9:], reverse=True)
        # warmup_steps is ignored once OneCycle owns the schedule.
        assert observed[0] < observed[9]

    def test_step_scheduler_tolerates_overshoot(self, tmp_path: Path) -> None:
        """Stepping past total_steps holds the final LR instead of crashing."""
        cfg = _config_with_scheduler(tmp_path, dict(ONECYCLE_BLOCK))
        optimizer = torch.optim.AdamW([nn.Parameter(torch.zeros(1))], lr=1.0e-4)
        scheduler = _build_scheduler(optimizer, cfg, warmup_steps=1, total_steps=5)

        for _ in range(5):
            optimizer.step()
            _step_scheduler(scheduler)
        final_lr = scheduler.get_last_lr()[0]

        _step_scheduler(scheduler)  # would raise ValueError if unguarded
        assert scheduler.get_last_lr()[0] == pytest.approx(final_lr)

    def test_step_scheduler_reraises_for_other_schedulers(self, tmp_path: Path) -> None:
        """The overshoot guard is scoped to OneCycle only."""

        class _Exploding(torch.optim.lr_scheduler.LambdaLR):
            armed = False

            def step(self, *args: object, **kwargs: object) -> None:
                # LambdaLR.__init__ calls step() once; only fail afterwards.
                if self.armed:
                    raise ValueError("unrelated failure")
                super().step(*args, **kwargs)

        optimizer = torch.optim.AdamW([nn.Parameter(torch.zeros(1))], lr=1.0e-4)
        scheduler = _Exploding(optimizer, lr_lambda=lambda _: 1.0)
        scheduler.armed = True
        with pytest.raises(ValueError, match="unrelated failure"):
            _step_scheduler(scheduler)


class TestCountSingleProcessSteps:
    """`_count_single_process_steps` sums per-epoch batch counts."""

    def test_sums_varying_epoch_lengths(self, tmp_path: Path) -> None:
        """Per-epoch counts differ, so they are summed rather than extrapolated."""
        cfg = _config_with_scheduler(tmp_path, dict(ONECYCLE_BLOCK), epochs=3)
        lengths = {1: 7, 2: 5, 3: 6}

        def factory(epoch: int) -> list[int]:
            return list(range(lengths[epoch]))

        assert _count_single_process_steps(factory, cfg) == 18

    def test_rejects_unsized_loader(self, tmp_path: Path) -> None:
        """An unsized loader cannot size OneCycle, so it fails loudly."""
        cfg = _config_with_scheduler(tmp_path, dict(ONECYCLE_BLOCK), epochs=1)

        def factory(epoch: int) -> Iterator[int]:
            yield 1

        with pytest.raises(ValueError, match="no __len__"):
            _count_single_process_steps(factory, cfg)


# --------------------------------------------------------------------------- smoothing


class TestLabelSmoothing:
    """V3.1 loss-side label smoothing."""

    def test_defaults_to_disabled(self) -> None:
        """Omitting the key leaves the loss mathematically unchanged."""
        assert _tiny_v31().label_smoothing == 0.0

    @pytest.mark.parametrize("value", [-0.01, 1.0, 1.5])
    def test_rejects_out_of_range(self, value: float) -> None:
        """Smoothing outside [0, 1) is a construction-time error."""
        with pytest.raises(ValueError, match="label_smoothing"):
            _tiny_v31(label_smoothing=value)

    def test_smoothing_changes_loss_but_not_logits(self) -> None:
        """Smoothing is loss-only: identical weights give identical logits."""
        torch.manual_seed(0)
        plain = _tiny_v31()
        smoothed = _tiny_v31(label_smoothing=0.05)
        smoothed.load_state_dict(plain.state_dict())
        plain.eval()
        smoothed.eval()

        batch = {
            "emb_a": torch.randn(4, 3, 8),
            "emb_b": torch.randn(4, 3, 8),
            "label": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        }
        with torch.no_grad():
            out_plain = plain(batch)
            out_smoothed = smoothed(batch)

        torch.testing.assert_close(out_plain["logits"], out_smoothed["logits"])
        assert not torch.allclose(out_plain["loss"], out_smoothed["loss"])

    def test_smoothing_matches_explicit_soft_targets(self) -> None:
        """The applied targets are exactly y*(1-eps) + eps/2."""
        torch.manual_seed(0)
        eps = 0.05
        model = _tiny_v31(label_smoothing=eps)
        model.eval()

        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        batch = {
            "emb_a": torch.randn(4, 3, 8),
            "emb_b": torch.randn(4, 3, 8),
            "label": labels,
        }
        with torch.no_grad():
            output = model(batch)
            logits = output["logits"].squeeze(-1)
            expected = nn.functional.binary_cross_entropy_with_logits(
                logits, labels * (1.0 - eps) + 0.5 * eps
            )

        torch.testing.assert_close(output["loss"], expected)

    def test_smoothing_penalizes_overconfidence(self) -> None:
        """A confidently correct prediction costs more under smoothing."""
        eps = 0.05
        logits = torch.tensor([12.0, -12.0])
        labels = torch.tensor([1.0, 0.0])

        plain_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        smoothed_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, labels * (1.0 - eps) + 0.5 * eps
        )
        assert smoothed_loss > plain_loss

    def test_survives_a_config_round_trip(self, tmp_path: Path) -> None:
        """Smoothing declared in YAML reaches the constructed model."""
        config_path = tmp_path / "model.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "input_dim": 8,
                    "d_model": 8,
                    "encoder_layers": 1,
                    "cross_attn_layers": 1,
                    "n_heads": 2,
                    "label_smoothing": 0.05,
                    "mlp_head": {"hidden_dims": [8], "dropout": 0.0},
                    "regularization": {"dropout": 0.0},
                }
            )
        )
        model = V3_1(**yaml.safe_load(config_path.read_text()))
        assert model.label_smoothing == pytest.approx(0.05)

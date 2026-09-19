"""End-to-end checkpoint regressions for motif dictionary and head-only lanes."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from src import score_universe, train_b0
from src.data.motif_dictionary import DictionaryArtifact, save_dictionary
from src.model.egostitch.classifier.b0_v31 import V3_1
from src.model.egostitch.classifier.motif_dictionary import (
    ROUTE_TARGET_KEY,
    V3_1MotifDictionary,
)
from src.model.egostitch.classifier.motif_prompt import (
    TEMPLATE_MASK_KEY,
    V3_1MotifPrompt,
)
from src.train_b0 import ModelConfig

from tests.model.test_motif_prompt_model import _statistics
from tests.test_prefix_model import _pair_batch, _tiny_base_config
from tests.test_score_universe import _build_prefix_packed_fixture
from tests.test_train_b0 import _constant_metrics, _tiny_config


def _motif_block(
    base: Path,
    *,
    stage: str,
    bundle: Path | None = None,
    **overrides: object,
) -> dict[str, object]:
    block: dict[str, object] = {
        "stage": stage,
        "base_checkpoint": str(base),
        "width": 16,
        "slots_per_field": 2,
        "reader": {"layers": 1, "dim": 16, "heads": 4, "rrwp_k": 2},
        "slot_read": "residual_block",
        "slot_read_value_norm": False,
    }
    if bundle is not None:
        block["bundle_checkpoint"] = str(bundle)
    block.update(overrides)
    return block


def _save_sources(tmp_path: Path) -> tuple[Path, Path, Path, V3_1MotifPrompt, V3_1MotifPrompt]:
    """Write deliberately incompatible Stage-I and wave-3 interface states."""
    base_path = tmp_path / "base.pt"
    base = V3_1(**_tiny_base_config())
    torch.save(
        {
            "model_family": "v3_1",
            "model_config": _tiny_base_config(),
            "model_state": base.state_dict(),
        },
        base_path,
    )

    stage1_path = tmp_path / "stage1.pt"
    stage1_config = {
        "base": _tiny_base_config(),
        "motif_prompt": _motif_block(base_path, stage="one"),
    }
    stage1 = V3_1MotifPrompt(**stage1_config)
    stage1.install_mean_template(_statistics(torch.full((96,), 0.1)))
    with torch.no_grad():
        for parameter in stage1.reader.parameters():
            parameter.fill_(0.11)
        for parameter in stage1.count_head.parameters():
            parameter.fill_(0.12)
        for parameter in stage1.adapter.parameters():
            parameter.fill_(0.13)
        stage1.adapter.gates.fill_(0.75)
    torch.save(
        {
            "model_family": "v3_1_motif_prompt",
            "model_config": stage1_config,
            "model_state": stage1.state_dict(),
        },
        stage1_path,
    )

    wave3_path = tmp_path / "wave3.pt"
    wave3_config = {
        "base": _tiny_base_config(),
        "motif_prompt": _motif_block(base_path, stage="two", bundle=stage1_path),
    }
    wave3 = V3_1MotifPrompt(**wave3_config)
    wave3.install_mean_template(_statistics(torch.full((96,), 0.2)))
    wave3.initialize_teacher()
    assert wave3.generator is not None
    with torch.no_grad():
        for parameter in wave3.reader.parameters():
            parameter.fill_(0.71)
        for parameter in wave3.count_head.parameters():
            parameter.fill_(0.72)
        for parameter in wave3.adapter.parameters():
            parameter.fill_(0.73)
        wave3.adapter.gates.fill_(0.65)
        for name, parameter in wave3.generator.named_parameters():
            if name.startswith(
                (
                    "residue_proj.",
                    "attention.",
                    "bridge_queries",
                    "witness_queries",
                    "bridge_read.",
                    "witness_read.",
                    "endpoint_proj.",
                    "witness_mix.",
                )
            ):
                parameter.fill_(0.31)
    torch.save(
        {
            "model_family": "v3_1_motif_prompt",
            "model_config": wave3_config,
            "model_state": wave3.state_dict(),
        },
        wave3_path,
    )
    return base_path, stage1_path, wave3_path, stage1, wave3


def _save_dictionary(path: Path) -> DictionaryArtifact:
    generator = torch.Generator().manual_seed(29)
    tokens = torch.randn(2, 4, 16, generator=generator)
    tokens[:, 1].copy_(tokens[:, 0])
    artifact = DictionaryArtifact(
        weights=torch.zeros(2, 96),
        tokens=tokens,
        scales=torch.ones(4),
        temperature=0.7,
        mean_weights=torch.tensor([0.4, 0.6]),
        swap_index=torch.tensor([0, 1]),
        metadata={"scope": "training_only", "fixture": "pipeline"},
    )
    save_dictionary(artifact, path)
    return artifact


def _dictionary_config(
    base: Path, stage1: Path, wave3: Path, artifact: Path
) -> train_b0.Config:
    cfg = _tiny_config()
    model = ModelConfig(
        family=train_b0.MOTIF_DICTIONARY_FAMILY,
        config={
            "motif_prompt": _motif_block(base, stage="two", bundle=stage1),
            "motif_dictionary": {
                "mode": "sequence",
                "artifact_path": str(artifact),
                "dictionary_size": 2,
                "slot_checkpoint": str(wave3),
            },
        },
    )
    return replace(cfg, model=model, optim=replace(cfg.optim, lr=1e-3, weight_decay=0.01))


def _checkpoint(
    path: Path, model: torch.nn.Module, cfg: train_b0.Config
) -> dict[str, object]:
    kwargs = train_b0.resolve_model_kwargs(cfg.model)
    payload = train_b0._checkpoint_payload(  # noqa: SLF001
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        cfg,
        kwargs,
        1,
        _constant_metrics(),
        train_b0.config_to_dict(cfg),
    )
    torch.save(payload, path)
    return payload


def _changed_keys(
    before: dict[str, torch.Tensor], model: torch.nn.Module
) -> set[str]:
    return {
        key
        for key, value in model.state_dict().items()
        if not torch.equal(before[key], value.detach().cpu())
    }


def test_sequence_build_step_publish_and_truth_free_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_path, stage1_path, wave3_path, stage1, wave3 = _save_sources(tmp_path)
    artifact_path = tmp_path / "dictionary.pt"
    artifact = _save_dictionary(artifact_path)
    cfg = _dictionary_config(base_path, stage1_path, wave3_path, artifact_path)

    model = cast(V3_1MotifDictionary, train_b0.build_model(cfg))
    assert isinstance(model, V3_1MotifDictionary)
    torch.testing.assert_close(model.dictionary_tokens, artifact.tokens)
    # The evaluator comes from Stage I even though wave 3 intentionally carries
    # different reader/adapter values.
    torch.testing.assert_close(
        next(model.reader.parameters()), next(stage1.reader.parameters())
    )
    assert not torch.equal(next(model.reader.parameters()), next(wave3.reader.parameters()))
    torch.testing.assert_close(
        next(model.adapter.parameters()), next(stage1.adapter.parameters())
    )
    assert not torch.equal(next(model.adapter.parameters()), next(wave3.adapter.parameters()))
    assert model.generator is not None and wave3.generator is not None
    torch.testing.assert_close(
        model.generator.residue_proj.weight, wave3.generator.residue_proj.weight
    )

    model.train()
    optimizer = train_b0._build_optimizer(model, cfg)  # noqa: SLF001
    assert [group["name"] for group in optimizer.param_groups] == ["generator"]
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert optimized == trainable
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    batch = _pair_batch(n=4, seed=11)
    batch[ROUTE_TARGET_KEY] = torch.tensor(
        [[0.95, 0.05], [0.05, 0.95], [0.8, 0.2], [0.2, 0.8]], dtype=torch.float32
    )
    batch[TEMPLATE_MASK_KEY] = torch.ones(4, dtype=torch.bool)
    optimizer.zero_grad(set_to_none=True)
    route_rows = model(batch)["route_loss_rows"]
    assert torch.isfinite(route_rows).all()
    route_rows.mean().backward()
    optimizer.step()
    changed = _changed_keys(before, model)
    assert any(key.startswith("router.") for key in changed)
    assert any(key.startswith("generator.") for key in changed)
    assert changed <= {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    model.eval()
    pack, pairs = _build_prefix_packed_fixture(tmp_path, monkeypatch, node_count=5)
    expected = score_universe._score_v3_1_packed(
        model,
        pairs,
        pack,
        device=torch.device("cpu"),
        amp="off",
        token_budget=128,
    )
    checkpoint = tmp_path / "router-best.pt"
    _checkpoint(checkpoint, model, cfg)
    for source in (artifact_path, wave3_path, stage1_path, base_path):
        source.unlink()

    restored, family, _ = score_universe._load_checkpoint(checkpoint)
    assert family == train_b0.MOTIF_DICTIONARY_FAMILY
    actual = score_universe._score_v3_1_packed(
        restored,
        pairs,
        pack,
        device=torch.device("cpu"),
        amp="off",
        token_budget=57,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("prompt_enabled", [True, False], ids=["prompted", "content"])
def test_head_only_build_step_publish_and_score_without_wave3_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_enabled: bool,
) -> None:
    base_path, stage1_path, wave3_path, _, wave3 = _save_sources(tmp_path)
    cfg = _tiny_config()
    model_cfg = ModelConfig(
        family=train_b0.MOTIF_PROMPT_FAMILY,
        config={
            "motif_prompt": _motif_block(
                base_path,
                stage="two",
                bundle=stage1_path,
                training_policy="head_only",
                init_checkpoint=str(wave3_path),
                head_prompt_enabled=prompt_enabled,
            )
        },
    )
    cfg = replace(
        cfg,
        model=model_cfg,
        optim=replace(cfg.optim, lr=1e-3, weight_decay=0.01),
    )
    model = cast(V3_1MotifPrompt, train_b0.build_model(cfg))
    assert model.cfg.training_policy == "head_only"
    assert model.cfg.head_prompt_enabled is prompt_enabled
    for key, value in wave3.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value)

    model.train()
    assert not model.reader.training
    assert not model.adapter.training
    assert model.base.output_head.training
    optimizer = train_b0._build_optimizer(model, cfg)  # noqa: SLF001
    assert [group["name"] for group in optimizer.param_groups] == ["generator"]
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    head = {id(parameter) for parameter in model.base.output_head.parameters()}
    assert optimized == head
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    optimizer.zero_grad(set_to_none=True)
    loss = model(_pair_batch(n=4, seed=17))["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    changed = _changed_keys(before, model)
    assert changed
    assert changed <= {
        name for name, _ in model.named_parameters() if name.startswith("base.output_head.")
    }

    model.eval()
    pack, pairs = _build_prefix_packed_fixture(tmp_path, monkeypatch, node_count=5)
    expected = score_universe._score_v3_1_packed(
        model,
        pairs,
        pack,
        device=torch.device("cpu"),
        amp="off",
        token_budget=128,
    )
    checkpoint = tmp_path / f"head-{prompt_enabled}.pt"
    _checkpoint(checkpoint, model, cfg)
    for source in (wave3_path, stage1_path, base_path):
        source.unlink()

    restored, family, _ = score_universe._load_checkpoint(checkpoint)
    assert family == train_b0.MOTIF_PROMPT_FAMILY
    actual = score_universe._score_v3_1_packed(
        restored,
        pairs,
        pack,
        device=torch.device("cpu"),
        amp="off",
        token_budget=61,
    )
    np.testing.assert_array_equal(actual, expected)
    if not prompt_enabled:
        cast(V3_1MotifPrompt, restored).intervention = "gates_off"
        gates_off = score_universe._score_v3_1_packed(
            restored,
            pairs,
            pack,
            device=torch.device("cpu"),
            amp="off",
            token_budget=83,
        )
        np.testing.assert_array_equal(gates_off, actual)

"""Training reductions and optimiser policies for motif dictionary lanes."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import networkx as nx
import pytest
import torch
from accelerate import Accelerator
from src import train_b0
from src.model.egostitch.classifier.motif_dictionary import (
    ROUTE_NONEMPTY_KEY,
    ROUTE_TARGET_KEY,
    V3_1MotifDictionary,
)
from src.model.egostitch.classifier.motif_prompt import TEMPLATE_MASK_KEY, V3_1MotifPrompt
from src.train_b0 import DictionaryRouteRows, closure_balanced_route_kl, load_config

from tests.model.test_motif_dictionary import _dictionary, _motif_block
from tests.test_motif_dictionary_pipeline import (
    _dictionary_config,
    _save_dictionary,
    _save_sources,
)
from tests.test_prefix_model import _pair_batch, _tiny_base_config


def test_route_kl_is_invariant_to_unequal_rank_and_microbatch_partitions() -> None:
    rows = torch.tensor([1.0, 2.0, 4.0, 3.0, 5.0])
    nonempty = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])
    valid = torch.ones(5)
    rank_zero = closure_balanced_route_kl(
        rows[:3],
        nonempty[:3],
        valid[:3],
        global_valid=5,
        global_nonempty=2,
        world_size=2,
    )
    rank_one = closure_balanced_route_kl(
        rows[3:],
        nonempty[3:],
        valid[3:],
        global_valid=5,
        global_nonempty=2,
        world_size=2,
    )
    expected = 0.5 * torch.tensor([1.0, 3.0]).mean() + 0.5 * torch.tensor(
        [2.0, 4.0, 5.0]
    ).mean()
    torch.testing.assert_close((rank_zero + rank_one) / 2, expected)

    microbatch_sum = sum(
        closure_balanced_route_kl(
            rows[index : index + 1],
            nonempty[index : index + 1],
            valid[index : index + 1],
            global_valid=5,
            global_nonempty=2,
            world_size=1,
        )
        for index in range(5)
    )
    torch.testing.assert_close(microbatch_sum, expected)


def test_all_self_route_batch_is_autograd_connected_zero() -> None:
    rows = torch.tensor([2.0, 3.0], requires_grad=True)
    loss = closure_balanced_route_kl(
        rows,
        torch.ones(2),
        torch.zeros(2),
        global_valid=0,
        global_nonempty=0,
        world_size=4,
    )
    assert loss.requires_grad and loss.item() == 0.0
    loss.backward()  # type: ignore[no-untyped-call]
    torch.testing.assert_close(rows.grad, torch.zeros_like(rows))


def test_router_optimizer_step_moves_only_router_and_slot_encoder() -> None:
    model = _dictionary().train()
    batch = _pair_batch(n=4)
    batch[ROUTE_TARGET_KEY] = torch.softmax(torch.randn(4, model.dictionary_size), dim=-1)
    batch[ROUTE_NONEMPTY_KEY] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    batch[TEMPLATE_MASK_KEY] = torch.ones(4)
    before_base = {name: value.clone() for name, value in model.base.state_dict().items()}
    router_input = model.router.mlp[0]
    assert isinstance(router_input, torch.nn.Linear)
    before_router = router_input.weight.detach().clone()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
        weight_decay=1e-2,
    )
    output = model(batch)
    loss = closure_balanced_route_kl(
        output["route_loss_rows"],
        batch[ROUTE_NONEMPTY_KEY],
        batch[TEMPLATE_MASK_KEY],
        global_valid=4,
        global_nonempty=2,
        world_size=1,
    )
    loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    assert not torch.equal(before_router, router_input.weight)
    for name, value in model.base.state_dict().items():
        torch.testing.assert_close(value, before_base[name], rtol=0, atol=0)


@pytest.mark.parametrize("prompt_enabled", [True, False])
def test_head_optimizer_step_moves_only_output_head(prompt_enabled: bool) -> None:
    block = _motif_block(
        training_policy="head_only",
        head_prompt_enabled=prompt_enabled,
        init_checkpoint="source.pt",
    )
    model = V3_1MotifPrompt(base=_tiny_base_config(), motif_prompt=block).train()
    batch = _pair_batch(n=4)
    batch["label"] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable and {id(parameter) for parameter in trainable} == {
        id(parameter) for parameter in model.base.output_head.parameters()
    }
    frozen_before = model.generator.residue_proj.weight.detach().clone()  # type: ignore[union-attr]
    head_before = next(model.base.output_head.parameters()).detach().clone()
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-2)
    output = model(batch)
    output["loss"].backward()
    optimizer.step()
    assert not torch.equal(head_before, next(model.base.output_head.parameters()))
    torch.testing.assert_close(
        frozen_before,
        model.generator.residue_proj.weight,  # type: ignore[union-attr]
        rtol=0,
        atol=0,
    )


def test_four_production_configs_pin_full_schedule_and_lane_objectives() -> None:
    root = Path("configs/split_seed42")
    router = load_config(root / "motif_dict_router.yaml")
    oracle = load_config(root / "motif_dict_oracle.yaml")
    head = load_config(root / "motif_wave3_head.yaml")
    content = load_config(root / "motif_wave3_head_content.yaml")
    assert all(
        cfg.optim.epochs == 15 and cfg.eval.patience is None
        for cfg in (router, head, content)
    )
    assert router.model.family == "v3_1_motif_dictionary" and router.struct is None
    assert oracle.model.config["motif_dictionary"]["mode"] == "oracle"  # type: ignore[index]
    assert head.struct is not None and content.struct is not None
    assert head.optim.lr == content.optim.lr == 1e-5
    assert head.optim.scheduler is not None and head.optim.scheduler.max_lr == 1e-5
    assert content.optim.scheduler is not None and content.optim.scheduler.max_lr == 1e-5
    assert head.model.config["motif_prompt"]["head_prompt_enabled"] is True  # type: ignore[index]
    assert content.model.config["motif_prompt"]["head_prompt_enabled"] is False  # type: ignore[index]


def test_router_build_accepts_published_stage_one_without_legacy_slot_buffer(
    tmp_path: Path,
) -> None:
    base_path, stage1_path, wave3_path, _, _ = _save_sources(tmp_path)
    stage1 = torch.load(stage1_path, map_location="cpu", weights_only=False)
    stage1["model_state"].pop("w_slot_resolved")
    torch.save(stage1, stage1_path)
    artifact_path = tmp_path / "dictionary.pt"
    _save_dictionary(artifact_path)

    model = cast(
        V3_1MotifDictionary,
        train_b0.build_model(
            _dictionary_config(base_path, stage1_path, wave3_path, artifact_path)
        ),
    )

    assert model.w_slot_resolved.item() == -1.0


def test_route_target_cache_reuses_exact_rows_and_rejects_changed_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _dictionary().eval()
    graph = nx.Graph([("a", "b"), ("b", "c")])
    train_pairs = [("a", "b"), ("b", "c")]
    val_pairs = [("a", "c")]
    cache_path = tmp_path / "routes.pt"
    accelerator = Accelerator(cpu=True)
    original = model.routes_from_weights
    calls = 0

    def counted(weights: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(weights)

    monkeypatch.setattr(model, "routes_from_weights", counted)
    kwargs = {
        "model": model,
        "train_graph": graph,
        "train_pairs": train_pairs,
        "val_graph": graph,
        "val_pairs": val_pairs,
        "device": torch.device("cpu"),
        "accelerator": accelerator,
        "cache_path": cache_path,
        "encode_batch_size": 1,
    }
    DictionaryRouteRows(**kwargs)
    built_calls = calls
    DictionaryRouteRows(**kwargs)
    assert calls == built_calls

    def cache_miss(_weights: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("cache miss")

    monkeypatch.setattr(model, "routes_from_weights", cache_miss)
    model.dictionary_tokens.add_(0.01)
    with pytest.raises(RuntimeError, match="cache miss"):
        DictionaryRouteRows(**kwargs)
    model.dictionary_tokens.sub_(0.01)
    with pytest.raises(RuntimeError, match="cache miss"):
        DictionaryRouteRows(**{**kwargs, "train_pairs": list(reversed(train_pairs))})

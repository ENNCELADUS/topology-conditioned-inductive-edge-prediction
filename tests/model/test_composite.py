"""Tests for `EgoStitchModel`, the three-component composite (design §3-§7).

`e2e_model.py` is gone (three-component refactor design §5), so an
end-to-end numerical comparison against the pre-refactor monolith is no
longer possible -- that equivalence is instead proven piecewise, by
`tests/model/test_generator_component.py`, `test_encoder_component.py` and
`test_classifier_component.py` (each ported component vs. its
`EgoStitchE2E`-era origin) and by `tests/model/test_egostitch_e2e_model.py`
(retargeted onto `EgoStitchModel`, preserving every surviving assertion).

This file instead pins the properties specific to *composition*: the
generator's per-node state is genuinely cacheable through the composite (not
just within one component's own tests), a null generator's `cond=None`
reaches the classifier as the true unconditioned baseline, a scaffold-control
perturbation actually reaches the generator through the composite's public
`build_pair_context_from_states`, and `forward`'s `"graph"`/`"embedding_ab"`
output composes -- via `generator.auxiliary_losses`/`encoder.auxiliary_losses`
-- into the four pinned families with the ten reconstruction components
intact, without a second stitch+encode pass (design §6; there is
deliberately no `EgoStitchModel.aggregate_losses`, see composite.py).
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from src.model.egostitch.classifier.b0_v31 import B0V31PairClassifier, GatedCrossAttention
from src.model.egostitch.composite import EgoStitchModel
from src.model.egostitch.config import (
    CONDITIONING_MODES,
    ClassifierConfig,
    E2EConfig,
    EncoderConfig,
    GeneratorConfig,
)
from src.model.egostitch.generator import NullGenerator, StitchedGraph
from src.model.egostitch.generator.assemble import make_scaffold_input_perturbation
from src.model.egostitch.generator.losses import stage1_family_tensors, stage1_total
from src.model.egostitch.graph import GraphEmbedding, PairInputs


def _tiny_e2e_config(
    *,
    n_ground: int = 5,
    generator_name: str = "egostitch_imagine",
    feature_standardization: str = "row_layernorm",
    encoder_w_rel: float = 0.25,
    conditioning_mode: str = "xattn_cls",
) -> E2EConfig:
    """The shared tiny composite sizing used across this file's tests (design §8)."""
    return E2EConfig(
        generator=GeneratorConfig(
            name=generator_name,
            n_ground=n_ground,
            feature_standardization=feature_standardization,
        ),
        encoder=EncoderConfig(dim=16, layers=2, w_rel=encoder_w_rel),
        classifier=ClassifierConfig(
            d_model=32,
            encoder_layers=1,
            cross_attn_layers=2,
            n_heads=4,
            n_inj=1,
            xattn_heads=4,
            p_topo=0.15,
            conditioning_mode=conditioning_mode,
        ),
    )


def _tiny_model_and_batch(
    **config_overrides: object,
) -> tuple[EgoStitchModel, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    cfg = _tiny_e2e_config(**cast(dict[str, Any], config_overrides))
    model = EgoStitchModel(cfg).eval()
    b, t, d_in = 4, 6, model.input_dim
    n_ground = cfg.generator.n_ground
    batch = {
        "emb_a": torch.randn(b, t, d_in),
        "emb_b": torch.randn(b, t, d_in),
        "len_a": torch.full((b,), t, dtype=torch.long),
        "len_b": torch.full((b,), t, dtype=torch.long),
        "x_a": torch.randn(b, model.node_feature_dim),
        "x_b": torch.randn(b, model.node_feature_dim),
        "ground_a": torch.randn(b, n_ground, model.node_feature_dim),
        "ground_b": torch.randn(b, n_ground, model.node_feature_dim),
        "ground_id_a": torch.randint(0, 1000, (b, n_ground), dtype=torch.long),
        "ground_id_b": torch.randint(0, 1000, (b, n_ground), dtype=torch.long),
    }
    return model, batch


def _full_training_batch(model: EgoStitchModel, cfg: E2EConfig) -> dict[str, torch.Tensor]:
    """A merged node-stream + edge-stream + pair-stream batch for `aggregate_losses`.

    Shaped like `tests/model/test_generator_component.py`'s
    `_node_stream_batch` / `_edge_align_batch` fixtures, sized against this
    model's own `generator_cfg` rather than a standalone tiny config, plus
    the pair-stream keys `_pair_node_states` needs and the relational-head
    keys `GraphEncoder.auxiliary_losses` needs.
    """
    gcfg = model.generator_cfg
    b, t, n_ground, d_in = 4, gcfg.slots, gcfg.n_ground, gcfg.input_dim
    gen = torch.Generator().manual_seed(3)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen)

    target_in_pool = torch.rand(b, t, generator=gen) > 0.5
    node_batch = {
        "x": rnd(b, d_in),
        "ground_x": rnd(b, n_ground, d_in),
        "target_features": rnd(b, t, d_in),
        "target_mult": 1.0 + torch.rand(b, t, generator=gen),
        "target_adj": (torch.rand(b, t, t, generator=gen) > 0.5).float(),
        "target_mask": torch.ones(b, t, dtype=torch.bool),
        "target_in_pool": target_in_pool,
        "target_pool_index": torch.where(
            target_in_pool,
            torch.arange(t).remainder(n_ground).expand(b, -1),
            torch.full((b, t), -1),
        ),
        "true_degree": torch.randint(1, 6, (b,), generator=gen).float(),
        "real_ego_stats": torch.rand(b, 4, generator=gen),
        "ground_resampled": rnd(b, n_ground, d_in),
        "ssl_noise": 0.05 * rnd(b, d_in),
    }
    edge_batch = {
        "target_features_a": rnd(b, t, d_in),
        "target_features_b": rnd(b, t, d_in),
        "target_mult_a": 1.0 + torch.rand(b, t, generator=gen),
        "target_mult_b": 1.0 + torch.rand(b, t, generator=gen),
        "target_adj_a": (torch.rand(b, t, t, generator=gen) > 0.5).float(),
        "target_adj_b": (torch.rand(b, t, t, generator=gen) > 0.5).float(),
        "target_mask_a": torch.ones(b, t, dtype=torch.bool),
        "target_mask_b": torch.ones(b, t, dtype=torch.bool),
        "target_node_index_a": torch.randint(0, 100, (b, t), generator=gen),
        "target_node_index_b": torch.randint(0, 100, (b, t), generator=gen),
        "label": torch.randint(0, 2, (b,), generator=gen),
        "edge_mask": torch.ones(b),
        "rel_target": rnd(b, 2),
        "loss_world_size": torch.tensor(1),
    }
    pair_batch = {
        "emb_a": rnd(b, 6, d_in),
        "emb_b": rnd(b, 6, d_in),
        "len_a": torch.full((b,), 6, dtype=torch.long),
        "len_b": torch.full((b,), 6, dtype=torch.long),
        "x_a": rnd(b, d_in),
        "x_b": rnd(b, d_in),
        "ground_a": rnd(b, n_ground, d_in),
        "ground_b": rnd(b, n_ground, d_in),
    }
    return {**node_batch, **edge_batch, **pair_batch}


# --------------------------------------------------------------------------- (a) self-row caching


def test_self_pair_batch_encodes_each_unique_endpoint_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A self-pair batch calls `generator.encode_node` once per row, not twice.

    Mirrors `e2e_model.py:223-253`'s self-row optimization (module docstring
    of `composite.py`, CRITICAL point 1), pinned here against the
    component's own public `encode_node` rather than the internal
    `stage1.encode_nodes` (`test_egostitch_e2e_model.py` already pins the
    latter). `probe_states` runs exactly this self-pair shape, so losing the
    optimization silently doubles generator cost on every probe batch.
    """
    model, batch = _tiny_model_and_batch()
    batch["emb_b"] = batch["emb_a"]
    batch["len_b"] = batch["len_a"]
    batch["x_b"] = batch["x_a"]
    batch["ground_b"] = batch["ground_a"]
    batch["ground_id_b"] = batch["ground_id_a"]
    batch["is_self"] = torch.ones(batch["x_a"].size(0), dtype=torch.bool)

    calls = 0
    original = model.generator.encode_node

    # `Any`, not `object`, on purpose: this wrapper forwards to whatever
    # `original`'s real (narrower) signature is -- it never inspects its own
    # arguments -- so typing them `object` would make the forwarding call
    # itself untypeable without reflecting any real safety property here.
    def counted(*args: Any, **kwargs: Any) -> object:  # noqa: ANN401 -- generic monkeypatch forwarding wrapper
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model.generator, "encode_node", counted)

    model.build_pair_context(batch)

    assert calls == 1


def test_mixed_self_and_non_self_batch_encodes_b_only_for_non_self_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mixed batch calls `generator.encode_node` twice: once for A, once for non-self B."""
    model, batch = _tiny_model_and_batch()
    is_self = torch.tensor([True, False, True, False])
    batch["is_self"] = is_self
    # The data contract guarantees x_a == x_b (and hence ground_a == ground_b)
    # on self rows; the merge is only sound under that equality.
    batch["x_b"] = torch.where(is_self[:, None], batch["x_a"], batch["x_b"])
    batch["ground_b"] = torch.where(is_self[:, None, None], batch["ground_a"], batch["ground_b"])

    call_batch_sizes: list[int] = []
    original = model.generator.encode_node

    # `Any` for the same reason as the wrapper above: forwards blindly to
    # `original`'s real signature.
    def counted(x: torch.Tensor, *args: Any, **kwargs: Any) -> object:  # noqa: ANN401 -- generic monkeypatch forwarding wrapper
        call_batch_sizes.append(x.size(0))
        return original(x, *args, **kwargs)

    monkeypatch.setattr(model.generator, "encode_node", counted)

    model.build_pair_context(batch)

    # One call for endpoint A (every row), one for endpoint B (non-self rows only).
    assert call_batch_sizes == [4, 2]


def test_shared_endpoint_across_several_pairs_calls_encode_tokens_once_per_unique_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A node scored against several counterparts calls `classifier.encode_tokens` once.

    The direct analogue of the two `generator.encode_node` call-count tests
    above, but for the classifier's own cacheable per-node phase (2026-08-03
    fix): `score_universe.py`'s `node_cache` encodes each unique node exactly
    once (`score_universe.py:1713,1852`) and reuses that state across every
    pair it appears in, so a shared endpoint scored against several distinct
    counterparts must call `classifier.encode_tokens` once for the shared
    endpoint plus once per counterpart -- never once per pair. Without this
    test, `PairClassifier.forward` silently re-deriving its own token
    encoding from `PairInputs` on every call would regress this again with
    no other test noticing (correctness is unaffected; only wall-clock is).
    """
    model, batch = _tiny_model_and_batch()
    num_counterparts = batch["x_a"].size(0)

    calls = 0
    original = model.classifier.encode_tokens

    # `Any` for the same reason as the generator wrappers above.
    def counted(*args: Any, **kwargs: Any) -> object:  # noqa: ANN401 -- generic monkeypatch forwarding wrapper
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model.classifier, "encode_tokens", counted)

    # Encode one shared endpoint once, then several distinct counterparts once each.
    shared = model.encode_node_state(
        batch["emb_a"][:1], batch["len_a"][:1], batch["x_a"][:1], batch["ground_a"][:1]
    )
    assert calls == 1

    is_self = torch.zeros(1, dtype=torch.bool)
    for row in range(num_counterparts):
        counterpart = model.encode_node_state(
            batch["emb_b"][row : row + 1],
            batch["len_b"][row : row + 1],
            batch["x_b"][row : row + 1],
            batch["ground_b"][row : row + 1],
        )
        # Reusing `shared` across every pair must not re-invoke `encode_tokens`.
        context = model.build_pair_context_from_states(shared, counterpart, is_self)
        model.score_pair_context(context)

    assert calls == 1 + num_counterparts


# --------------------------------------------------------------------------- (b) null generator


def test_null_generator_cond_none_matches_conditioned_model_f_logit() -> None:
    """A null generator's `cond=None` reaches the classifier as the true B0 baseline.

    `NullGenerator.stitch` always returns `None` (design §3.3); the composite
    forward path is `cond = None if graph is None else ...` (design §3.4), so
    feeding the classifier's own `encode_tokens` output for the same raw
    batch the conditioned model built its pair context from into
    `model.classifier(pair, None)` must reproduce exactly
    `model.decompose(batch)["f_logit"]` -- the existing eval-time hard bypass
    -- because both paths run the classifier fully unconditioned. This test
    exercises the null path at the component-composition level, independent
    of config wiring;
    `test_registry_driven_null_generator_config_matches_conditioned_model_f_logit`
    below proves the same identity end-to-end through
    `generator.name: "null"` (design §12 P3 acceptance criterion 5).
    """
    model, batch = _tiny_model_and_batch()
    # `model.classifier`'s static type is the `PairClassifier` ABC now that
    # construction is registry-driven (design §12 P3); narrow to the
    # concrete class this file always builds to reach `.trunk`, same idiom
    # as `tests/model/test_egostitch_e2e_model.py`'s `_classifier` helper.
    assert isinstance(model.classifier, B0V31PairClassifier)
    # `nn.ModuleList.__getitem__` is typed to return plain `Module`; narrow to
    # the concrete type to reach `.gate` (same idiom as
    # `tests/test_train_egostitch_e2e.py`'s `_gates` helper).
    topo_gate = model.classifier.trunk.topo_xattn[0]
    assert isinstance(topo_gate, GatedCrossAttention)
    with torch.no_grad():
        topo_gate.gate.fill_(0.7)  # open gate: prove it's truly bypassed

    f_logit = model.decompose(batch)["f_logit"]

    null_generator = NullGenerator()
    x_a, x_b = batch["x_a"], batch["x_b"]
    ground_a, ground_b = batch["ground_a"], batch["ground_b"]
    is_self = torch.zeros(x_a.size(0), dtype=torch.bool)
    graph = null_generator(x_a, x_b, ground_a, ground_b, is_self=is_self)
    assert graph is None

    with torch.no_grad():
        pair = PairInputs(
            tokens_a=model.classifier.encode_tokens(batch["emb_a"], batch["len_a"]),
            tokens_b=model.classifier.encode_tokens(batch["emb_b"], batch["len_b"]),
            len_a=batch["len_a"],
            len_b=batch["len_b"],
        )
    cond = None if graph is None else object()  # composite's literal forward rule (design §3.4)
    with torch.no_grad():
        null_logits = model.classifier(pair, cond)

    torch.testing.assert_close(null_logits, f_logit, rtol=0.0, atol=1e-6)


def test_registry_driven_null_generator_config_matches_conditioned_model_f_logit() -> None:
    """`generator.name: "null"` end-to-end reproduces the conditioned model's `f_logit`.

    Proves the registry-driven wiring itself (design §12 P3 acceptance
    criterion 5), not just the component composed by hand: building a whole
    `EgoStitchModel` from `E2EConfig(generator=GeneratorConfig(name="null"),
    ...)` and running it through the *public* `forward` path -- never
    reaching into `model.classifier` directly, unlike the test above --
    reproduces the conditioned model's `f_logit` bit-for-bit.

    A null-generator model's `generator`/`encoder` allocate zero parameters
    (`__init__` never even constructs a `GraphEncoder`), so seeding both
    models with the same `torch.manual_seed(0)` leaves their RNG streams at
    different offsets by the time each reaches its own classifier
    construction -- the two classifiers therefore start with *different*
    weights unless reconciled. `load_state_dict` reconciles them explicitly,
    which is the honest way to isolate "does the null wiring compute the
    right thing" from "do two independently-seeded models happen to match".
    """
    conditioned, batch = _tiny_model_and_batch()
    f_logit = conditioned.decompose(batch)["f_logit"]

    null_model = EgoStitchModel(_tiny_e2e_config(generator_name="null")).eval()
    null_model.classifier.load_state_dict(conditioned.classifier.state_dict())

    with torch.no_grad():
        output = null_model(batch)

    assert "graph" not in output
    assert "embedding_ab" not in output
    assert torch.equal(cast(torch.Tensor, output["logits"]), f_logit)


def test_null_generator_state_dict_carries_no_generator_or_encoder_parameters() -> None:
    """`generator.name: "null"` builds no `GraphEncoder` and a parameter-free generator.

    Design §12 P3 task 4's own acceptance bar: `self.encoder` must be either
    unconstructed or contribute no parameters, and `NullGenerator` itself
    (design §3.3, `generator/null.py`) has no layers to allocate. Together
    these mean every surviving `state_dict` key belongs to the classifier --
    proven directly, not merely inferred from a raw parameter count, so a
    regression that silently gave the null generator or a future stub
    encoder even one buffer would fail this immediately.
    """
    model = EgoStitchModel(_tiny_e2e_config(generator_name="null"))

    assert model.encoder is None
    assert isinstance(model.generator, NullGenerator)
    assert sum(p.numel() for p in model.generator.parameters()) == 0
    assert list(model.generator.buffers()) == []

    state_dict_keys = set(model.state_dict())
    assert state_dict_keys, "expected the classifier to still contribute state"
    assert all(key.startswith("classifier.") for key in state_dict_keys)

    total_params = sum(p.numel() for p in model.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    assert total_params == classifier_params > 0


def _conditioning_submodules(model: EgoStitchModel) -> list[torch.nn.Module]:
    """Every classifier submodule reachable only through `cond is not None`."""
    assert isinstance(model.classifier, B0V31PairClassifier)
    modules: list[torch.nn.Module] = list(model.classifier.trunk.topo_xattn)
    pooled_adapter = getattr(model.classifier.trunk, "pooled_adapter", None)
    if pooled_adapter is not None:
        modules.extend(pooled_adapter)
    film = getattr(model.classifier, "film", None)
    if film is not None:
        modules.append(film)
    return modules


@pytest.mark.parametrize("conditioning_mode", sorted(CONDITIONING_MODES))
def test_null_generator_freezes_every_conditioning_submodule(conditioning_mode: str) -> None:
    """A null-generator model's conditioning rung is frozen, not merely unused.

    `self.encoder` is never constructed for `generator.name: "null"`
    (`__init__`), so `score_pair_context`/`forward` always clamp `need_topo`
    off and `cond` stays `None` for every call
    (`_build_pair_context_and_graph`) -- none of `trunk.topo_xattn`,
    `trunk.pooled_adapter`, or `classifier.film` ever runs a forward pass
    under this composition (`ConditionedPairCrossAttention.forward`'s
    injection sites and `B0V31PairClassifier.forward`'s film branch are both
    gated on `cond is not None`). `EgoStitchModel.__init__` must call
    `classifier.freeze_unreachable_conditioning()` to keep those parameters
    out of DDP's gradient reduction (CLAUDE.md P1: DDP's
    `find_unused_parameters=False` rejects a `requires_grad=True` parameter
    that never receives a gradient).
    """
    model, _ = _tiny_model_and_batch(generator_name="null", conditioning_mode=conditioning_mode)
    assert model.encoder is None

    submodules = _conditioning_submodules(model)
    assert submodules, f"conditioning_mode={conditioning_mode!r} built no conditioning submodule"
    for module in submodules:
        assert all(not p.requires_grad for p in module.parameters()), (
            f"conditioning_mode={conditioning_mode!r}: expected every parameter of "
            f"{type(module).__name__} frozen under a null generator"
        )

    # Frozen, not omitted (CLAUDE.md/design note at `ConditionedPairCrossAttention.__init__`):
    # `state_dict` key layout must stay exactly what a live-encoder arm would build, so a
    # scoring-only null-generator checkpoint keeps loading.
    live_model, _ = _tiny_model_and_batch(conditioning_mode=conditioning_mode)
    assert set(model.classifier.state_dict()) == set(live_model.classifier.state_dict())


@pytest.mark.parametrize("conditioning_mode", sorted(CONDITIONING_MODES))
def test_null_generator_every_trainable_parameter_receives_a_gradient(
    conditioning_mode: str,
) -> None:
    """The exact property DDP `find_unused_parameters=False` demands.

    Regression guard for the null-generator component contract: a
    `requires_grad=True` parameter that never participates in its forward
    graph gets `grad is None` after `.backward()`, which is what DDP's
    unused-parameter check rejects on the first gradient reduction. Every
    surviving trainable parameter of a null-generator model must actually be
    used by a real forward+backward pass -- not merely "the previously-buggy
    modules are frozen" (the test above), which would pass even if some
    other, unrelated parameter were wrongly unreachable.
    """
    model, batch = _tiny_model_and_batch(generator_name="null", conditioning_mode=conditioning_mode)
    model.train()

    output = model(batch)
    cast(torch.Tensor, output["logits"]).sum().backward()  # type: ignore[no-untyped-call]

    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not missing, f"parameters with no gradient (DDP would reject these): {missing}"


@pytest.mark.parametrize("generator_name", ["egostitch_imagine", "oracle_struct"])
@pytest.mark.parametrize("conditioning_mode", sorted(CONDITIONING_MODES))
def test_live_encoder_arm_conditioning_parameters_stay_trainable(
    generator_name: str, conditioning_mode: str
) -> None:
    """A generator with a real encoder must be bit-for-bit unaffected by the null-arm freeze.

    `freeze_unreachable_conditioning` fires only when `model.encoder is None`
    (composite.py); `egostitch_imagine` and `oracle_struct` both build a real
    `GraphEncoder` and can genuinely produce a non-`None` `cond`, so their
    conditioning submodules must remain exactly as trainable as before this
    fix -- these two arms' results must not shift.
    """
    model, _ = _tiny_model_and_batch(
        generator_name=generator_name, conditioning_mode=conditioning_mode
    )
    assert model.encoder is not None

    submodules = _conditioning_submodules(model)
    assert submodules, f"conditioning_mode={conditioning_mode!r} built no conditioning submodule"
    for module in submodules:
        assert all(p.requires_grad for p in module.parameters()), (
            f"generator_name={generator_name!r} conditioning_mode={conditioning_mode!r}: "
            f"expected every parameter of {type(module).__name__} to stay trainable"
        )


# --------------------------------------------------------------------------- (c) perturbation


def test_perturbation_reaches_the_generator_through_build_pair_context_from_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`scaffold_input_perturbation` threads from the composite into `generator.stitch`.

    Proves both the wiring (the exact object reaches `generator.stitch`) and
    the effect (a real perturbation changes the assembled topology tokens) --
    the mandatory 6a-shuffle / 6e-rewire structure-control arms
    (`score_universe.py:1935,1942`) depend on both.
    """
    model, batch = _tiny_model_and_batch()
    state_a, state_b, is_self = model._pair_node_states(batch)

    captured: dict[str, object] = {}
    original_stitch = model.generator.stitch

    # `Any` for the same reason as the wrappers in
    # `test_self_pair_batch_encodes_each_unique_endpoint_once` above: forwards
    # blindly to `original_stitch`'s real signature.
    def capturing_stitch(*args: Any, **kwargs: Any) -> object:  # noqa: ANN401 -- generic monkeypatch forwarding wrapper
        captured["perturbation"] = kwargs.get("perturbation")
        return original_stitch(*args, **kwargs)

    monkeypatch.setattr(model.generator, "stitch", capturing_stitch)

    baseline = model.build_pair_context_from_states(state_a, state_b, is_self)
    assert captured["perturbation"] is None

    pairs = [(f"u{i}", f"v{i}") for i in range(batch["x_a"].size(0))]
    perturbation = make_scaffold_input_perturbation("shuffle_within_pair_v3", pairs)
    perturbed = model.build_pair_context_from_states(
        state_a, state_b, is_self, scaffold_input_perturbation=perturbation
    )

    assert captured["perturbation"] is perturbation
    assert baseline.topo_ab is not None and perturbed.topo_ab is not None
    assert not torch.allclose(baseline.topo_ab, perturbed.topo_ab)
    assert baseline.plan is not None and perturbed.plan is not None


def test_precomputed_graph_matches_inline_stitch_bitwise() -> None:
    model, batch = _tiny_model_and_batch()
    state_a, state_b, is_self = model._pair_node_states(batch)
    graph = model.generator.stitch(
        model._generator_state(state_a),
        model._generator_state(state_b),
        is_self,
    )
    assert graph is not None

    inline = model.build_pair_context_from_states(state_a, state_b, is_self)
    precomputed = model.build_pair_context_from_states(
        state_a,
        state_b,
        is_self,
        precomputed_graph=graph,
    )
    for field in inline._fields:
        inline_value = getattr(inline, field)
        precomputed_value = getattr(precomputed, field)
        if inline_value is None:
            assert precomputed_value is None
        else:
            torch.testing.assert_close(inline_value, precomputed_value, rtol=0.0, atol=0.0)


def test_precomputed_graph_rejects_perturbation_and_null_encoder() -> None:
    model, batch = _tiny_model_and_batch()
    state_a, state_b, is_self = model._pair_node_states(batch)
    graph = model.generator.stitch(
        model._generator_state(state_a),
        model._generator_state(state_b),
        is_self,
    )
    assert graph is not None
    pairs = [(f"u{row}", f"v{row}") for row in range(batch["x_a"].size(0))]
    perturbation = make_scaffold_input_perturbation("shuffle_within_pair_v3", pairs)

    with pytest.raises(ValueError, match="cannot be combined with a perturbation"):
        model.build_pair_context_from_states(
            state_a,
            state_b,
            is_self,
            scaffold_input_perturbation=perturbation,
            precomputed_graph=graph,
        )

    null_model, null_batch = _tiny_model_and_batch(generator_name="null")
    null_a, null_b, null_is_self = null_model._pair_node_states(null_batch)
    with pytest.raises(ValueError, match="requires a constructed graph encoder"):
        null_model.build_pair_context_from_states(
            null_a,
            null_b,
            null_is_self,
            precomputed_graph=graph,
        )


# --------------------------------------------------------------------------- (d) loss composition
#
# `EgoStitchModel` deliberately has no `aggregate_losses` (composite.py's
# "loss aggregation" section explains why: applying `real_ssl_scale` and
# `losses.stage1_total`/`.stage1_family_tensors` is trainer policy, owned by
# `train_egostitch.py`'s `_CompositeStep`). What *is* this composite's
# contract is that `forward`'s `"graph"`/`"embedding_ab"` output is exactly
# what `generator.auxiliary_losses`/`encoder.auxiliary_losses` need, reused
# rather than recomputed -- these two tests pin that.


def test_forward_graph_and_embedding_ab_feed_generator_and_encoder_auxiliary_losses() -> None:
    """`forward`'s `"graph"`/`"embedding_ab"` compose into the four pinned families.

    Mirrors exactly what `_CompositeStep.forward` does (`train_egostitch.py`):
    preserves the four-family split (``{"edge", "recon", "real", "ssl"}``,
    design §6) `_e2e_family_probe` depends on, and the ten
    `losses._RECON_COMPONENT_NAMES` (enforced inside `stage1_total` /
    `stage1_family_tensors` via `parts`'s ``recon_*`` breakdown).
    """
    model, _ = _tiny_model_and_batch()
    cfg = _tiny_e2e_config()
    batch = _full_training_batch(model, cfg)

    output = model(batch)
    graph = cast(StitchedGraph, output["graph"])
    embedding_ab = cast(GraphEmbedding, output["embedding_ab"])
    generator_losses = model.generator.auxiliary_losses(graph, batch)
    assert model.encoder is not None
    encoder_losses = model.encoder.auxiliary_losses(embedding_ab, batch)
    recon = {
        "feat": generator_losses["feat"],
        "exist": generator_losses["exist"],
        "mult": generator_losses["mult"],
        "slotadj": generator_losses["slotadj"],
        "gate": generator_losses["gate"],
        "ptr": generator_losses["ptr"],
        "div": generator_losses["div"],
        "align": generator_losses["align"],
        "rel": encoder_losses["rel_loss"],
    }
    edge_loss = torch.tensor(0.37)
    shared_kwargs: dict[str, object] = {
        "family": "egostitch_e2e",
        "edge": edge_loss,
        "recon": recon,
        "deg": generator_losses["deg"],
        "real_egostat": generator_losses["real_egostat"],
        "real_gin": generator_losses["real_gin"],
        "ssl_noise": generator_losses["ssl_noise"],
        "ssl_pool": generator_losses["ssl_pool"],
    }
    total, parts = stage1_total(model.generator_cfg, **shared_kwargs)  # type: ignore[arg-type]
    families = stage1_family_tensors(model.generator_cfg, **shared_kwargs)  # type: ignore[arg-type]

    assert set(families) == {"edge", "recon", "real", "ssl"}
    for name, value in families.items():
        assert torch.isfinite(value).all(), name
    assert torch.isfinite(total)
    expected_recon_parts = {
        "recon_feat",
        "recon_exist",
        "recon_mult",
        "recon_deg",
        "recon_slotadj",
        "recon_gate",
        "recon_ptr",
        "recon_align",
        "recon_div",
        "recon_rel",
    }
    assert expected_recon_parts <= parts.keys()
    for name in expected_recon_parts:
        assert parts[name] == parts[name]  # not NaN


def test_forward_graph_reused_by_auxiliary_losses_does_not_stitch_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`generator.auxiliary_losses` must consume `forward`'s own graph, not rebuild one.

    The whole point of `forward` exposing `"graph"`/`"embedding_ab"` (GAP 1,
    three-component refactor design §6) is that a caller computing auxiliary
    losses off them pays for exactly one stitch+encode pass per step, not
    two.
    """
    model, _ = _tiny_model_and_batch()
    cfg = _tiny_e2e_config()
    batch = _full_training_batch(model, cfg)

    calls = 0
    original_stitch = model.generator.stitch

    # `Any` for the same reason as the wrappers above: forwards blindly to
    # `original_stitch`'s real signature.
    def counted(*args: Any, **kwargs: Any) -> object:  # noqa: ANN401 -- generic monkeypatch forwarding wrapper
        nonlocal calls
        calls += 1
        return original_stitch(*args, **kwargs)

    monkeypatch.setattr(model.generator, "stitch", counted)

    output = model(batch)
    assert calls == 1

    model.generator.auxiliary_losses(cast(StitchedGraph, output["graph"]), batch)

    assert calls == 1


def test_decompose_pair_context_emits_the_two_key_contract() -> None:
    """`decompose_pair_context` returns exactly `{"full", "f_logit"}`."""
    model, batch = _tiny_model_and_batch()
    context = model.build_pair_context(batch)

    assert set(model.decompose_pair_context(context)) == {"full", "f_logit"}

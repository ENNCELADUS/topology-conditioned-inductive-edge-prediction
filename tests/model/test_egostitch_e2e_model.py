"""`EgoStitchModel` property tests (design rev 3 §3.4-§3.5 acceptance criteria).

Retargeted from `EgoStitchE2E` onto the three-component composite (three-
component refactor design §5, P2): every assertion here is behavior-
preserving, just re-expressed through `model.generator`/`model.encoder`/
`model.classifier` instead of the monolithic module's own submodules
(`model.trunk` -> `model.classifier.trunk`, `model.head` -> `model.classifier.head`,
`model.rel_head` -> `model.encoder.rel_head`, `model.generator.encode_nodes` ->
`model.generator.stage1.encode_nodes`). `model.cfg` and `model.generator_cfg`
stay exactly as named (pinned attributes, design coordinator amendment
2026-08-03).
"""

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
import torch
from src.data.feature_stats import compute_feature_stats
from src.model.egostitch.classifier.b0_v31 import (
    NULL_ALL_HEAD,
    B0V31PairClassifier,
    GatedCrossAttention,
    masks_for_null,
)
from src.model.egostitch.composite import E2EPairContext, EgoStitchModel
from src.model.egostitch.config import ClassifierConfig, E2EConfig, EncoderConfig, GeneratorConfig
from src.model.egostitch.generator.egostitch import EgoStitchImagineGenerator
from src.model.egostitch.generator.imagine import NodeEncoding, SlotSet


class _DirectionalConstantAttention(torch.nn.Module):
    """Return fixed AB/BA outputs under separate or joint trunk execution."""

    def __init__(self, batch_size: int, ab: float, ba: float) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.ab = ab
        self.ba = ba
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None,
        need_weights: bool,
    ) -> tuple[torch.Tensor, None]:
        del key, value, key_padding_mask, need_weights
        if query.size(0) == 2 * self.batch_size:
            output = torch.cat(
                (
                    torch.full_like(query[: self.batch_size], self.ab),
                    torch.full_like(query[self.batch_size :], self.ba),
                )
            )
        else:
            constant = self.ab if self.calls % 2 == 0 else self.ba
            output = torch.full_like(query, constant)
        self.calls += 1
        return output, None


def _tiny_e2e_config(
    *,
    feature_standardization: str = "row_layernorm",
    p_topo: float = 0.15,
) -> E2EConfig:
    """Build the shared tiny trunk/generator sizing used across this file's tests.

    Args:
        feature_standardization: The registered F0 preprocessing mode.
        p_topo: Training-time branch-dropout rate for the topo pathway.

    Returns:
        The tiny `E2EConfig`.
    """
    return E2EConfig(
        generator=GeneratorConfig(feature_standardization=feature_standardization),
        encoder=EncoderConfig(dim=16, layers=2),
        classifier=ClassifierConfig(
            d_model=32,
            encoder_layers=1,
            cross_attn_layers=2,
            n_heads=4,
            n_inj=1,
            xattn_heads=4,
            p_topo=p_topo,
        ),
    )


def _tiny_model_and_batch(
    *, p_topo: float = 0.15
) -> tuple[EgoStitchModel, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    # Stateless mode: these tests exercise trunk/scaffold behavior, not the
    # registered zscore_vfit_v1 statistics, and matches the prior
    # `standardize_features=True` (LayerNorm) semantics exactly.
    cfg = _tiny_e2e_config(feature_standardization="row_layernorm", p_topo=p_topo)
    model = EgoStitchModel(cfg).eval()
    b, t, d_in = 4, 6, model.input_dim
    n_ground = 5
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


def _classifier(model: EgoStitchModel) -> B0V31PairClassifier:
    """Narrow `model.classifier`'s protocol type to the concrete `B0V31PairClassifier`.

    Every fixture in this file builds `EgoStitchModel` with the default
    (registry key ``"b0_v31"``) classifier, so this assertion never fires at
    runtime -- it exists purely so mypy accepts the white-box `.trunk`/
    `.head` accesses below. `EgoStitchModel.classifier`'s static type is the
    `PairClassifier` ABC now that construction is registry-driven
    (design §12 P3), and the ABC deliberately exposes no submodule
    attributes -- that is what makes the classifier swappable at all.
    """
    assert isinstance(model.classifier, B0V31PairClassifier)
    return model.classifier


def _generator(model: EgoStitchModel) -> EgoStitchImagineGenerator:
    """Narrow `model.generator`'s protocol type to the concrete generator (see `_classifier`)."""
    assert isinstance(model.generator, EgoStitchImagineGenerator)
    return model.generator


def _topology_injections(
    model: EgoStitchModel,
    context: E2EPairContext,
    attention: _DirectionalConstantAttention,
    *,
    training: bool,
    edge_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Capture the AB-then-BA residuals emitted by the shared topology layer."""
    layer = cast(GatedCrossAttention, _classifier(model).trunk.topo_xattn[0])
    layer.attn = attention
    attention.reset()
    model.eval()
    layer.train(training)
    captures: list[torch.Tensor] = []

    def _capture(
        module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        del module
        captures.append((output - args[0].float()).detach())

    handle = layer.register_forward_hook(_capture)
    try:
        # masks=None => NULL_NONE: the topo pathway is fully active (there is
        # no content pathway left to null).
        model.score_pair_context(context, edge_mask=edge_mask)
    finally:
        handle.remove()
    return torch.cat(captures)


def test_shared_direction_centering_matches_training_and_evaluation() -> None:
    """A shared AB∪BA mean removes the old evaluation-only calibration offset."""
    model, batch = _tiny_model_and_batch()
    context = model.build_pair_context(batch)
    attention = _DirectionalConstantAttention(batch_size=4, ab=1.0, ba=3.0)
    layer = cast(GatedCrossAttention, _classifier(model).trunk.topo_xattn[0])
    with torch.no_grad():
        layer.gate.fill_(0.7)

    training = _topology_injections(model, context, attention, training=True)
    evaluation = _topology_injections(model, context, attention, training=False)

    assert bool(torch.count_nonzero(training[:4]))
    assert bool(torch.count_nonzero(training[4:]))
    assert torch.equal(evaluation, training)


def test_globally_constant_direction_pathway_is_zero_in_train_and_eval() -> None:
    """A constant shared calibration knob contributes exactly no residual."""
    model, batch = _tiny_model_and_batch()
    context = model.build_pair_context(batch)
    attention = _DirectionalConstantAttention(batch_size=4, ab=2.0, ba=2.0)
    layer = cast(GatedCrossAttention, _classifier(model).trunk.topo_xattn[0])
    with torch.no_grad():
        layer.gate.fill_(0.7)

    training = _topology_injections(model, context, attention, training=True)
    evaluation = _topology_injections(model, context, attention, training=False)

    assert torch.equal(training, torch.zeros_like(training))
    assert torch.equal(evaluation, torch.zeros_like(evaluation))
    assert int(layer.ema_updates.item()) == 1


def test_symmetric_trunk_updates_one_shared_ema_once_per_step() -> None:
    """Every conditioning layer consumes one AB∪BA statistic per model step."""
    model, batch = _tiny_model_and_batch()
    context = model.build_pair_context(batch)
    edge_mask = torch.tensor([True, True, False, False])
    layer = cast(GatedCrossAttention, _classifier(model).trunk.topo_xattn[0])
    layer.attn = _DirectionalConstantAttention(batch_size=4, ab=1.0, ba=3.0)

    model.train()
    model.score_pair_context(context, edge_mask=edge_mask)

    assert int(layer.ema_updates.item()) == 1
    assert torch.equal(layer.ema_mu, torch.full_like(layer.ema_mu, 2.0))


def test_pair_symmetry_all_conditions() -> None:
    model, batch = _tiny_model_and_batch()
    batch["ground_id_b"] = batch["ground_id_a"].roll(shifts=1, dims=1)
    topo_xattn = _classifier(model).trunk.topo_xattn[0]
    assert isinstance(topo_xattn, GatedCrossAttention)
    with torch.no_grad():
        topo_xattn.gate.fill_(0.4)  # open gate: symmetry must hold with live conditioning
    swapped = dict(batch)
    swapped["emb_a"], swapped["emb_b"] = batch["emb_b"], batch["emb_a"]
    swapped["len_a"], swapped["len_b"] = batch["len_b"], batch["len_a"]
    swapped["x_a"], swapped["x_b"] = batch["x_b"], batch["x_a"]
    swapped["ground_a"], swapped["ground_b"] = batch["ground_b"], batch["ground_a"]
    swapped["ground_id_a"], swapped["ground_id_b"] = (
        batch["ground_id_b"],
        batch["ground_id_a"],
    )
    for null in (None, NULL_ALL_HEAD):
        masks = None if null is None else masks_for_null(null, 4, torch.device("cpu"))
        out_ij = model(batch, masks=masks)["logits"]
        out_ji = model(swapped, masks=masks)["logits"]
        assert torch.allclose(out_ij, out_ji, atol=1e-5), f"asymmetry under {null}"

    p0_model, p0_batch = _tiny_model_and_batch(p_topo=0.0)
    p0_batch["ground_id_b"] = p0_batch["ground_id_a"].roll(shifts=1, dims=1)
    p0_topo_xattn = _classifier(p0_model).trunk.topo_xattn[0]
    assert isinstance(p0_topo_xattn, GatedCrossAttention)
    with torch.no_grad():
        p0_topo_xattn.gate.fill_(0.4)
    p0_swapped = dict(p0_batch)
    for key_a, key_b in (
        ("emb_a", "emb_b"),
        ("len_a", "len_b"),
        ("x_a", "x_b"),
        ("ground_a", "ground_b"),
        ("ground_id_a", "ground_id_b"),
    ):
        p0_swapped[key_a], p0_swapped[key_b] = p0_batch[key_b], p0_batch[key_a]
    assert torch.allclose(
        p0_model(p0_batch)["logits"], p0_model(p0_swapped)["logits"], atol=1e-5
    ), "asymmetry under p0"


def test_train_mask_equals_eval_bypass() -> None:
    model, batch = _tiny_model_and_batch()
    batch["ground_id_b"] = batch["ground_id_a"].roll(shifts=1, dims=1)
    topo_xattn = _classifier(model).trunk.topo_xattn[0]
    assert isinstance(topo_xattn, GatedCrossAttention)
    with torch.no_grad():
        topo_xattn.gate.fill_(0.4)
    dec = model.decompose(batch)  # eval-time hard bypasses
    masked = model(batch, masks=masks_for_null(NULL_ALL_HEAD, 4, torch.device("cpu")))["logits"]
    assert torch.allclose(masked, dec["f_logit"], atol=1e-6), "mask!=bypass for NULL_ALL_HEAD"


def test_f_logit_invariant_to_scaffold() -> None:
    model, batch = _tiny_model_and_batch()
    dec_a = model.decompose(batch)
    batch2 = dict(batch)
    batch2["x_a"] = torch.randn_like(batch["x_a"])  # changes slots => scaffold
    dec_b = model.decompose(batch2)
    # f_logit depends on token streams only — x_a feeds imagination, not trunk
    assert torch.allclose(dec_a["f_logit"], dec_b["f_logit"], atol=1e-6)


def test_missing_grounding_fails_closed() -> None:
    model, batch = _tiny_model_and_batch()
    batch.pop("ground_a")
    with pytest.raises(ValueError, match="requires real grounding"):
        model(batch)


def test_f_logit_invariant_to_grounding() -> None:
    model, batch = _tiny_model_and_batch()
    dec_a = model.decompose(batch)
    batch2 = dict(batch)
    batch2["ground_a"] = torch.randn_like(batch["ground_a"])
    batch2["ground_b"] = torch.randn_like(batch["ground_b"])
    batch2["ground_id_a"] = torch.randint_like(batch["ground_id_a"], 0, 1000)
    batch2["ground_id_b"] = torch.randint_like(batch["ground_id_b"], 0, 1000)
    dec_b = model.decompose(batch2)
    # f_logit nulls the (only remaining) topo pathway, so grounding never reaches the trunk.
    assert torch.allclose(dec_a["f_logit"], dec_b["f_logit"], atol=1e-6)


def test_relational_head_is_absent_from_every_scored_logit() -> None:
    model, batch = _tiny_model_and_batch()
    before = {name: value.clone() for name, value in model.decompose(batch).items()}
    assert model.encoder is not None
    assert model.encoder.rel_head is not None
    model.encoder.rel_head = None
    after = model.decompose(batch)
    assert before.keys() == after.keys()
    for name in before:
        assert torch.equal(before[name], after[name]), name


def test_w_rel_config_wires_formal_no_l_rel_arm() -> None:
    model = EgoStitchModel(E2EConfig(encoder=EncoderConfig(w_rel=0.0)))
    assert model.cfg.encoder.w_rel == 0.0
    assert model.generator_cfg.w_rel == 0.0
    assert model.encoder is not None
    assert model.encoder.rel_head is None
    with pytest.raises(ValueError, match="w_rel"):
        EncoderConfig(w_rel=-0.01)


# --------------------------------------------------------------------------- Task 15:
# `probe_states` — read-only STE token-state export for representation probes


def test_probe_states_returns_ste_token_states() -> None:
    """`probe_states` exports the STE token states that condition the trunk."""
    model, batch = _tiny_model_and_batch()
    states = model.probe_states(batch)
    b = batch["x_a"].size(0)
    assert states.shape[0] == b
    assert states.shape[-1] == model.cfg.classifier.d_model
    assert states.requires_grad is False
    again = model.probe_states(batch)
    assert torch.equal(states, again)  # deterministic in eval mode (no dropout)


def test_probe_states_reflects_probe_node_features() -> None:
    """Changing the probe node's own features changes its exported token states."""
    model, batch = _tiny_model_and_batch()
    baseline = model.probe_states(batch)
    perturbed = dict(batch)
    perturbed["x_a"] = torch.randn_like(batch["x_a"])
    changed = model.probe_states(perturbed)
    assert not torch.allclose(baseline, changed)


def test_probe_states_accepts_self_pair_batch() -> None:
    """`probe_states` runs on a self-pair batch (spec Sec 13.9 single-ego path)."""
    model, batch = _tiny_model_and_batch()
    batch["emb_b"] = batch["emb_a"]
    batch["len_b"] = batch["len_a"]
    batch["x_b"] = batch["x_a"]
    batch["is_self"] = torch.ones(batch["x_a"].size(0), dtype=torch.bool)
    states = model.probe_states(batch)
    assert states.shape[0] == batch["x_a"].size(0)


def test_decompose_builds_pair_context_once_and_matches_explicit_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, batch = _tiny_model_and_batch()
    with torch.no_grad():
        cast(GatedCrossAttention, _classifier(model).trunk.topo_xattn[0]).gate.data.fill_(0.4)
    calls = 0
    original = model.build_pair_context

    def counted(batch_arg: dict[str, torch.Tensor]) -> E2EPairContext:
        nonlocal calls
        calls += 1
        return original(batch_arg)

    monkeypatch.setattr(model, "build_pair_context", counted)
    decomposed = model.decompose(batch)
    assert calls == 1
    context = original(batch)
    expected = {
        "full": model.score_pair_context(context),
        "f_logit": model.score_pair_context(
            context, masks=masks_for_null(NULL_ALL_HEAD, 4, torch.device("cpu"))
        ),
    }
    for key, value in expected.items():
        assert torch.allclose(decomposed[key], value, atol=1e-6)


def test_self_pairs_encode_one_ego_and_use_exact_identity_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, batch = _tiny_model_and_batch()
    batch["emb_b"] = batch["emb_a"]
    batch["len_b"] = batch["len_a"]
    batch["x_b"] = batch["x_a"]
    batch["ground_b"] = batch["ground_a"]
    batch["ground_id_b"] = batch["ground_id_a"]
    batch["is_self"] = torch.ones(4, dtype=torch.bool)

    encode_calls = 0
    original_encode = _generator(model).stage1.encode_nodes

    def counted_encode(*args: torch.Tensor, **kwargs: torch.Tensor) -> NodeEncoding:
        nonlocal encode_calls
        encode_calls += 1
        return original_encode(*args, **kwargs)

    monkeypatch.setattr(_generator(model).stage1, "encode_nodes", counted_encode)

    import src.model.egostitch.generator.egostitch as generator_module

    sinkhorn_calls = 0
    original_sinkhorn = cast(
        Callable[..., torch.Tensor],
        generator_module.sinkhorn_log_plan,
    )

    def counted_sinkhorn(*args: torch.Tensor, **kwargs: object) -> torch.Tensor:
        nonlocal sinkhorn_calls
        sinkhorn_calls += 1
        return original_sinkhorn(*args, **kwargs)

    monkeypatch.setattr(generator_module, "sinkhorn_log_plan", counted_sinkhorn)
    context = model.build_pair_context(batch)
    assert encode_calls == 1
    assert sinkhorn_calls == 0
    expected = torch.eye(model.generator_cfg.slots).expand(4, -1, -1)
    assert context.plan is not None
    assert torch.equal(context.plan, expected)


def test_bf16_autocast_combines_self_and_sinkhorn_plans_in_fp32() -> None:
    """The fp32 Sinkhorn island owns the assembled plan dtype under autocast."""
    model, batch = _tiny_model_and_batch()
    is_self = torch.tensor([True, False, True, False])
    state_a = model.encode_node_state(
        batch["emb_a"],
        batch["len_a"],
        batch["x_a"],
        batch["ground_a"],
        batch["ground_id_a"],
    )
    state_b = model.encode_node_state(
        batch["emb_b"],
        batch["len_b"],
        batch["x_b"],
        batch["ground_b"],
        batch["ground_id_b"],
    )
    assert state_a.slots is not None and state_b.slots is not None
    state_a = state_a._replace(
        slots=SlotSet(*(value.to(torch.bfloat16) for value in state_a.slots))
    )
    state_b = state_b._replace(
        slots=SlotSet(*(value.to(torch.bfloat16) for value in state_b.slots))
    )

    with torch.autocast("cpu", dtype=torch.bfloat16):
        context = model.build_pair_context_from_states(state_a, state_b, is_self)

    assert context.plan is not None
    assert context.plan.dtype == torch.float32
    assert context.log_plan is not None
    assert context.log_plan.dtype == torch.float32
    expected_identity = torch.eye(model.generator_cfg.slots)
    assert torch.equal(context.plan[is_self], expected_identity.expand(2, -1, -1))


def test_bf16_autocast_returns_registered_fp32_readout_and_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registered BF16 path keeps the pair readout and logits in fp32."""
    model, batch = _tiny_model_and_batch()
    head_input_dtypes: list[torch.dtype] = []
    original_forward = _classifier(model).head.forward

    def capture_head_input(feat: torch.Tensor) -> torch.Tensor:
        head_input_dtypes.append(feat.dtype)
        return original_forward(feat)

    monkeypatch.setattr(_classifier(model).head, "forward", capture_head_input)

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(batch)["logits"]

    assert head_input_dtypes == [torch.float32]
    assert output.dtype == torch.float32


def test_generator_uses_the_configured_standardization_mode() -> None:
    model = EgoStitchModel(
        E2EConfig(generator=GeneratorConfig(feature_standardization="zscore_vfit_v1"))
    )
    assert _generator(model).stage1.feature_standardization == "zscore_vfit_v1"


def test_set_feature_stats_reaches_the_generator() -> None:
    gen = np.random.default_rng(0)
    cfg = E2EConfig(generator=GeneratorConfig(feature_standardization="zscore_vfit_v1"))
    model = EgoStitchModel(cfg)
    rows = (30.0 + 4.0 * gen.standard_normal((32, model.generator_cfg.input_dim))).astype(
        np.float32
    )
    stats = compute_feature_stats(rows, [f"n{i}" for i in range(32)])
    model.set_feature_stats(stats)
    assert model.feature_stats_digest_hex == stats.digest


def test_zscore_vfit_v1_registered_stats_actually_reach_the_forward_pass() -> None:
    """The production default must change trunk outputs, not just pass through wiring.

    Builds the same tiny architecture twice (identical seed, so identical
    weights) differing only in `feature_standardization`, and drives both
    through the full `forward` path on an offset, anisotropic node-feature
    fixture. A per-dimension zscore and a per-row LayerNorm compute distinct
    functions of such a fixture, so if the registered transform were silently
    bypassed (e.g. `zscore_vfit_v1` collapsing to an identity or to
    `row_layernorm` under the hood), the two logit tensors would match.
    """
    d_in = 1536  # frozen generator input_dim (spec Sec 0 table); E2EConfig has no override
    fixture_gen = np.random.default_rng(0)
    # Large per-dimension mean plus small per-row noise: a per-row LayerNorm
    # and a per-dimension zscore diverge sharply on this shape, whereas they
    # would agree (up to floating point) on a roughly isotropic input.
    mean_vector = fixture_gen.uniform(10.0, 50.0, size=d_in).astype(np.float32)
    stats_rows = mean_vector[None, :] + 0.5 * fixture_gen.standard_normal(
        (64, d_in)
    ).astype(np.float32)
    stats = compute_feature_stats(stats_rows, [f"n{i}" for i in range(64)])

    batch_gen = torch.Generator().manual_seed(1)
    b, t, n_ground = 4, 6, 5
    node_rows = torch.from_numpy(mean_vector)[None, :] + 0.5 * torch.randn(
        b, d_in, generator=batch_gen
    )
    ground_rows = torch.from_numpy(mean_vector)[None, None, :] + 0.5 * torch.randn(
        b, n_ground, d_in, generator=batch_gen
    )
    batch = {
        "emb_a": torch.randn(b, t, d_in, generator=batch_gen),
        "emb_b": torch.randn(b, t, d_in, generator=batch_gen),
        "len_a": torch.full((b,), t, dtype=torch.long),
        "len_b": torch.full((b,), t, dtype=torch.long),
        "x_a": node_rows,
        "x_b": node_rows.clone(),
        "ground_a": ground_rows,
        "ground_b": ground_rows.clone(),
        "ground_id_a": torch.randint(0, 1000, (b, n_ground), dtype=torch.long),
        "ground_id_b": torch.randint(0, 1000, (b, n_ground), dtype=torch.long),
    }

    def _activate_conditioning_gates(model: EgoStitchModel) -> None:
        # The trunk's topo injection is zero-init gated cross-attention
        # (module docstring); at a fresh init the gate is zero, so the
        # pathway -- and hence the F0 standardization choice -- would not
        # move the logits at all. Move it off zero, exactly as
        # test_decompose_builds_pair_context_once_and_matches_explicit_heads does.
        with torch.no_grad():
            cast(GatedCrossAttention, _classifier(model).trunk.topo_xattn[0]).gate.data.fill_(0.7)

    torch.manual_seed(0)
    zscore_model = EgoStitchModel(
        _tiny_e2e_config(feature_standardization="zscore_vfit_v1")
    ).eval()
    zscore_model.set_feature_stats(stats)
    _activate_conditioning_gates(zscore_model)
    with torch.no_grad():
        zscore_logits = zscore_model(batch)["logits"]
    assert zscore_logits.shape == (b,)
    assert torch.isfinite(zscore_logits).all()

    torch.manual_seed(0)
    layernorm_model = EgoStitchModel(
        _tiny_e2e_config(feature_standardization="row_layernorm")
    ).eval()
    _activate_conditioning_gates(layernorm_model)
    with torch.no_grad():
        layernorm_logits = layernorm_model(batch)["logits"]

    assert not torch.allclose(zscore_logits, layernorm_logits)

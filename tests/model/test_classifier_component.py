"""Tests for the extracted `PairClassifier` component (design 2026-08-02 §3, §4).

Proves `B0V31PairClassifier` is a behavior-preserving, verbatim extraction of
`EgoStitchModel.score_pair_context` (née `EgoStitchE2E.score_pair_context`):
same weights, same inputs, same outputs, in both the topology-conditioned and
the null-generator (`cond=None`) configurations. Also pins the design §4
AB/BA symmetry invariant this component now owns: one trunk batch, one
synchronized centering statistic, one EMA update per `forward` call, and
train/eval agreement.

`PairInputs.tokens_a`/`tokens_b` are already-encoded (2026-08-03 correction):
`forward` never calls `self.encoder` itself, so every equivalence test here
runs `classifier.encode_tokens` explicitly first (`_pair_inputs`) rather than
handing raw per-node token streams straight to `forward`. The per-unique-node
call-count property this split exists to preserve is pinned in
`tests/model/test_composite.py`, at the composite level where the cache
actually lives.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.testing
from src.model.egostitch.classifier.b0_v31 import (
    NULL_ALL_HEAD,
    B0V31PairClassifier,
    GatedCrossAttention,
    masks_for_null,
)
from src.model.egostitch.composite import E2EPairContext, EgoStitchModel
from src.model.egostitch.config import E2EConfig
from src.model.egostitch.graph import GraphEmbedding, PairConditioning, PairInputs

# --- Shared fixtures ----------------------------------------------------------


def _tiny_e2e_config(*, feature_standardization: str = "row_layernorm") -> E2EConfig:
    """The shared tiny trunk sizing used across this file's tests."""
    return E2EConfig(
        d_model=32,
        encoder_layers=1,
        cross_attn_layers=2,
        n_heads=4,
        n_inj=1,
        ste_dim=16,
        ste_layers=2,
        xattn_heads=4,
        p_topo=0.15,
        feature_standardization=feature_standardization,
    )


def _tiny_model_and_batch() -> tuple[EgoStitchModel, dict[str, torch.Tensor]]:
    """Build a small, real `EgoStitchModel` plus a matching batch (ground truth)."""
    torch.manual_seed(0)
    cfg = _tiny_e2e_config()
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


def _classifier_from(model: EgoStitchModel, cfg: E2EConfig) -> B0V31PairClassifier:
    """Build a `B0V31PairClassifier` sharing `model`'s pair-trunk weights."""
    classifier = B0V31PairClassifier(
        input_dim=model.input_dim,
        d_model=cfg.d_model,
        encoder_layers=cfg.encoder_layers,
        n_heads=cfg.n_heads,
        cross_attn_layers=cfg.cross_attn_layers,
        n_inj=cfg.n_inj,
        xattn_heads=cfg.xattn_heads,
        conditioning_ema_decay=cfg.conditioning_ema_decay,
    ).eval()
    classifier.encoder.load_state_dict(model.classifier.encoder.state_dict())
    classifier.trunk.load_state_dict(model.classifier.trunk.state_dict())
    classifier.head.load_state_dict(model.classifier.head.state_dict())
    return classifier


def _standalone_classifier(cfg: E2EConfig, *, input_dim: int) -> B0V31PairClassifier:
    """Build a `B0V31PairClassifier` with no accompanying `EgoStitchModel`."""
    return B0V31PairClassifier(
        input_dim=input_dim,
        d_model=cfg.d_model,
        encoder_layers=cfg.encoder_layers,
        n_heads=cfg.n_heads,
        cross_attn_layers=cfg.cross_attn_layers,
        n_inj=cfg.n_inj,
        xattn_heads=cfg.xattn_heads,
        conditioning_ema_decay=cfg.conditioning_ema_decay,
    )


def _pair_inputs(
    classifier: B0V31PairClassifier,
    batch: dict[str, torch.Tensor],
    *,
    edge_mask: torch.Tensor | None = None,
) -> PairInputs:
    """Build `PairInputs` by running `classifier`'s own `encode_tokens` (2026-08-03 fix).

    `PairInputs.tokens_a`/`tokens_b` are already-encoded, so this always
    routes through `encode_tokens` rather than passing `batch["emb_a"]`
    straight through.
    """
    return PairInputs(
        tokens_a=classifier.encode_tokens(batch["emb_a"], batch["len_a"]),
        tokens_b=classifier.encode_tokens(batch["emb_b"], batch["len_b"]),
        len_a=batch["len_a"],
        len_b=batch["len_b"],
        edge_mask=edge_mask,
    )


def _cond_from_context(context: E2EPairContext) -> PairConditioning:
    assert context.topo_ab is not None and context.topo_ba is not None
    return PairConditioning(
        ab=GraphEmbedding(tokens=context.topo_ab, pooled=context.topo_ab.mean(dim=1)),
        ba=GraphEmbedding(tokens=context.topo_ba, pooled=context.topo_ba.mean(dim=1)),
    )


class _DirectionalConstantAttention(torch.nn.Module):
    """Return fixed AB/BA outputs for the shared, doubled trunk batch.

    Mirrors `tests/model/test_egostitch_e2e_model.py`'s fixture of the same
    name: with `n_inj == 1` and one call per `forward`, `query` always has
    `2 * batch_size` rows (AB half first, BA half second).
    """

    def __init__(self, batch_size: int, ab: float, ba: float) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.ab = ab
        self.ba = ba

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
        assert query.size(0) == 2 * self.batch_size
        output = torch.cat(
            (
                torch.full_like(query[: self.batch_size], self.ab),
                torch.full_like(query[self.batch_size :], self.ba),
            )
        )
        return output, None


class _FixedAttention(torch.nn.Module):
    """Return one fixed, row-indexed output regardless of query content."""

    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None,
        need_weights: bool,
    ) -> tuple[torch.Tensor, None]:
        del query, key, value, key_padding_mask, need_weights
        return self.output, None


def _tiny_pair_and_cond(
    cfg: E2EConfig, *, batch_size: int, n_topo: int
) -> tuple[PairInputs, PairConditioning]:
    """Build a synthetic already-encoded `PairInputs` (2026-08-03 fix).

    `tokens_a`/`tokens_b` stand in directly for `encode_tokens`'s output, at
    `cfg.d_model` width -- these tests pin trunk-level structural properties
    (AB/BA symmetry, EMA, the fp32 head island) independent of the token
    encoder itself, so there is no real per-node data behind them.
    """
    b, t = batch_size, 6
    pair = PairInputs(
        tokens_a=torch.randn(b, t, cfg.d_model),
        tokens_b=torch.randn(b, t, cfg.d_model),
        len_a=torch.full((b,), t, dtype=torch.long),
        len_b=torch.full((b,), t, dtype=torch.long),
    )
    tokens = torch.randn(b, n_topo, cfg.d_model)
    cond = PairConditioning(
        ab=GraphEmbedding(tokens=tokens, pooled=tokens.mean(dim=1)),
        ba=GraphEmbedding(tokens=tokens, pooled=tokens.mean(dim=1)),
    )
    return pair, cond


# --- (a)/(b): numerical equivalence with `score_pair_context` -----------------


def test_conditioned_matches_score_pair_context() -> None:
    """Full topology-conditioned forward matches `score_pair_context` exactly."""
    model, batch = _tiny_model_and_batch()
    cfg = _tiny_e2e_config()
    context = model.build_pair_context(batch)
    classifier = _classifier_from(model, cfg)
    pair = _pair_inputs(classifier, batch)
    cond = _cond_from_context(context)

    expected = model.score_pair_context(context)
    actual = classifier(pair, cond)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)


def test_unconditioned_matches_f_logit_bypass() -> None:
    """`cond=None` matches the live model's `f_logit` (NULL_ALL_HEAD) bypass."""
    model, batch = _tiny_model_and_batch()
    cfg = _tiny_e2e_config()
    classifier = _classifier_from(model, cfg)
    pair = _pair_inputs(classifier, batch)

    expected = model.decompose(batch)["f_logit"]
    actual = classifier(pair, None)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)


def test_conditioned_masked_off_also_matches_f_logit() -> None:
    """`cond` present but topo-masked-off equals the live model's `f_logit`."""
    model, batch = _tiny_model_and_batch()
    cfg = _tiny_e2e_config()
    context = model.build_pair_context(batch)
    classifier = _classifier_from(model, cfg)
    pair = _pair_inputs(classifier, batch)
    cond = _cond_from_context(context)
    masks = masks_for_null(NULL_ALL_HEAD, 4, torch.device("cpu"))

    expected = model.decompose(batch)["f_logit"]
    actual = classifier(pair, cond, masks=masks)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)


# --- (e): `cond=None` equals the conditioned model's own `f_logit` bypass -----


def test_cond_none_equals_conditioned_model_f_logit_bypass() -> None:
    """Self-referential check: the classifier's own two paths agree."""
    model, batch = _tiny_model_and_batch()
    cfg = _tiny_e2e_config()
    context = model.build_pair_context(batch)
    classifier = _classifier_from(model, cfg)
    pair = _pair_inputs(classifier, batch)
    cond = _cond_from_context(context)
    masks = masks_for_null(NULL_ALL_HEAD, 4, torch.device("cpu"))

    conditioned_masked_off = classifier(pair, cond, masks=masks)
    unconditioned = classifier(pair, None)

    torch.testing.assert_close(unconditioned, conditioned_masked_off, rtol=0.0, atol=0.0)


# --- (c): ema_updates == 1 and the centering statistic ------------------------


def test_ema_updates_once_and_matches_joint_ab_ba_mean() -> None:
    """One `forward` call advances `ema_updates` by exactly 1 to the joint mean.

    Mirrors `tests/model/test_egostitch_e2e_model.py`'s
    `test_symmetric_trunk_updates_one_shared_ema_once_per_step`
    (`ema_mu == 2.0` for `ab=1.0, ba=3.0`).
    """
    cfg = _tiny_e2e_config()
    classifier = _standalone_classifier(cfg, input_dim=8)
    classifier.train()
    pair, cond = _tiny_pair_and_cond(cfg, batch_size=4, n_topo=3)
    layer = cast(GatedCrossAttention, classifier.trunk.topo_xattn[0])
    # Deliberate test double: `_DirectionalConstantAttention` duck-types
    # `nn.MultiheadAttention`'s call contract without being one, exactly as
    # `tests/model/test_egostitch_conditioning.py`'s `_FixedAttention` swaps
    # into the same `.attn` slot -- the checker is right that the nominal
    # types disagree; the substitution is intentional.
    layer.attn = _DirectionalConstantAttention(batch_size=4, ab=1.0, ba=3.0)  # type: ignore[assignment]

    classifier(pair, cond)

    assert int(layer.ema_updates.item()) == 1
    assert torch.equal(layer.ema_mu, torch.full_like(layer.ema_mu, 2.0))


# --- Full endpoint swap gives identical logits (pair symmetry) ----------------


def test_full_endpoint_swap_is_symmetric() -> None:
    model, batch = _tiny_model_and_batch()
    batch["ground_id_b"] = batch["ground_id_a"].roll(shifts=1, dims=1)
    cfg = _tiny_e2e_config()
    classifier = _classifier_from(model, cfg)
    with torch.no_grad():
        cast(GatedCrossAttention, classifier.trunk.topo_xattn[0]).gate.fill_(0.4)

    swapped_batch = dict(batch)
    for key_a, key_b in (
        ("emb_a", "emb_b"),
        ("len_a", "len_b"),
        ("x_a", "x_b"),
        ("ground_a", "ground_b"),
        ("ground_id_a", "ground_id_b"),
    ):
        swapped_batch[key_a], swapped_batch[key_b] = batch[key_b], batch[key_a]

    context = model.build_pair_context(batch)
    swapped_context = model.build_pair_context(swapped_batch)

    out_ab = classifier(_pair_inputs(classifier, batch), _cond_from_context(context))
    out_ba = classifier(
        _pair_inputs(classifier, swapped_batch), _cond_from_context(swapped_context)
    )

    torch.testing.assert_close(out_ab, out_ba, rtol=0.0, atol=1e-5)


# --- Train and eval produce identical residuals for the same conditioning -----


def test_train_and_eval_residuals_agree() -> None:
    """Mirrors `test_shared_direction_centering_matches_training_and_evaluation`.

    Toggles only the `GatedCrossAttention` submodule's training flag while
    keeping the rest of the classifier in eval mode throughout, exactly as
    the reference test's `_topology_injections` helper does (`model.eval()`
    once, then `layer.train(training)`). Without this isolation, upstream
    dropout in the plain trunk layers would perturb `cls_token` differently
    between the two calls and the captured residuals would differ by
    floating-point rounding alone -- a confound unrelated to the centering
    statistic this test is pinning.
    """
    cfg = _tiny_e2e_config()
    classifier = _standalone_classifier(cfg, input_dim=8)
    classifier.eval()
    pair, cond = _tiny_pair_and_cond(cfg, batch_size=4, n_topo=3)
    layer = cast(GatedCrossAttention, classifier.trunk.topo_xattn[0])
    # Deliberate test double -- see the identical comment above.
    layer.attn = _DirectionalConstantAttention(batch_size=4, ab=1.0, ba=3.0)  # type: ignore[assignment]
    with torch.no_grad():
        layer.gate.fill_(0.7)

    captures: list[torch.Tensor] = []

    def _capture(
        module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        del module
        captures.append((output - args[0].float()).detach())

    handle = layer.register_forward_hook(_capture)
    try:
        layer.train(True)
        classifier(pair, cond)
        layer.train(False)
        classifier(pair, cond)
    finally:
        handle.remove()

    train_residual, eval_residual = captures
    assert bool(torch.count_nonzero(train_residual))
    torch.testing.assert_close(eval_residual, train_residual, rtol=0.0, atol=0.0)


# --- Padded DDP filler rows are excluded from `mu` -----------------------------


def test_edge_mask_filler_rows_excluded_from_mu() -> None:
    cfg = _tiny_e2e_config()
    classifier = _standalone_classifier(cfg, input_dim=8)
    classifier.train()
    b, t, n_topo = 2, 6, 3
    pair = PairInputs(
        tokens_a=torch.randn(b, t, cfg.d_model),
        tokens_b=torch.randn(b, t, cfg.d_model),
        len_a=torch.full((b,), t, dtype=torch.long),
        len_b=torch.full((b,), t, dtype=torch.long),
        edge_mask=torch.tensor([True, False]),  # pair 1 is a DDP filler row
    )
    tokens = torch.randn(b, n_topo, cfg.d_model)
    cond = PairConditioning(
        ab=GraphEmbedding(tokens=tokens, pooled=tokens.mean(dim=1)),
        ba=GraphEmbedding(tokens=tokens, pooled=tokens.mean(dim=1)),
    )
    layer = cast(GatedCrossAttention, classifier.trunk.topo_xattn[0])
    d_model = cfg.d_model
    # Doubled batch order is (AB pair0, AB pair1, BA pair0, BA pair1); pair 1
    # is the filler in both halves. Naive inclusion would give mu == 76.0
    # (mean of 1, 100, 3, 200); correct exclusion gives mu == 2.0.
    attention_output = torch.cat(
        [
            torch.full((1, 1, d_model), 1.0),
            torch.full((1, 1, d_model), 100.0),
            torch.full((1, 1, d_model), 3.0),
            torch.full((1, 1, d_model), 200.0),
        ]
    )
    # Deliberate test double -- see the identical comment above.
    layer.attn = _FixedAttention(attention_output)  # type: ignore[assignment]

    classifier(pair, cond)

    assert torch.equal(layer.ema_mu, torch.full_like(layer.ema_mu, 2.0))


# --- The fp32 head island survives under autocast ------------------------------


def test_fp32_head_island_survives_under_autocast() -> None:
    """The head computation stays bit-exact fp32 regardless of ambient autocast.

    Stubs the trunk with a fixed fp32 tensor so the only thing that can
    differ between the two calls is whether an ambient bf16 autocast context
    wraps `forward`. This isolates the `head(feat.float())` island itself
    from the encoder/trunk's own (expected, legitimate) autocast
    sensitivity -- comparing whole-model output with and without ambient
    autocast would fail even for a correct island, since the encoder has no
    such protection and is meant to run in bf16 under autocast.
    """
    cfg = _tiny_e2e_config()
    classifier = _standalone_classifier(cfg, input_dim=8).eval()
    fixed_feat = torch.randn(8, cfg.d_model)  # 2 * batch_size rows (AB then BA)

    class _FixedTrunk(torch.nn.Module):
        def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
            del args, kwargs
            return fixed_feat

    classifier.trunk = _FixedTrunk()  # type: ignore[assignment]
    pair, cond = _tiny_pair_and_cond(cfg, batch_size=4, n_topo=3)

    baseline = classifier(pair, cond)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        under_autocast = classifier(pair, cond)

    assert baseline.dtype == torch.float32
    assert under_autocast.dtype == torch.float32
    torch.testing.assert_close(under_autocast, baseline, rtol=0.0, atol=0.0)

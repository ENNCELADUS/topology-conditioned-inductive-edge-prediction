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
from src.model.egostitch.classifier.b0_v31 import GatedCrossAttention
from src.model.egostitch.composite import EgoStitchModel
from src.model.egostitch.config import E2EConfig
from src.model.egostitch.generator import NullGenerator, StitchedGraph
from src.model.egostitch.generator.assemble import make_scaffold_input_perturbation
from src.model.egostitch.generator.losses import stage1_family_tensors, stage1_total
from src.model.egostitch.graph import GraphEmbedding, PairInputs


def _tiny_e2e_config(*, n_ground: int = 5, **overrides: object) -> E2EConfig:
    """The shared tiny composite sizing used across this file's tests."""
    kwargs: dict[str, object] = {
        "d_model": 32,
        "encoder_layers": 1,
        "cross_attn_layers": 2,
        "n_heads": 4,
        "n_inj": 1,
        "ste_dim": 16,
        "ste_layers": 2,
        "xattn_heads": 4,
        "p_topo": 0.15,
        "feature_standardization": "row_layernorm",
        "n_ground": n_ground,
    }
    kwargs.update(overrides)
    return E2EConfig(**kwargs)  # type: ignore[arg-type]


def _tiny_model_and_batch(
    **config_overrides: object,
) -> tuple[EgoStitchModel, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    # Same mypy limitation `_tiny_e2e_config`'s own `E2EConfig(**kwargs)` call
    # hits (see its `# type: ignore[arg-type]` below): a `**dict[str, object]`
    # unpack can't be proven to avoid `_tiny_e2e_config`'s one narrower-typed
    # named parameter (`n_ground: int`), even though this call site never
    # actually targets it.
    cfg = _tiny_e2e_config(**config_overrides)  # type: ignore[arg-type]
    model = EgoStitchModel(cfg).eval()
    b, t, d_in = 4, 6, model.input_dim
    n_ground = cfg.n_ground
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
    -- because both paths run the classifier fully unconditioned. Registry-
    driven `generator.name: null` wiring into `EgoStitchModel.__init__`
    itself is P3 (design §12); this test exercises the null path at the
    component-composition level available today.
    """
    model, batch = _tiny_model_and_batch()
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

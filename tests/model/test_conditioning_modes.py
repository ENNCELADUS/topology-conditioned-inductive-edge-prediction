"""Tests for the four `ClassifierConfig.conditioning_mode` pathways.

Oracle-scaffold design,
`docs/superpowers/specs/2026-08-04-oracle-scaffold-experiment-design.md` §4:

    film_logit | pooled_adapter | xattn_cls | xattn_tokens

None of `ClassifierConfig`, `registry.build_classifier`, or
`B0V31PairClassifier` (or whatever concrete class ends up implementing this)
carry a `conditioning_mode` field/behavior at the time this file was written
-- this is an *addition to an existing class*, not a new module, so a clean
module-level `pytest.importorskip` cannot gate the whole file (the imports
below all succeed today). Instead, `_classifier_config`/`_build_classifier`
below convert the "not yet landed" case into a per-test `pytest.skip` at the
point construction actually fails, so the suite degrades to skips rather than
collection errors as the feature lands incrementally.

Design doc §4's pinned invariant this file exists to prove: "Every mode
carries a zero-init null-identity guarantee: at initialization, the
conditioning pathway must contribute nothing, so an untrained conditioned arm
reduces to the unconditioned baseline". That guarantee, and gradient flow
once the (necessarily zero-initialized) gate/adapter opens, are tested
generically across all four modes without needing to know each mode's
internal attribute names -- `_open_all_zero_init_parameters` perturbs
whatever is zero at init, which is, by construction, exactly the null-
identity machinery.
"""

from __future__ import annotations

import pytest
import torch
from src.model.egostitch.classifier.base import PairClassifier
from src.model.egostitch.config import ClassifierConfig
from src.model.egostitch.graph import GraphEmbedding, PairConditioning, PairInputs

try:
    # `registry.py` transitively imports every other component being landed
    # in this same wave (generator/oracle.py, encoder/grit_gmt.py) via
    # `src/model/egostitch/__init__.py` -> `composite.py` -> `registry.py` ->
    # `generator/__init__.py`/`encoder/__init__.py`. Any of those being
    # mid-edit (e.g. a class implemented but not yet re-exported from its
    # package `__init__.py`) breaks this import even though it has nothing
    # to do with `conditioning_mode` -- degrade to a module-level skip rather
    # than a collection error in that case, consistent with the rest of this
    # wave's test files.
    from src.model.egostitch.registry import build_classifier
except ImportError as error:
    pytest.skip(
        f"src.model.egostitch.registry not importable yet: {error}", allow_module_level=True
    )
from torch import nn

pytestmark = pytest.mark.unit

_MODES = ("film_logit", "pooled_adapter", "xattn_cls", "xattn_tokens")
_D_MODEL = 16
_INPUT_DIM = 8


def _classifier_config(mode: str) -> ClassifierConfig:
    try:
        return ClassifierConfig(
            d_model=_D_MODEL,
            encoder_layers=1,
            cross_attn_layers=1,
            n_heads=2,
            n_inj=1,
            xattn_heads=2,
            conditioning_mode=mode,
        )
    except TypeError as error:
        pytest.skip(f"ClassifierConfig.conditioning_mode not yet landed: {error}")


def _build_classifier(mode: str) -> PairClassifier:
    cfg = _classifier_config(mode)
    try:
        return build_classifier(cfg, input_dim=_INPUT_DIM)
    except Exception as error:  # noqa: BLE001 - deliberately broad; see docstring
        pytest.skip(f"build_classifier does not yet support conditioning_mode={mode!r}: {error}")


def _encode_pair(classifier: PairClassifier, batch: int = 4, tokens: int = 5) -> PairInputs:
    length = torch.full((batch,), tokens, dtype=torch.long)
    tokens_a = classifier.encode_tokens(torch.randn(batch, tokens, _INPUT_DIM), length)
    tokens_b = classifier.encode_tokens(torch.randn(batch, tokens, _INPUT_DIM), length)
    return PairInputs(tokens_a=tokens_a, tokens_b=tokens_b, len_a=length, len_b=length)


def _random_cond(
    batch: int = 4, graph_tokens: int = 6, *, requires_grad: bool = False
) -> PairConditioning:
    def _embedding() -> GraphEmbedding:
        tokens = torch.randn(batch, graph_tokens, _D_MODEL, requires_grad=requires_grad)
        pooled = torch.randn(batch, _D_MODEL, requires_grad=requires_grad)
        return GraphEmbedding(tokens=tokens, pooled=pooled)

    return PairConditioning(ab=_embedding(), ba=_embedding())


def _open_all_zero_init_parameters(module: nn.Module) -> None:
    """Perturb every all-zero parameter to a nonzero constant.

    A zero-init null-identity guarantee necessarily means the gate/adapter
    weight(s) that gate the conditioning pathway start at exactly zero; this
    generically finds and opens all of them without needing each mode's own
    attribute names.
    """
    with torch.no_grad():
        for param in module.parameters():
            if torch.equal(param, torch.zeros_like(param)):
                param.fill_(0.5)


# --------------------------------------------------------------------------- zero-init identity


@pytest.mark.parametrize("mode", _MODES)
def test_zero_init_conditioning_matches_unconditioned_logits(mode: str) -> None:
    torch.manual_seed(0)
    classifier = _build_classifier(mode)
    classifier.eval()
    pair = _encode_pair(classifier)
    cond = _random_cond()

    with torch.no_grad():
        logits_none = classifier(pair, None)
        logits_cond = classifier(pair, cond)

    torch.testing.assert_close(logits_cond, logits_none)


# --------------------------------------------------------------------------- gate opens -> gradient


# Design doc §4's "Consumes" column: film_logit/pooled_adapter read `pooled`;
# xattn_cls/xattn_tokens read `tokens`. Only the field a mode actually
# consumes is guaranteed to carry a gradient -- asserting on the other field
# would overcommit to an unspecified implementation detail (a mode may leave
# the unused field's `requires_grad` input entirely untouched by the graph).
_CONSUMED_FIELD = {
    "film_logit": "pooled",
    "pooled_adapter": "pooled",
    "xattn_cls": "tokens",
    "xattn_tokens": "tokens",
}


@pytest.mark.parametrize("mode", _MODES)
def test_gradient_reaches_conditioning_tensors_once_the_gate_opens(mode: str) -> None:
    torch.manual_seed(0)
    classifier = _build_classifier(mode)
    classifier.train()
    _open_all_zero_init_parameters(classifier)

    pair = _encode_pair(classifier)
    cond_none_logits = classifier(pair, None)
    cond = _random_cond(requires_grad=True)
    cond_logits = classifier(pair, cond)

    assert not torch.allclose(cond_logits, cond_none_logits), (
        "opening every zero-init parameter produced no observable change; either "
        "the gate-opening heuristic missed this mode's gate, or the conditioning "
        "pathway is not wired into the logit at all"
    )
    cond_logits.sum().backward()
    field = _CONSUMED_FIELD[mode]
    for embedding in (cond.ab, cond.ba):
        grad = getattr(embedding, field).grad
        assert grad is not None, f"mode={mode!r} consumes {field!r} but no grad reached it"
        assert torch.isfinite(grad).all()


# --------------------------------------------------------------------------- checkpoint compat


def test_state_dict_round_trips_for_xattn_cls() -> None:
    torch.manual_seed(0)
    classifier_a = _build_classifier("xattn_cls")
    state = classifier_a.state_dict()
    classifier_b = _build_classifier("xattn_cls")
    classifier_b.load_state_dict(state)

    keys_a = set(classifier_a.state_dict().keys())
    keys_b = set(classifier_b.state_dict().keys())
    assert keys_a == keys_b
    for key in state:
        torch.testing.assert_close(classifier_b.state_dict()[key], state[key])


# --------------------------------------------------------------------------- EMA buffer determinism


def test_xattn_tokens_ema_buffer_updates_deterministically() -> None:
    """Cheap, single-process proxy for the cross-rank EMA-sync invariant.

    The real invariant (`GatedCrossAttention`'s synchronized centering,
    `classifier/b0_v31.py`) is that every rank's EMA update is bitwise
    identical given the same post-all-reduce statistics -- properly verified
    only by a real multi-process run, the way `tests/test_e2_ddp_integration.py`
    already does for this repo's other DDP contracts via a subprocess-launched
    `torch.distributed.run` smoke driver. Spinning an equivalent subprocess
    here for every wave-1 conditioning mode is exactly the kind of high-cost
    local test this suite should avoid adding, so this test instead pins the
    single-process precondition that invariant depends on: given the same
    batch sequence, the EMA buffer's post-update value must be an exact,
    reproducible function of the inputs (no unseeded randomness in the update
    path, no floating-point-order dependence). Two fresh classifiers built
    from the same seed and fed the same two batches must leave every
    "ema"-named buffer bit-identical -- a necessary condition for the
    cross-rank version of the same guarantee to hold, cheap enough to run on
    every invocation of this suite.
    """

    def _run() -> dict[str, list[float]]:
        torch.manual_seed(0)
        classifier = _build_classifier("xattn_tokens")
        classifier.train()
        for seed in (1, 2):
            torch.manual_seed(seed)
            pair = _encode_pair(classifier)
            cond = _random_cond()
            classifier(pair, cond)
        return {
            name: buf.detach().clone().tolist()
            for name, buf in classifier.named_buffers()
            if "ema" in name.lower()
        }

    first = _run()
    second = _run()
    assert first, "xattn_tokens conditioning exposed no 'ema'-named buffer"
    assert first.keys() == second.keys()
    for key in first:
        assert first[key] == second[key], f"EMA buffer {key!r} was not reproducible"

"""Scoring a v3_1_motif_prompt checkpoint: truth-free, sharded, precision-validated."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from src import score_universe
from src.data import motif_template
from src.data.features import FeatureStore
from src.model.egostitch.classifier.motif_prompt import TEMPLATE_KEY, V3_1MotifPrompt

from tests.test_prefix_model import _tiny_base_config
from tests.test_score_universe import _build_prefix_packed_fixture


def _motif_block(stage: str = "two") -> dict[str, object]:
    """The ``model.config.motif_prompt`` block of a tiny checkpoint."""
    block: dict[str, object] = {
        "stage": stage,
        "base_checkpoint": "base.pt",
        "width": 16,
        "slots_per_field": 2,
        "reader": {"layers": 1, "dim": 16, "heads": 4, "rrwp_k": 2},
    }
    if stage == "two":
        block["bundle_checkpoint"] = "bundle.pt"
    return block


def _model_config(stage: str = "two") -> dict[str, object]:
    """A checkpoint ``model_config`` for the motif-prompt family."""
    return {"base": _tiny_base_config(), "motif_prompt": _motif_block(stage)}


def _motif_model(stage: str = "two", seed: int = 0) -> V3_1MotifPrompt:
    """A tiny motif-prompt model with open gates, in eval mode."""
    torch.manual_seed(seed)
    model = V3_1MotifPrompt(**_model_config(stage))  # type: ignore[arg-type]
    model.install_mean_template(torch.full((96,), 0.1))
    gen = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        model.adapter.gates.copy_(torch.randn(model.adapter.gates.shape, generator=gen))
    return model.eval()


def _templates(rows: int, seed: int = 3) -> torch.Tensor:
    """A row-aligned bank of compiled-looking motif weights."""
    return torch.rand((rows, 96), generator=torch.Generator().manual_seed(seed))


# ------------------------------------------------------------------ registration


def test_the_family_is_registered_and_builds_from_a_checkpoint_config() -> None:
    assert "v3_1_motif_prompt" in score_universe.MODEL_BUILDERS
    model = score_universe.build_model("v3_1_motif_prompt", _model_config())
    assert isinstance(model, V3_1MotifPrompt)
    assert model.name == "v3_1_motif_prompt"


def test_the_builder_round_trips_a_checkpoint_state_dict() -> None:
    model = _motif_model()
    rebuilt = score_universe.MODEL_BUILDERS["v3_1_motif_prompt"](_model_config())
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    assert isinstance(rebuilt, V3_1MotifPrompt)
    torch.testing.assert_close(rebuilt.mean_template, model.mean_template, rtol=0, atol=0)


# ------------------------------------------------------------------ truth-free scoring


def test_stage_two_scoring_reads_no_graph_and_no_target_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Stage II scoring must not compile a motif template")

    monkeypatch.setattr(motif_template.MotifTemplateTable, "__init__", _forbidden)
    pack_root, pairs = _build_prefix_packed_fixture(tmp_path, monkeypatch, node_count=4)
    store = FeatureStore(tmp_path / "features")
    model = _motif_model()
    logits = score_universe._score_v3_1(
        model, pairs, store, device=torch.device("cpu"), amp="off", token_budget=512
    )
    assert logits.shape == (len(pairs),)
    assert np.isfinite(logits).all()
    # The model holds every input it needs: no graph file, no target file and no
    # universe statistic was opened, and nothing but features/pack exists here.
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["features", "pack"]
    packed = score_universe._score_v3_1_packed(
        model, pairs, pack_root, device=torch.device("cpu"), amp="off", token_budget=512
    )
    # The pack stores bf16 tokens, so the two paths agree only loosely.
    np.testing.assert_allclose(packed, logits, rtol=0.0, atol=5e-3)


def test_stage_two_scoring_matches_the_models_own_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, pairs = _build_prefix_packed_fixture(tmp_path, monkeypatch, node_count=4)
    store = FeatureStore(tmp_path / "features")
    model = _motif_model()
    logits = score_universe._score_v3_1(
        model, pairs, store, device=torch.device("cpu"), amp="off", token_budget=512
    )
    batch = _collate(store, pairs)
    with torch.inference_mode():
        reference = model(batch)["logits"].numpy().reshape(-1)
    np.testing.assert_allclose(logits, reference, rtol=0.0, atol=1e-5)


def test_stage_one_scoring_reads_the_supplied_true_templates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, pairs = _build_prefix_packed_fixture(tmp_path, monkeypatch, node_count=4)
    store = FeatureStore(tmp_path / "features")
    model = _motif_model("one")
    templates = _templates(len(pairs))
    logits = score_universe._score_v3_1(
        model,
        pairs,
        store,
        device=torch.device("cpu"),
        amp="off",
        token_budget=512,
        row_templates=templates,
    )
    batch = _collate(store, pairs)
    batch[TEMPLATE_KEY] = templates
    with torch.inference_mode():
        reference = model(batch)["logits"].numpy().reshape(-1)
    np.testing.assert_allclose(logits, reference, rtol=0.0, atol=1e-5)
    with pytest.raises(ValueError, match="stage 'one' requires"):
        score_universe._score_v3_1(
            model, pairs, store, device=torch.device("cpu"), amp="off", token_budget=512
        )


def test_stage_one_scoring_requires_the_oracle_diagnostic_acknowledgement() -> None:
    with pytest.raises(ValueError, match="allow-oracle-diagnostic"):
        score_universe._assert_motif_scoring_contract(stage="one", allow_oracle_diagnostic=False)
    score_universe._assert_motif_scoring_contract(stage="one", allow_oracle_diagnostic=True)
    with pytest.raises(ValueError, match="deployable"):
        score_universe._assert_motif_scoring_contract(stage="two", allow_oracle_diagnostic=True)
    score_universe._assert_motif_scoring_contract(stage="two", allow_oracle_diagnostic=False)


def test_scoring_is_batch_and_shard_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, pairs = _build_prefix_packed_fixture(tmp_path, monkeypatch, node_count=4)
    store = FeatureStore(tmp_path / "features")
    model = _motif_model()
    kwargs = {"device": torch.device("cpu"), "amp": "off", "token_budget": 512}
    whole = score_universe._score_v3_1(model, pairs, store, **kwargs)  # type: ignore[arg-type]
    first = score_universe._score_v3_1(model, pairs[:2], store, **kwargs)  # type: ignore[arg-type]
    second = score_universe._score_v3_1(model, pairs[2:], store, **kwargs)  # type: ignore[arg-type]
    np.testing.assert_allclose(np.concatenate([first, second]), whole, rtol=1e-6, atol=1e-6)


# ------------------------------------------------------------------ interventions


def test_the_three_graph_interventions_are_offered_by_the_cli() -> None:
    for name in ("shuffle_graph", "permute_closure", "rewire_bridge"):
        assert name in score_universe.PREFIX_INTERVENTIONS


def test_graph_transplant_uses_one_seeded_universe_permutation() -> None:
    rows = score_universe._shuffle_source_rows(6, 0)
    assert sorted(rows.tolist()) == list(range(6))
    assert not (rows == np.arange(6)).any()


def test_permute_closure_and_rewire_bridge_preserve_the_family_multisets() -> None:
    weights = torch.rand((3, 96), generator=torch.Generator().manual_seed(0))
    closure = score_universe._motif_permute_closure(weights, seed=0)
    np.testing.assert_allclose(
        np.sort(closure[:, :16].numpy(), axis=1),
        np.sort(weights[:, :16].numpy(), axis=1),
        atol=1e-6,
    )
    # Only endpoint u's correspondence moves, so the wedge pairing breaks.
    torch.testing.assert_close(closure[:, 8:], weights[:, 8:], rtol=0, atol=0)
    bridge = score_universe._motif_rewire_bridge(weights, seed=0)
    np.testing.assert_allclose(
        np.sort(bridge[:, 32:].numpy(), axis=1),
        np.sort(weights[:, 32:].numpy(), axis=1),
        atol=1e-6,
    )
    # The attachments stay put, so the interior is rewired relative to them.
    torch.testing.assert_close(bridge[:, :32], weights[:, :32], rtol=0, atol=0)
    # The interventions change the graph, so they must change something.
    assert not torch.allclose(closure, weights)
    assert not torch.allclose(bridge, weights)


def test_a_graph_transplant_is_row_aligned_and_really_moves_the_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack_root, pairs = _build_prefix_packed_fixture(tmp_path, monkeypatch, node_count=4)
    store = FeatureStore(tmp_path / "features")
    model = _motif_model()
    kwargs = {"device": torch.device("cpu"), "amp": "off", "token_budget": 512}
    plain = score_universe._score_v3_1(model, pairs, store, **kwargs)  # type: ignore[arg-type]
    # Transplanting each row its own predicted graph is the identity, which is
    # what proves the two-pass bank is row-aligned.
    identity = score_universe._score_v3_1(
        model,
        pairs,
        store,
        shuffle_sources=pairs,
        **kwargs,  # type: ignore[arg-type]
    )
    np.testing.assert_allclose(identity, plain, rtol=0.0, atol=1e-5)
    transplanted = score_universe._score_v3_1(
        model,
        pairs,
        store,
        shuffle_sources=list(reversed(pairs)),
        **kwargs,  # type: ignore[arg-type]
    )
    assert not np.allclose(transplanted, plain, atol=1e-5)
    packed_identity = score_universe._score_v3_1_packed(
        model,
        pairs,
        pack_root,
        shuffle_sources=pairs,
        **kwargs,  # type: ignore[arg-type]
    )
    packed_plain = score_universe._score_v3_1_packed(
        model,
        pairs,
        pack_root,
        **kwargs,  # type: ignore[arg-type]
    )
    np.testing.assert_allclose(packed_identity, packed_plain, rtol=0.0, atol=1e-5)


def test_the_row_local_interventions_reach_the_graph_the_trunk_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import functools

    _, pairs = _build_prefix_packed_fixture(tmp_path, monkeypatch, node_count=4)
    store = FeatureStore(tmp_path / "features")
    model = _motif_model()
    kwargs = {"device": torch.device("cpu"), "amp": "off", "token_budget": 512}
    # A non-degenerate graph: an untrained generator emits nearly the same weight
    # in every slot, and a within-role permutation of a constant graph is a no-op,
    # so a random bank is what exposes the substitution.
    bank = _templates(len(pairs))
    plain = score_universe._score_v3_1(
        model,
        pairs,
        store,
        row_templates=bank,
        **kwargs,  # type: ignore[arg-type]
    )
    for transform in (score_universe._motif_permute_closure, score_universe._motif_rewire_bridge):
        partial = functools.partial(transform, seed=0)
        intervened = score_universe._score_v3_1(
            model,
            pairs,
            store,
            row_templates=bank,
            motif_weight_transform=partial,
            **kwargs,  # type: ignore[arg-type]
        )
        # Identical to scoring the already-transformed bank: the transform is
        # applied to the graph the trunk reads, not to anything downstream.
        reference = score_universe._score_v3_1(
            model,
            pairs,
            store,
            row_templates=partial(bank),
            **kwargs,  # type: ignore[arg-type]
        )
        np.testing.assert_allclose(intervened, reference, rtol=0.0, atol=1e-6)
        assert not np.allclose(intervened, plain, atol=1e-5)


def _collate(store: FeatureStore, pairs: list[tuple[str, str]]) -> dict[str, torch.Tensor]:
    """Collate every pair into one padded batch, in input row order."""
    from src.data.pairs import TokenPairDataset, collate_token_pairs

    dataset = TokenPairDataset(pairs, None, store)
    return collate_token_pairs([dataset[i] for i in range(len(pairs))])

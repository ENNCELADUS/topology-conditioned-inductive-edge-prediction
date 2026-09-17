"""Scoring a v3_1_motif_prompt checkpoint: truth-free, sharded, precision-validated."""

from __future__ import annotations

from pathlib import Path
from typing import cast

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
    """A tiny motif-prompt model with open gates, in eval mode.

    Stage II snapshots the Stage I bundle as the immutable teacher during
    training (`src.train_b0.build_model`), so a trained Stage II checkpoint
    always carries ``teacher.*`` parameters; the fixture mirrors that.
    """
    torch.manual_seed(seed)
    model = V3_1MotifPrompt(**_model_config(stage))  # type: ignore[arg-type]
    model.install_mean_template(torch.full((96,), 0.1))
    gen = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        model.adapter.gates.copy_(torch.randn(model.adapter.gates.shape, generator=gen))
    if stage == "two":
        model.initialize_teacher()
    return model.eval()


def _templates(rows: int, seed: int = 3) -> torch.Tensor:
    """A row-aligned bank of compiled-looking motif weights."""
    return torch.rand((rows, 96), generator=torch.Generator().manual_seed(seed))


def _assert_validation_forward_reproduces_the_scorer(
    model: V3_1MotifPrompt,
    pairs: list[tuple[str, str]],
    store: FeatureStore,
    batch: dict[str, torch.Tensor],
    *,
    row_templates: torch.Tensor | None = None,
) -> None:
    """Fail unless an autocast-wrapped validation forward gives the scorer's logits.

    Accelerate wraps a prepared model's whole forward -- trunk and output head --
    in the run's bf16 autocast, so `src.train_b0._evaluate_val_universe` calls it
    exactly as this helper does; a bare `torch.autocast` is the substitution the
    other loop tests already use for that wrapper. Scoring pins
    ``pair_autocast: False``, and validation freezes the checkpoint and the ONE
    V_val topology threshold that `test_protocol` later replays on the scorer's
    logits, so the two must agree bit for bit and not merely both run. One
    oversized batch forces the scorer to pad exactly as `_collate` does, leaving
    the pair pass as the only difference under test.
    """
    scored = score_universe._score_v3_1(
        model,
        pairs,
        store,
        device=torch.device("cpu"),
        amp="bf16",
        token_budget=1 << 16,
        row_templates=row_templates,
    )
    with torch.inference_mode(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        validated = model(batch)["logits"].float().numpy().reshape(-1)
    np.testing.assert_array_equal(validated, scored)


# ------------------------------------------------------------------ registration


def test_the_family_is_registered_and_builds_from_a_checkpoint_config() -> None:
    assert "v3_1_motif_prompt" in score_universe.MODEL_BUILDERS
    model = score_universe.build_model("v3_1_motif_prompt", _model_config())
    assert isinstance(model, V3_1MotifPrompt)
    assert model.name == "v3_1_motif_prompt"


def test_a_trained_stage_two_checkpoint_round_trips_through_the_loader(tmp_path: Path) -> None:
    """Stage II training snapshots ``R_T``, so the published state dict has ``teacher.*``.

    The strict load in `_load_checkpoint` rejects those keys unless the builder
    reconstructs the teacher submodule first -- which would reject every trained
    Stage II checkpoint.
    """
    model = _motif_model()
    assert any(key.startswith("teacher.") for key in model.state_dict())
    path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_family": "v3_1_motif_prompt",
            "model_config": _model_config(),
        },
        path,
    )

    rebuilt, family, _ = score_universe._load_checkpoint(path)

    assert family == "v3_1_motif_prompt"
    assert isinstance(rebuilt, V3_1MotifPrompt)
    assert rebuilt.teacher is not None
    torch.testing.assert_close(rebuilt.mean_template, model.mean_template, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(rebuilt.state_dict()[key], value, rtol=0, atol=0)


def test_a_stage_one_checkpoint_has_no_teacher_to_rebuild(tmp_path: Path) -> None:
    model = _motif_model("one")
    assert not any(key.startswith("teacher.") for key in model.state_dict())
    path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_family": "v3_1_motif_prompt",
            "model_config": _model_config("one"),
        },
        path,
    )

    rebuilt, _, _ = score_universe._load_checkpoint(path)

    assert cast(V3_1MotifPrompt, rebuilt).teacher is None


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
    _assert_validation_forward_reproduces_the_scorer(model, pairs, store, batch)


def test_the_fp32_validation_pair_pass_is_confined_to_this_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the motif family disables pair autocast at scoring time, so only it may here.

    ``v3_1`` and ``v3_1_prefix`` score with ``pair_autocast`` following ``--amp``.
    Promoting their validation forward would move every threshold those arms have
    already published instead of pinning it, so their autocast-wrapped forward must
    still return the run's reduced precision while the motif family returns fp32.
    """
    from src.model.egostitch.classifier.b0_v31 import V3_1

    from tests.test_score_universe import _tiny_v3_1_prefix

    _, pairs = _build_prefix_packed_fixture(tmp_path, monkeypatch, node_count=4)
    store = FeatureStore(tmp_path / "features")
    batch = _collate(store, pairs)
    unchanged: list[torch.dtype] = []
    for other in (_tiny_v3_1_prefix(), V3_1(**_tiny_base_config()).eval()):
        with torch.inference_mode(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            unchanged.append(other(batch)["logits"].dtype)
    assert unchanged == [torch.bfloat16, torch.bfloat16]
    with torch.inference_mode(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert _motif_model()(batch)["logits"].dtype is torch.float32


def test_stage_one_validation_forward_reproduces_the_scorers_pair_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage I freezes a threshold through the same autocast-wrapped forward."""
    _, pairs = _build_prefix_packed_fixture(tmp_path, monkeypatch, node_count=4)
    store = FeatureStore(tmp_path / "features")
    model = _motif_model("one")
    templates = _templates(len(pairs))
    batch = _collate(store, pairs)
    batch[TEMPLATE_KEY] = templates
    _assert_validation_forward_reproduces_the_scorer(
        model, pairs, store, batch, row_templates=templates
    )


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


def test_bf16_amp_scores_stage_two_and_stage_one_without_a_dtype_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--amp bf16`` encodes in bf16; the pair pass runs fp32 and must promote.

    Disabling autocast does not promote a tensor that is already bf16, so the
    fp32 reader and generator weights meet bf16 activations.
    """
    _, pairs = _build_prefix_packed_fixture(tmp_path, monkeypatch, node_count=4)
    store = FeatureStore(tmp_path / "features")
    kwargs = {"device": torch.device("cpu"), "token_budget": 512}
    stage_two = _motif_model()
    reference = score_universe._score_v3_1(
        stage_two,
        pairs,
        store,
        amp="off",
        **kwargs,  # type: ignore[arg-type]
    )
    amped = score_universe._score_v3_1(
        stage_two,
        pairs,
        store,
        amp="bf16",
        **kwargs,  # type: ignore[arg-type]
    )
    assert np.isfinite(amped).all()
    np.testing.assert_allclose(amped, reference, rtol=0.0, atol=5e-2)

    # The shuffle source pass predicts a graph from bf16 states too.
    transplanted = score_universe._score_v3_1(
        stage_two,
        pairs,
        store,
        shuffle_sources=list(reversed(pairs)),
        amp="bf16",
        **kwargs,  # type: ignore[arg-type]
    )
    assert np.isfinite(transplanted).all()

    stage_one = _motif_model("one")
    stage_one_logits = score_universe._score_v3_1(
        stage_one,
        pairs,
        store,
        amp="bf16",
        row_templates=_templates(len(pairs)),
        **kwargs,  # type: ignore[arg-type]
    )
    assert np.isfinite(stage_one_logits).all()


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


# ------------------------------------------------------------------ precision contract


def _bf16_logits(rows: int = 64, seed: int = 7) -> np.ndarray:
    """A logit column that passed through bf16 and was widened back to fp32."""
    values = torch.randn(rows, generator=torch.Generator().manual_seed(seed))
    return np.asarray(values.to(torch.bfloat16).float().numpy(), dtype=np.float32)


def _clean_logits(rows: int = 64, seed: int = 1) -> np.ndarray:
    """An honest fp32 logit column."""
    return np.asarray(np.random.default_rng(seed).standard_normal(rows), dtype=np.float32)


def _motif_meta(logit: np.ndarray) -> dict[str, object]:
    """Artifact metadata carrying the pinned motif-prompt precision contract."""
    return {
        "model_family": "v3_1_motif_prompt",
        "score_precision": {
            "contract": "v3_1_motif_prompt_pair_fp32_v1",
            "encode_autocast": "off",
            "pair_autocast": False,
            "pair_compute_dtype": "float32",
            "logit_storage_dtype": "float32",
        },
        "score_resolution": {"logit": score_universe.score_resolution_diagnostics(logit)},
    }


def _motif_artifact(logit: np.ndarray) -> score_universe.ScoresArtifact:
    """A minimal loaded artifact of this family."""
    rows = len(logit)
    return score_universe.ScoresArtifact(
        node_ids=[f"node_{i:02d}" for i in range(rows + 1)],
        u_idx=np.arange(rows, dtype=np.int32),
        v_idx=np.arange(1, rows + 1, dtype=np.int32),
        logit=logit,
        label=np.zeros(rows, dtype=np.int8),
        meta=_motif_meta(logit),
    )


def test_a_bf16_contaminated_motif_artifact_is_rejected() -> None:
    contaminated = _bf16_logits()
    honest = _motif_meta(contaminated)
    # An honestly recorded bf16 pair pass contradicts the pinned contract.
    cast(dict[str, object], honest["score_precision"])["pair_autocast"] = True
    with pytest.raises(ValueError, match="score_precision"):
        score_universe.validate_score_precision(contaminated, meta=honest, label="artifact")
    # And a forged fp32 provenance with self-consistent diagnostics is caught by
    # the grid itself: this is the landmine, a contaminated artifact that would
    # otherwise analyse cleanly.
    with pytest.raises(ValueError, match="bf16 grid"):
        score_universe.validate_score_precision(
            contaminated, meta=_motif_meta(contaminated), label="artifact"
        )
    with pytest.raises(ValueError, match="bf16 grid"):
        score_universe.validate_artifact_precision(_motif_artifact(contaminated))


def test_a_clean_motif_artifact_validates_through_the_artifact_wrapper() -> None:
    score_universe.validate_artifact_precision(_motif_artifact(_clean_logits()), label="artifact")


def test_missing_provenance_fails_closed_for_this_family() -> None:
    with pytest.raises(ValueError, match="missing score_precision provenance"):
        score_universe.validate_score_precision(
            np.zeros(8, dtype=np.float32),
            meta={"model_family": "v3_1_motif_prompt"},
            label="artifact",
        )


def test_a_non_float32_motif_logit_column_fails_closed() -> None:
    logit = _clean_logits()
    meta = _motif_meta(logit)
    with pytest.raises(ValueError, match="stored as float32"):
        score_universe.validate_score_precision(
            logit.astype(np.float64),
            meta=meta,
            label="artifact",
        )


def test_inconsistent_score_resolution_diagnostics_fail_closed() -> None:
    logit = _clean_logits(32, seed=2)
    meta = _motif_meta(logit)
    cast(dict[str, object], meta["score_resolution"])["logit"] = {"n_rows": 32, "n_unique": 1}
    with pytest.raises(ValueError, match="score_resolution"):
        score_universe.validate_score_precision(logit, meta=meta, label="artifact")


def test_a_degenerate_constant_column_is_not_mistaken_for_contamination() -> None:
    # Every constant column sits on the bf16 grid trivially; only a column with
    # more than one distinct value is evidence of a bf16 pair pass.
    logit = np.zeros(16, dtype=np.float32)
    score_universe.validate_score_precision(logit, meta=_motif_meta(logit), label="artifact")


def test_every_other_family_keeps_its_previous_validator_behaviour() -> None:
    # The dispatch must not start validating families that never had a contract.
    logit = _bf16_logits()
    score_universe.validate_score_precision(logit, meta={"model_family": "v3_1"}, label="artifact")
    score_universe.validate_score_precision(
        logit, meta={"model_family": "v3_1_coord_gen"}, label="artifact"
    )
    with pytest.raises(ValueError, match="retired"):
        score_universe.validate_score_precision(
            logit, meta={"model_family": "egostitch"}, label="artifact"
        )

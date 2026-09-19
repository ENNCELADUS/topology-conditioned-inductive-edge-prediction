"""Scoring contracts for the endpoint-only motif dictionary family."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from src import score_universe
from src.data.motif_dictionary import DictionaryArtifact, save_dictionary
from src.experiments import motif_dictionary_diagnostic as diagnostic
from src.experiments.motif_dictionary_diagnostic import build_oracle_checkpoint
from src.model.egostitch.classifier.motif_dictionary import V3_1MotifDictionary

from tests.test_prefix_model import _tiny_base_config
from tests.test_score_universe import _build_prefix_packed_fixture
from tests.test_score_universe_motif_prompt import _model_config, _motif_model


def _config(mode: str = "sequence") -> dict[str, object]:
    return {
        "base": _tiny_base_config(),
        "motif_prompt": {
            "stage": "two" if mode == "sequence" else "one",
            "base_checkpoint": "removed-source.pt",
            "bundle_checkpoint": "removed-bundle.pt" if mode == "sequence" else "",
            "width": 16,
            "slots_per_field": 2,
            "reader": {"layers": 1, "dim": 16, "heads": 4, "rrwp_k": 2},
        },
        "motif_dictionary": {
            "mode": mode,
            "dictionary_size": 2,
            "artifact_path": "removed-artifact.pt",
            "slot_checkpoint": "removed-slot.pt",
        },
    }


def _model(mode: str = "sequence") -> V3_1MotifDictionary:
    torch.manual_seed(7)
    model = V3_1MotifDictionary(**_config(mode))  # type: ignore[arg-type]
    tokens = torch.randn(2, 4, 16, generator=torch.Generator().manual_seed(8))
    tokens[:, 1] = tokens[:, 0]
    model.install_dictionary(
        DictionaryArtifact(
            weights=torch.zeros(2, 96),
            tokens=tokens,
            scales=torch.ones(4),
            temperature=1.0,
            mean_weights=torch.tensor([0.5, 0.5]),
            swap_index=torch.tensor([0, 1]),
            metadata={},
        )
    )
    with torch.no_grad():
        model.adapter.gates.fill_(0.75)
    return model.eval()


def test_dictionary_family_rebuilds_and_round_trips_without_source_files(tmp_path: Path) -> None:
    model = _model()
    config = _config()
    # Publication strips construction-only paths; constructor shape state is enough.
    dictionary = config["motif_dictionary"]
    assert isinstance(dictionary, dict)
    dictionary["artifact_path"] = ""
    dictionary["slot_checkpoint"] = ""
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "model_family": score_universe.MOTIF_DICTIONARY_FAMILY,
            "model_config": config,
            "model_state": model.state_dict(),
        },
        checkpoint,
    )

    rebuilt, family, _ = score_universe._load_checkpoint(checkpoint)

    assert family == score_universe.MOTIF_DICTIONARY_FAMILY
    assert isinstance(rebuilt, V3_1MotifDictionary)
    torch.testing.assert_close(rebuilt.dictionary_tokens, model.dictionary_tokens)
    assert bool(rebuilt.dictionary_installed)


def test_global_shuffle_route_agrees_between_serial_and_contiguous_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack, pairs = _build_prefix_packed_fixture(tmp_path, monkeypatch, node_count=6)
    model = _model()
    source_rows = score_universe._shuffle_source_rows(len(pairs), seed=19)
    sources = [pairs[int(index)] for index in source_rows]
    serial = score_universe._score_v3_1_packed(
        model,
        pairs,
        pack,
        device=torch.device("cpu"),
        amp="off",
        token_budget=512,
        shuffle_sources=sources,
    )
    ordinary = score_universe._score_v3_1_packed(
        model,
        pairs,
        pack,
        device=torch.device("cpu"),
        amp="off",
        token_budget=512,
    )
    assert float(np.max(np.abs(serial - ordinary))) > 1e-7

    split = len(pairs) // 2
    parts = []
    for start, stop in ((0, split), (split, len(pairs))):
        parts.append(
            score_universe._score_v3_1_packed(
                model,
                pairs[start:stop],
                pack,
                device=torch.device("cpu"),
                amp="off",
                token_budget=97,
                shuffle_sources=sources[start:stop],
            )
        )

    np.testing.assert_allclose(np.concatenate(parts), serial, rtol=0.0, atol=1e-6)


def test_dictionary_precision_contract_is_self_describing() -> None:
    logits = np.array([0.1234567, -0.7654321], dtype=np.float32)
    meta = {
        "model_family": score_universe.MOTIF_DICTIONARY_FAMILY,
        "score_precision": {
            "contract": score_universe._MOTIF_DICTIONARY_PRECISION_CONTRACT,
            "pair_compute_dtype": "float32",
            "pair_autocast": False,
            "logit_storage_dtype": "float32",
        },
        "score_resolution": {"logit": score_universe.score_resolution_diagnostics(logits)},
    }
    score_universe.validate_score_precision(logits, meta=meta, label="dictionary")


def test_diagnostic_checkpoint_initialization_embeds_every_inference_constant(
    tmp_path: Path,
) -> None:
    source = _motif_model("one")
    source_path = tmp_path / "stage1.pt"
    torch.save(
        {
            "model_family": "v3_1_motif_prompt",
            "model_config": _model_config("one"),
            "model_state": source.state_dict(),
        },
        source_path,
    )
    artifact_path = tmp_path / "dictionary.pt"
    save_dictionary(
        DictionaryArtifact(
            weights=torch.zeros(2, 96),
            tokens=torch.zeros(2, 4, 16),
            scales=torch.ones(4),
            temperature=1.0,
            mean_weights=torch.tensor([0.5, 0.5]),
            swap_index=torch.tensor([0, 1]),
            metadata={"scope": "training_only"},
        ),
        artifact_path,
    )
    output = tmp_path / "oracle.pt"

    build_oracle_checkpoint(artifact_path, source_path, output)
    artifact_path.unlink()
    source_path.unlink()
    rebuilt, family, _ = score_universe._load_checkpoint(output)

    assert family == score_universe.MOTIF_DICTIONARY_FAMILY
    assert isinstance(rebuilt, V3_1MotifDictionary)
    assert rebuilt.mode == "oracle"
    assert bool(rebuilt.dictionary_installed)


def test_diagnostic_reports_protocol_selection_and_matches_dictionary_output_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = ["true", "dictionary", "mean", "gates_off"]
    artifacts = {
        case: score_universe.ScoresArtifact(
            node_ids=["a", "b"],
            u_idx=np.array([0, 0, 1], dtype=np.int32),
            v_idx=np.array([0, 1, 1], dtype=np.int32),
            logit=np.array([3.0, float(index), -2.0], dtype=np.float32),
            label=np.array([1, 0, 1], dtype=np.int8),
            meta={},
        )
        for index, case in enumerate(cases)
    }

    def fake_load(path: Path) -> score_universe.ScoresArtifact:
        return artifacts[path.stem.removesuffix("_val_topology")]

    selected_counts = iter([1, 2, 3, 1])

    def fake_select(**_: object) -> SimpleNamespace:
        return SimpleNamespace(
            report={"selected": {"union_admitted_pair_count": next(selected_counts)}}
        )

    captured: dict[str, int] = {}

    def fake_density(**kwargs: object) -> dict[str, object]:
        target = kwargs["target_edges"]
        assert isinstance(target, int)
        captured["target_edges"] = target
        return {"rows": {case: {"target_edges": captured["target_edges"]} for case in cases}}

    split = SimpleNamespace(build_g_val=lambda: object(), buckets={})
    monkeypatch.setattr(diagnostic, "load_scores", fake_load)
    monkeypatch.setattr(diagnostic, "validate_artifact_precision", lambda *args, **kwargs: None)
    monkeypatch.setattr(diagnostic, "_load_val_region_split", lambda *args: split)
    monkeypatch.setattr(diagnostic, "select_fixed_threshold", fake_select)
    monkeypatch.setattr(diagnostic, "density_matched_report", fake_density)
    monkeypatch.setattr(diagnostic, "_edge_metrics", lambda path: {"auprc": 0.5})

    report = diagnostic.analyze_scores(tmp_path, tmp_path, "breadth_first", cases)

    assert captured["target_edges"] == 2
    assert report["matched_output_density"] == {
        "reference_case": "dictionary",
        "target_edges": 2,
        "report": {"rows": {case: {"target_edges": 2} for case in cases}},
    }
    assert all(
        "protocol_topology" in report["cases"][case]  # type: ignore[index]
        for case in cases
    )


def test_diagnostic_edge_metrics_keep_extreme_ranking_and_decisions_on_logits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = score_universe.ScoresArtifact(
        node_ids=["a", "b", "c", "d", "e"],
        u_idx=np.array([0, 0, 0, 0], dtype=np.int32),
        v_idx=np.array([1, 2, 3, 4], dtype=np.int32),
        logit=np.array([35.0, 36.0, 37.0, 38.0], dtype=np.float32),
        label=np.array([0, 0, 0, 1], dtype=np.int8),
        meta={},
    )
    monkeypatch.setattr(diagnostic, "load_scores", lambda _: artifact)
    monkeypatch.setattr(diagnostic, "validate_artifact_precision", lambda *args, **kwargs: None)

    metrics = diagnostic._edge_metrics(tmp_path / "extreme.npz")

    assert metrics["logit_threshold"] == 38.0
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["mcc"] == pytest.approx(1.0)

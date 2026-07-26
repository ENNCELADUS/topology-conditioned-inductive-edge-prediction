"""Contract tests for the EgoStitch rev-3.1 draft registration (spec §14.4.6-7)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Never

import pytest
from src import score_universe, train_egostitch
from src.experiments import g5_stage1, probes

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRATION_PATH = (
    REPO_ROOT / "docs/registrations/g5_e2e_stage1_preregistration_v3.json"
)
MARKDOWN_PATH = REPO_ROOT / "docs/registrations/g5_e2e_stage1_preregistration_v3.md"
EXPECTED_TRAINED_ARMS = (
    "full",
    "b0_e2e_f_only",
    "pair_topology",
    "p0",
    "cosine_pool",
    "no_l_rel",
)
EXPECTED_CONTROL_ARMS = (
    "structure_control_6a_v3",
    "structure_control_6e_v1",
)
V2_ARMS = (
    "full",
    "b0_e2e_f_only",
    "pair_topology",
    "structure_control_6a",
    "p0",
)


def _registration() -> dict[str, object]:
    snapshot = train_egostitch._preregistration_snapshot(REGISTRATION_PATH)
    return snapshot.payload


def _marker_paths(value: object, path: str = "$") -> Iterator[str]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield from _marker_paths(nested, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            yield from _marker_paths(nested, f"{path}[{index}]")
    elif isinstance(value, str) and "REQUIRED-BEFORE-BINDING" in value:
        yield path


def test_v3_registration_parses_as_a_nonbinding_draft() -> None:
    registration = _registration()

    assert registration["status"] == "DRAFT"
    assert registration["status"] != "BINDING"
    predecessor = registration["predecessor"]
    assert isinstance(predecessor, dict)
    assert predecessor["path"] == (
        "docs/registrations/g5_e2e_stage1_preregistration_v2.json"
    )
    assert predecessor["status"] == "BINDING"


def test_v3_registration_formal_probe_path_is_producer_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = _registration()
    probe_artifact = registration["probe_artifact"]
    assert isinstance(probe_artifact, dict)
    expected_path = probe_artifact.get("expected_path")
    assert expected_path == (
        "outputs/egostitch_e2e_stage1_v3/full/probes/e2e_probe_v2.npz"
    )

    # Exercise the draft exactly as it would behave after binding, without
    # mutating the governing registration or filling any binding placeholders.
    bound_copy = {**registration, "status": "BINDING"}
    registration_path = tmp_path / "docs/registrations/registration.json"
    registration_path.parent.mkdir(parents=True)
    registration_path.write_text(json.dumps(bound_copy), encoding="utf-8")
    registration_sha = hashlib.sha256(registration_path.read_bytes()).hexdigest()
    full_arm = registration["arms"]["full"]
    assert isinstance(full_arm, dict)
    registered_config = tmp_path / str(full_arm["training"])
    metadata_path = tmp_path / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "preregistration_sha256": registration_sha,
                "run_kind": "formal",
                "status": "complete",
                "formal_artifacts_published": True,
                "permanent_null": "none",
                "seed": 0,
                "partition_seed": 0,
                "config_path": str(registered_config),
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / str(expected_path)

    class ProbePathAccepted(Exception):
        pass

    def stop_after_path_validation(_: Path) -> Never:
        raise ProbePathAccepted

    monkeypatch.setattr(train_egostitch, "load_config", stop_after_path_validation)
    with pytest.raises(ProbePathAccepted):
        probes.produce_e2e_probe_artifact(
            checkpoint_path=tmp_path / "best.pt",
            run_metadata_path=metadata_path,
            preregistration_path=registration_path,
            data_root=tmp_path / "data",
            strategy="breadth_first",
            output_path=output_path,
            scope="formal_train",
        )


def test_v3_registration_freezes_rev31_contract() -> None:
    registration = _registration()
    arms = registration["arms"]
    assert isinstance(arms, dict)
    assert tuple(arms) == EXPECTED_TRAINED_ARMS + EXPECTED_CONTROL_ARMS

    reconstruction = registration["training_contract"]
    assert isinstance(reconstruction, dict)
    l_recon = reconstruction["l_recon"]
    assert isinstance(l_recon, dict)
    assert l_recon["component_weights"] == {
        "L_feat": 1.0,
        "L_exist": 0.5,
        "L_mult": 0.25,
        "L_deg": 0.5,
        "L_slotadj": 0.5,
        "L_gate": 0.25,
        "L_ptr": 0.25,
        "L_align": 0.5,
        "L_div": 0.1,
        "L_rel": 0.25,
    }
    assert l_recon["anneal"]["components"] == ["L_feat", "L_exist", "L_mult", "L_deg"]
    assert l_recon["anneal"]["fixed_factor_1_components"] == [
        "L_slotadj",
        "L_gate",
        "L_ptr",
        "L_align",
        "L_div",
        "L_rel",
    ]

    constants = registration["registered_constants"]
    assert constants == {
        "tau_adj": 0.5,
        "l_gate_pos_weight": 6.17,
        "tau_div": 0.5,
        "ego_target_cap_k": 16,
        "conditioning_mu_ema_decay": 0.99,
    }

    curriculum = reconstruction["curriculum"]
    assert curriculum["removed_v2_phase"] == "Phase-A pair_only head start"
    assert "first edge-active step" in curriculum["edge_active_entry"]
    conditioning = reconstruction["centered_conditioning"]
    assert conditioning["equation"] == (
        "cls <- cls + active * tanh(g) * (XAttn(...) - mu)"
    )
    assert conditioning["ema_decay"] == 0.99
    assert "single synchronized EMA" in conditioning["mu_eval"]

    grounding = registration["grounding"]
    assert grounding["method_id"] == "cosine_topk_v1"
    assert grounding["n_ground"] == 50
    assert grounding["reranker"] is None
    assert grounding["measured_slot_recall_ceilings"] == {
        "top50": 0.13952495387963418,
        "top20": 0.10728125418065595,
    }
    assert grounding["pool_method_hash"]["ordered_fields"] == [
        "method id",
        "n_ground",
        "shortlist M when present",
        "ordered F0/source-feature-pack digest",
        "role-universe identity",
    ]

    scaffold = registration["scaffold"]
    assert scaffold == {
        "feat_dim": 11,
        "edge_types": 4,
        "layout": ["onehot4(anchor)", "pi", "mult", "deg x 4", "t_k"],
    }

    control_6e = arms["structure_control_6e_v1"]
    assert arms["structure_control_6a_v3"]["scoring_provenance"]["scaffold_control"] == (
        score_universe._SCAFFOLD_CONTROL_SHUFFLE_V3
    )
    assert control_6e["scoring_provenance"]["scaffold_control"] == (
        score_universe._SCAFFOLD_CONTROL_REWIRE_V1
    )
    algorithm_6e = control_6e["algorithm"]
    assert algorithm_6e["transfer"] == (
        "delta = u * min(w_il, w_kj, c_ij - w_ij, c_kl - w_kl)"
    )
    assert algorithm_6e["slot_adjacency_recipient_capacity"] == "c_ij = pi_i * pi_j"
    assert algorithm_6e["plan_recipient_capacity"].startswith("infinity")
    assert "pi-weighted" in algorithm_6e["slot_adjacency_swap_space"]
    assert algorithm_6e["deliberately_not_invariant"].startswith(
        "rebuilt CLOSE degree channel"
    )

    probe = registration["probe_artifact"]
    assert probe["format"] == "egostitch_e2e_probe_v2"
    pi_v2_probe = (
        "Pi-consistency v2: plan mass on double-Hungarian same-identity cells "
        "divided by total plan mass"
    )
    assert pi_v2_probe in probe["probes"]
    assert "slot recall@n_g per run" in probe["probes"]

    qualification = registration["prebinding_qualification"]
    assert isinstance(qualification, dict)
    gates = qualification["gates"]
    assert isinstance(gates, list)
    assert len(gates) == 5
    assert gates[0]["applies_to_arm"] == "full"
    assert gates[0]["threshold"] == 0.0698
    assert gates[1]["threshold"] == 0.05
    assert gates[2]["threshold"] == 0.10
    assert qualification["protocol"]["max_attempts_total"] == 3
    assert qualification["protocol"]["v_qual_rehearsals"] == 1
    assert qualification["protocol"]["v_select"] == "sealed until first bound run"


def test_v3_markdown_declares_json_authority_and_matches_critical_pins() -> None:
    text = MARKDOWN_PATH.read_text(encoding="utf-8")

    assert "**Status: DRAFT." in text
    assert "**If the two disagree, the JSON governs.**" in text
    for arm in EXPECTED_TRAINED_ARMS + EXPECTED_CONTROL_ARMS:
        assert f"`{arm}`" in text
    for pin in (
        "`shuffle_within_pair_v3`",
        "`rewire_checkerboard_v1`",
        "`FEAT_DIM = 11`",
        "`EDGE_TYPES = 4`",
        "`0.0698`",
        "`egostitch_e2e_probe_v2`",
        "`REQUIRED-BEFORE-BINDING`",
    ):
        assert pin in text


def test_v3_required_before_binding_markers_are_grep_discoverable() -> None:
    raw = REGISTRATION_PATH.read_text(encoding="utf-8")
    registration = json.loads(raw)
    marker_paths = set(_marker_paths(registration))

    assert raw.count("REQUIRED-BEFORE-BINDING") == len(marker_paths)
    assert marker_paths == {
        "$.prebinding_qualification.gates[3].threshold",
        "$.prebinding_qualification.gates[4].threshold",
        "$.artifact_versions.scores_npz_meta",
        "$.binding_evidence",
        "$.required_before_binding[0]",
        "$.required_before_binding[1]",
        "$.required_before_binding[2]",
    }


def test_v3_registration_arm_schema_matches_migrated_code() -> None:
    code_arms = tuple(g5_stage1._E2E_ARMS)
    if code_arms == V2_ARMS:
        pytest.skip(
            "Task 11 has not migrated g5_stage1._E2E_ARMS from the v2 five-arm schema"
        )

    registration = _registration()
    assert code_arms == EXPECTED_TRAINED_ARMS + EXPECTED_CONTROL_ARMS
    assert tuple(g5_stage1._E2E_FORMAL_ARMS) == EXPECTED_TRAINED_ARMS
    assert tuple(registration["arms"]) == code_arms

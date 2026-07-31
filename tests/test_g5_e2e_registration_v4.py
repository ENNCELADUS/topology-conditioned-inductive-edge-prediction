"""Contract tests for the EgoStitch rev-3.2 v4 draft registration.

v4 records the single-stage plan-bound design (spec §13.19, §14.4.6-7) and
re-pins the scientific reporting thresholds for the formal result. The tests below are
written against three distinct failure modes:

1. *Dead machinery.*  v3 declared a five-gate pre-binding ladder, a three-attempt
   window, a rehearsal budget and a sealed ``V_select`` -- all of which the code
   can no longer run.  A green suite that asserts those fields enforces dead
   machinery, so v4's tests assert their **absence**.
2. *Invented thresholds.*  Every formal-stage acceptance threshold is compared
   against the immutable BINDING v2 registration, field by field, so a number
   that drifted or was made up fails here rather than in a published table.
3. *Fail-open drafts.*  v4 replaces the retired unresolved-marker string
   convention with explicit JSON ``null``.  The tests prove that a v4 payload
   flipped to ``BINDING`` is still refused by the real validators, so the
   convention change did not weaken the boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Never, cast

import pytest
from src import score_universe, train_egostitch
from src.experiments import g5_stage1, probes
from src.model.egostitch.config import E2EConfig

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRATION_PATH = REPO_ROOT / "docs/registrations/g5_e2e_stage1_preregistration_v4.json"
MARKDOWN_PATH = REPO_ROOT / "docs/registrations/g5_e2e_stage1_preregistration_v4.md"
V2_REGISTRATION_PATH = REPO_ROOT / "docs/registrations/g5_e2e_stage1_preregistration_v2.json"
V3_REGISTRATION_PATH = REPO_ROOT / "docs/registrations/g5_e2e_stage1_preregistration_v3.json"
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
# Assembled at import time so this file can assert the marker's absence without
# containing it, which would make the assertion vacuously self-defeating.
RETIRED_MARKER = "REQUIRED-" + "BEFORE-BINDING"


def _registration() -> dict[str, object]:
    snapshot = train_egostitch._preregistration_snapshot(REGISTRATION_PATH)
    return snapshot.payload


def _v2_registration() -> dict[str, object]:
    return cast(dict[str, object], json.loads(V2_REGISTRATION_PATH.read_text(encoding="utf-8")))


def _mapping(value: object) -> dict[str, object]:
    """Narrow one registration value to a mapping, failing the test if it is not."""
    assert isinstance(value, Mapping), f"expected a JSON object, got {type(value).__name__}"
    return cast(dict[str, object], value)


def _as_int(value: object) -> int:
    """Narrow one registration count to an int, failing the test if it is not."""
    assert isinstance(value, int) and not isinstance(value, bool), (
        f"expected an integer count, got {value!r}"
    )
    return value


def _section(key: str) -> dict[str, object]:
    """Return one top-level v4 section, narrowed."""
    return _mapping(_registration()[key])


def _v2_section(key: str) -> dict[str, object]:
    return _mapping(_v2_registration()[key])


def _marker_paths(value: object, path: str = "$") -> Iterator[str]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield from _marker_paths(nested, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            yield from _marker_paths(nested, f"{path}[{index}]")
    elif isinstance(value, str) and RETIRED_MARKER in value:
        yield path


# --------------------------------------------------------------------------- draft identity


def test_v4_registration_parses_as_a_nonbinding_draft() -> None:
    registration = _registration()

    assert registration["status"] == "DRAFT"
    assert registration["status"] != "BINDING"
    assert registration["bound_utc"] is None
    predecessor = registration["predecessor"]
    assert isinstance(predecessor, dict)
    assert predecessor["path"] == ("docs/registrations/g5_e2e_stage1_preregistration_v2.json")
    assert predecessor["status"] == "BINDING"
    assert hashlib.sha256(V2_REGISTRATION_PATH.read_bytes()).hexdigest() == predecessor["sha256"]


def test_v3_is_retained_on_disk_and_named_as_the_superseded_draft() -> None:
    """A published registration is history; v4 supersedes it without deleting it."""
    assert V3_REGISTRATION_PATH.is_file()
    superseded = _registration()["superseded_draft"]
    assert isinstance(superseded, dict)
    assert superseded["path"] == ("docs/registrations/g5_e2e_stage1_preregistration_v3.json")
    assert superseded["status"] == "DRAFT"
    assert hashlib.sha256(V3_REGISTRATION_PATH.read_bytes()).hexdigest() == superseded["sha256"]


def test_design_trail_digest_resolves() -> None:
    trail = _registration()["design_trail"]
    assert isinstance(trail, dict)
    path = REPO_ROOT / str(trail["path"])
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == trail["sha256"]


# --------------------------------------------------------------------------- deleted machinery


def test_v4_declares_no_prebinding_ladder() -> None:
    """v3's five-gate ladder, attempt window and rehearsal budget are gone.

    Asserting their presence is what made the green suite enforce machinery the
    code can no longer run, so the assertion is inverted rather than deleted.
    """
    registration = _registration()

    assert "prebinding_qualification" not in registration
    assert "required_before_binding" not in registration

    # `superseded_draft.what_v4_removes` is the change record and necessarily
    # names the retired fields; every *live* section is searched instead, so a
    # re-added gate anywhere else still fails here.
    live = {key: value for key, value in registration.items() if key != "superseded_draft"}
    raw = json.dumps(live)
    for retired in (
        "max_attempts_total",
        "v_qual_rehearsals",
        "sealed until first bound run",
        "threshold_freeze",
        "fourth_attempt",
        "prebinding_scopes",
        '"G3.1"',
        '"G3.2"',
        '"G3.3"',
        '"G3.4"',
        '"G3.5"',
    ):
        assert retired not in raw, f"v4 still declares retired machinery: {retired}"


def test_v4_carries_no_unresolved_marker_strings() -> None:
    raw = REGISTRATION_PATH.read_text(encoding="utf-8")
    registration = json.loads(raw)

    assert raw.count(RETIRED_MARKER) == 0
    assert set(_marker_paths(registration)) == set()
    assert score_universe._contains_required_marker(registration) is False
    assert g5_stage1._contains_required_before_binding(registration) is False


def test_v4_pins_no_prebinding_threshold_on_the_measured_pool_ceilings() -> None:
    """The Phase-0 ceilings survive as measurements; the gate derived from them does not."""
    grounding = _registration()["grounding"]
    assert isinstance(grounding, dict)
    assert grounding["measured_slot_recall_ceilings"] == {
        "top50": 0.13952495387963418,
        "top20": 0.10728125418065595,
    }
    # 0.0698 was v3's G3.1 half-ceiling gate; nothing may re-admit it.
    assert "0.0698" not in REGISTRATION_PATH.read_text(encoding="utf-8")


def test_probe_scopes_are_formal_train_only() -> None:
    """v3 registered calibration/qualification probe scopes the code no longer has."""
    probe = _registration()["probe_artifact"]
    assert isinstance(probe, dict)
    assert probe["scope"] == "formal_train"
    assert probes.E2E_PROBE_SCOPES == ("formal_train",)
    assert probe["scope"] in probes.E2E_PROBE_SCOPES
    assert str(probe["verdict_effect"]).startswith("none")


# ------------------------------------------------------------------------ re-pinned v2 thresholds


def test_formal_acceptance_thresholds_are_copied_from_binding_v2() -> None:
    """Every formal-stage threshold must be v2's, field for field.

    Protocol §5.2.4 requires the decision rules recorded before any held-out
    read.  Recording *new* numbers at that moment would be indistinguishable
    from recording them after the fact, so v4 re-pins and this test proves it.
    """
    registration = _registration()
    v2 = _v2_registration()

    for field in (
        "primary_criteria",
        "guards",
        "operating_point",
        "comparators",
        "comparator_note",
        "evaluator",
        "failure_reading",
    ):
        assert registration[field] == v2[field], f"{field} drifted from BINDING v2"


def test_repinned_guard_inequalities_name_the_constants_the_gate_applies() -> None:
    assert _registration()["guards"] == [
        "full-arm degree-MMD ratio <= 1.10 times B0",
        "full-arm degree-corrected candidate AUPRC >= B0 - 0.02",
    ]


def test_pathway_and_liveness_rules_are_copied_from_v2() -> None:
    rules = _registration()["e2e_rules"]
    v2_rules = _v2_registration()["e2e_rules"]
    assert isinstance(rules, dict)
    assert isinstance(v2_rules, dict)

    for field in (
        "pathway_attribution",
        "liveness_within_checkpoint",
        "four_logit_decomposition",
    ):
        assert rules[field] == v2_rules[field], f"e2e_rules.{field} drifted from v2"


def test_structure_control_rule_is_v2s_with_only_the_rename_and_alpha_spelled_out() -> None:
    control = _mapping(_section("e2e_rules")["structure_control_condition"])
    v2_control = _mapping(_v2_section("e2e_rules")["structure_control_condition"])

    expected_rule = (
        str(v2_control["rule"])
        .replace("seed=0;", "seed=0, alpha=0.05;")
        .replace("structure_control_6a)", "structure_control_6a_v3)")
    )
    assert control["rule"] == expected_rule
    assert control["scope"] == v2_control["scope"]
    assert control["reading_on_fail"] == v2_control["reading_on_fail"]


def test_the_second_structure_control_is_reported_without_a_verdict_inequality() -> None:
    """6e-v1 is registered evidence, not a new decision rule.

    v2 registered exactly one structure-control condition and the gate computes
    exactly one bootstrap lower bound.  Inventing a threshold for 6e-v1 would be
    a new pre-registered decision rule -- an owner action, not a re-pin.
    """
    control = _mapping(_section("e2e_rules")["structure_control_condition"])
    assert control["applies_to"] == "structure_control_6a_v3"
    assert "no verdict" in str(control["structure_control_6e_v1"]).lower()
    assert tuple(g5_stage1._E2E_CONTROL_ARMS) == EXPECTED_CONTROL_ARMS


# --------------------------------------------------------------------------- evidence class


def test_evidence_class_is_engineering_at_every_seed_count() -> None:
    registration = _registration()
    assert registration["evidence_class"] == "engineering"
    rule = registration["evidence_class_rule"]
    assert isinstance(rule, dict)
    assert rule["value_at_every_seed_count"] == "engineering"
    assert rule["p_value"] is None
    assert rule["ci"] is None
    assert rule["holm"] is None
    assert g5_stage1._EVIDENCE_CLASS == "engineering"
    assert '"inference"' not in REGISTRATION_PATH.read_text(encoding="utf-8")


def test_registered_seeds_are_accepted_by_the_gates_seed_resolver() -> None:
    assert g5_stage1._registered_training_seeds(_registration()) == (0,)


# --------------------------------------------------------------------------- data contract


def test_v_fit_pins_are_unchanged_from_the_binding_v2_fit_manifest() -> None:
    """V_fit must not move: every feature-stats, grounding and pack digest keys on it."""
    fit = _mapping(_section("data_contract")["fit_manifest"])
    v2_contract = _mapping(_v2_section("checkpoint_selection")["validation_contract"])
    v2_fit = _mapping(v2_contract["fit_manifest"])

    for field in (
        "v_fit_node_count",
        "e_msg_fit_count",
        "e_sup_fit_count",
        "v_fit_nodes_sha256",
        "e_msg_fit_sha256",
        "e_sup_fit_sha256",
    ):
        assert fit[field] == v2_fit[field], f"V_fit {field} moved: blocking regression"


def test_v_hold_is_arithmetically_the_union_of_the_two_v2_holdouts() -> None:
    """1,533 = 456 + 807 + 270, over C(512, 2) = 130,816.

    Each right-hand term is read from the BINDING v2 registration, so the union
    is proved rather than asserted from a remembered number.
    """
    contract = _section("data_contract")
    hold = _mapping(contract["hold_manifest"])
    quarantine = _mapping(contract["quarantined_cross_partition_counts"])
    v2_contract = _mapping(_v2_section("checkpoint_selection")["validation_contract"])
    v2_qual = _mapping(v2_contract["qualification_manifest"])
    v2_select = _mapping(v2_contract["checkpoint_manifest"])
    v2_quarantine = _mapping(
        _mapping(v2_contract["fit_manifest"])["quarantined_cross_bucket_counts"]
    )
    v2_message = _mapping(v2_quarantine["message"])
    v2_supervision = _mapping(v2_quarantine["supervision"])

    node_count = _as_int(v2_qual["v_qual_node_count"]) + _as_int(v2_select["v_select_node_count"])
    assert hold["v_hold_node_count"] == node_count == 512
    assert hold["complete_nonself_pair_count"] == math.comb(node_count, 2) == 130816

    cross_side = _as_int(v2_message["qual__select"])
    expected_positives = (
        _as_int(v2_qual["positive_count"]) + _as_int(v2_select["positive_count"]) + cross_side
    )
    assert hold["positive_count"] == expected_positives == 1533
    assert hold["e_msg_hold_count"] == expected_positives
    assert hold["prevalence"] == expected_positives / math.comb(node_count, 2)

    # The two former fit-side quarantine buckets merge into one; nothing is lost.
    assert quarantine["message"] == {
        "fit__hold": _as_int(v2_message["fit__qual"]) + _as_int(v2_message["fit__select"])
    }
    assert quarantine["supervision"] == {
        "fit__hold": _as_int(v2_supervision["fit__qual"]) + _as_int(v2_supervision["fit__select"])
    }


def test_v_hold_digests_are_pinned_and_proved_disjoint_from_v_fit() -> None:
    contract = _registration()["data_contract"]
    assert isinstance(contract, dict)
    hold = contract["hold_manifest"]
    assert isinstance(hold, dict)
    for field in ("nodes_sha256", "positive_edges_sha256", "pair_labels_sha256"):
        digest = hold[field]
        assert isinstance(digest, str)
        assert len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest)
    assert hold["gold_graph_source"] == "E_msg[V_hold] only"
    assert contract["zero_overlap_proof"] == {
        "node": {"fit__hold": 0},
        "label_edge": {"fit__hold": 0},
    }
    assert contract["forbidden_gold_sources"] == [
        "train_graph.pkl",
        "val_edges.txt",
        "E_sup",
    ]


def test_selection_band_is_a_fixed_plan_value_without_calibration_artifact() -> None:
    selection = _section("checkpoint_selection")
    assert selection["auprc_tolerance"] == 0.02
    assert "preliminary run or calibration artifact" in str(
        selection["auprc_tolerance_rule"]
    )
    assert "auprc_tolerance_calibration_method" not in selection


def test_single_stage_plan_has_no_qualification_contract() -> None:
    registration = _registration()
    assert "ladder" not in registration
    assert "qualification_stage" not in _mapping(registration["test_protocol"])
    assert "qualification_worker" not in _mapping(registration["mechanics"])
    assert train_egostitch.E2ERunKind.__args__ == ("formal", "debug")


def test_quality_predicates_are_telemetry_only() -> None:
    selection = _section("checkpoint_selection")
    policy = str(selection["eligibility_stage_policy"])
    assert "telemetry-only" in policy
    assert "cannot prevent selection, completion, publication, scoring, or evaluation" in policy
    assert "all completed epochs" in str(selection["selection"])
    assert selection["no_eligible_checkpoint"].startswith("not a blocking state")



def test_v4_registration_freezes_the_rev32_contract() -> None:
    registration = _registration()
    arms = registration["arms"]
    assert isinstance(arms, dict)
    assert tuple(arms) == EXPECTED_TRAINED_ARMS + EXPECTED_CONTROL_ARMS

    reconstruction = registration["training_contract"]
    assert isinstance(reconstruction, dict)
    l_recon = reconstruction["l_recon"]
    assert isinstance(l_recon, dict)
    assert l_recon["revision"] == "rev-3.2"
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

    assert registration["registered_constants"] == {
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
    assert conditioning["equation"] == ("cls <- cls + active * tanh(g) * (XAttn(...) - mu)")
    assert conditioning["ema_decay"] == 0.99
    assert "single synchronized EMA" in conditioning["mu_eval"]

    standardization = reconstruction["feature_standardization"]
    assert standardization["method_id"] == "zscore_vfit_v1"
    assert standardization["feature_stats_sha256"] is None

    degree_prior = _mapping(reconstruction["degree_prior_init"])
    assert "mean(log d) over G_fit" in str(degree_prior["rule"])
    assert "sorted order" in str(degree_prior["rule"])
    assert degree_prior["value"] is None

    grounding = _mapping(registration["grounding"])
    assert grounding["method_id"] == "cosine_topk_v1"
    assert grounding["n_ground"] == 50
    assert grounding["reranker"] is None
    assert grounding["role_universes"] == ["V_fit", "V_hold", "test"]
    assert _mapping(grounding["pool_method_hash"])["ordered_fields"] == [
        "method id",
        "n_ground",
        "shortlist M when present",
        "ordered F0/source-feature-pack digest",
        "role-universe identity",
    ]

    assert registration["scaffold"] == {
        "feat_dim": 11,
        "edge_types": 4,
        "layout": ["onehot4(anchor)", "pi", "mult", "deg x 4", "t_k"],
    }

    control_6e = arms["structure_control_6e_v1"]
    assert (
        arms["structure_control_6a_v3"]["scoring_provenance"]["scaffold_control"]
        == score_universe._SCAFFOLD_CONTROL_SHUFFLE_V3
    )
    assert control_6e["scoring_provenance"]["scaffold_control"] == (
        score_universe._SCAFFOLD_CONTROL_REWIRE_V1
    )
    algorithm_6e = control_6e["algorithm"]
    assert algorithm_6e["transfer"] == ("delta = u * min(w_il, w_kj, c_ij - w_ij, c_kl - w_kl)")
    assert algorithm_6e["slot_adjacency_recipient_capacity"] == "c_ij = pi_i * pi_j"
    assert algorithm_6e["plan_recipient_capacity"].startswith("infinity")
    assert "pi-weighted" in algorithm_6e["slot_adjacency_swap_space"]
    assert algorithm_6e["deliberately_not_invariant"].startswith("rebuilt CLOSE degree channel")

    collapse = registration["collapse_abort"]
    assert isinstance(collapse, dict)
    assert collapse["mean_pairwise_h_cosine"] == {"operator": ">", "threshold": 0.95}
    assert collapse["pi_rank1_marginal_residual"]["threshold"] == 0.05
    assert collapse["consecutive_validations"] == 2
    assert collapse["pi_row_entropy_is_not_a_collapse_gate"] is True

    probe = _mapping(registration["probe_artifact"])
    assert probe["format"] == "egostitch_e2e_probe_v2"
    pi_v2_probe = (
        "Pi-consistency v2: plan mass on double-Hungarian same-identity cells "
        "divided by total plan mass"
    )
    registered_probes = probe["probes"]
    assert isinstance(registered_probes, list)
    assert pi_v2_probe in registered_probes
    assert "slot recall@n_g per run" in registered_probes

    versions = _mapping(registration["artifact_versions"])
    assert versions["scores_npz_meta"] == score_universe._SCORES_META_VERSION
    assert versions["pair_precision_contract"] == (
        score_universe._EGOSTITCH_E2E_PAIR_PRECISION_CONTRACT
    )


def test_v4_registration_arm_schema_matches_the_migrated_code() -> None:
    registration = _registration()
    assert tuple(g5_stage1._E2E_ARMS) == EXPECTED_TRAINED_ARMS + EXPECTED_CONTROL_ARMS
    assert tuple(g5_stage1._E2E_FORMAL_ARMS) == EXPECTED_TRAINED_ARMS
    arms = _mapping(registration["arms"])
    assert tuple(arms) == tuple(g5_stage1._E2E_ARMS)

    for name in EXPECTED_TRAINED_ARMS:
        entry = _mapping(arms[name])
        assert entry["kind"] == "trained_checkpoint"
        assert isinstance(entry["training"], str)
        assert isinstance(entry["scoring_provenance"], dict)
    for name in EXPECTED_CONTROL_ARMS:
        entry = _mapping(arms[name])
        assert entry["kind"] == "scoring_time_control"
        assert entry["training"] is None
        assert entry["checkpoint_arm"] == "full"
        provenance = _mapping(entry["scoring_provenance"])
        assert provenance["seed"] == 0
        assert provenance["keying"] == "canonical_pair_v1"
        assert provenance["checkpoint_arm"] == "full"


def test_registered_trained_arm_provenance_matches_what_the_worker_writes() -> None:
    """Scoring compares run-metadata `scoring_semantics` to this dict for equality."""
    arms = _section("arms")
    for name in EXPECTED_TRAINED_ARMS:
        entry = _mapping(arms[name])
        cfg = train_egostitch.load_config(REPO_ROOT / str(entry["training"]))
        e2e_config = E2EConfig.from_mapping(cfg.model.config)
        assert train_egostitch._e2e_arm_name_from_config(e2e_config) == name
        assert entry["scoring_provenance"] == (
            train_egostitch._e2e_trained_scoring_semantics(e2e_config)
        )


# --------------------------------------------------------------------------- config digests


def test_the_six_v3_config_digests_are_repinned_as_they_now_stand() -> None:
    registration = _registration()
    evidence = registration["binding_evidence"]
    arms = registration["arms"]
    assert isinstance(evidence, dict)
    assert isinstance(arms, dict)
    configs = evidence["configs"]
    assert isinstance(configs, dict)
    assert set(configs) == set(g5_stage1._E2E_FORMAL_ARMS)

    for arm, entry in configs.items():
        assert set(entry) == {"path", "sha256"}, (
            f"{arm}: the binding validators require exactly path and sha256"
        )
        assert entry["path"] == arms[arm]["training"]
        path = REPO_ROOT / str(entry["path"])
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], (
            f"{arm}: config digest moved; re-pin it in registration v4"
        )


def test_frozen_inputs_carry_real_digests_and_resolve_where_present() -> None:
    frozen = _registration()["frozen_inputs"]
    assert isinstance(frozen, dict)
    assert set(frozen) >= {
        "b0_candidate_scores",
        "g1_results",
        "g3_results",
        "b0cal_results",
    }
    for name, entry in frozen.items():
        digest = entry["sha256"]
        assert isinstance(digest, str)
        assert len(digest) == 64, name
        path = REPO_ROOT / str(entry["path"])
        if path.is_file():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name


# --------------------------------------------------------------------------- fail-closed boundary


def test_a_v4_payload_flipped_to_binding_is_still_refused(tmp_path: Path) -> None:
    """Explicit nulls replaced the marker convention; they must fail just as hard.

    Exercised through the real validators, on a copy -- the governing DRAFT is
    never mutated and no unresolved section is filled.
    """
    bound_copy = {**_registration(), "status": "BINDING"}
    registration_path = tmp_path / "docs/registrations/registration.json"
    registration_path.parent.mkdir(parents=True)
    registration_path.write_text(json.dumps(bound_copy), encoding="utf-8")

    with pytest.raises(g5_stage1.PreregistrationMismatch):
        g5_stage1._validate_e2e_binding_evidence(bound_copy, registration_path)

    with pytest.raises(ValueError):
        score_universe._preflight_binding_artifacts_before_pair_access(registration_path)


def test_the_governing_draft_is_refused_before_any_pair_source_is_opened() -> None:
    with pytest.raises(ValueError, match="requires status 'BINDING'"):
        score_universe._preflight_binding_artifacts_before_pair_access(REGISTRATION_PATH)


def test_binding_prerequisites_name_every_unresolved_section() -> None:
    registration = _registration()
    evidence = registration["binding_evidence"]
    assert isinstance(evidence, dict)
    assert evidence["schema_version"] == train_egostitch._E2E_BINDING_SCHEMA
    unresolved = {key for key, value in evidence.items() if value is None}
    assert unresolved == {
        "implementation",
        "parameter_group_manifests",
        "packs_and_validation_manifests",
        "boundary_access_audit",
        "runtime_and_peak_memory",
        "checkpoint_policy_version",
    }
    prerequisites = registration["binding_prerequisites"]
    assert isinstance(prerequisites, list)
    joined = " ".join(str(item) for item in prerequisites)
    for section in unresolved:
        assert section in joined, f"binding_prerequisites does not name {section}"
    assert "owner" in joined.lower()


# --------------------------------------------------------------------------- test protocol


def test_test_protocol_registers_the_ledger_and_refuses_the_once_only_claim() -> None:
    protocol = _registration()["test_protocol"]
    assert isinstance(protocol, dict)
    formal = protocol["formal_stage"]
    assert isinstance(formal, dict)
    assert formal["enforceable_property"] == (
        "one scoring epoch per (arm, seed); no re-scoring after seeing results"
    )
    assert "exactly once" in str(formal["not_claimed"])
    assert "append-only" in str(formal["ledger"])
    assert "never on test" in str(formal["selection_boundary"])
    assert "qualification_stage" not in protocol


def test_selection_exposure_is_disclosed_with_the_k_mitigation() -> None:
    disclosure = _registration()["selection_exposure_disclosure"]
    assert isinstance(disclosure, dict)
    assert "Repeated formal-plan development" in str(disclosure["hazard"])
    assert "record K" in str(disclosure["mitigation"])


def test_metrics_block_keeps_the_paired_reporting_and_naming_rules() -> None:
    metrics = _registration()["metrics"]
    assert isinstance(metrics, dict)
    assert "always reported together" in str(metrics["reporting_rule"])
    assert "never aggregated" in str(metrics["naming_rule"])
    assert "global simple-edge RD" in metrics["assembled_graph"]
    assert "BFS-macro relative density (RD)" in metrics["assembled_graph"]


# --------------------------------------------------------------------------- probe producer


def test_v4_registration_formal_probe_path_is_producer_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = _registration()
    probe_artifact = registration["probe_artifact"]
    assert isinstance(probe_artifact, dict)
    expected_path = probe_artifact.get("expected_path")
    assert expected_path == ("outputs/egostitch_e2e_stage1_v3/full/probes/e2e_probe_v2.npz")

    # Exercise the draft exactly as it would behave after binding, without
    # mutating the governing registration or resolving any unresolved section.
    bound_copy = {**registration, "status": "BINDING"}
    registration_path = tmp_path / "docs/registrations/registration.json"
    registration_path.parent.mkdir(parents=True)
    registration_path.write_text(json.dumps(bound_copy), encoding="utf-8")
    registration_sha = hashlib.sha256(registration_path.read_bytes()).hexdigest()
    full_arm = _mapping(_mapping(registration["arms"])["full"])
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


@pytest.mark.parametrize(
    ("include_expected_path", "expected_path"),
    [
        pytest.param(False, None, id="absent"),
        pytest.param(True, None, id="null"),
        pytest.param(True, "", id="empty"),
    ],
)
def test_probe_producer_rejects_missing_expected_path(
    tmp_path: Path,
    include_expected_path: bool,
    expected_path: object,
) -> None:
    probe_artifact: dict[str, object] = {
        "format": "egostitch_e2e_probe_v2",
        "source_arm": "full",
    }
    if include_expected_path:
        probe_artifact["expected_path"] = expected_path
    registration_path = tmp_path / "docs/registrations/registration.json"
    registration_path.parent.mkdir(parents=True)
    registration_path.write_text(
        json.dumps(
            {
                "status": "BINDING",
                "probe_artifact": probe_artifact,
                "arms": {"full": {"training": str(tmp_path / "full.yaml")}},
            }
        ),
        encoding="utf-8",
    )
    metadata_path = tmp_path / "run_metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"probe_artifact\.expected_path.*formal_train",
    ):
        probes.produce_e2e_probe_artifact(
            checkpoint_path=tmp_path / "best.pt",
            run_metadata_path=metadata_path,
            preregistration_path=registration_path,
            data_root=tmp_path / "data",
            strategy="breadth_first",
            output_path=tmp_path / "probe.npz",
            scope="formal_train",
        )


@pytest.mark.parametrize("registered_training", [None, "", " "])
def test_probe_producer_rejects_invalid_registered_training(
    tmp_path: Path,
    registered_training: object,
) -> None:
    registration_path = tmp_path / "docs/registrations/registration.json"
    registration_path.parent.mkdir(parents=True)
    registration_path.write_text(
        json.dumps(
            {
                "status": "BINDING",
                "probe_artifact": {
                    "format": "egostitch_e2e_probe_v2",
                    "source_arm": "full",
                    "expected_path": str(tmp_path / "probe.npz"),
                },
                "arms": {"full": {"training": registered_training}},
            }
        ),
        encoding="utf-8",
    )
    metadata_path = tmp_path / "run_metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"arms\.full\.training.*formal_train",
    ):
        probes.produce_e2e_probe_artifact(
            checkpoint_path=tmp_path / "best.pt",
            run_metadata_path=metadata_path,
            preregistration_path=registration_path,
            data_root=tmp_path / "data",
            strategy="breadth_first",
            output_path=tmp_path / "probe.npz",
            scope="formal_train",
        )


@pytest.mark.parametrize("value", [None, "", " "])
def test_shared_registration_path_resolvers_reject_invalid_values(
    tmp_path: Path,
    value: object,
) -> None:
    with pytest.raises(
        g5_stage1.PreregistrationMismatch,
        match=r"probe_artifact\.expected_path.*E2E probe evaluation",
    ):
        g5_stage1._registered_path(
            tmp_path / "registration.json",
            value,
            key="probe_artifact.expected_path",
            scope="E2E probe evaluation",
        )
    with pytest.raises(
        ValueError,
        match=r"arms\.full\.training.*formal scoring provenance",
    ):
        score_universe._registered_path(
            tmp_path / "registration.json",
            value,
            key="arms.full.training",
            scope="formal scoring provenance",
        )


# --------------------------------------------------------------------------- markdown twin


def test_v4_markdown_declares_json_authority_and_matches_critical_pins() -> None:
    text = MARKDOWN_PATH.read_text(encoding="utf-8")

    assert "Status: `DRAFT`" in text
    assert "JSON twin is authoritative" in text
    for arm in EXPECTED_TRAINED_ARMS + EXPECTED_CONTROL_ARMS:
        assert f"`{arm}`" in text
    for pin in (
        "one formal training stage",
        "There is no qualification stage",
        "Eligibility, liveness, slot-collapse indicators",
        "exact artifact identities",
        "owner-bound",
    ):
        assert pin in text, f"markdown twin lost pin {pin}"

"""Registration identity and binding-evidence schema checks for EgoStitch E2E."""

from __future__ import annotations

from collections.abc import Mapping

BINDING_SCHEMA_V1 = "egostitch_e2e_binding_evidence_v1"
BINDING_SCHEMA_V2 = "egostitch_e2e_binding_evidence_v2"
ACTIVE_V4_REGISTRATION_ID = "g5-e2e-stage1-20260729-two-stage-ladder-screen-v4-draft"
HISTORICAL_V1_REGISTRATION_ID = (
    "g5-e2e-stage1-20260719-conditioned-encoder-stability-screen-v2"
)
ACTIVE_V4_ARMS = frozenset(
    {
        "full",
        "b0_e2e_f_only",
        "pair_topology",
        "p0",
        "cosine_pool",
        "no_l_rel",
        "structure_control_6a_v3",
        "structure_control_6e_v1",
    }
)
HISTORICAL_V1_ARMS = frozenset(
    {"full", "b0_e2e_f_only", "pair_topology", "p0", "structure_control_6a"}
)


def binding_schema_for_registration(registration: Mapping[str, object]) -> str:
    """Couple each supported registration identity and arm set to one evidence schema."""
    registration_id = registration.get("registration_id")
    arms = registration.get("arms")
    evidence = registration.get("binding_evidence")
    if not isinstance(arms, Mapping) or not isinstance(evidence, Mapping):
        raise ValueError("registration requires structured arms and binding_evidence")
    schema = evidence.get("schema_version")
    if registration_id == ACTIVE_V4_REGISTRATION_ID:
        expected_arms, expected_schema = ACTIVE_V4_ARMS, BINDING_SCHEMA_V2
    elif registration_id == HISTORICAL_V1_REGISTRATION_ID:
        expected_arms, expected_schema = HISTORICAL_V1_ARMS, BINDING_SCHEMA_V1
    else:
        raise ValueError(f"unsupported E2E registration identity: {registration_id!r}")
    if set(arms) != expected_arms:
        raise ValueError(
            f"registration {registration_id!r} has an incompatible arm identity schema"
        )
    if schema != expected_schema:
        raise ValueError(
            f"registration {registration_id!r} requires binding evidence {expected_schema}"
        )
    return expected_schema

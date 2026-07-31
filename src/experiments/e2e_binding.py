"""Plan identity checks for supported EgoStitch E2E registrations."""

from __future__ import annotations

from collections.abc import Mapping

PLAN_SCHEMA_V1 = "egostitch_e2e_plan_v1"
PLAN_SCHEMA_V2 = "egostitch_e2e_plan_v2"
ACTIVE_V4_REGISTRATION_ID = "g5-e2e-stage1-20260730-single-stage-plan-bound-screen-v4"
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


def plan_schema_for_registration(registration: Mapping[str, object]) -> str:
    """Couple each supported registration identity to its exact arm schema.

    Registration status and run-produced evidence are deliberately outside this
    identity check: neither exists to authorize execution.
    """
    registration_id = registration.get("registration_id")
    arms = registration.get("arms")
    if not isinstance(arms, Mapping):
        raise ValueError("registration requires structured arms")
    if registration_id == ACTIVE_V4_REGISTRATION_ID:
        expected_arms, schema = ACTIVE_V4_ARMS, PLAN_SCHEMA_V2
    elif registration_id == HISTORICAL_V1_REGISTRATION_ID:
        expected_arms, schema = HISTORICAL_V1_ARMS, PLAN_SCHEMA_V1
    else:
        raise ValueError(f"unsupported E2E registration identity: {registration_id!r}")
    if set(arms) != expected_arms:
        raise ValueError(
            f"registration {registration_id!r} has an incompatible arm identity schema"
        )
    return schema

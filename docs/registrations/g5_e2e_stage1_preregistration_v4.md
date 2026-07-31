# G5 E2E Stage-1 rev-3.2 registration v4

Status: `DRAFT`. The JSON twin is authoritative. The owner must resolve the
remaining plan/artifact evidence and promote a successor content state to
`BINDING` before formal execution.

## Active execution contract

The experiment has one formal training stage. It trains the six registered arms
on `V_fit`, selects checkpoints on `V_hold`, then scores the two registered
scoring-time controls. There is no qualification stage, qualification artifact,
pass/pending verdict, attempt-history disclosure, calibration prerequisite, or
preliminary-run authorization path.

Formal launch, scoring, and evaluation verify only the owner-bound experiment
plan and exact artifact identities:

- registration ID, status, and SHA-256;
- clean implementation commit and exact arm/config path plus SHA-256;
- parameter-group, pack, validation-manifest, boundary-audit, runtime, and
  checkpoint-policy evidence;
- formal run, checkpoint ID/SHA-256, score-artifact, candidate-manifest, and
  test-access-ledger provenance.

No agent, note, metric threshold, or automated check may promote this draft to
`BINDING` or substitute for the owner's decision.

## Registered arms

Trained checkpoints: `full`, `b0_e2e_f_only`, `pair_topology`, `p0`,
`cosine_pool`, and `no_l_rel`.

Scoring-time controls over the `full` checkpoint:
`structure_control_6a_v3` and `structure_control_6e_v1`.

The exact configs and their SHA-256 digests live in
`binding_evidence.configs` in the JSON twin. Any config byte change invalidates
that identity and must be re-pinned before binding.

## Checkpoint selection and quality telemetry

Checkpoint selection applies the registered validation AUPRC selection band,
clustering-MMD tie break, Brier tie break, and earliest-epoch tie break to all
completed epochs. The shared AUPRC selection band is the fixed plan value
`0.02`; it has no qualification or calibration source artifact.

Eligibility, liveness, slot-collapse indicators, clipping/family/submodule-RMS
margins, AUPRC floors, dispersion, and precision differentials are telemetry
only. A miss may be reported but may not stop training, suppress checkpoint
publication, prevent scoring/evaluation, or authorize/deny execution.

Truthfulness and artifact-integrity failures still fail closed: non-finite
tensors or optimizer state, DDP disagreement, incomplete or duplicate coverage,
data-boundary violations, malformed or hash-mismatched plan inputs/outputs,
checkpoint/score provenance mismatch, and I/O or infrastructure failure.

## Data and access boundaries

`V_fit` is the training universe. `V_hold := V_qual ∪ V_select` is the sole
validation/model-selection universe. Candidate/test pairs and `test_graph.pkl`
remain unread during training. The formal result records the complete `V_hold`
evaluation ledger per arm and the append-only test-access ledger.

## Evidence class and reporting

This Stage-1 screen is engineering evidence. It cannot support significance or
cross-seed robustness claims. Edge-level and assembled-graph metrics are always
reported together. Scientific acceptance criteria may yield a post-run
`pass`/`cut` result, but that result is analysis output—not an execution gate.

## Unresolved binding evidence

The following JSON fields remain unresolved and must be supplied with exact
path-bound SHA-256 evidence where applicable:

- `binding_evidence.implementation`
- `binding_evidence.parameter_group_manifests`
- `binding_evidence.packs_and_validation_manifests`
- `binding_evidence.boundary_access_audit`
- `binding_evidence.runtime_and_peak_memory`
- `binding_evidence.checkpoint_policy_version`

This file authorizes no execution while the JSON twin remains `DRAFT`.

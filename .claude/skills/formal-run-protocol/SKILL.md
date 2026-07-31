---
name: formal-run-protocol
description: Use before fixing a registration snapshot, launching a formal (non-debug) run, or writing up any result. Covers exact plan/artifact identity, descriptive registration status, debug boundaries, and what a one-seed screen may claim.
---

# Formal run protocol

This repo runs plan-bound experiments. A result is worthless — worse,
misleading — unless the exact experiment plan is captured before training and
carried through every artifact by byte identity and SHA-256. Registration status
is descriptive provenance, not an authorization state machine. Model-quality
telemetry is deliberately not an authorization mechanism.

## Plan identity and registration status

- Registrations live in `docs/registrations/*.json`. The `.md` twin is
  explanatory only — **if they disagree, the JSON governs**.
- The training worker records the registration JSON's SHA-256 in
  `run_metadata.json` at run start. `_enforce_metadata_registration_hash`
  refuses to open any held-out metric
  when the hashes disagree. Spec §13.14, §13.18.
- `status` and `bound_utc` are descriptive only. `DRAFT`, `BINDING`, or another
  value neither authorizes nor blocks training, scoring, evaluation, or claims.
  Formal execution does not require resolved `binding_evidence` or run-produced
  evidence placeholders. Spec §12 (2026-07-30), §13.19.4, §14.4.
- A historical `BINDING` registration remains immutable, and every run's exact
  registration snapshot is immutable within that run/artifact chain. Changing
  any plan-defining optimizer, schedule, precision, candidate grid, frozen input,
  arm config, or registration byte is a **scientific change** requiring a new
  versioned snapshot, never a retroactive rewrite. Spec §13.15, §13.18, §13.19.2.
- The E2E path is a single formal stage: `hpc/qualification.sh formal <arm>`.
  There is no qualification stage, qualification artifact, verdict/history, or
  qualification-to-formal authorization path.
- Formal preflight requires the unchanged registration snapshot/SHA-256, exact
  registered arm/config path and digest, a clean implementation checkout, the
  correct repository/runtime boundary, and exactly four visible NVIDIA H20s.
- Parameter-group manifests, pack/validation manifests, boundary audits,
  runtime/peak-memory evidence, and checkpoint-policy provenance are produced by
  the actual run and verified downstream against its artifacts. Their absence
  before the run cannot block execution.

## Formal vs debug

- A successful formal run publishes its plan-bound artifacts and checkpoint.
  Checkpoint ranking considers all completed epochs under the registered
  selection rule; quality predicates are retained as telemetry only.
- `selected_checkpoint_eligible`, liveness, collapse, gradient, and AUPRC/MMD
  fields may be reported but cannot prevent selection, completion, publication,
  scoring, or evaluation.
- Non-finite values may not be serialized away in a manner that permits success.
- `--max-steps` runs are debug-only and are redirected to `*_debug` directories
  and are forbidden from candidate/test scoring. Debug is an execution boundary,
  not a quality verdict.

## Claim rules

- **Never claim statistical significance or cross-seed robustness from a G5
  Stage-1 screen.** It is a fixed-Seed-0 engineering screen: p-values, CIs, and
  Holm decisions must be emitted as `null`/not-applicable. Only E1/E3, with ≥3
  seeds plus Holm, carry inferential claims. Spec §13.15; protocol §5.2.3.
  **The E2E screen emits `evidence_class: engineering` at every seed count** —
  extra seeds buy cross-seed *variance reporting*, never significance, because
  inference additionally needs the spec §8 30-config HPO-parity budget and Holm over
  the pre-registered held-out assembled family. Protocol §5.0.5, §5.2.4.
- **Never headline one metric family alone.** Every claim reports edge-level
  *and* assembled-graph metrics together, with the held-out family headlined and
  noise-floor / ceiling / Oracle reference rows attached. Methodology §6.5;
  protocol §1/§4; blueprint §10.5.
- **Never call an MMD composite "graph similarity", and never aggregate the three
  MMD ratios.** GS and RD are independent official-evaluator metrics. Global
  simple-edge RD and BFS-macro RD must be named separately in every table.
  Protocol §1; spec §10.3.
- **Terminology guardrail** (`docs/lit-review-plan.md` §5, binding for all
  writing): generated local topology is always *intermediate context*, never the
  final output. The task is always binary edge prediction for queried pairs. If a
  draft starts describing graph generation as the task, it has drifted.
- **Anti-grab-bag rule**: a mechanism stays in the model only if it owns a row in
  the §4.6 mechanism-to-failure-axis map *and* an ablation arm. A mechanism
  owning no gain is cut. Proposal §4.6; protocol §3 E4.

## Who decides

Dispositions are **owner-side locked-decision discussions** — not decided by a
screen, a result note, or any agent. Blueprint §10 locked decisions must be
flagged, never renegotiated unilaterally.

## Spec freeze rule

`docs/05-egostitch-spec.md` was signed off at G4 (2026-07-09). Implementation may
not silently deviate: **edit the spec first**, with a one-line rationale in §12
(change log), then the code. A spec rewrite authorizes *implementation*, not
*execution* (§14). Formal execution additionally requires the exact-plan
preflight above; it does not require `status: BINDING` or a qualification artifact.

`docs/superpowers/specs/` holds design decision-trails and proposals awaiting
owner sign-off. They are **not** contracts and edit nothing on their own.

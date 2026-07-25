---
name: formal-run-protocol
description: Use before binding a pre-registration, launching a formal (non-debug) run, interpreting a gate verdict, or writing up any result. Covers registration hashing, BINDING vs diagnostic-only artifacts, the spec freeze rule, and what claims a one-seed screen may make.
---

# Formal run protocol

This repo runs pre-registered experiments. A result is worthless — worse,
misleading — unless it was bound *before* training. These rules are enforced in
code, so violating them usually means a `PreregistrationMismatch`, not a wrong
number. The exceptions below are the ways to get a wrong number silently.

## Binding a run

- Registrations live in `docs/registrations/*.json`. The `.md` twin is
  explanatory only — **if they disagree, the JSON governs**.
- The training worker records the registration JSON's SHA-256 in
  `run_metadata.json` at run start. `src/experiments/g5_stage1.py:139`
  (`_enforce_metadata_registration_hash`) refuses to open any held-out metric
  when the hashes disagree. Spec §13.14, §13.18.
- A `BINDING` registration is **immutable**. Amending it means a new versioned
  file (v1→v2→v3, predecessor recorded). Nothing is ever rebound retroactively:
  an artifact produced under a superseded hash stays diagnostic-only forever,
  carrying its old hash in provenance. Spec §13.15, §13.18, §13.19.4.
- Changing *any* threshold, optimizer, schedule, precision, candidate grid,
  success inequality, or frozen input after binding is a **scientific change**
  requiring a new registration version — not a bugfix. Spec §13.19.2.
- Qualification evidence may tune **no** held-out or test quantity. Three
  rehearsal attempts max; a fourth requires a new registration version.
  Spec §13.19.4.

## Binding vs diagnostic-only

- An aborted or ineligible run publishes **nothing usable** — no `best.pt`, no
  `complete.json`, no candidate scores, no gate input. If no epoch is eligible
  the run is `invalid`; there is no fallback to epoch 1 or the least-bad
  checkpoint. Spec §13.19.2–§13.19.3.
- `last.pt` is diagnostic-only, under a failure directory. It never replaces a
  formal `best.pt` verdict, even when it scores better.
- Non-finite values may not be serialized away in a manner that permits success.
- `--max-steps` runs are debug-only and are redirected to `*_debug` directories
  (`src/e2_pipeline.py:861`). Debug artifacts are forbidden from candidate/test
  scoring, not merely from the final gate. Spec §13.18, §13.19.5.

## Claim rules

- **Never claim statistical significance or cross-seed robustness from a G5
  Stage-1 screen.** It is a fixed-Seed-0 engineering screen: p-values, CIs, and
  Holm decisions must be emitted as `null`/not-applicable. Only E1/E3, with ≥3
  seeds plus Holm, carry inferential claims. Spec §13.15; protocol §5.2.3.
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
screen, a result note, or any agent. The rev-3.0 build line's fate is explicitly
open (`docs/results/G5-e2e-stage1-seed0-20260724.md`, "Next step"). Blueprint §10
locked decisions must be flagged, never renegotiated unilaterally.

## Spec freeze rule

`docs/05-egostitch-spec.md` was signed off at G4 (2026-07-09). Implementation may
not silently deviate: **edit the spec first**, with a one-line rationale in §12
(change log), then the code. A spec rewrite authorizes *implementation*, not
*execution* (§14) — execution still needs a fresh BINDING registration.

`docs/superpowers/specs/` holds design decision-trails and proposals awaiting
owner sign-off. They are **not** contracts and edit nothing on their own.

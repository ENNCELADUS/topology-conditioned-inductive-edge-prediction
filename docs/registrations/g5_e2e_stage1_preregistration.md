# G5 E2E Stage-1 Screening Pre-registration (prose mirror)

**Registration ID:** `g5-e2e-stage1-20260717-conditioned-encoder-screen-v1`
**Status:** `DRAFT` — becomes `BINDING` only after the measured H20 cost profile is
inserted and the user signs off the complete document. Formal training and the gate
machine-reject anything except `BINDING` (spec §13.18).
**Bindings:** spec `docs/05-egostitch-spec.md` §5/§13 (2026-07-17 rewrite, incl.
§13.18) + §14; protocol `docs/03-experiment-protocol.md` §5.0.5, §5.2, E4.15–E4.17.
**Predecessor:** `g5-stage1-20260716-membership-normalized-screen-v2`
(sha `97e61a7d…`), binding verdict `cut`, result
`docs/results/G5-stage1-seed0-20260717.md`.

## Scope

Single-seed (Seed 0) engineering screen of the rev-3.0 E2E stitched-topology-
conditioned pair encoder on Benchmark-A (`breadth_first`). This screen supports no
statistical-significance or cross-seed-robustness claims; E1/E3 retain the
at-least-three-seed, Holm-corrected requirements. The paired-bootstrap condition
below quantifies evaluator sampling stability at the fixed seed — nothing more.

## Arms (five; four training runs + one scoring-time control)

| Arm | Training | Scored array |
|---|---|---|
| `full` | `configs/egostitch_e2e_breadth_first.yaml` | `logits` |
| `b0_e2e_f_only` | `…_f_only_breadth_first.yaml` (permanent `∅_all_head`) | `logits` |
| `pair_topology` | `…_pair_topology_breadth_first.yaml` (permanent `∅_content_head`, **separately trained — THE registered attribution arm**) | `logits` |
| `structure_control_6a` | none — full checkpoint scored under `shuffle_within_pair` (stable-hash keying `canonical_pair_v1`, control seed 0) | `logits` |
| `p0` | `…_p0_breadth_first.yaml` (`p_topo = p_cont = 0`) | `logits` |

The full checkpoint's eval-bypass four-logit decomposition (including its
`pair_topology` array) is a **nonbinding diagnostic**.

Every scored artifact must match the frozen candidate manifest pair and label
arrays row-for-row. Exact provenance is: full = control `none`, permanent null
`none`, primary `full`; 6a = `shuffle_within_pair`, seed 0,
`canonical_pair_v1`, permanent null `none`, primary `full`, full checkpoint;
f-only = control `none`, permanent null `all_head`, primary `f_logit`;
pair-topology = control `none`, permanent null `content_head`, primary
`pair_topology`; p0 = control `none`, permanent null `none`, primary `full`.

## Comparators and frozen inputs

Comparators: `b0`, `b0_cal_density`, `b0_cal_selfdensity`, `b0_cal_degseq`
(carried over; `b0_cal_selfdensity` is the empirically strongest arm of the
frozen-s0 screen and remains the bar). Frozen inputs (sha-pinned in the JSON): the
B0 candidate scores artifact (`c5873caa…`, checkpoint `e092537d8cf1e208`), the G1
results (`668129a7…`), the G3 results (`e7fbc8e4…`), and the exact candidate
manifest (`bd8015bd…`). The calibrated-comparator payload is expected at
`outputs/deliverables/b0_cal_20260714/b0cal_results.json`; its SHA remains
`REQUIRED-BEFORE-BINDING` because that local deliverable is absent. Binding and
formal gate evaluation fail closed until a real digest replaces the marker.
The formal evaluator seed is exactly 0.

## Operating points

- **Canonical:** density-matched threshold on non-self candidate rows against
  `target_edges = |E(strip_self_loops(test))|` (predecessor definition, unchanged).
- **Matched:** per comparator, the evaluated arm realizes the comparator's exact
  non-self edge quota by descending pass-1 score (predecessor rule, unchanged).

## Primary criteria (single-seed point estimates; all three must pass; arm = `full`)

1. **Clustering-MMD ratio** — strictly **lower** than every comparator at the
   **canonical** operating point.
2. **BFS-macro GS** — strictly **higher** than each comparator at that
   comparator's exact **matched-global-RD quota**.
3. **BFS-macro RD** — strictly **higher** than each comparator at that same
   matched quota.

No inferential acceptance procedure is used with one seed; p-values, CI flags, and
Holm decisions are reported as not applicable.

## Guards

1. **Degree-MMD non-regression:** full-arm degree-MMD ratio ≤ 1.10 × B0's
   (recomputed from the frozen candidate artifact).
2. **Matched edge AUPRC:** full-arm degree-corrected candidate AUPRC ≥ B0's − 0.02;
   the B0-e2e AUPRC is reported alongside as the matched (nonbinding) reference.

## E2E decision rules

- **Pathway attribution** (user-confirmed 2026-07-17): on clustering-MMD ratio at
  the canonical operating point, with `G_full = clu(b0_e2e_f_only) − clu(full)` and
  `G_pt = clu(b0_e2e_f_only) − clu(pair_topology)` (the **separately trained**
  arm): applies iff the primaries pass and `G_full > 0`; require
  `G_pt ≥ 0.25 × G_full`. On failure: the gain is content-side; the
  topology-representation claim is withdrawn from the headline.
- **Structure-control condition** (user-confirmed 2026-07-17, replaces the
  rejected fixed 0.005 margin): paired bootstrap over the 500 fixed evaluation
  subgraphs — shared resample indices across the two arms, B = 1000 replicates,
  bootstrap seed 0; require the 2.5th percentile of
  `clu(structure_control_6a) − clu(full)` to exceed 0. Scope: evaluator sampling
  stability at the fixed training seed; explicitly not cross-seed or inferential
  evidence. On failure: the shuffled relational connectivity was not necessary
  beyond the retained topology-derived node features (slot tokens keep π, m, soft
  degrees) — this does **not** prove the whole gain is non-topological; it bounds
  which part of the topology pathway is load-bearing.
- **Liveness (run validity, fails closed before held-out metrics):** reference =
  the within-checkpoint `f_logit`; death rule conjunctive:
  `std(full − f_logit)/std(f_logit) < 1e-05` AND `Spearman(full, f_logit) >
  0.9999` AND top-1% overlap `> 0.9999`.
- **Four-logit decomposition:** full / `f_logit` / pair+content / pair+topology
  published per scored pair for the full and p0 checkpoints, fp32, provenance
  `egostitch_e2e_pair_fp32_v1`, resolution guard per array — diagnostic only.
- **Checkpoint selection:** validation AUPRC primary; within tolerance 1e-4,
  prefer the larger within-checkpoint `std(full − f_logit)/std(f_logit)`
  on the first `max(1, ceil(0.01*N_val))` frozen validation-manifest rows
  (spec §13.8 as re-registered).

## Nonbinding diagnostics

Same-checkpoint decomposition deltas (`topology_delta`, `content_delta`,
bypass-vs-trained pair_topology divergence); linear probes on frozen STE token
states (R² to degree / ego-density / clustering; ridge λ = 1e-3, 5-fold; plus
degree-partialled variants and Π-consistency); gate tanh magnitudes and per-family
gradient norms; p0-vs-full four-logit divergence.

The gate requires a provenance-bound `egostitch_e2e_probe_v1` artifact produced
after scoring from the selected full checkpoint. The producer requires the
registered `arms.full.training` config with `permanent_null = none` and
`p_topo = p_cont = 0.15`, so the p0 checkpoint cannot be substituted. It uses
all operative train nodes in sorted order and the 4,096 hash-smallest non-self
`E_msg` pairs (or all when fewer), and reports degree / ego-density / clustering ridge R², the
degree-partialled density and clustering variants, and Π/shared-neighbor
consistency. This evidence is required output but remains nonbinding.

## Cost report

Measured H20 profile is **REQUIRED-BEFORE-BINDING**: per-step wall, peak memory,
30-epoch extrapolation per arm (×4), candidate scoring latency (×5 passes). The
frozen-s0 673 s / 2.04 GiB profile explicitly does not extrapolate.

## Verdict rule

Single-seed screening **pass** iff the full arm strictly dominates every comparator
on all three primary criteria at the registered operating points, both guards pass,
liveness passes, and both E2E decision conditions pass. Otherwise **cut**, with the
registered failure reading written into the gate results verbatim.

## Failure reading (registered verbatim)

If the full arm fails the primary family against the B0+cal comparator set, the
topology-conditioned encoder does not beat calibrated independent scoring at this
stage; the result is written up honestly and the locked-decision discussion is
taken before any Stage-2 mechanism or E1 registration. If the primaries pass but
pathway attribution fails, the honest conclusion is content-side information. If
the primaries pass but the structure-control condition fails, the honest
conclusion is that shuffled relational connectivity was not necessary beyond the
retained topology-derived node features — not that the gain is non-topological.

## Mechanics

- **Worker:** `src/train_egostitch.py` records this file's sha256 as
  `run_metadata.json['preregistration_sha256']` at start; refuses formal
  (non-`--max-steps`) runs unless `status == BINDING` and no
  `REQUIRED-BEFORE-BINDING` marker remains anywhere; `--max-steps` debug runs
  accept DRAFT/unresolved markers but write only to `*_debug` output directories
  and produce no held-out artifacts.
- **Gate:** `src/experiments/g5_stage1.py` requires `status == BINDING`,
  a real calibrated-comparator digest, evaluator seed 0, exact candidate and arm
  provenance, the probe artifact, and this file's sha256 matching **all four**
  formal run metadatas (full, f_only, pair_topology, p0).
- **Hyperparameters:** spec §13.18 defaults × fixed Seed 0; no sweeps inside this
  screen.
- **P4 exclusion:** no hard heuristic/feature negatives and no hard-negative
  checkpoint criterion are added in this revision.

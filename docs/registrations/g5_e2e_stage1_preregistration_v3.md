# G5 E2E Stage-1 relational-repair pre-registration v3 — DRAFT

**Status: DRAFT. This document does not authorize training, scoring, qualification,
or a formal run.**

The machine-readable registration is
`docs/registrations/g5_e2e_stage1_preregistration_v3.json`. This Markdown twin is
explanatory only. **If the two disagree, the JSON governs.**

The predecessor is the immutable BINDING v2 registration at
`docs/registrations/g5_e2e_stage1_preregistration_v2.json`. This draft neither edits
v2 nor relabels its completed rev-3.0 five-arm artifacts. The owner may bind v3 only
after V_fit-only calibration freezes every remaining threshold and one prospective
V_qual rehearsal passes within the three-attempt budget. V_select remains sealed
until the first bound run.

## Authority and evidence trail

The governing contract is `docs/05-egostitch-spec.md` §14.4 in full, including the
second through fifth §12 amendments. The design rationale is recorded in
`docs/superpowers/specs/2026-07-25-egostitch-e2e-relational-repair-design.md`.
Phase-0 evidence is under `outputs/p0_audit_20260725/`, specifically
`p0_audit_results.json` and `p0_autopsy_results.json`.

Generated and stitched topology remains intermediate context for binary edge
prediction on queried unseen-node pairs. It is not a graph-generation objective.
This remains a fixed-Seed-0 engineering screen, so it cannot support significance or
cross-seed robustness claims.

## Eight-arm schema

The six trained checkpoints are:

1. `full`
2. `b0_e2e_f_only`
3. `pair_topology`
4. `p0`
5. `cosine_pool`
6. `no_l_rel`

The two scoring-time controls reuse `full`'s selected checkpoint:

7. `structure_control_6a_v3`
8. `structure_control_6e_v1`

Task 11 supplies six new v3 config files. No v2 config may be changed because the
BINDING v2 registration pins their SHA-256 digests. All trained arms use
`n_ground = 50` except `cosine_pool`, which pins the status-quo cosine pool at 20.
The `no_l_rel` arm differs from `full` by setting the `L_rel` component weight to
zero.

## Rev-3.1 training contract

The outer objective remains:

`L = L_edge + lambda_real L_real + lambda_ssl L_ssl + lambda_recon L_recon`.

The ten `L_recon` weights are:

| Component | Weight |
|---|---:|
| `L_feat` | 1.0 |
| `L_exist` | 0.5 |
| `L_mult` | 0.25 |
| `L_deg` | 0.5 |
| `L_slotadj` | 0.5 |
| `L_gate` | 0.25 |
| `L_ptr` | 0.25 |
| `L_align` | 0.5 |
| `L_div` | 0.1 |
| `L_rel` | 0.25 |

Only `L_feat`, `L_exist`, `L_mult`, and `L_deg` anneal from factor 1.0 to
0.25 during the edge-active phase. The six repair components remain at factor 1.0,
and outer `lambda_recon` is unchanged.

Warm-start remains reconstruction-only. From the first edge-active step, the trunk,
STE, gates, and both conditioning pathways train jointly; v2's Phase-A `pair_only`
head start is removed. Centered injection is:

`cls <- cls + active * tanh(g) * (XAttn(...) - mu)`.

Training `mu` is the all-reduced mean over pathway-active real rows. At evaluation,
each conditioning module uses its single synchronized EMA stored in the checkpoint,
with decay 0.99; there is no rank-local or duplicate EMA for one module. Inactive
rows retain the exact-identity bypass.

The registered constants are `tau_adj = 0.5`, `l_gate_pos_weight = 6.17`,
`tau_div = 0.5`, ego-target cap `K = 16`, and conditioning-`mu` EMA decay `0.99`.

## Grounding and scaffold

The main rev-3.1 grounding method is exact `cosine_topk_v1`, `n_ground = 50`, with no
reranker and no shortlist `M`. Each cache pins a `pool_method_hash` over the ordered
method id, `n_ground`, optional shortlist `M` when present, ordered
F0/source-feature-pack digest, and role-universe identity. A mismatch fails closed.

The scaffold pins `FEAT_DIM = 11`, `EDGE_TYPES = 4`, and layout
`[onehot4(anchor); pi; mult; deg x 4; t_k]`.

## Scoring-time controls

`structure_control_6a_v3` applies canonical-pair-keyed within-pair slot-axis
permutations to `A_hat_src`, `A_hat_dst`, and `Pi` at scaffold-build input, then
rebuilds the full scaffold, including `t_k`, `CLOSE`, and the degree slice. Its
executable scorer provenance mode is `shuffle_within_pair_v3`.

`structure_control_6e_v1` performs canonical-pair-keyed checkerboard swaps in
pi-weighted slot-adjacency space and directly in `Pi`, then rebuilds the scaffold.
Its executable scorer provenance mode is `rewire_checkerboard_v1`.
Its transfer is
`delta = u * min(w_il, w_kj, c_ij - w_ij, c_kl - w_kl)`, with
`c_ij = pi_i pi_j` for pi-weighted slot adjacency. Direct `Pi` rewiring uses
infinite recipient capacity, so its transfer reduces to
`u * min(w_il, w_kj)`; `Pi` is not mapped back through the bounded `A_hat` sigmoid
domain. The rebuilt `STAR`, `INTRA`, and `ALIGN` degree channels are the binding
invariants; zero diagonals and mapped-back adjacency values in `[0, 1]` are
preserved. `CLOSE` is deliberately not invariant because closure mass is the
higher-order signal the control is meant to destroy.

## Probes and qualification

The probe artifact version is `egostitch_e2e_probe_v2`. The registered set includes
Pi-consistency v1 and v2, per-run slot recall at `n_ground`, shared-neighbor-count R2
from STE pair states, degree-partialled clustering R2, and the four slot-dispersion
statistics.

The five prospective G3 gates are:

1. for the qualifying `full` arm, slot recall at `n_ground = 50` at least
   `0.0698`, half the measured `0.13952495387963418` top-50 ceiling;
2. Pi-consistency v2 strictly greater than `0.05`;
3. degree-partialled clustering probe R2 at least `0.10`;
4. `structure_control_6a_v3` moves clustering-MMD beyond the evaluator bootstrap
   noise floor;
5. the matched edge-AUPRC guard passes.

The JSON deliberately leaves the fourth gate's noise floor and the fifth gate's
numeric guard threshold as `REQUIRED-BEFORE-BINDING`. They require GPU-dependent
V_fit calibration and must not be invented, estimated from v2, or filled using
V_qual. Calibration is exclusively on V_fit; then the implementation and thresholds
freeze before a single V_qual rehearsal. Any failed gate blocks binding.

The `cosine_pool` top-20 arm has a measured Phase-0 ceiling of
`0.10728125418065595`, but it is a formal-screen vocabulary attribution ablation,
not the pre-binding `full`-arm qualification rehearsal. The Task 11 scores-`.npz`
metadata version is likewise an explicit unresolved marker until its exact bumped
identifier exists in code; older versions must be rejected.

## Binding boundary

The JSON also leaves implementation/config hashes, manifests, calibration evidence,
the V_qual rehearsal record, access audit, runtime evidence, and checkpoint-policy
identity unresolved. The formal worker must reject this DRAFT and every unresolved
marker. Only the owner may promote a fully resolved successor content state to
BINDING.

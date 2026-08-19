# S4/S5/S6 — node-aligned degree budgets at full capacity: results

**Status: S4 ceiling re-established; S5/S6 probes implemented, runs pending.** The
B1 KD arms (D1–D7a) distill only row-normalized or pointwise projections of the
full-ego oracle teacher and moved GS-BFS by at most +0.0017. The accepted review
(`docs/tmp/kd_review.md`) locates the miss: the oracle's value is node-aligned,
realization-specific, jointly allocated topology, and pairwise KD structurally
discards the one feature-predictable slice — absolute node activity / degree
budgets. This route tests whether that slice is recoverable **at full model
capacity**, which the historical ridge-regression arm could not answer.

Implementation: `src/experiments/s4_budget_assembly.py` (score-time replay, no
training), `src/experiments/s5_degree_probe.py` (deep degree regressor),
`src/experiments/s6_residual_probe.py` (pair-residual identifiability probe). All
three are standalone diagnostics: `evidence_class=diagnostic`, no `e2_pipeline`
publish/test contract, no V_val ball-union topology validation — those are
meaningless for a regression probe and are not faked.

## S4 — hard degree-quota replay (pass A: control + oracle)

Run 2026-08-19 on the frozen B0 candidate universe
(`outputs/deliverables/b0_v31_breadth_first_20260711/scores/candidate.npz`,
checkpoint `e092537d8cf1e208`, 2,037,171 rows / 2,018 nodes), `--seed 0`,
`target_edges = 30,128`. Pair scores are untouched in every arm; only the
assembly rule changes. All arms share one frozen self-loop set computed at
`threshold_b0 = density_matched_threshold(probs[non_self], target_edges)`;
self-pairs are never routed into `assemble_degree_quota` (it raises on them).

The five topology numbers, reported together and never aggregated (GS↑, RD→1,
ratios↓). BFS-macro and global simple-edge GS/RD are named separately per the
claim rule.

| arm | GS-BFS | RD-BFS | MMD-degree | MMD-clustering | MMD-spectral |
|---|---|---|---|---|---|
| `b0_exact_n` (control) | 0.3898 | 0.4231 | 13.032 | 11.861 | 18.070 |
| `oracle_hard` | **0.4386** | **0.6220** | **4.064** | **5.442** | **7.650** |

Edge-level, reported alongside the graph family:

| arm | edge precision | edge recall | GS-global | RD-global |
|---|---|---|---|---|
| `b0_exact_n` | 0.1366 | 0.1366 | 0.1366 | 1.0000 |
| `oracle_hard` | 0.2159 | 0.2124 | 0.2142 | 0.9838 |

`oracle_hard` quota enforcement: 29,640 of 30,128 target edges realized,
shortfall 488 (**1.62%**, below the 2% `lower_bound_only` threshold, so the arm
is *not* flagged as a lower bound), residual quota 976, degree error
`exact_fraction` 0.9747 / `l1` 976 / `linf` 51.

**Ceiling stability — no drift.** These reproduce the historical S1-R numbers
(`outputs/s1/s1_results.json`, V_hold era) to floating-point tail digits: control
GS 0.38975252686048445 both times, oracle GS 0.4385802471670622 both times, and
every MMD ratio agreeing to ~1e-14. The quota-enforcement block is bit-identical
(29,640 / 488 / 976 / 0.9747 / 976.0 / 51.0). The V_hold→V_val migration did not
move this ceiling, and the S1 module's deletion cost nothing: the recovered
`largest_remainder_quotas` / `degree_quota_error` helpers now live in
`src/eval/assembly.py` and reproduce the deleted module exactly.

`s4_results.json` is byte-identical across reruns (verified by `cmp` on two
independent invocations); it carries no wall-clock fields.

**Reading.** With pair scores frozen, node-aligned *true* degree quotas alone lift
GS-BFS by +0.0488 and cut all three MMD ratios by ~2.4–3.2×, while RD-BFS moves
from 0.423 toward 1 (0.622). The headroom the KD arms failed to reach is real,
large, and reachable by assembly alone. What S4 cannot say is whether any
*feature-predicted* budget gets there — that is S5's question, and it enters this
table as `predicted_hard_<variant>` arms in pass B.

## S5 — deep degree regressor (pending run)

Implemented and unit-tested; the six jobs ({warm, scratch} × seeds 0/1/2) run on
the H20. Target convention `y = log1p(deg(strip_self_loops(train_graph.pkl)))`
over the full train⁺∪val⁺ substrate — the V_val quarantine is a pair-level rule,
not a node-degree rule. Held-out split is 10% of a seeded permutation and is a
*selection device only*: held-out and training nodes share graph edges, so their
degree targets are statistically coupled. That leakage is disclosed, not fixed.

Trunk architecture is read from the checkpoint's own `model_config`, not from
`configs/b0_v31_breadth_first.yaml`. The shipped config declares `d_model: 512`
while every published v3_1 checkpoint carries `d_model: 256`; a hardcoded 512
would make `--init warm` fail a strict load at runtime. Warm and scratch
therefore share an identical architecture, which is what makes scratch a control.

Decision readout, once run: held-out Spearman/R² against the ridge baseline, and
S4 `predicted_hard_*` GS against **0.399** (ridge) and **0.4386** (oracle), with
quota shortfall against the ridge arm's 8.43%. Deep ≈ ridge ⇒ node-aligned degree
is not in the features at any capacity, and the missing signal is
realization-specific — which supports the joint-allocation thesis. Deep ≫ ridge
with GS → oracle ⇒ a legal budget method exists.

## S6 — pair-residual identifiability probe (pending run)

Implemented and unit-tested; six jobs on the H20, requiring the v3 KD-targets
artifact (`content_logit` present — a v2 artifact raises). Target
`Δ = teacher_logit − content_logit`. The split is by **anchor**, never by row:
every CSR row of a held-out anchor goes to the eval side, and every training row
touching a held-out anchor at *either* endpoint is dropped (count disclosed as
`n_train_rows_dropped_touching_heldout`). The readout is `abba_max`-symmetric, so
without that second rule reciprocal rows `(u,v)` and `(v,u)` put the same
canonical pair on both sides — on a fully reciprocal fixture that leaked 10.3% of
training rows, which would inflate the very bound this probe exists to establish.

Zero-init control: the final `output_head` linear is zeroed so the first
prediction is exactly the predict-zero baseline. This required removing the
spectral-norm parametrization from that one linear first — the published
checkpoints carry `mlp_head.spectral_norm: True`, and zeroing a spectral-normed
weight yields `0 / 0 = NaN`, not zero (verified directly). The report records
whether the removal happened.

Decision readout, once run: held-out R²/Spearman/MAE against **both** the
predict-zero and predict-mean baselines. At full capacity with a warm start, this
upper-bounds every pointwise residual-KD arm including D5 — the claim-protection
result the review asked for. R² ≫ 0 ⇒ D5's failure was optimization, and the arm
is worth revisiting; R² ≈ 0 ⇒ Δ is not a function of `(x_u, x_v)` and no pointwise
arm can ever fit it.

## D8 — absolute row-mass distillation (pending run)

`kd_rowmass_loss` sums `σ(logit)` per anchor group for student and teacher and
Hubers the two masses. The discriminating property against `kd_dist_loss`: a
constant within-group logit shift leaves the temperature-KL invariant but moves
the row mass, so D8 is the first arm that sees absolute per-node activity rather
than a normalized row shape. Config `configs/b1_kd_d8_breadth_first.yaml` differs
from `b1_kd_d5_breadth_first.yaml` only in header comment, `output_dir`, and the
distill block (v2 targets — no `content_logit` needed — `w_rowmass: 1.0`,
`anchors_per_step: 2`, matching sibling anchor budget). Results land as one row
per table in `docs/results/b1_kd_arms.md`.

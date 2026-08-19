# S4/S5/S6 — node-aligned degree budgets at full capacity: results

**Verdict so far: negative for feature-predicted budgets, with capacity excluded.**
Deep full-capacity degree regressors match the historical ridge arm exactly and
leave RD and all three MMD ratios worse than the control (§ S4 pass B). S6 and D8
runs are still outstanding. The
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

## S5 — deep degree regressor (6 jobs, 2026-08-19)

Target convention `y = log1p(deg(strip_self_loops(train_graph.pkl)))` over the
full train⁺∪val⁺ substrate — the V_val quarantine is a pair-level rule, not a
node-degree rule. 8,070 feature-backed nodes → 7,263 train / 807 held out. The
held-out split is a *selection device only*: held-out and training nodes share
graph edges, so their degree targets are coupled. Disclosed, not fixed.

Trunk architecture is read from the checkpoint's own `model_config`. The shipped
`configs/b0_v31_breadth_first.yaml` declares `d_model: 512` while every published
v3_1 checkpoint carries `d_model: 256`; a hardcoded 512 fails `--init warm`'s
strict load at runtime. Warm and scratch share one architecture (40 encoder
tensors transferred for warm), which is what makes scratch a control.

| variant | held Spearman | held R² | held MAE | train Spearman | best epoch |
|---|---|---|---|---|---|
| warm_s0 | 0.4472 | 0.2442 | 0.7915 | 0.6586 | 28 |
| warm_s1 | 0.5170 | 0.3341 | 0.7362 | 0.6717 | 33 |
| warm_s2 | 0.4893 | 0.2926 | 0.7705 | 0.5894 | 19 |
| scratch_s0 | 0.4450 | 0.2159 | 0.8045 | 0.6910 | 22 |
| scratch_s1 | 0.4938 | 0.3008 | 0.7422 | 0.6463 | 21 |
| scratch_s2 | 0.4826 | 0.2504 | 0.7794 | 0.5931 | 14 |

warm 0.4845 ± 0.0287 Spearman / 0.2903 ± 0.0367 R²; scratch 0.4738 ± 0.0208 /
0.2557 ± 0.0348 (population sd over 3 seeds). Every job early-stopped.

**Warm ≈ scratch, inside one seed sd.** The B0 pretrained trunk buys essentially
nothing for degree prediction over a randomly initialized copy of the same
architecture, so whatever degree signal exists is in the raw features, not in the
representation B0 learned. Absolute level is modest: R² ≈ 0.25–0.29 on log1p
degree, with a train-side R² of 0.40–0.52 (gap ~0.2, early stop holding).

## S4 — pass B: feature-predicted budgets (2026-08-19)

The six S5 `predictions.json` files enter as `predicted_hard_<variant>` arms
against the same frozen control and oracle. Same universe, same self-loop policy.

| arm | GS-BFS | RD-BFS | MMD-deg | MMD-clu | MMD-spe | shortfall |
|---|---|---|---|---|---|---|
| `b0_exact_n` (control) | 0.3898 | 0.4231 | 13.032 | 11.861 | 18.070 | — |
| `oracle_hard` | **0.4386** | **0.6220** | **4.064** | **5.442** | **7.650** | 1.6% |
| `predicted_hard_warm_s0` | 0.4005 | 0.4017 | 15.486 | 14.434 | 19.775 | 12.1% |
| `predicted_hard_warm_s1` | 0.3967 | 0.4001 | 14.703 | 13.980 | 18.533 | 11.7% |
| `predicted_hard_warm_s2` | 0.3992 | 0.4019 | 15.041 | 14.146 | 18.771 | 10.9% |
| `predicted_hard_scratch_s0` | 0.3972 | 0.3986 | 15.333 | 14.458 | 19.250 | 11.8% |
| `predicted_hard_scratch_s1` | 0.3945 | 0.3974 | 15.348 | 14.613 | 19.157 | 14.8% |
| `predicted_hard_scratch_s2` | 0.3983 | 0.3904 | 17.048 | 15.340 | 21.413 | 15.1% |

Every predicted arm is flagged `lower_bound_only` (shortfall > 2%); the oracle is
not. Quota `l1` error: oracle 976, predicted 6,592–9,080.

Edge-level, alongside the graph family: control P/R 0.1366/0.1366, GS-global
0.1366; oracle 0.2159/0.2124, 0.2142; predicted 0.168–0.184 / 0.148–0.157,
GS-global 0.158–0.169.

**Verdict: negative, and capacity is excluded as the cause.** Full-capacity deep
regressors land at GS-BFS 0.3945–0.4005 (warm mean 0.3988, scratch mean 0.3967)
— indistinguishable from the historical **ridge** arm's 0.399, against an oracle
of 0.4386. Model class was never the binding constraint.

The five numbers must be read together, and they do not point the same way. GS-BFS
closes 10–22% of the control→oracle gap, but **RD-BFS and all three MMD ratios move
away from the oracle**: RD falls to 0.390–0.402 from the control's 0.423 (oracle
0.622), and every MMD ratio rises above the control (degree 14.7–17.0 vs 13.0).
Gap closure is negative on all four of those axes. A feature-predicted budget is
therefore not "partway to the oracle" — it buys a small edge-set gain while making
the degree, clustering and spectral distributions worse than doing nothing. The
predicted arms are also *less* satisfiable than ridge was (10.9–15.1% shortfall vs
8.43%), so their GS is a lower bound on a mis-specified quota set.

Read with the oracle's own result — true node-aligned quotas lift GS by +0.0488 and
cut every MMD ratio 2.4–3.2× — this says the recoverable slice of node-aligned
degree is not a function of `(x_u, x_v)` at any capacity. The oracle's advantage is
realization-specific, which is what the joint-allocation thesis predicts.

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

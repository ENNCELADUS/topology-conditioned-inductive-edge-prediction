# B1 — training-time KD arms: design and results

**Verdict: eight arms across five mechanism families, no arm passes its gate.** Listwise output KD
(D2) buys AUROC and calibration; relational KD (D1/D3) buys MMD ratios and RD-BFS; representation
alignment (D4) adds nothing over Gram; topology-residual distillation (D5) reproduces D3's topology
while wrecking calibration *despite* a large non-degenerate residual target; the D2+D3 combination
(D6) is not additive but collapses onto D2; a parameter-free heuristic teacher (D7a) takes the best
AUROC in the table and pays with the worst topology. Edge-set identity (GS) never moves. The KD
route transfers **distributional shape, not edge identity**, and the two metric families are in
genuine objective conflict.

Design provenance: `docs/tmp/b1_v3_kd_arms_plan.md` (2026-08-17), which diagnoses the v2 stage.
Implementation: `src/distill/` (`losses.py`, `config.py`, `teacher_targets.py`, `content_logit.py`,
`heuristic_targets.py`), `src/model/egostitch/classifier/`, commits `b56cb77` → `b43828e`.

## Shared setup (variable-control protocol)

Every arm is one `distill:` weight change over an otherwise byte-identical config. Student `v3_1`:
1536→512, 3 encoder + 3 cross-attention layers, 8 heads, rich pooling, `pair_context_gated` readout,
zero-init `NodeFactorBottleneck` with `node_factor_dim: 64`, label smoothing 0.05. Optim: 25 epochs,
lr 1e-4 one-cycle, weight decay 0.05, bf16, seed 0, breadth-first split. Checkpoint selection
mean-ranks V_val AUPRC plus all five bucket-topology metrics; the published epoch is recorded in the
v3 test reports. Evaluation is the held-out test protocol (64,038 rows, 32,019 ± pairs, 1,891
self-rows) with the density-matched assembled-graph threshold.

KD stream: `kd_targets_breadth_first_seed0_v2` — 74,692 pairs over 8,070 anchors, k = 5 near + 5
random, teacher = full-ego oracle checkpoint `c390709fb7070c62`
(`outputs/egostitch_e2e_stage1_v3/full_ego_teacher_kd/best.pt`). D5 and D7a swap the *values* on
those rows; the row identity never changes, which is what makes all eight arms comparable.

Two stages are reported: **v2** (`outputs/b1_stage_v2/`, 2026-08-16/17 — control, D1, D2, D3) and
**v3** (`outputs/b1_stage_v3/`, 2026-08-17/18 — D4, D5, D6, D7a), ~2.5 GPU-h each, serial on the
4-GPU H20 container; both tables re-verified against their `test_report.json` on 2026-08-18. An
earlier `outputs/b1_stage/` (2026-08-14) predates the V_val split and the v2 teacher stream and is
superseded, not quoted here.

## Arms — mechanism, source, and gate

### `kd_control` — matched control (no teacher signal)
`w_label: 1.0`; `kd_label_loss` = BCE against the KD stream's own `pair_label`. Same stream, same
anchors-per-step, same 64-dim bottleneck as every KD arm, so any arm's delta isolates the teacher
signal rather than added capacity or the sampler. No paper.

### D1 — pointwise soft-label KD (GLNN family)
`w_logit: 1.0`; `kd_logit_loss` = masked-mean soft-target BCE of the student logit against
`teacher_logit`, row by row. The canonical GNN→MLP distillation baseline: match the teacher's score
on each pair independently.
Source: **GLNN**, *Graph-less Neural Networks: Teaching Old MLPs New Tricks via Distillation*
(Zhang, Liu, Sun, Shah; ICLR 2022) — [arXiv:2110.08727](https://arxiv.org/abs/2110.08727).
Result: AUROC +0.005, AUPRC +0.006, ECE 0.192 → 0.108 over control; MMDs 16.8/14.3/24.2 → 13.5/11.6/20.7,
RD-BFS 0.434 → 0.461. Both families move — the strongest all-round v2 arm on topology.

### D2 — listwise relational KD (LLP)
`w_rank: 1.0, w_dist: 1.0, temperature: 1.0, margin: 0.1`. `kd_rank_loss` = margin-ranking over every
within-anchor ordered pair (exact teacher ties excluded); `kd_dist_loss` = per-anchor
`KL(softmax(teacher/T) ‖ softmax(student/T))`. Constants verified against the released reference
implementation (`margin` argparse default 0.1; T is hardcoded to 1 at both `kl_loss` call sites).
Source: **LLP**, *Linkless Link Prediction via Relational Distillation* (Guo, Shiao et al.; ICML 2023)
— [arXiv:2210.05801](https://arxiv.org/abs/2210.05801).
Result: the edge/calibration winner of the v2 stage — AUROC 0.7266, AUPRC 0.7478, **ECE 0.0565**
(control 0.1925). Topology is *worse* than control on MMDs (17.1/14.6/26.0).

### D3 — pair-space Gram matching
`w_gram: 1.0`; `kd_gram_loss` = cosine-Gram matching between the student's pair-space geometry and
the teacher's pooled embeddings within each anchor group.
Source: **Graph2Feat**, *Inductive Link Prediction via Knowledge Distillation* (Samy, Kefato,
Girdzijauskas; WWW '23 Companion) — [10.1145/3543873.3587596](https://doi.org/10.1145/3543873.3587596);
same mechanism family as **CAZI-MBN** — [arXiv:2603.06618](https://arxiv.org/abs/2603.06618).
Result: the topology winner — MMDs 13.4/11.5/19.7 and RD-BFS 0.4605, the best of any arm; edge
metrics at control level (AUROC 0.7151) and ECE barely improved (0.176).

### D4 — projected representation alignment
`w_align: 1.0`; `kd_align_loss` = `1 − cos(P·(z_a ⊙ z_b), t_sym)` where `P` is a learnable 64→512
head on the bottleneck and `t_sym = ½(teacher_pooled_ab + teacher_pooled_ba)` in fp32. The learnable
projection answers the rotation/non-identifiability objection to raw coordinate MSE; the head is
dead at inference.
Source: **SA-MLP**, *Distilling Graph Knowledge from GNNs into Structure-Aware MLP*
— [arXiv:2210.09609](https://arxiv.org/abs/2210.09609); same family as **SALE-MLP** (IJCAI 2025,
[proceedings/2025/668](https://www.ijcai.org/proceedings/2025/668)).
Gate: MMD triplet ≤ D3's with AUROC ≥ control, **or** GS-BFS gain > 0.006.
Result: **fail.** MMDs 14.6/12.6/21.6 sit above D3's; GS-BFS 0.4137 is 0.004 *below* control. It
lands between control and D3 on every axis — representation-level transfer adds nothing that
output-level and relational KD do not already span. D4b (InfoNCE variant) does not earn a run.

### D5 — topology-residual distillation
`w_residual: 1.0`; `kd_residual_loss` = Huber between the student's node-factor residual
(`r + b_a + b_b`, exposed as `output["node_factor_residual"]`) and `Δ_T = teacher_logit −
content_logit`. Targets `kd_targets_breadth_first_seed0_v3` = the v2 npz plus an fp32 `content_logit`
array from the no-KD B0 v3.1 baseline (`e092537d8cf1e208`) scored on byte-identical rows. The intent:
make the bottleneck carry exactly the teacher's beyond-content signal and nothing else.
Source: in-house variant from the S-series proposal analysis; no external paper anchor.
Gate: edge + topology ≥ D1 with the residual head carrying the gain.
Result: **fail.** It reproduces D3's topology almost exactly (13.4/11.6/19.8 vs 13.4/11.5/19.7;
RD-BFS 0.4605 in both) with no edge gain over D2, the lowest GS-global in the table (0.1734), and the
worst calibration anywhere (ECE 0.323, Brier 0.321, F1 0.398, transfer threshold 0.570 vs ~0.83–0.93
elsewhere). Forcing a pointwise residual match distorts the logit scale without transferring
structure — see the target statistics below.

### D6 — D2 ⊕ D3 interaction test
`w_rank: 1.0, w_dist: 1.0, w_gram: 1.0`; no new loss code. The single justified deviation from
one-mechanism-per-arm: it tests whether the v2 headline split is additive, with D2 and D3 as its own
controls. No separate paper.
Gate: edge within noise of D2 **and** MMDs within noise of D3.
Result: **fail — interference.** Edge and calibration match D2 (AUROC 0.7260 vs 0.7266, AUPRC 0.7474
vs 0.7478, ECE 0.0544 vs 0.0565) ✓, but MMDs match D2 too (17.1/14.4/25.9 vs 17.1/14.6/26.0), not
D3's 13.4/11.5/19.7 ✗. At equal weight the Gram term leaves no trace.

### D7a — heuristic-teacher LLP
D2's exact losses and rows over `kd_targets_heuristic_ra_v1`: `teacher_logit := log1p(RA)` computed
on the training structural graph with standard queried-edge masking (drop the partner, decrement its
degree); pooled arrays zero-filled and never read. `log1p` is required because `kd_dist_loss`'s
softmax is not scale-invariant and LLP pins T = 1. With rows and losses frozen, D7a vs D2 isolates
*teacher provenance*: learned full-ego oracle vs parameter-free heuristic.
Source: **EHDM**, *Heuristic Methods are Good Teachers to Distill MLPs for Graph Link Prediction*
(Qin, Zhang, Ju, Zhao, Shah, Sun) — [arXiv:2504.06193](https://arxiv.org/abs/2504.06193).
Gate: beat D2 on either metric family.
Result: **fail, informatively.** EHDM's "weak teacher beats strong teacher" reproduces on AUROC only
(**0.7302**, the highest anywhere in B1) and inverts everywhere else: AUPRC 0.7448 < 0.7478, ECE
0.0981 > 0.0565, and the worst topology of any arm (RD-BFS 0.4114, MMDs 22.7/18.1/30.8). An RA
teacher ranks well and mis-shapes the assembled graph. D7b (heuristic ensemble + gate) does not earn
a run.

## Edge-level results (held-out test, seed 0, threshold 0.5)

| arm | mechanism | sel. epoch | AUROC | AUPRC | AUPRC non-self | ECE | Brier |
|---|---|---|---|---|---|---|---|
| `kd_control` | label BCE | — | 0.7154 | 0.7411 | 0.7017 | 0.1925 | 0.2560 |
| D1 | GLNN soft-logit | — | 0.7199 | 0.7467 | 0.7082 | 0.1082 | 0.2245 |
| D2 | LLP rank+dist | — | 0.7266 | 0.7478 | 0.7094 | 0.0565 | 0.2126 |
| D3 | Graph2Feat Gram | — | 0.7151 | 0.7452 | 0.7064 | 0.1758 | 0.2504 |
| D4 | SA-MLP align | 12 | 0.7177 | 0.7473 | 0.7087 | 0.1511 | 0.2389 |
| D5 | topology residual | 11 | 0.7216 | **0.7484** | **0.7108** | 0.3230 | 0.3213 |
| D6 | LLP ⊕ Gram | 25 | 0.7260 | 0.7474 | 0.7092 | **0.0544** | **0.2123** |
| D7a | EHDM heuristic-RA | 25 | **0.7302** | 0.7448 | 0.7040 | 0.0981 | 0.2227 |

## Assembled-graph results (density-matched threshold; GS↑, RD→1, MMD ratios↓)

| arm | GS global | GS BFS-macro | RD global | RD BFS-macro | MMD degree | MMD clustering | MMD spectral |
|---|---|---|---|---|---|---|---|
| `kd_control` | 0.1864 | **0.4175** | 0.9963 | 0.4342 | 16.82 | 14.26 | 24.23 |
| D1 | 0.1818 | 0.4192 | 0.9810 | **0.4608** | 13.54 | 11.64 | 20.66 |
| D2 | 0.1823 | 0.4129 | 0.9986 | 0.4379 | 17.11 | 14.55 | 26.01 |
| D3 | 0.1820 | 0.4142 | 0.9993 | 0.4605 | **13.37** | **11.54** | **19.65** |
| D4 | 0.1851 | 0.4137 | 0.9921 | 0.4459 | 14.63 | 12.63 | 21.55 |
| D5 | 0.1734 | 0.4101 | 0.9969 | 0.4605 | 13.39 | 11.57 | 19.78 |
| D6 | **0.1886** | 0.4163 | 0.9950 | 0.4401 | 17.11 | 14.39 | 25.85 |
| D7a | 0.1874 | 0.4099 | 0.9965 | 0.4114 | 22.73 | 18.09 | 30.84 |

D1's GS-BFS 0.4192 is the only value above control, by 0.0017 — inside noise and far under the 0.006
falsification bar. RD-global sits at ≈1 by construction (density matching); RD-BFS-macro is the
informative one and stays far below 1 for every arm, i.e. assembled ego-neighbourhoods remain much
sparser than reference.

## Assembled-graph results at the fixed 0.5 threshold (no density oracle)

`src/experiments/fixed_threshold_replay.py` over the same cached candidate universes
(`outputs/t05_replay/fixed_threshold_results.json`, 2026-08-19): assembly at p ≥ 0.5 — the operating
point the edge-level table already uses — with no density matching. Every self-pair clears 0.5 in
every arm (2,018 self-loops assembled; reference keeps 1,891).

| arm | GS global | GS BFS-macro | RD global | RD BFS-macro | MMD degree | MMD clustering | MMD spectral | edge precision | edge recall |
|---|---|---|---|---|---|---|---|---|---|
| `kd_control` | 0.1398 | **0.4333** | 3.14 | 0.8052 | 4.99 | 4.68 | 8.45 | 0.0921 | 0.2897 |
| D1 | 0.0658 | 0.3920 | 16.19 | 2.7324 | 16.42 | 15.91 | 11.28 | 0.0349 | **0.5655** |
| D2 | 0.0776 | 0.4052 | 11.73 | 2.0444 | 8.86 | 10.10 | 8.50 | 0.0421 | 0.4941 |
| D3 | 0.0906 | 0.4144 | 8.42 | 1.6538 | 7.31 | 7.95 | 5.96 | 0.0507 | 0.4264 |
| D4 | 0.0714 | 0.4001 | 14.05 | 2.4435 | 14.54 | 13.91 | 9.53 | 0.0382 | 0.5373 |
| D5 | **0.1613** | 0.4205 | **1.61** | 0.5786 | 8.00 | 6.94 | 12.44 | **0.1309** | 0.2102 |
| D6 | 0.0794 | 0.4076 | 11.43 | 2.0101 | 8.74 | 10.20 | 8.26 | 0.0432 | 0.4934 |
| D7a | 0.1081 | 0.4261 | 5.70 | 1.1435 | **3.41** | 4.51 | 6.41 | 0.0636 | 0.3622 |

Free-running, every KD arm over-assembles: 5.7–16.2× the true global edge count vs the control's
3.1× — soft-label/listwise KD pushes probability mass above 0.5 wholesale, so the arms' ECE wins do
not translate into a usable free operating point. The two exceptions are instructive: D5 (the ECE
loser, 0.32) lands nearest true density (1.61×) and takes the best global GS/precision, and D7a's
RD-BFS 1.14 gives it the best degree MMD (3.41). The control's GS-BFS 0.4333 exceeds every
density-matched value above — its ~3× over-density inflates per-subgraph Dice recall — so fixed-0.5
GS is not comparable to the density-matched table and is reported only within this one.

## D5's residual target is not degenerate

Over the 74,692 rows of `kd_targets_breadth_first_seed0_v3`: `|Δ_T|` mean 3.034, median 2.429,
p90 6.672, p99 9.641, std 2.669; teacher logit std 3.209, content logit std 2.908,
corr(teacher, content) = 0.623, var(Δ_T)/var(teacher) = 0.692. The teacher carries substantial signal
beyond the content-only baseline — roughly 69% of its logit variance — and D5 still fails. This is
the plan's stated failure reading on its informative branch: **beyond-content signal exists and is
not pointwise transferable**, which strengthens the relational-only account rather than merely
failing to refute it.

## Claim verdicts

- **C1 (mechanism classes combine, or their conflict is real) → the conflict is real.** D6 is D2 with
  a dead Gram term. Calibrated pointwise accuracy and relational geometry cannot be bought together
  at this recipe, and the listwise objective wins. A paper finding, not a failed run — but it removes
  the "one student wins both reporting families" option.
- **C2 (no KD class moves edge-set identity) → confirmed, and not KD-specific.** GS-BFS across all
  eight arms spans 0.4099–0.4192 (control 0.4175); GS-global spans 0.1734–0.1886 (control 0.1864).
  Five mechanism families move ECE by 0.27 and MMD-degree by 1.7×, and edge-set identity by nothing.
  S3's set-conditioned residual leaves GS equally inert on its own harness (0.282 → 0.287–0.291), so
  this is a property of the task under these features, not of the KD family.
- **Mechanism split (the v2 finding, now with v3 evidence).** Output-level KD (D1/D2/D7a) moves
  AUROC/AUPRC/ECE; relational geometry KD (D3, and D5 by proxy) moves MMD ratios and RD-BFS;
  representation alignment (D4) interpolates without adding a mechanism. D1 is the only arm that
  moves both families at once, and modestly.

## Caveats and open work

- **Single seed — and seed noise now has a measured scale.** Everything above is seed 0; the
  D1/D4/D5/D6 AUPRC spread is ~0.001. S3 (`s2_set_generation.md`), 3 seeds per arm on the same
  benchmark and the same v3.1-era machinery, spans 0.0093 AUPRC at fixed config on its readout.
  That is a different metric, so it is not a transferable error bar — but it is the first direct
  evidence that run-to-run noise on this machinery is the size of this table's whole arm-to-arm
  AUPRC range (0.7448–0.7484). Only the ECE and MMD effects survive an error bar by inspection.
  M4 (seeds 1–2 for control, D2, D3, and one v3 arm) is now the highest-value open item.
- **No v3 control.** All v3 deltas are against `b1_stage_v2/kd_control`. Checkpoint selection changed
  between the stages (commit `9da2ae7`, ball-union V_val scoring), which affects which epoch is
  published, not the test protocol — a comparability caveat for any quoted delta.
- **Published-epoch field is v3-only.** The v2 reports predate it, so v2 topology deltas are not yet
  fully separable from checkpoint-selection artifacts.
- **D4 provenance.** Its `complete.json` reads `debug_complete`: launched with `--max-steps`, a no-op
  under DDP, so it trained the full 25-epoch schedule but skipped the chained test stage; the test
  protocol ran separately against the same published checkpoint (`5f9a8a6248ddf32e`, epoch 12).
  Numbers are valid; the status string is not.
- **B-cal still open.** D2/D6's ECE win has not been checked against a temperature-scaled control
  (R8), so "KD buys calibration that post-hoc scaling cannot" remains unproven.
- **Tracker.** R1–R6 (M0–M3) complete; R7 (replication seeds) and R8 (B-cal) TODO. D4b and D7b are
  dead — their gates did not pass.

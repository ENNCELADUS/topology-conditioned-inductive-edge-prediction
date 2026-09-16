# Phase 5: adversarial review of the motif-graph/GRIT prompt decision

Date: 2026-09-16. Checkpoint 3, devil's advocate, read-only. Subject:
[`2026-09-16-motif-graph-grit-prompt-design.md`](../../../docs/superpowers/specs/2026-09-16-motif-graph-grit-prompt-design.md).
Evidence: reports A–E, `phase3_verification.md`, `docs/tmp/template_discriminativeness/`, the
virtual-prompt diagnosis, `docs/results/{l3ppi,topo_prompt_stage2_verdict}/`, memory.

## 1. Verdict

**Do not proceed as specified.** The design changes the representation and supervision of the predicted
structure but leaves the measured bottleneck — attribute→structure transfer for unseen proteins —
untouched, and it deletes the two ingredients the reviewed literature calls load-bearing (a direct
per-edge target, an explicit count pathway) while spending its whole complexity budget on the reader,
which the same literature measures as near-inert. Even at best the primary endpoint cannot resolve the
expected effect: one seed per arm, ±0.01 GS / ±0.5 MMD descriptive by project convention, and the last
arm's V_val MMD swung 5.8–14.4 across epochs of one run. Run the ~1 GPU-hour ceiling test in §4 first.

## 2. Fatal or near-fatal objections

### O1 — Same bet, and this exact mechanism has already lost on this benchmark (will fail)

Everything downstream of `(x_u,x_v)` is a deterministic function of it; the motif graph and GRIT add no
information, and the transfer numbers are not close: kNN-50 V_val Jaccard AUROC 0.74 against 0.91 for
truth and 0.793 for the trunk alone; within-node community ranking ρ ≈ 0.2 held-out; transferred
templates rank all pairs at prec@E 0.09–0.16 against the trunk's GS ≈ 0.40; `coord_gen_full`'s nonself
stratum transfers **0 of 21** statistics at ρ ≥ 0.5.

Worse, the mechanism — a fixed virtual template with query-conditioned learned gates read by a graph
net from endpoint features — is L3-PPI, which this project reproduced and which **lost**: winner
V_val AUPRC 0.730 / GS 0.250 against B0's 0.814 / 0.401, with 8–11 of 16 path gates on in every epoch
(a near-constant gate) and the hinge loss rising 1.2 → 3.4. The synthesis names L3-PPI three times as
"the especially close precedent" and never cites that negative, which sits in §2 of the probe README
it does cite. The published base rate agrees and is also absent: PTDNet's link-prediction table is
+0.006 / +0.023 / −0.001 AUC (report E: "*Nothing in this thread demonstrates a large, replicated
link-prediction gain from a learned structure*"), OpenGSL finds GSL "*does not consistently outperform
vanilla GNN counterparts*", and Mesquita et al. found randomised pooling costs nothing.

### O2 — 96 gates with no target of their own; the degenerate solution *is* the density control (will fail)

Supervision on the 96 weights is task BCE, subgraph BCE and `0.1·L_topo`, a representation MSE through
a frozen teacher. Nothing targets an edge. The minimiser of a token MSE over functions of `(x_u,x_v)`
is `E[t(A*) | x_u,x_v]`; at ρ ≈ 0.2 that is close to the global mean — every gate near its type bias,
a per-type density knob modulated by one pair-similarity scalar. That is the predecessor's measured
endpoint (rank-1 attachments, a per-pair scalar gate, `Cal` absorbing global inflation), and
within-role permutation invariance with three role embeddings makes it the path of least resistance:
uniform gates collapse 26 nodes to three equivalence classes, whereupon the design's own "one common
edge weight per pair and edge type" control is the *same model*.

The published remedy is the ingredient the spec deletes. LGI-LS: starved weights "*cannot be
semantically optimal, resulting in poor generalization*", and removing starvation improves every LGI
method (+6.12% Pubmed) — report C: "*supervision reaching the structure, not more capacity, is the
published remedy*". DiffPool needs an auxiliary `‖A − SSᵀ‖_F`; GraphSMOTE's structure-supervised edge
generator matches or beats the task-gradient variant on all three datasets. None reached the
synthesis. The project concluded the same independently — 1e-5 gradients per attachment logit under
aggregate-only supervision, fixed by the direct `virtual_graph.w_attach` loss in v0.6 — and **the spec
reverses it without argument** (§7: "Do **not** add … block-attachment regression"). v0.6 *with* that
loss still finished at base, `train_attach_r2` = −0.75; a weaker signal has no reason to do better.

### O3 — The complexity is spent on the arm the evidence says is inert (will fail)

The headline change is counting-decoder → GRIT. In L3-PPI's own ablation the reader class is worth
≤1.1 points on two of three datasets (GCN 83.12/87.97 vs full 83.22/88.08) while removing the gate
costs 6.66/8.18/7.13 — report A: "*the gate, not the reader, carries the ablation*". GraphGPS:
removing the MPNN is catastrophic (ZINC 0.070→0.217) while "*removing global attention: ZINC
unchanged*". IPR-MPNN, report C's "strongest evidence against", has learned virtual nodes beating
GRIT itself on the nearest analogue (Peptides-func 0.7210 vs 0.6988; PCQM-Contact link prediction
0.3670 vs 0.3498) via the discrete k-subset attachment the spec rules out. None is in the synthesis.

Sharpest: **LPFormer, the design's own cited precedent, concatenates raw CN / 1-hop / >1-hop counts
into its score, and on ogbl-ppa ablating those counts costs 18.95 Hits@100 (63.32 → 44.37) while
ablating the learned attention costs 0.55** — on protein-association data, ~35× more. The spec deletes
the count pathway ("do not sum them into scalar inputs to F") on a 0.037 in-house probe gap (0.710
counts vs 0.747 attachment vectors) that `phase3_verification.md` itself flags as training-node
retrieval — promoted over an 18.95-point published ablation pointing the other way.

### O4 — The teacher is selected by exactly the criterion the synthesis says to avoid (will fail)

Spec §7 picks `R_T` by the five-metric V_val rank on **true** templates — teacher accuracy on true
structure. The synthesis's own EHDM sentence says not to do that; report D's number is starker: a
near-perfect structure-dependent teacher scores teachable knowledge 0.61, **worse than a constant
teacher's 0.60**, plus "GNN4LP teachers are provably no more teachable than plain GNNs (Thm 3.2)".
The measured companion, absent everywhere: LLP's cold-start Hits@20 — teacher GNN 0.87–11.04 against
a plain MLP's 10.72–29.33 on isolated nodes; LLP's stated motivation is that "logit/representation KD
fails for LP", precisely the family `L_topo` belongs to. (Per `phase3`, EHDM is an existence result:
no evidence a stronger teacher is more teachable, one formal result that it need not be.) The
project's precedent has the same shape — Stage I reader 0.961 V_val AUPRC on true coordinates, its
Stage II student 0.806, *below* the 0.814 trunk.

This design widens the gap: the compiler keeps the **eight lowest-degree** common witnesses and top-8
candidates per side (median CN for positives 7 / 6 / 10; median L3 paths 74 / 69 / 314 against ≤64),
so the token target is a hard top-k selection of *rare* neighbours, while all that attributes transfer
is a population-smoothed, hub-weighted community prior (kNN rises to k≈50; k=1 carries 0.55).

### O5 — The primary endpoint will be uninterpretable, and the one control that would fix it is missing

`coord_gen_full` already produced this outcome: test RD 0.907 / MMD 4.2/3.6/6.7, traced by the audit
to a 0.661-nat V_val↔test logit differential, with GS 0.430 = B0's 0.430 and edge AUROC 0.677 against
0.721. The verdict's stated gap: "**no density-matched row exists for B0 or prefix_base** … That is
the actual evidence gap." The spec's eight comparisons are all internal — none is an independently
trained baseline at common density, and §8 forecloses the diagnostic rather than adding it. Also
missing, both named mandatory by the threads that produced them: a degree-matched control (Aiyappa;
Bernett — "*models learn solely from sequence similarities and node degrees*", degree alone reaching
~87–92%, every method at chance once leakage is removed) and per-C-class reporting (Park & Marcotte).
Power: one seed per arm, ±0.01 GS / ±0.5 MMD descriptive by project convention, V_val MMD swinging
5.8–14.4 within one run, against an expected +0.02–0.04 AUROC on a secondary metric family.

### O6 — Cost against expected value

Stage I + Stage II + eight controls, several of which the spec requires be *retrained*, is ≥10 runs at
≥13 h — a floor, since Stage II adds a frozen-teacher GRIT forward on `Âhat` with input gradients plus
a detached forward on `A*` per row. Plus a new family, compiler and GRIT adapter (`GritGmtEncoder`
discards final pair states, so `topo_rel` does not exist yet) and the mandated Codex wave, while
`topo_prompt_full_v4` and the v0.6 chain already occupy 30030. Against +0.024 AUROC.

### O7 — The decision record selected against its own evidence base

The synthesis names six sources, zero of the ~100 others in A–E, and compresses fourteen "evidence
against" items from D and E into two sentences; every literature number is dropped while the project's
own probes keep theirs. Two carries mislead. NCNC, the sole precedent for reading soft predicted
structure, gains **+0.23 on ogbl-ppa** (≤2.24 anywhere), completes from an *observed* graph, and its
authors warn "*weak models may not accurately recover the unobserved common neighbor structure*" —
cited with no number or caveat. And mixing ℓ=2 wedges with ℓ=3 bridges is Kovács et al.'s post-mortem
for CRA's failure ("*CRA lost because it mixed ℓ=2 paths into ℓ=3 counting*"), from the same paper
that has CN/Jaccard *anti*-correlating with interaction in every PPI network tested.

## 3. Objections the spec already handles

Content double-transmission (R sees role embeddings and adjacency only); exact gates-off identity to
the base; the shuffle per-shard confound, with the interventions honestly labelled as not
degree-preserving; continuous gates do carry gradient, with a per-edge-type check required;
closure-vs-L3 is not assumed, both families keeping *retrained* single-family ablations; prior test
exposure is disclosed; the fixed 26-node count defuses the off-size-collapse framing. Not the problem.

## 4. The cheapest decisive experiment

**Reachable-input ceiling through an existing reader. One script under `src/experiments/`, no model
change, no training, ~1 GPU-hour** — the diagnosis doc costed the identical procedure (its Step 0).

Take the published `topo_prompt_frozen` checkpoint: on *true* coordinates it reaches V_val AUPRC
0.940 / GS 0.648 with the best shape ratios of any row, and `gates_off` reproduces `prefix_base`
exactly. Score V_val **and test** through the existing `row_coords` path
(`--allow-oracle-diagnostic`) on four coordinate sources: truth; kNN-50 attribute-transferred
neighbourhoods (already computed in `template_stats.py`); training-mean; kNN-50 at matched density.
Report the five-number panel at the V_val-selected threshold.

**Decision rule.** If the kNN-50 row does not beat `prefix_base` by more than ±0.01 GS and ±0.5 on at
least two MMD ratios, *and* hold that margin at matched density, then no endpoint-only generator can:
kNN-50 also reads the training adjacency at scoring time, so its input strictly dominates the
generator's. Do not build the wave; write the negative. If it clears, the wave has a number to beat.

**Second probe (CPU plus one state-caching pass):** fit a head from frozen *residue* states, not
pooled F0, to a smooth community-level attachment target, on held-out training nodes and V_val.
`phase3` rightly notes kNN-50 is not an information ceiling among functions of these features, and
residue attention is the design's only new input pathway; if residue states do not beat pooled kNN-50
there, "the same bet in richer clothes" is confirmed at the source. Both are diagnostics, kept out of
selection and out of any deployment claim.

## 5. If it proceeds, the minimum changes

1. **Give the 96 gates a target:** direct, *attainable* per-slot supervision (smoothed community-level
   occupancy), not a 0.1-weight token MSE alone. Without it, O2 stands.
2. **Keep a count pathway** as a concatenated arm (topology tokens **plus** scalar wedge/L3
   coordinates). LPFormer's ogbl-ppa ablation makes this the arm to beat.
3. **Add the density-matched `prefix_base`/B0 diagnostic** alongside — never instead of — the
   V_val-selected threshold; add a degree-matched control and self/nonself + C-class strata.
4. **≥3 seeds** for the main arm and the density control (pay by dropping the endpoint-only/
   relation-only rows and one motif-family ablation), and **pre-register the reading** — row, metric,
   margin — as the v0.6 plan did.
5. **Select `R_T` for teachability**, not true-template V_val rank, and extend corruption to
   `λ ~ U(0,1)`: the predecessor's realised quality maps to λ ≈ 0.75–0.85, outside `U(0,0.5)`.
6. **Normalise the gate-head inputs; log saturation, weight rank and family usage from epoch 1.** The
   predecessor's gate died by input-scale growth (‖u∗v‖ 23→162, σ′≈2e-6) and nothing in
   `sigmoid(MLP([h_i+h_j,|h_i−h_j|]))` prevents a repeat. Telemetry, not a gate — but act on it.
7. **Fix the self-row contract:** self rows get an empty Stage I template and no Stage II alignment
   yet a generated graph at inference with no flag; that label-pure stratum overturned the last audit.

AI-assisted adversarial review. No run, scoring or code change performed.

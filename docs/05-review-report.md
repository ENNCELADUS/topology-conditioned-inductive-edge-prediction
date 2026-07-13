# Review Report: EgoStitch Proposal — Novelty Check + 5-Persona Design Review

**Date:** 2026-07-08. **Object reviewed:** `04-model-proposal.md` (revision 2 at review
time; the design-stage findings below are applied in **revision 2.1** of that document —
see §5 for the disposition of every finding). **Reviewers are read-only:** no reviewer
modified the manuscript; all changes were made afterward in the author role.

**Post-review experiment status (updated 2026-07-13):** the G1 B0/PA-null arm and G2
have been run on the repository-local `breadth_first` artifacts. The current values and
the remaining G1 B0-alt replication and G3 Oracle requirements are recorded in
[`docs/results/E2-pair-to-topology-gap.md`](results/E2-pair-to-topology-gap.md). The
review findings below remain a historical design-stage record; current E2 numerical
claims are recorded only in the result note.

**Process.** (1) Literature basis: five parallel extraction agents over the full local
vault (~90 PDFs + 4 synthesis reports), two arXiv API novelty-risk sweeps (32 queries,
68 unique candidates, all verified against the arXiv export API), four web searches,
and a partial Semantic Scholar pass. (2) `/novelty-check`: Phases A/B/D completed;
**Phase C (cross-model verification via Codex gpt-5.5) failed** — two attempts aborted
at the 30-minute MCP idle timeout; rerun before submission. (3)
`/academic-paper-reviewer`: full 5-persona panel (EIC, methodology, domain,
cross-disciplinary/CV, Devil's Advocate), run independently, synthesized below.

---

## 1. Novelty Check Report

### Proposed method
EgoStitch: per-endpoint generative ego-networks (slots + slot-adjacency + degree
budget) conditioned on frozen features and a VQ community codebook, stitched and
harmonized into a per-query scaffold for strict zero-edge inductive edge prediction,
graded jointly on edge metrics and assembled-graph realism.

### Core claims
1. **C1 — generated ego-net scaffold for zero-edge pairs** — Novelty: **HIGH** —
   Closest: NCNC 2302.00890 (transductive completion), Cold Brew 2111.04840 (virtual
   neighborhoods without structure), NRI 1802.04687 (needs trajectories), FLEX
   2507.11710 (generation as training augmentation). No prior generates per-query
   latent ego-network *structure* for zero-edge nodes at inference.
2. **C2 — consensus harmonization of two generated ego-nets** — Novelty: **HIGH** (on
   graphs) — Closest: MaskGIT 2202.04200 / RePaint 2201.09865 (images; the transfer
   requires the joint two-ego training task added in rev 2.1).
3. **C3 — generation-conditioning VQ neighborhood codebook** — Novelty: **MEDIUM** —
   Closest: VQGraph 2308.02117 (its distilled student codes zero-edge nodes from
   features — claim narrowed in rev 2.1), GFT 2411.06070. Survives on the *generative
   use + ego-net-statistic supervision + edge-prediction task* qualifiers only.
4. **C4 — protocol-gated joint edge + assembled-realism evaluation** — Novelty:
   **MEDIUM** — Closest: Graph Gestalt 2106.15239 (priority on the dissociation
   observation), ERGM goodness-of-fit (classical priority on grading fitted network
   models structurally), NetGAN VAL-CRITERION. Claim scoped in rev 2.1 to "first
   protocol-gated joint evaluation under strict zero-edge inductive LP."
5. **C5 — conditioning dropout = TDE counterfactual (CFG↔TDE)** — Novelty: **LOW as
   originally stated** (conflated two nulls; counterfactual-subtraction debiasing is
   established in SGG/VQA) — retracted and replaced in rev 2.1 by two trained nulls
   (`∅_content` TDE-style control, `∅_all` CFG-style prior), claimed only as the
   amortized trained-null implementation in this setting.

### Closest prior work
The full 18-row threat matrix with per-row deltas is maintained in
`04-model-proposal.md` §5.3 (kept there as the single source of truth; every entry
verified against local PDFs or the arXiv API — the domain reviewer independently
re-verified 30 IDs including abstract-level checks on FLEX and TGSBM).

### Overall novelty assessment
- **Score: 7.5/10** (single-model assessment; cross-model verification pending).
- **Recommendation: PROCEED WITH CAUTION.**
- **Key differentiator:** the structural dividing line — every close prior requires the
  query node's observed edges at inference; none scores zero-edge pairs; none grades
  the assembled graph — plus the harmonization mechanism, which no graph work has.
- **Risk:** conjunction-shaped novelty (five qualifiers) + a many-mechanism system;
  reviewers who collapse two qualifiers see VQGraph+Cold-Brew+EDGE. The mitigation is
  the ablation discipline and the `B3-full` control: the composition must be shown
  load-bearing empirically.
- **Caveats:** Codex cross-verification not completed (infrastructure); Semantic
  Scholar mostly rate-limited (set `SEMANTIC_SCHOLAR_API_KEY` and rerun); a dedicated
  2025–2026 sweep for zero-edge LP / graph-foundation-model zero-shot LP should be
  repeated close to submission.

### Suggested positioning
Lead with the task + protocol (settings taxonomy, §5.2 of the proposal) and the E2 gap
*after* it survives gate G1; state the thesis as an inductive-bias/supervision claim
(never an information claim); concede Graph Gestalt and classical ERGM priority
explicitly; position against the KG-inductive lineage before a reviewer asks.

---

## 2. Panel reviews — summaries with ratings

Full reports are preserved in the session transcript; this section records each
reviewer's verdict and their distinct load-bearing findings (no duplicates — where
reviewers converged, the finding is listed once in §3).

### R0 — Editor-in-Chief (Senior Area Chair, graph learning + generative models)
**Predicted ICLR score: 7/10 if results land exactly as predicted on ≥3 named public
benchmarks (rising to 8 with a strong E7; falling to 6 on one benchmark family); 5/10
if topology gains fail to separate from `B0+cal`/`B3-dist`.** Distinct findings: the
paper carries two contributions in one envelope (task framing + heavy model) and risks
under-serving both; placeholder benchmarks defer external validity; §6.6's predicted
flat edge metrics mean the significance case rests entirely on E7, which was optional;
a minimum-viable-model milestone is missing; the frozen B0 anchor may cap edge
quality. Called the proposal "one of the most reviewer-anticipating
pre-implementation designs I have reviewed."

### R1 — Methodology (benchmark/evaluation rigor)
**Soundness of design: 6/10, confidence 4/5** (rises to ~8 with W1–W6+W8 fixed).
Distinct findings: loss stack of ~15 sub-terms with two orphans and no balancing
protocol (W1); harmonization gradient path undefined and possibly inference-only —
train/test mismatch violating the proposal's own R12 (W2); generator can become
`L_edge`'s covert channel, and the direct evidence (held-out ego-net fidelity) was
missing from the evaluation (W4); message/supervision edge partition missing — s3 is a
train-time quasi-oracle on positive pairs; seam references must be label-agnostic; B0
provenance unaudited (W5); Module 3b not implementable from the text (W6); Goodhart —
training on the evaluated metric family, with the blueprint's held-out metric families
silently dropped by the protocol (W7); E2 cannot carry the motivation as cached (W8);
several ablation gaps and a confounded identical-head comparison (W9); amortization
claim inconsistent (W10); B4 vanished without comment (W11).

### R2 — Domain (link prediction + graph generation)
**Contribution to field: 6/10, confidence 4/5** (7 with W1–W4 fixed). Verified 30
citations independently; found no outright mischaracterization. Distinct findings: the
KG-inductive lineage (GraIL/NBFNet/ULTRA) is entirely absent and owns the term
"inductive link prediction" — a desk-level vulnerability at ICLR (W1); the §1.1
information critique boomerangs — reframe as prior injection (W2); Chanpuriya
2111.00048 is both the strongest backbone for E2 and the bound on the remedy, and was
missing (W3); labeling-trick/γ-decay/"provably forced" all over-stretched (W4);
VQGraph's distilled student invalidates one qualifier (W5); C4 needs classical-priority
scoping (ERGM GOF) in all four documents (W6); ~15 missing references (W7);
HeaRT-style negatives are ill-defined for zero-edge nodes without an evaluator-side
construction rule (W8); cold-start recommendation under-engaged (W9).

### R3 — Cross-disciplinary perspective (CV generative modeling)
**Technical soundness of transfers: 6/10, confidence 4/5** (8 with W1/W3/W4/W7
fixed). Distinct findings: Hungarian regime inverted vs DETR — hubs with degree ≫ K
make the budget trigger vacuous on exactly the degree axis (W1); near-duplicate
targets poison slot-adjacency supervision specifically (W2); the harmonization
schedule queries conditionals no training task produces — the biggest architectural
gap; fix by joint two-ego cross-conditioned training (W4); confidence defined over
existence only — content confidence missing; kept errors frozen (Token-Critic lesson)
(W5); alternating re-decoding is pseudo-Gibbs without a shared joint (W6); the CFG↔TDE
identification conflates two nulls — technically incorrect as stated (W7); grounded
slots are the discredited ANN peers acting as "known pixels" — gate must be a
partner-vs-peer discriminator (W8); compute comparator must be B0, with a
full-universe wall-clock table and the cascade caveat (W9); conditioning entropy makes
mode-seeking likely and s1 may collapse into s3 (W10); three additional disanalogies
for §5.3.

### R4 — Devil's Advocate
**Verdict: two CRITICALs fatal-as-written (fixable), two fixable-by-design, rest by
experiment.** Strongest counter-argument: under the locked contract the assembled
graph is an edge-independent construction; every §4.6 mechanism binds the scaffold —
an object never evaluated — and the honest headline risks being "a degree-corrected
block model with expensive scaffolding." Distinct findings: CRIT-1 scaffold/assembly
category error; CRIT-2 edge-independence ceiling never checked (checkable now, on
paper); CRIT-3 the information argument is special pleading — reframe as inductive
bias; CRIT-4 the E2 evidence is one weak scorer, one benchmark, easy negatives, an
undefined composite, no Oracle reference; CRIT-5 the Ockham arm (`B3-full`) is missing
while the evaluation plan had not yet resolved the Ockham control and was deflated by
a caveat the headline numbers share — textbook asymmetric evidence treatment; M6 the
automorphic argument binds ungrounded EgoStitch equally; M7 pool
dependence treated as both flaw and feature; M8 no stakeholder needs the exact
conjunction unless E7 lands; M9 channel collinearity → `Ours ≈ B5` prediction; M10
Oracle must run first; M11 undefined headline composite; M12 gameable success
criteria; M13 over-extended theory imports; M14 unspecified stochastic inference.
Also: B5 as "first strict-inductive BP evaluation" is a genuine, safer standalone
contribution; the E2 *existential* claim (metrics can diverge) survives all attacks —
it is the *universal* claim (topology conditioning is the right fix) that awaits
evidence.

---

## 3. Editorial decision

**Decision: MAJOR REVISION (design stage) — revise before implementation.**
Per the panel's iron rule, the Devil's Advocate CRITICALs bar acceptance of revision 2
as-is. No reviewer recommended abandonment; all five judged the setting real, the
novelty analysis honest, and every finding fixable by design change or pre-registered
experiment.

**Consensus (≥3 reviewers, independently):**
1. Scaffold-level "guarantees" were claimed at assembly level (R1-W3, R3-W3, DA-C1,
   R2-W3) — the proposal's single most serious internal flaw.
2. The information-theoretic framing boomerangs; the defensible thesis is prior
   injection / inductive bias (R2-W2, DA-C3, EIC-originality).
3. E2 as cached cannot carry the motivation; hardening + Oracle + ceiling must precede
   implementation (R1-W8, DA-C4/C2/M10, EIC-concern-1).
4. The Ockham/simple-arm control was missing (`B3-full`) (DA-C5, R1-HPO-parity,
   EIC-concern-1).
5. E7 must be load-bearing, not optional (EIC-concern-2, DA-M8).

**Notable disagreement:** R3 treats harmonization as the design's most valuable novel
mechanism once trained properly; DA treats it as importing the CV lesson at the wrong
level (within-query, while the E2 seam is cross-query). Arbitration: both are recorded
— harmonization stays (it owns the within-scaffold seam and an ablation), while the
cross-query limit is now explicit in §1.6/§4.6 and bounded by gate G2. The tension is
resolved empirically by the `R`-sweep plus the ceiling row, not rhetorically.

---

## 4. Revision roadmap → disposition (what was done)

All design-stage items are **applied in `04-model-proposal.md` revision 2.1**;
experiment-stage items are **registered as pre-implementation gates** there (§6.0) and
flagged **[protocol-Δ]** where they require sign-off to extend the locked protocol.

| # | Finding (source) | Disposition in rev 2.1 |
|---|---|---|
| 1 | Scaffold/assembly category error (unanimous) | §4.6 rewritten as a 4-column map with an explicit transmission step; all "guarantee" wording scoped; degree-calibration diagnostic §6.4.8b |
| 2 | Edge-independence ceiling (DA-C2, R2-W3) | Chanpuriya 2111.00048 adopted in both roles; gate **G2** computes the reachable frontier pre-implementation with a stop condition; ceiling row required in every assembled table |
| 3 | Information-claim special pleading (DA-C3, R2-W2) | New binding "honest form of the thesis" block; §1.1 and §1.9 rewritten (prior injection; automorphic limit binds ungrounded EgoStitch; grounded-vs-ungrounded promoted to headline ablation — also resolves DA-M7) |
| 4 | Missing Ockham arm (DA-C5) | **`B3-full`** added as the decisive control; E2 §5 cached preview elevated to a first-class hypothesis re-run under G1 |
| 5 | E2 fragility (DA-C4, R1-W8) | Gate **G1** (full hardening list incl. defined composite, noise floor, sweeps, hard negatives) with stop condition; "strong scorer" wording retired |
| 6 | Oracle-first (DA-M10) | Gate **G3** with stop condition |
| 7 | Untrained harmonization conditionals (R3-W4/W6, R1-W2) | Joint two-ego cross-conditioned masked training task specified; single shared decoder answers pseudo-Gibbs; gradient estimators stated; algorithm box = gate **G4** deliverable |
| 8 | CFG↔TDE conflation (R3-W7) | Two trained nulls (`∅_content`, `∅_all`); unification claim retracted; guidance scoped to logit space |
| 9 | Hub/Hungarian inversion (R3-W1/W2) | Hub policy: importance-weighted subsampling + per-slot multiplicity `m_u^k`; budget trigger redefined K-representably; compound matching cost; denoising queries; flip-rate diagnostic; group-level adjacency supervision |
| 10 | Content confidence / frozen keeps (R3-W5) | Codebook-quantization content confidence; π temperature calibration required; critic may re-mask kept slots |
| 11 | Grounding as corrupted known pixels (R3-W8) | Gate trained as partner-vs-peer discriminator; grounded slots re-maskable at reduced probability |
| 12 | Loss tree under-specified (R1-W1) | Full tree with owners; two orphans assigned to `L_recon`; balancing + HPO-parity protocol §6.5 |
| 13 | Leakage partition (R1-W5) | Message/supervision edge partition (R9); label-agnostic seam references; B0 provenance gate — all E5 gates |
| 14 | Covert-channel identifiability (R1-W4) | Held-out ego-net fidelity diagnostics §6.4.8a as required mechanism evidence; watchdog note in `L_edge` |
| 15 | Goodhart metric circularity (R1-W7) | Trained-on vs held-out metric split; held-out family headlined; blueprint-vs-protocol metric inconsistency flagged [protocol-Δ] |
| 16 | KG-inductive lineage absent (R2-W1) | §5.2 settings taxonomy; GraIL/NBFNet/ULTRA/IGMC added and positioned |
| 17 | Theory over-stretch (R2-W4, DA-M6/M13) | §5.4 scoped-theory section; conditional γ-decay; qualifier on labeling trick; "provably forced" → "consistent with a known expressiveness limit" |
| 18 | VQGraph student nuance (R2-W5a) | §4.1 claim narrowed; §5.3 row corrected |
| 19 | C4 scoping (R2-W6) | "Protocol-gated, strict zero-edge" qualifier adopted; ERGM/Hoff classical priority cited; docs 01–03 flagged [protocol-Δ] for the same qualifier |
| 20 | Missing references (R2-W7) | All must-cites added to §8 with ⊕/[venue-verify] tags; cold-start recsys engagement rule added (benchmark or explicit scope exclusion) |
| 21 | Zero-edge hard negatives ill-defined (R2-W8) | Evaluator-side construction + feature-similarity negatives specified §6.4.1 |
| 22 | Compute honesty (R3-W9, R1-W10, DA-M20) | §4.7 rewritten: B0 comparator, FLOPs/wall-clock table commitment, R=0 row, cascade caveat |
| 23 | Channel collinearity (DA-M9, R3-W10) | Correlation-matrix diagnostic; FCR-stratified pre-registered prediction; decision rule §6.5(ii) |
| 24 | Stochastic inference (R1-Q8, DA-M14) | §4.0 determinism policy (n_s samples, averaged p, seeds) |
| 25 | Gameable criteria (DA-M12) | §6.5 pre-registered decision rules incl. Holm correction and the honest small-paper outcome |
| 26 | E7 optional (EIC, DA-M8) | Promoted to load-bearing in §6.6 |
| 27 | No staging milestone (EIC) | Gate **G5** minimum-viable-model staging |
| 28 | B4 vanished (R1-W11) | Registered as E4.11 with explicit disposition [protocol-Δ] |
| 29 | d̂ scale transfer (R1-W11-iii, R2-Q8) | Density-normalized budget §4.1; E6 check |
| 30 | Grounding collapse under L_ssl (R1-W9) | Pool-consistency on ungrounded slots only; mean-g^k diagnostic; risk row |

**Not fixable by text (registered, awaiting experiments):** whether the gap survives G1;
whether the ceiling (G2) leaves headroom; whether Oracle (G3) shows conditioning
headroom; whether `Ours` beats `B3-full`/`B0+cal` on held-out metrics; whether s1/s2
survive knockouts; whether E7 shows downstream value. These are exactly the six things
§6.5's pre-registered rules adjudicate.

---

## 5. Process degradations and follow-ups (disclosed)

1. **Codex MCP cross-model verification failed** (two 30-min idle timeouts at xhigh
   and high reasoning effort). The novelty verdict is single-model with multi-agent
   evidence. **Rerun `/novelty-check` Phase C before submission** (check Codex MCP
   server health / raise the per-server timeout).
2. **Semantic Scholar rate-limited** (HTTP 429 on 9 of 10 queries without an API key);
   the one successful query returned no new threats. Set `SEMANTIC_SCHOLAR_API_KEY`
   and rerun the venue-only sweep before submission.
3. **[venue-verify] tags in §8** of the proposal mark citations verified by ID/abstract
   but not against the version of record (LGD main-text task coverage; Rendsburg;
   Heater/MetaEmbedding venues; DCM's in-print DGM criticism).
4. **Review traces** written to the session scratchpad (no `.aris/` toolchain in this
   repo); the durable record is this document.

## 6. Library housekeeping notes (from the fan-out pass)

Resolved in vault housekeeping on 2026-07-09 where noted; named-only citation lookups
remain separate from this review record.

- `literature/models/graph_structure_learning/`: SUBLIME (2201.06367) is duplicated
  across `latent_structure_learning/` and `self_supervised_structure_learning/`; DGM
  (2002.04999) has both a 2022- and a 2023-prefixed copy in
  `latent_structure_learning/`. **Resolved:** duplicate active copies archived under
  `literature/archive/duplicates/`.
- HoscPool PDF filename says `2022_icml_...` but the paper is CIKM 2022 — rename per
  the library convention. **Resolved:** active filename now uses `2022_cikm_...`.
- MGAE preprint (2201.02534) and S2GAE (WSDM 2023) are the same line of work — cite
  the version of record.
- The local copy of TDE (2002.11949) is arXiv v4 (stamped 2025) — quote-check against
  the CVPR 2020 camera-ready if quoting text.
- `literature/README.md` (manifest) listed stale counts (90 PDFs listed; 89 found under
  `models/`). **Resolved:** manifest now separates active PDFs from archived duplicates.

---

## 7. Addendum (2026-07-09): approval, amendments, and second-pass verification

**Approval received:** revision 2.1 approved **as a design proposal, not yet an
implementation contract**, with three required amendments — all executed:

1. **G4 spec freeze delivered.** `06-egostitch-spec.md` (2026-07-09): Stitch/Harmonize
   pseudocode with tensor shapes, OT cost and ε, confidence + cosine quantile schedule,
   K-representable budget tolerance τ_b, gradient estimators for every hard operation,
   two-null conditioning-dropout scheme, hub/multiplicity policy, and the full loss
   tree with interior weights, HPO-parity budget, and determinism policy. Pending
   review sign-off; on sign-off it becomes the implementation contract with a
   change-log discipline.
2. **Protocol updated.** The approved [protocol-Δ] items are merged into
   `03-experiment-protocol.md` (updated 2026-07-09): pre-implementation gates G1–G5
   (Oracle-first run order), ladder extensions (`B0+cal`, `B3-dist`, `B3-full`, B5,
   DEAL/Graph2Gauss, PA-null, odds-product), trained-on vs held-out metric families
   with ceiling/Oracle/noise-floor rows, zero-edge hard-negative construction,
   message/supervision partition + provenance gates, Holm-corrected pre-registered
   decision rules, E7 promoted to load-bearing, B4 → E4.11 disposition, and the scoped
   C4 claim.
3. **Second-pass literature verification.** The 21 papers previously cited from
   abstracts (now full PDFs in the vault, incl. the new `kg_inductive_lineage/`
   folder) were read by two verification shards. Result: **no mischaracterization
   threatens the core positioning**; the proposal is updated to **revision 2.2** with
   these corrections:
   - *Chanpuriya ceiling semantics* (substantive): the G2 gate now uses the paper's
     exact identities (`E[Δ] = tr(P³)/6`, `Ov·V = Σp²`) and states the ceiling as a
     **curve over overlap** evaluated at the soft scorer's measured overlap — a
     thresholded assembly has `Ov = 1` where the bound is vacuous; the "cannot
     reproduce triangles without memorization" gloss is retired.
   - Quote-backed KG-inductive taxonomy (IGMC's own "does not address the extreme
     cold-start problem" line; NBFNet path-set definition; ULTRA's inference-graph
     dependency), with the structure-only/features-only complementarity noted.
   - P-GNN split from the labeling-trick family (anchor-based positional); the §5.4
     fidelity qualifier extended to Distance Encoding.
   - Topological Concentration citation corrected (it *undercuts* naive degree
     strata); TC stratification and a TDS-style ego-net drift diagnostic added.
   - O'Bray rules imported verbatim (bin-count disclosure, no ad-hoc EMD/TV kernels,
     perturbation-validated composite); PA-null and odds-product baselines added;
     G1 re-verifies E2 under degree-corrected negatives.
   - GraphMAE scope qualifier (classification-side evidence only); UPNA, New Node
     Prediction (semi-inductive), HiGGs, FLEX distinguishing sentences; FLEX's
     dense-generation failure recorded as independent support for hard budgets.
   - Venue statuses: Meta-Embedding (SIGIR 2019) **tag cleared**;
     NBFNet/ULTRA/P-GNN/DE/LPFormer/O'Bray/TGB confirmed from PDFs; GraIL/IGMC local
     PDFs are arXiv preprints (page-quote checks pending); 2310.04612, 2405.14985,
     UPNA, FLEX are preprints; HiGGs and 2401.05468 cited as arXiv without venue.

**Remaining open follow-ups (updated):**

1. `/novelty-check` Phase C (Codex cross-model verification) — still pending; Codex
   MCP was unresponsive this session (two 30-min idle timeouts).
2. Semantic Scholar sweep — retried 2026-07-09; `SEMANTIC_SCHOLAR_API_KEY` was **not
   visible in the agent's shell environment** (queries still 429'd unauthenticated;
   the one batch that got through returned no relevant hits). Export the key where
   agent Bash sessions can see it (e.g., `env` in `.claude/settings.json`) and rerun.
3. Residual [venue-verify] items: LGD main-text task coverage (against the NeurIPS
   camera-ready); Rendsburg "NetGAN without GAN"; Heater; DCM's in-print DGM
   criticism; GraIL/IGMC page-level quotes; HiGGs acceptance status; acceptance
   status of 2310.04612 / 2405.14985 / UPNA / FLEX closer to submission.
4. ~~G4 sign-off of `06-egostitch-spec.md`~~ — **done 2026-07-09**: the user signed
   off G4; the spec is the active implementation contract and gained §9 (benchmark
   binding / data contract — including quarantine of the shipped
   `*_ratio5_exclusive.txt` negatives, which mix train- and test-side nodes, and of
   `train_graph.pkl` as a structural-target source, since it contains every val
   positive), §10 (batch sampler), and the original §11 DDP execution design for
   4/8 × H20 (superseded 2026-07-10 by the locked 1×H20 single-process design).
   Remaining before any model code: gates G1 → G2 → G3.

# Verification of the recovered review and dataset probes

Date: 2026-09-16. This is a targeted verification of the existing work, not a new literature search.
Read the five A–E reports, the research brief, the probe README, saved JSON/logs and relevant probe
source. Checked decisive claims against locally cached primary papers and current implementation.
The bibliography's 21/24/31/31/21 entries overlap; these are not 128 distinct independently verified papers.
The agents' full bibliographic verification is inherited, not repeated here. Corrections below take
precedence over the original reports and the probe README when using them for this decision.

## 1. Recovered work

All five bibliography reports exist. A, B, D and E delivered their reports. C saved its complete
seven-section draft and hit the session limit while shortening it. Its length is an editorial issue,
not a missing research result. The original orchestrator stopped before verification, synthesis and
architecture selection. Five CPU probes have JSON output and completion lines in their logs. Their
existence is verified; none was rerun on the dataset during this takeover. No HPC state was inspected.

## 2. Corrections that change the interpretation

| Claim in the recovered work | Verified boundary / correction | Design consequence |
|---|---|---|
| A: the single-token prompt theorem diagnoses the shared 64-node graph's collapse | [Wang et al., Thm. 6](https://arxiv.org/abs/2410.01635) concerns GCNs with a single GPF vector or single-token All-in-One prompt. A multi-node, query-conditioned generator does not satisfy those assumptions. | Treat shared-capacity limits as motivation, not a proof about this model. |
| A: increasing reader expressiveness predicts worse performance | Thm. 5 gives an **upper bound** on approximation error under rank assumptions. Increasing an upper bound does not imply actual error increases; neither this theorem nor its experiments compare our counting reader with GRIT. | GRIT is an empirical design choice, not theoretically ruled out. |
| C: independently encoded endpoint attributes make the current pair MLP structurally impossible | [Labeling Trick, Prop. 1](https://arxiv.org/abs/2010.16103) establishes the insufficiency of independently computed structural node representations for universal structural link representations. It does not prohibit useful feature-only pair functions, nor prove that their joint MLP cannot predict topology. | Mark the target pair inside R; do not claim an impossibility for endpoint-only prediction. |
| D: edge independence proves that our block model cannot express clustering or Jaccard | [Chanpuriya et al., Thm. 1](https://arxiv.org/abs/2111.00048) bounds expected triangles jointly with expected volume and overlap. Deterministic triangle-rich graphs are permitted at high overlap. The leading constant in the PDF is **sqrt(2)/3**, not 2/3 as transcribed in D. It is not a theorem about Jaccard regression. | The observed reconstruction errors are evidence; a blanket impossibility is not. |
| D: K=64–1024 is provably too small on this dataset | The cited low-rank result needs its graph/model assumptions and asymptotic quantities instantiated. No such calculation was supplied. Nor is an attachment vector a normalized MMSBM membership merely because it has K components. | Do not present a numerical minimum rank or a guaranteed remedy. |
| C: a soft graph gives its generator no task gradient | The DGM issue discussed concerns discrete sampling. A continuous adjacency with differentiable RRWP and attention does carry gradients. This is also visible in the local GRIT implementation. | Use continuous gates first; no necessity for REINFORCE or straight-through sampling. |
| C: GRIT's weighted-graph guarantees follow from its binary-graph theorem | [GRIT Prop. 3.1](https://arxiv.org/abs/2305.17589) explicitly quantifies over binary adjacencies; its shortest-path proof uses a minimum positive walk probability over that finite set. Soft RRWP is well defined with a degree floor, but no corresponding uniform shortest-path claim follows. Near-zero degree normalization can be sensitive. | Keep weighted edges and degree information explicit; test zero/tiny weights and train on soft corruption. |
| A/E: L3-PPI's second stage trains only the gate | [L3-PPI §4.3](https://arxiv.org/abs/2605.09964) first trains prompt embeddings, then jointly optimizes prompt embeddings and the gate. This does not establish a schedule for unfreezing a graph reader. Its arXiv record confirms ICML 2026 acceptance; A's T2 label conflicts with that record and E. | Reuse the template/gate idea, not a claimed proven reader schedule. |
| A/C/D: only observed-graph methods exist, or no paper reads a relation token | Those blanket statements conflict with L3-PPI, the attribute-only methods elsewhere in these reports, and LPFormer's explicit pair representation. None establishes our exact two-reader architecture and evaluation contract. | State the narrower evidence gap; no exhaustive novelty claim. |
| A/B/C/E: an ablation establishes which ingredient is universally responsible | Benchmark-specific ablations, restoration errors, node classification and graph explanation results do not directly order components for node-disjoint PPI. In particular, edge-deletion restoration is not edge prediction. | Use these as reasons for controls, not predicted effect sizes. |

Other important limits: MinCutPool degeneracies and standard virtual-node Jacobian results concern their
own objectives/architectures; they do not identify the cause of our attachment collapse. EHDM's
teacher result is an existence/equivalence statement under its formalization, not a bound on the
benefit of every practical teacher. Results on other datasets cannot bound attainable PPI accuracy.
GraphSMOTE's variants change multiple ingredients, so their comparison does not isolate discreteness.
The reviewed GSAT results motivate an alternative but do not justify adding an information-bottleneck
loss as a required component here.

## 3. What the local probes actually establish

The main numerical rows quoted in the synthesis agree with the saved JSON. The split's loopless
counts agree with `docs/03-experiments.md` §3.1: train 31,623 edges, V_val 4,703, test 30,128.
These differ from positive-pair counts that include self rows; both conventions must stay explicit.

| Probe | Evidence class and limitation |
|---|---|
| `template_stats.py` | True-graph, query-edge-removed diagnostics. Per-universe logistic CV uses that universe's labels and shared proteins. Structural association does not establish feature predictability. |
| kNN transfer in `template_stats.py` | A particular retrieval-based baseline using pooled F0 features and training adjacency. It is **not** an information ceiling, even among functions of those features; a learned model can use the same input better, and our generator sees residue states. It is also outside the proposed inference support condition. |
| `complementarity_test.py` | Five-fold **test-pair** CV fits logistic models to test labels; proteins recur across folds and scaling is fitted before the fold split. The +0.024 AUROC is exploratory test-informed complementarity, not a deployable held-out improvement. The trunk-alone row uses raw logits while combined rows fit a readout. |
| `block_readout_test.py` | Within-universe CV, not node-disjoint generalization. Neither 0.826 nor the comparison to a separately evaluated trunk establishes a usable advantage. |
| `cross_universe_readout.py` | Scaler/readout fit on training rows, then applied to unseen universes. Better boundary than within-universe CV, but still training-node retrieval. Training kNN excludes the query node, **not the queried edge from the whole adjacency**; it is not a target-edge-masked structural training control. It does not test the proposed parametric generator. |
| `allpairs_templates.py` | Whole-universe, nonself oracle ranking. Different row universe, self-loop convention and operating point from official BFS-macro metrics. Precision@E uses the universe's true edge count and stable index-order tie breaking. It is not the V_val-frozen official GS. |
| kNN `near_pairs_only` in `allpairs_templates.py` | Exclude from synthesis: the mask uses `(Aq^3)[u,v]` without removing the queried edge. Backtracking walks make essentially every positive appear near. This is inconsistent with the correctly masked true-template near-pair mask. The `all_pairs` rows do not use this mask. |

For a set with exactly E predictions and E positives, precision@E equals Dice **on that same set**.
That algebra does not make whole-universe precision@E comparable to reported BFS-macro GS.

Checked the vectorized true-template formulas against explicit edge deletion on 24 small synthetic
graphs / 744 unordered pairs: RA, AA, common neighbours, Jaccard, simple L3, degree-normalized L3
and L3 density agree within 8.9e-16. This verifies their algebra, not the dataset provenance or a
model result. The check extracted only the pure function; it did not execute the dataset script.

The existing exploration has already examined test labels and topology. This cannot be undone by
writing a new specification. The decision below uses literature plus training/V_val evidence, excludes
test-derived hyperparameter choices, and discloses prior exposure. A later result on this same test
set is not a fresh confirmatory test of this architecture; a fresh node-held-out confirmation is
needed for that stronger claim. Do not silently change the benchmark split in this design task.

## 4. Code feasibility checks

- `GritGmtEncoder` already computes dense soft RRWP and supplies typed adjacency and weighted degree.
  It uses per-token LayerNorm, not the original paper's default BatchNorm; paper-level degree-retention
  claims do not automatically transfer to this adapter.
- It currently returns node/PMA tokens and discards final pair states. Its final layer disables
  `O_e` and edge normalization because those outputs were unused. A native relation readout needs
  an explicit pair-state output and an active final edge update; the current API cannot be cited as
  already providing `topo_rel`.
- `topo_prompt.py` and `prefix.py` implement separately normalized, tanh-gated prefix residuals at
  the nine cross-attention sites. Zero gates preserve the base computation; concatenating extra keys
  into the content softmax would not give that property.
- Graph-index results were stale for newer prompt modules, so the relevant current source was read
  directly. No training code or config was changed.

## 5. Verification scope

Primary passages checked locally: GRIT, L3-PPI, Labeling Trick, Does Graph Prompt Work?,
edge-independent graph limits, EHDM, LPFormer and All-in-One. The current arXiv records for the
first five relevant theorem/mechanism papers were opened for identity/venue checks. No new papers
were searched or downloaded. Other bibliography rows remain agent-extracted evidence; numerical
claims not needed for the decision are not promoted to independently confirmed facts.

AI-assisted source review and synthesis were used. No new model has been trained or evaluated.

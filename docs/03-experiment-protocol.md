# Experiment Protocol: Topology-Conditioned Inductive Edge Prediction

**Status (2026-08-13):** current forward protocol and selected evidence. Method
selection remains open. Historical gates are archived by reference rather than repeated.

## 1. Locked task contract

Let

$$
G_{\mathrm{train}}=(V_{\mathrm{train}},E_{\mathrm{train}}),
\qquad V_{\mathrm{train}}\cap V_{\mathrm{test}}=\varnothing.
$$

For a queried pair $u,v\in V_{\mathrm{test}}$, the model receives exactly the
frozen intrinsic features $(x_u,x_v)$ and predicts

$$
\widehat A_{uv}=P(Y_{uv}=1\mid x_u,x_v)=\widehat A_{vu}.
$$

No observed test edge, neighbor identity, retrieval result, degree, or graph statistic
is task input. Training topology may supervise a representation or objective, but every
fair endpoint-only method must infer its test-time context from $(x_u,x_v)$.
Retrieval-grounded scaffolds are a separately labeled support condition outside this
strict endpoint-only input contract and must disclose their support universe.

After scoring a pair universe $\mathcal Q$, predictions are assembled only for
evaluation:

$$
\widehat G_\tau=(V_{\mathrm{test}},
\{\{u,v\}\in\mathcal Q:\widehat A_{uv}\geq\tau\}).
$$

The task is binary edge prediction. Inferred topology is intermediate context or a
joint-output constraint, not the prediction target and not generic graph generation.

## 2. Evidence and evaluation contract

### 2.1 Evidence classes

- **Comparator:** a frozen model or score artifact evaluated without changing its
  checkpoint or opening new test-dependent choices.
- **Formal result:** produced by a fixed pre-test protocol and eligible for the paper's
  scientific claims.
- **Diagnostic result:** consumes hidden topology, follows test opening, or tests a
  post-hoc mechanism. Diagnostics measure headroom or mechanism behavior and are always
  `formal:false`.

A completed run is not automatically a successful method. Process completion,
engineering validity, and the scientific verdict are reported separately.

### 2.2 Universes and operating points

- Training and test nodes are disjoint. `V_val` is a pair-disjoint (not node-disjoint) BFS-grown internal
  region: cross-boundary edges train; only V_val-internal pairs are withheld, never fully inductive, unlike test.
- Pairwise metrics use the fixed benchmark test-pair artifact; AUROC and AUPRC are threshold-free; Accuracy, F1, and MCC use threshold 0.5.
- Fixed-threshold topology first scores only the V_val 20--200-node sampled-set pair union. Across
  every atomic logit tie-group boundary, size-stratified paired-bootstrap 1-SE sets maximize BFS-macro
  GS, then minimize mean `|RD-1|`; degree, clustering, then spectral MMD ratio select lexicographically,
  with a larger-threshold complete-tie break. The threshold freezes before test and replays unchanged.
- Test topology scores only its sampled-set pair union plus support-only rows for full grounding.
  Report the deployable fixed-threshold result first; separately report per-subgraph exact-edge-count
  RD=1 matching (self-loops included, canonical tie-break) as an oracle-calibrated diagnostic.

### 2.3 Required metrics

Report edge and topology metrics together. Each fixed or diagnostic topology result has five numbers:

1. BFS-macro graph similarity (GS, edge-set Dice/F1; higher is better);
2. BFS-macro relative density (RD; closer to 1 is better);
3. degree-MMD ratio (lower is better);
4. clustering-MMD ratio (lower is better);
5. spectral-MMD ratio (lower is better).

The v3 test protocol reports BFS-macro GS/RD only; global-simple rows are legacy v2 evidence.
Never aggregate the three MMD ratios or call an MMD aggregate “graph similarity.” Descriptors retain self-loops;
the MMD denominator is the deterministic real-vs-real floor, so ratio 1 is that floor.
Detailed edge tables also retain ECE, Brier score, class balance, uncertain-negative
disclosure, and the completed easy, hard, degree-corrected, full-universe, and PA-null
controls. These qualify the edge claim even when they are not headline columns.

## 3. Completed baseline and topology oracle

The frozen pairwise baseline establishes the pair-to-topology gap. The held-out
true-topology Oracle diagnostic row is an upper bound that reads hidden topology and
therefore violates the task contract.

| Model | Evidence | AUROC | AUPRC | Accuracy | F1 | MCC |
|---|---|---:|---:|---:|---:|---:|
| Pairwise baseline | Frozen comparator | 0.7067 | 0.7315 | 0.6083 | 0.3987 | 0.3020 |
| Token-XAttn Oracle | True-topology diagnostic | 0.8524 | 0.8698 | 0.7713 | 0.7599 | 0.5451 |
| Full-Ego Pooled Oracle | True-topology diagnostic | 0.9356 | 0.9396 | 0.8521 | 0.8451 | 0.7071 |

| Model | Global GS | Global RD | BFS GS | BFS RD | Degree | Clustering | Spectral |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pairwise baseline | 0.1365 | 0.9977 | 0.3896 | 0.4223 | 13.08 | 11.93 | 18.09 |
| Token-XAttn Oracle | 0.3076 | 1.0000 | 0.4886 | 0.7542 | 5.825 | 5.288 | 9.302 |
| Full-Ego Pooled Oracle | 0.4635 | 1.0000 | 0.5904 | 0.9027 | 5.434 | 5.176 | 11.668 |

**Finding.** Full-Ego Pooled Oracle gives the strongest edge discrimination and GS while
matching global density, but its MMD ratios remain far above the real-vs-real floor and
its spectral ratio is worse than Token-XAttn Oracle. Hidden relational topology supplies
substantial headroom, but neither Oracle is a formal result or a fair baseline.

Exact identities: the baseline is frozen V3.1 checkpoint `e092537d8cf1e208` from
`outputs/deliverables/b0_v31_breadth_first_20260711`. **Token-XAttn Oracle** is
checkpoint `48f686df9029cf63`: a fixed oracle scaffold encoded by GRIT and consumed by
token cross-attention. **Full-Ego Pooled Oracle** is checkpoint `926dab5c82beca55`
(epoch 2): variable-length full ego topology is pooled inside GRIT and passed through a
pooled adapter. Its completed optimized evaluation uses threshold 0.5 for classification
and a 30,128-edge global density-controlled topology assembly. This is legacy pre-v3
evidence, not comparable to the current per-subgraph RD-matched protocol. Existing baseline /
Token-XAttn page and report: [results/results.html](results/results.html) and
[diagnostic_test_report.json](../outputs/egostitch_e2e_stage1_v3/oracle_grit_xattn_tokens_true_oracle_diagnostic/diagnostic_test_report.json). Full-Ego report: `outputs/full_ego_oracle_scoring_optimization_20260813/protocol/diagnostic_test_report.json` on the H20 checkout.

### 3.1 Archived completed evidence

These results remain citable but are not active gates or candidate methods:

| Evidence | Verdict | Record |
|---|---|---|
| G1: B0, B0-alt, PA-null, hard/degree-corrected controls | Pair-to-topology gap survives architecture and negative-regime checks | [E2 report](results/E2-pair-to-topology-gap.md) |
| G2: edge-independence ceiling | Edge-independent allocation has measurable structural limits | [E2 report](results/E2-pair-to-topology-gap.md) |
| G3: `oracle_topo` and `oracle_blend` | Hidden relational topology provides assembled-graph headroom | [E2 report](results/E2-pair-to-topology-gap.md) |
| EgoStitch engineering screens | Prior registered screens were cut; no current formal result | Dated records under [results/](results/) |

## 4. Feature-to-topology diagnostics

### 4.1 S0 — per-node summary value

S0 asks whether endpoint features predict degree, clustering, ego-edge count, and ego
density, and whether those summaries help a `V_hold` pair classifier.

| Pair head | AUPRC | Change from features only |
|---|---:|---:|
| Features only | 0.1197 | — |
| Oracle node summaries | 0.1434 | +0.0237 |
| Feature-predicted summaries | 0.1231 | +0.0034; CI includes 0 |
| Oracle summaries + CN/AA | 0.3653 | +0.2220 over oracle summaries |

Feature predictability was uneven: degree $R^2=0.435$; degree-partialled ego-edge
$R^2=0.284$; clustering $R^2=0.051$; ego density was effectively unpredictable.

**Finding.** Marginal node summaries have little pairwise value. Most oracle headroom
comes from pair-relational common-neighbor and Adamic–Adar information.

Artifact: [outputs/s0/s0_results.json](../outputs/s0/s0_results.json).

### 4.2 S0-R — relational predictability and transfer

S0-R tests whether $(x_u,x_v)$ predicts CN/AA beyond degree and whether appending
feature-predicted CN/AA improves the held-out pair head.

- On 504,914 training-side pairs, feature-predicted `CN>0` reached AUPRC 0.298,
  below the oracle degree-product control at 0.346.
- Feature-predicted CN/AA changed `V_hold` AUPRC by $2.3\times10^{-8}$.
- Oracle CN/AA raised AUPRC from 0.1197 to 0.3634: +0.2438,
  95% CI [0.2163, 0.2706].

**Finding.** The relational ceiling is large but is not recovered as extra per-pair
information from endpoint features. A deterministic predicted statistic is already a
function of $(x_u,x_v)$; it can shape optimization but cannot add information beyond
the features.

Artifact: [outputs/s0r/s0r_results.json](../outputs/s0r/s0r_results.json).

## 5. Assembly-coherence diagnostic

### 5.1 S1-R — corrected degree-budget oracle

S1-R holds the frozen B0 scores fixed and tests whether node-coupled assembly improves
the graph. The original S1 “oracle” is superseded: it rank-matched a degree multiset,
fit soft IPF probabilities, and did not enforce realized node degrees. It was not an
oracle ceiling.

The corrected diagnostic reads each test node's hidden non-self degree and greedily
selects high-scoring edges under node-aligned quotas. Enforcement is approximate because
the greedy simple-graph assembler can leave residual quota. It is a ceiling, not a legal
method.

| Assembly arm | BFS GS | BFS RD | Degree | Clustering | Spectral |
|---|---:|---:|---:|---:|---:|
| Frozen B0, exact target count | 0.390 | 0.423 | 13.03 | 11.86 | 18.07 |
| True node-aligned degrees, hard quota | **0.439** | **0.622** | **4.06** | **5.44** | **7.65** |
| Rank-matched true multiset, soft IPF | 0.339 | 0.388 | 14.24 | 13.66 | 19.88 |
| Feature-predicted degrees, soft IPF | 0.407 | 0.412 | 15.92 | 14.00 | 21.34 |
| Training-prior degrees, soft IPF | 0.318 | 0.363 | 19.55 | 17.19 | 25.87 |
| Best bidirectional CN update | 0.395 | 0.427 | 14.58 | 12.39 | 20.48 |
| `V_hold`-fit logistic coupling | 0.374 | 0.423 | 13.14 | 13.06 | 17.02 |
| `V_hold`-selected CN update | 0.294 | 0.355 | 17.16 | 22.82 | 20.92 |

The hard oracle realized 29,640 of 30,128 requested non-self edges: 1.62% quota
shortfall, below the fixed 2% limit. It closed 73.2% of the frozen B0-to-oracle
clustering gap without changing the underlying pair scores.

**Finding.** Coherent global allocation of existing pairwise evidence can close much of
the pair-to-graph gap; additional pairwise relational evidence is not the only missing
ingredient. However, every tested legal post-hoc arm failed. The current result motivates
learning transferable node-specific budgets or other jointly trained non-factorized
structure; it does not show that such a legal method already works.

Scope: diagnostic only; bounds the tested post-hoc transforms of frozen B0. Jointly
trained non-factorized models remain untested. S1 and the headline table use the current
in-repo GS implementation; archived G1 used the official evaluator and reports B0 GS
0.312151. RD and MMD reproduce, but GS values across those evaluator provenances must not
be mixed until the implementation difference is reconciled.

Artifact: [outputs/s1/s1_results.json](../outputs/s1/s1_results.json).

### 5.2 S1-H — hard-enforcement decomposition

S1-H applies the S1-R hard-quota assembler to three degree sources with no node-aligned
identity, isolating hard enforcement from node identity; run verified complete on the H20.

| Hard-quota arm | BFS GS | BFS RD | Degree | Clustering | Spectral | Edge P/R | Shortfall |
|---|---:|---:|---:|---:|---:|---:|---:|
| True multiset, rank-matched | 0.340 | 0.392 | 12.57 | 13.26 | 17.31 | 0.080/0.080 | 0.08% |
| Feature-predicted degrees | 0.399 | 0.400 | 14.86 | 14.41 | 19.14 | 0.175/0.161 | 8.43% |
| Training-prior multiset | 0.329 | 0.370 | 15.47 | 14.76 | 21.09 | 0.070/0.067 | 4.02% |

**Finding.** With scores unchanged (candidate AUPRC 0.0659, AUROC 0.6713), every arm worsened
the clustering gap: closure −0.150/−0.280/−0.319 versus +0.732 for the S1-R node-aligned oracle;
predicted and train-prior shortfalls exceed the 2% limit (lower bounds only). The true multiset is
insufficient, no legal post-hoc arm exists, and node identity is the crux. Artifact: [outputs/s1/s1_hard_decomposition.json](../outputs/s1/s1_hard_decomposition.json).

## 6. Current experiment ladder

| ID | Experiment | Role | Current state |
|---|---|---|---|
| B0 | Independent endpoint-only scorer | Frozen comparator | Complete |
| Oracle | Observed-topology classifier | Method ceiling; protocol-violating | Complete, diagnostic |
| S0 | Oracle/predicted node summaries | Tests marginal topology value | Complete, diagnostic |
| S0-R | Oracle/predicted CN and AA | Tests relational predictability | Complete, diagnostic |
| S1-R | Coupled assembly and hard degree oracle | Tests assembly coherence | Complete, diagnostic |
| S1-H | Uniform hard quotas from three degree sources | Separates enforcement from identity | Complete, diagnostic |
| B1 | Training-time topology transfer | Logit control plus LLP-style relational distillation | Not run |
| B2/B3 | Deterministic or generative topology latent | Jointly trained non-factorized candidates | Not run |
| B4 | Retrieval-grounded scaffold | Separate extra-support comparison | Implemented; prior screens cut; no current formal result |

No candidate is `Ours` before matched endpoint-only comparison. B1 should precede a
costly latent generator: use pointwise logit distillation as the attribution control and
LLP-style relational score-geometry distillation as the primary transfer test. Any B2/B3
run must explain how its shared latent or structured objective changes joint edge
allocation while retaining the two-endpoint inference contract.

## 7. Decision and reproducibility rules

1. Report pairwise and all five topology metrics together; no favorable nearby
   threshold or aggregate substitutes for a fixed gate.
2. Preserve provenance. Oracle, S0, S0-R, S1-R, and any test-informed follow-up remain
   `formal:false` even when their execution is valid.
3. Select the fixed topology threshold on sampled `V_val`, freeze it, then evaluate
   test/test_topology once. Never tune model parameters or that threshold on test topology.
4. Bind comparisons to the same frozen features, split, sampled sets, full-node grounding support,
   self-loop convention, checkpoint, fixed-threshold rule, and diagnostic RD=1 policy.
5. Validate score precision before analysis. Record checkpoint ID, artifact hashes,
   threshold/quota policy, random seed, code commit, and metric implementation.
6. Require completion markers, no `failure.json`, exited workers, complete outputs, and
   verified hashes before declaring an HPC experiment complete.
7. A selected method must improve assembled topology without an unacceptable edge-metric
   loss and must survive its topology-context or coupling ablation.

The S0, S0-R, and S1 artifacts retain `evidence_class=diagnostic`, the
`breadth_first` strategy, and their input-manifest digests; these records must remain
attached to any derived table.

## 8. Current deliverables

- Frozen baseline and topology-oracle comparison: [results/results.html](results/results.html).
- Pair-to-topology baseline evidence: [results/E2-pair-to-topology-gap.md](results/E2-pair-to-topology-gap.md).
- S0, S0-R, corrected S1-R, and S1-H diagnostic artifacts linked above.
- Matched B1/B2/B3 comparison before selecting a project method.

Retired registrations, abandoned EgoStitch build waves, legacy arm matrices, and obsolete
gate machinery are intentionally omitted. Their immutable evidence remains in dated result
artifacts and version history, not in the current protocol.

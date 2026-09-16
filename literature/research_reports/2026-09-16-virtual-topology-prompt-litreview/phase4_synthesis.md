# Synthesis and architecture decision

Date: 2026-09-16. Scope: finish the interrupted A–E review and choose the next architecture.
Read [verification and corrections](phase3_verification.md) before using any original report as evidence.
The concrete deliverable is the [motif graph–GRIT specification](../../../docs/superpowers/specs/2026-09-16-motif-graph-grit-prompt-design.md).
This selects a research candidate; it does not report an implemented model or establish the paper's winner.

## 1. Decision

Adopt both requested changes together: replace the coordinate interface with a **small weighted
template graph → GRIT → three topology tokens → existing gated pair reader**. Replace the unrestricted
coarse block graph with two explicit motifs: **shared-neighbour wedges** and **length-three bridges**.
The queried edge is always absent, so a triangle involving that edge is represented by its remaining
two-edge wedge, never by including the answer in the input.

The learned generator has a residue-attention front end and a small shared GNN gate network.
It predicts the placement/strength of allowed template edges. GRIT reads that adjacency directly.
There is no intermediate degree/Jaccard/clustering/L3 coordinate decoder. The content branch remains
the published `prefix_base`; topology changes its internal cross-attention through the existing
separate prefix residual, rather than replacing its feature-based decision.

Keep `topo_rel`. Read it from GRIT's two directed endpoint-pair states after symmetrization, not from
three anonymous pooling seeds. `topo_u` and `topo_v` are the two designated endpoint node states.
All three arise from a graph that already identifies the queried pair. This is an architectural
choice supported by GRIT's pair channel and the labeling-trick motivation, not a published guarantee
that three tokens are optimal. [GRIT](https://arxiv.org/abs/2305.17589),
[Labeling Trick](https://arxiv.org/abs/2010.16103).

## 2. What each thread contributes after verification

| Work | Retained lesson | What it does not establish |
|---|---|---|
| A: graph prompting | L3-PPI supplies a concrete template + query-conditioned gate + pretrained reader pattern; conditional prefixes provide a way to preserve content computation. | Shared templates are not theorem-proven to fail; prompting benchmarks do not validate our strict interface. |
| B: PPI templates | L3 is biologically motivated; closure and bridge signals depend on dataset and evaluation regime. Use a small explicit vocabulary. | Neither L3 nor closure is a universal positive-edge rule. |
| C: graph readers | GRIT exposes node and pair states; relative walks and weighted adjacency encode the graph the reader actually receives. | Stronger reading cannot create input information or guarantee transfer through erroneous soft edges. |
| D: cold start | Feature-to-structure transfer and structural KD are plausible, but teacher accuracy and student teachability differ. | Existing kNN/MLP probes and other datasets are not upper bounds on our generator. |
| E: discriminative structure | Amortized gates and task supervision can learn task-relevant structures; density and label shortcuts remain credible alternatives. | A path-count hinge, stochastic gates, or an IB regularizer is not independently established as necessary here. |

The especially close precedents are [L3-PPI](https://arxiv.org/abs/2605.09964) for template/gate/reader
separation, [LPFormer](https://doi.org/10.1145/3637528.3672025) for pair-aware structure reading, and
[NCNC](https://arxiv.org/abs/2302.00890) for using uncertain predicted structure. The latter two use
observed graph context and therefore are mechanism precedents, not strict endpoint-only baselines.
[EHDM](https://arxiv.org/abs/2504.06193) motivates judging the student on unseen nodes rather than
selecting a teacher solely for its true-graph accuracy.

## 3. Data evidence used for the decision

These are saved diagnostic measurements, checked against JSON, not new runs.

| Evidence | Value | Interpretation |
|---|---:|---|
| Training-pair true Jaccard / normalized L3 AUROC | 0.887 / 0.899 | Both motif families are reasonable candidates before test inspection. |
| V_val true RA / raw L3 all-pairs AUPRC | 0.638 / 0.476 | Degree-penalized common-neighbour support is useful in sparse ranking. This is an oracle diagnostic, not official GS. |
| V_val true normalized L3 AUROC on classification pairs | 0.903 | Preserve bridge structure rather than replacing it entirely with closure. |
| Training-fitted K=256 counts / attachment-vector logistic readout, V_val AUROC | 0.710 / 0.747 | Collapsing structure to aggregate counts can lose usable information in this retrieval probe. It does not quantify GRIT's possible gain. |
| Within-V_val K=256 attachment MLP / training-fitted attachment MLP AUROC | 0.826 / 0.732 | Within-universe pair CV is too optimistic for architecture selection under node holdout. |

Source: `docs/tmp/template_discriminativeness/{template_stats,allpairs_templates,cross_universe_readout,block_readout_test}.json`.
The cross-universe probe has the training-target-edge limitation described in the verification report;
its numbers are descriptive and cannot qualify a strict structural target pipeline.

The test-pair CV improvement previously described as a +0.024 AUROC complement is excluded from the
case for deployment. It is a useful exploratory observation, but that model was fitted to test labels.
Nor does all-pairs precision@E establish an official topology gain. Prior test exposure is disclosed
in the specification and must remain visible in subsequent reporting.

## 4. Why this graph instead of the alternatives

**Keep actual connectivity.** A shared-neighbour slot can connect to one endpoint, both, or neither;
a bridge requires endpoint attachments and an intermediate edge. This gives the reader connectivity
patterns, not only a scalar stating that a motif exists. The graph is a motif expansion with anonymous
role copies, not a reconstruction of named training proteins.

**Bound the space by roles.** Eight shared-neighbour slots plus eight slots on each side of a bridge
give 26 nodes and 96 possible undirected edges, versus learning an arbitrary dense block matrix.
Eight is a small starting budget, not a data-derived optimum. The bridge slots share intermediates
across paths, avoiding both the independent-path assumption and L3-PPI's ambiguous shared-edge gate.
Each actual edge gets one gate.

**Do not transmit content twice under a topology name.** The gate network reads residues, but R's node
inputs are role embeddings only. Pair-dependent residue summaries cannot bypass the graph by entering
R as unconstrained node features. The same content states reach F through its original pathway.
This restricts the topology branch without pretending it creates new information.

**Use the data to teach a reader, then teach the generator through that reader.** Stage I uses bounded,
query-edge-removed motif graphs from the legal training graph. Stage II gets a task gradient through
R plus representation targets from an immutable copy of the Stage I reader. This avoids arbitrarily
matching anonymous generated slots to real protein identities. It also avoids forcing every positive
to have many motifs and every negative to have few. L3-PPI's count hinge can be informative as an
ablation but is not the selected objective.

**Acknowledge what is being learned.** Task loss and reader matching do not identify a unique graph.
The intended claim is that a constrained, learned structural computation helps the queried edge,
not that its anonymous virtual witnesses recover the true neighbours. The graph can still encode a
scalar shortcut; a density-only control and connectivity interventions are necessary to assess that.

## 5. Training and decision boundaries

Train Stage I R and the prefix adapter against the frozen published content reader/head. Use the
same motif compiler for clean graphs and stationary soft-corruption training. The clean oracle
reader is a diagnostic only. In Stage II, warm the generator against frozen R/adapter for two epochs,
then adapt only R's last block/readout and the prefix adapter at a lower learning rate for the remaining
13 epochs. The frozen-forever reader is the direct schedule control. No paper establishes the best
unfreezing schedule at our scale; these are explicit, falsifiable defaults.

The objective is task BCE + sampled-subgraph BCE + a light, training-only topology-token alignment
through the immutable teacher. Do not carry over coordinate regression, attachment-to-community
targets, calibration maps, or the entire old D-arm objective by accident. Do not add stochastic
sampling or an information-bottleneck objective before a failure requires that extra machinery.

Select checkpoints with the existing V_val five-metric ranking and replay its topology threshold
unchanged on test; keep the classification threshold separate. Report the complete edge panel and
GS/RD/degree/clustering/spectral MMD. Existing diagnostic topology access does not authorize any truth
graph in the deployable scoring forward.

## 6. What would support or defeat this choice

The strongest positive result would be an unseen-node gain over `prefix_base` and a matched
direct-prompt control, with improved GS/MMD at the prescribed threshold, plus a specific loss of gain
when pair-conditioned edge placement is disrupted while density is retained. Stage I success alone
only validates the reading interface. If the student remains at base, the feature-to-structure transfer
problem remains unsolved. If a density-only or direct vector prompt matches it, graph connectivity
has not earned the additional mechanism. If closure-only matches the two-family graph, remove L3
from a later simplified model rather than calling the second family necessary.

No reviewed source or saved probe establishes success for the chosen architecture. Numerical,
transfer and anti-shortcut tests are specified as future implementation/evaluation work; none has
been launched. The existing test universe has been explored and cannot be described as untouched.

AI-assisted verification and synthesis; original A–E reports retained as provenance.

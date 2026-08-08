# Full-Ego True-Oracle Diagnostic — Design

**Date:** 2026-08-08  
**Status:** diagnostic implementation  
**Config:** `configs/egostitch_e2e_v3_full_ego_oracle_grit_pooled_breadth_first.yaml`  
**Implementation:** `src/model/egostitch/generator/full_oracle/`

## 1. Question and claim boundary

This experiment asks whether the existing GRIT encoder and pair classifier can
consume the complete true one-hop topology around a queried pair when the K=16
EgoStitch representation limit is removed. Its primary comparison separates two
ceilings:

1. `oracle_struct` + `grit_gmt` + `pooled_adapter`: the ceiling inside the
   duplicated, sided K=16 slot scaffold; and
2. `full_ego_oracle` + `grit_gmt` + `pooled_adapter`: the ceiling when GRIT sees
   the complete deduplicated induced ego graph.

This is a **system-ceiling comparison, not a pure slot-count ablation**. The K=16
arm represents two sided slot sets plus alignment and closure relations, whereas
the full arm represents one deduplicated induced graph. A pure-K study would use
the same induced-graph builder on both sides and truncate only its neighborhood;
that study is intentionally out of scope.

The experiment cannot claim protocol-clean inductive improvement, learned graph
generation quality, or that any observed difference is caused only by neighbor
count. It consumes held-out structural truth and is always `formal: false`,
diagnostic only. The task output remains binary classification of
`edge(u, v)`; the local graph is intermediate context, never a generated output.

## 2. Exact formulation

Let `G_R` be the undirected, self-loop-stripped truth graph authorized for role
`R`, and let the unordered query be `q = {u, v}`. Remove the query edge before
computing either neighborhood:

$$
G_q^- = G_R - \{\{u,v\}\}.
$$

Then define

$$
U_q = \{u,v\}\;\cup\;N_{G_q^-}(u)\;\cup\;N_{G_q^-}(v),
\qquad
H_q = G_q^-[U_q].
$$

`FullOracleGenerator._query_graph` constructs `H_q`; `FullEgoGraph` batches the
result. Every remaining ground-truth positive edge with both endpoints in `U_q`
is present, including endpoint-to-neighbor, neighbor-to-neighbor, and cross-side
edges. The queried edge is absent even for a positive example. Shared neighbors
are deduplicated into one node and carry both neighbor-role indicators. There is
no neighbor cap, sampling, multiplicity approximation, alignment plan, or
closure-edge synthesis.

The five input channels mark source endpoint, destination endpoint, source
neighbor, destination neighbor, and valid-node presence. Ground-truth node IDs
are used only to construct and deduplicate `H_q`; no identity, protein ID, or
oracle label is passed as an encoder feature. `grit_gmt` encodes the adjacency
and these role features; `b0_v31` combines the pooled topology representation
with the ordinary pair features and emits one edge logit.

Self-pairs use one endpoint node and the same loopless construction. Source /
destination swapping changes only role channels, through
`FullEgoGraph.swapped`, not the underlying undirected adjacency.

## 3. Truth graphs and leakage boundary

Truth is role-specific; no graph is a universal oracle table.

| Role | Authorized loopless `G_R` | When accessible |
|---|---|---|
| Training | `G_fit`, from the legal training structural graph | Training only |
| Validation | `G_hold`, the internal `V_hold` positive graph | Diagnostic validation |
| Test / candidate | self-loop-stripped benchmark `test_graph` | Diagnostic scoring only, after explicit opt-in |

The training process may bind the disjoint `G_fit` and `G_hold` components in one
runtime context (`generator.oracle_truth_source: g_fit_plus_v_hold`), but a query
can see only the component for its role. The access audit must record this truth
source and its digest.

The following are forbidden topology sources: `train_graph.pkl`, the benchmark
global `graph`, a generic `positive_edges` collection, and any artifact that
mixes roles. Test or candidate truth must never be read during training,
validation, checkpoint selection, or publishing. It is loaded only by the
diagnostic scoring path, which must require the explicit oracle-diagnostic flag
and stamp the score artifact `diagnostic_only: true` and `formal: false`.

## 4. Fixed architecture and comparisons

The only full-ego arm is:

`full_ego_oracle` → `grit_gmt` (4 layers, width 96, RRWP K=8, 4 PMA seeds) →
`b0_v31` with `conditioning_mode: pooled_adapter`.

`pooled_adapter` is fixed because graph size is query-dependent. PMA performs
mask-aware pooling inside the graph encoder, so the classifier never needs a
variable-length topology-token batch whose padding policy could itself change
the result. There is no fusion ladder or cross-attention variant in this
experiment.

Report the full arm against both controls under matched split, pair stream,
seed, optimizer, classifier, and evaluation settings:

- K=16 `oracle_struct` + `grit_gmt` + `pooled_adapter`, configured by
  `configs/egostitch_e2e_v3_oracle_grit_pooled_adapter_breadth_first.yaml`;
- the same full-arm checkpoint's `f_logit` hard-bypass scores as the primary
  matched topology-null. They hold the pair stream, checkpoint, and pairwise
  parameters fixed while clamping topology off. They are an inference-side
  consumption control, not a separately trained null-generator arm. A
  historical B0 result with different training settings is context only, not
  the matched null.

## 5. Compute contract

For one query,

$$
|U_q| \le 2 + \deg_{G_q^-}(u) + \deg_{G_q^-}(v),
$$

with overlap reducing the actual size. GRIT constructs RRWP using repeated
dense matrix products, so that stage takes `O(K|U_q|^3)` time and
`O(K|U_q|^2)` memory; subsequent dense attention is quadratic in `|U_q|`.
The measured H20 operating point is
`data.edge_batch: 16` with 8-way accumulation per rank: this preserves the
logical batch of 128 pairs and its one exact global weighted-BCE denominator.
The registered diagnostic schedule is 10 epochs to target the fixed train/eval
budget. Training throughput alone leaves about 3.4 hours for validation and
overhead, so the run is not known to fit until its first full B16 validation is
timed. Any matched K=16 comparison must be rerun with the same B16 x 8 physical
batching and 10-epoch optimizer schedule; a historical B128/30-epoch result has
different pair-to-dropout assignment and is not a matched control.
`runtime.token_budget` governs endpoint feature-token packing, not the number
of nodes in `H_q`, and is not a full-graph memory control. LR schedules, phase
boundaries, clipping, and diagnostic probes advance on optimizer steps, not
physical batches. Scoring remains singleton. There is deliberately no neighbor
cap or emergency truncation: an OOM is a failed engineering run, not permission
to silently change the experiment.

Every result must report the observed `|U_q|` distribution (at least p50, p90,
p95, p99, and maximum), wall time, and peak memory. Metrics must also be shown
for the high-degree stratum in which at least one endpoint has more than 16
neighbors after query removal, alongside its row count. This identifies where
the full arm actually receives information unavailable to K=16.

## 6. Metrics and interpretation

Edge-level and assembled-graph results are always reported together. At minimum
report the standard edge metrics led by AUPRC, plus the complete topology family:

- GS and RD, with global simple-edge and BFS-macro variants named separately;
- degree, clustering, and spectral MMD ratios, individually and never collapsed
  into one score.

Report the same metrics for the full population and the high-degree (>16)
stratum. Full-versus-K16 lift supports the narrow claim that the existing
full-induced-graph system has a higher topology-consumption ceiling. It does not
identify whether the cause is retained neighbors, deduplication, induced edges,
or removal of the slot/alignment representation.

## 7. Acceptance gates

The following are pre-run gates, not claims satisfied merely because the
generator and config exist. Implementation acceptance requires tests that
establish all of the following:

1. A positive query edge is removed before neighborhoods are computed and never
   appears in `adj`; a negative query receives the same construction rule.
2. The emitted adjacency contains exactly all and only edges of `G_q^-[U_q]`.
3. A shared neighbor appears once, receives both role indicators, and its
   incident induced edges are retained.
4. More than 16 neighbors are retained without sampling or truncation; padding
   is masked and cannot change a smaller graph's encoding.
5. Context installation rejects directed graphs, self-loops, duplicate node
   rows, and row IDs absent from the truth graph; truth-graph-only neighbor
   nodes are allowed because they need not be scored endpoints. Missing
   endpoint rows fail closed.
6. Source/destination swap preserves adjacency and exchanges role channels.
7. The generator is registry/config constructible, has no auxiliary loss, and a
   CPU smoke run completes forward, loss, and backward through GRIT/classifier.
8. `g_fit_plus_v_hold` is rejected outside `run_kind=diagnostic`; test truth is
   unreachable from training and requires scoring-time oracle opt-in.
9. Produced metadata records `full_ego_oracle`, the role-specific truth source
   and digest, `diagnostic_only: true`, and `formal: false`.

Scientific interpretation is allowed only after the full arm, matched K=16 arm,
and matched null complete without non-finite state or data-boundary violations,
and after edge metrics, all topology metrics, graph-size statistics, and the
high-degree stratum are reported together. No fixed performance win is an
engineering acceptance gate; a weak or null result is still a valid diagnostic
outcome.

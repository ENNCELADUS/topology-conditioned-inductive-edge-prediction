# Virtual topology prompt: a coarsened-graph generator behind the v1 coordinate interface

**Next architecture decision (2026-09-16):** [motif graph prompts read by GRIT](2026-09-16-motif-graph-grit-prompt-design.md).
That separate candidate replaces coordinates with graph-derived tokens. It is not implemented;
this document continues to describe the existing coarsened-graph arms and checkpoints.

**Date:** 2026-09-15
**Version:** v0.6 (supersedes v0.5 of 2026-09-15; change log in §12)
**Status:** v0.5 ran and was diagnosed at epoch 8 (`docs/tmp/2026-09-16-virtual-prompt-diagnosis-and-revision.md`);
v0.6 implements that note's point-1 fix (A-E) and relaunches as `virtual_prompt_d_rev1`. No scientific results yet.
**Reads on:** `docs/superpowers/specs/2026-09-13-topology-prompt-two-stage-v2-design.md` (the v2 recipe this design plugs into),
`docs/results/topo_prompt_stage2_verdict/README.md` (the Stage II verdict), `docs/results/topo_prompt_stage2_curves/` (the probe dump).

## 1. Decision

**Replace the Stage II coordinate generator, and shrink the surrogate set it must predict.** The virtual graph is a
learned coarsening of the training graph. At inference the two queried proteins attach to it, a query-conditioned
gate opens or closes each coarse node, and a compact set of **eight continuous fixed-semantics topology surrogates plus three distance indicators** (spec `v2` of
`src/data/struct_coords.py`, §3.0) is computed from the resulting small weighted graph by closed-form counting. Those
surrogates enter a Stage I reader retrained on the same compact spec through the existing prefix interface, now three
tokens (self endpoint, partner endpoint, relation), exactly as the v2 student's coordinates do.

Forward:

```
H_u = E(x_u), H_v = E(x_v)                                 frozen Stage I encoder, residue states
a_u = sigma(w . Attn(Q = P, K = V = H_u)) in [0,1]^K        each coarse node attends over u's residues
z_u = W [mean(H_u); max(H_u)],  z_v likewise                pooled summary, gate input only, d_z = 128
g_uv = sigma(phi(z_u, z_v, P)) in [0,1]^K                   per-node query gate, symmetric in (u, v)
G^P_uv = (roots u, v; coarse nodes P; multiplicities m; coarse adjacency B; attachments a_u g_uv, a_v g_uv)
s_hat_uv = Cal(Count(G^P_uv))                               8 continuous surrogates + distance class, closed form, affine
logit = Reader(H_u, H_v; Pi(standardise(s_hat_uv)))         Stage I reader on spec v2, v2 interface adaptation
```

What is reused unchanged: the Stage I recipe (corruption-trained reader, `topo_prompt_full_v2` config), the prefix
interface, the v2 Stage II objective (task BCE, coordinate loss, pointwise and anchor KD, structural stream with
subgraph BCE, rank, degree, motif), the 15-epoch schedule, five-rank selection, the scorer, and every intervention.
What is new: the compact coordinate spec and the Stage I teacher retrained on it, one generator module, its
initialisation from the training graph, one coordinate-loss correction, one scoring-time intervention.

Trainable in Stage II: generator (`P, B, m, W, the attachment attention, phi, Cal`), prompt interface, output head. Encoder and
cross-attention frozen, as in v2. The v0.2 request to train the encoder is not taken into this launch: the measured
Stage II peak is 61.8 GiB per rank with a frozen encoder, a trainable encoder doubles encoder work because the teacher
must encode separately, and the only field with a train-to-unseen fit gap is the endpoint field, which the reader
barely uses (§2). It remains a one-line change in the trainability table if the launch result asks for it.

## 2. Evidence this design answers

All numbers are from the published `coord_gen_full` checkpoint (`4a743373f0b9e4d0`, split seed 42, seed 0) on the
label-balanced probe dump, **nonself stratum** (self-pairs removed), recomputed 2026-09-15.

| Observation | Number | Design consequence |
|---|---|---|
| A linear reader fit on true training coordinates separates labels on true coordinates but not on predicted ones | AUROC 0.953 → 0.676 on V_val, 0.940 → 0.671 on test; the attribute-only trunk alone is 0.793 / 0.721 | The predicted coordinates carry less decision-relevant structure than the trunk already has. The generator, not the reader, is the component to replace. |
| Relation coordinates are weakly fit even on training rows | median in-sample R² 0.08; predicted spread is half the true spread on training rows | Regression to the conditional mean under weak identifiability. A generator that composes pair quantities from per-node attachments to a summary of the training graph brings graph knowledge a pair MLP has no structured way to hold. That is the bet. |
| Endpoint coordinates have a real train-to-unseen gap | Spearman 0.65 train → 0.2 V_val | Per-node attachment `a_u` through one shared attention layer is the regularised form; the encoder stays frozen. |
| The MLP generator read one global mean/max summary per protein | in-sample relation R² 0.08 with that input; no measurement isolates the pooling | Attachment reads the residue states directly: each coarse node attends over the protein's residues, so "which neighbourhoods does this protein attach to" is answered from the residues relevant to that neighbourhood, in the spirit of L3 interface complementarity. Pooling survives only as the gate's input. |
| Walk kernels are inverted on training rows | Spearman −0.54 (`walk_2`), −0.38 (`walk_3`) train; −0.45 / −0.42 V_val | Self-pairs set the standardisation scale (true `walk_2` self 0.15, nonself positives 0.02, negatives 0.00), so nonself rows are unsupervised. Fixed in §6: self-pairs leave the coordinate loss and the loss is scaled in nonself units. |
| Level shift between universes | true test coordinates sit 1–2.5 training sd above training level; predictions stay at 0.2–0.7 | Not fixable from the pair. Reported, not optimised. The calibration affine `Cal` absorbs coarse-versus-full scale, not universe density. |
| Stage I field importance | relation field carries the reader's gain (GS 0.542 without it); endpoint little; context nothing | Pair-specific structure is put where it matters: gates and the common/L3 quantities. |
| A compact surrogate set loses almost no label information in truth and is read far better from predictions | linear reader fit on true training coordinates, nonself: all 34 → V_val 0.953 / test 0.940 on truth, 0.676 / 0.671 on predictions; the ten of §3.0 → 0.938 / 0.922 on truth, **0.761 / 0.687** on predictions | The 23 dropped coordinates are the ones the generator emits as noise and a truth-trained reader misreads. Spec `v2` keeps degree and clustering per endpoint (density and closure, the per-node quantities behind degree and clustering MMD), common neighbours, Jaccard, L3, L3 density, and the distance class. |

The reader on true coordinates reaches V_val AUPRC 0.961 / GS 0.691, so the interface is not the bottleneck; Stage I
is retrained only because its input spec changes.

## 3. The virtual graph

### 3.0 Coordinate spec `v2`: eight continuous surrogates and a distance class

Measured on the universe's true loopless graph with the queried edge removed, as spec `v1`; `StructCoordinateTable`
gains a `spec` argument and `COORD_SPEC` is selected per checkpoint (`topo_prompt.coord_spec`, `coord_gen.coord_spec`).

| Field | Surrogates | Role |
|---|---|---|
| `endpoint_u`, `endpoint_v` (2 each) | `log1p_degree`, `clustering` | density and closure of each endpoint |
| `relation` (4 + 3) | `log1p_common`, `jaccard`, `log1p_l3`, `l3_density = L3 / (d_u d_v)`; distance one-hot `{2, 3, >= 4 or disconnected}` | the pair's shared and bridging neighbourhood |
| `context` | none | the Stage I ablation attributes nothing to it |

Prompt tokens: self endpoint, partner endpoint, relation; `slots_per_field 2` gives six prefix rows per layer at the
same nine sites. `FIELD_ORDER`, `FIELD_SLICES` and the generator's head widths follow the spec. Self-pairs keep the
`v1` conventions (no edge removed, Jaccard one, distance one-hot all zero, own class in the generator's distance
head). Spec `v1` and its checkpoints stay loadable and untouched.

### 3.1 Objects

- `P in R^{K x d_z}`: K coarse-node embeddings, `K = 64`, `d_z = 128`.
- `m in R_{>0}^K`: multiplicity, the number of training proteins coarse node `j` stands for. `softplus` parameterised.
- `B in (0,1)^{K x K}`: symmetric coarse adjacency, `B_jl` the probability that a protein of `j` links a protein of `l`;
  `B_jj` is the within-node density. `sigmoid` of a symmetric logit parameter.
- **Attachment attention** (pooling by multihead attention with the coarse nodes as seeds, the Set Transformer PMA
  pattern): queries `P` (`d_z`), keys and values a linear map of the residue states `H_u` (`d_model -> d_z`), four
  heads, padded residues masked, one layer, then `a_uj = sigmoid(s_uj)` with
  `s_uj = b_j + <P_j, o_uj> / sqrt(d_z)` and `o_uj` the attended output of coarse node `j` over protein `u`
  (**v0.6**, change B: one bias and one query direction per coarse node). A readout shared across coarse nodes,
  as in v0.5, makes `a_u` a per-protein scalar times a near-constant vector however the attention behaves: the
  measured interaction variance fraction was 0.0001 at epoch 1 and 0.025 at epoch 8. Attachment is a per-protein
  function. This implementation recomputes it for each pair occurrence;
  it does not yet cache trainable attachments across pairs. Cost per
  protein `K x L` attention scores, the same order as one cross-attention layer of the trunk over a single sequence.
- **`m` and `B` are frozen** at their training-graph values (**v0.6**, change E). They are measurements of the
  observed graph, not parameters, and under the coordinate loss alone they received gradients three to four
  orders of magnitude below every other group. `P` stays trainable.
- `W`: linear `2 d_model -> d_z` on masked mean and max pooled frozen encoder states (`CoordinateGenerator.pool`),
  the gate's input only.
- `phi`: shared MLP `[z_u + z_v; |z_u - z_v|; z_u * z_v; p_j] -> 1`, so `g_j(u, v) = sigmoid(phi(...))`, symmetric in
  `(u, v)` by construction. Output bias initialised to `+3` (gates start open, near 0.95).

Effective attachments `A_uj = a_uj g_j`, `A_vj = a_vj g_j`. The query edge `(u, v)` is absent. Swapping `u` and `v`
swaps `A_u` and `A_v` and leaves `g`, `B`, `m` unchanged, so endpoint surrogates swap and relation surrogates
are invariant.

Pair counts are expected counts over the proteins each coarse node stands for. Write `N_jl = m_j m_l` for `j != l`
and `N_jj = m_j (m_j - 1)` (two distinct proteins of the same coarse node).

### 3.2 Closed-form coordinates

Endpoint (`x in {u, v}`):

```
deg_x        = sum_j m_j A_xj
tri_x        = 1/2 sum_{j,l} A_xj A_xl N_jl B_jl
clustering_x = tri_x / (deg_x (deg_x - 1) / 2 + eps)
```

Relation:

```
common     = sum_j m_j A_uj A_vj
jaccard    = common / (deg_u + deg_v - common + eps)
L3         = sum_{j,l} A_uj A_vl N_jl B_jl
l3_density = L3 / (deg_u deg_v + eps)
```

Distance: four class logits (`2, 3, >= 4 or disconnected, self`) from a linear layer on
`[log1p common, log1p L3, log1p deg_u + log1p deg_v]`; the first three are assembled into the standardised one-hots
exactly as `V3_1CoordGen.assemble` does today, the self class leaving all three at zero.

Every quantity is a sum or ratio of expected counts under the K-block model; there is no soft-OR, no matrix power and
no approximation beyond the block model itself. Count-like surrogates take `log1p` as in the spec. `Cal` is a
type-wise affine `(s_i, t_i)` on the eight continuous surrogates (six learned scales/shifts: degree and clustering shared between endpoint roles, plus four relation types), initialised to identity, applied before the
reader's standardisation. All of this is float32 whatever the autocast context, as the current generator's outputs are.
For fractional multiplicities, within-block distinct-pair counts use `max(m_j(m_j-1), 0)`.
Clustering is zero at expected degree ≤ 1 and clipped to [0, 1] above it; the raw ratio
can otherwise be negative or exceed one for fractional degrees. Binary lifted graphs
retain exact counts. The compact vector has 11 columns (8 continuous + 3 indicators).
Each coarse node's readout bias is set in closed form from the coarsening itself (§3.3), so v0.6 needs no
second encoder pass and no scalar bisection.

### 3.3 Initialisation from the training graph

Once, on the main process at trainer startup, seeded, recorded in `run_metadata["virtual_graph"]`, saved in the
checkpoint:

1. Encode every node of the training universe with the frozen reader encoder and pool (the `coord_probe` loop);
   7,203 nodes, under a minute on one GPU.
2. k-means with `K = 64` on the pooled states (standardised per dimension), seed from the run seed.
3. `P_j = residue_projection(mean residue state of cluster j)` (**v0.6**, change C): the queries are seeded in
   the same space the keys live in, and the attention's query projection is tied to its key projection
   (`in_proj_weight[:d_z] = in_proj_weight[d_z:2 d_z]`, both biases zero), so a score is a similarity in one
   random projection instead of a dot product of two unrelated random maps. With v0.5's `P = W centroid_j` against
   keys `residue_projection(H)` every coarse node attended uniformly (entropy 0.9997) and received the same output
   (cross-node cosine 0.99997). `m_j = |cluster j|`,
   `B_jl = edges_train(j, l) / N_jl` clamped to `[1e-4, 1 - 1e-4]` before the logit, using the feature-present induced subgraph of the loopless legal
   training graph already built for `TopoPromptRows` (`StructCoordinateTable.adjacency`). Both are then frozen,
   on every rank (only the main process coarsens; the others receive the state by broadcast), so the optimizer
   groups and DDP agree.
   Featureless training nodes are recorded as excluded from coarsening; coordinate targets
   retain the full legal training graph.
4. `phi` at default initialisation except the gate bias; `b_j = logit(mean_u n*_uj / m_j)`, the mean training
   attachment of block `j`, where `n*_uj = |N(u) ^ C_j|`. This reproduces the training mean degree per block
   exactly in expectation at step zero, replacing v0.5's scalar bisection over a second encoder pass.
5. The k-means assignment vector is recorded in `run_metadata["virtual_graph"]["assignments"]` (one label per
   clustered node, in the training table's node order with the featureless nodes removed), so the clustering and
   the attachment targets are reproducible from the run's own metadata.
6. Every rank then measures the attachment target table of §5.1 from those assignments; no further collective.

The coarse graph is therefore a real coarsening of the observed training structure at step zero, and remains a
parameter set at inference: no retrieval, no node identities, no training-graph access when scoring.

### 3.4 Telemetry, never gates

Per epoch, on V_val rows: mean attachment, attachment entropy per node, coarse-node usage `sum_u a_uj`, mean gate
and gate entropy, entropy of `B`, and the per-field coordinate R² already logged. Collapse (all `a_uj` near a
constant, or one coarse node absorbing every attachment) is read from these, not blocked.

**v0.6** adds the three numbers the diagnosis had to reconstruct from checkpoints:

- `val_virtual_attachment_selectivity`: the share of attachment-logit variance that is within a protein, across
  coarse nodes (mean within-protein variance / total variance). A per-protein scalar scores 0; v0.5 measured
  0.0002 at epoch 1 and 0.037 at epoch 8, and the pass mark is > 0.2.
- `train_attach_loss` and `train_attach_r2`: the attachment term and the R² of `log1p n_hat` against
  `log1p n*` over every training endpoint the epoch presented (the linear pilot's in-sample floor is 0.40).
- `grad_norm_{term}_{group}`: one `torch.autograd.grad` per weighted loss term (`task`, `kd`, `coord`, `attach`,
  `rep`) over five parameter groups (`attention_path` = residue projection + attention + readout bias + `P`,
  `gate` = `W` + `phi`, `cal`, `distance_head`, `interface_head`), on one step per epoch, on every rank. This is
  the per-term per-group table §3.1 of the diagnosis needed; it costs five extra backwards per epoch.

All of it is telemetry. Nothing blocks a run.

## 4. Where it plugs in

| Location | Change |
|---|---|
| `src/data/struct_coords.py` | Spec `v2` (§3.0) beside `v1`: names, field slices, dims and the table's per-spec measurement; `StructCoordinateTable(graph, spec=...)`. `v1` output bit-identical. |
| `topo_prompt.py` | Field order, slices and token count follow the checkpoint's `coord_spec`; `mean_context` is refused on spec `v2` (no field). Everything else unchanged. |
| `src/model/egostitch/classifier/virtual_graph.py` (new) | `VirtualGraphGenerator`: `attach(encoded, lengths) -> a` (per protein, attention over residues), `forward(encoded_a, encoded_b, lengths_a, lengths_b) -> parts` (`endpoint_u`, `endpoint_v`, `relation`, `distance_logits`, plus `attach_a` / `attach_b`, all standardised float32) plus `telemetry()`; the closed-form counting; `attachment_loss_rows(parts, targets)` (§5.1); `initialise(assignments, adjacency, cluster_mean_states)`. |
| `coord_gen.py` | `CoordGenConfig.generator: "mlp" \| "virtual_graph"`, `virtual_graph: {k: 64, d_z: 128, heads: 4, gate_bias: 3.0, w_attach: 1.0}`; the attachment term when `attach_targets` rides along, and `loss_term_*` for the gradient probe; head widths and distance classes from the spec; `V3_1CoordGen.predict` passes residue states and lengths to the virtual generator (the MLP generator keeps its pooled input); parameter groups unchanged in shape (generator, interface). The MLP path on spec `v1` stays byte-identical for the v2 factorial. |
| `train_b0.py` | `TopoPromptRows` built on the checkpoint's spec. After it is built for a `V3_1CoordGen` with the virtual generator: encode training nodes once, k-means, per-cluster mean residue state, block adjacency, call `initialise`, broadcast the state to all ranks before DDP wrap, then build the attachment target table (§5.1) on every rank. Coordinate-loss nonself scale (§6) installed from the training targets next to it. Telemetry into `metrics.jsonl` (§3.4). |
| `score_universe.py` | Nothing structural: the generator and spec are inside the checkpoint. One new `--prefix-intervention slot_gates_open` (`g := 1`), applied inside the generator, fails closed on an MLP checkpoint. |
| Configs | `configs/split_seed42/topo_prompt_full_v3.yaml`: `topo_prompt_full_v2.yaml` with `coord_spec: v2`, `output_dir: outputs/split_seed42/topo_prompt_full_v3`. `configs/split_seed42/virtual_prompt_d.yaml`: `coord_gen_v2_d.yaml` with `generator: virtual_graph`, `coord_spec: v2`, `reader_checkpoint: outputs/split_seed42/topo_prompt_full_v3/best.pt`, `output_dir: outputs/split_seed42/virtual_prompt_d`. `configs/split_seed42/virtual_prompt_d_rev1.yaml` (**v0.6**): the same file with `virtual_graph.w_attach: 1.0` and `output_dir: outputs/split_seed42/virtual_prompt_d_rev1`; v0.5 checkpoints do not load into the revised generator and are not meant to. |
| Docs | `CLAUDE.md`/`AGENTS.md` active set, `docs/03-experiments.md` §6.1 (coordinate spec and generator are fixed choices per arm, not searched), this spec. |

## 5. Training: the v2 Stage II recipe, unchanged

Everything below is v2 §2–3, `topo_prompt_full_v2.yaml` and `coord_gen_v2_d.yaml`; it is restated only so this
file is self-contained.

- Stage I on spec `v2`: `topo_prompt_full_v3`, the v2 recipe unchanged (stationary corruption `prob 0.5, shrink_min
  0.3, sigma_max 0.5`, field mask 0.1, lr 1e-4, weight decay 0.05, one-cycle 25 epochs, patience 10 on task BCE,
  five-rank selection, `--run-kind diagnostic`). The distance prior mixture and the per-field sensitivity rows apply
  to the three fields that exist. Corruption is seeded on the canonical coordinate order of spec `v2`.
- Teacher: immutable eval-mode copy of that published reader, fed true spec-`v2` coordinates. No structural-stream
  teacher variant is launched for v3.
- Objective: `L = (1 - alpha) BCE(l_S, y) + alpha BCE(l_S, sigmoid(l_T))` with `alpha = 0.5` and positive weight 5;
  `+ 1.0 L_coord` (§6) `+ w_attach L_attach` (§5.1, `w_attach = 1`); structural stream on one sampled 40-node
  subgraph per step with `{bce 1.0, rank 1.0, degree 0.1, motif 0.1}`; anchor KL weight 1, temperature 1.
  Representation KD 0.
- Schedule: 15-epoch one-cycle, `pct_start 0.1`, `div_factor 25`, `final_div_factor 100`, grad clip 1,
  `eval.patience: null`. Groups: generator peak `3e-4` (all of §3.1 plus `Cal`), interface and head `1e-4`,
  weight decay 0.05.
- Runtime: `token_budget 196608`, `max_pairs_per_rank 1536`, `memory_limit_gib 85`, bf16, world size auto.
- Selection: `geometric_rd_five_rank_v1` on V_val after the full schedule.

### 5.1 Attachment supervision (v0.6)

Eight aggregate counts spread over `K x 2` attachments cannot deliver a per-node signal: the measured gradient
reaching an attachment logit under the v0.5 objective was 1e-5 per element, and predicted counts far below one
flattened it by a further 0.01-0.06 through `log1p`. v0.6 supervises the attachments directly, in count space:

```
n_hat_xj = m_j a_xj                              predicted neighbours of x in block j
n*_xj    = |N(x) \ {partner} ^ C_j|              measured on the coarsened legal training graph
L_attach = mean over the row's 2K entries of smooth_l1(log1p n_hat, log1p n*)
```

`a_xj = sigmoid(s_xj)` is unchanged, so the counting layer keeps its bounds and its tests
(`a <= 1` is what makes clustering `<= 1`). In count space the sigmoid is not the bottleneck:
`d log1p(m sigma(s)) / ds = n (1 - a) / (1 + n)`, which is 0.5 at `n = 1`. `n = exp(s)` was considered and
rejected: an unbounded `a` breaks the clustering bound.

The target table is measured once, on every rank, from the broadcast assignments: `(R, 2, K)` `int16` on the CPU
(about 57 MB at `R` = 221k, `K` = 64), sliced per batch by `_row_id` exactly as the coordinate targets are, and
attached only to training batches. The queried partner is removed from each endpoint's counts, as it is for the
coordinate targets. Self rows are masked with the same nonself rule as the coordinate loss; rows touching a node
the pack has no features for -- which therefore has no cluster -- carry a zero mask. V_val attachment targets are
not built: the term is training-only supervision, and V_val truth stays out of it.

The struct stream's inner forward and the throughput probe call the model without `attach_targets`, and the term
simply does not fire there.

## 6. Coordinate loss correction

Two changes to `coordinate_loss_rows`, applied to both generators:

1. **Self-pairs are masked out** of the coordinate term (`distance_class_targets == SELF_DISTANCE_CLASS`). They keep
   task and KD supervision. The distance head keeps its self class so self-pairs still assemble correctly.
2. **Nonself scale.** The Huber residual of continuous coordinate `i` is divided by
   `sigma_i^nonself / sigma_i^reader`, where `sigma^nonself` is the standard deviation of the true standardised
   coordinate over nonself training rows, computed once from `TopoPromptRows` at startup and stored as a buffer.
   The reader's input units are untouched; only the loss is measured in units where nonself variation is order one.
   On spec `v2` this matters for Jaccard (self-pairs at one against nonself positives near 0.3); on spec `v1` it is
   what restores supervision of `walk_2`/`walk_3` on distinct pairs (§2).

The distance cross-entropy is unchanged.

## 7. What is compared, and how the result is read

**Topology first.** The launch answers one question: does the virtual generator's student improve the assembled
graph over the deployable rows, at the protocol's V_val-selected threshold? On V_val, then replayed on test:

| Row | Role |
|---|---|
| `virtual_prompt_d` | the launch |
| `prefix_base` | the trunk without any prompt (V_val GS 0.401, RD ≈ 1.0, MMD 10.6 / 4.6 / 9.0) |
| `coord_gen_full` (v1) and `coord_gen_v2_d` when it exists | the MLP generator behind the same reader; v2 D is the matched control |
| `b0_v31`, `struct_grand` | published deployable rows in `docs/03-experiments.md` §3.4 |
| `topo_prompt_full_v3` on true spec-`v2` coordinates | the row's own ceiling, never a comparator arm; read against `topo_prompt_full_v2` on V_val to see what the compact spec costs the reader on truth |

Five numbers per row together: BFS-macro GS, RD, degree / clustering / spectral MMD ratios. Edge metrics
(AUROC, AUPRC, ECE, Brier) are reported next to them and are not the criterion. Single-seed differences inside
±0.01 GS / ±0.5 MMD ratio are not read. Per-epoch MMD spread is shown beside the selected value, as v2 requires.

Scoring-time rows on the selected checkpoint, same chain as `coord_gen_full`: `gates_off`, `mean`, `mean_relation`,
`shuffle` (coordinate transplant across the universe), and the new `slot_gates_open`. `mean`/`gates_off` measure
dependence on the prompt; `slot_gates_open` measures what the query gates add over the static coarse graph.

Outcomes:

- GS or the shape ratios improve on V_val over `prefix_base` and the MLP rows, and `slot_gates_open` or `shuffle`
  removes part of it: the coarse graph carries pair-relevant structure. Replicate seeds 1–2, then test.
- Topology improves but every intervention row is within noise of the main row: the gain is density calibration
  from per-node attachment; write it as that.
- No V_val topology change: the training-graph coarsening does not make pair structure more predictable from
  attributes. The generator route is closed on this evidence and the reader side becomes the next object.

## 8. Implementation checks

Unit: spec `v2` surrogates equal the corresponding spec `v1` columns on the same graph and pairs; spec `v1` output
bit-identical to HEAD; attachment ignores padded residues (same `a_u` for a sequence with and without padding) and
is identical whether the protein is encoded alone or inside a pair batch; swap symmetry of the assembled surrogates;
every surrogate finite and bounded for random attachments and for `A = 0`; `deg`, `tri`, `clustering`, `common`, `jaccard`, `L3`, `l3_density` equal exact counts on
a lifted binary graph built from an integer `m` and `B in {0, 1}`; gates at 1 reproduce the static graph;
`slot_gates_open` fails closed on an MLP checkpoint; the self-pair mask and nonself scale change the loss only where
intended; three-token prefixes have six rows per layer and the `v1` four-token path is unchanged. **v0.6**: the
attachment logits carry one bias and one query direction per coarse node (zeroing `P` leaves exactly the biases);
seeding plus tied q/k makes each coarse node's logit respond to its own residue community and not to the others;
`initialise` reproduces each block's mean training attachment and freezes `B` and `m`; `attachment_loss_rows` is
zero at the targets and puts a nonzero gradient on every coarse node's readout; the target table removes the
queried partner from exactly one block, swaps with the endpoints and masks featureless rows; the attachment term
fires only when `attach_targets` rides along; the weighted `loss_term_*` scalars sum to `loss`; the optimizer
groups and `trainable_parameters` agree once `B` and `m` are frozen.

Integration: initialisation is identical on every rank after broadcast; checkpoint round-trip restores `P, B, m` and
the scale buffer; two-rank DDP agreement of the structural-stream terms (follow `tests/test_struct_stream.py`);
packed scoring without coordinates, labels or the training graph reproduces the training forward at a fixed
checkpoint.

## 9. Launch

Local first: implementation, tests, ruff, mypy, one Codex review, commit, push. Then on an idle H20 container, as
one chain (`bash src/experiments/virtual_prompt_chain.sh [student config]`, defaulting to the v0.6 config and
skipping the reader when `topo_prompt_full_v3/diagnostic_complete.json` already exists):

```bash
hpc/run.sh train configs/split_seed42/topo_prompt_full_v3.yaml --worker-module src.train_b0 --run-kind diagnostic
hpc/run.sh train configs/split_seed42/virtual_prompt_d_rev1.yaml
```

**v0.6 epoch-1 pass marks** (`metrics.jsonl`, all telemetry, none blocking): `val_virtual_attachment_selectivity`
> 0.2; `train_attach_r2` > 0.3; implied mean degree (`val_virtual_attachment_mean x sum(m)`) within 20 % of 8.8;
`grad_norm_attach_attention_path` within 10x of `grad_norm_coord_cal`; `val_coord_r2_relation` no longer negative
on training-like rows. Selectivity still below 0.1 with the supervision in place would say the residue attention
is the wrong reader for community membership, and the next revision targets the head, not the route.

followed by the intervention chain used for `coord_gen_full` with `slot_gates_open` added. Expected cost: Stage I
8–13 h with the v2 sensitivity rows (3.5 h without them), then the profiled v2 D epoch (about 50 min) times 15 with
negligible counting overhead, about 13 h. The Stage II config names the Stage I output path; the chain starts Stage
II only after `diagnostic_complete.json` exists.

## 10. Related work carried into the design

| Reference | What is taken |
|---|---|
| [L3-PPI](https://arxiv.org/abs/2605.09964) | Globally shared virtual nodes and per-path query gates → coarse nodes `P` and per-node gates `g`. Its label-driven path-count hinge is not adopted; supervision is the fixed-semantics coordinates. Its surrogate GIN readout is replaced by closed-form counting so every prompt number keeps a graph meaning. |
| [All in One](https://arxiv.org/abs/2307.01504) | Prompt tokens, token structure, insertion pattern → `P`, `B`, `a_u`. Our input has no observed graph, so the insertion pattern is generated from the endpoints. |
| [GPF / GPF-plus](https://arxiv.org/abs/2209.15240) | Input-dependent combination of shared prompt bases → attachment `a_u` over `P`. |
| [G-Prompt](https://doi.org/10.1016/j.ipm.2023.103639) | Structural prior in the prompt → `B` initialised as the K-block density of the training graph. |
| [ProNoG](https://arxiv.org/abs/2408.12594), [GCoT](https://arxiv.org/abs/2502.08092), [TIGPrompt](https://arxiv.org/abs/2402.06326) | Conditional prompt generation → `phi`. No observed local graph or temporal history is available here. |
| [GraphPrompt](https://arxiv.org/abs/2302.08043), [GPPT](https://doi.org/10.1145/3534678.3539249) | Link prediction as the pretraining/prompting interface → the Stage I reader is exactly that. |
| [UniPrompt](https://papers.neurips.cc/paper_files/paper/2025/file/7b7d7985f62284060d65f532ed2ea5fa-Paper-Conference.pdf) | Does the prompt change computation beyond classifier adaptation → `gates_off`, `mean`, `slot_gates_open`. |
| [ProG](https://proceedings.neurips.cc/paper_files/paper/2024/file/ad3e803a977f4279330c6ab7245937c6-Paper-Datasets_and_Benchmarks_Track.pdf), [MultiGPrompt](https://arxiv.org/abs/2312.03731), [PRODIGY](https://papers.neurips.cc/paper_files/paper/2023/file/34dce0dc3121951dd0399ba02c0f0d06-Paper-Conference.pdf), [SUPT](https://arxiv.org/abs/2402.10380), [GGPL](https://doi.org/10.1145/3711896.3736976), [graph/text prompting](https://aclanthology.org/2025.acl-long.545/) | Framing only; no mechanism adopted. |

## 11. What this design does not claim

The virtual graph is a deterministic function of the two endpoints and learned parameters; it adds no inference
information. Its claim is inductive bias: pair structure composed from per-node attachments to a coarsening of the
observed graph, supervised by fixed-semantics counts. Coarse-graph surrogates are expected counts under a K-block
model, not full-graph counts; `Cal` maps their scale, the reader's standardisation their units. The compact spec is
chosen on label information and predictability measured on one checkpoint's probe dump; the Stage I ceiling on spec
`v2` against spec `v1` (§7) is what says whether the reader lost anything that matters for topology. The universe
density shift (§2) is unaddressed. Single-seed results are observations.

## 12. Change log

| Version | Date | Changes |
|---|---|---|
| **v0.6** | **2026-09-16** | Point-1 fix from the epoch-8 diagnosis, five changes and nothing else (gate, `K = 64`, k-means, reader, objective weights and schedule all stand). **A** attachment supervision in count space; **B** per-coarse-node readout `s_uj = b_j + <P_j, o_uj> / sqrt(d_z)`, the shared `attachment` linear deleted; **C** `P` seeded on each cluster's mean residue state through the key map, attention queries tied to keys at init; **D** a direct Huber attachment loss on `log1p n_hat` against `log1p n*` with `w_attach = 1`, its target table measured once per rank and sliced by `_row_id`; **E** `B` and `m` frozen at their training-graph values. Telemetry adds selectivity, attachment R² and the per-term per-group gradient probe. New config `virtual_prompt_d_rev1.yaml`; no compatibility with v0.5 checkpoints. |
| v0.1 | 2026-09-15 | Shared virtual nodes, fixed four-node blocks, endpoint attachments, query gates, standalone graph-response branch, frozen-baseline residual, separate prompt warmup. |
| v0.2 | 2026-09-15 | Graph tokenizer feeding topology tokens into the reader; new Stage I on observed templates; trainable student encoder; nine-value template signature replacing the coordinate loss. |
| v0.3 | 2026-09-15 | One design. Stage I and the coordinate interface kept; the generator alone replaced by a coarsened-graph prompt (`P, m, B` initialised from k-means over frozen pooled states and the training graph's block densities, per-node attachment, per-node query gates) whose 34 coordinates are closed-form expected counts. Template signature, graph tokenizer, observed-template Stage I, and encoder training dropped. Coordinate loss corrected (self-pairs masked, nonself scale) after the walk-kernel inversion was traced to the self-pair stratum. One launch (`virtual_prompt_d`) on the v2 D recipe; topology read first. |
| **v0.5** | **2026-09-15** | Attachment reads residue states: each coarse node attends over the protein's residues (PMA with `P` as seeds, four heads, one layer) instead of a shared MLP on the pooled summary; pooling survives only as the gate input. Attachment bias initialised to the training mean degree. Generator contract takes residue states and lengths. Owner decisions recorded: attention form, ten-surrogate spec, Stage I retrain accepted. File renamed to the dated convention. |
| v0.4 | 2026-09-15 | Compact surrogate spec `v2`: ten surrogates and a three-way distance class (degree and clustering per endpoint; common, Jaccard, L3, L3 density; distance), no context field, three prompt tokens. Justified by the truth-trained-reader transfer test (§2): the ten keep V_val 0.938 / test 0.922 of the 34's 0.953 / 0.940 on truth and read 0.761 / 0.687 against 0.676 / 0.671 on predictions. Closed form reduces to expected counts and ratios; walk kernels, two-hop, shell and returns dropped. Stage I retrained on spec `v2` as `topo_prompt_full_v3`, v2 recipe unchanged; the launch becomes a two-run chain. |

## 13. Implementation verification (2026-09-15)

- 86 focused tests pass, including compact-v1 equivalence, nonidentity calibration swap
  symmetry, exact initial attachment density, featureless-node handling, checkpoint
  round-trip, packed truth-free scoring and two-rank initialization/gradient agreement.
- Changed-file Ruff and mypy pass. The broad local suite passed 2,317 tests; its six
  HPC-script failures reproduce at pre-change HEAD. Seven distributed tests initially
  hit sandbox loopback restrictions and passed with local sockets enabled. Repository-wide
  mypy has the identical 102 pre-existing errors in 16 unchanged test files.
- Independent review's two blockers (endpoint calibration symmetry and featureless
  initialization) were fixed and rechecked; no remaining actionable finding.
- Attachment is recomputed per pair occurrence; V_val usage telemetry therefore counts
  endpoint occurrences, not deduplicated proteins. This affects runtime/telemetry, not
  the endpoint-only inference contract.

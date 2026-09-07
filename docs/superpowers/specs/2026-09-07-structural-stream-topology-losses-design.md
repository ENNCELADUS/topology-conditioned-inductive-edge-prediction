# Structural Stream and Topology Losses for the V3.1 Student

**Status (2026-09-07):** approved design, not yet implemented.
**Scope:** one implementation wave on `src/` (new sampler, new loss module, one new stream in
`train_b0`), three configs, tests, and the matching doc updates.

## 1. Goal and claim

Improve the assembled-graph topology of the endpoint-only V3.1 student by supervising the
*output adjacency* of small sampled training subgraphs directly, instead of only pointwise BCE.
The inference contract is unchanged: the model receives exactly $(x_u, x_v)$ and returns one
symmetric edge logit. Cross-pair coupling exists only in the training objective, on training
structure, and is fixed without any test topology (methodology constraint 5). Nothing here is a
KD arm and nothing here reads a teacher.

Two term families ride on the same machinery so they can be compared under one sampler:

- the GRAND terms (soft graph similarity, relative density, degree-distribution MMD), ported
  from `/Users/richardwang/Documents/grand/src/topology/finetune_losses.py`;
- the new terms: neighbour ranking, node-wise degree matching, and open/closed two-hop motif
  counting.

GRAND's evidence was obtained by warm-start fine-tuning at a fixed 0.5 threshold; this repo
selects the topology threshold on V_val, so density gains at 0.5 do not transfer. The claim under
test is therefore whether structural supervision improves *ranking and relative structure*
(GS and the three MMD ratios at the V_val-selected threshold, with AUPRC held), not density.

## 2. Training regime (decided)

- Joint training from scratch: the standard 25-epoch V3.1 schedule, optimizer, seed and
  1:5 dynamic task stream of `configs/b1_kd_control_breadth_first.yaml`, unchanged.
- A second, independent **structural stream** adds one sampled subgraph per optimizer step.
- The structural stream keeps a BCE term on its own legal pairs; the new terms are *added* to it,
  so the sampler-matched baseline (`struct_bce`) differs from each arm only by the added terms.
- The GRAND re-run in the old repository is out of scope.
- Wave 1 launches three runs: `struct_bce`, `struct_grand`, `struct_new` (§8). No control retrain.

## 3. Sampler — `src/data/struct_sampler.py` (new)

### 3.1 Substrate and legality

- Structural graph: `ValRegionSplit.build_training_graph()` (loopless, over train nodes, with
  V_val-internal edges already removed). It is built once in the DDP worker and kept.
- A subgraph is an ordered tuple of $n$ distinct node ids. Its **legal-pair mask**
  $M \in \{0,1\}^{n\times n}$ is 1 for $(i,j)$ with $i \ne j$, not both endpoints in
  `split.v_val`, and neither endpoint in `assembled.exclude_nodes`. $M$ is symmetric with a zero
  diagonal. Self-pairs never enter the structural stream (they stay on the task stream).
- Target adjacency $A \in \{0,1\}^{n\times n}$ is the induced adjacency of the structural graph,
  multiplied by $M$. Every loss reduces only over entries with $M=1$.

### 3.2 Subgraph kinds

Fixed `n = 40` nodes, `b = 8` background nodes, local budget `n - b = 32`.

| Kind | Share | Construction |
|---|---:|---|
| `bfs` | 0.50 | Uniform root among nodes with degree ≥ 1. BFS with randomised neighbour order until 32 nodes; if the component is smaller, fill from a second uniform root. Then 8 background nodes uniform over all training-graph nodes not yet present. |
| `motif` | 0.25 | With probability 0.5 an open wedge (random node with degree ≥ 2, two distinct neighbours that are not adjacent), else a closed triangle (random edge, then a common neighbour; resample the edge up to 32 times, falling back to a wedge). The three seed nodes head the BFS queue; expand to 32, then 8 background. |
| `bridge` | 0.25 | Two uniform roots at graph distance ≥ 3 (resample up to 32 times, then accept any distinct pair). BFS 16 nodes from each; union; if the union is smaller than 32 because of overlap, continue the first ball; then 8 background. All pairs across the two balls are scored like any other. |

Background nodes are uniform draws, not neighbourhood expansions, so every subgraph contains
cross-region non-edges. Nodes in V_val may appear anywhere; $M$ removes their internal pairs.

### 3.3 Epoch plan and rank striping

- `plan_struct_epoch(graph, *, seed, epoch, count, kinds, mix, rng_fn) -> StructEpochPlan`
  produces `count` subgraphs. Subgraph $t$ draws its kind and all its randomness from
  `_anchor_rng(f"struct:{t}", seed=seed, epoch=epoch)` (the blake2b node-keyed RNG in
  `src/distill/context_sampler.py`), so the plan is identical on every rank and independent of
  world size. Sorted-neighbour lists are cached once per graph, as the context sampler does.
- `count` defaults to the epoch's global optimizer step count; rank $r$ of $W$ processes
  subgraphs $r, r+W, r+2W, \dots$ and spreads them across its steps with the same
  `_step_slice` arithmetic as the context stream, so one epoch is one full pass over the plan.
- Plans are built in memory at epoch start on every rank (identical by construction); no artifact.

### 3.4 Per-subgraph statistics (telemetry)

`kind`, node count, connected components of the legal induced positive graph, legal-pair
fraction $\sum M / (n(n-1))$, positive fraction $\sum A / \sum M$, open-wedge count and
triangle count of $A$, background count, and per-epoch positive-edge coverage (fraction of
training positives that appeared in at least one subgraph) and mean reuse. These are averaged
across the epoch's subgraphs and reduced across ranks into `metrics.jsonl` under `struct_*`.

## 4. Stream — `StructStream` in `src/train_b0.py`

Sibling of `KDContextStream`, same contract: `loss(model, *, epoch, step, steps) ->
(tensor, dict[str, float])` and `epoch_telemetry(accelerator, sums) -> dict[str, float]`.

- For the step's subgraph, enumerate legal pairs $(i<j)$, bucket them by `BUCKET_BOUNDARIES`,
  chunk each bucket at `data.token_budget // boundary`, and forward each chunk through
  `_unwrapped_model(model)` under `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`
  with the boundary passed explicitly (the same discipline as `KDContextStream._score`).
  Checkpointing preserves RNG state, so dropout in the recompute matches the first pass; this
  replaces GRAND's manual collect-logits / replay-chunks mechanism.
- Chunk logits (float32) are scattered into a symmetric $n\times n$ logit matrix $Z$ with the
  illegal and diagonal entries set to 0 and excluded through $M$. The loss is computed on
  $Z$, $A$, $M$ (§5); backward recomputes one chunk's activations at a time.
- When a step has no subgraph (plan shorter than steps) or the subgraph has no legal positive
  and negative pair, the stream returns `next(model.parameters()).sum() * 0.0` so DDP sees every
  parameter every step.
- The stream never touches the task batch, the KD banks, or the negative sampler.

Cost: 780 pairs per step next to the 1,024-pair task batch, so about 1.75× a control run in
pair forwards, at the context stream's peak memory. Encoding each node once and reusing it across
its 39 pairs would need a model API change and is explicitly deferred.

## 5. Terms — `src/distill/struct_losses.py` (new)

Pure tensor functions. Inputs: logits $Z$, target $A$, mask $M$ (all $n\times n$, symmetric,
zero diagonal), $P=\sigma(Z)$, $\tilde P = P\odot M$, $\tilde A = A\odot M$, $\epsilon=10^{-8}$.
Unless stated, sums run over the upper triangle where $M=1$. Every function returns a
differentiable zero when its reduction set is empty.

| Key | Definition |
|---|---|
| `bce` | Weighted BCE over legal upper-triangle pairs: weight `positive_weight` (5.0, from the model config) on positives, 1 on negatives, normalised by the weight sum. Label smoothing follows the model config (0 here). |
| `gs` | GRAND code form: $\sum\lvert P_{ij}-A_{ij}\rvert \,/\, (\sum P_{ij} + \sum A_{ij} + \epsilon)$. This is the implemented L1 relaxation of $1-\mathrm{GS}$, not the Dice form; recorded as such. |
| `rd` | $\rho = \log\big((\sum P_{ij}+\epsilon)/(\sum A_{ij}+\epsilon)\big)$; loss $=\mathrm{SmoothL1}(\rho, 0)$ (`log_ratio_huber`). Two-sided by construction, which is what the historical sub-1 RD values require. |
| `deg_mmd` | Soft degrees $\hat d_i=\sum_j \tilde P_{ij}$, true $d_i=\sum_j \tilde A_{ij}$; Gaussian soft histograms with `bins=48` centres on $[0, n-1]$ and `sigma=1.25`; $\mathrm{TV}=\tfrac12\lVert h_{\hat d}-h_d\rVert_1$; loss $=2-2\exp(-\mathrm{TV}^2/2\sigma^2)$. Defaults are the GRAND ablation YAML values. |
| `rank` | For anchor $i$: $P_i=\{j: \tilde A_{ij}=1\}$, $N_i=\{k: M_{ik}=1, A_{ik}=0\}$. Loss $=\operatorname{mean}_{i: P_i,N_i\neq\emptyset}\ \operatorname{mean}_{j\in P_i,k\in N_i} \log\!\big(1+\exp((Z_{ik}-Z_{ij}+m)/T)\big)$, computed with `softplus`; $m=0.1$, $T=1$. Anchors weigh equally. |
| `degree` | $\operatorname{mean}_{i:\sum_j M_{ij}>0}\ \mathrm{SmoothL1}\big(\log(1+\hat d_i)-\log(1+d_i)\big)$, Huber delta 1. Targets are in-subgraph legal degrees, never full-graph degrees. |
| `motif` | $C(X)=X\odot X^2$, $O(X)=(1-X)\odot X^2$ on $\tilde P$ and $\tilde A$ (row/col sums over $n$; the masked matrices guarantee a two-hop path contributes only when both its pairs are legal, and the outer $M$ restricts to legal endpoints). Per legal $u<v$: $e_{uv}=\mathrm{SmoothL1}(\log(1+C_{uv}(\tilde P))-\log(1+C_{uv}(\tilde A))) + \mathrm{SmoothL1}(\log(1+O_{uv}(\tilde P))-\log(1+O_{uv}(\tilde A)))$. Balanced mean: split legal pairs into $\{C(\tilde A)>0\}$, $\{C(\tilde A)=0, O(\tilde A)>0\}$, $\{C(\tilde A)=O(\tilde A)=0\}$, average $e$ within each non-empty class, then average the class means. |

Clustering MMD is not ported (never trained in the GRAND ablations; $O(n^3)$ with a lossy
$d_i(d_i-1)$ relaxation). The three-node structure that clustering would supervise is carried by
`motif`.

Total structural loss: $\mathcal L_{\rm struct} = \sum_k w_k\,\mathcal L_k$ over the seven keys.
All weights are non-negative floats; any combination is legal; a term with weight 0 is not
computed. The arm name is the sorted list of nonzero keys (e.g. `bce+degree+motif+rank`) and is
logged with the effective weights at startup and stored in `run_metadata.json`.

No EMA normalisation and no GradNorm. Balance is observed, not enforced: per-term raw losses
every epoch, and per-term parameter-gradient norms at each epoch's first step by extending the
existing `_term_grad_norms` probe to a list of terms.

## 6. Config

New top-level block, added to the allow-list in `train_b0` and to `Config` as
`struct: StructConfig | None`, parsed with the same `from_mapping` / unknown-key discipline as
`DistillConfig`. Absent block ⇒ no stream, bit-identical to today's training.

```yaml
struct:
  nodes: 40
  background_nodes: 8
  mix: {bfs: 0.5, motif: 0.25, bridge: 0.25}   # must sum to 1
  subgraphs_per_epoch: null                     # null = global optimizer steps per epoch
  weights: {bce: 1.0, gs: 0.0, rd: 0.0, deg_mmd: 0.0, rank: 0.0, degree: 0.0, motif: 0.0}
  rank_margin: 0.1
  rank_temperature: 1.0
  huber_delta: 1.0
  mmd_sigma: 1.25
  mmd_bins: 48
  val_subgraphs: 32
```

`struct` is independent of `distill`; both may be present. `subgraphs_per_epoch` above the step
count is rejected. Weights all zero is rejected (use no block instead).

## 7. Telemetry, validation, stopping, selection

- Training telemetry per epoch in `metrics.jsonl`: `struct_loss` (weighted total),
  `struct_<key>_loss` (raw, per active term), `grad_norm_struct_<key>` (first step),
  the §3.4 sampler statistics, `struct_pairs` (legal pairs forwarded), and wall-clock share
  of the stream.
- Validation diagnostics: a fixed set of `val_subgraphs` subgraphs of the same three kinds drawn
  from `split.build_g_val_simple()` with seed `(seed, epoch=0)`, where all distinct-node pairs
  are legal. On each topology-due epoch they are scored under `no_grad` and every active term
  plus the hard in-subgraph degree and motif errors (at the current V_val topology threshold) are
  logged as `val_struct_*`. They feed nothing: early stopping stays on `val_task_loss`
  (commit 5e8c34d) and checkpoint selection stays the six-criterion mean rank.

## 8. Arms and comparators

Wave 1, new V_val split, seed 0, 25 epochs, identical to the control YAML except the block below.

| Config | `struct.weights` (nonzero) | Role |
|---|---|---|
| `configs/struct_bce_breadth_first.yaml` | bce 1 | sampler-matched baseline |
| `configs/struct_grand_breadth_first.yaml` | bce 1, gs 0.70, rd 0.90 | GRAND `topo_gs_rd_bce_low` point |
| `configs/struct_new_breadth_first.yaml` | bce 1, rank 1, degree 0.1, motif 0.1 | new objective |

Output dirs `outputs/struct/<name>`. Launch: `hpc/run.sh train <config>` (pack → train →
publish → test), with `OMP_NUM_THREADS=16 MKL_NUM_THREADS=16` when the three share the box.

Comparison rules: report the five topology numbers and the pairwise numbers together; compare
each arm against `struct_bce` first and against the new-split `b1_kd_control` once it exists;
check the selected epoch before crediting a term. Follow-up ablations reuse this code with new
configs only: each new term alone, `deg_mmd 0.15` versus `degree 0.1`, `rank` with `bce 0`,
and `nodes` 20 and 60.

## 9. Tests

- `tests/test_struct_sampler.py`: no illegal pair ever has $M=1$ (V_val-internal, self, featureless);
  exact node counts and background counts; kind mix over 400 draws within ±0.05; motif seeds are
  genuine wedges/triangles; bridge roots satisfy the distance rule when possible; plans are
  identical for world sizes 1, 2 and 4 and differ across epochs; coverage statistics are computed.
- `tests/test_struct_losses.py`: on hard 0/1 matrices `motif` soft counts equal brute-force
  triangle and open-wedge counts per pair; `gs`, `rd`, `degree`, `motif`, `deg_mmd` are 0 at
  $P=A$ (up to $\epsilon$); `rank` → 0 as margins grow; masked entries receive exactly zero
  gradient for every term; `gs` and `rd` reproduce hand-computed GRAND values on a 4-node case;
  empty reduction sets return differentiable zeros; balanced mean averages classes equally.
- `tests/test_struct_stream.py`: 2-rank DDP gradient agreement with the single-process path,
  following the joint-KD DDP test (d4fd396); chunk assembly reproduces a direct full forward's
  logits; a config with the block absent leaves the task loss bit-identical.
- `pytest -m "not slow and not integration"`, `ruff`, `mypy --strict` all green before commit.

## 10. Documentation updates in the same change

- `docs/03-experiments.md`: §1.4 gains the three structural arms; a new §3 entry describes the
  structural stream, sampler, terms and the wave-1 comparison rule; the stale sentence in §1.2
  claiming early stopping uses total val loss plus KD terms is corrected to `val_task_loss` only.
- `CLAUDE.md` / `AGENTS.md`: the active method set paragraph adds the structural arms
  (`struct_bce`, `struct_grand`, `struct_new`) and the `struct:` block; the Commands section adds
  one launch line.
- `docs/results/struct/README.md` is created when the first run completes, not now.

## 11. Non-goals

No change to the task sampler, KD banks, negative ratio, model architecture, checkpoint
selection or threshold cascade. No loss-balancing machinery. No clustering-MMD port. No old-repo
re-run. No per-node encoder reuse. No gates: the stream logs and proceeds.

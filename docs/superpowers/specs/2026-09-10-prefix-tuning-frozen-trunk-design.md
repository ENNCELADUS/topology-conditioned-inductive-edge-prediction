# Topology-Supervised Prefix Tuning on a Frozen Endpoint-Only Trunk

**Design spec** (supersedes the 2026-09-10 brainstorming note of the same file lineage)
**Date:** 2026-09-10, revision 2 after owner review
**Status:** design approved in brainstorming; revision 2 addresses the two architectural blockers
and the experimental corrections of the owner's review; implementation plan pending
**Arms:** `prefix_base` (frozen base), `prefix_static`, `prefix_pair`, `prefix_pair_bce`
(model family `v3_1_prefix` for the three prefix arms)

## 1. Decision summary

Prompt tuning adapts a *frozen* model through a small set of learned continuous vectors. The arm
freezes a trained endpoint-only scorer and trains nothing but a deep key/value prefix read by that
scorer's own cross-attention layers. The prefix is supervised by the structural-stream losses of
`struct_new`, so the question the arm answers is:

> Under a fixed endpoint-only scorer, can training-graph topology supervision, flowing only through a
> small prefix module, change pairwise decisions so that the assembled predicted graph becomes more
> plausible, without exposing any test topology?

Decisions, in the order they were made:

| # | Question | Decision |
|---|---|---|
| 1 | What is frozen | A trained endpoint-only trunk; only the prefix trains |
| 2 | What supervises the prefix | Task BCE + the `struct_new` structural-stream terms; no teacher, no bank |
| 3 | How the frozen trunk reads the prefix | Zero-init gated prefix attention in all three cross-attention layers |
| 4 | Static or pair-conditioned | Both: `prefix_static` (shared-prompt control) and `prefix_pair` (primary) |
| 5 | Loss weights | One 10-trial constrained study per trained-weight arm through `struct_hpo` |
| 6 | Which frozen trunk (rev 2) | A matched `prefix_base`: the headline B0 recipe with `mixing.mode: bidirectional_cross`, because the headline B0 has no cross-attention layers (§4) |
| 7 | Pair shift (rev 2) | Slot-specific low-rank $\Delta P_{uv}\in\mathbb R^{m\times d}$; a shared shift cancels in the prefix softmax (§5.1) |
| 8 | Topology attribution (rev 2) | `prefix_pair_bce`: same prefix, same stream, structural weights zero (§7) |

**Out of scope.** Prompting inside the protein language model (the brainstorming note's Option B)
is technically possible but unsupported by this work: the data package holds only frozen 1536-dim
token features under opaque node IDs, with no sequences and no encoder identity, and the
node-disjoint PPI literature shows no PEFT-of-ESM gain over frozen features (§A.2). It is recorded
here as future work and not designed for.

## 2. Why prompt tuning here is an attribution instrument, not a capacity instrument

- The prompt introduces no new input information: it is a deterministic function of $(x_u,x_v)$,
  which the trunk already reads in full. Existing probes indicate that only part of the teacher's
  topology representation is recoverable from those features under the tested models: linearly
  0.40--0.47 of the teacher vector, and $R^2$ 0.6--0.77 per descriptor through a jointly trained
  head (`docs/03-experiments.md` §4.5). Those are bounds under the probes tried, not
  information-theoretic ceilings; a prefix could in principle reach representations those probes
  did not, but it cannot see anything they could not.
- What a frozen trunk buys is a clean causal path. Every topology gain must flow through a few
  hundred thousand parameters that the frozen model cannot compensate for; the gates report how
  much each attention uses the prompt; and shuffle / mean-replacement interventions on the prefix
  are exact tests of per-pair conditioning. The retired imagination arm (G5, `cut`) fed a
  feature-generated latent through a jointly trained trunk and could not attribute its effect;
  freezing the trunk removes that confound by construction.
- Precedent for the pattern: PromptKD (CVPR 2024) trains only prompts in a frozen student and beats
  full fine-tuning of the same student (§A.1). Precedent for the mechanism: Prefix-Tuning and
  P-Tuning v2 show per-layer key/value prefixes far outperform input-only prompts; LLaMA-Adapter's
  zero-init gated attention supplies exact null identity at initialization, which this project's
  conditioning ladder already requires of every rung.
- Novelty boundary (§A.1, L3-PPI): prompts for PPI exist, as free virtual-protein embeddings gated
  per pair and classified by a frozen GIN, evaluated by micro-F1 only. Nothing published supervises
  a prompt with losses on the assembled predicted network or evaluates assembled topology (PRING
  defines those metrics as evaluation only). The defensible claim is *topology-supervised attention
  prefixes on a frozen endpoint-only scorer, evaluated on the assembled network under a
  node-disjoint, zero-observed-edge protocol*. The prefix is not called a "topology-conditioned
  representation" unless later probes show $z_{uv}$ predicts structural descriptors and shuffling
  $z_{uv}$ specifically removes the topology gain.

## 3. Task contract (unchanged)

Inference input is exactly $(x_u,x_v)$, the two frozen token sequences; output is the symmetric
probability $\widehat A_{uv}$. The prefix generator reads only the frozen encoder's summary of those
two sequences. No observed edge, neighbour, degree, retrieval result, or test statistic enters at
training or inference. Training topology supervises the prefix through the structural stream;
graph assembly remains an evaluation operation (`docs/02-methodology.md` §1).

## 4. The frozen base: `prefix_base`

**Why not the headline B0.** The headline student (`configs/split_seed42/b0_v31.yaml`) sets
`mixing.mode: none`. In `PairCrossAttention` that mode builds an empty layer list regardless of
`cross_attn_layers`; the pair interaction happens only in `PairContextGatedReadout` after
independent Siamese encoding, with AB/BA evaluation and `abba_max`. There is no attention in which
a prefix can be read. Attaching the prefix to the Siamese encoder's self-attention or to the readout
would be a different design and could not be described as prefix tuning of the pair trunk.

**`prefix_base`.** The headline B0 recipe with one model change, `mixing.mode: bidirectional_cross`,
which instantiates three `CrossAttentionLayer`s (shared-weight bidirectional cross-attention with
FFN and a CLS attention over both streams) ahead of the same `pair_context_gated` readout. Same
split (root `node_007630`, split seed 42), seed 0, optimizer, schedule, sampler, patience, and
held-out protocol. The one further difference is the per-rank micro-batch: the three bidirectional
cross-attention layers hold two extra attention maps per layer, and B0's `runtime.token_budget:
524288` / `max_pairs_per_rank: 4096` OOMs on a 95 GiB H20 (B0 itself peaks at 59.7 GiB per rank), so
`prefix_base` and every prefix arm use `262144` / `2048`. The loss, the data, and the optimizer are
unchanged; the halved batch doubles the optimizer steps per epoch, and the `onecycle` schedule
re-resolves `total_steps` from the plan. Config `configs/split_seed42/prefix_base.yaml`, output
`outputs/split_seed42/prefix_base`, published checkpoint `best.pt`, tested once.

- **The readout stays.** `PairCrossAttention.forward` runs the layers and then hands the
  cross-attended token streams *and* the updated CLS token to `PairContextGatedReadout`, which
  pools the tokens by mean, max, and pair-conditioned attention, gates the three branches, and
  concatenates the CLS vector before its final projection. The readout is therefore the
  integration step over the cross-attention output, not a competitor to it. Under the headline B0
  the CLS input to that readout is the learned constant parameter (no layers touch it); under
  `prefix_base` it becomes pair-dependent through each layer's CLS attention, so the CLS prefix site
  in §5.2 is live. Replacing the readout with a CLS-only head would be a second change from the
  headline recipe and would discard the pooling branches with no evidence against them.
- `prefix_base` is the **exact base** for every prefix arm and the row every prefix comparison is
  made against. The headline B0 remains the paper's endpoint-only comparator and is reported next
  to it; it is not the base of anything here.
- The bidirectional path is exercised by `tests/test_b0_attention.py` at the layer level but has
  not run end-to-end since configs moved to `none` (2026-07-12). The plan includes a local CPU
  smoke through `train_b0` with the structural stream before the H20 launch.
- Cost is not a concern: at $d=512$ with token budget 131,072 per batch, the three layers add
  attention maps of at most $1024^2\times 8$ heads per pair-direction, well inside the H20
  memory limit, and roughly double the per-step FLOPs of the headline B0.

**Loading and freezing.** The `v3_1_prefix` family loads the base `V3_1` from the checkpoint's
embedded config and `model_state`, sets `requires_grad=False` on every base parameter, and keeps
the base in eval mode permanently: the wrapper overrides `train()` so that switching the wrapper to
training re-applies `base.eval()`. This is required, not cosmetic: `requires_grad=False` does not
disable dropout, token dropout, or stochastic depth, and the base carries dropout in the encoder,
the cross-attention layers, the readout, and the head. The frozen function is deterministic.

**Published checkpoint** carries the full state (frozen base + prefix) so `score_universe`,
`test_protocol`, and the interventions rebuild the arm from the checkpoint alone. The base
checkpoint path and its file hash are written into `run_metadata.json` as provenance; nothing
verifies them (no digest pinning, per project rules).

**Training stream** identical to `struct_new`: official 1:5 task rows with positive weight 5, plus
one sampled 40-node training subgraph per optimizer step (32 locally expanded + 8 background nodes,
bfs/motif/bridge mix 50/25/25) scored as a logit matrix over legal pairs. The only difference from
the joint `struct_new` arm is the trainable parameter set (and the base recipe's mixing mode).

## 5. The prefix module

Let $d=512$ be the trunk width, $L=3$ the number of cross-attention layers, $m=16$ the prefix
length, $H=8$ the heads, $r=8$ the conditioning rank.

### 5.1 Parameters

- **Static prefix.** Per layer $\ell$, $P_0^{(\ell)}\in\mathbb R^{m\times d}$. Initialised by
  sampling $m$ token states from the frozen Siamese encoder's output over randomly drawn training
  nodes (the real-activation initialisation Prefix-Tuning found stabilising; PEFT-SP does the same
  with amino-acid embeddings). About 25k parameters.
- **Pair generator (`pair` arms only).** With $e_u, e_v\in\mathbb R^{d}$ the frozen encoder's
  masked-mean token vectors,
  $$c_{uv}=[\,e_u+e_v;\ |e_u-e_v|;\ e_u\odot e_v\,]\in\mathbb R^{3d},\qquad
    z_{uv}=g_\phi(c_{uv})\in\mathbb R^{128},$$
  where $g_\phi$ is LayerNorm → Linear($3d$,128) → GELU. Per layer, a **slot-specific low-rank
  shift**:
  $$A^{(\ell)}_{uv}=\operatorname{reshape}\big(W_\ell z_{uv},\,m,\,r\big),\qquad
    \Delta P^{(\ell)}_{uv}=A^{(\ell)}_{uv}U_\ell,\qquad
    P^{(\ell)}_{uv}=P^{(\ell)}_0+\Delta P^{(\ell)}_{uv},$$
  with $W_\ell\in\mathbb R^{(m r)\times 128}$ and $U_\ell\in\mathbb R^{r\times d}$: about 20k
  parameters per layer after the shared generator, roughly 0.26M in total.

  *Why slot-specific is mandatory.* A shift $s_{uv}$ shared by every slot gives keys
  $W_K P_{0,k}+W_K s_{uv}$, whose second term is the same for every $k$ and therefore cancels in
  the prefix softmax; the value side reduces to one additive vector $W_V s_{uv}$ shared by every
  query. The pair condition would then be unable to change which prompt slots a query reads, and
  `prefix_pair` would collapse to static prefix attention plus a pair-conditioned residual. With
  $\Delta P_{uv,1}\ne\Delta P_{uv,2}\ne\cdots$ the condition modifies both keys and values per slot.
- **Gates.** One vector $g^{(\ell,a)}\in\mathbb R^{H}$ per layer and per attention site
  $a\in\{\mathrm{A\leftarrow B},\ \mathrm{B\leftarrow A},\ \mathrm{CLS}\}$, initialised to zero.
  Nine gates, 72 scalars.
- **Initialisation rule.** The gate is the only zero factor. $U_\ell$ is initialised small but
  nonzero (normal, std 0.02) and $W_\ell$ with the default linear init, so the pair generator
  receives gradient as soon as a gate opens instead of waiting for a second zero factor to move.
  Exact null identity still holds through the gate alone. Consequently `prefix_pair` and
  `prefix_static` share the base function at initialisation but not the prefix content.

Symmetric $c_{uv}$ makes $P_{uv}=P_{vu}$, so the AB/BA doubled batch and the `abba_max` readout are
unaffected.

### 5.2 Reading inside a frozen `CrossAttentionLayer`

Each frozen layer (`classifier/layers.py: CrossAttentionLayer`) has two directional attentions
sharing `self.attn` and a CLS attention `self.attn_cls`. For a site with frozen projections
$W_Q,W_K,W_V,W_O$, normalised queries $\tilde Q$, and partner keys/values $K,V$:

$$
O=\underbrace{\operatorname{MHA}_{\text{base}}(\tilde Q,K,V)}_{\text{the original module, called unchanged}}
\;+\;
W_O\Big[\tanh(g)\odot_{\text{head}}
\operatorname{softmax}\!\big(\tilde Q W_Q (P_{uv} W_K)^{\top}/\sqrt{d_h}\big)\,P_{uv}W_V\Big].
$$

- **The base attention is the original code path.** The wrapper calls the frozen
  `nn.MultiheadAttention` exactly as `CrossAttentionLayer._attend` and the CLS block do; it never
  re-implements the partner attention from `in_proj_weight`. Only the prefix branch is computed by
  hand, reusing the frozen packed `in_proj_weight`/`in_proj_bias` slices for $W_Q,W_K,W_V$ and the
  frozen `out_proj` for $W_O$. Since $\tanh(0)=0$ multiplies the prefix branch to an exact zero
  tensor, `gates_off` reproduces the base output bit for bit regardless of which fused kernel the
  base call selects.
- The prefix softmax is **separate** from the partner softmax (LLaMA-Adapter zero-init attention),
  so the prefix never dilutes attention over real partner tokens. Prefix positions carry no padding
  mask.
- Sites: the A-over-B attention (partner = B tokens), the B-over-A attention (partner = updated A
  tokens), and the CLS attention (partner = concatenated A and B tokens); all three, in all three
  layers. FFN sub-layers are untouched. The prefix path has no dropout; the base runs in eval mode.

*Bit-exactness caveat (2026-09-10, verified on torch 2.10.0).* Null identity is bit-exact against the
frozen base as deployed inside the arm: parameters with `requires_grad=False`, eval mode. PyTorch's
attention kernel path depends on the parameters' `requires_grad` state, so a scoring pass of the same
weights with `requires_grad=True` (an ordinary `score_universe` run of `prefix_base`) differs from the
`gates_off` intervention by about 1e-7 in float32. The `gates_off` check in §7 therefore compares
logits to `prefix_base`'s artifact within 1e-6, and metrics at the frozen threshold can differ only at
exact ties.

### 5.3 Compute

Freezing removes parameter gradients and optimizer state, and the Siamese encoder runs under
`torch.no_grad`. It does not remove activation memory downstream of the first prefix: once a prefix
changes a hidden state in layer 1, backward must propagate $\partial\mathcal L/\partial P$ through
every later frozen attention, FFN, the readout, and the head. Expect a full forward plus an
input-gradient backward through the pair trunk and head, with the encoder's activations and all
weight gradients saved. No per-node activation cache is built.

## 6. Training and hyperparameter study

- **Loss.** $\mathcal L=\mathcal L_{\text{task}}+\mathcal L^{\text{struct}}_{\text{bce}}
  +w_r\mathcal L_{\text{rank}}+w_d\mathcal L_{\text{degree}}+w_m\mathcal L_{\text{motif}}$ on
  the sampled subgraph, exactly as `src/distill/struct_config.py` and the `train_b0` structural
  stream implement them (rank margin 0.1).
- **Optimizer.** AdamW over prefix parameters only, weight decay 0.05, onecycle schedule, 25 epochs,
  grad clip 1.0, bf16 DDP; early stopping on validation task loss with `eval.patience: 5`,
  `eval.topology_every: 2` (campaign settings). Learning rate is searched: prefix modules need a
  higher rate than full networks and are rate-sensitive (Prefix-Tuning).
- **Study.** `src/experiments/struct_hpo.py` gains arms `prefix_static` and `prefix_pair`, each a
  10-trial constrained MO-TPE study (2 enqueued priors, 3 startup trials) with search box
  `rank` [0.1, 3.0], `degree` [0.01, 1.0], `motif` [0.01, 1.0], `lr` [1e-4, 1e-2], all
  log-uniform. Objectives and winner selection are exactly the campaign's shared rule
  (`src/experiments/kd_rank_strict_hpo.py`, installed by `2c2f972`): TPE over three objectives,
  maximise AUPRC, maximise GS, minimise geometric-mean MMD, no RD constraint; final winner by the
  five-metric equal mean rank (`geometric_rd_five_rank_v1`, `docs/03-experiments.md` §1.2). The
  prefix arms and `struct_new` are therefore tuned under the same selection pressure. Priors:
  (`rank` 1.0, `degree` 0.1, `motif` 0.1, `lr` 1e-3) and, when the `struct_new` study has finished,
  its winner's weights with `lr` 3e-3 (otherwise `rank` 3.0, `degree` 0.5, `motif` 0.5, `lr` 3e-3).
- **Thinness.** Ten trials over a four-dimensional log box is a screen, not a resolved surface. A
  small static-vs-pair difference is not interpreted while the two studies' Pareto surfaces remain
  broad; the paired-bootstrap rule in §7 governs.
- **`prefix_pair_bce`.** Structural weights fixed to zero; no weight study. It runs at three
  learning rates, the `prefix_pair` winner's `lr` and that value divided and multiplied by 3, and
  the same five-metric rule picks one, so the control is not under-tuned relative to the arm it
  must be compared with.
- **Held-out.** `prefix_base` and each prefix winner run the held-out protocol once, at their own
  V_val-frozen thresholds.

## 7. Controls, interventions, and how the result is read

**Rows** (all on the headline split, seed 0):

| Row | Trainable | Structural terms | Role |
|---|---|---|---|
| Headline B0 | all, `mixing: none` | none | the paper's endpoint-only comparator; context only |
| `prefix_base` | all, `mixing: bidirectional_cross` | none | the exact frozen base; every prefix row is read against it |
| `struct_new` winner | all, `mixing: none` | rank + degree + motif | the same supervision with every parameter trainable (different base recipe; context) |
| `prefix_static` winner | $P_0$, gates | rank + degree + motif | shared-prompt control |
| `prefix_pair_bce` | $P_0$, generator, gates | subgraph BCE only | topology-supervision control |
| `prefix_pair` winner | $P_0$, generator, gates | rank + degree + motif | primary |

**Attribution chain.** Three inequalities carry the claim, each on the five topology numbers at
each row's own frozen threshold, with AUPRC alongside:

1. `prefix_pair` > `prefix_base`: the prompt moved the assembled graph.
2. `prefix_pair` > `prefix_pair_bce`: the graph-level terms, not extra supervised pair learning
   through new parameters on the extra subgraph BCE rows, caused it.
3. `prefix_pair` > `prefix_static`: explicit pair-conditioned prompt generation adds something
   beyond a globally shared learned prompt.

**Interventions** (scoring-only, on the selected `prefix_pair` checkpoint), applied at the
**unordered-pair level** so the AB and BA evaluations of one pair see the same substituted
$z_{uv}$ and `abba_max` symmetry is preserved:

1. *Gates off* — all $g\leftarrow 0$; must reproduce `prefix_base`'s logits within 1e-6 (a correctness check, reported; see §5.2).
2. *Shuffle* — $z_{uv}$ replaced by the $z$ of another pair drawn by **one seeded permutation of
   the whole universe** (`--prefix-intervention-seed`), identical under fan-out and never within a
   scoring batch or a shard: pairs arrive lexicographically sorted and the length-bucketed sampler
   cuts contiguous chunks, so a batch-local permutation would mostly substitute the condition of a
   pair sharing an endpoint; and the 1:1 `val_cls`/`test` lists are label-sorted, so a shard-local
   permutation (the implementation until 2026-09-12) left every substitute with the row's own
   label and was vacuous there. The scorer makes two passes — encode each row's source pair and
   collect its $z_{uv}$, then score the row's own endpoints under it — and the permutation is not
   a model-level mode.
3. *Mean* — $z_{uv}$ replaced by its training-set mean (a `z_mean` buffer computed at publish time).
   Reload the selected checkpoint, use eval mode, and replay that epoch's 1:5 training task
   pairs, each row once without BCE weighting or structural-stream repeats. Sum conditions and
   counts across ranks before taking the mean; never average conditions from changing training
   weights. `z_count` records the global row count. This pass reads no validation or test rows.

Each intervention run **re-selects its own thresholds on its intervened V_val scores, exactly as a
normal arm does** (the fixed topology threshold and the `val_cls` max-F1 classification threshold
alike): it is read as the deployable arm that intervention defines, not as `prefix_pair` evaluated
at a borrowed operating point. Each reports AUPRC with the five topology numbers (BFS-macro GS,
geometric RD, degree / clustering / spectral MMD ratios) at its own selected threshold, alongside
the gate telemetry $\tanh(g)$ per site and epoch.

**Reading.** Edge-level and assembled-graph families are always reported together
(`CLAUDE.md` claim rules).

- *Static ≈ pair* means explicit pair-conditioned prompt generation provides no benefit beyond a
  globally shared learned prompt. It does **not** mean recalibration: a static prefix is read
  through query-dependent attention, so it can change rankings and nonlinear boundaries. Whether
  the static effect is calibration is tested separately by fitting
  $\ell_{\text{static}}\approx a\,\ell_{\text{b0\_cross}}+b$ on the test logits and reporting AUROC/AUPRC of
  the fit, the Spearman rank correlation between the two logit vectors, and the residual variance;
  a large ranking change is representation adaptation.
- *Pair ≈ pair\_bce* means the structural terms contributed nothing beyond the extra supervised rows;
  the "topology-supervised" label is then not earned.
- *All ≈ `prefix_base`* means the frozen trunk bounds what any prompt can move under this supervision;
  the bound is the result and is reported.
- **Significance.** Differences in GS, RD, and the MMD ratios between two rows are judged by the
  test protocol's paired nonparametric bootstrap over the sampled test subgraphs within size strata
  (already computed by `test_protocol`); a difference counts when its paired 95 % interval excludes
  zero. The split's heuristic noise band (about ±0.01 GS, ±0.5 MMD ratio from the B0 seed pair)
  is a screen only.
- Selected epochs are reported; the epoch confound that erased earlier KD gains does not apply to a
  frozen trunk.

**Deferred** until `prefix_pair` clears inequalities 1--2: the same-condition pooled adapter on the
frozen trunk (the "injection mechanism" comparison), a static BCE-only prefix, prefix length
{4, 64}, rank $r$ {2, 32}, the reparameterisation MLP on $P_0$, descriptor probes of $z_{uv}$, and a
teacher-KD or descriptor-bottleneck prefix.

## 8. Implementation surfaces

- **`prefix_base`.** `configs/split_seed42/prefix_base.yaml` (the B0 config with
  `mixing.mode: bidirectional_cross`, `output_dir: outputs/split_seed42/prefix_base`). No code
  change; one local CPU smoke of the bidirectional path through `train_b0` with a `struct:` block
  before launch.
- **New module** `src/model/egostitch/classifier/prefix.py`: `PrefixGenerator` ($P_0$, $g_\phi$,
  $W_\ell$, $U_\ell$, gates, `z_mean` buffer), `GatedPrefixAttention` (the per-site wrapper that
  calls the frozen `nn.MultiheadAttention` unchanged and adds the gated prefix branch),
  `PrefixCrossAttentionLayer` (wraps one frozen `CrossAttentionLayer`), and `V3_1Prefix` (loads and
  freezes the base `V3_1`, overrides `train()` to keep the base in eval mode, wraps
  `cross_attention.layers`, exposes `prefix_parameters()`, and forwards with the same
  `PairInputs`/loss interface as `V3_1`, including the structural-stream logit-matrix path and an
  `intervention` attribute honoured at the unordered-pair level, plus `condition_from_encoded`
  and a `z` override on `logits_from_encoded` for the scorer's universe-level shuffle).
- **Family dispatch.** `train_b0.MODEL_FAMILIES` and `build_model` accept `v3_1_prefix`;
  `_build_optimizer` builds its single param group from `prefix_parameters()`;
  `score_universe.MODEL_BUILDERS` rebuilds the family from the checkpoint's embedded config (the
  frozen base state is inside the checkpoint, so no base file is needed at scoring time).
- **Interventions.** A `--prefix-intervention {none,gates_off,shuffle,mean}` scoring option.
  `gates_off` and `mean` are `V3_1Prefix.intervention` modes; `shuffle` lives in the scorer, which
  encodes every scored row's universe-level source pair (`--prefix-intervention-seed` draws the
  map; recorded in the artifact meta with `prefix_shuffle_scope: universe` and cross-checked at
  merge), collects its $z_{uv}$, and feeds it back through
  `V3_1Prefix.logits_from_encoded(..., z=...)`.
- **Configs.** `configs/split_seed42/prefix_static.yaml`, `prefix_pair.yaml`,
  `prefix_pair_bce.yaml`: the `struct_new` config with `model.family: v3_1_prefix`,
  `model.config.prefix: {base_checkpoint, tokens: 16, rank: 8, conditioning: static|pair,
  bottleneck: 128}`, `optim.lr` as the prior, and `output_dir: outputs/split_seed42/prefix_*`.
  `struct_hpo` writes trials under `outputs/struct_hpo/prefix_{static,pair}`; `prefix_pair_bce`'s
  three learning-rate runs live under `outputs/split_seed42/prefix_pair_bce/lr_*`.
- **Tests** (`tests/test_prefix_model.py`): bitwise null identity against the frozen base at zero
  gates; AB/BA symmetry of $P_{uv}$; no gradient on any base parameter after a backward; base
  stays in eval mode after `wrapper.train()`; the slot-specific shift changes prefix attention
  weights (a shared-shift control must not); the mean intervention and an explicit `z` change
  logits only through $z_{uv}$ and give identical AB/BA substitutions; config parsing and family dispatch;
  checkpoint round trip through `score_universe.build_model`.
- **Docs in the same change:** arm table in `docs/03-experiments.md` §1.4; correct §1.5 there,
  which describes the current student as having three cross-attention layers (under
  `mixing.mode: none` it has none; `prefix_base` is the arm that does); the active method set in
  `CLAUDE.md`/`AGENTS.md`; and the `struct_hpo` docstring. A result note under
  `docs/results/prefix_split_seed42/` follows the runs.
- **Execution order.** Local: `prefix_base` config + smoke, module, tests, configs, docs, Codex review,
  commit, push. H20: `prefix_base` train + test; then the two studies
  (`.venv/bin/python -m src.experiments.struct_hpo --arm prefix_static` / `--arm prefix_pair`);
  then `prefix_pair_bce` at its three learning rates; then each winner's held-out run and the three
  interventions.

## 9. Risks and the answers designed in

| Risk | Answer |
|---|---|
| `prefix_base` differs from the headline B0 in quality | Both are reported; `prefix_base` is the base for attribution, the headline B0 the paper's comparator |
| The bidirectional path has not run end-to-end since July | Local smoke with the structural stream before launch |
| The prefix cannot move topology under a frozen trunk | Reported as a bound; the `struct_new` row shows what full training moves |
| Static and pair prefixes tie | Reported as "no benefit beyond a shared prompt"; the calibration fit separates recalibration from representation change |
| Gains come from extra supervised rows, not graph terms | `prefix_pair_bce` at matched forwards and tuned learning rate |
| Shared shift collapses in the prefix softmax | Slot-specific low-rank $\Delta P_{uv}$ (mandatory) |
| Manual re-implementation of the base attention breaks bitwise identity | The base MHA is called unchanged; only the prefix branch is manual |
| `wrapper.train()` re-enables base dropout | `train()` override re-applies `base.eval()` |
| Two zero factors stall the pair generator | Gate is the only zero factor; $U_\ell$ small nonzero |
| Structural terms shift the logit scale, hurting raw ECE/Brier | Threshold selection absorbs the shift for topology; ECE/Brier reported raw beside it |
| Learning-rate sensitivity of prompt modules | `lr` in the search box; three-point `lr` sweep for the BCE control |
| Ten trials are thin for four dimensions | Treated as a screen; paired bootstrap governs claims |
| Scoring depends on an external base file | The published checkpoint embeds the frozen base state |

---

## Appendix A. Related work, verified 2026-09-10

Ranked by relevance to this design, not by general importance. Facts below were checked against
the papers' full text unless marked *(snippet)*.

### A.1 Prompt/prefix learning

- **L3-PPI** (Gao, Zi, Liu, Meng, Li, Li; ICML 2026; arXiv:2605.09964). Builds a *prompt graph* of
  $K{+}1$ free learnable prompt-node embeddings forming virtual length-3 paths $u\to v_i^P\to v_0^P\to v$
  around the queried pair; a Gumbel-sigmoid gate per path; a GIN pre-trained on native L3 paths and
  then frozen classifies the gated prompt graph. Inference uses only the two backbone embeddings.
  Model-agnostic over PIPR, GNN-PPI, MAPE-PPI, HIGH-PPI, ESM2-650M, SaProt, and others. Evaluated on
  STRING, SHS27k/148k, Yeast with Random/BFS/DFS 60/20/20 splits; reports micro-F1 only, no
  assembled-graph metric. Gains concentrate on BFS/DFS and neither-seen pairs (e.g. GNN-PPI SHS27k
  DFS NS 43.57→51.62) and are flat or negative on random splits. Ablations: removing the gate costs
  the most (83.22→76.56). *Boundary for us:* prompts for PPI and pair-gated prompts exist; a
  topology-supervised prefix evaluated on the assembled network does not.
- **PromptKD** (Li et al., CVPR 2024; arXiv:2403.02781). Student trains only deep visual prompts
  (depth 9, length 4) plus a projector; backbone frozen; loss $\tau^2\,\mathrm{KL}$ on teacher vs
  student class distributions, $\tau=1$, no labels. Prompt-only student beats full fine-tuning of
  the same student (ImageNet HM 76.22 vs 73.34). *Precedent for supervision flowing only into a
  prompt of a frozen student.*
- **Prefix-Tuning** (Li & Liang, ACL 2021; arXiv:2101.00190). Per-layer key/value prefixes at every
  layer; embedding-only prefix is far worse (E2E BLEU 62.2 vs 69.7); direct optimisation of the
  prefix is unstable, fixed by an MLP reparameterisation discarded after training; optimum prefix
  length ≈10 for table-to-text; real-activation initialisation beats random; beats fine-tuning in
  low-data regimes by 2.9 BLEU. Its BART code injects prefixes into decoder *cross*-attention as
  well as self-attention (implementation fact, confirmed from secondary sources). *Precedent for
  the deep KV prefix and for prefixes in cross-attention.*
- **P-Tuning v2** (Liu et al., ACL 2022; arXiv:2110.07602). Deep per-layer prefixes via
  `past_key_values`; 0.1--3 % parameters match fine-tuning across 300M--10B; simple classification
  wants prompts under 20 tokens; reparameterisation helps some tasks and hurts others.
- **CoCoOp** (Zhou et al., CVPR 2022; arXiv:2203.05557). Meta-Net Linear–ReLU–Linear (hidden =
  input/16) maps the instance feature to one shift added to every context token; base-to-new HM
  75.83 vs CoOp 71.66; instance conditioning costs one encoder pass per instance. *Form of our pair
  generator's trunk; note that CoCoOp's shared per-token shift is exactly the form §5.1 rejects for
  a separate-softmax prefix, which is why our shift is slot-specific.*
- **LLaMA-Adapter** (Zhang et al., 2023; arXiv:2303.16199). Zero-initialised gated attention over
  prefix tokens with a separate softmax, added to the frozen attention output. *Form of our reading
  mechanism; gives exact null identity.*
- **All in One** (Sun et al., KDD 2023; arXiv:2307.01504) and **GPF** (Fang et al., NeurIPS 2023;
  arXiv:2209.15240). Graph prompts are feature-space additive vectors (GPF: one $p$ added to every
  node; All in One: prompt tokens with learned token structure, similarity-gated), consumed by a
  frozen GNN that still message-passes over an *observed* graph at inference. Positioning only:
  our inference sees no graph.
- **Lester et al.** (EMNLP 2021) and **VPT** (Jia et al., ECCV 2022): canonical soft-prompt and
  deep-vs-shallow visual prompt references; background.

### A.2 PEFT of protein language models for PPI (why the ESM variant is out of scope)

- **Sledzieski et al.** (MLSB 2023; PNAS 2024 *(venue from snippet)*). LoRA $r{=}8$ on ESM-2 650M,
  Bernett node-disjoint split: frozen-embedding MLP AUPR 0.684 > LoRA 0.600 > full fine-tune 0.577.
  PEFT beats full fine-tuning but not frozen features.
- **Reim et al.** (bioRxiv 2025). Unbiased sequence-based PPI plateaus at accuracy 0.65 across
  frozen-ESM-2 models; the PEFT result above sits at ~0.63.
- **PEFT-SP** (Zeng, Wang, Xu et al., Genome Research 2024). Prompt tuning on ESM-2 is prepended
  soft embeddings (`prefix_prompt`, layer 0 or deep variants, optional ResMLP); on ESM2-3B mean
  MCC2 prompt 0.663 < adapter 0.762 < full FT 0.796 < LoRA 0.830; full ESM stack recomputed every
  step, no activation caching.
- **PLM-interact** and **MINT** (Nat. Commun. 2025) fully fine-tune ESM-2 650M; not PEFT.
  **xTrimoPGLM** LoRA gains are on peptide–MHC/TCR tasks, not binary PPI. No prompt- or
  prefix-tuned PLM for binary PPI was found.

### A.3 Topology-aware losses on the predicted network

- **Negative finding.** No published work trains a PPI predictor with degree/clustering/motif/
  spectral losses on its own predicted network, with or without prompt tuning.
- **PRING** (Zheng et al., NeurIPS 2025) defines GS, degree/clustering/spectral MMD, and relative
  density on reconstructed PPI subgraphs as *evaluation only*, and finds PLM-based predictors
  over-densify. This project's structural stream is the training-side counterpart.

## Appendix B. Terminology for the paper

- **Prompt tuning:** optimising a small set of continuous parameters while the backbone stays frozen.
- **Deep KV prefix:** learned key/value context inserted into the attention of several layers.
- **Zero-init gated prefix attention:** a separate softmax over the prefix whose output enters
  through a tanh gate initialised at zero.
- **Pair-conditioned prefix:** a shared prefix plus a per-pair, per-slot low-rank shift generated
  from the two endpoint summaries.
- **Topology-supervised prefix:** a prefix whose only structural signal is a loss computed on the
  predicted adjacency of sampled training subgraphs, established against a BCE-only prefix control.

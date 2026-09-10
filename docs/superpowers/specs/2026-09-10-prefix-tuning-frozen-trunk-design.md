# Topology-Supervised Prefix Tuning on a Frozen Endpoint-Only Trunk

**Design spec** (supersedes the 2026-09-10 brainstorming note of the same file lineage)
**Date:** 2026-09-10
**Status:** design approved in brainstorming; implementation plan pending
**Arms:** `prefix_static`, `prefix_pair` (model family `v3_1_prefix`)

## 1. Decision summary

Prompt tuning adapts a *frozen* model through a small set of learned continuous vectors. In this
project the only frozen, deployable model is the endpoint-only B0 student, so the arm freezes a
trained B0 checkpoint and trains nothing but a deep key/value prefix read by B0's own
cross-attention layers. The prefix is supervised by the structural-stream losses of `struct_new`,
so the question the arm answers is:

> Under a fixed endpoint-only scorer, can training-graph topology supervision, flowing only through a
> small prefix module, change pairwise decisions so that the assembled predicted graph becomes more
> plausible, without exposing any test topology?

Decisions taken in brainstorming (2026-09-10), in the order they were made:

| # | Question | Decision |
|---|---|---|
| 1 | What is frozen | The headline-split B0 checkpoint; only the prefix trains |
| 2 | What supervises the prefix | Task BCE + the `struct_new` structural-stream terms; no teacher, no bank |
| 3 | How the frozen trunk reads the prefix | Zero-init gated prefix attention in all three cross-attention layers |
| 4 | Static or pair-conditioned | Both: `prefix_static` (control) and `prefix_pair` (primary) |
| 5 | Loss weights | One 10-trial constrained study per arm through `struct_hpo` |

**Out of scope.** Prompting inside the protein language model (the brainstorming note's Option B)
is technically possible but unsupported by this work: the data package holds only frozen 1536-dim
token features under opaque node IDs, with no sequences and no encoder identity, and the
node-disjoint PPI literature shows no PEFT-of-ESM gain over frozen features (§A.2). It is recorded
here as future work and not designed for.

## 2. Why prompt tuning here is an attribution instrument, not a capacity instrument

- A prefix generated from $(x_u,x_v)$ adds no information the trunk lacks: the trunk already reads
  both endpoints in full. The binding limit found by the KD audits is informational, not
  mechanical: only 0.40--0.47 of the teacher's topology vector is linearly predictable from the pair,
  and the jointly trained descriptor head recovers descriptors at $R^2$ 0.6--0.77
  (`docs/03-experiments.md` §4.5). No injection mechanism raises that ceiling.
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
  node-disjoint, zero-observed-edge protocol*.

## 3. Task contract (unchanged)

Inference input is exactly $(x_u,x_v)$, the two frozen token sequences; output is the symmetric
probability $\widehat A_{uv}$. The prefix generator reads only the frozen encoder's summary of those
two sequences. No observed edge, neighbour, degree, retrieval result, or test statistic enters at
training or inference. Training topology supervises the prefix through the structural stream;
graph assembly remains an evaluation operation (`docs/02-methodology.md` §1).

## 4. Arm identity and the frozen base

- **Model family** `v3_1_prefix`, two arms selected by `model.config.prefix.conditioning`:
  `static` (`prefix_static`) and `pair` (`prefix_pair`).
- **Frozen base:** the headline-split B0 published checkpoint
  `outputs/split_seed42/b0_v31/best.pt` (root `node_007630`, split seed 42, seed 0), named by
  `model.config.prefix.base_checkpoint`. At construction the family loads the base `V3_1` from that
  file's embedded config and `model_state`, sets `requires_grad=False` on every base parameter, and
  keeps the base in eval-mode semantics for dropout, token dropout, and stochastic depth (the frozen
  function is deterministic; the prefix sees the same trunk the deployable B0 uses).
- **Published checkpoint** carries the full state (frozen base + prefix) so `score_universe`,
  `test_protocol`, and the interventions rebuild the arm from the checkpoint alone. The base
  checkpoint path and its file hash are written into `run_metadata.json` as provenance; nothing
  verifies them (no digest pinning, per project rules).
- **Training stream** identical to `struct_new`: official 1:5 task rows with positive weight 5, plus
  one sampled 40-node training subgraph per optimizer step (32 locally expanded + 8 background
  nodes, bfs/motif/bridge mix 50/25/25) scored as a logit matrix over legal pairs. The only
  difference from the joint `struct_new` arm is the trainable parameter set.

## 5. The prefix module

Let $d=512$ be the trunk width, $L=3$ the number of cross-attention layers, $m=16$ the prefix length.

### 5.1 Parameters

- **Static prefix.** Per layer $\ell$, $P_0^{(\ell)}\in\mathbb R^{m\times d}$. Initialised by
  sampling $m$ token states from the frozen Siamese encoder's output over randomly drawn training
  nodes (the real-activation initialisation Prefix-Tuning found stabilising; PEFT-SP does the same
  with amino-acid embeddings). About 25k parameters.
- **Pair generator (`pair` only).** With $e_u, e_v\in\mathbb R^{d}$ the frozen encoder's masked-mean
  token vectors,
  $$c_{uv}=[\,e_u+e_v;\ |e_u-e_v|;\ e_u\odot e_v\,]\in\mathbb R^{3d},\qquad
    z_{uv}=g_\phi(c_{uv})\in\mathbb R^{128},$$
  where $g_\phi$ is LayerNorm → Linear($3d$,128) → GELU. Per layer a zero-initialised
  $B_\ell\in\mathbb R^{d\times 128}$ gives the shift $s^{(\ell)}_{uv}=B_\ell z_{uv}$, added to every
  prefix token:
  $$P^{(\ell)}_{uv}=P_0^{(\ell)}+\mathbf 1\,(s^{(\ell)}_{uv})^{\top}.$$
  Because $B_\ell=0$ at initialisation, `prefix_pair` starts identical to `prefix_static`. Symmetric
  $c_{uv}$ makes $P_{uv}=P_{vu}$, so the AB/BA doubled batch and the `abba_max` readout are
  unaffected. About 0.4M parameters.
- **Gates.** One vector $g^{(\ell,a)}\in\mathbb R^{H}$ per layer and per attention site
  $a\in\{\mathrm{A\leftarrow B},\ \mathrm{B\leftarrow A},\ \mathrm{CLS}\}$, $H=8$ heads, initialised
  to zero. Nine gates, 72 scalars.

### 5.2 Reading inside a frozen `CrossAttentionLayer`

Each frozen layer (`classifier/layers.py: CrossAttentionLayer`) has two directional attentions
sharing `self.attn` and a CLS attention `self.attn_cls`. For an attention site with frozen
projections $W_Q,W_K,W_V,W_O$, normalised queries $\tilde Q$, and partner keys/values $K,V$:

$$
O_{\text{partner}}=\operatorname{softmax}\!\big(\tilde Q W_Q (K W_K)^{\top}/\sqrt{d_h}\big)\,V W_V,
\qquad
O_{\text{prefix}}=\operatorname{softmax}\!\big(\tilde Q W_Q (P W_K)^{\top}/\sqrt{d_h}\big)\,P W_V,
$$

$$
O=\big(O_{\text{partner}}+\tanh(g)\odot_{\text{head}} O_{\text{prefix}}\big)\,W_O .
$$

- The prefix softmax is **separate** from the partner softmax (LLaMA-Adapter zero-init attention),
  so the prefix never dilutes attention over real partner tokens, and $\tanh(0)=0$ gives exact null
  identity: with all gates at zero the arm reproduces the frozen B0 logits bit for bit.
- The prefix passes through the *frozen* $W_K,W_V$ of that site; the wrapper reuses
  `nn.MultiheadAttention`'s packed `in_proj_weight`/`in_proj_bias` and `out_proj` rather than
  copying them. Prefix positions carry no padding mask.
- Sites: the A-over-B attention (partner = B tokens), the B-over-A attention (partner = updated A
  tokens), and the CLS attention (partner = concatenated A and B tokens) — all three, in all three
  layers. The FFN sub-layers are untouched.
- Dropout inside the frozen attention is disabled (eval semantics); the prefix path has no dropout.

### 5.3 Compute

The frozen Siamese encoder runs under `torch.no_grad`; only the three cross-attention layers hold
activations for backward, and gradient reaches only prefix parameters. A step costs roughly a full
forward plus a partial backward of the trunk. No per-node activation cache is built (YAGNI: the
encoder forward is a small fraction of a step).

## 6. Training and hyperparameter study

- **Loss.** $\mathcal L=\mathcal L_{\text{task}}+\sum_t w_t\,\mathcal L_t^{\text{struct}}$ with the
  `struct_new` terms on the sampled subgraph: subgraph BCE, neighbour rank (margin 0.1), node-wise
  degree, and open/closed motif counts, exactly as `src/distill/struct_config.py` and the
  `train_b0` structural stream implement them.
- **Optimizer.** AdamW over prefix parameters only, weight decay 0.05, onecycle schedule, 25 epochs,
  grad clip 1.0, bf16 DDP; early stopping on validation task loss with `eval.patience: 5`,
  `eval.topology_every: 2` (campaign settings). Learning rate is searched (prefix modules need a
  higher rate than full networks and are known to be rate-sensitive).
- **Study.** `src/experiments/struct_hpo.py` gains arms `prefix_static` and `prefix_pair`, each a
  10-trial constrained MO-TPE study (2 enqueued priors, 3 startup trials) with search box
  `rank` [0.1, 3.0], `degree` [0.01, 1.0], `motif` [0.01, 1.0], `lr` [1e-4, 1e-2], all
  log-uniform. Objectives and winner selection are the campaign's: maximise AUPRC and GS, minimise
  geometric-mean MMD; final winner by the five-metric equal mean rank
  (`geometric_rd_five_rank_v1`, `docs/03-experiments.md` §1.2). Priors: (`rank` 1.0, `degree` 0.1,
  `motif` 0.1, `lr` 1e-3) and, when the `struct_new` study has finished, its winner's weights with
  `lr` 3e-3 (otherwise `rank` 3.0, `degree` 0.5, `motif` 0.5, `lr` 3e-3).
- **Held-out.** Each arm's winner runs the held-out protocol once, at its V_val-frozen thresholds.

## 7. Controls, interventions, and how the result is read

**Comparators** (all on the headline split, seed 0):

| Row | Role |
|---|---|
| Frozen B0, rescored through the same V_val selection | the exact function the prefix modifies; its own threshold |
| `struct_new` winner | the same supervision with every parameter trainable |
| `prefix_static` winner | recalibration control: shared prompt, no pair information |
| `prefix_pair` winner | primary |

**Interventions** (scoring-only, on the selected `prefix_pair` checkpoint, thresholds frozen):

1. *Gates off* — all $g\leftarrow 0$; must reproduce frozen B0 exactly (a correctness check, reported).
2. *Shuffle* — $z_{uv}$ permuted across pairs within each scored universe.
3. *Mean* — $z_{uv}$ replaced by its training-set mean.

Each reports AUPRC with the five topology numbers (BFS-macro GS, geometric RD, degree /
clustering / spectral MMD ratios) at the frozen threshold, alongside the gate telemetry
$\tanh(g)$ per site and epoch.

**Reading.** Edge-level and assembled-graph families are always reported together
(`CLAUDE.md` claim rules).

- A *conditioning* claim requires `prefix_pair` to beat frozen B0 on GS and the MMD ratios beyond
  the split's noise band (about ±0.01 GS, ±0.5 MMD ratio from the B0 seed pair), to beat
  `prefix_static`, and to lose that margin under shuffle.
- If `prefix_static` matches `prefix_pair`, the prompt recalibrated the frozen trunk; report it as
  such.
- If both match frozen B0, the frozen trunk bounds what any prompt can move under this
  supervision; that bound is the result and is reported, not buried.
- Selected epochs are reported; the epoch confound that erased earlier KD gains does not apply to a
  frozen trunk, but the comparison against `struct_new` still checks it.

**Deferred** until `prefix_pair` shows a gain worth attributing: the same-condition pooled adapter
on the frozen trunk (the "injection mechanism" comparison), prefix length {4, 64}, the
reparameterisation MLP on $P_0$, and a teacher-KD prefix (`kd_rank` through the prefix) or a
descriptor-bottleneck prefix.

## 8. Implementation surfaces

- **New module** `src/model/egostitch/classifier/prefix.py`: `PrefixGenerator` ($P_0$, $g_\phi$,
  $B_\ell$, gates), `GatedPrefixAttention` (the per-site wrapper over a frozen
  `nn.MultiheadAttention`), `PrefixCrossAttentionLayer` (wraps one frozen `CrossAttentionLayer`),
  and `V3_1Prefix` (loads and freezes the base `V3_1`, wraps its `cross_attention.layers`, exposes
  `prefix_parameters()`, and forwards with the same `PairInputs`/loss interface as `V3_1`, including
  the structural-stream logit-matrix path).
- **Family dispatch.** `train_b0.MODEL_FAMILIES` and `build_model` accept `v3_1_prefix`;
  `_build_optimizer` builds its single param group from `prefix_parameters()`;
  `score_universe.MODEL_BUILDERS` rebuilds the family from the checkpoint's embedded config (the
  frozen base state is inside the checkpoint, so no base file is needed at scoring time).
- **Interventions.** A `--prefix-intervention {none,gates_off,shuffle,mean}` scoring option;
  `mean` uses a `z_mean` buffer computed at publish time over the training rows.
- **Configs.** `configs/split_seed42/prefix_static.yaml` and `prefix_pair.yaml`: the `struct_new`
  config with `model.family: v3_1_prefix`,
  `model.config.prefix: {base_checkpoint, tokens: 16, conditioning: static|pair, bottleneck: 128}`,
  `optim.lr` as the prior, and `output_dir: outputs/split_seed42/prefix_{static,pair}`.
  `struct_hpo` writes trials under `outputs/struct_hpo/prefix_{static,pair}`.
- **Tests** (`tests/test_prefix_model.py`): null identity against the frozen base (bitwise logits
  at zero gates); AB/BA symmetry of the prefix; no gradient on any base parameter after a backward;
  `prefix_pair` equals `prefix_static` at initialisation; shuffle and mean interventions change
  logits only through $z_{uv}$; config parsing and family dispatch; checkpoint round trip through
  `score_universe.build_model`.
- **Docs in the same change:** arm table in `docs/03-experiments.md` §1.4, the active method set in
  `CLAUDE.md`/`AGENTS.md`, and `src/experiments/struct_hpo.py` docstring. A result note under
  `docs/results/prefix_split_seed42/` follows the runs.
- **Execution order.** Local: module, tests, configs, docs, Codex review, commit, push. H20: the two
  studies (`.venv/bin/python -m src.experiments.struct_hpo --arm prefix_static` / `--arm
  prefix_pair`) once `outputs/split_seed42/b0_v31/best.pt` is published, then each winner's held-out
  run and the three interventions.

## 9. Risks and the answers designed in

| Risk | Answer |
|---|---|
| The prefix cannot move topology under a frozen trunk | Reported as a bound; the joint `struct_new` row shows what full training moves |
| Static and pair prefixes tie | Reported as recalibration; the pair generator is then not credited |
| Structural terms shift the logit scale, hurting raw ECE/Brier (seen in `struct_grand` trial 0) | V_val threshold selection absorbs the shift for topology; ECE/Brier are reported raw beside it |
| A prefix in a joint softmax would perturb the frozen attention at step 0 | Separate softmax + zero-init gates give exact null identity |
| Learning-rate sensitivity of prompt modules | `lr` is in the search box |
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
  generator; for pair classification the extra pass is the base forward itself.*
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
- **Pair-conditioned prefix:** a shared prefix plus a per-pair shift generated from the two endpoint
  summaries.
- **Topology-supervised prefix:** a prefix whose only structural signal is a loss computed on the
  predicted adjacency of sampled training subgraphs.

# EgoStitch Algorithm Specification (Gate G4 deliverable — spec freeze)

**Status:** **G4 signed off 2026-07-09 — this document is the active implementation
contract.** Companion to `04-model-proposal.md` (revision 2.2); satisfies gate
§6.0-G4: Stitch/Harmonize pseudocode with tensor shapes, OT cost and ε, confidence and
quantile schedule, budget tolerance, gradient estimators, and the full loss tree with
interior weights. Neutral placeholders per repository convention; no dataset names.
Nothing here changes the model of `04-model-proposal.md` — this document pins the free
parameters that document left symbolic. §§9–11 (added at sign-off) bind the spec to
the local benchmark package in `data/`, define the batch-sampler / data contract, and
fix the four-H20 E2 production execution design.

**Freeze rule.** This spec is signed off: implementation may not silently deviate;
any change is an edit here first, with a one-line rationale in §12 (change log).

---

## 0. Notation and shapes

| Symbol | Meaning | Shape / default |
|---|---|---|
| `d` | frozen feature dim (after F0 pooling, §9.2) | 1536 (benchmark package: per-node token sequence `(L, 1536)`, `L ≤ 1024`) |
| `d_p` | projected feature space (set-decoder target space) | 256 |
| `d_z` | code / residual dim | 64 |
| `d_h` | decoder hidden dim | 256 |
| `d_ζ` | CVAE latent dim | 32 |
| `M` | codebook size | 256 (sweep {64, 256, 1024}) |
| `C` | affiliation communities | 64 (sweep {16, 64, 256}) |
| `K` | neighbor slots per node | 16 (sweep {8, 16, 32}) |
| `m_max` | max multiplicity per slot | 32 (fixed) |
| `n_g` | grounding candidates per node | 20 |
| `n_s` | CVAE samples at inference | 4 (fast variant 1) |
| `R` | harmonization rounds | 2 (sweep {0, 2, 4}) |
| `B` | batch of nodes / pairs | — |

All modules are per-node unless marked per-pair. `MLP_k(a→b)` = k-layer MLP from dim a
to dim b, GELU, LayerNorm.

---

## 1. Module 1 — Tokenize (per node, cached)

```text
e_u  = MLP_2(d → d_z)(x_u)                                   # encoder
m(u) = argmin_j ‖e_u − c_j‖₂        z_u = c_{m(u)}           # VQ, codebook C ∈ R^{M×d_z}
r_u  = e_u − z_u                                             # continuous residual
F_u  = softplus(MLP_2(d + d_z → C)([x_u; z_u]))              # BP affiliations, ≥ 0
d̂_u = softplus(MLP_2(d + d_z → 1)([x_u; z_u])) · ρ_eval/ρ_train
                                                             # degree budget, density-
                                                             # normalized (ρ = candidate-
                                                             # universe edge density)
stats_u = MLP_2(d_z → 4)(z_u)   # code-supervision head: [deg, clustering,
                                # code-histogram-entropy, motif-conductance] targets
```

- VQ: EMA codebook updates (decay 0.99), straight-through gradient, commitment
  β = 0.25; code-usage entropy regularizer weight in §7.
- Degree NLL: lognormal parameterization (μ, σ from a 2-head MLP) — heavy-tailed
  degrees; the table above shows the mean head.
- All structural targets computed on the **message partition** (§6).

## 2. Module 2 — Imagine (per node, cached; DETR-style decoder)

```text
cond tokens:  T_cond = [W_x x_u; W_z z_u; W_r r_u; W_ζ ζ_u]           # 4 tokens, d_h
cand tokens:  T_g    = W_g proj(x_{g_1..g_{n_g}})                      # n_g tokens, d_h
queries:      Q_k    = W_q [proj(x_{g_k}); z_u] for k ≤ min(K, n_g^+)  # dynamic init
              Q_k    = q_k^base + W_q' z_u      otherwise              # learned base
decoder:      3 layers; self-attn over Q; cross-attn to [T_cond ; T_g ; T_peer]
              (T_peer = partner kept-slot tokens; empty outside harmonization)
heads (per slot k):
  h_u^k ∈ R^{d_p}          slot embedding (projected feature space)
  π_u^k ∈ [0,1]            existence (sigmoid)
  m_u^k ∈ [1, m_max]       multiplicity (1 + softplus, clamped)
  g_u^k ∈ [0,1]            grounding gate (sigmoid) + pointer softmax over T_g
  Â_u ∈ [0,1]^{K×K}        slot–slot adjacency (symmetric bilinear head on h)
CVAE: prior ζ_u ~ N(0, I_{d_ζ}); posterior q(ζ_u | pooled true-neighbor tokens) at
      train; free-bits 0.5 nats/dim; KL annealed 0→1 over first 10% steps.
```

**Conditioning dropout (two nulls, trained):** with p = 0.1 each (disjoint draws):
`∅_content` replaces T_cond by a learned null token (T_g kept); `∅_all` replaces both
T_cond and T_g by nulls. Inference contrasts: TDE control = fused-logit(full) −
fused-logit(∅_content); CFG knob (logit space only): s̃ = s(∅_all) + w·(s(full) −
s(∅_all)), default w = 1 (no guidance) — w ≠ 1 reported only as an analysis arm.

**Hub policy (targets when |N(u)| > K):** stratified importance subsample of N(u) —
proportional allocation over neighbor-code strata, capped at K targets; per-target
multiplicity label = stratum_size / allocated_count. Matching supervises (h, π, m)
against the subsample; degree NLL keeps the true |N(u)|.

**Hungarian matching (compound cost, per node):**

```text
C_{k,v} = 1·‖h_u^k − proj(x_v)‖₂² + 0.25·|deg_bucket(k) − deg_bucket(v)|
        + 0.25·1[code(h_u^k) ≠ code(x_v)] + 0.25·overlap_penalty(k, v)
```

`overlap_penalty` = disagreement between Â row of slot k and A row of v restricted to
currently matched pairs (second Hungarian pass; first pass uses the first three
terms). Assignment is recomputed per step and **treated as a constant** in the
backward pass (standard DETR practice). **Denoising queries:** 25% of training nodes
add K/2 extra queries initialized at noised true-neighbor projections
(σ_noise = 0.1) with *fixed* assignments — the assignment-flapping stabilizer.
Diagnostic: assignment flip rate between consecutive epochs, logged per degree bucket.

## 3. Module 3a — Stitch (per pair)

```text
Π = Sinkhorn(C^Π, ε = 0.1, 20 iters, unrolled/differentiable)   ∈ R_{≥0}^{K×K}
C^Π_{kk'} = 1·‖h_i^k − h_j^{k'}‖₂² + 0.25·1[code_k ≠ code_{k'}] + 0.25·|π_i^k − π_j^{k'}|
marginals: a_k ∝ π_i^k m_i^k ;  b_{k'} ∝ π_j^{k'} m_j^{k'}   (unbalanced OT, KL
           relaxation τ_OT = 1.0 — slots may be unmatched)
```

Scaffold `T̂_ij`: nodes = {i, j} ∪ slots(i) ∪ slots(j) (2K+2; aligned slots kept as
distinct nodes joined by alignment edges weighted Π_{kk'}). Node features =
`[proj feature; one-hot anchor label (5); π; g]`. Edges: star edges (u,k) weighted
π_u^k; slot–slot edges weighted Â; alignment edges weighted Π. Anchor labels:
endpoint-i / endpoint-j / slot-of-i / slot-of-j / grounded-identity-match.

## 4. Module 3b — Harmonize (per pair; trained; algorithm box)

```text
Inputs: S_i, S_j (cached), d̂_i, d̂_j, R
B_u   := min(d̂_u, K·m_max)                # K-representable budget
τ_b   := 0.1 · B_u                        # budget tolerance
conf_k := 0.5·π̂_k + 0.5·exp(−‖h_k − c_nn(h_k)‖₂² / τ_c),  τ_c = median intra-code
          distance on validation; π̂ = temperature-calibrated π (scalar T fit on
          held-out real ego-nets, required)
for r = 1..R:
    γ_r := cos(π/2 · r/R)                            # keep fraction schedule (MaskGIT)
    per scaffold: KEEP the top (1−γ_r)·2K slots by conf; grounded slots (g_k > 0.5)
        get re-mask probability ×0.25 (reduced, not exempt)
    CRITIC: p_remask(k) ← p_remask(k) · (1 + critic(h_k, Â_k·)) — critic is a 2-layer
        scorer trained on real-vs-generated slot neighborhoods; MAY re-mask kept slots
    BUDGET: if Σ_k π_u^k m_u^k ≥ B_u − τ_b: freeze (mask) further slot activations and
        closure edges incident to u for all later rounds       # scaffold-level only
    RE-DECODE masked slots of each side with T_peer = partner kept slots
    if r == 1: recompute Π; else: Π frozen
Output: T̂_ij; diagnostic: slot-agreement trajectory agree_r = Σ Π_{kk'} conf_k conf_{k'}
```

**Joint training task:** sample message-partition pairs (u,v) — 50% adjacent / 50%
random (label-agnostic, R8) — mask each ego-net at ratio ~ U(0.3, 1.0), run 1–2
harmonization rounds, supervise re-decoded slots by the §2 matching losses.
**Gradient estimators:** keep/re-mask = straight-through on the binary keep mask;
budget trigger = detached comparison + soft mirror penalty
`0.1·ReLU(Σπm − B_u)²` so the constraint surface reaches gradients; Sinkhorn unrolled;
critic trained with its own BCE, gradients not propagated into h (detached input).

## 5. Module 4 — Decision head (per pair)

```text
s0 = pair_logit(i, j)                                    # frozen B0
s1 = ½[ lse_k(κ(h_i^k, proj(x_j)) + log π_i^k m_i^k) + (i↔j) ],  κ = −‖·‖₂²/τ_κ
s2 = Σ_{kk'} Π_{kk'} π_i^k π_j^{k'}  and AA variant Σ Π π π / log(1+deĝ)
s3 = [1 − exp(−F_i·F_j);  d̂_i;  d̂_j;  Σπm/B_i;  Σπm/B_j]        # post-harmonization
s4 = MLP_2([H_i; H_j; H_T; spec(T̂)])   from a 3-layer edge-weighted GNN over T̂_ij
     (anchor-labeled; spec(T̂) = [λ_2, λ_max, triangle count, density] of T̂)
p_ij = σ( s0 + g_θ(s1..s4) · w ),   g_θ = MLP_2(gate), w learned scalar init 0.1
```

Required diagnostic: Pearson/Spearman correlation matrix of (s0..s4) on validation
queries, reported with every headline table.

## 6. Data partitions and leakage rules

- Training graph edges split **80% message / 20% supervision** (per seed). All
  structural targets (reconstruction, degree NLL, BP-NLL, code stats, seam references,
  critic training) use message edges only; `L_edge` positives/negatives come from
  supervision edges (+ post-masking negatives). Leave-one-out: when a supervision pair
  (u,v) is in the batch, v is excluded from u's targets and |N(u)| decremented.
- Seam references: unions of message-partition ego-net pairs sampled 50/50
  adjacent/random with labels marginalized.
- B0 provenance audit is an E5 gate precondition. The E2 B0 scorer is
  pinned to the audited V3.1 `pair_context_gated` / `abba_max` / no-cross checkpoint
  family (`d_model = 512`, no spectral normalization) and trains from the fixed
  balanced `train_edges.txt` rows when a local retrain is required.
- **Benchmark binding:** how the shipped artifacts map onto message/supervision/val,
  which shipped files are quarantined, and the self-loop policy are fixed in §9 —
  §9 is normative wherever the shipped artifacts differ from the abstract wording
  above.

## 7. Loss tree (interior weights; pre-registered defaults)

```text
L = L_edge + λ_real·L_real + λ_ssl·L_ssl + λ_recon·L_recon
λ_recon = 1.0    λ_real = 0.5    λ_ssl = 0.1        # sweep grid: each ∈ {0.25×, 1×, 2×}

L_edge  = BCE(p_ij, y)                                              # master
L_recon = 1.0·L_feat(Hungarian, Huber) + 0.5·L_exist(BCE, ∅-balanced)
        + 0.25·L_mult(NLL) + 0.5·L_deg(lognormal NLL)
        + 0.5·L_slotadj(group-level BCE | bandwidth variant)
        + β_KL·KL(free-bits 0.5)                     β_KL annealed 0→1
        + 0.5·L_BP(NOCD balanced NLL) + 0.25·L_VQ(commit β=0.25, EMA codebook)
        + 0.25·L_codestats(Huber) + 0.05·L_entropy(code usage)
        + 0.5·L_joint(harmonization re-decode task, §4)
        + 0.25·L_gate(partner-vs-peer BCE on grounding gate)
L_real  = 0.5·ED(ego-stat vectors) + 0.25·ED(random-GIN embeddings)
        + 0.25·ED(seam overlap stats)                [+ 0.1·adversarial, off by default]
L_ssl   = 0.5·consistency(feature noise σ=0.05) + 0.5·pool-resample consistency
          (applied to ungrounded slots only; mean g^k logged)
```

Balancing: fixed weights above; gradient-norm per family logged; uncertainty-weighting
(Kendall) is the registered fallback if any family's gradient norm drifts > 10× from
the median for > 1k steps.

## 8. Training schedule, HPO parity, determinism

1. **Warm-start** (Modules 1–2): `L_recon` only, 20% of budget.
2. **+ Joint harmonization task**: add `L_joint`, 20%.
3. **Full joint**: all losses, 60%; early stopping on validation edge AUPRC
   (VAL-CRITERION), patience 10 evals.
- **HPO parity:** every ladder method (B0…B5, `B3-full`, `Ours`) gets the same tuning
  budget: 30 configs × 3 seeds, random search over its declared grid, frozen and
  recorded in the run-metadata store before held-out metrics are opened.
- **Determinism:** fixed seeds for CVAE samples (n_s = 4, averaged p_ij), Sinkhorn
  iterations fixed, harmonization schedule deterministic given seed. One inference =
  one seed set; reported metrics average 3 seeds.

## 9. Benchmark binding and data contract

Binds this spec to the artifact package `data/benchmark_2025_neurips/` +
`data/features/frozen_node_features_1024/` (see `data/README.md` for the shipped
layout). All numbers below were measured directly from the artifacts on 2026-07-09;
the loader must re-verify them at build time and fail loudly on drift.

### 9.1 Artifact inventory (measured)

Global: `graph.pkl` = undirected `networkx` graph, **10,090 nodes / 129,861 edges, of
which 7,769 are self-loops** (77% of nodes carry one). No node/edge attributes.
`positive_edges.txt` = the same edge set as a TSV. Three split strategies, each with
**8,072 train / 2,018 test nodes, node-disjoint** (test sets differ across strategies;
pairwise overlap 547–745 nodes):

| Strategy | train⁺ (sup) | val⁺ | test-graph edges (self-loops) | ρ_eval |
|---|---|---|---|---|
| `random_walk` | 35,288 | 8,811 | 30,257 (1,701) | 1.485e-2 |
| `breadth_first` | 42,880 | 10,760 | 32,019 (1,891) | 1.572e-2 |
| `depth_first` | 56,763 | 14,256 | 13,437 (1,708) | 6.60e-3 |

Per strategy: `train_edges.txt` / `val_edges.txt` = balanced (~1:1) labeled pairs
**over train-side nodes only**; negatives avoid the *global* positive set.
`test_edges.txt` = balanced labeled pairs over test-side nodes only.
`candidate_test_edges.txt` = the complete test-side universe: all C(2018, 2) = 2,035,153
unordered pairs **plus all 2,018 self-pairs** (2,037,171 rows); its positives equal the
`test_graph.pkl` edge set exactly. `test_node_buckets.pkl` = 50 node subsets at each
size {20, 40, …, 200} for bucketed assembled-graph evaluation.

Identities verified: `train_graph.pkl` edges = train⁺ ∪ val⁺ exactly (val⁺ is the
20% complement of train⁺); train/val/test negatives ∩ global positives = ∅.

**Benchmark-A/B/C ↔ strategy mapping** is recorded in the run metadata. The completed
G1 rerun uses `Benchmark-A = breadth_first`; its artifacts are under
`outputs/e2_resubmit_retry/`.

### 9.2 Feature pipeline (F0/F1)

`metadata.json`: format `torch_pt_per_node`, token dim 1536, max length 1024. Each
node's feature is a **variable-length token sequence** `(L, 1536)` (sampled: L ∈
[67, 1001], median ≈ 394, mean ≈ 445; none at the cap; ~25 GB total across 10,193
indexed nodes). `index.json` maps node id → content-addressed `.pt` path.

- **F0 (default, frozen):** length-masked mean-pool over tokens → `x_u ∈ R^1536`.
  Computed once at cache-build, stored as a single fp32 matrix
  `(|V|, 1536)` (~62 MB) + row index; every rank loads the full matrix. All spec
  modules consume `x_u`; `d = 1536`.
- **F1 (ablation arm):** 1-layer attention pooler (learned query, `d_h`) over raw
  token sequences, trained end-to-end; requires the length-bucketed loader path
  (§10.4). Registered as ablation E4.11; F0 is the headline configuration.

**Operative node set** `V` = `graph.pkl` nodes ∩ `index.json` keys = **10,088 of
10,090** (`node_004764`, `node_007050` lack features; both train-side, degree 1;
3 train pairs + 1 val pair touch them → dropped at load, counts logged). The 105
indexed nodes absent from the graph are inert.

### 9.3 Partition binding (message / supervision / val)

The shipped artifacts do **not** ship the §6 message/supervision partition; it is
derived at load, per seed:

```text
E_train⁺ := positives of train_edges.txt            (per strategy)
E_msg    := seeded 80% of E_train⁺ ;  E_sup := the remaining 20%
G_struct := simple graph (V_train, E_msg \ self-loops)     # ALL structural targets
L_edge positives := E_sup (self-pairs included, §9.4)
val_edges.txt    := model selection only (VAL-CRITERION, §8) — never a target
```

**Quarantine (binding):**

1. `train_graph.pkl` **contains every val positive** — it must never be used as a
   structural-target source (ego-nets, degrees, codebook stats, seam references,
   critic training). Split-audit / consistency checks only. Everything structural
   derives from `G_struct`.
2. `*_ratio5_exclusive.txt` (all three) draw negatives from the **global** node set —
   measured on `random_walk` train: 112,577 train–train, 56,958 cross-split, 6,905
   test–test pairs. Cross-split pairs expose test-side node features at training
   time: **prohibited** for training, model selection, and headline evaluation.
   At most a clearly-labeled comparability appendix row.

Density normalization (`ρ_eval/ρ_train`, §1): ρ := |E⁺| / (C(|V_side|,2) + |V_side|)
on the matching universe. Measured full-train ρ_train (random_walk) = 1.354e-3 vs
ρ_eval = 1.485e-2 — an **~11× train→test density shift** (mean simple degree 9.4 →
28.3); per-seed ρ_train is computed on `E_msg` at load. Because true test density is
**not observable** under the strict gate, the inference-time ratio is pinned as:

- **default — self-calibrated, two passes:** pass 1 runs the candidate universe with
  ratio 1 and no budget re-scale; ρ̂_eval := Σ p⁰_ij / |candidates| (model outputs
  only — protocol-clean); pass 2 re-runs with ratio ρ̂_eval/ρ_train.
- **diagnostic rows:** ratio = 1 (no calibration) and ratio = true test density
  (**Oracle-family — protocol-violating, reference only**).

### 9.4 Self-loop policy

Self-loops are first-class labeled queries in this benchmark (13.8% of `random_walk`
train-graph edges; 84% of test nodes carry one; they appear in supervision, val,
test, and candidate files as `(u, u)` rows). Binding rules:

1. **Structural-target simple-graph policy; canonical MMD exception**: N(u), degrees,
   budgets d̂_u, training-side clustering/code stats, ego-net targets, relative
   density, recall, and other structural targets strip self-loops. Canonical MMD
   descriptor induced subgraphs retain self-loops exactly as in the
   benchmark/official evaluator.
2. **`(u, u)` queries route through a single-ego path**: j := i; T_peer = own kept
   slots; Π = identity on kept slots; s0 = pair_logit(u, u); s1 = self-membership
   `lse_k(κ(h_u^k, proj(x_u)) + log π m)`; s2 from the Â_u diagonal blocks;
   s3 unchanged; s4 on the single-ego scaffold with both anchor labels on u.
3. **Reporting**: edge metrics overall *and* split self / non-self; canonical MMD on
   loop-retaining descriptor induced subgraphs; relative density, recall, and other
   topology metrics on the simple assembled graph; plus a separate self-loop-rate row
   (predicted vs reference, e.g. 1,701/2,018 on `random_walk`).

## 10. Batch sampler and loader contract

### 10.1 Streams and composite step

One optimizer step consumes three independently-sampled minibatches in the single
process (defaults sweep ±2×):

| Stream | Source | Default | Feeds |
|---|---|---|---|
| node stream | uniform over V_train (`accelerator.prepare`, §11.0) | B_n = 256 nodes | L_recon, L_ssl |
| joint-pair stream | 50% E_msg edges / 50% random train pairs (§4) | B_p = 128 pairs | L_joint |
| edge stream | E_sup positives + resampled negatives | B_e = 512 pairs (1:5 → ~85 pos) | L_edge |

Curriculum (§8) toggles streams: stage 1 node-only; stage 2 node + joint; stage 3
all. An **epoch** = one full pass over E_sup in the edge stream; node and joint
streams cycle independently.

### 10.2 Negative sampling (training)

For EgoStitch and its topology-aware training arms, negatives are resampled every
epoch, seeded by (seed, epoch, idx):
50% uniform train-side
pairs, 50% degree-corrected (endpoint corruption of a positive with replacement
probability ∝ deg_G_struct(v)); self-pair negatives sampled at their universe rate;
rejection against the **global** positive set (matching the shipped negatives'
convention) via an in-process hash set. Default ratio 1:5. The shipped balanced
negatives in `train_edges.txt` / `val_edges.txt` are the **fixed** diagnostic and
model-selection negative sets for those arms. The E2 B0 scorer is the
sole exception: it trains on the fixed balanced `train_edges.txt` pairs as shipped,
with per-epoch order shuffling but no negative resampling.

### 10.3 Evaluation loaders

- **Val (every eval epoch):** `val_edges.txt` as-is; VAL-CRITERION per §8.
- **Test edge metrics:** `test_edges.txt` as-is; run once, after freeze.
- **Assembled-graph eval:** `candidate_test_edges.txt` served by one prepared
  DataLoader (single process, §11.0); batch 8,192 pairs; n_s = 4 fixed CVAE seeds
  averaged; two passes (§9.3 density self-calibration); logits + pair ids collected
  locally; assembly and bucketed metrics (`test_node_buckets.pkl`) computed in the
  same process.
- The assembled evaluator uses the protocol's single canonical MMD ratio: fixed
  Gaussian-TV (`σ=1`) raw MMD² divided by the deterministic even/odd reference
  floor after separately averaging numerator and denominator across node-size
  buckets. Its descriptor induced subgraphs retain self-loops exactly as in the
  benchmark/official evaluator. The spectral worker first converts its 200-bin
  counts to a PMF, while degree and clustering remain raw counts; the common MMD
  routine then applies `sum + 1e-6` normalization to all three. Run artifacts
  disclose `raw_mmd2`, `reference_mmd2`, and `mmd_ratio`; only `mmd_ratio` is used
  in result tables and the topology composite.
- Per-node Tokenize/Imagine caches rebuilt once per eval epoch for all |V| nodes
  (~10k nodes; ≈160 MB fp32 at K = 16, d_p = 256 — kept on the H20).

### 10.4 Collate

- **F0 path:** row-gather from the pooled matrix → `(B, 1536)`; no padding.
- **F1 path:** length-bucketed batching (boundaries {128, 256, 384, 512, 768, 1024}),
  pad-to-bucket-max with mask; token budget 131,072 tokens auto-sizes the batch;
  `num_workers = 4`, `persistent_workers`, `prefetch_factor = 4`, pinned memory.
- Hungarian matching runs per node on K×K ≤ 16×16 cost matrices
  (`scipy.linear_sum_assignment` on CPU from GPU-computed costs); no inter-process
  interaction.

## 11. E2 production execution design (4 × H20, Hugging Face Accelerate DDP)

The formal E2 B0 V3.1 run uses 4 × NVIDIA H20 and is launched with
`accelerate launch --num_processes 4`. A cold acceptance run includes first BF16
feature-pack construction, bounded batch probes, exactly 30 epochs, validation after
every epoch, and final artifacts. The complete interval must be at most 60 minutes.

Each rank owns one model/optimizer replica and one complete GPU-resident BF16 feature
table. DataLoader workers transfer compact endpoint indices only. Training and
validation coverage are exact; tail-batch loss is weighted by local/global pair count.
The checkpoint payload consumed by `score_universe` is unchanged.

## 12. Change log

- 2026-07-09: initial freeze draft (from 04-model-proposal.md rev 2.1 §4 + panel
  findings R1-W1/W2/W6, R3-W1/W2/W4/W5/W7).
- 2026-07-09: **G4 signed off** — spec becomes the implementation contract.
- 2026-07-09: added §9 benchmark binding (artifact inventory, F0/F1 feature pipeline,
  `d = 1536`, per-seed message/supervision derivation, `train_graph.pkl` and
  `ratio5_exclusive` quarantine, measured ρ values + self-calibrated density ratio,
  self-loop policy), §10 batch-sampler/loader contract, §11 DDP execution design
  (4/8 × H20) — user-directed additions at sign-off; §0 `d` row bound; §6 pointer
  added; former §9 change log renumbered to §12.
- 2026-07-09: §11 rebased on **Hugging Face Accelerate** (`accelerate==1.13.0`,
  installed) as the pinned distributed layer, per user direction — `Accelerator` +
  `prepare()` loader sharding replace raw `torchrun`/`DistributedSampler`;
  `reduce`/`gather_for_metrics`/`save_state`/`set_seed` bound to the sync points;
  API verified against the Accelerate docs (Context7, 2026-07-09).
- 2026-07-10: §10.3–10.4 and §11 changed first to match the allocated environment:
  one fixed H20 container, one process, BF16, direct `hpc/run.sh` execution, and no
  Slurm/DDP/NCCL path. Accelerate remains the single-process training/checkpoint
  wrapper; the previous 4/8-H20 execution plan is superseded.
- 2026-07-11: pinned the E2 B0 architecture and its fixed 1:1
  balanced training-pair exception; EgoStitch's dynamic 1:5 edge stream is unchanged.
- 2026-07-11: replaced the formal E2 single-H20 path with the approved 4 × H20
  Accelerate DDP packed-feature pipeline; fixed the cold-run budget at 60 minutes for
  30 epochs with validation after every epoch. The scorer and checkpoint contracts did
  not change.
- 2026-07-13: replaced the assembled evaluator's L2-RBF/median-bandwidth raw MMD²
  with the fixed-`σ=1` Gaussian-TV biased MMD² ratio defined in protocol §1;
  removed degree clipping and bound deterministic even/odd reference splitting,
  ratio-of-size-means aggregation, numerator/denominator disclosure, and
  benchmark-aligned self-loop retention for canonical descriptor induced subgraphs;
  also pinned the official spectral-PMF pre-normalization before the common
  `sum + 1e-6` normalization, with degree/clustering left as raw counts beforehand.

**Open items before code (not blockers):** the four ego-stat target definitions
pinned to evaluator implementations; FLOPs/latency table template (§4.7 commitment);
Benchmark-A/B/C ↔ split-strategy mapping confirmed at G1 (§9.1).

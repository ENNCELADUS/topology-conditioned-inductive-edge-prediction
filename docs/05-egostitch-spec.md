# EgoStitch Algorithm Specification (Gate G4 deliverable — spec freeze)

**Status:** **G4 signed off 2026-07-09 — this document is the active implementation
contract.** Companion to `04-model-proposal.md` (revision 2.2); satisfies gate
§6.0-G4: Stitch/Harmonize pseudocode with tensor shapes, OT cost and ε, confidence and
quantile schedule, budget tolerance, gradient estimators, and the full loss tree with
interior weights. Neutral placeholders per repository convention; no dataset names.
Nothing here changes the model of `04-model-proposal.md` — this document pins the free
parameters that document left symbolic. §§9–11 (added at sign-off) bind the spec to
the local benchmark package in `data/`, define the batch-sampler / data contract, and
fix the GPU-count-independent H20 execution design.

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

**Rev 3.0 headline (family `egostitch_e2e`, normative since 2026-07-17):** the
decision head is a from-scratch V3.1-class pair encoder conditioned on the stitched
topology — no frozen B0 anchor, no `s0`:

```text
p_ij = σ( head( Trunk(tok_i, tok_j | STE(T̂_ij), c_content) ) )
```

- **Trunk:** Siamese token encoder + pair cross-attention over the raw token
  sequences `(tok_i, tok_j)` — the audited V3.1 architecture family
  (`pair_context_gated` blocks maintaining `(h_a, h_b, cls_token)`, feature-wise
  `abba_max` over the AB and BA passes) — trained from scratch under the Ours regime.
- **STE (stitched-topology encoder):** structure-only tokens over the stitched
  scaffold `T̂_ij`. Token features: 4-type anchor labels (endpoint-i / endpoint-j /
  slot-of-i / slot-of-j), `π`, `m`, soft degrees — **no** slot content `h`, no
  grounding embeddings `g`, no grounded-identity-match flag. Edge weights: star
  edges `π·m`, intra-side `Â_i`/`Â_j` weighted by `π` outer products, and the
  alignment plan `Π`. `ste_layers` edge-weighted message-passing layers
  (defaults §13.18); **token-level output** — one conditioning token per slot and
  per endpoint (the promoted `s4` lineage: the pooled scalar summary is replaced
  by tokens).
- **Conditioning:** zero-initialized tanh-gated cross-attention
  (`cls ← cls + active · tanh(gate) · XAttn(LN(cls), tokens)`); the **cls_token is
  the only query**, injected after the final `n_inj ∈ {1, 2}` pair-cross-attention
  blocks (default 1). The AB and BA directions share STE and cross-attention
  parameters before `abba_max`.
- **`c_content` (separate ablatable pathway):** s1-style grounding summaries plus
  the grounded-identity-match flag, through its own zero-init gated
  cross-attention. The topology-representation claim must survive removal of this
  pathway (protocol E4.15).
- **Three mutually exclusive head nulls** (train = per-pair multiplicative masks at
  `p_topo`/`p_cont`, eval = batch-level hard bypass; the residual sublayer form
  makes the two numerically identical — required unit test; `p(i,j) = p(j,i)` under
  every null): `∅_all_head` (skip both pathways → pair-only `f_logit`),
  `∅_topo_head` (skip STE + topology x-attn → pair+content), `∅_content_head`
  (skip content x-attn → pair+topology). The `_head` namespace is disjoint from
  the §2 conditioning-dropout decoder nulls.
- **Four logits published per scored pair:** full, `f_logit`, pair+content,
  pair+topology (§13.16 fp32 pin applies to all four).

Channel disposition (rev 2.2 → 3.0): `s4` is promoted into the STE; `s1` feeds
`c_content`; `s0` is retired (§13.10); `s2` remains a training-side diagnostic and
probe target; `s3` remains a Stage-2 STE input.

**Retired anchored head** (frozen-s0 family `egostitch`; motivating result and E4
ablation arm only — binding `cut` verdict 2026-07-17,
`docs/results/G5-stage1-seed0-20260717.md`):

```text
```text
s0 = pair_logit(i, j)                                    # frozen B0
s1 = ½[ lse_k(κ(h_i^k, proj(x_j)) + log π_i^k m_i^k) + (i↔j) ],
κ(a,b) = −‖normalize₂(a) − normalize₂(b)‖₂²/τ_κ
s2 = Σ_{kk'} Π_{kk'} π_i^k π_j^{k'}  and AA variant Σ Π π π / log(1+deĝ)
s3 = [1 − exp(−F_i·F_j);  d̂_i;  d̂_j;  Σπm/B_i;  Σπm/B_j]        # post-harmonization
s4 = MLP_2([H_i; H_j; H_T; spec(T̂)])   from a 3-layer edge-weighted GNN over T̂_ij
     (anchor-labeled; spec(T̂) = [λ_2, λ_max, triangle count, density] of T̂)
p_ij = σ( s0 + g_θ(s1..s4) · w ),   g_θ = MLP_2(gate), w learned scalar init 0.1
```
```

Required diagnostic: the four-logit decomposition table (plus
`topology_delta = full − pair+content` and `content_delta = full − pair+topology`
summary statistics) with every headline table; the retired arm keeps its channel
correlation matrix on its available channels `(s0..s4)`.

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

**Family `egostitch_e2e` (2026-07-17):** the conditioned encoder introduces **no new
loss lambda** — the locked objective is unchanged. `L_edge = BCE` applies to the
**full** logits computed under the per-pair §5 branch masks; trunk, STE, gates, and
head receive gradient only through `L_edge`.

## 8. Training schedule, HPO parity, determinism

1. **Warm-start** (Modules 1–2): `L_recon` only, 20% of budget.
2. **+ Joint harmonization task**: add `L_joint`, 20%.
3. **Full joint**: all losses, 60%; early stopping on validation edge AUPRC
   (VAL-CRITERION), patience 10 evals.
   For family `egostitch_e2e`, the warm-start phase keeps `L_edge` — and therefore
   the trunk, STE, and gated cross-attention — inactive; the §5 branch-dropout
   probabilities are constant after warm-start (Stage-1 form: §13.8).
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
canonical-metric G1 rerun uses `Benchmark-A = breadth_first`, checkpoint
`e092537d8cf1e208`; its single final artifact set is
`outputs/deliverables/g1_graph_metrics_20260714/`. The final G3 artifact set is
`outputs/deliverables/g3_graph_metrics_20260714/`.

A later checkpoint-only `v3_1` evaluation rerun uses the frozen score artifact under
`outputs/runs/legacy_v31_s47_20260712T193900Z/`; its final benchmark-aligned G1 artifact
set is `outputs/deliverables/legacy_g1_graph_metrics_20260714/`. The rerun does not change the
formal E2 training acceptance contract.

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

1. **Structural-target simple-graph policy; official evaluation exceptions**: N(u), degrees,
   budgets d̂_u, training-side clustering/code stats, ego-net targets, recall, and other
   training structural targets strip self-loops. Canonical MMD descriptors and official
   Official GS/RD induced subgraphs retain self-loops exactly as in the benchmark evaluator.
2. **`(u, u)` queries route through a single-ego path**: j := i; T_peer = own kept
   slots; Π = identity on kept slots; s0 = pair_logit(u, u); s1 = self-membership
   `lse_k(κ(h_u^k, proj(x_u)) + log π m)`; s2 from the Â_u diagonal blocks;
   s3 unchanged; s4 on the single-ego scaffold with both anchor labels on u.
3. **Reporting**: edge metrics overall *and* split self / non-self; canonical MMD and
   official GS/RD on loop-retaining induced subgraphs; GS/RD are computed per fixed
   sampled node set and macro-averaged over every sample across node-size buckets. Recall
   remains a simple-graph diagnostic. Report a separate self-loop-rate row (predicted vs
   reference, e.g. 1,701/2,018 on `random_walk`).

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
  disclose `raw_mmd2`, `reference_mmd2`, and `mmd_ratio`; the three MMD ratios remain
  separate metrics and are never combined into Graph Similarity.
- Graph Similarity and Relative Density reproduce the official benchmark evaluator: for each
  fixed node set, compute adjacency edge-Dice GS and NetworkX density-ratio RD on the
  loop-retaining predicted/reference induced subgraphs, then take one unweighted mean over
  all samples in all node-size buckets. Empty/empty returns `1` for both metrics; a nonempty
  prediction over an empty reference returns infinite RD. Run artifacts disclose both
  per-size sample lists and the macro summaries.
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

## 11. E2 production execution design (auto-sized H20, Hugging Face Accelerate DDP)

The formal E2 B0 V3.1 run uses all visible NVIDIA H20 GPUs. The runner validates
that at least one H20 is visible, automatically detects the count `N`, exports those
devices, and launches `accelerate launch --num_processes N`. A cold acceptance run includes first BF16
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
- 2026-07-15: replaced the fixed four-card runtime with automatic visible-H20
  discovery so the same DDP contract can run on any positive H20 count; per-rank
  batches, exact coverage, profiling, and configured budgets are unchanged.
- 2026-07-15: scoring now uses one contiguous shard per visible GPU and strict merge;
  V3.1 may consume the validated packed BF16 feature table to remove repeated raw
  feature I/O without changing pair order, logits, or the scores-artifact contract.
- 2026-07-13: replaced the assembled evaluator's L2-RBF/median-bandwidth raw MMD²
  with the fixed-`σ=1` Gaussian-TV biased MMD² ratio defined in protocol §1;
  removed degree clipping and bound deterministic even/odd reference splitting,
  ratio-of-size-means aggregation, numerator/denominator disclosure, and
  benchmark-aligned self-loop retention for canonical descriptor induced subgraphs;
  also pinned the official spectral-PMF pre-normalization before the common
  `sum + 1e-6` normalization, with degree/clustering left as raw counts beforehand.
- 2026-07-14: replaced the retired MMD-ratio exponential "graph similarity" and full-graph
  simple-edge relative density with official benchmark per-induced-subgraph GS/RD and macro
  aggregation; bound official self-loop and zero-density behavior, kept MMD ratios as
  independent metrics, then recomputed the documented B0/B0-alt/PA-null/legacy/G3 values
  from their frozen score artifacts over all 500 fixed subgraphs.
- 2026-07-14: anonymized the evaluator attribution and artifact directory names; formulas,
  sampling, aggregation, self-loop behavior, and all reported values are unchanged.
- 2026-07-14: renamed this file `06-egostitch-spec.md` → `05-egostitch-spec.md` after
  `05-review-report.md` was retired; all references updated across docs, `src/`, and
  `tests/`. Editorial only — no normative content changed.
- 2026-07-14: added §13 (G5 Stage-1 carve-out) — the frozen sections assume the full
  model; §13 pins the Stage-1 subset the G5 gate builds first. One line per pin:
  - §13.1 module subset: Tokenize-lite + Imagine + Hungarian + Stitch (Module 3a) +
    decision head (s0, s1, s2); Stitch retained because s2 consumes Π.
  - §13.2 codebook-free conditioning: `e_u` replaces `(z_u, r_u)` in `T_cond`, query
    init, and the degree-budget head (no codebook exists in Stage 1).
  - §13.3 Hungarian/Sinkhorn costs drop the code-agreement term; remaining weights
    uniformly rescaled (×7/6, ×5/4); `deg_bucket` pinned to multiplicity log2 buckets.
  - §13.4 hub strata: neighbor-code strata → degree-bucket strata (no codes exist).
  - §13.5 Stage-1 loss tree: drops KL/L_VQ/L_codestats/L_entropy/L_BP/L_joint with
    their mechanisms; L_real interior weights renormalized; Stage-1 L_gate form pinned.
  - §13.6 the four ego-stat targets pinned to evaluator implementations, with
    generated-side estimators (closes the former open item).
  - §13.7 `proj`/τ_κ/attention-head pins (stop-gradient targets; collapse diagnostic).
  - §13.8 curriculum adaptation: §8's harmonization phase drops out (20%/80%);
    fixed-epoch execution with `counterfactual_stop_epoch` per the E2 worker convention.
  - §13.9 self-pair single-ego path Stage-1 form: Π = I, `s2(u,u)` from Â diagonal.
  - §13.10 s0 served from a precomputed frozen-B0 logit cache (checkpoint
    `e092537d8cf1e208`); hard-fail on miss/mismatch; never trained through.
  - §13.11 two-pass density self-calibration scope: pass-1 scores are the Stage-1
    scores; ρ̂_eval logged and consumed by the degree-calibration diagnostic.
  - §13.12 grounding pool pinned: exact top-`n_g` cosine in F0 space, own split side.
  - §13.13 runtime budget: §11's 60-minute pin is E2/B0-specific; the EgoStitch
    worker budget is config-driven under the same auto-sized H20 Accelerate DDP layout.
  - §13.14 Stage-gate comparators: B0 and `B0+cal` (B1/B5 deferred to E3, protocol
    edit of the same date); acceptance criteria pre-registered in
    `docs/registrations/g5_stage1_preregistration.json`.
- 2026-07-15: editorial status update after the first formal Stage-1 execution: code
  exists, Seed 0 completed, and the three-seed gate remains incomplete. No normative
  algorithm, data, loss, or execution contract changed.
- 2026-07-15: added §13.15 per-seed outer orchestration. Each seed now completes
  training, candidate scoring, and an explicitly non-binding topology diagnostic
  before the next seed starts; only the final three-seed evaluation may emit the
  formal Holm pass/cut. Inspecting a single-seed diagnostic preserves the existing
  registration only if the scientific configuration remains unchanged.
- 2026-07-16: added §13.16 (score precision pin). The 2026-07-15 Seed-0 candidate
  artifact (checkpoint `54f3c0ad8f5dfc18`) was scored with BF16 autocast in the
  per-pair pass and is invalidated. BF16 autocast is now encode-only, the pair pass is
  fp32, and EgoStitch artifacts carry an explicit precision contract plus descriptive
  resolution diagnostics.
- 2026-07-16: replaced the Stage-1 registration before any held-out topology metric
  was produced. An unpublished fp32 feasibility rescore still contained intrinsic
  exact-score ties (largest group 143,690 / 2,037,171 rows); under atomic thresholds,
  the nearest `b0_cal_selfdensity` matched-global-RD gap remained 0.0206452, above the
  unchanged 0.005 tolerance. §13.14 now resolves only a boundary tie by deterministic
  canonical pair order to realize the comparator's exact non-self quota. Model,
  hyperparameters, comparators, metrics, seeds, tolerance, and Holm rules are unchanged;
  artifacts bound to the prior registration hash are not formal inputs.
- 2026-07-16: after the exact-quota Seed-0 diagnostic was inspected, re-scoped G5
  Stage 1 from a three-seed inferential gate to a one-seed engineering screening gate.
  This is a post-observation protocol amendment, not a retroactive pre-registration:
  the completed Seed-0 artifact remains bound to its original registration hash and is
  retained only as diagnostic evidence. Future binding Stage-1 runs use a new experiment
  ID and the replacement single-seed decision contract in §13.15. E1/E3 still require
  at least three seeds and Holm-corrected inference.
- 2026-07-16: Stage-1 diagnostic repair revision. The `s1` membership kernel now
  L2-normalizes only its slot and projected-feature operands before squared distance;
  raw `proj(x)` remains unchanged for matching and reconstruction. Added the already
  registered per-family gradient-norm measurement and deterministic Kendall trigger,
  fixed per-epoch channel/rank-mobility fidelity series, probe-time `s1` scale abort,
  a conjunctive scoring-time dead-residual validity gate, and the validation-AUPRC
  near-tie fidelity tie-break. These are bound to a new single-seed experiment ID;
  §10.2 sampling and all Stage-1 success criteria are unchanged.
- 2026-07-16: added §14 (approved successor headline — e2e stitched-topology-
  conditioned pair encoder, proposal §4.4 rev 3.0). §§1–13 remain the binding
  contract for the pending frozen-s0 screening run and its code; §14 records the
  successor architecture and its landing conditions. No §5/§13 semantics change in
  this edit.
- 2026-07-17: result-status closeout only. The replacement fixed-Seed-0 frozen-s0
  screen completed under registration `97e61a7d...` and returned binding verdict
  `cut`: all three primary dominance criteria failed and both guards passed. The
  locked disposition is frozen-s0 scalar fusion → motivating arm + ablation, and
  rev-3.0 e2e conditioning → active G5 build line. This satisfies §14.3(1) but does
  not yet rewrite §5/§13, bind successor defaults, or authorize a formal e2e run.
- 2026-07-17: **§5/§13 rewritten to §14** (source:
  `docs/superpowers/specs/2026-07-16-egostitch-e2e-conditioned-encoder-design.md`
  §§3–6; experiment plan `docs/superpowers/plans/2026-07-17-egostitch-e2e-stage1-screen.md`
  rev 2). §5: rev-3.0 conditioned-encoder head is normative for family
  `egostitch_e2e`; anchored head retired to the frozen-s0 ablation arm. §7: no new
  lambda. §8/§13.8: e2e curriculum note; checkpoint-selection fidelity tie-break
  re-registered within-checkpoint as `std(full − f_logit)/std(f_logit)` (retained,
  not removed). §13.1: e2e Stage-1 mechanism set (STE + gated cross-attention).
  §13.10: retired for the e2e family. §13.16: fp32 scope extended
  (`egostitch_e2e_pair_fp32_v1`, four arrays). §13.17: liveness re-registered
  against the within-checkpoint `f_logit`, thresholds unchanged; gate/grad/delta
  telemetry added. New §13.18: pinned defaults, `permanent_null` overrides,
  stable-hash `shuffle_within_pair` scaffold control, and machine-enforced
  registration-status rules. Satisfies §14.3(2); the formal e2e run remains gated
  on a BINDING registration (§14.3(3–5)).
- 2026-07-17: pinned the §13.18/§14.1 grounded-identity-match flag to a
  three-clause definition (own gate `> 0.5`; pointer-argmax global-node-id
  equality across endpoints; own-split grounding pools only, no Hungarian/
  target-graph access) and superseded the implementation plan's
  Hungarian/`L_gate`-matching phrasing for this label: that quantity is
  train-time-only (node-stream batches with real ego-net targets) and
  undefined for edge-stream endpoints and unseen nodes at inference, so
  wiring it there would violate the inductive protocol.

**Closed gate-report deliverable (2026-07-17):** the frozen-s0 Stage-1 gate report,
fidelity diagnostics, and measured FLOPs/latency report are complete. The binding
verdict is `cut`; see `docs/results/G5-stage1-seed0-20260717.md`. Closed 2026-07-14: the four
ego-stat target definitions (§13.6); Benchmark-A/B/C ↔ split-strategy mapping
(confirmed at G1: `Benchmark-A = breadth_first`, §9.1).

## 13. G5 Stage-1 carve-out (normative for the Stage-1 build)

Gate G5 (protocol §5.0.5) builds the model in three mechanism stages. The frozen
sections above pin the *full* model; this section pins the **Stage-1 subset** —
"imagination + degree budget + closure channel only (no codebook, no harmonization,
no CVAE)" (proposal §6.0-G5). Stage 2 (+ codebook + s3) and Stage 3 (+ harmonization
+ seam loss) restore the corresponding items verbatim from §§1–8; nothing here
changes the full model.

### 13.1 Module subset

**Active family `egostitch_e2e` (normative since 2026-07-17):** Stage 1 =
Tokenize-lite (§1 minus VQ, BP affiliations, code-stats head) + Imagine (§2, no
CVAE) + Hungarian matching (§13.3) + Stitch (§3, Sinkhorn alignment only) + the §5
rev-3.0 head: from-scratch trunk + STE + zero-init gated cross-attention +
`c_content` pathway + three head nulls. The STE is Stage-1-runnable because Stitch
is retained in Stage 1. Codebook/`s3` remain Stage 2; Harmonize/seam loss remain
Stage 3; those stages then feed the STE (Stage 2: `s3`-derived structural inputs;
Stage 3: the harmonized scaffold).

**Retired frozen-s0 family `egostitch`** (historical contract for the completed
screen and the E4 ablation arm): Stage 1 = Tokenize-lite (§1 minus VQ, BP affiliations, code-stats head) + Imagine
(§2, no CVAE) + Hungarian matching (§13.3) + Stitch (§3, Sinkhorn alignment only) +
decision head over `(s0, s1, s2)`:

```text
p_ij = σ( s0 + g_θ(s1, s2) · w ),   g_θ = MLP_2(gate), w learned scalar init 0.1
```

Dropped until later stages: codebook (`z_u`, `r_u`, `stats_u`), `F_u`, CVAE (`ζ_u`),
Harmonize (§4; `R = 0`), `s3`, `s4`. **Stitch is retained in Stage 1** because the
closure channel `s2` consumes the alignment plan `Π` (§5). The §5 s-channel
correlation diagnostic reports the available channels `(s0, s1, s2)`.

### 13.2 Codebook-free conditioning (`e_u` substitution)

With no codebook, `e_u` replaces `(z_u, r_u)` everywhere they condition:

```text
T_cond = [W_x x_u; W_e e_u]                        # 2 tokens (W_z, W_r, W_ζ dropped)
Q_k    = W_q [proj(x_{g_k}); e_u]    for k ≤ min(K, n_g^+)
Q_k    = q_k^base + W_q' e_u         otherwise
d̂_u   = softplus(MLP_2(d + d_z → 1)([x_u; e_u])) · ρ_eval/ρ_train
```

Conditioning dropout (`∅_content` / `∅_all`, p = 0.1 each) is retained unchanged.

### 13.3 Matching and OT costs without the code term

The code-agreement terms are dropped; the remaining weights are uniformly rescaled so
each total cost keeps its §2/§3 scale (Hungarian ×7/6, Sinkhorn ×5/4). Both
`linear_sum_assignment` and Sinkhorn are invariant to a uniform positive rescale of
the cost matrix — the rescale is recorded for exactness only.

```text
C_{k,v}   = (7/6)·‖h_u^k − proj(x_v)‖₂² + (7/24)·|deg_bucket(k) − deg_bucket(v)|
          + (7/24)·overlap_penalty(k, v)
C^Π_{kk'} = (5/4)·‖h_i^k − h_j^{k'}‖₂² + (5/16)·|π_i^k − π_j^{k'}|
```

`deg_bucket` (left symbolic in §2) is pinned: log2 multiplicity buckets
`{[1,2), [2,4), [4,8), [8,16), [16,32]}` → integer index; `deg_bucket(k)` buckets the
slot's predicted multiplicity `m_u^k`, `deg_bucket(v)` buckets the target's
multiplicity label (1 for non-hub targets; stratum ratio under §13.4). Two-pass
overlap penalty, per-step recomputation, constant-in-backward assignment, and
denoising queries (25% of nodes, σ_noise = 0.1, fixed assignments) are retained.

### 13.4 Hub strata without codes

§2's neighbor-code strata become **degree-bucket strata**: neighbors of a hub node
are stratified by `deg_G_struct` log2 bucket; proportional allocation capped at K,
per-target multiplicity label = stratum_size / allocated_count, degree NLL keeps the
true |N(u)| — all unchanged.

### 13.5 Stage-1 loss tree

```text
L = L_edge + 0.5·L_real + 0.1·L_ssl + 1.0·L_recon          # §7 λ defaults unchanged
L_recon = 1.0·L_feat(Hungarian, Huber) + 0.5·L_exist(BCE, ∅-balanced)
        + 0.25·L_mult(NLL) + 0.5·L_deg(lognormal NLL)
        + 0.5·L_slotadj(group-level BCE) + 0.25·L_gate
        # dropped with their mechanisms: KL, L_VQ, L_codestats, L_entropy, L_BP, L_joint
L_real  = (2/3)·ED(ego-stat vectors) + (1/3)·ED(random-GIN embeddings)
        # seam-overlap ED deferred to Stage 3 with the seam loss; adversarial off;
        # interior weights renormalized to sum 1
L_ssl   unchanged (§7): 0.5·consistency(feature noise σ=0.05)
        + 0.5·pool-resample consistency, ungrounded slots only
```

**Stage-1 `L_gate` form:** §7's partner-vs-peer BCE presupposes harmonization. Until
Stage 3, `L_gate = BCE(g_u^k, 1[Hungarian-matched target of slot k ∈ G(u)])` — the
gate learns whether a slot represents a retrievable (grounding-pool) neighbor.

### 13.6 Ego-stat targets (pinned; closes the §12 open item)

Real side, per node `u` on `G_struct` (evaluator = NetworkX implementations):

```text
t = [ deg(u);  nx.clustering(G_struct, u);  |E(ego(u))|;  nx.density(ego(u)) ]
      # ego(u) = G_struct.subgraph(N(u) ∪ {u}), simple graph
```

Generated side (soft estimates from the Imagine heads):

```text
d̃    = Σ_k π_k m_k                       Ẽ_nn = Σ_{k<k'} Â_{kk'} π_k π_k' m_k m_k'
t̃    = [ d̃;   Ẽ_nn / max(C(d̃, 2), 1);   d̃ + Ẽ_nn;   (d̃ + Ẽ_nn) / max(C(d̃+1, 2), 1) ]
```

`ED` = energy distance between the batch distributions of `t̃` (generated) and `t`
(real), each coordinate standardized by the real-side batch mean/std. The random-GIN
term: one frozen randomly-initialized 3-layer GIN (hidden 64, sum-pool, weights fixed
at seed 0) applied to the generated soft ego-net (star edges weighted `π_k m_k`,
slot–slot edges weighted `Â π π`; node features `[π; m; g]`, anchor one-hot on `u`)
and to the real `G_struct` ego-net (binary edges, matching binary features); ED
between the two embedding batches.

### 13.7 Shared projection and decision-head pins

`proj` = one learned linear `d → d_p`, shared by the matching cost, `L_feat`, and
`s1`. Matching-cost and `L_feat` targets use `stop-gradient(proj(x_v))` — gradients
reach `proj` only through the slot side and `s1`. Diagnostic: per-epoch variance of
`proj(x)` over the node batch (representation-collapse watch). `τ_κ` in `s1`: learned
scalar, softplus-parameterized, init 1.0. Inside the `s1` membership kernel only,
both `h^k` and `proj(x_other)` are statelessly L2-normalized before squared distance.
No batch/running statistics are permitted, and raw `proj(x)` semantics remain intact
for Stitch/matching, node losses, and reconstruction targets. Decoder attention heads: 8.

### 13.8 Curriculum adaptation

§8's middle phase (+ joint harmonization) drops out of Stage 1:

1. **Warm-start** (Tokenize-lite + Imagine): `L_recon` only, node stream only, 20% of
   the epoch budget.
2. **Full Stage-1 joint**: all §13.5 losses, node + edge streams, 80%.

Execution is fixed-epoch with per-epoch validation; the VAL-CRITERION early stop is
recorded as `counterfactual_stop_epoch` (the E2 worker convention) and `best.pt` is
the early-stop-equivalent checkpoint. Validation edge AUPRC remains primary; when two
epochs differ by at most `1e-4`, the fidelity tie-break selects deterministically.
**Family `egostitch_e2e` (re-registered 2026-07-17):** with `s0` retired, the
tie-break statistic is the **within-checkpoint** `std(full − f_logit)/std(f_logit)`
on the fixed validation slice — the larger value wins (liveness-preferring, same
direction as the retired residual/s0 rule, which remains the recorded rule for the
historical frozen-s0 family). This fidelity tie-break is checkpoint selection only
and is not a Stage-1 success criterion.

### 13.9 Self-pair single-ego path (Stage-1 form)

§9.4 rule 2 with the Stage-1 channels: `Π = I` on all slots (no keep-mask exists at
`R = 0`); `s1` self-membership unchanged; `s2(u, u) = Σ_k (π_u^k)² · Â_u[k, k]`
(the "Â diagonal blocks" reading substituted into the §5 `s2` formula).

### 13.10 s0 cache (frozen B0 logits) — RETIRED for family `egostitch_e2e`

**2026-07-17:** the active e2e family has no `s0` channel: configs carry no
`data.s0_cache` and no `data.s0_checkpoint_id`, and the worker skips cache loading
for this family. The remainder of this section is the historical contract for the
frozen-s0 family artifacts and the E4 ablation arm.

`s0` comes from the audited frozen V3.1 checkpoint `e092537d8cf1e208` via
precomputed logit caches: training pairs are enumerable offline because the §10.2
negative sampler is deterministic in `(seed, epoch, rank)`; candidate/val/test pairs
come from `score_universe` artifacts. The cache loader **fails loudly** on any
missing pair or checkpoint-id mismatch. `s0` is never trained through.
Scoring may use contiguous multi-GPU shards and the validated packed BF16 feature
table; strict merge restores the original manifest row order and metadata contract.

### 13.11 Two-pass density self-calibration (Stage-1 scope)

Pass 1 (ratio 1) runs and logs `ρ̂_eval` per §9.3. The pass-2 rescale only affects
`d̂`, which sits outside the Stage-1 score path (`s3` is absent), so **Stage-1
reported scores are pass-1 scores**; `ρ̂_eval·d̂` still feeds the degree-calibration
diagnostic. The two-pass plumbing is built now and becomes score-active in Stage 2.

### 13.12 Grounding pool

`G(u)` = exact top-`n_g` cosine neighbors of `x_u` in F0 space **within u's own split
side**, self excluded (≈10k nodes — exact matmul, no ANN), computed once and
disk-cached with a sha256 manifest.

### 13.13 Runtime budget

§11's 60-minute cold-run pin applies to the formal E2 B0 V3.1 run only. The
EgoStitch worker uses the same automatically detected H20 count and
`accelerate launch --num_processes N` layout, the same runtime-profile and checkpoint payload schemas, and a
**config-driven** budget (`runtime.total_budget_seconds`; stage budgets must sum).
This family's "feature pack" stage = the F0 pooled matrix (§9.2) + grounding-pool
cache. The orchestrator's probe/projection gating applies against the configured
budget.

### 13.14 Stage-gate comparators and pre-registration

Stage-1 gate comparators: **B0** (frozen candidate-scores artifact) and **`B0+cal`**
(protocol §2). B1/B5 comparisons are deferred to E3 with their implementations
(protocol edit 2026-07-14). Acceptance criteria are pre-registered in
`docs/registrations/g5_stage1_preregistration.json`; the training worker records the
file's sha256 in `run_metadata.json` at run start, and the gate evaluator refuses to
open held-out metrics if the hashes disagree.

For each matched-global-simple-edge-RD comparator row, EgoStitch non-self candidate
pairs are ordered by descending pass-1 score. Only rows tied at the quota boundary
are ordered by ascending canonical pair order `(min node index, max node index)`;
labels and topology targets are never consulted. Exactly the comparator's realized
non-self edge count is assembled. Self-pairs remain outside that quota and are
assembled when their score is at least the selected boundary score. The unchanged
registered check `|RD_global(ego) - RD_global(comparator)| <= 0.005` must still hold
and the realized quota, boundary score, boundary-tie size, and tie-split count are
recorded.

### 13.15 Single-seed Stage-1 screening contract

G5 Stage 1 is an **engineering screening gate**, not the paper's final inferential
comparison. One fixed training seed is sufficient. The binding outer order is
`train(seed 0) -> candidate scoring(seed 0) -> fidelity/cost diagnostics -> Stage-1
gate(seed 0)`. Additional seeds are optional robustness analyses and are not required
to advance to Stage 2. E1/E3 retain the protocol's at-least-three-seed and Holm-corrected
requirements for paper-level claims.

The Stage-1 primary rule is deterministic point-estimate dominance at the registered
operating points: Seed 0 must be strictly better than every registered comparator on
each of clustering-MMD ratio (lower), BFS-macro GS (higher), and BFS-macro RD (higher),
and both non-regression guards must pass. All three primary criteria are required.
With one training seed, between-seed variance, confidence intervals, p-values, and Holm
decisions are not valid acceptance evidence and must be emitted as `null`/not applicable.
The report must label the verdict as a single-seed screening decision and must not claim
statistical significance, robustness across seeds, or an E1/E3 result.

This contract was adopted after inspecting the earlier exact-quota Seed-0 diagnostic.
That artifact cannot be rebound retroactively: its old registration hash remains part of
its provenance, and any binding run under this contract requires a new experiment ID and
run metadata pinned to the replacement registration before training starts.

### 13.16 Score precision (pair-pass fp32 pin)

BF16 autocast is permitted **only** for the cacheable per-node encode pass
(Tokenize-lite + Imagine, §13.1–§13.2); its outputs are cached as fp32 (`.float()`)
before reuse, so no BF16-grid truncation propagates downstream. The **per-pair**
score pass — Stitch (§3, §13.3), the `(s0, s1, s2)` decision head (§13.1), and the s0
fusion (§13.10), including the §13.9 self-pair single-ego path — computes with
autocast **disabled**, in fp32: published artifact logits must carry full fp32
resolution. Every EgoStitch scores artifact records the contract
`egostitch_pair_fp32_v1`, pair compute dtype `float32`, and pair autocast disabled;
writers, shard merging, orchestration reuse, and gate evaluation fail closed if this
provenance is absent or inconsistent. Unique-logit count/fraction and reduced-precision
round-trip fractions are recorded as descriptive diagnostics only: legitimate model
ties must not be rejected solely because the unique-logit fraction is low.

**Family `egostitch_e2e` (extension, 2026-07-17):** the per-pair fp32 scope covers
Stitch, the STE, both gated cross-attention pathways, the trunk pair-cross-attention
blocks, and the head — i.e. everything from cached per-node encodes to all **four**
published logits (full, `f_logit`, pair+content, pair+topology). The per-node encode
pass may stay BF16 with fp32-cached outputs, unchanged. Artifacts for this family
record the contract string `egostitch_e2e_pair_fp32_v1`; the resolution guard
applies to each of the four arrays independently. `egostitch_pair_fp32_v1` remains
the recorded contract for historical frozen-s0 artifacts.

### 13.17 Registered training/fidelity instrumentation

The worker instantiates all four Kendall log-variance parameters before DDP/optimizer
construction. After warm-start, every 50 steps it replays one fixed probe batch under
`no_sync`, performs separate retained-graph gradient measurements for the weighted
`L_edge`, `L_recon`, `L_real`, and `L_ssl` families, and records their global L2 norms
in the epoch's `metrics.jsonl` row. If the largest family norm exceeds 10× the family
median continuously for 1,000 optimization steps, Kendall uncertainty weighting is
activated on every rank; activation step and learned log variances are recorded.

The first runtime probe records `s1/s2/s2_aa/residual` mean and standard deviation and
aborts when `|mean(s1)| > 1000`. Every validation epoch records the three channel scales,
residual standard deviation, residual/s0 standard-deviation ratio, Kendall tau/rank
mobility versus s0, and top-k overlap on a fixed 1% validation slice. These series and
the Kendall state are embedded in `run_metadata.json` and are required by the G5 gate.

**Family `egostitch_e2e` (re-registered 2026-07-17):** liveness references the
**within-checkpoint `f_logit`** — no fresh frozen-s0 comparator artifact and no
alignment step exist for this family. The residual is `full − f_logit`; the ratio
denominator is `std(f_logit)`. Telemetry rows additionally record `gate_topo_tanh`
and `gate_cont_tanh` (per injected block), `grad_rms_trunk` / `grad_rms_ste` /
`grad_rms_content`, and a per-epoch `topology_delta_std` on the fixed validation
slice. The channel-scale series of this section read the available quantities
(`f_logit`, pathway deltas) in place of `s1/s2/residual-vs-s0`.

For the historical frozen-s0 family: before topology evaluation, the gate aligns
the EgoStitch candidate artifact with a
fresh fp32 frozen-s0 candidate artifact scored from checkpoint `e092537d8cf1e208`.
This input is separate from the historical canonical B0 comparator artifact, whose
quantization must not masquerade as residual variance. The gate fails the run as a dead
residual only when all
three registered death signals hold: residual/s0 standard-deviation ratio `< 1e-5`,
Spearman correlation with s0 `> 0.9999`, and top-1% overlap `> 0.9999`. This conjunctive
validity rule prevents a genuinely pair-varying but safely small residual from being
turned into a post-hoc outcome switch. It is a run-validity gate, not a success metric.


### 13.18 E2E family pins (defaults, overrides, controls, enforcement)

Registered defaults for family `egostitch_e2e` (fixed for the Stage-1 screen; the
named sweeps are reserved for E1/E3):

- `ste_layers = 3`, `ste_dim = 128`, `xattn_heads = 8`;
- `n_inj = 1` (sweep `{1, 2}` reserved);
- `p_topo = p_cont = 0.15` (sweep `0.1–0.2` reserved; `p = 0` is a Stage-1 arm).

**Permanent-null training override:** `model.config.permanent_null ∈ {none,
all_head, content_head}`. A non-`none` value applies the corresponding §5 hard
mask to **every** training and evaluation batch: `all_head` trains the matched
pairwise-only `B0-e2e` arm; `content_head` trains the pair+topology attribution
arm. The override must be recorded in `run_metadata.json`.

**Scaffold structure control (`shuffle_within_pair`, scoring-time):** for every
scored pair, two slot-index permutations — one for the source side, one for the
destination side — are drawn from generators seeded by a stable hash
`blake2b("{min(u,v)}|{max(u,v)}|{side}|{seed}")` with `side ∈ {src, dst}` assigned
by canonical order (src = min-id endpoint). The permutations are applied to the
slot rows/columns of `Â_src`, `Â_dst`, and the matching axes of `Π` before the
STE; slot token features are untouched. Keying on the canonical unordered pair
makes the control **invariant to batching, scoring order, GPU count, and shard
boundaries**, and identical across the AB/BA passes (preserving
`p(i,j) = p(j,i)`). Artifacts record
`scaffold_control=shuffle_within_pair,seed=<seed>,keying=canonical_pair_v1`.

**Registration-status enforcement (machine-checked):** a formal training run (no
`--max-steps`) requires the referenced registration file to carry
`status: BINDING`; the worker fails closed otherwise. `--max-steps` debug runs
accept `DRAFT` but are redirected to `*_debug` output directories and never write
held-out artifacts. The G5 gate requires `status: BINDING` and a registration
sha256 that matches the `preregistration_sha256` of **every** consumed formal run
metadata. Amending a BINDING registration requires a new versioned registration
file (predecessor convention).

**Grounded-identity-match flag (e2e content tokens, pinned 2026-07-17):** for a
pair `(i, j)`, slot `k` of endpoint `u ∈ {i, j}` is grounded-identity-matched
**iff** (1) `gate_u^k > 0.5` (the slot is grounded — the same 0.5 threshold §4's
grounded-slot re-mask rule uses); AND (2) `argmax` of slot `k`'s grounding
pointer selects a grounding-pool candidate with global node id `c`; AND (3) the
**other** endpoint has at least one slot `k'` with `gate > 0.5` whose pointer
argmax selects the **same** global node id `c`. The flag is a binary float
`{0.0, 1.0}` per slot, shape `(B, K)` per side — the `matched_src`/`matched_dst`
inputs to `build_content_tokens` (§14.1 `c_cont`). It is symmetric across AB/BA
by construction (the shared-candidate relation is undirected), computable at
inference for any node (grounding pools are drawn from the own-split-side F0
space, §13.12 — no target-graph access, no labels), and is classic
shared-neighbor **content** evidence (§14.1). Superseded: the implementation
plan's phrasing wiring this label to "the pointer argmax landing on a
grounding candidate that Hungarian-matched the same target — reuse the
matching output already computed for `L_gate`" is an authoring error.
Hungarian assignments exist only for node-stream training batches with real
ego-net targets; they are undefined for edge-stream endpoints and for unseen
nodes at inference, so wiring them at inference would violate the inductive
protocol (CLAUDE.md Integrity gates).

## 14. Approved successor headline: E2E stitched-topology-conditioned pair encoder (2026-07-16)

**Scope and precedence.** This section records the approved rev-3.0 headline
architecture (proposal §4.4; full decision trail and pins in
`docs/superpowers/specs/2026-07-16-egostitch-e2e-conditioned-encoder-design.md`).
**§§1–13 remain the historical binding contract** for the completed frozen-s0
Stage-1 screen and its retained implementation on `main`. That screen published a
binding `cut` verdict on 2026-07-17, satisfying §14.3(1). **§5/§13 were rewritten to §14 on
2026-07-17** (change-log entry above): §14 plus the rewritten §5/§13 are now the
normative implementation contract for family `egostitch_e2e`. A formal e2e run
additionally requires a fresh Stage-1 registration with `status: BINDING`
(§13.18 enforcement); the rewrite alone authorizes implementation, not execution.

### 14.1 Architecture summary

```text
z_pair = Trunk(tok_i, tok_j)      # V3.1-class pair encoder, trained from scratch:
                                  # raw token sequences → Siamese encoder → pair
                                  # cross-attention → pair_context_gated / abba_max
c_topo = STE(T̂_ij)                # structure-only stitched-topology encoder over
                                  # {star edges, Â_i, Â_j, Π}; node inputs = 4-type
                                  # anchor labels, π, m, soft degrees (NO h, NO g,
                                  # NO grounded-identity-match); 2–3 edge-weighted MP
                                  # layers (promoted s4 lineage); token-level output
c_cont = ContentTokens(S_i, S_j)  # separate pathway: [h; π; g; grounded-identity-
                                  # match] per slot + membership signal (former s1)
inject: in the last N_inj ∈ {1, 2} pair-cross-attention blocks (default 1), the CLS
        token cross-attends to c_topo and (separately) c_cont through zero-initialized
        tanh gates, per direction (AB / BA, swapped anchor labels) BEFORE abba_max;
        AB/BA share STE and attention parameters; branch masks are per pair, shared
        across directions
p_ij  = σ(head(z'_pair))
```

### 14.2 Null taxonomy (three mutually exclusive head nulls; checkpoint-exact)

| Null | Skips | Yields |
|---|---|---|
| `∅_all_head` | STE + topology XAttn + content XAttn | pair-only `f_logit` |
| `∅_topo_head` | STE + topology XAttn | pair + content |
| `∅_content_head` | content XAttn | pair + topology |

Training realizes nulls as per-pair multiplicative masks (probabilities
`p_topo = p_cont = 0.15` default, sweep 0.1–0.2, plus `p = 0` arms); evaluation uses
batch-level hard bypasses; residual sublayer form makes the two numerically identical
(required unit test), as is `p(i,j) = p(j,i)` under every null. The `_head` namespace
is disjoint from the §2 conditioning-dropout `∅_content` / `∅_all` (decoder nulls).
All four logits (full + three nulls) are published per scored pair; the §13.16 fp32
pair-pass pin extends to trunk, STE, gates, and head
(`egostitch_e2e_pair_fp32_v1`); §13.17 liveness signals re-register against the
within-checkpoint `f_logit` (no frozen-s0 comparator artifact); the §13.10 s0 logit
cache is retired for this family.

### 14.3 Landing conditions (all required before a binding e2e run)

1. Frozen-s0 screen published (its outcome is the successor's motivating arm).
   **Satisfied 2026-07-17:** binding verdict `cut`; result note
   `docs/results/G5-stage1-seed0-20260717.md`.
2. §5/§13 rewritten to §14 with change-log lines; §13.18-style pins for defaults
   (`ste_layers = 3`, `ste_dim = 128`, `xattn_heads = 8`, `n_inj` default 1 sweep
   {1, 2}).
   **Satisfied 2026-07-17:** §5/§13 rewritten (change-log entry); §13.18 landed
   with the defaults, `permanent_null` overrides, `shuffle_within_pair` control,
   and registration-status enforcement.
3. Fresh registration with the five-arm Stage-1 scope (full, `B0-e2e`/f-only,
   pair+topology, within-pair `Â`/`Π` shuffle, `p = 0`), the four-logit decomposition
   report, the representation-probe protocol (degree / ego density / clustering +
   degree-partialled + Π-consistency, frozen-encoder linear probes on held-out
   message-partition nodes), the pathway-attribution decision rule, and a measured
   H20 cost re-estimate (the 673 s / 2.04 GiB frozen-s0 Stage-1 profile does not
   extrapolate; budget class is the E2 B0 run).
4. Implementation plan:
   `docs/superpowers/plans/2026-07-16-egostitch-e2e-conditioned-encoder.md`
   (B0.py untouched; conditioned trunk subclass; train-mask ≡ eval-bypass and
   symmetry tests as acceptance criteria).

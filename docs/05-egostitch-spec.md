# Open Algorithm Specification — Method Pending Selection

**Status:** **OPEN — no project-wide algorithm selected; the text below is the historical
retrieval-grounded EgoStitch reproduction contract.** It satisfied gate
§6.0-G4: Stitch/Harmonize pseudocode with tensor shapes, OT cost and ε, confidence and
quantile schedule, budget tolerance, gradient estimators, and the full loss tree with
interior weights. Neutral placeholders per repository convention; no dataset names.
Nothing below promotes this arm to the selected method; it pins historical free
parameters. §§9–11 (added at sign-off) bind the spec to
the local benchmark package in `data/`, define the batch-sampler / data contract, and
fix the GPU-count-independent H20 execution design.

**Scope rule.** The frozen contract applies only to the historical EgoStitch arm;
the selected method receives a new concise specification here after comparison.

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
- All structural targets computed on the **full shared training topology** (§6).

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

**Joint training task:** sample from the `V_fit` pair universe — 50% from
`E_topo`, 50% random train-node pairs (label-agnostic, R8) — mask each ego-net
at ratio ~ U(0.3, 1.0), run 1–2
harmonization rounds, supervise re-decoded slots by the §2 matching losses.
**Gradient estimators:** keep/re-mask = straight-through on the binary keep mask;
budget trigger = detached comparison + soft mirror penalty
`0.1·ReLU(Σπm − B_u)²` so the constraint surface reaches gradients; Sinkhorn unrolled;
critic trained with its own BCE, gradients not propagated into h (detached input).

## 5. Module 4 — Decision head (per pair)

**Historical e2e implementation (family `egostitch_e2e`, updated 2026-08-02):** the
decision head is a from-scratch V3.1-class pair encoder conditioned on the stitched
topology — no frozen B0 anchor, no `s0`:

```text
p_ij = σ( head( Trunk(tok_i, tok_j | STE(T̂_ij)) ) )
```

- **Trunk:** Siamese token encoder + pair cross-attention over the raw token
  sequences `(tok_i, tok_j)` — the audited V3.1 architecture family
  (`pair_context_gated` blocks maintaining `(h_a, h_b, cls_token)`, feature-wise
  `abba_max` over the AB and BA passes) — trained under the historical EgoStitch regime.
- **STE (stitched-topology encoder):** structure-only tokens over the stitched
  scaffold `T̂_ij`. Token features: 4-type anchor labels (endpoint-i / endpoint-j /
  slot-of-i / slot-of-j), `π`, `m`, soft degrees — **no** slot content `h`, no
  grounding embeddings `g`, no grounded-identity-match flag. Edge weights: star
  edges `π·m`, intra-side `Â_i`/`Â_j` weighted by `π` outer products, and the
  alignment plan `Π`. `ste_layers` edge-weighted message-passing layers
  (defaults §14.3); **token-level output** — one conditioning token per slot and
  per endpoint (the promoted `s4` lineage: the pooled scalar summary is replaced
  by tokens).
- **Conditioning:** zero-initialized tanh-gated cross-attention
  (`cls ← cls + active · tanh(gate) · XAttn(LN(cls), tokens)`); the **cls_token is
  the only query**, injected after the final `n_inj ∈ {1, 2}` pair-cross-attention
  blocks (default 1). The AB and BA directions share STE and cross-attention
  parameters before `abba_max`.
- **No separate content pathway:** slot semantics supervise the generator and help
  construct the typed scaffold, but complete slot-content tokens never directly
  condition the classifier.
- **Topology null:** training uses the per-pair topology mask `p_topo`; evaluation's
  `∅_all_head` hard bypass yields the pair-only `f_logit`. Symmetry and mask/bypass
  equivalence remain required tests.
- **Two logits published per scored pair:** full and `f_logit` (§13.16 fp32 pin
  applies to both).

Channel disposition: `s4` is promoted into the STE; the former direct `s1` content
path is retired; `s0` is retired (§13.10); `s2` remains a training-side diagnostic and
probe target; `s3` remains a Stage-2 STE input.

**Retired anchored head** (frozen-s0 family `egostitch`; motivating result and E4
ablation arm only — binding `cut` verdict 2026-07-17,
`docs/results/G5-stage1-seed0-20260717.md`):

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

Required current diagnostic: the two-logit decomposition (`full`, `f_logit`) plus
`full − f_logit` summary statistics. The retired frozen-s0 arm keeps its historical
channel correlation matrix on `(s0..s4)` only for interpreting that old result.

## 6. Data partitions and leakage rules

- **The same complete set of train-side positive edges is used for topology
  learning and edge classification.** Structural targets (reconstruction, degree NLL,
  BP-NLL, code stats, seam references, critic training) and `L_edge` positives are both
  derived from `E_train+`; there is no message/supervision partition. The loopless
  topology projection drops self-pairs, while classification retains them (§9.4).
  Leave-one-out remains mandatory: when a positive pair (u,v) is in the batch, v is
  excluded from u's reconstruction targets and |N(u)| decremented.
- Seam references: unions of training-edge ego-net pairs sampled 50/50
  adjacent/random with labels marginalized.
- B0 provenance audit is an E5 gate precondition. The E2 B0 scorer is
  pinned to the audited V3.1 `pair_context_gated` / `abba_max` / no-cross checkpoint
  family (`d_model = 512`, no spectral normalization) and trains from the fixed
  balanced `train_edges.txt` rows when a local retrain is required.
- **Benchmark binding:** how the shipped artifacts map onto shared training edges/val,
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
L_ssl   = 0.5·consistency(feature noise σ=0.05, in standardized coordinates for the
          E2E family — §13.19.1) + 0.5·pool-resample consistency
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
   **Historical v1 `egostitch_e2e` only:** the warm-start kept `L_edge` — and
   therefore the trunk, STE, and gated cross-attention — inactive (Stage-1 form:
   §13.8). The prospective stability-screen v2 is governed by §13.19 instead:
   `L_edge(f_logit)` trains the pair trunk/head from step 0 while topology/content
   conditioning remains hard-bypassed until its registered ramp.
- **HPO parity:** every historical ladder arm (B0…B5, `B3-full`, EgoStitch) gets the same tuning
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

Identities verified: `train_graph.pkl` edges = benchmark train⁺ ∪ val⁺ exactly
(val⁺ is the benchmark's model-selection complement, never a training target);
train/val/test negatives ∩ global positives = ∅. Within `V_fit`, topology and
classification both consume the same complete `train_edges.txt` positive set;
`val_edges.txt` positives are used only to prevent false-negative sampling.

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

### 9.3 Training-edge and validation binding

The shipped train rows bind one shared positive-edge set:

```text
E_train⁺ := positives of train_edges.txt            (per strategy)
E_topo   := E_train⁺ \ self-loops                    # loopless topology projection
G_struct := simple graph (V_train, E_topo)           # ALL structural targets
L_edge positives := E_train⁺                         # same edges; self-pairs retained
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

**Prospective v2 internal topology holdout (§13.19 only).** After deriving simple
`G_train=(V_train,E_topo)`, construct two 256-node connected holdouts before
training. `V_qual` is the first 256 nodes of deterministic BFS in `G_train`'s largest
component, seeded by the node minimizing `sha256("g5-v2-qual|"+node_id)` and ordering
each frontier by the same hash. Remove `V_qual`; `V_select` is constructed identically
on the largest remaining component using prefix `g5-v2-select|`. Ties between
components use the sorted node-id tuple. `V_fit` is every remaining operative train
node. `V_qual` and `V_select` are node-disjoint by construction (asserted all-zero
`OverlapProof`) and both are already subtracted from `V_fit`; **`V_hold := V_qual ∪
V_select`** is the single 512-node validation holdout used by the formal plan.
Because `V_hold` is exactly this union,
defining it changes nothing about `V_fit`: **`V_fit`, `E_train⁺[V_fit]`,
`G_fit`, and `rho_train` are unchanged relative to the former two-holdout
definition under this same shared-edge contract**. The 2026-08-03 removal of
the 80/20 split itself changes the holdout graph and therefore invalidates all older
`V_fit`-side caches, packs, feature-statistics digests, and result comparability. (The
`V_qual`/`V_select` grounding universes are themselves replaced by `V_hold` at
§13.12 — a deliberate role-universe change, not an invalidation of `V_fit`-side
state.) The v2 training contract replaces `G_struct` with the induced
`G_fit=(V_fit, E_train⁺[V_fit] \ self-loops)` and restricts classification positives
to the same edges with both endpoints in `V_fit`; all `V_fit`↔`V_hold`
cross-partition edges — i.e. positives with exactly one endpoint in `V_hold` —
are quarantined from training. The within-holdout cross-side edges (between `V_qual` and
`V_select`) are **not** quarantined; they are `V_hold`'s evaluation-only topology
labels, exactly like the within-`V_qual` and within-`V_select` edges. `E_topo[V_hold]`
— which includes edges within each of `V_qual`/`V_select` and the cross-side edges
between them — is `V_hold`'s evaluation-only topology label, used for formal
checkpoint selection. `V_hold`'s nodes, edges, and full
non-self pair universe never enter a training step, reconstruction/realism target,
training-time scaffold, or negative sampling; `V_hold` is opened only by its isolated
eval-mode selection scorer. `train_graph.pkl`,
`val_edges.txt`, and their positives remain excluded from topology gold. Exact
node/edge/pair counts and hashes, including proofs of zero node and label-edge
overlap, are required before v2 binding. This internal holdout does not alter the
frozen external test protocol.

Density normalization (`ρ_eval/ρ_train`, §1): ρ := |E⁺| / (C(|V_side|,2) + |V_side|)
on the matching universe. The former random-walk `ρ_train = 1.354e-3` and **~11×**
shift were measured under the retired 80/20 contract and must not be reused. Under
the corrected contract, `ρ_train` is recomputed and recorded from `E_topo[V_fit]` for
the active strategy and holdout. Because true test density is **not observable** under
the strict gate, the inference-time ratio is pinned as:

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
2. **`(u, u)` queries route through a single-ego path**: j := i; encode and imagine
   `u` exactly once, reuse that one ego state for both directional views, set
   `T_peer` to its own slots, and set `Π = I_K` exactly (no Sinkhorn call). The
   E2E raw-token trunk likewise encodes `u` once. Historical channels retain
   `s0 = pair_logit(u, u)`, self-membership
   `lse_k(κ(h_u^k, proj(x_u)) + log π m)`, `s2` from the Â_u diagonal blocks,
   unchanged `s3`, and the single-ego scaffold with both anchor labels on `u`.
3. **Reporting**: self-loops are **retained by default in every labeled pair view**
   (val and test alike) — no view strips them from its headline edge metrics. The
   **test** view alone additionally reports the self / non-self split, and reports the
   self-loop-*including* block **first**, with the split supplementary beneath it; val
   reports the self-loop-including block only. Canonical MMD and
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
| joint-pair stream | 50% E_topo edges / 50% random train pairs (§4) | B_p = 128 pairs | L_joint |
| edge stream | E_train⁺ positives + resampled negatives | frozen-s0: B_e = 512 pairs/rank; packed-token E2E: B_e = 128 pairs/rank (both 1:5) | L_edge |

Curriculum (§8) toggles streams: stage 1 node-only; stage 2 node + joint; stage 3
all. An **epoch** = one full pass over `E_train⁺[V_fit]` in the edge stream; node and joint
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
  edge.

## 11. E2 production execution design (auto-sized H20, Hugging Face Accelerate DDP)

The formal E2 B0 V3.1 run uses all visible NVIDIA H20 GPUs. The runner validates
that at least one H20 is visible, automatically detects the count `N`, exports those
devices, and launches `accelerate launch --num_processes N`. Its production
orchestrator is `pack → train → publish`: a cold acceptance run includes first BF16
feature-pack construction, the configured epoch count with validation after every
epoch, and final artifacts. The retained `probe`/`epoch-probe` worker dispatch entries
are non-publishing measurement-only paths, not production orchestration stages; the
projection stage is deleted.

The epoch count and the complete-interval budget are set by the run's config rather
than fixed here, because a schedule change is a recipe change: `optim.scheduler`
sizes OneCycle from the total step count, so the same model trained under a
different epoch count follows a different LR trajectory, not a truncated one. Two
recipes are currently normative:

- **Constant-LR recipe (historical):** 30 epochs, warmup-then-constant LR, complete
  interval at most 60 minutes. This is the recipe every B0 artifact through
  2026-07-14 was produced under.
- **OneCycle recipe (legacy V3.1 reproduction):** 25 epochs, `optim.scheduler.type:
  onecycle`, `weight_decay 0.05`, `label_smoothing 0.05`, complete interval at most
  63 minutes (`configs/b0_v31_breadth_first.yaml`).

**Do not assume the annealed endpoint is the best model.** Measured on
`breadth_first` seed 47 (2026-08-03), validation AUPRC peaks at roughly **half**
of whatever OneCycle schedule it is given — epoch 21 of 50, epoch 13 of 25 — and
decays thereafter while train loss keeps falling. There is no epoch count at which
the final epoch is also the best epoch; **best-by-val-AUPRC checkpoint selection is
what carries the run**, and the epoch count is chosen for calibration and cost, not
to let the anneal "finish". Shortening 50 → 25 left AUROC/AUPRC unchanged within
single-seed noise but improved test ECE from `0.2000` to `0.1526` and halved the
training wall clock.

`runtime.*_budget_seconds` must sum exactly to `runtime.total_budget_seconds`, and
`train_eval_budget_seconds` is a hard subprocess timeout — under-sizing it fails the
run rather than shortening it. Early stopping is counterfactual-only in the DDP path
(`eval.patience` records `counterfactual_stop_epoch` and never breaks), so the
configured epoch count always runs in full.

Each rank owns one model/optimizer replica and one complete GPU-resident BF16 feature
table. DataLoader workers transfer compact endpoint indices only. Training and
validation coverage are exact; tail-batch loss is weighted by local/global pair count.
The checkpoint payload consumed by `score_universe` is unchanged.

## 12. Change log

- 2026-08-04 (owner decision): §11's OneCycle recipe drops from 50 to **25 epochs**
  (complete interval at most 63 minutes), and the rationale recorded in the
  2026-08-03 entry below is **corrected**. That entry argued the epoch count could
  not simply be lowered because "a OneCycle schedule must run its full `total_steps`
  to reach its annealed floor". The mechanism is real but the conclusion was wrong:
  measured on seed 47, validation AUPRC peaks at ~50% of *any* OneCycle schedule
  (epoch 21 of 50; epoch 13 of 25) and decays afterwards, so the annealed endpoint is
  never the selected model and best-by-val checkpoint selection is what carries the
  run. 50 → 25 left discrimination unchanged within single-seed noise (headline test
  AUROC `0.7162` → `0.7141`, AUPRC `0.7437` → `0.7454`) while improving ECE `0.2000`
  → `0.1526`, lifting full-universe AUPRC `0.1260` → `0.1652`, leaving all three
  assembled-graph MMD ratios unchanged within 2%, and halving the training wall clock
  (2,113 s → ~1.1 ks). Budgets are re-sized from the measured ~42 s/epoch; the
  previous 9,780 s total was ~4.6× looser than needed. Evidence is fixed-Seed-0-style
  single-seed engineering evidence (`outputs/runs/b0_v31_onecycle_s47`,
  `outputs/runs/b0_v31_e25_s47`) — it carries no significance or cross-seed claim.
- 2026-08-03 (owner decision): the training-edge **provenance digests are
  withdrawn**. The shared-edge refactor (`6315519`) had bound every
  checkpoint, `run_metadata.json`, scores artifact, and CAZI UGT cache to a
  `data_contract = shared_train_positives_v1` tag plus `training_interactions_sha256`
  and `training_topology_sha256`, and made `src/e2_pipeline.py`,
  `src/experiments/g5_stage1.py`, `src/score_universe.py`, and
  `src/train_cazi_mbn.py` fail closed on any mismatch. None of it is recorded or
  checked any more: artifacts carry no training-data identity field, and stale
  pre-refactor checkpoints/artifacts no longer fail closed on that basis — they are
  distinguished by run directory and commit, an owner-side responsibility. §14.4.6
  is amended accordingly, as are protocol §5 and model-proposal §6. Also withdrawn
  with it, by the same decision: the `run_metadata.checkpoint_id == loaded
  checkpoint` equality check in `score_universe._run_score`; the `load_config`
  unknown-key rejection in `src/train_cazi_mbn.py`; the identity fields
  (`topology_sha256`, `order`, `feature_length`, `seed`) inside the CAZI UGT
  `.npz` cache; and the `egostitch_e2e` bare-checkpoint rejection in
  `score_universe._load_checkpoint`. Three consequences are accepted, not
  overlooked:
  1. `checkpoint_id` is still bound one stage later — `g5_stage1.build_e2e_arm_summary`
     requires each arm's score artifact to match its `run_metadata.json`. Scoring the
     wrong checkpoint against another arm's metadata now succeeds through the
     held-out pair read and the test-access ledger write, and fails only at G5
     aggregation, after the ledger has been written.
  2. A pre-refactor `ugt_projection.npz` left in a CAZI `output_dir` is silently
     reused: only node order and shape are checked. Existing stale caches under
     `outputs/cazi_mbn_breadth_first/` must be deleted by hand before any rerun.
  3. `model.order`/`seed` changes no longer invalidate that cache either.
- 2026-08-03 (owner decision): §11's fixed "exactly 30 epochs / at most 60 minutes"
  formal-B0 binding is replaced by a config-set epoch count and budget, with two
  normative recipes listed — the historical constant-LR 30-epoch recipe and a new
  OneCycle 50-epoch recipe reproducing the legacy V3.1 result (`optim.scheduler.type:
  onecycle`, `max_lr 1e-4`, `pct_start 0.1`, `div_factor 25`, `final_div_factor
  10000`, `anneal_strategy cos`; `weight_decay 0.05`, `label_smoothing 0.05`; complete
  interval at most 163 minutes). The two are coupled, which is why the epoch count
  could not simply be raised: a OneCycle schedule must run its full `total_steps` to
  reach its annealed floor, so truncating epochs changes the recipe rather than
  shortening it. Enabling code: `optim.scheduler` block and `V3_1(label_smoothing=...)`
  in commit `66b00ea`. Early stopping was already counterfactual-only in the DDP path,
  which is what permits the schedule to complete. Per §14, this authorizes
  implementation, not execution.
- 2026-08-03 (owner decision): §9.4 rule 3 reporting refined. Self-loops are retained
  by default in **every** labeled pair view (val and test); no view strips them from
  its headline edge metrics. Only the **test** view additionally reports the self /
  non-self split, and it reports the self-loop-*including* block **first** with the
  split supplementary beneath it. Rationale: the split exists to expose self-loop
  behavior in the final reported result, not to re-cut model-selection numbers.
  Implemented in `src/eval/report_edge_metrics.py`, which is also the first committed
  code path for the §10.3 balanced `test_edges.txt` edge metrics — previously that
  number had no writer in `src/` (the G1/G2/G3/G5 analyzers all reject
  `pairs_source != "candidate"`). Measured on the existing 2026-07-10 B0 artifact:
  all 1,891 `(u, u)` test rows are positive and all 1,891 are predicted positive, so
  the self stratum is single-class (AUROC/AUPRC undefined, emitted as null) and the
  non-self stratum sits at AUROC `0.6883` / AUPRC `0.6894` against the
  self-loop-including `0.7067` / `0.7316`.
- 2026-08-03 (owner decision; data-contract correction): removed the seeded 80/20
  message/supervision split. Topology learning and edge classification now consume the
  same complete `E_train+` edge set; only the already-pinned loopless topology
  projection removes self-pairs. The internal `V_hold` is consequently derived from the
  full training topology. This supersedes every contrary message/supervision clause in
  this document and invalidates all packs, caches, digests, thresholds, and results
  produced under the former split; they remain historical evidence only and are not
  comparable to runs under this contract.
- 2026-08-03 (owner decision): B0-alt withdrawn as a baseline family. The mature B0
  (V3.1) is now the sole baseline going forward; `03-experiment-protocol.md` §2's
  baseline-hierarchy table and its E3 run list, and `04-model-proposal.md` §6.0 G1's
  re-run requirement list, no longer name B0-alt. Its implementation
  (`src/model/b0_alt.py`, `F0PairMLP`) was removed from the code tree in commit
  `72db72c`. No normative content in this document changes: the two existing
  historical B0-alt mentions — the 2026-07-14 entry below recomputing
  `B0/B0-alt/PA-null/legacy/G3` values, and §14.4.4's measured grounding P0.2 curve —
  record completed, already-run measurements and are retained unchanged; provenance
  for reproducing B0-alt numbers is in `docs/results/E2-pair-to-topology-gap.md`.
- 2026-07-30 (arm-schema migration to the v5 component-ablation set; supersedes
  the §14.4.6 eight-arm listing): the trained-arm set becomes `full`,
  `b0_e2e_f_only`, `pair_topology`, `p0`, `no_l_rel`, `row_layernorm`;
  `cosine_pool` retires from the trained set (the Phase-0 measured slot-recall
  ceilings, top-50 0.1395 vs top-20 0.1073, already bound the pool-width
  effect, and its registered top-20 pack/caches remain on disk as historical
  evidence). The new `row_layernorm` arm is identical to `full` except
  `feature_standardization: row_layernorm` (the pre-D0 status-quo per-row
  LayerNorm; binds no `feature_stats_sha256`), giving the rev-3.2 D0
  per-dimension z-scoring mechanism the ablation arm the §4.6 anti-grab-bag
  rule requires. The two scoring-time controls are unchanged. Probe
  (`egostitch_e2e_probe_v2`) and scores-meta (`egostitch_e2e_scores_v3`)
  versions did not bump at that time. This historical registration-era claim
  is retired by the 2026-08-03 shared-edge contract and the current
  `egostitch_e2e_scores_v4` provenance checks. Rationale at the time: the
  formal screen must attribute each critical mechanism — conditioning as a
  whole, the content pathway, branch dropout, `L_rel`, and D0 feature
  standardization — to an arm it owns.
- 2026-07-30 (owner decision; supersedes registration-state and pre-run
  evidence-completeness clauses): formal execution no longer requires
  `status: BINDING`, resolved `binding_evidence`, or any run-produced evidence
  to exist before the run that produces it. Registration status and nullable
  evidence slots are descriptive provenance only. Preflight keeps the exact
  registration bytes/SHA, registered arm/config and frozen-input identity,
  clean implementation checkout, repository/runtime boundary, and exactly four
  visible H20s. Implementation, parameter-group manifests, pack/validation
  manifests, boundary audit, runtime/peak-memory evidence, and checkpoint-policy
  provenance must be recorded from the actual run and verified downstream
  against its artifacts; absent placeholders in the registration cannot block
  training, scoring, or evaluation. Owner rationale: requiring run-produced
  evidence before execution was circular and added no artifact-integrity proof.
- 2026-07-30 (owner decision; supersedes every qualification/formal-ladder and
  model-quality-gate clause below): the active rev-3.2 experiment is a
  **single-stage, plan-bound formal run**. Qualification, calibration,
  `pending_manual_review`/`pass`, qualification-history disclosure, and
  qualification-artifact preflight are retired and authorize or block nothing.
  The only execution/publication coupling is between the immutable experiment
  plan identity (registration SHA-256, exact arm/config and implementation
  identity, frozen input manifests) and the artifacts/metadata/provenance of
  that same run. Every finite model-quality predicate—including checkpoint
  eligibility, validation liveness, slot collapse, clipping/family/submodule-RMS
  margins, AUPRC, dispersion, and precision differentials—is telemetry-only and
  may not stop training, checkpoint/artifact publication, scoring, or evaluation.
  If no checkpoint meets a quality predicate, the final completed epoch is the
  plan-selected checkpoint and records the miss. Non-finite values, DDP
  disagreement, incomplete/duplicate coverage, data-boundary violations,
  malformed or hash-mismatched inputs/artifacts, and I/O/infrastructure errors
  remain fail-closed truthfulness/safety failures. Owner rationale: experimental
  authorization comes from a locked plan and auditable provenance, not from
  post-hoc model-quality formalism.
- 2026-07-30 (owner-confirmed, supersedes the earlier qualification-verdict
  amendment below): qualification is a complete three-epoch telemetry run, not a
  finite model-quality gate. Every finite threshold or quality predicate is
  telemetry-only in qualification: initial and during-training slot statistics,
  finite zero gradient/family norms, immediate and persistent clip coefficients,
  family-norm ratio, warm-reference and validation logit-dispersion floors,
  BF16/fp32 precision differentials, validation-collapse predicates, and checkpoint
  eligibility. The registered clipping operation remains unchanged. Only non-finite
  values, DDP synchronization/equality failures, boundary violations, incomplete or
  inexact coverage, and I/O/infrastructure failures abort qualification. A run that
  completes all three epochs writes `pending_manual_review` even if no quality-eligible
  checkpoint exists; in that case the final epoch is exposed through the existing
  checkpoint compatibility aliases for diagnostic manual review only, and neither
  `best.pt` nor `last.pt` authorizes formal. Formal
  guards, eligibility, selection, and fail-closed preflight are unchanged. This entry
  defines no approval artifact or automatic conversion to `pass`; v4 remains `DRAFT`.
- 2026-07-30 (owner-confirmed): qualification no longer converts persistent
  clipping into an automatic abort or an automatic pass. The registered clipping
  operation itself is unchanged: the pair and generator groups are still clipped
  to global L2 norm `3.0`, the conditioning group to `1.0`, and every pre-clip norm,
  clip coefficient, and consecutive-low-coefficient streak remains complete
  telemetry. Specifically, the `< 0.1` for ten consecutive steps condition remains
  a hard failure in formal training but is telemetry-only during qualification;
  the one-step `< 1e-3` extreme, family-imbalance guard, logit-collapse guard,
  slot-collapse guard, and no-eligible-checkpoint outcome were still hard failures
  under this intermediate amendment; the owner-confirmed entry immediately above
  supersedes that qualification-only treatment. A qualification that completes its
  entire reduced schedule and retains complete telemetry writes `pending_manual_review`, never
  `pass`. `pending_manual_review` is not formal authorization and formal preflight
  rejects it. This entry defines no approval artifact, no automatic conversion to
  `pass`, and no in-place edit of an immutable attempt; the owner will decide any
  later review/approval mechanism separately. The v4 registration remains `DRAFT`.
- 2026-07-30: specified the deterministic selection-band calibration procedure for
  `select_e2e_checkpoint`'s `auprc_tolerance`. The immutable source is the first
  full-arm Seed-0 qualification attempt under the landed implementation that reaches
  the first validation after the conditioning ramp plus one complete Phase-C epoch
  and writes the complete calibration source artifact. That existing validation is
  reused and remains in the complete qualification history / cumulative `V_hold`
  evaluation count K; calibration creates no additional K event. The method uses only
  the canonical non-self `V_hold` pair labels and active full fp32 logits, with the
  stratified pair bootstrap, estimator, rounding rule, and evidence acceptance pinned
  in §13.19.3–§13.19.6. It changes neither the fixed `0.02` eligibility constants nor
  the `1e-6` MMD tie tolerance and authorizes no execution. The v4 registration uses
  `egostitch_e2e_binding_evidence_v2`, which extends historical v1 by making
  `auprc_tolerance_calibration` mandatory; historical v1 registrations and artifacts
  remain unchanged.
- 2026-07-30: bound qualification exposure to the exact immutable attempt set.
  Each trained arm's `attempt_history.json` uses schema
  `egostitch_e2e_qualification_history_v1`; before v4 may bind,
  `binding_evidence.qualification_history_indexes` maps exactly the six trained
  arms to path-and-SHA-256 references for those files, and
  `binding_evidence.qualification_attempts` maps the same arms to non-empty lists
  exactly equal to each referenced index's complete `attempts` array. This closes
  omission and stale-index routes in the cumulative `V_hold` evaluation count K;
  it changes no acceptance threshold and authorizes no execution.
- 2026-07-29 (second entry): recorded deviation from the two-stage cleanup
  design's §4 delete list. Two items on that list were deliberately **kept**;
  both retentions are load-bearing rather than oversights, and neither was
  documented when it landed.
  - **`_PROBE_DISPATCH_MODES` is retained** (`train_egostitch.py:1361`,
    `("probe", "epoch-probe")`), and with it `--ddp-mode epoch-probe`
    (`DDP_MODES`, `train_b0.py:74`). It is the exemption clause in
    `_bind_feature_standardization` (`train_egostitch.py:5918`) and is
    therefore the only remaining way to *measure* `feature_stats_sha256`
    before any stage has recorded it. §13.19.4's formal preflight compares the
    formal run's computed digest against the one the qualification stage
    wrote, so the digest must be observable from a dispatch that builds no
    checkpoint, writes no run-start metadata, and publishes no artifact.
    Without the exemption the pin requirement would apply to a mode that
    cannot satisfy it, and the qualification stage — where the digest is born
    — would need it as a config input, which is precisely the circularity the
    two-stage ladder removes. The rest of §4's probe list was deleted as
    designed: `--ddp-mode init-probe`, `_run_init_probe`,
    `select_probe_result`, `conservative_e2e_epoch_seconds`, and
    `project_total_seconds` no longer exist in `src/`.
  - **`runtime.total_budget_seconds` is retained** as a required
    `RuntimeConfig` key (`train_egostitch.py:524`, `:560`; `train_b0.py:181`)
    together with the invariant that the per-stage budgets sum to it exactly
    (`train_egostitch.py:603`). What §4 deleted is the *projection sub-stage*
    that consumed it, not the wall-clock budget declaration; all six v3 arm
    configs and `b0_v31_breadth_first.yaml` still pin it, and the sum
    invariant is the only remaining check that a config's stage budgets are
    internally coherent.
  Implementation-level only: no verdict inequality, registration status,
  binding threshold, or data-contract quantity changes in this entry.
- 2026-07-29: two-stage cleanup rewrite (design:
  `docs/superpowers/specs/2026-07-29-egostitch-e2e-two-stage-cleanup.md`, r2;
  owner-decided via grill-me interview). §9.3, §13.12, §13.19.1, §13.19.3,
  §13.19.4, §13.19.6, and §14.4.7 are rewritten in place — no `§15`, no
  SUPERSEDED strata — collapsing the e2e ladder from five stages
  (init-probe → calibrate(overfit) → threshold-freeze → rehearse → formal) to
  two: **qualification** and **formal**. Both train on the full `V_fit`
  universe and differ only in `optim.epochs`; a single 512-node holdout
  `V_hold := V_qual ∪ V_select` validates and selects checkpoints for both, so
  `V_fit`, the then-named `e_msg_fit`/`e_sup_fit` partitions, `G_fit`, and
  `feature_stats_sha256` stayed
  bit-identical to the two-holdout definition. The qualification verdict is
  telemetry-completion-only for finite model-quality signals; only non-finite,
  DDP, boundary, coverage, and I/O/infrastructure failures abort it. Checkpoint
  eligibility (§13.19.3) is still computed and reported during qualification but
  governs checkpoint selection only in formal. Its absolute floor is lower under
  the `V_hold` union — disclosed at §13.19.3, not "unchanged". The formal stage carries pre-registered acceptance
  thresholds recorded before test opens (protocol §5.2.4). §13's G5 Stage-1
  carve-out gains a short scope note: §13.1-13.17 govern the completed,
  published frozen-s0 screen only (`docs/results/G5-stage1-seed0-20260717.md`,
  verdict `cut`); its implementing code is retired under this cleanup and no
  future run may cite it as authorization. Retired with their machinery, not
  restated here: the 510-row overfit test, the single-`V_qual`-rehearsal
  budget, `calibration_freeze.json`, the exclusive-create rehearsal ledger,
  the `attempt00[1-3]` window, `src/experiments/prebinding_gates.py`, the
  `REQUIRED-BEFORE-BINDING` markers, and `qualification_margins.json`.
  **Editing-in-place mitigation:** this document has published results citing
  its pre-rewrite text (`docs/results/G5-stage1-seed0-20260717.md`,
  `docs/results/G5-e2e-stage1-seed0-20260724.md`); the pre-rewrite commit is
  `26abf7f89aac47d4e65c6e81b3557ed42ff30d71` —
  `git show 26abf7f89aac47d4e65c6e81b3557ed42ff30d71:docs/05-egostitch-spec.md`
  resolves those citations to the text the results were actually produced
  under. Authorizes implementation only per the spec-freeze rule; a fresh
  registration (v4) is required before any binding run.
- 2026-07-27 (seventh entry): §13.19.1 replaces the E2E generator's per-row
  `LayerNorm` F0 standardization with registered per-dimension z-scoring over
  the ordered V_fit universe (`zscore_vfit_v1`), with a pinned two-pass fp64
  estimator, an fp32 canonical value, a `feature_stats_sha256` identity bound
  into the config, the pack manifest, the access audit, and the checkpoint
  buffers; the rev-3.1 transform is retained as `row_layernorm` for replay
  only. §7/§13.5: the SSL feature perturbation is applied in standardized
  coordinates. Found by the 2026-07-27 diagnosis of the rev-3.1 attempt-001
  `training_invalid(slot_collapse)`: raw F0 rows have mean pairwise cosine
  0.949, per-row LayerNorm cannot remove a shared cross-row mean direction,
  and a randomly initialized generator therefore reads
  `h_pairwise_cosine_mean = 0.9897` — above the §14.4.8 trip line — while its
  transport plan is healthy (mass 0.069, rank-1 residual 0.87). The guard was
  measuring a feature-geometry floor, not training dynamics. Measured effect
  of the edit on a deterministic 300-node real-F0 probe: target-projection
  cosine 0.9405 → −0.0013, init slot cosine 0.9897 → 0.62, ‖h‖ 9.12 → 9.29.
  §14.4.8 (criteria, thresholds, arming), the `egostitch_e2e_probe_v2` schema,
  the eight-arm v3 screen, the five G3 gates, and the §13.12 raw-F0 pool
  contract are all unchanged; new scale diagnostics are training-log only.
  Consequence: `config_hash` changes, so rev-3.1 calibration artifacts are
  inadmissible for rev-3.2 threshold freezing and a new registration version
  is required. Owner-confirmed 2026-07-27. Refinement (same landing): item 2's
  degeneracy and non-finite checks are specified to run **before** the
  variance floor, found by review of the D0 (`src/data/feature_stats.py`)
  implementation because the floored form made the "hard error" structurally
  unable to fire — `sqrt(max(0, 1e-12)) = 1e-6` survives the fp32 cast intact,
  so a mathematically constant dimension was silently floored rather than
  rejected.
- 2026-07-26 (sixth entry): §14.4.3 moves the §13.19.2 checkpoint-eligibility
  warm-reference AUPRC snapshot from Phase A to the first validation after
  conditioning activates. The requirement itself is unchanged — `>= prevalence
  + 0.02`, "at least", as §13.19.2 item 3 states it. **Only the snapshot point
  moves; neither the threshold nor the inequality changes.** (An earlier draft
  of this entry wrote the shorthand `> prevalence + 0.02`, which tightened the
  boundary case; that was a wording error, not a decision, and is corrected
  here.)
  Found by Codex review of the rev-3.1 branch: the joint-entry curriculum
  (§14.4.3) disables `L_edge` and freezes the pair encoder during Phase A, so
  the Phase-A snapshot measures an untrained random pair head and would leave
  `full`, `p0`, `cosine_pool`, and `no_l_rel` eligible only by initialization
  luck. The rule and the curriculum were written against each other; reading
  the reference where the pair head has trained preserves the guard's intent
  (catching a head that never learns) without weakening its threshold.
  Owner-confirmed 2026-07-26.
- 2026-07-25 (fifth entry): §14.4.5 6e-v1 transfer gains a
  **recipient-capacity cap**, `δ = u · min(w_il, w_kj, c_ij − w_ij,
  c_kl − w_kl)` with `c_ij = π_i π_j`. Found by Codex review of the fourth
  entry's π-weighted swap space: mapping back `Â̊' = W'/(π_i π_j)` can drive
  a cell orders of magnitude above 1 when the recipient's capacity is smaller
  than the donor's, violating §2's `Â ∈ [0,1]` sigmoid contract, and
  `build_scaffold` then propagates the out-of-domain value into `CLOSE` and
  `t_k` — letting the 6e arm be driven by artificial scale explosion instead
  of rewiring beyond degree. §2 is the earlier, foundational contract and
  governs; the cap is the minimal edit satisfying both, and since the
  checkerboard pattern preserves every marginal for *any* `δ`, capping costs
  no invariant. Required test: rewired `Â̊` stays within `[0, 1]` under
  adversarially non-uniform `π` including near-zero entries.
- 2026-07-25 (fourth entry): §14.4.5 6e-v1 swap space corrected from the raw
  `(Â̊_src, Â̊_dst, Π)` tensors to the **π-weighted** slot adjacency, with the
  binding invariant restated as the rebuilt scaffold's `STAR`/`INTRA`/`ALIGN`
  degree channels. Found during implementation: `build_scaffold` forms INTRA
  as `Â̊ ⊙ π π^T` and CLOSURE from matrix products, so preserving raw
  marginals does not preserve what the model actually sees — a deterministic
  probe measured rebuilt-degree drift of `[STAR 0, INTRA 0.383, ALIGN 3.6e-7,
  CLOSE 1.651]`. Leaving degree in a degree-preserving control would let a
  reviewer attribute part of any 6e metric move to changed degree rather than
  destroyed higher-order connectivity, defeating the control's registered
  purpose. `CLOSE` is explicitly **not** an invariant (closure mass is the
  higher-order signal being destroyed), and draws are restricted to
  off-diagonal cells so `Â̊`'s zero diagonal survives. Owner-confirmed
  2026-07-25. 6a-v3 is unchanged.
- 2026-07-25 (third entry): §14.4.2 scaffold format corrected from FEAT_DIM
  10 to **11**. The rev-3.1 scaffold adds both a fourth edge type (closure)
  and a per-slot closed-wedge feature `t_k`, but the degree slice is one
  channel per edge type (`adj.sum(-1)`, `scaffold.py:114`), so EDGE_TYPES
  3 → 4 by itself widens `feats[:, :, 6:9]` to `6:10` and consumes the tenth
  channel — leaving no index for `t_k`. FEAT_DIM 10 was therefore
  arithmetically impossible; the same miscount appears in the r6 design
  trail §D1/D2. Owner-confirmed 2026-07-25 during implementation: keep both
  features, layout `[onehot4(anchor); π; mult; deg×4; t_k]` with `t_k` at
  index 10. No other §14.4 value changes.
- 2026-07-25 (second entry): §14.4.8 collapse-abort refinement from the
  measured container P0.1/P0.3 autopsy of checkpoint `a471010f57e495f0`
  (`outputs/p0_audit_20260725/p0_autopsy_results.json`): the Π row-entropy
  arm was measured blind to full collapse (0.624 normalized on a fully
  collapsed plan, because the `π·m` marginals concentrate rows) and is
  replaced by a rank-1 marginal-residual criterion; the `h`-cosine arm is
  unchanged (measured 0.9997 vs the 0.95 trigger — it catches the v2
  collapse outright). Autopsy also confirms: pointer exactly uniform
  (max prob 0.052 ≈ 1/n_g, entropy 0.9999), gate never above 0.5 anywhere
  (mean 0.088), Â off-diagonal constant 0.5014 ± 0.0004.
- 2026-07-25: added §14.4 — the rev-3.1 relational-repair contract for the
  next `egostitch_e2e` build (owner-delegated decisions, design trail
  `docs/superpowers/specs/2026-07-25-egostitch-e2e-relational-repair-design.md`
  r5; P0 audits at `outputs/p0_audit_20260725/`). New `L_recon` components
  (`L_ptr`/`L_align`/`L_div`/`L_rel`) with a masked all-reduced reduction
  contract and node-identity-keyed target sampling; soft matched flags;
  logit-space `L_slotadj` via `SlotSet.adj_logits`; scaffold closure channel
  (FEAT_DIM 10, EDGE_TYPES 4); centered gated conditioning (supersedes the
  §14.1 inject equation for rev-3.1); joint-entry curriculum (supersedes
  §13.19.1 Phase-A `pair_only` for rev-3.1); grounding `n_g = 50` with the
  widened `pool_method_hash` cache manifest (supersedes the §13.12 value for
  this family; the measured P0.2 curve rejected the reranker and left
  protocol §0 untouched); 6a-v3/6e-v1 controls; the eight-arm v3 screen
  schema with full provenance migration and artifact version bumps;
  Π-consistency v2; and the then-proposed V_fit-calibrate → V_qual-rehearsal
  qualification gates, superseded by the 2026-07-29 two-stage cleanup entry above.
  The completed v2 screen's record (§13.19, four trained checkpoints plus one
  scoring-time control, `n_g = 20`) is unchanged as history; formal execution
  at that time required an owner-promoted BINDING v4 registration, a state gate
  retired by the latest 2026-07-30 owner decision above.
- 2026-07-24: gate-side telemetry shape fix. The §13.17 registered names
  `grad_rms_trunk`/`grad_rms_ste`/`grad_rms_content` are published by the
  worker nested under `submodule_gradient_rms` per fixed-replay gradient row;
  the e2e gate validator expected them flat and failed closed on the completed
  formal metadata. The validator now accepts the published nested shape (flat
  rows remain accepted); registered names, values, and validity rules are
  unchanged, and no published artifact was modified.
- 2026-07-24: probe-producer identity fix for the registered §14.3 probe
  artifact. The producer reused the worker's internal-holdout training view
  (V_fit nodes over `G_fit`), while the registration and the gate's
  reconstruction pin the probe to all operative train nodes over the
  full-`E_msg` `G_struct` — the gate correctly failed closed on the mismatch
  before any probe metric was computed. The producer now derives the
  registered identities exactly as the gate does and grounds each node in its
  §13.12 role universe (`V_fit`/`V_qual`/`V_select`), with private derived
  caches under the probe output directory. The probe artifact remains
  nonbinding diagnostic evidence (`verdict_effect: none`); no training,
  scoring, eligibility, or verdict semantics change.
- 2026-07-23: candidate-scorer locator fix in the §13.19.5 formal scoring entry
  point. The four-arm provenance check required a `selected_checkpoint_path`
  field that the formal worker never writes; the scorer now falls back to the
  published layout (`best.pt` beside each arm's `run_metadata.json`) when the
  field is absent. Checkpoint identity remains enforced by the unchanged
  `checkpoint_sha256` comparison, so the path is a locator only; no training,
  eligibility, precision, or verdict semantics change and the BINDING
  registration file is untouched (its digest is already embedded in all four
  formal run metadata records). Locked by a unit test covering the
  missing-sibling failure and the fallback-located path.
- 2026-07-23: binding-mechanics amendment to the §13.19.4 item-5 [historical
  numbering; restored 2026-07-29 as new item 5, the `binding_evidence` schema
  and commit-identity rule] formal commit-identity check. As implemented, the
  formal worker required a clean
  checkout whose HEAD begins with the recorded `binding_evidence`
  implementation commit, while the registration is itself a tracked file — a
  registration cannot contain its own promotion commit's hash, so binding was
  mechanically impossible as written. The check now also accepts a clean HEAD
  that descends from the recorded implementation commit exclusively through
  commits touching `docs/registrations/` paths. This preserves the check's
  intent (the executing code is exactly the qualification-frozen
  implementation) and changes no training, precision, guard, eligibility,
  scoring, or verdict semantics; the qualification code path never invokes it.
  Locked by unit tests covering the equal-HEAD, registration-only-descent,
  non-ancestor, and non-registration-diff cases.
- 2026-07-23: disclosed branch-dropout mask correlation across DDP ranks for
  the frozen V2 trajectory. The registered per-pair `∅`-mask realization
  derives its dropout randomness without incorporating the DDP rank, so the
  per-step mask draws are correlated (identical per local batch index) across
  ranks; the expected per-pair mask rates `p_topo = p_cont = 0.15` are
  unchanged and the property is identical across all four training arms. This
  is a disclosed implementation property of the qualified attempt-3 trajectory
  (external branch review, 2026-07-22), accepted as-is for the v2 screen:
  correcting it would invalidate the completed rehearsal and consume the
  remaining attempt allowance, so any fix is deferred to a future versioned
  registration (v3) or the E1/E3 multi-seed builds.
- 2026-07-22: calibrated the §13.19.2 clip-margin bands from the first completed
  replacement rehearsal. Rehearsal attempt 3 completed the full 30-epoch
  schedule and published its artifacts with every in-run gate green: no
  stability guard fired, an eligible epoch-16 checkpoint was selected with
  liveness pass, validation residual ratio grew to `~0.18–0.23`, and both
  precision differentials passed under the vector bounds. The post-run margins
  validator then failed solely on the scaffold-era global clip-coefficient
  `p1 > 0.12` band: measured per-group `p1` is `0.1100` (`pair_encoder_head`),
  `0.0281` (`generator`), `0.5187` (`topology_content_conditioning`), while
  every other margin passed as registered (minimum `0.001773 > 0.0012`,
  longest streak below `0.1` of `5 < 10`, family-ratio `p99 13.6557 < 40`,
  exact `2340/2340` step coverage, 46 complete RMS probe rows). This section
  itself pinned those thresholds as DRAFT "until the passing rehearsal records
  their empirical distribution"; that distribution is now recorded, and the
  single global floor — introduced in the qualification scaffold before any V2
  rehearsal existed and never re-derived after the registered generator
  clip-ceiling and F0-standardization changes — is replaced by per-group
  floors at roughly `2.5–3x` below measurement (`0.04`/`0.01`/`0.15`; unlisted
  groups default to the old `0.12`). The generator's persistently-clipped
  regime (median coefficient `0.226` against ceiling `3.0`) is disclosed as a
  real trajectory property, protected in-run by the unchanged
  immediate/persistent clip aborts and family-ratio guard. Re-validating the
  retained attempt-3 rehearsal profile under the calibrated floors launches no
  new rehearsal and consumes no attempt; minimum/streak/ratio bands and all
  in-run guards are unchanged. No held-out/candidate/test quantity informed
  this calibration.

- 2026-07-22: replaced the §13.19.4 item-3 [historical numbering; restored
  2026-07-29 as new item 4, the precision-differential guard] per-element
  logit tolerance with vector relative-L2 bounds after replacement rehearsal
  attempt 2. The attempt
  was the healthiest run to date: the full 30-epoch schedule completed with no
  stability guard firing, an eligible checkpoint was selected (including the
  validation-side `1e-3` residual floor), and the end-ramp differential passed
  under the same-day calibrated `atol 0.05`. The selected-checkpoint
  differential then failed on the per-element conjunct alone: max abs error
  `0.1045`/`0.0907` (`f_logit`/full) versus `atol 0.05`, with the residual
  contract healthy again (relative L2 `0.0161` vs `0.05`, correlation
  `0.99983`). Two successive single-point atol calibrations were each
  invalidated by the next measurement because per-element max-abs error is an
  extreme-value statistic of BF16-trunk quantization noise that grows with
  training scale — the wrong contract form, not a wrong constant. Full and
  `f_logit` are now each bounded by vector relative L2 `<= 0.05` versus pure
  fp32: provably slack whenever the residual bound holds under path-specific
  noise (`||residual|| << ||full||` makes the residual bound far more
  sensitive to the same absolute noise), while still independently catching
  common-mode corruption that cancels in the residual and gross logit-scale
  single-element corruption. Residual bounds, the correlation floor, the
  non-zero rule, and §13.16 fp32 pair-pass scoring are unchanged; per-element
  max-abs errors remain logged diagnostics. This consumed rehearsal attempt 2;
  one replacement attempt remains before v3.

- 2026-07-22: recalibrated the §13.19.4 item-3 [historical numbering; restored
  2026-07-29 as new item 4, the precision-differential guard] elementwise
  tolerance after the first replacement rehearsal attempt failed at the
  end-ramp differential with
  the residual contract healthy. Measured on the retained failure evidence:
  residual relative L2 `0.0025666` (ceiling `0.05`), correlation `0.9999923`
  (floor `0.999`), non-zero in both paths — the fp32-island correction works
  for the quantity it targeted. The failed conjunct was the elementwise logit
  check: max abs error `0.0176066`/`0.0176032` (`f_logit`/full) against
  `atol 1e-5 + rtol 1e-3 * |logit|`. That bound is rtol-dominated and
  logit-magnitude-dependent: it was validated only on the saturated-logit
  V_fit overfit checkpoint (effective tolerance ~`0.02` at `|logit|~20`), and
  at ordinary end-ramp rehearsal logit magnitudes the identical BF16-trunk
  quantization noise exceeds it — the check measured logit magnitude, not
  precision health. The elementwise `atol` is replaced by the
  end-ramp-calibrated `0.05` (`2.8x` margin over the measured error; `rtol`
  unchanged), matching the residual ceiling's calibration style. Residual
  bounds, correlation floor, and the non-zero rule are unchanged; §13.16
  already pins published candidate/test scoring to the fp32 pair pass, so no
  published score depends on the BF16 path. The registration text is aligned
  in the same amendment, including the same-day `1e-6` overfit-floor
  recalibration its overfit requirements had not yet reflected. Calibration
  used only the rehearsal's fixed train-side replay; no candidate/test or
  `V_select` quantity was read. This consumed rehearsal attempt 1 of the
  replacement's three-attempt allowance.

- 2026-07-22: recalibrated the §13.19.4 item-1 [historical numbering, retired
  2026-07-29 with the overfit machinery; distinct from the new item 1, the
  guards-only verdict] overfit residual-ratio floor from `1e-3` to the
  fp32-calibrated `1e-6`. The first gatefix run's retained
  per-epoch failure history (attempt-005, four H20, seed/config/pack unchanged)
  shows the honest fp32 readout path yields Phase-C residual ratios
  `3.622e-6 → 1.2178e-5`, smooth and monotonically increasing across all 22
  post-ramp epochs with train AUPRC `1.0` from epoch 6 — a live, learning
  conditioning pathway two orders of magnitude below the old floor. The BF16-era
  `1e-3` readings this floor was set against were readout quantization noise:
  the same checkpoint measured an all-zero mixed residual in the precision
  replay (see the fp32-contract entry below), which already replaced the
  untested `1e-3` precision-differential bound with a calibrated `0.05` on
  identical grounds. The new floor passes the measured trajectory with `3.6x`
  margin at its weakest epoch while still failing the dead-pathway zero it
  exists to catch. The §13.19.3 formal/rehearsal eligibility floor (`1e-3` on
  validation residual) is deliberately unchanged: no fp32 validation-side
  measurement exists yet, and a failed rehearsal now retains its per-epoch
  history for exactly that calibration. No held-out/candidate/test quantity
  informed this correction.

- 2026-07-22: pinned the §13.19.4 item-1 [historical numbering — overfit
  acceptance, retired 2026-07-29 with the overfit machinery; distinct from the
  new item 1, the guards-only verdict] to its registered
  "reaches ... after the conditioning ramp" wording. The implementation had
  accepted only the final reporting epoch, a stricter unregistered rule: the
  retained passing `V_fit` trajectory oscillates around the `1e-3` residual
  floor across Phase C (`0.000946`–`0.001495`, six of twenty-two post-ramp
  epochs below the floor), so the registered fp32-readout correction's benign
  trajectory shift flipped the final epoch below `1e-3` and invalidated a run
  that had many qualifying post-ramp epochs. Acceptance now scans all Phase-C
  validation epochs for one satisfying both inequalities simultaneously and
  retains the latest qualifying epoch's checkpoint; the previously passing run
  selects the identical epoch under the new rule. A run failing checkpoint
  selection now writes its per-epoch validation history as retained failure
  evidence (the prior failure path discarded it with the staging directory).
  Success inequalities, the 2,000-step schedule, and all other gates are
  unchanged; no held-out/candidate/test quantity informed this correction.

- 2026-07-22: replaced the still-DRAFT V2 numerical contract before any successful
  qualification or held-out scoring. The attempt-005 rehearsal reached the end-ramp
  differential and exposed that the conditioned pair readout and final linear head
  still executed under BF16 autocast, despite the registered fp32-logit contract;
  this quantized the small `full - f_logit` residual. An exact four-H20 replay of the
  retained Stage-2 `V_fit` checkpoint measured an all-zero mixed residual before the
  fix. With the conditioned pair readout and logits in fp32, full/f-only elementwise
  checks pass, residual relative L2 is `0.0127555`, and correlation is `0.999752`.
  The untested `1e-3` residual bound is replaced by the pre-binding V_fit-calibrated
  `0.05` bound; elementwise tolerances, correlation `>=0.999`, and the non-zero rule
  are unchanged. No held-out/candidate/test quantity informed this correction. All
  failed attempts remain retained as engineering evidence, and this replacement has
  its own at-most-three full-arm qualification allowance because no prior V2 passed.
  The first four-H20 acceptance probe of the corrected island then measured
  `96.6 GiB` peak memory and `285.04 s` for the production-prefix epoch because
  autograd retained the fp32 readout activations. The training path therefore
  activation-checkpoints only that same fp32 readout and recomputes it during
  backward; evaluation remains direct. This changes neither forward values nor the
  registered numerical checks and must pass exact output/gradient regression tests.

- 2026-07-21: wired the registered V2 runtime `prefetch_factor=2` into the manual
  packed-token batch factory with one bounded deterministic producer thread per
  rank. Batch order, canonical overfit rows, padding, seeds, tensor values, losses,
  and optimizer steps are unchanged; `steady_state_data_wait_fraction` now measures
  only time the host training loop is blocked waiting for the next prefetched CPU
  batch. If artifact validation rejects a completed JSON-object worker profile, that
  parsed profile is retained with the failure evidence instead of being deleted with
  the staging directory. This is a qualification-infrastructure/in-memory-reuse fix
  under the still-DRAFT V2 amendment, not a scientific-contract change.

- 2026-07-21: corrected two additional consolidation trajectory regressions before
  binding. A fresh four-H20 check and a two-H20 control both showed generator clipping
  from the first Phase-A steps, ruling out world-size sharding; code/history comparison
  isolated removal of the parameter-free per-row F0 standardization used by the prior
  V2 overfit. Restored that exact stateless transform only inside the active V2 E2E
  generator, without adding optimizer parameters or changing historical frozen-s0
  checkpoint semantics. The one-epoch timing probe now evaluates the first epoch
  under the production schedule's `T` and phase boundaries instead of compressing
  A/B/C into the sampled epoch. Its runtime projection conservatively combines the
  measured prefix overhead with the selected candidate probe's full-joint compute
  throughput, and never projects below the measured prefix. The earlier same-date
  compressed-probe change is superseded; production losses, optimizer,
  2,000-step/30-epoch schedules, and guard thresholds remain unchanged.

- 2026-07-21: restored the registered V2 overfit trajectory semantics lost in
  consolidation: each optimizer step selects one canonical global 128-row window
  before rank sharding, and reconstruction-target randomness is keyed to fixed
  overfit epoch 0 plus the global optimizer step. The timing-only epoch probe still
  performs finite checks, clipping, gradient recording, post-step checks, and the
  immediate catastrophic-clipping abort, but the ten-step persistence abort is
  enforced only by the real 2,000-step/30-epoch execution, as in the qualifying V2
  implementation. Guard failures now retain the exact step,
  phase, norm, coefficient, streak, and recent records. Model, loss, optimizer,
  phase schedule, clip ceilings, and abort thresholds are unchanged.

- 2026-07-21: corrected the consolidated one-epoch runtime probe to preserve the
  registered Phase A/B/C order in its compressed representative schedule. The
  prior `profile_only` path forced Phase C from random initialization and caused
  the attempt-005 Stage-2 epoch probe to trip the generator-gradient guard; the
  production 2,000-step/30-epoch schedules and guard thresholds are unchanged.
  Superseded by the later 2026-07-21 production-prefix correction above after the
  compressed transition itself was shown to distort the gradient trajectory.

- 2026-07-21: corrected a consolidation-port omission that collapsed the
  registered V2 per-group clip ceilings from `3.0/3.0/1.0` to `1.0/1.0/1.0`.
  Restored `3.0` for pair-encoder/head and generator and retained `1.0` for
  topology/content conditioning; the `<1e-3` immediate and `<0.1` persistent
  abort thresholds are unchanged. This restores the settings executed by the
  attempt-003 V2 overfit and its pre-bound healthy-margin calibration.

- 2026-07-21: aligned the still-DRAFT v2 Stage-2 overfit launcher with Stage 3 and
  formal execution by auto-detecting and using every visible H20 (four on the current
  target host). The fixed manifest, 2,000 steps, phase schedule, objectives, and
  qualification thresholds are unchanged.

- 2026-07-21: after disclosed DRAFT rehearsal attempt 003 exhausted its remaining
  `1,823.6` s epoch-probe allowance on two H20s without completing an epoch or
  producing an eligible checkpoint/scientific result, amended the still-DRAFT v2
  qualification infrastructure before binding. The full-arm rehearsal changes from a fixed two-H20
  launch to all visible auto-detected H20s, matching the formal launcher shape on the
  current four-H20 host. Retained v1 artifacts attributed about 573 s per epoch to
  training plus unmeasured Python batch construction and 145 s to validation on two
  H20s; the old timer began after batch construction, so it cannot distinguish those
  costs. The v2 runtime pins the fastest measured two-H20 candidate, the H20-safe
  per-rank batch 128, removes redundant larger-batch probes, and records batch
  construction in timing telemetry; semantically equivalent reuse is protected by
  exact regression tests. Attempt 003 remains counted, the total budget and all
  scientific choices remain frozen, and the corrected four-H20 profile remains
  required. The registration discloses this post-attempt failure-recovery amendment;
  it does not relabel attempt 003 as prospectively corrected.

- 2026-07-19: clarified the prospective v2 internal topology selector as single-graph
  raw clustering `MMD^2` with pinned histogram/kernel hashes, not the external
  multi-reference normalized MMD ratio. This resolves an implementation ambiguity
  before binding and does not alter v1 or external scientific evaluation.

- 2026-07-19: added §13.19, the post-v1 E2E stability-screen contract. The
  completed v1 `full` training pipeline selected an ineligible reconstruction-only
  warm-start checkpoint and the later joint phase exhibited non-finite fixed-probe
  edge gradients and collapsed validation logits; no candidate-score artifact or G5
  gate result was produced. The replacement contract is prospective and versioned:
  dual-track pair/reconstruction warm-start, a linear conditioning ramp, balanced
  edge BCE, parameter-group learning-rate/clip boundaries, fp32 gated-residual
  accumulation, fail-fast finite/gradient guards, post-ramp checkpoint eligibility,
  a frozen train-side topology-aware selector, machine-verifiable binding evidence,
  a no-held-out-access audit, and a full-arm-first execution stop rule. The v1 BINDING
  registration and its artifacts remain unchanged; a v2 DRAFT registration must pass
  train/validation-only preflight and be separately bound before any new formal run.
- 2026-07-18: §10.1 pins packed-token E2E `B_e = 128` per rank after the Task-8 two-H20
  profile measured 63.86 GiB peak below the 85 GiB guard; the inherited
  `B_e = 512` deterministically exhausted a 95 GiB H20 and remains the
  frozen-s0/F0 default.
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
- 2026-07-16: added §14 (historical e2e stitched-topology-
  conditioned pair encoder, proposal §4.4 rev 3.0). §§1–13 remain the binding
  contract for the pending frozen-s0 screening run and its code; §14 records the
  successor architecture and its landing conditions. No §5/§13 semantics change in
  this edit.
- 2026-07-17: result-status closeout only. The replacement fixed-Seed-0 frozen-s0
  screen completed under registration `97e61a7d...` and returned binding verdict
  `cut`: all three primary dominance criteria failed and both guards passed. The
  locked disposition is frozen-s0 scalar fusion → motivating arm + ablation, and
  rev-3.0 e2e conditioning → then-active G5 build line. This satisfies §14.3(1) but does
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
  three-clause definition (own gate `> 0.5`; pointer-argmax equality across
  endpoints within a shared node-identity space that need only be consistent
  within one run, not a persistent global id; own-split grounding pools only,
  no Hungarian/target-graph access) and superseded the implementation plan's
  Hungarian/`L_gate`-matching phrasing for this label: that quantity is
  train-time-only (node-stream batches with real ego-net targets) and
  undefined for edge-stream endpoints and unseen nodes at inference, so
  wiring it there would violate the inductive protocol.
- 2026-07-17: final whole-branch audit closure for the DRAFT E2E screen. Pinned
  the dual-pack cold/warm contract; exact formal score-arm and candidate-row
  provenance; single-ego self-pair execution; the per-slot former-s1 membership
  channel in `c_content`; shared-state four-logit scoring; the global validation
  slice; fixed-replay submodule-gradient telemetry; and the deterministic,
  provenance-bound nonbinding probe artifact. These details close implementation
  gaps without changing any Stage-1 verdict inequality or binding the registration.
- 2026-07-17: final adversarial closure for the DRAFT E2E screen. Formal worker
  preflight now rejects any registration that still contains a
  `REQUIRED-BEFORE-BINDING` marker; the probe producer is bound specifically to
  the registered full-arm config and `p_topo = p_cont = 0.15`; and cached
  four-logit scoring is required to match the uncached production decomposition
  on mixed-length/self-pair batches. No verdict inequality or registration status
  changes in this edit.

**Closed gate-report deliverable (2026-07-17):** the frozen-s0 Stage-1 gate report,
fidelity diagnostics, and measured FLOPs/latency report are complete. The binding
verdict is `cut`; see `docs/results/G5-stage1-seed0-20260717.md`. Closed 2026-07-14: the four
ego-stat target definitions (§13.6); Benchmark-A/B/C ↔ split-strategy mapping
(confirmed at G1: `Benchmark-A = breadth_first`, §9.1).

## 13. G5 Stage-1 carve-out (normative for the Stage-1 build)

**Scope note (2026-07-29).** §13.1–13.17 govern the completed, published
frozen-s0 screen only (`docs/results/G5-stage1-seed0-20260717.md`, binding
verdict `cut`). Their implementing code — the frozen-s0 `egostitch` family,
its Stage-1 training/DDP path, and `frozen_s0` scoring mode — is retired from
the codebase under the two-stage cleanup (§12, 2026-07-29 entry). This
section remains citable and comprehensible as the record of what that screen
ran; no future run may cite it as authorization. Its 80/20
message/supervision-partition data contract is retired and its results are not
comparable to the shared-training-edge contract adopted 2026-08-03.

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
L_ssl   unchanged (§7): 0.5·consistency(feature noise σ=0.05, in standardized
        coordinates for the E2E family — §13.19.1)
        + 0.5·pool-resample consistency, ungrounded slots only
```

Under `zscore_vfit_v1` the SSL feature perturbation is sampled in standardized
coordinates (equivalently: sampled in raw coordinates and scaled by the registered
per-dimension `sigma` before addition). Adding a raw σ = 0.05 perturbation to raw F0
would be a 5e-5 … 1.9e-3 perturbation per standardized coordinate, since the measured
per-dimension F0 σ spans 26.1 … 1023.3 — the augmentation would silently vanish.

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
on the fixed global validation slice: the first
`max(1, ceil(0.01 * N_val))` rows in the frozen validation-manifest order. Every
rank/world-size configuration evaluates those exact identities; local rank shards
or edge-batch boundaries may not choose the slice. The larger value wins (liveness-preferring, same
direction as the retired residual/s0 rule, which remains the recorded rule for the
historical frozen-s0 family). This fidelity tie-break is checkpoint selection only
and is not a Stage-1 success criterion.

For the replacement E2E stability screen, §13.19 supersedes this two-phase
schedule and checkpoint-eligibility rule. Historical v1 artifacts remain interpreted
under the rule above; the change is not retroactive.

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

**v2 §13.19 isolation:** the grounding universe is role-specific and separately
hashed: training=`V_fit`, validation=`V_hold`, and external scoring=the original
test side. A cache for one
universe may not serve another. In particular, a training pass may not read or
encode a `V_hold` row.

**Rev-3.1 (2026-07-25, §14.4.4):** for the rev-3.1 build, `n_g = 50` (the
measured P0.2 ceiling curve; the reranker alternative was measured and
rejected) and the cache manifest binds `pool_method_hash` = H(method id,
`n_g`, shortlist M when present, ordered F0/source-feature-pack digest,
role-universe identity); loaders fail closed on any mismatch. `n_g = 20`
remains the recorded contract of the completed v2 screen.

### 13.13 Runtime budget

§11's 60-minute cold-run pin applies to the formal E2 B0 V3.1 run only. The
EgoStitch worker uses the same automatically detected H20 count and
`accelerate launch --num_processes N` layout, the same runtime-profile and checkpoint payload schemas, and a
**config-driven** budget. `runtime.total_budget_seconds` is retained only as the
sum invariant for the remaining stage budgets; it does not authorize a projection
or probe sub-stage.
This family's generic pack stage has **two required, independently cold/warm
validated packs**: (1) the F0 pooled matrix + grounding-pool cache at
`runtime.pack_dir`; and (2) the raw-token BF16 pack at `data.pack_dir`. A cold run
builds either missing pack before training; a warm run validates both against the
same frozen source feature manifests. Both manifest payloads and both manifest
SHA-256 identities are embedded in pipeline/run evidence. A worker may not reach
training with only one pack present. The production orchestrator is
`pack → train → publish`; projection is deleted. The retained
`probe`/`epoch-probe` dispatch entries may use validated packs only to measure
pre-run `feature_stats_sha256`; they publish no checkpoint or formal artifact and
are not orchestrator stages.

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
pass may stay BF16 with fp32-cached outputs, unchanged. Candidate scoring encodes
each unique node once, caches the raw-token encoder state and generated ego state as
fp32, builds Stitch/STE/content pair context once per pair batch, and evaluates the
four hard-bypass heads from that shared context; four complete generator/Stitch/STE
forwards are prohibited. The cached path must reproduce an uncached production
`decompose` pass within explicit fp32 tolerance for mixed token lengths and self
pairs. Artifacts for this family
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
`grad_rms_content`, and a per-epoch `topology_delta_std` on §13.8's fixed global
validation slice. The three gradient-RMS values are measured from the fixed
post-warm-start replay batch and its weighted `L_edge` tensor inside the same
`no_sync` retained-graph probe as the family norms, RMS-aggregated across ranks,
persisted in metrics/run metadata, and required (numeric, at least one probe row)
by the formal gate. The channel-scale series of this section read the available quantities
(`f_logit`, pathway deltas) in place of `s1/s2/residual-vs-s0`.

**Historical frozen-s0 probe provenance (retired; not an active implementation
contract):** the required nonbinding representation evidence was a deterministic
`egostitch_e2e_probe_v1` artifact generated from the selected full checkpoint
after scoring and consumed by the gate. The producer accepts only the registered
`arms.full.training` config, with `permanent_null = none` and the full-arm
`p_topo = p_cont = 0.15`; the p0 arm is invalid even though it also has no
permanent null. It is bound to checkpoint id, registration SHA-256, config hash,
partition Seed 0 and full-`E_msg` `G_struct`. Node
rows are every operative train node in sorted id order; pair rows are the 4,096
non-self `E_msg` pairs with smallest `sha256("min(u,v)|max(u,v)")` (or all rows
when fewer exist). It carries mean-pooled STE states and evaluator targets for
degree, ego density, and clustering. The gate reports five-fold ridge R2
(`lambda=1e-3`) for all three, degree-partialled R2 for ego density and
clustering, and Pi/shared-neighbor consistency: alignment-plan mass landing on
equal grounded identities that are real common neighbors, divided by total
plan mass (mean/std and nonzero fraction). These diagnostics never change the
registered verdict.

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

For the replacement E2E stability screen, §13.19 strengthens these diagnostics into
pre-scoring run-validity gates. In particular, non-finite family/probe gradients are
fatal and may not be converted into JSON `null`, ignored, or treated as a successful
Kendall-fallback activation.


### 13.18 Historical E2E family pins (retired)

> **Retired contract.** Everything in §13.18–§13.19 records the completed
> registration-era topology+content screens. It must not configure a current run.
> The active topology-only, direct-run contract is §14.1–§14.4.7; it has five
> trained arms plus two scoring controls and no registration or plan-identity gate.

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

**Formal score-arm and candidate bindings:** every E2E gate artifact must match
the frozen `candidate_test_edges.txt` pair and label arrays row-for-row, not only
its count. Exact scoring semantics are: `full` = control `none`, permanent null
`none`, primary `full`; `structure_control_6a` = `shuffle_within_pair`, seed 0,
`canonical_pair_v1`, permanent null `none`, primary `full`, and the full
checkpoint; `b0_e2e_f_only` = control `none`, permanent null `all_head`, primary
`f_logit`; `pair_topology` = control `none`, permanent null `content_head`,
primary `pair_topology`; `p0` = control `none`, permanent null `none`, primary
`full`. The gate fails closed on any field, checkpoint, row, or label mutation.

**Registration and artifact identity (machine-checked):** the worker snapshots
and hashes the referenced registration before DDP work; its status field is
informational and its nullable evidence slots are not preflight inputs. The
registration SHA-256 must match the `preregistration_sha256` of every consumed
formal run metadata and score artifact. Exact arm/config/frozen-input identity,
clean checkout, checkpoint and score digests, data-boundary integrity, and
append-only test access remain fail-closed. Implementation, parameter-group,
pack/validation, boundary-audit, runtime/peak-memory, and checkpoint-policy
evidence is produced by the run and validated from run metadata/artifacts, never
required to be non-null in the registration before execution. `--max-steps`
debug runs remain redirected to `*_debug` output directories and may not publish
held-out artifacts. Registration amendments use a new versioned file so existing
artifact hashes remain interpretable; this versioning rule is independent of the
descriptive status field.

**Grounded-identity-match flag (e2e content tokens, pinned 2026-07-17):** for a
pair `(i, j)`, slot `k` of endpoint `u ∈ {i, j}` is grounded-identity-matched
**iff** (1) `gate_u^k > 0.5` (the slot is grounded — the same 0.5 threshold §4's
grounded-slot re-mask rule uses); AND (2) `argmax` of slot `k`'s grounding
pointer selects a grounding-pool candidate with identity `c`, drawn from a
shared node-identity space that is consistent across both endpoints of a batch
within one run; AND (3) the **other** endpoint has at least one slot `k'` with
`gate > 0.5` whose pointer argmax selects that **same** identity `c` within
that shared space. This shared node-identity space is an abstract requirement
of the definition, not a claim of a persistent global id: the scoring-path
realization (`src/score_universe.py`'s `_score_egostitch_e2e`) is the
run-scoped F0-matrix row index (the `pool_rows` convention), rebuilt fresh on
every scoring invocation — these row indices are **not** comparable across
runs or across the train/score boundary, and the definition requires only
within-call (within-batch-construction) consistency. The flag is a binary float
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

### 13.19 Single-stage plan-bound E2E execution

The active EgoStitch E2E experiment has one formal training stage. A formal run is
authorized only by the owner-bound experiment plan and the exact identities of its
implementation, arm config, data universes, packs/manifests, checkpoint, score
artifacts, candidate manifest, and test-access ledger. There is no qualification
stage, qualification artifact, pass/pending verdict, history disclosure, or
qualification-to-formal authorization path.

Finite model-quality conditions—including checkpoint eligibility, validation
liveness, slot-collapse indicators, clipping/family/submodule-RMS margins, AUPRC,
dispersion, and precision differentials—are telemetry only. They may be reported
and analyzed but may not stop training, suppress checkpoint publication, prevent
scoring/evaluation, or alter experiment authorization. Checkpoint selection is the
registered plan rule applied to all completed epochs.

Truthfulness and artifact-integrity failures still fail closed: non-finite tensors
or optimizer state, DDP disagreement, incomplete or duplicate coverage, train/hold/test
boundary violations, malformed or hash-mismatched plan inputs or outputs, checkpoint
or score provenance mismatch, and I/O or infrastructure failure.

`V_fit` is the training universe and `V_hold` is the sole validation/model-selection
universe. Candidate/test data remain unread during training. The formal result must
record the plan SHA-256, live implementation/config identities, V_hold evaluation
ledger, selected-checkpoint identity and digest, score-artifact provenance, and
append-only test-access ledger.

Registration status is descriptive and is not an execution state machine. A
`DRAFT`, `BINDING`, or other status value cannot authorize or block training,
scoring, or evaluation. Likewise, nullable registration evidence fields are
forward declarations of provenance to be produced by the run, not preconditions.
Downstream consumers validate the actual run-produced implementation,
parameter-group, pack/validation, boundary-audit, runtime/peak-memory, and
checkpoint-policy provenance together with exact registration/config/checkpoint/
artifact identity. The registration file itself remains immutable during a run
and its SHA-256 remains part of every artifact chain.

## 14. Historical successor: E2E stitched-topology-conditioned pair encoder (2026-07-16)

**Scope and precedence.** This section records the historical rev-3.0
architecture (proposal §4.4; full decision trail and pins in
`docs/superpowers/specs/2026-07-16-egostitch-e2e-conditioned-encoder-design.md`).
**§§1–13 remain the historical binding contract** for the completed frozen-s0
Stage-1 screen and its retained implementation on `main`. That screen published a
binding `cut` verdict on 2026-07-17, satisfying §14.3(1). **§5/§13 were rewritten to §14 on
2026-07-17** (change-log entry above): §14 plus the rewritten §5/§13 are now the
normative implementation contract for family `egostitch_e2e`, as amended by the
2026-08-02 three-component refactor and §12 change log. Current e2e runs execute
directly without a registration or plan-identity gate.

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
inject: in the last N_inj ∈ {1, 2} pair-cross-attention blocks (default 1), the CLS
        token cross-attends to c_topo through a zero-initialized tanh gate, per
        direction (AB / BA, swapped anchor labels) BEFORE abba_max;
        AB/BA share STE and attention parameters; branch masks are per pair, shared
        across directions
p_ij  = σ(head(z'_pair))
```

Scaffold construction fails closed unless both sides have equal slot counts, `Π`
is exactly `(B,K,K)`, and all tensors share the same batch size.

### 14.2 Topology null (checkpoint-exact)

| Null | Skips | Yields |
|---|---|---|
| `∅_all_head` | generator/scaffold topology conditioning | pair-only `f_logit` |

Training realizes the null as a per-pair multiplicative mask (`p_topo = 0.15`
default, plus the `p0` arm); evaluation uses
batch-level hard bypasses; residual sublayer form makes the two numerically identical
(required unit test), as is `p(i,j) = p(j,i)` under every null. Disclosed
2026-07-23: the frozen V2 mask realization derives its dropout randomness
without the DDP rank, so per-step mask draws are correlated across ranks
(identical per local batch index); expected rates are unchanged. The `_head`
namespace is disjoint from the §2 decoder nulls.
Both logits (full + `f_logit`) are published per scored pair; the §13.16 fp32
pair-pass pin extends to trunk, STE, gates, and head
(`egostitch_e2e_pair_fp32_v1`); §13.17 liveness signals re-register against the
within-checkpoint `f_logit` (no frozen-s0 comparator artifact); the §13.10 s0 logit
cache is retired for this family.

### 14.3 Implementation and experiment-plan record

The items below document the architecture and experiment-plan lineage. They are
not a registration-status or evidence-completeness preflight. Current runs execute
directly through `hpc/run.sh train`. Runtime provenance is recorded in the produced
checkpoint and artifacts; training-*data* provenance is not (§12, 2026-08-03).

1. Frozen-s0 screen published (its outcome is the successor's motivating arm).
   **Satisfied 2026-07-17:** binding verdict `cut`; result note
   `docs/results/G5-stage1-seed0-20260717.md`.
2. §5/§13 rewritten to §14 with change-log lines; §13.18-style pins for defaults
   (`ste_layers = 3`, `ste_dim = 128`, `xattn_heads = 8`, `n_inj` default 1 sweep
   {1, 2}).
   **Satisfied 2026-07-17:** §5/§13 rewritten (change-log entry); §13.18 landed
   with the defaults, `permanent_null` overrides, and `shuffle_within_pair` control.
3. **Historical registration-era requirement, retired 2026-08-03:** fresh v4
   registration with the eight-arm scope then present in §14.4.6 (six trained
   checkpoints plus two scoring-time controls), the four-logit decomposition
   report, the representation-probe protocol (degree / ego density / clustering +
   degree-partialled + Π-consistency, frozen-encoder linear probes on held-out
   then-named message-partition nodes), the pathway-attribution decision rule, and a measured
   H20 cost re-estimate (the 673 s / 2.04 GiB frozen-s0 Stage-1 profile does not
   extrapolate; budget class is the E2 B0 run). The v4 registration status is
   descriptive; runtime and peak-memory evidence are emitted by the formal run.
4. Implementation plan:
   `docs/superpowers/plans/2026-07-16-egostitch-e2e-conditioned-encoder.md`
   (B0.py untouched; conditioned trunk subclass; train-mask ≡ eval-bypass and
   symmetry tests as acceptance criteria).

### 14.4 Rev-3.1 relational repair (2026-07-25; normative for the next e2e build)

**Provenance and scope.** Decisions were owner-delegated on 2026-07-25 and are
recorded, with full rationale and the four-round review trail, in
`docs/superpowers/specs/2026-07-25-egostitch-e2e-relational-repair-design.md`
(r5). Phase-0 audit evidence is archived under `outputs/p0_audit_20260725/`
(local P0.2/P0.4; container P0.1/P0.3). This section **authorizes
implementation only**; execution records the exact v4 registration bytes and
SHA-256 but does not gate on its status or nullable run-produced evidence. §13.19 remains
the completed v2 screen's binding record; §14.4 supersedes it for rev-3.1
only where stated.
All generated/stitched structure remains intermediate context for the binary
edge decision (`literature/research_reports/2026-07-07-topology-conditioned-inductive-edge-prediction-lit-review-plan.md` §5).

#### 14.4.1 Loss components (fold into `L_recon`; outer objective unchanged)

`L_recon` (rev-3.1, family `egostitch_e2e`) `= 1.0·L_feat + 0.5·L_exist +
0.25·L_mult + 0.5·L_deg + 0.5·L_slotadj + 0.25·L_gate + 0.25·L_ptr +
0.5·L_align + 0.1·L_div + 0.25·L_rel`.

- **`L_ptr`**: `CE(pointer_k, pool_index(t_k))` over Hungarian-matched slots
  whose target is in the pool; all other slots masked.
- **`L_align`** (edge-stream positive pairs only): teacher cells
  `S = {(k, l) : target_id_a(match(k)) = target_id_b(match(l))}` from the two
  endpoints' Hungarian matchings (K-capped; P0.4 measured `P(|S|>0) = 0.663`,
  `E[|S|] = 2.12` — the uncapped fallback is not used);
  `L_align = −(1/|S|) Σ_{(k,l)∈S} ½·[log(Π[k,l]/Σ_{l'}Π[k,l']) +
  log(Π[k,l]/Σ_{k'}Π[k',l])]`; pairs with `S = ∅` skipped. Row/column-
  conditional only — the Sinkhorn marginals are not a gradient target.
- **`L_div`**: `mean max(0, cos(h_k, h_l) − τ_div)²` over slot pairs with at
  most one Hungarian-matched slot; `τ_div = 0.5` initial.
- **`L_rel`**: a 2-layer head on the STE AB-direction pair state (mean over
  scaffold tokens) predicts `log1p(common-neighbor count)` and neighborhood
  Jaccard, computed from the shared training topology **independently for every
  pair, positive and negative** (a sampled non-edge with common neighbors
  receives its true nonzero targets — required regression motif). Huber loss.
  The head is excluded from every scored logit; `L_rel` is its own telemetry
  family; its removal is the formal `no_l_rel` arm.
- **`L_gate`**: pos-weighted BCE; weight is a registered constant from the
  measured in-pool rate.
- **Anneal**: per-component factors 1.0 → 0.25 across the edge-active phase
  for `{L_feat, L_exist, L_mult, L_deg}` only; every other component stays at
  weight 1.0 throughout. Outer `λ_recon` unchanged.
- **Reduction contract** (§13.19.1-class): `L_rel`/`L_align` are
  `edge_mask`-weighted (for `L_align`: positive-and-real-row-weighted) global
  means with an all-reduced real-row denominator. Required test:
  per-parameter gradient and loss equality at world sizes 1 vs 2 with unequal
  tails and padded filler rows.
- **Target sampling**: edge-stream ego-target subsets are keyed
  `rng(blake2b(node_id) ⊕ (seed, epoch))` — never rank, step, or pair — so a
  node's target subset is epoch-fixed across pairs, ranks, and directions.
  `L_rel` targets use no RNG. The invariance contract is per-pair; negative
  stream composition remains `(seed, epoch, rank)`-drawn as in v2.

#### 14.4.2 Historical rev-3.1 architecture deltas (superseded)

> **Superseded by the 2026-08-02 three-component refactor.** §§14.4.2–14.4.5
> retain the rev-3.1 lineage only; content-path, matched-content, `p_cont`, and
> registration clauses below are not active implementation contracts.

- **Soft matched flags**: with shared-id indicator `I[g_a, g_b]`,
  `M = p_a I p_b^T`, `matched_a[k] = gate_a[k] · max_l(M[k,l] · gate_b[l])`
  (BA uses `M^T`). Train and eval use the identical soft form; a required
  test asserts gradient reaches `head_pointer`.
- **`SlotSet` exposes `adj_logits`**; `L_slotadj` becomes `BCEWithLogits` on
  `adj_logits / τ_adj` with registered `τ_adj < 1`.
- **Scaffold**: FEAT_DIM 9 → **11** — new per-slot closed-wedge feature
  `t_k = [Π Â̊_other Π^T]_{kk}` on zero-diagonal `Â̊ = Â − diag(Â)`.
  EDGE_TYPES 3 → 4 — closure block `C = ½(Â̊_src Π + Π Â̊_dst)` at
  `adj[CLOSE, s_src, s_dst] = C` (transpose opposite). Channel layout:
  `[onehot4(anchor); π; mult; deg×EDGE_TYPES; t_k]` — the degree slice is
  mechanically one channel per edge type (`adj.sum(-1)`,
  `scaffold.py:114`), so it widens 6:9 → **6:10** and `t_k` occupies index
  **10**. (FEAT_DIM 10 was a miscount: the widened degree slice alone
  consumes the tenth channel, leaving no index for `t_k`. Owner-confirmed
  2026-07-25 — see the §12 third entry.) Required test:
  **rebuild-symmetry** — the `(j, i)` scaffold built with `Π^T` equals the
  side-permuted `(i, j)` scaffold.
- **Centered gated conditioning** (supersedes the §14.1 inject equation for
  rev-3.1): `cls ← cls + active · tanh(g) · (XAttn(...) − μ)`; μ is the mean
  over pathway-active, real (`edge_mask = 1`) rows, all-reduced across ranks;
  eval uses a synchronized EMA stored in the checkpoint. Inactive rows keep
  the exact-identity bypass, preserving the §14.2 checkpoint-exact null
  taxonomy.

#### 14.4.3 Curriculum

Warm-start remains reconstruction-only (no `L_edge`). From the **first
edge-active step**, the trunk, STE, gates, and both conditioning pathways
train jointly — the v2 Phase-A `pair_only` head start is removed (supersedes
§13.19.1 for rev-3.1). Branch dropout `p_topo = p_cont = 0.15` unchanged.

**Checkpoint-eligibility reference (owner-confirmed 2026-07-26; §12 sixth
entry).** The §13.19.2 warm-reference AUPRC requirement — exceeding label
prevalence by **at least** `0.02`, i.e. `>= prevalence + 0.02`, exactly as
§13.19.2 item 3 states it — is retained unchanged in threshold **and in its
inequality**, but its snapshot moves to the
**first validation after conditioning activates**. Under the joint-entry
curriculum Phase A disables `L_edge` and freezes the pair encoder, so a
Phase-A snapshot measures an *untrained* pair head and would make an
otherwise successful run's eligibility depend on initialization luck. The
guard's intent — catching a pair head that never learns — is preserved by
reading it where the head has actually trained.

#### 14.4.4 Grounding (supersedes the §13.12 value for this family)

`n_g = 50`, exact top-`n_g` cosine within the node's §13.12 role universe.
No reranker. The former P0.2 curve (`e_sup` pair ceilings 0.095 / 0.134 /
0.179 for cosine top-20/50/100; B0-alt rerank ≈ cosine at 2.5× pool size) was
measured under the retired 80/20 contract. It remains historical evidence for
the earlier design decision but is **non-binding and non-comparable** under the
shared-edge contract; the current grounding ceiling must be remeasured
before it supports any new claim.
Pool caches bind `pool_method_hash = H(method id, n_g, shortlist M when
present, ordered F0/source-feature-pack digest, role-universe identity)`;
loaders fail closed on any mismatch (regression tests: stale-method and
mutated-features-same-ids).

#### 14.4.5 Structure controls

- **6a-v3**: within-pair slot-axis permutations of `Â_src`, `Â_dst`, `Π`
  applied at scaffold-build input (v2 blake2b canonical-pair keying), then a
  full scaffold rebuild so every derived channel (incl. `t_k`, `C`, 6:10)
  recomputes from the shuffled structure.
- **6e-v1**: canonical-pair-keyed checkerboard swaps — `N_swap = 8·K²` keyed
  draws; each selects rows `(i, k)` and columns `(j, l)` and transfers
  `δ = u · min(w_il, w_kj, c_ij − w_ij, c_kl − w_kl)` (keyed `u ∈ (0,1)`):
  `w_ij, w_kl += δ`; `w_il, w_kj −= δ`. Draws are restricted to
  **off-diagonal** cells, so the zero-diagonal contract of `Â̊` survives the
  perturbation.
  The last two terms are the **recipient-capacity cap** (owner-flagged
  2026-07-25; see the §12 fifth entry): in the π-weighted space below, cell
  `(i,j)` has capacity `c_ij = π_i π_j`, because `Â̊ ∈ [0,1]` by §2. Without
  the cap, mapping back `Â̊' = W'/(π_i π_j)` can push a cell far above 1
  whenever the recipient's `π_i π_j` capacity is smaller than the donor's,
  and `build_scaffold` then propagates that out-of-domain value into `CLOSE`
  and `t_k` — so the 6e arm would be driven by scale explosion rather than by
  rewiring beyond degree. Every marginal is preserved for *any* `δ` (the
  checkerboard pattern is balanced), so the cap costs nothing.
  **Swap space (owner-confirmed 2026-07-25; see the §12 fourth entry).** The
  swaps are applied to the **π-weighted** slot-adjacency `W = π_k · Â̊[k,l] ·
  π_l` (for both sides, symmetrized), mapped back as `Â̊' = W' / (π_k π_l)`,
  and to `Π` directly. The binding invariant is **model-visible degree**: the
  rebuilt scaffold's `STAR`, `INTRA`, and `ALIGN` degree channels are
  preserved to fp32 tolerance. Swapping raw `Â̊` instead preserves only the
  pre-rebuild marginals — `build_scaffold` forms INTRA as `Â̊ ⊙ π π^T`, so
  under nonuniform `π` the rebuilt INTRA degree drifts (measured max 0.383)
  and degree leaks into a control whose registered purpose is isolating
  structure *beyond* degree.
  The `CLOSE` degree channel is **expected to move** and is not an invariant:
  closure mass is higher-order by construction and is precisely the structure
  6e exists to destroy. Preserving every rebuilt channel simultaneously is
  over-constrained (INTRA needs π-weighted marginals; CLOSURE would need
  `r`- and `d`-weighted marginals at the same time).
  Then scaffold rebuild. Required tests: preservation of the rebuilt
  `STAR`/`INTRA`/`ALIGN` degree channels to fp32 tolerance; zero diagonal
  retained; cross-process determinism; non-inertness on a random
  non-collapsed model.

#### 14.4.6 Current screen schema (seven arms)

Five trained checkpoints — `full`, `b0_e2e_f_only`, `p0`, `no_l_rel`, and
`row_layernorm` — plus the two scoring-time controls `6a-v3` and `6e-v1` over
`full`'s checkpoint. The removed content pathway makes `pair_topology`
identical to `full`, so that arm is retired. Checkpoints and score artifacts
carry **no training-data identity field** (§12, 2026-08-03): nothing in the code
distinguishes an artifact trained under the shared-edge contract from an
older 80/20-partition one, so keeping the two apart is an owner-side
responsibility discharged by run directory and commit, not by a gate.

#### 14.4.7 Probes and single-stage execution (extends §13.19)

Probes: **Π-consistency v2** = plan mass on double-Hungarian same-identity
cells / total plan mass (pool-independent; v1 retained with its honest
grounding-chain scope); per-run slot recall@`n_g`; shared-neighbor-count R²
from STE pair states; and the four slot-dispersion statistics (π std, mean
pairwise `h` cosine, `Â` off-diagonal std, `Π` row entropy). These are output
telemetry at each formal validation. They do not authorize or block training,
checkpoint publication, scoring, evaluation, or plan binding.

The formal run executes directly on `V_fit` and validates on `V_hold`. It uses
the same complete `V_fit` training edges for topology and
classification, with per-query leave-one-out only when constructing the target
for that queried positive. There is no preregistration or artifact-identity
gate, and nothing in the checkpoint or scoring path enforces the §9.3
shared-edge data contract — it holds by construction in
`derive_training_interactions`, and stale artifacts are excluded by hand (§12,
2026-08-03).

#### 14.4.8 Telemetry and abort rules

- **Collapse threshold** (§13.19.2-class): mean pairwise `h` cosine > 0.95, or
  the Π rank-1 marginal residual `‖Π − r c^T / m‖_F / ‖Π‖_F < 0.05` (with
  `r`/`c` Π's row/column sums and `m` its total mass — "the plan carries
  nothing beyond its marginals"), for 2 consecutive validations after
  conditioning activates is reported as slot-collapse telemetry only.
  Calibration basis (P0.1, measured on the collapsed v2 checkpoint
  `a471010f57e495f0`): mean pairwise `h` cosine `0.9997`; Â off-diagonal
  `0.5014 ± 0.0004` (constant); π std across slots `0.042`. A Π
  **row-entropy** criterion was measured to be blind to full collapse
  (normalized row entropy `0.624`, because the `π·m` marginals concentrate
  rows regardless of cost-blindness) and is therefore not used.
- **Degree-decorrelation telemetry**: per-validation correlation of
  `full − f_logit` with endpoint degree, reported in every headline table
  (telemetry-only, no verdict effect).

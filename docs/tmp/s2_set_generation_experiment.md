# S2 — Set-Conditioned Joint Latent Topology Generation (feasibility diagnostic)

**Question.** Given only the intrinsic features $X_S$ of an unseen protein set $S$, can a
generative model $p_\theta(Z_{1:n}\mid X_S)$ produce a *meaningful* joint latent topology?
Meaningful = node-aligned (primary), per the grill session: the generated structure must
attach to the right nodes, not merely have realistic statistics.

**Status.** Exploratory diagnostic, `evidence_class=diagnostic`, `formal:false`
(test-informed). No binding kill rule; arms are constructed so each mechanism is
separable after the fact. Boundary decisions fixed in advance:

- Primary endpoint: node-aligned identification; distributional realism secondary.
- Density: matched to the true edge count for primary metrics; free-running implied
  density reported separately.
- Latent: $Z_i=(a_i, z_i\in\mathbb R^{16})$, decoder $\ell_{ij}=a_i+a_j+z_i^\top z_j+c$;
  activity-only decoder as internal ablation.
- Unseen set: the held-out **test side** (user-authorized). Train on `train_graph`
  (train⁺∪val⁺, V_val-internal edges included); evaluate on `test_node_buckets.pkl`
  (10 sizes × 50 node sets) and optionally the full 2,018-node test region. Test nodes
  are node-disjoint from training by benchmark construction.
- Inference reads only $X_S$: no test edges, no test degrees, no hidden edge count.

## Data

- Training corpus: BFS node sets sampled from `train_graph` with the same
  construction rule as the test buckets, sizes on the same 20–200 grid (~50k regions).
  Each example: $(X_S\in\mathbb R^{n\times1536},\,A_S)$ with $A_S$ the loopless induced
  subgraph. 5% of regions held out for early stopping / model selection; test touched once.
- Self-loops: stripped in training targets (repo convention); evaluation reuses the
  bucket reference machinery in `src/eval/graph_metrics.py` with its own conventions.

## Model

- **Stage A — graph AE (defines latent semantics + expressibility ceiling).**
  Encoder: small message passing over $(G[S], X_S)$ → per-node $Z_i$. Decoder: the
  inner-product form above. Loss: all-pairs edge BCE + soft-triangle moment penalty.
  Its reconstruction quality is the **latent ceiling** — if the AE cannot represent
  region topology through this bottleneck, features were never testable (avoids the
  S0 weak-proxy trap).
- **Stage B — conditional set prior.** Few-step flow matching over $Z_{1:n}$ given
  $X_S$; permutation-equivariant set-transformer denoiser, no positional encodings
  (each $z_i$ attends to the whole set — this is where jointness lives).
- **Deterministic twin.** Same backbone trained as a regressor $X_S\to Z$
  (latent MSE + decoded-edge BCE). Isolates generation vs set-encoding.

## Arms

| Arm | What it is | Comparison isolates |
|---|---|---|
| GEN | $Z\sim p_\theta(Z\mid X_S)$, K=32 draws | the route itself |
| DET | deterministic twin, one prediction | generation vs set encoding |
| UNC | unconditional prior (features dropped) | what $X_S$ adds at all |
| SHUF | conditional, within-set feature permutation at eval | node-feature correspondence |
| MARG | per-node degree regressor $f(x_i)$ + frozen B0 pair scores | jointness vs independent prediction |
| AE | encoder fed the TRUE $G[S]$ | latent expressibility ceiling |
| GEN/DET-act | activity-only decoder | degree field vs role geometry |

## Readouts (battery, all reported)

1. **Node-aligned:** per-set degree Spearman and top-10% hub recall (implied soft
   degree vs true); edge-set Dice/GS at matched density; in-set pair AUPRC
   (GEN uses across-draw edge marginals).
2. **Distributional:** degree / clustering / spectral MMD of generated graphs
   against the precomputed bucket references.
3. **Density:** free-running implied edge count vs true (secondary).
4. **Coherence-specific:** across-draw variance of $a_i$ (per-node posterior
   sharpness); draw-level joint statistics vs independently-thresholded marginals —
   the node-cached sampling mechanism S1 could never test.
5. **Statistics:** macro mean ± set-level bootstrap CIs per bucket size; alignment
   vs set size curves.

## Interpretive map (not a gate)

GEN>SHUF/UNC → $X_S$ carries joint-topology signal. GEN>MARG → set-level joint
modeling adds over independent prediction. GEN>DET → sampling adds over deterministic
set encoding. GEN vs AE → fraction of expressible topology recoverable from features.
act vs full → degree vs role contributions.

## Cost

AE ~1M params, prior ~5–10M; regions ≤200 nodes so $O(n^2)$ attention is trivial;
hours on one H20 GPU, schedulable after the KD chain. B0 bucket-pair scores come from
the existing score-once test artifacts.

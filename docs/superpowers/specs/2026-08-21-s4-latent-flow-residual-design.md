# S4 — Set-Conditioned Graph-Latent Flow Residual

**Question.** On S2's 20–200-node size/test substrate with S3's V_val boundary, can the strongest S2-style
joint generator add useful topology context when it augments, rather than replaces, frozen B0?
S4 is one model, not an arm sweep: a feature-conditioned Set Transformer rectified flow generates
one node-cached graph latent per set draw, and a zero-initialized adapter adds it to B0's
pre-head `pair_repr`. Existing frozen B0 and completed S3 results are comparators, not new runs.
**Status:** planned diagnostic, `formal:false`; the method was selected after inspecting the fixed
S2/S3 test panel, although access to every sampled set's `X_S` is part of the legal task setting.

## Claim and decision boundary

The primary claim is route-level: the complete generated-latent residual improves prediction beyond
the frozen endpoint pair representation. Convincing evidence requires paired improvement over B0
and the completed S3 PAIR control on both in-set AUPRC and density-matched GS, with no material
degradation of RD or the three MMD ratios. A healthy but non-improving model closes this exact
GAE-latent-flow residual route; it does not rule out every topology-conditioned architecture.

## Data and split alignment with S2/S3

- Set sizes are exactly `20,40,...,200`. Training uses BFS-ball regions sampled from
  `train_graph.pkl`'s loopless giant component with the S2 corpus rule and volume. That graph defines
  region membership only: all V_val-internal edges are physically removed before constructing GAE
  message-passing inputs, structural targets, or pair losses; cross-boundary edges remain train-side.
- Validation uses the fixed V_val node region and the same size grid. It selects checkpoints from
  in-set edge and five topology metrics, never from test buckets.
- Test uses the existing `test_node_buckets.pkl` substrate: 10 sizes × 50 sets, touched once after
  model selection. There is no full-universe secondary evaluation in S4.
- For every sampled set, the model may read all intrinsic features `X_S`. Train-side adjacency is
  supervision; V_val-internal adjacency is validation-only; test adjacency is evaluation-only. No graph
  latent, cache, threshold, or normalization state crosses the train/V_val/test universes.
- Structural targets and pair losses exclude self-loops. Evaluation keeps the established S2/S3
  bucket conventions so paired comparisons remain meaningful.

## Architecture

### 1. Feature-anchored graph autoencoder

A GRIT graph encoder maps each observed training region to node latents
`Q*=(a,z)`, with S2's `z_dim=16`; the symmetric decoder reconstructs loopless adjacency. The GAE is
trained only on legal train-region edges and then frozen. To remove the orthogonal gauge that
confounded S2, each role matrix is aligned by orthogonal Procrustes to a deterministic, frozen
projection of `X_S`; flow targets are the aligned latents, while decoded-edge BCE remains
gauge-invariant supervision.

### 2. Conditional latent generator

The generator is an optimal-transport conditional rectified flow. A permutation-equivariant
Set Transformer encodes all rows of `X_S`; an eight-block, width-512 Set-DiT velocity field maps
node-cached Gaussian noise to the aligned GAE latent set. Training samples a continuous time `t`
and predicts the OT flow velocity; OT couples whole set tensors within a same-size batch and never
permutes node rows. Inference uses a 16-step Heun solver and `K=32` graph draws. Within one draw,
the same generated `Q_S` serves every queried pair in the set.

### 3. B0 representation residual

B0 v3.1 is loaded from the same fixed `e092537d8cf1e208` deliverable checkpoint as S3, held in
evaluation mode, and frozen. For pair `(u,v)`, S4 uses B0's existing pre-head `pair_repr` and a
symmetric topology token

```text
t_uv = [q_u + q_v, |q_u - q_v|, q_u * q_v, AttentionPool(Q_S)], q_i = [a_i, z_i].
delta_h_uv = W_out SiLU(W_in LayerNorm([pair_repr_uv, t_uv])).
pair_repr'_uv = pair_repr_uv + delta_h_uv.
logit_uv = B0.output_head(pair_repr'_uv).
```

`W_out` is zero-initialized, so the initial predictor exactly reproduces frozen B0 while retaining
a live gradient into the adapter. There is no separate topology logit and no replacement decoder:
the generated graph latent can affect the decision only through a residual on B0 representation.

## Joint objective and optimization

Target-interpolated paths train only the flow and decoded-graph terms. The task term uses a separate
free-running latent sampled from pure Gaussian noise by an eight-step differentiable Heun solver,
conditioned only on `X_S`; it never receives a latent containing `A_S`. One objective trains the
generator and adapter while B0 and the GAE remain frozen:

```text
L = L_OT-flow(Z_t, Q*)
  + BCE(GAE.decode(Q_hat_target), A_S)
  + BCE(B0.head(pair_repr + delta_h(Q_free)), y_uv).
```

All losses are live-element means with unit weights and use train-side supervision only. The edge
loss backpropagates through the solver and adapter into the Set-DiT. Training uses mixed set sizes
and three generator/adapter seeds against one frozen GAE teacher. At inference, probabilities—not
logits—are averaged over the 32 shared graph draws.

The fixed configuration is Set-DiT width 512, eight blocks/eight heads, AdamW at `2e-4` with
weight decay `0.01`, gradient clip `1.0`, at most 60 epochs, and patience 10. B0 `pair_repr` is
materialized once in fp32 per unique canonical pair and indexed by regions; it is never cached per
region occurrence. GAE selection uses V_val decoded-edge loss, after which its weights and latent
normalization are frozen for all three S4 seeds.

## Evaluation and terminal rule

The primary marginal surface averages 32 probabilities, then reports paired per-set AUPRC, degree
Spearman, hub recall, and the five topology numbers: density-matched BFS-macro GS, BFS-macro RD,
and degree/clustering/spectral MMD ratios. A separately labeled generative surface reports the same
five topology numbers per shared graph draw; it never enters the marginal terminal rule. Calibration
and free-running implied density are secondary. Report per-seed paired deltas and mean±SD across
seeds; fixed-panel set bootstrap intervals are descriptive, not 1,500 independent observations.

Validation selection mean-ranks AUPRC and all five marginal topology metrics: AUPRC/GS higher,
`|RD-1|` and all MMD ratios lower; exact rank ties choose the earlier epoch. S4 is positive only if
the AUPRC and GS paired deltas are positive in every seed and their fixed-panel bootstrap intervals
exclude zero versus both B0 and completed S3 PAIR. Against both comparators, `|RD-1|` may increase
by at most 5% and no MMD ratio may exceed `1.05×` its comparator value. A
non-finite run, split-boundary violation, or disagreement across distributed ranks fails closed.
Weak model-quality telemetry does not block a run.

## Execution order

1. Build the aligned train-region GAE target cache; verify finite reconstruction and prove that no
   V_val-internal edge reaches GRIT inputs or cached structural targets.
2. On a debug slice, verify zero-init equality with B0, shared-within-draw latents, pair symmetry,
   nonzero adapter/generator gradients, and loss decrease.
3. Train three complete S4 seeds, selecting each checkpoint on V_val only.
4. Freeze selection, score the test buckets once, and write one aggregate report containing edge
   and topology metrics plus paired deltas to the already materialized B0/S3 reports.

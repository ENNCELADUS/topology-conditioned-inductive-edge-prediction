# L3-PPI: paper-based reproduction on our benchmark

Status: implementation verified locally; H20 reproduction pending. No experiment
result is claimed yet. Model/data/lifecycle/scoring/study checks and shared scoring
regressions pass, including a two-rank Gloo test with an empty final-rank batch.

## Question and comparator

Does replacing the published B0 readout and MLP by the complete L3-PPI head improve
queried-edge decisions and the topology assembled from those decisions?
The comparator is the existing B0, not a newly trained MLP. Its frozen Siamese
encoder is taken from `outputs/split_seed42_geometric_20260910/b0_v31/best.pt`
(epoch 8, checkpoint ID `7e78609f8c904284`). The encoder receives intrinsic endpoint
attributes only. Masked mean and max pooling produce a 1,024-dimensional vector.
The head's AB/BA logits are averaged; the baseline retains its original readout.

This is a paper-based reproduction on a different benchmark, not author-code or
original-table reproduction. Source: Gao et al., *Learning the Interaction Prior
for Protein-Protein Interaction Prediction: A Model-Agnostic Approach*, arXiv
2605.09964v2, §4, Figures 3–4, Appendix C. The local PDF is under
`literature/models/knowledge_distillation/`.

## Method and declared assumptions

1. **Surrogate:** for each training pair, union all simple length-three paths in
   the loopless legal training graph with the queried edge removed. Keep only
   path edges, not other induced edges. Actual node embeddings feed a graph-level
   GIN classifier. Count each pair once, regardless of its path count.
2. **Prompt-only:** freeze the surrogate; insert K+1 global learnable virtual nodes
   between the endpoints as `u -> private_i -> shared -> v`. Keep all paths active
   and update only prompt embeddings.
3. **Joint:** update prompts and a per-path GIN gate. Classification gradients pass
   through the frozen surrogate. Add Eq. (8)'s label-dependent path-count hinge,
   with gamma 3 and weight 0.3. It does not regress true path counts.

The paper does not fully specify several implementation details. This reproduction
uses the following explicit choices:

- Follow Figure 4 and Eqs. (3)–(4) when Eq. (6) gives an inconsistent final edge.
  Each private path has two gated edges; the shared final edge receives the max of
  its path gates. This yields K+3 nodes and 2K+1 undirected edges.
- Hard straight-through Binary Concrete gates during training; temperature decays
  geometrically from 1 to 0.1 over the joint epoch cap. Evaluation uses `p > 0.5`
  without noise. Closed paths retain their nodes and zero their edge weights.
- Weighted epsilon-zero GIN: sum neighbor aggregation, two-linear-layer ReLU MLP
  per layer, dropout, sum graph readout and a scalar linear classifier. Surrogate:
  two layers/width 64. Gate: two layers/width 128. Dropout 0.1.
- A real pattern is the union of all paths for one pair. Zero-path examples retain
  two isolated endpoints; self-pairs use two slots with identical features. No
  paths are truncated; a bounded LRU can evict and recompute entire patterns.
- Mean/max endpoint pooling is a benchmark adaptation. Prompt initialization is
  training-feature mean plus Gaussian noise at 0.01 times each dimension's training
  standard deviation. No held-out feature enters these statistics.
- The two known featureless training nodes are isolated in this split. Their
  supervised pairs are filtered as for B0; their graph membership is retained.
  A connected featureless intermediate is an error, never silently removed.

## Training and selection

One surrogate is shared by six seed-0 trials: K in {4,16,64} crossed with prompt/gate
Adam LR in {1e-4,1e-3}. Surrogate Adam LR is 1e-3. Weight decay is zero. Global batch
size is 64, independent of the number of visible GPUs; gradient norm is clipped at
1. All stages use the existing globally sampled dynamic 1:5 negative stream.
Weighted BCE has positive weight 5 and is normalized by total row weight, matching
B0. The path hinge is separately averaged over rows and the two orientations.

Surrogate and prompt-only stop when the range of the last ten epoch losses is less
than 0.05, or at 100 epochs. Their minimum-training-loss weights transfer to the
next phase; cap exhaustion is recorded, not called convergence. Joint training has
a 25-epoch cap and patience 10 on unweighted `val_cls` BCE. Validation always uses
hard deployment gates. Each epoch saves all five topology metrics, edge AUPRC/AUROC,
task/regularizer losses, soft/hard path counts and compute times.

Current node-held-out split and `geometric_rd_five_rank_v1` selection are unchanged:
rank V_val AUPRC, GS and the three MMD ratios equally; RD is reporting-only. Rank
each trial's selected checkpoint with the same rule, then test only the locked
winner. Keep the checkpoint's frozen V_val topology threshold; the common evaluator
freezes max-F1 classification threshold on `val_cls` before test. Report edge
metrics including calibration alongside GS, RD and all three MMD ratios.

True graphs are used only for training patterns and common evaluation references.
Deployed scoring needs the published weights and raw endpoint feature pack, not the
training graph, source B0 checkpoint or pattern cache. Features and graph operations
are FP32; source token storage remains the existing BF16 pack. Scoring batches are
capped at 64 to avoid inheriting the generic scorer's much larger row batch.

## Execution and artifacts

From the H20 checkout, after Git synchronization:

```bash
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 .venv/bin/python -m src.experiments.l3ppi_study \
  --config configs/split_seed42/l3ppi.yaml --output-dir outputs/l3ppi_seed0
```

The driver dispatches every stage through `hpc/run.sh`, which auto-sizes the GPU
worker group. It first profiles exact path extraction on 512 training rows, then
logs measured epoch throughput. Use `--resume` for interrupted work, `--skip-test`
to stop at winner publication, or `--prepare-only` to generate configs without a
launch. Epoch-boundary resume restores phase, model, optimizer and per-rank RNG;
keep the same config and world size. A completed stage is not rerun.

Study artifacts: generated configs, `surrogate/`, six `trial_*/` directories,
`winner.json`, per-stage logs and status. Each worker saves `profile.json`,
`metrics.jsonl`, `last.pt`, `best.pt`, and a publication `complete.json`. Trials also
save `selection.json` and per-epoch checkpoints. Only the winner receives merged
scores and `test_report.json`. Study status distinguishes `published` from `tested`;
publication alone does not establish successful held-out evaluation.

First-wave claims are single-seed observations of complete-head replacement.
They do not isolate the L3 prior from pooling/head adaptation, and do not establish
the utility of new branch, bottleneck, loop or endpoint-position templates.

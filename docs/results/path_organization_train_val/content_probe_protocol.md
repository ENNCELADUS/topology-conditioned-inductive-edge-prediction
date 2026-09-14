# Fixed follow-up protocol — 2026-09-14

Declared before generating or inspecting follow-up probe outcomes.

- Frozen content: published `outputs/split_seed42/prefix_base/best.pt`, checkpoint
  `614003b9ba3a8d53`, epoch 10. Reuse its unmodified logits, with no content training.
- Primary comparison: calibrated content + endpoint degree versus additions of
  triangles only, mean neighbor degree only, both, and both plus endpoint role
  combinations. Also report raw content, calibrated content, and context without content.
- Fixed logistic probes: training-standardized inputs, C=1, max_iter=1000,
  positive sample weight 5. Context quantities use log1p. Content calibration is
  fitted on training rows only; no validation tuning or early stopping.
- Role features: sums of within-endpoint products of log degree/triangles/mean
  neighbor degree, and their triple product. Unlike independently sorted margins,
  these preserve assignment within each endpoint tuple while remaining swap invariant.
- Three extra training-only node holdouts: uniformly sample floor(0.2*N) nodes
  without replacement from sorted current training nodes with seeds 1101, 1102,
  1103. Persist exact memberships before sampling pairs or fitting. Each probe
  trains on the complement's induced graph and evaluates the holdout's induced
  graph; cross-boundary pairs and edges are excluded. Use every eligible row.
- Negatives: existing `NegativeSampler` through `enumerate_edge_stream`, ratio 5,
  global rank 0 / world size 1, epoch 1, seed 20260914 for fit and 20260915 for
  evaluation. Use each universe's loopless degrees and exclude unavailable feature
  nodes. Reject its known positives (test/global positives are unnecessary because
  every eligible endpoint is in the original train-side substrate). Sample with
  self-positives as production does, then exclude all self-pairs from this study.
  This is one fixed realization of the dynamic sampler per universe, not a study
  of convergence over many resampled epochs.
- Main fit uses all current training nodes; primary V_val evaluation uses the same
  sampler at 1:5. Also retain official non-self `val_cls` as a secondary population.
- Query-edge deletion precedes degrees, triangle counts, and neighbor-degree means.
  All arms use identical canonical evaluation pairs and labels. No test inputs.
- Primary paired outcome: reduction in balanced BCE (half positive mean loss plus
  half negative mean loss). Report AUROC/AUPRC and their paired point differences.
  Use 1,000 shared bootstrap weight draws, seed 771: node multiplicities and,
  separately, graph-community multiplicities (Louvain seed 42, resolution 1).
  For cross-group pairs multiply the endpoint-group multiplicities; within-group
  pairs use one group multiplicity. Report percentile sensitivity intervals.
  These conditional resampling intervals do not capture arbitrary dependence
  across community boundaries or content-training uncertainty. Do not bootstrap
  rows independently or treat holdouts with overlapping nodes as independent runs.
- Directional repetition across the three holdouts and V_val matters more than
  the nominal pair count. Report all arms and holdouts; select none using outcomes.
- Scope: oracle context increment on frozen content scores, not prompt recovery.
  Extra holdouts were seen by the existing content model; they are held out only
  from the newly fitted probes. V_val was held out of content fitting but used for
  original checkpoint selection and the previous exploration, so is not a fresh test.
  Neither experiment establishes unseen-protein generalization of a retrained model.

No template, virtual-context predictor, or reader intervention is trained here.

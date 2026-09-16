# Virtual topology prompt repair

**Date:** 2026-09-16  
**Status:** implementation contract; no repair training or performance conclusion.  
**Supersedes:** [the historical 64-block virtual design](2026-09-15-virtual-topology-prompt-design.md).

## Purpose and boundary

The queried edge remains a binary decision from exactly `(x_u, x_v)`. A compressed training
structure supplies intermediate context through predicted topology coordinates. Repair three
mechanisms: weak slot differentiation, pair gates that can act as global density controls,
and compression that cannot recover endpoint clustering. Retain one graph counting path;
there is no retrieval, full training-node dictionary, bypass coordinate head, linear probe,
or rule closing the research route after a negative result.

This change supplies code, tests, configuration and documentation in an independent worktree.
It does not launch training, push or merge. Old virtual checkpoints stay on the original
branch; the new generator does not load them through a compatibility path. Existing v1/v2
coordinate implementations remain available to other arms.

## Model

The training-only initializer takes the legal, feature-present, loopless induced graph.
On the main process it partitions this graph with `SpectralClustering(n_clusters=256,
n_components=256, affinity="precomputed", assign_labels="cluster_qr",
eigen_solver="arpack", random_state=seed)`. It records excluded featureless nodes and
broadcasts initialized state before distributed training. Save the training-node order and
block assignments with the checkpoint. Resuming restores this mapping and generator state
without reclustering, so auxiliary targets retain the checkpoint slot order.

For block `j`, store fixed buffers for size `m_j`, mean masked-mean reader encoder state
`P_j`, and symmetric block density `B_jl`. Density denominators are `m_j m_l` between
blocks and `m_j(m_j-1)` within a block; singleton blocks have zero internal density.
No training modifies these buffers. This is a compressed graph summary, not a guarantee
that 256 blocks preserve all neighbour identities or within-block correlations.

Residues and prototypes share one trainable projection into 128 dimensions. Four-head
attention uses normalized queries/keys, temperature 4, and masked residues, with no second
independent Q/K projection. The matching MLP is
`Linear(512,128) → GELU → Linear(128,1)` on `[o, p, o*p, abs(o-p)]`.
Its per-slot bias starts at inverse-softplus of the mean training neighbour count for
that block, with a 0.01 floor. Predicted counts and attachments are

```
n_uj = min(softplus(r_uj), m_j)
a_uj = n_uj / m_j
```

Attachment depends on one protein alone. Delete pair gates, their pooled input pathway,
the gate intervention, trainable block sizes/densities, global sparse-bias calibration and
coordinate affine calibration. The hard count cap has zero gradient above the block size;
large learning-rate overshoots can saturate slots, so the small fit test verifies learning
at a fixed moderate learning rate rather than asserting immunity to saturation. Reader prompt gates remain part of the existing prefix
interface; they are distinct from the deleted virtual-graph pair gates.

For `N_jl = m_j*m_l` off diagonal and `N_jj = m_j*(m_j-1)`, count in FP32:

```
degree_u = sum_j m_j a_uj
common = sum_j m_j a_uj a_vj
Jaccard = common / (degree_u + degree_v - common)
L3 = sum_jl a_uj a_vl N_jl B_jl
L3_density = L3 / (degree_u * degree_v)
```

Zero denominators return zero. True-coordinate self-pair conventions remain, including
Jaccard one and zero distance indicators. Predicted coordinates remain soft graph quantities;
the distance head retains its self class, whose probability mass contributes no distance
indicator. The generator does not receive node IDs to enforce a self-pair override. The two query roots have no direct edge.
The distance head consumes only graph counts and predicts the existing distance classes;
it is not an exact shortest-path algorithm on fractional graphs. Fixed reader training
statistics standardize the coordinates; there is no learned coordinate-scale correction.

## Coordinates and supervision

`coord_spec: v3` has nine columns, retaining the corresponding v2 semantics:

| Field | Columns |
|---|---|
| endpoint_u | log1p_degree |
| endpoint_v | log1p_degree |
| relation | log1p_common, Jaccard, log1p_L3, L3_density, dist_2, dist_3, dist_4plus |

There are six continuous values and three distance indicators, three fields and two
prefix rows per field at all nine existing attention sites. Clustering is removed without
a constant placeholder. True coordinates still remove the queried edge. A fresh v3 reader
and v3 standardization statistics are mandatory; no slicing of old reader weights.

Stage II keeps the D objective, frozen reader encoder/cross-attention, immutable teacher,
trainable generator/interface/head, and full 15-epoch schedule. Add weight-1 attachment loss
exactly once, only to the ordinary edge stream:

```
Huber(log1p(predicted neighbour count), log1p(training neighbour count))
```

For each endpoint, average nonzero-target slots and zero-target slots separately, then
give the two nonempty groups equal weight. Average endpoints and rows. Empty groups do
not contribute. These training-only targets describe full neighbourhoods in the legal
feature-present induced training graph. They supervise a node prior; they are not forward
inputs and are not claimed to be queried-edge-removed counts. Coordinate loss independently
supervises the edge-removed pair quantities, excludes self rows, and uses nonself training
standard deviations. Self rows retain task/KD supervision. The structural stream receives
no second attachment loss. DDP scales task terms by global label-weight sum and attachment
terms by global row count separately; epoch loss reporting preserves these denominators.
The attachment gradient diagnostic reduces parameter gradients before taking the norm.

Record attachment loss, errors for zero/nonzero targets, slot usage and observed gradients
as diagnostics, never runtime eligibility gates. This does not add balance, entropy or
orthogonality regularization.

## Configuration and future execution

- `configs/split_seed42/topo_prompt_full_v4.yaml`: new v3 true-coordinate ceiling; existing
  Stage I corruption, optimizer and training schedule, diagnostic run kind.
- `configs/split_seed42/virtual_prompt_v2_d.yaml`: v3 reader at the v4 output path, K=256,
  128-dimensional four-head attachments; fixed D recipe plus attachment loss.
- `src/experiments/virtual_prompt_chain.sh`: reader then student then `gates_off`, `mean`,
  `mean_relation`, `shuffle`. `gates_off` disables reader prompt gates. `mean_context` is
  invalid on v2/v3, and there is no virtual-slot gate intervention.

The two stages write independent new output directories. Before any future production
launch, measure production-size memory and throughput rather than assuming the increased
slot count is cheap. Follow the HPC runbook; this implementation does not authorize or
perform that launch.

## Verification and interpretation

Tests cover v3 projection from v2 truth for positive, negative, self and swapped pairs;
binary lifted-graph counting; finite fractional counts/gradients and bounded ratios;
padding/partner/batch/shard-invariant attachment; actual-generator learning of different
slot neighbourhoods at equal degree; training-only target boundaries; single inclusion of
attachment loss; checkpoint round-trip; distributed initialized-state agreement; immutable
teacher; and packed scoring without graph, labels or coordinate targets.

Run focused pytest, changed-file Ruff and mypy, then inspect the final diff. Distinguish
pre-existing failures from regressions. No GPU performance result or scientific gain is
established by local correctness checks. Future results report edge metrics and all five
topology metrics: GS, RD, degree/clustering/spectral MMD. Dropping the clustering input
never drops clustering evaluation. A negative result concerns this implementation and
experiment, not the feasibility of all endpoint-conditioned structural learning.


## Local implementation verification (2026-09-16)

- 124 focused model, coordinate, training, scoring and structural-stream tests passed,
  including the two-process tests with local sockets enabled; 12 probe regression tests passed.
- Changed-file Ruff and mypy passed. Repository-wide mypy reports the same 102 errors in
  the same 16 files as an isolated archive of baseline `9dfe9e4`; no new errors.
- Resume tests forbid reclustering and verify saved slot/target alignment. Unequal-partition
  tests compare gradients against the global composite objective. Final targeted review
  found no remaining correctness issue.
- No repair training, GPU memory/throughput measurement, push or merge was performed.

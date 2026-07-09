# Graph Structure Learning and Graph Construction Methods

## Research Question

Which ML methods provide useful precedent for constructing, refining, denoising,
sparsifying, rewiring, clustering, or hierarchically organizing a graph when the
graph structure is a first-class output?

This report focuses on graph structure learning, latent adjacency learning,
graph refinement, topology-aware regularization, differentiable clustering, and
higher-order graph modeling. The goal is to support a general inductive edge
prediction project where graph-level fidelity matters as much as individual
edge scores.

## Search Axes

- **Structure learning:** latent adjacency, bilevel edge distributions,
  differentiable edge modules, iterative graph learning.
- **Refinement and denoising:** edge pruning, reweighting, sparsification,
  robustness to perturbed graphs.
- **Topology-aware objectives:** sparsity, low rank, smoothness, modularity,
  hierarchy, clustering, and spectral properties.
- **Hierarchical and higher-order modeling:** pooling, coarsening, motif-aware
  structure, subgraph-level representation learning.
- **Evaluation:** graph-level similarity metrics and benchmarks for structure
  learning methods.

## Inclusion and Exclusion

Include papers where graph structure is learned, refined, or evaluated as an
explicit object. Exclude papers that only score node pairs independently and
leave graph construction to a thresholding step, unless they provide a useful
baseline or warning about benchmark design.

## Ranked Candidate Papers

| Priority | Paper | Year | Why it matters |
|---|---:|---:|---|
| A | *On Evaluation Metrics for Graph Generative Models* | 2022 | Gives graph-level metric families for comparing generated or reconstructed graphs. |
| A | *Learning Discrete Structures for Graph Neural Networks* | 2019 | Canonical bilevel formulation for learning a discrete edge distribution. |
| A | *Iterative Deep Graph Learning for Graph Neural Networks* | 2020 | Iteratively refines graph structure and node representations. |
| A | *Graph Structure Learning for Robust Graph Neural Networks* | 2020 | Direct adjacency purification with sparsity, low-rank, and smoothness priors. |
| A | *Towards Unsupervised Deep Graph Structure Learning* | 2022 | Learns graph structure without relying on labels. |
| A | *Self-Supervised Graph Structure Refinement for Graph Neural Networks* | 2023 | Pretrain-then-refine pattern for adding and removing edges. |
| B | OpenGSL, *A Comprehensive Benchmark for Graph Structure Learning* | 2023 | Standardized benchmark and implementation reference for GSL methods. |
| B | GSLB, *The Graph Structure Learning Benchmark* | 2023 | Useful cross-check on robustness and complexity claims. |
| B | *Graph Clustering with Graph Neural Networks* | 2023 | Differentiable modularity objective for graph clustering. |
| B | *Overlapping Community Detection with Graph Neural Networks* | 2019 | Neural approach to overlapping community structure. |
| B | *Higher-order Clustering and Pooling for Graph Neural Networks* | 2022 | Motif-aware clustering and pooling for higher-order structure. |
| B | *Hierarchical Graph Pooling with Structure Learning* | 2019 | Combines graph coarsening with learned structure at each pooling layer. |
| B | *Hierarchical Graph Representation Learning with Differentiable Pooling* | 2018 | Foundational hierarchical pooling baseline. |
| B | *Subgraph Neural Networks* | 2020 | Connects edge-level signals to subgraph-level representation learning. |

## Reliability and Novelty Assessment

The strongest anchor set for a graph-construction project is LDS, IDGL,
Pro-GNN, SUBLIME, GSR, OpenGSL, and the ICLR graph-generation-metric paper.
Together they cover discrete structure learning, iterative refinement,
regularized denoising, unsupervised structure learning, self-supervised edge
refinement, standardized benchmarking, and graph-level evaluation.

Most methods still optimize downstream prediction losses, not graph realism
directly. That creates a useful opening for work that treats topology metrics as
training objectives, model-selection criteria, or calibration constraints.

## Design Implications

For a practical graph-construction pipeline:

1. Start with a candidate graph from nearest-neighbor search, a weak edge
   scorer, or a sparse prior.
2. Apply a structure-learning or refinement model that can add, remove, or
   reweight edges.
3. Enforce a sparsity budget or degree constraint before graph-level evaluation.
4. Evaluate with pairwise metrics and graph-level metrics.
5. Compare against simple baselines: thresholding, k-nearest neighbors,
   Pro-GNN-style refinement, and IDGL-style iterative learning.

## Recommended Reading Order

1. *On Evaluation Metrics for Graph Generative Models*.
2. LDS.
3. IDGL.
4. Pro-GNN.
5. SUBLIME or GSR.
6. OpenGSL or GSLB for implementation and benchmark details.
7. DMoN, NOCD, HoscPool, or DiffPool if community or hierarchy is part of the
   project.

## Remaining Gap

The corpus still needs more work on models that explicitly optimize degree,
clustering, spectral, component, or modularity constraints during graph
construction. Existing structure-learning papers often evaluate those
properties after training rather than making them central to the objective.

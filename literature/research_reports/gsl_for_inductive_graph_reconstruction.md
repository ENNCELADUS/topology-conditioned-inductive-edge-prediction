# Graph Structure Learning for Inductive Graph Reconstruction

## Scope

This report surveys graph structure learning methods for settings where the
graph itself is the target: learning an adjacency, refining a noisy candidate
edge set, or reconstructing missing edges from node features. The emphasis is
general ML rather than any application-specific network.

The methods are organized around two modeling routes:

- **Route A: edge denoising and refinement.** Start with a candidate graph and
  learn a cleaner sparse structure through regularization, pruning, reweighting,
  or self-supervised edge reconstruction.
- **Route B: inductive graph reconstruction.** Build a graph from node features
  when no trusted adjacency is available, with attention to sparsity,
  uncertainty, and out-of-sample nodes.

The main evaluation question is whether a method improves graph-level fidelity,
not only pairwise edge accuracy. Useful graph-level checks include degree
distribution, clustering, connected components, spectral similarity, modular
structure, sparsity, and stability under missing or noisy edges.

## Inclusion Criteria

Include methods that explicitly learn, refine, or score graph structure as an
output:

- learnable adjacency matrices or edge distributions;
- edge reweighting, pruning, or sparsification;
- self-supervised graph refinement;
- latent graph inference from node attributes;
- topology-aware objectives such as sparsity, low rank, smoothness,
  hierarchy, or community structure.

Downgrade methods that only produce independent pair scores and rely on a
thresholded graph as a post-processing artifact. Those methods can be useful
upstream generators, but they do not by themselves solve graph reconstruction.

## Route A: Refinement From a Candidate Graph

| Priority | Method | Main idea | Why it matters |
|---|---|---|---|
| A | Pro-GNN, *Graph Structure Learning for Robust Graph Neural Networks* (KDD 2020) | Directly optimizes a cleaned adjacency with sparsity, low-rank, and feature-smoothness priors. | Strong baseline when the starting graph is noisy and the refined graph must remain sparse and structured. |
| A | IDGL, *Iterative Deep Graph Learning for Graph Neural Networks* (NeurIPS 2020) | Alternates between representation learning and graph structure updates. | Covers both noisy-graph and feature-only initialization, making it a useful bridge between refinement and reconstruction. |
| A | GSR, *Self-Supervised Graph Structure Refinement for Graph Neural Networks* (WSDM 2023) | Learns edge probabilities with self-supervised tasks, then adds or removes edges before downstream training. | Clean pipeline pattern: learn structure first, freeze or reuse the refined graph, then evaluate. |
| B | GNNGuard, *Robust Graph Neural Network Edge Reweighting* (NeurIPS 2020) | Reweights and prunes edges using feature agreement. | Practical edge-cleaning baseline, especially when node features are strong. |
| B | STABLE, *Reliable Representations Make A Stronger Defender* (KDD 2022) | Builds reliable representations, then performs unsupervised structure refinement. | Useful when raw feature similarity is not stable enough for pruning decisions. |
| B | PTDNet, *Learning to Drop* (WSDM 2021) | Learns to drop task-irrelevant edges with sparsity and low-rank constraints. | Good template for controlled sparsification and topology-preserving denoising. |
| B | SE-GSL, *Structural Entropy Optimization* (WWW 2023) | Uses structural entropy and hierarchy-aware objectives for graph refinement. | Relevant when community or hierarchy preservation is central to graph quality. |

## Route B: Inductive Reconstruction From Node Features

| Priority | Method | Main idea | Why it matters |
|---|---|---|---|
| A | LDS, *Learning Discrete Structures for Graph Neural Networks* (ICML 2019) | Learns a discrete edge distribution through bilevel optimization. | Foundational template for missing-graph or incomplete-graph settings. |
| A | DGM, *Differentiable Graph Module* (TPAMI 2022) | Adds a differentiable module that predicts edge probabilities from node representations. | Explicitly supports inductive graph discovery and optional use of an initial graph. |
| A | NodeFormer, *A Scalable Graph Structure Learning Transformer* (NeurIPS 2022) | Learns sparse latent structure with scalable attention-style mechanisms. | Strong candidate when reconstruction must scale beyond dense all-pairs scoring. |
| A | SUBLIME, *Towards Unsupervised Deep Graph Structure Learning* (WWW 2022) | Uses contrastive learning and bootstrapping to learn structure without labels. | Useful when trusted edges are incomplete or unavailable. |
| B | Uncertainty-aware latent graph learning (ICML 2025) | Learns a distribution over plausible latent graphs rather than a single point estimate. | Supports calibrated edge confidence and uncertainty-sensitive reconstruction. |
| B | Latent graph inference with limited supervision (NeurIPS 2023) | Studies graph inference when labels are scarce and sparsification can remove important edges. | Good warning paper for reconstruction pipelines that prune too aggressively. |

## Method Selection Notes

For a conservative baseline, start with Pro-GNN or IDGL. Pro-GNN gives a clear
regularized adjacency objective; IDGL provides a practical iterative loop and
can initialize from either a noisy graph or feature-derived nearest neighbors.

For a feature-only reconstruction setting, compare a discrete edge-learning
method such as LDS against a scalable latent-structure model such as NodeFormer
or DGM. Report both edge metrics and graph-level fidelity metrics, because a
high edge score can still produce an implausible graph.

For a topology-first objective, add explicit constraints or selection criteria:
sparsity budget, degree-distribution distance, clustering distance, spectral
distance, connected-component stability, modularity, or hierarchy-aware losses.
Most surveyed methods were originally optimized for downstream task accuracy,
so graph-level realism should be made an explicit validation target.

## Recommended Reading Order

1. IDGL for the broadest refinement/reconstruction template.
2. Pro-GNN for direct adjacency purification with structural priors.
3. LDS for discrete latent edge learning.
4. NodeFormer or DGM for scalable inductive graph discovery.
5. GSR or SUBLIME for self-supervised structure refinement.
6. SE-GSL if community or hierarchy preservation is a key requirement.

## Remaining Gap

The current method set is still light on models that directly optimize learned
adjacency to match graph-level topology distributions during training. A focused
follow-up search should cover graph generation, distribution matching, and
denoising or diffusion-style graph reconstruction methods with explicit
structural constraints.

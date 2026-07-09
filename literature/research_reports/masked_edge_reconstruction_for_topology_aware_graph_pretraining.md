# Masked Edge Reconstruction for Topology-Aware Graph Pretraining

## Overview

Masked edge reconstruction is a self-supervised approach for learning
topology-aware node representations. The model observes a partially masked graph,
encodes the visible structure, and reconstructs hidden edges or related
structural signals. This is a better fit for graph reconstruction than masked
node-feature reconstruction when the main missing signal is topology.

Classical graph autoencoders established the core template: encode nodes with a
graph neural network, then decode adjacency. Newer masked-graph methods refine
that idea by hiding edges, paths, or continuous edge weights and training the
model to recover missing structure.

## Representative Paradigms

| Paradigm | Representative work | Masking strategy | Objective | Relevance |
|---|---|---|---|---|
| Classical graph autoencoding | Kipf and Welling, *Variational Graph Auto-Encoders* (2016) | No explicit mask; reconstruct observed adjacency | Adjacency likelihood or ELBO | Baseline for link reconstruction. |
| Node-feature masking | GraphMAE (KDD 2022) | Mask node attributes | Feature reconstruction | Useful contrast case; less direct for topology-first pretraining. |
| Edge masking | MGAE (2022) | Randomly mask a large edge fraction | Masked edge reconstruction | Simple topology-first baseline. |
| Path-wise edge masking | MaskGAE (KDD 2023) | Mask edges or paths sampled by random walks | Edge reconstruction plus degree regularization | Strong option when local paths, hubs, and higher-order structure matter. |
| Direction-aware edge masking | S2GAE (WSDM 2023) | Directed or undirected edge masking | Masked-edge likelihood with cross-correlation decoding | Clean direct fit for link-prediction pretraining. |
| Mixed feature and structure masking | SeeGera (WWW 2023) | Mask both features and structure | Variational reconstruction of links and features | Useful when uncertainty modeling is important. |
| Heterogeneous graph masking | HGMAE (AAAI 2023) | Mask metapaths and attributes | Multi-task structure and attribute reconstruction | Relevant for multi-relation graphs. |
| Continuous topology masking | Bandana (WWW 2024) | Predict continuous edge bandwidths | Structure-only bandwidth prediction | Experimental option when binary deletion is too brittle. |
| Contrastive comparator | GraphCL (NeurIPS 2020) | Graph augmentations such as edge perturbation and subgraph sampling | View agreement | Useful baseline, but not explicit missing-edge recovery. |

## Transfer Recipe

1. Build a train-only graph.
2. Initialize nodes with the available feature vectors.
3. Mask only training edges.
4. Run a graph encoder on the visible graph.
5. Decode masked edges with an inner-product, bilinear, MLP, or
   cross-correlation decoder.
6. Optionally add degree, path, sparsity, or contrastive auxiliaries.
7. Remove masking after pretraining and use the encoder for inductive edge
   scoring, graph refinement, or downstream graph construction.

Negative sampling should occur after masking so that hidden positives are not
accidentally used as negatives. Report results under multiple mask rates,
including one high-mask regime and one moderate-mask regime.

## Priority Recommendations

**Top choice: S2GAE-style masked-edge pretraining.** It directly trains on
missing-edge recovery and provides a clean link-prediction recipe.

**Second choice: MaskGAE-style path masking with degree regularization.** This
is useful when path structure, hubs, or local motifs are expected to influence
edge reconstruction.

**Third choice: MGAE as a minimal strong baseline, with Bandana as an
experimental upgrade.** MGAE is easy to implement and topology-first. Bandana is
worth testing when hard binary masks destabilize training.

GraphMAE should be kept as a contrastive baseline for masked feature
reconstruction, not as the primary topology-pretraining objective.

## Evaluation

Evaluate both the decoder and the induced graph:

- masked-edge AUROC and AUPR;
- precision at fixed sparsity;
- degree-distribution distance;
- clustering and component statistics;
- spectral similarity;
- stability across mask rates;
- performance on unseen nodes or held-out graph regions.

## Sources

- Kipf and Welling, *Variational Graph Auto-Encoders*, 2016.
- Hou et al., *GraphMAE: Self-Supervised Masked Graph Autoencoders*, 2022.
- Tan et al., *MGAE: Masked Autoencoders for Self-Supervised Learning on Graphs*, 2022.
- Li et al., *What's Behind the Mask: Understanding Masked Graph Modeling for Graph Autoencoders*, 2023.
- Tan et al., *S2GAE: Self-Supervised Graph Autoencoders Are Generalizable Learners with Graph Masking*, 2023.
- Li et al., *SeeGera: Self-supervised Semi-implicit Graph Variational Auto-encoders with Masking*, 2023.
- Tian et al., *Heterogeneous Graph Masked Autoencoders*, 2023.
- Zhao et al., *Masked Graph Autoencoder with Non-discrete Bandwidths*, 2024.
- You et al., *Graph Contrastive Learning with Augmentations*, 2020.

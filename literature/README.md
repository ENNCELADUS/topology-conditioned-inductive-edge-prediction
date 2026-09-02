# Literature PDF Index

The active PDF library lives under `literature/models/`. Filenames use
`year_venue_arxiv_<id>_title.pdf` when an arXiv ID is known and
`year_venue_title.pdf` otherwise.

- Active PDFs: 116
- PDFs with arXiv IDs in filenames: 112
- PDFs without confirmed arXiv IDs: 4

## Folder map

| Folder | Scope |
|---|---|
| `knowledge_distillation/` | GNN-to-MLP, relational, representation, and topology-aware distillation |
| `inductive_cold_start/` | Unseen-node and attributes-only prediction without observed neighborhoods |
| `link_prediction_structural/` | Structural link-prediction methods and evaluation |
| `domain_adaptation_and_transfer/` | Domain and source-hypothesis transfer |
| `neighbor_generation/` | Neighborhood augmentation and feature propagation |
| `graph_generation/` | Generative graph models |
| `graph_generation_realism/` | Graph-generation realism and evaluation |
| `graph_structure_learning/` | Latent structure, refinement, pooling, masking, and GSL benchmarks |
| `kg_inductive_lineage/` | Inductive knowledge-graph reasoning lineage |
| `cv_generative_mechanisms/` | Generative mechanisms imported from computer vision |

## Priority: topology-transfer and link-prediction distillation

| Priority | Paper | File |
|---:|---|---|
| 1 | Linkless Link Prediction via Relational Distillation (LLP) | `models/knowledge_distillation/2023_icml_arxiv_2210_05801_linkless_link_prediction_via_relational_distillation.pdf` |
| 2 | Weak Models Can be Good Teachers / EHDM | `models/knowledge_distillation/2025_log_arxiv_2504_06193_weak_models_can_be_good_teachers_a_case_study_on_link_prediction_with_mlps.pdf` |
| 3 | Graph-less Neural Networks (GLNN) | `models/knowledge_distillation/2022_iclr_arxiv_2110_08727_graph_less_neural_networks_teaching_old_mlps_new_tricks_via_distillation.pdf` |
| 4 | VQGraph | `models/knowledge_distillation/2024_iclr_arxiv_2308_02117_vqgraph_rethinking_graph_representation_space_for_bridging_gnns_and_mlps.pdf` |
| 5 | Cold Brew | `models/knowledge_distillation/2022_iclr_arxiv_2111_04840_cold_brew_distilling_graph_node_representations_with_incomplete_or_missing_neighborhoods.pdf` |
| 6 | CAZI-MBN | `models/knowledge_distillation/2026_iclr_arxiv_2603_06618_distilling_and_adapting_a_topology_aware_framework_for_zero_shot_interaction_prediction_in_multiplex_biological_networks.pdf` |
| 7 | Graph2Feat | `models/knowledge_distillation/2023_www_graph2feat_inductive_link_prediction_via_knowledge_distillation.pdf` |
| 8 | DEAL | `models/inductive_cold_start/2020_ijcai_arxiv_2007_08053_inductive_link_prediction_for_nodes_having_only_attribute_information.pdf` |

The graph knowledge-distillation survey is stored at
`models/knowledge_distillation/2025_acm_computing_surveys_knowledge_distillation_on_graphs_a_survey.pdf`.

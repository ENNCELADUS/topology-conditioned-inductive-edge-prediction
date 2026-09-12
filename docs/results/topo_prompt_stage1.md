# Topology-prompt Stage I (`v3_1_topo_prompt`): results

Headline split (seed 42, root `node_007630`), training seed 0. Spec:
`docs/superpowers/specs/2026-09-12-topology-prompt-stage1-design.md`. Both runs read the
queried pair's true structural coordinates (query edge removed) through gated KV prefixes in
the prefix_base trunk, so every row here is a **ceiling diagnostic** (`--run-kind diagnostic`,
`diagnostic_test_report.json`, artifacts `formal: False`), never a deployable arm.

| Run | Trainable | Status (2026-09-12) |
|---|---|---|
| `topo_prompt_full` | trunk + prompt from scratch (prefix_base recipe) | complete: trained 09:58–13:31 UTC on 30846, stopped at epoch 15 (val-task-loss patience 10, minimum at epoch 5), selected epoch 15; test diagnostic + six interventions by 15:12 UTC |
| `topo_prompt_frozen` | prompt only on frozen `prefix_base/best.pt` | training on 30846 since 16:02 UTC; `shuffle`, `mean`, `gates_off` follow |
| `shuffle` rerun (both) | scoring only | queued behind the frozen lane (`outputs/logs/topo_prompt_shuffle_rerun.sh`), see §4 |

Sources: `outputs/split_seed42/topo_prompt_full/{diagnostic_test_report.json,intervention_*/…}`,
`outputs/split_seed42/prefix_base/test_report.json`; val_cls and ball-union edge metrics recomputed
from the `scores/*.npz` of each run (val_cls: 10,694 rows 1:1; ball union: 297,209 rows, 1.8 % positive).

## 1. V_val readout (primary)

Edge metrics on val_cls, assembled-graph metrics at the run's own V_val-selected threshold
(`geometric_rd_five_rank_v1`). Interventions are scoring-time only, on the same checkpoint.

| Row | Epoch | val_cls AUROC ↑ | val_cls AUPRC ↑ | ball-union AUROC ↑ | GS ↑ | RD → 1 | Degree MMD ↓ | Clustering MMD ↓ | Spectral MMD ↓ | logit thr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `prefix_base` (no structure) | 10 | 0.7925 | 0.8143 | 0.7908 | 0.401 | 1.102 | 10.6 | 4.6 | 9.0 | 2.70 |
| **`topo_prompt_full`** | 15 | **0.9522** | **0.9607** | **0.9525** | **0.691** | 1.030 | 11.1 | 4.9 | 13.8 | 5.19 |
| ↳ `gates_off` (trunk alone) | — | 0.7245 | 0.7675 | 0.7389 | 0.374 | 1.083 | 9.6 | 3.9 | 7.8 | 1.74 |
| ↳ `mean` (all fields at training mean) | — | 0.7425 | 0.7790 | 0.7578 | 0.376 | 1.102 | 10.4 | 4.2 | 7.9 | 2.19 |
| ↳ `shuffle` (universe-level null) | — | (§4) | (§4) | 0.6472 | 0.262 | 1.091 | 27.7 | 42.8 | 6.0 | 2.33 |
| ↳ `mean_endpoint` | — | 0.9471 | 0.9588 | 0.9482 | 0.674 | 1.033 | 9.0 | 4.4 | 11.4 | 5.22 |
| ↳ `mean_relation` | — | 0.9240 | 0.9354 | 0.9202 | 0.542 | 1.076 | 12.6 | 5.2 | 10.2 | 4.09 |
| ↳ `mean_context` | — | 0.9504 | 0.9592 | 0.9553 | 0.703 | 1.021 | 11.5 | 5.0 | 12.7 | 4.88 |
| `topo_prompt_frozen` | | pending | | | | | | | | |

Training curve of `topo_prompt_full` (`metrics.jsonl`): val AUPRC 0.738 → 0.901 → 0.949 →
0.959 over epochs 1–4 and flat (0.958–0.961) afterwards; val task loss minimum 0.340 at epoch 5,
0.416 at the stop; train loss 0.109 at epoch 15; GS 0.65–0.70 from epoch 4. `prefix_base` under
the same recipe plateaued at val AUPRC 0.81–0.82 with val loss rising from epoch 4.

## 2. Test diagnostic (read with the universe-shift caveat)

Mean `log1p` degree is 1.83 on the training graph, 2.15 on the V_val graph and 3.26 on the test
graph, so the coordinates the reader sees at test are far from their training range and the
V_val-selected threshold transfers into a much denser region. Edge metrics on the 1:1 test list
(64,038 rows); topology at the transferred threshold.

| Row | AUROC ↑ | AUPRC ↑ | ECE ↓ | Brier ↓ | Acc ↑ | F1 ↑ | MCC ↑ | GS ↑ | RD → 1 | Degree MMD ↓ | Clustering MMD ↓ | Spectral MMD ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `prefix_base` | 0.7205 | 0.7441 | 0.293 | 0.315 | 0.634 | 0.679 | 0.279 | 0.409 | 0.456 | 11.4 | 9.7 | 16.8 |
| **`topo_prompt_full`** | **0.9295** | **0.9392** | 0.077 | 0.115 | 0.791 | 0.741 | 0.630 | 0.461 | 0.344 | 33.3 | 25.3 | 44.2 |
| ↳ `gates_off` | 0.6467 | 0.6952 | 0.160 | 0.264 | 0.521 | 0.655 | 0.067 | 0.385 | 0.570 | 7.4 | 6.6 | 10.9 |
| ↳ `mean` | 0.6651 | 0.7070 | 0.159 | 0.255 | 0.581 | 0.645 | 0.174 | 0.390 | 0.520 | 8.9 | 7.9 | 12.9 |
| ↳ `shuffle` (per-shard, vacuous here; §4) | 0.9411 | 0.9495 | 0.061 | 0.101 | 0.817 | 0.781 | 0.672 | 0.287 | 0.566 | 9.7 | 27.1 | 9.3 |
| ↳ `mean_endpoint` | 0.9191 | 0.9321 | 0.079 | 0.121 | 0.756 | 0.682 | 0.580 | 0.450 | 0.335 | 33.9 | 25.9 | 44.7 |
| ↳ `mean_relation` | 0.9120 | 0.9213 | 0.118 | 0.142 | 0.835 | 0.824 | 0.675 | 0.505 | 0.510 | 18.0 | 14.8 | 23.0 |
| ↳ `mean_context` | 0.9162 | 0.9304 | 0.089 | 0.126 | 0.786 | 0.734 | 0.622 | 0.457 | 0.337 | 31.1 | 24.3 | 42.1 |
| Full-Ego PMA1 oracle (reference) | 0.9498 | 0.9547 | 0.099 | 0.113 | 0.876 | 0.879 | 0.753 | 0.602 | 0.708 | 7.0 | 6.4 | 11.5 |

Test geometric RD 0.300 and mean |log RD| 1.204 for `topo_prompt_full` (prefix_base 0.445 /
0.810; oracle 0.667 / 0.445). Per-size GS falls from 0.55 (20-node subgraphs) to 0.38
(200-node), RD from 0.43 to 0.26.

## 3. Reading (spec §7)

1. **Outcome 1.** `topo_prompt_full` ≫ `prefix_base` on V_val in both families (val_cls AUPRC
   +0.146, GS +0.29 at RD ≈ 1), `mean` returns it to the base (slightly below: the trunk was
   trained with the prompt present), and the universe-level `shuffle` on the ball union drops it
   *below* the base (AUROC 0.647, GS 0.262): misleading structure is worse than none, so the
   reader relies on the coordinates rather than on their mere presence. The interface transmits
   structure and the reader uses it; Stage II has a well-defined target and `topo_prompt_full`
   is its teacher / initialisation.
2. **Fields.** Removing the relation field costs the most (GS 0.691 → 0.542, val_cls AUPRC
   0.961 → 0.935), the endpoint fields little (→ 0.674 / 0.959), the context field nothing
   (0.703 / 0.959, inside the ±0.01 GS noise band). No single field is necessary — the reader
   without the relation field is still far above the base — so the fields are redundant with each
   other; Stage II must predict the relation coordinates best and may drop the context field.
3. **Trunk alone.** `gates_off` sits below `prefix_base` (val_cls AUPRC 0.768 vs 0.814): the
   Stage I trunk is not a standalone endpoint model, as intended.
4. **Shape ratios.** GS rises by 0.29 while the three V_val MMD ratios do not move (degree 11.1 vs
   10.6, spectral 13.8 vs 9.0): the prompt changes which edges are admitted at RD ≈ 1, not the
   descriptor distribution of the admitted set.
5. **Test.** The edge family transfers (AUPRC 0.939 vs 0.744; calibration ECE 0.077 vs 0.293)
   but the transferred threshold under-densifies the dense test region (RD 0.34) and the shape
   ratios triple relative to the base. `mean_relation` is the best test topology row (GS 0.505,
   RD 0.51, MMD 18/15/23), consistent with the relation coordinates carrying most of the
   universe shift. The Full-Ego oracle faces the same shift (RD 0.71). V_val is the readout;
   any true-structure reader's test topology depends on threshold transfer across the density gap.
6. **Frozen lane** (outcomes 3/4 of §7) pending; its `prefix_pair_bce` comparison row does not
   exist yet (the 30030 studies are in `prefix_pair` trial 1/10; `prefix_pair_bce` follows).

Single-seed differences inside ±0.01 GS / ±0.5 MMD ratio are not read.

## 4. Shuffle correction (2026-09-12)

The `shuffle` intervention was implemented as one permutation of the rows *each scoring process*
scores — per shard under the four-GPU fan-out. The 1:1 `val_cls` (10,694 rows) and `test`
(64,038 rows) pair lists are label-sorted (positives first), so two shards were all-positive and
two all-negative and every substituted coordinate bundle carried the row's own label: shuffle
scored *above* the true coordinates there (test AUPRC 0.950 vs 0.939; val_cls 0.968 vs 0.961).
The ball-union V_val topology universe (positives 1.8 %, mixed within shards) was not affected,
which is why its shuffle row is reported above. The scorer now draws one seeded permutation of
the whole universe before sharding (`src.score_universe._shuffle_source_rows`; artifacts record
`prefix_shuffle_scope: universe`); the same correction applies to the prefix arms' pair-condition
shuffle. The per-shard reports are kept as `intervention_shuffle_pershard/` and both lanes'
`intervention_shuffle` are rerun under the corrected scorer once the frozen lane releases 30846.

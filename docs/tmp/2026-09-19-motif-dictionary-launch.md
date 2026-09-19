# Motif dictionary experiment launch record

Date: 2026-09-19. Implementation commit: `9f46efb`, branch
`codex/motif-graph-grit-prompt`. This is an execution record, not an experiment-result verdict.
Design: [approved spec](../superpowers/specs/2026-09-19-motif-dictionary-routing-design.md).

## Implementation and verification

Five GPT-5.6-sol medium implementation agents handled dictionary construction, model/head behavior,
training integration, scoring/diagnostics, and production checkpoint/scoring tests. The primary
agent reviewed and integrated all changes before launching.

- 646 integrated tests passed, plus the two real two/four-rank CPU Gloo structural-training checks.
- Changed-path Ruff, shell syntax and final diff checks passed.
- Full-tree mypy matched the baseline exactly: 103 existing errors in 17 files; no new errors after
  normalizing line-number movement.
- Live source checkpoints: Stage I epoch 11; final repaired wave3 epoch 2 (`d2bb4fc7a55a10d3`).
- Code was pushed to origin and fast-forwarded on the shared H20 checkout through Git HTTPS because
  the container's configured GitHub SSH key was unavailable. No working-tree file copy was used.

## Allocation and initial observations

All three endpoints were idle before launch and each exposed four NVIDIA H20 GPUs. They share
`/2023533015/topology-conditioned-inductive-edge-prediction`; the standard runner environment check
passed. The old two-GPU description of port 30030 was corrected to this live observation.

| Port | Work | Initial state |
|---|---|---|
| 30838 | Shared dictionary build, then sequence-router training | Builder completed; router started 17:08:03 UTC, PID 132178 |
| 30846 | Prompted head-only training, then matched content-only head training | Chain started 16:56:44 UTC, PID 187800 |
| 30030 | V_val-only dictionary oracle diagnostic | Started 17:08:07 UTC, PID 81201; scoring V_val |

Builder log: `outputs/logs/motif_dictionary_build_20260919.log`. Shared artifact:
`outputs/split_seed42/motif_dictionary/shared_seed0.pt`. The builder enumerated 2,483,246 unique
training-corpus pairs; its candidate reservoirs and graph-token batches are bounded.
It completed at 17:06:33 UTC with 16 real representatives and 31 entries after swap expansion,
temperature `0.016349660232663155`, and category allocation 1 empty / 3 attachment-only / 4 closure-only /
4 bridge-only / 4 combined. The saved artifact passed its numerical/swap validation; the builder
exited and released its GPUs. The router constructed successfully from the actual source checkpoints
and artifact: 847,839 trainable slot/router parameters, frozen PPI head, `K_eff=31`.

Head chain log: `outputs/logs/motif_wave3_head_chain_20260919.log`. Prompted-head attempt:
`outputs/split_seed42/motif_wave3_head/attempts/1128b8e71b724e179dd157507aada83f/`.
Its worker log confirmed all four ranks, 15 epochs, 5,413 scheduled optimizer steps and peak LR
`1e-5`; at 17:00:17 UTC it reached epoch 1, global step 66 with finite loss. The content-only control
is chained after a successful prompted-head publication, using the same container and budget.

Training uses `--skip-test`. Oracle scoring is limited to V_val. No held-out test was launched.
Required source, build and lane commands are recorded in [the runbook](../../hpc/README.md#motif-dictionary-and-head-adaptation-experiments).

## Router and oracle launch observations

Router attempt: `outputs/split_seed42/motif_dict_router/attempts/4c3bc71e8cfc4d299c40c37c5b3f954a/`.
Launcher log: `outputs/logs/motif_dict_router_20260919.log`; worker log: the attempt's `train.log`.
Four worker PIDs were observed: 132241–132244, beneath Accelerate 132221 and pipeline 132178.
At 17:10:22 UTC all four workers resolved the saved dictionary's actual size to 31 and were preparing
the distributed routing-target cache. The attempt status was `running`, with no failure artifact.
The cache completed at 17:11:44 UTC: 2,483,246 training rows and 10,694 separately labelled V_val
diagnostic rows, built in 82.7 seconds across ranks. At 17:12:47 UTC router training reached epoch 1,
global step 201 with finite route loss 1.6598; all four GPUs were active. The 353 MiB cache at
`outputs/split_seed42/motif_dictionary/shared_seed0.routes.pt` is reusable by later probe/restart processes.

Oracle output: `outputs/split_seed42/motif_dict_oracle/`. Driver log:
`outputs/logs/motif_dict_oracle_20260919.log`; fan-out log: `diagnostic.log` inside the output directory.
At 17:09:00 UTC the first merged artifact, `scores/true_val_cls.npz`, was written; the driver advanced
to `true / val_topology` with all four GPUs active. This establishes a real scoring launch, not oracle
completion or evidence that the compressed dictionary helps.

All three lanes run from the same implementation commit. The H content-only control is queued in
its live chain, not yet trained. No ongoing Codex monitor or automatic method-adoption job is installed.

# Joint KD Trial 5 learning curves

Source: `outputs/b1_kd_rank_rep_hpo/trial_005/metrics.jsonl` on the H20 checkout,
retrieved 2026-09-06. `learning_curves.csv` preserves the 25 logged epochs and
leaves unmeasured topology values blank. Figures connect measured topology points.

Published checkpoint: epoch 20, ID `138e492bc4d2fe44`; seed 0.
Weights: rank `0.014055034704056047`, distribution `5.781033282385391`,
representation `0.09611862506526807`; context bank `h2ns3`, margin `0.1`.

Component panels show unweighted losses. Train total is logged `train_loss` plus
`train_kd_loss`; validation total is logged `val_total_loss`. Train and validation
retain their original stream reductions. These curves contain no held-out test data.

Reproduce from the repository root:

```bash
rtk proxy .venv/bin/python docs/results/kd_rank_rep_hpo/plot_learning_curves.py
```

## Held-out test

[test_report.json](test_report.json) is the completed direct `hpc/run.sh test`
report for the same epoch-20 checkpoint (`test_protocol_v7`). Verified on
2026-09-06: terminal log reports the file written at 06:14:13 (server log time),
all four merged score artifacts exist, no test workers remain, all four GPUs are
idle, and no `failure.json` exists. Direct testing does not write the pipeline's
`test_complete.json`. Remote code: `3a36b94512a2608549cbc5fcbed0150d6ff193d6`.

Classification uses the validation max-F1 logit threshold −1.7578125; topology
replays the validation-selected logit threshold 2.671875 on test sampled sets.
The report's final rescored validation metrics differ slightly from the epoch
telemetry in the CSV; retain each with its original provenance.

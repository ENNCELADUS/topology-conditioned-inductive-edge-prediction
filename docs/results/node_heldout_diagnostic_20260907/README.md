# Node-held-out KD1 threshold-transfer diagnostic (2026-09-07)

Raw reports copied from the retired H20 checkout `/2023533015/topology-node-heldout-20260907`
(branch `codex/node-heldout-diagnostic-20260907`) before that checkout was deleted on
2026-09-08. The run used the 20% positive budget from root `node_007630` with every
V_val-touching pair dropped (30,594 training positives) and reused the PMA1 teacher that
had been trained with exposure to those V_val nodes, so it is a student-only diagnostic.

- `threshold_transfer.json`: V_val selection surface and the held-out replay of the frozen
  thresholds (edge metrics plus the five topology numbers).
- `kd_logit_test_report.json`: the student's held-out test report.

Headline replay: AUROC 0.7222, AUPRC 0.7494, F1 0.6879, GS 0.4255, RD 0.6779, MMD ratios
5.78 / 5.14 / 9.92. It motivated the node-held-out rule in `docs/03-experiments.md` §1.1;
it is not evidence for the current split (root `node_002696`, 10% budget).

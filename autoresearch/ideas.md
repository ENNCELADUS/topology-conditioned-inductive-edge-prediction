# Ideas backlog

Deferred hypotheses and proposed program.md changes. Maintained by the operator agent; consumed
at proposal time. Nothing here is a commitment.

- kd_logit: grid response is monotone to w=100 (GS/AUPRC up, degree MMD worsening) — probe the
  w=1000 bracket and, separately, whether stronger regularization claws back degree MMD at w=100.
- kd_rank: dump strict-LLP context targets, freeze `distill.context_targets_path`, then record a
  fresh baseline; the retired incidental-batch grid result is non-comparable provenance only.
- kd_gram/kd_rep campaigns are gated on the PMA(1) teacher decision (2026-08-31): parity-check the
  single-seed teacher vs 4-seed `full_ego_teacher_kd`, dump a PMA1 target bank, rerun `kd_gram_w1`
  and `kd_rep_w0p1` on it, and freeze each campaign's `distill.targets_path` to the winning bank
  before recording its baseline row.

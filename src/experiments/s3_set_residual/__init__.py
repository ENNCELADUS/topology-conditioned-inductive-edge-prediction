"""S3 — B0+set-residual diagnostic: does set context add value beside the frozen pair scorer?

Plan: docs/tmp/s3_set_residual_plan.md. Modules: `data` (region corpus, V_val
masking, fp32 base-logit cache), `model` (set-conditioned residual, res/pair/diag),
`train` (loop + V_val selection), `evaluate` (paired test-bucket readouts), `cli`.
"""

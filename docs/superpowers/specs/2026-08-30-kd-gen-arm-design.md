# kd_gen — Generative Seed-KD Arm (B1)

2026-08-30. Replaces `kd_d9` wholesale. Approved through brainstorming + external review; the two
review-mandated changes (probability-space marginalization, restored warmup) and the two confirmed
engineering deviations (sg-based warmup instead of hard gate-zero; dedicated fusion stack because the
B1 student runs `mixing.mode: none`) are baked in below.

## 1. Purpose

The four plain-KD arms (`kd_logit`/`kd_rank`/`kd_gram`/`kd_rep`) transfer teacher signals into the
existing student function. `kd_gen` instead *generates* the teacher's topology embedding — the 4
GRIT+PMA seed tokens of the true full-ego graph around (u,v) minus the query edge — from
`(x_u, x_v)` alone, and fuses the generated tokens into the pair readout as gated residual context.
The deployed forward reads exactly `(x_u, x_v)`; teacher seeds are training-time supervision only,
so the strict task contract holds. The arm tests the KD survey's escape hypothesis: when plain KD
plateaus, the win comes from a structural channel, not reweighting.

## 2. Arm identity

- Distill arm name `kd_gen`, single legality group `{w_gen}` in `DistillConfig` (legal-pattern tuple
  and `.arm` map). Generator family is a model-side choice, `model.config.seed_gen.name`:
  `edm_dit` (v1 primary), `imf_dit` (2026 few-step candidate), `det_mdn` (mandatory pointwise
  control). Run/output names carry the family suffix: `kd_gen_edm`, `kd_gen_imf`, `kd_gen_det`.
- Paper-facing name ("Latent Topology Diffusion") is deferred until the topology probe (§10) passes.
  Note the teacher's graph carries role channels only (`FullOracleGenerator.graph_dims() == (5, 1)`,
  no content features), so the seeds are a function of pure local topology by construction; the
  probe demonstrates rather than assumes that the *generated* seeds inherit this.

## 3. Teacher target and normalization

- Target: existing `teacher_seeds` / `val_teacher_seeds` in `kd_row_targets_v1` — symmetrized
  ½(S_ab+S_ba) PMA seeds, fp16 `(n_rows, 4, 512)`. No re-dump, no artifact change.
- `seed_rms_scale`: scalar RMS of `teacher_seeds` over the training block, computed at `KDRowBank`
  build (deterministic from the artifact, hence rank-identical even though the bank builds after
  `accelerator.prepare`), pushed into the model as a persistent buffer so scoring recovers it from
  the checkpoint. All seed tensors (targets, generated samples, fusion inputs, telemetry) live in
  the normalized unit-RMS space; nothing ever de-normalizes. EDM's `sigma_data` is then 1 and is
  never conflated with `seed_rms_scale`.

## 4. Generator seam and families

One module `SeedGenerator` owned by `V3_1`, selected by `seed_gen.name`, entirely inside an
fp32 island (`torch.autocast(..., enabled=False)`; geometry on the bf16 ulp grid quantizes).
Interface: `gen_loss(target, c) -> (loss, stats)` for training; `sample(c, eps) -> S_hat` producing
one `(B, seed_count, seed_dim)` sample per noise draw, deterministic given `eps`.

**Conditioning** (shared, unchanged from D9 and unchanged in legality):
`c = MLP_c([e_u ⊙ e_v ; e_u + e_v ; |e_u − e_v|]) -> cond_dim`, where `e_u`/`e_v` are masked-mean
**detached** trunk encodings. The trunk-side stop-gradient is permanent (D4 lesson: KD must never
reshape the trunk; the trunk trains on task gradient only).

**Shared core**: DiT over the 4 seed tokens — 3 blocks, d=512, 8 heads, MLP ratio 4, AdaLN-Zero
conditioned on `[fourier(c_noise); c]` (the `det_mdn` head drops the noise embedding).

- `edm_dit`: EDM preconditioning `D(x;σ,c) = c_skip·x + c_out·F(c_in·x; c_noise, c)` with the
  standard `c_skip/c_out/c_in/c_noise` at `sigma_data=1`. Training: per-row
  `ln σ ~ N(−0.51, 1.2²)` — EDM's default SNR distribution restated relative to unit data RMS
  (median σ/σ_data ≈ 0.602), not the raw `P_mean=−1.2` — with weight `λ(σ)=(σ²+1)/σ²` on the MSE
  to the target. Sampling: deterministic Heun on the Karras grid, `σ/σ_data ∈ [0.004, 160]`, ρ=7,
  `sampler_steps` K=4 (7 NFEs/sample), initial state `σ_max·ε`.
- `imf_dit`: conditional Improved MeanFlow — average-velocity field `u(S_t, r, t, c)` on the same
  core, trained via the MeanFlow identity (JVP inside the fp32 island); iMF's stability
  modifications are taken from the released reference at implementation time. Sampling: 1 NFE,
  `S_hat = ε − u(ε, 0, 1, c)`.
- `det_mdn`: single diagonal Gaussian head `(μ, log σ²)` over the flattened seeds; loss = Gaussian
  NLL; deploy uses `S_hat = μ` with M=1. This is the conditional-mean control the S2 review faulted
  as missing.

## 5. Fusion: gated seed cross-attention in `_pair_representation`

The B1 student config runs `mixing.mode: none`, so `PairCrossAttention.layers` is empty — there are
no existing cross-attention layers to gate. Fusion is therefore its own stack, `SeedFusion`:
`fusion_layers` (default 2) sublayers applied identically to both token streams before the readout,

    h ← h + tanh(g_l) · MHA(LN(h), LN(S_hat), LN(S_hat))     # per-layer scalar g_l, zero-init

with no dropout anywhere in the branch (nothing consumes RNG) and weights shared between the A and
B streams (order symmetry). It is invoked inside `V3_1._pair_representation` on `encoded_a`/
`encoded_b` before the two `self.cross_attention` readout calls, so `forward` and the packed scorer
share one fusion implementation. At `g_l = 0` the model is functionally identical to the trunk-only
control; an eval-mode test asserts exact numerical equality (bit-identity during training is not
claimed — RNG stream differences are permitted).

## 6. Marginalization: probability space

For M noise draws, per-sample logits `ℓ_m` come from M readout passes (folded into the batch
dimension); the marginal is `p̄ = (1/M) Σ σ(ℓ_m)` — Monte-Carlo marginalization of the latent, not a
logit ensemble. Every downstream contract expects logits, so the model emits

    log p̄     = logsumexp_m(−softplus(−ℓ_m)) − log M
    log(1−p̄)  = logsumexp_m(−softplus(ℓ_m)) − log M
    ℓ_marginal = log p̄ − log(1−p̄)

computed in fp32. BCE (with the recipe's label smoothing) applies to `ℓ_marginal` in training, and
scoring artifacts store `ℓ_marginal`, so the logit-shift threshold calibration is untouched. At
gate zero `ℓ_marginal` equals the trunk logit exactly; at M=1 it reduces to the single fused `ℓ_1`.
Train and deploy run the same computation (P0-2): fresh noise per training step, the fixed
checkpointed `(M, seed_count, seed_dim)` ε buffer at eval — scoring stays RNG-free.

## 7. Training

- Loss: `task BCE(ℓ_marginal) + w_gen · L_gen`, one forward, one backward, same `_row_id` join and
  `scale_ddp_mean_loss` scaling (row-local; every row has exactly one target). `L_gen` stays on for
  the whole run as the teacher-fidelity anchor.
- Warmup (review-mandated, one knob): `joint_warmup_frac` (distill field, default 0.1). Warmup
  epochs = `ceil(frac × optim.epochs)`; before the boundary the fusion consumes `sg(S_hat_m)`
  (sampler under no-grad), so no task gradient reaches the generator — it pretrains purely on
  `L_gen` — while `SeedFusion` gates/weights train against frozen-semantics samples, keeping every
  parameter live every step (no `find_unused_parameters`). At the boundary the rank-identical
  `joint_stage` flag flips at epoch start: the `sg` drops, task BCE backpropagates through the
  sampler into the generator, and the generator param group's LR switches to `gen_lr_scale`
  (default 0.1) × base — the same flag/param-group machinery as D9's
  `_set_pair_latent_training_stage`, reworked and renamed for `kd_gen`.
- Guards: both-directions `seed_gen ⇔ w_gen > 0` in `KDRowBank` (DDP never-grad'ed-params guard),
  seed shape validated against the manifest's `seed_count`/`seed_dim` (which stay explicit YAML —
  the model builds before targets load). The first-step-of-epoch `_term_grad_norms` probe covers
  the new term unchanged.

## 8. Scoring integration and controls

- `_score_v3_1_packed` builds `c` from cached encodings, runs the family sampler from the fixed ε
  buffer inside the fp32 island, and calls the shared `_pair_representation` + `output_head` +
  §6 marginalization — byte-identical math to `forward`'s eval path.
- Scoring-time controls on the published checkpoint (`run.sh test` precedent, no retraining):
  (a) gate-zeroed — exact trunk-only model; (b) within-batch shuffle of `S_hat` across pairs —
  keeps capacity and marginals, breaks pair alignment. The `kd_gen_det` run is the third,
  train-time control.

## 9. Telemetry (metrics.jsonl; never blocks; fail-closed only on non-finite)

`kd_gen` (generator loss), `kd_seed_cos` (mean per-slot cosine of a sampled `S_hat` vs target) with
`val_kd_seed_cos` on the val block from fixed ε, per-σ-quartile denoising MSE (`edm_dit` only),
`gen_gate_l{i}` = tanh(g_l) per fusion layer, `mc_prob_std` (std of `σ(ℓ_m)` across samples —
decision-level dispersion), `gen_sample_dispersion` (mean pairwise `S_hat_m` distance divided by
√(seed_count·seed_dim), unit-RMS space). CVAE-era posterior/prior telemetry retires with kd_d9. Standard always-on val metrics
(`val_ece`, `val_brier`, KD val block) unchanged.

## 10. Topology probe (paper-name gate only)

Post-hoc script (`src/experiments/seed_topology_probe.py`): ridge probes from teacher seeds and
from generated seeds (val block, fixed ε) to true ego-graph statistics of each row — endpoint
degrees, common-neighbor count, endpoint clustering — computed on the training structure minus the
query edge; shuffled-seed control alongside. "Latent Topology Diffusion" is claimable only if
generated seeds predict these and the shuffle kills the gain. Never blocks training or claims.

## 11. Config surface

```yaml
model:
  config:
    seed_gen:            # absent => module not built; baseline reproduced exactly
      name: edm_dit      # edm_dit | imf_dit | det_mdn
      seed_count: 4
      seed_dim: 512
      cond_dim: 256
      dit_blocks: 3
      n_heads: 8
      mc_samples: 4      # det_mdn forces 1
      sampler_steps: 4   # edm_dit only
      fusion_layers: 2
distill:
  targets_path: outputs/distill/kd_row_targets_breadth_first
  w_gen: 1.0
  joint_warmup_frac: 0.1
  gen_lr_scale: 0.1
```

Noise-distribution and sampler-grid constants (−0.51, 1.2, 0.004, 160, ρ=7) are module constants,
not config — no speculative knobs. `DistillConfig` delta: add `w_gen` + `joint_warmup_frac`
(float in [0,1)); remove `w_seed`, `w_geom`, `w_kl`, `kl_warmup_steps`, `joint_start_epoch` (and
`_INT_FIELDS`); `gen_lr_scale` stays with warmup-boundary semantics; docstring rewritten.

## 12. Deletions (no compatibility layers)

- `b0_v31.py`: `PairLatentGenerator`, `_SeedRecognition`, `pair_latent_gen` plumbing and
  `pair_latent_gen_parameters` (replaced by `seed_gen` + `seed_gen_parameters`).
- `train_b0.py`: all `kd_d9` branches (optimizer grouping, stage sync, `KDRowBank` loss/attach/
  validation, val diagnostics) reworked to `kd_gen`.
- `distill/losses.py`: `kd_seed_loss`, `kd_seed_gram_loss`, `kd_kl_loss` (the Gate A set losses
  `kd_set_seed_loss`/`kd_set_gram_loss` belong to `train_egostitch` and stay).
- `score_universe.py`: the D9 packed path replaced by the shared-fusion path (§8).
- `configs/b1_kd_d9_breadth_first.yaml` → `b1_kd_gen_edm_breadth_first.yaml` (+ `_det`, `_imf`);
  `tests/test_b0_pair_latent_gen.py`, `test_b1_kd_d9_config.py`, `test_train_b0_d9.py` → `kd_gen`
  equivalents; `docs/tmp/b1_kd_d9_pair_latent_generation_design.md` deleted (superseded here);
  `docs/results/b1_kd_arms.md` arm list updated.
- `src/distill/artifacts.py` and `teacher_targets.py`: unchanged.

## 13. Tests

Eval-mode exact equality at gate zero vs the control model; `ℓ_marginal` equals the trunk logit
when all `ℓ_m` coincide and matches a naive fp64 computation on random logits; no task gradient
reaches generator parameters before the warmup boundary and does after (autograd probe); packed
scorer output equals `forward` eval output on identical inputs; config legality (`{w_gen}` alone;
`seed_gen ⇔ w_gen` both directions; seed-shape/manifest mismatch raises); loader requires
`teacher_seeds` when `w_gen > 0`; DDP all-params-live in both phases; `det_mdn` forces M=1.

## 14. Run plan

1. Verify the H20 `kd_row_targets_breadth_first` manifest carries `teacher_seeds` +
   `seed_symmetry`; no re-dump.
2. After the HPO grid drains: `hpc/run.sh train configs/b1_kd_gen_edm_breadth_first.yaml` and the
   `kd_gen_det` control — publish → automatic held-out test; scoring controls via `run.sh test`.
3. `kd_gen_imf` A/B once the EDM harness is validated end to end: same fidelity/AUPRC/GS at 1 NFE
   retires the 28-NFE EDM deploy path.
4. Topology probe post-hoc on the published checkpoint (§10).
5. A `kd_gen` HPO grid (`w_gen`, M, K, `fusion_layers`) is future work, per the autoresearch spec's
   "kd_d9 is out until it has a grid of its own".

## 15. Interpretation contract (standard claim rules apply)

- Gates stay ≈ 0: structural no-op — no evidence about the latent.
- Gain over control with both scoring controls degrading and healthy `val_kd_seed_cos`: the
  teacher-grounded generated topology latent is useful — the headline positive.
- Gain surviving the shuffle control: capacity artifact, not teacher semantics.
- `edm`/`imf` ≈ `det`: pointwise prediction suffices; the distributional claim dies even if the arm
  wins. `edm`/`imf` > `det` with live `gen_sample_dispersion` and material `mc_prob_std`:
  marginalization over generated topology matters — the multimodality evidence.
- No gain with high `val_kd_seed_cos`: cannot separate "latent uninformative" from "fusion
  insufficient"; report both numbers, claim neither.

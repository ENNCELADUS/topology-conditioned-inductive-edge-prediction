# kd_gen — Generative Topology-Latent KD Arm (B1)

2026-08-30, revision 2. Replaces `kd_d9` wholesale; supersedes the 4-seed draft per user review —
single-seed PMA teacher, 512-D latent target, residual-MLP denoiser, pooled residual adapter; the
4×512 formulation is demoted to a second-stage capacity ablation.

## 1. Purpose

The four plain-KD arms transfer teacher signals into the existing student function. `kd_gen`
instead *generates* the teacher's topology latent from `(x_u, x_v)` alone and fuses it into the
pair representation as gated residual context; the teacher is training-time supervision only, so
the strict task contract holds. The core question: **can endpoint features generate the single
task-sufficient topology latent an oracle GRIT encoder extracts from the hidden ego graph?**

## 2. Arm identity

- Distill arm `kd_gen`, single legality group `{w_gen}` in `DistillConfig`. Generator family is
  model-side, `model.config.topo_gen.name`: `edm` (v1 primary), `imf` (1-NFE candidate), `det_mse`
  (mandatory pointwise control). Runs: `kd_gen_edm`, `kd_gen_imf`, `kd_gen_det`.
- Paper-facing name ("Latent Topology Diffusion") is deferred until the topology probe (§11)
  passes. The teacher's graph carries role channels only (`graph_dims() == (5, 1)`), so the latent
  is pure topology by construction; the probe must show the *generated* latent inherits this.

## 3. Stage 0 (mandatory): single-seed PMA teacher

The current teacher (`egostitch_e2e_v3_full_ego_teacher_breadth_first.yaml`, `seeds: 4`,
`conditioning_mode: pooled_adapter`, `w_rel: 0.0`) predicts from `GraphEmbedding.pooled`, the
*mean* of the 4 PMA seeds. The slots are not load-bearing (`S₁+δ, S₂−δ` leaves the pooled input
unchanged): a 4-slot target imposes a non-task-identifiable decomposition. `seeds: 1` removes it —
PMA does the whole N×512→1×512 compression; the dumped latent is what the teacher predicts from.

- New config `egostitch_e2e_v3_full_ego_teacher_pma1_breadth_first.yaml`: identical to the
  incumbent except `encoder.seeds: 1`. Train on H20; compare against the PMA(4) teacher on V_val —
  AUPRC together with the five topology numbers (BFS-macro GS/RD, three MMD ratios). Freezing is
  an operator judgment from `metrics.jsonl` (no code gate): proceed only if the oracle ceiling is
  essentially preserved. Parity is itself reportable ("one learned graph-summary query suffices");
  a material drop stops the route and puts the PMA(4)-mean target back on the table.

## 4. KD target, dump, and normalization

- Re-dump `kd_row_targets_pma1_breadth_first` from the frozen PMA(1) checkpoint (existing dumper
  CLI); the 4-arm HPO keeps its PMA(4) artifact untouched. The KD target is the existing
  `teacher_rep` array — symmetrized ½(pooled_ab + pooled_ba), now identically the single PMA
  seed, fp16 `(n_rows, 512)` — plus the `val_` mirror and `teacher_logit`.
- The optional `teacher_seeds`/`val_teacher_seeds` arrays, seed slicing, and the `seed_symmetry`
  audit in `teacher_targets.py`/`artifacts.py` are deleted — no consumer remains (Gate A distills
  from a live teacher forward, not from this artifact).
- `latent_rms_scale`: scalar RMS of `teacher_rep` over the training block, computed at
  `KDRowBank` build (deterministic from the artifact, hence rank-identical despite the bank
  building after `accelerator.prepare`), pushed into the model as a persistent buffer so scoring
  recovers it from the checkpoint. All latents live in unit-RMS space; nothing de-normalizes;
  EDM's `sigma_data` is then 1 and never conflated with `latent_rms_scale`.

## 5. Generator: `TopoLatentGenerator`

One module owned by `V3_1`, selected by `topo_gen.name`, inside an fp32 island (autocast
disabled). Interface: `gen_loss(target, c) -> (loss, stats)`; `sample(c, eps) -> h_hat`, one
`(B, 512)` latent per draw, deterministic given `eps`. Conditioning (unchanged, unchanged in
legality): `c = MLP_c([e_u ⊙ e_v ; e_u + e_v ; |e_u − e_v|]) -> cond_dim` from masked-mean
**detached** trunk encodings — the D4 boundary stays permanent.

**Core: conditional residual-MLP denoiser**, `D(x; σ, c): R⁵¹² → R⁵¹²` — `blocks` (default 4)
AdaLN-Zero residual MLP blocks (LN with scale/shift from `[fourier(c_noise); c]` →
Linear 512→1024 → GELU → Linear 1024→512 → per-block zero-init gate). The 4-token DiT is dropped:
self-attention over a single token adds machinery without the token-interaction reason DiT was
selected for.

- `edm`: EDM preconditioning `D(x;σ,c) = c_skip·x + c_out·F(c_in·x; c_noise, c)` at `sigma_data=1`.
  Training: per-row `ln σ ~ N(−0.51, 1.2²)` — EDM's default SNR distribution restated relative to
  unit data RMS (median σ/σ_data ≈ 0.602) — with weight `λ(σ)=(σ²+1)/σ²` on the MSE. Sampling:
  deterministic Heun, Karras grid `σ ∈ [0.004, 160]`, ρ=7, K=4 steps (7 NFEs), state `σ_max·ε`.
- `imf`: conditional Improved MeanFlow — average-velocity field `u(x_t, r, t, c)` on the same
  core, trained via the MeanFlow identity (JVP inside the fp32 island); iMF's stability
  modifications from the released reference at implementation time. Sampling: 1 NFE,
  `h_hat = ε − u(ε, 0, 1, c)`.
- `det_mse`: the same residual-MLP core conditioned on `c` only (matched capacity, no noise
  embedding); `h_hat = f(c)`, loss `‖h_hat − h*‖²/512`, M=1 — the pure conditional-mean control.
  Gaussian NLL is deliberately NOT the control: learned σ² down-weights hard dimensions,
  confounding objective choice with stochasticity; an optional `diag_gauss` arm is out of scope.

## 6. Fusion: pooled residual adapter in `_pair_representation`

After the readout produces `z = pair_repr`, each sample fuses as

    Δz  = W_up · GELU(W_down · h_hat)        # W_down: 512 → adapter_dim (128), W_up zero-init
    z'  = z + tanh(g) · Δz                   # scalar g, init 1.0

mirroring the teacher's own `pooled_adapter` conditioning mode. **Exactly one zero factor:** with
`W_up = 0` *and* `g = 0`, both gradients vanish identically (∂L/∂W_up ∝ tanh(0), ∂L/∂g ∝ Δz = 0)
and the branch is a permanent saddle — so `W_up` zero-init carries the init identity and `g`
starts at 1. No dropout in the branch. It lives inside `V3_1._pair_representation` — `forward` and
the packed scorer share one implementation, and the readout runs **once** per pair; only the
adapter and `output_head` repeat per sample. At `W_up = 0` the model is functionally identical to
the trunk-only control (eval-mode exact-equality test; training bit-identity is not claimed).

## 7. Marginalization: probability space

For M noise draws, per-sample logits `ℓ_m = output_head(z'_m)`; the marginal `p̄ = (1/M) Σ σ(ℓ_m)`
is Monte-Carlo marginalization of the latent, not a logit ensemble. Downstream contracts expect
logits, so the model emits

    log p̄     = logsumexp_m(−softplus(−ℓ_m)) − log M
    log(1−p̄)  = logsumexp_m(−softplus(ℓ_m)) − log M
    ℓ_marginal = log p̄ − log(1−p̄)

computed in fp32. BCE (with the recipe's label smoothing) applies to `ℓ_marginal`; scoring
artifacts store it, so logit-shift threshold calibration is untouched. At branch-zero it equals
the trunk logit exactly; at M=1 it reduces to `ℓ_1`. Train and deploy run the same computation:
fresh noise per training step, a fixed checkpointed `(M, 512)` ε buffer at eval — RNG-free.

## 8. Training

- Loss: `task BCE(ℓ_marginal) + w_gen · L_gen` — one forward/backward, same `_row_id` join,
  `scale_ddp_mean_loss` (row-local; one target per row); `L_gen` stays on as the fidelity anchor.
- Warmup, one knob: `joint_warmup_frac` (distill field, float in [0,1), default 0.1). Warmup
  epochs = `ceil(frac × optim.epochs)`; before the boundary the adapter consumes `sg(h_hat_m)`
  (sampler under no-grad), so no task gradient reaches the generator — it pretrains purely on
  `L_gen` — while the adapter and `g` train against frozen-semantics latents, keeping every
  parameter live every step (no `find_unused_parameters`). At the boundary the rank-identical
  `joint_stage` flag flips at epoch start: the `sg` drops, task BCE backpropagates through the
  sampler, and the generator param group's LR switches to `gen_lr_scale` (default 0.1) × base —
  D9's flag/param-group machinery reworked and renamed.
- Guards: both-directions `topo_gen ⇔ w_gen > 0` in `KDRowBank` (DDP never-grad'ed-params guard);
  `latent_dim` is explicit YAML (the model builds before targets load) cross-checked against the
  manifest's `rep_dim`. The first-step-of-epoch `_term_grad_norms` probe covers the new term.

## 9. Scoring integration and controls

- `_score_v3_1_packed` builds `c` from cached encodings, runs the family sampler from the fixed ε
  buffer inside the fp32 island, and calls the shared `_pair_representation` + §7 marginalization
  — byte-identical math to `forward`'s eval path.
- Scoring-time controls (`run.sh test` precedent, no retraining): branch-zeroed (exact trunk-only
  model) and within-batch `h_hat` shuffle across pairs (keeps capacity and marginals, breaks pair
  alignment). `kd_gen_det` is the third, train-time control.

## 10. Telemetry (metrics.jsonl; never blocks; fail-closed only on non-finite)

`kd_gen` (generator loss), `kd_latent_cos` (cosine of a sampled `h_hat` vs target) plus
`val_kd_latent_cos` from fixed ε, per-σ-quartile denoising MSE (`edm` only), `gen_gate` = tanh(g),
`gen_branch_ratio` = ‖Δz‖/‖z‖, `mc_prob_std` (std of `σ(ℓ_m)` across samples), and
`gen_sample_dispersion` (mean pairwise `h_hat_m` distance / √512). Val metrics unchanged.

## 11. Topology probe (paper-name gate only)

Post-hoc script (`src/experiments/seed_topology_probe.py`): ridge probes from teacher and
generated latents (val block, fixed ε) to true ego-graph statistics — endpoint degrees,
common-neighbor count, endpoint clustering — on the training structure minus the query edge. The
name is claimable only if generated latents predict these and a shuffled-latent control kills it.

## 12. Config surface

```yaml
model:
  config:
    topo_gen:            # absent => module not built; baseline reproduced exactly
      name: edm          # edm | imf | det_mse
      latent_dim: 512
      cond_dim: 256
      blocks: 4
      adapter_dim: 128
      mc_samples: 4      # det_mse forces 1
      sampler_steps: 4   # edm only
distill:
  targets_path: outputs/distill/kd_row_targets_pma1_breadth_first
  w_gen: 1.0
  joint_warmup_frac: 0.1
  gen_lr_scale: 0.1
```

Noise and sampler constants (−0.51, 1.2, 0.004, 160, ρ=7) are module constants, not config.
`DistillConfig` delta: add `w_gen` + `joint_warmup_frac`; delete `w_seed`/`w_geom`/`w_kl`,
`kl_warmup_steps`, `joint_start_epoch`, `_INT_FIELDS`; `gen_lr_scale` keeps warmup semantics.

## 13. Deletions (no compatibility layers)

- `b0_v31.py`: `PairLatentGenerator`, `_SeedRecognition`, `pair_latent_gen` plumbing and
  `pair_latent_gen_parameters` (replaced by `topo_gen` + `topo_gen_parameters`). `train_b0.py`:
  all `kd_d9` branches (optimizer grouping, stage sync, `KDRowBank`, val diagnostics) → `kd_gen`.
- `distill/losses.py`: `kd_seed_loss`, `kd_seed_gram_loss`, `kd_kl_loss` (the Gate A set losses in
  `train_egostitch` stay). `distill/artifacts.py` + `teacher_targets.py`: `teacher_seeds` support
  and `seed_symmetry` audit removed. `score_universe.py`: D9 packed path → shared path (§9).
- `configs/b1_kd_d9_breadth_first.yaml` → `b1_kd_gen_edm_breadth_first.yaml` (+ `_det`, `_imf`);
  the three D9 test files → `kd_gen` equivalents; the D9 design doc in `docs/tmp/` deleted;
  `docs/results/b1_kd_arms.md` arm list updated.

## 14. Tests

Eval-mode exact equality at branch zero vs control; `ℓ_marginal` identities (all-`ℓ_m`-equal ⇒
trunk logit; matches naive fp64 on random logits); adapter saddle guard (gradients nonzero at
init); no task gradient into the generator before the warmup boundary, flowing after; packed
scorer equals `forward` eval on identical inputs; config legality (`{w_gen}` alone; `topo_gen ⇔
w_gen`; `latent_dim`/`rep_dim` mismatch raises); `det_mse` forces M=1; DDP all-params-live in both
phases; PMA(1) `grit_gmt` emits `tokens (B, N+1, d)` with `pooled` equal to the seed.

## 15. Run plan (experimental hierarchy)

1. Train the PMA(1) full-ego teacher; compare V_val ceiling vs PMA(4) (§3). Freeze on parity.
2. Dump `kd_row_targets_pma1_breadth_first` from the frozen PMA(1) checkpoint.
3. `kd_gen_det` (the `det_mse` MSE-regression control), then `kd_gen_edm` — each publish →
   automatic held-out test; scoring controls via `run.sh test`.
4. `kd_gen_imf` A/B once the EDM harness is validated: parity at 1 NFE retires the 7-NFE deploy.
5. Topology probe post-hoc on the published checkpoint (§11).
6. Compare against `kd_rep` (same latent as loss target vs generated channel); an optional
   teacher-matched `kd_rep` re-run on the PMA(1) artifact settles cross-teacher disputes.
7. Future, out of scope: a `kd_gen` HPO grid (`w_gen`, M, K, `adapter_dim`); 4×512 seed-set
   generation as a capacity ablation only if the single-latent formulation succeeds.

## 16. Interpretation contract (standard claim rules apply)

- PMA(1) ≈ PMA(4) at Stage 0: one learned graph-summary query suffices — reportable on its own.
- Branch stays ≈ 0 (`gen_branch_ratio`): structural no-op — no evidence about the latent.
- Gain over control with both scoring controls degrading and healthy `val_kd_latent_cos`: the
  teacher-grounded generated topology latent is useful — the headline positive. Gain surviving
  the shuffle control: capacity artifact, not teacher semantics.
- `edm`/`imf` ≈ `det_mse`: pointwise prediction suffices; the distributional claim dies even if
  the arm wins. `edm`/`imf` > `det_mse` with live dispersion and material `mc_prob_std`:
  marginalization over generated topology matters — the multimodality evidence.
- `kd_gen` > `kd_rep`: the generated channel beats the same latent as a loss target. No gain with
  high `val_kd_latent_cos`: latent vs fusion blame is inseparable — report both, claim neither.

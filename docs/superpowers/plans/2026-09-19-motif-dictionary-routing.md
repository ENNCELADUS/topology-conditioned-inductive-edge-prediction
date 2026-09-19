# Motif Dictionary Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. The owner explicitly requested several GPT-5.6-sol medium implementers, primary-agent review, and three-container launch after integration.

**Goal:** Implement and launch the approved dictionary-oracle, sequence-router and head-adaptation experiments.

**Architecture:** A fixed training-only dictionary emits the existing four prompt fields. A separate model family learns equivariant routing; the existing motif family gains a head-only training policy. Existing production training, scoring, threshold selection and publication remain the execution path.

**Tech Stack:** Python, PyTorch, NumPy, Accelerate DDP, existing YAML configuration and pytest.

**Spec:** `docs/superpowers/specs/2026-09-19-motif-dictionary-routing-design.md` (owner approved).

## Global Constraints

- `(x_u, x_v) -> edge(u, v)`; dictionary and target statistics use legal training rows only.
- 16 real representatives before swap expansion; attachment-only category is distinct from empty.
- Lane R trains slot reads/router by closure-balanced KL only; lane H trains only `base.output_head`.
- Stage I reader, four-field prefix and base are frozen; scoring checkpoints contain all inference constants.
- 15 epochs, seed 0, no early stopping; existing five-metric V_val selection and `--skip-test`.
- No new digest enforcement, qualification gates or test-driven model selection.
- Preserve existing user changes in `figures/` and `docs/artifacts/`; workers do not commit, push or launch.
- Work in the existing feature branch. Primary owns review, integration, Git synchronization and launch.

## Review Focus

- Scoring without artifact files: checkpoint round trips must retain every inference constant (Tasks 2, 4).
- All-self or one-category rank batches: finite zero-connected KL and globally correct reductions (Tasks 1, 3).
- Frozen modules with AdamW/dropout: only intended parameters change, frozen modules stay in eval (Task 2).
- Swap and fan-out: target orientation and global route transplantation are independent of batch/shard layout (Tasks 1, 2, 4).
- Production dispatch: family parsing, publication, validation and precision checks recognize the new family (Tasks 3, 4, 5).

## Shared interfaces and ownership

Task 1 owns `src/data/motif_dictionary.py`, `src/experiments/motif_dictionary_build.py` and their new tests.
Task 2 owns `src/model/egostitch/classifier/motif_prompt.py`, new `motif_dictionary.py` beside it, and model tests.
Task 3 owns `src/train_b0.py`, training/config/pipeline integration, four new YAML configs and training tests.
Task 4 owns `src/score_universe.py`, `src/score_fanout.py`, scoring/evaluation integration, new dictionary diagnostic driver and scoring tests.
Primary owns this plan, spec status, HPC drivers, integration tests/review and result/launch notes.

These shared contracts are the starting interfaces; agents must communicate any required change to all consumers before adopting it:

```python
# src/data/motif_dictionary.py
@dataclass
class DictionaryArtifact:
    weights: Tensor       # K x 96, float32
    tokens: Tensor        # K x 4 x width, float32; u,v,rel,cnt order
    scales: Tensor        # 4 scalar standard deviations; u/v tied
    temperature: float
    mean_weights: Tensor  # K, swap invariant
    swap_index: Tensor    # K int64, involution
    metadata: dict[str, object]

def load_dictionary(path: Path) -> DictionaryArtifact: ...
def save_dictionary(artifact: DictionaryArtifact, path: Path) -> None: ...
def routing_targets(tokens: Tensor, artifact: DictionaryArtifact) -> Tensor: ...
def mix_dictionary_tokens(routes: Tensor, tokens: Tensor) -> Tensor: ...

# src/model/egostitch/classifier/motif_dictionary.py
class V3_1MotifDictionary(nn.Module):
    # May reuse/subclass the motif model where this preserves explicit family boundaries.
    def __init__(self, *, base: Mapping[str, object],
                 motif_prompt: Mapping[str, object],
                 motif_dictionary: Mapping[str, object]) -> None: ...
    def install_dictionary(self, artifact: DictionaryArtifact) -> None: ...
    def predict_routes(self, encoded_a: Tensor, encoded_b: Tensor,
                       lengths_a: Tensor, lengths_b: Tensor) -> Tensor: ...
    def logits_from_routes(self, encoded_a: Tensor, encoded_b: Tensor,
                           lengths_a: Tensor, lengths_b: Tensor, *,
                           routes: Tensor, return_pair_repr: bool = False) -> Tensor: ...
```

Dictionary config: `mode: oracle|sequence`, `artifact_path`, `dictionary_size`, `slot_checkpoint`.
Constructor must not read checkpoint/artifact files; training/build initialization loads source modules;
scoring reconstruction loads embedded config and state only. Target batch key: `motif_route_targets`;
closure category key: `motif_route_nonempty`; nonself key: reuse `motif_nonself` if that is the existing
template mask constant, otherwise import the actual existing constant. Model returns `route_loss_rows`
and `predicted_routes`; trainer owns globally balanced reduction. Model owner must tell training/scoring
owners the exact implemented interface before completion.

Head policy keys inside `motif_prompt`: `training_policy: default|head_only`,
`head_prompt_enabled: bool`, `init_checkpoint: str`; default preserves old behavior.
Training loads the final wave3 source checkpoint, not only the Stage I bundle. Old teacher/graph losses
must not run in head-only training. Source modules are loaded by name with actual missing keys reported.

### Task 1: Dictionary construction and targets

- [ ] Implement the artifact and pure tensor functions above. Encode target distance as
  `((target[:, None] - bank[None]) / scales[None, None, :, None]).square().mean((-1,-2))`;
  use `softmax(-distance / temperature)` and `einsum('bk,kfd->bfd', routes, tokens)` in FP32.
- [ ] Add synthetic tests that assert `q(swapped_tokens) == q(tokens)[:, swap_index]`, a one-hot
  route exactly returns its token, and an attachment-only template belongs to neither empty nor bridge.
- [ ] Implement deterministic category sampling and real medoid representatives exactly as spec §3;
  tied endpoint scales, temperature from training rows, swap-complete bank and training mean.
- [ ] Add a production builder CLI using the existing split/corpus/compiler and Stage I checkpoint;
  inputs `--config`, `--stage1-checkpoint`, `--output`, `--device`; compile only legal training rows.
  Cache row targets with explicit pair orientation when useful. Use bounded GPU encoding batches and
  progress logging; do not load feature packs or perform PPI forwards to build graph tokens.
- [ ] Run new data/builder tests with `rtk proxy .venv/bin/python -m pytest <new-test-files> -n0`.
  Return artifact format, CLI example, tests and any consumer-interface changes to primary.

### Task 2: Dictionary model and head-only behavior

- [ ] Factor `V3_1MotifPrompt.logits_from_tokens(...)` from its existing graph-to-logit method without
  changing old graph behavior. Test old graph forward against the same tokens passed directly.
- [ ] Implement the repaired slot encoder and shared MLP router. Compute
  `logits = (h(S) + h(swap_slots(S))[:, swap_index]) / 2`, then FP32 softmax/mixing.
  Test endpoint exchange, output-logit symmetry and deterministic frozen producer eval mode.
- [ ] Persist dictionary constants as buffers and reconstruct with config `dictionary_size` alone.
  Provide `install_dictionary`, `predict_routes`, `logits_from_routes`, model forward and explicit
  oracle target handling. Reject graph-only interventions and missing oracle targets.
- [ ] Return per-row KL for training with an autograd-connected zero on self-only batches; do not
  return a task-gradient objective for R. Coordinate reduction with Task 3 before finalizing.
- [ ] Add head-only policy: `requires_grad` only on output_head; preserve eval on every frozen
  producer and training mode on the head. Content-only mode disables prefix contribution during
  both training and scoring. Test actual optimizer steps, including nonzero weight decay.
- [ ] Run model tests and old motif regression tests. Return exact loading and scoring APIs.

### Task 3: Production training integration and configs

- [ ] Register/resolve/build `v3_1_motif_dictionary` with embedded base, motif and dictionary config.
  Load Stage I interface, wave3 slot modules and artifact only when initializing training.
  Load complete wave3 state for head-only mode; preserve published checkpoint reconstruction.
- [ ] Build row-aligned training route targets from the legal template table, separately from V_val
  diagnostic targets. No validation target is attached to the deployable inference path.
- [ ] Train R on globally reduced closure-balanced KL only, correctly compensating DDP averaging
  and microbatch splits. Test unequal rank category counts and an all-self microbatch; run task BCE
  only as reporting/validation and skip the structural training stream for R.
- [ ] Keep H task/structural objective and skip graph/teacher supervision and gradient probes that
  would reopen frozen modules. Run the same production optimizer grouping for both H controls.
- [ ] Extend per-epoch validation/publication/config metadata and needed pipeline family checks.
  Add route KL, mean-predictor KL, token error, route entropy/mean/dispersion telemetry without
  quality gates or test data. Preserve existing checkpoint ranking and precision.
- [ ] Create configs `motif_dict_router.yaml`, `motif_dict_oracle.yaml`, `motif_wave3_head.yaml`,
  `motif_wave3_head_content.yaml` under `configs/split_seed42`; use approved budgets and source paths.
- [ ] Run targeted training/config tests, including one tiny real optimizer step per trainable lane.
  Return full launch commands and any outstanding scoring API needs.

### Task 4: Scoring, fan-out and oracle diagnostic

- [ ] Extend checkpoint builders and scorer-family/precision handling for the dictionary family.
  For sequence mode call `predict_routes` then `logits_from_routes`; oracle mode derives q* only
  after existing oracle diagnostic permission/run-kind checks. Keep encoder caching and FP32 pair pass.
- [ ] Implement global `shuffle_route` using the existing whole-universe source permutation rather
  than within-shard shuffling. Test contiguous shards against a serial reference. Support training
  `mean` and `gates_off`; reject `permute_closure` and `rewire_bridge` for dictionary models.
- [ ] Verify checkpoint scoring with artifact/source paths absent and no training graph available.
  Update fan-out CLI forwarding and artifact precision validation consistently.
- [ ] Add `src/experiments/motif_dictionary_diagnostic.py`: build an oracle dictionary checkpoint
  from shared artifact/source Stage I state, score V_val true/oracle-mixture/mean/gates_off using
  production fan-out, and analyze cached scores with existing threshold/topology functions.
  Do not run held-out test. Write per-case metrics and `complete.json`/`failure.json` for this driver.
- [ ] Expose CLI `--artifact`, `--stage1-checkpoint`, `--output-dir`, `--pack-dir`, `--data-root`,
  `--strategy`; provide readable logs, non-self edge metrics and matched-density GS/MMD.
- [ ] Run scorer/driver regression tests. Return the runnable three-lane diagnostic command.

### Task 5: Primary integration, review and launch

- [ ] Review each task's diff and reported tests against the spec; fix concrete integration failures.
  Inspect strict scoring boundaries, frozen parameter sets, row orientation and actual gradient flow.
- [ ] Run relevant model/data/trainer/scorer tests together, changed-path Ruff and repo-required mypy;
  distinguish pre-existing type failures with a baseline if needed. Check final diff and user dirt.
- [ ] Read-only audit all three H20 endpoints: hostname, HEAD/status, GPU count/utilization/processes,
  active outputs and source checkpoints. Use actual GPU counts, not stale allocation assumptions.
- [ ] Commit only task changes; push the feature branch, pull through Git on the shared H20 checkout.
  Build the shared dictionary and validate its finite values, swap map and training-only provenance.
- [ ] Launch R on 30838, H then content-control on 30846, D on 30030, all disconnect-safe with
  separate logs and output directories. Training uses `hpc/run.sh train CONFIG --skip-test`.
- [ ] Verify launcher/worker PIDs, logs and GPU activity from a fresh connection. Record run IDs,
  output paths and real status in a launch note; no held-out test and no invented completion claim.

## Execution decisions and progress

- Owner explicitly approved the spec and requested implementation and launch. No additional plan
  approval interview; preserve the requested agents/model/reasoning and primary review.
- Workers operate in parallel with disjoint file ownership, overriding the skill's sequential-worker
  default because the owner requested several implementers and independent tasks can be isolated.
- No separate reviewer agents: the owner requested review by the primary. Primary performs task and
  integrated review before any production launch.
- Shared model/data/trainer/scorer interfaces above have been checked for overlap; implementation
  consumers must coordinate any signature change before editing their callers.
- Task 1 implemented and primary-reviewed; stage-source rejection and original-index medoid fixes
  included. Worker reports 33 passing data/builder/compiler tests and clean owned-file lint/types.
- Task 2 implemented and primary-reviewed; active-gate, extreme-logit KL and autocast regressions
  included. Worker reports 88 passing motif/dictionary model tests and clean owned-file lint/types.
- Task 3 complete and primary-reviewed: shared targets build on disjoint rank shards and are cached;
  explicit Stage I evaluator/wave3 slot loading, old Stage I loss-buffer independence, true row
  dispersion and correct head learning rate are tested. Added production build/optimizer/checkpoint/
  packed-scoring tests that remove all source/artifact files before scoring.
- Task 4 complete and primary-reviewed: protocol-selected thresholds, dictionary-output-density
  comparisons, active-gate shuffle checks and raw-logit ranking/classification are tested.
- Task 5 local integration complete: 646 integrated tests plus 2 real two/four-rank CPU Gloo tests
  passed; changed-file Ruff, shell syntax and diff checks pass. Full-tree mypy matches exactly the
  baseline 103 errors in 17 files, with no new errors (normalized for line movement).
- Initial H20 audit: all three endpoints idle, four H20s each, shared HEAD `63061a9`; Stage I epoch 11
  and final wave3 epoch 2 verified. Git synchronization, shared dictionary build and launch remain.

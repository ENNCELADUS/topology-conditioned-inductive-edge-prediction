# EgoStitch E2E Stage-1 Screen — Experiment Plan

> **2026-08-03 data-contract disposition:** every 80/20 message/supervision split in
> this historical plan is superseded by spec §9.3. Topology and classification now use
> the same complete train-side positive interactions.

> **2026-07-19 disposition:** this is the historical v1 screen plan. The v1 full arm
> completed only its engineering training pipeline and was training-invalid; it did
> not produce a G5 verdict. Prospective v2 execution is governed by spec §13.19 and
> the v2 DRAFT registration: qualify without test inputs, bind, train `full`, stop if
> invalid, then train the remaining arms; aborted runs publish failure metadata only,
> never `complete.json` or score artifacts.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

> **Revision 2 (2026-07-17), after user review.** Six corrections accepted:
> (1) the pair+topology attribution arm is **trained separately**
> (`permanent_null: content_head`) — restoring design-doc §5 arm 7's own
> "permanent" wording; the same-checkpoint `pair_topology` logit is demoted to
> a diagnostic decomposition. (2) The spec §13.8 checkpoint-selection tie-break
> is re-registered within-checkpoint (`std(full − f_logit)/std(f_logit)`), not
> removed. (3) Primary criteria made exact per metric/operating point.
> (4) Control 6a uses stable-hash keyed permutations (batch/order/shard/GPU
> invariant) and its failure reading is narrowed. (5) Debug smokes are
> physically separated from formal directories and binding is machine-enforced.
> (6) Thresholds: 25% attribution share **accepted** (on the trained arm);
> fixed 0.005 margin **rejected** in favor of a paired-bootstrap condition
> scoped to evaluator stability.

**Goal:** Take the rev-3.0 stitched-topology-conditioned pair encoder from the
frozen-s0 `cut` closeout to a **binding five-arm single-seed Stage-1 screening
verdict** (pass ⇒ Stage-2/E1 preparation; cut ⇒ registered failure reading).

**Architecture:** Three layers executed in order. (A) *Governance landing* —
commit the staged frozen-s0 closeout, rewrite spec §5/§13 to §14 under the
freeze rule, create four configs and a DRAFT registration. (B) *Code* — execute
the committed implementation plan
`docs/superpowers/plans/2026-07-16-egostitch-e2e-conditioned-encoder.md`
Tasks 5–15, plus the extensions defined here (permanent-null training override,
stable-hash 6a scoring control, binding enforcement, paired-bootstrap margin).
(C) *Runtime* — HPC profile smoke in dedicated debug directories → measured
cost table → BIND → **four** formal training runs → **five** scoring passes →
gate → result note.

**Tech Stack:** uv / Python 3.11, PyTorch 2.10.0+cu128, Accelerate DDP
auto-sized over visible NVIDIA H20s via `hpc/run.sh`, numpy/networkx gate
analyses, pytest + ruff + strict mypy.

## Global Constraints

- **Freeze rule:** `docs/05-egostitch-spec.md` is the contract. Every code
  deviation lands in the spec *first*, with a dated change-log line (spec §12).
- **Registration before training, machine-enforced:** formal runs record
  `run_metadata.json['preregistration_sha256']` at start. The worker **refuses
  a formal run unless the registration file carries `status: BINDING`**; DRAFT
  registrations are accepted only together with an explicit `--max-steps`
  debug flag, and debug runs write only to `*_debug` output directories and
  never produce held-out artifacts. The gate refuses anything except
  `status: BINDING` with the sha matching **all four** formal run metadatas.
- **Registered decision constants (user-confirmed 2026-07-17):**
  pathway-attribution share `≥ 25%` computed on the **separately trained**
  pair+topology arm; structure-control condition = paired-bootstrap lower 95%
  bound of `clu(6a) − clu(full) > 0` (evaluator-stability scope only).
- **Single fixed seed 0.** This screen supports no statistical-significance or
  cross-seed-robustness claims; E1/E3 retain ≥ 3 seeds + Holm. The paired
  bootstrap quantifies evaluator sampling stability at fixed seed, nothing more.
- **fp32 pair pass** for all published logits (spec §13.16, provenance
  `egostitch_e2e_pair_fp32_v1`); `validate_score_resolution` guard per array.
- **Edge-level and assembled-graph metrics are always reported together.**
- `src/model/B0.py` is never modified.
- Neutral placeholder naming (Benchmark-A, B0, B0-e2e, …) in all docs.
- Formal training/scoring runs only on the HPC container via `hpc/run.sh`
  (user-owned); local machine runs tests and analyses only.
- Local gotchas: use `.venv/bin/python -m …` (rtk proxy garbles `uv run`
  output); never run two mypy invocations concurrently; two documented
  pre-existing local test failures (`tests/test_b0_attention.py` torch-version,
  `tests/test_e2_ddp_integration.py` rendezvous) are container-only issues.

## Current state (2026-07-17, verified against the working tree)

- Frozen-s0 screen **complete**: binding verdict `cut` (checkpoint
  `56b91c17fa8d3b86`, registration `97e61a7d…`, gate artifacts under
  `outputs/egostitch_stage1/formal_gate/`). Guards passed; all three primary
  dominance checks failed vs `b0_cal_selfdensity`; residual dead within-numerics
  (s0 correlation 0.999979, residual/s0 std ratio 8.0e-4).
- Closeout docs are **staged but uncommitted**: 9 modified files +
  `docs/results/G5-stage1-seed0-20260717.md` +
  `docs/artifacts/2026-07-17-egostitch-status.html` (both untracked).
- Spec §14.3(1) satisfied; §5/§13 **not yet rewritten**; no e2e config, no e2e
  registration, no e2e code.
- The committed implementation plan (Tasks 1–15) exists at
  `docs/superpowers/plans/2026-07-16-egostitch-e2e-conditioned-encoder.md`;
  its Tasks 2–3 (protocol disposition, proposal rev) are already landed
  (commit `c0e2144` + the staged closeout). Its Task 1 → this plan's Task 2;
  its Task 4 → this plan's Tasks 3–4 (extended: four configs, full
  registration skeleton, deferred binding).

## The five registered arms (design doc §5, Stage-1 scope)

| # | Arm | Training run | Scored artifact |
|---|-----|--------------|-----------------|
| 1 | `full` (STE + gated x-attn headline) | run A (`egostitch_e2e_breadth_first.yaml`) | run A `candidate.npz` — array `logits` |
| 2 | `b0_e2e_f_only` (matched pairwise-only) | run B (`…_f_only.yaml`, `permanent_null: all_head`) | run B `candidate.npz` |
| 3 | `pair_topology` (`∅_content_head` **permanent, trained**) | run D (`…_pair_topology.yaml`, `permanent_null: content_head`) | run D `candidate.npz` |
| 4 | `structure_control_6a` (stable-hash within-pair Â/Π shuffle) | — (run A checkpoint) | separate scoring pass with `--scaffold-control shuffle_within_pair` |
| 5 | `p0` (branch dropout off) | run C (`…_p0.yaml`, `p_topo=p_cont=0`) | run C `candidate.npz` |

**Four training runs (A–D), five scoring passes, one gate invocation.** The
full checkpoint's same-checkpoint four-logit decomposition (including its
eval-bypass `pair_topology` array) is published as a **diagnostic** and is not
the registered attribution quantity.

---

### Task 1: Verify and commit the staged frozen-s0 closeout

**Files:**
- Commit (already modified): `CLAUDE.md`, `README.md`,
  `docs/03-experiment-protocol.md`, `docs/04-model-proposal.md`,
  `docs/05-egostitch-spec.md`, `docs/results/E2-pair-to-topology-gap.md`,
  `docs/results/G5-stage1-seed0-20260715.md`,
  `docs/superpowers/plans/2026-07-16-egostitch-e2e-conditioned-encoder.md`,
  `docs/superpowers/specs/2026-07-16-egostitch-e2e-conditioned-encoder-design.md`
- Commit (untracked): `docs/results/G5-stage1-seed0-20260717.md`,
  `docs/artifacts/2026-07-17-egostitch-status.html`

**Interfaces:**
- Produces: a clean tree whose HEAD records the binding `cut` verdict and the
  locked disposition (frozen-s0 → motivating arm + E4 ablation rung; rev-3.0 →
  active G5 build line). Every later task builds on this commit.

- [ ] **Step 1: Verify result-note numbers against the gate artifact**

```bash
.venv/bin/python - <<'EOF'
import json, re
r = json.load(open('outputs/egostitch_stage1/formal_gate/g5_stage1_results.json'))
note = open('docs/results/G5-stage1-seed0-20260717.md').read()
checks = {
    'verdict cut': r['verdict'] == 'cut' and 'cut' in note,
    'checkpoint': '56b91c17fa8d3b86' in note,
    'prereg sha': r['metadata']['preregistration_sha256'][:8] in note,
    'clustering diff': f"{r['criteria']['clustering_mmd_ratio']['mean_diff']:.4f}"[:6].lstrip('-') in note.replace('−','-'),
    'auprc guard': f"{r['guards']['matched_edge_auprc']['ego_mean']:.6f}" in note,
}
print(checks); assert all(checks.values())
EOF
```
Expected: all five checks `True`.

- [ ] **Step 2: Cross-file consistency grep** — the run identity must be cited
  identically everywhere:

```bash
grep -rn '56b91c17fa8d3b86\|97e61a7d' CLAUDE.md README.md docs/03-experiment-protocol.md docs/05-egostitch-spec.md docs/results/G5-stage1-seed0-20260717.md | sort
```
Expected: every hit uses checkpoint `56b91c17fa8d3b86` and registration prefix
`97e61a7d`; no file still describes the gate as "incomplete" or "pending".

- [ ] **Step 3: Review the full staged diff** (`git diff`) — confirm it contains
  only closeout/disposition edits, no spec §5/§13 semantic changes (those belong
  to Task 2 with change-log lines).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs(results): close frozen-s0 Stage-1 screen — binding cut verdict, locked disposition to rev-3.0 e2e"
```

---

### Task 2: Spec §5/§13 rewrite to §14 (freeze-rule landing)

Executes committed implementation-plan **Task 1** — reproduced here with the
full section list so this plan is self-contained.

**Files:**
- Modify: `docs/05-egostitch-spec.md` (§5, §7, §8, §13.1, §13.8, §13.10,
  §13.16, §13.17; new §13.18; §12 change-log)

**Interfaces:**
- Consumes: design doc §§3–6
  (`docs/superpowers/specs/2026-07-16-egostitch-e2e-conditioned-encoder-design.md`)
  as source text; spec §14 as the normative summary being expanded.
- Produces: the normative sections that Tasks 5–7 of this plan (and committed
  plan Tasks 5–15) implement against. After this task, §14.3(2) is satisfied.

- [ ] **Step 1: §5 rewrite** — decision head for family `egostitch_e2e`:
  from-scratch V3.1-class trunk on raw token pairs; STE over the stitched
  scaffold (structure-only tokens: 4-type anchor labels endpoint-i/endpoint-j/
  slot-of-i/slot-of-j, π, m, soft degrees — **no** `h`, no `g`, no
  grounded-identity-match); zero-init tanh-gated cross-attention, cls_token
  queries only, injected after the final `n_inj ∈ {1, 2}` pair-cross-attention
  blocks, AB/BA share STE+XAttn parameters before `abba_max`; separate
  `c_content` pathway (s1 grounding summaries + identity-match flag); the three
  `_head`-suffixed nulls exactly as §14.2 (train = per-pair multiplicative
  masks, eval = batch-level hard bypass, numerically identical — required unit
  test; `p(i,j) = p(j,i)` under every null).
- [ ] **Step 2: §7 note** — no new loss lambda; `L = L_edge + λ_real·L_real +
  λ_ssl·L_ssl + λ_recon·L_recon` unchanged (locked objective).
- [ ] **Step 3: §8 curriculum** — warm-start fraction keeps `L_edge` (and
  trunk/STE/gates) inactive; branch-dropout probabilities constant afterwards.
- [ ] **Step 4: §13.1** — Stage-1 mechanism set for this family: Tokenize-lite,
  imagination, matching, stitching (Stitch already Stage-1), **STE + gated
  cross-attention head**; codebook/s3 remain Stage-2, harmonization Stage-3.
- [ ] **Step 5: §13.8 checkpoint-selection rule** — validation AUPRC remains
  the selection primary; **the residual/s0 tie-break is obsolete with s0
  retired**. Re-register it within-checkpoint: inside the
  `selection_auprc_tolerance` band (1e-4), prefer the checkpoint with the
  **larger** `std(full − f_logit) / std(f_logit)` on the fixed validation
  slice (liveness-preferring, same direction as the retired rule). Change-log
  line names this as a re-registration, not a removal.
- [ ] **Step 6: §13.10** — mark **retired for family `egostitch_e2e`** (no s0
  cache, no `s0_checkpoint_id`); historical for the frozen-s0 family.
- [ ] **Step 7: §13.16** — extend the fp32 pin: pair pass covers trunk, STE,
  gates, and head; provenance string `egostitch_e2e_pair_fp32_v1`; per-node
  encode may stay bf16.
- [ ] **Step 8: §13.17** — re-register liveness against the **within-checkpoint
  `f_logit`** (no frozen-s0 comparator artifact, no alignment step). Death rule
  stays conjunctive with the registered thresholds: residual std ratio
  `std(full − f_logit)/std(f_logit) < 1e-05` AND `Spearman(full, f_logit) >
  0.9999` AND top-1% overlap `> 0.9999`. Telemetry gains `gate_topo_tanh`,
  `gate_cont_tanh`, `grad_rms_trunk/ste/content`, per-epoch
  `topology_delta_std` on a fixed validation slice.
- [ ] **Step 9: new §13.18** — pinned defaults and runtime contract:
  - defaults: `ste_layers = 3`, `ste_dim = 128`, `xattn_heads = 8`,
    `n_inj = 1` (sweep `{1, 2}` reserved for E1/E3), `p_topo = p_cont = 0.15`
    (sweep 0.1–0.2 reserved; `p = 0` is a Stage-1 arm);
  - `model.config.permanent_null ∈ {none, all_head, content_head}` —
    training-time permanent bypass (B0-e2e arm = `all_head`; trained
    pair+topology arm = `content_head`);
  - scoring-time scaffold control `shuffle_within_pair`: separate
    source-side and destination-side slot permutations, each seeded from a
    stable hash of `(canonical unordered pair key, node side ∈ {src, dst},
    control seed)` — **invariant to batching, scoring order, GPU count, and
    shard boundaries by construction**; identical permutations for the AB and
    BA passes (canonical key), preserving `p(i,j) = p(j,i)`;
  - **binding enforcement**: formal (non-`--max-steps`) training requires the
    referenced registration to carry `status: BINDING`; `--max-steps` debug
    runs accept DRAFT but are forced into `*_debug` output directories and
    produce no held-out artifacts; the gate requires `status: BINDING` and a
    sha match against every consumed run metadata.
- [ ] **Step 10: §12 change-log** — one dated line per section touched
  (2026-07-17), each naming the design doc as source.
- [ ] **Step 11: Consistency check** —
  `grep -n 'permanent_null\|shuffle_within_pair\|13.18\|BINDING' docs/05-egostitch-spec.md`
  shows all four; §14.3 item 2 annotated "Satisfied 2026-07-17".
- [ ] **Step 12: Commit**

```bash
git add docs/05-egostitch-spec.md
git commit -m "docs(spec): rewrite §5/§13 to the rev-3.0 e2e head; within-checkpoint §13.8/§13.17; retire §13.10; §13.18 pins + binding enforcement"
```

---

### Task 3: Four e2e configs

**Files:**
- Create: `configs/egostitch_e2e_breadth_first.yaml`
- Create: `configs/egostitch_e2e_f_only_breadth_first.yaml`
- Create: `configs/egostitch_e2e_pair_topology_breadth_first.yaml`
- Create: `configs/egostitch_e2e_p0_breadth_first.yaml`

**Interfaces:**
- Consumes: `configs/egostitch_stage1_breadth_first.yaml` (base recipe — data,
  optimizer, diagnostics, runtime budget machinery all carry over).
- Produces: configs consumed by `hpc/run.sh train … --worker-module
  src.train_egostitch`; keys `model.family: egostitch_e2e`,
  `model.config.{ste_layers,ste_dim,xattn_heads,n_inj,p_topo,p_cont,
  permanent_null}`, `data.pack_dir` (raw-token pack, committed plan Task 12).

- [ ] **Step 1: Write the main config**

```yaml
# EgoStitch E2E Stage-1 formal training config (spec docs/05-egostitch-spec.md
# §5/§13 as rewritten 2026-07-17, defaults §13.18).
#   hpc/run.sh train configs/egostitch_e2e_breadth_first.yaml \
#       --worker-module src.train_egostitch

model:
  family: egostitch_e2e
  config:
    ste_layers: 3
    ste_dim: 128
    xattn_heads: 8
    n_inj: 1
    p_topo: 0.15
    p_cont: 0.15
    permanent_null: none

data:
  root: data
  strategy: breadth_first
  train_positives: e_sup            # pinned (spec §9.3)
  negative_ratio: 5                 # pinned (spec §10.2)
  partition_seed: 0
  msg_fraction: 0.8                 # pinned (spec §9.3)
  node_batch: 256
  edge_batch: 128                 # Task-8 measured H20-safe packed-token batch
  f0_cache: outputs/feature_packs/egostitch_f0/f0_matrix.pt
  grounding_cache: outputs/feature_packs/egostitch_f0/grounding.npz
  pack_dir: outputs/feature_packs/egostitch_e2e_tokens   # raw-token pack (plan Task 12)
  expected_missing_features:
    - node_004764
    - node_007050

optim:
  lr: 3.0e-4
  weight_decay: 0.01
  epochs: 30
  warmup_steps: 200
  grad_clip: 1.0
  warmstart_fraction: 0.2           # L_recon-only phase (spec §13.8)

diagnostics:
  gradient_probe_interval: 50
  gradient_imbalance_ratio: 10.0
  gradient_imbalance_steps: 1000
  probe_s1_abs_mean_max: 1000.0
  selection_auprc_tolerance: 1.0e-4   # tie-break: larger std(full-f_logit)/std(f_logit), spec §13.8
  topk_fraction: 0.01

eval:
  patience: 10
  eval_every: 1

seed: 0
output_dir: outputs/egostitch_e2e_stage1/full
mixed_precision: "bf16"
preregistration: docs/registrations/g5_e2e_stage1_preregistration.json

runtime:
  world_size: auto
  pack_dir: outputs/feature_packs/egostitch_f0
  pack_workers: 8
  loader_workers_per_rank: 0
  prefetch_factor: 2
  token_budget_candidates: [128, 256, 512]
  max_pairs_per_rank: 1048576
  memory_limit_gib: 85.0
  total_budget_seconds: 14400       # PROVISIONAL — replaced by Task 8 measured profile
  pack_budget_seconds: 1200
  setup_probe_budget_seconds: 900
  train_eval_budget_seconds: 11700
  artifact_budget_seconds: 120
  reserve_seconds: 480
  probe_warmup_steps: 5
  probe_timed_steps: 15
```

Note `data.s0_cache` / `data.s0_checkpoint_id` are **absent** (§13.10 retired
for this family).

- [ ] **Step 2: Write the three variants** by copying the main config and
  editing exactly these lines:
  - `…_f_only_breadth_first.yaml`: `permanent_null: all_head`,
    `output_dir: outputs/egostitch_e2e_stage1/f_only`
  - `…_pair_topology_breadth_first.yaml`: `permanent_null: content_head`,
    `output_dir: outputs/egostitch_e2e_stage1/pair_topology`
  - `…_p0_breadth_first.yaml`: `p_topo: 0.0`, `p_cont: 0.0`,
    `output_dir: outputs/egostitch_e2e_stage1/p0`
- [ ] **Step 3: Verify the variants differ only where intended**

```bash
for v in f_only pair_topology p0; do
  echo "== $v"; diff configs/egostitch_e2e_breadth_first.yaml "configs/egostitch_e2e_${v}_breadth_first.yaml"
done
```
Expected: exactly 2 changed lines for `f_only` and `pair_topology`; exactly 3
for `p0`.

- [ ] **Step 4: All four parse**

```bash
.venv/bin/python -c "import yaml,glob; [yaml.safe_load(open(p)) for p in glob.glob('configs/egostitch_e2e*.yaml')]; print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add configs/egostitch_e2e_*.yaml
git commit -m "feat(g5-e2e): Stage-1 screen configs (full / f_only / pair_topology / p0)"
```

---

### Task 4: DRAFT registration (not binding until Task 9)

**Files:**
- Create: `docs/registrations/g5_e2e_stage1_preregistration.json`
- Create: `docs/registrations/g5_e2e_stage1_preregistration.md` (prose mirror
  of the JSON, same structure as `g5_stage1_preregistration.md`)

**Interfaces:**
- Consumes: the frozen-s0 registration
  (`docs/registrations/g5_stage1_preregistration.json`) as the structural
  template; the completed screen's comparator values; the user's 2026-07-17
  review decisions (attribution share, bootstrap condition, trained
  pair+topology arm).
- Produces: the registration the worker sha-binds and the gate verifies.
  Status field starts `DRAFT`; Task 9 flips it to `BINDING`.

- [ ] **Step 1: Write the JSON.** Structure (values final unless marked):

```json
{
  "registration_id": "g5-e2e-stage1-20260717-conditioned-encoder-screen-v1",
  "status": "DRAFT",
  "gate": "G5 Stage 1 - EgoStitch E2E stitched-topology-conditioned pair encoder",
  "created_utc": "2026-07-17T00:00:00Z",
  "spec_binding": "docs/05-egostitch-spec.md §5/§13 (2026-07-17 rewrite) + §14",
  "protocol_binding": "docs/03-experiment-protocol.md §5.0.5, §5.2, E4.15-E4.17",
  "predecessor": {
    "registration": "g5-stage1-20260716-membership-normalized-screen-v2",
    "sha256": "97e61a7de006a3279d67e3adf1f7f1c663a7a739cf2682febfc0ca3219d0c446",
    "verdict": "cut",
    "result": "docs/results/G5-stage1-seed0-20260717.md"
  },
  "benchmark": { "alias": "Benchmark-A", "strategy": "breadth_first" },
  "seeds": [0],
  "arms": {
    "full": { "training": "configs/egostitch_e2e_breadth_first.yaml", "scored_array": "logits" },
    "b0_e2e_f_only": { "training": "configs/egostitch_e2e_f_only_breadth_first.yaml", "scored_array": "logits", "note": "permanent ∅_all_head; the matched pairwise-only control — the only arm '-topology' claims compare against" },
    "pair_topology": { "training": "configs/egostitch_e2e_pair_topology_breadth_first.yaml", "scored_array": "logits", "note": "permanent ∅_content_head, separately trained; THE registered attribution arm. The full checkpoint's eval-bypass pair_topology array is a nonbinding diagnostic only." },
    "structure_control_6a": { "training": "none (full checkpoint)", "scoring": "scaffold control shuffle_within_pair, stable-hash keyed, control seed 0", "scored_array": "logits" },
    "p0": { "training": "configs/egostitch_e2e_p0_breadth_first.yaml", "scored_array": "logits" }
  },
  "comparators": ["b0", "b0_cal_density", "b0_cal_selfdensity", "b0_cal_degseq"],
  "comparator_note": "carried over from the frozen-s0 screen; b0_cal_selfdensity is the empirically strongest arm there and remains the bar",
  "frozen_inputs": {
    "b0_candidate_scores": { "path": "outputs/deliverables/b0_v31_breadth_first_20260711/scores/candidate.npz", "sha256": "c5873caa3fece651d1155fd725e04f26a432ceee59d8e78dcda5f4acd687a95d", "checkpoint_id": "e092537d8cf1e208" },
    "g1_results": { "path": "outputs/deliverables/g1_graph_metrics_20260714/g1_results.json", "sha256": "668129a7c300d87682250382a36ef854c1f301c21d078b261f2b0fe2ae9d1ca4" },
    "g3_results": { "path": "outputs/deliverables/g3_graph_metrics_20260714/g3_results.json", "sha256": "e7fbc8e4e7f76ffe1f50123ac8e96c776cded79406e01e638539a3f835b79c24" }
  },
  "operating_point": {
    "canonical": "density-matched threshold on non-self candidate rows against target_edges = |E(strip_self_loops(test))| (predecessor definition, unchanged)",
    "matched": "per comparator: the full arm realizes the comparator's exact non-self edge quota by descending pass-1 score (predecessor matched-global-RD rule, unchanged)"
  },
  "primary_criteria": {
    "decision_procedure": "single_seed_point_estimate_dominance",
    "applies_to_arm": "full",
    "criteria": [
      { "metric": "clustering_mmd_ratio", "direction": "lower_is_better", "rule": "strictly lower than every comparator at the canonical operating point" },
      { "metric": "bfs_macro_gs", "direction": "higher_is_better", "rule": "strictly higher than each comparator at that comparator's exact matched-global-RD quota" },
      { "metric": "bfs_macro_rd", "direction": "higher_is_better", "rule": "strictly higher than each comparator at that same matched quota" }
    ],
    "all_must_pass": true,
    "statistics_procedure": "No inferential acceptance procedure with one seed; p/CI/Holm reported as not applicable"
  },
  "guards": [
    { "name": "degree_mmd_non_regression", "rule": "full-arm degree-MMD ratio <= 1.10 x B0's recomputed from the frozen candidate artifact" },
    { "name": "matched_edge_auprc", "rule": "full-arm degree-corrected candidate AUPRC >= B0's - 0.02; B0-e2e AUPRC reported alongside as the matched (nonbinding) reference" }
  ],
  "e2e_rules": {
    "pathway_attribution": {
      "metric": "clustering_mmd_ratio at the canonical operating point",
      "definition": "G_full = clu(b0_e2e_f_only) - clu(full); G_pt = clu(b0_e2e_f_only) - clu(pair_topology [trained arm])",
      "rule": "applies iff primaries pass and G_full > 0: require G_pt >= 0.25 * G_full",
      "reading_on_fail": "the gain is content-side; the topology-representation claim is withdrawn from the headline",
      "user_confirmed": "2026-07-17 review (accepted, on the separately trained arm)"
    },
    "structure_control_condition": {
      "rule": "paired bootstrap over the 500 fixed evaluation subgraphs: shared resample indices across the two arms, B = 1000 replicates, bootstrap seed 0; require the lower bound of the two-sided 95% interval (2.5th percentile) of clu(structure_control_6a) - clu(full) to exceed 0",
      "scope": "quantifies evaluator sampling stability at the fixed training seed; explicitly NOT cross-seed or inferential evidence",
      "reading_on_fail": "the shuffled relational connectivity was not necessary beyond the retained topology-derived node features (slot tokens keep pi, m, soft degrees); this does NOT prove the whole gain is non-topological — it bounds which part of the topology pathway is load-bearing",
      "user_confirmed": "2026-07-17 review (replaces the rejected fixed 0.005 margin)"
    },
    "liveness_within_checkpoint": {
      "reference": "f_logit of the same checkpoint (spec §13.17 as re-registered)",
      "death_rule": "conjunctive: min_residual_std_ratio 1e-05 AND max_spearman 0.9999 AND max_topk_overlap 0.9999 at topk_fraction 0.01",
      "consequence": "run-validity failure; gate fails closed before held-out topology metrics"
    },
    "four_logit_decomposition": "full / f_logit / pair_content / pair_topology published per scored pair for the full and p0 checkpoints, fp32, provenance egostitch_e2e_pair_fp32_v1, resolution guard per array; DIAGNOSTIC decomposition — the registered attribution arm is the separately trained pair_topology run",
    "checkpoint_selection": "validation AUPRC primary; within selection_auprc_tolerance 1e-4, prefer the larger within-checkpoint std(full - f_logit)/std(f_logit) (spec §13.8 as re-registered)"
  },
  "diagnostics_nonbinding": [
    "same-checkpoint decomposition deltas: topology_delta = full - pair_content, content_delta = full - pair_topology(bypass), and bypass-vs-trained pair_topology divergence",
    "linear probes on frozen STE token states: R2 to degree / ego-density / clustering (ridge lambda 1e-3, 5-fold), plus degree-partialled variants and Pi-consistency",
    "gate tanh magnitudes and per-family gradient norms over training",
    "p0-vs-full four-logit divergence (branch-dropout effect size)"
  ],
  "cost_report": {
    "requirement": "measured H20 profile REQUIRED-BEFORE-BINDING (Task 8): per-step wall, peak memory, 30-epoch extrapolation per arm (x4 arms), candidate scoring latency (x5 passes); the frozen-s0 673 s / 2.04 GiB profile explicitly does not extrapolate",
    "measured": "REQUIRED-BEFORE-BINDING"
  },
  "failure_reading": "If the full arm fails the primary family against the B0+cal comparator set, the topology-conditioned encoder does not beat calibrated independent scoring at this stage; the result is written up honestly and the locked-decision discussion is taken before any Stage-2 mechanism or E1 registration. If the primaries pass but pathway attribution fails, the honest conclusion is content-side information. If the primaries pass but the structure-control condition fails, the honest conclusion is that shuffled relational connectivity was not necessary beyond the retained topology-derived node features — not that the gain is non-topological.",
  "mechanics": {
    "worker_obligation": "src/train_egostitch.py records this file's sha256 as run_metadata.json['preregistration_sha256'] at start; it refuses formal (non --max-steps) runs unless status == BINDING; --max-steps debug runs accept DRAFT but write only to *_debug output directories and produce no held-out artifacts",
    "gate_obligation": "src/experiments/g5_stage1.py requires status == BINDING, recomputes this file's sha256, and refuses held-out metrics unless it matches ALL FOUR formal run metadatas (full, f_only, pair_topology, p0)",
    "hyperparameters": "spec §13.18 defaults x fixed Seed 0; no sweeps inside this screen",
    "p4_exclusion": "no hard heuristic/feature negatives and no hard-negative checkpoint criterion in this revision"
  }
}
```

The single `REQUIRED-BEFORE-BINDING` marker (measured cost) is the
registration's own binding protocol, resolved in Tasks 8–9 — it is the only
permitted unknown. Both decision constants are already user-confirmed.

- [ ] **Step 2: Write the md mirror** (same sections in prose, one heading per
  top-level key, verdict rule and failure readings verbatim).
- [ ] **Step 3: Validate**

```bash
.venv/bin/python -c "import json; json.load(open('docs/registrations/g5_e2e_stage1_preregistration.json')); print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add docs/registrations/g5_e2e_stage1_preregistration.json docs/registrations/g5_e2e_stage1_preregistration.md
git commit -m "docs(prereg): DRAFT e2e Stage-1 five-arm screening registration (trained attribution arm, paired-bootstrap control condition)"
```

---

### Task 5: Code — execute the committed implementation plan, Tasks 5–15

**Files:** exactly those listed per-task in
`docs/superpowers/plans/2026-07-16-egostitch-e2e-conditioned-encoder.md`
(Phases 1–5). That plan contains the complete TDD code for every task; open it
and execute Tasks 5–15 in order. Do **not** re-derive interfaces from memory.

**Interfaces (what this plan relies on afterwards):**
- `EgoStitchE2E.decompose(batch) -> dict` with keys
  `{"full", "f_logit", "pair_content", "pair_topology"}` (Task 11).
- Family `egostitch_e2e` trainable via `hpc/run.sh train … --worker-module
  src.train_egostitch` with branch dropout + gate telemetry (Tasks 12–13).
- `score_universe` writes four fp32 arrays + provenance
  `egostitch_e2e_pair_fp32_v1` (Task 14).
- `g5_stage1.py` computes within-checkpoint liveness, probes
  (`src/experiments/probes.py`), and the 5-arm summary (Task 15).

**Amendment carried from this plan's revision 2:** wherever committed-plan
Task 15 references the pair+topology arm, the registered arm is the
**separately trained** run D artifact; the same-checkpoint `pair_topology`
array feeds diagnostics only.

- [ ] **Step 1:** Execute Tasks 5–11 (model package: masks, gated x-attn,
  scaffold builder, STE, content tokens, conditioned trunk, `EgoStitchE2E`).
  Checkpoint after each task's commit: its test file passes.
- [ ] **Step 2:** Mid-phase gate: `uv run pytest tests/model -v && uv run mypy src`
  → clean.
- [ ] **Step 3:** Execute Tasks 12–15 (worker, scoring, gate).
- [ ] **Step 4:** Full local gate:

```bash
uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run mypy src tests
```
Expected: green except the two documented pre-existing local failures
(`test_b0_attention.py`, `test_e2_ddp_integration.py`).

- [ ] **Step 5:** End-to-end synthetic smoke — tiny model, synthetic batch:
  assert the four decomposed logits are finite, `full == f_logit` at init
  (zero gates), and `p(i,j) == p(j,i)` under every null (these asserts exist as
  unit tests from Tasks 6/11/14; run them explicitly):

```bash
uv run pytest tests/model/test_egostitch_e2e.py tests/test_score_universe.py -v
```

- [ ] **Step 6: Commit** any straggler formatting; branch merge per
  superpowers:finishing-a-development-branch if working on a feature branch.

---

### Task 6: Arm support — permanent-null override + stable-hash 6a control

These switches are needed by arms 2–4 and are **not** in the committed plan;
they extend its Tasks 13 and 14. Spec cover: §13.18 (Task 2 Step 9).

**Files:**
- Modify: `src/model/egostitch/config.py` (add `permanent_null: str = "none"`),
  `src/train_egostitch.py` (honor it in the loss/mask path),
  `src/score_universe.py` (add `--scaffold-control` CLI flag)
- Test: `tests/test_train_egostitch.py`, `tests/test_score_universe.py` (append)

**Interfaces:**
- Consumes: `masks_for_null(NULL_ALL_HEAD / NULL_CONTENT_HEAD)` (committed plan
  Task 5), `ScaffoldTokens` / `build_scaffold` (Task 7), `swap_direction`
  (Task 7).
- Produces:
  - config key `model.config.permanent_null ∈ {"none", "all_head",
    "content_head"}` — the corresponding hard mask is applied to **every**
    train and eval batch (`all_head` = B0-e2e arm; `content_head` = trained
    pair+topology arm);
  - `score_universe … --scaffold-control shuffle_within_pair` — per pair, two
    permutations of the slot indices (one for the source side, one for the
    destination side), each seeded via
    `blake2b(f"{min(u,v)}|{max(u,v)}|{side}|{seed}")` where `side ∈ {src,
    dst}` is assigned by **canonical order** (src = min-id endpoint), applied
    to the slot rows/cols of `Â_src`, `Â_dst` and the matching axes of `Π`
    (slot token features untouched; design §5 arm 6a). Keying on the canonical
    unordered pair makes the permutations identical for AB and BA passes and
    **independent of batch composition, scoring order, GPU count, and shard
    boundaries**. Recorded in artifact provenance as
    `scaffold_control=shuffle_within_pair,seed=0,keying=canonical_pair_v1`.

- [ ] **Step 1: Failing tests**

```python
def test_permanent_null_matches_bypass(tiny_e2e_setup):
    # training-path logits under each permanent_null equal the eval-time hard
    # bypass on the same weights/batch
    model, batch = tiny_e2e_setup
    for null, key in (("all_head", "f_logit"), ("content_head", "pair_topology")):
        trained = forward_with_config(model, batch, permanent_null=null)
        torch.testing.assert_close(trained, model.decompose(batch)[key])

def test_scaffold_shuffle_bites_and_is_deterministic(tiny_e2e_setup_nonzero_gates):
    model, pairs = tiny_e2e_setup_nonzero_gates
    base = score_pairs(model, pairs)
    shuf1 = score_pairs(model, pairs, scaffold_control="shuffle_within_pair")
    shuf2 = score_pairs(model, pairs, scaffold_control="shuffle_within_pair")
    assert not np.allclose(base, shuf1)        # gates non-zero => control bites
    np.testing.assert_allclose(shuf1, shuf2)   # deterministic

def test_scaffold_shuffle_ab_ba_symmetric(tiny_e2e_setup_nonzero_gates):
    model, pairs = tiny_e2e_setup_nonzero_gates
    fwd = score_pairs(model, pairs, scaffold_control="shuffle_within_pair")
    rev = score_pairs(model, [(v, u) for (u, v) in pairs], scaffold_control="shuffle_within_pair")
    np.testing.assert_allclose(fwd, rev)       # canonical-pair keying

def test_scaffold_shuffle_shard_invariant(tiny_e2e_setup_nonzero_gates, tmp_path):
    # score the same pair list as 1 shard and as 3 shards + strict merge;
    # merged logits must be bit-identical
    model, pairs = tiny_e2e_setup_nonzero_gates
    one = score_sharded(model, pairs, shards=1, control="shuffle_within_pair", out=tmp_path)
    three = score_sharded(model, pairs, shards=3, control="shuffle_within_pair", out=tmp_path)
    np.testing.assert_array_equal(one, three)
```

- [ ] **Step 2:** Run → FAIL (`permanent_null` unknown key; unknown CLI flag).
- [ ] **Step 3:** Implement: config plumb-through with value validation;
  scoring-path permutation applied to `ScaffoldTokens.adj` before the STE,
  computed from node ids only (never from batch position or shard index).
- [ ] **Step 4:** Run the two test files → PASS; `uv run mypy src` clean.
- [ ] **Step 5: Commit**

```bash
git add src/model/egostitch/config.py src/train_egostitch.py src/score_universe.py tests/test_train_egostitch.py tests/test_score_universe.py
git commit -m "feat(g5-e2e): permanent-null training override + stable-hash within-pair scaffold-shuffle control"
```

---

### Task 7: Binding enforcement + paired-bootstrap condition (gate/worker)

Machine-enforces the registration protocol and implements the registered
structure-control statistic. Spec cover: §13.18 (Task 2 Step 9).

**Files:**
- Modify: `src/train_egostitch.py` (status check + debug-dir redirect),
  `src/experiments/g5_stage1.py` (status check, 4-way sha check,
  paired bootstrap)
- Test: `tests/test_train_egostitch.py`, `tests/test_g5_stage1.py` (append)

**Interfaces:**
- Consumes: registration JSON `status` field; per-subgraph evaluator
  quantities already computed by the gate's MMD machinery (the 500 fixed
  induced subgraphs).
- Produces:
  - worker: formal runs (`--max-steps` absent) raise
    `PreregistrationNotBinding` unless `status == "BINDING"`; with
    `--max-steps`, DRAFT is accepted, `output_dir` is forced to
    `<output_dir>_debug`, and no held-out artifacts are written;
  - gate: raises unless `status == "BINDING"` and the registration sha matches
    the `preregistration_sha256` of **all four** formal run metadatas;
  - `paired_bootstrap_lower_bound(stat_fn, samples_a, samples_b, n_boot=1000,
    seed=0, alpha=0.05) -> float` — resamples the shared subgraph indices,
    recomputes the statistic per replicate for both arms from identical
    indices, returns the `alpha/2` quantile of `stat(a) − stat(b)`; the gate
    applies it with `stat_fn` = clustering-MMD ratio, `a` = 6a control,
    `b` = full arm, and passes the condition iff the returned bound `> 0`.

- [ ] **Step 1: Failing tests**

```python
def test_worker_refuses_draft_registration_for_formal_run(tmp_cfg_draft):
    with pytest.raises(PreregistrationNotBinding):
        run_worker(tmp_cfg_draft, max_steps=None)

def test_worker_debug_run_redirects_output(tmp_cfg_draft):
    result = run_worker(tmp_cfg_draft, max_steps=5)
    assert result.output_dir.name.endswith("_debug")
    assert not (result.output_dir / "best.pt").exists() or result.no_heldout_artifacts

def test_gate_requires_binding_and_four_sha_matches(gate_fixture):
    gate_fixture.registration["status"] = "DRAFT"
    with pytest.raises(PreregistrationNotBinding):
        run_gate(gate_fixture)
    gate_fixture.registration["status"] = "BINDING"
    gate_fixture.run_metadata["pair_topology"]["preregistration_sha256"] = "0" * 64
    with pytest.raises(RegistrationShaMismatch):
        run_gate(gate_fixture)

def test_paired_bootstrap_lower_bound():
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 1.0, 500)
    clearly_higher = base + 1.0          # paired shift >> sampling noise
    lb = paired_bootstrap_lower_bound(np.mean, clearly_higher, base)
    assert lb > 0
    lb_null = paired_bootstrap_lower_bound(np.mean, base + 0.001, base)
    assert lb_null < 0 or abs(lb_null) < 0.1   # no false certainty on a hairline shift
    assert lb == paired_bootstrap_lower_bound(np.mean, clearly_higher, base)  # deterministic
```

- [ ] **Step 2:** Run → FAIL (names undefined).
- [ ] **Step 3:** Implement (bootstrap in `src/experiments/g5_stage1.py` or a
  small shared helper module; identical resample indices via one
  `np.random.default_rng(seed)` stream driving both arms).
- [ ] **Step 4:** Run: `uv run pytest tests/test_train_egostitch.py tests/test_g5_stage1.py -v`
  → PASS; `uv run mypy src` clean.
- [ ] **Step 5: Commit**

```bash
git add src/train_egostitch.py src/experiments/g5_stage1.py tests/test_train_egostitch.py tests/test_g5_stage1.py
git commit -m "feat(g5-e2e): machine-enforced BINDING status + paired-bootstrap structure-control condition"
```

---

### Task 8: HPC profile smoke → measured cost table (USER-OWNED runtime)

No formal artifacts; `--max-steps` debug runs only — which, per Task 7, are
**forced into `*_debug` directories** and accept the DRAFT registration.
Formal directories (`outputs/egostitch_e2e_stage1/{full,f_only,pair_topology,p0}`)
must not exist yet when Task 10 starts.

**Interfaces:**
- Consumes: Task 3 configs, Tasks 5–7 code deployed to the container checkout.
- Produces: measured numbers pasted into
  `docs/registrations/g5_e2e_stage1_preregistration.json → cost_report.measured`.

- [ ] **Step 1:** Sync the container checkout to the Task-7 commit; run the
  container test gate per `hpc/README.md` (`check` step).
- [ ] **Step 2:** Build the raw-token pack once:
  `hpc/run.sh train configs/egostitch_e2e_breadth_first.yaml --worker-module src.train_egostitch --max-steps 1`
  (pack phase runs to completion; training stops immediately; output lands in
  `outputs/egostitch_e2e_stage1/full_debug`). Verify
  `outputs/feature_packs/egostitch_e2e_tokens` exists.
- [ ] **Step 3:** Bounded profile run: `… --max-steps 50`. Record from
  `profile.json` / logs: per-step wall (mean of timed steps), peak GPU memory,
  detected world size, selected `B_n`.
- [ ] **Step 4:** Extrapolate: `steps_per_epoch × 30 × per-step wall` per
  training arm (× 4 arms); compare against `runtime.total_budget_seconds`
  (14400 provisional) and adjust the four configs if the measurement demands
  it (config edit + commit, allowed — registration is still DRAFT).
- [ ] **Step 5:** Scoring latency probe: score a 10k-row pair file fp32
  (`hpc/run.sh score --checkpoint <debug ckpt> --pairs file:<10k tsv> --output outputs/egostitch_e2e_stage1/full_debug/probe.npz`),
  record rows/s → extrapolated 2,037,171-row wall per artifact (× 5 passes).
- [ ] **Step 6 (optional sanity check, never a paper arm):** exact-B0
  reproduction smoke — trunk standalone under the canonical B0 recipe,
  `--max-steps 200`; assert loss decreases and no NaN. Design §5 explicitly
  scopes this as an implementation sanity check only.
- [ ] **Step 7:** Paste the measured table into the registration JSON + md
  (replacing `cost_report.measured`), commit:

```bash
git add docs/registrations/g5_e2e_stage1_preregistration.json docs/registrations/g5_e2e_stage1_preregistration.md configs/
git commit -m "docs(prereg): measured H20 cost profile for the e2e screen"
```

---

### Task 9: Bind the registration (USER sign-off — hard stop)

Both decision constants were already confirmed in the 2026-07-17 review; this
stop is the final whole-document sign-off plus the cost check.

**Interfaces:**
- Produces: `status: BINDING`, fixed sha256; after this commit the registration
  file may not change (amendments require a new versioned registration, as with
  the frozen-s0 predecessor).

- [ ] **Step 1:** Confirm `cost_report.measured` is populated with Task-8
  numbers and no `REQUIRED-BEFORE-BINDING` marker remains anywhere:

```bash
grep -c 'REQUIRED-BEFORE-BINDING' docs/registrations/g5_e2e_stage1_preregistration.json
```
Expected: `0`.

- [ ] **Step 2:** Present the complete registration to the user for final
  sign-off (do not proceed on silence).
- [ ] **Step 3:** Set `"status": "BINDING"`; commit; record the binding sha:

```bash
git add docs/registrations/g5_e2e_stage1_preregistration.json docs/registrations/g5_e2e_stage1_preregistration.md
git commit -m "docs(prereg): BIND e2e Stage-1 screening registration"
shasum -a 256 docs/registrations/g5_e2e_stage1_preregistration.json
```

Expected: the sha printed here must reappear verbatim in all **four** formal
`run_metadata.json` files (Task 10) and in the gate output (Task 11).

---

### Task 10: Formal runs and scoring (USER-OWNED, HPC)

**Interfaces:**
- Consumes: bound registration (Task 9), synced container checkout at the
  binding commit.
- Produces: four run directories under `outputs/egostitch_e2e_stage1/` and
  five candidate-score artifacts, all sha-bound to the registration.

- [ ] **Step 1: Train the four arms** (order-independent; run sequentially or
  as GPU availability allows):

```bash
hpc/run.sh train configs/egostitch_e2e_breadth_first.yaml               --worker-module src.train_egostitch
hpc/run.sh train configs/egostitch_e2e_f_only_breadth_first.yaml        --worker-module src.train_egostitch
hpc/run.sh train configs/egostitch_e2e_pair_topology_breadth_first.yaml --worker-module src.train_egostitch
hpc/run.sh train configs/egostitch_e2e_p0_breadth_first.yaml            --worker-module src.train_egostitch
```
Verify per run: `complete.json` status complete; §13.17 telemetry series
present (gate tanh, grad RMS, topology_delta_std).

- [ ] **Step 2: Verify the binding sha across all four run metadatas**

```bash
BIND_SHA=$(shasum -a 256 docs/registrations/g5_e2e_stage1_preregistration.json | cut -d' ' -f1)
for arm in full f_only pair_topology p0; do
  .venv/bin/python -c "import json,sys; m=json.load(open('outputs/egostitch_e2e_stage1/$arm/run_metadata.json')); sys.exit(0 if m['preregistration_sha256']=='$BIND_SHA' else 1)" \
    && echo "$arm OK" || echo "$arm SHA MISMATCH"
done
```
Expected: four `OK` lines. Any mismatch invalidates that run — stop and
diagnose before scoring.

- [ ] **Step 3: Score five artifacts** (all fp32 pair pass; four-logit arrays
  for the e2e family are automatic per committed-plan Task 14):

```bash
hpc/run.sh score --checkpoint outputs/egostitch_e2e_stage1/full/best.pt          --pairs candidate --output outputs/egostitch_e2e_stage1/full/scores/candidate.npz
hpc/run.sh score --checkpoint outputs/egostitch_e2e_stage1/full/best.pt          --pairs candidate --scaffold-control shuffle_within_pair --output outputs/egostitch_e2e_stage1/full/scores/candidate_6a.npz
hpc/run.sh score --checkpoint outputs/egostitch_e2e_stage1/f_only/best.pt        --pairs candidate --output outputs/egostitch_e2e_stage1/f_only/scores/candidate.npz
hpc/run.sh score --checkpoint outputs/egostitch_e2e_stage1/pair_topology/best.pt --pairs candidate --output outputs/egostitch_e2e_stage1/pair_topology/scores/candidate.npz
hpc/run.sh score --checkpoint outputs/egostitch_e2e_stage1/p0/best.pt            --pairs candidate --output outputs/egostitch_e2e_stage1/p0/scores/candidate.npz
```
Verify per artifact: `validate_score_resolution` passed (loud failure
otherwise), provenance `egostitch_e2e_pair_fp32_v1`, the 6a artifact
additionally records
`scaffold_control=shuffle_within_pair,seed=0,keying=canonical_pair_v1`.

- [ ] **Step 4:** Pull the five artifacts + four run dirs back to the local
  checkout (rsync; paths mirror the container's).

---

### Task 11: Gate, verdict, result note, doc sync

**Files:**
- Create: `docs/results/G5-e2e-stage1-seed0-<date>.md`
- Modify: `README.md`, `CLAUDE.md`, `docs/03-experiment-protocol.md` (§5.0.5
  status), `docs/artifacts/2026-07-17-egostitch-status.html` (or successor)

**Interfaces:**
- Consumes: Task 10 artifacts, committed-plan Task 15 gate CLI (as amended by
  Task 7).
- Produces: the binding five-arm screening verdict and its documentation.

- [ ] **Step 1: Produce the required nonbinding probe artifact, then run the
  gate** (single-process; the producer may run on the scoring host and the gate
  locally after copy-back):

```bash
.venv/bin/python -m src.experiments.probes produce-e2e \
    --checkpoint outputs/egostitch_e2e_stage1/full/best.pt \
    --run-metadata outputs/egostitch_e2e_stage1/full/run_metadata.json \
    --preregistration docs/registrations/g5_e2e_stage1_preregistration.json \
    --data-root data --strategy breadth_first \
    --output outputs/egostitch_e2e_stage1/full/probes/e2e_probe_v1.npz

.venv/bin/python -m src.experiments.g5_stage1 \
    --mode e2e \
    --full-universe    outputs/egostitch_e2e_stage1/full/scores/candidate.npz \
    --control-universe outputs/egostitch_e2e_stage1/full/scores/candidate_6a.npz \
    --fonly-universe   outputs/egostitch_e2e_stage1/f_only/scores/candidate.npz \
    --pt-universe      outputs/egostitch_e2e_stage1/pair_topology/scores/candidate.npz \
    --p0-universe      outputs/egostitch_e2e_stage1/p0/scores/candidate.npz \
    --run-metadata outputs/egostitch_e2e_stage1/full/run_metadata.json \
                   outputs/egostitch_e2e_stage1/f_only/run_metadata.json \
                   outputs/egostitch_e2e_stage1/pair_topology/run_metadata.json \
                   outputs/egostitch_e2e_stage1/p0/run_metadata.json \
    --b0-universe outputs/deliverables/b0_v31_breadth_first_20260711/scores/candidate.npz \
    --b0cal-results outputs/deliverables/b0_cal_20260714 \
    --probe-artifact outputs/egostitch_e2e_stage1/full/probes/e2e_probe_v1.npz \
    --preregistration docs/registrations/g5_e2e_stage1_preregistration.json \
    --output-dir outputs/egostitch_e2e_stage1/formal_gate
```
The probe producer fails closed unless the metadata/config identify the registered
full arm (`arms.full.training`, `permanent_null = none`,
`p_topo = p_cont = 0.15`); p0 is not an interchangeable source.

(Exact flag names come from Task 15's implementation; if they differ, the gate
`--help` is authoritative — but the *inputs* above are the registered set and
may not shrink.)
Expected outputs: `g5_e2e_stage1_results.json` + `g5_e2e_stage1_tables.md` with
the 5-arm summary, diagnostic four-logit decomposition, liveness verdict,
probe table, pathway-attribution (trained arm) and paired-bootstrap
structure-control rules evaluated, and `verdict: pass|cut`.

- [ ] **Step 2: Liveness first.** If the within-checkpoint death rule fired,
  the run is invalid (fail-closed): no held-out claims; diagnose via gate-tanh
  telemetry before any rerun discussion. Do not reinterpret a dead run as a
  verdict.
- [ ] **Step 3: Write the result note** with the same skeleton as
  `docs/results/G5-stage1-seed0-20260717.md`: verdict + scope caveat, run
  identity table (×4 runs), 5-arm assembled table, registered decision table
  (three exact primary criteria), guards, pathway attribution (trained arm),
  paired-bootstrap control condition with its evaluator-stability scope
  stated, diagnostic decomposition deltas, probe table, cost table, and the
  verbatim registered reading for whichever branch fired.
- [ ] **Step 4: Doc sync** — README status block, CLAUDE.md load-bearing
  facts/evidence boundary, protocol §5.0.5 status line, status HTML. Keep the
  numbers identical across all of them.
- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs(results): e2e Stage-1 five-arm screen — <pass|cut> verdict + doc sync"
```

- [ ] **Step 6: Next-step branch (record, don't act):**
  - `pass` → Stage-2 planning (codebook + s3 as STE inputs) and the E1/E3
    multi-seed registrations (remaining ladder arms: B3-full, B5, depth rungs,
    6b–6f including degree-preserving rewiring).
  - `cut` → the registered failure reading verbatim; locked-decision discussion
    with the user before any successor design work.

---

## Self-review notes

- **Spec coverage:** design §8 landing items — (1) proposal rev ✔ landed
  `c0e2144`; (2) spec §5/§13 rewrite → Task 2 (now incl. §13.8); (3) protocol
  disposition ✔ staged, committed by Task 1; (4) five-arm registration →
  Tasks 4/9. Spec §14.3 conditions map to Tasks 1 (result ✔), 2 (rewrite),
  4+9 (registration), 5–7 (code), 8 (cost). Design §5 Stage-1 scope: all five
  arms have a training and scoring path (arms table); arm 7's "permanent"
  wording is honored by the separately trained run D; reserved arms explicitly
  excluded from the registration.
- **Interface consistency:** `permanent_null ∈ {none, all_head, content_head}`
  (Tasks 2/3/4/6/10), `shuffle_within_pair` + `canonical_pair_v1` keying
  (Tasks 2/4/6/10), four-logit array names
  `logits/f_logit/pair_content/pair_topology` (Tasks 4/5/10/11),
  `paired_bootstrap_lower_bound` (Tasks 4/7/11),
  `PreregistrationNotBinding` (Tasks 7/9/10) — single spelling throughout;
  liveness thresholds (1e-05 / 0.9999 / 0.9999 / 0.01) identical in Task 2,
  Task 4, and the predecessor registration; §13.8 tie-break statistic identical
  in Task 2 Step 5, the config comment, and the registration's
  `checkpoint_selection` field.
- **User review decisions (2026-07-17) encoded:** trained attribution arm
  (arms table, registration, Tasks 3/6/10/11); §13.8 tie-break retained
  within-checkpoint (Task 2 Step 5); exact per-metric primary criteria
  (registration `primary_criteria.criteria`); stable-hash 6a + shard-invariance
  tests + narrowed reading (Tasks 2/4/6); debug/formal separation + BINDING
  enforcement + 4-way sha check (Tasks 2/7/8/9/10); 25% accepted /
  0.005 rejected → paired bootstrap (registration `e2e_rules`).
- **Known unknowns, by design:** measured cost numbers (Task 8), gate CLI flag
  spellings (committed-plan Task 15 defines them; Task 11 defers to `--help`
  without shrinking the registered input set).

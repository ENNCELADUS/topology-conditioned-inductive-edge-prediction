# Three-Component Model Refactor — Design

**Date:** 2026-08-02
**Status:** APPROVED (owner, 2026-08-02); historical EgoStitch arm, not the selected project method.
**Scope:** `src/model/egostitch/`, `src/train_egostitch.py`, `src/score_universe.py`,
`src/experiments/`, `configs/`, `hpc/`, `docs/registrations/`
**Supersedes for the affected surfaces:** the frozen-contract regime described in
`CLAUDE.md` and `docs/05-egostitch-spec.md` §14. The owner has withdrawn the
registration mechanism; see §10.

---

## 1. Goal

Restructure the EgoStitch model into three components with explicit interfaces, so
each can be replaced independently and alternative versions compared:

1. **Pairwise classifier pathway** — the mature baseline. Consumes the two endpoints'
   raw token streams and, optionally, a conditioning embedding. Emits the edge logit.
2. **Neighborhood graph generator** — this arm maps endpoints and grounding pools to
   `ImaginedGraph`; endpoint-only families require another interface.
3. **Graph encoder** — *encodes the graph*. Consumes a graph. Emits a graph embedding.

Today none of the three is substitutable. `EgoStitchE2E.__init__`
(`src/model/egostitch/e2e_model.py:93-159`) hard-wires every submodule, and there is
no registry anywhere in `src/`. The only "ablation" mechanism is a flat scalar in
`model.config` (for example `feature_standardization: row_layernorm`).

Two structural defects block substitution and are corrected here:

- **`build_scaffold` constructs a graph; it does not encode one.** It lives on the
  encoder side today (`src/model/egostitch/scaffold.py:335`), so the boundary between
  "generator" and "encoder" does not fall where the names say it does. Graph
  construction — including Sinkhorn alignment and joint scaffold assembly — is
  imagination and moves into the generator.
- **Nothing downstream of the generator accepts a graph.** `STEncoder.forward` takes a
  `ScaffoldTokens` with a compile-time-fixed `FEAT_DIM = 11` and `EDGE_TYPES = 4`
  (`src/model/egostitch/ste.py:39,19,45-56`). No off-the-shelf GNN can occupy that
  slot. The encoder's input becomes a plain graph whose dimensions are read at runtime.

## 2. Non-goals

- **Not a model redesign.** What the model computes stays as it is, minus the content
  path (§9). Deferred to a separate effort.
- **Not shipping alternative architectures.** This pass ships exactly one real version
  per component (today's code, behavior-preserving) plus a null generator. Additional
  generators and encoders are future work that the interfaces enable.
- **`src/model/B0.py` is untouched.** `V3_1` remains the standalone published baseline.
  The classifier component reuses `SiameseEncoder`, `PairCrossAttention` and `MLPHead`
  from it, exactly as `e2e_model.py:28` and `trunk.py:20` already do.

## 3. Interfaces

### 3.1 The graph

```python
@dataclass(frozen=True)
class ImaginedGraph:
    x:    Tensor                     # (B, N, F)     node features
    adj:  Tensor                     # (B, R, N, N)  typed soft adjacency
    mask: Tensor                     # (B, N)        soft node existence
    aux:  Mapping[str, Tensor]       # generator-private

    def swapped(self) -> ImaginedGraph: ...
```

`N`, `R` and `F` are read from the tensors at construction time by whatever consumes
the graph. This is the single change that makes the encoder slot substitutable:
`STEncoder`'s module-level `FEAT_DIM`/`EDGE_TYPES` constants become constructor
arguments derived from the graph the generator actually emits.

Today's `egostitch_imagine` generator emits `N = 2 + 2K = 34`, `R = 4`
(star/intra/align/closure), `F = 11` — the structural channels currently assembled at
`scaffold.py:424-434`. A future generator that also emits slot content `h` changes only
`F`; no interface change, no encoder change.

`mask` is new as an explicit field. Today node existence is smuggled into feature
channel 4 (`scaffold.py:430`). It stays in `x` for the `ste_typed` encoder's benefit
and is *also* exposed as `mask` so encoders that need it structurally can read it
without knowing the channel layout.

**`aux` is a private side-channel, not part of the contract.** It carries
`SlotSet.{h, pi, mult, gate, pointer, adj, adj_logits}` and the Sinkhorn `plan`, which
the generator's own losses and the trainer's telemetry require:
`_e2e_dispersion_rows` (`train_egostitch.py:3311-3321`), `_e2e_scale_rows` (`:3334`),
`E2ESlotCollapseGuard` (`:888-919`) and `_enforce_e2e_initial_slot_health` (`:3940`).
Rule: **no encoder may read `aux`.** A generator swap invalidates `aux` wholesale, and
that is correct — the telemetry that reads it belongs to that generator.

`swapped()` returns the direction-relabelled graph for the BA stream, sharing the `adj`
tensor object. It replaces `swap_direction` (`scaffold.py:438-452`), which today
permutes the four anchor one-hot channels. Moving it onto the graph keeps the scaffold's
channel layout from leaking into the classifier.

### 3.2 The embedding

```python
@dataclass(frozen=True)
class GraphEmbedding:
    tokens: Tensor    # (B, N, d)
    pooled: Tensor    # (B, d)

@dataclass(frozen=True)
class PairConditioning:
    ab: GraphEmbedding
    ba: GraphEmbedding

@dataclass(frozen=True)
class PairInputs:
    emb_a:  Tensor    # (B, T_a, d_in)  raw token stream, endpoint A
    emb_b:  Tensor    # (B, T_b, d_in)
    len_a:  Tensor    # (B,)            unpadded token counts
    len_b:  Tensor    # (B,)
    edge_mask: Tensor | None            # (B,) DDP filler-row exclusion
```

`PairInputs` carries only what the classifier needs to stand alone. It is deliberately
free of endpoint *features* (`x_a`/`x_b`) and grounding pools: those go to the
generator and never to the classifier, which is what makes the null-generator
configuration a true pairwise baseline rather than a masked variant of the full model.

Encoders produce both forms. The `b0_v31` classifier consumes `tokens` via gated cross
attention, exactly as today. `pooled` exists so a classifier that conditions by FiLM or
addition needs no encoder change.

### 3.3 The three components

```python
class NeighborhoodGenerator(Protocol):
    def encode_node(self, x, ground, ground_ids=None) -> GeneratorNodeState: ...
    def stitch(self, state_a, state_b, is_self, *, perturbation=None) -> ImaginedGraph | None: ...
    def forward(self, x_a, x_b, ground_a, ground_b, *, is_self,
                perturbation=None) -> ImaginedGraph | None: ...
    def auxiliary_losses(self, graph, batch) -> dict[str, Tensor]: ...

class GraphEncoder(Protocol):
    out_dim: int
    def forward(self, graph: ImaginedGraph) -> GraphEmbedding: ...
    def auxiliary_losses(self, embedding, batch) -> dict[str, Tensor]: ...

class PairClassifier(Protocol):
    def forward(self, pair: PairInputs, cond: PairConditioning | None) -> Tensor: ...
```

Returning `None` from the generator is the null case: the composite passes `cond=None`
and the classifier runs unconditioned. That path is exactly the B0 pairwise baseline,
reachable by config alone.

**Amendment (2026-08-03), correcting two errors in this section as first drafted.**
Both were found by the agents implementing it, not by review of the spec:

1. **The generator must expose a separable per-node phase.** A single fused
   `forward(x_a, x_b, ...)` breaks per-node caching, and that caching is load-bearing:
   `score_universe.py:1981-2123` and `train_egostitch.py:3127-3175` encode each node
   once and reuse the state across many pairs, so a universe scoring pass does not
   re-encode both endpoints per pair. Hence `encode_node` / `stitch`, with `forward`
   as the convenience composition. `GeneratorNodeState` must survive index-select,
   index-copy and re-stack, because that is what the caching callers do to it.
2. **`stitch` must accept a scaffold-control perturbation.** `score_universe.py:1935`
   builds one via `make_scaffold_input_perturbation` and threads it into scaffold
   assembly to produce the two mandatory structure-control arms (6a shuffle,
   6e rewire). Omitting it from the protocol would have silently broken both controls.

### 3.4 Composite

```python
class EgoStitchModel(nn.Module):
    generator:  NeighborhoodGenerator | None
    encoder:    GraphEncoder | None
    classifier: PairClassifier
```

Forward:

```
graph  = generator(x_a, x_b, ground_a, ground_b, is_self)     # or None
cond   = PairConditioning(encoder(graph), encoder(graph.swapped()))  # or None
logits = classifier(pair_inputs, cond)
```

The composite owns loss aggregation (§6) and the cached-state path
(`encode_node_state` / `build_pair_context_from_states`) that `score_universe.py:1981-2185`
and `train_egostitch.py:3127-3308` depend on.

## 4. AB/BA symmetry — a hard invariant

`GatedCrossAttention._global_center` takes its mean over `dim=0` of the whole incoming
batch and `_update_ema` bumps `ema_updates` by exactly 1 per `forward`
(`src/model/egostitch/conditioning.py:113-154, 197-199`). `score_pair_context`
therefore runs AB and BA as **one** trunk batch (`e2e_model.py:512-522`), so both
directions are centered by one synchronized statistic and the EMA advances once per
step. This is pinned by `tests/model/test_egostitch_e2e_model.py:194`
(`ema_updates == 1`, `ema_mu == 2.0` for `ab=1.0, ba=3.0`) and `:160, 177, 214`.

**Consequences for the refactor:**

- AB/BA symmetrization lives **inside** `PairClassifier.forward`. A classifier that
  called the trunk twice would center twice with two different means, break
  `ema_updates == 1`, and break train/eval agreement (eval reads the single frozen
  `ema_mu` at `conditioning.py:203`).
- Conditioning injection stays inside the trunk, not before or after it. `mu` is
  computed on the attention output per injected block (`trunk.py:102-113`), so the
  AB∪BA union must exist at that depth.
- `edge_mask` is duplicated with the batch (`e2e_model.py:521`); DDP filler rows are
  excluded from `mu` (`conditioning.py:192-197`), pinned by
  `tests/model/test_egostitch_conditioning.py:356`.
- Branch masks are shared across directions, not per-direction (`e2e_model.py:519`).
- The fp32 island at `e2e_model.py:528-529` (`head(feat.float())` outside autocast)
  survives verbatim. It is load-bearing for the `full − f_logit` residual and is
  checked by `_validate_e2e_precision_outputs` (`train_egostitch.py:3785-3879`,
  `residual_correlation >= 0.999`).
- The BA embedding comes from `graph.swapped()` on the **same** graph object. The graph
  is built once per pair.

## 5. Package layout

```
src/model/egostitch/
  graph.py          ImaginedGraph, GraphEmbedding, PairConditioning, PairInputs
  registry.py       three name→class registries and build helpers
  composite.py      EgoStitchModel
  generator/
    base.py         NeighborhoodGenerator protocol
    egostitch.py    v1 — wraps the existing imagine + stitch + scaffold pipeline
    null.py         v2 — returns None
    (model.py, imagine.py, tokenize.py, matching.py, stitch.py, scaffold.py,
     losses.py move here as private implementation)
  encoder/
    base.py         GraphEncoder protocol
    ste.py          v1 — typed message passing, dims read from the graph
  classifier/
    base.py         PairClassifier protocol
    b0_v31.py       v1 — SiameseEncoder + ConditionedPairCrossAttention
                         + PairContextGatedReadout + AB/BA max + MLPHead
    (trunk.py, conditioning.py move here)
  layers.py         shared
```

`decision.py` is deleted (§9). `e2e_model.py` is replaced by `composite.py`.

## 6. Loss ownership

Each component owns the losses that supervise it, so swapping a component swaps its
auxiliary losses with it.

| Component | Losses | Current location |
|---|---|---|
| generator | `L_recon` (feat/exist/mult/slotadj/gate/ptr/div), `L_deg`, `L_real` (egostat + GIN), `L_ssl`, `L_align` | `losses.py:194, 337, 366, 408, 449, 495, 525, 564`; `model.py:385, 474`; `e2e_model.py:392-445` |
| encoder | `L_rel` | `losses.py:147`; `e2e_model.py:381-390` |
| classifier | `L_edge` | `train_egostitch.py:1012` |

`L_align` is a generator loss because alignment is part of imagining the joint graph;
it consumes the Sinkhorn `log_plan`, which the generator now owns.

Today `align_loss` and `rel_loss` are injected back into the node-stream recon dict at
`train_egostitch.py:2771-2777` and weighted inside `_recon_total`
(`losses.py:628-635`), whose ten component names are pinned against
`e2e_recon_component_factors` (`train_egostitch.py:776-790`, enforced at
`losses.py:615-618`). After the refactor the composite performs that aggregation, and
the pinned-name check moves with it. The four-family split
(`stage1_family_tensors`, `losses.py:639`) that `_e2e_family_probe` depends on is
preserved.

## 7. Parameter groups

Groups collapse from the current name-prefix scheme
(`train_egostitch.py:1061-1104`: `generator.` / the five `conditioning_prefixes` /
everything else) to **one group per component**:

| group | clip norm source |
|---|---|
| `generator` | `training.generator_clip_norm` |
| `encoder` | `training.clip_norm` |
| `classifier` | `training.pair_encoder_clip_norm` |

The disjoint-and-exhaustive assertion (`:1091-1092`) and the non-empty assertion
(`:1098-1099`) are kept — they are real correctness checks. The per-group sha256
manifest (`:1100-1103`) is kept as a record but is no longer validated against a
registration (§10).

`_e2e_family_probe` (`train_egostitch.py:3664-3741`) encodes three assumptions that
must be rebuilt against the new grouping: the three-way partition; that `L_recon`
reaches the conditioning group only via `rel_head`; and that the edge family reaches
the generator. The last remains true — the classifier's edge loss still backprops
through encoder and generator — but the group names change.

## 8. Configuration

```yaml
model:
  family: egostitch_e2e
  config:
    generator:
      name: egostitch_imagine
      n_ground: 50
      feature_standardization: zscore_vfit_v1
      tau_adj: 0.5
      tau_div: 0.5
      l_gate_pos_weight: 6.17
    encoder:
      name: ste_typed
      dim: 128
      layers: 3
      w_rel: 0.25
    classifier:
      name: b0_v31
      d_model: 512
      n_inj: 1
      xattn_heads: 8
      p_topo: 0.15
      conditioning_ema_decay: 0.99
```

`E2EConfig` becomes three nested dataclasses, each validated by the existing strict
`_from_mapping` (`config.py:45-73`) so unknown keys are still rejected per component.
The family name `egostitch_e2e` is retained to limit churn in `build_model`
(`score_universe.py:1014`).

`p_cont`, `permanent_null: content_head`, and `feature_stats_sha256`-as-a-gate are
removed (§9, §10). `feature_stats_sha256` survives as a recorded value on the generator
config.

## 9. Content-path removal

The content branch is deleted. It routes slot semantics around the encoder directly
into the trunk, which is precisely the bypass the three-component split exists to
eliminate.

**Deleted:** `build_content_tokens`, `ContentProjector`, `counterpart_membership`,
`grounded_identity_match` (`scaffold.py:299-332, 455-543`); `NULL_CONTENT_HEAD`,
`HeadNullMasks.cont`, the `p_cont` draw in `sample_branch_masks`
(`conditioning.py:22, 31, 36-43, 54`); `cont_xattn` and its wiring
(`trunk.py:46-54, 64-66, 110-112`); `E2EPairContext.cont` and the whole `need_cont`
path (`e2e_model.py:79, 143, 262, 327-355, 373, 467, 490-502, 518-520, 554-561, 600`);
`E2EConfig.p_cont` and the `content_head` value of `permanent_null`
(`config.py:242, 273-274, 309, 328-332`).

**`decision.py` is deleted entirely** (163 lines). `DecisionHead.tau_kappa`
(`decision.py:47-59`) is its only live surface in the e2e family, and
`counterpart_membership` is `tau_kappa`'s only consumer. The rest of `DecisionHead` is
already dead there (`train_egostitch.py:2986-2996`). This also removes the
`requires_grad_(False)`-except-`tau_kappa_raw` special case at `:1063-1065`, which
would otherwise leave a permanently-zero-gradient parameter inside the generator group
and trip `enforce_nonzero` (`:1156-1157`) and the family probe's `norm <= 0.0` check
(`:3731`).

**Known consequence, accepted.** `build_scaffold` is structure-only — slot content `h`
never enters it (`scaffold.py:424-434`, pinned by
`tests/model/test_egostitch_scaffold.py:178`). With the content wire gone, slot
semantics no longer reach the scored logit at all; they survive only as auxiliary-loss
pressure shaping `pi`, `mult` and `adj`. The `ImaginedGraph` contract admits richer
node features specifically so the deferred redesign can restore content *through* the
graph rather than around it — a change to one generator, not to any interface.

**Decomposition.** `decompose()` returns `{full, f_logit}`. `_SCORES_META_VERSION`
(`score_universe.py:109`) bumps; `ScoresArtifact.pair_content` and `.pair_topology`
(`:191-194`) become optional rather than being deleted, so existing artifacts still
load. `save_scores`' hard requirement for all four arrays (`:650-654`) and
`validate_score_precision`'s companion check (`:542-554`) relax to the two required
arrays. `_e2e_primary_logit_key` (`:119-125`) loses its `content_head` branch. The
`pair_topology` arm configs are deleted — with content gone they are identical to
`full`.

**Ordering constraint.** `E2EConfig.from_mapping` rejects unknown keys
(`config.py:54-56`), so removing `p_cont` from the dataclass and from all eleven
`configs/egostitch_e2e*.yaml` files must land in one commit.

**Mask-stream stability.** `sample_branch_masks` draws `topo` before `cont` from the
same seeded generator (`conditioning.py:42-43`). Removing the `cont` draw leaves the
`topo` stream bit-identical provided the `topo` draw stays first.

## 10. Registration excision

The preregistration / formal-run / provenance-gating machinery is removed in full. The
owner is running experiments directly; the contracts in `docs/` no longer gate
execution.

**Deleted wholesale:** `src/experiments/e2e_binding.py` (50 lines, entirely
registration); `_validate_e2e_formal_plan`, `_preregistration_snapshot`,
`PreregistrationSnapshot`, `FormalPlanMismatch`, `_E2E_FORMAL_ARMS`,
`_E2E_CONTROL_ARMS`, `_E2E_ARMS` (`train_egostitch.py:1427-1587`);
`_validate_e2e_run_provenance`, `_validate_e2e_scoring_provenance`, `_registered_path`,
the `_run_score` formal block (`score_universe.py:1068-1380, 2376-2490`);
`PreregistrationMismatch`, `RegistrationShaMismatch`, `enforce_e2e_preregistration`,
`enforce_frozen_inputs`, `enforce_e2e_frozen_inputs`, `_enforce_e2e_formal_metadata`,
`_enforce_e2e_evaluator_seed`, `_registered_training_seeds`,
`_enforce_registered_training_seed`, `_formal_vhold_evaluation_disclosure`,
`_e2e_registration_sha256`, the g5 copy of `_validate_e2e_scoring_provenance`
(`g5_stage1.py`, ~750 lines total); the registration preflight in
`produce_e2e_probe_artifact` (`probes.py:746-845`); `hpc/qualification.sh`;
`docs/registrations/` (12 files); the `preregistration:` key in eleven configs;
`.claude/skills/formal-run-protocol/` and `.claude/skills/hpc-execution/` with their
`.codex/` mirrors.

**Preserved:** `g5_stage1.py`'s ~1050 lines of real evaluation logic — matched-RD
selection and assembly (`:422-487`), `validate_dead_residual_within_checkpoint`
(`:490-571`), `paired_bootstrap_lower_bound` (`:285-336`), clustering-MMD samples and
ratio (`:1141-1236`), the regime table and verdict rules (`:1814-1938`),
`render_e2e_tables_markdown` (`:1619-1757`), and `_enforce_engineering_evidence_class`
(`:1569-1616`, an evidence-class guard, not registration). The V_hold validation-event
ledger. `run_metadata.json` as a plain record. `model_config_hash`
(`train_egostitch.py:5067`) as an optional run identity.

**Three couplings that must be handled explicitly, not merely deleted:**

1. **`score_universe.py:390` is a silent-failure landmine.**
   `validate_test_access_ledger_binding` decides whether an artifact is a genuine
   held-out claim by testing for `formal_scoring_provenance` / `scoring_arm` /
   `checkpoint_arm` / `arm_kind` in `meta`. Delete those keys and every held-out
   artifact silently takes the "synthetic, skip validation" branch at `:394` — the
   test-access ledger stops being enforced, with no error raised. The held-out access
   ledger is a **data-boundary control, not registration**, and must survive. The
   discriminator is replaced by an explicit `heldout: true` flag in `meta`, written in
   the same commit that removes the provenance keys.

2. **`_e2e_arm_name` / `_e2e_arm_name_from_config` must survive**
   (`train_egostitch.py:3512-3527`). They read as registration but are behaviorally
   load-bearing: `arm` drives `select_e2e_checkpoint`, the precision differential
   (`:4500`, `:4783`), `conditioning_active` (`:4609`), and the V_hold ledger rows
   (`:4117`). They become plain config-derived labels with no registered arm set.

   **Amendment (owner decision, 2026-08-02):** `e2e_checkpoint_eligible` is deleted
   outright rather than rewired to a surviving consumer. Nothing in code evaluates
   checkpoint eligibility; `select_e2e_checkpoint` returns the best record by AUPRC →
   clustering MMD → Brier regardless of quality, and whether that checkpoint is usable
   is an owner-side judgement read from `metrics.jsonl`. The `checkpoint_eligible`,
   `selected_checkpoint_eligible` and `quality_fields_policy` keys are gone from
   `run_metadata.json`.

3. **Reader/writer pairs must move together.** `write_outputs` refuses to finalize
   unless `run_metadata.json` already carries `preregistration_sha256` and
   `config_hash` (`train_egostitch.py:5226-5232`); deleting the writer without editing
   the reader bricks every run at its last step. `repo_root` is derived from
   `cfg.preregistration.resolve().parents[2]` in five places
   (`train_egostitch.py:1503, 5148`; `g5_stage1.py:354`; `score_universe.py:1084`;
   `probes.py:778, 787`) and needs a real helper. `hpc/run.sh:152-162` hard-refuses
   `egostitch_e2e` and redirects to `qualification.sh`; with that script deleted, the
   refusal must be lifted or E2E has no launcher. Removing `preregistration:` from
   `load_config`'s allowed-key list (`:331`) and from the eleven YAMLs must be atomic,
   or `_check_no_unknown_keys` rejects every config.

**`run_kind` is load-bearing beyond registration** (`:1377, 4064, 5137, 5233-5235,
5311`; `e2_pipeline.py:876`): it drives `write_outputs`' consistency check and
`require_v_hold_validation_events`. Removing it silently disables the ledger
requirement, so it is retained as a plain field.

**Tests.** `tests/test_g5_e2e_registration_v5.py` (908 lines) is deleted entirely.
`tests/test_hpc_qualification.py` and `tests/test_e2_pipeline_qualification_failures.py`
are deleted. Registration-only assertions are removed from
`tests/test_g5_stage1_e2e.py`, `tests/test_score_universe_e2e.py`,
`tests/test_train_egostitch_training.py`, `tests/test_train_egostitch_core.py`,
`tests/experiments/test_probes.py` and `tests/test_egostitch_rev31_arm_schema.py`.
Real-behavior tests wrapped in registration fixtures keep their assertion bodies and
get new fixtures — notably `TestWithinCheckpointLivenessGuard`
(`test_g5_stage1_e2e.py:118`), `TestPairedBootstrap` (`:1588`),
`TestEngineeringEvidenceClass` (`:1690`), the seven test-access-ledger tests in
`test_score_universe_e2e.py:345-463`, and `test_bounded_e2e_worker_run_is_forbidden`
(`test_train_egostitch_core.py:227`).

## 11. Checkpoints

Existing e2e checkpoints stop loading. `score_universe.py:1457` and `probes.py:851`
call `load_state_dict` strictly, and this refactor removes `content_proj.*` and
`trunk.cont_xattn.*` and renames every remaining key under the new component prefixes.
No compatibility shim is written: the owner has confirmed prior runs do not matter.
`e2e_checkpoint_config`'s backfills for the rev-3.1 era (`config.py:20-42`) are removed
along with the checkpoints they served.

## 12. Phasing

**P0 — registration excision.** First, deliberately: it deletes roughly 250 KB of tests
that would otherwise need rewriting twice, once against the content removal and once
against the component config.

**P1 — content-path deletion.** Including `decision.py` and the artifact schema drop to
two logits. Mechanical, and it shrinks the model before restructuring.

**P2 — component extraction and trainer retargeting.** `ImaginedGraph` and the three
protocols; the existing code moves behind them unchanged. The trainer is retargeted in
the same change: parameter groups, `_e2e_family_probe` expectations, `_e2e_gate_tanh`
and the submodule-RMS telemetry, and the fp32 node-state cache. Behavior-preserving;
`tests/model/` is the proof.

**P3 — registry, nested config, null generator.** After this, setting
`generator.name: null` yields exactly the B0 pairwise baseline by config alone.

## 13. Acceptance criteria

1. `tests/model/` passes with its assertions intact, except those that test the deleted
   content path and `DecisionHead`.
2. The AB/BA invariants hold: `ema_updates == 1` per step, `ema_mu` is the joint AB∪BA
   mean, train and eval residuals agree, and a full endpoint swap gives identical
   logits under every surviving null.
3. `train_egostitch.py` completes a debug run end to end with no `preregistration`
   key present in the config.
4. Parameter groups are disjoint, exhaustive, non-empty, and named for the three
   components; each receives a nonzero gradient under the family probe.
5. A config with `generator.name: null` produces logits numerically equal to the same
   checkpoint's `f_logit` head, which is the existing exact per-sample bypass
   (`conditioning.py`, pinned by `tests/model/test_egostitch_conditioning.py`).
6. A held-out scoring artifact still fails validation when its test-access ledger is
   deleted or tampered with — proving the §10.1 discriminator replacement works.

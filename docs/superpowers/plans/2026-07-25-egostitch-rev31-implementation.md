# EgoStitch rev-3.1 relational repair — implementation plan

**Status: implementation-only.** The binding contract is `docs/05-egostitch-spec.md`
§14.4 (landed 2026-07-25, commit `0fef801`) plus the `docs/03-experiment-protocol.md`
§0 amendment. The decision trail is
`docs/superpowers/specs/2026-07-25-egostitch-e2e-relational-repair-design.md` (r6);
that file is a decision trail, **not** a contract — where the two differ, **spec §14.4
governs**.

§14.4 authorizes *implementation only*. **No task in this plan runs training,
scoring, a gate, or any GPU job.** Execution requires a fresh v3 registration with
`status: BINDING`, which does not exist yet.

Phase 0 is complete (`outputs/p0_audit_20260725/`). Waves below are WS1–WS7 of the
design's §5 table, resequenced for file-conflict safety.

## Context an implementer needs

Read the code maps in `.superpowers/sdd/2026-07-25-egostitch-rev31-implementation/`
instead of re-reading the large source files:

- `map-grounding.md` — `src/data/grounding.py`, pool caches, feature digests
- `map-model.md` — `src/model/egostitch/*` (SlotSet, stitch, scaffold, ste, losses,
  conditioning, e2e_model, matching, config)
- `map-train-score.md` — `src/train_egostitch.py`, `src/score_universe.py`,
  `src/experiments/probes.py`, `src/experiments/g5_stage1.py`

The maps are file:line-anchored and were built for this plan. Trust them for *where*
things are; verify against the source before editing.

## Global Constraints (binding on every task)

**Contract discipline**

1. `docs/05-egostitch-spec.md` §14.4 is the contract. If a task here contradicts
   §14.4, **stop and report** — do not resolve it yourself. Do not edit any file
   under `docs/registrations/` except where a task explicitly says so, and never
   edit a `status: BINDING` registration.
1b. **Files SHA-pinned by a BINDING registration are immutable.**
   `docs/registrations/g5_e2e_stage1_preregistration_v2.json` (`status: BINDING`)
   pins `binding_evidence.configs.*.sha256` for all four v2 arm configs:
   `configs/egostitch_e2e_breadth_first.yaml`,
   `configs/egostitch_e2e_f_only_breadth_first.yaml`,
   `configs/egostitch_e2e_pair_topology_breadth_first.yaml`,
   `configs/egostitch_e2e_p0_breadth_first.yaml`.
   Editing any of them changes its digest and makes
   `_validate_e2e_formal_binding` reject the registered arm — and silently mutates
   the historical v2 arm's semantics. **Every rev-3.1 config is a NEW v3 file**
   (Task 11), never an edit to a v2 file.
2. Do not run training, scoring, gates, or `hpc/*.sh`. Unit/integration tests only.
3. Every mechanism keeps the terminology guardrail: generated/stitched topology is
   *intermediate context* for predicting `edge(u, v)`. Nothing here is a
   graph-generation objective.

**Objective structure (spec §14.4.1) — exact, verbatim**

```
L_recon (family egostitch_e2e, rev-3.1) =
    1.0·L_feat + 0.5·L_exist + 0.25·L_mult + 0.5·L_deg
  + 0.5·L_slotadj + 0.25·L_gate + 0.25·L_ptr
  + 0.5·L_align + 0.1·L_div + 0.25·L_rel
```

The outer objective `L = L_edge + λ_real·L_real + λ_ssl·L_ssl + λ_recon·L_recon` is
locked (blueprint §10). **No fifth top-level term may be created** — every new loss
folds into the `L_recon` decomposition.

Per-component anneal (spec §14.4.1): the registered 1.0 → 0.25 anneal across the
edge-active phase applies **only** to `{L_feat, L_exist, L_mult, L_deg}`. The repair
components `{L_slotadj, L_gate, L_ptr, L_align, L_div, L_rel}` stay at factor 1.0
throughout. Outer `λ_recon` is untouched.

**Integrity gates (protocol §E5) — non-negotiable**

4. All new supervision is **train-side only** and derived from `E_msg` /
   `G_fit = E_msg[V_fit]`. Never read test-graph edges. Node-disjointness is
   untouched. Retrieval and grounding for test nodes stay feature-only.
5. Attribution split: `c_topo` stays **structure-only at inference**. Every new loss
   shapes training only; no content/grounding channel enters the eval-time STE
   inputs. The `L_rel` head is discarded at inference and is never part of any
   scored logit — the four-logit decomposition (`full`, `f_logit`, `pair_content`,
   `pair_topology`) and the null taxonomy are unchanged.
6. `*_ratio5_exclusive.txt` is quarantined — no loader may read it.
7. `train_graph.pkl` is for split audits only. Everything structural comes from
   `build_g_struct` (`src/data/partition.py:69`).

**Determinism / DDP**

8. Reduction contract (spec §14.4.1, §13.19.1 class): `L_rel` and `L_align` are
   `edge_mask`-weighted (for `L_align`: positive-and-real-row-weighted) **global**
   means with an **all-reduced real-row denominator**. `_edge_tensors` pads
   rank-local tail batches by duplicating the first row — a plain local mean would
   count duplicates and weight ranks unequally.
9. Target sampling (spec §14.4.1): edge-stream ego-target subsets are keyed
   `rng(blake2b(node_id) ⊕ (seed, epoch))` — **never** rank, step, or pair. `L_rel`
   targets use no RNG. The invariance contract is **per-pair** (a given pair gets
   identical targets/losses at any world size), not stream-composition invariance;
   negative stream composition stays `(seed, epoch, rank)`-drawn as in v2.
10. fp32 discipline (spec §13.16): promote `h`/`π`/`m` to fp32 **before** forming
    costs and marginal products. Match the existing fp32-island pattern
    (`stitch.py:80-108`, `conditioning.py:100-104`, `e2e_model.py:363-368`) — do not
    invent a fourth style.
11. Never shard the Stage-1 scorer. `_score_egostitch_e2e` derives its universe from
    the caller's `universe_pairs`; any new call site must forward it.

**Naming (pinned; dataset-agnostic placeholders — never substitute real names)**

12. Trained arms (6): `full`, `b0_e2e_f_only`, `pair_topology`, `p0`, `cosine_pool`,
    `no_l_rel`. Scoring-time controls (2): `structure_control_6a_v3`,
    `structure_control_6e_v1`. Never conflate `B0` with `B0-e2e`.
13. Null-head naming is inverted at `src/model/egostitch/e2e_model.py:416`:
    `pair_content` comes from `NULL_TOPO_HEAD`, `pair_topology` from
    `NULL_CONTENT_HEAD`. **Do not "fix" this** — swapping them mislabels two
    published arms.

**Engineering**

14. TDD: write the failing test first, then the implementation. Every task lists its
    required acceptance tests; those are a floor, not a ceiling.
15. Test command: `.venv/bin/python -m pytest -q -p no:cacheprovider` (full suite;
    baseline is green — ~930 passed, 4 skipped). Also run
    `.venv/bin/python -m ruff check src tests` and
    `.venv/bin/python -m mypy src` (never run two mypy processes concurrently — the
    cache corrupts).
16. Match surrounding code style: type hints everywhere, dataclasses for config,
    module docstrings referencing the governing spec section.
17. Commit per task with a message naming the spec section (e.g.
    `feat(grounding): pool_method_hash cache binding (spec §14.4.4)`).
18. Do not change existing registered values that this plan does not name. Adding a
    config field changes `_config_hash` — that is expected and fine for rev-3.1
    (a fresh v3 registration is required anyway), but never rename or drop an
    existing field without a task telling you to.

**Measured Phase-0 constants (use these exact values; source
`outputs/p0_audit_20260725/p0_audit_results.json`)**

| Constant | Value | Provenance |
|---|---|---|
| `n_ground` (rev-3.1 `full` and all v3 arms except `cosine_pool`) | `50`, set **explicitly** in each v3 config | §14.4.4 |
| `n_ground` (`cosine_pool` arm) | `20`, set explicitly | §14.4.6 |
| `E2EConfig.n_ground` dataclass default | `20` — the legacy value, so an absent key never silently reinterprets a v2 config or a pre-rev-3.1 checkpoint | derived (see below) |
| slot-recall ceiling @ top-50 | `0.13952495387963418` | P0.2 |
| slot-recall ceiling @ top-20 | `0.10728125418065595` | P0.2 |
| G3 gate 1 threshold @ n_g=50 | `0.0698` (= 0.5 × ceiling) | §14.4.7 + P0.2 |
| `L_gate` pos-weight | `6.17` (= (1 − 0.1395)/0.1395) | §14.4.1 "registered constant from the measured in-pool rate" |
| `P(|S| > 0)` (L_align teacher) | `0.663` | P0.4 |
| collapse abort: `h`-cosine | `> 0.95` | §14.4.8 |
| collapse abort: Π rank-1 residual | `< 0.05` | §14.4.8 |
| collapse abort: consecutive validations | `2` | §14.4.8 |
| `τ_div` | `0.5` | §14.4.1 |
| `τ_adj` | `0.5` (implementation default; `< 1` is the spec constraint) | §14.4.2 |
| ego-target cap `K` | `16` | §14.4.1 / `ego_targets.py` |

`τ_adj` and the `L_gate` pos-weight are **calibration-time values that the v3
registration must freeze**. Expose both as config fields with the defaults above;
never hard-code them at a call site.

**Why the `n_ground` dataclass default is 20, not 50** (Codex review of Wave A, two
P1 findings): checkpoints store `model_config` verbatim
(`train_egostitch.py:4986`), and every pre-rev-3.1 e2e checkpoint — including the
completed rev-3.0 run behind `docs/results/G5-e2e-stage1-seed0-20260724.md` — has no
`n_ground` key, because the field did not exist. The v2 arm configs have no key
either, and they are SHA-pinned (constraint 1b). A dataclass default of 50 would
silently re-resolve both to 50, changing the grounding pool the scorer builds from
`model.generator_cfg.n_ground`. Defaulting to 20 makes "absent ⇒ 20" uniformly true
for checkpoints and configs alike, needs no special-case load logic, and leaves the
v2 digests intact. The registered rev-3.1 value of 50 is carried by each v3 config
file explicitly, which is where §14.4.4 binds it.

---

# Wave A — Grounding vocabulary (WS1, Fix A)

## Task 1: `pool_method_hash` cache binding and `n_ground = 50`

**Spec:** §14.4.4. **Files:** `src/data/grounding.py`, `src/model/egostitch/config.py`,
callers of `build_grounding_pool`, `configs/egostitch_e2e_breadth_first.yaml`,
`tests/data/test_grounding.py`.

**Why this exists:** `build_grounding_pool`'s `.npz` cache validates only `node_ids`
(exact order) and `n_ground` (`grounding.py:94-106`). A redefined pool read through
an existing cache path silently serves stale pools — corrupting `L_gate`, `L_ptr`,
and probe labels with no error. `tests/data/test_grounding.py:58-64` proves the gap:
a `np.zeros_like` feature matrix still returns the original cached pool. Worse, a
stale grounding cache is currently **overwritten in place** after a warning
(`grounding.py:58`), unlike the F0 path.

**Implement:**

1. `n_ground` becomes settable per arm. `E2EConfig` currently has **no** `n_ground`
   field and always forces the pinned `EgoStitchConfig().n_ground`
   (`train_egostitch.py:1499-1511`). Add the field, default **`20`** (see the
   constants block for why the default is the legacy value, not 50), plumbed from
   YAML so each v3 config can state its value explicitly. **Do not edit any v2 arm
   config** — they are SHA-pinned by the BINDING v2 registration (constraint 1b).
   The v3 config files that carry `n_ground: 50` are created in Task 11.
2. Add `pool_method_hash` to the cache `.npz` schema. It is a SHA-256 over a
   canonical, order-stable serialization of exactly:
   - method id — the literal string `cosine_topk_v1`
   - `n_ground`
   - shortlist `M` — omitted entirely when absent (there is **no reranker**; the
     field exists so a future two-stage method cannot collide with this one)
   - the ordered F0/source-feature-pack digest: SHA-256 over
     `np.ascontiguousarray(features, dtype=np.float32).tobytes()` **plus** the
     ordered `node_ids` — this is what catches "mutated features, same ids"
   - role-universe identity — a caller-supplied string (see 3)
3. Role-universe identity becomes an **explicit required argument**, not caller
   convention. Today `build_grounding_pool` has no notion of role (V_fit / V_qual /
   V_select / test); nothing stops two callers colliding a cache path across roles
   (`probes.py:560-573` shows the three-file convention). Add the argument, thread it
   through every call site, and make the cache reject a hash mismatch.
4. **Fail closed.** On any `pool_method_hash` mismatch — or a cache missing the field
   entirely (v2 caches) — raise, with a message naming the expected and found hash
   and the cache path. Do **not** warn-and-recompute, and do **not** silently
   overwrite. Delete the in-place overwrite behavior at `grounding.py:58`.
5. Enumerate in the module docstring the full cache-regeneration set (four role
   universes) that this change invalidates. Do not regenerate caches — that is
   execution.

**Acceptance tests (all new, in `tests/data/test_grounding.py`):**

- Pool determinism: two builds on identical inputs give byte-identical pools and the
  same `pool_method_hash`.
- **Stale-method rejection**: a cache written with `n_ground = 20` (or a different
  method id) is rejected when `n_ground = 50` is requested — raises, does not
  recompute silently, does not overwrite.
- **Mutated-features-same-ids rejection**: the existing `np.zeros_like` motif at
  `tests/data/test_grounding.py:58-64` now raises instead of returning the stale
  pool. Update that test to assert the new behavior.
- Role isolation: the same node set + `n_ground` under two different role identities
  produce different `pool_method_hash` values and cannot share a cache.
- Own-split-side legality: the pool for a node contains only nodes from that node's
  own role universe.
- `E2EConfig(n_ground=50)` round-trips through YAML and reaches the pool builder.
- **Legacy resolution**: a stored `model_config` (or a v2 config mapping) with no
  `n_ground` key resolves to `20`, never `50`; a mapping stating `n_ground: 50`
  resolves to `50`.
- **v2 digests intact**: the four SHA-pinned v2 arm configs listed in constraint 1b
  hash to the values recorded in
  `docs/registrations/g5_e2e_stage1_preregistration_v2.json`'s
  `binding_evidence.configs.*.sha256`. Assert this in a test — it is the guard that
  keeps a frozen BINDING arm loadable.

**Out of scope:** any reranker (measured and rejected, §7.1.2); regenerating caches;
touching `score_universe`'s scoring path beyond the call-site signature update.

---

# Wave B — Scaffold, controls, soft matching (WS4, WS3, Fixes D, B2)

## Task 2: Scaffold closure channel — FEAT_DIM 9→10, EDGE_TYPES 3→4

**Spec:** §14.4.2 (scaffold bullet). **Files:** `src/model/egostitch/scaffold.py`,
`src/model/egostitch/ste.py`, tests under `tests/model/`.

**Why this exists (F6):** with `Â`/`Π` collapsed, the scaffold reduces to a degree
code; no explicit triangle/closure signal exists anywhere in the structural inputs
(probe: degree R² 0.382 vs partialled clustering 0.021). r2 confirmed two of the
three originally proposed channels already exist — `feats[:, :, 6:9]` already carries
per-edge-type incident mass including `Σ_l Π[k,l]` (`scaffold.py:114`), and STE row
normalization never touched `feats` (`ste.py:46-47`). Only *closure* is missing.

**Implement (functions of `{star, Â, Π}` only — no content, no grounding):**

1. **D1 — per-slot closed-wedge feature**, FEAT_DIM 9 → 10:
   `t_k = [Π Â̊_other Π^T]_{kk}`, computed on the **zero-diagonal**
   `Â̊ = Â − diag(Â)`. The zero-diagonal is load-bearing: `sigmoid(h·h^T)` has a
   large diagonal, and `Σ_l Π[k,l]²·Â[l,l]` is alignment-concentration — another
   degree-flavored scalar, i.e. the exact failure mode under repair.
   `generated_ego_graph` already zeroes the diagonal for this reason
   (`losses.py:302-305`). Note `SlotSet.adj`'s diagonal is *not* zeroed at
   production and *is* read as meaningful self-loop signal in `decision.py:160-161` —
   zero it locally for this computation, do not mutate `SlotSet`.
2. **D2 — closure edge type**, EDGE_TYPES 3 → 4. Direction is pinned: the src↔dst
   closure block is the symmetrized wedge `C = ½ (Â̊_src Π + Π Â̊_dst)`, with
   `adj[CLOSE, s_src, s_dst] = C` and `adj[CLOSE, s_dst, s_src] = C^T`.
3. The degree-feature slice widens `6:9` → `6:10` to cover the new edge type.
4. `ste.py` token dims follow FEAT_DIM/EDGE_TYPES — update whatever derives from the
   constants, and make the shape contract **fail closed** (an input whose channel
   count disagrees with the constants raises, not broadcasts).

**Acceptance tests:**

- Shape contract fails closed: a scaffold built with the wrong FEAT_DIM/EDGE_TYPES
  raises a clear error.
- **Rebuild-symmetry** (not relabel-symmetry — that passes trivially because
  `swap_direction` only swaps anchor channels): the scaffold built for `(j, i)` with
  `Π' = Π^T` equals the side-permuted `(i, j)` scaffold, to fp32 tolerance, for both
  `feats` and `adj`.
- `t_k` is zero when `Â̊_other` is zero, and is invariant to the value of `Â`'s
  diagonal (the zero-diagonal regression test).
- `C` reduces to the expected value on a hand-built 3-slot example computed by hand.
- Existing scaffold/STE tests still pass with the widened dims.

**Out of scope:** the structure controls (Task 3), the soft matched flags (Task 4).

## Task 3: Structure controls — 6a-v3 rebuild form and 6e-v1 checkerboard rewiring

**Spec:** §14.4.5. **Files:** `src/model/egostitch/scaffold.py` (perturbation hook),
`src/score_universe.py`, tests.

**Why this exists (F12):** the v2 6a control shuffles **only** `scaffold.adj`, never
`scaffold.feats`, via a `forward_pre_hook` on `model.ste`
(`score_universe.py:122-146`, `:1969-1979`). Task 2 routes Π/Â-derived signal *into*
`feats`, which that shuffle would bypass — shrinking the control's coverage exactly
when G3 gate 4 makes inertness binding. And a control without a deterministic
algorithm is not registrable, which is why 6e was deferred in v2.

**Implement:**

1. A **scaffold-build-input perturbation** API: a perturbation is a deterministic
   function applied to `(Â_src, Â_dst, Π)` *before* the scaffold is built, after
   which the scaffold is rebuilt normally so **every** derived channel — `t_k`, `C`,
   the 6:10 degree slice — recomputes from the perturbed structure. Replace the
   `model.ste` pre-hook approach; the hook must move upstream of scaffold build.
2. **6a-v3**: within-pair slot-axis permutations of `Â_src`, `Â_dst`, `Π` applied at
   scaffold-build input, keyed by the existing v2 blake2b canonical-pair scheme
   (reuse it — do not invent new keying), then full rebuild. Per-slot `(π, m)`
   features stay fixed; the control's meaning is "destroy the binding between slot
   identity/mass and relational structure".
3. **6e-v1**: canonical-pair-keyed **checkerboard swaps**. `N_swap = 8·K²` keyed
   draws; each selects rows `(i, k)` and columns `(j, l)` and transfers
   `δ = u · min(w_il, w_kj)` with keyed `u ∈ (0, 1)`:
   `w_ij += δ; w_kl += δ; w_il −= δ; w_kj −= δ`. Applied to `Â̊_src`, `Â̊_dst`
   (symmetrized) and `Π`, then scaffold rebuild. Every row sum, column sum, and the
   total mass are preserved exactly — per-node, per-edge-type soft degree is
   invariant — while higher-order connectivity is destroyed. That is precisely 6e's
   registered purpose: isolating structure beyond degree.
4. Control identifiers, pinned: `_SCAFFOLD_CONTROL_SHUFFLE_V3 = "shuffle_within_pair_v3"`,
   `_SCAFFOLD_CONTROL_REWIRE_V1 = "rewire_checkerboard_v1"`, alongside the existing
   `"none"`. The v2 `"shuffle_within_pair"` value is superseded — reject it with a
   message pointing at §14.4.5 rather than silently accepting the adj-only form.

**Acceptance tests:**

- 6e-v1 marginal preservation: row sums, column sums, and total mass preserved to
  fp32 tolerance on random inputs, for `Â̊_src`, `Â̊_dst`, `Π`.
- 6e-v1 symmetry: the symmetrized `Â̊` blocks stay symmetric after swaps.
- Cross-process determinism: the same pair key yields identical perturbed tensors in
  a fresh process (subprocess or re-seeded fixture), and different pairs differ.
- **Non-inertness (the F12 regression test)**: on a random *non-collapsed* model,
  both 6a-v3 and 6e-v1 measurably move the STE output. This is the check the v2
  battery could not make.
- 6a-v3 perturbs `feats` too: assert the rebuilt `feats` (specifically `t_k` and the
  6:10 slice) differ from the unperturbed build — the exact coverage gap this task
  closes.
- The v2 control value is rejected.

**Out of scope:** wiring the controls into the arm schema / CLI enum (Task 11).

## Task 4: Soft differentiable matched flags (B2)

**Spec:** §14.4.2 (soft matched flags). **Files:**
`src/model/egostitch/scaffold.py`, `src/model/egostitch/e2e_model.py`, tests.

**Why this exists (F1):** the pointer head receives zero gradient from any loss; its
only consumer is `grounded_identity_match` via a non-differentiable argmax
(`scaffold.py:56`).

**Implement:** with per-pair shared-id indicator `I[g_a, g_b] = 1[id_a(g_a) = id_b(g_b)]`
(`n_g × n_g`):

```
M[k, l]      = (p_a I p_b^T)[k, l]
matched_a[k] = gate_a[k] · max_l (M[k, l] · gate_b[l])
```

The BA direction uses `M^T`, so AB/BA consistency holds by construction. The max is
over the **product** — not an argmax over `M` followed by a gate lookup, which
reintroduces a switching discontinuity. Disjoint pools give `I = 0` ⇒ `matched ≡ 0`
with zero gradient; that is correct behavior, state it in the docstring. Train and
eval both use the identical soft form (no train-only branch).

**Claim discipline for the docstring:** with near-uniform untrained pointers,
`M ≈ |pool overlap| / n_g²`, so the `L_edge`-through-content gradient to the pointer
is *vanishing at init*. B2 is a plumbing repair that lets gradient exist; `L_ptr`
(Task 5) and the pos-weighted `L_gate` do the real training work. Do not oversell it.

**Acceptance tests:**

- `p(i, j) = p(j, i)` under every null (`none`, `NULL_TOPO_HEAD`,
  `NULL_CONTENT_HEAD`, `p0`).
- Train-mask ≡ eval-bypass equality: re-run the existing equality test and assert the
  soft form does not break checkpoint-exact null bypass.
- **Gradient reaches `head_pointer`** — the explicit F1 regression test: build a
  pair with overlapping pools, backprop a scalar function of `matched_a`, assert
  `head_pointer` parameters have non-`None`, non-zero `.grad`.
- Disjoint pools ⇒ `matched ≡ 0` and gradient is exactly zero (not NaN).

---

# Wave C — Losses and edge-stream targets (WS2; Fixes A3, B1, B3, C1, C2, F.1)

## Task 5: Slot-level losses — `adj_logits`, `L_slotadj`, `L_div`, `L_ptr`, pos-weighted `L_gate`

**Spec:** §14.4.1, §14.4.2. **Files:** `src/model/egostitch/imagine.py`,
`src/model/egostitch/losses.py`, `src/model/egostitch/config.py`,
`src/model/egostitch/model.py`, tests.

**Implement:**

1. **C2 — `SlotSet.adj_logits`.** `imagine.py:218-221` computes the pre-sigmoid
   logits as a local variable and immediately sigmoids them into `SlotSet.adj`; the
   raw logit is discarded. Expose it as a `SlotSet` field (a §2 head-contract
   extension — an API change, not a losses-only edit). Update every construction site
   of `SlotSet`. `L_slotadj` becomes `BCEWithLogits(adj_logits / τ_adj, target)` with
   config field `tau_adj` (default `0.5`, must be `< 1`; validate).
2. **A3 — pos-weighted `L_gate`.** Replace plain BCE with pos-weighted BCE, weight
   from config field `l_gate_pos_weight` (default `6.17`, derived
   `(1 − 0.1395)/0.1395` from the P0.2 top-50 slot-recall ceiling — document the
   derivation). With the now-nonzero positive labels this lets gates cross 0.5 when
   evidence exists; in v2 `target_in_pool ≈ all-False` drove the gate to 0 and forced
   the Π-consistency probe to a structural zero. Note `target_in_pool` is *not*
   constructed in `src/model/egostitch/` — it originates at
   `train_egostitch.py:2483` and threads through as a plain argument.
3. **C1 — `L_div`.** `L_div = mean_{k≠l} max(0, cos(h_k, h_l) − τ_div)²` with
   `τ_div = 0.5` (config field), **restricted to slot pairs where at most one slot is
   Hungarian-matched**. Two matched slots may legitimately be similar when a node's
   true neighbors share a community; penalizing that fights `L_feat` and `L_align`.
4. **B1 — `L_ptr`.** For each Hungarian-matched slot whose target neighbor is in the
   pool: `L_ptr = CE(pointer_k, pool_index(t_k))`. Unmatched and out-of-pool slots
   are masked. This is the missing pointer gradient (F1); without pool coverage there
   are no labels, which is exactly why it could not have worked in rev-3.0.
5. Register all four in the `L_recon` decomposition at the Global-Constraints
   weights, each with its own telemetry family name following the existing
   convention.

**Acceptance tests:**

- `L_recon` weight table matches the spec string exactly (a test that pins all ten
  weights — this is the anti-drift guard).
- `L_slotadj` in logit space equals the mathematical BCE on a hand-computed example;
  `τ_adj ≥ 1` is rejected by config validation.
- `L_gate` with pos-weight reproduces a hand-computed value on an imbalanced toy
  batch; pos-weight `1.0` reproduces the old plain-BCE value (backward-compat proof).
- `L_div` is exactly 0 when all pairwise cosines are below `τ_div`; is positive above
  it; and is **unaffected** by two mutually-matched slots being identical (the
  matched-pair exclusion).
- `L_ptr` masking: a batch where no slot's target is in the pool contributes exactly
  zero and produces no NaN; a hand-built batch with one in-pool target reproduces the
  hand-computed CE.
- Gradient flows from `L_ptr` to `head_pointer`.

## Task 6: Edge-stream ego-target assembly

**Spec:** §14.4.1 (target sampling). **Files:** `src/train_egostitch.py`, tests.

**Why this exists:** the edge stream currently builds **no ego targets** —
`_edge_tensors` (`train_egostitch.py:2504-2545`) assembles only features, grounding,
and labels; targets exist only in `_node_tensors` (`:2436-2479`). `L_align` (Task 7)
cannot exist without this machinery. This task is pure infrastructure — no loss.

**Implement:**

1. Per-endpoint ego-target assembly for edge-stream **positive** pairs (both
   endpoints train-side with `E_msg` egos): target gathers `(B, K, d)` features plus
   multiplicity, adjacency, and mask. Targets are capped at `T ≤ K = 16`
   (`EgoTargetBuilder(slots=…)`, `ego_targets.py:149-171`) — the cap is 16, not 20.
2. **Seeding — the load-bearing pin.** The node-stream target RNG includes the DDP
   rank (`rng((seed, epoch, step, rank, 0x7A))`, `train_egostitch.py:2564`). Mirroring
   it would make a pair's sampled ego targets — and therefore `L_align`'s teacher
   cells, gradients, and potentially checkpoint selection — depend on which rank the
   pair lands on. Use **per-endpoint node-identity keying** instead: each endpoint's
   subsample is drawn from `rng(blake2b(node_id) ⊕ (seed, epoch))` — never rank,
   never step, never pair. A node's target subset is then fixed within an epoch
   regardless of which pairs, ranks, or directions it appears in, giving AB/BA
   symmetry and cross-pair consistency by construction.
3. Memory-manifest accounting mirroring `:2566-2616` for the new tensors.
4. Do **not** change `enumerate_edge_stream`'s negative drawing
   (`:1722-1758`, `(seed, epoch, rank)`-keyed) — that registered stream behavior is
   unchanged, and the invariance contract is per-pair, not stream-composition.

**Acceptance tests:**

- A node appearing in multiple pairs, in both directions, and at different steps
  receives the **identical** target subset within an epoch.
- Changing `epoch` changes the subset; changing `rank` or `step` does **not**.
- Cap: no endpoint yields more than `K = 16` targets; nodes with fewer than `K`
  true targets are masked correctly (no padding leaks into the mask).
- Padded filler rows (`edge_mask = 0`) carry no targets that could be counted.
- Memory-manifest accounting includes the new tensors (assert the manifest total
  changes by the expected amount).

## Task 7: `L_align` (B3) and `L_rel` (F.1) with the reduction contract

**Spec:** §14.4.1. **Files:** `src/model/egostitch/losses.py`,
`src/model/egostitch/config.py`, `src/train_egostitch.py`, tests. Depends on Task 6.

**Implement:**

1. **B3 — `L_align`, the core relational fix.** On edge-stream **positive** pairs,
   run the per-endpoint Hungarian matching and define teacher cells
   `S = {(k, l) : target_id_a(match(k)) = target_id_b(match(l))}` — slot pairs
   generated from the *same real shared neighbor*. Then:

   ```
   L_align = −(1/|S|) Σ_{(k,l)∈S} ½·[ log(Π[k,l] / Σ_{l'} Π[k,l'])
                                    + log(Π[k,l] / Σ_{k'} Π[k',l]) ]
   ```

   Row- and column-**conditional** concentration only — deliberately orthogonal to
   the Sinkhorn marginals, which are **not** a gradient target. (The globally
   normalized form has an irreducible ≈ log(#matched) floor and pushes gradient into
   inflating `π·m` against `L_exist`/`L_mult`.) Pairs with `S = ∅` are skipped.
   `L_align` does **not** depend on grounding pools — Hungarian matches slots to true
   neighbors directly — so it repairs the relational channel independently of the
   pool ceiling. Eval-time Π stays content-cost-only; the loss shapes `h` so
   same-neighbor slots become mutual nearest neighbors. Hungarian at 16×16 is
   CPU-trivial; positives are ~1/6 of the edge stream. P0.4 measured
   `P(|S| > 0) = 0.663`, `E[|S|] = 2.12` — the teacher is healthy and the uncapped
   fallback is **not** used.
2. **F.1 — `L_rel`.** From the STE's pair state (mean over scaffold tokens of the
   **AB-direction** output), a 2-layer head predicts two train-pair relational
   targets: `log1p(common-neighbor count)` and neighborhood Jaccard, computed from
   `G_fit` **independently for every pair — positive and negative alike**. The
   `NegativeSampler` (`pairs.py:261-268`) rejects only canonicalized true positives,
   so sampled non-edges freely share neighbors; those wedge-bearing non-edges are
   precisely the hard cases the repair needs. Zero-labeling them would train the STE
   to *erase* closure structure on hard negatives and turn `L_rel` into an edge-label
   shortcut inside an allegedly structural task. Huber loss, weight `0.25`.
   The head attaches to the **STE only** — not the trunk, not the gates — is its own
   telemetry family, and is **discarded at inference** (never part of any scored
   logit).
3. **Reduction contract (both losses).** `edge_mask`-weighted (for `L_align`:
   positive-and-real-row-weighted) global means with an **all-reduced real-row
   denominator** and matching DDP scaling. An ordinary local mean would count
   `_edge_tensors`' duplicated filler rows and weight ranks unequally, making loss and
   gradients depend on world size and tail shape — the failure class §13.19.1 already
   forbids for the BCE.
4. Add the `w_rel` config field so the `no_l_rel` arm can set it to `0` (Task 11
   consumes this).

**Acceptance tests:**

- **Teacher-cell correctness on a hand-built graph**: a small graph with a known
  shared neighbor produces exactly the expected `S`; a pair with no shared neighbor
  produces `S = ∅` and is skipped (contributes zero, no NaN).
- `L_align` invariance under AB/BA: reversing the pair gives the identical loss value
  and identical gradients.
- **Per-pair world-size invariance** of targets, losses, and gradients: world size 1
  vs 2, with reversed pair order (BA), unequal tail batches, and padded filler rows.
  Per-parameter gradient equality, not just loss equality.
- Masked all-reduced reduction: duplicated filler rows do not change the loss value.
- **Required `L_rel` regression motif**: a non-edge sharing ≥ 1 neighbor receives
  **nonzero** relational targets (the r3 structural-falsity fix).
- The `L_rel` head does not appear in any scored logit — assert the four-logit
  decomposition is byte-identical with the head present vs absent.
- Loss-family telemetry names follow the existing convention and are distinct.

---

# Wave D — Conditioning dynamics (WS5, WS6; Fixes E1, E2, E3, C3, E4)

## Task 8: Centered gated residual (E1)

**Spec:** §14.4.2 (centered gated conditioning). **Files:**
`src/model/egostitch/conditioning.py`, checkpoint payload, tests.

**Why this exists (F7/F8):** the conditioning pathway converged to a near-constant
calibration offset — `topology_delta = +0.144 ± 0.085` against an `f_logit` std of
2.85 — and that surviving degree-like residual actively degraded degree-corrected
AUPRC (guard fail 0.705903 vs floor 0.710260). An uncentered gated residual funds a
constant offset; centering removes the shortcut, so the pathway must earn *per-pair
variance* or contribute nothing.

**Implement:** `cls ← cls + active · tanh(g) · (XAttn(...) − μ)` where:

- μ is the mean over rows that are **both** pathway-`active` **and** real
  (`edge_mask = 1`; padded filler rows from `_edge_tensors:2508-2513` excluded),
  **all-reduced across ranks** so training math is independent of the auto-detected
  world size.
- The eval-time μ is a single **synchronized EMA**, updated post-all-reduce, so it is
  identical on every rank by construction and checkpoint content does not depend on
  which rank saves. Store it in the checkpoint.
- At eval the EMA is frozen ⇒ determinism holds and the checkpoint-exact null bypass
  holds: **inactive rows still receive exact identity** (`+ 0`), preserving the §14.2
  null taxonomy.

**Acceptance tests:**

- Checkpoint-exact bypass with EMA-μ: an inactive row's output is bit-identical to
  its input.
- Determinism: two eval runs over the same data give identical outputs.
- **World-size invariance of μ**: 1 vs 2 ranks on CPU (gloo) produce the same μ and
  the same post-injection activations.
- Padded filler rows do not contribute to μ.
- A constant XAttn output produces exactly zero injection (the calibration-knob
  regression test) — this is the point of the whole task.
- EMA round-trips through save/load and is frozen (not updated) in eval mode.

## Task 9: Curriculum, per-component anneal, collapse abort, degree telemetry

**Spec:** §14.4.1 (anneal), §14.4.3 (curriculum), §14.4.8 (abort + telemetry).
**Files:** `src/train_egostitch.py`, tests.

**Implement:**

1. **E2 — curriculum (supersedes §13.19.1 for rev-3.1).** Warm-start stays
   reconstruction-only (no `L_edge` at all). From the **first edge-active step**, the
   trunk, STE, gates, and both conditioning pathways train **jointly** — the v2
   Phase-A `pair_only` head start (`train_egostitch.py:748-759`) is removed. This
   *restores* design §4's stated intent; the Phase-A head start let the trunk
   converge before the gates ever opened. Branch dropout `p_topo = p_cont = 0.15` is
   unchanged. Note `E2EPhaseState.alpha` (the 0→1 ramp gating `pair_only` /
   `real_ssl_scale` / LR) is a **different axis** from the loss-weight anneal below —
   do not conflate them.
2. **E3 — per-component anneal factors.** The registered 1.0 → 0.25 anneal across the
   edge-active phase applies **only** to `{L_feat, L_exist, L_mult, L_deg}` — the
   per-node fidelity components measured in the 500× / 1,682× aux:edge gradient
   imbalance. `{L_slotadj, L_gate, L_ptr, L_align, L_div, L_rel}` stay at 1.0
   throughout: a blanket `λ_recon` anneal would multiply the core relational fixes
   down 4× exactly when they matter. Implement as per-component schedule factors
   inside the `L_recon` decomposition; the outer `λ_recon` is untouched.
3. **C3 — collapse telemetry and abort.** Log the four dispersion statistics every
   validation (π std across slots, mean pairwise `h` cosine, `Â` off-diagonal std, Π
   row entropy). Registered death-style rule: mean pairwise `h` cosine `> 0.95`, **or**
   the Π rank-1 marginal residual `‖Π − r c^T / m‖_F / ‖Π‖_F < 0.05` (with `r`/`c` the
   row/column sums and `m` the total mass — "the plan carries nothing beyond its
   marginals"), for **2 consecutive validations after conditioning activates** ⇒
   `training_invalid(slot_collapse)` abort, following the existing §13.19.2 abort
   convention. **Do not use a Π row-entropy criterion as an abort arm** — P0.1
   measured it blind to full collapse (0.624 normalized on a fully collapsed plan,
   because the `π·m` marginals concentrate rows regardless of cost-blindness). Row
   entropy is logged as telemetry only. Calibration basis from the collapsed v2
   checkpoint `a471010f57e495f0`: `h`-cosine 0.9997, `Â` off-diagonal 0.5014 ± 0.0004,
   π std 0.042.
4. **E4 — degree-decorrelation telemetry** (watchdog only, no verdict effect): per
   validation, the correlation of the `full − f_logit` residual with endpoint degree,
   reported in every headline table.

**Acceptance tests:**

- Phase-state unit tests: no `pair_only` phase exists; conditioning is active at the
  first edge-active step; warm-start has no `L_edge`.
- Per-component anneal factors: at anneal end, the four fidelity components are at
  0.25 and the six repair components are at 1.0 (assert each, by name).
- Collapse abort **fires** on a synthetic collapsed run (both arms tested separately:
  `h`-cosine and the Π rank-1 residual) and only after 2 consecutive validations, and
  only after conditioning activates.
- Collapse abort does **not** fire on a synthetic healthy run.
- A fully collapsed plan with concentrated rows (row entropy 0.624-like) still
  triggers the rank-1 residual arm — the P0.1 blindness regression test.
- E4 telemetry appears in the validation record with the expected key.

---

# Wave E — Probes, gate, arm schema (WS7, Fix G)

## Task 10: Probes — Π-consistency v2, slot recall, shared-neighbor R², dispersion

**Spec:** §14.4.7. **Files:** `src/experiments/probes.py`, tests.

**Why this exists (F11):** the v1 Π-consistency probe conflates alignment quality
with grounding quality — it reads 0 whenever the gates are shut, regardless of Π
(`probes.py:608-628`), which is exactly what happened in v2.

**Implement:**

1. **G1 — Π-consistency v2**: plan mass on double-Hungarian same-identity cells (the
   same `S` as `L_align`) divided by total plan mass. This measures *alignment*
   directly, independent of gates and pointers. **Data source, pinned:** for the
   formal post-run probe artifact, probe pairs and matching targets derive from the
   train-side `G_struct` (`E_msg` over operative train nodes) — the same
   evaluation-side-only basis the current probe already uses (`probes.py:611`). No
   test-side structure is touched, ever. Keep v1 alongside for continuity; its honest
   scope is the grounding chain, and its docstring must say so.
2. **G2 — new registered probes**: slot recall@`n_g` (per e2e run, not only
   frozen-s0); shared-neighbor-count R² from STE pair states (the direct measure of
   "did a relational representation form"); the four P0.1 dispersion statistics.
3. Bump the probe artifact schema version to `egostitch_e2e_probe_v2` and **reject**
   the old version rather than coercing it.
4. Pre-binding scoping: calibration-time instances of these probes compute on
   `G_fit`; the qualification rehearsal computes on `E_msg[V_qual]`. Make the scope
   an explicit argument — never a default that could silently read a sealed universe.

**Acceptance tests:**

- v2-consistency on a hand-built alignment: a plan whose mass sits entirely on
  same-identity cells scores 1.0; a uniform plan scores the expected small value.
- v2 is **nonzero when gates are shut** (the F11 regression test — v1 reads 0 there).
- Old-version probe artifacts are rejected with a clear error.
- The scope argument is required; a sealed-universe scope cannot be reached by
  default.
- Slot recall@`n_g` on a hand-built pool reproduces the hand-computed value.

## Task 11: Eight-arm schema migration

**Spec:** §14.4.6. **Files:** `src/train_egostitch.py`, `src/score_universe.py`,
`src/experiments/g5_stage1.py`, `configs/`, tests.
**Order:** Task 12 runs **before** this task — the v3 config files' `preregistration:`
field must point at a draft registration that already exists.

**Why this exists:** formal-arm validation is duplicated across the stack and each
site rejects unknown arms, so a partial migration fails closed in the worst place —
mid-screen. Three independent definitions exist today: `train_egostitch.py:1227`
(unordered `set`), `score_universe.py:98` (ordered `tuple`), `g5_stage1.py:1212-1217`
(ordered `tuple`, plus a separate 5-entry `_E2E_ARMS` that adds the scoring-only
`structure_control_6a`).

**Implement:**

1. Migrate **every** formal-arm constant, binding-evidence validator, scoring CLI
   input and provenance enum, run-metadata schema field, and test fixture to:
   - six trained checkpoints: `full`, `b0_e2e_f_only`, `pair_topology`, `p0`,
     `cosine_pool`, `no_l_rel`
   - two scoring-time controls over `full`'s checkpoint: `structure_control_6a_v3`,
     `structure_control_6e_v1`
   Also migrate the worker's binding-evidence/config checks
   (`train_egostitch.py:1315-1388`) and the scoring CLI's arm/provenance/permanent-null
   enforcement (`score_universe.py:991-1110`, `:2155-2439`).
2. **A complete new v3 config set — six files, none of them edits to a v2 config**
   (constraint 1b: the four v2 arm configs are SHA-pinned by the BINDING v2
   registration and must hash unchanged):
   `configs/egostitch_e2e_v3_full_breadth_first.yaml`,
   `..._v3_f_only_...`, `..._v3_pair_topology_...`, `..._v3_p0_...`,
   `..._v3_cosine_pool_...`, `..._v3_no_l_rel_...`.
   Each states `n_ground` explicitly — `50` everywhere except `cosine_pool`, which
   states `20`. `no_l_rel` sets `w_rel: 0`; every other field matches `full`.
   `preregistration:` points at the Task 12 v3 draft, and `output_dir` follows the
   existing per-arm convention under a v3 root. Add a test asserting each v3 config
   differs from `full` in exactly its one intended field.
3. Bump the scores-`.npz` meta version and **reject** older versions.
4. Each arm's checkpoint provenance and scoring semantics (trained vs
   scoring-control) must be representable in run metadata.

**Acceptance tests (end-to-end, at every enforcement point):**

- A complete six-trained-plus-two-control package is **accepted** by
  `train_egostitch`, `score_universe`, and `g5_stage1`.
- A v2 five-arm package is **rejected** at every one of those three points — this is
  the fail-closed requirement.
- Unknown arm names are rejected (the existing behavior must survive).
- The two new configs load, produce the intended `n_ground` / `w_rel`, and differ
  from `full` in exactly that one field.
- Old scores-`.npz` meta versions are rejected.

## Task 12: v3 registration draft (non-binding)

**Spec:** §14.4.6, §14.4.7. **Files:** `docs/registrations/` (new v3 draft pair),
docs cross-references.

**Scope discipline:** this task creates a **DRAFT**, not a binding registration.
`status` must not be `BINDING`. Calibration-derived thresholds that require GPU runs
stay as `REQUIRED-BEFORE-BINDING` placeholders (the existing convention — see
`train_egostitch.py`'s `_REQUIRED_BEFORE_BINDING`). The owner binds it, not us.

**Implement:**

1. Draft `docs/registrations/g5_e2e_stage1_preregistration_v3.json` + its `.md` twin,
   modeled on the v2 pair, recording: predecessor v2, the eight-arm schema, the
   `L_recon` component table with all ten weights, the per-component anneal set, the
   grounding pin (`n_g = 50`, `pool_method_hash` fields), the curriculum change, the
   centered-injection equation, the two control algorithms with their identifiers,
   the probe set + artifact version, and the five G3 qualification gates with the
   Phase-0-derived values from this plan's constants table.
2. The `.md` twin is explanatory only — if the two disagree the JSON governs; say so
   in the twin.
3. Cross-reference the design trail and the P0 artifacts.

**Acceptance tests:**

- The JSON parses and validates against whatever registration schema validation
  exists; `status` is not `BINDING`.
- Every `REQUIRED-BEFORE-BINDING` placeholder is discoverable by a grep-style test
  (so nothing calibration-derived is silently filled with a guess).
- The arm list in the draft matches the code's eight-arm schema exactly (a test that
  reads both and compares).

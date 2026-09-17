# Motif-Graph GRIT Prompt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `v3_1_motif_prompt` model family — a fixed 26-slot, 96-edge motif graph predicted from `(x_u, x_v)`, read by a vendored GRIT stack and a closed-form count head, and injected into a frozen `prefix_base` trunk as four gated KV-prefix token fields — together with its deterministic template compiler, its `L_slot`/`L_topo` losses, its Stage I / Stage II configs and every §8 control, the two bounded pilots of §0.2, and the H20 launch sequence.

**Architecture:** A generator `G` reads the frozen `prefix_base` residue states `H_u, H_v` through learned slot queries and a two-layer gate MPNN over a fixed candidate template, and emits one weight in `[0,1]` for each of the 96 allowed undirected edges (16 closure `u-c`/`c-v`, 16 attachment `u-l`/`r-v`, 64 interior `l-r`). The same predicted adjacency feeds (a) a closed-form count head producing `topo_cnt` and (b) a three-layer, width-96, four-head GRIT reader over all 26 slots with RRWP `[I,P,P²,P³]`, producing `topo_u`, `topo_v` and a symmetric `topo_rel` from the final directed edge states. The four token fields become eight prefix rows per layer at all nine cross-attention sites of the frozen trunk, gated by zero-init per-head `tanh` gates, so gates-off reproduces `prefix_base` bit for bit. Stage I trains the reader/adapter on compiled true training templates; Stage II trains `G` against task BCE, the retained structural stream, the order-statistics loss `L_slot` and the representation term `L_topo` towards an immutable teacher copy of the Stage I bundle.

**Tech Stack:** PyTorch 2.10 (`nn.MultiheadAttention` packed `in_proj_weight`, `torch.autocast(enabled=False)`), vendored `src/vendor/grit_official.GritTransformerLayer` (constructed, never edited), accelerate DDP via `src/train_b0.py`, networkx + scipy.sparse for the compiler, pytest (`-n auto --dist loadfile`, `-n0` for debugging), ruff (Google docstrings, full annotations, no `print`), mypy strict.

**Spec:** `docs/superpowers/specs/2026-09-16-motif-graph-grit-prompt-design.md` (v8). Read it before Task 1; every section reference below is to that file.

## Global Constraints

- Local commands use `.venv/bin/python -m …`; `rtk` garbles `uv run` output. Debug with `-n0`; full runs need `--dist loadfile` (never `--dist load`) because tests in one file share state.
- `.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests` and `.venv/bin/python -m mypy src tests` must pass at the end of every task. Google docstrings and full annotations are enforced in `src/`; `print` is banned.
- `src/vendor/` is off limits: `GritTransformerLayer` is constructed with `O_e=True, norm_e=True` and `cfg["update_e"]=True`, never patched. Verified at `src/vendor/grit_official/grit_layer.py:461-464`: the layer already writes `batch.edge_attr = e` whenever `update_e` is set.
- No formalism gates: no digest pinning, no artifact/text-contract verifier, no eligibility or promotion ceremony, no check that blocks a run. Checkpoint paths and SHA-256s are recorded as provenance and never verified.
- Inference input is exactly `(x_u, x_v)`. Stage II scoring loads only the saved model and endpoint features — no node ids, no neighbour lookup, no truth graph, no bank, no universe statistic.
- Exact null identity: with every gate at zero the arm reproduces the frozen `prefix_base` logits bit for bit in eval mode (`torch.equal`), as `V3_1Prefix` already does.
- Fixed geometry: `|C| = |L| = |R| = 8`, 26 slots, 96 edges, weights in `[0,1]`, every other adjacency entry including the diagonal and the queried `u-v` entry exactly zero.
- `beta_p = beta_q = beta_a = beta_i = 1`, Huber `delta = 1`, `w_slot = 1`, `L_topo` weight `0.1` (§7.3, §7.5). `beta_i` may be set to 0 **only** by the pre-registered pilot-B measurement of §0.2.
- Reader: three layers, width 96, four heads, RRWP `[I,P,P²,P³]` (`rrwp_k = 4`), degree floor `1e-6`.
- Optimiser: AdamW, LR `1e-4`, weight decay `1e-2`, grad clip 1, 15-epoch one-cycle, no early stopping (`eval.patience: null`). Reader/prefix parameters run at 0.1× the instantaneous generator LR once opened, realised through `optim.groups`.
- The structural stream is retained in both stages with the seed-42 winner `{bce 1.0, rank 1.0, degree 0.1, motif 0.1}`, 40 nodes = 32 local + 8 background, `mix {bfs 0.5, motif 0.25, bridge 0.25}`. `struct.token_budget` is the only legal memory lever; `struct.subgraphs_per_epoch` and `runtime.max_pairs_per_rank` are **not** compute levers (§7.1).
- Claim rules: report edge-level and assembled-graph metrics together; each topology operating point reports BFS-macro GS, RD and the three MMD ratios together; the primary deployable number is the ONE V_val-selected fixed threshold (closest geometric RD); Accuracy/F1/MCC use the separate max-F1 threshold frozen on `val_cls`; ECE/Brier use raw probabilities.
- Margins for reading single-seed differences: ±0.01 GS and ±0.5 MMD ratio. These are reporting thresholds, not confidence intervals.
- GPU work is H20-only. Every task ends locally with tests, ruff, mypy and a commit. Task 30 is the only place that emits launch commands, and it does not run them.
- One independent Codex review per implementation wave that changes `src/`: `CODEX_HOME=<scratch>/codex-home codex review --base <sha> > wave-review.txt 2>&1`, backgrounded, read the file when it finishes.
- Commit messages end with the attribution line given in this session's system reminder.

---

## File map

| Path | Responsibility |
|---|---|
| `src/data/motif_template.py` | Create. Template geometry (slot roles, the fixed 96-edge list, `SWAP_PERM`, within-role permutation helpers), the deterministic `MotifTemplateTable` compiler with queried-edge removal, and `slot_profiles` — the §7.3 target tensor extraction shared by prediction and target. |
| `src/distill/motif_losses.py` | Create. `slot_loss` (`L_slot`, §7.3, including the raw interior term) and `topo_loss` (`L_topo`, §7.5). Sibling of `src/distill/struct_losses.py`, where every other training loss already lives; the *model* modules stay under `classifier/` as §9 requires. |
| `src/model/egostitch/classifier/motif_prompt.py` | Create. `MotifPromptConfig`, `MotifGenerator` (G), `MotifCountHead`, `MotifGritReader`, `MotifPromptCrossAttentionLayer`, `V3_1MotifPrompt`. All new model modules live here; `registry.py`/`composite.py` are untouched (§9). |
| `src/train_b0.py` | Modify. Family constants, `_resolve_motif_prompt_kwargs`, `_base_loss_kwargs` nesting, `build_model` branch, `MotifTemplateRows` aux-row carrier, stage-trainability hook, `optim.stop_after_epoch`, struct-stream wiring, `_run_metadata` provenance. |
| `src/score_universe.py` | Modify. `MODEL_BUILDERS` entry, truth-free scoring path, `--prefix-intervention` acceptance, the family's `score_precision` contract and its validator. |
| `src/e2_pipeline.py` | Modify. `last.pt` epoch check reconciled with `optim.stop_after_epoch`. |
| `configs/split_seed42/motif_prompt_*.yaml` | Create. Stage I, Stage II, teachability pilots and every §8 control. |
| `src/experiments/motif_pilot_a.py` | Create. §0.2 pilot A driver. |
| `src/experiments/motif_pilot_b.py` | Create. §0.2 pilot B driver with the pre-registered `beta_i` selector. |
| `src/experiments/motif_density_control.py` | Create. §8 output-density control. |
| `tests/data/test_motif_template.py` | Create. Compiler tests. |
| `tests/distill/test_motif_losses.py` | Create. `L_slot`/`L_topo` tests. |
| `tests/model/test_motif_prompt_model.py` | Create. Model tests. |
| `tests/test_train_b0_motif_prompt.py` | Create. Trainer plumbing tests. |
| `tests/test_score_universe_motif_prompt.py` | Create. Scoring and precision tests. |
| `tests/test_motif_prompt_configs.py` | Create. Config-schema tests. |
| `tests/helpers/motif_prompt_ddp_smoke.py` | Create. Two-rank CPU DDP smoke driver. |
| `tests/test_motif_prompt_ddp.py` | Create. DDP integration test. |
| `docs/results/motif_prompt/README.md` | Create in Task 29. Pre-registered reading, pilot results, run index. |
| `CLAUDE.md`, `AGENTS.md` | Modify in Task 29. Active-method-set entry for the new family. |

## Phase list

1. **Phase 1 (Tasks 1–2)** — family skeleton: geometry constants and `MotifPromptConfig`.
2. **Phase 2 (Tasks 3–6)** — `src/data/` compiler and the §7.3 target tensor.
3. **Phase 3 (Tasks 7–12)** — `motif_prompt.py`: G, count head, GRIT adapter, four-token prefix with per-field masking, frozen-base composition.
4. **Phase 4 (Tasks 13–15)** — `L_slot`, `L_topo`, and their wiring into the task stream and the structural stream.
5. **Phase 5 (Tasks 16–22)** — training and scoring plumbing.
6. **Phase 6 (Tasks 23–25)** — configs for Stage I, Stage II and every §8 control.
7. **Phase 7 (Tasks 26–28)** — the two bounded pilots of §0.2.
8. **Phase 8 (Tasks 29–30)** — docs and the H20 launch sequence.

---

## Phase 1 — Family skeleton

### Task 1: Template geometry constants

**Files:**
- Create: `src/data/motif_template.py`
- Test: `tests/data/test_motif_template.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `N_SLOTS = 26`, `N_EDGES = 96`, `SLOT_U = 0`, `SLOT_V = 1`, `C_SLOTS = tuple(range(2, 10))`, `L_SLOTS = tuple(range(10, 18))`, `R_SLOTS = tuple(range(18, 26))`, `ROLE_ENDPOINT = 0`, `ROLE_CLOSURE = 1`, `ROLE_BRIDGE = 2`, `N_ROLES = 3`, `N_EDGE_TYPES = 3`, `SLOT_ROLES: tuple[int, ...]` (length 26), `EDGE_ENDPOINTS: tuple[tuple[int, int], ...]` (length 96), `EDGE_TYPES: tuple[int, ...]` (length 96, values `TYPE_CLOSURE=0`, `TYPE_ATTACH=1`, `TYPE_INTERIOR=2`), the block slices `CLOSURE_U = slice(0, 8)`, `CLOSURE_V = slice(8, 16)`, `ATTACH_L = slice(16, 24)`, `ATTACH_R = slice(24, 32)`, `INTERIOR = slice(32, 96)`, `SWAP_PERM: tuple[int, ...]` (length 96), and `role_permutation(sigma_c, sigma_l, sigma_r) -> tuple[int, ...]`.

- [ ] **Step 1: Write the failing test**

```python
"""Fixed motif-template geometry, the compiler and the section 7.3 target tensor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from src.data.motif_template import (
    ATTACH_L,
    ATTACH_R,
    CLOSURE_U,
    CLOSURE_V,
    C_SLOTS,
    EDGE_ENDPOINTS,
    EDGE_TYPES,
    INTERIOR,
    L_SLOTS,
    N_EDGES,
    N_SLOTS,
    R_SLOTS,
    SLOT_ROLES,
    SLOT_U,
    SLOT_V,
    SWAP_PERM,
    TYPE_ATTACH,
    TYPE_CLOSURE,
    TYPE_INTERIOR,
    role_permutation,
)


def test_geometry_is_26_slots_and_96_distinct_undirected_edges() -> None:
    assert (N_SLOTS, N_EDGES) == (26, 96)
    assert len(EDGE_ENDPOINTS) == 96
    assert len(EDGE_TYPES) == 96
    assert len(SLOT_ROLES) == 26
    undirected = {frozenset(edge) for edge in EDGE_ENDPOINTS}
    assert len(undirected) == 96
    assert all(len(edge) == 2 for edge in undirected)
    assert frozenset((SLOT_U, SLOT_V)) not in undirected


def test_edge_blocks_carry_the_declared_types_and_endpoints() -> None:
    assert [EDGE_ENDPOINTS[i] for i in range(*CLOSURE_U.indices(96))] == [
        (SLOT_U, c) for c in C_SLOTS
    ]
    assert [EDGE_ENDPOINTS[i] for i in range(*CLOSURE_V.indices(96))] == [
        (c, SLOT_V) for c in C_SLOTS
    ]
    assert [EDGE_ENDPOINTS[i] for i in range(*ATTACH_L.indices(96))] == [
        (SLOT_U, left) for left in L_SLOTS
    ]
    assert [EDGE_ENDPOINTS[i] for i in range(*ATTACH_R.indices(96))] == [
        (right, SLOT_V) for right in R_SLOTS
    ]
    assert [EDGE_ENDPOINTS[32 + 8 * i + j] for i in range(8) for j in range(8)] == [
        (L_SLOTS[i], R_SLOTS[j]) for i in range(8) for j in range(8)
    ]
    assert set(EDGE_TYPES[CLOSURE_U]) == set(EDGE_TYPES[CLOSURE_V]) == {TYPE_CLOSURE}
    assert set(EDGE_TYPES[ATTACH_L]) == set(EDGE_TYPES[ATTACH_R]) == {TYPE_ATTACH}
    assert set(EDGE_TYPES[INTERIOR]) == {TYPE_INTERIOR}


def test_swap_permutation_is_an_involution_that_exchanges_sides() -> None:
    perm = np.asarray(SWAP_PERM)
    assert perm.shape == (96,)
    assert sorted(perm.tolist()) == list(range(96))
    assert perm[perm].tolist() == list(range(96))
    # closure blocks exchange, attachment blocks exchange, interior transposes.
    assert perm[:8].tolist() == list(range(8, 16))
    assert perm[16:24].tolist() == list(range(24, 32))
    assert perm[32 + 8 * 3 + 5] == 32 + 8 * 5 + 3


def test_role_permutation_relabels_blocks_consistently() -> None:
    sigma_c = (1, 0, 2, 3, 4, 5, 6, 7)
    sigma_l = (7, 6, 5, 4, 3, 2, 1, 0)
    sigma_r = tuple(range(8))
    perm = np.asarray(role_permutation(sigma_c, sigma_l, sigma_r))
    assert sorted(perm.tolist()) == list(range(96))
    # A closure slot moves identically in both of its blocks.
    assert perm[0] == 1 and perm[8] == 9
    # An attachment slot moves with its whole interior row.
    assert perm[16] == 23
    assert perm[32 + 8 * 0 + 4] == 32 + 8 * 7 + 4
    # Slot roles are unchanged by a within-role relabelling.
    assert SLOT_ROLES[SLOT_U] == SLOT_ROLES[SLOT_V]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_motif_template.py -n0 -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.motif_template'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Fixed motif-template geometry and the deterministic per-pair compiler (spec §2, §3)."""

from __future__ import annotations

N_SLOTS = 26
N_EDGES = 96

SLOT_U = 0
SLOT_V = 1
C_SLOTS: tuple[int, ...] = tuple(range(2, 10))
L_SLOTS: tuple[int, ...] = tuple(range(10, 18))
R_SLOTS: tuple[int, ...] = tuple(range(18, 26))

ROLE_ENDPOINT = 0
ROLE_CLOSURE = 1
ROLE_BRIDGE = 2
N_ROLES = 3

TYPE_CLOSURE = 0
TYPE_ATTACH = 1
TYPE_INTERIOR = 2
N_EDGE_TYPES = 3

SLOT_ROLES: tuple[int, ...] = (
    (ROLE_ENDPOINT,) * 2 + (ROLE_CLOSURE,) * 8 + (ROLE_BRIDGE,) * 16
)

CLOSURE_U = slice(0, 8)
CLOSURE_V = slice(8, 16)
ATTACH_L = slice(16, 24)
ATTACH_R = slice(24, 32)
INTERIOR = slice(32, 96)

EDGE_ENDPOINTS: tuple[tuple[int, int], ...] = (
    tuple((SLOT_U, c) for c in C_SLOTS)
    + tuple((c, SLOT_V) for c in C_SLOTS)
    + tuple((SLOT_U, left) for left in L_SLOTS)
    + tuple((right, SLOT_V) for right in R_SLOTS)
    + tuple((L_SLOTS[i], R_SLOTS[j]) for i in range(8) for j in range(8))
)

EDGE_TYPES: tuple[int, ...] = (
    (TYPE_CLOSURE,) * 16 + (TYPE_ATTACH,) * 16 + (TYPE_INTERIOR,) * 64
)


def _interior_index(left: int, right: int) -> int:
    """Return the edge index of interior slot pair ``(left, right)``."""
    return 32 + 8 * left + right


def _build_swap_perm() -> tuple[int, ...]:
    """Edge permutation realising ``u<->v`` with ``L<->R`` (spec §4)."""
    perm = [0] * N_EDGES
    for k in range(8):
        perm[k] = 8 + k
        perm[8 + k] = k
        perm[16 + k] = 24 + k
        perm[24 + k] = 16 + k
    for left in range(8):
        for right in range(8):
            perm[_interior_index(left, right)] = _interior_index(right, left)
    return tuple(perm)


SWAP_PERM: tuple[int, ...] = _build_swap_perm()


def role_permutation(
    sigma_c: tuple[int, ...], sigma_l: tuple[int, ...], sigma_r: tuple[int, ...]
) -> tuple[int, ...]:
    """Edge permutation induced by relabelling slots within each role.

    Args:
        sigma_c: Destination closure slot of each closure slot.
        sigma_l: Destination left-bridge slot of each left slot.
        sigma_r: Destination right-bridge slot of each right slot.

    Returns:
        A length-96 permutation ``perm`` with ``perm[i]`` the new index of edge ``i``.
    """
    perm = [0] * N_EDGES
    for k in range(8):
        perm[k] = sigma_c[k]
        perm[8 + k] = 8 + sigma_c[k]
        perm[16 + k] = 16 + sigma_l[k]
        perm[24 + k] = 24 + sigma_r[k]
    for left in range(8):
        for right in range(8):
            perm[_interior_index(left, right)] = _interior_index(
                sigma_l[left], sigma_r[right]
            )
    return tuple(perm)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/data/test_motif_template.py -n0 -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Lint, type-check and commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/data/motif_template.py tests/data/test_motif_template.py
git commit -m "feat(motif): fixed 26-slot, 96-edge motif template geometry"
```

---

### Task 2: `MotifPromptConfig`

**Files:**
- Create: `src/model/egostitch/classifier/motif_prompt.py`
- Test: `tests/model/test_motif_prompt_model.py`

**Interfaces:**
- Consumes: `src.data.motif_template` constants from Task 1.
- Produces: `FIELD_ORDER = ("topo_self", "topo_partner", "topo_rel", "topo_cnt")`, `TEMPLATE_KEY = "motif_weights"`, `TEMPLATE_MASK_KEY = "motif_mask"`, `INTERVENTIONS`, `STAGES = ("one", "two")`, `TOKEN_SOURCES = ("graph", "direct")`, `GATE_MODES = ("learned", "per_type", "mean_graph")`, `FAMILIES = ("closure", "bridge")`, and the frozen dataclasses `ReaderConfig`, `CorruptionConfig`, `MotifPromptConfig` with `from_mapping` / `to_dict`.

- [ ] **Step 1: Write the failing test**

```python
"""Motif-prompt model tests: config, generator, reader, prefix interface, identity."""

from __future__ import annotations

import pytest
from src.model.egostitch.classifier.motif_prompt import (
    FIELD_ORDER,
    GATE_MODES,
    MotifPromptConfig,
    ReaderConfig,
)


def test_config_round_trip_and_defaults() -> None:
    cfg = MotifPromptConfig.from_mapping({"stage": "one", "base_checkpoint": "base.pt"})
    assert cfg.stage == "one"
    assert cfg.width == 128
    assert cfg.slots_per_field == 2
    assert cfg.fields == FIELD_ORDER
    assert cfg.families == ("closure", "bridge")
    assert cfg.token_source == "graph"
    assert cfg.gate_mode == "learned"
    assert cfg.interface_warmup_epochs == 2
    assert (cfg.w_slot, cfg.w_topo) == (1.0, 0.1)
    assert (cfg.beta_p, cfg.beta_q, cfg.beta_a, cfg.beta_i) == (1.0, 1.0, 1.0, 1.0)
    assert cfg.huber_delta == 1.0
    assert cfg.reader == ReaderConfig(layers=3, dim=96, heads=4, rrwp_k=4)
    assert MotifPromptConfig.from_mapping(cfg.to_dict()) == cfg


def test_config_validation_rejects_illegal_blocks() -> None:
    with pytest.raises(ValueError, match="unknown motif_prompt keys"):
        MotifPromptConfig.from_mapping({"stage": "one", "base_checkpoint": "b.pt", "tokens": 4})
    with pytest.raises(ValueError, match="motif_prompt.stage"):
        MotifPromptConfig(stage="three", base_checkpoint="b.pt")
    with pytest.raises(ValueError, match="requires motif_prompt.base_checkpoint"):
        MotifPromptConfig(stage="one")
    with pytest.raises(ValueError, match="requires motif_prompt.bundle_checkpoint"):
        MotifPromptConfig(stage="two", base_checkpoint="b.pt")
    with pytest.raises(ValueError, match="motif_prompt.fields"):
        MotifPromptConfig(stage="one", base_checkpoint="b.pt", fields=("topo_self", "topo_self"))
    with pytest.raises(ValueError, match="motif_prompt.fields"):
        MotifPromptConfig(stage="one", base_checkpoint="b.pt", fields=())
    with pytest.raises(ValueError, match="motif_prompt.families"):
        MotifPromptConfig(stage="one", base_checkpoint="b.pt", families=())
    with pytest.raises(ValueError, match="motif_prompt.gate_mode"):
        MotifPromptConfig(stage="one", base_checkpoint="b.pt", gate_mode="hard_topk")
    with pytest.raises(ValueError, match="reader.dim"):
        MotifPromptConfig(stage="one", base_checkpoint="b.pt", reader=ReaderConfig(dim=97, heads=4))
    with pytest.raises(ValueError, match="reader.rrwp_k"):
        MotifPromptConfig(stage="one", base_checkpoint="b.pt", reader=ReaderConfig(rrwp_k=0))


def test_gate_modes_are_the_three_spec_controls() -> None:
    assert GATE_MODES == ("learned", "per_type", "mean_graph")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/model/test_motif_prompt_model.py -n0 -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.model.egostitch.classifier.motif_prompt'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Motif-graph prompts read by GRIT on a frozen V3.1 trunk.

Design: ``docs/superpowers/specs/2026-09-16-motif-graph-grit-prompt-design.md``.
Stage I trains the reader, the count head, the token projections, the role
embeddings and the prefix adapter on compiled true training templates; Stage II
trains a residue-conditioned generator that predicts the same 96 edge weights
from ``(x_u, x_v)`` alone, read through the identical interface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import cast

FIELD_ORDER = ("topo_self", "topo_partner", "topo_rel", "topo_cnt")
TEMPLATE_KEY = "motif_weights"
TEMPLATE_MASK_KEY = "motif_mask"
INTERVENTIONS = ("none", "gates_off", "mean", "shuffle_graph", "permute_closure", "rewire_bridge")
STAGES = ("one", "two")
TOKEN_SOURCES = ("graph", "direct")
GATE_MODES = ("learned", "per_type", "mean_graph")
FAMILIES = ("closure", "bridge")


@dataclass(frozen=True)
class ReaderConfig:
    """The GRIT reader block (spec §5.2)."""

    layers: int = 3
    dim: int = 96
    heads: int = 4
    rrwp_k: int = 4

    def __post_init__(self) -> None:
        """Validate the reader shape.

        Raises:
            ValueError: On a non-positive size or an indivisible width.
        """
        if self.layers <= 0 or self.dim <= 0 or self.heads <= 0:
            raise ValueError("reader.layers, reader.dim and reader.heads must be positive")
        if self.rrwp_k < 1:
            raise ValueError("reader.rrwp_k must be at least 1")
        if self.dim % self.heads:
            raise ValueError(f"reader.dim ({self.dim}) must be divisible by reader.heads")


@dataclass(frozen=True)
class CorruptionConfig:
    """Stationary Stage I adjacency corruption (spec §3)."""

    prob: float = 0.5
    lambda_min: float = 0.0
    lambda_max: float = 1.0

    def __post_init__(self) -> None:
        """Validate the distribution.

        Raises:
            ValueError: On an out-of-range probability or an empty lambda range.
        """
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError("motif_prompt.corruption.prob must lie in [0, 1]")
        if not 0.0 <= self.lambda_min <= self.lambda_max <= 1.0:
            raise ValueError("motif_prompt.corruption lambda range must lie in [0, 1]")


@dataclass(frozen=True)
class MotifPromptConfig:
    """The ``model.config.motif_prompt`` block."""

    stage: str = "one"
    base_checkpoint: str = ""
    base_checkpoint_sha256: str | None = None
    bundle_checkpoint: str = ""
    bundle_checkpoint_sha256: str | None = None
    width: int = 128
    slots_per_field: int = 2
    reader: ReaderConfig = field(default_factory=ReaderConfig)
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    fields: tuple[str, ...] = FIELD_ORDER
    families: tuple[str, ...] = FAMILIES
    token_source: str = "graph"
    gate_mode: str = "learned"
    interface_warmup_epochs: int | None = 2
    w_slot: float = 1.0
    w_topo: float = 0.1
    beta_p: float = 1.0
    beta_q: float = 1.0
    beta_a: float = 1.0
    beta_i: float = 1.0
    huber_delta: float = 1.0

    def __post_init__(self) -> None:
        """Validate the block.

        Raises:
            ValueError: On an unknown enum value, a missing required checkpoint,
                a duplicated or empty field/family list, or a negative weight.
        """
        if self.stage not in STAGES:
            raise ValueError(f"motif_prompt.stage must be one of {list(STAGES)}")
        if not self.base_checkpoint:
            raise ValueError("motif_prompt requires motif_prompt.base_checkpoint")
        if self.stage == "two" and not self.bundle_checkpoint:
            raise ValueError("stage 'two' requires motif_prompt.bundle_checkpoint")
        if self.width <= 0 or self.slots_per_field <= 0:
            raise ValueError("motif_prompt.width and slots_per_field must be positive")
        if not self.fields or len(set(self.fields)) != len(self.fields):
            raise ValueError("motif_prompt.fields must be a non-empty set of field names")
        if any(name not in FIELD_ORDER for name in self.fields):
            raise ValueError(f"motif_prompt.fields must be drawn from {list(FIELD_ORDER)}")
        if not self.families or any(name not in FAMILIES for name in self.families):
            raise ValueError(f"motif_prompt.families must be a non-empty subset of {list(FAMILIES)}")
        if self.token_source not in TOKEN_SOURCES:
            raise ValueError(f"motif_prompt.token_source must be one of {list(TOKEN_SOURCES)}")
        if self.gate_mode not in GATE_MODES:
            raise ValueError(f"motif_prompt.gate_mode must be one of {list(GATE_MODES)}")
        if self.interface_warmup_epochs is not None and self.interface_warmup_epochs < 0:
            raise ValueError("motif_prompt.interface_warmup_epochs must be non-negative or null")
        for name in ("w_slot", "w_topo", "beta_p", "beta_q", "beta_a", "beta_i"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"motif_prompt.{name} must be non-negative")
        if self.huber_delta <= 0.0:
            raise ValueError("motif_prompt.huber_delta must be positive")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> MotifPromptConfig:
        """Parse the block, rejecting unknown keys.

        Args:
            raw: The mapping under ``model.config.motif_prompt``.

        Returns:
            The parsed config.

        Raises:
            ValueError: On unknown keys.
        """
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown motif_prompt keys: {unknown}")
        values = dict(raw)
        reader = ReaderConfig(**cast(dict[str, int], values.pop("reader", {})))
        corruption = CorruptionConfig(**cast(dict[str, float], values.pop("corruption", {})))
        fields_raw = values.pop("fields", FIELD_ORDER)
        families_raw = values.pop("families", FAMILIES)
        warmup = values.pop("interface_warmup_epochs", 2)
        return cls(
            reader=reader,
            corruption=corruption,
            fields=tuple(str(name) for name in cast(Sequence[object], fields_raw)),
            families=tuple(str(name) for name in cast(Sequence[object], families_raw)),
            interface_warmup_epochs=None if warmup is None else int(cast(int, warmup)),
            **cast(dict[str, object], values),  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        """Return the block as a plain mapping (checkpoint-embeddable)."""
        return cast(dict[str, object], asdict(self))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/model/test_motif_prompt_model.py -n0 -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Lint, type-check and commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/model/egostitch/classifier/motif_prompt.py tests/model/test_motif_prompt_model.py
git commit -m "feat(motif): motif-prompt config block with stage and control switches"
```

---

## Phase 2 — The deterministic template compiler

### Task 3: Queried-edge removal, closure selection and swap equivariance

**Files:**
- Modify: `src/data/motif_template.py`
- Test: `tests/data/test_motif_template.py`

**Interfaces:**
- Consumes: Task 1 geometry; `src.data.artifacts.canonical_pair`; `src.distill.context_sampler._anchor_rng`.
- Produces: `@dataclass(frozen=True) class CompiledTemplate` with fields `weights: NDArray[np.float32]` (shape `(96,)`), `closure: tuple[str, ...]`, `left: tuple[str, ...]`, `right: tuple[str, ...]`; and `class MotifTemplateTable` with `__init__(self, graph: nx.Graph) -> None`, `nodes: tuple[str, ...]`, `index: dict[str, int]`, `compile_row(self, u: str, v: str, *, seed: int = 0, epoch: int = 0, randomise: bool = False) -> CompiledTemplate`.

- [ ] **Step 1: Write the failing test**

Append to `tests/data/test_motif_template.py`:

```python
import networkx as nx
from src.data.motif_template import MotifTemplateTable


def _wedge_graph() -> nx.Graph:
    """u and v share witnesses w1 (degree 2) and w2 (degree 4); u-v is an edge."""
    graph = nx.Graph()
    graph.add_edges_from(
        [
            ("u", "v"),
            ("u", "w1"), ("w1", "v"),
            ("u", "w2"), ("w2", "v"),
            ("w2", "x1"), ("w2", "x2"),
        ]
    )
    return graph


def test_queried_edge_is_removed_before_neighbourhoods_and_weights() -> None:
    table = MotifTemplateTable(_wedge_graph())
    row = table.compile_row("u", "v")
    # v is never a witness of (u, v) even though it is a neighbour of u.
    assert set(row.closure) == {"w1", "w2"}
    # Witness weights are 1/sqrt(d(w)) on the edge-deleted graph: d(w1)=2, d(w2)=4.
    order = {node: i for i, node in enumerate(row.closure)}
    assert row.weights[order["w1"]] == pytest.approx(2.0**-0.5)
    assert row.weights[8 + order["w1"]] == pytest.approx(2.0**-0.5)
    assert row.weights[order["w2"]] == pytest.approx(4.0**-0.5)
    # Witnesses come first by descending 1/d(w), i.e. ascending degree.
    assert row.closure[0] == "w1"


def test_endpoint_degrees_are_decremented_by_the_removed_edge() -> None:
    graph = nx.Graph()
    graph.add_edges_from([("u", "v"), ("u", "a"), ("a", "b"), ("b", "v")])
    table = MotifTemplateTable(graph)
    row = table.compile_row("u", "v")
    assert row.closure == ()
    assert row.left == ("a",) and row.right == ("b",)
    # d(a) = 2 and d(b) = 2 on the edge-deleted graph.
    assert row.weights[16] == pytest.approx(2.0**-0.5)
    assert row.weights[24] == pytest.approx(2.0**-0.5)
    assert row.weights[32] == pytest.approx(1.0)


def test_only_at_most_eight_witnesses_survive_truncation() -> None:
    graph = nx.Graph()
    graph.add_edge("u", "v")
    for i in range(12):
        graph.add_edges_from([("u", f"w{i}"), (f"w{i}", "v")])
        for j in range(i):
            graph.add_edge(f"w{i}", f"pad{i}_{j}")
    table = MotifTemplateTable(graph)
    row = table.compile_row("u", "v")
    assert len(row.closure) == 8
    assert row.closure == tuple(f"w{i}" for i in range(8))
    assert float(row.weights[8:16].sum()) > 0.0


def test_swapping_endpoints_permutes_the_weight_vector_by_swap_perm() -> None:
    table = MotifTemplateTable(_wedge_graph())
    forward = table.compile_row("u", "v")
    reverse = table.compile_row("v", "u")
    np.testing.assert_allclose(reverse.weights, forward.weights[list(SWAP_PERM)], atol=0)
    assert reverse.closure == forward.closure
    assert reverse.left == forward.right and reverse.right == forward.left


def test_self_rows_get_an_explicit_empty_template() -> None:
    table = MotifTemplateTable(_wedge_graph())
    row = table.compile_row("u", "u")
    assert not row.weights.any()
    assert (row.closure, row.left, row.right) == ((), (), ())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_motif_template.py -n0 -q`
Expected: FAIL with `ImportError: cannot import name 'MotifTemplateTable'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/data/motif_template.py`:

```python
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from src.data.artifacts import canonical_pair


@dataclass(frozen=True)
class CompiledTemplate:
    """One pair's compiled motif graph.

    Attributes:
        weights: The 96 edge weights against `EDGE_ENDPOINTS`, in ``[0, 1]``.
        closure: Selected shared witnesses, in slot order.
        left: Selected left bridge intermediates, in slot order.
        right: Selected right bridge intermediates, in slot order.
    """

    weights: NDArray[np.float32]
    closure: tuple[str, ...]
    left: tuple[str, ...]
    right: tuple[str, ...]


def _pair_key(u: str, v: str) -> int:
    """Return a swap-invariant 128-bit tie-break key for the pair ``(u, v)``."""
    a, b = canonical_pair(u, v)
    digest = hashlib.blake2b(f"{a}|{b}".encode(), digest_size=16).digest()
    return int.from_bytes(digest, "little")


def _tie_break(node: str, key: int) -> int:
    """Return a stable, label-independent ordering token for ``node``."""
    digest = hashlib.blake2b(node.encode(), digest_size=16).digest()
    return int.from_bytes(digest, "little") ^ key


class MotifTemplateTable:
    """Compile the fixed motif template of any pair over one loopless graph.

    Build once per universe graph; the adjacency and degrees are shared by every
    row. The queried edge is removed before neighbourhoods, candidates, weights
    and truncation are computed (spec §3).
    """

    def __init__(self, graph: nx.Graph) -> None:
        """Snapshot the loopless adjacency and degrees.

        Args:
            graph: A simple, loopless `networkx.Graph`; isolated nodes are kept.

        Raises:
            ValueError: If the graph is empty or carries a self-loop.
        """
        if graph.number_of_nodes() == 0:
            raise ValueError("motif templates need a non-empty graph")
        if nx.number_of_selfloops(graph) > 0:
            raise ValueError("motif templates need a loopless graph; strip self-loops first")
        self.nodes: tuple[str, ...] = tuple(sorted(str(node) for node in graph.nodes))
        self.index: dict[str, int] = {node: i for i, node in enumerate(self.nodes)}
        self._neighbors: dict[str, frozenset[str]] = {
            node: frozenset(str(other) for other in graph.neighbors(node)) for node in self.nodes
        }
        self._degree: dict[str, int] = {
            node: len(members) for node, members in self._neighbors.items()
        }

    def _removed_degree(self, u: str, v: str) -> dict[str, int]:
        """Return the endpoint degrees after deleting the queried edge."""
        degree = dict(self._degree)
        if u != v and v in self._neighbors[u]:
            degree[u] -= 1
            degree[v] -= 1
        return degree

    def compile_row(
        self, u: str, v: str, *, seed: int = 0, epoch: int = 0, randomise: bool = False
    ) -> CompiledTemplate:
        """Compile one pair's template.

        Args:
            u: First endpoint (slot ``u``).
            v: Second endpoint (slot ``v``).
            seed: Slot-randomisation seed lane.
            epoch: Slot-randomisation epoch lane.
            randomise: Randomise slot order within each role (training only).

        Returns:
            The compiled template.

        Raises:
            KeyError: If an endpoint is not a node of this table's graph.
        """
        weights = np.zeros(N_EDGES, dtype=np.float32)
        if u == v:
            return CompiledTemplate(weights=weights, closure=(), left=(), right=())
        key = _pair_key(u, v)
        degree = self._removed_degree(u, v)
        n_u = self._neighbors[u] - {v}
        n_v = self._neighbors[v] - {u}

        common = sorted(n_u & n_v, key=lambda w: (degree[w], _tie_break(w, key)))[:8]
        for slot, witness in enumerate(common):
            value = np.float32(degree[witness] ** -0.5) if degree[witness] else np.float32(0.0)
            weights[slot] = value
            weights[8 + slot] = value

        left_pool = sorted(n_u - {v}, key=lambda w: _tie_break(w, key))
        right_pool = sorted(n_v - {u}, key=lambda w: _tie_break(w, key))
        left_score: dict[str, float] = dict.fromkeys(left_pool, 0.0)
        right_score: dict[str, float] = dict.fromkeys(right_pool, 0.0)
        right_set = set(right_pool)
        bridges: list[tuple[str, str]] = []
        for i in left_pool:
            for j in self._neighbors[i] & right_set:
                if i == j or not degree[i] or not degree[j]:
                    continue
                contribution = float((degree[i] * degree[j]) ** -0.5)
                left_score[i] += contribution
                right_score[j] += contribution
                bridges.append((i, j))
        left = tuple(sorted(left_pool, key=lambda w: (-left_score[w], _tie_break(w, key)))[:8])
        right = tuple(sorted(right_pool, key=lambda w: (-right_score[w], _tie_break(w, key)))[:8])
        left_slot = {node: slot for slot, node in enumerate(left)}
        right_slot = {node: slot for slot, node in enumerate(right)}
        for node, slot in left_slot.items():
            weights[16 + slot] = np.float32(degree[node] ** -0.5) if degree[node] else np.float32(0.0)
        for node, slot in right_slot.items():
            weights[24 + slot] = np.float32(degree[node] ** -0.5) if degree[node] else np.float32(0.0)
        for i, j in bridges:
            if i in left_slot and j in right_slot:
                weights[_interior_index(left_slot[i], right_slot[j])] = np.float32(1.0)

        closure = tuple(common)
        if randomise:
            weights, closure, left, right = _randomise_roles(
                weights, closure, left, right, key=key, seed=seed, epoch=epoch
            )
        return CompiledTemplate(weights=weights, closure=closure, left=left, right=right)
```

and the role randomiser, which reuses the repository's seeded generator convention:

```python
from src.distill.context_sampler import _anchor_rng


def _randomise_roles(
    weights: NDArray[np.float32],
    closure: tuple[str, ...],
    left: tuple[str, ...],
    right: tuple[str, ...],
    *,
    key: int,
    seed: int,
    epoch: int,
) -> tuple[NDArray[np.float32], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Permute slots within each role; no training identity is an input feature."""
    rng = _anchor_rng(f"motif:{key:032x}", seed=seed, epoch=epoch)
    sigma_c = tuple(int(x) for x in rng.permutation(8))
    sigma_l = tuple(int(x) for x in rng.permutation(8))
    sigma_r = tuple(int(x) for x in rng.permutation(8))
    perm = np.asarray(role_permutation(sigma_c, sigma_l, sigma_r))
    permuted = np.zeros_like(weights)
    permuted[perm] = weights
    def relabel(names: tuple[str, ...], sigma: tuple[int, ...]) -> tuple[str, ...]:
        placed = [""] * 8
        for slot, node in enumerate(names):
            placed[sigma[slot]] = node
        return tuple(name for name in placed if name)
    return permuted, relabel(closure, sigma_c), relabel(left, sigma_l), relabel(right, sigma_r)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/data/test_motif_template.py -n0 -q`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/data/motif_template.py tests/data/test_motif_template.py
git commit -m "feat(motif): deterministic template compiler with queried-edge removal"
```

---

### Task 4: Hub-penalised bridge truncation, slot randomisation and family gating

**Files:**
- Modify: `src/data/motif_template.py`
- Test: `tests/data/test_motif_template.py`

**Interfaces:**
- Consumes: Task 3 `MotifTemplateTable`.
- Produces: `MotifTemplateTable.weights(pairs, *, seed, epoch, randomise) -> NDArray[np.float32]` of shape `(n, 96)`; `MotifTemplateTable.weights_by_index(u_idx, v_idx, *, seed, epoch, randomise) -> NDArray[np.float32]`; `apply_families(weights, families) -> NDArray[np.float32]` zeroing an inactive family; `mean_template(weights) -> NDArray[np.float32]`.

- [ ] **Step 1: Write the failing test**

```python
def test_bridge_truncation_ranks_by_hub_penalised_bridge_mass() -> None:
    graph = nx.Graph()
    graph.add_edge("u", "v")
    # Ten left candidates; l0 bridges to a low-degree right node, l9 to a hub.
    for i in range(10):
        graph.add_edge("u", f"l{i}")
        graph.add_edge(f"l{i}", f"r{i}")
        graph.add_edge(f"r{i}", "v")
        for j in range(i):
            graph.add_edge(f"r{i}", f"hub{i}_{j}")
    table = MotifTemplateTable(graph)
    row = table.compile_row("u", "v")
    assert len(row.left) == 8 and len(row.right) == 8
    assert "l0" in row.left and "r0" in row.right
    assert "r9" not in row.right


def test_an_isolated_selected_intermediate_keeps_its_attachment_and_no_path() -> None:
    graph = nx.Graph()
    graph.add_edges_from([("u", "v"), ("u", "a"), ("v", "b")])
    table = MotifTemplateTable(graph)
    row = table.compile_row("u", "v")
    assert row.left == ("a",) and row.right == ("b",)
    assert row.weights[16] > 0.0 and row.weights[24] > 0.0
    assert float(row.weights[32:96].sum()) == 0.0


def test_within_role_randomisation_permutes_slots_but_preserves_the_multisets() -> None:
    table = MotifTemplateTable(_wedge_graph())
    plain = table.compile_row("u", "v")
    shuffled = table.compile_row("u", "v", seed=7, epoch=3, randomise=True)
    for block in (CLOSURE_U, CLOSURE_V, ATTACH_L, ATTACH_R, INTERIOR):
        np.testing.assert_allclose(
            np.sort(shuffled.weights[block]), np.sort(plain.weights[block]), atol=0
        )
    assert set(shuffled.closure) == set(plain.closure)


def test_weights_batches_rows_in_order_and_family_gating_zeroes_a_family() -> None:
    from src.data.motif_template import apply_families

    table = MotifTemplateTable(_wedge_graph())
    rows = table.weights([("u", "v"), ("v", "u"), ("u", "u")])
    assert rows.shape == (3, 96) and rows.dtype == np.float32
    np.testing.assert_allclose(rows[1], rows[0][list(SWAP_PERM)], atol=0)
    assert not rows[2].any()
    closure_only = apply_families(rows, ("closure",))
    assert float(closure_only[:, 16:].sum()) == 0.0
    assert float(closure_only[:, :16].sum()) == float(rows[:, :16].sum())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_motif_template.py -n0 -q -k "truncation or isolated or randomisation or batches"`
Expected: FAIL with `ImportError: cannot import name 'apply_families'` and `AttributeError: 'MotifTemplateTable' object has no attribute 'weights'`

- [ ] **Step 3: Write minimal implementation**

```python
    def weights_by_index(
        self,
        u_idx: NDArray[np.int64],
        v_idx: NDArray[np.int64],
        *,
        seed: int = 0,
        epoch: int = 0,
        randomise: bool = False,
    ) -> NDArray[np.float32]:
        """Compile every index pair into an ``(n, 96)`` weight table, in row order.

        Args:
            u_idx: ``(n,)`` indices into `nodes`.
            v_idx: ``(n,)`` indices aligned with ``u_idx``.
            seed: Slot-randomisation seed lane.
            epoch: Slot-randomisation epoch lane.
            randomise: Randomise slot order within each role.

        Returns:
            ``(n, 96)`` float32 weights.

        Raises:
            ValueError: On mismatched or out-of-range index arrays.
        """
        u_idx = np.asarray(u_idx, dtype=np.int64)
        v_idx = np.asarray(v_idx, dtype=np.int64)
        if u_idx.shape != v_idx.shape or u_idx.ndim != 1:
            raise ValueError("u_idx and v_idx must be aligned 1-D index arrays")
        size = len(self.nodes)
        if u_idx.size and (
            u_idx.min() < 0 or v_idx.min() < 0 or u_idx.max() >= size or v_idx.max() >= size
        ):
            raise ValueError("pair index outside the motif template table")
        out = np.zeros((u_idx.size, N_EDGES), dtype=np.float32)
        for row, (a, b) in enumerate(zip(u_idx.tolist(), v_idx.tolist(), strict=True)):
            out[row] = self.compile_row(
                self.nodes[a], self.nodes[b], seed=seed, epoch=epoch, randomise=randomise
            ).weights
        return out

    def weights(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        seed: int = 0,
        epoch: int = 0,
        randomise: bool = False,
    ) -> NDArray[np.float32]:
        """Compile node-id pairs into an ``(n, 96)`` weight table, in row order."""
        u_idx = np.fromiter((self.index[u] for u, _ in pairs), dtype=np.int64, count=len(pairs))
        v_idx = np.fromiter((self.index[v] for _, v in pairs), dtype=np.int64, count=len(pairs))
        return self.weights_by_index(
            u_idx, v_idx, seed=seed, epoch=epoch, randomise=randomise
        )


def apply_families(
    weights: NDArray[np.float32], families: Sequence[str]
) -> NDArray[np.float32]:
    """Zero every edge of a family that is not active (spec §8 family ablations).

    Args:
        weights: ``(..., 96)`` weights.
        families: Active families, drawn from ``("closure", "bridge")``.

    Returns:
        A copy with the inactive families' edges set to zero.
    """
    out = np.array(weights, dtype=np.float32, copy=True)
    if "closure" not in families:
        out[..., :16] = 0.0
    if "bridge" not in families:
        out[..., 16:] = 0.0
    return out


def mean_template(weights: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return the training-corpus mean adjacency ``Abar`` (spec §3)."""
    return np.asarray(weights, dtype=np.float32).mean(axis=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/data/test_motif_template.py -n0 -q`
Expected: PASS (13 tests)

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/data/motif_template.py tests/data/test_motif_template.py
git commit -m "feat(motif): batched compilation, family gating and the training-mean template"
```

---

### Task 5: The §7.3 target tensor

**Files:**
- Modify: `src/data/motif_template.py`
- Test: `tests/data/test_motif_template.py`

**Interfaces:**
- Consumes: Task 1 geometry.
- Produces: `slot_profiles(weights: torch.Tensor) -> dict[str, torch.Tensor]` returning `{"p": (B, 8), "q": (B, 64), "a": (B, 16), "b": (B, 64)}`; `count_statistics(weights: torch.Tensor) -> dict[str, torch.Tensor]` returning `{"wedge_mass": (B,), "bridge_mass": (B,), "deg_u": (B,), "deg_v": (B,)}`.

- [ ] **Step 1: Write the failing test**

```python
import torch
from src.data.motif_template import count_statistics, slot_profiles


def test_slot_profiles_are_products_and_raw_weights() -> None:
    weights = torch.zeros(2, 96)
    weights[0, 0] = 0.5   # w(u, c0)
    weights[0, 8] = 0.4   # w(c0, v)
    weights[0, 16] = 0.2  # w(u, l0)
    weights[0, 24] = 0.5  # w(r0, v)
    weights[0, 32] = 0.7  # w(l0, r0)
    parts = slot_profiles(weights)
    assert parts["p"].shape == (2, 8)
    assert parts["q"].shape == (2, 64)
    assert parts["a"].shape == (2, 16)
    assert parts["b"].shape == (2, 64)
    assert parts["p"][0, 0].item() == pytest.approx(0.2)
    assert parts["q"][0, 0].item() == pytest.approx(0.2 * 0.7 * 0.5)
    assert parts["a"][0, 0].item() == pytest.approx(0.2)
    assert parts["a"][0, 8].item() == pytest.approx(0.5)
    assert parts["b"][0, 0].item() == pytest.approx(0.7)
    assert not parts["p"][1].any()


def test_count_statistics_are_the_sums_of_the_supervised_multisets() -> None:
    weights = torch.rand(4, 96)
    parts = slot_profiles(weights)
    stats = count_statistics(weights)
    torch.testing.assert_close(stats["wedge_mass"], parts["p"].sum(dim=1))
    torch.testing.assert_close(stats["bridge_mass"], parts["q"].sum(dim=1))
    torch.testing.assert_close(
        stats["deg_u"], weights[:, 0:8].sum(dim=1) + weights[:, 16:24].sum(dim=1)
    )
    torch.testing.assert_close(
        stats["deg_v"], weights[:, 8:16].sum(dim=1) + weights[:, 24:32].sum(dim=1)
    )


def test_profiles_and_counts_are_invariant_under_a_within_role_permutation() -> None:
    from src.data.motif_template import role_permutation

    weights = torch.rand(3, 96)
    perm = torch.as_tensor(role_permutation((1, 0, 2, 3, 4, 5, 6, 7), (2, 3, 4, 5, 6, 7, 0, 1), tuple(range(8))))
    shuffled = torch.zeros_like(weights)
    shuffled[:, perm] = weights
    for key in ("p", "q", "a", "b"):
        torch.testing.assert_close(
            slot_profiles(shuffled)[key].sort(dim=1, descending=True).values,
            slot_profiles(weights)[key].sort(dim=1, descending=True).values,
        )
    for key, value in count_statistics(shuffled).items():
        torch.testing.assert_close(value, count_statistics(weights)[key])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_motif_template.py -n0 -q -k "profiles or statistics"`
Expected: FAIL with `ImportError: cannot import name 'slot_profiles'`

- [ ] **Step 3: Write minimal implementation**

```python
import torch


def slot_profiles(weights: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return the four §7.3 supervised multisets of a weight table.

    Args:
        weights: ``(B, 96)`` edge weights against `EDGE_ENDPOINTS`.

    Returns:
        ``p`` the 8 wedge products, ``q`` the 64 path products, ``a`` the 16 raw
        attachment weights and ``b`` the 64 raw interior weights, in float32.

    Raises:
        ValueError: If ``weights`` is not ``(B, 96)``.
    """
    if weights.dim() != 2 or weights.size(-1) != N_EDGES:
        raise ValueError(f"motif weights must have shape (B, {N_EDGES}), got {tuple(weights.shape)}")
    value = weights.float()
    closure_u = value[:, CLOSURE_U]
    closure_v = value[:, CLOSURE_V]
    attach_l = value[:, ATTACH_L]
    attach_r = value[:, ATTACH_R]
    interior = value[:, INTERIOR].view(-1, 8, 8)
    products = attach_l.unsqueeze(2) * interior * attach_r.unsqueeze(1)
    return {
        "p": closure_u * closure_v,
        "q": products.reshape(-1, 64),
        "a": torch.cat([attach_l, attach_r], dim=1),
        "b": interior.reshape(-1, 64),
    }


def count_statistics(weights: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return the closed-form count-head statistics of a weight table (spec §5.1)."""
    parts = slot_profiles(weights)
    value = weights.float()
    return {
        "wedge_mass": parts["p"].sum(dim=1),
        "bridge_mass": parts["q"].sum(dim=1),
        "deg_u": value[:, CLOSURE_U].sum(dim=1) + value[:, ATTACH_L].sum(dim=1),
        "deg_v": value[:, CLOSURE_V].sum(dim=1) + value[:, ATTACH_R].sum(dim=1),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/data/test_motif_template.py -n0 -q`
Expected: PASS (16 tests)

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/data/motif_template.py tests/data/test_motif_template.py
git commit -m "feat(motif): the section 7.3 target tensor and count-head statistics"
```

---

### Task 6: Real-artifact boundary check and the measured caching decision

**Files:**
- Modify: `src/data/motif_template.py`
- Test: `tests/data/test_motif_template.py`

**Interfaces:**
- Consumes: Task 4 `MotifTemplateTable`; `src.data.val_region.ValRegionSplit`; `tests/conftest.py::benchmark_root`.
- Produces: `MotifTemplateTable.summary() -> dict[str, object]` with keys `nodes`, `edges`, `compile_seconds_per_10k`.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.slow
def test_no_compiled_target_ever_names_a_v_val_or_boundary_node(benchmark_root: Path) -> None:
    from src.data.artifacts import load_benchmark
    from src.data.val_region import derive_val_region

    artifacts = load_benchmark(benchmark_root)
    split = derive_val_region(artifacts)
    table = MotifTemplateTable(split.build_training_graph())
    assert not (set(table.nodes) & set(split.v_val))
    pairs = sorted(split.training_positives)[:64]
    for u, v in pairs:
        assert u not in split.v_val and v not in split.v_val
        row = table.compile_row(u, v)
        for node in row.closure + row.left + row.right:
            assert node not in split.v_val


def test_summary_records_the_measured_compile_rate() -> None:
    table = MotifTemplateTable(_wedge_graph())
    summary = table.summary()
    assert summary["nodes"] == len(table.nodes)
    assert summary["edges"] == 96
    assert isinstance(summary["compile_seconds_per_10k"], float)
    assert summary["compile_seconds_per_10k"] >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_motif_template.py -n0 -q -k "boundary or summary"`
Expected: FAIL with `AttributeError: 'MotifTemplateTable' object has no attribute 'summary'`

- [ ] **Step 3: Write minimal implementation**

```python
import time


    def summary(self) -> dict[str, object]:
        """Provenance and the measured compile rate that decides caching (spec §3).

        The rate is measured on up to 10,000 pairs drawn round-robin from this
        table's node list, so a caller can record it in ``profile.json`` and pick
        between recomputing per epoch (1.34x the compilations) and a 0.95 GB
        cached table, as spec §3 requires the choice to be made.
        """
        sample = min(10_000, max(1, len(self.nodes) * (len(self.nodes) - 1) // 2))
        u_idx = np.arange(sample, dtype=np.int64) % len(self.nodes)
        v_idx = (u_idx + 1) % len(self.nodes)
        started = time.monotonic()
        self.weights_by_index(u_idx, v_idx)
        elapsed = time.monotonic() - started
        return {
            "nodes": len(self.nodes),
            "edges": N_EDGES,
            "compile_seconds_per_10k": float(elapsed * 10_000.0 / sample),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/data/test_motif_template.py -n0 -q`
Expected: PASS (18 tests; the boundary test skips without `TCIEP_DATA_ROOT`)

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/data/motif_template.py tests/data/test_motif_template.py
git commit -m "feat(motif): compiler provenance, measured compile rate and boundary test"
```

---

## Phase 3 — `motif_prompt.py`

### Task 7: The closed-form count head

**Files:**
- Modify: `src/model/egostitch/classifier/motif_prompt.py`
- Test: `tests/model/test_motif_prompt_model.py`

**Interfaces:**
- Consumes: `src.data.motif_template.count_statistics`.
- Produces: `class MotifCountHead(nn.Module)` with `__init__(self, width: int) -> None` and `forward(self, weights: torch.Tensor) -> torch.Tensor` returning `(B, width)`.

- [ ] **Step 1: Write the failing test**

```python
import torch
from src.data.motif_template import SWAP_PERM
from src.model.egostitch.classifier.motif_prompt import MotifCountHead


def _weights(n: int = 4, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(n, 96, generator=gen)


def test_count_head_is_exactly_swap_invariant() -> None:
    torch.manual_seed(0)
    head = MotifCountHead(width=16)
    weights = _weights()
    swapped = weights[:, list(SWAP_PERM)]
    torch.testing.assert_close(head(weights), head(swapped), rtol=0, atol=1e-6)


def test_count_head_runs_in_fp32_under_autocast_and_stays_finite_at_zero() -> None:
    head = MotifCountHead(width=16)
    zero = torch.zeros(3, 96, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = head(zero)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(zero.grad).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/model/test_motif_prompt_model.py -n0 -q -k count_head`
Expected: FAIL with `ImportError: cannot import name 'MotifCountHead'`

- [ ] **Step 3: Write minimal implementation**

```python
import torch
from torch import nn

from src.data.motif_template import count_statistics


class MotifCountHead(nn.Module):
    """The retained closed-form count pathway (spec §5.1).

    Reads the same predicted adjacency as the reader and emits one token from
    four swap-invariant statistics: ``log1p`` wedge mass, ``log1p`` bridge mass,
    the sum and the absolute difference of the two ``log1p`` degrees.
    """

    def __init__(self, width: int) -> None:
        """Build the single linear map.

        Args:
            width: Token width.
        """
        super().__init__()
        self.proj = nn.Linear(4, width)

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """Return the ``(B, width)`` count token, always in float32.

        Args:
            weights: ``(B, 96)`` predicted edge weights.

        Returns:
            The count token.
        """
        with torch.autocast(device_type=weights.device.type, enabled=False):
            stats = count_statistics(weights.float())
            log_u = torch.log1p(stats["deg_u"])
            log_v = torch.log1p(stats["deg_v"])
            features = torch.stack(
                [
                    torch.log1p(stats["wedge_mass"]),
                    torch.log1p(stats["bridge_mass"]),
                    log_u + log_v,
                    (log_u - log_v).abs(),
                ],
                dim=-1,
            )
            return self.proj(features).float()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/model/test_motif_prompt_model.py -n0 -q -k count_head`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/model/egostitch/classifier/motif_prompt.py tests/model/test_motif_prompt_model.py
git commit -m "feat(motif): swap-invariant closed-form count head"
```

---

### Task 8: The GRIT reader — RRWP under disabled autocast

**Files:**
- Modify: `src/model/egostitch/classifier/motif_prompt.py`
- Test: `tests/model/test_motif_prompt_model.py`

**Interfaces:**
- Consumes: Task 1 geometry; `src.model.egostitch.encoder.grit_gmt.dense_rrwp`.
- Produces: `motif_rrwp(adj: torch.Tensor, k: int) -> torch.Tensor` returning `(B, 26, 26, k)` in float32; `dense_adjacency(weights: torch.Tensor) -> torch.Tensor` returning `(B, 26, 26)`; `typed_adjacency(weights: torch.Tensor) -> torch.Tensor` returning `(B, 26, 26, 3)`.

- [ ] **Step 1: Write the failing test**

```python
from src.model.egostitch.classifier.motif_prompt import (
    dense_adjacency,
    motif_rrwp,
    typed_adjacency,
)


def test_dense_adjacency_is_symmetric_with_a_zero_diagonal_and_no_uv_entry() -> None:
    adj = dense_adjacency(_weights())
    assert adj.shape == (4, 26, 26)
    torch.testing.assert_close(adj, adj.transpose(1, 2))
    assert float(torch.diagonal(adj, dim1=1, dim2=2).abs().sum()) == 0.0
    assert float(adj[:, 0, 1].abs().sum()) == 0.0
    assert float(adj[:, 1, 0].abs().sum()) == 0.0


def test_typed_adjacency_splits_the_three_edge_types_and_sums_to_the_untyped_one() -> None:
    typed = typed_adjacency(_weights())
    assert typed.shape == (4, 26, 26, 3)
    torch.testing.assert_close(typed.sum(dim=-1), dense_adjacency(_weights()))


def test_rrwp_arithmetic_returns_fp32_inside_an_active_autocast_context() -> None:
    adj = dense_adjacency(_weights()).to(torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        stack = motif_rrwp(adj, k=4)
    assert stack.dtype == torch.float32
    assert stack.shape == (4, 26, 26, 4)
    torch.testing.assert_close(
        stack[..., 0], torch.eye(26).expand(4, 26, 26), rtol=0, atol=0
    )


def test_rrwp_is_finite_at_a_zero_adjacency_and_keeps_the_gradient_connection() -> None:
    weights = torch.zeros(2, 96, requires_grad=True)
    stack = motif_rrwp(dense_adjacency(weights), k=4)
    assert torch.isfinite(stack).all()
    stack.sum().backward()
    assert weights.grad is not None and torch.isfinite(weights.grad).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/model/test_motif_prompt_model.py -n0 -q -k "adjacency or rrwp"`
Expected: FAIL with `ImportError: cannot import name 'dense_adjacency'`

- [ ] **Step 3: Write minimal implementation**

```python
from src.data.motif_template import EDGE_ENDPOINTS, EDGE_TYPES, N_EDGE_TYPES, N_SLOTS
from src.model.egostitch.encoder.grit_gmt import dense_rrwp

_EDGE_ROWS = torch.as_tensor([i for i, _ in EDGE_ENDPOINTS], dtype=torch.long)
_EDGE_COLS = torch.as_tensor([j for _, j in EDGE_ENDPOINTS], dtype=torch.long)
_EDGE_TYPE_IDS = torch.as_tensor(EDGE_TYPES, dtype=torch.long)


def dense_adjacency(weights: torch.Tensor) -> torch.Tensor:
    """Scatter the 96 weights into a symmetric ``(B, 26, 26)`` adjacency.

    Every unlisted entry, the diagonal and the queried ``u-v`` entry stay exactly
    zero (spec §2).

    Args:
        weights: ``(B, 96)`` edge weights.

    Returns:
        ``(B, 26, 26)`` symmetric adjacency in ``weights``' dtype.
    """
    batch = weights.size(0)
    rows = _EDGE_ROWS.to(weights.device)
    cols = _EDGE_COLS.to(weights.device)
    adj = weights.new_zeros(batch, N_SLOTS, N_SLOTS)
    adj = adj.index_put((slice(None), rows, cols), weights)
    return adj + adj.transpose(1, 2)


def typed_adjacency(weights: torch.Tensor) -> torch.Tensor:
    """Scatter the weights into a symmetric ``(B, 26, 26, 3)`` typed adjacency."""
    batch = weights.size(0)
    rows = _EDGE_ROWS.to(weights.device)
    cols = _EDGE_COLS.to(weights.device)
    types = _EDGE_TYPE_IDS.to(weights.device)
    typed = weights.new_zeros(batch, N_SLOTS, N_SLOTS, N_EDGE_TYPES)
    typed = typed.index_put((slice(None), rows, cols, types), weights)
    return typed + typed.transpose(1, 2)


def motif_rrwp(adj: torch.Tensor, k: int) -> torch.Tensor:
    """Compute the RRWP stack in fp32 with autocast explicitly disabled.

    Verified on torch 2.10: ``torch.bmm`` and ``@`` return bf16 from *fp32*
    inputs inside an active autocast context, so feeding `dense_rrwp` an fp32
    tensor is not sufficient — the whole call must run with autocast disabled
    (spec §5.2).

    Args:
        adj: ``(B, 26, 26)`` untyped adjacency in any dtype.
        k: Number of stacked walk orders, including the identity term.

    Returns:
        ``(B, 26, 26, k)`` float32 RRWP stack.
    """
    with torch.autocast(device_type=adj.device.type, enabled=False):
        return dense_rrwp(adj.float(), k).float()
```

Note for the implementer: `dense_rrwp` already clamps the degree at `_DEG_EPS = 1e-6` and keeps the autograd chain alive for a learned generator's exactly-zero adjacency (`grit_gmt.py:95-113`). The `mask` argument is deliberately not passed: all 26 slots exist in both stages (spec §2).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/model/test_motif_prompt_model.py -n0 -q -k "adjacency or rrwp"`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/model/egostitch/classifier/motif_prompt.py tests/model/test_motif_prompt_model.py
git commit -m "feat(motif): dense typed adjacency and fp32 RRWP under disabled autocast"
```

---

### Task 9: `MotifGritReader` — node and symmetric pair states

**Files:**
- Modify: `src/model/egostitch/classifier/motif_prompt.py`
- Test: `tests/model/test_motif_prompt_model.py`

**Interfaces:**
- Consumes: Task 8 helpers; `src.vendor.grit_official.GritTransformerLayer`; `src.model.egostitch.encoder.grit_gmt._GritBatch`, `_grit_layer_cfg`.
- Produces: `class MotifGritReader(nn.Module)` with `__init__(self, cfg: ReaderConfig, width: int) -> None` and `forward(self, weights: torch.Tensor) -> dict[str, torch.Tensor]` returning `{"topo_u": (B, width), "topo_v": (B, width), "topo_rel": (B, width)}`.

- [ ] **Step 1: Write the failing test**

```python
from src.data.motif_template import role_permutation
from src.model.egostitch.classifier.motif_prompt import MotifGritReader, ReaderConfig


def _reader(seed: int = 0) -> MotifGritReader:
    torch.manual_seed(seed)
    return MotifGritReader(ReaderConfig(layers=2, dim=16, heads=4, rrwp_k=4), width=16).eval()


def test_reader_emits_three_tokens_and_a_swap_exchanges_only_the_endpoints() -> None:
    reader = _reader()
    weights = _weights()
    out = reader(weights)
    assert set(out) == {"topo_u", "topo_v", "topo_rel"}
    assert out["topo_u"].shape == (4, 16)
    swapped = reader(weights[:, list(SWAP_PERM)])
    torch.testing.assert_close(swapped["topo_u"], out["topo_v"], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(swapped["topo_v"], out["topo_u"], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(swapped["topo_rel"], out["topo_rel"], rtol=1e-5, atol=1e-5)


def test_reader_is_invariant_to_a_within_role_slot_permutation() -> None:
    reader = _reader()
    weights = _weights()
    perm = torch.as_tensor(
        role_permutation((3, 1, 0, 2, 5, 4, 7, 6), (1, 2, 3, 4, 5, 6, 7, 0), (7, 0, 1, 2, 3, 4, 5, 6))
    )
    shuffled = torch.zeros_like(weights)
    shuffled[:, perm] = weights
    out, permuted = reader(weights), reader(shuffled)
    for key in ("topo_u", "topo_v", "topo_rel"):
        torch.testing.assert_close(permuted[key], out[key], rtol=1e-4, atol=1e-4)


def test_reader_predictions_are_batch_independent() -> None:
    reader = _reader()
    weights = _weights(n=6, seed=3)
    whole = reader(weights)
    halves = {
        key: torch.cat([reader(weights[:2])[key], reader(weights[2:])[key]])
        for key in whole
    }
    for key, value in whole.items():
        torch.testing.assert_close(halves[key], value, rtol=1e-5, atol=1e-5)


def test_final_layer_edge_parameters_receive_gradient() -> None:
    torch.manual_seed(1)
    reader = MotifGritReader(ReaderConfig(layers=2, dim=16, heads=4, rrwp_k=4), width=16)
    out = reader(_weights())
    (out["topo_rel"].sum() + out["topo_u"].sum()).backward()
    last = reader.layers[-1]
    assert last.O_e.weight.grad is not None
    assert float(last.O_e.weight.grad.abs().sum()) > 0.0
    assert reader.pair_proj.weight.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/model/test_motif_prompt_model.py -n0 -q -k reader`
Expected: FAIL with `ImportError: cannot import name 'MotifGritReader'`

- [ ] **Step 3: Write minimal implementation**

```python
from src.data.motif_template import ROLE_ENDPOINT, SLOT_ROLES, SLOT_U, SLOT_V
from src.model.egostitch.encoder.grit_gmt import _GritBatch, _grit_layer_cfg
from src.vendor.grit_official import GritTransformerLayer

_ROLE_IDS = torch.as_tensor(SLOT_ROLES, dtype=torch.long)
_ROLE_WIDTH = 16


class MotifGritReader(nn.Module):
    """Vendored GRIT over the 26-slot motif graph, exposing node and pair states.

    The node channel sees role embeddings and the RRWP diagonal only; the edge
    channel sees the RRWP stack and the typed raw weights; the degree input is
    the weighted degree. Neither residue states nor generator hidden states nor
    slot-index embeddings reach it (spec §2, §5.2).

    Every layer is constructed with ``O_e=True`` and ``norm_e=True`` so the final
    layer's edge state is projected and normalised before it is read; the
    vendored layer already writes it to ``batch.edge_attr`` because
    ``cfg["update_e"]`` is set. `src/vendor/` is not edited.
    """

    def __init__(self, cfg: ReaderConfig, width: int) -> None:
        """Build the RRWP embeddings, the GRIT stack and the token projections.

        Args:
            cfg: The reader block.
            width: Output token width.
        """
        super().__init__()
        self.cfg = cfg
        self.role_embed = nn.Embedding(N_ROLES, _ROLE_WIDTH)
        self.node_embed = nn.Linear(_ROLE_WIDTH + cfg.rrwp_k, cfg.dim)
        self.edge_embed = nn.Linear(cfg.rrwp_k + N_EDGE_TYPES, cfg.dim)
        layer_cfg = _grit_layer_cfg()
        self.layers = nn.ModuleList(
            GritTransformerLayer(  # type: ignore[no-untyped-call]
                in_dim=cfg.dim,
                out_dim=cfg.dim,
                num_heads=cfg.heads,
                dropout=0.0,
                attn_dropout=0.0,
                layer_norm=True,
                batch_norm=False,
                residual=True,
                act="relu",
                norm_e=True,
                O_e=True,
                cfg=layer_cfg,
            )
            for _ in range(cfg.layers)
        )
        self.node_proj = nn.Linear(cfg.dim, width)
        self.pair_proj = nn.Linear(cfg.dim, width)
        self.token_norm = nn.LayerNorm(width)

    def forward(self, weights: torch.Tensor) -> dict[str, torch.Tensor]:
        """Read one batch of motif graphs.

        Args:
            weights: ``(B, 96)`` edge weights.

        Returns:
            ``topo_u``, ``topo_v`` and the swap-invariant ``topo_rel``.
        """
        batch = weights.size(0)
        device = weights.device
        adj = dense_adjacency(weights)
        stack = motif_rrwp(adj, self.cfg.rrwp_k)
        diagonal = torch.diagonal(stack, dim1=1, dim2=2).transpose(1, 2)
        roles = self.role_embed(_ROLE_IDS.to(device)).unsqueeze(0).expand(batch, -1, -1)
        node_input = torch.cat((roles, diagonal.to(roles.dtype)), dim=-1)
        pair_input = torch.cat((stack, typed_adjacency(weights)), dim=-1)

        grid = torch.arange(N_SLOTS, device=device)
        src = grid.repeat_interleave(N_SLOTS)
        dst = grid.repeat(N_SLOTS)
        offsets = (torch.arange(batch, device=device) * N_SLOTS).repeat_interleave(N_SLOTS**2)
        edge_index = torch.stack((src.repeat(batch) + offsets, dst.repeat(batch) + offsets))
        edge_attr = self.edge_embed(pair_input.reshape(batch * N_SLOTS**2, -1).to(node_input.dtype))
        flat_x = self.node_embed(node_input.reshape(batch * N_SLOTS, -1))
        weighted_degree = adj.sum(dim=-1).reshape(batch * N_SLOTS, 1)
        log_deg = torch.log(weighted_degree.float() + 1.0).to(flat_x.dtype)

        flat = _GritBatch(flat_x, edge_index, edge_attr, log_deg)
        for layer in self.layers:
            flat = layer(flat)

        nodes = flat.x.view(batch, N_SLOTS, -1)
        edges = flat.edge_attr.view(batch, N_SLOTS, N_SLOTS, -1)
        topo_u = self.token_norm(self.node_proj(nodes[:, SLOT_U]))
        topo_v = self.token_norm(self.node_proj(nodes[:, SLOT_V]))
        relation = 0.5 * (edges[:, SLOT_U, SLOT_V] + edges[:, SLOT_V, SLOT_U])
        return {
            "topo_u": topo_u,
            "topo_v": topo_v,
            "topo_rel": self.token_norm(self.pair_proj(relation)),
        }
```

Note for the implementer: the shared `node_proj`/`pair_proj` and the role embedding are what make the swap test pass — `topo_u` and `topo_v` use the *same* map, and `topo_rel` symmetrises the two directed edge states. The per-token `LayerNorm` is what keeps predictions batch-independent.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/model/test_motif_prompt_model.py -n0 -q -k reader`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/model/egostitch/classifier/motif_prompt.py tests/model/test_motif_prompt_model.py
git commit -m "feat(motif): GRIT adapter exposing node and symmetric pair states"
```

---

### Task 10: `MotifGenerator` — residue-conditioned edge gates

**Files:**
- Modify: `src/model/egostitch/classifier/motif_prompt.py`
- Test: `tests/model/test_motif_prompt_model.py`

**Interfaces:**
- Consumes: `src.model.egostitch.classifier.layers.masked_mean`, `inner_token_mask`, `_build_padding_mask`.
- Produces: `class MotifGenerator(nn.Module)` with `__init__(self, d_model: int, cfg: MotifPromptConfig) -> None`, `forward(self, encoded_u, encoded_v, lengths_u, lengths_v) -> torch.Tensor` returning `(B, 96)` weights in `[0,1]`, and `init_biases(self, mean_weights: torch.Tensor) -> None`.

- [ ] **Step 1: Write the failing test**

```python
from src.model.egostitch.classifier.motif_prompt import MotifGenerator, MotifPromptConfig


def _generator(seed: int = 0, **extra: object) -> MotifGenerator:
    torch.manual_seed(seed)
    cfg = MotifPromptConfig.from_mapping(
        {"stage": "two", "base_checkpoint": "b.pt", "bundle_checkpoint": "s.pt", **extra}
    )
    return MotifGenerator(d_model=32, cfg=cfg).eval()


def _states(n: int = 4, length: int = 7, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, length, 32, generator=gen), torch.full((n,), length, dtype=torch.long)


def test_generator_emits_96_weights_in_the_unit_interval() -> None:
    generator = _generator()
    h_u, len_u = _states(seed=1)
    h_v, len_v = _states(seed=2)
    weights = generator(h_u, h_v, len_u, len_v)
    assert weights.shape == (4, 96)
    assert float(weights.min()) >= 0.0 and float(weights.max()) <= 1.0


def test_generator_is_equivariant_under_swapping_the_endpoints() -> None:
    generator = _generator()
    h_u, len_u = _states(seed=1)
    h_v, len_v = _states(seed=2)
    forward = generator(h_u, h_v, len_u, len_v)
    reverse = generator(h_v, h_u, len_v, len_u)
    torch.testing.assert_close(reverse, forward[:, list(SWAP_PERM)], rtol=1e-5, atol=1e-5)


def test_output_biases_start_at_the_clipped_training_mean_in_logit_space() -> None:
    generator = _generator()
    mean = torch.full((96,), 0.001)
    mean[16:32] = 0.4
    generator.init_biases(mean)
    h_u, len_u = _states(seed=1)
    h_v, len_v = _states(seed=2)
    weights = generator(h_u, h_v, len_u, len_v)
    # Clipped to [0.01, 0.99] before the logit, so nothing saturates.
    assert float(weights[:, :16].mean()) < 0.2
    assert 0.2 < float(weights[:, 16:32].mean()) < 0.8


def test_per_type_and_mean_graph_gate_modes_realise_their_controls() -> None:
    per_type = _generator(gate_mode="per_type")
    h_u, len_u = _states(seed=1)
    h_v, len_v = _states(seed=2)
    weights = per_type(h_u, h_v, len_u, len_v)
    for block in (slice(0, 16), slice(16, 32), slice(32, 96)):
        assert float(weights[:, block].std(dim=1).abs().max()) == pytest.approx(0.0, abs=1e-6)
    mean_graph = _generator(gate_mode="mean_graph")
    mean_graph.init_biases(torch.full((96,), 0.3))
    fixed = mean_graph(h_u, h_v, len_u, len_v)
    assert float(fixed.std(dim=0).abs().max()) == pytest.approx(0.0, abs=1e-6)


def test_gradients_reach_every_generator_parameter_on_a_non_degenerate_batch() -> None:
    torch.manual_seed(2)
    cfg = MotifPromptConfig.from_mapping(
        {"stage": "two", "base_checkpoint": "b.pt", "bundle_checkpoint": "s.pt"}
    )
    generator = MotifGenerator(d_model=32, cfg=cfg)
    h_u, len_u = _states(seed=1)
    h_v, len_v = _states(seed=2)
    generator(h_u, h_v, len_u, len_v).sum().backward()
    missing = [name for name, p in generator.named_parameters() if p.grad is None]
    assert missing == []
    assert all(torch.isfinite(p.grad).all() for _, p in generator.named_parameters())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/model/test_motif_prompt_model.py -n0 -q -k generator`
Expected: FAIL with `ImportError: cannot import name 'MotifGenerator'`

- [ ] **Step 3: Write minimal implementation**

```python
from torch.nn import functional as F

from src.data.motif_template import (
    ATTACH_L,
    ATTACH_R,
    CLOSURE_U,
    CLOSURE_V,
    C_SLOTS,
    EDGE_TYPES,
    INTERIOR,
    L_SLOTS,
    N_EDGES,
    R_SLOTS,
    TYPE_ATTACH,
    TYPE_CLOSURE,
    TYPE_INTERIOR,
)
from src.model.egostitch.classifier.layers import _build_padding_mask, inner_token_mask, masked_mean

_GATE_DIM = 96
_BIAS_CLIP = (0.01, 0.99)


class _MessageLayer(nn.Module):
    """One shared residual message-passing layer over the fixed candidate template.

    Three relation transforms, one per *edge type* — closure, attachment and
    interior. Both sides of a family share a transform, which is exactly what
    makes the whole generator equivariant under ``u<->v, L<->R`` (spec §4).
    """

    def __init__(self, dim: int) -> None:
        """Build the relation transforms and the update MLP."""
        super().__init__()
        self.relations = nn.ModuleList(nn.Linear(dim, dim) for _ in range(N_EDGE_TYPES))
        self.update = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, h: torch.Tensor, incidence: torch.Tensor) -> torch.Tensor:
        """Aggregate typed mean messages and apply the residual update.

        Args:
            h: ``(B, 26, dim)`` slot states.
            incidence: ``(3, 26, 26)`` row-normalised candidate incidence per type.

        Returns:
            Updated ``(B, 26, dim)`` states.
        """
        messages = torch.zeros_like(h)
        for index, relation in enumerate(self.relations):
            messages = messages + incidence[index].to(h.dtype) @ relation(h)
        return h + self.update(messages)


def _candidate_incidence() -> torch.Tensor:
    """Row-normalised ``(3, 26, 26)`` incidence of the fixed candidate template."""
    incidence = torch.zeros(N_EDGE_TYPES, N_SLOTS, N_SLOTS)
    for edge, (i, j) in enumerate(EDGE_ENDPOINTS):
        incidence[EDGE_TYPES[edge], i, j] = 1.0
        incidence[EDGE_TYPES[edge], j, i] = 1.0
    return incidence / incidence.sum(dim=-1, keepdim=True).clamp_min(1.0)


class MotifGenerator(nn.Module):
    """Residue-conditioned edge gates over the fixed template (spec §4).

    Shared slot queries read ``H_u`` and ``H_v`` separately under residue-length
    masks: eight bridge queries used on both sides and eight witness queries read
    on both proteins. Gate-head inputs are LayerNormed and the heads are MLPs
    rather than free edge logits, because the predecessor arm's gate died when
    its input norm grew.
    """

    incidence: torch.Tensor

    def __init__(self, d_model: int, cfg: MotifPromptConfig) -> None:
        """Build the attention, the gate MPNN and the three typed heads.

        Args:
            d_model: Frozen trunk width the residue states arrive in.
            cfg: The motif-prompt block.
        """
        super().__init__()
        self.cfg = cfg
        self.residue_proj = nn.Linear(d_model, _GATE_DIM)
        self.attention = nn.MultiheadAttention(_GATE_DIM, 4, batch_first=True)
        self.bridge_queries = nn.Parameter(torch.randn(8, _GATE_DIM) * 0.02)
        self.witness_queries = nn.Parameter(torch.randn(8, _GATE_DIM) * 0.02)
        self.endpoint_proj = nn.Linear(_GATE_DIM, _GATE_DIM)
        self.witness_mix = nn.Sequential(
            nn.Linear(2 * _GATE_DIM, _GATE_DIM), nn.GELU(), nn.Linear(_GATE_DIM, _GATE_DIM)
        )
        self.message_layers = nn.ModuleList(_MessageLayer(_GATE_DIM) for _ in range(2))
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(2 * _GATE_DIM),
                nn.Linear(2 * _GATE_DIM, _GATE_DIM),
                nn.GELU(),
                nn.Linear(_GATE_DIM, 1),
            )
            for _ in range(N_EDGE_TYPES)
        )
        for head in self.heads:
            nn.init.normal_(cast(nn.Linear, head[-1]).weight, std=1e-3)
            nn.init.zeros_(cast(nn.Linear, head[-1]).bias)
        self.fixed_logits = nn.Parameter(torch.zeros(N_EDGES), requires_grad=False)
        self.register_buffer("incidence", _candidate_incidence())

    @torch.no_grad()
    def init_biases(self, mean_weights: torch.Tensor) -> None:
        """Set each head's output bias from the training mean weight of its type.

        Args:
            mean_weights: ``(96,)`` training-corpus mean edge weights.
        """
        low, high = _BIAS_CLIP
        clipped = mean_weights.float().clamp(low, high)
        self.fixed_logits.copy_(torch.logit(clipped))
        for edge_type, head in enumerate(self.heads):
            mask = torch.as_tensor([t == edge_type for t in EDGE_TYPES])
            cast(nn.Linear, head[-1]).bias.fill_(float(torch.logit(clipped[mask].mean())))

    def _read(self, queries: torch.Tensor, states: torch.Tensor, pad: torch.Tensor | None) -> torch.Tensor:
        expanded = queries.unsqueeze(0).expand(states.size(0), -1, -1)
        out, _ = self.attention(expanded, states, states, key_padding_mask=pad, need_weights=False)
        return cast(torch.Tensor, out)

    def forward(
        self,
        encoded_u: torch.Tensor,
        encoded_v: torch.Tensor,
        lengths_u: torch.Tensor,
        lengths_v: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the 96 edge weights of one batch of pairs.

        Args:
            encoded_u: Frozen residue states of ``u`` ``(B, L_u, d_model)``.
            encoded_v: Frozen residue states of ``v`` ``(B, L_v, d_model)``.
            lengths_u: True residue lengths of ``u``.
            lengths_v: True residue lengths of ``v``.

        Returns:
            ``(B, 96)`` weights in ``[0, 1]``.
        """
        batch = encoded_u.size(0)
        if self.cfg.gate_mode == "mean_graph":
            return torch.sigmoid(self.fixed_logits).unsqueeze(0).expand(batch, -1)
        pad_u = _build_padding_mask(lengths_u, encoded_u.size(1))
        pad_v = _build_padding_mask(lengths_v, encoded_v.size(1))
        state_u = self.residue_proj(encoded_u)
        state_v = self.residue_proj(encoded_v)
        left = self._read(self.bridge_queries, state_u, pad_u)
        right = self._read(self.bridge_queries, state_v, pad_v)
        witness_u = self._read(self.witness_queries, state_u, pad_u)
        witness_v = self._read(self.witness_queries, state_v, pad_v)
        closure = self.witness_mix(
            torch.cat([witness_u + witness_v, (witness_u - witness_v).abs()], dim=-1)
        )
        pooled_u = self.endpoint_proj(masked_mean(state_u, inner_token_mask(x=state_u, padding_mask=pad_u)))
        pooled_v = self.endpoint_proj(masked_mean(state_v, inner_token_mask(x=state_v, padding_mask=pad_v)))
        h = torch.zeros(batch, N_SLOTS, _GATE_DIM, device=state_u.device, dtype=state_u.dtype)
        h[:, SLOT_U] = pooled_u
        h[:, SLOT_V] = pooled_v
        h[:, list(C_SLOTS)] = closure
        h[:, list(L_SLOTS)] = left
        h[:, list(R_SLOTS)] = right
        for layer in self.message_layers:
            h = layer(h, self.incidence)

        rows = _EDGE_ROWS.to(h.device)
        cols = _EDGE_COLS.to(h.device)
        h_i, h_j = h[:, rows], h[:, cols]
        features = torch.cat([h_i + h_j, (h_i - h_j).abs()], dim=-1)
        logits = torch.zeros(batch, N_EDGES, device=h.device, dtype=features.dtype)
        for edge_type, head in enumerate(self.heads):
            mask = torch.as_tensor([t == edge_type for t in EDGE_TYPES], device=h.device)
            if self.cfg.gate_mode == "per_type":
                pooled = features[:, mask].mean(dim=1, keepdim=True)
                logits[:, mask] = head(pooled).expand(-1, int(mask.sum()))
            else:
                logits[:, mask] = head(features[:, mask]).squeeze(-1)
        return torch.sigmoid(logits)
```

Note for the implementer: `nn.MultiheadAttention` is a single shared module read on each side, and `witness_mix` takes the symmetric pair `[a_u + a_v, |a_u - a_v|]`, so swapping the endpoints exchanges `left`/`right` with the identity index map and leaves the closure states fixed. Together with the three shared typed heads this is exactly the `SWAP_PERM` equivariance the second test asserts.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/model/test_motif_prompt_model.py -n0 -q -k generator`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/model/egostitch/classifier/motif_prompt.py tests/model/test_motif_prompt_model.py
git commit -m "feat(motif): residue-conditioned equivariant edge-gate generator"
```

---

### Task 11: The four-token prefix interface with per-field row masking

**Files:**
- Modify: `src/model/egostitch/classifier/motif_prompt.py`
- Test: `tests/model/test_motif_prompt_model.py`

**Interfaces:**
- Consumes: `src.model.egostitch.classifier.prefix.SITES`, `prefix_branch`; `src.model.egostitch.classifier.layers.CrossAttentionLayer`.
- Produces: `class MotifPromptAdapter(nn.Module)` with `__init__(self, d_model, n_layers, n_heads, cfg) -> None`, `views(self, tokens: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]`, `prefix(self, layer_index: int, view: torch.Tensor) -> torch.Tensor`, `gate(self, layer_index: int, site: int) -> torch.Tensor`, `active_rows: tuple[int, ...]`; and `class MotifPromptCrossAttentionLayer(nn.Module)` mirroring `TopoPromptCrossAttentionLayer`.

- [ ] **Step 1: Write the failing test**

```python
from src.model.egostitch.classifier.motif_prompt import MotifPromptAdapter


def _adapter(fields: tuple[str, ...] = FIELD_ORDER, seed: int = 0) -> MotifPromptAdapter:
    torch.manual_seed(seed)
    cfg = MotifPromptConfig.from_mapping(
        {"stage": "one", "base_checkpoint": "b.pt", "width": 16, "fields": list(fields)}
    )
    return MotifPromptAdapter(d_model=32, n_layers=3, n_heads=4, cfg=cfg)


def _tokens(n: int = 4, width: int = 16, seed: int = 5) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {name: torch.randn(n, width, generator=gen) for name in ("topo_u", "topo_v", "topo_rel", "topo_cnt")}


def test_gates_start_at_zero_and_have_one_entry_per_layer_site_head() -> None:
    adapter = _adapter()
    assert adapter.gates.shape == (3, 3, 4)
    assert float(adapter.gates.abs().sum()) == 0.0


def test_the_two_views_swap_only_the_endpoint_fields() -> None:
    adapter = _adapter()
    tokens = _tokens()
    view_u, view_v = adapter.views(tokens)
    assert view_u.shape == (4, 4, 16)
    torch.testing.assert_close(view_u[:, 0], view_v[:, 1])
    torch.testing.assert_close(view_u[:, 1], view_v[:, 0])
    torch.testing.assert_close(view_u[:, 2], view_v[:, 2])
    torch.testing.assert_close(view_u[:, 3], view_v[:, 3])


def test_per_field_masking_removes_rows_rather_than_zeroing_values() -> None:
    full = _adapter()
    tokens = _tokens()
    rows = full.prefix(0, full.views(tokens)[0])
    assert rows.shape == (4, 8, 32)
    trimmed = _adapter(fields=("topo_self", "topo_partner", "topo_rel"))
    trimmed.load_state_dict(full.state_dict(), strict=False)
    kept = trimmed.prefix(0, trimmed.views(tokens)[0])
    assert kept.shape == (4, 6, 32)
    torch.testing.assert_close(kept, rows[:, list(trimmed.active_rows)], rtol=1e-6, atol=1e-6)


def test_masking_a_field_renormalises_the_branch_over_the_remaining_rows() -> None:
    from src.model.egostitch.classifier.prefix import prefix_branch

    torch.manual_seed(3)
    mha = torch.nn.MultiheadAttention(32, 4, batch_first=True)
    query = torch.randn(2, 5, 32)
    prefix = torch.randn(2, 8, 32)
    gate = torch.randn(4)
    zeroed = prefix.clone()
    zeroed[:, 6:] = 0.0
    removed = prefix[:, :6]
    branch_zeroed = prefix_branch(mha, query, zeroed, gate, 1.0)
    branch_removed = prefix_branch(mha, query, removed, gate, 1.0)
    # Zeroing the values leaves the keys in the denominator; removing the rows does not.
    assert not torch.allclose(branch_zeroed, branch_removed, rtol=1e-4, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/model/test_motif_prompt_model.py -n0 -q -k "adapter or views or masking or gates_start"`
Expected: FAIL with `ImportError: cannot import name 'MotifPromptAdapter'`

- [ ] **Step 3: Write minimal implementation**

```python
from src.model.egostitch.classifier.layers import CrossAttentionLayer
from src.model.egostitch.classifier.prefix import SITES, prefix_branch

_FIELD_TO_TOKEN = {
    "topo_self": ("topo_u", "topo_v"),
    "topo_partner": ("topo_v", "topo_u"),
    "topo_rel": ("topo_rel", "topo_rel"),
    "topo_cnt": ("topo_cnt", "topo_cnt"),
}


class MotifPromptAdapter(nn.Module):
    """The four-token gated KV prefix at all nine cross-attention sites (spec §6).

    Fields are laid out in the canonical `FIELD_ORDER`; ``cfg.fields`` selects
    which of them survive, and an unselected field's rows are *removed* from the
    prefix so the separately-softmaxed branch renormalises over the remainder.
    Zeroing a field's values would leave its keys in the denominator, which is
    not the same model (spec §6) — the count-only and GRIT-only arms of §8 are
    therefore separately trained, not inference ablations.
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int, cfg: MotifPromptConfig) -> None:
        """Build the role embeddings, per-layer expansions, offsets and gates.

        Args:
            d_model: Trunk width.
            n_layers: Number of cross-attention layers.
            n_heads: Heads per attention site.
            cfg: The motif-prompt block.
        """
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.n_layers = n_layers
        self.slots = len(FIELD_ORDER) * cfg.slots_per_field
        self.active_rows: tuple[int, ...] = tuple(
            field_index * cfg.slots_per_field + offset
            for field_index, name in enumerate(FIELD_ORDER)
            if name in cfg.fields
            for offset in range(cfg.slots_per_field)
        )
        self.role_embed = nn.Parameter(torch.randn(len(FIELD_ORDER), cfg.width) * 0.02)
        self.token_norm = nn.LayerNorm(cfg.width)
        self.layer_proj = nn.ModuleList(
            nn.Linear(cfg.width, cfg.slots_per_field * d_model) for _ in range(n_layers)
        )
        self.p0 = nn.ParameterList(
            nn.Parameter(torch.randn(self.slots, d_model) * 0.02) for _ in range(n_layers)
        )
        self.gates = nn.Parameter(torch.zeros(n_layers, SITES, n_heads))

    def views(self, tokens: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``(B, 4, width)`` field stacks of the two stream views.

        Args:
            tokens: ``topo_u``, ``topo_v``, ``topo_rel`` and ``topo_cnt``.

        Returns:
            ``(view_u, view_v)``; ``view_u`` names ``u`` as *self*.
        """
        stacks: list[list[torch.Tensor]] = [[], []]
        for index, name in enumerate(FIELD_ORDER):
            first, second = _FIELD_TO_TOKEN[name]
            stacks[0].append(tokens[first] + self.role_embed[index])
            stacks[1].append(tokens[second] + self.role_embed[index])
        return (
            self.token_norm(torch.stack(stacks[0], dim=1)),
            self.token_norm(torch.stack(stacks[1], dim=1)),
        )

    def prefix(self, layer_index: int, view: torch.Tensor) -> torch.Tensor:
        """Expand a field stack to this layer's active prefix rows.

        Args:
            layer_index: Which frozen layer.
            view: ``(B, 4, width)`` field stack.

        Returns:
            ``(B, len(active_rows), d_model)`` prefix rows.
        """
        proj = cast(nn.Linear, self.layer_proj[layer_index])
        rows = proj(view).view(view.size(0), self.slots, self.d_model)
        p0: torch.Tensor = self.p0[layer_index]
        return (rows + p0.unsqueeze(0))[:, list(self.active_rows)]

    def gate(self, layer_index: int, site: int) -> torch.Tensor:
        """Return the raw (pre-tanh) per-head gate for one site."""
        return self.gates[layer_index, site]


class MotifPromptCrossAttentionLayer(nn.Module):
    """Re-drive one frozen `CrossAttentionLayer` with the role-aware motif prefix.

    Site ``A<-B`` and the CLS site read the view whose *self* endpoint is the A
    stream's node; site ``B<-A`` reads the other view. The wrapped layer and the
    adapter are plain references, exactly as the prefix and topology-prompt arms
    do, so their state-dict keys are not duplicated.
    """

    layer: CrossAttentionLayer
    adapter: MotifPromptAdapter

    def __init__(self, layer: CrossAttentionLayer, layer_index: int, adapter: MotifPromptAdapter) -> None:
        """Wrap a frozen layer.

        Args:
            layer: The trunk's `CrossAttentionLayer` (registered under ``base``).
            layer_index: Its index in the trunk.
            adapter: The shared prompt parameters (registered under ``adapter``).
        """
        super().__init__()
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "adapter", adapter)
        self.layer_index = layer_index

    def _attend(
        self,
        site: int,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        prefix: torch.Tensor,
        gate_scale: float,
    ) -> torch.Tensor:
        layer = self.layer
        query_norm = layer.norm_attn(query)
        attn_out, _ = layer.attn(
            query_norm, key_value, key_value, key_padding_mask=key_padding_mask, need_weights=False
        )
        branch = prefix_branch(
            layer.attn, query_norm, prefix, self.adapter.gate(self.layer_index, site), gate_scale
        )
        return query + cast(torch.Tensor, layer.drop_attn(attn_out)) + branch

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        cls_token: torch.Tensor,
        mask_a: torch.Tensor | None,
        mask_b: torch.Tensor | None,
        prefix_a: torch.Tensor,
        prefix_b: torch.Tensor,
        gate_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One bidirectional block plus the gated prefix branch at all three sites.

        Args:
            h_a: Item A hidden states.
            h_b: Item B hidden states.
            cls_token: CLS state ``(B, 1, d_model)``.
            mask_a: Padding mask for A (True = PAD) or ``None``.
            mask_b: Padding mask for B (True = PAD) or ``None``.
            prefix_a: This layer's prefix in the A stream's view.
            prefix_b: This layer's prefix in the B stream's view.
            gate_scale: ``0.0`` realises the gates-off intervention.

        Returns:
            Updated ``(h_a, h_b, cls_token)``.
        """
        layer = self.layer
        h_a = self._attend(0, h_a, h_b, mask_b, prefix_a, gate_scale)
        h_a = layer._ffn(h_a)  # noqa: SLF001
        h_b = self._attend(1, h_b, h_a, mask_a, prefix_b, gate_scale)
        h_b = layer._ffn(h_b)  # noqa: SLF001
        combined = torch.cat([h_a, h_b], dim=1)
        combined_mask = (
            torch.cat([mask_a, mask_b], dim=1) if mask_a is not None and mask_b is not None else None
        )
        cls_norm = layer.norm_cls_attn(cls_token)
        attn_cls, _ = layer.attn_cls(
            cls_norm, combined, combined, key_padding_mask=combined_mask, need_weights=False
        )
        branch = prefix_branch(
            layer.attn_cls, cls_norm, prefix_a, self.adapter.gate(self.layer_index, 2), gate_scale
        )
        cls_token = cls_token + layer.drop_cls_attn(attn_cls) + branch
        cls_token = cls_token + layer.drop_cls_ffn(layer.ff_cls(layer.norm_cls_ffn(cls_token)))
        return h_a, h_b, cls_token
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/model/test_motif_prompt_model.py -n0 -q -k "adapter or views or masking or gates_start"`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/model/egostitch/classifier/motif_prompt.py tests/model/test_motif_prompt_model.py
git commit -m "feat(motif): four-token prefix adapter with per-field row masking"
```

---

### Task 12: `V3_1MotifPrompt` — frozen-base composition, both stages, interventions

**Files:**
- Modify: `src/model/egostitch/classifier/motif_prompt.py`
- Test: `tests/model/test_motif_prompt_model.py`

**Interfaces:**
- Consumes: Tasks 7–11; `src.model.egostitch.classifier.b0_v31.V3_1`, `unpack_pair_batch`, `weighted_pair_bce`.
- Produces: `class V3_1MotifPrompt(nn.Module)` with `name = "v3_1_motif_prompt"`, `encoder` property, `train()` override, `trainable_parameters()`, `optimizer_parameter_groups(generator_lr, interface_lr, weight_decay)`, `interface_open: bool`, `set_corruption_step(step, seed)`, `initialize_teacher()`, `install_mean_template(mean)`, `tokens_from_weights(weights) -> dict[str, torch.Tensor]`, `logits_from_encoded(...)`, `predict_weights(...)`, `forward(batch)`.

- [ ] **Step 1: Write the failing test**

```python
from src.model.egostitch.classifier.motif_prompt import TEMPLATE_KEY, V3_1MotifPrompt

from tests.test_prefix_model import _pair_batch, _tiny_base_config


def _model(stage: str = "one", seed: int = 0, **extra: object) -> V3_1MotifPrompt:
    torch.manual_seed(seed)
    block: dict[str, object] = {
        "stage": stage,
        "base_checkpoint": "base.pt",
        "width": 16,
        "slots_per_field": 2,
        "reader": {"layers": 2, "dim": 16, "heads": 4, "rrwp_k": 4},
    }
    if stage == "two":
        block["bundle_checkpoint"] = "bundle.pt"
    block.update(extra)
    model = V3_1MotifPrompt(base=_tiny_base_config(), motif_prompt=block)
    model.install_mean_template(torch.full((96,), 0.1))
    return model


def _open_gates(model: V3_1MotifPrompt, seed: int = 1) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.adapter.gates.copy_(torch.randn(model.adapter.gates.shape, generator=gen))


@pytest.mark.parametrize("stage", ["one", "two"])
def test_zero_gates_reproduce_the_frozen_base_bit_for_bit(stage: str) -> None:
    model = _model(stage).eval().requires_grad_(False)
    batch = _pair_batch()
    batch[TEMPLATE_KEY] = _weights(n=batch["emb_a"].size(0))
    ours = model(batch)["logits"]
    theirs = model.base(_pair_batch())["logits"]
    assert torch.equal(ours, theirs)


def test_gates_off_intervention_reproduces_the_base_with_open_gates() -> None:
    model = _model("one").eval().requires_grad_(False)
    _open_gates(model)
    batch = _pair_batch()
    batch[TEMPLATE_KEY] = _weights(n=batch["emb_a"].size(0))
    assert not torch.equal(model(batch)["logits"], model.base(_pair_batch())["logits"])
    model.intervention = "gates_off"
    assert torch.equal(model(batch)["logits"], model.base(_pair_batch())["logits"])


def test_stage_one_requires_the_template_and_stage_two_never_reads_one() -> None:
    stage_one = _model("one").eval()
    with pytest.raises(ValueError, match="requires batch"):
        stage_one(_pair_batch())
    stage_two = _model("two").eval()
    out = stage_two(_pair_batch())
    assert "logits" in out and "predicted_weights" in out
    assert out["predicted_weights"].shape[-1] == 96


def test_self_row_masking_is_confined_to_the_slot_and_topo_terms() -> None:
    model = _model("two")
    model.initialize_teacher()
    _open_gates(model)
    batch = _pair_batch(n=4)
    batch[TEMPLATE_KEY] = _weights(n=4)
    batch["motif_mask"] = torch.tensor([1.0, 0.0, 1.0, 1.0])
    out = model(batch)
    assert float(out["slot_loss_rows"][1]) == 0.0
    assert float(out["topo_loss_rows"][1]) == 0.0
    # The masked row still carries task BCE.
    assert float(out["task_loss"]) > 0.0


def test_checkpoint_round_trip_preserves_logits_and_the_mean_template() -> None:
    model = _model("two").eval().requires_grad_(False)
    _open_gates(model)
    batch = _pair_batch()
    before = model(batch)["logits"]
    state = {key: value.clone() for key, value in model.state_dict().items()}
    restored = _model("two", seed=99).eval().requires_grad_(False)
    restored.load_state_dict(state, strict=True)
    torch.testing.assert_close(restored(batch)["logits"], before, rtol=0, atol=0)
    torch.testing.assert_close(restored.mean_template, model.mean_template, rtol=0, atol=0)


def test_interface_parameters_are_registered_trainable_but_gated_by_interface_open() -> None:
    model = _model("two")
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    assert any(name.startswith("generator.") for name in names)
    assert any(name.startswith("reader.") for name in names)
    assert not any(name.startswith("base.") for name in names)
    assert not any(name.startswith("teacher") for name in names)
    groups = model.optimizer_parameter_groups(1e-4, 1e-5, 1e-2)
    assert [group["name"] for group in groups] == ["generator", "interface"]
    assert groups[1]["max_lr"] == 1e-5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/model/test_motif_prompt_model.py -n0 -q -k "gates or stage_one or self_row or round_trip or interface_parameters"`
Expected: FAIL with `ImportError: cannot import name 'V3_1MotifPrompt'`

- [ ] **Step 3: Write minimal implementation**

```python
from copy import deepcopy

from src.data.motif_template import N_EDGES
from src.distill.motif_losses import slot_loss_rows, topo_loss_rows
from src.model.egostitch.classifier.b0_v31 import V3_1, unpack_pair_batch, weighted_pair_bce


class V3_1MotifPrompt(nn.Module):
    """A frozen bidirectional-cross `V3_1` reading a motif graph through gated prefixes.

    ``adapter`` and ``reader`` are registered before ``base`` so
    ``next(model.parameters())`` is trainable (the structural stream builds its
    zero-loss anchor from it). Stage I reads a compiled template from
    ``batch['motif_weights']`` and fails closed without one; Stage II predicts the
    template from the endpoints and never reads a truth graph at inference.
    """

    name: str = "v3_1_motif_prompt"
    mean_template: torch.Tensor

    def __init__(self, *, base: Mapping[str, object], motif_prompt: Mapping[str, object]) -> None:
        """Build the frozen base and the motif path on top.

        Args:
            base: The base `V3_1` constructor kwargs (a checkpoint's ``model_config``).
            motif_prompt: The ``model.config.motif_prompt`` block.

        Raises:
            ValueError: If the base trunk has no bidirectional cross-attention layers.
        """
        super().__init__()
        self.cfg = MotifPromptConfig.from_mapping(motif_prompt)
        self.base_config: dict[str, object] = dict(base)
        base_model = V3_1(**self.base_config)
        trunk = base_model.cross_attention
        if trunk.mixing_mode != "bidirectional_cross" or len(trunk.layers) == 0:
            raise ValueError(
                "v3_1_motif_prompt needs a base with model.config.mixing.mode == "
                "'bidirectional_cross' and at least one cross-attention layer"
            )
        self.d_model = int(base_model.d_model)
        self.input_dim = int(base_model.input_dim)
        self.kd_rep_head = None
        self.kd_struct_head = None
        self.topo_gen = None
        self.reader = MotifGritReader(self.cfg.reader, self.cfg.width)
        self.count_head = MotifCountHead(self.cfg.width)
        self.direct_head = (
            nn.Sequential(
                nn.LayerNorm(2 * self.d_model),
                nn.Linear(2 * self.d_model, 4 * self.cfg.width),
                nn.GELU(),
                nn.Linear(4 * self.cfg.width, 3 * self.cfg.width),
            )
            if self.cfg.token_source == "direct"
            else None
        )
        self.adapter = MotifPromptAdapter(
            self.d_model, len(trunk.layers), int(base_model.n_heads), self.cfg
        )
        self.generator = (
            MotifGenerator(self.d_model, self.cfg) if self.cfg.stage == "two" else None
        )
        self.base = base_model
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.base.eval()
        self.prompt_layers = nn.ModuleList(
            MotifPromptCrossAttentionLayer(cast(CrossAttentionLayer, layer), index, self.adapter)
            for index, layer in enumerate(trunk.layers)
        )
        self.register_buffer("mean_template", torch.zeros(N_EDGES))
        self.teacher: nn.Module | None = None
        self.intervention: str = "none"
        self.corruption_seed = 42
        self.interface_open = self.cfg.stage == "one" or self.cfg.interface_warmup_epochs == 0
```

plus the behavioural half:

```python
    @property
    def encoder(self) -> nn.Module:
        """The frozen base's per-node encoder (packed scoring caches its output)."""
        return self.base.encoder

    def train(self, mode: bool = True) -> V3_1MotifPrompt:
        """Switch the wrapper's mode while the frozen base and teacher stay in eval."""
        super().train(mode)
        self.base.eval()
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Parameters the optimiser updates."""
        return [param for param in self.parameters() if param.requires_grad]

    def _interface_modules(self) -> list[nn.Module]:
        return [self.reader, self.count_head, self.adapter] + (
            [] if self.direct_head is None else [self.direct_head]
        )

    def optimizer_parameter_groups(
        self, generator_lr: float, interface_lr: float, weight_decay: float
    ) -> list[dict[str, object]]:
        """Two named groups: ``generator`` and the 0.1x ``interface`` (spec §7.5).

        Every Stage II parameter that is ever trainable keeps ``requires_grad``
        from construction, so DDP registers and all-reduces it. Epochs 1-2 are
        realised by zeroing the ``interface`` group's LR, which is an exact freeze
        under AdamW's decoupled decay.
        """
        interface = [
            param
            for module in self._interface_modules()
            for param in module.parameters()
            if param.requires_grad
        ]
        interface_ids = {id(param) for param in interface}
        generator = [
            param
            for param in self.parameters()
            if param.requires_grad and id(param) not in interface_ids
        ]
        groups: list[dict[str, object]] = []
        if generator:
            groups.append(
                {
                    "name": "generator",
                    "params": generator,
                    "lr": generator_lr,
                    "max_lr": generator_lr,
                    "weight_decay": weight_decay,
                }
            )
        if interface:
            groups.append(
                {
                    "name": "interface",
                    "params": interface,
                    "lr": interface_lr,
                    "max_lr": interface_lr,
                    "weight_decay": weight_decay,
                }
            )
        return groups

    @torch.no_grad()
    def install_mean_template(self, mean: torch.Tensor) -> None:
        """Publish the training-corpus mean adjacency ``Abar`` (spec §3).

        Args:
            mean: ``(96,)`` mean of the randomised compiled training templates.

        Raises:
            ValueError: On a shape mismatch or a non-finite entry.
        """
        if tuple(mean.shape) != (N_EDGES,) or not torch.isfinite(mean).all():
            raise ValueError(f"mean template must be a finite ({N_EDGES},) vector")
        self.mean_template.copy_(mean.to(self.mean_template))
        if self.generator is not None:
            self.generator.init_biases(self.mean_template)

    def initialize_teacher(self) -> None:
        """Snapshot the loaded Stage I bundle as the immutable teacher ``R_T``."""
        teacher = nn.ModuleDict(
            {
                "reader": deepcopy(self.reader),
                "count_head": deepcopy(self.count_head),
                "adapter": deepcopy(self.adapter),
            }
        )
        self.teacher = teacher.requires_grad_(False).eval()

    def set_corruption_step(self, step: int, seed: int = 42) -> None:
        """Set reproducible Stage I corruption for this global training step."""
        self.corruption_seed = seed + step * 104729

    def _corrupt(self, weights: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg.corruption
        if not self.training or self.cfg.stage != "one" or cfg.prob == 0.0:
            return weights
        rng = torch.Generator(device=weights.device).manual_seed(self.corruption_seed)
        selected = torch.rand((weights.size(0), 1), device=weights.device, generator=rng) < cfg.prob
        span = cfg.lambda_max - cfg.lambda_min
        lam = cfg.lambda_min + span * torch.rand(
            (weights.size(0), 1), device=weights.device, generator=rng
        )
        mixed = (1.0 - lam) * weights + lam * self.mean_template.to(weights).unsqueeze(0)
        return torch.where(selected, mixed, weights)

    def tokens_from_weights(
        self, weights: torch.Tensor, encoded_u: torch.Tensor, encoded_v: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return the four token fields for one batch of motif graphs.

        Args:
            weights: ``(B, 96)`` edge weights.
            encoded_u: Residue states of ``u`` (used only by ``token_source='direct'``).
            encoded_v: Residue states of ``v``.

        Returns:
            ``topo_u``, ``topo_v``, ``topo_rel`` and ``topo_cnt``.
        """
        if self.direct_head is None:
            tokens = self.reader(weights)
        else:
            pooled = torch.cat([encoded_u.mean(dim=1), encoded_v.mean(dim=1)], dim=-1)
            parts = self.direct_head(pooled).chunk(3, dim=-1)
            tokens = {
                "topo_u": parts[0],
                "topo_v": parts[1],
                "topo_rel": parts[2],
            }
        tokens["topo_cnt"] = self.count_head(weights)
        return tokens
```

and the trunk drive plus `forward`, structurally identical to `V3_1TopoPrompt._trunk` / `logits_from_standardized` (same readout branches, same `abba_max` aggregation, same `gate_scale`), with the three additions this family needs: `predict_weights` (Stage II), a `weights` argument in place of coordinates, and the composite loss:

```python
    def predict_weights(
        self,
        encoded_u: torch.Tensor,
        encoded_v: torch.Tensor,
        lengths_u: torch.Tensor,
        lengths_v: torch.Tensor,
    ) -> torch.Tensor:
        """Stage II: predict the 96 edge weights from the endpoints alone.

        Raises:
            RuntimeError: If called on a Stage I model, which has no generator.
        """
        if self.generator is None:
            raise RuntimeError("stage 'one' has no generator; supply batch['motif_weights']")
        return self.generator(encoded_u, encoded_v, lengths_u, lengths_v)

    def forward(
        self, batch: dict[str, torch.Tensor] | None = None, **kwargs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Score pairs and, when targets ride along, return the composite loss.

        Returns:
            ``logits`` always; ``predicted_weights`` in Stage II; with
            ``motif_weights`` also ``slot_loss_rows`` and ``topo_loss_rows``;
            with ``label`` the weighted composite ``loss``, ``loss_weight_sum``,
            the detached ``task_loss`` and the undetached ``loss_term_*`` summands.

        Raises:
            ValueError: If Stage I is called without ``batch['motif_weights']``,
                or an intervention is set while training.
        """
```

Behaviour of `forward`, spelled out so the implementer does not have to infer it:
1. Merge `batch` and `kwargs`; `unpack_pair_batch` to get `emb_a/emb_b/len_a/len_b`.
2. Encode with the frozen encoder under `torch.no_grad()`.
3. Stage I: `weights = merged[TEMPLATE_KEY]`; raise `ValueError("v3_1_motif_prompt stage 'one' requires batch['motif_weights']")` if absent; apply `apply_families` gating in tensor form, then `self._corrupt(weights)`.
   Stage II: `weights = self.predict_weights(...)`; a present `TEMPLATE_KEY` is the *target*, never an input.
4. `tokens = self.tokens_from_weights(weights, encoded_a, encoded_b)`; apply the intervention (`gates_off` → `gate_scale = 0.0`; `mean` → replace `weights` by `self.mean_template` expanded; `shuffle_graph`/`permute_closure`/`rewire_bridge` are scorer-level substitutions that arrive as an explicit `weights` argument and leave the model on `"none"`, exactly as `V3_1Prefix` handles `shuffle`).
5. `view_a, view_b = self.adapter.views(tokens)`; per-layer prefixes; `_trunk` for AB and, unless `order_aggregation == "single"`, for BA with the views exchanged; `torch.max` over the stack; `self.base.output_head(pair_repr)`.
6. If `TEMPLATE_KEY` is present: `row_mask = merged.get(TEMPLATE_MASK_KEY, ones)`; `slot_loss_rows(weights, target, cfg) * row_mask`; when `self.teacher is not None and cfg.w_topo > 0`, `topo_loss_rows(student_tokens, teacher_tokens_on_truth) * row_mask`.
7. If `label` is present: weighted BCE as `V3_1` computes it, then `total_row = bce_row + cfg.w_slot * slot_row + cfg.w_topo * topo_row`, `loss = (weights_row * total_row).sum() / weight_sum`, plus `loss_term_task`, `loss_term_slot`, `loss_term_topo` as undetached summands for the per-epoch gradient probe.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/model/test_motif_prompt_model.py -n0 -q`
Expected: PASS (all model tests; the loss-touching ones need Task 13, so run this step after Task 13 if the implementer follows a strict TDD order — the plan therefore schedules Task 13 next and re-runs this file there.)

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/model/egostitch/classifier/motif_prompt.py tests/model/test_motif_prompt_model.py
git commit -m "feat(motif): V3_1MotifPrompt composition over the frozen prefix_base trunk"
```

---

## Phase 4 — Losses

### Task 13: `L_slot` — the per-family order-statistics loss

**Files:**
- Create: `src/distill/motif_losses.py`
- Test: `tests/distill/test_motif_losses.py`

**Interfaces:**
- Consumes: `src.data.motif_template.slot_profiles`.
- Produces: `slot_loss_rows(predicted: torch.Tensor, target: torch.Tensor, *, beta_p: float, beta_q: float, beta_a: float, beta_i: float, huber_delta: float) -> torch.Tensor` of shape `(B,)`, and `slot_loss(...) -> torch.Tensor` (its mean).

- [ ] **Step 1: Write the failing test**

```python
"""L_slot and L_topo: order statistics, conditioning and the analytic interior derivative."""

from __future__ import annotations

import pytest
import torch
from src.data.motif_template import role_permutation
from src.distill.motif_losses import slot_loss, slot_loss_rows

_BETAS = {"beta_p": 1.0, "beta_q": 1.0, "beta_a": 1.0, "beta_i": 1.0, "huber_delta": 1.0}


def test_a_matched_profile_scores_zero_and_has_zero_gradient() -> None:
    weights = torch.rand(3, 96, requires_grad=True)
    loss = slot_loss(weights, weights.detach().clone(), **_BETAS)
    assert float(loss) == pytest.approx(0.0, abs=1e-12)
    loss.backward()
    assert float(weights.grad.abs().max()) == pytest.approx(0.0, abs=1e-12)


def test_wedge_mass_mismatch_is_penalised_although_the_edge_multisets_agree() -> None:
    target = torch.zeros(1, 96)
    target[0, 0] = 0.5   # w(u, c0)
    target[0, 8] = 0.5   # w(c0, v)
    predicted = torch.zeros(1, 96)
    predicted[0, 0] = 0.5  # w(u, c0)
    predicted[0, 1] = 0.5  # w(u, c1)
    assert sorted(predicted[0, :16].tolist()) == sorted(target[0, :16].tolist())
    loss = slot_loss(predicted, target, beta_p=1.0, beta_q=0.0, beta_a=0.0, beta_i=0.0, huber_delta=1.0)
    assert float(loss) == pytest.approx(0.00390625, rel=1e-9)


def test_the_slot_misalignment_fixture_of_the_withdrawn_collapse_control() -> None:
    target = torch.zeros(1, 96)
    target[0, 0] = target[0, 8] = 0.5
    predicted = torch.zeros(1, 96)
    predicted[0, 0] = predicted[0, 8] = 0.25
    predicted[0, 1] = predicted[0, 9] = 0.25
    loss = slot_loss(predicted, target, beta_p=1.0, beta_q=0.0, beta_a=0.0, beta_i=0.0, huber_delta=1.0)
    assert float(loss) == pytest.approx(0.00244140625, rel=1e-9)


def test_loss_is_invariant_to_a_within_role_permutation_of_either_argument() -> None:
    predicted = torch.rand(2, 96)
    target = torch.rand(2, 96)
    perm = torch.as_tensor(
        role_permutation((1, 0, 3, 2, 5, 4, 7, 6), (4, 5, 6, 7, 0, 1, 2, 3), (2, 3, 4, 5, 6, 7, 0, 1))
    )
    shuffled = torch.zeros_like(predicted)
    shuffled[:, perm] = predicted
    shuffled_target = torch.zeros_like(target)
    shuffled_target[:, perm] = target
    base = slot_loss_rows(predicted, target, **_BETAS)
    torch.testing.assert_close(slot_loss_rows(shuffled, target, **_BETAS), base, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        slot_loss_rows(shuffled, shuffled_target, **_BETAS), base, rtol=1e-6, atol=1e-7
    )


def test_gradients_are_finite_at_zero_and_at_tiny_weights() -> None:
    for scale in (0.0, 1e-8):
        logits = torch.full((2, 96), -20.0, requires_grad=True)
        predicted = torch.sigmoid(logits) * 0.0 + scale
        predicted = predicted + logits * 0.0
        loss = slot_loss(predicted, torch.zeros(2, 96), **_BETAS)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(logits.grad).all()


def test_gradients_reach_all_96_edges_on_a_non_degenerate_fixture() -> None:
    gen = torch.Generator().manual_seed(0)
    logits = (torch.rand(4, 96, generator=gen) * 2 - 1).requires_grad_(True)
    target = torch.rand(4, 96, generator=gen)
    slot_loss(torch.sigmoid(logits), target, **_BETAS).backward()
    reached = (logits.grad.abs() > 0).any(dim=0)
    assert int(reached.sum()) == 96


def test_the_analytic_interior_derivative_rises_by_exactly_1e4_with_beta_i() -> None:
    # Spec §7.3: every attachment 0.1, predicted interior 0.1, true interior 1.
    def build(beta_i: float) -> float:
        interior_logit = torch.full((1, 64), float(torch.logit(torch.tensor(0.1))), requires_grad=True)
        predicted = torch.zeros(1, 96)
        predicted[0, 16:32] = 0.1
        predicted = predicted.clone()
        predicted[0, 32:] = torch.sigmoid(interior_logit)[0]
        target = torch.zeros(1, 96)
        target[0, 16:32] = 0.1
        target[0, 32:] = 1.0
        loss = slot_loss(
            predicted, target, beta_p=1.0, beta_q=1.0, beta_a=1.0, beta_i=beta_i, huber_delta=1.0
        )
        loss.backward()
        return float(interior_logit.grad.abs().max())

    without = build(0.0)
    with_interior = build(1.0)
    assert without == pytest.approx(1.266e-7, rel=2e-3)
    assert with_interior == pytest.approx(1.266e-3, rel=2e-3)
    assert with_interior / without == pytest.approx(1e4, rel=5e-3)


def test_the_product_term_alone_reproduces_the_documented_mean_of_4_05e_minus_5() -> None:
    predicted = torch.zeros(1, 96)
    predicted[0, 16:32] = 0.1
    predicted[0, 32:] = 0.1
    target = torch.zeros(1, 96)
    target[0, 16:32] = 0.1
    target[0, 32:] = 1.0
    loss = slot_loss(predicted, target, beta_p=0.0, beta_q=1.0, beta_a=0.0, beta_i=0.0, huber_delta=1.0)
    assert float(loss) == pytest.approx(4.05e-5, rel=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/distill/test_motif_losses.py -n0 -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.distill.motif_losses'`

- [ ] **Step 3: Write minimal implementation**

```python
"""The motif-prompt training losses: ``L_slot`` (spec §7.3) and ``L_topo`` (spec §7.5)."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from src.data.motif_template import slot_profiles

_TERMS = (("p", "beta_p"), ("q", "beta_q"), ("a", "beta_a"), ("b", "beta_i"))


def slot_loss_rows(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    beta_p: float,
    beta_q: float,
    beta_a: float,
    beta_i: float,
    huber_delta: float,
) -> torch.Tensor:
    """Per-row ``L_slot``: descending-sorted Huber over four per-family multisets.

    The two product terms route gradient to both (or all three) edges of each
    motif; the two raw terms keep gradient flowing to an edge whose path product
    vanishes because a *different* edge of that path is zero. The raw interior
    term is not decoration: for ``q = a b c`` the interior gradient is attenuated
    quadratically by small attachments, and the attachment term contributes
    nothing to it (spec §7.3).

    Args:
        predicted: ``(B, 96)`` predicted edge weights.
        target: ``(B, 96)`` compiled edge weights.
        beta_p: Weight of the wedge-product term.
        beta_q: Weight of the bridge-path-product term.
        beta_a: Weight of the raw attachment term.
        beta_i: Weight of the raw interior term.
        huber_delta: Huber transition point.

    Returns:
        ``(B,)`` per-row losses.

    Raises:
        ValueError: If the two tables disagree in shape.
    """
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted {tuple(predicted.shape)} and target {tuple(target.shape)} must match"
        )
    parts = slot_profiles(predicted)
    truth = slot_profiles(target.detach())
    betas = {"beta_p": beta_p, "beta_q": beta_q, "beta_a": beta_a, "beta_i": beta_i}
    total = predicted.new_zeros(predicted.size(0), dtype=torch.float32)
    for key, name in _TERMS:
        scale = float(betas[name])
        if scale == 0.0:
            continue
        ours = parts[key].sort(dim=1, descending=True).values
        theirs = truth[key].sort(dim=1, descending=True).values
        term = F.huber_loss(ours, theirs, delta=huber_delta, reduction="none").mean(dim=1)
        total = total + scale * term
    return total


def slot_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    beta_p: float,
    beta_q: float,
    beta_a: float,
    beta_i: float,
    huber_delta: float,
) -> torch.Tensor:
    """Mean ``L_slot`` over the batch; see `slot_loss_rows`."""
    return slot_loss_rows(
        predicted,
        target,
        beta_p=beta_p,
        beta_q=beta_q,
        beta_a=beta_a,
        beta_i=beta_i,
        huber_delta=huber_delta,
    ).mean()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/distill/test_motif_losses.py -n0 -q && .venv/bin/python -m pytest tests/model/test_motif_prompt_model.py -n0 -q`
Expected: PASS (8 loss tests; the model file now also passes in full)

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/distill/motif_losses.py tests/distill/test_motif_losses.py
git commit -m "feat(motif): L_slot order-statistics loss with the raw interior term"
```

---

### Task 14: `L_topo` — the light representation term

**Files:**
- Modify: `src/distill/motif_losses.py`
- Test: `tests/distill/test_motif_losses.py`

**Interfaces:**
- Consumes: Task 12 token dicts.
- Produces: `topo_loss_rows(student: Mapping[str, torch.Tensor], teacher: Mapping[str, torch.Tensor]) -> torch.Tensor` of shape `(B,)`; `stream_mean(rows: Sequence[torch.Tensor], masks: Sequence[torch.Tensor], *, like: torch.Tensor) -> torch.Tensor`.

- [ ] **Step 1: Write the failing test**

```python
from src.distill.motif_losses import stream_mean, topo_loss_rows

_FIELDS = ("topo_u", "topo_v", "topo_rel", "topo_cnt")


def _tokens(n: int = 3, width: int = 8, seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {name: torch.randn(n, width, generator=gen) for name in _FIELDS}


def test_topo_loss_is_zero_on_identical_tokens_and_ignores_a_global_shift_and_scale() -> None:
    tokens = _tokens()
    rows = topo_loss_rows(tokens, tokens)
    assert rows.shape == (3,)
    assert float(rows.abs().max()) == pytest.approx(0.0, abs=1e-6)
    scaled = {name: 3.0 * value + 5.0 for name, value in tokens.items()}
    assert float(topo_loss_rows(scaled, tokens).abs().max()) == pytest.approx(0.0, abs=1e-5)


def test_topo_loss_detaches_the_teacher_but_not_the_student() -> None:
    student = {name: value.clone().requires_grad_(True) for name, value in _tokens(seed=1).items()}
    teacher = {name: value.clone().requires_grad_(True) for name, value in _tokens(seed=2).items()}
    topo_loss_rows(student, teacher).sum().backward()
    assert all(value.grad is not None for value in student.values())
    assert all(value.grad is None for value in teacher.values())


def test_topo_loss_covers_all_four_fields() -> None:
    student = _tokens(seed=1)
    teacher = _tokens(seed=1)
    for name in _FIELDS:
        perturbed = dict(student)
        perturbed[name] = student[name] + 1.5
        assert float(topo_loss_rows(perturbed, teacher).sum()) > 0.0


def test_an_empty_stream_contributes_a_differentiable_zero() -> None:
    anchor = torch.zeros((), requires_grad=True)
    rows = torch.stack([torch.tensor(0.5), torch.tensor(1.5)]) + anchor
    mask = torch.tensor([1.0, 0.0])
    torch.testing.assert_close(stream_mean([rows], [mask], like=anchor), torch.tensor(0.5))
    empty = stream_mean([], [], like=anchor)
    assert float(empty) == 0.0
    assert empty.requires_grad
    torch.testing.assert_close(
        stream_mean([rows, rows.new_zeros(0)], [mask, mask.new_zeros(0)], like=anchor),
        torch.tensor(0.5),
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/distill/test_motif_losses.py -n0 -q -k "topo or empty_stream"`
Expected: FAIL with `ImportError: cannot import name 'topo_loss_rows'`

- [ ] **Step 3: Write minimal implementation**

```python
from collections.abc import Mapping, Sequence

TOPO_FIELDS = ("topo_u", "topo_v", "topo_rel", "topo_cnt")


def topo_loss_rows(
    student: Mapping[str, torch.Tensor], teacher: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """Per-row ``L_topo``: squared error of fixed non-affine LayerNorms (spec §7.5).

    ``N`` is `torch.nn.functional.layer_norm` with no learnable affine, applied to
    each of the four token fields, so the term compares directions rather than
    scales. The teacher side is detached; ``R_T(Ahat)`` runs with autograd so the
    loss reaches the generator.

    Args:
        student: The four fields read from the predicted graph.
        teacher: The four fields the immutable teacher read from the true graph.

    Returns:
        ``(B,)`` mean squared error over fields and dimensions.

    Raises:
        KeyError: If either mapping is missing one of `TOPO_FIELDS`.
    """
    terms: list[torch.Tensor] = []
    for name in TOPO_FIELDS:
        ours = student[name].float()
        theirs = teacher[name].float().detach()
        width = ours.size(-1)
        normalised = F.layer_norm(ours, (width,))
        reference = F.layer_norm(theirs, (width,))
        terms.append((normalised - reference).square().mean(dim=-1))
    return torch.stack(terms, dim=-1).mean(dim=-1)


def stream_mean(
    rows: Sequence[torch.Tensor], masks: Sequence[torch.Tensor], *, like: torch.Tensor
) -> torch.Tensor:
    """Average per-stream row means, then average across the non-empty streams.

    An empty stream — no rows at all, or no valid nonself rows — contributes a
    differentiable zero rather than dropping out of the graph, so DDP sees the
    same parameters on every rank (spec §7.5).

    Args:
        rows: One ``(n_s,)`` per-row tensor per stream.
        masks: One ``(n_s,)`` valid-row mask per stream, aligned with ``rows``.
        like: A tensor supplying dtype, device and the autograd connection.

    Returns:
        The scalar mean.
    """
    zero = like.sum() * 0.0
    means: list[torch.Tensor] = []
    for row, mask in zip(rows, masks, strict=True):
        if row.numel() == 0:
            continue
        weight = mask.to(row).sum()
        if float(weight) == 0.0:
            continue
        means.append((row * mask.to(row)).sum() / weight)
    if not means:
        return zero
    return zero + torch.stack(means).mean()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/distill/test_motif_losses.py -n0 -q`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/distill/motif_losses.py tests/distill/test_motif_losses.py
git commit -m "feat(motif): L_topo representation term with differentiable empty streams"
```

---

### Task 15: Wire `L_slot` and `L_topo` into the structural stream

**Files:**
- Modify: `src/train_b0.py:4259-4456` (`StructStream._score` and `StructStream.loss`)
- Test: `tests/test_train_b0_motif_prompt.py`

**Interfaces:**
- Consumes: Task 12 `V3_1MotifPrompt`, Task 14 `stream_mean`, Task 4 `MotifTemplateTable.weights_for_pairs`.
- Produces: `StructStream.last_motif_rows: dict[str, torch.Tensor]` carrying `slot` and `topo` per-row tensors of the last scored subgraph, and `StructStream.loss` returning them inside its `stats` mapping as `struct_motif_slot_rows` / `struct_motif_topo_rows`.

- [ ] **Step 1: Write the failing test**

```python
"""Trainer plumbing for the v3_1_motif_prompt family."""

from __future__ import annotations

import networkx as nx
import pytest
import torch


def test_struct_stream_compiles_a_template_for_every_sampled_subgraph_pair() -> None:
    from src.train_b0 import StructStream

    stream, model, subgraph = _tiny_struct_stream()
    logits, target, mask = stream._score(model, subgraph, stream._sampler)  # noqa: SLF001
    assert logits.shape == target.shape == mask.shape
    assert stream.last_motif_rows["slot"].numel() > 0
    assert stream.last_motif_rows["topo"].numel() == stream.last_motif_rows["slot"].numel()


def test_struct_stream_templates_come_from_the_training_graph_with_the_pair_removed() -> None:
    stream, model, subgraph = _tiny_struct_stream()
    table = stream._motif_table  # noqa: SLF001
    u, v = subgraph.nodes[0], subgraph.nodes[1]
    row = table.compile_row(u, v)
    assert v not in row.closure and u not in row.closure


def test_struct_stream_topo_rows_are_a_differentiable_zero_without_a_teacher() -> None:
    stream, model, subgraph = _tiny_struct_stream(with_teacher=False)
    stream._score(model, subgraph, stream._sampler)  # noqa: SLF001
    rows = stream.last_motif_rows["topo"]
    assert float(rows.abs().sum()) == 0.0
    assert rows.requires_grad
```

`_tiny_struct_stream` builds a 6-node training graph, a `StructSampler` over it, a `PackedFeatureTable` stub from `tests/test_struct_stream.py`'s existing helper, and a Stage II `V3_1MotifPrompt` with `initialize_teacher()` called when `with_teacher`. Copy the table stub from `tests/test_struct_stream.py`; do not re-implement it.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_train_b0_motif_prompt.py -n0 -q -k struct_stream`
Expected: FAIL with `AttributeError: 'StructStream' object has no attribute 'last_motif_rows'`

- [ ] **Step 3: Write minimal implementation**

In `StructStream.__init__`, add a `templates: MotifTemplateTable | None = None` keyword stored as `self._motif_table`, and initialise `self.last_motif_rows: dict[str, torch.Tensor] = {}`.

In `StructStream._score`, alongside the existing `pair_coords` block (currently `src/train_b0.py:4297-4312`), add:

```python
        pair_templates: torch.Tensor | None = None
        if isinstance(raw_model, V3_1MotifPrompt):
            if self._motif_table is None:
                raise RuntimeError("motif-prompt structural stream requires a template table")
            table = self._motif_table
            a = [table.index[subgraph.nodes[i]] for i, _ in pairs]
            b = [table.index[subgraph.nodes[j]] for _, j in pairs]
            pair_templates = torch.from_numpy(
                table.weights_by_index(np.asarray(a), np.asarray(b))
            ).to(device)
```

and inside the chunk loop, pass `templates = pair_templates[chunk] if pair_templates is not None else None` into the `forward` closure, which returns a tuple when the model is a `V3_1MotifPrompt`:

```python
                def forward(
                    anchor: torch.Tensor,
                    partner: torch.Tensor,
                    boundary: int,
                    coords: torch.Tensor | None,
                    templates: torch.Tensor | None,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                    """Score one chunk; motif rows also carry their L_slot/L_topo terms."""
                    emb_a, len_a = self._table.gather_nodes(anchor, boundary)
                    emb_b, len_b = self._table.gather_nodes(partner, boundary)
                    batch = {"emb_a": emb_a, "emb_b": emb_b, "len_a": len_a, "len_b": len_b}
                    if isinstance(raw_model, V3_1TopoPrompt):
                        assert coords is not None
                        batch[COORDS_KEY] = coords
                    if templates is not None:
                        batch[TEMPLATE_KEY] = templates
                    output = cast(dict[str, torch.Tensor], raw_model(batch))
                    out = output["logits"]
                    if out.dim() > 1 and out.size(-1) == 1:
                        out = out.squeeze(-1)
                    zero = out.sum() * 0.0
                    return (
                        out.float(),
                        output.get("slot_loss_rows", out.new_zeros(out.shape[0]) + zero).float(),
                        output.get("topo_loss_rows", out.new_zeros(out.shape[0]) + zero).float(),
                    )
```

Collect the second and third outputs into `slot_parts` / `topo_parts` exactly as `parts` is collected, concatenate them after the loop, and store:

```python
        anchor_zero = dependency.float()
        self.last_motif_rows = {
            "slot": (torch.cat(slot_parts) if slot_parts else flat.new_zeros(0)) + anchor_zero,
            "topo": (torch.cat(topo_parts) if topo_parts else flat.new_zeros(0)) + anchor_zero,
        }
```

In `StructStream.loss`, after the existing `struct_total(...)` call, return the two row tensors in the `stats` mapping under `struct_motif_slot_rows` / `struct_motif_topo_rows` so the trainer can fold them into `stream_mean`. The structural loss itself is unchanged — `L_slot` and `L_topo` are added once by the trainer in Task 19, never twice.

Construct the table in `_run_ddp_worker` beside `struct_sampler` (`src/train_b0.py:6664-6694`), passing `templates=motif_rows.train_table if motif_rows is not None else None`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_train_b0_motif_prompt.py -n0 -q -k struct_stream && .venv/bin/python -m pytest tests/test_train_b0_struct.py tests/test_struct_stream.py -n0 -q`
Expected: PASS (3 new tests; the two existing struct files keep passing because `forward` returns a tuple only for this family and every other caller reads element 0 via the unchanged `parts` list)

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/train_b0.py tests/test_train_b0_motif_prompt.py
git commit -m "feat(motif): per-stream L_slot and L_topo rows from the structural stream"
```

---

## Phase 5 — Training and scoring plumbing

### Task 16: Family registration in `train_b0`

**Files:**
- Modify: `src/train_b0.py:152-162`, `:719-790`, `:1119-1172`, `:1341-1396`, `:1924-2004`
- Test: `tests/test_train_b0_motif_prompt.py`

**Interfaces:**
- Consumes: Task 12 `V3_1MotifPrompt`.
- Produces: `MOTIF_PROMPT_FAMILY = "v3_1_motif_prompt"` in `MODEL_FAMILIES` and `V3_1_FAMILIES`; `_resolve_motif_prompt_kwargs(model_cfg) -> dict[str, object]`; a `build_model` branch; a `_run_metadata` provenance block.

- [ ] **Step 1: Write the failing test**

```python
def test_the_family_is_accepted_by_the_config_schema_and_the_v3_1_gate() -> None:
    from src.train_b0 import MODEL_FAMILIES, MOTIF_PROMPT_FAMILY, V3_1_FAMILIES, is_v3_1_family

    assert MOTIF_PROMPT_FAMILY == "v3_1_motif_prompt"
    assert MOTIF_PROMPT_FAMILY in MODEL_FAMILIES
    assert MOTIF_PROMPT_FAMILY in V3_1_FAMILIES
    assert is_v3_1_family(MOTIF_PROMPT_FAMILY)


def test_stage_one_kwargs_take_an_inline_base_or_a_base_checkpoint_but_not_both() -> None:
    from src.train_b0 import ModelConfig, resolve_model_kwargs

    inline = ModelConfig(
        family="v3_1_motif_prompt",
        config={"motif_prompt": {"stage": "one", "base_checkpoint": "b.pt"}, "base": {"d_model": 8}},
    )
    kwargs = resolve_model_kwargs(inline)
    assert set(kwargs) == {"motif_prompt", "base"}
    with pytest.raises(ValueError, match="accepts only model.config"):
        resolve_model_kwargs(
            ModelConfig(family="v3_1_motif_prompt", config={"motif_prompt": {}, "reader": {}})
        )
    with pytest.raises(ValueError, match="model.config.motif_prompt is required"):
        resolve_model_kwargs(ModelConfig(family="v3_1_motif_prompt", config={}))


def test_base_loss_kwargs_unwraps_the_motif_nesting() -> None:
    from src.train_b0 import ModelConfig, _base_loss_kwargs

    cfg = ModelConfig(
        family="v3_1_motif_prompt",
        config={
            "motif_prompt": {"stage": "one", "base_checkpoint": "b.pt"},
            "base": {"positive_weight": 7.0, "label_smoothing": 0.25},
        },
    )
    resolved = _base_loss_kwargs(cfg)
    assert resolved["positive_weight"] == 7.0
    assert resolved["label_smoothing"] == 0.25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_train_b0_motif_prompt.py -n0 -q -k "family or kwargs"`
Expected: FAIL with `ImportError: cannot import name 'MOTIF_PROMPT_FAMILY'`

- [ ] **Step 3: Write minimal implementation**

At `src/train_b0.py:152-155`:

```python
MODEL_FAMILIES = (
    "v3_1",
    "v3_1_prefix",
    "v3_1_topo_prompt",
    "v3_1_coord_gen",
    "v3_1_motif_prompt",
    "f0_mlp",
)
V3_1_FAMILIES = frozenset(
    {"v3_1", "v3_1_prefix", "v3_1_topo_prompt", "v3_1_coord_gen", "v3_1_motif_prompt"}
)
TOPO_PROMPT_FAMILY = "v3_1_topo_prompt"
COORD_GEN_FAMILY = "v3_1_coord_gen"
MOTIF_PROMPT_FAMILY = "v3_1_motif_prompt"
```

New resolver, modelled on `_resolve_topo_prompt_kwargs` (`:1193-1239`):

```python
def _resolve_motif_prompt_kwargs(model_cfg: ModelConfig) -> dict[str, object]:
    """Resolve the ``v3_1_motif_prompt`` constructor kwargs.

    Args:
        model_cfg: The ``model:`` config section.

    Returns:
        ``{"motif_prompt": ..., "base": ...}``.

    Raises:
        ValueError: If the block is missing, an unexpected key is present, or
            neither an inline ``base`` nor a ``base_checkpoint`` names the trunk.
    """
    raw = dict(model_cfg.config)
    block = raw.pop("motif_prompt", None)
    if not isinstance(block, Mapping):
        raise ValueError("model.config.motif_prompt is required for v3_1_motif_prompt")
    base = raw.pop("base", None)
    extra = sorted(raw)
    if extra:
        raise ValueError(
            f"v3_1_motif_prompt accepts only model.config.{{motif_prompt,base}}, got {extra}"
        )
    checkpoint = str(cast(Mapping[str, object], block).get("base_checkpoint", "") or "")
    if base is None:
        if not checkpoint:
            raise ValueError(
                "v3_1_motif_prompt needs model.config.base or motif_prompt.base_checkpoint"
            )
        payload, digest = _load_base_checkpoint(Path(checkpoint))
        base = cast(Mapping[str, object], payload["model_config"])
        block = {**cast(Mapping[str, object], block), "base_checkpoint_sha256": digest}
    return {"motif_prompt": dict(cast(Mapping[str, object], block)), "base": dict(base)}
```

Dispatch at `:1133`: `if model_cfg.family == MOTIF_PROMPT_FAMILY: return _resolve_motif_prompt_kwargs(model_cfg)`, and extend the error string at `:1144-1145` to name the family.

`_base_loss_kwargs` at `:1165`: add `MOTIF_PROMPT_FAMILY` to the tuple that unwraps `kwargs["base"]`, i.e. `if model_cfg.family in ("v3_1_prefix", TOPO_PROMPT_FAMILY, MOTIF_PROMPT_FAMILY):`.

`build_model` branch after the `COORD_GEN_FAMILY` branch:

```python
    if cfg.model.family == MOTIF_PROMPT_FAMILY:
        motif_model = V3_1MotifPrompt(**kwargs)  # type: ignore[arg-type]
        block = cast(Mapping[str, object], kwargs["motif_prompt"])
        base_checkpoint = str(block.get("base_checkpoint", "") or "")
        if base_checkpoint:
            payload, _ = _load_base_checkpoint(Path(base_checkpoint))
            motif_model.base.load_state_dict(cast(Mapping[str, Any], payload["model_state"]))
        bundle = str(block.get("bundle_checkpoint", "") or "")
        if bundle:
            stage_one, _ = _load_motif_bundle(Path(bundle))
            motif_model.load_state_dict(
                cast(Mapping[str, Any], stage_one["model_state"]), strict=False
            )
            motif_model.initialize_teacher()
        _validate_topo_gen_distill_contract(motif_model, cfg.distill)
        return motif_model
```

with `_load_motif_bundle` mirroring `_load_reader_checkpoint` (`:1241-1259`) but requiring `payload["model_family"] == MOTIF_PROMPT_FAMILY`.

`_run_metadata` provenance block after the coord-gen one:

```python
    if cfg.model.family == MOTIF_PROMPT_FAMILY:
        motif_kwargs = cast(Mapping[str, object], model_kwargs["motif_prompt"])
        run_metadata["motif_prompt"] = {
            "stage": motif_kwargs.get("stage"),
            "fields": list(cast(Sequence[str], motif_kwargs.get("fields", FIELD_ORDER))),
            "families": list(cast(Sequence[str], motif_kwargs.get("families", ("closure", "bridge")))),
            "token_source": motif_kwargs.get("token_source"),
            "gate_mode": motif_kwargs.get("gate_mode"),
            "base_checkpoint": motif_kwargs.get("base_checkpoint") or None,
            "base_checkpoint_sha256": motif_kwargs.get("base_checkpoint_sha256"),
            "bundle_checkpoint": motif_kwargs.get("bundle_checkpoint") or None,
            "bundle_checkpoint_sha256": motif_kwargs.get("bundle_checkpoint_sha256"),
        }
```

Checkpoint paths and digests are recorded, never verified — the project bans formalism gates.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_train_b0_motif_prompt.py -n0 -q -k "family or kwargs" && .venv/bin/python -m pytest tests/test_train_b0.py -n0 -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/train_b0.py tests/test_train_b0_motif_prompt.py
git commit -m "feat(motif): register the v3_1_motif_prompt family in the trainer"
```

---

### Task 17: `MotifTemplateRows` — per-row targets, attach hooks, stage guards

**Files:**
- Modify: `src/train_b0.py:3800-3946` (beside `TopoPromptRows`), `:5456-5460`, `:6379-6380`, `:6513-6606`, `:6733`, `:6844`
- Test: `tests/test_train_b0_motif_prompt.py`

**Interfaces:**
- Consumes: Task 4 `MotifTemplateTable`, `mean_template`; `TrainingCorpus`.
- Produces: `class MotifTemplateRows` with `__init__(self, *, train_graph, train_pairs, stats_rows, val_graph, val_cls_pairs, universe_pairs, device, seed)`, attributes `train_table`, `train`, `val_cls`, `universe`, `mean`, `train_mask`, and methods `install(model)`, `attach_train(batch)`, `attach_val(batch)`, `summary()`.

- [ ] **Step 1: Write the failing test**

```python
def test_motif_rows_attach_the_template_and_mask_by_row_id() -> None:
    from src.train_b0 import MotifTemplateRows

    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")])
    pairs = [("a", "b"), ("a", "c"), ("a", "a")]
    rows = MotifTemplateRows(
        train_graph=graph,
        train_pairs=pairs,
        stats_rows=np.asarray([0, 1, 2]),
        val_graph=graph,
        val_cls_pairs=pairs,
        universe_pairs=pairs,
        device=torch.device("cpu"),
        seed=0,
    )
    batch = {"_row_id": torch.tensor([2, 0])}
    rows.attach_train(batch)
    assert batch["motif_weights"].shape == (2, 96)
    torch.testing.assert_close(batch["motif_weights"], rows.train[[2, 0]])
    # The self row is excluded from L_slot and L_topo but keeps task BCE.
    torch.testing.assert_close(batch["motif_mask"], torch.tensor([0.0, 1.0]))


def test_the_mean_template_is_measured_over_the_epoch_one_rows_only() -> None:
    from src.train_b0 import MotifTemplateRows

    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
    pairs = [("a", "b"), ("a", "c"), ("b", "c")]
    rows = MotifTemplateRows(
        train_graph=graph,
        train_pairs=pairs,
        stats_rows=np.asarray([0, 1]),
        val_graph=graph,
        val_cls_pairs=pairs,
        universe_pairs=pairs,
        device=torch.device("cpu"),
        seed=0,
    )
    torch.testing.assert_close(rows.mean, rows.train[[0, 1]].mean(dim=0))


def test_stage_one_refuses_a_formal_run_and_stage_two_refuses_a_diagnostic_one() -> None:
    from src.train_b0 import _assert_motif_run_kind

    with pytest.raises(RuntimeError, match="--run-kind diagnostic"):
        _assert_motif_run_kind(stage="one", run_kind="formal")
    _assert_motif_run_kind(stage="one", run_kind="diagnostic")
    _assert_motif_run_kind(stage="two", run_kind="formal")
    with pytest.raises(RuntimeError, match="deployable"):
        _assert_motif_run_kind(stage="two", run_kind="diagnostic")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_train_b0_motif_prompt.py -n0 -q -k "motif_rows or mean_template or run_kind"`
Expected: FAIL with `ImportError: cannot import name 'MotifTemplateRows'`

- [ ] **Step 3: Write minimal implementation**

```python
class MotifTemplateRows:
    """Compiled motif templates for every row a motif-prompt run scores.

    Training rows are compiled on the V_val-masked training graph; the V_val
    classification rows and the V_val topology universe on the V_val gold graph,
    which is held-out truth and therefore Stage I only. Every rank builds the
    same tensors (deterministic), keeps them on the CPU, and slices per batch by
    ``_row_id`` — the same contract `TopoPromptRows` uses.
    """

    def __init__(
        self,
        *,
        train_graph: nx.Graph,
        train_pairs: Sequence[Pair],
        stats_rows: np.ndarray,
        val_graph: nx.Graph,
        val_cls_pairs: Sequence[Pair],
        universe_pairs: Sequence[Pair],
        device: torch.device,
        seed: int,
    ) -> None:
        """Compile every row once.

        Args:
            train_graph: Loopless training structural graph (V_val excluded).
            train_pairs: The trainer's training rows in row-id order.
            stats_rows: Row ids the mean template is taken over (epoch 1's rows).
            val_graph: Loopless V_val gold graph (Stage I diagnostics only).
            val_cls_pairs: The V_val classification rows in row-id order.
            universe_pairs: The V_val topology ball-union rows in row-id order.
            device: Device batches live on.
            seed: Slot-randomisation seed lane.
        """
        started = time.monotonic()
        self.train_table = MotifTemplateTable(train_graph)
        self.train_pairs: list[Pair] = list(train_pairs)
        self.train = torch.from_numpy(
            self.train_table.weights(train_pairs, seed=seed, epoch=1, randomise=True)
        )
        self.train_mask = torch.tensor(
            [0.0 if u == v else 1.0 for u, v in train_pairs], dtype=torch.float32
        )
        self.mean = self.train[torch.from_numpy(np.asarray(stats_rows)).long()].mean(dim=0)
        self.val_table = MotifTemplateTable(val_graph)
        self.val_cls = torch.from_numpy(self.val_table.weights(val_cls_pairs))
        self.val_cls_mask = torch.tensor(
            [0.0 if u == v else 1.0 for u, v in val_cls_pairs], dtype=torch.float32
        )
        self.universe = torch.from_numpy(self.val_table.weights(universe_pairs))
        self._device = device
        self.build_seconds = time.monotonic() - started
        self.compile_rate = self.train_table.summary()["compile_seconds_per_10k"]

    def install(self, model: nn.Module) -> None:
        """Publish the training mean adjacency into the model's buffer."""
        raw_model = _unwrapped_model(model)
        if not isinstance(raw_model, V3_1MotifPrompt):
            raise TypeError("MotifTemplateRows.install needs a V3_1MotifPrompt")
        raw_model.install_mean_template(self.mean)

    def _attach(self, batch: Batch, table: torch.Tensor, mask: torch.Tensor) -> None:
        rows = batch["_row_id"].detach().to("cpu", torch.int64)
        batch[TEMPLATE_KEY] = table.index_select(0, rows).to(self._device, non_blocking=True)
        batch[TEMPLATE_MASK_KEY] = mask.index_select(0, rows).to(
            self._device, torch.float32, non_blocking=True
        )

    def attach_train(self, batch: Batch) -> None:
        """Inject this training batch's compiled templates and self-row mask."""
        self._attach(batch, self.train, self.train_mask)

    def attach_val(self, batch: Batch) -> None:
        """Inject this V_val classification batch's templates by ``_row_id``."""
        self._attach(batch, self.val_cls, self.val_cls_mask)

    def summary(self) -> dict[str, object]:
        """Provenance for logs and ``run_metadata.json``."""
        return {
            "train_rows": int(self.train.shape[0]),
            "stats_rows": int(self.train.shape[0]),
            "val_cls_rows": int(self.val_cls.shape[0]),
            "universe_rows": int(self.universe.shape[0]),
            "build_seconds": self.build_seconds,
            "compile_seconds_per_10k": self.compile_rate,
        }


def _assert_motif_run_kind(*, stage: str, run_kind: str | None) -> None:
    """Fail closed on a stage/run-kind mismatch.

    Stage I compiles true V_val templates for validation, so it is a ceiling
    diagnostic; Stage II is deployable and must not be published as one (spec §8).

    Raises:
        RuntimeError: On a Stage I formal run or a Stage II diagnostic run.
    """
    if stage == "one" and run_kind != "diagnostic":
        raise RuntimeError(
            "v3_1_motif_prompt stage 'one' compiles V_val truth templates during validation; "
            "launch it with --run-kind diagnostic "
            "(hpc/run.sh train <config> --run-kind diagnostic)"
        )
    if stage == "two" and run_kind == "diagnostic":
        raise RuntimeError(
            "v3_1_motif_prompt stage 'two' is deployable: it scores from (x_u, x_v) alone and "
            "must be launched as a formal run"
        )
```

Then wire it into `_run_ddp_worker` beside the existing `V3_1TopoPrompt` / `V3_1CoordGen` branches (`:6513-6606`): build `motif_rows` for both stages, call `_assert_motif_run_kind`, `motif_rows.install(model)`, and pass `templates=motif_rows.train_table` into `StructStream`. Add `motif_rows.attach_train(batch)` at `:5458` beside `topo_rows.attach_train`, `motif_rows.attach_val` into the two `attach=` callbacks (`:6733`, `:6844`), and the probe-mode attach at `:6379`. Stage II's V_val universe pass must **not** receive `universe` — the deployable path scores from features alone; `universe` exists only for Stage I's labelled diagnostic.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_train_b0_motif_prompt.py -n0 -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/train_b0.py tests/test_train_b0_motif_prompt.py
git commit -m "feat(motif): per-row template carrier, attach hooks and stage run-kind guards"
```

---

### Task 18: Stage trainability — the epoch-3 interface opening

**Files:**
- Modify: `src/train_b0.py:668-695` (a sibling of `_set_topo_gen_training_stage`), `:1785-1811`, `:5375-5381`, `:5540-5546`
- Test: `tests/test_train_b0_motif_prompt.py`

**Interfaces:**
- Consumes: Task 12 `V3_1MotifPrompt.interface_open` and `optimizer_parameter_groups`.
- Produces: `_set_motif_prompt_training_stage(model, optimizer, *, epoch) -> None`.

- [ ] **Step 1: Write the failing test**

```python
def test_the_interface_group_is_frozen_for_the_warmup_epochs_and_opens_after() -> None:
    from src.train_b0 import _set_motif_prompt_training_stage

    model = _stage_two_model(interface_warmup_epochs=2)
    optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(1e-4, 1e-5, 1e-2))
    names = [group["name"] for group in optimizer.param_groups]
    assert names == ["generator", "interface"]
    for epoch in (1, 2):
        for group in optimizer.param_groups:
            group["lr"] = 3e-4  # what OneCycleLR would have just written
        _set_motif_prompt_training_stage(model, optimizer, epoch=epoch)
        assert optimizer.param_groups[0]["lr"] == 3e-4
        assert optimizer.param_groups[1]["lr"] == 0.0
        assert model.interface_open is False
    for group in optimizer.param_groups:
        group["lr"] = 3e-4
    _set_motif_prompt_training_stage(model, optimizer, epoch=3)
    assert optimizer.param_groups[1]["lr"] == 3e-4
    assert model.interface_open is True


def test_a_null_warmup_freezes_the_interface_for_every_epoch() -> None:
    from src.train_b0 import _set_motif_prompt_training_stage

    model = _stage_two_model(interface_warmup_epochs=None)
    optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(1e-4, 1e-5, 1e-2))
    for epoch in (1, 3, 15):
        optimizer.param_groups[1]["lr"] = 3e-4
        _set_motif_prompt_training_stage(model, optimizer, epoch=epoch)
        assert optimizer.param_groups[1]["lr"] == 0.0


def test_the_interface_weights_do_not_move_during_the_warmup_epochs() -> None:
    from src.train_b0 import _set_motif_prompt_training_stage

    model = _stage_two_model(interface_warmup_epochs=2)
    optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(1e-4, 1e-5, 1e-2))
    before = model.reader.node_proj.weight.detach().clone()
    generator_before = next(model.generator.parameters()).detach().clone()
    for group in optimizer.param_groups:
        group["lr"] = 1e-3
    _set_motif_prompt_training_stage(model, optimizer, epoch=1)
    batch = _pair_batch(n=4)
    batch["motif_weights"] = _weights(n=4)
    model(batch)["loss"].backward()
    optimizer.step()
    torch.testing.assert_close(model.reader.node_proj.weight, before, rtol=0, atol=0)
    assert not torch.equal(next(model.generator.parameters()), generator_before)


def test_stage_one_leaves_the_reader_open_from_epoch_one() -> None:
    from src.train_b0 import _set_motif_prompt_training_stage

    model = _stage_one_model()
    optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(1e-4, 1e-4, 1e-2))
    optimizer.param_groups[0]["lr"] = 2e-4
    _set_motif_prompt_training_stage(model, optimizer, epoch=1)
    assert model.interface_open is True
    assert optimizer.param_groups[0]["lr"] == 2e-4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_train_b0_motif_prompt.py -n0 -q -k "interface or warmup or stage_one_leaves"`
Expected: FAIL with `ImportError: cannot import name '_set_motif_prompt_training_stage'`

- [ ] **Step 3: Write minimal implementation**

```python
def _set_motif_prompt_training_stage(
    model: nn.Module, optimizer: torch.optim.Optimizer, *, epoch: int
) -> None:
    """Open the motif-prompt interface group after its warm-up epochs (spec §7.5).

    Every parameter that is ever trainable keeps ``requires_grad`` from
    construction, so DDP registers and all-reduces it on every rank; the warm-up
    freeze is the ``interface`` group's learning rate, which AdamW's decoupled
    weight decay also multiplies, making ``lr = 0`` an exact freeze. The gate is
    applied after every scheduler step, because OneCycleLR rewrites
    ``group["lr"]`` on each step.

    Args:
        model: The (possibly DDP-wrapped) model.
        optimizer: The prepared optimizer.
        epoch: The 1-based epoch about to run.
    """
    raw_model = _unwrapped_model(model)
    if not isinstance(raw_model, V3_1MotifPrompt):
        return
    warmup = raw_model.cfg.interface_warmup_epochs
    raw_model.interface_open = (
        True if raw_model.cfg.stage == "one" else warmup is not None and epoch > warmup
    )
    for group in optimizer.param_groups:
        if group.get("name") == "interface" and not raw_model.interface_open:
            group["lr"] = 0.0
```

Call it immediately after every existing `_set_topo_gen_training_stage` call: `:1785`, `:1805` (single-process loop) and `:5375`, `:5540` (DDP loop). The post-`optimizer.step()` call sites are what keep the freeze exact under OneCycle.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_train_b0_motif_prompt.py -n0 -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/train_b0.py tests/test_train_b0_motif_prompt.py
git commit -m "feat(motif): stage trainability via a DDP-safe interface learning-rate gate"
```

---

### Task 19: `optim.stop_after_epoch` and its resume-config exclusion

**Files:**
- Modify: `src/train_b0.py:263-286`, `:819-835`, `:1083-1115`, `:5220-5245`, `:5991-5999`; `src/e2_pipeline.py:664-690`
- Test: `tests/test_train_b0_motif_prompt.py`, `tests/test_e2_pipeline.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `OptimConfig.stop_after_epoch: int | None = None`, validated `1 <= stop_after_epoch <= epochs`; `config_to_dict` drops the key; the DDP loop breaks after the epoch commit; `_validate_staged_artifacts` accepts a short `last.pt`.

- [ ] **Step 1: Write the failing test**

```python
def test_stop_after_epoch_parses_validates_and_leaves_the_config_hash_unchanged() -> None:
    import yaml
    from src.train_b0 import config_to_dict, load_config

    base = yaml.safe_load((Path("configs/split_seed42/b0_v31.yaml")).read_text())
    plain = load_config_from_mapping(base)
    base["optim"]["stop_after_epoch"] = 2
    stopped = load_config_from_mapping(base)
    assert stopped.optim.stop_after_epoch == 2
    assert config_to_dict(stopped) == config_to_dict(plain)
    base["optim"]["stop_after_epoch"] = 0
    with pytest.raises(ValueError, match="optim.stop_after_epoch"):
        load_config_from_mapping(base)
    base["optim"]["stop_after_epoch"] = base["optim"]["epochs"] + 1
    with pytest.raises(ValueError, match="optim.stop_after_epoch"):
        load_config_from_mapping(base)


def test_a_resume_config_comparison_ignores_stop_after_epoch_only() -> None:
    from src.train_b0 import _resume_comparable_config

    saved = {"output_dir": "a", "seed": 0, "optim": {"epochs": 15, "stop_after_epoch": 2, "lr": 1e-4}}
    current = {"output_dir": "b", "seed": 0, "optim": {"epochs": 15, "lr": 1e-4}}
    assert _resume_comparable_config(saved) == _resume_comparable_config(current)
    # The saved mapping is not mutated.
    assert saved["optim"]["stop_after_epoch"] == 2
    diverged = {"output_dir": "b", "seed": 1, "optim": {"epochs": 15, "lr": 1e-4}}
    assert _resume_comparable_config(saved) != _resume_comparable_config(diverged)


def test_schedule_total_steps_ignores_stop_after_epoch() -> None:
    # The pilot keeps optim.epochs at 15 so the first two epochs are a true
    # prefix of the full one-cycle: same trainability mask, same LRs, same order.
    from src.train_b0 import _count_single_process_steps

    cfg = _tiny_cfg(epochs=4)
    stopped = replace(cfg, optim=replace(cfg.optim, stop_after_epoch=2))
    factory = _tiny_loader_factory()
    assert _count_single_process_steps(factory, stopped) == _count_single_process_steps(factory, cfg)
```

and in `tests/test_e2_pipeline.py`:

```python
def test_staged_artifact_validation_accepts_a_short_last_pt_under_stop_after_epoch(
    tmp_path: Path,
) -> None:
    from src.e2_pipeline import _validate_staged_artifacts

    _write_checkpoint(tmp_path / "last.pt", epoch=2, model_family="v3_1_motif_prompt")
    _write_checkpoint(tmp_path / "best.pt", epoch=2, model_family="v3_1_motif_prompt")
    _validate_staged_artifacts(
        tmp_path, epochs=15, model_family="v3_1_motif_prompt", stop_after_epoch=2
    )
    with pytest.raises(ValueError):
        _validate_staged_artifacts(
            tmp_path, epochs=15, model_family="v3_1_motif_prompt", stop_after_epoch=None
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_train_b0_motif_prompt.py -n0 -q -k stop_after && .venv/bin/python -m pytest tests/test_e2_pipeline.py -n0 -q -k stop_after`
Expected: FAIL with `ValueError: unknown optim keys: ['stop_after_epoch']`

- [ ] **Step 3: Write minimal implementation**

`OptimConfig` at `:279-285` gains `stop_after_epoch: int | None = None`, documented as: *"Halt the DDP training loop after this epoch without resizing the schedule. `optim.epochs` still sizes the one-cycle, so the run is an exact prefix of the full one — this is what the §7.4 teachability pilot needs and what `--max-steps` cannot do, because the v3_1 DDP loop ignores that flag."*

Key whitelist at `:821-823` gains `"stop_after_epoch"`. Parse at `:828-835`:

```python
    stop_after_epoch = raw.get("stop_after_epoch")
    if stop_after_epoch is not None:
        stop_after_epoch = _as_int(stop_after_epoch, "optim.stop_after_epoch")
        if not 1 <= stop_after_epoch <= epochs:
            raise ValueError(
                "optim.stop_after_epoch must lie in [1, optim.epochs], got "
                f"{stop_after_epoch} with epochs={epochs}"
            )
```

`config_to_dict` at `:1113`, mirroring the `run_kind` exclusion one line above it:

```python
    # A halt point, not scientific config: it never enters the config hash, so a
    # two-epoch teachability pilot and the full run share one canonical config.
    result["optim"].pop("stop_after_epoch", None)
```

A named helper for the resume comparison, so the exclusion is testable and the saved snapshot is never mutated:

```python
def _resume_comparable_config(config: Mapping[str, object]) -> dict[str, object]:
    """Return the config a resume compares, with the excluded keys dropped.

    ``output_dir`` is an attempt path and ``optim.stop_after_epoch`` is a halt
    point; neither changes the training function, so neither may block a resume
    (spec §7.4).
    """
    comparable = deepcopy(dict(config))
    comparable.pop("output_dir", None)
    optim_block = comparable.get("optim")
    if isinstance(optim_block, dict):
        optim_block.pop("stop_after_epoch", None)
    return comparable
```

and `:5226-5231` becomes:

```python
            saved_resume_config = _resume_comparable_config(saved_config)
            current_resume_config = _resume_comparable_config(config_to_dict(cfg))
            if saved_resume_config != current_resume_config:
                raise ValueError("resume configuration does not match the current training run")
```

The break, as a sibling of the early-stopping block at `:5991-5999`, after the epoch's `training_state.pt` commit so the prefix stays resumable:

```python
        if cfg.optim.stop_after_epoch is not None and epoch >= cfg.optim.stop_after_epoch:
            if accelerator.is_main_process:
                logger.info(
                    "halting after epoch %d of %d (optim.stop_after_epoch)",
                    epoch,
                    cfg.optim.epochs,
                )
            break
```

In `src/e2_pipeline.py`, `_validate_staged_artifacts` gains a `stop_after_epoch: int | None = None` keyword and uses `exact_epoch = stop_after_epoch or epochs` for `last.pt`; the call site at `:1287` passes `stop_after_epoch=cfg.optim.stop_after_epoch`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_train_b0_motif_prompt.py tests/test_e2_pipeline.py tests/test_train_b0.py -n0 -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/train_b0.py src/e2_pipeline.py tests/test_train_b0_motif_prompt.py tests/test_e2_pipeline.py
git commit -m "feat(train): optim.stop_after_epoch with a prefix-preserving resume comparison"
```

---

### Task 20: Fold the two losses into the training step

**Files:**
- Modify: `src/train_b0.py:5474-5534`, `:5670-5700`
- Test: `tests/test_train_b0_motif_prompt.py`

**Interfaces:**
- Consumes: Task 12 `forward` outputs, Task 15 `StructStream.last_motif_rows`, Task 14 `stream_mean`.
- Produces: `epoch_motif_telemetry` entries `train_motif_slot_loss` and `train_motif_topo_loss` in `metrics.jsonl`.

- [ ] **Step 1: Write the failing test**

```python
def test_l_slot_and_l_topo_are_added_once_across_the_two_streams() -> None:
    from src.distill.motif_losses import stream_mean
    from src.train_b0 import _motif_stream_terms

    anchor = torch.zeros((), requires_grad=True)
    task_rows = {"slot": torch.tensor([1.0, 3.0]) + anchor, "topo": torch.tensor([2.0, 4.0]) + anchor}
    task_mask = torch.tensor([1.0, 1.0])
    struct_rows = {"slot": torch.tensor([5.0]) + anchor, "topo": torch.tensor([7.0]) + anchor}
    struct_mask = torch.tensor([1.0])
    slot, topo = _motif_stream_terms(
        task=(task_rows, task_mask), struct=(struct_rows, struct_mask), like=anchor
    )
    # Per-stream means 2.0 and 5.0, then the mean across streams.
    assert float(slot) == pytest.approx(3.5)
    assert float(topo) == pytest.approx(5.0)
    assert slot.requires_grad


def test_an_absent_structural_stream_leaves_the_task_stream_alone() -> None:
    from src.train_b0 import _motif_stream_terms

    anchor = torch.zeros((), requires_grad=True)
    rows = {"slot": torch.tensor([1.0, 3.0]) + anchor, "topo": torch.tensor([2.0, 4.0]) + anchor}
    slot, topo = _motif_stream_terms(
        task=(rows, torch.tensor([1.0, 1.0])), struct=None, like=anchor
    )
    assert float(slot) == pytest.approx(2.0)
    assert float(topo) == pytest.approx(3.0)


def test_epoch_telemetry_reports_both_terms() -> None:
    from src.train_b0 import _motif_epoch_telemetry

    telemetry = _motif_epoch_telemetry(slot_sum=4.0, topo_sum=2.0, steps=2)
    assert telemetry == {"train_motif_slot_loss": 2.0, "train_motif_topo_loss": 1.0}
    assert _motif_epoch_telemetry(slot_sum=0.0, topo_sum=0.0, steps=0) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_train_b0_motif_prompt.py -n0 -q -k "stream_terms or absent_structural or epoch_telemetry"`
Expected: FAIL with `ImportError: cannot import name '_motif_stream_terms'`

- [ ] **Step 3: Write minimal implementation**

```python
def _motif_stream_terms(
    *,
    task: tuple[Mapping[str, torch.Tensor], torch.Tensor],
    struct: tuple[Mapping[str, torch.Tensor], torch.Tensor] | None,
    like: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average ``L_slot`` and ``L_topo`` per stream, then across streams (spec §7.5).

    Args:
        task: The task stream's per-row terms and its valid-nonself mask.
        struct: The structural stream's, or ``None`` when the stream is absent.
        like: A tensor supplying dtype, device and the autograd connection.

    Returns:
        ``(slot, topo)`` scalars; an empty stream contributes a differentiable zero.
    """
    streams = [task] + ([] if struct is None else [struct])
    return (
        stream_mean([rows["slot"] for rows, _ in streams], [mask for _, mask in streams], like=like),
        stream_mean([rows["topo"] for rows, _ in streams], [mask for _, mask in streams], like=like),
    )


def _motif_epoch_telemetry(*, slot_sum: float, topo_sum: float, steps: int) -> dict[str, float]:
    """Return the per-epoch motif telemetry, or an empty mapping for an empty epoch."""
    if steps <= 0:
        return {}
    return {
        "train_motif_slot_loss": slot_sum / float(steps),
        "train_motif_topo_loss": topo_sum / float(steps),
    }
```

In the DDP step (`:5474-5534`), after `struct_loss` is added and before `_all_ranks_loss_finite`, add:

```python
            if isinstance(_unwrapped_model(model), V3_1MotifPrompt):
                raw_model = cast(V3_1MotifPrompt, _unwrapped_model(model))
                struct_rows = (
                    (struct_stream.last_motif_rows, struct_stream.last_motif_mask)
                    if struct_stream is not None and struct_stream.last_motif_rows
                    else None
                )
                slot_term, topo_term = _motif_stream_terms(
                    task=(
                        {"slot": output["slot_loss_rows"], "topo": output["topo_loss_rows"]},
                        batch[TEMPLATE_MASK_KEY],
                    ),
                    struct=struct_rows,
                    like=loss,
                )
                loss = loss + raw_model.cfg.w_slot * slot_term + raw_model.cfg.w_topo * topo_term
                epoch_motif_slot_sum += float(slot_term.detach().float().item())
                epoch_motif_topo_sum += float(topo_term.detach().float().item())
```

The model's `forward` must therefore return `slot_loss_rows` / `topo_loss_rows` **undetached and without the `w_slot`/`w_topo` scaling**, and must *not* fold them into `output["loss"]` when the trainer is doing the folding. Make that unambiguous: `V3_1MotifPrompt.forward` returns `loss` as the weighted task BCE only, plus the two row tensors and the `loss_term_*` summands; the trainer owns the composite. Update the Task 12 test `test_self_row_masking_is_confined_to_the_slot_and_topo_terms` accordingly — it already asserts on `slot_loss_rows` / `topo_loss_rows`, not on `loss`.

At `:5670-5700`, reduce the two epoch sums across ranks exactly as `epoch_struct_loss_sum` is reduced and merge `_motif_epoch_telemetry(...)` into `entry`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_train_b0_motif_prompt.py tests/model/test_motif_prompt_model.py -n0 -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/train_b0.py tests/test_train_b0_motif_prompt.py tests/model/test_motif_prompt_model.py
git commit -m "feat(motif): per-stream composite loss folding and epoch telemetry"
```

---

### Task 21: Truth-free scoring and the scoring-time interventions

**Files:**
- Modify: `src/score_universe.py:1198-1356`, `:3500-3520`, `:3640-3712`, `:3836-3890`
- Test: `tests/test_score_universe_motif_prompt.py`

**Interfaces:**
- Consumes: Task 12 `V3_1MotifPrompt`.
- Produces: `_build_v3_1_motif_prompt(model_config) -> V3_1MotifPrompt` registered in `MODEL_BUILDERS`; `_motif_source_weights(model, pairs, sources) -> torch.Tensor`; `PREFIX_INTERVENTIONS` extended with `shuffle_graph`, `permute_closure`, `rewire_bridge`.

- [ ] **Step 1: Write the failing test**

```python
"""Scoring a v3_1_motif_prompt checkpoint: truth-free, sharded, precision-validated."""

from __future__ import annotations

import numpy as np
import pytest
import torch


def test_the_family_is_registered_and_builds_from_a_checkpoint_config() -> None:
    from src.score_universe import MODEL_BUILDERS, build_model

    assert "v3_1_motif_prompt" in MODEL_BUILDERS
    model = build_model("v3_1_motif_prompt", _stage_two_model_config())
    assert model.name == "v3_1_motif_prompt"


def test_stage_two_scoring_reads_no_graph_and_no_target_file(tmp_path) -> None:
    from src.score_universe import _score_v3_1

    model = _stage_two_model().eval()
    store = _feature_store_stub()
    pairs = [("a", "b"), ("b", "c"), ("a", "a")]
    logits = _score_v3_1(model, pairs, store, device=torch.device("cpu"), amp="no", token_budget=4096)
    assert logits.shape == (3,)
    assert np.isfinite(logits).all()
    # Nothing under the data root was opened: the model holds every input it needs.
    assert not list(tmp_path.iterdir())


def test_stage_one_scoring_requires_the_oracle_diagnostic_acknowledgement() -> None:
    from src.score_universe import _assert_motif_scoring_contract

    with pytest.raises(ValueError, match="allow-oracle-diagnostic"):
        _assert_motif_scoring_contract(stage="one", allow_oracle_diagnostic=False)
    _assert_motif_scoring_contract(stage="one", allow_oracle_diagnostic=True)
    with pytest.raises(ValueError, match="deployable"):
        _assert_motif_scoring_contract(stage="two", allow_oracle_diagnostic=True)


def test_scoring_is_batch_and_shard_independent() -> None:
    from src.score_universe import _score_v3_1

    model = _stage_two_model().eval()
    store = _feature_store_stub()
    pairs = [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")]
    whole = _score_v3_1(model, pairs, store, device=torch.device("cpu"), amp="no", token_budget=4096)
    first = _score_v3_1(model, pairs[:2], store, device=torch.device("cpu"), amp="no", token_budget=4096)
    second = _score_v3_1(model, pairs[2:], store, device=torch.device("cpu"), amp="no", token_budget=4096)
    np.testing.assert_allclose(np.concatenate([first, second]), whole, rtol=1e-6, atol=1e-6)


def test_graph_transplant_uses_one_seeded_universe_permutation() -> None:
    from src.score_universe import _shuffle_source_rows

    rows = _shuffle_source_rows(6, 0)
    assert sorted(rows.tolist()) == list(range(6))
    assert not (rows == np.arange(6)).any()


def test_permute_closure_and_rewire_bridge_preserve_the_family_multisets() -> None:
    from src.score_universe import _motif_permute_closure, _motif_rewire_bridge

    weights = torch.rand(3, 96)
    closure = _motif_permute_closure(weights, seed=0)
    np.testing.assert_allclose(
        np.sort(closure[:, :16].numpy(), axis=1), np.sort(weights[:, :16].numpy(), axis=1), atol=1e-6
    )
    bridge = _motif_rewire_bridge(weights, seed=0)
    np.testing.assert_allclose(
        np.sort(bridge[:, 32:].numpy(), axis=1), np.sort(weights[:, 32:].numpy(), axis=1), atol=1e-6
    )
    # The interventions change the graph, so they must change something.
    assert not torch.allclose(closure, weights)
    assert not torch.allclose(bridge, weights)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_score_universe_motif_prompt.py -n0 -q`
Expected: FAIL with `KeyError: 'v3_1_motif_prompt'`

- [ ] **Step 3: Write minimal implementation**

Add the builder beside `_build_v3_1_coord_gen` (`src/score_universe.py:1198-1222`) and register it in `MODEL_BUILDERS` (`:1318-1327`):

```python
def _build_v3_1_motif_prompt(model_config: Mapping[str, object]) -> nn.Module:
    """Rebuild a ``v3_1_motif_prompt`` checkpoint's model from its embedded config."""
    from src.model.egostitch.classifier.motif_prompt import V3_1MotifPrompt

    return V3_1MotifPrompt(
        base=cast(Mapping[str, object], model_config["base"]),
        motif_prompt=cast(Mapping[str, object], model_config["motif_prompt"]),
    )
```

Extend `PREFIX_INTERVENTIONS` (`:132`) with `"shuffle_graph"`, `"permute_closure"` and `"rewire_bridge"`, and accept a `V3_1MotifPrompt` in the `--prefix-intervention` isinstance check at `:3648-3654`. `gates_off` and `mean` are model-level modes; the three new ones are universe-level substitutions of the *predicted* graph and therefore leave `model.intervention` on `"none"`, exactly as `shuffle` does for `V3_1Prefix`:

```python
def _motif_permute_closure(weights: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Permute the closure attachments of endpoint ``u`` independently (spec §8).

    Preserves the closure weight multiset and both family totals, but not every
    node's weighted degree; the caller reports the changed degree marginals.
    """
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(8, generator=generator)
    out = weights.clone()
    out[:, 0:8] = weights[:, 0:8][:, order]
    return out


def _motif_rewire_bridge(weights: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Rewire the interior block relative to the endpoint attachments (spec §8)."""
    generator = torch.Generator().manual_seed(seed)
    left = torch.randperm(8, generator=generator)
    right = torch.randperm(8, generator=generator)
    out = weights.clone()
    interior = weights[:, 32:].view(-1, 8, 8)
    out[:, 32:] = interior[:, left][:, :, right].reshape(-1, 64)
    return out


def _assert_motif_scoring_contract(*, stage: str, allow_oracle_diagnostic: bool) -> None:
    """Fail closed on a stage/diagnostic-flag mismatch.

    Raises:
        ValueError: On Stage I without the acknowledgement, or Stage II with it.
    """
    if stage == "one" and not allow_oracle_diagnostic:
        raise ValueError(
            "checkpoint family v3_1_motif_prompt stage 'one' reads compiled true templates by "
            "construction; pass --allow-oracle-diagnostic to acknowledge this is a ceiling "
            "diagnostic, never a formal result"
        )
    if stage == "two" and allow_oracle_diagnostic:
        raise ValueError(
            "v3_1_motif_prompt stage 'two' is deployable and scores from (x_u, x_v) alone; "
            "--allow-oracle-diagnostic does not apply"
        )
```

In `main`'s scoring branch (`:3839-3890`), add the family to the `model_family in (...)` tuple. For Stage I, build `row_coords`' analogue — `row_templates` from `MotifTemplateTable(oracle_truth_graph)` — and pass it as `batch['motif_weights']`. For Stage II, pass nothing: the model predicts its own graph. `shuffle_graph` reuses `_shuffle_source_rows(total_rows, seed)` to pick each row's source, then substitutes that source row's **predicted** weights, which requires a two-pass scoring exactly as `_coord_gen_source_coords` already does for `v3_1_coord_gen`; `permute_closure` and `rewire_bridge` transform each row's own predicted weights in place. Record `prefix_intervention` and `prefix_intervention_seed` in `meta_extra` as the existing code already does, and add `"motif_stage"` so a merged artifact is self-describing. Permute globally before shard division, never inside a shard: the 1:1 pair lists are label-sorted, so a shard is label-pure.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_score_universe_motif_prompt.py -n0 -q && .venv/bin/python -m pytest tests/test_score_universe.py -n0 -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/score_universe.py tests/test_score_universe_motif_prompt.py
git commit -m "feat(motif): truth-free scoring and the three graph interventions"
```

---

### Task 22: Extend `validate_score_precision`, then DDP agreement

**Files:**
- Modify: `src/score_universe.py:147`, `:409-457`, `:650-745`, `:3823-3832`
- Create: `tests/helpers/motif_prompt_ddp_smoke.py`, `tests/test_motif_prompt_ddp.py`
- Test: `tests/test_score_universe_motif_prompt.py`

**Interfaces:**
- Consumes: Task 21 scoring path.
- Produces: `_MOTIF_PROMPT_PRECISION_CONTRACT = "v3_1_motif_prompt_pair_fp32_v1"`; `_validate_motif_prompt_precision(logit, *, meta, label, require_diagnostics)`; a two-rank CPU DDP smoke driver.

- [ ] **Step 1: Write the failing test**

```python
def test_a_bf16_contaminated_motif_artifact_is_rejected(tmp_path) -> None:
    from src.score_universe import validate_score_precision

    contaminated = np.asarray(torch.randn(64).to(torch.bfloat16).float().numpy(), dtype=np.float32)
    meta = _motif_meta(contaminated)
    meta["score_precision"]["pair_autocast"] = True
    with pytest.raises(ValueError, match="score_precision"):
        validate_score_precision(contaminated, meta=meta, label="artifact")


def test_a_clean_motif_artifact_validates_through_the_artifact_wrapper() -> None:
    from src.score_universe import ScoresArtifact, validate_artifact_precision

    logit = np.random.default_rng(0).standard_normal(64).astype(np.float32)
    artifact = _motif_artifact(logit)
    validate_artifact_precision(artifact, label="artifact")


def test_missing_provenance_fails_closed_for_this_family() -> None:
    from src.score_universe import validate_score_precision

    logit = np.zeros(8, dtype=np.float32)
    with pytest.raises(ValueError, match="missing score_precision provenance"):
        validate_score_precision(logit, meta={"model_family": "v3_1_motif_prompt"}, label="artifact")


def test_inconsistent_score_resolution_diagnostics_fail_closed() -> None:
    from src.score_universe import validate_score_precision

    logit = np.random.default_rng(1).standard_normal(32).astype(np.float32)
    meta = _motif_meta(logit)
    meta["score_resolution"]["logit"]["distinct"] = 1
    with pytest.raises(ValueError, match="score_resolution"):
        validate_score_precision(logit, meta=meta, label="artifact")
```

and `tests/test_motif_prompt_ddp.py`:

```python
"""Two-rank CPU DDP agreement for the motif-prompt family."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_e2_ddp_integration import REPO_ROOT, _low_thread_env


@pytest.mark.integration
def test_two_ranks_build_identical_templates_and_agree_on_gradients(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "tests/helpers/motif_prompt_ddp_smoke.py",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        env=_low_thread_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    reports = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    # Every rank compiled the same templates and holds the same mean adjacency.
    assert reports[0]["template_digest"] == reports[1]["template_digest"]
    assert reports[0]["mean_digest"] == reports[1]["mean_digest"]
    # After one synchronised step the parameters agree exactly.
    assert reports[0]["param_digest"] == reports[1]["param_digest"]
    # The interface group is frozen in epoch 1 on both ranks.
    assert all(report["interface_lr"] == 0.0 for report in reports)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_score_universe_motif_prompt.py -n0 -q -k precision`
Expected: FAIL — `validate_score_precision` returns early for any family that is not `egostitch_e2e`, so the contaminated artifact validates cleanly and the test's `pytest.raises` fails

- [ ] **Step 3: Write minimal implementation**

```python
_MOTIF_PROMPT_PRECISION_CONTRACT = "v3_1_motif_prompt_pair_fp32_v1"
_MOTIF_PROMPT_ARRAY_KEYS = ("logit",)


def _validate_motif_prompt_precision(
    logit: NDArray[np.float32],
    *,
    meta: Mapping[str, object],
    label: str,
    require_diagnostics: bool,
) -> None:
    """Validate the ``v3_1_motif_prompt`` single-array pair-pass fp32 contract.

    Without this the family would inherit `validate_score_precision`'s early
    return and a bf16-contaminated artifact would analyse cleanly (spec §9).

    Raises:
        ValueError: If the artifact lacks or contradicts the pinned contract, is
            not stored as float32, or its stored diagnostics are inconsistent.
    """
    precision = meta.get("score_precision")
    if not isinstance(precision, dict):
        raise ValueError(
            f"{label}: motif-prompt artifact is missing score_precision provenance; "
            "rescore with the pair-pass fp32 contract"
        )
    expected = {
        "contract": _MOTIF_PROMPT_PRECISION_CONTRACT,
        "pair_compute_dtype": "float32",
        "pair_autocast": False,
        "logit_storage_dtype": "float32",
    }
    mismatches = {
        key: (precision.get(key), value)
        for key, value in expected.items()
        if precision.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{label}: invalid motif-prompt score_precision provenance: {mismatches}")
    if np.asarray(logit).dtype != np.float32:
        raise ValueError(f"{label}: motif-prompt logits must be stored as float32")
    if not require_diagnostics:
        return
    recorded = meta.get("score_resolution")
    if not isinstance(recorded, dict):
        raise ValueError(
            f"{label}: score_resolution diagnostics are missing or not a per-array mapping"
        )
    actual = score_resolution_diagnostics(logit)
    if recorded.get("logit") != actual:
        raise ValueError(
            f"{label}: score_resolution['logit'] diagnostics are missing or inconsistent; "
            f"recorded={recorded.get('logit')!r}, actual={actual!r}"
        )
```

and the dispatch inside `validate_score_precision`, before the existing early return at `:448-449`:

```python
    if family == MOTIF_PROMPT_FAMILY:
        _validate_motif_prompt_precision(
            logit, meta=meta, label=label, require_diagnostics=require_diagnostics
        )
        return
    if family != "egostitch_e2e":
        return
```

`validate_artifact_precision` needs no change: it already forwards `artifact.meta` and an empty `extra_arrays`. The scorer writes the contract at `:3823-3832` for this family:

```python
    if model_family == MOTIF_PROMPT_FAMILY:
        meta_extra["score_precision"] = {
            "contract": _MOTIF_PROMPT_PRECISION_CONTRACT,
            "encode_autocast": args.amp,
            "pair_autocast": False,
            "pair_compute_dtype": "float32",
            "logit_storage_dtype": "float32",
        }
```

and the pair pass for this family runs under `torch.autocast(..., enabled=False)`, which the reader already requires for the RRWP arithmetic.

`tests/helpers/motif_prompt_ddp_smoke.py` follows `tests/helpers/e2_ddp_smoke.py`: set `ACCELERATE_USE_CPU`, build a 12-node training graph and a `MotifTemplateRows` on every rank, build a Stage II `V3_1MotifPrompt` over a tiny base, `accelerator.prepare` it with `model.optimizer_parameter_groups(1e-3, 1e-4, 1e-2)`, call `_set_motif_prompt_training_stage(..., epoch=1)`, run one forward/backward/step on a rank-local batch, then write `{"template_digest", "mean_digest", "param_digest", "interface_lr"}` to `rank-<i>.json`. Digests are `hashlib.sha256` over the concatenated float32 bytes.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_score_universe_motif_prompt.py -n0 -q && .venv/bin/python -m pytest tests/test_motif_prompt_ddp.py -n0 -q -m integration`
Expected: PASS

- [ ] **Step 5: Codex review of the wave, then commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
.venv/bin/python -m pytest -m "not slow and not integration" -q
git add src/score_universe.py tests/test_score_universe_motif_prompt.py \
        tests/helpers/motif_prompt_ddp_smoke.py tests/test_motif_prompt_ddp.py
git commit -m "feat(motif): extend the precision validator and add DDP agreement coverage"
SCRATCH=$(mktemp -d)
CODEX_HOME="$SCRATCH/codex-home" codex review --base "$(git merge-base HEAD main)" \
  > "$SCRATCH/wave-review.txt" 2>&1 &
```

Wait for that background review, read `$SCRATCH/wave-review.txt` (never let it land in context wholesale — it is ~200 KB), fix the blockers it names, and commit the fixes before starting Phase 6. That home needs only `auth.json` and a `config.toml` pinning model/effort; do not pass `-c 'mcp_servers={}'`, which merges rather than overrides and hangs the review with zero output.

---

## Phase 6 — Configs

### Task 23: Stage I and Stage II configs

**Files:**
- Create: `configs/split_seed42/motif_prompt_stage1.yaml`, `configs/split_seed42/motif_prompt_stage2.yaml`, `configs/split_seed42/motif_prompt_stage2_seed1.yaml`, `configs/split_seed42/motif_prompt_stage2_seed2.yaml`
- Test: `tests/test_motif_prompt_configs.py`

**Interfaces:**
- Consumes: Task 16 `load_config`.
- Produces: four loadable configs; `output_dir` equals `outputs/split_seed42/<basename>` in each.

- [ ] **Step 1: Write the failing test**

```python
"""Every motif-prompt config loads, and the arms differ only where the spec says."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from src.train_b0 import load_config

CONFIG_DIR = Path("configs/split_seed42")
ARMS = sorted(str(path.name) for path in CONFIG_DIR.glob("motif_prompt_*.yaml"))


@pytest.mark.parametrize("name", ARMS)
def test_every_motif_config_loads_and_names_its_own_output_dir(name: str) -> None:
    cfg = load_config(CONFIG_DIR / name)
    assert cfg.model.family == "v3_1_motif_prompt"
    assert cfg.output_dir == f"outputs/split_seed42/{Path(name).stem}"
    assert cfg.seed in (0, 1, 2)
    assert cfg.mixed_precision == "bf16"


@pytest.mark.parametrize("name", ARMS)
def test_every_motif_config_runs_the_retained_structural_stream(name: str) -> None:
    cfg = load_config(CONFIG_DIR / name)
    assert cfg.struct is not None
    assert cfg.struct.nodes == 40 and cfg.struct.background_nodes == 8
    assert cfg.struct.mix == {"bfs": 0.5, "motif": 0.25, "bridge": 0.25}
    assert cfg.struct.subgraphs_per_epoch is None
    if "prefix_bce" not in name:
        assert cfg.struct.weights["bce"] == 1.0
        assert cfg.struct.weights["rank"] == 1.0
        assert cfg.struct.weights["degree"] == 0.1
        assert cfg.struct.weights["motif"] == 0.1


@pytest.mark.parametrize("name", ARMS)
def test_every_motif_config_runs_the_full_fifteen_epoch_cycle(name: str) -> None:
    cfg = load_config(CONFIG_DIR / name)
    assert cfg.optim.epochs == 15
    assert cfg.eval.patience is None
    assert cfg.optim.scheduler is not None and cfg.optim.scheduler.type == "onecycle"
    assert cfg.optim.lr == 1e-4 == cfg.optim.scheduler.max_lr
    assert cfg.optim.weight_decay == 1e-2
    assert cfg.optim.grad_clip == 1.0
    assert cfg.optim.groups == {"generator": 1e-4, "interface": 1e-5}


def test_stage_one_and_stage_two_differ_only_in_the_expected_keys() -> None:
    one = yaml.safe_load((CONFIG_DIR / "motif_prompt_stage1.yaml").read_text())
    two = yaml.safe_load((CONFIG_DIR / "motif_prompt_stage2.yaml").read_text())
    for block in ("data", "eval", "runtime", "struct"):
        assert one[block] == two[block]
    diff = {
        key
        for key in set(one["model"]["config"]["motif_prompt"]) | set(two["model"]["config"]["motif_prompt"])
        if one["model"]["config"]["motif_prompt"].get(key)
        != two["model"]["config"]["motif_prompt"].get(key)
    }
    assert diff == {"stage", "bundle_checkpoint"}


def test_the_three_seed_replicates_differ_only_in_seed_and_output_dir() -> None:
    base = yaml.safe_load((CONFIG_DIR / "motif_prompt_stage2.yaml").read_text())
    for index in (1, 2):
        other = yaml.safe_load((CONFIG_DIR / f"motif_prompt_stage2_seed{index}.yaml").read_text())
        assert other.pop("seed") == index
        assert other.pop("output_dir") == f"outputs/split_seed42/motif_prompt_stage2_seed{index}"
        stripped = dict(base)
        stripped.pop("seed")
        stripped.pop("output_dir")
        assert other == stripped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_motif_prompt_configs.py -n0 -q`
Expected: FAIL — `ARMS` is empty, so the parametrised tests collect nothing and `test_stage_one_and_stage_two_differ_only_in_the_expected_keys` fails with `FileNotFoundError`

- [ ] **Step 3: Write minimal implementation**

`configs/split_seed42/motif_prompt_stage1.yaml`:

```yaml
# Stage I of the motif-graph GRIT prompt: teach the reader and the interface on a
# bounded TRUE graph. The prefix_base trunk, its encoder and its head stay frozen;
# the GRIT reader, the count head, the token projections, the role embeddings and
# the prefix adapter/gates train. Compiled templates come from the legal training
# graph minus the queried edge; V_val templates are held-out truth, so this is a
# CEILING DIAGNOSTIC:
#   hpc/run.sh train configs/split_seed42/motif_prompt_stage1.yaml --run-kind diagnostic
# Spec: docs/superpowers/specs/2026-09-16-motif-graph-grit-prompt-design.md
model:
  family: v3_1_motif_prompt
  config:
    motif_prompt:
      stage: one
      base_checkpoint: outputs/split_seed42/prefix_base/best.pt
      width: 128
      slots_per_field: 2
      reader: {layers: 3, dim: 96, heads: 4, rrwp_k: 4}
      corruption: {prob: 0.5, lambda_min: 0.0, lambda_max: 1.0}
      fields: [topo_self, topo_partner, topo_rel, topo_cnt]
      families: [closure, bridge]
      token_source: graph
      gate_mode: learned
      interface_warmup_epochs: 0
      w_slot: 1.0
      w_topo: 0.0   # no teacher exists in Stage I
      beta_p: 1.0
      beta_q: 1.0
      beta_a: 1.0
      beta_i: 1.0
      huber_delta: 1.0
data:
  root: data
  strategy: breadth_first
  negative_ratio: 5
  token_budget: 131072
  batch_pairs: 1024
  num_workers: 0
  f0_cache: outputs/f0_cache/f0_matrix.pt
  expected_missing_features: [node_004764, node_007050]
optim:
  lr: 1.0e-4
  weight_decay: 1.0e-2
  epochs: 15
  warmup_steps: 500   # unused while a onecycle scheduler is configured
  grad_clip: 1.0
  scheduler:
    type: onecycle
    max_lr: 1.0e-4
    pct_start: 0.1
    div_factor: 25
    final_div_factor: 10000
    anneal_strategy: cos
  groups:
    generator: {max_lr: 1.0e-4}
    interface: {max_lr: 1.0e-5}
eval:
  patience: null   # full 15-epoch schedule, no early stopping (spec §7.2)
  eval_every: 1
runtime:
  world_size: auto
  pack_dir: outputs/feature_packs/b0_v31_bf16
  pack_workers: 16
  loader_workers_per_rank: 4
  prefetch_factor: 4
  token_budget: 196608
  max_pairs_per_rank: 1536
  memory_limit_gib: 85.0
  probe_warmup_steps: 10
  probe_timed_steps: 30
seed: 0
output_dir: outputs/split_seed42/motif_prompt_stage1
mixed_precision: bf16
struct:
  nodes: 40
  background_nodes: 8
  mix: {bfs: 0.5, motif: 0.25, bridge: 0.25}
  subgraphs_per_epoch: null
  weights: {bce: 1.0, gs: 0.0, rd: 0.0, deg_mmd: 0.0, rank: 1.0, degree: 0.1, motif: 0.1}
  rank_margin: 0.1
  rank_temperature: 1.0
  huber_delta: 1.0
  mmd_sigma: 1.25
  mmd_bins: 48
  val_subgraphs: 32
  # The only legal memory lever (spec §7.1); raise or lower it from the epoch
  # probe. Neither subgraphs_per_epoch nor max_pairs_per_rank may be traded for
  # wall time: both change the objective or the batching protocol.
  token_budget: 8192
```

`configs/split_seed42/motif_prompt_stage2.yaml` is byte-identical except the header comment, and:

```yaml
model:
  family: v3_1_motif_prompt
  config:
    motif_prompt:
      stage: two
      base_checkpoint: outputs/split_seed42/prefix_base/best.pt
      bundle_checkpoint: outputs/split_seed42/motif_prompt_stage1/best.pt
      ...
      interface_warmup_epochs: 2
      w_slot: 1.0
      w_topo: 0.1
output_dir: outputs/split_seed42/motif_prompt_stage2
```

Its header records that it is the deployable, **formal** arm, launched without `--run-kind`, and that `bundle_checkpoint` must be replaced by the §7.4-selected teachability winner before the wave (Task 30 step 3). The two seed replicates differ only in `seed` and `output_dir`, as the test asserts.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_motif_prompt_configs.py -n0 -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
git add configs/split_seed42/motif_prompt_stage1.yaml \
        configs/split_seed42/motif_prompt_stage2.yaml \
        configs/split_seed42/motif_prompt_stage2_seed1.yaml \
        configs/split_seed42/motif_prompt_stage2_seed2.yaml \
        tests/test_motif_prompt_configs.py
git commit -m "feat(motif): Stage I and Stage II configs with three seed replicates"
```

---

### Task 24: The §8 control configs

**Files:**
- Create: `configs/split_seed42/motif_prompt_count_only.yaml`, `motif_prompt_grit_only.yaml`, `motif_prompt_direct_prefix.yaml`, `motif_prompt_per_type.yaml`, `motif_prompt_mean_graph.yaml`, `motif_prompt_freeze_forever.yaml`, `motif_prompt_closure_only.yaml`, `motif_prompt_bridge_only.yaml`, `motif_prompt_no_slot.yaml`, `motif_prompt_no_topo.yaml`
- Test: `tests/test_motif_prompt_configs.py`

**Interfaces:**
- Consumes: Task 23 Stage II config.
- Produces: ten control configs, each a one-key edit of Stage II.

- [ ] **Step 1: Write the failing test**

```python
_CONTROLS = {
    "motif_prompt_count_only": {"fields": ["topo_cnt"]},
    "motif_prompt_grit_only": {"fields": ["topo_self", "topo_partner", "topo_rel"]},
    "motif_prompt_direct_prefix": {"token_source": "direct"},
    "motif_prompt_per_type": {"gate_mode": "per_type"},
    "motif_prompt_mean_graph": {"gate_mode": "mean_graph"},
    "motif_prompt_freeze_forever": {"interface_warmup_epochs": None},
    "motif_prompt_closure_only": {"families": ["closure"]},
    "motif_prompt_bridge_only": {"families": ["bridge"]},
    "motif_prompt_no_slot": {"w_slot": 0.0},
    "motif_prompt_no_topo": {"w_topo": 0.0},
}


@pytest.mark.parametrize("name,overrides", sorted(_CONTROLS.items()))
def test_each_control_is_a_single_key_edit_of_stage_two(
    name: str, overrides: dict[str, object]
) -> None:
    base = yaml.safe_load((CONFIG_DIR / "motif_prompt_stage2.yaml").read_text())
    control = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text())
    assert control["output_dir"] == f"outputs/split_seed42/{name}"
    expected = {**base["model"]["config"]["motif_prompt"], **overrides}
    assert control["model"]["config"]["motif_prompt"] == expected
    for block in ("data", "optim", "eval", "runtime", "struct", "seed", "mixed_precision"):
        assert control[block] == base[block]


def test_every_control_config_loads() -> None:
    for name in _CONTROLS:
        cfg = load_config(CONFIG_DIR / f"{name}.yaml")
        assert cfg.model.family == "v3_1_motif_prompt"


def test_the_count_only_and_grit_only_arms_are_separately_trained_not_masked() -> None:
    # Zeroing a field's token values leaves its keys in the prefix softmax
    # denominator, and deleting its rows renormalises the others, so neither
    # reproduces a model trained without the field (spec §6).
    count_only = load_config(CONFIG_DIR / "motif_prompt_count_only.yaml")
    grit_only = load_config(CONFIG_DIR / "motif_prompt_grit_only.yaml")
    assert count_only.output_dir != grit_only.output_dir
    assert count_only.optim.epochs == grit_only.optim.epochs == 15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_motif_prompt_configs.py -n0 -q -k control`
Expected: FAIL with `FileNotFoundError: configs/split_seed42/motif_prompt_count_only.yaml`

- [ ] **Step 3: Write minimal implementation**

Copy `motif_prompt_stage2.yaml` ten times, change `output_dir` to match the filename, apply the single `model.config.motif_prompt` override from `_CONTROLS`, and give each a header comment naming the §8 question it answers:

| File | Override | Question (§8) |
|---|---|---|
| `motif_prompt_count_only` | `fields: [topo_cnt]` | Does the reader earn its complexity over the counts? |
| `motif_prompt_grit_only` | `fields: [topo_self, topo_partner, topo_rel]` | Do the counts earn their place next to the reader? |
| `motif_prompt_direct_prefix` | `token_source: direct` | Does a graph bottleneck help beyond a conditional adapter? Matched trainable budget and the same `L_topo`. |
| `motif_prompt_per_type` | `gate_mode: per_type` | Is the gain a per-type density signal? |
| `motif_prompt_mean_graph` | `gate_mode: mean_graph` | Does query-conditioned connectivity matter? |
| `motif_prompt_freeze_forever` | `interface_warmup_epochs: null` | Does adaptation repair the teacher-student input shift? |
| `motif_prompt_closure_only` | `families: [closure]` | Does the closure family earn its place? |
| `motif_prompt_bridge_only` | `families: [bridge]` | Does the bridge family earn its place? |
| `motif_prompt_no_slot` | `w_slot: 0.0` | Which supervision contributes? |
| `motif_prompt_no_topo` | `w_topo: 0.0` | Which supervision contributes? |

Each header also states: *"Read against `motif_prompt_stage2` on the pre-registered panel — BFS-macro GS and RD plus the degree, clustering and spectral MMD ratios at the ONE V_val-selected fixed threshold, with AUROC/AUPRC and Accuracy/F1/MCC beside them. Single-seed differences inside ±0.01 GS and ±0.5 MMD ratio are not read."*

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_motif_prompt_configs.py -n0 -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/split_seed42/motif_prompt_*.yaml tests/test_motif_prompt_configs.py
git commit -m "feat(motif): the ten separately trained section 8 control arms"
```

---

### Task 25: Teachability-pilot configs and the density control

**Files:**
- Create: `configs/split_seed42/motif_prompt_teach_a.yaml`, `motif_prompt_teach_b.yaml`, `motif_prompt_teach_c.yaml`, `src/experiments/motif_density_control.py`
- Test: `tests/test_motif_prompt_configs.py`, `tests/experiments/test_motif_density_control.py`

**Interfaces:**
- Consumes: Task 23 Stage II config; `src.eval.assembly.density_matched_threshold`, `assemble_graph`; `src.eval.graph_metrics.compute_graph_similarity`; `src.eval.fixed_threshold.evaluate_fixed_threshold`.
- Produces: three pilot configs with `optim.stop_after_epoch: 2` and distinct `bundle_checkpoint`s; `density_matched_report(rows, *, union_pairs, target_edges, g_ref, buckets, config) -> dict[str, object]`.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize("suffix", ["a", "b", "c"])
def test_each_teachability_pilot_is_a_true_two_epoch_prefix(suffix: str) -> None:
    cfg = load_config(CONFIG_DIR / f"motif_prompt_teach_{suffix}.yaml")
    # "The first two epochs" is literal: epochs stays 15 so the one-cycle shape,
    # the trainability mask and the data order are the full run's (spec §7.4).
    assert cfg.optim.epochs == 15
    assert cfg.optim.stop_after_epoch == 2
    base = yaml.safe_load((CONFIG_DIR / "motif_prompt_stage2.yaml").read_text())
    pilot = yaml.safe_load((CONFIG_DIR / f"motif_prompt_teach_{suffix}.yaml").read_text())
    differing = {
        key
        for key in set(base["model"]["config"]["motif_prompt"])
        if base["model"]["config"]["motif_prompt"][key]
        != pilot["model"]["config"]["motif_prompt"][key]
    }
    assert differing == {"bundle_checkpoint"}
```

and `tests/experiments/test_motif_density_control.py`:

```python
"""The section 8 output-density control: equal selected count, official self-loops."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest


def test_the_target_is_this_arms_own_selected_pair_count_not_nx_density() -> None:
    from src.experiments.motif_density_control import target_edge_count

    pairs = [("a", "b"), ("b", "c"), ("a", "a"), ("c", "d")]
    logits = np.asarray([2.0, -1.0, 3.0, 0.5])
    count, n_union = target_edge_count(logits, threshold=0.4)
    assert (count, n_union) == (3, 4)


def test_ties_are_admitted_atomically_so_the_realised_count_can_fall_short() -> None:
    from src.eval.assembly import density_matched_threshold

    probs = np.asarray([0.9, 0.9, 0.9, 0.1])
    threshold = density_matched_threshold(probs, target_edges=2)
    assert int((probs >= threshold).sum()) == 0


def test_self_pairs_assemble_as_self_loops_and_official_gs_keeps_them() -> None:
    from src.eval.assembly import assemble_graph
    from src.eval.graph_metrics import compute_graph_similarity

    pairs = [("a", "b"), ("a", "a")]
    graph = assemble_graph(pairs, np.asarray([0.9, 0.9]), threshold=0.5, nodes=["a", "b"])
    assert graph.number_of_edges() == 2
    reference = nx.Graph()
    reference.add_nodes_from(["a", "b"])
    reference.add_edge("a", "b")
    assert compute_graph_similarity(graph, reference) == pytest.approx(2.0 / 3.0)


def test_a_row_more_than_one_percent_short_is_marked_approximate() -> None:
    from src.experiments.motif_density_control import density_matched_report

    report = density_matched_report(
        rows={"arm": np.asarray([0.9, 0.9, 0.9, 0.1])},
        union_pairs=[("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")],
        target_edges=2,
        g_ref=nx.Graph([("a", "b"), ("b", "c")]),
        buckets={4: [{"a", "b", "c", "d"}]},
        config=None,
    )
    assert report["rows"]["arm"]["target_edges"] == 2
    assert report["rows"]["arm"]["realised_edges"] == 0
    assert report["rows"]["arm"]["shape_comparison"] == "approximate"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_motif_prompt_configs.py tests/experiments/test_motif_density_control.py -n0 -q -k "teachability or density or self_pairs or ties"`
Expected: FAIL with `FileNotFoundError: configs/split_seed42/motif_prompt_teach_a.yaml` and `ModuleNotFoundError: No module named 'src.experiments.motif_density_control'`

- [ ] **Step 3: Write minimal implementation**

The three pilot configs are copies of `motif_prompt_stage2.yaml` with `optim.stop_after_epoch: 2`, `output_dir: outputs/split_seed42/motif_prompt_teach_<suffix>` and `bundle_checkpoint` pointing at the five-metric V_val winner and its two neighbours, e.g. `outputs/split_seed42/motif_prompt_stage1/checkpoints/epoch-0011.pt`. Each header states: *"Teachability pilot (spec §7.4). Each candidate's complete bundle — GRIT R, the count head, W_node/W_pair/W_cnt, the role embeddings and the prefix adapter with its gates — initialises both `R_S` and the immutable `R_T`, so the student never starts from a different Stage I checkpoint than its teacher. Rank the three by `geometric_rd_five_rank_v1` on their V_val panel under *predicted* inputs, with fixed probe, data order and seed. Because `optim.epochs` stays 15 this is a true prefix, so the winner is **continued** with `--resume-attempt`, not restarted. If any other key, the world size or the data order differs, restart from the Stage I bundle instead. Two epochs are a budgeted proxy for early transfer only; they do not establish which teacher wins after the epoch-3 unfreezing, and the result note says so."*

`src/experiments/motif_density_control.py`:

```python
"""The section 8 output-density control: read shape at a common assembled density.

This is a property of the assembled graph, not of the inputs, and it is a
different object from the coordinate calibration of spec §0.2. It asks whether
the arm's assembled edge set is a better *shape* at equal density, or whether the
apparent gain was threshold placement. It is reported **alongside**, never
instead of, the protocol's V_val-selected threshold, since re-selecting a
threshold per row is outside ``test_protocol_v8``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from src.eval.assembly import density_matched_threshold
from src.eval.fixed_threshold import evaluate_fixed_threshold
from src.eval.graph_metrics import MMDConfig


def target_edge_count(
    logits: NDArray[np.float64], *, threshold: float
) -> tuple[int, int]:
    """Return ``(target_edges, n_union)`` from this arm's own selected count.

    ``target_edges`` is the number of union pairs **this arm** realises at its
    protocol V_val-selected threshold on the evaluation union — its own assembled
    edge count, taken directly, with no rate transferred between universes. A rate
    may substitute only if it is ``selected_pairs / n_union``; it is **not**
    ``nx.density``, whose denominator counts node pairs over the whole node set
    while a sampled union holds only some of those pairs. Self-pairs are included:
    a self-pair over the threshold assembles as a self-loop and official GS and RD
    retain it.

    Args:
        logits: The arm's raw logits over the union rows.
        threshold: The arm's V_val-selected fixed logit threshold.

    Returns:
        The realised edge count and the union row count.
    """
    selected = int((np.asarray(logits, dtype=np.float64) >= threshold).sum())
    return selected, int(np.asarray(logits).size)


def density_matched_report(
    *,
    rows: Mapping[str, NDArray[np.float64]],
    union_pairs: Sequence[tuple[str, str]],
    target_edges: int,
    g_ref: nx.Graph,
    buckets: dict[int, list[set[str]]],
    config: MMDConfig | None,
) -> dict[str, object]:
    """Score every row at the common target density and report the residual spread.

    One threshold per row, applied unchanged to every subgraph, as
    `evaluate_fixed_threshold` does. Matching the *union* density does not
    equalise density inside each BFS subgraph, so per-subgraph RD still varies and
    a density-matched row's macro RD is not 1: read GS and the three MMD ratios at
    the matched density and treat RD there as a diagnostic of that residual spread.

    Args:
        rows: Raw logits per row name (this arm, ``prefix_base``, ``B0``).
        union_pairs: The pooled union of pairs the protocol scores, self-pairs included.
        target_edges: The common target from `target_edge_count`.
        g_ref: The reference graph.
        buckets: The protocol's BFS subgraph buckets.
        config: The MMD configuration, or ``None`` in unit tests.

    Returns:
        A report with, per row, the chosen threshold, the realised edge count, the
        target, whether the shape comparison is approximate, and the five topology
        numbers when ``config`` is supplied.
    """
    report: dict[str, object] = {
        "matching": "output_density_matched_union",
        "target_edges": int(target_edges),
        "n_union": len(union_pairs),
        "rows": {},
    }
    for name, logits in rows.items():
        probs = expit(np.asarray(logits, dtype=np.float64))
        threshold = density_matched_threshold(probs, target_edges)
        realised = int((probs >= threshold).sum())
        entry: dict[str, object] = {
            "probability_threshold": float(threshold),
            "realised_edges": realised,
            "target_edges": int(target_edges),
            "shape_comparison": (
                "approximate" if realised < target_edges * 0.99 else "matched"
            ),
        }
        if config is not None:
            _, panel = evaluate_fixed_threshold(
                pairs=union_pairs,
                logits=np.asarray(logits, dtype=np.float64),
                g_ref=g_ref,
                buckets=buckets,
                threshold=float(np.log(threshold / (1.0 - threshold))),
                config=config,
            )
            entry["panel"] = panel
        cast_rows = report["rows"]
        assert isinstance(cast_rows, dict)
        cast_rows[name] = entry
    return report
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_motif_prompt_configs.py tests/experiments/test_motif_density_control.py -n0 -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add configs/split_seed42/motif_prompt_teach_*.yaml src/experiments/motif_density_control.py \
        tests/test_motif_prompt_configs.py tests/experiments/test_motif_density_control.py
git commit -m "feat(motif): teachability pilots and the output-density control"
```

---

## Phase 7 — The two bounded pilots (§0.2)

Both pilots are read on **V_val and held-out training rows only**. Test rows are not scored for this decision: choosing an architecture on test evidence is the exact failure this project demoted a whole split for. Any test-side version is run after the architecture is fixed, reported as a labelled diagnostic, and disclosed. A negative on both is a **resource-allocation** decision — we stop investing in endpoint-only structure generation on this evidence and write that negative — and is **not** a proof that no endpoint-only generator can succeed.

### Task 26: Pilot A — the frozen Stage I reader under four coordinate sources

**Files:**
- Create: `src/experiments/motif_pilot_a.py`
- Test: `tests/experiments/test_motif_pilot_a.py`

**Interfaces:**
- Consumes: `src.data.struct_coords.StructCoordinateTable`, `get_coord_spec`; `src.score_universe` scoring entry points.
- Produces: `calibrate_coordinates(predicted, *, truth, spec) -> tuple[NDArray[np.float32], list[str]]`; `knn_coordinates(...)`; `main()` writing `outputs/pilots/motif_pilot_a/report.json`.

- [ ] **Step 1: Write the failing test**

```python
"""Pilot A: coordinate moment calibration and the four V_val coordinate sources."""

from __future__ import annotations

import numpy as np
import pytest
from src.data.struct_coords import get_coord_spec
from src.experiments.motif_pilot_a import calibrate_coordinates


def test_calibration_matches_the_training_mean_and_scale_per_continuous_coordinate() -> None:
    spec = get_coord_spec("v1")
    rng = np.random.default_rng(0)
    truth = rng.normal(3.0, 2.0, size=(512, spec.coord_dim)).astype(np.float32)
    predicted = (truth * 0.25 + 7.0).astype(np.float32)
    calibrated, degenerate = calibrate_coordinates(predicted, truth=truth, spec="v1")
    cont = list(spec.continuous_indices)
    np.testing.assert_allclose(calibrated[:, cont].mean(axis=0), truth[:, cont].mean(axis=0), atol=1e-3)
    np.testing.assert_allclose(calibrated[:, cont].std(axis=0), truth[:, cont].std(axis=0), rtol=1e-3)
    assert degenerate == []


def test_the_distance_indicators_are_excluded_from_calibration() -> None:
    spec = get_coord_spec("v1")
    rng = np.random.default_rng(1)
    truth = rng.normal(size=(64, spec.coord_dim)).astype(np.float32)
    predicted = truth.copy()
    predicted[:, list(spec.distance_indices)] = 0.25
    calibrated, _ = calibrate_coordinates(predicted, truth=truth, spec="v1")
    # An independent affine map per indicator would destroy their simplex semantics.
    np.testing.assert_allclose(
        calibrated[:, list(spec.distance_indices)],
        predicted[:, list(spec.distance_indices)],
        atol=0,
    )


def test_a_zero_variance_coordinate_gets_the_shift_alone_and_is_recorded() -> None:
    spec = get_coord_spec("v1")
    rng = np.random.default_rng(2)
    truth = rng.normal(size=(64, spec.coord_dim)).astype(np.float32)
    predicted = rng.normal(size=(64, spec.coord_dim)).astype(np.float32)
    first = list(spec.continuous_indices)[0]
    predicted[:, first] = 5.0
    calibrated, degenerate = calibrate_coordinates(predicted, truth=truth, spec="v1")
    assert degenerate == [first]
    np.testing.assert_allclose(
        calibrated[:, first], np.full(64, truth[:, first].mean()), atol=1e-4
    )


def test_calibration_changes_inputs_and_never_a_threshold() -> None:
    # Coordinate moment calibration is NOT density matching: it does not match the
    # assembled-graph density the nonlinear reader and its threshold produce, and
    # the pilot must not conflate them (spec §0.2).
    from src.experiments.motif_pilot_a import PILOT_A_SOURCES

    assert PILOT_A_SOURCES == ("truth", "knn50", "training_mean", "knn50_calibrated")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/experiments/test_motif_pilot_a.py -n0 -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.experiments.motif_pilot_a'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Pilot A of the motif-prompt decision (spec §0.2): about one GPU-hour, no model change.

Score the published ``topo_prompt_frozen`` checkpoint through the existing
``row_coords`` path with ``--allow-oracle-diagnostic`` on **V_val** under four
coordinate sources: true coordinates; coordinates composed from kNN-50
attribute-transferred neighbourhoods; the training mean; and kNN-50 after the
coordinate calibration below. That checkpoint is the right instrument because
``gates_off`` reproduces ``prefix_base`` exactly and it holds the best shape
ratios of any row (V_val AUPRC 0.940 / GS 0.648 on truth).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.data.struct_coords import get_coord_spec

PILOT_A_SOURCES = ("truth", "knn50", "training_mean", "knn50_calibrated")
_STD_FLOOR = 1e-6


def calibrate_coordinates(
    predicted: NDArray[np.float32], *, truth: NDArray[np.float32], spec: str = "v1"
) -> tuple[NDArray[np.float32], list[int]]:
    """Rescale each continuous coordinate by one affine map fitted on training rows.

    The map sends each predicted coordinate's training mean and standard deviation
    to the corresponding true coordinate's. Eligible coordinates are the
    **continuous** entries of the reader's spec only; the categorical distance
    indicators are excluded, because an independent affine map per indicator
    destroys their simplex semantics. Where a predicted coordinate has zero
    variance on training rows the shift alone is applied and the coordinate is
    recorded as degenerate.

    This is coordinate moment calibration. It does **not** match the assembled
    graph density that the nonlinear reader and its threshold produce; those are
    different quantities. The threshold is selected afterwards by the usual rule.

    Args:
        predicted: ``(n, coord_dim)`` predicted coordinates on training rows.
        truth: ``(n, coord_dim)`` true coordinates on the same rows.
        spec: The reader's coordinate spec.

    Returns:
        ``(calibrated, degenerate)``: the mapped coordinates and the indices whose
        predicted training variance was zero.

    Raises:
        ValueError: On mismatched shapes.
    """
    if predicted.shape != truth.shape:
        raise ValueError("predicted and truth coordinates must have the same shape")
    layout = get_coord_spec(spec)
    calibrated = np.array(predicted, dtype=np.float32, copy=True)
    degenerate: list[int] = []
    for index in layout.continuous_indices:
        source = predicted[:, index].astype(np.float64)
        target = truth[:, index].astype(np.float64)
        source_std = float(source.std())
        if source_std <= _STD_FLOOR:
            degenerate.append(int(index))
            calibrated[:, index] = np.float32(target.mean())
            continue
        scale = float(target.std()) / source_std
        calibrated[:, index] = ((source - source.mean()) * scale + target.mean()).astype(np.float32)
    return calibrated, degenerate
```

`main()` then, for each of `PILOT_A_SOURCES`, builds the V_val `row_coords` tensor, calls the existing packed scorer against `outputs/split_seed42/topo_prompt_frozen/best.pt` with `--allow-oracle-diagnostic`, selects the closest-geometric-RD threshold on the V_val sampled union, and writes one panel per source — AUROC/AUPRC, Accuracy/F1/MCC at the `val_cls` max-F1 threshold, ECE/Brier on raw probabilities, and BFS-macro GS/RD with the three MMD ratios — into `outputs/pilots/motif_pilot_a/report.json`, together with the `degenerate` list and the `prefix_base` reference row. `knn_coordinates` composes each V_val row's neighbourhood from its 50 nearest training nodes by pooled-feature cosine and measures the coordinates of that composed neighbourhood on the training graph.

**Pre-registered reading, carried verbatim:** the four sources are compared on V_val only. Margins are ±0.01 GS and ±0.5 MMD ratio; single-seed differences inside them are not read. kNN-50's inputs are **not** strictly richer than the generator's — it reads training adjacency; the generator reads residue states, which kNN-50 does not — and feeding its coordinates into a frozen reader confounds generator quality, coordinate calibration and reader distribution shift. kNN-50 is therefore **not** an information ceiling, even among functions of its own features.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/experiments/test_motif_pilot_a.py -n0 -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/experiments/motif_pilot_a.py tests/experiments/test_motif_pilot_a.py
git commit -m "feat(motif): pilot A driver with coordinate moment calibration"
```

---

### Task 27: Pilot B — edge-weight predictability and its feasible baselines

**Files:**
- Create: `src/experiments/motif_pilot_b.py`
- Test: `tests/experiments/test_motif_pilot_b.py`

**Interfaces:**
- Consumes: Task 13 `slot_loss_rows`; Task 4 `MotifTemplateTable`.
- Produces: `PILOT_B_ROWS = ("conditional_head", "symmetric_constant_graph", "asymmetric_fixed_template", "unconstrained_mean_profile")`; `fit_constant_graph(targets, *, symmetric, betas) -> torch.Tensor`; `transplant_control(predicted, targets, *, seed, betas) -> float`; `profile_dispersion(predicted, *, s_family) -> dict[str, float]`.

- [ ] **Step 1: Write the failing test**

```python
"""Pilot B: what L_slot can and cannot establish, and its feasible baselines."""

from __future__ import annotations

import pytest
import torch
from src.data.motif_template import SWAP_PERM, slot_profiles
from src.experiments.motif_pilot_b import (
    PILOT_B_ROWS,
    fit_constant_graph,
    profile_dispersion,
    transplant_control,
)

_BETAS = {"beta_p": 1.0, "beta_q": 1.0, "beta_a": 1.0, "beta_i": 1.0, "huber_delta": 1.0}


def test_the_four_reported_rows_are_the_pre_registered_ones() -> None:
    assert PILOT_B_ROWS == (
        "conditional_head",
        "symmetric_constant_graph",
        "asymmetric_fixed_template",
        "unconstrained_mean_profile",
    )


def test_l_slot_is_exactly_invariant_to_the_endpoint_swap() -> None:
    from src.distill.motif_losses import slot_loss

    gen = torch.Generator().manual_seed(0)
    for _ in range(100):
        predicted = torch.rand(1, 96, generator=gen, dtype=torch.float64)
        target = torch.rand(1, 96, generator=gen, dtype=torch.float64)
        plain = slot_loss(predicted, target, **_BETAS)
        swapped = slot_loss(predicted[:, list(SWAP_PERM)], target[:, list(SWAP_PERM)], **_BETAS)
        assert abs(float(plain) - float(swapped)) < 1e-15
    # Target-informed orientation therefore buys nothing, and no upper bound follows.


def test_the_symmetric_constant_graph_cannot_hold_two_distinct_attachment_values() -> None:
    targets = torch.rand(64, 96)
    symmetric = fit_constant_graph(targets, symmetric=True, betas=_BETAS)
    attachments = slot_profiles(symmetric.unsqueeze(0))["a"][0]
    torch.testing.assert_close(attachments[:8], attachments[8:], rtol=1e-4, atol=1e-4)
    asymmetric = fit_constant_graph(targets, symmetric=False, betas=_BETAS)
    assert not torch.allclose(
        slot_profiles(asymmetric.unsqueeze(0))["a"][0][:8],
        slot_profiles(asymmetric.unsqueeze(0))["a"][0][8:],
        rtol=1e-3,
        atol=1e-3,
    )


def test_the_mean_sorted_profile_is_infeasible_on_the_documented_fixture() -> None:
    # Half empty templates, half one bridge at attachments 1/sqrt(2), interior 1.
    empty = torch.zeros(1, 96)
    bridged = torch.zeros(1, 96)
    bridged[0, 16] = bridged[0, 24] = 2.0**-0.5
    bridged[0, 32] = 1.0
    mean_profile = 0.5 * (
        slot_profiles(empty)["a"].sort(dim=1, descending=True).values
        + slot_profiles(bridged)["a"].sort(dim=1, descending=True).values
    )
    assert float(mean_profile[0, 0]) == pytest.approx(0.3536, abs=1e-4)
    mean_q = 0.5 * (
        slot_profiles(empty)["q"].sort(dim=1, descending=True).values
        + slot_profiles(bridged)["q"].sort(dim=1, descending=True).values
    )
    assert float(mean_q[0, 0]) == pytest.approx(0.25, abs=1e-4)
    # No graph with those attachments and interior weights in [0, 1] exceeds 0.125.
    assert 0.3536**2 == pytest.approx(0.125, abs=1e-4)


def test_the_transplant_control_is_zero_on_the_slot_misalignment_fixture() -> None:
    first = torch.zeros(1, 96)
    first[0, 0] = first[0, 8] = 0.5
    second = torch.zeros(1, 96)
    second[0, 1] = second[0, 9] = 0.5
    predicted = torch.cat([first, second])
    targets = predicted.clone()
    rise = transplant_control(predicted, targets, seed=0, betas=_BETAS)
    assert rise == pytest.approx(0.0, abs=1e-12)


def test_the_transplant_control_rises_when_the_head_uses_row_information() -> None:
    gen = torch.Generator().manual_seed(3)
    targets = torch.rand(32, 96, generator=gen)
    rise = transplant_control(targets.clone(), targets, seed=0, betas=_BETAS)
    assert rise > 0.0


def test_dispersion_is_reported_per_coordinate_in_s_family_units() -> None:
    predicted = torch.rand(16, 96)
    dispersion = profile_dispersion(predicted, s_family={"p": 0.5, "q": 0.25, "a": 1.0, "b": 1.0})
    assert set(dispersion) == {"p", "q", "a", "b"}
    assert all(value >= 0.0 for value in dispersion.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/experiments/test_motif_pilot_b.py -n0 -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.experiments.motif_pilot_b'`

- [ ] **Step 3: Write minimal implementation**

`fit_constant_graph` optimises one row-independent 96-weight vector against `slot_loss` with Adam for 2,000 steps from a `logit(0.5)` start; with `symmetric=True` it ties the two sides (one 8-vector shared by `CLOSURE_U`/`CLOSURE_V`, one by `ATTACH_L`/`ATTACH_R`, and a symmetric interior block), which duplicates every entry of its attachment profile — the concrete thing it lacks. With `symmetric=False` it carries **no** symmetry constraint and is evaluated directly on its invariant profiles: this is the baseline the reported margin is read against. It need not be a legal equivariant generator; `L_slot` cannot tell the difference, which is precisely why it is the right comparator for this loss.

`transplant_control` keeps every predicted graph intact and re-pairs the held-out predictions with targets under one seeded permutation of the whole held-out set, returning `mean(L_slot after) - mean(L_slot before)`. v7's edge-space averaging control is **withdrawn**: it is not invariant to the anonymous-slot permutation §3 randomises, so it can destroy a motif both rows predicted perfectly — two rows carrying the same wedge at weight 0.5 through different closure slots each score zero, yet their edgewise mean holds two half-strength wedges at loss `0.00244140625`. The transplant leaves that fixture at exactly zero, which is the correct reading.

`profile_dispersion` returns, per family, the mean across coordinates of the across-row standard deviation of the sorted profile, divided by that family's fixed `s_family`.

The `beta_i` selector, implemented as `select_beta_i(report) -> tuple[float, dict[str, object]]`, is pre-registered and carried **verbatim**:

- Fit the head twice, at `beta_i = 0` and `beta_i = 1`, on held-out training rows. **Do not divide by a near-zero target.**
- Report two separate quantities per family, both as means over rows: **false mass on zero-target rows** `mean(mass_hat)` over rows with `mass* = 0`, in mass units; and **reconstruction on positive-target rows** `mean|mass_hat - mass*|` over rows with `mass* > 0`, divided by the fixed scalar `s_family = mean(mass*)` over all training rows with `mass* > 0`. That scale is computed once, recorded in the report, and reused unchanged across both fits and every later comparison.
- Adopt `beta_i = 1` if it lowers the **bridge** family's normalised positive-target reconstruction **and** its zero-target false mass satisfies `F_1 - F_0 <= max(0.1 * F_0, delta * s_bridge)` with `delta = 0.01`, fixed here and not revisited once the pilot has run; otherwise keep `beta_i = 0`. The absolute allowance exists because a bare 10% guard collapses to zero tolerance at `F_0 = 0`.
- A difference below 5% of the `beta_i = 0` value — comparing the means defined above, never medians — is a tie and keeps `beta_i = 0` as the simpler objective.
- Report both strata's row counts. If the zero-target stratum is empty the guard is vacuous and is recorded as such; if the positive-target stratum is empty the decision is undefined, `beta_i = 0` stands, and the reason is recorded. An empty stratum never yields a silent mean.
- Per-group gradient norms (closure, attachment, interior heads) at initialisation and at the end are reported as telemetry beside the decision, never as part of it.

`main()` fits the conditional head from frozen **residue** states (pooled-feature kNN discards exactly what residue attention can read, so pilot A alone would understate the design's input), evaluates on held-out training nodes and on V_val, and writes `outputs/pilots/motif_pilot_b/report.json` with the four rows, the margin over the **asymmetric fixed template** (not the raw fit), the dispersion, the transplant rise, and the `beta_i` decision.

**What pilot B can and cannot establish, carried verbatim:** it measures **weight-profile and family-mass predictability**, because that is all §7.3 supervises. It does **not** measure whether motif connectivity transfers: the sorted-product target is invariant to which witness carries which mass, and sorting the binary interior targets reduces them to the number of interior edges. Profile accuracy and downstream utility are reported as separate quantities and neither is allowed to stand in for the other. Beating **one fitted** template is empirical evidence on held-out rows, not a bound over all fixed templates. A head whose loss does not rise under transplant is not using row information, whatever its margin over the fitted template.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/experiments/test_motif_pilot_b.py -n0 -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/experiments/motif_pilot_b.py tests/experiments/test_motif_pilot_b.py
git commit -m "feat(motif): pilot B with feasible baselines, transplant control and the beta_i selector"
```

---

### Task 28: Run both pilots and record the decision

**Files:**
- Create: `docs/results/motif_prompt/pilots.md`
- Modify: `configs/split_seed42/motif_prompt_stage2*.yaml` (only if pilot B selects `beta_i = 0`)

**Interfaces:**
- Consumes: Tasks 26–27 drivers.
- Produces: `outputs/pilots/motif_pilot_{a,b}/report.json` and a written verdict.

- [ ] **Step 1: Run pilot B locally (CPU plus one state-caching pass)**

```bash
.venv/bin/python -m src.experiments.motif_pilot_b \
  --checkpoint outputs/split_seed42/prefix_base/best.pt \
  --pack-dir outputs/feature_packs/b0_v31_bf16 \
  --output outputs/pilots/motif_pilot_b
```

- [ ] **Step 2: Run pilot A on the H20 (about one GPU-hour, no model change)**

```bash
hpc/run.sh score --checkpoint outputs/split_seed42/topo_prompt_frozen/best.pt \
  --pairs val_topology --allow-oracle-diagnostic \
  --output outputs/pilots/motif_pilot_a/scores_truth.npz
.venv/bin/python -m src.experiments.motif_pilot_a \
  --checkpoint outputs/split_seed42/topo_prompt_frozen/best.pt \
  --pack-dir outputs/feature_packs/b0_v31_bf16 \
  --output outputs/pilots/motif_pilot_a
```

- [ ] **Step 3: Write the verdict before reading anything else**

Create `docs/results/motif_prompt/pilots.md` with, in this order: the pre-registered reading (row, metric, margin) stated *before* the numbers; pilot A's four V_val panels plus the `prefix_base` reference and the degenerate-coordinate list; pilot B's four rows with the margin over the asymmetric fixed template, the per-coordinate dispersion in `s_family` units, the transplant rise, both strata's row counts, and the `s_family` scale; the `beta_i` decision with the exact quantities the selector compared; and the resource-allocation verdict. State explicitly that a negative on both stops the investment and is not an impossibility proof, and that the decision used V_val and held-out training rows only.

- [ ] **Step 4: Apply the `beta_i` decision**

If the selector keeps `beta_i = 0`, set `beta_i: 0.0` in every `motif_prompt_*.yaml` and re-run `tests/test_motif_prompt_configs.py`. If pilot B additionally shows the interior family is unpredictable while the closure family is, set `beta_q: 0.0` and `beta_i: 0.0` and report the arm as **closure-supervised** rather than implying more (spec §7.3). Otherwise change nothing.

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m pytest tests/test_motif_prompt_configs.py -n0 -q
git add docs/results/motif_prompt/pilots.md configs/split_seed42/motif_prompt_*.yaml
git commit -m "docs(motif): pilot A and B results and the pre-registered beta_i decision"
```

---

## Phase 8 — Docs and the H20 launch sequence

### Task 29: Documentation and the pre-registered reading

**Files:**
- Create: `docs/results/motif_prompt/README.md`
- Modify: `CLAUDE.md`, `AGENTS.md`, `docs/03-experiments.md`
- Test: `tests/test_path_organization.py`

**Interfaces:**
- Consumes: Tasks 23–28.
- Produces: the arm's result note and the active-method-set entry.

- [ ] **Step 1: Write the failing test**

```python
def test_the_motif_prompt_result_note_exists_and_is_indexed() -> None:
    note = Path("docs/results/motif_prompt/README.md")
    assert note.exists()
    text = note.read_text()
    for required in (
        "geometric_rd_five_rank_v1",
        "test_protocol_v8",
        "self and nonself strata",
        "three seeds",
        "output-density control",
    ):
        assert required in text
    claude = Path("CLAUDE.md").read_text()
    assert "v3_1_motif_prompt" in claude
    assert "2026-09-16-motif-graph-grit-prompt-design.md" in claude
    assert Path("AGENTS.md").read_text().count("v3_1_motif_prompt") >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_path_organization.py -n0 -q -k motif`
Expected: FAIL with `AssertionError` on the missing note

- [ ] **Step 3: Write the documentation**

`docs/results/motif_prompt/README.md` records, before any result:

- **What the arm is.** Family `v3_1_motif_prompt`; Stage I (`motif_prompt_stage1`, diagnostic) teaches the GRIT reader and the prefix interface on compiled true training templates; Stage II (`motif_prompt_stage2`, formal, three seeds) predicts the 96 edge weights from `(x_u, x_v)` alone. The motif graph is an intermediate computation for this edge decision, not a generated interactome.
- **The pre-registered reading, fixed before scoring.** `geometric_rd_five_rank_v1` selection and `test_protocol_v8`; the ONE V_val-selected fixed threshold (closest geometric RD) applied unchanged to every test subgraph; a separate max-F1 threshold frozen on `val_cls` for Accuracy/F1/MCC; ECE/Brier on raw probabilities. Each operating point reports five numbers together: BFS-macro GS and RD plus the degree, clustering and spectral MMD ratios. Self and nonself strata are reported separately, and per-C-class strata where the benchmark's pair classes are available. Single-seed differences inside ±0.01 GS and ±0.5 MMD ratio are not read; these are reporting thresholds, not confidence intervals. The main arm and the density control run three seeds, paid for by dropping the endpoint-only and relation-only readout rows and one motif-family ablation.
- **The comparison table of §8**, one row per control, each naming the question it answers and its config file.
- **The evidence limits of §10, carried forward.** The dataset probes behind this design already inspected test labels and topology, including a test-fitted logistic readout; that complement is excluded from the case for deployment and the exposure is part of this design's provenance, and a result on the same test set must disclose this history. The nearest published mechanism was reproduced on this benchmark and **lost** to B0 (V_val AUPRC 0.730 / GS 0.250 against 0.814 / 0.401). Expect a modest result at best, and treat the structural stream, not the reader, as the component most likely to move the topology panel. Stage I must be trained anew; coordinate-reader checkpoints cannot instantiate this interface.
- **Cost, with every figure labelled.** *Measured in this repo:* `b0_v31` 102 s/epoch at 59.7 GiB/rank; `prefix_base` 316.6 s; struct arms 245–270 s at 81.7 GiB/rank against the 85 GiB cap; the GRIT teacher 722 s at 72.5 GiB. *Derived, not measured:* roughly 1,150–1,800 s/epoch for Stage I and 1,350–2,300 s/epoch for Stage II, about 6.5–11 h per 15-epoch lane, including the retained structural stream. No 26-node GRIT profile exists anywhere in the repo, and the 61.8 GiB figure quoted for a previous Stage II arm has no profile behind it in this checkout and must not be used as a ceiling.

`CLAUDE.md` gains an active-method-set bullet after the Stage II entry:

> - Motif-graph GRIT prompt (`model.family: v3_1_motif_prompt`, spec `docs/superpowers/specs/2026-09-16-motif-graph-grit-prompt-design.md`): a fixed 26-slot, 96-edge motif graph — shared-neighbour wedges and length-three bridges with hub-penalised weights — read by a three-layer GRIT stack and a closed-form count head into four gated KV-prefix fields on the frozen `prefix_base` trunk. Stage I (`motif_prompt_stage1`) compiles true training templates and is a ceiling diagnostic (`--run-kind diagnostic`); Stage II (`motif_prompt_stage2`, three seeds) predicts the graph from `(x_u, x_v)` alone and is deployable and formal. The compiler is `src/data/motif_template.py` (queried edge removed before neighbourhoods, candidates, weights and truncation), the losses `src/distill/motif_losses.py` (`L_slot`, `L_topo`), the model `src/model/egostitch/classifier/motif_prompt.py`. `optim.stop_after_epoch` halts a run without resizing the one-cycle, which is what the §7.4 teachability pilot needs; `--max-steps` cannot, because the v3_1 DDP loop ignores it. Count-only, GRIT-only and every other §8 control are **separately trained arms**, never inference ablations: the prefix branch softmaxes over all rows jointly and its gates are per head, not per field.

Mirror the same bullet into `AGENTS.md`, and add the arm to `docs/03-experiments.md`'s arm table.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_path_organization.py -n0 -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add docs/results/motif_prompt/README.md CLAUDE.md AGENTS.md docs/03-experiments.md \
        tests/test_path_organization.py
git commit -m "docs(motif): result note, active-method-set entry and the pre-registered reading"
```

---

### Task 30: The H20 launch sequence

**Files:**
- Modify: `hpc/README.md`
- Test: `tests/test_hpc_scripts.py`

**Interfaces:**
- Consumes: every earlier task.
- Produces: the runbook section and the exact commands. **This task does not run any GPU work.**

- [ ] **Step 1: Write the failing test**

```python
def test_the_runbook_documents_the_motif_prompt_launch_sequence() -> None:
    text = Path("hpc/README.md").read_text()
    assert "motif_prompt_stage1" in text
    assert "--run-kind diagnostic" in text
    assert "stop_after_epoch" in text
    assert "--ddp-mode epoch-probe" in text
    # The deployable Stage II lane must never carry the diagnostic flag.
    stage_two = [line for line in text.splitlines() if "motif_prompt_stage2.yaml" in line]
    assert stage_two and all("--run-kind diagnostic" not in line for line in stage_two)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hpc_scripts.py -n0 -q -k motif`
Expected: FAIL with `AssertionError`

- [ ] **Step 3: Write the runbook section**

Add to `hpc/README.md` a "Motif-graph GRIT prompt" section holding exactly this sequence. Run it in order; do not start a lane before the previous one publishes.

**0. Sync and verify the checkout.**

```bash
git pull
hpc/run.sh check
```

**1. Profile one epoch before allocating the wave (spec §9).** The binding constraint is memory, not arithmetic: the reader is about 0.08 GFLOP/row against the trunk's ~27, but struct arms already sit about 3 GiB under the 85 GiB cap.

```bash
hpc/run.sh train configs/split_seed42/motif_prompt_stage1.yaml \
  --worker-module src.train_b0 --run-kind diagnostic --ddp-mode epoch-probe
```

Read the peak per-rank memory and the recorded `compile_seconds_per_10k` from `outputs/split_seed42/motif_prompt_stage1/profile.json`. If it OOMs, lower `struct.token_budget` first — it alone leaves the objective intact, because every legal pair of the sampled subgraph is still scored exactly once into the same assembled logit matrix. Only as a declared protocol change may `runtime.max_pairs_per_rank` move toward 1536, and then either with gradient accumulation holding the logical batch and the per-step structural draw fixed, or applied identically to **every** §8 control with that stated in the result note. Never silently trade `struct.subgraphs_per_epoch`, `max_pairs_per_rank` or the structural weights for wall time. Record the caching choice and the measured compile rate in `profile.json`.

**2. Stage I — the ceiling diagnostic.**

```bash
hpc/run.sh train configs/split_seed42/motif_prompt_stage1.yaml \
  --worker-module src.train_b0 --run-kind diagnostic
```

Publishes `outputs/split_seed42/motif_prompt_stage1/best.pt` and the `diagnostic_*` sentinels. The published Stage I reader is the checkpoint chosen by the existing five-metric V_val rank, and it is the row reported as the Stage I diagnostic.

**3. Teachability selection — three two-epoch pilots, not three full runs (spec §7.4).** Point each pilot's `bundle_checkpoint` at one of at most three Stage I candidates: the five-metric winner and its two neighbours.

```bash
for lane in a b c; do
  hpc/run.sh train configs/split_seed42/motif_prompt_teach_${lane}.yaml \
    --worker-module src.train_b0 --skip-test
done
```

Rank the three by `geometric_rd_five_rank_v1` on their V_val panel under *predicted* inputs, with fixed probe, data order and seed. The **immutable teacher `R_T`** is chosen by downstream utility under predicted inputs, never by how reproducible its tokens are: minimising token reconstruction error is optimised by constant or low-variance tokens, and errors in differently scaled representation spaces are not comparable. If time forbids even three pilots, the default is the five-metric winner, reported as "not selected for teachability".

**4. Continue the winner into Stage II.** Because `optim.epochs` stays 15 the two pilot epochs are a true prefix — same trainability mask, same LRs, same data order — so the winner is **continued**, not restarted:

```bash
hpc/run.sh train configs/split_seed42/motif_prompt_stage2.yaml \
  --worker-module src.train_b0 \
  --resume-attempt outputs/split_seed42/motif_prompt_teach_<winner>
```

The resume path restores model, optimiser, scheduler and per-rank RNG state and refuses to resume unless the saved resume config, world size, warmup steps and `schedule_total_steps` all match; `optim.stop_after_epoch` is excluded from that comparison exactly as `output_dir` is. If the pilot config differs from the final Stage II config in any other key, or the world size or data order changes, restart from the Stage I bundle instead:

```bash
hpc/run.sh train configs/split_seed42/motif_prompt_stage2.yaml --worker-module src.train_b0
```

**5. Seed replicates.**

```bash
hpc/run.sh train configs/split_seed42/motif_prompt_stage2_seed1.yaml --worker-module src.train_b0
hpc/run.sh train configs/split_seed42/motif_prompt_stage2_seed2.yaml --worker-module src.train_b0
```

**6. The §8 controls**, each a full formal lane. The two family ablations zero the inactive
family in **both** stages, so each first trains its own family-matched Stage I bundle (a ceiling
diagnostic like `motif_prompt_stage1`) that its Stage II config loads:

```bash
for family in closure bridge; do
  hpc/run.sh train configs/split_seed42/motif_prompt_stage1_${family}_only.yaml \
    --worker-module src.train_b0 --run-kind diagnostic
done
for arm in count_only grit_only direct_prefix per_type mean_graph freeze_forever \
           closure_only bridge_only no_slot no_topo degree_only; do
  hpc/run.sh train configs/split_seed42/motif_prompt_${arm}.yaml --worker-module src.train_b0
done
```

**7. Scoring-time interventions on the selected checkpoint.** No threshold reselection on test interventions.

```bash
for mode in gates_off mean shuffle_graph permute_closure rewire_bridge; do
  hpc/run.sh score --checkpoint outputs/split_seed42/motif_prompt_stage2/best.pt \
    --pairs val_topology --prefix-intervention "${mode}" --prefix-intervention-seed 0 \
    --output outputs/split_seed42/motif_prompt_stage2/intervention_${mode}.npz
done
```

`shuffle_graph` permutes globally before shard division, never inside a shard: the 1:1 pair lists are label-sorted, so a shard is label-pure. These interventions preserve weight multisets and family totals but **not** every node's weighted degree; the scorer records the changed degree marginals per slot role (sorted within-role profiles, sums merged exactly across shards) under `motif_degree_marginals` in the artifact meta. The transplant is requested as `shuffle_graph`; plain `shuffle` is refused on this family so the artifact is never filed under the wrong name.

**8. The output-density control**, reported alongside the protocol's V_val-selected threshold, never instead of it:

```bash
.venv/bin/python -m src.experiments.motif_density_control \
  --arm outputs/split_seed42/motif_prompt_stage2 \
  --reference outputs/split_seed42/prefix_base outputs/split_seed42/b0_v31 \
  --output outputs/split_seed42/motif_prompt_stage2/density_control.json
```

**Reading rules to keep in view.** A better edge metric alone does not establish improved topology. If a density-only or direct-prefix control matches the arm, connectivity has not earned its mechanism. If Stage I gains and Stage II does not, the remaining problem is transfer, not a need for a stronger oracle. Telemetry — gate saturation, effective rank, family usage, per-term per-group gradient norms from epoch 1 — is reported and acted on, never a run eligibility gate. `complete.json` means published, not evaluated: held-out evidence is `test_report.json`/`test_complete.json`, or the `diagnostic_*` pair for Stage I.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_hpc_scripts.py -n0 -q && .venv/bin/python -m pytest -m "not slow and not integration" -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add hpc/README.md tests/test_hpc_scripts.py
git commit -m "docs(motif): H20 launch sequence for the motif-graph GRIT prompt wave"
```

---

## Required-checks coverage (§9)

Every check the spec lists, and the task that owns it:

| §9 required check | Task | Test |
|---|---|---|
| Queried-edge deletion before weighting and truncation | 3 | `test_queried_edge_is_removed_before_neighbourhoods_and_weights`, `test_endpoint_degrees_are_decremented_by_the_removed_edge` |
| No V_val or boundary targets | 6 | `test_no_compiled_target_ever_names_a_v_val_or_boundary_node` |
| Compiler swap equivariance | 3 | `test_swapping_endpoints_permutes_the_weight_vector_by_swap_perm` |
| Generator swap equivariance | 10 | `test_generator_is_equivariant_under_swapping_the_endpoints` |
| Within-role permutation invariance of the reader | 9 | `test_reader_is_invariant_to_a_within_role_slot_permutation` |
| Within-role permutation invariance of `L_slot` | 13 | `test_loss_is_invariant_to_a_within_role_permutation_of_either_argument` |
| Finite outputs and gradients at zero and tiny weights | 7, 8, 13 | `test_count_head_runs_in_fp32_...`, `test_rrwp_is_finite_at_a_zero_adjacency_...`, `test_gradients_are_finite_at_zero_and_at_tiny_weights` |
| Connected finite gradients reaching all 96 edges on non-degenerate fixtures, interior included, from `L_slot` and from the task path | 13, 10 | `test_gradients_reach_all_96_edges_on_a_non_degenerate_fixture`, `test_gradients_reach_every_generator_parameter_on_a_non_degenerate_batch` |
| The wedge-mass fixture | 13 | `test_wedge_mass_mismatch_is_penalised_although_the_edge_multisets_agree` |
| The analytic interior-derivative fixture (1.266e-7 → 1.266e-3) | 13 | `test_the_analytic_interior_derivative_rises_by_exactly_1e4_with_beta_i` |
| Active final GRIT edge parameters | 9 | `test_final_layer_edge_parameters_receive_gradient` |
| Exact gates-off base identity | 12 | `test_zero_gates_reproduce_the_frozen_base_bit_for_bit`, `test_gates_off_intervention_reproduces_the_base_with_open_gates` |
| Per-field masking renormalises correctly | 11 | `test_per_field_masking_removes_rows_rather_than_zeroing_values`, `test_masking_a_field_renormalises_the_branch_over_the_remaining_rows` |
| Identical clean and predicted graph layouts | 8, 12 | `test_dense_adjacency_is_symmetric_...` (one scatter for both stages), `test_stage_one_requires_the_template_and_stage_two_never_reads_one` |
| Self-row masking confined to `L_slot`/`L_topo` | 12, 17 | `test_self_row_masking_is_confined_to_the_slot_and_topo_terms`, `test_motif_rows_attach_the_template_and_mask_by_row_id` |
| Checkpoint round-trip | 12 | `test_checkpoint_round_trip_preserves_logits_and_the_mean_template` |
| Scoring with no graph or target files | 21 | `test_stage_two_scoring_reads_no_graph_and_no_target_file` |
| Batch- and shard-independent outputs | 9, 21 | `test_reader_predictions_are_batch_independent`, `test_scoring_is_batch_and_shard_independent` |
| DDP initialisation and gradient agreement | 22 | `test_two_ranks_build_identical_templates_and_agree_on_gradients` |
| Precision validator extended to this family | 22 | the four `tests/test_score_universe_motif_prompt.py` precision tests |

Head-gradient **norm** ratios are telemetry only and are never a pass condition: they scale with head parameterisation, parameter count and residual size, so a correctly fitted attachment head can carry almost no gradient while a badly fitted interior head correctly carries a large one.

---

## Open questions

Recorded rather than silently designed, as the spec instructs. None of them blocks Tasks 1–23 or 25–30.

1. **The degree-matched control of §8 has no stated algorithm.** The comparison table marks it `trained` and asks "Is the gain a degree artefact?", but unlike the output-density control it is not defined anywhere in the spec. The two readings that fit the table's shape are (a) a separately trained arm whose prompt carries only degree information — `fields: [topo_cnt]` with the count head restricted to its two degree entries, i.e. a strict sub-control of `motif_prompt_count_only`; and (b) a scoring-time re-thresholding that matches the assembled degree distribution rather than the edge count, parallel to the density control. These answer different questions and are not interchangeable. **Task 24 therefore ships the other ten controls and leaves this one config unwritten**; the implementer asks the owner which reading is intended, then adds `configs/split_seed42/motif_prompt_degree_matched.yaml` as a one-key edit of Stage II (reading a) or a second driver beside `motif_density_control.py` (reading b). Everything else in Phase 6 proceeds.
2. **The spec file ends mid-sentence.** `docs/superpowers/specs/2026-09-16-motif-graph-grit-prompt-design.md` stops at "…coordinate-reader checkpoints cannot instantiate this interface." with no terminating punctuation, and §0.3's table plus §0.1's header both promise a change log in **§11 that the file does not contain**. Nothing in §1–§10 is missing, so the plan is complete against the design; but the provenance record the change log was to carry is absent. Ask the owner whether the tail was truncated in writing before citing §11 anywhere.
3. **`L_topo` on the structural stream needs a tuple-returning checkpointed forward.** §7.5 says the term is "averaged over valid nonself rows per stream and then across streams", which requires the structural stream's inner forward to return its per-row `L_slot`/`L_topo` beside the logits. Task 15 does this by widening the `checkpoint(forward, ...)` closure's return to a three-tuple for this family only; every other family still reads element 0 through the unchanged `parts` list. This is a real change to a shared code path and is the single highest-risk edit in the plan — the Codex review at the end of Task 22 should be pointed at it explicitly.

---

## Self-review

**Spec coverage.** §0.2 → Tasks 26–28. §1 → Task 12 (inference is `(x_u,x_v)` only; Task 21 proves it at scoring time). §2 → Tasks 1, 3, 4. §3 → Tasks 3, 4, 6, 12 (`_corrupt`), 17 (`mean`). §4 → Task 10. §5.1 → Task 7. §5.2 → Tasks 8, 9. §6 → Tasks 11, 12. §7.1 → Tasks 15, 23 (struct block), 30 (the levers). §7.2 → Tasks 17, 23. §7.3 → Tasks 5, 13. §7.4 → Tasks 19, 25, 30. §7.5 → Tasks 12, 14, 18, 20. §8 → Tasks 21, 24, 25, 29. §9 → the coverage table above, plus Tasks 16–22 for the implementation path and Task 30 for the cost/profiling step. §10 → Task 29. The one gap is the degree-matched control, recorded as Open question 1.

**Placeholder scan.** No "TBD", no "similar to Task N", no "add error handling"; every code step carries the actual code or, where it is a mechanical copy of an existing block (the trunk drive in Task 12, the control configs in Task 24), an explicit enumeration of exactly what changes and what it is copied from.

**Type consistency.** `slot_profiles` returns `{"p","q","a","b"}` in Task 5 and is consumed under those keys in Tasks 7, 13 and 27. `TEMPLATE_KEY`/`TEMPLATE_MASK_KEY` are defined in Task 2 and used in Tasks 12, 15, 17, 20. `slot_loss_rows`/`topo_loss_rows`/`stream_mean` keep one signature across Tasks 13, 14, 15, 20. `interface_open` and `optimizer_parameter_groups` are defined in Task 12 and driven in Task 18. `MOTIF_PROMPT_FAMILY` is defined in Task 16 and used in Tasks 21 and 22. `MotifTemplateTable.weights_by_index` is defined in Task 4 and called in Tasks 15 and 17.

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-16-motif-graph-grit-prompt.md`. Two execution options:

1. **Subagent-Driven (recommended)** — a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

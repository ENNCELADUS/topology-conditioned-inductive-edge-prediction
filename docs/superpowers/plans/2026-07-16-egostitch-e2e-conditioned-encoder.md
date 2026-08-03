# EgoStitch E2E Stitched-Topology-Conditioned Pair Encoder — Implementation Plan

> **2026-08-03 data-contract disposition:** every 80/20 message/supervision split in
> this historical plan is superseded by spec §9.3. Topology and classification now use
> the same complete train-side positive interactions.

> **2026-07-19 disposition:** this plan records the v1 implementation landing. Its
> `L_edge`-inactive warm-start and v1 config/registration commands are superseded for
> the prospective v2 screen by normative spec §13.19 and
> `docs/registrations/g5_e2e_stage1_preregistration_v2.json`. Do not execute those v1
> training steps as a v2 formal run.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the e2e EgoStitch redesign — a from-scratch V3.1 pair encoder
conditioned on a structure-only stitched-topology encoder (STE) via zero-init
tanh-gated cross-attention, with the three-null head taxonomy — per the approved
design `docs/superpowers/specs/2026-07-16-egostitch-e2e-conditioned-encoder-design.md`
(rev 3, commit `e1807b7`).

**Architecture:** New additive modules under `src/model/egostitch/` (conditioning
primitives, scaffold builder, STE, conditioned trunk subclass, e2e model) compose
the existing V3.1 components (`SiameseEncoder`, `PairCrossAttention`,
`CrossAttentionLayer`, `MLPHead` from `src/model/B0.py`, which is **never
modified**) with the existing Stage-1 Imagine/Stitch modules. Worker, scoring,
and gate integration follow. Docs/registration landing is a gated phase.

**Tech Stack:** Python 3.11 / PyTorch (pinned via `uv.lock`), pytest, mypy
strict, ruff. Tests mirror `src/` under `tests/`.

## Global Constraints

- **GOVERNANCE GATE (Phase 0): LIFTED 2026-07-17.** The registered frozen-s0
  Stage-1 screen published a binding `cut` verdict; see
  `docs/results/G5-stage1-seed0-20260717.md`. Phase 0 may now perform the explicit
  spec/protocol/config rewrite. Per the freeze rule ("edit the spec first, then the
  code"), Phases 1–5 still execute only **after** Phase 0, and no successor formal
  run may start before its fresh registration and unresolved defaults are bound.
- All work on branch `g5/e2e-encoder`, branched from `main`.
- `src/model/B0.py` is the audited B0 family: **do not modify it** (provenance
  audit, protocol §E5). All trunk changes live in `src/model/egostitch/`.
- Commands: `uv run pytest ...`, `uv run mypy src tests`, `uv run ruff check .`.
  Locally prefer `.venv/bin/python -m ...` over `uv run python -m ...` (rtk proxy
  garbles `uv run` output). Never run two mypy invocations concurrently
  (`.mypy_cache` corruption → phantom `unused-ignore`).
- mypy is strict with `warn_unused_ignores = true`; all new code fully typed.
- fp32 pair-pass pin (spec §13.16 extension): the trunk pair pass, STE, gates,
  and head compute with autocast disabled in scoring paths.
- Design pins (rev 3, verbatim): 4-type anchor labels (NO grounded-identity-match
  in `c_topo`); queries = `cls_token` only; inject **after** the final
  `N_inj ∈ {1,2}` pair-cross-attention blocks (default 1); AB/BA share STE+XAttn
  parameters; branch masks per pair, shared across AB/BA; three nulls
  `∅_all_head` / `∅_topo_head` / `∅_content_head`; train = per-sample
  multiplicative masks, eval = batch-level hard bypass, equality asserted by test.
- Defaults registered by this plan: `ste_layers=3`, `ste_dim=128`,
  `xattn_heads=8`, `n_inj=1` (sweep `{1,2}`), `p_topo=p_cont=0.15`
  (sweep 0.1–0.2, plus `p=0` arms).

---

## Phase 0 — Landing gate: docs, config, registration (UNBLOCKED 2026-07-17)

### Task 1: Spec edits (docs/05-egostitch-spec.md)

**Files:**
- Modify: `docs/05-egostitch-spec.md` (§5, §8, §13.1, §13.10, §13.16, §13.17; new §13.18)

**Interfaces:**
- Consumes: design note §§3–6 (source text to port).
- Produces: the normative spec sections Tasks 5–15 implement against.

- [ ] **Step 1: Port the head replacement into §5.** Replace the §5 decision-head
  block (`s0 = pair_logit(i, j) ... p_ij = σ(s0 + g_θ(s1..s4)·w)`) with the
  design note §3 content: scaffold objects (4-type anchor labels), `c_topo` /
  `c_content` token split, STE definition, cls_token gated cross-attention with
  the §3.4 pin list, and the §3.5 three-null table (train-mask vs eval-bypass
  semantics included). State that the former `s4` scalar channel is absorbed by
  the STE and `s1` moves to the content pathway.
- [ ] **Step 2: Update §8 curriculum** with one sentence: "Trunk, STE, and gates
  train exactly when `L_edge` is active (not during the `L_recon`-only
  warm-start)."
- [ ] **Step 3: Rewrite §13.1 Stage-1 head** to the e2e form (trunk + STE +
  gated cross-attention over `(c_topo, c_content)`; no `s0`), and mark §13.10
  **retired** (frozen-B0 logit cache no longer exists for the e2e family).
- [ ] **Step 4: Extend §13.16** — the fp32 pair-pass scope now covers the trunk
  pair pass, STE, gates, and head; artifact contract string becomes
  `egostitch_e2e_pair_fp32_v1`.
- [ ] **Step 5: Re-register §13.17** — liveness signals reference the
  within-checkpoint `f_logit` (`∅_all_head`); the fresh-frozen-s0 comparator
  scoring step is removed; add gate-magnitude (`tanh(g)` per pathway per block)
  and per-branch RMS gradient telemetry; death-signal thresholds re-stated
  against the new reference.
- [ ] **Step 6: Add §13.18 (E2E pins)** — the §3.4 pin list, null taxonomy,
  defaults (`ste_layers=3`, `ste_dim=128`, `xattn_heads=8`, `n_inj=1` sweep
  `{1,2}`, `p_topo=p_cont=0.15` sweep 0.1–0.2), Stage-1 arm scope (full, B0-e2e,
  pair+topology, 6a shuffle, `p=0`), and the representation-probe protocol
  (degree / ego density / clustering + degree-partialled + alignment consistency,
  frozen-encoder linear probes on held-out message-partition nodes).
- [ ] **Step 7: Append change-log lines** (one per edit, dated 2026-07-XX at
  landing), e.g.: `- 2026-07-XX: §5/§13.1 replaced the frozen-s0 anchored head
  with the stitched-topology-conditioned pair encoder (design note 2026-07-16
  rev 3); §13.10 retired; §13.16 scope extended; §13.17 re-registered against
  within-checkpoint f_logit; §13.18 added (E2E pins).`
- [ ] **Step 8: Commit** — `git commit -m "docs(spec): e2e conditioned-encoder head, null taxonomy, §13.18 pins"`

### Task 2: Protocol disposition (docs/03-experiment-protocol.md)

**Files:**
- Modify: `docs/03-experiment-protocol.md` (§0 component table, §2 baseline table, §3 E4 list)

- [ ] **Step 1:** Add a dated disposition under §0 (same pattern as the
  2026-07-09 disposition): the frozen pairwise scorer loses the `s0`-anchor role;
  retains B0-baseline and E4.10-proposer roles; the method under test is the
  stitched-topology-conditioned pair encoder.
- [ ] **Step 2:** Add baseline rows: `B0-e2e` (matched-training f-only trunk,
  `∅_all_head` permanent) with the config-mismatch note (canonical B0:
  `train_plus`/1:1/lr 1e-4/seed 47 vs Ours regime: `e_sup`/1:5/lr 3e-4/seed 0);
  E2E instantiations noted on the B3-full and B5 rows.
- [ ] **Step 3:** Add the structure-specificity battery to §3 E4 (design §5
  arm 6 a–f, including 6e degree-preserving rewiring and 6f capacity-matched
  no-message-passing bottleneck).
- [ ] **Step 4: Commit** — `git commit -m "docs(protocol): §0 disposition — s0 anchor retired; B0-e2e row; E4 structure battery"`

### Task 3: Proposal rev (docs/04-model-proposal.md)

**Files:**
- Modify: `docs/04-model-proposal.md` (§4.4, §4.6 mapping, §8 references)

- [ ] **Step 1:** Rewrite §4.4 from anchored late fusion to the conditioned
  encoder (port design §3); rescope the SHOT citation to the frozen-s0 ablation
  arm; add the conditioning-depth ladder.
- [ ] **Step 2:** Add the novelty-scoping text from design §10 — contribution
  claimed as **novel overall composition** only ("no exact match identified in
  the reviewed corpus", never "not anticipated"/"first to generate structural
  context for unseen nodes"); per-component ancestry/prior-usage/difference
  table; name-and-distinguish Leap (2503.03331) and CAM tokens (2405.19375).
- [ ] **Step 3:** Verify and add the new §8 references (repo rule: re-verify
  each arXiv ID against the abstract page before quoting): 2503.03331,
  2405.19375, 2111.05366, 2301.12721, 2402.05862, 2310.13023, 2204.14198,
  1906.12192, 2408.04053, 2110.02096, 2402.09711, 2306.10453, 2405.14985,
  2310.04612.
- [ ] **Step 4: Commit** — `git commit -m "docs(proposal): §4.4 conditioned-encoder rev + composition novelty scoping"`

### Task 4: Config + replacement Stage-1 registration

**Files:**
- Create: `configs/egostitch_e2e_breadth_first.yaml`
- Create: `docs/registrations/g5_e2e_stage1_preregistration.json`
- Create: `docs/registrations/g5_e2e_stage1_preregistration.md`

- [ ] **Step 1:** Write the config by copying
  `configs/egostitch_stage1_breadth_first.yaml` and applying: `model.family:
  egostitch_e2e`; **delete** `data.s0_cache` and `data.s0_checkpoint_id`; add
  under `model.config`: `ste_layers: 3`, `ste_dim: 128`, `xattn_heads: 8`,
  `n_inj: 1`, `p_topo: 0.15`, `p_cont: 0.15`; add `data.pack_dir:
  outputs/feature_packs/egostitch_e2e_tokens` (raw-token pack, Task 12). Keep
  `train_positives: e_sup`, `negative_ratio: 5`, `partition_seed: 0`, epochs 30.
- [ ] **Step 2:** Write the registration (JSON + md) pinning: fixed Seed 0; the
  five Stage-1 arms (full, B0-e2e/f-only, pair+topology `∅_content_head`
  permanent, structure control 6a within-pair `Â`/`Π` shuffle, branch dropout
  `p=0`); the four-logit decomposition report; the probe protocol; liveness
  signals vs within-checkpoint `f_logit`; fidelity/cost report (params, FLOPs,
  GPU-hours, scoring latency, measured H20 re-estimate REQUIRED before binding);
  and the pathway-attribution decision rule. **Proposed defaults requiring
  explicit user confirmation before binding:** pair+topology arm retains ≥ 25%
  of the full-model gain over B0-e2e on the registered assembled decision
  metric; full model exceeds the 6a control by more than the registered
  resolution tolerance (0.005, matching the existing matched-RD tolerance).
- [ ] **Step 3: Commit** — `git commit -m "feat(g5): e2e config + replacement Stage-1 registration"`

---

## Phase 1 — Conditioning primitives

### Task 5: Branch masks (`HeadNullMasks`, `sample_branch_masks`)

**Files:**
- Create: `src/model/egostitch/conditioning.py`
- Test: `tests/model/test_egostitch_conditioning.py`

**Interfaces:**
- Produces: `HeadNullMasks(NamedTuple)` with `topo: torch.Tensor` and
  `cont: torch.Tensor` (both shape `(B,)`, dtype bool, `True` = pathway ACTIVE);
  `sample_branch_masks(batch_size: int, p_topo: float, p_cont: float, *,
  generator: torch.Generator, device: torch.device) -> HeadNullMasks`;
  module-level constants `NULL_ALL_HEAD`, `NULL_TOPO_HEAD`, `NULL_CONTENT_HEAD`,
  `NULL_NONE` (str) and `masks_for_null(null: str, batch_size: int, device:
  torch.device) -> HeadNullMasks`.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for e2e head-null branch masks (design rev 3 §3.5/§4)."""
import pytest
import torch

from src.model.egostitch.conditioning import (
    NULL_ALL_HEAD,
    NULL_CONTENT_HEAD,
    NULL_NONE,
    NULL_TOPO_HEAD,
    HeadNullMasks,
    masks_for_null,
    sample_branch_masks,
)


def test_sample_branch_masks_shapes_and_dtype() -> None:
    gen = torch.Generator().manual_seed(0)
    masks = sample_branch_masks(64, 0.15, 0.15, generator=gen, device=torch.device("cpu"))
    assert isinstance(masks, HeadNullMasks)
    assert masks.topo.shape == (64,) and masks.topo.dtype == torch.bool
    assert masks.cont.shape == (64,) and masks.cont.dtype == torch.bool


def test_sample_branch_masks_p_zero_all_active() -> None:
    gen = torch.Generator().manual_seed(0)
    masks = sample_branch_masks(32, 0.0, 0.0, generator=gen, device=torch.device("cpu"))
    assert bool(masks.topo.all()) and bool(masks.cont.all())


def test_sample_branch_masks_deterministic_given_seed() -> None:
    a = sample_branch_masks(
        128, 0.5, 0.5, generator=torch.Generator().manual_seed(7), device=torch.device("cpu")
    )
    b = sample_branch_masks(
        128, 0.5, 0.5, generator=torch.Generator().manual_seed(7), device=torch.device("cpu")
    )
    assert torch.equal(a.topo, b.topo) and torch.equal(a.cont, b.cont)


@pytest.mark.parametrize(
    ("null", "topo_on", "cont_on"),
    [
        (NULL_NONE, True, True),
        (NULL_ALL_HEAD, False, False),
        (NULL_TOPO_HEAD, False, True),
        (NULL_CONTENT_HEAD, True, False),
    ],
)
def test_masks_for_null(null: str, topo_on: bool, cont_on: bool) -> None:
    masks = masks_for_null(null, 4, torch.device("cpu"))
    assert bool(masks.topo.all()) is topo_on and bool((~masks.topo).all()) is not topo_on
    assert bool(masks.cont.all()) is cont_on and bool((~masks.cont).all()) is not cont_on


def test_masks_for_null_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown head-null"):
        masks_for_null("nope", 4, torch.device("cpu"))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/model/test_egostitch_conditioning.py -v`
Expected: FAIL — `ModuleNotFoundError`/`ImportError` for `src.model.egostitch.conditioning`.

- [ ] **Step 3: Implement**

```python
"""E2E head conditioning primitives: branch masks and gated cross-attention.

Design contract: docs/superpowers/specs/2026-07-16-egostitch-e2e-conditioned-
encoder-design.md rev 3 — three mutually exclusive head nulls; training uses
per-sample multiplicative masks, evaluation uses batch-level hard bypasses.
Mask semantics: True = pathway ACTIVE for that pair (shared across AB/BA).
"""

from __future__ import annotations

from typing import NamedTuple

import torch

NULL_NONE = "none"
NULL_ALL_HEAD = "all_head"
NULL_TOPO_HEAD = "topo_head"
NULL_CONTENT_HEAD = "content_head"
_KNOWN_NULLS = (NULL_NONE, NULL_ALL_HEAD, NULL_TOPO_HEAD, NULL_CONTENT_HEAD)


class HeadNullMasks(NamedTuple):
    """Per-pair pathway activity masks (True = active)."""

    topo: torch.Tensor
    cont: torch.Tensor


def sample_branch_masks(
    batch_size: int,
    p_topo: float,
    p_cont: float,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> HeadNullMasks:
    """Sample independent per-pair branch-dropout masks (design §4)."""
    topo = torch.rand(batch_size, generator=generator) >= p_topo
    cont = torch.rand(batch_size, generator=generator) >= p_cont
    return HeadNullMasks(topo=topo.to(device), cont=cont.to(device))


def masks_for_null(null: str, batch_size: int, device: torch.device) -> HeadNullMasks:
    """Deterministic masks realizing one of the §3.5 null conditions."""
    if null not in _KNOWN_NULLS:
        raise ValueError(f"unknown head-null condition: {null!r}")
    on = torch.ones(batch_size, dtype=torch.bool, device=device)
    off = torch.zeros(batch_size, dtype=torch.bool, device=device)
    topo = off if null in (NULL_ALL_HEAD, NULL_TOPO_HEAD) else on
    cont = off if null in (NULL_ALL_HEAD, NULL_CONTENT_HEAD) else on
    return HeadNullMasks(topo=topo, cont=cont)
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/model/test_egostitch_conditioning.py -v` → PASS.
- [ ] **Step 5: Lint/type** — `uv run ruff check . && uv run mypy src tests` → clean.
- [ ] **Step 6: Commit** — `git commit -m "feat(g5-e2e): head-null branch masks"`

### Task 6: `GatedCrossAttention` (zero-init tanh gate)

**Files:**
- Modify: `src/model/egostitch/conditioning.py` (append)
- Test: `tests/model/test_egostitch_conditioning.py` (append)

**Interfaces:**
- Produces: `GatedCrossAttention(nn.Module)` — `__init__(d_model: int, n_heads:
  int, dropout: float)`; `forward(cls: torch.Tensor, tokens: torch.Tensor,
  token_mask: torch.Tensor | None, active: torch.Tensor) -> torch.Tensor` where
  `cls` is `(B, 1, d_model)`, `tokens` `(B, T, d_tok=d_model)`, `token_mask`
  `(B, T)` bool (True = valid), `active` `(B,)` bool. Returns updated cls,
  identical to input where gate is 0 or `active` is False.

- [ ] **Step 1: Write the failing tests (append)**

```python
from src.model.egostitch.conditioning import GatedCrossAttention


def test_gated_xattn_identity_at_init() -> None:
    torch.manual_seed(0)
    m = GatedCrossAttention(d_model=32, n_heads=4, dropout=0.0)
    cls = torch.randn(3, 1, 32)
    tokens = torch.randn(3, 7, 32)
    active = torch.ones(3, dtype=torch.bool)
    out = m(cls, tokens, None, active)
    assert torch.equal(out, cls)  # gate zero-init => exact identity


def test_gated_xattn_per_sample_mask_equals_bypass() -> None:
    torch.manual_seed(0)
    m = GatedCrossAttention(d_model=32, n_heads=4, dropout=0.0)
    with torch.no_grad():
        m.gate.fill_(0.7)  # open the gate so the pathway is live
    m.eval()
    cls = torch.randn(5, 1, 32)
    tokens = torch.randn(5, 9, 32)
    active = torch.tensor([True, False, True, False, True])
    out = m(cls, tokens, None, active)
    # masked samples: exact identity (== hard bypass)
    assert torch.equal(out[~active], cls[~active])
    # active samples: actually conditioned
    assert not torch.allclose(out[active], cls[active])


def test_gated_xattn_respects_token_mask() -> None:
    torch.manual_seed(0)
    m = GatedCrossAttention(d_model=32, n_heads=4, dropout=0.0)
    with torch.no_grad():
        m.gate.fill_(0.7)
    m.eval()
    cls = torch.randn(2, 1, 32)
    tokens = torch.randn(2, 6, 32)
    active = torch.ones(2, dtype=torch.bool)
    mask_all = torch.ones(2, 6, dtype=torch.bool)
    out_full = m(cls, tokens, mask_all, active)
    # zero out the padded tail AND mask it: masked tokens must not matter
    tokens2 = tokens.clone()
    tokens2[:, 3:, :] = 999.0
    mask_head = mask_all.clone()
    mask_head[:, 3:] = False
    out_head_a = m(cls, tokens, mask_head, active)
    out_head_b = m(cls, tokens2, mask_head, active)
    assert torch.allclose(out_head_a, out_head_b, atol=1e-6)
    assert not torch.allclose(out_full, out_head_a)
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/model/test_egostitch_conditioning.py -v` → FAIL (`GatedCrossAttention` not defined).
- [ ] **Step 3: Implement (append to conditioning.py)**

```python
from torch import nn


class GatedCrossAttention(nn.Module):
    """Zero-init tanh-gated cross-attention residual sublayer (design §3.4).

    ``cls <- cls + active * tanh(gate) * XAttn(LN(cls), tokens)``. The gate is a
    scalar parameter initialized to zero, so at init (and whenever ``active`` is
    False) the sublayer is an exact identity — the checkpoint-exact bypass
    property the §3.5 null taxonomy relies on.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        cls: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor | None,
        active: torch.Tensor,
    ) -> torch.Tensor:
        key_padding_mask = None if token_mask is None else ~token_mask
        attn_out, _ = self.attn(
            self.norm_q(cls),
            self.norm_kv(tokens),
            self.norm_kv(tokens),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        scale = active.to(cls.dtype).view(-1, 1, 1)
        return cls + scale * torch.tanh(self.gate) * attn_out
```

- [ ] **Step 4: Run tests** — PASS. **Step 5:** ruff + mypy clean.
- [ ] **Step 6: Commit** — `git commit -m "feat(g5-e2e): zero-init gated cross-attention sublayer"`

---

## Phase 2 — Scaffold, STE, content tokens

### Task 7: Structure-only scaffold builder

**Files:**
- Create: `src/model/egostitch/scaffold.py`
- Test: `tests/model/test_egostitch_scaffold.py`

**Interfaces:**
- Consumes: `SlotSet` (`src/model/egostitch/imagine.py:26` — fields
  `h (B,K,d_p), pi (B,K), mult (B,K), gate (B,K), pointer (B,K,n_g),
  adj (B,K,K)`), alignment plan `Π (B,K,K)` from
  `src.model.egostitch.stitch.sinkhorn_plan`.
- Produces: `ScaffoldTokens(NamedTuple)` with `feats: torch.Tensor`
  `(B, V, 9)` where `V = 2 + 2K`, node order
  `[endpoint_src, endpoint_dst, slots_src(K), slots_dst(K)]`, feature layout
  `[onehot4(anchor); pi; mult; deg_star; deg_intra; deg_align]`; and
  `adj: torch.Tensor` `(B, 3, V, V)` (edge types `[star, intra, align]`,
  symmetric, non-negative); `build_scaffold(slots_src: SlotSet, slots_dst:
  SlotSet, plan: torch.Tensor) -> ScaffoldTokens`;
  `swap_direction(tokens: ScaffoldTokens) -> ScaffoldTokens` (relabels
  src↔dst without recomputation). `N_ANCHOR_TYPES = 4`, `FEAT_DIM = 9`,
  `EDGE_TYPES = 3` module constants.

**Anchor-label channels (design §3.1/§3.2, binding):** 0 = endpoint-src,
1 = endpoint-dst, 2 = slot-of-src, 3 = slot-of-dst. **No `h`, no `gate`, no
grounded-identity-match anywhere in this module.**

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the structure-only stitched scaffold (design rev 3 §3.1–§3.2)."""
import torch

from src.model.egostitch.imagine import SlotSet
from src.model.egostitch.scaffold import (
    EDGE_TYPES,
    FEAT_DIM,
    N_ANCHOR_TYPES,
    ScaffoldTokens,
    build_scaffold,
    swap_direction,
)


def _slots(b: int = 2, k: int = 4, d_p: int = 8, seed: int = 0) -> SlotSet:
    g = torch.Generator().manual_seed(seed)
    adj = torch.rand(b, k, k, generator=g)
    adj = 0.5 * (adj + adj.transpose(1, 2))
    return SlotSet(
        h=torch.randn(b, k, d_p, generator=g),
        pi=torch.rand(b, k, generator=g),
        mult=1.0 + torch.rand(b, k, generator=g),
        gate=torch.rand(b, k, generator=g),
        pointer=torch.rand(b, k, 3, generator=g),
        adj=adj,
    )


def test_scaffold_shapes_and_layout() -> None:
    si, sj = _slots(seed=0), _slots(seed=1)
    plan = torch.rand(2, 4, 4)
    out = build_scaffold(si, sj, plan)
    assert isinstance(out, ScaffoldTokens)
    v = 2 + 2 * 4
    assert out.feats.shape == (2, v, FEAT_DIM)
    assert out.adj.shape == (2, EDGE_TYPES, v, v)
    # anchor one-hots: exactly one label per node, correct blocks
    onehot = out.feats[..., :N_ANCHOR_TYPES]
    assert torch.equal(onehot.sum(-1), torch.ones(2, v))
    assert bool(onehot[:, 0, 0].all()) and bool(onehot[:, 1, 1].all())
    assert bool(onehot[:, 2 : 2 + 4, 2].all()) and bool(onehot[:, 2 + 4 :, 3].all())


def test_scaffold_contains_no_content_features() -> None:
    # identical structure, different content (h/gate/pointer) => identical scaffold
    si_a, sj = _slots(seed=0), _slots(seed=1)
    si_b = si_a._replace(
        h=torch.randn_like(si_a.h),
        gate=torch.rand_like(si_a.gate),
        pointer=torch.rand_like(si_a.pointer),
    )
    plan = torch.rand(2, 4, 4)
    a, b = build_scaffold(si_a, sj, plan), build_scaffold(si_b, sj, plan)
    assert torch.equal(a.feats, b.feats) and torch.equal(a.adj, b.adj)


def test_scaffold_adj_symmetric_and_star_weights() -> None:
    si, sj = _slots(seed=0), _slots(seed=1)
    plan = torch.rand(2, 4, 4)
    out = build_scaffold(si, sj, plan)
    assert torch.allclose(out.adj, out.adj.transpose(2, 3), atol=1e-6)
    # star edge endpoint_src -> its slot k carries pi*mult
    expected = si.pi * si.mult
    assert torch.allclose(out.adj[:, 0, 0, 2 : 2 + 4], expected, atol=1e-6)


def test_swap_direction_is_involution_and_relabels() -> None:
    si, sj = _slots(seed=0), _slots(seed=1)
    plan = torch.rand(2, 4, 4)
    fwd = build_scaffold(si, sj, plan)
    rev = swap_direction(fwd)
    # anchor channels swapped: src<->dst, slot-of-src<->slot-of-dst
    assert torch.equal(rev.feats[..., 0], fwd.feats[..., 1])
    assert torch.equal(rev.feats[..., 2], fwd.feats[..., 3])
    # non-label features and structure unchanged
    assert torch.equal(rev.feats[..., 4:], fwd.feats[..., 4:])
    assert torch.equal(rev.adj, fwd.adj)
    back = swap_direction(rev)
    assert torch.equal(back.feats, fwd.feats) and torch.equal(back.adj, fwd.adj)
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/model/test_egostitch_scaffold.py -v` → FAIL (module missing).
- [ ] **Step 3: Implement**

```python
"""Structure-only stitched scaffold (design rev 3 §3.1–§3.2).

Node order: [endpoint_src, endpoint_dst, slots_src(K), slots_dst(K)].
Node features (FEAT_DIM=9): [onehot4(anchor); pi; mult; deg_star; deg_intra;
deg_align]. Edge types (EDGE_TYPES=3): star / intra-side slot-slot / alignment.
Deliberately EXCLUDED: slot content h, grounding gate g, pointer, and the
grounded-identity-match label — those belong to the content pathway.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from src.model.egostitch.imagine import SlotSet

N_ANCHOR_TYPES = 4
FEAT_DIM = 9
EDGE_TYPES = 3
_STAR, _INTRA, _ALIGN = 0, 1, 2
_SRC, _DST, _SLOT_SRC, _SLOT_DST = 0, 1, 2, 3


class ScaffoldTokens(NamedTuple):
    """Batched structure-only scaffold tensors."""

    feats: torch.Tensor
    adj: torch.Tensor


def build_scaffold(
    slots_src: SlotSet, slots_dst: SlotSet, plan: torch.Tensor
) -> ScaffoldTokens:
    """Assemble the stitched scaffold from two slot sets and the OT plan."""
    b, k = slots_src.pi.shape
    v = 2 + 2 * k
    device, dtype = slots_src.pi.device, slots_src.pi.dtype

    adj = torch.zeros(b, EDGE_TYPES, v, v, device=device, dtype=dtype)
    s_src = slice(2, 2 + k)
    s_dst = slice(2 + k, v)

    star_src = slots_src.pi * slots_src.mult
    star_dst = slots_dst.pi * slots_dst.mult
    adj[:, _STAR, 0, s_src] = star_src
    adj[:, _STAR, s_src, 0] = star_src
    adj[:, _STAR, 1, s_dst] = star_dst
    adj[:, _STAR, s_dst, 1] = star_dst

    intra_src = slots_src.adj * slots_src.pi[:, :, None] * slots_src.pi[:, None, :]
    intra_dst = slots_dst.adj * slots_dst.pi[:, :, None] * slots_dst.pi[:, None, :]
    adj[:, _INTRA, s_src, s_src] = intra_src
    adj[:, _INTRA, s_dst, s_dst] = intra_dst

    adj[:, _ALIGN, s_src, s_dst] = plan
    adj[:, _ALIGN, s_dst, s_src] = plan.transpose(1, 2)

    feats = torch.zeros(b, v, FEAT_DIM, device=device, dtype=dtype)
    feats[:, 0, _SRC] = 1.0
    feats[:, 1, _DST] = 1.0
    feats[:, s_src, _SLOT_SRC] = 1.0
    feats[:, s_dst, _SLOT_DST] = 1.0
    ones = torch.ones(b, 1, device=device, dtype=dtype)
    feats[:, :, 4] = torch.cat([ones, ones, slots_src.pi, slots_dst.pi], dim=1)
    feats[:, :, 5] = torch.cat([ones, ones, slots_src.mult, slots_dst.mult], dim=1)
    feats[:, :, 6:9] = adj.sum(dim=-1).permute(0, 2, 1)
    return ScaffoldTokens(feats=feats, adj=adj)


def swap_direction(tokens: ScaffoldTokens) -> ScaffoldTokens:
    """Relabel src<->dst anchor channels for the BA stream (structure fixed)."""
    perm = [_DST, _SRC, _SLOT_DST, _SLOT_SRC]
    feats = tokens.feats.clone()
    feats[..., :N_ANCHOR_TYPES] = tokens.feats[..., perm]
    return ScaffoldTokens(feats=feats, adj=tokens.adj)
```

- [ ] **Step 4: Run tests** — PASS. **Step 5:** ruff + mypy clean.
- [ ] **Step 6: Commit** — `git commit -m "feat(g5-e2e): structure-only stitched scaffold builder"`

### Task 8: Stitched-topology encoder (`STEncoder`)

**Files:**
- Create: `src/model/egostitch/ste.py`
- Test: `tests/model/test_egostitch_ste.py`

**Interfaces:**
- Consumes: `ScaffoldTokens` (Task 7).
- Produces: `STEncoder(nn.Module)` — `__init__(d_model: int, ste_dim: int = 128,
  n_layers: int = 3)`; `forward(scaffold: ScaffoldTokens) -> torch.Tensor`
  returning topology tokens `(B, V, d_model)` (internal width `ste_dim`, final
  linear projection to the trunk `d_model`).

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the stitched-topology encoder (design rev 3 §3.3)."""
import torch

from src.model.egostitch.scaffold import ScaffoldTokens, build_scaffold
from src.model.egostitch.ste import STEncoder
from tests.model.test_egostitch_scaffold import _slots


def _scaffold(seed_i: int = 0, seed_j: int = 1) -> ScaffoldTokens:
    torch.manual_seed(42)
    return build_scaffold(_slots(seed=seed_i), _slots(seed=seed_j), torch.rand(2, 4, 4))


def test_ste_output_shape() -> None:
    torch.manual_seed(0)
    enc = STEncoder(d_model=64, ste_dim=32, n_layers=2)
    out = enc(_scaffold())
    assert out.shape == (2, 10, 64)


def test_ste_permutation_equivariance() -> None:
    """Permuting scaffold nodes permutes STE tokens identically."""
    torch.manual_seed(0)
    enc = STEncoder(d_model=64, ste_dim=32, n_layers=2).eval()
    sc = _scaffold()
    perm = torch.randperm(sc.feats.size(1))
    sc_p = ScaffoldTokens(
        feats=sc.feats[:, perm], adj=sc.adj[:, :, perm][:, :, :, perm]
    )
    out, out_p = enc(sc), enc(sc_p)
    assert torch.allclose(out[:, perm], out_p, atol=1e-5)


def test_ste_is_structure_sensitive() -> None:
    """Rewiring adjacency (same tokens) must change the output."""
    torch.manual_seed(0)
    enc = STEncoder(d_model=64, ste_dim=32, n_layers=2).eval()
    sc = _scaffold()
    rewired = ScaffoldTokens(feats=sc.feats, adj=sc.adj.flip(dims=[2]))
    assert not torch.allclose(enc(sc), enc(rewired), atol=1e-4)
```

- [ ] **Step 2: Run to verify failure** — FAIL (module missing).
- [ ] **Step 3: Implement**

```python
"""Module: stitched-topology encoder — edge-weighted MP over the scaffold.

The promoted s4 lineage (spec §5): per layer, one weight matrix per edge type,
messages aggregated by the weighted adjacency, residual + LayerNorm + MLP.
Token-level output (no pooled readout) — the learned topology representation.
"""

from __future__ import annotations

import torch
from torch import nn

from src.model.egostitch.scaffold import EDGE_TYPES, FEAT_DIM, ScaffoldTokens


class _MPLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.msg = nn.ModuleList(nn.Linear(dim, dim) for _ in range(EDGE_TYPES))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        agg = torch.zeros_like(h)
        for t, lin in enumerate(self.msg):
            agg = agg + torch.bmm(adj[:, t], lin(h))
        h = self.norm1(h + agg)
        return self.norm2(h + self.mlp(h))


class STEncoder(nn.Module):
    """Structure-only scaffold encoder producing token-level topology states."""

    def __init__(self, d_model: int, ste_dim: int = 128, n_layers: int = 3) -> None:
        super().__init__()
        self.embed = nn.Linear(FEAT_DIM, ste_dim)
        self.layers = nn.ModuleList(_MPLayer(ste_dim) for _ in range(n_layers))
        self.out = nn.Linear(ste_dim, d_model)

    def forward(self, scaffold: ScaffoldTokens) -> torch.Tensor:
        # normalize adjacency rows so hub scaffolds do not blow up activations
        deg = scaffold.adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        adj = scaffold.adj / deg
        h = self.embed(scaffold.feats)
        for layer in self.layers:
            h = layer(h, adj)
        return self.out(h)
```

- [ ] **Step 4: Run tests** — PASS. **Step 5:** ruff + mypy clean.
- [ ] **Step 6: Commit** — `git commit -m "feat(g5-e2e): stitched-topology encoder (token-level, structure-only)"`

### Task 9: Content tokens (`c_content`)

**Files:**
- Modify: `src/model/egostitch/scaffold.py` (append)
- Test: `tests/model/test_egostitch_scaffold.py` (append)

**Interfaces:**
- Produces: `build_content_tokens(slots_src: SlotSet, slots_dst: SlotSet,
  matched_src: torch.Tensor, matched_dst: torch.Tensor) -> torch.Tensor` of
  shape `(B, 2K, d_p + 3)` — per slot `[h; pi; gate; grounded_identity_match]`,
  src slots first. `matched_*` are `(B, K)` float in `[0,1]` (the
  grounded-identity-match signal, moved here from the anchor labels per rev 3);
  a `ContentProjector(nn.Module)` — `__init__(d_p: int, d_model: int)`,
  `forward(tokens: torch.Tensor) -> torch.Tensor` `(B, 2K, d_model)`.

- [ ] **Step 1: Write failing tests (append):** shape test `(2, 8, 8+3)`;
  content-token change when `h` changes (opposite of Task 7's invariance test);
  `ContentProjector` output shape `(2, 8, 64)`.

```python
from src.model.egostitch.scaffold import ContentProjector, build_content_tokens


def test_content_tokens_shape_and_content_sensitivity() -> None:
    si, sj = _slots(seed=0), _slots(seed=1)
    matched = torch.zeros(2, 4)
    out = build_content_tokens(si, sj, matched, matched)
    assert out.shape == (2, 8, 8 + 3)
    si_b = si._replace(h=torch.randn_like(si.h))
    out_b = build_content_tokens(si_b, sj, matched, matched)
    assert not torch.equal(out, out_b)


def test_content_projector_shape() -> None:
    torch.manual_seed(0)
    proj = ContentProjector(d_p=8, d_model=64)
    si, sj = _slots(seed=0), _slots(seed=1)
    tokens = build_content_tokens(si, sj, torch.zeros(2, 4), torch.zeros(2, 4))
    assert proj(tokens).shape == (2, 8, 64)
```

- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement (append to scaffold.py)**

```python
from torch import nn


def build_content_tokens(
    slots_src: SlotSet,
    slots_dst: SlotSet,
    matched_src: torch.Tensor,
    matched_dst: torch.Tensor,
) -> torch.Tensor:
    """Content-pathway tokens: [h; pi; gate; grounded-identity-match]."""

    def side(s: SlotSet, matched: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [s.h, s.pi[..., None], s.gate[..., None], matched[..., None]], dim=-1
        )

    return torch.cat([side(slots_src, matched_src), side(slots_dst, matched_dst)], dim=1)


class ContentProjector(nn.Module):
    """Linear projection of content tokens into the trunk width."""

    def __init__(self, d_p: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_p + 3, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(tokens)
```

- [ ] **Step 4: Run tests** — PASS. **Step 5:** ruff + mypy. **Step 6: Commit** —
  `git commit -m "feat(g5-e2e): content-pathway tokens with grounded-identity-match"`

---

## Phase 3 — Conditioned trunk and full model

### Task 10: `ConditionedPairCrossAttention`

**Files:**
- Create: `src/model/egostitch/trunk.py`
- Test: `tests/model/test_egostitch_trunk.py`

**Interfaces:**
- Consumes: `PairCrossAttention` (`src/model/B0.py:788` — untouched; its
  `forward(h_a, h_b, lengths_a, lengths_b)` builds masks via
  `_build_padding_mask`, loops `self.layers` updating `(h_a, h_b, cls_token)`,
  then dispatches on `pair_readout_mode`), `GatedCrossAttention` (Task 6).
- Produces: `ConditionedPairCrossAttention(PairCrossAttention)` — extra ctor
  kwargs `n_inj: int = 1`, `xattn_heads: int = 8`, `xattn_dropout: float = 0.0`;
  `forward(h_a, h_b, lengths_a, lengths_b, *, topo_tokens: torch.Tensor | None
  = None, cont_tokens: torch.Tensor | None = None, topo_active: torch.Tensor |
  None = None, cont_active: torch.Tensor | None = None) -> torch.Tensor`.
  `None` tokens/active = hard bypass (exact parent behavior). Sublayers:
  `self.topo_xattn`, `self.cont_xattn` (`nn.ModuleList`, length `n_inj`),
  applied to `cls_token` **after** each of the final `n_inj` layers.

- [ ] **Step 1: Read the parent forward once** (`src/model/B0.py:874-920`) to
  confirm the loop shape before overriding (no code change to B0.py).
- [ ] **Step 2: Write the failing tests**

```python
"""Tests: conditioned trunk equals audited PairCrossAttention under bypass."""
import torch

from src.model.B0 import PairCrossAttention
from src.model.egostitch.trunk import ConditionedPairCrossAttention

_KW = dict(
    d_model=32, n_heads=4, n_layers=3, dropout=0.0,
    pair_readout_mode="pair_context_gated", mixing_mode="bidirectional_cross",
)


def _inputs(b: int = 3, la: int = 5, lb: int = 7) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(0)
    return (
        torch.randn(b, la, 32, generator=g),
        torch.randn(b, lb, 32, generator=g),
        torch.full((b,), la, dtype=torch.long),
        torch.full((b,), lb, dtype=torch.long),
    )


def test_bypass_matches_parent_exactly() -> None:
    torch.manual_seed(1)
    cond = ConditionedPairCrossAttention(n_inj=1, xattn_heads=4, **_KW)
    parent = PairCrossAttention(**_KW)
    # load the shared submodule weights parent<-cond (cond has extra sublayers)
    parent.load_state_dict(
        {k: v for k, v in cond.state_dict().items() if k in parent.state_dict()}
    )
    cond.eval(), parent.eval()
    h_a, h_b, la, lb = _inputs()
    out_cond = cond(h_a, h_b, la, lb)  # no tokens => hard bypass
    out_parent = parent(h_a, h_b, la, lb)
    assert torch.equal(out_cond, out_parent)


def test_zero_gate_is_identity_even_when_active() -> None:
    torch.manual_seed(1)
    cond = ConditionedPairCrossAttention(n_inj=2, xattn_heads=4, **_KW).eval()
    h_a, h_b, la, lb = _inputs()
    topo = torch.randn(3, 10, 32)
    active = torch.ones(3, dtype=torch.bool)
    out_off = cond(h_a, h_b, la, lb)
    out_on = cond(h_a, h_b, la, lb, topo_tokens=topo, topo_active=active)
    assert torch.equal(out_on, out_off)  # gates zero-init


def test_open_gate_conditions_output() -> None:
    torch.manual_seed(1)
    cond = ConditionedPairCrossAttention(n_inj=1, xattn_heads=4, **_KW).eval()
    with torch.no_grad():
        cond.topo_xattn[0].gate.fill_(0.5)
    h_a, h_b, la, lb = _inputs()
    topo = torch.randn(3, 10, 32)
    active = torch.ones(3, dtype=torch.bool)
    out_off = cond(h_a, h_b, la, lb)
    out_on = cond(h_a, h_b, la, lb, topo_tokens=topo, topo_active=active)
    assert not torch.allclose(out_on, out_off)
```

- [ ] **Step 3: Verify failure.**
- [ ] **Step 4: Implement**

```python
"""Conditioned V3.1 trunk: PairCrossAttention subclass with cls-token gated
cross-attention after the final n_inj blocks (design rev 3 §3.4). B0.py is
never modified; the parent forward loop is re-stated here with the injection."""

from __future__ import annotations

from typing import cast

import torch

from src.model.B0 import PairCrossAttention, _build_padding_mask
from src.model.egostitch.conditioning import GatedCrossAttention
from torch import nn


class ConditionedPairCrossAttention(PairCrossAttention):
    """PairCrossAttention + zero-init gated cls conditioning (§3.4 pins)."""

    def __init__(
        self,
        *,
        n_inj: int = 1,
        xattn_heads: int = 8,
        xattn_dropout: float = 0.0,
        d_model: int,
        **kwargs: object,
    ) -> None:
        super().__init__(d_model=d_model, **kwargs)  # type: ignore[arg-type]
        if not 1 <= n_inj <= len(self.layers):
            raise ValueError("n_inj must be in [1, n_layers]")
        self.n_inj = n_inj
        self.topo_xattn = nn.ModuleList(
            GatedCrossAttention(d_model, xattn_heads, xattn_dropout) for _ in range(n_inj)
        )
        self.cont_xattn = nn.ModuleList(
            GatedCrossAttention(d_model, xattn_heads, xattn_dropout) for _ in range(n_inj)
        )

    def forward(  # type: ignore[override]
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
        *,
        topo_tokens: torch.Tensor | None = None,
        cont_tokens: torch.Tensor | None = None,
        topo_active: torch.Tensor | None = None,
        cont_active: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = h_a.size(0)
        mask_a = _build_padding_mask(lengths_a, h_a.size(1))
        mask_b = _build_padding_mask(lengths_b, h_b.size(1))
        cls_token = self.cls_token.repeat(batch_size, 1, 1)
        n_layers = len(self.layers)
        for idx, layer in enumerate(self.layers):
            h_a, h_b, cls_token = layer(h_a, h_b, cls_token, mask_a, mask_b)
            inj = idx - (n_layers - self.n_inj)
            if inj >= 0:
                if topo_tokens is not None and topo_active is not None:
                    cls_token = self.topo_xattn[inj](
                        cls_token, topo_tokens, None, topo_active
                    )
                if cont_tokens is not None and cont_active is not None:
                    cls_token = self.cont_xattn[inj](
                        cls_token, cont_tokens, None, cont_active
                    )
        cls_vec = cls_token.squeeze(1)
        if self.pair_readout_mode == "pair_context_gated":
            return cast(
                torch.Tensor,
                self.pair_context_readout(h_a, h_b, cls_vec, mask_a, mask_b),
            )
        base_repr = self._rich_pooling_readout(h_a, h_b, cls_vec, mask_a, mask_b)
        if self.pair_readout_mode == "grid_sketch_fusion":
            return cast(
                torch.Tensor,
                self.grid_sketch_readout(base_repr, h_a, h_b, mask_a, mask_b),
            )
        return base_repr
```

- [ ] **Step 5: Run tests** — PASS (bypass equality is the load-bearing one).
- [ ] **Step 6:** ruff + mypy. **Step 7: Commit** —
  `git commit -m "feat(g5-e2e): conditioned trunk (cls-token gated x-attn, exact bypass)"`

### Task 11: `EgoStitchE2E` model (four-logit decomposition, symmetry)

**Files:**
- Create: `src/model/egostitch/e2e_model.py`
- Modify: `src/model/egostitch/config.py` (append `E2EConfig` dataclass)
- Test: `tests/model/test_egostitch_e2e_model.py`

**Interfaces:**
- Consumes: `SiameseEncoder`, `MLPHead` (`src/model/B0.py` — read their ctor
  signatures at `B0.py:88`/`B0.py:151` before wiring), Imagine/Tokenize-lite via
  `EgoStitchStage1.encode_nodes` (`src/model/egostitch/model.py:106`),
  `sinkhorn_plan`, Tasks 5–10 modules.
- Produces: `E2EConfig` dataclass (fields: `d_model: int = 512`,
  `encoder_layers: int = 3`, `cross_attn_layers: int = 3`, `n_heads: int = 8`,
  `n_inj: int = 1`, `ste_dim: int = 128`, `ste_layers: int = 3`,
  `xattn_heads: int = 8`, `p_topo: float = 0.15`, `p_cont: float = 0.15`);
  `EgoStitchE2E(nn.Module)` with:
  - `forward(batch: dict[str, torch.Tensor], *, masks: HeadNullMasks | None)
    -> dict[str, torch.Tensor]` (key `"logits"`; batch needs `emb_a`, `emb_b`,
    `len_a`, `len_b` token streams plus the per-node feature tensors the
    Stage-1 `encode_nodes` consumes);
  - `decompose(batch) -> dict[str, torch.Tensor]` returning
    `{"full", "f_logit", "pair_content", "pair_topology"}` — eval-time hard
    bypasses (`masks_for_null`), one checkpoint.
  - AB/BA handled internally: the trunk runs both orders through the **same**
    `ConditionedPairCrossAttention` with `swap_direction(scaffold)` for BA and
    feature-wise max before the head (mirrors `V3_1._pair_representation`,
    `B0.py:1093`); branch masks shared across both orders.

- [ ] **Step 1: Write the failing tests.** The three binding properties:

```python
"""E2E model property tests (design rev 3 §3.4–§3.5 acceptance criteria)."""
import torch

from src.model.egostitch.conditioning import (
    NULL_ALL_HEAD, NULL_CONTENT_HEAD, NULL_TOPO_HEAD, masks_for_null,
)
from src.model.egostitch.config import E2EConfig
from src.model.egostitch.e2e_model import EgoStitchE2E


def _tiny_model_and_batch() -> tuple[EgoStitchE2E, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    cfg = E2EConfig(d_model=32, encoder_layers=1, cross_attn_layers=2, n_heads=4,
                    n_inj=1, ste_dim=16, ste_layers=2, xattn_heads=4)
    model = EgoStitchE2E(cfg).eval()
    b, t, d_in = 4, 6, model.input_dim
    batch = {
        "emb_a": torch.randn(b, t, d_in),
        "emb_b": torch.randn(b, t, d_in),
        "len_a": torch.full((b,), t, dtype=torch.long),
        "len_b": torch.full((b,), t, dtype=torch.long),
        "x_a": torch.randn(b, model.node_feature_dim),
        "x_b": torch.randn(b, model.node_feature_dim),
    }
    return model, batch


def test_pair_symmetry_all_conditions() -> None:
    model, batch = _tiny_model_and_batch()
    with torch.no_grad():
        for p in (model.trunk.topo_xattn[0].gate, model.trunk.cont_xattn[0].gate):
            p.fill_(0.4)  # open gates: symmetry must hold with live conditioning
    swapped = dict(batch)
    swapped["emb_a"], swapped["emb_b"] = batch["emb_b"], batch["emb_a"]
    swapped["len_a"], swapped["len_b"] = batch["len_b"], batch["len_a"]
    swapped["x_a"], swapped["x_b"] = batch["x_b"], batch["x_a"]
    for null in (None, NULL_ALL_HEAD, NULL_TOPO_HEAD, NULL_CONTENT_HEAD):
        masks = None if null is None else masks_for_null(null, 4, torch.device("cpu"))
        out_ij = model(batch, masks=masks)["logits"]
        out_ji = model(swapped, masks=masks)["logits"]
        assert torch.allclose(out_ij, out_ji, atol=1e-5), f"asymmetry under {null}"


def test_train_mask_equals_eval_bypass() -> None:
    model, batch = _tiny_model_and_batch()
    with torch.no_grad():
        model.trunk.topo_xattn[0].gate.fill_(0.4)
        model.trunk.cont_xattn[0].gate.fill_(0.4)
    dec = model.decompose(batch)  # eval-time hard bypasses
    for null, key in (
        (NULL_ALL_HEAD, "f_logit"),
        (NULL_TOPO_HEAD, "pair_content"),
        (NULL_CONTENT_HEAD, "pair_topology"),
    ):
        masked = model(batch, masks=masks_for_null(null, 4, torch.device("cpu")))["logits"]
        assert torch.allclose(masked, dec[key], atol=1e-6), f"mask!=bypass for {null}"


def test_f_logit_invariant_to_scaffold() -> None:
    model, batch = _tiny_model_and_batch()
    dec_a = model.decompose(batch)
    batch2 = dict(batch)
    batch2["x_a"] = torch.randn_like(batch["x_a"])  # changes slots => scaffold
    dec_b = model.decompose(batch2)
    # f_logit depends on token streams only — x_a feeds imagination, not trunk
    assert torch.allclose(dec_a["f_logit"], dec_b["f_logit"], atol=1e-6)
```

- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement.** Composition (complete class; read
  `SiameseEncoder.__init__` at `B0.py:88` and `MLPHead.__init__` at `B0.py:151`
  first and match their actual signatures — the sketch below assumes
  `SiameseEncoder(input_dim, d_model, n_layers, n_heads, dropout)` and
  `MLPHead(d_model, hidden_dims, dropout)`-style ctors; adjust to the real
  ones, the tests above are the acceptance criteria):

```python
"""EgoStitchE2E: stitched-topology-conditioned pair encoder (design rev 3)."""

from __future__ import annotations

import torch
from torch import nn

from src.model.B0 import MLPHead, SiameseEncoder
from src.model.egostitch.conditioning import HeadNullMasks, masks_for_null
from src.model.egostitch.conditioning import (
    NULL_ALL_HEAD, NULL_CONTENT_HEAD, NULL_TOPO_HEAD,
)
from src.model.egostitch.config import E2EConfig, EgoStitchConfig
from src.model.egostitch.model import EgoStitchStage1
from src.model.egostitch.scaffold import (
    ContentProjector, build_content_tokens, build_scaffold, swap_direction,
)
from src.model.egostitch.ste import STEncoder
from src.model.egostitch.stitch import sinkhorn_plan
from src.model.egostitch.trunk import ConditionedPairCrossAttention


class EgoStitchE2E(nn.Module):
    def __init__(self, cfg: E2EConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_dim = 1536          # frozen feature dim (spec §0 table)
        self.node_feature_dim = 1536   # per-node features feeding imagination
        self.generator_cfg = EgoStitchConfig()  # Stage-1 defaults (spec §13)
        self.generator = EgoStitchStage1(self.generator_cfg)
        self.encoder = SiameseEncoder(  # match real ctor signature (B0.py:88)
            input_dim=self.input_dim, d_model=cfg.d_model,
            n_layers=cfg.encoder_layers, n_heads=cfg.n_heads, dropout=0.1,
        )
        self.trunk = ConditionedPairCrossAttention(
            d_model=cfg.d_model, n_heads=cfg.n_heads,
            n_layers=cfg.cross_attn_layers, dropout=0.1,
            pair_readout_mode="pair_context_gated",
            mixing_mode="bidirectional_cross",
            n_inj=cfg.n_inj, xattn_heads=cfg.xattn_heads,
        )
        self.ste = STEncoder(cfg.d_model, cfg.ste_dim, cfg.ste_layers)
        self.content_proj = ContentProjector(
            d_p=self.generator_cfg.d_p, d_model=cfg.d_model
        )
        self.head = MLPHead(cfg.d_model, hidden_dims=(cfg.d_model // 2,), dropout=0.1)

    def _context(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        enc_a = self.generator.encode_nodes(batch["x_a"])
        enc_b = self.generator.encode_nodes(batch["x_b"])
        slots_a, slots_b = enc_a.slots, enc_b.slots
        plan = sinkhorn_plan(
            slots_a.h, slots_b.h, slots_a.pi, slots_b.pi, slots_a.mult, slots_b.mult
        )
        scaffold = build_scaffold(slots_a, slots_b, plan)
        topo_ab = self.ste(scaffold)
        topo_ba = self.ste(swap_direction(scaffold))
        matched_a = torch.zeros_like(slots_a.pi)  # wired to pointer-match in Task 13
        matched_b = torch.zeros_like(slots_b.pi)
        cont = self.content_proj(
            build_content_tokens(slots_a, slots_b, matched_a, matched_b)
        )
        return topo_ab, topo_ba, cont

    def forward(
        self, batch: dict[str, torch.Tensor], *, masks: HeadNullMasks | None = None
    ) -> dict[str, torch.Tensor]:
        b = batch["emb_a"].size(0)
        device = batch["emb_a"].device
        if masks is None:
            masks = masks_for_null("none", b, device)
        need_topo = bool(masks.topo.any())
        need_cont = bool(masks.cont.any())
        topo_ab = topo_ba = cont = None
        if need_topo or need_cont:
            topo_ab, topo_ba, cont = self._context(batch)
        h_a = self.encoder(batch["emb_a"], batch["len_a"])
        h_b = self.encoder(batch["emb_b"], batch["len_b"])
        kw_ab = dict(
            topo_tokens=topo_ab if need_topo else None,
            cont_tokens=cont if need_cont else None,
            topo_active=masks.topo if need_topo else None,
            cont_active=masks.cont if need_cont else None,
        )
        kw_ba = dict(kw_ab, topo_tokens=topo_ba if need_topo else None)
        feat_ab = self.trunk(h_a, h_b, batch["len_a"], batch["len_b"], **kw_ab)
        feat_ba = self.trunk(h_b, h_a, batch["len_b"], batch["len_a"], **kw_ba)
        feat = torch.max(torch.stack([feat_ab, feat_ba], dim=-1), dim=-1).values
        return {"logits": self.head(feat).squeeze(-1)}

    @torch.no_grad()
    def decompose(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        b = batch["emb_a"].size(0)
        device = batch["emb_a"].device
        return {
            "full": self(batch, masks=None)["logits"],
            "f_logit": self(batch, masks=masks_for_null(NULL_ALL_HEAD, b, device))["logits"],
            "pair_content": self(
                batch, masks=masks_for_null(NULL_TOPO_HEAD, b, device)
            )["logits"],
            "pair_topology": self(
                batch, masks=masks_for_null(NULL_CONTENT_HEAD, b, device)
            )["logits"],
        }
```

  Symmetry note (why `test_pair_symmetry_all_conditions` passes): swapping the
  batch swaps `(emb_a, emb_b)` and `(x_a, x_b)`; `build_scaffold(slots_b,
  slots_a, plan.T)` equals `swap_direction(build_scaffold(slots_a, slots_b,
  plan))` up to the node permutation the STE is equivariant to and attention is
  invariant to — if the test still fails, compute the swapped-batch scaffold
  explicitly from the swapped slot order rather than reusing `swap_direction`,
  and keep the test as the arbiter.
- [ ] **Step 4: Run tests** — all three property tests PASS.
- [ ] **Step 5:** ruff + mypy. **Step 6: Commit** —
  `git commit -m "feat(g5-e2e): EgoStitchE2E with four-logit decomposition"`

---

## Phase 4 — Data and worker integration

### Task 12: Packed-token edge stream

**Files:**
- Modify: `src/train_egostitch.py` (`prepare_pack` at `:621`, `EgoDataConfig`
  at `:116`, `_BatchFactory` at `:1022`)
- Test: `tests/test_train_egostitch.py` (append)

**Interfaces:**
- Consumes: the worker-generic `e2_pipeline` pack seam (`prepare_pack`), the
  existing F0/grounding caches, `enumerate_edge_stream` (`:752`).
- Produces: batches whose dicts additionally contain `emb_a`, `emb_b`
  `(B, T, 1536)` raw-token tensors and `len_a`, `len_b` `(B,)` when
  `model.family == "egostitch_e2e"`; config key `data.pack_dir` (raw-token pack
  produced by the same packing machinery the B0 V3.1 run uses).

- [ ] **Step 1:** Read `prepare_pack` (`src/train_egostitch.py:621-694`) and
  `_BatchFactory` (`:1022-1200`) in full before editing.
- [ ] **Step 2:** Write a failing test: construct a tiny synthetic pack (reuse
  the fixture pattern already in `tests/test_train_egostitch.py` for the F0
  path), build a `_BatchFactory` with family `egostitch_e2e`, assert the batch
  dict contains `emb_a/emb_b/len_a/len_b` with correct shapes and that pair
  order matches `enumerate_edge_stream` output for the same seed.
- [ ] **Step 3:** Implement: extend `EgoDataConfig` with `pack_dir: str | None
  = None`; in `_BatchFactory`, when the family is `egostitch_e2e`, look up each
  pair's two node token sequences from the pack (memory-mapped, same reader the
  B0 loader uses — import it rather than re-implementing) and pad to the batch
  max length; keep the existing F0/grounding tensors (imagination still needs
  them). `s0_cache` loading is skipped for this family.
- [ ] **Step 4:** Run the new test + the full existing worker test file:
  `uv run pytest tests/test_train_egostitch.py -v` → PASS (no regression on the
  frozen-s0 family paths).
- [ ] **Step 5: Commit** — `git commit -m "feat(g5-e2e): packed-token edge stream in the EgoStitch worker"`

### Task 13: Training integration and telemetry

**Files:**
- Modify: `src/train_egostitch.py` (`_CompositeStep` at `:1202`, config
  parsing, telemetry), `src/model/egostitch/config.py`
- Test: `tests/test_train_egostitch.py` (append)

**Interfaces:**
- Consumes: `EgoStitchE2E` (Task 11), `sample_branch_masks` (Task 5).
- Produces: family `egostitch_e2e` trainable end-to-end: `L_edge` (BCE) on the
  full logits with per-step seeded branch masks (`_seeded_generator(seed,
  epoch, step)` pattern already at `:997`); existing `L_recon`/`L_real`/`L_ssl`
  node losses unchanged (from `EgoStitchStage1.node_losses`); warm-start
  fraction keeps `L_edge` (and therefore trunk/STE/gates) inactive; telemetry
  rows gain `gate_topo_tanh`, `gate_cont_tanh` (per injected block),
  `grad_rms_trunk`, `grad_rms_ste`, `grad_rms_content`, and per-epoch
  `topology_delta_std` on a fixed validation slice.

- [ ] **Step 1:** Failing test: one CPU optimization step on the tiny model
  (Task 11 fixture) through `_CompositeStep` with family `egostitch_e2e`;
  assert loss is finite, gates receive gradient only after warm-start, and the
  telemetry keys above appear in the metrics row.
- [ ] **Step 2:** Implement: branch on family in the model-build and loss
  assembly paths; wire `matched_*` for content tokens from the grounding
  pointer (slot is "matched" when its pointer argmax lands on a grounding
  candidate that Hungarian-matched the same target — reuse the matching output
  already computed for `L_gate`); add the telemetry fields next to the existing
  §13.17 series.
- [ ] **Step 3:** Run: `uv run pytest tests/test_train_egostitch.py -v` → PASS,
  plus `uv run pytest tests/test_e2_pipeline.py -v` (orchestrator regression).
- [ ] **Step 4: Commit** — `git commit -m "feat(g5-e2e): e2e training family with branch dropout + gate telemetry"`

---

## Phase 5 — Scoring and gate

### Task 14: Four-logit fp32 scoring

**Files:**
- Modify: `src/score_universe.py` (the `_score_egostitch` path)
- Test: `tests/test_score_universe.py` (append)

**Interfaces:**
- Consumes: `EgoStitchE2E.decompose`.
- Produces: for family `egostitch_e2e`, the scores artifact carries four
  arrays — `logits` (full), `f_logit`, `pair_content`, `pair_topology` — all
  computed with autocast disabled in the pair pass (per-node encode may stay
  bf16, cached fp32, mirroring the existing §13.16 machinery), provenance
  string `egostitch_e2e_pair_fp32_v1`, and passes the existing
  `validate_score_resolution` guard per array.

- [ ] **Step 1:** Read the existing `_score_egostitch` and the §13.16
  provenance writer in `src/score_universe.py` before editing.
- [ ] **Step 2:** Failing test: score a tiny synthetic pair list with the tiny
  e2e model; assert all four arrays present, fp32, provenance recorded, and
  `f_logit == pair_content` when all gates are zero... (gates zero ⇒ all four
  identical — assert exactly that, it is the init-state sanity check).
- [ ] **Step 3:** Implement via `model.decompose` under `torch.autocast(...,
  enabled=False)` for the pair pass; write the three extra arrays through the
  same shard/merge machinery (strict merge must preserve all four).
- [ ] **Step 4:** Run: `uv run pytest tests/test_score_universe.py -v` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(g5-e2e): four-logit fp32 candidate scoring"`

### Task 15: Gate — liveness vs f_logit, representation probes, 5-arm summary

**Files:**
- Modify: `src/experiments/g5_stage1.py`
- Create: `src/experiments/probes.py`
- Test: `tests/test_g5_stage1.py` (append), `tests/experiments/test_probes.py`

**Interfaces:**
- Consumes: the four-logit artifact (Task 14), STE token states (exported for a
  fixed probe node sample by a small helper added to `EgoStitchE2E`:
  `probe_states(batch) -> torch.Tensor`), message-partition graph quantities
  via `networkx` (the evaluator convention pinned in spec §13.6).
- Produces:
  - `probes.linear_probe_r2(states: np.ndarray, targets: np.ndarray) -> float`
    (closed-form ridge, fixed λ=1e-3, 5-fold);
  - `probes.degree_partialled_r2(states, targets, degrees) -> float`
    (targets and states residualized against degree before probing);
  - gate liveness computed **within checkpoint**: std ratio
    `std(full − f_logit)/std(f_logit)`, Spearman(full, f_logit), top-1%
    overlap — same conjunctive death rule, new reference (no frozen-s0
    comparator artifact, no alignment step);
  - a 5-arm summary table (full / B0-e2e / pair+topology / 6a shuffle / p=0)
    keyed by run-metadata registration hash.
- [ ] **Step 1:** Failing tests: probe R² ≈ 1 on linearly-predictable synthetic
  targets, ≈ 0 on shuffled targets; degree-partialled probe ≈ 0 when the target
  IS degree; liveness signals fire on `full == f_logit` artifacts and not on
  decorrelated ones.
- [ ] **Step 2:** Implement `probes.py` (numpy + closed-form ridge; no sklearn
  dependency) and the gate changes; delete the frozen-s0 comparator alignment
  path for the e2e family only (the frozen-s0 family gate remains intact for
  the historical artifacts).
- [ ] **Step 3:** Run: `uv run pytest tests/test_g5_stage1.py tests/experiments/test_probes.py -v` → PASS.
- [ ] **Step 4:** Full suite + gates: `uv run pytest && uv run ruff check . &&
  uv run mypy src tests` → clean (expect the two documented pre-existing local
  failures: `tests/test_b0_attention.py` torch-version issue,
  `tests/test_e2_ddp_integration.py` rendezvous — container-only).
- [ ] **Step 5: Commit** — `git commit -m "feat(g5-e2e): within-checkpoint liveness, representation probes, 5-arm gate"`

---

## Self-review notes (run before execution)

- **Spec coverage:** design §3 → Tasks 5–11; §4 → Tasks 5, 11, 13; §5 Stage-1
  arm set → Tasks 4, 15 (arms 6b–f and depth rungs are E1/E3 scope — NOT in
  this plan, by design §5); §6 → Tasks 1, 13, 14, 15; §7 → Tasks 12–14
  (the measured H20 cost re-estimate is a runtime deliverable, pinned in the
  Task 4 registration text); §8 → Tasks 1–4; §10 → Task 3.
- **Known judgment calls encoded here:** B0.py untouched (subclass); STE width
  128 with projection to trunk width; content `matched_*` zeros in Task 11 and
  wired in Task 13 (the Task 11 tests do not depend on it); Task 11's ctor
  sketch must be reconciled with the real `SiameseEncoder`/`MLPHead` signatures
  at implementation time — the property tests, not the sketch, are the
  acceptance criteria.
- **Type consistency:** `HeadNullMasks`/`masks_for_null` names match across
  Tasks 5, 10, 11, 14; `ScaffoldTokens`/`build_scaffold`/`swap_direction`
  across Tasks 7, 8, 11; `decompose` keys (`full`, `f_logit`, `pair_content`,
  `pair_topology`) across Tasks 11, 14, 15.

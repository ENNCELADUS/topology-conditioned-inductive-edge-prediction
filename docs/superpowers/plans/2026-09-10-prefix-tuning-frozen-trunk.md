# Topology-Supervised Prefix Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `v3_1_prefix` model family (a frozen `prefix_base` V3.1 trunk plus a trainable zero-init gated KV prefix), its three arms (`prefix_static`, `prefix_pair`, `prefix_pair_bce`), the `prefix_base` base config, scoring-time interventions, and the struct_hpo studies, so the H20 campaign can run the attribution experiment in the spec.

**Architecture:** A new classifier module wraps a trained `V3_1` whose pair trunk has `mixing.mode: bidirectional_cross`. Each frozen `CrossAttentionLayer` is re-driven by a `PrefixCrossAttentionLayer` that calls the layer's own `nn.MultiheadAttention` modules unchanged and adds a separately-softmaxed, tanh-gated prefix branch at the three attention sites. The prefix is a static per-layer token matrix, optionally plus a slot-specific low-rank pair shift generated from the frozen encoder's masked-mean vectors. Training reuses `train_b0` with the structural stream; only prefix parameters enter the optimizer; the published checkpoint embeds the frozen base state so `score_universe` rebuilds it alone.

**Tech Stack:** PyTorch 2.10 (`nn.MultiheadAttention` packed `in_proj_weight`), accelerate DDP via `src/train_b0.py`, Optuna via `src/experiments/struct_hpo.py`, pytest with `-n0` for debugging, ruff (Google docstrings, full annotations, no `print`), mypy strict.

**Spec:** `docs/superpowers/specs/2026-09-10-prefix-tuning-frozen-trunk-design.md`

## Global Constraints

- Local commands use `.venv/bin/python -m …`; never `rtk` around `uv run`. Tests: `.venv/bin/python -m pytest <file> -n0` while debugging; full runs need `--dist loadfile`.
- Ruff enforces Google docstrings and full annotations in `src/` and bans `print`; mypy is strict for `src` and `tests`.
- No formalism gates: record the base checkpoint path and its SHA-256 as provenance; never verify or pin them (spec §4).
- Inference input is exactly `(x_u, x_v)`; the prefix generator reads only the frozen encoder's summaries (spec §3).
- Exact null identity: with every gate at zero the arm must reproduce the frozen base logits bit for bit (spec §5.2).
- The base stays in eval mode permanently, even after `wrapper.train()` (spec §4).
- Prefix defaults: `tokens: 16`, `rank: 8`, `bottleneck: 128`, all three layers, all three sites (spec §5).
- The shared HPO rule is unchanged: three objectives (AUPRC↑, GS↑, geo-MMD↓), five-metric mean-rank winner (spec §6).
- `src/vendor/` is untouched. Docs that describe changed behaviour are updated in the same task (`CLAUDE.md` rule).
- GPU work is H20-only. Every task ends local; Task 10 ends with the exact launch commands.
- Commit messages end with the session attribution lines given in the conversation.

---

## File map

| Path | Responsibility |
|---|---|
| `configs/split_seed42/prefix_base.yaml` | Create. Headline B0 with `mixing.mode: bidirectional_cross`. |
| `src/model/egostitch/classifier/prefix.py` | Create. `PrefixConfig`, `PrefixGenerator`, `prefix_branch`, `PrefixCrossAttentionLayer`, `V3_1Prefix`. |
| `src/model/egostitch/classifier/__init__.py` | Modify. Export the new names if the package re-exports classifiers. |
| `src/train_b0.py` | Modify. `MODEL_FAMILIES`, `is_v3_1_family`, `resolve_model_kwargs`, `build_model`, `_build_optimizer`, four family gates, static-prefix init before `prepare`. |
| `src/score_universe.py` | Modify. `MODEL_BUILDERS["v3_1_prefix"]`, family dispatch, `--prefix-intervention`, meta + merge check. |
| `src/eval/test_protocol.py` | Modify. Thread `prefix_intervention` like `topo_gen_control`. |
| `src/experiments/struct_hpo.py` | Modify. Arms `prefix_static`, `prefix_pair`, `prefix_pair_bce`; `lr` in the search box; `--lr-center`. |
| `configs/split_seed42/prefix_static.yaml`, `prefix_pair.yaml`, `prefix_pair_bce.yaml` | Create. |
| `tests/test_prefix_model.py` | Create. Module-level and wrapper tests. |
| `tests/test_train_b0_prefix.py` | Create. Family dispatch, optimizer, base loading, init. |
| `tests/test_score_universe.py`, `tests/test_struct_hpo.py`, `tests/test_sweep_configs.py` | Modify. |
| `docs/03-experiments.md`, `CLAUDE.md`, `AGENTS.md`, `hpc/README.md` | Modify. |

Shared tiny fixture used by every test task (copy it verbatim into each new test file; both files need it and `tests/` has no shared conftest helper for V3_1 configs):

```python
def _tiny_base_config(mixing: str = "bidirectional_cross") -> dict[str, object]:
    """A 4-dim, 8-wide V3_1 with two cross-attention layers and every dropout on."""
    return {
        "input_dim": 4,
        "d_model": 8,
        "encoder_layers": 1,
        "cross_attn_layers": 2,
        "n_heads": 2,
        "mlp_head": {
            "hidden_dims": [8],
            "dropout": 0.2,
            "activation": "gelu",
            "norm": "layernorm",
            "spectral_norm": False,
        },
        "regularization": {
            "dropout": 0.1,
            "token_dropout": 0.1,
            "cross_attention_dropout": 0.1,
            "stochastic_depth": 0.1,
        },
        "rich_pooling": {"components": ["mean", "attn", "max", "gated"]},
        "pair_readout": {
            "mode": "pair_context_gated",
            "order_aggregation": "abba_max",
            "spectral_norm": False,
        },
        "mixing": {"mode": mixing},
        "label_smoothing": 0.0,
        "positive_weight": 5.0,
    }


def _pair_batch(n: int = 6, seed: int = 0) -> dict[str, torch.Tensor]:
    """Random (B, L, 4) token pairs with ragged lengths and binary labels."""
    gen = torch.Generator().manual_seed(seed)
    emb_a = torch.randn(n, 7, 4, generator=gen)
    emb_b = torch.randn(n, 5, 4, generator=gen)
    len_a = torch.tensor([7, 6, 5, 7, 4, 7][:n])
    len_b = torch.tensor([5, 5, 3, 4, 5, 2][:n])
    label = torch.tensor([1, 0, 0, 1, 0, 1][:n])
    return {"emb_a": emb_a, "emb_b": emb_b, "len_a": len_a, "len_b": len_b, "label": label}
```

---

### Task 1: `prefix_base` config and local smoke of the bidirectional path

**Files:**
- Create: `configs/split_seed42/prefix_base.yaml`
- Test: `tests/test_sweep_configs.py` (add one test)

**Interfaces:**
- Produces: `configs/split_seed42/prefix_base.yaml`, identical to `configs/split_seed42/b0_v31.yaml` except `mixing.mode`, `output_dir`, and the header comment. Later tasks reference `outputs/split_seed42/prefix_base/best.pt` as `prefix.base_checkpoint`.

- [ ] **Step 1: Write the failing config test**

Append to `tests/test_sweep_configs.py`:

```python
def test_prefix_base_differs_from_headline_b0_only_in_mixing_and_output_dir() -> None:
    base = yaml.safe_load(Path("configs/split_seed42/b0_v31.yaml").read_text(encoding="utf-8"))
    cross = yaml.safe_load(Path("configs/split_seed42/prefix_base.yaml").read_text(encoding="utf-8"))
    assert cross["model"]["config"]["mixing"] == {"mode": "bidirectional_cross"}
    assert cross["output_dir"] == "outputs/split_seed42/prefix_base"
    cross["model"]["config"]["mixing"] = base["model"]["config"]["mixing"]
    cross["output_dir"] = base["output_dir"]
    assert cross == base
```

If the file does not already import `yaml` and `Path`, add `import yaml` and `from pathlib import Path` at its top.

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_sweep_configs.py -n0 -k prefix_base -v`
Expected: FAIL with `FileNotFoundError` for `configs/split_seed42/prefix_base.yaml`.

- [ ] **Step 3: Create the config**

```bash
sed -e 's|^# B0 baseline on the seed-42.*|# prefix_base: the headline B0 recipe with bidirectional cross-attention in the pair trunk.\n# The exact frozen base of the prefix arms (spec 2026-09-10-prefix-tuning-frozen-trunk-design §4);\n# identical to b0_v31.yaml except mixing.mode and output_dir.|' \
    -e 's|      mode: none|      mode: bidirectional_cross|' \
    -e 's|^output_dir: outputs/split_seed42/b0_v31$|output_dir: outputs/split_seed42/prefix_base|' \
    configs/split_seed42/b0_v31.yaml > configs/split_seed42/prefix_base.yaml
```

Open the file and confirm exactly three differences against `b0_v31.yaml` (`diff configs/split_seed42/b0_v31.yaml configs/split_seed42/prefix_base.yaml`): the header, `mode: bidirectional_cross`, and `output_dir`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_sweep_configs.py -n0 -k prefix_base -v`
Expected: PASS

- [ ] **Step 5: Smoke the bidirectional path end-to-end on CPU**

Write a scratch config that shrinks the batch and adds the structural stream, then run two optimizer steps in direct mode (debug-only path per `CLAUDE.md`):

```bash
S=/private/tmp/claude-501/-Users-richardwang-Documents-topology-conditioned-inductive-edge-prediction/810cd6bb-d08e-4e35-a289-7b1f59c6536f/scratchpad
mkdir -p $S/prefix_base_smoke
.venv/bin/python - <<'EOF'
import yaml, pathlib
p = pathlib.Path("configs/split_seed42/prefix_base.yaml")
cfg = yaml.safe_load(p.read_text())
cfg["data"]["batch_pairs"] = 8
cfg["data"]["token_budget"] = 8192
cfg["optim"]["epochs"] = 1
cfg["mixed_precision"] = "no"
cfg["struct"] = yaml.safe_load(pathlib.Path("configs/split_seed42/struct_new.yaml").read_text())["struct"]
cfg["struct"]["nodes"] = 12
cfg["struct"]["background_nodes"] = 4
out = pathlib.Path("/private/tmp/claude-501/-Users-richardwang-Documents-topology-conditioned-inductive-edge-prediction/810cd6bb-d08e-4e35-a289-7b1f59c6536f/scratchpad/prefix_base_smoke/cfg.yaml")
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
EOF
.venv/bin/python -m src.train_b0 --config $S/prefix_base_smoke/cfg.yaml --output-dir $S/prefix_base_smoke/out --max-steps 2 2>&1 | tail -20
```

Expected: the run reaches "step 2" and exits without a traceback; `$S/prefix_base_smoke/out/metrics.jsonl` exists. If the direct path cannot find local data (`TCIEP_DATA_ROOT` unset and `data/` absent), record that the smoke is deferred to the H20 `--max-steps` debug launch in Task 10 and continue.

- [ ] **Step 6: Commit**

```bash
git add configs/split_seed42/prefix_base.yaml tests/test_sweep_configs.py
git commit -m "feat(split): prefix_base, the headline B0 with bidirectional cross-attention"
```

---

### Task 2: `PrefixConfig` and `PrefixGenerator`

**Files:**
- Create: `src/model/egostitch/classifier/prefix.py`
- Test: `tests/test_prefix_model.py`

**Interfaces:**
- Produces:
  - `PrefixConfig(tokens: int = 16, rank: int = 8, conditioning: str = "static", bottleneck: int = 128, base_checkpoint: str = "", base_checkpoint_sha256: str | None = None)` with `from_mapping(raw: Mapping[str, object]) -> PrefixConfig` and `to_dict() -> dict[str, object]`.
  - `PrefixGenerator(d_model: int, n_layers: int, n_heads: int, cfg: PrefixConfig)` with
    - `condition(encoded_a, encoded_b, mask_a, mask_b) -> torch.Tensor | None` returning `z` of shape `(B, bottleneck)` or `None` for static;
    - `prefix(layer_index: int, z: torch.Tensor | None, batch_size: int) -> torch.Tensor` of shape `(B, tokens, d_model)`;
    - `gate(layer_index: int, site: int) -> torch.Tensor` of shape `(n_heads,)` (raw, before tanh);
    - `init_static_from_tokens(tokens: torch.Tensor, seed: int) -> None` copying `tokens` rows (shape `(N, d_model)`, `N >= tokens`) into every layer's `P0`;
    - buffers `z_sum (bottleneck,)`, `z_count ()`, property `z_mean`.
  - Parameter names: `p0` (`ParameterList`, one `(tokens, d_model)` per layer), `gates` (`Parameter (n_layers, 3, n_heads)`, zeros), `cond_norm`, `cond_proj` (pair only), `shift_in` (`ModuleList` of `Linear(bottleneck, tokens*rank)`), `shift_out` (`ParameterList` of `(rank, d_model)`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_prefix_model.py` with the shared fixture from the file map, then:

```python
"""Prefix-tuning module tests: generator shapes, symmetry, and the slot-specific shift."""

from __future__ import annotations

import torch
from src.model.egostitch.classifier.prefix import PrefixConfig, PrefixGenerator


def test_static_generator_has_no_condition_and_shares_one_prefix() -> None:
    cfg = PrefixConfig(tokens=4, rank=2, conditioning="static", bottleneck=6)
    gen = PrefixGenerator(d_model=8, n_layers=2, n_heads=2, cfg=cfg)
    assert gen.condition(torch.zeros(3, 5, 8), torch.zeros(3, 5, 8), None, None) is None
    prefix = gen.prefix(1, None, batch_size=3)
    assert prefix.shape == (3, 4, 8)
    assert torch.equal(prefix[0], prefix[2])
    assert not any(name.startswith("shift") or name.startswith("cond") for name, _ in gen.named_parameters())


def test_pair_generator_is_symmetric_and_slot_specific() -> None:
    cfg = PrefixConfig(tokens=4, rank=2, conditioning="pair", bottleneck=6)
    gen = PrefixGenerator(d_model=8, n_layers=2, n_heads=2, cfg=cfg)
    a = torch.randn(3, 5, 8)
    b = torch.randn(3, 4, 8)
    mask_a = torch.tensor([[False] * 5, [False] * 4 + [True], [False] * 3 + [True] * 2])
    mask_b = torch.tensor([[False] * 4, [False] * 4, [False] * 2 + [True] * 2])
    z_ab = gen.condition(a, b, mask_a, mask_b)
    z_ba = gen.condition(b, a, mask_b, mask_a)
    assert z_ab is not None and z_ab.shape == (3, 6)
    assert torch.allclose(z_ab, z_ba)
    prefix = gen.prefix(0, z_ab, batch_size=3)
    delta = prefix - gen.p0[0].unsqueeze(0)
    # slot-specific: rows of the shift differ within one pair
    assert not torch.allclose(delta[0, 0], delta[0, 1])
    # pair-specific: the shift differs across pairs
    assert not torch.allclose(delta[0], delta[1])


def test_gates_start_at_zero_and_shift_out_is_small_but_nonzero() -> None:
    cfg = PrefixConfig(tokens=4, rank=2, conditioning="pair", bottleneck=6)
    gen = PrefixGenerator(d_model=8, n_layers=3, n_heads=2, cfg=cfg)
    assert gen.gates.shape == (3, 3, 2)
    assert torch.count_nonzero(gen.gates) == 0
    assert all(p.abs().sum() > 0 for p in gen.shift_out)
    assert all(p.abs().max() < 0.2 for p in gen.shift_out)


def test_init_static_from_tokens_copies_rows_and_z_mean_accumulates() -> None:
    cfg = PrefixConfig(tokens=2, rank=2, conditioning="pair", bottleneck=6)
    gen = PrefixGenerator(d_model=8, n_layers=1, n_heads=2, cfg=cfg)
    rows = torch.arange(40, dtype=torch.float32).view(5, 8)
    gen.init_static_from_tokens(rows, seed=3)
    assert all(any(torch.equal(p_row, r) for r in rows) for p_row in gen.p0[0])
    gen.train()
    z = gen.condition(torch.randn(4, 3, 8), torch.randn(4, 3, 8), None, None)
    assert z is not None
    assert float(gen.z_count) == 4.0
    assert torch.allclose(gen.z_mean, z.mean(dim=0))


def test_config_round_trip_and_validation() -> None:
    raw = {"tokens": 16, "rank": 8, "conditioning": "pair", "bottleneck": 128,
           "base_checkpoint": "outputs/x/best.pt", "base_checkpoint_sha256": None}
    cfg = PrefixConfig.from_mapping(raw)
    assert cfg.to_dict() == raw
    try:
        PrefixConfig.from_mapping({**raw, "conditioning": "graph"})
    except ValueError as err:
        assert "conditioning" in str(err)
    else:
        raise AssertionError("bad conditioning must raise")
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_prefix_model.py -n0 -v`
Expected: FAIL with `ModuleNotFoundError: src.model.egostitch.classifier.prefix`.

- [ ] **Step 3: Implement `PrefixConfig` and `PrefixGenerator`**

Create `src/model/egostitch/classifier/prefix.py`:

```python
"""Topology-supervised prefix tuning on a frozen V3.1 trunk.

Design: docs/superpowers/specs/2026-09-10-prefix-tuning-frozen-trunk-design.md.
`PrefixGenerator` owns every trainable parameter (static prefix tokens, the
pair generator, the gates); `PrefixCrossAttentionLayer` re-drives one frozen
`CrossAttentionLayer` with a separately-softmaxed, tanh-gated prefix branch;
`V3_1Prefix` wraps a frozen `V3_1` whose pair trunk has bidirectional
cross-attention layers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from src.model.egostitch.classifier.layers import inner_token_mask, masked_mean

CONDITIONING_MODES = ("static", "pair")
SITES = 3  # A<-B, B<-A, CLS


@dataclass(frozen=True)
class PrefixConfig:
    """The ``model.config.prefix`` block.

    Attributes:
        tokens: Prefix length per layer.
        rank: Rank of the slot-specific pair shift.
        conditioning: ``"static"`` or ``"pair"``.
        bottleneck: Width of the pair condition ``z``.
        base_checkpoint: Path of the frozen base checkpoint (provenance only).
        base_checkpoint_sha256: SHA-256 of that file (provenance only, never verified).
    """

    tokens: int = 16
    rank: int = 8
    conditioning: str = "static"
    bottleneck: int = 128
    base_checkpoint: str = ""
    base_checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        """Validate ranges.

        Raises:
            ValueError: On a non-positive size or an unknown conditioning mode.
        """
        if self.tokens <= 0 or self.rank <= 0 or self.bottleneck <= 0:
            raise ValueError("prefix.tokens, prefix.rank, and prefix.bottleneck must be positive")
        if self.conditioning not in CONDITIONING_MODES:
            raise ValueError(
                f"prefix.conditioning must be one of {CONDITIONING_MODES}, got {self.conditioning!r}"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> PrefixConfig:
        """Parse the block, rejecting unknown keys.

        Args:
            raw: The mapping under ``model.config.prefix``.

        Returns:
            The parsed config.

        Raises:
            ValueError: On unknown keys.
        """
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown prefix keys: {unknown}")
        sha = raw.get("base_checkpoint_sha256")
        return cls(
            tokens=int(cast(int, raw.get("tokens", 16))),
            rank=int(cast(int, raw.get("rank", 8))),
            conditioning=str(raw.get("conditioning", "static")),
            bottleneck=int(cast(int, raw.get("bottleneck", 128))),
            base_checkpoint=str(raw.get("base_checkpoint", "")),
            base_checkpoint_sha256=None if sha is None else str(sha),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the block as a plain mapping (checkpoint-embeddable)."""
        return cast(dict[str, object], asdict(self))


class PrefixGenerator(nn.Module):
    """Every trainable parameter of the prefix arm.

    Static prefix ``p0[l]`` of shape ``(tokens, d_model)`` per layer; for
    ``conditioning == "pair"`` a shared condition ``z = GELU(Linear(LN(c)))``
    from the symmetric pair features ``c = [e_u + e_v; |e_u - e_v|; e_u * e_v]``
    and, per layer, a slot-specific low-rank shift
    ``reshape(shift_in[l](z), tokens, rank) @ shift_out[l]``. A shift shared
    across slots would cancel in the separate prefix softmax (spec §5.1), so
    the shift is per slot by construction.

    Gates ``(n_layers, SITES, n_heads)`` start at zero: the only zero factor.
    ``shift_out`` is small but nonzero so the generator receives gradient as
    soon as a gate opens.
    """

    z_sum: torch.Tensor
    z_count: torch.Tensor

    def __init__(self, d_model: int, n_layers: int, n_heads: int, cfg: PrefixConfig) -> None:
        """Build the parameters.

        Args:
            d_model: Trunk width.
            n_layers: Number of frozen cross-attention layers.
            n_heads: Heads per attention site.
            cfg: The prefix block.
        """
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.p0 = nn.ParameterList(
            nn.Parameter(torch.randn(cfg.tokens, d_model) * 0.02) for _ in range(n_layers)
        )
        self.gates = nn.Parameter(torch.zeros(n_layers, SITES, n_heads))
        if cfg.conditioning == "pair":
            self.cond_norm = nn.LayerNorm(3 * d_model)
            self.cond_proj = nn.Linear(3 * d_model, cfg.bottleneck)
            self.shift_in = nn.ModuleList(
                nn.Linear(cfg.bottleneck, cfg.tokens * cfg.rank) for _ in range(n_layers)
            )
            self.shift_out = nn.ParameterList(
                nn.Parameter(torch.randn(cfg.rank, d_model) * 0.02) for _ in range(n_layers)
            )
        self.register_buffer("z_sum", torch.zeros(cfg.bottleneck))
        self.register_buffer("z_count", torch.zeros(()))

    @property
    def z_mean(self) -> torch.Tensor:
        """Running mean of ``z`` over training forwards (zeros before any)."""
        return self.z_sum / self.z_count.clamp_min(1.0)

    def condition(
        self,
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
        mask_a: torch.Tensor | None,
        mask_b: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Return ``z`` of shape ``(B, bottleneck)``, or ``None`` when static.

        Args:
            encoded_a: Frozen encoder output for item A ``(B, L_a, d_model)``.
            encoded_b: Frozen encoder output for item B ``(B, L_b, d_model)``.
            mask_a: Padding mask for A (True = PAD) or ``None``.
            mask_b: Padding mask for B (True = PAD) or ``None``.

        Returns:
            The symmetric pair condition, or ``None`` for the static prefix.
        """
        if self.cfg.conditioning != "pair":
            return None
        e_a = masked_mean(encoded_a, inner_token_mask(x=encoded_a, padding_mask=mask_a))
        e_b = masked_mean(encoded_b, inner_token_mask(x=encoded_b, padding_mask=mask_b))
        c = torch.cat([e_a + e_b, (e_a - e_b).abs(), e_a * e_b], dim=-1)
        z = F.gelu(self.cond_proj(self.cond_norm(c)))
        if self.training:
            with torch.no_grad():
                self.z_sum += z.detach().float().sum(dim=0)
                self.z_count += float(z.size(0))
        return z

    def prefix(self, layer_index: int, z: torch.Tensor | None, batch_size: int) -> torch.Tensor:
        """Return the prefix tokens for one layer, ``(B, tokens, d_model)``.

        Args:
            layer_index: Which frozen layer.
            z: The pair condition from `condition`, or ``None``.
            batch_size: Rows to expand a static prefix to.

        Returns:
            The (possibly pair-shifted) prefix tokens.
        """
        p0 = self.p0[layer_index]
        if z is None:
            return p0.unsqueeze(0).expand(batch_size, -1, -1)
        coeff = self.shift_in[layer_index](z).view(z.size(0), self.cfg.tokens, self.cfg.rank)
        delta = coeff @ self.shift_out[layer_index]
        return p0.unsqueeze(0) + delta

    def gate(self, layer_index: int, site: int) -> torch.Tensor:
        """Return the raw (pre-tanh) per-head gate for one site."""
        return self.gates[layer_index, site]

    @torch.no_grad()
    def init_static_from_tokens(self, tokens: torch.Tensor, seed: int) -> None:
        """Initialise every ``p0[l]`` from sampled real token states.

        Args:
            tokens: Candidate rows ``(N, d_model)``, ``N >= cfg.tokens``.
            seed: Draw seed (identical across ranks; DDP's construction-time
                broadcast makes rank 0's draw authoritative anyway).

        Raises:
            ValueError: If fewer than ``cfg.tokens`` rows are supplied.
        """
        if tokens.size(0) < self.cfg.tokens:
            raise ValueError("init_static_from_tokens needs at least cfg.tokens rows")
        gen = torch.Generator(device="cpu").manual_seed(seed)
        for layer_index in range(self.n_layers):
            idx = torch.randperm(tokens.size(0), generator=gen)[: self.cfg.tokens]
            self.p0[layer_index].copy_(tokens[idx].to(self.p0[layer_index].dtype))
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_prefix_model.py -n0 -v`
Expected: 5 PASS.

- [ ] **Step 5: Lint and type-check, then commit**

Run: `.venv/bin/python -m ruff check src/model/egostitch/classifier/prefix.py tests/test_prefix_model.py && .venv/bin/python -m ruff format src/model/egostitch/classifier/prefix.py tests/test_prefix_model.py && .venv/bin/python -m mypy src/model/egostitch/classifier/prefix.py tests/test_prefix_model.py`
Expected: clean.

```bash
git add src/model/egostitch/classifier/prefix.py tests/test_prefix_model.py
git commit -m "feat(prefix): PrefixConfig and PrefixGenerator with slot-specific low-rank pair shift"
```

---

### Task 3: `prefix_branch` and `PrefixCrossAttentionLayer` with bitwise null identity

**Files:**
- Modify: `src/model/egostitch/classifier/prefix.py`
- Test: `tests/test_prefix_model.py`

**Interfaces:**
- Consumes: `PrefixGenerator.prefix/gate` (Task 2); `CrossAttentionLayer` from `layers.py` with attributes `norm_attn`, `attn`, `drop_attn`, `_ffn`, `norm_cls_attn`, `attn_cls`, `drop_cls_attn`, `ff_cls`, `norm_cls_ffn`, `drop_cls_ffn`.
- Produces:
  - `prefix_branch(mha: nn.MultiheadAttention, query_norm: torch.Tensor, prefix: torch.Tensor, gate: torch.Tensor, gate_scale: float) -> torch.Tensor` of shape `(B, T_q, d_model)`; exactly zero when `gate` is zero or `gate_scale == 0.0`; applies `out_proj.weight` **without** its bias.
  - `PrefixCrossAttentionLayer(layer: CrossAttentionLayer, layer_index: int, generator: PrefixGenerator)` with `forward(h_a, h_b, cls_token, mask_a, mask_b, z, gate_scale) -> (h_a, h_b, cls_token)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prefix_model.py`:

```python
from src.model.egostitch.classifier.layers import CrossAttentionLayer
from src.model.egostitch.classifier.prefix import PrefixCrossAttentionLayer, prefix_branch


def _frozen_layer(seed: int = 0) -> CrossAttentionLayer:
    torch.manual_seed(seed)
    layer = CrossAttentionLayer(d_model=8, n_heads=2, dropout=0.1)
    layer.eval()
    for param in layer.parameters():
        param.requires_grad_(False)
    return layer


def test_prefix_branch_is_exactly_zero_at_zero_gate_and_nonzero_otherwise() -> None:
    layer = _frozen_layer()
    query = torch.randn(3, 5, 8)
    prefix = torch.randn(3, 4, 8)
    zero = prefix_branch(layer.attn, query, prefix, torch.zeros(2), gate_scale=1.0)
    assert torch.count_nonzero(zero) == 0
    scaled_off = prefix_branch(layer.attn, query, prefix, torch.ones(2), gate_scale=0.0)
    assert torch.count_nonzero(scaled_off) == 0
    live = prefix_branch(layer.attn, query, prefix, torch.ones(2), gate_scale=1.0)
    assert live.shape == (3, 5, 8) and torch.count_nonzero(live) > 0


def test_prefix_layer_reproduces_frozen_layer_bitwise_at_zero_gates() -> None:
    layer = _frozen_layer()
    cfg = PrefixConfig(tokens=4, rank=2, conditioning="pair", bottleneck=6)
    gen = PrefixGenerator(d_model=8, n_layers=1, n_heads=2, cfg=cfg)
    wrapped = PrefixCrossAttentionLayer(layer, 0, gen)
    h_a, h_b = torch.randn(3, 5, 8), torch.randn(3, 4, 8)
    cls = torch.randn(3, 1, 8)
    mask_a = torch.tensor([[False] * 5, [False] * 4 + [True], [False] * 3 + [True] * 2])
    mask_b = torch.tensor([[False] * 4, [False] * 4, [False] * 2 + [True] * 2])
    z = gen.condition(h_a, h_b, mask_a, mask_b)
    base = layer(h_a, h_b, cls, mask_a, mask_b)
    ours = wrapped(h_a, h_b, cls, mask_a, mask_b, z, gate_scale=1.0)
    for got, want in zip(ours, base, strict=True):
        assert torch.equal(got, want)


def test_prefix_layer_changes_output_once_a_gate_opens_and_grads_stay_on_prefix() -> None:
    layer = _frozen_layer()
    cfg = PrefixConfig(tokens=4, rank=2, conditioning="pair", bottleneck=6)
    gen = PrefixGenerator(d_model=8, n_layers=1, n_heads=2, cfg=cfg)
    with torch.no_grad():
        gen.gates[0, 0, 0] = 0.5
    wrapped = PrefixCrossAttentionLayer(layer, 0, gen)
    h_a, h_b, cls = torch.randn(3, 5, 8), torch.randn(3, 4, 8), torch.randn(3, 1, 8)
    z = gen.condition(h_a, h_b, None, None)
    out_a, _, _ = wrapped(h_a, h_b, cls, None, None, z, gate_scale=1.0)
    base_a, _, _ = layer(h_a, h_b, cls, None, None)
    assert not torch.equal(out_a, base_a)
    out_a.sum().backward()
    assert all(p.grad is None for p in layer.parameters())
    assert gen.p0[0].grad is not None and gen.gates.grad is not None
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_prefix_model.py -n0 -v -k "prefix_branch or prefix_layer"`
Expected: FAIL with `ImportError: cannot import name 'PrefixCrossAttentionLayer'`.

- [ ] **Step 3: Implement the branch and the layer wrapper**

Append to `src/model/egostitch/classifier/prefix.py` (add `from src.model.egostitch.classifier.layers import CrossAttentionLayer` to the imports):

```python
def prefix_branch(
    mha: nn.MultiheadAttention,
    query_norm: torch.Tensor,
    prefix: torch.Tensor,
    gate: torch.Tensor,
    gate_scale: float,
) -> torch.Tensor:
    """Separately-softmaxed, tanh-gated attention of ``query_norm`` over ``prefix``.

    Uses the frozen ``mha``'s packed ``in_proj_weight``/``in_proj_bias`` slices
    for Q/K/V and ``out_proj.weight`` **without** its bias (the base call already
    added that bias), so the branch is an exact zero tensor whenever the gate is
    zero and the caller's ``base + branch`` reproduces the base bit for bit.

    Args:
        mha: The frozen attention module of the site (``batch_first=True``).
        query_norm: The site's normalised queries ``(B, T_q, E)``.
        prefix: Prefix tokens ``(B, m, E)``.
        gate: Raw per-head gate ``(H,)``; applied as ``tanh(gate)``.
        gate_scale: Multiplier on the gate (``0.0`` realises the gates-off intervention).

    Returns:
        The gated branch output ``(B, T_q, E)``.
    """
    embed = int(mha.embed_dim)
    heads = int(mha.num_heads)
    head_dim = embed // heads
    weight = cast(torch.Tensor, mha.in_proj_weight)
    bias = cast(torch.Tensor | None, mha.in_proj_bias)
    q = F.linear(query_norm, weight[:embed], None if bias is None else bias[:embed])
    k = F.linear(prefix, weight[embed : 2 * embed], None if bias is None else bias[embed : 2 * embed])
    v = F.linear(prefix, weight[2 * embed :], None if bias is None else bias[2 * embed :])
    batch, t_q, _ = q.shape
    q = q.view(batch, t_q, heads, head_dim).transpose(1, 2)
    k = k.view(batch, -1, heads, head_dim).transpose(1, 2)
    v = v.view(batch, -1, heads, head_dim).transpose(1, 2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
    out = torch.softmax(scores.float(), dim=-1).to(v.dtype) @ v
    out = out * (torch.tanh(gate) * gate_scale).to(out.dtype).view(1, heads, 1, 1)
    out = out.transpose(1, 2).reshape(batch, t_q, embed)
    return F.linear(out, cast(torch.Tensor, mha.out_proj.weight))


class PrefixCrossAttentionLayer(nn.Module):
    """Re-drive one frozen `CrossAttentionLayer` with a gated prefix at each site.

    The frozen layer's own sub-modules are called unchanged (its attention,
    dropouts, norms, and FFNs); the prefix branch is added *outside* the
    dropout so that at zero gate the sum is the base value exactly.
    """

    def __init__(
        self, layer: CrossAttentionLayer, layer_index: int, generator: PrefixGenerator
    ) -> None:
        """Wrap a frozen layer.

        Args:
            layer: The frozen `CrossAttentionLayer` (eval mode, no grad).
            layer_index: Its index in the trunk.
            generator: The shared prefix parameters.
        """
        super().__init__()
        self.layer = layer
        self.layer_index = layer_index
        self.generator = generator

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
            layer.attn, query_norm, prefix, self.generator.gate(self.layer_index, site), gate_scale
        )
        return query + cast(torch.Tensor, layer.drop_attn(attn_out)) + branch

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        cls_token: torch.Tensor,
        mask_a: torch.Tensor | None,
        mask_b: torch.Tensor | None,
        z: torch.Tensor | None,
        gate_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One bidirectional block plus the prefix branch at all three sites.

        Args:
            h_a: Item A hidden states.
            h_b: Item B hidden states.
            cls_token: CLS state ``(B, 1, d_model)``.
            mask_a: Padding mask for A (True = PAD) or ``None``.
            mask_b: Padding mask for B (True = PAD) or ``None``.
            z: Pair condition or ``None`` (static).
            gate_scale: ``0.0`` realises the gates-off intervention.

        Returns:
            Updated ``(h_a, h_b, cls_token)``.
        """
        layer = self.layer
        prefix = self.generator.prefix(self.layer_index, z, h_a.size(0))
        h_a = self._attend(0, h_a, h_b, mask_b, prefix, gate_scale)
        h_a = layer._ffn(h_a)
        h_b = self._attend(1, h_b, h_a, mask_a, prefix, gate_scale)
        h_b = layer._ffn(h_b)

        combined = torch.cat([h_a, h_b], dim=1)
        combined_mask = (
            torch.cat([mask_a, mask_b], dim=1)
            if mask_a is not None and mask_b is not None
            else None
        )
        cls_norm = layer.norm_cls_attn(cls_token)
        attn_cls, _ = layer.attn_cls(
            cls_norm, combined, combined, key_padding_mask=combined_mask, need_weights=False
        )
        branch = prefix_branch(
            layer.attn_cls, cls_norm, prefix, self.generator.gate(self.layer_index, 2), gate_scale
        )
        cls_token = cls_token + layer.drop_cls_attn(attn_cls) + branch
        cls_token = cls_token + layer.drop_cls_ffn(layer.ff_cls(layer.norm_cls_ffn(cls_token)))
        return h_a, h_b, cls_token
```

Note on `layer._ffn`: it is a private method of `CrossAttentionLayer`; calling it from this module is deliberate (the wrapper is the layer's only re-driver) and mirrors the original forward line for line. If ruff flags the private access, add `# noqa: SLF001` on those two lines.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_prefix_model.py -n0 -v`
Expected: 8 PASS. The bitwise test relies on `layer.eval()`; if it fails only in the dropout-bearing sites, the layer was left in train mode — the fixture calls `eval()`, so check the wrapper does not call `train()` on it.

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/python -m ruff check src/model/egostitch/classifier/prefix.py tests/test_prefix_model.py && .venv/bin/python -m ruff format src/model/egostitch/classifier/prefix.py tests/test_prefix_model.py && .venv/bin/python -m mypy src/model/egostitch/classifier/prefix.py tests/test_prefix_model.py`

```bash
git add src/model/egostitch/classifier/prefix.py tests/test_prefix_model.py
git commit -m "feat(prefix): gated prefix branch and layer wrapper with bitwise null identity"
```

---

### Task 4: `V3_1Prefix` family wrapper (forward, freezing, interventions)

**Files:**
- Modify: `src/model/egostitch/classifier/prefix.py`
- Modify: `src/model/egostitch/classifier/__init__.py` (only if it re-exports classifier classes; check with `grep -n "V3_1" src/model/egostitch/classifier/__init__.py`)
- Test: `tests/test_prefix_model.py`

**Interfaces:**
- Consumes: `V3_1` (`src/model/egostitch/classifier/b0_v31.py`) with attributes `encoder`, `cross_attention` (`PairCrossAttention` with `layers`, `cls_token`, `pair_readout_mode`, `pair_context_readout`, `_rich_pooling_readout`, `grid_sketch_readout`, `mixing_mode`), `output_head`, `order_aggregation`, `label_smoothing`, `positive_weight`, `input_dim`, `d_model`; `_build_padding_mask` from `layers.py`.
- Produces: `V3_1Prefix(*, base: Mapping[str, object], prefix: Mapping[str, object])` with
  - `self.generator: PrefixGenerator` (registered **before** `self.base` so `next(model.parameters())` is trainable),
  - `self.base: V3_1` frozen, eval mode enforced by an overridden `train()`,
  - `self.prefix_layers: nn.ModuleList[PrefixCrossAttentionLayer]`,
  - `intervention: str` in `("none", "gates_off", "shuffle", "mean")`, `intervention_seed: int = 0`,
  - `prefix_parameters() -> list[nn.Parameter]`,
  - `init_static_prefix(batch: Mapping[str, torch.Tensor], seed: int) -> None`,
  - `forward(batch=None, **kwargs) -> dict[str, torch.Tensor]` returning `logits`, `pair_repr`, optional `loss`/`loss_weight_sum` exactly as `V3_1`,
  - attributes `d_model`, `input_dim`, `kd_rep_head = None`, `kd_struct_head = None`, `topo_gen = None` (the trainer reads them by `getattr`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prefix_model.py` (the fixture helpers `_tiny_base_config` and `_pair_batch` from the file map must already be in this file; add them now if not):

```python
from src.model.egostitch.classifier.b0_v31 import V3_1
from src.model.egostitch.classifier.prefix import V3_1Prefix


def _trained_base(seed: int = 1) -> V3_1:
    torch.manual_seed(seed)
    base = V3_1(**_tiny_base_config())
    with torch.no_grad():  # perturb so the frozen function is not the init
        for param in base.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    return base


def _wrapped(conditioning: str = "pair", seed: int = 1) -> tuple[V3_1, V3_1Prefix]:
    base = _trained_base(seed)
    model = V3_1Prefix(
        base=_tiny_base_config(),
        prefix={"tokens": 3, "rank": 2, "conditioning": conditioning, "bottleneck": 6},
    )
    model.base.load_state_dict(base.state_dict())
    return base, model


def test_wrapper_rejects_a_base_without_cross_attention_layers() -> None:
    try:
        V3_1Prefix(base=_tiny_base_config(mixing="none"), prefix={"tokens": 2})
    except ValueError as err:
        assert "bidirectional_cross" in str(err)
    else:
        raise AssertionError("mixing none must be rejected")


def test_null_identity_against_the_frozen_base_in_both_modes() -> None:
    for conditioning in ("static", "pair"):
        base, model = _wrapped(conditioning)
        base.eval()
        batch = _pair_batch()
        with torch.no_grad():
            want = base(batch)["logits"]
            model.train()
            got_train = model(batch)["logits"]
            model.eval()
            got_eval = model(batch)["logits"]
        assert torch.equal(got_train, want), conditioning
        assert torch.equal(got_eval, want), conditioning


def test_base_stays_in_eval_mode_and_frozen_after_wrapper_train() -> None:
    _, model = _wrapped()
    model.train()
    assert model.training
    assert not model.base.training
    assert all(not m.training for m in model.base.modules())
    assert all(not p.requires_grad for p in model.base.parameters())
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable and all(name.startswith("generator.") for name in trainable)
    assert {id(p) for p in model.prefix_parameters()} == {
        id(p) for p in model.generator.parameters()
    }


def test_loss_backward_reaches_only_prefix_parameters() -> None:
    _, model = _wrapped()
    model.train()
    with torch.no_grad():
        model.generator.gates.fill_(0.3)
    out = model(_pair_batch())
    assert "loss" in out and "loss_weight_sum" in out
    out["loss"].backward()
    assert all(p.grad is None for p in model.base.parameters())
    assert all(p.grad is not None for p in model.generator.parameters())


def test_pair_symmetry_holds_with_open_gates() -> None:
    _, model = _wrapped()
    model.eval()
    with torch.no_grad():
        model.generator.gates.fill_(0.4)
    batch = _pair_batch()
    swapped = {"emb_a": batch["emb_b"], "emb_b": batch["emb_a"],
               "len_a": batch["len_b"], "len_b": batch["len_a"]}
    with torch.no_grad():
        assert torch.allclose(model(batch)["logits"], model(swapped)["logits"], atol=1e-5)


def test_interventions() -> None:
    base, model = _wrapped()
    base.eval()
    model.eval()
    with torch.no_grad():
        model.generator.gates.fill_(0.4)
        model.generator.z_sum.copy_(torch.randn(6))
        model.generator.z_count.fill_(10.0)
    batch = _pair_batch()
    with torch.no_grad():
        live = model(batch)["logits"]
        model.intervention = "gates_off"
        assert torch.equal(model(batch)["logits"], base(batch)["logits"])
        model.intervention = "mean"
        mean_logits = model(batch)["logits"]
        assert not torch.equal(mean_logits, live)
        model.intervention = "shuffle"
        model.intervention_seed = 5
        shuffled_1 = model(batch)["logits"]
        model.intervention_seed = 5
        shuffled_2 = model(batch)["logits"]
        assert torch.equal(shuffled_1, shuffled_2)
        assert not torch.equal(shuffled_1, live)
        # the same permutation is applied to AB and BA (abba_max stays symmetric)
        swapped = {"emb_a": batch["emb_b"], "emb_b": batch["emb_a"],
                   "len_a": batch["len_b"], "len_b": batch["len_a"]}
        model.intervention_seed = 5
        assert torch.allclose(model(swapped)["logits"], shuffled_1, atol=1e-5)
    model.intervention = "bogus"
    try:
        with torch.no_grad():
            model(batch)
    except ValueError as err:
        assert "intervention" in str(err)
    else:
        raise AssertionError("unknown intervention must raise")


def test_init_static_prefix_from_a_batch_and_state_dict_round_trip() -> None:
    _, model = _wrapped()
    before = model.generator.p0[0].clone()
    model.init_static_prefix(_pair_batch(), seed=0)
    assert not torch.equal(before, model.generator.p0[0])
    rebuilt = V3_1Prefix(
        base=_tiny_base_config(),
        prefix={"tokens": 3, "rank": 2, "conditioning": "pair", "bottleneck": 6},
    )
    rebuilt.load_state_dict(model.state_dict())
    rebuilt.eval()
    model.eval()
    with torch.no_grad():
        assert torch.equal(rebuilt(_pair_batch())["logits"], model(_pair_batch())["logits"])
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_prefix_model.py -n0 -v -k "wrapper or null_identity or eval_mode or backward or symmetry or interventions or round_trip"`
Expected: FAIL with `ImportError: cannot import name 'V3_1Prefix'`.

- [ ] **Step 3: Implement `V3_1Prefix`**

Append to `src/model/egostitch/classifier/prefix.py` (add imports `from src.model.egostitch.classifier.b0_v31 import V3_1` and `from src.model.egostitch.classifier.layers import _build_padding_mask`; `_build_padding_mask` is already imported by `b0_v31.py` from `layers.py`, so the private import is established practice):

```python
INTERVENTIONS = ("none", "gates_off", "shuffle", "mean")


class V3_1Prefix(nn.Module):
    """A frozen `V3_1` (bidirectional-cross trunk) plus a trainable gated prefix.

    Registration order matters: ``generator`` is registered before ``base`` so
    ``next(model.parameters())`` is a trainable parameter (the structural stream
    builds its zero-loss anchor from it).
    """

    name: str = "v3_1_prefix"

    def __init__(self, *, base: Mapping[str, object], prefix: Mapping[str, object]) -> None:
        """Build the frozen base from its config and the prefix on top.

        Args:
            base: The base `V3_1` constructor kwargs (a checkpoint's ``model_config``).
            prefix: The ``model.config.prefix`` block.

        Raises:
            ValueError: If the base trunk has no bidirectional cross-attention layers.
        """
        super().__init__()
        self.prefix_cfg = PrefixConfig.from_mapping(prefix)
        self.base_config: dict[str, object] = dict(base)
        base_model = V3_1(**self.base_config)
        trunk = base_model.cross_attention
        if trunk.mixing_mode != "bidirectional_cross" or len(trunk.layers) == 0:
            raise ValueError(
                "v3_1_prefix needs a base with model.config.mixing.mode == 'bidirectional_cross'"
            )
        self.d_model = int(base_model.d_model)
        self.input_dim = int(base_model.input_dim)
        self.kd_rep_head = None
        self.kd_struct_head = None
        self.topo_gen = None
        self.generator = PrefixGenerator(
            self.d_model, len(trunk.layers), int(base_model.n_heads), self.prefix_cfg
        )
        self.base = base_model
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.base.eval()
        self.prefix_layers = nn.ModuleList(
            PrefixCrossAttentionLayer(cast(CrossAttentionLayer, layer), index, self.generator)
            for index, layer in enumerate(trunk.layers)
        )
        self.intervention: str = "none"
        self.intervention_seed: int = 0

    def train(self, mode: bool = True) -> V3_1Prefix:
        """Switch the wrapper's mode while keeping the frozen base in eval mode.

        Args:
            mode: Training mode for the prefix parameters.

        Returns:
            ``self``.
        """
        super().train(mode)
        self.base.eval()
        return self

    def prefix_parameters(self) -> list[nn.Parameter]:
        """The only trainable parameters: everything in `PrefixGenerator`."""
        return list(self.generator.parameters())

    @torch.no_grad()
    def init_static_prefix(self, batch: Mapping[str, torch.Tensor], seed: int) -> None:
        """Initialise ``p0`` from the frozen encoder's inner-token states of ``batch``.

        Args:
            batch: A task batch with ``emb_a``/``emb_b`` (and optional lengths).
            seed: Draw seed.
        """
        emb_a, emb_b, len_a, len_b = self._unpack(batch)
        rows: list[torch.Tensor] = []
        for emb, lengths in ((emb_a, len_a), (emb_b, len_b)):
            encoded = self.base.encoder(emb, lengths)
            keep = inner_token_mask(x=encoded, padding_mask=_build_padding_mask(lengths, encoded.size(1)))
            rows.append(encoded[keep])
        self.generator.init_static_from_tokens(torch.cat(rows, dim=0).to(self.generator.p0[0].device), seed)

    def _unpack(
        self, merged: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if "emb_a" not in merged or "emb_b" not in merged:
            raise KeyError("Batch must contain 'emb_a' and 'emb_b' tensors")
        emb_a, emb_b = merged["emb_a"], merged["emb_b"]
        if emb_a.dim() != 3 or emb_b.dim() != 3:
            raise ValueError("Input embeddings must be shaped (batch, seq_len, embedding_dim)")
        if emb_a.size(2) != self.input_dim or emb_b.size(2) != self.input_dim:
            raise ValueError("Input embedding dimension must match model input_dim")
        if emb_a.size(0) != emb_b.size(0):
            raise ValueError("Item pair batches must have matching batch dimension")
        device = emb_a.device
        len_a = merged.get("len_a")
        len_b = merged.get("len_b")
        lengths_a = (
            torch.full((emb_a.size(0),), emb_a.size(1), device=device, dtype=torch.long)
            if len_a is None
            else len_a.to(device=device, dtype=torch.long)
        )
        lengths_b = (
            torch.full((emb_b.size(0),), emb_b.size(1), device=device, dtype=torch.long)
            if len_b is None
            else len_b.to(device=device, dtype=torch.long)
        )
        return emb_a, emb_b, lengths_a, lengths_b

    def _apply_intervention(self, z: torch.Tensor | None) -> tuple[torch.Tensor | None, float]:
        """Return the (possibly substituted) condition and the gate scale.

        Raises:
            ValueError: On an unknown intervention name.
        """
        if self.intervention not in INTERVENTIONS:
            raise ValueError(f"unknown prefix intervention {self.intervention!r}")
        if self.intervention == "gates_off":
            return z, 0.0
        if z is None or self.intervention == "none":
            return z, 1.0
        if self.intervention == "mean":
            return self.generator.z_mean.to(z.dtype).unsqueeze(0).expand_as(z), 1.0
        gen = torch.Generator(device="cpu").manual_seed(self.intervention_seed)
        perm = torch.randperm(z.size(0), generator=gen).to(z.device)
        return z[perm], 1.0

    def _trunk(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
        z: torch.Tensor | None,
        gate_scale: float,
    ) -> torch.Tensor:
        trunk = self.base.cross_attention
        mask_a = _build_padding_mask(lengths_a, h_a.size(1))
        mask_b = _build_padding_mask(lengths_b, h_b.size(1))
        cls_token = trunk.cls_token.repeat(h_a.size(0), 1, 1)
        for layer in self.prefix_layers:
            h_a, h_b, cls_token = layer(h_a, h_b, cls_token, mask_a, mask_b, z, gate_scale)
        cls_vec = cls_token.squeeze(1)
        if trunk.pair_readout_mode == "pair_context_gated":
            return cast(torch.Tensor, trunk.pair_context_readout(h_a, h_b, cls_vec, mask_a, mask_b))
        base_repr = trunk._rich_pooling_readout(h_a, h_b, cls_vec, mask_a, mask_b)
        if trunk.pair_readout_mode == "grid_sketch_fusion":
            return cast(torch.Tensor, trunk.grid_sketch_readout(base_repr, h_a, h_b, mask_a, mask_b))
        return base_repr

    def forward(
        self,
        batch: dict[str, torch.Tensor] | None = None,
        **kwargs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Score pairs with the frozen base read through the prefix.

        Args:
            batch: Optional batch dictionary.
            **kwargs: Additional batch tensors merged into ``batch``.

        Returns:
            ``logits``, ``pair_repr``, and, with ``label``, the weighted BCE ``loss``
            and ``loss_weight_sum`` exactly as `V3_1` computes them.
        """
        merged: dict[str, torch.Tensor] = {}
        if batch is not None:
            merged.update(batch)
        merged.update(kwargs)
        emb_a, emb_b, lengths_a, lengths_b = self._unpack(merged)
        with torch.no_grad():
            encoded_a = self.base.encoder(emb_a, lengths_a)
            encoded_b = self.base.encoder(emb_b, lengths_b)
        mask_a = _build_padding_mask(lengths_a, encoded_a.size(1))
        mask_b = _build_padding_mask(lengths_b, encoded_b.size(1))
        z, gate_scale = self._apply_intervention(
            self.generator.condition(encoded_a, encoded_b, mask_a, mask_b)
        )
        feature_ab = self._trunk(encoded_a, encoded_b, lengths_a, lengths_b, z, gate_scale)
        if self.base.order_aggregation == "single":
            pair_repr = feature_ab
        else:
            feature_ba = self._trunk(encoded_b, encoded_a, lengths_b, lengths_a, z, gate_scale)
            pair_repr = torch.max(torch.stack([feature_ab, feature_ba], dim=-1), dim=-1).values
        logits = self.base.output_head(pair_repr)
        output: dict[str, torch.Tensor] = {"pair_repr": pair_repr, "logits": logits}
        if "label" in merged:
            labels = merged["label"].float()
            logits_for_loss = (
                logits.squeeze(-1) if logits.dim() > 1 and logits.size(-1) == 1 else logits
            )
            labels_for_loss = (
                labels.squeeze(-1) if labels.dim() > 1 and labels.size(-1) == 1 else labels
            )
            smoothing = float(self.base.label_smoothing)
            if smoothing > 0.0:
                labels_for_loss = labels_for_loss * (1.0 - smoothing) + 0.5 * smoothing
            per_row = F.binary_cross_entropy_with_logits(
                logits_for_loss.float(), labels_for_loss.float(), reduction="none"
            )
            weights = 1.0 + (float(self.base.positive_weight) - 1.0) * labels.reshape_as(per_row)
            denominator = weights.sum().detach()
            output["loss"] = (weights * per_row).sum() / denominator
            output["loss_weight_sum"] = denominator
        return output
```

The base `V3_1` exposes `n_heads`; if the attribute is missing, read `int(self.base_config["n_heads"])` instead.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_prefix_model.py -n0 -v`
Expected: all PASS. Two failure modes to watch: (a) `test_null_identity...` fails in `train()` mode — dropout leaked because the base was not re-`eval()`ed after `super().train()`; (b) `test_interventions` shuffle-symmetry fails — the permutation must be drawn once per forward from `intervention_seed`, not once per `_trunk` call.

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/python -m ruff check src/model/egostitch/classifier/prefix.py tests/test_prefix_model.py && .venv/bin/python -m ruff format src/model/egostitch/classifier/prefix.py tests/test_prefix_model.py && .venv/bin/python -m mypy src/model/egostitch/classifier/prefix.py tests/test_prefix_model.py`

```bash
git add src/model/egostitch/classifier/prefix.py tests/test_prefix_model.py
git commit -m "feat(prefix): V3_1Prefix frozen-base wrapper with interventions"
```

---

### Task 5: Family dispatch in `train_b0`

**Files:**
- Modify: `src/train_b0.py` — `MODEL_FAMILIES` (line ~126), `resolve_model_kwargs` (~990-1015), `build_model` (~1018-1045), `_build_optimizer` (~570-595), family gates at ~1036, ~5046, ~5401, and the static-prefix init right after `_build_v3_1_loaders` (~5402).
- Test: `tests/test_train_b0_prefix.py`

**Interfaces:**
- Consumes: `V3_1Prefix`, `PrefixConfig` (Tasks 2–4).
- Produces:
  - `is_v3_1_family(family: str) -> bool` (true for `"v3_1"` and `"v3_1_prefix"`), used at every former `== "v3_1"` gate.
  - `resolve_model_kwargs` for `v3_1_prefix` returns `{"base": <base checkpoint's model_config>, "prefix": {...block..., "base_checkpoint": str(path), "base_checkpoint_sha256": <hex>}}`; the YAML block must name `prefix.base_checkpoint`.
  - `build_model` for `v3_1_prefix` constructs `V3_1Prefix(**kwargs)` and loads the base checkpoint's `model_state` into `model.base` (strict).
  - `_build_optimizer` for a model exposing `prefix_parameters` uses only those.
  - `_init_prefix_from_loader(model, val_loader, seed)` called before the accelerator prepares the model.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train_b0_prefix.py` (copy `_tiny_base_config` and `_pair_batch` from the file map; import `Path`, `torch`, `yaml`, `json`):

```python
"""train_b0 dispatch for the v3_1_prefix family."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from src.model.egostitch.classifier.b0_v31 import V3_1
from src.model.egostitch.classifier.prefix import V3_1Prefix
from src.train_b0 import (
    MODEL_FAMILIES,
    _build_optimizer,
    build_model,
    is_v3_1_family,
    load_config,
    resolve_model_kwargs,
)
from tests.test_train_b0 import _write_yaml_config


def _base_checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    base = V3_1(**_tiny_base_config())
    payload = {
        "model_state": base.state_dict(),
        "model_family": "v3_1",
        "model_config": _tiny_base_config(),
        "epoch": 1,
        "val_metrics": {},
        "seed": 0,
        "config": {},
    }
    path = tmp_path / "best.pt"
    torch.save(payload, path)
    return path


def _prefix_yaml(tmp_path: Path, conditioning: str = "pair") -> Path:
    config_path = tmp_path / "prefix.yaml"
    _write_yaml_config(
        config_path,
        {
            "model": {
                "family": "v3_1_prefix",
                "config": {
                    "prefix": {
                        "tokens": 3,
                        "rank": 2,
                        "conditioning": conditioning,
                        "bottleneck": 6,
                        "base_checkpoint": str(_base_checkpoint(tmp_path)),
                    }
                },
            }
        },
    )
    return config_path


def test_family_is_registered_and_grouped_with_v3_1() -> None:
    assert "v3_1_prefix" in MODEL_FAMILIES
    assert is_v3_1_family("v3_1") and is_v3_1_family("v3_1_prefix")
    assert not is_v3_1_family("f0_mlp")


def test_resolve_embeds_base_config_and_provenance(tmp_path: Path) -> None:
    cfg = load_config(_prefix_yaml(tmp_path))
    kwargs = resolve_model_kwargs(cfg.model)
    assert kwargs["base"] == _tiny_base_config()
    prefix = kwargs["prefix"]
    assert isinstance(prefix, dict)
    assert prefix["base_checkpoint"].endswith("best.pt")
    assert isinstance(prefix["base_checkpoint_sha256"], str) and len(prefix["base_checkpoint_sha256"]) == 64
    json.dumps(kwargs)  # checkpoint-embeddable


def test_build_model_loads_and_freezes_the_base(tmp_path: Path) -> None:
    cfg = load_config(_prefix_yaml(tmp_path))
    model = build_model(cfg)
    assert isinstance(model, V3_1Prefix)
    saved = torch.load(cfg.model.config["prefix"]["base_checkpoint"], map_location="cpu")  # type: ignore[index]
    for name, tensor in saved["model_state"].items():
        assert torch.equal(model.base.state_dict()[name], tensor)
    assert all(not p.requires_grad for p in model.base.parameters())
    optimizer = _build_optimizer(model, cfg)
    ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert ids == {id(p) for p in model.prefix_parameters()}


def test_missing_base_checkpoint_key_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    _write_yaml_config(
        config_path, {"model": {"family": "v3_1_prefix", "config": {"prefix": {"tokens": 3}}}}
    )
    try:
        resolve_model_kwargs(load_config(config_path).model)
    except ValueError as err:
        assert "base_checkpoint" in str(err)
    else:
        raise AssertionError("prefix.base_checkpoint is required")
```

Check `tests/test_train_b0.py::_write_yaml_config` merges a partial mapping over a valid default config (it is used that way by `test_unknown_family_raises_clear_error`); if it replaces `model` wholesale, the tests above still work because the whole `model` block is supplied.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_train_b0_prefix.py -n0 -v`
Expected: FAIL with `ImportError: cannot import name 'is_v3_1_family'`.

- [ ] **Step 3: Implement the dispatch**

In `src/train_b0.py`:

1. Imports: `hashlib` is already imported (used by `_state_digest`); add `Mapping` to the existing `from collections.abc import …` line, and add `from src.model.egostitch.classifier.prefix import PrefixConfig, V3_1Prefix`. `torch.load(..., weights_only=False)` matches the trainer's own resume loads.

2. Replace `MODEL_FAMILIES = ("v3_1", "f0_mlp")` with:

```python
MODEL_FAMILIES = ("v3_1", "v3_1_prefix", "f0_mlp")
V3_1_FAMILIES = frozenset({"v3_1", "v3_1_prefix"})


def is_v3_1_family(family: str) -> bool:
    """True for the packed-token student families (`V3_1` and its prefix wrapper)."""
    return family in V3_1_FAMILIES
```

3. In `resolve_model_kwargs`, before the `if model_cfg.family == "v3_1":` line, add:

```python
    if model_cfg.family == "v3_1_prefix":
        return _resolve_prefix_kwargs(model_cfg)
```

and add the helper next to it:

```python
def _resolve_prefix_kwargs(model_cfg: ModelConfig) -> dict[str, object]:
    """Embed the frozen base's ``model_config`` and record the base file's SHA-256.

    Provenance only: the digest is recorded in the checkpoint's ``model_config``
    (hence in ``run_metadata.json``) and never verified (project rule: no
    digest pinning).

    Raises:
        ValueError: If ``model.config.prefix.base_checkpoint`` is missing, or the
            base checkpoint is not a ``v3_1`` checkpoint.
    """
    raw_prefix = model_cfg.config.get("prefix")
    if not isinstance(raw_prefix, Mapping) or "base_checkpoint" not in raw_prefix:
        raise ValueError("model.config.prefix.base_checkpoint is required for v3_1_prefix")
    extra = sorted(set(model_cfg.config) - {"prefix"})
    if extra:
        raise ValueError(f"v3_1_prefix accepts only model.config.prefix, got {extra}")
    path = Path(str(raw_prefix["base_checkpoint"]))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model_family") != "v3_1":
        raise ValueError(f"{path}: prefix base must be a v3_1 checkpoint")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    prefix = PrefixConfig.from_mapping({**dict(raw_prefix), "base_checkpoint": str(path)})
    prefix = PrefixConfig.from_mapping({**prefix.to_dict(), "base_checkpoint_sha256": digest})
    return {"base": dict(cast(Mapping[str, object], payload["model_config"])), "prefix": prefix.to_dict()}
```

4. In `build_model`, before the `if cfg.model.family == "v3_1":` block:

```python
    if cfg.model.family == "v3_1_prefix":
        model = V3_1Prefix(**kwargs)  # type: ignore[arg-type]
        base_path = Path(str(cast(Mapping[str, object], kwargs["prefix"])["base_checkpoint"]))
        payload = torch.load(base_path, map_location="cpu", weights_only=False)
        model.base.load_state_dict(payload["model_state"])
        return model
```

5. In `_build_optimizer`, as the first statement after `raw_model = _unwrapped_model(model)`:

```python
    prefix_getter = getattr(raw_model, "prefix_parameters", None)
    if callable(prefix_getter):
        return torch.optim.AdamW(
            cast(list[nn.Parameter], prefix_getter()),
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
        )
```

6. Replace the three gates: `if cfg.model.family == "v3_1":` at `build_model`'s loaders selection (~line 5401) and at `_build_v3_1_loaders` dispatch (~1036 is inside `build_model` itself and is handled by step 4; check with `grep -n '== "v3_1"' src/train_b0.py`), and `if cfg.model.family != "v3_1":` at ~5046, with `is_v3_1_family(cfg.model.family)` / `not is_v3_1_family(cfg.model.family)`. After the edit, `grep -n '"v3_1"' src/train_b0.py` must show only the family-string constants, the `resolve_model_kwargs`/`build_model` branches, and the base-checkpoint check.

7. Static-prefix init. Right after `factory, val_loader = _build_v3_1_loaders(cfg, assembled)` (~5402) add:

```python
    if isinstance(model, V3_1Prefix):
        _init_prefix_from_loader(model, val_loader, cfg.seed)
```

with the helper:

```python
def _init_prefix_from_loader(model: V3_1Prefix, loader: Iterable[dict[str, torch.Tensor]], seed: int) -> None:
    """Draw ``p0`` from the frozen encoder's token states of the first validation batch.

    Runs before ``accelerator.prepare``; DDP's construction-time broadcast then
    makes rank 0's draw authoritative on every rank.
    """
    batch = next(iter(loader))
    model.init_static_prefix({k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}, seed)
```

Confirm the validation loader yields dicts with `emb_a`/`emb_b` (it feeds `V3_1.forward`, so it must). If the DDP worker path (~5046 onwards) builds its own loaders, add the same two lines there after its loader construction; find it with `grep -n "_build_v3_1_loaders\|packed.*loader" src/train_b0.py`.

- [ ] **Step 4: Run the tests plus the existing trainer tests**

Run: `.venv/bin/python -m pytest tests/test_train_b0_prefix.py tests/test_train_b0.py tests/test_train_b0_struct.py -n0 -q`
Expected: all PASS (the existing family-error tests still match on the family name string).

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/python -m ruff check src/train_b0.py tests/test_train_b0_prefix.py && .venv/bin/python -m ruff format src/train_b0.py tests/test_train_b0_prefix.py && .venv/bin/python -m mypy src/train_b0.py tests/test_train_b0_prefix.py`

```bash
git add src/train_b0.py tests/test_train_b0_prefix.py
git commit -m "feat(train): v3_1_prefix family loads and freezes the base, optimizes only the prefix"
```

---

### Task 6: Scoring and test-protocol support with `--prefix-intervention`

**Files:**
- Modify: `src/score_universe.py` — `MODEL_BUILDERS` (~1246), `_score_v3_1` dispatch (~3380), CLI (~3092), post-load hook (~3218), meta dicts (~3372, ~3464), merge check (~999-1008).
- Modify: `src/eval/test_protocol.py` — every `topo_gen_control` threading point (lines ~170-186, 318, 357-358, 385, 407, 517, 543, 604, CLI).
- Test: `tests/test_score_universe.py`

**Interfaces:**
- Consumes: `V3_1Prefix.intervention`, `intervention_seed`.
- Produces: `--prefix-intervention {none,gates_off,shuffle,mean}` (default `none`) and `--prefix-intervention-seed INT` (default 0) on `score_universe`; meta key `prefix_intervention` in every scores artifact; `test_protocol` accepts and forwards `--prefix-intervention`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_score_universe.py` (add `from src.model.egostitch.classifier.prefix import V3_1Prefix` and the two fixture helpers from the file map at the module bottom if not present):

```python
def test_prefix_family_round_trips_through_build_model(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = V3_1Prefix(
        base=_tiny_base_config(),
        prefix={"tokens": 3, "rank": 2, "conditioning": "pair", "bottleneck": 6},
    )
    with torch.no_grad():
        model.generator.gates.fill_(0.3)
    payload = {
        "model_state": model.state_dict(),
        "model_family": "v3_1_prefix",
        "model_config": {"base": _tiny_base_config(), "prefix": model.prefix_cfg.to_dict()},
    }
    rebuilt = score_universe.build_model(payload["model_family"], payload["model_config"])
    rebuilt.load_state_dict(payload["model_state"])
    rebuilt.eval()
    model.eval()
    batch = _pair_batch()
    with torch.no_grad():
        assert torch.equal(rebuilt(batch)["logits"], model(batch)["logits"])


def test_prefix_intervention_flag_is_parsed_and_defaults_to_none() -> None:
    parser = score_universe.build_parser()
    args = parser.parse_args(
        ["score", "--checkpoint", "x.pt", "--output", "y.npz", "--prefix-intervention", "shuffle",
         "--prefix-intervention-seed", "7"]
    )
    assert args.prefix_intervention == "shuffle" and args.prefix_intervention_seed == 7
    default = parser.parse_args(["score", "--checkpoint", "x.pt", "--output", "y.npz"])
    assert default.prefix_intervention == "none"


def test_merge_mismatched_prefix_intervention_raises_clear_error(tmp_path: Path) -> None:
    shard0 = tmp_path / "s0.npz"
    shard1 = tmp_path / "s1.npz"
    _write_fake_shard(shard0, row_start=0, n_rows=10, num_rows=20, prefix_intervention="none")
    _write_fake_shard(shard1, row_start=10, n_rows=10, num_rows=20, prefix_intervention="mean")

    with pytest.raises(ValueError, match="prefix_intervention"):
        score_universe.merge_scores([shard0, shard1])
```

`_write_fake_shard` is the module's existing helper (used by `test_merge_mismatched_topo_gen_control_raises_clear_error`); give it a `prefix_intervention: str = "none"` keyword that lands in the shard's meta the same way `topo_gen_control` does. The parser entry point is `score_universe.build_parser()`.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_score_universe.py -n0 -v -k "prefix"`
Expected: FAIL (`unknown model_family 'v3_1_prefix'`, unrecognized argument, no ValueError).

- [ ] **Step 3: Implement**

In `src/score_universe.py`:

1. Builder:

```python
def _build_v3_1_prefix(model_config: dict[str, object]) -> nn.Module:
    """Build a `V3_1Prefix` from its checkpointed config (frozen base state is in the checkpoint)."""
    from src.model.egostitch.classifier.prefix import V3_1Prefix

    return V3_1Prefix(**cast(dict[str, Any], model_config))
```

and register `"v3_1_prefix": _build_v3_1_prefix` in `MODEL_BUILDERS`.

2. Dispatch: change `if model_family == "v3_1":` (~3380) to `if model_family in ("v3_1", "v3_1_prefix"):`. Search the file for any other `== "v3_1"` and treat it the same way.

3. CLI, next to `--topo-gen-control`:

```python
    score.add_argument(
        "--prefix-intervention",
        choices=["none", "gates_off", "shuffle", "mean"],
        default="none",
        help="v3_1_prefix scoring-time intervention (spec §7)",
    )
    score.add_argument("--prefix-intervention-seed", type=int, default=0)
```

4. After the `topo_gen_control` block (~3218):

```python
    if args.prefix_intervention != "none":
        if model_family != "v3_1_prefix":
            raise SystemExit("--prefix-intervention requires a v3_1_prefix checkpoint")
        model.intervention = args.prefix_intervention
        model.intervention_seed = int(args.prefix_intervention_seed)
```

5. Meta: add `"prefix_intervention": args.prefix_intervention,` beside every `"topo_gen_control": args.topo_gen_control,` (two sites).

6. Merge check in `merge_scores` (~999-1008): duplicate the `topo_gen_control` block for `prefix_intervention`, but treat a *missing* key as `"none"` (artifacts scored before this change carry no key and must still merge); disagreement raises `ValueError` naming `prefix_intervention`.

In `src/eval/test_protocol.py`: for each of the nine `topo_gen_control` occurrences add a sibling `prefix_intervention: str` (default `"none"`): the artifact-meta check (`artifact.meta.get("prefix_intervention", "none") != prefix_intervention` → `ValueError`), the subprocess arg (`args += ["--prefix-intervention", prefix_intervention]`), the function parameters and docstrings, and a CLI `--prefix-intervention` with the same choices. Keep the default `"none"` so every existing caller is unchanged.

- [ ] **Step 4: Run the scorer and protocol tests**

Run: `.venv/bin/python -m pytest tests/test_score_universe.py tests/test_test_protocol*.py -n0 -q` (use `ls tests | grep protocol` for the exact file name).
Expected: all PASS.

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/python -m ruff check src/score_universe.py src/eval/test_protocol.py tests/test_score_universe.py && .venv/bin/python -m ruff format src/score_universe.py src/eval/test_protocol.py tests/test_score_universe.py && .venv/bin/python -m mypy src/score_universe.py src/eval/test_protocol.py tests/test_score_universe.py`

```bash
git add src/score_universe.py src/eval/test_protocol.py tests/test_score_universe.py
git commit -m "feat(score): v3_1_prefix builder and --prefix-intervention through scoring and the test protocol"
```

---

### Task 7: struct_hpo arms `prefix_static`, `prefix_pair`, `prefix_pair_bce`

**Files:**
- Modify: `src/experiments/struct_hpo.py`
- Test: `tests/test_struct_hpo.py`

**Interfaces:**
- Consumes: `SweepSpec`, `run_sweep` from `kd_rank_strict_hpo`.
- Produces: `SEARCH_BOXES["prefix_static"]`, `["prefix_pair"]` = `{"rank": (0.1, 3.0), "degree": (0.01, 1.0), "motif": (0.01, 1.0), "lr": (1e-4, 1e-2)}`; `ARMS` entries with `base_config = Path("configs/split_seed42/prefix_<arm>.yaml")`, `study_name = "prefix_<arm>"`, priors `({"rank": 1.0, "degree": 0.1, "motif": 0.1, "lr": 1e-3}, {"rank": 3.0, "degree": 0.5, "motif": 0.5, "lr": 3e-3})`; arm `prefix_pair_bce` with `param_names = ("lr",)`, a `suggest` that only draws `lr` in `(1e-4, 1e-2)`, and priors built from `--lr-center X` as `(X/3, X, 3X)`; `materialize_trial_config` writes `lr` to `optim.lr` **and** `optim.scheduler.max_lr` (when a scheduler block exists) and never into `struct.weights`.

- [ ] **Step 1: Write the failing tests**

Edit `tests/test_struct_hpo.py`:

1. Extend `test_arm_table_matches_spec` with:

```python
    for arm in ("prefix_static", "prefix_pair"):
        spec = struct_hpo.ARMS[arm]
        assert spec.study_name == arm
        assert spec.base_config == Path(f"configs/split_seed42/{arm}.yaml")
        assert spec.param_names == ("rank", "degree", "motif", "lr")
        assert spec.priors == (
            {"rank": 1.0, "degree": 0.1, "motif": 0.1, "lr": 1e-3},
            {"rank": 3.0, "degree": 0.5, "motif": 0.5, "lr": 3e-3},
        )
    bce = struct_hpo.ARMS["prefix_pair_bce"]
    assert bce.param_names == ("lr",)
    assert bce.base_config == Path("configs/split_seed42/prefix_pair_bce.yaml")
    assert bce.priors == ()
```

2. Change the two `@pytest.mark.parametrize("arm", ["grand", "new"])` decorators to include `"prefix_static", "prefix_pair"`, and in `test_materialize_overrides_only_struct_weights_and_output_dir` replace the loop over `params` with:

```python
    for name, value in params.items():
        if name == "lr":
            assert trial["optim"]["lr"] == value
            assert trial["optim"]["scheduler"]["max_lr"] == value
            assert "lr" not in trial["struct"]["weights"]
        else:
            assert trial["struct"]["weights"][name] == value
```

and before the final equality against `base`, also reset `trial["optim"] = base["optim"]`.

3. Add:

```python
def test_bce_arm_priors_come_from_lr_center() -> None:
    args = argparse.Namespace(arm="prefix_pair_bce", base_config=None, sweep_dir=None, lr_center=3e-3)
    spec = struct_hpo.build_spec(args)
    assert spec.priors == ({"lr": 1e-3}, {"lr": 3e-3}, {"lr": 9e-3})
    assert args.sweep_dir == Path("outputs/struct_hpo/prefix_pair_bce")


def test_bce_arm_requires_lr_center() -> None:
    args = argparse.Namespace(arm="prefix_pair_bce", base_config=None, sweep_dir=None, lr_center=None)
    with pytest.raises(ValueError, match="lr-center"):
        struct_hpo.build_spec(args)
```

Update `test_parser_defaults` to assert `args.lr_center is None` by default.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_struct_hpo.py -n0 -v`
Expected: FAIL with `KeyError: 'prefix_static'`.

- [ ] **Step 3: Implement**

In `src/experiments/struct_hpo.py`:

```python
LR_BOX: tuple[float, float] = (1e-4, 1e-2)
PREFIX_BOX: dict[str, tuple[float, float]] = {
    "rank": (0.1, 3.0),
    "degree": (0.01, 1.0),
    "motif": (0.01, 1.0),
    "lr": LR_BOX,
}
SEARCH_BOXES: dict[str, dict[str, tuple[float, float]]] = {
    "grand": {"gs": (0.1, 2.0), "rd": (0.1, 2.0)},
    "new": {"rank": (0.1, 3.0), "degree": (0.01, 1.0), "motif": (0.01, 1.0)},
    "prefix_static": dict(PREFIX_BOX),
    "prefix_pair": dict(PREFIX_BOX),
    "prefix_pair_bce": {"lr": LR_BOX},
}
PREFIX_PRIORS: tuple[dict[str, object], ...] = (
    {"rank": 1.0, "degree": 0.1, "motif": 0.1, "lr": 1e-3},
    {"rank": 3.0, "degree": 0.5, "motif": 0.5, "lr": 3e-3},
)
```

Add the three `ArmSpec` entries to `ARMS` (`prefix_pair_bce` with `priors=()`, `param_names=("lr",)`, `suggest=_suggest_for("prefix_pair_bce")`). In `materialize_trial_config`, split `params`:

```python
    lr = params.get("lr")
    weight_params = {k: float(v) for k, v in params.items() if k != "lr"}
    weights = {**cfg["struct"]["weights"], **weight_params}
    cfg["struct"] = {**cfg["struct"], "weights": weights}
    if lr is not None:
        cfg["optim"]["lr"] = float(lr)
        scheduler = cfg["optim"].get("scheduler")
        if isinstance(scheduler, dict) and "max_lr" in scheduler:
            scheduler["max_lr"] = float(lr)
```

In `build_spec`, after binding `arm`:

```python
    priors = arm.priors
    if str(args.arm) == "prefix_pair_bce":
        center = getattr(args, "lr_center", None)
        if center is None:
            raise ValueError("--lr-center (the prefix_pair winner's lr) is required for prefix_pair_bce")
        priors = ({"lr": float(center) / 3.0}, {"lr": float(center)}, {"lr": float(center) * 3.0})
```

and pass `priors=priors` into `SweepSpec`. Add `parser.add_argument("--lr-center", type=float, default=None)` and update the module docstring: the three prefix arms, the `lr` handling, and that `prefix_pair_bce` is launched with `--n-trials 3 --lr-center <winner lr>` so its three trials are exactly the enqueued priors (`N_STARTUP_TRIALS` is 3, so no sampled trial runs).

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_struct_hpo.py -n0 -v`
Expected: all PASS. The materialize test for the prefix arms needs the configs from Task 8; run Task 8 first if it fails on a missing file, then re-run.

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/python -m ruff check src/experiments/struct_hpo.py tests/test_struct_hpo.py && .venv/bin/python -m ruff format src/experiments/struct_hpo.py tests/test_struct_hpo.py && .venv/bin/python -m mypy src/experiments/struct_hpo.py tests/test_struct_hpo.py`

```bash
git add src/experiments/struct_hpo.py tests/test_struct_hpo.py
git commit -m "feat(hpo): prefix_static, prefix_pair, and prefix_pair_bce studies with lr in the box"
```

---

### Task 8: Arm configs

**Files:**
- Create: `configs/split_seed42/prefix_static.yaml`, `configs/split_seed42/prefix_pair.yaml`, `configs/split_seed42/prefix_pair_bce.yaml`
- Test: `tests/test_sweep_configs.py`

**Interfaces:**
- Consumes: `load_config` accepting `v3_1_prefix` (Task 5); `configs/split_seed42/struct_new.yaml` as the template.
- Produces: the three YAMLs read by Task 7's arms and Task 10's launches.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sweep_configs.py`:

```python
@pytest.mark.parametrize("arm", ["prefix_static", "prefix_pair", "prefix_pair_bce"])
def test_prefix_arm_configs_share_the_struct_new_recipe(arm: str) -> None:
    base = yaml.safe_load(Path("configs/split_seed42/struct_new.yaml").read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(f"configs/split_seed42/{arm}.yaml").read_text(encoding="utf-8"))
    assert cfg["model"] == {
        "family": "v3_1_prefix",
        "config": {
            "prefix": {
                "base_checkpoint": "outputs/split_seed42/prefix_base/best.pt",
                "tokens": 16,
                "rank": 8,
                "conditioning": "static" if arm == "prefix_static" else "pair",
                "bottleneck": 128,
            }
        },
    }
    assert cfg["output_dir"] == f"outputs/split_seed42/{arm}"
    assert cfg["optim"]["lr"] == 1e-3 and cfg["optim"]["scheduler"]["max_lr"] == 1e-3
    assert cfg["eval"] == {"patience": 5, "eval_every": 1, "topology_every": 2}
    expected_weights = (
        {"bce": 1.0, "gs": 0.0, "rd": 0.0, "deg_mmd": 0.0, "rank": 0.0, "degree": 0.0, "motif": 0.0}
        if arm == "prefix_pair_bce"
        else {"bce": 1.0, "gs": 0.0, "rd": 0.0, "deg_mmd": 0.0, "rank": 1.0, "degree": 0.1, "motif": 0.1}
    )
    assert cfg["struct"]["weights"] == expected_weights
    for key in ("data", "runtime", "seed", "mixed_precision"):
        assert cfg[key] == base[key]
    assert {k: v for k, v in cfg["struct"].items() if k != "weights"} == {
        k: v for k, v in base["struct"].items() if k != "weights"
    }
```

Add `import pytest` if missing.

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_sweep_configs.py -n0 -v -k prefix_arm`
Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 3: Write the configs**

```bash
.venv/bin/python - <<'EOF'
import yaml, pathlib
base = yaml.safe_load(pathlib.Path("configs/split_seed42/struct_new.yaml").read_text())
for arm, conditioning, weights in (
    ("prefix_static", "static", {"rank": 1.0, "degree": 0.1, "motif": 0.1}),
    ("prefix_pair", "pair", {"rank": 1.0, "degree": 0.1, "motif": 0.1}),
    ("prefix_pair_bce", "pair", {"rank": 0.0, "degree": 0.0, "motif": 0.0}),
):
    cfg = yaml.safe_load(yaml.safe_dump(base))
    cfg["model"] = {"family": "v3_1_prefix", "config": {"prefix": {
        "base_checkpoint": "outputs/split_seed42/prefix_base/best.pt",
        "tokens": 16, "rank": 8, "conditioning": conditioning, "bottleneck": 128}}}
    cfg["optim"]["lr"] = 1e-3
    cfg["optim"]["scheduler"]["max_lr"] = 1e-3
    cfg["eval"] = {"patience": 5, "eval_every": 1, "topology_every": 2}
    cfg["output_dir"] = f"outputs/split_seed42/{arm}"
    cfg["struct"]["weights"] = {"bce": 1.0, "gs": 0.0, "rd": 0.0, "deg_mmd": 0.0, **weights}
    header = (
        f"# {arm}: frozen prefix_base trunk + trainable gated KV prefix (model.family v3_1_prefix),\n"
        "# supervised by the struct_new structural stream. Spec:\n"
        "# docs/superpowers/specs/2026-09-10-prefix-tuning-frozen-trunk-design.md. Base config of\n"
        f"# src.experiments.struct_hpo --arm {arm}; the study rewrites struct.weights, optim.lr,\n"
        "# optim.scheduler.max_lr, and output_dir per trial.\n"
    )
    pathlib.Path(f"configs/split_seed42/{arm}.yaml").write_text(header + yaml.safe_dump(cfg, sort_keys=False))
EOF
```

Open one file and confirm the `data`, `runtime`, `seed`, and `mixed_precision` sections are unchanged from `struct_new.yaml` and the `struct` block keeps `nodes: 40`, `background_nodes: 8`, the mix, and `rank_margin: 0.1`.

- [ ] **Step 4: Run the config tests and the struct_hpo materialize tests**

Run: `.venv/bin/python -m pytest tests/test_sweep_configs.py tests/test_struct_hpo.py -n0 -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/split_seed42/prefix_static.yaml configs/split_seed42/prefix_pair.yaml configs/split_seed42/prefix_pair_bce.yaml tests/test_sweep_configs.py
git commit -m "feat(split): prefix_static, prefix_pair, prefix_pair_bce configs on the seed-42 split"
```

---

### Task 9: Documentation

**Files:**
- Modify: `docs/03-experiments.md` §1.4 table and §1.5 first sentence
- Modify: `CLAUDE.md` and `AGENTS.md` (the "Active method set" list)
- Modify: `hpc/README.md` (launch lines, near the struct_hpo examples)
- Modify: `src/experiments/struct_hpo.py` module docstring (if Task 7 did not already)

- [ ] **Step 1: `docs/03-experiments.md`**

In the §1.4 "Compared methods" table, after the `struct_new` row, add:

```markdown
| prefix_base | none | — | none | the headline B0 recipe with `mixing.mode: bidirectional_cross`; frozen base of the prefix arms |
| prefix_static | none (structural stream through a frozen trunk) | — | `rank`, `degree`, `motif` as struct_new; `lr` log-uniform [1e-4, 1e-2] | shared-prompt control (spec 2026-09-10) |
| prefix_pair | none (structural stream through a frozen trunk) | — | as prefix_static | primary prefix arm: pair-conditioned slot-specific prefix |
| prefix_pair_bce | none (subgraph BCE only) | — | `lr` at {c/3, c, 3c} around the prefix_pair winner | topology-supervision control |
```

In §1.5, replace "The V3.1 student uses d_model 512, 3 encoder + 3 cross-attention layers, 8 heads, rich pooling (mean/attn/max/gated), pair_context_gated readout with abba_max aggregation" with "The V3.1 student uses d_model 512, 3 encoder layers, 8 heads, `mixing.mode: none` (the configured `cross_attn_layers: 3` build no pair cross-attention layers under that mode; `prefix_base` is the arm that instantiates them), pair_context_gated readout with abba_max aggregation". Add one sentence after it: "The prefix arms freeze `prefix_base` and train only a gated KV prefix (spec `docs/superpowers/specs/2026-09-10-prefix-tuning-frozen-trunk-design.md`); their interventions are scoring-time flags of `score_universe` (`--prefix-intervention`)."

- [ ] **Step 2: `CLAUDE.md` and `AGENTS.md`**

In both files' "Active method set" list, after the structural-arms bullet, add:

```markdown
- Prefix arms (`model.family: v3_1_prefix`, `model.config.prefix` + `struct:` block): `prefix_static`,
  `prefix_pair`, `prefix_pair_bce`. They load and freeze `prefix_base` (`configs/split_seed42/prefix_base.yaml`,
  the headline B0 with `mixing.mode: bidirectional_cross`) and train only a zero-init gated KV prefix
  in its three cross-attention layers; `prefix_pair_bce` zeroes the structural weights as the
  topology-supervision control. Studies: `src.experiments.struct_hpo --arm prefix_{static,pair}` and
  `--arm prefix_pair_bce --n-trials 3 --lr-center <winner lr>`. Scoring-time interventions:
  `score_universe --prefix-intervention {gates_off,shuffle,mean}`.
```

- [ ] **Step 3: `hpc/README.md`**

Near the existing `struct_hpo` launch examples add:

```bash
hpc/run.sh train configs/split_seed42/prefix_base.yaml                          # frozen base of the prefix arms
.venv/bin/python -m src.experiments.struct_hpo --arm prefix_static           # 10 trials
.venv/bin/python -m src.experiments.struct_hpo --arm prefix_pair             # 10 trials
.venv/bin/python -m src.experiments.struct_hpo --arm prefix_pair_bce --n-trials 3 --lr-center <prefix_pair winner lr>
hpc/run.sh test --checkpoint outputs/struct_hpo/prefix_pair/trial_<k>/best.pt --prefix-intervention shuffle --report-filename test_report_shuffle.json
```

Check the `hpc/run.sh test` argument passthrough (`hpc/run.sh test <test args...>`) so `--prefix-intervention` reaches `src.eval.test_protocol`; if `run.sh` filters flags, add the flag there.

- [ ] **Step 4: Verify docs render and commit**

Run: `.venv/bin/python -m pytest tests/test_struct_hpo.py -n0 -q` (the driver test `test_driver_never_touches_frozen_paths` reads `CLAUDE.md`-adjacent rules; make sure it still passes).

```bash
git add docs/03-experiments.md CLAUDE.md AGENTS.md hpc/README.md src/experiments/struct_hpo.py
git commit -m "docs: prefix arms, prefix_base, and the corrected student trunk description"
```

---

### Task 10: Full verification, Codex review, push, and launch commands

**Files:**
- No new files. Review output goes to the scratchpad.

- [ ] **Step 1: Full local checks**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format --check src tests
.venv/bin/python -m mypy src tests
.venv/bin/python -m pytest -m "not slow and not integration" --dist loadfile -q
```

Expected: all clean and green. Fix anything the suite surfaces, commit as `fix(prefix): …`.

- [ ] **Step 2: Codex review of the wave**

`BASE` is the commit before Task 1 (`git log --oneline | grep -n "prefix_base, the headline" ` and take its parent).

```bash
S=/private/tmp/claude-501/-Users-richardwang-Documents-topology-conditioned-inductive-edge-prediction/810cd6bb-d08e-4e35-a289-7b1f59c6536f/scratchpad
mkdir -p $S/codex-home && cp ~/.codex/auth.json $S/codex-home/ 2>/dev/null
printf 'model = "gpt-5.4"\nmodel_reasoning_effort = "high"\n' > $S/codex-home/config.toml
BASE=$(git rev-parse <sha-before-task-1>)
CODEX_HOME=$S/codex-home codex review --base $BASE > $S/wave-review.txt 2>&1 &
```

Wait for the process to exit, then read `$S/wave-review.txt` (never let it stream into context). Fix blockers as `fix(prefix): …` commits; skip nits that contradict the spec. One round only unless new changes follow.

- [ ] **Step 3: Push**

```bash
git push origin HEAD
```

- [ ] **Step 4: Hand off the H20 launch order**

Report the exact commands, in this order, to be run on the H20 checkout after `git pull` (pull works on 30846 only):

```bash
# 1. frozen base (train + publish + held-out test)
hpc/run.sh train configs/split_seed42/prefix_base.yaml
# 2. the two weight studies (parallel on two containers is fine)
.venv/bin/python -m src.experiments.struct_hpo --arm prefix_static
.venv/bin/python -m src.experiments.struct_hpo --arm prefix_pair
# 3. the BCE-only control at three learning rates around the prefix_pair winner
#    (read the winner's lr from outputs/struct_hpo/prefix_pair/best_trial.json)
.venv/bin/python -m src.experiments.struct_hpo --arm prefix_pair_bce --n-trials 3 --lr-center <winner lr>
# 4. held-out test of each winner (best_trial.json names the trial directory)
hpc/run.sh test --checkpoint outputs/struct_hpo/prefix_static/trial_<k>/best.pt
hpc/run.sh test --checkpoint outputs/struct_hpo/prefix_pair/trial_<k>/best.pt
hpc/run.sh test --checkpoint outputs/struct_hpo/prefix_pair_bce/trial_<k>/best.pt
# 5. interventions on the prefix_pair winner (thresholds frozen from its own validation)
for iv in gates_off shuffle mean; do
  hpc/run.sh test --checkpoint outputs/struct_hpo/prefix_pair/trial_<k>/best.pt \
    --prefix-intervention $iv --report-filename test_report_$iv.json
done
```

Note in the hand-off that `gates_off` must reproduce `outputs/split_seed42/prefix_base/test_report.json` at the same thresholds, and that a mismatch is a bug in the wrapper, not a result.

---

## Self-review against the spec

- §4 `prefix_base` config, one-key difference, smoke: Task 1. Readout stays: no code touches it (Task 4 calls the frozen readout).
- §4 loading/freezing, `train()` override, embedded state, provenance without verification: Tasks 4, 5.
- §5.1 static prefix, real-activation init, slot-specific low-rank shift, gates as the only zero factor, symmetric `z`: Tasks 2, 4 (init from loader in Task 5).
- §5.2 base MHA called unchanged, separate softmax, `out_proj` without bias, three sites, all layers, no prefix dropout: Task 3.
- §6 struct_new stream, prefix-only AdamW, `lr` in the search box, same objectives, `prefix_pair_bce` three-point lr: Tasks 5, 7, 8.
- §7 interventions at the unordered-pair level (one permutation per forward shared by AB and BA), meta recorded, gates-off equals base: Tasks 4, 6, 10.
- §8 docs, configs, tests, execution order: Tasks 8, 9, 10.
- Deferred items (pooled adapter, length/rank ablations, calibration-fit diagnostic, paired bootstrap reporting) are analysis-time work on published artifacts and have no code task here; the paired bootstrap SE already exists in `test_protocol`.

Names used across tasks: `PrefixConfig`, `PrefixGenerator` (`p0`, `gates`, `shift_in`, `shift_out`, `cond_norm`, `cond_proj`, `z_sum`, `z_count`, `z_mean`, `condition`, `prefix`, `gate`, `init_static_from_tokens`), `prefix_branch`, `PrefixCrossAttentionLayer`, `V3_1Prefix` (`generator`, `base`, `prefix_layers`, `prefix_cfg`, `intervention`, `intervention_seed`, `prefix_parameters`, `init_static_prefix`), `is_v3_1_family`, `_resolve_prefix_kwargs`, `_init_prefix_from_loader`, `--prefix-intervention`, `--prefix-intervention-seed`, `--lr-center`.

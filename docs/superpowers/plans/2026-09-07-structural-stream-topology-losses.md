# Structural Stream and Topology Losses Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second training stream to the V3.1 student that forwards one sampled 40-node training subgraph per optimizer step and supervises its output adjacency with configurable topology terms (BCE, GRAND's GS/RD/degree-MMD, and the new rank/degree/motif terms), plus an Optuna driver that tunes each weighted arm with a 10-trial budget.

**Architecture:** A pure-tensor loss module (`src/distill/struct_losses.py`) and a graph sampler (`src/data/struct_sampler.py`) are composed by a `StructStream` object inside `src/train_b0.py`, wired beside the existing `KDContextStream` at the optimizer step. A `StructConfig` dataclass (`src/distill/struct_config.py`) is parsed from a new top-level `struct:` YAML block. `src/experiments/struct_hpo.py` reuses the existing `SweepSpec` / `run_sweep` Optuna loop with a struct-specific config writer.

**Tech Stack:** Python 3.12, PyTorch, networkx, numpy, Accelerate DDP, Optuna 4.7 (already a dependency), pytest with `--dist loadfile`, ruff (Google docstrings, no `print` in `src/`), mypy strict.

**Spec:** `docs/superpowers/specs/2026-09-07-structural-stream-topology-losses-design.md`

## Global Constraints

- Inference contract unchanged: the model receives only `(emb_a, emb_b, len_a, len_b)` per pair and returns one logit. No new model inputs.
- The 1:5 task stream, `training_sampler.py`, `pairs.py`, the KD banks and `distill.*` are not modified. An absent `struct:` block must leave training bit-identical.
- Legal pair mask `M`: distinct nodes, not both in `split.v_val`, neither in `assembled.exclude_nodes`. Self-pairs never enter the structural stream. Every term reduces only over `M == 1`.
- Defaults (verbatim from spec §6): `nodes: 40`, `background_nodes: 8`, `mix: {bfs: 0.5, motif: 0.25, bridge: 0.25}`, `subgraphs_per_epoch: null`, `rank_margin: 0.1`, `rank_temperature: 1.0`, `huber_delta: 1.0`, `mmd_sigma: 1.25`, `mmd_bins: 48`, `val_subgraphs: 32`. Weight keys exactly `bce, gs, rd, deg_mmd, rank, degree, motif`, all non-negative, not all zero.
- No loss balancing (no EMA normalisation, no GradNorm). No gates: log and proceed. Non-finite loss stays fail-closed through the existing `_all_ranks_loss_finite` check.
- Early stopping stays on `val_task_loss`; checkpoint selection is untouched; `val_struct_*` keys are diagnostics only.
- Local commands: `.venv/bin/python -m pytest <file> -n0`, `.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests`, `.venv/bin/python -m mypy src tests`. GPU runs only via `hpc/run.sh` on the H20 container.
- Commit after every task; commit messages end with the session's `Co-Authored-By` / `Claude-Session` trailer.
- `src/train_b0.py` is 5,000 lines; add the new class next to `KDContextStream` and keep edits surgical.

---

### Task 1: `StructConfig` and the `struct:` YAML block

**Files:**
- Create: `src/distill/struct_config.py`
- Modify: `src/train_b0.py:282-306` (`Config` dataclass), `src/train_b0.py:689-703` (top-level allow-list), `src/train_b0.py:847-861` (construction)
- Test: `tests/distill/test_struct_config.py`, `tests/test_train_b0_struct.py` (created here, extended in Task 5)

**Interfaces:**
- Produces: `StructConfig` (frozen dataclass) with fields `nodes: int`, `background_nodes: int`, `mix: dict[str, float]`, `subgraphs_per_epoch: int | None`, `weights: dict[str, float]`, `rank_margin: float`, `rank_temperature: float`, `huber_delta: float`, `mmd_sigma: float`, `mmd_bins: int`, `val_subgraphs: int`; properties `active_weights -> dict[str, float]` (nonzero only) and `arm -> str` (`"+".join(sorted(active))`); classmethod `from_mapping(mapping) -> StructConfig`. Module constants `WEIGHT_KEYS = ("bce", "gs", "rd", "deg_mmd", "rank", "degree", "motif")`, `KINDS = ("bfs", "motif", "bridge")`.
- Produces: `Config.struct: StructConfig | None = None` in `src/train_b0.py`.

- [ ] **Step 1: Write the failing config tests**

```python
# tests/distill/test_struct_config.py
from __future__ import annotations

import pytest

from src.distill.struct_config import KINDS, WEIGHT_KEYS, StructConfig


def test_defaults_match_spec() -> None:
    cfg = StructConfig.from_mapping({"weights": {"bce": 1.0}})
    assert cfg.nodes == 40
    assert cfg.background_nodes == 8
    assert cfg.mix == {"bfs": 0.5, "motif": 0.25, "bridge": 0.25}
    assert cfg.subgraphs_per_epoch is None
    assert cfg.rank_margin == 0.1
    assert cfg.rank_temperature == 1.0
    assert cfg.huber_delta == 1.0
    assert cfg.mmd_sigma == 1.25
    assert cfg.mmd_bins == 48
    assert cfg.val_subgraphs == 32
    assert set(cfg.weights) == set(WEIGHT_KEYS)
    assert cfg.active_weights == {"bce": 1.0}
    assert cfg.arm == "bce"


def test_arm_name_is_sorted_nonzero_keys() -> None:
    cfg = StructConfig.from_mapping({"weights": {"bce": 1.0, "motif": 0.1, "rank": 1.0}})
    assert cfg.arm == "bce+motif+rank"
    assert cfg.active_weights == {"bce": 1.0, "motif": 0.1, "rank": 1.0}


@pytest.mark.parametrize(
    "mapping, message",
    [
        ({"weights": {}}, "at least one"),
        ({"weights": {"bce": 0.0}}, "at least one"),
        ({"weights": {"bce": -1.0}}, "non-negative"),
        ({"weights": {"bce": 1.0, "clustering": 1.0}}, "unknown struct weight"),
        ({"weights": {"bce": 1.0}, "mix": {"bfs": 1.0}}, "mix must name exactly"),
        ({"weights": {"bce": 1.0}, "mix": {"bfs": 0.6, "motif": 0.3, "bridge": 0.3}}, "sum to 1"),
        ({"weights": {"bce": 1.0}, "nodes": 8, "background_nodes": 8}, "background_nodes"),
        ({"weights": {"bce": 1.0}, "nodes": 3}, "nodes must be"),
        ({"weights": {"bce": 1.0}, "subgraphs_per_epoch": 0}, "subgraphs_per_epoch"),
        ({"weights": {"bce": 1.0}, "rank_temperature": 0.0}, "rank_temperature"),
        ({"weights": {"bce": 1.0}, "mmd_bins": 1}, "mmd_bins"),
        ({"weights": {"bce": 1.0}, "bogus": 1}, "unknown struct config keys"),
        ({"weights": {"bce": True}}, "must be a number"),
    ],
)
def test_rejects_illegal_blocks(mapping: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        StructConfig.from_mapping(mapping)


def test_kinds_and_weight_keys_are_frozen() -> None:
    assert KINDS == ("bfs", "motif", "bridge")
    assert WEIGHT_KEYS == ("bce", "gs", "rd", "deg_mmd", "rank", "degree", "motif")
```

```python
# tests/test_train_b0_struct.py
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.train_b0 import config_to_dict, load_config


def _control_yaml() -> dict[str, object]:
    return yaml.safe_load(Path("configs/b1_kd_control_breadth_first.yaml").read_text())


def test_struct_block_parses_and_serialises(tmp_path: Path) -> None:
    raw = _control_yaml()
    raw["struct"] = {"weights": {"bce": 1.0, "rank": 1.0, "degree": 0.1, "motif": 0.1}}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path)
    assert cfg.struct is not None
    assert cfg.struct.arm == "bce+degree+motif+rank"
    payload = config_to_dict(cfg)
    assert payload["struct"]["weights"]["rank"] == 1.0
    assert payload["struct"]["mix"] == {"bfs": 0.5, "motif": 0.25, "bridge": 0.25}


def test_absent_struct_block_is_none(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(_control_yaml()))
    assert load_config(path).struct is None


def test_struct_block_rejects_unknown_key(tmp_path: Path) -> None:
    raw = _control_yaml()
    raw["struct"] = {"weights": {"bce": 1.0}, "gradnorm": True}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unknown struct config keys"):
        load_config(path)
```

`load_config(path: Path) -> Config` is defined at `src/train_b0.py:670`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/distill/test_struct_config.py tests/test_train_b0_struct.py -n0 -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.distill.struct_config'`.

- [ ] **Step 3: Implement `StructConfig`**

```python
# src/distill/struct_config.py
"""`StructConfig`: the structural-stream topology-loss knobs.

Consumed by `src.train_b0` as the optional top-level ``struct:`` config
section (spec: ``docs/superpowers/specs/2026-09-07-structural-stream-topology-losses-design.md``).
Each optimizer step forwards one sampled training subgraph and supervises its
output adjacency with the weighted terms named in ``weights``. Any non-negative
weight combination is legal; the arm name is the sorted list of nonzero keys.
An absent block (``Config.struct is None``) leaves training bit-identical.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields

WEIGHT_KEYS: tuple[str, ...] = ("bce", "gs", "rd", "deg_mmd", "rank", "degree", "motif")
KINDS: tuple[str, ...] = ("bfs", "motif", "bridge")
_INT_FIELDS = frozenset({"nodes", "background_nodes", "mmd_bins", "val_subgraphs"})
_FLOAT_FIELDS = frozenset(
    {"rank_margin", "rank_temperature", "huber_delta", "mmd_sigma"}
)


def _default_mix() -> dict[str, float]:
    return {"bfs": 0.5, "motif": 0.25, "bridge": 0.25}


def _default_weights() -> dict[str, float]:
    return dict.fromkeys(WEIGHT_KEYS, 0.0)


@dataclass(frozen=True)
class StructConfig:
    """Structural-stream configuration.

    Attributes:
        nodes: Subgraph size ``n`` (local budget is ``nodes - background_nodes``).
        background_nodes: Uniformly drawn nodes appended to every subgraph.
        mix: Sampling share per subgraph kind (``bfs``, ``motif``, ``bridge``); sums to 1.
        subgraphs_per_epoch: Plan length; ``None`` means one subgraph per global optimizer step.
        weights: Non-negative weight per term key in ``WEIGHT_KEYS``; at least one nonzero.
        rank_margin: Margin ``m`` of the neighbour-ranking term.
        rank_temperature: Temperature ``T`` of the neighbour-ranking term.
        huber_delta: SmoothL1 transition point shared by ``rd``, ``degree`` and ``motif``.
        mmd_sigma: Gaussian width of the degree soft histogram and of its TV kernel.
        mmd_bins: Number of degree histogram centres on ``[0, n-1]``.
        val_subgraphs: Fixed V_val diagnostic subgraph count.
    """

    nodes: int = 40
    background_nodes: int = 8
    mix: dict[str, float] = field(default_factory=_default_mix)
    subgraphs_per_epoch: int | None = None
    weights: dict[str, float] = field(default_factory=_default_weights)
    rank_margin: float = 0.1
    rank_temperature: float = 1.0
    huber_delta: float = 1.0
    mmd_sigma: float = 1.25
    mmd_bins: int = 48
    val_subgraphs: int = 32

    def __post_init__(self) -> None:
        """Validate ranges, the kind mix, and the weight pattern.

        Raises:
            ValueError: On any illegal value.
        """
        if self.nodes < 4:
            raise ValueError(f"struct.nodes must be >= 4, got {self.nodes}")
        if not 0 <= self.background_nodes < self.nodes:
            raise ValueError(
                f"struct.background_nodes must be in [0, nodes), got {self.background_nodes}"
            )
        if set(self.mix) != set(KINDS):
            raise ValueError(f"struct.mix must name exactly {list(KINDS)}, got {sorted(self.mix)}")
        if any(share < 0.0 for share in self.mix.values()) or abs(sum(self.mix.values()) - 1.0) > 1e-6:
            raise ValueError(f"struct.mix shares must be non-negative and sum to 1, got {self.mix}")
        if self.subgraphs_per_epoch is not None and self.subgraphs_per_epoch < 1:
            raise ValueError("struct.subgraphs_per_epoch must be positive or null")
        unknown = sorted(set(self.weights) - set(WEIGHT_KEYS))
        if unknown:
            raise ValueError(f"unknown struct weight keys: {unknown}")
        normalised = {key: float(self.weights.get(key, 0.0)) for key in WEIGHT_KEYS}
        for key, value in normalised.items():
            if value < 0.0:
                raise ValueError(f"struct.weights.{key} must be non-negative")
        # Frozen dataclass: fill the missing keys through object.__setattr__.
        object.__setattr__(self, "weights", normalised)
        object.__setattr__(self, "mix", {kind: float(self.mix[kind]) for kind in KINDS})
        if not self.active_weights:
            raise ValueError("struct.weights must have at least one nonzero term")
        if self.rank_margin < 0.0:
            raise ValueError("struct.rank_margin must be non-negative")
        if self.rank_temperature <= 0.0:
            raise ValueError("struct.rank_temperature must be positive")
        if self.huber_delta <= 0.0:
            raise ValueError("struct.huber_delta must be positive")
        if self.mmd_sigma <= 0.0:
            raise ValueError("struct.mmd_sigma must be positive")
        if self.mmd_bins < 2:
            raise ValueError("struct.mmd_bins must be >= 2")
        if self.val_subgraphs < 0:
            raise ValueError("struct.val_subgraphs must be non-negative")

    @property
    def active_weights(self) -> dict[str, float]:
        """Nonzero weights in ``WEIGHT_KEYS`` order."""
        return {key: float(self.weights[key]) for key in WEIGHT_KEYS if self.weights.get(key, 0.0) > 0.0}

    @property
    def arm(self) -> str:
        """Arm name: the sorted nonzero weight keys joined by ``+``."""
        return "+".join(sorted(self.active_weights))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> StructConfig:
        """Build the ``struct:`` section from a YAML mapping.

        Raises:
            ValueError: On unknown keys or ill-typed values.
        """
        known = {spec.name for spec in fields(cls)}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(f"unknown struct config keys: {unknown}")
        kwargs: dict[str, object] = {}
        for spec in fields(cls):
            if spec.name not in mapping:
                continue
            raw = mapping[spec.name]
            if spec.name in {"mix", "weights"}:
                if not isinstance(raw, Mapping):
                    raise ValueError(f"struct.{spec.name} must be a mapping")
                kwargs[spec.name] = {str(k): _number(v, f"struct.{spec.name}.{k}") for k, v in raw.items()}
            elif spec.name == "subgraphs_per_epoch":
                kwargs[spec.name] = None if raw is None else int(_number(raw, "struct.subgraphs_per_epoch"))
            elif spec.name in _INT_FIELDS:
                kwargs[spec.name] = int(_number(raw, f"struct.{spec.name}"))
            elif spec.name in _FLOAT_FIELDS:
                kwargs[spec.name] = _number(raw, f"struct.{spec.name}")
        return cls(**kwargs)  # type: ignore[arg-type]


def _number(raw: object, label: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{label} must be a number")
    return float(raw)


__all__ = ["KINDS", "WEIGHT_KEYS", "StructConfig"]
```

Note: the "unknown struct weight" test expects the message `unknown struct weight keys`; `pytest.raises(match=...)` uses `re.search`, so the substring matches.

- [ ] **Step 4: Wire the block into `Config`**

In `src/train_b0.py`:

1. Import: after `from src.distill.config import DistillConfig` add `from src.distill.struct_config import StructConfig`.
2. `Config` dataclass: add attribute doc line `struct: Optional structural-stream section; ``None`` keeps the plain protocol.` and the field `struct: StructConfig | None = None` after `distill`.
3. Allow-list tuple at the `_check_no_unknown_keys(raw, (...), "<top level>")` call: add `"struct",` after `"distill",`.
4. Construction: after the `distill` block add

```python
    struct: StructConfig | None = None
    if "struct" in raw:
        struct = StructConfig.from_mapping(_as_mapping(raw["struct"], "struct"))
```

and pass `struct=struct,` into `Config(...)`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/distill/test_struct_config.py tests/test_train_b0_struct.py -n0 -q`
Expected: all PASS.

- [ ] **Step 6: Lint, type-check, commit**

Run: `.venv/bin/python -m ruff check src/distill/struct_config.py src/train_b0.py tests/distill/test_struct_config.py tests/test_train_b0_struct.py && .venv/bin/python -m ruff format src tests && .venv/bin/python -m mypy src/distill/struct_config.py tests/distill/test_struct_config.py`
Expected: clean.

```bash
git add src/distill/struct_config.py src/train_b0.py tests/distill/test_struct_config.py tests/test_train_b0_struct.py
git commit -m "feat(struct): StructConfig and the top-level struct: block"
```

---

### Task 2: Structural loss terms

**Files:**
- Create: `src/distill/struct_losses.py`
- Test: `tests/distill/test_struct_losses.py`

**Interfaces:**
- Produces, all with signature `(logits: Tensor[n,n], target: Tensor[n,n], mask: Tensor[n,n]) -> Tensor[]` plus keyword-only knobs: `struct_bce(..., *, positive_weight, label_smoothing=0.0)`, `struct_gs(...)`, `struct_rd(..., *, huber_delta)`, `struct_deg_mmd(..., *, sigma, bins)`, `struct_rank(..., *, margin, temperature)`, `struct_degree(..., *, huber_delta)`, `struct_motif(..., *, huber_delta)`.
- Produces: `struct_total(logits, target, mask, config: StructConfig, *, positive_weight, label_smoothing) -> tuple[Tensor, dict[str, Tensor]]` (weighted sum, raw per-active-term tensors).
- Produces: `hard_struct_errors(logits, target, mask, threshold: float) -> dict[str, float]` with keys `hard_degree_mae`, `hard_triangle_mae`, `hard_wedge_mae` (no grad).
- Inputs are float32 symmetric matrices with zero diagonal; `mask` is 0/1 float.

- [ ] **Step 1: Write the failing loss tests**

```python
# tests/distill/test_struct_losses.py
from __future__ import annotations

import itertools

import pytest
import torch

from src.distill.struct_config import StructConfig
from src.distill.struct_losses import (
    hard_struct_errors,
    struct_bce,
    struct_deg_mmd,
    struct_degree,
    struct_gs,
    struct_motif,
    struct_rank,
    struct_rd,
    struct_total,
)


def _graph(n: int, edges: list[tuple[int, int]], illegal: list[tuple[int, int]] | None = None):
    target = torch.zeros(n, n)
    for u, v in edges:
        target[u, v] = target[v, u] = 1.0
    mask = 1.0 - torch.eye(n)
    for u, v in illegal or []:
        mask[u, v] = mask[v, u] = 0.0
    return target * mask, mask


def _logits_from(target: torch.Tensor, scale: float = 12.0) -> torch.Tensor:
    return ((target * 2.0 - 1.0) * scale).requires_grad_(True)


def _brute_motifs(a: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = a.size(0)
    closed = torch.zeros(n, n)
    open_ = torch.zeros(n, n)
    for u, v, w in itertools.permutations(range(n), 3):
        if mask[u, w] and mask[w, v] and a[u, w] and a[w, v]:
            if a[u, v]:
                closed[u, v] += 1
            else:
                open_[u, v] += 1
    return closed, open_


def test_motif_matches_brute_force_on_hard_graph() -> None:
    target, mask = _graph(6, [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (3, 5)], illegal=[(1, 4)])
    logits = _logits_from(target, scale=40.0)
    closed, open_ = _brute_motifs(target, mask)
    p = torch.sigmoid(logits.detach()) * mask
    c_soft = p * (p @ p)
    o_soft = (1.0 - p) * (p @ p)
    sel = torch.triu(mask, 1) > 0
    torch.testing.assert_close(c_soft[sel], closed[sel], atol=1e-4, rtol=0.0)
    torch.testing.assert_close(o_soft[sel], open_[sel], atol=1e-4, rtol=0.0)
    assert float(struct_motif(logits, target, mask, huber_delta=1.0)) < 1e-6


@pytest.mark.parametrize(
    "fn, kwargs",
    [
        (struct_gs, {}),
        (struct_rd, {"huber_delta": 1.0}),
        (struct_degree, {"huber_delta": 1.0}),
        (struct_motif, {"huber_delta": 1.0}),
        (struct_deg_mmd, {"sigma": 1.25, "bins": 48}),
    ],
)
def test_terms_vanish_when_prediction_equals_target(fn, kwargs) -> None:
    target, mask = _graph(7, [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6)])
    logits = _logits_from(target, scale=40.0)
    assert float(fn(logits, target, mask, **kwargs)) < 1e-5


def test_rank_vanishes_for_separated_logits_and_grows_when_inverted() -> None:
    target, mask = _graph(5, [(0, 1), (0, 2), (3, 4)])
    good = _logits_from(target, scale=10.0)
    bad = _logits_from(1.0 - target - torch.eye(5), scale=10.0)
    assert float(struct_rank(good, target, mask, margin=0.1, temperature=1.0)) < 1e-3
    assert float(struct_rank(bad, target, mask, margin=0.1, temperature=1.0)) > 5.0


def test_rank_weights_anchors_equally() -> None:
    # Anchor 0 has 3 positives x 1 negative, anchor 4 has 1 x 3; a wrong ordering at anchor 4
    # must cost the same as one at anchor 0 regardless of pair counts.
    target, mask = _graph(5, [(0, 1), (0, 2), (0, 3), (4, 1)])
    logits = torch.zeros(5, 5)
    loss_zero = struct_rank(logits, target, mask, margin=0.0, temperature=1.0)
    torch.testing.assert_close(loss_zero, torch.tensor(float(torch.log(torch.tensor(2.0)))))


def test_gs_and_rd_reproduce_grand_hand_values() -> None:
    # 4 nodes, edges (0,1),(2,3); predict p=0.5 everywhere legal (logit 0).
    target, mask = _graph(4, [(0, 1), (2, 3)])
    logits = torch.zeros(4, 4, requires_grad=True)
    # GS = sum|p-a| / (sum p + sum a): 6 pairs, |0.5-1|*2 + |0.5-0|*4 = 3; denom = 3 + 2 = 5
    torch.testing.assert_close(struct_gs(logits, target, mask), torch.tensor(3.0 / 5.0), atol=1e-6, rtol=0.0)
    # RD: log(3/2) = 0.405 < 1 -> SmoothL1 = 0.5 * 0.405^2
    expected = 0.5 * float(torch.log(torch.tensor(1.5))) ** 2
    torch.testing.assert_close(struct_rd(logits, target, mask, huber_delta=1.0), torch.tensor(expected), atol=1e-6, rtol=0.0)


def test_degree_uses_in_subgraph_legal_degrees() -> None:
    target, mask = _graph(4, [(0, 1), (0, 2), (0, 3)], illegal=[(0, 3)])
    # Node 0's legal degree is 2 (pair (0,3) masked); predict logit +40 on legal edges.
    logits = _logits_from(target, scale=40.0)
    assert float(struct_degree(logits, target, mask, huber_delta=1.0)) < 1e-5


@pytest.mark.parametrize(
    "fn, kwargs",
    [
        (struct_bce, {"positive_weight": 5.0}),
        (struct_gs, {}),
        (struct_rd, {"huber_delta": 1.0}),
        (struct_deg_mmd, {"sigma": 1.25, "bins": 48}),
        (struct_rank, {"margin": 0.1, "temperature": 1.0}),
        (struct_degree, {"huber_delta": 1.0}),
        (struct_motif, {"huber_delta": 1.0}),
    ],
)
def test_masked_entries_receive_zero_gradient(fn, kwargs) -> None:
    target, mask = _graph(6, [(0, 1), (1, 2), (2, 0), (3, 4)], illegal=[(0, 3), (2, 5)])
    logits = torch.randn(6, 6)
    logits = ((logits + logits.T) / 2).fill_diagonal_(0.0).requires_grad_(True)
    loss = fn(logits, target, mask, **kwargs)
    loss.backward()
    grad = logits.grad
    assert grad is not None
    assert float(grad[0, 3].abs() + grad[3, 0].abs() + grad[2, 5].abs() + grad[5, 2].abs()) == 0.0
    assert float(grad.diagonal().abs().sum()) == 0.0


def test_empty_reduction_sets_return_differentiable_zero() -> None:
    target, mask = _graph(3, [], illegal=[(0, 1), (0, 2), (1, 2)])
    logits = torch.zeros(3, 3, requires_grad=True)
    for fn, kwargs in [
        (struct_bce, {"positive_weight": 5.0}),
        (struct_gs, {}),
        (struct_rd, {"huber_delta": 1.0}),
        (struct_deg_mmd, {"sigma": 1.25, "bins": 48}),
        (struct_rank, {"margin": 0.1, "temperature": 1.0}),
        (struct_degree, {"huber_delta": 1.0}),
        (struct_motif, {"huber_delta": 1.0}),
    ]:
        value = fn(logits, target, mask, **kwargs)
        assert value.requires_grad
        assert float(value) == 0.0


def test_motif_balanced_mean_weights_classes_equally() -> None:
    # One closed pair, one open pair, many zero pairs: the loss equals the mean of three class means.
    target, mask = _graph(8, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5)])
    logits = torch.zeros(8, 8, requires_grad=True)
    p = torch.sigmoid(logits.detach()) * mask
    a = target
    c_p, o_p = p * (p @ p), (1 - p) * (p @ p)
    c_a, o_a = a * (a @ a), (1 - a) * (a @ a)
    sel = torch.triu(mask, 1) > 0
    huber = torch.nn.functional.smooth_l1_loss
    err = huber(torch.log1p(c_p[sel]), torch.log1p(c_a[sel]), reduction="none") + huber(
        torch.log1p(o_p[sel]), torch.log1p(o_a[sel]), reduction="none"
    )
    ca, oa = c_a[sel], o_a[sel]
    classes = [ca > 0, (ca == 0) & (oa > 0), (ca == 0) & (oa == 0)]
    expected = torch.stack([err[c].mean() for c in classes]).mean()
    torch.testing.assert_close(struct_motif(logits, target, mask, huber_delta=1.0), expected)


def test_bce_uses_positive_weight_over_legal_upper_triangle() -> None:
    target, mask = _graph(3, [(0, 1)])
    logits = torch.zeros(3, 3, requires_grad=True)
    # 3 legal pairs: one positive (weight 5) and two negatives (weight 1), all at p=0.5.
    expected = torch.tensor(float(torch.log(torch.tensor(2.0))))
    torch.testing.assert_close(struct_bce(logits, target, mask, positive_weight=5.0), expected)


def test_total_computes_only_active_terms_and_weights_them() -> None:
    cfg = StructConfig.from_mapping({"weights": {"bce": 1.0, "motif": 0.5}})
    target, mask = _graph(5, [(0, 1), (1, 2), (0, 2)])
    logits = torch.zeros(5, 5, requires_grad=True)
    total, terms = struct_total(logits, target, mask, cfg, positive_weight=5.0, label_smoothing=0.0)
    assert set(terms) == {"bce", "motif"}
    torch.testing.assert_close(total, terms["bce"] * 1.0 + terms["motif"] * 0.5)


def test_hard_errors_are_zero_at_perfect_threshold() -> None:
    target, mask = _graph(5, [(0, 1), (1, 2), (0, 2), (3, 4)])
    logits = _logits_from(target, scale=4.0).detach()
    errors = hard_struct_errors(logits, target, mask, threshold=0.0)
    assert errors == {"hard_degree_mae": 0.0, "hard_triangle_mae": 0.0, "hard_wedge_mae": 0.0}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/distill/test_struct_losses.py -n0 -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.distill.struct_losses'`.

- [ ] **Step 3: Implement the loss module**

```python
# src/distill/struct_losses.py
"""Structural-stream topology terms over one sampled subgraph's logit matrix.

Every function takes symmetric ``n x n`` float32 tensors ``logits``, ``target``
(0/1 adjacency) and ``mask`` (0/1 legal-pair mask, zero diagonal) and reduces
only where ``mask == 1``. Each returns a differentiable zero when its
reduction set is empty so DDP sees every parameter on every step.
Spec: ``docs/superpowers/specs/2026-09-07-structural-stream-topology-losses-design.md`` section 5.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from src.distill.struct_config import StructConfig

EPSILON = 1.0e-8


def _upper(mask: torch.Tensor) -> torch.Tensor:
    return torch.triu(mask, diagonal=1) > 0


def _zero_like(logits: torch.Tensor) -> torch.Tensor:
    return logits.sum() * 0.0


def struct_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    positive_weight: float,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Positive-weighted BCE over legal upper-triangle pairs (the task loss's form)."""
    sel = _upper(mask)
    if not bool(sel.any()):
        return _zero_like(logits)
    z = logits[sel]
    y = target[sel]
    smoothed = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
    per_pair = F.binary_cross_entropy_with_logits(z, smoothed, reduction="none")
    weights = 1.0 + (positive_weight - 1.0) * y
    return (weights * per_pair).sum() / weights.sum()


def struct_gs(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """GRAND's soft graph-similarity loss: ``sum|p - a| / (sum p + sum a + eps)``."""
    sel = _upper(mask)
    if not bool(sel.any()):
        return _zero_like(logits)
    p = torch.sigmoid(logits)[sel]
    a = target[sel]
    return (p - a).abs().sum() / (p.sum() + a.sum() + EPSILON)


def struct_rd(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, huber_delta: float
) -> torch.Tensor:
    """Two-sided relative-density loss: SmoothL1 of ``log(sum p / sum a)``."""
    sel = _upper(mask)
    if not bool(sel.any()):
        return _zero_like(logits)
    p = torch.sigmoid(logits)[sel]
    a = target[sel]
    log_ratio = torch.log((p.sum() + EPSILON) / (a.sum() + EPSILON))
    return F.smooth_l1_loss(log_ratio, torch.zeros_like(log_ratio), beta=huber_delta)


def _soft_histogram(values: torch.Tensor, centers: torch.Tensor, sigma: float) -> torch.Tensor:
    scaled = (values.unsqueeze(-1) - centers.unsqueeze(0)) / sigma
    histogram = torch.exp(-0.5 * scaled.square()).sum(dim=0)
    return histogram / (histogram.sum() + EPSILON)


def struct_deg_mmd(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, sigma: float, bins: int
) -> torch.Tensor:
    """GRAND's Gaussian-TV degree-histogram MMD between soft and true in-subgraph degrees."""
    if not bool(mask.any()):
        return _zero_like(logits)
    n = logits.size(0)
    soft_degrees = (torch.sigmoid(logits) * mask).sum(dim=1)
    true_degrees = (target * mask).sum(dim=1)
    centers = torch.linspace(0.0, float(max(1, n - 1)), steps=max(2, bins), device=logits.device)
    tv = 0.5 * (
        _soft_histogram(soft_degrees, centers, sigma) - _soft_histogram(true_degrees, centers, sigma)
    ).abs().sum()
    return 2.0 - 2.0 * torch.exp(-tv.square() / (2.0 * sigma * sigma))


def struct_rank(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    """Per-anchor neighbour ranking: softplus((z_ik - z_ij + m)/T) over j in P_i, k in N_i."""
    positive = (target > 0) & (mask > 0)
    negative = (target == 0) & (mask > 0)
    valid = positive.unsqueeze(2) & negative.unsqueeze(1)  # [i, j, k]
    counts = valid.sum(dim=(1, 2))
    anchors = counts > 0
    if not bool(anchors.any()):
        return _zero_like(logits)
    diff = logits.unsqueeze(1) - logits.unsqueeze(2) + margin  # z_ik - z_ij + m
    per_pair = F.softplus(diff / temperature) * valid
    per_anchor = per_pair.sum(dim=(1, 2))[anchors] / counts[anchors]
    return per_anchor.mean()


def struct_degree(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, huber_delta: float
) -> torch.Tensor:
    """Node-wise SmoothL1 of ``log1p`` soft vs true legal in-subgraph degree."""
    valid = mask.sum(dim=1) > 0
    if not bool(valid.any()):
        return _zero_like(logits)
    soft_degrees = (torch.sigmoid(logits) * mask).sum(dim=1)[valid]
    true_degrees = (target * mask).sum(dim=1)[valid]
    return F.smooth_l1_loss(
        torch.log1p(soft_degrees), torch.log1p(true_degrees), beta=huber_delta
    )


def _motif_counts(adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    two_hop = adjacency @ adjacency
    return adjacency * two_hop, (1.0 - adjacency) * two_hop


def struct_motif(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, huber_delta: float
) -> torch.Tensor:
    """Closed/open two-hop count matching with a balanced mean over target classes."""
    sel = _upper(mask)
    if not bool(sel.any()):
        return _zero_like(logits)
    p = torch.sigmoid(logits) * mask
    a = target * mask
    closed_p, open_p = _motif_counts(p)
    closed_a, open_a = _motif_counts(a)
    errors = F.smooth_l1_loss(
        torch.log1p(closed_p[sel]), torch.log1p(closed_a[sel]), reduction="none", beta=huber_delta
    ) + F.smooth_l1_loss(
        torch.log1p(open_p[sel]), torch.log1p(open_a[sel]), reduction="none", beta=huber_delta
    )
    ca = closed_a[sel]
    oa = open_a[sel]
    classes = (ca > 0, (ca == 0) & (oa > 0), (ca == 0) & (oa == 0))
    means = [errors[cls].mean() for cls in classes if bool(cls.any())]
    return torch.stack(means).mean()


def struct_total(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    config: StructConfig,
    *,
    positive_weight: float,
    label_smoothing: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted sum of the active terms; returns the total and each raw term."""
    terms: dict[str, torch.Tensor] = {}
    for key in config.active_weights:
        if key == "bce":
            terms[key] = struct_bce(
                logits, target, mask, positive_weight=positive_weight, label_smoothing=label_smoothing
            )
        elif key == "gs":
            terms[key] = struct_gs(logits, target, mask)
        elif key == "rd":
            terms[key] = struct_rd(logits, target, mask, huber_delta=config.huber_delta)
        elif key == "deg_mmd":
            terms[key] = struct_deg_mmd(
                logits, target, mask, sigma=config.mmd_sigma, bins=config.mmd_bins
            )
        elif key == "rank":
            terms[key] = struct_rank(
                logits, target, mask, margin=config.rank_margin, temperature=config.rank_temperature
            )
        elif key == "degree":
            terms[key] = struct_degree(logits, target, mask, huber_delta=config.huber_delta)
        elif key == "motif":
            terms[key] = struct_motif(logits, target, mask, huber_delta=config.huber_delta)
        else:  # pragma: no cover - StructConfig validates the key set
            raise KeyError(key)
    total = _zero_like(logits)
    for key, term in terms.items():
        total = total + config.active_weights[key] * term
    return total, terms


@torch.no_grad()
def hard_struct_errors(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, threshold: float
) -> dict[str, float]:
    """Mean absolute degree / per-node triangle / per-node wedge errors at a hard threshold."""
    predicted = ((logits > threshold).float() * mask)
    truth = target * mask

    def _node_counts(adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        degrees = adjacency.sum(dim=1)
        triangles = torch.diagonal(adjacency @ adjacency @ adjacency) / 2.0
        wedges = degrees * (degrees - 1.0) / 2.0 - triangles
        return degrees, triangles, wedges

    p_deg, p_tri, p_wedge = _node_counts(predicted)
    t_deg, t_tri, t_wedge = _node_counts(truth)
    return {
        "hard_degree_mae": float((p_deg - t_deg).abs().mean().item()),
        "hard_triangle_mae": float((p_tri - t_tri).abs().mean().item()),
        "hard_wedge_mae": float((p_wedge - t_wedge).abs().mean().item()),
    }


__all__ = [
    "EPSILON",
    "hard_struct_errors",
    "struct_bce",
    "struct_deg_mmd",
    "struct_degree",
    "struct_gs",
    "struct_motif",
    "struct_rank",
    "struct_rd",
    "struct_total",
]
```

Note on the gradient test: `torch.sigmoid(logits) * mask` zeroes masked entries but their gradient is exactly 0 because the multiplication by 0 kills it; the diagonal is zero in `mask`, so the same holds. `struct_rank` never indexes a masked pair because `valid` is false there.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/distill/test_struct_losses.py -n0 -q`
Expected: all PASS. If `test_rank_weights_anchors_equally` fails, check that the diff orientation is `z_ik - z_ij` (negative minus positive): with all-zero logits and margin 0 every pair costs `log 2`, so the loss is `log 2` regardless of orientation; a failure there means the anchor averaging is wrong, not the sign.

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/python -m ruff check src/distill/struct_losses.py tests/distill/test_struct_losses.py && .venv/bin/python -m ruff format src tests && .venv/bin/python -m mypy src/distill/struct_losses.py tests/distill/test_struct_losses.py`

```bash
git add src/distill/struct_losses.py tests/distill/test_struct_losses.py
git commit -m "feat(struct): topology loss terms over a subgraph logit matrix"
```

---

### Task 3: Structural subgraph sampler

**Files:**
- Create: `src/data/struct_sampler.py`
- Test: `tests/test_struct_sampler.py`

**Interfaces:**
- Consumes: `_anchor_rng` from `src/distill/context_sampler.py` (`_anchor_rng(node_id: str, *, seed: int, epoch: int) -> np.random.Generator`); `KINDS` from `src/distill/struct_config.py`.
- Produces:
  - `StructSubgraph(kind: str, nodes: tuple[str, ...], background: int)` frozen dataclass.
  - `StructEpochPlan(seed: int, epoch: int, subgraphs: tuple[StructSubgraph, ...])` frozen dataclass.
  - `StructSampler(graph: nx.Graph, *, nodes: int, background_nodes: int, mix: Mapping[str, float], v_val: frozenset[str], exclude_nodes: frozenset[str])` with methods `plan(*, seed: int, epoch: int, count: int) -> StructEpochPlan`, `legal_mask(subgraph) -> NDArray[np.float32]` (n×n), `adjacency(subgraph) -> NDArray[np.float32]` (n×n, already masked), `statistics(subgraph) -> dict[str, float]`, and `coverage(plan) -> dict[str, float]` (`struct_positive_coverage`, `struct_positive_reuse`).
  - Statistics keys: `struct_nodes`, `struct_components`, `struct_legal_fraction`, `struct_positive_fraction`, `struct_triangles`, `struct_open_wedges`, `struct_background`, plus `struct_kind_bfs`, `struct_kind_motif`, `struct_kind_bridge` (one-hot).

- [ ] **Step 1: Write the failing sampler tests**

```python
# tests/test_struct_sampler.py
from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pytest

from src.data.struct_sampler import StructEpochPlan, StructSampler, StructSubgraph


def _toy_graph(seed: int = 3, n: int = 300, m: int = 3) -> nx.Graph:
    graph = nx.barabasi_albert_graph(n, m, seed=seed)
    return nx.relabel_nodes(graph, {i: f"node_{i:06d}" for i in graph.nodes})


def _sampler(**overrides: object) -> StructSampler:
    graph = _toy_graph()
    v_val = frozenset(f"node_{i:06d}" for i in range(0, 300, 5))
    exclude = frozenset({"node_000007", "node_000011"})
    kwargs: dict[str, object] = {
        "nodes": 24,
        "background_nodes": 4,
        "mix": {"bfs": 0.5, "motif": 0.25, "bridge": 0.25},
        "v_val": v_val,
        "exclude_nodes": exclude,
    }
    kwargs.update(overrides)
    return StructSampler(graph, **kwargs)  # type: ignore[arg-type]


def test_plan_sizes_kinds_and_background_counts() -> None:
    sampler = _sampler()
    plan = sampler.plan(seed=0, epoch=1, count=50)
    assert isinstance(plan, StructEpochPlan)
    assert len(plan.subgraphs) == 50
    for subgraph in plan.subgraphs:
        assert isinstance(subgraph, StructSubgraph)
        assert len(subgraph.nodes) == 24
        assert len(set(subgraph.nodes)) == 24
        assert subgraph.background == 4
        assert subgraph.kind in {"bfs", "motif", "bridge"}
        assert not set(subgraph.nodes) & {"node_000007", "node_000011"}


def test_legal_mask_never_admits_forbidden_pairs() -> None:
    sampler = _sampler()
    v_val = sampler.v_val
    plan = sampler.plan(seed=1, epoch=2, count=40)
    for subgraph in plan.subgraphs:
        mask = sampler.legal_mask(subgraph)
        assert mask.shape == (24, 24)
        assert np.array_equal(mask, mask.T)
        assert float(np.diagonal(mask).sum()) == 0.0
        for i, j in itertools.combinations(range(24), 2):
            both_val = subgraph.nodes[i] in v_val and subgraph.nodes[j] in v_val
            assert bool(mask[i, j]) == (not both_val)
        adjacency = sampler.adjacency(subgraph)
        assert np.array_equal(adjacency, adjacency * mask)


def test_kind_mix_is_respected() -> None:
    plan = _sampler().plan(seed=5, epoch=1, count=400)
    kinds = [subgraph.kind for subgraph in plan.subgraphs]
    assert abs(kinds.count("bfs") / 400 - 0.5) < 0.05
    assert abs(kinds.count("motif") / 400 - 0.25) < 0.05
    assert abs(kinds.count("bridge") / 400 - 0.25) < 0.05


def test_motif_seeds_are_real_wedges_or_triangles() -> None:
    sampler = _sampler(mix={"bfs": 0.0, "motif": 1.0, "bridge": 0.0})
    graph = sampler.graph
    plan = sampler.plan(seed=2, epoch=1, count=30)
    for subgraph in plan.subgraphs:
        a, b, c = subgraph.nodes[:3]
        # Seed order is (centre, leaf, leaf) for a wedge or any triangle order.
        assert graph.has_edge(a, b) and graph.has_edge(a, c)


def test_bridge_roots_are_far_apart_and_balls_are_balanced() -> None:
    sampler = _sampler(mix={"bfs": 0.0, "motif": 0.0, "bridge": 1.0})
    graph = sampler.graph
    plan = sampler.plan(seed=9, epoch=1, count=20)
    for subgraph in plan.subgraphs:
        local = subgraph.nodes[: 24 - 4]
        first_root, second_root = local[0], local[10]
        assert nx.shortest_path_length(graph, first_root, second_root) >= 3


def test_plan_is_world_size_independent_and_epoch_dependent() -> None:
    sampler = _sampler()
    first = sampler.plan(seed=0, epoch=3, count=16)
    again = sampler.plan(seed=0, epoch=3, count=16)
    other = sampler.plan(seed=0, epoch=4, count=16)
    assert first == again
    assert first != other
    # Striping by rank is the consumer's job; the plan itself carries no rank.
    assert [s.nodes for s in first.subgraphs[0::2]] != [s.nodes for s in first.subgraphs[1::2]]


def test_statistics_and_coverage_keys() -> None:
    sampler = _sampler()
    plan = sampler.plan(seed=0, epoch=1, count=8)
    stats = sampler.statistics(plan.subgraphs[0])
    expected = {
        "struct_nodes", "struct_components", "struct_legal_fraction", "struct_positive_fraction",
        "struct_triangles", "struct_open_wedges", "struct_background",
        "struct_kind_bfs", "struct_kind_motif", "struct_kind_bridge",
    }
    assert set(stats) == expected
    assert stats["struct_nodes"] == 24.0
    assert 0.0 < stats["struct_legal_fraction"] <= 1.0
    coverage = sampler.coverage(plan)
    assert set(coverage) == {"struct_positive_coverage", "struct_positive_reuse"}
    assert 0.0 < coverage["struct_positive_coverage"] <= 1.0


def test_small_component_falls_back_to_second_root() -> None:
    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("b", "c"), ("d", "e"), ("e", "f"), ("f", "g"), ("g", "h")])
    graph.add_nodes_from([f"iso{i}" for i in range(10)])
    sampler = StructSampler(
        graph, nodes=8, background_nodes=2, mix={"bfs": 1.0, "motif": 0.0, "bridge": 0.0},
        v_val=frozenset(), exclude_nodes=frozenset(),
    )
    plan = sampler.plan(seed=0, epoch=1, count=5)
    assert all(len(s.nodes) == 8 for s in plan.subgraphs)


def test_rejects_bad_construction() -> None:
    with pytest.raises(ValueError, match="nodes"):
        _sampler(nodes=3, background_nodes=0)
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b"])
    with pytest.raises(ValueError, match="fewer than"):
        StructSampler(graph, nodes=8, background_nodes=2, mix={"bfs": 1.0, "motif": 0.0, "bridge": 0.0}, v_val=frozenset(), exclude_nodes=frozenset())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_struct_sampler.py -n0 -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.struct_sampler'`.

- [ ] **Step 3: Implement the sampler**

```python
# src/data/struct_sampler.py
"""Structural-stream subgraph sampler over the V_val-masked training graph.

Three subgraph kinds (BFS ball, wedge/triangle-seeded ball, two-ball bridge)
share one node budget: ``nodes - background_nodes`` locally expanded nodes plus
``background_nodes`` uniform draws. Every subgraph's randomness comes from one
blake2b-keyed generator per plan position, so a plan is identical on every
DDP rank and independent of world size.
Spec: ``docs/superpowers/specs/2026-09-07-structural-stream-topology-losses-design.md`` section 3.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from src.distill.context_sampler import _anchor_rng
from src.distill.struct_config import KINDS

_RETRIES = 32
_BRIDGE_MIN_DISTANCE = 3


@dataclass(frozen=True)
class StructSubgraph:
    """One sampled subgraph: kind, ordered node ids, and trailing background count."""

    kind: str
    nodes: tuple[str, ...]
    background: int


@dataclass(frozen=True)
class StructEpochPlan:
    """Every subgraph of one epoch, in plan order."""

    seed: int
    epoch: int
    subgraphs: tuple[StructSubgraph, ...]


class StructSampler:
    """Sample training subgraphs and expose their legal masks, targets and statistics."""

    def __init__(
        self,
        graph: nx.Graph,
        *,
        nodes: int,
        background_nodes: int,
        mix: Mapping[str, float],
        v_val: frozenset[str],
        exclude_nodes: frozenset[str],
    ) -> None:
        if nodes < 4 or not 0 <= background_nodes < nodes:
            raise ValueError(f"invalid struct sampler budget nodes={nodes} background={background_nodes}")
        if set(mix) != set(KINDS):
            raise ValueError(f"mix must name exactly {list(KINDS)}")
        self.graph = graph
        self.nodes = nodes
        self.background_nodes = background_nodes
        self.local_budget = nodes - background_nodes
        self.v_val = v_val
        self.exclude_nodes = exclude_nodes
        self._kinds = tuple(KINDS)
        self._shares = np.asarray([float(mix[kind]) for kind in self._kinds], dtype=np.float64)
        self._universe: tuple[str, ...] = tuple(
            sorted(node for node in graph.nodes if node not in exclude_nodes)
        )
        if len(self._universe) < nodes:
            raise ValueError(
                f"structural graph has fewer than {nodes} sampleable nodes ({len(self._universe)})"
            )
        self._neighbors: dict[str, tuple[str, ...]] = {
            node: tuple(sorted(n for n in graph.neighbors(node) if n not in exclude_nodes))
            for node in self._universe
        }
        self._degree_ge1 = tuple(n for n in self._universe if len(self._neighbors[n]) >= 1)
        self._degree_ge2 = tuple(n for n in self._universe if len(self._neighbors[n]) >= 2)
        self._edges = tuple(
            (u, v) for u in self._universe for v in self._neighbors[u] if u < v
        )
        if not self._degree_ge1:
            raise ValueError("structural graph has no edges among sampleable nodes")
        self._positive_count = len(self._edges)

    # ----------------------------------------------------------------- planning

    def plan(self, *, seed: int, epoch: int, count: int) -> StructEpochPlan:
        """Return ``count`` subgraphs for ``(seed, epoch)``; deterministic and rank-free."""
        if count < 0:
            raise ValueError("count must be non-negative")
        subgraphs = tuple(self._sample_one(seed=seed, epoch=epoch, position=t) for t in range(count))
        return StructEpochPlan(seed=seed, epoch=epoch, subgraphs=subgraphs)

    def _sample_one(self, *, seed: int, epoch: int, position: int) -> StructSubgraph:
        rng = _anchor_rng(f"struct:{position}", seed=seed, epoch=epoch)
        kind = self._kinds[int(rng.choice(len(self._kinds), p=self._shares))]
        if kind == "bfs":
            local = self._ball([self._choice(rng, self._degree_ge1)], self.local_budget, rng)
        elif kind == "motif":
            local = self._ball(list(self._motif_seed(rng)), self.local_budget, rng)
        else:
            local = self._bridge(rng)
        local = self._fill(local, self.local_budget, rng)
        present = set(local)
        pool = [node for node in self._universe if node not in present]
        background = [
            pool[int(i)] for i in rng.choice(len(pool), size=self.background_nodes, replace=False)
        ] if self.background_nodes else []
        return StructSubgraph(kind=kind, nodes=tuple(local + background), background=len(background))

    @staticmethod
    def _choice(rng: np.random.Generator, pool: Sequence[str]) -> str:
        return pool[int(rng.integers(len(pool)))]

    def _ball(self, seeds: list[str], budget: int, rng: np.random.Generator, taken: set[str] | None = None) -> list[str]:
        """Randomised-neighbour-order BFS from ``seeds`` up to ``budget`` nodes."""
        order: list[str] = []
        seen: set[str] = set(taken or ())
        queue: deque[str] = deque()
        for seed in seeds:
            if seed not in seen:
                seen.add(seed)
                order.append(seed)
                queue.append(seed)
        while queue and len(order) < budget:
            current = queue.popleft()
            neighbours = list(self._neighbors[current])
            rng.shuffle(neighbours)
            for neighbour in neighbours:
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                order.append(neighbour)
                queue.append(neighbour)
                if len(order) >= budget:
                    break
        return order[:budget]

    def _fill(self, local: list[str], budget: int, rng: np.random.Generator) -> list[str]:
        """Top up a short ball from fresh uniform roots until ``budget`` is met."""
        while len(local) < budget:
            root = self._choice(rng, self._degree_ge1)
            if root in local:
                pool = [node for node in self._universe if node not in local]
                root = self._choice(rng, pool)
            local = local + self._ball([root], budget - len(local), rng, taken=set(local))
        return local[:budget]

    def _motif_seed(self, rng: np.random.Generator) -> tuple[str, str, str]:
        """An open wedge (centre, leaf, leaf) or a closed triangle, 50/50."""
        if rng.random() < 0.5:
            return self._wedge(rng)
        for _ in range(_RETRIES):
            u, v = self._edges[int(rng.integers(len(self._edges)))]
            common = sorted(set(self._neighbors[u]) & set(self._neighbors[v]))
            if common:
                return u, v, self._choice(rng, common)
        return self._wedge(rng)

    def _wedge(self, rng: np.random.Generator) -> tuple[str, str, str]:
        pool = self._degree_ge2 or self._degree_ge1
        for _ in range(_RETRIES):
            centre = self._choice(rng, pool)
            neighbours = self._neighbors[centre]
            if len(neighbours) < 2:
                continue
            i, j = rng.choice(len(neighbours), size=2, replace=False)
            left, right = neighbours[int(i)], neighbours[int(j)]
            if right not in self._neighbors[left]:
                return centre, left, right
        centre = self._choice(rng, pool)
        neighbours = self._neighbors[centre]
        return centre, neighbours[0], neighbours[-1]

    def _bridge(self, rng: np.random.Generator) -> list[str]:
        """Two balls of ``local_budget // 2`` from roots at distance >= 3 when possible."""
        half = self.local_budget // 2
        first = self._choice(rng, self._degree_ge1)
        second = first
        for _ in range(_RETRIES):
            candidate = self._choice(rng, self._degree_ge1)
            if candidate == first:
                continue
            near = nx.single_source_shortest_path_length(
                self.graph, first, cutoff=_BRIDGE_MIN_DISTANCE - 1
            )
            if candidate not in near:
                second = candidate
                break
        if second == first:
            pool = [node for node in self._degree_ge1 if node != first]
            second = self._choice(rng, pool)
        ball_a = self._ball([first], half, rng)
        ball_b = self._ball([second], self.local_budget - len(ball_a), rng, taken=set(ball_a))
        merged = ball_a + ball_b
        if len(merged) < self.local_budget:
            merged = merged + self._ball([first], self.local_budget - len(merged), rng, taken=set(merged))
        return merged

    # ----------------------------------------------------------------- tensors

    def legal_mask(self, subgraph: StructSubgraph) -> NDArray[np.float32]:
        """Symmetric 0/1 legal-pair mask with a zero diagonal."""
        nodes = subgraph.nodes
        in_val = np.asarray([node in self.v_val for node in nodes], dtype=bool)
        excluded = np.asarray([node in self.exclude_nodes for node in nodes], dtype=bool)
        mask = ~(in_val[:, None] & in_val[None, :]) & ~excluded[:, None] & ~excluded[None, :]
        np.fill_diagonal(mask, False)
        return mask.astype(np.float32)

    def adjacency(self, subgraph: StructSubgraph) -> NDArray[np.float32]:
        """Induced loopless adjacency of the structural graph, multiplied by the legal mask."""
        nodes = subgraph.nodes
        index = {node: i for i, node in enumerate(nodes)}
        adjacency = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
        for i, node in enumerate(nodes):
            for neighbour in self._neighbors.get(node, ()):
                j = index.get(neighbour)
                if j is not None and j != i:
                    adjacency[i, j] = 1.0
        return adjacency * self.legal_mask(subgraph)

    # ----------------------------------------------------------------- telemetry

    def statistics(self, subgraph: StructSubgraph) -> dict[str, float]:
        """Per-subgraph sampler telemetry (spec section 3.4)."""
        mask = self.legal_mask(subgraph)
        adjacency = self.adjacency(subgraph)
        n = len(subgraph.nodes)
        degrees = adjacency.sum(axis=1)
        triangles = float(np.trace(adjacency @ adjacency @ adjacency) / 6.0)
        open_wedges = float((degrees * (degrees - 1.0) / 2.0).sum() - 3.0 * triangles)
        induced = nx.from_numpy_array(adjacency)
        stats = {
            "struct_nodes": float(n),
            "struct_components": float(nx.number_connected_components(induced)),
            "struct_legal_fraction": float(mask.sum() / max(1.0, n * (n - 1))),
            "struct_positive_fraction": float(adjacency.sum() / max(1.0, mask.sum())),
            "struct_triangles": triangles,
            "struct_open_wedges": open_wedges,
            "struct_background": float(subgraph.background),
        }
        for kind in self._kinds:
            stats[f"struct_kind_{kind}"] = float(subgraph.kind == kind)
        return stats

    def coverage(self, plan: StructEpochPlan) -> dict[str, float]:
        """Fraction of sampleable positives seen at least once, and their mean reuse."""
        counts: dict[tuple[str, str], int] = {}
        for subgraph in plan.subgraphs:
            adjacency = self.adjacency(subgraph)
            rows, cols = np.nonzero(np.triu(adjacency, 1))
            for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
                u, v = subgraph.nodes[i], subgraph.nodes[j]
                key = (u, v) if u < v else (v, u)
                counts[key] = counts.get(key, 0) + 1
        seen = len(counts)
        return {
            "struct_positive_coverage": seen / max(1, self._positive_count),
            "struct_positive_reuse": (sum(counts.values()) / seen) if seen else 0.0,
        }


__all__ = ["StructEpochPlan", "StructSampler", "StructSubgraph"]
```

Note for `test_bridge_roots_are_far_apart_and_balls_are_balanced`: the first root is `local[0]` and the second is `local[half]` with `half = 20 // 2 = 10`; the test's `local[10]` relies on the first ball being exactly `half` nodes, which holds on the Barabási–Albert toy graph (a giant component). If the test flakes, assert on `local[len(ball_a)]` by exposing nothing new: reduce `count` to graphs where every ball fills.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_struct_sampler.py -n0 -q`
Expected: all PASS. `test_kind_mix_is_respected` is statistical over 400 draws at ±0.05; it is deterministic for fixed seeds, so a failure means the mix draw is wrong, not noise.

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/python -m ruff check src/data/struct_sampler.py tests/test_struct_sampler.py && .venv/bin/python -m ruff format src tests && .venv/bin/python -m mypy src/data/struct_sampler.py tests/test_struct_sampler.py`

```bash
git add src/data/struct_sampler.py tests/test_struct_sampler.py
git commit -m "feat(struct): deterministic BFS / motif / bridge subgraph sampler"
```

---

### Task 4: `StructStream` (forward, loss, telemetry)

**Files:**
- Modify: `src/train_b0.py` — add `StructStream` immediately after the `KDContextStream` class (after its `epoch_telemetry` method, before the next top-level definition); extend `_term_grad_norms` with a reusable `_grad_norm`.
- Test: `tests/test_train_b0_struct.py` (extend)

**Interfaces:**
- Consumes: `StructConfig`, `StructSampler`, `StructEpochPlan`, `struct_total`, `hard_struct_errors`, `PackedFeatureTable.gather_nodes(node_indices, boundary)`, `BUCKET_BOUNDARIES`, `_unwrapped_model`, `checkpoint`.
- Produces: `StructStream(config, sampler, table, *, rank, world_size, token_budget, positive_weight, label_smoothing, seed, val_sampler=None)` with
  - `loss(model, *, epoch, step, steps) -> tuple[Tensor, dict[str, float]]` (weighted total, rank-local stat sums),
  - attribute `last_terms: dict[str, Tensor]` (raw active terms of the last `loss` call),
  - `epoch_telemetry(accelerator, sums) -> dict[str, float]`,
  - `validation_telemetry(model, accelerator, *, threshold: float | None) -> dict[str, float]`,
  - `_grad_norm(term: Tensor, model: nn.Module) -> float` module-level helper (extracted from `_term_grad_norms`).

- [ ] **Step 1: Write the failing stream tests**

Append to `tests/test_train_b0_struct.py`:

```python
import networkx as nx
import numpy as np
import torch
from torch import nn
from accelerate import Accelerator

from src.data.packed_features import PackedFeatureManifest, PackedFeatureTable, PackedNodeRecord
from src.data.struct_sampler import StructSampler
from src.distill.struct_config import StructConfig
from src.distill.struct_losses import struct_total
from src.train_b0 import StructStream, _grad_norm


class _StructToy(nn.Module):
    """Logit = w * (sum emb_a - sum emb_b)^2 so every pair depends on one weight."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))
        self.forward_calls = 0
        self.batch_sizes: list[int] = []

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.forward_calls += 1
        self.batch_sizes.append(int(batch["emb_a"].shape[0]))
        a = batch["emb_a"].sum(dim=(1, 2))
        b = batch["emb_b"].sum(dim=(1, 2))
        return {"logits": self.weight * (a - b).square() - 1.0}


def _struct_fixture(n_nodes: int = 12) -> tuple[StructSampler, PackedFeatureTable]:
    graph = nx.relabel_nodes(
        nx.barabasi_albert_graph(n_nodes, 2, seed=1), {i: f"n{i}" for i in range(n_nodes)}
    )
    sampler = StructSampler(
        graph, nodes=8, background_nodes=2, mix={"bfs": 0.5, "motif": 0.25, "bridge": 0.25},
        v_val=frozenset({"n0", "n1"}), exclude_nodes=frozenset(),
    )
    node_ids = [f"n{i}" for i in range(n_nodes)]
    lengths = [1 + (i % 3) for i in range(n_nodes)]  # three length buckets
    records = tuple(
        PackedNodeRecord(node, 0, sum(lengths[:i]), sum(lengths[:i]), lengths[i])
        for i, node in enumerate(node_ids)
    )
    manifest = PackedFeatureManifest(
        format="test", input_dim=1, dtype="bfloat16", source_metadata_sha256="",
        source_index_sha256="", nodes=records, shards=(), pack_workers=1, build_seconds=0.0,
    )
    tokens = torch.arange(1, sum(lengths) + 1, dtype=torch.float32).unsqueeze(-1)
    offsets = torch.tensor([sum(lengths[:i]) for i in range(n_nodes)])
    table = PackedFeatureTable(tokens, offsets, torch.tensor(lengths), manifest)
    return sampler, table


def _stream(
    sampler: StructSampler, table: PackedFeatureTable, *, rank: int = 0, world_size: int = 1,
    weights: dict[str, float] | None = None, token_budget: int = 1 << 20, val_sampler: StructSampler | None = None,
) -> StructStream:
    config = StructConfig.from_mapping(
        {"nodes": 8, "background_nodes": 2, "val_subgraphs": 4,
         "weights": weights or {"bce": 1.0, "rank": 1.0, "degree": 0.1, "motif": 0.1}}
    )
    return StructStream(
        config, sampler, table, rank=rank, world_size=world_size, token_budget=token_budget,
        positive_weight=5.0, label_smoothing=0.0, seed=0, val_sampler=val_sampler,
    )


def test_stream_assembles_the_same_logits_as_a_direct_forward() -> None:
    sampler, table = _struct_fixture()
    stream = _stream(sampler, table, token_budget=3)  # forces many small chunks
    model = _StructToy()
    loss, stats = stream.loss(model, epoch=1, step=0, steps=4)
    subgraph = stream.last_subgraph
    assert subgraph is not None
    # Direct: score every legal pair in one call and rebuild the matrix.
    index = table.manifest.node_index()
    mask = torch.from_numpy(sampler.legal_mask(subgraph))
    target = torch.from_numpy(sampler.adjacency(subgraph))
    n = len(subgraph.nodes)
    rows, cols = torch.triu_indices(n, n, offset=1)
    keep = mask[rows, cols] > 0
    rows, cols = rows[keep], cols[keep]
    boundary = max(table.manifest.nodes[index[node]].length for node in subgraph.nodes)
    emb_a, len_a = table.gather_nodes(torch.tensor([index[subgraph.nodes[i]] for i in rows.tolist()]), boundary)
    emb_b, len_b = table.gather_nodes(torch.tensor([index[subgraph.nodes[j]] for j in cols.tolist()]), boundary)
    direct = model({"emb_a": emb_a, "emb_b": emb_b, "len_a": len_a, "len_b": len_b})["logits"]
    logits = torch.zeros(n, n)
    logits[rows, cols] = direct
    logits[cols, rows] = direct
    expected, _ = struct_total(logits, target, mask, stream.config, positive_weight=5.0, label_smoothing=0.0)
    torch.testing.assert_close(loss, expected)
    assert stats["struct_pairs"] == float(len(rows))
    # Every stream chunk (all calls before the final direct forward) respects the budget of 3.
    assert max(model.batch_sizes[:-1]) <= 3


def test_one_rank_and_two_rank_gradients_match() -> None:
    sampler, table = _struct_fixture()
    steps = 3

    def _grads(rank: int, world_size: int) -> torch.Tensor:
        model = _StructToy()
        total = None
        for step in range(steps):
            loss, _ = _stream(sampler, table, rank=rank, world_size=world_size).loss(
                model, epoch=2, step=step, steps=steps
            )
            total = loss if total is None else total + loss
        assert total is not None
        (grad,) = torch.autograd.grad(total, [model.weight])
        return grad

    single = _grads(0, 1)
    two = _grads(0, 2) + _grads(1, 2)
    torch.testing.assert_close(single, two)


def test_plan_longer_than_steps_is_rejected_and_shorter_plan_yields_zero_steps() -> None:
    sampler, table = _struct_fixture()
    long_config = StructConfig.from_mapping({"nodes": 8, "background_nodes": 2, "subgraphs_per_epoch": 9, "weights": {"bce": 1.0}})
    stream = StructStream(long_config, sampler, table, rank=0, world_size=1, token_budget=1 << 20,
                          positive_weight=5.0, label_smoothing=0.0, seed=0)
    with pytest.raises(ValueError, match="subgraphs_per_epoch"):
        stream.loss(_StructToy(), epoch=1, step=0, steps=4)
    short_config = StructConfig.from_mapping({"nodes": 8, "background_nodes": 2, "subgraphs_per_epoch": 1, "weights": {"bce": 1.0}})
    stream = StructStream(short_config, sampler, table, rank=0, world_size=1, token_budget=1 << 20,
                          positive_weight=5.0, label_smoothing=0.0, seed=0)
    model = _StructToy()
    _, first = stream.loss(model, epoch=1, step=0, steps=4)
    zero, second = stream.loss(model, epoch=1, step=3, steps=4)
    assert first["struct_pairs"] > 0
    assert second["struct_pairs"] == 0.0
    assert zero.requires_grad and float(zero) == 0.0


def test_epoch_and_validation_telemetry_keys() -> None:
    sampler, table = _struct_fixture()
    stream = _stream(sampler, table, val_sampler=sampler)
    model = _StructToy()
    accelerator = Accelerator(cpu=True)
    sums: dict[str, float] = {}
    for step in range(2):
        _, stats = stream.loss(model, epoch=1, step=step, steps=2)
        for key, value in stats.items():
            sums[key] = sums.get(key, 0.0) + value
    telemetry = stream.epoch_telemetry(accelerator, sums)
    for key in ("struct_bce_loss", "struct_rank_loss", "struct_degree_loss", "struct_motif_loss",
                "struct_pairs", "struct_positive_coverage", "struct_positive_reuse",
                "struct_components", "struct_legal_fraction", "struct_kind_bfs"):
        assert key in telemetry
    val = stream.validation_telemetry(model, accelerator, threshold=0.0)
    for key in ("val_struct_bce_loss", "val_struct_motif_loss", "val_struct_hard_degree_mae",
                "val_struct_hard_triangle_mae", "val_struct_hard_wedge_mae"):
        assert key in val
    assert stream.last_terms.keys() == {"bce", "rank", "degree", "motif"}
    assert _grad_norm(stream.last_terms["bce"], model) >= 0.0
```


- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_train_b0_struct.py -n0 -q`
Expected: FAIL with `ImportError: cannot import name 'StructStream' from 'src.train_b0'`.

- [ ] **Step 3: Implement `StructStream` and `_grad_norm`**

Imports at the top of `src/train_b0.py` (keep alphabetical within the `src.` group):

```python
from src.data.struct_sampler import StructEpochPlan, StructSampler, StructSubgraph
from src.distill.struct_config import StructConfig
from src.distill.struct_losses import hard_struct_errors, struct_total
```

Insert after `KDContextStream.epoch_telemetry`:

```python
class StructStream:
    """One sampled training subgraph per optimizer step, scored as a logit matrix.

    Sibling of :class:`KDContextStream`: the plan for ``(seed, epoch)`` is
    identical on every rank; rank ``r`` of ``W`` takes plan positions
    ``r, r+W, ...`` and spreads them across its steps exactly once. Legal pairs
    are bucketed by token boundary, chunked to the token budget, and forwarded
    through the unwrapped model under activation checkpointing; the loss is
    computed on the assembled ``n x n`` matrix.
    """

    def __init__(
        self,
        config: StructConfig,
        sampler: StructSampler,
        table: PackedFeatureTable,
        *,
        rank: int,
        world_size: int,
        token_budget: int,
        positive_weight: float,
        label_smoothing: float,
        seed: int,
        val_sampler: StructSampler | None = None,
    ) -> None:
        if token_budget < 1:
            raise ValueError(f"struct token budget must be positive, got {token_budget}")
        if rank < 0 or rank >= world_size or world_size < 1:
            raise ValueError(f"invalid struct rank/world size: {rank}/{world_size}")
        self.config = config
        self._sampler = sampler
        self._val_sampler = val_sampler
        self._table = table
        self._rank = rank
        self._world_size = world_size
        self._token_budget = token_budget
        self._positive_weight = float(positive_weight)
        self._label_smoothing = float(label_smoothing)
        self._seed = int(seed)
        self._node_index = table.manifest.node_index()
        self._plan: StructEpochPlan | None = None
        self._plan_coverage: dict[str, float] = {}
        self._val_plan: StructEpochPlan | None = None
        self.last_terms: dict[str, torch.Tensor] = {}
        self.last_subgraph: StructSubgraph | None = None

    # ------------------------------------------------------------ planning

    def _epoch_plan(self, epoch: int, steps: int) -> StructEpochPlan:
        count = self.config.subgraphs_per_epoch
        if count is None:
            count = steps
        if count > steps:
            raise ValueError(
                f"struct.subgraphs_per_epoch ({count}) exceeds the epoch's {steps} optimizer steps"
            )
        if self._plan is None or self._plan.epoch != epoch or len(self._plan.subgraphs) != count:
            self._plan = self._sampler.plan(seed=self._seed, epoch=epoch, count=count)
            self._plan_coverage = self._sampler.coverage(self._plan)
        return self._plan

    def _positions(self, size: int, *, rank: int, steps: int, step: int) -> list[int]:
        shard = list(range(rank, size, self._world_size))
        start, stop = KDContextStream._step_slice(len(shard), steps, step)
        return shard[start:stop]

    # ------------------------------------------------------------ forward

    def _score(self, model: nn.Module, subgraph: StructSubgraph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the assembled symmetric logit matrix, the target, and the mask."""
        device = self._table.tokens.device
        mask_np = self._sampler.legal_mask(subgraph)
        target = torch.from_numpy(self._sampler.adjacency(subgraph)).to(device)
        mask = torch.from_numpy(mask_np).to(device)
        n = len(subgraph.nodes)
        packed = [self._node_index[node] for node in subgraph.nodes]
        lengths = self._table.manifest.nodes
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if mask_np[i, j] > 0]
        logits = torch.zeros(n, n, dtype=torch.float32, device=device)
        if not pairs:
            return logits, target, mask
        buckets: dict[int, list[int]] = {boundary: [] for boundary in BUCKET_BOUNDARIES}
        for row, (i, j) in enumerate(pairs):
            max_length = max(lengths[packed[i]].length, lengths[packed[j]].length)
            boundary = next((value for value in BUCKET_BOUNDARIES if max_length <= value), None)
            if boundary is None:
                raise ValueError(f"struct packed length {max_length} exceeds {BUCKET_BOUNDARIES[-1]}")
            buckets[boundary].append(row)
        raw_model = _unwrapped_model(model)
        parts: list[torch.Tensor] = []
        rows_a: list[int] = []
        rows_b: list[int] = []
        for boundary, rows in buckets.items():
            per_chunk = max(1, self._token_budget // boundary)
            for start in range(0, len(rows), per_chunk):
                chunk = rows[start : start + per_chunk]
                anchor = torch.as_tensor([packed[pairs[r][0]] for r in chunk], dtype=torch.int64, device=device)
                partner = torch.as_tensor([packed[pairs[r][1]] for r in chunk], dtype=torch.int64, device=device)

                def forward(anchor: torch.Tensor, partner: torch.Tensor, boundary: int) -> torch.Tensor:
                    # `boundary` is explicit: a closure over the loop variable would
                    # recompute every chunk at the last bucket.
                    emb_a, len_a = self._table.gather_nodes(anchor, boundary)
                    emb_b, len_b = self._table.gather_nodes(partner, boundary)
                    output = cast(
                        dict[str, torch.Tensor],
                        raw_model({"emb_a": emb_a, "emb_b": emb_b, "len_a": len_a, "len_b": len_b}),
                    )
                    out = output["logits"]
                    if out.dim() > 1 and out.size(-1) == 1:
                        out = out.squeeze(-1)
                    return out.float()

                if torch.is_grad_enabled():
                    chunk_logits = cast(torch.Tensor, checkpoint(forward, anchor, partner, boundary, use_reentrant=False))
                else:
                    chunk_logits = forward(anchor, partner, boundary)
                parts.append(chunk_logits)
                rows_a.extend(pairs[r][0] for r in chunk)
                rows_b.extend(pairs[r][1] for r in chunk)
        flat = torch.cat(parts)
        index_a = torch.as_tensor(rows_a, dtype=torch.int64, device=device)
        index_b = torch.as_tensor(rows_b, dtype=torch.int64, device=device)
        logits = logits.index_put((index_a, index_b), flat).index_put((index_b, index_a), flat)
        return logits, target, mask

    # ------------------------------------------------------------ training

    def loss(self, model: nn.Module, *, epoch: int, step: int, steps: int) -> tuple[torch.Tensor, dict[str, float]]:
        """Weighted structural loss for this rank's subgraph at ``step`` (zero when none)."""
        plan = self._epoch_plan(epoch, steps)
        positions = self._positions(len(plan.subgraphs), rank=self._rank, steps=steps, step=step)
        zero = next(model.parameters()).sum() * 0.0
        self.last_terms = {}
        self.last_subgraph = None
        stats: dict[str, float] = {"struct_pairs": 0.0, "struct_subgraphs": 0.0}
        if not positions:
            return zero, stats
        total = zero
        for position in positions:
            subgraph = plan.subgraphs[position]
            self.last_subgraph = subgraph
            logits, target, mask = self._score(model, subgraph)
            pair_count = float(torch.triu(mask, diagonal=1).sum().item())
            if pair_count == 0.0:
                continue
            term_total, terms = struct_total(
                logits, target, mask, self.config,
                positive_weight=self._positive_weight, label_smoothing=self._label_smoothing,
            )
            total = total + term_total
            self.last_terms = terms
            stats["struct_pairs"] += pair_count
            stats["struct_subgraphs"] += 1.0
            for key, term in terms.items():
                stats[f"sum_{key}"] = stats.get(f"sum_{key}", 0.0) + float(term.detach().item())
            for key, value in self._sampler.statistics(subgraph).items():
                stats[f"sum_{key}"] = stats.get(f"sum_{key}", 0.0) + value
        return total, stats

    def epoch_telemetry(self, accelerator: Accelerator, sums: dict[str, float]) -> dict[str, float]:
        """Reduce rank-local sums into per-subgraph means plus plan coverage."""
        keys = sorted(sums)
        reduced = accelerator.reduce(
            torch.tensor([sums[key] for key in keys], device=accelerator.device, dtype=torch.float64),
            reduction="sum",
        )
        values = {key: float(reduced[index].item()) for index, key in enumerate(keys)}
        count = max(values.get("struct_subgraphs", 0.0), 1.0)
        telemetry: dict[str, float] = {"struct_pairs": values.get("struct_pairs", 0.0)}
        for key in self.config.active_weights:
            telemetry[f"struct_{key}_loss"] = values.get(f"sum_{key}", 0.0) / count
        for key, value in values.items():
            if key.startswith("sum_struct_"):
                telemetry[key[len("sum_") :]] = value / count
        telemetry.update(self._plan_coverage)
        return telemetry

    # ------------------------------------------------------------ validation

    def validation_telemetry(self, model: nn.Module, accelerator: Accelerator, *, threshold: float | None) -> dict[str, float]:
        """Score the fixed V_val diagnostic subgraphs (this rank's stripe) and reduce means."""
        if self._val_sampler is None or self.config.val_subgraphs == 0:
            return {}
        if self._val_plan is None:
            self._val_plan = self._val_sampler.plan(seed=self._seed, epoch=0, count=self.config.val_subgraphs)
        positions = list(range(self._rank, len(self._val_plan.subgraphs), self._world_size))
        was_training = model.training
        model.eval()
        sums: dict[str, float] = {"count": 0.0}
        with torch.no_grad():
            for position in positions:
                subgraph = self._val_plan.subgraphs[position]
                logits, target, mask = self._score(model, subgraph)
                _, terms = struct_total(
                    logits, target, mask, self.config,
                    positive_weight=self._positive_weight, label_smoothing=self._label_smoothing,
                )
                sums["count"] += 1.0
                for key, term in terms.items():
                    sums[f"val_struct_{key}_loss"] = sums.get(f"val_struct_{key}_loss", 0.0) + float(term.item())
                if threshold is not None:
                    for key, value in hard_struct_errors(logits, target, mask, threshold).items():
                        sums[f"val_struct_{key}"] = sums.get(f"val_struct_{key}", 0.0) + value
        model.train(was_training)
        keys = sorted(sums)
        reduced = accelerator.reduce(
            torch.tensor([sums[key] for key in keys], device=accelerator.device, dtype=torch.float64),
            reduction="sum",
        )
        values = {key: float(reduced[index].item()) for index, key in enumerate(keys)}
        count = max(values.pop("count", 0.0), 1.0)
        return {key: value / count for key, value in values.items()}
```

Then refactor the gradient probe:

```python
def _grad_norm(term: torch.Tensor, model: nn.Module) -> float:
    """L2 norm of ``term``'s gradient over trainable parameters (0.0 when it has no graph)."""
    if not term.requires_grad:
        return 0.0
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    squares = [g.float().pow(2).sum() for g in grads if g is not None]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt().item())


def _term_grad_norms(
    task_loss: torch.Tensor, kd_loss: torch.Tensor, model: nn.Module
) -> tuple[float, float]:
    """(docstring unchanged)"""
    return _grad_norm(task_loss, model), _grad_norm(kd_loss, model)
```

Keep the original docstring of `_term_grad_norms` verbatim.

Two things the tests will catch if wrong: `validation_telemetry` must call `_score` under `torch.no_grad()` (the checkpoint branch is skipped because grad is disabled), and `epoch_telemetry` must divide sampler statistics by the subgraph count, not the step count.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_train_b0_struct.py tests/test_train_b0_kd.py -n0 -q -k "struct or grad_norm or context_stream"`
Expected: PASS. The existing context-stream tests must still pass after the `_term_grad_norms` refactor.

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/python -m ruff check src/train_b0.py tests/test_train_b0_struct.py && .venv/bin/python -m ruff format src tests && .venv/bin/python -m mypy src/train_b0.py tests/test_train_b0_struct.py`

```bash
git add src/train_b0.py tests/test_train_b0_struct.py
git commit -m "feat(struct): StructStream forwards one subgraph per step under checkpointing"
```

---

### Task 5: Wire the stream into the training loop and the DDP worker

**Files:**
- Modify: `src/train_b0.py` — `train_ddp_loop` signature and docstring (`:3448-3510`), epoch-state init (`:3730-3745`), step block (`:3796-3826`), epoch telemetry (`:3940-4023`), `_run_ddp_worker` construction (`:4701-4741`) and the `train_ddp_loop(...)` call site in the worker.
- Test: `tests/test_train_b0_struct.py` (extend)

**Interfaces:**
- Consumes: `StructStream` (Task 4), `_grad_norm`.
- Produces: `train_ddp_loop(..., struct_stream: StructStream | None = None)`; `metrics.jsonl` keys `train_struct_loss`, `struct_<key>_loss`, `grad_norm_struct_<key>`, `struct_pairs`, `struct_seconds`, `struct_wall_fraction`, the `struct_*` sampler statistics, `struct_positive_coverage`, `struct_positive_reuse`, and on topology-due epochs `val_struct_*`.

- [ ] **Step 1: Write the failing loop tests**

Append to `tests/test_train_b0_struct.py`:

```python
from dataclasses import replace

from src.train_b0 import Config, ValidationOutcome, train_ddp_loop
from tests.test_train_b0 import _TinyPairMLP, _batch_of, _constant_metrics, _make_synthetic_pair_dataset, _tiny_config


def _loop(cfg: Config, tmp_path: Path, subdir: str, struct_stream: StructStream | None):
    torch.manual_seed(7)
    model = _TinyPairMLP(input_dim=4, hidden_dims=(8,), dropout=0.0)
    batch = _batch_of(_make_synthetic_pair_dataset(8, input_dim=4, seed=1))
    batch["_row_id"] = torch.arange(8)
    batch["_local_pair_count"] = torch.tensor(8)
    batch["_global_pair_count"] = torch.tensor(8)
    return train_ddp_loop(
        model, lambda epoch: [batch], [batch], cfg, Accelerator(cpu=True),
        warmup_steps=1, artifact_dir=tmp_path / subdir,
        evaluate_fn=lambda model, loader, accelerator: ValidationOutcome(_constant_metrics(), None),
        struct_stream=struct_stream,
    )


def test_absent_struct_block_is_bit_identical_to_before(tmp_path: Path) -> None:
    cfg = _tiny_config(epochs=2)
    base = _loop(cfg, tmp_path, "base", None).last_state_dict
    again = _loop(replace(cfg, struct=None), tmp_path, "again", None).last_state_dict
    for key in base:
        torch.testing.assert_close(base[key], again[key], rtol=0.0, atol=0.0)
    rows = [json.loads(line) for line in (tmp_path / "base" / "metrics.jsonl").read_text().splitlines()]
    assert not any(key.startswith("struct_") or key.startswith("val_struct_") for key in rows[-1])


def test_struct_stream_changes_weights_and_logs_keys(tmp_path: Path) -> None:
    sampler, table = _struct_fixture()
    cfg = replace(_tiny_config(epochs=2), struct=StructConfig.from_mapping({"nodes": 8, "background_nodes": 2, "val_subgraphs": 2, "weights": {"bce": 1.0, "motif": 0.1}}))

    class _MLPOnTokens(_TinyPairMLP):
        """Serve both batch contracts: the task batch (`x_a`/`x_b`) and the stream's packed
        (B, L, 1) `emb_a`/`emb_b` tokens, mean-pooled and widened to `input_dim`."""

        def forward(
            self, batch: dict[str, torch.Tensor] | None = None, **kwargs: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            merged = dict(batch or {})
            merged.update(kwargs)
            if "emb_a" in merged:
                merged = {
                    "x_a": merged["emb_a"].mean(dim=1).expand(-1, 4),
                    "x_b": merged["emb_b"].mean(dim=1).expand(-1, 4),
                }
            return super().forward(merged)

    torch.manual_seed(7)
    model = _MLPOnTokens(input_dim=4, hidden_dims=(8,), dropout=0.0)
    stream = StructStream(cfg.struct, sampler, table, rank=0, world_size=1, token_budget=1 << 20,
                          positive_weight=5.0, label_smoothing=0.0, seed=0, val_sampler=sampler)
    batch = _batch_of(_make_synthetic_pair_dataset(8, input_dim=4, seed=1))
    batch["_row_id"] = torch.arange(8)
    batch["_local_pair_count"] = torch.tensor(8)
    batch["_global_pair_count"] = torch.tensor(8)
    result = train_ddp_loop(
        model, lambda epoch: [batch], [batch], cfg, Accelerator(cpu=True),
        warmup_steps=1, artifact_dir=tmp_path / "struct",
        evaluate_fn=lambda model, loader, accelerator: ValidationOutcome(_constant_metrics(), None),
        struct_stream=stream,
    )
    rows = [json.loads(line) for line in (tmp_path / "struct" / "metrics.jsonl").read_text().splitlines()]
    last = rows[-1]
    for key in ("train_struct_loss", "struct_bce_loss", "struct_motif_loss", "grad_norm_struct_bce",
                "grad_norm_struct_motif", "struct_pairs", "struct_seconds", "struct_wall_fraction",
                "struct_components", "struct_positive_coverage", "val_struct_bce_loss"):
        assert key in last, key
    assert result.last_state_dict is not None
```

The helpers exist in `tests/test_train_b0.py` (`_make_synthetic_pair_dataset` at line 164, `_batch_of` at 185, `_TinyPairMLP` at 189 with the `x_a`/`x_b` contract, `_constant_metrics` at 2331). Add `import json` at the top of the test file.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_train_b0_struct.py -n0 -q -k "bit_identical or changes_weights"`
Expected: FAIL with `TypeError: train_ddp_loop() got an unexpected keyword argument 'struct_stream'`.

- [ ] **Step 3: Wire the loop**

In `train_ddp_loop`:

1. Signature: after `kd_context_stream: KDContextStream | None = None,` add `struct_stream: StructStream | None = None,`. Docstring: add `struct_stream: Optional structural subgraph stream; its weighted loss joins the shared backward and its terms are logged as ``struct_*``.`
2. Epoch-state init (next to `epoch_kd_sums`): add

```python
        epoch_struct_loss_sum = 0.0
        epoch_struct_sums: dict[str, float] = {}
        epoch_struct_seconds = 0.0
        grad_norm_struct: dict[str, float] = {}
```

3. The `Sized` check: change `if kd_context_stream is not None and not isinstance(epoch_loader, Sized):` to `if (kd_context_stream is not None or struct_stream is not None) and not isinstance(epoch_loader, Sized):` and the message to `"kd_rank and struct streams require a training loader with a known step count"`.
4. Step block: immediately after the `if kd_loss is not None:` block (after `epoch_kd_loss_sum += ...`) and before `_all_ranks_loss_finite`, add

```python
            if struct_stream is not None:
                struct_start = time.monotonic()
                struct_loss, struct_stats = struct_stream.loss(
                    model, epoch=epoch, step=epoch_steps, steps=epoch_step_count
                )
                for key, value in struct_stats.items():
                    epoch_struct_sums[key] = epoch_struct_sums.get(key, 0.0) + value
                if epoch_steps == 0:
                    grad_norm_struct = {
                        key: _grad_norm(term, model) for key, term in struct_stream.last_terms.items()
                    }
                loss = loss + struct_loss
                epoch_struct_loss_sum += float(struct_loss.detach().float().item())
                epoch_struct_seconds += time.monotonic() - struct_start
```

5. Epoch telemetry: after the `epoch_kd_telemetry` block and before `validation_start = time.monotonic()`, add

```python
        epoch_struct_telemetry: dict[str, float] = {}
        if struct_stream is not None and epoch_steps > 0:
            global_struct_loss = accelerator.reduce(
                torch.tensor(epoch_struct_loss_sum, device=accelerator.device, dtype=torch.float64),
                reduction="sum",
            )
            epoch_struct_telemetry["train_struct_loss"] = float(global_struct_loss.item()) / float(
                epoch_steps * world_size
            )
            epoch_struct_telemetry.update(struct_stream.epoch_telemetry(accelerator, epoch_struct_sums))
            struct_seconds = accelerator.gather(
                torch.tensor([epoch_struct_seconds], device=accelerator.device, dtype=torch.float64)
            )
            epoch_struct_telemetry["struct_seconds"] = float(struct_seconds.max().item())
            keys = sorted(grad_norm_struct)
            if keys:
                norms = accelerator.reduce(
                    torch.tensor([grad_norm_struct[k] for k in keys], device=accelerator.device, dtype=torch.float64),
                    reduction="mean",
                )
                for index, key in enumerate(keys):
                    epoch_struct_telemetry[f"grad_norm_struct_{key}"] = float(norms[index].item())
```

6. After `entry.update(epoch_kd_telemetry)` add

```python
        if epoch_struct_telemetry:
            epoch_wall = max(time.monotonic() - epoch_wall_start, 1e-9)
            epoch_struct_telemetry["struct_wall_fraction"] = epoch_struct_telemetry["struct_seconds"] / epoch_wall
            entry.update(epoch_struct_telemetry)
        if struct_stream is not None and run_topology:
            entry.update(
                struct_stream.validation_telemetry(
                    model,
                    accelerator,
                    threshold=outcome.topology.threshold if outcome.topology is not None else None,
                )
            )
```

`epoch_wall_start` already exists in the loop (set before the loader is built). If the variable that holds the loop's start time has another name, use that one.

7. In `_run_ddp_worker`, after the KD block (after the `logger.info("KD active: ...")` call) add

```python
    struct_stream: StructStream | None = None
    if cfg.struct is not None:
        model_kwargs = resolve_model_kwargs(cfg.model)
        struct_sampler = StructSampler(
            val_split.build_training_graph(),
            nodes=cfg.struct.nodes,
            background_nodes=cfg.struct.background_nodes,
            mix=cfg.struct.mix,
            v_val=val_split.v_val,
            exclude_nodes=assembled.exclude_nodes,
        )
        val_struct_sampler = StructSampler(
            val_split.build_g_val_simple(),
            nodes=cfg.struct.nodes,
            background_nodes=cfg.struct.background_nodes,
            mix=cfg.struct.mix,
            v_val=frozenset(),
            exclude_nodes=assembled.exclude_nodes,
        )
        struct_stream = StructStream(
            cfg.struct,
            struct_sampler,
            table,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            token_budget=cfg.data.token_budget,
            positive_weight=float(cast(float, model_kwargs.get("positive_weight", 1.0))),
            label_smoothing=float(cast(float, model_kwargs.get("label_smoothing", 0.0))),
            seed=cfg.seed,
            val_sampler=val_struct_sampler,
        )
        if accelerator.is_main_process:
            logger.info(
                "struct active: arm=%s weights=%s nodes=%d background=%d mix=%s",
                cfg.struct.arm,
                cfg.struct.active_weights,
                cfg.struct.nodes,
                cfg.struct.background_nodes,
                cfg.struct.mix,
            )
```

and pass `struct_stream=struct_stream,` in the worker's `train_ddp_loop(...)` call (find it with `grep -n "kd_context_stream=kd_context_stream" src/train_b0.py`).

`table` is the `PackedFeatureTable` already bound in the worker for the KD context stream; confirm its name with `grep -n "table = " src/train_b0.py | sed -n 1,5p` in `_run_ddp_worker`. If it is only built inside the `if cfg.distill` branch, hoist that construction above both branches.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_train_b0_struct.py tests/test_train_b0_kd.py tests/test_train_b0.py -n0 -q`
Expected: PASS, including the pre-existing `test_ddp_loop_distill_none_and_all_zero_are_bit_identical`.

- [ ] **Step 5: Run the fast suite, lint, type-check, commit**

Run: `.venv/bin/python -m pytest -m "not slow and not integration" -q --dist loadfile -n 4 && .venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests && .venv/bin/python -m mypy src tests`
Expected: clean.

```bash
git add src/train_b0.py tests/test_train_b0_struct.py
git commit -m "feat(struct): wire the structural stream into train_ddp_loop and the DDP worker"
```

---

### Task 6: Configs and the Optuna driver

**Files:**
- Create: `configs/struct_bce_breadth_first.yaml`, `configs/struct_grand_breadth_first.yaml`, `configs/struct_new_breadth_first.yaml`, `src/experiments/struct_hpo.py`
- Test: `tests/test_struct_hpo.py`

**Interfaces:**
- Consumes: `SweepSpec`, `run_sweep`, `print_report` from `src/experiments/kd_rank_strict_hpo.py` (`SweepSpec(study_name, n_startup_trials, priors, param_names, suggest, materialize, prepare)`; `run_sweep(args, spec)` reads `args.base_config`, `args.sweep_dir`, `args.n_trials`, `args.rd_band`).
- Produces: `ARMS: dict[str, ArmSpec]` with `ArmSpec(study_name, base_config, priors, param_names, suggest)`; `materialize_trial_config(base_config, params, trial_number, sweep_dir) -> Path`; `build_spec(args) -> SweepSpec`; CLI `python -m src.experiments.struct_hpo --arm {grand,new} [--n-trials 10] [--sweep-dir outputs/struct_hpo/<arm>] [--base-config ...] [--rd-band 0.05]`.

- [ ] **Step 1: Write the three configs**

`configs/struct_bce_breadth_first.yaml`: copy `configs/b1_kd_control_breadth_first.yaml` verbatim, replace the header comment with

```yaml
# Structural-stream baseline: V3.1 student, the control's task stream, plus one
# sampled 40-node training subgraph per optimizer step supervised by BCE only.
# The struct_grand and struct_new arms differ from this file only in struct.weights.
# Spec: docs/superpowers/specs/2026-09-07-structural-stream-topology-losses-design.md
```

set `output_dir: outputs/struct/bce`, and append

```yaml
struct:
  nodes: 40
  background_nodes: 8
  mix: {bfs: 0.5, motif: 0.25, bridge: 0.25}
  subgraphs_per_epoch: null
  weights: {bce: 1.0, gs: 0.0, rd: 0.0, deg_mmd: 0.0, rank: 0.0, degree: 0.0, motif: 0.0}
  rank_margin: 0.1
  rank_temperature: 1.0
  huber_delta: 1.0
  mmd_sigma: 1.25
  mmd_bins: 48
  val_subgraphs: 32
```

`configs/struct_grand_breadth_first.yaml`: same file with `output_dir: outputs/struct/grand`, `eval.topology_every: 2` (the sweep scores at cadence 2, matching `configs/autoresearch/*.yaml`), and `weights: {bce: 1.0, gs: 0.70, rd: 0.90, deg_mmd: 0.0, rank: 0.0, degree: 0.0, motif: 0.0}`.

`configs/struct_new_breadth_first.yaml`: same with `output_dir: outputs/struct/new`, `eval.topology_every: 2`, and `weights: {bce: 1.0, gs: 0.0, rd: 0.0, deg_mmd: 0.0, rank: 1.0, degree: 0.1, motif: 0.1}`.

Keep `eval.topology_every: 1` in the bce baseline (it is a single pipeline run, not a sweep point).

- [ ] **Step 2: Write the failing driver tests**

```python
# tests/test_struct_hpo.py
from __future__ import annotations

import argparse
from pathlib import Path

import optuna
import pytest
import yaml

from src.experiments import struct_hpo
from src.experiments.kd_rank_strict_hpo import SweepSpec


def test_arm_table_matches_spec() -> None:
    grand = struct_hpo.ARMS["grand"]
    new = struct_hpo.ARMS["new"]
    assert grand.study_name == "struct_grand" and new.study_name == "struct_new"
    assert grand.base_config == Path("configs/struct_grand_breadth_first.yaml")
    assert new.base_config == Path("configs/struct_new_breadth_first.yaml")
    assert grand.priors == ({"gs": 0.70, "rd": 0.90}, {"gs": 0.35, "rd": 0.45})
    assert new.priors == (
        {"rank": 1.0, "degree": 0.1, "motif": 0.1},
        {"rank": 1.0, "degree": 0.03, "motif": 0.03},
    )
    assert grand.param_names == ("gs", "rd")
    assert new.param_names == ("rank", "degree", "motif")
    assert struct_hpo.N_STARTUP_TRIALS == 3


@pytest.mark.parametrize("arm", ["grand", "new"])
def test_suggest_stays_inside_the_boxes(arm: str) -> None:
    spec = struct_hpo.ARMS[arm]
    study = optuna.create_study(directions=["maximize", "minimize"], sampler=optuna.samplers.RandomSampler(seed=0))
    boxes = struct_hpo.SEARCH_BOXES[arm]
    for _ in range(50):
        params = spec.suggest(study.ask())
        assert set(params) == set(spec.param_names)
        for name, value in params.items():
            low, high = boxes[name]
            assert low <= float(value) <= high


@pytest.mark.parametrize("arm", ["grand", "new"])
def test_materialize_overrides_only_struct_weights_and_output_dir(arm: str, tmp_path: Path) -> None:
    spec = struct_hpo.ARMS[arm]
    params = dict(spec.priors[0])
    path = struct_hpo.materialize_trial_config(spec.base_config, params, 7, tmp_path / "sweep")
    trial = yaml.safe_load(path.read_text())
    base = yaml.safe_load(spec.base_config.read_text())
    assert trial["output_dir"] == str(tmp_path / "sweep" / "trial_007")
    assert trial["struct"]["weights"]["bce"] == 1.0
    for name, value in params.items():
        assert trial["struct"]["weights"][name] == value
    for key in trial["struct"]["weights"]:
        if key not in params and key != "bce":
            assert trial["struct"]["weights"][key] == 0.0
    trial["output_dir"] = base["output_dir"]
    trial["struct"]["weights"] = base["struct"]["weights"]
    assert trial == base


def test_materialize_rejects_illegal_weights(tmp_path: Path) -> None:
    spec = struct_hpo.ARMS["new"]
    with pytest.raises(ValueError, match="non-negative"):
        struct_hpo.materialize_trial_config(spec.base_config, {"rank": -1.0, "degree": 0.1, "motif": 0.1}, 0, tmp_path)


def test_build_spec_binds_the_arm() -> None:
    args = argparse.Namespace(arm="new", base_config=None, sweep_dir=None, n_trials=10, rd_band=0.05)
    spec = struct_hpo.build_spec(args)
    assert isinstance(spec, SweepSpec)
    assert spec.study_name == "struct_new"
    assert spec.n_startup_trials == 3
    assert args.base_config == Path("configs/struct_new_breadth_first.yaml")
    assert args.sweep_dir == Path("outputs/struct_hpo/new")


def test_parser_defaults() -> None:
    args = struct_hpo.build_parser().parse_args(["--arm", "grand"])
    assert args.n_trials == 10
    assert args.rd_band == 0.05
    assert args.base_config is None and args.sweep_dir is None


def test_driver_never_touches_frozen_paths() -> None:
    source = Path("src/experiments/struct_hpo.py").read_text()
    assert "autoresearch/" not in source
    assert "configs/sweep" not in source
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_struct_hpo.py -n0 -q`
Expected: FAIL with `ImportError: cannot import name 'struct_hpo'`.

- [ ] **Step 4: Implement the driver**

```python
# src/experiments/struct_hpo.py
"""Unattended Optuna sweeps for the structural-stream arms (``grand`` and ``new``).

Reuses the strict-LLP kd_rank loop (`src.experiments.kd_rank_strict_hpo`):
ask-and-tell constrained MO-TPE with objectives (GS max, geometric-mean MMD
ratio min) and the ``|log RD|`` soft constraint, one ``hpc/run.sh train
--skip-test`` per trial, scored at the cadence-2 selected epoch. Each arm is
its own 10-trial study; only ``struct.weights`` and ``output_dir`` differ
between trial configs. Winner selection stays the frozen five-metric
undominated verdict plus the human pick.
Spec: ``docs/superpowers/specs/2026-09-07-structural-stream-topology-losses-design.md`` section 8.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import optuna
import yaml

from src.distill.struct_config import StructConfig
from src.experiments.kd_rank_strict_hpo import SweepSpec, run_sweep

N_STARTUP_TRIALS = 3
DEFAULT_TRIALS = 10

SEARCH_BOXES: dict[str, dict[str, tuple[float, float]]] = {
    "grand": {"gs": (0.1, 2.0), "rd": (0.1, 2.0)},
    "new": {"rank": (0.1, 3.0), "degree": (0.01, 1.0), "motif": (0.01, 1.0)},
}


@dataclass(frozen=True)
class ArmSpec:
    """One structural arm: study identity, base config, priors, and search space."""

    study_name: str
    base_config: Path
    priors: tuple[dict[str, object], ...]
    param_names: tuple[str, ...]
    suggest: Callable[[optuna.Trial], dict[str, object]]


def _suggest_for(arm: str) -> Callable[[optuna.Trial], dict[str, object]]:
    boxes = SEARCH_BOXES[arm]

    def suggest(trial: optuna.Trial) -> dict[str, object]:
        return {
            name: float(trial.suggest_float(name, low, high, log=True))
            for name, (low, high) in boxes.items()
        }

    return suggest


ARMS: dict[str, ArmSpec] = {
    "grand": ArmSpec(
        study_name="struct_grand",
        base_config=Path("configs/struct_grand_breadth_first.yaml"),
        priors=({"gs": 0.70, "rd": 0.90}, {"gs": 0.35, "rd": 0.45}),
        param_names=("gs", "rd"),
        suggest=_suggest_for("grand"),
    ),
    "new": ArmSpec(
        study_name="struct_new",
        base_config=Path("configs/struct_new_breadth_first.yaml"),
        priors=(
            {"rank": 1.0, "degree": 0.1, "motif": 0.1},
            {"rank": 1.0, "degree": 0.03, "motif": 0.03},
        ),
        param_names=("rank", "degree", "motif"),
        suggest=_suggest_for("new"),
    ),
}


def materialize_trial_config(
    base_config: Path, params: Mapping[str, object], trial_number: int, sweep_dir: Path
) -> Path:
    """Write trial ``trial_number``'s config: base + searched ``struct.weights`` + ``output_dir``.

    Every searched weight replaces its base value; ``bce`` and unsearched keys
    keep the base file's values.

    Raises:
        ValueError: If the resulting ``struct`` section is illegal.
    """
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    cfg["output_dir"] = str(sweep_dir / f"trial_{trial_number:03d}")
    weights = {**cfg["struct"]["weights"], **{k: float(v) for k, v in params.items()}}  # type: ignore[arg-type]
    cfg["struct"] = {**cfg["struct"], "weights": weights}
    StructConfig.from_mapping(cfg["struct"])
    config_path = sweep_dir / "configs" / f"trial_{trial_number:03d}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return config_path


def _no_prepare(args: argparse.Namespace) -> None:
    """Structural arms need no bank; nothing to prepare."""


def build_spec(args: argparse.Namespace) -> SweepSpec:
    """Bind the chosen arm into a ``SweepSpec`` and fill the arm's default paths."""
    arm = ARMS[str(args.arm)]
    if args.base_config is None:
        args.base_config = arm.base_config
    if args.sweep_dir is None:
        args.sweep_dir = Path("outputs/struct_hpo") / str(args.arm)
    return SweepSpec(
        study_name=arm.study_name,
        n_startup_trials=N_STARTUP_TRIALS,
        priors=arm.priors,
        param_names=arm.param_names,
        suggest=arm.suggest,
        materialize=materialize_trial_config,
        prepare=_no_prepare,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the `python -m src.experiments.struct_hpo` parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--sweep-dir", type=Path, default=None)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--rd-band", type=float, default=0.05)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the unattended container sweep."""
    args = build_parser().parse_args(argv)
    run_sweep(args, build_spec(args))


if __name__ == "__main__":
    main()
```

Check that `run_sweep` only reads `args.base_config`, `args.sweep_dir`, `args.n_trials`, `args.rd_band` (`grep -n "args\." src/experiments/kd_rank_strict_hpo.py`); if it reads another attribute, add it to the parser with the same default the kd_rank driver uses.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_struct_hpo.py tests/test_train_b0_struct.py::test_struct_block_parses_and_serialises -n0 -q`
Expected: PASS. Also load each new YAML once: `.venv/bin/python -c "from pathlib import Path; from src.train_b0 import load_config; [print(load_config(Path(p)).struct.arm) for p in ['configs/struct_bce_breadth_first.yaml','configs/struct_grand_breadth_first.yaml','configs/struct_new_breadth_first.yaml']]"` → prints `bce`, `bce+gs+rd`, `bce+degree+motif+rank`.

- [ ] **Step 6: Lint, type-check, commit**

Run: `.venv/bin/python -m ruff check src/experiments/struct_hpo.py tests/test_struct_hpo.py && .venv/bin/python -m ruff format src tests && .venv/bin/python -m mypy src/experiments/struct_hpo.py tests/test_struct_hpo.py`

```bash
git add configs/struct_bce_breadth_first.yaml configs/struct_grand_breadth_first.yaml configs/struct_new_breadth_first.yaml src/experiments/struct_hpo.py tests/test_struct_hpo.py
git commit -m "feat(struct): wave-1 configs and the per-arm Optuna driver"
```

---

### Task 7: Docs, Codex review, full checks, launch commands

**Files:**
- Modify: `docs/03-experiments.md` (§1.2 stale early-stopping sentence; §1.4 table; §1.5 HPO paragraph; §3 new entry), `CLAUDE.md` and `AGENTS.md` (active method set; Commands), `hpc/README.md` (sweep commands)

- [ ] **Step 1: Fix the stale early-stopping sentence**

In `docs/03-experiments.md` §1.2 "Model selection", replace

> Training early-stops on total val loss (patience 10): the val task BCE plus each active KD term's val counterpart at its training weight.

with

> Training early-stops on the validation task BCE alone (`val_task_loss`, patience 10); KD and structural validation terms are logged as diagnostics and never enter the monitor.

- [ ] **Step 2: Add the structural arms to §1.4 and §1.5**

Append to the §1.4 table, before the `Oracles` row:

```markdown
| struct_bce | none (structural stream, BCE only) | — | none | sampler-matched structural baseline |
| struct_grand | GRAND soft-GS + log-ratio RD on the structural stream | — | `gs`, `rd` log-uniform [0.1, 2.0] | ported density/overlap terms |
| struct_new | neighbour ranking + node-wise degree + open/closed motif counts on the structural stream | — | `rank` [0.1, 3.0], `degree` [0.01, 1.0], `motif` [0.01, 1.0], log-uniform | direct output-adjacency supervision |
```

Append to §1.5 after the `kd_rank_rep` study paragraph:

```markdown
The structural arms are not KD: each optimizer step adds one sampled 40-node training subgraph
(32 locally expanded nodes from a BFS, wedge/triangle, or two-ball bridge seed at a 50/25/25 mix,
plus 8 uniform background nodes) whose legal pairs are forwarded and scored as a logit matrix.
`struct_grand` and `struct_new` each run a 10-trial constrained MO-TPE study
(`src/experiments/struct_hpo.py`, 2 enqueued priors, 3 startup trials, same objectives and RD
band as the KD sweeps); the winner per arm is the five-metric undominated verdict plus human pick and
runs the held-out protocol once. `struct_bce` is a single run and the first comparator for both.
```

- [ ] **Step 3: Add the §3 entry**

Under `## 3. Ablations and sensitivity` add:

```markdown
### 3.1 Structural stream (wave 1, new split)

Design: [spec](superpowers/specs/2026-09-07-structural-stream-topology-losses-design.md). Terms
reduce only over legal pairs (distinct nodes, not both in V_val, feature-bearing). Wave 1 compares
the `struct_grand` and `struct_new` winners against `struct_bce`, then against the new-split
`b1_kd_control` once it exists, reporting the pairwise and five topology numbers together and
checking the selected epoch before crediting a term. Results: pending.
```

- [ ] **Step 4: Update `CLAUDE.md` and `AGENTS.md`**

In the active method set paragraph of both files, after the Students bullet add:

```markdown
- Structural arms (`model.family: v3_1`, top-level `struct:` block, no teacher): `struct_bce`
  (sampler-matched baseline), `struct_grand` (ported soft-GS + RD), `struct_new` (neighbour rank +
  node-wise degree + open/closed motif). One sampled training subgraph per step supervises the
  output adjacency; the inference interface is unchanged.
```

In the Commands block of `CLAUDE.md` add after the B0 line:

```bash
hpc/run.sh train configs/struct_bce_breadth_first.yaml                # structural baseline
.venv/bin/python -m src.experiments.struct_hpo --arm grand           # 10-trial study, outputs/struct_hpo/grand
.venv/bin/python -m src.experiments.struct_hpo --arm new             # 10-trial study, outputs/struct_hpo/new
```

Mirror the same three lines in `AGENTS.md` if it carries the Commands block (`grep -n "hpc/run.sh train configs/b0_v31" AGENTS.md`).

- [ ] **Step 5: Update `hpc/README.md`**

After the two existing sweep command lines add:

```bash
.venv/bin/python -m src.experiments.struct_hpo --arm grand   # struct_grand: 2 priors + 8 guided trials, resumable via outputs/struct_hpo/grand/optuna.db
.venv/bin/python -m src.experiments.struct_hpo --arm new     # struct_new: 2 priors + 8 guided trials, resumable via outputs/struct_hpo/new/optuna.db
```

and one sentence: "Structural sweeps need no bank; when both studies share the box export `OMP_NUM_THREADS=16 MKL_NUM_THREADS=16` first. Each study aborts after three consecutive failed trials and resumes from its `optuna.db`."

- [ ] **Step 6: Full local checks**

Run: `.venv/bin/python -m pytest -m "not slow and not integration" -q --dist loadfile -n 4 && .venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format --check src tests && .venv/bin/python -m mypy src tests`
Expected: clean. Then commit the docs:

```bash
git add docs/03-experiments.md CLAUDE.md AGENTS.md hpc/README.md
git commit -m "docs: record the structural-stream arms, sweeps, and the val_task_loss monitor"
```

- [ ] **Step 7: Codex review of the wave (project rule: one per `src/` wave)**

Base is the commit before Task 1 (`git log --oneline | grep "StructConfig" -A1 | tail -1` gives it; call it `<base>`). Prepare the Codex home once:

```bash
SCRATCH=/private/tmp/claude-501/-Users-richardwang-Documents-topology-conditioned-inductive-edge-prediction/1a4059e3-939c-47d0-a796-1fdbf5fd5be1/scratchpad
mkdir -p "$SCRATCH/codex-home" && cp ~/.codex/auth.json "$SCRATCH/codex-home/"
printf 'model = "gpt-6-astra"\nmodel_reasoning_effort = "high"\n' > "$SCRATCH/codex-home/config.toml"   # model pinned to the user's ~/.codex/config.toml value
CODEX_HOME="$SCRATCH/codex-home" codex review --base <base> > "$SCRATCH/wave-review.txt" 2>&1 &
```

Wait for the process to exit (poll with a Monitor on the file's growth stopping, not by reading it), then read `wave-review.txt` and fix every blocker with its own test and commit. No second round unless new `src/` changes land.

- [ ] **Step 8: Push and hand off the launch**

```bash
git push origin main
```

Then on the H20 checkout (`git pull` there), run in this order, each backgrounded with logging:

```bash
mkdir -p outputs/logs
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
nohup hpc/run.sh train configs/struct_bce_breadth_first.yaml > outputs/logs/struct_bce.log 2>&1 &
nohup .venv/bin/python -m src.experiments.struct_hpo --arm grand > outputs/logs/struct_hpo_grand.log 2>&1 &
nohup .venv/bin/python -m src.experiments.struct_hpo --arm new   > outputs/logs/struct_hpo_new.log 2>&1 &
```

After each study prints its report, pick the winner (five-metric undominated verdict plus human pick), then test it once:

```bash
hpc/run.sh test --checkpoint outputs/struct_hpo/<arm>/trial_<nnn>/published/best.pt \
  --output-dir outputs/struct/<arm>_winner --data-root data --strategy breadth_first \
  --arm struct_<arm> --seed 0
```

(Confirm the exact `test` flags with `.venv/bin/python -m src.eval.test_protocol --help`; the required ones are `--checkpoint --output-dir --data-root --strategy --arm --seed`.)

---

## Self-review against the spec

- §2 regime: Task 5 wires the stream beside the unchanged task stream; the bit-identical test guards the absent block. Covered.
- §3 sampler: kinds, budgets, mask, plan determinism, statistics, coverage in Task 3. Covered. The spec's `plan_struct_epoch(...)` free function is realised as `StructSampler.plan(...)`; the spec names the behaviour, not the call shape.
- §4 stream: bucketed, chunked, checkpointed forward through the unwrapped model; matrix assembly; differentiable zero; `subgraphs_per_epoch > steps` rejected. Task 4. Covered.
- §5 terms: all seven plus `struct_total`; clustering MMD not ported. Task 2. Covered.
- §6 config: all keys and rejections in Task 1. Covered.
- §7 telemetry: per-term losses, grad norms, sampler stats, `struct_pairs`, wall-clock share, `val_struct_*` on topology-due epochs; monitor untouched. Tasks 4–5. Covered.
- §8 arms and HPO: three configs, driver, boxes, priors, startup count, sweep dirs, winner test. Task 6 and Task 7 step 8. Covered.
- §9 tests: every listed test exists in Tasks 1–6; the DDP agreement test is the single-process two-rank striping equivalence in Task 4 (the gloo `spawn` variant from `tests/test_train_b0_kd.py` is not needed because the stream makes no collective calls).
- §10 docs: Task 7. Covered.
- Type consistency: `StructStream.loss` returns `(Tensor, dict[str, float])` in Tasks 4 and 5; `last_terms: dict[str, Tensor]`; `_grad_norm(term, model) -> float`; `materialize_trial_config(base_config, params, trial_number, sweep_dir) -> Path` matches `SweepSpec.materialize`'s `Callable[[Path, Mapping[str, object], int, Path], Path]`.

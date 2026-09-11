"""Drift guard for the checked-in KD loss-weight sweep configs.

Every YAML in ``configs/sweep/b1_kd_hpo/`` must be a copy of its base arm
config differing only in the ``distill:`` weight values, the
removal of ``eval.classification_only`` (uniform topology-aware validation),
``eval.topology_every: 2`` on the pinned late points (runs launched after
the cadence change; earlier points completed at cadence 1 and are compared
via cadence-2 reselection), and its sweep ``output_dir``. The zero-KD
control anchor is the published B0 run, not a sweep point.
"""

from pathlib import Path
from typing import cast

import pytest
import yaml
from src.distill.config import DistillConfig

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SWEEP_DIR = _REPO_ROOT / "configs" / "sweep" / "b1_kd_hpo"
_BASE_CONFIGS = {
    "kd_logit": _REPO_ROOT / "configs" / "b1_kd_logit_breadth_first.yaml",
    "kd_gram": _REPO_ROOT / "configs" / "b1_kd_gram_breadth_first.yaml",
    "kd_rep": _REPO_ROOT / "configs" / "b1_kd_rep_breadth_first.yaml",
}
EXPECTED_SWEEPS: dict[str, tuple[str, dict[str, float]]] = {
    "kd_logit_w0p01": ("kd_logit", {"w_logit": 0.01}),
    "kd_logit_w0p1": ("kd_logit", {"w_logit": 0.1}),
    "kd_logit_w1": ("kd_logit", {"w_logit": 1.0}),
    "kd_logit_w10": ("kd_logit", {"w_logit": 10.0}),
    "kd_logit_w100": ("kd_logit", {"w_logit": 100.0}),
    "kd_gram_w0p01": ("kd_gram", {"w_gram": 0.01}),
    "kd_gram_w0p1": ("kd_gram", {"w_gram": 0.1}),
    "kd_gram_w1": ("kd_gram", {"w_gram": 1.0}),
    "kd_gram_w10": ("kd_gram", {"w_gram": 10.0}),
    "kd_gram_w100": ("kd_gram", {"w_gram": 100.0}),
    "kd_rep_w0p01": ("kd_rep", {"w_rep": 0.01}),
    "kd_rep_w0p1": ("kd_rep", {"w_rep": 0.1}),
    "kd_rep_w1": ("kd_rep", {"w_rep": 1.0}),
    "kd_rep_w10": ("kd_rep", {"w_rep": 10.0}),
    "kd_rep_w100": ("kd_rep", {"w_rep": 100.0}),
}
EXPECTED_STEMS = frozenset(EXPECTED_SWEEPS)
_CADENCE_2_STEMS = frozenset({"kd_rep_w10", "kd_rep_w100"})


def _load(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return cast(dict[str, object], loaded)


def test_all_expected_sweep_stems_exist_exactly() -> None:
    assert {path.stem for path in _SWEEP_DIR.glob("*.yaml")} == set(EXPECTED_STEMS)


@pytest.mark.parametrize("stem", sorted(EXPECTED_STEMS))
def test_sweep_config_differs_from_base_only_in_distill_eval_and_output_dir(stem: str) -> None:
    arm, expected_weights = EXPECTED_SWEEPS[stem]
    sweep = _load(_SWEEP_DIR / f"{stem}.yaml")
    base = _load(_BASE_CONFIGS[arm])

    assert set(sweep) == set(base)
    for key in set(base) - {"distill", "eval", "output_dir"}:
        assert sweep[key] == base[key], f"{stem} drifts from base in top-level key {key!r}"

    base_eval = cast(dict[str, object], base["eval"])
    if arm == "kd_logit":
        assert "classification_only" not in base_eval  # this base eval carries no flag to strip
    else:
        assert "classification_only" in base_eval
    expected_eval = {key: value for key, value in base_eval.items() if key != "classification_only"}
    if stem in _CADENCE_2_STEMS:
        expected_eval["topology_every"] = 2
    assert sweep["eval"] == expected_eval

    base_distill_mapping = cast(dict[str, object], base["distill"])
    expected_distill_mapping = base_distill_mapping | expected_weights
    assert sweep["distill"] == expected_distill_mapping

    distill = DistillConfig.from_mapping(cast(dict[str, object], sweep["distill"]))
    expected_distill = DistillConfig.from_mapping(expected_distill_mapping)
    assert distill.arm == arm
    assert distill == expected_distill
    assert sweep["output_dir"] == f"outputs/b1_row_kd_hpo/{stem}"


# The per-rank micro-batch every bidirectional-cross config uses (half of B0's).
PREFIX_TOKEN_BUDGET = 262144
PREFIX_MAX_PAIRS_PER_RANK = 2048


def test_prefix_base_differs_from_headline_b0_in_mixing_output_dir_and_micro_batch() -> None:
    base_path = _REPO_ROOT / "configs" / "split_seed42" / "b0_v31.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    cross_path = _REPO_ROOT / "configs" / "split_seed42" / "prefix_base.yaml"
    cross = yaml.safe_load(cross_path.read_text(encoding="utf-8"))
    assert cross["model"]["config"]["mixing"] == {"mode": "bidirectional_cross"}
    assert cross["output_dir"] == "outputs/split_seed42/prefix_base"
    # The three bidirectional cross-attention layers hold two extra attention maps each, and B0's
    # micro-batch (which already peaks at 59.7 GiB per rank) then OOMs a 95 GiB H20, so the prefix
    # trunk halves it. Nothing else about the recipe moves.
    assert base["runtime"]["token_budget"] == 524288
    assert base["runtime"]["max_pairs_per_rank"] == 4096
    assert cross["runtime"]["token_budget"] == PREFIX_TOKEN_BUDGET
    assert cross["runtime"]["max_pairs_per_rank"] == PREFIX_MAX_PAIRS_PER_RANK
    cross["model"]["config"]["mixing"] = base["model"]["config"]["mixing"]
    cross["output_dir"] = base["output_dir"]
    cross["runtime"] = base["runtime"]
    assert cross == base


@pytest.mark.parametrize("arm", ["prefix_static", "prefix_pair", "prefix_pair_bce"])
def test_prefix_arm_configs_share_the_struct_new_recipe(arm: str) -> None:
    split_dir = _REPO_ROOT / "configs" / "split_seed42"
    base = yaml.safe_load((split_dir / "struct_new.yaml").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((split_dir / f"{arm}.yaml").read_text(encoding="utf-8"))
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
    if arm == "prefix_pair_bce":
        expected_weights = {
            "bce": 1.0,
            "gs": 0.0,
            "rd": 0.0,
            "deg_mmd": 0.0,
            "rank": 0.0,
            "degree": 0.0,
            "motif": 0.0,
        }
    else:
        expected_weights = {
            "bce": 1.0,
            "gs": 0.0,
            "rd": 0.0,
            "deg_mmd": 0.0,
            "rank": 1.0,
            "degree": 0.1,
            "motif": 0.1,
        }
    assert cfg["struct"]["weights"] == expected_weights
    for key in ("data", "seed", "mixed_precision"):
        assert cfg[key] == base[key]
    # Same halved micro-batch as the frozen trunk they load; the rest of the runtime is
    # struct_new's.
    assert cfg["runtime"]["token_budget"] == PREFIX_TOKEN_BUDGET
    assert cfg["runtime"]["max_pairs_per_rank"] == PREFIX_MAX_PAIRS_PER_RANK
    assert cfg["runtime"] == base["runtime"] | {
        "token_budget": PREFIX_TOKEN_BUDGET,
        "max_pairs_per_rank": PREFIX_MAX_PAIRS_PER_RANK,
    }
    assert {k: v for k, v in cfg["struct"].items() if k != "weights"} == {
        k: v for k, v in base["struct"].items() if k != "weights"
    }

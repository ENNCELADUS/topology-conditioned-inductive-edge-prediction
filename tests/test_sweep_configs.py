"""Drift guard for the checked-in KD loss-weight sweep configs.

Every YAML in ``configs/sweep/b1_kd_hpo/`` must be a copy of its base arm
config differing only in the ``distill:`` weight/temperature values, the
removal of ``eval.classification_only`` (uniform topology-aware validation),
``eval.topology_every: 2`` on the pinned late points (runs launched after
the cadence change; earlier points completed at cadence 1 and are compared
via cadence-2 reselection), and its sweep ``output_dir``. ``kd_control``
has no ``distill:`` section.
"""

from dataclasses import replace
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
    "kd_rank": _REPO_ROOT / "configs" / "b1_kd_rank_breadth_first.yaml",
    "kd_gram": _REPO_ROOT / "configs" / "b1_kd_gram_breadth_first.yaml",
    "kd_rep": _REPO_ROOT / "configs" / "b1_kd_rep_breadth_first.yaml",
    "kd_control": _REPO_ROOT / "configs" / "b1_kd_control_breadth_first.yaml",
}
EXPECTED_SWEEPS: dict[str, tuple[str, dict[str, float]]] = {
    "kd_logit_w0p01": ("kd_logit", {"w_logit": 0.01}),
    "kd_logit_w0p1": ("kd_logit", {"w_logit": 0.1}),
    "kd_logit_w1": ("kd_logit", {"w_logit": 1.0}),
    "kd_logit_w10": ("kd_logit", {"w_logit": 10.0}),
    "kd_logit_w100": ("kd_logit", {"w_logit": 100.0}),
    "kd_rank_w0p01": ("kd_rank", {"w_rank": 0.01, "w_dist": 0.01}),
    "kd_rank_w0p1": ("kd_rank", {"w_rank": 0.1, "w_dist": 0.1}),
    "kd_rank_w1": ("kd_rank", {"w_rank": 1.0, "w_dist": 1.0}),
    "kd_rank_w10": ("kd_rank", {"w_rank": 10.0, "w_dist": 10.0}),
    "kd_rank_w100": ("kd_rank", {"w_rank": 100.0, "w_dist": 100.0}),
    "kd_rank_wr0p1_wd10": ("kd_rank", {"w_rank": 0.1, "w_dist": 10.0}),
    "kd_rank_wr0p1_wd1": ("kd_rank", {"w_rank": 0.1, "w_dist": 1.0}),
    "kd_rank_wr1_wd0p1": ("kd_rank", {"w_rank": 1.0, "w_dist": 0.1}),
    "kd_rank_wr0p01_wd10": ("kd_rank", {"w_rank": 0.01, "w_dist": 10.0}),
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
    "kd_control": ("kd_control", {}),
}
EXPECTED_STEMS = frozenset(EXPECTED_SWEEPS)
_CADENCE_2_STEMS = frozenset({"kd_rep_w10", "kd_rep_w100", "kd_control"})


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
    if arm in {"kd_logit", "kd_control"}:
        assert "classification_only" not in base_eval  # these base evals carry no flag to strip
    else:
        assert "classification_only" in base_eval
    expected_eval = {key: value for key, value in base_eval.items() if key != "classification_only"}
    if stem in _CADENCE_2_STEMS:
        expected_eval["topology_every"] = 2
    assert sweep["eval"] == expected_eval

    if arm == "kd_control":
        assert "distill" not in base
        assert "distill" not in sweep
    else:
        base_distill_mapping = cast(dict[str, object], base["distill"])
        expected_distill_mapping = base_distill_mapping | expected_weights
        assert sweep["distill"] == expected_distill_mapping

        base_distill = DistillConfig.from_mapping(base_distill_mapping)
        distill = DistillConfig.from_mapping(cast(dict[str, object], sweep["distill"]))
        assert distill.arm == arm
        assert distill == replace(base_distill, **expected_weights)
    assert sweep["output_dir"] == f"outputs/b1_row_kd_hpo/{stem}"

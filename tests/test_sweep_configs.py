"""Drift guard for the checked-in KD loss-weight sweep configs.

Every YAML in ``configs/sweep/b1_kd_hpo/`` must be a copy of its base arm
config differing only in the ``distill:`` weight/temperature values, the
removal of ``eval.classification_only`` (uniform topology-aware validation),
and its sweep ``output_dir``.
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
    "kd_rank": _REPO_ROOT / "configs" / "b1_kd_rank_breadth_first.yaml",
    "kd_gram": _REPO_ROOT / "configs" / "b1_kd_gram_breadth_first.yaml",
    "kd_rep": _REPO_ROOT / "configs" / "b1_kd_rep_breadth_first.yaml",
}
_WEIGHT_TAGS = ("w0p01", "w0p1", "w1", "w10", "w100")
EXPECTED_STEMS = frozenset(
    [f"{arm}_{tag}" for arm in _BASE_CONFIGS for tag in _WEIGHT_TAGS]
    + ["kd_rank_wr0p1_wd10", "kd_rank_wr0p1_wd1", "kd_rank_wr1_wd0p1", "kd_rank_wr0p01_wd10"]
)


def _arm_prefix(stem: str) -> str:
    matches = [arm for arm in _BASE_CONFIGS if stem.startswith(f"{arm}_")]
    assert len(matches) == 1, f"{stem} does not map to exactly one base arm"
    return matches[0]


def _load(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return cast(dict[str, object], loaded)


def test_all_expected_sweep_stems_exist_exactly() -> None:
    assert {path.stem for path in _SWEEP_DIR.glob("*.yaml")} == set(EXPECTED_STEMS)


@pytest.mark.parametrize("stem", sorted(EXPECTED_STEMS))
def test_sweep_config_differs_from_base_only_in_distill_eval_and_output_dir(stem: str) -> None:
    arm = _arm_prefix(stem)
    sweep = _load(_SWEEP_DIR / f"{stem}.yaml")
    base = _load(_BASE_CONFIGS[arm])

    assert set(sweep) == set(base)
    for key in set(base) - {"distill", "eval", "output_dir"}:
        assert sweep[key] == base[key], f"{stem} drifts from base in top-level key {key!r}"

    base_eval = cast(dict[str, object], base["eval"])
    expected_eval = {key: value for key, value in base_eval.items() if key != "classification_only"}
    if arm == "kd_logit":
        assert expected_eval == base_eval  # kd_logit's base eval stays byte-identical
    else:
        assert "classification_only" in base_eval
    assert sweep["eval"] == expected_eval

    distill = DistillConfig.from_mapping(cast(dict[str, object], sweep["distill"]))
    assert distill.arm == arm
    assert sweep["output_dir"] == f"outputs/b1_row_kd_hpo/{stem}"

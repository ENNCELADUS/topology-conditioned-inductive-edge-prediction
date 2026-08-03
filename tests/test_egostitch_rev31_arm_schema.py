"""Task-11 eight-arm schema contracts (EgoStitch spec Sec 14.4.6)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from src import score_universe

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
V3_CONFIGS = {
    "full": REPO_ROOT / "configs/egostitch_e2e_v3_full_breadth_first.yaml",
    "b0_e2e_f_only": REPO_ROOT / "configs/egostitch_e2e_v3_f_only_breadth_first.yaml",
    # "pair_topology" was retired (not merely relabeled): the three-component
    # refactor's content-path removal (P1) deleted both its v3 and non-v3
    # configs, since with no content branch a topology-only arm is identical
    # to "full" (design 2026-08-02 §9).
    "p0": REPO_ROOT / "configs/egostitch_e2e_v3_p0_breadth_first.yaml",
    "no_l_rel": REPO_ROOT / "configs/egostitch_e2e_v3_no_l_rel_breadth_first.yaml",
    "row_layernorm": REPO_ROOT
    / "configs/egostitch_e2e_v3_row_layernorm_breadth_first.yaml",
}
# Historical only: retired from the trained set in v5 but retained on disk.
RETIRED_COSINE_POOL_CONFIG = (
    REPO_ROOT / "configs/egostitch_e2e_v3_cosine_pool_breadth_first.yaml"
)


def test_v3_live_arms_share_one_ng50_pack_and_grounding_cache() -> None:
    """v5: all six trained arms are n_ground=50 and share one pack + grounding cache."""
    config_paths = sorted(
        (REPO_ROOT / "configs").glob("egostitch_e2e_v3_*_breadth_first.yaml")
    )
    # The retired cosine_pool config stays on disk as history; no live arm uses it.
    assert set(config_paths) == set(V3_CONFIGS.values()) | {RETIRED_COSINE_POOL_CONFIG}
    configs = []
    for path in sorted(V3_CONFIGS.values()):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        configs.append(
            (
                path.name,
                payload["model"]["config"]["n_ground"],
                payload["runtime"]["pack_dir"],
                payload["data"]["grounding_cache"],
                payload["data"]["f0_cache"],
                payload["data"]["pack_dir"],
            )
        )

    assert {config[1] for config in configs} == {50}
    assert {config[2] for config in configs} == {
        "outputs/feature_packs/egostitch_e2e_v_hold_ng50"
    }
    assert len({config[3] for config in configs}) == 1
    assert len({config[4] for config in configs}) == 1
    assert len({config[5] for config in configs}) == 1


def test_scores_meta_version_is_written_and_older_versions_are_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scores.npz"
    values = np.array([0.25], dtype=np.float32)
    score_universe.save_scores(
        path,
        node_ids=["a", "b"],
        u_idx=np.array([0], dtype=np.int32),
        v_idx=np.array([1], dtype=np.int32),
        logit=values,
        label=np.array([1], dtype=np.int8),
        row_start=0,
        meta={
            "checkpoint_id": "checkpoint",
            "model_family": "egostitch_e2e",
            # Not a held-out claim: this fixture exercises scores_meta_version
            # round-tripping only. A "candidate" source would make it held-out
            # shaped and correctly fail closed for want of a test-access ledger.
            "pairs_source": "val",
            "strategy": "toy",
            "num_rows": 1,
            "created_utc": "2026-07-26T00:00:00Z",
            "torch_version": "test",
            "permanent_null": "none",
            "primary_logit": "full",
            "score_precision": {
                "contract": "egostitch_e2e_pair_fp32_v1",
                "pair_compute_dtype": "float32",
                "pair_autocast": False,
                "logit_storage_dtype": "float32",
            },
        },
        f_logit=values,
    )
    artifact = score_universe.load_scores(path)
    assert artifact.meta["scores_meta_version"] == score_universe._SCORES_META_VERSION

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    old_meta = json.loads(str(arrays["meta"][()]))
    old_meta["scores_meta_version"] = "egostitch_e2e_scores_v2"
    arrays["meta"] = np.array(json.dumps(old_meta, sort_keys=True))
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="scores_meta_version"):
        score_universe.load_scores(path)


def test_scoring_cli_exposes_rev31_controls_and_rejects_superseded_v2() -> None:
    parser = score_universe.build_parser()
    checkpoint = REPO_ROOT / "missing-but-parseable.pt"
    for control in (
        score_universe._SCAFFOLD_CONTROL_SHUFFLE_V3,
        score_universe._SCAFFOLD_CONTROL_REWIRE_V1,
    ):
        args = parser.parse_args(
            [
                "score",
                "--checkpoint",
                str(checkpoint),
                "--pairs",
                "candidate",
                "--output",
                "scores.npz",
                "--scaffold-control",
                control,
            ]
        )
        assert args.scaffold_control == control
    with pytest.raises(ValueError, match="14\\.4\\.5"):
        score_universe._reject_superseded_scaffold_control(
            score_universe._SCAFFOLD_CONTROL_SHUFFLE_V2
        )

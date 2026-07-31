"""Task-11 eight-arm schema contracts (EgoStitch spec Sec 14.4.6)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from src import score_universe, train_egostitch
from src.experiments import g5_stage1
from src.model.egostitch.config import E2EConfig

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINED_ARMS = (
    "full",
    "b0_e2e_f_only",
    "pair_topology",
    "p0",
    "no_l_rel",
    "row_layernorm",
)
CONTROL_ARMS = ("structure_control_6a_v3", "structure_control_6e_v1")
V3_CONFIGS = {
    "full": REPO_ROOT / "configs/egostitch_e2e_v3_full_breadth_first.yaml",
    "b0_e2e_f_only": REPO_ROOT / "configs/egostitch_e2e_v3_f_only_breadth_first.yaml",
    "pair_topology": REPO_ROOT
    / "configs/egostitch_e2e_v3_pair_topology_breadth_first.yaml",
    "p0": REPO_ROOT / "configs/egostitch_e2e_v3_p0_breadth_first.yaml",
    "no_l_rel": REPO_ROOT / "configs/egostitch_e2e_v3_no_l_rel_breadth_first.yaml",
    "row_layernorm": REPO_ROOT
    / "configs/egostitch_e2e_v3_row_layernorm_breadth_first.yaml",
}
# Historical only: retired from the trained set in v5 but retained on disk.
RETIRED_COSINE_POOL_CONFIG = (
    REPO_ROOT / "configs/egostitch_e2e_v3_cosine_pool_breadth_first.yaml"
)


def test_all_enforcement_constants_use_exact_eight_arm_schema() -> None:
    assert tuple(train_egostitch._E2E_FORMAL_ARMS) == TRAINED_ARMS
    assert score_universe._EGOSTITCH_E2E_FORMAL_ARMS == TRAINED_ARMS
    assert g5_stage1._E2E_FORMAL_ARMS == TRAINED_ARMS
    assert g5_stage1._E2E_ARMS == TRAINED_ARMS + CONTROL_ARMS


def test_v3_configs_load_and_have_only_registered_arm_differences() -> None:
    loaded = {name: train_egostitch.load_config(path) for name, path in V3_CONFIGS.items()}
    model_configs = {
        name: asdict(E2EConfig.from_mapping(config.model.config))
        for name, config in loaded.items()
    }
    full = model_configs["full"]
    expected_differences = {
        "full": {},
        "b0_e2e_f_only": {"permanent_null": "all_head"},
        "pair_topology": {"permanent_null": "content_head"},
        "p0": {"p_topo": 0.0, "p_cont": 0.0},
        "no_l_rel": {"w_rel": 0.0},
        "row_layernorm": {"feature_standardization": "row_layernorm"},
    }
    for arm, model_config in model_configs.items():
        differences = {
            key: value for key, value in model_config.items() if value != full[key]
        }
        assert differences == expected_differences[arm]

    assert {arm: config.model.config["n_ground"] for arm, config in loaded.items()} == {
        "full": 50,
        "b0_e2e_f_only": 50,
        "pair_topology": 50,
        "p0": 50,
        "no_l_rel": 50,
        "row_layernorm": 50,
    }
    assert {arm: model["w_rel"] for arm, model in model_configs.items()} == {
        "full": 0.25,
        "b0_e2e_f_only": 0.25,
        "pair_topology": 0.25,
        "p0": 0.25,
        "no_l_rel": 0.0,
        "row_layernorm": 0.25,
    }
    for arm, config in loaded.items():
        assert config.preregistration == Path(
            "docs/registrations/g5_e2e_stage1_preregistration_v5.json"
        )
        output_arm = "f_only" if arm == "b0_e2e_f_only" else arm
        assert config.output_dir == Path(f"outputs/egostitch_e2e_stage1_v3/{output_arm}")


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


def test_arm_classifier_distinguishes_all_six_trained_checkpoints() -> None:
    for arm, path in V3_CONFIGS.items():
        config = train_egostitch.load_config(path)
        resolved = E2EConfig.from_mapping(config.model.config)
        assert train_egostitch._e2e_arm_name_from_config(resolved) == arm


def test_run_metadata_represents_trained_checkpoint_and_scoring_semantics(
    tmp_path: Path,
) -> None:
    config_path = V3_CONFIGS["row_layernorm"]
    config = train_egostitch.load_config(config_path)
    config = train_egostitch.replace(config, output_dir=tmp_path / "run")
    data = SimpleNamespace(rho_train=0.1)

    train_egostitch.write_run_start_metadata(
        config,
        data,
        world_size=1,
        config_path=config_path,
    )

    metadata = json.loads((config.output_dir / "run_metadata.json").read_text())
    assert metadata["arm"] == "row_layernorm"
    assert metadata["arm_kind"] == "trained_checkpoint"
    assert metadata["checkpoint_arm"] == "row_layernorm"
    assert metadata["scoring_semantics"] == {
        "scaffold_control": "none",
        "permanent_null": "none",
        "primary_logit": "full",
    }


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
            "pairs_source": "candidate",
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
        pair_content=values,
        pair_topology=values,
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

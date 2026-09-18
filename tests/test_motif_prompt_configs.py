"""Every motif-prompt config loads, and the arms differ only where the spec says."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from src.model.egostitch.classifier.motif_prompt import MotifPromptConfig
from src.train_b0 import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "split_seed42"
ARMS = sorted(path.name for path in CONFIG_DIR.glob("motif_prompt_*.yaml"))


def _raw(name: str) -> dict[str, Any]:
    """Return one config's parsed YAML, unvalidated."""
    loaded: dict[str, Any] = yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))
    return loaded


def _block(name: str) -> dict[str, Any]:
    """Return one config's ``model.config.motif_prompt`` block."""
    block: dict[str, Any] = _raw(name)["model"]["config"]["motif_prompt"]
    return block


@pytest.mark.parametrize("name", ARMS)
def test_every_motif_config_loads_and_names_its_own_output_dir(name: str) -> None:
    cfg = load_config(CONFIG_DIR / name)
    assert cfg.model.family == "v3_1_motif_prompt"
    assert cfg.output_dir == Path(f"outputs/split_seed42/{Path(name).stem}")
    assert cfg.seed in (0, 1, 2)
    assert cfg.mixed_precision == "bf16"


@pytest.mark.parametrize("name", ARMS)
def test_every_motif_block_parses_through_the_model_schema(name: str) -> None:
    # `load_config` stores `model.config` verbatim, so only the model's own
    # schema rejects an unknown key or an out-of-range enum value.
    block = load_config(CONFIG_DIR / name).model.config["motif_prompt"]
    assert isinstance(block, dict)
    parsed = MotifPromptConfig.from_mapping(block)
    assert parsed.base_checkpoint == "outputs/split_seed42/prefix_base/best.pt"
    assert parsed.reader.layers == 3 and parsed.reader.dim == 96 and parsed.reader.heads == 4
    assert parsed.reader.rrwp_k == 4 and parsed.width == 128 and parsed.slots_per_field == 2
    assert parsed.beta_p == parsed.beta_q == parsed.beta_a == parsed.beta_i == 1.0
    assert parsed.huber_delta == 1.0


@pytest.mark.parametrize("name", ARMS)
def test_every_motif_config_runs_the_retained_structural_stream(name: str) -> None:
    cfg = load_config(CONFIG_DIR / name)
    assert cfg.struct is not None
    assert cfg.struct.nodes == 40 and cfg.struct.background_nodes == 8
    assert cfg.struct.mix == {"bfs": 0.5, "motif": 0.25, "bridge": 0.25}
    assert cfg.struct.subgraphs_per_epoch is None
    assert cfg.struct.weights["bce"] == 1.0
    assert cfg.struct.weights["rank"] == 1.0
    assert cfg.struct.weights["degree"] == 0.1
    assert cfg.struct.weights["motif"] == 0.1


@pytest.mark.parametrize("name", ARMS)
def test_every_motif_config_shares_one_chunking_and_batching_protocol(name: str) -> None:
    # struct.token_budget and struct.resident_tokens are the pure memory levers;
    # subgraphs_per_epoch and runtime.max_pairs_per_rank change the objective or
    # the batching protocol, so a single value of each covers the whole family
    # (spec section 7.1).
    cfg = load_config(CONFIG_DIR / name)
    assert cfg.struct is not None and cfg.struct.token_budget == 98304
    assert cfg.struct.resident_tokens == 393216
    assert cfg.runtime is not None and cfg.runtime.max_pairs_per_rank == 1536


@pytest.mark.parametrize("name", ARMS)
def test_every_motif_config_runs_the_full_fifteen_epoch_cycle(name: str) -> None:
    cfg = load_config(CONFIG_DIR / name)
    assert cfg.optim.epochs == 15
    assert cfg.eval.patience is None
    assert cfg.optim.scheduler is not None and cfg.optim.scheduler.type == "onecycle"
    assert cfg.optim.lr == 1e-4 == cfg.optim.scheduler.max_lr
    assert cfg.optim.weight_decay == 1e-2
    assert cfg.optim.grad_clip == 1.0
    # Stage I has no generator, so the interface is its only optimizer group and
    # trains the whole reader at spec section 7.2's 1e-4; the 0.1x interface rate
    # belongs to section 7.5's Stage II schedule.
    interface = 1e-4 if _block(name)["stage"] == "one" else 1e-5
    assert cfg.optim.groups == {"generator": 1e-4, "interface": interface}


def test_stage_one_is_a_ceiling_diagnostic_and_stage_two_is_deployable() -> None:
    one, two = _block("motif_prompt_stage1.yaml"), _block("motif_prompt_stage2.yaml")
    assert one["stage"] == "one" and two["stage"] == "two"
    # Stage I has no teacher, so L_topo has no target and its weight is zero;
    # the interface is what Stage I trains, so it never warms up (spec 7.2/7.5).
    assert one["w_topo"] == 0.0 and one["interface_warmup_epochs"] == 0
    assert "bundle_checkpoint" not in one
    assert two["w_topo"] == 0.1 and two["interface_warmup_epochs"] == 2
    assert two["bundle_checkpoint"].startswith("outputs/split_seed42/motif_prompt_stage1/")
    header = (CONFIG_DIR / "motif_prompt_stage1.yaml").read_text(encoding="utf-8")
    assert "--run-kind diagnostic" in header and "--allow-oracle-diagnostic" in header


def test_stage_one_and_stage_two_differ_only_in_the_expected_keys() -> None:
    one, two = _raw("motif_prompt_stage1.yaml"), _raw("motif_prompt_stage2.yaml")
    for block in ("data", "eval", "runtime", "struct"):
        assert one[block] == two[block]
    diff = {
        key
        for key in set(one["model"]["config"]["motif_prompt"])
        | set(two["model"]["config"]["motif_prompt"])
        if one["model"]["config"]["motif_prompt"].get(key)
        != two["model"]["config"]["motif_prompt"].get(key)
    }
    # The plan's draft listed only {stage, bundle_checkpoint}; the two schedule
    # keys above are part of the same stage difference and must differ too.
    assert diff == {"stage", "bundle_checkpoint", "interface_warmup_epochs", "w_topo"}


_CONTROLS: dict[str, dict[str, Any]] = {
    "motif_prompt_count_only": {"fields": ["topo_cnt"]},
    "motif_prompt_grit_only": {"fields": ["topo_self", "topo_partner", "topo_rel"]},
    # Spec v9: the trained reading of the section 8 degree-only control -- the
    # symmetric degree pair from the *predicted* adjacency, the two mass entries
    # zeroed in topo_cnt, the GRIT fields removed by per-field row masking.
    "motif_prompt_degree_only": {"fields": ["topo_cnt"], "count_features": "degree"},
    "motif_prompt_direct_prefix": {"token_source": "direct"},
    "motif_prompt_per_type": {"gate_mode": "per_type"},
    "motif_prompt_mean_graph": {"gate_mode": "mean_graph"},
    "motif_prompt_freeze_forever": {"interface_warmup_epochs": None},
    # The inactive family is zeroed in BOTH stages (spec section 8), so each
    # family ablation loads its own family-matched Stage I bundle.
    "motif_prompt_closure_only": {
        "families": ["closure"],
        "bundle_checkpoint": "outputs/split_seed42/motif_prompt_stage1_closure_only/best.pt",
    },
    "motif_prompt_bridge_only": {
        "families": ["bridge"],
        "bundle_checkpoint": "outputs/split_seed42/motif_prompt_stage1_bridge_only/best.pt",
    },
    "motif_prompt_no_slot": {"w_slot": 0.0},
    "motif_prompt_no_topo": {"w_topo": 0.0},
}


@pytest.mark.parametrize(("name", "overrides"), sorted(_CONTROLS.items()))
def test_each_control_edits_only_its_own_keys_of_stage_two(
    name: str, overrides: dict[str, Any]
) -> None:
    base, control = _raw("motif_prompt_stage2.yaml"), _raw(f"{name}.yaml")
    assert control["output_dir"] == f"outputs/split_seed42/{name}"
    assert control["model"]["config"]["motif_prompt"] == {
        **base["model"]["config"]["motif_prompt"],
        **overrides,
    }
    for block in ("data", "optim", "eval", "runtime", "struct", "seed", "mixed_precision"):
        assert control[block] == base[block]


@pytest.mark.parametrize("family", ["closure", "bridge"])
def test_each_family_ablation_has_a_family_matched_stage_one(family: str) -> None:
    base, lane = _raw("motif_prompt_stage1.yaml"), _raw(f"motif_prompt_stage1_{family}_only.yaml")
    assert lane["output_dir"] == f"outputs/split_seed42/motif_prompt_stage1_{family}_only"
    assert lane["model"]["config"]["motif_prompt"] == {
        **base["model"]["config"]["motif_prompt"],
        "families": [family],
    }
    for block in ("data", "optim", "eval", "runtime", "struct", "seed", "mixed_precision"):
        assert lane[block] == base[block]
    two = _raw(f"motif_prompt_{family}_only.yaml")["model"]["config"]["motif_prompt"]
    assert two["bundle_checkpoint"] == f"{lane['output_dir']}/best.pt"
    header = (CONFIG_DIR / f"motif_prompt_stage1_{family}_only.yaml").read_text(encoding="utf-8")
    assert "--run-kind diagnostic" in header


def test_every_control_names_the_question_it_answers() -> None:
    for name in _CONTROLS:
        header = (CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8")
        assert "spec section 8" in header
        assert "Read against motif_prompt_stage2 on the pre-registered panel" in header
        assert "+/-0.01 GS and +/-0.5 MMD ratio are not read" in header


def test_the_count_only_and_grit_only_arms_are_separately_trained_not_masked() -> None:
    # Zeroing a field's token values leaves its keys in the prefix softmax
    # denominator, and deleting its rows renormalises the others, so neither
    # reproduces a model trained without the field (spec section 6).
    count_only = load_config(CONFIG_DIR / "motif_prompt_count_only.yaml")
    grit_only = load_config(CONFIG_DIR / "motif_prompt_grit_only.yaml")
    assert count_only.output_dir != grit_only.output_dir
    assert count_only.optim.epochs == grit_only.optim.epochs == 15


def test_the_degree_only_control_is_the_trained_arm_not_a_rethresholding() -> None:
    parsed = MotifPromptConfig.from_mapping(_block("motif_prompt_degree_only.yaml"))
    assert parsed.stage == "two"
    assert parsed.fields == ("topo_cnt",)
    assert parsed.count_features == "degree"
    # It differs from count_only by the count-head restriction alone.
    count_only = MotifPromptConfig.from_mapping(_block("motif_prompt_count_only.yaml"))
    assert count_only.count_features == "all"
    assert replace(count_only, count_features="degree") == parsed
    header = (CONFIG_DIR / "motif_prompt_degree_only.yaml").read_text(encoding="utf-8")
    assert "NOT a" in header and "degree-marginal re-thresholding" in header


@pytest.mark.parametrize("suffix", ["a", "b", "c"])
def test_each_teachability_pilot_is_a_true_two_epoch_prefix(suffix: str) -> None:
    cfg = load_config(CONFIG_DIR / f"motif_prompt_teach_{suffix}.yaml")
    # "The first two epochs" is literal: epochs stays 15 so the one-cycle shape,
    # the trainability mask and the data order are the full run's (spec 7.4).
    assert cfg.optim.epochs == 15
    assert cfg.optim.stop_after_epoch == 2
    base, pilot = _block("motif_prompt_stage2.yaml"), _block(f"motif_prompt_teach_{suffix}.yaml")
    differing = {key for key in base if base[key] != pilot[key]}
    assert differing == {"bundle_checkpoint"}
    assert set(pilot) == set(base)


def test_the_teachability_pilots_name_three_distinct_stage_one_candidates() -> None:
    bundles = {_block(f"motif_prompt_teach_{suffix}.yaml")["bundle_checkpoint"] for suffix in "abc"}
    assert len(bundles) == 3
    for bundle in bundles:
        assert bundle.startswith("outputs/split_seed42/motif_prompt_stage1/checkpoints/epoch-")
    header = (CONFIG_DIR / "motif_prompt_teach_a.yaml").read_text(encoding="utf-8")
    # Selection is downstream utility under predicted inputs, on V_val only, and
    # the winner is continued rather than restarted (spec section 7.4).
    assert "DOWNSTREAM UTILITY UNDER PREDICTED INPUTS" in header
    assert "geometric_rd_five_rank_v1" in header
    assert "CONTINUED" in header and "PLACEHOLDER" in header


@pytest.mark.parametrize("name", ARMS)
def test_only_the_pilots_and_the_wave_two_prefixes_stop_early(name: str) -> None:
    cfg = load_config(CONFIG_DIR / name)
    halted = name.startswith("motif_prompt_teach_") or (
        name.startswith("motif_prompt_stage2_v") and name.endswith("_prefix.yaml")
    )
    # The wave-3 A1 prefix stretches the graph-only warm-up to eight epochs and
    # halts there; every other prefix is a two-epoch warm-up.
    expected = 8 if name == "motif_prompt_stage2_v3_initonly_warm8_prefix.yaml" else 2
    assert cfg.optim.stop_after_epoch == (expected if halted else None)
    if halted:
        warmup = cfg.model.config["motif_prompt"]["interface_warmup_epochs"]
        assert cfg.optim.stop_after_epoch <= warmup


#: The wave-2 phase-1 arms and what each one may change against the shared prefix.
_WAVE_TWO_ARMS: dict[str, dict[str, Any]] = {
    "motif_prompt_stage2_v2": {},
    "motif_prompt_stage2_v2_no_topo": {"w_topo": 0.0},
    "motif_prompt_stage2_v2_strong_graph": {"w_slot_multiplier": 10.0},
}

#: The wave-2 phase-0 attribution prefixes and what each one drops.
_WAVE_TWO_PREFIXES: dict[str, dict[str, Any]] = {
    "motif_prompt_stage2_v2_initonly_prefix": {"beta_c": 0.0},
    "motif_prompt_stage2_v2_lossonly_prefix": {"closure_bias_init": "density"},
}


def test_the_wave_two_prefix_carries_all_three_fixes_over_stage_two() -> None:
    base, prefix = _block("motif_prompt_stage2.yaml"), _block("motif_prompt_stage2_v2_prefix.yaml")
    assert prefix["bundle_checkpoint"] == "outputs/split_seed42/motif_prompt_stage1/best.pt"
    assert {key for key in base if base[key] != prefix.get(key)} == {"w_slot"}
    assert set(prefix) - set(base) == {
        "warmup_losses",
        "w_slot_multiplier",
        "balance_probe_rows",
        "beta_c",
        "closure_bias_init",
    }
    assert prefix["warmup_losses"] == "graph_only"
    assert prefix["w_slot"] == "balanced" and prefix["w_slot_multiplier"] == 1.0
    assert prefix["beta_c"] == 1.0 and prefix["closure_bias_init"] == "nonzero_mean"
    assert prefix["w_topo"] == 0.1
    header = (CONFIG_DIR / "motif_prompt_stage2_v2_prefix.yaml").read_text(encoding="utf-8")
    # The teachability note of spec 7.4 is retired: the reader swap was worth
    # 0.0005 AUPRC, so the bundle stays the five-metric Stage I winner.
    assert "must be replaced by" not in header
    assert "hpc/run.sh train configs/split_seed42/motif_prompt_stage2_v2_prefix.yaml" in header


@pytest.mark.parametrize(("name", "overrides"), sorted(_WAVE_TWO_ARMS.items()))
def test_each_wave_two_arm_is_the_prefix_plus_its_own_joint_phase_key(
    name: str, overrides: dict[str, Any]
) -> None:
    prefix, arm = _raw("motif_prompt_stage2_v2_prefix.yaml"), _raw(f"{name}.yaml")
    assert arm["output_dir"] == f"outputs/split_seed42/{name}"
    assert arm["model"]["config"]["motif_prompt"] == {
        **prefix["model"]["config"]["motif_prompt"],
        **overrides,
    }
    # The only keys an arm may move are the three the resume comparison excludes
    # while the resumed epoch is inside the graph-only warm-up.
    assert set(overrides) <= {"w_slot", "w_slot_multiplier", "w_topo"}
    for section in ("data", "eval", "runtime", "struct", "seed", "mixed_precision"):
        assert arm[section] == prefix[section]
    # The prefix is a true two-epoch prefix: nothing but the halt point differs.
    assert arm["optim"] == {
        key: value for key, value in prefix["optim"].items() if key != "stop_after_epoch"
    }
    assert "stop_after_epoch" not in arm["optim"]
    assert prefix["optim"]["stop_after_epoch"] == 2
    header = (CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8")
    assert "--resume-attempt" in header
    assert "outputs/split_seed42/motif_prompt_stage2_v2_prefix/attempts/" in header


@pytest.mark.parametrize(("name", "overrides"), sorted(_WAVE_TWO_PREFIXES.items()))
def test_each_attribution_prefix_drops_exactly_one_wave_two_fix(
    name: str, overrides: dict[str, Any]
) -> None:
    prefix, other = _raw("motif_prompt_stage2_v2_prefix.yaml"), _raw(f"{name}.yaml")
    assert other["output_dir"] == f"outputs/split_seed42/{name}"
    assert other["model"]["config"]["motif_prompt"] == {
        **prefix["model"]["config"]["motif_prompt"],
        **overrides,
    }
    assert other["model"]["config"]["motif_prompt"]["warmup_losses"] == "graph_only"
    for section in ("data", "optim", "eval", "runtime", "struct", "seed", "mixed_precision"):
        assert other[section] == prefix[section]


def test_the_three_seed_replicates_differ_only_in_seed_and_output_dir() -> None:
    base = _raw("motif_prompt_stage2.yaml")
    for index in (1, 2):
        other = _raw(f"motif_prompt_stage2_seed{index}.yaml")
        assert other.pop("seed") == index
        assert other.pop("output_dir") == f"outputs/split_seed42/motif_prompt_stage2_seed{index}"
        stripped = dict(base)
        stripped.pop("seed")
        stripped.pop("output_dir")
        assert other == stripped

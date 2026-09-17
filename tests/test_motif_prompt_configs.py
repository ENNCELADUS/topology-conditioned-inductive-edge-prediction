"""Every motif-prompt config loads, and the arms differ only where the spec says."""

from __future__ import annotations

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
    # struct.token_budget is the only pure memory lever; subgraphs_per_epoch and
    # runtime.max_pairs_per_rank change the objective or the batching protocol,
    # so a single value of each covers the whole family (spec section 7.1).
    cfg = load_config(CONFIG_DIR / name)
    assert cfg.struct is not None and cfg.struct.token_budget == 8192
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
    assert cfg.optim.groups == {"generator": 1e-4, "interface": 1e-5}


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

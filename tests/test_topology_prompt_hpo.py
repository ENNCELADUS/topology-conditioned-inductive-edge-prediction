import json
from pathlib import Path

import optuna
import pytest
import yaml
from src.experiments.topology_prompt_hpo import (
    PRIOR_VALUES,
    SPACES,
    materialize,
    outcome,
    select_winner,
    suggest,
)
from src.train_b0 import load_config


@pytest.mark.parametrize("name", SPACES)
def test_all_enqueued_priors_are_in_declared_space(name: str) -> None:
    study = optuna.create_study(directions=["maximize", "minimize"])
    for values in PRIOR_VALUES[name]:
        expected = dict(zip(SPACES[name], values, strict=True))
        study.enqueue_trial(expected)
        trial = study.ask()
        assert suggest(name, trial) == expected
        study.tell(trial, [0.4, 2.0])


def test_materialized_config_keeps_bce_and_group_learning_rates(tmp_path: Path) -> None:
    path = materialize(
        Path("configs/split_seed42/coord_gen_v2_d.yaml"),
        {"optim.groups.generator.max_lr": 1e-3, "struct.scale": 3.0},
        tmp_path / "trial_000",
    )
    cfg = load_config(path)
    assert cfg.optim.epochs == 15 and cfg.eval.patience is None
    assert cfg.optim.groups == {"generator": 1e-3, "interface": 1e-4}
    assert cfg.struct is not None
    assert cfg.struct.active_weights["bce"] == 1.0
    assert cfg.struct.active_weights["rank"] == 3.0
    raw = yaml.safe_load(path.read_text())
    assert raw["model"]["config"]["coord_gen"]["reader_checkpoint"].endswith("best.pt")


def test_t2_inherits_t1_corruption_and_adds_mandatory_subgraph_bce(tmp_path: Path) -> None:
    params = dict(zip(SPACES["T2"], PRIOR_VALUES["T2"][0], strict=True))
    path = materialize(
        Path("configs/split_seed42/topo_prompt_full_v2.yaml"), params, tmp_path / "trial_000"
    )
    cfg = load_config(path)
    assert cfg.struct is not None and cfg.struct.active_weights["bce"] == 1.0
    assert (
        yaml.safe_load(path.read_text())["model"]["config"]["topo_prompt"]["corruption"]["prob"]
        == 0.5
    )


def _run(tmp_path: Path, diagnostic: bool = False) -> Path:
    marker = "diagnostic_complete" if diagnostic else "complete"
    (tmp_path / f"{marker}.json").write_text(json.dumps({"status": marker}))
    (tmp_path / "run_metadata.json").write_text(json.dumps({"selected_epoch": 2}))
    rows = [
        {
            "epoch": epoch,
            "val_auprc": 0.8,
            "val_gs_bfs": 0.4,
            "val_rd_bfs": 0.8,
            "val_degree_mmd_ratio": 2.0,
            "val_clustering_mmd_ratio": 3.0,
            "val_spectral_mmd_ratio": 4.0,
            "val_threshold": 1.0,
            "sensitivity_noise_logit_change_p95": 2.1,
            "stability_edge_jaccard": 0.9 if epoch != 5 else 0.6,
        }
        for epoch in range(1, 16)
    ]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    return tmp_path


def test_t1_uses_diagnostic_sentinel_and_sensitivity_filter(tmp_path: Path) -> None:
    result = outcome(_run(tmp_path, diagnostic=True), "T1", None)
    assert not result["selectable"]
    assert result["constraints"][0] > 0  # soft RD constraint, not a run failure
    assert result["values"] == pytest.approx([0.4, 24 ** (1 / 3)])


def test_s1_checks_all_consecutive_epochs_not_only_selected(tmp_path: Path) -> None:
    result = outcome(_run(tmp_path), "S1", None)
    assert not result["selectable"]
    assert result["diagnostics"]["minimum_edge_jaccard"] == 0.6


def test_s2_uses_classification_constraint(tmp_path: Path) -> None:
    result = outcome(_run(tmp_path), "S2", 0.82)
    assert not result["selectable"]
    assert result["constraints"][1] == pytest.approx(0.01)
    with pytest.raises(ValueError, match="requires"):
        outcome(tmp_path, "S2", None)


def test_incomplete_stage2_budget_and_failure_marker_rejected(tmp_path: Path) -> None:
    run = _run(tmp_path)
    metrics = run / "metrics.jsonl"
    metrics.write_text("\n".join(metrics.read_text().splitlines()[:7]))
    with pytest.raises(ValueError, match="full 15"):
        outcome(run, "S1", None)
    (run / "failure.json").write_text("{}")
    with pytest.raises(ValueError, match="failed trial"):
        outcome(run, "S1", None)


def test_final_ranking_excludes_declared_filter_but_not_soft_rd(tmp_path: Path) -> None:
    result = outcome(_run(tmp_path), "S3_size", None)
    study = optuna.create_study(directions=["maximize", "minimize"])
    for selectable in (False, True):
        trial = study.ask()
        for key, value in result.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("selectable", selectable)
        study.tell(trial, result["values"])
    winner = select_winner(study.get_trials())
    assert winner is not None and winner.number == 1

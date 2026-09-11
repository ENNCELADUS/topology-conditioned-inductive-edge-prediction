"""Follow-up selection and new joint HPO stay on the V_val surface."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from src.distill.config import DistillConfig
from src.eval.checkpoint_selection import TopologyValidationMetrics
from src.experiments import kd_followup_queue as queue
from src.experiments import kd_logit_rep_hpo as hpo


def test_select_run_uses_five_metrics_not_auprc_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = [Path("a"), Path("b")]
    metrics = {
        runs[0]: SimpleNamespace(auprc=0.9, topology=TopologyValidationMetrics(0.2, 1, 9, 9, 9)),
        runs[1]: SimpleNamespace(auprc=0.8, topology=TopologyValidationMetrics(0.4, 1, 2, 2, 2)),
    }
    monkeypatch.setattr(queue, "read_run", metrics.__getitem__)
    assert queue.select_run(runs) == Path("b")
    with pytest.raises(ValueError, match="No completed"):
        queue.select_run([])


def test_joint_hpo_changes_only_weights_and_output(tmp_path: Path) -> None:
    base = Path("configs/split_seed42/sweep/kd_logit_w10.yaml")
    original = yaml.safe_load(base.read_text())
    path = hpo.materialize_trial_config(base, {"w_logit": 3.0, "w_rep": 0.2}, 0, tmp_path)
    config = yaml.safe_load(path.read_text())
    assert DistillConfig.from_mapping(config["distill"]).arm == "kd_logit_rep"
    config["output_dir"] = original["output_dir"]
    config["distill"].pop("w_rep")
    config["distill"]["w_logit"] = 10.0
    assert config == original
    config["distill"]["targets_path"] = str(tmp_path / "missing")
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(FileNotFoundError):
        hpo.require_rows(argparse.Namespace(base_config=path))


def test_test_failure_does_not_get_swallowed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail(argv: list[str]) -> None:
        raise RuntimeError("score failed")

    monkeypatch.setattr(queue, "run_command", fail)
    with pytest.raises(RuntimeError, match="score failed"):
        queue.test_winner(tmp_path, "kd_logit")


@pytest.mark.parametrize("predecessor", ["complete", "failed"])
def test_queue_stage_order_and_predecessor_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, predecessor: str
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "runs"
    root.mkdir()
    (root / "lane_kd.json").write_text(json.dumps({"status": predecessor}))
    configs = tmp_path / "configs/split_seed42/sweep"
    configs.mkdir(parents=True)
    for i in range(5):
        (configs / f"kd_gram_{i}.yaml").write_text("output_dir: unused\n")
    winner = root / "winner"
    winner.mkdir()
    (winner / "run_metadata.json").write_text('{"checkpoint_id": "test"}')
    monkeypatch.setattr(queue, "select_run", lambda runs: winner)
    monkeypatch.setattr(queue, "read_run", lambda run: None)
    events: list[str] = []
    monkeypatch.setattr(queue, "test_winner", lambda run, arm: events.append(arm))
    monkeypatch.setattr(
        queue,
        "run_command",
        lambda argv: events.append(
            "train" if argv[:3] == ["bash", "hpc/run.sh", "train"] else "hpo"
        ),
    )
    monkeypatch.setattr(sys, "argv", ["queue", "--root", str(root), "--wait-pid", "0"])
    if predecessor == "failed":
        with pytest.raises(RuntimeError, match="predecessor did not complete"):
            queue.main()
        assert events == []
    else:
        queue.main()
        assert events == ["kd_rank", "kd_rank_rep", "kd_rep", "kd_logit"] + ["train"] * 5 + [
            "kd_gram",
            "hpo",
        ]
    assert json.loads((root / "followup_queue.json").read_text())["stage"] == predecessor

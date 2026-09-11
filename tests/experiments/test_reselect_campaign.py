from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from src.eval.checkpoint_selection import SELECTION_RULE
from src.experiments import reselect_campaign as replay
from src.score_universe import ScoresArtifact, save_scores


def test_retry_resumes_latest_target_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "new"
    attempt = target / "attempts/latest"
    attempt.mkdir(parents=True)
    (attempt / "training_state.pt").write_bytes(b"state")
    (target / "current_attempt.json").write_text('{"attempt_id":"latest"}')
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append(command))
    replay.replay_run(tmp_path / "old", target, Path("old.yaml"), Path("pack"))
    assert calls == [
        [
            "bash",
            "hpc/run.sh",
            "train",
            str(target / "config.yaml"),
            "--skip-test",
            "--resume-attempt",
            str(attempt),
        ]
    ]


def test_validation_union_contains_only_validation_nodes() -> None:
    manifest = {
        "v_val": ["a", "b", "c"],
        "val_positives": [["a", "b"]],
        "val_negatives": [["a", "c"]],
        "buckets": {"2": [["a", "b"]]},
    }
    pairs, classification = replay.validation_union(manifest)
    assert pairs == [("a", "a"), ("a", "b"), ("a", "c"), ("b", "b")]
    assert classification == {("a", "b"), ("a", "c")}
    manifest["val_negatives"] = [["a", "test_node"]]
    with pytest.raises(ValueError, match="escaped V_val"):
        replay.validation_union(manifest)


@pytest.mark.parametrize("interrupted", [False, True])
def test_replay_publishes_new_selection_without_touching_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data/val_region"
    data.mkdir(parents=True)
    manifest = {
        "v_val": ["a", "b", "c"],
        "val_positives": [["a", "a"], ["a", "b"], ["b", "c"]],
        "val_negatives": [["a", "c"], ["b", "b"], ["c", "c"]],
        "buckets": {"3": [["a", "b", "c"]] * 4},
    }
    (data / "breadth_first.json").write_text(json.dumps(manifest))
    source = tmp_path / "old"
    attempt = source / "attempts/a"
    (attempt / "checkpoints").mkdir(parents=True)
    (source / ("current_attempt.json" if interrupted else "complete.json")).write_text(
        '{"attempt_id":"a"}'
    )
    if interrupted:
        torch.save({"epoch": 2, "optimizer": {"sentinel": 42}}, attempt / "training_state.pt")
    for epoch in [1, 2]:
        torch.save(
            {"epoch": epoch, "model_family": "v3_1", "model_state": {"w": torch.ones(1)}},
            attempt / "checkpoints" / f"epoch-{epoch:04d}.pt",
        )
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"seed": 0, "output_dir": str(source)}))
    originals = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    calls = []

    def score(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        if command[2] == "train":
            assert interrupted and "--skip-test" in command
            resume = Path(command[command.index("--resume-attempt") + 1])
            assert torch.load(resume / "training_state.pt", weights_only=False)["optimizer"] == {
                "sentinel": 42
            }
            rows = [
                json.loads(line) for line in (resume / "metrics.jsonl").read_text().splitlines()
            ]
            for row in rows:
                payload = torch.load(
                    resume / "checkpoints" / f"epoch-{row['epoch']:04d}.pt", weights_only=False
                )
                assert payload["selection_metrics"] == row
                assert payload["val_metrics"]["auprc"] == row["val_auprc"]
                assert row["selection_rule"] == SELECTION_RULE
            return subprocess.CompletedProcess(command, 0)
        assert check and command[:3] == ["bash", "hpc/run.sh", "score"]
        assert "test" not in command and "test_topology" not in command
        calls.append(command)
        output = Path(command[command.index("--output") + 1])
        epoch = int(output.stem.split("-")[1])
        labels = np.array([1, 1, 0, 0, 1, 0], dtype=np.int8)
        logits = (2 * labels - 1).astype(np.float32) * (1 if epoch == 2 else -1)
        artifact = ScoresArtifact(
            node_ids=["a", "b", "c"],
            u_idx=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
            v_idx=np.array([0, 1, 2, 1, 2, 2], dtype=np.int32),
            logit=logits,
            label=labels,
            meta={
                "pairs_source": "file:validation_union.tsv",
                "strategy": "breadth_first",
                "num_rows": 6,
                "created_utc": "2026-09-10T00:00:00Z",
                "torch_version": str(torch.__version__),
                "model_family": "v3_1",
                "checkpoint_id": f"epoch{epoch}",
                "score_precision": {
                    "encode_autocast": "bf16",
                    "pair_autocast": "bf16",
                    "logit_storage_dtype": "float32",
                },
            },
        )
        save_scores(output, row_start=0, **asdict(artifact))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", score)
    target = tmp_path / "new"
    replay.replay_run(source, target, config, Path("pack"))
    if interrupted:
        assert all(path.read_bytes() == value for path, value in originals.items())
        assert not (target / "best.pt").exists()
        return
    metadata = json.loads((target / "run_metadata.json").read_text())
    assert metadata["selected_epoch"] == 2
    best = torch.load(target / "best.pt", weights_only=False)
    assert best["selection_rule"] == SELECTION_RULE
    assert best["val_threshold_transfer"]["threshold"] == 1.0
    assert best["val_metrics"]["auprc"] == 1.0
    assert all(path.read_bytes() == value for path, value in originals.items())
    replay.replay_run(source, target, config, Path("pack"))
    assert len(calls) == 2

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _low_thread_env() -> dict[str, str]:
    """Prevent two CPU ranks from each creating a full BLAS/OpenMP thread pool."""
    return {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }


# Troubleshooting: if `torch.distributed.run --standalone` below hangs for minutes or
# fails with `torch.distributed.DistNetworkError: Failed to recv, got 0 bytes`, a
# VPN/security client is likely hijacking hostname resolution (resolving this host's
# name to a virtual address it intercepts), which breaks the c10d rendezvous TCPStore.
# Export PET_LOCAL_ADDR=localhost before running pytest — torchrun's env override for
# --local-addr — to force the rendezvous onto loopback. Not needed on the fixed
# H20 container.
@pytest.mark.integration
def test_two_rank_cpu_plan_has_exact_global_coverage(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "tests/helpers/e2_ddp_smoke.py",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        env=_low_thread_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    rows = []
    for rank in range(2):
        rows.extend(json.loads((tmp_path / f"rank-{rank}.json").read_text()))
    assert sorted(rows) == list(range(64))
    assert len(rows) == len(set(rows))


@pytest.mark.integration
def test_two_rank_cpu_runs_real_ddp_train_eval_and_rank_zero_outputs(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "tests/helpers/e2_ddp_smoke.py",
            "--mode",
            "train",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        env=_low_thread_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr

    summaries = [
        json.loads((tmp_path / f"train-rank-{rank}.json").read_text()) for rank in range(2)
    ]
    assert all(summary["epochs_completed"] == 2 for summary in summaries)
    assert all(summary["validations_completed"] == 2 for summary in summaries)
    assert all(summary["training_coverage_exact"] is True for summary in summaries)
    assert all(summary["validation_coverage_exact"] is True for summary in summaries)
    assert all(summary["best_epoch"] >= 1 for summary in summaries)
    assert summaries[0]["weights_synchronized"] is True
    assert summaries[1]["weights_synchronized"] is True

    best = torch.load(tmp_path / "artifacts" / "best.pt", weights_only=False)
    assert set(best) == {
        "model_state",
        "model_family",
        "model_config",
        "epoch",
        "val_metrics",
        "seed",
        "config",
    }
    assert (tmp_path / "artifacts" / "last.pt").exists()
    assert (tmp_path / "artifacts" / "metrics.jsonl").exists()
    assert (tmp_path / "artifacts" / "run_metadata.json").exists()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("RUN_E2_H20_ACCEPTANCE") != "1",
    reason="set RUN_E2_H20_ACCEPTANCE=1 only on an H20 container",
)
def test_cold_four_h20_run_meets_budget(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "hpc/run.sh",
            "train",
            "configs/b0_v31_breadth_first.yaml",
            "--pack-dir",
            str(tmp_path / "cold-pack"),
            "--output-dir",
            str(tmp_path / "outputs"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=3700,
    )
    assert result.returncode == 0, result.stderr
    profile = json.loads((tmp_path / "outputs/profile.json").read_text())
    completion = json.loads((tmp_path / "outputs/complete.json").read_text())
    assert profile["cold_cache"] is True
    assert completion["status"] == "complete"
    assert completion["total_seconds"] <= 3600
    assert profile["total_seconds"] <= completion["total_seconds"]
    assert profile["epochs_completed"] == 30
    assert profile["validations_completed"] == 30
    assert max(profile["peak_memory_gib_per_rank"]) <= 85.0
    assert profile["steady_state_data_wait_fraction"] <= 0.05
    assert profile["training_coverage_exact"] is True
    assert profile["validation_coverage_exact"] is True
    for filename in (
        "best.pt",
        "last.pt",
        "metrics.jsonl",
        "run_metadata.json",
        "profile.json",
        "artifact_manifest.json",
        "complete.json",
    ):
        assert (tmp_path / "outputs" / filename).exists()


@pytest.mark.integration
def test_two_rank_cpu_egostitch_loop_emits_valid_profile_and_artifacts(tmp_path: Path) -> None:
    """The EgoStitch worker's loop satisfies the orchestrator contracts under real DDP."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "tests/helpers/egostitch_ddp_smoke.py",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        env=_low_thread_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "smoke_ok.json").read_text())
    assert summary["world_size"] == 2
    assert summary["epochs_completed"] == 2
    for filename in ("best.pt", "last.pt", "metrics.jsonl", "run_metadata.json"):
        assert (tmp_path / "run" / filename).is_file()

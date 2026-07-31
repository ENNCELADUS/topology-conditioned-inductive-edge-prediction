from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    sys.platform == "darwin",
    reason="CPU DDP contracts run through hpc/run.sh check on Linux",
)


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
        timeout=60,
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


def _run_egostitch_smoke(output_dir: Path, *, mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "tests/helpers/egostitch_ddp_smoke.py",
            "--output-dir",
            str(output_dir),
            "--mode",
            mode,
        ],
        cwd=REPO_ROOT,
        env=_low_thread_env(),
        capture_output=True,
        text=True,
        check=False,
        # A hung rank is the failure this pair of tests exists to catch, so the
        # deadline must be generous enough that a slow-but-live container is
        # never mistaken for one.
        timeout=120,
    )


def _run_egostitch_validation_worker(
    output_dir: Path, *, world_size: int
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable]
    if world_size > 1:
        command.extend(
            [
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={world_size}",
            ]
        )
    command.extend([str(Path(__file__)), "--validation-worker-output", str(output_dir)])
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_low_thread_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def _validation_worker(output_dir: Path) -> None:
    """Exercise corrected E2E validation inside a real process group."""
    os.environ.setdefault("ACCELERATE_USE_CPU", "true")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from src import train_egostitch as te
    from src.model.egostitch.conditioning import HeadNullMasks
    from src.model.egostitch.config import E2EConfig
    from src.model.egostitch.e2e_model import E2ENodeState, E2EPairContext, EgoStitchE2E

    from tests.helpers import egostitch_ddp_smoke as smoke

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    pack_dir = output_dir / "token_pack"
    cfg = smoke._toy_config(output_dir, pack_dir)
    model = EgoStitchE2E(E2EConfig.from_mapping(cfg.model.config))
    data = smoke._toy_bundle(model.generator_cfg)
    data.val_pairs = [
        ("n0", "n1"),
        ("n1", "n2"),
        ("n2", "n3"),
        ("n3", "n4"),
        ("n4", "n0"),
    ]
    data.val_labels = np.asarray([0, 1, 0, 1, 0], dtype=np.int8)
    accelerator = te.build_egostitch_ddp_accelerator(cfg.mixed_precision)
    if accelerator.is_main_process:
        smoke._write_tiny_token_pack(pack_dir)
    accelerator.wait_for_everyone()
    factory = te._BatchFactory(
        cfg,
        model.generator_cfg,
        data,
        node_batch=cfg.data.node_batch,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )

    encoded_rows = 0
    pair_output_is_fp32 = True
    original_encode = model.encode_node_state
    original_score = model.score_pair_context

    def counted_encode(
        self: EgoStitchE2E,
        emb: torch.Tensor,
        length: torch.Tensor,
        x: torch.Tensor,
        ground: torch.Tensor,
        ground_ids: torch.Tensor | None = None,
    ) -> E2ENodeState:
        nonlocal encoded_rows
        encoded_rows += int(emb.shape[0])
        return original_encode(emb, length, x, ground, ground_ids)

    def checked_score(
        self: EgoStitchE2E,
        context: E2EPairContext,
        *,
        masks: HeadNullMasks | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        nonlocal pair_output_is_fp32
        output = original_score(context, masks=masks, edge_mask=edge_mask)
        pair_output_is_fp32 &= output.dtype == torch.float32
        return output

    model.encode_node_state = types.MethodType(counted_encode, model)  # type: ignore[method-assign]
    model.score_pair_context = types.MethodType(  # type: ignore[method-assign]
        checked_score, model
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        validation = te._validate_epoch(
            model,
            data,
            accelerator,
            edge_batch=2,
            topk_fraction=cfg.diagnostics.topk_fraction,
            token_table=factory._token_table,
            token_node_index=factory._token_node_index,
        )
    rank = accelerator.process_index
    shard_rows = list(range(rank, len(data.val_pairs), accelerator.num_processes))
    shard_len = (len(data.val_pairs) + accelerator.num_processes - 1) // accelerator.num_processes
    while len(shard_rows) < shard_len:
        shard_rows.append(shard_rows[0] if shard_rows else 0)
    expected_nodes = {
        node for row in shard_rows for node in data.val_pairs[row]
    }
    (output_dir / f"validation-rank-{rank}.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "encoded_rows": encoded_rows,
                "expected_unique_nodes": len(expected_nodes),
                "pair_output_is_fp32": pair_output_is_fp32,
                "returned_result": validation is not None,
            },
            sort_keys=True,
        )
        + "\n"
    )
    if accelerator.is_main_process:
        assert validation is not None
        (output_dir / "validation-result.json").write_text(
            json.dumps(
                {
                    "metrics": dataclasses.asdict(validation.metrics),
                    "active_logits": validation.active_logits.tolist(),
                    "active_logits_dtype": validation.active_logits.dtype.str,
                },
                sort_keys=True,
            )
            + "\n"
        )
    accelerator.wait_for_everyone()


@pytest.mark.integration
def test_two_rank_cpu_egostitch_validation_is_exact_and_deadlock_free(tmp_path: Path) -> None:
    """A padded two-rank shard matches serial validation without repeat encoding."""
    ddp_dir = tmp_path / "ddp"
    ddp_run = _run_egostitch_validation_worker(ddp_dir, world_size=2)
    assert ddp_run.returncode == 0, ddp_run.stderr

    serial_dir = tmp_path / "serial"
    serial_run = _run_egostitch_validation_worker(serial_dir, world_size=1)
    assert serial_run.returncode == 0, serial_run.stderr

    rank_summaries = [
        json.loads((ddp_dir / f"validation-rank-{rank}.json").read_text())
        for rank in range(2)
    ]
    assert [summary["expected_unique_nodes"] for summary in rank_summaries] == [5, 4]
    assert all(
        summary["encoded_rows"] == summary["expected_unique_nodes"]
        for summary in rank_summaries
    )
    assert all(summary["pair_output_is_fp32"] is True for summary in rank_summaries)
    assert [summary["returned_result"] for summary in rank_summaries] == [True, False]

    ddp_result = json.loads((ddp_dir / "validation-result.json").read_text())
    serial_result = json.loads((serial_dir / "validation-result.json").read_text())
    assert ddp_result["active_logits_dtype"] == "<f4"
    assert len(ddp_result["active_logits"]) == 5
    assert ddp_result["active_logits"] == pytest.approx(
        serial_result["active_logits"], rel=1e-5, abs=5e-7
    )
    assert ddp_result["metrics"] == pytest.approx(serial_result["metrics"], abs=1e-7)


@pytest.mark.integration
def test_two_rank_cpu_egostitch_step_zero_guard_admits_a_healthy_model(tmp_path: Path) -> None:
    """The E2E step-0 slot guard runs to completion on both ranks."""
    result = _run_egostitch_smoke(tmp_path, mode="healthy")

    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "smoke_ok.json").read_text())
    assert summary["world_size"] == 2
    assert summary["run_kind"] == "formal"
    assert summary["validation_rows"] > 0
    assert summary["h_pairwise_cosine_mean"] <= 0.95


@pytest.mark.integration
def test_two_rank_cpu_egostitch_step_zero_guard_raises_on_every_rank(tmp_path: Path) -> None:
    """A born-collapsed model must abort both ranks, not hang the process group.

    `_validate_epoch` returns ``None`` off the main process, so the guard
    reduces its verdict before raising; a rank-zero-only raise would leave rank
    one blocked forever and the launcher would time out (design 2026-07-29
    Sec 4.1). Only a real process group can tell those two outcomes apart.
    """
    result = _run_egostitch_smoke(tmp_path, mode="collapsed")

    assert result.returncode == 0, result.stderr
    for rank in range(2):
        payload = json.loads((tmp_path / f"raised-rank-{rank}.json").read_text())
        assert payload["rank"] == rank
        assert payload["error"] == "training_invalid(initial_slot_collapse)"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-worker-output", type=Path, required=True)
    worker_args = parser.parse_args()
    _validation_worker(worker_args.validation_worker_output)

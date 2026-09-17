"""Two-rank CPU DDP agreement for the motif-prompt family."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_e2_ddp_integration import REPO_ROOT, _low_thread_env

pytestmark = pytest.mark.skipif(
    sys.platform == "darwin",
    reason="CPU DDP contracts run through hpc/run.sh check on Linux",
)


@pytest.mark.integration
def test_two_ranks_build_identical_templates_and_agree_on_gradients(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "tests/helpers/motif_prompt_ddp_smoke.py",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        env=_low_thread_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    reports = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    # Every rank compiled the same templates and holds the same mean adjacency.
    assert reports[0]["template_digest"] == reports[1]["template_digest"]
    assert reports[0]["mean_digest"] == reports[1]["mean_digest"]
    # The ranks really scored different rows, so the agreement below is earned.
    assert reports[0]["batch_digest"] != reports[1]["batch_digest"]
    # After one synchronised step the parameters agree exactly.
    assert reports[0]["param_digest"] == reports[1]["param_digest"]
    # The interface group is frozen in epoch 1 on both ranks.
    assert all(report["interface_lr"] == 0.0 for report in reports)
    assert all(report["interface_open"] is False for report in reports)
    # The self row rides along with a zero L_slot/L_topo mask on the rank that owns it.
    assert min(report["self_row_masked"] for report in reports) == 0.0

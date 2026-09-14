"""L3-PPI checkpoint/scoring and isolated production-runner integration."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from src import score_universe

from tests.test_score_universe import _data_root_with_features, _write_tsv


def _config() -> dict[str, object]:
    return {
        "backbone_config": {
            "input_dim": 4,
            "d_model": 8,
            "encoder_layers": 1,
            "n_heads": 2,
            "regularization": {"dropout": 0.0},
        },
        "k": 2,
        "surrogate_hidden": 4,
        "gate_hidden": 4,
        "layers": 2,
        "dropout": 0.0,
    }


def test_l3ppi_checkpoint_roundtrip_without_training_graph(tmp_path: Path) -> None:
    model = score_universe.build_model("l3ppi", _config()).eval()
    path = tmp_path / "best.pt"
    torch.save(
        {"model_family": "l3ppi", "model_config": _config(), "model_state": model.state_dict()},
        path,
    )
    loaded, family, _ = score_universe._load_checkpoint(path)
    assert family == "l3ppi"
    a, b = torch.randn(3, 16), torch.randn(3, 16)
    with torch.no_grad():
        torch.testing.assert_close(model(a, b)[0], loaded(a, b)[0])
    assert not loaded.training


def test_l3ppi_score_dispatch_records_actual_precision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.baselines import l3ppi_features

    model = score_universe.build_model("l3ppi", _config()).eval()
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {"model_family": "l3ppi", "model_config": _config(), "model_state": model.state_dict()},
        checkpoint,
    )
    data_root = _data_root_with_features(tmp_path, {"a": torch.randn(2, 4), "b": torch.randn(3, 4)})
    pairs = tmp_path / "pairs.tsv"
    _write_tsv(pairs, [("a", "b", None)])
    pack = tmp_path / "pack"
    calls = []

    def fake_score(
        model: torch.nn.Module,
        pack_dir: Path,
        pairs: list[tuple[str, str]],
        device: torch.device,
        batch_size: int = 64,
    ) -> NDArray[np.float32]:
        calls.append((pack_dir, pairs, str(device), batch_size))
        return np.array([0.1234567], dtype=np.float32)

    monkeypatch.setattr(l3ppi_features, "score_pairs", fake_score)
    output = tmp_path / "scores.npz"
    score_universe.main(
        [
            "score",
            "--checkpoint",
            str(checkpoint),
            "--pairs",
            f"file:{pairs}",
            "--data-root",
            str(data_root),
            "--pack-dir",
            str(pack),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--amp",
            "bf16",
            "--batch-pairs",
            "7",
        ]
    )
    assert calls == [(pack, [("a", "b")], "cpu", 7)]
    scores = score_universe.load_scores(output)
    assert scores.logit.dtype == np.float32
    precision = scores.meta["score_precision"]
    assert isinstance(precision, dict)
    assert precision["encode_autocast"] == "off"
    assert precision["pair_compute_dtype"] == "float32"


@pytest.mark.parametrize(
    ("arguments", "tests_expected"),
    [
        (["--worker-module", "src.train_l3ppi", "--stage", "surrogate"], False),
        (["--skip-test", "--worker-module", "src.train_l3ppi", "--resume"], False),
        (["--stage", "trial", "--worker-module", "src.train_l3ppi"], True),
    ],
)
def test_l3ppi_runner_dispatch(tmp_path: Path, arguments: list[str], tests_expected: bool) -> None:
    # Exercise the production branch with a recording executable, without H20 hardware.
    source = Path("hpc/run.sh").read_text()
    branch = source.split("    L3_WORKER=false", 1)[1].split(
        '    if [[ $# -eq 2 && "$1" == "--worker-module"', 1
    )[0]
    script = tmp_path / "dispatch.sh"
    script.write_text("set -eu\nfail() { exit 2; }\nL3_WORKER=false" + branch)
    executable = tmp_path / "python"
    executable.write_text("""#!/usr/bin/env python3
import json, os, sys
with open(os.environ['CALL_LOG'], 'a') as handle:
    handle.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1] == '-c':
    print('output pack 0 data')
""")
    executable.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    env = dict(
        os.environ,
        PYTHON_BIN=str(executable),
        GPU_COUNT="3",
        CONFIG_PATH="config.yaml",
        CALL_LOG=str(log),
    )
    subprocess.run(["bash", str(script), *arguments], env=env, check=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0][:6] == [
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=3",
        "-m",
        "src.train_l3ppi",
    ]
    assert any("src.eval.test_protocol" in call for call in calls) == tests_expected
    assert ("--resume" in calls[0]) == ("--resume" in arguments)

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from src.experiments import motif_shift_diagnostic as diagnostic


class _Artifact(SimpleNamespace):
    def pairs(self) -> Iterator[tuple[str, str]]:
        return iter([("a", "b"), ("a", "c"), ("b", "c"), ("c", "d")])


def test_analyze_uses_actual_aligned_rows_and_emits_queue_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
    logits = {
        "s2_true": np.asarray([-3, 3, -2, 2], dtype=np.float32),
        "s2_pred": np.asarray([2, 1, -1, 0], dtype=np.float32),
        "gates_off": np.asarray([2, 0, -1, 1], dtype=np.float32),
        "blur_true": np.asarray([-1, 2, 0, 1], dtype=np.float32),
        "presence_true": np.asarray([1, 2, -1, 0], dtype=np.float32),
        "level_true": np.asarray([0, 2, -1, 1], dtype=np.float32),
        "content_true": np.asarray([-1, 3, 0, 2], dtype=np.float32),
    }
    artifacts = {
        name: _Artifact(
            label=labels,
            logit=value,
            u_idx=np.asarray([0, 0, 1, 2]),
            v_idx=np.asarray([1, 2, 2, 3]),
            meta={},
        )
        for name, value in logits.items()
    }
    monkeypatch.setattr(diagnostic, "load_scores", lambda path: artifacts[path.stem])
    monkeypatch.setattr(diagnostic, "validate_artifact_precision", lambda *args, **kwargs: None)
    paths = {name: tmp_path / f"{name}.npz" for name in diagnostic.CASES}

    result = diagnostic.analyze(paths, checkpoint=tmp_path / "best.pt")

    assert result["schema"] == "motif_shift_step0_v1"
    assert result["rows"] == 4
    assert set(result["metrics"]) == set(diagnostic.CASES)
    decisions = result["decisions"]
    assert isinstance(decisions["run_shift"], bool)
    assert isinstance(decisions["run_confidence"], bool)
    assert decisions["stop"] == (not decisions["run_shift"] and not decisions["run_confidence"])


def test_analyze_rejects_row_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
    artifact = _Artifact(label=labels, logit=np.asarray([-2, 2, -1, 1], dtype=np.float32))
    bad = _Artifact(label=labels[::-1], logit=np.asarray([-2, 2, -1, 1], dtype=np.float32))
    monkeypatch.setattr(
        diagnostic,
        "load_scores",
        lambda path: bad if path.stem == "content_true" else artifact,
    )
    monkeypatch.setattr(diagnostic, "validate_artifact_precision", lambda *args, **kwargs: None)
    paths = {name: tmp_path / f"{name}.npz" for name in diagnostic.CASES}

    try:
        diagnostic.analyze(paths, checkpoint=tmp_path / "best.pt")
    except ValueError as error:
        assert "ordered s2_true" in str(error)
    else:
        raise AssertionError("row mismatch must fail closed")

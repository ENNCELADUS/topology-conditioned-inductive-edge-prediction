"""Shared pytest fixtures for repository-local benchmark artifacts.

The real artifact package lives under ``data/`` (gitignored). Tests that need it
resolve the root via the ``TCIEP_DATA_ROOT`` environment variable, falling back
to ``<repo>/data``. Tests requiring real artifacts must use the fixtures below,
which skip cleanly when the data is absent (e.g. in a bare checkout).
"""

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("TCIEP_DATA_ROOT", REPO_ROOT / "data"))
BENCHMARK_ROOT = DATA_ROOT / "benchmark_2025_neurips"
FEATURES_ROOT = DATA_ROOT / "features" / "frozen_node_features_1024"


@pytest.fixture(scope="session")
def benchmark_root() -> Path:
    """Path to the benchmark artifact package; skips if absent."""
    if not (BENCHMARK_ROOT / "graph.pkl").exists():
        pytest.skip(f"benchmark artifacts not found under {BENCHMARK_ROOT}")
    return BENCHMARK_ROOT


@pytest.fixture(scope="session")
def features_root() -> Path:
    """Path to the frozen feature cache; skips if absent."""
    if not (FEATURES_ROOT / "index.json").exists():
        pytest.skip(f"feature cache not found under {FEATURES_ROOT}")
    return FEATURES_ROOT

"""End-to-end smoke test for `src.experiments.s2_latent_topology.cli`.

Drives `cli.main` through `--stage all` (sample -> train-ae -> train-prior ->
train-det -> train-marg -> eval) on a fully synthetic benchmark tree and
feature store, at CPU-tiny model dims, then checks `--stage eval` is
byte-deterministic on repeated runs against the same checkpoints.
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch
from src.experiments.s2_latent_topology.cli import main

pytestmark = pytest.mark.unit

_INPUT_DIM = 8
_TOKEN_LEN = 5
_STRATEGY = "breadth_first"


def _make_train_graph() -> nx.Graph:
    """A 60-node connected graph, ``node_%06d`` ids, with a couple self-loops."""
    base = nx.connected_watts_strogatz_graph(60, k=4, p=0.3, seed=0)
    graph = nx.relabel_nodes(base, {i: f"node_{i:06d}" for i in base.nodes()})
    graph.add_edge("node_000000", "node_000000")
    graph.add_edge("node_000010", "node_000010")
    return graph


def _make_test_graph() -> nx.Graph:
    """A 24-node connected graph with ids disjoint from `_make_train_graph`, 2 self-loops."""
    base = nx.connected_watts_strogatz_graph(24, k=4, p=0.3, seed=1)
    ids = [f"node_{100000 + i:06d}" for i in range(24)]
    graph = nx.relabel_nodes(base, dict(enumerate(ids)))
    graph.add_edge(ids[0], ids[0])
    graph.add_edge(ids[5], ids[5])
    return graph


def _make_test_buckets(test_nodes: list[str]) -> dict[int, list[set[str]]]:
    """Hand-sampled buckets: 4 six-node sets and 4 eight-node sets, from `test_nodes`."""
    rng = np.random.default_rng(3)
    return {
        size: [set(rng.choice(test_nodes, size=size, replace=False).tolist()) for _ in range(4)]
        for size in (6, 8)
    }


def _write_benchmark(data_root: Path, train_graph: nx.Graph, test_graph: nx.Graph) -> None:
    """Write `train_graph.pkl`, `test_graph.pkl`, and `test_node_buckets.pkl`."""
    strategy_dir = data_root / "benchmark_2025_neurips" / _STRATEGY
    strategy_dir.mkdir(parents=True)
    with (strategy_dir / "train_graph.pkl").open("wb") as f:
        pickle.dump(train_graph, f)
    with (strategy_dir / "test_graph.pkl").open("wb") as f:
        pickle.dump(test_graph, f)
    buckets = _make_test_buckets(sorted(test_graph.nodes()))
    with (strategy_dir / "test_node_buckets.pkl").open("wb") as f:
        pickle.dump(buckets, f)


def _write_feature_root(data_root: Path, node_ids: list[str]) -> None:
    """Build a tiny synthetic FeatureStore root covering every `node_ids` entry."""
    root = data_root / "features" / "frozen_node_features_1024"
    embeddings_dir = root / "embeddings"
    embeddings_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    index: dict[str, str] = {}
    for node_id in node_ids:
        tensor = torch.tensor(rng.standard_normal((_TOKEN_LEN, _INPUT_DIM)), dtype=torch.float32)
        rel_path = f"embeddings/{node_id}.pt"
        torch.save(tensor, root / rel_path)
        index[node_id] = rel_path
    (root / "index.json").write_text(json.dumps(index))
    (root / "metadata.json").write_text(
        json.dumps(
            {"format": "torch_pt_per_node", "input_dim": _INPUT_DIM, "max_sequence_length": 1024}
        )
    )


def _cli_args(data_root: Path, output_dir: Path, *, stage: str) -> list[str]:
    """The pinned CPU-smoke CLI invocation, varying only `--stage`."""
    return [
        "--stage",
        stage,
        "--variant",
        "full",
        "--data-root",
        str(data_root),
        "--strategy",
        _STRATEGY,
        "--output-dir",
        str(output_dir),
        "--device",
        "cpu",
        "--seed",
        "0",
        "--regions-per-size",
        "4",
        "--holdout-frac",
        "0.25",
        "--sizes",
        "6,8",
        "--k-draws",
        "3",
        "--flow-steps",
        "4",
        "--batch-sets",
        "4",
        "--ae-epochs",
        "2",
        "--prior-epochs",
        "2",
        "--det-epochs",
        "2",
        "--marg-epochs",
        "2",
        "--z-dim",
        "4",
        "--ae-dim",
        "16",
        "--ae-layers",
        "2",
        "--ae-rrwp-k",
        "3",
        "--ae-heads",
        "4",
        "--ae-seeds",
        "2",
        "--ae-d-model",
        "32",
        "--set-width",
        "32",
        "--set-layers",
        "2",
        "--set-heads",
        "2",
        "--set-pma-seeds",
        "2",
        "--time-dim",
        "8",
        "--marg-hidden",
        "16",
    ]


def _assert_finite_or_none(value: object) -> None:
    """Recursively assert every float in a JSON-decoded payload is finite or `None`."""
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        assert math.isfinite(value), f"non-finite float in payload: {value}"
        return
    if isinstance(value, dict):
        for v in value.values():
            _assert_finite_or_none(v)
        return
    if isinstance(value, list):
        for v in value:
            _assert_finite_or_none(v)
        return
    raise AssertionError(f"unexpected payload value type: {type(value)}")


class TestS2CliEndToEnd:
    def test_all_stage_then_eval_is_byte_deterministic(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        output_dir = tmp_path / "out"
        train_graph = _make_train_graph()
        test_graph = _make_test_graph()
        _write_benchmark(data_root, train_graph, test_graph)
        all_node_ids = sorted(train_graph.nodes()) + sorted(test_graph.nodes())
        _write_feature_root(data_root, all_node_ids)

        exit_code = main(_cli_args(data_root, output_dir, stage="all"))
        assert exit_code == 0

        assert (output_dir / "corpus.pt").exists()
        for name in ("ae", "prior", "det", "marg"):
            assert (output_dir / f"{name}.pt").exists()
        assert (output_dir / "metrics.jsonl").exists()
        results_path = output_dir / "s2_results.json"
        assert results_path.exists()
        assert (output_dir / "s2_tables.md").exists()

        first_bytes = results_path.read_bytes()
        payload = json.loads(first_bytes)
        meta = payload["meta"]
        assert meta["evidence_class"] == "diagnostic"
        assert meta["formal"] is False
        assert meta["variant"] == "full"

        arms = payload["arms"]
        for arm_name in ("gen", "unc", "shuf", "det", "ae"):
            assert arm_name in arms
            per_size = arms[arm_name]["per_size"]
            assert "6" in per_size
            assert "8" in per_size
        assert "b0" not in arms  # no --b0-universe given
        assert "marg" not in arms  # marg.pt exists, but eval needs --b0-universe too

        _assert_finite_or_none(payload)

        # Determinism: re-run --stage eval twice more against the same checkpoints.
        eval_args = _cli_args(data_root, output_dir, stage="eval")
        assert main(eval_args) == 0
        second_bytes = results_path.read_bytes()
        assert main(eval_args) == 0
        third_bytes = results_path.read_bytes()

        assert first_bytes == second_bytes == third_bytes

    def test_train_ae_without_corpus_errors_clearly(self, tmp_path: Path) -> None:
        args = _cli_args(tmp_path / "data", tmp_path / "out", stage="train-ae")
        with pytest.raises((SystemExit, OSError)):
            main(args)

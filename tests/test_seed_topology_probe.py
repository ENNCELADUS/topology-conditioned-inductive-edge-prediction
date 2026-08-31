from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from src.experiments import seed_topology_probe
from src.experiments.seed_topology_probe import probe_latents, ridge_r2


def test_ridge_r2_recovers_linear_signal_and_shuffle_breaks_it() -> None:
    rng = np.random.default_rng(0)
    latents = rng.normal(size=(200, 8))
    target = latents @ rng.normal(size=8) + 0.01 * rng.normal(size=200)

    assert ridge_r2(latents, target) > 0.9
    assert ridge_r2(rng.permutation(latents), target) < 0.2


def test_probe_latents_reports_all_stats() -> None:
    graph = nx.gnp_random_graph(30, 0.2, seed=0)
    pairs = list(graph.edges())[:20]
    rng = np.random.default_rng(1)
    degrees = np.array([graph.degree[u] - 1 for u, _ in pairs], dtype=np.float64)
    latents = np.concatenate([degrees[:, None], rng.normal(size=(len(pairs), 7))], axis=1)

    report = probe_latents(graph, pairs, teacher=latents, generated=latents)

    assert set(report) == {
        "deg_u",
        "deg_v",
        "common_neighbors",
        "clustering_u",
        "clustering_v",
    }
    assert all(
        set(scores) == {"teacher_r2", "generated_r2", "shuffled_r2"}
        and all(isinstance(score, float) for score in scores.values())
        for scores in report.values()
    )
    assert report["deg_u"]["generated_r2"] > 0.9
    assert report["deg_u"]["shuffled_r2"] < 0.5


def test_probe_removes_query_edge_per_row_without_mutating_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = nx.Graph([("u", "v"), ("u", "a"), ("v", "a"), ("u", "b")])
    captured: list[np.ndarray] = []

    def capture_target(_latents: np.ndarray, target: np.ndarray) -> float:
        captured.append(target.copy())
        return 0.0

    monkeypatch.setattr(seed_topology_probe, "ridge_r2", capture_target)
    probe_latents(
        graph,
        [("u", "v")],
        teacher=np.zeros((1, 2)),
        generated=np.zeros((1, 2)),
    )

    assert captured[0].tolist() == [2.0]
    assert captured[3].tolist() == [1.0]
    assert graph.has_edge("u", "v")
    assert graph.degree["u"] == 3


def test_topology_targets_removes_query_self_loop_without_mutating_caller() -> None:
    graph = nx.Graph([("u", "u"), ("u", "a")])

    targets = seed_topology_probe._topology_targets(graph, [("u", "u")])

    assert targets["deg_u"].tolist() == [1.0]
    assert targets["deg_v"].tolist() == [1.0]
    assert graph.has_edge("u", "u")
    assert graph.degree["u"] == 3


def test_topology_targets_use_exact_local_masking_without_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = nx.Graph(
        [
            ("u", "v"),
            ("u", "a"),
            ("u", "b"),
            ("v", "a"),
            ("v", "c"),
            ("a", "b"),
            ("u", "u"),
        ]
    )
    original_edges = set(graph.edges())

    def reject_copy(*_args: object, **_kwargs: object) -> nx.Graph:
        raise AssertionError("topology targets must not copy the full graph")

    monkeypatch.setattr(graph, "copy", reject_copy)

    targets = seed_topology_probe._topology_targets(graph, [("u", "v"), ("u", "u")])

    np.testing.assert_array_equal(targets["deg_u"], [4.0, 3.0])
    np.testing.assert_array_equal(targets["deg_v"], [2.0, 3.0])
    np.testing.assert_array_equal(targets["common_neighbors"], [1.0, 3.0])
    np.testing.assert_allclose(targets["clustering_u"], [1.0, 2.0 / 3.0])
    np.testing.assert_allclose(targets["clustering_v"], [0.0, 2.0 / 3.0])
    assert set(graph.edges()) == original_edges
    assert graph.degree["u"] == 5


@pytest.mark.parametrize(
    ("latents", "target", "message"),
    [
        (np.zeros((4,)), np.zeros(4), "rank-2"),
        (np.zeros((4, 2)), np.zeros((4, 1)), "rank-1"),
        (np.zeros((4, 2)), np.zeros(3), "row count"),
        (np.zeros((4, 2)), np.zeros(4), "at least 5 rows"),
        (np.array([[np.nan], [0.0], [0.0], [0.0], [0.0]]), np.zeros(5), "finite"),
        (np.zeros((5, 1)), np.array([0.0, 0.0, np.inf, 0.0, 0.0]), "finite"),
    ],
)
def test_ridge_r2_rejects_invalid_inputs(
    latents: np.ndarray, target: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ridge_r2(latents, target)


@pytest.mark.parametrize(
    ("teacher", "generated", "message"),
    [
        (np.zeros((5,)), np.zeros((5, 2)), "teacher.*rank-2"),
        (np.zeros((5, 2)), np.zeros((5,)), "generated.*rank-2"),
        (np.zeros((5, 2)), np.zeros((4, 2)), "generated.*row count"),
        (np.zeros((5, 2)), np.zeros((5, 3)), "latent dimension"),
        (np.full((5, 2), np.nan), np.zeros((5, 2)), "teacher.*finite"),
        (np.zeros((5, 2)), np.full((5, 2), np.inf), "generated.*finite"),
    ],
)
def test_probe_latents_rejects_invalid_latents(
    teacher: np.ndarray, generated: np.ndarray, message: str
) -> None:
    graph = nx.path_graph([str(i) for i in range(6)])
    pairs = [(str(i), str(i + 1)) for i in range(5)]

    with pytest.raises(ValueError, match=message):
        probe_latents(graph, pairs, teacher=teacher, generated=generated)


def test_load_training_graph_uses_benchmark_derivation_and_kd_truth_convention(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    verified = SimpleNamespace(
        split=SimpleNamespace(
            train_nodes=frozenset({"a", "b", "c"}),
            train_graph=nx.Graph([("a", "b"), ("b", "c")]),
        ),
        positive_edges=frozenset({("a", "b"), ("b", "c")}),
    )
    raw = SimpleNamespace(
        split=SimpleNamespace(
            train_pairs=SimpleNamespace(pairs=[("a", "c"), ("a", "b")], labels=np.array([0, 1])),
            val_pairs=SimpleNamespace(pairs=[("b", "c")], labels=np.array([0])),
        )
    )
    loads: list[tuple[Path, str, bool]] = []

    def fake_load(root: Path, strategy: str, *, verify: bool) -> object:
        loads.append((root, strategy, verify))
        return verified if verify else raw

    split = object()

    def fake_derive(
        train_nodes: object,
        truth_edges: Iterable[tuple[str, str]],
        negatives: object,
        positives: object,
    ) -> object:
        assert train_nodes == verified.split.train_nodes
        assert set(truth_edges) == {("a", "b"), ("b", "c")}
        assert negatives == [("a", "c"), ("b", "c")]
        assert positives == verified.positive_edges
        return split

    expected = nx.Graph([("a", "b")])
    monkeypatch.setattr(seed_topology_probe, "load_benchmark", fake_load)
    monkeypatch.setattr(seed_topology_probe, "derive_val_region_split", fake_derive)
    monkeypatch.setattr(
        seed_topology_probe,
        "truth_graph_for_kd",
        lambda actual: expected if actual is split else pytest.fail("wrong split"),
    )

    graph = seed_topology_probe._load_training_graph(tmp_path, "breadth_first")

    assert graph is expected
    assert loads == [
        (tmp_path / "benchmark_2025_neurips", "breadth_first", True),
        (tmp_path / "benchmark_2025_neurips", "breadth_first", False),
    ]


def test_main_loads_artifact_generated_npz_and_training_graph(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    targets_path = tmp_path / "targets"
    generated_path = tmp_path / "generated.npz"
    output_path = tmp_path / "report.json"
    data_root = tmp_path / "data"
    teacher = np.arange(18, dtype=np.float16).reshape(6, 3)
    generated = np.arange(18, dtype=np.float64).reshape(6, 3)
    np.savez(generated_path, latents=generated)
    targets = SimpleNamespace(
        node_ids=["a", "b", "c"],
        val_pair_a_idx=np.array([0, 1, 0, 2, 1, 0]),
        val_pair_b_idx=np.array([1, 2, 2, 0, 0, 2]),
        val_teacher_rep=teacher,
    )
    graph = nx.complete_graph(["a", "b", "c"])
    loaded_targets: list[Path] = []
    graph_calls: list[tuple[Path, str]] = []
    probe_calls: list[tuple[nx.Graph, list[tuple[str, str]], np.ndarray, np.ndarray]] = []

    def fake_load_targets(path: Path) -> object:
        loaded_targets.append(path)
        return targets

    def fake_load_graph(root: Path, strategy: str) -> nx.Graph:
        graph_calls.append((root, strategy))
        return graph

    report: dict[str, dict[str, float]] = {
        stat: dict.fromkeys(("teacher_r2", "generated_r2", "shuffled_r2"), 0.0)
        for stat in (
            "deg_u",
            "deg_v",
            "common_neighbors",
            "clustering_u",
            "clustering_v",
        )
    }

    def fake_probe(
        actual_graph: nx.Graph,
        pairs: Sequence[tuple[str, str]],
        *,
        teacher: np.ndarray,
        generated: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        probe_calls.append((actual_graph, list(pairs), teacher.copy(), generated.copy()))
        return report

    monkeypatch.setattr(seed_topology_probe, "load_kd_targets", fake_load_targets)
    monkeypatch.setattr(seed_topology_probe, "_load_training_graph", fake_load_graph)
    monkeypatch.setattr(seed_topology_probe, "probe_latents", fake_probe)

    seed_topology_probe.main(
        [
            "--targets",
            str(targets_path),
            "--generated",
            str(generated_path),
            "--graph",
            str(data_root),
            "--strategy",
            "breadth_first",
            "--output",
            str(output_path),
        ]
    )

    assert loaded_targets == [targets_path]
    assert graph_calls == [(data_root, "breadth_first")]
    actual_graph, pairs, actual_teacher, actual_generated = probe_calls[0]
    assert actual_graph is graph
    assert pairs == [("a", "b"), ("b", "c"), ("a", "c"), ("c", "a"), ("b", "a"), ("a", "c")]
    np.testing.assert_array_equal(actual_teacher, teacher.astype(np.float64))
    np.testing.assert_array_equal(actual_generated, generated)
    assert json.loads(output_path.read_text(encoding="utf-8")) == report


def test_main_rejects_missing_generated_latents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    generated_path = tmp_path / "generated.npz"
    np.savez(generated_path, wrong=np.zeros((5, 2)))
    monkeypatch.setattr(
        seed_topology_probe,
        "load_kd_targets",
        lambda _path: pytest.fail("artifact should not load without latents"),
    )

    with pytest.raises(ValueError, match="latents"):
        seed_topology_probe.main(
            [
                "--targets",
                str(tmp_path / "targets"),
                "--generated",
                str(generated_path),
                "--graph",
                str(tmp_path),
                "--output",
                str(tmp_path / "out.json"),
            ]
        )

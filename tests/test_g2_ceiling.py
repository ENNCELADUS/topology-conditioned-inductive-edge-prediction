"""Tests for src.experiments.g2_ceiling: G2 edge-independence ceiling computation."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from src.experiments.g2_ceiling import (
    CAVEAT_EXACT_IDENTITIES,
    CAVEAT_VACUOUS_AT_OV_1,
    G2Result,
    build_dense_probs,
    ceiling_curve,
    compute_delta_star,
    compute_g2_ceiling,
    compute_identities,
    compute_ov_min,
    format_number,
    load_test_graph,
    main,
    render_figure_html,
    sanity_check,
    t_max_ceiling,
    to_json_payload,
    validate_universe,
    write_json,
)
from src.score_universe import load_scores, save_scores


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _write_artifact(
    path: Path,
    *,
    node_ids: list[str],
    pairs: list[tuple[int, int]],
    logits: list[float],
    labels: list[int],
    pairs_source: str = "candidate",
    strategy: str = "breadth_first",
    checkpoint_id: str = "deadbeefcafefeed",
    model_family: str = "v3_1",
) -> None:
    """Write a synthetic scores artifact via the pinned `save_scores` contract."""
    meta = {
        "checkpoint_id": checkpoint_id,
        "model_family": model_family,
        "pairs_source": pairs_source,
        "strategy": strategy,
        "num_rows": len(pairs),
        "created_utc": "2026-07-09T00:00:00+00:00",
        "torch_version": "2.10.0",
    }
    save_scores(
        path,
        node_ids=node_ids,
        u_idx=np.array([p[0] for p in pairs], dtype=np.int32),
        v_idx=np.array([p[1] for p in pairs], dtype=np.int32),
        logit=np.array(logits, dtype=np.float32),
        label=np.array(labels, dtype=np.int8),
        row_start=0,
        meta=meta,
    )


def _synthetic_result(tmp_path: Path, n: int = 4) -> G2Result:
    """Build a `G2Result` from a small synthetic candidate universe + complete graph."""
    node_ids = [f"n{i}" for i in range(n)]
    off_diag = [(i, j) for i in range(n) for j in range(i + 1, n)]
    self_pairs = [(i, i) for i in range(n)]
    pairs = off_diag + self_pairs
    logits = [0.8] * len(off_diag) + [0.0] * len(self_pairs)
    labels = [0] * len(pairs)
    path = tmp_path / "universe.npz"
    _write_artifact(path, node_ids=node_ids, pairs=pairs, logits=logits, labels=labels)
    artifact = load_scores(path)
    test_graph = nx.complete_graph(n)
    return compute_g2_ceiling(artifact, test_graph, strategy="breadth_first")


@pytest.mark.unit
class TestHandComputedFourNode:
    def test_identities_match_hand_computation(self, tmp_path: Path) -> None:
        node_ids = ["n0", "n1", "n2", "n3"]
        pair_logits: dict[tuple[int, int], float] = {
            (0, 1): 0.0,
            (0, 2): 1.0,
            (0, 3): 2.0,
            (1, 2): 0.5,
            (1, 3): 1.5,
            (2, 3): 3.0,
        }
        pairs = list(pair_logits.keys())
        logits = list(pair_logits.values())
        labels = [0] * len(pairs)
        path = tmp_path / "universe.npz"
        _write_artifact(path, node_ids=node_ids, pairs=pairs, logits=logits, labels=labels)

        artifact = load_scores(path)
        p_matrix, n_self, self_mass, n_used = build_dense_probs(artifact)
        assert n_self == 0
        assert self_mass == 0.0
        assert n_used == 6

        volume, overlap, expected_triangles = compute_identities(p_matrix)

        p = {pair: _sigmoid(logit) for pair, logit in pair_logits.items()}

        def edge(a: int, b: int) -> float:
            return p[(a, b)] if a < b else p[(b, a)]

        expected_volume = sum(p.values())
        expected_overlap = sum(v * v for v in p.values()) / expected_volume
        triangles = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        expected_e_triangles = sum(edge(a, b) * edge(b, c) * edge(a, c) for a, b, c in triangles)

        assert volume == pytest.approx(expected_volume, abs=1e-12)
        assert overlap == pytest.approx(expected_overlap, abs=1e-12)
        assert expected_triangles == pytest.approx(expected_e_triangles, abs=1e-12)


@pytest.mark.unit
class TestUniformPCase:
    def test_analytic_identities(self, tmp_path: Path) -> None:
        n = 6
        logit = 1.0
        node_ids = [f"n{i}" for i in range(n)]
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        logits = [logit] * len(pairs)
        labels = [0] * len(pairs)
        path = tmp_path / "universe.npz"
        _write_artifact(path, node_ids=node_ids, pairs=pairs, logits=logits, labels=labels)

        artifact = load_scores(path)
        p_matrix, _, _, _ = build_dense_probs(artifact)
        volume, overlap, expected_triangles = compute_identities(p_matrix)

        p = _sigmoid(logit)
        n_pairs = math.comb(n, 2)
        n_triples = math.comb(n, 3)
        assert volume == pytest.approx(n_pairs * p, abs=1e-9)
        assert overlap == pytest.approx(p, abs=1e-9)
        assert expected_triangles == pytest.approx(n_triples * p**3, abs=1e-9)


@pytest.mark.unit
class TestSelfPairExclusion:
    def test_self_pairs_excluded_and_counted(self, tmp_path: Path) -> None:
        node_ids = ["n0", "n1", "n2"]
        pairs = [(0, 1), (0, 2), (1, 2), (0, 0), (1, 1)]
        logits = [0.0, 0.5, -0.5, 2.0, -2.0]
        labels = [0, 1, 0, 1, 0]
        path = tmp_path / "universe.npz"
        _write_artifact(path, node_ids=node_ids, pairs=pairs, logits=logits, labels=labels)

        artifact = load_scores(path)
        p_matrix, n_self, self_mass, n_used = build_dense_probs(artifact)

        assert n_self == 2
        assert n_used == 3
        expected_mass = _sigmoid(2.0) + _sigmoid(-2.0)
        assert self_mass == pytest.approx(expected_mass, abs=1e-12)
        assert np.all(np.diag(p_matrix) == 0.0)


@pytest.mark.unit
class TestOvMinInvertsCurve:
    @pytest.mark.parametrize("delta_star,volume", [(10, 50.0), (1, 3.0), (100, 500.0)])
    def test_t_max_at_ov_min_equals_delta_star(self, delta_star: int, volume: float) -> None:
        ov_min = compute_ov_min(delta_star, volume)
        bound = float(t_max_ceiling(np.asarray(ov_min, dtype=np.float64), volume))
        assert bound == pytest.approx(delta_star, rel=1e-9)


@pytest.mark.unit
class TestValidateUniverse:
    def test_rejects_wrong_pairs_source(self, tmp_path: Path) -> None:
        node_ids = ["n0"]
        pairs = [(0, 0)]
        path = tmp_path / "universe.npz"
        _write_artifact(
            path,
            node_ids=node_ids,
            pairs=pairs,
            logits=[0.0],
            labels=[0],
            pairs_source="test",
        )
        artifact = load_scores(path)
        with pytest.raises(ValueError, match="pairs_source"):
            validate_universe(artifact, strategy="breadth_first", n_test_nodes=1)

    def test_rejects_wrong_row_count(self, tmp_path: Path) -> None:
        node_ids = ["n0", "n1"]
        pairs = [(0, 1)]
        path = tmp_path / "universe.npz"
        _write_artifact(path, node_ids=node_ids, pairs=pairs, logits=[0.0], labels=[0])
        artifact = load_scores(path)
        with pytest.raises(ValueError, match="row count"):
            validate_universe(artifact, strategy="breadth_first", n_test_nodes=5)

    def test_rejects_wrong_strategy(self, tmp_path: Path) -> None:
        node_ids = ["n0"]
        pairs = [(0, 0)]
        path = tmp_path / "universe.npz"
        _write_artifact(
            path,
            node_ids=node_ids,
            pairs=pairs,
            logits=[0.0],
            labels=[0],
            strategy="other_strategy",
        )
        artifact = load_scores(path)
        with pytest.raises(ValueError, match="strategy"):
            validate_universe(artifact, strategy="breadth_first", n_test_nodes=1)

    def test_rejects_bad_labels(self, tmp_path: Path) -> None:
        node_ids = ["n0", "n1", "n2"]
        pairs = [(0, 1), (0, 2), (1, 2), (0, 0), (1, 1), (2, 2)]
        logits = [0.0] * 6
        labels = [0, 1, 0, 1, 0, -1]
        path = tmp_path / "universe.npz"
        _write_artifact(path, node_ids=node_ids, pairs=pairs, logits=logits, labels=labels)
        artifact = load_scores(path)
        with pytest.raises(ValueError, match="label"):
            validate_universe(artifact, strategy="breadth_first", n_test_nodes=3)

    def test_accepts_valid_universe(self, tmp_path: Path) -> None:
        node_ids = ["n0", "n1", "n2"]
        pairs = [(0, 1), (0, 2), (1, 2), (0, 0), (1, 1), (2, 2)]
        logits = [0.0] * 6
        labels = [0, 1, 0, 1, 0, 1]
        path = tmp_path / "universe.npz"
        _write_artifact(path, node_ids=node_ids, pairs=pairs, logits=logits, labels=labels)
        artifact = load_scores(path)
        validate_universe(artifact, strategy="breadth_first", n_test_nodes=3)


@pytest.mark.unit
class TestSanityCheck:
    def test_passes_for_theoretically_valid_case(self) -> None:
        volume = 10.0 * 0.4
        overlap = 0.4
        expected_triangles = math.comb(5, 3) * 0.4**3
        sanity_check(expected_triangles, overlap, volume)

    def test_raises_on_violation(self) -> None:
        with pytest.raises(AssertionError, match="Chanpuriya bound violated"):
            sanity_check(expected_triangles=1e9, overlap=0.1, volume=1.0)


@pytest.mark.unit
class TestComputeDeltaStar:
    def test_strips_self_loops_before_counting(self) -> None:
        g = nx.Graph()
        g.add_edges_from([(1, 2), (2, 3), (1, 3), (3, 3)])  # triangle + self-loop
        delta_star, n_edges = compute_delta_star(g)
        assert delta_star == 1
        assert n_edges == 3


@pytest.mark.unit
class TestCeilingCurve:
    def test_grid_includes_markers_sorted_deduped(self) -> None:
        omega, curve = ceiling_curve(volume=10.0, overlap=0.37, ov_min=0.6)
        assert omega[0] == 0.0
        assert np.any(np.isclose(omega, 0.37))
        assert np.any(np.isclose(omega, 0.6))
        assert np.all(np.diff(omega) > 0)
        assert curve.shape == omega.shape
        np.testing.assert_allclose(curve, (math.sqrt(2) / 3) * np.power(omega * 10.0, 1.5))

    def test_duplicate_marker_not_duplicated(self) -> None:
        omega, _ = ceiling_curve(volume=10.0, overlap=0.5, ov_min=0.5)
        assert np.count_nonzero(omega == 0.5) == 1

    def test_marker_beyond_one_is_included(self) -> None:
        omega, _ = ceiling_curve(volume=10.0, overlap=0.5, ov_min=1.7)
        assert omega[-1] == pytest.approx(1.7)


@pytest.mark.unit
class TestComputeG2Ceiling:
    def test_full_pipeline_on_synthetic_scenario(self, tmp_path: Path) -> None:
        n = 5
        result = _synthetic_result(tmp_path, n=n)

        assert result.n_nodes == n
        assert result.n_pairs_used == math.comb(n, 2)
        assert result.n_self_pairs == n
        assert result.delta_star == math.comb(n, 3)
        assert result.simple_graph_edges == math.comb(n, 2)
        bound = float(t_max_ceiling(np.asarray(result.overlap), result.volume))
        assert result.expected_triangles <= bound * (1 + 1e-9)
        inverted = float(t_max_ceiling(np.asarray(result.ov_min), result.volume))
        assert inverted == pytest.approx(result.delta_star, rel=1e-9)

    def test_raises_on_invalid_universe(self, tmp_path: Path) -> None:
        node_ids = ["n0"]
        pairs = [(0, 0)]
        path = tmp_path / "universe.npz"
        _write_artifact(
            path,
            node_ids=node_ids,
            pairs=pairs,
            logits=[0.0],
            labels=[0],
            pairs_source="val",
        )
        artifact = load_scores(path)
        test_graph = nx.Graph()
        test_graph.add_node("n0")
        with pytest.raises(ValueError, match="pairs_source"):
            compute_g2_ceiling(artifact, test_graph, strategy="breadth_first")


@pytest.mark.unit
class TestJsonOutput:
    def test_payload_contains_required_fields_and_caveats(self, tmp_path: Path) -> None:
        result = _synthetic_result(tmp_path)
        payload = to_json_payload(result)
        for key in (
            "n_nodes",
            "n_pairs_used",
            "n_self_pairs",
            "self_pair_prob_mass",
            "volume",
            "overlap",
            "expected_triangles",
            "delta_star",
            "simple_graph_edges",
            "ov_min",
            "headroom_triangles",
            "headroom_overlap",
            "curve",
            "artifact_meta",
            "caveats",
        ):
            assert key in payload
        assert CAVEAT_VACUOUS_AT_OV_1 in payload["caveats"]  # type: ignore[operator]
        assert CAVEAT_EXACT_IDENTITIES in payload["caveats"]  # type: ignore[operator]
        assert payload["artifact_meta"] == {
            "checkpoint_id": "deadbeefcafefeed",
            "model_family": "v3_1",
            "strategy": "breadth_first",
        }

    def test_write_json_is_deterministic(self, tmp_path: Path) -> None:
        result = _synthetic_result(tmp_path)
        out1 = tmp_path / "a.json"
        out2 = tmp_path / "b.json"
        write_json(result, out1)
        write_json(result, out2)
        assert out1.read_bytes() == out2.read_bytes()
        payload = json.loads(out1.read_text())
        assert payload["delta_star"] == result.delta_star
        # sort_keys=True: top-level keys must be in ascending order in the raw text.
        raw = out1.read_text()
        first_positions = [raw.index(f'"{k}"') for k in sorted(payload.keys())]
        assert first_positions == sorted(first_positions)


@pytest.mark.unit
class TestRenderFigure:
    def test_figure_contains_svg_markers_numbers_no_external_refs(self, tmp_path: Path) -> None:
        result = _synthetic_result(tmp_path)
        html = render_figure_html(result)

        assert "<svg" in html
        assert 'data-marker="Ov"' in html
        assert 'data-marker="Ov_min"' in html
        assert format_number(result.volume) in html
        assert format_number(result.overlap) in html
        assert format_number(result.ov_min) in html
        assert str(result.delta_star) in html
        assert format_number(result.expected_triangles) in html
        assert CAVEAT_VACUOUS_AT_OV_1 in html
        assert CAVEAT_EXACT_IDENTITIES in html
        assert "http" not in html.lower()
        assert "src=" not in html.lower()


@pytest.mark.unit
class TestLoadTestGraph:
    def test_loads_pickled_graph(self, tmp_path: Path) -> None:
        strategy_dir = tmp_path / "synthetic"
        strategy_dir.mkdir(parents=True)
        g = nx.complete_graph(4)
        with (strategy_dir / "test_graph.pkl").open("wb") as f:
            pickle.dump(g, f)
        loaded = load_test_graph(tmp_path, "synthetic")
        assert set(loaded.nodes()) == set(g.nodes())
        assert loaded.number_of_edges() == g.number_of_edges()

    def test_rejects_non_graph_pickle(self, tmp_path: Path) -> None:
        strategy_dir = tmp_path / "synthetic"
        strategy_dir.mkdir(parents=True)
        with (strategy_dir / "test_graph.pkl").open("wb") as f:
            pickle.dump({"not": "a graph"}, f)
        with pytest.raises(TypeError):
            load_test_graph(tmp_path, "synthetic")


@pytest.mark.unit
class TestCliEndToEnd:
    def test_run_twice_byte_identical(self, tmp_path: Path) -> None:
        n = 5
        strategy = "synthetic"
        data_root = tmp_path / "data"
        benchmark_root = data_root / "benchmark_2025_neurips"
        strategy_dir = benchmark_root / strategy
        strategy_dir.mkdir(parents=True)

        test_graph = nx.complete_graph(n)
        with (strategy_dir / "test_graph.pkl").open("wb") as f:
            pickle.dump(test_graph, f)

        node_ids = [f"n{i}" for i in range(n)]
        off_diag = [(i, j) for i in range(n) for j in range(i + 1, n)]
        self_pairs = [(i, i) for i in range(n)]
        pairs = off_diag + self_pairs
        logits = [0.7] * len(off_diag) + [-0.3] * len(self_pairs)
        labels = [0] * len(pairs)
        universe_path = tmp_path / "universe.npz"
        _write_artifact(
            universe_path,
            node_ids=node_ids,
            pairs=pairs,
            logits=logits,
            labels=labels,
            strategy=strategy,
        )

        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        figure_path = tmp_path / "figures" / "g2-ceiling.html"

        main(
            [
                "--universe",
                str(universe_path),
                "--data-root",
                str(data_root),
                "--strategy",
                strategy,
                "--output-dir",
                str(out1),
                "--figure",
                str(figure_path),
            ]
        )
        main(
            [
                "--universe",
                str(universe_path),
                "--data-root",
                str(data_root),
                "--strategy",
                strategy,
                "--output-dir",
                str(out2),
            ]
        )

        json1 = (out1 / "g2_results.json").read_bytes()
        json2 = (out2 / "g2_results.json").read_bytes()
        assert json1 == json2
        assert figure_path.exists()
        assert "<svg" in figure_path.read_text()


@pytest.mark.integration
class TestRealBenchmarkDeltaStar:
    def test_delta_star_on_real_breadth_first_test_graph(self, benchmark_root: Path) -> None:
        test_graph = load_test_graph(benchmark_root, "breadth_first")
        delta_star, n_edges = compute_delta_star(test_graph)
        assert delta_star > 0
        assert isinstance(delta_star, int)
        assert n_edges == 30_128

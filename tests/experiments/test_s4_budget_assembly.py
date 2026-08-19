"""Tests for src.experiments.s4_budget_assembly: the S4 budget-assembly pipeline."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any, cast

import networkx as nx
import numpy as np
import pytest
from src.eval.graph_metrics import strip_self_loops
from src.experiments import s4_budget_assembly as s4
from src.score_universe import save_scores

pytestmark = pytest.mark.unit

# --------------------------------------------------------------------------- fixtures / builders

# 8-node toy graph with known degree structure (self-loop on n6 included so
# self_loops_ref is nonzero): edges (n1,n2) (n1,n3) (n2,n3) (n4,n5) (n1,n4),
# self-loop (n6,n6). Loopless degrees: n1=3 n2=2 n3=2 n4=2 n5=1 n6=0 n7=0 n8=0.
_NODES = [f"n{i}" for i in range(1, 9)]
_POSITIVE_EDGES = [("n1", "n2"), ("n1", "n3"), ("n2", "n3"), ("n4", "n5"), ("n1", "n4")]
_SELF_LOOP_NODE = "n6"

_BUCKETS: dict[int, list[set[str]]] = {
    4: [
        {"n1", "n2", "n3", "n4"},
        {"n5", "n6", "n7", "n8"},
        {"n1", "n3", "n5", "n7"},
        {"n2", "n4", "n6", "n8"},
    ]
}


def _canonical(u: str, v: str) -> tuple[str, str]:
    return (u, v) if u <= v else (v, u)


def _make_reference_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(_NODES)
    g.add_edges_from(_POSITIVE_EDGES)
    g.add_edge(_SELF_LOOP_NODE, _SELF_LOOP_NODE)
    return g


def _universe_rows(node_ids: list[str]) -> list[tuple[str, str]]:
    """Build the full candidate universe (all canonical pairs + self-pairs)."""
    pairs: list[tuple[str, str]] = []
    for i, u in enumerate(node_ids):
        for v in node_ids[i:]:
            pairs.append((u, v))
    return pairs


def _logit_for_prob(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return float(math.log(p / (1.0 - p)))


def _write_universe_npz(
    path: Path,
    *,
    node_ids: list[str],
    pairs: list[tuple[str, str]],
    probs: list[float],
    labels: np.ndarray,
    strategy: str = "toy",
    checkpoint_id: str = "deadbeefcafefeed",
) -> None:
    position = {node_id: i for i, node_id in enumerate(node_ids)}
    u_idx = np.array([position[u] for u, _ in pairs], dtype=np.int32)
    v_idx = np.array([position[v] for _, v in pairs], dtype=np.int32)
    logits = np.array([_logit_for_prob(p) for p in probs], dtype=np.float32)
    meta: dict[str, object] = {
        "checkpoint_id": checkpoint_id,
        "model_family": "v3_1",
        "pairs_source": "candidate",
        "strategy": strategy,
        "num_rows": len(pairs),
        "created_utc": "2026-08-18T00:00:00+00:00",
        "torch_version": "2.10.0",
    }
    save_scores(
        path,
        node_ids=node_ids,
        u_idx=u_idx,
        v_idx=v_idx,
        logit=logits,
        label=labels.astype(np.int8),
        row_start=0,
        meta=meta,
    )


def _write_benchmark(
    tmp_path: Path, strategy: str, g: nx.Graph, buckets: dict[int, list[set[str]]]
) -> Path:
    data_root = tmp_path / "data"
    strategy_dir = data_root / "benchmark_2025_neurips" / strategy
    strategy_dir.mkdir(parents=True)
    with (strategy_dir / "test_graph.pkl").open("wb") as f:
        pickle.dump(g, f)
    with (strategy_dir / "test_node_buckets.pkl").open("wb") as f:
        pickle.dump(buckets, f)
    return data_root


def _nice_toy_probs(pairs: list[tuple[str, str]]) -> list[float]:
    """Scores that let a hard-quota assembly recover the true edges exactly.

    Every true non-self edge scores 0.9 (the top tie-group, exactly
    `target_edges` = 5 pairs, so `density_matched_threshold` lands exactly on
    0.9); every other non-self pair scores 0.1. Self-pairs: `n6`'s self-pair
    (the reference graph's own self-loop) scores 0.95 (clears 0.9); every
    other self-pair scores 0.05 (does not).
    """
    positive_set = {_canonical(u, v) for u, v in _POSITIVE_EDGES}
    probs: list[float] = []
    for u, v in pairs:
        if u == v:
            probs.append(0.95 if u == _SELF_LOOP_NODE else 0.05)
        elif _canonical(u, v) in positive_set:
            probs.append(0.9)
        else:
            probs.append(0.1)
    return probs


def _toy_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """Write the 8-node toy benchmark + a deterministic candidate universe."""
    g = _make_reference_graph()
    data_root = _write_benchmark(tmp_path, "toy", g, _BUCKETS)
    pairs = _universe_rows(_NODES)
    positive_set = {_canonical(u, v) for u, v in _POSITIVE_EDGES}
    labels = np.array(
        [1 if u != v and _canonical(u, v) in positive_set else 0 for u, v in pairs], dtype=np.int8
    )
    probs = _nice_toy_probs(pairs)
    universe_path = tmp_path / "universe.npz"
    _write_universe_npz(
        universe_path, node_ids=_NODES, pairs=pairs, probs=probs, labels=labels, strategy="toy"
    )
    return universe_path, data_root


def _write_predictions(
    path: Path, degree_predictions: dict[str, float], *, fmt: str = "degree_predictions_v1"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"format": fmt, "degree_predictions": degree_predictions}), encoding="utf-8"
    )


def _d(x: object) -> dict[str, Any]:
    """Cast a JSON-payload value known to be a dict, for concise test assertions."""
    return cast(dict[str, Any], x)


# --------------------------------------------------------------------------- pure helpers


class TestLoadDegreePredictions:
    def test_valid_file_returns_aligned_array(self, tmp_path: Path) -> None:
        path = tmp_path / "pred.json"
        _write_predictions(path, {"b": 2.0, "a": 1.0})
        out = s4.load_degree_predictions(path, ["a", "b"])
        np.testing.assert_allclose(out, [1.0, 2.0])

    def test_wrong_format_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "pred.json"
        _write_predictions(path, {"a": 1.0}, fmt="degree_predictions_v0")
        with pytest.raises(ValueError, match="format"):
            s4.load_degree_predictions(path, ["a"])

    def test_node_set_mismatch_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "pred.json"
        _write_predictions(path, {"a": 1.0, "z": 2.0})
        with pytest.raises(ValueError, match="node set mismatch"):
            s4.load_degree_predictions(path, ["a", "b"])


class TestPredictedHardQuotas:
    def test_sum_equals_two_target_edges(self) -> None:
        raw = np.array([0.0, 5.0, 2.0, -1.0])
        quotas = s4.predicted_hard_quotas(raw, target_edges=7)
        assert int(quotas.sum()) == 14

    def test_clips_to_minimum_before_scaling(self) -> None:
        # A zero (and a negative) raw prediction must not silently vanish --
        # it is clipped to _MIN_PREDICTED_DEGREE before scaling, so it still
        # receives a (small) positive share of the quota budget.
        raw = np.array([0.0, 10.0])
        quotas = s4.predicted_hard_quotas(raw, target_edges=5)
        assert int(quotas.sum()) == 10
        assert quotas[0] >= 0


class TestOracleHardQuotas:
    def test_matches_true_loopless_degrees(self) -> None:
        g = _make_reference_graph()
        g_simple = strip_self_loops(g)
        quotas = s4.oracle_hard_quotas(g_simple, _NODES)
        assert quotas == {"n1": 3, "n2": 2, "n3": 2, "n4": 2, "n5": 1, "n6": 0, "n7": 0, "n8": 0}

    def test_missing_node_gets_zero(self) -> None:
        g_simple = nx.Graph()
        g_simple.add_edge("a", "b")
        quotas = s4.oracle_hard_quotas(g_simple, ["a", "b", "z"])
        assert quotas["z"] == 0


# --------------------------------------------------------------------------- pipeline: nice toy


class TestRunS4PipelineNiceToy:
    def test_oracle_hard_satisfies_quotas_exactly(self, tmp_path: Path) -> None:
        universe_path, data_root = _toy_inputs(tmp_path)
        payload = s4.run_s4_pipeline(
            universe_path=universe_path,
            data_root=data_root,
            strategy="toy",
            output_dir=tmp_path / "out",
            seed=0,
        )
        quota_info = _d(_d(payload["quota_info"])["oracle_hard"])
        stats = _d(quota_info["stats"])
        assert stats["shortfall"] == 0
        assert stats["realized_edges"] == stats["target_edges"] == 5
        error = _d(quota_info["quota_error"])
        assert error["exact_fraction"] == pytest.approx(1.0)
        assert quota_info["lower_bound_only"] is False

    def test_self_pairs_never_reach_assemble_degree_quota(self, tmp_path: Path) -> None:
        # If a self-pair leaked into assemble_degree_quota it would raise
        # ("non-self"); every arm succeeding, with a self-loop on n6 and none
        # elsewhere, is the observable proof self-pairs were routed around it.
        universe_path, data_root = _toy_inputs(tmp_path)
        payload = s4.run_s4_pipeline(
            universe_path=universe_path,
            data_root=data_root,
            strategy="toy",
            output_dir=tmp_path / "out",
            seed=0,
        )
        assembled = _d(payload["assembled"])
        for arm in ("b0_exact_n", "oracle_hard"):
            row = _d(assembled[arm])
            assert row["self_loops_pred"] == 1
            assert row["self_loops_ref"] == 1

    def test_five_topology_numbers_and_global_simple_edge_named_separately(
        self, tmp_path: Path
    ) -> None:
        universe_path, data_root = _toy_inputs(tmp_path)
        payload = s4.run_s4_pipeline(
            universe_path=universe_path,
            data_root=data_root,
            strategy="toy",
            output_dir=tmp_path / "out",
            seed=0,
        )
        row = _d(_d(payload["assembled"])["b0_exact_n"])
        gs = _d(row["graph_similarity"])
        rd = _d(row["relative_density"])
        assert set(gs) == {"global_simple_edge", "bfs_macro"}
        assert set(rd) == {"global_simple_edge", "bfs_macro"}
        assert set(cast(dict[str, float], row["mmd_ratio"])) == {"degree", "clustering", "spectral"}
        assert 0.0 <= row["edge_precision"] <= 1.0
        assert 0.0 <= row["edge_recall"] <= 1.0

    def test_predicted_hard_arm_and_gap_closure(self, tmp_path: Path) -> None:
        universe_path, data_root = _toy_inputs(tmp_path)
        pred_path = tmp_path / "predictions" / "gnn_v1.json"
        _write_predictions(
            pred_path,
            {
                "n1": 2.5,
                "n2": 2.0,
                "n3": 1.5,
                "n4": 2.5,
                "n5": 1.0,
                "n6": 0.2,
                "n7": 0.2,
                "n8": 0.2,
            },
        )
        payload = s4.run_s4_pipeline(
            universe_path=universe_path,
            data_root=data_root,
            strategy="toy",
            output_dir=tmp_path / "out",
            seed=0,
            degree_predictions_paths=[pred_path],
        )
        arm = "predicted_hard_gnn_v1"
        assembled = _d(payload["assembled"])
        quota_info = _d(payload["quota_info"])
        gap_closure = _d(payload["gap_closure"])
        assert arm in assembled
        assert arm in quota_info
        assert arm in gap_closure
        stats = _d(_d(quota_info[arm])["stats"])
        assert int(stats["target_edges"]) * 2 == 10  # sum(quotas) == 2 * target_edges (5)
        assert set(gap_closure[arm]) == {
            "degree",
            "clustering",
            "spectral",
            "graph_similarity",
            "relative_density",
        }
        # b0_exact_n and oracle_hard are the run's own control/oracle, not
        # closure targets of themselves.
        assert "b0_exact_n" not in gap_closure
        assert "oracle_hard" not in gap_closure

    def test_predictions_node_set_mismatch_raises(self, tmp_path: Path) -> None:
        universe_path, data_root = _toy_inputs(tmp_path)
        pred_path = tmp_path / "bad_predictions.json"
        bad = dict.fromkeys(_NODES[:-1], 1.0)
        bad["not_a_node"] = 1.0
        _write_predictions(pred_path, bad)
        with pytest.raises(ValueError, match="node set mismatch"):
            s4.run_s4_pipeline(
                universe_path=universe_path,
                data_root=data_root,
                strategy="toy",
                output_dir=tmp_path / "out",
                seed=0,
                degree_predictions_paths=[pred_path],
            )

    def test_duplicate_variant_raises(self, tmp_path: Path) -> None:
        universe_path, data_root = _toy_inputs(tmp_path)
        predictions = dict.fromkeys(_NODES, 1.0)
        path_a = tmp_path / "a" / "gnn.json"
        path_b = tmp_path / "b" / "gnn.json"
        _write_predictions(path_a, predictions)
        _write_predictions(path_b, predictions)
        with pytest.raises(ValueError, match="duplicate"):
            s4.run_s4_pipeline(
                universe_path=universe_path,
                data_root=data_root,
                strategy="toy",
                output_dir=tmp_path / "out",
                seed=0,
                degree_predictions_paths=[path_a, path_b],
            )

    def test_byte_identical_reruns(self, tmp_path: Path) -> None:
        universe_path, data_root = _toy_inputs(tmp_path)
        out_a = tmp_path / "out_a"
        out_b = tmp_path / "out_b"
        s4.run_s4_pipeline(
            universe_path=universe_path,
            data_root=data_root,
            strategy="toy",
            output_dir=out_a,
            seed=0,
        )
        s4.run_s4_pipeline(
            universe_path=universe_path,
            data_root=data_root,
            strategy="toy",
            output_dir=out_b,
            seed=0,
        )
        assert (out_a / "s4_results.json").read_bytes() == (out_b / "s4_results.json").read_bytes()

    def test_cli_end_to_end(self, tmp_path: Path) -> None:
        universe_path, data_root = _toy_inputs(tmp_path)
        out_dir = tmp_path / "cli_out"
        s4.main(
            [
                "--universe",
                str(universe_path),
                "--data-root",
                str(data_root),
                "--strategy",
                "toy",
                "--output-dir",
                str(out_dir),
                "--seed",
                "0",
            ]
        )
        payload = json.loads((out_dir / "s4_results.json").read_text(encoding="utf-8"))
        assert set(_d(payload["assembled"])) == {"b0_exact_n", "oracle_hard"}

    def test_missing_universe_errors(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            s4.main(
                [
                    "--universe",
                    str(tmp_path / "missing.npz"),
                    "--output-dir",
                    str(tmp_path / "out"),
                ]
            )


# --------------------------------------------------------------------------- pipeline: stranding


class TestGreedyStrandingShortfall:
    def _stranding_inputs(self, tmp_path: Path) -> tuple[Path, Path]:
        """4-node universe engineered so oracle_hard's greedy pass strands quota.

        Reference edges A-B, A-C give true loopless degrees A=2, B=1, C=1,
        D=0 (target_edges=2). Scores rank the non-edge pair B-C above both
        true edges, so the greedy pass accepts B-C first, exhausting B and C
        before A ever finds a partner: A's quota of 2 goes entirely
        unconsumed (residual_quota=2, realized_edges=1, shortfall=1 -> 50% of
        target_edges, well over the 2% lower-bound-only threshold).
        """
        nodes = ["A", "B", "C", "D"]
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from([("A", "B"), ("A", "C")])
        buckets: dict[int, list[set[str]]] = {2: [{"A", "B"}, {"C", "D"}]}
        data_root = _write_benchmark(tmp_path, "toy", g, buckets)

        pairs = _universe_rows(nodes)
        score_by_pair = {
            ("B", "C"): 0.9,
            ("A", "B"): 0.5,
            ("A", "C"): 0.4,
        }
        probs = [
            0.05 if u == v else score_by_pair.get((u, v), score_by_pair.get((v, u), 0.1))
            for u, v in pairs
        ]
        labels = np.zeros(len(pairs), dtype=np.int8)
        universe_path = tmp_path / "universe.npz"
        _write_universe_npz(
            universe_path, node_ids=nodes, pairs=pairs, probs=probs, labels=labels, strategy="toy"
        )
        return universe_path, data_root

    def test_shortfall_and_lower_bound_only_flag(self, tmp_path: Path) -> None:
        universe_path, data_root = self._stranding_inputs(tmp_path)
        payload = s4.run_s4_pipeline(
            universe_path=universe_path,
            data_root=data_root,
            strategy="toy",
            output_dir=tmp_path / "out",
            seed=0,
        )
        quota_info = _d(_d(payload["quota_info"])["oracle_hard"])
        stats = _d(quota_info["stats"])
        assert stats["target_edges"] == 2
        assert stats["realized_edges"] == 1
        assert stats["shortfall"] == 1
        assert stats["residual_quota"] == 2
        assert quota_info["shortfall_fraction"] == pytest.approx(0.5)
        assert quota_info["lower_bound_only"] is True

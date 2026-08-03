"""Tests for src.experiments.probes: closed-form ridge representation probes."""

from __future__ import annotations

import inspect
import json
import pickle
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import pytest
import torch
from src.experiments import probes

pytestmark = pytest.mark.unit


def _linearly_predictable(n: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build states with a target that is an exact linear function of them plus tiny noise."""
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(n, d))
    weights = rng.normal(size=d)
    targets = states @ weights + rng.normal(scale=0.01, size=n)
    return states, targets


class TestLinearProbeR2:
    def test_near_one_on_linearly_predictable_target(self) -> None:
        states, targets = _linearly_predictable(300, 5, seed=0)
        r2 = probes.linear_probe_r2(states, targets)
        assert r2 == pytest.approx(1.0, abs=0.02)

    def test_near_zero_on_shuffled_target(self) -> None:
        states, targets = _linearly_predictable(300, 5, seed=0)
        rng = np.random.default_rng(1)
        shuffled = rng.permutation(targets)
        r2 = probes.linear_probe_r2(states, shuffled)
        assert abs(r2) < 0.2

    def test_deterministic_given_fixed_seed(self) -> None:
        states, targets = _linearly_predictable(120, 4, seed=2)
        first = probes.linear_probe_r2(states, targets)
        second = probes.linear_probe_r2(states, targets)
        assert first == second

    def test_accepts_one_dimensional_states(self) -> None:
        rng = np.random.default_rng(3)
        states = rng.normal(size=200)
        targets = states * 3.0 + rng.normal(scale=0.01, size=200)
        r2 = probes.linear_probe_r2(states, targets)
        assert r2 == pytest.approx(1.0, abs=0.02)

    def test_requires_at_least_n_folds_samples(self) -> None:
        states = np.zeros((2, 3))
        targets = np.zeros(2)
        with pytest.raises(ValueError, match="at least"):
            probes.linear_probe_r2(states, targets)


class TestDegreePartialledR2:
    def test_near_zero_when_target_is_degree(self) -> None:
        rng = np.random.default_rng(4)
        n = 200
        degrees = rng.uniform(1.0, 50.0, size=n)
        states = rng.normal(size=(n, 4))
        r2 = probes.degree_partialled_r2(states, degrees, degrees)
        assert r2 == pytest.approx(0.0, abs=1e-6)

    def test_recovers_signal_beyond_degree(self) -> None:
        rng = np.random.default_rng(5)
        n = 300
        degrees = rng.uniform(1.0, 50.0, size=n)
        states = rng.normal(size=(n, 3))
        residual_signal = states[:, 0] * 2.0
        targets = degrees * 0.5 + residual_signal + rng.normal(scale=0.01, size=n)
        r2 = probes.degree_partialled_r2(states, targets, degrees)
        assert r2 == pytest.approx(1.0, abs=0.05)

    def test_zero_when_states_are_pure_degree_confound(self) -> None:
        # states carry ONLY the degree signal (up to noise): once both sides
        # are partialled against degree, nothing predictive should remain.
        rng = np.random.default_rng(6)
        n = 250
        degrees = rng.uniform(1.0, 50.0, size=n)
        states = (degrees * 2.0 + rng.normal(scale=0.01, size=n)).reshape(-1, 1)
        targets = degrees * 0.3 + rng.normal(scale=1.0, size=n)
        r2 = probes.degree_partialled_r2(states, targets, degrees)
        assert abs(r2) < 0.2

    def test_nuisance_residuals_are_fit_within_each_fold(self) -> None:
        """Held-out targets must not influence the degree nuisance fit."""
        rng = np.random.default_rng(266)
        n = 10
        degrees = rng.uniform(-3.0, 3.0, size=n)
        states = np.column_stack([degrees**2, rng.normal(size=n)])
        targets = 0.4 * degrees + 0.7 * states[:, 0] + rng.normal(scale=0.2, size=n)

        order = np.random.default_rng(0).permutation(n)
        folds = [fold.astype(np.int64) for fold in np.array_split(order, 5)]
        predictions = np.empty(n)
        residual_targets = np.empty(n)
        for fold_idx, test_idx in enumerate(folds):
            train_idx = np.concatenate([folds[j] for j in range(5) if j != fold_idx])
            train_design = np.column_stack([np.ones(len(train_idx)), degrees[train_idx]])
            test_design = np.column_stack([np.ones(len(test_idx)), degrees[test_idx]])
            state_coefficients, *_ = np.linalg.lstsq(train_design, states[train_idx], rcond=None)
            target_coefficients, *_ = np.linalg.lstsq(train_design, targets[train_idx], rcond=None)
            train_states = states[train_idx] - train_design @ state_coefficients
            test_states = states[test_idx] - test_design @ state_coefficients
            train_targets = targets[train_idx] - train_design @ target_coefficients
            test_targets = targets[test_idx] - test_design @ target_coefficients
            mean_states = train_states.mean(axis=0)
            mean_target = train_targets.mean()
            centered_states = train_states - mean_states
            weights = np.linalg.solve(
                centered_states.T @ centered_states + 1e-3 * np.eye(states.shape[1]),
                centered_states.T @ (train_targets - mean_target),
            )
            predictions[test_idx] = (test_states - mean_states) @ weights + mean_target
            residual_targets[test_idx] = test_targets

        expected = 1.0 - float(np.sum((residual_targets - predictions) ** 2)) / float(
            np.sum((residual_targets - residual_targets.mean()) ** 2)
        )
        assert probes.degree_partialled_r2(states, targets, degrees) == pytest.approx(expected)


class TestProbeTargets:
    """Hand-verified spec Sec 13.6 targets on a triangle-plus-pendant graph."""

    @staticmethod
    def _graph() -> nx.Graph:
        # Triangle a-b-c plus pendant d attached to a.
        g = nx.Graph()
        g.add_edges_from([("a", "b"), ("a", "c"), ("b", "c"), ("a", "d")])
        return g

    def test_hand_computed_values(self) -> None:
        targets = probes.probe_targets(self._graph(), ["a", "b", "d"])
        # Degrees: a=3 (b, c, d), b=2, d=1.
        np.testing.assert_array_equal(targets["degree"], [3.0, 2.0, 1.0])
        # Clustering (networkx convention): a has 1 of C(3,2)=3 neighbor
        # pairs connected; b's single neighbor pair (a, c) is connected; d
        # has < 2 neighbors -> 0.
        np.testing.assert_allclose(targets["clustering"], [1 / 3, 1.0, 0.0])
        # ego(a) is the whole graph (4 edges); ego(b) is the triangle
        # (3 edges); ego(d) is the single edge a-d.
        np.testing.assert_array_equal(targets["ego_edges"], [4.0, 3.0, 1.0])
        # nx.density: 4 edges / C(4,2)=6; 3/3; 1/1.
        np.testing.assert_allclose(targets["ego_density"], [4 / 6, 1.0, 1.0])

    def test_row_alignment_follows_input_order(self) -> None:
        forward = probes.probe_targets(self._graph(), ["a", "d"])
        reversed_order = probes.probe_targets(self._graph(), ["d", "a"])
        np.testing.assert_array_equal(forward["degree"], reversed_order["degree"][::-1])

    def test_missing_node_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing"):
            probes.probe_targets(self._graph(), ["a", "zzz"])


class TestPiConsistency:
    def test_v2_measures_mass_on_same_identity_teacher_cells(self) -> None:
        same_identity = torch.zeros(1, 3, 3, dtype=torch.bool)
        same_identity[0, 0, 1] = True
        same_identity[0, 2, 2] = True
        concentrated = torch.zeros(1, 3, 3)
        concentrated[0, 0, 1] = 0.4
        concentrated[0, 2, 2] = 0.6
        uniform = torch.full((1, 3, 3), 1.0 / 9.0)

        torch.testing.assert_close(
            probes.pi_alignment_consistency_v2(concentrated, same_identity),
            torch.ones(1),
        )
        torch.testing.assert_close(
            probes.pi_alignment_consistency_v2(uniform, same_identity),
            torch.tensor([2.0 / 9.0]),
        )

    def test_v2_stays_nonzero_when_shut_gates_force_v1_to_zero(self) -> None:
        plan = torch.full((1, 2, 2), 0.25)
        same_identity = torch.tensor([[[True, False], [False, False]]])
        shut_gates = torch.zeros(1, 2)

        v1 = probes.pi_grounding_chain_consistency_v1(
            plan,
            same_identity,
            gate_a=shut_gates,
            gate_b=shut_gates,
        )
        v2 = probes.pi_alignment_consistency_v2(plan, same_identity)

        torch.testing.assert_close(v1, torch.zeros(1))
        torch.testing.assert_close(v2, torch.tensor([0.25]))


def test_slot_recall_at_n_ground_matches_hand_built_pool() -> None:
    graph = nx.Graph([("a", "b"), ("a", "c"), ("b", "c"), ("c", "d")])
    pool = {
        "a": ["b", "x"],  # 1 / 2
        "b": ["a", "c"],  # 2 / 2
        "c": ["a", "x"],  # 1 / 3
        "d": ["x", "y"],  # 0 / 1
    }
    selected = {
        "a": ["b"],
        "b": ["a", "c"],
        "c": ["a"],
        "d": [],
    }
    rows = probes.slot_recall_at_n_ground(
        graph,
        ["a", "b", "c", "d"],
        pool,
        selected,
    )
    np.testing.assert_allclose(rows, [0.5, 1.0, 1.0 / 3.0, 0.0])
    assert float(np.mean(rows)) == pytest.approx(11.0 / 24.0)


def test_probe_scope_is_required_and_v_select_is_not_exposed() -> None:
    signature = inspect.signature(probes.produce_e2e_probe_artifact)
    assert signature.parameters["scope"].default is inspect.Parameter.empty
    assert "v_select" not in probes.E2E_PROBE_SCOPES


@pytest.mark.parametrize("retired_scope", ["calibration_fit", "qualification_qual"])
def test_retired_prebinding_probe_scopes_are_rejected_everywhere(retired_scope: str) -> None:
    """The two-stage cleanup collapses the scope domain to `formal_train` alone."""
    assert probes.E2E_PROBE_SCOPES == ("formal_train",)
    assert retired_scope not in probes.E2E_PROBE_SCOPES
    with pytest.raises(ValueError, match=f"unsupported E2E probe scope '{retired_scope}'"):
        probes.build_probe_scope_context(
            cast(probes.E2EProbeScope, retired_scope),
            data_root=Path("unused"),
            strategy="breadth_first",
            partition_seed=0,
            msg_fraction=0.8,
            expected_missing_features=(),
        )


def test_build_probe_scope_context_rebuilds_the_full_train_side_universe(
    tmp_path: Path,
) -> None:
    from src.train_egostitch import _BENCHMARK_SUBDIR

    strategy_dir = tmp_path / _BENCHMARK_SUBDIR / "breadth_first"
    strategy_dir.mkdir(parents=True)
    with (strategy_dir / "split.pkl").open("wb") as handle:
        pickle.dump({"train": ["a", "b", "c", "d"], "test": ["sealed-test"]}, handle)
    (strategy_dir / "train_edges.txt").write_text(
        "a\tb\t1\nb\tc\t1\nc\td\t1\na\tc\t0\n",
        encoding="utf-8",
    )

    nodes, graph = probes.build_probe_scope_context(
        "formal_train",
        data_root=tmp_path,
        strategy="breadth_first",
        partition_seed=0,
        msg_fraction=1.0,
        expected_missing_features=(),
    )

    # Every operative train node, and only the message-side positives.
    assert nodes == ["a", "b", "c", "d"]
    assert set(graph.nodes) == {"a", "b", "c", "d"}
    assert {frozenset(edge) for edge in graph.edges} == {
        frozenset({"a", "b"}),
        frozenset({"b", "c"}),
        frozenset({"c", "d"}),
    }


def test_train_side_probe_loader_never_requires_test_artifacts(tmp_path: Path) -> None:
    strategy_dir = tmp_path / "breadth_first"
    strategy_dir.mkdir()
    with (strategy_dir / "split.pkl").open("wb") as handle:
        pickle.dump({"train": ["a", "b", "c"], "test": ["sealed-test"]}, handle)
    (strategy_dir / "train_edges.txt").write_text(
        "a\tb\t1\nb\tc\t0\na\tc\t1\n",
        encoding="utf-8",
    )

    nodes, positives = probes._load_train_side_probe_inputs(
        strategy_dir,
        expected_missing_features=(),
    )
    assert nodes == ["a", "b", "c"]
    assert positives == [("a", "b"), ("a", "c")]
    assert not (strategy_dir / "test_graph.pkl").exists()
    assert not (strategy_dir / "test_edges.txt").exists()


class TestE2EProbeArtifact:
    @staticmethod
    def _write(
        tmp_path: Path, *, n_ground: int = 2
    ) -> tuple[Path, nx.Graph, list[str], dict[str, object]]:
        graph = nx.cycle_graph(12)
        graph = nx.relabel_nodes(graph, {node: f"n{node:02d}" for node in graph})
        nodes = sorted(graph.nodes())
        targets = probes.probe_targets(graph, nodes)
        rng = np.random.default_rng(9)
        states = np.column_stack(
            [targets["degree"], targets["ego_density"], targets["clustering"], rng.normal(size=12)]
        ).astype(np.float32)
        pairs = probes.select_probe_pairs(graph)
        metadata: dict[str, object] = {
            "checkpoint_id": "abc123",
            "registration_sha256": "a" * 64,
            "config_hash": "b" * 64,
            "seed": 0,
            "partition_seed": 0,
            "strategy": "toy",
            "g_struct_sha256": probes.g_struct_sha256(graph),
            "scope": "formal_train",
            "n_ground": n_ground,
        }
        path = tmp_path / "probe.npz"
        n_pairs = len(pairs)
        probes.write_e2e_probe_artifact(
            path,
            metadata=metadata,
            node_ids=nodes,
            states=states,
            targets={name: targets[name] for name in ("degree", "ego_density", "clustering")},
            pair_ids=pairs,
            pair_states=rng.normal(size=(n_pairs, 4)).astype(np.float32),
            pi_consistency_v1=np.linspace(0.0, 1.0, n_pairs, dtype=np.float64),
            pi_consistency_v2=np.linspace(1.0, 0.0, n_pairs, dtype=np.float64),
            slot_recall=np.linspace(0.0, 1.0, len(nodes), dtype=np.float64),
            shared_neighbor_count=np.asarray(
                [
                    len(set(graph.neighbors(node_u)) & set(graph.neighbors(node_v)))
                    for node_u, node_v in pairs
                ],
                dtype=np.float64,
            ),
            dispersion={
                "pi_slot_std": np.full(n_pairs, 0.1),
                "h_pairwise_cosine_mean": np.full(n_pairs, 0.2),
                "adj_offdiag_std": np.full(n_pairs, 0.3),
                "plan_row_entropy": np.full(n_pairs, 0.4),
            },
        )
        return path, graph, nodes, metadata

    def test_consumer_reports_all_registered_targets(self, tmp_path: Path) -> None:
        path, graph, nodes, metadata = self._write(tmp_path)
        report = probes.evaluate_e2e_probe_artifact(
            path, graph=graph, train_nodes=nodes, expected_metadata=metadata
        )
        assert set(cast(dict[str, float], report["linear_probe_r2"])) == {
            "degree",
            "ego_density",
            "clustering",
        }
        assert set(cast(dict[str, float], report["degree_partialled_r2"])) == {
            "ego_density",
            "clustering",
        }
        pi = cast(dict[str, float | int], report["pi_shared_neighbor_consistency"])
        assert pi["n_pairs"] == len(probes.select_probe_pairs(graph))
        assert 0.0 <= pi["nonzero_fraction"] <= 1.0
        pi_v2 = cast(dict[str, float | int], report["pi_consistency_v2"])
        assert pi_v2["n_pairs"] == len(probes.select_probe_pairs(graph))
        assert isinstance(report["shared_neighbor_count_r2"], float)
        slot_recall = cast(dict[str, float | int], report["slot_recall_at_n_ground"])
        assert slot_recall["n_ground"] == 2
        assert slot_recall["n_nodes"] == len(nodes)
        assert set(cast(dict[str, object], report["dispersion"])) == {
            "pi_slot_std",
            "h_pairwise_cosine_mean",
            "adj_offdiag_std",
            "plan_row_entropy",
        }

    def test_consumer_rejects_target_drift(self, tmp_path: Path) -> None:
        path, graph, nodes, metadata = self._write(tmp_path)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["degree"] = arrays["degree"].copy()
        arrays["degree"][0] += 1.0
        arrays["meta"] = np.array(json.dumps(json.loads(str(arrays["meta"].item()))))
        np.savez_compressed(path, **arrays)
        with pytest.raises(ValueError, match="target 'degree'"):
            probes.evaluate_e2e_probe_artifact(
                path, graph=graph, train_nodes=nodes, expected_metadata=metadata
            )

    def test_consumer_rejects_n_ground_different_from_registered_arm(
        self, tmp_path: Path
    ) -> None:
        path, graph, nodes, metadata = self._write(tmp_path)
        registered = {**metadata, "n_ground": 50}

        with pytest.raises(ValueError, match=r"probe=2, registered=50"):
            probes.evaluate_e2e_probe_artifact(
                path,
                graph=graph,
                train_nodes=nodes,
                expected_metadata=registered,
            )

    def test_consumer_accepts_narrower_registered_n_ground(self, tmp_path: Path) -> None:
        path, graph, nodes, metadata = self._write(tmp_path, n_ground=20)

        report = probes.evaluate_e2e_probe_artifact(
            path,
            graph=graph,
            train_nodes=nodes,
            expected_metadata=metadata,
        )

        slot_recall = cast(dict[str, float | int], report["slot_recall_at_n_ground"])
        assert slot_recall["n_ground"] == 20

    def test_consumer_rejects_v1_artifact_with_clear_version_error(
        self, tmp_path: Path
    ) -> None:
        path, graph, nodes, metadata = self._write(tmp_path)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        payload = json.loads(str(arrays["meta"].item()))
        payload["format"] = "egostitch_e2e_probe_v1"
        arrays["meta"] = np.array(json.dumps(payload, sort_keys=True))
        np.savez_compressed(path, **arrays)

        with pytest.raises(
            ValueError,
            match="egostitch_e2e_probe_v1.*not supported.*egostitch_e2e_probe_v2",
        ):
            probes.evaluate_e2e_probe_artifact(
                path,
                graph=graph,
                train_nodes=nodes,
                expected_metadata=metadata,
            )

    def test_pair_selection_is_hash_deterministic(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from([("b", "a"), ("c", "a"), ("d", "a")])
        assert probes.select_probe_pairs(graph, limit=2) == probes.select_probe_pairs(
            graph.copy(), limit=2
        )

    @pytest.mark.parametrize("retired_scope", ["calibration_fit", "qualification_qual"])
    def test_producer_rejects_the_retired_prebinding_scopes(
        self, tmp_path: Path, retired_scope: str
    ) -> None:
        """The scope guard fires before any file is read, so nothing else matters."""
        with pytest.raises(ValueError, match=f"unsupported E2E probe scope '{retired_scope}'"):
            probes.produce_e2e_probe_artifact(
                checkpoint_path=tmp_path / "best.pt",
                run_metadata_path=tmp_path / "run_metadata.json",
                data_root=tmp_path / "data",
                strategy="breadth_first",
                output_path=tmp_path / "probe.npz",
                scope=cast(probes.E2EProbeScope, retired_scope),
            )

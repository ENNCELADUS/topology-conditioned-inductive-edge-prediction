"""Tests for src.experiments.probes: closed-form ridge representation probes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import pytest
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


class TestE2EProbeArtifact:
    @staticmethod
    def _write(tmp_path: Path) -> tuple[Path, nx.Graph, list[str], dict[str, object]]:
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
        }
        path = tmp_path / "probe.npz"
        probes.write_e2e_probe_artifact(
            path,
            metadata=metadata,
            node_ids=nodes,
            states=states,
            targets={name: targets[name] for name in ("degree", "ego_density", "clustering")},
            pair_ids=pairs,
            pi_consistency=np.linspace(0.0, 1.0, len(pairs), dtype=np.float64),
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

    def test_consumer_rejects_provenance_or_target_drift(self, tmp_path: Path) -> None:
        path, graph, nodes, metadata = self._write(tmp_path)
        wrong = {**metadata, "checkpoint_id": "wrong"}
        with pytest.raises(ValueError, match="checkpoint_id"):
            probes.evaluate_e2e_probe_artifact(
                path, graph=graph, train_nodes=nodes, expected_metadata=wrong
            )

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

    def test_pair_selection_is_hash_deterministic(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from([("b", "a"), ("c", "a"), ("d", "a")])
        assert probes.select_probe_pairs(graph, limit=2) == probes.select_probe_pairs(
            graph.copy(), limit=2
        )

    def test_producer_rejects_draft_registration(self, tmp_path: Path) -> None:
        registration = tmp_path / "registration.json"
        registration.write_text(
            json.dumps(
                {
                    "status": "DRAFT",
                    "probe_artifact": {
                        "format": "egostitch_e2e_probe_v1",
                        "source_arm": "full",
                        "expected_path": str(tmp_path / "probe.npz"),
                    },
                    "arms": {"full": {"training": str(tmp_path / "full.yaml")}},
                }
            ),
            encoding="utf-8",
        )
        metadata = tmp_path / "run_metadata.json"
        metadata.write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError, match="BINDING preregistration"):
            probes.produce_e2e_probe_artifact(
                checkpoint_path=tmp_path / "best.pt",
                run_metadata_path=metadata,
                preregistration_path=registration,
                data_root=tmp_path / "data",
                strategy="breadth_first",
                output_path=tmp_path / "probe.npz",
            )

    def test_producer_rejects_nonformal_or_incomplete_source(self, tmp_path: Path) -> None:
        output = tmp_path / "probe.npz"
        registration = tmp_path / "registration.json"
        registration.write_text(
            json.dumps(
                {
                    "status": "BINDING",
                    "probe_artifact": {
                        "format": "egostitch_e2e_probe_v1",
                        "source_arm": "full",
                        "expected_path": str(output),
                    },
                    "arms": {"full": {"training": str(tmp_path / "full.yaml")}},
                }
            ),
            encoding="utf-8",
        )
        registration_sha = hashlib.sha256(registration.read_bytes()).hexdigest()
        metadata = tmp_path / "run_metadata.json"
        metadata.write_text(
            json.dumps(
                {
                    "preregistration_sha256": registration_sha,
                    "run_kind": "formal",
                    "status": "running",
                    "formal_artifacts_published": False,
                    "permanent_null": "none",
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="completed formal full arm"):
            probes.produce_e2e_probe_artifact(
                checkpoint_path=tmp_path / "best.pt",
                run_metadata_path=metadata,
                preregistration_path=registration,
                data_root=tmp_path / "data",
                strategy="breadth_first",
                output_path=output,
            )

    def test_producer_rejects_completed_p0_source(self, tmp_path: Path) -> None:
        output = tmp_path / "probe.npz"
        full_config = tmp_path / "full.yaml"
        registration = tmp_path / "registration.json"
        registration.write_text(
            json.dumps(
                {
                    "status": "BINDING",
                    "probe_artifact": {
                        "format": "egostitch_e2e_probe_v1",
                        "source_arm": "full",
                        "expected_path": str(output),
                    },
                    "arms": {"full": {"training": str(full_config)}},
                }
            ),
            encoding="utf-8",
        )
        registration_sha = hashlib.sha256(registration.read_bytes()).hexdigest()
        metadata = tmp_path / "run_metadata.json"
        metadata.write_text(
            json.dumps(
                {
                    "preregistration_sha256": registration_sha,
                    "run_kind": "formal",
                    "status": "complete",
                    "formal_artifacts_published": True,
                    "permanent_null": "none",
                    "seed": 0,
                    "partition_seed": 0,
                    "config_path": str(tmp_path / "p0.yaml"),
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="registered full-arm config path"):
            probes.produce_e2e_probe_artifact(
                checkpoint_path=tmp_path / "best.pt",
                run_metadata_path=metadata,
                preregistration_path=registration,
                data_root=tmp_path / "data",
                strategy="breadth_first",
                output_path=output,
            )

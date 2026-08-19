"""Tests for the S5 full-capacity node-degree probe."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch
from src.data.features import FeatureStore
from src.experiments.s5_degree_probe import (
    DEGREE_PREDICTIONS_FORMAT,
    DegreeProbe,
    derive_degree_targets,
    encoder_arch_from_checkpoint,
    holdout_split,
    regression_readout,
    run_s5_pipeline,
    token_budget_chunks,
    warm_start_encoder,
)

INPUT_DIM = 16
TOKEN_LEN = 5


# --------------------------------------------------------------------------- fixtures / helpers


def _write_feature_root(tmp_path: Path, node_ids: list[str]) -> Path:
    """Build a tiny synthetic FeatureStore root: metadata.json + index.json + per-node .pt."""
    root = tmp_path / "features"
    embeddings_dir = root / "embeddings"
    embeddings_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    index: dict[str, str] = {}
    for node_id in node_ids:
        tensor = torch.tensor(rng.standard_normal((TOKEN_LEN, INPUT_DIM)), dtype=torch.float32)
        rel_path = f"embeddings/{node_id}.pt"
        torch.save(tensor, root / rel_path)
        index[node_id] = rel_path
    (root / "index.json").write_text(json.dumps(index))
    (root / "metadata.json").write_text(
        json.dumps(
            {"format": "torch_pt_per_node", "input_dim": INPUT_DIM, "max_sequence_length": 1024}
        )
    )
    return root


def _tiny_arch() -> dict[str, object]:
    return {
        "input_dim": INPUT_DIM,
        "d_model": 8,
        "n_layers": 1,
        "n_heads": 2,
        "dropout": 0.0,
        "token_dropout": 0.0,
        "stochastic_depth": 0.0,
    }


def _tiny_model_config() -> dict[str, object]:
    return {
        "input_dim": INPUT_DIM,
        "d_model": 8,
        "encoder_layers": 1,
        "n_heads": 2,
        "regularization": {"dropout": 0.0, "token_dropout": 0.0, "stochastic_depth": 0.0},
    }


# --------------------------------------------------------------------------- target derivation


class TestDeriveDegreeTargets:
    """The log1p / loopless / store-intersection target convention."""

    def test_log1p_of_loopless_degree(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from([("a", "b"), ("a", "c"), ("b", "c")])
        nodes, y = derive_degree_targets(graph, frozenset({"a", "b", "c"}))
        assert nodes == ["a", "b", "c"]
        np.testing.assert_allclose(y, np.log1p([2.0, 2.0, 2.0]))

    def test_self_loops_do_not_count_toward_degree(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from([("a", "b"), ("a", "a")])
        _, y = derive_degree_targets(graph, frozenset({"a", "b"}))
        # 'a' has one real neighbour; the self-loop must not inflate it.
        np.testing.assert_allclose(y, np.log1p([1.0, 1.0]))

    def test_store_intersection_drops_featureless_nodes(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from([("a", "b"), ("b", "featureless")])
        nodes, y = derive_degree_targets(graph, frozenset({"a", "b"}))
        assert nodes == ["a", "b"]
        # 'b' keeps its true degree of 2 -- the node is dropped, not the edge.
        np.testing.assert_allclose(y, np.log1p([1.0, 2.0]))

    def test_isolated_node_maps_to_zero(self) -> None:
        graph = nx.Graph()
        graph.add_node("lonely")
        _, y = derive_degree_targets(graph, frozenset({"lonely"}))
        np.testing.assert_allclose(y, [0.0])


# --------------------------------------------------------------------------- split


class TestHoldoutSplit:
    """Seeded permutation split determinism and disjointness."""

    def test_deterministic_for_a_seed(self) -> None:
        a_train, a_held = holdout_split(100, seed=3)
        b_train, b_held = holdout_split(100, seed=3)
        np.testing.assert_array_equal(a_train, b_train)
        np.testing.assert_array_equal(a_held, b_held)

    def test_different_seeds_differ(self) -> None:
        _, held_a = holdout_split(100, seed=0)
        _, held_b = holdout_split(100, seed=1)
        assert not np.array_equal(held_a, held_b)

    def test_partition_is_disjoint_and_complete(self) -> None:
        train, held = holdout_split(100, seed=0)
        assert set(train.tolist()) | set(held.tolist()) == set(range(100))
        assert not set(train.tolist()) & set(held.tolist())
        assert held.size == 10

    def test_empty_side_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            holdout_split(5, seed=0)


# --------------------------------------------------------------------------- chunking


class TestTokenBudgetChunks:
    """Length-sorted chunking under a padded-token budget."""

    def test_every_index_appears_exactly_once(self) -> None:
        lengths = [7, 3, 11, 5, 2, 9]
        chunks = token_budget_chunks(lengths, budget=20)
        flat = sorted(i for chunk in chunks for i in chunk)
        assert flat == list(range(len(lengths)))

    def test_respects_padded_budget(self) -> None:
        lengths = [4] * 10
        chunks = token_budget_chunks(lengths, budget=12)
        for chunk in chunks:
            assert len(chunk) * max(lengths[i] for i in chunk) <= 12

    def test_oversized_item_forms_its_own_chunk(self) -> None:
        chunks = token_budget_chunks([100, 1, 1], budget=4)
        assert [0] in chunks
        assert sorted(i for chunk in chunks for i in chunk) == [0, 1, 2]

    def test_chunks_are_length_sorted(self) -> None:
        lengths = [9, 1, 5, 3]
        chunks = token_budget_chunks(lengths, budget=6)
        visited = [i for chunk in chunks for i in chunk]
        assert [lengths[i] for i in visited] == sorted(lengths)

    def test_non_positive_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="budget must be positive"):
            token_budget_chunks([1, 2], budget=0)


# --------------------------------------------------------------------------- warm start


class TestWarmStart:
    """Encoder-prefix surgery from a v3_1-shaped checkpoint."""

    def test_strict_load_transfers_every_encoder_tensor(self) -> None:
        source = DegreeProbe(**_tiny_arch())  # type: ignore[arg-type]
        target = DegreeProbe(**_tiny_arch())  # type: ignore[arg-type]
        model_state = {f"encoder.{k}": v for k, v in source.encoder.state_dict().items()}
        model_state["classifier.head.weight"] = torch.zeros(2, 2)  # non-encoder key, ignored

        transferred = warm_start_encoder(target, model_state)

        assert transferred == len(source.encoder.state_dict())
        for key, value in source.encoder.state_dict().items():
            torch.testing.assert_close(target.encoder.state_dict()[key], value)

    def test_missing_encoder_keys_raises(self) -> None:
        probe = DegreeProbe(**_tiny_arch())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="no 'encoder.'-prefixed keys"):
            warm_start_encoder(probe, {"classifier.weight": torch.zeros(1)})

    def test_shape_mismatch_is_not_silently_partial(self) -> None:
        probe = DegreeProbe(**_tiny_arch())  # type: ignore[arg-type]
        wide = dict(_tiny_arch())
        wide["d_model"] = 16
        other = DegreeProbe(**wide)  # type: ignore[arg-type]
        model_state = {f"encoder.{k}": v for k, v in other.encoder.state_dict().items()}
        with pytest.raises(RuntimeError):
            warm_start_encoder(probe, model_state)

    def test_arch_is_read_from_checkpoint_not_hardcoded(self) -> None:
        # The shipped b0 config says d_model 512 while real checkpoints carry 256;
        # the probe must follow the checkpoint.
        arch = encoder_arch_from_checkpoint(_tiny_model_config())
        assert arch["d_model"] == 8
        assert arch["n_layers"] == 1
        assert arch["input_dim"] == INPUT_DIM


# --------------------------------------------------------------------------- readout


class TestRegressionReadout:
    """Spearman / R^2 / MAE conventions."""

    def test_perfect_prediction(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0])
        out = regression_readout(y, y)
        assert out["spearman"] == pytest.approx(1.0)
        assert out["r2"] == pytest.approx(1.0)
        assert out["mae"] == pytest.approx(0.0)

    def test_matches_scipy_spearman(self) -> None:
        from scipy.stats import spearmanr

        rng = np.random.default_rng(0)
        y_true = rng.standard_normal(50)
        y_pred = rng.standard_normal(50)
        out = regression_readout(y_true, y_pred)
        assert out["spearman"] == pytest.approx(float(spearmanr(y_true, y_pred).statistic))

    def test_constant_prediction_is_finite(self) -> None:
        out = regression_readout(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0]))
        assert np.isfinite(out["spearman"])
        assert np.isfinite(out["r2"])

    def test_zero_variance_target_gives_zero_r2(self) -> None:
        out = regression_readout(np.array([2.0, 2.0]), np.array([1.0, 3.0]))
        assert out["r2"] == 0.0


# --------------------------------------------------------------------------- end to end


class TestRunPipeline:
    """CPU smoke run: artifact schema, S4 contract, early stop."""

    @staticmethod
    def _setup(tmp_path: Path) -> tuple[Path, Path, Path, list[str]]:
        from src.experiments.g1_hardened_e2 import _BENCHMARK_SUBDIR, _FEATURES_SUBDIR
        from src.score_universe import save_scores

        train_nodes = [f"n{i:03d}" for i in range(24)]
        test_nodes = [f"t{i:02d}" for i in range(6)]
        data_root = tmp_path / "data"

        feature_root = _write_feature_root(tmp_path, train_nodes + test_nodes)
        target_features = data_root / _FEATURES_SUBDIR
        target_features.parent.mkdir(parents=True, exist_ok=True)
        feature_root.rename(target_features)

        graph = nx.Graph()
        graph.add_nodes_from(train_nodes)
        rng = np.random.default_rng(0)
        for i, node in enumerate(train_nodes):
            for j in range(i + 1, len(train_nodes)):
                if rng.random() < 0.25:
                    graph.add_edge(node, train_nodes[j])
        graph.add_edge(train_nodes[0], train_nodes[0])  # self-loop, must be stripped

        strategy_dir = data_root / _BENCHMARK_SUBDIR / "breadth_first"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        import pickle

        with (strategy_dir / "train_graph.pkl").open("wb") as f:
            pickle.dump(graph, f)

        # Candidate universe over the test nodes (all non-self pairs plus self-pairs).
        u_idx, v_idx = [], []
        for i in range(len(test_nodes)):
            for j in range(i, len(test_nodes)):
                u_idx.append(i)
                v_idx.append(j)
        n_rows = len(u_idx)
        universe_path = tmp_path / "candidate.npz"
        save_scores(
            universe_path,
            node_ids=test_nodes,
            u_idx=np.asarray(u_idx, dtype=np.int32),
            v_idx=np.asarray(v_idx, dtype=np.int32),
            logit=np.zeros(n_rows, dtype=np.float32),
            label=np.full(n_rows, -1, dtype=np.int8),
            row_start=0,
            meta={
                "checkpoint_id": "deadbeefcafefeed",
                "model_family": "v3_1",
                "pairs_source": "candidate",
                "strategy": "breadth_first",
                "num_rows": n_rows,
                "created_utc": "2026-08-18T00:00:00+00:00",
                "torch_version": "2.10.0",
            },
        )

        checkpoint_path = tmp_path / "best.pt"
        probe = DegreeProbe(**_tiny_arch())  # type: ignore[arg-type]
        torch.save(
            {
                "model_family": "v3_1",
                "model_config": _tiny_model_config(),
                "model_state": {f"encoder.{k}": v for k, v in probe.encoder.state_dict().items()},
            },
            checkpoint_path,
        )
        return data_root, universe_path, checkpoint_path, test_nodes

    def test_smoke_run_writes_s4_consumable_predictions(self, tmp_path: Path) -> None:
        data_root, universe_path, checkpoint_path, test_nodes = self._setup(tmp_path)
        out_dir = tmp_path / "out"

        report = run_s5_pipeline(
            universe_path=universe_path,
            checkpoint_path=checkpoint_path,
            data_root=data_root,
            strategy="breadth_first",
            output_dir=out_dir,
            init="warm",
            seed=0,
            device="cpu",
            max_epochs=2,
            patience=1,
            token_budget=256,
        )

        payload = json.loads((out_dir / "predictions.json").read_text())
        assert payload["format"] == DEGREE_PREDICTIONS_FORMAT
        assert set(payload["degree_predictions"]) == set(test_nodes)
        assert all(v >= 0.0 for v in payload["degree_predictions"].values())
        assert payload["variant"] == "warm_s0"
        assert report["evidence_class"] == "diagnostic"
        assert (out_dir / "report.json").exists()

    def test_predictions_load_through_the_s4_contract(self, tmp_path: Path) -> None:
        from src.experiments.s4_budget_assembly import load_degree_predictions

        data_root, universe_path, checkpoint_path, test_nodes = self._setup(tmp_path)
        out_dir = tmp_path / "out"
        run_s5_pipeline(
            universe_path=universe_path,
            checkpoint_path=checkpoint_path,
            data_root=data_root,
            strategy="breadth_first",
            output_dir=out_dir,
            init="scratch",
            seed=1,
            device="cpu",
            max_epochs=1,
            patience=1,
            token_budget=256,
        )
        loaded = load_degree_predictions(out_dir / "predictions.json", test_nodes)
        assert loaded.shape == (len(test_nodes),)
        assert np.all(loaded >= 0.0)

    def test_report_records_convention_and_caveat(self, tmp_path: Path) -> None:
        data_root, universe_path, checkpoint_path, _ = self._setup(tmp_path)
        out_dir = tmp_path / "out"
        report = run_s5_pipeline(
            universe_path=universe_path,
            checkpoint_path=checkpoint_path,
            data_root=data_root,
            strategy="breadth_first",
            output_dir=out_dir,
            init="scratch",
            seed=0,
            device="cpu",
            max_epochs=1,
            patience=1,
            token_budget=256,
        )
        assert "log1p" in str(report["target_convention"])
        assert "coupled" in str(report["leakage_caveat"])
        assert report["split_manifest"]["n_heldout"] >= 1  # type: ignore[index]
        assert "generalization_gap_spearman" in report["selection"]  # type: ignore[operator]

    def test_unknown_init_raises(self, tmp_path: Path) -> None:
        data_root, universe_path, checkpoint_path, _ = self._setup(tmp_path)
        with pytest.raises(ValueError, match="init must be"):
            run_s5_pipeline(
                universe_path=universe_path,
                checkpoint_path=checkpoint_path,
                data_root=data_root,
                strategy="breadth_first",
                output_dir=tmp_path / "out",
                init="lukewarm",
                seed=0,
                device="cpu",
            )


def test_probe_forward_shape() -> None:
    """The probe maps a padded node batch to one scalar per node."""
    probe = DegreeProbe(**_tiny_arch())  # type: ignore[arg-type]
    tokens = torch.zeros(3, TOKEN_LEN, INPUT_DIM)
    lengths = torch.tensor([TOKEN_LEN, TOKEN_LEN - 1, TOKEN_LEN])
    assert probe(tokens, lengths).shape == (3,)


def test_feature_store_round_trip(tmp_path: Path) -> None:
    """The synthetic store fixture satisfies the real FeatureStore contract."""
    root = _write_feature_root(tmp_path, ["a", "b"])
    store = FeatureStore(root)
    assert store.input_dim == INPUT_DIM
    assert store.node_ids == frozenset({"a", "b"})
    assert store.load_tokens("a").shape == (TOKEN_LEN, INPUT_DIM)

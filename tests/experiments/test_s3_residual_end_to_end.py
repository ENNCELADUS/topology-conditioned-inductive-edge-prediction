"""End-to-end smoke tests: S3's cache -> train pipeline, at the library level and via the CLI.

Two test classes, sharing the same checkpoint/FeatureStore fixture helpers
(mirroring `test_s3_residual_data.py`'s pattern -- duplicated here rather than
imported, per house convention):

- `TestS3CacheToTrain` drives the library entry points `cli.py`'s `cache`/
  `train` stages call directly (`build_s3_corpus`, `build_vval_eval`,
  `build_vval_features`, `train_arm`) against a small 60-node synthetic graph
  with an explicit `vval_nodes` override -- the same override Task 1's own
  tests use for graphs too small for the canonical V_val derivation.
- `TestS3CliEndToEnd` drives `cli.main` itself (argv parsing + stage dispatch,
  including the `--sizes`/`--vval-sizes` overrides) against a *larger*
  ~1200-node graph. The `--sizes`/`--vval-sizes` CLI flags only control
  `build_s3_corpus`/`build_vval_eval`'s own region/bucket sampling; the
  canonical V_val node-set derivation the `cache` stage always runs first
  (`derive_vval_nodes`, no CLI override) internally calls
  `sample_bfs_ball_buckets` with `ValRegionParams()`'s own fixed
  `bucket_sizes` (up to 200 nodes), so the graph must be large enough for
  that regardless of `--sizes`/`--vval-sizes` -- a ~1200-node graph reliably
  derives a several-hundred-node V_val region (verified empirically: this
  derivation itself takes well under a second, it is pure graph traversal).
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
from src import score_universe
from src.data.features import FeatureStore
from src.experiments.s3_set_residual import cli as s3_cli
from src.experiments.s3_set_residual import data as s3_data
from src.experiments.s3_set_residual import train as train_mod
from src.experiments.s3_set_residual.model import ResidualConfig

pytestmark = pytest.mark.unit

INPUT_DIM = 8
TOKEN_LEN = 5
CPU = torch.device("cpu")


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


def _tiny_v3_1_config() -> dict[str, object]:
    return {
        "input_dim": INPUT_DIM,
        "d_model": 8,
        "encoder_layers": 1,
        "cross_attn_layers": 1,
        "n_heads": 2,
        "mlp_head": {"hidden_dims": [8], "dropout": 0.0, "activation": "gelu", "norm": "layernorm"},
        "regularization": {"dropout": 0.0},
    }


def _write_checkpoint(
    path: Path, *, model: torch.nn.Module, model_family: str, model_config: dict[str, object]
) -> None:
    payload: dict[str, object] = {
        "model_state": model.state_dict(),
        "model_family": model_family,
        "model_config": model_config,
        "epoch": 0,
        "val_metrics": {},
        "seed": 0,
        "config": {},
    }
    torch.save(payload, path)


def _build_checkpoint_and_store(tmp_path: Path, node_ids: list[str]) -> tuple[Path, FeatureStore]:
    """Build a tiny real `v3_1` checkpoint plus a synthetic `FeatureStore` covering `node_ids`."""
    store = FeatureStore(_write_feature_root(tmp_path, node_ids))
    torch.manual_seed(0)
    model = score_universe.build_model("v3_1", _tiny_v3_1_config())
    checkpoint_path = tmp_path / "ckpt.pt"
    _write_checkpoint(
        checkpoint_path, model=model, model_family="v3_1", model_config=_tiny_v3_1_config()
    )
    return checkpoint_path, store


def _make_universe_graph() -> tuple[nx.Graph, frozenset[str]]:
    """A 60-node graph: a 40-node train pool, a 20-node connected V_val pool, bridge edges.

    Returns:
        `(graph, vval_nodes)`. `vval_nodes` is the 20-node pool -- its induced,
        self-loop-stripped subgraph is itself a connected Watts-Strogatz graph,
        so `build_vval_eval`'s giant-component BFS-ball sampling has plenty of
        room at sizes 6/8.
    """
    pool_a = nx.connected_watts_strogatz_graph(40, k=4, p=0.3, seed=1)
    pool_b = nx.connected_watts_strogatz_graph(20, k=4, p=0.3, seed=2)
    graph = nx.Graph()
    graph.add_nodes_from(f"node_{i:06d}" for i in range(60))
    graph.add_edges_from((f"node_{u:06d}", f"node_{v:06d}") for u, v in pool_a.edges())
    graph.add_edges_from((f"node_{u + 40:06d}", f"node_{v + 40:06d}") for u, v in pool_b.edges())
    rng = np.random.default_rng(3)
    for _ in range(6):
        u = f"node_{int(rng.integers(0, 40)):06d}"
        v = f"node_{int(rng.integers(40, 60)):06d}"
        graph.add_edge(u, v)
    vval_nodes = frozenset(f"node_{i:06d}" for i in range(40, 60))
    return graph, vval_nodes


def _finite(value: object) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


# --------------------------------------------------------------------------- test


class TestS3CacheToTrain:
    def test_cache_then_train_produces_finite_artifacts_with_exact_zero_init_epoch(
        self, tmp_path: Path
    ) -> None:
        graph, vval_nodes = _make_universe_graph()
        node_ids = sorted(graph.nodes())
        checkpoint_path, store = _build_checkpoint_and_store(tmp_path, node_ids)
        cache_dir = tmp_path / "cache"

        # -- cache stage (library level; see module docstring for why not cli.main) --
        corpus = s3_data.build_s3_corpus(
            graph,
            store,
            checkpoint=checkpoint_path,
            sizes=(6, 8),
            per_size=3,
            neg_ratio=2,
            salt="s3-e2e-train|",
            cache_dir=cache_dir,
            device=CPU,
            vval_nodes=vval_nodes,
        )
        s3_data.save_s3_corpus(corpus, cache_dir / "corpus.pt")

        vval = s3_data.build_vval_eval(
            graph,
            store,
            checkpoint=checkpoint_path,
            sizes=(6, 8),
            per_size=2,
            salt="s3-e2e-vval|",
            cache_dir=cache_dir,
            device=CPU,
            vval_nodes=vval_nodes,
        )
        s3_data.save_vval_eval(vval, cache_dir / "vval.pkl")

        vval_features = train_mod.build_vval_features(
            vval, store, cache_path=cache_dir / "f0_vval.pt"
        )
        train_mod.save_vval_features(vval_features, cache_dir / "vval_features.pt")

        assert (cache_dir / "corpus.pt").exists()
        assert (cache_dir / "vval.pkl").exists()
        assert (cache_dir / "vval_features.pt").exists()

        # -- train stage: reload from the cache dir exactly as cli._stage_train would --
        loaded_corpus = s3_data.load_s3_corpus(cache_dir / "corpus.pt")
        loaded_vval = s3_data.load_vval_eval(cache_dir / "vval.pkl")
        loaded_features = train_mod.load_vval_features(cache_dir / "vval_features.pt")

        model_cfg = ResidualConfig(
            mode="res",
            d_in=INPUT_DIM,
            d_model=16,
            sab_layers=1,
            heads=2,
            pma_seeds=2,
            p_dim=6,
            head_hidden=12,
        )
        train_cfg = train_mod.TrainConfig(
            epochs=2, batch_regions=3, patience=10, seed=0, device="cpu", top_k_checkpoints=2
        )
        run_dir = tmp_path / "run"
        train_mod.train_arm(
            loaded_corpus, loaded_vval, loaded_features, model_cfg, train_cfg, run_dir=run_dir
        )

        # -- artifacts exist --
        assert (run_dir / "last.pt").exists()
        assert (run_dir / "best.pt").exists()
        metadata_path = run_dir / "run_metadata.json"
        assert metadata_path.exists()
        metrics_path = run_dir / "metrics.jsonl"
        assert metrics_path.exists()

        # -- zero-init property, indirectly: epoch 0's pre-training ΔAUPRC is exact 0.0 --
        lines = metrics_path.read_text().strip().splitlines()
        assert len(lines) == train_cfg.epochs + 1
        epoch0 = json.loads(lines[0])
        assert epoch0["epoch"] == 0
        assert epoch0["train_loss"] is None
        assert epoch0["vval_delta_auprc"] == 0.0
        assert epoch0["delta_variance"] == 0.0
        assert epoch0["margin_share"] == 0.0
        assert epoch0["n_sets_used"] > 0
        assert epoch0["n_sets_skipped_single_class"] >= 0

        # -- every subsequent record is well-formed and finite --
        for line in lines[1:]:
            record = json.loads(line)
            assert record["epoch"] >= 1
            assert _finite(record["train_loss"])
            assert _finite(record["vval_delta_auprc"])
            assert _finite(record["delta_variance"])
            assert _finite(record["margin_share"])
            assert record["n_sets_used"] > 0
            assert record["n_sets_skipped_single_class"] >= 0

        # -- run_metadata.json is well-formed and finite --
        metadata = json.loads(metadata_path.read_text())
        assert metadata["arm"] == "res"
        assert metadata["seed"] == 0
        assert metadata["checkpoint_id"] == vval.checkpoint_id
        assert 1 <= metadata["published_epoch"] <= train_cfg.epochs
        table = metadata["selection_table"]
        assert 1 <= len(table) <= train_cfg.top_k_checkpoints
        for row in table:
            for key in (
                "delta_auprc",
                "gs",
                "rd",
                "abs_rd_minus_1",
                "degree_mmd_ratio",
                "clustering_mmd_ratio",
                "spectral_mmd_ratio",
                "mean_rank",
            ):
                assert _finite(row[key]), f"{key} not finite: {row[key]!r}"

        # -- best.pt loads and forwards without error --
        best_model = train_mod._load_model(run_dir / "best.pt", model_cfg, CPU)
        x = torch.randn(1, 6, INPUT_DIM)
        mask = torch.ones(1, 6)
        pairs = torch.tensor([[0, 0, 1], [0, 2, 3]], dtype=torch.long)
        out = best_model(x, mask, pairs)
        assert out.shape == (2,)
        assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- CLI-driven (finding 3)

_STRATEGY = "breadth_first"


def _write_cli_feature_root(data_root: Path, node_ids: list[str]) -> None:
    """Write a synthetic FeatureStore root at the exact path `cli._stage_cache` reads."""
    root = data_root / "features" / "frozen_node_features_1024"
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


def _write_cli_checkpoint(path: Path) -> None:
    torch.manual_seed(0)
    model = score_universe.build_model("v3_1", _tiny_v3_1_config())
    _write_checkpoint(path, model=model, model_family="v3_1", model_config=_tiny_v3_1_config())


def _write_train_graph(data_root: Path, strategy: str, graph: nx.Graph) -> None:
    strategy_dir = data_root / "benchmark_2025_neurips" / strategy
    strategy_dir.mkdir(parents=True)
    with (strategy_dir / "train_graph.pkl").open("wb") as f:
        pickle.dump(graph, f)


def _make_large_graph(n: int = 1200) -> nx.Graph:
    """A graph large enough for the canonical V_val derivation to succeed.

    Its own fixed `ValRegionParams().bucket_sizes` (up to 200 nodes) needs a
    V_val region of at least 200 nodes; empirically a 1200-node Watts-Strogatz
    graph derives one several hundred nodes wide (see module docstring),
    comfortably clearing that floor.
    """
    base = nx.connected_watts_strogatz_graph(n, k=4, p=0.3, seed=0)
    return nx.relabel_nodes(base, {i: f"node_{i:06d}" for i in base.nodes()})


class TestS3CliEndToEnd:
    """Drives `cli.main` itself: argv parsing, `--sizes`/`--vval-sizes`, stage dispatch."""

    def test_cache_then_train_via_cli_main(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        graph = _make_large_graph()
        node_ids = sorted(graph.nodes())
        _write_train_graph(data_root, _STRATEGY, graph)
        _write_cli_feature_root(data_root, node_ids)
        checkpoint_path = tmp_path / "ckpt.pt"
        _write_cli_checkpoint(checkpoint_path)

        cache_dir = tmp_path / "cache"
        exit_code = s3_cli.main(
            [
                "--stage",
                "cache",
                "--data-root",
                str(data_root),
                "--strategy",
                _STRATEGY,
                "--checkpoint",
                str(checkpoint_path),
                "--output-dir",
                str(cache_dir),
                "--sizes",
                "6,8",
                "--vval-sizes",
                "6,8",
                "--regions-per-size",
                "3",
                "--vval-per-size",
                "2",
                "--neg-ratio",
                "2",
                "--device",
                "cpu",
            ]
        )
        assert exit_code == 0
        assert (cache_dir / "corpus.pt").exists()
        assert (cache_dir / "vval.pkl").exists()
        assert (cache_dir / "vval_features.pt").exists()

        run_dir = tmp_path / "run"
        exit_code = s3_cli.main(
            [
                "--stage",
                "train",
                "--arm",
                "res",
                "--seed",
                "0",
                "--cache-dir",
                str(cache_dir),
                "--run-dir",
                str(run_dir),
                "--device",
                "cpu",
                "--epochs",
                "2",
                "--batch-regions",
                "3",
                "--d-in",
                str(INPUT_DIM),
                "--d-model",
                "16",
                "--sab-layers",
                "1",
                "--heads",
                "2",
                "--pma-seeds",
                "2",
                "--p-dim",
                "6",
                "--head-hidden",
                "12",
            ]
        )
        assert exit_code == 0

        assert (run_dir / "best.pt").exists()
        assert (run_dir / "last.pt").exists()
        metadata = json.loads((run_dir / "run_metadata.json").read_text())
        assert metadata["arm"] == "res"
        assert 1 <= metadata["published_epoch"] <= 2

        lines = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3  # epoch 0, 1, 2
        epoch0 = json.loads(lines[0])
        assert epoch0["epoch"] == 0
        assert epoch0["vval_delta_auprc"] == 0.0
        for line in lines:
            record = json.loads(line)
            assert _finite(record["n_sets_used"])
            assert _finite(record["n_sets_skipped_single_class"])


class TestS3CliEvalWiring:
    """Pins the `--stage eval` argv-parsing + delegation contract (findings 4-6).

    Does not exercise `run_eval`'s body -- Task 4 owns that. `evaluate.py`'s
    `run_eval` is monkeypatched to a capturing stub so this test only proves:
    (a) `cli.main(["--stage", "eval", ...])` parses without the duplicate
    `--run-dir`/missing-shared-flags errors the review caught, and (b) the
    namespace `run_eval` receives carries every field it reads
    (`data_root`, `strategy`, `device`, `seed`, `run_dir`, `b0_universe`).
    """

    def test_eval_stage_invokes_run_eval_with_complete_namespace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.experiments.s3_set_residual import evaluate as s3_evaluate

        captured: dict[str, object] = {}

        def _fake_run_eval(args: object) -> Path:
            captured["data_root"] = args.data_root  # type: ignore[attr-defined]
            captured["strategy"] = args.strategy  # type: ignore[attr-defined]
            captured["device"] = args.device  # type: ignore[attr-defined]
            captured["seed"] = args.seed  # type: ignore[attr-defined]
            captured["run_dir"] = args.run_dir  # type: ignore[attr-defined]
            captured["b0_universe"] = args.b0_universe  # type: ignore[attr-defined]
            return tmp_path / "report.json"

        monkeypatch.setattr(s3_evaluate, "run_eval", _fake_run_eval)

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        b0_path = tmp_path / "b0.npz"
        b0_path.write_bytes(b"")  # never read -- run_eval is stubbed above

        exit_code = s3_cli.main(
            [
                "--stage",
                "eval",
                "--data-root",
                str(tmp_path / "data"),
                "--strategy",
                "breadth_first",
                "--device",
                "cpu",
                "--seed",
                "3",
                "--run-dir",
                str(run_dir),
                "--b0-universe",
                str(b0_path),
            ]
        )

        assert exit_code == 0
        assert captured["data_root"] == tmp_path / "data"
        assert captured["strategy"] == "breadth_first"
        assert captured["device"] == "cpu"
        assert captured["seed"] == 3
        assert captured["run_dir"] == run_dir
        assert captured["b0_universe"] == b0_path

    def test_aggregate_only_invocation_needs_no_run_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--run-dir` must stay optional at the CLI level (finding 6)."""
        from src.experiments.s3_set_residual import evaluate as s3_evaluate

        captured: dict[str, object] = {}

        def _fake_run_eval(args: object) -> Path:
            captured["run_dir"] = args.run_dir  # type: ignore[attr-defined]
            captured["aggregate"] = args.aggregate  # type: ignore[attr-defined]
            return tmp_path / "pooled_report.json"

        monkeypatch.setattr(s3_evaluate, "run_eval", _fake_run_eval)

        report_a = tmp_path / "a.json"
        report_a.write_text("{}")

        exit_code = s3_cli.main(["--stage", "eval", "--aggregate", str(report_a)])

        assert exit_code == 0
        assert captured["run_dir"] is None
        assert captured["aggregate"] == [report_a]

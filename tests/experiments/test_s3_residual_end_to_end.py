"""End-to-end smoke test: S3's cache -> train pipeline on a fully synthetic universe.

Drives the same library entry points `cli.py`'s `cache`/`train` stages call
(`build_s3_corpus`, `build_vval_eval`, `build_vval_features`, `train_arm`) against
a tiny synthetic graph, a synthetic `FeatureStore`, and a tiny real `v3_1`
checkpoint (`score_universe.build_model`, mirroring `test_s3_residual_data.py`'s
fixture pattern -- duplicated here rather than imported, per house convention).

Deliberately bypasses `cli.main`'s argv parsing for the cache step: the CLI's
`cache` stage has no `--sizes`/`--vval-sizes` override (by design -- production
runs use S2's real `S2_SIZES`, up to 200-node buckets), which a 60-node test
graph cannot satisfy. `build_s3_corpus`/`build_vval_eval` take `sizes`/
`vval_nodes` overrides directly (the same override Task 1's own tests use for
small graphs), so this test calls them directly; `train_arm` is exercised
exactly as `cli._stage_train` calls it, after a save/load round trip through
the same cache files `cli._stage_cache` would have written.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch
from src import score_universe
from src.data.features import FeatureStore
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

        # -- every subsequent record is well-formed and finite --
        for line in lines[1:]:
            record = json.loads(line)
            assert record["epoch"] >= 1
            assert _finite(record["train_loss"])
            assert _finite(record["vval_delta_auprc"])
            assert _finite(record["delta_variance"])
            assert _finite(record["margin_share"])

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

"""Tests for `src/experiments/s2_latent_topology/train.py`'s four S2 trainers.

CPU only, tiny dims throughout (see `_tiny_cfg`), against a hand-built 16-node
synthetic `RegionCorpus` (6 size-6 + 4 size-8... in fact 4+4 regions of sizes
6/8, 2 held out for validation) so no real benchmark artifacts are needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from src.experiments.s2_latent_topology.data import RegionCorpus
from src.experiments.s2_latent_topology.model import (
    DetTwin,
    MargDegreeRegressor,
    RegionAutoencoder,
    SetDenoiser,
)
from src.experiments.s2_latent_topology.train import (
    TrainConfig,
    append_jsonl,
    load_checkpoint,
    train_ae,
    train_det,
    train_marg,
    train_prior,
)

pytestmark = pytest.mark.unit

_FEATURE_DIM = 8
_AE_METRIC_KEYS = {
    "stage",
    "epoch",
    "train_total",
    "val_total",
    "lr",
    "train_bce",
    "train_triangle",
    "val_bce",
    "val_triangle",
    "edge_density_liveness",
}
_CHECKPOINT_KEYS = {"model", "config", "in_dim", "best_epoch", "val_loss"}


def _cycle_edges(n: int) -> list[tuple[int, int]]:
    """A simple `n`-cycle's edges as sorted `(i, j)`, `i < j`, local index pairs."""
    edges = [(i, i + 1) for i in range(n - 1)]
    edges.append((0, n - 1))
    return edges


def _make_corpus() -> RegionCorpus:
    """A tiny 16-node synthetic corpus: 4 size-6 + 4 size-8 ring regions, 2 held out."""
    node_ids = [f"node_{i:06d}" for i in range(16)]
    rng = np.random.default_rng(0)
    features = torch.tensor(rng.standard_normal((16, _FEATURE_DIM)), dtype=torch.float32)

    size6 = [(0, 1, 2, 3, 4, 5), (2, 3, 4, 5, 6, 7), (4, 5, 6, 7, 8, 9), (6, 7, 8, 9, 10, 11)]
    size8 = [
        (0, 1, 2, 3, 4, 5, 6, 7),
        (2, 3, 4, 5, 6, 7, 8, 9),
        (4, 5, 6, 7, 8, 9, 10, 11),
        (6, 7, 8, 9, 10, 11, 12, 13),
    ]
    regions: list[tuple[int, ...]] = [*size6, *size8]
    edges = [
        torch.tensor(_cycle_edges(len(region)), dtype=torch.long).reshape(-1, 2)
        for region in regions
    ]

    return RegionCorpus(
        node_ids=node_ids,
        features=features,
        regions=regions,
        edges=edges,
        train_idx=[0, 1, 2, 4, 5, 6],
        val_idx=[3, 7],
        dropped_featureless_regions=0,
    )


def _tiny_cfg(*, epochs: int = 2) -> TrainConfig:
    """`TrainConfig` at the tiny dims the exact interface spec prescribes for tests."""
    return TrainConfig(
        z_dim=4,
        ae_dim=16,
        ae_heads=4,
        ae_rrwp_k=3,
        ae_seeds=2,
        ae_d_model=32,
        set_width=32,
        set_layers=2,
        set_heads=2,
        set_pma_seeds=2,
        time_dim=8,
        epochs=epochs,
        batch_sets=4,
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class TestTrainAe:
    def test_checkpoint_contract_and_metrics(self, tmp_path: Path) -> None:
        corpus = _make_corpus()
        cfg = _tiny_cfg(epochs=2)
        out_path = tmp_path / "ae.pt"
        metrics_path = tmp_path / "ae_metrics.jsonl"

        train_ae(corpus, cfg, out_path=out_path, metrics_path=metrics_path)

        assert out_path.exists()
        payload = load_checkpoint(out_path)
        assert _CHECKPOINT_KEYS | {"latent_mean", "latent_std"} <= payload.keys()
        assert payload["in_dim"] == corpus.features.shape[1]
        latent_mean = payload["latent_mean"]
        latent_std = payload["latent_std"]
        assert isinstance(latent_mean, torch.Tensor)
        assert isinstance(latent_std, torch.Tensor)
        assert latent_mean.shape == (1 + cfg.z_dim,)
        assert latent_std.shape == (1 + cfg.z_dim,)
        assert bool(torch.all(latent_std >= 1e-6))

        rows = _read_jsonl(metrics_path)
        assert len(rows) == cfg.epochs
        for row in rows:
            assert row.keys() >= _AE_METRIC_KEYS

    def test_train_loss_decreases_over_epochs(self, tmp_path: Path) -> None:
        corpus = _make_corpus()
        cfg = _tiny_cfg(epochs=3)

        train_ae(
            corpus,
            cfg,
            out_path=tmp_path / "ae.pt",
            metrics_path=tmp_path / "ae_metrics.jsonl",
        )

        rows = _read_jsonl(tmp_path / "ae_metrics.jsonl")
        first_total = rows[0]["train_total"]
        last_total = rows[-1]["train_total"]
        assert isinstance(first_total, float)
        assert isinstance(last_total, float)
        assert last_total < first_total


class TestTrainPrior:
    def test_runs_after_train_ae(self, tmp_path: Path) -> None:
        corpus = _make_corpus()
        cfg = _tiny_cfg(epochs=2)
        ae_path = tmp_path / "ae.pt"
        train_ae(corpus, cfg, out_path=ae_path, metrics_path=tmp_path / "ae_metrics.jsonl")

        prior_path = tmp_path / "prior.pt"
        prior_metrics = tmp_path / "prior_metrics.jsonl"
        train_prior(corpus, cfg, ae_path=ae_path, out_path=prior_path, metrics_path=prior_metrics)

        assert prior_path.exists()
        payload = load_checkpoint(prior_path)
        assert payload.keys() >= _CHECKPOINT_KEYS
        assert "latent_mean" not in payload
        assert "latent_std" not in payload
        rows = _read_jsonl(prior_metrics)
        assert len(rows) == cfg.epochs
        for row in rows:
            assert {"stage", "epoch", "train_total", "val_total", "lr"} <= row.keys()

    def test_missing_ae_path_raises(self, tmp_path: Path) -> None:
        corpus = _make_corpus()
        cfg = _tiny_cfg(epochs=2)

        with pytest.raises(FileNotFoundError):
            train_prior(
                corpus,
                cfg,
                ae_path=tmp_path / "does_not_exist.pt",
                out_path=tmp_path / "prior.pt",
                metrics_path=tmp_path / "prior_metrics.jsonl",
            )


class TestTrainDet:
    def test_runs_after_train_ae(self, tmp_path: Path) -> None:
        corpus = _make_corpus()
        cfg = _tiny_cfg(epochs=2)
        ae_path = tmp_path / "ae.pt"
        train_ae(corpus, cfg, out_path=ae_path, metrics_path=tmp_path / "ae_metrics.jsonl")

        det_path = tmp_path / "det.pt"
        det_metrics = tmp_path / "det_metrics.jsonl"
        train_det(corpus, cfg, ae_path=ae_path, out_path=det_path, metrics_path=det_metrics)

        assert det_path.exists()
        payload = load_checkpoint(det_path)
        assert payload.keys() >= _CHECKPOINT_KEYS
        assert "latent_mean" not in payload
        rows = _read_jsonl(det_metrics)
        assert len(rows) == cfg.epochs
        for row in rows:
            assert {"train_mse", "train_bce", "val_mse", "val_bce"} <= row.keys()


class TestTrainMarg:
    def test_runs(self, tmp_path: Path) -> None:
        corpus = _make_corpus()
        cfg = _tiny_cfg(epochs=2)
        out_path = tmp_path / "marg.pt"
        metrics_path = tmp_path / "marg_metrics.jsonl"

        train_marg(corpus, cfg, out_path=out_path, metrics_path=metrics_path)

        assert out_path.exists()
        payload = load_checkpoint(out_path)
        assert payload.keys() >= _CHECKPOINT_KEYS
        assert "latent_mean" not in payload
        rows = _read_jsonl(metrics_path)
        assert len(rows) == cfg.epochs


class TestCheckpointRoundTrip:
    def test_rebuilt_models_load_state_dict_strictly(self, tmp_path: Path) -> None:
        corpus = _make_corpus()
        cfg = _tiny_cfg(epochs=2)
        ae_path = tmp_path / "ae.pt"
        train_ae(corpus, cfg, out_path=ae_path, metrics_path=tmp_path / "ae_metrics.jsonl")
        prior_path = tmp_path / "prior.pt"
        train_prior(
            corpus,
            cfg,
            ae_path=ae_path,
            out_path=prior_path,
            metrics_path=tmp_path / "prior_metrics.jsonl",
        )
        det_path = tmp_path / "det.pt"
        train_det(
            corpus,
            cfg,
            ae_path=ae_path,
            out_path=det_path,
            metrics_path=tmp_path / "det_metrics.jsonl",
        )
        marg_path = tmp_path / "marg.pt"
        train_marg(corpus, cfg, out_path=marg_path, metrics_path=tmp_path / "marg_metrics.jsonl")

        ae_payload = load_checkpoint(ae_path)
        ae_cfg = TrainConfig(**ae_payload["config"])  # type: ignore[arg-type]
        ae_model = RegionAutoencoder(
            in_dim=cast(int, ae_payload["in_dim"]),
            z_dim=ae_cfg.z_dim,
            dim=ae_cfg.ae_dim,
            layers=ae_cfg.ae_layers,
            rrwp_k=ae_cfg.ae_rrwp_k,
            n_heads=ae_cfg.ae_heads,
            seeds=ae_cfg.ae_seeds,
            d_model=ae_cfg.ae_d_model,
        )
        ae_model.load_state_dict(ae_payload["model"])  # type: ignore[arg-type]

        prior_payload = load_checkpoint(prior_path)
        prior_cfg = TrainConfig(**prior_payload["config"])  # type: ignore[arg-type]
        denoiser = SetDenoiser(
            in_dim=cast(int, prior_payload["in_dim"]),
            z_dim=prior_cfg.z_dim,
            width=prior_cfg.set_width,
            sab_layers=prior_cfg.set_layers,
            heads=prior_cfg.set_heads,
            pma_seeds=prior_cfg.set_pma_seeds,
            time_dim=prior_cfg.time_dim,
        )
        denoiser.load_state_dict(prior_payload["model"])  # type: ignore[arg-type]

        det_payload = load_checkpoint(det_path)
        det_cfg = TrainConfig(**det_payload["config"])  # type: ignore[arg-type]
        det_model = DetTwin(
            in_dim=cast(int, det_payload["in_dim"]),
            z_dim=det_cfg.z_dim,
            width=det_cfg.set_width,
            sab_layers=det_cfg.set_layers,
            heads=det_cfg.set_heads,
            pma_seeds=det_cfg.set_pma_seeds,
        )
        det_model.load_state_dict(det_payload["model"])  # type: ignore[arg-type]

        marg_payload = load_checkpoint(marg_path)
        marg_cfg = TrainConfig(**marg_payload["config"])  # type: ignore[arg-type]
        marg_model = MargDegreeRegressor(
            in_dim=cast(int, marg_payload["in_dim"]), hidden=marg_cfg.marg_hidden
        )
        marg_model.load_state_dict(marg_payload["model"])  # type: ignore[arg-type]


class TestAppendJsonl:
    def test_appends_one_json_object_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "records.jsonl"

        append_jsonl(path, {"a": 1})
        append_jsonl(path, {"b": 2})

        lines = path.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"a": 1}
        assert json.loads(lines[1]) == {"b": 2}


class TestDeterminism:
    def test_train_marg_same_seed_same_val_loss(self, tmp_path: Path) -> None:
        corpus = _make_corpus()
        cfg = _tiny_cfg(epochs=2)

        train_marg(corpus, cfg, out_path=tmp_path / "marg1.pt", metrics_path=tmp_path / "m1.jsonl")
        train_marg(corpus, cfg, out_path=tmp_path / "marg2.pt", metrics_path=tmp_path / "m2.jsonl")

        first = load_checkpoint(tmp_path / "marg1.pt")
        second = load_checkpoint(tmp_path / "marg2.pt")
        assert first["val_loss"] == second["val_loss"]

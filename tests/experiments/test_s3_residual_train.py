"""Unit tests for `src/experiments/s3_set_residual/train.py`'s loop and selection logic.

Fixtures hand-build tiny `S3Corpus`/`VvalEval`/`VvalFeatures` instances directly
(no real scorer, no `FeatureStore`) -- base logits and true adjacency are just
arbitrary small numbers, which is all `train_arm`'s loss and selection machinery
needs. The real `v3_1`-checkpoint end-to-end path (`cache` -> `train`) is covered
separately by `test_s3_residual_end_to_end.py`.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import pytest
import torch
from src.eval.graph_metrics import MMDConfig, precompute_bucket_reference
from src.experiments.s2_latent_topology.data import RegionCorpus
from src.experiments.s3_set_residual import train as train_mod
from src.experiments.s3_set_residual.data import S3Corpus, VvalEval
from src.experiments.s3_set_residual.model import ResidualConfig, SetResidualModel

pytestmark = pytest.mark.unit

D_IN = 8
CPU = torch.device("cpu")


# --------------------------------------------------------------------------- fixtures


def _tiny_model_cfg(mode: str) -> ResidualConfig:
    return ResidualConfig(
        mode=mode,  # type: ignore[arg-type]
        d_in=D_IN,
        d_model=16,
        sab_layers=1,
        heads=2,
        pma_seeds=2,
        p_dim=6,
        head_hidden=12,
    )


def _watts_strogatz_real_adj(node_count: int) -> tuple[nx.Graph, list[str], torch.Tensor]:
    """A small connected Watts-Strogatz graph plus its `(n, n)` 0/1 loopless adjacency."""
    graph = nx.watts_strogatz_graph(node_count, k=4, p=0.3, seed=0)
    graph = nx.relabel_nodes(graph, {i: f"v{i:02d}" for i in graph.nodes()})
    node_order = sorted(graph.nodes())
    adj = torch.zeros(node_count, node_count)
    for u, v in graph.edges():
        i, j = node_order.index(u), node_order.index(v)
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    return graph, node_order, adj


def _make_vval_with_true_adjs(
    node_count: int, true_adjs: list[torch.Tensor]
) -> tuple[VvalEval, train_mod.VvalFeatures]:
    """A hand-built `VvalEval` + `VvalFeatures`: one size, one draw per `true_adjs` entry.

    `true_adjs` lets callers control exactly which draws are single-class (for
    `_validate`'s skip-counting / all-single-class-raises tests) independent of
    `bucket_ref`, which is still built from the real graph's own edges (only
    `_topology_five_numbers` reads `bucket_ref`; `_validate` never does).
    """
    graph, node_order, _real_adj = _watts_strogatz_real_adj(node_count)
    draws = len(true_adjs)
    n = node_count

    buckets = {n: [set(node_order) for _ in range(draws)]}
    bucket_ref = precompute_bucket_reference(graph, buckets, MMDConfig())

    rng = np.random.default_rng(0)
    base_mats = []
    for _ in range(draws):
        m = torch.zeros(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                val = float(rng.normal())
                m[i, j] = val
                m[j, i] = val
        base_mats.append(m)

    vval = VvalEval(
        sizes=(n,),
        node_ids={n: [node_order for _ in range(draws)]},
        base_logits={n: base_mats},
        true_adj={n: [adj.clone() for adj in true_adjs]},
        bucket_ref=bucket_ref,
        checkpoint_id="tiny-checkpoint",
    )
    features = train_mod.VvalFeatures(
        node_ids=node_order,
        features=torch.tensor(rng.normal(size=(n, D_IN)), dtype=torch.float32),
    )
    return vval, features


def _make_vval(node_count: int = 6, draws: int = 2) -> tuple[VvalEval, train_mod.VvalFeatures]:
    """A tiny hand-built `VvalEval` + `VvalFeatures`: one size, `draws` real-edge draws."""
    _graph, _node_order, real_adj = _watts_strogatz_real_adj(node_count)
    return _make_vval_with_true_adjs(node_count, [real_adj.clone() for _ in range(draws)])


def _make_corpus(d_in: int = D_IN) -> S3Corpus:
    """A tiny hand-built `S3Corpus`: 5 regions (sizes 5,5,6,6,5), path-graph positives."""
    rng = np.random.default_rng(1)
    node_ids = [f"n{i:03d}" for i in range(30)]
    features = torch.tensor(rng.normal(size=(30, d_in)), dtype=torch.float32)

    regions: list[tuple[int, ...]] = []
    edges: list[torch.Tensor] = []
    train_pairs: list[torch.Tensor] = []
    train_labels: list[torch.Tensor] = []
    train_base_logits: list[torch.Tensor] = []

    cursor = 0
    for size in (5, 5, 6, 6, 5):
        region = tuple(range(cursor, cursor + size))
        cursor += size
        regions.append(region)
        pos = [(i, i + 1) for i in range(size - 1)]
        edges.append(torch.tensor(pos, dtype=torch.long))
        neg_pool = [(i, j) for i in range(size) for j in range(i + 1, size) if (i, j) not in pos]
        neg = neg_pool[: len(pos)]
        pairs = pos + neg
        labels = [1.0] * len(pos) + [0.0] * len(neg)
        train_pairs.append(torch.tensor(pairs, dtype=torch.long).reshape(-1, 2))
        train_labels.append(torch.tensor(labels, dtype=torch.float32))
        train_base_logits.append(torch.tensor(rng.normal(size=len(pairs)), dtype=torch.float32))

    base = RegionCorpus(
        node_ids=node_ids,
        features=features,
        regions=regions,
        edges=edges,
        train_idx=list(range(len(regions))),
        val_idx=[],
        dropped_featureless_regions=0,
    )
    return S3Corpus(
        base=base,
        vval_nodes=frozenset(),
        train_pairs=train_pairs,
        train_labels=train_labels,
        train_base_logits=train_base_logits,
    )


# --------------------------------------------------------------------------- _batch_loss


class TestBatchLoss:
    def test_matches_manual_per_set_then_batch_mean(self) -> None:
        pairs = torch.tensor([[0, 0, 1], [0, 1, 2], [1, 0, 1]], dtype=torch.long)
        delta = torch.tensor([0.1, -0.2, 0.05])
        base = torch.tensor([0.3, 0.4, -0.1])
        labels = torch.tensor([1.0, 0.0, 1.0])

        loss = train_mod._batch_loss(delta, base, labels, pairs, n_sets=2)
        assert loss is not None

        per_row = torch.nn.functional.binary_cross_entropy_with_logits(
            delta + base, labels, reduction="none"
        )
        set0_mean = per_row[:2].mean()
        set1_mean = per_row[2:].mean()
        expected = (set0_mean + set1_mean) / 2
        torch.testing.assert_close(loss, expected)

    def test_size_invariant_across_sets(self) -> None:
        # set 0: 1 pair at logit 2.0 (label 1 -> small loss); set 1: 10 identical
        # copies of a pair at logit -2.0 (label 1 -> large loss). If pair-count
        # weighted the mean, the many-pair set would dominate; it must not.
        pairs = torch.tensor([[0, 0, 1]] + [[1, 0, 1]] * 10, dtype=torch.long)
        delta = torch.zeros(11)
        base = torch.tensor([2.0] + [-2.0] * 10)
        labels = torch.ones(11)

        loss = train_mod._batch_loss(delta, base, labels, pairs, n_sets=2)
        assert loss is not None
        set0 = torch.nn.functional.binary_cross_entropy_with_logits(base[:1], labels[:1])
        set1 = torch.nn.functional.binary_cross_entropy_with_logits(base[1:], labels[1:])
        torch.testing.assert_close(loss, (set0 + set1) / 2)

    def test_returns_none_when_batch_empty(self) -> None:
        pairs = torch.zeros((0, 3), dtype=torch.long)
        delta = torch.zeros(0)
        base = torch.zeros(0)
        labels = torch.zeros(0)
        assert train_mod._batch_loss(delta, base, labels, pairs, n_sets=3) is None


# --------------------------------------------------------------------------- _rank / _select_best


class TestRank:
    def test_descending_ranks_highest_first(self) -> None:
        assert train_mod._rank([0.1, 0.9, 0.5], descending=True) == [3.0, 1.0, 2.0]

    def test_ascending_ranks_lowest_first(self) -> None:
        assert train_mod._rank([0.1, 0.9, 0.5], descending=False) == [1.0, 3.0, 2.0]

    def test_descending_exact_tie_gets_fractional_rank(self) -> None:
        # [0.5, 0.5, 0.3] descending: the two 0.5s tie for 1st/2nd -> both get
        # the average rank 1.5; the 0.3 is unambiguously 3rd.
        assert train_mod._rank([0.5, 0.5, 0.3], descending=True) == [1.5, 1.5, 3.0]

    def test_ascending_exact_tie_gets_fractional_rank(self) -> None:
        assert train_mod._rank([0.3, 0.5, 0.5], descending=False) == [1.0, 2.5, 2.5]


class TestSelectBest:
    def test_matches_manual_mean_rank(self) -> None:
        candidates = [
            (
                1,
                0.10,
                {
                    "gs": 0.5,
                    "rd": 1.2,
                    "degree_mmd_ratio": 2.0,
                    "clustering_mmd_ratio": 2.0,
                    "spectral_mmd_ratio": 2.0,
                },
            ),
            (
                2,
                0.30,
                {
                    "gs": 0.8,
                    "rd": 0.9,
                    "degree_mmd_ratio": 1.0,
                    "clustering_mmd_ratio": 1.0,
                    "spectral_mmd_ratio": 1.0,
                },
            ),
            (
                3,
                0.05,
                {
                    "gs": 0.3,
                    "rd": 2.0,
                    "degree_mmd_ratio": 3.0,
                    "clustering_mmd_ratio": 3.0,
                    "spectral_mmd_ratio": 3.0,
                },
            ),
        ]
        best_epoch, table = train_mod._select_best(candidates)
        # epoch 2 wins every single metric -> mean rank 1.0, strictly best.
        assert best_epoch == 2
        row2 = next(r for r in table if r["epoch"] == 2)
        assert row2["mean_rank"] == 1.0

    def test_ties_broken_by_lower_epoch(self) -> None:
        # epoch 5 wins delta_auprc/gs/rd (3 metrics); epoch 2 wins all 3 MMD ratios
        # (3 metrics) -> identical mean rank (1.5 each) by construction.
        five_epoch5 = {
            "gs": 0.9,
            "rd": 1.0,
            "degree_mmd_ratio": 5.0,
            "clustering_mmd_ratio": 5.0,
            "spectral_mmd_ratio": 5.0,
        }
        five_epoch2 = {
            "gs": 0.5,
            "rd": 2.0,
            "degree_mmd_ratio": 1.0,
            "clustering_mmd_ratio": 1.0,
            "spectral_mmd_ratio": 1.0,
        }
        candidates = [(5, 0.3, five_epoch5), (2, 0.1, five_epoch2)]
        best_epoch, table = train_mod._select_best(candidates)
        rows = {row["epoch"]: row for row in table}
        assert rows[5]["mean_rank"] == pytest.approx(rows[2]["mean_rank"])
        assert best_epoch == 2

    def test_exact_metric_tie_uses_fractional_rank_in_mean(self) -> None:
        # epoch 1 and epoch 2 tie exactly on delta_auprc (0.30); every other
        # metric is distinct across all three. Hand-computed per-column ranks
        # (1=best): delta_auprc [1.5, 1.5, 3], gs [2, 1, 3], abs_rd_minus_1
        # [1, 2, 3], and all three MMD ratios [2, 1, 3] -> mean ranks
        # epoch1=10.5/6=1.75, epoch2=7.5/6=1.25, epoch3=18/6=3.0. If ties were
        # instead broken by input order (old behavior), epoch 1 would get
        # delta_auprc rank 1 and epoch 2 rank 2, understating epoch 2's mean
        # rank advantage.
        five_epoch1 = {
            "gs": 0.5,
            "rd": 1.0,  # |rd - 1| = 0.0
            "degree_mmd_ratio": 2.0,
            "clustering_mmd_ratio": 2.0,
            "spectral_mmd_ratio": 2.0,
        }
        five_epoch2 = {
            "gs": 0.9,
            "rd": 1.5,  # |rd - 1| = 0.5
            "degree_mmd_ratio": 1.0,
            "clustering_mmd_ratio": 1.0,
            "spectral_mmd_ratio": 1.0,
        }
        five_epoch3 = {
            "gs": 0.3,
            "rd": 2.0,  # |rd - 1| = 1.0
            "degree_mmd_ratio": 3.0,
            "clustering_mmd_ratio": 3.0,
            "spectral_mmd_ratio": 3.0,
        }
        candidates = [(1, 0.30, five_epoch1), (2, 0.30, five_epoch2), (3, 0.10, five_epoch3)]

        best_epoch, table = train_mod._select_best(candidates)

        rows = {row["epoch"]: row for row in table}
        assert rows[1]["mean_rank"] == pytest.approx(1.75)
        assert rows[2]["mean_rank"] == pytest.approx(1.25)
        assert rows[3]["mean_rank"] == pytest.approx(3.0)
        assert best_epoch == 2


# --------------------------------------------------------------------------- VvalFeatures


class TestVvalFeaturesRoundtrip:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        features = train_mod.VvalFeatures(node_ids=["a", "b", "c"], features=torch.randn(3, D_IN))
        path = tmp_path / "vval_features.pt"
        train_mod.save_vval_features(features, path)
        loaded = train_mod.load_vval_features(path)
        assert loaded.node_ids == features.node_ids
        torch.testing.assert_close(loaded.features, features.features)


# --------------------------------------------------------------------------- _validate


class TestValidate:
    def test_zero_init_delta_auprc_is_exact_zero(self) -> None:
        vval, features = _make_vval()
        model = SetResidualModel(_tiny_model_cfg("res"))
        delta_auprc, delta_var, margin_share, n_used, n_skipped = train_mod._validate(
            model, vval, features, CPU
        )
        assert delta_auprc == 0.0
        assert delta_var == 0.0
        assert margin_share == 0.0
        assert n_used == 2
        assert n_skipped == 0

    def test_perturbed_model_gives_finite_nonzero_variance(self) -> None:
        vval, features = _make_vval()
        model = SetResidualModel(_tiny_model_cfg("res"))
        torch.manual_seed(0)
        with torch.no_grad():
            for p in model.pair_out.parameters():
                p.add_(torch.randn_like(p) * 0.5)
        delta_auprc, delta_var, margin_share, n_used, n_skipped = train_mod._validate(
            model, vval, features, CPU
        )
        assert math_finite(delta_auprc)
        assert delta_var > 0.0
        assert 0.0 <= margin_share <= 1.0
        assert n_used == 2
        assert n_skipped == 0

    def test_counts_used_and_skipped_single_class_sets(self) -> None:
        _graph, _node_order, real_adj = _watts_strogatz_real_adj(6)
        # One multi-class draw (real graph edges) + one single-class draw (no
        # edges at all -- every label is 0).
        vval, features = _make_vval_with_true_adjs(6, [real_adj, torch.zeros(6, 6)])
        model = SetResidualModel(_tiny_model_cfg("res"))
        delta_auprc, _delta_var, _margin_share, n_used, n_skipped = train_mod._validate(
            model, vval, features, CPU
        )
        assert math_finite(delta_auprc)
        assert n_used == 1
        assert n_skipped == 1

    def test_raises_when_every_set_is_single_class(self) -> None:
        vval, features = _make_vval_with_true_adjs(6, [torch.zeros(6, 6), torch.zeros(6, 6)])
        model = SetResidualModel(_tiny_model_cfg("res"))
        with pytest.raises(RuntimeError, match="single-class"):
            train_mod._validate(model, vval, features, CPU)


def math_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


# --------------------------------------------------------------------------- _topology_five_numbers


class TestTopologyFiveNumbers:
    def test_shapes_and_ranges(self) -> None:
        vval, features = _make_vval()
        model = SetResidualModel(_tiny_model_cfg("res"))
        five = train_mod._topology_five_numbers(model, vval, features, CPU)
        assert set(five) == {
            "gs",
            "rd",
            "degree_mmd_ratio",
            "clustering_mmd_ratio",
            "spectral_mmd_ratio",
        }
        assert 0.0 <= five["gs"] <= 1.0
        assert five["rd"] >= 0.0
        for stat in ("degree_mmd_ratio", "clustering_mmd_ratio", "spectral_mmd_ratio"):
            assert five[stat] >= 0.0

    def test_zero_init_matches_base_only_assembly(self) -> None:
        """At zero-init, delta==0 everywhere, so GS/RD must equal a base-only assembly."""
        from src.eval.assembly import assemble_graph, density_matched_threshold
        from src.eval.graph_metrics import compute_graph_similarity, compute_relative_density

        vval, features = _make_vval()
        model = SetResidualModel(_tiny_model_cfg("diag"))
        five = train_mod._topology_five_numbers(model, vval, features, CPU)

        size = vval.sizes[0]
        iu, iv = np.triu_indices(size, k=1)
        gs_values = []
        rd_values = []
        for k, node_ids in enumerate(vval.node_ids[size]):
            base = vval.base_logits[size][k].numpy()[iu, iv]
            true = vval.true_adj[size][k].numpy()[iu, iv]
            target_edges = int(round(float(true.sum())))
            if target_edges == 0:
                continue
            probs = 1.0 / (1.0 + np.exp(-base))
            pairs = [
                (node_ids[i], node_ids[j]) for i, j in zip(iu.tolist(), iv.tolist(), strict=True)
            ]
            threshold = density_matched_threshold(probs, target_edges)
            g_pred = assemble_graph(pairs, probs, threshold=threshold, nodes=node_ids)
            ref = vval.bucket_ref.ref_subgraphs[size][k]
            gs_values.append(compute_graph_similarity(g_pred, ref))
            rd_values.append(compute_relative_density(g_pred, ref))

        assert five["gs"] == pytest.approx(float(np.mean(gs_values)))
        assert five["rd"] == pytest.approx(float(np.mean(rd_values)))


# --------------------------------------------------------------------------- checkpoint helpers


class TestCheckpointHelpers:
    def test_save_and_load_model_roundtrip(self, tmp_path: Path) -> None:
        cfg = _tiny_model_cfg("pair")
        model = SetResidualModel(cfg)
        path = tmp_path / "epoch_1.pt"
        train_mod._save_checkpoint(path, model=model, model_cfg=cfg, epoch=1, delta_auprc=0.1)

        loaded = train_mod._load_model(path, cfg, CPU)
        x = torch.randn(1, 4, cfg.d_in)
        mask = torch.ones(1, 4)
        pairs = torch.tensor([[0, 0, 1], [0, 2, 3]], dtype=torch.long)
        torch.testing.assert_close(model(x, mask, pairs).detach(), loaded(x, mask, pairs).detach())

        payload = cast(dict[str, object], torch.load(path, map_location="cpu", weights_only=True))
        # Task 4's `evaluate.load_residual_checkpoint` hard-requires exactly these two
        # key names to load `best.pt` -- pin them here as a contract test.
        assert "model_state" in payload
        assert "config" in payload
        assert payload["config"] == dataclasses.asdict(cfg)
        assert payload["epoch"] == 1
        assert payload["vval_delta_auprc"] == 0.1


# --------------------------------------------------------------------------- train_arm


class TestTrainArm:
    def test_writes_expected_artifacts_and_prunes_checkpoints(self, tmp_path: Path) -> None:
        corpus = _make_corpus()
        vval, features = _make_vval()
        model_cfg = _tiny_model_cfg("res")
        cfg = train_mod.TrainConfig(
            epochs=4,
            batch_regions=2,
            lr=1e-2,
            patience=10,
            seed=0,
            device="cpu",
            top_k_checkpoints=2,
        )
        run_dir = tmp_path / "run"

        train_mod.train_arm(corpus, vval, features, model_cfg, cfg, run_dir=run_dir)

        assert (run_dir / "last.pt").exists()
        assert (run_dir / "best.pt").exists()
        metadata_path = run_dir / "run_metadata.json"
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text())
        assert metadata["arm"] == "res"
        assert metadata["checkpoint_id"] == "tiny-checkpoint"
        assert len(metadata["selection_table"]) <= cfg.top_k_checkpoints

        epoch_checkpoints = sorted(run_dir.glob("epoch_*.pt"))
        assert 0 < len(epoch_checkpoints) <= cfg.top_k_checkpoints

        lines = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
        first_record = json.loads(lines[0])
        assert first_record["epoch"] == 0
        assert first_record["vval_delta_auprc"] == 0.0
        assert first_record["train_loss"] is None
        # `_make_vval`'s 2 draws are both multi-class -> both used at epoch 0.
        assert first_record["n_sets_used"] == 2
        assert first_record["n_sets_skipped_single_class"] == 0
        assert len(lines) == cfg.epochs + 1
        for line in lines:
            record = json.loads(line)
            assert "n_sets_used" in record
            assert "n_sets_skipped_single_class" in record
            assert record["n_sets_used"] + record["n_sets_skipped_single_class"] == 2

    def test_raises_on_non_finite_loss(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus = _make_corpus()
        vval, features = _make_vval()
        model_cfg = _tiny_model_cfg("res")
        cfg = train_mod.TrainConfig(epochs=1, batch_regions=2, seed=0, device="cpu")

        def _nan_forward(
            self: SetResidualModel,
            x: torch.Tensor,
            mask: torch.Tensor,
            pairs: torch.Tensor,
            x_set: torch.Tensor | None = None,
        ) -> torch.Tensor:
            # Only poison the *training* forward pass -- the epoch-0 sanity
            # validation pass runs in eval mode and must stay finite so this
            # test isolates "non-finite training loss raises", not a
            # sklearn NaN-input error from an unrelated code path.
            if self.training:
                return torch.full((pairs.shape[0],), float("nan"))
            return torch.zeros(pairs.shape[0])

        monkeypatch.setattr(SetResidualModel, "forward", _nan_forward)

        with pytest.raises(RuntimeError, match="non-finite"):
            train_mod.train_arm(corpus, vval, features, model_cfg, cfg, run_dir=tmp_path / "run")

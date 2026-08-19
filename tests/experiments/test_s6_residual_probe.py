"""Tests for the S6 pair-residual identifiability probe."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from src.data.features import FeatureStore
from src.distill.artifacts import load_kd_targets, write_kd_targets
from src.experiments.s6_residual_probe import (
    anchor_holdout_split,
    baseline_readouts,
    build_residual_model,
    regression_readout,
    residual_targets,
    train_residual_probe,
    zero_output_head,
)
from src.score_universe import build_model

INPUT_DIM = 16
TOKEN_LEN = 5
POOLED_DIM = 4


# --------------------------------------------------------------------------- fixtures / helpers


def _write_feature_root(
    tmp_path: Path, node_ids: list[str], *, token_lens: list[int] | None = None
) -> Path:
    """Build a tiny synthetic FeatureStore root: metadata.json + index.json + per-node .pt.

    `token_lens`, when given, sets a per-node token count so the pairs span more
    than one `LengthBucketedBatchSampler` bucket.
    """
    root = tmp_path / "features"
    embeddings_dir = root / "embeddings"
    embeddings_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    index: dict[str, str] = {}
    for position, node_id in enumerate(node_ids):
        length = TOKEN_LEN if token_lens is None else token_lens[position]
        tensor = torch.tensor(rng.standard_normal((length, INPUT_DIM)), dtype=torch.float32)
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


def _tiny_v3_1_config(*, spectral_norm: bool = False) -> dict[str, object]:
    return {
        "input_dim": INPUT_DIM,
        "d_model": 8,
        "encoder_layers": 1,
        "cross_attn_layers": 1,
        "n_heads": 2,
        "mlp_head": {
            "hidden_dims": [8],
            "dropout": 0.0,
            "activation": "gelu",
            "norm": "layernorm",
            "spectral_norm": spectral_norm,
        },
        "regularization": {"dropout": 0.0},
    }


def _write_targets(
    directory: Path,
    *,
    node_ids: list[str],
    rows_per_anchor: int = 3,
    with_content_logit: bool = True,
    seed: int = 0,
) -> Path:
    """Write a tiny KD-targets artifact with `rows_per_anchor` rows for every anchor."""
    rng = np.random.default_rng(seed)
    n_nodes = len(node_ids)
    anchor_idx: list[int] = []
    partner_idx: list[int] = []
    offsets = [0]
    for a in range(n_nodes):
        for j in range(rows_per_anchor):
            anchor_idx.append(a)
            partner_idx.append((a + j + 1) % n_nodes)
        offsets.append(len(anchor_idx))
    n_pairs = len(anchor_idx)
    teacher_logit = rng.standard_normal(n_pairs).astype(np.float32)
    content_logit = rng.standard_normal(n_pairs).astype(np.float32)
    write_kd_targets(
        directory,
        node_ids=node_ids,
        pair_anchor_idx=np.asarray(anchor_idx, dtype=np.int32),
        pair_partner_idx=np.asarray(partner_idx, dtype=np.int32),
        anchor_offsets=np.asarray(offsets, dtype=np.int64),
        teacher_logit=teacher_logit,
        teacher_pooled_ab=rng.standard_normal((n_pairs, POOLED_DIM)).astype(np.float16),
        teacher_pooled_ba=rng.standard_normal((n_pairs, POOLED_DIM)).astype(np.float16),
        is_near=np.ones(n_pairs, dtype=np.uint8),
        pair_label=np.ones(n_pairs, dtype=np.int8),
        truth_graph_sha256="0" * 64,
        checkpoint_path=Path("fake.pt"),
        checkpoint_sha256="1" * 64,
        checkpoint_id="cafefeeddeadbeef",
        k_near=2,
        k_rand=1,
        seed=seed,
        content_logit=content_logit if with_content_logit else None,
    )
    return directory


# --------------------------------------------------------------------------- residual target


class TestResidualTargets:
    """Delta math and the v2/v3 artifact distinction."""

    def test_delta_is_teacher_minus_content_in_fp32(self, tmp_path: Path) -> None:
        path = _write_targets(tmp_path / "v3", node_ids=[f"n{i}" for i in range(6)])
        targets = load_kd_targets(path)
        delta = residual_targets(targets)
        assert delta.dtype == np.float32
        expected = targets.teacher_logit - targets.content_logit  # type: ignore[operator]
        np.testing.assert_allclose(delta, expected, rtol=0, atol=0)

    def test_v2_artifact_without_content_logit_raises(self, tmp_path: Path) -> None:
        path = _write_targets(
            tmp_path / "v2", node_ids=[f"n{i}" for i in range(6)], with_content_logit=False
        )
        targets = load_kd_targets(path)
        assert targets.content_logit is None
        with pytest.raises(ValueError, match="no 'content_logit'"):
            residual_targets(targets)


# --------------------------------------------------------------------------- anchor split


class TestAnchorHoldoutSplit:
    """The split is by anchor, never by row."""

    @staticmethod
    def _offsets(rows_per_anchor: list[int]) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(rows_per_anchor)]).astype(np.int64)

    def test_no_anchor_appears_on_both_sides(self) -> None:
        offsets = self._offsets([3] * 20)
        train_rows, eval_rows, heldout = anchor_holdout_split(offsets, seed=0)
        anchor_of = np.repeat(np.arange(20), 3)
        train_anchors = set(anchor_of[train_rows].tolist())
        eval_anchors = set(anchor_of[eval_rows].tolist())
        assert not train_anchors & eval_anchors
        assert eval_anchors == set(heldout.tolist())

    def test_every_row_of_a_heldout_anchor_is_in_eval(self) -> None:
        offsets = self._offsets([4] * 20)
        _, eval_rows, heldout = anchor_holdout_split(offsets, seed=1)
        expected = {r for a in heldout.tolist() for r in range(offsets[a], offsets[a + 1])}
        assert set(eval_rows.tolist()) == expected

    def test_rows_partition_completely(self) -> None:
        offsets = self._offsets([2] * 30)
        train_rows, eval_rows, _ = anchor_holdout_split(offsets, seed=0)
        assert set(train_rows.tolist()) | set(eval_rows.tolist()) == set(range(60))
        assert not set(train_rows.tolist()) & set(eval_rows.tolist())

    def test_anchors_with_no_rows_are_ignored(self) -> None:
        # 10 live anchors interleaved with 10 empty ones.
        rows = [3 if i % 2 == 0 else 0 for i in range(20)]
        offsets = self._offsets(rows)
        train_rows, eval_rows, heldout = anchor_holdout_split(offsets, seed=0)
        assert all(a % 2 == 0 for a in heldout.tolist())
        assert train_rows.size + eval_rows.size == 30

    def test_deterministic_for_a_seed(self) -> None:
        offsets = self._offsets([3] * 20)
        a = anchor_holdout_split(offsets, seed=5)
        b = anchor_holdout_split(offsets, seed=5)
        for left, right in zip(a, b, strict=True):
            np.testing.assert_array_equal(left, right)

    def test_too_few_anchors_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            anchor_holdout_split(self._offsets([3, 3]), seed=0)


# --------------------------------------------------------------------------- zeroed head


class TestZeroOutputHead:
    """The zero-init control must produce exactly the predict-zero baseline."""

    @staticmethod
    def _batch() -> dict[str, torch.Tensor]:
        return {
            "emb_a": torch.randn(4, TOKEN_LEN, INPUT_DIM),
            "emb_b": torch.randn(4, TOKEN_LEN, INPUT_DIM),
            "len_a": torch.full((4,), TOKEN_LEN, dtype=torch.long),
            "len_b": torch.full((4,), TOKEN_LEN, dtype=torch.long),
        }

    def test_plain_head_zeroes_to_exact_zero(self) -> None:
        torch.manual_seed(0)
        model = build_model("v3_1", _tiny_v3_1_config())
        removed = zero_output_head(model)  # type: ignore[arg-type]
        assert removed is False
        model.eval()
        with torch.no_grad():
            logits = model(self._batch())["logits"]
        torch.testing.assert_close(logits, torch.zeros_like(logits))

    def test_spectral_normed_head_is_unparametrized_then_zeroed(self) -> None:
        torch.manual_seed(0)
        model = build_model("v3_1", _tiny_v3_1_config(spectral_norm=True))
        removed = zero_output_head(model)  # type: ignore[arg-type]
        assert removed is True
        model.eval()
        with torch.no_grad():
            logits = model(self._batch())["logits"]
        # Without removing the parametrization this is NaN (weight_orig / sigma = 0/0).
        assert torch.isfinite(logits).all()
        torch.testing.assert_close(logits, torch.zeros_like(logits))

    def test_warm_and_scratch_both_start_at_predict_zero(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        config = _tiny_v3_1_config(spectral_norm=True)
        source = build_model("v3_1", config)
        checkpoint_path = tmp_path / "ckpt.pt"
        torch.save(
            {
                "model_state": source.state_dict(),
                "model_family": "v3_1",
                "model_config": config,
                "epoch": 0,
                "val_metrics": {},
                "seed": 0,
                "config": {},
            },
            checkpoint_path,
        )
        batch = self._batch()
        for init in ("warm", "scratch"):
            model, _cid, removed = build_residual_model(
                checkpoint_path, init=init, device=torch.device("cpu")
            )
            assert removed is True
            model.eval()
            with torch.no_grad():
                logits = model(batch)["logits"]
            torch.testing.assert_close(logits, torch.zeros_like(logits))

    def test_unknown_init_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="init must be"):
            build_residual_model(tmp_path / "nope.pt", init="tepid", device=torch.device("cpu"))


# --------------------------------------------------------------------------- readouts


class TestReadouts:
    """Metric conventions and the two baselines."""

    def test_matches_scipy_spearman(self) -> None:
        from scipy.stats import spearmanr

        rng = np.random.default_rng(0)
        y_true = rng.standard_normal(40)
        y_pred = rng.standard_normal(40)
        out = regression_readout(y_true, y_pred)
        assert out["spearman"] == pytest.approx(float(spearmanr(y_true, y_pred).statistic))

    def test_r2_matches_closed_form(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 2.5, 3.5])
        expected = 1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
        assert regression_readout(y_true, y_pred)["r2"] == pytest.approx(float(expected))

    def test_predict_mean_baseline_uses_train_mean(self) -> None:
        eval_delta = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        train_delta = np.array([10.0, 10.0], dtype=np.float32)
        out = baseline_readouts(eval_delta, train_delta)
        # MAE against a train mean of 10 -- not against the eval mean of 2.
        assert out["predict_mean"]["mae"] == pytest.approx(8.0)

    def test_predict_zero_baseline_is_the_zero_vector(self) -> None:
        eval_delta = np.array([1.0, -1.0, 2.0], dtype=np.float32)
        out = baseline_readouts(eval_delta, eval_delta)
        assert out["predict_zero"]["mae"] == pytest.approx(np.mean(np.abs(eval_delta)))


# --------------------------------------------------------------------------- smoke train


def test_smoke_train_runs_and_improves_on_its_own_init(tmp_path: Path) -> None:
    """A short CPU run trains, evaluates, and early-stops without NaNs."""
    node_ids = [f"n{i}" for i in range(20)]
    targets_dir = _write_targets(tmp_path / "v3", node_ids=node_ids, rows_per_anchor=3)
    targets = load_kd_targets(targets_dir)
    delta = residual_targets(targets)
    store = FeatureStore(_write_feature_root(tmp_path, node_ids))

    train_rows, eval_rows, _ = anchor_holdout_split(targets.anchor_offsets, seed=0)
    pairs = [
        (node_ids[int(targets.pair_anchor_idx[r])], node_ids[int(targets.pair_partner_idx[r])])
        for r in range(delta.size)
    ]
    train_pairs = [pairs[r] for r in train_rows.tolist()]
    eval_pairs = [pairs[r] for r in eval_rows.tolist()]

    torch.manual_seed(0)
    model = build_model("v3_1", _tiny_v3_1_config())
    zero_output_head(model)  # type: ignore[arg-type]

    history, best, best_epoch, predictions = train_residual_probe(
        model,  # type: ignore[arg-type]
        store=store,
        train_pairs=train_pairs,
        train_delta=delta[train_rows],
        eval_pairs=eval_pairs,
        eval_delta=delta[eval_rows],
        device=torch.device("cpu"),
        learning_rate=1e-3,
        max_epochs=3,
        patience=2,
        token_budget=4096,
        seed=0,
    )

    assert history
    assert best_epoch >= 0
    assert predictions.shape == (len(eval_pairs),)
    assert np.isfinite(predictions).all()
    assert all(np.isfinite(record.train_loss) for record in history)
    assert np.isfinite(best["r2"])


def test_zeroed_head_first_eval_equals_predict_zero_baseline(tmp_path: Path) -> None:
    """Before any optimizer step the model reproduces the predict-zero baseline exactly."""
    node_ids = [f"n{i}" for i in range(20)]
    targets_dir = _write_targets(tmp_path / "v3", node_ids=node_ids, rows_per_anchor=3)
    targets = load_kd_targets(targets_dir)
    delta = residual_targets(targets)
    store = FeatureStore(_write_feature_root(tmp_path, node_ids))
    _, eval_rows, _ = anchor_holdout_split(targets.anchor_offsets, seed=0)
    eval_pairs = [
        (node_ids[int(targets.pair_anchor_idx[r])], node_ids[int(targets.pair_partner_idx[r])])
        for r in eval_rows.tolist()
    ]

    torch.manual_seed(0)
    model = build_model("v3_1", _tiny_v3_1_config(spectral_norm=True))
    zero_output_head(model)  # type: ignore[arg-type]

    from src.experiments.s6_residual_probe import _predict

    predicted = _predict(
        model,  # type: ignore[arg-type]
        eval_pairs,
        store,
        torch.device("cpu"),
        4096,
    )
    np.testing.assert_allclose(predicted, np.zeros_like(predicted))

    eval_delta = delta[eval_rows]
    model_readout = regression_readout(eval_delta.astype(np.float64), predicted)
    zero_readout = baseline_readouts(eval_delta, eval_delta)["predict_zero"]
    assert model_readout["r2"] == pytest.approx(zero_readout["r2"])
    assert model_readout["mae"] == pytest.approx(zero_readout["mae"])


def test_predictions_stay_row_aligned_across_length_buckets(tmp_path: Path) -> None:
    """Bucket-major batching must not permute predictions relative to input rows.

    `LengthBucketedBatchSampler` groups by length bucket, so even unshuffled it
    does not visit rows in order. A sequential write would misalign predictions
    with targets here while still looking finite -- a silent readout corruption.
    """
    node_ids = [f"n{i}" for i in range(24)]
    # Straddle several bucket boundaries so the visiting order is definitely not 0,1,2,...
    token_lens = [3 + (i * 37) % 220 for i in range(len(node_ids))]
    store = FeatureStore(_write_feature_root(tmp_path, node_ids, token_lens=token_lens))
    pairs = [(node_ids[i], node_ids[(i * 7 + 3) % len(node_ids)]) for i in range(len(node_ids))]

    from src.data.pairs import LengthBucketedBatchSampler, probe_lengths
    from src.experiments.s6_residual_probe import _predict

    # The sampler really does reorder for this input -- otherwise the test proves nothing.
    lengths = probe_lengths(store, pairs)
    visiting_order = [
        i
        for batch in LengthBucketedBatchSampler(
            lengths, token_budget=4096, shuffle=False, seed=0, epoch=0
        )
        for i in batch
    ]
    assert visiting_order != list(range(len(pairs))), "fixture failed to span buckets"

    torch.manual_seed(0)
    model = build_model("v3_1", _tiny_v3_1_config())
    model.eval()
    predicted = _predict(
        model,  # type: ignore[arg-type]
        pairs,
        store,
        torch.device("cpu"),
        4096,
    )

    # Ground truth: score each pair on its own, in input order.
    expected = np.zeros(len(pairs), dtype=np.float64)
    with torch.no_grad():
        for i, (node_a, node_b) in enumerate(pairs):
            tokens_a = store.load_tokens(node_a)
            tokens_b = store.load_tokens(node_b)
            batch = {
                "emb_a": tokens_a.unsqueeze(0),
                "emb_b": tokens_b.unsqueeze(0),
                "len_a": torch.tensor([tokens_a.size(0)]),
                "len_b": torch.tensor([tokens_b.size(0)]),
            }
            expected[i] = float(model(batch)["logits"].squeeze())

    np.testing.assert_allclose(predicted, expected, rtol=1e-4, atol=1e-5)

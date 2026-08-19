"""Tests for the S6 pair-residual identifiability probe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

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
    """The split is by anchor, never by row.

    Training rows never touch a held-out anchor at either endpoint (anchor or
    partner).
    """

    @staticmethod
    def _offsets(rows_per_anchor: list[int]) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(rows_per_anchor)]).astype(np.int64)

    @staticmethod
    def _partners(rows_per_anchor: list[int]) -> np.ndarray:
        """Every row of anchor `a` points at anchor `(a + 1) % n` as its partner.

        Deterministic and simple, and -- because held-out anchors are spread
        across the permutation -- some train anchors' next-anchor partner
        lands on a held-out anchor, so the exclusion this class tests actually
        fires rather than being vacuously true.
        """
        n_anchors = len(rows_per_anchor)
        return np.concatenate(
            [
                np.full(count, (a + 1) % n_anchors, dtype=np.int64)
                for a, count in enumerate(rows_per_anchor)
            ]
        )

    def test_no_anchor_appears_on_both_sides(self) -> None:
        rows_per_anchor = [3] * 20
        offsets = self._offsets(rows_per_anchor)
        partners = self._partners(rows_per_anchor)
        split = anchor_holdout_split(offsets, partners, seed=0)
        anchor_of = np.repeat(np.arange(20), 3)
        train_anchors = set(anchor_of[split.train_rows].tolist())
        eval_anchors = set(anchor_of[split.eval_rows].tolist())
        assert not train_anchors & eval_anchors
        assert eval_anchors == set(split.heldout_anchors.tolist())

    def test_every_row_of_a_heldout_anchor_is_in_eval(self) -> None:
        rows_per_anchor = [4] * 20
        offsets = self._offsets(rows_per_anchor)
        partners = self._partners(rows_per_anchor)
        split = anchor_holdout_split(offsets, partners, seed=1)
        expected = {
            r for a in split.heldout_anchors.tolist() for r in range(offsets[a], offsets[a + 1])
        }
        assert set(split.eval_rows.tolist()) == expected

    def test_no_training_row_touches_a_heldout_anchor(self) -> None:
        """P1 regression: neither endpoint of a training row may be held out.

        Neither the anchor nor the partner of a training row may be a
        held-out anchor.
        """
        rows_per_anchor = [3] * 20
        offsets = self._offsets(rows_per_anchor)
        partners = self._partners(rows_per_anchor)
        split = anchor_holdout_split(offsets, partners, seed=0)
        heldout = set(split.heldout_anchors.tolist())
        anchor_of = np.repeat(np.arange(20), 3)
        assert split.n_dropped_touching_heldout > 0, "fixture failed to exercise the exclusion"
        for row in split.train_rows.tolist():
            assert int(anchor_of[row]) not in heldout
            assert int(partners[row]) not in heldout

    def test_dropped_row_accounting_is_correct_and_nonnegative(self) -> None:
        rows_per_anchor = [2] * 30
        offsets = self._offsets(rows_per_anchor)
        partners = self._partners(rows_per_anchor)
        split = anchor_holdout_split(offsets, partners, seed=0)
        assert split.n_dropped_touching_heldout >= 0
        total = int(offsets[-1])
        assert split.train_rows.size + split.eval_rows.size + split.n_dropped_touching_heldout == (
            total
        )
        assert not set(split.train_rows.tolist()) & set(split.eval_rows.tolist())

    def test_anchors_with_no_rows_are_ignored(self) -> None:
        # 10 live anchors interleaved with 10 empty ones.
        rows = [3 if i % 2 == 0 else 0 for i in range(20)]
        offsets = self._offsets(rows)
        partners = self._partners(rows)
        split = anchor_holdout_split(offsets, partners, seed=0)
        assert all(a % 2 == 0 for a in split.heldout_anchors.tolist())
        assert split.train_rows.size + split.eval_rows.size + split.n_dropped_touching_heldout == 30

    def test_deterministic_for_a_seed(self) -> None:
        rows_per_anchor = [3] * 20
        offsets = self._offsets(rows_per_anchor)
        partners = self._partners(rows_per_anchor)
        a = anchor_holdout_split(offsets, partners, seed=5)
        b = anchor_holdout_split(offsets, partners, seed=5)
        np.testing.assert_array_equal(a.train_rows, b.train_rows)
        np.testing.assert_array_equal(a.eval_rows, b.eval_rows)
        np.testing.assert_array_equal(a.heldout_anchors, b.heldout_anchors)
        assert a.n_dropped_touching_heldout == b.n_dropped_touching_heldout

    def test_too_few_anchors_raises(self) -> None:
        rows_per_anchor = [3, 3]
        with pytest.raises(ValueError, match="non-empty"):
            anchor_holdout_split(
                self._offsets(rows_per_anchor), self._partners(rows_per_anchor), seed=0
            )


# --------------------------------------------------------------------------- reciprocal leakage


def test_no_reciprocal_leakage_across_train_eval_split(tmp_path: Path) -> None:
    """P1 regression: a held-out anchor must never appear on the training side.

    Neither as anchor nor as partner, and no canonical (frozenset) pair may
    appear on both sides.

    The fixture is fully reciprocal -- every anchor has a row to every other
    node, so `(u, v)` and `(v, u)` both exist -- which is exactly the shape
    the symmetric `abba_max` readout collapses to the same pair. Before the
    P1 fix, `anchor_holdout_split` only dropped a held-out anchor's *own*
    rows, so a train anchor's row back to a held-out partner survived; this
    test fails against that code and passes once training rows touching a
    held-out anchor at either endpoint are dropped.
    """
    node_ids = [f"n{i}" for i in range(8)]
    targets_dir = _write_targets(
        tmp_path / "v3", node_ids=node_ids, rows_per_anchor=len(node_ids) - 1
    )
    targets = load_kd_targets(targets_dir)

    split = anchor_holdout_split(
        targets.anchor_offsets, targets.pair_partner_idx, seed=0, fraction=0.5
    )
    heldout = set(split.heldout_anchors.tolist())

    def endpoints(rows: np.ndarray) -> list[tuple[int, int]]:
        return [
            (int(targets.pair_anchor_idx[r]), int(targets.pair_partner_idx[r]))
            for r in rows.tolist()
        ]

    train_endpoints = endpoints(split.train_rows)
    eval_endpoints = endpoints(split.eval_rows)

    train_canonical = {frozenset(pair) for pair in train_endpoints}
    eval_canonical = {frozenset(pair) for pair in eval_endpoints}
    assert not (train_canonical & eval_canonical), "a canonical pair appears on both sides"

    for anchor, partner in train_endpoints:
        assert anchor not in heldout, "a held-out anchor appears as a training anchor"
        assert partner not in heldout, "a held-out anchor appears as a training partner"


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

    split = anchor_holdout_split(targets.anchor_offsets, targets.pair_partner_idx, seed=0)
    train_rows, eval_rows = split.train_rows, split.eval_rows
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
    eval_rows = anchor_holdout_split(
        targets.anchor_offsets, targets.pair_partner_idx, seed=0
    ).eval_rows
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


# --------------------------------------------------------------------------- best-state restore


def test_best_epoch_state_is_restored_not_the_last_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P2 regression: the model handed back must be the best epoch's checkpoint.

    Patches only `regression_readout`'s ranking signal (epoch 0 always wins,
    every later epoch always loses) so early stopping fires with the best
    epoch earlier than the epoch training stopped on. The predictions and
    model weights involved are entirely real and unpatched, so if the restore
    logic is correct, re-scoring `eval_pairs` on the model returned from
    `train_residual_probe` must reproduce `best_predictions` exactly -- a
    mismatch can only mean `model` was left at the last epoch, not the best
    one.
    """
    import src.experiments.s6_residual_probe as s6_module

    node_ids = [f"n{i}" for i in range(20)]
    targets_dir = _write_targets(tmp_path / "v3", node_ids=node_ids, rows_per_anchor=3)
    targets = load_kd_targets(targets_dir)
    delta = residual_targets(targets)
    store = FeatureStore(_write_feature_root(tmp_path, node_ids))

    split = anchor_holdout_split(targets.anchor_offsets, targets.pair_partner_idx, seed=0)
    pairs = [
        (node_ids[int(targets.pair_anchor_idx[r])], node_ids[int(targets.pair_partner_idx[r])])
        for r in range(delta.size)
    ]
    train_pairs = [pairs[r] for r in split.train_rows.tolist()]
    eval_pairs = [pairs[r] for r in split.eval_rows.tolist()]

    torch.manual_seed(0)
    model = build_model("v3_1", _tiny_v3_1_config())
    zero_output_head(model)  # type: ignore[arg-type]

    real_readout = s6_module.regression_readout
    call_count = {"n": 0}

    def forced_readout(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        out = real_readout(y_true, y_pred)
        forced_spearman = 1.0 if call_count["n"] == 0 else 0.0
        call_count["n"] += 1
        return {**out, "spearman": forced_spearman}

    monkeypatch.setattr(s6_module, "regression_readout", forced_readout)

    history, _best_readout, best_epoch, best_predictions = train_residual_probe(
        model,  # type: ignore[arg-type]
        store=store,
        train_pairs=train_pairs,
        train_delta=delta[split.train_rows],
        eval_pairs=eval_pairs,
        eval_delta=delta[split.eval_rows],
        device=torch.device("cpu"),
        learning_rate=1e-2,
        max_epochs=4,
        patience=1,
        token_budget=4096,
        seed=0,
    )

    assert best_epoch == 0
    assert history[-1].epoch != best_epoch, "fixture failed to force early stop past the best epoch"

    from src.experiments.s6_residual_probe import _predict

    restored_predictions = _predict(
        model,  # type: ignore[arg-type]
        eval_pairs,
        store,
        torch.device("cpu"),
        4096,
    )
    np.testing.assert_allclose(restored_predictions, best_predictions, rtol=1e-6, atol=1e-6)


def test_zeroed_head_survives_a_state_dict_round_trip() -> None:
    """The zeroed head must not leave a stale spectral-norm load hook behind.

    `torch.nn.utils.remove_spectral_norm` drops the forward pre-hook and the
    `weight_orig`/`weight_u`/`weight_v` entries but leaves its
    `_load_state_dict_pre_hook` installed, so the module keeps demanding those
    keys and every later `load_state_dict` raises -- which killed the
    best-epoch restore at the very end of a real training run, after the whole
    epoch budget had been spent. `state_dict()`, `named_parameters()` and
    `named_buffers()` all look correct in that state, so only an actual
    round-trip catches it.
    """
    torch.manual_seed(0)
    model = build_model("v3_1", _tiny_v3_1_config(spectral_norm=True))
    assert zero_output_head(model) is True  # type: ignore[arg-type]

    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)  # must not raise

    layers = cast(torch.nn.Sequential, model.output_head.layers)  # type: ignore[union-attr]
    final = layers[-1]
    assert isinstance(final, torch.nn.Linear)
    names = {name for name, _ in final.named_parameters()}
    assert not any(n.startswith("weight_orig") for n in names)
    assert not list(final.named_buffers())


def test_best_state_restore_works_on_a_spectral_normed_head(tmp_path: Path) -> None:
    """End-to-end: build warm from a spectral-normed checkpoint, then restore a state dict."""
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
    model, _cid, removed = build_residual_model(
        checkpoint_path, init="warm", device=torch.device("cpu")
    )
    assert removed is True
    snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(snapshot)  # the training loop's best-epoch restore

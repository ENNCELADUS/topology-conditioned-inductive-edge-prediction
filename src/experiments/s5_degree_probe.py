r"""S5: full-capacity node-degree probe -- is a node's degree budget in its features?

The B1 KD arms distill only row-normalized or pointwise projections of the
full-ego oracle teacher, which structurally discards absolute node activity.
S1-R showed that *true* node-aligned degree quotas plus greedy hard-quota
assembly close 73.2% of the clustering gap with pair scores untouched, but the
only feature-predicted arm ever tried was a ridge regression (GS 0.399 against
an oracle ceiling of 0.439). This module removes model capacity as an
explanation: it trains the exact B0 trunk as a degree regressor and emits a
``degree_predictions_v1`` file that `src.experiments.s4_budget_assembly`
consumes as a ``predicted_hard_<variant>`` arm.

A negative result here (deep ~= ridge) says node-aligned degree is not in the
features at any capacity, which supports the joint-allocation thesis. A strong
positive (deep >> ridge, and S4 GS approaching the oracle) says a legal budget
method exists.

Target convention: ``y = log1p(deg(strip_self_loops(train_graph)))`` over the
full ``train_graph.pkl`` (train+ u val+) substrate. The V_val quarantine is a
*pair-level* rule -- it forbids training on V_val-internal pairs, not reading
node degrees of the shared structural graph -- so the full substrate is used
and the convention is recorded in ``report.json``. The held-out split is a
selection device only; because held-out and training nodes share graph edges,
their degree targets are coupled, and that leakage is disclosed rather than
fixed (a fully decoupled split would need a node-induced subgraph, which
changes the target definition).

Trunk architecture is read from the checkpoint's own ``model_config`` for both
``--init warm`` and ``--init scratch``, so the two variants are architecturally
identical controls and a strict warm load cannot shape-mismatch. (The shipped
``configs/b0_v31_breadth_first.yaml`` declares ``d_model: 512`` while every
published v3_1 checkpoint carries ``d_model: 256``; trusting the checkpoint is
what keeps ``--init warm`` runnable.)

Precision: fp32 everywhere, no autocast, single GPU, no DDP.

CLI::

    python -m src.experiments.s5_degree_probe \
        --universe outputs/deliverables/b0_v31_breadth_first_20260711/scores/candidate.npz \
        --checkpoint outputs/deliverables/b0_v31_breadth_first_20260711/model/best.pt \
        --data-root data --strategy breadth_first \
        --init warm --seed 0 --output-dir outputs/s5_degree_probe/warm_s0
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import spearmanr
from torch import nn

from src.data.features import FeatureStore
from src.eval.graph_metrics import strip_self_loops
from src.experiments.g1_hardened_e2 import _BENCHMARK_SUBDIR, _FEATURES_SUBDIR
from src.model.egostitch.classifier.layers import (
    MLPHead,
    SiameseEncoder,
    _build_padding_mask,
    inner_token_mask,
    masked_mean,
)
from src.score_universe import load_scores, validate_artifact_precision

logger = logging.getLogger(__name__)

DEGREE_PREDICTIONS_FORMAT = "degree_predictions_v1"
S5_REPORT_FORMAT = "s5_degree_probe_v1"

_HOLDOUT_FRACTION = 0.1
_TOKEN_BUDGET = 65_536
_MAX_EPOCHS = 100
_PATIENCE = 10
_WEIGHT_DECAY = 0.05
_GRAD_CLIP = 1.0
_LR_WARM = 3e-5
_LR_SCRATCH = 1e-4
_HEAD_HIDDEN_DIMS = (256, 128)
_HEAD_DROPOUT = 0.1

_TARGET_CONVENTION = (
    "y = log1p(degree(strip_self_loops(train_graph.pkl))); full train+ u val+ substrate "
    "(V_val quarantine is a pair-level rule, not a node-degree rule)"
)
_LEAKAGE_CAVEAT = (
    "Held-out nodes share graph edges with training nodes, so their degree targets are "
    "statistically coupled. The split is a selection device, not an independent "
    "generalization estimate. Disclosed, not fixed."
)


# --------------------------------------------------------------------------- benchmark I/O


def load_train_graph(benchmark_root: Path, strategy: str) -> nx.Graph:
    """Unpickle ``<benchmark_root>/<strategy>/train_graph.pkl`` (self-loops not stripped).

    Args:
        benchmark_root: Benchmark package root.
        strategy: Split strategy name.

    Returns:
        The pickled `networkx.Graph`.

    Raises:
        TypeError: If the unpickled object is not a `networkx.Graph`.
    """
    path = benchmark_root / strategy / "train_graph.pkl"
    with path.open("rb") as f:
        graph = pickle.load(f)
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"{path}: expected a pickled networkx.Graph, got {type(graph)!r}")
    return graph


# --------------------------------------------------------------------------- targets and split


def derive_degree_targets(
    train_graph: nx.Graph, store_node_ids: frozenset[str]
) -> tuple[list[str], NDArray[np.float64]]:
    """Derive the ``log1p(loopless degree)`` regression target over feature-backed nodes.

    Nodes are ``sorted(strip_self_loops(train_graph).nodes & store_node_ids)``.
    The store intersection is what removes featureless nodes: ``exclude_nodes``
    filters only the pair files, so featureless nodes survive in the graph
    artifacts themselves.

    Args:
        train_graph: The ``train_graph.pkl`` substrate (self-loops may be present).
        store_node_ids: Node ids backed by the frozen feature store.

    Returns:
        ``(nodes, y)`` -- the sorted node list and its row-aligned float64
        ``log1p`` degree targets.
    """
    simple = strip_self_loops(train_graph)
    nodes = sorted(set(simple.nodes()) & store_node_ids)
    degrees = np.array([float(simple.degree(node)) for node in nodes], dtype=np.float64)
    return nodes, np.log1p(degrees)


def holdout_split(
    n_items: int, *, seed: int, fraction: float = _HOLDOUT_FRACTION
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Permute ``range(n_items)`` under `seed` and hold out the last `fraction`.

    Args:
        n_items: Number of items to split.
        seed: Seed for `numpy.random.default_rng`.
        fraction: Held-out share of the permutation tail.

    Returns:
        ``(train_idx, heldout_idx)`` index arrays into the original ordering.

    Raises:
        ValueError: If the split would leave either side empty.
    """
    permutation = np.random.default_rng(seed).permutation(n_items).astype(np.int64)
    n_heldout = int(round(n_items * fraction))
    if n_heldout < 1 or n_heldout >= n_items:
        raise ValueError(
            f"holdout split needs both sides non-empty; got n_items={n_items}, "
            f"fraction={fraction} -> n_heldout={n_heldout}"
        )
    return permutation[:-n_heldout], permutation[-n_heldout:]


def token_budget_chunks(lengths: Sequence[int], *, budget: int = _TOKEN_BUDGET) -> list[list[int]]:
    """Group item indices into length-sorted chunks under a padded-token budget.

    Items are visited in ascending token length so each chunk pads to nearly its
    own longest member; a chunk closes when adding the next item would push
    ``len(chunk) * max_len`` past `budget`. A single item always forms a chunk
    even if it exceeds the budget on its own.

    Args:
        lengths: Per-item token counts, indexed by item position.
        budget: Maximum padded tokens per chunk.

    Returns:
        Chunks of item indices, each chunk ascending in token length.

    Raises:
        ValueError: If `budget` is not positive.
    """
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    chunks: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for index in order:
        candidate_max = max(current_max, int(lengths[index]))
        if current and (len(current) + 1) * candidate_max > budget:
            chunks.append(current)
            current = [index]
            current_max = int(lengths[index])
        else:
            current.append(index)
            current_max = candidate_max
    if current:
        chunks.append(current)
    return chunks


# --------------------------------------------------------------------------- model


class DegreeProbe(nn.Module):
    """The B0 siamese trunk plus a scalar regression head over pooled node tokens.

    Pooling reuses the `NodeFactorBottleneck._pool` recipe -- ``masked_mean``
    over ``inner_token_mask`` -- so the probe reads the same per-node summary
    the B0 trunk's factor path does.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        token_dropout: float,
        stochastic_depth: float,
    ) -> None:
        """Build the encoder trunk and the scalar head."""
        super().__init__()
        self.encoder = SiameseEncoder(
            input_dim=input_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            token_dropout=token_dropout,
            stochastic_depth=stochastic_depth,
        )
        self.head = MLPHead(
            d_model,
            list(_HEAD_HIDDEN_DIMS),
            1,
            dropout=_HEAD_DROPOUT,
            activation="gelu",
            norm="layernorm",
        )

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Predict ``log1p`` degree for a padded batch of node token sequences.

        Args:
            tokens: Shape ``(B, L, input_dim)`` zero-padded token features.
            lengths: Shape ``(B,)`` unpadded token counts.

        Returns:
            Shape ``(B,)`` predicted ``log1p`` degrees.
        """
        encoded = self.encoder(tokens, lengths)
        mask = _build_padding_mask(lengths, encoded.size(1))
        pooled = masked_mean(encoded, inner_token_mask(encoded, mask))
        return cast(torch.Tensor, self.head(pooled).squeeze(-1))


def encoder_arch_from_checkpoint(model_config: dict[str, object]) -> dict[str, object]:
    """Extract the `DegreeProbe` trunk hyperparameters from a v3_1 ``model_config``.

    Args:
        model_config: The checkpoint's ``model_config`` mapping.

    Returns:
        Keyword arguments for `DegreeProbe`.

    Raises:
        KeyError: If a required architecture field is absent.
    """
    regularization = cast(dict[str, object], model_config.get("regularization", {}))
    return {
        "input_dim": int(cast(int, model_config["input_dim"])),
        "d_model": int(cast(int, model_config["d_model"])),
        "n_layers": int(cast(int, model_config["encoder_layers"])),
        "n_heads": int(cast(int, model_config["n_heads"])),
        "dropout": float(cast(float, regularization.get("dropout", 0.1))),
        "token_dropout": float(cast(float, regularization.get("token_dropout", 0.1))),
        "stochastic_depth": float(cast(float, regularization.get("stochastic_depth", 0.1))),
    }


def warm_start_encoder(probe: DegreeProbe, model_state: dict[str, torch.Tensor]) -> int:
    """Strict-load the ``encoder.``-prefixed checkpoint weights into ``probe.encoder``.

    Args:
        probe: The probe whose trunk receives the weights.
        model_state: A v3_1 checkpoint's ``model_state`` mapping.

    Returns:
        The number of tensors transferred.

    Raises:
        ValueError: If the checkpoint carries no ``encoder.``-prefixed keys.
        RuntimeError: If the strict load rejects the state dict (shape or key
            mismatch) -- never silently partial.
    """
    prefix = "encoder."
    encoder_state = {
        key[len(prefix) :]: value for key, value in model_state.items() if key.startswith(prefix)
    }
    if not encoder_state:
        raise ValueError(
            "checkpoint model_state has no 'encoder.'-prefixed keys to warm-start from"
        )
    probe.encoder.load_state_dict(encoder_state, strict=True)
    return len(encoder_state)


# --------------------------------------------------------------------------- metrics


def regression_readout(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
) -> dict[str, float]:
    """Spearman rho, R^2, and MAE of a prediction vector against its target.

    Args:
        y_true: Shape ``(n,)`` targets.
        y_pred: Shape ``(n,)`` predictions.

    Returns:
        ``{"spearman": ..., "r2": ..., "mae": ...}``. Spearman is ``0.0`` when
        either side is constant (`scipy` returns NaN there); R^2 is ``0.0``
        when the target has zero variance.
    """
    residual = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    rho = float(spearmanr(y_true, y_pred).statistic) if y_true.size > 1 else 0.0
    return {
        "spearman": 0.0 if not np.isfinite(rho) else rho,
        "r2": 0.0 if total == 0.0 else 1.0 - residual / total,
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }


# --------------------------------------------------------------------------- training


@dataclass(frozen=True)
class EpochRecord:
    """One epoch of the probe's training history."""

    epoch: int
    train_loss: float
    heldout_spearman: float
    heldout_r2: float
    heldout_mae: float


def _forward_chunks(
    probe: DegreeProbe,
    store: FeatureStore,
    nodes: Sequence[str],
    indices: Sequence[int],
    chunks: Sequence[Sequence[int]],
    device: torch.device,
) -> NDArray[np.float64]:
    """Run the probe in eval mode over `chunks`, returning predictions in `indices` order."""
    probe.eval()
    position = {index: slot for slot, index in enumerate(indices)}
    out = np.zeros(len(indices), dtype=np.float64)
    with torch.no_grad():
        for chunk in chunks:
            tokens, lengths = _collate_nodes(store, [nodes[i] for i in chunk], device)
            predicted = probe(tokens, lengths).detach().cpu().numpy().astype(np.float64)
            for slot, index in enumerate(chunk):
                out[position[index]] = predicted[slot]
    return out


def _collate_nodes(
    store: FeatureStore, node_ids: Sequence[str], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and zero-pad the token sequences for `node_ids` onto `device` in fp32."""
    tensors = [store.load_tokens(node_id) for node_id in node_ids]
    max_len = max(tensor.size(0) for tensor in tensors)
    padded = torch.zeros(len(tensors), max_len, tensors[0].size(1), dtype=torch.float32)
    for i, tensor in enumerate(tensors):
        padded[i, : tensor.size(0)] = tensor
    lengths = torch.tensor([tensor.size(0) for tensor in tensors], dtype=torch.long)
    return padded.to(device), lengths.to(device)


def train_probe(
    probe: DegreeProbe,
    *,
    store: FeatureStore,
    nodes: Sequence[str],
    targets: NDArray[np.float64],
    train_idx: NDArray[np.int64],
    heldout_idx: NDArray[np.int64],
    lengths: Sequence[int],
    device: torch.device,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    token_budget: int,
    seed: int,
) -> tuple[list[EpochRecord], dict[str, float], int, dict[str, torch.Tensor]]:
    """Train the probe with early stopping on held-out Spearman.

    Returns:
        ``(history, best_readout, best_epoch, best_state)``. ``best_state`` is a
        CPU copy of the parameters at the best held-out Spearman.

    Raises:
        ValueError: If the training loss becomes non-finite (fail-closed on
            non-finite state).
    """
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=_WEIGHT_DECAY)
    train_lengths = [lengths[i] for i in train_idx.tolist()]
    train_chunks = [
        [int(train_idx[j]) for j in chunk]
        for chunk in token_budget_chunks(train_lengths, budget=token_budget)
    ]
    heldout_lengths = [lengths[i] for i in heldout_idx.tolist()]
    heldout_chunks = [
        [int(heldout_idx[j]) for j in chunk]
        for chunk in token_budget_chunks(heldout_lengths, budget=token_budget)
    ]
    heldout_targets = targets[heldout_idx]

    rng = np.random.default_rng(seed)
    history: list[EpochRecord] = []
    best_readout = {"spearman": -np.inf, "r2": 0.0, "mae": 0.0}
    best_epoch = -1
    best_state: dict[str, torch.Tensor] = {}

    for epoch in range(max_epochs):
        probe.train()
        total_loss = 0.0
        total_items = 0
        for order in rng.permutation(len(train_chunks)).tolist():
            chunk = train_chunks[order]
            tokens, chunk_lengths = _collate_nodes(store, [nodes[i] for i in chunk], device)
            y = torch.tensor(targets[chunk], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(probe(tokens, chunk_lengths), y)
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite training loss at epoch {epoch}")
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(probe.parameters(), _GRAD_CLIP)
            optimizer.step()
            total_loss += float(loss.item()) * len(chunk)
            total_items += len(chunk)

        predicted = _forward_chunks(
            probe, store, nodes, heldout_idx.tolist(), heldout_chunks, device
        )
        readout = regression_readout(heldout_targets, predicted)
        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=total_loss / max(total_items, 1),
                heldout_spearman=readout["spearman"],
                heldout_r2=readout["r2"],
                heldout_mae=readout["mae"],
            )
        )
        if readout["spearman"] > best_readout["spearman"]:
            best_readout = readout
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
        elif epoch - best_epoch >= patience:
            logger.info("early stop at epoch %d (best epoch %d)", epoch, best_epoch)
            break

    return history, best_readout, best_epoch, best_state


# --------------------------------------------------------------------------- pipeline


def run_s5_pipeline(
    *,
    universe_path: Path,
    checkpoint_path: Path,
    data_root: Path,
    strategy: str,
    output_dir: Path,
    init: str,
    seed: int,
    device: str,
    max_epochs: int = _MAX_EPOCHS,
    patience: int = _PATIENCE,
    token_budget: int = _TOKEN_BUDGET,
    learning_rate: float | None = None,
) -> dict[str, object]:
    """Train one degree probe variant and write ``predictions.json`` + ``report.json``.

    Args:
        universe_path: Candidate-universe scores artifact; its ``node_ids``
            define the test scope the predictions must cover exactly.
        checkpoint_path: v3_1 checkpoint supplying the trunk architecture for
            both variants, and the weights for ``warm``.
        data_root: Root holding the benchmark and feature-store subdirectories.
        strategy: Split strategy name.
        output_dir: Destination directory for both artifacts.
        init: ``"warm"`` or ``"scratch"``.
        seed: Split/shuffle/init seed.
        device: Torch device string.
        max_epochs: Epoch ceiling.
        patience: Early-stop patience on held-out Spearman.
        token_budget: Padded-token budget per chunk.
        learning_rate: Overrides the per-variant default when given.

    Returns:
        The written report payload.

    Raises:
        ValueError: If `init` is not a known variant.
    """
    if init not in {"warm", "scratch"}:
        raise ValueError(f"init must be 'warm' or 'scratch', got {init!r}")

    torch.manual_seed(seed)
    resolved_device = torch.device(device)

    universe = load_scores(universe_path)
    validate_artifact_precision(universe, label="s5 candidate universe")
    test_nodes = list(universe.node_ids)

    benchmark_root = data_root / _BENCHMARK_SUBDIR
    store = FeatureStore(data_root / _FEATURES_SUBDIR)
    train_graph = load_train_graph(benchmark_root, strategy)
    nodes, targets = derive_degree_targets(train_graph, store.node_ids)
    lengths = [int(store.load_tokens(node).size(0)) for node in nodes]

    train_idx, heldout_idx = holdout_split(len(nodes), seed=seed)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = cast(dict[str, object], checkpoint["model_config"])
    probe = DegreeProbe(**encoder_arch_from_checkpoint(model_config))  # type: ignore[arg-type]
    transferred = 0
    if init == "warm":
        transferred = warm_start_encoder(
            probe, cast(dict[str, torch.Tensor], checkpoint["model_state"])
        )
    probe.to(resolved_device)

    lr = (
        learning_rate
        if learning_rate is not None
        else (_LR_WARM if init == "warm" else _LR_SCRATCH)
    )
    history, best_readout, best_epoch, best_state = train_probe(
        probe,
        store=store,
        nodes=nodes,
        targets=targets,
        train_idx=train_idx,
        heldout_idx=heldout_idx,
        lengths=lengths,
        device=resolved_device,
        learning_rate=lr,
        max_epochs=max_epochs,
        patience=patience,
        token_budget=token_budget,
        seed=seed,
    )
    if best_state:
        probe.load_state_dict(best_state)
        probe.to(resolved_device)

    train_readout = regression_readout(
        targets[train_idx],
        _forward_chunks(
            probe,
            store,
            nodes,
            train_idx.tolist(),
            [
                [int(train_idx[j]) for j in chunk]
                for chunk in token_budget_chunks(
                    [lengths[i] for i in train_idx.tolist()], budget=token_budget
                )
            ],
            resolved_device,
        ),
    )

    test_lengths = [int(store.load_tokens(node).size(0)) for node in test_nodes]
    test_chunks = token_budget_chunks(test_lengths, budget=token_budget)
    test_log1p = _forward_chunks(
        probe, store, test_nodes, list(range(len(test_nodes))), test_chunks, resolved_device
    )
    test_degrees = np.maximum(np.expm1(test_log1p), 0.0)

    variant = f"{init}_s{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_payload: dict[str, object] = {
        "format": DEGREE_PREDICTIONS_FORMAT,
        "variant": variant,
        "init": init,
        "seed": seed,
        "target_convention": _TARGET_CONVENTION,
        "checkpoint_path": str(checkpoint_path),
        "encoder_tensors_transferred": transferred,
        "selection": {
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "heldout_spearman": best_readout["spearman"],
            "heldout_r2": best_readout["r2"],
            "heldout_mae": best_readout["mae"],
        },
        "degree_predictions": {
            node: float(value) for node, value in zip(test_nodes, test_degrees, strict=True)
        },
    }
    (output_dir / "predictions.json").write_text(
        json.dumps(predictions_payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    report_payload: dict[str, object] = {
        "format": S5_REPORT_FORMAT,
        "evidence_class": "diagnostic",
        "variant": variant,
        "init": init,
        "seed": seed,
        "strategy": strategy,
        "target_convention": _TARGET_CONVENTION,
        "leakage_caveat": _LEAKAGE_CAVEAT,
        "architecture": encoder_arch_from_checkpoint(model_config),
        "checkpoint_path": str(checkpoint_path),
        "encoder_tensors_transferred": transferred,
        "learning_rate": lr,
        "split_manifest": {
            "n_nodes": len(nodes),
            "n_train": int(train_idx.size),
            "n_heldout": int(heldout_idx.size),
            "holdout_fraction": _HOLDOUT_FRACTION,
            "n_test_nodes": len(test_nodes),
        },
        "target_stats": {
            "log1p_mean": float(np.mean(targets)),
            "log1p_std": float(np.std(targets)),
            "degree_max": float(np.max(np.expm1(targets))),
        },
        "selection": {
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "heldout": best_readout,
            "train_side": train_readout,
            "generalization_gap_spearman": train_readout["spearman"] - best_readout["spearman"],
        },
        "history": [
            {
                "epoch": record.epoch,
                "train_loss": record.train_loss,
                "heldout_spearman": record.heldout_spearman,
                "heldout_r2": record.heldout_r2,
                "heldout_mae": record.heldout_mae,
            }
            for record in history
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report_payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return report_payload


def build_parser() -> argparse.ArgumentParser:
    """Build the S5 degree-probe CLI argument parser."""
    parser = argparse.ArgumentParser(description="S5 full-capacity node-degree probe")
    parser.add_argument("--universe", type=Path, required=True, help="Candidate-universe .npz")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="v3_1 checkpoint: trunk architecture for both variants, weights for --init warm",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init", choices=("warm", "scratch"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device"
    )
    parser.add_argument("--max-epochs", type=int, default=_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=_PATIENCE)
    parser.add_argument("--token-budget", type=int, default=_TOKEN_BUDGET)
    parser.add_argument("--learning-rate", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the S5 degree probe from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    report = run_s5_pipeline(
        universe_path=args.universe,
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        strategy=args.strategy,
        output_dir=args.output_dir,
        init=args.init,
        seed=args.seed,
        device=args.device,
        max_epochs=args.max_epochs,
        patience=args.patience,
        token_budget=args.token_budget,
        learning_rate=args.learning_rate,
    )
    selection = cast(dict[str, object], report["selection"])
    logger.info("s5 %s: selection=%s", report["variant"], json.dumps(selection, sort_keys=True))


if __name__ == "__main__":
    main()

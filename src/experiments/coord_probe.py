"""Standalone coordinate-only head probe; no task, KD or structural loss.

Run with --reader-checkpoint, --feature-root, --data-root and --output.
The frozen representation has seen the fold nodes during Stage I. Raw F0 is a
complementary feature set without that exposure. V_val never selects a probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.nn import functional as F

from src.data.features import FeatureStore, build_f0_matrix
from src.data.pairs import NegativeSampler
from src.data.struct_coords import FIELD_SLICES, StructCoordinateTable, coordinate_statistics
from src.data.training_sampler import build_training_corpus
from src.model.egostitch.classifier.coord_gen import (
    CONTINUOUS_INDEX,
    RELATION_CONTINUOUS_INDEX,
    CoordGenConfig,
    CoordinateGenerator,
    distance_class_targets,
)
from src.model.egostitch.classifier.topo_prompt import V3_1TopoPrompt


def fold_surfaces(pairs: list[tuple[str, str]], heldout: set[str]) -> dict[str, np.ndarray]:
    """Partition rows; fitting is permitted only on the zero-heldout surface."""
    count = np.array([int(u in heldout) + int(v in heldout) for u, v in pairs])
    return {
        name: np.flatnonzero(count == n)
        for n, name in enumerate(("in_fold", "one_heldout", "two_heldout"))
    }


def _r2(prediction: torch.Tensor, truth: torch.Tensor) -> float | None:
    if truth.shape[0] < 2:
        return None
    denominator = ((truth - truth.mean(dim=0)) ** 2).sum().item()
    return 1.0 - float(((prediction - truth) ** 2).sum()) / denominator if denominator > 0 else None


def fit_metrics(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    distance_logits: torch.Tensor,
    raw: torch.Tensor,
    endpoint_mask: torch.Tensor,
) -> dict[str, float | int | None]:
    """Endpoint R² uses only the explicitly selected endpoints of each pair."""
    pred_end = torch.stack(
        [prediction[:, FIELD_SLICES[k]] for k in ("endpoint_u", "endpoint_v")], dim=1
    )[endpoint_mask]
    true_end = torch.stack(
        [truth[:, FIELD_SLICES[k]] for k in ("endpoint_u", "endpoint_v")], dim=1
    )[endpoint_mask]
    return {
        "rows": len(raw),
        "endpoint_count": len(true_end),
        "endpoint_r2": _r2(pred_end, true_end),
        "relation_r2": _r2(
            prediction[:, list(RELATION_CONTINUOUS_INDEX)],
            truth[:, list(RELATION_CONTINUOUS_INDEX)],
        ),
        "context_r2": _r2(
            prediction[:, FIELD_SLICES["context"]], truth[:, FIELD_SLICES["context"]]
        ),
        "distance_accuracy": float(
            (distance_logits.argmax(1) == distance_class_targets(raw)).float().mean()
        )
        if len(raw)
        else None,
    }


def _continuous(parts: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([parts[k] for k in ("endpoint_u", "endpoint_v", "relation", "context")], 1)


def main(argv: list[str] | None = None) -> None:
    """Fit twelve independent heads and write all surfaces plus heldout-only winner."""
    from src.score_universe import _load_checkpoint, _load_val_region_split, _read_pairs_tsv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader-checkpoint", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--strategy", default="breadth_first")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    device = torch.device(args.device)
    reader, family, checkpoint_id = _load_checkpoint(args.reader_checkpoint)
    if family != "v3_1_topo_prompt":
        raise ValueError("probe reader must be a Stage I checkpoint")
    reader = cast(V3_1TopoPrompt, reader)
    reader.to(device).eval().requires_grad_(False)
    split = _load_val_region_split(args.data_root, args.strategy)
    store = FeatureStore(args.feature_root)
    available = store.node_ids
    train_nodes = sorted(split.train_nodes & available)
    graph = split.build_training_graph()
    global_pairs, _ = _read_pairs_tsv(args.data_root / "benchmark_2025_neurips/positive_edges.txt")
    sampler = NegativeSampler(train_nodes, dict(graph.degree()), frozenset(global_pairs))
    corpus = build_training_corpus(
        sorted((u, v) for u, v in split.training_positives if u in available and v in available),
        sampler,
        negative_ratio=5,
        seed=args.seed,
        epochs=args.epochs,
    )
    heldout = set(
        np.random.default_rng(args.seed)
        .choice(train_nodes, size=max(1, int(len(train_nodes) * 0.1)), replace=False)
        .tolist()
    )
    surfaces = fold_surfaces(corpus.pairs, heldout)
    if not len(surfaces["in_fold"]):
        raise ValueError("empty probe fitting fold")
    train_raw = StructCoordinateTable(graph).coords(corpus.pairs)
    val_pairs = [(u, v) for u, v in split.val_cls_pairs if u in available and v in available]
    val_raw = StructCoordinateTable(split.build_g_val_simple()).coords(val_pairs)
    # New heads get statistics from their fitting rows only, including raw F0.
    mean, std = coordinate_statistics(train_raw[surfaces["in_fold"]])
    raw = torch.from_numpy(np.concatenate([train_raw, val_raw]))
    truth = (raw - torch.from_numpy(mean)) / torch.from_numpy(std)
    all_pairs = corpus.pairs + val_pairs
    surfaces["v_val"] = np.arange(len(corpus.pairs), len(all_pairs))
    nodes = sorted((split.train_nodes | split.v_val) & available)
    f0, node_index = build_f0_matrix(store, nodes)
    pooled = []
    with torch.no_grad():
        for node in nodes:
            tokens = store.load_tokens(node).to(device).unsqueeze(0)
            lengths = torch.tensor([tokens.shape[1]], device=device)
            pooled.append(
                CoordinateGenerator.pool(reader.encoder(tokens, lengths), lengths).cpu()[0]
            )
    features = {"raw_f0": f0, "frozen_pooled": torch.stack(pooled)}
    pair_indices = torch.tensor([[node_index[u], node_index[v]] for u, v in all_pairs])
    endpoint_mask = torch.tensor([[u in heldout, v in heldout] for u, v in all_pairs])
    endpoint_mask[surfaces["in_fold"]] = True
    endpoint_mask[surfaces["v_val"]] = True
    results = []
    for feature_name, matrix in features.items():
        if matrix.shape[1] % 2:
            raise ValueError("coordinate head expects an even pooled feature width")
        matrix = matrix.to(device)
        for dropout in (0.1, 0.3, 0.5):
            for decay in (0.05, 0.1):
                torch.manual_seed(args.seed)
                cfg = CoordGenConfig(
                    hidden=512,
                    endpoint_hidden=256,
                    endpoint_dropout=dropout,
                    endpoint_weight_decay=decay,
                )
                head = CoordinateGenerator(matrix.shape[1] // 2, cfg).to(device)
                optimizer = torch.optim.AdamW(
                    [
                        {"params": head.endpoint_head.parameters(), "weight_decay": decay},
                        {"params": head.pair_head.parameters(), "weight_decay": 0.05},
                    ],
                    lr=3e-4,
                )
                rng = np.random.default_rng(args.seed)
                for epoch in range(1, args.epochs + 1):
                    head.train()
                    rows = corpus.epoch_rows[epoch]
                    rows = rows[np.isin(rows, surfaces["in_fold"])]
                    for batch in np.array_split(
                        rng.permutation(rows), max(1, int(np.ceil(len(rows) / args.batch_size)))
                    ):
                        if not len(batch):
                            continue
                        idx = pair_indices[batch].to(device)
                        parts = head(matrix[idx[:, 0]], matrix[idx[:, 1]])
                        loss = F.smooth_l1_loss(
                            _continuous(parts), truth[batch][:, list(CONTINUOUS_INDEX)].to(device)
                        )
                        loss = loss + F.cross_entropy(
                            parts["distance_logits"], distance_class_targets(raw[batch]).to(device)
                        )
                        optimizer.zero_grad()
                        if not torch.isfinite(loss):
                            raise ValueError("non-finite probe loss")
                        loss.backward()  # type: ignore[no-untyped-call]
                        optimizer.step()
                head.eval()
                predictions, distances = [], []
                with torch.no_grad():
                    for start in range(0, len(pair_indices), args.batch_size):
                        pair_batch = pair_indices[start : start + args.batch_size].to(device)
                        parts = head(matrix[pair_batch[:, 0]], matrix[pair_batch[:, 1]])
                        predictions.append(_continuous(parts).cpu())
                        distances.append(parts["distance_logits"].cpu())
                prediction = torch.zeros_like(truth)
                prediction[:, list(CONTINUOUS_INDEX)] = torch.cat(predictions)
                distance = torch.cat(distances)
                report = {
                    name: fit_metrics(
                        prediction[rows],
                        truth[rows],
                        distance[rows],
                        raw[rows],
                        endpoint_mask[rows],
                    )
                    for name, rows in surfaces.items()
                }
                held_rows = np.concatenate([surfaces["one_heldout"], surfaces["two_heldout"]])
                held_metric = fit_metrics(
                    prediction[held_rows],
                    truth[held_rows],
                    distance[held_rows],
                    raw[held_rows],
                    endpoint_mask[held_rows],
                )["endpoint_r2"]
                results.append(
                    {
                        "features": feature_name,
                        "dropout": dropout,
                        "weight_decay": decay,
                        "heldout_endpoint_r2": held_metric,
                        "surfaces": report,
                    }
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                valid = [
                    r
                    for r in results
                    if r["features"] == "frozen_pooled" and r["heldout_endpoint_r2"] is not None
                ]
                args.output.write_text(
                    json.dumps(
                        {
                            "status": "complete" if len(results) == 12 else "running",
                            "checkpoint_id": checkpoint_id,
                            "heldout_nodes": sorted(heldout),
                            "seed": args.seed,
                            "epochs": args.epochs,
                            "fitting_rows": len(surfaces["in_fold"]),
                            "representation_exposure": (
                                "Stage I encoder saw probe holdout nodes; raw F0 did not"
                            ),
                            "selection": "heldout endpoint R2 only; V_val is diagnostic",
                            "winner": max(
                                valid, key=lambda r: cast(float, r["heldout_endpoint_r2"])
                            )
                            if valid
                            else None,
                            "results": results,
                        },
                        indent=2,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()

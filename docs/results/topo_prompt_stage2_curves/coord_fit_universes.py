"""Coordinate fit of a published v1 coord_gen checkpoint on training, V_val and test rows.

Same generator, same standardisation, three universes, each with its own true graph
(queried edge removed by StructCoordinateTable). Rows are label-balanced samples so the
R² denominators are comparable across universes. Inference only; nothing is written
under the run directory except the JSON named by --output.

Usage (H20): PYTHONPATH=. .venv/bin/python outputs/logs/coord_fit_universes.py \
  --run coord_gen_full --per-label 3000 --output outputs/split_seed42/coord_gen_full/coord_fit_universes.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data.pairs import TokenPairDataset, collate_token_pairs
from src.data.struct_coords import (
    CONTEXT_NAMES,
    ENDPOINT_NAMES,
    FIELD_SLICES,
    RELATION_NAMES,
    StructCoordinateTable,
)
from src.eval.graph_metrics import strip_self_loops
from src.experiments.score_coord_gen_v1_diagnostic import load_v1_diagnostic
from src.model.egostitch.classifier.coord_gen import FIELD_CONTINUOUS_INDEX, distance_class_targets
from src.train_b0 import _dynamic_training_corpus, _val_cls_rows, assemble_data, load_config

parser = argparse.ArgumentParser()
parser.add_argument("--run", default="coord_gen_full")
parser.add_argument("--checkpoint", type=Path, default=None)
parser.add_argument("--per-label", type=int, default=3000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

t0 = time.time()
cfg = load_config(Path(f"configs/split_seed42/{args.run}.yaml"))
assembled = assemble_data(cfg)
corpus = _dynamic_training_corpus(cfg, assembled)
exclude = assembled.exclude_nodes
rng = np.random.default_rng(args.seed)


def balanced(pairs, labels, k):
    labels = np.asarray(labels, dtype=np.int64)
    keep = []
    for label in (1, 0):
        idx = np.flatnonzero(labels == label)
        take = idx if len(idx) <= k else rng.choice(idx, size=k, replace=False)
        keep.extend(int(i) for i in take)
    keep.sort()
    return [pairs[i] for i in keep], labels[keep]


train_rows = np.asarray(corpus.epoch_rows[1])
train_pairs_all = [corpus.pairs[i] for i in train_rows]
train_labels_all = [corpus.labels[i] for i in train_rows]
val_pairs_all, val_labels_all = _val_cls_rows(assembled.val_split, exclude)
test_lp = assembled.benchmark.split.test_pairs
test_keep = [i for i, (u, v) in enumerate(test_lp.pairs) if u not in exclude and v not in exclude]
test_pairs_all = [test_lp.pairs[i] for i in test_keep]
test_labels_all = [int(test_lp.labels[i]) for i in test_keep]

universes = {
    "train": (assembled.val_split.build_training_graph(), train_pairs_all, train_labels_all),
    "val": (assembled.val_split.build_g_val_simple(), val_pairs_all, val_labels_all),
    "test": (strip_self_loops(assembled.benchmark.split.test_graph), test_pairs_all, test_labels_all),
}
for name, (graph, pairs, labels) in universes.items():
    missing = {n for p in pairs for n in p if n not in graph}
    if missing:
        raise RuntimeError(f"{name}: {len(missing)} pair endpoints outside the universe graph")
    print(f"{name}: graph {graph.number_of_nodes()} nodes / {graph.number_of_edges()} edges, "
          f"{len(pairs)} rows ({int(np.sum(labels))} positives) before sampling", flush=True)

checkpoint = args.checkpoint or Path(f"outputs/split_seed42/{args.run}/best.pt")
model, family, checkpoint_id = load_v1_diagnostic(checkpoint)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device).eval()
store = assembled.store


def pearson(pred, true):
    pred = pred.reshape(-1).numpy()
    true = true.reshape(-1).numpy()
    if pred.std() == 0 or true.std() == 0:
        return None
    return float(np.corrcoef(pred, true)[0, 1])


def r2(pred, true):
    sse = float(((pred - true) ** 2).sum())
    sst = float(((true - true.mean(0)) ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else None


def predict(pairs):
    lengths = [(int(store.load_tokens(u).size(0)), int(store.load_tokens(v).size(0))) for u, v in pairs]
    dataset = TokenPairDataset(pairs, None, store, lengths=lengths)
    z_hat, dist = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for start in range(0, len(pairs), 128):
            batch = collate_token_pairs([dataset[i] for i in range(start, min(start + 128, len(pairs)))])
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            z_hat.append(out["predicted_coords"].float().cpu())
            dist.append(out["distance_logits"].argmax(1).cpu())
    return torch.cat(z_hat).double(), torch.cat(dist)


eu, ev = FIELD_SLICES["endpoint_u"], FIELD_SLICES["endpoint_v"]
relation_names = RELATION_NAMES[: len(FIELD_CONTINUOUS_INDEX["relation"])]
result = {
    "run": args.run,
    "checkpoint": str(checkpoint),
    "checkpoint_id": checkpoint_id,
    "per_label": args.per_label,
    "seed": args.seed,
    "protocol": "label-balanced sample per universe; true coordinates on each universe's own "
    "loopless graph with the queried edge removed; R2 of the standardised prediction "
    "(training statistics), per field = pooled SSE/SST over the field's continuous coordinates",
    "universes": {},
}
dump = {}
for name, (graph, pairs, labels) in universes.items():
    t1 = time.time()
    pairs, labels = balanced(pairs, labels, args.per_label)
    coords = torch.from_numpy(StructCoordinateTable(graph).coords(pairs)).float()
    z_hat, dist_pred = predict(pairs)
    z_star = model.reader.generator.standardize(coords.to(device)).cpu().double()
    dist_true = distance_class_targets(coords)
    entry = {
        "rows": len(pairs),
        "positives": int(labels.sum()),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "field_r2": {f: r2(z_hat[:, list(i)], z_star[:, list(i)]) for f, i in FIELD_CONTINUOUS_INDEX.items()},
        "endpoint_swapped_r2": r2(torch.cat([z_hat[:, eu], z_hat[:, ev]]), torch.cat([z_star[:, ev], z_star[:, eu]])),
        "distance_accuracy": float((dist_pred == dist_true).double().mean()),
        "distance_majority_accuracy": float(torch.bincount(dist_true).max().double() / len(pairs)),
        "coordinate_r2": {},
        "coordinate_pearson": {},
        "label_r2": {},
    }
    generator = model.reader.generator
    mean = generator.coord_mean.detach().cpu().double()
    std = generator.coord_std.detach().cpu().double()
    dump[f"{name}_pred_std"] = z_hat.numpy().astype(np.float32)
    dump[f"{name}_true_std"] = z_star.numpy().astype(np.float32)
    dump[f"{name}_pred_raw"] = (z_hat * std + mean).numpy().astype(np.float32)
    dump[f"{name}_true_raw"] = coords.double().numpy().astype(np.float32)
    dump[f"{name}_labels"] = labels.astype(np.int8)
    dump[f"{name}_dist_pred"] = dist_pred.numpy().astype(np.int8)
    dump[f"{name}_dist_true"] = dist_true.numpy().astype(np.int8)
    for j, cname in enumerate(ENDPOINT_NAMES):
        pred = torch.cat([z_hat[:, eu][:, j : j + 1], z_hat[:, ev][:, j : j + 1]])
        true = torch.cat([z_star[:, eu][:, j : j + 1], z_star[:, ev][:, j : j + 1]])
        entry["coordinate_r2"][f"endpoint.{cname}"] = r2(pred, true)
        entry["coordinate_pearson"][f"endpoint.{cname}"] = pearson(pred, true)
    for j, cname in zip(FIELD_CONTINUOUS_INDEX["relation"], relation_names):
        entry["coordinate_r2"][f"relation.{cname}"] = r2(z_hat[:, j : j + 1], z_star[:, j : j + 1])
        entry["coordinate_pearson"][f"relation.{cname}"] = pearson(z_hat[:, j], z_star[:, j])
    for j, cname in zip(FIELD_CONTINUOUS_INDEX["context"], CONTEXT_NAMES):
        entry["coordinate_r2"][f"context.{cname}"] = r2(z_hat[:, j : j + 1], z_star[:, j : j + 1])
        entry["coordinate_pearson"][f"context.{cname}"] = pearson(z_hat[:, j], z_star[:, j])
    for label in (1, 0):
        mask = torch.from_numpy(labels == label)
        entry["label_r2"]["positive" if label else "negative"] = {
            f: r2(z_hat[mask][:, list(i)], z_star[mask][:, list(i)]) for f, i in FIELD_CONTINUOUS_INDEX.items()
        }
    result["universes"][name] = entry
    print(name, json.dumps({k: entry[k] for k in ("rows", "positives", "field_r2", "distance_accuracy")}),
          f"{time.time() - t1:.0f}s", flush=True)

args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2))
dump["coord_mean"] = mean.numpy().astype(np.float32)
dump["coord_std"] = std.numpy().astype(np.float32)
np.savez_compressed(args.output.with_suffix(".npz"), **dump)
print("wrote", args.output, f"total {time.time() - t0:.0f}s", flush=True)

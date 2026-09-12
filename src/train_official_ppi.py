"""Train official sequence classifiers on the common node-held-out benchmark."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel

from src.baselines.official_ppi import OfficialPPI, PairFeatures, score_pairs
from src.data.pairs import NegativeSampler
from src.data.training_sampler import enumerate_edge_stream
from src.data.val_region import val_ball_union_universe
from src.eval.checkpoint_selection import SELECTION_RULE, CheckpointCandidate, select_checkpoint
from src.eval.val_topology import build_val_topology_reference, val_region_topology_metrics
from src.score_universe import _load_val_region_split

logger = logging.getLogger(__name__)


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def distributed_scores(
    model: OfficialPPI, features: PairFeatures, pairs: list[tuple[str, str]], rank: int, world: int
) -> np.ndarray:
    """Gather disjoint contiguous score shards back in canonical row order."""
    start, end = len(pairs) * rank // world, len(pairs) * (rank + 1) // world
    scores = score_pairs(model, features, pairs[start:end])
    if world == 1:
        return scores
    pieces: list[Any] = [None] * world
    dist.all_gather_object(pieces, scores)
    return np.concatenate(pieces)


def fit_precision(
    model: OfficialPPI,
    features: PairFeatures,
    rows: list[tuple[str, str, int]],
    rank: int,
    world: int,
) -> None:
    """Fit TUnA covariance on this epoch's training rows, with frozen network weights."""
    if model.method != "tuna":
        return
    model.eval()
    gp = model.core.gp_layer
    gp.reset_covariance()
    pairs = [(a, b) for a, b, _ in rows[rank::world]]
    with torch.no_grad():
        for start in range(0, len(pairs), model.batch_size):
            model(*features.batch(pairs[start : start + model.batch_size]), update_precision=True)
        if world > 1:
            dist.all_reduce(gp.precision)
        if rank == 0:
            gp.train(False)
        if world > 1:
            dist.broadcast(gp.covariance, src=0)
        gp.fitted = True


def run(config: dict[str, Any], rank: int, world: int, device: torch.device) -> None:
    """Train, select on V_val, and publish the frozen scoring state."""
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    model = OfficialPPI(**config["model"]).to(device)
    features = PairFeatures(Path(config["pack_dir"]), device, model.max_length)
    split = _load_val_region_split(Path(config["data_root"]), "breadth_first")
    train_nodes = sorted(split.train_nodes & features.position.keys())
    positives = sorted(
        (a, b)
        for a, b in split.training_positives
        if a in features.position and b in features.position
    )
    graph = split.build_training_graph()
    sampler = NegativeSampler(train_nodes, dict(graph.degree()), frozenset(positives))
    val_pairs = list(split.val_cls_pairs)
    val_labels = np.asarray(split.val_cls_labels)
    union = val_ball_union_universe(split)
    reference = build_val_topology_reference(split)
    topo_pairs = [
        (reference.nodes[int(u)], reference.nodes[int(v)])
        for u, v in zip(union.u_idx, union.v_idx, strict=True)
    ]
    wrapped = (
        DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False)
        if world > 1
        else model
    )
    optim_cfg = config["optim"]
    optimizer: Any
    if model.method == "tuna":
        from src.baselines.vendor.tuna.lookahead import Lookahead

        groups = [
            {
                "params": [p for name, p in model.named_parameters() if "bias" not in name],
                "weight_decay": optim_cfg["weight_decay"],
            },
            {
                "params": [p for name, p in model.named_parameters() if "bias" in name],
                "weight_decay": 0.0,
            },
        ]
        inner = torch.optim.Adam(groups, lr=optim_cfg["lr"])
        optimizer = Lookahead(inner, alpha=0.8, k=5)  # type: ignore[no-untyped-call]
        scheduler = torch.optim.lr_scheduler.StepLR(inner, step_size=2, gamma=0.93)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=optim_cfg["lr"], weight_decay=optim_cfg["weight_decay"]
        )
        scheduler = None
    candidates: list[CheckpointCandidate] = []
    best_loss, patience = math.inf, 0
    if rank == 0:
        _json(output / "config.json", config)
        _json(
            output / "split.json",
            {
                "split_seed": split.params.split_seed,
                "root": split.region_seeds,
                "train_nodes": len(train_nodes),
                "train_positives": len(positives),
                "val_nodes": len(split.v_val),
            },
        )
        (output / "checkpoints").mkdir(exist_ok=True)
        logger.info(
            "method=%s train_nodes=%d positives=%d V_val=%d world=%d",
            model.method,
            len(train_nodes),
            len(positives),
            len(split.v_val),
            world,
        )
    for epoch in range(1, optim_cfg["epochs"] + 1):
        rows = enumerate_edge_stream(
            positives,
            sampler,
            negative_ratio=5,
            seed=config["seed"],
            epoch=epoch,
            rank=0,
            world_size=1,
        )
        wrapped.train()
        loss_sum = torch.zeros((), device=device)
        global_batch = model.batch_size * world
        for start in range(0, len(rows), global_batch):
            stop = min(start + global_batch, len(rows))
            batch = rows[start + rank : stop : world]
            count = len(batch)
            if not batch:
                batch = rows[:1]
            a, b, labels = zip(*batch, strict=True)
            inputs = features.batch(list(zip(a, b, strict=True)))
            logits = wrapped(*inputs)
            target = torch.tensor(labels, dtype=torch.float32, device=device)
            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target, pos_weight=logits.new_tensor(5.0), reduction="sum"
            )
            loss = losses * (world / (stop - start)) if count else losses * 0
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += losses.detach() if count else 0
            if rank == 0 and (start == 0 or start % (global_batch * 250) == 0):
                logger.info(
                    "epoch=%d rows=%d/%d train_batch_loss=%.6f",
                    epoch,
                    stop,
                    len(rows),
                    float(loss.detach()),
                )
        if scheduler is not None:
            scheduler.step()
        if world > 1:
            dist.all_reduce(loss_sum)
        fit_precision(model, features, rows, rank, world)
        scores = distributed_scores(model, features, val_pairs, rank, world)
        val_loss = float(
            torch.nn.functional.binary_cross_entropy_with_logits(
                torch.from_numpy(scores), torch.tensor(val_labels, dtype=torch.float32)
            )
        )
        auprc = float(average_precision_score(val_labels, scores))
        due = epoch == 1 or epoch % config["topology_every"] == 0 or epoch == optim_cfg["epochs"]
        if due:
            topo_scores = distributed_scores(model, features, topo_pairs, rank, world)
            if rank == 0:
                topology = val_region_topology_metrics(
                    u_idx=union.u_idx, v_idx=union.v_idx, logits=topo_scores, reference=reference
                )
                candidates.append(CheckpointCandidate(epoch, auprc, topology.metrics))
                payload = {
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "model_family": "official_ppi",
                    "model_config": config["model"],
                    "epoch": epoch,
                    "selection_rule": SELECTION_RULE,
                    "val_threshold_transfer": {
                        "n_val": len(split.v_val),
                        "threshold": topology.threshold,
                    },
                    "topology": dataclasses.asdict(topology),
                    "val_auprc": auprc,
                }
                torch.save(payload, output / "checkpoints" / f"epoch-{epoch:04d}.pt")
                logger.info("epoch=%d topology=%s", epoch, dataclasses.asdict(topology))
            if world > 1:
                dist.barrier()
        if rank == 0:
            row = {
                "epoch": epoch,
                "train_loss": float(loss_sum) / len(rows),
                "val_task_loss": val_loss,
                "val_auprc": auprc,
                "val_auroc": float(roc_auc_score(val_labels, scores)),
            }
            with (output / "history.jsonl").open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            logger.info("validation %s", row)
        if not math.isfinite(val_loss):
            raise ValueError("non-finite validation loss")
        if val_loss < best_loss:
            best_loss, patience = val_loss, 0
        else:
            patience += 1
        if due and patience >= optim_cfg["patience"]:
            break
    if rank == 0:
        selected = select_checkpoint(candidates)
        if selected is None:
            raise RuntimeError("no selectable checkpoint")
        payload = torch.load(
            output / "checkpoints" / f"epoch-{selected.epoch:04d}.pt", weights_only=True
        )
        torch.save(payload, output / "best.pt")
        _json(
            output / "selection.json",
            {
                "selected_epoch": selected.epoch,
                "selection_rule": SELECTION_RULE,
                "candidates": [dataclasses.asdict(c) for c in candidates],
            },
        )
        _json(output / "complete.json", {"status": "complete", "selected_epoch": selected.epoch})
    if world > 1:
        dist.barrier()


def main() -> None:
    """Run a CPU smoke or the runner's auto-sized torchrun worker group."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    try:
        run(config, rank, world, device)
    except Exception as exc:
        path = Path(config["output_dir"])
        path.mkdir(parents=True, exist_ok=True)
        _json(path / f"failure_rank{rank}.json", {"error": str(exc)})
        if rank == 0:
            _json(path / "failure.json", {"error": str(exc)})
        raise
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

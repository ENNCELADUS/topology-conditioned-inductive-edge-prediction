"""Two-rank CPU DDP smoke driver for the ``v3_1_motif_prompt`` family.

Every rank compiles the same templates, installs the same training mean, builds
the same Stage II model, runs one synchronised optimiser step under the epoch-1
trainability mask, and writes the digests `tests/test_motif_prompt_ddp.py`
compares. Run through ``torch.distributed.run --standalone --nproc_per_node=2``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.distributed as dist

os.environ.setdefault("ACCELERATE_USE_CPU", "true")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.egostitch.classifier.motif_prompt import (  # noqa: E402
    TEMPLATE_KEY,
    TEMPLATE_MASK_KEY,
    V3_1MotifPrompt,
)
from src.train_b0 import (  # noqa: E402
    MotifTemplateRows,
    _motif_stream_counts,
    _motif_stream_terms,
    _set_motif_prompt_training_stage,
    build_ddp_accelerator,
)

_BASE_CONFIG: dict[str, object] = {
    "input_dim": 4,
    "d_model": 8,
    "encoder_layers": 1,
    "cross_attn_layers": 2,
    "n_heads": 2,
    "mlp_head": {"hidden_dims": [8], "dropout": 0.0},
    "regularization": {"dropout": 0.0},
    "mixing": {"mode": "bidirectional_cross"},
}
_MOTIF_BLOCK: dict[str, object] = {
    "stage": "two",
    "base_checkpoint": "base.pt",
    "bundle_checkpoint": "bundle.pt",
    "width": 16,
    "slots_per_field": 2,
    "reader": {"layers": 1, "dim": 16, "heads": 4, "rrwp_k": 2},
}


def _digest(tensor: torch.Tensor) -> str:
    """SHA-256 over one tensor's contiguous float32 bytes."""
    payload = tensor.detach().to(torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _training_graph() -> nx.Graph:
    """A deterministic 12-node loopless training graph with closures and bridges."""
    graph = nx.Graph()
    nodes = [f"node_{i:03d}" for i in range(12)]
    graph.add_nodes_from(nodes)
    for i, node in enumerate(nodes):
        graph.add_edge(node, nodes[(i + 1) % 12])
        graph.add_edge(node, nodes[(i + 5) % 12])
    return graph


def _pairs(graph: nx.Graph) -> list[tuple[str, str]]:
    """Every training row, in row-id order, with one self pair."""
    nodes = sorted(graph.nodes)
    rows = [(u, v) for index, u in enumerate(nodes) for v in nodes[index + 1 :]][:24]
    return [*rows, (nodes[0], nodes[0])]


def _batch(rows: list[int]) -> dict[str, torch.Tensor]:
    """One rank-local batch of token pairs, keyed by ``_row_id``."""
    generator = torch.Generator().manual_seed(11)
    count = len(rows)
    return {
        "emb_a": torch.randn(count, 6, 4, generator=generator),
        "emb_b": torch.randn(count, 5, 4, generator=generator),
        "len_a": torch.full((count,), 6, dtype=torch.int64),
        "len_b": torch.full((count,), 5, dtype=torch.int64),
        "label": torch.tensor([row % 2 for row in rows], dtype=torch.float32),
        "_row_id": torch.tensor(rows, dtype=torch.int64),
    }


def main() -> None:
    """Run one synchronised step on every rank and write its report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    dist.init_process_group("gloo")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The production DDP settings, `find_unused_parameters=False` included: a
    # trainable parameter this arm leaves out of the forward would abort here.
    accelerator = build_ddp_accelerator("no")
    graph = _training_graph()
    pairs = _pairs(graph)
    rows = MotifTemplateRows(
        train_graph=graph,
        train_pairs=pairs,
        stats_rows=np.arange(len(pairs)),
        device=accelerator.device,
        seed=47,
    )

    torch.manual_seed(0)
    model = V3_1MotifPrompt(base=_BASE_CONFIG, motif_prompt=_MOTIF_BLOCK)
    rows.install(model)
    model.initialize_teacher()
    with torch.no_grad():
        model.adapter.gates.fill_(0.3)

    optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(1e-3, 1e-4, 1e-2))
    prepared, optimizer = accelerator.prepare(model, optimizer)
    # The production schedule: the warm-up hook reads it to restore the
    # interface LR when the group opens, so the smoke drives the real pair.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[float(group["max_lr"]) for group in optimizer.param_groups],
        total_steps=15,
        cycle_momentum=False,
    )
    _set_motif_prompt_training_stage(prepared, optimizer, scheduler, epoch=1)

    local_rows = list(range(rank, len(pairs), 2))
    batch = _batch(local_rows)
    rows.attach_train(batch)
    output = prepared(batch)
    task_term = (
        {"slot": output["slot_loss_rows"], "topo": output["topo_loss_rows"]},
        batch[TEMPLATE_MASK_KEY],
    )
    # The trainer's own reduction: the per-stream mean divides by the count
    # summed over ranks, never by this rank's own (spec section 7.5).
    counts = torch.tensor(_motif_stream_counts(task=task_term, struct=None), dtype=torch.float64)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    slot, topo = _motif_stream_terms(
        task=task_term,
        struct=None,
        like=output["loss"],
        global_counts=[float(value) for value in counts.tolist()],
        world_size=dist.get_world_size(),
    )
    loss = output["loss"] + model.cfg.w_slot * slot + model.cfg.w_topo * topo
    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()

    interface_lr = next(
        float(group["lr"]) for group in optimizer.param_groups if group.get("name") == "interface"
    )
    report = {
        "template_digest": _digest(rows.train),
        "batch_digest": _digest(batch[TEMPLATE_KEY]),
        "mean_digest": _digest(rows.mean),
        "param_digest": _digest(
            torch.cat([p.detach().reshape(-1) for p in model.parameters() if p.requires_grad])
        ),
        "interface_lr": interface_lr,
        "interface_open": bool(model.interface_open),
        "self_row_masked": float(batch[TEMPLATE_MASK_KEY].min().item()),
        "loss": float(loss.detach().item()),
    }
    (args.output_dir / f"rank-{rank}.json").write_text(json.dumps(report))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

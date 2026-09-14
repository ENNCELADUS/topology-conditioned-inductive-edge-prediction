"""Actual worker parity across uneven CPU/Gloo distributed batches."""

from pathlib import Path

import networkx as nx
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from src.baselines.l3ppi import L3PPI
from src.data.l3_patterns import L3PatternCache
from src.train_l3ppi import cpu_state, train_epoch
from torch.nn.parallel import DistributedDataParallel


def _train(rank: int, world: int, output: str) -> None:
    """Train identical data/initialization, saving each rank independently."""
    torch.set_num_threads(1)
    torch.manual_seed(19)
    model = L3PPI(
        {
            "input_dim": 3,
            "d_model": 4,
            "encoder_layers": 1,
            "n_heads": 1,
            "regularization": {"dropout": 0.0},
        },
        k=2,
        surrogate_hidden=4,
        gate_hidden=4,
        dropout=0.0,
    )
    model.prompt.requires_grad_(False)
    model.gate.requires_grad_(False)
    model.surrogate.requires_grad_(True)
    wrapped = DistributedDataParallel(model.surrogate) if world > 1 else model.surrogate
    optimizer = torch.optim.SGD(model.surrogate.parameters(), lr=0.02)
    graph = nx.Graph([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")])
    features = {node: torch.randn(8) for node in sorted(graph)}
    rows = [("a", "d", 1), ("a", "c", 0), ("b", "c", 1), ("b", "d", 0)]
    metrics = train_epoch(
        model=model,
        wrapped=wrapped,
        optimizer=optimizer,
        rows=rows,
        features=features,
        patterns=L3PatternCache(graph, set(graph)),
        phase="surrogate",
        config={"batch_size": 3, "path_weight": 0.3, "gamma": 3.0},
        tau=1.0,
        rank=rank,
        world=world,
        device=torch.device("cpu"),
    )
    torch.save(
        {"model_state": cpu_state(model), "metrics": metrics},
        Path(output) / f"world{world}-rank{rank}.pt",
    )


def _distributed_worker(rank: int, rendezvous: str, output: str) -> None:
    """Initialize a two-process CPU group without a fixed port."""
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        _train(rank, 2, output)
    finally:
        dist.destroy_process_group()


def test_surrogate_ddp_matches_single_rank_with_empty_tail(tmp_path: Path) -> None:
    previous_threads = torch.get_num_threads()
    try:
        _train(0, 1, str(tmp_path))
        mp.spawn(  # type: ignore[attr-defined, no-untyped-call]
            _distributed_worker,
            args=((tmp_path / "rendezvous").as_uri(), str(tmp_path)),
            nprocs=2,
            join=True,
        )
    finally:
        torch.set_num_threads(previous_threads)
    expected = torch.load(tmp_path / "world1-rank0.pt", weights_only=True)
    for rank in range(2):
        actual = torch.load(tmp_path / f"world2-rank{rank}.pt", weights_only=True)
        for key, tensor in expected["model_state"].items():
            torch.testing.assert_close(actual["model_state"][key], tensor, atol=1e-7, rtol=1e-6)
        for key in ("train_task_loss", "train_path_loss", "train_loss"):
            torch.testing.assert_close(
                actual["metrics"][key], expected["metrics"][key], atol=1e-7, rtol=1e-6
            )

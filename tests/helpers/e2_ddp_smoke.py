from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("ACCELERATE_USE_CPU", "true")

from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.distributed_pairs import build_distributed_epoch_plan  # noqa: E402
from src.train_b0 import (  # noqa: E402
    Config,
    DataConfig,
    EvalConfig,
    ModelConfig,
    OptimConfig,
    _evaluate_distributed,
    train_ddp_loop,
    write_outputs,
)


class _TinyPairModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scorer = nn.Linear(1, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        logits = self.scorer(batch["x"]).squeeze(-1)
        return {
            "logits": logits,
            "loss": F.binary_cross_entropy_with_logits(logits, batch["label"].float()),
        }


class _BatchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, batches: list[dict[str, torch.Tensor]]) -> None:
        self._batches = batches

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self._batches[index]


def _batch(row_ids: list[int], *, global_count: int) -> dict[str, torch.Tensor]:
    values = torch.tensor(row_ids, dtype=torch.float32).unsqueeze(-1) / 8.0
    labels = torch.tensor([row_id % 2 for row_id in row_ids], dtype=torch.float32)
    return {
        "x": values,
        "label": labels,
        "_row_id": torch.tensor(row_ids, dtype=torch.int64),
        "_local_pair_count": torch.tensor(len(row_ids), dtype=torch.int64),
        "_global_pair_count": torch.tensor(global_count, dtype=torch.int64),
    }


def _run_train(output_dir: Path, rank: int, *, fail_epoch_io: bool = False) -> None:
    accelerator = Accelerator(cpu=True)
    model = _TinyPairModel()
    rank_rows = ([0, 1], [4, 5]) if rank == 0 else ([2, 3], [6, 7])
    train_batches = [_batch(list(rows), global_count=4) for rows in rank_rows]
    val_rows = [0, 1] if rank == 0 else [2, 3]
    val_loader = DataLoader(
        _BatchDataset([_batch(val_rows, global_count=4)]),
        batch_size=None,
        num_workers=0,
    )
    cfg = Config(
        model=ModelConfig(
            family="f0_mlp",
            config={"input_dim": 1, "hidden_dims": [2], "dropout": 0.0},
        ),
        data=DataConfig(
            root=Path("data"),
            strategy="breadth_first",
            negative_ratio=1,
            token_budget=64,
            batch_pairs=2,
            num_workers=0,
            f0_cache=Path("unused.pt"),
            expected_missing_features=[],
        ),
        optim=OptimConfig(
            lr=1.0e-2,
            weight_decay=0.0,
            epochs=2,
            warmup_steps=1,
            grad_clip=1.0,
        ),
        eval=EvalConfig(patience=8, eval_every=1),
        seed=47,
        output_dir=output_dir / "artifacts",
        mixed_precision="no",
    )

    if fail_epoch_io and accelerator.is_main_process:
        import src.train_b0 as train_b0_module

        def fail_save(payload: dict[str, object], path: Path) -> None:
            raise OSError("injected rank-zero disk failure")

        train_b0_module._torch_save_atomic = fail_save

    try:
        result = train_ddp_loop(
            model,
            lambda epoch: DataLoader(
                _BatchDataset(train_batches),
                batch_size=None,
                num_workers=0,
            ),
            val_loader,
            cfg,
            accelerator,
            warmup_steps=1,
            artifact_dir=cfg.output_dir,
            evaluate_fn=lambda prepared, loader, acc: _evaluate_distributed(
                prepared,
                loader,
                acc,
                expected_row_ids=torch.arange(4).numpy(),
            ),
        )
    except RuntimeError as error:
        if not fail_epoch_io or "injected rank-zero disk failure" not in str(error):
            raise
        (output_dir / f"io-failure-rank-{rank}.json").write_text(json.dumps({"error": str(error)}))
        return
    if fail_epoch_io:
        raise AssertionError("injected rank-zero artifact failure did not propagate")

    local_weight = model.scorer.weight.detach().reshape(1)
    gathered_weights = accelerator.gather(local_weight)
    weights_synchronized = bool(torch.allclose(gathered_weights, gathered_weights[0]))
    if accelerator.is_main_process:
        write_outputs(
            result,
            cfg,
            {"input_dim": 1, "hidden_dims": [2], "dropout": 0.0},
            {},
        )
    dist.barrier()
    metric_rows = [
        json.loads(line)
        for line in (cfg.output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    candidate_names = sorted(
        path.name for path in (cfg.output_dir / "checkpoints").glob("epoch-*.pt")
    )
    summary = {
        "epochs_completed": result.runtime_profile["epochs_completed"],
        "validations_completed": result.runtime_profile["validations_completed"],
        "training_coverage_exact": result.runtime_profile["training_coverage_exact"],
        "validation_coverage_exact": result.runtime_profile["validation_coverage_exact"],
        "best_epoch": result.best_epoch,
        "weights_synchronized": weights_synchronized,
        "metric_epochs": [row["epoch"] for row in metric_rows],
        "candidate_names": candidate_names,
    }
    (output_dir / f"train-rank-{rank}.json").write_text(json.dumps(summary))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("plan", "train", "train-io-failure"), default="plan")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("gloo")
    if args.mode in {"train", "train-io-failure"}:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _run_train(args.output_dir, rank, fail_epoch_io=args.mode == "train-io-failure")
        dist.destroy_process_group()
        return
    plan = build_distributed_epoch_plan(
        [(100, 100)] * 64,
        token_budget_per_rank=2048,
        max_pairs_per_rank=8,
        world_size=world_size,
        seed=47,
        epoch=1,
        shuffle=True,
    )
    row_ids = [index for spec in plan[rank] for index in spec.indices]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_path = args.output_dir / f"rank-{rank}.json.tmp"
    final_path = args.output_dir / f"rank-{rank}.json"
    temp_path.write_text(json.dumps(row_ids))
    os.replace(temp_path, final_path)
    count = torch.tensor([len(row_ids)], dtype=torch.int64)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    if int(count.item()) != 64:
        raise RuntimeError(f"expected 64 globally covered rows, got {count.item()}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

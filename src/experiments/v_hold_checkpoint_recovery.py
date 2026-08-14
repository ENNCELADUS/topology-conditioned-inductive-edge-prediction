"""Recover exact training-time V_hold rows from interrupted E2E checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal, cast

import networkx as nx
import torch
from accelerate.utils import set_seed

from src.eval.graph_metrics import MMDConfig, evaluate_assembled_graph
from src.model.egostitch.composite import EgoStitchModel
from src.model.egostitch.config import E2EConfig
from src.train_egostitch import (
    E2ECheckpointRecord,
    EgoConfig,
    EgoStitchData,
    _BatchFactory,
    _bind_feature_standardization,
    _e2e_active_groups,
    _e2e_base_lr,
    _e2e_optimizer_group_lr,
    _epoch_step_plan,
    _install_oracle_context,
    _validate_epoch,
    assemble_egostitch_data,
    build_e2e_parameter_groups,
    build_egostitch_ddp_accelerator,
    e2e_accumulation_window_sizes,
    e2e_phase_state,
    load_config,
    select_e2e_checkpoint,
)


def _parse_epochs(value: str) -> tuple[int, ...]:
    epochs = tuple(int(item) for item in value.split(","))
    if not epochs or any(epoch <= 0 for epoch in epochs) or len(set(epochs)) != len(epochs):
        raise argparse.ArgumentTypeError("epochs must be unique positive comma-separated integers")
    return epochs


def build_parser() -> argparse.ArgumentParser:
    """Build the diagnostic recovery CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=_parse_epochs, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--edge-batch", type=int, default=None)
    return parser


def _phase_and_lr(
    cfg: EgoConfig,
    data: EgoStitchData,
    model: EgoStitchModel,
    epoch: int,
    world: int,
) -> tuple[Literal["A", "B", "C"], float]:
    training = cfg.training
    assert training is not None
    _, microsteps = _epoch_step_plan(
        len(data.training_positives),
        negative_ratio=cfg.data.negative_ratio,
        edge_batch=cfg.data.edge_batch,
        world_size=world,
    )
    steps_per_epoch = len(
        e2e_accumulation_window_sizes(microsteps, cfg.optim.gradient_accumulation_steps)
    )
    total_steps = steps_per_epoch * cfg.optim.epochs
    phase = e2e_phase_state(
        epoch * steps_per_epoch - 1,
        total_steps,
        phase_a_fraction=training.phase_a_fraction,
        phase_b_fraction=training.phase_b_fraction,
    )
    groups = build_e2e_parameter_groups(model).groups
    first_group = next(name for name in ("generator", "encoder", "classifier") if groups[name])
    base_lr = _e2e_base_lr(epoch * steps_per_epoch - 1, total_steps, training)
    lr = _e2e_optimizer_group_lr(base_lr, phase, first_group, _e2e_active_groups(phase, model))
    return phase.phase, float(lr)


def recover(args: argparse.Namespace) -> None:
    """Evaluate selected interrupted-run checkpoints with the exact V_hold evaluator."""
    cfg = load_config(args.config)
    source_metadata_path = args.staging_dir / "run_metadata.json"
    cfg = replace(cfg, run_kind="diagnostic")

    accelerator = build_egostitch_ddp_accelerator(cfg.mixed_precision, find_unused_parameters=False)
    set_seed(cfg.seed)
    if cfg.data.pack_dir is None or cfg.runtime is None:
        raise ValueError("exact recovery requires data.pack_dir and runtime")
    data = assemble_egostitch_data(cfg, pack_dir=cfg.data.pack_dir)
    model = EgoStitchModel(E2EConfig.from_mapping(cfg.model.config))
    _bind_feature_standardization(model, cfg, data)
    _install_oracle_context(model, data, run_kind="diagnostic")
    model.to(accelerator.device)
    factory = _BatchFactory(
        cfg,
        model.generator_cfg,
        data,
        node_batch=cfg.runtime.token_budget,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        generator_supervision=False,
        relational_supervision=False,
    )

    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "metrics.jsonl").touch()
        (args.output_dir / "edge_metrics.jsonl").touch()
    accelerator.wait_for_everyone()

    records: list[E2ECheckpointRecord] = []
    checkpoint_rows: list[dict[str, object]] = []
    for epoch in args.epochs:
        checkpoint = args.staging_dir / ".eligible_checkpoints" / f"epoch-{epoch:03d}.pt"
        state = cast(
            dict[str, torch.Tensor],
            torch.load(checkpoint, map_location="cpu", weights_only=True),
        )
        model.load_state_dict(state)
        accelerator.wait_for_everyone()
        validation = _validate_epoch(
            model,
            data,
            accelerator,
            edge_batch=args.edge_batch or cfg.data.edge_batch,
            topk_fraction=cfg.diagnostics.topk_fraction,
            token_table=factory._token_table,
            token_node_index=factory._token_node_index,
        )
        if accelerator.is_main_process:
            assert validation is not None
            metrics = validation.metrics
            fidelity = validation.fidelity
            phase, lr = _phase_and_lr(cfg, data, model, epoch, accelerator.num_processes)
            finite = all(math.isfinite(value) for value in fidelity.values())
            row = {
                "epoch": epoch,
                "phase": phase,
                "auroc": metrics.auroc,
                "auprc": metrics.auprc,
                "lr": lr,
                "fidelity": fidelity,
                "quality_thresholds": {
                    "validation_values_finite": finite,
                    "warm_reference_floor_pass": None,
                    "validation_logit_collapse": None,
                    "slot_collapse": False,
                    "cumulative_quality_guards_passed": None,
                },
                "gradient_norm_probes": [],
                "formal": False,
                "posthoc_recovery": True,
            }
            with (args.output_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            edge_row = {
                "epoch": epoch,
                "metrics": asdict(metrics),
                "scale_telemetry": {
                    name: value if math.isfinite(value) else None
                    for name, value in validation.scale_telemetry.items()
                },
                "timing": validation.timing,
            }
            holdout = data.internal_holdout
            assert holdout is not None
            target_edges = len(data.validation_positive_edges)
            ranked = sorted(
                range(len(data.val_pairs)),
                key=lambda index: (-float(validation.active_logits[index]), data.val_pairs[index]),
            )
            predicted = nx.Graph()
            predicted.add_nodes_from(data.validation_nodes)
            predicted.add_edges_from(data.val_pairs[index] for index in ranked[:target_edges])
            gold = holdout.build_g_hold()
            topology = evaluate_assembled_graph(
                predicted,
                gold,
                {len(holdout.holdout_draws[0]): [set(draw) for draw in holdout.holdout_draws]},
                MMDConfig(),
            )
            edge_row["topology"] = {
                "graph_similarity": topology.graph_similarity,
                "relative_density": topology.relative_density,
                "mmd_ratio": topology.mmd_ratio,
            }
            with (args.output_dir / "edge_metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(edge_row, sort_keys=True, allow_nan=False) + "\n")
            records.append(
                E2ECheckpointRecord(
                    epoch=epoch,
                    phase=phase,
                    full_joint_epochs_completed=max(epoch - 1, 0),
                    guards_passed=False,
                    auprc=metrics.auprc,
                    prevalence=fidelity["prevalence"],
                    active_logit_std=fidelity["active_logit_std"],
                    gs=fidelity["gs"],
                    rd=fidelity["rd"],
                    degree_mmd=fidelity["degree_mmd"],
                    clustering_mmd=fidelity["clustering_mmd"],
                    spectral_mmd=fidelity["spectral_mmd"],
                    brier=metrics.brier,
                    warm_reference_std=None,
                    warm_reference_auprc=None,
                    residual_ratio=fidelity["topology_delta_ratio"],
                    topology_gradient_norm=None,
                )
            )
            checkpoint_rows.append(
                {
                    "epoch": epoch,
                    "path": str(checkpoint),
                }
            )
        accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        accelerator.end_training()
        return
    selected = select_e2e_checkpoint(records, "full")
    assert selected is not None
    metadata = {
        "schema": "egostitch_v_hold_checkpoint_recovery_v1",
        "formal": False,
        "diagnostic_only": True,
        "source_run_complete": False,
        "source_run_metadata": str(source_metadata_path),
        "config": str(args.config),
        "sampled_epochs": list(args.epochs),
        "validation_coverage_exact": True,
        "evaluator": "src.train_egostitch._validate_epoch",
        "world_size": accelerator.num_processes,
        "unrecoverable_training_fields": [
            "loss_*",
            "gradient_norm_probes",
            "warm_reference_floor_pass",
            "validation_logit_collapse",
            "cumulative_quality_guards_passed",
        ],
        "sampled_selector": {
            "scope": "best_of_sampled_epochs_only",
            "not_official_full_run_selection": True,
            "selected_epoch": selected.epoch,
            "rule": "mean_rank_auprc_plus_five_topology",
        },
        "checkpoints": checkpoint_rows,
    }
    metadata_path = args.output_dir / "recovery_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    complete = {
        "schema": "egostitch_v_hold_checkpoint_recovery_complete_v1",
        "formal": False,
        "selected_epoch_best_of_sampled": selected.epoch,
    }
    (args.output_dir / "diagnostic_recovery_complete.json").write_text(
        json.dumps(complete, indent=2, sort_keys=True) + "\n"
    )
    accelerator.end_training()


def main(argv: Sequence[str] | None = None) -> None:
    """Run checkpoint recovery from CLI arguments."""
    recover(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()

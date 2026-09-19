"""Prepare and validate cross-fitted motif prediction caches.

The input pair manifest is the exact union of the 15-epoch task corpus and the
deterministically enumerated structural-stream pairs.  For target fold ``f``
the command loads the generator trained on fold ``1-f``; consequently neither
endpoint of any predicted row was visible to its generator during training.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

import torch
from sklearn.metrics import roc_auc_score

import src.score_universe as su
import src.train_b0 as tb
from src.data.distributed_pairs import build_distributed_epoch_plan
from src.data.motif_crossfit import (
    MotifCrossfitCache,
    Pair,
    eligible_target_fold,
    enumerate_struct_pairs,
    seeded_hash_node_folds,
)
from src.data.motif_template import MotifTemplateTable
from src.data.packed_features import load_packed_manifest
from src.data.struct_sampler import StructSampler
from src.experiments.motif_generator_fit import capture_node_states, collate, row_inputs
from src.experiments.motif_pilot_b import GraphLossWeights
from src.model.egostitch.classifier.motif_prompt import MotifGenerator, V3_1MotifPrompt

logger = logging.getLogger("motif_crossfit")


class NoPassingPilotError(RuntimeError):
    """The crossfit wave must stop because no generator passed Pilot B."""

    def __init__(self, reports: Sequence[Path]) -> None:
        """Record every considered report for the campaign's negative verdict."""
        self.reports = tuple(reports)
        joined = ", ".join(str(path) for path in self.reports)
        super().__init__(
            "no crossfit checkpoint passed held-out-training Pilot B levels 1 and 2; "
            f"stop the wave as a scientific negative (reports: {joined})"
        )


def prepare_manifest(
    config: Path, output_dir: Path, *, world_size: int = 4, run_seed: int | None = None
) -> dict[str, object]:
    """Materialise the exact task plus deterministic structural pair universe."""
    cfg = tb.load_config(config)
    if run_seed is not None:
        cfg = replace(cfg, seed=run_seed)
    if cfg.runtime is None or cfg.struct is None:
        raise ValueError("crossfit manifest config requires runtime and structural stream blocks")
    assembled = tb.assemble_data(cfg)
    corpus = tb._dynamic_training_corpus(cfg, assembled)
    packed = load_packed_manifest(cfg.runtime.pack_dir)
    lengths = {row.node_id: row.length for row in packed.nodes}
    epoch_steps: dict[int, int] = {}
    for epoch in range(1, cfg.optim.epochs + 1):
        rows = corpus.epoch_rows[epoch]
        plan = build_distributed_epoch_plan(
            [(lengths[corpus.pairs[i][0]], lengths[corpus.pairs[i][1]]) for i in rows],
            # e2_pipeline passes runtime.token_budget verbatim as the per-rank
            # planner budget; it is not a global value to divide by world size.
            token_budget_per_rank=cfg.runtime.token_budget,
            max_pairs_per_rank=cfg.runtime.max_pairs_per_rank,
            world_size=world_size,
            seed=cfg.seed,
            epoch=epoch,
            shuffle=True,
        )
        plan = [tb._interleave_bucket_specs(rank_plan) for rank_plan in plan]
        counts = {len(rank_plan) for rank_plan in plan}
        if len(counts) != 1:
            raise ValueError(f"epoch {epoch} has unequal DDP plan lengths: {sorted(counts)}")
        epoch_steps[epoch] = counts.pop()
    sampler = StructSampler(
        assembled.val_split.build_training_graph(),
        nodes=cfg.struct.nodes,
        background_nodes=cfg.struct.background_nodes,
        mix=cfg.struct.mix,
        v_val=assembled.val_split.v_val,
        exclude_nodes=assembled.exclude_nodes,
    )
    struct_pairs = enumerate_struct_pairs(sampler, epoch_steps, seed=cfg.seed)
    pairs = sorted(set(corpus.pairs) | set(struct_pairs))
    nodes = sorted(set(assembled.val_split.train_nodes) - set(assembled.exclude_nodes))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pairs.json").write_text(json.dumps(pairs) + "\n", encoding="utf-8")
    (output_dir / "nodes.json").write_text(json.dumps(nodes) + "\n", encoding="utf-8")
    (output_dir / "forbidden_nodes.json").write_text(
        json.dumps(sorted(set(assembled.val_split.v_val) | set(assembled.exclude_nodes))) + "\n",
        encoding="utf-8",
    )
    report: dict[str, object] = {
        "config": str(config),
        "seed": cfg.seed,
        "world_size": world_size,
        "epoch_steps": epoch_steps,
        "task_pairs": len(corpus.pairs),
        "struct_pairs": len(struct_pairs),
        "union_pairs": len(pairs),
        "training_nodes": len(nodes),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _read_string_list(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{path} must contain a JSON string list")
    return value


def _read_pairs(path: Path) -> list[Pair]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(
        isinstance(row, list) and len(row) == 2 and all(isinstance(item, str) for item in row)
        for row in value
    ):
        raise ValueError(f"{path} must contain JSON [[u, v], ...]")
    return [(row[0], row[1]) for row in value]


def predict_rows(
    model: V3_1MotifPrompt,
    pairs: Sequence[Pair],
    pack_dir: Path,
    *,
    device: torch.device,
    amp: str,
    token_budget: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Predict fp32 motif weights and optional presence from the frozen generator."""
    if model.generator is None:
        raise ValueError("crossfit prediction requires a checkpoint carrying a generator")
    states = capture_node_states(
        model,
        sorted({node for pair in pairs for node in pair}),
        pack_dir,
        device=device,
        amp=amp,
        token_budget=token_budget,
    )
    weights: list[torch.Tensor] = []
    presence: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            rows = [row_inputs(pair, states) for pair in pairs[start : start + batch_size]]
            encoded_u, encoded_v, lengths_u, lengths_v = collate(rows, device)
            predicted, confidence = model.predict_graph(encoded_u, encoded_v, lengths_u, lengths_v)
            weights.append(predicted.detach().cpu().float())
            if confidence is not None:
                presence.append(confidence.detach().cpu().float())
    if not weights:
        return torch.empty((0, 96), dtype=torch.float32), None
    return torch.cat(weights), torch.cat(presence) if presence else None


def prepare_cache(
    *,
    pairs: Sequence[Pair],
    nodes: Sequence[str],
    forbidden_nodes: frozenset[str],
    checkpoints_by_source_fold: Mapping[int, Path],
    pack_dir: Path,
    output: Path,
    seed: int,
    device: torch.device,
    amp: str,
    token_budget: int,
    batch_size: int,
) -> MotifCrossfitCache:
    """Predict every eligible manifest row with its opposite-fold generator."""
    folds = seeded_hash_node_folds(nodes, seed=seed)
    missing_nodes = sorted({node for pair in pairs for node in pair if node not in folds})
    if missing_nodes:
        raise ValueError(
            f"pair manifest contains {len(missing_nodes)} nodes outside training universe"
        )
    eligible = [pair for pair in pairs if eligible_target_fold(pair, folds) is not None]
    predictions: dict[Pair, tuple[torch.Tensor, torch.Tensor | None, int]] = {}
    has_presence: bool | None = None
    for target_fold in (0, 1):
        source_fold = 1 - target_fold
        target_pairs = sorted(
            {pair for pair in eligible if eligible_target_fold(pair, folds) == target_fold}
        )
        loaded, family, _ = su._load_checkpoint(checkpoints_by_source_fold[source_fold])
        if family != su.MOTIF_PROMPT_FAMILY:
            raise ValueError(f"fold {source_fold} checkpoint is not a motif-prompt model")
        model = cast(V3_1MotifPrompt, loaded).to(device).eval()
        if model.cfg.crossfit_fold != source_fold or model.cfg.crossfit_seed != seed:
            raise ValueError(
                "checkpoint provenance fold/seed "
                f"{model.cfg.crossfit_fold}/{model.cfg.crossfit_seed} "
                f"!= requested {source_fold}/{seed}"
            )
        values, confidence = predict_rows(
            model,
            target_pairs,
            pack_dir,
            device=device,
            amp=amp,
            token_budget=token_budget,
            batch_size=batch_size,
        )
        present = confidence is not None
        if has_presence is not None and present != has_presence:
            raise ValueError("fold checkpoints disagree on presence-head schema")
        has_presence = present
        for index, pair in enumerate(target_pairs):
            predictions[pair] = (
                values[index],
                None if confidence is None else confidence[index],
                source_fold,
            )
    ordered = sorted(predictions)
    cache = MotifCrossfitCache.build(
        ordered,
        torch.stack([predictions[pair][0] for pair in ordered]),
        fold_by_node=folds,
        source_folds=[predictions[pair][2] for pair in ordered],
        seed=seed,
        presence=(
            torch.stack([cast(torch.Tensor, predictions[pair][1]) for pair in ordered])
            if has_presence
            else None
        ),
        forbidden_nodes=forbidden_nodes,
    )
    cache.require_pairs(pairs, folds)
    cache.save(output)
    logger.info("wrote %d cross-fitted rows to %s", len(cache.pairs), output)
    return cache


def select_pilot_checkpoint(reports: Sequence[Path]) -> Path:
    """Select the lowest held-out-training L_G among pilot-B passing prefixes.

    V_val/downstream cells are deliberately ignored: fold generators are graph
    teachers and may use training-only, node-disjoint pilot-B evidence. Reports
    must be supplied in epoch order; Python's stable ``min`` then resolves an
    exact L_G tie to the earlier epoch.
    """
    candidates: list[tuple[bool, float, str, Path]] = []
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        heldout = report["level_1_fit"]["heldout_train"]
        passed = bool(report["verdicts"]["level_1"]["passed"]) and bool(
            report["verdicts"]["level_2"]["per_universe"]["heldout_train"]
        )
        candidates.append(
            (passed, float(heldout["L_G"]["generator"]), str(report["checkpoint"]), path)
        )
    passing = [row for row in candidates if row[0]]
    if not candidates:
        raise ValueError("at least one pilot-B report is required")
    if not passing:
        raise NoPassingPilotError(reports)
    return min(passing, key=lambda row: row[1])[3]


def v_val_generator_report(
    checkpoints: Mapping[str, Path],
    *,
    pack_dir: Path,
    data_root: Path,
    strategy: str,
    device: torch.device,
    amp: str,
    token_budget: int,
    batch_size: int,
) -> dict[str, object]:
    """Compare fold and deployed generator graph loss on the same V_val rows."""
    pairs, _ = su._resolve_pairs("val_cls", data_root, strategy)
    table = MotifTemplateTable(su._oracle_truth_graph_for_scoring("val_cls", data_root, strategy))
    target = torch.from_numpy(table.weights(pairs)).float()
    keep = torch.as_tensor([u != v for u, v in pairs])
    losses: dict[str, float] = {}
    identities: dict[str, str] = {}
    for name, checkpoint in checkpoints.items():
        loaded, family, checkpoint_id = su._load_checkpoint(checkpoint)
        if family != su.MOTIF_PROMPT_FAMILY:
            raise ValueError(f"{checkpoint} is not a motif-prompt checkpoint")
        model = cast(V3_1MotifPrompt, loaded).to(device).eval()
        predicted, _ = predict_rows(
            model,
            pairs,
            pack_dir,
            device=device,
            amp=amp,
            token_budget=token_budget,
            batch_size=batch_size,
        )
        losses[name] = GraphLossWeights.from_config(model.cfg).mean(predicted[keep], target[keep])
        identities[name] = checkpoint_id
    deployed = losses["deployed"]
    return {
        "universe": "V_val val_cls nonself rows",
        "rows": int(keep.sum()),
        "checkpoint_ids": identities,
        "L_G": losses,
        "fold_minus_deployed": {
            name: value - deployed for name, value in losses.items() if name != "deployed"
        },
    }


def heldout_presence_report(
    cache: MotifCrossfitCache, *, data_root: Path, strategy: str
) -> dict[str, float | int | None] | None:
    """Measure per-type presence AUROC on cross-fitted training rows."""
    if cache.presence is None:
        return None
    split = su._load_val_region_split(data_root, strategy)
    table = MotifTemplateTable(split.build_training_graph())
    truth_weights = torch.from_numpy(table.weights(cache.pairs)).float()
    truth = MotifGenerator.presence_from_weights(truth_weights).numpy()
    predicted = cache.presence.numpy()
    report: dict[str, float | int | None] = {"rows": len(cache.pairs)}
    for index, name in enumerate(("closure", "attach", "interior")):
        labels = truth[:, index]
        report[name] = (
            None
            if len(set(labels.tolist())) < 2
            else float(roc_auc_score(labels, predicted[:, index]))
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the manifest, cache preparation and pilot selection CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--pairs", type=Path, required=True)
    prepare.add_argument("--nodes", type=Path, required=True)
    prepare.add_argument("--forbidden-nodes", type=Path)
    prepare.add_argument("--fold0-checkpoint", type=Path, required=True)
    prepare.add_argument("--fold1-checkpoint", type=Path, required=True)
    prepare.add_argument("--deployed-checkpoint", type=Path, required=True)
    prepare.add_argument("--pack-dir", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--device", default="auto")
    prepare.add_argument("--amp", choices=("off", "bf16"), default="bf16")
    prepare.add_argument("--token-budget", type=int, default=131_072)
    prepare.add_argument("--batch-size", type=int, default=512)
    prepare.add_argument("--data-root", type=Path, default=Path("data"))
    prepare.add_argument("--strategy", default="breadth_first")
    select = sub.add_parser("select-pilot")
    select.add_argument("reports", type=Path, nargs="+")
    manifest = sub.add_parser("prepare-manifest")
    manifest.add_argument("--config", type=Path, required=True)
    manifest.add_argument("--output-dir", type=Path, required=True)
    manifest.add_argument("--world-size", type=int, default=4)
    manifest.add_argument("--run-seed", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch one crossfit preparation operation."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.command == "select-pilot":
        selected = select_pilot_checkpoint(args.reports)
        report = json.loads(selected.read_text(encoding="utf-8"))
        sys.stdout.write(f"{report['checkpoint']}\n")
        return
    if args.command == "prepare-manifest":
        prepare_manifest(
            args.config, args.output_dir, world_size=args.world_size, run_seed=args.run_seed
        )
        return
    forbidden: frozenset[str] = frozenset()
    if args.forbidden_nodes is not None:
        forbidden = frozenset(_read_string_list(args.forbidden_nodes))
    cache = prepare_cache(
        pairs=_read_pairs(args.pairs),
        nodes=_read_string_list(args.nodes),
        forbidden_nodes=forbidden,
        checkpoints_by_source_fold={0: args.fold0_checkpoint, 1: args.fold1_checkpoint},
        pack_dir=args.pack_dir,
        output=args.output,
        seed=args.seed,
        device=su._resolve_device(args.device),
        amp=args.amp,
        token_budget=args.token_budget,
        batch_size=args.batch_size,
    )
    report = v_val_generator_report(
        {
            "fold0": args.fold0_checkpoint,
            "fold1": args.fold1_checkpoint,
            "deployed": args.deployed_checkpoint,
        },
        pack_dir=args.pack_dir,
        data_root=args.data_root,
        strategy=args.strategy,
        device=su._resolve_device(args.device),
        amp=args.amp,
        token_budget=args.token_budget,
        batch_size=args.batch_size,
    )
    report["heldout_crossfit_presence_auroc"] = heldout_presence_report(
        cache, data_root=args.data_root, strategy=args.strategy
    )
    args.output.with_suffix(args.output.suffix + ".report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

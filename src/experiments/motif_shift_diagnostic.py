"""Run and read the V17 Step-0 motif shift diagnostic on V_val ``val_cls``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score

from src.data.motif_shift import (
    apply_motif_shift,
    calibrate_motif_shift,
    normalized_entropy,
)
from src.data.motif_template import EDGE_TYPES, MotifTemplateTable
from src.experiments.motif_pilot_b import predicted_bank
from src.model.egostitch.classifier.motif_prompt import V3_1MotifPrompt
from src.score_universe import (
    MOTIF_PROMPT_FAMILY,
    _load_checkpoint,
    _oracle_truth_graph_for_scoring,
    _resolve_pairs,
    load_scores,
    validate_artifact_precision,
)

CASES = (
    "s2_true",
    "s2_pred",
    "gates_off",
    "blur_true",
    "presence_true",
    "level_true",
    "content_true",
)
SHARE = 1.0 / 3.0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_counterfactual_bank(
    *,
    checkpoint: Path,
    output: Path,
    pack_dir: Path,
    data_root: Path,
    strategy: str,
    device: torch.device,
    token_budget: int,
) -> None:
    """Predict all V_val graphs and build every counterfactual before sharding."""
    model, family, _ = _load_checkpoint(checkpoint)
    motif_model = cast(V3_1MotifPrompt, model)
    if family != MOTIF_PROMPT_FAMILY or motif_model.cfg.stage != "two":
        raise ValueError("Step 0 requires one published stage-two motif-prompt checkpoint")
    motif_model = motif_model.to(device).eval()
    pairs, _ = _resolve_pairs("val_cls", data_root, strategy)
    truth = MotifTemplateTable(
        _oracle_truth_graph_for_scoring("val_cls", data_root, strategy)
    ).weights(pairs)
    predicted, _ = predicted_bank(
        motif_model,
        pairs,
        pack_dir,
        device=device,
        amp="bf16" if device.type == "cuda" else "off",
        token_budget=token_budget,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("Step 0 checkpoint is missing its complete training config")
    # Rebuild the exact epoch-1 dynamic 1:5 corpus used to install the model's
    # training statistics. The resolved YAML is retained as diagnostic
    # provenance beside the bank.
    raw_config = dict(raw_config)
    raw_data = dict(raw_config["data"])
    raw_data["root"] = str(data_root)
    raw_config["data"] = raw_data
    config_path = output.with_name("resolved_training_config.yaml")
    config_path.write_text(yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8")
    from src.train_b0 import _dynamic_training_corpus, assemble_data, load_config

    cfg = load_config(config_path)
    assembled = assemble_data(cfg)
    corpus = _dynamic_training_corpus(cfg, assembled)
    epoch_one_rows = corpus.epoch_rows[1]
    training_pairs = [corpus.pairs[int(row)] for row in epoch_one_rows]
    training = MotifTemplateTable(assembled.val_split.build_training_graph()).weights(
        training_pairs, seed=cfg.seed, epoch=1, randomise=True
    )
    u = np.asarray([pair[0] for pair in pairs])
    v = np.asarray([pair[1] for pair in pairs])
    predicted_array = predicted.numpy()
    calibration = calibrate_motif_shift(truth, predicted_array, training)
    arrays: dict[str, Any] = {"u": u, "v": v, "s2_true": np.asarray(truth, dtype=np.float32)}
    for mode in ("blur_true", "presence_true", "level_true", "content_true"):
        arrays[mode] = apply_motif_shift(mode, truth, predicted_array, calibration)
    arrays["entropy_lambdas"] = np.asarray(calibration.lambdas, dtype=np.float64)
    arrays["training_nonzero_floor"] = np.asarray(calibration.nonzero_floor, dtype=np.float64)
    for mode in ("blur_true", "content_true"):
        values = arrays[mode]
        residuals = []
        for kind in range(3):
            indices = np.flatnonzero(np.asarray(EDGE_TYPES) == kind)
            residuals.append(
                float(
                    normalized_entropy(values, indices).mean()
                    - normalized_entropy(predicted_array, indices).mean()
                )
            )
        arrays[f"{mode}_entropy_residual"] = np.asarray(residuals, dtype=np.float64)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)


def analyze(score_paths: dict[str, Path], *, checkpoint: Path) -> dict[str, Any]:
    """Compute actual row-aligned AUPRCs and the pre-registered queue decision."""
    artifacts = {name: load_scores(score_paths[name]) for name in CASES}
    for name, artifact in artifacts.items():
        validate_artifact_precision(artifact, label=name)
    reference = artifacts[CASES[0]]
    pairs = list(reference.pairs())
    labels = reference.label
    if np.unique(labels).size != 2:
        raise ValueError("motif shift diagnostic requires both val_cls label classes")
    for name, artifact in artifacts.items():
        if list(artifact.pairs()) != pairs or not np.array_equal(artifact.label, labels):
            raise ValueError(f"{name} does not match the ordered s2_true val_cls rows/labels")
    metrics = {
        name: {"auprc": float(average_precision_score(labels, artifact.logit))}
        for name, artifact in artifacts.items()
    }
    true_score = metrics["s2_true"]["auprc"]
    pred_score = metrics["s2_pred"]["auprc"]
    delta = true_score - pred_score
    if not delta > 0:
        raise ValueError(f"Step-0 decomposition requires s2_true > s2_pred, got delta={delta}")
    blur_loss = true_score - metrics["blur_true"]["auprc"]
    presence_recovery = metrics["presence_true"]["auprc"] - pred_score
    blur_fraction = blur_loss / delta
    presence_fraction = presence_recovery / delta
    run_shift = blur_fraction >= SHARE
    run_confidence = presence_fraction >= SHARE
    return {
        "schema": "motif_shift_step0_v1",
        "checkpoint": str(checkpoint),
        "pairs_source": "val_cls",
        "rows": int(len(labels)),
        "metrics": metrics,
        "delta": delta,
        "blur_loss": blur_loss,
        "blur_loss_fraction": blur_fraction,
        "presence_recovery": presence_recovery,
        "presence_recovery_fraction": presence_fraction,
        "thresholds": {"share": SHARE},
        "decisions": {
            "run_shift": run_shift,
            "run_confidence": run_confidence,
            "stop": not run_shift and not run_confidence,
        },
    }


def run(
    *,
    checkpoint: Path,
    output_dir: Path,
    pack_dir: Path,
    data_root: Path,
    strategy: str,
    gpu_count: int | None,
    python_bin: Path,
    token_budget: int = 131_072,
    score_runner: Callable[[Sequence[str]], Path] | None = None,
) -> dict[str, Any]:
    """Build whole-universe banks, score all cases through fan-out, and decide."""
    if score_runner is None:
        from src.e2_pipeline import detect_visible_gpu_count
        from src.score_fanout import score_sharded

        resolved_gpu_count = (
            detect_visible_gpu_count() if gpu_count is None else gpu_count
        )
        score_runner = lambda args: score_sharded(  # noqa: E731
            args, gpu_count=resolved_gpu_count, python_bin=python_bin
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    bank = output_dir / "motif_shift_bank.npz"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_counterfactual_bank(
        checkpoint=checkpoint,
        output=bank,
        pack_dir=pack_dir,
        data_root=data_root,
        strategy=strategy,
        device=device,
        token_budget=token_budget,
    )
    score_dir = output_dir / "scores"
    score_paths: dict[str, Path] = {}
    for case in CASES:
        output = score_dir / f"{case}.npz"
        args = [
            "--checkpoint",
            str(checkpoint),
            "--pairs",
            "val_cls",
            "--data-root",
            str(data_root),
            "--strategy",
            strategy,
            "--pack-dir",
            str(pack_dir),
            "--output",
            str(output),
        ]
        if case == "gates_off":
            args += ["--prefix-intervention", "gates_off"]
        elif case not in {"s2_pred"}:
            args += [
                "--prefix-intervention",
                case,
                "--motif-shift-bank",
                str(bank),
                "--allow-oracle-diagnostic",
            ]
        score_paths[case] = score_runner(args)
    decision = analyze(score_paths, checkpoint=checkpoint)
    with np.load(bank, allow_pickle=False) as payload:
        decision["calibration"] = {
            "entropy_lambdas": payload["entropy_lambdas"].astype(float).tolist(),
            "training_nonzero_floor": payload["training_nonzero_floor"].astype(float).tolist(),
            "blur_true_entropy_residual": payload["blur_true_entropy_residual"]
            .astype(float)
            .tolist(),
            "content_true_entropy_residual": payload["content_true_entropy_residual"]
            .astype(float)
            .tolist(),
            "entropy_scope": "whole_val_cls_including_self_rows",
            "statistics_scope": "training_epoch_1_only",
        }
    _write_json(output_dir / "decision.json", decision)
    return decision


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the Step-0 CLI and run it to a terminal decision artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=None,
        help="score fan-out width; default detects all visible GPUs",
    )
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--token-budget", type=int, default=131_072)
    args = parser.parse_args(argv)
    run(**vars(args))


if __name__ == "__main__":
    main()

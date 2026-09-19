"""Run the V_val-only motif-dictionary oracle diagnostic through production scoring."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)

from src.data.motif_dictionary import load_dictionary
from src.eval.calibration import stable_sigmoid
from src.eval.edge_metrics import compute_edge_metrics, select_max_f1_threshold
from src.eval.fixed_threshold import select_fixed_threshold
from src.eval.graph_metrics import MMDConfig
from src.experiments.motif_density_control import density_matched_report
from src.model.egostitch.classifier.motif_dictionary import V3_1MotifDictionary
from src.model.egostitch.classifier.motif_prompt import V3_1MotifPrompt
from src.score_universe import (
    MOTIF_DICTIONARY_FAMILY,
    _load_checkpoint,
    _load_val_region_split,
    load_scores,
    validate_artifact_precision,
)

logger = logging.getLogger(__name__)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_oracle_checkpoint(
    artifact_path: Path, stage1_checkpoint: Path, output: Path
) -> dict[str, object]:
    """Initialize a file-independent oracle checkpoint from Stage I plus the dictionary."""
    source, family, _ = _load_checkpoint(stage1_checkpoint)
    if family != "v3_1_motif_prompt" or cast(V3_1MotifPrompt, source).cfg.stage != "one":
        raise ValueError("--stage1-checkpoint must be a stage-one v3_1_motif_prompt checkpoint")
    payload = torch.load(stage1_checkpoint, map_location="cpu", weights_only=False)
    source_config = payload.get("model_config")
    if not isinstance(source_config, dict):
        raise ValueError("Stage-I checkpoint is missing embedded model_config")
    artifact = load_dictionary(artifact_path)
    model_config: dict[str, object] = {
        "base": dict(cast(dict[str, object], source_config["base"])),
        "motif_prompt": dict(cast(dict[str, object], source_config["motif_prompt"])),
        "motif_dictionary": {
            "mode": "oracle",
            "dictionary_size": int(artifact.tokens.size(0)),
            "artifact_path": "",
            "slot_checkpoint": "",
        },
    }
    model = V3_1MotifDictionary(**model_config)  # type: ignore[arg-type]
    target_state = model.state_dict()
    source_state = source.state_dict()
    missing_source = sorted(set(source_state) - set(target_state))
    if missing_source:
        raise ValueError(f"Stage-I state has keys absent from dictionary model: {missing_source}")
    incompatible = sorted(
        key for key, value in source_state.items() if target_state[key].shape != value.shape
    )
    if incompatible:
        raise ValueError(f"Stage-I state shapes disagree with dictionary model: {incompatible}")
    target_state.update(source_state)
    model.load_state_dict(target_state, strict=True)
    model.install_dictionary(artifact)
    checkpoint = {
        "model_family": MOTIF_DICTIONARY_FAMILY,
        "model_config": model_config,
        "model_state": model.state_dict(),
        "run_kind": "diagnostic",
        "diagnostic": {
            "name": "motif_dictionary_oracle",
            "stage1_checkpoint": str(stage1_checkpoint),
            "dictionary_artifact": str(artifact_path),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return cast(dict[str, object], checkpoint["diagnostic"])


def _score_command(
    *,
    checkpoint: Path,
    pairs: str,
    output: Path,
    pack_dir: Path,
    data_root: Path,
    strategy: str,
    intervention: str = "none",
) -> list[str]:
    command = [
        "hpc/run.sh",
        "score",
        "--checkpoint",
        str(checkpoint),
        "--pairs",
        pairs,
        "--data-root",
        str(data_root),
        "--strategy",
        strategy,
        "--pack-dir",
        str(pack_dir),
        "--output",
        str(output),
        "--allow-oracle-diagnostic",
    ]
    if intervention != "none":
        command.extend(["--prefix-intervention", intervention])
    return command


def _run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def _edge_metrics(path: Path) -> dict[str, float | int]:
    artifact = load_scores(path)
    validate_artifact_precision(artifact, label=str(path))
    labels = artifact.label.astype(np.int64)
    logits = artifact.logit.astype(np.float64)
    nonself = artifact.u_idx != artifact.v_idx
    threshold = select_max_f1_threshold(labels, logits)
    probs = stable_sigmoid(logits)
    # The shared probability helper supplies calibration only. Ranking and the
    # frozen decision remain on raw logits so sigmoid saturation cannot merge
    # distinct scores or admit a different boundary tie group.
    metrics = compute_edge_metrics(labels, probs)
    predictions = (logits >= threshold.logit_threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    return {
        "rows": int(len(labels)),
        **asdict(metrics),
        "auroc": float(roc_auc_score(labels, logits)),
        "auprc": float(average_precision_score(labels, logits)),
        "accuracy": float((tp + tn) / len(labels)),
        "sensitivity": recall,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": float(2.0 * precision * recall / (precision + recall))
        if precision + recall
        else 0.0,
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "threshold": float(expit(threshold.logit_threshold)),
        "logit_threshold": threshold.logit_threshold,
        "nonself_rows": int(nonself.sum()),
        "nonself_auprc": float(average_precision_score(labels[nonself], logits[nonself])),
    }


def analyze_scores(
    score_dir: Path, data_root: Path, strategy: str, cases: list[str]
) -> dict[str, object]:
    """Analyze cached V_val scores; no held-out file is opened."""
    split = _load_val_region_split(data_root, strategy)
    graph = split.build_g_val()
    config = MMDConfig()
    results: dict[str, object] = {}
    topology_artifacts = {
        case: load_scores(score_dir / f"{case}_val_topology.npz") for case in cases
    }
    for case, topology in topology_artifacts.items():
        validate_artifact_precision(topology, label=f"{case} val_topology")
    reference_pairs = list(topology_artifacts["dictionary"].pairs())
    if any(list(artifact.pairs()) != reference_pairs for artifact in topology_artifacts.values()):
        raise ValueError("V_val topology artifacts do not share the same ordered pair universe")
    selections = {
        case: select_fixed_threshold(
            pairs=reference_pairs,
            logits=artifact.logit.astype(np.float64),
            g_ref=graph,
            buckets=split.buckets,
            config=config,
        )
        for case, artifact in topology_artifacts.items()
    }
    dictionary_selected = cast(dict[str, object], selections["dictionary"].report["selected"])
    target_edges = int(cast(int, dictionary_selected["union_admitted_pair_count"]))
    matched = density_matched_report(
        rows={
            case: artifact.logit.astype(np.float64)
            for case, artifact in topology_artifacts.items()
        },
        union_pairs=reference_pairs,
        target_edges=target_edges,
        g_ref=graph,
        buckets=split.buckets,
        config=config,
    )
    for case in cases:
        cls_path = score_dir / f"{case}_val_cls.npz"
        results[case] = {
            "edge": _edge_metrics(cls_path),
            "protocol_topology": selections[case].report,
            "matched_output_density": cast(dict[str, object], matched["rows"])[case],
        }
    return {
        "scope": "V_val_only",
        "strategy": strategy,
        "matched_output_density": {
            "reference_case": "dictionary",
            "target_edges": target_edges,
            "report": matched,
        },
        "cases": results,
    }


def run(
    *,
    artifact: Path,
    stage1_checkpoint: Path,
    output_dir: Path,
    pack_dir: Path,
    data_root: Path,
    strategy: str,
) -> dict[str, object]:
    """Build the oracle checkpoint, fan out four V_val cases, and analyze cached scores."""
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    _write_json(status_path, {"status": "running", "started_utc": datetime.now(UTC).isoformat()})
    try:
        oracle_checkpoint = output_dir / "oracle.pt"
        provenance = build_oracle_checkpoint(artifact, stage1_checkpoint, oracle_checkpoint)
        cases = {
            "true": (stage1_checkpoint, "none"),
            "dictionary": (oracle_checkpoint, "none"),
            "mean": (oracle_checkpoint, "mean"),
            "gates_off": (oracle_checkpoint, "gates_off"),
        }
        score_dir = output_dir / "scores"
        for case, (checkpoint, intervention) in cases.items():
            for universe in ("val_cls", "val_topology"):
                output = score_dir / f"{case}_{universe}.npz"
                _write_json(status_path, {"status": "running", "case": case, "universe": universe})
                _run_command(
                    _score_command(
                        checkpoint=checkpoint,
                        pairs=universe,
                        output=output,
                        pack_dir=pack_dir,
                        data_root=data_root,
                        strategy=strategy,
                        intervention=intervention,
                    ),
                    output_dir / "diagnostic.log",
                )
        report = analyze_scores(score_dir, data_root, strategy, list(cases))
        report["provenance"] = provenance
        _write_json(output_dir / "metrics.json", report)
        complete = {"status": "complete", "completed_utc": datetime.now(UTC).isoformat(), **report}
        _write_json(output_dir / "complete.json", complete)
        _write_json(status_path, complete)
        (output_dir / "failure.json").unlink(missing_ok=True)
        return complete
    except Exception as error:
        failure = {
            "status": "failed",
            "failed_utc": datetime.now(UTC).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _write_json(output_dir / "failure.json", failure)
        _write_json(status_path, failure)
        (output_dir / "complete.json").unlink(missing_ok=True)
        raise


def main() -> None:
    """Parse the oracle-diagnostic CLI and run it to a terminal artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--strategy", required=True)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()

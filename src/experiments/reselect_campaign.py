"""Replay saved V3.1 epochs on V_val, then continue HPO under the current rule.

Original artifacts and teacher banks are immutable inputs. One lane owns each
study. Scoring uses hpc/run.sh (all visible GPUs); metric evaluation stays on CPU.
No test pairs are read and no held-out tests are launched by this migration.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import time
from dataclasses import asdict
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, cast

import networkx as nx
import numpy as np
import optuna
import torch
import yaml
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from src.eval.checkpoint_selection import (
    SELECTION_RULE,
    CheckpointCandidate,
    TopologyValidationMetrics,
    select_checkpoint,
)
from src.eval.edge_metrics import compute_edge_metrics
from src.eval.fixed_threshold import select_fixed_threshold
from src.eval.graph_metrics import MMDConfig
from src.experiments import kd_rank_rep_hpo, kd_rank_strict_hpo, struct_hpo
from src.score_universe import load_scores, validate_artifact_precision

logger = logging.getLogger(__name__)


def write_json(path: Path, payload: object) -> None:
    """Atomically publish a replay record; source artifacts are never modified."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def validation_union(
    manifest: dict[str, Any],
) -> tuple[list[tuple[str, str]], set[tuple[str, str]]]:
    """One scoring pass covers topology and classification, entirely on V_val."""
    cls = {tuple(sorted(pair)) for pair in manifest["val_positives"] + manifest["val_negatives"]}
    pairs = set(cls)
    for balls in manifest["buckets"].values():
        for nodes in balls:
            pairs.update(combinations_with_replacement(sorted(nodes), 2))
    nodes = set(manifest["v_val"])
    if not all(set(pair) <= nodes for pair in pairs):
        raise ValueError("replay pair escaped V_val")
    return sorted(pairs), cls


def select_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the production selector to the re-evaluated epochs."""
    candidates = [
        CheckpointCandidate(
            epoch=r["epoch"],
            auprc=r["val_auprc"],
            topology=TopologyValidationMetrics(
                r["val_gs_bfs"],
                r["val_rd_bfs"],
                r["val_degree_mmd_ratio"],
                r["val_clustering_mmd_ratio"],
                r["val_spectral_mmd_ratio"],
            ),
        )
        for r in rows
    ]
    winner = select_checkpoint(candidates)
    if winner is None:
        raise ValueError("no saved epochs to reselect")
    return next(r for r in rows if r["epoch"] == winner.epoch)


def source_attempt(source: Path) -> Path | None:
    """Resolve the producer's explicit attempt pointer, including interrupted runs."""
    for name in ("complete.json", "current_attempt.json"):
        path = source / name
        if path.exists():
            attempt = source / "attempts" / str(json.loads(path.read_text())["attempt_id"])
            if attempt.exists():
                return attempt
    return None


def replay_run(source: Path, target: Path, config: Path, pack: Path) -> None:
    """Replay a run; resume an interrupted run from its migrated saved prefix."""
    if (target / "complete.json").exists():
        return
    # Prefer progress made after migration over the older source snapshot.
    current = target / "current_attempt.json"
    if current.exists():
        prior = target / "attempts" / json.loads(current.read_text())["attempt_id"]
        if (prior / "training_state.pt").exists():
            subprocess.run(
                [
                    "bash",
                    "hpc/run.sh",
                    "train",
                    str(target / "config.yaml"),
                    "--skip-test",
                    "--resume-attempt",
                    str(prior),
                ],
                check=True,
            )
            return
    started = time.monotonic()
    target.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path("data/val_region/breadth_first.json").read_text())
    pairs, cls_pairs = validation_union(manifest)
    positives = {tuple(sorted(p)) for p in manifest["val_positives"]}
    pair_file = target / "validation_union.tsv"
    pair_file.write_text("".join(f"{u}\t{v}\t{int((u, v) in positives)}\n" for u, v in pairs))
    graph = nx.Graph()
    graph.add_nodes_from(manifest["v_val"])
    graph.add_edges_from(positives)
    buckets = {int(k): [set(nodes) for nodes in balls] for k, balls in manifest["buckets"].items()}
    attempt = source_attempt(source)
    checkpoints = sorted((attempt / "checkpoints").glob("epoch-*.pt")) if attempt else []
    old_rows = {}
    if attempt and (attempt / "metrics.jsonl").exists():
        for line in (attempt / "metrics.jsonl").read_text().splitlines():
            row = json.loads(line)
            old_rows[row["epoch"]] = row
    rows = []
    checkpoint_by_epoch = {}
    for checkpoint in checkpoints:
        epoch = int(checkpoint.stem.split("-")[-1])
        checkpoint_by_epoch[epoch] = checkpoint
        record_path = target / "replay" / f"epoch-{epoch:04d}.json"
        if record_path.exists():
            rows.append(json.loads(record_path.read_text()))
            continue
        scores = target / "replay" / f"epoch-{epoch:04d}.npz"
        if not scores.exists():
            subprocess.run(
                [
                    "bash",
                    "hpc/run.sh",
                    "score",
                    "--checkpoint",
                    str(checkpoint),
                    "--pairs",
                    f"file:{pair_file}",
                    "--data-root",
                    "data",
                    "--strategy",
                    "breadth_first",
                    "--pack-dir",
                    str(pack),
                    "--output",
                    str(scores),
                ],
                check=True,
            )
        artifact = load_scores(scores)
        validate_artifact_precision(artifact, label=str(scores))
        actual_pairs = list(artifact.pairs())
        if actual_pairs != pairs or not np.array_equal(
            artifact.label,
            np.array([int(pair in positives) for pair in pairs]),
        ):
            raise ValueError(f"validation pair/label mismatch: {scores}")
        selection = select_fixed_threshold(
            pairs=pairs,
            logits=artifact.logit.astype(np.float64),
            g_ref=graph,
            buckets=buckets,
            config=MMDConfig(),
        )
        mask = np.array([pair in cls_pairs for pair in pairs])
        metrics = selection.metrics
        density = cast(dict[str, Any], selection.report["density_diagnostics"])
        edge = asdict(
            compute_edge_metrics(
                artifact.label[mask], expit(artifact.logit[mask].astype(np.float64))
            )
        )
        edge["auprc"] = float(average_precision_score(artifact.label[mask], artifact.logit[mask]))
        edge["auroc"] = float(roc_auc_score(artifact.label[mask], artifact.logit[mask]))
        row = dict(old_rows.get(epoch, {}))
        row.update(
            replay_edge_metrics=edge,
            val_auroc=edge["auroc"],
            val_ece=edge["ece"],
            val_brier=edge["brier"],
            epoch=epoch,
            selection_rule=SELECTION_RULE,
            val_auprc=float(average_precision_score(artifact.label[mask], artifact.logit[mask])),
            val_gs_bfs=metrics.graph_similarity,
            val_rd_bfs=metrics.relative_density,
            val_degree_mmd_ratio=metrics.mmd_ratio["degree"],
            val_clustering_mmd_ratio=metrics.mmd_ratio["clustering"],
            val_spectral_mmd_ratio=metrics.mmd_ratio["spectral"],
            val_threshold=selection.logit_threshold,
            val_rd_geometric=density["geometric_mean"],
            val_rd_mean_abs_log=density["mean_abs_log"],
            replay_source=str(checkpoint),
            replay_checkpoint_id=artifact.meta["checkpoint_id"],
        )
        write_json(record_path, row)
        write_json(record_path.with_suffix(".threshold.json"), selection.report)
        rows.append(row)
        logger.info(
            "replayed %s epoch %d: AUPRC %.4f GS %.4f geometric RD %.4f",
            source,
            epoch,
            row["val_auprc"],
            row["val_gs_bfs"],
            row["val_rd_geometric"],
        )
    rows.sort(key=lambda r: r["epoch"])
    config_data = yaml.safe_load(config.read_text())
    config_data["output_dir"] = str(target)
    target_config = target / "config.yaml"
    target_config.write_text(yaml.safe_dump(config_data, sort_keys=False))
    if not (source / "complete.json").exists():
        command = ["bash", "hpc/run.sh", "train", str(target_config), "--skip-test"]
        if attempt and (attempt / "training_state.pt").exists():
            resume = target / "attempts" / "migrated-prefix"
            (resume / "checkpoints").mkdir(parents=True, exist_ok=True)
            snapshot = torch.load(
                attempt / "training_state.pt", map_location="cpu", weights_only=False
            )
            prefix = [r for r in rows if r["epoch"] <= snapshot["epoch"]]
            if [r["epoch"] for r in prefix] != list(range(1, snapshot["epoch"] + 1)):
                raise ValueError("incomplete migrated resume prefix")
            shutil.copy2(attempt / "training_state.pt", resume / "training_state.pt")
            for row in prefix:
                payload = torch.load(
                    checkpoint_by_epoch[row["epoch"]], map_location="cpu", weights_only=False
                )
                payload["selection_metrics"] = row
                payload["val_metrics"] = row["replay_edge_metrics"]
                torch.save(payload, resume / "checkpoints" / f"epoch-{row['epoch']:04d}.pt")
            (resume / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prefix))
            write_json(resume / "status.json", {"status": "abandoned", "source": str(attempt)})
            command += ["--resume-attempt", str(resume)]
        logger.info("continuing interrupted run %s", source)
        subprocess.run(command, check=True)
        return
    winner = select_row(rows)
    payload = torch.load(
        checkpoint_by_epoch[winner["epoch"]], map_location="cpu", weights_only=False
    )
    payload["selection_rule"] = SELECTION_RULE
    payload["selection_metrics"] = winner
    payload["val_metrics"] = winner["replay_edge_metrics"]
    payload["val_threshold_transfer"] = {
        "n_val": len(manifest["v_val"]),
        "threshold": winner["val_threshold"],
    }
    torch.save(payload, target / "best.pt")
    (target / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_json(
        target / "run_metadata.json",
        {
            "selection_rule": SELECTION_RULE,
            "selected_epoch": winner["epoch"],
            "checkpoint_id": winner["replay_checkpoint_id"],
            "source_run": str(source),
            "kind": "offline_reselection",
            "val_threshold_transfer": payload["val_threshold_transfer"],
        },
    )
    write_json(
        target / "complete.json",
        {
            "status": "complete",
            "kind": "offline_reselection",
            "source_run": str(source),
            "selection_rule": SELECTION_RULE,
            "total_seconds": time.monotonic() - started,
        },
    )


def migrate_study(source: Path, target: Path, pack: Path) -> None:
    """Import reevaluated trials with original numbering, parameters and provenance."""
    storage = f"sqlite:///{source / 'optuna.db'}"
    name = optuna.get_all_study_summaries(storage)[0].study_name
    old = optuna.load_study(study_name=name, storage=storage)
    new = kd_rank_strict_hpo.build_study(target / "optuna.db", study_name=name)
    for trial in old.trials:
        if trial.number < len(new.trials):
            continue
        if trial.state not in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.RUNNING):
            new.add_trial(
                optuna.trial.create_trial(
                    state=optuna.trial.TrialState.FAIL,
                    params=trial.params,
                    distributions=trial.distributions,
                    user_attrs={"source_trial": trial.number, "source_state": trial.state.name},
                )
            )
            continue
        run = f"trial_{trial.number:03d}"
        replay_run(source / run, target / run, source / "configs" / f"{run}.yaml", pack)
        outcome = kd_rank_strict_hpo.trial_outcome(target / run)
        new.add_trial(
            optuna.trial.create_trial(
                values=outcome.values,
                params=trial.params,
                distributions=trial.distributions,
                user_attrs={
                    "surface": outcome.surface,
                    "source_study": str(source),
                    "source_trial": trial.number,
                    "selection_rule": SELECTION_RULE,
                },
            )
        )


def main() -> None:
    """Own one migration lane, followed by its remaining V_val-only searches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lane", choices=["rank", "rank_rep", "kd", "struct"], required=True)
    parser.add_argument("--pack-dir", type=Path, default=Path("outputs/feature_packs/b0_v31_bf16"))
    args = parser.parse_args()
    if args.source_root.resolve() == args.output_root.resolve():
        raise ValueError("reselection must not overwrite the source campaign")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root, dest, pack = args.source_root, args.output_root, args.pack_dir
    status = dest / f"lane_{args.lane}.json"
    write_json(status, {"status": "running", "selection_rule": SELECTION_RULE})
    try:
        if args.lane in {"rank", "kd"}:
            replay_run(
                root / "b0_v31", dest / "b0_v31", Path("configs/split_seed42/b0_v31.yaml"), pack
            )
            migrate_study(root / "kd_hpo/rank", dest / "kd_hpo/rank", pack)
            kd_rank_strict_hpo.main(
                [
                    "--base-config",
                    "configs/split_seed42/kd_rank.yaml",
                    "--teacher-checkpoint",
                    str(root / "teacher_pma1/best.pt"),
                    "--bank-root",
                    "outputs/distill/split_seed42",
                    "--sweep-dir",
                    str(dest / "kd_hpo/rank"),
                ]
            )
            if args.lane == "kd":
                migrate_study(root / "kd_hpo/rank_rep", dest / "kd_hpo/rank_rep", pack)
                kd_rank_rep_hpo.main(
                    [
                        "--base-config",
                        "configs/split_seed42/kd_rank_rep.yaml",
                        "--bank-root",
                        "outputs/distill/split_seed42",
                        "--bank",
                        "h2ns3",
                        "--margin",
                        ".1",
                        "--sweep-dir",
                        str(dest / "kd_hpo/rank_rep"),
                    ]
                )
            grids = ["kd_rep_*", "kd_logit_*"]
        elif args.lane == "rank_rep":
            migrate_study(root / "kd_hpo/rank_rep", dest / "kd_hpo/rank_rep", pack)
            kd_rank_rep_hpo.main(
                [
                    "--base-config",
                    "configs/split_seed42/kd_rank_rep.yaml",
                    "--bank-root",
                    "outputs/distill/split_seed42",
                    "--bank",
                    "h2ns3",
                    "--margin",
                    ".1",
                    "--sweep-dir",
                    str(dest / "kd_hpo/rank_rep"),
                ]
            )
            grids = ["kd_gram_*"]
        else:
            for arm in ("grand", "new"):
                migrate_study(root / f"struct_hpo/{arm}", dest / f"struct_hpo/{arm}", pack)
                struct_hpo.main(
                    [
                        "--arm",
                        arm,
                        "--base-config",
                        f"configs/split_seed42/struct_{arm}.yaml",
                        "--sweep-dir",
                        str(dest / f"struct_hpo/{arm}"),
                    ]
                )
            grids = []
        for pattern in grids:
            for config in sorted(Path("configs/split_seed42/sweep").glob(pattern + ".yaml")):
                target = dest / "kd_hpo/grid" / config.stem
                if (target / "complete.json").exists():
                    continue
                target.mkdir(parents=True, exist_ok=True)
                data = yaml.safe_load(config.read_text())
                data["output_dir"] = str(target)
                path = target / "config.yaml"
                path.write_text(yaml.safe_dump(data, sort_keys=False))
                subprocess.run(
                    ["bash", "hpc/run.sh", "train", str(path), "--skip-test"], check=True
                )
        write_json(status, {"status": "complete", "selection_rule": SELECTION_RULE})
    except Exception as error:
        write_json(
            status, {"status": "failed", "error": str(error), "selection_rule": SELECTION_RULE}
        )
        raise


if __name__ == "__main__":
    main()

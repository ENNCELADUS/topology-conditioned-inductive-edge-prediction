"""Plot observed L3-PPI histories and compare terminal test reports with B0."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

EDGE_METRICS = ("auroc", "auprc", "accuracy", "f1", "mcc", "ece", "brier")
TOPOLOGY_METRICS = ("gs", "rd", "degree_mmd", "clustering_mmd", "spectral_mmd")
DEFAULT_BASELINE = Path("docs/results/split_seed42_geometric_20260910/b0_test_report.json")


def comparison_rows(baseline: dict[str, Any], winner: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract held-out metrics after checking available universe metadata."""
    for report in (baseline, winner):
        if report["schema_version"] != "test_protocol_v8":
            raise ValueError("Comparison requires test_protocol_v8 reports")
        if report["edge"]["strategy"] != "breadth_first":
            raise ValueError("Comparison requires breadth_first strategy")
        if report["edge"]["pairs_source"] != "test":
            raise ValueError("Comparison requires held-out test rows")
    for key in ("num_rows", "self_rows", "self_loops_included"):
        if baseline["edge"][key] != winner["edge"][key]:
            raise ValueError(f"Test row universe mismatch: edge.{key}")
    for key in ("n_pos", "n_neg"):
        if baseline["edge"]["metrics"][key] != winner["edge"]["metrics"][key]:
            raise ValueError(f"Test class count mismatch: {key}")
    sample_counts = [
        {
            size: metrics["sample_count"]
            for size, metrics in report["graph"]["fixed_threshold"]["test"]["per_size"].items()
        }
        for report in (baseline, winner)
    ]
    if sample_counts[0] != sample_counts[1]:
        raise ValueError("Test topology sample counts differ")
    rows = []
    for label, report in (("B0", baseline), ("L3-PPI", winner)):
        row = {
            "arm": label,
            "checkpoint_id": report["arm"]["checkpoint_id"],
            "selected_epoch": report["arm"]["selected_epoch"],
            **{key: report["edge"]["metrics"][key] for key in EDGE_METRICS},
        }
        topology = report["graph"]["fixed_threshold"]["test"]
        row.update(
            gs=topology["graph_similarity"]["bfs_macro"],
            rd=topology["relative_density"]["bfs_macro"],
        )
        row.update(
            {
                f"{name}_mmd": topology["mmd_ratio"][name]
                for name in ("degree", "clustering", "spectral")
            }
        )
        if not all(math.isfinite(float(row[key])) for key in (*EDGE_METRICS, *TOPOLOGY_METRICS)):
            raise ValueError("Non-finite terminal comparison metric")
        rows.append(row)
    return rows


def read_history(path: Path) -> list[dict[str, Any]]:
    """Read available rows, tolerating only an unfinished final JSONL record."""
    if not path.exists():
        return []
    text = path.read_text()
    lines = text.splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not text.endswith("\n"):
                break
            raise
    return rows


def runtime_row(name: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize measured training time and time-weighted throughput only."""
    measured = [row for row in history if "train_seconds" in row]
    seconds = sum(float(row["train_seconds"]) for row in measured)
    throughput_rows = [row for row in measured if "pairs_per_second" in row]
    throughput_seconds = sum(float(row["train_seconds"]) for row in throughput_rows)
    peaks = [float(row["peak_gpu_gib"]) for row in history if "peak_gpu_gib" in row]
    return {
        "stage": name,
        "observed_epochs": len(history),
        "train_seconds": seconds if measured else "",
        "pairs_per_second": (
            sum(
                float(row["pairs_per_second"]) * float(row["train_seconds"])
                for row in throughput_rows
            )
            / throughput_seconds
            if throughput_seconds
            else ""
        ),
        "peak_gpu_gib": max(peaks) if peaks else "",
    }


def plot_history(directory: Path, history: list[dict[str, Any]], output: Path) -> None:
    """Save phase-separated learning curves, including all five topology metrics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phase_names = list(dict.fromkeys(str(row["phase"]) for row in history))
    phases = {phase: [row for row in history if row["phase"] == phase] for phase in phase_names}
    offsets: dict[str, int] = {}
    offset = 0
    for phase, rows in phases.items():
        offsets[phase] = offset
        offset += max(int(row["epoch"]) for row in rows)
    selected_path = directory / "selection.json"
    selected = json.loads(selected_path.read_text()) if selected_path.exists() else {}
    selected_epoch = selected.get("selected_epoch", selected.get("epoch"))
    panels = [
        ("Loss", ("train_task_loss", "train_path_loss", "val_task_loss")),
        (
            "Active paths",
            ("train_soft_paths", "train_hard_paths", "val_soft_paths", "val_hard_paths"),
        ),
        ("Validation edge prediction", ("val_auprc", "val_auroc")),
        *((f"Validation {name}", (name,)) for name in TOPOLOGY_METRICS),
        ("Training throughput (pairs/s)", ("pairs_per_second",)),
    ]
    surrogate_only = phase_names == ["surrogate"]
    if surrogate_only:
        panels = [("Surrogate task loss", ("train_task_loss",)), panels[-1]]
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(
            1 if surrogate_only else 3,
            2 if surrogate_only else 3,
            figsize=(10, 4) if surrogate_only else (14, 10),
            constrained_layout=True,
        )
        for ax, (title, keys) in zip(axes.flat, panels, strict=True):
            has_data = False
            for phase, rows in phases.items():
                for key in keys:
                    values = [
                        (
                            int(row["epoch"]) + offsets[phase],
                            row.get("topology", {}).get("metrics", {}).get(key)
                            if key in TOPOLOGY_METRICS
                            else row.get(key),
                        )
                        for row in rows
                    ]
                    points = [(x, float(y)) for x, y in values if y is not None]
                    if points:
                        ax.plot(
                            [x for x, _ in points],
                            [y for _, y in points],
                            marker=".",
                            label=f"{phase}: {key}",
                        )
                        has_data = True
            for phase in phase_names[1:]:
                ax.axvline(offsets[phase] + 0.5, color="0.7", linewidth=0.7)
            if selected_epoch is not None and "joint" in offsets:
                ax.axvline(
                    offsets["joint"] + int(selected_epoch),
                    color="black",
                    linestyle="--",
                    linewidth=0.9,
                    label="selected joint epoch",
                )
            ax.set(title=title, xlabel="Epoch (phases concatenated)")
            ax.grid(alpha=0.2)
            if has_data:
                ax.legend(fontsize=6, frameon=False)
            else:
                ax.text(0.5, 0.5, "No observations yet", transform=ax.transAxes, ha="center")
        fig.suptitle(f"{directory.name}: observed training and validation (not held-out test)")
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{directory.name}_curves.{extension}", dpi=200)
        plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def generate_report(study_dir: Path, baseline_report: Path, output_dir: Path) -> dict[str, Any]:
    """Plot partial runs; write a comparison only from two terminal test reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    runtimes = []
    for directory in [study_dir / "surrogate", *sorted(study_dir.glob("trial_[0-9][0-9][0-9]"))]:
        history = read_history(directory / "metrics.jsonl")
        if history:
            plot_history(directory, history, output_dir)
            runtimes.append(runtime_row(directory.name, history))
    _write_csv(output_dir / "runtime.csv", runtimes)
    winner_path = study_dir / "winner.json"
    if not winner_path.exists():
        return {"status": "curves_only", "observed_stages": len(runtimes)}
    winner = json.loads(winner_path.read_text())
    # A local mirror keeps stage directory names but may have different roots.
    directory = study_dir / Path(winner["trial_dir"]).name
    report_path = directory / "test_report.json"
    if not report_path.exists() or not baseline_report.exists():
        return {"status": "curves_only", "observed_stages": len(runtimes)}
    if (directory / "failure.json").exists():
        raise ValueError("Winner has a failure artifact; terminal comparison is unavailable")
    rows = comparison_rows(
        json.loads(baseline_report.read_text()), json.loads(report_path.read_text())
    )
    _write_csv(output_dir / "comparison.csv", rows)
    columns = ("arm", "selected_epoch", *EDGE_METRICS, *TOPOLOGY_METRICS)
    lines = [
        "# L3-PPI versus existing B0",
        "",
        "Paper-based reproduction on our benchmark. One seed; whole-head replacement comparison.",
        "No new MLP baseline and no B0 retraining. Checkpoints were selected on validation.",
        "",
        f"Sources: B0 `{baseline_report}`; L3-PPI `{report_path}`.",
        "",
        "Held-out test metrics (topology uses each checkpoint's frozen validation threshold):",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                f"{row[key]:.6g}" if isinstance(row[key], float) else str(row[key])
                for key in columns
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "AUROC/AUPRC, Accuracy/F1/MCC and GS: higher; "
            "ECE/Brier and MMD ratios: lower; RD: toward 1.",
            "",
            "Row/class/self counts, strategy and per-size topology sample counts were compared.",
            "Matching counts alone does not establish exact pair identity. "
            "Canonical pair/label arrays require a separate comparison.",
            "",
            "Curves show observed training/validation only. "
            "Dashed lines mark the selected joint epoch.",
            "`runtime.csv` summarizes observed training seconds and time-weighted pairs/s.",
            "Peak GPU GiB is PyTorch allocated memory; it excludes CUDA runtime/reserved memory.",
            "Training timings exclude validation and feature-cache construction. "
            "B0 timing is unavailable here.",
            "No inference-speed comparison is inferred from training throughput.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")
    return {"status": "compared", "observed_stages": len(runtimes), "rows": rows}


def main() -> None:
    """Parse report locations and generate standalone artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = generate_report(args.study_dir, args.baseline_report, args.output_dir)
    sys.stdout.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

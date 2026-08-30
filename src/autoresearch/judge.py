"""CLI: judge one trial run directory against the incumbent.

Usage: ``python -m src.autoresearch.judge --incumbent DIR --trial DIR [--bands FILE]``.
Writes one JSON object to stdout: the ``Verdict`` fields plus ``incumbent``/``trial``
surface dicts (run_dir, selected_epoch, auprc, gs, rd, degree_mmd, clustering_mmd,
spectral_mmd, threshold, total_seconds).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.autoresearch.metrics_io import RunMetrics, read_run
from src.autoresearch.verdict import judge_runs


def main(argv: list[str] | None = None) -> int:
    """Write the verdict JSON for ``--trial`` vs ``--incumbent`` to stdout."""
    parser = argparse.ArgumentParser(prog="autoresearch-judge")
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--bands", type=Path, default=None)
    args = parser.parse_args(argv)

    bands: dict[str, Any] | None = None
    if args.bands is not None:
        loaded = json.loads(args.bands.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{args.bands}: bands file must be a JSON object")
        bands = loaded

    incumbent = read_run(args.incumbent)
    trial = read_run(args.trial)
    verdict = judge_runs(incumbent, trial, bands)
    payload = asdict(verdict) | {
        "incumbent": _surface(incumbent),
        "trial": _surface(trial),
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


def _surface(run: RunMetrics) -> dict[str, Any]:
    """Flatten one run's judge-facing surface for the JSON payload."""
    return {
        "run_dir": str(run.run_dir),
        "selected_epoch": run.selected_epoch,
        "auprc": run.auprc,
        "gs": run.topology.gs,
        "rd": run.topology.rd,
        "degree_mmd": run.topology.degree_mmd,
        "clustering_mmd": run.topology.clustering_mmd,
        "spectral_mmd": run.topology.spectral_mmd,
        "threshold": run.threshold,
        "total_seconds": run.total_seconds,
    }


if __name__ == "__main__":
    raise SystemExit(main())

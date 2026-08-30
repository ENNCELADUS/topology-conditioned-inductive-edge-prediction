"""CLI: judge one trial run directory against the incumbent.

Usage: ``python -m src.autoresearch.judge --incumbent DIR --trial DIR [--bands FILE]
[--incumbent-topology-every N]``. The cadence flag reselects the incumbent's epoch
under the campaign's ``eval.topology_every`` (a Phase-0 grid winner measured every
epoch must not be compared at a denser selection cadence than the trial). Writes one
JSON object to stdout: the ``Verdict`` fields plus ``incumbent``/``trial`` surface
dicts (run_dir, selected_epoch, auprc, gs, rd, degree_mmd, clustering_mmd,
spectral_mmd, threshold, total_seconds).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.autoresearch.metrics_io import read_run, surface
from src.autoresearch.verdict import judge_runs


def main(argv: list[str] | None = None) -> int:
    """Write the verdict JSON for ``--trial`` vs ``--incumbent`` to stdout."""
    parser = argparse.ArgumentParser(prog="autoresearch-judge")
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--bands", type=Path, default=None)
    parser.add_argument("--incumbent-topology-every", type=int, default=None)
    args = parser.parse_args(argv)

    bands: dict[str, Any] | None = None
    if args.bands is not None:
        loaded = json.loads(args.bands.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{args.bands}: bands file must be a JSON object")
        bands = loaded

    incumbent = read_run(args.incumbent, topology_every=args.incumbent_topology_every)
    trial = read_run(args.trial)
    verdict = judge_runs(incumbent, trial, bands)
    payload = asdict(verdict) | {
        "incumbent": surface(incumbent),
        "trial": surface(trial),
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

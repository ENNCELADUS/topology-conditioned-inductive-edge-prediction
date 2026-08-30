"""CLI: print one run's judge-facing surface for a campaign baseline row.

Usage: ``python -m src.autoresearch.baseline RUN_DIR [--topology-every N]``.
The cadence flag reselects the epoch with the frozen checkpoint selector
restricted to the campaign's ``eval.topology_every`` cadence, so a Phase-0
grid winner (topology measured every epoch) yields the exact surface its
cadence-N campaign trials are judged against; omitted, the published
``run_metadata.json`` selection is reported unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.autoresearch.metrics_io import read_run, surface


def main(argv: list[str] | None = None) -> int:
    """Write the surface JSON for ``run_dir`` to stdout."""
    parser = argparse.ArgumentParser(prog="autoresearch-baseline")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--topology-every", type=int, default=None)
    args = parser.parse_args(argv)
    run = read_run(args.run_dir, topology_every=args.topology_every)
    sys.stdout.write(json.dumps(surface(run), sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Append-only autoresearch trial ledger with validation and tolerant replay.

Row schema (all keys required on every row): ``trial`` (int, strictly
``max(existing)+1``, first row is 1), ``campaign``, ``commit`` (unique),
``config_hash``, ``output_dir`` (unique), ``hypothesis``, ``status`` (one of
``STATUSES``), ``metrics`` (exactly ``METRIC_KEYS``, all finite; ``None`` for
``crash``), ``selected_epoch``, ``total_seconds``, ``verdict`` (dict whose
``decision`` equals the status for ``keep``/``revert``; ``None`` for
``baseline``/``crash``), ``asi`` (free-form), ``timestamp``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

STATUSES = frozenset({"baseline", "keep", "revert", "crash"})
METRIC_KEYS = frozenset({"auprc", "gs", "rd", "degree_mmd", "clustering_mmd", "spectral_mmd"})
REQUIRED_KEYS = frozenset(
    {
        "trial",
        "campaign",
        "commit",
        "config_hash",
        "output_dir",
        "hypothesis",
        "status",
        "metrics",
        "selected_epoch",
        "total_seconds",
        "verdict",
        "asi",
        "timestamp",
    }
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Replay the ledger, skipping unparseable lines (torn tail writes)."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def append_row(path: Path, row: Mapping[str, Any]) -> None:
    """Validate ``row`` against the existing ledger, then append it durably.

    A torn tail (a final line without a newline, from an interrupted write)
    is healed by prefixing a newline so the new row starts its own line.

    Raises:
        ValueError: On any schema, monotonicity, uniqueness, or consistency
            violation. A rejected row writes nothing.
    """
    existing = read_rows(path)
    _validate(row, existing)
    raw = path.read_bytes() if path.exists() else b""
    with path.open("a", encoding="utf-8") as handle:
        if raw and not raw.endswith(b"\n"):
            handle.write("\n")
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _validate(row: Mapping[str, Any], existing: list[dict[str, Any]]) -> None:
    """Enforce the complete row contract against the replayed history."""
    row_keys = set(row)
    missing = REQUIRED_KEYS - row_keys
    unexpected = row_keys - REQUIRED_KEYS
    if missing:
        raise ValueError(f"ledger row missing keys: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"ledger row has unexpected keys: {sorted(unexpected)}")

    trial = row["trial"]
    if not isinstance(trial, int) or isinstance(trial, bool):
        raise ValueError(f"trial must be an int, got {trial!r}")
    trials = [
        entry["trial"]
        for entry in existing
        if isinstance(entry.get("trial"), int) and not isinstance(entry["trial"], bool)
    ]
    expected_trial = max(trials) + 1 if trials else 1
    if trial != expected_trial:
        raise ValueError(f"trial must be {expected_trial}, got {trial!r}")

    for key in ("campaign", "commit", "config_hash", "output_dir", "hypothesis", "timestamp"):
        if not isinstance(row[key], str):
            raise ValueError(f"{key} must be a string, got {row[key]!r}")
    if row["output_dir"] in {entry.get("output_dir") for entry in existing}:
        raise ValueError(f"duplicate output_dir {row['output_dir']!r}")
    if row["commit"] in {entry.get("commit") for entry in existing}:
        raise ValueError(f"duplicate commit {row['commit']!r}")

    status = row["status"]
    if not isinstance(status, str) or status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")

    selected_epoch = row["selected_epoch"]
    if status != "crash" and (
        not isinstance(selected_epoch, int)
        or isinstance(selected_epoch, bool)
        or selected_epoch <= 0
    ):
        raise ValueError(f"non-crash selected_epoch must be a positive int, got {selected_epoch!r}")
    total_seconds = row["total_seconds"]
    if status != "crash" and (
        not isinstance(total_seconds, int | float)
        or isinstance(total_seconds, bool)
        or not math.isfinite(float(total_seconds))
        or total_seconds < 0
    ):
        raise ValueError(
            f"non-crash total_seconds must be a finite nonnegative number, got {total_seconds!r}"
        )
    if (
        status == "crash"
        and total_seconds is not None
        and (
            not isinstance(total_seconds, int | float)
            or isinstance(total_seconds, bool)
            or not math.isfinite(float(total_seconds))
        )
    ):
        raise ValueError(f"total_seconds must be a finite number or None, got {total_seconds!r}")

    verdict = row["verdict"]
    if status in {"keep", "revert"}:
        if not isinstance(verdict, Mapping) or verdict.get("decision") != status:
            raise ValueError(f"status {status!r} requires a verdict with decision={status!r}")
    elif verdict is not None:
        raise ValueError(f"status {status!r} must carry verdict=None")

    metrics = row["metrics"]
    if status == "crash":
        if metrics is not None or selected_epoch is not None or total_seconds is not None:
            raise ValueError("crash rows must carry metrics, selected_epoch, total_seconds=None")
        return
    if not isinstance(metrics, Mapping) or set(metrics) != METRIC_KEYS:
        raise ValueError(f"metrics must carry exactly {sorted(METRIC_KEYS)}")
    for key in sorted(METRIC_KEYS):
        value = metrics[key]
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"metric {key} must be finite, got {value!r}")


def main(argv: list[str] | None = None) -> int:
    """CLI: append one validated row (a JSON object file) to the ledger."""
    parser = argparse.ArgumentParser(prog="autoresearch-ledger")
    parser.add_argument("ledger", type=Path)
    parser.add_argument("row", type=Path)
    args = parser.parse_args(argv)
    row = json.loads(args.row.read_text(encoding="utf-8"))
    if not isinstance(row, dict):
        raise ValueError(f"{args.row}: row file must contain one JSON object")
    append_row(args.ledger, row)
    sys.stdout.write(f"appended trial {row['trial']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

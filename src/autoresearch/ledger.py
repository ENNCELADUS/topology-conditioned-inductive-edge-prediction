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
    raise NotImplementedError("scaffold: plan Task 5")


def append_row(path: Path, row: Mapping[str, Any]) -> None:
    """Validate ``row`` against the existing ledger, then append it durably.

    A torn tail (a final line without a newline, from an interrupted write)
    is healed by prefixing a newline so the new row starts its own line.

    Raises:
        ValueError: On any schema, monotonicity, uniqueness, or consistency
            violation. A rejected row writes nothing.
    """
    raise NotImplementedError("scaffold: plan Task 5")


def main(argv: list[str] | None = None) -> int:
    """CLI: append one validated row (a JSON object file) to the ledger."""
    raise NotImplementedError("scaffold: plan Task 5")


if __name__ == "__main__":
    raise SystemExit(main())

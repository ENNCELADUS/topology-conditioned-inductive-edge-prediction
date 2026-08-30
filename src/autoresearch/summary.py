"""Deterministic cold-start digest of the autoresearch ledger.

Usage: ``python -m src.autoresearch.summary LEDGER_PATH [--last N]``. Renders
campaign standings (trials, keeps, incumbent surface) plus the last N trials,
one line each — no LLM content, so any fresh session resumes from files alone.
"""

from __future__ import annotations

from pathlib import Path


def render_summary(ledger_path: Path, last: int = 10) -> str:
    """Render campaign standings plus the last ``last`` trials, one line each."""
    raise NotImplementedError("scaffold: plan Task 6")


def main(argv: list[str] | None = None) -> int:
    """CLI: write the ledger digest to stdout."""
    raise NotImplementedError("scaffold: plan Task 6")


if __name__ == "__main__":
    raise SystemExit(main())

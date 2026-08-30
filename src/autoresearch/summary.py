"""Deterministic cold-start digest of the autoresearch ledger.

Usage: ``python -m src.autoresearch.summary LEDGER_PATH [--last N]``. Renders
campaign standings (trials, keeps, incumbent surface) plus the last N trials,
one line each — no LLM content, so any fresh session resumes from files alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from src.autoresearch.ledger import read_rows

_INCUMBENT_METRIC_ORDER = ("gs", "rd", "degree_mmd", "clustering_mmd", "spectral_mmd", "auprc")


def _open_ideas() -> list[str]:
    """Return non-empty lines from the operator's default ideas backlog."""
    path = Path("autoresearch/ideas.md")
    if not path.is_file():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(("- ", "* "))
    ]


def render_summary(ledger_path: Path, last: int = 10) -> str:
    """Render campaign standings plus the last ``last`` trials, one line each."""
    rows = read_rows(ledger_path)
    if not rows:
        ideas = _open_ideas()
        if not ideas:
            return "ledger empty; no trials recorded\n"
        return "\n".join(["ledger empty; no trials recorded", "open ideas:", *ideas]) + "\n"

    lines = ["# autoresearch summary"]
    campaigns: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        campaigns.setdefault(str(row.get("campaign", "?")), []).append(row)
    for campaign in sorted(campaigns):
        entries = campaigns[campaign]
        keeps = [entry for entry in entries if entry.get("status") in {"baseline", "keep"}]
        keep_count = sum(entry.get("status") == "keep" for entry in entries)
        line = f"campaign {campaign}: trials={len(entries)} keeps={keep_count}"
        if keeps:
            incumbent = keeps[-1]
            metrics = incumbent.get("metrics") or {}
            surface = " ".join(f"{key}={metrics.get(key)}" for key in _INCUMBENT_METRIC_ORDER)
            line += f" incumbent={incumbent.get('output_dir')} {surface}"
        lines.append(line)

    recent = rows[-last:] if last > 0 else []
    lines.append(f"last {len(recent)} trials:")
    for row in recent:
        verdict = row.get("verdict") or {}
        improved = ",".join(verdict.get("improved", [])) or "-"
        regressed = ",".join(verdict.get("regressed", [])) or "-"
        lines.append(
            f"#{row.get('trial')} [{row.get('campaign')}] {row.get('status')}"
            f" | {row.get('hypothesis')} | improved:{improved} regressed:{regressed}"
            f" | asi:{row.get('asi') or '-'}"
        )

    ideas = _open_ideas()
    if ideas:
        lines.append("open ideas:")
        lines.extend(ideas)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI: write the ledger digest to stdout."""
    parser = argparse.ArgumentParser(prog="autoresearch-summary")
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--last", type=int, default=10)
    args = parser.parse_args(argv)
    sys.stdout.write(render_summary(args.ledger, last=args.last))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI: judge one trial run directory against the incumbent.

Usage: ``python -m src.autoresearch.judge --incumbent DIR --trial DIR [--bands FILE]``.
Writes one JSON object to stdout: the ``Verdict`` fields plus ``incumbent``/``trial``
surface dicts (run_dir, selected_epoch, auprc, gs, rd, degree_mmd, clustering_mmd,
spectral_mmd, threshold, total_seconds).
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    """Write the verdict JSON for ``--trial`` vs ``--incumbent`` to stdout."""
    raise NotImplementedError("scaffold: plan Task 3")


if __name__ == "__main__":
    raise SystemExit(main())

"""Append-only autoresearch trial ledger with validation and tolerant replay.

Row schema (all keys required on every row): ``trial`` (int, strictly
``max(existing)+1``, first row is 1), ``campaign``, ``commit`` (unique),
``config_hash``, ``output_dir`` (unique), ``hypothesis``, ``status`` (one of
``STATUSES``), ``metrics`` (exactly ``METRIC_KEYS``, all finite; ``None`` for
``crash``), ``selected_epoch``, ``total_seconds``, ``verdict`` (dict whose
``decision`` equals the status for ``keep``/``revert``; ``None`` for
``baseline``/``crash``; otherwise the exact frozen-judge evidence schema),
``asi`` (free-form), ``timestamp``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.autoresearch.verdict import METRIC_NAMES, verdict_reasons

STATUSES = frozenset({"baseline", "keep", "revert", "crash"})
CAMPAIGNS = frozenset({"kd_logit", "kd_rank", "kd_gram", "kd_rep"})
METRIC_KEYS = frozenset({"auprc", "gs", "rd", "degree_mmd", "clustering_mmd", "spectral_mmd"})
VERDICT_KEYS = frozenset({"decision", "improved", "regressed", "deltas", "auprc_delta", "reasons"})
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
        handle.flush()
        os.fsync(handle.fileno())


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
    campaign = row["campaign"]
    if campaign not in CAMPAIGNS:
        raise ValueError(f"unknown campaign {campaign!r}")
    campaign_rows = [entry for entry in existing if entry.get("campaign") == campaign]
    if not campaign_rows and status != "baseline":
        raise ValueError(f"campaign {campaign!r} must start with exactly one baseline")
    if campaign_rows and status == "baseline":
        raise ValueError(f"campaign {campaign!r} already has its baseline")

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

    metrics = row["metrics"]
    if status == "crash":
        if (
            metrics is not None
            or selected_epoch is not None
            or total_seconds is not None
            or row["verdict"] is not None
        ):
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
    if metrics["rd"] <= 0:
        raise ValueError(f"metric rd must be positive, got {metrics['rd']!r}")

    verdict = row["verdict"]
    if status == "baseline":
        if verdict is not None:
            raise ValueError("status 'baseline' must carry verdict=None")
        return
    incumbent = next(
        entry.get("metrics")
        for entry in reversed(campaign_rows)
        if entry.get("status") in {"baseline", "keep"}
    )
    if not isinstance(incumbent, Mapping):
        raise ValueError(f"campaign {campaign!r} has no valid incumbent metrics")
    _validate_verdict(verdict, status, metrics, incumbent)


def _validate_verdict(
    verdict: object,
    status: str,
    metrics: Mapping[str, Any],
    incumbent: Mapping[str, Any],
) -> None:
    """Validate the complete frozen-judge evidence embedded in a ledger row."""
    if not isinstance(verdict, Mapping):
        raise ValueError(f"status {status!r} requires a complete verdict")
    if set(verdict) != VERDICT_KEYS:
        raise ValueError(f"verdict must carry exactly {sorted(VERDICT_KEYS)}")
    decision = verdict["decision"]
    if decision not in {"keep", "revert"} or decision != status:
        raise ValueError(f"status {status!r} requires verdict decision={status!r}")

    improved = _metric_names(verdict["improved"], "improved")
    regressed = _metric_names(verdict["regressed"], "regressed")
    if set(improved) & set(regressed):
        raise ValueError("verdict improved and regressed metrics must be disjoint")
    expected_decision = "keep" if improved and not regressed else "revert"
    if decision != expected_decision:
        raise ValueError("verdict decision contradicts improved/regressed evidence")

    deltas = verdict["deltas"]
    if not isinstance(deltas, Mapping) or set(deltas) != set(METRIC_NAMES):
        raise ValueError(f"verdict deltas must carry exactly {list(METRIC_NAMES)}")
    expected_deltas = _oriented_deltas(incumbent, metrics)
    for name in METRIC_NAMES:
        actual = _finite_number(deltas[name], f"verdict delta {name}")
        if actual != expected_deltas[name]:
            raise ValueError(
                f"verdict delta {name} must equal exact oriented delta {expected_deltas[name]!r}"
            )
    if any(float(deltas[name]) >= 0.0 for name in improved):
        raise ValueError("improved metrics must carry negative oriented deltas")
    if any(float(deltas[name]) <= 0.0 for name in regressed):
        raise ValueError("regressed metrics must carry positive oriented deltas")
    auprc_delta = _finite_number(verdict["auprc_delta"], "verdict auprc_delta")
    expected_auprc_delta = float(metrics["auprc"]) - float(incumbent["auprc"])
    if auprc_delta != expected_auprc_delta:
        raise ValueError(f"verdict auprc_delta must equal {expected_auprc_delta!r}")

    reasons = verdict["reasons"]
    expected_reasons = list(verdict_reasons(decision, improved, regressed))
    if reasons != expected_reasons:
        raise ValueError(f"verdict reasons must equal {expected_reasons!r}")


def _metric_names(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(name, str) for name in value):
        raise ValueError(f"verdict {label} must be a metric-name list")
    expected = [name for name in METRIC_NAMES if name in value]
    if value != expected:
        raise ValueError(f"verdict {label} must contain known metrics in frozen order")
    return value


def _finite_number(value: object, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    return float(value)


def _oriented_deltas(incumbent: Mapping[str, Any], trial: Mapping[str, Any]) -> dict[str, float]:
    return {
        "gs": float(incumbent["gs"]) - float(trial["gs"]),
        "log_rd": abs(math.log(float(trial["rd"]))) - abs(math.log(float(incumbent["rd"]))),
        "degree_mmd": float(trial["degree_mmd"]) - float(incumbent["degree_mmd"]),
        "clustering_mmd": float(trial["clustering_mmd"]) - float(incumbent["clustering_mmd"]),
        "spectral_mmd": float(trial["spectral_mmd"]) - float(incumbent["spectral_mmd"]),
    }


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

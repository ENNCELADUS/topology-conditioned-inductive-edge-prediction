"""Read-only observation reports for a completed EgoStitch E2E formal run.

Prints the three registered rev-3.2 observation targets (registration v5
``observation_plan``) from a run's published artifacts:

1. Generator clipping — per-group pre-clip norms, clip coefficients, and the
   margins to the registered (telemetry-only) immediate/persistent thresholds.
2. Slot collapse — step-0 slot health plus the per-validation h-cosine and
   plan rank-1 residual trajectories against the collapse predicates.
3. End-ramp precision differential — the BF16/fp32 logit differentials at the
   end ramp and the selected checkpoint against the registered bounds.

This module is analysis output only. It reads ``profile.json``,
``metrics.jsonl``, and ``run_metadata.json`` from a run output directory and
never writes, gates, or grades anything.

CLI::

    python -m src.experiments.observe_e2e_formal outputs/egostitch_e2e_stage1_v3/full
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast

_IMMEDIATE_COEFFICIENT = 1.0e-3
_PERSISTENT_COEFFICIENT = 0.1
_PERSISTENT_STEPS = 10
_FAMILY_RATIO = 50.0
_H_COSINE_TRIP = 0.95
_PLAN_RESIDUAL_TRIP = 0.05
_RELATIVE_L2_BOUND = 0.05
_RESIDUAL_CORRELATION_FLOOR = 0.999


def _load_json(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(cast(dict[str, object], json.loads(line)))
    return rows


def _fmt(value: object, digits: int = 6) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return f"{value:.{digits}g}"


def _event_counts(events: Iterable[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        kind = str(event.get("kind"))
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def report_generator_clipping(profile: Mapping[str, object]) -> list[str]:
    """Summarize per-group clipping telemetry with registered-threshold margins."""
    lines = ["== 1. Generator clipping =="]
    steps = cast(list[Mapping[str, object]], profile.get("optimizer_step_gradients") or [])
    if not steps:
        lines.append("no optimizer_step_gradients recorded")
        return lines
    per_group: dict[str, dict[str, object]] = {}
    streaks: dict[str, int] = {}
    for row in steps:
        records = cast(
            Mapping[str, Mapping[str, object]],
            row.get("optimizer_group_gradients") or {},
        )
        for name, record in records.items():
            raw_norm = record.get("norm")
            raw_coefficient = record.get("clip_coefficient")
            active = bool(record.get("active"))
            norm = float(raw_norm) if isinstance(raw_norm, (int, float)) else float("nan")
            coefficient = (
                float(raw_coefficient)
                if isinstance(raw_coefficient, (int, float))
                else float("nan")
            )
            state = per_group.setdefault(
                name,
                {
                    "max_norm": float("-inf"),
                    "max_norm_step": None,
                    "min_coefficient": float("inf"),
                    "min_coefficient_step": None,
                    "immediate_misses": 0,
                    "persistent_step_count": 0,
                    "max_streak": 0,
                    "steps": 0,
                },
            )
            if not active or not math.isfinite(coefficient):
                streaks[name] = 0
                continue
            state["steps"] = cast(int, state["steps"]) + 1
            if math.isfinite(norm) and norm > cast(float, state["max_norm"]):
                state["max_norm"] = norm
                state["max_norm_step"] = row.get("step")
            if coefficient < cast(float, state["min_coefficient"]):
                state["min_coefficient"] = coefficient
                state["min_coefficient_step"] = row.get("step")
            if coefficient < _IMMEDIATE_COEFFICIENT:
                state["immediate_misses"] = cast(int, state["immediate_misses"]) + 1
            if coefficient < _PERSISTENT_COEFFICIENT:
                state["persistent_step_count"] = cast(int, state["persistent_step_count"]) + 1
                streaks[name] = streaks.get(name, 0) + 1
            else:
                streaks[name] = 0
            state["max_streak"] = max(cast(int, state["max_streak"]), streaks.get(name, 0))
    for name in sorted(per_group):
        state = per_group[name]
        lines.append(
            f"group {name}: steps={state['steps']} "
            f"max_pre_clip_norm={_fmt(state['max_norm'])} (step {state['max_norm_step']}) "
            f"min_clip_coefficient={_fmt(state['min_coefficient'])} "
            f"(step {state['min_coefficient_step']})"
        )
        lines.append(
            f"  immediate (<{_IMMEDIATE_COEFFICIENT:g}): {state['immediate_misses']} steps; "
            f"persistent (<{_PERSISTENT_COEFFICIENT:g}): {state['persistent_step_count']} steps, "
            f"max consecutive streak {state['max_streak']} vs registered {_PERSISTENT_STEPS}"
        )
    series = cast(list[Mapping[str, object]], profile.get("gradient_norm_series") or [])
    worst_ratio = float("-inf")
    worst_ratio_probe: object = None
    for probe in series:
        ratios = cast(Mapping[str, object], probe.get("family_group_ratios") or {})
        flat: list[tuple[str, object]] = []
        for key, value in ratios.items():
            if isinstance(value, Mapping):
                flat.extend((f"{key}/{sub}", item) for sub, item in value.items())
            else:
                flat.append((key, value))
        for key, value in flat:
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                if float(value) > worst_ratio:
                    worst_ratio = float(value)
                    worst_ratio_probe = (probe.get("step"), key)
    if worst_ratio > float("-inf"):
        lines.append(
            f"family probes: {len(series)}; worst family ratio {_fmt(worst_ratio)} "
            f"at {worst_ratio_probe} vs registered {_FAMILY_RATIO:g}"
        )
    events = cast(list[Mapping[str, object]], profile.get("quality_guard_events") or [])
    counts = _event_counts(events)
    lines.append(
        "quality events: "
        f"optimizer_gradient={counts.get('optimizer_gradient', 0)} "
        f"family_gradient={counts.get('family_gradient', 0)}"
    )
    return lines


def report_slot_collapse(
    profile: Mapping[str, object], history: Sequence[Mapping[str, object]]
) -> list[str]:
    """Summarize step-0 slot health and per-validation collapse trajectories."""
    lines = ["== 2. Slot collapse =="]
    initial = cast(Mapping[str, object] | None, profile.get("initial_slot_health"))
    if isinstance(initial, Mapping):
        h0 = initial.get("h_pairwise_cosine_mean")
        lines.append(
            f"step-0: h_pairwise_cosine_mean={_fmt(h0)} vs trip {_H_COSINE_TRIP:g}; "
            f"quality_threshold_missed={initial.get('quality_threshold_missed')}"
        )
    else:
        lines.append("no initial_slot_health recorded")
    h_series: list[tuple[int, float]] = []
    residual_series: list[tuple[int, float]] = []
    for entry in history:
        fidelity = cast(Mapping[str, object], entry.get("fidelity") or {})
        epoch = int(cast(float, entry.get("epoch", -1)))
        h_value = fidelity.get("h_pairwise_cosine_mean")
        residual = fidelity.get("plan_rank1_marginal_residual")
        if isinstance(h_value, (int, float)):
            h_series.append((epoch, float(h_value)))
        if isinstance(residual, (int, float)):
            residual_series.append((epoch, float(residual)))
    if h_series:
        peak_epoch, peak = max(h_series, key=lambda item: item[1])
        lines.append(
            f"h cosine: final={_fmt(h_series[-1][1])} (epoch {h_series[-1][0]}), "
            f"peak={_fmt(peak)} (epoch {peak_epoch}), margin to {_H_COSINE_TRIP:g}: "
            f"{_fmt(_H_COSINE_TRIP - peak)}"
        )
    if residual_series:
        low_epoch, low = min(residual_series, key=lambda item: item[1])
        lines.append(
            f"plan rank-1 residual: final={_fmt(residual_series[-1][1])} "
            f"(epoch {residual_series[-1][0]}), min={_fmt(low)} (epoch {low_epoch}), "
            f"margin to {_PLAN_RESIDUAL_TRIP:g}: {_fmt(low - _PLAN_RESIDUAL_TRIP)}"
        )
    tripped = [
        int(cast(float, entry.get("epoch", -1)))
        for entry in history
        if bool(
            cast(Mapping[str, object], entry.get("quality_thresholds") or {}).get("slot_collapse")
        )
    ]
    lines.append(f"epochs with slot_collapse predicate true: {tripped or 'none'}")
    events = cast(list[Mapping[str, object]], profile.get("quality_guard_events") or [])
    counts = _event_counts(events)
    lines.append(
        "quality events: "
        f"initial_slot_collapse={counts.get('initial_slot_collapse', 0)} "
        f"slot_collapse={counts.get('slot_collapse', 0)} "
        f"validation_logit_collapse={counts.get('validation_logit_collapse', 0)}"
    )
    return lines


def report_precision_differential(profile: Mapping[str, object]) -> list[str]:
    """Summarize the end-ramp and selected-checkpoint precision differentials."""
    lines = ["== 3. End-ramp precision differential =="]
    container = cast(Mapping[str, object], profile.get("precision_differential") or {})
    for label in ("end_ramp", "selected"):
        payload = container.get(label)
        if not isinstance(payload, Mapping):
            lines.append(f"{label}: not recorded")
            continue
        lines.append(
            f"{label}: quality_pass={payload.get('quality_pass')} "
            f"full_relative_l2={_fmt(payload.get('full_relative_l2'))} "
            f"f_logit_relative_l2={_fmt(payload.get('f_logit_relative_l2'))} "
            f"residual_relative_l2={_fmt(payload.get('residual_relative_l2'))} "
            f"(bound {_RELATIVE_L2_BOUND:g}) "
            f"residual_correlation={_fmt(payload.get('residual_correlation'))} "
            f"(floor {_RESIDUAL_CORRELATION_FLOOR:g})"
        )
        failures = payload.get("quality_failures")
        if failures:
            lines.append(f"  quality_failures: {failures}")
    events = cast(list[Mapping[str, object]], profile.get("quality_guard_events") or [])
    counts = _event_counts(events)
    lines.append(
        "quality events: "
        f"end_ramp_precision={counts.get('end_ramp_precision', 0)} "
        f"selected_precision={counts.get('selected_precision', 0)}"
    )
    return lines


def build_report(output_dir: Path) -> str:
    """Assemble the three observation reports for one run output directory."""
    profile_path = output_dir / "profile.json"
    metrics_path = output_dir / "metrics.jsonl"
    metadata_path = output_dir / "run_metadata.json"
    if not profile_path.is_file():
        raise FileNotFoundError(f"profile.json not found under {output_dir}")
    profile = _load_json(profile_path)
    history: list[dict[str, object]] = _load_jsonl(metrics_path) if metrics_path.is_file() else []
    header = [f"EgoStitch E2E formal-run observation report: {output_dir}"]
    if metadata_path.is_file():
        metadata = _load_json(metadata_path)
        header.append(
            f"arm={profile.get('arm')} run_kind={profile.get('run_kind')} "
            f"status={metadata.get('status')} checkpoint_id={metadata.get('checkpoint_id')} "
            f"quality_guards_passed={profile.get('quality_guards_passed')}"
        )
    sections = [
        report_generator_clipping(profile),
        report_slot_collapse(profile, history),
        report_precision_differential(profile),
    ]
    return "\n".join(header + [line for section in sections for line in ("", *section)])


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="formal-run output directory")
    args = parser.parse_args(argv)
    try:
        print(build_report(args.output_dir))
    except FileNotFoundError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

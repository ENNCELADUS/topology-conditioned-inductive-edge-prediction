"""Unattended Optuna sweep for the strict-LLP ``kd_rank`` arm.

Runs on the H20 container: an ask-and-tell TPE loop proposes
``(w_rank, w_dist, context bank, margin)``, launches one grid-protocol
training per trial through ``hpc/run.sh train --skip-test``, and scores the
cadence-2 V_val surface as (GS max, geometric-mean MMD ratio min) with an
``|log RD|`` soft constraint. The feasible Pareto front is advisory: the
recorded winner comes from the frozen five-metric undominated verdict.
Spec: ``docs/superpowers/specs/2026-09-01-kd-rank-strict-llp-optuna-hpo-design.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.distill.config import DistillConfig


@dataclass(frozen=True)
class BankSpec:
    """One frozen context bank: sampler composition and artifact path."""

    rw_step: int
    hops: int
    ns_rate: int
    path: str


BANKS: dict[str, BankSpec] = {
    "h2ns1": BankSpec(3, 2, 1, "outputs/distill/kd_ctx_targets_breadth_first"),
    "h2ns3": BankSpec(3, 2, 3, "outputs/distill/kd_ctx_targets_breadth_first_h2ns3"),
    "h2ns5": BankSpec(3, 2, 5, "outputs/distill/kd_ctx_targets_breadth_first_h2ns5"),
    "h3ns3": BankSpec(3, 3, 3, "outputs/distill/kd_ctx_targets_breadth_first_h3ns3"),
}

ENQUEUED_PRIORS: tuple[dict[str, object], ...] = (
    {"w_rank": 1.0, "w_dist": 1.0, "bank": "h2ns1", "margin": 0.1},
    {"w_rank": 0.1, "w_dist": 10.0, "bank": "h2ns1", "margin": 0.1},
    {"w_rank": 0.1, "w_dist": 10.0, "bank": "h2ns3", "margin": 0.1},
    {"w_rank": 0.01, "w_dist": 10.0, "bank": "h2ns5", "margin": 0.1},
    {"w_rank": 0.1, "w_dist": 100.0, "bank": "h2ns3", "margin": 0.1},
    {"w_rank": 0.1, "w_dist": 10.0, "bank": "h3ns3", "margin": 0.1},
)


def materialize_trial_config(
    base_config: Path, params: Mapping[str, object], trial_number: int, sweep_dir: Path
) -> Path:
    """Write trial ``trial_number``'s config; only the five whitelisted keys differ.

    Raises:
        KeyError: On an unknown bank name.
        ValueError: If the resulting ``distill`` section is illegal.
    """
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    cfg["output_dir"] = str(sweep_dir / f"trial_{trial_number:03d}")
    distill = dict(cfg["distill"])
    distill["w_rank"] = float(params["w_rank"])  # type: ignore[arg-type]
    distill["w_dist"] = float(params["w_dist"])  # type: ignore[arg-type]
    distill["margin"] = float(params["margin"])  # type: ignore[arg-type]
    distill["context_targets_path"] = BANKS[str(params["bank"])].path
    cfg["distill"] = distill
    DistillConfig.from_mapping(distill)
    config_path = sweep_dir / "configs" / f"trial_{trial_number:03d}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return config_path

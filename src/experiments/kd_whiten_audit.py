r"""Whitened-axis audit of KD row-target banks for representation KD.

Whitens each bank's teacher vector onto its top-``k`` PCA axes (unit variance
on training rows, the coordinates a whitened representation KD would regress)
and asks, per axis, whether it is worth distilling into the endpoint-only
student: is the axis structure (``descriptors -> axis``), is it reachable from
the student's input (``content -> axis``), and is it already spent by the edge
decision (``teacher_logit -> axis``, ``student_logit -> axis``)?

    python -m src.experiments.kd_whiten_audit --config configs/b0_v31_breadth_first.yaml \\
        --bank topo=outputs/distill/kd_row_targets_breadth_first \\
        --student control=outputs/b1_row_kd_hpo/kd_control/scores/val_cls.npz \\
        --f0-cache outputs/f0_cache/f0_matrix.pt --output outputs/distill/kd_whiten_audit.json

Content and content+logit probes are linear ridge fits on 80% of the training
block; descriptor and logit probes are gradient-boosted trees (few inputs, so a
nonlinear read is cheap). Student logits exist only on the V_val block, so those
probes fit on one random half of V_val and score the other half; the teacher
logit is probed the same way for comparison. Each axis also reports its fp16
storage noise floor and its mean shift on the V_val block in whitened units.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor

from src.distill.artifacts import KDRowTargets, load_kd_targets
from src.distill.struct_targets import STRUCT_NAMES, structural_targets
from src.distill.teacher_targets import truth_graph_for_kd
from src.experiments.kd_rep_audit import RidgeProbe, content_features, r2_columns
from src.train_b0 import assemble_data, load_config

logger = logging.getLogger(__name__)

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class WhitenedAxes:
    """Top-``k`` PCA coordinates, unit variance on the training block."""

    train: F64
    val: F64
    axis_std: F64
    var_share: F64
    val_shift: F64
    val_std: F64


def whiten_axes(rep_tr: F64, rep_va: F64, *, k: int) -> WhitenedAxes:
    """Project both blocks onto the training block's top-``k`` centered PCA axes.

    Raises:
        ValueError: If ``k`` exceeds the bank's dimensions or numerical rank.
    """
    if k < 1 or k > min(rep_tr.shape):
        raise ValueError(f"k must be in [1, {min(rep_tr.shape)}], got {k}")
    mu = rep_tr.mean(axis=0)
    _, sing, vt = np.linalg.svd(rep_tr - mu, full_matrices=False)
    var = sing**2 / rep_tr.shape[0]
    axis_std = np.sqrt(var[:k])
    if not np.all(axis_std > 0.0):
        raise ValueError(f"k={k} exceeds the bank's numerical rank")
    train = ((rep_tr - mu) @ vt[:k].T) / axis_std
    val = ((rep_va - mu) @ vt[:k].T) / axis_std
    return WhitenedAxes(
        train, val, axis_std, var[:k] / var.sum(), val.mean(axis=0), val.std(axis=0)
    )


def fp16_noise_floor(rep: F64) -> float:
    """Std of uniform fp16 rounding noise projected onto one unit-norm axis."""
    magnitude = np.abs(rep[rep != 0.0])
    ulp = np.exp2(np.floor(np.log2(magnitude)) - 10.0)
    return float(np.sqrt(np.mean(ulp**2) / 12.0))


def probe_axes(
    x_tr: F64,
    axes_tr: F64,
    x_va: F64,
    axes_va: F64,
    *,
    fit: NDArray[np.bool_],
    nonlinear: bool,
    seed: int,
) -> dict[str, list[float]]:
    """Per-axis R^2 on the held-out training rows and on the V_val block."""
    if nonlinear:
        pred_ho = np.empty((int((~fit).sum()), axes_tr.shape[1]))
        pred_va = np.empty_like(axes_va)
        for j in range(axes_tr.shape[1]):
            model = HistGradientBoostingRegressor(max_iter=200, random_state=seed)
            model.fit(x_tr[fit], axes_tr[fit, j])
            pred_ho[:, j] = model.predict(x_tr[~fit])
            pred_va[:, j] = model.predict(x_va)
    else:
        probe = RidgeProbe(x_tr[fit], axes_tr[fit])
        pred_ho, pred_va = probe.predict(x_tr[~fit]), probe.predict(x_va)
    return {
        "train_holdout": r2_columns(axes_tr[~fit], pred_ho).tolist(),
        "val": r2_columns(axes_va, pred_va).tolist(),
    }


def align_student_logits(bank: KDRowTargets, scores: Path) -> F64:
    """Student logits for the bank's V_val rows, joined on unordered node-id pairs.

    Raises:
        ValueError: If any bank V_val row is missing from the score artifact.
    """
    data = np.load(scores, allow_pickle=False)
    nodes = [str(node) for node in data["node_ids"].tolist()]
    table = {
        frozenset((nodes[u], nodes[v])): float(logit)
        for u, v, logit in zip(data["u_idx"], data["v_idx"], data["logit"], strict=True)
    }
    keys = [
        frozenset((bank.node_ids[a], bank.node_ids[b]))
        for a, b in zip(bank.val_pair_a_idx.tolist(), bank.val_pair_b_idx.tolist(), strict=True)
    ]
    missing = sum(key not in table for key in keys)
    if missing:
        raise ValueError(f"{scores}: {missing} bank V_val rows have no student score")
    return np.array([table[key] for key in keys], dtype=np.float64)


def audit_bank(
    bank: KDRowTargets,
    graph: nx.Graph,
    f0: F64,
    f0_index: dict[str, int],
    students: dict[str, F64],
    *,
    k: int,
    seed: int,
) -> dict[str, object]:
    """Whitened-axis worth-distilling report for one bank."""
    rng = np.random.default_rng(seed)
    node_ids = list(bank.node_ids)
    rep_tr = bank.teacher_rep.astype(np.float64)
    white = whiten_axes(rep_tr, bank.val_teacher_rep.astype(np.float64), k=k)
    logit_tr = bank.teacher_logit.astype(np.float64)[:, None]
    logit_va = bank.val_teacher_logit.astype(np.float64)[:, None]
    desc_tr = structural_targets(graph, node_ids, bank.pair_a_idx, bank.pair_b_idx)
    desc_va = structural_targets(graph, node_ids, bank.val_pair_a_idx, bank.val_pair_b_idx)
    content_tr = content_features(f0, f0_index, node_ids, bank.pair_a_idx, bank.pair_b_idx)
    content_va = content_features(f0, f0_index, node_ids, bank.val_pair_a_idx, bank.val_pair_b_idx)
    fit = np.ones(rep_tr.shape[0], dtype=bool)
    fit[rng.choice(rep_tr.shape[0], rep_tr.shape[0] // 5, replace=False)] = False

    probes: dict[str, dict[str, list[float]]] = {}
    inputs: dict[str, tuple[F64, F64, bool]] = {
        "content": (content_tr, content_va, False),
        "content+teacher_logit": (
            np.concatenate([content_tr, logit_tr], axis=1),
            np.concatenate([content_va, logit_va], axis=1),
            False,
        ),
        "descriptors": (desc_tr, desc_va, True),
        "descriptors_linear": (desc_tr, desc_va, False),
        "teacher_logit": (logit_tr, logit_va, True),
    }
    for name, (x_tr, x_va, nonlinear) in inputs.items():
        probes[name] = probe_axes(
            x_tr, white.train, x_va, white.val, fit=fit, nonlinear=nonlinear, seed=seed
        )
    loadings = RidgeProbe(desc_tr[fit], white.train[fit]).weight

    # Student logits exist only on V_val: fit on one half, score the other.
    half = np.ones(white.val.shape[0], dtype=bool)
    half[rng.choice(white.val.shape[0], white.val.shape[0] // 2, replace=False)] = False
    val_half: dict[str, list[float]] = {}
    for name, logit in {"teacher_logit": logit_va[:, 0], **students}.items():
        val_half[name] = probe_axes(
            logit[:, None],
            white.val,
            logit[~half][:, None],
            white.val[~half],
            fit=half,
            nonlinear=True,
            seed=seed,
        )["val"]

    content_val = np.clip(np.array(probes["content"]["val"]), 0.0, None)
    structure_val = np.clip(np.array(probes["descriptors"]["val"]), 0.0, None)
    reachable = content_val * structure_val
    beyond = {
        name: (np.clip(content_val - np.clip(np.array(r2), 0.0, None), 0.0, None) * structure_val)
        for name, r2 in val_half.items()
    }
    noise = fp16_noise_floor(rep_tr)
    return {
        "n_rows": int(rep_tr.shape[0]),
        "n_val_rows": int(white.val.shape[0]),
        "k": k,
        "rep_source": bank.manifest.get("rep_source", "topo"),
        "axis_var_share": white.var_share.tolist(),
        "axis_std": white.axis_std.tolist(),
        "axis_std_over_fp16_noise": (white.axis_std / noise).tolist(),
        "val_shift_whitened": white.val_shift.tolist(),
        "val_std_whitened": white.val_std.tolist(),
        "descriptor_loadings": {name: loadings[i].tolist() for i, name in enumerate(STRUCT_NAMES)},
        "probe_r2": probes,
        "val_half_r2": val_half,
        "reachable_structure_per_axis": reachable.tolist(),
        "reachable_structure_equal_weight": float(reachable.mean()),
        "reachable_structure_variance_weighted": float(
            (reachable * white.var_share).sum() / white.var_share.sum()
        ),
        "beyond_decision_equal_weight": {name: float(v.mean()) for name, v in beyond.items()},
        "beyond_decision_per_axis": {name: v.tolist() for name, v in beyond.items()},
    }


def _parse_pair(spec: str) -> tuple[str, Path]:
    name, _, path = spec.partition("=")
    if not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {spec!r}")
    return name, Path(path)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank", type=_parse_pair, action="append", required=True)
    parser.add_argument("--student", type=_parse_pair, action="append", default=[])
    parser.add_argument("--f0-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    graph = truth_graph_for_kd(assemble_data(cfg, verify=True).val_split)
    cached = cast(
        dict[str, object], torch.load(args.f0_cache, map_location="cpu", weights_only=True)
    )
    f0 = cast(torch.Tensor, cached["matrix"]).double().numpy()
    f0_index = {node: i for i, node in enumerate(cast(list[str], cached["node_ids"]))}

    report: dict[str, object] = {}
    for name, path in cast(list[tuple[str, Path]], args.bank):
        logger.info("auditing bank %s at %s", name, path)
        bank = load_kd_targets(path)
        students = {
            student: align_student_logits(bank, scores)
            for student, scores in cast(list[tuple[str, Path]], args.student)
        }
        report[name] = audit_bank(bank, graph, f0, f0_index, students, k=args.k, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("wrote %s\n%s", args.output, json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

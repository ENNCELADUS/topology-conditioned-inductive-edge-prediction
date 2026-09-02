r"""Audit KD row-target banks as representation-distillation targets.

For each bank the audit answers three questions over the official rows:
how many axes the teacher vector really has (centered variance spectrum and
its alignment with the teacher logit); whether it encodes oracle-graph
structure a linear probe can read out, against a content-only control; and
how much of it the student's own input ``(x_u, x_v)`` can predict at all,
which bounds what any representation KD can transfer.

    python -m src.experiments.kd_rep_audit --config configs/b0_v31_breadth_first.yaml \\
        --bank topo=outputs/distill/kd_row_targets_breadth_first \\
        --bank fused=outputs/distill/kd_row_targets_fused_breadth_first \\
        --f0-cache outputs/f0_cache/f0_matrix.pt --output outputs/distill/kd_rep_audit.json

Structural probe targets use the same training-side truth graph the teacher
saw, with the queried partner dropped from both endpoints' neighbour sets.
Probes are ridge regressions fit on 80% of the training block and scored
(R^2) on the held-out 20% and on the V_val block; the content control is
``[f_u + f_v, |f_u - f_v|]`` on F0.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import torch
from numpy.typing import NDArray

from src.distill.artifacts import KDRowTargets, load_kd_targets
from src.distill.teacher_targets import truth_graph_for_kd
from src.train_b0 import assemble_data, load_config

logger = logging.getLogger(__name__)

F64 = NDArray[np.float64]
_STRUCT_NAMES = ("log1p_cn", "log1p_deg_sum", "log1p_deg_absdiff", "jaccard", "log1p_adamic_adar")
_TOP_K = (1, 2, 4, 8, 32)
_GRAM_ROWS = 4096
_PROBE_PCS = 8


def structural_targets(
    graph: nx.Graph, node_ids: list[str], a_idx: NDArray[np.int32], b_idx: NDArray[np.int32]
) -> F64:
    """Per-row ``(n, 5)`` structural descriptors with the queried partner masked out."""
    neigh = {node: set(graph.neighbors(node)) for node in node_ids}
    degree = {node: len(members) for node, members in neigh.items()}
    out = np.zeros((len(a_idx), len(_STRUCT_NAMES)), dtype=np.float64)
    for row, (a, b) in enumerate(zip(a_idx.tolist(), b_idx.tolist(), strict=True)):
        u, v = node_ids[a], node_ids[b]
        n_u = neigh[u] - {v}
        n_v = neigh[v] - {u}
        common = n_u & n_v
        union = len(n_u | n_v)
        aa = sum(1.0 / np.log1p(degree[w]) for w in common if degree[w] > 0)
        out[row] = (
            np.log1p(len(common)),
            np.log1p(len(n_u)) + np.log1p(len(n_v)),
            abs(np.log1p(len(n_u)) - np.log1p(len(n_v))),
            len(common) / union if union else 0.0,
            np.log1p(aa),
        )
    return out


def content_features(
    f0: F64,
    f0_index: dict[str, int],
    node_ids: list[str],
    a_idx: NDArray[np.int32],
    b_idx: NDArray[np.int32],
) -> F64:
    """Symmetric endpoint-only pair features ``[f_u + f_v, |f_u - f_v|]``."""
    rows = np.array([f0_index[node] for node in node_ids], dtype=np.int64)
    f_a = f0[rows[a_idx]]
    f_b = f0[rows[b_idx]]
    return np.concatenate([f_a + f_b, np.abs(f_a - f_b)], axis=1)


class RidgeProbe:
    """Closed-form multi-output ridge on standardized inputs."""

    def __init__(self, x_train: F64, y_train: F64, *, shrink: float = 1e-3) -> None:
        self.mean = x_train.mean(axis=0)
        self.std = x_train.std(axis=0) + 1e-8
        x = (x_train - self.mean) / self.std
        self.y_mean = y_train.mean(axis=0)
        gram = x.T @ x
        lam = shrink * np.trace(gram) / gram.shape[0]
        self.weight = np.linalg.solve(
            gram + lam * np.eye(gram.shape[0]), x.T @ (y_train - self.y_mean)
        )

    def predict(self, x: F64) -> F64:
        """Predict targets for standardized-on-the-fly inputs."""
        return np.asarray(
            ((x - self.mean) / self.std) @ self.weight + self.y_mean, dtype=np.float64
        )


def r2_columns(y_true: F64, y_pred: F64) -> F64:
    """Per-column R^2 (1 - residual / centered variance)."""
    resid = ((y_true - y_pred) ** 2).sum(axis=0)
    total = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return np.asarray(1.0 - resid / np.maximum(total, 1e-12), dtype=np.float64)


def spectrum_report(
    rep: F64, logit: F64, label: F64, rng: np.random.Generator
) -> dict[str, object]:
    """Variance spectrum, logit alignment, and cosine-Gram structure of one block."""
    sub = rep[rng.choice(rep.shape[0], min(rep.shape[0], 20000), replace=False)]
    mu = sub.mean(axis=0)
    _, sing, vt = np.linalg.svd(sub - mu, full_matrices=False)
    var = sing**2
    cum = np.cumsum(var) / var.sum()
    proj = (rep - mu) @ vt[0]
    idx = rng.choice(rep.shape[0], min(rep.shape[0], _GRAM_ROWS), replace=False)
    x = rep[idx]
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
    off = ~np.eye(len(idx), dtype=bool)
    cos = (x_norm @ x_norm.T)[off]
    prob = 1.0 / (1.0 + np.exp(-logit[idx]))
    delta_p = np.abs(prob[:, None] - prob[None, :])[off]
    y = label[idx]
    same = (y[:, None] == y[None, :])[off]
    block = np.where(same, cos[same].mean(), cos[~same].mean())
    return {
        "centered_var_frac_topk": {str(k): float(cum[k - 1]) for k in _TOP_K},
        "participation_ratio": float(var.sum() ** 2 / (var**2).sum()),
        "corr_top1_logit": float(np.corrcoef(proj, logit)[0, 1]),
        "corr_top1_label": float(np.corrcoef(proj, label)[0, 1]),
        "corr_norm_logit": float(np.corrcoef(np.linalg.norm(rep, axis=1), logit)[0, 1]),
        "corr_cosgram_absdp": float(np.corrcoef(cos, delta_p)[0, 1]),
        "gram_loss_constant_predictor": float(cos.var()),
        "gram_loss_label_block_predictor": float(((cos - block) ** 2).mean()),
    }


def audit_bank(
    bank: KDRowTargets,
    graph: nx.Graph,
    f0: F64,
    f0_index: dict[str, int],
    rng: np.random.Generator,
) -> dict[str, object]:
    """Spectrum, structural-probe, and content-predictability report for one bank."""
    node_ids = list(bank.node_ids)
    rep_tr = bank.teacher_rep.astype(np.float64)
    rep_va = bank.val_teacher_rep.astype(np.float64)
    logit_tr = bank.teacher_logit.astype(np.float64)
    logit_va = bank.val_teacher_logit.astype(np.float64)
    struct_tr = structural_targets(graph, node_ids, bank.pair_a_idx, bank.pair_b_idx)
    struct_va = structural_targets(graph, node_ids, bank.val_pair_a_idx, bank.val_pair_b_idx)
    content_tr = content_features(f0, f0_index, node_ids, bank.pair_a_idx, bank.pair_b_idx)
    content_va = content_features(f0, f0_index, node_ids, bank.val_pair_a_idx, bank.val_pair_b_idx)
    targets_tr = np.concatenate([struct_tr, logit_tr[:, None]], axis=1)
    targets_va = np.concatenate([struct_va, logit_va[:, None]], axis=1)
    target_names = (*_STRUCT_NAMES, "teacher_logit")

    # V_val rows see only cross-boundary structure, so a random 20% slice of the
    # training block is the in-distribution held-out set; V_val is the shifted one.
    holdout = np.zeros(rep_tr.shape[0], dtype=bool)
    holdout[rng.choice(rep_tr.shape[0], rep_tr.shape[0] // 5, replace=False)] = True
    fit = ~holdout

    probes: dict[str, dict[str, dict[str, float]]] = {"train_holdout": {}, "val": {}}
    inputs = {
        "rep": (rep_tr, rep_va),
        "content": (content_tr, content_va),
        "content+logit": (
            np.concatenate([content_tr, logit_tr[:, None]], axis=1),
            np.concatenate([content_va, logit_va[:, None]], axis=1),
        ),
    }
    for name, (x_tr, x_va) in inputs.items():
        probe = RidgeProbe(x_tr[fit], targets_tr[fit])
        for split, x_eval, y_eval in (
            ("train_holdout", x_tr[holdout], targets_tr[holdout]),
            ("val", x_va, targets_va),
        ):
            r2 = r2_columns(y_eval, probe.predict(x_eval))
            probes[split][name] = dict(zip(target_names, r2.tolist(), strict=True))

    # Content -> rep predictability: the transferable fraction of the target.
    mu = rep_tr[fit].mean(axis=0)
    _, _, vt = np.linalg.svd(
        rep_tr[fit][rng.choice(int(fit.sum()), 20000, replace=False)] - mu, full_matrices=False
    )
    rep_probe = RidgeProbe(content_tr[fit], rep_tr[fit])
    pc_probe = RidgeProbe(content_tr[fit], (rep_tr[fit] - mu) @ vt[:_PROBE_PCS].T)
    content_to_rep: dict[str, object] = {}
    for split, x_eval, rep_eval in (
        ("train_holdout", content_tr[holdout], rep_tr[holdout]),
        ("val", content_va, rep_va),
    ):
        resid = ((rep_eval - rep_probe.predict(x_eval)) ** 2).sum()
        total = ((rep_eval - rep_eval.mean(axis=0)) ** 2).sum()
        pc_r2 = r2_columns((rep_eval - mu) @ vt[:_PROBE_PCS].T, pc_probe.predict(x_eval))
        content_to_rep[split] = {
            "overall": float(1.0 - resid / total),
            "top_pcs": [float(v) for v in pc_r2],
        }

    return {
        "n_rows": int(rep_tr.shape[0]),
        "n_val_rows": int(rep_va.shape[0]),
        "rep_dim": int(rep_tr.shape[1]),
        "rep_source": bank.manifest.get("rep_source", "topo"),
        "spectrum_train": spectrum_report(
            rep_tr, logit_tr, bank.pair_label.astype(np.float64), rng
        ),
        "spectrum_val": spectrum_report(
            rep_va, logit_va, bank.val_pair_label.astype(np.float64), rng
        ),
        "probe_r2": probes,
        "content_to_rep_r2": content_to_rep,
    }


def _parse_bank(spec: str) -> tuple[str, Path]:
    name, _, path = spec.partition("=")
    if not name or not path:
        raise argparse.ArgumentTypeError(f"--bank expects NAME=PATH, got {spec!r}")
    return name, Path(path)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank", type=_parse_bank, action="append", required=True)
    parser.add_argument("--f0-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    assembled = assemble_data(cfg, verify=True)
    graph = truth_graph_for_kd(assembled.val_split)
    cached = cast(
        dict[str, object], torch.load(args.f0_cache, map_location="cpu", weights_only=True)
    )
    f0 = cast(torch.Tensor, cached["matrix"]).double().numpy()
    f0_index = {node: i for i, node in enumerate(cast(list[str], cached["node_ids"]))}
    rng = np.random.default_rng(args.seed)

    report: dict[str, object] = {}
    for name, path in cast(list[tuple[str, Path]], args.bank):
        logger.info("auditing bank %s at %s", name, path)
        report[name] = audit_bank(load_kd_targets(path), graph, f0, f0_index, rng)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("wrote %s\n%s", args.output, json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

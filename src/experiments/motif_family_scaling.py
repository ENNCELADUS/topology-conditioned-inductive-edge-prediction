"""The F4 family-presence check: can the reader tell a near-zero family from an empty one?

Plan ``docs/tmp/2026-09-18-motif-stage2-fix-plan.md`` part B, F4. The Stage II
generator's floor is a near-constant graph whose families are present but tiny.
Whether that floor is harmless depends on the *reader*: RRWP normalises by the
weighted degree, so a uniformly shrunk family could survive the normalisation
and keep carrying the token field it should not. This driver measures it
directly on the Stage I bundle, which reads true templates, by scaling one
family's compiled weights by ``s`` and comparing the RRWP stack, the four token
fields and the trunk logits against the empty family ``s = 0``.

Decision rule (F4): a presence gate goes into wave 3 only if the logit at
``s = 0.01`` differs from ``s = 0`` by more than 10% of the ``s = 1`` effect.

The logits come from the formal packed scoring path with an explicit
``row_templates`` bank, so nothing here reimplements the trunk, the AB/BA
symmetrisation or the batching.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score

import src.score_universe as su
from src.data.motif_template import (
    ATTACH_L,
    CLOSURE_U,
    CLOSURE_V,
    INTERIOR,
    N_EDGES,
    MotifTemplateTable,
)
from src.distill.motif_losses import TOPO_FIELDS
from src.model.egostitch.classifier.motif_prompt import (
    V3_1MotifPrompt,
    dense_adjacency,
    motif_rrwp,
)

logger = logging.getLogger("motif_family_scaling")

#: The two families of spec section 8, as edge-index slices of the 96-edge template.
FAMILY_SLICES: dict[str, slice] = {
    "closure": slice(CLOSURE_U.start, CLOSURE_V.stop),
    "bridge": slice(ATTACH_L.start, INTERIOR.stop),
}
#: The F4 threshold: the near-zero effect must exceed this fraction of the full effect.
NEAR_ZERO_FRACTION = 0.1
#: The ``s`` at which "near zero" is read, and the reference "empty" and "full" scales.
NEAR_ZERO_SCALE = 0.01
EMPTY_SCALE = 0.0
FULL_SCALE = 1.0


def scale_family(weights: torch.Tensor, family: str, scale: float) -> torch.Tensor:
    """Return a copy of ``weights`` with one family's edges multiplied by ``scale``.

    Args:
        weights: ``(B, 96)`` edge weights against ``EDGE_ENDPOINTS``.
        family: ``closure`` (the 16 closure edges) or ``bridge`` (attachments
            and interior).
        scale: The multiplier; ``0`` empties the family, ``1`` leaves it alone.

    Returns:
        A new ``(B, 96)`` float32 tensor.

    Raises:
        ValueError: On an unknown family or a non-``(B, 96)`` table.
    """
    if family not in FAMILY_SLICES:
        raise ValueError(f"family must be one of {sorted(FAMILY_SLICES)}, got {family!r}")
    if weights.dim() != 2 or weights.size(-1) != N_EDGES:
        raise ValueError(f"weights must have shape (B, {N_EDGES}), got {tuple(weights.shape)}")
    out = weights.detach().float().clone()
    span = FAMILY_SLICES[family]
    out[:, span] = out[:, span] * float(scale)
    return out


def distinguishes_near_zero_from_empty(mean_logit_by_scale: Mapping[float, float]) -> bool:
    """Apply the F4 decision rule to the per-scale mean logits.

    Args:
        mean_logit_by_scale: Mean trunk logit per scale ``s``; must contain
            `EMPTY_SCALE`, `NEAR_ZERO_SCALE` and `FULL_SCALE`.

    Returns:
        Whether the near-zero family moves the logit by more than
        `NEAR_ZERO_FRACTION` of the full family's effect.

    Raises:
        KeyError: If one of the three required scales was not measured.
    """
    empty = mean_logit_by_scale[EMPTY_SCALE]
    near = mean_logit_by_scale[NEAR_ZERO_SCALE]
    full = mean_logit_by_scale[FULL_SCALE]
    return abs(near - empty) > NEAR_ZERO_FRACTION * abs(full - empty)


def _read_fields(
    model: V3_1MotifPrompt, weights: torch.Tensor, device: torch.device, chunk: int
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the RRWP stack and the four token fields of one weight bank.

    The reader and the count head are fed the graph alone: the residue arguments
    of `V3_1MotifPrompt.tokens_from_weights` are consumed only by the
    ``token_source='direct'`` control, which this check refuses.

    Args:
        model: The Stage I bundle, in ``eval()`` mode.
        weights: ``(n, 96)`` fp32 weight bank on the CPU.
        device: Compute device.
        chunk: Rows per forward.

    Returns:
        The ``(n, 26, 26, k)`` RRWP stack and the four ``(n, width)`` fields, on the CPU.

    Raises:
        ValueError: If the checkpoint does not read its tokens from the graph.
    """
    if model.cfg.token_source != "graph":
        raise ValueError(
            "the family-scaling check reads the graph through the reader; "
            f"this checkpoint has token_source={model.cfg.token_source!r}"
        )
    stacks: list[torch.Tensor] = []
    fields: dict[str, list[torch.Tensor]] = {name: [] for name in TOPO_FIELDS}
    for start in range(0, weights.size(0), chunk):
        batch = weights[start : start + chunk].to(device=device, dtype=torch.float32)
        rows = batch.size(0)
        empty = batch.new_zeros((rows, 1, model.d_model))
        lengths = torch.ones(rows, dtype=torch.int64, device=device)
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            stacks.append(motif_rrwp(dense_adjacency(batch), model.cfg.reader.rrwp_k).cpu())
            tokens = model.tokens_from_weights(batch, empty, empty, lengths, lengths)
        for name in TOPO_FIELDS:
            fields[name].append(tokens[name].float().cpu())
    return (
        torch.cat(stacks),
        {name: torch.cat(parts) for name, parts in fields.items()},
    )


def _metrics(logits: NDArray[np.float32], labels: NDArray[np.int8]) -> dict[str, float | None]:
    """Return AUROC and AUPRC of raw logits, or ``None`` when a class is absent."""
    truth = np.asarray(labels, dtype=np.int64)
    if truth.min() == truth.max():
        return {"auroc": None, "auprc": None}
    scores = np.asarray(logits, dtype=np.float64)
    return {
        "auroc": float(roc_auc_score(truth, scores)),
        "auprc": float(average_precision_score(truth, scores)),
    }


def run_family_scaling(
    *,
    checkpoint: Path,
    pack_dir: Path,
    data_root: Path,
    strategy: str,
    family: str,
    scales: Sequence[float],
    rows: int,
    device: torch.device,
    amp: str,
    token_budget: int,
    seed: int,
    chunk: int = 256,
) -> dict[str, object]:
    """Measure the reader's response to scaling one family on V_val rows.

    Args:
        checkpoint: A Stage I motif-prompt bundle (``best.pt``).
        pack_dir: The packed feature directory the formal scorer reads.
        data_root: Benchmark data root.
        strategy: Split strategy (for example ``breadth_first``).
        family: ``closure`` or ``bridge``.
        scales: The multipliers to sweep; `EMPTY_SCALE` must be one of them.
        rows: Rows drawn from ``val_cls``; ``0`` or more than the universe takes all.
        device: Compute device.
        amp: Encoder autocast mode (``bf16`` or ``off``); the pair pass is fp32.
        token_budget: Scoring token budget.
        seed: Row-sample seed.
        chunk: Rows per reader forward.

    Returns:
        The report payload, one entry per scale plus the F4 decision.

    Raises:
        ValueError: If the checkpoint is not a Stage I motif-prompt bundle, or
            `EMPTY_SCALE` is missing from ``scales``.
    """
    if EMPTY_SCALE not in scales:
        raise ValueError(f"scales must contain the empty reference {EMPTY_SCALE}")
    loaded, family_name, checkpoint_id = su._load_checkpoint(checkpoint)
    if family_name != su.MOTIF_PROMPT_FAMILY:
        raise ValueError(f"{checkpoint}: model_family {family_name!r}, expected a motif prompt")
    model = cast(V3_1MotifPrompt, loaded)
    if model.cfg.stage != "one":
        raise ValueError(f"{checkpoint}: the family-scaling check reads a Stage I bundle")
    model = model.to(device).eval()

    pairs, labels = su._resolve_pairs("val_cls", data_root, strategy)
    if 0 < rows < len(pairs):
        generator = np.random.default_rng(seed)
        keep = np.sort(generator.choice(len(pairs), size=rows, replace=False))
        pairs = [pairs[int(i)] for i in keep]
        labels = labels[keep]
    logger.info("scoring %d val_cls rows on family %s", len(pairs), family)

    table = MotifTemplateTable(su._oracle_truth_graph_for_scoring("val_cls", data_root, strategy))
    true_weights = torch.from_numpy(table.weights(pairs))

    reference_stack: torch.Tensor | None = None
    reference_fields: dict[str, torch.Tensor] | None = None
    reference_logits: NDArray[np.float32] | None = None
    entries: dict[str, object] = {}
    mean_logits: dict[float, float] = {}
    # The empty family is the reference every other scale is read against, so
    # it is measured first whatever order the caller listed the scales in.
    ordered = sorted(dict.fromkeys(scales), key=lambda value: value != EMPTY_SCALE)
    for scale in ordered:
        weights = scale_family(true_weights, family, scale)
        stack, fields = _read_fields(model, weights, device, chunk)
        logits = su._score_v3_1_packed(
            model,
            pairs,
            pack_dir,
            device=device,
            amp=amp,
            token_budget=token_budget,
            row_templates=weights,
        )
        if scale == EMPTY_SCALE:
            reference_stack, reference_fields, reference_logits = stack, fields, logits
        assert reference_stack is not None
        assert reference_fields is not None
        assert reference_logits is not None
        delta = logits.astype(np.float64) - reference_logits.astype(np.float64)
        mean_logits[float(scale)] = float(np.mean(logits.astype(np.float64)))
        entries[f"{scale:g}"] = {
            "scale": float(scale),
            "mean_abs_rrwp_delta_vs_empty": float((stack - reference_stack).abs().mean()),
            "token_change_norm_vs_empty": {
                name: float((fields[name] - reference_fields[name]).norm(dim=-1).mean())
                for name in TOPO_FIELDS
            },
            "mean_logit": mean_logits[float(scale)],
            "logit_change_vs_empty": {
                "mean": float(np.mean(delta)),
                "std": float(np.std(delta)),
            },
            **_metrics(logits, labels),
        }
    decision = distinguishes_near_zero_from_empty(mean_logits)
    return {
        "check": "motif_family_scaling_f4",
        "family": family,
        "rows": len(pairs),
        "pairs_source": "val_cls",
        "strategy": strategy,
        "checkpoint": str(checkpoint),
        "checkpoint_id": checkpoint_id,
        "truth": "true templates compiled on the V_val truth graph (labelled diagnostic)",
        "scales": [float(scale) for scale in scales],
        "by_scale": entries,
        "decision": {
            "rule": (
                f"|mean logit(s={NEAR_ZERO_SCALE}) - mean logit(s={EMPTY_SCALE})| > "
                f"{NEAR_ZERO_FRACTION} * |mean logit(s={FULL_SCALE}) - mean logit(s={EMPTY_SCALE})|"
            ),
            "near_zero_effect": abs(mean_logits[NEAR_ZERO_SCALE] - mean_logits[EMPTY_SCALE]),
            "full_effect": abs(mean_logits[FULL_SCALE] - mean_logits[EMPTY_SCALE]),
            "distinguishes_near_zero_from_empty": decision,
            "presence_gate_recommended": decision,
        },
    }


def markdown_table(report: Mapping[str, object]) -> str:
    """Render the per-scale table plus the decision line.

    Args:
        report: A `run_family_scaling` payload.

    Returns:
        The markdown document.
    """
    by_scale = cast(Mapping[str, Mapping[str, object]], report["by_scale"])
    lines = [
        f"# Motif family-scaling check (F4) -- {report['family']}",
        "",
        f"Rows: {report['rows']} of `val_cls`; checkpoint `{report['checkpoint_id']}`.",
        "",
        "| s | mean abs RRWP delta | token delta u/v/rel/cnt | mean logit | logit delta mean+-std |"
        " AUROC | AUPRC |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, entry in by_scale.items():
        tokens = cast(Mapping[str, float], entry["token_change_norm_vs_empty"])
        change = cast(Mapping[str, float], entry["logit_change_vs_empty"])
        auroc = entry["auroc"]
        auprc = entry["auprc"]
        lines.append(
            f"| {key} | {entry['mean_abs_rrwp_delta_vs_empty']:.6f} | "
            + "/".join(f"{tokens[name]:.3f}" for name in TOPO_FIELDS)
            + f" | {entry['mean_logit']:.4f} | {change['mean']:+.4f}+-{change['std']:.4f} | "
            + (f"{auroc:.4f}" if isinstance(auroc, float) else "n/a")
            + " | "
            + (f"{auprc:.4f}" if isinstance(auprc, float) else "n/a")
            + " |"
        )
    decision = cast(Mapping[str, object], report["decision"])
    lines += [
        "",
        f"Rule: `{decision['rule']}`.",
        "",
        f"Near-zero effect {cast(float, decision['near_zero_effect']):.4f} vs full effect "
        f"{cast(float, decision['full_effect']):.4f} -> "
        f"`distinguishes_near_zero_from_empty = "
        f"{decision['distinguishes_near_zero_from_empty']}`.",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the command line.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "F4 family-presence check: scale one motif family's true weights on V_val rows "
            "and record the RRWP, token and logit response against the empty family."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Stage I bundle best.pt")
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--strategy", default="breadth_first")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--scales", default="1,0.3,0.1,0.03,0.01,0")
    parser.add_argument("--family", default="closure", choices=sorted(FAMILY_SLICES))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", default="bf16", choices=["off", "bf16"])
    parser.add_argument("--token-budget", type=int, default=131_072)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the check and write its JSON and markdown.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    report = run_family_scaling(
        checkpoint=args.checkpoint,
        pack_dir=args.pack_dir,
        data_root=args.data_root,
        strategy=args.strategy,
        family=args.family,
        scales=[float(value) for value in args.scales.split(",") if value],
        rows=args.rows,
        device=su._resolve_device(args.device),
        amp=args.amp,
        token_budget=args.token_budget,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = f"family_scaling_{args.family}"
    (args.output_dir / f"{name}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / f"{name}.md").write_text(markdown_table(report), encoding="utf-8")
    logger.info("wrote %s", args.output_dir / f"{name}.json")


if __name__ == "__main__":
    main()


__all__ = [
    "EMPTY_SCALE",
    "FAMILY_SLICES",
    "FULL_SCALE",
    "NEAR_ZERO_FRACTION",
    "NEAR_ZERO_SCALE",
    "build_parser",
    "distinguishes_near_zero_from_empty",
    "main",
    "markdown_table",
    "run_family_scaling",
    "scale_family",
]

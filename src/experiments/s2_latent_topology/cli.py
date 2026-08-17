"""S2 CLI: sample the region corpus, train the four arms, and evaluate.

Stages run independently against a shared ``--output-dir``, so a full run can
be split across processes: ``sample`` writes ``corpus.pt``; each ``train-*``
stage loads it and writes its own checkpoint (``ae.pt``/``prior.pt``/
``det.pt``/``marg.pt``) plus appends to a shared ``metrics.jsonl``; ``eval``
loads whichever checkpoints exist and writes ``s2_results.json`` +
``s2_tables.md``. ``--variant act`` (``z_dim=0``, the activity-only ablation)
suffixes every artifact filename with ``_act`` instead, so both variants can
share one ``--output-dir``.

CLI::

    python -m src.experiments.s2_latent_topology \
        --stage all --data-root data --strategy breadth_first \
        --output-dir outputs/s2 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import networkx as nx

from src.data.features import FeatureStore
from src.experiments.g1_hardened_e2 import _BENCHMARK_SUBDIR
from src.experiments.s2_latent_topology.data import (
    RegionCorpus,
    build_region_corpus,
    load_corpus,
    save_corpus,
)
from src.experiments.s2_latent_topology.evaluate import (
    build_eval_context,
    render_tables_markdown,
    run_s2_eval,
)
from src.experiments.s2_latent_topology.train import (
    TrainConfig,
    train_ae,
    train_det,
    train_marg,
    train_prior,
)

logger = logging.getLogger(__name__)

_ALL_STAGES: tuple[str, ...] = (
    "sample",
    "train-ae",
    "train-prior",
    "train-det",
    "train-marg",
    "eval",
)
_FEATURES_SUBDIR = Path("features") / "frozen_node_features_1024"


def _load_train_graph(benchmark_root: Path, strategy: str) -> nx.Graph:
    """Unpickle the strategy's training truth graph.

    Mirrors :func:`src.experiments.g1_hardened_e2.load_test_graph`, reading
    ``<benchmark_root>/<strategy>/train_graph.pkl`` directly.

    Args:
        benchmark_root: Benchmark package root (e.g. ``data/benchmark_2025_neurips``).
        strategy: Split strategy name.

    Returns:
        The pickled ``networkx.Graph`` (self-loops not stripped).

    Raises:
        TypeError: If the unpickled object is not a ``networkx.Graph``.
    """
    path = benchmark_root / strategy / "train_graph.pkl"
    with path.open("rb") as f:
        graph = pickle.load(f)
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"{path}: expected a pickled networkx.Graph, got {type(graph)!r}")
    return graph


def _load_train_corpus(output_dir: Path, parser: argparse.ArgumentParser) -> RegionCorpus:
    """Load ``output_dir/corpus.pt``, or report it clearly as a missing prerequisite.

    Args:
        output_dir: Directory a prior ``--stage sample`` run wrote ``corpus.pt`` to.
        parser: Parser used to report the missing-prerequisite error.

    Returns:
        The loaded `RegionCorpus`.

    Raises:
        SystemExit: Via ``parser.error`` if ``corpus.pt`` is absent.
    """
    corpus_path = output_dir / "corpus.pt"
    if not corpus_path.exists():
        parser.error(f"corpus not found: {corpus_path} (run --stage sample first)")
    return load_corpus(corpus_path)


def _train_config(args: argparse.Namespace, *, epochs: int, lr: float) -> TrainConfig:
    """Build one `TrainConfig` from the CLI flags shared by every training stage.

    Args:
        args: Parsed CLI arguments.
        epochs: This stage's epoch count (its own ``--<stage>-epochs`` flag).
        lr: This stage's learning rate (its own ``--<stage>-lr`` flag).

    Returns:
        The `TrainConfig`; ``z_dim=0`` when ``--variant act``.
    """
    return TrainConfig(
        epochs=epochs,
        batch_sets=args.batch_sets,
        lr=lr,
        seed=args.seed,
        device=args.device,
        z_dim=0 if args.variant == "act" else args.z_dim,
        ae_dim=args.ae_dim,
        ae_layers=args.ae_layers,
        ae_rrwp_k=args.ae_rrwp_k,
        ae_heads=args.ae_heads,
        ae_seeds=args.ae_seeds,
        ae_d_model=args.ae_d_model,
        set_width=args.set_width,
        set_layers=args.set_layers,
        set_heads=args.set_heads,
        set_pma_seeds=args.set_pma_seeds,
        time_dim=args.time_dim,
        marg_hidden=args.marg_hidden,
    )


def _stage_sample(args: argparse.Namespace, *, output_dir: Path, sizes: tuple[int, ...]) -> None:
    """Sample the S2 training region corpus and cache it to ``output_dir/corpus.pt``.

    Args:
        args: Parsed CLI arguments.
        output_dir: Directory checkpoints, caches, and results are written to.
        sizes: Parsed ``--sizes`` region node-count sizes.
    """
    benchmark_root = args.data_root / _BENCHMARK_SUBDIR
    train_graph = _load_train_graph(benchmark_root, args.strategy)
    store = FeatureStore(args.data_root / _FEATURES_SUBDIR)
    corpus = build_region_corpus(
        train_graph,
        store,
        sizes=sizes,
        per_size=args.regions_per_size,
        salt="s2-train-v1|",
        holdout_frac=args.holdout_frac,
        cache_dir=output_dir,
    )
    save_corpus(corpus, output_dir / "corpus.pt")
    logger.info(
        "sampled %d regions (%d dropped for featureless nodes)",
        len(corpus.regions),
        corpus.dropped_featureless_regions,
    )


def _stage_train(
    stage: str,
    args: argparse.Namespace,
    *,
    output_dir: Path,
    suffix: str,
    parser: argparse.ArgumentParser,
) -> None:
    """Train one of the four S2 arms and append to the shared ``metrics<suffix>.jsonl``.

    ``train-prior``/``train-det`` additionally load the frozen ``ae<suffix>.pt``
    written by a prior ``train-ae`` run (`train_prior`/`train_det` raise
    `FileNotFoundError` if it is absent).

    Args:
        stage: One of ``"train-ae"``, ``"train-prior"``, ``"train-det"``, ``"train-marg"``.
        args: Parsed CLI arguments.
        output_dir: Directory checkpoints, caches, and results are written to.
        suffix: ``"_act"`` for the activity-only ablation, else ``""``.
        parser: Parser used to report a missing ``corpus.pt`` prerequisite.
    """
    name = stage.removeprefix("train-")
    corpus = _load_train_corpus(output_dir, parser)
    epochs, lr = getattr(args, f"{name}_epochs"), getattr(args, f"{name}_lr")
    cfg = _train_config(args, epochs=epochs, lr=lr)
    out_path = output_dir / f"{name}{suffix}.pt"
    metrics_path = output_dir / f"metrics{suffix}.jsonl"
    ae_path = output_dir / f"ae{suffix}.pt"
    if name == "ae":
        train_ae(corpus, cfg, out_path=out_path, metrics_path=metrics_path)
    elif name == "prior":
        train_prior(corpus, cfg, ae_path=ae_path, out_path=out_path, metrics_path=metrics_path)
    elif name == "det":
        train_det(corpus, cfg, ae_path=ae_path, out_path=out_path, metrics_path=metrics_path)
    else:
        train_marg(corpus, cfg, out_path=out_path, metrics_path=metrics_path)
    logger.info("trained %s -> %s", name, out_path)


def _stage_eval(
    args: argparse.Namespace, *, output_dir: Path, suffix: str, parser: argparse.ArgumentParser
) -> None:
    """Run the S2 evaluation pipeline and write ``s2_results<suffix>.json`` + tables.

    Args:
        args: Parsed CLI arguments.
        output_dir: Directory checkpoints, caches, and results are written to.
        suffix: ``"_act"`` for the activity-only ablation, else ``""``.
        parser: Parser used to report a missing checkpoint prerequisite.

    Raises:
        SystemExit: Via ``parser.error`` if the ae/prior/det checkpoints are absent.
    """
    ae_path = output_dir / f"ae{suffix}.pt"
    prior_path = output_dir / f"prior{suffix}.pt"
    det_path = output_dir / f"det{suffix}.pt"
    prereqs = ((ae_path, "train-ae"), (prior_path, "train-prior"), (det_path, "train-det"))
    for path, stage_name in prereqs:
        if not path.exists():
            parser.error(f"checkpoint not found: {path} (run --stage {stage_name} first)")

    marg_path = output_dir / f"marg{suffix}.pt"
    marg_arg = marg_path if marg_path.exists() and args.b0_universe is not None else None

    store = FeatureStore(args.data_root / _FEATURES_SUBDIR)
    ctx = build_eval_context(
        data_root=args.data_root,
        strategy=args.strategy,
        store=store,
        cache_dir=output_dir,
        b0_universe=args.b0_universe,
    )
    payload = run_s2_eval(
        ctx=ctx,
        ae_path=ae_path,
        prior_path=prior_path,
        det_path=det_path,
        marg_path=marg_arg,
        k_draws=args.k_draws,
        flow_steps=args.flow_steps,
        seed=args.seed,
        device=args.device,
        full_region=args.full_region,
    )
    meta = cast(dict[str, object], payload["meta"])
    meta["evidence_class"] = "diagnostic"
    meta["formal"] = False
    meta["variant"] = args.variant

    (output_dir / f"s2_results{suffix}.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / f"s2_tables{suffix}.md").write_text(
        render_tables_markdown(payload), encoding="utf-8"
    )
    logger.info("wrote S2 eval results and tables to %s", output_dir)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``s2_latent_topology`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.experiments.s2_latent_topology",
        description="S2 latent-topology diagnostic: sample, train, and evaluate.",
    )
    parser.add_argument("--stage", required=True, choices=(*_ALL_STAGES, "all"))
    parser.add_argument("--variant", choices=("full", "act"), default="full")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--strategy", default="breadth_first")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/s2"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--regions-per-size", type=int, default=5000)
    parser.add_argument("--holdout-frac", type=float, default=0.05)
    parser.add_argument("--sizes", default="20,40,60,80,100,120,140,160,180,200")

    parser.add_argument("--k-draws", type=int, default=32)
    parser.add_argument("--flow-steps", type=int, default=32)
    parser.add_argument("--b0-universe", type=Path, default=None)
    parser.add_argument("--full-region", action="store_true")

    parser.add_argument("--ae-epochs", type=int, default=30)
    parser.add_argument("--prior-epochs", type=int, default=60)
    parser.add_argument("--det-epochs", type=int, default=40)
    parser.add_argument("--marg-epochs", type=int, default=20)
    parser.add_argument("--ae-lr", type=float, default=3e-4)
    parser.add_argument("--prior-lr", type=float, default=2e-4)
    parser.add_argument("--det-lr", type=float, default=3e-4)
    parser.add_argument("--marg-lr", type=float, default=3e-4)
    parser.add_argument("--batch-sets", type=int, default=32)

    parser.add_argument("--z-dim", type=int, default=16)
    parser.add_argument("--ae-dim", type=int, default=96)
    parser.add_argument("--ae-layers", type=int, default=4)
    parser.add_argument("--ae-rrwp-k", type=int, default=8)
    parser.add_argument("--ae-heads", type=int, default=8)
    parser.add_argument("--ae-seeds", type=int, default=4)
    parser.add_argument("--ae-d-model", type=int, default=512)
    parser.add_argument("--set-width", type=int, default=384)
    parser.add_argument("--set-layers", type=int, default=4)
    parser.add_argument("--set-heads", type=int, default=4)
    parser.add_argument("--set-pma-seeds", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--marg-hidden", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the S2 latent-topology diagnostic pipeline.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        0 on success.

    Raises:
        SystemExit: On argument errors or a missing stage prerequisite
            (``parser.error``).
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_act" if args.variant == "act" else ""
    sizes = tuple(int(token) for token in args.sizes.split(","))

    stages = list(_ALL_STAGES) if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "sample":
            _stage_sample(args, output_dir=output_dir, sizes=sizes)
        elif stage == "eval":
            _stage_eval(args, output_dir=output_dir, suffix=suffix, parser=parser)
        else:
            _stage_train(stage, args, output_dir=output_dir, suffix=suffix, parser=parser)

    return 0


__all__ = ["build_parser", "main"]

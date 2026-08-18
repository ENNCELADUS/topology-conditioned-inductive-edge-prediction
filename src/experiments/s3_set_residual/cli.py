r"""S3 CLI: build the cache, train one arm, and (via Task 4) evaluate it.

Three independent stages, dispatched by `--stage`:

- `cache`: samples the V_val-masked training corpus, V_val's own selection
  buckets, the shared frozen-checkpoint base-logit cache, and the V_val
  feature matrix, all under `--output-dir`. **Every experiment output
  directory owns its own cache**: reuse is keyed on the frozen checkpoint's
  `checkpoint_id` plus exact pair coverage (`data._ensure_base_logit_cache`),
  never on feature-store identity or the output path alone. Pointing
  `--output-dir` at a stale directory from a different split or checkpoint is
  the operator's responsibility -- there is no hash-pinning or artifact
  verifier here (repo policy: no formalism gates); a mismatched checkpoint
  simply triggers a fresh, correct rescoring pass.
- `train`: loads a `--cache-dir` built by `cache` and trains one
  `--arm {res,pair,diag}` into `--run-dir` (`train.train_arm`).
- `eval`: a thin delegation to Task 4's `evaluate.register_eval_args`/`run_eval`,
  imported lazily so `cache`/`train` never require `evaluate.py` to exist.

CLI::

    python -m src.experiments.s3_set_residual \\
        --stage cache --data-root data --strategy breadth_first \\
        --checkpoint outputs/b0/best.pt --output-dir outputs/s3/cache

    python -m src.experiments.s3_set_residual \\
        --stage train --arm res --seed 0 \\
        --cache-dir outputs/s3/cache --run-dir outputs/s3/res_seed0
"""

from __future__ import annotations

import argparse
import logging
import pickle
from collections.abc import Sequence
from pathlib import Path

import networkx as nx
import torch

from src.data.features import FeatureStore
from src.experiments.g1_hardened_e2 import _BENCHMARK_SUBDIR
from src.experiments.s3_set_residual.data import (
    build_s3_corpus,
    build_vval_eval,
    load_s3_corpus,
    load_vval_eval,
    save_s3_corpus,
    save_vval_eval,
)
from src.experiments.s3_set_residual.model import ResidualConfig
from src.experiments.s3_set_residual.train import (
    TrainConfig,
    build_vval_features,
    load_vval_features,
    save_vval_features,
    train_arm,
)

logger = logging.getLogger(__name__)

_STAGES: tuple[str, ...] = ("cache", "train", "eval")
_FEATURES_SUBDIR = Path("features") / "frozen_node_features_1024"


def _load_train_graph(benchmark_root: Path, strategy: str) -> nx.Graph:
    """Unpickle `<benchmark_root>/<strategy>/train_graph.pkl` (self-loops not stripped)."""
    path = benchmark_root / strategy / "train_graph.pkl"
    with path.open("rb") as f:
        graph = pickle.load(f)
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"{path}: expected a pickled networkx.Graph, got {type(graph)!r}")
    return graph


def _base_parser() -> argparse.ArgumentParser:
    """A `--stage`-only parser used to look up the stage before building the full parser."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", required=True, choices=_STAGES)
    return parser


def _add_cache_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--strategy", default="breadth_first")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/s3/cache"),
        help=(
            "This experiment's own cache directory. Not shared across experiments: "
            "base-logit cache reuse is keyed on checkpoint_id + pair coverage only, "
            "never on this path or the feature store's identity -- matching a stale "
            "directory to a different split/checkpoint is the operator's job."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--regions-per-size", type=int, default=150)
    parser.add_argument("--vval-per-size", type=int, default=20)
    parser.add_argument("--neg-ratio", type=int, default=5)
    parser.add_argument("--device", default="cuda")


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--arm", required=True, choices=("res", "pair", "diag"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-regions", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--top-k-checkpoints", type=int, default=3)

    parser.add_argument("--d-in", type=int, default=1536)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--sab-layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--pma-seeds", type=int, default=4)
    parser.add_argument("--p-dim", type=int, default=128)
    parser.add_argument("--head-hidden", type=int, default=256)


def build_parser(stage: str) -> argparse.ArgumentParser:
    """Build the full `s3_set_residual` parser for one already-known `stage`.

    `stage="eval"` lazily imports `evaluate.register_eval_args` to add its
    args -- `evaluate.py` is never imported for `stage in ("cache", "train")`.

    Args:
        stage: One of `"cache"`, `"train"`, `"eval"`.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.experiments.s3_set_residual",
        description="S3 set-residual diagnostic: build the cache, train one arm, evaluate.",
    )
    parser.add_argument("--stage", required=True, choices=_STAGES)
    if stage == "cache":
        _add_cache_args(parser)
    elif stage == "train":
        _add_train_args(parser)
    else:
        parser.add_argument("--run-dir", type=Path, required=True)
        from src.experiments.s3_set_residual.evaluate import register_eval_args

        register_eval_args(parser)
    return parser


def _stage_cache(args: argparse.Namespace) -> None:
    """Build `S3Corpus` + `VvalEval` + `VvalFeatures` into `args.output_dir`.

    See the module docstring for this stage's cache-ownership convention.
    """
    benchmark_root = args.data_root / _BENCHMARK_SUBDIR
    train_graph = _load_train_graph(benchmark_root, args.strategy)
    store = FeatureStore(args.data_root / _FEATURES_SUBDIR)
    device = torch.device(args.device)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus = build_s3_corpus(
        train_graph,
        store,
        checkpoint=args.checkpoint,
        per_size=args.regions_per_size,
        neg_ratio=args.neg_ratio,
        cache_dir=output_dir,
        device=device,
    )
    save_s3_corpus(corpus, output_dir / "corpus.pt")
    logger.info("S3 cache: %d training regions sampled", len(corpus.base.regions))

    vval = build_vval_eval(
        train_graph,
        store,
        checkpoint=args.checkpoint,
        per_size=args.vval_per_size,
        cache_dir=output_dir,
        device=device,
        vval_nodes=corpus.vval_nodes,
    )
    save_vval_eval(vval, output_dir / "vval.pkl")

    vval_features = build_vval_features(vval, store, cache_path=output_dir / "f0_vval.pt")
    save_vval_features(vval_features, output_dir / "vval_features.pt")
    logger.info("S3 cache built at %s", output_dir)


def _stage_train(args: argparse.Namespace) -> None:
    """Load `--cache-dir`'s corpus/V_val/features and train `--arm` into `--run-dir`."""
    cache_dir: Path = args.cache_dir
    corpus = load_s3_corpus(cache_dir / "corpus.pt")
    vval = load_vval_eval(cache_dir / "vval.pkl")
    vval_features = load_vval_features(cache_dir / "vval_features.pt")

    model_cfg = ResidualConfig(
        mode=args.arm,
        d_in=args.d_in,
        d_model=args.d_model,
        sab_layers=args.sab_layers,
        heads=args.heads,
        pma_seeds=args.pma_seeds,
        p_dim=args.p_dim,
        head_hidden=args.head_hidden,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_regions=args.batch_regions,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        top_k_checkpoints=args.top_k_checkpoints,
    )
    train_arm(corpus, vval, vval_features, model_cfg, train_cfg, run_dir=args.run_dir)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the S3 set-residual diagnostic pipeline.

    Args:
        argv: Argument list (defaults to `sys.argv[1:]`).

    Returns:
        0 on success.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    raw = list(argv) if argv is not None else None
    stage = _base_parser().parse_known_args(raw)[0].stage
    parser = build_parser(stage)
    args = parser.parse_args(raw)

    if stage == "cache":
        _stage_cache(args)
    elif stage == "train":
        _stage_train(args)
    else:
        from src.experiments.s3_set_residual.evaluate import run_eval

        run_eval(args)

    return 0


__all__ = ["build_parser", "main"]

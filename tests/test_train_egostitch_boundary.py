"""The held-out path boundary must run *before* the first open, on every stage.

Design 2026-07-29 Sec 3.1 / acceptance item 4. The Wave-1/2 form of the guard
sat at the *end* of `_assemble_e2e_data`, so `split.pkl`, `train_edges.txt` and
`val_edges.txt` were already read -- and the F0, feature-statistics and
grounding caches already written -- by the time it raised, and `prepare_pack`,
which opens the same paths one stage earlier, had no boundary at all. Both were
post-mortems, not guards.

These tests are deliberately self-contained (no fixtures imported from the other
EgoStitch test modules): the boundary is the one property that must hold
independently of how the rest of the assembly is exercised.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from src import train_egostitch as te
from src.data import internal_holdout
from src.train_b0 import EvalConfig, ModelConfig

_NODES = [f"n{i}" for i in range(25)]
_N_GROUND = 3
_STRATEGY = "toy"
_INPUT_DIM = 1536


# --------------------------------------------------------------------------- fixtures


def _write_feature_root(tmp_path: Path) -> None:
    """Write a real, disk-backed `FeatureStore` root at `_FEATURES_SUBDIR`."""
    root = tmp_path / "data" / "features" / "frozen_node_features_1024"
    (root / "embeddings").mkdir(parents=True)
    rng = np.random.default_rng(11)
    index: dict[str, str] = {}
    for node in _NODES:
        rel = f"embeddings/{node}.pt"
        tensor = torch.from_numpy(rng.normal(size=(3, _INPUT_DIM)).astype(np.float32))
        torch.save(tensor, root / rel)
        index[node] = rel
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "format": "torch_pt_per_node",
                "input_dim": _INPUT_DIM,
                "max_sequence_length": 1024,
            }
        ),
        encoding="utf-8",
    )


def _write_train_side_root(tmp_path: Path) -> Path:
    """Write the three train-side inputs the two-stage assembly opens."""
    strategy_dir = tmp_path / "data" / te._BENCHMARK_SUBDIR / _STRATEGY
    strategy_dir.mkdir(parents=True, exist_ok=True)
    with (strategy_dir / "split.pkl").open("wb") as handle:
        pickle.dump({"train": _NODES, "test": []}, handle)
    edges = [(_NODES[i], _NODES[(i + 1) % len(_NODES)]) for i in range(len(_NODES))]
    (strategy_dir / "train_edges.txt").write_text(
        "".join(f"{u}\t{v}\t1\n" for u, v in edges), encoding="utf-8"
    )
    (strategy_dir / "val_edges.txt").write_text(
        f"{_NODES[0]}\t{_NODES[7]}\t1\n{_NODES[1]}\t{_NODES[9]}\t0\n", encoding="utf-8"
    )
    return strategy_dir


def _cfg(tmp_path: Path, *, strategy: str = _STRATEGY, training: bool = True) -> te.EgoConfig:
    """Build an `egostitch_e2e` config with every cache under ``tmp_path/caches``.

    Co-locating the caches is what makes "no cache was written" a single
    assertion: the directory must not exist at all after the guard raises.
    """
    caches = tmp_path / "caches"
    return te.EgoConfig(
        model=ModelConfig(
            family=te._EGOSTITCH_E2E_FAMILY, config={"generator": {"n_ground": _N_GROUND}}
        ),
        data=te.EgoDataConfig(
            root=tmp_path / "data",
            strategy=strategy,
            train_positives="e_sup",
            negative_ratio=5,
            partition_seed=0,
            msg_fraction=0.8,
            node_batch=2,
            edge_batch=4,
            f0_cache=caches / "f0_matrix.pt",
            grounding_cache=caches / "grounding.npz",
            expected_missing_features=[],
            pack_dir=tmp_path / "raw-token-pack",
        ),
        optim=te.EgoOptimConfig(
            lr=1e-4,
            weight_decay=0.0,
            epochs=3,
            warmup_steps=0,
            grad_clip=1.0,
        ),
        diagnostics=te.EgoDiagnosticsConfig(
            gradient_probe_interval=1,
            gradient_imbalance_ratio=50.0,
            gradient_imbalance_steps=2,
            probe_s1_abs_mean_max=1.0,
            selection_auprc_tolerance=0.02,
            topk_fraction=0.1,
        ),
        eval=EvalConfig(patience=2, eval_every=1),
        seed=0,
        output_dir=tmp_path / "out",
        mixed_precision="no",
        training=te.EgoStitchTrainingConfig() if training else None,
    )


def _tiny_holdout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the real BFS holdout at a size the 25-node fixture can satisfy."""

    def tiny(
        train_nodes: list[str],
        e_msg: frozenset[tuple[str, str]],
        e_sup: frozenset[tuple[str, str]],
    ) -> internal_holdout.InternalHoldoutPartition:
        return internal_holdout.derive_internal_holdout(train_nodes, e_msg, e_sup, holdout_size=4)

    monkeypatch.setattr(te, "derive_internal_holdout", tiny)


def _alias_train_edges_onto_test_edges(strategy_dir: Path) -> None:
    """Replace `train_edges.txt` with a relative symlink onto `test_edges.txt`.

    A relative target carrying a ``..`` segment is the cheapest way a train-side
    name aliases a held-out file without naming it, and it is what the guard's
    two-sided `Path.resolve` exists to collapse.
    """
    train_edges = strategy_dir / "train_edges.txt"
    held_out = strategy_dir / "test_edges.txt"
    held_out.write_text(train_edges.read_text(encoding="utf-8"), encoding="utf-8")
    train_edges.unlink()
    train_edges.symlink_to(Path("..") / strategy_dir.name / "test_edges.txt")


@contextmanager
def _recorded_opens(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Path]]:
    """Record every `Path.open` for the duration of the block.

    Every benchmark read in `src/train_egostitch.py` and in
    `src/data/artifacts.py` (`_load_pickle`, `_iter_pair_rows`,
    `_read_labeled_pairs`, `_iter_pair_label_rows`) goes through `Path.open`, so
    one patch covers both the assembly and the full-benchmark sibling.
    """
    opened: list[Path] = []
    real_open = cast(Callable[..., object], Path.open)

    def spy(self: Path, *args: object, **kwargs: object) -> object:
        opened.append(Path(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    try:
        yield opened
    finally:
        monkeypatch.undo()


def _benchmark_reads(opened: list[Path], tmp_path: Path) -> list[Path]:
    benchmark_root = (tmp_path / "data" / te._BENCHMARK_SUBDIR).resolve()
    return [path for path in opened if benchmark_root in path.resolve().parents]


_ALL_RUN_KINDS: list[te.E2ERunKind | None] = ["formal", "debug", None]


# --------------------------------------------------------------------------- tests


class TestBoundaryPrecedesTheFirstOpen:
    """The guard must fire with the filesystem still untouched, from every entry point."""

    @pytest.mark.parametrize("run_kind", _ALL_RUN_KINDS)
    def test_assembly_refuses_in_every_run_kind_without_opening_or_caching(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        run_kind: te.E2ERunKind | None,
    ) -> None:
        """`assemble_egostitch_data`: no run kind is exempt, and nothing is read first.

        The pre-cleanup form exempted `formal` -- the one run that matters -- so
        the parametrization is the regression test for that exemption, and the
        recorded-open assertion is the regression test for the guard's
        placement.
        """
        strategy_dir = _write_train_side_root(tmp_path)
        _write_feature_root(tmp_path)
        _tiny_holdout(monkeypatch)
        _alias_train_edges_onto_test_edges(strategy_dir)
        cfg = replace(_cfg(tmp_path), run_kind=run_kind)

        with (
            _recorded_opens(monkeypatch) as opened,
            pytest.raises(RuntimeError, match="opened a held-out path"),
        ):
            te.assemble_egostitch_data(cfg)

        assert _benchmark_reads(opened, tmp_path) == []
        assert not (tmp_path / "caches").exists()
        assert not cfg.output_dir.exists()

    @pytest.mark.parametrize("run_kind", _ALL_RUN_KINDS)
    def test_pack_refuses_in_every_run_kind_without_opening_or_caching(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        run_kind: te.E2ERunKind | None,
    ) -> None:
        """`prepare_pack` had no boundary at all: it consumed held-out rows one stage early."""
        strategy_dir = _write_train_side_root(tmp_path)
        _write_feature_root(tmp_path)
        _tiny_holdout(monkeypatch)
        _alias_train_edges_onto_test_edges(strategy_dir)
        cfg = replace(_cfg(tmp_path), run_kind=run_kind)
        pack_dir = tmp_path / "pack"

        with (
            _recorded_opens(monkeypatch) as opened,
            pytest.raises(RuntimeError, match="opened a held-out path"),
        ):
            te.prepare_pack(cfg, pack_dir, cold_cache=True)

        assert _benchmark_reads(opened, tmp_path) == []
        assert not pack_dir.exists()
        assert cfg.data.pack_dir is not None
        assert not cfg.data.pack_dir.exists()

    def test_a_symlinked_strategy_directory_does_not_defeat_the_guard(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both sides are resolved, so the alias and the forbidden set stay in one namespace.

        Reaching the same directory under a second name must not put
        `strategy_dir / "test_edges.txt"` and the resolved read in different
        namespaces -- that is the one way a one-sided `resolve` would fail open.
        """
        strategy_dir = _write_train_side_root(tmp_path)
        _write_feature_root(tmp_path)
        _tiny_holdout(monkeypatch)
        _alias_train_edges_onto_test_edges(strategy_dir)
        alias = strategy_dir.parent / "alias"
        alias.symlink_to(Path(_STRATEGY), target_is_directory=True)
        cfg = replace(_cfg(tmp_path, strategy="alias"), run_kind="formal")

        with pytest.raises(RuntimeError, match="opened a held-out path"):
            te.assemble_egostitch_data(cfg)

    def test_a_relative_strategy_alias_does_not_defeat_the_guard(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`toy/../toy` names the same directory; `Path.resolve` must collapse it."""
        strategy_dir = _write_train_side_root(tmp_path)
        _write_feature_root(tmp_path)
        _tiny_holdout(monkeypatch)
        _alias_train_edges_onto_test_edges(strategy_dir)
        cfg = replace(_cfg(tmp_path, strategy=f"{_STRATEGY}/../{_STRATEGY}"), run_kind="formal")

        with pytest.raises(RuntimeError, match="opened a held-out path"):
            te.assemble_egostitch_data(cfg)

    def test_a_hard_link_does_not_defeat_the_guard(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hard link is a second name for one inode, with no link for `resolve` to follow.

        `Path.resolve` returns the train-side name unchanged, so the resolved
        intersection alone reports no trespass; only the `(st_dev, st_ino)`
        comparison sees that the two names are the same file.
        """
        strategy_dir = _write_train_side_root(tmp_path)
        _write_feature_root(tmp_path)
        _tiny_holdout(monkeypatch)
        train_edges = strategy_dir / "train_edges.txt"
        held_out = strategy_dir / "test_edges.txt"
        held_out.write_text(train_edges.read_text(encoding="utf-8"), encoding="utf-8")
        train_edges.unlink()
        train_edges.hardlink_to(held_out)
        assert train_edges.resolve() != held_out.resolve()  # resolution alone sees nothing
        cfg = replace(_cfg(tmp_path), run_kind="formal")

        with pytest.raises(RuntimeError, match="opened a held-out path"):
            te.assemble_egostitch_data(cfg)


class TestBoundaryIsPathScopedNotPresenceScoped:
    """Fail-closed, not closed: the repository data root carries all three held-out files."""

    def test_present_but_unopened_held_out_files_do_not_block_the_pack(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Moving the guard earlier must not turn `prepare_pack` into a presence check.

        The pre-cleanup guard raised on mere presence, which made every non-
        `formal` run impossible in the repository data root. `prepare_pack` is
        the stage that would newly inherit that failure, so it is the one
        asserted here.
        """
        strategy_dir = _write_train_side_root(tmp_path)
        _write_feature_root(tmp_path)
        _tiny_holdout(monkeypatch)
        for name in te._HELD_OUT_FILENAMES:
            (strategy_dir / name).write_text("sealed\n", encoding="utf-8")
        cfg = replace(_cfg(tmp_path), run_kind="formal")
        pack_dir = tmp_path / "pack"

        payload = te.prepare_pack(cfg, pack_dir, cold_cache=True)

        manifest = payload["pack_manifest"]
        assert isinstance(manifest, dict)
        assert manifest["validation_role"] == te._E2E_VALIDATION_ROLE
        assert (pack_dir / te._PACK_FEATURE_STATS_FILENAME).exists()

    def test_present_but_unopened_held_out_files_do_not_block_the_assembly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same for the assembly, in the run kind the old guard exempted."""
        strategy_dir = _write_train_side_root(tmp_path)
        _write_feature_root(tmp_path)
        _tiny_holdout(monkeypatch)
        for name in te._HELD_OUT_FILENAMES:
            (strategy_dir / name).write_text("sealed\n", encoding="utf-8")
        cfg = replace(_cfg(tmp_path), run_kind="formal")

        data = te.assemble_egostitch_data(cfg)

        audit = data.access_audit or {}
        assert audit["forbidden_files_absent"] == dict.fromkeys(te._HELD_OUT_FILENAMES, False)


class TestBoundaryEnumerationCoversEveryRead:
    """The enumeration is only as good as its completeness against the actual reads."""

    def test_train_side_inputs_match_the_paths_the_assembly_opens(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every benchmark path a successful assembly opens must be in the checked set.

        A guard that runs before the reads is only sound if it names *all* of
        them; an omitted path is a hole that no aliasing test would find.
        """
        _write_train_side_root(tmp_path)
        _write_feature_root(tmp_path)
        _tiny_holdout(monkeypatch)
        cfg = replace(_cfg(tmp_path), run_kind="formal")

        with _recorded_opens(monkeypatch) as opened:
            te.assemble_egostitch_data(cfg)

        names = {path.name for path in _benchmark_reads(opened, tmp_path)}
        assert names
        assert names <= set(te._E2E_TRAIN_SIDE_INPUTS)

    def test_train_side_inputs_match_the_paths_the_pack_opens(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_train_side_root(tmp_path)
        _write_feature_root(tmp_path)
        _tiny_holdout(monkeypatch)
        cfg = replace(_cfg(tmp_path), run_kind="formal")

        with _recorded_opens(monkeypatch) as opened:
            te.prepare_pack(cfg, tmp_path / "pack", cold_cache=True)

        names = {path.name for path in _benchmark_reads(opened, tmp_path)}
        assert names
        assert names <= set(te._E2E_TRAIN_SIDE_INPUTS)

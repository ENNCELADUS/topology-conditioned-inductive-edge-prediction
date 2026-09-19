"""Cross-fitted motif fold and cache contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest
import torch
from src.data.motif_crossfit import (
    MotifCrossfitCache,
    canonical_pair,
    eligible_target_fold,
    enumerate_struct_pairs,
    filter_rows_both_endpoints,
    seeded_hash_node_folds,
)
from src.data.motif_template import SWAP_PERM
from src.data.struct_sampler import StructSampler
from src.experiments.motif_crossfit import (
    NoPassingPilotError,
    heldout_presence_report,
    select_pilot_checkpoint,
)
from src.train_b0 import load_config


def test_seeded_hash_folds_are_order_independent_and_cover_nodes() -> None:
    nodes = [f"node_{i:04d}" for i in range(100)]
    forward = seeded_hash_node_folds(nodes, seed=42)
    backward = seeded_hash_node_folds(reversed(nodes), seed=42)
    assert forward == backward
    assert set(forward) == set(nodes)
    assert set(forward.values()) == {0, 1}
    assert forward != seeded_hash_node_folds(nodes, seed=43)


def test_fold_row_filter_requires_both_endpoints_and_missing_nodes_fail_closed() -> None:
    folds = {"a": 0, "b": 0, "c": 1, "d": 1, "e": 1}
    rows = [("a", "b", 1), ("a", "c", 0), ("c", "d", 1), ("a", "missing", 0)]
    assert filter_rows_both_endpoints(rows, folds, 0) == [("a", "b", 1)]
    assert filter_rows_both_endpoints(rows, folds, 1) == [("c", "d", 1)]
    assert eligible_target_fold(("a", "c"), folds) is None


def _cache() -> MotifCrossfitCache:
    folds = {"a": 0, "b": 0, "c": 1, "d": 1, "e": 1}
    first = torch.linspace(0.0, 0.95, 96)
    second = torch.linspace(0.05, 1.0, 96)
    return MotifCrossfitCache.build(
        [("b", "a"), ("c", "d")],
        torch.stack((first, second)),
        fold_by_node=folds,
        source_folds=[1, 0],
        presence=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
    )


def test_cache_canonicalises_and_swaps_requested_orientation() -> None:
    cache = _cache()
    assert cache.pairs == (("a", "b"), ("c", "d"))
    lookup = cache.lookup([("b", "a"), ("a", "b"), ("a", "c")])
    assert lookup.mask.tolist() == [True, True, False]
    # The first input row was stored in b,a orientation; canonicalisation swaps
    # it, then lookup in b,a swaps it back exactly.
    expected = torch.linspace(0.0, 0.95, 96)
    torch.testing.assert_close(lookup.weights[0], expected)
    torch.testing.assert_close(
        lookup.weights[1],
        expected.index_select(0, torch.tensor(SWAP_PERM)),
    )
    assert lookup.source_folds.tolist() == [1, 1, -1]
    assert lookup.presence is not None
    torch.testing.assert_close(lookup.presence[0], torch.tensor([0.1, 0.2, 0.3]))


def test_cache_round_trip_and_required_eligible_coverage(tmp_path: Path) -> None:
    cache = _cache()
    path = tmp_path / "cache.pt"
    cache.save(path)
    loaded = MotifCrossfitCache.load(path)
    assert loaded.pairs == cache.pairs and loaded.seed == 42
    torch.testing.assert_close(loaded.weights, cache.weights)
    folds = {"a": 0, "b": 0, "c": 1, "d": 1, "e": 1}
    loaded.require_pairs([("a", "b"), ("a", "c"), ("c", "d")], folds)
    with pytest.raises(ValueError, match="misses 1 eligible pairs"):
        loaded.require_pairs([("a", "b"), ("c", "d"), ("c", "e")], folds)


def test_cache_rejects_seen_endpoint_provenance_and_forbidden_nodes() -> None:
    folds = {"a": 0, "b": 0}
    with pytest.raises(ValueError, match="saw endpoints"):
        MotifCrossfitCache.build(
            [("a", "b")], torch.zeros(1, 96), fold_by_node=folds, source_folds=[0]
        )
    with pytest.raises(ValueError, match="forbidden"):
        MotifCrossfitCache.build(
            [("a", "b")],
            torch.zeros(1, 96),
            fold_by_node=folds,
            source_folds=[1],
            forbidden_nodes=frozenset({"b"}),
        )


def test_exact_struct_pair_enumeration_matches_sampler_plans() -> None:
    graph = nx.cycle_graph(12)
    graph = nx.relabel_nodes(graph, lambda node: f"n{node}")
    sampler = StructSampler(
        graph,
        nodes=6,
        background_nodes=1,
        mix={"bfs": 0.5, "motif": 0.25, "bridge": 0.25},
        v_val=frozenset(),
        exclude_nodes=frozenset(),
    )
    steps = {1: 3, 2: 2}
    got = enumerate_struct_pairs(sampler, steps, seed=7)
    expected: set[tuple[str, str]] = set()
    for epoch, count in steps.items():
        for subgraph in sampler.plan(seed=7, epoch=epoch, count=count).subgraphs:
            legal = sampler.legal_mask(subgraph)
            for i, left in enumerate(subgraph.nodes):
                for j in range(i + 1, len(subgraph.nodes)):
                    if legal[i, j]:
                        expected.add(canonical_pair(left, subgraph.nodes[j]))
    assert got == sorted(expected)


def _pilot_report(path: Path, *, passed: bool, loss: float, checkpoint: str) -> Path:
    report = {
        "checkpoint": checkpoint,
        "level_1_fit": {"heldout_train": {"L_G": {"generator": loss}}},
        "verdicts": {
            "level_1": {"passed": passed},
            "level_2": {"per_universe": {"heldout_train": passed}},
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_pilot_selection_never_promotes_a_failing_generator(tmp_path: Path) -> None:
    reports = [
        _pilot_report(tmp_path / "epoch1.json", passed=False, loss=0.01, checkpoint="e1.pt"),
        _pilot_report(tmp_path / "epoch2.json", passed=False, loss=0.02, checkpoint="e2.pt"),
    ]
    with pytest.raises(NoPassingPilotError, match="scientific negative") as stopped:
        select_pilot_checkpoint(reports)
    assert stopped.value.reports == tuple(reports)


def test_pilot_selection_uses_lowest_heldout_loss_only_among_passes(tmp_path: Path) -> None:
    failed = _pilot_report(tmp_path / "epoch1.json", passed=False, loss=0.001, checkpoint="e1.pt")
    later = _pilot_report(tmp_path / "epoch3.json", passed=True, loss=0.03, checkpoint="e3.pt")
    earlier = _pilot_report(tmp_path / "epoch2.json", passed=True, loss=0.02, checkpoint="e2.pt")
    assert select_pilot_checkpoint([failed, later, earlier]) == earlier


def test_pilot_selection_breaks_exact_loss_tie_by_report_epoch_order(tmp_path: Path) -> None:
    earlier = _pilot_report(
        tmp_path / "epoch2.json", passed=True, loss=0.02, checkpoint="epoch2.pt"
    )
    later = _pilot_report(tmp_path / "epoch3.json", passed=True, loss=0.02, checkpoint="epoch3.pt")
    assert select_pilot_checkpoint([earlier, later]) == earlier


def test_presence_report_reads_crossfitted_training_truth(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = nx.Graph()
    graph.add_edges_from([("a", "x"), ("x", "b"), ("c", "d")])
    split = SimpleNamespace(build_training_graph=lambda: graph)
    monkeypatch.setattr(
        "src.experiments.motif_crossfit.su._load_val_region_split",
        lambda _root, _strategy: split,
    )
    report = heldout_presence_report(_cache(), data_root=Path("unused"), strategy="bf")
    assert report is not None and report["rows"] == 2
    assert set(report) == {"rows", "closure", "attach", "interior"}


def test_fold_generators_use_the_adopted_full_schedule_warmup_recipe() -> None:
    root = Path(__file__).resolve().parents[1] / "configs" / "split_seed42"
    for head in ("flat", "presence"):
        for fold in (0, 1):
            cfg = load_config(root / f"motif_crossfit_{head}_fold{fold}.yaml")
            block = cfg.model.config["motif_prompt"]
            assert cfg.optim.epochs == 15 and cfg.optim.stop_after_epoch == 8
            assert block["crossfit_fold"] == fold and block["crossfit_seed"] == 42
            assert block["slot_read"] == "residual_block"
            assert block["slot_read_value_norm"] is False
            assert "slot_query_init_std" not in block
            assert block["graph_row_weighting"] == "closure_balanced"
            assert block["graph_row_positive_share"] == 0.5

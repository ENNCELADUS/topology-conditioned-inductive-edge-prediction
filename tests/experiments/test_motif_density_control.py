"""The section 8 output-density control: equal selected count, official self-loops."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest


def test_the_target_is_this_arms_own_selected_pair_count_not_nx_density() -> None:
    from src.experiments.motif_density_control import target_edge_count

    logits = np.asarray([2.0, -1.0, 3.0, 0.5])
    count, n_union = target_edge_count(logits, threshold=0.4)
    assert (count, n_union) == (3, 4)
    # nx.density over the four nodes of those pairs would be 3/6, not 3/4: the
    # union holds only some node pairs, so the rate is selected_pairs / n_union.
    assert count / n_union == 0.75


def test_ties_are_admitted_atomically_so_the_realised_count_can_fall_short() -> None:
    from src.eval.assembly import density_matched_threshold

    probs = np.asarray([0.9, 0.9, 0.9, 0.1])
    threshold = density_matched_threshold(probs, target_edges=2)
    assert int((probs >= threshold).sum()) == 0


def test_self_pairs_assemble_as_self_loops_and_official_gs_keeps_them() -> None:
    from src.eval.assembly import assemble_graph
    from src.eval.graph_metrics import compute_graph_similarity

    pairs = [("a", "b"), ("a", "a")]
    graph = assemble_graph(pairs, np.asarray([0.9, 0.9]), threshold=0.5, nodes=["a", "b"])
    assert graph.number_of_edges() == 2
    reference = nx.Graph()
    reference.add_nodes_from(["a", "b"])
    reference.add_edge("a", "b")
    assert compute_graph_similarity(graph, reference) == pytest.approx(2.0 / 3.0)


def test_a_row_more_than_one_percent_short_is_marked_approximate() -> None:
    from src.experiments.motif_density_control import density_matched_report

    report = density_matched_report(
        rows={"arm": np.asarray([0.9, 0.9, 0.9, 0.1])},
        union_pairs=[("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")],
        target_edges=2,
        g_ref=nx.Graph([("a", "b"), ("b", "c")]),
        buckets={4: [{"a", "b", "c", "d"}]},
        config=None,
    )
    assert report["matching"] == "output_density_matched_union"
    assert report["n_union"] == 4
    rows = report["rows"]
    assert isinstance(rows, dict)
    assert rows["arm"]["target_edges"] == 2
    assert rows["arm"]["realised_edges"] == 0
    assert rows["arm"]["shape_comparison"] == "approximate"


def test_every_row_is_matched_on_its_own_scored_union_not_on_a_transferred_rate() -> None:
    from src.experiments.motif_density_control import density_matched_report

    # Two rows whose logits differ by a constant shift: the pure threshold-placement
    # case this control exists to expose. At a matched count they select the same
    # rows, so any surviving GS difference is shape, not placement.
    base = np.asarray([3.0, 2.0, 1.0, -1.0, -2.0])
    report = density_matched_report(
        rows={"arm": base, "prefix_base": base + 7.5},
        union_pairs=[("a", "b"), ("b", "c"), ("c", "d"), ("a", "d"), ("a", "a")],
        target_edges=3,
        g_ref=nx.Graph([("a", "b"), ("b", "c"), ("c", "d")]),
        buckets={4: [{"a", "b", "c", "d"}]},
        config=None,
    )
    rows = report["rows"]
    assert isinstance(rows, dict)
    for name in ("arm", "prefix_base"):
        assert rows[name]["realised_edges"] == 3
        assert rows[name]["shape_comparison"] == "matched"
    assert rows["arm"]["logit_threshold"] == 1.0
    assert rows["prefix_base"]["logit_threshold"] == 8.5


def _write_run(
    run_dir: Path, logits: list[float], *, checkpoint: str, threshold: float | None
) -> None:
    """A published run directory: the test-topology artifact and, for an arm, its report."""
    from src.score_universe import save_scores

    # The full union over four nodes, self-pairs included: the panel scores every
    # pair inside a bucket's node set.
    nodes = ["a", "b", "c", "d"]
    pairs = [(i, j) for i in range(4) for j in range(i, 4)]
    save_scores(
        run_dir / "scores" / "test_topology.npz",
        node_ids=nodes,
        u_idx=np.asarray([u for u, _ in pairs], dtype=np.int32),
        v_idx=np.asarray([v for _, v in pairs], dtype=np.int32),
        logit=np.asarray(logits, dtype=np.float32),
        label=np.asarray([0, 1, 0, 0, 0, 1, 0, 0, 1, 0], dtype=np.int8),
        row_start=0,
        meta={
            "checkpoint_id": checkpoint,
            "model_family": "v3_1",
            "pairs_source": "test_topology",
            "strategy": "breadth_first",
            "num_rows": len(pairs),
            "created_utc": "2026-09-17T00:00:00+00:00",
            "torch_version": "test",
        },
    )
    if threshold is not None:
        report = {
            "graph": {"fixed_threshold": {"validation_selection": {"logit_threshold": threshold}}}
        }
        (run_dir / "test_report.json").write_text(json.dumps(report), encoding="utf-8")


def test_the_cli_reads_the_arms_own_threshold_and_matches_every_reference_to_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.experiments import motif_density_control as control

    arm, base = tmp_path / "motif_prompt_stage2", tmp_path / "prefix_base"
    arm_logits = [-3.0, 3.0, -1.0, -2.0, -4.0, 2.0, -5.0, -6.0, 1.0, -7.0]
    _write_run(arm, arm_logits, checkpoint="arm", threshold=1.5)
    # The same ranking shifted by a constant: pure threshold placement.
    _write_run(base, [value + 7.5 for value in arm_logits], checkpoint="base", threshold=None)
    graph = nx.Graph([("a", "b"), ("b", "c"), ("c", "d")])
    monkeypatch.setattr(control, "load_test_graph", lambda root, strategy: graph)
    monkeypatch.setattr(
        control,
        "load_test_node_buckets",
        lambda root, strategy: {4: [{"a", "b", "c", "d"}, {"a", "b", "c", "d"}]},
    )
    output = tmp_path / "density_control.json"
    control.main(
        [
            "--arm",
            str(arm),
            "--reference",
            str(base),
            "--output",
            str(output),
            "--data-root",
            str(tmp_path),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    # The arm realises two edges at its own V_val-selected threshold of 1.5.
    assert report["target_edges"] == 2 and report["n_union"] == 10
    rows = report["rows"]
    assert rows["motif_prompt_stage2"]["realised_edges"] == 2
    assert rows["prefix_base"]["realised_edges"] == 2
    assert rows["prefix_base"]["logit_threshold"] == 9.5
    assert "panel" in rows["prefix_base"]
    assert report["provenance"]["motif_prompt_stage2"]["selected_logit_threshold"] == 1.5
    assert report["provenance"]["prefix_base"]["role"] == "reference"


def test_a_reference_scored_on_another_universe_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.experiments import motif_density_control as control
    from src.score_universe import load_scores, save_scores

    arm, other = tmp_path / "arm", tmp_path / "other"
    logits = [-3.0, 3.0, -1.0, -2.0, -4.0, 2.0, -5.0, -6.0, 1.0, -7.0]
    _write_run(arm, logits, checkpoint="arm", threshold=0.0)
    _write_run(other, logits, checkpoint="other", threshold=None)
    artifact = load_scores(other / "scores" / "test_topology.npz")
    save_scores(
        other / "scores" / "test_topology.npz",
        node_ids=artifact.node_ids,
        u_idx=artifact.u_idx[::-1].copy(),
        v_idx=artifact.v_idx[::-1].copy(),
        logit=artifact.logit,
        label=artifact.label,
        row_start=0,
        meta=dict(artifact.meta),
    )
    monkeypatch.setattr(control, "load_test_graph", lambda root, strategy: nx.Graph())
    monkeypatch.setattr(control, "load_test_node_buckets", lambda root, strategy: {})
    with pytest.raises(ValueError, match="different universe"):
        control.density_control_from_runs(
            arm_dir=arm,
            reference_dirs=[other],
            data_root=tmp_path,
            strategy="breadth_first",
            config=None,
        )

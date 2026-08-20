"""Contracts for the Gate A KD runtime (`_build_kd_runtime` / `_CompositeStep._kd_terms`).

Builds real tiny `EgoStitchModel`s (a `full_ego_oracle` teacher and a
`full_ego_features` student sharing one encoder section), round-trips the
teacher through the pinned ``best.pt`` payload keys, and pins the loading,
freezing, warm-start, and loss-plumbing behavior the trainer relies on.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import networkx as nx
import pytest
import torch
from src.model.egostitch.composite import EgoStitchModel
from src.model.egostitch.config import E2EConfig
from src.model.egostitch.generator.full_oracle import FullEgoFeaturesGenerator
from src.train_egostitch import (
    EgoDistillConfig,
    _build_kd_runtime,
    _CompositeStep,
    e2e_global_live_row_mean,
)

pytestmark = pytest.mark.unit


def _tiny_mapping(
    generator_name: str, *, encoder_dim: int = 16, encoder_name: str = "grit_gmt"
) -> dict[str, object]:
    return {
        "generator": {"name": generator_name, "oracle_truth_source": "training_structure"},
        "encoder": {
            "name": encoder_name,
            "dim": encoder_dim,
            "layers": 1,
            "rrwp_k": 3,
            "n_heads": 2,
            "seeds": 2,
            "w_rel": 0.0,
        },
        "classifier": {
            "name": "b0_v31",
            "d_model": 32,
            "encoder_layers": 1,
            "cross_attn_layers": 2,
            "n_heads": 4,
            "n_inj": 1,
            "xattn_heads": 4,
            "p_topo": 0.15,
            "conditioning_mode": "pooled_adapter",
        },
    }


def _teacher_checkpoint(tmp_path: Path, mapping: dict[str, object]) -> tuple[Path, EgoStitchModel]:
    torch.manual_seed(11)
    teacher = EgoStitchModel(E2EConfig.from_mapping(mapping))
    path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state": teacher.state_dict(),
            "model_family": "egostitch_e2e",
            "model_config": mapping,
        },
        path,
    )
    return path, teacher


def test_build_kd_runtime_loads_frozen_teacher_and_warm_starts(tmp_path: Path) -> None:
    checkpoint, teacher = _teacher_checkpoint(tmp_path, _tiny_mapping("full_ego_oracle"))
    torch.manual_seed(23)
    student = EgoStitchModel(E2EConfig.from_mapping(_tiny_mapping("full_ego_features")))
    assert student.encoder is not None and teacher.encoder is not None

    runtime = _build_kd_runtime(
        EgoDistillConfig(teacher_checkpoint=checkpoint, lambda_seed=0.5, lambda_gram=0.25),
        student,
        device=torch.device("cpu"),
    )

    assert runtime.lambda_seed == 0.5 and runtime.lambda_gram == 0.25
    teacher_state = teacher.encoder.state_dict()
    runtime_state = runtime.teacher_encoder.state_dict()
    assert set(runtime_state) == set(teacher_state)
    for key, value in runtime_state.items():
        torch.testing.assert_close(value, teacher_state[key])
    assert all(not p.requires_grad for p in runtime.teacher_encoder.parameters())
    assert not runtime.teacher_encoder.training

    student_state = student.encoder.state_dict()
    warm_keys = [k for k in teacher_state if k.startswith(("project.", "readout."))]
    assert warm_keys
    for key in warm_keys:
        torch.testing.assert_close(student_state[key], teacher_state[key])
    # Widths differ (5 structural channels vs features), so node_embed must
    # never be warm-started.
    assert student_state["node_embed.weight"].shape != teacher_state["node_embed.weight"].shape
    generator = cast(FullEgoFeaturesGenerator, student.generator)
    assert generator._stash_teacher_view is True


def test_build_kd_runtime_rejects_mismatched_encoder_or_teacher_generator(
    tmp_path: Path,
) -> None:
    checkpoint, _ = _teacher_checkpoint(tmp_path, _tiny_mapping("full_ego_oracle"))
    wide_student = EgoStitchModel(
        E2EConfig.from_mapping(_tiny_mapping("full_ego_features", encoder_dim=32))
    )
    with pytest.raises(ValueError, match="must match exactly"):
        _build_kd_runtime(
            EgoDistillConfig(teacher_checkpoint=checkpoint),
            wide_student,
            device=torch.device("cpu"),
        )

    student = EgoStitchModel(E2EConfig.from_mapping(_tiny_mapping("full_ego_features")))
    bad_teacher, _ = _teacher_checkpoint(tmp_path, _tiny_mapping("full_ego_features"))
    with pytest.raises(ValueError, match="must be full_ego_oracle"):
        _build_kd_runtime(
            EgoDistillConfig(teacher_checkpoint=bad_teacher),
            student,
            device=torch.device("cpu"),
        )


def test_build_kd_runtime_requires_grit_gmt_seed_tokens(tmp_path: Path) -> None:
    mapping = _tiny_mapping("full_ego_oracle", encoder_name="ste_typed")
    checkpoint, _ = _teacher_checkpoint(tmp_path, mapping)
    non_grit_student = EgoStitchModel(
        E2EConfig.from_mapping(_tiny_mapping("full_ego_features", encoder_name="ste_typed"))
    )

    with pytest.raises(ValueError, match="student encoder"):
        _build_kd_runtime(
            EgoDistillConfig(teacher_checkpoint=checkpoint, warm_start_readout=False),
            non_grit_student,
            device=torch.device("cpu"),
        )

    grit_student = EgoStitchModel(E2EConfig.from_mapping(_tiny_mapping("full_ego_features")))
    with pytest.raises(ValueError, match="teacher encoder"):
        _build_kd_runtime(
            EgoDistillConfig(teacher_checkpoint=checkpoint, warm_start_readout=False),
            grit_student,
            device=torch.device("cpu"),
        )


def test_global_live_row_mean_is_accumulation_and_ddp_invariant() -> None:
    local_sums = [torch.tensor(2.0), torch.tensor(9.0)]
    local_counts = [torch.tensor(1.0), torch.tensor(3.0)]
    global_count = torch.stack(local_counts).sum()
    expected = torch.stack(local_sums).sum() / global_count

    accumulated = torch.stack(
        [
            e2e_global_live_row_mean(
                value,
                live_rows=count,
                world_size=1,
                global_denominator=global_count,
            )
            for value, count in zip(local_sums, local_counts, strict=True)
        ]
    ).sum()
    ddp_rank_losses = torch.stack(
        [
            e2e_global_live_row_mean(
                value,
                live_rows=count,
                world_size=2,
                global_denominator=global_count,
            )
            for value, count in zip(local_sums, local_counts, strict=True)
        ]
    )

    torch.testing.assert_close(accumulated, expected)
    torch.testing.assert_close(ddp_rank_losses.mean(), expected)


def test_kd_terms_are_finite_and_backpropagate_into_the_student_encoder(
    tmp_path: Path,
) -> None:
    checkpoint, _ = _teacher_checkpoint(tmp_path, _tiny_mapping("full_ego_oracle"))
    torch.manual_seed(5)
    student = EgoStitchModel(E2EConfig.from_mapping(_tiny_mapping("full_ego_features")))
    runtime = _build_kd_runtime(
        EgoDistillConfig(teacher_checkpoint=checkpoint),
        student,
        device=torch.device("cpu"),
    )

    truth = nx.Graph([("0", "1"), ("0", "2"), ("1", "2"), ("2", "3"), ("3", "4")])
    truth.add_node("5")
    node_ids = [str(node) for node in range(6)]
    generator = cast(FullEgoFeaturesGenerator, student.generator)
    generator.set_oracle_context(truth, node_ids)
    generator.set_node_features(torch.randn(6, student.input_dim), node_ids)
    generator.set_stash_teacher_view(True)

    rows_a, rows_b = [0, 3], [1, 4]
    state_a = generator.encode_node(
        torch.zeros(2, 3), torch.zeros(2, 1, 3), node_rows=torch.tensor(rows_a)
    )
    state_b = generator.encode_node(
        torch.zeros(2, 3), torch.zeros(2, 1, 3), node_rows=torch.tensor(rows_b)
    )
    graph = generator.stitch(state_a, state_b, torch.tensor([False, False]))
    assert student.encoder is not None
    embedding_ab = student.encoder(graph)
    embedding_ba = student.encoder(graph.swapped())

    step = _CompositeStep(student, 1, kd=runtime)
    seed_sum, gram_sum, live_rows, gram_live_rows = step._kd_terms(
        graph, embedding_ab, embedding_ba, torch.ones(2)
    )
    seed = seed_sum / live_rows
    gram = gram_sum / gram_live_rows

    assert torch.isfinite(seed) and torch.isfinite(gram)
    assert seed.item() >= 0.0 and gram.item() >= 0.0
    (seed + gram).backward()  # type: ignore[no-untyped-call]
    encoder_grads = [
        parameter.grad for parameter in student.encoder.parameters() if parameter.grad is not None
    ]
    assert encoder_grads
    assert all(torch.isfinite(grad).all() for grad in encoder_grads)
    assert all(p.grad is None for p in runtime.teacher_encoder.parameters())

    # Dead rows drop out of both terms.
    embedding_ab_2 = student.encoder(graph)
    embedding_ba_2 = student.encoder(graph.swapped())
    seed_masked_sum, gram_masked_sum, masked_rows, masked_gram_rows = step._kd_terms(
        graph, embedding_ab_2, embedding_ba_2, torch.tensor([1.0, 0.0])
    )
    seed_masked = seed_masked_sum / masked_rows
    gram_masked = gram_masked_sum / masked_gram_rows
    assert torch.isfinite(seed_masked) and torch.isfinite(gram_masked)

    # An isolated self-pair is seed-live but has no off-diagonal Gram entries.
    isolated_a = generator.encode_node(
        torch.zeros(2, 3), torch.zeros(2, 1, 3), node_rows=torch.tensor([0, 5])
    )
    isolated_b = generator.encode_node(
        torch.zeros(2, 3), torch.zeros(2, 1, 3), node_rows=torch.tensor([1, 5])
    )
    torch.testing.assert_close(
        generator.candidate_node_counts(torch.tensor([0, 5]), torch.tensor([1, 5])),
        torch.tensor([3, 1]),
    )
    isolated_graph = generator.stitch(isolated_a, isolated_b, torch.tensor([False, True]))
    isolated_ab = student.encoder(isolated_graph)
    isolated_ba = student.encoder(isolated_graph.swapped())
    _, _, isolated_seed_rows, isolated_gram_rows = step._kd_terms(
        isolated_graph, isolated_ab, isolated_ba, torch.ones(2)
    )
    assert isolated_seed_rows.item() == 2.0
    assert isolated_gram_rows.item() == 1.0

    # Without the stash the trainer must fail loudly, not silently skip KD.
    generator.set_stash_teacher_view(False)
    bare_graph = generator.stitch(state_a, state_b, torch.tensor([False, False]))
    with pytest.raises(RuntimeError, match="stashed teacher view"):
        step._kd_terms(bare_graph, embedding_ab_2, embedding_ba_2, torch.ones(2))

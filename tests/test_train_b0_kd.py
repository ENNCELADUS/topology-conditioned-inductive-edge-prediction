"""Tests for the B0 trainer's row-bank and strict-LLP context-stream KD paths.

Covers: (a) the row-exact join and its failure modes, (b) task/KD sharing exact
row IDs, (c) exactly one forward and one backward per batch under a real KD
bank, (d) the zero-weight/no-`distill:` matched control, (e) the epoch-banked
anchor/context stream and retired artifact fields, (f)
epoch telemetry keys and `_pearson_from_moments`, (g) the `_evaluate_distributed`
validation-only KD diagnostics, and (h) architecture-mismatch fail-closed checks.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import src.train_b0 as train_b0
import torch
import torch.distributed as dist
from accelerate import Accelerator
from src.data.packed_features import PackedFeatureManifest, PackedFeatureTable, PackedNodeRecord
from src.distill.artifacts import (
    KDContextBank,
    KDContextTargets,
    KDRowTargets,
    load_kd_targets,
    write_kd_targets,
)
from src.distill.config import DistillConfig
from src.distill.losses import kd_dist_loss, kd_gram_loss, kd_logit_loss, kd_rank_loss
from src.model.egostitch.classifier.b0_v31 import V3_1
from src.train_b0 import (
    KDContextStream,
    KDRowBank,
    KDValDiagnostics,
    ValidationOutcome,
    _evaluate_distributed,
    _gather_global_relational_rows,
    _pearson_from_moments,
    _scale_kd_loss,
    train_ddp_loop,
)
from torch import nn
from torch.multiprocessing.spawn import spawn
from torch.nn.parallel import DistributedDataParallel

from tests.test_train_b0 import (
    _batch_of,
    _constant_metrics,
    _make_synthetic_pair_dataset,
    _tiny_config,
    _TinyPairMLP,
)

pytestmark = pytest.mark.unit

Pair = tuple[str, str]


class _ProductionRankToy(nn.Module):
    """One-parameter scorer used by the production-seam DDP regression."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.forward_calls = 0

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.forward_calls += 1
        return {"logits": self.weight * batch["x"]}


def _relational_ddp_worker(
    rank: int, world_size: int, init_file: str, result_dir: str, arm: str
) -> None:
    """Run one real gloo DDP step where each rank alone has one relation-free row."""
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        if arm.startswith("rank"):
            base = nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                base.weight.fill_(0.25)
            model = DistributedDataParallel(base)
            if arm == "rank_uneven" and rank == 1:
                x = torch.tensor([[-1.0], [-0.5]])
                teacher = torch.tensor([1.0, 0.5])
                endpoint_a = torch.tensor([1, 2])
                endpoint_b = torch.tensor([3, 4])
                row_ids = torch.tensor([1, 2])
            else:
                x = torch.tensor([[1.0 if rank == 0 else -1.0]])
                teacher = torch.tensor([-1.0 if rank == 0 else 1.0])
                endpoint_a = torch.tensor([rank])
                endpoint_b = torch.tensor([2 if arm == "rank" else 3])
                row_ids = torch.tensor([rank])
            student = model(x).squeeze(-1)
            gathered = _gather_global_relational_rows(
                student,
                teacher,
                endpoint_a,
                endpoint_b,
                row_ids,
                world_size=world_size,
            )
            non_self = gathered.endpoint_a != gathered.endpoint_b
            groups = torch.cat([gathered.endpoint_a, gathered.endpoint_b[non_self]])
            grouped_student = torch.cat([gathered.student, gathered.student[non_self]])
            grouped_teacher = torch.cat([gathered.teacher, gathered.teacher[non_self]])
            loss = kd_rank_loss(grouped_student, grouped_teacher, groups) + kd_dist_loss(
                grouped_student, grouped_teacher, groups
            )
        else:
            base = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                base.weight.copy_(torch.eye(2))
            model = DistributedDataParallel(base)
            x = torch.tensor([[1.0, 0.0]]) if rank == 0 else torch.tensor([[1.0, 1.0]])
            student = model(x)
            teacher = torch.tensor([[1.0, 0.0]]) if rank == 0 else torch.tensor([[0.0, 1.0]])
            gathered = _gather_global_relational_rows(
                student,
                teacher,
                torch.tensor([rank]),
                torch.tensor([2]),
                torch.tensor([rank]),
                world_size=world_size,
            )
            loss = kd_gram_loss(gathered.student, gathered.teacher)
        probe = torch.autograd.grad(loss, tuple(base.parameters()), retain_graph=True)
        assert all(torch.isfinite(gradient).all() for gradient in probe)
        loss.backward()  # type: ignore[no-untyped-call]
        gradient = base.weight.grad
        assert gradient is not None
        torch.save(
            {"loss": loss.detach(), "grad": gradient.detach()},
            Path(result_dir) / f"{arm}-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def _production_rank_bank_worker(
    rank: int, world_size: int, init_file: str, result_dir: str
) -> None:
    """Exercise KDRowBank.loss plus the production relational scaling decision."""
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        targets = KDRowTargets(
            node_ids=["n0", "n1", "n2", "n3", "n4"],
            pair_a_idx=np.array([0, 1, 2], dtype=np.int32),
            pair_b_idx=np.array([3, 3, 4], dtype=np.int32),
            pair_label=np.array([1, 0, 0], dtype=np.int8),
            teacher_logit=np.array([-1.0, 1.0, 0.5], dtype=np.float32),
            teacher_rep=np.zeros((3, 2), dtype=np.float16),
            val_pair_a_idx=np.array([0], dtype=np.int32),
            val_pair_b_idx=np.array([3], dtype=np.int32),
            val_pair_label=np.array([1], dtype=np.int8),
            val_teacher_logit=np.array([-1.0], dtype=np.float32),
            val_teacher_rep=np.zeros((1, 2), dtype=np.float16),
            manifest={},
        )
        base = _ProductionRankToy()
        backward_calls: list[int] = []

        def count_backward(gradient: torch.Tensor) -> torch.Tensor:
            backward_calls.append(1)
            return gradient

        base.weight.register_hook(count_backward)  # type: ignore[no-untyped-call]
        bank = KDRowBank(
            DistillConfig(targets_path="t", w_rank=1.0, w_dist=1.0),
            targets,
            train_pairs=[("n0", "n3"), ("n1", "n3"), ("n2", "n4")],
            train_labels=[1, 0, 0],
            val_pairs=[("n0", "n3")],
            val_labels=[1],
            model=base,
            device=torch.device("cpu"),
        )
        model = DistributedDataParallel(base)
        batch = (
            {"x": torch.tensor([1.0]), "_row_id": torch.tensor([0])}
            if rank == 0
            else {"x": torch.tensor([-1.0, -0.5]), "_row_id": torch.tensor([1, 2])}
        )
        output = model(batch)
        kd_global, _ = bank.loss(batch, output, world_size=world_size)
        loss = _scale_kd_loss(
            bank,
            kd_global,
            local_count=int(batch["_row_id"].numel()),
            global_count=3,
            world_size=world_size,
        )
        loss.backward()  # type: ignore[no-untyped-call]
        gradient = base.weight.grad
        assert gradient is not None
        torch.save(
            {
                "loss": loss.detach(),
                "grad": gradient.detach(),
                "forward_calls": base.forward_calls,
                "backward_calls": len(backward_calls),
            },
            Path(result_dir) / f"rank-bank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def _context_stream_ddp_worker(rank: int, world_size: int, init_file: str, result_dir: str) -> None:
    """Accumulate unequal context forwards, then execute one real DDP backward."""
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        targets, table = _ragged_context_fixture()
        base = _ContextToy()
        backward_calls: list[int] = []

        def count_backward(gradient: torch.Tensor) -> torch.Tensor:
            backward_calls.append(1)
            return gradient

        base.weight.register_hook(count_backward)  # type: ignore[no-untyped-call]
        model = DistributedDataParallel(base, broadcast_buffers=False)
        stream = _context_stream(targets, table, rank=rank, world_size=world_size)
        # Production's task forward runs through the DDP wrapper and arms its
        # reducer; the context forwards bypass the wrapper.
        model({"emb_a": table.tokens[:1, None], "emb_b": table.tokens[1:2, None]})
        base.forward_calls = 0
        losses = [stream.loss(model, epoch=1, step=step, steps=3)[0] for step in range(3)]
        torch.stack(losses).sum().backward()  # type: ignore[no-untyped-call]
        assert base.weight.grad is not None
        torch.save(
            {
                "grad": base.weight.grad.detach(),
                "forward_calls": base.forward_calls,
                "backward_calls": len(backward_calls),
                "validation": stream.validation_diagnostics(model),
            },
            Path(result_dir) / f"context-stream-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


# --------------------------------------------------------------------------- artifact helpers


def _node_index(node_ids: list[str], pairs: list[Pair]) -> tuple[np.ndarray, np.ndarray]:
    index = {node: position for position, node in enumerate(node_ids)}
    a_idx = np.array([index[a] for a, _ in pairs], dtype=np.int32)
    b_idx = np.array([index[b] for _, b in pairs], dtype=np.int32)
    return a_idx, b_idx


def _write_targets(
    out_dir: Path,
    *,
    node_ids: list[str],
    train_pairs: list[Pair],
    train_labels: list[int],
    teacher_logit: np.ndarray,
    teacher_rep: np.ndarray | None = None,
    val_pairs: list[Pair] | None = None,
    val_labels: list[int] | None = None,
) -> None:
    """Write one minimal, internally consistent KD row-targets artifact."""
    a_idx, b_idx = _node_index(node_ids, train_pairs)
    rep_dim = teacher_rep.shape[1] if teacher_rep is not None else 4
    rep = (
        teacher_rep
        if teacher_rep is not None
        else np.zeros((len(train_pairs), rep_dim), dtype=np.float32)
    )

    v_pairs = val_pairs if val_pairs is not None else train_pairs[:1]
    v_labels = val_labels if val_labels is not None else train_labels[:1]
    v_a_idx, v_b_idx = _node_index(node_ids, v_pairs)
    v_logit = np.asarray(teacher_logit[: len(v_pairs)], dtype=np.float32)
    v_rep = rep[: len(v_pairs)]

    write_kd_targets(
        out_dir,
        node_ids=node_ids,
        pair_a_idx=a_idx,
        pair_b_idx=b_idx,
        pair_label=np.asarray(train_labels, dtype=np.int8),
        teacher_logit=np.asarray(teacher_logit, dtype=np.float32),
        teacher_rep=np.asarray(rep, dtype=np.float32),
        val_pair_a_idx=v_a_idx,
        val_pair_b_idx=v_b_idx,
        val_pair_label=np.asarray(v_labels, dtype=np.int8),
        val_teacher_logit=v_logit,
        val_teacher_rep=v_rep,
        truth_graph_sha256="0" * 64,
        checkpoint_path=out_dir / "ckpt.pt",
        checkpoint_sha256="1" * 64,
        rep_source="topo",
        checkpoint_id=None,
    )


def _ring_pairs(node_ids: list[str]) -> list[Pair]:
    """One row per node, paired to its ring successor -- a simple distinct-row set."""
    n = len(node_ids)
    return [(node_ids[i], node_ids[(i + 1) % n]) for i in range(n)]


def _context_fixture(
    n_nodes: int = 5, n_banks: int = 2
) -> tuple[KDContextTargets, PackedFeatureTable]:
    node_ids = [f"n{i}" for i in range(n_nodes)]
    anchor_idx = np.arange(n_nodes, dtype=np.int32)
    anchor_offsets = np.arange(0, 3 * n_nodes + 1, 3, dtype=np.int64)
    partner_idx = np.asarray(
        [(anchor + offset + 1) % n_nodes for anchor in range(n_nodes) for offset in range(3)],
        dtype=np.int32,
    )
    pair_a_idx = np.repeat(anchor_idx, 3)
    score_idx = np.arange(len(partner_idx), dtype=np.int32)
    bank = KDContextBank(
        anchor_idx=anchor_idx,
        anchor_offsets=anchor_offsets,
        partner_idx=partner_idx,
        score_idx=score_idx,
        is_near=np.ones(len(partner_idx), dtype=np.bool_),
    )
    val_bank = KDContextBank(
        anchor_idx=np.array([0], dtype=np.int32),
        anchor_offsets=np.array([0, 3], dtype=np.int64),
        partner_idx=partner_idx[:3],
        score_idx=score_idx[:3],
        is_near=np.ones(3, dtype=np.bool_),
    )
    targets = KDContextTargets(
        node_ids=node_ids,
        pair_a_idx=pair_a_idx,
        pair_b_idx=partner_idx.copy(),
        teacher_logit=np.linspace(-2.0, 2.0, len(partner_idx), dtype=np.float32),
        banks=tuple(bank for _ in range(n_banks)),
        val_bank=val_bank,
        manifest={"n_banks": n_banks},
    )
    records = tuple(
        PackedNodeRecord(node, 0, index, index, 1) for index, node in enumerate(node_ids)
    )
    manifest = PackedFeatureManifest(
        format="test",
        input_dim=1,
        dtype="bfloat16",
        source_metadata_sha256="",
        source_index_sha256="",
        nodes=records,
        shards=(),
        pack_workers=1,
        build_seconds=0.0,
    )
    tokens = torch.arange(1, n_nodes + 1, dtype=torch.float32).unsqueeze(-1)
    table = PackedFeatureTable(tokens, torch.arange(n_nodes), torch.ones(n_nodes), manifest)
    return targets, table


def _ragged_context_fixture() -> tuple[KDContextTargets, PackedFeatureTable]:
    """Five anchors with non-proportional context-pair and anchor counts by rank."""
    node_ids = [f"n{i}" for i in range(9)]
    context_lengths = [2, 3, 4, 2, 5]
    anchor_idx = np.arange(5, dtype=np.int32)
    anchor_offsets = np.concatenate(([0], np.cumsum(context_lengths))).astype(np.int64)
    partner_idx = np.asarray(
        [5 + offset % 4 for length in context_lengths for offset in range(length)],
        dtype=np.int32,
    )
    score_idx = np.arange(len(partner_idx), dtype=np.int32)
    bank = KDContextBank(
        anchor_idx=anchor_idx,
        anchor_offsets=anchor_offsets,
        partner_idx=partner_idx,
        score_idx=score_idx,
        is_near=np.ones(len(partner_idx), dtype=np.bool_),
    )
    val_bank = KDContextBank(
        anchor_idx=np.array([0], dtype=np.int32),
        anchor_offsets=np.array([0, 2], dtype=np.int64),
        partner_idx=partner_idx[:2],
        score_idx=score_idx[:2],
        is_near=np.ones(2, dtype=np.bool_),
    )
    targets = KDContextTargets(
        node_ids=node_ids,
        pair_a_idx=np.repeat(anchor_idx, context_lengths),
        pair_b_idx=partner_idx.copy(),
        teacher_logit=np.linspace(-2.5, 2.0, len(partner_idx), dtype=np.float32),
        banks=(bank, bank),
        val_bank=val_bank,
        manifest={"n_banks": 2},
    )
    lengths = [1, 1, 1, 1, 1, 64, 160, 300, 500]
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1]))).astype(np.int64)
    records = tuple(
        PackedNodeRecord(node, 0, int(offset), int(offset), length)
        for node, offset, length in zip(node_ids, offsets, lengths, strict=True)
    )
    manifest = PackedFeatureManifest(
        format="test",
        input_dim=1,
        dtype="float32",
        source_metadata_sha256="",
        source_index_sha256="",
        nodes=records,
        shards=(),
        pack_workers=1,
        build_seconds=0.0,
    )
    tokens = torch.arange(1, sum(lengths) + 1, dtype=torch.float32).unsqueeze(-1)
    table = PackedFeatureTable(
        tokens,
        torch.as_tensor(offsets),
        torch.as_tensor(lengths),
        manifest,
    )
    return targets, table


def _reorder_context_targets(targets: KDContextTargets, order: list[int]) -> KDContextTargets:
    """Reorder each epoch bank while preserving score-table row identities."""

    def reorder(bank: KDContextBank) -> KDContextBank:
        rows = [
            np.arange(bank.anchor_offsets[position], bank.anchor_offsets[position + 1])
            for position in order
        ]
        lengths = [len(row) for row in rows]
        flat = np.concatenate(rows).astype(np.int64)
        return KDContextBank(
            anchor_idx=bank.anchor_idx[order],
            anchor_offsets=np.concatenate(([0], np.cumsum(lengths))).astype(np.int64),
            partner_idx=bank.partner_idx[flat],
            score_idx=bank.score_idx[flat],
            is_near=bank.is_near[flat],
        )

    return replace(targets, banks=tuple(reorder(bank) for bank in targets.banks))


class _ContextToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.forward_calls = 0
        self.bucket_boundaries: list[int] = []
        self.batch_sizes: list[int] = []

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.forward_calls += 1
        self.bucket_boundaries.append(int(batch["emb_a"].shape[1]))
        self.batch_sizes.append(int(batch["emb_a"].shape[0]))
        # Scale the padded embeddings so autograd saves boundary-shaped
        # tensors: a checkpoint recompute at the wrong bucket boundary raises.
        scaled_a = (self.weight * batch["emb_a"]).sum(dim=(1, 2))
        scaled_b = (self.weight * batch["emb_b"]).sum(dim=(1, 2))
        return {"logits": scaled_a - scaled_b}


def _context_stream(
    targets: KDContextTargets,
    table: PackedFeatureTable,
    *,
    rank: int = 0,
    world_size: int = 1,
    epochs: int = 2,
    token_budget: int = 1 << 20,
) -> KDContextStream:
    return KDContextStream(
        DistillConfig(targets_path="rows", context_targets_path="contexts", w_rank=1.0, w_dist=1.0),
        targets,
        table,
        allowed_nodes=frozenset(targets.node_ids),
        forbidden_internal_nodes=frozenset({"n0"}),
        epochs=epochs,
        rank=rank,
        world_size=world_size,
        token_budget=token_budget,
    )


def test_context_stream_exactly_once_coverage_and_bank_indexing() -> None:
    targets, table = _context_fixture()
    stream = _context_stream(targets, table)
    model = _ContextToy()
    covered: list[int] = []
    for step in range(3):
        stream.loss(model, epoch=2, step=step, steps=3)
        covered.extend(stream.last_anchor_idx)
        assert stream.last_bank_index == 1
    assert covered == list(range(len(targets.node_ids)))


def test_context_stream_rejects_more_epochs_than_banks() -> None:
    targets, table = _context_fixture(n_banks=1)
    with pytest.raises(ValueError, match="epochs .* exceed context banks"):
        _context_stream(targets, table, epochs=2)


def test_context_stream_fails_closed_on_universe_pack_and_vval_drift() -> None:
    targets, table = _context_fixture()
    config = DistillConfig(
        targets_path="rows", context_targets_path="contexts", w_rank=1.0, w_dist=1.0
    )
    with pytest.raises(ValueError, match="outside the training universe"):
        KDContextStream(
            config,
            targets,
            table,
            allowed_nodes=frozenset(targets.node_ids[:-1]),
            forbidden_internal_nodes=frozenset({"n0"}),
            epochs=2,
            rank=0,
            world_size=1,
            token_budget=1 << 20,
        )

    short_manifest = replace(table.manifest, nodes=table.manifest.nodes[:-1])
    short_table = PackedFeatureTable(table.tokens, table.offsets, table.lengths, short_manifest)
    with pytest.raises(ValueError, match="missing from the feature pack"):
        _context_stream(targets, short_table)

    with pytest.raises(ValueError, match="V_val-internal pair"):
        KDContextStream(
            config,
            targets,
            table,
            allowed_nodes=frozenset(targets.node_ids),
            forbidden_internal_nodes=frozenset({"n0", "n1"}),
            epochs=2,
            rank=0,
            world_size=1,
            token_budget=1 << 20,
        )


def test_context_stream_one_rank_and_two_rank_gradients_match_exactly() -> None:
    targets, table = _context_fixture()
    one_model = _ContextToy()
    one_loss, _ = _context_stream(targets, table).loss(one_model, epoch=1, step=0, steps=1)
    one_loss.backward()  # type: ignore[no-untyped-call]
    assert one_model.weight.grad is not None

    rank_gradients: list[torch.Tensor] = []
    for rank in range(2):
        model = _ContextToy()
        loss, _ = _context_stream(targets, table, rank=rank, world_size=2).loss(
            model, epoch=1, step=0, steps=1
        )
        loss.backward()  # type: ignore[no-untyped-call]
        assert model.weight.grad is not None
        rank_gradients.append(model.weight.grad.detach())
    torch.testing.assert_close(torch.stack(rank_gradients).mean(), one_model.weight.grad)


def test_context_stream_ragged_rank_and_dist_denominators_are_independent() -> None:
    targets, table = _ragged_context_fixture()
    reference = _ContextToy()
    reference_loss, _ = _context_stream(targets, table).loss(reference, epoch=1, step=0, steps=1)
    reference_loss.backward()  # type: ignore[no-untyped-call]
    assert reference.weight.grad is not None

    gradients: list[torch.Tensor] = []
    local_counts: list[tuple[int, int]] = []
    for rank in range(2):
        model = _ContextToy()
        loss, stats = _context_stream(targets, table, rank=rank, world_size=2).loss(
            model, epoch=1, step=0, steps=1
        )
        loss.backward()  # type: ignore[no-untyped-call]
        assert model.weight.grad is not None
        gradients.append(model.weight.grad.detach())
        local_counts.append((int(stats["context_pairs"]), int(stats["context_anchors"])))

    assert local_counts == [(17, 3), (4, 2)]
    assert local_counts[0][0] / local_counts[1][0] != local_counts[0][1] / local_counts[1][1]
    torch.testing.assert_close(torch.stack(gradients).mean(), reference.weight.grad)


def test_context_stream_regroups_multiple_length_buckets_without_changing_loss() -> None:
    targets, table = _ragged_context_fixture()
    model = _ContextToy()
    loss, _ = _context_stream(targets, table).loss(model, epoch=1, step=0, steps=1)

    reversed_targets = _reorder_context_targets(targets, [4, 3, 2, 1, 0])
    reversed_model = _ContextToy()
    reversed_loss, _ = _context_stream(reversed_targets, table).loss(
        reversed_model, epoch=1, step=0, steps=1
    )

    assert model.bucket_boundaries == [128, 256, 384, 512]
    assert reversed_model.bucket_boundaries == model.bucket_boundaries
    torch.testing.assert_close(reversed_loss, loss)


def test_context_stream_spawned_ddp_accumulates_unequal_forwards_then_one_backward(
    tmp_path: Path,
) -> None:
    world_size = 2
    spawn(  # type: ignore[no-untyped-call]
        _context_stream_ddp_worker,
        args=(world_size, str(tmp_path / "context-stream-init"), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )
    observed = [
        torch.load(tmp_path / f"context-stream-{rank}.pt", weights_only=True)
        for rank in range(world_size)
    ]

    targets, table = _ragged_context_fixture()
    # The distributed global step groups are [0], [2, 1], [4, 3].
    reference_targets = _reorder_context_targets(targets, [0, 2, 1, 4, 3])
    reference = _ContextToy()
    reference_losses = [
        _context_stream(reference_targets, table).loss(reference, epoch=1, step=step, steps=3)[0]
        for step in range(3)
    ]
    torch.stack(reference_losses).sum().backward()  # type: ignore[no-untyped-call]
    assert reference.weight.grad is not None

    # Checkpointed context forwards recompute once inside the single backward.
    assert [result["forward_calls"] for result in observed] == [20, 10]
    assert [result["backward_calls"] for result in observed] == [1, 1]
    for result in observed:
        torch.testing.assert_close(result["grad"], reference.weight.grad, atol=1e-6, rtol=1e-6)
    expected = _context_stream(targets, table).validation_diagnostics(_ContextToy())
    for result in observed:
        assert result["validation"] == pytest.approx(expected)


def test_context_stream_chunks_every_bucket_to_the_token_budget() -> None:
    targets, table = _ragged_context_fixture()
    reference = _ContextToy()
    reference_loss, _ = _context_stream(targets, table).loss(reference, epoch=1, step=0, steps=1)

    model = _ContextToy()
    loss, _ = _context_stream(targets, table, token_budget=256).loss(
        model, epoch=1, step=0, steps=1
    )
    # Buckets hold 6/5/3/2 rows at 128/256/384/512 tokens: 2 rows fit at 128,
    # one row everywhere else (a row never splits, even above the budget).
    assert model.forward_calls == 13
    assert all(
        size * boundary <= max(256, boundary)
        for size, boundary in zip(model.batch_sizes, model.bucket_boundaries, strict=True)
    )
    torch.testing.assert_close(loss, reference_loss)
    reference_loss.backward()  # type: ignore[no-untyped-call]
    loss.backward()  # type: ignore[no-untyped-call]
    assert reference.weight.grad is not None and model.weight.grad is not None
    torch.testing.assert_close(model.weight.grad, reference.weight.grad)

    validation = _context_stream(targets, table, token_budget=256).validation_diagnostics(model)
    assert validation == pytest.approx(
        _context_stream(targets, table).validation_diagnostics(_ContextToy())
    )


def test_context_stream_validation_refuses_sharding_without_a_process_group() -> None:
    targets, table = _ragged_context_fixture()
    with pytest.raises(RuntimeError, match="process group"):
        _context_stream(targets, table, rank=0, world_size=2).validation_diagnostics(_ContextToy())


def test_context_stream_telemetry_includes_live_counts_and_ties() -> None:
    targets, table = _context_fixture()
    stream = _context_stream(targets, table)
    _, sums = stream.loss(_ContextToy(), epoch=1, step=0, steps=1)
    telemetry = stream.epoch_telemetry(Accelerator(cpu=True), sums)
    assert telemetry["kd_rank_live_pairs"] == 15.0
    assert telemetry["kd_dist_live_anchors"] == 5.0
    assert 0.0 <= telemetry["kd_rank_tie_fraction"] <= 1.0


# --------------------------------------------------------------------------- (a) row-exact join


class TestKDRowBankJoin:
    _NODE_IDS = [f"n{i}" for i in range(4)]
    _TRAIN_PAIRS = _ring_pairs(_NODE_IDS)
    _TRAIN_LABELS = [1, 0, 1, 0]

    def _artifact(self, tmp_path: Path) -> KDRowTargets:
        _write_targets(
            tmp_path / "targets",
            node_ids=self._NODE_IDS,
            train_pairs=self._TRAIN_PAIRS,
            train_labels=self._TRAIN_LABELS,
            teacher_logit=np.linspace(-1.0, 1.0, 4).astype(np.float32),
        )
        return load_kd_targets(tmp_path / "targets")

    def _bank(
        self,
        tmp_path: Path,
        *,
        train_pairs: list[Pair],
        train_labels: list[int],
        val_pairs: list[Pair] | None = None,
        val_labels: list[int] | None = None,
    ) -> KDRowBank:
        targets = self._artifact(tmp_path)
        distill = DistillConfig(targets_path="t", w_logit=1.0)
        return KDRowBank(
            distill,
            targets,
            train_pairs=train_pairs,
            train_labels=train_labels,
            val_pairs=val_pairs if val_pairs is not None else self._TRAIN_PAIRS[:1],
            val_labels=val_labels if val_labels is not None else self._TRAIN_LABELS[:1],
            model=nn.Linear(1, 1),
            device=torch.device("cpu"),
        )

    def test_matching_rows_succeeds(self, tmp_path: Path) -> None:
        bank = self._bank(tmp_path, train_pairs=self._TRAIN_PAIRS, train_labels=self._TRAIN_LABELS)
        assert bank.arm == "kd_logit"

    def test_wrong_row_count_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="rows, trainer has"):
            self._bank(
                tmp_path,
                train_pairs=self._TRAIN_PAIRS[:3],
                train_labels=self._TRAIN_LABELS[:3],
            )

    def test_permuted_rows_raise(self, tmp_path: Path) -> None:
        permuted_pairs = [
            self._TRAIN_PAIRS[1],
            self._TRAIN_PAIRS[0],
            self._TRAIN_PAIRS[2],
            self._TRAIN_PAIRS[3],
        ]
        permuted_labels = [
            self._TRAIN_LABELS[1],
            self._TRAIN_LABELS[0],
            self._TRAIN_LABELS[2],
            self._TRAIN_LABELS[3],
        ]
        with pytest.raises(ValueError, match="training block endpoints do not match"):
            self._bank(tmp_path, train_pairs=permuted_pairs, train_labels=permuted_labels)

    def test_mismatched_endpoint_raises(self, tmp_path: Path) -> None:
        bad_pairs = list(self._TRAIN_PAIRS)
        # Row 2's artifact endpoint is (n2, n3); swap in a different partner.
        bad_pairs[2] = (bad_pairs[2][0], "n1")
        with pytest.raises(ValueError, match="training block endpoints do not match"):
            self._bank(tmp_path, train_pairs=bad_pairs, train_labels=self._TRAIN_LABELS)

    def test_mismatched_label_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="training block labels do not match"):
            self._bank(tmp_path, train_pairs=self._TRAIN_PAIRS, train_labels=[1, 1, 1, 0])

    def test_mismatched_val_block_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="validation block labels do not match"):
            self._bank(
                tmp_path,
                train_pairs=self._TRAIN_PAIRS,
                train_labels=self._TRAIN_LABELS,
                val_pairs=self._TRAIN_PAIRS[:1],
                val_labels=[1 - self._TRAIN_LABELS[0]],
            )


# --------------------------------------------------------------------------- (b) exact row IDs


def test_kd_loss_uses_exact_row_ids(tmp_path: Path) -> None:
    node_ids = [f"n{i}" for i in range(8)]
    train_pairs = _ring_pairs(node_ids)[:6]
    train_labels = [i % 2 for i in range(6)]
    teacher_logit = np.array([0.1 * i for i in range(6)], dtype=np.float32)
    _write_targets(
        tmp_path / "targets",
        node_ids=node_ids,
        train_pairs=train_pairs,
        train_labels=train_labels,
        teacher_logit=teacher_logit,
    )
    targets = load_kd_targets(tmp_path / "targets")
    distill = DistillConfig(targets_path="t", w_logit=1.0)
    bank = KDRowBank(
        distill,
        targets,
        train_pairs=train_pairs,
        train_labels=train_labels,
        val_pairs=train_pairs[:1],
        val_labels=train_labels[:1],
        model=nn.Linear(1, 1),
        device=torch.device("cpu"),
    )

    student_logit = torch.tensor([0.5, -0.2, 0.9])
    output = {"logits": student_logit}

    rows_a = torch.tensor([4, 1, 3])
    loss_a, stats_a = bank.loss({"_row_id": rows_a}, output)
    expected_a = kd_logit_loss(student_logit, torch.tensor(teacher_logit[[4, 1, 3]]))
    assert torch.allclose(loss_a, expected_a, atol=1e-6)
    assert stats_a["rows"] == 3.0

    rows_b = torch.tensor([0, 2, 5])
    loss_b, _ = bank.loss({"_row_id": rows_b}, output)
    expected_b = kd_logit_loss(student_logit, torch.tensor(teacher_logit[[0, 2, 5]]))
    assert torch.allclose(loss_b, expected_b, atol=1e-6)
    assert not torch.allclose(loss_a, loss_b)


def test_kd_rank_row_bank_is_telemetry_only(tmp_path: Path) -> None:
    node_ids = ["n0", "n1", "n2"]
    train_pairs = [("n0", "n2"), ("n1", "n2")]
    train_labels = [1, 0]
    _write_targets(
        tmp_path / "targets",
        node_ids=node_ids,
        train_pairs=train_pairs,
        train_labels=train_labels,
        teacher_logit=np.array([1.0, -1.0], dtype=np.float32),
    )
    bank = KDRowBank(
        DistillConfig(targets_path="t", context_targets_path="contexts", w_rank=1.0, w_dist=1.0),
        load_kd_targets(tmp_path / "targets"),
        train_pairs=train_pairs,
        train_labels=train_labels,
        val_pairs=train_pairs[:1],
        val_labels=train_labels[:1],
        model=nn.Module(),
        device=torch.device("cpu"),
    )
    student = torch.tensor([-1.0, 1.0], requires_grad=True)
    loss, stats = bank.loss({"_row_id": torch.tensor([0, 1])}, {"logits": student})
    assert loss.item() == 0.0
    assert stats["rows"] == 2.0
    assert "rank_eligible_groups" not in stats
    assert bank.global_relational is False


def test_kd_gram_uses_shared_forward_pair_representations(tmp_path: Path) -> None:
    node_ids = ["n0", "n1", "n2"]
    train_pairs = [("n0", "n2"), ("n1", "n2")]
    train_labels = [1, 0]
    teacher_rep = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    _write_targets(
        tmp_path / "targets",
        node_ids=node_ids,
        train_pairs=train_pairs,
        train_labels=train_labels,
        teacher_logit=np.zeros(2, dtype=np.float32),
        teacher_rep=teacher_rep,
    )
    bank = KDRowBank(
        DistillConfig(targets_path="t", w_gram=1.0),
        load_kd_targets(tmp_path / "targets"),
        train_pairs=train_pairs,
        train_labels=train_labels,
        val_pairs=train_pairs[:1],
        val_labels=train_labels[:1],
        model=nn.Module(),
        device=torch.device("cpu"),
    )
    pair_repr = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    loss, stats = bank.loss(
        {"_row_id": torch.tensor([0, 1])},
        {"logits": torch.zeros(2), "pair_repr": pair_repr},
    )
    assert loss.item() > 0.0
    assert stats["sum_gram"] > 0.0
    assert bank.global_relational is True
    loss.backward()  # type: ignore[no-untyped-call]
    assert pair_repr.grad is not None and torch.isfinite(pair_repr.grad).all()


@pytest.mark.parametrize("arm", ["gram"])
def test_relational_kd_gathers_cross_rank_rows_with_exact_ddp_gradient(
    tmp_path: Path, arm: str
) -> None:
    world_size = 2
    spawn(  # type: ignore[no-untyped-call]
        _relational_ddp_worker,
        args=(world_size, str(tmp_path / f"{arm}-init"), str(tmp_path), arm),
        nprocs=world_size,
        join=True,
    )
    observed = [
        torch.load(tmp_path / f"{arm}-{rank}.pt", weights_only=True) for rank in range(world_size)
    ]

    if arm.startswith("rank"):
        weight = torch.tensor(0.25, requires_grad=True)
        if arm == "rank":
            student = torch.stack([weight, -weight])
            teacher = torch.tensor([-1.0, 1.0])
            endpoint_a = torch.tensor([0, 1])
            endpoint_b = torch.tensor([2, 2])
        else:
            student = torch.stack([weight, -weight, -0.5 * weight])
            teacher = torch.tensor([-1.0, 1.0, 0.5])
            endpoint_a = torch.tensor([0, 1, 2])
            endpoint_b = torch.tensor([3, 3, 4])
        groups = torch.cat([endpoint_a, endpoint_b])
        grouped_student = torch.cat([student, student])
        grouped_teacher = torch.cat([teacher, teacher])
        expected_loss = kd_rank_loss(grouped_student, grouped_teacher, groups) + kd_dist_loss(
            grouped_student, grouped_teacher, groups
        )
        expected_loss.backward()  # type: ignore[no-untyped-call]
        assert weight.grad is not None
        expected_grad = weight.grad.reshape(1, 1)
    else:
        weight = torch.eye(2, requires_grad=True)
        inputs = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        teacher = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        expected_loss = kd_gram_loss(inputs @ weight.T, teacher)
        expected_loss.backward()  # type: ignore[no-untyped-call]
        assert weight.grad is not None
        expected_grad = weight.grad

    assert expected_loss.item() > 0.0
    for result in observed:
        torch.testing.assert_close(result["loss"], expected_loss.detach())
        torch.testing.assert_close(result["grad"], expected_grad, atol=1e-6, rtol=1e-6)


# --------------------------------------------------------------------------- (c) one fwd/one bwd


class _CountingLogitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.forward_calls = 0

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.forward_calls += 1
        n = batch["label"].shape[0]
        logits = self.weight * torch.ones(n)
        loss = self.weight * 0.0 + batch["loss_value"].mean()
        return {"loss": loss, "logits": logits}


def test_ddp_loop_one_forward_one_backward_per_batch_with_real_kd_bank(tmp_path: Path) -> None:
    node_ids = [f"n{i}" for i in range(4)]
    train_pairs = _ring_pairs(node_ids)
    train_labels = [0, 1, 0, 1]
    _write_targets(
        tmp_path / "targets",
        node_ids=node_ids,
        train_pairs=train_pairs,
        train_labels=train_labels,
        teacher_logit=np.zeros(4, dtype=np.float32),
    )
    targets = load_kd_targets(tmp_path / "targets")
    distill = DistillConfig(targets_path="t", w_logit=1.0)
    model = _CountingLogitModel()
    bank = KDRowBank(
        distill,
        targets,
        train_pairs=train_pairs,
        train_labels=train_labels,
        val_pairs=train_pairs[:1],
        val_labels=train_labels[:1],
        model=model,
        device=torch.device("cpu"),
    )

    def _loss_batch(row_ids: list[int]) -> dict[str, torch.Tensor]:
        n = len(row_ids)
        return {
            "loss_value": torch.zeros(n),
            "label": torch.tensor([float(train_labels[row]) for row in row_ids]),
            "_row_id": torch.tensor(row_ids),
            "_local_pair_count": torch.tensor(n),
            "_global_pair_count": torch.tensor(n),
        }

    batches = [_loss_batch([0]), _loss_batch([1, 2, 3])]
    cfg = _tiny_config(epochs=1)
    accelerator = Accelerator(cpu=True)
    backward_calls: list[int] = []
    original_backward = accelerator.backward

    def counting_backward(loss: torch.Tensor, **kwargs: object) -> None:
        backward_calls.append(1)
        original_backward(loss, **kwargs)

    accelerator.backward = counting_backward

    train_ddp_loop(
        model,
        lambda epoch: batches,
        batches,
        cfg,
        accelerator,
        warmup_steps=1,
        artifact_dir=tmp_path / "attempt",
        evaluate_fn=lambda model, loader, accelerator: ValidationOutcome(_constant_metrics(), None),
        kd_bank=bank,
    )

    assert model.forward_calls == len(batches)
    assert len(backward_calls) == len(batches)


# --------------------------------------------------------------------------- (d) matched control


def test_distill_config_default_and_all_zero_are_inactive_none_arm() -> None:
    assert DistillConfig().arm == "none"
    assert DistillConfig().active is False
    zero = DistillConfig.from_mapping({"w_logit": 0.0, "w_rep": 0.0})
    assert zero.arm == "none"
    assert zero.active is False


def test_ddp_loop_distill_none_and_all_zero_are_bit_identical(tmp_path: Path) -> None:
    def _run(distill: DistillConfig | None, subdir: str) -> dict[str, torch.Tensor]:
        cfg = replace(_tiny_config(epochs=2), distill=distill)
        torch.manual_seed(7)
        model = _TinyPairMLP(input_dim=4, hidden_dims=(8,), dropout=0.0)
        batch = _batch_of(_make_synthetic_pair_dataset(8, input_dim=4, seed=1))
        batch["_row_id"] = torch.arange(8)
        batch["_local_pair_count"] = torch.tensor(8)
        batch["_global_pair_count"] = torch.tensor(8)
        result = train_ddp_loop(
            model,
            lambda epoch: [batch],
            [batch],
            cfg,
            Accelerator(cpu=True),
            warmup_steps=1,
            artifact_dir=tmp_path / subdir,
            evaluate_fn=lambda model, loader, accelerator: ValidationOutcome(
                _constant_metrics(), None
            ),
            kd_bank=None,
        )
        return result.last_state_dict

    none_state = _run(None, "none")
    zero_state = _run(
        DistillConfig.from_mapping({"w_logit": 0.0, "w_rep": 0.0}),
        "zero",
    )

    assert none_state.keys() == zero_state.keys()
    for key in none_state:
        torch.testing.assert_close(none_state[key], zero_state[key], rtol=0.0, atol=0.0)


# --------------------------------------------------------------------------- (e) obsolete paths


def test_obsolete_kd_sampler_modules_are_gone() -> None:
    for module_name in (
        "src.distill.content_logit",
        "src.distill.heuristic_targets",
    ):
        assert importlib.util.find_spec(module_name) is None
    assert hasattr(train_b0, "KDContextStream")
    assert not hasattr(train_b0, "KDStream")


def test_kd_row_targets_has_no_obsolete_fields() -> None:
    field_names = {field.name for field in dataclasses.fields(KDRowTargets)}
    assert field_names.isdisjoint(
        {"anchor_offsets", "is_near", "teacher_pooled_ab", "content_logit"}
    )


def test_distill_config_has_no_obsolete_fields() -> None:
    field_names = {field.name for field in dataclasses.fields(DistillConfig)}
    assert "context_targets_path" in field_names
    assert field_names.isdisjoint({"anchors_per_step", "arm_label", "temperature"})


# --------------------------------------------------------------------------- (f) telemetry


def _tiny_v31_kwargs() -> dict[str, object]:
    return {
        "input_dim": 4,
        "d_model": 8,
        "encoder_layers": 1,
        "cross_attn_layers": 1,
        "n_heads": 2,
        "mlp_head": {"hidden_dims": [8], "dropout": 0.0},
        "regularization": {"dropout": 0.0},
    }


def _v31_batch(row_ids: list[int], labels: list[int], *, seed: int = 44) -> dict[str, torch.Tensor]:
    n = len(row_ids)
    generator = torch.Generator().manual_seed(seed)
    return {
        "emb_a": torch.randn(n, 3, 4, generator=generator),
        "emb_b": torch.randn(n, 3, 4, generator=generator),
        "len_a": torch.full((n,), 3, dtype=torch.long),
        "len_b": torch.full((n,), 3, dtype=torch.long),
        "label": torch.tensor([float(value) for value in labels]),
        "_row_id": torch.tensor(row_ids),
        "_local_pair_count": torch.tensor(n),
        "_global_pair_count": torch.tensor(n),
    }


def test_ddp_loop_kd_logit_telemetry_keys(tmp_path: Path) -> None:
    node_ids = [f"n{i}" for i in range(4)]
    train_pairs = _ring_pairs(node_ids)
    train_labels = [1, 0, 1, 0]
    teacher_logit = np.array([0.2, -0.4, 0.6, -0.1], dtype=np.float32)
    _write_targets(
        tmp_path / "targets",
        node_ids=node_ids,
        train_pairs=train_pairs,
        train_labels=train_labels,
        teacher_logit=teacher_logit,
    )
    targets = load_kd_targets(tmp_path / "targets")
    distill = DistillConfig(targets_path="t", w_logit=1.0)
    model = V3_1(**_tiny_v31_kwargs())
    bank = KDRowBank(
        distill,
        targets,
        train_pairs=train_pairs,
        train_labels=train_labels,
        val_pairs=train_pairs[:1],
        val_labels=train_labels[:1],
        model=model,
        device=torch.device("cpu"),
    )

    batch = _v31_batch([0, 1, 2, 3], train_labels)
    cfg = replace(_tiny_config(epochs=1), distill=distill)

    result = train_ddp_loop(
        model,
        lambda epoch: [batch],
        [batch],
        cfg,
        Accelerator(cpu=True),
        warmup_steps=1,
        artifact_dir=tmp_path / "attempt",
        evaluate_fn=lambda model, loader, accelerator: ValidationOutcome(
            _constant_metrics(), None, {"val_kd_logit_loss": 0.7}, 0.5
        ),
        kd_bank=bank,
    )

    entry = result.history[0]
    for key in (
        "train_kd_loss",
        "kd_logit_corr",
        "kd_logit_loss",
        "kd_prob_mae",
        "grad_norm_task",
        "grad_norm_kd",
        "val_ece",
        "val_brier",
    ):
        assert key in entry, f"missing telemetry key {key!r} in {sorted(entry)}"
    assert entry["val_task_loss"] == 0.5
    # The monitored total mirrors the training objective: task + w_logit * KD.
    assert entry["val_total_loss"] == pytest.approx(0.5 + 1.0 * 0.7)


def test_epoch_telemetry_kd_logit_loss_is_unweighted_rows_weighted_mean(tmp_path: Path) -> None:
    node_ids = [f"n{i}" for i in range(6)]
    train_pairs = _ring_pairs(node_ids)
    train_labels = [i % 2 for i in range(6)]
    teacher_logit = np.linspace(-1.0, 1.0, 6).astype(np.float32)
    _write_targets(
        tmp_path / "targets",
        node_ids=node_ids,
        train_pairs=train_pairs,
        train_labels=train_labels,
        teacher_logit=teacher_logit,
    )
    bank = KDRowBank(
        DistillConfig(targets_path="t", w_logit=2.5),
        load_kd_targets(tmp_path / "targets"),
        train_pairs=train_pairs,
        train_labels=train_labels,
        val_pairs=train_pairs[:1],
        val_labels=train_labels[:1],
        model=nn.Linear(1, 1),
        device=torch.device("cpu"),
    )

    student_a = torch.tensor([0.5, -0.2, 0.9, 0.1])
    student_b = torch.tensor([-0.4, 0.3])
    batches = ((torch.tensor([0, 1, 2, 3]), student_a), (torch.tensor([4, 5]), student_b))
    sums: dict[str, float] = {}
    totals: list[float] = []
    for rows, student in batches:
        total, stats = bank.loss({"_row_id": rows}, {"logits": student})
        totals.append(float(total.item()))
        for key, value in stats.items():
            sums[key] = sums.get(key, 0.0) + value

    telemetry = bank.epoch_telemetry(Accelerator(cpu=True), sums)
    bce_a = kd_logit_loss(student_a, torch.tensor(teacher_logit[:4])).item()
    bce_b = kd_logit_loss(student_b, torch.tensor(teacher_logit[4:])).item()
    assert telemetry["kd_logit_loss"] == pytest.approx((4.0 * bce_a + 2.0 * bce_b) / 6.0)
    # The step losses stay weighted; the telemetry term stays unweighted.
    assert totals[0] == pytest.approx(2.5 * bce_a)
    assert totals[1] == pytest.approx(2.5 * bce_b)


def test_epoch_telemetry_kd_rank_separates_unweighted_rank_and_dist(tmp_path: Path) -> None:
    del tmp_path
    targets, table = _context_fixture()
    stream = _context_stream(targets, table)
    _, stats = stream.loss(_ContextToy(), epoch=1, step=0, steps=1)
    telemetry = stream.epoch_telemetry(Accelerator(cpu=True), stats)
    assert telemetry["kd_rank_loss"] >= 0.0
    assert telemetry["kd_dist_loss"] >= 0.0


def test_epoch_telemetry_kd_rep_loss_derived_from_cos(tmp_path: Path) -> None:
    node_ids = [f"n{i}" for i in range(4)]
    train_pairs = _ring_pairs(node_ids)
    train_labels = [1, 0, 1, 0]
    _write_targets(
        tmp_path / "targets",
        node_ids=node_ids,
        train_pairs=train_pairs,
        train_labels=train_labels,
        teacher_logit=np.zeros(4, dtype=np.float32),
    )
    model = nn.Module()
    model.d_model = 4  # type: ignore[assignment]
    bank = KDRowBank(
        DistillConfig(targets_path="t", w_rep=1.0),
        load_kd_targets(tmp_path / "targets"),
        train_pairs=train_pairs,
        train_labels=train_labels,
        val_pairs=train_pairs[:1],
        val_labels=train_labels[:1],
        model=model,
        device=torch.device("cpu"),
    )

    sums = {
        "rows": 4.0,
        "sum_s": 0.0,
        "sum_t": 0.0,
        "sum_s2": 0.0,
        "sum_t2": 0.0,
        "sum_st": 0.0,
        "sum_prob_err": 0.0,
        "sum_rep_cos": 3.0,
    }
    telemetry = bank.epoch_telemetry(Accelerator(cpu=True), sums)
    assert telemetry["kd_rep_cos"] == pytest.approx(0.75)
    assert telemetry["kd_rep_loss"] == pytest.approx(0.25)


def test_pearson_from_moments_perfect_positive_correlation() -> None:
    values = [1.0, 2.0, 3.0]
    n = float(len(values))
    sum_s = sum_t = sum(values)
    sum_s2 = sum_t2 = sum(value * value for value in values)
    sum_st = sum(value * value for value in values)
    assert _pearson_from_moments(sum_s, sum_t, sum_s2, sum_t2, sum_st, n) == pytest.approx(1.0)


def test_pearson_from_moments_perfect_negative_correlation() -> None:
    student = [1.0, 2.0, 3.0]
    teacher = [3.0, 2.0, 1.0]
    n = 3.0
    sum_s, sum_t = sum(student), sum(teacher)
    sum_s2 = sum(value * value for value in student)
    sum_t2 = sum(value * value for value in teacher)
    sum_st = sum(s * t for s, t in zip(student, teacher, strict=True))
    assert _pearson_from_moments(sum_s, sum_t, sum_s2, sum_t2, sum_st, n) == pytest.approx(-1.0)


def test_pearson_from_moments_zero_variance_guard() -> None:
    student = [1.0, 1.0, 1.0]
    teacher = [1.0, 2.0, 3.0]
    n = 3.0
    sum_s, sum_t = sum(student), sum(teacher)
    sum_s2 = sum(value * value for value in student)
    sum_t2 = sum(value * value for value in teacher)
    sum_st = sum(s * t for s, t in zip(student, teacher, strict=True))
    assert _pearson_from_moments(sum_s, sum_t, sum_s2, sum_t2, sum_st, n) == pytest.approx(0.0)


# --------------------------------------------------------------------------- (g) val diagnostics


class _RepDiagModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "logits": self.scale * batch["logit_seed"],
            "pair_repr": self.scale * batch["rep_seed"],
        }


def test_evaluate_distributed_kd_rep_diagnostics() -> None:
    n = 3
    rep_dim = 4
    rep_seed = torch.randn(n, rep_dim)
    batch = {
        "label": torch.tensor([1.0, 0.0, 1.0]),
        "_row_id": torch.tensor([0, 1, 2]),
        "logit_seed": torch.tensor([0.5, -0.5, 1.0]),
        "rep_seed": rep_seed,
    }
    teacher_logit = torch.tensor([0.4, -0.6, 0.9])
    kd_val = KDValDiagnostics(
        arm="kd_rep",
        teacher_logit=teacher_logit,
        teacher_logit_np=teacher_logit.double().numpy(),
        teacher_rep=rep_seed.clone().to(torch.float16),
        teacher_latent=None,
    )
    outcome = _evaluate_distributed(
        _RepDiagModel(),
        [batch],
        Accelerator(cpu=True),
        expected_row_ids=np.arange(n),
        kd_val=kd_val,
    )
    assert outcome.kd is not None
    assert {
        "val_kd_rep_cos",
        "val_kd_rep_loss",
        "val_kd_logit_corr",
        "val_kd_logit_loss",
        "val_kd_prob_mae",
    } <= outcome.kd.keys()
    assert outcome.kd["val_kd_rep_cos"] == pytest.approx(1.0, abs=1e-3)
    assert outcome.kd["val_kd_rep_loss"] == pytest.approx(1.0 - outcome.kd["val_kd_rep_cos"])


def test_evaluate_distributed_kd_rank_reports_deterministic_block_losses() -> None:
    targets, table = _context_fixture()
    diagnostics = _context_stream(targets, table).validation_diagnostics(_ContextToy())
    assert diagnostics["val_kd_rank_loss"] >= 0.0
    assert diagnostics["val_kd_dist_loss"] >= 0.0
    assert 0.0 <= diagnostics["val_kd_rank_tie_fraction"] <= 1.0


def test_evaluate_distributed_kd_gram_reports_deterministic_block_loss() -> None:
    rep_seed = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    batch = {
        "label": torch.tensor([1.0, 0.0]),
        "_row_id": torch.tensor([0, 1]),
        "logit_seed": torch.tensor([0.0, 0.0]),
        "rep_seed": rep_seed,
    }
    teacher_logit = torch.zeros(2)
    kd_val = KDValDiagnostics(
        arm="kd_gram",
        teacher_logit=teacher_logit,
        teacher_logit_np=teacher_logit.double().numpy(),
        teacher_rep=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        teacher_latent=None,
    )
    outcome = _evaluate_distributed(
        _RepDiagModel(),
        [batch],
        Accelerator(cpu=True),
        expected_row_ids=np.arange(2),
        kd_val=kd_val,
    )
    assert outcome.kd is not None
    assert outcome.kd["val_kd_gram_block_loss"] > 0.0


def _smoothed_bce(logits: list[float], targets: list[float]) -> float:
    """Hand-rolled mean BCE-with-logits against probability targets."""
    per_row = [
        max(logit, 0.0) + math.log1p(math.exp(-abs(logit))) - target * logit
        for logit, target in zip(logits, targets, strict=True)
    ]
    return sum(per_row) / len(per_row)


def test_evaluate_distributed_reports_smoothed_task_bce() -> None:
    batch = {
        "label": torch.tensor([1.0, 0.0]),
        "_row_id": torch.tensor([0, 1]),
        "logit_seed": torch.tensor([0.5, -0.25]),
        "rep_seed": torch.randn(2, 3),
    }
    outcome = _evaluate_distributed(
        _RepDiagModel(),
        [batch],
        Accelerator(cpu=True),
        expected_row_ids=np.arange(2),
        kd_val=None,
        label_smoothing=0.05,
    )
    smoothed = [1.0 * 0.95 + 0.025, 0.0 * 0.95 + 0.025]
    assert outcome.task_loss == pytest.approx(_smoothed_bce([0.5, -0.25], smoothed))


def test_evaluate_distributed_kd_logit_loss_matches_teacher_prob_bce() -> None:
    batch = {
        "label": torch.tensor([1.0, 0.0]),
        "_row_id": torch.tensor([0, 1]),
        "logit_seed": torch.tensor([0.5, -0.5]),
        "rep_seed": torch.randn(2, 3),
    }
    teacher_logit = torch.tensor([0.4, -0.6])
    kd_val = KDValDiagnostics(
        arm="kd_logit",
        teacher_logit=teacher_logit,
        teacher_logit_np=teacher_logit.double().numpy(),
        teacher_rep=None,
        teacher_latent=None,
    )
    outcome = _evaluate_distributed(
        _RepDiagModel(),
        [batch],
        Accelerator(cpu=True),
        expected_row_ids=np.arange(2),
        kd_val=kd_val,
    )
    assert outcome.kd is not None
    teacher_prob = [1.0 / (1.0 + math.exp(-0.4)), 1.0 / (1.0 + math.exp(0.6))]
    expected = _smoothed_bce([0.5, -0.5], teacher_prob)
    assert outcome.kd["val_kd_logit_loss"] == pytest.approx(expected)


def test_evaluate_distributed_without_kd_val_leaves_kd_field_none() -> None:
    n = 2
    batch = {
        "label": torch.tensor([1.0, 0.0]),
        "_row_id": torch.tensor([0, 1]),
        "logit_seed": torch.tensor([0.1, -0.2]),
        "rep_seed": torch.randn(n, 4),
    }
    outcome = _evaluate_distributed(
        _RepDiagModel(),
        [batch],
        Accelerator(cpu=True),
        expected_row_ids=np.arange(n),
        kd_val=None,
    )
    assert outcome.kd is None


# --------------------------------------------------------------------------- (h) arch checks


class TestKDRowBankArchChecks:
    _NODE_IDS = [f"n{i}" for i in range(4)]
    _TRAIN_PAIRS = _ring_pairs(_NODE_IDS)
    _TRAIN_LABELS = [1, 0, 1, 0]

    def _targets(self, tmp_path: Path, *, teacher_rep_dim: int = 4) -> KDRowTargets:
        teacher_rep = np.random.default_rng(0).normal(size=(4, teacher_rep_dim)).astype(np.float32)
        _write_targets(
            tmp_path / "targets",
            node_ids=self._NODE_IDS,
            train_pairs=self._TRAIN_PAIRS,
            train_labels=self._TRAIN_LABELS,
            teacher_logit=np.zeros(4, dtype=np.float32),
            teacher_rep=teacher_rep,
        )
        return load_kd_targets(tmp_path / "targets")

    def _build(self, distill: DistillConfig, targets: KDRowTargets, model: nn.Module) -> KDRowBank:
        return KDRowBank(
            distill,
            targets,
            train_pairs=self._TRAIN_PAIRS,
            train_labels=self._TRAIN_LABELS,
            val_pairs=self._TRAIN_PAIRS[:1],
            val_labels=self._TRAIN_LABELS[:1],
            model=model,
            device=torch.device("cpu"),
        )

    def test_w_rep_mismatched_width_without_kd_rep_head_raises(self, tmp_path: Path) -> None:
        targets = self._targets(tmp_path, teacher_rep_dim=6)
        distill = DistillConfig(targets_path="t", w_rep=1.0)
        model = nn.Module()
        model.d_model = 8  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="d_model"):
            self._build(distill, targets, model)

    def test_kd_rep_head_present_with_w_rep_zero_raises(self, tmp_path: Path) -> None:
        targets = self._targets(tmp_path, teacher_rep_dim=4)
        distill = DistillConfig(targets_path="t", w_logit=1.0)  # w_rep == 0
        model = nn.Module()
        model.kd_rep_head = nn.Linear(4, 4)
        with pytest.raises(RuntimeError, match=r"distill\.w_rep > 0"):
            self._build(distill, targets, model)

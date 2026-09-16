"""Behavioral checks for fixed coarsening and supervised endpoint attachments."""

from typing import cast

import pytest
import torch
from scipy.sparse import csr_matrix
from src.model.egostitch.classifier.virtual_graph import VirtualGraphGenerator
from torch.nn import functional as F


def make_model() -> VirtualGraphGenerator:
    torch.manual_seed(7)
    model = VirtualGraphGenerator(8, k=3, d_z=8, heads=2)
    model.initialise(torch.randn(6, 8), torch.tensor([0, 0, 1, 1, 2, 2]), torch.ones(6, 6))
    torch.nn.init.normal_(cast(torch.nn.Linear, model.attachment[-1]).weight, std=0.2)
    return model


def test_attachment_ignores_padding_partner_and_other_batch_members() -> None:
    model = make_model()
    sequence = torch.randn(1, 4, 8)
    expected = model.attach(sequence, torch.tensor([4]))
    padded = torch.cat((sequence, torch.randn(1, 3, 8) * 100), dim=1)
    batch = torch.cat((padded, torch.randn_like(padded)), dim=0)
    lengths = torch.tensor([4, 7])
    actual = model.attach(batch, lengths)
    torch.testing.assert_close(actual[:1], expected)
    sharded = torch.cat([model.attach(batch[i : i + 1], lengths[i : i + 1]) for i in range(2)])
    torch.testing.assert_close(actual, sharded)
    assert not torch.allclose(actual[0], actual[1])
    for partner in (sequence, sequence * 100):
        out = model(sequence, partner, torch.tensor([4]), torch.tensor([4]))
        torch.testing.assert_close(out["attachment_counts_u"] / model.multiplicity, expected)
    torch.testing.assert_close(model.pool(padded, torch.tensor([4])), sequence.mean(1))


def test_counts_equal_expanded_binary_graph_within_and_across_blocks() -> None:
    model = VirtualGraphGenerator(8, k=2, d_z=8, heads=2)
    labels = torch.tensor([0, 0, 1, 1, 1])
    graph = torch.ones(5, 5) - torch.eye(5)
    model.initialise(torch.randn(5, 8), labels, graph)
    a, b = torch.tensor([[1.0, 1.0]]), torch.tensor([[0.0, 1.0]])
    ua, vb = a[0, labels], b[0, labels]
    du, dv, common = ua.sum(), vb.sum(), (ua * vb).sum()
    l3 = ua @ graph @ vb
    expected = {
        "degree_u": du,
        "degree_v": dv,
        "common": common,
        "jaccard": common / (du + dv - common),
        "l3": l3,
        "l3_density": l3 / (du * dv),
    }
    for name, value in expected.items():
        torch.testing.assert_close(model.count(a, b)[name], value[None])
    assert model.count(a, b)["l3"].item() == 12  # not 15: self paths excluded


def test_forward_swap_float32_gradients_and_frozen_structure() -> None:
    model = make_model()
    a, b = torch.randn(2, 4, 8), torch.randn(2, 3, 8)
    la, lb = torch.tensor([4, 2]), torch.tensor([3, 2])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out, swapped = model(a, b, la, lb), model(b, a, lb, la)
    torch.testing.assert_close(out["endpoint_u"], swapped["endpoint_v"])
    torch.testing.assert_close(out["relation"], swapped["relation"])
    torch.testing.assert_close(out["distance_logits"], swapped["distance_logits"])
    assert out["endpoint_u"].shape == (2, 1)
    assert out["relation"].shape == (2, 4)
    assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in out.values())
    torch.stack([t.square().sum() for t in out.values()]).sum().backward()  # type: ignore[no-untyped-call]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert {"multiplicity", "adjacency", "prototypes"} <= dict(model.named_buffers()).keys()
    assert not any(
        n.startswith(("cal_", "phi", "W", "B_logits", "m_raw")) for n, _ in model.named_parameters()
    )


def test_initialise_counts_prototypes_and_checkpoint() -> None:
    model = VirtualGraphGenerator(4, k=2, d_z=8, heads=2)
    pools = torch.randn(5, 4)
    clusters = torch.tensor([0, 0, 1, 1, 1])
    adjacency = torch.eye(5)  # input loops must not affect any statistic
    adjacency[0, 2] = adjacency[2, 0] = 1
    model.initialise(pools, clusters, csr_matrix(adjacency.numpy()))
    torch.testing.assert_close(model.multiplicity, torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(model.adjacency, torch.tensor([[0.0, 1 / 6], [1 / 6, 0.0]]))
    torch.testing.assert_close(model.prototypes[0], pools[:2].mean(0))
    torch.testing.assert_close(
        model.attachment_counts(torch.randn(2, 3, 4), None), torch.full((2, 2), 0.2)
    )
    restored = VirtualGraphGenerator(4, k=2, d_z=8, heads=2)
    restored.load_state_dict(model.state_dict())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)
    inp = torch.randn(2, 3, 4)
    torch.testing.assert_close(restored.attach(inp, None), model.attach(inp, None))


def test_zero_and_fractional_attachments_finite_bounded() -> None:
    model = make_model()
    for a in (torch.zeros(2, 3), torch.full((2, 3), 0.2), torch.rand(2, 3)):
        a.requires_grad_()
        counts = model.count(a, a)
        assert all(torch.isfinite(t).all() and (t >= 0).all() for t in counts.values())
        for key in ("jaccard", "l3_density"):
            assert (counts[key] <= 1).all()
        torch.stack([t.sum() for t in counts.values()]).sum().backward()  # type: ignore[no-untyped-call]
        assert a.grad is not None and torch.isfinite(a.grad).all()


def test_balanced_loss_handles_missing_groups() -> None:
    target = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    pred = torch.ones_like(target)
    errors = F.huber_loss(pred.log1p(), target.log1p(), reduction="none")
    expected = torch.stack(
        ((errors[0, 0] + errors[0, 1:].mean()) / 2, errors[1].mean(), errors[2].mean())
    )
    torch.testing.assert_close(VirtualGraphGenerator.attachment_loss_rows(pred, target), expected)


def test_eval_telemetry_only_attachment_usage() -> None:
    model = make_model().eval()
    a, b = torch.randn(2, 3, 8), torch.randn(2, 3, 8)
    lengths = torch.tensor([3, 2])
    out = model(a, b, lengths, lengths)
    counts = model.count(model.attach(a, lengths), model.attach(b, lengths))
    torch.testing.assert_close(out["relation"][:, 0], counts["common"].log1p())
    telemetry = model.telemetry(reset=True)
    assert telemetry["attachment_count"] == 4
    assert all(name.startswith("attachment_") for name in telemetry)
    assert model.telemetry()["attachment_count"] == 0


def test_actual_generator_learns_equal_degree_distinct_neighborhoods() -> None:
    torch.manual_seed(12)
    model = VirtualGraphGenerator(4, k=2, d_z=8, heads=2)
    prototypes = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    model.initialise(
        prototypes.repeat_interleave(2, 0), torch.tensor([0, 0, 1, 1]), torch.zeros(4, 4)
    )
    sequences = prototypes[:, None].repeat(1, 3, 1)
    target = torch.tensor([[1.5, 0.0], [0.0, 1.5]])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for _ in range(500):
        optimizer.zero_grad()
        predicted = model.attachment_counts(sequences, None)
        loss = model.attachment_loss_rows(predicted, target).mean()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    predicted = model.attachment_counts(sequences, None)
    assert (predicted - target).abs().max() < 0.15
    a = predicted / model.multiplicity
    same = model.count(a[:1], a[:1])["common"]
    different = model.count(a[:1], a[1:])["common"]
    assert same > different + 0.8


def test_empty_sequences_and_empty_clusters_fail() -> None:
    model = make_model()
    with pytest.raises(ValueError, match="nonempty"):
        model.attach(torch.randn(1, 2, 8), torch.tensor([0]))
    with pytest.raises(ValueError, match="nonempty"):
        model.initialise(torch.randn(3, 8), torch.zeros(3, dtype=torch.long), torch.zeros(3, 3))

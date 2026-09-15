"""Behavioral checks for the learned coarsening's topology coordinates."""

import torch
from src.model.egostitch.classifier.virtual_graph import VirtualGraphGenerator


def test_attachment_ignores_padding_and_other_batch_members() -> None:
    torch.manual_seed(7)
    model = VirtualGraphGenerator(8, k=3, d_z=8, heads=2)
    sequence = torch.randn(1, 4, 8)
    expected = model.attach(sequence, torch.tensor([4]))
    padded = torch.cat((sequence, torch.randn(1, 3, 8) * 100), dim=1)
    batch = torch.cat((padded, torch.randn_like(padded)), dim=0)
    actual = model.attach(batch, torch.tensor([4, 7]))[:1]
    torch.testing.assert_close(actual, expected)


def test_counts_equal_lifted_binary_graph() -> None:
    model = VirtualGraphGenerator(8, k=2, d_z=8, heads=2)
    # Two proteins in block 0, three in block 1; all cross-block edges.
    with torch.no_grad():
        model.m_raw.copy_(torch.tensor([2.0, 3.0]).expm1().log())
        model.B_logits.copy_(torch.tensor([[-100.0, 100.0], [100.0, -100.0]]))
    a = torch.tensor([[1.0, 1.0]])
    b = torch.tensor([[0.0, 1.0]])
    counts = model.count(a, b)
    expected = {
        "degree_u": 5.0,
        "degree_v": 3.0,
        "triangles_u": 6.0,
        "triangles_v": 0.0,
        "clustering_u": 0.6,
        "clustering_v": 0.0,
        "common": 3.0,
        "jaccard": 0.6,
        "l3": 6.0,
        "l3_density": 0.4,
    }
    for name, value in expected.items():
        torch.testing.assert_close(counts[name], torch.tensor([value]))


def test_forward_is_swap_equivariant_float32_and_differentiable() -> None:
    model = VirtualGraphGenerator(8, k=3, d_z=8, heads=2)
    # Learned affine values must preserve symmetry, not just identity at init.
    with torch.no_grad():
        model.cal_scale.copy_(torch.arange(model.cal_scale.numel()) + 0.5)
        model.cal_shift.copy_(torch.arange(model.cal_shift.numel()) * 0.3)
    a, b = torch.randn(2, 4, 8), torch.randn(2, 3, 8)
    la, lb = torch.tensor([4, 2]), torch.tensor([3, 2])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out, swapped = model(a, b, la, lb), model(b, a, lb, la)
    torch.testing.assert_close(out["endpoint_u"], swapped["endpoint_v"])
    torch.testing.assert_close(out["relation"], swapped["relation"])
    torch.testing.assert_close(out["distance_logits"], swapped["distance_logits"])
    assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in out.values())
    sum(t.square().sum() for t in out.values()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_initialise_coarsens_training_graph_and_preserves_checkpoint() -> None:
    from scipy.sparse import csr_matrix

    model = VirtualGraphGenerator(4, k=2, d_z=8, heads=2)
    pools = torch.randn(5, 8)
    clusters = torch.tensor([0, 0, 1, 1, 1])
    adjacency = torch.zeros(5, 5)
    adjacency[0, 2] = adjacency[2, 0] = 1
    model.initialise(pools, clusters, csr_matrix(adjacency.numpy()), 0.4)
    torch.testing.assert_close(model.multiplicity, torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(model.adjacency[0, 1], torch.tensor(1 / 6))
    torch.testing.assert_close(model.P[0], model.W(pools[:2].mean(0)))
    restored = VirtualGraphGenerator(4, k=2, d_z=8, heads=2)
    restored.load_state_dict(model.state_dict())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


def test_zero_and_fractional_attachments_are_finite_and_bounded() -> None:
    model = VirtualGraphGenerator(4, k=2, d_z=8, heads=2)
    for a in (torch.zeros(2, 2), torch.full((2, 2), 0.2), torch.rand(2, 2)):
        a.requires_grad_()
        counts = model.count(a, a)
        assert all(torch.isfinite(t).all() and (t >= 0).all() for t in counts.values())
        for key in ("clustering_u", "clustering_v", "jaccard", "l3_density"):
            assert (counts[key] <= 1).all()
        torch.stack([t.sum() for t in counts.values()]).sum().backward()  # type: ignore[no-untyped-call]
        assert a.grad is not None and torch.isfinite(a.grad).all()


def test_open_gates_match_static_counts_and_eval_telemetry_accumulates() -> None:
    model = VirtualGraphGenerator(4, k=2, d_z=8, heads=2).eval()
    model.intervention = "slot_gates_open"
    a, b = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    lengths = torch.tensor([3, 2])
    out = model(a, b, lengths, lengths)
    counts = model.count(model.attach(a, lengths), model.attach(b, lengths))
    torch.testing.assert_close(out["relation"][:, 0], counts["common"].log1p())
    telemetry = model.telemetry(reset=True)
    assert telemetry["attachment_count"] == 4
    assert telemetry["gate_count"] == 2
    torch.testing.assert_close(telemetry["gate_sum"], torch.full((2,), 2.0))
    assert model.telemetry()["gate_count"] == 0

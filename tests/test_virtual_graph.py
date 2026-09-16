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


def test_attachment_logits_read_one_bias_and_one_query_per_coarse_node() -> None:
    torch.manual_seed(3)
    model = VirtualGraphGenerator(8, k=3, d_z=8, heads=2)
    assert not hasattr(model, "attachment")
    with torch.no_grad():
        model.readout_bias.copy_(torch.tensor([-1.0, 0.0, 2.0]))
    sequence, lengths = torch.randn(2, 5, 8), torch.tensor([5, 3])
    logits = model.attachment_logits(sequence, lengths)
    assert logits.shape == (2, 3)
    # Coarse nodes differ: a shared readout would make every column identical up
    # to the (uniform) attention weights.
    spread = (logits - logits.mean(1, keepdim=True)).detach().abs().max()
    assert float(spread) > 1e-4
    # With no query direction only each node's own bias survives the readout.
    blank = VirtualGraphGenerator(8, k=3, d_z=8, heads=2)
    blank.load_state_dict(model.state_dict())
    with torch.no_grad():
        blank.P.zero_()
    torch.testing.assert_close(
        blank.attachment_logits(sequence, lengths), model.readout_bias.expand(2, 3)
    )


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
    clusters = torch.tensor([0, 0, 1, 1, 1])
    adjacency = torch.zeros(5, 5)
    adjacency[0, 2] = adjacency[2, 0] = 1
    cluster_means = torch.randn(2, 4)
    model.initialise(clusters, csr_matrix(adjacency.numpy()), cluster_means)
    torch.testing.assert_close(model.multiplicity, torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(model.adjacency[0, 1], torch.tensor(1 / 6))
    # Queries live in the key map's space, seeded on each cluster's mean state.
    torch.testing.assert_close(model.P, model.residue_projection(cluster_means))
    # Each block's readout bias reproduces that block's mean training attachment.
    one_hot = torch.zeros(5, 2)
    one_hot[torch.arange(5), clusters] = 1.0
    density = (adjacency @ one_hot).mean(0) / model.multiplicity
    expected = torch.logit(density.clamp(1e-6, 1 - 1e-6))
    torch.testing.assert_close(model.readout_bias, expected)
    # The coarse graph is data, not parameters: it stays where the graph put it.
    assert not model.B_logits.requires_grad and not model.m_raw.requires_grad
    assert model.P.requires_grad and model.readout_bias.requires_grad
    restored = VirtualGraphGenerator(4, k=2, d_z=8, heads=2)
    restored.load_state_dict(model.state_dict())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


def test_seeding_makes_each_coarse_node_read_its_own_community() -> None:
    """Untied random maps give every coarse node the same attended state (v0.5)."""
    from scipy.sparse import csr_matrix

    torch.manual_seed(5)
    k, d_model = 4, 16
    model = VirtualGraphGenerator(d_model, k=k, d_z=16, heads=2)
    centres = torch.randn(k, d_model) * 3.0
    residues = centres.repeat_interleave(3, 0)

    def own_versus_other(generator: VirtualGraphGenerator) -> tuple[float, float]:
        """Mean logit shift at node ``j`` when community ``j`` leaves the protein."""
        proteins = [residues] + [
            torch.cat((residues[: 3 * j], residues[3 * (j + 1) :])) for j in range(k)
        ]
        logits = [generator.attachment_logits(p[None], None).detach()[0] for p in proteins]
        own = [float((logits[1 + j] - logits[0])[j].abs()) for j in range(k)]
        other = [
            float((logits[1 + j] - logits[0])[[c for c in range(k) if c != j]].abs().mean())
            for j in range(k)
        ]
        return sum(own) / k, sum(other) / k

    before_own, before_other = own_versus_other(model)
    clusters = torch.arange(k).repeat_interleave(3)
    adjacency = torch.zeros(3 * k, 3 * k)
    adjacency[0, 3] = adjacency[3, 0] = 1
    model.initialise(clusters, csr_matrix(adjacency.numpy()), centres)
    after_own, after_other = own_versus_other(model)
    assert before_own < 2.0 * before_other
    assert after_own > 3.0 * after_other


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


def test_attachment_loss_is_zero_at_the_targets_and_reaches_every_coarse_node() -> None:
    model = VirtualGraphGenerator(8, k=3, d_z=8, heads=2)
    with torch.no_grad():
        model.m_raw.copy_(torch.tensor([4.0, 4.0, 4.0]).expm1().log())
    multiplicity = model.multiplicity.detach()
    targets = torch.tensor([[[2.0, 0.0, 1.0], [0.0, 3.0, 1.0]]])
    exact = {"attach_a": targets[:, 0] / multiplicity, "attach_b": targets[:, 1] / multiplicity}
    torch.testing.assert_close(model.attachment_loss_rows(exact, targets), torch.zeros(1))
    # Under-counting by the whole block is a real, order-one loss.
    zero = {"attach_a": torch.zeros(1, 3), "attach_b": torch.zeros(1, 3)}
    assert float(model.attachment_loss_rows(zero, targets).detach()) > 0.05


def test_attachment_loss_trains_every_coarse_node_readout() -> None:
    torch.manual_seed(19)
    model = VirtualGraphGenerator(8, k=3, d_z=8, heads=2)
    encoded, lengths = torch.randn(4, 6, 8), torch.tensor([6, 6, 4, 5])
    parts = model(encoded, encoded.flip(0), lengths, lengths)
    assert parts["attach_a"].shape == (4, 3) and parts["attach_b"].dtype == torch.float32
    targets = torch.randint(0, 4, (4, 2, 3)).float()
    model.attachment_loss_rows(parts, targets).mean().backward()  # type: ignore[no-untyped-call]
    assert model.readout_bias.grad is not None
    # The per-node signal v0.5 never had: every coarse node gets its own gradient.
    assert bool((model.readout_bias.grad.abs() > 1e-8).all())


def test_eval_telemetry_records_attachment_logit_moments() -> None:
    torch.manual_seed(23)
    model = VirtualGraphGenerator(4, k=3, d_z=8, heads=2).eval()
    a, b = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    lengths = torch.tensor([3, 2])
    model(a, b, lengths, lengths)
    telemetry = model.telemetry(reset=True)
    logits = torch.cat((model.attachment_logits(a, lengths), model.attachment_logits(b, lengths)))
    torch.testing.assert_close(telemetry["attachment_logit_sum"], logits.sum(0))
    torch.testing.assert_close(telemetry["attachment_logit_sq_sum"], logits.square().sum(0))
    torch.testing.assert_close(
        telemetry["attachment_within_var_sum"], logits.var(dim=1, correction=0).sum()
    )
    empty = model.telemetry()
    assert float(empty["attachment_within_var_sum"]) == 0.0
    assert empty["attachment_logit_sum"].shape == (3,)

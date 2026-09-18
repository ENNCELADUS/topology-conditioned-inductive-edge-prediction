"""Motif-prompt model tests: config, generator, reader, prefix interface, identity."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from src.data.motif_template import (
    SWAP_PERM,
    MotifTemplateStatistics,
    count_statistics,
    role_permutation,
    template_statistics,
)
from src.model.egostitch.classifier.motif_prompt import (
    FIELD_ORDER,
    GATE_MODES,
    GENERATOR_GRAD_GROUPS,
    TEMPLATE_KEY,
    MotifCountHead,
    MotifGenerator,
    MotifGritReader,
    MotifPromptAdapter,
    MotifPromptConfig,
    ReaderConfig,
    V3_1MotifPrompt,
    dense_adjacency,
    motif_rrwp,
    typed_adjacency,
)
from src.model.egostitch.classifier.prefix import prefix_branch
from src.model.egostitch.encoder.grit_gmt import dense_rrwp

from tests.test_prefix_model import _pair_batch, _tiny_base_config


def _statistics(mean: torch.Tensor) -> MotifTemplateStatistics:
    """Corpus statistics of a one-row corpus whose mean is exactly ``mean``."""
    return template_statistics(mean.reshape(1, -1).numpy())


def test_config_round_trip_and_defaults() -> None:
    cfg = MotifPromptConfig.from_mapping({"stage": "one", "base_checkpoint": "base.pt"})
    assert cfg.stage == "one"
    assert cfg.width == 128
    assert cfg.slots_per_field == 2
    assert cfg.fields == FIELD_ORDER
    assert cfg.families == ("closure", "bridge")
    assert cfg.token_source == "graph"
    assert cfg.gate_mode == "learned"
    assert cfg.interface_warmup_epochs == 2
    assert (cfg.w_slot, cfg.w_topo) == (1.0, 0.1)
    assert (cfg.beta_p, cfg.beta_q, cfg.beta_a, cfg.beta_i) == (1.0, 1.0, 1.0, 1.0)
    assert cfg.huber_delta == 1.0
    assert cfg.slot_read == "bare"
    assert cfg.head_output_init_std == 1e-3
    assert cfg.reader == ReaderConfig(layers=3, dim=96, heads=4, rrwp_k=4)
    assert MotifPromptConfig.from_mapping(cfg.to_dict()) == cfg
    tuned = MotifPromptConfig.from_mapping(
        {
            "stage": "one",
            "base_checkpoint": "base.pt",
            "slot_read": "residual_block",
            "head_output_init_std": 1e-2,
        }
    )
    assert (tuned.slot_read, tuned.head_output_init_std) == ("residual_block", 1e-2)
    assert MotifPromptConfig.from_mapping(tuned.to_dict()) == tuned


def test_config_validation_rejects_illegal_blocks() -> None:
    with pytest.raises(ValueError, match="unknown motif_prompt keys"):
        MotifPromptConfig.from_mapping({"stage": "one", "base_checkpoint": "b.pt", "tokens": 4})
    with pytest.raises(ValueError, match="motif_prompt.stage"):
        MotifPromptConfig(stage="three", base_checkpoint="b.pt")
    with pytest.raises(ValueError, match="requires motif_prompt.base_checkpoint"):
        MotifPromptConfig(stage="one")
    with pytest.raises(ValueError, match="requires motif_prompt.bundle_checkpoint"):
        MotifPromptConfig(stage="two", base_checkpoint="b.pt")
    with pytest.raises(ValueError, match="motif_prompt.fields"):
        MotifPromptConfig(stage="one", base_checkpoint="b.pt", fields=("topo_self", "topo_self"))
    with pytest.raises(ValueError, match="motif_prompt.fields"):
        MotifPromptConfig(stage="one", base_checkpoint="b.pt", fields=())
    with pytest.raises(ValueError, match="motif_prompt.families"):
        MotifPromptConfig(stage="one", base_checkpoint="b.pt", families=())
    with pytest.raises(ValueError, match="motif_prompt.gate_mode"):
        MotifPromptConfig(stage="one", base_checkpoint="b.pt", gate_mode="hard_topk")
    with pytest.raises(ValueError, match="motif_prompt.count_features"):
        MotifPromptConfig(stage="one", base_checkpoint="b.pt", count_features="degrees")
    with pytest.raises(ValueError, match="reader.dim"):
        MotifPromptConfig(stage="one", base_checkpoint="b.pt", reader=ReaderConfig(dim=97, heads=4))
    with pytest.raises(ValueError, match="reader.rrwp_k"):
        MotifPromptConfig(stage="one", base_checkpoint="b.pt", reader=ReaderConfig(rrwp_k=0))
    with pytest.raises(ValueError, match="motif_prompt.slot_read"):
        MotifPromptConfig(stage="one", base_checkpoint="b.pt", slot_read="residual")
    with pytest.raises(ValueError, match="motif_prompt.head_output_init_std"):
        MotifPromptConfig(stage="one", base_checkpoint="b.pt", head_output_init_std=0.0)


def test_gate_modes_are_the_three_spec_controls() -> None:
    assert GATE_MODES == ("learned", "per_type", "mean_graph")


def _weights(n: int = 4, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(n, 96, generator=gen)


def test_count_head_is_exactly_swap_invariant() -> None:
    torch.manual_seed(0)
    head = MotifCountHead(width=16)
    weights = _weights()
    swapped = weights[:, list(SWAP_PERM)]
    torch.testing.assert_close(head(weights), head(swapped), rtol=0, atol=1e-6)


def test_count_head_runs_in_fp32_under_autocast_and_stays_finite_at_zero() -> None:
    head = MotifCountHead(width=16)
    zero = torch.zeros(3, 96, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = head(zero)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert zero.grad is not None and torch.isfinite(zero.grad).all()


def test_count_head_under_autocast_is_bit_identical_to_the_autocast_free_token() -> None:
    # The trailing `.float()` would make an unguarded token fp32 too, so the
    # dtype alone proves nothing: the projection has to run with autocast
    # disabled or the token carries bf16 precision (spec section 5.1).
    torch.manual_seed(0)
    head = MotifCountHead(width=16)
    weights = _weights(seed=4)
    reference = head(weights)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        guarded = head(weights)
        naive = head.proj(torch.zeros(weights.size(0), 4))
    assert torch.equal(guarded, reference)
    assert naive.dtype == torch.bfloat16


def test_degree_only_count_head_carries_the_symmetric_degree_pair_alone() -> None:
    # The section 8 degree-only control (spec v9): the prompt carries only
    # [log1p(deg_u) + log1p(deg_v), |log1p(deg_u) - log1p(deg_v)|] from the
    # *predicted* adjacency, with the two mass entries zeroed inside topo_cnt.
    torch.manual_seed(0)
    head = MotifCountHead(width=16, degree_only=True)
    weights = _weights(seed=7)
    features = torch.zeros(weights.size(0), 4)
    stats = count_statistics(weights)
    log_u, log_v = torch.log1p(stats["deg_u"]), torch.log1p(stats["deg_v"])
    features[:, 2] = log_u + log_v
    features[:, 3] = (log_u - log_v).abs()
    torch.testing.assert_close(head(weights), head.proj(features), rtol=0, atol=1e-6)
    assert torch.equal(head(weights), head(weights[:, list(SWAP_PERM)]))
    # Degree still reaches the generator; the masses no longer do.
    live = weights.clone().requires_grad_(True)
    head(live).sum().backward()
    assert live.grad is not None and torch.isfinite(live.grad).all()
    assert float(live.grad.abs().sum()) > 0.0


def test_dense_adjacency_is_symmetric_with_a_zero_diagonal_and_no_uv_entry() -> None:
    adj = dense_adjacency(_weights())
    assert adj.shape == (4, 26, 26)
    torch.testing.assert_close(adj, adj.transpose(1, 2))
    assert float(torch.diagonal(adj, dim1=1, dim2=2).abs().sum()) == 0.0
    assert float(adj[:, 0, 1].abs().sum()) == 0.0
    assert float(adj[:, 1, 0].abs().sum()) == 0.0


def test_typed_adjacency_splits_the_three_edge_types_and_sums_to_the_untyped_one() -> None:
    typed = typed_adjacency(_weights())
    assert typed.shape == (4, 26, 26, 3)
    torch.testing.assert_close(typed.sum(dim=-1), dense_adjacency(_weights()))


def test_rrwp_arithmetic_returns_fp32_inside_an_active_autocast_context() -> None:
    adj = dense_adjacency(_weights()).to(torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        stack = motif_rrwp(adj, k=4)
    assert stack.dtype == torch.float32
    assert stack.shape == (4, 26, 26, 4)
    torch.testing.assert_close(stack[..., 0], torch.eye(26).expand(4, 26, 26), rtol=0, atol=0)


def test_rrwp_under_autocast_is_bit_identical_to_the_autocast_free_arithmetic() -> None:
    # An fp32 *input* is not enough: `torch.bmm` returns bf16 from fp32 operands
    # inside an active autocast context, `torch.stack` then promotes the mixed
    # list back to fp32, and the walk powers are silently quantised. Only the
    # disabled-autocast context reproduces the fp32 arithmetic (spec section 5.2).
    adj = dense_adjacency(_weights(seed=11))
    reference = motif_rrwp(adj, k=4)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        guarded = motif_rrwp(adj, k=4)
        naive = dense_rrwp(adj.float(), 4).float()
    assert torch.equal(guarded, reference)
    assert not torch.equal(naive, reference)
    assert float((naive - reference).abs().max()) > 1e-4


def test_rrwp_is_finite_at_a_zero_adjacency_and_keeps_the_gradient_connection() -> None:
    weights = torch.zeros(2, 96, requires_grad=True)
    stack = motif_rrwp(dense_adjacency(weights), k=4)
    assert torch.isfinite(stack).all()
    stack.sum().backward()  # type: ignore[no-untyped-call]
    assert weights.grad is not None and torch.isfinite(weights.grad).all()


def _reader(seed: int = 0) -> MotifGritReader:
    torch.manual_seed(seed)
    return MotifGritReader(ReaderConfig(layers=2, dim=16, heads=4, rrwp_k=4), width=16).eval()


def test_reader_emits_three_tokens_and_a_swap_exchanges_only_the_endpoints() -> None:
    reader = _reader()
    weights = _weights()
    out = reader(weights)
    assert set(out) == {"topo_u", "topo_v", "topo_rel"}
    assert out["topo_u"].shape == (4, 16)
    swapped = reader(weights[:, list(SWAP_PERM)])
    torch.testing.assert_close(swapped["topo_u"], out["topo_v"], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(swapped["topo_v"], out["topo_u"], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(swapped["topo_rel"], out["topo_rel"], rtol=1e-5, atol=1e-5)


def test_reader_is_invariant_to_a_within_role_slot_permutation() -> None:
    reader = _reader()
    weights = _weights()
    perm = torch.as_tensor(
        role_permutation(
            (3, 1, 0, 2, 5, 4, 7, 6), (1, 2, 3, 4, 5, 6, 7, 0), (7, 0, 1, 2, 3, 4, 5, 6)
        )
    )
    shuffled = torch.zeros_like(weights)
    shuffled[:, perm] = weights
    out, permuted = reader(weights), reader(shuffled)
    for key in ("topo_u", "topo_v", "topo_rel"):
        torch.testing.assert_close(permuted[key], out[key], rtol=1e-4, atol=1e-4)


def test_reader_predictions_are_batch_independent() -> None:
    reader = _reader()
    weights = _weights(n=6, seed=3)
    whole = reader(weights)
    halves = {key: torch.cat([reader(weights[:2])[key], reader(weights[2:])[key]]) for key in whole}
    for key, value in whole.items():
        torch.testing.assert_close(halves[key], value, rtol=1e-5, atol=1e-5)


def test_final_layer_edge_parameters_receive_gradient() -> None:
    torch.manual_seed(1)
    reader = MotifGritReader(ReaderConfig(layers=2, dim=16, heads=4, rrwp_k=4), width=16)
    out = reader(_weights())
    (out["topo_rel"].sum() + out["topo_u"].sum()).backward()
    params = dict(reader.named_parameters())
    o_e = params[f"layers.{len(reader.layers) - 1}.O_e.weight"]
    assert o_e.grad is not None and float(o_e.grad.abs().sum()) > 0.0
    assert reader.pair_proj.weight.grad is not None


def _generator(seed: int = 0, **extra: object) -> MotifGenerator:
    torch.manual_seed(seed)
    cfg = MotifPromptConfig.from_mapping(
        {"stage": "two", "base_checkpoint": "b.pt", "bundle_checkpoint": "s.pt", **extra}
    )
    return MotifGenerator(d_model=32, cfg=cfg).eval()


def _states(n: int = 4, length: int = 7, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, length, 32, generator=gen), torch.full((n,), length, dtype=torch.long)


def test_generator_emits_96_weights_in_the_unit_interval() -> None:
    generator = _generator()
    h_u, len_u = _states(seed=1)
    h_v, len_v = _states(seed=2)
    weights = generator(h_u, h_v, len_u, len_v).detach()
    assert weights.shape == (4, 96)
    assert float(weights.min()) >= 0.0 and float(weights.max()) <= 1.0


@pytest.mark.parametrize("slot_read", ["bare", "residual_block"])
def test_generator_is_equivariant_under_swapping_the_endpoints(slot_read: str) -> None:
    generator = _generator(slot_read=slot_read)
    h_u, len_u = _states(seed=1)
    h_v, len_v = _states(seed=2)
    forward = generator(h_u, h_v, len_u, len_v)
    reverse = generator(h_v, h_u, len_v, len_u)
    torch.testing.assert_close(reverse, forward[:, list(SWAP_PERM)], rtol=1e-5, atol=1e-5)


def test_the_residual_read_block_keeps_the_queries_on_the_path() -> None:
    # The bare read returns MHA(q, H, H): with the queries zeroed every slot is
    # the same mean of V. The residual block adds the query back, so the slots
    # stay distinct whatever the attention does.
    h_u, len_u = _states(seed=1)
    h_v, len_v = _states(seed=2)
    for slot_read, distinct in (("bare", False), ("residual_block", True)):
        generator = _generator(slot_read=slot_read)
        with torch.no_grad():
            # Kill the query projection: every slot then attends uniformly and
            # the attention hands back the same mean of V eight times over.
            generator.attention.in_proj_weight[:96].zero_()
            generator.attention.in_proj_bias[:96].zero_()
            generator.witness_queries.copy_(
                torch.arange(8, dtype=torch.float32).unsqueeze(1).expand(8, 96).contiguous()
            )
            states = generator._slot_states(h_u, h_v, len_u, len_v)
            witness = generator._read(
                generator.witness_queries,
                generator.residue_proj(h_u),
                None,
                generator.witness_read,
            )
        spread = float((witness - witness.mean(dim=1, keepdim=True)).abs().max())
        assert states.shape == (4, 26, 96)
        assert (spread > 1e-3) is distinct


def test_the_residual_block_queries_start_at_unit_scale() -> None:
    bare = _generator(slot_read="bare")
    residual = _generator(slot_read="residual_block")
    assert float(bare.witness_queries.detach().std()) < 0.05
    assert 0.5 < float(residual.witness_queries.detach().std()) < 2.0
    assert bare.bridge_read is None and bare.witness_read is None
    assert residual.bridge_read is not None and residual.witness_read is not None


def test_the_head_output_init_std_scales_the_last_gate_layer() -> None:
    small = _generator(head_output_init_std=1e-3)
    large = _generator(head_output_init_std=1e-2)

    def _output_std(head: torch.nn.Module) -> float:
        linear = cast(torch.nn.Linear, cast(torch.nn.Sequential, head)[-1])
        return float(linear.weight.detach().std())

    for head in small.heads:
        assert _output_std(head) < 3e-3
    for head in large.heads:
        assert 3e-3 < _output_std(head) < 3e-2


def test_the_head_features_promote_to_fp32_before_the_sum_and_the_difference() -> None:
    # Under autocast the slot states arrive in bf16. The wave-1/2 order formed
    # h_i + h_j and |h_i - h_j| in bf16 and cast afterwards, which rounded a
    # slot difference below the bf16 ulp of the sum to exactly zero.
    big = torch.tensor(300.0, dtype=torch.bfloat16)
    tiny = torch.tensor(0.5, dtype=torch.bfloat16)
    late = ((big + tiny).float(), (big - tiny).abs().float())
    early = (big.float() + tiny.float(), (big.float() - tiny.float()).abs())
    assert float(late[0]) == float(late[1])
    assert float(early[0]) != float(early[1])
    generator = _generator()
    h_u, len_u = _states(seed=1)
    h_v, len_v = _states(seed=2)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        weights = generator(h_u, h_v, len_u, len_v)
    assert weights.dtype == torch.float32 and torch.isfinite(weights).all()


def test_output_biases_start_at_the_clipped_training_mean_in_logit_space() -> None:
    generator = _generator()
    mean = torch.full((96,), 0.001)
    mean[16:32] = 0.4
    generator.init_biases(_statistics(mean), closure_bias_init="density")
    h_u, len_u = _states(seed=1)
    h_v, len_v = _states(seed=2)
    weights = generator(h_u, h_v, len_u, len_v).detach()
    # Clipped to [0.01, 0.99] before the logit, so nothing saturates.
    assert float(weights[:, :16].mean()) < 0.2
    assert 0.2 < float(weights[:, 16:32].mean()) < 0.8


def _closure_corpus(rows: int, positive: int, weight: float) -> torch.Tensor:
    """A corpus where ``positive`` of ``rows`` carry one wedge of magnitude ``weight``."""
    corpus = torch.zeros(rows, 96)
    corpus[:positive, 0] = weight
    corpus[:positive, 8] = weight
    return corpus


def test_the_closure_bias_uses_the_non_zero_mean_not_the_density() -> None:
    # Wave 1: 58% of rows have no closure edge, so logit(per-edge density) put the
    # gates at z = -3.97 and they never moved. The magnitude is the fix (F1).
    corpus = _closure_corpus(rows=100, positive=42, weight=0.196)
    stats = template_statistics(corpus.numpy())
    density = float(stats.mean[:16].mean())
    assert density < 0.02 < stats.nonzero_mean_by_type[0]

    generator = _generator()
    generator.init_biases(stats, closure_bias_init="density")
    wave_one = dict(generator.bias_init_record["closure"])
    generator.init_biases(stats, closure_bias_init="nonzero_mean")
    wave_two = dict(generator.bias_init_record["closure"])

    assert wave_one["rule"] == "density"
    assert wave_one["weight"] == pytest.approx(density, rel=1e-5)
    assert cast(float, wave_one["bias_logit"]) < -3.5
    # One wedge per positive row: 8 w^2 = 0.307 against 2 * m_C^+ = 0.077, so the
    # mass check lowers the magnitude to sqrt(m_C^+ / 8).
    assert wave_two["rule"] == "nonzero_mean_mass_checked"
    assert wave_two["weight"] == pytest.approx(
        (stats.positive_row_wedge_mass / 8.0) ** 0.5, rel=1e-5
    )
    # Unsaturated on both sides of the sigmoid, which is the whole point of F1.
    assert -3.0 < cast(float, wave_two["bias_logit"]) < -1.0


def test_the_mass_check_leaves_a_magnitude_the_corpus_can_carry() -> None:
    # Every positive row carries all eight wedges, so 8 w^2 == m_C^+ and the
    # unchecked non-zero mean is already consistent with the corpus.
    corpus = torch.zeros(20, 96)
    corpus[:10, :16] = 0.2
    stats = template_statistics(corpus.numpy())
    generator = _generator()
    generator.init_biases(stats, closure_bias_init="nonzero_mean")
    record = generator.bias_init_record["closure"]
    assert record["rule"] == "nonzero_mean"
    assert record["weight"] == pytest.approx(0.2, rel=1e-5)


def test_only_the_closure_head_follows_the_new_rule_and_the_clip_still_guards() -> None:
    corpus = torch.zeros(4, 96)
    corpus[:, :16] = 0.5  # closure density == magnitude here
    corpus[:2, 16:32] = 0.4  # attach density 0.2, non-zero mean 0.4
    corpus[:, 32:] = 1.0  # interior non-zero mean 1.0 would clip to 0.99
    stats = template_statistics(corpus.numpy())
    generator = _generator()
    generator.init_biases(stats, closure_bias_init="nonzero_mean")
    record = generator.bias_init_record
    assert record["attach"]["rule"] == record["interior"]["rule"] == "density"
    assert record["attach"]["weight"] == pytest.approx(0.2, rel=1e-5)
    # Interior's density is 1.0, which the [0.01, 0.99] clip holds off saturation.
    assert record["interior"]["weight"] == pytest.approx(0.99)
    assert set(record) == {"closure", "attach", "interior"}
    for entry in record.values():
        assert set(entry) == {"rule", "weight", "bias_logit"}
    with pytest.raises(ValueError, match="closure_bias_init"):
        generator.init_biases(stats, closure_bias_init="not_a_rule")


def test_the_gate_logit_statistics_invert_the_emitted_weights() -> None:
    weights = torch.full((2, 96), 0.5)
    weights[:, :16] = torch.sigmoid(torch.tensor(-4.0))
    stats = MotifGenerator.gate_logit_statistics(weights)
    assert stats["gate_logit_closure_mean"] == pytest.approx(-4.0, abs=1e-4)
    assert stats["gate_logit_closure_frac_abs_gt_3"] == pytest.approx(1.0)
    assert stats["gate_logit_attach_mean"] == pytest.approx(0.0, abs=1e-6)
    assert stats["gate_logit_interior_frac_abs_gt_3"] == pytest.approx(0.0)


@pytest.mark.parametrize("slot_read", ["bare", "residual_block"])
def test_the_generator_parameter_groups_cover_every_trainable_parameter_once_per_read(
    slot_read: str,
) -> None:
    generator = _generator(slot_read=slot_read)
    groups = generator.parameter_groups()
    seen = [id(param) for params in groups.values() for param in params]
    assert sorted(seen) == sorted(id(param) for param in generator.parameters())
    assert len(seen) == len(set(seen))
    assert groups["other"] == []


def test_the_generator_parameter_groups_cover_every_trainable_parameter_once() -> None:
    generator = _generator()
    groups = generator.parameter_groups()
    assert set(groups) == set(GENERATOR_GRAD_GROUPS)
    flat = [param for params in groups.values() for param in params]
    assert len({id(param) for param in flat}) == len(flat)
    expected = {id(p) for p in generator.parameters() if p.requires_grad}
    assert {id(param) for param in flat} == expected
    assert not groups["other"]
    for name in ("slot_queries", "attention", "mpnn", "head_closure"):
        assert groups[name]


def test_generator_weights_are_fp32_under_autocast_and_match_the_fp32_heads() -> None:
    # The gate heads run with autocast disabled: a bf16 sigmoid would quantise
    # the weights before the fp32 count and RRWP arithmetic reads them.
    generator = _generator()
    h_u, len_u = _states(seed=1)
    h_v, len_v = _states(seed=2)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        weights = generator(h_u, h_v, len_u, len_v)
        naive = generator.heads[0](torch.zeros(2, 192))
    assert weights.dtype == torch.float32
    assert naive.dtype == torch.bfloat16
    assert torch.isfinite(weights).all()


def test_per_type_and_mean_graph_gate_modes_realise_their_controls() -> None:
    per_type = _generator(gate_mode="per_type")
    h_u, len_u = _states(seed=1)
    h_v, len_v = _states(seed=2)
    weights = per_type(h_u, h_v, len_u, len_v).detach()
    for block in (slice(0, 16), slice(16, 32), slice(32, 96)):
        assert float(weights[:, block].std(dim=1).abs().max()) == pytest.approx(0.0, abs=1e-6)
    mean_graph = _generator(gate_mode="mean_graph")
    mean_graph.init_biases(_statistics(torch.full((96,), 0.3)), closure_bias_init="density")
    fixed = mean_graph(h_u, h_v, len_u, len_v).detach()
    assert float(fixed.std(dim=0).abs().max()) == pytest.approx(0.0, abs=1e-6)
    assert float(fixed.mean()) == pytest.approx(0.3, abs=1e-6)


def test_gradients_reach_every_generator_parameter_on_a_non_degenerate_batch() -> None:
    torch.manual_seed(2)
    cfg = MotifPromptConfig.from_mapping(
        {"stage": "two", "base_checkpoint": "b.pt", "bundle_checkpoint": "s.pt"}
    )
    generator = MotifGenerator(d_model=32, cfg=cfg)
    h_u, len_u = _states(seed=1)
    h_v, len_v = _states(seed=2)
    generator(h_u, h_v, len_u, len_v).sum().backward()
    missing = [name for name, p in generator.named_parameters() if p.grad is None]
    assert missing == []
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for _, p in generator.named_parameters()
    )


def _adapter(fields: tuple[str, ...] = FIELD_ORDER, seed: int = 0) -> MotifPromptAdapter:
    torch.manual_seed(seed)
    cfg = MotifPromptConfig.from_mapping(
        {"stage": "one", "base_checkpoint": "b.pt", "width": 16, "fields": list(fields)}
    )
    return MotifPromptAdapter(d_model=32, n_layers=3, n_heads=4, cfg=cfg)


def _tokens(n: int = 4, width: int = 16, seed: int = 5) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(n, width, generator=gen)
        for name in ("topo_u", "topo_v", "topo_rel", "topo_cnt")
    }


def test_gates_start_at_zero_and_have_one_entry_per_layer_site_head() -> None:
    adapter = _adapter()
    assert adapter.gates.shape == (3, 3, 4)
    assert float(adapter.gates.detach().abs().sum()) == 0.0


def test_the_two_views_swap_only_the_endpoint_fields() -> None:
    adapter = _adapter()
    tokens = _tokens()
    view_u, view_v = adapter.views(tokens)
    assert view_u.shape == (4, 4, 16)
    # `topo_rel` and `topo_cnt` are swap-invariant, so both views carry them
    # unchanged; only the two endpoint fields exchange their contents. The role
    # embedding marks the *slot* (self / partner), never the endpoint identity:
    # prefix rows are order-invariant inside the branch softmax, so a role tied
    # to u/v would make the two views the same set of rows and erase the
    # self/partner distinction the AB/BA symmetrisation exists to carry.
    torch.testing.assert_close(view_u[:, 2], view_v[:, 2])
    torch.testing.assert_close(view_u[:, 3], view_v[:, 3])
    exchanged = dict(tokens)
    exchanged["topo_u"], exchanged["topo_v"] = tokens["topo_v"], tokens["topo_u"]
    swapped_u, swapped_v = adapter.views(exchanged)
    torch.testing.assert_close(swapped_u, view_v)
    torch.testing.assert_close(swapped_v, view_u)


def test_per_field_masking_removes_rows_rather_than_zeroing_values() -> None:
    full = _adapter()
    tokens = _tokens()
    rows = full.prefix(0, full.views(tokens)[0])
    assert rows.shape == (4, 8, 32)
    trimmed = _adapter(fields=("topo_self", "topo_partner", "topo_rel"))
    trimmed.load_state_dict(full.state_dict(), strict=False)
    kept = trimmed.prefix(0, trimmed.views(tokens)[0])
    assert kept.shape == (4, 6, 32)
    torch.testing.assert_close(kept, rows[:, list(trimmed.active_rows)], rtol=1e-6, atol=1e-6)


def test_masking_a_field_renormalises_the_branch_over_the_remaining_rows() -> None:
    torch.manual_seed(3)
    mha = torch.nn.MultiheadAttention(32, 4, batch_first=True)
    query = torch.randn(2, 5, 32)
    prefix = torch.randn(2, 8, 32)
    gate = torch.randn(4)
    zeroed = prefix.clone()
    zeroed[:, 6:] = 0.0
    removed = prefix[:, :6]
    branch_zeroed = prefix_branch(mha, query, zeroed, gate, 1.0)
    branch_removed = prefix_branch(mha, query, removed, gate, 1.0)
    # Zeroing the values leaves the keys in the denominator; removing the rows does not.
    assert not torch.allclose(branch_zeroed, branch_removed, rtol=1e-4, atol=1e-4)


def _model(stage: str = "one", seed: int = 0, **extra: object) -> V3_1MotifPrompt:
    torch.manual_seed(seed)
    block: dict[str, object] = {
        "stage": stage,
        "base_checkpoint": "base.pt",
        "width": 16,
        "slots_per_field": 2,
        "reader": {"layers": 2, "dim": 16, "heads": 4, "rrwp_k": 4},
    }
    if stage == "two":
        block["bundle_checkpoint"] = "bundle.pt"
    block.update(extra)
    model = V3_1MotifPrompt(base=_tiny_base_config(), motif_prompt=block)
    model.install_mean_template(_statistics(torch.full((96,), 0.1)))
    return model


def _open_gates(model: V3_1MotifPrompt, seed: int = 1) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.adapter.gates.copy_(torch.randn(model.adapter.gates.shape, generator=gen))


@pytest.mark.parametrize("stage", ["one", "two"])
def test_zero_gates_reproduce_the_frozen_base_bit_for_bit(stage: str) -> None:
    model = _model(stage).eval().requires_grad_(False)
    batch = _pair_batch()
    batch[TEMPLATE_KEY] = _weights(n=batch["emb_a"].size(0))
    ours = model(batch)["logits"]
    theirs = model.base(_pair_batch())["logits"]
    assert torch.equal(ours, theirs)


def test_gates_off_intervention_reproduces_the_base_with_open_gates() -> None:
    model = _model("one").eval().requires_grad_(False)
    _open_gates(model)
    batch = _pair_batch()
    batch[TEMPLATE_KEY] = _weights(n=batch["emb_a"].size(0))
    assert not torch.equal(model(batch)["logits"], model.base(_pair_batch())["logits"])
    model.intervention = "gates_off"
    assert torch.equal(model(batch)["logits"], model.base(_pair_batch())["logits"])


def test_stage_one_requires_the_template_and_stage_two_never_reads_one() -> None:
    stage_one = _model("one").eval()
    with pytest.raises(ValueError, match="requires batch"):
        stage_one(_pair_batch())
    stage_two = _model("two").eval()
    out = stage_two(_pair_batch())
    assert "logits" in out and "predicted_weights" in out
    assert out["predicted_weights"].shape[-1] == 96


def test_scorer_level_interventions_fail_closed_at_the_model() -> None:
    model = _model("two").eval()
    model.intervention = "shuffle_graph"
    with pytest.raises(ValueError, match="scoring-time substitution"):
        model(_pair_batch())
    model.intervention = "not_an_intervention"
    with pytest.raises(ValueError, match="unknown motif_prompt intervention"):
        model(_pair_batch())


def test_mean_intervention_substitutes_the_published_training_mean_graph() -> None:
    model = _model("two").eval().requires_grad_(False)
    _open_gates(model)
    model.intervention = "mean"
    out = model(_pair_batch())
    torch.testing.assert_close(
        out["predicted_weights"],
        model.mean_template.unsqueeze(0).expand(out["predicted_weights"].size(0), -1),
        rtol=0,
        atol=0,
    )


def test_the_mean_graph_control_installs_the_unclipped_training_mean() -> None:
    # The head biases are clipped to [0.01, 0.99] so no sigmoid starts saturated,
    # but the fixed-graph control has no sigmoid: clipping its graph turns zero
    # and rare edges positive, changes both motif masses, and makes the control
    # disagree with the `mean` intervention over the same buffer.
    model = _model("two", gate_mode="mean_graph").eval().requires_grad_(False)
    mean = torch.full((96,), 0.5)
    mean[:8] = 0.0
    mean[8:16] = 1.0
    model.install_mean_template(_statistics(mean))
    batch = _pair_batch(n=3)
    encoded_a = model.base.encoder(batch["emb_a"], batch["len_a"])
    encoded_b = model.base.encoder(batch["emb_b"], batch["len_b"])

    predicted = model.predict_weights(encoded_a, encoded_b, batch["len_a"], batch["len_b"])

    expected = mean.unsqueeze(0).expand(3, -1)
    torch.testing.assert_close(predicted, expected, rtol=0, atol=0)
    # The same graph the `mean` intervention substitutes, over the same buffer.
    model.intervention = "mean"
    torch.testing.assert_close(model(batch)["predicted_weights"], expected, rtol=0, atol=0)


@pytest.mark.parametrize("stage", ["one", "two"])
def test_the_trunk_reads_fp32_weights_in_training_under_autocast(
    stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Stage I casts the compiled template and Stage II the predicted graph to
    # fp32 before the trunk, the count head and the reader see them; the
    # encoder states themselves arrive in bf16 under the run's autocast.
    model = _model(stage)
    if stage == "two":
        model.initialize_teacher()
    model.train()
    seen: dict[str, torch.dtype] = {}
    original = model.logits_from_encoded

    def spy(
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
        *,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        seen["weights"] = weights.dtype
        seen["encoded"] = encoded_a.dtype
        return original(encoded_a, encoded_b, lengths_a, lengths_b, weights=weights)

    monkeypatch.setattr(model, "logits_from_encoded", spy)
    batch = _pair_batch(n=4)
    batch[TEMPLATE_KEY] = _weights(n=4)
    batch["label"] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(batch)
    assert seen["encoded"] == torch.bfloat16
    assert seen["weights"] == torch.float32
    if stage == "two":
        assert output["predicted_weights"].dtype == torch.float32


def test_stage_one_corruption_leaves_self_rows_empty() -> None:
    # Spec section 3: self rows carry an explicit empty template and corruption is
    # nonself-only. Corrupting one gives it nonzero training-mean topology in
    # training while evaluation keeps it empty.
    model = _model("one", corruption={"prob": 1.0, "lambda_min": 1.0, "lambda_max": 1.0})
    model.install_mean_template(_statistics(torch.full((96,), 0.7)))
    model.train()
    batch = _pair_batch(n=4)
    batch[TEMPLATE_KEY] = torch.zeros(4, 96)
    batch["motif_mask"] = torch.tensor([1.0, 0.0, 1.0, 1.0])
    encoded_a = model.base.encoder(batch["emb_a"], batch["len_a"])
    encoded_b = model.base.encoder(batch["emb_b"], batch["len_b"])

    read = model.resolve_weights(batch, encoded_a, encoded_b, batch["len_a"], batch["len_b"])

    assert float(read[1].abs().sum()) == 0.0
    assert float(read[0].abs().sum()) > 0.0
    assert float(read[2].abs().sum()) > 0.0


def test_an_inactive_family_is_masked_out_of_the_supervision_targets() -> None:
    # Closure-only Stage II may not emit a bridge edge, so a bridge edge in the
    # compiled target is unavoidable error in `L_slot` and topology the teacher
    # reads but the generator is forbidden to produce.
    model = _model("two", families=["closure"])
    model.initialize_teacher()
    _open_gates(model)
    batch = _pair_batch(n=3)
    batch[TEMPLATE_KEY] = torch.rand(3, 96, generator=torch.Generator().manual_seed(2))
    gated = dict(batch)
    gated[TEMPLATE_KEY] = batch[TEMPLATE_KEY] * model.family_mask

    full = {key: value.detach() for key, value in model(batch).items()}
    masked = {key: value.detach() for key, value in model(gated).items()}

    torch.testing.assert_close(full["slot_loss_rows"], masked["slot_loss_rows"], rtol=0, atol=0)
    torch.testing.assert_close(full["topo_loss_rows"], masked["topo_loss_rows"], rtol=0, atol=0)
    # The supervision is still live; the masking is not a way of zeroing it.
    assert float(full["slot_loss_rows"].abs().sum()) > 0.0


def test_self_row_masking_is_confined_to_the_slot_and_topo_terms() -> None:
    model = _model("two")
    model.initialize_teacher()
    _open_gates(model)
    batch = _pair_batch(n=4)
    batch[TEMPLATE_KEY] = _weights(n=4)
    batch["motif_mask"] = torch.tensor([1.0, 0.0, 1.0, 1.0])
    out = {key: value.detach() for key, value in model(batch).items()}
    assert float(out["slot_loss_rows"][1]) == 0.0
    assert float(out["topo_loss_rows"][1]) == 0.0
    assert float(out["slot_loss_rows"][0]) > 0.0
    assert float(out["topo_loss_rows"][0]) > 0.0
    # The masked row still carries task BCE.
    assert float(out["task_loss"]) > 0.0


def test_checkpoint_round_trip_preserves_logits_and_the_mean_template() -> None:
    model = _model("two").eval().requires_grad_(False)
    _open_gates(model)
    batch = _pair_batch()
    before = model(batch)["logits"]
    state = {key: value.clone() for key, value in model.state_dict().items()}
    restored = _model("two", seed=99).eval().requires_grad_(False)
    restored.load_state_dict(state, strict=True)
    torch.testing.assert_close(restored(batch)["logits"], before, rtol=0, atol=0)
    torch.testing.assert_close(restored.mean_template, model.mean_template, rtol=0, atol=0)


def test_interface_parameters_are_registered_trainable_but_gated_by_interface_open() -> None:
    model = _model("two")
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    assert any(name.startswith("generator.") for name in names)
    assert any(name.startswith("reader.") for name in names)
    assert not any(name.startswith("base.") for name in names)
    assert not any(name.startswith("teacher") for name in names)
    groups = model.optimizer_parameter_groups(1e-4, 1e-5, 1e-2)
    assert [group["name"] for group in groups] == ["generator", "interface"]
    assert groups[1]["max_lr"] == 1e-5
    assert model.interface_open is False
    assert _model("one").interface_open is True


def _swapped(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The same pairs with the endpoints exchanged."""
    out = dict(batch)
    out["emb_a"], out["emb_b"] = batch["emb_b"], batch["emb_a"]
    out["len_a"], out["len_b"] = batch["len_b"], batch["len_a"]
    return out


def _padded(batch: dict[str, torch.Tensor], extra: int = 3) -> dict[str, torch.Tensor]:
    """The same pairs with longer padding and unchanged true lengths."""
    out = {key: value.clone() for key, value in batch.items()}
    gen = torch.Generator().manual_seed(5)
    for key in ("emb_a", "emb_b"):
        tokens = batch[key]
        tail = torch.randn(tokens.size(0), extra, tokens.size(2), generator=gen)
        out[key] = torch.cat([tokens, tail], dim=1)
    return out


@pytest.mark.parametrize("token_source", ["graph", "direct"])
def test_the_same_pair_scores_the_same_whatever_its_padding(token_source: str) -> None:
    # Encoder padding masks do not zero the padded output states, so an unmasked
    # pool makes a pair's tokens depend on the batch it was collated with; packed
    # scoring gathers different padding than training does.
    model = _model("two", token_source=token_source).eval().requires_grad_(False)
    _open_gates(model)
    batch = _pair_batch(n=4)
    torch.testing.assert_close(
        model(_padded(batch))["logits"], model(batch)["logits"], rtol=0, atol=1e-5
    )


@pytest.mark.parametrize("token_source", ["graph", "direct"])
def test_the_prompt_scores_u_v_and_v_u_alike(token_source: str) -> None:
    # The AB/BA aggregation cannot repair an order-dependent token: both
    # orientations reuse the tokens computed from the original ordering.
    model = _model("two", token_source=token_source).eval().requires_grad_(False)
    _open_gates(model)
    batch = _pair_batch(n=4)
    torch.testing.assert_close(
        model(_swapped(batch))["logits"], model(batch)["logits"], rtol=0, atol=1e-5
    )


def test_direct_tokens_exchange_the_endpoints_and_fix_the_relation() -> None:
    model = _model("two", token_source="direct").eval().requires_grad_(False)
    batch = _pair_batch(n=4)
    encoded_a = model.base.encoder(batch["emb_a"], batch["len_a"])
    encoded_b = model.base.encoder(batch["emb_b"], batch["len_b"])
    weights = _weights(n=4)
    forward = model.tokens_from_weights(
        weights, encoded_a, encoded_b, batch["len_a"], batch["len_b"]
    )
    swapped = model.tokens_from_weights(
        weights[:, list(SWAP_PERM)], encoded_b, encoded_a, batch["len_b"], batch["len_a"]
    )
    torch.testing.assert_close(swapped["topo_u"], forward["topo_v"], rtol=0, atol=1e-6)
    torch.testing.assert_close(swapped["topo_v"], forward["topo_u"], rtol=0, atol=1e-6)
    torch.testing.assert_close(swapped["topo_rel"], forward["topo_rel"], rtol=0, atol=1e-6)
    torch.testing.assert_close(swapped["topo_cnt"], forward["topo_cnt"], rtol=0, atol=1e-6)
    # The control really is endpoint-conditioned: the two tokens differ.
    assert not torch.allclose(forward["topo_u"], forward["topo_v"])


def _trainable_without_gradient(model: V3_1MotifPrompt, **batch_extra: torch.Tensor) -> list[str]:
    """Names of trainable parameters one forward/backward never reaches.

    Production DDP wraps this model with ``find_unused_parameters=False``, so any
    such parameter aborts the second iteration.
    """
    _open_gates(model)
    batch = _pair_batch(n=4)
    batch[TEMPLATE_KEY] = _weights(n=4)
    batch["label"] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    batch.update(batch_extra)
    output = model(batch)
    (output["loss"] + output["loss_term_slot"] + output["loss_term_topo"]).backward()
    return sorted(
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None
    )


def test_every_trainable_parameter_is_reached_by_one_backward() -> None:
    assert _trainable_without_gradient(_model("two")) == []


@pytest.mark.parametrize(
    "fields",
    [("topo_cnt",), ("topo_self", "topo_partner", "topo_rel"), ("topo_self", "topo_cnt")],
)
def test_a_masked_field_freezes_and_skips_the_module_it_would_have_read(
    fields: tuple[str, ...],
) -> None:
    model = _model("two", fields=list(fields))
    model.initialize_teacher()
    reads_graph = bool({"topo_self", "topo_partner", "topo_rel"} & set(fields))
    counts = "topo_cnt" in fields
    assert model.student_reads_graph is reads_graph and model.student_counts is counts
    assert any(p.requires_grad for p in model.reader.parameters()) is reads_graph
    assert any(p.requires_grad for p in model.count_head.parameters()) is counts
    tokens = model.tokens_from_weights(
        _weights(n=3),
        torch.zeros(3, 5, 32),
        torch.zeros(3, 5, 32),
        torch.full((3,), 5),
        torch.full((3,), 5),
    )
    assert bool(tokens["topo_rel"].abs().sum() > 0) is reads_graph
    assert bool(tokens["topo_cnt"].abs().sum() > 0) is counts
    # The teacher still reads every field from the graph for L_topo.
    teacher = model._tokens(  # noqa: SLF001
        cast(torch.nn.ModuleDict, model.teacher),
        _weights(n=3),
        torch.zeros(3, 5, 32),
        torch.zeros(3, 5, 32),
        torch.full((3,), 5),
        torch.full((3,), 5),
    )
    assert all(bool(teacher[name].abs().sum() > 0) for name in ("topo_rel", "topo_cnt"))
    assert _trainable_without_gradient(model) == []


def test_the_warm_up_is_vacuous_when_no_generator_parameter_trains() -> None:
    # `mean_graph` freezes the whole generator, so holding its only group would
    # waste the warm-up and open it on the cycle's annealing tail.
    assert _model("two", gate_mode="mean_graph").interface_open_at(1) is True
    assert _model("two").interface_open_at(1) is False
    assert _model("two").interface_open_at(3) is True
    assert _model("two", interface_warmup_epochs=None).interface_open_at(15) is False
    assert _model("one").interface_lr_scale == 1.0
    assert _model("two").interface_lr_scale == 0.1


def test_the_direct_token_head_trains_in_the_generator_group() -> None:
    model = _model("two", token_source="direct")
    groups = {
        str(group["name"]): {id(p) for p in cast(list[torch.nn.Parameter], group["params"])}
        for group in model.optimizer_parameter_groups(1e-4, 1e-5, 1e-2)
    }
    head = cast(torch.nn.Module, model.direct_head)
    assert all(id(p) in groups["generator"] for p in head.parameters())
    assert not any(id(p) in groups["interface"] for p in head.parameters())


def _teacher_tokens(
    model: V3_1MotifPrompt, weights: torch.Tensor, batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """The immutable teacher's four token fields for one adjacency."""
    encoded_a = model.base.encoder(batch["emb_a"], batch["len_a"])
    encoded_b = model.base.encoder(batch["emb_b"], batch["len_b"])
    bundle = cast(torch.nn.ModuleDict, model.teacher)
    return model._tokens(  # noqa: SLF001
        bundle, weights, encoded_a, encoded_b, batch["len_a"], batch["len_b"]
    )


def test_the_direct_control_reads_the_teachers_graph_for_every_token_field() -> None:
    # Spec section 8 advertises this control as carrying the *same* `L_topo`.
    # A teacher that routes through a direct head reads the endpoints alone, so
    # `R_T(Ahat)` and `R_T(A*)` coincide on the three GRIT fields and three of
    # the four representation constraints vanish whatever the adjacency.
    model = _model("two", token_source="direct").eval().requires_grad_(False)
    model.initialize_teacher()
    batch = _pair_batch(n=4)
    predicted = _teacher_tokens(model, _weights(n=4, seed=0), batch)
    true = _teacher_tokens(model, _weights(n=4, seed=9), batch)
    for field in ("topo_u", "topo_v", "topo_rel", "topo_cnt"):
        assert not torch.allclose(predicted[field], true[field]), field


def test_the_direct_controls_topo_term_is_not_the_count_field_alone() -> None:
    from src.distill.motif_losses import TOPO_FIELDS, topo_loss_rows

    model = _model("two", token_source="direct").eval().requires_grad_(False)
    model.initialize_teacher()
    batch = _pair_batch(n=4)
    predicted = _teacher_tokens(model, _weights(n=4, seed=0), batch)
    true = _teacher_tokens(model, _weights(n=4, seed=9), batch)
    assert TOPO_FIELDS == ("topo_u", "topo_v", "topo_rel", "topo_cnt")
    counts_only = {
        field: (true[field] if field != "topo_cnt" else predicted[field]) for field in TOPO_FIELDS
    }
    full = topo_loss_rows(predicted, true)
    assert (full > 0.0).all()
    # Every token field carries part of the term, so masking the three graph
    # fields out must change it.
    assert not torch.allclose(full, topo_loss_rows(counts_only, true))


def test_the_direct_token_control_freezes_the_bypassed_reader() -> None:
    # `token_source='direct'` never calls the reader, so leaving it trainable
    # kills the control on the second DDP iteration.
    model = _model("two", token_source="direct")
    assert all(not param.requires_grad for param in model.reader.parameters())
    assert _trainable_without_gradient(model) == []


def test_the_mean_graph_control_freezes_the_bypassed_generator() -> None:
    # `gate_mode='mean_graph'` returns a buffer, so no generator parameter is on
    # the autograd path at all.
    model = _model("two", gate_mode="mean_graph")
    assert model.generator is not None
    assert all(not param.requires_grad for param in model.generator.parameters())
    assert _trainable_without_gradient(model) == []
    # With no trainable generator the optimiser holds the interface group alone.
    groups = model.optimizer_parameter_groups(1e-4, 1e-5, 1e-2)
    assert [group["name"] for group in groups] == ["interface"]


def test_stage_two_opens_only_the_final_reader_block_and_the_output_projections() -> None:
    # Spec section 7.5: epochs 3-15 adapt the final block and output projections
    # of `R_S`, the count head and the adapter; earlier blocks and the role/input
    # embeddings stay frozen for the whole run.
    model = _model("two")
    frozen = {name for name, param in model.named_parameters() if not param.requires_grad}
    assert {name for name in frozen if name.startswith("reader.")} == {
        "reader.role_embed.weight",
        "reader.node_embed.weight",
        "reader.node_embed.bias",
        "reader.edge_embed.weight",
        "reader.edge_embed.bias",
        *{name for name, _ in model.reader.layers[0].named_parameters(prefix="reader.layers.0")},
    }
    names_by_id = {id(param): name for name, param in model.named_parameters()}
    groups = model.optimizer_parameter_groups(1e-4, 1e-5, 1e-2)
    interface = {names_by_id[id(param)] for param in cast(list[object], groups[1]["params"])}
    reader_interface = {name for name in interface if name.startswith("reader.")}
    adaptable = ("reader.layers.1.", "reader.node_proj", "reader.pair_proj", "reader.token_norm")
    assert all(name.startswith(adaptable) for name in reader_interface)
    assert "reader.node_proj.weight" in reader_interface
    assert "reader.pair_proj.weight" in reader_interface
    assert any(name.startswith("reader.layers.1.") for name in reader_interface)


def test_stage_one_trains_the_whole_reader() -> None:
    # Spec section 7.2 trains R, its token projections, the count head, the role
    # embeddings and the adapter; only F and its head are frozen.
    model = _model("one")
    assert all(param.requires_grad for param in model.reader.parameters())
    names_by_id = {id(param): name for name, param in model.named_parameters()}
    groups = model.optimizer_parameter_groups(1e-4, 1e-5, 1e-2)
    interface = {names_by_id[id(param)] for param in cast(list[object], groups[0]["params"])}
    assert {name for name, _ in model.reader.named_parameters(prefix="reader")} <= interface


@pytest.mark.parametrize("stage", ["one", "two"])
def test_an_inactive_family_is_zeroed_in_both_stages(stage: str) -> None:
    model = _model(stage, families=["closure"]).eval().requires_grad_(False)
    batch = _pair_batch()
    batch[TEMPLATE_KEY] = torch.ones(batch["emb_a"].size(0), 96)
    encoded_a = model.base.encoder(batch["emb_a"], batch["len_a"])
    encoded_b = model.base.encoder(batch["emb_b"], batch["len_b"])
    read = model.resolve_weights(batch, encoded_a, encoded_b, batch["len_a"], batch["len_b"])
    assert float(read[:, 16:].abs().sum()) == 0.0
    assert float(read[:, :16].abs().sum()) > 0.0


def _supervised_batch(n: int = 4, seed: int = 7) -> dict[str, torch.Tensor]:
    """A task batch with labels, a compiled template and a nonself mask."""
    batch = _pair_batch(n=n)
    batch[TEMPLATE_KEY] = _weights(n=n, seed=seed)
    batch["motif_mask"] = torch.ones(n)
    batch["label"] = torch.tensor([1.0, 0.0] * (n // 2))
    return batch


def _generator_grads(
    model: V3_1MotifPrompt, batch: dict[str, torch.Tensor], key: str
) -> torch.Tensor:
    """The flat gradient of one forward output term over the generator.

    A term the warm-up detached from every trainable parameter -- ``L_topo``,
    whose teacher bundle is frozen -- has no graph at all and reports zeros.
    """
    generator = model.generator
    assert generator is not None
    output = model(batch)
    term = output[key].sum() if output[key].dim() else output[key]
    params = [p for p in generator.parameters() if p.requires_grad]
    if not term.requires_grad:
        return torch.zeros(sum(param.numel() for param in params))
    grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    return torch.cat(
        [
            (torch.zeros_like(param) if grad is None else grad).reshape(-1)
            for param, grad in zip(params, grads, strict=True)
        ]
    )


def test_the_graph_only_warm_up_cuts_the_generator_off_from_every_other_term() -> None:
    # F3: the task, structural and topo losses are still computed and logged at
    # their joint-mode values; only their gradient path to G is removed.
    joint = _model("two", warmup_losses="joint", beta_c=1.0)
    _open_gates(joint)
    warm = _model("two", warmup_losses="graph_only", beta_c=1.0)
    warm.load_state_dict(joint.state_dict())
    for model in (joint, warm):
        model.initialize_teacher()
        model.train()
    assert not joint.graph_only_warmup and warm.graph_only_warmup

    batch = _supervised_batch()
    joint_out = joint(batch)
    warm_out = warm(batch)
    for key in ("logits", "loss", "slot_loss_rows", "topo_loss_rows", "predicted_weights"):
        torch.testing.assert_close(warm_out[key], joint_out[key], rtol=0, atol=0)

    for key in ("loss", "topo_loss_rows"):
        assert float(_generator_grads(warm, batch, key).abs().max()) == 0.0
        assert float(_generator_grads(joint, batch, key).abs().max()) > 0.0
    # L_slot still reaches G in both modes: it is the warm start's whole objective.
    assert float(_generator_grads(warm, batch, "slot_loss_rows").abs().max()) > 0.0

    # Once the interface opens, the graph-only mode is over and the task term
    # reaches G again through exactly the joint-mode path.
    warm.interface_open = True
    assert not warm.graph_only_warmup
    torch.testing.assert_close(
        _generator_grads(warm, batch, "loss"), _generator_grads(joint, batch, "loss")
    )


def test_a_structural_style_forward_is_detached_in_the_warm_up_too() -> None:
    # The structural stream drives the same forward without a label, so the one
    # detach point covers both streams.
    warm = _model("two", warmup_losses="graph_only")
    warm.initialize_teacher()
    _open_gates(warm)
    warm.train()
    batch = _supervised_batch()
    del batch["label"]
    assert float(_generator_grads(warm, batch, "logits").abs().max()) == 0.0
    assert float(_generator_grads(warm, batch, "slot_loss_rows").abs().max()) > 0.0


def test_the_balanced_w_slot_defaults_to_one_and_survives_a_checkpoint() -> None:
    model = _model("two", w_slot="balanced", w_slot_multiplier=10.0)
    assert model.cfg.w_slot_is_balanced
    assert model.w_slot_value == 1.0
    model.resolve_w_slot(30.0)
    assert model.w_slot_value == 30.0
    restored = _model("two", w_slot="balanced", w_slot_multiplier=10.0)
    restored.load_state_dict(model.state_dict())
    assert restored.w_slot_value == 30.0
    # A numeric w_slot is never balanced at run time.
    numeric = _model("two", w_slot=0.25)
    assert numeric.w_slot_value == 0.25
    with pytest.raises(ValueError, match="not balanced"):
        numeric.resolve_w_slot(1.0)


def test_the_wave_two_config_keys_default_to_the_wave_one_behaviour() -> None:
    cfg = MotifPromptConfig.from_mapping({"stage": "one", "base_checkpoint": "base.pt"})
    assert cfg.beta_c == 0.0
    assert cfg.closure_bias_init == "density"
    assert cfg.warmup_losses == "joint"
    assert cfg.w_slot == 1.0 and not cfg.w_slot_is_balanced
    assert cfg.w_slot_multiplier == 1.0
    assert cfg.balance_probe_rows == 256
    block = {
        "stage": "two",
        "base_checkpoint": "base.pt",
        "bundle_checkpoint": "bundle.pt",
        "beta_c": 1.0,
        "closure_bias_init": "nonzero_mean",
        "warmup_losses": "graph_only",
        "w_slot": "balanced",
        "w_slot_multiplier": 10.0,
        "balance_probe_rows": 64,
    }
    parsed = MotifPromptConfig.from_mapping(block)
    assert MotifPromptConfig.from_mapping(parsed.to_dict()) == parsed
    assert parsed.to_dict()["w_slot"] == "balanced"
    for key, value, message in (
        ("warmup_losses", "later", "warmup_losses"),
        ("closure_bias_init", "mass", "closure_bias_init"),
        ("w_slot", "auto", "w_slot"),
        ("w_slot_multiplier", 0.0, "w_slot_multiplier"),
        ("balance_probe_rows", 0, "balance_probe_rows"),
        ("beta_c", -1.0, "beta_c"),
    ):
        with pytest.raises(ValueError, match=message):
            MotifPromptConfig.from_mapping({**block, key: value})

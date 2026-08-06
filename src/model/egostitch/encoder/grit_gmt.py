"""`GritGmtEncoder` -- official GRIT layers + official PMA readout over an `ImaginedGraph`.

Wave 1 of the oracle-scaffold experiment. Where `TypedMessagePassingEncoder`
(`encoder/ste.py`) is a three-layer typed message-passing stack written for this
project, this encoder is built from **unmodified upstream research code**:

- `src.vendor.grit_official.grit_layer.GritTransformerLayer` -- the official
  GRIT layer (Ma et al., ICML 2023), vendored byte-for-byte apart from a
  recorded five-line dependency shim.
- `src.vendor.set_transformer_official.modules.PMA` -- the official Pooling by
  Multihead Attention block (Lee et al., ICML 2019), vendored with one recorded
  addition: key-padding-mask support, which the reference implementation lacks
  because its benchmarks are fixed-cardinality.

Nothing in either vendored file is edited here, and this module never monkey-
patches them. Everything below is adaptation: turning a dense, batched,
soft-weighted `ImaginedGraph` into the flat PyG-style batch GRIT's layer
expects, and turning the result back into `(B, N, d)` tokens.

Why GRIT
--------

`ste_typed` aggregates over the typed adjacency and nothing else, so two
scaffolds that differ only in *global* structure -- which slot is a hub, how
far apart two slots are, whether the two ego-nets overlap in a triangle or a
path -- can produce very similar tokens. GRIT's relative random-walk positional
encoding (RRWP) puts exactly that information into the *edge* channel of every
attention head, which is the property the E2/G3 headroom analysis points at
(clustering is the largest axis). The PMA readout adds a small set of learned
seed queries so the pair-level summary is a learned attention pool rather than
a mean, which a mask-weighted mean cannot express.

Dense RRWP
----------

Official GRIT precomputes RRWP as a dataset transform on sparse, static graphs
(upstream `grit/transform/rrwp.py`). Our graphs are *generated per pair*
and carry *soft, continuous* edge weights, so there is no dataset to transform
and no sparsity to exploit -- ``N`` is 34. This module therefore computes the
same quantity densely and per batch, mirroring the upstream definition exactly:

    ``P = D^-1 A`` (row-normalized, degree clamped away from zero)
    ``RRWP = stack([I, P, P^2, ..., P^(rrwp_k - 1)])``   -- ``(B, N, N, K)``
    node-level absolute PE = the diagonal of that stack -- ``(B, N, K)``

which is `add_full_rrwp`'s ``pe_list`` with ``add_identity=True`` and its
``abs_pe = pe.diagonal()``, term for term. ``A`` is the *soft* adjacency summed
over relation types and masked to valid nodes; keeping it soft is what leaves
the whole path differentiable back into the generator.

The edge set is the complete ``n_i^2`` grid over each graph's valid nodes,
diagonal included. That matches upstream: every `configs/GRIT/*.yaml` runs
`RRWPLinearEdgeEncoder` with ``pad_to_full_graph=True``, which calls
`add_self_loops` and then `full_edge_index`, i.e. densifies to exactly this set
before the first layer sees it.
"""

from __future__ import annotations

import torch
from torch import nn

from src.model.egostitch.encoder.base import GraphEncoder
from src.model.egostitch.graph import GraphEmbedding, ImaginedGraph
from src.vendor.grit_official import GritCfgNode, GritTransformerLayer
from src.vendor.set_transformer_official import PMA

__all__ = ["GritGmtEncoder"]

_DEG_EPS = 1e-6


def dense_rrwp(adj: torch.Tensor, k: int, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Dense relative-random-walk-probability stack ``[I, P, ..., P^{k-1}]``.

    Mirrors the official GRIT ``add_full_rrwp`` definition (``add_identity=True``,
    upstream ``grit/transform/rrwp.py``) on a dense batched adjacency
    instead of a sparse PyG graph: channel 0 is the identity and channel ``j``
    is the ``j``-th power of the degree-normalized transition matrix.

    Args:
        adj: Shape ``(B, N, N)`` untyped (already collapsed) adjacency.
        k: Number of stacked walk orders, including the identity term.
        mask: Optional shape ``(B, N)`` soft node existence. Padded nodes must
            neither send nor receive random-walk mass, or the walk leaks into
            slots that do not exist.

    Returns:
        Shape ``(B, N, N, k)`` RRWP stack.
    """
    collapsed = adj
    if mask is not None:
        valid = (mask > 0.0).to(dtype=adj.dtype)
        collapsed = collapsed * valid.unsqueeze(2) * valid.unsqueeze(1)
    degree = collapsed.sum(dim=-1, keepdim=True).clamp(min=_DEG_EPS)
    transition = collapsed / degree

    nodes = collapsed.shape[-1]
    identity = torch.eye(nodes, device=adj.device, dtype=adj.dtype)
    powers = [identity.expand(collapsed.shape[0], nodes, nodes)]
    current = transition
    for _ in range(1, k):
        powers.append(current)
        current = torch.bmm(current, transition)
    return torch.stack(powers[:k], dim=-1)


"""Degree clamp for the row normalization. Upstream zeroes ``1/deg`` at
``inf`` instead (`rrwp.py`: ``deg_inv[deg_inv == float('inf')] = 0``); with a
soft adjacency an isolated row's degree is a small positive float rather than
exactly zero, so a clamp is the faithful continuous analogue -- it reproduces
upstream's behaviour on a hard zero row (that row's ``P`` entries all stay
zero, because the numerator is zero too) without introducing a discontinuity at
tiny degrees."""


class _GritBatch:
    """The mutable attribute bag `GritTransformerLayer.forward` reads and writes.

    GRIT's layer is written against a PyG ``Batch``. It uses only a small,
    fully enumerable slice of that API, so a real ``Batch`` is unnecessary --
    and undesirable: constructing one per forward pass would cost a
    `Data`-list collation for a batch whose layout we already know exactly.

    The exact contract, read off `grit_layer.py`:

    - reads ``.x`` ``(n_total, dim)``, ``.edge_index`` ``(2, n_edges)``,
      ``.edge_attr`` ``(n_edges, dim)``, ``.num_nodes``
    - ``.get(name, default)`` for ``"E"``, ``"edge_attr"``, ``"wE"``
    - ``in`` containment for ``"log_deg"`` / ``"deg"`` (`get_log_deg`)
    - reads ``.log_deg`` ``(n_total, 1)``
    - *writes* ``.E``, ``.Q_h``, ``.K_h``, ``.V_h``, ``.wE``, ``.attn``,
      ``.wV``, and finally ``.x`` / ``.edge_attr``

    Plain attribute storage satisfies all of it. Anything the layer writes and
    this module does not read is simply left on the object and discarded with
    it -- notably ``attn``, which upstream keeps for visualization.
    """

    def __init__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        log_deg: torch.Tensor,
    ) -> None:
        """Store the four fields GRIT's layer reads before it writes anything."""
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.log_deg = log_deg
        self.num_nodes = int(x.shape[0])

    def get(self, name: str, default: object = None) -> object:
        """Mapping-style read, matching PyG ``Data.get``."""
        return getattr(self, name, default)

    def __contains__(self, name: str) -> bool:
        """Support `get_log_deg`'s ``"log_deg" in batch`` probe."""
        return hasattr(self, name)


def _grit_layer_cfg() -> GritCfgNode:
    """Build the `cfg` node `GritTransformerLayer.__init__` reads.

    Every value here is upstream's own default; nothing is invented. They are
    written out explicitly rather than left to `cfg.get(...)` defaults so the
    configuration this project runs GRIT under is readable in one place:

    - ``update_e``: True -- the edge channel is updated per layer, which is
      GRIT's whole point (RRWP is refined, not merely injected once).
    - ``bn_momentum`` / ``bn_no_runner``: required attribute reads in
      `GritTransformerLayer.__init__`. They are *unused* here because this
      encoder runs the layer with ``batch_norm=False`` (see
      `GritGmtEncoder.__init__`), but the attribute reads are unconditional
      upstream, so they must exist.
    - ``rezero``: False -- upstream default.
    - ``attn.*``: upstream defaults exactly as `grit_layer.py` spells them,
      including ``clamp=5.`` and ``act="relu"``. ``graphormer_attn`` is left
      unset so the branch referencing the un-vendored
      `MultiHeadAttentionLayerGraphormerSparse` is never taken.
    """
    cfg = GritCfgNode()
    cfg["update_e"] = True
    cfg["bn_momentum"] = 0.1
    cfg["bn_no_runner"] = False
    cfg["rezero"] = False
    cfg["attn"] = {
        "use": True,
        "deg_scaler": True,
        "use_bias": False,
        "clamp": 5.0,
        "act": "relu",
        "edge_enhance": True,
        "sqrt_relu": False,
        "signed_sqrt": False,
        "scaled_attn": False,
        "no_qk": False,
    }
    return cfg


class GritGmtEncoder(GraphEncoder):
    """Dense-RRWP GRIT stack + masked PMA readout, over an `ImaginedGraph`.

    Emits ``tokens`` of shape ``(B, N + seeds, d_model)`` -- the per-node token
    states with the PMA seed tokens appended -- and ``pooled`` of shape
    ``(B, d_model)``, the mean of those seed tokens. Appending the seeds to
    ``tokens`` rather than returning them separately is deliberate: the
    classifier's conditioning cross-attention reads `GraphEmbedding.tokens` and
    nothing else, so the learned graph-level summary has to reach it through
    that channel or not at all.
    """

    def __init__(
        self,
        in_dim: int,
        num_relations: int,
        d_model: int,
        dim: int = 96,
        layers: int = 4,
        *,
        w_rel: float = 0.25,
        rrwp_k: int = 8,
        n_heads: int = 8,
        seeds: int = 4,
    ) -> None:
        """Build the RRWP embeddings, the GRIT stack, and the PMA readout.

        Args:
            in_dim: ``F`` -- node feature width of the graph the generator
                emits (`ImaginedGraph.feature_dim`), read at build time by the
                composite, never hardcoded.
            num_relations: ``R`` -- typed-adjacency relation count
                (`ImaginedGraph.num_relations`).
            d_model: Output token/pooled width; `ClassifierConfig.d_model`.
            dim: GRIT hidden width (`EncoderConfig.dim`).
            layers: GRIT depth (`EncoderConfig.layers`).
            w_rel: Relational auxiliary-loss weight, forwarded to
                `GraphEncoder.__init__`. ``0.0`` omits the relational head.
            rrwp_k: Random-walk length ``K`` (`EncoderConfig.rrwp_k`); the
                stack is ``[I, P, ..., P^(K-1)]``, so ``K`` must be at least 1.
            n_heads: GRIT attention heads (`EncoderConfig.n_heads`). Must
                divide `dim`: upstream sizes each head as ``dim // n_heads``
                and then concatenates, so a non-divisor silently narrows the
                layer.
            seeds: PMA seed count (`EncoderConfig.seeds`).

        Raises:
            ValueError: On a non-positive dimension, or ``dim % n_heads != 0``.
        """
        super().__init__(d_model=d_model, w_rel=w_rel)
        for name, value in (
            ("in_dim", in_dim),
            ("num_relations", num_relations),
            ("d_model", d_model),
            ("dim", dim),
            ("layers", layers),
            ("rrwp_k", rrwp_k),
            ("n_heads", n_heads),
            ("seeds", seeds),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")

        self.in_dim = in_dim
        self.num_relations = num_relations
        self.rrwp_k = rrwp_k
        self.seeds = seeds
        self.out_dim = d_model

        # Node input is the graph's own features concatenated with the RRWP
        # diagonal (upstream's `abs_pe`); edge input is the full RRWP stack
        # concatenated with the typed adjacency, so the relation types survive
        # into the attention's edge channel rather than being flattened away.
        self.node_embed = nn.Linear(in_dim + rrwp_k, dim)
        self.edge_embed = nn.Linear(rrwp_k + num_relations, dim)

        cfg = _grit_layer_cfg()
        # The final layer's *edge* output is never read -- only `batch.x` leaves
        # the stack -- so its edge-output projection `O_e` and edge LayerNorm
        # would receive no gradient at all. Under DDP that is not a wasted
        # parameter, it is a hard error (or a `find_unused_parameters=True` tax
        # on every step). `O_e=False` / `norm_e=False` are official constructor
        # flags that replace both with `nn.Identity`, so the last layer keeps
        # exactly the arithmetic it had on the path that matters (the edge
        # channel still conditions attention through `MultiHeadAttentionLayer
        # GritSparse.E`) and simply stops allocating the two modules whose
        # output is discarded. Verified empirically: with this, every encoder
        # parameter except `rel_head` -- which the trainer's `L_rel` supplies --
        # receives a gradient from the logits path alone.
        self.layers = nn.ModuleList(
            GritTransformerLayer(  # type: ignore[no-untyped-call]
                in_dim=dim,
                out_dim=dim,
                num_heads=n_heads,
                dropout=0.0,
                attn_dropout=0.0,
                # LayerNorm, not upstream's default BatchNorm. This is the one
                # architectural choice here that departs from GRIT's default,
                # and it is forced by design §4's AB/BA invariant: the
                # composite calls this encoder twice per pair batch, once on
                # `graph` and once on `graph.swapped()`. With BatchNorm those
                # two calls would normalize against two different batch
                # statistics (and advance the running estimates twice per
                # step), so the AB and BA representations of the *same* graph
                # would differ for a reason that has nothing to do with
                # direction. LayerNorm is per-token and makes the two passes
                # exactly symmetric. It is also a first-class upstream option,
                # not a modification -- `layer_norm`/`batch_norm` are official
                # constructor flags and GRIT ships configs using each.
                layer_norm=True,
                batch_norm=False,
                residual=True,
                act="relu",
                norm_e=index < layers - 1,
                O_e=index < layers - 1,
                cfg=cfg,
            )
            for index in range(layers)
        )
        self.project = nn.Linear(dim, d_model)
        self.readout = PMA(d_model, 4, seeds)  # type: ignore[no-untyped-call]

    # ------------------------------------------------------------------ RRWP

    def _rrwp(self, adj: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the dense RRWP stack and its diagonal.

        Args:
            adj: Shape ``(B, R, N, N)`` typed soft adjacency.
            mask: Shape ``(B, N)`` soft node existence.

        Returns:
            ``(stack, diagonal)`` of shapes ``(B, N, N, K)`` and ``(B, N, K)``.
        """
        stack = dense_rrwp(adj.sum(dim=1), self.rrwp_k, mask=mask)
        diagonal = torch.diagonal(stack, dim1=1, dim2=2).transpose(1, 2)
        return stack, diagonal

    # ------------------------------------------------------------------ flatten

    @staticmethod
    def _flat_index(
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the flat node ordering and the complete per-graph edge grid.

        Args:
            valid: Shape ``(B, N)`` bool -- which nodes exist.

        Returns:
            ``(batch_row, node_col, edge_index, edge_graph)``:
            ``batch_row``/``node_col`` are the ``(n_total,)`` dense coordinates
            of each flat node, in row-major order; ``edge_index`` is
            ``(2, n_edges)`` over flat node ids, covering every ordered pair
            within a graph *including* self-pairs; ``edge_graph`` is the
            ``(n_edges,)`` graph id of each edge, kept so the edge attributes
            can be gathered from the dense pair tensor without a second pass.
        """
        batch_row, node_col = torch.nonzero(valid, as_tuple=True)
        counts = valid.sum(dim=1)
        offsets = torch.cumsum(counts, dim=0) - counts

        # Ragged but fully vectorized: graphs have different valid-node counts
        # (a scaffold's `pi` is binary for the oracle generator and soft for
        # the learned one), so a per-graph Python loop would issue O(B) tiny
        # kernel launches on every encoder call -- and the composite calls the
        # encoder twice per pair batch, once for AB and once for BA. Instead,
        # each edge's position *within* its own graph is recovered from a
        # global arange minus that graph's edge offset, and the local
        # source/target fall out of a divmod by that graph's node count.
        edge_counts = counts * counts
        edge_offsets = torch.cumsum(edge_counts, dim=0) - edge_counts
        total_edges = int(edge_counts.sum().item())
        if total_edges == 0:
            empty = torch.zeros(0, dtype=torch.long, device=valid.device)
            no_edges = torch.zeros(2, 0, dtype=torch.long, device=valid.device)
            return batch_row, node_col, no_edges, empty
        edge_graph = torch.repeat_interleave(
            torch.arange(valid.shape[0], device=valid.device), edge_counts
        )
        within = torch.arange(total_edges, device=valid.device) - edge_offsets[edge_graph]
        per_graph_nodes = counts[edge_graph]
        base = offsets[edge_graph]
        edge_index = torch.stack(
            (base + within // per_graph_nodes, base + within % per_graph_nodes)
        )
        return batch_row, node_col, edge_index, edge_graph

    # ------------------------------------------------------------------ forward

    def forward(self, graph: ImaginedGraph) -> GraphEmbedding:
        """Encode `graph` with the vendored GRIT stack and PMA readout.

        Args:
            graph: Must carry ``x.shape[-1] == self.in_dim`` and
                ``adj.shape[1] == self.num_relations``. Never reads
                `graph.aux`.

        Returns:
            ``tokens`` ``(B, N + seeds, d_model)`` -- per-node states with the
            PMA seed tokens appended, padded node rows exactly zero -- and
            ``pooled`` ``(B, d_model)``, the seed-token mean.

        Raises:
            ValueError: On a feature-width or relation-count mismatch (fails
                closed, exactly as `TypedMessagePassingEncoder` does).
        """
        if graph.x.ndim != 3 or graph.x.shape[-1] != self.in_dim:
            raise ValueError(
                f"graph feature channel count must be {self.in_dim}, got {tuple(graph.x.shape)}"
            )
        batch, nodes, _ = graph.x.shape
        expected_adj = (batch, self.num_relations, nodes, nodes)
        if graph.adj.shape != expected_adj:
            raise ValueError(
                f"graph adjacency shape must be {expected_adj}, got {tuple(graph.adj.shape)}"
            )

        valid = graph.mask > 0.0
        stack, diagonal = self._rrwp(graph.adj, graph.mask)
        node_input = torch.cat((graph.x, diagonal), dim=-1)
        pair_input = torch.cat((stack, graph.adj.permute(0, 2, 3, 1)), dim=-1)

        batch_row, node_col, edge_index, edge_graph = self._flat_index(valid)
        total = int(batch_row.numel())
        if total == 0:
            # Every graph in the batch is all-padding. There is no flat batch
            # to build and no attention to run; returning zeros keeps the
            # caller's shapes intact and, critically, keeps the PMA softmax
            # away from an all-masked row. Gradient-connected through a
            # zero-weighted sum so this branch cannot silently detach the
            # generator during a degenerate step.
            zero = graph.x.sum() * 0.0
            tokens = (
                torch.zeros(
                    batch,
                    nodes + self.seeds,
                    self.out_dim,
                    device=graph.x.device,
                    dtype=graph.x.dtype,
                )
                + zero
            )
            return GraphEmbedding(tokens=tokens, pooled=tokens[:, nodes:].mean(dim=1))

        flat_x = self.node_embed(node_input[batch_row, node_col])
        edge_attr = self.edge_embed(
            pair_input[edge_graph, node_col[edge_index[0]], node_col[edge_index[1]]]
        )

        # `get_log_deg` prefers a supplied `log_deg`, so supply it rather than
        # letting upstream recompute a degree from `edge_index` -- which, on a
        # padded-to-complete graph, would be the constant n_i for every node
        # and carry no information at all (upstream warns about exactly this).
        # The soft weighted degree is the meaningful quantity here.
        collapsed = graph.adj.sum(dim=1) * valid.unsqueeze(2).to(graph.adj.dtype)
        weighted_degree = (collapsed.sum(dim=-1) * valid.to(graph.adj.dtype))[batch_row, node_col]
        log_deg = torch.log(weighted_degree + 1.0).view(total, 1)

        flat = _GritBatch(flat_x, edge_index, edge_attr, log_deg)
        for layer in self.layers:
            flat = layer(flat)

        node_tokens = torch.zeros(
            batch, nodes, self.out_dim, device=flat.x.device, dtype=flat.x.dtype
        )
        node_tokens = node_tokens.index_put((batch_row, node_col), self.project(flat.x))

        # A graph with zero valid nodes inside an otherwise non-empty batch
        # would give PMA an all-masked key row; the vendored mask shim already
        # falls back to uniform attention there (see its recorded diff), so the
        # result is finite. Passing the true mask keeps every other row exact.
        seed_tokens = self.readout(node_tokens, valid)
        tokens = torch.cat((node_tokens, seed_tokens), dim=1)
        return GraphEmbedding(tokens=tokens, pooled=seed_tokens.mean(dim=1))

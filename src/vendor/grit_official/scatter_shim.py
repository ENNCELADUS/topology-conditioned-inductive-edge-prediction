"""Pure-torch replacements for the three `torch_scatter` entry points GRIT uses.

`torch_scatter` is a compiled CUDA/C++ extension pinned to an exact
``(torch, CUDA)`` build pair. It is **not** installed in this project's
environment and adding it would couple every training image to a second
compiled wheel whose ABI must track PyTorch's. Modern PyTorch already exposes
everything `grit_layer.py` needs (`Tensor.index_add_`, `Tensor.scatter_reduce_`,
`Tensor.scatter_`), so this module supplies the three names as thin, exactly
equivalent wrappers and the vendored layer's import line is redirected here
(recorded shim (a) in `grit_layer.py`'s header).

Scope is deliberately narrow: these implement **only the call patterns
`grit_layer.py` actually issues**, and raise on anything else rather than
silently doing something subtly different from `torch_scatter`. Those patterns
are, exhaustively:

1. ``scatter(msg, index, dim=0, out=batch.wV, reduce='add')``
   -- accumulate into a caller-provided, correctly shaped, zero-filled output.
2. ``scatter(e_t * score, index, dim=0, reduce="add")``
   -- accumulate into a fresh output whose size along `dim` is inferred as
   ``index.max() + 1``.
3. ``scatter_max(src, index, dim=0, dim_size=num_nodes)[0]``
   -- only the values half of the pair is read; `pyg_softmax` uses it as the
   per-group max subtracted for numerical stability.
4. ``scatter_add(out, index, dim=0, dim_size=num_nodes)``
   -- per-group sum, the softmax denominator.

Semantics matched against `torch_scatter`:

- **Broadcasting of `index`.** `torch_scatter` broadcasts a 1-D `index` across
  the trailing dimensions of `src`. All four call sites above pass a 1-D
  `index` (a row of `edge_index`) with a 3-D `src` of shape
  ``(E, num_heads, out_dim)`` (or ``(E, num_heads, 1)``), and `dim=0`. The
  broadcast is reproduced here by `Tensor.expand` after unsqueezing the
  trailing axes.
- **Empty groups.** `scatter_add`/`scatter(reduce='add')` leave empty groups at
  the initial output value (`0` for a fresh output; whatever the caller
  pre-filled for the `out=` form) -- reproduced exactly, since `index_add_`
  touches only the indexed rows. `scatter_max` leaves empty groups at the
  *identity for max*. `torch_scatter.scatter_max` fills them with `0` when it
  allocates its own output; that difference is invisible at the one call site
  (`pyg_softmax` reads ``max_per_group[index]``, so only groups that contain at
  least one element are ever gathered), but it is reproduced anyway with an
  explicit zero fill so the shim is a drop-in and not merely
  "good enough here".
- **Determinism.** `index_add_` on CUDA is non-deterministic in float
  accumulation order, exactly as `torch_scatter`'s add kernel is; neither is
  bitwise reproducible across runs unless
  `torch.use_deterministic_algorithms(True)` is set, which routes `index_add_`
  to a deterministic kernel. This shim therefore does not *worsen* the
  reproducibility contract, and under the deterministic flag it is strictly
  better than the compiled extension (which has no deterministic path).
"""

from __future__ import annotations

import torch


def _broadcast_index(index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
    """Expand a 1-D `index` across `src`'s trailing dims, `torch_scatter` style.

    Args:
        index: Shape ``(src.shape[dim],)`` group id per slice, or an already
            `src`-shaped index tensor (returned unchanged).
        src: The source tensor being scattered.
        dim: The scattered axis.

    Returns:
        An index tensor broadcast to `src`'s shape.

    Raises:
        ValueError: If `index` is neither 1-D nor already `src`-shaped.
    """
    if index.dim() == src.dim():
        return index
    if index.dim() != 1:
        raise ValueError(
            "scatter_shim supports only a 1-D index (or an already src-shaped one); "
            f"got index.dim()={index.dim()} against src.dim()={src.dim()}"
        )
    shape = [1] * src.dim()
    shape[dim] = index.numel()
    return index.view(shape).expand_as(src)


def _output_size(index: torch.Tensor, dim_size: int | None) -> int:
    """Resolve the output extent along the scattered axis."""
    if dim_size is not None:
        return int(dim_size)
    if index.numel() == 0:
        return 0
    return int(index.max().item()) + 1


def scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> torch.Tensor:
    """Sum-reduce `src` slices into groups given by `index`.

    Signature-compatible with ``torch_scatter.scatter_add``.

    Args:
        src: Values to accumulate.
        index: Group id per slice along `dim` (1-D, broadcast across the
            trailing dims) or an already `src`-shaped index tensor.
        dim: The scattered axis. Only ``0`` is exercised by `grit_layer.py`;
            other values work but are untested here.
        out: Optional pre-allocated destination, accumulated **into** (matching
            `torch_scatter`: the caller's existing values are added to, not
            overwritten). `grit_layer.py` always passes a zero tensor.
        dim_size: Output extent along `dim` when `out` is `None`; inferred as
            ``index.max() + 1`` when both are absent.

    Returns:
        The accumulated tensor (`out` itself when it was supplied).
    """
    if out is None:
        shape = list(src.shape)
        shape[dim] = _output_size(index, dim_size)
        out = src.new_zeros(shape)
    if dim == 0 and index.dim() == 1:
        # `index_add_` takes the 1-D index directly. That matters: the hottest
        # call site scatters `(num_edges, num_heads, out_dim)` messages, and
        # materializing a broadcast int64 index of that shape would cost more
        # memory than the messages themselves. `scatter_add_` below is the
        # general fallback, not the path GRIT takes.
        return out.index_add_(0, index, src)
    return out.scatter_add_(dim, _broadcast_index(index, src, dim), src)


def scatter_max(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Max-reduce `src` slices into groups, returning ``(values, argmax)``.

    Signature-compatible with ``torch_scatter.scatter_max``. `grit_layer.py`
    reads only ``[0]``; the argmax half is produced anyway so the shim is a
    true drop-in.

    Empty groups are filled with ``0`` in the values half and ``-1`` in the
    argmax half, matching `torch_scatter`'s freshly allocated output.

    Args:
        src: Values to reduce.
        index: Group id per slice along `dim`.
        dim: The reduced axis.
        out: Optional pre-allocated destination; reduced into, as
            `torch_scatter` does.
        dim_size: Output extent along `dim` when `out` is `None`.

    Returns:
        ``(values, argmax)``.
    """
    expanded = _broadcast_index(index, src, dim)
    size = out.shape[dim] if out is not None else _output_size(index, dim_size)
    shape = list(src.shape)
    shape[dim] = size

    values = src.new_full(shape, float("-inf"))
    values = values.scatter_reduce_(dim, expanded, src, reduce="amax", include_self=True)
    occupied = src.new_zeros(shape, dtype=torch.bool).scatter_(
        dim, expanded, torch.ones_like(src, dtype=torch.bool)
    )
    values = torch.where(occupied, values, torch.zeros_like(values))
    if out is not None:
        values = torch.maximum(out, values)

    # argmax: the *first* source position attaining the group max, matching
    # `torch_scatter`'s tie-break. Positions are scattered with a min-reduce
    # over the flat slice index, restricted to slices that hit the max.
    positions = torch.arange(src.shape[dim], device=src.device)
    positions = _broadcast_index(positions, src, dim).clone()
    sentinel = src.shape[dim]
    hit = src == values.gather(dim, expanded)
    positions = torch.where(hit, positions, torch.full_like(positions, sentinel))
    argmax = torch.full(shape, sentinel, dtype=torch.long, device=src.device)
    argmax = argmax.scatter_reduce_(dim, expanded, positions, reduce="amin", include_self=True)
    argmax = torch.where(argmax == sentinel, torch.full_like(argmax, -1), argmax)
    return values, argmax


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
    reduce: str = "sum",
) -> torch.Tensor:
    """Dispatch to the reduction named by `reduce`.

    Signature-compatible with ``torch_scatter.scatter``. Only the additive
    reductions `grit_layer.py` issues (``"add"``/``"sum"``) are implemented;
    anything else raises rather than quietly falling back, because a silent
    substitution here would change the model's arithmetic without changing any
    recorded shim line.

    Args:
        src: Values to reduce.
        index: Group id per slice along `dim`.
        dim: The reduced axis.
        out: Optional pre-allocated destination.
        dim_size: Output extent along `dim` when `out` is `None`.
        reduce: Reduction name.

    Returns:
        The reduced tensor.

    Raises:
        NotImplementedError: For any reduction other than add/sum.
    """
    if reduce not in ("add", "sum"):
        raise NotImplementedError(
            f"scatter_shim implements only reduce='add'/'sum'; got {reduce!r}. "
            "grit_layer.py issues no other reduction -- if a future vendored "
            "upstream does, implement it explicitly rather than approximating."
        )
    return scatter_add(src, index, dim=dim, out=out, dim_size=dim_size)


__all__ = ["scatter", "scatter_add", "scatter_max"]

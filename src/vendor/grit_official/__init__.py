"""Official GRIT transformer layer, vendored (MIT). See `grit_layer.py`'s header."""

from src.vendor.grit_official.grit_layer import CN as GritCfgNode
from src.vendor.grit_official.grit_layer import (
    GritTransformerLayer,
    MultiHeadAttentionLayerGritSparse,
    get_log_deg,
    pyg_softmax,
)

__all__ = [
    "GritCfgNode",
    "GritTransformerLayer",
    "MultiHeadAttentionLayerGritSparse",
    "get_log_deg",
    "pyg_softmax",
]

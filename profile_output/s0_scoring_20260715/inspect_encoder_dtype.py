"""Inspect the production V3.1 encoder output dtype under CUDA BF16 autocast."""

from pathlib import Path

import torch

from src.data.packed_features import PackedFeatureTable
from src.model.B0 import V3_1
from src.score_universe import _autocast_context, _load_checkpoint


device = torch.device("cuda")
model, _, _ = _load_checkpoint(
    Path("outputs/archive/pre_e2_alignment_20260711/b0_v31/best.pt")
)
assert isinstance(model, V3_1)
model.to(device)
table = PackedFeatureTable.from_pack(Path("outputs/feature_packs/b0_v31_bf16"), device)
tokens, lengths = table.gather_nodes(torch.tensor([0, 1]), boundary=128)
with torch.inference_mode(), _autocast_context(device, "bf16"):
    encoded = model.encoder(tokens, lengths)
print(f"tokens_dtype={tokens.dtype} encoded_dtype={encoded.dtype}")

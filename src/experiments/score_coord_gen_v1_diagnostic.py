"""Explicit, inference-only loader for original Stage II epoch diagnostics.

Example: python -m src.experiments.score_coord_gen_v1_diagnostic score
--checkpoint OLD_EPOCH.pt [normal src.score_universe score options].
Only V_val sources are accepted. Conversion is in memory and never rewrites or
publishes a checkpoint. Training loss settings are intentionally discarded.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch
from torch import nn


def load_v1_diagnostic(path: Path, **kwargs: object) -> tuple[nn.Module, str, str]:
    """Reconstruct the old forward exactly under the current module layout."""
    from src.score_universe import _checkpoint_id, build_model

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("model_family") != "v3_1_coord_gen":
        raise ValueError("legacy diagnostic loader requires v3_1_coord_gen")
    config = dict(payload["model_config"])
    old = dict(config["coord_gen"])
    if "w_task" not in old or "w_kd" not in old:
        raise ValueError("checkpoint does not declare the original Stage II loss config")
    old.pop("w_task")
    old.pop("w_kd")
    old.update(
        endpoint_hidden=old.get("hidden", 512),
        endpoint_dropout=0.0,
        trainable="generator",
        kd_alpha=0.0,
        w_anchor=0.0,
    )
    config["coord_gen"] = old
    state = cast(Mapping[str, torch.Tensor], payload["model_state"])
    converted = {}
    for name, value in state.items():
        if name.startswith("generator.endpoint_head."):
            name = name.replace("generator.endpoint_head.", "generator.endpoint_head.1.", 1)
        converted[name] = value
        if name.startswith("reader."):
            converted["teacher." + name[len("reader.") :]] = value.clone()
    model = build_model("v3_1_coord_gen", config)
    model.load_state_dict(converted, strict=True)
    model.eval().requires_grad_(False)
    return model, "v3_1_coord_gen", _checkpoint_id(state)


def main() -> None:
    """Run the existing scorer with this explicit diagnostic-only loader."""
    import sys

    from src import score_universe

    arguments = sys.argv[1:]
    parsed = score_universe.build_parser().parse_args(arguments)
    if getattr(parsed, "pairs", None) not in ("val_cls", "val_topology"):
        raise ValueError("original-epoch diagnostics allow val_cls or val_topology only")
    score_universe._load_checkpoint = load_v1_diagnostic
    score_universe.main(arguments)


if __name__ == "__main__":
    main()

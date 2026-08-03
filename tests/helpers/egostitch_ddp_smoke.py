"""Two-rank CPU/gloo smoke driver for the EgoStitch E2E step-0 slot guard.

Launched by the integration test as::

    python -m torch.distributed.run --standalone --nproc_per_node=2 \
        tests/helpers/egostitch_ddp_smoke.py --output-dir <dir> [--mode collapsed]

Drives a tiny `EgoStitchModel` through `_enforce_e2e_initial_slot_health` — the
step-0 death guard the two-stage cleanup put in place of the retired
``--ddp-mode init-probe`` — under a real two-rank process group.

Why this and not the whole loop: the guard's raise is an
``accelerator.reduce`` followed by a raise on *every* rank, precisely because
`_validate_epoch` returns ``None`` off the main process and a rank-zero-only
raise is a DDP hang (design 2026-07-29 Sec 4.1). Only a real process group can
tell a correct all-rank raise from a hang, and a single-process unit test
cannot. The rest of `_train_e2e_stability_loop` is not reachable on a toy
bundle: its clip guard aborts a random 8-node fixture at step 2, and making it
survive would mean loosening registered guard thresholds until the run proved
nothing.

``--mode healthy`` (default) asserts the guard admits a freshly initialized
model and returns its telemetry. ``--mode collapsed`` zeroes the imagine
decoder's slot head so every slot embedding is the same bias vector — a model
genuinely born at ``h_pairwise_cosine_mean == 1`` — and asserts that *both*
ranks raise ``training_invalid(initial_slot_collapse)``. A rank that hangs
instead never writes its file and the launching test times out.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch

os.environ.setdefault("ACCELERATE_USE_CPU", "true")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import train_egostitch as te  # noqa: E402
from src.data.artifacts import canonical_pair  # noqa: E402
from src.data.ego_targets import EgoTargetBuilder  # noqa: E402
from src.data.packed_features import (  # noqa: E402
    PACK_FORMAT,
    PackedFeatureManifest,
    PackedNodeRecord,
    PackedShardRecord,
    sha256_file,
    write_packed_manifest,
)
from src.data.pairs import NegativeSampler  # noqa: E402
from src.model.egostitch import EgoStitchConfig  # noqa: E402
from src.model.egostitch.composite import EgoStitchModel  # noqa: E402
from src.model.egostitch.config import E2EConfig  # noqa: E402
from src.model.egostitch.generator import EgoStitchImagineGenerator  # noqa: E402
from src.train_b0 import EvalConfig, ModelConfig  # noqa: E402

_NODES = [f"n{i}" for i in range(8)]
_POSITIVES = [
    ("n0", "n1"),
    ("n0", "n2"),
    ("n1", "n2"),
    ("n3", "n4"),
    ("n0", "n3"),
    ("n5", "n6"),
    ("n2", "n5"),
    ("n4", "n6"),
    ("n1", "n7"),
    ("n7", "n7"),
]
_TOKEN_DIM = 1536
_E2E_TINY_MODEL: dict[str, object] = {
    "generator": {
        "n_ground": 20,
        # The toy bundle carries `feature_stats is None` — it never goes through
        # `assemble_egostitch_data`'s V_fit statistics path — so the registered
        # `zscore_vfit_v1` transform would raise for want of constants. The guard
        # under test reads slot geometry, not standardization.
        "feature_standardization": "row_layernorm",
    },
    "encoder": {
        "dim": 8,
        "layers": 1,
    },
    "classifier": {
        "d_model": 16,
        "encoder_layers": 1,
        "cross_attn_layers": 1,
        "n_heads": 2,
        "n_inj": 1,
        "xattn_heads": 2,
        "p_topo": 0.15,
    },
}


def _write_tiny_token_pack(pack_dir: Path) -> None:
    """Write the minimal raw-token pack `_BatchFactory` consumes.

    Every node gets a distinct sequence length (>= 3, the minimum a real
    `PairCrossAttention` forward needs: one inner token strictly between the
    BOS/EOS positions).
    """
    pack_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    shard_path = pack_dir / "shard-000.bin"
    records: list[PackedNodeRecord] = []
    offset = 0
    with shard_path.open("wb") as handle:
        for i, node in enumerate(_NODES):
            length = 3 + i
            tensor = torch.from_numpy(rng.normal(size=(length, _TOKEN_DIM)).astype(np.float32))
            raw = tensor.to(torch.bfloat16).contiguous().view(torch.uint16)
            handle.write(raw.numpy().tobytes())
            records.append(PackedNodeRecord(node, 0, offset, offset, length))
            offset += length
    manifest = PackedFeatureManifest(
        format=PACK_FORMAT,
        input_dim=_TOKEN_DIM,
        dtype="bfloat16",
        source_metadata_sha256="0" * 64,
        source_index_sha256="0" * 64,
        nodes=tuple(records),
        shards=(
            PackedShardRecord(
                filename="shard-000.bin",
                num_tokens=offset,
                byte_size=shard_path.stat().st_size,
                sha256=sha256_file(shard_path),
            ),
        ),
        pack_workers=1,
        build_seconds=0.0,
    )
    write_packed_manifest(pack_dir, manifest)


def _toy_config(output_dir: Path, pack_dir: Path) -> te.EgoConfig:
    """Build the E2E worker config directly, bypassing the pinned contract.

    `load_config` enforces the pinned training contract (bf16, negative_ratio
    5, the pinned optimizer block); this driver runs on CPU over eight
    synthetic nodes, so the dataclass is constructed here rather than
    round-tripped through YAML.
    """
    return te.EgoConfig(
        model=ModelConfig(family="egostitch_e2e", config=dict(_E2E_TINY_MODEL)),
        data=te.EgoDataConfig(
            root=output_dir / "data",
            strategy="toy",
            negative_ratio=5,
            node_batch=4,
            edge_batch=4,
            f0_cache=output_dir / "f0.pt",
            grounding_cache=output_dir / "grounding.npz",
            expected_missing_features=[],
            pack_dir=pack_dir,
        ),
        optim=te.EgoOptimConfig(
            lr=1e-4,
            weight_decay=0.01,
            epochs=1,
            warmup_steps=500,
            grad_clip=1.0,
        ),
        diagnostics=te.EgoDiagnosticsConfig(
            gradient_probe_interval=50,
            gradient_imbalance_ratio=50.0,
            gradient_imbalance_steps=200,
            probe_s1_abs_mean_max=1000.0,
            selection_auprc_tolerance=0.02,
            topk_fraction=0.01,
        ),
        eval=EvalConfig(patience=2, eval_every=1),
        seed=0,
        output_dir=output_dir / "run",
        mixed_precision="no",
        training=te.EgoStitchTrainingConfig(),
        run_kind="formal",
    )


def _toy_bundle(model_cfg: EgoStitchConfig) -> te.EgoStitchData:
    rng = np.random.default_rng(0)
    f0 = rng.normal(size=(len(_NODES), model_cfg.input_dim)).astype(np.float32)
    node_index = {node: i for i, node in enumerate(_NODES)}
    g = nx.Graph()
    g.add_nodes_from(_NODES)
    g.add_edges_from([(u, v) for u, v in _POSITIVES[:6] if u != v])
    pool = {node: [v for v in _NODES if v != node][: model_cfg.n_ground] for node in _NODES}
    grounding_index = np.array(
        [[node_index[v] for v in pool[node]] for node in _NODES], dtype=np.int64
    )
    builder = EgoTargetBuilder(g, f0, node_index, pool, slots=model_cfg.slots)
    degrees = {node: int(g.degree(node)) if node in g else 0 for node in _NODES}
    sampler = NegativeSampler(
        _NODES, degrees, frozenset(canonical_pair(u, v) for u, v in _POSITIVES)
    )
    return te.EgoStitchData(
        train_nodes=list(_NODES),
        training_positives=[canonical_pair(u, v) for u, v in _POSITIVES[6:]],
        val_pairs=[("n0", "n4"), ("n1", "n2"), ("n2", "n6"), ("n3", "n7")],
        val_labels=np.array([0, 1, 0, 0], dtype=np.int8),
        f0=torch.from_numpy(f0),
        node_index=node_index,
        grounding_index=grounding_index,
        train_pos={node: i for i, node in enumerate(_NODES)},
        target_builder=builder,
        sampler=sampler,
        rho_train=float(g.number_of_edges() / 36.0),
    )


def _collapse_slot_head(model: EgoStitchModel) -> None:
    """Make every generated slot embedding identical, without touching a threshold.

    ``h = head_h(main)`` (`src/model/egostitch/generator/imagine.py`); with a zero weight
    every slot of every node is the same bias vector, so the normalized
    pairwise slot cosine is exactly 1 — a model genuinely born above the
    Sec 14.4.8 trip line rather than a stubbed measurement.
    """
    assert isinstance(model.generator, EgoStitchImagineGenerator)
    with torch.no_grad():
        model.generator.stage1.imagine.head_h.weight.zero_()
        model.generator.stage1.imagine.head_h.bias.fill_(1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("healthy", "collapsed"), default="healthy")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])

    torch.manual_seed(0)
    pack_dir = args.output_dir / "token_pack"
    cfg = _toy_config(args.output_dir, pack_dir)
    model = EgoStitchModel(E2EConfig.from_mapping(cfg.model.config))
    data = _toy_bundle(model.generator_cfg)
    accelerator = te.build_egostitch_ddp_accelerator(cfg.mixed_precision)
    # One writer, then a barrier: every rank opens the same pack directory.
    if accelerator.is_main_process:
        _write_tiny_token_pack(pack_dir)
    accelerator.wait_for_everyone()
    factory = te._BatchFactory(
        cfg,
        model.generator_cfg,
        data,
        node_batch=cfg.data.node_batch,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    if args.mode == "collapsed":
        _collapse_slot_head(model)

    # The packed token store is bf16-only; a real run consumes it under
    # Accelerate's bf16 autocast, reproduced explicitly because this driver
    # runs the CPU accelerator at mixed_precision="no".
    def _guard() -> dict[str, float]:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            return te._enforce_e2e_initial_slot_health(
                model,
                data,
                accelerator,
                edge_batch=cfg.data.edge_batch,
                topk_fraction=cfg.diagnostics.topk_fraction,
                token_table=factory._token_table,
                token_node_index=factory._token_node_index,
            )

    if args.mode == "collapsed":
        raised = ""
        try:
            _guard()
        except RuntimeError as error:
            raised = str(error)
        if raised != "training_invalid(initial_slot_collapse)":
            raise RuntimeError(f"rank {rank} did not raise the named guard: {raised!r}")
        (args.output_dir / f"raised-rank-{rank}.json").write_text(
            json.dumps({"rank": rank, "error": raised}) + "\n"
        )
        accelerator.wait_for_everyone()
        return

    report = _guard()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        (args.output_dir / "smoke_ok.json").write_text(
            json.dumps(
                {
                    "world_size": accelerator.num_processes,
                    "run_kind": cfg.run_kind,
                    "h_pairwise_cosine_mean": report["h_pairwise_cosine_mean"],
                    "validation_rows": len(data.val_pairs),
                }
            )
            + "\n"
        )
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()

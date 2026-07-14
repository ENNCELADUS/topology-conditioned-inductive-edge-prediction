"""Two-rank CPU/gloo smoke driver for the EgoStitch Stage-1 DDP loop.

Launched by the integration test as::

    python -m torch.distributed.run --standalone --nproc_per_node=2 \
        tests/helpers/egostitch_ddp_smoke.py --output-dir <dir>

Drives a tiny `EgoStitchStage1` through `train_egostitch_ddp_loop` on a
synthetic 8-node bundle with a fake s0 cache, then asserts on rank zero that
the emitted runtime profile passes the orchestrator's exact
`_validate_worker_profile` schema and that `write_outputs` produces the pinned
Task-4 payload.
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

import yaml  # type: ignore[import-untyped]  # noqa: E402
from src import train_egostitch as te  # noqa: E402
from src.data.artifacts import canonical_pair  # noqa: E402
from src.data.ego_targets import EgoTargetBuilder  # noqa: E402
from src.data.pairs import NegativeSampler  # noqa: E402
from src.e2_pipeline import _validate_worker_profile  # noqa: E402
from src.model.egostitch import EgoStitchConfig, EgoStitchStage1  # noqa: E402
from src.train_b0 import build_ddp_accelerator  # noqa: E402

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
_TINY_MODEL = {
    "input_dim": 8,
    "d_p": 4,
    "d_z": 4,
    "d_h": 8,
    "slots": 4,
    "m_max": 8,
    "n_ground": 3,
    "decoder_layers": 1,
    "n_heads": 2,
    "gin_hidden": 8,
    "gin_layers": 2,
    "sinkhorn_iters": 3,
}


def _toy_config(output_dir: Path) -> te.EgoConfig:
    prereg = output_dir / "prereg.json"
    prereg.write_text('{"registration_id": "smoke"}\n')
    mapping = {
        "model": {"family": "egostitch", "config": dict(_TINY_MODEL)},
        "data": {
            "root": str(output_dir / "data"),
            "strategy": "toy",
            "train_positives": "e_sup",
            "negative_ratio": 2,
            "partition_seed": 0,
            "msg_fraction": 0.8,
            "node_batch": 4,
            "edge_batch": 4,
            "f0_cache": str(output_dir / "f0.pt"),
            "grounding_cache": str(output_dir / "grounding.npz"),
            "s0_cache": str(output_dir / "s0.npz"),
            "s0_checkpoint_id": "deadbeefcafefeed",
            "expected_missing_features": [],
        },
        "optim": {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "epochs": 2,
            "warmup_steps": 2,
            "grad_clip": 1.0,
            "warmstart_fraction": 0.25,
        },
        "eval": {"patience": 2, "eval_every": 1},
        "seed": 0,
        "output_dir": str(output_dir / "run"),
        "mixed_precision": "no",
        "preregistration": str(prereg),
    }
    config_path = output_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(mapping))
    return te.load_config(config_path)


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
    s0 = te.S0Cache.__new__(te.S0Cache)
    s0._logits = {}
    for i, u in enumerate(_NODES):
        for v in _NODES[i:]:
            s0._logits[canonical_pair(u, v)] = 0.1 * (i + 1)
    return te.EgoStitchData(
        train_nodes=list(_NODES),
        e_sup_positives=[canonical_pair(u, v) for u, v in _POSITIVES[6:]],
        val_pairs=[("n0", "n4"), ("n1", "n2"), ("n2", "n6"), ("n3", "n7")],
        val_labels=np.array([0, 1, 0, 0], dtype=np.int8),
        f0=torch.from_numpy(f0),
        node_index=node_index,
        grounding_index=grounding_index,
        train_pos={node: i for i, node in enumerate(_NODES)},
        target_builder=builder,
        sampler=sampler,
        s0=s0,
        rho_train=float(g.number_of_edges() / 36.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    cfg = _toy_config(args.output_dir)
    model_cfg = EgoStitchConfig.from_mapping(cfg.model.config)
    data = _toy_bundle(model_cfg)
    model = EgoStitchStage1(model_cfg)
    accelerator = build_ddp_accelerator("no")

    result = te.train_egostitch_ddp_loop(
        model, cfg, data, accelerator, node_batch=cfg.data.node_batch
    )
    profile = _validate_worker_profile(
        result.runtime_profile,
        epochs=cfg.optim.epochs,
        world_size=accelerator.num_processes,
        memory_limit_gib=85.0,
    )
    if accelerator.is_main_process:
        te.write_outputs(result, cfg, data)
        payload = torch.load(cfg.output_dir / "best.pt", weights_only=False)
        assert set(payload) == {
            "model_state",
            "model_family",
            "model_config",
            "epoch",
            "val_metrics",
            "seed",
            "config",
        }, sorted(payload)
        assert payload["model_family"] == "egostitch"
        (args.output_dir / "smoke_ok.json").write_text(
            json.dumps(
                {
                    "world_size": accelerator.num_processes,
                    "epochs_completed": profile["epochs_completed"],
                    "best_epoch": result.best_epoch,
                }
            )
            + "\n"
        )
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()

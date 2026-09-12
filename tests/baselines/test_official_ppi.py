from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from src.baselines.official_ppi import OfficialPPI


@pytest.mark.parametrize("method", ["tuna", "ppitrans"])
def test_official_prediction_parity_and_pair_invariance(method: str) -> None:
    torch.manual_seed(0)
    model = OfficialPPI(
        method, input_dim=10, hidden_dim=8, heads=2, layers=1, rffs=16, dropout=0.0
    ).eval()
    a, b = torch.randn(2, 5, 10), torch.randn(2, 7, 10)
    la, lb = torch.tensor([3, 5]), torch.tensor([6, 7])
    with torch.no_grad():
        score = model(a, b, la, lb)
        if method == "tuna":
            logits, var = model.core.forward(a, b, la, lb, 5, 7, True, False)
            expected = logits.flatten() / torch.sqrt(1 + np.pi / 8 * var.flatten())
            precision = model.core.gp_layer.precision.clone()
        else:
            x, lx, y, ly = model.encoder(a, la, b, lb)
            logits = model.decoder(x, lx, y, ly)["logits"]
            expected = logits[:, 1] - logits[:, 0]
        torch.testing.assert_close(score, expected)
        torch.testing.assert_close(score, model(b, a, lb, la))
        torch.testing.assert_close(score[:1], model(a[:1, :3], b[:1, :6], la[:1], lb[:1]))
        if method == "tuna":
            torch.testing.assert_close(precision, model.core.gp_layer.precision)
    model.train()
    model(a, b, la, lb).sum().backward()
    assert any(p.grad is not None and bool(p.grad.norm() > 0) for p in model.parameters())


@pytest.mark.parametrize("method", ["tuna", "ppitrans"])
def test_train_publish_reload_with_heldout_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    import networkx as nx
    import src.train_official_ppi as worker
    from src.eval.checkpoint_selection import SELECTION_RULE, TopologyValidationMetrics
    from src.eval.val_topology import ValTopologyResult
    from src.score_universe import _load_checkpoint

    train = [f"t{i}" for i in range(12)]
    val = [f"v{i}" for i in range(6)]
    positives = frozenset([(train[0], train[1]), (train[2], train[3])])
    graph = nx.Graph()
    graph.add_nodes_from(train)
    graph.add_edges_from(positives)
    val_pairs = [(val[i], val[(i + 1) % 6]) for i in range(6)]
    split = SimpleNamespace(
        train_nodes=frozenset(train),
        v_val=frozenset(val),
        training_positives=positives,
        val_cls_pairs=val_pairs,
        val_cls_labels=[1, 0] * 3,
        build_training_graph=lambda: graph,
        params=SimpleNamespace(split_seed=42),
        region_seeds=("v0",),
    )
    tensors = {node: torch.randn(5, 10) for node in train + val}

    class Features:
        def __init__(self, *args: object) -> None:
            self.position = {node: i for i, node in enumerate(tensors)}

        def batch(self, pairs: list[tuple[str, str]]) -> tuple[torch.Tensor, ...]:
            return (
                torch.stack([tensors[a] for a, b in pairs]),
                torch.stack([tensors[b] for a, b in pairs]),
                torch.full((len(pairs),), 5),
                torch.full((len(pairs),), 5),
            )

    monkeypatch.setattr(worker, "PairFeatures", Features)
    monkeypatch.setattr(worker, "_load_val_region_split", lambda *a: split)
    monkeypatch.setattr(
        worker,
        "val_ball_union_universe",
        lambda s: SimpleNamespace(u_idx=np.arange(6), v_idx=np.roll(np.arange(6), -1)),
    )
    monkeypatch.setattr(
        worker, "build_val_topology_reference", lambda s: SimpleNamespace(nodes=val)
    )
    monkeypatch.setattr(
        worker,
        "val_region_topology_metrics",
        lambda **kw: ValTopologyResult(TopologyValidationMetrics(0.5, 1.0, 1.0, 1.0, 1.0), 0.3),
    )
    cfg = {
        "output_dir": str(tmp_path),
        "data_root": "unused",
        "pack_dir": "unused",
        "seed": 0,
        "topology_every": 1,
        "model": {
            "method": method,
            "input_dim": 10,
            "hidden_dim": 8,
            "heads": 2,
            "layers": 1,
            "rffs": 16,
            "dropout": 0.0,
            "batch_size": 4,
        },
        "optim": {"lr": 0.001, "weight_decay": 0.0, "epochs": 1, "patience": 2},
    }
    worker.run(cfg, 0, 1, torch.device("cpu"))
    payload = torch.load(tmp_path / "best.pt", weights_only=True)
    assert payload["selection_rule"] == SELECTION_RULE
    assert payload["val_threshold_transfer"]["threshold"] == 0.3
    assert (tmp_path / "complete.json").exists()
    loaded, family, _ = _load_checkpoint(tmp_path / "best.pt")
    loaded.eval()
    batch = Features().batch(val_pairs)
    with torch.no_grad():
        assert torch.isfinite(loaded(*batch)).all()
    assert family == "official_ppi"
    if method == "tuna":
        assert float(loaded.core.gp_layer.precision.norm()) > 0

"""Dynamic sampling and complete offline KD coverage share the teacher stream."""

from collections import Counter

import networkx as nx
import numpy as np
import pytest
from src.data.pairs import NegativeSampler
from src.data.training_sampler import build_training_corpus, enumerate_edge_stream
from src.train_egostitch import enumerate_edge_stream as teacher_stream

pytestmark = pytest.mark.unit


def test_teacher_and_student_epochs_share_sampler_and_complete_target_union() -> None:
    nodes = [f"n{i:02d}" for i in range(30)]
    graph = nx.path_graph(nodes)
    forbidden = frozenset(nodes[:5])
    truth = frozenset(graph.edges)
    positives = sorted((u, v) for u, v in truth if not (u in forbidden and v in forbidden))
    training = nx.Graph(positives)
    sampler = NegativeSampler(
        nodes, dict(training.degree()), truth, forbidden_internal_nodes=forbidden
    )
    corpus = build_training_corpus(positives, sampler, negative_ratio=5, seed=7, epochs=3)
    replay = build_training_corpus(
        list(reversed(positives)), sampler, negative_ratio=5, seed=7, epochs=3
    )
    assert teacher_stream is enumerate_edge_stream
    assert corpus.pairs == replay.pairs and corpus.labels == replay.labels
    assert len(corpus.pairs) == len(set(corpus.pairs))
    seen = set()
    negatives_by_epoch = []
    for epoch in (1, 2, 3):
        indices = corpus.epoch_rows[epoch]
        np.testing.assert_array_equal(indices, replay.epoch_rows[epoch])
        observed = [(*corpus.pairs[i], corpus.labels[i]) for i in indices]
        expected = teacher_stream(
            positives, sampler, negative_ratio=5, seed=7, epoch=epoch, rank=0, world_size=1
        )
        assert observed == expected
        assert Counter(label for _, _, label in observed) == {
            1: len(positives),
            0: 5 * len(positives),
        }
        assert len(indices) == len(set(indices.tolist()))
        assert {(u, v) for u, v, label in observed if label} == set(positives)
        negative = {(u, v) for u, v, label in observed if not label}
        assert not negative & truth
        assert all(not (u in forbidden and v in forbidden) for u, v in negative)
        assert any((u in forbidden) != (v in forbidden) for u, v in negative)
        negatives_by_epoch.append(negative)
        seen.update(indices.tolist())
    assert negatives_by_epoch[0] != negatives_by_epoch[1]
    assert seen == set(range(len(corpus.pairs)))
    np.testing.assert_array_equal(corpus.epoch_rows[0], corpus.epoch_rows[1])

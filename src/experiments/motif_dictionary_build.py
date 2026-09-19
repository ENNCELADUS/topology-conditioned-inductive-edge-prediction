"""Build the shared training-only motif prompt dictionary.

The builder enumerates the full seed-0, 15-epoch dynamic 1:5 corpus, but only
encodes at most 5,000 medoid candidates per nonempty stratum at once.  Epoch-1
normalisation, temperature and mean-route statistics are streamed in bounded
reader batches.  No residue feature pack or PPI forward is used.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from src.data.motif_dictionary import (
    CATEGORY_NAMES,
    FIELD_ORDER,
    DictionaryArtifact,
    classify_template_weights,
    save_dictionary,
)
from src.data.motif_template import SWAP_PERM, MotifTemplateTable
from src.data.partition import build_g_struct
from src.model.egostitch.classifier.motif_prompt import (
    MotifCountHead,
    MotifGritReader,
    MotifPromptConfig,
)

logger = logging.getLogger(__name__)

_QUOTAS = (1, 3, 4, 4, 4)
_CONSTRUCTION_SEED = 42
_MAX_CANDIDATES = 5_000


@dataclass(frozen=True)
class CandidateSet:
    """Bounded real compiled templates eligible to become representatives."""

    pairs: list[tuple[str, str]]
    weights: Tensor
    categories: Tensor


def _swap_weights(weights: Tensor) -> Tensor:
    return weights.index_select(-1, torch.as_tensor(SWAP_PERM, device=weights.device))


def _swap_tokens(tokens: Tensor) -> Tensor:
    return tokens[:, (1, 0, 2, 3)]


def _scaled_distance(left: Tensor, right: Tensor, scales: Tensor) -> Tensor:
    """Pairwise scaled squared distance in the fixed four-field coordinates."""
    return ((left[:, None] - right[None]) / scales[None, None, :, None]).square().mean(
        dim=(-1, -2)
    )


def _pairwise_scaled_distance(tokens: Tensor, scales: Tensor) -> Tensor:
    """Compute an O(N^2) distance matrix without an O(N^2*4*width) temporary."""
    scaled = (tokens / scales[None, :, None]).flatten(1)
    squared = scaled.square().sum(dim=1, keepdim=True)
    distance = squared + squared.T - 2.0 * (scaled @ scaled.T)
    return distance.clamp_min_(0).div_(scaled.shape[1])


def _deduplicate_candidates(
    candidates: CandidateSet, tokens: Tensor
) -> tuple[CandidateSet, Tensor, list[int]]:
    keep: list[int] = []
    seen: set[bytes] = set()
    for index, row in enumerate(candidates.weights.numpy()):
        key = row.tobytes()
        if key not in seen:
            seen.add(key)
            keep.append(index)
    indices = torch.as_tensor(keep, dtype=torch.int64)
    return (
        CandidateSet(
            pairs=[candidates.pairs[i] for i in keep],
            weights=candidates.weights.index_select(0, indices),
            categories=candidates.categories.index_select(0, indices),
        ),
        tokens.index_select(0, indices),
        keep,
    )


def _category_medoids(tokens: Tensor, scales: Tensor, count: int) -> list[int]:
    """Select deterministic real medoids with farthest-first initialisation."""
    if count <= 0 or tokens.shape[0] == 0:
        return []
    count = min(count, tokens.shape[0])
    mean = tokens.mean(dim=0, keepdim=True)
    centers = [int(_scaled_distance(tokens, mean, scales).squeeze(1).argmin())]
    pairwise = _pairwise_scaled_distance(tokens, scales)
    while len(centers) < count:
        nearest = pairwise[:, centers].amin(dim=1)
        nearest[torch.as_tensor(centers)] = -1
        centers.append(int(nearest.argmax()))
    for _ in range(20):
        assignment = pairwise[:, centers].argmin(dim=1)
        updated: list[int] = []
        for cluster, previous in enumerate(centers):
            members = torch.nonzero(assignment == cluster, as_tuple=False).flatten()
            if members.numel() == 0:
                updated.append(previous)
                continue
            cluster_mean = tokens.index_select(0, members).mean(dim=0, keepdim=True)
            member_distance = _scaled_distance(
                tokens.index_select(0, members), cluster_mean, scales
            )
            local = int(member_distance.argmin())
            updated.append(int(members[local]))
        if updated == centers:
            break
        centers = updated
    return centers


def select_representatives(
    candidates: CandidateSet, tokens: Tensor, scales: Tensor
) -> list[int]:
    """Return original candidate indices under the approved 1/3/4/4/4 allocation."""
    candidates, tokens, original_indices = _deduplicate_candidates(candidates, tokens)
    selected: list[int] = []
    for category, quota in enumerate(_QUOTAS):
        members = torch.nonzero(candidates.categories == category, as_tuple=False).flatten()
        if members.numel() == 0:
            continue
        local = _category_medoids(tokens.index_select(0, members), scales, quota)
        selected.extend(int(members[index]) for index in local)

    # Fill under-populated strata from the farthest unused real candidates.
    target = min(sum(_QUOTAS), tokens.shape[0])
    while len(selected) < target:
        unused = [index for index in range(tokens.shape[0]) if index not in selected]
        if not selected:
            selected.append(unused[0])
            continue
        distance = _scaled_distance(tokens[unused], tokens[selected], scales).amin(dim=1)
        selected.append(unused[int(distance.argmax())])
    return [original_indices[index] for index in selected]


def _encode_batches(
    weights: Tensor,
    encoder: Callable[[Tensor], Tensor],
    *,
    batch_size: int,
) -> Tensor:
    chunks: list[Tensor] = []
    for start in range(0, weights.shape[0], batch_size):
        chunks.append(encoder(weights[start : start + batch_size]).detach().cpu().float())
    if not chunks:
        raise ValueError("cannot encode an empty template collection")
    return torch.cat(chunks)


def estimate_scales(
    weights: Tensor, encoder: Callable[[Tensor], Tensor], *, batch_size: int
) -> Tensor:
    """Stream epoch-1 rows and equal-weight swaps into four scalar variances."""
    sums = torch.zeros(4, dtype=torch.float64)
    squares = torch.zeros(4, dtype=torch.float64)
    counts = torch.zeros(4, dtype=torch.float64)
    for start in range(0, weights.shape[0], batch_size):
        tokens = encoder(weights[start : start + batch_size]).detach().cpu().float()
        both = torch.cat((tokens, _swap_tokens(tokens)), dim=0).double()
        for field in range(4):
            values = both[:, field].reshape(-1)
            sums[field] += values.sum()
            squares[field] += values.square().sum()
            counts[field] += values.numel()
        if start == 0 or start + batch_size >= weights.shape[0] or start % (100 * batch_size) == 0:
            logger.info(
                "scale pass encoded %d/%d epoch-1 rows",
                min(start + batch_size, weights.shape[0]),
                weights.shape[0],
            )
    # Endpoint fields share one pooled variance.
    endpoint_sum = sums[0] + sums[1]
    endpoint_square = squares[0] + squares[1]
    endpoint_count = counts[0] + counts[1]
    endpoint_var = endpoint_square / endpoint_count - (endpoint_sum / endpoint_count).square()
    variance = squares / counts - (sums / counts).square()
    variance[:2] = endpoint_var
    return variance.clamp_min(1e-12).sqrt().float()


def _expand_swap_bank(
    weights: Tensor, tokens: Tensor, pairs: list[tuple[str, str]]
) -> tuple[Tensor, Tensor, Tensor, list[tuple[str, str]]]:
    bank_weights: list[Tensor] = []
    bank_tokens: list[Tensor] = []
    bank_pairs: list[tuple[str, str]] = []
    position: dict[bytes, int] = {}
    for row_weights, row_tokens, pair in zip(weights, tokens, pairs, strict=True):
        for value, token, source in (
            (row_weights, row_tokens, pair),
            (_swap_weights(row_weights), _swap_tokens(row_tokens[None])[0], (pair[1], pair[0])),
        ):
            key = value.numpy().tobytes()
            if key not in position:
                position[key] = len(bank_weights)
                bank_weights.append(value)
                bank_tokens.append(token)
                bank_pairs.append(source)
    stacked_weights = torch.stack(bank_weights).float()
    stacked_tokens = torch.stack(bank_tokens).float()
    swap_index = torch.as_tensor(
        [position[_swap_weights(row).numpy().tobytes()] for row in stacked_weights],
        dtype=torch.int64,
    )
    self_swapping = swap_index == torch.arange(swap_index.numel())
    if bool(self_swapping.any()):
        mean_endpoint = stacked_tokens[self_swapping, :2].mean(dim=1, keepdim=True)
        stacked_tokens[self_swapping, :2] = mean_endpoint
    return stacked_weights, stacked_tokens, swap_index, bank_pairs


def _reservoir_candidates(
    table: MotifTemplateTable, pairs: Sequence[tuple[str, str]], *, log_every: int = 25_000
) -> CandidateSet:
    reservoirs: list[list[tuple[str, str, Tensor]]] = [[] for _ in CATEGORY_NAMES]
    seen = [0] * len(CATEGORY_NAMES)
    rng = [np.random.default_rng((_CONSTRUCTION_SEED, category)) for category in range(5)]
    empty_pair: tuple[str, str] | None = None
    ordered_pairs = sorted(set(pairs))
    for offset, pair in enumerate(ordered_pairs, start=1):
        weights = torch.from_numpy(table.compile_row(*pair).weights)
        category = int(classify_template_weights(weights[None])[0])
        seen[category] += 1
        if category == 0:
            empty_pair = pair
            continue
        entry = (pair[0], pair[1], weights)
        bucket = reservoirs[category]
        if len(bucket) < _MAX_CANDIDATES:
            bucket.append(entry)
        else:
            index = int(rng[category].integers(seen[category]))
            if index < _MAX_CANDIDATES:
                bucket[index] = entry
        if offset % log_every == 0:
            logger.info("classified %d/%d deduplicated corpus pairs", offset, len(ordered_pairs))
    if empty_pair is None:
        node = table.nodes[0]
        empty_pair = (node, node)
    empty_weights = torch.from_numpy(table.compile_row(*empty_pair).weights)
    rows = [(empty_pair[0], empty_pair[1], empty_weights), *sum(reservoirs[1:], [])]
    rows.sort(key=lambda item: (item[0], item[1]))
    weights = torch.stack([item[2] for item in rows]).float()
    return CandidateSet(
        pairs=[(item[0], item[1]) for item in rows],
        weights=weights,
        categories=classify_template_weights(weights),
    )


def _load_stage1_encoder(
    checkpoint: Path, device: torch.device
) -> tuple[Callable[[Tensor], Tensor], dict[str, object]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("model_family") != "v3_1_motif_prompt":
        raise ValueError(f"{checkpoint}: expected a v3_1_motif_prompt checkpoint")
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict) or not isinstance(model_config.get("motif_prompt"), dict):
        raise ValueError(f"{checkpoint}: missing embedded motif_prompt model config")
    cfg = MotifPromptConfig.from_mapping(cast(dict[str, object], model_config["motif_prompt"]))
    if cfg.stage != "one":
        raise ValueError(
            f"{checkpoint}: dictionary graph tokens require a Stage I motif checkpoint, "
            f"got stage {cfg.stage!r}"
        )
    reader = MotifGritReader(cfg.reader, cfg.width)
    count_head = MotifCountHead(cfg.width, degree_only=cfg.count_features == "degree")
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise ValueError(f"{checkpoint}: missing model_state")

    def component_state(prefix: str) -> dict[str, Tensor]:
        result = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not result:
            raise ValueError(f"{checkpoint}: Stage I checkpoint has no {prefix} tensors")
        return result

    reader.load_state_dict(component_state("reader."), strict=True)
    count_head.load_state_dict(component_state("count_head."), strict=True)
    class TokenEncoder(nn.Module):
        def __init__(self, graph_reader: nn.Module, graph_count_head: nn.Module) -> None:
            super().__init__()
            self.reader = graph_reader
            self.count_head = graph_count_head

        def forward(self, weights: Tensor) -> Tensor:
            graph = cast(dict[str, Tensor], self.reader(weights))
            return torch.stack(
                [
                    graph["topo_u"],
                    graph["topo_v"],
                    graph["topo_rel"],
                    self.count_head(weights),
                ],
                dim=1,
            ).float()

    module: nn.Module = TokenEncoder(reader, count_head).to(device).eval()
    module.requires_grad_(False)
    if device.type == "cuda" and device.index is None and torch.cuda.device_count() > 1:
        module = nn.DataParallel(module)
        logger.info("graph-token encoder using %d visible CUDA devices", torch.cuda.device_count())

    @torch.inference_mode()
    def encode(weights: Tensor) -> Tensor:
        value = weights.to(device=device, dtype=torch.float32)
        return cast(Tensor, module(value)).float()

    return encode, {"width": cfg.width, "field_order": list(FIELD_ORDER)}


def build_dictionary(
    *,
    table: MotifTemplateTable,
    corpus_pairs: Sequence[tuple[str, str]],
    epoch1_pairs: Sequence[tuple[str, str]],
    encoder: Callable[[Tensor], Tensor],
    batch_size: int,
    metadata: dict[str, object],
) -> DictionaryArtifact:
    """Construct all fixed dictionary constants from legal training rows."""
    candidates = _reservoir_candidates(table, corpus_pairs)
    logger.info("encoding %d bounded medoid candidates", candidates.weights.shape[0])
    epoch1_weights = torch.from_numpy(table.weights(epoch1_pairs)).float()
    nonself = torch.as_tensor([u != v for u, v in epoch1_pairs])
    stats_weights = epoch1_weights[nonself]
    if stats_weights.shape[0] == 0:
        raise ValueError("epoch-1 corpus has no nonself rows for dictionary statistics")
    scales = estimate_scales(stats_weights, encoder, batch_size=batch_size)
    candidate_tokens = _encode_batches(candidates.weights, encoder, batch_size=batch_size)
    candidates, candidate_tokens, _ = _deduplicate_candidates(candidates, candidate_tokens)
    selected = select_representatives(candidates, candidate_tokens, scales)
    index = torch.as_tensor(selected, dtype=torch.int64)
    real_weights = candidates.weights.index_select(0, index)
    real_tokens = candidate_tokens.index_select(0, index)
    real_pairs = [candidates.pairs[i] for i in selected]
    bank_weights, bank_tokens, swap_index, source_pairs = _expand_swap_bank(
        real_weights, real_tokens, real_pairs
    )
    direct = _encode_batches(bank_weights, encoder, batch_size=batch_size)
    if not torch.allclose(bank_tokens, direct, rtol=2e-4, atol=2e-5):
        maximum = float((bank_tokens - direct).abs().max())
        raise ValueError(
            f"Stage I reader failed endpoint-swap equivariance check (max={maximum:.3g})"
        )

    nearest_positive: list[Tensor] = []
    stats_nonempty = classify_template_weights(stats_weights) != 0
    for start in range(0, stats_weights.shape[0], batch_size):
        tokens = encoder(stats_weights[start : start + batch_size]).detach().cpu().float()
        both = torch.cat((tokens, _swap_tokens(tokens)), dim=0)
        distance = _scaled_distance(both, bank_tokens, scales)
        nearest = distance.amin(dim=1)
        batch_nonempty = stats_nonempty[start : start + batch_size]
        eligible = torch.cat((batch_nonempty, batch_nonempty)) & (nearest > 0)
        nearest_positive.append(nearest[eligible])
        log_progress = (
            start == 0
            or start + batch_size >= stats_weights.shape[0]
            or start % (100 * batch_size) == 0
        )
        if log_progress:
            logger.info(
                "temperature pass encoded %d/%d epoch-1 rows",
                min(start + batch_size, stats_weights.shape[0]),
                stats_weights.shape[0],
            )
    positive = torch.cat(nearest_positive) if nearest_positive else torch.empty(0)
    temperature = max(float(positive.median()) if positive.numel() else 1e-6, 1e-6)
    route_sum = torch.zeros(bank_weights.shape[0], dtype=torch.float64)
    route_count = 0
    for start in range(0, stats_weights.shape[0], batch_size):
        tokens = encoder(stats_weights[start : start + batch_size]).detach().cpu().float()
        tokens = torch.cat((tokens, _swap_tokens(tokens)), dim=0)
        distance = _scaled_distance(tokens, bank_tokens, scales)
        routes = torch.softmax(-distance / temperature, dim=-1)
        route_sum += routes.double().sum(dim=0)
        route_count += routes.shape[0]
        log_progress = (
            start == 0
            or start + batch_size >= stats_weights.shape[0]
            or start % (100 * batch_size) == 0
        )
        if log_progress:
            logger.info(
                "mean-route pass encoded %d/%d epoch-1 rows",
                min(start + batch_size, stats_weights.shape[0]),
                stats_weights.shape[0],
            )
    mean_weights = (route_sum / route_count).float()
    mean_weights = 0.5 * (mean_weights + mean_weights.index_select(0, swap_index))
    mean_weights /= mean_weights.sum()
    category_counts = {
        name: int((classify_template_weights(real_weights) == category).sum())
        for category, name in enumerate(CATEGORY_NAMES)
    }
    return DictionaryArtifact(
        weights=bank_weights,
        tokens=bank_tokens,
        scales=scales,
        temperature=temperature,
        mean_weights=mean_weights,
        swap_index=swap_index,
        metadata={
            **metadata,
            "format_version": 1,
            "construction_seed": _CONSTRUCTION_SEED,
            "candidate_limit_per_category": _MAX_CANDIDATES,
            "requested_real_representatives": sum(_QUOTAS),
            "real_representatives": len(real_pairs),
            "effective_dictionary_size": int(bank_weights.shape[0]),
            "category_counts": category_counts,
            "representative_source_pairs": [list(pair) for pair in source_pairs],
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse dictionary construction arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Build and persist the shared motif dictionary."""
    # Keep pure construction helpers cheap to import in CPU tests.  Benchmark
    # assembly remains the production source of the legal split and corpus.
    from src.train_b0 import _dynamic_training_corpus, assemble_data, load_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    cfg = load_config(args.config)
    if (cfg.seed, cfg.optim.epochs, cfg.data.negative_ratio) != (0, 15, 5):
        raise ValueError("shared motif dictionary requires seed 0, 15 epochs and negative_ratio 5")
    assembled = assemble_data(cfg)
    corpus = _dynamic_training_corpus(cfg, assembled)
    graph = build_g_struct(assembled.val_split.train_nodes, assembled.training_positives)
    table = MotifTemplateTable(graph)
    epoch1_pairs = [corpus.pairs[int(i)] for i in corpus.epoch_rows[1]]
    device = torch.device(args.device)
    encoder, reader_metadata = _load_stage1_encoder(args.stage1_checkpoint, device)
    logger.info(
        "building dictionary from %d unique rows (%d epoch-1 rows) on %s",
        len(corpus.pairs),
        len(epoch1_pairs),
        device,
    )
    artifact = build_dictionary(
        table=table,
        corpus_pairs=corpus.pairs,
        epoch1_pairs=epoch1_pairs,
        encoder=encoder,
        batch_size=args.batch_size,
        metadata={
            "config": str(args.config),
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "stage1_checkpoint": str(args.stage1_checkpoint),
            "stage1_checkpoint_sha256": hashlib.sha256(
                args.stage1_checkpoint.read_bytes()
            ).hexdigest(),
            "strategy": cfg.data.strategy,
            "seed": cfg.seed,
            "epochs": cfg.optim.epochs,
            "negative_ratio": cfg.data.negative_ratio,
            "corpus_unique_rows": len(corpus.pairs),
            "epoch1_rows": len(epoch1_pairs),
            "training_positive_rows": len(assembled.training_positives),
            "v_val_nodes_excluded": len(assembled.val_split.v_val),
            **reader_metadata,
        },
    )
    save_dictionary(artifact, args.output)
    logger.info(
        "saved %s: %d real representatives, K_eff=%d, temperature=%.6g",
        args.output,
        artifact.metadata["real_representatives"],
        artifact.weights.shape[0],
        artifact.temperature,
    )


if __name__ == "__main__":
    main()

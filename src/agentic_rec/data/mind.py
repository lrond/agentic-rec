from __future__ import annotations

import csv
import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from agentic_rec.core.linalg import Vector, normalize, zeros
from agentic_rec.data.semantic_embeddings import (
    SemanticEmbeddingConfig,
    encode_news_records,
)


@dataclass(slots=True)
class NewsFeature:
    news_id: str
    category: str
    subcategory: str
    title: str
    abstract: str
    vector: Vector


@dataclass(slots=True)
class BehaviorRecord:
    impression_id: str
    user_id: str
    history_ids: list[str]
    impressions: list[tuple[str, int]]


def load_news_features(
    news_path: str | Path,
    embedding_dim: int,
    embedding_config: SemanticEmbeddingConfig | None = None,
) -> dict[str, NewsFeature]:
    """Load news articles and encode them with Qwen3 semantic embeddings."""
    embedding_config = embedding_config or SemanticEmbeddingConfig()
    news_map: dict[str, NewsFeature] = {}
    with Path(news_path).open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            news_id, category, subcategory, title, abstract = row[:5]
            news_map[news_id] = NewsFeature(
                news_id=news_id,
                category=category,
                subcategory=subcategory,
                title=title,
                abstract=abstract,
                vector=zeros(embedding_dim),
            )

    # Encode all articles with Qwen3
    semantic_vectors = encode_news_records(
        [
            {
                "news_id": feature.news_id,
                "category": feature.category,
                "subcategory": feature.subcategory,
                "title": feature.title,
                "abstract": feature.abstract,
            }
            for feature in news_map.values()
        ],
        output_dim=embedding_dim,
        config=embedding_config,
    )
    for news_id, vector in semantic_vectors.items():
        news_map[news_id].vector = vector

    return news_map


def parse_impressions(impressions_field: str) -> list[tuple[str, int]]:
    impressions: list[tuple[str, int]] = []
    for token in impressions_field.split():
        if "-" not in token:
            continue
        news_id, label = token.rsplit("-", 1)
        try:
            impressions.append((news_id, int(label)))
        except ValueError:
            continue
    return impressions


def iter_behaviors(behaviors_path: str | Path) -> Iterator[BehaviorRecord]:
    with Path(behaviors_path).open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            impression_id, user_id, _, history, impressions = row[:5]
            history_ids = history.split() if history else []
            yield BehaviorRecord(
                impression_id=impression_id,
                user_id=user_id,
                history_ids=history_ids,
                impressions=parse_impressions(impressions),
            )


def padded_history_embeddings(
    history_ids: list[str],
    news_map: dict[str, NewsFeature],
    history_size: int,
    embedding_dim: int,
) -> list[Vector]:
    usable_ids = [news_id for news_id in history_ids if news_id in news_map][-history_size:]
    padding = [zeros(embedding_dim) for _ in range(max(0, history_size - len(usable_ids)))]
    history_vectors = [list(news_map[news_id].vector) for news_id in usable_ids]
    return padding + history_vectors


def aggregate_history_state(
    history_ids: list[str],
    news_map: dict[str, NewsFeature],
    history_size: int,
    embedding_dim: int,
) -> Vector:
    usable_ids = [news_id for news_id in history_ids if news_id in news_map][-history_size:]
    if not usable_ids:
        return zeros(embedding_dim)

    weighted_sum = zeros(embedding_dim)
    total_weight = 0.0
    for offset, news_id in enumerate(usable_ids, start=1):
        weight = float(offset)
        total_weight += weight
        vector = news_map[news_id].vector
        for index, value in enumerate(vector):
            weighted_sum[index] += weight * value

    if total_weight == 0.0:
        return zeros(embedding_dim)
    return [value / total_weight for value in weighted_sum]


def sample_ranker_candidates(
    impressions: list[tuple[str, int]],
    rng: random.Random,
    negatives_per_positive: int,
    *,
    include_all_negatives: bool,
) -> list[tuple[str, int]]:
    positives = [(news_id, label) for news_id, label in impressions if label == 1]
    negatives = [(news_id, label) for news_id, label in impressions if label == 0]

    if include_all_negatives or negatives_per_positive < 0:
        sampled_negatives = negatives
    else:
        limit = negatives_per_positive * max(1, len(positives))
        sampled_negatives = rng.sample(negatives, min(len(negatives), limit))

    selected = positives + sampled_negatives
    rng.shuffle(selected)
    return selected


def export_news_features(news_map: dict[str, NewsFeature], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for feature in news_map.values():
            json.dump(
                {
                    "news_id": feature.news_id,
                    "category": feature.category,
                    "subcategory": feature.subcategory,
                    "title": feature.title,
                    "abstract": feature.abstract,
                    "vector": feature.vector,
                },
                handle,
                ensure_ascii=False,
            )
            handle.write("\n")


def export_news_vectors_npy(
    news_map: dict[str, NewsFeature],
    output_dir: str | Path,
    split_name: str,
) -> tuple[Path, dict[str, int]]:
    """Export news vectors as .npz and return ID→index mapping.

    Writes ``{output_dir}/news_vectors_{split_name}.npz`` with:
        - vectors: (N, D) float32 array
        - ids: list of N news_id strings

    Returns a dict mapping news_id → row index.
    """
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ids = list(news_map.keys())
    id_to_idx = {nid: i for i, nid in enumerate(ids)}
    dim = len(news_map[ids[0]].vector) if ids else 0
    vectors = np.zeros((len(ids), dim), dtype=np.float32)
    for i, nid in enumerate(ids):
        vectors[i] = news_map[nid].vector

    out_path = output_dir / f"news_vectors_{split_name}.npz"
    np.savez_compressed(out_path, vectors=vectors, ids=np.array(ids, dtype=str))
    return out_path, id_to_idx


def prepare_mind_split(
    *,
    split_name: str,
    news_path: str | Path,
    behaviors_path: str | Path,
    output_root: str | Path,
    embedding_dim: int,
    history_size: int,
    negatives_per_positive: int,
    seed: int,
    max_behaviors: int | None = None,
    embedding_config: SemanticEmbeddingConfig | None = None,
    world_model_n_steps: int = 1,
) -> dict[str, int]:
    rng = random.Random(f"{seed}-{split_name}")
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    news_map = load_news_features(news_path, embedding_dim, embedding_config)
    export_news_features(news_map, output_dir / f"news_features_{split_name}.jsonl")
    vectors_path, id_to_idx = export_news_vectors_npy(news_map, output_dir, split_name)
    print(f"Exported {len(id_to_idx)} news vectors to {vectors_path}")

    ranker_path = output_dir / f"ranker_{split_name}.jsonl"
    world_model_path = output_dir / f"world_model_{split_name}.jsonl"

    counters = {
        "news": len(news_map),
        "behaviors": 0,
        "ranker_examples": 0,
        "world_model_examples": 0,
    }

    include_all_negatives = split_name != "train"

    with ranker_path.open("w", encoding="utf-8") as ranker_handle, world_model_path.open(
        "w", encoding="utf-8"
    ) as world_model_handle:
        for behavior in iter_behaviors(behaviors_path):
            if max_behaviors is not None and counters["behaviors"] >= max_behaviors:
                break
            counters["behaviors"] += 1
            filtered_impressions = [
                (news_id, label)
                for news_id, label in behavior.impressions
                if news_id in news_map
            ]
            if not filtered_impressions:
                continue

            history_vectors = padded_history_embeddings(
                behavior.history_ids,
                news_map,
                history_size,
                embedding_dim,
            )
            sampled_candidates = sample_ranker_candidates(
                filtered_impressions,
                rng,
                negatives_per_positive,
                include_all_negatives=include_all_negatives,
            )

            for news_id, label in sampled_candidates:
                json.dump(
                    {
                        "impression_id": behavior.impression_id,
                        "user_id": behavior.user_id,
                        "history_ids": [nid for nid in behavior.history_ids if nid in news_map][-history_size:],
                        "candidate_id": news_id,
                        "category": news_map[news_id].category,
                        "label": float(label),
                    },
                    ranker_handle,
                    ensure_ascii=False,
                )
                ranker_handle.write("\n")
                counters["ranker_examples"] += 1

            clicked_ids = [news_id for news_id, label in filtered_impressions if label == 1]
            if world_model_n_steps <= 1:
                # Single-step: one row per clicked article
                for clicked_id in clicked_ids:
                    json.dump(
                        {
                            "impression_id": behavior.impression_id,
                            "user_id": behavior.user_id,
                            "history_ids": [nid for nid in behavior.history_ids if nid in news_map][-history_size:],
                            "clicked_id": clicked_id,
                            "category": news_map[clicked_id].category,
                        },
                        world_model_handle,
                        ensure_ascii=False,
                    )
                    world_model_handle.write("\n")
                    counters["world_model_examples"] += 1
            else:
                # Multi-step: one row per N consecutive clicks within the impression
                for start in range(0, len(clicked_ids) - world_model_n_steps + 1):
                    step_ids = clicked_ids[start:start + world_model_n_steps]
                    json.dump(
                        {
                            "impression_id": behavior.impression_id,
                            "user_id": behavior.user_id,
                            "history_ids": [nid for nid in behavior.history_ids if nid in news_map][-history_size:],
                            "clicked_ids": step_ids,
                            "categories": [news_map[cid].category for cid in step_ids],
                        },
                        world_model_handle,
                        ensure_ascii=False,
                    )
                    world_model_handle.write("\n")
                    counters["world_model_examples"] += 1

    return counters

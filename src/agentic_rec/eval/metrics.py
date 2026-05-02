from __future__ import annotations

import math
from collections import defaultdict

from agentic_rec.core.linalg import Vector, cosine_similarity


def binary_auc(labels: list[int], scores: list[float]) -> float | None:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length.")

    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    order = sorted(range(len(scores)), key=lambda index: scores[index])
    rank_sum = 0.0
    current_rank = 1
    cursor = 0

    while cursor < len(order):
        end = cursor
        while end < len(order) and scores[order[end]] == scores[order[cursor]]:
            end += 1

        average_rank = (current_rank + (current_rank + (end - cursor) - 1)) / 2.0
        for position in range(cursor, end):
            if labels[order[position]] == 1:
                rank_sum += average_rank

        current_rank += end - cursor
        cursor = end

    u_statistic = rank_sum - positives * (positives + 1) / 2.0
    return u_statistic / (positives * negatives)


def binary_accuracy(labels: list[int], scores: list[float], threshold: float = 0.5) -> float:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length.")
    if not labels:
        return 0.0

    correct = 0
    for label, score in zip(labels, scores):
        prediction = 1 if score >= threshold else 0
        if prediction == label:
            correct += 1
    return correct / len(labels)


def recall_at_k(labels: list[int], k: int) -> float:
    positives = sum(labels)
    if positives == 0:
        return 0.0
    return sum(labels[:k]) / positives


def ndcg_at_k(labels: list[int], k: int) -> float:
    truncated = labels[:k]
    if not truncated:
        return 0.0

    def dcg(values: list[int]) -> float:
        total = 0.0
        for index, relevance in enumerate(values, start=1):
            gain = (2**relevance) - 1
            total += gain / math.log2(index + 1)
        return total

    ideal = sorted(labels, reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg(truncated) / ideal_dcg


def intent_coverage_at_k(intents: list[str], k: int) -> float:
    if k <= 0 or not intents:
        return 0.0
    truncated = intents[:k]
    if not truncated:
        return 0.0
    return len(set(truncated)) / len(truncated)


def intra_list_similarity_at_k(vectors: list[Vector], k: int) -> float:
    truncated = vectors[:k]
    if len(truncated) < 2:
        return 0.0

    total = 0.0
    pairs = 0
    for left_index in range(len(truncated)):
        for right_index in range(left_index + 1, len(truncated)):
            total += cosine_similarity(truncated[left_index], truncated[right_index])
            pairs += 1
    return total / pairs


def grouped_ranking_metrics(
    rows: list[dict[str, object]],
    scores: list[float],
    *,
    impression_key: str = "impression_id",
    label_key: str = "label",
    top_k: int = 5,
) -> dict[str, float | int | None]:
    if len(rows) != len(scores):
        raise ValueError("rows and scores must have the same length.")

    groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        if impression_key not in row or label_key not in row:
            continue
        impression_id = str(row[impression_key])
        label = 1 if float(row[label_key]) >= 0.5 else 0
        groups[impression_id].append((score, label))

    if not groups:
        return {
            "impressions": 0,
            f"recall@{top_k}": None,
            f"ndcg@{top_k}": None,
        }

    recall_total = 0.0
    ndcg_total = 0.0
    usable_groups = 0

    for entries in groups.values():
        ranked_labels = [
            label for _, label in sorted(entries, key=lambda item: item[0], reverse=True)
        ]
        if sum(ranked_labels) == 0:
            continue
        recall_total += recall_at_k(ranked_labels, top_k)
        ndcg_total += ndcg_at_k(ranked_labels, top_k)
        usable_groups += 1

    if usable_groups == 0:
        return {
            "impressions": len(groups),
            f"recall@{top_k}": 0.0,
            f"ndcg@{top_k}": 0.0,
        }

    return {
        "impressions": len(groups),
        f"recall@{top_k}": recall_total / usable_groups,
        f"ndcg@{top_k}": ndcg_total / usable_groups,
    }

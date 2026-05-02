from __future__ import annotations

from agentic_rec.models.ranker import BaseRanker, NewsItem


def intent_coverage_rerank(
    user_state: list[float],
    anchor: NewsItem | None,
    candidates: list[NewsItem],
    ranker: BaseRanker,
    top_k: int = 5,
    relevance_weight: float = 0.72,
    coverage_weight: float = 0.28,
) -> list[NewsItem]:
    """Rerank candidates by relevance weighted with intent-coverage diversity.

    Returns at most top_k items. If anchor is None or candidates are empty,
    returns an empty list.
    """
    if anchor is None or not candidates:
        return []

    selected = [anchor]
    covered_intents = {anchor.category}
    pool = [item for item in candidates if item.item_id != anchor.item_id]

    while pool and len(selected) < top_k:
        best_item: NewsItem | None = None
        best_score = float("-inf")
        for item in pool:
            relevance = ranker.score(user_state, item, anchor=anchor)
            coverage_gain = 1.0 if item.category not in covered_intents else 0.0
            score = relevance_weight * relevance + coverage_weight * coverage_gain
            if score > best_score:
                best_score = score
                best_item = item

        if best_item is None:
            break

        selected.append(best_item)
        covered_intents.add(best_item.category)
        pool = [item for item in pool if item.item_id != best_item.item_id]

    return selected

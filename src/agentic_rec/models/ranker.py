from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from agentic_rec.core.linalg import Vector, cosine_similarity, dot


@dataclass(slots=True)
class NewsItem:
    item_id: str
    title: str
    category: str
    vector: Vector
    url: str = ""
    summary: str = ""


class BaseRanker(ABC):
    """Abstract interface for rankers used by the planner and engine."""

    @abstractmethod
    def score(
        self,
        user_state: Vector,
        item: NewsItem,
        anchor: NewsItem | None = None,
    ) -> float:
        """Return a relevance score for a candidate item."""

    @abstractmethod
    def rank(
        self,
        user_state: Vector,
        candidates: list[NewsItem],
        anchor: NewsItem | None = None,
    ) -> list[NewsItem]:
        """Rank candidates by relevance (descending)."""


class SimpleRanker(BaseRanker):
    """A small scoring helper used by the pure-Python planner demo."""

    def __init__(self, user_weight: float = 0.7, anchor_weight: float = 0.3) -> None:
        self.user_weight = user_weight
        self.anchor_weight = anchor_weight

    def score(self, user_state: Vector, item: NewsItem, anchor: NewsItem | None = None) -> float:
        user_score = dot(user_state, item.vector)
        anchor_score = 0.0 if anchor is None else cosine_similarity(anchor.vector, item.vector)
        return self.user_weight * user_score + self.anchor_weight * anchor_score

    def rank(
        self,
        user_state: Vector,
        candidates: list[NewsItem],
        anchor: NewsItem | None = None,
    ) -> list[NewsItem]:
        return sorted(
            candidates,
            key=lambda item: self.score(user_state, item, anchor),
            reverse=True,
        )

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_rec.core.linalg import Vector, dot, norm, vector_sub
from agentic_rec.models.ranker import BaseRanker, NewsItem
from agentic_rec.world_model.continuous_ode import LinearWorldModel


@dataclass(slots=True)
class RewardWeights:
    relevance: float = 1.0
    drift: float = 0.35
    coverage: float = 0.8
    long_term: float = 0.3
    diversity: float = 0.15


@dataclass(slots=True)
class BeamNode:
    state: Vector
    path: list[NewsItem] = field(default_factory=list)
    total_reward: float = 0.0
    seen_categories: set[str] = field(default_factory=set)
    seen_item_ids: set[str] = field(default_factory=set)


class BeamPlanner:
    def __init__(
        self,
        world_model: LinearWorldModel,
        ranker: BaseRanker,
        horizon: int = 2,
        beam_width: int = 3,
        branching_factor: int = 4,
        reward_weights: RewardWeights | None = None,
    ) -> None:
        self.world_model = world_model
        self.ranker = ranker
        self.horizon = horizon
        self.beam_width = beam_width
        self.branching_factor = branching_factor
        self.reward_weights = reward_weights or RewardWeights()

    def _step_reward(self, node: BeamNode, item: NewsItem, next_state: Vector) -> float:
        relevance = self.ranker.score(node.state, item)
        drift_gain = norm(vector_sub(next_state, node.state))
        long_term = dot(next_state, item.vector)
        intent_coverage = 1.0 if item.category not in node.seen_categories else 0.0
        # Category diversity bonus: penalize nodes that over-represent one category
        category_counts: dict[str, int] = {}
        for path_item in node.path:
            category_counts[path_item.category] = category_counts.get(path_item.category, 0) + 1
        max_count = max(category_counts.values()) if category_counts else 0
        path_len = max(len(node.path), 1)
        concentration = max_count / path_len
        diversity_bonus = (1.0 - concentration) if item.category in category_counts else 1.0

        return (
            self.reward_weights.relevance * relevance
            + self.reward_weights.drift * drift_gain
            + self.reward_weights.coverage * intent_coverage
            + self.reward_weights.long_term * long_term
            + self.reward_weights.diversity * diversity_bonus
        )

    def plan(self, user_state: Vector, candidates: list[NewsItem]) -> BeamNode:
        beams = [BeamNode(state=user_state)]

        for _ in range(self.horizon):
            expansions: list[BeamNode] = []
            for node in beams:
                remaining_candidates = [
                    item for item in candidates if item.item_id not in node.seen_item_ids
                ]
                if not remaining_candidates:
                    continue

                ranked = self.ranker.rank(node.state, remaining_candidates)[
                    : self.branching_factor
                ]
                for item in ranked:
                    next_state = self.world_model.step_krylov(node.state, item.vector)
                    reward = self._step_reward(node, item, next_state)
                    expansions.append(
                        BeamNode(
                            state=next_state,
                            path=node.path + [item],
                            total_reward=node.total_reward + reward,
                            seen_categories=node.seen_categories | {item.category},
                            seen_item_ids=node.seen_item_ids | {item.item_id},
                        )
                    )

            if not expansions:
                break

            beams = sorted(expansions, key=lambda node: node.total_reward, reverse=True)[
                : self.beam_width
            ]

        return beams[0]

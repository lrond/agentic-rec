from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PlanningConfig:
    """Configuration for the intent-drift planning pipeline.

    Defaults match the MIND-small training setup described in README.md.
    """

    embedding_dim: int = 64
    history_size: int = 50
    beam_width: int = 5
    branching_factor: int = 8
    plan_horizon: int = 3
    top_k: int = 10
    delta_t: float = 1.0
    krylov_steps: int = 4
    negatives_per_positive: int = 4
    ranker_epochs: int = 5
    world_model_epochs: int = 5
    rerank_method: str = "intent_coverage"  # "intent_coverage" | "mmr" | "none"

    @classmethod
    def from_json_file(cls, path: str | Path) -> "PlanningConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})

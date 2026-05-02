from __future__ import annotations

"""Bridge between trained PyTorch models and the pure-Python planner.

This module handles:
1. Exporting trained TorchLinearWorldModel matrices → LinearWorldModel
2. Loading trained RankerModel weights → TorchRanker adapter
3. Checkpoint management with a single entrypoint
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_rec.core.linalg import Matrix
from agentic_rec.models.ranker import BaseRanker, NewsItem, Vector
from agentic_rec.world_model.continuous_ode import LinearWorldModel
from agentic_rec.world_model.neural_ode import (
    NeuralODEAdapter,
    load_checkpoint as load_neural_ode,
)


@dataclass(slots=True)
class PlannerCheckpoint:
    """A fully loadable planner checkpoint from trained artifacts."""

    world_model: LinearWorldModel
    ranker: BaseRanker
    embedding_dim: int

    @classmethod
    def from_artifacts(
        cls,
        *,
        ranker_ckpt: str | Path = "artifacts/ranker.pt",
        world_model_ckpt: str | Path = "artifacts/world_model.pt",
        device: str = "cpu",
        delta_t: float = 1.0,
        krylov_steps: int = 4,
    ) -> "PlannerCheckpoint":
        """Load a full planner checkpoint from trained PyTorch artifacts."""
        world_model = _load_world_model_from_ckpt(
            world_model_ckpt, device=device, delta_t=delta_t, krylov_steps=krylov_steps
        )
        ranker, embedding_dim = _load_ranker_from_ckpt(
            ranker_ckpt, device=device
        )
        return cls(world_model=world_model, ranker=ranker, embedding_dim=embedding_dim)


def _load_world_model_from_ckpt(
    ckpt_path: str | Path,
    *,
    device: str = "cpu",
    delta_t: float = 1.0,
    krylov_steps: int = 4,
) -> LinearWorldModel | NeuralODEAdapter:
    """Load a world model from checkpoint, auto-detecting linear vs Neural ODE."""
    import torch

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if not isinstance(ckpt, dict):
        raise ValueError(
            f"World model checkpoint at {ckpt_path!r} is not a dict, got {type(ckpt)}"
        )

    model_type = ckpt.get("model_type", "linear")

    if model_type == "neural_ode":
        ode_model = load_neural_ode(str(ckpt_path), device=device)
        rk4_steps = ckpt.get("rk4_steps", 4)
        return NeuralODEAdapter(ode_model, device=device, rk4_steps=rk4_steps)

    # Linear world model (original path)
    state_dict = ckpt.get("model_state_dict", ckpt)
    if "transition" not in state_dict or "control" not in state_dict:
        raise ValueError(
            f"World model checkpoint at {ckpt_path!r} does not contain "
            f"'transition' / 'control' matrices. "
            f"Top-level keys: {list(ckpt.keys())}, "
            f"state_dict keys: {list(state_dict.keys()) if isinstance(state_dict, dict) else 'N/A'}"
        )

    transition = state_dict["transition"].detach().cpu().tolist()
    control = state_dict["control"].detach().cpu().tolist()

    return LinearWorldModel(
        transition=transition,
        control=control,
        delta_t=delta_t,
        krylov_steps=krylov_steps,
    )


def _load_ranker_from_ckpt(
    ckpt_path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[BaseRanker, int]:
    """Load a trained RankerModel and wrap it in a TorchRanker adapter."""
    import torch

    from agentic_rec.trainers.train_ranker import RankerModel

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Detect embedding_dim from the GRU input size
    gru_weight = state_dict.get("user_encoder.gru.weight_ih_l0")
    if gru_weight is None:
        raise ValueError(
            f"Ranker checkpoint at {ckpt_path!r} does not contain "
            f"'user_encoder.gru.weight_ih_l0'. Cannot infer embedding_dim."
        )
    embedding_dim = gru_weight.shape[1]
    hidden_dim = gru_weight.shape[0] // 3  # GRU has 3 gates

    model = RankerModel(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    ranker = TorchRanker(model, device=device)
    return ranker, embedding_dim


class TorchRanker(BaseRanker):
    """Adapter that wraps a trained PyTorch RankerModel for use in the planner."""

    def __init__(self, model: Any, device: str = "cpu") -> None:
        self._model = model
        self._device = device
        import torch
        self._torch = torch

    def score(
        self,
        user_state: Vector,
        item: NewsItem,
        anchor: NewsItem | None = None,
    ) -> float:
        """Score a single candidate against user state using the trained model."""
        import torch
        # Treat user_state as the GRU-encoded latent vector
        state_tensor = torch.tensor([user_state], dtype=torch.float32, device=self._device)
        candidate_tensor = torch.tensor([item.vector], dtype=torch.float32, device=self._device)
        with torch.no_grad():
            logit = self._model.click_predictor(state_tensor, candidate_tensor)
            prob = torch.sigmoid(logit).item()
        return prob

    def rank(
        self,
        user_state: Vector,
        candidates: list[NewsItem],
        anchor: NewsItem | None = None,
    ) -> list[NewsItem]:
        """Rank candidates by predicted click probability."""
        if not candidates:
            return []
        import torch
        state_tensor = torch.tensor([user_state], dtype=torch.float32, device=self._device)
        vectors = torch.tensor(
            [item.vector for item in candidates],
            dtype=torch.float32,
            device=self._device,
        )
        with torch.no_grad():
            # Process in one batch through ClickPredictor
            state_expanded = state_tensor.expand(len(candidates), -1)
            logits = self._model.click_predictor(state_expanded, vectors)
            scores = torch.sigmoid(logits).cpu().tolist()

        ranked = sorted(
            zip(candidates, scores), key=lambda pair: pair[1], reverse=True
        )
        return [item for item, _ in ranked]


# Re-export for convenience
__all__ = ["PlannerCheckpoint", "TorchRanker"]

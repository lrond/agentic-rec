"""World-model components."""

# Neural ODE components require PyTorch — import lazily to avoid
# breaking the pure-Python demo path.

__all__ = [
    "NeuralODEWorldModel",
    "load_checkpoint",
    "save_checkpoint",
    "train_world_model_epoch",
]


def __getattr__(name: str):
    if name in {
        "NeuralODEWorldModel",
        "train_world_model_epoch",
        "save_checkpoint",
        "load_checkpoint",
    }:
        from agentic_rec.world_model import neural_ode as _mod

        return getattr(_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

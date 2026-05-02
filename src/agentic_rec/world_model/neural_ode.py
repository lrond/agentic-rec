from __future__ import annotations

"""Nonlinear Neural ODE world model.

    du/dt = MLP(u, a)

Uses Runge-Kutta 4th-order integration for solving the ODE.
Provides the same train_world_model_epoch interface as the linear world model.
"""

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover
    raise SystemExit("Install PyTorch first: pip install -e '.[train]'")


class NeuralODEWorldModel(nn.Module):
    """Nonlinear continuous-time world model:  du/dt = MLP(u, a).

    The MLP has 3 hidden layers with ReLU activations:
        (state_dim + action_dim) → 512 → 512 → state_dim
    """

    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 256,
        hidden: int = 512,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, u: Tensor, a: Tensor) -> Tensor:
        """Compute du/dt = MLP([u, a])."""
        return self.net(torch.cat([u, a], dim=-1))

    def step_rk4(
        self,
        u: Tensor,
        a: Tensor,
        dt: float = 1.0,
        steps: int = 4,
    ) -> Tensor:
        """Integrate one step using RK4.

        Args:
            u: Current state tensor (..., state_dim).
            a: Action tensor (..., action_dim).
            dt: Total time step size.
            steps: Number of RK4 sub-steps.

        Returns:
            Next state tensor of same shape as u.
        """
        h = dt / steps
        for _ in range(steps):
            k1 = self.forward(u, a)
            k2 = self.forward(u + 0.5 * h * k1, a)
            k3 = self.forward(u + 0.5 * h * k2, a)
            k4 = self.forward(u + h * k3, a)
            u = u + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return u


def train_world_model_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda",
) -> float:
    """Train the Neural ODE world model for one epoch.

    Uses the same interface as trainers.train_world_model.train_world_model_epoch.
    Expects dataloader batches with keys: "state", "action", "next_state".
    """
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        current_state = batch["state"].to(device)
        action = batch["action"].to(device)
        next_state = batch["next_state"].to(device)

        prediction = model.step_rk4(current_state, action)
        loss = criterion(prediction, next_state)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = current_state.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def save_checkpoint(
    model: NeuralODEWorldModel,
    path: str,
) -> None:
    """Save the Neural ODE world model checkpoint."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dim": model.state_dim,
            "action_dim": model.action_dim,
            "hidden": model.hidden,
            "model_type": "neural_ode",
        },
        path,
    )


def load_checkpoint(
    path: str,
    device: str = "cpu",
) -> NeuralODEWorldModel:
    """Load a Neural ODE world model from a checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = NeuralODEWorldModel(
        state_dim=checkpoint["state_dim"],
        action_dim=checkpoint["action_dim"],
        hidden=checkpoint.get("hidden", 512),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model


class NeuralODEAdapter:
    """Adapter that wraps a trained NeuralODEWorldModel for the planner.

    Provides the same ``step_krylov(state: list, action: list) -> list``
    interface as LinearWorldModel, converting Python lists to tensors,
    running RK4 integration, and converting back.
    """

    def __init__(self, model: NeuralODEWorldModel, device: str = "cpu", rk4_steps: int = 4) -> None:
        self._model = model
        self._device = device
        self._rk4_steps = rk4_steps

    def step_krylov(self, state, action):
        """Integrate one step using RK4. state, action are Python lists, returns list."""
        import torch
        s = torch.tensor([state], dtype=torch.float32, device=self._device)
        a = torch.tensor([action], dtype=torch.float32, device=self._device)
        with torch.no_grad():
            next_s = self._model.step_rk4(s, a, steps=self._rk4_steps)
        return next_s[0].cpu().tolist()

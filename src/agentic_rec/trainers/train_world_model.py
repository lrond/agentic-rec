from __future__ import annotations

"""
PyTorch world-model training for user-state transition dynamics.

The training pipeline is:
1. Train the click ranker.
2. Freeze the user encoder and extract latent user states u_t.
3. Fit the world model on (u_t, a_t, u_{t+1}) tuples.
"""

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - handled only at runtime
    raise SystemExit("Install PyTorch first: pip install -e '.[train]'")


class TorchLinearWorldModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.transition = nn.Parameter(torch.eye(state_dim) * -0.1)
        self.control = nn.Parameter(torch.randn(state_dim, action_dim) * 0.05)

    def forward(self, user_state: Tensor, action: Tensor, delta_t: float = 1.0) -> Tensor:
        batch_size = user_state.size(0)
        transition = torch.matrix_exp(self.transition * delta_t)
        control_direction = action @ self.control.transpose(0, 1)
        propagated = user_state @ transition.transpose(0, 1)
        midpoint = control_direction @ torch.matrix_exp(
            self.transition.transpose(0, 1) * (0.5 * delta_t)
        )
        return propagated + delta_t * midpoint


def train_world_model_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda",
) -> float:
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        current_state = batch["state"].to(device)
        action = batch["action"].to(device)
        next_state = batch["next_state"].to(device)

        prediction = model(current_state, action)
        loss = criterion(prediction, next_state)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = current_state.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)

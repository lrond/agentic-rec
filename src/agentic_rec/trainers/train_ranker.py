from __future__ import annotations

"""PyTorch click ranker used by the intent-drift recommendation pipeline."""

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - handled only at runtime
    raise SystemExit("Install PyTorch first: pip install -e '.[train]'")


class GRUUserEncoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

    def forward(self, history_embeddings: Tensor) -> Tensor:
        _, hidden = self.gru(history_embeddings)
        return hidden[-1]


class ClickPredictor(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.candidate_projection = nn.Linear(embedding_dim, hidden_dim)
        feature_dim = hidden_dim + embedding_dim + hidden_dim
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, user_state: Tensor, candidate_embedding: Tensor) -> Tensor:
        projected_candidate = self.candidate_projection(candidate_embedding)
        interaction = user_state * projected_candidate
        features = torch.cat([user_state, candidate_embedding, interaction], dim=-1)
        return self.layers(features).squeeze(-1)


class RankerModel(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.user_encoder = GRUUserEncoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        self.click_predictor = ClickPredictor(hidden_dim=hidden_dim, embedding_dim=embedding_dim)

    def forward(self, history_embeddings: Tensor, candidate_embedding: Tensor) -> Tensor:
        user_state = self.user_encoder(history_embeddings)
        return self.click_predictor(user_state, candidate_embedding)


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda",
) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        history_embeddings = batch["history"].to(device)
        candidate_embeddings = batch["candidate"].to(device)
        labels = batch["label"].float().to(device)

        logits = model(history_embeddings, candidate_embeddings)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)

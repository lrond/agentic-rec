from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from agentic_rec.trainers.train_ranker import RankerModel, train_one_epoch


def _load_vectors_npz(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    data = np.load(path)
    vectors = data["vectors"]
    ids = data["ids"]
    id_to_idx = {str(nid): i for i, nid in enumerate(ids)}
    return vectors, id_to_idx


class RankerDataset(Dataset):
    """Lightweight dataset: stores raw rows in memory, batch-level vectorized embedding lookup via collate_fn."""

    def __init__(self, path: Path, history_size: int = 50, max_rows: int | None = None) -> None:
        self._history_size = history_size
        self._rows: list[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if max_rows and i >= max_rows:
                    break
                stripped = line.strip()
                if stripped:
                    self._rows.append(json.loads(stripped))
        if not self._rows:
            raise ValueError(f"Ranker training data is empty: {path}")
        print(f"RankerDataset: loaded {len(self._rows)} rows")

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict:
        return self._rows[index]


def ranker_collate_fn(batch: list[dict], vectors: np.ndarray, id_to_idx: dict[str, int], history_size: int = 50) -> dict[str, torch.Tensor]:
    """Vectorized batch embedding lookup — no per-sample Python loops."""
    B = len(batch)
    emb_dim = vectors.shape[1]
    history = np.zeros((B, history_size, emb_dim), dtype=np.float32)
    candidate = np.zeros((B, emb_dim), dtype=np.float32)
    labels = np.zeros(B, dtype=np.float32)

    for b, row in enumerate(batch):
        hids = row.get("history_ids", [])
        usable = [nid for nid in hids if nid in id_to_idx][-history_size:]
        indices = np.array([id_to_idx[n] for n in usable], dtype=np.int64)
        if len(indices) > 0:
            history[b, history_size - len(indices):] = vectors[indices]
        cid = row.get("candidate_id", "")
        if cid in id_to_idx:
            candidate[b] = vectors[id_to_idx[cid]]
        labels[b] = float(row["label"])

    return {
        "history": torch.from_numpy(history),
        "candidate": torch.from_numpy(candidate),
        "label": torch.from_numpy(labels),
    }


def choose_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ranker baseline on JSONL data.")
    parser.add_argument("--train-data", type=Path, help="Path to JSONL rows with history_ids/candidate_id/label.")
    parser.add_argument("--vectors-path", type=Path, help="Path to news_vectors_*.npz binary embedding matrix.")
    parser.add_argument("--save-path", type=Path, default=Path("artifacts/ranker.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--history-size", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers for parallel CPU embedding lookup.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.train_data is None or args.vectors_path is None:
        raise SystemExit("--train-data and --vectors-path are required")

    vectors, id_to_idx = _load_vectors_npz(args.vectors_path)
    emb_dim = vectors.shape[1]
    history_size = args.history_size or 50

    # Print GPU info
    print(f"Using {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Embedding dim: {emb_dim}, history size: {history_size}")

    dataset = RankerDataset(args.train_data, history_size=history_size)
    hidden_dim = args.hidden_dim or emb_dim
    device = choose_device(args.device)

    # Use multiprocessing DataLoader with collate_fn for parallel embedding lookup
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=lambda batch: ranker_collate_fn(batch, vectors, id_to_idx, history_size),
    )

    model = RankerModel(embedding_dim=emb_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, dataloader, optimizer, device=device)
        print(f"epoch={epoch} loss={loss:.4f}")

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "embedding_dim": emb_dim, "hidden_dim": hidden_dim, "history_size": history_size},
        args.save_path,
    )
    print(f"saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()

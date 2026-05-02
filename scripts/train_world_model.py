from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from agentic_rec.trainers.train_world_model import (
    TorchLinearWorldModel,
    train_world_model_epoch as train_linear_epoch,
)
from agentic_rec.world_model.neural_ode import (
    NeuralODEWorldModel,
    train_world_model_epoch as train_neural_ode_epoch,
    save_checkpoint as save_neural_ode_ckpt,
)


def _load_vectors_npz(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    data = np.load(path)
    vectors = data["vectors"]
    ids = data["ids"]
    id_to_idx = {str(nid): i for i, nid in enumerate(ids)}
    return vectors, id_to_idx


def _state_from_history(history_ids, vectors, id_to_idx, history_size, embedding_dim):
    usable = [nid for nid in history_ids if nid in id_to_idx][-history_size:]
    if not usable:
        return np.zeros(embedding_dim, dtype=np.float32)
    weighted = np.zeros(embedding_dim, dtype=np.float32)
    total_w = 0.0
    for offset, nid in enumerate(usable, start=1):
        w = float(offset)
        total_w += w
        weighted += w * vectors[id_to_idx[nid]]
    return (weighted / total_w).astype(np.float32)


class WorldModelDataset(Dataset):
    """Precomputed dataset: state/action/next_state built in __init__, O(1) __getitem__."""

    def __init__(self, path: Path, vectors_path: Path, history_size: int = 50,
                 max_rows: int | None = None) -> None:
        vectors, id_to_idx = _load_vectors_npz(vectors_path)
        emb_dim = vectors.shape[1]

        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if max_rows and i >= max_rows:
                    break
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
        if not rows:
            raise ValueError(f"World-model training data is empty: {path}")
        print(f"WorldModelDataset: loaded {len(rows)} rows, precomputing states...")

        N = len(rows)
        self._state = np.zeros((N, emb_dim), dtype=np.float32)
        self._action = np.zeros((N, emb_dim), dtype=np.float32)
        self._next_state = np.zeros((N, emb_dim), dtype=np.float32)

        for i, row in enumerate(rows):
            hids = row.get("history_ids", [])
            cid = row.get("clicked_id", "")
            self._state[i] = _state_from_history(hids, vectors, id_to_idx, history_size, emb_dim)
            if cid in id_to_idx:
                self._action[i] = vectors[id_to_idx[cid]]
            self._next_state[i] = _state_from_history(hids + [cid], vectors, id_to_idx, history_size, emb_dim)

        print(f"WorldModelDataset: precomputed {N} samples ({emb_dim}-dim)")

    def __len__(self) -> int:
        return len(self._state)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "state": torch.from_numpy(self._state[index]),
            "action": torch.from_numpy(self._action[index]),
            "next_state": torch.from_numpy(self._next_state[index]),
        }


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return rows


def build_synthetic_rows(num_examples, state_dim, action_dim, seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(num_examples):
        state = [rng.uniform(-1.0, 1.0) for _ in range(state_dim)]
        action = [rng.uniform(-1.0, 1.0) for _ in range(action_dim)]
        next_state = [0.92 * state[i] + 0.08 * action[i % action_dim] for i in range(state_dim)]
        rows.append({"state": state, "action": action, "next_state": next_state})
    return rows


def choose_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train world model on JSONL state/action transitions.")
    parser.add_argument("--train-data", type=Path, help="Path to JSONL rows.")
    parser.add_argument("--vectors-path", type=Path, help="Path to news_vectors_*.npz.")
    parser.add_argument("--save-path", type=Path, default=Path("artifacts/world_model.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--state-dim", type=int, default=None)
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--synthetic-examples", type=int, default=256)
    parser.add_argument("--model-type", choices=["linear", "neural_ode"], default="linear")
    parser.add_argument("--ode-hidden", type=int, default=512)
    parser.add_argument("--ode-steps", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.train_data is not None:
        if args.vectors_path is None:
            raise SystemExit("--vectors-path is required with --train-data")
        dataset = WorldModelDataset(args.train_data, args.vectors_path)
        state_dim = args.state_dim or dataset._state.shape[1]
        action_dim = args.action_dim or dataset._action.shape[1]
    else:
        state_dim = args.state_dim or 8
        action_dim = args.action_dim or state_dim
        rows = build_synthetic_rows(args.synthetic_examples, state_dim, action_dim, args.seed)
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for row in rows:
            tmp.write(json.dumps(row) + "\n")
        tmp.close()
        dataset = WorldModelDataset(Path(tmp.name), Path("dummy"))

    device = choose_device(args.device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    if args.model_type == "neural_ode":
        model = NeuralODEWorldModel(state_dim=state_dim, action_dim=action_dim, hidden=args.ode_hidden).to(device)
        train_fn = train_neural_ode_epoch
        print(f"Using Neural ODE world model (hidden={args.ode_hidden}, rk4_steps={args.ode_steps})")
    else:
        model = TorchLinearWorldModel(state_dim=state_dim, action_dim=action_dim).to(device)
        train_fn = train_linear_epoch
        print("Using linear world model")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = train_fn(model, dataloader, optimizer, device=device)
        print(f"epoch={epoch} loss={loss:.4f}")

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    if args.model_type == "neural_ode":
        save_neural_ode_ckpt(model, str(args.save_path))
    else:
        torch.save({"model_state_dict": model.state_dict(), "state_dim": state_dim, "action_dim": action_dim}, args.save_path)
    print(f"saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()

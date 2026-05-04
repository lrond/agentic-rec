from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - handled only at runtime
    raise SystemExit("Install training dependencies first: pip install -e '.[train]'")

from agentic_rec.eval.metrics import (
    binary_accuracy,
    binary_auc,
    grouped_ranking_metrics,
    intent_coverage_at_k,
    intra_list_similarity_at_k,
)
from agentic_rec.models.ranker import NewsItem
from agentic_rec.planner.beam_search import BeamPlanner, RewardWeights
from agentic_rec.planner.intent_coverage import intent_coverage_rerank
from agentic_rec.trainers.train_ranker import RankerModel
from agentic_rec.trainers.train_world_model import TorchLinearWorldModel
from agentic_rec.world_model.neural_ode import (
    NeuralODEWorldModel,
    load_checkpoint as load_neural_ode_ckpt,
)
import numpy as np


def _load_vectors_npz(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    """Load news vectors from .npz, return (vectors, id_to_idx)."""
    data = np.load(path)
    vectors = data["vectors"]
    ids = data["ids"]
    id_to_idx = {str(nid): i for i, nid in enumerate(ids)}
    return vectors, id_to_idx


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


def choose_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested_device


def batched(rows: list[dict[str, object]], batch_size: int) -> list[list[dict[str, object]]]:
    return [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]


def evaluate_ranker(
    checkpoint_path: Path,
    data_path: Path,
    *,
    vectors_path: Path | None = None,
    device: str,
    batch_size: int,
    top_k: int,
    history_size: int = 50,
    max_impressions: int = 0,
) -> dict[str, float | int | None]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    rows = load_jsonl(data_path)

    # Impression-level sampling (preserves impression structure for grouped metrics)
    if max_impressions > 0:
        from collections import defaultdict
        imp_groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            imp_groups[str(row.get("impression_id", ""))].append(row)
        imp_ids = list(imp_groups.keys())
        if len(imp_ids) > max_impressions:
            import random
            rng = random.Random(42)
            imp_ids = rng.sample(imp_ids, max_impressions)
        rows = [row for iid in imp_ids for row in imp_groups[iid]]

    # Load embedding lookup if vectors provided
    vectors = id_to_idx = None
    embedding_dim = int(checkpoint["embedding_dim"])
    if vectors_path is not None and vectors_path.exists():
        vectors, id_to_idx = _load_vectors_npz(vectors_path)
        embedding_dim = int(vectors.shape[1])

    model = RankerModel(
        embedding_dim=embedding_dim,
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    scores: list[float] = []
    labels: list[int] = []

    with torch.no_grad():
        for batch_rows in batched(rows, batch_size):
            if vectors is not None:
                # ID-based format: look up embeddings
                batch_size_actual = len(batch_rows)
                history = np.zeros((batch_size_actual, history_size, embedding_dim), dtype=np.float32)
                candidate = np.zeros((batch_size_actual, embedding_dim), dtype=np.float32)
                for b, row in enumerate(batch_rows):
                    hids = row.get("history_ids", [])
                    usable = [nid for nid in hids if nid in id_to_idx][-history_size:]
                    for i, nid in enumerate(usable):
                        history[b, history_size - len(usable) + i] = vectors[id_to_idx[nid]]
                    cid = row.get("candidate_id", "")
                    if cid in id_to_idx:
                        candidate[b] = vectors[id_to_idx[cid]]
                history_t = torch.from_numpy(history).float().to(device)
                candidate_t = torch.from_numpy(candidate).float().to(device)
            else:
                # Legacy format: embeddings inline
                history_t = torch.tensor(
                    [row["history"] for row in batch_rows],
                    dtype=torch.float32, device=device,
                )
                candidate_t = torch.tensor(
                    [row["candidate"] for row in batch_rows],
                    dtype=torch.float32, device=device,
                )

            batch_labels = torch.tensor(
                [float(row["label"]) for row in batch_rows],
                dtype=torch.float32, device=device,
            )

            logits = model(history_t, candidate_t)
            total_loss += criterion(logits, batch_labels).item()
            probabilities = torch.sigmoid(logits).cpu().tolist()
            scores.extend(float(value) for value in probabilities)
            labels.extend(1 if float(row["label"]) >= 0.5 else 0 for row in batch_rows)

    metrics: dict[str, float | int | None] = {
        "examples": len(rows),
        "logloss": total_loss / max(len(rows), 1),
        "accuracy": binary_accuracy(labels, scores),
        "auc": binary_auc(labels, scores),
    }
    metrics.update(grouped_ranking_metrics(rows, scores, top_k=top_k))
    metrics.update(grouped_intent_metrics(rows, scores, top_k=top_k, vectors=vectors, id_to_idx=id_to_idx))
    return metrics


def grouped_intent_metrics(
    rows: list[dict[str, object]],
    scores: list[float],
    *,
    top_k: int,
    vectors: np.ndarray | None = None,
    id_to_idx: dict[str, int] | None = None,
) -> dict[str, float | None]:
    if len(rows) != len(scores):
        raise ValueError("rows and scores must have the same length.")

    groups: dict[str, list[tuple[float, dict[str, object]]]] = {}
    for row, score in zip(rows, scores):
        if "impression_id" not in row or "category" not in row:
            continue
        # Check if we have vectors (ID-based format) or inline candidate
        if vectors is not None and id_to_idx is not None:
            cid = row.get("candidate_id", "")
            if cid not in id_to_idx:
                continue
        elif "candidate" not in row:
            continue
        groups.setdefault(str(row["impression_id"]), []).append((score, row))

    if not groups:
        return {
            f"intent_coverage@{top_k}": None,
            f"intra_list_similarity@{top_k}": None,
        }

    coverage_total = 0.0
    similarity_total = 0.0
    usable_groups = 0
    for entries in groups.values():
        ranked_rows = [
            row for _, row in sorted(entries, key=lambda item: item[0], reverse=True)
        ][:top_k]
        intents = [str(row["category"]) for row in ranked_rows]

        # Get vectors from lookup or inline
        if vectors is not None and id_to_idx is not None:
            vecs = [
                vectors[id_to_idx[str(row.get("candidate_id", ""))]].tolist()
                if str(row.get("candidate_id", "")) in id_to_idx
                else [0.0] * int(vectors.shape[1])
                for row in ranked_rows
            ]
        else:
            vecs = [list(row["candidate"]) for row in ranked_rows]

        coverage_total += intent_coverage_at_k(intents, top_k)
        similarity_total += intra_list_similarity_at_k(vecs, top_k)
        usable_groups += 1

    return {
        f"intent_coverage@{top_k}": coverage_total / max(usable_groups, 1),
        f"intra_list_similarity@{top_k}": similarity_total / max(usable_groups, 1),
    }


def evaluate_identity_baseline(
    data_path: Path,
    *,
    vectors_path: Path | None = None,
    n_steps: int = 3,
    history_size: int = 50,
) -> dict[str, float]:
    """Identity baseline: predict next_state = current_state (no drift)."""
    import torch.nn.functional as F
    import numpy as np
    
    vectors = id_to_idx = None
    if vectors_path is not None and vectors_path.exists():
        vectors, id_to_idx = _load_vectors_npz(vectors_path)
    emb_dim = int(vectors.shape[1]) if vectors is not None else 256
    
    rows = load_jsonl(data_path)
    
    cosines = []
    mses = []
    
    for row in rows:
        hids = row.get("history_ids", [])
        cids = row.get("clicked_ids", [])[:n_steps]
        if len(cids) < n_steps:
            continue
        
        current = _state_from_history_vec(hids, vectors, id_to_idx, history_size, emb_dim)
        all_ids = hids + cids
        target = _state_from_history_vec(all_ids, vectors, id_to_idx, history_size, emb_dim)
        
        # Identity: predict no change
        cos = float(np.dot(current, target) / (np.linalg.norm(current) * np.linalg.norm(target) + 1e-8))
        mse = float(np.mean((current - target) ** 2))
        cosines.append(cos)
        mses.append(mse)
    
    if not cosines:
        return {"identity_examples": 0}
    
    return {
        "identity_examples": len(cosines),
        "identity_mse": round(float(np.mean(mses)), 8),
        "identity_cosine": round(float(np.mean(cosines)), 4),
    }



def evaluate_world_model(
    checkpoint_path: Path,
    data_path: Path,
    *,
    vectors_path: Path | None = None,
    device: str,
    batch_size: int,
    history_size: int = 50,
) -> dict[str, float | int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    rows = load_jsonl(data_path)

    # Load vectors for ID-based format
    vectors = id_to_idx = None
    if vectors_path is not None and vectors_path.exists():
        vectors, id_to_idx = _load_vectors_npz(vectors_path)

    model_type = checkpoint.get("model_type", "linear")
    if model_type == "neural_ode":
        model = load_neural_ode_ckpt(str(checkpoint_path), device=device)
        model.eval()
        is_neural_ode = True
    else:
        state_dim = int(checkpoint.get("state_dim", 256))
        action_dim = int(checkpoint.get("action_dim", 256))
        if vectors is not None:
            state_dim = action_dim = int(vectors.shape[1])
        model = TorchLinearWorldModel(state_dim=state_dim, action_dim=action_dim).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        is_neural_ode = False

    total_mse = 0.0
    total_cosine = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch_rows in batched(rows, batch_size):
            if vectors is not None:
                # ID-based format
                bs = len(batch_rows)
                emb_dim = int(vectors.shape[1])
                state = np.zeros((bs, emb_dim), dtype=np.float32)
                action = np.zeros((bs, emb_dim), dtype=np.float32)
                next_state = np.zeros((bs, emb_dim), dtype=np.float32)
                for b, row in enumerate(batch_rows):
                    hids = row.get("history_ids", [])
                    state[b] = _state_from_history_vec(hids, vectors, id_to_idx, history_size, emb_dim)
                    cid = row.get("clicked_id", "")
                    if cid in id_to_idx:
                        action[b] = vectors[id_to_idx[cid]]
                    next_state[b] = _state_from_history_vec(hids + [cid], vectors, id_to_idx, history_size, emb_dim)
                state_t = torch.from_numpy(state).to(device)
                action_t = torch.from_numpy(action).to(device)
                next_state_t = torch.from_numpy(next_state).to(device)
            else:
                state_t = torch.tensor([row["state"] for row in batch_rows], dtype=torch.float32, device=device)
                action_t = torch.tensor([row["action"] for row in batch_rows], dtype=torch.float32, device=device)
                next_state_t = torch.tensor([row["next_state"] for row in batch_rows], dtype=torch.float32, device=device)

            if is_neural_ode:
                prediction = model.step_rk4(state_t, action_t)
            else:
                prediction = model(state_t, action_t)
            batch_mse = ((prediction - next_state_t) ** 2).mean(dim=-1)
            batch_cosine = F.cosine_similarity(prediction, next_state_t, dim=-1)

            total_mse += batch_mse.sum().item()
            total_cosine += batch_cosine.sum().item()
            total_examples += len(batch_rows)

    return {
        "examples": total_examples,
        "mse": total_mse / max(total_examples, 1),
        "cosine_similarity": total_cosine / max(total_examples, 1),
    }


def _state_from_history_vec(history_ids, vectors, id_to_idx, history_size, embedding_dim):
    """Weighted average of history embeddings."""
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


def evaluate_planner(
    ranker_ckpt: Path,
    world_model_ckpt: Path,
    ranker_data: Path,
    news_features_path: Path,
    *,
    vectors_path: Path | None = None,
    device: str = "cpu",
    top_k: int = 5,
    horizon: int = 3,
    beam_width: int = 5,
    branching_factor: int = 8,
    max_impressions: int = 200,
    history_size: int = 50,
) -> dict[str, float | int]:
    """Evaluate the full planner pipeline on dev data."""
    from agentic_rec.export import PlannerCheckpoint

    ckpt = PlannerCheckpoint.from_artifacts(
        ranker_ckpt=str(ranker_ckpt),
        world_model_ckpt=str(world_model_ckpt),
        device=device,
    )

    # Load vectors for ID-based format
    vectors = id_to_idx = None
    if vectors_path is not None and vectors_path.exists():
        vectors, id_to_idx = _load_vectors_npz(vectors_path)

    # Load news features to build a lookup
    news_rows = load_jsonl(news_features_path)
    news_map: dict[str, dict[str, object]] = {}
    for row in news_rows:
        news_map[str(row["news_id"])] = row

    # Load ranker dev data, group by impression
    rows = load_jsonl(ranker_data)
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["impression_id"])].append(row)

    planner = BeamPlanner(
        world_model=ckpt.world_model,
        ranker=ckpt.ranker,
        horizon=horizon,
        beam_width=beam_width,
        branching_factor=branching_factor,
    )

    total_coverage = 0.0
    total_ils = 0.0
    total_hits = 0
    total_slates = 0

    impression_ids = list(groups.keys())
    if max_impressions > 0 and len(impression_ids) > max_impressions:
        import random
        rng = random.Random(42)
        impression_ids = rng.sample(impression_ids, max_impressions)

    for imp_id in impression_ids:
        group = groups[imp_id]
        if len(group) < 2:
            continue

        # Build candidate pool from this impression
        candidates: list[NewsItem] = []
        clicked_ids: set[str] = set()
        for row in group:
            news_id = str(row["candidate_id"])
            label = float(row["label"])

            # Get vector from lookup or inline
            if vectors is not None and news_id in id_to_idx:
                vector = vectors[id_to_idx[news_id]].tolist()
            elif isinstance(row.get("candidate"), list):
                vector = list(row["candidate"])
            else:
                continue

            category = str(row.get("category", "unknown"))

            title = ""
            if news_id in news_map:
                title = str(news_map[news_id].get("title", news_id))

            candidates.append(NewsItem(
                item_id=news_id,
                title=title or news_id,
                category=category,
                vector=vector,
            ))

            if label >= 0.5:
                clicked_ids.add(news_id)

        if not candidates:
            continue

        # Build user state from the first row's history
        if vectors is not None:
            # ID-based format
            history_ids = group[0].get("history_ids", [])
            if not history_ids:
                continue
            user_state = _state_from_history_vec(history_ids, vectors, id_to_idx, history_size, int(vectors.shape[1])).tolist()
        else:
            history = group[0].get("history", [])
            if not history:
                continue
            user_state = [0.0] * ckpt.embedding_dim
            for vec in history:
                for i, v in enumerate(vec):
                    if i < len(user_state):
                        user_state[i] += v
            n = max(len(history), 1)
            user_state = [v / n for v in user_state]

        # Run planner
        try:
            best_node = planner.plan(user_state, candidates)
            anchor_path = best_node.path
        except Exception:
            continue

        if not anchor_path:
            continue

        anchor = anchor_path[0]
        final_items = intent_coverage_rerank(
            user_state, anchor, candidates,
            ranker=ckpt.ranker, top_k=top_k,
        )

        if not final_items:
            continue

        # Slate metrics
        slate_intents = [item.category for item in final_items[:top_k]]
        slate_vectors = [item.vector for item in final_items[:top_k]]
        coverage = intent_coverage_at_k(slate_intents, top_k)
        ils = intra_list_similarity_at_k(slate_vectors, top_k)

        total_coverage += coverage
        total_ils += ils

        # Hit rate: did any clicked item make it into the slate?
        slate_ids = {item.item_id for item in final_items[:top_k]}
        if slate_ids & clicked_ids:
            total_hits += 1

        total_slates += 1

    if total_slates == 0:
        return {"planner_slates": 0}

    return {
        "planner_slates": total_slates,
        f"planner_intent_coverage@{top_k}": round(total_coverage / total_slates, 4),
        f"planner_ils@{top_k}": round(total_ils / total_slates, 4),
        f"planner_hit_rate@{top_k}": round(total_hits / total_slates, 4),
    }


def evaluate_world_model_multi_step(
    checkpoint_path: Path,
    data_path: Path,
    *,
    vectors_path: Path | None = None,
    device: str,
    batch_size: int,
    n_steps: int = 3,
    history_size: int = 50,
) -> dict[str, float | int]:
    """Evaluate Neural ODE world model with multi-step prediction.

    Reports per-step MSE and cosine similarity to measure error accumulation.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    rows = load_jsonl(data_path)

    vectors = id_to_idx = None
    if vectors_path is not None and vectors_path.exists():
        vectors, id_to_idx = _load_vectors_npz(vectors_path)

    model_type = checkpoint.get("model_type", "linear")
    if model_type != "neural_ode":
        return {"error": "multi_step eval requires neural_ode model"}

    model = load_neural_ode_ckpt(str(checkpoint_path), device=device)
    model.eval()
    emb_dim = int(vectors.shape[1]) if vectors is not None else int(checkpoint.get("state_dim", 256))

    # Accumulate per-step metrics
    step_mses = [0.0] * n_steps
    step_cosines = [0.0] * n_steps
    step_drift_mses = [0.0] * n_steps
    step_drift_cosines = [0.0] * n_steps
    total_examples = 0

    with torch.no_grad():
        for batch_rows in batched(rows, batch_size):
            bs = len(batch_rows)
            # Build state and actions
            state = np.zeros((bs, emb_dim), dtype=np.float32)
            actions = np.zeros((bs, n_steps, emb_dim), dtype=np.float32)
            final_target = np.zeros((bs, emb_dim), dtype=np.float32)

            for b, row in enumerate(batch_rows):
                hids = row.get("history_ids", [])
                cids = row.get("clicked_ids", [])[:n_steps]
                if len(cids) < n_steps:
                    continue  # skip rows with insufficient clicks

                state[b] = _state_from_history_vec(hids, vectors, id_to_idx, history_size, emb_dim)
                for j, cid in enumerate(cids):
                    if cid in id_to_idx:
                        actions[b, j] = vectors[id_to_idx[cid]]
                final_target[b] = _state_from_history_vec(hids + cids, vectors, id_to_idx, history_size, emb_dim)

            state_t = torch.from_numpy(state).to(device)
            actions_t = torch.from_numpy(actions).to(device)
            target_t = torch.from_numpy(final_target).to(device)

            # Propagate step by step, record per-step error
            current = state_t
            for step_idx in range(n_steps):
                current = model.step_rk4(current, actions_t[:, step_idx, :])
                # State-level metrics
                step_mse = ((current - target_t) ** 2).mean(dim=-1).mean().item()
                step_cosine = F.cosine_similarity(current, target_t, dim=-1).mean().item()
                step_mses[step_idx] += step_mse * bs
                step_cosines[step_idx] += step_cosine * bs
                # Drift-level metrics
                drift_pred = current - state_t
                drift_true = target_t - state_t
                drift_mse = ((drift_pred - drift_true) ** 2).mean(dim=-1).mean().item()
                drift_cos = F.cosine_similarity(drift_pred, drift_true + 1e-8, dim=-1).mean().item()
                step_drift_mses[step_idx] += drift_mse * bs
                step_drift_cosines[step_idx] += drift_cos * bs

            total_examples += bs

    if total_examples == 0:
        return {"multi_step_examples": 0}

    result = {"multi_step_examples": total_examples, "n_steps": n_steps}
    for i in range(n_steps):
        result[f"step_{i+1}_mse"] = round(step_mses[i] / total_examples, 8)
        result[f"step_{i+1}_cosine"] = round(step_cosines[i] / total_examples, 4)
        result[f"step_{i+1}_drift_mse"] = round(step_drift_mses[i] / total_examples, 8)
        result[f"step_{i+1}_drift_cos"] = round(step_drift_cosines[i] / total_examples, 4)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved checkpoints on offline dev data.")
    parser.add_argument("--ranker-ckpt", type=Path, help="Path to a saved ranker checkpoint.")
    parser.add_argument("--ranker-data", type=Path, help="Path to ranker JSONL eval data.")
    parser.add_argument("--world-model-ckpt", type=Path, help="Path to a saved world-model checkpoint.")
    parser.add_argument("--world-model-data", type=Path, help="Path to world-model JSONL eval data.")
    parser.add_argument("--world-model-n-steps", type=int, default=1,
                        help="Multi-step evaluation for world model (default: 1=single).")
    parser.add_argument("--identity-baseline", action="store_true",
                        help="Evaluate identity baseline (predict no drift).")
    parser.add_argument("--news-features", type=Path, default=Path("data/processed/news_features_dev.jsonl"),
                        help="Path to news features JSONL (for planner eval).")
    parser.add_argument("--planner", action="store_true",
                        help="Run full planner-level slate evaluation.")
    parser.add_argument("--max-impressions", type=int, default=2000,
                        help="Max impressions for planner eval (default: 200).")
    parser.add_argument("--max-dev-impressions", type=int, default=0,
                        help="Max impressions for ranker eval (0=full dev set).")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--branching-factor", type=int, default=8)
    parser.add_argument("--vectors-path", type=Path,
                        help="Path to news_vectors_*.npz for ID-based embedding lookup.")
    parser.add_argument("--output", type=Path, help="Optional path to save metrics as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    if not any([args.ranker_ckpt, args.world_model_ckpt]):
        raise SystemExit("Provide at least one checkpoint to evaluate.")

    report: dict[str, dict[str, float | int | None]] = {}

    if args.ranker_ckpt or args.ranker_data:
        if not args.ranker_ckpt or not args.ranker_data:
            raise SystemExit("Ranker evaluation requires both --ranker-ckpt and --ranker-data.")
        report["ranker"] = evaluate_ranker(
            args.ranker_ckpt,
            args.ranker_data,
            vectors_path=args.vectors_path,
            device=device,
            batch_size=args.batch_size,
            top_k=args.top_k,
            max_impressions=args.max_dev_impressions,
        )

    if args.world_model_ckpt or args.world_model_data:
        if not args.world_model_ckpt or not args.world_model_data:
            raise SystemExit(
                "World-model evaluation requires both --world-model-ckpt and --world-model-data."
            )
        if args.identity_baseline:
            report["identity_baseline"] = evaluate_identity_baseline(
                args.world_model_data,
                vectors_path=args.vectors_path,
                n_steps=args.world_model_n_steps,
            )
        if args.world_model_n_steps > 1:
            # Multi-step evaluation
            report["world_model"] = evaluate_world_model_multi_step(
                args.world_model_ckpt,
                args.world_model_data,
                vectors_path=args.vectors_path,
                device=device,
                batch_size=args.batch_size,
                n_steps=args.world_model_n_steps,
            )
        else:
            report["world_model"] = evaluate_world_model(
                args.world_model_ckpt,
                args.world_model_data,
                vectors_path=args.vectors_path,
                device=device,
                batch_size=args.batch_size,
            )

    if args.planner:
        if not args.ranker_ckpt or not args.world_model_ckpt:
            raise SystemExit("Planner evaluation requires both --ranker-ckpt and --world-model-ckpt.")
        if not args.ranker_data:
            raise SystemExit("Planner evaluation requires --ranker-data.")
        if not args.news_features.exists():
            print(f"Warning: news features not found at {args.news_features}, skipping planner eval.")
        else:
            report["planner"] = evaluate_planner(
                ranker_ckpt=args.ranker_ckpt,
                world_model_ckpt=args.world_model_ckpt,
                ranker_data=args.ranker_data,
                news_features_path=args.news_features,
                vectors_path=args.vectors_path,
                device=device,
                top_k=args.top_k,
                horizon=args.horizon,
                beam_width=args.beam_width,
                branching_factor=args.branching_factor,
                max_impressions=args.max_impressions,
            )

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

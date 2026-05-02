from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rec.config import PlanningConfig
from agentic_rec.data.mind import prepare_mind_split
from agentic_rec.data.semantic_embeddings import (
    DEFAULT_QWEN3_EMBEDDING_MODEL,
    SemanticEmbeddingConfig,
)


VARIANT_TO_PREFIX = {
    "small": "MINDsmall",
    "large": "MINDlarge",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MIND raw TSV files into JSONL rows for ranker and world-model training."
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/mind"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--config", type=Path, default=Path("configs/base.json"))
    parser.add_argument("--variant", choices=sorted(VARIANT_TO_PREFIX), default="small")
    parser.add_argument("--negatives-per-positive", type=int, default=4)
    parser.add_argument("--max-train-behaviors", type=int, default=0)
    parser.add_argument("--max-dev-behaviors", type=int, default=0)
    parser.add_argument(
        "--embedding-backend",
        choices=["hash", "qwen3", "qwen3-lora", "sentence-transformers"],
        default="hash",
        help=(
            "Article vector builder. 'hash' is dependency-free; 'qwen3' uses "
            "SentenceTransformers; 'qwen3-lora' loads a fine-tuned PEFT adapter."
        ),
    )
    parser.add_argument("--embedding-model", default=DEFAULT_QWEN3_EMBEDDING_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-device", default="auto")
    parser.add_argument("--embedding-max-length", type=int, default=512)
    parser.add_argument("--embedding-cache", type=Path, default=None)
    parser.add_argument("--no-embedding-progress", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def resolve_split_paths(raw_root: Path, variant: str) -> dict[str, tuple[Path, Path]]:
    prefix = VARIANT_TO_PREFIX[variant]
    splits = {
        "train": raw_root / f"{prefix}_train",
        "dev": raw_root / f"{prefix}_dev",
    }
    resolved: dict[str, tuple[Path, Path]] = {}
    for split_name, split_dir in splits.items():
        news_path = split_dir / "news.tsv"
        behaviors_path = split_dir / "behaviors.tsv"
        if not news_path.exists() or not behaviors_path.exists():
            raise FileNotFoundError(
                f"Missing MIND files for split '{split_name}' under {split_dir}."
            )
        resolved[split_name] = (news_path, behaviors_path)
    return resolved


def main() -> None:
    args = parse_args()
    config = PlanningConfig.from_json_file(args.config)
    split_paths = resolve_split_paths(args.raw_root, args.variant)
    embedding_config = SemanticEmbeddingConfig(
        backend=args.embedding_backend,
        model_name=args.embedding_model,
        batch_size=args.embedding_batch_size,
        device=args.embedding_device,
        max_length=args.embedding_max_length,
        cache_path=args.embedding_cache,
        show_progress=not args.no_embedding_progress,
    )

    summary: dict[str, dict[str, int]] = {}
    for split_name, (news_path, behaviors_path) in split_paths.items():
        behavior_limit = (
            args.max_train_behaviors if split_name == "train" else args.max_dev_behaviors
        )
        summary[split_name] = prepare_mind_split(
            split_name=split_name,
            news_path=news_path,
            behaviors_path=behaviors_path,
            output_root=args.output_root,
            embedding_dim=config.embedding_dim,
            history_size=config.history_size,
            negatives_per_positive=args.negatives_per_positive,
            seed=args.seed,
            max_behaviors=None if behavior_limit == 0 else behavior_limit,
            embedding_config=embedding_config,
        )

    manifest = {
        "variant": args.variant,
        "embedding_dim": config.embedding_dim,
        "embedding_backend": embedding_config.backend,
        "embedding_model": embedding_config.model_name
        if embedding_config.backend != "hash"
        else None,
        "embedding_batch_size": embedding_config.batch_size
        if embedding_config.backend != "hash"
        else None,
        "embedding_max_length": embedding_config.max_length
        if embedding_config.backend != "hash"
        else None,
        "embedding_cache": str(embedding_config.cache_path)
        if embedding_config.backend != "hash" and embedding_config.cache_path
        else None,
        "history_size": config.history_size,
        "negatives_per_positive": args.negatives_per_positive,
        "max_train_behaviors": None
        if args.max_train_behaviors == 0
        else args.max_train_behaviors,
        "max_dev_behaviors": None if args.max_dev_behaviors == 0 else args.max_dev_behaviors,
        "splits": summary,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "mind_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

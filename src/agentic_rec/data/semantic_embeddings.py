from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from agentic_rec.core.linalg import Vector, normalize


DEFAULT_QWEN3_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


@dataclass(frozen=True, slots=True)
class SemanticEmbeddingConfig:
    backend: str = "hash"
    model_name: str = DEFAULT_QWEN3_EMBEDDING_MODEL
    batch_size: int = 16
    device: str = "auto"
    max_length: int = 512
    cache_path: Path | None = None
    show_progress: bool = True


def format_news_text(
    *,
    category: str,
    subcategory: str,
    title: str,
    abstract: str,
) -> str:
    fields = [
        ("Category", category),
        ("Subcategory", subcategory),
        ("Title", title),
        ("Abstract", abstract),
    ]
    return "\n".join(f"{name}: {value.strip()}" for name, value in fields if value.strip())


def encode_news_records(
    records: Sequence[Mapping[str, str]],
    *,
    output_dim: int,
    config: SemanticEmbeddingConfig,
) -> dict[str, Vector]:
    if output_dim <= 0:
        raise ValueError("output_dim must be positive.")

    cache = SemanticEmbeddingCache(config.cache_path) if config.cache_path else None
    vectors: dict[str, Vector] = {}
    pending_records: list[Mapping[str, str]] = []
    pending_texts: list[str] = []
    pending_keys: list[str] = []

    for record in records:
        news_id = record["news_id"]
        text = format_news_text(
            category=record["category"],
            subcategory=record["subcategory"],
            title=record["title"],
            abstract=record["abstract"],
        )
        key = semantic_cache_key(
            model_name=config.model_name,
            output_dim=output_dim,
            text=text,
        )
        cached_vector = cache.get(key) if cache else None
        if cached_vector is not None:
            vectors[news_id] = cached_vector
            continue

        pending_records.append(record)
        pending_texts.append(text)
        pending_keys.append(key)

    if pending_texts:
        if config.backend == "qwen3-lora":
            encoder = PeftTransformerNewsEncoder(config)
        else:
            encoder = SentenceTransformerNewsEncoder(config)
        encoded_vectors = encoder.encode_texts(pending_texts, output_dim=output_dim)
        for record, key, text, vector in zip(
            pending_records,
            pending_keys,
            pending_texts,
            encoded_vectors,
            strict=True,
        ):
            news_id = record["news_id"]
            vectors[news_id] = vector
            if cache:
                cache.append(
                    {
                        "key": key,
                        "news_id": news_id,
                        "model_name": config.model_name,
                        "embedding_dim": output_dim,
                        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "vector": vector,
                    }
                )

    return vectors


def semantic_cache_key(*, model_name: str, output_dim: int, text: str) -> str:
    payload = f"{model_name}\n{output_dim}\n{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SemanticEmbeddingCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._vectors = self._load()

    def get(self, key: str) -> Vector | None:
        vector = self._vectors.get(key)
        return list(vector) if vector is not None else None

    def append(self, record: Mapping[str, object]) -> None:
        key = str(record["key"])
        vector = [float(value) for value in record["vector"]]  # type: ignore[index]
        self._vectors[key] = vector
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(dict(record), handle, ensure_ascii=False)
            handle.write("\n")

    def _load(self) -> dict[str, Vector]:
        if not self.path.exists():
            return {}

        vectors: dict[str, Vector] = {}
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                key = record.get("key")
                vector = record.get("vector")
                if isinstance(key, str) and isinstance(vector, list):
                    vectors[key] = [float(value) for value in vector]
        return vectors


class SentenceTransformerNewsEncoder:
    def __init__(self, config: SemanticEmbeddingConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - runtime guidance only
            raise RuntimeError(
                "Install semantic embedding dependencies first: "
                "python3 -m pip install -e '.[semantic]'"
            ) from exc

        resolved_device = resolve_sentence_transformer_device(config.device)
        if resolved_device is None:
            self.model = SentenceTransformer(config.model_name)
        else:
            self.model = SentenceTransformer(config.model_name, device=resolved_device)
        self.config = config

        if config.max_length > 0 and hasattr(self.model, "max_seq_length"):
            self.model.max_seq_length = config.max_length

    def encode_texts(self, texts: Sequence[str], *, output_dim: int) -> list[Vector]:
        kwargs: dict[str, object] = {
            "batch_size": max(1, int(self.config.batch_size)),
            "convert_to_numpy": True,
            "normalize_embeddings": False,
            "show_progress_bar": self.config.show_progress,
        }
        if "truncate_dim" in inspect.signature(self.model.encode).parameters:
            kwargs["truncate_dim"] = output_dim

        raw_embeddings = self.model.encode(list(texts), **kwargs)
        rows = raw_embeddings.tolist() if hasattr(raw_embeddings, "tolist") else raw_embeddings
        return [normalize_embedding_row(row, output_dim=output_dim) for row in rows]


def normalize_embedding_row(row: Sequence[float], *, output_dim: int) -> Vector:
    if len(row) < output_dim:
        raise ValueError(
            f"Embedding row has width {len(row)}, but output_dim={output_dim} was requested."
        )

    vector = [float(value) for value in row[:output_dim]]
    return normalize(vector)


def resolve_sentence_transformer_device(requested_device: str) -> str | None:
    if requested_device != "auto":
        return requested_device

    try:
        import torch
    except ImportError:
        return None

    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return None


class PeftTransformerNewsEncoder:
    def __init__(self, config: SemanticEmbeddingConfig) -> None:
        try:
            import torch
            from peft import PeftConfig, PeftModel
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - runtime guidance only
            raise RuntimeError(
                "Install Qwen fine-tuning dependencies first: "
                "python3 -m pip install -e '.[qwen-train]'"
            ) from exc

        self.torch = torch
        self.config = config
        self.device = resolve_transformer_device(config.device)

        peft_config = PeftConfig.from_pretrained(config.model_name)
        base_model_name = peft_config.base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        dtype = torch.bfloat16 if self.device == "cuda" else None
        base_model = AutoModel.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        self.model = PeftModel.from_pretrained(base_model, config.model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode_texts(self, texts: Sequence[str], *, output_dim: int) -> list[Vector]:
        vectors: list[Vector] = []
        batch_size = max(1, int(self.config.batch_size))
        with self.torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = list(texts[start : start + batch_size])
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max(1, int(self.config.max_length)),
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                output = self.model(**encoded)
                embeddings = last_token_pool(output.last_hidden_state, encoded["attention_mask"])
                embeddings = self.torch.nn.functional.normalize(embeddings, p=2, dim=-1)
                for row in embeddings.detach().cpu().tolist():
                    vectors.append(normalize_embedding_row(row, output_dim=output_dim))
        return vectors


def resolve_transformer_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def last_token_pool(hidden_states, attention_mask):
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = hidden_states.new_tensor(
        range(hidden_states.size(0)),
        dtype=sequence_lengths.dtype,
    )
    return hidden_states[batch_indices, sequence_lengths]

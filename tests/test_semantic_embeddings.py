from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic_rec.data.semantic_embeddings import (
    SemanticEmbeddingCache,
    format_news_text,
    normalize_embedding_row,
    semantic_cache_key,
)


class SemanticEmbeddingTests(unittest.TestCase):
    def test_format_news_text_keeps_title_and_abstract(self) -> None:
        text = format_news_text(
            category="news",
            subcategory="local",
            title="City council approves transit plan",
            abstract="The plan adds bus lanes downtown.",
        )

        self.assertIn("Title: City council approves transit plan", text)
        self.assertIn("Abstract: The plan adds bus lanes downtown.", text)

    def test_normalize_embedding_row_truncates_to_requested_dim(self) -> None:
        vector = normalize_embedding_row([3.0, 4.0, 12.0], output_dim=2)

        self.assertAlmostEqual(vector[0], 0.6)
        self.assertAlmostEqual(vector[1], 0.8)

    def test_semantic_cache_round_trips_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cache.jsonl"
            key = semantic_cache_key(
                model_name="Qwen/Qwen3-Embedding-0.6B",
                output_dim=2,
                text="Title: Example",
            )
            cache = SemanticEmbeddingCache(path)
            cache.append(
                {
                    "key": key,
                    "news_id": "N1",
                    "model_name": "Qwen/Qwen3-Embedding-0.6B",
                    "embedding_dim": 2,
                    "text_sha256": "demo",
                    "vector": [0.6, 0.8],
                }
            )

            reloaded = SemanticEmbeddingCache(path)
            self.assertEqual(reloaded.get(key), [0.6, 0.8])


if __name__ == "__main__":
    unittest.main()

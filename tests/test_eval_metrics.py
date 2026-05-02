from __future__ import annotations

import unittest

from agentic_rec.eval.metrics import (
    binary_accuracy,
    binary_auc,
    grouped_ranking_metrics,
    intent_coverage_at_k,
    intra_list_similarity_at_k,
)


class EvalMetricsTests(unittest.TestCase):
    def test_binary_auc_is_one_for_perfect_ranking(self) -> None:
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        auc = binary_auc(labels, scores)
        self.assertIsNotNone(auc)
        self.assertAlmostEqual(auc or 0.0, 1.0, places=6)

    def test_binary_accuracy_thresholds_scores(self) -> None:
        labels = [0, 1, 1, 0]
        scores = [0.2, 0.7, 0.8, 0.4]
        self.assertAlmostEqual(binary_accuracy(labels, scores), 1.0, places=6)

    def test_grouped_ranking_metrics_returns_recall_and_ndcg(self) -> None:
        rows = [
            {"impression_id": "1", "label": 1},
            {"impression_id": "1", "label": 0},
            {"impression_id": "2", "label": 0},
            {"impression_id": "2", "label": 1},
        ]
        scores = [0.9, 0.1, 0.2, 0.8]
        metrics = grouped_ranking_metrics(rows, scores, top_k=1)

        self.assertEqual(metrics["impressions"], 2)
        self.assertAlmostEqual(float(metrics["recall@1"] or 0.0), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["ndcg@1"] or 0.0), 1.0, places=6)

    def test_intent_coverage_at_k_counts_unique_intents(self) -> None:
        self.assertAlmostEqual(
            intent_coverage_at_k(["tech", "tech", "sports", "science"], 4),
            0.75,
            places=6,
        )

    def test_intra_list_similarity_at_k_averages_pairwise_similarity(self) -> None:
        score = intra_list_similarity_at_k(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            3,
        )
        self.assertAlmostEqual(score, 1.0 / 3.0, places=6)


if __name__ == "__main__":
    unittest.main()

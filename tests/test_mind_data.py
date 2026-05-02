from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentic_rec.data.mind import prepare_mind_split


class MindPreparationTests(unittest.TestCase):
    def test_prepare_mind_split_writes_ranker_and_world_model_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            news_path = root / "news.tsv"
            behaviors_path = root / "behaviors.tsv"
            output_root = root / "processed"

            news_path.write_text(
                "\n".join(
                    [
                        "N1\tnews\tlocal\tTitle One\tAbstract One\turl\t[]\t[]",
                        "N2\tnews\tglobal\tTitle Two\tAbstract Two\turl\t[]\t[]",
                        "N3\tsports\tlocal\tTitle Three\tAbstract Three\turl\t[]\t[]",
                    ]
                ),
                encoding="utf-8",
            )
            behaviors_path.write_text(
                "\n".join(
                    [
                        "1\tU1\t11/10/2019 10:00:00 AM\tN1\tN2-1 N3-0",
                        "2\tU2\t11/10/2019 11:00:00 AM\t\tN1-0 N2-1",
                    ]
                ),
                encoding="utf-8",
            )

            summary = prepare_mind_split(
                split_name="train",
                news_path=news_path,
                behaviors_path=behaviors_path,
                output_root=output_root,
                embedding_dim=8,
                history_size=3,
                negatives_per_positive=1,
                seed=7,
            )

            self.assertEqual(summary["behaviors"], 2)
            self.assertGreaterEqual(summary["ranker_examples"], 4)
            self.assertEqual(summary["world_model_examples"], 2)

            ranker_rows = [
                json.loads(line)
                for line in (output_root / "ranker_train.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            world_rows = [
                json.loads(line)
                for line in (output_root / "world_model_train.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

            self.assertEqual(len(ranker_rows[0]["history"]), 3)
            self.assertEqual(len(ranker_rows[0]["candidate"]), 8)
            self.assertIn("category", ranker_rows[0])
            self.assertEqual(len(world_rows[0]["state"]), 8)
            self.assertEqual(len(world_rows[0]["action"]), 8)
            self.assertEqual(len(world_rows[0]["next_state"]), 8)

    def test_prepare_mind_split_respects_behavior_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            news_path = root / "news.tsv"
            behaviors_path = root / "behaviors.tsv"
            output_root = root / "processed"

            news_path.write_text(
                "\n".join(
                    [
                        "N1\tnews\tlocal\tTitle One\tAbstract One\turl\t[]\t[]",
                        "N2\tbusiness\tglobal\tTitle Two\tAbstract Two\turl\t[]\t[]",
                    ]
                ),
                encoding="utf-8",
            )
            behaviors_path.write_text(
                "\n".join(
                    [
                        "1\tU1\t11/10/2019 10:00:00 AM\tN1\tN2-1",
                        "2\tU2\t11/10/2019 11:00:00 AM\tN2\tN1-1",
                        "3\tU3\t11/10/2019 12:00:00 PM\tN1\tN2-1",
                    ]
                ),
                encoding="utf-8",
            )

            summary = prepare_mind_split(
                split_name="train",
                news_path=news_path,
                behaviors_path=behaviors_path,
                output_root=output_root,
                embedding_dim=8,
                history_size=3,
                negatives_per_positive=1,
                seed=7,
                max_behaviors=2,
            )

            self.assertEqual(summary["behaviors"], 2)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import math
import unittest

from agentic_rec.core.linalg import cosine_similarity, dot, matvec


class LinalgTests(unittest.TestCase):
    def test_dot(self) -> None:
        self.assertEqual(dot([1.0, 2.0], [3.0, 4.0]), 11.0)

    def test_cosine_similarity(self) -> None:
        score = cosine_similarity([1.0, 0.0], [1.0, 1.0])
        self.assertTrue(math.isclose(score, math.sqrt(0.5), rel_tol=1e-6))

    def test_matvec(self) -> None:
        result = matvec([[1.0, 2.0], [0.0, -1.0]], [3.0, 4.0])
        self.assertEqual(result, [11.0, -4.0])


if __name__ == "__main__":
    unittest.main()

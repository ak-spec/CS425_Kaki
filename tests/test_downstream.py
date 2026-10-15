import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prep_downstream import build_rows, validate_splits


class DataTests(unittest.TestCase):
    def test_preserves_strategies_and_no_future_feedback(self):
        dialogue = {
            "dialog": [
                {"speaker": "seeker", "content": "exam worry", "annotation": {}},
                {
                    "speaker": "supporter",
                    "content": "I hear you",
                    "annotation": {"strategy": "Reflection of feelings"},
                },
                {
                    "speaker": "supporter",
                    "content": "What worries you?",
                    "annotation": {"strategy": "Question"},
                },
                {
                    "speaker": "seeker",
                    "content": "future secret",
                    "annotation": {"feedback": "4"},
                },
            ]
        }
        rows, pairs, stats = build_rows(1, dialogue, {})
        self.assertEqual(
            [r["strategy"] for r in rows], ["Reflection of feelings", "Question"]
        )
        self.assertNotIn("I hear you", rows[0]["text"])
        self.assertIn("I hear you", rows[1]["text"])
        self.assertNotIn("future secret", pairs[0]["text"] + pairs[0]["response"])
        self.assertEqual(pairs[0]["response"], "I hear you What worries you?")
        self.assertEqual(len(pairs), 1)

    def test_leakage_and_missing_ids_rejected(self):
        for split in [
            dict(train=[0], val=[0], test=[1]),
            dict(train=[0], val=[], test=[]),
        ]:
            with self.assertRaises(ValueError):
                validate_splits(split, [0, 1])
        validate_splits(dict(train=[0], val=[], test=[1]), [0, 1])

    def test_unpaired_feedback_not_invented(self):
        _, pairs, counts = build_rows(
            1,
            {
                "dialog": [
                    {
                        "speaker": "seeker",
                        "content": "hello",
                        "annotation": {"feedback": "5"},
                    }
                ]
            },
            {},
        )
        self.assertEqual(pairs, [])
        self.assertEqual(counts["unpaired_feedback"], 1)

    def test_initial_supporter_and_unknown_strategy_are_excluded(self):
        rows, _, counts = build_rows(
            0,
            {
                "dialog": [
                    {
                        "speaker": "supporter",
                        "content": "Hello",
                        "annotation": {"strategy": "Question"},
                    },
                    {"speaker": "seeker", "content": "Hi", "annotation": {}},
                    {
                        "speaker": "supporter",
                        "content": "Reply",
                        "annotation": {"strategy": "invalid"},
                    },
                ]
            },
            {},
        )
        self.assertEqual(rows, [])
        self.assertEqual(counts["empty_history"], 1)
        self.assertEqual(counts["unsupported_strategy"], 1)


if __name__ == "__main__":
    unittest.main()

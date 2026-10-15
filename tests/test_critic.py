"""Critic data split and evaluation helpers; no model is loaded.

Run from the repository root: python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from critic import to_rating, to_target
from train_critic import evaluate, other_replies, pair_accuracy, rating_pairs, split_halves


def row(dialogue, start, rating, reply="ok"):
    return {"dialogue_id": dialogue, "response_start_idx": start, "feedback": rating, "text": "seeker: hi",
            "response": reply}


ROWS = [row(1, 3, 3), row(1, 7, 5), row(1, 11, 4), row(2, 2, 5), row(2, 6, 2), row(3, 4, 4)]


class CriticHelperTests(unittest.TestCase):
    def test_halves_are_disjoint_complete_and_reproducible(self):
        ids = list(range(100)) * 3
        a, b = split_halves(ids, 42), split_halves(ids, 42)
        self.assertEqual(a, b)
        self.assertFalse(set(a["a"]) & set(a["b"]))
        self.assertEqual(sorted(a["a"] + a["b"]), list(range(100)))
        self.assertNotEqual(split_halves(ids, 1), a)

    def test_pairs_stay_within_a_dialogue(self):
        self.assertEqual(sorted(rating_pairs(ROWS)), [(1, 0), (1, 2), (2, 0), (3, 4)])

    def test_pair_accuracy_counts_ties_as_half(self):
        self.assertEqual(pair_accuracy([(0, 1), (2, 3)], [2.0, 1.0, 1.0, 1.0]), 0.75)

    def test_other_reply_comes_from_another_dialogue(self):
        rows = [row(d, 0, 4, reply=f"reply from {d}") for d in (1, 1, 2, 3, 3, 4)]
        for r, other in zip(rows, other_replies(rows, seed=0)):
            self.assertNotEqual(other, f"reply from {r['dialogue_id']}")

    def test_against_trend_pairs_and_baselines(self):
        preds = [3.0, 4.0, 5.0, 5.0, 1.0, 4.0]
        m = evaluate(ROWS, preds, [0.5] * len(ROWS), train_mean=4.0)
        self.assertEqual((m["pairs"], m["against_trend_pairs"]), (4, 2))  # (1,2) and (3,4) go against position
        self.assertEqual(m["pairwise_position_baseline"], 0.5)
        self.assertEqual(m["against_trend_accuracy"], 0.5)
        self.assertEqual(m["beats_other_reply"], 1.0)

    def test_rating_scale_round_trip(self):
        for rating in (1, 3, 5):
            self.assertAlmostEqual(to_rating(to_target(rating)), rating)


if __name__ == "__main__":
    unittest.main()

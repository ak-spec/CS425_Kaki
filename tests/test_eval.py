"""Evaluation tooling tests with tiny inputs and stand-in models; nothing is downloaded.

Run from the repository root: python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from select_eval_contexts import eligible, history_turns, select


def response_row(dialogue, idx, text, problem="academic pressure", in_scope=True):
    return {"dialogue_id": dialogue, "utterance_idx": idx, "text": text, "response": "ref",
            "strategy": "Question", "problem": problem, "in_scope": in_scope}


TWO_SEEKER = ("seeker: hi | supporter: Hello, what is on your mind? | seeker: my exams are next week and I "
              "cannot sleep because I keep thinking that I will fail all of them")


class SelectionTests(unittest.TestCase):
    def test_history_turns_round_trip(self):
        turns = history_turns(TWO_SEEKER)
        self.assertEqual([t["speaker"] for t in turns], ["seeker", "supporter", "seeker"])
        self.assertEqual(" | ".join(f"{t['speaker']}: {t['text']}" for t in turns), TWO_SEEKER)
        with self.assertRaises(ValueError):
            history_turns("no tags here")

    def test_eligibility(self):
        self.assertTrue(eligible(response_row(1, 3, TWO_SEEKER)))
        self.assertFalse(eligible(response_row(1, 1, "seeker: hi")))  # no earlier seeker message
        self.assertFalse(eligible(response_row(1, 4, TWO_SEEKER + " | supporter: ok")))  # ends with supporter
        self.assertFalse(eligible(response_row(1, 3, TWO_SEEKER), max_words=20))
        self.assertFalse(eligible(response_row(1, 3, TWO_SEEKER), min_seeker_words=30))

    def test_selection_is_balanced_distinct_and_reproducible(self):
        rows = []
        for d in range(30):
            problem = ["academic pressure", "problems with friends", "job crisis"][d % 3]
            rows += [response_row(d, i, TWO_SEEKER, problem) for i in (3, 5)]
        rows.append(response_row(99, 3, TWO_SEEKER, "other", in_scope=False))
        picks = select(rows, n=20, seed=1)
        self.assertEqual(picks, select(rows, n=20, seed=1))
        self.assertEqual(len({p["dialogue_id"] for p in picks}), 20)
        problems = [rows[2 * p["dialogue_id"]]["problem"] for p in picks]
        self.assertEqual(problems.count("academic pressure"), 10)
        self.assertEqual(problems.count("problems with friends"), 10)
        self.assertNotEqual(picks, select(rows, n=20, seed=2))

    def test_selection_fills_from_other_in_scope_problems(self):
        rows = [response_row(d, 3, TWO_SEEKER, "academic pressure" if d < 2 else "job crisis") for d in range(4)]
        rows.append(response_row(9, 3, TWO_SEEKER, "other", in_scope=False))
        picks = select(rows, n=4, seed=0)
        self.assertEqual(sorted(p["dialogue_id"] for p in picks), [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            select(rows, n=5, seed=0)


if __name__ == "__main__":
    unittest.main()

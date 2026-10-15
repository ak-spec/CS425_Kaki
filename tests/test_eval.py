"""Evaluation tooling tests with tiny inputs and stand-in models; nothing is downloaded.

Run from the repository root: python -m unittest discover -s tests -v
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generate_outputs import generate, load_rows, row_turns
from pipeline import CONFIGS, Kaki
from prep_downstream import history
from select_eval_contexts import eligible, history_turns, read_jsonl, select


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



class FakeGuard:
    def check(self, turns):
        flagged = "crisis" in turns[-1]["text"]
        return {"bypass_generation": flagged, "response_key": "crisis_sg" if flagged else None}

    def response(self, key):
        return "prepared"


class FakePerceiver:
    def predict(self, history):
        return {"emotion": "anxiety", "problem": "academic pressure"}


class FakePlanner:
    def predict(self, history):
        return [{"strategy": "Question", "probability": 0.5},
                {"strategy": "Reflection of feelings", "probability": 0.3},
                {"strategy": "Others", "probability": 0.1}]


class FakeSpeaker:
    def generate(self, history, emotion=None, strategy=None):
        return f"{strategy or 'plain'} reply to: {history.split(': ')[-1]}"

    def generate_candidates(self, history, emotion, strategies):
        return [{"strategy": s["strategy"], "planner_probability": s["probability"],
                 "response": self.generate(history, emotion, s["strategy"])} for s in strategies]


def fake_kaki():
    """The real Kaki.respond with stand-in components, so outputs have the pipeline's exact shape."""
    return Kaki(FakeGuard(), FakePerceiver(), FakePlanner(), {"history": FakeSpeaker(), "conditioned": FakeSpeaker()})


RAW = [{"dialog": [{"speaker": "seeker", "content": "hi"},
                   {"speaker": "supporter", "content": "Hello,  how are you?"},
                   {"speaker": "seeker", "content": "  my exams are soon"},
                   {"speaker": "supporter", "content": "That sounds stressful."},
                   {"speaker": "supporter", "content": "Which subject?"},
                   {"speaker": "seeker", "content": "I am in crisis"},
                   {"speaker": "supporter", "content": "Please reach out."}]}]


def test_rows():
    dialog = RAW[0]["dialog"]
    return [{"dialogue_id": 0, "utterance_idx": i, "text": history(dialog[:i]), "response": dialog[i]["content"],
             "strategy": "Question", "problem": "academic pressure", "emotion": "anxiety"} for i in (1, 3, 4, 6)]


class GenerateTests(unittest.TestCase):
    def test_turns_from_raw_match_turns_from_history(self):
        for row in test_rows():
            self.assertEqual(row_turns(row, RAW), row_turns(row))
        self.assertEqual(row_turns(test_rows()[1], RAW)[-1], {"speaker": "seeker", "text": "my exams are soon"})
        broken = dict(test_rows()[1], text="seeker: something else")
        with self.assertRaises(ValueError):
            row_turns(broken, RAW)

    def test_generate_writes_every_configuration_and_resumes(self):
        rows = test_rows()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            out = Path(tmp) / "outputs"
            self.assertEqual(generate(fake_kaki(), rows, CONFIGS, out, RAW), dict.fromkeys(CONFIGS, 4))
            lines = {c: read_jsonl(out / f"{c}.jsonl") for c in CONFIGS}
            self.assertTrue((out / ".gitignore").exists())
            direct, planning, lookahead = lines["direct"][1], lines["planning"][1], lines["lookahead"][1]
            self.assertEqual((direct["dialogue_id"], direct["utterance_idx"]), (0, 3))
            self.assertEqual(direct["reference"], "That sounds stressful.")
            self.assertEqual((direct["reply"], direct["strategy"]), ("plain reply to: my exams are soon", None))
            self.assertEqual((planning["strategy"], len(planning["candidates"])), ("Question", 1))
            self.assertEqual((lookahead["selected"], len(lookahead["candidates"])), (0, 3))
            self.assertEqual(lookahead["perceived"], {"emotion": "anxiety", "problem": "academic pressure"})
            self.assertEqual([r["after_seeker"] for r in lines["direct"]], [True, True, False, True])
            crisis = lines["lookahead"][3]
            self.assertEqual((crisis["bypassed"], crisis["reply"], crisis["strategy"]), (True, "prepared", None))
            self.assertEqual(generate(fake_kaki(), rows, CONFIGS, out, RAW), dict.fromkeys(CONFIGS, 0))
            generate(fake_kaki(), rows[:2], ["direct"], out, RAW, overwrite=True)
            self.assertEqual(len(read_jsonl(out / "direct.jsonl")), 2)

    def test_load_rows_follows_contexts(self):
        rows = test_rows()
        with tempfile.TemporaryDirectory() as tmp:
            data, contexts = Path(tmp) / "responses.jsonl", Path(tmp) / "contexts.json"
            data.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            contexts.write_text(json.dumps({"contexts": [{"dialogue_id": 0, "utterance_idx": 4},
                                                         {"dialogue_id": 0, "utterance_idx": 1}]}), encoding="utf-8")
            self.assertEqual([r["utterance_idx"] for r in load_rows(data, contexts)], [4, 1])
            self.assertEqual(len(load_rows(data, limit=3)), 3)
            contexts.write_text(json.dumps({"contexts": [{"dialogue_id": 5, "utterance_idx": 1}]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_rows(data, contexts)


if __name__ == "__main__":
    unittest.main()

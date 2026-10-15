"""Tests for the baselines, controllability, simulator and benchmark helpers; no model is downloaded.

Run from the repository root: python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baselines import detokenize
from benchmark import parameters
from controllability import agreement, confusion, ends_with_seeker
from simulate import (CLOSING, OPENING, clean, converse, per_chat, rating, repeats, seeker_messages,
                      situations, summarise)
from test_eval import fake_kaki, sheet_inputs, TWO_SEEKER
import json
import tempfile
from ablations import inputs as ablation_inputs
from human_eval_sheet import CRITERIA, build as build_sheet, query_outputs, query_rows
from judge import parse_scores, pick_contexts, prompt
from realism_sheet import build as build_realism, read_guesses, real_openers, score as score_realism

SESSION = {"dialogue_id": 7, "situation": "I failed a quiz.", "emotion": "sadness", "problem": "academic pressure"}


class FakeSeeker:
    """Scripted seeker: says something about exams, ends on its third message, rates 4."""

    def __init__(self):
        self.calls = []

    def __call__(self, chats, seed, sample=True, max_new_tokens=None):
        self.calls.append((len(chats), seed, sample))
        if not sample:
            return ["4" for _ in chats]
        return [f"Seeker: message {sum(m['role'] == 'assistant' for m in c) + 1} about exams"
                + (" bye [END]" if sum(m["role"] == "assistant" for m in c) >= 2 else "") for c in chats]


class BaselineTests(unittest.TestCase):
    def test_detokenize_blenderbot_output(self):
        self.assertEqual(detokenize("i ' m sorry to hear that . don ' t worry , it ' s ok ! have you tried ?"),
                         "I'm sorry to hear that. Don't worry, it's ok! Have you tried?")
        self.assertEqual(detokenize(""), "")


class ControllabilityTests(unittest.TestCase):
    def test_agreement_and_confusion(self):
        asked = ["Question", "Question", "Others", "Others"]
        tagged = ["Question", "Others", "Others", "Others"]
        result = agreement(asked, tagged)
        self.assertEqual(result["overall"], 0.75)
        self.assertEqual(result["per_strategy"], {"Others": 1.0, "Question": 0.5})
        self.assertEqual(confusion(asked, tagged, ["Question", "Others"]), [[1, 1], [0, 2]])

    def test_ends_with_seeker(self):
        self.assertTrue(ends_with_seeker("supporter: hi | seeker: I am sad"))
        self.assertFalse(ends_with_seeker("seeker: I am sad | supporter: why?"))


class SimulatorTests(unittest.TestCase):
    def test_situations_use_only_held_out_dialogues_and_are_reproducible(self):
        raw = [{"situation": f" s{i} ", "emotion_type": "Anxiety", "problem_type": "Job crisis"} for i in range(10)]
        splits = {"train": [0, 1, 2, 3], "val": [4, 5, 6], "test": [7, 8, 9]}
        picked = situations(raw, splits, 4, seed=1)
        self.assertEqual(picked, situations(raw, splits, 4, seed=1))
        self.assertTrue(all(p["dialogue_id"] >= 4 for p in picked))
        self.assertEqual(picked[0]["emotion"], "anxiety")
        self.assertEqual(picked[0]["situation"], f"s{picked[0]['dialogue_id']}")

    def test_seeker_sees_its_own_turns_as_assistant(self):
        turns = [{"speaker": "seeker", "text": "hi"}, {"speaker": "supporter", "text": "hello"}]
        msgs = seeker_messages(SESSION, turns, closing=True)
        self.assertIn("I failed a quiz.", msgs[0]["content"])
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "user", "user"])
        self.assertEqual((msgs[1]["content"], msgs[-1]["content"]), (OPENING, CLOSING))

    def test_clean_and_rating(self):
        self.assertEqual(clean('Seeker: "Thanks, that helps. Bye! [END]"'), ("Thanks, that helps. Bye!", True))
        self.assertEqual(clean("ok\nSupporter: I hear you"), ("ok", False))
        self.assertEqual(clean("[end]"), ("", True))
        self.assertEqual(rating("I'd say 4."), 4)
        self.assertIsNone(rating("not sure"))

    def test_repetition_counts_reused_trigrams(self):
        self.assertEqual(repeats(["that sounds really hard for you", "that sounds really hard for you today",
                                  "what happened at school", "ok"]), [False, True, False, False])

    def test_conversation_loop_with_stand_ins(self):
        seeker = FakeSeeker()
        sessions = [SESSION, {**SESSION, "dialogue_id": 8}]
        openers = ["Seeker: I failed my quiz", "I am in crisis"]
        chats = converse(fake_kaki(), seeker, sessions, openers, "lookahead", max_turns=6, seed=1)
        first, second = chats
        self.assertEqual(first["ended_by"], "seeker")
        self.assertEqual([t["speaker"] for t in first["turns"]],
                         ["seeker", "supporter", "seeker", "supporter", "seeker"])
        self.assertEqual(first["turns"][0]["text"], "I failed my quiz")
        self.assertEqual(first["turns"][-1]["text"], "message 3 about exams bye")
        self.assertEqual(len(first["replies"]), 2)
        self.assertEqual(first["replies"][0]["strategy"], "Question")
        self.assertEqual(first["replies"][0]["history"], "seeker: I failed my quiz")
        self.assertEqual(second["ended_by"], "guard")  # FakeGuard flags "crisis"
        self.assertEqual(second["turns"][-1]["text"], "prepared")
        self.assertEqual([c["seeker_rating"] for c in chats], [4, 4])
        self.assertEqual(seeker.calls[-1], (2, 0, False))  # one closing question for every chat

        for c in chats:
            for r in c["replies"]:
                r["eval_critic"] = 3.5
        summary = summarise(chats)
        self.assertEqual(summary["ended_by_seeker"], 0.5)
        self.assertEqual(summary["ended_by_guard"], 0.5)
        self.assertEqual(summary["kaki_replies"], 2)
        self.assertEqual(summary["eval_critic"], 3.5)
        self.assertEqual(summary["seeker_rating"], 4)
        self.assertEqual(per_chat(chats, "eval_critic"), {7: 3.5})
        self.assertEqual(per_chat(chats, "seeker_rating"), {7: 4, 8: 4})

    def test_opening_message_cannot_end_the_chat(self):
        chats = converse(fake_kaki(), FakeSeeker(), [SESSION], ["hi, thanks for being here [END]"], "direct",
                         max_turns=2, seed=1)
        self.assertEqual(chats[0]["turns"][0]["text"], "hi, thanks for being here")
        self.assertEqual(len(chats[0]["replies"]), 2)

    def test_turn_limit(self):
        chats = converse(fake_kaki(), FakeSeeker(), [SESSION], ["hello"], "direct", max_turns=2, seed=1)
        self.assertEqual(chats[0]["ended_by"], "turn_limit")
        self.assertEqual(len(chats[0]["replies"]), 2)


class BenchmarkTests(unittest.TestCase):
    def test_shared_parameters_count_once(self):
        a, b = torch.nn.Linear(3, 2), torch.nn.Linear(2, 2)
        self.assertEqual(parameters(a, b, a), 8 + 6)


class HumanEvalExtensionTests(unittest.TestCase):
    def test_four_systems_get_labels_a_to_d_and_five_criteria(self):
        contexts, outputs = sheet_inputs()
        outputs["blenderbot"] = {k: {"reply": f"blenderbot {k[0]}"} for k in outputs["direct"]}
        sheet, key = build_sheet(contexts, outputs, seed=1)
        self.assertEqual(sorted(key["contexts"][0]["labels"]), ["A", "B", "C", "D"])
        self.assertIn("safety", CRITERIA)
        self.assertIn("safety", sheet[0])
        outputs["extra"] = outputs["direct"]
        with self.assertRaises(ValueError):
            build_sheet(contexts, outputs)

    def test_handwritten_queries_become_contexts(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = Path(tmp) / "q.json"
            q.write_text(json.dumps({"queries": [{"id": "h1", "text": "exams  lah"}]}), encoding="utf-8")
            r = Path(tmp) / "r.json"
            r.write_text(json.dumps([{"system": "direct", "replies": [{"id": "h1", "reply": "hi"}]}]), encoding="utf-8")
            self.assertEqual(query_rows(q), [{"dialogue_id": "h1", "utterance_idx": 0, "text": "seeker: exams lah"}])
            self.assertEqual(query_outputs(r), {"direct": {("h1", 0): {"reply": "hi"}}})
            sheet, key = build_sheet(query_rows(q), {"direct": {("h1", 0): {"reply": "hi"}},
                                                     "planning": {("h1", 0): {"reply": "hello"}}})
            self.assertEqual(sheet[0]["history"], "Seeker: exams lah")


class JudgeTests(unittest.TestCase):
    def test_parse_scores(self):
        answer = '{"relevance": 4, "acknowledgement": 3, "helpfulness": "4", "fluency": 5, "safety": 5}'
        self.assertEqual(parse_scores(answer), {"relevance": 4, "acknowledgement": 3, "helpfulness": 4,
                                                "fluency": 5, "safety": 5})
        self.assertIsNone(parse_scores('{"relevance": 4}'))
        self.assertIsNone(parse_scores("no idea"))

    def test_prompt_shows_the_history_and_reply(self):
        msgs = prompt(TWO_SEEKER, "That sounds hard.")
        self.assertIn("Seeker: hi\nSupporter: Hello", msgs[1]["content"])
        self.assertIn("That sounds hard.", msgs[1]["content"])

    def test_contexts_exclude_bypassed_supporter_last_and_human_eval(self):
        def row(after, bypassed):
            return {"after_seeker": after, "bypassed": bypassed}
        base = {(1, 1): row(True, False), (2, 1): row(True, True), (3, 1): row(False, False), (4, 1): row(True, False)}
        rows = {s: dict(base) for s in ("direct", "planning", "lookahead", "blenderbot", "zeroshot")}
        self.assertEqual(pick_contexts(rows, exclude={(4, 1)}, n=5, seed=0), [(1, 1)])


class AblationTests(unittest.TestCase):
    def test_each_variant_changes_one_input(self):
        rows = [{"dialogue_id": 1, "utterance_idx": 2, "perceived": {"emotion": "fear"}, "emotion": "sadness",
                 "strategy": "Question", "reference_strategy": "Others"}]
        got = ablation_inputs(rows, {(1, 2): "seeker: hi"}, "anxiety")
        self.assertEqual(got["planning"], [("seeker: hi", "fear", "Question")])
        self.assertEqual(got["oracle_strategy"], [("seeker: hi", "fear", "Others")])
        self.assertEqual(got["oracle_emotion"], [("seeker: hi", "sadness", "Question")])
        self.assertEqual(got["constant_emotion"], [("seeker: hi", "anxiety", "Question")])


class RealismTests(unittest.TestCase):
    def test_build_score_and_openers(self):
        raw = [{"dialog": [{"speaker": "seeker", "content": "hi"},
                           {"speaker": "seeker", "content": "I lost my job and I do not know what to do next"}]}]
        self.assertEqual(real_openers(raw, [0]), ["I lost my job and I do not know what to do next"])
        sheet, key = build_realism([f"real {i}" for i in range(12)], [f"sim {i}" for i in range(12)], n=3, seed=1)
        self.assertEqual(len(sheet), 6)
        self.assertEqual(sorted(key["answers"].values()), ["real"] * 3 + ["simulated"] * 3)
        perfect = {i: a for i, a in key["answers"].items()}
        wrong = {i: "real" for i in key["answers"]}
        result = score_realism([perfect, wrong], key)
        self.assertEqual(result["accuracy_per_rater"], [1, 0.5])
        self.assertEqual(result["simulated_called_real"], 0.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.csv"
            path.write_text("item,message,guess\n1,x,Real\n2,y,maybe\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_guesses(path)


if __name__ == "__main__":
    unittest.main()

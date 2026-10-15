"""Demo tests: the fake Kaki, and app.py run with Streamlit's AppTest. No checkpoints or models are loaded.

Run from the repository root: python -m unittest discover -s tests -v
"""
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest

from app import fake_kaki, missing_files, model_set
from guard import decide
from pipeline import CONFIGS
from prep_downstream import STRATEGIES

APP = str(ROOT / "app.py")
CRISIS = json.loads((ROOT / "configs/guard.json").read_text(encoding="utf-8"))["responses"]["crisis_sg"]["text"]
ORDINARY = "I failed my midterm and feel awful."
RISKY = "I want to kill myself."
KEYS = {"config", "reply", "bypassed", "guard", "perceive", "strategies", "candidates", "selected"}
GUARD_KEYS = set(decide(0.0, [], {"model": 0.5, "lexicon": 0.3}))
CANDIDATE_KEYS = {"strategy", "planner_probability", "response", "critic_score", "score"}


def seeker(text):
    return [{"speaker": "seeker", "text": text}]


class FakeKakiTests(unittest.TestCase):
    def setUp(self):
        self.kaki = fake_kaki()

    def test_output_has_the_pipeline_shape_for_every_config(self):
        for config in CONFIGS:
            with self.subTest(config=config):
                out = self.kaki.respond(seeker(ORDINARY), config)
                self.assertEqual(set(out), KEYS)
                self.assertEqual(set(out["guard"]), GUARD_KEYS)
                self.assertFalse(out["bypassed"])
                p = out["perceive"]
                for label, probs in ((p["emotion"], p["emotion_probs"]), (p["problem"], p["problem_probs"])):
                    self.assertEqual(len(probs), 8)
                    self.assertAlmostEqual(sum(probs.values()), 1.0)
                    self.assertEqual(label, max(probs, key=probs.get))
                if config == "direct":
                    self.assertIsNone(out["strategies"])
                else:
                    self.assertEqual(len(out["strategies"]), 3)
                    self.assertTrue(all(s["strategy"] in STRATEGIES for s in out["strategies"]))
                self.assertEqual(len(out["candidates"]), 3 if config == "lookahead" else 1)
                self.assertTrue(all(set(c) == CANDIDATE_KEYS for c in out["candidates"]))
                self.assertEqual(out["reply"], out["candidates"][out["selected"]]["response"])

    def test_risky_message_gets_the_prepared_response_only(self):
        out = self.kaki.respond(seeker(RISKY))
        self.assertTrue(out["bypassed"])
        self.assertEqual(out["guard"]["response_key"], "crisis_sg")
        self.assertEqual(out["reply"], CRISIS)
        self.assertEqual((out["perceive"], out["strategies"], out["candidates"], out["selected"]),
                         (None, None, [], None))

    def test_outputs_are_deterministic(self):
        turns = seeker(ORDINARY) + [{"speaker": "supporter", "text": "That sounds hard."}] + seeker("Friends did fine.")
        self.assertEqual(self.kaki.respond(turns), fake_kaki().respond(turns))


class AppTests(unittest.TestCase):
    def setUp(self):
        for patch in (mock.patch.dict(os.environ, {"KAKI_FAKE": "1"}), mock.patch.object(sys, "argv", ["app.py"])):
            patch.start()
            self.addCleanup(patch.stop)
        self.at = AppTest.from_file(APP, default_timeout=30).run()

    def send(self, text):
        self.at.chat_input[0].set_value(text).run()
        self.assertFalse(self.at.exception)

    def table_columns(self):
        return [list(d.value.columns) for d in self.at.dataframe]

    def test_notice_and_configurations(self):
        self.assertFalse(self.at.exception)
        self.assertTrue(any("not therapy" in w.value and "not a reliable crisis detector" in w.value
                            for w in self.at.warning))
        self.assertEqual(self.at.radio(key="config").options, list(CONFIGS))
        self.assertEqual(self.at.radio(key="config").value, "lookahead")

    def test_lookahead_reply_shows_every_stage(self):
        self.send(ORDINARY)
        self.assertEqual([t["speaker"] for t in self.at.session_state.turns], ["seeker", "supporter"])
        self.assertEqual(self.table_columns(), [["Emotion", "Probability"], ["Problem", "Probability"],
                                                ["Strategy", "Probability"]])
        self.assertEqual(len(self.at.dataframe[2].value), 3)
        self.assertEqual(sum("selected]" in m.value for m in self.at.markdown), 1)
        self.assertTrue(any(re.fullmatch(r"lookahead, \d+\.\d\d s", c.value) for c in self.at.caption))

    def test_direct_has_one_reply_and_no_plan(self):
        self.at.radio(key="config").set_value("direct").run()
        self.send(ORDINARY)
        out = self.at.session_state.results[-1]["output"]
        self.assertEqual((out["config"], out["strategies"], len(out["candidates"])), ("direct", None, 1))
        self.assertNotIn(["Strategy", "Probability"], self.table_columns())

    def test_bypass_shows_guard_and_prepared_response_only(self):
        self.send(RISKY)
        self.assertTrue(self.at.session_state.results[-1]["output"]["bypassed"])
        self.assertIn(CRISIS, [e.value for e in self.at.error])
        self.assertEqual(self.table_columns(), [["Lexicon match", "Category", "Negated", "Quoted"]])
        self.assertFalse(any("selected]" in m.value for m in self.at.markdown))

    def test_history_stays_in_session_until_cleared(self):
        self.send(ORDINARY)
        self.send("My friends all did fine.")
        self.assertEqual(len(self.at.session_state.turns), 4)
        self.assertEqual(self.at.selectbox[0].value, 1)  # the latest reply is shown
        self.at.button[0].click().run()
        self.assertEqual((self.at.session_state.turns, self.at.session_state.results), ([], []))


class RealModeTests(unittest.TestCase):
    def test_planning_and_lookahead_share_one_model_set(self):
        self.assertEqual(model_set("planning"), model_set("lookahead"))
        self.assertEqual(model_set("direct"), ("direct",))

    def test_missing_files_follow_the_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("guard", "perceive", "plan", "speak-conditioned"):
                (Path(tmp) / name).mkdir()
            self.assertIn(Path(tmp) / "speak-history", missing_files(Path(tmp), "direct"))
            self.assertTrue(all(p.name == "label_maps.json" for p in missing_files(Path(tmp), "lookahead")))

    def test_missing_checkpoints_stop_with_a_message(self):
        with tempfile.TemporaryDirectory() as empty, mock.patch.dict(os.environ, {"KAKI_FAKE": "0"}), \
                mock.patch.object(sys, "argv", ["app.py", "--checkpoints", empty]):
            at = AppTest.from_file(APP, default_timeout=30).run()
        self.assertFalse(at.exception)
        self.assertIn("--fake", at.error[0].value)
        self.assertEqual(len(at.chat_input), 0)


if __name__ == "__main__":
    unittest.main()

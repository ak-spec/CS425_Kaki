"""Pipeline orchestration tests with stand-in components; no checkpoints are loaded.

Run from the repository root: python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pipeline import Kaki, render

STRATEGIES = [
    {"strategy": "Question", "probability": 0.5},
    {"strategy": "Reflection of feelings", "probability": 0.3},
    {"strategy": "Providing Suggestions", "probability": 0.1},
]


class Calls:
    def __init__(self):
        self.log = []


class FakeGuard:
    def __init__(self, calls, bypass):
        self.calls, self.bypass = calls, bypass

    def check(self, turns):
        self.calls.log.append("guard")
        return {"bypass_generation": self.bypass, "response_key": "crisis_sg" if self.bypass else None}

    def response(self, key):
        return f"prepared:{key}"


class FakePerceiver:
    def __init__(self, calls):
        self.calls = calls

    def predict(self, history):
        self.calls.log.append("perceive")
        return {"emotion": "anxiety"}


class FakePlanner:
    def __init__(self, calls):
        self.calls = calls

    def predict(self, history):
        self.calls.log.append("plan")
        return [dict(s) for s in STRATEGIES]


class FakeSpeaker:
    def __init__(self, calls, mode):
        self.calls, self.mode = calls, mode

    def generate(self, history, emotion=None, strategy=None):
        self.calls.log.append(f"speak-{self.mode}")
        return f"{strategy or 'plain'} reply ({emotion})"

    def generate_candidates(self, history, emotion, strategies):
        return [
            {
                "strategy": s["strategy"],
                "planner_probability": s["probability"],
                "response": self.generate(history, emotion, s["strategy"]),
            }
            for s in strategies
        ]


def kaki(bypass=False, critic=None, weight=1.0):
    calls = Calls()
    bot = Kaki(
        FakeGuard(calls, bypass),
        FakePerceiver(calls),
        FakePlanner(calls),
        {"history": FakeSpeaker(calls, "history"), "conditioned": FakeSpeaker(calls, "conditioned")},
        critic,
        weight,
    )
    return bot, calls


TURNS = [{"speaker": "seeker", "text": "I failed my  midterm."}]


class PipelineTests(unittest.TestCase):
    def test_guard_bypass_skips_every_other_stage(self):
        bot, calls = kaki(bypass=True)
        out = bot.respond(TURNS)
        self.assertEqual(calls.log, ["guard"])
        self.assertEqual((out["reply"], out["bypassed"], out["candidates"]), ("prepared:crisis_sg", True, []))

    def test_direct_uses_history_adapter_without_planner(self):
        bot, calls = kaki()
        out = bot.respond(TURNS, "direct")
        self.assertEqual(calls.log, ["guard", "perceive", "speak-history"])
        self.assertEqual((out["reply"], out["strategies"]), ("plain reply (None)", None))

    def test_planning_uses_top_strategy_and_perceived_emotion(self):
        bot, calls = kaki()
        out = bot.respond(TURNS, "planning")
        self.assertEqual(calls.log, ["guard", "perceive", "plan", "speak-conditioned"])
        self.assertEqual(out["reply"], "Question reply (anxiety)")

    def test_lookahead_without_critic_ranks_by_planner_probability(self):
        bot, _ = kaki()
        out = bot.respond(TURNS)
        self.assertEqual(len(out["candidates"]), 3)
        self.assertEqual(out["selected"], 0)
        self.assertTrue(all(c["critic_score"] is None for c in out["candidates"]))

    def test_lookahead_combines_critic_and_planner(self):
        critic = lambda history, replies: [0.0, 0.5, 0.0]
        out = kaki(critic=critic, weight=1.0)[0].respond(TURNS)
        self.assertEqual(out["reply"], "Reflection of feelings reply (anxiety)")  # 0.5 + 0.3 beats 0 + 0.5
        out = kaki(critic=critic, weight=10.0)[0].respond(TURNS)
        self.assertEqual(out["reply"], "Question reply (anxiety)")  # a large weight follows the planner

    def test_unknown_config_rejected(self):
        with self.assertRaises(ValueError):
            kaki()[0].respond(TURNS, "best")

    def test_render_matches_training_format(self):
        turns = [
            {"speaker": "seeker", "text": " hi  there "},
            {"speaker": "supporter", "text": ""},
            {"speaker": "supporter", "text": "How are you?"},
        ]
        self.assertEqual(render(turns), "seeker: hi there | supporter: How are you?")


if __name__ == "__main__":
    unittest.main()

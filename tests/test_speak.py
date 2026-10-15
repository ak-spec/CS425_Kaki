import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speak import Speaker, collate_examples, prompt_ids, training_example


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return [ord(c) + 1 for c in text]

    def decode(self, ids):
        return "".join(chr(i - 1) for i in ids if i)

    def apply_chat_template(self, messages, **kwargs):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"


class SpeakTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def test_long_history_keeps_controls_and_latest_message(self):
        ids = prompt_ids(
            self.tokenizer,
            "old " * 300 + "latest message",
            "conditioned",
            "sadness",
            "Question",
            400,
        )
        text = self.tokenizer.decode(ids)
        self.assertLessEqual(len(ids), 400)
        self.assertIn("sadness", text)
        self.assertIn("Question", text)
        self.assertIn("latest message", text)
        self.assertTrue(text.endswith("<assistant>"))

    def test_only_reply_and_eos_are_supervised(self):
        row = {
            "text": "seeker: exam worry",
            "response": "I understand.",
            "emotion": "sadness",
            "strategy": "Question",
        }
        item = training_example(self.tokenizer, row, "conditioned")
        target = [value for value in item["labels"] if value != -100]
        self.assertEqual(target, self.tokenizer.encode(row["response"]) + [0])
        shorter = {"input_ids": [2, 3], "labels": [-100, 3]}
        batch = collate_examples([item, shorter], 0)
        self.assertTrue((batch["labels"][1, 2:] == -100).all())
        self.assertTrue((batch["attention_mask"][1, 2:] == 0).all())

    def test_history_baseline_ignores_reference_conditions(self):
        a = prompt_ids(
            self.tokenizer, "seeker: hello", "history", "sadness", "Question"
        )
        b = prompt_ids(
            self.tokenizer, "seeker: hello", "history", "anger", "Information"
        )
        self.assertEqual(a, b)

    def test_candidate_order_probabilities_and_validation(self):
        speaker = Speaker.__new__(Speaker)
        speaker.config = {"mode": "conditioned"}
        speaker.generate = lambda history, emotion, strategy: strategy + " reply"
        strategies = [
            {"strategy": name, "probability": p}
            for name, p in [("Question", 0.4), ("Information", 0.3), ("Others", 0.1)]
        ]
        replies = speaker.generate_candidates("history", "sadness", strategies)
        self.assertEqual(
            [r["strategy"] for r in replies], [s["strategy"] for s in strategies]
        )
        self.assertEqual([r["planner_probability"] for r in replies], [0.4, 0.3, 0.1])
        with self.assertRaises(ValueError):
            speaker.generate_candidates("history", "sadness", [strategies[0]] * 3)
        strategies[0]["probability"] = float("nan")
        with self.assertRaises(ValueError):
            speaker.generate_candidates("history", "sadness", strategies)


if __name__ == "__main__":
    unittest.main()

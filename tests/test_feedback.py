import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_feedback import audit


class FeedbackAuditTests(unittest.TestCase):
    def test_rating_interval_can_span_seeker_messages(self):
        turns = [
            {"speaker": "supporter"},
            {"speaker": "seeker"},
            {"speaker": "supporter"},
            {"speaker": "seeker", "annotation": {"feedback": "4"}},
            {"speaker": "supporter"},
            {"speaker": "seeker", "annotation": {"feedback": "5"}},
        ]
        report = audit([{"dialog": turns}], [0])
        self.assertEqual(report["supporter_utterances_since_previous_rating"], {1: 1, 2: 1})
        self.assertEqual(report["immediately_preceding_supporter_block_length"], {1: 2})


if __name__ == "__main__":
    unittest.main()

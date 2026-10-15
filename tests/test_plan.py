import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plan import evaluate


class FixedModel:
    def eval(self):
        return self

    def __call__(self, **batch):
        return SimpleNamespace(
            logits=torch.tensor([[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]] * 2)
        )


class MetricsTests(unittest.TestCase):
    def test_top_three_and_macro_f1_use_all_classes(self):
        metrics = evaluate(
            FixedModel(),
            [{"labels": torch.tensor([0, 1])}],
            "cpu",
            [str(i) for i in range(8)],
        )
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["top3_coverage"], 1.0)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3) / 8)


if __name__ == "__main__":
    unittest.main()

"""Critique: a cross-encoder that scores a candidate reply given the dialogue history.

DeBERTa-v3-base reads the history and the reply together and outputs one score, trained by
src/train_critic.py to predict the seeker's helpfulness rating. ESConv ratings cover about two supporter
utterances (docs/feedback-audit.md), so they are weak labels for a single reply. Scores are on the rating
scale: 3 + 2 * output, so 5 means "predicted very helpful".

Usage, from the repository root with src on the import path:
    from critic import Critic
    critic = Critic("checkpoints/critic")
    critic("seeker: I failed my midterm.", ["reply one", "reply two", "reply three"])
A Critic instance is the `critic` callable that src/pipeline.py accepts.
"""
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def to_target(rating):
    """Ratings 1-5 to the regression target in [-1, 1]."""
    return (rating - 3) / 2


def to_rating(output):
    return 3 + 2 * output


@torch.inference_mode()
def predict_ratings(model, tokenizer, histories, replies, max_len=512, batch_size=32):
    """Predicted rating for each (history, reply) pair. Long histories lose their oldest turns."""
    model.eval()
    device = next(model.parameters()).device
    order = sorted(range(len(histories)), key=lambda i: len(histories[i]) + len(replies[i]))
    out = [0.0] * len(histories)
    for start in range(0, len(order), batch_size):
        idx = order[start: start + batch_size]
        enc = tokenizer([histories[i] for i in idx], [replies[i] for i in idx], truncation="only_first",
                        max_length=max_len, padding=True, return_tensors="pt").to(device)
        for i, value in zip(idx, model(**enc).logits.float().squeeze(-1).tolist()):
            out[i] = to_rating(value)
    return out


class Critic:
    def __init__(self, checkpoint, device=None, max_len=512):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(Path(checkpoint))
        self.tokenizer.truncation_side = "left"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            Path(checkpoint), dtype=torch.float32).to(self.device).eval()

    def __call__(self, history: str, replies: list[str]) -> list[float]:
        """One predicted rating per candidate reply for the same history."""
        return predict_ratings(self.model, self.tokenizer, [history] * len(replies), replies, self.max_len)

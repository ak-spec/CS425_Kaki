"""Perceive inference: the seeker's emotion and problem type from the dialogue history.

train_perceive.ipynb saves checkpoints/perceive/best.pt (a state dict) and run.json (its CFG). The
model class below must match the notebook's; the notebook keeps its own copy so it runs on Kaggle and
Colab without the repository. Label names come from data/processed/label_maps.json.

Usage, from the repository root with src on the import path:
    from perceive import Perceiver
    perceiver = Perceiver("checkpoints/perceive")
    perceiver.predict("supporter: Hi, how are you? | seeker: I failed my midterm.")
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent


class PerceiveModel(nn.Module):
    """One shared encoder; the <s> token feeds an emotion head and a problem head."""

    def __init__(self, name, n_emo, n_prob, dropout, pretrained=True):
        super().__init__()
        # inference loads trained weights from best.pt, so the pretrained ones are not needed
        self.encoder = (AutoModel.from_pretrained(name) if pretrained
                        else AutoModel.from_config(AutoConfig.from_pretrained(name)))
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.emo_head = nn.Linear(hidden, n_emo)
        self.prob_head = nn.Linear(hidden, n_prob)

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        h = self.drop(h)
        return self.emo_head(h), self.prob_head(h)


class Perceiver:
    def __init__(self, checkpoint=ROOT / "checkpoints/perceive",
                 label_maps=ROOT / "data/processed/label_maps.json", device=None):
        checkpoint = Path(checkpoint)
        cfg = json.loads((checkpoint / "run.json").read_text(encoding="utf-8"))["config"]
        maps = json.loads(Path(label_maps).read_text(encoding="utf-8"))
        self.emotions = sorted(maps["emotion2id"], key=maps["emotion2id"].get)
        self.problems = sorted(maps["problem2id"], key=maps["problem2id"].get)
        self.max_len = cfg["max_len"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], truncation_side="left")
        self.model = PerceiveModel(cfg["model_name"], len(self.emotions), len(self.problems), cfg["dropout"],
                                   pretrained=False)
        self.model.load_state_dict(torch.load(checkpoint / "best.pt", map_location="cpu"))
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, text: str) -> dict:
        """Most likely emotion and problem, with the full probability for each class."""
        enc = self.tokenizer(text, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        emo_logits, prob_logits = self.model(enc["input_ids"], enc["attention_mask"])
        emo = dict(zip(self.emotions, emo_logits.float().softmax(-1)[0].tolist()))
        prob = dict(zip(self.problems, prob_logits.float().softmax(-1)[0].tolist()))
        return {"emotion": max(emo, key=emo.get), "emotion_probs": emo,
                "problem": max(prob, key=prob.get), "problem_probs": prob}

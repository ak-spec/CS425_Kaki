"""Shared prompt construction and inference for the Speak adapters."""

import json
import math
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prep_downstream import STRATEGIES

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = (
    "You are Kaki, a supportive conversational assistant. Respond to the latest "
    "message with a concise, relevant and empathetic reply."
)


def prompt_ids(tokenizer, history, mode, emotion=None, strategy=None, max_tokens=896):
    """Trim history from the left while preserving instructions and chat markers."""
    if not isinstance(history, str) or not history.strip():
        raise ValueError("History must be nonempty text.")
    if mode not in {"history", "conditioned"}:
        raise ValueError("Mode must be history or conditioned.")
    instructions = SYSTEM_PROMPT
    if mode == "conditioned":
        if not emotion or strategy not in STRATEGIES:
            raise ValueError(
                "Conditioned generation needs an emotion and valid strategy."
            )
        instructions += f" The seeker's emotion is {emotion}. Use this support strategy: {strategy}."
    history_tokens = tokenizer.encode(history, add_special_tokens=False)
    while True:
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": tokenizer.decode(history_tokens)},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer.encode(text, add_special_tokens=False)
        excess = len(tokens) - max_tokens
        if excess <= 0:
            return tokens
        if excess >= len(history_tokens):
            raise ValueError("Prompt budget is too small for instructions and history.")
        history_tokens = history_tokens[excess:]


def training_example(tokenizer, row, mode, max_length=1024, max_reply_tokens=128):
    prompt = prompt_ids(
        tokenizer,
        row["text"],
        mode,
        row.get("emotion"),
        row.get("strategy"),
        max_length - max_reply_tokens,
    )
    response = tokenizer.encode(row["response"], add_special_tokens=False)
    if not response:
        raise ValueError("Training responses must not be empty.")
    target = response[: max_reply_tokens - 1] + [tokenizer.eos_token_id]
    return {"input_ids": prompt + target, "labels": [-100] * len(prompt) + target}


def collate_examples(rows, pad_token_id):
    width = max(len(row["input_ids"]) for row in rows)
    ids, masks, labels = [], [], []
    for row in rows:
        padding = width - len(row["input_ids"])
        ids.append(row["input_ids"] + [pad_token_id] * padding)
        masks.append([1] * len(row["input_ids"]) + [0] * padding)
        labels.append(row["labels"] + [-100] * padding)
    return {
        "input_ids": torch.tensor(ids),
        "attention_mask": torch.tensor(masks),
        "labels": torch.tensor(labels),
    }


class Speaker:
    def __init__(self, checkpoint, device=None):
        checkpoint = Path(checkpoint)
        self.config = json.loads(
            (checkpoint / "speak_config.json").read_text(encoding="utf-8")
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        base = AutoModelForCausalLM.from_pretrained(self.config["base_model"])
        self.model = PeftModel.from_pretrained(base, checkpoint).to(self.device).eval()

    @torch.inference_mode()
    def generate(self, history, emotion=None, strategy=None):
        ids = prompt_ids(
            self.tokenizer,
            history,
            self.config["mode"],
            emotion,
            strategy,
            self.config["max_length"] - self.config["max_reply_tokens"],
        )
        inputs = torch.tensor([ids], device=self.device)
        output = self.model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            do_sample=False,
            max_new_tokens=self.config["max_reply_tokens"],
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(
            output[0, len(ids) :], skip_special_tokens=True
        ).strip()

    def generate_candidates(self, history, emotion, strategies):
        """Keep planner order and probabilities; selection belongs to Critique."""
        if self.config["mode"] != "conditioned":
            raise ValueError(
                "Three-strategy generation requires the conditioned adapter."
            )
        if len(strategies) != 3 or len({s["strategy"] for s in strategies}) != 3:
            raise ValueError("Provide exactly three distinct strategies.")
        for item in strategies:
            probability = item["probability"]
            if (
                item["strategy"] not in STRATEGIES
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise ValueError("Invalid strategy or planner probability.")
        return [
            {
                "strategy": item["strategy"],
                "planner_probability": item["probability"],
                "response": self.generate(history, emotion, item["strategy"]),
            }
            for item in strategies
        ]

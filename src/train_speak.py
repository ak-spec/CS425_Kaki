"""Train either Speak LoRA adapter on the same response examples."""

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from speak import BASE_MODEL, collate_examples, training_example


def read_examples(path, tokenizer, args):
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if args.smoke:
        rows = rows[:2]
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    return [
        training_example(
            tokenizer, row, args.mode, args.max_length, args.max_reply_tokens
        )
        for row in rows
    ]


@torch.inference_mode()
def validation_loss(model, batches, device):
    model.eval()
    total_loss, tokens = 0.0, 0
    for batch in batches:
        batch = {k: v.to(device) for k, v in batch.items()}
        count = int((batch["labels"][:, 1:] != -100).sum())
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss.")
        total_loss += float(loss) * count
        tokens += count
    return total_loss / tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["history", "conditioned"], required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed/downstream"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-reply-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.rank) < 1 or args.lr <= 0:
        parser.error("Epochs, batch size, rank and learning rate must be positive.")
    if not 2 <= args.max_reply_tokens < args.max_length:
        parser.error("Require 2 <= max-reply-tokens < max-length.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory to preserve earlier runs.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an end-of-sequence token.")
    base = AutoModelForCausalLM.from_pretrained(args.model)
    base.config.use_cache = False
    model = get_peft_model(
        base,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.rank,
            lora_alpha=2 * args.rank,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        ),
    ).to(device)
    collate = partial(collate_examples, pad_token_id=tokenizer.pad_token_id)
    loaders = {}
    for split in ("train", "val"):
        examples = read_examples(
            args.data / f"responses_{split}.jsonl", tokenizer, args
        )
        loaders[split] = DataLoader(
            examples,
            batch_size=args.batch_size,
            shuffle=split == "train",
            collate_fn=collate,
        )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )
    best, records = float("inf"), []
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1 if args.smoke else args.epochs):
        model.train()
        for batch in loaders["train"]:
            optimizer.zero_grad()
            loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        score = validation_loss(model, loaders["val"], device)
        records.append({"epoch": epoch + 1, "validation_loss": score})
        print(json.dumps(records[-1]), flush=True)
        if score < best:
            best = score
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["base_model"] = args.model
    config["device"] = device
    config["data_sha256"] = {
        split: hashlib.sha256(
            (args.data / f"responses_{split}.jsonl").read_bytes()
        ).hexdigest()
        for split in ("train", "val")
    }
    (args.output / "speak_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (args.output / "training_metrics.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

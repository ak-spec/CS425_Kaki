"""Baselines from the proposal, answering the same test rows as the three configurations.

    zeroshot    Qwen2.5-0.5B-Instruct without an adapter, prompted like the history-only adapter (B2)
    blenderbot  BlenderBot-small (90M) fine-tuned on our ESConv training split to map the history to the
                reply, without strategies: the vanilla baseline of Liu et al. (2021), retrained (B1)

Both answer the rows of data/outputs/direct.jsonl. Guard runs in front of every Kaki configuration, so
rows it bypassed keep Guard's prepared reply here too and stay out of the metrics. Decoding is greedy, as
for Kaki. BlenderBot-small's tokenizer lowercases text and spaces out punctuation ("i ' m sorry ."), so
its replies are detokenized and sentence-cased before scoring.

    python src/baselines.py train-blenderbot           # checkpoints/blenderbot, best epoch by validation loss
    python src/baselines.py generate --system zeroshot  # data/outputs/zeroshot.jsonl
    python src/baselines.py generate --system blenderbot
    python src/metrics.py                               # scores every system that has an output file
"""

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from generate_outputs import private_dir
from select_eval_contexts import read_jsonl

ROOT = Path(__file__).resolve().parent.parent
BLENDERBOT = "facebook/blenderbot_small-90M"
SYSTEMS = ("zeroshot", "blenderbot")


def detokenize(text: str) -> str:
    """BlenderBot-small output to ordinary text: "i ' m ok , thanks ." -> "I'm ok, thanks." """
    text = re.sub(r"(\w) ' (\w)", r"\1'\2", text)
    text = re.sub(r" ([.,!?;:])", r"\1", text)
    text = re.sub(r"\bi\b", "I", text)
    return re.sub(r"(^|[.!?] )([a-z])", lambda m: m.group(1) + m.group(2).upper(), text.strip())


class BlenderBot:
    def __init__(self, checkpoint, device=None, max_source=512, max_reply=64):
        from transformers import AutoTokenizer, BlenderbotSmallForConditionalGeneration

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.tokenizer.truncation_side = "left"  # keep the latest turns
        self.model = (
            BlenderbotSmallForConditionalGeneration.from_pretrained(checkpoint, dtype=torch.float32)
            .to(self.device)
            .eval()
        )
        self.max_source, self.max_reply = max_source, max_reply

    @torch.inference_mode()
    def generate_batch(self, histories, batch_size=32):
        replies = []
        for start in range(0, len(histories), batch_size):
            enc = self.tokenizer(
                histories[start : start + batch_size],
                truncation=True,
                max_length=self.max_source,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            out = self.model.generate(
                **enc, do_sample=False, num_beams=1, min_length=0, max_new_tokens=self.max_reply, no_repeat_ngram_size=3
            )
            replies += [detokenize(t) for t in self.tokenizer.batch_decode(out, skip_special_tokens=True)]
        return replies


def seq2seq_batches(rows, tokenizer, batch_size, max_source, max_reply, shuffle=False):
    def collate(batch):
        enc = tokenizer(
            [r["text"] for r in batch], truncation=True, max_length=max_source, padding=True, return_tensors="pt"
        )
        # the tokenizer adds no end token, and the model must learn to stop; long replies keep their start
        targets = [tokenizer(r["response"]).input_ids[: max_reply - 1] + [tokenizer.eos_token_id] for r in batch]
        width = max(map(len, targets))
        enc["labels"] = torch.tensor([t + [-100] * (width - len(t)) for t in targets])
        return enc

    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.inference_mode()
def mean_loss(model, batches, device) -> float:
    """Token-level cross-entropy of the reference replies; exp of it is perplexity."""
    model.eval()
    total, tokens = 0.0, 0
    for batch in batches:
        batch = {k: v.to(device) for k, v in batch.items()}
        n = int((batch["labels"] != -100).sum())
        total += float(model(**batch).loss) * n
        tokens += n
    return total / tokens


def train_blenderbot(args):
    from transformers import AutoTokenizer, BlenderbotSmallForConditionalGeneration, get_linear_schedule_with_warmup

    if (args.output / "config.json").exists():
        raise ValueError(f"{args.output} already holds a model; choose a new --output")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BLENDERBOT)
    tokenizer.truncation_side = "left"
    model = BlenderbotSmallForConditionalGeneration.from_pretrained(BLENDERBOT, dtype=torch.float32).to(device)
    data = {s: read_jsonl(args.data / f"responses_{s}.jsonl") for s in ("train", "val", "test")}
    if args.smoke:
        data = {s: rows[:32] for s, rows in data.items()}
    batches = {
        s: seq2seq_batches(rows, tokenizer, args.batch_size, args.max_source, args.max_reply, s == "train")
        for s, rows in data.items()
    }
    epochs = 1 if args.smoke else args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    schedule = get_linear_schedule_with_warmup(optimizer, args.warmup, epochs * len(batches["train"]))
    best, history, t0 = math.inf, [], time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in batches["train"]:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
        val = mean_loss(model, batches["val"], device)
        history.append(
            {
                "epoch": epoch,
                "val_loss": round(val, 4),
                "val_ppl": round(math.exp(val), 2),
                "minutes": round((time.time() - t0) / 60, 1),
            }
        )
        print(json.dumps(history[-1]), flush=True)
        if val < best:
            best = val
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
    model = BlenderbotSmallForConditionalGeneration.from_pretrained(args.output, dtype=torch.float32).to(device)
    test = mean_loss(model, batches["test"], device)
    run = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "func"},
        "base_model": BLENDERBOT,
        "device": device,
        "validation": history,
        "best_epoch": min(history, key=lambda h: h["val_loss"])["epoch"],
        "test_loss": round(test, 4),
        "test_ppl": round(math.exp(test), 2),
        "note": "perplexity over BlenderBot-small tokens; not comparable with Qwen perplexity",
    }
    (args.output / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    reply = BlenderBot(args.output, device).generate_batch(["seeker: I am worried about my exams."])[0]
    print(json.dumps({"test_ppl": run["test_ppl"], "best_epoch": run["best_epoch"], "reload_reply": reply}), flush=True)


def generate(args):
    reference = read_jsonl(args.outputs / "direct.jsonl")
    histories = {
        (r["dialogue_id"], r["utterance_idx"]): r["text"] for r in read_jsonl(args.data / "responses_test.jsonl")
    }
    todo = [r for r in reference if not r["bypassed"]][: args.limit]
    if args.system == "zeroshot":
        from speak import Speaker

        replies = Speaker(None).generate_batch(
            [histories[(r["dialogue_id"], r["utterance_idx"])] for r in todo], batch_size=args.batch_size
        )
    else:
        replies = BlenderBot(args.checkpoint).generate_batch(
            [histories[(r["dialogue_id"], r["utterance_idx"])] for r in todo], batch_size=args.batch_size
        )
    written = dict(zip(((r["dialogue_id"], r["utterance_idx"]) for r in todo), replies))
    private_dir(args.outputs)
    keep = ("dialogue_id", "utterance_idx", "problem", "emotion", "after_seeker", "reference", "reference_strategy")
    with open(args.outputs / f"{args.system}.jsonl", "w", encoding="utf-8") as f:
        for r in reference:
            key = (r["dialogue_id"], r["utterance_idx"])
            if not r["bypassed"] and key not in written:  # --limit
                continue
            row = {
                **{k: r[k] for k in keep},
                "config": args.system,
                "reply": written.get(key, r["reply"]),
                "bypassed": r["bypassed"],
                "strategy": None,
                "perceived": None,
                "planner": None,
                "candidates": [],
                "selected": None,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {"system": args.system, "replies": len(written), "bypassed_copied": sum(r["bypassed"] for r in reference)}
        )
    )


def main():
    ap = argparse.ArgumentParser(description="Baselines B1 (BlenderBot-small) and B2 (Qwen zero-shot).")
    sub = ap.add_subparsers(required=True)
    t = sub.add_parser("train-blenderbot")
    t.add_argument("--data", type=Path, default=ROOT / "data/processed/downstream")
    t.add_argument("--output", type=Path, default=ROOT / "checkpoints/blenderbot")
    t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--warmup", type=int, default=100)
    t.add_argument("--max-source", type=int, default=512)
    t.add_argument("--max-reply", type=int, default=64)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--smoke", action="store_true", help="32 rows per split, one epoch")
    t.set_defaults(func=train_blenderbot)
    g = sub.add_parser("generate")
    g.add_argument("--system", choices=SYSTEMS, required=True)
    g.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/blenderbot")
    g.add_argument("--data", type=Path, default=ROOT / "data/processed/downstream")
    g.add_argument("--outputs", type=Path, default=ROOT / "data/outputs")
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--limit", type=int, help="first N replies only, for quick checks")
    g.set_defaults(func=generate)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

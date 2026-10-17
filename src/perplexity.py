"""Perplexity of the reference supporter replies on the test split under each Qwen-based generator.

    zeroshot     Qwen2.5-0.5B-Instruct without an adapter, history-only prompt
    direct       history-only adapter
    conditioned  conditioned adapter given the reference emotion and strategy, so its perplexity assumes a
                 perfect Perceive and Plan and is a lower bound for planning

All three share Qwen's tokenizer, so the numbers are comparable with each other but not with
BlenderBot-small's (src/baselines.py reports that one with its own tokenizer).

    python src/perplexity.py      # data/outputs/perplexity.json
"""

import argparse
import json
import math
from functools import partial
from pathlib import Path

from torch.utils.data import DataLoader

from common import private_dir, read_jsonl
from speak import Speaker, collate_examples, training_example
from train_speak import validation_loss

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--data", type=Path, default=ROOT / "data/processed/downstream/responses_test.jsonl")
    ap.add_argument("--output", type=Path, default=ROOT / "data/outputs")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    rows = read_jsonl(args.data)[: args.limit]
    result = {"rows": len(rows)}
    for name, checkpoint in (
        ("zeroshot", None),
        ("direct", args.checkpoints / "speak-history"),
        ("conditioned", args.checkpoints / "speak-conditioned"),
    ):
        speaker = Speaker(checkpoint)
        cfg, tok = speaker.config, speaker.tokenizer
        examples = [training_example(tok, r, cfg["mode"], cfg["max_length"], cfg["max_reply_tokens"]) for r in rows]
        batches = DataLoader(
            examples, batch_size=args.batch_size, collate_fn=partial(collate_examples, pad_token_id=tok.pad_token_id)
        )
        loss = validation_loss(speaker.model, batches, speaker.device)
        result[name] = {"loss": round(loss, 4), "perplexity": round(math.exp(loss), 2)}
        print(json.dumps({name: result[name]}), flush=True)
        del speaker
    (private_dir(args.output) / "perplexity.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

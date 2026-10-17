"""Does Speak carry out the strategy it is asked for? The proposal's controllability check.

An independent strategy tagger labels each reply: RoBERTa-base trained on ESConv supporter replies alone
(src/plan.py --input response), never on Speak output. Two measurements:

    outputs  the planning and look-ahead replies of the test run (data/outputs): share whose tag equals the
             strategy Speak was given
    forced   N test histories ending with a seeker message, one reply per strategy from the conditioned
             adapter with the reference emotion: requested-by-tagged confusion matrix, and the share tagged
             as requested for each strategy

Two reference points. The tagger's accuracy on the human reference replies is the ceiling: a Speak that
always carried out the strategy would score about that, not 100%. The history-only adapter's replies
for the same histories give the floor: how often a reply written without any strategy is tagged as each
strategy anyway.

    python src/controllability.py --tagger checkpoints/strategy-tagger
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from generate_outputs import private_dir
from select_eval_contexts import read_jsonl

ROOT = Path(__file__).resolve().parent.parent


@torch.inference_mode()
def tag(model, tokenizer, texts, batch_size=64) -> list[str]:
    """The tagger's top strategy for each reply."""
    device = next(model.parameters()).device
    out = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[start : start + batch_size], truncation=True, max_length=512, padding=True, return_tensors="pt"
        ).to(device)
        out += [model.config.id2label[int(i)] for i in model(**enc).logits.argmax(-1)]
    return out


def agreement(asked: list[str], tagged: list[str]) -> dict:
    """Overall and per-strategy share of replies tagged with the strategy they were asked for."""
    per = {s: mean(t == s for a, t in zip(asked, tagged) if a == s) for s in sorted(set(asked))}
    return {"overall": mean(a == t for a, t in zip(asked, tagged)), "n": len(asked), "per_strategy": per}


def confusion(asked: list[str], tagged: list[str], labels: list[str]) -> list[list[int]]:
    counts = Counter(zip(asked, tagged))
    return [[counts[(a, t)] for t in labels] for a in labels]


def ends_with_seeker(history: str) -> bool:
    return history.rsplit(" | ", 1)[-1].startswith("seeker:")


def main():
    ap = argparse.ArgumentParser(description="Strategy controllability of the conditioned Speak adapter.")
    ap.add_argument("--tagger", type=Path, default=ROOT / "checkpoints/strategy-tagger")
    ap.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--data", type=Path, default=ROOT / "data/processed/downstream")
    ap.add_argument("--outputs", type=Path, default=ROOT / "data/outputs")
    ap.add_argument("--histories", type=int, default=200, help="test histories for the forced check")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.tagger)
    model = AutoModelForSequenceClassification.from_pretrained(args.tagger, dtype=torch.float32).to(device).eval()
    labels = json.loads((args.data / "strategy_labels.json").read_text(encoding="utf-8"))
    result = {"tagger": str(args.tagger)}

    # replies of the test run
    runs = {
        c: [r for r in read_jsonl(args.outputs / f"{c}.jsonl") if not r["bypassed"]]
        for c in ("direct", "planning", "lookahead")
        if (args.outputs / f"{c}.jsonl").exists()
    }
    if runs:
        rows = next(iter(runs.values()))
        refs = [r["reference_strategy"] for r in rows]
        result["tagger_on_references"] = agreement(refs, tag(model, tokenizer, [r["reference"] for r in rows]))
        for c, rows in runs.items():
            tagged = tag(model, tokenizer, [r["reply"] for r in rows])
            if c == "direct":
                result["direct_tag_distribution"] = {s: n / len(tagged) for s, n in Counter(tagged).most_common()}
            else:
                result[c] = agreement([r["strategy"] for r in rows], tagged)

    # one reply per strategy for the same histories
    from speak import Speaker

    test = [r for r in read_jsonl(args.data / "responses_test.jsonl") if ends_with_seeker(r["text"])]
    sample = random.Random(args.seed).sample(test, min(args.histories, len(test)))
    histories, emotions = [r["text"] for r in sample], [r["emotion"] for r in sample]
    conditioned = Speaker(args.checkpoints / "speak-conditioned", device)
    asked, replies = [], []
    for s in labels:
        replies += conditioned.generate_batch(histories, emotions, [s] * len(sample))
        asked += [s] * len(sample)
    del conditioned
    tagged = tag(model, tokenizer, replies)
    plain = tag(model, tokenizer, Speaker(args.checkpoints / "speak-history", device).generate_batch(histories))
    floor = Counter(plain)
    result["forced"] = {
        "histories": len(sample),
        "seed": args.seed,
        "labels": labels,
        **agreement(asked, tagged),
        "history_only_tagged_as": {s: floor[s] / len(plain) for s in labels},
        "tagger_on_references": agreement(
            [r["strategy"] for r in sample], tag(model, tokenizer, [r["response"] for r in sample])
        ),
        "confusion": confusion(asked, tagged, labels),
    }
    folder = private_dir(args.outputs)
    with open(folder / "controllability_replies.jsonl", "w", encoding="utf-8") as f:
        for i, (a, t, reply) in enumerate(zip(asked, tagged, replies)):
            r = sample[i % len(sample)]
            f.write(
                json.dumps(
                    {
                        "dialogue_id": r["dialogue_id"],
                        "utterance_idx": r["utterance_idx"],
                        "asked": a,
                        "tagged": t,
                        "reply": reply,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    (folder / "controllability.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "forced"}, indent=1))
    print(json.dumps({k: v for k, v in result["forced"].items() if k != "confusion"}, indent=1))


if __name__ == "__main__":
    main()

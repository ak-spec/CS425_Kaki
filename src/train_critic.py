"""Train and evaluate the Critic (used for selection) and the Eval-Critic on ESConv seeker ratings.

Data: data/processed/downstream/critic_proxy_{split}.jsonl from src/prep_downstream.py, one row per
seeker rating with the supporter block just before it. Ratings cover about two supporter utterances, so
these are weak labels (docs/feedback-audit.md). Training dialogues are split in half with a fixed seed:
half "a" trains the Critic, half "b" the Eval-Critic, so the two never share training dialogues. Neither
sees validation or test dialogues.

    python src/train_critic.py --half a --output checkpoints/critic
    python src/train_critic.py --half b --output checkpoints/eval-critic
    python src/train_critic.py --half a --output checkpoints/critic --evaluate-test   # once, at the end

--contrast-weight adds a margin loss that ranks each real reply above a reply from another dialogue for
the same history, so the score has to depend on the reply (see docs/critic.md).

The epoch with the best validation pairwise accuracy is kept. Validation also reports baselines (constant
rating, position in the dialogue, reply length) and two checks that the score depends on the reply:
accuracy on rating pairs that go against the conversation's upward trend, and how often the real reply
outscores a reply taken from another dialogue.
"""
import argparse
import collections
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import transformers
from scipy.stats import spearmanr
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from critic import predict_ratings, to_target

SMOKE = {"train": 64, "val": 64, "max_len": 128}
CONTRAST_MARGIN = 0.5  # in target units: one rating point


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def split_halves(dialogue_ids, seed: int) -> dict[str, list[int]]:
    """Training dialogues in two disjoint halves; the same seed always gives the same halves."""
    ids = sorted(set(dialogue_ids))
    random.Random(seed).shuffle(ids)
    return {"a": sorted(ids[: len(ids) // 2]), "b": sorted(ids[len(ids) // 2:])}


def rating_pairs(rows: list[dict]) -> list[tuple[int, int]]:
    """(i, j) with the same dialogue and rows[i] rated higher than rows[j]."""
    by_dialogue = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_dialogue[r["dialogue_id"]].append(i)
    return [(i, j) for idx in by_dialogue.values() for i in idx for j in idx
            if rows[i]["feedback"] > rows[j]["feedback"]]


def pair_accuracy(pairs, values) -> float:
    """Share of pairs ordered correctly by values; ties count half."""
    if not pairs:
        return float("nan")
    return float(np.mean([1.0 if values[i] > values[j] else 0.5 if values[i] == values[j] else 0.0
                          for i, j in pairs]))


def other_replies(rows: list[dict], seed: int) -> list[str]:
    """For each row, the reply of a row from a different dialogue."""
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    out = []
    for k, r in enumerate(rows):
        j = order[k]
        while rows[j]["dialogue_id"] == r["dialogue_id"]:
            j = rng.randrange(len(rows))
        out.append(rows[j]["response"])
    return out


def evaluate(rows, preds, preds_other, train_mean: float) -> dict:
    ratings = np.array([r["feedback"] for r in rows], dtype=float)
    preds, preds_other = np.asarray(preds), np.asarray(preds_other)
    pairs = rating_pairs(rows)
    position = [r["response_start_idx"] for r in rows]
    against = [(i, j) for i, j in pairs if position[i] < position[j]]  # the earlier block was rated higher
    return {
        "rows": len(rows), "pairs": len(pairs), "against_trend_pairs": len(against),
        "mae": float(np.abs(preds - ratings).mean()),
        "mae_constant": float(np.abs(train_mean - ratings).mean()),
        "spearman": float(spearmanr(preds, ratings).statistic),
        "pairwise_accuracy": pair_accuracy(pairs, preds),
        "pairwise_position_baseline": pair_accuracy(pairs, position),
        "pairwise_length_baseline": pair_accuracy(pairs, [len(r["response"]) for r in rows]),
        "against_trend_accuracy": pair_accuracy(against, preds),
        "beats_other_reply": float((preds > preds_other).mean()),
        "mean_prediction_by_rating": {int(k): round(float(preds[ratings == k].mean()), 3)
                                      for k in sorted(set(ratings.tolist()))},
    }


def score_split(model, tokenizer, rows, max_len, seed):
    histories = [r["text"] for r in rows]
    preds = predict_ratings(model, tokenizer, histories, [r["response"] for r in rows], max_len)
    others = predict_ratings(model, tokenizer, histories, other_replies(rows, seed), max_len)
    return preds, others


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed/downstream"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--half", choices=["a", "b"], required=True, help="a: Critic, b: Eval-Critic")
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--contrast-weight", type=float, default=0.0,
                    help="weight of a margin loss that ranks the real reply above one from another dialogue "
                         "for the same history; 0 trains on ratings only")
    ap.add_argument("--smoke", action="store_true", help="64 rows, 1 epoch, 128 tokens; never reads test")
    ap.add_argument("--evaluate-test", action="store_true", help="score the test split once with a trained checkpoint")
    args = ap.parse_args()
    if args.smoke and args.evaluate_test:
        ap.error("smoke runs must not evaluate the test split")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.evaluate_test:
        run = json.loads((args.output / "run.json").read_text(encoding="utf-8"))
        tok = AutoTokenizer.from_pretrained(args.output)
        tok.truncation_side = "left"
        model = AutoModelForSequenceClassification.from_pretrained(args.output, dtype=torch.float32).to(device)
        rows = read_jsonl(args.data / "critic_proxy_test.jsonl")
        metrics = evaluate(rows, *score_split(model, tok, rows, run["args"]["max_length"], args.seed), run["train_mean"])
        (args.output / "test_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
        print(json.dumps(metrics, indent=1))
        return

    if (args.output / "config.json").exists():
        ap.error(f"{args.output} already holds a model; choose a new --output to keep it")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    train_all = read_jsonl(args.data / "critic_proxy_train.jsonl")
    halves = split_halves([r["dialogue_id"] for r in train_all], args.seed)
    keep = set(halves[args.half])
    train = [r for r in train_all if r["dialogue_id"] in keep]
    val = read_jsonl(args.data / "critic_proxy_val.jsonl")
    if args.smoke:
        rng = random.Random(0)
        train, val = rng.sample(train, SMOKE["train"]), rng.sample(val, SMOKE["val"])
        args.epochs, args.max_length = 1, SMOKE["max_len"]
    train_mean = float(np.mean([r["feedback"] for r in train]))
    print({"train": len(train), "val": len(val), "half": args.half, "device": device}, flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.truncation_side = "left"
    # the hub checkpoint is stored in fp16 and trains to NaN in that dtype, so load fp32 explicitly
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1,
                                                               dtype=torch.float32).to(device)
    steps = -(-len(train) // args.batch_size) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()  # DeBERTa-v3 is unstable in fp16
    args.output.mkdir(parents=True, exist_ok=True)

    history, best, t0 = [], -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train)))
        random.shuffle(order)
        others = other_replies(train, args.seed + epoch) if args.contrast_weight else None
        running = 0.0
        for start in range(0, len(order), args.batch_size):
            idx = order[start: start + args.batch_size]
            batch = [train[i] for i in idx]
            enc = tok([r["text"] for r in batch], [r["response"] for r in batch], truncation="only_first",
                      max_length=args.max_length, padding=True, return_tensors="pt").to(device)
            target = torch.tensor([to_target(r["feedback"]) for r in batch], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                out = model(**enc).logits.squeeze(-1)
            loss = nn.functional.mse_loss(out.float(), target)
            if args.contrast_weight:
                # the real reply should outscore a reply from another dialogue for the same history
                enc_other = tok([r["text"] for r in batch], [others[i] for i in idx], truncation="only_first",
                                max_length=args.max_length, padding=True, return_tensors="pt").to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    out_other = model(**enc_other).logits.squeeze(-1)
                margin = nn.functional.relu(CONTRAST_MARGIN - (out.float() - out_other.float())).mean()
                loss = loss + args.contrast_weight * margin
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            running += loss.item() * len(batch)
        metrics = evaluate(val, *score_split(model, tok, val, args.max_length, args.seed), train_mean)
        history.append({"epoch": epoch, "train_loss": running / len(train), "val": metrics})
        print(json.dumps({"epoch": epoch, "train_loss": round(running / len(train), 4),
                          **{k: round(v, 3) for k, v in metrics.items() if isinstance(v, float)}}), flush=True)
        if metrics["pairwise_accuracy"] > best:
            best = metrics["pairwise_accuracy"]
            model.save_pretrained(args.output)
            tok.save_pretrained(args.output)

    # reload the kept checkpoint from disk and report validation from it
    model = AutoModelForSequenceClassification.from_pretrained(args.output, dtype=torch.float32).to(device)
    val_metrics = evaluate(val, *score_split(model, tok, val, args.max_length, args.seed), train_mean)
    run = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           "halves": {k: len(v) for k, v in halves.items()}, "train_rows": len(train), "train_mean": train_mean,
           "device": device, "bf16": use_bf16, "torch": torch.__version__, "transformers": transformers.__version__,
           "history": history, "val": val_metrics, "minutes": round((time.time() - t0) / 60, 1)}
    (args.output / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    (args.output / "halves.json").write_text(json.dumps({"seed": args.seed, **halves}), encoding="utf-8")
    print(json.dumps({"val": val_metrics}, indent=1))


if __name__ == "__main__":
    main()

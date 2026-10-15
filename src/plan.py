"""Train and serve Kaki's RoBERTa strategy planner. Run from repository root.

With --input response the same classifier reads the supporter reply instead of the history. That model
tags which strategy a reply carries out, and src/controllability.py uses it to check Speak.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Planner:
    def __init__(self, checkpoint, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.tokenizer.truncation_side = "left"
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(checkpoint)
            .to(self.device)
            .eval()
        )

    @torch.inference_mode()
    def predict(self, text):
        """Return the top three names and probabilities from the full 8-class softmax."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Conversation history must be nonempty text.")
        tokens = self.tokenizer(
            text, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        probs = self.model(**tokens).logits.softmax(-1)[0]
        values, indices = probs.topk(3)
        return [
            {"strategy": self.model.config.id2label[int(i)], "probability": float(p)}
            for p, i in zip(values, indices)
        ]


def loader(path, tokenizer, batch_size, shuffle=False, limit=None, field="text"):
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No examples in {path}")

    def collate(batch):
        tokens = tokenizer(
            [r[field] for r in batch],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        tokens["labels"] = torch.tensor([r["label"] for r in batch])
        return tokens

    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.inference_mode()
def evaluate(model, batches, device, labels):
    model.eval()
    truth, pred, covered = [], [], []
    for batch in batches:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        target = batch["labels"]
        truth.extend(target.cpu().tolist())
        pred.extend(logits.argmax(-1).cpu().tolist())
        covered.extend(
            (logits.topk(3).indices == target[:, None]).any(-1).cpu().tolist()
        )
    return dict(
        accuracy=accuracy_score(truth, pred),
        macro_f1=f1_score(
            truth,
            pred,
            labels=list(range(len(labels))),
            average="macro",
            zero_division=0,
        ),
        top3_coverage=float(np.mean(covered)),
        examples=len(truth),
        per_class=classification_report(
            truth,
            pred,
            labels=list(range(len(labels))),
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed/downstream"))
    ap.add_argument("--output", type=Path, default=Path("checkpoints/plan"))
    ap.add_argument("--model", default="FacebookAI/roberta-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--class-weight-power",
        type=float,
        default=0.0,
        help="Weight classes by (N / (K * n_c)) ** power from the training split; 0 disables",
    )
    ap.add_argument(
        "--input",
        choices=["text", "response"],
        default="text",
        help="text: the history (Plan); response: the reply itself (strategy tagger)",
    )
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Explicit final-checkpoint test evaluation; no training",
    )
    args = ap.parse_args()
    if args.smoke and args.evaluate_test:
        ap.error("Smoke tests must not evaluate the test set.")
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        ap.error("Epochs, batch size and learning rate must be positive.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    labels = json.loads((args.data / "strategy_labels.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output.mkdir(parents=True, exist_ok=True)
    if args.evaluate_test:
        planner = Planner(args.output, device)
        if [planner.model.config.id2label[i] for i in range(len(labels))] != labels:
            raise ValueError("Checkpoint and dataset strategy labels differ.")
        metrics = evaluate(
            planner.model,
            loader(
                args.data / "responses_test.jsonl",
                planner.tokenizer,
                args.batch_size,
                field=args.input,
            ),
            device,
            labels,
        )
        (args.output / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2))
        return
    if (args.output / "config.json").exists():
        raise ValueError(
            "Output already contains a model. Choose a new output to preserve it."
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={name: i for i, name in enumerate(labels)},
    ).to(device)
    train = loader(
        args.data / "responses_train.jsonl",
        tokenizer,
        args.batch_size,
        True,
        16 if args.smoke else None,
        args.input,
    )
    val = loader(
        args.data / "responses_val.jsonl",
        tokenizer,
        args.batch_size,
        False,
        16 if args.smoke else None,
        args.input,
    )
    weights = None
    if args.class_weight_power:
        with (args.data / "responses_train.jsonl").open(encoding="utf-8") as f:
            counts = np.bincount(
                [json.loads(line)["label"] for line in f], minlength=len(labels)
            ).clip(min=1)
        weights = torch.tensor(
            (counts.sum() / (len(labels) * counts)) ** args.class_weight_power,
            dtype=torch.float,
            device=device,
        )
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best, records = -1, []
    for epoch in range(1 if args.smoke else args.epochs):
        model.train()
        for batch in train:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            target = batch.pop("labels")
            loss = loss_fn(model(**batch).logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        metrics = evaluate(model, val, device, labels)
        records.append({"epoch": epoch + 1, **metrics})
        print(
            json.dumps(
                {"epoch": epoch + 1, "validation_macro_f1": metrics["macro_f1"]}
            ),
            flush=True,
        )
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
    (args.output / "run.json").write_text(
        json.dumps(
            dict(
                config={
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                device=device,
                validation=records,
            ),
            indent=2,
        )
    )
    planner = Planner(args.output, device)
    result = planner.predict("seeker: I am worried about my exams.")
    assert len(result) == 3 and len({r["strategy"] for r in result}) == 3
    print(json.dumps({"reload_inference": result, "smoke_only": args.smoke}, indent=2))


if __name__ == "__main__":
    main()

"""Fine-tune and evaluate Guard's risk classifier: DistilRoBERTa with two classes (no_risk, risk).

Run from the repository root after src/prep_guard.py:
    python src/train_guard.py --smoke --output checkpoints/guard-smoke    # pipeline check, no test data
    python src/train_guard.py --output checkpoints/guard                  # train, select thresholds on val
    python src/train_guard.py --output checkpoints/guard --evaluate-test  # once, after development

Training keeps the epoch with the best validation average precision, reloads it from disk and selects
the classifier and lexicon thresholds on validation data: the highest risk recall on the labelled
validation posts among thresholds whose bypass rate on ESConv validation turns stays within
selection.max_bypass_rate. ESConv histories come from src/prep.py. The test splits are read only with
--evaluate-test.
"""
import argparse
import collections
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import transformers
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from guard import CONFIG, LABELS, ROOT, Guard, decide, guard_text, match_lexicon, risk_scores

SMOKE = {"train": 64, "val": 64, "esconv": 200, "max_len": 128}


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def binary_metrics(y, flags) -> dict:
    y, flags = np.asarray(y, bool), np.asarray(flags, bool)
    tp, fp = int((y & flags).sum()), int((~y & flags).sum())
    fn, tn = int((y & ~flags).sum()), int((~y & ~flags).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def affirmed(texts: list[str]) -> np.ndarray:
    """True where the lexicon has a match that is neither negated nor quoted."""
    return np.array([any(not m["negated"] and not m["quoted"] for m in match_lexicon(t)) for t in texts])


def flags_at(scores, lexicon, t_model: float, t_lex: float) -> np.ndarray:
    """The decision rule of guard.decide, vectorised."""
    return (scores >= t_model) | (lexicon & (scores >= t_lex))


def select_thresholds(scores, lexicon, y, ref_scores, ref_lexicon, max_bypass_rate: float,
                      step: float) -> tuple[float, float]:
    """(model, lexicon) threshold pair with the highest recall on the risk-labelled posts, among pairs that
    bypass at most max_bypass_rate of the reference ESConv turns. The lexicon threshold is at most the
    model threshold. Ties go to higher precision, then to higher thresholds (fewer flags)."""
    grid = np.round(np.arange(step, 1.0, step), 6)
    best_key, best = None, None
    for t_model in grid:
        for t_lex in grid[grid <= t_model]:
            if flags_at(ref_scores, ref_lexicon, t_model, t_lex).mean() > max_bypass_rate:
                continue
            m = binary_metrics(y, flags_at(scores, lexicon, t_model, t_lex))
            key = (m["recall"], m["precision"], t_model, t_lex)
            if best_key is None or key > best_key:
                best_key, best = key, (float(t_model), float(t_lex))
    if best is None:
        raise ValueError(f"no threshold keeps the ESConv bypass rate at or below {max_bypass_rate}")
    return best


def bypass_stats(rows: list[dict], flags) -> dict:
    """Share of ESConv seeker turns, and of dialogues, where Guard would bypass generation."""
    by_dialogue = collections.defaultdict(bool)
    for r, f in zip(rows, flags):
        by_dialogue[r["dialogue_id"]] |= bool(f)
    return {"turns": len(rows), "turn_rate": round(float(np.mean(flags)), 4),
            "dialogues": len(by_dialogue), "dialogue_rate": round(sum(by_dialogue.values()) / len(by_dialogue), 4)}


def bootstrap_ci(groups, y, flags, n_boot: int = 1000, seed: int = 0) -> dict:
    """95% percentile intervals, resampling users (not posts) with replacement."""
    rng = np.random.default_rng(seed)
    members = collections.defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    members = list(members.values())
    y, flags, stats = np.asarray(y), np.asarray(flags), []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(len(members), size=len(members))])
        m = binary_metrics(y[idx], flags[idx])
        stats.append((m["precision"], m["recall"], m["f1"]))
    lo, hi = np.percentile(stats, [2.5, 97.5], axis=0)
    return {k: [round(float(a), 3), round(float(b), 3)] for k, a, b in zip(("precision", "recall", "f1"), lo, hi)}


def report(rows: list[dict], scores, thresholds: dict) -> tuple[dict, list[dict]]:
    """Metrics for the deployed rule (decide) and its two parts, plus per-row predictions (no text)."""
    y = np.array([r["label"] for r in rows])
    groups = [r["group"] for r in rows]
    scores = np.asarray(scores)
    decisions = [decide(float(s), match_lexicon(r["text"]), thresholds) for r, s in zip(rows, scores)]
    rules = {
        "combined": np.array([d["risk_flag"] for d in decisions]),
        "classifier_only": scores >= thresholds["model_only"],
        "lexicon_only": affirmed([r["text"] for r in rows]),
    }
    out = {"posts": len(rows), "risk_posts": int(y.sum()), "users": len(set(groups)),
           "average_precision": float(average_precision_score(y, scores)),
           "roc_auc": float(roc_auc_score(y, scores)), "rules": {}}
    for name, flags in rules.items():
        out["rules"][name] = {**binary_metrics(y, flags), "ci95": bootstrap_ci(groups, y, flags)}
    flagged = collections.defaultdict(lambda: [0, 0])
    for r, d in zip(rows, decisions):
        flagged[r["label_raw"]][0] += 1
        flagged[r["label_raw"]][1] += d["risk_flag"]
    out["combined_flag_rate_by_label"] = {k: {"posts": n, "flagged": f, "rate": round(f / n, 3)}
                                          for k, (n, f) in sorted(flagged.items())}
    out["combined_triggers"] = dict(collections.Counter(d["trigger"] for d in decisions if d["trigger"]))
    preds = [{"id": r["id"], "label": r["label"], "label_raw": r["label_raw"], "score": round(float(s), 5),
              "lexicon_affirmed": bool(lex), "risk_flag": d["risk_flag"], "trigger": d["trigger"]}
             for r, s, lex, d in zip(rows, scores, rules["lexicon_only"], decisions)]
    return out, preds


def length_batches(n_rows: int, lengths: list[int], batch_size: int, rng: random.Random) -> list[list[int]]:
    """Shuffled batches of similar-length rows: shuffle, sort within chunks of 50 batches, shuffle the
    batches. Most posts are short, so this avoids padding every batch to the longest post."""
    idx = list(range(n_rows))
    rng.shuffle(idx)
    chunk, batches = batch_size * 50, []
    for start in range(0, n_rows, chunk):
        part = sorted(idx[start: start + chunk], key=lambda i: lengths[i])
        batches += [part[j: j + batch_size] for j in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return batches


def write_jsonl(path: Path, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def train(cfg: dict, args, data: dict, device: str) -> tuple[list[dict], list[float]]:
    """Fine-tune, keep the epoch with the best validation average precision. Returns the per-epoch
    history and the best epoch's validation scores."""
    tcfg = cfg["train"]
    tok = AutoTokenizer.from_pretrained(tcfg["model_name"], truncation_side="left")
    model = AutoModelForSequenceClassification.from_pretrained(
        tcfg["model_name"], num_labels=2, id2label=dict(enumerate(LABELS)),
        label2id={name: i for i, name in enumerate(LABELS)}).to(device)

    def collate(rows):
        enc = tok([r["text"] for r in rows], truncation=True, max_length=tcfg["max_len"],
                  padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([r["label"] for r in rows])
        return enc

    rows = data["train"]
    lengths = [len(ids) for ids in tok([r["text"] for r in rows], truncation=True, max_length=tcfg["max_len"])["input_ids"]]
    rng = random.Random(tcfg["seed"])
    n_batches = -(-len(rows) // tcfg["batch_size"])

    # tempered inverse-frequency weights from the train split only, as in Perceive
    counts = np.bincount([r["label"] for r in data["train"]], minlength=2).clip(min=1)
    weights = torch.tensor((counts.sum() / (2 * counts)) ** tcfg["weight_power"], dtype=torch.float)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

    steps = n_batches * tcfg["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = get_linear_schedule_with_warmup(opt, int(tcfg["warmup_ratio"] * steps), steps)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    val_texts = [r["text"] for r in data["val"]]
    val_y = [r["label"] for r in data["val"]]
    history, best_ap, best_scores = [], -1.0, None
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        t0, running = time.time(), 0.0
        for idx in length_batches(len(rows), lengths, tcfg["batch_size"], rng):
            b = {k: v.to(device) for k, v in collate([rows[i] for i in idx]).items()}
            labels = b.pop("labels")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**b).logits
            loss = loss_fn(logits.float(), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            running += loss.item()

        scores = risk_scores(model, tok, val_texts, tcfg["max_len"])
        ap = float(average_precision_score(val_y, scores))
        history.append({"epoch": epoch, "train_loss": running / n_batches, "val_average_precision": ap,
                        "seconds": round(time.time() - t0)})
        print(f"epoch {epoch}: loss {running / n_batches:.4f} | val AP {ap:.3f} | {time.time() - t0:.0f}s", flush=True)
        if ap > best_ap:
            best_ap, best_scores = ap, scores
            model.save_pretrained(args.output)
            tok.save_pretrained(args.output)
            print(f"  saved best checkpoint (val AP {ap:.3f})", flush=True)
    return history, best_scores


def select_on_val(output: Path, cfg: dict, val_rows: list[dict], esconv_rows: list[dict], device: str):
    """Reload the checkpoint from disk, score the risk-labelled validation posts and the ESConv validation
    turns, select thresholds under the configured bypass cap and write thresholds.json. Also returns the
    operating points for other caps, so the choice of cap is documented."""
    model = AutoModelForSequenceClassification.from_pretrained(output).to(device)
    tok = AutoTokenizer.from_pretrained(output, truncation_side="left")
    max_len = cfg["train"]["max_len"]
    scores = np.array(risk_scores(model, tok, [r["text"] for r in val_rows], max_len))
    ref_texts = [guard_text(r["text"], cfg["inference"]["context_turns"]) for r in esconv_rows]
    ref_scores = np.array(risk_scores(model, tok, ref_texts, max_len))
    y = np.array([r["label"] for r in val_rows])
    lex, ref_lex = affirmed([r["text"] for r in val_rows]), affirmed(ref_texts)
    sel = cfg["selection"]

    operating_points = {}
    for cap in sorted({0.01, 0.02, 0.05, 0.10, sel["max_bypass_rate"]}):
        point = {}
        for rule, l, rl in (("combined", lex, ref_lex),
                            ("classifier_only", np.zeros_like(lex), np.zeros_like(ref_lex))):
            t_model, t_lex = select_thresholds(scores, l, y, ref_scores, rl, cap, sel["grid_step"])
            point[rule] = {"model": t_model, "lexicon": t_lex,
                           **binary_metrics(y, flags_at(scores, l, t_model, t_lex)),
                           "esconv_val": bypass_stats(esconv_rows, flags_at(ref_scores, rl, t_model, t_lex))}
        operating_points[str(cap)] = point
    chosen = operating_points[str(sel["max_bypass_rate"])]
    thresholds = {"model": chosen["combined"]["model"], "lexicon": chosen["combined"]["lexicon"],
                  "model_only": chosen["classifier_only"]["model"],
                  "max_bypass_rate": sel["max_bypass_rate"], "selected_on": "validation"}
    (output / "thresholds.json").write_text(json.dumps(thresholds, indent=1))
    return scores, thresholds, operating_points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/processed/guard")
    ap.add_argument("--output", type=Path, default=ROOT / "checkpoints/guard")
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 1 epoch, short inputs; never reads test")
    ap.add_argument("--select", action="store_true",
                    help="re-select thresholds on validation for a trained checkpoint (e.g. after changing "
                         "selection.max_bypass_rate), without retraining")
    ap.add_argument("--evaluate-test", action="store_true", help="score the test splits once with a trained checkpoint")
    ap.add_argument("--flag-rate", type=Path, metavar="JSONL",
                    help="histories in src/prep.py format: report how often a trained Guard bypasses generation")
    args = ap.parse_args()
    if args.smoke and (args.select or args.evaluate_test or args.flag_rate):
        ap.error("smoke runs only train and validate")

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    esconv_dir = ROOT / cfg["selection"]["esconv_dir"]
    if not (esconv_dir / "val.jsonl").exists():
        ap.error(f"ESConv histories not found in {esconv_dir}; run src/prep.py first")
    logging.getLogger("guard").setLevel(logging.ERROR)  # one bypass warning per row is noise here

    if args.flag_rate:
        rows = read_jsonl(args.flag_rate)
        results = Guard(args.output, args.config, device).check_batch([r["text"] for r in rows])
        summary = {"file": str(args.flag_rate), **bypass_stats(rows, [r["bypass_generation"] for r in results]),
                   "triggers": dict(collections.Counter(r["trigger"] for r in results if r["trigger"]))}
        print(json.dumps(summary, indent=1))
        return

    if args.select:
        rows = read_jsonl(args.data / "val.jsonl")
        scores, thresholds, operating_points = select_on_val(args.output, cfg, rows,
                                                             read_jsonl(esconv_dir / "val.jsonl"), device)
        val_metrics, val_preds = report(rows, scores, thresholds)
        write_jsonl(args.output / "preds_val.jsonl", val_preds)
        run = json.loads((args.output / "run.json").read_text())
        run.update(config=cfg, thresholds=thresholds, val_operating_points=operating_points, val=val_metrics)
        (args.output / "run.json").write_text(json.dumps(run, indent=1))
        print(json.dumps({"thresholds": thresholds, "operating_points": operating_points}, indent=1))
        return

    if args.evaluate_test:
        guard = Guard(args.output, args.config, device)
        rows = read_jsonl(args.data / "test.jsonl")
        metrics, preds = report(rows, guard.score([r["text"] for r in rows]), guard.thresholds)
        esconv = read_jsonl(esconv_dir / "test.jsonl")
        results = guard.check_batch([r["text"] for r in esconv])
        metrics["esconv_test"] = {**bypass_stats(esconv, [r["bypass_generation"] for r in results]),
                                  "triggers": dict(collections.Counter(r["trigger"] for r in results if r["trigger"]))}
        (args.output / "test_metrics.json").write_text(json.dumps(metrics, indent=1))
        write_jsonl(args.output / "preds_test.jsonl", preds)
        print(json.dumps(metrics, indent=1))
        return

    if (args.output / "config.json").exists():
        ap.error(f"{args.output} already holds a model; choose a new --output to keep it")
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["train"]["seed"])
    data = {s: read_jsonl(args.data / f"{s}.jsonl") for s in ("train", "val")}
    esconv_val = read_jsonl(esconv_dir / "val.jsonl")
    if args.smoke:
        rng = random.Random(0)
        for s in data:  # keep both classes in the subset
            pos = [r for r in data[s] if r["label"]]
            neg = [r for r in data[s] if not r["label"]]
            data[s] = rng.sample(pos, SMOKE[s] // 2) + rng.sample(neg, SMOKE[s] // 2)
        esconv_val = rng.sample(esconv_val, SMOKE["esconv"])
        cfg["train"].update(epochs=1, max_len=SMOKE["max_len"])
        args.config = args.output / "smoke_config.json"
        args.config.write_text(json.dumps(cfg, indent=1))
    print({s: len(rows) for s, rows in data.items()}, "| device:", device, "| torch", torch.__version__,
          "| transformers", transformers.__version__, flush=True)

    t0 = time.time()
    history, best_scores = train(cfg, args, data, device)
    val_scores, thresholds, operating_points = select_on_val(args.output, cfg, data["val"], esconv_val, device)
    reload_diff = float(np.abs(val_scores - np.array(best_scores)).max())
    if reload_diff > 1e-4:
        raise RuntimeError(f"reloaded checkpoint disagrees with the trained model (max diff {reload_diff:.2e})")

    # end-to-end inference check through the public interface, using the reloaded files
    guard = Guard(args.output, args.config, device)
    check = guard.check([{"speaker": "supporter", "text": "How are you feeling today?"},
                         {"speaker": "seeker", "text": data["val"][0]["text"]}])
    if abs(check["model_score"] - val_scores[0]) > 1e-4:
        raise RuntimeError("Guard.check score differs from the trainer's score for the same text")

    val_metrics, val_preds = report(data["val"], val_scores, thresholds)
    write_jsonl(args.output / "preds_val.jsonl", val_preds)
    run = {"config": cfg, "smoke": args.smoke, "data": {s: len(rows) for s, rows in data.items()},
           "device": device, "torch": torch.__version__, "transformers": transformers.__version__,
           "history": history, "reload_max_abs_diff": reload_diff, "thresholds": thresholds,
           "val_operating_points": operating_points, "val": val_metrics,
           "minutes": round((time.time() - t0) / 60, 1)}
    (args.output / "run.json").write_text(json.dumps(run, indent=1))
    print(json.dumps({"thresholds": thresholds, "reload_max_abs_diff": reload_diff,
                      "val (thresholds chosen here, optimistic)": val_metrics["rules"]}, indent=1))

if __name__ == "__main__":
    main()

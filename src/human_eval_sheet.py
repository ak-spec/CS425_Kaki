"""Blinded human evaluation of the three configurations on the contexts in configs/eval_contexts.json.

build: one CSV row per (context, distinct reply). A context's replies appear in a random order under the
neutral labels A, B and C; when two configurations wrote the same reply it is shown once, so identical
replies cannot reveal which system wrote which. The history is shown on the context's first row. The key
that maps each label to its configurations goes to a separate file; raters only ever get the sheet.
score: reads the raters' filled sheets and the key, unblinds (a shared reply's rating counts for every
configuration that wrote it), and reports each configuration's mean for each criterion, inter-rater
agreement (Krippendorff's alpha, ordinal) and the ratings that differ by two or more points. Rubric and
procedure: docs/evaluation.md.

    python src/human_eval_sheet.py build      # data/human_eval/sheet.csv and data/human_eval/key.json
    python src/human_eval_sheet.py score --sheets data/human_eval/rater1.csv data/human_eval/rater2.csv
"""
import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean

from generate_outputs import load_rows, private_dir
from pipeline import CONFIGS
from select_eval_contexts import history_turns, read_jsonl

ROOT = Path(__file__).resolve().parent.parent
CRITERIA = ("relevance", "acknowledgement", "helpfulness", "fluency")
LABELS = "ABC"
FIELDS = ["context", "label", "history", "reply", *CRITERIA, "comment"]


def show_history(text: str) -> str:
    return "\n".join(f"{t['speaker'].capitalize()}: {t['text']}" for t in history_turns(text))


def build(contexts: list[dict], outputs: dict[str, dict], seed: int = 42) -> tuple[list[dict], dict]:
    """Sheet rows and key. contexts are response rows in sheet order; outputs maps each configuration to
    {(dialogue_id, utterance_idx): output row}."""
    configs = sorted(outputs)
    if len(configs) != len(LABELS):
        raise ValueError(f"need exactly {len(LABELS)} configurations, got {configs}")
    rng = random.Random(seed)
    sheet, key = [], {"seed": seed, "contexts": []}
    for number, row in enumerate(contexts, 1):
        ids = (row["dialogue_id"], row["utterance_idx"])
        missing = [c for c in configs if ids not in outputs[c]]
        if missing:
            raise ValueError(f"no {missing} output for dialogue {ids[0]}, utterance {ids[1]}")
        writers = {}  # reply text -> configurations that wrote it
        for config in configs:
            writers.setdefault(outputs[config][ids]["reply"], []).append(config)
        groups = list(writers.items())
        rng.shuffle(groups)
        key["contexts"].append({"context": number, "dialogue_id": ids[0], "utterance_idx": ids[1],
                                "labels": {label: group for label, (_, group) in zip(LABELS, groups)}})
        for i, (label, (reply, _)) in enumerate(zip(LABELS, groups)):
            sheet.append({"context": number, "label": label, "history": show_history(row["text"]) if i == 0 else "",
                          "reply": reply, **dict.fromkeys(CRITERIA, ""), "comment": ""})
    return sheet, key


def read_sheet(path: Path) -> dict[tuple[int, str], dict]:
    """{(context, label): {criterion: rating}} from a filled sheet; every rating must be a whole number 1-5."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    ratings, problems = {}, []
    for r in rows:
        if not (r.get("context") or "").strip():  # blank rows that spreadsheet exports add
            continue
        unit = (int(r["context"]), r["label"].strip())
        ratings[unit] = {}
        for c in CRITERIA:
            value = (r.get(c) or "").strip()
            try:
                number = float(value)
            except ValueError:
                number = 0.0
            if number not in (1, 2, 3, 4, 5):
                problems.append(f"context {unit[0]} reply {unit[1]} {c}: {value!r}")
                continue
            ratings[unit][c] = int(number)
    if problems:
        raise ValueError(f"{path}: ratings must be whole numbers from 1 to 5\n  " + "\n  ".join(problems))
    return ratings


def krippendorff_alpha(units: list[list], level: str = "ordinal") -> float:
    """Krippendorff's alpha. units holds, for each rated item, the values its raters gave (missing ones
    left out); items with fewer than two values are skipped. level: nominal, ordinal or interval."""
    units = [u for u in units if len(u) >= 2]
    values = sorted({v for u in units for v in u})
    index = {v: i for i, v in enumerate(values)}
    k = len(values)
    coincidence = [[0.0] * k for _ in range(k)]
    for u in units:
        for a, x in enumerate(u):
            for b, y in enumerate(u):
                if a != b:
                    coincidence[index[x]][index[y]] += 1 / (len(u) - 1)
    totals = [sum(row) for row in coincidence]
    n = sum(totals)

    def distance(c, d):
        if level == "nominal":
            return float(c != d)
        if level == "interval":
            return float(values[c] - values[d]) ** 2
        if level == "ordinal":
            lo, hi = min(c, d), max(c, d)
            return (sum(totals[lo: hi + 1]) - (totals[c] + totals[d]) / 2) ** 2
        raise ValueError(f"unknown level {level!r}")

    observed = sum(coincidence[c][d] * distance(c, d) for c in range(k) for d in range(k))
    expected = sum(totals[c] * totals[d] * distance(c, d) for c in range(k) for d in range(k))
    return 1 - (n - 1) * observed / expected if expected else float("nan")


def score(sheets: list[dict], key: dict, gap: int = 2) -> dict:
    """Unblind the raters' sheets and summarise them."""
    configs_of = {(c["context"], label): group for c in key["contexts"] for label, group in c["labels"].items()}
    for i, sheet in enumerate(sheets, 1):
        if set(sheet) != set(configs_of):
            raise ValueError(f"sheet {i} does not have the contexts and labels in the key")
    units = sorted(configs_of)
    configs = sorted({config for group in configs_of.values() for config in group})

    def means(raters):
        return {config: {c: mean(s[u][c] for s in raters for u in units if config in configs_of[u])
                         for c in CRITERIA} for config in configs}

    return {
        "raters": len(sheets), "units": len(units),
        "means": means(sheets),
        "means_per_rater": [means([s]) for s in sheets],
        "alpha_ordinal": {c: krippendorff_alpha([[s[u][c] for s in sheets] for u in units]) for c in CRITERIA},
        "disagreements": [{"context": u[0], "label": u[1], "configs": configs_of[u], "criterion": c,
                           "ratings": [s[u][c] for s in sheets]}
                          for u in units for c in CRITERIA
                          if max(s[u][c] for s in sheets) - min(s[u][c] for s in sheets) >= gap],
    }


def main_build(args):
    folder = private_dir(args.out)
    if (folder / "key.json").exists() and not args.overwrite:
        raise SystemExit(f"{folder / 'key.json'} exists; raters may already hold its sheet. Use --overwrite.")
    contexts = load_rows(args.data, args.contexts)
    outputs = {c: {(r["dialogue_id"], r["utterance_idx"]): r for r in read_jsonl(args.outputs / f"{c}.jsonl")}
               for c in CONFIGS}
    sheet, key = build(contexts, outputs, args.seed)
    with open(folder / "sheet.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sheet)
    (folder / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    same = sum(len(k["labels"]) < len(CONFIGS) for k in key["contexts"])
    print(json.dumps({"contexts": len(contexts), "replies_to_rate": len(sheet), "sheet": str(folder / "sheet.csv"),
                      "key": str(folder / "key.json"), "contexts_with_a_shared_reply": same}))


def main_score(args):
    if len(args.sheets) < 2:
        raise SystemExit("agreement needs at least two raters' sheets")
    key = json.loads(args.key.read_text(encoding="utf-8"))
    report = score([read_sheet(p) for p in args.sheets], key, args.gap)
    report["sheets"] = [str(p) for p in args.sheets]
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"{'':<12}" + "".join(f"{c:>17}" for c in CRITERIA))
    for config, row in report["means"].items():
        print(f"{config:<12}" + "".join(f"{row[c]:>17.2f}" for c in CRITERIA))
    print(f"{'alpha (ord.)':<12}" + "".join(f"{report['alpha_ordinal'][c]:>17.3f}" for c in CRITERIA))
    print(f"\n{len(report['disagreements'])} ratings differ by {args.gap} or more points:")
    for d in report["disagreements"]:
        print(f"  context {d['context']} reply {d['label']} ({', '.join(d['configs'])}), {d['criterion']}: {d['ratings']}")


def main():
    ap = argparse.ArgumentParser(description="Blinded human-evaluation sheet and scoring.")
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="write the blinded sheet and its key")
    b.add_argument("--contexts", type=Path, default=ROOT / "configs/eval_contexts.json")
    b.add_argument("--data", type=Path, default=ROOT / "data/processed/downstream/responses_test.jsonl")
    b.add_argument("--outputs", type=Path, default=ROOT / "data/outputs")
    b.add_argument("--out", type=Path, default=ROOT / "data/human_eval")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--overwrite", action="store_true")
    s = sub.add_parser("score", help="unblind filled sheets and report means and agreement")
    s.add_argument("--sheets", type=Path, nargs="+", required=True)
    s.add_argument("--key", type=Path, default=ROOT / "data/human_eval/key.json")
    s.add_argument("--gap", type=int, default=2, help="report ratings whose range is at least this")
    s.add_argument("--output", type=Path, default=ROOT / "data/human_eval/results.json")
    args = ap.parse_args()
    if args.command == "build":
        main_build(args)
    else:
        main_score(args)


if __name__ == "__main__":
    main()

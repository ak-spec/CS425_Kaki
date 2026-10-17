"""Can people tell the simulated seekers from real ones? The proposal's realism check for src/simulate.py.

build: 20 opening messages in a random order, 10 real (the first message of at least 8 words by the seeker
of an ESConv test dialogue) and 10 simulated (the first message of a simulated seeker in data/simulation),
with an empty `guess` column for "real" or "simulated". The answers go to a separate key.
score: each rater's accuracy against the key. About 50% means the raters could not tell them apart.

    python src/realism_sheet.py build
    python src/realism_sheet.py score --sheets data/human_eval/realism_rater1.csv data/human_eval/realism_rater2.csv
"""

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean

from generate_outputs import private_dir
from select_eval_contexts import read_jsonl

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["item", "message", "guess"]


def real_openers(raw: list[dict], ids: list[int], min_words: int = 8) -> list[str]:
    out = []
    for i in ids:
        for turn in raw[i]["dialog"]:
            text = " ".join(turn["content"].split())
            if turn["speaker"] == "seeker" and len(text.split()) >= min_words:
                out.append(text)
                break
    return out


def build(real: list[str], simulated: list[str], n: int = 10, seed: int = 42) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    items = [(m, "real") for m in rng.sample(real, n)] + [(m, "simulated") for m in rng.sample(simulated, n)]
    rng.shuffle(items)
    sheet = [{"item": i, "message": m, "guess": ""} for i, (m, _) in enumerate(items, 1)]
    return sheet, {"seed": seed, "answers": {str(i): a for i, (_, a) in enumerate(items, 1)}}


SYNONYMS = {"real": "real", "human": "real", "simulated": "simulated", "ai": "simulated"}


def read_guesses(path: Path) -> dict[str, str]:
    """{item: "real" or "simulated"}; "human" and "ai" are accepted too, in any case."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("item") or "").strip()]
    guesses = {r["item"].strip(): SYNONYMS.get((r.get("guess") or "").strip().lower(), "") for r in rows}
    bad = [i for i, g in guesses.items() if g not in ("real", "simulated")]
    if bad:
        raise ValueError(f"{path}: guess must be 'real' or 'simulated' for items {bad}")
    return guesses


def score(sheets: list[dict], key: dict) -> dict:
    answers = key["answers"]
    per = [mean(s[i] == a for i, a in answers.items()) for s in sheets]
    return {
        "raters": len(sheets),
        "items": len(answers),
        "accuracy_per_rater": per,
        "accuracy": mean(per),
        "simulated_called_real": mean(s[i] == "real" for s in sheets for i, a in answers.items() if a == "simulated"),
    }


def main():
    ap = argparse.ArgumentParser(description="Blind real-or-simulated sheet for the seeker simulator.")
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--raw", type=Path, default=ROOT / "data/raw/ESConv.json")
    b.add_argument("--splits", type=Path, default=ROOT / "data/processed/downstream/splits.json")
    b.add_argument("--simulation", type=Path, default=ROOT / "data/simulation/direct.jsonl")
    b.add_argument("--out", type=Path, default=ROOT / "data/human_eval")
    b.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("score")
    s.add_argument("--sheets", type=Path, nargs="+", required=True)
    s.add_argument("--key", type=Path, default=ROOT / "data/human_eval/realism_key.json")
    args = ap.parse_args()
    if args.command == "build":
        raw = json.loads(args.raw.read_text(encoding="utf-8"))
        test = json.loads(args.splits.read_text(encoding="utf-8"))["test"]
        simulated = [c["turns"][0]["text"] for c in read_jsonl(args.simulation) if c["turns"]]
        sheet, key = build(real_openers(raw, test), simulated, seed=args.seed)
        folder = private_dir(args.out)
        with open(folder / "realism_sheet.csv", "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(sheet)
        (folder / "realism_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
        print(json.dumps({"items": len(sheet), "sheet": str(folder / "realism_sheet.csv")}))
    else:
        key = json.loads(args.key.read_text(encoding="utf-8"))
        print(json.dumps(score([read_guesses(p) for p in args.sheets], key), indent=1))


if __name__ == "__main__":
    main()

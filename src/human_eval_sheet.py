"""Blinded human evaluation of the three configurations and BlenderBot-small (B1).

Contexts are the 20 test histories in configs/eval_contexts.json and, with --queries, the handwritten
Singlish messages in configs/eval_queries.json (their replies come from src/benchmark.py --queries).

build: one CSV row per (context, distinct reply). A context's replies appear in a random order under the
neutral labels A to D; when two systems wrote the same reply it is shown once, so identical replies
cannot reveal which system wrote which. The history is shown on the context's first row. The key
that maps each label to its configurations goes to a separate file; raters only ever get the sheet.
score: reads the raters' filled sheets and the key, unblinds (a shared reply's rating counts for every
configuration that wrote it), and reports each configuration's mean for each criterion, inter-rater
agreement (Krippendorff's alpha, ordinal) and the ratings that differ by two or more points. Rubric and
procedure: docs/evaluation.md.

    python src/human_eval_sheet.py build --queries configs/eval_queries.json   # sheet.csv and key.json
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
CRITERIA = ("relevance", "acknowledgement", "helpfulness", "fluency", "safety")
LABELS = "ABCD"
SYSTEMS = (*CONFIGS, "blenderbot")
FIELDS = ["context", "label", "history", "reply", *CRITERIA, "comment"]


def show_history(text: str) -> str:
    return "\n".join(f"{t['speaker'].capitalize()}: {t['text']}" for t in history_turns(text))


def build(contexts: list[dict], outputs: dict[str, dict], seed: int = 42) -> tuple[list[dict], dict]:
    """Sheet rows and key. contexts are response rows in sheet order; outputs maps each configuration to
    {(dialogue_id, utterance_idx): output row}."""
    configs = sorted(outputs)
    if not 2 <= len(configs) <= len(LABELS):
        raise ValueError(f"need 2 to {len(LABELS)} systems, got {configs}")
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


def read_sheet(path: Path, allow_missing: bool = False) -> dict[tuple[int, str], dict]:
    """{(context, label): {criterion: rating}} from a filled sheet; every rating must be a whole number 1-5.
    With allow_missing, a blank cell is left out (and reported by score) instead of stopping."""
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
            if allow_missing and not value:
                continue
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
        return {config: {c: mean(s[u][c] for s in raters for u in units if config in configs_of[u] and c in s[u])
                         for c in CRITERIA} for config in configs}

    def given(u, c):  # the ratings present for one reply and criterion; missing cells are left out
        return [s[u][c] for s in sheets if c in s[u]]

    return {
        "raters": len(sheets), "units": len(units),
        "missing": [{"rater": i, "context": u[0], "label": u[1], "criterion": c}
                    for i, s in enumerate(sheets, 1) for u in units for c in CRITERIA if c not in s[u]],
        "means": means(sheets),
        "means_per_rater": [means([s]) for s in sheets],
        "alpha_ordinal": {c: krippendorff_alpha([given(u, c) for u in units]) for c in CRITERIA},
        "disagreements": [{"context": u[0], "label": u[1], "configs": configs_of[u], "criterion": c,
                           "ratings": given(u, c)}
                          for u in units for c in CRITERIA
                          if len(given(u, c)) > 1 and max(given(u, c)) - min(given(u, c)) >= gap],
    }


def query_rows(path: Path) -> list[dict]:
    """Handwritten opening messages as context rows: a one-message history and no reference reply."""
    queries = json.loads(Path(path).read_text(encoding="utf-8"))["queries"]
    return [{"dialogue_id": q["id"], "utterance_idx": 0, "text": f"seeker: {' '.join(q['text'].split())}"}
            for q in queries]


def query_outputs(path: Path) -> dict[str, dict]:
    """{system: {(query id, 0): {"reply": ...}}} from the replies src/benchmark.py saved."""
    return {run["system"]: {(r["id"], 0): {"reply": r["reply"]} for r in run["replies"]}
            for run in json.loads(Path(path).read_text(encoding="utf-8"))}


def main_build(args):
    folder = private_dir(args.out)
    if (folder / "key.json").exists() and not args.overwrite:
        raise SystemExit(f"{folder / 'key.json'} exists; raters may already hold its sheet. Use --overwrite.")
    contexts = load_rows(args.data, args.contexts)
    outputs = {c: {(r["dialogue_id"], r["utterance_idx"]): r for r in read_jsonl(args.outputs / f"{c}.jsonl")}
               for c in args.systems}
    if args.queries:
        contexts += query_rows(args.queries)
        for system, replies in query_outputs(args.query_replies).items():
            if system in outputs:
                outputs[system].update(replies)
    sheet, key = build(contexts, outputs, args.seed)
    with open(folder / "sheet.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sheet)
    (folder / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    same = sum(len(k["labels"]) < len(args.systems) for k in key["contexts"])
    print(json.dumps({"contexts": len(contexts), "replies_to_rate": len(sheet), "sheet": str(folder / "sheet.csv"),
                      "key": str(folder / "key.json"), "contexts_with_a_shared_reply": same}))


def spearman(x: list, y: list) -> float:
    """Spearman's rank correlation, ties given their average rank."""
    def ranks(v):
        order = sorted(range(len(v)), key=v.__getitem__)
        r, i = [0.0] * len(v), 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = mean(rx), mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx, sy = sum((a - mx) ** 2 for a in rx) ** 0.5, sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (sx * sy) if sx and sy else float("nan")


def judge_agreement(sheets: list[dict], key: dict, judge_rows: list[dict]) -> dict:
    """Per criterion: Spearman of the judge model's rating with the raters' mean rating, and of the first two
    raters with each other, over the replies the judge also rated (the test contexts, not the Singlish ones)."""
    judged = {(r["dialogue_id"], r["utterance_idx"]): r["scores"] for r in judge_rows if r["subset"] == "human_eval"}
    units = [(c["context"], label, group[0], (c["dialogue_id"], c["utterance_idx"]))
             for c in key["contexts"] for label, group in c["labels"].items()]
    units = [u for u in units if u[3] in judged and judged[u[3]][u[2]]]
    out = {}
    for c in CRITERIA:
        rows = [(judged[k][system][c], [s[(ctx, label)][c] for s in sheets if c in s[(ctx, label)]])
                for ctx, label, system, k in units]
        rows = [(j, h) for j, h in rows if len(h) == len(sheets)]
        out[c] = {"replies": len(rows),
                  "judge_vs_raters": round(spearman([j for j, _ in rows], [mean(h) for _, h in rows]), 3),
                  "rater1_vs_rater2": round(spearman([h[0] for _, h in rows], [h[1] for _, h in rows]), 3)}
    return out


def per_context(sheets: list[dict], key: dict) -> dict:
    """{criterion or "overall": {system: {context: mean rating over raters}}}; "overall" averages the criteria."""
    out = {c: {} for c in (*CRITERIA, "overall")}
    for ctx in key["contexts"]:
        for label, group in ctx["labels"].items():
            unit = (ctx["context"], label)
            values = {c: [s[unit][c] for s in sheets if c in s[unit]] for c in CRITERIA}
            for system in group:
                for c in CRITERIA:
                    out[c].setdefault(system, {})[ctx["context"]] = mean(values[c])
                out["overall"].setdefault(system, {})[ctx["context"]] = mean(mean(v) for v in values.values())
    return out


def paired_differences(sheets: list[dict], key: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean difference between each pair of systems over the contexts both answered, with a 95% bootstrap
    interval over contexts, per criterion and overall; and each system's overall mean on the test contexts
    and on the handwritten messages separately."""
    import itertools
    import random
    scores = per_context(sheets, key)
    rng = random.Random(seed)
    diffs = {}
    for a, b in itertools.combinations(sorted(scores["overall"]), 2):
        for c, table in scores.items():
            d = [table[a][k] - table[b][k] for k in sorted(set(table[a]) & set(table[b]))]
            boots = sorted(mean(rng.choice(d) for _ in d) for _ in range(n_boot))
            diffs.setdefault(f"{a} - {b}", {})[c] = {"mean": round(mean(d), 3),
                                                      "ci95": [round(boots[int(0.025 * n_boot)], 3),
                                                               round(boots[int(0.975 * n_boot) - 1], 3)]}
    kind = {ctx["context"]: "test" if isinstance(ctx["dialogue_id"], int) else "handwritten"
            for ctx in key["contexts"]}
    subsets = {k: {s: round(mean(v for ctx, v in table.items() if kind[ctx] == k), 3)
                   for s, table in scores["overall"].items()} for k in sorted(set(kind.values()))}
    return {"differences": diffs, "overall_by_subset": subsets}


def main_score(args):
    if len(args.sheets) < 2:
        raise SystemExit("agreement needs at least two raters' sheets")
    key = json.loads(args.key.read_text(encoding="utf-8"))
    sheets = [read_sheet(p, args.allow_missing) for p in args.sheets]
    report = score(sheets, key, args.gap)
    report.update(paired_differences(sheets, key))
    if args.judge:
        report["judge_agreement"] = judge_agreement(sheets, key, read_jsonl(args.judge))
    report["sheets"] = [str(p) for p in args.sheets]
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"{'':<12}" + "".join(f"{c:>17}" for c in CRITERIA))
    for config, row in report["means"].items():
        print(f"{config:<12}" + "".join(f"{row[c]:>17.2f}" for c in CRITERIA))
    print(f"{'alpha (ord.)':<12}" + "".join(f"{report['alpha_ordinal'][c]:>17.3f}" for c in CRITERIA))
    if report["missing"]:
        print(f"missing ratings, left out: {report['missing']}")
    print("overall mean by subset:", json.dumps(report["overall_by_subset"]))
    for pair, d in report["differences"].items():
        print(f"{pair}: overall {d['overall']['mean']} {d['overall']['ci95']}")
    for c, a in report.get("judge_agreement", {}).items():
        print(f"{c}: judge vs raters {a['judge_vs_raters']}, rater 1 vs rater 2 {a['rater1_vs_rater2']} ({a['replies']} replies)")
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
    b.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    b.add_argument("--queries", type=Path, help="handwritten messages to add, e.g. configs/eval_queries.json")
    b.add_argument("--query-replies", type=Path, default=ROOT / "data/outputs/eval_queries/benchmark_cuda.json",
                   help="replies to --queries saved by src/benchmark.py")
    b.add_argument("--out", type=Path, default=ROOT / "data/human_eval")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--overwrite", action="store_true")
    s = sub.add_parser("score", help="unblind filled sheets and report means and agreement")
    s.add_argument("--sheets", type=Path, nargs="+", required=True)
    s.add_argument("--key", type=Path, default=ROOT / "data/human_eval/key.json")
    s.add_argument("--gap", type=int, default=2, help="report ratings whose range is at least this")
    s.add_argument("--output", type=Path, default=ROOT / "data/human_eval/results.json")
    s.add_argument("--allow-missing", action="store_true", help="leave blank cells out instead of stopping")
    s.add_argument("--judge", type=Path, help="judge_rows.jsonl from src/judge.py, to compare with the raters")
    args = ap.parse_args()
    if args.command == "build":
        main_build(args)
    else:
        main_score(args)


if __name__ == "__main__":
    main()

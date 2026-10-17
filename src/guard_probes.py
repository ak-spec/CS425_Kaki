"""Guard on the handwritten red-team probes in configs/guard_probes.json.

The proposal's safety comparison of three detectors on the same messages:
    lexicon     any affirmed lexicon match (not negated, not quoted), whatever the score
    classifier  model score at or above the model threshold
    guard       decide() in src/guard.py: the classifier, or an affirmed match with a score at or above
                the lower lexicon threshold
    either      the lexicon or the classifier; shown for comparison only, since choosing it after seeing
                these probes would tune Guard on its test cases
The probes were written before this script was run and are never used for training or thresholds.

    python src/guard_probes.py      # writes checkpoints/guard/probes.json and prints the tables
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from guard import Guard, decide, match_lexicon

ROOT = Path(__file__).resolve().parent.parent
METHODS = ("lexicon", "classifier", "guard", "either")


def flags(score: float, matches: list[dict], thresholds: dict) -> dict:
    lexicon = any(not m["negated"] and not m["quoted"] for m in matches)
    return {
        "lexicon": lexicon,
        "classifier": score >= thresholds["model"],
        "guard": decide(score, matches, thresholds)["bypass_generation"],
        "either": lexicon or score >= thresholds["model"],
    }


def counts(truth: list[bool], pred: list[bool]) -> dict:
    tp = sum(t and p for t, p in zip(truth, pred))
    fp = sum(p and not t for t, p in zip(truth, pred))
    fn = sum(t and not p for t, p in zip(truth, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": len(truth) - tp - fp - fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/guard")
    ap.add_argument("--probes", type=Path, default=ROOT / "configs/guard_probes.json")
    args = ap.parse_args()
    probes = json.loads(args.probes.read_text(encoding="utf-8"))["probes"]
    guard = Guard(args.checkpoint)
    scores = guard.score([p["text"] for p in probes])
    rows = []
    for p, s in zip(probes, scores):
        rows.append(
            {
                "id": p["id"],
                "group": p["group"],
                "escalate": p["escalate"],
                "score": round(s, 3),
                **flags(s, match_lexicon(p["text"]), guard.thresholds),
            }
        )
    truth = [r["escalate"] for r in rows]
    result = {
        "thresholds": guard.thresholds,
        "probes": len(rows),
        "overall": {m: counts(truth, [r[m] for r in rows]) for m in METHODS},
        "by_group": {},
        "rows": rows,
    }
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    for g, rs in groups.items():
        result["by_group"][g] = {
            "n": len(rs),
            "escalate": rs[0]["escalate"],
            **{m: sum(r[m] for r in rs) for m in METHODS},
        }
    (args.checkpoint / "probes.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("| Detector | Precision | Recall | F1 | False alarms | Missed |\n|---|---|---|---|---|---|")
    for m in METHODS:
        c = result["overall"][m]
        print(f"| {m} | {c['precision']:.2f} | {c['recall']:.2f} | {c['f1']:.2f} | {c['fp']} | {c['fn']} |")
    print(
        "\n| Group | Should escalate | Probes | Lexicon | Classifier | Guard | Either |\n|---|---|---|---|---|---|---|"
    )
    for g, c in result["by_group"].items():
        print(
            f"| {g} | {'yes' if c['escalate'] else 'no'} | {c['n']} | " + " | ".join(str(c[m]) for m in METHODS) + " |"
        )


if __name__ == "__main__":
    main()

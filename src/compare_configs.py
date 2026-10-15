"""Paired comparison of the three configurations on the same test histories.

For every history answered by all three configurations, BERTScore F1 and ROUGE-L of each reply against
the reference are computed per history, and each pair of configurations gets the mean difference with a
95% bootstrap interval (resampling histories). An interval that contains 0 means the test set does not
show a reliable difference. Bypassed histories are left out, as in src/metrics.py.

    python src/compare_configs.py                       # after src/generate_outputs.py
    python src/compare_configs.py --subset after_seeker
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer

from metrics import bert_scorer
from pipeline import CONFIGS

ROOT = Path(__file__).resolve().parent.parent


def paired_rows(outputs: Path, subset: str) -> dict[str, list[dict]]:
    """Rows answered by every configuration, aligned by (dialogue_id, utterance_idx)."""
    data = {}
    for c in CONFIGS:
        with open(outputs / f"{c}.jsonl", encoding="utf-8") as f:
            data[c] = {(r["dialogue_id"], r["utterance_idx"]): r for r in map(json.loads, f)}
    keys = sorted(set.intersection(*(set(d) for d in data.values())))
    keys = [k for k in keys if not data[CONFIGS[0]][k]["bypassed"]
            and (subset == "all" or data[CONFIGS[0]][k]["after_seeker"])]
    return {c: [data[c][k] for k in keys] for c in CONFIGS}


def bootstrap_diff(a, b, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean of a - b over histories with a 95% percentile interval."""
    diffs = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = [diffs[rng.integers(len(diffs), size=len(diffs))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": round(float(diffs.mean()), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, default=ROOT / "data/outputs")
    ap.add_argument("--subset", choices=["all", "after_seeker"], default="after_seeker")
    ap.add_argument("--bertscore-model", default="roberta-large")
    ap.add_argument("--bertscore-layer", type=int, default=17)
    args = ap.parse_args()

    rows = paired_rows(args.outputs, args.subset)
    references = [r["reference"] for r in rows[CONFIGS[0]]]
    score, bertscore_hash = bert_scorer(args.bertscore_model, args.bertscore_layer)
    rouge = rouge_scorer.RougeScorer(["rougeL"])
    per_row = {}
    for c in CONFIGS:
        replies = [r["reply"] for r in rows[c]]
        per_row[c] = {"bertscore_f1": [100 * x for x in score(replies, references)],
                      "rouge_l": [100 * rouge.score(ref, rep)["rougeL"].fmeasure for ref, rep in zip(references, replies)]}
    result = {"subset": args.subset, "histories": len(references), "bertscore": bertscore_hash,
              "means": {c: {m: round(float(np.mean(v)), 2) for m, v in per_row[c].items()} for c in CONFIGS},
              "differences": {f"{a} - {b}": {m: bootstrap_diff(per_row[a][m], per_row[b][m]) for m in per_row[a]}
                              for a, b in itertools.combinations(CONFIGS, 2)}}
    out = args.outputs / f"paired_{args.subset}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

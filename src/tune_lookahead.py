"""Choose the look-ahead weight on validation: critic score + weight * planner probability.

Reads look-ahead outputs for validation histories and, for each weight, scores the selected candidate
against the reference reply with BERTScore F1 (the same scorer as src/metrics.py). Every candidate is
scored once, so the sweep needs no new generation. "planner only" ignores the Critic, which reproduces
the planning reply; weight 0 uses the Critic alone. Only histories that end with a seeker turn and were
not bypassed by Guard are used. The chosen weight is written to CRITIC/lookahead.json.

    python src/generate_outputs.py --data data/processed/downstream/responses_val.jsonl \\
        --configs lookahead --critic checkpoints/critic --output data/outputs-val
    python src/tune_lookahead.py --outputs data/outputs-val/lookahead.jsonl --critic checkpoints/critic
"""

import argparse
import json
from pathlib import Path

import numpy as np

from metrics import bert_scorer

WEIGHTS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, None]  # None: planner probability only


def select(candidates: list[dict], weight) -> int:
    """Index of the candidate the pipeline would send with this weight."""

    def score(c):
        if weight is None:
            return c["planner_probability"]
        return c["critic_score"] + weight * c["planner_probability"]

    return max(range(len(candidates)), key=lambda i: score(candidates[i]))


def paired_interval(diffs, n_boot: int = 2000, seed: int = 0) -> list[float]:
    """95% bootstrap interval for the mean of per-history differences."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs)
    means = [diffs[rng.integers(len(diffs), size=len(diffs))].mean() for _ in range(n_boot)]
    return [round(float(x), 4) for x in np.percentile(means, [2.5, 97.5])]


def sweep(rows: list[dict], f1: list[list[float]]) -> list[dict]:
    """Mean F1 of the selected candidate for each weight, and how often it differs from planner only."""
    planner = [select(r["candidates"], None) for r in rows]
    table = []
    for weight in WEIGHTS:
        chosen = [select(r["candidates"], weight) for r in rows]
        table.append(
            {
                "weight": "planner only" if weight is None else weight,
                "bertscore_f1": round(float(np.mean([f[c] for f, c in zip(f1, chosen)])), 4),
                "differs_from_planning": round(float(np.mean([c != p for c, p in zip(chosen, planner)])), 3),
            }
        )
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, required=True, help="lookahead.jsonl from generate_outputs.py")
    ap.add_argument("--critic", type=Path, required=True, help="Critic checkpoint used for the outputs")
    ap.add_argument("--bertscore-model", default="roberta-large")
    ap.add_argument("--bertscore-layer", type=int, default=17)
    args = ap.parse_args()

    with open(args.outputs, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    rows = [r for r in rows if r["after_seeker"] and not r["bypassed"] and len(r["candidates"]) == 3]
    if any(c["critic_score"] is None for r in rows for c in r["candidates"]):
        raise ValueError("outputs have no critic scores; generate them with --critic")

    score, bertscore_hash = bert_scorer(args.bertscore_model, args.bertscore_layer)
    flat = score(
        [c["response"] for r in rows for c in r["candidates"]], [r["reference"] for r in rows for _ in r["candidates"]]
    )
    f1 = [list(flat[3 * i : 3 * i + 3]) for i in range(len(rows))]

    table = sweep(rows, f1)
    best = max(table, key=lambda t: t["bertscore_f1"])
    weight = None if best["weight"] == "planner only" else best["weight"]
    planner = [select(r["candidates"], None) for r in rows]
    chosen = [select(r["candidates"], weight) for r in rows]
    result = {
        "critic_weight": weight,
        "selected_on": "validation",
        "histories": len(rows),
        "bertscore": bertscore_hash,
        "table": table,
        "best_minus_planner_only": {
            "mean": round(float(np.mean([f[c] - f[p] for f, c, p in zip(f1, chosen, planner)])), 4),
            "ci95": paired_interval([f[c] - f[p] for f, c, p in zip(f1, chosen, planner)]),
        },
    }
    (args.critic / "lookahead.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

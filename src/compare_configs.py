"""Paired comparison of the systems on the same test histories.

For every history answered by all systems, BERTScore F1 and ROUGE-L of each reply against the reference
are computed per history, and each pair of systems gets the mean difference with a 95% bootstrap interval
(resampling histories). An interval that contains 0 means the test set does not show a reliable
difference. Bypassed histories are left out, as in src/metrics.py. Systems are the three configurations
and, when their output files exist, the baselines of src/baselines.py.

--critic adds the rating a held-out Critic predicts for each reply (the proposal's Eval-Critic, trained on
the other half of the training dialogues from the Critic that selects look-ahead replies). It needs no
reference, but it shares its training objective with that Critic, so a look-ahead advantage on it alone
is not evidence; docs/results.md reads it together with the reference metrics.

    python src/compare_configs.py                       # after src/generate_outputs.py
    python src/compare_configs.py --subset all --critic checkpoints/eval-critic-contrast
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer

from metrics import SYSTEMS, bert_scorer
from select_eval_contexts import read_jsonl

ROOT = Path(__file__).resolve().parent.parent


def paired_rows(outputs: Path, subset: str, systems=SYSTEMS) -> dict[str, list[dict]]:
    """Rows answered by every system with an output file, aligned by (dialogue_id, utterance_idx)."""
    data = {}
    for c in systems:
        if (outputs / f"{c}.jsonl").exists():
            with open(outputs / f"{c}.jsonl", encoding="utf-8") as f:
                data[c] = {(r["dialogue_id"], r["utterance_idx"]): r for r in map(json.loads, f)}
    first = next(iter(data.values()))
    keys = sorted(set.intersection(*(set(d) for d in data.values())))
    keys = [k for k in keys if not first[k]["bypassed"] and (subset == "all" or first[k]["after_seeker"])]
    return {c: [d[k] for k in keys] for c, d in data.items()}


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
    ap.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS),
                    help="systems to compare; those without an output file are skipped")
    ap.add_argument("--bertscore-model", default="roberta-large")
    ap.add_argument("--bertscore-layer", type=int, default=17)
    ap.add_argument("--critic", type=Path, help="held-out Critic checkpoint, e.g. checkpoints/eval-critic-contrast")
    ap.add_argument("--data", type=Path, default=ROOT / "data/processed/downstream/responses_test.jsonl",
                    help="test rows, for the histories the Critic reads")
    args = ap.parse_args()

    rows = paired_rows(args.outputs, args.subset, args.systems)
    systems = list(rows)
    first = rows[systems[0]]
    references = [r["reference"] for r in first]
    score, bertscore_hash = bert_scorer(args.bertscore_model, args.bertscore_layer)
    rouge = rouge_scorer.RougeScorer(["rougeL"])
    per_row = {}
    for c in systems:
        replies = [r["reply"] for r in rows[c]]
        per_row[c] = {"bertscore_f1": [100 * x for x in score(replies, references)],
                      "rouge_l": [100 * rouge.score(ref, rep)["rougeL"].fmeasure for ref, rep in zip(references, replies)]}
    if args.critic:
        from critic import Critic, predict_ratings
        critic = Critic(args.critic)
        text = {(r["dialogue_id"], r["utterance_idx"]): r["text"] for r in read_jsonl(args.data)}
        histories = [text[(r["dialogue_id"], r["utterance_idx"])] for r in first]
        for c in systems:
            per_row[c]["critic_rating"] = predict_ratings(critic.model, critic.tokenizer, histories,
                                                          [r["reply"] for r in rows[c]], critic.max_len)
        per_row["reference"] = {"critic_rating": predict_ratings(critic.model, critic.tokenizer, histories,
                                                                 references, critic.max_len)}
    result = {"subset": args.subset, "histories": len(references), "bertscore": bertscore_hash,
              "critic": str(args.critic) if args.critic else None,
              "means": {c: {m: round(float(np.mean(v)), 3) for m, v in per_row[c].items()} for c in per_row},
              "differences": {f"{a} - {b}": {m: bootstrap_diff(per_row[a][m], per_row[b][m]) for m in per_row[a]}
                              for a, b in itertools.combinations(systems, 2)}}
    out = args.outputs / f"paired_{args.subset}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

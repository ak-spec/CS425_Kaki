"""Ablations of the planning configuration: what changes when Speak gets better or no inputs.

Every variant uses the conditioned Speak adapter on the test histories of data/outputs/planning.jsonl that
Guard did not bypass, and changes one input:
    planning          Plan's top strategy and Perceive's emotion, as in the planning configuration
    oracle_strategy   the reference strategy instead of Plan's: what a perfect Plan would give
    oracle_emotion    the reference emotion instead of Perceive's
    constant_emotion  the most frequent training emotion for every history: the emotion signal removed
All four are generated here with batched greedy decoding, so compare them with each other rather than
with the per-turn replies in data/outputs (padding can change a reply slightly). Scores are for histories
ending with a seeker message, with paired bootstrap intervals against the planning variant.

    python src/ablations.py      # data/outputs/ablations.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from compare_configs import bootstrap_diff
from generate_outputs import private_dir
from metrics import bert_scorer, bleu, distinct, words
from select_eval_contexts import read_jsonl

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = ("planning", "oracle_strategy", "oracle_emotion", "constant_emotion")


def inputs(rows: list[dict], text: dict, constant: str) -> dict[str, list[tuple]]:
    """(history, emotion, strategy) per row for each variant."""
    out = {v: [] for v in VARIANTS}
    for r in rows:
        h = text[(r["dialogue_id"], r["utterance_idx"])]
        out["planning"].append((h, r["perceived"]["emotion"], r["strategy"]))
        out["oracle_strategy"].append((h, r["perceived"]["emotion"], r["reference_strategy"]))
        out["oracle_emotion"].append((h, r["emotion"], r["strategy"]))
        out["constant_emotion"].append((h, constant, r["strategy"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--outputs", type=Path, default=ROOT / "data/outputs")
    ap.add_argument("--data", type=Path, default=ROOT / "data/processed/downstream")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    rows = [r for r in read_jsonl(args.outputs / "planning.jsonl") if not r["bypassed"]][: args.limit]
    text = {(r["dialogue_id"], r["utterance_idx"]): r["text"] for r in read_jsonl(args.data / "responses_test.jsonl")}
    constant = Counter(r["emotion"] for r in read_jsonl(args.data / "responses_train.jsonl")).most_common(1)[0][0]
    from speak import Speaker
    speaker = Speaker(args.checkpoints / "speak-conditioned")
    replies = {}
    for v, triples in inputs(rows, text, constant).items():
        h, e, s = zip(*triples)
        replies[v] = speaker.generate_batch(list(h), list(e), list(s), batch_size=args.batch_size)
        print(json.dumps({"variant": v, "replies": len(replies[v])}), flush=True)
    del speaker

    keep = [i for i, r in enumerate(rows) if r["after_seeker"]]
    refs = [rows[i]["reference"] for i in keep]
    score, bertscore_hash = bert_scorer("roberta-large", 17)
    from rouge_score import rouge_scorer
    rouge = rouge_scorer.RougeScorer(["rougeL"])
    per = {}
    result = {"histories": len(keep), "constant_emotion": constant, "bertscore": bertscore_hash,
              "strategy_changed_by_oracle": round(mean(rows[i]["strategy"] != rows[i]["reference_strategy"]
                                                       for i in keep), 3),
              "emotion_changed_by_oracle": round(mean(rows[i]["perceived"]["emotion"] != rows[i]["emotion"]
                                                      for i in keep), 3),
              "variants": {}, "against_planning": {}}
    for v in VARIANTS:
        out = [replies[v][i] for i in keep]
        per[v] = {"bertscore_f1": [100 * x for x in score(out, refs)],
                  "rouge_l": [100 * rouge.score(ref, o)["rougeL"].fmeasure for ref, o in zip(refs, out)]}
        result["variants"][v] = {"bleu2": round(bleu(out, refs, 2)[0], 2),
                                 "rouge_l": round(mean(per[v]["rouge_l"]), 2),
                                 "bertscore_f1": round(mean(per[v]["bertscore_f1"]), 2),
                                 "distinct2": round(100 * distinct(out, 2), 2),
                                 "length": round(mean(len(words(o)) for o in out), 1)}
        if v != "planning":
            result["against_planning"][v] = {m: bootstrap_diff(per[v][m], per["planning"][m]) for m in per[v]}
    folder = private_dir(args.outputs)
    (folder / "ablations.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    with open(folder / "ablation_replies.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"dialogue_id": r["dialogue_id"], "utterance_idx": r["utterance_idx"],
                                **{v: replies[v][i] for v in VARIANTS}}, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

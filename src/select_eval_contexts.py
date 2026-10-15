"""Choose the human-evaluation contexts before any system output exists.

Twenty contexts from twenty distinct ESConv test dialogues (rows of
data/processed/downstream/responses_test.jsonl). A row is eligible when its history ends with a seeker
message, holds at least two seeker messages and at least 20 seeker words in total (so the reply has
something to connect to and feelings to acknowledge), and has at most 250 words, so two raters can read
every context in one sitting. Dialogues are drawn in turn from the priority problems (academic pressure,
problems with friends), then from the other in-scope problems if those run out, and one eligible row is
drawn from each chosen dialogue. The reference reply's text and strategy are never read. The output holds
IDs, the seed and the rule only.

    python src/select_eval_contexts.py      # writes configs/eval_contexts.json
"""
import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRIORITY = ("academic pressure", "problems with friends")
TAG = re.compile(r" \| (?=(?:seeker|supporter): )")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def history_turns(text: str) -> list[dict]:
    """'speaker: text | speaker: text' back into {"speaker", "text"} turns."""
    turns = []
    for part in TAG.split(text.strip()):
        speaker, sep, said = part.partition(": ")
        if speaker not in ("seeker", "supporter") or not sep or not said.strip():
            raise ValueError(f"not a tagged history turn: {part[:40]!r}")
        turns.append({"speaker": speaker, "text": said})
    return turns


def eligible(row: dict, min_seeker_turns: int = 2, min_seeker_words: int = 20, max_words: int = 250) -> bool:
    turns = history_turns(row["text"])
    seeker = [t["text"] for t in turns if t["speaker"] == "seeker"]
    return (turns[-1]["speaker"] == "seeker"
            and len(seeker) >= min_seeker_turns
            and sum(len(t.split()) for t in seeker) >= min_seeker_words
            and sum(len(t["text"].split()) for t in turns) <= max_words)


def select(rows: list[dict], n: int = 20, seed: int = 42, priority=PRIORITY, min_seeker_turns: int = 2,
           min_seeker_words: int = 20, max_words: int = 250) -> list[dict]:
    """n {dialogue_id, utterance_idx} from n distinct dialogues, sorted by dialogue."""
    by_dialogue = collections.defaultdict(list)
    for r in rows:
        if eligible(r, min_seeker_turns, min_seeker_words, max_words):
            by_dialogue[r["dialogue_id"]].append(r)
    problem = {d: rs[0]["problem"] for d, rs in by_dialogue.items()}
    in_scope = {d: rs[0]["in_scope"] for d, rs in by_dialogue.items()}
    rng = random.Random(seed)
    pools = [sorted(d for d in by_dialogue if problem[d] == p) for p in priority]
    rest = sorted(d for d in by_dialogue if in_scope[d] and problem[d] not in priority)
    for pool in pools + [rest]:
        rng.shuffle(pool)
    chosen = []
    while len(chosen) < n and any(pools):  # alternate between the priority problems
        for pool in pools:
            if pool and len(chosen) < n:
                chosen.append(pool.pop())
    chosen += rest[: n - len(chosen)]
    if len(chosen) < n:
        raise ValueError(f"only {len(chosen)} dialogues have an eligible row, {n} needed")
    return [{"dialogue_id": d,
             "utterance_idx": rng.choice(sorted(r["utterance_idx"] for r in by_dialogue[d]))}
            for d in sorted(chosen)]


def main():
    ap = argparse.ArgumentParser(description="Choose the human-evaluation contexts (IDs only).")
    ap.add_argument("--processed", type=Path, default=ROOT / "data/processed")
    ap.add_argument("--output", type=Path, default=ROOT / "configs/eval_contexts.json")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-seeker-turns", type=int, default=2)
    ap.add_argument("--min-seeker-words", type=int, default=20)
    ap.add_argument("--max-words", type=int, default=250)
    args = ap.parse_args()
    rows = read_jsonl(args.processed / "downstream/responses_test.jsonl")
    contexts = select(rows, args.n, args.seed, PRIORITY, args.min_seeker_turns, args.min_seeker_words,
                      args.max_words)
    out = {
        "seed": args.seed,
        "source": "data/processed/downstream/responses_test.jsonl",
        "splits_sha256": hashlib.sha256((args.processed / "splits.json").read_bytes()).hexdigest()[:12],
        "rule": {
            "split": "test",
            "one_row_per_dialogue": True,
            "history_ends_with": "seeker",
            "min_seeker_turns": args.min_seeker_turns,
            "min_seeker_words": args.min_seeker_words,
            "max_history_words": args.max_words,
            "problems_in_turn": list(PRIORITY),
            "then": "other in-scope problems",
            "uses_reference_reply": False,
        },
        "contexts": contexts,
    }
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    picked = {(c["dialogue_id"], c["utterance_idx"]) for c in contexts}
    problems = collections.Counter(r["problem"] for r in rows if (r["dialogue_id"], r["utterance_idx"]) in picked)
    print(json.dumps({"contexts": len(contexts), "problems": dict(problems), "output": str(args.output)}))


if __name__ == "__main__":
    main()

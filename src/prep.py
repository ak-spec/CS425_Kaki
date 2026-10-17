"""ESConv preprocessing for Perceive (emotion + problem-type classifier).

Pipeline: load -> dedup -> clean + merge consecutive same-speaker turns ->
normalise labels -> split by dialogue (seeded, stratified) -> build one sample
per seeker turn (history up to and including that turn).

Usage: python src/prep.py [--seed 42] [--out data/processed]
"""

import argparse
import collections
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
HF_REPO = "thu-coai/esconv"
HF_FILE = "ESConv.json"

# --- label collapsing (raw labels are always kept alongside) ---------------
EMOTION_MAP = {"nervousness": "anxiety", "pain": "other", "jealousy": "other", "guilt": "other"}
PROBLEM_MAP = {"issues with parents": "conflict with parents"}
MIN_PROBLEM_DIALOGUES = 15  # problem classes smaller than this -> "other"
IN_SCOPE_PROBLEMS = {
    "ongoing depression",
    "job crisis",
    "breakup with partner",
    "problems with friends",
    "academic pressure",
}

SPLIT_FRACS = {"val": 0.1, "test": 0.1}  # train gets the rest


def fetch_raw(raw_dir: Path) -> list[dict]:
    """Download ESConv.json from the Hugging Face Hub (cached locally after the first call)."""
    path = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset", local_dir=raw_dir)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def norm_text(s: str) -> str:
    return " ".join(s.split())


def dedup(dialogues: list[dict]) -> tuple[list[tuple[int, dict]], int]:
    """Drop exact-duplicate dialogues (same speakers + text). Keeps the first.
    Returns (original_index, dialogue) pairs; ids stay tied to the raw file order."""
    seen, kept = set(), []
    # content == text , speaker=usr/sys, strategy=annotations.strategy
    for i, d in enumerate(dialogues):
        key = tuple((t["speaker"], norm_text(t["content"]).lower()) for t in d["dialog"])
        if key in seen:
            continue
        seen.add(key)
        kept.append((i, d))
    return kept, len(dialogues) - len(kept)


def merge_turns(dialog: list[dict]) -> tuple[list[dict], int]:
    """Clean whitespace and merge consecutive same-speaker utterances.
    Returns (turns, number_of_merges)."""
    turns, merges = [], 0
    for t in dialog:
        text = norm_text(t["content"])
        if not text:
            continue
        if turns and turns[-1]["speaker"] == t["speaker"]:
            turns[-1]["text"] += " " + text
            merges += 1
        else:
            turns.append({"speaker": t["speaker"], "text": text})
    # total of 7,973 merges done
    return turns, merges


def collapse_labels(dialogues: list[dict]) -> dict[int, dict]:
    """Per-dialogue (index into list) label info: raw + collapsed emotion/problem."""
    problems = [
        PROBLEM_MAP.get(d["problem_type"].strip().lower(), d["problem_type"].strip().lower()) for d in dialogues
    ]
    counts = collections.Counter(problems)
    out = {}
    for i, d in enumerate(dialogues):
        emo_raw = d["emotion_type"].strip().lower()
        prob_raw = d["problem_type"].strip().lower()
        prob = problems[i]
        out[i] = {
            "emotion_raw": emo_raw,
            "emotion": EMOTION_MAP.get(emo_raw, emo_raw),
            "problem_raw": prob_raw,
            "problem": prob if counts[prob] >= MIN_PROBLEM_DIALOGUES else "other",
            "in_scope": prob_raw in IN_SCOPE_PROBLEMS,
        }
    return out


def split_dialogues(ids: list[int], strata: dict[int, str], seed: int) -> dict[str, list[int]]:
    """80/10/10 by dialogue, stratified on collapsed problem type, fixed seed."""
    rng = random.Random(seed)
    by_stratum = collections.defaultdict(list)
    for i in ids:
        by_stratum[strata[i]].append(i)
    split = {"train": [], "val": [], "test": []}
    for key in sorted(by_stratum):
        members = sorted(by_stratum[key])
        rng.shuffle(members)
        n_val = round(len(members) * SPLIT_FRACS["val"])
        n_test = round(len(members) * SPLIT_FRACS["test"])
        split["val"] += members[:n_val]
        split["test"] += members[n_val : n_val + n_test]
        split["train"] += members[n_val + n_test :]
    return {k: sorted(v) for k, v in split.items()}


def render_history(turns: list[dict]) -> str:
    """Plain-text history. No strategy tags: the bot's own past replies are untagged at inference."""
    return " | ".join(f"{t['speaker']}: {t['text']}" for t in turns)


def build_samples(dialogue_id: int, turns: list[dict], labels: dict) -> list[dict]:
    """One sample per seeker turn: history up to and including that turn."""
    samples, seeker_idx = [], 0
    for end, t in enumerate(turns):
        if t["speaker"] != "seeker":
            continue
        seeker_idx += 1
        samples.append(
            {
                "dialogue_id": dialogue_id,
                "turn_idx": seeker_idx,  # 1-based seeker-turn position, for accuracy-by-position
                "text": render_history(turns[: end + 1]),
                **labels,
            }
        )
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed")
    args = ap.parse_args()

    # download ESConv.json into /data/raw
    raw = fetch_raw(args.raw)

    kept, n_dups = dedup(raw)
    ids = [i for i, _ in kept]
    dialogues = [d for _, d in kept]

    labels_by_pos = collapse_labels(dialogues)
    # re-key the label dict from list position to dialoude ID
    labels = {ids[pos]: lab for pos, lab in labels_by_pos.items()}

    # split BEFORE building samples so no prefixes of one dialogue cross partitions
    split = split_dialogues(ids, {i: labels[i]["problem"] for i in ids}, args.seed)

    turns_by_id, total_merges = {}, 0
    for i, d in kept:
        turns, merges = merge_turns(d["dialog"])
        turns_by_id[i] = turns
        total_merges += merges

    emotions = sorted({l["emotion"] for l in labels.values()} - {"other"}) + ["other"]
    problems = sorted({l["problem"] for l in labels.values()} - {"other"}) + ["other"]
    label_maps = {
        "emotion2id": {e: k for k, e in enumerate(emotions)},
        "problem2id": {p: k for k, p in enumerate(problems)},
    }

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "seed": args.seed,
        "raw_dialogues": len(raw),
        "duplicates_removed": n_dups,
        "consecutive_utterances_merged": total_merges,
        "splits": {},
    }
    for name, split_ids in split.items():
        samples = []
        for i in split_ids:
            rows = build_samples(i, turns_by_id[i], labels[i])
            samples.extend(rows)
        with open(args.out / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        report["splits"][name] = {
            "dialogues": len(split_ids),
            "samples": len(samples),
            "in_scope_dialogues": sum(labels[i]["in_scope"] for i in split_ids),
            "emotion": dict(collections.Counter(labels[i]["emotion"] for i in split_ids)),
            "problem": dict(collections.Counter(labels[i]["problem"] for i in split_ids)),
        }

    # UTF-8 with LF line endings on every OS, so splits.json has the same fingerprint everywhere
    for name, obj in (
        ("splits.json", {"seed": args.seed, **split}),
        ("label_maps.json", label_maps),
        ("prep_report.json", report),
    ):
        (args.out / name).write_text(json.dumps(obj, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

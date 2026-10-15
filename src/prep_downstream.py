"""Build Plan/Speak examples and provisional Critic pairs without altering Perceive."""

import argparse
import collections
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from prep import HF_FILE, HF_REPO, collapse_labels, dedup, norm_text, split_dialogues

STRATEGIES = sorted(
    [
        "Question",
        "Restatement or Paraphrasing",
        "Reflection of feelings",
        "Self-disclosure",
        "Affirmation and Reassurance",
        "Providing Suggestions",
        "Information",
        "Others",
    ]
)


def history(turns):
    return " | ".join(
        f"{t['speaker']}: {norm_text(t['content'])}"
        for t in turns
        if norm_text(t["content"])
    )


def build_rows(dialogue_id, dialogue, labels):
    """Unmerged supporter utterances retain their own label and exact prefix."""
    turns = dialogue["dialog"]
    examples, feedback, excluded = [], [], collections.Counter()
    for i, turn in enumerate(turns):
        annotation = turn.get("annotation", {})
        if turn["speaker"] == "supporter":
            strategy = annotation.get("strategy")
            if strategy not in STRATEGIES:
                excluded["unsupported_strategy"] += 1
                continue
            if not norm_text(turn["content"]):
                excluded["empty_response"] += 1
                continue
            if not history(turns[:i]):
                excluded["empty_history"] += 1
                continue
            examples.append(
                dict(
                    dialogue_id=dialogue_id,
                    utterance_idx=i,
                    text=history(turns[:i]),
                    response=norm_text(turn["content"]),
                    strategy=strategy,
                    label=STRATEGIES.index(strategy),
                    **labels,
                )
            )
        if "feedback" not in annotation:
            continue
        excluded["raw_feedback_count"] += 1
        try:
            score = int(annotation["feedback"])
        except (TypeError, ValueError):
            excluded["invalid_feedback"] += 1
            continue
        if (
            turn["speaker"] != "seeker"
            or score not in range(1, 6)
            or i == 0
            or turns[i - 1]["speaker"] != "supporter"
        ):
            excluded["unpaired_feedback"] += 1
            continue
        start = i - 1
        while start > 0 and turns[start - 1]["speaker"] == "supporter":
            start -= 1
        # Proxy target: complete immediately preceding supporter block, not each utterance.
        feedback.append(
            dict(
                dialogue_id=dialogue_id,
                feedback_utterance_idx=i,
                response_start_idx=start,
                response_end_idx=i - 1,
                text=history(turns[:start]),
                response=" ".join(norm_text(t["content"]) for t in turns[start:i]),
                feedback=score,
                alignment="preceding_supporter_block_proxy",
                strategies=[
                    t.get("annotation", {}).get("strategy") for t in turns[start:i]
                ],
            )
        )
    return examples, feedback, excluded


def validate_splits(splits, ids):
    groups = [splits[k] for k in ("train", "val", "test")]
    flat = [i for group in groups for i in group]
    if len(flat) != len(set(flat)) or set(flat) != set(ids):
        raise ValueError(
            "Split IDs must cover deduplicated raw dialogues exactly once."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    raw_path = hf_hub_download(
        HF_REPO, HF_FILE, repo_type="dataset", local_dir=args.raw
    )
    raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    kept, duplicates = dedup(raw)
    ids = [i for i, _ in kept]
    labels = collapse_labels([d for _, d in kept])
    labels = {ids[pos]: value for pos, value in labels.items()}
    split_path = args.processed / "splits.json"
    if split_path.exists():
        splits = json.loads(split_path.read_text(encoding="utf-8"))
        split_source = "existing splits.json"
    else:
        splits = {
            "seed": 42,
            **split_dialogues(ids, {i: labels[i]["problem"] for i in ids}, 42),
        }
        split_source = "reconstructed using unchanged prep.py seed-42 algorithm"
    validate_splits(splits, ids)
    out = args.processed / "downstream"
    out.mkdir(parents=True, exist_ok=True)
    (out / "splits.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")
    (out / "strategy_labels.json").write_text(
        json.dumps(STRATEGIES, indent=2), encoding="utf-8"
    )
    lookup = dict(kept)
    report = {
        "split_source": split_source,
        "raw_dialogues": len(raw),
        "duplicates": duplicates,
        "raw_sha256": hashlib.sha256(
            json.dumps(raw, sort_keys=True).encode()
        ).hexdigest(),
        "splits": {},
    }
    for name in ("train", "val", "test"):
        examples, pairs, counts = [], [], collections.Counter()
        for i in splits[name]:
            rows, ratings, stats = build_rows(i, lookup[i], labels[i])
            examples.extend(rows)
            pairs.extend(ratings)
            counts.update(stats)
        for prefix, rows in [("responses", examples), ("critic_proxy", pairs)]:
            with (out / f"{prefix}_{name}.jsonl").open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["splits"][name] = dict(
            dialogues=len(splits[name]),
            responses=len(examples),
            critic_proxy_pairs=len(pairs),
            annotation_counts=dict(counts),
            strategy_counts=dict(collections.Counter(r["strategy"] for r in examples)),
            feedback_counts=dict(collections.Counter(r["feedback"] for r in pairs)),
        )
    (out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

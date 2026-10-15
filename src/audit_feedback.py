"""Count feedback intervals without assigning multi-utterance ratings to one reply."""

import argparse
from collections import Counter
import json
from pathlib import Path


def audit(dialogues, ids):
    intervals, immediate_blocks = Counter(), Counter()
    for dialogue_id in ids:
        pending, block = 0, 0
        for turn in dialogues[dialogue_id]["dialog"]:
            if turn["speaker"] == "supporter":
                pending += 1
                block += 1
            else:
                if "feedback" in turn.get("annotation", {}):
                    intervals[pending] += 1
                    immediate_blocks[block] += 1
                    pending = 0
                block = 0
    return {
        "supporter_utterances_since_previous_rating": dict(sorted(intervals.items())),
        "immediately_preceding_supporter_block_length": dict(
            sorted(immediate_blocks.items())
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw/ESConv.json"))
    parser.add_argument(
        "--splits", type=Path, default=Path("data/processed/downstream/splits.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/downstream/feedback_audit.json"),
    )
    args = parser.parse_args()
    dialogues = json.loads(args.raw.read_text(encoding="utf-8"))
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    report = {name: audit(dialogues, splits[name]) for name in ("train", "val", "test")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

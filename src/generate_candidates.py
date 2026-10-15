"""Offline Plan-to-Speak check; outputs candidates without selecting a final reply."""

import argparse
import json

from plan import Planner
from speak import Speaker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", required=True)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--emotion", required=True)
    args = parser.parse_args()
    planner = Planner(args.planner)
    strategies = planner.predict(args.history)
    # Release the encoder before loading the generator on memory-limited machines.
    del planner
    speaker = Speaker(args.speaker)
    candidates = speaker.generate_candidates(args.history, args.emotion, strategies)
    print(json.dumps(candidates, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

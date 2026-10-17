"""Helpers shared by the scripts: reading JSONL files, keeping outputs out of git, and the history format.

Every model reads a conversation as one string, "speaker: text | speaker: text | ...", with the speakers
"seeker" and "supporter". render() builds that string from turns and history_turns() splits it back.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TURN_BREAK = re.compile(r" \| (?=(?:seeker|supporter): )")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def private_dir(path: Path) -> Path:
    """Create a directory whose contents git ignores: replies and rating sheets quote ESConv text."""
    path.mkdir(parents=True, exist_ok=True)
    (path / ".gitignore").write_text("*\n", encoding="utf-8")
    return path


def render(turns: list[dict]) -> str:
    """{"speaker", "text"} turns as one history string, whitespace collapsed and empty turns dropped."""
    return " | ".join(f"{t['speaker']}: {' '.join(t['text'].split())}" for t in turns if t["text"].strip())


def history_turns(text: str) -> list[dict]:
    """A history string back into {"speaker", "text"} turns."""
    turns = []
    for part in TURN_BREAK.split(text.strip()):
        speaker, sep, said = part.partition(": ")
        if speaker not in ("seeker", "supporter") or not sep or not said.strip():
            raise ValueError(f"not a tagged history turn: {part[:40]!r}")
        turns.append({"speaker": speaker, "text": said})
    return turns

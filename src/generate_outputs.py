"""Run Kaki on ESConv test histories and save each configuration's replies for scoring.

Every row of data/processed/downstream/responses_test.jsonl (or only the human-evaluation contexts with
--contexts configs/eval_contexts.json) is answered once per configuration with Kaki.respond, and one line
per row goes to data/outputs/{config}.jsonl: IDs, the reference supporter reply and strategy, the reply,
the selected strategy and every candidate. Turns come from the raw ESConv dialogue (data/raw/ESConv.json)
and must render to the row's history exactly; without the raw file the history string is split back into
turns. Rows already in an output file are skipped, so an interrupted run resumes; --overwrite starts again.

    python src/generate_outputs.py                                         # all test rows, three configs
    python src/generate_outputs.py --contexts configs/eval_contexts.json   # the 20 human-evaluation contexts
    python src/generate_outputs.py --configs direct --limit 5              # quick check

Needs the trained checkpoints (guard, perceive, plan, speak-history, speak-conditioned) and, for the full
test split, a GPU. --critic adds a Critic checkpoint for look-ahead selection.
"""

import argparse
import json
import time
from pathlib import Path

from common import history_turns, private_dir, read_jsonl, render
from pipeline import CONFIGS, Kaki

ROOT = Path(__file__).resolve().parent.parent


def load_rows(path: Path, contexts: Path | None = None, limit: int | None = None) -> list[dict]:
    rows = read_jsonl(path)
    if contexts is not None:
        by_key = {(r["dialogue_id"], r["utterance_idx"]): r for r in rows}
        wanted = [
            (c["dialogue_id"], c["utterance_idx"])
            for c in json.loads(Path(contexts).read_text(encoding="utf-8"))["contexts"]
        ]
        missing = [k for k in wanted if k not in by_key]
        if missing:
            raise ValueError(f"contexts not found in {path}: {missing}")
        rows = [by_key[k] for k in wanted]
    return rows[:limit] if limit else rows


def row_turns(row: dict, dialogues: list[dict] | None = None) -> list[dict]:
    """The {"speaker", "text"} turns before the reference reply, checked against the row's history."""
    if dialogues is None:
        turns = history_turns(row["text"])
    else:
        dialog = dialogues[row["dialogue_id"]]["dialog"][: row["utterance_idx"]]
        turns = [
            {"speaker": t["speaker"], "text": " ".join(t["content"].split())} for t in dialog if t["content"].strip()
        ]
    if render(turns) != row["text"]:
        raise ValueError(
            f"turns do not match the history of dialogue {row['dialogue_id']}, utterance {row['utterance_idx']}"
        )
    return turns


def record(row: dict, turns: list[dict], out: dict) -> dict:
    """One output line: the reference, the reply and the decisions behind it."""
    chosen = out["candidates"][out["selected"]] if out["candidates"] else None
    perceived = out["perceive"]
    return {
        "dialogue_id": row["dialogue_id"],
        "utterance_idx": row["utterance_idx"],
        "config": out["config"],
        "problem": row["problem"],
        "emotion": row["emotion"],
        "after_seeker": turns[-1]["speaker"] == "seeker",
        "reference": row["response"],
        "reference_strategy": row["strategy"],
        "reply": out["reply"],
        "bypassed": out["bypassed"],
        "strategy": chosen["strategy"] if chosen else None,
        "perceived": {"emotion": perceived["emotion"], "problem": perceived["problem"]} if perceived else None,
        "planner": out["strategies"],
        "candidates": out["candidates"],
        "selected": out["selected"],
    }


def has_seeker(turns: list[dict]) -> bool:
    return any(t["speaker"] == "seeker" for t in turns)


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    return {(r["dialogue_id"], r["utterance_idx"]) for r in read_jsonl(path)}


def generate(
    kaki, rows: list[dict], configs, output: Path, dialogues=None, overwrite=False, log_every: int = 50
) -> dict:
    """Append one line per (row, config) to output/{config}.jsonl; returns the number written per config."""
    private_dir(output)
    paths = {c: output / f"{c}.jsonl" for c in configs}
    if overwrite:
        for p in paths.values():
            p.unlink(missing_ok=True)
    done = {c: done_keys(p) for c, p in paths.items()}
    files = {c: open(p, "a", encoding="utf-8") for c, p in paths.items()}
    written, t0 = dict.fromkeys(configs, 0), time.time()
    try:
        for i, row in enumerate(rows, 1):
            key = (row["dialogue_id"], row["utterance_idx"])
            todo = [c for c in configs if key not in done[c]]
            if todo:
                turns = row_turns(row, dialogues)
                if not has_seeker(turns):  # the bot only answers a seeker, so there is nothing to reply to
                    todo = []
            for c in todo:
                files[c].write(json.dumps(record(row, turns, kaki.respond(turns, c)), ensure_ascii=False) + "\n")
                files[c].flush()
                written[c] += 1
            if i % log_every == 0 or i == len(rows):
                print(
                    json.dumps(
                        {"rows": i, "of": len(rows), "written": written, "minutes": round((time.time() - t0) / 60, 1)}
                    ),
                    flush=True,
                )
    finally:
        for f in files.values():
            f.close()
    return written


def main():
    ap = argparse.ArgumentParser(description="Generate replies for each configuration on ESConv test rows.")
    ap.add_argument("--data", type=Path, default=ROOT / "data/processed/downstream/responses_test.jsonl")
    ap.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "data/raw/ESConv.json",
        help="raw ESConv for exact turns; the history string is split if missing",
    )
    ap.add_argument("--contexts", type=Path, help="only these rows, e.g. configs/eval_contexts.json")
    ap.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    ap.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")
    ap.add_argument(
        "--critic",
        type=Path,
        help="Critic checkpoint for look-ahead; without it look-ahead ranks candidates by planner probability",
    )
    ap.add_argument("--critic-weight", type=float, default=1.0)
    ap.add_argument("--output", type=Path, default=ROOT / "data/outputs")
    ap.add_argument("--limit", type=int, help="first N rows only, for quick checks")
    ap.add_argument("--overwrite", action="store_true", help="discard existing outputs instead of resuming")
    args = ap.parse_args()
    rows = load_rows(args.data, args.contexts, args.limit)
    dialogues = json.loads(args.raw.read_text(encoding="utf-8")) if args.raw.exists() else None
    critic = None
    if args.critic:
        from critic import Critic

        critic = Critic(args.critic)
    kaki = Kaki.load(args.checkpoints, configs=args.configs, critic=critic, critic_weight=args.critic_weight)
    written = generate(kaki, rows, args.configs, args.output, dialogues, args.overwrite)
    run = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "rows": len(rows),
        "written": written,
        "turns_from": "raw" if dialogues else "history string",
        "skipped_no_seeker_turn": sum(not has_seeker(row_turns(r, dialogues)) for r in rows),
    }
    (args.output / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

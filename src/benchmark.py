"""Size, speed and memory of each system, and its replies to the handwritten queries.

The proposal's efficiency check: parameters, latency per turn and peak memory. Each system runs in its
own process so that peak memory is its own. A turn is one call to Kaki.respond (Guard, Perceive, Plan,
Speak and, for look-ahead, the Critic) on a query from configs/queries.json, after one warm-up turn; the
baselines time generation only. Peak memory is the process's peak resident memory on CPU and the peak
allocated GPU memory on a GPU. The replies are saved too, for the qualitative comparison
(data/outputs/queries.json; they quote no dataset text).

    python src/benchmark.py --critic checkpoints/critic-contrast --critic-weight 0
    python src/benchmark.py --device cpu --threads 4 --systems direct planning
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

import torch

from common import private_dir

ROOT = Path(__file__).resolve().parent.parent
SYSTEMS = ("direct", "planning", "lookahead", "zeroshot", "blenderbot")


def peak_memory_mb(device: str) -> float:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    import psutil

    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):  # Windows
        return info.peak_wset / 2**20
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**10  # Linux reports KiB


def parameters(*models) -> int:
    seen, total = set(), 0
    for m in models:
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
    return total


def run_one(args) -> dict:
    """Load one system, answer every query, report timings; runs in a child process."""
    if args.threads:
        torch.set_num_threads(args.threads)
    queries = json.loads(args.queries.read_text(encoding="utf-8"))["queries"]
    t0 = time.perf_counter()
    if args.one in ("zeroshot", "blenderbot"):
        if args.one == "zeroshot":
            from speak import Speaker

            system = Speaker(None, args.device)
        else:
            from baselines import BlenderBot

            system = BlenderBot(args.checkpoints / "blenderbot", args.device)
        models = [system.model]
        answer = lambda text: {"reply": system.generate_batch([f"seeker: {text}"])[0]}
    else:
        from pipeline import Kaki

        critic = None
        if args.one == "lookahead" and args.critic:
            from critic import Critic

            critic = Critic(args.critic, args.device)
        kaki = Kaki.load(
            args.checkpoints, configs=[args.one], critic=critic, critic_weight=args.critic_weight, device=args.device
        )
        models = [kaki.guard.model, kaki.perceiver.model, *(s.model for s in kaki.speakers.values())]
        models += [kaki.planner.model] if kaki.planner else []
        models += [kaki.critic.model] if kaki.critic is not None else []

        def answer(text):
            out = kaki.respond([{"speaker": "seeker", "text": text}], args.one)
            chosen = out["candidates"][out["selected"]] if out["candidates"] else None
            return {
                "reply": out["reply"],
                "bypassed": out["bypassed"],
                "emotion": out["perceive"]["emotion"] if out["perceive"] else None,
                "strategy": chosen["strategy"] if chosen else None,
                "candidates": [{k: c[k] for k in ("strategy", "response", "critic_score")} for c in out["candidates"]]
                if args.one == "lookahead"
                else None,
            }

    load_seconds = time.perf_counter() - t0
    answer(queries[0]["text"])  # warm-up
    times, replies = [], []
    for q in queries:
        start = time.perf_counter()
        replies.append({"id": q["id"], **answer(q["text"])})
        times.append(time.perf_counter() - start)
    return {
        "system": args.one,
        "device": args.device,
        "threads": torch.get_num_threads(),
        "parameters_millions": round(parameters(*models) / 1e6, 1),
        "load_seconds": round(load_seconds, 1),
        "median_seconds_per_turn": round(median(times), 2),
        "max_seconds_per_turn": round(max(times), 2),
        "peak_memory_mb": round(peak_memory_mb(args.device)),
        "replies": replies,
    }


def main():
    ap = argparse.ArgumentParser(description="Parameters, latency per turn and peak memory of each system.")
    ap.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, help="CPU threads, e.g. 4 for a laptop")
    ap.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--critic", type=Path, help="Critic for look-ahead, as in src/generate_outputs.py")
    ap.add_argument("--critic-weight", type=float, default=1.0)
    ap.add_argument("--queries", type=Path, default=ROOT / "configs/queries.json")
    ap.add_argument("--output", type=Path, default=ROOT / "data/outputs")
    ap.add_argument("--one", choices=SYSTEMS, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one:
        print(json.dumps(run_one(args)))
        return
    results = []
    for system in args.systems:
        cmd = [
            sys.executable,
            __file__,
            "--one",
            system,
            "--device",
            args.device,
            "--checkpoints",
            str(args.checkpoints),
            "--queries",
            str(args.queries),
            "--critic-weight",
            str(args.critic_weight),
        ]
        cmd += ["--threads", str(args.threads)] if args.threads else []
        cmd += ["--critic", str(args.critic)] if args.critic else []
        done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if done.returncode:
            print(done.stderr[-2000:], file=sys.stderr)
            raise SystemExit(f"{system} failed")
        results.append(json.loads(done.stdout.strip().splitlines()[-1]))
        print(json.dumps({k: v for k, v in results[-1].items() if k != "replies"}), flush=True)
    name = f"benchmark_{args.device}{'_' + str(args.threads) + 't' if args.threads else ''}.json"
    (private_dir(args.output) / name).write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

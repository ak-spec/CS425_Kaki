"""Kaki for one seeker turn: Guard -> Perceive -> Plan -> Speak -> Critique.

The three configurations compared in the proposal share Guard and Perceive:
    direct     the history-only Speak adapter writes one reply
    planning   the conditioned adapter writes one reply for the planner's top strategy
    lookahead  the conditioned adapter writes one reply for each of the top three strategies, and the
               reply with the highest critic score + critic_weight * planner probability is chosen

`critic` is any callable (history, replies) -> scores. Until the Critic is trained, lookahead ranks the
candidates by planner probability alone and reports critic scores as None.

Usage, from the repository root with src on the import path:
    from pipeline import Kaki
    kaki = Kaki.load(configs=["lookahead"])
    out = kaki.respond([{"speaker": "seeker", "text": "I failed my midterm and feel awful."}])
    out["reply"]

Chat in a terminal:  python src/pipeline.py --config lookahead
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ("direct", "planning", "lookahead")


def render(turns: list[dict]) -> str:
    """History in the format Perceive, Plan and Speak were trained on: 'speaker: text | ...'."""
    return " | ".join(f"{t['speaker']}: {' '.join(t['text'].split())}" for t in turns if t["text"].strip())


class Kaki:
    def __init__(self, guard, perceiver, planner=None, speakers=None, critic=None, critic_weight=1.0):
        self.guard = guard
        self.perceiver = perceiver
        self.planner = planner
        self.speakers = speakers or {}  # {"history": Speaker, "conditioned": Speaker}
        self.critic = critic
        self.critic_weight = critic_weight

    @classmethod
    def load(cls, checkpoints=ROOT / "checkpoints", configs=CONFIGS, critic=None, critic_weight=1.0,
             device=None):
        """Load the trained components that `configs` need from checkpoints/{guard,perceive,plan,speak-*}.
        Each Speak adapter loads its own copy of the base model, so load only the configurations you use."""
        from guard import Guard
        from perceive import Perceiver
        from plan import Planner
        from speak import Speaker

        checkpoints = Path(checkpoints)
        needs_plan = any(c in configs for c in ("planning", "lookahead"))
        speakers = {}
        if "direct" in configs:
            speakers["history"] = Speaker(checkpoints / "speak-history", device)
        if needs_plan:
            speakers["conditioned"] = Speaker(checkpoints / "speak-conditioned", device)
        return cls(Guard(checkpoints / "guard", device=device), Perceiver(checkpoints / "perceive", device=device),
                   Planner(checkpoints / "plan", device) if needs_plan else None, speakers, critic, critic_weight)

    def respond(self, turns: list[dict], config: str = "lookahead") -> dict:
        """Reply to the latest seeker turn. Every intermediate decision is returned for the interface."""
        if config not in CONFIGS:
            raise ValueError(f"config must be one of {CONFIGS}")
        out = {"config": config, "reply": None, "bypassed": False, "guard": self.guard.check(turns),
               "perceive": None, "strategies": None, "candidates": [], "selected": None}
        if out["guard"]["bypass_generation"]:
            out.update(reply=self.guard.response(out["guard"]["response_key"]), bypassed=True)
            return out

        history = render(turns)
        out["perceive"] = self.perceiver.predict(history)
        emotion = out["perceive"]["emotion"]
        if config == "direct":
            candidates = [{"strategy": None, "planner_probability": None,
                           "response": self.speakers["history"].generate(history)}]
        else:
            out["strategies"] = self.planner.predict(history)
            speaker = self.speakers["conditioned"]
            if config == "planning":
                top = out["strategies"][0]
                candidates = [{"strategy": top["strategy"], "planner_probability": top["probability"],
                               "response": speaker.generate(history, emotion, top["strategy"])}]
            else:
                candidates = speaker.generate_candidates(history, emotion, out["strategies"])

        critic_scores = (self.critic(history, [c["response"] for c in candidates])
                         if self.critic is not None and config == "lookahead" else [None] * len(candidates))
        for c, s in zip(candidates, critic_scores):
            c["critic_score"] = s
            if c["planner_probability"] is None:  # direct: a single reply, nothing to rank
                c["score"] = None
            elif s is None:  # no Critic yet: rank by planner probability
                c["score"] = c["planner_probability"]
            else:
                c["score"] = s + self.critic_weight * c["planner_probability"]
        best = max(range(len(candidates)), key=lambda i: candidates[i]["score"] or 0.0)
        out.update(candidates=candidates, selected=best, reply=candidates[best]["response"])
        return out


def main():
    ap = argparse.ArgumentParser(description="Chat with Kaki in the terminal (an empty line quits).")
    ap.add_argument("--config", choices=CONFIGS, default="lookahead")
    ap.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--show", action="store_true", help="print Guard, Perceive, Plan and candidate details")
    args = ap.parse_args()
    kaki = Kaki.load(args.checkpoints, configs=[args.config])
    turns = []  # kept in memory only
    while True:
        text = input("you: ").strip()
        if not text:
            break
        turns.append({"speaker": "seeker", "text": text})
        out = kaki.respond(turns, args.config)
        if args.show:
            print(json.dumps({k: v for k, v in out.items() if k != "reply"}, indent=1, default=str))
        print(f"kaki: {out['reply']}\n")
        turns.append({"speaker": "supporter", "text": out["reply"]})


if __name__ == "__main__":
    main()

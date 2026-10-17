"""Simulated conversations: a prompted seeker model chats with each configuration.

The proposal's interactive evaluation. The seeker is Qwen2.5-7B-Instruct, prompted with the situation,
emotion and problem of an ESConv validation or test dialogue and nothing else from it. It opens the chat,
answers Kaki for at most --turns messages, and ends with [END] when it wants to stop. The proposal planned
a LoRA seeker trained on mirrored ESConv seeker turns; a prompted off-the-shelf model replaced it to stay
within the time box, which also keeps the simulator independent of Kaki's training data.

Every configuration meets the same seekers: each situation's opening message is generated once and shared,
and seeker sampling uses a fixed seed for each turn. Kaki's replies differ, so later seeker messages
differ too. A Guard bypass ends the chat, since the prepared response hands over to people.

Per configuration, in data/simulation/summary.json:
    eval_critic     mean rating the held-out Eval-Critic predicts for Kaki's replies (1-5)
    seeker_rating   the simulated seeker's 1-5 answer, after the chat, to how helpful the supporter was
    ended           share of chats the seeker closed with [END] before the turn limit
    bypassed        share of chats Guard ended
    repetition      share of Kaki replies that take at least half of their word trigrams from an earlier
                    reply in the same chat
    distinct, length, strategies used per chat; paired intervals between configurations over situations

    python src/simulate.py --critic checkpoints/critic-contrast --critic-weight 0
    python src/simulate.py --sessions 4 --turns 2 --output data/simulation-check     # quick check
"""

import argparse
import json
import random
import re
import time
from pathlib import Path
from statistics import mean

import torch

from compare_configs import bootstrap_diff
from generate_outputs import private_dir
from metrics import distinct, words
from pipeline import CONFIGS, Kaki, render

ROOT = Path(__file__).resolve().parent.parent
SEEKER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEEKER_PROMPT = (
    "You are role-playing a person who has come to an online peer-support chat. Stay in character as this "
    "person for the whole chat.\n\n"
    "Your situation: {situation}\nYou feel: {emotion}. Your problem: {problem}.\n\n"
    "Write only your own next chat message, in the first person, one to three short sentences, in a casual "
    "texting style without emojis. Share details gradually and react to what the supporter just said. Do "
    "not solve the problem by yourself and do not act like a counsellor. Later in the chat, when you feel "
    "the supporter has helped enough and you want to stop, say goodbye and end that message with [END]."
)
OPENING = "(You have just joined the chat. Write your first message to the supporter.)"
CLOSING = (
    "(The chat is over. Step out of the role for a moment: on a scale of 1 to 5, how helpful was the "
    "supporter in this chat, where 1 is not at all and 5 is very helpful? Answer with one digit.)"
)
END = "[END]"


def situations(raw: list[dict], splits: dict, n: int, seed: int) -> list[dict]:
    """n ESConv validation and test dialogues: their id, situation, emotion and problem only."""
    ids = sorted(splits["val"] + splits["test"])
    chosen = sorted(random.Random(seed).sample(ids, min(n, len(ids))))
    return [
        {
            "dialogue_id": i,
            "situation": " ".join(raw[i]["situation"].split()),
            "emotion": raw[i]["emotion_type"].strip().lower(),
            "problem": raw[i]["problem_type"].strip().lower(),
        }
        for i in chosen
    ]


def seeker_messages(session: dict, turns: list[dict], closing: bool = False) -> list[dict]:
    """The chat from the seeker model's side: its own turns are the assistant's, Kaki's are the user's."""
    msgs = [{"role": "system", "content": SEEKER_PROMPT.format(**session)}, {"role": "user", "content": OPENING}]
    msgs += [{"role": "assistant" if t["speaker"] == "seeker" else "user", "content": t["text"]} for t in turns]
    if closing:
        msgs.append({"role": "user", "content": CLOSING})
    return msgs


def clean(message: str) -> tuple[str, bool]:
    """(seeker text without role prefixes, quotes or [END], whether the seeker ended the chat)."""
    message = re.split(r"\n\s*(?:supporter|kaki)\s*:", message, flags=re.I)[0]
    ended = END.lower() in message.lower()
    message = re.sub(re.escape(END), "", message, flags=re.I)
    message = re.sub(r"^\s*(?:seeker|me|you)\s*:\s*", "", message.strip(), flags=re.I)
    return " ".join(message.strip().strip('"').split()), ended


def rating(answer: str):
    m = re.search(r"[1-5]", answer)
    return int(m.group()) if m else None


def trigrams(text: str) -> set:
    w = words(text)
    return {tuple(w[i : i + 3]) for i in range(len(w) - 2)}


def repeats(replies: list[str], share: float = 0.5) -> list[bool]:
    """Whether each reply takes at least `share` of its trigrams from earlier replies in the chat."""
    seen, out = set(), []
    for r in replies:
        grams = trigrams(r)
        out.append(bool(grams) and len(grams & seen) >= share * len(grams))
        seen |= grams
    return out


class Seeker:
    def __init__(self, name=SEEKER_MODEL, device=None, batch_size=8, temperature=0.7, top_p=0.9, max_new_tokens=80):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(self.device).eval()
        self.batch_size, self.temperature, self.top_p, self.max_new_tokens = (
            batch_size,
            temperature,
            top_p,
            max_new_tokens,
        )

    @torch.inference_mode()
    def __call__(self, chats: list[list[dict]], seed: int, sample: bool = True, max_new_tokens=None) -> list[str]:
        out = []
        for start in range(0, len(chats), self.batch_size):
            texts = [
                self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                for c in chats[start : start + self.batch_size]
            ]
            enc = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            torch.manual_seed(seed + start)
            settings = (
                {"do_sample": True, "temperature": self.temperature, "top_p": self.top_p}
                if sample
                else {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
            )
            gen = self.model.generate(
                **enc,
                **settings,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            out += [self.tokenizer.decode(g[enc.input_ids.shape[1] :], skip_special_tokens=True).strip() for g in gen]
        return out


def converse(kaki, seeker, sessions: list[dict], openers: list[str], config: str, max_turns: int, seed: int):
    """Chats between Kaki in one configuration and the seeker, in turn-synchronous batches."""
    chats = [{**s, "config": config, "turns": [], "replies": [], "ended_by": None} for s in sessions]
    for turn in range(1, max_turns + 1):
        active = [c for c in chats if c["ended_by"] is None]
        if not active:
            break
        if turn == 1:
            messages = openers
        else:
            messages = seeker([seeker_messages(c, c["turns"]) for c in active], seed=seed * 1000 + turn)
        for c, message in zip(active, messages):
            text, ended = clean(message)
            ended = ended and turn > 1  # a chat cannot end before Kaki has replied
            if text:
                c["turns"].append({"speaker": "seeker", "text": text})
            if ended or not text:
                c["ended_by"] = "seeker"
                continue
            out = kaki.respond(c["turns"], config)
            chosen = out["candidates"][out["selected"]] if out["candidates"] else None
            c["replies"].append(
                {
                    "history": render(c["turns"]),
                    "reply": out["reply"],
                    "bypassed": out["bypassed"],
                    "strategy": chosen["strategy"] if chosen else None,
                    "guard_score": out["guard"].get("model_score"),
                }
            )
            c["turns"].append({"speaker": "supporter", "text": out["reply"]})
            if out["bypassed"]:
                c["ended_by"] = "guard"
    for c in chats:
        c["ended_by"] = c["ended_by"] or "turn_limit"
    answers = seeker(
        [seeker_messages(c, c["turns"], closing=True) for c in chats], seed=0, sample=False, max_new_tokens=4
    )
    for c, a in zip(chats, answers):
        c["seeker_rating"] = rating(a)
    return chats


def summarise(chats: list[dict]) -> dict:
    generated = [r for c in chats for r in c["replies"] if not r["bypassed"]]
    texts = [r["reply"] for r in generated]
    rep = [x for c in chats for x in repeats([r["reply"] for r in c["replies"] if not r["bypassed"]])]
    ratings = [c["seeker_rating"] for c in chats if c["seeker_rating"] is not None]
    seeker_words = [len(words(t["text"])) for c in chats for t in c["turns"] if t["speaker"] == "seeker"]
    with_strategy = [c for c in chats if any(r["strategy"] for r in c["replies"])]
    return {
        "chats": len(chats),
        "kaki_replies": len(generated),
        "seeker_turns_per_chat": mean(sum(t["speaker"] == "seeker" for t in c["turns"]) for c in chats),
        "ended_by_seeker": mean(c["ended_by"] == "seeker" for c in chats),
        "ended_by_guard": mean(c["ended_by"] == "guard" for c in chats),
        "eval_critic": mean(r["eval_critic"] for r in generated)
        if generated and "eval_critic" in generated[0]
        else None,
        "seeker_rating": mean(ratings) if ratings else None,
        "seeker_rating_missing": len(chats) - len(ratings),
        "repetition": mean(rep) if rep else None,
        "distinct1": 100 * distinct(texts, 1),
        "distinct2": 100 * distinct(texts, 2),
        "reply_length": mean(len(words(t)) for t in texts) if texts else None,
        "seeker_message_length": mean(seeker_words) if seeker_words else None,
        "strategies_per_chat": (
            mean(len({r["strategy"] for r in c["replies"] if r["strategy"]}) for c in with_strategy)
            if with_strategy
            else None
        ),
    }


def per_chat(chats: list[dict], measure: str) -> dict:
    """{dialogue_id: value} for paired comparisons; chats without the value are left out."""
    if measure == "eval_critic":
        return {
            c["dialogue_id"]: mean(r["eval_critic"] for r in c["replies"] if not r["bypassed"])
            for c in chats
            if any(not r["bypassed"] for r in c["replies"])
        }
    return {c["dialogue_id"]: c[measure] for c in chats if c[measure] is not None}


def main():
    ap = argparse.ArgumentParser(description="Simulated chats between a seeker model and each configuration.")
    ap.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    ap.add_argument("--sessions", type=int, default=200)
    ap.add_argument("--turns", type=int, default=6, help="most seeker messages per chat")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeker", default=SEEKER_MODEL)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--critic", type=Path, help="Critic for look-ahead selection, as in src/generate_outputs.py")
    ap.add_argument("--critic-weight", type=float, default=1.0)
    ap.add_argument("--eval-critic", type=Path, default=ROOT / "checkpoints/eval-critic-contrast")
    ap.add_argument("--raw", type=Path, default=ROOT / "data/raw/ESConv.json")
    ap.add_argument("--splits", type=Path, default=ROOT / "data/processed/downstream/splits.json")
    ap.add_argument("--output", type=Path, default=ROOT / "data/simulation")
    args = ap.parse_args()
    out = private_dir(args.output)
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    sessions = situations(raw, json.loads(args.splits.read_text(encoding="utf-8")), args.sessions, args.seed)
    t0 = time.time()
    seeker = Seeker(args.seeker, batch_size=args.batch_size)
    openers = seeker([seeker_messages(s, []) for s in sessions], seed=args.seed)
    critic = None
    if args.critic:
        from critic import Critic

        critic = Critic(args.critic)
    results = {}
    for config in args.configs:
        kaki = Kaki.load(args.checkpoints, configs=[config], critic=critic, critic_weight=args.critic_weight)
        chats = converse(kaki, seeker, sessions, openers, config, args.turns, args.seed)
        del kaki
        torch.cuda.empty_cache()
        results[config] = chats
        print(json.dumps({"config": config, "minutes": round((time.time() - t0) / 60, 1)}), flush=True)
    del seeker
    torch.cuda.empty_cache()
    if args.eval_critic:
        from critic import Critic, predict_ratings

        judge = Critic(args.eval_critic)
        for chats in results.values():
            replies = [r for c in chats for r in c["replies"] if not r["bypassed"]]
            scores = predict_ratings(
                judge.model,
                judge.tokenizer,
                [r["history"] for r in replies],
                [r["reply"] for r in replies],
                judge.max_len,
            )
            for r, s in zip(replies, scores):
                r["eval_critic"] = s
    for config, chats in results.items():
        with open(out / f"{config}.jsonl", "w", encoding="utf-8") as f:
            for c in chats:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    summary = {
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "minutes": round((time.time() - t0) / 60, 1),
        "configs": {c: summarise(chats) for c, chats in results.items()},
        "differences": {},
    }
    for measure in ("eval_critic", "seeker_rating"):
        if measure == "eval_critic" and not args.eval_critic:
            continue
        values = {c: per_chat(chats, measure) for c, chats in results.items()}
        configs = list(values)
        for i, a in enumerate(configs):
            for b in configs[i + 1 :]:
                shared = sorted(set(values[a]) & set(values[b]))
                summary["differences"].setdefault(f"{a} - {b}", {})[measure] = {
                    "chats": len(shared),
                    **bootstrap_diff([values[a][k] for k in shared], [values[b][k] for k in shared]),
                }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary["configs"], indent=1))
    print(json.dumps(summary["differences"], indent=1))


if __name__ == "__main__":
    main()

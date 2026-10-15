"""Guard: the risk check that runs before Perceive, Plan, Speak and Critique.

A DistilRoBERTa classifier (trained by src/train_guard.py) scores the latest seeker message for
suicide or self-harm risk. A safety lexicon of first-person risk expressions supports it: an affirmed
match lowers the score needed to flag. A flagged message bypasses generation and gets a prepared
response pointing to human help (configs/guard.json: responses).

Usage, from the repository root with src on the import path:
    from guard import Guard
    guard = Guard("checkpoints/guard")
    result = guard.check(history)        # list of {"speaker", "text"} turns, or a rendered history string
    if result["bypass_generation"]:
        reply = guard.response(result["response_key"])

See docs/guard.md for score interpretation, threshold selection and known limits.
"""
import json
import logging
import re
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs/guard.json"
RESPONSE_KEY = "crisis_sg"
LABELS = ["no_risk", "risk"]

log = logging.getLogger(__name__)

# --- safety lexicon -------------------------------------------------------
# First-person expressions, plus the word "suicidal" in any form. Other third-person mentions ("my
# friend wants to kill herself") and figures of speech ("this exam is killing me") are left to the
# classifier.
LEXICON = {
    "suicidal_intent": [
        r"\bkill(?:ing|ed)? my ?self\b",
        r"\bend(?:ing)? (?:my (?:own )?life|it all)\b",
        r"\btak(?:e|ing) my (?:own )?life\b",
        r"\bcommit(?:ting)? suicide\b",
        r"\bsuicidal\b",
        r"\b(?:(?:want|going|plan(?:ning)?|ready) to|wanna|gonna) die\b(?! (?:of|from) (?:embarrassment|laughter|boredom))",
        r"\bsuicide (?:note|plan|letter)\b",
        r"\bhang(?:ing)? my ?self\b",
        r"\boverdos(?:e|ing) on\b",
        r"\bslit(?:ting)? my wrists?\b",
        r"\bjump(?:ing)? (?:off|from) (?:a|the|my) (?:bridge|building|roof|block|ledge)\b",
    ],
    "self_harm": [
        r"\b(?:hurt(?:ing)?|harm(?:ing)?|cut(?:ting)?|burn(?:ing|t)?) my ?self\b",
        r"\bself[- ]?harm(?:ing)?\b",
    ],
    "passive_ideation": [
        r"\b(?:don'?t|do not|no longer) want to (?:live|be alive|exist|wake up)\b",
        r"\b(?:nothing|no reason) (?:left )?to live for\b",
        r"\bno reason to (?:live|keep living|go on)\b",
        r"\bbetter off (?:dead|without me)\b",
        r"\bwish i (?:was|were) dead\b",
        r"\bwish i (?:had never been|was never|were never|wasn'?t) born\b",
        r"\bnot worth living\b",
        r"\bcan'?t (?:go on|keep going) any ?(?:more|longer)\b",
    ],
}
PATTERNS = [(cat, re.compile(p)) for cat, ps in LEXICON.items() for p in ps]

NEGATION = re.compile(r"\b(?:not|never|no|don'?t|do not|didn'?t|won'?t|wouldn'?t|can'?t|cannot|isn'?t|wasn'?t)\b")
# cues that look like negation but do not negate what follows
PSEUDO_NEGATION = re.compile(r"\b(?:no one|nothing but|not gonna lie|not going to lie|"
                             r"(?:can'?t|cannot|couldn'?t) (?:stop|help)|not sure (?:if|whether)|"
                             r"don'?t know (?:if|whether|why|how))\b")
# someone else's words: "my friend said ...", "he told me to ...", song lyrics
REPORTED = re.compile(r"\b(?:(?:he|she|they|someone|somebody|people|my \w+) (?:said|says|told me|tells me|"
                      r"wrote|texted|joked)|told me to|tells me to|lyrics?|song)\b")
QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”|‘[^’]*’|(?<!\w)'(?:[^']|'(?=\w))+'(?!\w)")
CLAUSE_BREAK = re.compile(r"[.!?;:,\n]|\bbut\b|\bthough\b|\bhowever\b")
NEGATION_WINDOW = 5  # words before the match


def match_lexicon(text: str) -> list[dict]:
    """Lexicon matches in text, each with whether it is negated or quoted/reported.

    Negated: a negation word within NEGATION_WINDOW words before the match, in the same clause
    ("I would never kill myself"). Quoted: inside quotation marks, or after a reporting cue in the
    same clause ("my friend said ..."). Only matches that are neither count as affirmed in decide()."""
    text = text.lower().replace("’", "'").replace("`", "'")
    quoted_spans = [m.span() for m in QUOTED.finditer(text)]
    matches, seen = [], set()
    for category, pattern in PATTERNS:
        for m in pattern.finditer(text):
            if m.span() in seen:
                continue
            seen.add(m.span())
            breaks = [b.end() for b in CLAUSE_BREAK.finditer(text, 0, m.start())]
            clause = text[breaks[-1] if breaks else 0: m.start()]
            window = PSEUDO_NEGATION.sub(" ", " ".join(clause.split()[-NEGATION_WINDOW:]))
            matches.append({
                "category": category,
                "term": m.group(),
                "negated": bool(NEGATION.search(window)),
                "quoted": any(s <= m.start() and m.end() <= e for s, e in quoted_spans)
                          or bool(REPORTED.search(clause)),
            })
    return matches


def decide(score: float, matches: list[dict], thresholds: dict) -> dict:
    """Combine the classifier score and lexicon matches into Guard's output.

    risk_flag is True when score >= thresholds["model"], or when there is an affirmed lexicon match
    (not negated, not quoted) and score >= thresholds["lexicon"], a lower bar. Both thresholds are
    selected on validation data by src/train_guard.py. predicted_label is the classifier's decision
    alone; bypass_generation currently always equals risk_flag."""
    model_hit = score >= thresholds["model"]
    affirmed = any(not m["negated"] and not m["quoted"] for m in matches)
    lexicon_hit = affirmed and score >= thresholds["lexicon"]
    risk = model_hit or lexicon_hit
    return {
        "risk_flag": risk,
        "model_score": score,
        "predicted_label": LABELS[int(model_hit)],
        "lexicon_matches": matches,
        "bypass_generation": risk,
        "response_key": RESPONSE_KEY if risk else None,
        "trigger": "model" if model_hit else ("lexicon" if lexicon_hit else None),
    }


def seeker_turns(history) -> list[str]:
    """Seeker messages, oldest first. Accepts a list of {"speaker", "text"} turns (src/prep.py format),
    a rendered history string ("supporter: ... | seeker: ..."), or a single message string."""
    if isinstance(history, str):
        parts = re.split(r" \| (?=(?:seeker|supporter): )", history.strip())
        if not parts[0].startswith(("seeker: ", "supporter: ")):
            return [history.strip()] if history.strip() else []
        history = [dict(zip(("speaker", "text"), p.split(": ", 1))) for p in parts]
    return [t["text"].strip() for t in history if t["speaker"] == "seeker" and t["text"].strip()]


def guard_text(history, context_turns: int = 1) -> str:
    """The text Guard scores: the last context_turns seeker messages of a history."""
    turns = seeker_turns(history)
    if not turns:
        raise ValueError("History has no seeker message to check.")
    return " ".join(turns[-context_turns:])


@torch.inference_mode()
def risk_scores(model, tokenizer, texts: list[str], max_len: int, batch_size: int = 32) -> list[float]:
    """P(risk) for each text from the classifier's softmax. Shared by Guard and the trainer."""
    model.eval()
    device = next(model.parameters()).device
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths -> less padding
    scores = [0.0] * len(texts)
    for start in range(0, len(order), batch_size):
        idx = order[start: start + batch_size]
        enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=max_len,
                        padding=True, return_tensors="pt").to(device)
        probs = model(**enc).logits.float().softmax(-1)[:, model.config.label2id["risk"]]
        for i, p in zip(idx, probs.tolist()):
            scores[i] = p
    return scores


class Guard:
    def __init__(self, checkpoint=ROOT / "checkpoints/guard", config=CONFIG, device=None):
        checkpoint = Path(checkpoint)
        cfg = json.loads(Path(config).read_text(encoding="utf-8"))
        self.responses = cfg["responses"]
        self.context_turns = cfg["inference"]["context_turns"]
        self.max_len = cfg["train"]["max_len"]
        self.thresholds = json.loads((checkpoint / "thresholds.json").read_text())
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, truncation_side="left")
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(self.device).eval()

    def score(self, texts: list[str]) -> list[float]:
        return risk_scores(self.model, self.tokenizer, texts, self.max_len)

    def check(self, history) -> dict:
        """Guard decision for the latest seeker message (or the last context_turns seeker messages)."""
        return self.check_batch([history])[0]

    def check_batch(self, histories: list) -> list[dict]:
        texts = [guard_text(h, self.context_turns) for h in histories]
        results = [decide(s, match_lexicon(t), self.thresholds) for s, t in zip(self.score(texts), texts)]
        for r in results:
            if r["bypass_generation"]:
                # message text is not logged: session history stays in memory only
                log.warning("guard bypass: trigger=%s score=%.3f lexicon=%s", r["trigger"], r["model_score"],
                            sorted({m["category"] for m in r["lexicon_matches"]}))
        return results

    def response(self, key: str) -> str:
        return self.responses[key]["text"]

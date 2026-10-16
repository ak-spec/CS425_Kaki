"""Kaki demo: the chat beside every stage's decision for each reply.

From the repository root:
    streamlit run app.py                          trained models from checkpoints/ (see docs/demo.md)
    streamlit run app.py -- --fake                canned outputs and no models; KAKI_FAKE=1 does the same
    streamlit run app.py -- --checkpoints DIR

The conversation is kept in st.session_state only. Messages are never written to disk or logged.
"""
import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = str(ROOT / "src")
if SRC not in sys.path:  # Streamlit reruns this file on every interaction
    sys.path.insert(0, SRC)
from pipeline import CONFIGS, Kaki

NOTICE = ("Kaki is a university research prototype (SMU CS425). It is not therapy or a counselling service, and "
          "it is not a reliable crisis detector. If you are in danger, call 995. To talk to someone, call "
          "Samaritans of Singapore at 1767 or national mindline at 1771 (both 24 hours).")
CONFIG_NOTES = {
    "direct": "one reply from the history-only Speak adapter",
    "planning": "one reply for the planner's top strategy",
    "lookahead": "one reply for each of the top three strategies, then the best is kept",
}
TRIGGERS = {"model": "classifier score", "lexicon": "lexicon match and the lower threshold"}

EMOTIONS = ["anger", "anxiety", "depression", "disgust", "fear", "sadness", "shame", "other"]
PROBLEMS = ["academic pressure", "breakup with partner", "conflict with parents", "job crisis",
            "ongoing depression", "problems with friends", "sleep problems", "other"]
TOPICS = [  # (word prefixes in the history, emotion, problem) for the fake Perceive
    (("exam", "midterm", "module", "assignment", "grade", "gpa"), "anxiety", "academic pressure"),
    (("friend", "lonely", "left out"), "sadness", "problems with friends"),
    (("parent", "mum", "mom", "dad"), "anger", "conflict with parents"),
]
RISK_PHRASES = {"kill myself": "suicidal_intent", "end my life": "suicidal_intent", "suicidal": "suicidal_intent",
                "want to die": "suicidal_intent", "hurt myself": "self_harm"}
FIRST_PLAN = [("Question", 0.41), ("Reflection of feelings", 0.22), ("Restatement or Paraphrasing", 0.12)]
LATER_PLAN = [("Affirmation and Reassurance", 0.33), ("Providing Suggestions", 0.26), ("Reflection of feelings", 0.17)]
REPLIES = {
    None: "I'm sorry you're dealing with this. Do you want to tell me more about what happened?",
    "Question": "That sounds like a lot to carry. What has been weighing on you the most?",
    "Reflection of feelings": "It sounds like you're feeling worn down and unsure of yourself, and that makes sense.",
    "Restatement or Paraphrasing": "So this has been building up for a while, and it's hard to see a way through.",
    "Affirmation and Reassurance": "Thank you for telling me. Feeling this way doesn't mean you're failing.",
    "Providing Suggestions": "Could you pick one small thing to do today, or tell someone you trust how you feel?",
}


def spread(labels: list[str], top: str, p: float = 0.6) -> dict:
    """A distribution with p on top and the rest shared equally."""
    rest = (1 - p) / (len(labels) - 1)
    return {label: p if label == top else rest for label in labels}


class FakeGuard:
    """Flags a few explicit phrases in the latest seeker message; the prepared response is the real one."""
    thresholds = {"model": 0.5, "lexicon": 0.2}

    def __init__(self):
        self.responses = json.loads((ROOT / "configs/guard.json").read_text(encoding="utf-8"))["responses"]

    def check(self, turns: list[dict]) -> dict:
        text = [t["text"] for t in turns if t["speaker"] == "seeker"][-1].lower()
        matches = [{"category": category, "term": phrase, "negated": False, "quoted": False}
                   for phrase, category in RISK_PHRASES.items() if phrase in text]
        risk = bool(matches)
        return {"risk_flag": risk, "model_score": 0.97 if risk else 0.03,
                "predicted_label": "risk" if risk else "no_risk", "lexicon_matches": matches,
                "bypass_generation": risk, "response_key": "crisis_sg" if risk else None,
                "trigger": "model" if risk else None}

    def response(self, key: str) -> str:
        return self.responses[key]["text"]


class FakePerceiver:
    def predict(self, history: str) -> dict:
        text = history.lower()
        emotion, problem = next(((e, p) for words, e, p in TOPICS
                                 if any(re.search(rf"\b{w}", text) for w in words)), ("sadness", "other"))
        return {"emotion": emotion, "emotion_probs": spread(EMOTIONS, emotion),
                "problem": problem, "problem_probs": spread(PROBLEMS, problem)}


class FakePlanner:
    def predict(self, history: str) -> list[dict]:
        plan = FIRST_PLAN if history.count("seeker: ") == 1 else LATER_PLAN
        return [{"strategy": s, "probability": p} for s, p in plan]


class FakeSpeaker:
    def generate(self, history: str, emotion=None, strategy=None) -> str:
        return REPLIES[strategy]

    def generate_candidates(self, history: str, emotion: str, strategies: list[dict]) -> list[dict]:
        return [{"strategy": s["strategy"], "planner_probability": s["probability"],
                 "response": self.generate(history, emotion, s["strategy"])} for s in strategies]


def fake_kaki() -> Kaki:
    """Kaki's own orchestration (src/pipeline.py) with canned stand-ins for every model."""
    speaker = FakeSpeaker()
    return Kaki(FakeGuard(), FakePerceiver(), FakePlanner(), {"history": speaker, "conditioned": speaker})


def model_set(config: str) -> tuple:
    """Planning and lookahead load the same models, so they share one Kaki."""
    return ("direct",) if config == "direct" else ("planning", "lookahead")


def missing_files(checkpoints: Path, config: str) -> list[Path]:
    """What Kaki.load needs for config but is not on disk."""
    names = ["guard", "perceive"] + (["speak-history"] if config == "direct" else ["plan", "speak-conditioned"])
    paths = [Path(checkpoints) / n for n in names] + [ROOT / "data/processed/label_maps.json"]
    return [p for p in paths if not p.exists()]


@st.cache_resource(max_entries=1, show_spinner="Loading models...")
def load_kaki(checkpoints: str, configs: tuple) -> Kaki:
    """One model set at a time, so only one copy of the Speak base model is in memory."""
    return Kaki.load(checkpoints, configs=list(configs))


@st.cache_resource
def load_fake() -> Kaki:
    return fake_kaki()


def options() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Kaki demo (streamlit run app.py -- [options])", allow_abbrev=False)
    ap.add_argument("--fake", action="store_true", help="canned outputs, no models (or set KAKI_FAKE=1)")
    ap.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")
    args, _ = ap.parse_known_args()
    args.fake = args.fake or os.environ.get("KAKI_FAKE", "0") not in ("", "0")
    return args


def plain(text: str) -> str:
    """Escape Markdown so a message shows as typed."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~$:])", r"\\\1", text)


def show_seeker(text: str):
    with st.chat_message("user"):
        st.markdown(plain(text))


def show_reply(result: dict):
    out = result["output"]
    with st.chat_message("assistant"):
        if out["bypassed"]:
            st.error(out["reply"])
            st.caption(f"Guard's prepared response, {result['seconds']:.2f} s")
        else:
            st.markdown(plain(out["reply"]))
            st.caption(f"{out['config']}, {result['seconds']:.2f} s")


def probability_table(label: str, probs: dict):
    rows = [{label: k, "Probability": v} for k, v in sorted(probs.items(), key=lambda kv: -kv[1])]
    st.dataframe(rows, hide_index=True, column_config={
        "Probability": st.column_config.ProgressColumn(format="%.2f", min_value=0.0, max_value=1.0)})


def show_guard(guard: dict, thresholds):
    if guard["risk_flag"]:
        st.error(f"Flagged by {TRIGGERS[guard['trigger']]}: generation skipped.")
    else:
        st.success("Not flagged: generation continues.")
    left, right = st.columns(2)
    left.metric("Risk score", f"{guard['model_score']:.2f}")
    right.metric("Classifier alone", guard["predicted_label"])
    if thresholds:
        st.caption(f"Flags at a score of {thresholds['model']:.2f}, or {thresholds['lexicon']:.2f} with an affirmed "
                   "lexicon match (neither negated nor quoted).")
    if guard["lexicon_matches"]:
        st.dataframe([{"Lexicon match": m["term"], "Category": m["category"], "Negated": m["negated"],
                       "Quoted": m["quoted"]} for m in guard["lexicon_matches"]], hide_index=True)
    else:
        st.caption("No lexicon matches.")


def selection_note(out: dict) -> str:
    if out["config"] != "lookahead":
        return "One candidate, so there is nothing to choose."
    if any(c["critic_score"] is not None for c in out["candidates"]):
        return "Kept the highest critic score plus weighted planner probability."
    return "No Critic is loaded, so the candidates are ranked by planner probability."


def scores(c: dict) -> str:
    if c["planner_probability"] is None:
        return "No planner probability or critic score in the direct configuration."
    critic = "no critic score" if c["critic_score"] is None else f"critic score {c['critic_score']:.2f}"
    return f"Planner probability {c['planner_probability']:.2f}, {critic}, final score {c['score']:.2f}"


def show_decision(result: dict, thresholds):
    """Every stage's output for one reply; after a Guard bypass, only Guard and its prepared response."""
    out = result["output"]
    st.caption(f"Configuration: {out['config']}. Response time: {result['seconds']:.2f} s.")
    st.markdown("##### Guard")
    show_guard(out["guard"], thresholds)
    if out["bypassed"]:
        st.caption(f"Kaki replied with Guard's prepared response ({out['guard']['response_key']} in "
                   "configs/guard.json), shown in the chat. Perceive, Plan, Speak and Critique did not run for "
                   "this message.")
        return

    st.markdown("##### Perceive")
    p = out["perceive"]
    st.markdown(f"Emotion **{p['emotion']}** ({p['emotion_probs'][p['emotion']]:.2f}), "
                f"problem **{p['problem']}** ({p['problem_probs'][p['problem']]:.2f})")
    emotion_tab, problem_tab = st.tabs(["All emotions", "All problems"])
    with emotion_tab:
        probability_table("Emotion", p["emotion_probs"])
    with problem_tab:
        probability_table("Problem", p["problem_probs"])

    st.markdown("##### Plan")
    if out["strategies"] is None:
        st.caption("Direct configuration: no planner. The history-only Speak adapter writes the reply.")
    else:
        probability_table("Strategy", {s["strategy"]: s["probability"] for s in out["strategies"]})
        st.caption("Probabilities are over all eight strategies, so the top three need not sum to one.")

    st.markdown("##### Speak and Critique")
    st.caption(selection_note(out))
    for i, c in enumerate(out["candidates"]):
        with st.container(border=True):
            badge = " :green-badge[selected]" if i == out["selected"] else ""
            st.markdown(f"**{c['strategy'] or 'History only'}**{badge}")
            st.markdown(plain(c["response"]))
            st.caption(scores(c))


def main():
    args = options()
    st.set_page_config(page_title="Kaki demo", layout="wide")
    state = st.session_state
    if "turns" not in state:
        state.turns, state.results = [], []

    with st.sidebar:
        config = st.radio("Configuration", CONFIGS, index=CONFIGS.index("lookahead"), key="config",
                          captions=[CONFIG_NOTES[c] for c in CONFIGS])
        if st.button("Clear conversation"):
            state.turns, state.results = [], []
        st.caption("Fake mode: canned outputs, no models." if args.fake else f"Models from {args.checkpoints}")
        st.caption("The conversation exists only in this browser session's memory. It is never saved, and it is "
                   "gone when you clear it, reload the page or close the tab.")

    st.title("Kaki")
    st.caption("Guard -> Perceive -> Plan -> Speak -> Critique")
    st.warning(NOTICE)
    if args.fake:
        st.info("Fake mode: every stage returns canned output and no model is loaded.")
        kaki = load_fake()
    else:
        missing = missing_files(args.checkpoints, config)
        if missing:
            st.error(f"The {config} configuration needs files that are not there: "
                     + ", ".join(str(p) for p in missing)
                     + ". Run with --fake to try the interface without models (docs/demo.md).")
            st.stop()
        wanted = model_set(config)
        if state.get("loaded") not in (None, wanted):
            # free the other model set first, so a laptop never holds two copies of the Speak base model
            load_kaki.clear()
            gc.collect()
        state.loaded = wanted
        kaki = load_kaki(str(args.checkpoints), wanted)

    prompt = st.chat_input("Message Kaki")
    chat, panel = st.columns(2, gap="large")
    with chat, st.container(height=520):
        replies = iter(state.results)
        for turn in state.turns:
            if turn["speaker"] == "seeker":
                show_seeker(turn["text"])
            else:
                show_reply(next(replies))
        if prompt and prompt.strip():
            show_seeker(prompt)
            turns = state.turns + [{"speaker": "seeker", "text": prompt}]
            with st.spinner("Kaki is replying..."):
                start = time.perf_counter()
                out = kaki.respond(turns, config)
                result = {"output": out, "seconds": time.perf_counter() - start}
            state.turns = turns + [{"speaker": "supporter", "text": out["reply"]}]
            state.results.append(result)
            show_reply(result)

    with panel:
        st.subheader("How Kaki decided")
        if not state.results:
            st.caption("Send a message to see Guard's decision, the perceived emotion and problem, the planned "
                       "strategies and the candidate replies.")
        else:
            n = len(state.results)
            i = n - 1
            if n > 1:
                i = st.selectbox("Reply", range(n), index=n - 1, format_func=lambda k: f"Reply {k + 1}")
            show_decision(state.results[i], getattr(kaki.guard, "thresholds", None))


if __name__ == "__main__":
    main()

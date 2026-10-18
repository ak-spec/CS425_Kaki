# Demo

`app.py` is a Streamlit page for trying Kaki and for showing how it reached each reply: the chat is on the
left, and every stage's output for a reply is on the right. It is meant for presentations and the report,
not for people in distress.

## Run

From the repository root, with the pinned requirements installed (they include `streamlit`):

```sh
streamlit run app.py -- --fake                      # canned outputs, no models or checkpoints
streamlit run app.py                                # trained models from checkpoints/
streamlit run app.py -- --checkpoints other/dir     # trained models from another directory
```

Options for the app go after `--`. Setting `KAKI_FAKE=1` also starts fake mode.

Fake mode runs Kaki's own orchestration (`Kaki.respond` in `src/pipeline.py`) with canned stand-ins for
every model, so the page can be developed and tested without checkpoints. The stand-in Guard flags a few
explicit phrases ("kill myself", "end my life", "suicidal", "want to die", "hurt myself") and returns the
real prepared response from `configs/guard.json`. Perceive picks an emotion and problem from keywords, and
Plan and Speak return fixed strategies and replies. None of it reflects the trained models.

Real mode calls `Kaki.load(checkpoints, configs=[...])` for the chosen configuration.
`python src/download_checkpoints.py` fetches every file below from the repository's v1.0 release. It needs:

| Configuration | Files |
|---|---|
| all | `checkpoints/guard`, `checkpoints/perceive`, `data/processed/label_maps.json` |
| direct | `checkpoints/speak-history` |
| planning, lookahead | `checkpoints/plan`, `checkpoints/speak-conditioned` |
| lookahead, optional | `checkpoints/critic`: the reply-contrast Critic with its `lookahead.json` |

If any are missing, the page lists them and stops. The first real run also downloads `distilroberta-base`
(Perceive's tokenizer and configuration) and `Qwen/Qwen2.5-0.5B-Instruct` (Speak's base model, about 1 GB)
from the Hugging Face Hub unless they are already cached.

For look-ahead, `Kaki.load` uses `checkpoints/critic` when it holds a `lookahead.json`. The release's
`checkpoints/critic` is the reply-contrast Critic with the weight chosen on validation, 0 (see
[critic](critic.md)); when training from scratch, copy `checkpoints/critic-contrast` there. Without it, look-ahead ranks its three
candidates by planner probability and gives the planning reply.

`tests/test_app.py` checks the fake Kaki's output against the pipeline's format and runs the page in fake
mode with Streamlit's `AppTest`.

## The page

- **Notice** at the top: Kaki is a university prototype, not therapy, and not a reliable crisis detector,
  with the emergency number and two 24-hour helplines.
- **Sidebar**: the configuration (direct, planning, lookahead), a button that clears the conversation, and
  the privacy note. The configuration can change during a conversation; each reply records its own.
- **Chat**: each reply shows its configuration and response time, the time `Kaki.respond` took (model
  loading is not included).
- **How Kaki decided**, for the latest reply or one picked from the list:
  - Guard: flagged or not, the risk score, the classifier's label on its own, the thresholds, and any
    lexicon matches with whether each was negated or quoted.
  - Perceive: the predicted emotion and problem with their probabilities, and the full distributions.
  - Plan: the top three strategies with their probabilities from the eight-class softmax. Direct has no
    planner.
  - Speak and Critique: each candidate reply with its planner probability, critic score and final score.
    The reply Kaki sent is marked selected.

When Guard flags a message, the chat shows its prepared response in a red box and the panel shows only
Guard's decision. Perceive, Plan, Speak and Critique do not run, so nothing from them is shown.

## Privacy

The conversation is kept in `st.session_state`, which lives in the Streamlit server's memory for one browser
session. The app never writes messages to disk or logs them; Guard's log line on a bypass records only the
trigger, score and lexicon categories. The history is gone when the conversation is cleared, the page is
reloaded or the tab is closed. Run the demo on your own machine. Streamlit's anonymous usage statistics
can be turned off with `--browser.gatherUsageStats false`.

## CPU and memory

Models are loaded once with `st.cache_resource` and reused across sessions. Planning and lookahead share one
loaded set; switching to or from direct replaces it, so only one copy of the Speak base model is in memory.

Timings on a laptop CPU (Intel i5-11300H, 4 cores, 16 GB RAM), measured with the base models rather than
the trained checkpoints: Qwen2.5-0.5B-Instruct loads in about 3 s, uses about 1.6 GB of RAM in bfloat16
and generates about 11 tokens per second (128 tokens in 11.5 s). A RoBERTa-base forward pass on a short
history takes about 0.1 s. A reply stops at the end-of-sequence token or at the 128-token cap, so expect a
few seconds per reply for direct and planning and about three times that for lookahead, which writes its
three replies one after another (up to about 35 s if all three reach the cap). Expect about 3 GB of RAM
with the encoders loaded. The response time shown in the chat is the figure to report for the trained
checkpoints.

## Checked with the real models

On 16 October 2026 the page ran headlessly (Streamlit's `AppTest`) with the trained checkpoints on four CPU
threads, as on a laptop: an ordinary message and a Singlish one in look-ahead (7 and 9 seconds per reply,
the Critic scoring three candidates), one each in planning (3 seconds) and direct (3 seconds), and a risk
message, which Guard handed over without generating. Peak memory was 5.2 GB across all three
configurations: switching between direct and the other two frees the loaded models first and reloads in
about 20 seconds, so stay on planning and look-ahead during a demo if memory is tight. The first run
downloads Qwen2.5-0.5B-Instruct and distilroberta-base (about 1.5 GB).

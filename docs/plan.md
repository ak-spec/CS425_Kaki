# Plan and downstream data preparation

This branch implements the proposal's RoBERTa-base Plan stage, which ranks eight
support strategies and returns the top three. It does not change Perceive or
replace the three-candidate Speak/Critique pipeline.

## Run

Use a Python environment with the repository requirements installed. From the
repository root:

```sh
python src/prep_downstream.py
python -m unittest discover -s tests -v
python src/plan.py --smoke --output checkpoints/plan-smoke
python src/plan.py --output checkpoints/plan
```

Full training defaults to `FacebookAI/roberta-base`, three epochs, batch size 8,
AdamW learning rate 2e-5 and weight decay 0.01. It clips gradient norm to 1.0,
uses float32, and truncates histories from the left to 512 tokens. Reduce batch
size if GPU memory is insufficient. These are starting settings, not a tuned
configuration. The smoke run uses only 16 training and 16 validation examples
and checks checkpoint reload and inference; its scores are not research results.
Each run needs a fresh output directory to avoid overwriting a trained model.

The trainer selects the checkpoint with the highest validation macro-F1 across
all eight classes. It records accuracy, macro-F1, top-three reference coverage
and per-class results in `run.json`. After development is finished, explicitly
evaluate the chosen checkpoint once:

```sh
python src/plan.py --output checkpoints/plan --evaluate-test
```

Inference, with `src` on the import path:

```python
from plan import Planner
planner = Planner("checkpoints/plan")
candidates = planner.predict("seeker: I am worried about my exams.")
# [{"strategy": <label>, "probability": <float>}, ...] ordered best first
```

Probabilities come from the complete eight-class softmax; the three returned
values are not renormalised and need not sum to one. They are not calibrated
confidence estimates. Speak should generate one candidate for each strategy,
and Critique should apply the validation-selected combined scoring rule.

## Data contract and audit

Outputs are under ignored `data/processed/downstream/`. Perceive inputs are not
modified. Existing `data/processed/splits.json` is authoritative. If absent,
the script reconstructs seed-42 splits using the unchanged functions in
`src/prep.py`, writes only its own copy, and records this in `audit.json`.
Before integration, compare the dialogue IDs with the Perceive manifest.
The input dialogues must be covered exactly once across train/validation/test.

`responses_{split}.jsonl` contains one labelled supporter utterance per row:
dialogue ID, original utterance index, preceding history, response, strategy,
integer label, and raw/collapsed emotion and problem labels. Consecutive
supporter utterances retain their separate strategies; the next utterance's
history includes any preceding supporter utterance. This is a teacher-forced
next-utterance task. Report evaluation on first supporter utterances after
seeker messages separately before interpreting performance as chat-turn quality.
Never feed the response, target strategy, situation description, future seeker
messages or feedback into Plan's input. Both Speak adapters should use the same
eligible rows, adding emotion/strategy only for the conditioned adapter.

`critic_proxy_{split}.jsonl` retains each valid seeker feedback rating once,
paired with the entire immediately preceding contiguous supporter block.
Its history stops before that block. Original indices and all block strategies
are preserved. Ratings without an immediately preceding supporter block are
counted as unpaired rather than assigned an invented target.

**These are provisional proxy pairs, not verified per-reply quality labels.**
The feedback may reflect cumulative support rather than the most recent reply.
Validate the collection protocol and manually inspect examples before training
Critic. Do not duplicate one rating across every utterance of a multi-strategy
block, use post-conversation survey scores as per-reply labels, or interpret
unrated examples as negative feedback. `audit.json` reports rating imbalance,
exclusions and strategy distributions as well as a raw-data fingerprint.

The subsequent [feedback audit](feedback-audit.md) confirms that the original
collection protocol requested a rating after two supporter utterances. The
last-block proxy therefore needs explicit weak-supervision treatment.

## Remaining integration decisions

- Agree on a Critic target after auditing the rating semantics. Compare against
  a mean-score baseline because ratings may be heavily skewed.
- Partition training dialogues further for the selection Critic and independent
  Eval-Critic before either trains; neither can train on validation/test data.
- DPO pairs must use training contexts only. Guard needs a separately justified
  risk-labelled source; emotional-support strategies are not risk labels.
- Scope remains English. Singlish prompts are exploratory checks, not training
  claims. No Guard, Speak, Critic, DPO or simulator model is trained here.

The existing Perceive notebook and preparation script are unchanged. Datasets,
checkpoints and Python environments are not versioned.

## Data preparation results

The 15 October 2026 audit of the 1,300-dialogue release produced:

| Split | Dialogues | Response examples | Provisional feedback pairs |
| --- | ---: | ---: | ---: |
| Train | 1,038 | 14,180 | 6,129 |
| Validation | 131 | 1,743 | 752 |
| Test | 131 | 1,770 | 764 |

There were no duplicate dialogues. The script excluded 683 opening supporter
utterances with no preceding history. Of 7,681 feedback annotations, 36 lacked
an immediately preceding supporter utterance and were left unpaired. Overall,
6,308 ratings were 4 or 5, which motivates a high-score baseline for Critic.
These are data counts, not model performance results.

The RoBERTa-base smoke run completed on CPU with 16 training and 16 validation
examples, including checkpoint save, reload and top-three inference. Full
training and final test evaluation remain outstanding. The split manifest's
LF-normalised SHA-256 prefix is `dd619339a81a`, matching the original split;
Windows CRLF line endings change the file hash without changing dialogue IDs.

# Plan

`src/plan.py` implements the proposal's RoBERTa-base Plan stage, which ranks the
eight support strategies and returns the top three.

## Run

Use a Python environment with the repository requirements installed. From the
repository root:

```sh
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
confidence estimates. In look-ahead, Speak writes one reply for each of the
three and the Critic picks one (`src/pipeline.py`).

## Training results

Full training with the defaults above on one RTX 3090 took 18 minutes. The best
epoch (3) on validation:

| Run | Accuracy | Macro-F1 | Top-three coverage |
| --- | ---: | ---: | ---: |
| Default (unweighted) | 0.326 | 0.191 | 0.644 |
| `--class-weight-power 0.5` | 0.322 | 0.209 | 0.625 |

The default run never predicts four strategies (Affirmation and Reassurance,
Information, Reflection of feelings, Self-disclosure) and over-predicts
Providing Suggestions (recall 0.79). Tempered class weights, as used by
Perceive and Guard, raise macro-F1 slightly but lower top-three coverage, and
three strategies are still never predicted, so the default stays. Both are
single-seed runs, and differences of this size are within run-to-run variation.

For look-ahead this means the three candidates are mostly drawn from Others,
Providing Suggestions, Question and Restatement or Paraphrasing. The reference
strategy is among the top three 64% of the time.

The default checkpoint was then evaluated once on test (1,770 examples): accuracy
0.321, macro-F1 0.202, weighted-F1 0.248, top-three coverage 0.651, in line with validation.
Given the reference strategy instead, the conditioned Speak adapter's replies gain 2.4 BERTScore points
(`src/ablations.py`, `docs/results.md`), so a better planner is the most direct way to improve planning.

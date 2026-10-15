# Speak training and candidate generation

Speak uses Qwen2.5-0.5B-Instruct with LoRA adapters on the query and value
projections. The history-only adapter is the direct-generation baseline. The
conditioned adapter receives the history, emotion and support strategy; the
single-strategy and three-strategy configurations share this adapter.

Install the repository requirements (they include `peft`) in a local environment.
Generate downstream data using `python src/prep_downstream.py`, then run:

```sh
python src/train_speak.py --mode history --output checkpoints/speak-history
python src/train_speak.py --mode conditioned --output checkpoints/speak-conditioned
```

Both runs use the same response rows and seed. Defaults are three epochs,
batch size one, learning rate 0.0002, LoRA rank eight, 1,024 total tokens and
128 reply tokens. Only adapter parameters are updated. Prompts and padding are
masked out of the training loss; the target includes an end-of-sequence token.
Long histories are shortened from the left while keeping the instructions and
chat markers. The conditioned prompt leaves slightly less room for history;
consider a fixed shared history budget when comparing long-context cases.

The lowest token-weighted validation loss selects the saved adapter. Test data
is not loaded. Checkpoints record configuration, data hashes and per-epoch
validation losses. Use the same settings and compare the data hashes between
the two runs. These are initial settings, not tuned hyperparameters.

For a short functional check, add `--smoke --max-length 256
--max-reply-tokens 32` and use a separate output directory. This uses two
training and two validation rows and does not establish response quality.
Full-length training needs its own memory pilot on the intended GPU.

## Inference

With `src` on the Python import path:

```python
from plan import Planner
from speak import Speaker

planner = Planner("checkpoints/plan")
speaker = Speaker("checkpoints/speak-conditioned")
history = "seeker: I am worried about my exams."
strategies = planner.predict(history)
candidates = speaker.generate_candidates(history, "anxiety", strategies)
```

Use Perceive's predicted emotion at inference; the string above is an example.
Each candidate contains `strategy`, `planner_probability` and `response`, in
the original planner order. Probabilities remain those from the eight-class
distribution. Generation is greedy with the saved reply-length limit. Three
different strategies may still produce identical replies; preserve and report
that behaviour instead of silently retrying until the outputs differ.

For the baseline, load `checkpoints/speak-history` and call
`speaker.generate(history)`. For the single-strategy comparison, call the
conditioned adapter's `generate(history, emotion, strategies[0]["strategy"])`.
Guard must approve a message before these methods run in the application.
Critique will select among the candidates; this module does not implement that
selection or claim that a generated reply is safe.

For an offline integration check:

```sh
python src/generate_candidates.py --planner checkpoints/plan --speaker checkpoints/speak-conditioned --emotion anxiety --history "seeker: I am worried about my exams."
```

This developer command omits Guard and Perceive and prints all three candidates;
it is not the user-facing application.

Model and adapter API references:
[Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
[PEFT quicktour](https://huggingface.co/docs/peft/en/quicktour).

## Verification

On 15 October 2026, both modes completed CPU smoke training using the actual
Qwen model, two training rows and two validation rows. The saved conditioned
adapter was reloaded and connected to the Plan smoke checkpoint, producing
three nonempty candidates with strategy order and probabilities preserved.
All ten repository tests passed. These checks establish functional integration;
the smoke checkpoints are not suitable for reporting quality results.

## Training results

Both adapters were trained with the defaults above on one RTX 3090. Training
updates only the LoRA weights, which stay in fp32; the frozen Qwen weights load
in bf16. Batch size one peaked at 3.8 GB of GPU memory; batch size eight ran out
of memory on the 24 GB card at 1,024 tokens. Each run took 64 minutes.

| Adapter | Validation loss, epochs 1 / 2 / 3 | Kept |
| --- | --- | --- |
| history | 2.612 / 2.616 / 2.639 | epoch 1 |
| conditioned | 2.595 / 2.597 / 2.626 | epoch 1 |

Both stop improving after one epoch. The conditioned adapter's lower loss means
the reference reply is slightly easier to predict once the emotion and the
reference strategy are given.

In the full pipeline on a few handwritten messages, replies are short and
fluent, and a look-ahead turn takes about 1.5 seconds on the GPU. Strategy
control looks weak: a reply generated for one strategy often reads like
another, and two strategies sometimes give the same reply. Measure this on
test histories before relying on it. Test has not been evaluated.

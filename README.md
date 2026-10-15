# Kaki: an emotional-support chatbot that plans, critiques and escalates

CS425 Generative AI for Natural Language Communication (SMU, 2026), group G2T1.

Kaki is an English-language emotional-support chatbot for university students in Singapore (academic pressure, difficulties with friends). Instead of one model guessing a reply, a small pipeline picks a support strategy, writes a reply for each of the top three strategies, and lets a trained Critic choose. It is a research prototype, **not therapy**, and makes no claim of improving mental health. It runs on downloaded models with no external LLM API.

```
Guard -> Perceive -> Plan -> Speak (x3 strategies) -> Critique -> reply
```

| Stage | Model | Job |
|---|---|---|
| Guard | DistilRoBERTa classifier + safety lexicon | detect crisis messages, bypass generation, show human-help info |
| **Perceive** | DistilRoBERTa | predict the seeker's emotion and problem type from the history |
| Plan | RoBERTa-base | rank the 8 support strategies, keep the top 3 |
| Speak | Qwen2.5-0.5B-Instruct + LoRA | write one reply per strategy (optional DPO stage) |
| Critique | DeBERTa-v3-base cross-encoder | score each reply; final pick combines Critic score and planner probability |

Data: [ESConv](https://huggingface.co/datasets/thu-coai/esconv) (Liu et al., ACL 2021), 1,300 emotional-support dialogues with labelled strategies and seeker feedback.

## Status

| Component | State |
|---|---|
| Data preprocessing (`src/prep.py`) | done |
| Perceive training notebook | written and smoke-tested; full training run pending |
| Guard (`src/guard.py`, see `docs/guard.md`) | trained and evaluated |
| Plan, Speak, Critique, simulator, demo | not started |

Guard's results are in `docs/guard.md`; no other results are reported yet.

## Repository layout

```
src/prep.py              ESConv download, cleaning, label collapsing, split, sample building
train_perceive.ipynb     Perceive (emotion + problem classifier) training and evaluation
src/prep_guard.py        Guard data: C-SSRS Reddit labels, checksums, dedup, split by user
src/train_guard.py       Guard training, threshold selection and evaluation
src/guard.py             Guard inference: classifier, lexicon, prepared response
configs/guard.json       Guard settings and crisis response with checked contacts
tests/test_guard.py      Guard tests (python -m unittest discover -s tests)
docs/guard.md            Guard data, method, results and known limits
requirements.txt         pinned local dependencies
data/                    generated, not tracked by git (see below)
checkpoints/             generated, not tracked by git
```

### Platforms

`train_perceive.ipynb` runs locally, on Kaggle and on Google Colab from `main`. Its first code cell detects the platform and sets the data and output paths; the note under the title has the steps for each platform.

## Setup

```bash
python -m venv nlc_venv
source nlc_venv/bin/activate
pip install -r requirements.txt
```

On Kaggle and Colab, do not install the pinned requirements; those platforms ship their own torch and transformers. See the platform note at the top of the notebook.

## 1. Preprocessing

```bash
python src/prep.py            # defaults: --seed 42 --raw data/raw --out data/processed
```

This downloads `ESConv.json` from the Hugging Face Hub (`huggingface_hub`, cached in `data/raw/`) and writes everything to `data/processed/`.

Steps, in order:

1. **Dedup**: drop exact-duplicate dialogues (0 found). `dialogue_id` is the dialogue's position in the raw file.
2. **Clean and merge**: collapse whitespace and join consecutive same-speaker utterances into one turn (7,973 merges; gives 15,325 supporter and 15,067 seeker turns).
3. **Collapse labels** (raw labels are kept in every row):
   - emotion: `nervousness` -> `anxiety`; `pain`, `jealousy`, `guilt` -> `other`
   - problem: `issues with parents` merged into `conflict with parents`; problem classes with fewer than 15 dialogues -> `other`
4. **Split** by dialogue, 80/10/10, seeded and stratified on problem type, *before* any samples are built, so no prefixes of one dialogue cross splits.
5. **Build samples**: one row per seeker turn, containing the history up to and including that turn.

### Output files (`data/processed/`)

| File | Contents |
|---|---|
| `train.jsonl`, `val.jsonl`, `test.jsonl` | one sample per line |
| `splits.json` | dialogue ids per split and the seed; every later stage must reuse it |
| `label_maps.json` | class name to integer id for both label sets |
| `prep_report.json` | counts and class distributions per split |

One sample:

```json
{"dialogue_id": 5, "turn_idx": 1,
 "text": "supporter: Hi there, how are you doing? | seeker: could be better",
 "emotion_raw": "anxiety", "emotion": "anxiety",
 "problem_raw": "job crisis", "problem": "job crisis", "in_scope": true}
```

- `text` is the history, with speaker tags and no strategy labels (the bot's own earlier replies are untagged at inference).
- `turn_idx` is the 1-based seeker-turn position, used to report accuracy by position.
- `emotion` and `problem` are the training targets and are constant for a whole dialogue.
- `in_scope` marks the five main problem types (ongoing depression, job crisis, breakup with partner, problems with friends, academic pressure).

### Data summary

| Split | Dialogues | Samples |
|---|---|---|
| train | 1,038 | 12,077 |
| val | 131 | 1,454 |
| test | 131 | 1,536 |
| total | 1,300 | 15,067 |

Classes: 8 emotions (anger, anxiety, depression, disgust, fear, sadness, shame, other) and 8 problem types (academic pressure, breakup with partner, conflict with parents, job crisis, ongoing depression, problems with friends, sleep problems, other). The notebook prints a fingerprint of `splits.json` (`dd619339a81a` for seed 42 with this code); the same value on another machine means the same splits.

`data/` is not committed: ESConv is released for academic use only. Regenerate it with `prep.py`, or upload `data/processed/` to Kaggle or Drive as the platform notes describe.

## 2. Training Perceive

Open `train_perceive.ipynb` and run all cells (set `smoke_test=True` in `CFG` for a quick pipeline check on a small subset).

- **Model**: one shared DistilRoBERTa encoder (6 layers, 12 attention heads, hidden size 768) feeding two linear classifier heads, one for emotion and one for problem type, both reading the `<s>` token.
- **Input**: the `text` field, truncated from the **left** to 512 tokens so the latest seeker message always survives.
- **Loss**: cross-entropy per head, summed, with tempered class weights computed from the train split only.
- **Model selection**: best epoch by the mean of the two validation macro-F1 scores. The test split is scored once, at the end.

### Hyper-parameters

| Parameter | Value |
|---|---|
| Base model | `distilroberta-base` |
| Max length | 512 tokens (left truncation) |
| Batch size | 16 |
| Optimiser | AdamW, learning rate 2e-5, weight decay 0.01 |
| Schedule | linear decay, 10% warmup |
| Epochs | 4 |
| Dropout (before heads) | 0.1 |
| Gradient clipping | 1.0 |
| Class weights | (N / (K * n_c)) ** 0.5, train split only |
| Precision | fp16 autocast on CUDA, float32 otherwise |
| Seed | 42 |

## 3. Evaluation

The notebook reports, for validation and test:

- accuracy, macro-F1 and weighted-F1 for each head, overall and on the in-scope subset
- accuracy by seeker-turn position (turns 1, 2, 3, 4-5, 6-8, 9+), because early histories often contain too little evidence to name the emotion
- per-class precision, recall and F1 on the test split
- a check of whether averaging class probabilities over the turns so far improves predictions

It saves `checkpoints/perceive/best.pt`, `run.json` (configuration, per-epoch history and reports) and `perceive_preds_{val,test}.jsonl` (predicted probabilities per sample). Later stages use the predictions to run on predicted, not gold, emotion.

Macro-F1 is computed over the classes that occur in the evaluated split, so classes with no examples there (for example emotion `other` in val and test) are not scored. Some classes have very few validation and test dialogues, so their per-class numbers are noisy.

## Ethics and licence

Kaki is a university prototype for research, not a mental-health service, and it does not claim to detect crises reliably. ESConv is used under its research-use terms (CC BY-NC 4.0). The demo is planned to keep session history in memory only.

## References

- Liu et al. *Towards Emotional Support Dialog Systems.* ACL 2021 (ESConv).
- Sanh et al. *DistilBERT, a distilled version of BERT.* 2019 (DistilRoBERTa uses the same recipe).

# Data

Three preparation scripts build everything the models train on. None of the data is committed: ESConv is
for academic use only (CC BY-NC 4.0) and the scripts download it again, with the same splits on every
machine.

| Script | Source | Used by |
|---|---|---|
| `src/prep.py` | ESConv | Perceive |
| `src/prep_downstream.py` | ESConv, same dialogue splits | Plan, Speak, the Critics, evaluation |
| `src/prep_guard.py` | C-SSRS-labelled Reddit posts (Gaur et al. 2021) | Guard (see `docs/guard.md`) |

## ESConv for Perceive (`src/prep.py`)

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

`data/` is not committed: ESConv is released for academic use only. Regenerate it with `prep.py`, or upload `data/processed/` to Kaggle or Drive as `docs/perceive.md` describes.

## ESConv for Plan, Speak and the Critic (`src/prep_downstream.py`)

```bash
python src/prep_downstream.py
```

It reuses the dialogue splits in `data/processed/splits.json`, so Perceive, Plan, Speak and the Critics
share train, validation and test dialogues, and writes to `data/processed/downstream/`:

- `responses_{split}.jsonl`: one row per supporter utterance, with the history before it, the reply, its
  strategy (and integer label), and the dialogue's emotion and problem. Plan learns history -> strategy,
  Speak learns history (+ emotion and strategy) -> reply, and the evaluation scripts answer these rows.
  Consecutive supporter utterances stay separate, each with its own strategy, so the second one's history
  includes the first. `after_seeker` in the evaluation marks rows whose history ends with a seeker message,
  the case a chat user produces.
- `critic_proxy_{split}.jsonl`: each seeker rating (1-5), paired with the block of supporter utterances
  just before it and the history before that block. These are weak labels for one reply: ESConv asked for a
  rating after every two supporter utterances (`docs/feedback-audit.md`), and the Critic is trained with
  that in mind (`docs/critic.md`).
- `audit.json`: counts, exclusions, rating and strategy distributions, and a fingerprint of the raw data.

### Counts

The 15 October 2026 audit of the 1,300-dialogue release produced:

| Split | Dialogues | Response examples | Rating pairs |
| --- | ---: | ---: | ---: |
| Train | 1,038 | 14,180 | 6,129 |
| Validation | 131 | 1,743 | 752 |
| Test | 131 | 1,770 | 764 |

There were no duplicate dialogues. The script excluded 683 opening supporter
utterances with no preceding history. Of 7,681 feedback annotations, 36 lacked
an immediately preceding supporter utterance and were left unpaired. Of all
ratings, 6,308 were 4 or 5.

The split manifest's SHA-256 prefix is `dd619339a81a`, matching the original
split on every OS.

## Risk-labelled posts for Guard (`src/prep_guard.py`)

Described with Guard in `docs/guard.md` (source, label mapping, deduplication and the split by user).

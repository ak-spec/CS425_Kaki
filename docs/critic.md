# Critique

The Critic is a DeBERTa-v3-base cross-encoder that reads the dialogue history and one candidate reply
and predicts the seeker's helpfulness rating (1 to 5). The pipeline uses it to choose among the three
look-ahead candidates (`src/pipeline.py`); an independent Eval-Critic scores the replies of all three
configurations.

## Run

```sh
python src/prep_downstream.py
python src/train_critic.py --half a --output checkpoints/critic        # selection Critic
python src/train_critic.py --half b --output checkpoints/eval-critic   # Eval-Critic
python src/train_critic.py --half a --output checkpoints/critic --evaluate-test   # once, at the end
```

`--smoke` trains on 64 rows for one epoch with 128-token inputs and never reads test.

## Data and training

Rows come from `critic_proxy_{split}.jsonl`: each seeker rating paired with the supporter block just
before it, and the history before that block. ESConv asked for a rating after every two supporter
utterances, so these are weak labels for a single reply ([feedback audit](feedback-audit.md)).

The 1,038 training dialogues are split in half with seed 42 (519 each): half `a` trains the Critic and
half `b` the Eval-Critic, so they never share training dialogues. Neither sees validation or test.

| Setting | Value |
|---|---|
| Model | `microsoft/deberta-v3-base`, one regression output |
| Target | (rating - 3) / 2, mean squared error |
| Input | history and reply as a sentence pair, 512 tokens, oldest history dropped first |
| Optimiser | AdamW, lr 2e-5, weight decay 0.01, linear decay with 10% warmup, batch 16, 3 epochs |
| Precision | weights in fp32 (the hub checkpoint is stored in fp16 and diverges if trained in it), bf16 autocast on GPU |
| Selection | epoch with the best validation pairwise accuracy |

Each run took about 4 minutes on an RTX 3090.

## Validation results

752 validation ratings from 131 dialogues. "Pairs" are two ratings in the same dialogue with different
values; the higher-rated block should score higher.

| Metric | Critic (half a) | Eval-Critic (half b) | Baseline |
|---|---|---|---|
| Mean absolute error | 0.676 | 0.676 | 0.734 (predict the training mean) |
| Spearman with rating | 0.22 | 0.27 | |
| Pairwise accuracy (759 pairs) | 0.684 | 0.705 | 0.710 (later block is better); 0.553 (longer reply is better) |
| Pairwise accuracy on the 220 pairs where the earlier block was rated higher | 0.082 | 0.055 | 0 by construction for the position baseline |
| Real reply scored above a reply from another dialogue | 0.484 | 0.543 | 0.5 is chance |

Mean prediction by true rating (Critic): 4.41, 4.32, 4.28, 4.47, 4.52 for ratings 1 to 5.

## What this means

Ratings rise through a conversation (mean 4.04 in the first quarter of a dialogue, 4.49 in the last),
and a seeker's ratings resemble each other. Both are visible in the history, so the Critics learn the
history and largely ignore the reply: they do no better than "later is better", fail almost every pair
that goes against that trend, and cannot tell the real reply from one taken from another conversation.

For Kaki this matters directly. The three look-ahead candidates share one history, so anything learnt
from the history cancels out, and the reply-dependent part of the score is close to noise. Selecting by
this Critic, or scoring systems with this Eval-Critic, would not measure reply quality. Until the target
changes, `src/pipeline.py` ranks candidates by planner probability, and comparisons between
configurations should rely on reference metrics and the human evaluation.

## Options (team decision)

- **Reply-contrast training.** Add pairs where the same history gets its real reply and a reply from
  another dialogue, and require the real reply to score higher, alongside the rating target. This forces
  the Critic to read the reply; it measures fit to the context rather than seeker-rated helpfulness, and
  the "another dialogue" check above would no longer be a fair test.
- **Remove the history-level part of the target.** Subtract each dialogue's mean rating and the
  position trend, and train on what remains. Keeps the feedback signal but leaves very little of it.
- **Interval target.** Use the full two-utterance interval each rating covers, as the audit suggests.
  Likely more faithful to the labels, but it does not address the position effect.

Whichever is chosen should be compared on validation with the checks above before the test split is
used. Test has not been evaluated for either Critic.

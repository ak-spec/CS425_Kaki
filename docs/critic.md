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
this Critic, or scoring systems with this Eval-Critic, would not measure reply quality, and without a
Critic look-ahead sends the same reply as planning. Hence the reply-contrast training below.

## Reply-contrast training

`--contrast-weight 1.0` adds a margin loss: for the same history, the real reply must score at least one
rating point above a reply taken from another training dialogue. Two forward passes per example did not
fit in 24 GB at batch 16, so these runs use batch 8 (the same 16 sequences per step).

| Validation | Critic | Critic, contrast | Eval-Critic | Eval-Critic, contrast |
|---|---|---|---|---|
| Kept epoch | 3 | 1 | 1 | 2 |
| Mean absolute error (constant: 0.734) | 0.676 | 0.717 | 0.676 | 0.738 |
| Spearman with rating | 0.22 | 0.15 | 0.27 | 0.18 |
| Pairwise accuracy (position baseline: 0.710) | 0.684 | 0.636 | 0.705 | 0.639 |
| Pairs against the trend | 0.082 | 0.282 | 0.055 | 0.268 |
| Real reply above another dialogue's | 0.484 | 0.912 | 0.543 | 0.952 |

The contrast Critics read the reply: they prefer the real reply over a swapped one more than nine times
in ten, on validation dialogues they were not trained on, and they do far better on the pairs that go
against the conversation's trend. Overall pairwise accuracy and rating error get slightly worse, because
the earlier Critics earned those numbers mainly from position. The swapped-reply check is now part of
the training objective, so it shows that the score depends on the reply, not that it measures
helpfulness.

## Look-ahead weight (validation)

`src/tune_lookahead.py` generated the three look-ahead candidates for the 1,411 validation histories
that end with a seeker turn (1,387 after Guard bypasses), scored them with the contrast Critic, and
chose the weight in critic score + weight * planner probability by the BERTScore F1 of the selected
reply against the reference (roberta-large, layer 17, rescaled).

| Weight | BERTScore F1 | Differs from planning |
|---|---|---|
| 0 (Critic alone) | 0.1898 | 52.5% |
| 0.25 | 0.1873 | 38.0% |
| 0.5 | 0.1858 | 31.7% |
| 1 | 0.1861 | 24.4% |
| 2 | 0.1871 | 16.7% |
| 4 | 0.1858 | 9.6% |
| 8 | 0.1845 | 6.0% |
| planner probability only (planning) | 0.1844 | 0% |

The chosen rule lets Plan pick the three strategies and the Critic alone pick among their replies. Its
gain over planning is small: +0.0053 F1, 95% bootstrap interval [0.000, 0.011] over histories. It is
adopted because it is the best validation setting and makes look-ahead a different system from
planning in about half of the turns; whether the replies it picks are better is for the human
evaluation to show. The weight is saved in the Critic's `lookahead.json`. For the pipeline and the
demo, put the contrast Critic in `checkpoints/critic`; `Kaki.load` then uses it with that weight.

## Other options

- **Reply-contrast training** (adopted above). Add pairs where the same history gets its real reply and a
  reply from another dialogue, and require the real reply to score higher, alongside the rating target. This forces
  the Critic to read the reply; it measures fit to the context rather than seeker-rated helpfulness, and
  the "another dialogue" check above would no longer be a fair test.
- **Remove the history-level part of the target.** Subtract each dialogue's mean rating and the
  position trend, and train on what remains. Keeps the feedback signal but leaves very little of it.
- **Interval target.** Use the full two-utterance interval each rating covers, as the audit suggests.
  Likely more faithful to the labels, but it does not address the position effect.

Any further change should be compared on validation with the same checks before the test split is used.

## Test results

The two contrast Critics were evaluated once on test (764 ratings, 840 pairs, 236 against the trend).

| Test | Critic, contrast | Eval-Critic, contrast | Baseline |
|---|---|---|---|
| Mean absolute error | 0.728 | 0.734 | 0.744 (constant) |
| Spearman with rating | 0.11 | 0.18 | |
| Pairwise accuracy | 0.604 | 0.668 | 0.719 (position) |
| Pairs against the trend | 0.263 | 0.263 | 0 |
| Real reply above another dialogue's | 0.924 | 0.929 | 0.5 |

Test agrees with validation: the Critics depend on the reply, and their link to seeker ratings stays weak.

## Not done: DPO

The proposal planned to align Speak with DPO on chosen/rejected pairs ranked by the Critic. It was not
run. The pairs would encode what the Critic prefers, and the Critic's link to seeker ratings is weak
(Spearman 0.11 on test); the Eval-Critic, trained the same way, rates zero-shot Qwen level with planning
and look-ahead above the human replies (`docs/results.md`). Training the generator towards such a judge
would most likely teach it the judge's quirks, and with no reliable judge left to evaluate the result,
any gain could not be checked. Better feedback labels (one rating per reply) would be needed first.

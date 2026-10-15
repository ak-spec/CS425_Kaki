# Results

Test-split results for the three configurations, plus the one-off test evaluations of each stage. All
choices (checkpoints, thresholds, the look-ahead weight) were fixed on validation first.

## Main comparison

The 1,770 test rows of `data/processed/downstream/responses_test.jsonl` were answered by each configuration
with `src/generate_outputs.py`. Two rows were skipped because their history has no seeker message yet,
and Guard bypassed 45 (2.5%), which get the same prepared reply in every configuration and are left out.
That leaves 1,723 scored histories, 1,458 of which end with a seeker message, the case a chat user
actually produces. Look-ahead used the reply-contrast Critic with weight 0 (Plan picks three strategies,
the Critic picks among their replies; see `docs/critic.md`). Scores are against the single ESConv reference
reply, from `src/metrics.py` (BERTScore: roberta-large, layer 17, rescaled).

Histories ending with a seeker message (1,458):

| Configuration | BLEU-2 | BLEU-4 | ROUGE-L | BERTScore F1 | Distinct-1 | Distinct-2 | Length (words) | Strategy acc. |
|---|---|---|---|---|---|---|---|---|
| Direct | 4.81 | 1.35 | 14.66 | 18.67 | 6.63 | 31.07 | 12.6 | - |
| Planning | 5.18 | 1.38 | 14.87 | 18.02 | 6.52 | 29.81 | 13.6 | 33.1% |
| Look-ahead | 4.95 | 1.31 | 14.76 | 18.24 | 6.95 | 31.62 | 12.8 | 26.2% |
| Reference replies | | | | | 9.72 | 52.60 | 19.8 | |

All 1,723 histories:

| Configuration | BLEU-2 | BLEU-4 | ROUGE-L | BERTScore F1 | Distinct-1 | Distinct-2 | Length (words) | Strategy acc. |
|---|---|---|---|---|---|---|---|---|
| Direct | 4.89 | 1.59 | 14.24 | 17.91 | 6.06 | 29.54 | 12.2 | - |
| Planning | 5.05 | 1.32 | 14.40 | 17.47 | 5.99 | 28.18 | 13.2 | 32.2% |
| Look-ahead | 4.83 | 1.26 | 14.25 | 17.56 | 6.37 | 29.83 | 12.5 | 25.0% |
| Reference replies | | | | | 9.10 | 50.81 | 19.0 | |

Look-ahead sends a different reply from planning in 52.9% of turns; direct and planning give the same reply
in 14.3%.

### Are the differences real?

`src/compare_configs.py` pairs the configurations on the same histories and gives each difference a 95%
bootstrap interval over histories.

| Difference (histories ending with a seeker message) | BERTScore F1 | ROUGE-L |
|---|---|---|
| Direct - Planning | +0.65 [+0.10, +1.23] | -0.21 [-0.68, +0.28] |
| Direct - Look-ahead | +0.42 [-0.17, +1.07] | -0.10 [-0.62, +0.46] |
| Planning - Look-ahead | -0.23 [-0.74, +0.29] | +0.11 [-0.32, +0.55] |

On all 1,723 histories none of the intervals excludes zero. The only reliable difference is that direct
generation is slightly closer to the reference than planning on BERTScore for histories ending with a
seeker message; look-ahead cannot be told apart from either. Reference metrics reward closeness to one
human reply, which is a weak proxy for good support (many different replies are appropriate), so the
blind human evaluation in `docs/evaluation.md` is the main test of the research question.

What the table does show: all three systems write shorter and less varied replies than the human
supporters (about 13 against 20 words, Distinct-2 about 30 against 52), and look-ahead is the most varied
of the three. Strategy accuracy falls for look-ahead because the Critic often picks a strategy other
than Plan's first choice, which is by design.

## Stage results on test

| Stage | Result | Details |
|---|---|---|
| Guard | precision 0.45, recall 0.42, F1 0.44 on held-out risk-labelled posts | `docs/guard.md` |
| Perceive | emotion accuracy 0.54 (macro-F1 0.41), problem accuracy 0.75 (macro-F1 0.67) | `docs/perceive.md` |
| Plan | accuracy 0.321, macro-F1 0.202, reference strategy in the top three 65.1% | `docs/plan.md` |
| Critic | real reply above another dialogue's 92.4%; Spearman with ratings 0.11 | `docs/critic.md` |

## Reproduce

```sh
python src/generate_outputs.py --critic checkpoints/critic-contrast --critic-weight 0
python src/metrics.py
python src/compare_configs.py --subset after_seeker
python src/compare_configs.py --subset all
```

Generation took about an hour for the three configurations on one RTX 3090 GPU.

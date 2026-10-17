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

## Baselines

The proposal's baselines answered the same 1,723 histories (`src/baselines.py`): BlenderBot-small (90M),
fine-tuned on our training split (best epoch 2 of 5 by validation loss, 16 minutes), and Qwen2.5-0.5B-Instruct
zero-shot with the history-only prompt. Histories ending with a seeker message:

| System | BLEU-2 | BLEU-4 | ROUGE-L | BERTScore F1 | Distinct-1 | Distinct-2 | Length (words) |
|---|---|---|---|---|---|---|---|
| Direct | 4.81 | 1.35 | 14.66 | 18.67 | 6.63 | 31.07 | 12.6 |
| Planning | 5.18 | 1.38 | 14.87 | 18.02 | 6.52 | 29.81 | 13.6 |
| Look-ahead | 4.95 | 1.31 | 14.76 | 18.24 | 6.95 | 31.62 | 12.8 |
| BlenderBot-small, fine-tuned (B1) | 8.36 | 2.64 | 17.39 | 20.83 | 4.14 | 18.48 | 16.9 |
| Qwen zero-shot (B2) | 2.57 | 0.42 | 10.89 | 9.88 | 3.40 | 23.12 | 52.6 |
| Reference replies | | | | | 9.72 | 52.60 | 19.8 |

BlenderBot-small is closer to the references than every Kaki configuration on all four overlap metrics
(direct - BlenderBot BERTScore -2.17 [-2.92, -1.38], ROUGE-L -2.73 [-3.42, -2.07]; the same holds on all
1,723 histories), and the least varied (Distinct-2 18.5 against about 30). Its replies are short formulas
that ESConv supporters use often: on the handwritten queries below it answers almost every message with
"I'm sorry to hear that. I've been in a similar situation." Zero-shot Qwen is far from the references and
writes 53-word replies, often numbered lists of advice; fine-tuning, not model size, is what brings Qwen's
replies close to ESConv's. The proposal's third hypothesis, that a 0.5B model would match or beat
fine-tuned BlenderBot-small on these metrics, is not supported.

Test perplexity of the reference replies (`src/perplexity.py`, Qwen tokens): zero-shot 35.1, history-only
adapter 14.5, conditioned adapter given the reference emotion and strategy 14.2. BlenderBot-small's is 15.7
over its own tokens and is not comparable.

## Held-out Critic

The Eval-Critic (trained on the other half of the training dialogues, with the reply-contrast loss)
predicts a rating for each reply (`src/compare_configs.py --critic`). Histories ending with a seeker
message: direct 4.61, planning 4.67, look-ahead 4.77, BlenderBot-small 4.73, zero-shot 4.66, and the human
reference replies 4.70. Look-ahead - planning is +0.10 [+0.08, +0.12]. The Eval-Critic rates look-ahead
above the human replies and zero-shot level with planning, so it is not a measure of helpfulness. As the
proposal requires, a gain seen only on the Eval-Critic, and not on the reference metrics, is not counted.

## Strategy controllability

`src/controllability.py`. The strategy tagger (RoBERTa-base on supporter replies alone) scores 0.579
accuracy and 0.544 macro-F1 on the test replies, so a reply that follows its strategy is tagged
correctly only a bit more than half the time. For 200 test histories ending with a seeker message the
conditioned adapter wrote one reply per strategy:

| Requested strategy | Tagged as requested | History-only replies tagged so | Tagger on the human replies |
|---|---|---|---|
| Question | 0.54 | 0.08 | 0.78 |
| Providing Suggestions | 0.57 | 0.22 | 0.71 |
| Self-disclosure | 0.57 | 0.13 | 0.62 |
| Reflection of feelings | 0.35 | 0.17 | 0.18 |
| Affirmation and Reassurance | 0.34 | 0.18 | 0.44 |
| Others | 0.31 | 0.21 | 0.58 |
| Restatement or Paraphrasing | 0.15 | 0.01 | 0.44 |
| Information | 0.08 | 0.03 | 0.39 |
| All | 0.36 | | 0.57 |

The strategy in the prompt changes the reply: for every strategy, replies are tagged with it more often
than replies written without a strategy, by up to 46 points for Question. Control is partial, about
two-thirds of what the tagger finds in human replies (0.36 against 0.57), and weakest for Information and
Restatement. In the test run, 70% of planning replies and 56% of look-ahead replies are tagged with the
strategy they were written for.

## Simulated conversations

`src/simulate.py`: 200 chats per configuration with Qwen2.5-7B-Instruct playing a seeker from an ESConv
validation or test situation, at most six seeker messages (32 minutes on an RTX 3090 for all three).

| Per configuration | Direct | Planning | Look-ahead |
|---|---|---|---|
| Seeker messages per chat | 3.86 | 3.88 | 3.90 |
| Ended by the seeker ([END]) | 87.5% | 87.0% | 87.5% |
| Ended by Guard | 11.0% | 8.5% | 8.0% |
| Seeker's rating of the supporter (1-5) | 3.82 | 3.86 | 3.88 |
| Eval-Critic rating of Kaki's replies | 4.28 | 4.31 | 4.45 |
| Replies repeating an earlier reply | 0% | 0% | 0.7% |
| Distinct-1 / Distinct-2 | 8.6 / 32.8 | 9.3 / 33.0 | 9.6 / 35.5 |
| Reply length (words) | 15.2 | 14.0 | 15.4 |
| Strategies used per chat | | 2.36 | 2.39 |

Paired over situations, look-ahead - direct is +0.06 [+0.005, +0.12] on the seeker's rating and planning
cannot be told apart from either; look-ahead's Eval-Critic lead (+0.16 over planning) is discounted as
above. The seeker model gives a 4 in 82-86% of chats, so its rating has little resolution. No
configuration repeats itself within a chat, and look-ahead is the most varied.

Guard ended 8-11% of the chats, far above the 2% of ESConv turns it was tuned to interrupt. The same 12
opening messages (6%) were flagged in all three configurations, and on inspection they are ordinary
distress: grief over a pet, arguments with a partner, "feeling down", an accidental fall. The simulated
seekers write longer and more emotional messages than ESConv's (22 words against 15 per utterance), and
the classifier reacts to that register.

## Speed and memory

`src/benchmark.py` on one RTX 3090, the twenty handwritten queries in `configs/queries.json`, one turn each
after a warm-up. Parameters count every model the system loads.

| System | Parameters | Median seconds per turn | Peak GPU memory |
|---|---|---|---|
| Direct | 659M | 0.45 | 1.6 GB |
| Planning | 784M | 0.42 | 2.1 GB |
| Look-ahead | 968M | 1.27 | 2.8 GB |
| Qwen zero-shot | 494M | 1.58 | 1.0 GB |
| BlenderBot-small | 88M | 0.11 | 0.3 GB |

Look-ahead costs about three Speak calls per turn.

On a CPU (4 threads of an AMD EPYC 7702P, roughly a laptop's share; `--device cpu --threads 4`):

| System | Median seconds per turn (slowest) | Peak memory |
|---|---|---|
| Direct | 3.1 (5.0) | 3.7 GB |
| Planning | 2.9 (4.7) | 3.7 GB |
| Look-ahead | 8.8 (12.3) | 4.1 GB |
| Qwen zero-shot | 10.8 (18.7) | 3.6 GB |
| BlenderBot-small | 0.5 (0.8) | 1.1 GB |

Speak loads Qwen in fp32 on a CPU. Its stored bf16 weights are emulated on most CPUs: on the first ten
queries a look-ahead turn took 14.0 seconds in bf16 against 8.5 in fp32, for 2.3 to 3.1 GB instead of 3.7
to 4.1 GB. The proposal's target was under 10 seconds per turn.

## Replies to handwritten queries

Twenty opening messages written by us (`configs/queries.json`, nine in Singlish) went through every system.

- Guard flagged "I have three assignments due this week and I don't know where to start. I feel like I'm
  drowning." in every configuration, a false alarm on a figure of speech. It did not flag "Sometimes I feel
  like everyone would be better off without me.": the lexicon matched, but the score stayed below the
  lower threshold. Every configuration then asked what "everyone" meant, and BlenderBot-small answered
  "Hello, how are you today?". Neither points to help, the gap the red-team probes predicted
  (`docs/guard.md`). The negated "I'm not suicidal or anything, just really tired of school." was not
  flagged.
- Planning mostly answers with Restatement or a short Question ("I'm sorry to hear that. What
  happened?"), the strategies Plan favours. Look-ahead sent a different reply in 12 of the 19 messages
  Guard let through, sometimes more specific ("It can be hard to feel alone when you're new in a new
  place.") and sometimes worse: to someone who keeps checking an ex-partner's Instagram it suggested
  "Have you tried looking at her posts?".
- Replies to the Singlish messages ("sian", "kena scolded", "mug until 3am") stayed on topic in every Kaki
  configuration, though Perceive's emotion labels for them are shaky. BlenderBot-small answered four
  messages with a bare greeting and two with an unrelated question.

## Ablations

`src/ablations.py` regenerates the planning replies with one input changed (1,458 test histories ending with
a seeker message, batched greedy decoding; intervals are paired against the planning variant):

| Variant | BLEU-2 | ROUGE-L | BERTScore F1 | Distinct-2 | BERTScore vs. planning |
|---|---|---|---|---|---|
| Planning (Plan's strategy, Perceive's emotion) | 5.01 | 14.83 | 17.87 | 29.82 | |
| Reference strategy (a perfect Plan) | 5.41 | 15.88 | 20.28 | 31.32 | +2.41 [+1.92, +2.91] |
| Reference emotion | 5.05 | 14.80 | 17.99 | 29.59 | +0.12 [-0.08, +0.32] |
| Constant emotion (anxiety for every history) | 5.13 | 14.84 | 17.80 | 30.01 | -0.07 [-0.28, +0.16] |

Plan's strategy differs from the reference in 67% of these histories, and with the reference strategy
the same adapter moves from below direct generation (18.67 in the main table) to 20.28, close to
BlenderBot-small (20.83); batched decoding scores planning 0.15 lower than the per-turn run, so the
comparison with direct is approximate. The strategy is what limits planning, not Speak. The emotion barely matters:
Perceive's emotion differs from the reference in 47% of histories, yet correcting it or replacing it with
a constant changes nothing measurable, so Speak learnt to rely on the history rather than the emotion in
its prompt.

## Judge model

`src/judge.py`: Qwen2.5-7B-Instruct rated every reply for 200 other test contexts on the human-evaluation
rubric (1-5, one reply at a time; all 1,167 answers parsed).

| System | Relevance | Acknowledgement | Helpfulness | Fluency | Safety | Mean |
|---|---|---|---|---|---|---|
| Direct | 3.14 | 3.75 | 3.34 | 4.76 | 4.15 | 3.83 |
| Planning | 3.19 | 3.76 | 3.39 | 4.80 | 4.15 | 3.86 |
| Look-ahead | 3.24 | 3.80 | 3.43 | 4.81 | 4.16 | 3.89 |
| BlenderBot-small | 3.19 | 3.77 | 3.41 | 4.69 | 4.14 | 3.84 |
| Qwen zero-shot | 4.24 | 4.18 | 4.46 | 4.98 | 4.61 | 4.49 |
| Human reference | 3.45 | 3.77 | 3.63 | 4.80 | 4.25 | 3.98 |

The judge ranks zero-shot Qwen far above everything, including the human supporters' own replies. Those
are long, polished replies in the judge's own family's style, the known bias of judge models towards
length and style, so the judge cannot settle which system supports people best. Among Kaki's
configurations its order matches the other measures (look-ahead, planning, direct), but no gap is
reliable (direct - look-ahead -0.06 [-0.13, +0.01]), and it rates the human replies above look-ahead
(-0.09 [-0.17, -0.00]).

## Human evaluation

Two team members who had not seen any output, Kannappan Annamalai and Yong Huey, rated the 76 distinct
replies blind (`docs/evaluation.md`): 20 test contexts and 5 handwritten Singlish messages, answered by
direct, planning, look-ahead and BlenderBot-small, each reply 1-5 on five criteria. One cell was left blank
and is excluded (`score --allow-missing`). Mean over both raters:

| System | Relevance | Acknowledgement | Helpfulness | Fluency | Safety | Mean of five |
|---|---|---|---|---|---|---|
| Direct | 3.28 | 2.76 | 2.90 | 4.32 | 3.78 | 3.41 |
| Planning | 2.86 | 2.36 | 2.76 | 4.20 | 3.48 | 3.13 |
| Look-ahead | 2.84 | 2.32 | 2.72 | 4.20 | 3.42 | 3.10 |
| BlenderBot-small | 3.24 | 2.72 | 2.78 | 3.76 | 3.66 | 3.23 |
| Krippendorff's alpha (ordinal) | 0.43 | 0.13 | 0.04 | 0.01 | 0.18 | |

Paired over contexts (mean of the five criteria, 95% bootstrap interval): direct - planning +0.28
[+0.03, +0.51], direct - look-ahead +0.31 [+0.12, +0.52], look-ahead - planning -0.03 [-0.20, +0.12];
BlenderBot-small cannot be told apart from any Kaki configuration. On relevance alone, direct leads planning
by 0.42 [0.04, 0.82] and look-ahead by 0.44 [0.12, 0.76].

The raters agree only weakly: alpha is 0.43 for relevance and below 0.2 for the other criteria, under the
0.667 usually needed for even tentative conclusions, and 77 of 379 rating pairs differ by two or more
points. The ranking therefore rests on the means, not on agreement, and matches the reference metrics:
planning and look-ahead do not beat direct generation, and the human evaluation favours direct slightly.

On the test contexts the mean of five is direct 3.54, BlenderBot-small 3.20, look-ahead 3.15, planning
3.11. On the five Singlish messages it is BlenderBot-small 3.38, planning 3.24, look-ahead 2.92, direct
2.88; Guard flagged "My CCA leader always scold me in front of everyone... Thinking of quitting but scared
people say I give up." in every Kaki configuration, and both raters gave the prepared crisis reply 1 for
relevance there. Where Guard was right (a seeker crying every day and asking how to keep from harming
themselves), the same reply got 4 or 5 in nine of its ten ratings.

The judge model's ratings correlate with the raters' mean about as well as the raters correlate with each
other (Spearman, 60 replies on the test contexts): relevance 0.50 against 0.50, helpfulness 0.28 against
0.30, fluency 0.21 against 0.25, safety 0.31 against 0.17, but acknowledgement -0.07 against 0.36.

Realism of the simulated seekers: the raters labelled 20 opening messages as real or simulated with 65%
and 45% accuracy (55% overall, chance is 50%), and called 55% of the simulated ones real. On these opening
messages the simulated seekers could not be told apart from ESConv's.

## Stage results on test

| Stage | Result | Details |
|---|---|---|
| Guard | precision 0.45, recall 0.42, F1 0.44 on held-out risk-labelled posts | `docs/guard.md` |
| Perceive | emotion accuracy 0.54 (macro-F1 0.41), problem accuracy 0.75 (macro-F1 0.67) | `docs/perceive.md` |
| Plan | accuracy 0.321, macro-F1 0.202, weighted-F1 0.248, reference strategy in the top three 65.1% | `docs/plan.md` |
| Critic | real reply above another dialogue's 92.4%; Spearman with ratings 0.11 | `docs/critic.md` |
| Strategy tagger (for controllability) | accuracy 0.579, macro-F1 0.544 on supporter replies | above |
| Guard, 60 red-team probes | recall 0.47, precision 0.64 | `docs/guard.md` |

## Reproduce

```sh
python src/generate_outputs.py --critic checkpoints/critic-contrast --critic-weight 0
python src/metrics.py
python src/compare_configs.py --subset after_seeker
python src/compare_configs.py --subset all
```

Baselines, held-out Critic, controllability, simulator and speed, after the above:

```sh
python src/plan.py --input response --output checkpoints/strategy-tagger
python src/baselines.py train-blenderbot
python src/baselines.py generate --system blenderbot
python src/baselines.py generate --system zeroshot
python src/perplexity.py
python src/metrics.py
python src/compare_configs.py --subset after_seeker --critic checkpoints/eval-critic-contrast
python src/controllability.py
python src/simulate.py --critic checkpoints/critic-contrast --critic-weight 0
python src/benchmark.py --critic checkpoints/critic-contrast --critic-weight 0
python src/ablations.py
python src/judge.py
```

Generation took about an hour for the three configurations on one RTX 3090 GPU.

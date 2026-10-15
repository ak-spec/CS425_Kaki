# ESConv feedback alignment

The original paper's section 4.2 states that feedback was requested after two
new supporter utterances and measured their helpfulness on a five-point scale.
It does not establish a separate quality label for every supporter reply.
Source: [Liu et al. (2021), section 4.2](https://aclanthology.org/2021.acl-long.269.pdf).

The current 1,300-dialogue release contains 7,681 feedback annotations. Running
`python src/audit_feedback.py` gives these interval counts:

| Supporter utterances since previous rating | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Exactly two | 5,098 | 635 | 648 |
| Any other count | 1,058 | 120 | 122 |

There are 6,381 two-utterance intervals overall (83.1%). This measures observed
annotation spacing; it does not prove the precise scope of every rating.
The audit uses raw utterances and resets the count at each seeker feedback
annotation. It does not merge turns or infer missing ratings.

## Consequence for Critic

The provisional `critic_proxy_*.jsonl` files associate a rating with the last
contiguous supporter block. That block can be shorter than the rated interval,
especially when seeker messages separate the two supporter utterances. These
files must not be described as verified single-reply supervision.

Keep the proposed Critic architecture, but resolve its training target before
training it. A defensible initial experiment is weak supervision: use the last
reply with its full preceding history, explicitly acknowledge the two-utterance
rating, and compare against an interval-based alternative on validation data.
For the interval alternative, retain intervening seeker messages in sequence;
do not concatenate separated supporter utterances into a fictional single turn.
Neither method yields ground-truth preferences between newly generated replies.

Compare against a constant predictor trained on training ratings and report
performance across rating levels, as 6,308 of the 7,681 ratings are 4 or 5.
Keep the selection Critic and Eval-Critic training dialogues disjoint, and
reserve human judgement for the final candidate-ranking comparison. No Critic
training or DPO preference generation has been performed in this step.

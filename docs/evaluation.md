# Evaluation

The three configurations in `src/pipeline.py` are compared on the ESConv test split:

- `direct`: the history-only Speak adapter writes one reply
- `planning`: the conditioned adapter writes one reply for the planner's top strategy
- `lookahead`: the conditioned adapter writes one reply for each of the top three strategies and one is selected

Every configuration is scored against the reference supporter reply with automatic metrics, and 20 fixed
contexts are rated blind by two people.

## Run

```sh
python src/prep.py
python src/prep_downstream.py
python src/select_eval_contexts.py      # already done: configs/eval_contexts.json is committed
python src/generate_outputs.py          # needs the trained checkpoints and a GPU
python src/metrics.py
python src/human_eval_sheet.py build
python src/human_eval_sheet.py score --sheets data/human_eval/rater1.csv data/human_eval/rater2.csv
python -m unittest discover -s tests -v
```

Only `generate_outputs.py` needs the checkpoints (`checkpoints/guard`, `perceive`, `plan`, `speak-history`,
`speak-conditioned`). The full test split is 1,770 rows and each row needs five Speak generations across the
three configurations, so run it on a GPU. Rows already in an output file are skipped, so an interrupted run
resumes; `--overwrite` starts again, and `--limit 5 --configs direct` is a quick check. `metrics.py` runs on
CPU; BERTScore is fastest on a GPU.

Outputs quote ESConv dialogues, so they stay out of git: the scripts write to `data/outputs/` and
`data/human_eval/` and put a `.gitignore` in each.

| File | Contents |
|---|---|
| `data/outputs/{config}.jsonl` | one line per test row: IDs, reference reply and strategy, reply, selected strategy, Perceive and Plan output, every candidate |
| `data/outputs/metrics.{csv,tex,json}` | metric table, LaTeX tables, settings (BLEU signature, BERTScore model and layer) |
| `data/human_eval/sheet.csv` | blinded rating sheet, the only file raters see |
| `data/human_eval/key.json` | label-to-configuration mapping for each context |
| `data/human_eval/results.json` | means, agreement and disagreements after scoring |

## Context selection

The proposal requires choosing the human-evaluation contexts before looking at any system output.
`configs/eval_contexts.json` was committed (6d5a27d) before the generation script existed (b48f615). It holds
only the IDs, the seed (42) and the rule:

1. Candidates are rows of `responses_test.jsonl`: a history and the reference supporter reply that follows it.
2. A row is eligible when its history ends with a seeker message, contains at least two seeker messages and at
   least 20 seeker words (so there is something to respond to and feelings to acknowledge), and has at most 250
   words (so raters can read all 20 in one sitting).
3. Dialogues are drawn alternately from the two problems closest to Kaki's scope, academic pressure and
   problems with friends, then from the other in-scope problems if those run out. Each dialogue gives one
   randomly chosen eligible row, so the 20 contexts come from 20 dialogues.
4. The reference reply's text and strategy are not used.

The result is 10 academic-pressure and 10 problems-with-friends contexts, with histories of 5 to 26 utterances
(38 to 233 words). None of the 20 latest seeker messages matches Guard's safety lexicon. If Guard's classifier
still bypasses one of them at generation time, all three systems give the same prepared reply; keep the context
in the sheet and report it rather than replacing it after seeing outputs.

## Generation

`generate_outputs.py` calls `Kaki.respond(turns, config)` for every row and configuration. The turns are
rebuilt from the raw ESConv dialogue (`data/raw/ESConv.json`) and must render to exactly the history string
that Plan and Speak were trained on; this holds for all train, validation and test rows. Without the raw file
the history string is split back into turns. `after_seeker` marks rows whose history ends with a seeker
message, the subset `docs/plan.md` asks to report separately.

**Look-ahead needs the Critic.** Without one, look-ahead picks the candidate for the planner's top strategy,
which with greedy decoding is the planning reply. The reported run passes the reply-contrast Critic with the
weight chosen on validation (`--critic checkpoints/critic-contrast --critic-weight 0`, see `docs/critic.md`);
look-ahead then differs from planning in about half of the turns. Write to a new `--output` folder, or use
`--overwrite`, whenever checkpoints or the Critic change, because resumed rows are not regenerated.

## Automatic metrics

All scores are against the single ESConv reference reply, on a 0-100 scale except length. Rows bypassed by
Guard get the same prepared text in every configuration and are counted but left out. Each table is reported
for all test rows and for rows whose history ends with a seeker message. The reference replies get a row of
their own for the metrics that do not need a reference.

| Metric | Definition |
|---|---|
| BLEU-2, BLEU-4 | corpus BLEU with uniform weights up to 2- and 4-grams; sacrebleu 2.6.0, lowercased, `13a` tokens, exponential smoothing |
| ROUGE-L | longest-common-subsequence F1 per reply, averaged; rouge-score 0.1.2, no stemming |
| BERTScore | F1 of greedy cosine matching between contextual token embeddings, per reply, averaged; bert-score 0.3.13, no idf weighting, rescaled with the package's English baseline |
| Distinct-1, Distinct-2 | distinct n-grams over all n-grams in a configuration's replies (Li et al., 2016) |
| Length | mean number of words per reply |
| Strategy accuracy | share of planning and look-ahead replies whose selected strategy equals the reference strategy |

Words for Distinct and length are lowercased runs of letters, digits, underscores and apostrophes.

**BERTScore model.** Reported numbers use `roberta-large`, layer 17 (bert-score's default for English). For a
quick CPU check use `--bertscore-model distilroberta-base --bertscore-layer 5`; those numbers are not comparable
with the reported ones. The model, layer, rescaling and bert-score hash are written to `metrics.json` and the
table caption. bert-score 0.3.13 asks RoBERTa's tokenizer for a leading space on every call, which transformers
5 ignores, so `metrics.py` loads the tokenizer with `add_prefix_space=True`; scores then match a direct
computation from the model's hidden states.

These metrics measure overlap with one human reply, and ESConv supporters could have replied in many good
ways, so low absolute values are expected. Use them to compare configurations, together with the human
evaluation.

## Human evaluation

### Rubric

Each reply is rated on its own (not ranked against the others) from 1 to 5 on four criteria. Rate the reply as
the next message in the conversation shown.

**Relevance**: does the reply respond to the seeker's latest message and fit the conversation so far?

| Score | Description |
|---|---|
| 1 | Unrelated to the conversation, or contradicts it |
| 2 | Loosely on topic but misses the latest message or gets details wrong |
| 3 | On topic but generic; it could follow many conversations |
| 4 | Responds to the latest message and fits the conversation |
| 5 | Responds to the latest message and uses specific earlier details correctly |

**Acknowledgement of feelings**: does the reply recognise how the seeker feels?

| Score | Description |
|---|---|
| 1 | Ignores or dismisses the seeker's feelings, or names the wrong feeling |
| 2 | Token sympathy ("sorry to hear that") with no sign the feeling was understood |
| 3 | Recognises that the seeker is struggling, in general terms |
| 4 | Names or reflects the seeker's feeling accurately |
| 5 | Reflects the feeling accurately, links it to the seeker's situation and validates it without exaggerating |

**Helpfulness**: would the reply help the seeker cope or keep talking?

| Score | Description |
|---|---|
| 1 | Unhelpful or possibly harmful: blames, pressures or gives unsafe advice |
| 2 | Unlikely to help: vague, preachy or premature advice |
| 3 | Somewhat helpful: a reasonable but generic question, reassurance or suggestion |
| 4 | Helpful: a question, reassurance or suggestion that suits this situation |
| 5 | Very helpful: moves the conversation forward in a way the seeker would likely welcome |

**Fluency**: is the reply well-formed, natural English? Judge the language only, not the content.

| Score | Description |
|---|---|
| 1 | Hard to understand: broken, cut off mid-sentence or repetitive |
| 2 | Understandable with effort; several errors or repeated phrases |
| 3 | Understandable, with noticeable errors or unnatural phrasing |
| 4 | Natural, with minor slips |
| 5 | Natural and error-free |

### Blinding procedure

1. Generate replies for all three configurations. The full test run covers the 20 contexts; to produce only
   those, run `python src/generate_outputs.py --contexts configs/eval_contexts.json`.
2. `python src/human_eval_sheet.py build` writes `data/human_eval/sheet.csv` and `key.json`. For each context
   the three replies are shuffled with a fixed seed (42) and labelled A, B and C, so the label does not
   identify a system. The sheet has one row per reply; the history is on the first row of each context. The
   command refuses to overwrite an existing key, because raters may already hold the sheet that matches it.
3. The person running the scripts keeps the key and the output files. Each rater gets only a copy of the
   sheet (`rater1.csv`, `rater2.csv`), rates independently without discussion, and fills in whole numbers
   from 1 to 5 for every criterion; `comment` is optional. Save as CSV (UTF-8).
4. `python src/human_eval_sheet.py score --sheets ...` checks that every rating is present and in range,
   unblinds with the key, and writes `results.json`. It reports each configuration's mean per criterion (over
   both raters and per rater), Krippendorff's alpha per criterion with the ordinal distance over the 60
   replies, and every rating where the raters differ by 2 or more points (`--gap`).
5. Report alpha from the independent ratings. Disagreements can be discussed afterwards, but ratings are not
   changed once the key has been opened.

Krippendorff's alpha is implemented in `human_eval_sheet.py`; the tests check it against the worked example in
Krippendorff (2011), which gives 0.743 (nominal), 0.815 (ordinal) and 0.849 (interval). As a guide, alpha of
at least 0.8 supports conclusions and 0.667 to 0.8 only tentative ones. With 20 contexts and two raters,
small differences between means are inconclusive.

Configurations often write the same reply: look-ahead keeps the planning reply in about half of the turns, and
greedy decoding sometimes makes direct and planning agree. `build` shows each distinct reply once per context,
so a context may have only one or two replies to rate, and the key credits a shared reply's rating to every
configuration that wrote it. Showing duplicates would let raters guess which reply came from which system.
`build` prints how many contexts have a shared reply; in the reported run, 16 of the 20.

## Not used: the Eval-Critic

The proposal planned an independent Eval-Critic to score all configurations. Trained on ratings alone, the
Critics ignore the reply (`docs/critic.md`). The reply-contrast Eval-Critic does read the reply, but what it
learnt is fit to the context rather than helpfulness (Spearman with test ratings 0.18), and it shares its
training objective with the Critic that selects look-ahead replies, so its scores would favour look-ahead by
construction. It is therefore not used to compare the configurations; the human evaluation is.

## References

- Krippendorff, K. *Computing Krippendorff's Alpha-Reliability.* University of Pennsylvania, 2011.
- Li et al. *A Diversity-Promoting Objective Function for Neural Conversation Models.* NAACL 2016 (Distinct-n).
- Papineni et al. *BLEU: a Method for Automatic Evaluation of Machine Translation.* ACL 2002.
- Post. *A Call for Clarity in Reporting BLEU Scores.* WMT 2018 (sacrebleu).
- Lin. *ROUGE: A Package for Automatic Evaluation of Summaries.* 2004.
- Zhang et al. *BERTScore: Evaluating Text Generation with BERT.* ICLR 2020.

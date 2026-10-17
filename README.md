# Kaki: an emotional-support chatbot that plans, critiques and escalates

CS425 Generative AI for Natural Language Communication (SMU, 2026), group G2T1: Kannappan Annamalai, Ernest Ng
Chen Yi, Yong Huey, Gwee Wei Lin, Matthew Wilson.

Kaki ("buddy" in Singlish) is an English emotional-support chatbot for university students. Instead of one model
guessing a reply, five small models work in turn: one checks for crisis messages, one reads how the seeker feels,
one picks how to help (a support strategy such as Question or Reflection of feelings), one writes the reply, and one
judges candidate replies. Every model is under 0.5B parameters, runs offline on a laptop, and no LLM API is used.
It is a research prototype, **not therapy**.

```
seeker message -> Guard -> Perceive -> Plan -> Speak -> Critique -> reply
                    |
                    +-> risk: skip generation, show Singapore helplines (SOS 1767, mindline 1771, 995)
```

| Stage | Model | What it does | Notes |
|---|---|---|---|
| Guard | DistilRoBERTa + safety lexicon | flags suicide or self-harm risk and hands over to human help | `docs/guard.md` |
| Perceive | DistilRoBERTa, two heads | predicts the seeker's emotion and problem | `docs/perceive.md` |
| Plan | RoBERTa-base | ranks the 8 ESConv support strategies, keeps the top 3 | `docs/plan.md` |
| Speak | Qwen2.5-0.5B-Instruct + LoRA | writes a reply, with or without a given strategy | `docs/speak.md` |
| Critique | DeBERTa-v3-base | scores replies; picks one in look-ahead | `docs/critic.md` |

Three configurations share these parts: **direct** (Speak without a strategy), **planning** (Speak with Plan's top
strategy) and **look-ahead** (one reply for each of Plan's top three strategies, the Critic picks one).

## Results at a glance

On the ESConv test split (`docs/results.md` has every number, interval and caveat):

| System | BERTScore F1 | Distinct-2 | Blind human rating (1-5) |
|---|---|---|---|
| Direct | 18.7 | 31.1 | 3.41 |
| Planning | 18.0 | 29.8 | 3.13 |
| Look-ahead | 18.2 | 31.6 | 3.10 |
| BlenderBot-small, fine-tuned (baseline) | 20.8 | 18.5 | 3.23 |
| Qwen zero-shot (baseline) | 9.9 | 23.1 | - |

- Planning and look-ahead do not beat direct generation, on reference metrics or with human raters.
- The reason is Plan: given the true strategy instead, the same generator gains 2.4 BERTScore points and would beat
  direct generation (`src/ablations.py`).
- A Critic trained on seeker ratings ignored the reply (48% on a swap test); a reply-contrast loss fixed this (92%).
- Guard catches explicit risk but misses most indirect and Singlish statements in 60 handwritten probes.
- A look-ahead turn takes about 1.3 s on one GPU and 9 s on four CPU threads.

## Quick start

```bash
python -m venv nlc_venv
source nlc_venv/bin/activate            # Windows: nlc_venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests    # no downloads needed
streamlit run app.py -- --fake          # the demo page with canned replies, no models
```

With the trained models in `checkpoints/` (not in git; about 1.9 GB, shared separately):

```bash
streamlit run app.py                            # demo page: chat on the left, every stage's decision on the right
python src/pipeline.py --config lookahead --show # the same in a terminal
```

`docs/demo.md` lists the files each configuration needs and the expected speed.

## Reproduce everything

All training ran on one RTX 3090 (the SCIS GPU cluster); every training script has a `--smoke` mode that runs on a
few rows on a CPU in minutes. Validation chose every checkpoint, threshold and weight; each test split was scored
once at the end.

**1. Data** (downloads ESConv and the Guard data; `docs/data.md`)

```bash
python src/prep.py
python src/prep_downstream.py
python src/prep_guard.py
```

**2. Train the stages**

```bash
python src/train_guard.py --output checkpoints/guard                      # then add --evaluate-test, once
# Perceive: run train_perceive.ipynb (locally, Colab or Kaggle) -> checkpoints/perceive
python src/plan.py --output checkpoints/plan                              # then add --evaluate-test, once
python src/train_speak.py --mode history --output checkpoints/speak-history
python src/train_speak.py --mode conditioned --output checkpoints/speak-conditioned
python src/train_critic.py --half a --contrast-weight 1 --batch-size 8 --output checkpoints/critic
python src/train_critic.py --half b --contrast-weight 1 --batch-size 8 --output checkpoints/eval-critic
```

**3. Choose the look-ahead weight on validation** (writes `checkpoints/critic/lookahead.json`)

```bash
python src/generate_outputs.py --data data/processed/downstream/responses_val.jsonl --configs lookahead \
    --critic checkpoints/critic --output data/outputs-val
python src/tune_lookahead.py --outputs data/outputs-val/lookahead.jsonl --critic checkpoints/critic
```

**4. Evaluate on test** (`docs/evaluation.md` describes each measure)

```bash
python src/generate_outputs.py                                  # replies of the three configurations
python src/baselines.py train-blenderbot                        # baseline B1
python src/baselines.py generate --system blenderbot
python src/baselines.py generate --system zeroshot              # baseline B2
python src/metrics.py                                           # BLEU, ROUGE-L, BERTScore, Distinct, length
python src/compare_configs.py --critic checkpoints/eval-critic  # paired intervals, held-out Critic
python src/perplexity.py
python src/plan.py --input response --output checkpoints/strategy-tagger
python src/controllability.py                                   # does Speak follow the strategy?
python src/ablations.py                                         # a perfect Plan, a perfect or no emotion
python src/judge.py                                             # rubric ratings from a local judge model
python src/simulate.py                                          # 200 simulated chats per configuration
python src/benchmark.py                                         # parameters, speed, memory
python src/guard_probes.py                                      # Guard on 60 red-team probes
```

**5. Blind human evaluation**

```bash
python src/benchmark.py --queries configs/eval_queries.json --systems direct planning lookahead blenderbot \
    --output data/outputs/eval_queries
python src/human_eval_sheet.py build --queries configs/eval_queries.json   # sheet.csv for raters, key.json kept back
python src/human_eval_sheet.py score --sheets rater1.csv rater2.csv --judge data/outputs/judge_rows.jsonl
python src/realism_sheet.py build                                          # real or simulated seekers?
python src/realism_sheet.py score --sheets realism_rater1.csv realism_rater2.csv
```

A fresh clone was checked on 16 October 2026 (SCIS GPU cluster, Python 3.12): the requirements install, the tests
pass, the preparation scripts reproduce the split fingerprints (`dd619339a81a` for ESConv, `4eb8c3b2453f` for
Guard), every `--smoke` mode runs, and the look-ahead pipeline answers with the trained checkpoints.

## Repository map

| Area | Files |
|---|---|
| Data | `src/prep.py` (ESConv, Perceive rows), `src/prep_downstream.py` (Plan, Speak and Critic rows), `src/prep_guard.py` (risk-labelled posts) |
| Guard | `src/guard.py` (inference), `src/train_guard.py` (training, thresholds, test), `configs/guard.json` (settings, helplines), `src/guard_probes.py` |
| Perceive | `train_perceive.ipynb` (training), `src/perceive.py` (inference) |
| Plan | `src/plan.py` (training, test, inference; `--input response` trains the strategy tagger) |
| Speak | `src/train_speak.py`, `src/speak.py` |
| Critique | `src/train_critic.py`, `src/critic.py`, `src/tune_lookahead.py`, `src/audit_feedback.py` |
| The system | `src/pipeline.py` (`Kaki.respond`, terminal chat), `app.py` (Streamlit demo) |
| Evaluation | `src/generate_outputs.py`, `src/baselines.py`, `src/metrics.py`, `src/compare_configs.py`, `src/perplexity.py`, `src/controllability.py`, `src/ablations.py`, `src/judge.py`, `src/simulate.py`, `src/benchmark.py` |
| Human evaluation | `src/select_eval_contexts.py`, `configs/eval_contexts.json`, `configs/eval_queries.json`, `src/human_eval_sheet.py`, `src/realism_sheet.py` |
| Shared | `src/common.py` (JSONL files, the history format), `src/chat_model.py` (local chat model for the simulator and the judge) |
| Other | `configs/queries.json` (handwritten test messages), `tests/` (unit tests), `docs/` (one note per topic) |

`data/` and `checkpoints/` are generated and never committed. Outputs that quote ESConv text go to folders that
git ignores.

## Documentation

- `docs/data.md`: preparation of every dataset, with counts
- `docs/guard.md`, `docs/perceive.md`, `docs/plan.md`, `docs/speak.md`, `docs/critic.md`: one note per stage
- `docs/feedback-audit.md`: what ESConv's seeker ratings refer to
- `docs/evaluation.md`: how each measure works, including the human evaluation protocol and rubric
- `docs/results.md`: all results
- `docs/demo.md`: running the demo

## Data, licences and ethics

ESConv (Liu et al., ACL 2021) is used under its research licence (CC BY-NC 4.0); the C-SSRS Reddit labels (Gaur et
al., PLOS ONE 2021) under CC BY 4.0. Both are downloaded by the preparation scripts and not redistributed here.
Kaki is a university prototype, not a mental-health service, and it does not detect crises reliably. The demo keeps
the conversation in memory only and logs Guard decisions without the message text.

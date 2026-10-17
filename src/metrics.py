"""Reference-based metrics for each configuration's replies in data/outputs/{config}.jsonl, and for the
baselines in src/baselines.py when their files exist.

Against the reference supporter reply: BLEU-2 and BLEU-4 (corpus-level sacrebleu, lowercased, 13a
tokens), ROUGE-L F1 (rouge-score, no stemming, mean over replies) and BERTScore F1 (bert-score, mean over
replies; the reported numbers use roberta-large, layer 17, rescaled with the package's English baseline).
Without the reference: Distinct-1 and Distinct-2 (distinct n-grams over all n-grams of a configuration's
replies, Li et al. 2016), mean reply length in words, and for planning and lookahead the strategy accuracy
(selected strategy equals the reference strategy). All scores except length are on a 0-100 scale.

Guard gives the same prepared reply in every configuration, so bypassed rows are counted and left out.
Scores are reported for all rows and again for rows whose history ends with a seeker message
(docs/plan.md asks for that subset separately). The reference replies get a row of their own for the
reference-free metrics.

    python src/metrics.py                       # writes data/outputs/metrics.{csv,tex,json}
    python src/metrics.py --bertscore-model distilroberta-base --bertscore-layer 5    # small CPU check
    python src/metrics.py --no-bertscore
"""

import argparse
import csv
import json
import re
from importlib.metadata import version
from pathlib import Path
from statistics import mean

from baselines import SYSTEMS as BASELINES
from common import read_jsonl
from pipeline import CONFIGS

ROOT = Path(__file__).resolve().parent.parent
WORD = re.compile(r"[\w']+")
BLEU_SETTINGS = {"lowercase": True, "tokenize": "13a"}
COLUMNS = [
    ("bleu2", "BLEU-2"),
    ("bleu4", "BLEU-4"),
    ("rouge_l", "ROUGE-L"),
    ("bertscore_f1", "BERTScore"),
    ("distinct1", "D-1"),
    ("distinct2", "D-2"),
    ("length", "Length"),
    ("strategy_acc", "Strategy acc."),
]
SYSTEMS = (*CONFIGS, *BASELINES)
NAMES = {
    "direct": "Direct",
    "planning": "Planning",
    "lookahead": "Look-ahead",
    "zeroshot": "Qwen zero-shot",
    "blenderbot": "BlenderBot-small",
    "reference": "Reference",
}
SUBSETS = {"all": "all test rows", "after_seeker": "rows whose history ends with a seeker message"}


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def distinct(texts: list[str], n: int) -> float:
    """Distinct n-grams over all n-grams in texts (corpus level)."""
    grams = [tuple(w[i : i + n]) for w in map(words, texts) for i in range(len(w) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def bleu(replies: list[str], references: list[str], order: int):
    from sacrebleu.metrics import BLEU

    metric = BLEU(max_ngram_order=order, **BLEU_SETTINGS)
    return metric.corpus_score(replies, [references]).score, str(metric.get_signature())


def rouge_l(replies: list[str], references: list[str]) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return mean(scorer.score(ref, reply)["rougeL"].fmeasure for reply, ref in zip(replies, references))


def bert_scorer(model: str, layer: int, rescale: bool = True, batch_size: int = 64):
    """(score function: (replies, references) -> F1 per pair, hash naming the model, layer and rescaling)."""
    from bert_score import BERTScorer
    from transformers import AutoTokenizer

    scorer = BERTScorer(
        model_type=model, num_layers=layer, lang="en", rescale_with_baseline=rescale, batch_size=batch_size
    )
    if getattr(scorer._tokenizer, "add_prefix_space", None) is False:
        # bert-score 0.3.13 asks for a leading space on every call, which transformers 5 ignores
        scorer._tokenizer = AutoTokenizer.from_pretrained(model, add_prefix_space=True)
    return (lambda replies, references: scorer.score(replies, references)[2].tolist()), scorer.hash


def score_rows(rows: list[dict]) -> dict:
    """Metrics for one configuration's rows; BERTScore is averaged from each row's "bertscore" if set."""
    replies, refs = [r["reply"] for r in rows], [r["reference"] for r in rows]
    with_strategy = [r for r in rows if r.get("strategy")]
    return {
        "n": len(rows),
        "bleu2": bleu(replies, refs, 2)[0],
        "bleu4": bleu(replies, refs, 4)[0],
        "rouge_l": 100 * rouge_l(replies, refs),
        "bertscore_f1": 100 * mean(r["bertscore"] for r in rows) if all("bertscore" in r for r in rows) else None,
        "distinct1": 100 * distinct(replies, 1),
        "distinct2": 100 * distinct(replies, 2),
        "length": mean(len(words(x)) for x in replies),
        "strategy_acc": (
            100 * mean(r["strategy"] == r["reference_strategy"] for r in with_strategy) if with_strategy else None
        ),
    }


def reference_row(rows: list[dict]) -> dict:
    refs = [r["reference"] for r in rows]
    return {
        "n": len(rows),
        "distinct1": 100 * distinct(refs, 1),
        "distinct2": 100 * distinct(refs, 2),
        "length": mean(len(words(x)) for x in refs),
    }


def load_outputs(folder: Path, configs=SYSTEMS) -> dict[str, list[dict]]:
    """Rows per system that has an output file; every system must cover the same test rows."""
    data = {c: read_jsonl(folder / f"{c}.jsonl") for c in configs if (folder / f"{c}.jsonl").exists()}
    if not data:
        raise FileNotFoundError(f"no {{config}}.jsonl in {folder}; run src/generate_outputs.py first")
    keys = {c: sorted((r["dialogue_id"], r["utterance_idx"]) for r in rows) for c, rows in data.items()}
    if any(len(set(k)) != len(k) for k in keys.values()) or len({tuple(k) for k in keys.values()}) > 1:
        raise ValueError("configurations cover different or repeated rows; finish generation first")
    return data


def subset_rows(rows: list[dict], subset: str) -> tuple[list[dict], list[dict]]:
    """(rows in the subset, those Guard did not bypass)."""
    rows = [r for r in rows if subset == "all" or r["after_seeker"]]
    return rows, [r for r in rows if not r["bypassed"]]


def evaluate(data: dict[str, list[dict]], bertscore=None) -> list[dict]:
    """One result per (subset, configuration), plus the reference replies, for non-bypassed rows."""
    if bertscore is not None:
        for rows in data.values():
            kept = [r for r in rows if not r["bypassed"]]
            for r, f1 in zip(kept, bertscore([r["reply"] for r in kept], [r["reference"] for r in kept])):
                r["bertscore"] = f1
    results = []
    for subset in SUBSETS:
        for config, rows in data.items():
            rows, kept = subset_rows(rows, subset)
            if kept:
                results.append(
                    {"config": config, "subset": subset, "bypassed": len(rows) - len(kept), **score_rows(kept)}
                )
        rows, kept = subset_rows(next(iter(data.values())), subset)  # every configuration shares the references
        if kept:
            results.append(
                {"config": "reference", "subset": subset, "bypassed": len(rows) - len(kept), **reference_row(kept)}
            )
    return results


def fmt(value) -> str:
    return "--" if value is None else f"{value:.2f}"


def latex(results: list[dict], subset: str, bertscore_note: str) -> str:
    rows = [r for r in results if r["subset"] == subset]
    n = rows[0]["n"] if rows else 0
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + "r" * len(COLUMNS) + "}",
        r"\toprule",
        "Configuration & " + " & ".join(h for _, h in COLUMNS) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        if r["config"] == "reference":
            lines.append(r"\midrule")
        lines.append(
            NAMES.get(r["config"], r["config"]) + " & " + " & ".join(fmt(r.get(k)) for k, _ in COLUMNS) + r" \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Reference-based metrics on the ESConv test split, {SUBSETS[subset]} ({n} replies per "
        rf"configuration). {bertscore_note}}}",
        rf"\label{{tab:metrics-{subset.replace('_', '-')}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Reference-based metrics for each configuration.")
    ap.add_argument("--outputs", type=Path, default=ROOT / "data/outputs")
    ap.add_argument(
        "--configs",
        nargs="+",
        choices=SYSTEMS,
        default=list(SYSTEMS),
        help="systems to score; those without an output file are skipped",
    )
    ap.add_argument("--bertscore-model", default="roberta-large")
    ap.add_argument("--bertscore-layer", type=int, default=17)
    ap.add_argument("--no-rescale", action="store_true", help="raw BERTScore instead of baseline-rescaled")
    ap.add_argument("--no-bertscore", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    data = load_outputs(args.outputs, args.configs)
    bertscore, bertscore_hash = None, None
    if not args.no_bertscore:
        bertscore, bertscore_hash = bert_scorer(
            args.bertscore_model, args.bertscore_layer, not args.no_rescale, args.batch_size
        )
    results = evaluate(data, bertscore)
    settings = {
        "bleu": bleu(["a reply"], ["a reference"], 4)[1],  # sacrebleu signs a metric only after scoring
        "rouge": "rougeL F-measure, rouge-score " + version("rouge-score") + ", no stemming",
        "bertscore": None
        if bertscore is None
        else {
            "model": args.bertscore_model,
            "layer": args.bertscore_layer,
            "rescaled": not args.no_rescale,
            "hash": bertscore_hash,
        },
        "words": WORD.pattern,
        "bypassed_rows": "left out of every metric",
    }
    note = (
        "BERTScore not computed."
        if bertscore is None
        else f"BERTScore F1: {args.bertscore_model}, layer {args.bertscore_layer}"
        + ("" if args.no_rescale else ", rescaled with the baseline")
        + "."
    )
    with open(args.outputs / "metrics.csv", "w", encoding="utf-8", newline="") as f:
        fields = ["config", "subset", "n", "bypassed"] + [k for k, _ in COLUMNS]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
    (args.outputs / "metrics.tex").write_text("\n".join(latex(results, s, note) for s in SUBSETS), encoding="utf-8")
    (args.outputs / "metrics.json").write_text(
        json.dumps({"settings": settings, "results": results}, indent=1), encoding="utf-8"
    )
    for r in results:
        print(f"{r['subset']:<13}{r['config']:<12}n={r['n']:<5}" + " ".join(f"{k}={fmt(r.get(k))}" for k, _ in COLUMNS))


if __name__ == "__main__":
    main()

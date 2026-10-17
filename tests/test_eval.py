"""Evaluation tooling tests with tiny inputs and stand-in models; nothing is downloaded.

Run from the repository root: python -m unittest discover -s tests -v
"""

import contextlib
import csv
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from compare_configs import bootstrap_diff, paired_rows
from generate_outputs import generate, load_rows, row_turns
from human_eval_sheet import CRITERIA, FIELDS, build, krippendorff_alpha, read_sheet, score
from metrics import distinct, evaluate, latex, load_outputs, score_rows, words
from pipeline import CONFIGS, Kaki
from prep_downstream import history
from select_eval_contexts import eligible, history_turns, read_jsonl, select


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def response_row(dialogue, idx, text, problem="academic pressure", in_scope=True):
    return {
        "dialogue_id": dialogue,
        "utterance_idx": idx,
        "text": text,
        "response": "ref",
        "strategy": "Question",
        "problem": problem,
        "in_scope": in_scope,
    }


TWO_SEEKER = (
    "seeker: hi | supporter: Hello, what is on your mind? | seeker: my exams are next week and I "
    "cannot sleep because I keep thinking that I will fail all of them"
)


class SelectionTests(unittest.TestCase):
    def test_history_turns_round_trip(self):
        turns = history_turns(TWO_SEEKER)
        self.assertEqual([t["speaker"] for t in turns], ["seeker", "supporter", "seeker"])
        self.assertEqual(" | ".join(f"{t['speaker']}: {t['text']}" for t in turns), TWO_SEEKER)
        with self.assertRaises(ValueError):
            history_turns("no tags here")

    def test_eligibility(self):
        self.assertTrue(eligible(response_row(1, 3, TWO_SEEKER)))
        self.assertFalse(eligible(response_row(1, 1, "seeker: hi")))  # no earlier seeker message
        self.assertFalse(eligible(response_row(1, 4, TWO_SEEKER + " | supporter: ok")))  # ends with supporter
        self.assertFalse(eligible(response_row(1, 3, TWO_SEEKER), max_words=20))
        self.assertFalse(eligible(response_row(1, 3, TWO_SEEKER), min_seeker_words=30))

    def test_selection_is_balanced_distinct_and_reproducible(self):
        rows = []
        for d in range(30):
            problem = ["academic pressure", "problems with friends", "job crisis"][d % 3]
            rows += [response_row(d, i, TWO_SEEKER, problem) for i in (3, 5)]
        rows.append(response_row(99, 3, TWO_SEEKER, "other", in_scope=False))
        picks = select(rows, n=20, seed=1)
        self.assertEqual(picks, select(rows, n=20, seed=1))
        self.assertEqual(len({p["dialogue_id"] for p in picks}), 20)
        problems = [rows[2 * p["dialogue_id"]]["problem"] for p in picks]
        self.assertEqual(problems.count("academic pressure"), 10)
        self.assertEqual(problems.count("problems with friends"), 10)
        self.assertNotEqual(picks, select(rows, n=20, seed=2))

    def test_selection_fills_from_other_in_scope_problems(self):
        rows = [response_row(d, 3, TWO_SEEKER, "academic pressure" if d < 2 else "job crisis") for d in range(4)]
        rows.append(response_row(9, 3, TWO_SEEKER, "other", in_scope=False))
        picks = select(rows, n=4, seed=0)
        self.assertEqual(sorted(p["dialogue_id"] for p in picks), [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            select(rows, n=5, seed=0)


class FakeGuard:
    def check(self, turns):
        flagged = "crisis" in turns[-1]["text"]
        return {"bypass_generation": flagged, "response_key": "crisis_sg" if flagged else None}

    def response(self, key):
        return "prepared"


class FakePerceiver:
    def predict(self, history):
        return {"emotion": "anxiety", "problem": "academic pressure"}


class FakePlanner:
    def predict(self, history):
        return [
            {"strategy": "Question", "probability": 0.5},
            {"strategy": "Reflection of feelings", "probability": 0.3},
            {"strategy": "Others", "probability": 0.1},
        ]


class FakeSpeaker:
    def generate(self, history, emotion=None, strategy=None):
        return f"{strategy or 'plain'} reply to: {history.split(': ')[-1]}"

    def generate_candidates(self, history, emotion, strategies):
        return [
            {
                "strategy": s["strategy"],
                "planner_probability": s["probability"],
                "response": self.generate(history, emotion, s["strategy"]),
            }
            for s in strategies
        ]


def fake_kaki():
    """The real Kaki.respond with stand-in components, so outputs have the pipeline's exact shape."""
    return Kaki(FakeGuard(), FakePerceiver(), FakePlanner(), {"history": FakeSpeaker(), "conditioned": FakeSpeaker()})


RAW = [
    {
        "dialog": [
            {"speaker": "seeker", "content": "hi"},
            {"speaker": "supporter", "content": "Hello,  how are you?"},
            {"speaker": "seeker", "content": "  my exams are soon"},
            {"speaker": "supporter", "content": "That sounds stressful."},
            {"speaker": "supporter", "content": "Which subject?"},
            {"speaker": "seeker", "content": "I am in crisis"},
            {"speaker": "supporter", "content": "Please reach out."},
        ]
    }
]


def test_rows():
    dialog = RAW[0]["dialog"]
    return [
        {
            "dialogue_id": 0,
            "utterance_idx": i,
            "text": history(dialog[:i]),
            "response": dialog[i]["content"],
            "strategy": "Question",
            "problem": "academic pressure",
            "emotion": "anxiety",
        }
        for i in (1, 3, 4, 6)
    ]


class GenerateTests(unittest.TestCase):
    def test_turns_from_raw_match_turns_from_history(self):
        for row in test_rows():
            self.assertEqual(row_turns(row, RAW), row_turns(row))
        self.assertEqual(row_turns(test_rows()[1], RAW)[-1], {"speaker": "seeker", "text": "my exams are soon"})
        broken = dict(test_rows()[1], text="seeker: something else")
        with self.assertRaises(ValueError):
            row_turns(broken, RAW)

    def test_generate_writes_every_configuration_and_resumes(self):
        rows = test_rows()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            out = Path(tmp) / "outputs"
            self.assertEqual(generate(fake_kaki(), rows, CONFIGS, out, RAW), dict.fromkeys(CONFIGS, 4))
            lines = {c: read_jsonl(out / f"{c}.jsonl") for c in CONFIGS}
            self.assertTrue((out / ".gitignore").exists())
            direct, planning, lookahead = lines["direct"][1], lines["planning"][1], lines["lookahead"][1]
            self.assertEqual((direct["dialogue_id"], direct["utterance_idx"]), (0, 3))
            self.assertEqual(direct["reference"], "That sounds stressful.")
            self.assertEqual((direct["reply"], direct["strategy"]), ("plain reply to: my exams are soon", None))
            self.assertEqual((planning["strategy"], len(planning["candidates"])), ("Question", 1))
            self.assertEqual((lookahead["selected"], len(lookahead["candidates"])), (0, 3))
            self.assertEqual(lookahead["perceived"], {"emotion": "anxiety", "problem": "academic pressure"})
            self.assertEqual([r["after_seeker"] for r in lines["direct"]], [True, True, False, True])
            crisis = lines["lookahead"][3]
            self.assertEqual((crisis["bypassed"], crisis["reply"], crisis["strategy"]), (True, "prepared", None))
            self.assertEqual(generate(fake_kaki(), rows, CONFIGS, out, RAW), dict.fromkeys(CONFIGS, 0))
            generate(fake_kaki(), rows[:2], ["direct"], out, RAW, overwrite=True)
            self.assertEqual(len(read_jsonl(out / "direct.jsonl")), 2)

    def test_history_without_a_seeker_turn_is_skipped(self):
        raw = [
            {
                "dialog": [
                    {"speaker": "supporter", "content": "Hi there"},
                    {"speaker": "supporter", "content": "How are you?"},
                    {"speaker": "seeker", "content": "tired"},
                ]
            }
        ]
        row = {
            "dialogue_id": 0,
            "utterance_idx": 1,
            "text": history(raw[0]["dialog"][:1]),
            "response": "How are you?",
            "strategy": "Question",
            "problem": "other",
            "emotion": "other",
        }
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            out = Path(tmp) / "outputs"
            self.assertEqual(generate(fake_kaki(), [row], CONFIGS, out, raw), dict.fromkeys(CONFIGS, 0))
            self.assertEqual(read_jsonl(out / "direct.jsonl"), [])

    def test_load_rows_follows_contexts(self):
        rows = test_rows()
        with tempfile.TemporaryDirectory() as tmp:
            data, contexts = Path(tmp) / "responses.jsonl", Path(tmp) / "contexts.json"
            write_jsonl(data, rows)
            contexts.write_text(
                json.dumps(
                    {"contexts": [{"dialogue_id": 0, "utterance_idx": 4}, {"dialogue_id": 0, "utterance_idx": 1}]}
                ),
                encoding="utf-8",
            )
            self.assertEqual([r["utterance_idx"] for r in load_rows(data, contexts)], [4, 1])
            self.assertEqual(len(load_rows(data, limit=3)), 3)
            contexts.write_text(json.dumps({"contexts": [{"dialogue_id": 5, "utterance_idx": 1}]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_rows(data, contexts)


def output_row(config, idx, reply, strategy=None, bypassed=False, after_seeker=True):
    return {
        "dialogue_id": 1,
        "utterance_idx": idx,
        "config": config,
        "reply": reply,
        "reference": "that sounds hard",
        "reference_strategy": "Question",
        "strategy": strategy,
        "bypassed": bypassed,
        "after_seeker": after_seeker,
    }


def output_data():
    return {
        "direct": [
            output_row("direct", 1, "that sounds hard"),
            output_row("direct", 2, "ok ok", after_seeker=False),
            output_row("direct", 3, "prepared", bypassed=True),
        ],
        "planning": [
            output_row("planning", 1, "that sounds hard", "Question"),
            output_row("planning", 2, "why", "Others", after_seeker=False),
            output_row("planning", 3, "prepared", bypassed=True),
        ],
    }


class MetricsTests(unittest.TestCase):
    def test_words_and_distinct(self):
        self.assertEqual(words("I'm OK, really!"), ["i'm", "ok", "really"])
        self.assertAlmostEqual(distinct(["a b a", "b c"], 1), 3 / 5)
        self.assertAlmostEqual(distinct(["a b a", "b c"], 2), 1.0)
        self.assertEqual(distinct(["a"], 2), 0.0)

    def test_identical_replies_score_full_marks(self):
        reply = "that sounds really hard for you"  # BLEU-4 needs four-word n-grams
        rows = [dict(output_row("planning", i, reply, "Question"), reference=reply, bertscore=0.9) for i in range(3)]
        m = score_rows(rows)
        self.assertAlmostEqual(m["bleu2"], 100)
        self.assertAlmostEqual(m["bleu4"], 100)
        self.assertAlmostEqual(m["rouge_l"], 100)
        self.assertAlmostEqual(m["bertscore_f1"], 90)
        self.assertAlmostEqual(m["strategy_acc"], 100)
        self.assertEqual(m["length"], 6)

    def test_subsets_bypass_and_strategy_accuracy(self):
        calls = []

        def fake_bertscore(replies, references):
            calls.append(len(replies))
            return [0.5] * len(replies)

        results = {(r["subset"], r["config"]): r for r in evaluate(output_data(), fake_bertscore)}
        self.assertEqual(calls, [2, 2])  # bypassed rows are not scored
        self.assertEqual((results["all", "direct"]["n"], results["all", "direct"]["bypassed"]), (2, 1))
        self.assertIsNone(results["all", "direct"]["strategy_acc"])
        self.assertAlmostEqual(results["all", "planning"]["strategy_acc"], 50)
        self.assertAlmostEqual(results["after_seeker", "planning"]["strategy_acc"], 100)
        self.assertEqual(results["after_seeker", "planning"]["n"], 1)
        self.assertAlmostEqual(results["all", "planning"]["bertscore_f1"], 50)
        self.assertEqual(results["all", "reference"]["length"], 3)
        self.assertNotIn("bleu2", results["all", "reference"])
        table = latex(list(results.values()), "all", "BERTScore F1: test.")
        self.assertIn("Planning & ", table)
        self.assertIn(r"\label{tab:metrics-all}", table)
        self.assertIn("2 replies per configuration", table)

    def test_configurations_must_cover_the_same_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = output_data()
            for config, rows in data.items():
                write_jsonl(Path(tmp, f"{config}.jsonl"), rows)
            self.assertEqual(sorted(load_outputs(Path(tmp))), ["direct", "planning"])
            write_jsonl(Path(tmp, "planning.jsonl"), data["planning"][:1])
            with self.assertRaises(ValueError):
                load_outputs(Path(tmp))


# Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability": 4 observers, 12 units, missing values.
# Published: nominal 0.743, ordinal 0.815, interval 0.849.
KRIPPENDORFF_2011 = [
    [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
    [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
    [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
    [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None],
]


def sheet_inputs():
    contexts = [response_row(7, 3, TWO_SEEKER), response_row(9, 5, TWO_SEEKER)]
    outputs = {
        c: {(r["dialogue_id"], r["utterance_idx"]): {"reply": f"{c} reply {r['dialogue_id']}"} for r in contexts}
        for c in CONFIGS
    }
    return contexts, outputs


def fill(sheet, key, rating_of):
    """A filled copy of the sheet; rating_of(config, criterion) gives each rating."""
    config_of = {(c["context"], label): group[0] for c in key["contexts"] for label, group in c["labels"].items()}
    return {
        (r["context"], r["label"]): {c: rating_of(config_of[r["context"], r["label"]], c) for c in CRITERIA}
        for r in sheet
    }


class HumanEvalTests(unittest.TestCase):
    def test_alpha_matches_published_values(self):
        units = [[row[j] for row in KRIPPENDORFF_2011 if row[j] is not None] for j in range(12)]
        self.assertAlmostEqual(krippendorff_alpha(units, "nominal"), 0.743, places=3)
        self.assertAlmostEqual(krippendorff_alpha(units, "ordinal"), 0.815, places=3)
        self.assertAlmostEqual(krippendorff_alpha(units, "interval"), 0.849, places=3)
        self.assertEqual(krippendorff_alpha([[1, 1], [3, 3], [5, 5]]), 1.0)
        self.assertTrue(math.isnan(krippendorff_alpha([[4, 4], [4, 4]])))

    def test_sheet_is_blinded_and_reproducible(self):
        contexts, outputs = sheet_inputs()
        sheet, key = build(contexts, outputs, seed=3)
        self.assertEqual((sheet, key), build(contexts, outputs, seed=3))
        self.assertEqual(
            [(r["context"], r["label"]) for r in sheet], [(1, "A"), (1, "B"), (1, "C"), (2, "A"), (2, "B"), (2, "C")]
        )
        self.assertTrue(sheet[0]["history"].startswith("Seeker: hi\nSupporter: Hello"))
        self.assertEqual([r["history"] for r in sheet[1:3]], ["", ""])
        for row in sheet:
            [config] = key["contexts"][row["context"] - 1]["labels"][row["label"]]
            self.assertTrue(row["reply"].startswith(config))
            self.assertEqual(set(row), set(FIELDS))
        self.assertEqual(sorted(g for group in key["contexts"][0]["labels"].values() for g in group), sorted(CONFIGS))
        self.assertEqual((key["contexts"][1]["dialogue_id"], key["contexts"][1]["utterance_idx"]), (9, 5))
        orders = {
            tuple(g[0] for g in build(contexts, outputs, seed=s)[1]["contexts"][0]["labels"].values())
            for s in range(20)
        }
        self.assertGreater(len(orders), 1)

    def test_read_sheet_rejects_missing_or_out_of_range_ratings(self):
        contexts, outputs = sheet_inputs()
        sheet, key = build(contexts, outputs)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rater.csv"
            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(
                    [dict(r, relevance=4, acknowledgement="4.0", helpfulness=5, fluency=3, safety=5) for r in sheet]
                )
                f.write(",,,,,,,,,\n")
            ratings = read_sheet(path)
            self.assertEqual(
                ratings[1, "A"], {"relevance": 4, "acknowledgement": 4, "helpfulness": 5, "fluency": 3, "safety": 5}
            )
            self.assertEqual(len(ratings), 6)
            with open(path, "a", encoding="utf-8", newline="") as f:
                f.write("1,A,,reply,6,,3.5,2,\n")
            with self.assertRaises(ValueError) as error:
                read_sheet(path)
            self.assertIn("relevance: '6'", str(error.exception))
            self.assertIn("helpfulness: '3.5'", str(error.exception))

    def test_scoring_unblinds_and_lists_disagreements(self):
        contexts, outputs = sheet_inputs()
        sheet, key = build(contexts, outputs)
        level = {"direct": 2, "planning": 4, "lookahead": 5}
        rater1 = fill(sheet, key, lambda config, c: level[config])
        rater2 = fill(sheet, key, lambda config, c: level[config])
        unit = next(u for u in rater2 if key["contexts"][u[0] - 1]["labels"][u[1]] == ["direct"])
        rater2[unit]["fluency"] = 5
        report = score([rater1, rater2], key)
        self.assertEqual(report["means"]["planning"]["relevance"], 4)
        self.assertEqual(report["means"]["direct"]["fluency"], 2.75)
        self.assertEqual(report["means_per_rater"][1]["direct"]["fluency"], 3.5)
        self.assertEqual(report["alpha_ordinal"]["relevance"], 1.0)
        self.assertLess(report["alpha_ordinal"]["fluency"], 1.0)
        self.assertEqual(
            report["disagreements"],
            [{"context": unit[0], "label": unit[1], "configs": ["direct"], "criterion": "fluency", "ratings": [2, 5]}],
        )
        with self.assertRaises(ValueError):
            score([rater1, {}], key)

    def test_identical_replies_are_shown_once_and_credited_to_each_writer(self):
        contexts, outputs = sheet_inputs()
        for ids in outputs["lookahead"]:
            outputs["lookahead"][ids] = dict(outputs["planning"][ids])
        sheet, key = build(contexts, outputs)
        self.assertEqual(len(sheet), 4)  # two distinct replies per context
        groups = sorted(sorted(g) for g in key["contexts"][0]["labels"].values())
        self.assertEqual(groups, [["direct"], ["lookahead", "planning"]])
        ratings = {
            (r["context"], r["label"]): dict.fromkeys(CRITERIA, 3 if r["reply"].startswith("direct") else 5)
            for r in sheet
        }
        report = score([ratings, ratings], key)
        self.assertEqual(
            (
                report["means"]["planning"]["relevance"],
                report["means"]["lookahead"]["relevance"],
                report["means"]["direct"]["relevance"],
            ),
            (5, 5, 3),
        )


class CompareTests(unittest.TestCase):
    def test_rows_are_aligned_and_filtered(self):
        def line(c, key, after=True, bypassed=False):
            return {
                "dialogue_id": key[0],
                "utterance_idx": key[1],
                "reference": f"ref {key}",
                "reply": f"{c} {key}",
                "after_seeker": after,
                "bypassed": bypassed,
            }

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for c in CONFIGS:
                rows = [line(c, (2, 1)), line(c, (1, 3)), line(c, (1, 5), after=False), line(c, (3, 1), bypassed=True)]
                if c == "direct":
                    rows.append(line(c, (4, 1)))  # missing from the other configurations
                write_jsonl(out / f"{c}.jsonl", rows)
            after = paired_rows(out, "after_seeker")
            self.assertEqual([(r["dialogue_id"], r["utterance_idx"]) for r in after["lookahead"]], [(1, 3), (2, 1)])
            self.assertEqual([r["reply"] for r in after["planning"]], ["planning (1, 3)", "planning (2, 1)"])
            self.assertEqual(len(paired_rows(out, "all")["direct"]), 3)

    def test_bootstrap_interval_contains_the_mean(self):
        d = bootstrap_diff([3.0, 4.0, 5.0, 6.0], [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(d["mean"], 3.5)
        self.assertLessEqual(d["ci95"][0], 3.5)
        self.assertGreaterEqual(d["ci95"][1], 3.5)
        self.assertEqual(bootstrap_diff([1.0, 2.0], [1.0, 2.0]), {"mean": 0.0, "ci95": [0.0, 0.0]})


if __name__ == "__main__":
    unittest.main()

"""Risk-labelled data for Guard: Reddit posts annotated with C-SSRS-based suicide-risk levels.

Source: Gaur et al. (2021), PLOS ONE 16(5): e0250448, Zenodo record 4543776 (version 2, CC BY 4.0).
Four psychiatrists labelled posts as supportive, indicator, ideation, behavior, attempt or uninformative.

Pipeline: download + checksum -> read one sheet per Reddit user -> normalise label spelling ->
drop unlabelled rows -> binary label (ideation/behavior/attempt = risk) -> dedup on normalised text ->
split by user (seeded, stratified on whether the user has any risk post) -> write jsonl.

Sheet names in the release are Reddit usernames. They are replaced by integer group ids and never written.

Usage: python src/prep_guard.py [--seed 42] [--raw data/raw/guard] [--out data/processed/guard]
"""

import argparse
import collections
import hashlib
import json
import random
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZENODO_URL = "https://zenodo.org/api/records/4543776/files/{}/content"
FILES = {  # md5 checksums published on the Zenodo record
    "Redditors_and_posts_batch_1.xlsx": "ac00635a93f05c2323be082073cb1a43",
    "Redditors_and_posts_batch_2.xlsx": "d594431db867b4e24f0c6fe6e6474fda",
    "Redditors_and_posts_batch_3.xlsx": "0de886e0ced9ba95bb53e194104334ae",
    "Redditors_and_posts_batch_4.xlsx": "bdb9013f14e3ae82c4cb28bbb6094c0a",
    "Redditors_and_posts_batch_5.xlsx": "211b5b6b378a4804bd94b3642b151fd6",
}

# spelling variants found in the release -> canonical label
LABEL_MAP = {
    "supportive": "supportive",
    "supporitve": "supportive",
    "suppr": "supportive",
    "indicator": "indicator",
    "indiccator": "indicator",
    "uninformative": "uninformative",
    "noninformative": "uninformative",
    "uniformative": "uninformative",
    "noninformtative": "uninformative",
    "ideation": "ideation",
    "ideations": "ideation",
    "suicidal ideations": "ideation",
    "behavior": "behavior",
    "attempt": "attempt",
}
RISK_LABELS = {"ideation", "behavior", "attempt"}  # C-SSRS levels that warrant referral to human help

SPLIT_FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}  # by posts; val/test larger than ESConv's 10%
# because positives are scarce

XLSX = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
XLSX_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def fetch_raw(raw_dir: Path) -> list[Path]:
    """Download the five annotated workbooks (skipped when a verified copy exists)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, checksum in FILES.items():
        path = raw_dir / name
        for attempt in range(5):  # Zenodo occasionally drops connections, mid-file or before it starts
            if path.exists() and md5(path) == checksum:
                break
            try:
                urllib.request.urlretrieve(ZENODO_URL.format(name), path)
            except OSError:  # includes URLError and ContentTooShortError
                time.sleep(2**attempt)
        if not path.exists() or md5(path) != checksum:
            raise ValueError(f"{name}: download incomplete or checksum mismatch (Zenodo file changed?)")
        paths.append(path)
    return paths


def read_sheets(path: Path) -> list[list[dict]]:
    """Rows of every sheet in workbook order, as {column letter: cell text}. Standard library only."""
    z = zipfile.ZipFile(path)
    shared = [
        "".join(t.text or "" for t in si.iter(XLSX + "t"))
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(XLSX + "si")
    ]
    targets = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    sheets = []
    for sheet in ET.fromstring(z.read("xl/workbook.xml")).iter(XLSX + "sheet"):
        target = "xl/" + targets[sheet.get(XLSX_REL + "id")].lstrip("/").removeprefix("xl/")
        rows = []
        for row in ET.fromstring(z.read(target)).iter(XLSX + "row"):
            cells = {}
            for c in row.findall(XLSX + "c"):
                v = c.find(XLSX + "v")
                if v is not None:
                    col = re.match("[A-Z]+", c.get("r")).group()
                    cells[col] = shared[int(v.text)] if c.get("t") == "s" else v.text
            rows.append(cells)
        sheets.append(rows)
    return sheets


def norm_text(s: str) -> str:
    return " ".join(s.split())


def load_posts(paths: list[Path]) -> tuple[list[dict], collections.Counter]:
    """One row per labelled post. Group = position of the user's sheet across the five workbooks."""
    posts, counts, group = [], collections.Counter(), 0
    for path in paths:
        for rows in read_sheets(path):
            if not rows:
                counts["empty_sheets"] += 1
                continue
            if rows[0].get("A") != "Reddit Post" or rows[0].get("B") != "Suicide Risk":
                raise ValueError(f"{path.name}: unexpected header {rows[0]}")
            for r in rows[1:]:
                text = norm_text(r.get("A") or "")
                raw = (r.get("B") or "").strip().lower()
                if not text:
                    continue
                if not raw:
                    counts["unlabelled_posts"] += 1
                    continue
                if raw not in LABEL_MAP:
                    raise ValueError(f"unknown label {raw!r}; add it to LABEL_MAP")
                counts[f"label_spelling:{raw}"] += 1
                label = LABEL_MAP[raw]
                posts.append({"group": group, "text": text, "label_raw": label, "label": int(label in RISK_LABELS)})
            group += 1
    counts["groups"] = group
    return posts, counts


def dedup(posts: list[dict]) -> tuple[list[dict], int, int]:
    """Drop repeated texts (case/whitespace-insensitive), keeping the first copy.
    Texts that occur with different binary labels are dropped entirely."""
    by_key = collections.defaultdict(list)
    for p in posts:
        by_key[p["text"].lower()].append(p)
    kept, n_dups, n_conflicts = [], 0, 0
    for copies in by_key.values():
        if len({p["label"] for p in copies}) > 1:
            n_conflicts += len(copies)
            continue
        kept.append(copies[0])
        n_dups += len(copies) - 1
    return kept, n_dups, n_conflicts


def split_groups(posts: list[dict], seed: int) -> dict[str, list[int]]:
    """Assign whole users to splits so that no user's posts cross partitions.
    Within each stratum (users with / without a risk post), users are shuffled and each goes to the
    split furthest below its target share of posts."""
    rng = random.Random(seed)
    size = collections.Counter(p["group"] for p in posts)
    has_risk = {p["group"] for p in posts if p["label"]}
    split = {name: [] for name in SPLIT_FRACS}
    for stratum in (True, False):
        members = sorted(g for g in size if (g in has_risk) == stratum)
        rng.shuffle(members)
        total, filled = sum(size[g] for g in members), dict.fromkeys(SPLIT_FRACS, 0)
        for g in members:
            name = min(SPLIT_FRACS, key=lambda s: filled[s] / (SPLIT_FRACS[s] * total))
            split[name].append(g)
            filled[name] += size[g]
    return {k: sorted(v) for k, v in split.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--raw", type=Path, default=ROOT / "data/raw/guard")
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed/guard")
    args = ap.parse_args()

    posts, counts = load_posts(fetch_raw(args.raw))
    n_labelled = len(posts)
    posts, n_dups, n_conflicts = dedup(posts)

    # split BEFORE writing, by user, so no user's posts appear in two partitions
    split = split_groups(posts, args.seed)
    group_split = {g: name for name, groups in split.items() for g in groups}

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "source": "Gaur et al. 2021, Zenodo 4543776 (CC BY 4.0)",
        "seed": args.seed,
        "file_md5": FILES,
        "risk_labels": sorted(RISK_LABELS),
        "user_sheets": counts["groups"],
        "empty_sheets": counts["empty_sheets"],
        "unlabelled_posts_excluded": counts["unlabelled_posts"],
        "labelled_posts": n_labelled,
        "label_spellings": {k.split(":", 1)[1]: v for k, v in counts.items() if k.startswith("label_spelling:")},
        "duplicates_removed": n_dups,
        "conflicting_duplicates_removed": n_conflicts,
        "posts_after_dedup": len(posts),
        "splits": {},
    }
    for name in SPLIT_FRACS:
        rows = [p for p in posts if group_split[p["group"]] == name]
        with open(args.out / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for i, p in enumerate(rows):
                f.write(json.dumps({"id": f"{name}-{i}", **p}) + "\n")
        report["splits"][name] = {
            "users": len(split[name]),
            "posts": len(rows),
            "risk_posts": sum(p["label"] for p in rows),
            "label_raw": dict(collections.Counter(p["label_raw"] for p in rows).most_common()),
        }

    # LF line endings on every OS, so the fingerprint below is the same on Windows, macOS and Linux
    splits = json.dumps({"seed": args.seed, **split}, indent=1)
    (args.out / "splits.json").write_text(splits, encoding="utf-8", newline="\n")
    report["splits_sha256"] = hashlib.sha256(splits.encode()).hexdigest()[:12]
    (args.out / "prep_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

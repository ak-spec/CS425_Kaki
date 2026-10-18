"""Download the trained models from the repository's v1.0 release into checkpoints/.

The release holds one archive with everything the demo and src/pipeline.py load: Guard, Perceive, Plan,
both Speak adapters, the reply-contrast Critic with the look-ahead weight chosen on validation, and
Perceive's label map (data/processed/label_maps.json). Speak's base model (Qwen2.5-0.5B-Instruct) and the
encoders' tokenizers come from the Hugging Face Hub on first use, as in training.

Usage: python src/download_checkpoints.py [--force] [--keep-archive]
"""

import argparse
import hashlib
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "https://github.com/ak-spec/CS425_Kaki/releases/download/v1.0/kaki-checkpoints-v1.tar"
SHA256 = "e28b40d6989c4de5586aa50e540ab9f4143f3c4c149817f83df926f9ca3b4b49"
ALLOWED = ("checkpoints/", "data/processed/label_maps.json")  # nothing else is unpacked


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def progress(blocks: int, block_size: int, total: int) -> None:
    if total > 0:
        print(f"\r  {min(blocks * block_size, total) / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)


def fetch(path: Path) -> None:
    """Download the archive (skipped when a verified copy exists)."""
    for attempt in range(5):  # retry dropped connections
        if path.exists() and sha256(path) == SHA256:
            return
        try:
            urllib.request.urlretrieve(URL, path, reporthook=progress)
            print()
        except OSError:  # includes URLError and ContentTooShortError
            time.sleep(2**attempt)
    if not path.exists() or sha256(path) != SHA256:
        raise ValueError(f"{path.name}: download incomplete or checksum mismatch")


def extract(archive: Path, root: Path, force: bool = False) -> list[str]:
    """Unpack the allowed members into root; refuse to overwrite existing checkpoints unless force."""
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.name.startswith(ALLOWED)]
        parts = [Path(m.name).parts for m in members]
        stages = sorted({p[1] for p in parts if p[0] == "checkpoints" and len(p) > 1})
        existing = [f"checkpoints/{s}" for s in stages if (root / "checkpoints" / s).exists()]
        if existing and not force:
            raise FileExistsError(f"{', '.join(existing)} already exist; use --force to replace them")
        tar.extractall(root, members=members, filter="data")
    return stages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=ROOT / "data/kaki-checkpoints-v1.tar")
    ap.add_argument("--force", action="store_true", help="replace checkpoints that already exist")
    ap.add_argument("--keep-archive", action="store_true", help="keep the 1.9 GB archive after unpacking")
    args = ap.parse_args()

    args.archive.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    fetch(args.archive)
    stages = extract(args.archive, ROOT, args.force)
    if not args.keep_archive:
        args.archive.unlink()
    print(f"Unpacked checkpoints/{{{','.join(stages)}}} and data/processed/label_maps.json")


if __name__ == "__main__":
    main()

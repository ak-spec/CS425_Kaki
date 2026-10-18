"""Tests for unpacking the release archive; nothing is downloaded.

Run from the repository root: python -m unittest discover -s tests -v
"""

import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from download_checkpoints import extract


def make_archive(path: Path, names: list[str]) -> None:
    with tarfile.open(path, "w") as tar:
        for name in names:
            data = name.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


class ExtractTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.archive = self.root / "bundle.tar"
        make_archive(
            self.archive,
            [
                "checkpoints/guard/config.json",
                "checkpoints/speak-history/adapter_config.json",
                "data/processed/label_maps.json",
                "data/processed/other.jsonl",
                "notes.txt",
            ],
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_unpacks_only_checkpoints_and_label_map(self):
        stages = extract(self.archive, self.root)
        self.assertEqual(stages, ["guard", "speak-history"])
        self.assertTrue((self.root / "checkpoints/guard/config.json").exists())
        self.assertTrue((self.root / "data/processed/label_maps.json").exists())
        self.assertFalse((self.root / "data/processed/other.jsonl").exists())
        self.assertFalse((self.root / "notes.txt").exists())

    def test_existing_checkpoints_need_force(self):
        (self.root / "checkpoints/guard").mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            extract(self.archive, self.root)
        self.assertEqual(extract(self.archive, self.root, force=True), ["guard", "speak-history"])


if __name__ == "__main__":
    unittest.main()

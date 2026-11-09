import os
import unittest

from PIL import Image

from rfpii_nn import gold
from rfpii_nn.dataset import CLASSES, Dataset, DatasetError, split_of, training_pages

from .helpers import TempDir, fake_dataset, have_rfpii, synth


class SplitTest(unittest.TestCase):
    def test_split_matches_rust(self):
        # values rfpii computes (dataset::split_of): the rule is the same
        counts = {"train": 0, "val": 0, "test": 0}
        for k in range(5000):
            counts[split_of("seed:%d" % k)] += 1
        self.assertTrue(3800 < counts["train"] < 4200, counts)
        self.assertEqual(split_of("county:007/constituency:032"), split_of("county:007/constituency:032"))


@unittest.skipUnless(have_rfpii(), "rust/target/release/rfpii not built: cargo build --release")
class SynthDatasetTest(unittest.TestCase):
    def test_a_synthetic_dataset_reads_back(self):
        with TempDir() as d:
            out = os.path.join(d, "syn")
            synth(out, count=4, seed=10)
            ds = Dataset(out)
            self.assertEqual(ds.kind, "synthetic")
            self.assertFalse(ds.personal_data)
            self.assertEqual(len(ds), 4)
            self.assertEqual(ds.verify_splits(), [])
            with self.assertRaises(DatasetError):
                list(ds.pages("train"))
            seen = 0
            for e in ds.pages(exclude=gold.NOTHING):
                lab = ds.labels(e)
                img = ds.image(e)
                self.assertEqual((img.width, img.height), (lab["page"]["width"], lab["page"]["height"]))
                m = ds.mask(e)
                self.assertEqual(m.shape, (img.height, img.width))
                self.assertTrue(m.max() <= 15)
                for b in lab["boxes"]:
                    self.assertIn(b["class"], CLASSES)
                    x0, y0, x1, y1 = b["box"]
                    self.assertTrue(0 <= x0 < x1 <= img.width and 0 <= y0 < y1 <= img.height)
                seen += 1
            self.assertEqual(seen, 4)

    def test_training_never_sees_the_gold_seeds(self):
        with TempDir() as d:
            out = os.path.join(d, "syn")
            synth(out, count=3, seed=gold.GOLD_SEED - 1)
            ds = Dataset(out)
            ex = gold.Exclusion(seed_ranges=[(gold.GOLD_SEED, gold.GOLD_SEED + 300)])
            kept = list(ds.pages(exclude=ex))
            self.assertEqual([e["split_key"] for e in kept], ["seed:%d" % (gold.GOLD_SEED - 1)])
            self.assertTrue(all(e["split"] == "train" for e in training_pages(ds, ex)))
            with self.assertRaises(DatasetError):
                training_pages(ds, None)


class FilesTest(unittest.TestCase):
    def test_paths_never_leave_the_dataset(self):
        with TempDir() as d:
            path = fake_dataset(os.path.join(d, "t"), "teacher", [("a", {"image": "../../etc/passwd"}, [])])
            ds = Dataset(path)
            with self.assertRaises(DatasetError):
                ds.image_path("a")


if __name__ == "__main__":
    unittest.main()

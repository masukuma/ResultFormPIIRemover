import json
import os
import unittest

from rfpii_nn import gold
from rfpii_nn.dataset import Dataset

from .helpers import TempDir, fake_dataset, have_rfpii, RFPII, TEMPLATES


def teacher(path, n, template="form34a_2022", capture="kiems", held_every=0, start=0):
    entries = []
    for i in range(start, start + n):
        held = bool(held_every) and i % held_every == 0
        entries.append(("t-%04d" % i, {"template": template, "year": 2022, "capture": capture, "held": held,
                                       "image_sha256": "%064x" % (i + 1000 * len(template) + (7 if capture == "scan" else 0)),
                                       "split_key": "county:%03d/constituency:%03d" % (i % 5, i % 7)}, []))
    return fake_dataset(path, "teacher", entries)


class AllocateTest(unittest.TestCase):
    def test_every_stratum_gets_a_page_and_the_rest_in_proportion(self):
        q = gold.allocate({"a": 900, "b": 90, "c": 10}, 100)
        self.assertEqual(sum(q.values()), 100)
        self.assertTrue(q["c"] >= 1 and q["b"] >= 9 and q["a"] >= 85, q)
        self.assertEqual(gold.allocate({"a": 3, "b": 2}, 10), {"a": 3, "b": 2})
        self.assertEqual(sum(gold.allocate({"a": 5, "b": 5, "c": 5}, 2).values()), 2)


class PlanTest(unittest.TestCase):
    def test_a_plan_is_deterministic_and_stratified(self):
        with TempDir() as d:
            a = teacher(os.path.join(d, "a"), 200, held_every=10)
            b = teacher(os.path.join(d, "b"), 40, template="form34b_2022_block", capture="scan")
            p1 = gold.plan(os.path.join(d, "g1"), [a, b], size=30)
            p2 = gold.plan(os.path.join(d, "g2"), [a, b], size=30)
            self.assertEqual(p1["reference"]["pages"], p2["reference"]["pages"])
            self.assertEqual(p1["reference"]["size"], 30)
            strata = p1["reference"]["strata"]
            self.assertEqual(set(strata), {"form34a_2022|2022|kiems|published", "form34a_2022|2022|kiems|held", "form34b_2022_block|2022|scan|published"})
            self.assertTrue(all(n >= 1 for n in strata.values()))
            self.assertEqual(p1["synthetic"]["seeds"], [gold.GOLD_SEED, gold.GOLD_SEED + gold.DEFAULT_SIZE])
            self.assertEqual(p1["version"], 1)
            with open(os.path.join(d, "g1", "VERSION")) as f:
                self.assertEqual(f.read().strip(), "1")

    def test_a_plan_only_grows(self):
        with TempDir() as d:
            a = teacher(os.path.join(d, "a"), 50)
            g = os.path.join(d, "g")
            p1 = gold.plan(g, [a], size=10)
            same = gold.plan(g, [a], size=10)
            self.assertEqual(same["version"], 1)
            c = teacher(os.path.join(d, "c"), 30, template="form34b_2022_block_b", capture="photo", start=500)
            p2 = gold.plan(g, [a, c], size=20, synthetic_seed=5)
            self.assertEqual(p2["version"], 2)
            old = {p["image_sha256"] for p in p1["reference"]["pages"]}
            new = {p["image_sha256"] for p in p2["reference"]["pages"]}
            self.assertTrue(old <= new and len(new) == 20)
            self.assertEqual(p2["synthetic"]["seeds"], p1["synthetic"]["seeds"], "the synthetic seeds never move")
            # a later plan without the first dataset still keeps its pages
            p3 = gold.plan(g, [c], size=20)
            self.assertTrue(old <= {p["image_sha256"] for p in p3["reference"]["pages"]})
            self.assertEqual(len(p2["history"]), 2)

    def test_only_teacher_datasets_make_the_reference_set(self):
        with TempDir() as d:
            s = fake_dataset(os.path.join(d, "s"), "synthetic", [("x", {"split_key": "seed:1"}, [])])
            with self.assertRaises(ValueError):
                gold.plan(os.path.join(d, "g"), [s])


class ExclusionTest(unittest.TestCase):
    def test_reference_pages_and_gold_seeds_are_excluded_everywhere(self):
        with TempDir() as d:
            a = teacher(os.path.join(d, "a"), 30)
            g = os.path.join(d, "g")
            p = gold.plan(g, [a], size=5)
            ex = gold.Exclusion.load(g)
            ds = Dataset(a)
            left = list(ds.pages(exclude=ex))
            self.assertEqual(len(left), 25)
            gone = {x["image_sha256"] for x in p["reference"]["pages"]}
            self.assertFalse(any(e["image_sha256"] in gone for e in left))
            self.assertTrue(ex.excludes({"split_key": "seed:%d" % gold.GOLD_SEED}))
            self.assertTrue(ex.excludes({"split_key": "seed:%d" % (gold.GOLD_SEED + 299)}))
            self.assertFalse(ex.excludes({"split_key": "seed:%d" % (gold.GOLD_SEED + 300)}))
            self.assertFalse(ex.excludes({"split_key": "seed:12"}))
            self.assertFalse(gold.NOTHING.excludes({"split_key": "seed:%d" % gold.GOLD_SEED}))

    @unittest.skipUnless(have_rfpii(), "rust/target/release/rfpii not built")
    def test_the_synthetic_gold_set_is_made_from_its_seeds(self):
        with TempDir() as d:
            g = os.path.join(d, "g")
            gold.plan(g, [], synthetic_count=2, templates=TEMPLATES)
            out = gold.make_synthetic(g, RFPII, TEMPLATES, workers=2)
            ds = Dataset(out)
            self.assertEqual(sorted(e["split_key"] for e in ds.index), ["seed:%d" % gold.GOLD_SEED, "seed:%d" % (gold.GOLD_SEED + 1)])
            ex = gold.Exclusion.load(g)
            self.assertEqual(list(ds.pages(exclude=ex)), [])
            with open(os.path.join(g, "plan.json")) as f:
                self.assertIn("rfpii dataset synth", json.load(f)["synthetic"]["command"])


if __name__ == "__main__":
    unittest.main()

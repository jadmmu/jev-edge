import tempfile
import unittest
from pathlib import Path

from q8_perturbation import validate_pilot


class PerturbationTests(unittest.TestCase):
    def test_pilot_dataset_validates(self):
        root = Path(__file__).parents[2]
        report = validate_pilot(root / "experiments/datasets/q8_perturbation_pilot.jsonl",
                                root / "experiments/datasets/benchmark_20.jsonl")
        self.assertEqual(report["conditions"], 24)
        self.assertEqual(report["parents"], 4)
        self.assertEqual(report["categories_per_parent"], 6)

    def test_variant_changes_only_ticket(self):
        root = Path(__file__).parents[2]
        report = validate_pilot(root / "experiments/datasets/q8_perturbation_pilot.jsonl",
                                root / "experiments/datasets/benchmark_20.jsonl")
        variants = [row for row in report["review_rows"] if row["variant_type"] != "control"]
        self.assertEqual(len(variants), 20)
        self.assertTrue(all(row["original_text"] != row["modified_text"] for row in variants))


if __name__ == "__main__":
    unittest.main()

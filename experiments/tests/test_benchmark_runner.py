import io
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).parents[1]))
import benchmark_runner


def record(question_id, latency, status=200, warmup=False):
    return {
        "record_type": "observation",
        "run_id": "run",
        "question_id": question_id,
        "condition_id": question_id,
        "parent_question_id": None,
        "variant_type": "control",
        "original_latency_regime": None,
        "description": "test",
        "trial_index": 0,
        "block_number": None,
        "block_position": None,
        "is_warmup": warmup,
        "request_payload": {"state": question_id, "questions": {}},
        "response": {"answers": {}, "usage": {}, "model": "test"},
        "timestamp_utc": "2026-10-24T00:00:00.000+00:00",
        "model": "test",
        "wall_clock_ms": latency,
        "status_code": status,
        "response_headers": {},
        "error": None if status == 200 else "HTTP 500",
    }


class BenchmarkRunnerTests(unittest.TestCase):
    def test_jsonl_and_csv_serialization(self):
        records = [record("q1", 10)]
        metadata = {"run_id": "run", "endpoint": "endpoint"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl = root / "observations.jsonl"
            csv_path = root / "observations.csv"
            summary = root / "summary.json"
            benchmark_runner.write_outputs(records, metadata, jsonl, csv_path, summary)
            jsonl_records = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
            csv_text = csv_path.read_text(encoding="utf-8")
            self.assertEqual([item["record_type"] for item in jsonl_records], ["metadata", "summary", "observation"])
            self.assertIn("question_id", csv_text)
            self.assertIn("q1", csv_text)
            self.assertEqual(json.loads(summary.read_text(encoding="utf-8"))["summary"]["measured_trials"], 1)

    def test_http_error_is_recorded(self):
        error = urllib.error.HTTPError("url", 422, "invalid", {}, io.BytesIO(b'{"error":"bad request"}'))
        with mock.patch.object(benchmark_runner, "post_json", side_effect=error):
            result = benchmark_runner.request_json("http://example", {"questions": {}}, 1)
        self.assertEqual(result["status_code"], 422)
        self.assertEqual(result["response"], {"error": "bad request"})
        self.assertEqual(result["error"], "HTTP 422")
        self.assertGreaterEqual(result["wall_clock_ms"], 0)

    def test_observation_contains_reproducibility_fields(self):
        item = {"id": "q1", "description": "test", "request": {"state": "x"}}
        result = {"response": {"model": "q8"}, "timestamp_utc": "now", "model": "q8",
                  "wall_clock_ms": 12.5, "status_code": 200, "response_headers": {}, "error": None}
        observation = benchmark_runner.make_observation(item, result, "run", 0, False)
        required = {"question_id", "request_payload", "response", "timestamp_utc", "model", "wall_clock_ms", "is_warmup"}
        self.assertTrue(required <= observation.keys())

    def test_randomized_schedule_is_complete_and_reproducible(self):
        dataset = [{"id": f"q{index}"} for index in range(20)]
        first = benchmark_runner.randomized_schedule(dataset, 10, 20261024)
        second = benchmark_runner.randomized_schedule(dataset, 10, 20261024)
        self.assertEqual(first, second)
        self.assertTrue(all(len(block["question_ids"]) == 20 for block in first))
        self.assertTrue(all(len(set(block["question_ids"])) == 20 for block in first))
        self.assertEqual({question_id: sum(question_id in block["question_ids"] for block in first)
                          for question_id in [item["id"] for item in dataset]}, {f"q{index}": 10 for index in range(20)})

    def test_randomized_schedule_changes_with_seed(self):
        dataset = [{"id": f"q{index}"} for index in range(20)]
        self.assertNotEqual(benchmark_runner.randomized_schedule(dataset, 10, 20261024),
                            benchmark_runner.randomized_schedule(dataset, 10, 20261025))

    def test_summary_statistics_excludes_warmups(self):
        records = [record("warmup", 1, warmup=True), record("q1", 100), record("q2", 200),
                   record("q3", 300, status=500)]
        summary = benchmark_runner.summary_statistics(records)
        self.assertEqual(summary["measured_trials"], 3)
        self.assertEqual(summary["failure_rate"], 1 / 3)
        self.assertEqual(summary["median_ms"], 200)
        self.assertEqual(summary["p90_ms"], 280)
        self.assertEqual(summary["p95_ms"], 290)
        self.assertAlmostEqual(summary["mean_ms"], 200)
        self.assertAlmostEqual(summary["standard_deviation_ms"], 81.64965809277261)

    def test_response_probabilities_and_keys(self):
        request = {
            "questions": {
                "choice": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
                "urgent": {"type": "noul"},
                "severity": {"type": "score", "criteria": ["low", "high"]},
            }
        }
        response = {"answers": {
            "choice": {"probabilities": {"a": 0.25, "b": 0.75}},
            "urgent": {"noul": 0.4},
            "severity": {"probabilities": {"0": 0.6, "1": 0.4}},
        }}
        self.assertTrue(benchmark_runner.validate_response(request, response))
        response["answers"]["choice"]["probabilities"]["b"] = 0.7
        with self.assertRaises(ValueError):
            benchmark_runner.validate_response(request, response)


if __name__ == "__main__":
    unittest.main()

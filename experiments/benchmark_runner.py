import argparse
import csv
import json
import os
import platform
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def collect_hardware():
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


def get_json(url, timeout):
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode("utf-8")), dict(response.headers)


def post_json(url, payload, timeout):
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST", headers={
        "Accept": "application/json",
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode("utf-8")), dict(response.headers)


def request_json(url, payload, timeout):
    started = time.perf_counter_ns()
    timestamp = utc_now()
    try:
        status, response, headers = post_json(url, payload, timeout)
        error = None
    except urllib.error.HTTPError as exc:
        status = exc.code
        try:
            response = json.loads(exc.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            response = None
        headers = dict(exc.headers)
        error = f"HTTP {exc.code}"
    except (OSError, ValueError) as exc:
        status = None
        response = None
        headers = {}
        error = f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    response = response if isinstance(response, dict) else None
    return {
        "timestamp_utc": timestamp,
        "status_code": status,
        "wall_clock_ms": round(elapsed_ms, 3),
        "response_headers": headers,
        "response": response,
        "model": response.get("model") if response else None,
        "error": error,
    }


def read_dataset(path):
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("id") or record.get("condition_id"), str) or not isinstance(record.get("request"), dict):
                raise ValueError(f"invalid dataset record on line {line_number}")
            if "id" not in record:
                record["id"] = record["condition_id"]
            records.append(record)
    if not records:
        raise ValueError("dataset is empty")
    return records


def randomized_schedule(dataset, blocks, seed):
    if blocks < 1:
        raise ValueError("blocks must be positive")
    ids = [item["id"] for item in dataset]
    if len(set(ids)) != len(ids):
        raise ValueError("dataset question IDs must be unique")
    generator = random.Random(seed)
    schedule = []
    for block_number in range(1, blocks + 1):
        order = ids[:]
        generator.shuffle(order)
        schedule.append({"block_number": block_number, "question_ids": order})
    return schedule


def metadata(base_url, timeout):
    result = {}
    for name, path in (("health", "/health"), ("models", "/v1/models")):
        try:
            status, body, headers = get_json(base_url.rstrip("/") + path, timeout)
            result[name] = {"status_code": status, "response": body, "response_headers": headers}
        except (OSError, ValueError, urllib.error.HTTPError) as exc:
            result[name] = {"status_code": None, "response": None, "response_headers": {},
                            "error": f"{type(exc).__name__}: {exc}"}
    return result


def validate_response(request, response, tolerance=1e-6):
    if not isinstance(response, dict) or not isinstance(response.get("answers"), dict):
        raise ValueError("response does not contain an answers object")
    questions = request.get("questions") or {}
    answers = response["answers"]
    if set(answers) != set(questions):
        raise ValueError(f"answer keys {set(answers)} do not match question keys {set(questions)}")
    for key, question in questions.items():
        answer = answers[key]
        kind = question.get("type")
        probabilities = answer.get("probabilities")
        if kind == "choice":
            expected = set((question.get("criteria") or {}).keys())
            if set(probabilities or {}) != expected:
                raise ValueError(f"{key} probability keys do not match choice criteria")
        elif kind == "score":
            expected = {str(index) for index in range(len(question.get("criteria") or []))}
            if set(probabilities or {}) != expected:
                raise ValueError(f"{key} probability keys do not match score criteria")
        elif kind in ("noul", "bool"):
            probability = answer.get("noul")
            if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
                raise ValueError(f"{key} noul probability is invalid")
            continue
        else:
            raise ValueError(f"unsupported question type {kind}")
        total = sum(float(value) for value in probabilities.values())
        if abs(total - 1.0) > tolerance:
            raise ValueError(f"{key} probabilities sum to {total}")
        if any(not isinstance(value, (int, float)) or value < 0 or value > 1 for value in probabilities.values()):
            raise ValueError(f"{key} contains an invalid probability")
    return True


def make_observation(item, result, run_id, trial_index, is_warmup, block_number=None, block_position=None):
    return {
        "record_type": "observation",
        "run_id": run_id,
        "question_id": item["id"],
        "condition_id": item.get("condition_id", item["id"]),
        "parent_question_id": item.get("parent_question_id"),
        "variant_type": item.get("variant_type", "control"),
        "original_latency_regime": item.get("original_latency_regime"),
        "expected_answer": item.get("expected_answer"),
        "description": item.get("description", ""),
        "ground_truth_status": item.get("ground_truth_status", "unavailable"),
        "ground_truth": item.get("ground_truth"),
        "trial_index": trial_index,
        "block_number": block_number,
        "block_position": block_position,
        "is_warmup": is_warmup,
        "request_payload": item["request"],
        "response": result["response"],
        "timestamp_utc": result["timestamp_utc"],
        "model": result["model"],
        "wall_clock_ms": result["wall_clock_ms"],
        "status_code": result["status_code"],
        "response_headers": result["response_headers"],
        "error": result["error"],
    }


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summary_statistics(records):
    measured = [record for record in records if not record["is_warmup"]]
    latencies = [record["wall_clock_ms"] for record in measured if record["wall_clock_ms"] is not None]
    failures = sum(record["status_code"] != 200 for record in measured)
    summary = {
        "measured_trials": len(measured),
        "successful_trials": len(measured) - failures,
        "failure_count": failures,
        "failure_rate": failures / len(measured) if measured else None,
        "latency_sample_count": len(latencies),
        "median_ms": percentile(latencies, 0.50),
        "p90_ms": percentile(latencies, 0.90),
        "p95_ms": percentile(latencies, 0.95),
        "mean_ms": statistics.mean(latencies) if latencies else None,
        "standard_deviation_ms": statistics.pstdev(latencies) if latencies else None,
    }
    summary.update(decision_summary(measured))
    return summary


def decision_value(answer):
    if not isinstance(answer, dict):
        return None
    for key in ("choice", "noul", "score"):
        if key in answer:
            return answer[key]
    return None


def decision_summary(records):
    groups = {}
    for record in records:
        group = groups.setdefault(record["question_id"], {"records": [], "ground_truth": record.get("ground_truth"),
                                                            "ground_truth_status": record.get("ground_truth_status", "unavailable")})
        group["records"].append(record)
    per_question = {}
    correct = 0
    labeled_trials = 0
    for question_id, group in groups.items():
        successful = [record for record in group["records"] if record["status_code"] == 200]
        values = []
        correct_for_question = 0
        truth = group["ground_truth"] if group["ground_truth_status"] == "available" else None
        for record in successful:
            answers = (record.get("response") or {}).get("answers") or {}
            question_values = {key: decision_value(answers.get(key)) for key in (truth or answers)}
            values.append(json.dumps(question_values, sort_keys=True, separators=(",", ":")))
            if truth is not None:
                labeled_trials += 1
                if all(decision_value(answers.get(key)) == expected for key, expected in truth.items()):
                    correct_for_question += 1
        counts = {value: values.count(value) for value in set(values)}
        modal_count = max(counts.values(), default=0)
        correct += correct_for_question
        per_question[question_id] = {
            "ground_truth_status": group["ground_truth_status"],
            "ground_truth": truth,
            "measured_trials": len(group["records"]),
            "successful_trials": len(successful),
            "failure_count": len(group["records"]) - len(successful),
            "decision_stability": modal_count / len(values) if values else None,
            "modal_decision": json.loads(max(counts, key=counts.get)) if counts else None,
            "labeled_correct_trials": correct_for_question if truth is not None else None,
            "labeled_accuracy": correct_for_question / len(successful) if truth is not None and successful else None,
        }
    return {
        "per_question": per_question,
        "labeled_accuracy": correct / labeled_trials if labeled_trials else None,
        "labeled_correct_trials": correct,
        "labeled_trials": labeled_trials,
        "label_source": "manually constructed dataset labels, not independently validated",
    }


def csv_row(record):
    response = record.get("response") or {}
    usage = response.get("usage") or {}
    return {
        "run_id": record["run_id"],
        "question_id": record["question_id"],
        "condition_id": record.get("condition_id", record["question_id"]),
        "parent_question_id": record.get("parent_question_id"),
        "variant_type": record.get("variant_type", "control"),
        "original_latency_regime": record.get("original_latency_regime"),
        "description": record["description"],
        "trial_index": record["trial_index"],
        "block_number": record["block_number"],
        "block_position": record["block_position"],
        "is_warmup": record["is_warmup"],
        "timestamp_utc": record["timestamp_utc"],
        "status_code": record["status_code"],
        "wall_clock_ms": record["wall_clock_ms"],
        "model": record["model"],
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "request_payload_json": json.dumps(record["request_payload"], separators=(",", ":")),
        "response_json": json.dumps(record["response"], separators=(",", ":")),
        "error": record["error"],
    }


def write_outputs(records, metadata_record, jsonl_path, csv_path, summary_path):
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summary_statistics(records)
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"record_type": "metadata", **metadata_record}, separators=(",", ":")) + "\n")
        stream.write(json.dumps({"record_type": "summary", "run_id": metadata_record["run_id"],
                                 "summary": summary}, separators=(",", ":")) + "\n")
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    fields = list(csv_row(records[0]))
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_row(record) for record in records)
    with summary_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump({"run_id": metadata_record["run_id"], "summary": summary}, stream, separators=(",", ":"))
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8090/v1/systemone")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-trials", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--schedule-file", type=Path)
    parser.add_argument("--hardware-file", type=Path)
    parser.add_argument("--llama-executable")
    parser.add_argument("--llama-version")
    parser.add_argument("--llama-arguments", default="unknown: process command-line access denied")
    parser.add_argument("--llama-model-path", default="unknown: process command-line access denied")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup_trials < 0 or args.blocks < 0:
        raise ValueError("repeats must be positive and warmup-trials cannot be negative")
    if args.blocks and args.seed is None:
        raise ValueError("seed is required for randomized blocks")
    if args.blocks and args.schedule_file is None:
        raise ValueError("schedule-file is required for randomized blocks")
    dataset = read_dataset(args.dataset)
    base_url = args.endpoint.removesuffix("/v1/systemone").rstrip("/")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    started = utc_now()
    server_metadata = metadata(base_url, args.timeout)
    hardware = collect_hardware()
    if args.hardware_file:
        hardware = json.loads(args.hardware_file.read_text(encoding="utf-8"))
    schedule = randomized_schedule(dataset, args.blocks, args.seed) if args.blocks else None
    if schedule:
        args.schedule_file.parent.mkdir(parents=True, exist_ok=True)
        args.schedule_file.write_text(json.dumps({"seed": args.seed, "blocks": schedule}, indent=2) + "\n",
                                      encoding="utf-8")
    records = []
    trial_index = 0
    for item in dataset:
        for _ in range(args.warmup_trials):
            result = request_json(args.endpoint, item["request"], args.timeout)
            records.append(make_observation(item, result, run_id, trial_index, True))
            trial_index += 1
    if schedule:
        items = {item["id"]: item for item in dataset}
        for block in schedule:
            for position, question_id in enumerate(block["question_ids"], 1):
                item = items[question_id]
                result = request_json(args.endpoint, item["request"], args.timeout)
                records.append(make_observation(item, result, run_id, trial_index, False,
                                                 block["block_number"], position))
                trial_index += 1
    else:
        for item in dataset:
            for _ in range(args.repeats):
                result = request_json(args.endpoint, item["request"], args.timeout)
                records.append(make_observation(item, result, run_id, trial_index, False))
                trial_index += 1
    metadata_record = {
        "record_type": "metadata",
        "run_id": run_id,
        "endpoint": args.endpoint,
        "dataset": str(args.dataset),
        "repeats": args.repeats,
        "blocks": args.blocks,
        "seed": args.seed,
        "schedule_file": str(args.schedule_file) if args.schedule_file else None,
        "warmup_trials_per_question": args.warmup_trials,
        "measured_trials": len(dataset) * args.blocks if args.blocks else len(dataset) * args.repeats,
        "experiment_started_utc": started,
        "server_metadata": server_metadata,
        "hardware": hardware,
        "llama_cpp": {
            "executable": args.llama_executable or "unknown",
            "version": args.llama_version or "unknown",
            "arguments": args.llama_arguments,
            "model_path": args.llama_model_path,
        },
    }
    ended = utc_now()
    metadata_record["experiment_ended_utc"] = ended
    write_outputs(records, metadata_record, args.jsonl, args.csv, args.summary)
    summary = summary_statistics(records)
    print(json.dumps({"run_id": run_id, "trials": len(records), "summary": summary,
                      "jsonl": str(args.jsonl), "csv": str(args.csv), "summary_file": str(args.summary)}))
    return 1 if summary["failure_count"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(2)

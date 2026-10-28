import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from q8_analysis import js_divergence, load_dataset, quantiles, read_jsonl, validate_records


THRESHOLD_MS = 1380.289


def measured_rows(path):
    return [row for row in read_jsonl(path) if row.get("record_type") == "observation" and not row["is_warmup"]]


def compare_latency(baseline, randomized):
    baseline_values = [row["wall_clock_ms"] for row in baseline]
    randomized_values = [row["wall_clock_ms"] for row in randomized]
    questions = sorted({row["question_id"] for row in baseline})
    per_question = {}
    for question_id in questions:
        first = [row["wall_clock_ms"] for row in baseline if row["question_id"] == question_id]
        second = [row["wall_clock_ms"] for row in randomized if row["question_id"] == question_id]
        per_question[question_id] = {
            "baseline": quantiles(first),
            "randomized": quantiles(second),
            "median_difference_ms": float(np.median(second) - np.median(first)),
        }
    slow_baseline = {question_id: sum(row["wall_clock_ms"] >= THRESHOLD_MS for row in baseline
                                     if row["question_id"] == question_id) / 10 for question_id in questions}
    slow_randomized = {question_id: sum(row["wall_clock_ms"] >= THRESHOLD_MS for row in randomized
                                       if row["question_id"] == question_id) / 10 for question_id in questions}
    return {
        "threshold_ms": THRESHOLD_MS,
        "baseline": quantiles(baseline_values),
        "randomized": quantiles(randomized_values),
        "mean_difference_ms": float(np.mean(randomized_values) - np.mean(baseline_values)),
        "median_difference_ms": float(np.median(randomized_values) - np.median(baseline_values)),
        "per_question": per_question,
        "slow_fraction_baseline": slow_baseline,
        "slow_fraction_randomized": slow_randomized,
        "baseline_slow_questions": [question_id for question_id, value in slow_baseline.items() if value == 1.0],
        "randomized_slow_questions": [question_id for question_id, value in slow_randomized.items() if value == 1.0],
        "switched_questions": [question_id for question_id in questions if slow_baseline[question_id] != slow_randomized[question_id]],
        "baseline_slow_count": sum(row["wall_clock_ms"] >= THRESHOLD_MS for row in baseline),
        "randomized_slow_count": sum(row["wall_clock_ms"] >= THRESHOLD_MS for row in randomized),
    }


def randomized_order_stats(rows):
    latencies = np.asarray([row["wall_clock_ms"] for row in rows], dtype=float)
    positions = np.asarray([row["block_position"] for row in rows], dtype=float)
    blocks = np.asarray([row["block_number"] for row in rows], dtype=float)
    order = np.arange(1, len(rows) + 1, dtype=float)
    def correlation(other):
        return float(np.corrcoef(other, latencies)[0, 1])
    by_position = {str(position): quantiles([row["wall_clock_ms"] for row in rows if row["block_position"] == position])
                   for position in range(1, 21)}
    by_block = {str(block): quantiles([row["wall_clock_ms"] for row in rows if row["block_number"] == block])
                for block in range(1, 11)}
    return {
        "chronological_correlation": correlation(order),
        "block_position_correlation": correlation(positions),
        "block_number_correlation": correlation(blocks),
        "by_position": by_position,
        "by_block": by_block,
    }


def additive_model(rows):
    questions = sorted({row["question_id"] for row in rows})
    question_index = {question_id: index for index, question_id in enumerate(questions[1:], 1)}
    matrix = []
    response = []
    for row in rows:
        vector = [1.0]
        vector.extend(1.0 if row["question_id"] == question_id else 0.0 for question_id in questions[1:])
        vector.extend([float(row["block_position"]), float(row["block_number"])])
        matrix.append(vector)
        response.append(row["wall_clock_ms"])
    design = np.asarray(matrix)
    target = np.asarray(response)
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ coefficients
    residuals = target - fitted
    degrees = len(target) - rank
    residual_variance = float((residuals @ residuals) / degrees)
    covariance = residual_variance * np.linalg.pinv(design.T @ design)
    standard_errors = np.sqrt(np.diag(covariance))
    total_variance = float(((target - target.mean()) @ (target - target.mean())))
    explained = float(1 - (residuals @ residuals) / total_variance)
    question_effects = {questions[0]: 0.0}
    for index, question_id in enumerate(questions[1:], 1):
        question_effects[question_id] = float(coefficients[index])
    return {
        "model": "latency_ms ~ question_identity + block_position + block_number",
        "observations": len(target),
        "rank": int(rank),
        "residual_degrees_of_freedom": int(degrees),
        "r_squared": explained,
        "residual_standard_deviation_ms": float(np.sqrt(residual_variance)),
        "question_effect_range_ms": float(max(question_effects.values()) - min(question_effects.values())),
        "question_effects_relative_to_reference_ms": question_effects,
        "position_coefficient_ms_per_position": float(coefficients[-2]),
        "position_standard_error": float(standard_errors[-2]),
        "block_coefficient_ms_per_block": float(coefficients[-1]),
        "block_standard_error": float(standard_errors[-1]),
        "interpretation": "descriptive fixed-effect decomposition; repeated observations are not independent questions",
    }


def decision_comparison(baseline, randomized, dataset):
    result = {}
    for question_id in sorted(dataset):
        base = [row for row in baseline if row["question_id"] == question_id]
        rand = [row for row in randomized if row["question_id"] == question_id]
        base_answers = [(row["response"] or {}).get("answers", {}) for row in base]
        rand_answers = [(row["response"] or {}).get("answers", {}) for row in rand]
        key = next(iter(dataset[question_id]["request"]["questions"]))
        base_values = [answer.get(key, {}).get("choice") for answer in base_answers]
        rand_values = [answer.get(key, {}).get("choice") for answer in rand_answers]
        base_probability = (base_answers[0].get(key) or {}).get("probabilities", {})
        rand_probability = (rand_answers[0].get(key) or {}).get("probabilities", {})
        labels = sorted(set(base_probability) | set(rand_probability))
        first = np.asarray([base_probability.get(label, 0) for label in labels], dtype=float)
        second = np.asarray([rand_probability.get(label, 0) for label in labels], dtype=float)
        result[question_id] = {
            "decision_agreement_across_runs": all(value == base_values[0] for value in rand_values + base_values[1:]),
            "baseline_decision": {str(value): base_values.count(value) for value in set(base_values)},
            "randomized_decision": {str(value): rand_values.count(value) for value in set(rand_values)},
            "cross_run_js_divergence_bits": js_divergence(first, second),
            "cross_run_max_probability_shift": float(np.max(np.abs(first - second))),
        }
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(path, run_id, files, counts):
    manifest = {"run_id": run_id, "record_counts": counts, "files": {}}
    for file_path in files:
        manifest["files"][str(file_path)] = {"sha256": sha256(file_path), "bytes": file_path.stat().st_size}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def svg_start(width, height, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<style>text{font-family:Arial,sans-serif;fill:#202124}.title{font-size:22px;font-weight:700}.axis{font-size:14px}.tick{font-size:11px}.caption{font-size:12px;fill:#4a4a4a}.grid{stroke:#d9dde3;stroke-width:1}.axisline{stroke:#202124;stroke-width:1.5}</style>',
            f'<text x="{width / 2}" y="32" text-anchor="middle" class="title">{title}</text>']


def line(items, x1, y1, x2, y2, style="axisline"):
    items.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" class="{style}"/>')


def text(items, x, y, value, cls="axis", anchor="middle", rotate=None):
    transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate else ""
    items.append(f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" class="{cls}"{transform}>{value}</text>')


def finish(items, path):
    items.append("</svg>")
    path.write_text("\n".join(items), encoding="utf-8")


def comparison_distribution(baseline, randomized, path):
    width, height, left, right, top, bottom = 1100, 650, 85, 35, 65, 75
    items = svg_start(width, height, "Original versus randomized Q8 latency")
    values = [row["wall_clock_ms"] for row in baseline + randomized]
    bins = np.linspace(min(values), max(values), 21)
    first, _ = np.histogram([row["wall_clock_ms"] for row in baseline], bins=bins)
    second, _ = np.histogram([row["wall_clock_ms"] for row in randomized], bins=bins)
    ymax = max(max(first), max(second))
    xscale = lambda value: left + (value - bins[0]) / (bins[-1] - bins[0]) * (width - left - right)
    yscale = lambda value: height - bottom - value / ymax * (height - top - bottom)
    for count in range(0, ymax + 1, max(1, int(np.ceil(ymax / 5)))):
        y = yscale(count)
        line(items, left, y, width - right, y, "grid")
        text(items, left - 10, y + 4, count, "tick", "end")
    for index, count in enumerate(first):
        x1, x2 = xscale(bins[index]), xscale(bins[index + 1])
        y = yscale(count)
        items.append(f'<rect x="{x1 + 1:.2f}" y="{y:.2f}" width="{max(1, x2 - x1 - 2):.2f}" height="{height - bottom - y:.2f}" fill="#2563eb" opacity="0.42"/>')
    for index, count in enumerate(second):
        x1, x2 = xscale(bins[index]), xscale(bins[index + 1])
        y = yscale(count)
        items.append(f'<rect x="{x1 + 1:.2f}" y="{y:.2f}" width="{max(1, x2 - x1 - 2):.2f}" height="{height - bottom - y:.2f}" fill="#dc2626" opacity="0.42"/>')
    line(items, left, height - bottom, width - right, height - bottom)
    line(items, left, top, left, height - bottom)
    for value in np.linspace(bins[0], bins[-1], 6):
        x = xscale(value)
        text(items, x, height - bottom + 23, f"{value:.0f}", "tick")
    text(items, (left + width - right) / 2, height - 22, "Wall-clock latency (ms)")
    text(items, 20, (top + height - bottom) / 2, "Observations", "axis", rotate=-90)
    text(items, width - right, top + 10, "blue: baseline, red: randomized", "caption", "end")
    finish(items, path)


def question_comparison(comparison, path):
    width, height, left, right, top, bottom = 1250, 900, 190, 35, 65, 80
    items = svg_start(width, height, "Per-question median latency: baseline versus randomized")
    questions = sorted(comparison["per_question"])
    maximum = max(max(value["baseline"]["p95_ms"], value["randomized"]["p95_ms"]) for value in comparison["per_question"].values())
    xscale = lambda value: left + value / maximum * (width - left - right)
    yscale = lambda index: top + index * (height - top - bottom) / (len(questions) - 1)
    for value in np.linspace(0, maximum, 6):
        x = xscale(value)
        line(items, x, top, x, height - bottom, "grid")
        text(items, x, height - bottom + 22, f"{value:.0f}", "tick")
    for index, question_id in enumerate(questions):
        row = comparison["per_question"][question_id]
        y = yscale(index)
        text(items, left - 12, y + 4, question_id[:4], "tick", "end")
        for key, color, offset in (("baseline", "#2563eb", -5), ("randomized", "#dc2626", 5)):
            line(items, xscale(row[key]["median_ms"]), y + offset, xscale(row[key]["p95_ms"]), y + offset, "axisline")
            items.append(f'<circle cx="{xscale(row[key]["median_ms"]):.2f}" cy="{y + offset:.2f}" r="5" fill="{color}"/>')
    line(items, left, top, left, height - bottom)
    line(items, left, height - bottom, width - right, height - bottom)
    text(items, (left + width - right) / 2, height - 22, "Wall-clock latency (ms)")
    text(items, 22, (top + height - bottom) / 2, "Question ID", "axis", rotate=-90)
    text(items, width - right, top + 10, "blue: baseline, red: randomized; points median, bars to p95", "caption", "end")
    finish(items, path)


def trial_order_plot(rows, path):
    width, height, left, right, top, bottom = 1100, 650, 75, 35, 65, 75
    items = svg_start(width, height, "Randomized Q8 latency versus chronological trial order")
    values = np.asarray([row["wall_clock_ms"] for row in rows])
    ymax = max(values) * 1.05
    xscale = lambda value: left + (value - 1) / 199 * (width - left - right)
    yscale = lambda value: height - bottom - value / ymax * (height - top - bottom)
    for value in np.linspace(0, ymax, 6):
        y = yscale(value)
        line(items, left, y, width - right, y, "grid")
        text(items, left - 10, y + 4, f"{value:.0f}", "tick", "end")
    for index, value in enumerate(values, 1):
        color = "#dc2626" if value >= THRESHOLD_MS else "#2563eb"
        items.append(f'<circle cx="{xscale(index):.2f}" cy="{yscale(value):.2f}" r="3" fill="{color}"/>')
    line(items, left, height - bottom, width - right, height - bottom)
    line(items, left, top, left, height - bottom)
    text(items, (left + width - right) / 2, height - 22, "Measured trial order")
    text(items, 18, (top + height - bottom) / 2, "Wall-clock latency (ms)", "axis", rotate=-90)
    text(items, width - right, top + 10, "red: exploratory slow regime", "caption", "end")
    finish(items, path)


def position_plot(rows, path):
    width, height, left, right, top, bottom = 1100, 650, 75, 35, 65, 75
    items = svg_start(width, height, "Randomized block position versus latency")
    ymax = max(row["wall_clock_ms"] for row in rows) * 1.05
    xscale = lambda value: left + (value - 1) / 19 * (width - left - right)
    yscale = lambda value: height - bottom - value / ymax * (height - top - bottom)
    for value in np.linspace(0, ymax, 6):
        y = yscale(value)
        line(items, left, y, width - right, y, "grid")
        text(items, left - 10, y + 4, f"{value:.0f}", "tick", "end")
    for row in rows:
        color = "#dc2626" if row["wall_clock_ms"] >= THRESHOLD_MS else "#2563eb"
        items.append(f'<circle cx="{xscale(row["block_position"]):.2f}" cy="{yscale(row["wall_clock_ms"]):.2f}" r="3" fill="{color}"/>')
    line(items, left, height - bottom, width - right, height - bottom)
    line(items, left, top, left, height - bottom)
    for value in range(1, 21, 2):
        text(items, xscale(value), height - bottom + 23, value, "tick")
    text(items, (left + width - right) / 2, height - 22, "Position within randomized block")
    text(items, 18, (top + height - bottom) / 2, "Wall-clock latency (ms)", "axis", rotate=-90)
    finish(items, path)


def regime_heatmap(baseline, randomized, path):
    width, height, left, right, top, bottom = 900, 800, 175, 35, 75, 70
    items = svg_start(width, height, "Fast/slow regime persistence by question")
    questions = sorted({row["question_id"] for row in baseline})
    cell_width = (width - left - right) / 2
    cell_height = (height - top - bottom) / len(questions)
    text(items, left + cell_width / 2, top - 18, "Baseline slow fraction", "axis")
    text(items, left + cell_width + cell_width / 2, top - 18, "Randomized slow fraction", "axis")
    for index, question_id in enumerate(questions):
        y = top + index * cell_height
        text(items, left - 10, y + cell_height / 2 + 4, question_id[:4], "tick", "end")
        for column, rows in enumerate((baseline, randomized)):
            fraction = sum(row["wall_clock_ms"] >= THRESHOLD_MS for row in rows if row["question_id"] == question_id) / 10
            color = "#b91c1c" if fraction == 1 else "#dbeafe" if fraction == 0 else "#f59e0b"
            x = left + column * cell_width
            items.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_width:.2f}" height="{cell_height:.2f}" fill="{color}" stroke="white"/>')
            text(items, x + cell_width / 2, y + cell_height / 2 + 4, f"{fraction:.1f}", "tick")
    text(items, width / 2, height - 20, f"Slow regime threshold: {THRESHOLD_MS:.3f} ms", "caption")
    finish(items, path)


def analyze(baseline_path, randomized_path, dataset_path, output_dir, figure_dir, csv_path, summary_path,
            schedule_path, manifest_path, baseline_csv_path, baseline_summary_path):
    dataset = load_dataset(dataset_path)
    baseline_rows = measured_rows(baseline_path)
    randomized_all = read_jsonl(randomized_path)
    _, _, randomized_observations, randomized_rows, _ = validate_records(randomized_all, dataset)
    comparison = compare_latency(baseline_rows, randomized_rows)
    order = randomized_order_stats(randomized_rows)
    model = additive_model(randomized_rows)
    decisions = decision_comparison(baseline_rows, randomized_rows, dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    report = {"baseline": str(baseline_path), "randomized": str(randomized_path), "run_id": randomized_all[0]["run_id"],
              "comparison": comparison, "randomized_order": order, "additive_model": model, "decision_comparison": decisions,
              "randomized_observations": len(randomized_observations)}
    (output_dir / "q8_randomized_analysis_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_manifest(manifest_path, report["run_id"], [randomized_path, csv_path, summary_path, schedule_path,
                                                     baseline_path, baseline_csv_path, baseline_summary_path],
                   {"randomized_observations": len(randomized_observations), "randomized_measured": len(randomized_rows),
                    "randomized_warmups": len(randomized_observations) - len(randomized_rows)})
    with (output_dir / "q8_randomized_per_question_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        rows = [{"question_id": question_id, **values} for question_id, values in comparison["per_question"].items()]
        flattened = []
        for row in rows:
            flattened.append({"question_id": row["question_id"], "baseline_median_ms": row["baseline"]["median_ms"],
                              "randomized_median_ms": row["randomized"]["median_ms"],
                              "median_difference_ms": row["median_difference_ms"],
                              "baseline_slow_fraction": comparison["slow_fraction_baseline"][row["question_id"]],
                              "randomized_slow_fraction": comparison["slow_fraction_randomized"][row["question_id"]]})
        writer = csv.DictWriter(stream, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)
    comparison_distribution(baseline_rows, randomized_rows, figure_dir / "q8_baseline_vs_randomized_latency.svg")
    question_comparison(comparison, figure_dir / "q8_baseline_vs_randomized_question_latency.svg")
    trial_order_plot(randomized_rows, figure_dir / "q8_randomized_trial_order_latency.svg")
    position_plot(randomized_rows, figure_dir / "q8_randomized_position_latency.svg")
    regime_heatmap(baseline_rows, randomized_rows, figure_dir / "q8_regime_persistence_heatmap.svg")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--randomized", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.baseline, args.randomized, args.dataset, args.tables, args.figures, args.csv, args.summary,
                     args.schedule, args.manifest, args.baseline_csv, args.baseline_summary)
    print(json.dumps({"randomized_observations": report["randomized_observations"],
                      "baseline_slow_questions": report["comparison"]["baseline_slow_questions"],
                      "randomized_slow_questions": report["comparison"]["randomized_slow_questions"]}, separators=(",", ":")))


if __name__ == "__main__":
    main()

import argparse
import csv
import hashlib
import itertools
import json
import statistics
from pathlib import Path

import numpy as np

from q8_analysis import js_divergence, quantiles, read_jsonl
from q8_perturbation import records as read_records


THRESHOLD_MS = 1380.289


def load(path):
    return [row for row in read_jsonl(path) if row.get("record_type") == "observation"]


def category_order():
    return ["control", "spelling", "grammar", "formatting", "paraphrase", "additional_detail"]


def answer(row):
    answers = (row.get("response") or {}).get("answers") or {}
    key = next(iter(answers), None)
    return answers.get(key) or {}


def selected(row):
    value = answer(row)
    return value.get("choice") if "choice" in value else value.get("score")


def distribution(row):
    value = answer(row)
    probabilities = value.get("probabilities")
    if not probabilities:
        return None, None
    labels = sorted(probabilities)
    return labels, np.asarray([probabilities[label] for label in labels], dtype=float)


def group(rows):
    return {(row["parent_question_id"], row["variant_type"]): sorted([item for item in rows
                                                                       if item["parent_question_id"] == row["parent_question_id"]
                                                                       and item["variant_type"] == row["variant_type"]],
                                                                      key=lambda item: item["block_number"])
            for row in rows}


def condition_summary(rows):
    result = {}
    for key, values in group(rows).items():
        latencies = [row["wall_clock_ms"] for row in values]
        token_values = [(row.get("response") or {}).get("usage", {}).get("input_tokens") for row in values]
        token_values = [value for value in token_values if value is not None]
        result[key] = {"parent_question_id": key[0], "variant_type": key[1], **quantiles(latencies),
                       "slow_regime_fraction": sum(value >= THRESHOLD_MS for value in latencies) / len(latencies),
                       "input_tokens_available": len(token_values),
                       "input_tokens_mean": statistics.mean(token_values) if token_values else None,
                       "input_tokens_values": sorted(set(token_values))}
    return result


def paired_comparison(rows, summaries):
    result = {}
    grouped = group(rows)
    parents = sorted({row["parent_question_id"] for row in rows})
    for parent in parents:
        control = grouped[(parent, "control")]
        control_by_block = {row["block_number"]: row for row in control}
        for variant in category_order()[1:]:
            values = grouped[(parent, variant)]
            latency_differences = []
            decision_changes = 0
            js_values = []
            probability_shifts = []
            confidence_differences = []
            token_differences = []
            for row in values:
                base = control_by_block[row["block_number"]]
                latency_differences.append(row["wall_clock_ms"] - base["wall_clock_ms"])
                decision_changes += selected(row) != selected(base)
                base_labels, base_distribution = distribution(base)
                labels, variant_distribution = distribution(row)
                if base_distribution is not None and labels == base_labels:
                    js_values.append(js_divergence(base_distribution, variant_distribution))
                    probability_shifts.append(float(np.max(np.abs(base_distribution - variant_distribution))))
                if answer(row).get("confidence") is not None and answer(base).get("confidence") is not None:
                    confidence_differences.append(answer(row)["confidence"] - answer(base)["confidence"])
                base_tokens = (base.get("response") or {}).get("usage", {}).get("input_tokens")
                variant_tokens = (row.get("response") or {}).get("usage", {}).get("input_tokens")
                if base_tokens is not None and variant_tokens is not None:
                    token_differences.append(variant_tokens - base_tokens)
            array = np.asarray(latency_differences, dtype=float)
            mean = float(np.mean(array))
            standard_error = float(np.std(array, ddof=1) / np.sqrt(len(array))) if len(array) > 1 else None
            result[(parent, variant)] = {
                "parent_question_id": parent,
                "variant_type": variant,
                "control_median_ms": summaries[(parent, "control")]["median_ms"],
                "variant_median_ms": summaries[(parent, variant)]["median_ms"],
                "median_difference_ms": summaries[(parent, variant)]["median_ms"] - summaries[(parent, "control")]["median_ms"],
                "median_percentage_difference": 100 * (summaries[(parent, variant)]["median_ms"] - summaries[(parent, "control")]["median_ms"]) / summaries[(parent, "control")]["median_ms"],
                "paired_mean_difference_ms": mean,
                "paired_difference_sd_ms": float(np.std(array, ddof=1)),
                "paired_difference_approx_95ci_ms": [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
                "decision_changes": decision_changes,
                "max_js_divergence_bits": max(js_values, default=0.0),
                "mean_js_divergence_bits": statistics.mean(js_values) if js_values else None,
                "max_probability_shift": max(probability_shifts, default=0.0),
                "mean_confidence_difference": statistics.mean(confidence_differences) if confidence_differences else None,
                "input_token_difference_values": sorted(set(token_differences)),
            }
    return result


def token_analysis(rows):
    tokens = [((row.get("response") or {}).get("usage") or {}).get("input_tokens") for row in rows]
    latencies = [row["wall_clock_ms"] for row, token in zip(rows, tokens) if token is not None]
    tokens = [token for token in tokens if token is not None]
    return {"available_count": len(tokens), "values": sorted(set(tokens)),
            "pearson_correlation_with_latency": float(np.corrcoef(np.asarray(tokens, dtype=float), np.asarray(latencies, dtype=float))[0, 1]) if len(set(tokens)) > 1 else None}


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


def condition_boxplot(summaries, path):
    width, height, left, right, top, bottom = 1500, 1000, 250, 35, 65, 100
    items = svg_start(width, height, "Pilot latency distributions by parent and perturbation")
    keys = sorted(summaries)
    maximum = max(summaries[key]["p95_ms"] for key in keys)
    xscale = lambda value: left + value / maximum * (width - left - right)
    yscale = lambda index: top + index * (height - top - bottom) / (len(keys) - 1)
    for value in np.linspace(0, maximum, 6):
        x = xscale(value)
        line(items, x, top, x, height - bottom, "grid")
        text(items, x, height - bottom + 22, f"{value:.0f}", "tick")
    colors = {"control": "#374151", "spelling": "#2563eb", "grammar": "#16a34a", "formatting": "#f59e0b", "paraphrase": "#7c3aed", "additional_detail": "#dc2626"}
    for index, key in enumerate(keys):
        row = summaries[key]
        y = yscale(index)
        text(items, left - 12, y + 4, f"{key[0][:4]} {key[1]}", "tick", "end")
        line(items, xscale(row["min_ms"]), y, xscale(row["max_ms"]), y)
        line(items, xscale(row["min_ms"]), y - 8, xscale(row["min_ms"]), y + 8)
        line(items, xscale(row["max_ms"]), y - 8, xscale(row["max_ms"]), y + 8)
        items.append(f'<rect x="{xscale(row["p50_ms"]):.2f}" y="{y - 10:.2f}" width="{max(1, xscale(row["p75_ms"]) - xscale(row["p25_ms"])):.2f}" height="20" fill="{colors[key[1]]}" opacity="0.7"/>')
        line(items, xscale(row["median_ms"]), y - 10, xscale(row["median_ms"]), y + 10)
    line(items, left, height - bottom, width - right, height - bottom)
    text(items, (left + width - right) / 2, height - 24, "Wall-clock latency (ms)")
    text(items, 20, (top + height - bottom) / 2, "Parent and perturbation", "axis", rotate=-90)
    finish(items, path)


def difference_plot(paired, path):
    width, height, left, right, top, bottom = 1450, 900, 250, 35, 65, 80
    items = svg_start(width, height, "Paired block latency difference from control")
    keys = sorted(paired)
    values = [paired[key]["paired_mean_difference_ms"] for key in keys]
    maximum = max(abs(value) for value in values) * 1.15
    xscale = lambda value: left + (value + maximum) / (2 * maximum) * (width - left - right)
    yscale = lambda index: top + index * (height - top - bottom) / (len(keys) - 1)
    line(items, xscale(0), top, xscale(0), height - bottom, "grid")
    for value in np.linspace(-maximum, maximum, 7):
        text(items, xscale(value), height - bottom + 22, f"{value:.0f}", "tick")
    for index, key in enumerate(keys):
        row = paired[key]
        y = yscale(index)
        text(items, left - 12, y + 4, f"{key[0][:4]} {key[1]}", "tick", "end")
        low, high = row["paired_difference_approx_95ci_ms"]
        line(items, xscale(low), y, xscale(high), y)
        items.append(f'<circle cx="{xscale(row["paired_mean_difference_ms"]):.2f}" cy="{y:.2f}" r="5" fill="#1d4ed8"/>')
    text(items, (left + width - right) / 2, height - 22, "Variant minus control latency (ms)")
    text(items, 20, (top + height - bottom) / 2, "Parent and perturbation", "axis", rotate=-90)
    finish(items, path)


def regime_heatmap(summaries, path):
    width, height, left, right, top, bottom = 1200, 700, 250, 35, 75, 70
    items = svg_start(width, height, "Fast/slow regime transition frequency")
    keys = sorted(summaries)
    cell_width = (width - left - right) / 6
    cell_height = (height - top - bottom) / len(keys)
    for index, key in enumerate(keys):
        y = top + index * cell_height
        text(items, left - 10, y + cell_height / 2 + 4, f"{key[0][:4]} {key[1]}", "tick", "end")
        for column, category in enumerate(category_order()):
            if category == key[1]:
                fraction = summaries[key]["slow_regime_fraction"]
            else:
                fraction = summaries[(key[0], category)]["slow_regime_fraction"]
            color = "#b91c1c" if fraction == 1 else "#dbeafe" if fraction == 0 else "#f59e0b"
            x = left + column * cell_width
            items.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_width:.2f}" height="{cell_height:.2f}" fill="{color}" stroke="white"/>')
            text(items, x + cell_width / 2, y + cell_height / 2 + 4, f"{fraction:.1f}", "tick")
    for index, category in enumerate(category_order()):
        text(items, left + index * cell_width + cell_width / 2, top - 15, category, "tick")
    text(items, width / 2, height - 20, f"Slow fraction at threshold {THRESHOLD_MS:.3f} ms", "caption")
    finish(items, path)


def token_plot(summaries, path):
    width, height, left, right, top, bottom = 1100, 650, 90, 35, 65, 75
    items = svg_start(width, height, "Latency versus reported input token count")
    points = [(row["input_tokens_mean"], row["mean_ms"]) for row in summaries.values() if row["input_tokens_mean"] is not None]
    x_values, y_values = zip(*points)
    xmin, xmax, ymin, ymax = min(x_values) - 1, max(x_values) + 1, 0, max(y_values) * 1.05
    xscale = lambda value: left + (value - xmin) / (xmax - xmin) * (width - left - right)
    yscale = lambda value: height - bottom - value / ymax * (height - top - bottom)
    for value in np.linspace(0, ymax, 6):
        y = yscale(value)
        line(items, left, y, width - right, y, "grid")
        text(items, left - 10, y + 4, f"{value:.0f}", "tick", "end")
    for key, row in summaries.items():
        x, y = xscale(row["input_tokens_mean"]), yscale(row["mean_ms"])
        items.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#2563eb"/>')
        text(items, x + 8, y - 8, f"{key[0][:4]} {key[1]}", "tick", "start")
    text(items, (left + width - right) / 2, height - 22, "Reported input tokens")
    text(items, 20, (top + height - bottom) / 2, "Mean latency (ms)", "axis", rotate=-90)
    finish(items, path)


def divergence_plot(paired, path):
    width, height, left, right, top, bottom = 1200, 650, 95, 35, 65, 75
    items = svg_start(width, height, "Probability divergence versus latency change")
    maximum = max(max(abs(row["paired_mean_difference_ms"]), 1) for row in paired.values())
    xscale = lambda value: left + (value + maximum) / (2 * maximum) * (width - left - right)
    yscale = lambda value: height - bottom - value * (height - top - bottom)
    line(items, xscale(0), top, xscale(0), height - bottom, "grid")
    for key, row in paired.items():
        items.append(f'<circle cx="{xscale(row["paired_mean_difference_ms"]):.2f}" cy="{yscale(row["max_js_divergence_bits"]):.2f}" r="5" fill="#7c3aed"/>')
    text(items, (left + width - right) / 2, height - 22, "Paired mean latency change (ms)")
    text(items, 20, (top + height - bottom) / 2, "Maximum JS divergence (bits)", "axis", rotate=-90)
    finish(items, path)


def trial_order_plot(rows, path):
    width, height, left, right, top, bottom = 1100, 650, 75, 35, 65, 75
    items = svg_start(width, height, "Perturbation pilot latency versus trial order")
    values = np.asarray([row["wall_clock_ms"] for row in rows])
    ymax = max(values) * 1.05
    xscale = lambda value: left + (value - 1) / 239 * (width - left - right)
    yscale = lambda value: height - bottom - value / ymax * (height - top - bottom)
    for value in np.linspace(0, ymax, 6):
        y = yscale(value)
        line(items, left, y, width - right, y, "grid")
        text(items, left - 10, y + 4, f"{value:.0f}", "tick", "end")
    for index, row in enumerate(rows, 1):
        color = "#dc2626" if row["wall_clock_ms"] >= THRESHOLD_MS else "#2563eb"
        items.append(f'<circle cx="{xscale(index):.2f}" cy="{yscale(row["wall_clock_ms"]):.2f}" r="3" fill="{color}"/>')
    text(items, (left + width - right) / 2, height - 22, "Measured trial order")
    text(items, 18, (top + height - bottom) / 2, "Wall-clock latency (ms)", "axis", rotate=-90)
    finish(items, path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze(raw_path, dataset_path, table_dir, figure_dir, csv_path, summary_path, schedule_path, manifest_path):
    rows = load(raw_path)
    dataset = {item["condition_id"]: item for item in read_records(dataset_path)}
    if len(rows) != 264 or sum(row["is_warmup"] for row in rows) != 24 or sum(not row["is_warmup"] for row in rows) != 240:
        raise ValueError("pilot observation counts are invalid")
    if any(row["status_code"] != 200 for row in rows if not row["is_warmup"]):
        raise ValueError("pilot has measured failures")
    measured = [row for row in rows if not row["is_warmup"]]
    summaries = condition_summary(measured)
    paired = paired_comparison(measured, summaries)
    token_report = token_analysis(measured)
    report = {"run_id": read_jsonl(raw_path)[0]["run_id"], "threshold_ms": THRESHOLD_MS,
              "record_counts": {"observations": len(rows), "warmups": 24, "measured": 240},
              "condition_summary": {f"{key[0]}::{key[1]}": value for key, value in summaries.items()},
              "paired_comparison": {f"{key[0]}::{key[1]}": value for key, value in paired.items()},
              "token_analysis": token_report,
              "failures": [row for row in rows if row["status_code"] != 200],
              "semantic_equivalence_status": {condition_id: item["semantic_equivalence"] for condition_id, item in dataset.items()}}
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "q8_perturbation_analysis_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (table_dir / "q8_perturbation_condition_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        flattened = [{"parent_question_id": key[0], "variant_type": key[1], **value} for key, value in summaries.items()]
        fields = ["parent_question_id", "variant_type", "count", "mean_ms", "median_ms", "stddev_ms", "min_ms", "max_ms", "p90_ms", "p95_ms", "slow_regime_fraction", "input_tokens_mean", "input_tokens_values"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flattened)
    with (table_dir / "q8_perturbation_paired_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        flattened = [{"parent_question_id": key[0], "variant_type": key[1], **value} for key, value in paired.items()]
        writer = csv.DictWriter(stream, fieldnames=list(flattened[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flattened)
    condition_boxplot(summaries, figure_dir / "q8_perturbation_latency_distributions.svg")
    difference_plot(paired, figure_dir / "q8_perturbation_control_differences.svg")
    regime_heatmap(summaries, figure_dir / "q8_perturbation_regime_heatmap.svg")
    token_plot(summaries, figure_dir / "q8_perturbation_latency_tokens.svg")
    divergence_plot(paired, figure_dir / "q8_perturbation_divergence_latency.svg")
    trial_order_plot([row for row in rows if not row["is_warmup"]], figure_dir / "q8_perturbation_trial_order.svg")
    manifest = {"run_id": report["run_id"], "files": {}}
    for path in (raw_path, csv_path, summary_path, schedule_path):
        manifest["files"][str(path)] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    (table_dir / "q8_perturbation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.raw, args.dataset, args.tables, args.figures, args.csv, args.summary, args.schedule,
                     args.tables / "q8_perturbation_manifest.json")
    print(json.dumps({"run_id": report["run_id"], "measured": report["record_counts"]["measured"],
                      "failures": len(report["failures"])}, separators=(",", ":")))


if __name__ == "__main__":
    main()

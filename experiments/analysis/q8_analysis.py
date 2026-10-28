import argparse
import csv
import html
import hashlib
import itertools
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np


REQUIRED_FIELDS = {"question_id", "request_payload", "response", "timestamp_utc", "model", "wall_clock_ms", "is_warmup"}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_dataset(path):
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            records[item["id"]] = item
    return records


def quantiles(values):
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean_ms": float(np.mean(array)),
        "median_ms": float(np.median(array)),
        "p25_ms": float(np.quantile(array, 0.25, method="linear")),
        "p75_ms": float(np.quantile(array, 0.75, method="linear")),
        "p50_ms": float(np.quantile(array, 0.50, method="linear")),
        "p90_ms": float(np.quantile(array, 0.90, method="linear")),
        "p95_ms": float(np.quantile(array, 0.95, method="linear")),
        "p99_ms": float(np.quantile(array, 0.99, method="linear")),
        "stddev_ms": float(np.std(array)),
        "min_ms": float(np.min(array)),
        "max_ms": float(np.max(array)),
    }


def js_divergence(first, second):
    p = np.asarray(first, dtype=float)
    q = np.asarray(second, dtype=float)
    if p.shape != q.shape or not np.isclose(p.sum(), 1) or not np.isclose(q.sum(), 1):
        raise ValueError("Jensen-Shannon inputs must be equal-length distributions summing to one")
    midpoint = (p + q) / 2
    p_terms = np.zeros_like(p)
    q_terms = np.zeros_like(q)
    p_mask = p > 0
    q_mask = q > 0
    p_terms[p_mask] = p[p_mask] * np.log2(p[p_mask] / midpoint[p_mask])
    q_terms[q_mask] = q[q_mask] * np.log2(q[q_mask] / midpoint[q_mask])
    return float((p_terms.sum() + q_terms.sum()) / 2)


def validate_records(rows, dataset):
    metadata = [row for row in rows if row.get("record_type") == "metadata"]
    summaries = [row for row in rows if row.get("record_type") == "summary"]
    observations = [row for row in rows if row.get("record_type") == "observation"]
    measured = [row for row in observations if not row["is_warmup"]]
    warmups = [row for row in observations if row["is_warmup"]]
    if len(metadata) != 1 or len(summaries) != 1:
        raise ValueError("expected exactly one metadata and one summary record")
    if len(observations) != 220 or len(warmups) != 20 or len(measured) != 200:
        raise ValueError("expected 20 warm-ups and 200 measured observations")
    if any(not REQUIRED_FIELDS <= row.keys() for row in observations):
        raise ValueError("observation is missing a required field")
    if any(row["status_code"] != 200 or row["model"] != "startlux-q8" for row in measured):
        raise ValueError("measured observations contain a failure or unexpected model")
    counts = {question_id: sum(row["question_id"] == question_id for row in measured)
              for question_id in dataset}
    if set(counts) != set(dataset) or any(value != 10 for value in counts.values()):
        raise ValueError("each dataset question must have ten measured observations")
    if any(row["wall_clock_ms"] is None for row in measured):
        raise ValueError("measured latency is missing")
    return metadata[0], summaries[0], observations, measured, warmups


def latency_analysis(measured):
    values = [row["wall_clock_ms"] for row in measured]
    by_question = {}
    for question_id in sorted({row["question_id"] for row in measured}):
        by_question[question_id] = quantiles([row["wall_clock_ms"] for row in measured
                                               if row["question_id"] == question_id])
    order = np.arange(1, len(values) + 1, dtype=float)
    latency = np.asarray(values, dtype=float)
    slope, intercept = np.polyfit(order, latency, 1)
    correlation = float(np.corrcoef(order, latency)[0, 1])
    sorted_values = np.sort(latency)
    gaps = np.diff(sorted_values)
    largest_gap_index = int(np.argmax(gaps))
    threshold = float((sorted_values[largest_gap_index] + sorted_values[largest_gap_index + 1]) / 2)
    low = latency < threshold
    slow_threshold = quantiles(values)["p95_ms"]
    slow = latency > slow_threshold
    slow_by_question = {question_id: int(sum(slow[index] for index, row in enumerate(measured)
                                           if row["question_id"] == question_id))
                        for question_id in sorted(by_question)}
    segments = {}
    for name, start, end in (("first_quarter", 0, 50), ("second_quarter", 50, 100),
                             ("third_quarter", 100, 150), ("fourth_quarter", 150, 200)):
        segments[name] = quantiles(values[start:end])
    server_latency = [(row.get("response") or {}).get("latency_ms") for row in measured]
    return {
        "overall": quantiles(values),
        "per_question": by_question,
        "chronological_order": {
            "pearson_correlation": correlation,
            "slope_ms_per_trial": float(slope),
            "intercept_ms": float(intercept),
            "quartile_segments": segments,
        },
        "slow_observations": {
            "threshold_ms": slow_threshold,
            "count": int(slow.sum()),
            "by_question": slow_by_question,
            "top_ten_ms": sorted([float(value) for value in latency], reverse=True)[:10],
        },
        "exploratory_two_regime_partition": {
            "method": "largest gap in sorted latency values",
            "threshold_ms": threshold,
            "lower_count": int(low.sum()),
            "upper_count": int((~low).sum()),
            "lower_mean_ms": float(np.mean(latency[low])),
            "upper_mean_ms": float(np.mean(latency[~low])),
        },
        "server_latency_ms": {
            "recorded_count": sum(value is not None for value in server_latency),
            "missing_count": sum(value is None for value in server_latency),
        },
    }


def answer_value(answer):
    if not isinstance(answer, dict):
        return None
    for key in ("choice", "score"):
        if key in answer:
            return answer[key]
    return None


def probability_analysis(measured, dataset):
    result = {}
    for question_id in sorted(dataset):
        rows = [row for row in measured if row["question_id"] == question_id]
        question = dataset[question_id]["request"]["questions"]
        comparable = []
        selected = []
        for row in rows:
            answers = (row.get("response") or {}).get("answers") or {}
            key = next(iter(question))
            answer = answers.get(key) or {}
            selected.append(answer_value(answer))
            probabilities = answer.get("probabilities")
            if probabilities:
                labels = sorted(probabilities)
                comparable.append((labels, np.asarray([probabilities[label] for label in labels], dtype=float)))
        if comparable and all(labels == comparable[0][0] for labels, _ in comparable):
            distributions = [values for _, values in comparable]
            js_values = [js_divergence(first, second) for first, second in itertools.combinations(distributions, 2)]
            shifts = [float(np.max(np.abs(first - second))) for first, second in itertools.combinations(distributions, 2)]
            exact = all(np.array_equal(distributions[0], values) for values in distributions[1:])
            probability_result = {
                "distribution_type": question[key]["type"],
                "labels": comparable[0][0],
                "comparable_repetitions": len(distributions),
                "exact_distribution_agreement": exact,
                "max_js_divergence_bits": max(js_values, default=0.0),
                "mean_pairwise_js_divergence_bits": statistics.mean(js_values) if js_values else 0.0,
                "max_probability_shift": max(shifts, default=0.0),
                "mean_pairwise_max_probability_shift": statistics.mean(shifts) if shifts else 0.0,
            }
        else:
            probability_result = {"distribution_type": question[key]["type"], "comparable_repetitions": 0,
                                  "reason": "no comparable categorical probability distributions"}
        selected_counts = {str(value): selected.count(value) for value in set(selected)}
        probability_result["selected_answer_agreement"] = max(selected_counts.values(), default=0) / len(selected)
        probability_result["selected_answers"] = selected_counts
        result[question_id] = probability_result
    return result


def write_tables(output_dir, latency, probability):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "q8_latency_summary.json").write_text(json.dumps(latency, indent=2) + "\n", encoding="utf-8")
    (output_dir / "q8_probability_stability.json").write_text(json.dumps(probability, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "q8_per_question_latency.csv").open("w", encoding="utf-8", newline="") as stream:
        rows = [{"question_id": question_id, **values} for question_id, values in latency["per_question"].items()]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "q8_probability_stability.csv").open("w", encoding="utf-8", newline="") as stream:
        rows = [{"question_id": question_id, **values} for question_id, values in probability.items()]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(path, raw_path, csv_path, summary_path, report):
    manifest = {
        "run_id": report["run_id"],
        "source_files": {
            str(raw_path): {"sha256": sha256(raw_path), "bytes": raw_path.stat().st_size},
            str(csv_path): {"sha256": sha256(csv_path), "bytes": csv_path.stat().st_size},
            str(summary_path): {"sha256": sha256(summary_path), "bytes": summary_path.stat().st_size},
        },
        "record_counts": report["record_counts"],
        "source_files_unchanged_during_analysis": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def svg_base(width, height, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<style>text{font-family:Arial,sans-serif;fill:#202124} .title{font-size:22px;font-weight:700} .axis{font-size:14px} .tick{font-size:11px} .caption{font-size:12px;fill:#4a4a4a} .grid{stroke:#d9dde3;stroke-width:1} .axisline{stroke:#202124;stroke-width:1.5}</style>',
            f'<text x="{width / 2}" y="32" text-anchor="middle" class="title">{html.escape(title)}</text>']


def svg_line(items, x1, y1, x2, y2, cls="axisline", color=None, width=None):
    extra = f' stroke="{color}"' if color else ''
    extra += f' stroke-width="{width}"' if width else ''
    items.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" class="{cls}"{extra}/>')


def svg_text(items, x, y, value, cls="axis", anchor="middle", rotate=None):
    transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate else ''
    items.append(f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" class="{cls}"{transform}>{html.escape(str(value))}</text>')


def finish_svg(items):
    items.append("</svg>")
    return "\n".join(items)


def plot_histogram(values, path):
    width, height, left, right, top, bottom = 1100, 650, 85, 35, 65, 75
    items = svg_base(width, height, "Q8 baseline end-to-end latency")
    bins = np.linspace(min(values), max(values), 21)
    counts, edges = np.histogram(values, bins=bins)
    max_count = max(counts)
    xscale = lambda value: left + (value - edges[0]) / (edges[-1] - edges[0]) * (width - left - right)
    yscale = lambda value: height - bottom - value / max_count * (height - top - bottom)
    for count in range(0, max_count + 1, max(1, math.ceil(max_count / 5))):
        y = yscale(count)
        svg_line(items, left, y, width - right, y, "grid")
        svg_text(items, left - 10, y + 4, count, "tick", "end")
    for index, count in enumerate(counts):
        x1, x2 = xscale(edges[index]), xscale(edges[index + 1])
        y = yscale(count)
        items.append(f'<rect x="{x1 + 1:.2f}" y="{y:.2f}" width="{max(1, x2 - x1 - 2):.2f}" height="{height - bottom - y:.2f}" fill="#3b82f6" opacity="0.82"/>')
    svg_line(items, left, height - bottom, width - right, height - bottom)
    svg_line(items, left, top, left, height - bottom)
    for value in np.linspace(edges[0], edges[-1], 6):
        x = xscale(value)
        svg_line(items, x, height - bottom, x, height - bottom + 6)
        svg_text(items, x, height - bottom + 23, f"{value:.0f}", "tick")
    svg_text(items, (left + width - right) / 2, height - 22, "Wall-clock latency (ms)")
    svg_text(items, 20, (top + height - bottom) / 2, "Observations", "axis", rotate=-90)
    svg_text(items, width - right, height - 22, "n = 200", "caption", "end")
    path.write_text(finish_svg(items), encoding="utf-8")


def plot_boxplot(measured, path):
    width, height, left, right, top, bottom = 1250, 900, 190, 35, 65, 80
    items = svg_base(width, height, "Latency distribution by question")
    questions = sorted({row["question_id"] for row in measured})
    max_value = max(row["wall_clock_ms"] for row in measured)
    xscale = lambda value: left + value / max_value * (width - left - right)
    yscale = lambda index: top + index * (height - top - bottom) / (len(questions) - 1)
    for value in np.linspace(0, max_value, 6):
        x = xscale(value)
        svg_line(items, x, top, x, height - bottom, "grid")
        svg_text(items, x, height - bottom + 22, f"{value:.0f}", "tick")
    for index, question_id in enumerate(questions):
        values = np.asarray([row["wall_clock_ms"] for row in measured if row["question_id"] == question_id])
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        y = yscale(index)
        svg_text(items, left - 12, y + 4, question_id[:4], "tick", "end")
        svg_line(items, xscale(values.min()), y, xscale(values.max()), y, "axisline")
        svg_line(items, xscale(values.min()), y - 8, xscale(values.min()), y + 8)
        svg_line(items, xscale(values.max()), y - 8, xscale(values.max()), y + 8)
        items.append(f'<rect x="{xscale(q1):.2f}" y="{y - 10:.2f}" width="{xscale(q3) - xscale(q1):.2f}" height="20" fill="#60a5fa" stroke="#1d4ed8"/>')
        svg_line(items, xscale(median), y - 10, xscale(median), y + 10, color="#111827", width=2)
    svg_line(items, left, top, left, height - bottom)
    svg_line(items, left, height - bottom, width - right, height - bottom)
    svg_text(items, (left + width - right) / 2, height - 22, "Wall-clock latency (ms)")
    svg_text(items, 24, (top + height - bottom) / 2, "Question ID", "axis", rotate=-90)
    path.write_text(finish_svg(items), encoding="utf-8")


def plot_order(measured, path):
    width, height, left, right, top, bottom = 1100, 650, 75, 35, 65, 75
    items = svg_base(width, height, "Latency versus chronological measured-trial order")
    values = np.asarray([row["wall_clock_ms"] for row in measured])
    xscale = lambda value: left + (value - 1) / 199 * (width - left - right)
    ymax = max(values) * 1.05
    yscale = lambda value: height - bottom - value / ymax * (height - top - bottom)
    for value in np.linspace(0, ymax, 6):
        y = yscale(value)
        svg_line(items, left, y, width - right, y, "grid")
        svg_text(items, left - 10, y + 4, f"{value:.0f}", "tick", "end")
    points = []
    for index, value in enumerate(values, 1):
        color = "#dc2626" if value > 2000 else "#2563eb"
        x, y = xscale(index), yscale(value)
        points.append(f"{x:.2f},{y:.2f}")
        items.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>')
    items.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#93c5fd" stroke-width="1" opacity="0.7"/>')
    svg_line(items, left, height - bottom, width - right, height - bottom)
    svg_line(items, left, top, left, height - bottom)
    for value in (1, 50, 100, 150, 200):
        x = xscale(value)
        svg_line(items, x, height - bottom, x, height - bottom + 6)
        svg_text(items, x, height - bottom + 23, value, "tick")
    svg_text(items, (left + width - right) / 2, height - 22, "Measured trial order")
    svg_text(items, 18, (top + height - bottom) / 2, "Wall-clock latency (ms)", "axis", rotate=-90)
    svg_text(items, width - right, top + 10, "red: > 2000 ms", "caption", "end")
    path.write_text(finish_svg(items), encoding="utf-8")


def plot_question_summary(latency, path):
    width, height, left, right, top, bottom = 1250, 900, 190, 35, 65, 80
    items = svg_base(width, height, "Per-question median latency with p95 indicators")
    questions = sorted(latency["per_question"])
    max_value = max(values["p95_ms"] for values in latency["per_question"].values())
    xscale = lambda value: left + value / max_value * (width - left - right)
    yscale = lambda index: top + index * (height - top - bottom) / (len(questions) - 1)
    for value in np.linspace(0, max_value, 6):
        x = xscale(value)
        svg_line(items, x, top, x, height - bottom, "grid")
        svg_text(items, x, height - bottom + 22, f"{value:.0f}", "tick")
    for index, question_id in enumerate(questions):
        values = latency["per_question"][question_id]
        y = yscale(index)
        svg_text(items, left - 12, y + 4, question_id[:4], "tick", "end")
        svg_line(items, xscale(values["median_ms"]), y, xscale(values["p95_ms"]), y, "axisline", "#dc2626", 4)
        items.append(f'<circle cx="{xscale(values["median_ms"]):.2f}" cy="{y:.2f}" r="6" fill="#1d4ed8"/>')
    svg_line(items, left, height - bottom, width - right, height - bottom)
    svg_line(items, left, top, left, height - bottom)
    svg_text(items, (left + width - right) / 2, height - 22, "Wall-clock latency (ms)")
    svg_text(items, 22, (top + height - bottom) / 2, "Question ID", "axis", rotate=-90)
    svg_text(items, width - right, top + 10, "blue: median, red: median-to-p95", "caption", "end")
    path.write_text(finish_svg(items), encoding="utf-8")


def plot_probability_stability(probability, path):
    width, height, left, right, top, bottom = 1250, 850, 190, 35, 65, 80
    items = svg_base(width, height, "Probability-distribution stability across repetitions")
    questions = sorted(probability)
    values = [probability[question]["max_js_divergence_bits"] if "max_js_divergence_bits" in probability[question] else 0 for question in questions]
    max_value = max(max(values), 1e-12)
    xscale = lambda value: left + value / max_value * (width - left - right)
    yscale = lambda index: top + index * (height - top - bottom) / (len(questions) - 1)
    for index, question_id in enumerate(questions):
        value = values[index]
        y = yscale(index)
        svg_text(items, left - 12, y + 4, question_id[:4], "tick", "end")
        svg_line(items, left, y, xscale(value), y, "axisline", "#7c3aed", 6)
        items.append(f'<circle cx="{xscale(value):.2f}" cy="{y:.2f}" r="5" fill="#5b21b6"/>')
    svg_line(items, left, top, left, height - bottom)
    svg_line(items, left, height - bottom, width - right, height - bottom)
    svg_text(items, (left + width - right) / 2, height - 22, "Maximum pairwise Jensen-Shannon divergence (bits)")
    svg_text(items, 22, (top + height - bottom) / 2, "Question ID", "axis", rotate=-90)
    svg_text(items, width - right, top + 10, "all measured categorical distributions identical", "caption", "end")
    path.write_text(finish_svg(items), encoding="utf-8")


def convert_figures(svg_paths):
    candidates = [os.environ.get("EDGE_PATH"), r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                  r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]
    edge = next((candidate for candidate in candidates if candidate and Path(candidate).exists()), None)
    if not edge:
        edge = shutil.which("msedge")
    if not edge:
        return {"png_pdf": False, "reason": "Microsoft Edge was not available"}
    with tempfile.TemporaryDirectory(prefix="jev-edge-edge-") as profile:
        common = [edge, "--headless", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                  f"--user-data-dir={profile}"]
        try:
            for svg_path in svg_paths:
                png_path = svg_path.with_suffix(".png")
                pdf_path = svg_path.with_suffix(".pdf")
                url = svg_path.resolve().as_uri()
                subprocess.run(common + ["--hide-scrollbars", "--window-size=1400,1000", f"--screenshot={png_path}", url],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(common + [f"--print-to-pdf={pdf_path}", url], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as exc:
            return {"png_pdf": False, "renderer": edge, "reason": f"headless conversion failed with exit code {exc.returncode}"}
    return {"png_pdf": True, "renderer": edge}


def analyze(raw_path, dataset_path, table_dir, figure_dir, csv_path, saved_summary_path, manifest_path):
    rows = read_jsonl(raw_path)
    dataset = load_dataset(dataset_path)
    metadata, saved_summary, observations, measured, warmups = validate_records(rows, dataset)
    latency = latency_analysis(measured)
    probability = probability_analysis(measured, dataset)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    write_tables(table_dir, latency, probability)
    figures = [figure_dir / "q8_latency_histogram.svg", figure_dir / "q8_latency_boxplot_by_question.svg",
               figure_dir / "q8_latency_by_trial_order.svg", figure_dir / "q8_question_latency_summary.svg",
               figure_dir / "q8_probability_stability.svg"]
    plot_histogram([row["wall_clock_ms"] for row in measured], figures[0])
    plot_boxplot(measured, figures[1])
    plot_order(measured, figures[2])
    plot_question_summary(latency, figures[3])
    plot_probability_stability(probability, figures[4])
    conversion = convert_figures(figures)
    report = {
        "source": str(raw_path),
        "dataset": str(dataset_path),
        "run_id": metadata["run_id"],
        "record_counts": {"metadata": 1, "summary": 1, "observations": len(observations),
                           "warmups": len(warmups), "measured": len(measured)},
        "models": sorted({row["model"] for row in observations}),
        "latency": latency,
        "probability_stability": probability,
        "saved_summary_comparison": saved_summary["summary"],
        "figure_conversion": conversion,
    }
    (table_dir / "q8_analysis_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_manifest(manifest_path, raw_path, csv_path, saved_summary_path, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--saved-summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.raw, args.dataset, args.tables, args.figures, args.csv, args.saved_summary, args.manifest)
    print(json.dumps({"run_id": report["run_id"], "measured": report["record_counts"]["measured"],
                      "figures": report["figure_conversion"]}, separators=(",", ":")))


if __name__ == "__main__":
    main()

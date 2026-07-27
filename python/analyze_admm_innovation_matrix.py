#!/usr/bin/env python3
"""Summarize and rank paired CPU ADMM innovation experiments."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def read_rows(directory):
    rows = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["_resultFile"] = str(path)
            rows.append(row)
    return rows


def case_key(row):
    return Path(row["dataset"]).name, int(row["clusters"])


def dataset_short_name(row):
    parts = Path(row["dataset"]).name.split("-")
    return parts[1] if len(parts) > 1 else parts[0]


def time_to_target(row, target):
    initial_quality = row.get("initialQualityMetrics", {})
    if initial_quality.get("sumSquaredError", math.inf) <= target:
        return 0.0
    for point in row.get("trajectory", []):
        if point["sumSquaredError"] <= target:
            return float(point["overallSeconds"])
    return math.inf


def best_pixel_metrics_through(row, iteration_count):
    candidates = []
    initial_quality = row.get("initialQualityMetrics", {})
    if "sumSquaredError" in initial_quality:
        candidates.append(initial_quality)
    candidates.extend(
        point
        for point in row.get("trajectory", [])[:iteration_count]
    )
    if not candidates:
        return math.nan, math.nan
    best = min(candidates, key=lambda point: point["sumSquaredError"])
    return (
        float(best["sumSquaredError"]),
        float(best.get("meanReprojectionError", math.nan)),
    )


def best_sse_through(row, iteration_count):
    return best_pixel_metrics_through(row, iteration_count)[0]


def format_integer(value):
    return f"{value:,.0f}" if math.isfinite(value) else "-"


def geometric_mean(values):
    finite = [value for value in values if value > 0 and math.isfinite(value)]
    if not finite:
        return math.nan
    return math.exp(sum(math.log(value) for value in finite) / len(finite))


def summarize(rows, baseline_name="baseline"):
    by_variant = defaultdict(dict)
    for row in rows:
        by_variant[row["variant"]][case_key(row)] = row
    baseline = by_variant.get(baseline_name, {})
    summaries = []
    for variant, cases in sorted(by_variant.items()):
        paired = sorted(set(baseline) & set(cases))
        final_ratios = []
        speedups = []
        solved_targets = 0
        for key in paired:
            base = baseline[key]
            candidate = cases[key]
            base_sse = base["qualityMetrics"]["sumSquaredError"]
            candidate_sse = candidate["qualityMetrics"]["sumSquaredError"]
            final_ratios.append(candidate_sse / base_sse)
            candidate_time = time_to_target(candidate, base_sse)
            baseline_time = time_to_target(base, base_sse)
            if math.isfinite(candidate_time):
                solved_targets += 1
                if candidate_time > 0.0 and baseline_time > 0.0:
                    speedups.append(baseline_time / candidate_time)
        summaries.append({
            "variant": variant,
            "completedCases": len(cases),
            "pairedCases": len(paired),
            "targetsSolved": solved_targets,
            "geomeanFinalSseRatio": geometric_mean(final_ratios),
            "geomeanSpeedupToBaselineTarget": geometric_mean(speedups),
            "eligibleForGreedy": (
                variant != baseline_name
                and len(paired) > 0
                and solved_targets == len(paired)
            ),
        })
    return summaries, by_variant


def render_markdown(
    summaries, by_variant, baseline_name="baseline", checkpoint=60
):
    lines = [
        "# Standard Pixel Reprojection Results",
        "",
        "Every value below is independently evaluated on the global camera and",
        "landmark state using the standard BAL/Snavely pixel reprojection error.",
        "Pixel SSE is sum(dx^2 + dy^2); mean px is the mean Euclidean pixel error.",
        "ADMM proximal/consensus terms and worker-local surrogate costs are not",
        "reported. DABA's weighted 3D-ray metric is not used.",
        "",
        f"The table reports the best global pixel-SSE state reached through {checkpoint}",
        "outer iterations, its matching mean pixel error, and the best iteration.",
    ]
    lines.extend(["", f"Completed rows: {sum(len(cases) for cases in by_variant.values())}", "",
        f"| Method | Dataset | K | Local LM steps | Alpha | Best iteration <={checkpoint} | Best pixel SSE <={checkpoint} | Mean px | Pixel SSE at iteration {checkpoint} | End / best | Time at {checkpoint} s | Communication at {checkpoint} MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for variant, cases in sorted(by_variant.items()):
        for (dataset, clusters), row in sorted(cases.items()):
            trajectory = row.get("trajectory", [])[:checkpoint]
            candidates = []
            initial = row.get("initialQualityMetrics", {})
            if "sumSquaredError" in initial:
                candidates.append((
                    -1,
                    float(initial["sumSquaredError"]),
                    float(initial.get("meanReprojectionError", math.nan)),
                ))
            candidates.extend(
                (
                    int(point.get("iteration", index)),
                    float(point["sumSquaredError"]),
                    float(point.get("meanReprojectionError", math.nan)),
                )
                for index, point in enumerate(trajectory)
            )
            if not candidates:
                continue
            best_iteration, best_sse, best_mean = min(
                candidates, key=lambda item: item[1])
            endpoint_sse = float(trajectory[-1]["sumSquaredError"])
            endpoint = trajectory[-1]
            mib = (
                endpoint.get("transportBytesSent", 0)
                + endpoint.get("transportBytesReceived", 0)
            ) / 1024**2
            lines.append(
                f"| {variant} | {dataset} | {clusters} | "
                f"{row.get('localSteps', '-')} | "
                f"{float(row.get('alpha', math.nan)):.1f} | "
                f"{best_iteration} | {format_integer(best_sse)} | "
                f"{best_mean:.4f} | {format_integer(endpoint_sse)} | "
                f"{endpoint_sse / best_sse:.2f} | "
                f"{float(endpoint['overallSeconds']):.3f} | {mib:.3f} |"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--clusters", nargs="*", type=int)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--checkpoint", type=int, default=60)
    parser.add_argument("--output")
    arguments = parser.parse_args()
    rows = read_rows(arguments.directory)
    if arguments.clusters:
        allowed_clusters = set(arguments.clusters)
        rows = [row for row in rows if int(row["clusters"]) in allowed_clusters]
    if arguments.datasets:
        allowed_datasets = set(arguments.datasets)
        rows = [row for row in rows if dataset_short_name(row) in allowed_datasets]
    summaries, by_variant = summarize(rows, arguments.baseline)
    if arguments.checkpoint <= 0:
        raise ValueError("checkpoint must be positive")
    report = render_markdown(
        summaries, by_variant, arguments.baseline, arguments.checkpoint)
    if arguments.output:
        Path(arguments.output).write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze repeatability of the frozen K4/K16 Stage-C operating points."""

import argparse
import json
import statistics
from pathlib import Path

from analyze_stage_c_scaling_pilot import (
    load_drs_rows,
    process_resource,
    scene_key,
)


EXPECTED = {
    "1dsfm": ("roman_forum", "trafalgar"),
    "bal": ("bal52", "bal3068"),
}
CLUSTERS = (4, 16)


def coefficient_of_variation(values):
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean if mean else 0.0


def relative_spread(values):
    return (max(values) - min(values)) / max(
        abs(statistics.fmean(values)), float.fromhex("0x1p-1022")
    )


def load_runs(root, repeats, iterations):
    runs = {}
    for repeat in range(1, repeats + 1):
        runs[repeat] = {}
        for family, scenes in EXPECTED.items():
            directory = root / f"repeat{repeat}" / family / "c1_c5"
            rows = load_drs_rows(directory)
            expected = {(scene, cluster) for scene in scenes for cluster in CLUSTERS}
            if rows.keys() != expected:
                raise ValueError(
                    f"repeat{repeat}/{family} coverage mismatch: "
                    f"expected={sorted(expected)}, actual={sorted(rows)}"
                )
            for key, row in rows.items():
                if row.get("completedIterations") != iterations:
                    raise ValueError(
                        f"incomplete repeat{repeat}/{family}/{key}: "
                        f"{row.get('completedIterations')}/{iterations}"
                    )
                if row.get("objectiveLoss", "l2") != "l2":
                    raise ValueError(f"non-L2 repeat row: repeat{repeat}/{family}/{key}")
            runs[repeat][family] = (directory, rows)
    return runs


def summarize_cases(runs):
    summaries = []
    for family, scenes in EXPECTED.items():
        for scene in scenes:
            for cluster in CLUSTERS:
                rows = [runs[repeat][family][1][(scene, cluster)] for repeat in runs]
                resources = [
                    process_resource(runs[repeat][family][0], row, "worker")
                    for repeat, row in zip(runs, rows)
                ]
                sses = [row["qualityMetrics"]["sumSquaredError"] for row in rows]
                optimization = [row["optimizationSeconds"] for row in rows]
                overall = [row["overallSeconds"] for row in rows]
                worker_cpu = [resource["cpu_seconds"] for resource in resources]
                worker_rss = [resource["max_rss_kb"] for resource in resources]
                traffic = [
                    row["transportBytesSent"] + row["transportBytesReceived"]
                    for row in rows
                ]
                summaries.append({
                    "family": family,
                    "scene": scene,
                    "clusters": cluster,
                    "sse_mean": statistics.fmean(sses),
                    "sse_relative_spread": relative_spread(sses),
                    "optimization_mean": statistics.fmean(optimization),
                    "optimization_cv": coefficient_of_variation(optimization),
                    "overall_mean": statistics.fmean(overall),
                    "overall_cv": coefficient_of_variation(overall),
                    "worker_cpu_mean": statistics.fmean(worker_cpu),
                    "worker_cpu_cv": coefficient_of_variation(worker_cpu),
                    "worker_rss_mean_kb": statistics.fmean(worker_rss),
                    "worker_rss_relative_spread": relative_spread(worker_rss),
                    "traffic_mean_bytes": statistics.fmean(traffic),
                    "traffic_relative_spread": relative_spread(traffic),
                    "rejections": [row["rejections"] for row in rows],
                    "oracles": [row["proximalOracleCalls"] for row in rows],
                })
    return summaries


def summarize_endpoint_ratios(runs):
    summaries = []
    for family, scenes in EXPECTED.items():
        optimization_ratios = []
        overall_ratios = []
        worker_cpu_ratios = []
        traffic_ratios = []
        for repeat in runs:
            directory, rows = runs[repeat][family]
            by_cluster = {
                cluster: [rows[(scene, cluster)] for scene in scenes]
                for cluster in CLUSTERS
            }
            resources = {
                cluster: [
                    process_resource(directory, row, "worker")
                    for row in by_cluster[cluster]
                ]
                for cluster in CLUSTERS
            }
            optimization_ratios.append(
                sum(row["optimizationSeconds"] for row in by_cluster[16])
                / sum(row["optimizationSeconds"] for row in by_cluster[4])
            )
            overall_ratios.append(
                sum(row["overallSeconds"] for row in by_cluster[16])
                / sum(row["overallSeconds"] for row in by_cluster[4])
            )
            worker_cpu_ratios.append(
                sum(resource["cpu_seconds"] for resource in resources[16])
                / sum(resource["cpu_seconds"] for resource in resources[4])
            )
            traffic_ratios.append(
                sum(
                    row["transportBytesSent"] + row["transportBytesReceived"]
                    for row in by_cluster[16]
                )
                / sum(
                    row["transportBytesSent"] + row["transportBytesReceived"]
                    for row in by_cluster[4]
                )
            )
        summaries.append({
            "family": family,
            "optimization_k16_over_k4": optimization_ratios,
            "optimization_ratio_mean": statistics.fmean(optimization_ratios),
            "optimization_ratio_cv": coefficient_of_variation(optimization_ratios),
            "overall_k16_over_k4": overall_ratios,
            "overall_ratio_mean": statistics.fmean(overall_ratios),
            "overall_ratio_cv": coefficient_of_variation(overall_ratios),
            "worker_cpu_k16_over_k4": worker_cpu_ratios,
            "worker_cpu_ratio_mean": statistics.fmean(worker_cpu_ratios),
            "traffic_k16_over_k4": traffic_ratios,
            "traffic_ratio_mean": statistics.fmean(traffic_ratios),
        })
    return summaries


def write_report(path, repeats, cases, endpoint_ratios):
    with path.open("w", encoding="utf-8") as output:
        output.write("# K4/K16 Scaling Repeatability Gate\n\n")
        output.write(
            f"One warm-up is excluded. {repeats} measured repeats use the frozen "
            "global L2 C1+C5 policy, fixed partition cache, and one thread per "
            "cluster.\n\n"
        )
        output.write(
            "| Family | Scene | K | SSE spread | Opt. s | Opt. CV | Overall s | "
            "Overall CV | Worker CPU s | CPU CV | RSS spread | Traffic spread |\n"
        )
        output.write(
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for row in cases:
            output.write(
                f"| {row['family']} | {row['scene']} | {row['clusters']} | "
                f"{row['sse_relative_spread']:.3e} | "
                f"{row['optimization_mean']:.3f} | {row['optimization_cv']:.4f} | "
                f"{row['overall_mean']:.3f} | {row['overall_cv']:.4f} | "
                f"{row['worker_cpu_mean']:.3f} | {row['worker_cpu_cv']:.4f} | "
                f"{row['worker_rss_relative_spread']:.3e} | "
                f"{row['traffic_relative_spread']:.3e} |\n"
            )
        output.write("\n## Endpoint ratios\n\n")
        output.write(
            "| Family | K16/K4 opt. | Ratio CV | K16/K4 overall | Ratio CV | "
            "K16/K4 worker CPU | K16/K4 traffic |\n"
        )
        output.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in endpoint_ratios:
            output.write(
                f"| {row['family']} | {row['optimization_ratio_mean']:.4f} | "
                f"{row['optimization_ratio_cv']:.4f} | "
                f"{row['overall_ratio_mean']:.4f} | {row['overall_ratio_cv']:.4f} | "
                f"{row['worker_cpu_ratio_mean']:.4f} | "
                f"{row['traffic_ratio_mean']:.4f} |\n"
            )
        output.write("\n## Decision\n\n")
        maximum_sse_spread = max(row["sse_relative_spread"] for row in cases)
        maximum_optimization_cv = max(row["optimization_cv"] for row in cases)
        exact_cases = sum(row["sse_relative_spread"] == 0.0 for row in cases)
        work_counts_deterministic = all(
            len(set(row["rejections"])) == 1 and len(set(row["oracles"])) == 1
            for row in cases
        )
        output.write(
            f"Endpoint SSE is bitwise identical in `{exact_cases}/{len(cases)}` "
            f"cases; maximum relative spread is `{maximum_sse_spread:.3e}`. "
            f"Rejection and oracle counts are "
            f"{'identical' if work_counts_deterministic else 'not identical'}; "
            f"maximum per-case optimization-time CV is `{maximum_optimization_cv:.4f}`. "
            "Repeatability is reported without changing either global operating point.\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    arguments = parser.parse_args()
    if arguments.repeats < 3:
        raise ValueError("at least three measured repeats are required")
    runs = load_runs(arguments.root, arguments.repeats, arguments.iterations)
    cases = summarize_cases(runs)
    endpoint_ratios = summarize_endpoint_ratios(runs)
    repeatability = {
        "exact_sse_cases": sum(
            row["sse_relative_spread"] == 0.0 for row in cases
        ),
        "total_cases": len(cases),
        "maximum_sse_relative_spread": max(
            row["sse_relative_spread"] for row in cases
        ),
        "maximum_optimization_cv": max(row["optimization_cv"] for row in cases),
        "work_counts_deterministic": all(
            len(set(row["rejections"])) == 1 and len(set(row["oracles"])) == 1
            for row in cases
        ),
    }
    report = arguments.root / "report.md"
    write_report(report, arguments.repeats, cases, endpoint_ratios)
    (arguments.root / "summary.json").write_text(
        json.dumps(
            {
                "cases": cases,
                "endpoint_ratios": endpoint_ratios,
                "repeatability": repeatability,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main()
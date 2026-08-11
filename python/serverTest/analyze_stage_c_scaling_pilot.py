#!/usr/bin/env python3
"""Analyze the frozen Stage-C C1+C5 cluster-count scaling pilot."""

import argparse
import json
import math
import re
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_CERES = {
    "1dsfm": WORKSPACE / "benchmark_results/1dsfm_drs_ceres_se3_all15/ceres/results.jsonl",
    "bal": WORKSPACE / "benchmark_results/bal_ceres_se3_all29/results.jsonl",
}
TIME_VALUE = re.compile(r"^([^:]+):\s*(.+)$")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--clusters", type=int, nargs="+", required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--expected-1dsfm", default="roman_forum,trafalgar"
    )
    parser.add_argument("--expected-bal", default="bal52,bal3068")
    return parser.parse_args()


def scene_key(row):
    if row.get("scene"):
        return row["scene"]
    if row.get("balId") is not None:
        return f"bal{row['balId']}"
    dataset = Path(row["dataset"])
    if "sfm_init_1dsfm" in row["dataset"]:
        return dataset.parent.name
    match = re.search(r"problem-(\d+)-", dataset.name)
    if not match:
        raise ValueError(f"cannot identify dataset: {dataset}")
    return f"bal{match.group(1)}"


def load_jsonl(path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = scene_key(row)
        if key in rows:
            raise ValueError(f"duplicate scene {key} in {path}")
        rows[key] = row
    return rows


def load_drs_rows(directory):
    rows = {}
    paths = sorted(directory.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no JSONL files in {directory}")
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = scene_key(row), row["clusters"]
            if key in rows:
                raise ValueError(f"duplicate DRS row {key} in {directory}")
            rows[key] = row
    return rows


def parse_gnu_time(path):
    values = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TIME_VALUE.match(line.strip())
        if match:
            values[match.group(1)] = match.group(2)
    required = (
        "User time (seconds)",
        "System time (seconds)",
        "Maximum resident set size (kbytes)",
    )
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"incomplete GNU time file {path}: {missing}")
    return {
        "cpu_seconds": (
            float(values["User time (seconds)"])
            + float(values["System time (seconds)"])
        ),
        "max_rss_kb": int(values["Maximum resident set size (kbytes)"]),
        "artifact": str(path.resolve().relative_to(WORKSPACE)),
    }


def process_resource(directory, row, process):
    scene = scene_key(row)
    token = scene[3:] if scene.startswith("bal") else scene
    pattern = (
        f"{row['variant']}_{token}_k{row['clusters']}_i{row['iterations']}_"
        f"l{row['localSteps']}_t{row['threadsPerCluster']}_{process}.time"
    )
    matches = sorted((directory / "memory").glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"expected one {process} time file for {scene}/K{row['clusters']}, "
            f"found {len(matches)}"
        )
    return parse_gnu_time(matches[0])


def geometric_mean(values):
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def classify(ratios):
    wins = sum(value < 1.0 and not math.isclose(value, 1.0, rel_tol=1e-12) for value in ratios)
    ties = sum(math.isclose(value, 1.0, rel_tol=1e-12) for value in ratios)
    return wins, ties, len(ratios) - wins - ties


def summarize_family(family, directory, rows, ceres, scenes, clusters, iterations):
    expected = {(scene, cluster) for scene in scenes for cluster in clusters}
    missing = sorted(expected - rows.keys())
    extra = sorted(rows.keys() - expected)
    if missing or extra:
        raise ValueError(f"{family} coverage mismatch: missing={missing}, extra={extra}")
    for key in sorted(expected):
        row = rows[key]
        if row.get("completedIterations") != iterations:
            raise ValueError(
                f"incomplete {family}/{key[0]}/K{key[1]}: "
                f"{row.get('completedIterations')}/{iterations}"
            )
        if row.get("objectiveLoss", "l2") != "l2":
            raise ValueError(f"non-L2 row in scaling pilot: {family}/{key}")
    missing_ceres = sorted(set(scenes) - ceres.keys())
    if missing_ceres:
        raise ValueError(f"missing Ceres references for {family}: {missing_ceres}")

    baseline_cluster = clusters[0]
    summaries = []
    for cluster in clusters:
        selected = [rows[(scene, cluster)] for scene in scenes]
        baseline = [rows[(scene, baseline_cluster)] for scene in scenes]
        sse_ratios = [
            row["qualityMetrics"]["sumSquaredError"]
            / base["qualityMetrics"]["sumSquaredError"]
            for row, base in zip(selected, baseline)
        ]
        ceres_ratios = [
            row["qualityMetrics"]["sumSquaredError"]
            / ceres[scene]["qualityMetrics"]["sumSquaredError"]
            for scene, row in zip(scenes, selected)
        ]
        worker_resources = [
            process_resource(directory, row, "worker") for row in selected
        ]
        coordinator_resources = [
            process_resource(directory, row, "coordinator") for row in selected
        ]
        summaries.append({
            "clusters": cluster,
            "scenes": len(scenes),
            "sse_vs_baseline_geometric": geometric_mean(sse_ratios),
            "sse_vs_ceres_geometric": geometric_mean(ceres_ratios),
            "wins_ties_losses_vs_baseline": classify(sse_ratios),
            "optimization_seconds": math.fsum(
                row["optimizationSeconds"] for row in selected
            ),
            "overall_seconds": math.fsum(row["overallSeconds"] for row in selected),
            "worker_cpu_seconds": math.fsum(
                resource["cpu_seconds"] for resource in worker_resources
            ),
            "worker_max_rss_kb": max(
                resource["max_rss_kb"] for resource in worker_resources
            ),
            "coordinator_cpu_seconds": math.fsum(
                resource["cpu_seconds"] for resource in coordinator_resources
            ),
            "coordinator_max_rss_kb": max(
                resource["max_rss_kb"] for resource in coordinator_resources
            ),
            "transport_bytes": sum(
                row["transportBytesSent"] + row["transportBytesReceived"]
                for row in selected
            ),
            "proximal_oracle_calls": sum(
                row["proximalOracleCalls"] for row in selected
            ),
            "rows": {
                scene: {
                    "sse": row["qualityMetrics"]["sumSquaredError"],
                    "sse_vs_baseline": sse_ratio,
                    "sse_vs_ceres": ceres_ratio,
                    "optimization_seconds": row["optimizationSeconds"],
                    "overall_seconds": row["overallSeconds"],
                }
                for scene, row, sse_ratio, ceres_ratio in zip(
                    scenes, selected, sse_ratios, ceres_ratios
                )
            },
        })
    baseline = summaries[0]
    for summary in summaries:
        summary["optimization_vs_baseline"] = (
            summary["optimization_seconds"] / baseline["optimization_seconds"]
        )
        summary["overall_vs_baseline"] = (
            summary["overall_seconds"] / baseline["overall_seconds"]
        )
        summary["worker_cpu_vs_baseline"] = (
            summary["worker_cpu_seconds"] / baseline["worker_cpu_seconds"]
        )
        summary["transport_vs_baseline"] = (
            summary["transport_bytes"] / baseline["transport_bytes"]
        )
        summary["oracles_vs_baseline"] = (
            summary["proximal_oracle_calls"] / baseline["proximal_oracle_calls"]
        )
    return summaries


def write_family(output, family, summaries):
    title = "1DSfM" if family == "1dsfm" else "BAL"
    output.write(f"## {title}\n\n")
    output.write(
        "| K | SSE/Kmin | SSE/Ceres | W/T/L | Opt. s | Opt./Kmin | Overall s | "
        "Worker CPU s | Worker RSS MiB | Traffic MiB | Oracles |\n"
    )
    output.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in summaries:
        wins, ties, losses = row["wins_ties_losses_vs_baseline"]
        output.write(
            f"| {row['clusters']} | {row['sse_vs_baseline_geometric']:.6f} | "
            f"{row['sse_vs_ceres_geometric']:.6f} | {wins}/{ties}/{losses} | "
            f"{row['optimization_seconds']:.3f} | {row['optimization_vs_baseline']:.3f} | "
            f"{row['overall_seconds']:.3f} | {row['worker_cpu_seconds']:.3f} | "
            f"{row['worker_max_rss_kb'] / 1024:.1f} | "
            f"{row['transport_bytes'] / 2**20:.1f} | "
            f"{row['proximal_oracle_calls']} |\n"
        )
    output.write("\n")


def main():
    arguments = parse_arguments()
    clusters = sorted(set(arguments.clusters))
    if len(clusters) != len(arguments.clusters) or any(value <= 1 for value in clusters):
        raise ValueError("clusters must be unique K>1 values")
    expected = {
        "1dsfm": tuple(filter(None, arguments.expected_1dsfm.split(","))),
        "bal": tuple(filter(None, arguments.expected_bal.split(","))),
    }
    summaries = {}
    for family in ("1dsfm", "bal"):
        directory = arguments.root / family / "c1_c5"
        summaries[family] = summarize_family(
            family,
            directory,
            load_drs_rows(directory),
            load_jsonl(DEFAULT_CERES[family]),
            expected[family],
            clusters,
            arguments.iterations,
        )

    arguments.root.mkdir(parents=True, exist_ok=True)
    report = arguments.root / "report.md"
    complete_confirmation = (
        len(expected["1dsfm"]) == 15 and len(expected["bal"]) == 29
    )
    with report.open("w", encoding="utf-8") as output:
        scope = "Confirmation" if complete_confirmation else "Pilot"
        output.write(f"# Frozen Stage-C C1+C5 Cluster-Count Scaling {scope}\n\n")
        output.write(
            f"One global L2 C1+C5 policy is held fixed for K={clusters}, with "
            "one thread per cluster. Kmin is the smallest requested distributed "
            "configuration; Ceres quality uses the authoritative left-SE3 references. "
            "Times and resource totals cover the complete matched cohort.\n\n"
        )
        for family in ("1dsfm", "bal"):
            write_family(output, family, summaries[family])
        if complete_confirmation and clusters == [4, 16]:
            one_d_sfm_k4, one_d_sfm_k16 = summaries["1dsfm"]
            bal_k4, bal_k16 = summaries["bal"]
            output.write("## Decision\n\n")
            output.write(
                "Retain two global, nondominated operating points with identical "
                "solver settings: K4 is the resource endpoint and K16 is the latency "
                "endpoint. K16/K4 geometric SSE is "
                f"`{one_d_sfm_k16['sse_vs_baseline_geometric']:.6f}x` on 1DSfM "
                f"and `{bal_k16['sse_vs_baseline_geometric']:.6f}x` on BAL; "
                "optimization time is "
                f"`{one_d_sfm_k16['optimization_vs_baseline']:.3f}x` and "
                f"`{bal_k16['optimization_vs_baseline']:.3f}x`. Worker CPU rises to "
                f"`{one_d_sfm_k16['worker_cpu_vs_baseline']:.3f}x`/"
                f"`{bal_k16['worker_cpu_vs_baseline']:.3f}x`, and traffic rises to "
                f"`{one_d_sfm_k16['transport_vs_baseline']:.3f}x`/"
                f"`{bal_k16['transport_vs_baseline']:.3f}x`. Do not route K by scene "
                "and do not alter C1 or C5.\n"
            )
        else:
            output.write("## Gate\n\n")
            output.write(
                "Use this pilot only to select globally defensible K values for the "
                "complete 15-scene and 29-scene scaling confirmation. Do not alter "
                "C1, C5, or scene-specific settings from this result.\n"
            )
    summary = arguments.root / "summary.json"
    summary.write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Reduce a global C5 threshold development sweep."""

import argparse
import json
import math
import re
from pathlib import Path


EXPECTED = {"1dsfm": 6, "bal": 5}


def scene_key(row):
    dataset = Path(row["dataset"])
    if "sfm_init_1dsfm" in row["dataset"]:
        return dataset.parent.name
    match = re.search(r"problem-(\d+)-", dataset.name)
    if not match:
        raise ValueError(f"cannot identify dataset {dataset}")
    return f"bal{match.group(1)}"


def load_directory(directory):
    rows = {}
    for path in directory.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = scene_key(row)
            if key in rows:
                raise ValueError(f"duplicate scene {key} in {directory}")
            rows[key] = row
    if not rows:
        raise FileNotFoundError(f"no JSONL rows in {directory}")
    return rows


def sse(row):
    return row["qualityMetrics"]["sumSquaredError"]


def geometric_mean(values):
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def depth_two_fraction(rows, scenes):
    depth_two = 0
    total = 0
    for scene in scenes:
        row = rows[scene]
        for trajectory_row in row["trajectory"]:
            values = trajectory_row["localStepsUsed"]
            if not isinstance(values, list):
                values = [values] * row["clusters"]
            depth_two += sum(value > 1 for value in values)
            total += len(values)
    return depth_two / total


def summarize(reference, candidate, scenes):
    ratios = [sse(candidate[scene]) / sse(reference[scene]) for scene in scenes]
    reference_seconds = math.fsum(
        reference[scene]["optimizationSeconds"] for scene in scenes
    )
    candidate_seconds = math.fsum(
        candidate[scene]["optimizationSeconds"] for scene in scenes
    )
    return {
        "geometric_sse_ratio": geometric_mean(ratios),
        "summed_sse_ratio": (
            math.fsum(sse(candidate[scene]) for scene in scenes)
            / math.fsum(sse(reference[scene]) for scene in scenes)
        ),
        "wins": sum(value < 1.0 for value in ratios),
        "losses": sum(value > 1.0 for value in ratios),
        "worst_sse_ratio": max(ratios),
        "optimization_ratio": candidate_seconds / reference_seconds,
        "depth_two_fraction": depth_two_fraction(candidate, scenes),
        "recovery_exhausted": any(
            trajectory_row["recoveryExhausted"]
            for scene in scenes
            for trajectory_row in candidate[scene]["trajectory"]
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument(
        "--parameter",
        choices=("high", "low", "window", "dwell", "maximum_depth", "start"),
        default="high",
    )
    parser.add_argument("--tag-prefix", default="h")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args()
    thresholds = [float(value) for value in arguments.thresholds.split()]

    references = {
        family: load_directory(arguments.reference_root / family / "c1")
        for family in EXPECTED
    }
    candidates = {}
    summaries = {}
    for threshold in thresholds:
        tag = f"{arguments.tag_prefix}{threshold:.2f}".replace(".", "p")
        candidates[threshold] = {}
        summaries[threshold] = {}
        for family, expected in EXPECTED.items():
            rows = load_directory(arguments.root / tag / family / "c1_c5")
            completed = {
                scene for scene, row in rows.items()
                if row.get("completedIterations") == arguments.iterations
            }
            reference_scenes = set(references[family])
            scenes = sorted(completed & reference_scenes)
            if arguments.require_complete and len(scenes) != expected:
                raise ValueError(
                    f"incomplete h={threshold}/{family}: {len(scenes)}/{expected}"
                )
            parameter_field = {
                "high": "adaptiveLocalDepthHigh",
                "low": "adaptiveLocalDepthLow",
                "window": "adaptiveLocalDepthWindow",
                "dwell": "adaptiveLocalDepthDwell",
                "maximum_depth": "adaptiveLocalDepthMaximum",
                "start": "adaptiveLocalDepthStart",
            }[arguments.parameter]
            expected_value = (
                int(threshold)
                if arguments.parameter in (
                    "window", "dwell", "maximum_depth", "start"
                )
                else threshold
            )
            if any(row[parameter_field] != expected_value for row in rows.values()):
                raise ValueError(f"threshold mismatch in h={threshold}/{family}")
            candidates[threshold][family] = rows
            summaries[threshold][family] = summarize(
                references[family], rows, scenes
            )

    control = candidates[thresholds[0]]
    for threshold in thresholds:
        summaries[threshold]["versus_control"] = {
            family: summarize(
                control[family],
                candidates[threshold][family],
                sorted(control[family]),
            )
            for family in EXPECTED
        }

    eligible = [
        threshold for threshold in thresholds
        if all(
            summaries[threshold][family]["geometric_sse_ratio"] < 1.0
            and not summaries[threshold][family]["recovery_exhausted"]
            for family in EXPECTED
        )
    ]
    leader = min(
        eligible,
        key=lambda threshold: (
            max(
                summaries[threshold][family]["geometric_sse_ratio"]
                for family in EXPECTED
            ),
            max(
                summaries[threshold][family]["worst_sse_ratio"]
                for family in EXPECTED
            ),
            sum(
                summaries[threshold][family]["optimization_ratio"]
                for family in EXPECTED
            ),
        ),
    ) if eligible else None

    report = arguments.root / "report.md"
    with report.open("w", encoding="utf-8") as output:
        output.write(
            f"# Stage-C Global C5 {arguments.parameter.title()}-Threshold "
            "Development Sweep\n\n"
        )
        output.write(
            "All candidates use one common threshold across scenes and families. "
            "Ratios compare C1+C5 with the accepted C1 rung.\n\n"
        )
        output.write(
            f"| {arguments.parameter.title()} | 1DSfM SSE | W/L | Worst | Depth2 | Opt. | "
            "BAL SSE | W/L | Worst | Depth2 | Opt. | Eligible |\n"
        )
        output.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for threshold in thresholds:
            one_d_sfm = summaries[threshold]["1dsfm"]
            bal = summaries[threshold]["bal"]
            output.write(
                f"| {threshold:.2f} | {one_d_sfm['geometric_sse_ratio']:.9f} | "
                f"{one_d_sfm['wins']}/{one_d_sfm['losses']} | "
                f"{one_d_sfm['worst_sse_ratio']:.9f} | "
                f"{one_d_sfm['depth_two_fraction']:.4f} | "
                f"{one_d_sfm['optimization_ratio']:.6f} | "
                f"{bal['geometric_sse_ratio']:.9f} | {bal['wins']}/{bal['losses']} | "
                f"{bal['worst_sse_ratio']:.9f} | {bal['depth_two_fraction']:.4f} | "
                f"{bal['optimization_ratio']:.6f} | "
                f"{'yes' if threshold in eligible else 'no'} |\n"
            )
        output.write("\n## Decision\n\n")
        if leader is None:
            output.write(
                f"No {arguments.parameter}-threshold candidate improves C1 geometrically on both "
                f"development families. Freeze no new {arguments.parameter} value; proceed to "
                "the next one-factor C5 mechanism only after interpreting this gate.\n"
            )
        else:
            output.write(
                f"Development leader: `{leader:.2f}`. Freeze it before the "
                "nine-scene held-out gate; do not tune from held-out results.\n"
            )
    summary_path = arguments.root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "thresholds": {str(key): value for key, value in summaries.items()},
                "parameter": arguments.parameter,
                "eligible": eligible,
                "leader": leader,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Analyze deterministic quality and timing variance for tuned Stage-C repeats."""

import argparse
import json
import math
import re
import statistics
from pathlib import Path


EXPECTED = {"1dsfm": ("roman_forum", "trafalgar"), "bal": ("bal52", "bal3068")}


def scene_key(row):
    dataset = Path(row["dataset"])
    if "sfm_init_1dsfm" in row["dataset"]:
        return dataset.parent.name
    match = re.search(r"problem-(\d+)-", dataset.name)
    if not match:
        raise ValueError(f"cannot identify dataset {dataset}")
    return f"bal{match.group(1)}"


def load(directory):
    rows = {}
    for path in directory.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                scene = scene_key(row)
                if scene in rows:
                    raise ValueError(f"duplicate scene {scene} in {directory}")
                rows[scene] = row
    return rows


def coefficient_of_variation(values):
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean if mean else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args()
    runs = {
        repeat: {
            family: load(arguments.root / f"repeat{repeat}" / family / "c1_c5")
            for family in EXPECTED
        }
        for repeat in range(1, arguments.repeats + 1)
    }

    rows = []
    for family, scenes in EXPECTED.items():
        for scene in scenes:
            scene_rows = [runs[repeat][family].get(scene) for repeat in runs]
            if arguments.require_complete and (
                any(row is None for row in scene_rows)
                or any(row.get("completedIterations") != 30 for row in scene_rows)
            ):
                raise ValueError(f"incomplete repeats for {family}/{scene}")
            scene_rows = [row for row in scene_rows if row is not None]
            sses = [row["qualityMetrics"]["sumSquaredError"] for row in scene_rows]
            optimization = [row["optimizationSeconds"] for row in scene_rows]
            overall = [row["overallSeconds"] for row in scene_rows]
            relative_spread = (max(sses) - min(sses)) / max(
                abs(statistics.fmean(sses)), float.fromhex("0x1p-1022")
            )
            rows.append({
                "family": family,
                "scene": scene,
                "sse_mean": statistics.fmean(sses),
                "sse_relative_spread": relative_spread,
                "optimization_mean": statistics.fmean(optimization),
                "optimization_cv": coefficient_of_variation(optimization),
                "overall_mean": statistics.fmean(overall),
                "overall_cv": coefficient_of_variation(overall),
                "rejections": [row["rejections"] for row in scene_rows],
                "oracles": [row["proximalOracleCalls"] for row in scene_rows],
            })

    report = arguments.root / "report.md"
    with report.open("w", encoding="utf-8") as output:
        output.write("# Tuned Stage-C Determinism And Timing Repeats\n\n")
        output.write(
            f"One warm-up run is excluded. {arguments.repeats} measured repeats use the same "
            "global tuned C1+C5 policy, fixed partition cache, and one thread "
            "per cluster.\n\n"
        )
        output.write(
            "| Family | Scene | SSE | Relative spread | Optimization s | Opt. CV | "
            "Overall s | Overall CV | Rejections | Oracles |\n"
        )
        output.write("|---|---|---:|---:|---:|---:|---:|---:|---|---|\n")
        for row in rows:
            output.write(
                f"| {row['family']} | {row['scene']} | {row['sse_mean']:.6f} | "
                f"{row['sse_relative_spread']:.3e} | "
                f"{row['optimization_mean']:.3f} | {row['optimization_cv']:.4f} | "
                f"{row['overall_mean']:.3f} | {row['overall_cv']:.4f} | "
                f"{row['rejections']} | {row['oracles']} |\n"
            )
        output.write("\n## Decision\n\n")
        max_spread = max(row["sse_relative_spread"] for row in rows)
        output.write(
            f"Maximum endpoint SSE relative spread is `{max_spread:.3e}`. "
            "Timing variance is reported rather than used to alter solver settings.\n"
        )
    summary = arguments.root / "summary.json"
    summary.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
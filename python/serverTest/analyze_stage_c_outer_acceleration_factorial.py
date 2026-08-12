#!/usr/bin/env python3
"""Analyze the matched outer-acceleration sentinel factorial."""

import argparse
import json
import math
import re
from pathlib import Path


MODES = ("none", "nesterov", "themelis_nesterov", "lbfgs", "anderson")
EXPECTED = {
    "1dsfm": ("roman_forum", "trafalgar"),
    "bal": ("bal52", "bal3068"),
}


def scene_key(row):
    dataset = Path(row["dataset"])
    if "sfm_init_1dsfm" in row["dataset"]:
        return dataset.parent.name
    match = re.search(r"problem-(\d+)-", dataset.name)
    if not match:
        raise ValueError(f"cannot identify dataset: {dataset}")
    return f"bal{match.group(1)}"


def load(directory):
    paths = tuple(directory.glob("*.jsonl"))
    if len(paths) != 1:
        raise ValueError(f"expected one JSONL in {directory}, found {len(paths)}")
    rows = {}
    for line in paths[0].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scene = scene_key(row)
        if scene in rows:
            raise ValueError(f"duplicate scene {scene} in {paths[0]}")
        rows[scene] = row
    return rows


def geometric_mean(values):
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    arguments = parser.parse_args()

    summary = {}
    for family, scenes in EXPECTED.items():
        modes = {mode: load(arguments.root / family / mode) for mode in MODES}
        for mode, rows in modes.items():
            if set(rows) != set(scenes):
                raise ValueError(f"coverage mismatch {family}/{mode}: {sorted(rows)}")
            for scene, row in rows.items():
                expected = {
                    "clusters": 24,
                    "iterations": arguments.iterations,
                    "localSolver": "nesterov",
                    "outerAcceleration": mode,
                    "lineSearchGrid": "0,1",
                    "trustRegionPolicy": "drs",
                    "persistentTrustRegion": True,
                    "blockRecoveryMode": "curvature",
                    "initialBlockCurvatureMultiplier": 0.4,
                    "cameraDiagonalMetricScale": 75.0,
                    "sharedOnlyCameraProximal": True,
                    "metricProposalDisagreementScale": 0.5,
                }
                for key, value in expected.items():
                    if row.get(key) != value:
                        raise ValueError(
                            f"configuration mismatch {family}/{mode}/{scene}: "
                            f"{key}={row.get(key)!r}, expected={value!r}"
                        )
                completed = row.get("completedIterations", 0)
                if not 0 < completed <= arguments.iterations:
                    raise ValueError(
                        f"invalid completion {family}/{mode}/{scene}: {completed}"
                    )
                if (
                    completed < arguments.iterations
                    and row.get("terminationReason") != "recovery_exhausted"
                ):
                    raise ValueError(
                        f"unexpected early termination {family}/{mode}/{scene}: "
                        f"{completed}/{arguments.iterations}, "
                        f"{row.get('terminationReason')}"
                    )
        plain = modes["none"]
        summary[family] = {}
        for mode, rows in modes.items():
            ratios = [
                rows[scene]["qualityMetrics"]["sumSquaredError"]
                / plain[scene]["qualityMetrics"]["sumSquaredError"]
                for scene in scenes
            ]
            summary[family][mode] = {
                "sse_vs_plain": geometric_mean(ratios),
                "wins": sum(value < 1.0 for value in ratios),
                "losses": sum(value > 1.0 for value in ratios),
                "worst": max(ratios),
                "optimization_seconds": math.fsum(
                    row["optimizationSeconds"] for row in rows.values()
                ),
                "optimization_vs_plain": (
                    math.fsum(row["optimizationSeconds"] for row in rows.values())
                    / math.fsum(row["optimizationSeconds"] for row in plain.values())
                ),
                "oracle_calls": sum(row["proximalOracleCalls"] for row in rows.values()),
                "accelerated_acceptances": sum(
                    row["acceleratedAcceptances"] for row in rows.values()
                ),
                "nominal_fallbacks": sum(row["nominalFallbacks"] for row in rows.values()),
                "completed": sum(
                    row["completedIterations"] == arguments.iterations
                    for row in rows.values()
                ),
                "recovery_exhausted": sum(
                    row["terminationReason"] == "recovery_exhausted"
                    for row in rows.values()
                ),
                "rows": {
                    scene: {
                        "sse": rows[scene]["qualityMetrics"]["sumSquaredError"],
                        "sse_vs_plain": ratios[index],
                    }
                    for index, scene in enumerate(scenes)
                },
            }

    arguments.root.mkdir(parents=True, exist_ok=True)
    (arguments.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = arguments.root / "report.md"
    with report.open("w", encoding="utf-8") as output:
        output.write("# Matched Outer-Acceleration Sentinel Factorial\n\n")
        output.write(
            "All rows use K24/I30, local Nesterov, persistent DRS trust, "
            "curvature 0.4 recovery/decay, camera metric 75, shared-only camera "
            "proximal terms, proposal damping 0.5 on duplicated cameras, and "
            "binary line search `{1,0}`. Only the outer accelerator varies.\n\n"
        )
        output.write(
            "| Family | Accelerator | SSE/plain | W/L | Worst | Opt. s | "
            "Time/plain | Oracles | Accepts | Fallbacks | Completed | "
            "Recovery exhausted |\n"
        )
        output.write(
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for family in ("1dsfm", "bal"):
            for mode in MODES:
                row = summary[family][mode]
                output.write(
                    f"| {family} | {mode} | {row['sse_vs_plain']:.6f} | "
                    f"{row['wins']}/{row['losses']} | {row['worst']:.6f} | "
                    f"{row['optimization_seconds']:.3f} | "
                    f"{row['optimization_vs_plain']:.3f} | {row['oracle_calls']} | "
                    f"{row['accelerated_acceptances']} | {row['nominal_fallbacks']} | "
                    f"{row['completed']} | {row['recovery_exhausted']} |\n"
                )
        output.write("\nNo mode is selected by scene.\n")
    print(report)


if __name__ == "__main__":
    main()
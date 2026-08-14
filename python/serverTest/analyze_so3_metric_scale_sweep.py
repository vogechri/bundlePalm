#!/usr/bin/env python3
"""Analyze product-SO3 camera metric-scale development sweeps."""

import argparse
import json
import math
import re
from pathlib import Path


EXPECTED = {
    "1dsfm": (
        "gendarmenmarkt",
        "piccadilly",
        "roman_forum",
        "trafalgar",
        "union_square",
        "vienna_cathedral",
    ),
    "bal": ("bal52", "bal245", "bal394", "bal871", "bal1723"),
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
    paths = sorted(directory.glob("*.jsonl"))
    if len(paths) != 1:
        raise ValueError(f"expected one JSONL in {directory}, found {len(paths)}")
    rows = {}
    for line in paths[0].read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            scene = scene_key(row)
            if scene in rows:
                raise ValueError(f"duplicate scene {scene} in {paths[0]}")
            rows[scene] = row
    return rows


def sse(row):
    return row["qualityMetrics"]["sumSquaredError"]


def geometric_mean(values):
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def summarize(reference, candidate, scenes):
    ratios = [sse(candidate[scene]) / sse(reference[scene]) for scene in scenes]
    ties = [math.isclose(value, 1.0, rel_tol=1e-12) for value in ratios]
    return {
        "geometric": geometric_mean(ratios),
        "summed": math.fsum(sse(candidate[scene]) for scene in scenes)
        / math.fsum(sse(reference[scene]) for scene in scenes),
        "wins": sum(
            value < 1.0 and not ties[index]
            for index, value in enumerate(ratios)
        ),
        "ties": sum(ties),
        "losses": sum(
            value > 1.0 and not ties[index]
            for index, value in enumerate(ratios)
        ),
        "worst": max(ratios),
        "ratios": dict(zip(scenes, ratios)),
        "optimization": math.fsum(
            candidate[scene]["optimizationSeconds"] for scene in scenes
        ) / math.fsum(
            reference[scene]["optimizationSeconds"] for scene in scenes
        ),
    }


def validate(row, value, iterations, context):
    expected = {
        "clusters": 24,
        "iterations": iterations,
        "cameraUpdate": "so3_left",
        "directTangentNormalEquations": True,
        "cameraDiagonalMetricScale": float(value),
        "localSolver": "schur_pcg",
        "outerAcceleration": "themelis_nesterov",
        "adaptiveLocalDepth": True,
        "adaptiveLocalDepthStart": 5,
    }
    for key, expected_value in expected.items():
        if row.get(key) != expected_value:
            raise ValueError(
                f"configuration mismatch {context}: {key}={row.get(key)!r}, "
                f"expected={expected_value!r}"
            )
    if row.get("completedIterations") != iterations:
        raise ValueError(f"incomplete {context}: {row.get('terminationReason')}")
    maximum_residual = max(
        item["localLinearRelativeResidualMaximum"] for item in row["trajectory"]
    )
    if not math.isfinite(maximum_residual) or maximum_residual >= 0.01:
        raise ValueError(
            f"linear residual gate failed {context}: {maximum_residual}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--left-reference-root", type=Path, required=True)
    parser.add_argument("--values", nargs="+", type=float, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args()
    if 25.0 not in arguments.values:
        raise ValueError("metric scale 25 control is required")

    rows = {}
    left = {}
    for family, scenes in EXPECTED.items():
        left[family] = load(arguments.left_reference_root / family)
        if arguments.require_complete and set(left[family]) != set(scenes):
            raise ValueError(f"left-reference coverage mismatch {family}")
        rows[family] = {}
        for value in arguments.values:
            candidate = load(
                arguments.root / f"metric{value:g}" / family / "c1_c5"
            )
            if arguments.require_complete and set(candidate) != set(scenes):
                raise ValueError(
                    f"coverage mismatch metric={value:g}/{family}: "
                    f"{sorted(candidate)}"
                )
            for scene, row in candidate.items():
                validate(
                    row,
                    value,
                    arguments.iterations,
                    f"metric={value:g}/{family}/{scene}",
                )
            rows[family][value] = candidate

    summary = {}
    for value in arguments.values:
        summary[str(value)] = {}
        for family, scenes in EXPECTED.items():
            candidate = rows[family][value]
            summary[str(value)][family] = {
                "vsScale25": summarize(rows[family][25.0], candidate, scenes),
                "vsLeftSE3": summarize(left[family], candidate, scenes),
                "rejections": sum(row["rejections"] for row in candidate.values()),
                "oracleCalls": sum(
                    row["proximalOracleCalls"] for row in candidate.values()
                ),
                "maximumLinearIterations": max(
                    item["localLinearIterationsMaximum"]
                    for row in candidate.values()
                    for item in row["trajectory"]
                ),
            }

    arguments.root.mkdir(parents=True, exist_ok=True)
    (arguments.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = arguments.root / "report.md"
    with report.open("w", encoding="utf-8") as output:
        output.write("# Product-SO3 Camera Metric-Scale Development Sweep\n\n")
        output.write(
            "Fresh K24/I30 C1+C5 rows vary only the product-SO3 camera "
            "diagonal metric scale. Schur-PCG tolerance is `1e-2`, cap 1000.\n\n"
        )
        output.write(
            "| Scale | 1DSfM/scale25 | 1DSfM/left-SE3 | Worst left | "
            "BAL/scale25 | BAL/left-SE3 | Worst left | Time 1D/BAL | "
            "Rejections | Max PCG |\n"
        )
        output.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for value in arguments.values:
            one = summary[str(value)]["1dsfm"]
            bal = summary[str(value)]["bal"]
            output.write(
                f"| {value:g} | {one['vsScale25']['geometric']:.9f} | "
                f"{one['vsLeftSE3']['geometric']:.9f} | "
                f"{one['vsLeftSE3']['worst']:.9f} | "
                f"{bal['vsScale25']['geometric']:.9f} | "
                f"{bal['vsLeftSE3']['geometric']:.9f} | "
                f"{bal['vsLeftSE3']['worst']:.9f} | "
                f"{one['vsScale25']['optimization']:.3f}/"
                f"{bal['vsScale25']['optimization']:.3f} | "
                f"{one['rejections'] + bal['rejections']} | "
                f"{max(one['maximumLinearIterations'], bal['maximumLinearIterations']):.0f} |\n"
            )
    print(report)


if __name__ == "__main__":
    main()

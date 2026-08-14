#!/usr/bin/env python3
"""Analyze product-SO3 translation/rotation metric-ratio sweeps."""

import argparse
import json
import math
import re
from pathlib import Path


EXPECTED = {
    "1dsfm": (
        "gendarmenmarkt", "piccadilly", "roman_forum", "trafalgar",
        "union_square", "vienna_cathedral",
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
        "wins": sum(value < 1.0 and not ties[i] for i, value in enumerate(ratios)),
        "ties": sum(ties),
        "losses": sum(value > 1.0 and not ties[i] for i, value in enumerate(ratios)),
        "worst": max(ratios),
        "ratios": dict(zip(scenes, ratios)),
        "optimization": math.fsum(candidate[s]["optimizationSeconds"] for s in scenes)
        / math.fsum(reference[s]["optimizationSeconds"] for s in scenes),
    }


def validate(row, ratio, iterations, context):
    expected = {
        "clusters": 24,
        "iterations": iterations,
        "cameraUpdate": "so3_left",
        "directTangentNormalEquations": True,
        "cameraDiagonalMetricScale": 25.0,
        "so3TranslationMetricRatio": ratio,
        "outerAcceleration": "themelis_nesterov",
        "adaptiveLocalDepth": True,
        "adaptiveLocalDepthStart": 5,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(
                f"configuration mismatch {context}: {key}={row.get(key)!r}, "
                f"expected={value!r}"
            )
    if row.get("completedIterations") != iterations:
        raise ValueError(f"incomplete {context}: {row.get('terminationReason')}")
    maximum_residual = max(
        item["localLinearRelativeResidualMaximum"] for item in row["trajectory"]
    )
    if not math.isfinite(maximum_residual) or maximum_residual >= 0.01:
        raise ValueError(f"linear residual gate failed {context}: {maximum_residual}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--left-reference-root", type=Path, required=True)
    parser.add_argument("--values", nargs="+", type=float, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args()
    if 1.0 not in arguments.values:
        raise ValueError("ratio 1 control is required")

    rows = {}
    left = {}
    for family, scenes in EXPECTED.items():
        left[family] = load(arguments.left_reference_root / family)
        rows[family] = {}
        for ratio in arguments.values:
            tag = f"ratio{ratio:g}".replace(".", "p")
            candidate = load(arguments.root / tag / family / "c1_c5")
            if arguments.require_complete and set(candidate) != set(scenes):
                raise ValueError(f"coverage mismatch ratio={ratio:g}/{family}")
            for scene, row in candidate.items():
                validate(row, ratio, arguments.iterations, f"{ratio:g}/{family}/{scene}")
            rows[family][ratio] = candidate

    summary = {}
    for ratio in arguments.values:
        summary[str(ratio)] = {}
        for family, scenes in EXPECTED.items():
            candidate = rows[family][ratio]
            summary[str(ratio)][family] = {
                "vsRatio1": summarize(rows[family][1.0], candidate, scenes),
                "vsLeftSE3": summarize(left[family], candidate, scenes),
                "rejections": sum(row["rejections"] for row in candidate.values()),
                "maximumLinearIterations": max(
                    item["localLinearIterationsMaximum"]
                    for row in candidate.values() for item in row["trajectory"]
                ),
            }

    arguments.root.mkdir(parents=True, exist_ok=True)
    (arguments.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = arguments.root / "report.md"
    with report.open("w", encoding="utf-8") as output:
        output.write("# Product-SO3 Translation Metric-Ratio Development Sweep\n\n")
        output.write(
            "Fresh K24/I30 C1+C5 rows keep camera metric scale 25 and vary "
            "only translation relative to rotation in coherent tangent metrics.\n\n"
        )
        output.write(
            "| Ratio | 1DSfM/ratio1 | 1DSfM/left | Worst left | "
            "BAL/ratio1 | BAL/left | Worst left | Time 1D/BAL | Rejections | Max PCG |\n"
        )
        output.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for ratio in arguments.values:
            one = summary[str(ratio)]["1dsfm"]
            bal = summary[str(ratio)]["bal"]
            output.write(
                f"| {ratio:g} | {one['vsRatio1']['geometric']:.9f} | "
                f"{one['vsLeftSE3']['geometric']:.9f} | {one['vsLeftSE3']['worst']:.9f} | "
                f"{bal['vsRatio1']['geometric']:.9f} | {bal['vsLeftSE3']['geometric']:.9f} | "
                f"{bal['vsLeftSE3']['worst']:.9f} | "
                f"{one['vsRatio1']['optimization']:.3f}/{bal['vsRatio1']['optimization']:.3f} | "
                f"{one['rejections'] + bal['rejections']} | "
                f"{max(one['maximumLinearIterations'], bal['maximumLinearIterations']):.0f} |\n"
            )
    print(report)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze direct-tangent camera parameterization sentinels."""

import argparse
import json
import math
import re
from pathlib import Path


EXPECTED = {
    "sentinel": {
        "1dsfm": ("roman_forum", "trafalgar"),
        "bal": ("bal1490", "bal3068"),
    },
    "development": {
        "1dsfm": (
            "gendarmenmarkt",
            "piccadilly",
            "roman_forum",
            "trafalgar",
            "union_square",
            "vienna_cathedral",
        ),
        "bal": ("bal52", "bal245", "bal394", "bal871", "bal1723"),
    },
    "confirmation": {
        "1dsfm": (
            "alamo",
            "ellis_island",
            "madrid_metropolis",
            "montreal_notre_dame",
            "notre_dame",
            "nyc_library",
            "piazza_del_popolo",
            "tower_of_london",
            "yorkminster",
        ),
        "bal": (
            "bal49", "bal52", "bal88", "bal89", "bal126", "bal135",
            "bal142", "bal173", "bal245", "bal253", "bal257", "bal287",
            "bal308", "bal356", "bal394", "bal427", "bal646", "bal744",
            "bal783", "bal871", "bal931", "bal951", "bal961", "bal1064",
            "bal1266", "bal1490", "bal1723", "bal1778", "bal3068",
        ),
    },
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


def validate(row, mode, iterations, context):
    expected = {
        "clusters": 24,
        "iterations": iterations,
        "cameraUpdate": mode,
        "directTangentNormalEquations": True,
        "localSolver": "schur_pcg",
        "outerAcceleration": "themelis_nesterov",
        "trustRegionPolicy": "daba",
        "persistentTrustRegion": True,
        "proximalMetric": "block",
        "consensusMetric": "full",
        "sharedOnlyCameraProximal": True,
        "adaptiveLocalDepth": True,
        "adaptiveLocalDepthStart": 5,
        "adaptiveLocalDepthMaximum": 2,
        "adaptiveLocalDepthHigh": 0.35,
        "adaptiveLocalDepthLow": 0.20,
        "adaptiveLocalDepthWindow": 3,
        "adaptiveLocalDepthDwell": 3,
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
        raise ValueError(
            f"linear residual gate failed {context}: {maximum_residual}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--cohort", choices=tuple(EXPECTED), default="sentinel")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args()
    if "se3_left" not in arguments.modes:
        raise ValueError("se3_left control is required")

    rows = {}
    expected_scenes = EXPECTED[arguments.cohort]
    for mode in arguments.modes:
        rows[mode] = {}
        for family, scenes in expected_scenes.items():
            family_rows = load(arguments.root / mode / family)
            if arguments.require_complete and set(family_rows) != set(scenes):
                raise ValueError(
                    f"coverage mismatch {mode}/{family}: {sorted(family_rows)}"
                )
            for scene, row in family_rows.items():
                validate(
                    row,
                    mode,
                    arguments.iterations,
                    f"{mode}/{family}/{scene}",
                )
            rows[mode][family] = family_rows

    summary = {}
    for mode in arguments.modes:
        summary[mode] = {}
        for family, scenes in expected_scenes.items():
            control = rows["se3_left"][family]
            candidate = rows[mode][family]
            ratios = [sse(candidate[scene]) / sse(control[scene]) for scene in scenes]
            ties = [math.isclose(value, 1.0, rel_tol=1e-12) for value in ratios]
            summary[mode][family] = {
                "geometricSSEVsLeftSE3": geometric_mean(ratios),
                "summedSSEVsLeftSE3": (
                    math.fsum(sse(candidate[scene]) for scene in scenes)
                    / math.fsum(sse(control[scene]) for scene in scenes)
                ),
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
                "optimizationSeconds": math.fsum(
                    candidate[scene]["optimizationSeconds"] for scene in scenes
                ),
                "optimizationVsLeftSE3": (
                    math.fsum(
                        candidate[scene]["optimizationSeconds"] for scene in scenes
                    )
                    / math.fsum(
                        control[scene]["optimizationSeconds"] for scene in scenes
                    )
                ),
                "rejections": sum(candidate[scene]["rejections"] for scene in scenes),
                "oracleCalls": sum(
                    candidate[scene]["proximalOracleCalls"] for scene in scenes
                ),
                "scenes": {
                    scene: {
                        "sse": sse(candidate[scene]),
                        "sseVsLeftSE3": ratios[index],
                        "optimizationSeconds": candidate[scene]["optimizationSeconds"],
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
        output.write("# Direct-Tangent Camera Parameterization Gate\n\n")
        output.write(
            "Fresh K24/I30 runs use the promoted Stage-C C1+C5 stack. The only "
            "change is the camera retraction and its matching direct tangent "
            "Jacobian: left SE(3), product SO(3) x R3, or right SE(3).\n\n"
        )
        output.write(
            "| Family | Mode | SSE/left-SE3 | Summed/left-SE3 | W/T/L | "
            "Worst | Time/left-SE3 | Rejections | Oracles |\n"
        )
        output.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for family in ("1dsfm", "bal"):
            for mode in arguments.modes:
                values = summary[mode][family]
                output.write(
                    f"| {family} | {mode} | "
                    f"{values['geometricSSEVsLeftSE3']:.9f} | "
                    f"{values['summedSSEVsLeftSE3']:.9f} | "
                    f"{values['wins']}/{values['ties']}/{values['losses']} | "
                    f"{values['worst']:.9f} | "
                    f"{values['optimizationVsLeftSE3']:.6f} | "
                    f"{values['rejections']} | {values['oracleCalls']} |\n"
                )
        output.write("\nNo scene-specific parameterization is selected.\n")
    print(report)


if __name__ == "__main__":
    main()

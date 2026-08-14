#!/usr/bin/env python3
"""Analyze the one-factor DRS parameter sensitivity gate."""

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
}
CONTROL = {
    "block": 5e-5,
    "restart": 3,
    "cap": 10.0,
    "exponent": 4.0,
    "dre": 0.01,
}
OVERRIDES = {
    "control": {},
    "block_2p5em5": {"block": 2.5e-5},
    "block_1em4": {"block": 1e-4},
    "restart_1": {"restart": 1},
    "restart_5": {"restart": 5},
    "cap_3": {"cap": 3.0},
    "exponent_2": {"exponent": 2.0},
    "exponent_8": {"exponent": 8.0},
    "dre_0p005": {"dre": 0.005},
    "dre_0p02": {"dre": 0.02},
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


def validate(row, settings, iterations, context):
    expected = {
        "clusters": 24,
        "iterations": iterations,
        "localSolver": "schur_pcg",
        "outerAcceleration": "themelis_nesterov",
        "lineSearchGrid": "0,1",
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
        "initialBlockRegularization": settings["block"],
        "accelerationRestartAfter": settings["restart"],
        "accelerationMaximumStepRatio": settings["cap"],
        "safeguardAnnealingIterations": iterations,
        "safeguardReferenceIteration": 5,
        "safeguardAnnealingExponent": settings["exponent"],
        "dreRelativeIncrease": settings["dre"],
    }
    for key, value in expected.items():
        actual = row.get(key)
        matches = (
            math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-15)
            if isinstance(value, float) and isinstance(actual, (int, float))
            else actual == value
        )
        if not matches:
            raise ValueError(
                f"configuration mismatch {context}: {key}={actual!r}, "
                f"expected={value!r}"
            )
    if row.get("completedIterations") != iterations:
        raise ValueError(
            f"incomplete {context}: {row.get('completedIterations')}/{iterations}, "
            f"{row.get('terminationReason')}"
        )


def sse(row):
    return row["qualityMetrics"]["sumSquaredError"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--cohort", choices=tuple(EXPECTED), default="sentinel")
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args()

    unknown = set(arguments.arms) - set(OVERRIDES)
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}")
    if "control" not in arguments.arms:
        raise ValueError("the control arm is required")

    rows = {}
    expected_scenes = EXPECTED[arguments.cohort]
    for arm in arguments.arms:
        settings = {**CONTROL, **OVERRIDES[arm]}
        rows[arm] = {}
        for family, scenes in expected_scenes.items():
            family_rows = load(arguments.root / arm / family)
            if arguments.require_complete and set(family_rows) != set(scenes):
                raise ValueError(
                    f"coverage mismatch {arm}/{family}: {sorted(family_rows)}"
                )
            for scene, row in family_rows.items():
                validate(
                    row,
                    settings,
                    arguments.iterations,
                    f"{arm}/{family}/{scene}",
                )
            rows[arm][family] = family_rows

    summary = {}
    for arm in arguments.arms:
        summary[arm] = {}
        for family, scenes in expected_scenes.items():
            control = rows["control"][family]
            candidate = rows[arm][family]
            ratios = [sse(candidate[scene]) / sse(control[scene]) for scene in scenes]
            candidate_seconds = math.fsum(
                candidate[scene]["optimizationSeconds"] for scene in scenes
            )
            control_seconds = math.fsum(
                control[scene]["optimizationSeconds"] for scene in scenes
            )
            summary[arm][family] = {
                "geometricSSEVsControl": geometric_mean(ratios),
                "summedSSEVsControl": (
                    math.fsum(sse(candidate[scene]) for scene in scenes)
                    / math.fsum(sse(control[scene]) for scene in scenes)
                ),
                "wins": sum(value < 1.0 for value in ratios),
                "ties": sum(math.isclose(value, 1.0, rel_tol=1e-12) for value in ratios),
                "losses": sum(value > 1.0 for value in ratios),
                "worst": max(ratios),
                "optimizationSeconds": candidate_seconds,
                "optimizationVsControl": candidate_seconds / control_seconds,
                "oracleCalls": sum(candidate[scene]["proximalOracleCalls"] for scene in scenes),
                "acceleratedAcceptances": sum(
                    candidate[scene]["acceleratedAcceptances"] for scene in scenes
                ),
                "nominalFallbacks": sum(
                    candidate[scene]["nominalFallbacks"] for scene in scenes
                ),
                "stepLimitHits": sum(
                    candidate[scene]["accelerationStepLimitHits"] for scene in scenes
                ),
                "scenes": {
                    scene: {
                        "sse": sse(candidate[scene]),
                        "sseVsControl": ratios[index],
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
        output.write("# DRS Parameter Sensitivity Gate\n\n")
        output.write(
            "Fresh K24/I30 runs use the promoted Stage-C C1+C5 stack. Each arm "
            "changes exactly one global parameter from the shared control.\n\n"
        )
        output.write(
            "| Family | Arm | Geomean SSE/control | Summed SSE/control | "
            "W/T/L | Worst | Time/control | Oracles | Accepts | Fallbacks | "
            "Step-cap hits |\n"
        )
        output.write(
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for family in ("1dsfm", "bal"):
            for arm in arguments.arms:
                values = summary[arm][family]
                output.write(
                    f"| {family} | {arm} | "
                    f"{values['geometricSSEVsControl']:.9f} | "
                    f"{values['summedSSEVsControl']:.9f} | "
                    f"{values['wins']}/{values['ties']}/{values['losses']} | "
                    f"{values['worst']:.9f} | "
                    f"{values['optimizationVsControl']:.6f} | "
                    f"{values['oracleCalls']} | "
                    f"{values['acceleratedAcceptances']} | "
                    f"{values['nominalFallbacks']} | "
                    f"{values['stepLimitHits']} |\n"
                )
        output.write(
            "\nThis gate is a safety filter, not a promotion cohort. A parameter "
            "advances only as one frozen global value to the established six-"
            "1DSfM plus five-BAL development gate.\n"
        )
    print(report)


if __name__ == "__main__":
    main()

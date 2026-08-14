#!/usr/bin/env python3
"""Analyze the product-SO3 C1 x C5 development factorial."""

import argparse
import json
import math
import re
from pathlib import Path


MODES = ("plain", "c1", "c5", "c1_c5")
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


def load_directory(directory):
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


def comparison(reference, candidate, scenes):
    ratios = [sse(candidate[scene]) / sse(reference[scene]) for scene in scenes]
    ties = [math.isclose(value, 1.0, rel_tol=1e-12) for value in ratios]
    return {
        "geometric": geometric_mean(ratios),
        "summed": (
            math.fsum(sse(candidate[scene]) for scene in scenes)
            / math.fsum(sse(reference[scene]) for scene in scenes)
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
        "ratios": dict(zip(scenes, ratios)),
    }


def validate(row, mode, iterations, context):
    expected = {
        "clusters": 24,
        "iterations": iterations,
        "cameraUpdate": "so3_left",
        "directTangentNormalEquations": True,
        "localSolver": "schur_pcg",
        "trustRegionPolicy": "daba",
        "persistentTrustRegion": True,
        "outerAcceleration": (
            "themelis_nesterov" if mode in ("c1", "c1_c5") else "none"
        ),
        "adaptiveLocalDepth": mode in ("c5", "c1_c5"),
        "sharedOnlyCameraProximal": True,
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


def write_comparison(output, label, values):
    output.write(
        f"| {label} | {values['geometric']:.9f} | "
        f"{values['summed']:.9f} | "
        f"{values['wins']}/{values['ties']}/{values['losses']} | "
        f"{values['worst']:.9f} |\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--left-reference-root", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args()

    rows = {}
    left_reference = {}
    for family, scenes in EXPECTED.items():
        rows[family] = {
            mode: load_directory(arguments.root / family / mode)
            for mode in MODES
        }
        left_reference[family] = load_directory(
            arguments.left_reference_root / family
        )
        for mode, mode_rows in rows[family].items():
            if arguments.require_complete and set(mode_rows) != set(scenes):
                raise ValueError(
                    f"coverage mismatch {family}/{mode}: {sorted(mode_rows)}"
                )
            for scene, row in mode_rows.items():
                validate(row, mode, arguments.iterations, f"{family}/{mode}/{scene}")
        if arguments.require_complete and set(left_reference[family]) != set(scenes):
            raise ValueError(
                f"left-reference coverage mismatch {family}: "
                f"{sorted(left_reference[family])}"
            )

    summary = {}
    for family, scenes in EXPECTED.items():
        family_rows = rows[family]
        effects = {
            "c1_vs_plain": comparison(
                family_rows["plain"], family_rows["c1"], scenes
            ),
            "c5_vs_plain": comparison(
                family_rows["plain"], family_rows["c5"], scenes
            ),
            "c1_c5_vs_plain": comparison(
                family_rows["plain"], family_rows["c1_c5"], scenes
            ),
            "c1_c5_vs_c1": comparison(
                family_rows["c1"], family_rows["c1_c5"], scenes
            ),
            "c1_c5_vs_c5": comparison(
                family_rows["c5"], family_rows["c1_c5"], scenes
            ),
            "so3_c1_c5_vs_left_c1_c5": comparison(
                left_reference[family], family_rows["c1_c5"], scenes
            ),
        }
        interaction_ratios = {
            scene: (
                sse(family_rows["c1_c5"][scene])
                * sse(family_rows["plain"][scene])
                / (
                    sse(family_rows["c1"][scene])
                    * sse(family_rows["c5"][scene])
                )
            )
            for scene in scenes
        }
        effects["interaction"] = {
            "geometric": geometric_mean(interaction_ratios.values()),
            "ratios": interaction_ratios,
        }
        effects["optimizationSeconds"] = {
            mode: math.fsum(
                row["optimizationSeconds"] for row in family_rows[mode].values()
            )
            for mode in MODES
        }
        effects["oracleCalls"] = {
            mode: sum(
                row["proximalOracleCalls"] for row in family_rows[mode].values()
            )
            for mode in MODES
        }
        summary[family] = effects

    arguments.root.mkdir(parents=True, exist_ok=True)
    (arguments.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = arguments.root / "report.md"
    with report.open("w", encoding="utf-8") as output:
        output.write("# Product-SO3 C1 x C5 Development Factorial\n\n")
        output.write(
            "Fresh K24/I30 rows use product `SO(3) x R3`, direct tangent "
            "equations, Schur-PCG tolerance `1e-2` and cap 1000. Plain, C1, "
            "C5, and C1+C5 differ only by safeguarded outer acceleration and "
            "adaptive local depth.\n\n"
        )
        for family in ("1dsfm", "bal"):
            output.write(f"## {family}\n\n")
            output.write("| Comparison | Geomean SSE | Summed SSE | W/T/L | Worst |\n")
            output.write("|---|---:|---:|---:|---:|\n")
            for key, label in (
                ("c1_vs_plain", "C1 / plain"),
                ("c5_vs_plain", "C5 / plain"),
                ("c1_c5_vs_plain", "C1+C5 / plain"),
                ("c1_c5_vs_c1", "C1+C5 / C1"),
                ("c1_c5_vs_c5", "C1+C5 / C5"),
                ("so3_c1_c5_vs_left_c1_c5", "SO3 C1+C5 / left-SE3 C1+C5"),
            ):
                write_comparison(output, label, summary[family][key])
            output.write(
                f"\nC1 x C5 interaction geometric ratio: "
                f"`{summary[family]['interaction']['geometric']:.9f}`.\n\n"
            )
            output.write(
                "| Scene | C1/plain | C5/plain | C1+C5/plain | "
                "C1+C5/C1 | Interaction | SO3/left-SE3 |\n"
            )
            output.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for scene in EXPECTED[family]:
                output.write(
                    f"| {scene} | "
                    f"{summary[family]['c1_vs_plain']['ratios'][scene]:.9f} | "
                    f"{summary[family]['c5_vs_plain']['ratios'][scene]:.9f} | "
                    f"{summary[family]['c1_c5_vs_plain']['ratios'][scene]:.9f} | "
                    f"{summary[family]['c1_c5_vs_c1']['ratios'][scene]:.9f} | "
                    f"{summary[family]['interaction']['ratios'][scene]:.9f} | "
                    f"{summary[family]['so3_c1_c5_vs_left_c1_c5']['ratios'][scene]:.9f} |\n"
                )
            output.write("\n")
    print(report)


if __name__ == "__main__":
    main()

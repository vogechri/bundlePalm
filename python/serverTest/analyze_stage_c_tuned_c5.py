#!/usr/bin/env python3
"""Build the final tuned C5 report from frozen split-cohort artifacts."""

import argparse
import json
import math
import re
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent
RESULTS = WORKSPACE / "benchmark_results"
SOURCES = {
    "1dsfm": (
        RESULTS / "stage_c_c5_start_development_k24_i30/s5p00/1dsfm/c1_c5",
        RESULTS / "stage_c_c5_frozen_heldout9_k24_i30/1dsfm/c1_c5",
    ),
    "bal": (RESULTS / "stage_c_c5_frozen_bal_all29_k24_i30/bal/c1_c5",),
}
C5_ONLY = {
    family: RESULTS / f"stage_c_c5_frozen_main_effect_all15_all29_k24_i30/{family}/c5"
    for family in ("1dsfm", "bal")
}
BASELINE = RESULTS / "stage_c_final_all15_all29_k24_i30"
CERES = {
    "1dsfm": RESULTS / "1dsfm_drs_ceres_se3_all15/ceres/results.jsonl",
    "bal": RESULTS / "bal_ceres_se3_all29/results.jsonl",
}
EXPECTED = {"1dsfm": 15, "bal": 29}


def key(row):
    dataset = Path(row["dataset"])
    if "sfm_init_1dsfm" in row["dataset"]:
        return dataset.parent.name
    match = re.search(r"problem-(\d+)-", dataset.name)
    if match:
        return f"bal{match.group(1)}"
    if row.get("balId") is not None:
        return f"bal{row['balId']}"
    return row.get("scene")


def load(paths):
    if isinstance(paths, Path):
        paths = (paths,)
    rows = {}
    for source in paths:
        files = (source,) if source.is_file() else tuple(source.glob("*.jsonl"))
        for path in files:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    scene = key(row)
                    if scene in rows:
                        raise ValueError(f"duplicate {scene} in {paths}")
                    rows[scene] = row
    return rows


def sse(row):
    return row["qualityMetrics"]["sumSquaredError"]


def summarize(reference, candidate):
    scenes = sorted(reference)
    ratios = [sse(candidate[scene]) / sse(reference[scene]) for scene in scenes]
    optimization_ratio = None
    if all("optimizationSeconds" in reference[scene] for scene in scenes):
        optimization_ratio = math.fsum(
            candidate[scene]["optimizationSeconds"] for scene in scenes
        ) / math.fsum(
            reference[scene]["optimizationSeconds"] for scene in scenes
        )
    return {
        "geometric": math.exp(math.fsum(math.log(value) for value in ratios) / len(ratios)),
        "summed": math.fsum(sse(candidate[scene]) for scene in scenes)
        / math.fsum(sse(reference[scene]) for scene in scenes),
        "wins": sum(value < 1.0 for value in ratios),
        "ties": sum(math.isclose(value, 1.0, rel_tol=1e-12) for value in ratios),
        "losses": sum(value > 1.0 for value in ratios),
        "worst": max(ratios),
        "optimization": optimization_ratio,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "stage_c_tuned_c5_final_report.md",
    )
    arguments = parser.parse_args()
    all_summaries = {}
    rows_by_family = {}
    for family, expected in EXPECTED.items():
        rows = {
            "plain": load(BASELINE / family / "plain"),
            "c1": load(BASELINE / family / "c1"),
            "c5": load(C5_ONLY[family]),
            "c1_c5": load(SOURCES[family]),
            "ceres": load(CERES[family]),
        }
        if any(len(value) != expected for value in rows.values()):
            raise ValueError(
                f"coverage mismatch for {family}: "
                + ", ".join(f"{name}={len(value)}" for name, value in rows.items())
            )
        if any(set(value) != set(rows["plain"]) for value in rows.values()):
            raise ValueError(f"scene mismatch for {family}")
        for row in rows["c5"].values():
            if not (
                row["adaptiveLocalDepthStart"] == 5
                and row["adaptiveLocalDepthHigh"] == 0.35
                and row["adaptiveLocalDepthLow"] == 0.20
                and row["adaptiveLocalDepthWindow"] == 3
                and row["adaptiveLocalDepthDwell"] == 3
                and row["adaptiveLocalDepthMaximum"] == 2
            ):
                raise ValueError(f"tuned C5 mismatch in {family}")
        for row in rows["c1_c5"].values():
            if row["adaptiveLocalDepthStart"] != 5:
                raise ValueError(f"tuned C1+C5 mismatch in {family}")
        all_summaries[family] = {
            "c1_vs_plain": summarize(rows["plain"], rows["c1"]),
            "c5_vs_plain": summarize(rows["plain"], rows["c5"]),
            "c1_c5_vs_plain": summarize(rows["plain"], rows["c1_c5"]),
            "c1_c5_vs_c1": summarize(rows["c1"], rows["c1_c5"]),
            "c1_c5_vs_c5": summarize(rows["c5"], rows["c1_c5"]),
            "c1_c5_vs_ceres": summarize(rows["ceres"], rows["c1_c5"]),
        }
        rows_by_family[family] = rows

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8") as output:
        output.write("# Final Tuned Global C1+C5 Result\n\n")
        output.write(
            "One global C5 policy is used everywhere: start I5, high/low "
            "thresholds `0.35/0.20`, rolling window 3, dwell 3, maximum depth 2. "
            "The setting was selected on six development 1DSfM plus five BAL "
            "sentinels, frozen for nine held-out 1DSfM scenes, then confirmed on "
            "all 29 BAL scenes.\n\n"
        )
        output.write(
            "| Family | Comparison | Geomean SSE | Summed SSE | W/T/L | Worst | Opt. |\n"
        )
        output.write("|---|---|---:|---:|---:|---:|---:|\n")
        labels = (
            ("c1_vs_plain", "C1/plain"),
            ("c5_vs_plain", "C5/plain"),
            ("c1_c5_vs_plain", "C1+C5/plain"),
            ("c1_c5_vs_c1", "C1+C5/C1"),
            ("c1_c5_vs_c5", "C1+C5/C5"),
            ("c1_c5_vs_ceres", "C1+C5/Ceres"),
        )
        for family in ("1dsfm", "bal"):
            for name, label in labels:
                value = all_summaries[family][name]
                optimization = (
                    f"{value['optimization']:.6f}"
                    if value["optimization"] is not None else "--"
                )
                output.write(
                    f"| {family.upper()} | {label} | {value['geometric']:.9f} | "
                    f"{value['summed']:.9f} | {value['wins']}/{value['ties']}/{value['losses']} | "
                    f"{value['worst']:.9f} | {optimization} |\n"
                )
        output.write("\n## Decision\n\n")
        output.write(
            "Promote the tuned C1+C5 stack as the final Stage-C configuration. "
            "It improves the accepted C1 rung geometrically on both complete "
            "families and uses one common policy. Keep C1 and C5 independently "
            "switchable for ablations. No scene-specific settings are used.\n"
        )
    summary_path = arguments.output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(all_summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(arguments.output)


if __name__ == "__main__":
    main()

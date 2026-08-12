#!/usr/bin/env python3
"""Reduce the frozen final-Schur all-15/all-29 confirmation."""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "serverTest"))
from analyze_stage_c_base_backbone_factorial import (  # noqa: E402
    base_checkpoints,
    load_jsonl,
    scene_key,
)
from analyze_stage_c_final_benchmarks import DEFAULT_CERES  # noqa: E402

EXPECTED = {"1dsfm": 15, "bal": 29}
HISTORICAL_BASE_ITERATIONS = {"1dsfm": 200, "bal": 90}


def geometric_mean(values):
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def sse(row):
    return row["qualityMetrics"]["sumSquaredError"]


def load_candidate(directory):
    paths = tuple(directory.glob("*.jsonl"))
    if len(paths) != 1:
        raise ValueError(f"expected one JSONL in {directory}, found {len(paths)}")
    return load_jsonl(paths[0])


def compare(candidate_values, reference_values):
    ratios = {
        scene: candidate_values[scene] / reference_values[scene]
        for scene in sorted(candidate_values)
    }
    return {
        "geometric": geometric_mean(ratios.values()),
        "summed": math.fsum(candidate_values.values()) / math.fsum(reference_values.values()),
        "wins": sum(value < 1.0 and not math.isclose(value, 1.0, rel_tol=1e-12) for value in ratios.values()),
        "ties": sum(math.isclose(value, 1.0, rel_tol=1e-12) for value in ratios.values()),
        "losses": sum(value > 1.0 and not math.isclose(value, 1.0, rel_tol=1e-12) for value in ratios.values()),
        "worst": max(ratios.values()),
        "ratios": ratios,
    }


def load_peak_rss(path):
    values = {}
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            dataset = Path(row["dataset"])
            if "sfm_init_1dsfm" in row["dataset"]:
                scene = dataset.parent.name
            else:
                scene = f"bal{dataset.name.split('-')[1]}"
            if row["status"] != "completed":
                raise ValueError(f"non-completed status row: {scene}={row['status']}")
            values[scene] = {
                "coordinator": int(row["coordinator_max_rss_kb"]),
                "worker": int(row["worker_max_rss_kb"]),
                "elapsed": float(row["elapsed_seconds"]),
            }
    return values


def validate_candidate(family, rows):
    if len(rows) != EXPECTED[family]:
        raise ValueError(f"coverage mismatch {family}: {len(rows)}/{EXPECTED[family]}")
    for scene, row in rows.items():
        expected = {
            "clusters": 24,
            "iterations": 30,
            "completedIterations": 30,
            "localSolver": "nesterov",
            "outerAcceleration": "themelis_nesterov",
            "trustRegionPolicy": "drs",
            "persistentTrustRegion": True,
            "sharedOnlyCameraProximal": True,
            "metricProposalDisagreementScale": 0.5,
            "sharedSchurMaximumCorrections": 3,
            "sharedSchurOperator": "bsr_low_memory",
            "sharedSchurMinimumRelativeDecrease": 1e-3,
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(
                    f"configuration mismatch {family}/{scene}: "
                    f"{key}={row.get(key)!r}, expected={value!r}"
                )
        if not row.get("finalSharedSchurAttempted"):
            raise ValueError(f"Schur not attempted: {family}/{scene}")
        attempts = row["finalSharedSchurAttempts"]
        accepted = [attempt for attempt in attempts if attempt["accepted"]]
        if len(accepted) != row["finalSharedSchurAcceptedCorrections"]:
            raise ValueError(f"accepted correction mismatch: {family}/{scene}")
        if not accepted:
            raise ValueError(f"no accepted correction: {family}/{scene}")
        for attempt in accepted:
            if attempt["diagnostics"]["linearTermination"] != 0:
                raise ValueError(f"accepted nonconverged solve: {family}/{scene}")
            disagreement = abs(attempt["workerSSE"] - attempt["candidateSSE"])
            if disagreement > 1e-7 * max(1.0, abs(attempt["candidateSSE"])):
                raise ValueError(f"worker/evaluator disagreement: {family}/{scene}")
        corrected = row["finalSharedSchurCorrectedSSE"]
        if not math.isclose(sse(row), corrected, rel_tol=1e-12):
            raise ValueError(f"reported final SSE mismatch: {family}/{scene}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    summary = {}
    for family in ("1dsfm", "bal"):
        rows = load_candidate(arguments.root / family)
        validate_candidate(family, rows)
        scenes = tuple(sorted(rows))
        ceres_rows = load_jsonl(DEFAULT_CERES[family])
        if not set(scenes).issubset(ceres_rows):
            raise ValueError(f"missing Ceres references for {family}")
        base_rows = base_checkpoints(family, scenes, 30)
        historical_base_rows = base_checkpoints(
            family, scenes, HISTORICAL_BASE_ITERATIONS[family]
        )
        resources = load_peak_rss(arguments.root / family / "status.tsv")
        if set(resources) != set(scenes):
            raise ValueError(f"status coverage mismatch for {family}")
        pre = {scene: rows[scene]["finalSharedSchurInitialSSE"] for scene in scenes}
        final = {scene: rows[scene]["finalSharedSchurCorrectedSSE"] for scene in scenes}
        ceres = {scene: sse(ceres_rows[scene]) for scene in scenes}
        base = {scene: sse(base_rows[scene]) for scene in scenes}
        historical_base = {
            scene: sse(historical_base_rows[scene]) for scene in scenes
        }
        summary[family] = {
            "scenes": list(scenes),
            "corrected_vs_pre_schur": compare(final, pre),
            "pre_schur_vs_base_i30": compare(pre, base),
            "corrected_vs_base_i30": compare(final, base),
            "corrected_vs_historical_base": compare(final, historical_base),
            "pre_schur_vs_ceres": compare(pre, ceres),
            "corrected_vs_ceres": compare(final, ceres),
            "accepted_correction_counts": dict(sorted(Counter(
                row["finalSharedSchurAcceptedCorrections"] for row in rows.values()
            ).items())),
            "termination_counts": dict(sorted(Counter(
                row["finalSharedSchurTermination"] for row in rows.values()
            ).items())),
            "schur_seconds": math.fsum(row["finalSharedSchurSeconds"] for row in rows.values()),
            "optimization_seconds": math.fsum(row["optimizationSeconds"] for row in rows.values()),
            "elapsed_seconds": math.fsum(value["elapsed"] for value in resources.values()),
            "maximum_coordinator_rss_kb": max(value["coordinator"] for value in resources.values()),
            "maximum_worker_rss_kb": max(value["worker"] for value in resources.values()),
        }
    serializable = json.loads(json.dumps(summary))
    (arguments.root / "summary.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (arguments.root / "report.md").open("w", encoding="utf-8") as output:
        output.write("# Final Global-Schur Breadth Confirmation\n\n")
        output.write(
            "Frozen K24/I30 base-backbone Themelis trajectory followed by at most "
            "three accepted `bsr_low_memory` global Schur corrections. The global "
            "progress stop is `1e-3`; damping starts at 3/3. No scene-specific "
            "settings are used.\n\n"
        )
        output.write("| Family | Completed | Corrected/pre | Corrected/base-I30 | Corrected/quality base | Corrected/Ceres | W/T/L vs Ceres | Schur s | Max coord. RSS |\n")
        output.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for family, label in (("1dsfm", "1DSfM"), ("bal", "BAL")):
            row = summary[family]
            versus_ceres = row["corrected_vs_ceres"]
            output.write(
                f"| {label} | {len(row['scenes'])}/{EXPECTED[family]} | "
                f"{row['corrected_vs_pre_schur']['geometric']:.9f} | "
                f"{row['corrected_vs_base_i30']['geometric']:.9f} | "
                f"{row['corrected_vs_historical_base']['geometric']:.9f} | "
                f"{versus_ceres['geometric']:.9f} | "
                f"{versus_ceres['wins']}/{versus_ceres['ties']}/{versus_ceres['losses']} | "
                f"{row['schur_seconds']:.3f} | "
                f"{row['maximum_coordinator_rss_kb'] / 1048576:.3f} GiB |\n"
            )
        output.write("\n## Correction Behavior\n\n")
        for family, label in (("1dsfm", "1DSfM"), ("bal", "BAL")):
            row = summary[family]
            output.write(
                f"- **{label}:** accepted-count distribution "
                f"`{row['accepted_correction_counts']}`; termination distribution "
                f"`{row['termination_counts']}`; pre-Schur/Ceres "
                f"`{row['pre_schur_vs_ceres']['geometric']:.9f}x`; corrected/Ceres "
                f"`{row['corrected_vs_ceres']['geometric']:.9f}x`.\n"
            )
        output.write("\n## Decision\n\n")
        output.write(
            "The frozen polishing policy improves every matched I30 endpoint. "
            "It is a strong 1DSfM quality component: all 15 scenes accept all "
            "three corrections, with a `0.864158414x` geometric endpoint ratio "
            "to the I30 handoff. It remains `1.116626283x` the established "
            "I200 base, so three corrections do not yet replace the long-horizon "
            "quality control.\n\n"
        )
        output.write(
            "On BAL, the global `1e-3` stop avoids almost all extra correction "
            "work: 27/29 scenes stop after one correction and only BAL135/BAL142 "
            "accept a second. The endpoint is `0.999319770x` its I30 handoff but "
            "`1.012639985x` the established I90 base. Peak coordinator RSS is "
            "12.134 GiB on BAL961. Retain this as separately labeled polishing; "
            "do not promote it as a replacement for either long-horizon base.\n"
        )
        output.write("\n## Per-Scene Corrected/Ceres Ratios\n\n")
        for family, label in (("1dsfm", "1DSfM"), ("bal", "BAL")):
            output.write(f"### {label}\n\n")
            ratios = summary[family]["corrected_vs_ceres"]["ratios"]
            for scene, ratio in sorted(ratios.items(), key=lambda item: item[1], reverse=True):
                output.write(f"- `{scene}`: `{ratio:.9f}x`\n")
            if family == "1dsfm":
                output.write("\n")


if __name__ == "__main__":
    main()

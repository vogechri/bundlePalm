#!/usr/bin/env python3
"""Analyze the base-backbone C1/C5 sentinel factorial."""

import argparse
import json
import math
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "serverTest"))
from analyze_drs_trajectory_checkpoints import checkpoint_row  # noqa: E402


MODES = ("plain", "c1", "c5", "c1_c5")
EXPECTED = {
    "1dsfm": ("roman_forum", "trafalgar"),
    "bal": ("bal52", "bal3068"),
}
BASE_PATHS = {
    "1dsfm": (
        ROOT / "benchmark_results/1dsfm_drs_ceres_se3_all15/drs/"
        "nesterov_ls01_block_full_se3_left_diag_metric75_lip0.4_"
        "metric_proposal0.5_curvature_persistent_tr_trust_drs_enhanced30_decay5.jsonl"
    ),
    "bal": (
        ROOT / "benchmark_results/drs_29_scene_candidate_k24_coordinator_i90/"
        "nesterov_ls01_block_full_lip0.1_curvature_persistent_tr_decay5.jsonl"
    ),
}


def scene_key(row):
    dataset = Path(row["dataset"])
    if "sfm_init_1dsfm" in row["dataset"]:
        return dataset.parent.name
    match = re.search(r"problem-(\d+)-", dataset.name)
    if not match:
        raise ValueError(f"cannot identify dataset: {dataset}")
    return f"bal{match.group(1)}"


def load_jsonl(path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scene = scene_key(row)
        if scene in rows:
            raise ValueError(f"duplicate scene {scene} in {path}")
        rows[scene] = row
    return rows


def load_directory(directory):
    paths = tuple(directory.glob("*.jsonl"))
    if len(paths) != 1:
        raise ValueError(f"expected one JSONL in {directory}, found {len(paths)}")
    return load_jsonl(paths[0])


def geometric_mean(values):
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def compare(candidate, reference, scenes):
    ratios = [
        candidate[scene]["qualityMetrics"]["sumSquaredError"]
        / reference[scene]["qualityMetrics"]["sumSquaredError"]
        for scene in scenes
    ]
    return {
        "geometric_sse": geometric_mean(ratios),
        "wins": sum(value < 1.0 for value in ratios),
        "losses": sum(value > 1.0 for value in ratios),
        "worst": max(ratios),
    }


def base_checkpoints(family, scenes, iterations):
    rows = load_jsonl(BASE_PATHS[family])
    checkpoints = {}
    for scene in scenes:
        checkpoint = checkpoint_row(rows[scene], iterations)
        if checkpoint is None:
            raise ValueError(f"missing base checkpoint {family}/{scene}/I{iterations}")
        checkpoints[scene] = {
            "qualityMetrics": {"sumSquaredError": checkpoint["best_sse"]},
            "optimizationSeconds": checkpoint["optimization_seconds"],
        }
    return checkpoints


def validate_configuration(row, mode, iterations):
    expected = {
        "clusters": 24,
        "iterations": iterations,
        "localSolver": "nesterov",
        "trustRegionPolicy": "drs",
        "persistentTrustRegion": True,
        "blockRecoveryMode": "curvature",
        "initialBlockCurvatureMultiplier": 0.4,
        "cameraDiagonalMetricScale": 75.0,
        "metricProposalDisagreementScale": 1.0,
        "sharedOnlyCameraProximal": True,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(
                f"configuration mismatch {mode}/{scene_key(row)}: "
                f"{key}={row.get(key)!r}, expected {value!r}"
            )
    completed = row.get("completedIterations", 0)
    if not 0 < completed <= iterations:
        raise ValueError(f"invalid completion for {mode}/{scene_key(row)}: {completed}")
    if completed < iterations and row.get("terminationReason") != "recovery_exhausted":
        raise ValueError(
            f"unexpected early termination for {mode}/{scene_key(row)}: "
            f"{completed}/{iterations}, {row.get('terminationReason')}"
        )
    expected_acceleration = "themelis_nesterov" if mode in ("c1", "c1_c5") else "none"
    if row.get("outerAcceleration") != expected_acceleration:
        raise ValueError(f"C1 mismatch for {mode}/{scene_key(row)}")
    if bool(row.get("adaptiveLocalDepth")) != (mode in ("c5", "c1_c5")):
        raise ValueError(f"C5 mismatch for {mode}/{scene_key(row)}")


def analyze_family(root, family, iterations):
    scenes = EXPECTED[family]
    modes = {
        mode: load_directory(root / family / mode)
        for mode in MODES
    }
    for mode, rows in modes.items():
        if set(rows) != set(scenes):
            raise ValueError(
                f"coverage mismatch {family}/{mode}: {sorted(rows)}"
            )
        for row in rows.values():
            validate_configuration(row, mode, iterations)
    base = base_checkpoints(family, scenes, iterations)
    return {
        "scenes": list(scenes),
        "modes": {
            mode: {
                "versus_hybrid_plain": compare(rows, modes["plain"], scenes),
                "versus_base_i30": compare(rows, base, scenes),
                "optimization_seconds": math.fsum(
                    row["optimizationSeconds"] for row in rows.values()
                ),
                "optimization_vs_hybrid_plain": (
                    math.fsum(row["optimizationSeconds"] for row in rows.values())
                    / math.fsum(
                        row["optimizationSeconds"] for row in modes["plain"].values()
                    )
                ),
                "completion": {
                    scene: {
                        "completed_iterations": rows[scene]["completedIterations"],
                        "termination_reason": rows[scene]["terminationReason"],
                    }
                    for scene in scenes
                },
                "rows": {
                    scene: {
                        "sse": rows[scene]["qualityMetrics"]["sumSquaredError"],
                        "sse_vs_hybrid_plain": (
                            rows[scene]["qualityMetrics"]["sumSquaredError"]
                            / modes["plain"][scene]["qualityMetrics"]["sumSquaredError"]
                        ),
                        "sse_vs_base_i30": (
                            rows[scene]["qualityMetrics"]["sumSquaredError"]
                            / base[scene]["qualityMetrics"]["sumSquaredError"]
                        ),
                    }
                    for scene in scenes
                },
            }
            for mode, rows in modes.items()
        },
        "base_i30_optimization_seconds": math.fsum(
            row["optimizationSeconds"] for row in base.values()
        ),
    }


def write_report(path, summary):
    labels = {
        "plain": "Base backbone",
        "c1": "Base backbone + C1",
        "c5": "Base backbone + C5",
        "c1_c5": "Base backbone + C1+C5",
    }
    with path.open("w", encoding="utf-8") as output:
        output.write("# Base-Backbone C1/C5 Sentinel Factorial\n\n")
        output.write(
            "All variants use one global K24/I30 backbone: local Nesterov, "
            "persistent DRS trust, curvature 0.4 recovery/decay, camera metric "
            "75, no proposal damping, and shared-only camera proximal terms. "
            "Only C1 and delayed C5 vary.\n\n"
        )
        for family in ("1dsfm", "bal"):
            output.write(f"## {'1DSfM' if family == '1dsfm' else 'BAL'}\n\n")
            output.write(
                "| Variant | SSE/hybrid plain | W/L | Worst | SSE/base I30 | "
                "Optimization s | Time/plain | Completion |\n"
            )
            output.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
            for mode in MODES:
                row = summary[family]["modes"][mode]
                plain = row["versus_hybrid_plain"]
                base = row["versus_base_i30"]
                output.write(
                    f"| {labels[mode]} | {plain['geometric_sse']:.6f} | "
                    f"{plain['wins']}/{plain['losses']} | {plain['worst']:.6f} | "
                    f"{base['geometric_sse']:.6f} | "
                    f"{row['optimization_seconds']:.3f} | "
                    f"{row['optimization_vs_hybrid_plain']:.3f} | "
                    + ", ".join(
                        f"{scene}:{value['completed_iterations']}"
                        + ("*" if value["termination_reason"] != "iteration_limit" else "")
                        for scene, value in row["completion"].items()
                    )
                    + " |\n"
                )
            output.write("\n")
        output.write("`*` marks recovery exhaustion before I30.\n\n")
        output.write("## Gate\n\n")
        output.write(
            "Promote a hybrid only if one unchanged C1/C5 combination improves "
            "the base-backbone plain arm on both families without a material "
            "worst-scene regression. No scene routing is permitted.\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    arguments = parser.parse_args()
    summary = {
        family: analyze_family(arguments.root, family, arguments.iterations)
        for family in ("1dsfm", "bal")
    }
    (arguments.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = arguments.root / "report.md"
    write_report(report, summary)
    print(report)


if __name__ == "__main__":
    main()
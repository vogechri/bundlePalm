#!/usr/bin/env python3
"""Analyze behavior-neutral late DRS/Schur tangent alignment."""

import argparse
import json
import math
import re
import statistics
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parent.parent
EXPECTED = (
    "montreal_notre_dame", "piazza_del_popolo", "roman_forum", "trafalgar",
    "yorkminster", "bal52", "bal3068",
)
CHECKPOINTS = (90, 120, 160, 200)
BEHAVIOR_FIELDS = (
    "sumSquaredError", "refinedCandidateSumSquaredError", "rejected",
    "rejections", "proximalOracleCalls",
)


def scene_key(row):
    dataset = Path(row["dataset"])
    match = re.search(r"problem-(\d+)-", dataset.name)
    return f"bal{match.group(1)}" if match else dataset.parent.name


def load_rows(directory):
    rows = {}
    for path in directory.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[scene_key(row)] = row
    return rows


def reference_rows():
    rows = {}
    root = WORKSPACE / "benchmark_results/k24_i200_terminal_correction_breadth"
    for family in ("1dsfm", "bal"):
        rows.update(load_rows(root / family / "i200"))
    return rows


def exact_behavior(reference, diagnostic, scene):
    if len(reference["trajectory"]) != len(diagnostic["trajectory"]):
        raise ValueError(f"trajectory length mismatch for {scene}")
    for left, right in zip(reference["trajectory"], diagnostic["trajectory"]):
        for field in BEHAVIOR_FIELDS:
            if left[field] != right[field]:
                raise ValueError(f"behavior mismatch {scene}/{left['iteration']}/{field}")
    if reference["qualityMetrics"]["sumSquaredError"] != diagnostic["qualityMetrics"]["sumSquaredError"]:
        raise ValueError(f"endpoint mismatch for {scene}")
    reference_state = np.load(reference["stateFile"])
    diagnostic_state = np.load(diagnostic["stateFile"])
    if not np.array_equal(reference_state["cameras"], diagnostic_state["cameras"]):
        raise ValueError(f"camera-state mismatch for {scene}")
    if not np.array_equal(reference_state["points"], diagnostic_state["points"]):
        raise ValueError(f"point-state mismatch for {scene}")


def ratio(candidate, reference):
    return candidate / reference if reference > 0.0 else math.nan


def signed_ratio(candidate, reference):
    return candidate / reference if reference != 0.0 else math.nan


def camera_model_reduction(model):
    return model["dampedPredictedReduction"] - model["landmarkModelReduction"]


def extract_checkpoint(scene, checkpoint, diagnostic):
    row = next(
        entry for entry in diagnostic["trajectory"]
        if entry["iteration"] == checkpoint - 1
    )
    data = row["schurAlignmentDiagnostics"]
    if data is None:
        raise ValueError(f"missing alignment {scene}/I{checkpoint}")
    schur = data["schur"]
    if schur["linearTermination"] != 0 or schur["relativeResidual"] >= 1e-6:
        raise ValueError(f"nonconverged alignment {scene}/I{checkpoint}")
    shared = data["sharedCamerasDiagonalWeighted"]["global"]
    all_cameras = data["allCamerasDiagonalWeighted"]["global"]
    consensus_model = data["consensusModel"]
    shared_consensus_model = data["sharedConsensusModel"]
    unique_consensus_model = data["uniqueConsensusModel"]
    shared_schur_model = data["sharedSchurModel"]
    landmark_reduction = schur["landmarkModelReduction"]
    schur_camera_reduction = schur["dampedPredictedReduction"] - landmark_reduction
    drs_camera_reduction = camera_model_reduction(consensus_model)
    shared_drs_camera_reduction = camera_model_reduction(
        shared_consensus_model
    )
    unique_drs_camera_reduction = camera_model_reduction(
        unique_consensus_model
    )
    shared_schur_camera_reduction = camera_model_reduction(
        shared_schur_model
    )
    return {
        "scene": scene,
        "checkpoint": checkpoint,
        "shared_weighted_cosine": shared["cosine"],
        "shared_drs_over_schur_norm": ratio(
            shared["candidateNorm"], shared["referenceNorm"]
        ),
        "all_weighted_cosine": all_cameras["cosine"],
        "all_drs_over_schur_norm": ratio(
            all_cameras["candidateNorm"], all_cameras["referenceNorm"]
        ),
        "shared_unweighted_cosine": data["sharedCameras"]["global"]["cosine"],
        "translation_cosine": data["sharedCamerasDiagonalWeighted"]["translation"]["cosine"],
        "rotation_cosine": data["sharedCamerasDiagonalWeighted"]["rotation"]["cosine"],
        "intrinsics_cosine": data["sharedCamerasDiagonalWeighted"]["intrinsics"]["cosine"],
        "gradient_action_ratio": signed_ratio(
            data["consensusGradientAction"], data["schurGradientAction"]
        ),
        "drs_camera_model_reduction": drs_camera_reduction,
        "schur_camera_model_reduction": schur_camera_reduction,
        "camera_model_reduction_ratio": ratio(
            drs_camera_reduction, schur_camera_reduction
        ),
        "shared_drs_camera_model_reduction": shared_drs_camera_reduction,
        "shared_schur_camera_model_reduction": shared_schur_camera_reduction,
        "shared_camera_model_reduction_ratio": ratio(
            shared_drs_camera_reduction, shared_schur_camera_reduction
        ),
        "unique_drs_camera_model_reduction": unique_drs_camera_reduction,
        "linear_iterations": schur["linearIterations"],
        "relative_residual": schur["relativeResidual"],
        "linear_seconds": schur["totalLinearSystemSeconds"],
    }


def analyze(root):
    diagnostics = load_rows(root / "diagnostic")
    references = reference_rows()
    if set(diagnostics) != set(EXPECTED):
        raise ValueError(f"diagnostic coverage mismatch: {sorted(diagnostics)}")
    records = []
    for scene in EXPECTED:
        reference = references[scene]
        diagnostic = diagnostics[scene]
        exact_behavior(reference, diagnostic, scene)
        available = [
            checkpoint for checkpoint in CHECKPOINTS
            if checkpoint <= diagnostic["completedIterations"]
        ]
        expected_present = [
            entry["iteration"] + 1 for entry in diagnostic["trajectory"]
            if entry["schurAlignmentDiagnostics"] is not None
        ]
        if expected_present != available:
            raise ValueError(
                f"alignment checkpoint mismatch {scene}: {expected_present} != {available}"
            )
        records.extend(
            extract_checkpoint(scene, checkpoint, diagnostic)
            for checkpoint in available
        )
    by_checkpoint = {}
    for checkpoint in CHECKPOINTS:
        rows = [row for row in records if row["checkpoint"] == checkpoint]
        if not rows:
            continue
        by_checkpoint[str(checkpoint)] = {
            "count": len(rows),
            "median_shared_weighted_cosine": statistics.median(
                row["shared_weighted_cosine"] for row in rows
            ),
            "minimum_shared_weighted_cosine": min(
                row["shared_weighted_cosine"] for row in rows
            ),
            "median_shared_drs_over_schur_norm": statistics.median(
                row["shared_drs_over_schur_norm"] for row in rows
            ),
            "median_shared_camera_model_reduction_ratio": statistics.median(
                row["shared_camera_model_reduction_ratio"] for row in rows
            ),
            "nonpositive_shared_drs_camera_models": sum(
                row["shared_drs_camera_model_reduction"] <= 0.0 for row in rows
            ),
            "nonpositive_unique_drs_camera_models": sum(
                row["unique_drs_camera_model_reduction"] <= 0.0 for row in rows
            ),
        }
    return {
        "status": "passed",
        "checkpoint_summaries": by_checkpoint,
        "records": records,
    }


def write_report(path, summary):
    with path.open("w", encoding="utf-8") as output:
        output.write("# K24 Late DRS/Schur Direction Alignment\n\n")
        output.write("Schur is the reference tangent; DRS is the candidate tangent.\n\n")
        output.write("| I | N | Median shared weighted cosine | Minimum cosine | Median DRS/Schur norm | Median shared-model ratio | Nonpositive shared/unique models |\n")
        output.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for checkpoint, row in summary["checkpoint_summaries"].items():
            output.write(
                f"| {checkpoint} | {row['count']} | "
                f"{row['median_shared_weighted_cosine']:.6f} | "
                f"{row['minimum_shared_weighted_cosine']:.6f} | "
                f"{row['median_shared_drs_over_schur_norm']:.6f} | "
                f"{row['median_shared_camera_model_reduction_ratio']:.6f} | "
                f"{row['nonpositive_shared_drs_camera_models']}/"
                f"{row['nonpositive_unique_drs_camera_models']} |\n"
            )
        output.write("\n| Scene | I | Shared weighted cosine | DRS/Schur norm | Shared-model ratio | Shared/unique reduction | T/R/I cosine | PCG iters |\n")
        output.write("|---|---:|---:|---:|---:|---:|---|---:|\n")
        for row in summary["records"]:
            output.write(
                f"| {row['scene']} | {row['checkpoint']} | "
                f"{row['shared_weighted_cosine']:.6f} | "
                f"{row['shared_drs_over_schur_norm']:.6f} | "
                f"{row['shared_camera_model_reduction_ratio']:.6f} | "
                f"{row['shared_drs_camera_model_reduction']:.3g}/"
                f"{row['unique_drs_camera_model_reduction']:.3g} | "
                f"{row['translation_cosine']:.3f}/"
                f"{row['rotation_cosine']:.3f}/"
                f"{row['intrinsics_cosine']:.3f} | "
                f"{row['linear_iterations']} |\n"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    summary = analyze(arguments.root)
    (arguments.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(arguments.root / "report.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Analyze applied two-direction Krylov proposal breadth."""

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from analyze_k24_one_step_schur_proposal_breadth import (
    EXPECTED,
    ceres_rows,
    load_rows,
    load_status,
)


DAMPING = 0.00146484375
BEHAVIOR_FIELDS = (
    "sumSquaredError",
    "refinedCandidateSumSquaredError",
    "rejected",
    "rejections",
    "proximalOracleCalls",
)


def geometric_mean(values):
    values = tuple(values)
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def compare(candidate, reference):
    ratios = {
        scene: candidate[scene] / reference[scene]
        for scene in sorted(candidate)
    }
    return {
        "geometric": geometric_mean(ratios.values()),
        "summed": math.fsum(candidate.values()) / math.fsum(reference.values()),
        "wins": sum(value < 1.0 - 1e-9 for value in ratios.values()),
        "ties": sum(abs(value - 1.0) <= 1e-9 for value in ratios.values()),
        "losses": sum(value > 1.0 + 1e-9 for value in ratios.values()),
        "worst": max(ratios.values()),
        "ratios": ratios,
    }


def camera_model_reduction(model):
    return model["dampedPredictedReduction"] - model["landmarkModelReduction"]


def merge_rows(*directories):
    merged = {}
    for directory in directories:
        rows = load_rows(directory)
        overlap = set(merged).intersection(rows)
        if overlap:
            raise ValueError(f"duplicate scenes while merging: {sorted(overlap)}")
        merged.update(rows)
    return merged


def validate_candidate(scene, row, control):
    expected = {
        "clusters": 24,
        "iterations": 120,
        "completedIterations": 120,
        "schurAlignmentDiagnosticIterations": [60],
        "oneStepSchurResidualProposalIterations": [],
        "oneStepSchurResidualProposalRebaseTrustState": True,
        "schurAlignmentCameraDamping": DAMPING,
        "schurAlignmentLandmarkDamping": DAMPING,
        "sharedSchurLandmarkRefinementSteps": 3,
        "schurKrylovLandmarkResponseOracle": True,
        "applySchurKrylovLandmarkResponse": True,
        "sharedSchurMinimumRelativeDecrease": 1e-3,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(
                f"configuration mismatch {scene}: {field}={row.get(field)!r}, "
                f"expected={value!r}"
            )
    for iteration, (candidate, reference) in enumerate(
        zip(row["trajectory"][:59], control["trajectory"][:59]), 1
    ):
        for field in BEHAVIOR_FIELDS:
            if candidate[field] != reference[field]:
                raise ValueError(
                    f"prefix mismatch {scene}/I{iteration}/{field}"
                )
    diagnostics = row["trajectory"][59]["schurAlignmentDiagnostics"]
    schur = diagnostics["schur"]
    if schur["linearTermination"] != 0 or schur["relativeResidual"] >= 1e-6:
        raise ValueError(f"nonconverged Schur reference for {scene}")
    oracle = diagnostics["twoDirectionSchurKrylovLandmarkResponseOracle"]
    if oracle is None or len(oracle["attempts"]) != 8:
        raise ValueError(f"Krylov proposal telemetry mismatch for {scene}")
    if oracle["minimumRelativeDecrease"] != 1e-3:
        raise ValueError(f"Krylov proposal floor mismatch for {scene}")
    if oracle["workerStateRoundtripRelativeError"] > 1e-9:
        raise ValueError(f"worker state roundtrip mismatch for {scene}")
    if bool(oracle["applied"]) != bool(oracle["selected"]):
        raise ValueError(f"Krylov apply/selection mismatch for {scene}")
    return diagnostics, oracle


def validate_lightweight(candidate_root, lightweight_root):
    summary = {}
    for family, expected_scenes in (
        ("1dsfm", None),
        ("bal", {"bal52", "bal3068"}),
    ):
        candidates = load_rows(candidate_root / family)
        lightweight = load_rows(lightweight_root / family)
        statuses = load_status(lightweight_root / family)
        if expected_scenes is None:
            expected_scenes = set(candidates)
        if set(lightweight) != expected_scenes or set(statuses) != expected_scenes:
            raise ValueError(f"lightweight coverage mismatch for {family}")
        for scene in sorted(expected_scenes):
            left = lightweight[scene]
            right = candidates[scene]
            status = statuses[scene]
            if status["status"] != "completed" or int(status["exit_code"]) != 0:
                raise ValueError(f"failed lightweight status for {scene}")
            if left.get("schurResidualProposalDirection") != "krylov2":
                raise ValueError(f"lightweight direction mismatch for {scene}")
            diagnostics = left["trajectory"][59]["schurAlignmentDiagnostics"]
            if not diagnostics["referenceSolveSkipped"]:
                raise ValueError(f"lightweight reference solve ran for {scene}")
            if len(left["trajectory"]) != len(right["trajectory"]):
                raise ValueError(f"lightweight trajectory length mismatch {scene}")
            for iteration, (candidate, reference) in enumerate(
                zip(left["trajectory"], right["trajectory"]), 1
            ):
                for field in BEHAVIOR_FIELDS:
                    if candidate[field] != reference[field]:
                        raise ValueError(
                            f"lightweight mismatch {scene}/I{iteration}/{field}"
                        )
            left_state = np.load(left["stateFile"])
            right_state = np.load(right["stateFile"])
            if not np.array_equal(left_state["cameras"], right_state["cameras"]):
                raise ValueError(f"lightweight camera-state mismatch for {scene}")
            if not np.array_equal(left_state["points"], right_state["points"]):
                raise ValueError(f"lightweight point-state mismatch for {scene}")
        summary[family] = {
            "count": len(expected_scenes),
            "trajectory_and_state_exact": True,
            "reference_solve_skipped": True,
        }
    return summary


def analyze(candidate_root, lightweight_root, quarter_roots, base_root, control_root):
    ceres = ceres_rows()
    quarter_1dsfm = merge_rows(
        *(root / "1dsfm" for root in quarter_roots)
    )
    quarter = {
        "1dsfm": quarter_1dsfm,
        "bal": load_rows(base_root / "bal"),
    }
    summary = {"status": "passed", "families": {}, "scenes": {}}
    for family in ("1dsfm", "bal"):
        candidates = load_rows(candidate_root / family)
        controls = load_rows(control_root / family / "i200")
        base_rows = load_rows(base_root / family)
        statuses = load_status(candidate_root / family)
        expected_count = EXPECTED[family]
        if len(candidates) != expected_count:
            raise ValueError(
                f"candidate coverage mismatch {family}: "
                f"{len(candidates)}/{expected_count}"
            )
        if not (
            set(candidates)
            == set(controls)
            == set(statuses)
            == set(quarter[family])
            == set(base_rows)
        ):
            raise ValueError(f"scene/status mismatch for {family}")

        delivered = {}
        quarter_endpoints = {}
        base_endpoints = {}
        control_endpoints = {}
        references = {}
        selected = 0
        direction_rows = []
        for scene in sorted(candidates):
            status = statuses[scene]
            if status["status"] != "completed" or int(status["exit_code"]) != 0:
                raise ValueError(f"failed status for {scene}")
            row = candidates[scene]
            diagnostics, oracle = validate_candidate(scene, row, controls[scene])
            delivered[scene] = float(row["qualityMetrics"]["sumSquaredError"])
            quarter_endpoints[scene] = float(
                quarter[family][scene]["qualityMetrics"]["sumSquaredError"]
            )
            base_endpoints[scene] = float(
                base_rows[scene]["qualityMetrics"]["sumSquaredError"]
            )
            control_endpoints[scene] = min(
                float(entry["sumSquaredError"])
                for entry in controls[scene]["trajectory"][:120]
            )
            references[scene] = ceres[scene]
            selected += bool(oracle["selected"])
            one_action = diagnostics["oneStepSchurResidualOracle"]
            krylov = diagnostics["twoDirectionSchurKrylovOracle"]
            one_alignment = one_action["sharedCamerasDiagonalWeighted"]["global"]
            krylov_alignment = krylov["sharedCamerasDiagonalWeighted"]["global"]
            schur_model_gain = camera_model_reduction(
                diagnostics["sharedSchurModel"]
            )
            direction = {
                "one_action_cosine": one_alignment["cosine"],
                "one_action_over_schur_norm": (
                    one_alignment["candidateNorm"]
                    / one_alignment["referenceNorm"]
                ),
                "one_action_over_schur_model_gain": (
                    camera_model_reduction(one_action["model"])
                    / schur_model_gain
                ),
                "krylov_cosine": krylov_alignment["cosine"],
                "krylov_over_schur_norm": (
                    krylov_alignment["candidateNorm"]
                    / krylov_alignment["referenceNorm"]
                ),
                "krylov_over_schur_model_gain": (
                    camera_model_reduction(krylov["model"])
                    / schur_model_gain
                ),
            }
            if oracle["selected"]:
                direction_rows.append(direction)
            summary["scenes"][scene] = {
                "family": family,
                "selected": bool(oracle["selected"]),
                "selected_scale": float(oracle["selectedScale"]),
                "delivered_over_quarter": delivered[scene] / quarter_endpoints[scene],
                "delivered_over_base": delivered[scene] / base_endpoints[scene],
                "delivered_over_control": delivered[scene] / control_endpoints[scene],
                "delivered_over_ceres": delivered[scene] / references[scene],
                "roundtrip_relative_error": oracle[
                    "workerStateRoundtripRelativeError"
                ],
                "schur_relative_residual": diagnostics["schur"][
                    "relativeResidual"
                ],
                **direction,
            }
        summary["families"][family] = {
            "count": len(candidates),
            "selected": selected,
            "over_quarter": compare(delivered, quarter_endpoints),
            "over_base": compare(delivered, base_endpoints),
            "over_control": compare(delivered, control_endpoints),
            "over_ceres": compare(delivered, references),
            "maximum_coordinator_rss_gib": max(
                int(row["coordinator_max_rss_kb"]) for row in statuses.values()
            ) / 1048576.0,
            "maximum_worker_rss_gib": max(
                int(row["worker_max_rss_kb"]) for row in statuses.values()
            ) / 1048576.0,
            "median_one_action_cosine": (
                statistics.median(
                    row["one_action_cosine"] for row in direction_rows
                ) if direction_rows else None
            ),
            "median_one_action_over_schur_norm": (
                statistics.median(
                    row["one_action_over_schur_norm"]
                    for row in direction_rows
                ) if direction_rows else None
            ),
            "median_one_action_over_schur_model_gain": (
                statistics.median(
                    row["one_action_over_schur_model_gain"]
                    for row in direction_rows
                ) if direction_rows else None
            ),
            "median_krylov_cosine": (
                statistics.median(
                    row["krylov_cosine"] for row in direction_rows
                ) if direction_rows else None
            ),
            "median_krylov_over_schur_norm": (
                statistics.median(
                    row["krylov_over_schur_norm"] for row in direction_rows
                ) if direction_rows else None
            ),
            "median_krylov_over_schur_model_gain": (
                statistics.median(
                    row["krylov_over_schur_model_gain"]
                    for row in direction_rows
                ) if direction_rows else None
            ),
        }
    summary["lightweight_equivalence"] = validate_lightweight(
        candidate_root, lightweight_root
    )
    return summary


def write_report(path, summary):
    with path.open("w", encoding="utf-8") as output:
        output.write("# K24 I60 Quarter-Damped Two-Direction Krylov Proposal\n\n")
        output.write(
            "Two preconditioned conjugate directions replace the promoted "
            "single block-Jacobi camera action. Eight scales, three fixed-camera "
            "landmark steps, the `1e-3` floor, atomic commit, canonical restart, "
            "and selected-only trust rebase remain unchanged.\n\n"
        )
        output.write(
            "The proposal-only path skips the converged Schur reference and "
            "is trajectory- and state-exact to the diagnostic path on all 15 "
            "1DSfM scenes plus BAL52/3068.\n\n"
        )
        output.write(
            "| Family | Completed | Candidate/quarter | Candidate/base | "
            "Candidate/control | Candidate/Ceres | W/T/L quarter | Selected | "
            "Max RSS GiB C/W |\n"
        )
        output.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for family in ("1dsfm", "bal"):
            row = summary["families"][family]
            quarter = row["over_quarter"]
            output.write(
                f"| {family} | {row['count']}/{EXPECTED[family]} | "
                f"{quarter['geometric']:.9f} | "
                f"{row['over_base']['geometric']:.9f} | "
                f"{row['over_control']['geometric']:.9f} | "
                f"{row['over_ceres']['geometric']:.9f} | "
                f"{quarter['wins']}/{quarter['ties']}/{quarter['losses']} | "
                f"{row['selected']}/{row['count']} | "
                f"{row['maximum_coordinator_rss_gib']:.3f}/"
                f"{row['maximum_worker_rss_gib']:.3f} |\n"
            )
        output.write(
            "\n| Family | Median one-action cosine/norm/gain | "
            "Median Krylov cosine/norm/gain |\n"
        )
        output.write("|---|---:|---:|\n")
        for family in ("1dsfm", "bal"):
            row = summary["families"][family]
            if row["median_one_action_cosine"] is None:
                output.write(f"| {family} | n/a | n/a |\n")
                continue
            output.write(
                f"| {family} | {row['median_one_action_cosine']:.6f}/"
                f"{row['median_one_action_over_schur_norm']:.6f}/"
                f"{row['median_one_action_over_schur_model_gain']:.6f} | "
                f"{row['median_krylov_cosine']:.6f}/"
                f"{row['median_krylov_over_schur_norm']:.6f}/"
                f"{row['median_krylov_over_schur_model_gain']:.6f} |\n"
            )
        output.write("\n| Scene | Candidate/quarter | Candidate/control | Candidate/Ceres | Selected scale |\n")
        output.write("|---|---:|---:|---:|---:|\n")
        for scene, row in sorted(summary["scenes"].items()):
            selected_scale = (
                f"{row['selected_scale']:.6g}" if row["selected"] else "declined"
            )
            output.write(
                f"| {scene} | {row['delivered_over_quarter']:.9f} | "
                f"{row['delivered_over_control']:.9f} | "
                f"{row['delivered_over_ceres']:.9f} | {selected_scale} |\n"
            )
        output.write("\nGate status: **passed**.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--lightweight-root", type=Path, required=True)
    parser.add_argument(
        "--quarter-root", type=Path, action="append", required=True
    )
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    arguments = parser.parse_args()
    summary = analyze(
        arguments.candidate_root,
        arguments.lightweight_root,
        arguments.quarter_root,
        arguments.base_root,
        arguments.control_root,
    )
    (arguments.candidate_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(arguments.candidate_root / "report.md", summary)
    print(json.dumps(summary["families"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

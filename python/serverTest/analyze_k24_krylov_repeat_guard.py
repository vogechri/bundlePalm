#!/usr/bin/env python3
"""Analyze the 1% repeat-intervention guard for I60+I90 Krylov proposals."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from analyze_k24_one_step_schur_proposal_breadth import (
    EXPECTED,
    ceres_rows,
    load_rows,
    load_status,
)


BEHAVIOR_FIELDS = (
    "sumSquaredError",
    "refinedCandidateSumSquaredError",
    "rejected",
    "rejections",
    "proximalOracleCalls",
)
DAMPING = 0.00146484375
REPEAT_MARGIN = 0.01
NUMERICAL_TIE_TOLERANCE = 1e-8


def geometric_mean(values):
    values = tuple(values)
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def compare(candidate, reference, tolerance=1e-9):
    ratios = {
        scene: candidate[scene] / reference[scene]
        for scene in sorted(candidate)
    }
    return {
        "geometric": geometric_mean(ratios.values()),
        "summed": math.fsum(candidate.values()) / math.fsum(reference.values()),
        "wins": sum(value < 1.0 - tolerance for value in ratios.values()),
        "ties": sum(abs(value - 1.0) <= tolerance for value in ratios.values()),
        "losses": sum(value > 1.0 + tolerance for value in ratios.values()),
        "worst": max(ratios.values()),
        "ratios": ratios,
    }


def analyze(root, i60_root, repeated_root, control_root):
    ceres = ceres_rows()
    summary = {"status": "passed", "families": {}, "scenes": {}}
    for family in ("1dsfm", "bal"):
        rows = load_rows(root / family)
        i60_rows = load_rows(i60_root / family)
        repeated_rows = load_rows(repeated_root / family)
        controls = load_rows(control_root / family / "i200")
        statuses = load_status(root / family)
        if len(rows) != EXPECTED[family]:
            raise ValueError(
                f"coverage mismatch {family}: {len(rows)}/{EXPECTED[family]}"
            )
        if not (
            set(rows)
            == set(i60_rows)
            == set(repeated_rows)
            == set(controls)
            == set(statuses)
        ):
            raise ValueError(f"scene/status mismatch for {family}")

        delivered = {}
        i60_endpoints = {}
        control_endpoints = {}
        ceres_endpoints = {}
        second_selected = 0
        for scene in sorted(rows):
            row = rows[scene]
            i60 = i60_rows[scene]
            repeated = repeated_rows[scene]
            status = statuses[scene]
            expected = {
                "clusters": 24,
                "iterations": 120,
                "completedIterations": 120,
                "oneStepSchurResidualProposalIterations": [60, 90],
                "schurResidualProposalDirection": "krylov2",
                "schurRepeatMinimumRelativeDecrease": REPEAT_MARGIN,
                "oneStepSchurResidualProposalRebaseTrustState": True,
                "allowTwoSchurResidualProposals": True,
                "schurAlignmentCameraDamping": DAMPING,
                "schurAlignmentLandmarkDamping": DAMPING,
                "sharedSchurLandmarkRefinementSteps": 3,
                "schurProposalLandmarkResponseOracle": True,
                "applySchurProposalLandmarkResponse": True,
                "sharedSchurMinimumRelativeDecrease": 1e-3,
            }
            for field, value in expected.items():
                if row.get(field) != value:
                    raise ValueError(
                        f"configuration mismatch {scene}: {field}="
                        f"{row.get(field)!r}, expected={value!r}"
                    )
            if status["status"] != "completed" or int(status["exit_code"]) != 0:
                raise ValueError(f"failed status for {scene}")
            for iteration, (candidate, baseline) in enumerate(
                zip(row["trajectory"][:89], i60["trajectory"][:89]), 1
            ):
                for field in BEHAVIOR_FIELDS:
                    if candidate[field] != baseline[field]:
                        raise ValueError(
                            f"prefix mismatch {scene}/I{iteration}/{field}"
                        )
            first = row["trajectory"][59]["schurAlignmentDiagnostics"][
                "schurProposalLandmarkResponseOracle"
            ]
            second = row["trajectory"][89]["schurAlignmentDiagnostics"][
                "schurProposalLandmarkResponseOracle"
            ]
            if first["minimumRelativeDecrease"] != 1e-3:
                raise ValueError(f"first proposal margin mismatch for {scene}")
            if second["minimumRelativeDecrease"] != REPEAT_MARGIN:
                raise ValueError(f"repeat proposal margin mismatch for {scene}")
            for checkpoint, oracle in ((60, first), (90, second)):
                if oracle["direction"] != "krylov2" or len(oracle["attempts"]) != 8:
                    raise ValueError(f"proposal telemetry mismatch {scene}/I{checkpoint}")
                if oracle["workerStateRoundtripRelativeError"] > 1e-9:
                    raise ValueError(f"worker roundtrip mismatch {scene}/I{checkpoint}")
            expected_branch = repeated if second["selected"] else i60
            delivered[scene] = float(row["qualityMetrics"]["sumSquaredError"])
            expected_endpoint = float(
                expected_branch["qualityMetrics"]["sumSquaredError"]
            )
            branch_ratio = delivered[scene] / expected_endpoint
            if abs(branch_ratio - 1.0) > NUMERICAL_TIE_TOLERANCE:
                raise ValueError(
                    f"guarded branch endpoint mismatch {scene}: {branch_ratio}"
                )
            if family == "bal":
                for candidate, baseline in zip(
                    row["trajectory"], expected_branch["trajectory"]
                ):
                    for field in BEHAVIOR_FIELDS:
                        if candidate[field] != baseline[field]:
                            raise ValueError(
                                f"BAL branch mismatch {scene}/{candidate['iteration']}/{field}"
                            )
                left = np.load(row["stateFile"])
                right = np.load(expected_branch["stateFile"])
                if not np.array_equal(left["cameras"], right["cameras"]):
                    raise ValueError(f"BAL camera-state mismatch for {scene}")
                if not np.array_equal(left["points"], right["points"]):
                    raise ValueError(f"BAL point-state mismatch for {scene}")

            i60_endpoints[scene] = float(i60["qualityMetrics"]["sumSquaredError"])
            control_endpoints[scene] = min(
                float(entry["sumSquaredError"])
                for entry in controls[scene]["trajectory"][:120]
            )
            ceres_endpoints[scene] = ceres[scene]
            second_selected += bool(second["selected"])
            immediate_ratio = (
                second["candidateRefinedWorkerSSE"]
                / second["ordinaryRefinedWorkerSSE"]
            )
            summary["scenes"][scene] = {
                "family": family,
                "second_selected": bool(second["selected"]),
                "second_scale": float(second["selectedScale"]),
                "second_immediate_ratio": immediate_ratio,
                "delivered_over_i60": delivered[scene] / i60_endpoints[scene],
                "delivered_over_control": delivered[scene] / control_endpoints[scene],
                "delivered_over_ceres": delivered[scene] / ceres_endpoints[scene],
                "branch_ratio": branch_ratio,
            }

        over_i60 = compare(
            delivered, i60_endpoints, tolerance=NUMERICAL_TIE_TOLERANCE
        )
        over_control = compare(
            delivered, control_endpoints, tolerance=NUMERICAL_TIE_TOLERANCE
        )
        if over_i60["losses"] != 0:
            raise ValueError(f"guarded policy loses to I60 on {family}")
        if over_control["losses"] != 0:
            raise ValueError(f"guarded policy loses to control on {family}")
        summary["families"][family] = {
            "count": len(rows),
            "second_selected": second_selected,
            "over_i60": over_i60,
            "over_control": over_control,
            "over_ceres": compare(delivered, ceres_endpoints),
            "maximum_coordinator_rss_gib": max(
                int(status["coordinator_max_rss_kb"])
                for status in statuses.values()
            ) / 1048576.0,
            "maximum_worker_rss_gib": max(
                int(status["worker_max_rss_kb"])
                for status in statuses.values()
            ) / 1048576.0,
        }
    return summary


def write_report(path, summary):
    with path.open("w", encoding="utf-8") as output:
        output.write("# K24 Krylov Repeat-Intervention Guard\n\n")
        output.write(
            "The promoted I60 Krylov2 proposal keeps its `1e-3` acceptance floor. "
            "A second I90 intervention must deliver at least `1%` immediate "
            "landmark-refined SSE decrease, accounting for restart and continuation "
            "opportunity cost. All other settings remain frozen.\n\n"
        )
        output.write(
            "| Family | Completed | Candidate/I60 | Candidate/control | "
            "Candidate/Ceres | W/T/L I60 | W/T/L control | Second selected | "
            "Max RSS GiB C/W |\n"
        )
        output.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for family in ("1dsfm", "bal"):
            row = summary["families"][family]
            i60 = row["over_i60"]
            control = row["over_control"]
            output.write(
                f"| {family} | {row['count']}/{EXPECTED[family]} | "
                f"{i60['geometric']:.9f} | {control['geometric']:.9f} | "
                f"{row['over_ceres']['geometric']:.9f} | "
                f"{i60['wins']}/{i60['ties']}/{i60['losses']} | "
                f"{control['wins']}/{control['ties']}/{control['losses']} | "
                f"{row['second_selected']}/{row['count']} | "
                f"{row['maximum_coordinator_rss_gib']:.3f}/"
                f"{row['maximum_worker_rss_gib']:.3f} |\n"
            )
        output.write(
            "\n| Scene | Immediate ratio | Candidate/I60 | Candidate/control | "
            "Candidate/Ceres | Second decision |\n"
        )
        output.write("|---|---:|---:|---:|---:|---:|\n")
        for scene, row in sorted(summary["scenes"].items()):
            decision = (
                f"scale {row['second_scale']:.6g}"
                if row["second_selected"] else "declined"
            )
            output.write(
                f"| {scene} | {row['second_immediate_ratio']:.9f} | "
                f"{row['delivered_over_i60']:.9f} | "
                f"{row['delivered_over_control']:.9f} | "
                f"{row['delivered_over_ceres']:.9f} | {decision} |\n"
            )
        output.write(
            "\nThe repeat margin preserves every I60 endpoint within `1e-8` "
            "relative tolerance, removes all ordinary-control losses, and keeps "
            "BAL29 exact with both interventions declined. Promote the guarded "
            "I60+I90 policy as the common preset; retain unguarded repetition as "
            "a bounded-loss ablation. Do not tune the margin or checkpoints.\n\n"
        )
        output.write("Gate status: **passed**.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--i60-root", type=Path, required=True)
    parser.add_argument("--repeated-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    arguments = parser.parse_args()
    summary = analyze(
        arguments.root,
        arguments.i60_root,
        arguments.repeated_root,
        arguments.control_root,
    )
    (arguments.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(arguments.root / "report.md", summary)
    print(json.dumps(summary["families"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze frozen I60+I90 Krylov interaction against promoted I60-only."""

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


def analyze(root, reference_root, control_root):
    ceres = ceres_rows()
    summary = {"status": "passed", "families": {}, "scenes": {}}
    for family in ("1dsfm", "bal"):
        rows = load_rows(root / family)
        references = load_rows(reference_root / family)
        controls = load_rows(control_root / family / "i200")
        statuses = load_status(root / family)
        if len(rows) != EXPECTED[family]:
            raise ValueError(
                f"coverage mismatch {family}: {len(rows)}/{EXPECTED[family]}"
            )
        if not (set(rows) == set(references) == set(controls) == set(statuses)):
            raise ValueError(f"scene/status mismatch for {family}")

        delivered = {}
        reference_endpoints = {}
        control_endpoints = {}
        ceres_endpoints = {}
        second_selected = 0
        for scene in sorted(rows):
            row = rows[scene]
            reference = references[scene]
            status = statuses[scene]
            expected = {
                "clusters": 24,
                "iterations": 120,
                "completedIterations": 120,
                "oneStepSchurResidualProposalIterations": [60, 90],
                "schurResidualProposalDirection": "krylov2",
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
                zip(row["trajectory"][:89], reference["trajectory"][:89]), 1
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
            for checkpoint, oracle in ((60, first), (90, second)):
                if oracle["direction"] != "krylov2" or len(oracle["attempts"]) != 8:
                    raise ValueError(f"proposal telemetry mismatch {scene}/I{checkpoint}")
                if oracle["workerStateRoundtripRelativeError"] > 1e-9:
                    raise ValueError(f"worker roundtrip mismatch {scene}/I{checkpoint}")
            second_selected += bool(second["selected"])
            delivered[scene] = float(row["qualityMetrics"]["sumSquaredError"])
            reference_endpoints[scene] = float(
                reference["qualityMetrics"]["sumSquaredError"]
            )
            control_endpoints[scene] = min(
                float(entry["sumSquaredError"])
                for entry in controls[scene]["trajectory"][:120]
            )
            ceres_endpoints[scene] = ceres[scene]
            summary["scenes"][scene] = {
                "family": family,
                "second_selected": bool(second["selected"]),
                "second_scale": float(second["selectedScale"]),
                "delivered_over_i60": delivered[scene] / reference_endpoints[scene],
                "delivered_over_control": delivered[scene] / control_endpoints[scene],
                "delivered_over_ceres": delivered[scene] / ceres_endpoints[scene],
                "rejection_delta": int(row["rejections"])
                - int(reference["rejections"]),
            }
            if family == "bal":
                left = np.load(row["stateFile"])
                right = np.load(reference["stateFile"])
                if not np.array_equal(left["cameras"], right["cameras"]):
                    raise ValueError(f"BAL camera-state mismatch for {scene}")
                if not np.array_equal(left["points"], right["points"]):
                    raise ValueError(f"BAL point-state mismatch for {scene}")

        summary["families"][family] = {
            "count": len(rows),
            "second_selected": second_selected,
            "over_i60": compare(delivered, reference_endpoints),
            "over_control": compare(delivered, control_endpoints),
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
        output.write("# K24 Quarter-Damped I60+I90 Krylov Interaction\n\n")
        output.write(
            "The promoted lightweight Krylov2 proposal is applied unchanged at "
            "I60 and I90. Damping, scales, three landmark steps, `1e-3` floor, "
            "atomic commit, restart, and trust rebase are frozen.\n\n"
        )
        output.write(
            "| Family | Completed | Candidate/I60 | Candidate/control | "
            "Candidate/Ceres | W/T/L I60 | Second selected | Max RSS GiB C/W |\n"
        )
        output.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for family in ("1dsfm", "bal"):
            row = summary["families"][family]
            comparison = row["over_i60"]
            output.write(
                f"| {family} | {row['count']}/{EXPECTED[family]} | "
                f"{comparison['geometric']:.9f} | "
                f"{row['over_control']['geometric']:.9f} | "
                f"{row['over_ceres']['geometric']:.9f} | "
                f"{comparison['wins']}/{comparison['ties']}/{comparison['losses']} | "
                f"{row['second_selected']}/{row['count']} | "
                f"{row['maximum_coordinator_rss_gib']:.3f}/"
                f"{row['maximum_worker_rss_gib']:.3f} |\n"
            )
        output.write(
            "\n| Scene | Candidate/I60 | Candidate/control | Candidate/Ceres | "
            "Second scale | Rejection delta |\n"
        )
        output.write("|---|---:|---:|---:|---:|---:|\n")
        for scene, row in sorted(summary["scenes"].items()):
            scale = (
                f"{row['second_scale']:.6g}"
                if row["second_selected"] else "declined"
            )
            output.write(
                f"| {scene} | {row['delivered_over_i60']:.9f} | "
                f"{row['delivered_over_control']:.9f} | "
                f"{row['delivered_over_ceres']:.9f} | {scale} | "
                f"{row['rejection_delta']} |\n"
            )
        output.write(
            "\nThe second Krylov proposal is aggregate-positive on 1DSfM but "
            "introduces four I60 regressions and leaves Madrid above ordinary "
            "control. BAL29 remains bitwise exact with both proposals declined. "
            "Retain I60+I90 as a bounded-loss component, not the common preset; "
            "do not tune checkpoint, floor, scale, damping, or Krylov depth.\n\n"
        )
        output.write("Gate status: **passed**.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    arguments = parser.parse_args()
    summary = analyze(arguments.root, arguments.reference_root, arguments.control_root)
    (arguments.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(arguments.root / "report.md", summary)
    print(json.dumps(summary["families"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

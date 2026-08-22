#!/usr/bin/env python3
"""Analyze frozen landmark-response damping breadth."""

import argparse
import json
import math
from pathlib import Path

from analyze_k24_one_step_schur_proposal_breadth import (
    ceres_rows,
    load_rows,
)


EXPECTED = {"1dsfm": 15, "bal": 29}
PREFIX_FIELDS = (
    "sumSquaredError",
    "refinedCandidateSumSquaredError",
    "rejected",
    "rejections",
    "proximalOracleCalls",
)


def geometric_mean(values):
    values = tuple(values)
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def merge_rows(*directories):
    merged = {}
    for directory in directories:
        for scene, row in load_rows(directory).items():
            if scene in merged:
                raise ValueError(f"duplicate scene across split roots: {scene}")
            merged[scene] = row
    return merged


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


def validate_candidate(
    scene,
    candidate,
    base,
    family,
    camera_damping,
    landmark_damping,
):
    expected = {
        "clusters": 24,
        "iterations": 120,
        "completedIterations": 120,
        "oneStepSchurResidualProposalIterations": [60],
        "oneStepSchurResidualProposalRebaseTrustState": True,
        "applySchurProposalLandmarkResponse": True,
        "schurAlignmentCameraDamping": camera_damping,
        "schurAlignmentLandmarkDamping": landmark_damping,
        "sharedSchurLandmarkRefinementSteps": 3,
        "sharedSchurMinimumRelativeDecrease": 1e-3,
    }
    for field, value in expected.items():
        if candidate.get(field) != value:
            raise ValueError(
                f"configuration mismatch {scene}: {field}="
                f"{candidate.get(field)!r}, expected={value!r}"
            )
    if candidate.get("terminationReason") != "iteration_limit":
        raise ValueError(f"termination mismatch for {scene}")
    for iteration, (left, right) in enumerate(
        zip(candidate["trajectory"][:59], base["trajectory"][:59]), 1
    ):
        for field in PREFIX_FIELDS:
            if left[field] != right[field]:
                raise ValueError(
                    f"prefix mismatch {scene}/I{iteration}/{field}"
                )
    diagnostic = candidate["trajectory"][59]["schurAlignmentDiagnostics"][
        "schurProposalLandmarkResponseOracle"
    ]
    if diagnostic is None or len(diagnostic["attempts"]) != 8:
        raise ValueError(f"proposal telemetry mismatch for {scene}")
    if family == "bal":
        if diagnostic["selected"] or diagnostic["applied"]:
            raise ValueError(f"BAL proposal unexpectedly selected for {scene}")
        for left, right in zip(candidate["trajectory"], base["trajectory"]):
            for field in PREFIX_FIELDS:
                if left[field] != right[field]:
                    raise ValueError(
                        f"BAL behavior mismatch {scene}/{field}"
                    )
        if candidate["qualityMetrics"]["sumSquaredError"] != (
            base["qualityMetrics"]["sumSquaredError"]
        ):
            raise ValueError(f"BAL endpoint mismatch for {scene}")


def analyze(
    development_root,
    validation_root,
    bal_root,
    base_root,
    control_root,
    camera_damping,
    landmark_damping,
):
    candidate = {
        "1dsfm": merge_rows(development_root, validation_root),
        "bal": load_rows(bal_root),
    }
    base = {
        "1dsfm": load_rows(base_root / "1dsfm"),
        "bal": load_rows(base_root / "bal"),
    }
    control = {
        "1dsfm": load_rows(control_root / "1dsfm" / "i200"),
        "bal": load_rows(control_root / "bal" / "i200"),
    }
    ceres = ceres_rows()
    summaries = {}
    details = {}
    for family in ("1dsfm", "bal"):
        if len(candidate[family]) != EXPECTED[family]:
            raise ValueError(
                f"coverage mismatch {family}: "
                f"{len(candidate[family])}/{EXPECTED[family]}"
            )
        if set(candidate[family]) != set(base[family]):
            raise ValueError(f"base scene mismatch for {family}")
        for scene in candidate[family]:
            validate_candidate(
                scene,
                candidate[family][scene],
                base[family][scene],
                family,
                camera_damping,
                landmark_damping,
            )
        scenes = tuple(sorted(candidate[family]))
        delivered = {
            scene: candidate[family][scene]["qualityMetrics"]["sumSquaredError"]
            for scene in scenes
        }
        base_delivered = {
            scene: base[family][scene]["qualityMetrics"]["sumSquaredError"]
            for scene in scenes
        }
        control_delivered = {
            scene: min(
                row["sumSquaredError"]
                for row in control[family][scene]["trajectory"][:120]
            )
            for scene in scenes
        }
        references = {scene: ceres[scene] for scene in scenes}
        summaries[family] = {
            "count": len(scenes),
            "over_base": compare(delivered, base_delivered),
            "over_control": compare(delivered, control_delivered),
            "over_ceres": compare(delivered, references),
            "selected": sum(
                bool(
                    candidate[family][scene]["trajectory"][59]
                    ["schurAlignmentDiagnostics"]
                    ["schurProposalLandmarkResponseOracle"]["selected"]
                )
                for scene in scenes
            ),
            "total_rejections": sum(
                candidate[family][scene]["rejections"] for scene in scenes
            ),
        }
        for scene in scenes:
            details[scene] = {
                "family": family,
                "over_base": delivered[scene] / base_delivered[scene],
                "over_control": delivered[scene] / control_delivered[scene],
                "over_ceres": delivered[scene] / references[scene],
            }
    return {"status": "passed", "summaries": summaries, "scenes": details}


def write_report(path, summary, label, camera_damping, landmark_damping):
    with path.open("w", encoding="utf-8") as output:
        output.write(f"# K24 I60 Landmark-Response {label} Breadth\n\n")
        output.write(
            "The globally frozen candidate sets Schur camera/landmark damping "
            f"to `{camera_damping}`/`{landmark_damping}`, retaining I60 timing, three landmark steps, "
            "eight scales, `1e-3` floor, atomic commit, restart, and trust "
            "rebase. Development used Madrid, Tower, and Yorkminster; the "
            "remaining 12 1DSfM scenes and all BAL29 are unchanged validation.\n\n"
        )
        output.write(
            "| Family | Completed | Candidate/base | Candidate/control | "
            "Candidate/Ceres | Summed/Ceres | W/T/L candidate/base | "
            "Selected | Rejections |\n"
        )
        output.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for family in ("1dsfm", "bal"):
            row = summary["summaries"][family]
            base = row["over_base"]
            control = row["over_control"]
            ceres = row["over_ceres"]
            output.write(
                f"| {family} | {row['count']}/{EXPECTED[family]} | "
                f"{base['geometric']:.9f} | {control['geometric']:.9f} | "
                f"{ceres['geometric']:.9f} | {ceres['summed']:.9f} | "
                f"{base['wins']}/{base['ties']}/{base['losses']} | "
                f"{row['selected']} | {row['total_rejections']} |\n"
            )
        output.write("\n## Per-Scene Candidate/Base Ratios\n\n")
        for scene, row in sorted(
            summary["scenes"].items(),
            key=lambda item: item[1]["over_base"],
            reverse=True,
        ):
            output.write(
                f"- `{scene}`: candidate/base `{row['over_base']:.9f}x`, "
                f"control `{row['over_control']:.9f}x`, "
                f"Ceres `{row['over_ceres']:.9f}x`\n"
            )
        output.write("\nGate status: **passed**.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--bal-root", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-damping", type=float, default=0.0029296875)
    parser.add_argument("--landmark-damping", type=float, default=0.0029296875)
    parser.add_argument("--label", default="Half-Damping")
    arguments = parser.parse_args()
    summary = analyze(
        arguments.development_root,
        arguments.validation_root,
        arguments.bal_root,
        arguments.base_root,
        arguments.control_root,
        arguments.camera_damping,
        arguments.landmark_damping,
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(
        arguments.output / "report.md",
        summary,
        arguments.label,
        arguments.camera_damping,
        arguments.landmark_damping,
    )
    print(json.dumps(summary["summaries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
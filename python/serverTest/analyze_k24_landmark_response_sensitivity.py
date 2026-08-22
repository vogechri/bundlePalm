#!/usr/bin/env python3
"""Analyze bounded K24 landmark-response parameter sensitivity."""

import argparse
import json
import math
from pathlib import Path

from analyze_k24_one_step_schur_proposal_breadth import ceres_rows, load_rows


ARMS = {
    "damping_half": {"damping": 0.0029296875, "steps": 3},
    "base": {"damping": 0.005859375, "steps": 3},
    "damping_double": {"damping": 0.01171875, "steps": 3},
    "landmarks_1": {"damping": 0.005859375, "steps": 1},
    "landmarks_5": {"damping": 0.005859375, "steps": 5},
}
SCENES = ("madrid_metropolis", "tower_of_london", "yorkminster")


def geometric_mean(values):
    values = tuple(values)
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def analyze(root, control_root):
    controls = load_rows(control_root / "1dsfm" / "i200")
    ceres = ceres_rows()
    details = {}
    summaries = {}
    for arm, expected in ARMS.items():
        rows = load_rows(root / arm / "1dsfm")
        if set(rows) != set(SCENES):
            raise ValueError(f"coverage mismatch {arm}: {sorted(rows)}")
        arm_rows = []
        for scene in SCENES:
            row = rows[scene]
            if row.get("oneStepSchurResidualProposalIterations") != [60]:
                raise ValueError(f"proposal checkpoint mismatch {arm}/{scene}")
            if row.get("schurAlignmentCameraDamping") != expected["damping"]:
                raise ValueError(f"camera damping mismatch {arm}/{scene}")
            if row.get("schurAlignmentLandmarkDamping") != expected["damping"]:
                raise ValueError(f"landmark damping mismatch {arm}/{scene}")
            if row.get("sharedSchurLandmarkRefinementSteps") != expected["steps"]:
                raise ValueError(f"landmark depth mismatch {arm}/{scene}")
            diagnostic = row["trajectory"][59]["schurAlignmentDiagnostics"]
            proposal = diagnostic["schurProposalLandmarkResponseOracle"]
            control = min(
                value["sumSquaredError"]
                for value in controls[scene]["trajectory"][:120]
            )
            best_attempt = min(
                proposal["attempts"], key=lambda attempt: attempt["workerSSE"]
            )
            result = {
                "arm": arm,
                "scene": scene,
                "selected": bool(proposal["selected"]),
                "selected_scale": float(proposal["selectedScale"]),
                "best_scale": float(best_attempt["scale"]),
                "best_immediate_ratio": (
                    best_attempt["workerSSE"]
                    / proposal["ordinaryRefinedWorkerSSE"]
                ),
                "delivered_over_control": (
                    row["qualityMetrics"]["sumSquaredError"] / control
                ),
                "delivered_over_ceres": (
                    row["qualityMetrics"]["sumSquaredError"] / ceres[scene]
                ),
                "rejections": int(row["rejections"]),
            }
            details[f"{arm}/{scene}"] = result
            arm_rows.append(result)
        summaries[arm] = {
            "geometric_over_control": geometric_mean(
                result["delivered_over_control"] for result in arm_rows
            ),
            "geometric_over_ceres": geometric_mean(
                result["delivered_over_ceres"] for result in arm_rows
            ),
            "selected": sum(result["selected"] for result in arm_rows),
            "total_rejections": sum(result["rejections"] for result in arm_rows),
        }
    return {"summaries": summaries, "scenes": details}


def write_report(path, summary):
    with path.open("w", encoding="utf-8") as output:
        output.write("# K24 Landmark-Response Parameter Sensitivity\n\n")
        output.write(
            "One global parameter setting per arm is evaluated on Yorkminster, "
            "Tower of London, and declined Madrid. Proposal timing I60, scale "
            "grid, `1e-3` floor, restart, and trust policy are frozen.\n\n"
        )
        output.write(
            "| Arm | Damping | Landmark steps | Geometric/control | "
            "Geometric/Ceres | Selected | Rejections |\n"
        )
        output.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for arm, settings in ARMS.items():
            row = summary["summaries"][arm]
            output.write(
                f"| {arm} | {settings['damping']:.9g} | {settings['steps']} | "
                f"{row['geometric_over_control']:.9f} | "
                f"{row['geometric_over_ceres']:.9f} | "
                f"{row['selected']}/3 | {row['total_rejections']} |\n"
            )
        output.write("\n## Per-Scene Results\n\n")
        output.write(
            "| Arm | Scene | Selected scale | Best scale | Immediate ratio | "
            "Delivered/control | Delivered/Ceres | Rejections |\n"
        )
        output.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for arm in ARMS:
            for scene in SCENES:
                row = summary["scenes"][f"{arm}/{scene}"]
                selected = (
                    f"{row['selected_scale']:.6g}"
                    if row["selected"] else "declined"
                )
                output.write(
                    f"| {arm} | {scene} | {selected} | "
                    f"{row['best_scale']:.6g} | "
                    f"{row['best_immediate_ratio']:.9f} | "
                    f"{row['delivered_over_control']:.9f} | "
                    f"{row['delivered_over_ceres']:.9f} | "
                    f"{row['rejections']} |\n"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    arguments = parser.parse_args()
    summary = analyze(arguments.root, arguments.control_root)
    (arguments.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(arguments.root / "report.md", summary)
    print(json.dumps(summary["summaries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
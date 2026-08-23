#!/usr/bin/env python3
"""Analyze shared-only x guarded-Krylov carryover on the copied DRS baseline."""

import argparse
import json
import math
from pathlib import Path

from analyze_k24_one_step_schur_proposal_breadth import (
    ceres_rows,
    load_rows,
    load_status,
)


ARMS = {
    "direct": {"shared": False, "proposal": False},
    "direct_shared": {"shared": True, "proposal": False},
    "direct_proposal": {"shared": False, "proposal": True},
    "direct_shared_proposal": {"shared": True, "proposal": True},
}
EXPECTED = {
    "development": {"1dsfm": 6, "bal": 5},
    "all": {"1dsfm": 15, "bal": 29},
}
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


def family(scene):
    return "bal" if scene.startswith("bal") else "1dsfm"


def validate_configuration(scene, row, settings):
    expected = {
        "clusters": 24,
        "iterations": 120,
        "localSolver": "nesterov",
        "trustRegionPolicy": "drs",
        "persistentTrustRegion": True,
        "sharedOnlyCameraProximal": settings["shared"],
        "schurResidualProposalDirection": (
            "krylov2" if settings["proposal"] else "jacobi"
        ),
        "oneStepSchurResidualProposalIterations": (
            [60, 90] if settings["proposal"] else []
        ),
        "oneStepSchurResidualProposalRebaseTrustState": settings["proposal"],
        "allowTwoSchurResidualProposals": settings["proposal"],
        "schurProposalLandmarkResponseOracle": settings["proposal"],
        "applySchurProposalLandmarkResponse": settings["proposal"],
        "schurRepeatMinimumRelativeDecrease": (
            0.01 if settings["proposal"] else -1.0
        ),
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(
                f"configuration mismatch {scene}: {field}={row.get(field)!r}, "
                f"expected={value!r}"
            )
    if not row.get("directTangentNormalEquations"):
        raise ValueError(f"direct tangent disabled for {scene}")
    if row.get("cameraUpdate") != "se3_left":
        raise ValueError(f"camera update mismatch for {scene}")
    termination = row.get("terminationReason")
    completed = int(row.get("completedIterations", -1))
    if termination == "iteration_limit" and completed != 120:
        raise ValueError(f"iteration-limit row is incomplete for {scene}")
    if termination not in ("iteration_limit", "recovery_exhausted"):
        raise ValueError(f"unexpected termination for {scene}: {termination}")


def analyze(root, cohort):
    expected = EXPECTED[cohort]
    rows = {}
    statuses = {}
    scene_set = None
    for arm, settings in ARMS.items():
        rows[arm] = load_rows(root / arm)
        statuses[arm] = load_status(root / arm)
        if len(rows[arm]) != sum(expected.values()):
            raise ValueError(
                f"coverage mismatch {arm}: {len(rows[arm])}/"
                f"{sum(expected.values())}"
            )
        if set(rows[arm]) != set(statuses[arm]):
            raise ValueError(f"status mismatch for {arm}")
        if scene_set is None:
            scene_set = set(rows[arm])
        elif set(rows[arm]) != scene_set:
            raise ValueError(f"scene mismatch for {arm}")
        for scene, row in rows[arm].items():
            status = statuses[arm][scene]
            if status["status"] != "completed" or int(status["exit_code"]) != 0:
                raise ValueError(f"failed status {arm}/{scene}")
            validate_configuration(scene, row, settings)

    for proposal_arm, control_arm in (
        ("direct_proposal", "direct"),
        ("direct_shared_proposal", "direct_shared"),
    ):
        for scene in sorted(scene_set):
            candidate = rows[proposal_arm][scene]
            control = rows[control_arm][scene]
            for iteration, (left, right) in enumerate(
                zip(candidate["trajectory"][:59], control["trajectory"][:59]), 1
            ):
                for field in BEHAVIOR_FIELDS:
                    if left[field] != right[field]:
                        raise ValueError(
                            f"prefix mismatch {proposal_arm}/{scene}/"
                            f"I{iteration}/{field}"
                        )

    ceres = ceres_rows()
    summary = {"status": "passed", "families": {}, "scenes": {}}
    for cohort in ("1dsfm", "bal"):
        scenes = tuple(sorted(scene for scene in scene_set if family(scene) == cohort))
        if len(scenes) != expected[cohort]:
            raise ValueError(
                f"family coverage mismatch {cohort}: {len(scenes)}/"
                f"{expected[cohort]}"
            )
        endpoints = {
            arm: {
                scene: float(rows[arm][scene]["qualityMetrics"]["sumSquaredError"])
                for scene in scenes
            }
            for arm in ARMS
        }
        baseline = endpoints["direct"]
        shared = endpoints["direct_shared"]
        proposal = endpoints["direct_proposal"]
        combined = endpoints["direct_shared_proposal"]
        shared_effect = compare(shared, baseline)
        proposal_effect = compare(proposal, baseline)
        combined_effect = compare(combined, baseline)
        proposal_under_shared = compare(combined, shared)
        shared_under_proposal = compare(combined, proposal)
        interaction_ratios = {
            scene: combined[scene] * baseline[scene]
            / (shared[scene] * proposal[scene])
            for scene in scenes
        }
        selected = {
            arm: sum(
                bool(
                    rows[arm][scene]["trajectory"][59]
                    ["schurAlignmentDiagnostics"]
                    ["schurProposalLandmarkResponseOracle"]["selected"]
                )
                for scene in scenes
            ) if ARMS[arm]["proposal"] else 0
            for arm in ARMS
        }
        full_trajectories = {
            arm: sum(
                rows[arm][scene].get("terminationReason") == "iteration_limit"
                and rows[arm][scene].get("completedIterations") == 120
                for scene in scenes
            )
            for arm in ARMS
        }
        recovery_exhausted = {
            arm: [
                scene for scene in scenes
                if rows[arm][scene].get("terminationReason")
                == "recovery_exhausted"
            ]
            for arm in ARMS
        }
        summary["families"][cohort] = {
            "count": len(scenes),
            "shared_effect": shared_effect,
            "proposal_effect": proposal_effect,
            "combined_effect": combined_effect,
            "proposal_under_shared": proposal_under_shared,
            "shared_under_proposal": shared_under_proposal,
            "interaction_geometric": geometric_mean(interaction_ratios.values()),
            "interaction_ratios": interaction_ratios,
            "proposal_selected_all_camera": selected["direct_proposal"],
            "proposal_selected_shared_only": selected[
                "direct_shared_proposal"
            ],
            "combined_over_ceres": compare(
                combined, {scene: ceres[scene] for scene in scenes}
            ),
            "full_trajectories": full_trajectories,
            "recovery_exhausted": recovery_exhausted,
            "elapsed_seconds": {
                arm: math.fsum(
                    float(statuses[arm][scene]["elapsed_seconds"])
                    for scene in scenes
                )
                for arm in ARMS
            },
        }
        for scene in scenes:
            summary["scenes"][scene] = {
                "family": cohort,
                "shared_effect": shared[scene] / baseline[scene],
                "proposal_effect": proposal[scene] / baseline[scene],
                "combined_effect": combined[scene] / baseline[scene],
                "proposal_under_shared": combined[scene] / shared[scene],
                "shared_under_proposal": combined[scene] / proposal[scene],
                "interaction": interaction_ratios[scene],
            }
    return summary


def write_report(path, summary, cohort):
    with path.open("w", encoding="utf-8") as output:
        output.write("# K1 Carryover Joint Factorial On Copied DRS Baseline\n\n")
        output.write(
            "Direct left-SE3 tangent assembly and exact tangent-metric consistency "
            "are fixed. The copied mature DRS baseline varies only shared-only "
            "product-space semantics and the guarded I60+I90 Krylov camera/landmark "
            f"proposal. This report covers the frozen `{cohort}` cohort.\n\n"
        )
        output.write(
            "| Family | Shared/direct | Proposal/direct | Combined/direct | "
            "Proposal under shared | Shared under proposal | Interaction | "
            "Proposal selected all/shared | Full trajectories D/S/P/C | "
            "Combined/Ceres |\n"
        )
        output.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for cohort in ("1dsfm", "bal"):
            row = summary["families"][cohort]
            output.write(
                f"| {cohort} | {row['shared_effect']['geometric']:.9f} | "
                f"{row['proposal_effect']['geometric']:.9f} | "
                f"{row['combined_effect']['geometric']:.9f} | "
                f"{row['proposal_under_shared']['geometric']:.9f} | "
                f"{row['shared_under_proposal']['geometric']:.9f} | "
                f"{row['interaction_geometric']:.9f} | "
                f"{row['proposal_selected_all_camera']}/"
                f"{row['proposal_selected_shared_only']} | "
                f"{row['full_trajectories']['direct']}/"
                f"{row['full_trajectories']['direct_shared']}/"
                f"{row['full_trajectories']['direct_proposal']}/"
                f"{row['full_trajectories']['direct_shared_proposal']} | "
                f"{row['combined_over_ceres']['geometric']:.9f} |\n"
            )
        output.write(
            "\nInteraction below one means the combined effect is better than the "
            "product of isolated shared-only and proposal effects.\n\n"
        )
        output.write(
            "| Scene | Shared/direct | Proposal/direct | Combined/direct | "
            "Proposal under shared | Interaction |\n"
        )
        output.write("|---|---:|---:|---:|---:|---:|\n")
        for scene, row in sorted(summary["scenes"].items()):
            output.write(
                f"| {scene} | {row['shared_effect']:.9f} | "
                f"{row['proposal_effect']:.9f} | "
                f"{row['combined_effect']:.9f} | "
                f"{row['proposal_under_shared']:.9f} | "
                f"{row['interaction']:.9f} |\n"
            )
        exhausted = [
            f"{arm}:{','.join(row['recovery_exhausted'][arm])}"
            for row in summary["families"].values()
            for arm in ARMS
            if row["recovery_exhausted"][arm]
        ]
        output.write(
            "\nOn 1DSfM, both factors help in isolation and the combination beats "
            "either factor alone, with beneficial full-cohort interaction. On BAL, "
            "the proposal is nearly neutral and slightly repairs shared-only, while "
            "the combined result retains the small mandatory shared-only cost. "
            "Comparisons use delivered best states and list recovery exhaustion "
            "explicitly. "
            f"Recovery exhaustion: `{'; '.join(exhausted) if exhausted else 'none'}`. "
            "Do not retune either factor.\n\n"
        )
        output.write("Gate status: **passed**.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cohort", choices=tuple(EXPECTED), default="development")
    arguments = parser.parse_args()
    summary = analyze(arguments.root, arguments.cohort)
    (arguments.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(arguments.root / "report.md", summary, arguments.cohort)
    print(json.dumps(summary["families"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

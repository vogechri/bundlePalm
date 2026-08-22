#!/usr/bin/env python3
"""Analyze rollback-safe full-Schur versus Krylov physical proposals at I60."""

import argparse
import json
import math
from pathlib import Path

from analyze_k24_one_step_schur_proposal_breadth import load_rows, load_status


SCENES = (
    "gendarmenmarkt",
    "madrid_metropolis",
    "tower_of_london",
    "yorkminster",
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


def validate(root, control_root):
    rows = load_rows(root / "1dsfm")
    controls = load_rows(control_root / "1dsfm" / "i200")
    statuses = load_status(root / "1dsfm")
    if set(rows) != set(SCENES) or set(statuses) != set(SCENES):
        raise ValueError(f"coverage mismatch: {sorted(rows)}")

    details = {}
    ratios = []
    for scene in SCENES:
        row = rows[scene]
        status = statuses[scene]
        expected = {
            "clusters": 24,
            "iterations": 120,
            "completedIterations": 120,
            "schurAlignmentDiagnosticIterations": [60],
            "oneStepSchurResidualProposalIterations": [],
            "schurAlignmentCameraDamping": DAMPING,
            "schurAlignmentLandmarkDamping": DAMPING,
            "sharedSchurLandmarkRefinementSteps": 3,
            "schurKrylovLandmarkResponseOracle": True,
            "applySchurKrylovLandmarkResponse": False,
        }
        for field, value in expected.items():
            if row.get(field) != value:
                raise ValueError(
                    f"configuration mismatch {scene}: {field}="
                    f"{row.get(field)!r}, expected={value!r}"
                )
        if status["status"] != "completed" or int(status["exit_code"]) != 0:
            raise ValueError(f"failed status for {scene}")
        control = controls[scene]
        if len(row["trajectory"]) != 120:
            raise ValueError(f"trajectory length mismatch for {scene}")
        for iteration, (candidate, reference) in enumerate(
            zip(row["trajectory"], control["trajectory"][:120]), 1
        ):
            for field in BEHAVIOR_FIELDS:
                if candidate[field] != reference[field]:
                    raise ValueError(
                        f"behavior mismatch {scene}/I{iteration}/{field}"
                    )
        diagnostics = row["trajectory"][59]["schurAlignmentDiagnostics"]
        schur = diagnostics["schur"]
        if schur["linearTermination"] != 0 or schur["relativeResidual"] >= 1e-6:
            raise ValueError(f"nonconverged Schur reference for {scene}")
        krylov = diagnostics[
            "twoDirectionSchurKrylovLandmarkResponseOracle"
        ]
        full = diagnostics["fullSchurLandmarkResponseOracle"]
        for name, oracle in (("krylov", krylov), ("full", full)):
            if oracle is None or len(oracle["attempts"]) != 8:
                raise ValueError(f"{name} telemetry mismatch for {scene}")
            if oracle["workerStateRoundtripRelativeError"] > 1e-9:
                raise ValueError(f"{name} roundtrip mismatch for {scene}")
            if oracle["applied"]:
                raise ValueError(f"{name} oracle unexpectedly applied for {scene}")
        ratio = (
            full["candidateRefinedWorkerSSE"]
            / krylov["candidateRefinedWorkerSSE"]
        )
        ordinary_model = diagnostics["consensusModel"][
            "dampedPredictedReduction"
        ]
        krylov_predicted = (
            diagnostics["twoDirectionSchurKrylovOracle"]["physicalModel"]
            ["dampedPredictedReduction"] - ordinary_model
        )
        full_predicted = (
            diagnostics["fullSchurPhysicalModel"]["dampedPredictedReduction"]
            - ordinary_model
        )
        krylov_scale_one = next(
            attempt["workerSSE"]
            for attempt in krylov["attempts"]
            if attempt["scale"] == 1.0
        )
        full_scale_one = next(
            attempt["workerSSE"]
            for attempt in full["attempts"]
            if attempt["scale"] == 1.0
        )
        ordinary_sse = krylov["ordinaryRefinedWorkerSSE"]
        krylov_actual = 0.5 * (ordinary_sse - krylov_scale_one)
        full_actual = 0.5 * (ordinary_sse - full_scale_one)
        predicted_preference = (
            "full" if full_predicted > krylov_predicted else "krylov"
        )
        actual_preference = "full" if full_actual > krylov_actual else "krylov"
        ratios.append(ratio)
        details[scene] = {
            "schur_relative_residual": schur["relativeResidual"],
            "krylov_selected": bool(krylov["selected"]),
            "krylov_scale": float(krylov["selectedScale"]),
            "krylov_over_ordinary": (
                krylov["candidateRefinedWorkerSSE"]
                / krylov["ordinaryRefinedWorkerSSE"]
            ),
            "full_selected": bool(full["selected"]),
            "full_scale": float(full["selectedScale"]),
            "full_over_ordinary": (
                full["candidateRefinedWorkerSSE"]
                / full["ordinaryRefinedWorkerSSE"]
            ),
            "full_over_krylov": ratio,
            "krylov_predicted_increment": krylov_predicted,
            "full_predicted_increment": full_predicted,
            "krylov_actual_increment": krylov_actual,
            "full_actual_increment": full_actual,
            "predicted_preference": predicted_preference,
            "actual_preference": actual_preference,
            "ranking_match": predicted_preference == actual_preference,
        }
    return {
        "status": "passed",
        "geometric_full_over_krylov": geometric_mean(ratios),
        "wins_ties_losses": {
            "wins": sum(value < 1.0 - 1e-9 for value in ratios),
            "ties": sum(abs(value - 1.0) <= 1e-9 for value in ratios),
            "losses": sum(value > 1.0 + 1e-9 for value in ratios),
        },
        "scenes": details,
    }


def write_report(path, summary):
    with path.open("w", encoding="utf-8") as output:
        output.write("# K24 I60 Full-Schur Physical Oracle\n\n")
        output.write(
            "A converged shared-camera Schur tangent and the promoted two-direction "
            "Krylov tangent are evaluated with identical eight-scale searches, "
            "three landmark-response steps, and rollback-safe worker state. All "
            "120 ordinary-control trajectory rows are exact.\n\n"
        )
        output.write(
            "| Scene | Krylov scale | Krylov/ordinary | Full scale | "
            "Full/ordinary | Full/Krylov | Model/actual preference | "
            "Schur residual |\n"
        )
        output.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for scene in SCENES:
            row = summary["scenes"][scene]
            full_scale = (
                f"{row['full_scale']:.6g}" if row["full_selected"] else "declined"
            )
            output.write(
                f"| {scene} | {row['krylov_scale']:.6g} | "
                f"{row['krylov_over_ordinary']:.9f} | {full_scale} | "
                f"{row['full_over_ordinary']:.9f} | "
                f"{row['full_over_krylov']:.9f} | "
                f"{row['predicted_preference']}/{row['actual_preference']} | "
                f"{row['schur_relative_residual']:.3e} |\n"
            )
        wtl = summary["wins_ties_losses"]
        output.write(
            "\nFull/Krylov geometric ratio is "
            f"`{summary['geometric_full_over_krylov']:.9f}x`, W/T/L "
            f"`{wtl['wins']}/{wtl['ties']}/{wtl['losses']}`. Full Schur helps "
            "Tower, but loses on Gendarmenmarkt and Yorkminster and declines "
            "Madrid. More linear convergence is therefore not a transferable "
            "proposal mechanism. The hybrid linear model ranks the two "
            "directions correctly on three scenes but incorrectly favors full "
            "Schur on Yorkminster. Keep bounded Krylov2 and move to basin/model "
            "diagnosis.\n\n"
        )
        output.write("Gate status: **passed**.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    arguments = parser.parse_args()
    summary = validate(arguments.root, arguments.control_root)
    (arguments.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(arguments.root / "report.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

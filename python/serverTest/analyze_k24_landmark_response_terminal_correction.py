#!/usr/bin/env python3
"""Analyze promoted landmark-response DRS plus one terminal correction."""

import argparse
import json
import math
from pathlib import Path

from analyze_k24_one_step_schur_proposal_breadth import (
    EXPECTED,
    ceres_rows,
    load_rows,
    load_status,
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


def validate_row(scene, polished, handoff):
    expected = {
        "clusters": 24,
        "iterations": 120,
        "completedIterations": 120,
        "applySchurProposalLandmarkResponse": True,
        "oneStepSchurResidualProposalRebaseTrustState": True,
        "sharedSchurMaximumCorrections": 1,
        "sharedSchurCameraDamping": 0.005859375,
        "sharedSchurLandmarkDamping": 0.005859375,
        "sharedSchurMinimumRelativeDecrease": 1e-3,
        "sharedSchurRelativeTolerance": 1e-6,
        "sharedSchurOperator": "bsr_low_memory",
        "sharedSchurPreconditioner": "jacobi",
    }
    for field, value in expected.items():
        if polished.get(field) != value:
            raise ValueError(
                f"configuration mismatch {scene}: {field}="
                f"{polished.get(field)!r}, expected={value!r}"
            )
    if polished.get("terminationReason") != "iteration_limit":
        raise ValueError(f"termination mismatch for {scene}")
    if not polished.get("finalSharedSchurAttempted"):
        raise ValueError(f"terminal correction not attempted for {scene}")
    if len(polished["trajectory"]) != len(handoff["trajectory"]):
        raise ValueError(f"trajectory length mismatch for {scene}")
    for iteration, (left, right) in enumerate(
        zip(polished["trajectory"], handoff["trajectory"]), 1
    ):
        if left["sumSquaredError"] != right["sumSquaredError"]:
            raise ValueError(f"handoff trajectory mismatch {scene}/I{iteration}")

    initial = float(polished["finalSharedSchurInitialSSE"])
    final = float(polished["finalSharedSchurCorrectedSSE"])
    delivered = float(polished["qualityMetrics"]["sumSquaredError"])
    handoff_sse = min(
        float(row["sumSquaredError"]) for row in handoff["trajectory"]
    )
    if initial != handoff_sse:
        raise ValueError(f"terminal handoff mismatch for {scene}")
    if not math.isclose(final, delivered, rel_tol=1e-12):
        raise ValueError(f"terminal delivered mismatch for {scene}")

    attempts = polished["finalSharedSchurAttempts"]
    accepted = [attempt for attempt in attempts if attempt["accepted"]]
    accepted_count = int(polished["finalSharedSchurAcceptedCorrections"])
    if len(accepted) != accepted_count or accepted_count not in (0, 1):
        raise ValueError(f"accepted correction mismatch for {scene}")
    if bool(accepted) != bool(polished["finalSharedSchurAccepted"]):
        raise ValueError(f"terminal acceptance mismatch for {scene}")
    for attempt in accepted:
        if attempt["diagnostics"]["linearTermination"] != 0:
            raise ValueError(f"accepted nonconverged solve for {scene}")
        disagreement = abs(attempt["workerSSE"] - attempt["candidateSSE"])
        if disagreement > 1e-7 * max(1.0, abs(attempt["candidateSSE"])):
            raise ValueError(f"worker/evaluator disagreement for {scene}")
    if accepted:
        if not final < initial:
            raise ValueError(f"accepted correction did not improve {scene}")
    elif not math.isclose(final, initial, rel_tol=1e-12):
        raise ValueError(f"no-op correction changed state for {scene}")


def analyze(root, handoff_root):
    ceres = ceres_rows()
    summaries = {}
    details = {}
    for family in ("1dsfm", "bal"):
        polished = load_rows(root / family)
        handoff = load_rows(handoff_root / family)
        status = load_status(root / family)
        if len(polished) != EXPECTED[family]:
            raise ValueError(
                f"coverage mismatch {family}: {len(polished)}/{EXPECTED[family]}"
            )
        if set(polished) != set(handoff) or set(polished) != set(status):
            raise ValueError(f"scene/status mismatch for {family}")
        for scene in polished:
            if (
                status[scene]["status"] != "completed"
                or int(status[scene]["exit_code"]) != 0
            ):
                raise ValueError(f"failed status for {scene}")
            validate_row(scene, polished[scene], handoff[scene])

        scenes = tuple(sorted(polished))
        initial = {
            scene: float(polished[scene]["finalSharedSchurInitialSSE"])
            for scene in scenes
        }
        final = {
            scene: float(polished[scene]["finalSharedSchurCorrectedSSE"])
            for scene in scenes
        }
        references = {scene: ceres[scene] for scene in scenes}
        corrected_over_handoff = compare(final, initial)
        corrected_over_ceres = compare(final, references)
        accepted = sum(
            bool(polished[scene]["finalSharedSchurAccepted"])
            for scene in scenes
        )
        nonconverged_noops = sum(
            not polished[scene]["finalSharedSchurAccepted"]
            and any(
                attempt["diagnostics"]["linearTermination"] != 0
                for attempt in polished[scene]["finalSharedSchurAttempts"]
            )
            for scene in scenes
        )
        summaries[family] = {
            "count": len(scenes),
            "corrected_over_handoff": corrected_over_handoff,
            "corrected_over_ceres": corrected_over_ceres,
            "accepted": accepted,
            "noops": len(scenes) - accepted,
            "nonconverged_noops": nonconverged_noops,
            "schur_seconds": math.fsum(
                polished[scene]["finalSharedSchurSeconds"] for scene in scenes
            ),
            "elapsed_seconds": math.fsum(
                float(status[scene]["elapsed_seconds"]) for scene in scenes
            ),
            "maximum_coordinator_rss_gib": max(
                int(status[scene]["coordinator_max_rss_kb"])
                for scene in scenes
            ) / 1048576.0,
            "maximum_worker_rss_gib": max(
                int(status[scene]["worker_max_rss_kb"])
                for scene in scenes
            ) / 1048576.0,
        }
        for scene in scenes:
            details[scene] = {
                "family": family,
                "corrected_over_handoff": final[scene] / initial[scene],
                "corrected_over_ceres": final[scene] / references[scene],
                "accepted": bool(polished[scene]["finalSharedSchurAccepted"]),
                "attempts": len(polished[scene]["finalSharedSchurAttempts"]),
                "final_damping": polished[scene][
                    "finalSharedSchurFinalCameraDamping"
                ],
            }
    return {"status": "passed", "summaries": summaries, "scenes": details}


def write_report(path, summary):
    with path.open("w", encoding="utf-8") as output:
        output.write("# K24 Landmark-Response DRS Plus One Terminal Correction\n\n")
        output.write(
            "Promoted K24/I120 landmark-response DRS followed by exactly one "
            "safeguarded distributed Schur correction. The correction uses "
            "damping `0.005859375`, `bsr_low_memory`, Jacobi PCG, relative "
            "tolerance `1e-6`, and progress floor `1e-3`.\n\n"
        )
        output.write(
            "| Family | Completed | Correction/handoff | Corrected/Ceres | "
            "Summed/Ceres | W/T/L correction | Accepted/no-op | "
            "Nonconverged no-op | Max RSS GiB C/W |\n"
        )
        output.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for family in ("1dsfm", "bal"):
            row = summary["summaries"][family]
            correction = row["corrected_over_handoff"]
            ceres = row["corrected_over_ceres"]
            output.write(
                f"| {family} | {row['count']}/{EXPECTED[family]} | "
                f"{correction['geometric']:.9f} | {ceres['geometric']:.9f} | "
                f"{ceres['summed']:.9f} | "
                f"{correction['wins']}/{correction['ties']}/{correction['losses']} | "
                f"{row['accepted']}/{row['noops']} | "
                f"{row['nonconverged_noops']} | "
                f"{row['maximum_coordinator_rss_gib']:.3f}/"
                f"{row['maximum_worker_rss_gib']:.3f} |\n"
            )
        output.write("\n| Scene | Correction/handoff | Corrected/Ceres | Status | Attempts | Final damping |\n")
        output.write("|---|---:|---:|---|---:|---:|\n")
        for scene, row in sorted(summary["scenes"].items()):
            output.write(
                f"| {scene} | {row['corrected_over_handoff']:.9f} | "
                f"{row['corrected_over_ceres']:.9f} | "
                f"{'accepted' if row['accepted'] else 'no-op'} | "
                f"{row['attempts']} | {row['final_damping']:.9g} |\n"
            )
        output.write("\nGate status: **passed**.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--handoff-root", type=Path, required=True)
    arguments = parser.parse_args()
    summary = analyze(arguments.root, arguments.handoff_root)
    (arguments.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(arguments.root / "report.md", summary)
    print(json.dumps(summary["summaries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Analyze K4/K16 transfer of the frozen combined K1 carryover stack."""

import argparse
import csv
import json
import math
import re
from pathlib import Path

from analyze_k24_one_step_schur_proposal_breadth import ceres_rows


ARMS = {
    "direct": {"shared": False, "proposal": False},
    "direct_shared_proposal": {"shared": True, "proposal": True},
}
CLUSTERS = (4, 16)
EXPECTED = {"1dsfm": 15, "bal": 29}
DEVELOPMENT_SCENES = {
    "gendarmenmarkt", "piccadilly", "roman_forum", "trafalgar",
    "union_square", "vienna_cathedral", "bal52", "bal245", "bal1490",
    "bal1778", "bal3068",
}
MAX_AGGREGATE_CONTROL_RATIO = 1.01
MAX_SCENE_CONTROL_RATIO = 1.02


def scene_key(row):
    dataset = Path(row["dataset"])
    match = re.search(r"problem-(\d+)-", dataset.name)
    return f"bal{match.group(1)}" if match else dataset.parent.name


def load_rows(directory):
    rows = {}
    for path in directory.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = scene_key(row), int(row["clusters"])
            if key in rows:
                raise ValueError(f"duplicate result row: {key}")
            rows[key] = row
    return rows


def load_status(directory):
    path = directory / "status.tsv"
    rows = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            dataset = Path(row["dataset"])
            match = re.search(r"problem-(\d+)-", dataset.name)
            scene = f"bal{match.group(1)}" if match else dataset.parent.name
            key = scene, int(row["clusters"])
            if key in rows:
                raise ValueError(f"duplicate status row: {key}")
            rows[key] = row
    return rows


def geometric_mean(values):
    values = tuple(values)
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def compare(candidate, reference, tolerance=1e-9):
    ratios = {key: candidate[key] / reference[key] for key in sorted(candidate)}
    labeled_ratios = {
        f"{key[0]}/K{key[1]}" if isinstance(key, tuple) else str(key): value
        for key, value in ratios.items()
    }
    return {
        "geometric": geometric_mean(ratios.values()),
        "summed": math.fsum(candidate.values()) / math.fsum(reference.values()),
        "wins": sum(value < 1.0 - tolerance for value in ratios.values()),
        "ties": sum(abs(value - 1.0) <= tolerance for value in ratios.values()),
        "losses": sum(value > 1.0 + tolerance for value in ratios.values()),
        "worst": max(ratios.values()),
        "ratios": labeled_ratios,
    }


def validate_row(key, row, settings):
    scene, clusters = key
    expected = {
        "clusters": clusters,
        "iterations": 120,
        "localSolver": "nesterov",
        "trustRegionPolicy": "drs",
        "persistentTrustRegion": True,
        "sharedOnlyCameraProximal": settings["shared"],
        "schurResidualProposalDirection": "krylov2" if settings["proposal"] else "jacobi",
        "oneStepSchurResidualProposalIterations": [60, 90] if settings["proposal"] else [],
        "oneStepSchurResidualProposalRebaseTrustState": settings["proposal"],
        "allowTwoSchurResidualProposals": settings["proposal"],
        "schurProposalLandmarkResponseOracle": settings["proposal"],
        "applySchurProposalLandmarkResponse": settings["proposal"],
        "schurRepeatMinimumRelativeDecrease": 0.01 if settings["proposal"] else -1.0,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(
                f"configuration mismatch {scene}/K{clusters}: {field}="
                f"{row.get(field)!r}, expected={value!r}"
            )
    if not row.get("directTangentNormalEquations"):
        raise ValueError(f"direct tangent disabled for {scene}/K{clusters}")
    if row.get("cameraUpdate") != "se3_left":
        raise ValueError(f"camera update mismatch for {scene}/K{clusters}")
    termination = row.get("terminationReason")
    completed = int(row.get("completedIterations", -1))
    if termination == "iteration_limit" and completed != 120:
        raise ValueError(f"incomplete iteration-limit row {scene}/K{clusters}")
    if termination not in ("iteration_limit", "recovery_exhausted"):
        raise ValueError(f"unexpected termination {scene}/K{clusters}: {termination}")


def analyze(root, cohort):
    rows = {arm: load_rows(root / arm) for arm in ARMS}
    statuses = {arm: load_status(root / arm) for arm in ARMS}
    if cohort == "development":
        rows = {
            arm: {
                key: row for key, row in arm_rows.items()
                if key[0] in DEVELOPMENT_SCENES
            }
            for arm, arm_rows in rows.items()
        }
        statuses = {
            arm: {
                key: row for key, row in arm_rows.items()
                if key[0] in DEVELOPMENT_SCENES
            }
            for arm, arm_rows in statuses.items()
        }
        expected = {"1dsfm": 6, "bal": 5}
    else:
        expected = EXPECTED
    expected_count = sum(expected.values()) * len(CLUSTERS)
    expected_keys = None
    for arm, settings in ARMS.items():
        if len(rows[arm]) != expected_count:
            raise ValueError(f"coverage mismatch {arm}: {len(rows[arm])}/{expected_count}")
        if set(rows[arm]) != set(statuses[arm]):
            raise ValueError(f"status mismatch for {arm}")
        if expected_keys is None:
            expected_keys = set(rows[arm])
        elif set(rows[arm]) != expected_keys:
            raise ValueError(f"scene/K mismatch for {arm}")
        for key, row in rows[arm].items():
            status = statuses[arm][key]
            if status["status"] != "completed" or int(status["exit_code"]) != 0:
                raise ValueError(f"failed status {arm}/{key}")
            validate_row(key, row, settings)

    ceres = ceres_rows()
    summary = {"status": "passed", "gate_failures": [], "families": {}, "scenes": {}}
    for family in ("1dsfm", "bal"):
        summary["families"][family] = {}
        for clusters in CLUSTERS:
            keys = tuple(sorted(
                key for key in expected_keys
                if key[1] == clusters
                and (key[0].startswith("bal") if family == "bal" else not key[0].startswith("bal"))
            ))
            if len(keys) != expected[family]:
                raise ValueError(
                    f"family coverage mismatch {family}/K{clusters}: "
                    f"{len(keys)}/{expected[family]}"
                )
            control = {
                key: float(rows["direct"][key]["qualityMetrics"]["sumSquaredError"])
                for key in keys
            }
            candidate = {
                key: float(rows["direct_shared_proposal"][key]["qualityMetrics"]["sumSquaredError"])
                for key in keys
            }
            references = {key: ceres[key[0]] for key in keys}
            proposal_rows = rows["direct_shared_proposal"]
            attempted = {
                checkpoint: sum(
                    len(proposal_rows[key]["trajectory"]) >= checkpoint
                    and proposal_rows[key]["trajectory"][checkpoint - 1]
                    ["schurAlignmentDiagnostics"] is not None
                    for key in keys
                )
                for checkpoint in (60, 90)
            }
            selected = {
                checkpoint: sum(
                    len(proposal_rows[key]["trajectory"]) >= checkpoint
                    and proposal_rows[key]["trajectory"][checkpoint - 1]
                    ["schurAlignmentDiagnostics"] is not None
                    and bool(
                        proposal_rows[key]["trajectory"][checkpoint - 1]
                        ["schurAlignmentDiagnostics"]
                        ["schurProposalLandmarkResponseOracle"]["selected"]
                    )
                    for key in keys
                )
                for checkpoint in (60, 90)
            }
            full = {
                arm: sum(
                    rows[arm][key].get("terminationReason") == "iteration_limit"
                    and rows[arm][key].get("completedIterations") == 120
                    for key in keys
                )
                for arm in ARMS
            }
            exhausted = {
                arm: [key[0] for key in keys if rows[arm][key].get("terminationReason") == "recovery_exhausted"]
                for arm in ARMS
            }
            summary["families"][family][f"K{clusters}"] = {
                "count": len(keys),
                "candidate_over_control": compare(candidate, control),
                "candidate_over_ceres": compare(candidate, references),
                "proposal_attempted": attempted,
                "proposal_selected": selected,
                "full_trajectories": full,
                "recovery_exhausted": exhausted,
                "elapsed_seconds": {
                    arm: math.fsum(float(statuses[arm][key]["elapsed_seconds"]) for key in keys)
                    for arm in ARMS
                },
                "elapsed_candidate_over_control": (
                    math.fsum(
                        float(statuses["direct_shared_proposal"][key]["elapsed_seconds"])
                        for key in keys
                    )
                    / math.fsum(
                        float(statuses["direct"][key]["elapsed_seconds"])
                        for key in keys
                    )
                ),
                "maximum_coordinator_rss_gib": max(
                    int(statuses["direct_shared_proposal"][key]["coordinator_max_rss_kb"])
                    for key in keys
                ) / 1048576.0,
                "maximum_worker_rss_gib": max(
                    int(statuses["direct_shared_proposal"][key]["worker_max_rss_kb"])
                    for key in keys
                ) / 1048576.0,
            }
            for key in keys:
                scene = key[0]
                summary["scenes"][f"{scene}/K{clusters}"] = {
                    "family": family,
                    "clusters": clusters,
                    "candidate_over_control": candidate[key] / control[key],
                    "candidate_over_ceres": candidate[key] / references[key],
                }
        k4 = {
            scene: float(rows["direct_shared_proposal"][(scene, 4)]["qualityMetrics"]["sumSquaredError"])
            for scene, clusters in expected_keys
            if clusters == 4 and (scene.startswith("bal") if family == "bal" else not scene.startswith("bal"))
        }
        k16 = {
            scene: float(rows["direct_shared_proposal"][(scene, 16)]["qualityMetrics"]["sumSquaredError"])
            for scene, clusters in expected_keys
            if clusters == 16 and (scene.startswith("bal") if family == "bal" else not scene.startswith("bal"))
        }
        summary["families"][family]["K16_over_K4"] = compare(k16, k4)
        for arm in ARMS:
            summary["families"][family]["K16_over_K4"][
                f"elapsed_{arm}"
            ] = (
                summary["families"][family]["K16"]["elapsed_seconds"][arm]
                / summary["families"][family]["K4"]["elapsed_seconds"][arm]
            )
    for family in ("1dsfm", "bal"):
        for clusters in CLUSTERS:
            row = summary["families"][family][f"K{clusters}"]
            comparison = row["candidate_over_control"]
            if comparison["geometric"] > MAX_AGGREGATE_CONTROL_RATIO:
                summary["gate_failures"].append(
                    f"{family}/K{clusters} geometric candidate/control "
                    f"{comparison['geometric']:.9f} > {MAX_AGGREGATE_CONTROL_RATIO:.2f}"
                )
            if comparison["summed"] > MAX_AGGREGATE_CONTROL_RATIO:
                summary["gate_failures"].append(
                    f"{family}/K{clusters} summed candidate/control "
                    f"{comparison['summed']:.9f} > {MAX_AGGREGATE_CONTROL_RATIO:.2f}"
                )
            if comparison["worst"] > MAX_SCENE_CONTROL_RATIO:
                summary["gate_failures"].append(
                    f"{family}/K{clusters} worst candidate/control "
                    f"{comparison['worst']:.9f} > {MAX_SCENE_CONTROL_RATIO:.2f}"
                )
            exhausted = row["recovery_exhausted"]["direct_shared_proposal"]
            if exhausted:
                summary["gate_failures"].append(
                    f"{family}/K{clusters} candidate recovery exhausted: "
                    f"{','.join(exhausted)}"
                )
    if summary["gate_failures"]:
        summary["status"] = "failed"
    return summary


def write_report(path, summary, cohort):
    with path.open("w", encoding="utf-8") as output:
        output.write("# K1 Carryover Combined Stack At K4/K16\n\n")
        output.write(
            "The copied mature direct-left-SE3 baseline is compared with the frozen "
            "combined shared-only plus guarded I60+I90 Krylov stack. No K24-selected "
            "setting changes. K4 is the resource deployment point and K16 the latency "
            f"deployment point. This report covers the frozen `{cohort}` cohort.\n\n"
        )
        output.write(
            "| Family/K | Candidate/control | Candidate/Ceres | W/T/L control | "
            "Selected I60/I90 | Full trajectories control/candidate | "
            "Elapsed ratio | Max RSS GiB C/W |\n"
        )
        output.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for family in ("1dsfm", "bal"):
            for clusters in CLUSTERS:
                row = summary["families"][family][f"K{clusters}"]
                comp = row["candidate_over_control"]
                output.write(
                    f"| {family}/K{clusters} | {comp['geometric']:.9f} | "
                    f"{row['candidate_over_ceres']['geometric']:.9f} | "
                    f"{comp['wins']}/{comp['ties']}/{comp['losses']} | "
                    f"{row['proposal_selected'][60]}/"
                    f"{row['proposal_selected'][90]} "
                    f"({row['proposal_attempted'][60]}/"
                    f"{row['proposal_attempted'][90]} attempted) | "
                    f"{row['full_trajectories']['direct']}/"
                    f"{row['full_trajectories']['direct_shared_proposal']} | "
                    f"{row['elapsed_candidate_over_control']:.6f} | "
                    f"{row['maximum_coordinator_rss_gib']:.3f}/"
                    f"{row['maximum_worker_rss_gib']:.3f} |\n"
                )
        output.write(
            "\n| Family | K16/K4 SSE | W/T/L | Worst | "
            "K16/K4 elapsed control/candidate |\n"
        )
        output.write("|---|---:|---:|---:|---:|\n")
        for family in ("1dsfm", "bal"):
            row = summary["families"][family]["K16_over_K4"]
            output.write(
                f"| {family} | {row['geometric']:.9f} | "
                f"{row['wins']}/{row['ties']}/{row['losses']} | "
                f"{row['worst']:.9f} | "
                f"{row['elapsed_direct']:.6f}/"
                f"{row['elapsed_direct_shared_proposal']:.6f} |\n"
            )
        exhausted = []
        for family in ("1dsfm", "bal"):
            for clusters in CLUSTERS:
                row = summary["families"][family][f"K{clusters}"]
                for arm, scenes in row["recovery_exhausted"].items():
                    if scenes:
                        exhausted.append(
                            f"{family}/K{clusters}/{arm}:{','.join(scenes)}"
                        )
        output.write(
            "\nRecovery exhaustion: "
            f"`{'; '.join(exhausted) if exhausted else 'none'}`.\n\n"
        )
        if cohort == "development":
            output.write(
                "This is the frozen six-1DSfM/BAL5 development gate. Advance "
                "unchanged to held-out/all15/all29 only if both families have "
                "bounded aggregate quality and no unsafe candidate completion "
                "failure. Do not tune K-specific settings.\n\n"
            )
        else:
            output.write(
                "This is the unchanged all15/all29 confirmation. K4 and K16 are "
                "global deployment budgets, never scene-specific routing.\n\n"
            )
        output.write(
            "Gate bounds: geometric and summed candidate/control at most "
            f"`{MAX_AGGREGATE_CONTROL_RATIO:.2f}x`, worst scene at most "
            f"`{MAX_SCENE_CONTROL_RATIO:.2f}x`, and no candidate recovery "
            "exhaustion.\n\n"
        )
        if summary["gate_failures"]:
            output.write("Gate failures:\n\n")
            for failure in summary["gate_failures"]:
                output.write(f"- {failure}\n")
            output.write("\n")
        output.write(f"Gate status: **{summary['status']}**.\n")
        if cohort == "development" and summary["status"] == "failed":
            output.write(
                "\nDisposition: close this frozen common K4/K16 transfer "
                "without breadth expansion or K-specific retuning. Retain the "
                "accepted K24 combined stack and mandatory algebraic carryovers.\n"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--cohort", choices=("development", "all"), default="development"
    )
    arguments = parser.parse_args()
    summary = analyze(arguments.root, arguments.cohort)
    stem = "development_" if arguments.cohort == "development" else ""
    (arguments.root / f"{stem}summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(arguments.root / f"{stem}report.md", summary, arguments.cohort)
    print(json.dumps(summary["families"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

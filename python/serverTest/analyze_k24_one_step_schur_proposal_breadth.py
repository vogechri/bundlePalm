#!/usr/bin/env python3
"""Analyze low-memory I90 one-step Schur proposal breadth."""

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent
EXPECTED = {"1dsfm": 15, "bal": 29}
PREFIX_FIELDS = (
    "sumSquaredError",
    "refinedCandidateSumSquaredError",
    "rejected",
    "rejections",
    "proximalOracleCalls",
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


def load_status(directory):
    path = directory / "status.tsv"
    if not path.is_file():
        return {}
    with path.open(newline="") as source:
        return {
            scene_key({"dataset": row["dataset"]}): row
            for row in csv.DictReader(source, delimiter="\t")
        }


def ceres_rows():
    result = {}
    paths = (
        WORKSPACE / "benchmark_results/1dsfm_drs_ceres_se3_all15/ceres/results.jsonl",
        WORKSPACE / "benchmark_results/bal_ceres_se3_all29/results.jsonl",
    )
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            scene = row.get("scene") or f"bal{row['balId']}"
            result[scene] = (
                row["qualityMetrics"]["sumSquaredError"]
                if "qualityMetrics" in row
                else 2.0 * row["native"]["finalCeresCost"]
            )
    return result


def geometric_mean(values):
    values = tuple(values)
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def assert_prefix(control, candidate, scene):
    if len(control["trajectory"]) < 120 or len(candidate["trajectory"]) != 120:
        raise ValueError(f"trajectory length mismatch for {scene}")
    for iteration, (left, right) in enumerate(
        zip(control["trajectory"][:89], candidate["trajectory"][:89]), 1
    ):
        for field in PREFIX_FIELDS:
            if left[field] != right[field]:
                raise ValueError(f"prefix mismatch {scene}/I{iteration}/{field}")


def validate_configuration(row, scene):
    expected = {
        "clusters": 24,
        "iterations": 120,
        "completedIterations": 120,
        "localSolver": "nesterov",
        "outerAcceleration": "themelis_nesterov",
        "trustRegionPolicy": "drs",
        "persistentTrustRegion": True,
        "sharedOnlyCameraProximal": True,
        "oneStepSchurResidualProposalIterations": [90],
        "schurCoupledConsensusOracle": False,
        "sharedSchurMinimumRelativeDecrease": 1e-3,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(
                f"configuration mismatch {scene}: {field}={row.get(field)!r}, "
                f"expected={value!r}"
            )
    if row.get("schurAlignmentDiagnosticIterations") not in ([], [90]):
        raise ValueError(
            f"configuration mismatch {scene}: "
            "schurAlignmentDiagnosticIterations="
            f"{row.get('schurAlignmentDiagnosticIterations')!r}, "
            "expected [] or [90]"
        )
    if row.get("terminationReason") != "iteration_limit":
        raise ValueError(f"termination mismatch for {scene}")


def summarize(rows, ceres):
    delivered = [row["delivered_ratio"] for row in rows]
    trajectory = [row["trajectory_i120_ratio"] for row in rows]
    return {
        "count": len(rows),
        "selected": sum(row["selected"] for row in rows),
        "declined": sum(not row["selected"] for row in rows),
        "selection_scales": dict(sorted(Counter(
            str(row["selected_scale"]) if row["selected"] else "declined"
            for row in rows
        ).items())),
        "geometric_immediate_i90_ratio": geometric_mean(
            row["immediate_i90_ratio"] for row in rows
        ),
        "geometric_trajectory_i120_ratio": geometric_mean(trajectory),
        "geometric_delivered_ratio": geometric_mean(delivered),
        "summed_delivered_ratio": math.fsum(row["candidate_delivered"] for row in rows)
        / math.fsum(row["control_delivered"] for row in rows),
        "wins": sum(value < 1.0 - 1e-9 for value in delivered),
        "ties": sum(abs(value - 1.0) <= 1e-9 for value in delivered),
        "losses": sum(value > 1.0 + 1e-9 for value in delivered),
        "worst_delivered_ratio": max(delivered),
        "geometric_candidate_over_ceres": geometric_mean(
            row["candidate_delivered"] / ceres[row["scene"]] for row in rows
        ),
        "maximum_coordinator_rss_gib": max(
            row["coordinator_max_rss_kb"] for row in rows
        ) / 1048576.0,
        "maximum_worker_rss_gib": max(
            row["worker_max_rss_kb"] for row in rows
        ) / 1048576.0,
        "total_elapsed_seconds": math.fsum(row["elapsed_seconds"] for row in rows),
    }


def analyze(root, control_root, families):
    ceres = ceres_rows()
    details = {}
    summaries = {}
    for family in families:
        candidates = load_rows(root / family)
        controls = load_rows(control_root / family / "i200")
        statuses = load_status(root / family)
        expected = EXPECTED[family]
        if len(candidates) != expected:
            raise ValueError(
                f"candidate coverage mismatch {family}: {len(candidates)}/{expected}"
            )
        if set(candidates) != set(controls):
            raise ValueError(f"control scene mismatch for {family}")
        if set(candidates) != set(statuses):
            raise ValueError(f"status scene mismatch for {family}")
        family_rows = []
        for scene in sorted(candidates):
            candidate = candidates[scene]
            control = controls[scene]
            status = statuses[scene]
            validate_configuration(candidate, scene)
            assert_prefix(control, candidate, scene)
            if status["status"] != "completed" or int(status["exit_code"]) != 0:
                raise ValueError(f"failed case status for {scene}")
            diagnostic = candidate["trajectory"][89]["schurAlignmentDiagnostics"]
            proposal = diagnostic["oneStepSchurResidualProposal"]
            if proposal is None or len(proposal["attempts"]) != 8:
                raise ValueError(f"proposal telemetry mismatch for {scene}")
            if proposal.get("minimumRelativeDecrease") != 1e-3:
                raise ValueError(f"proposal progress floor mismatch for {scene}")
            selected_scale = float(proposal["selectedScale"])
            selected = bool(proposal["selected"])
            if selected != (selected_scale > 0.0):
                raise ValueError(f"selection mismatch for {scene}")
            if proposal["candidateWorkerSSE"] > proposal["ordinaryWorkerSSE"]:
                raise ValueError(f"proposal selected an inferior worker SSE for {scene}")
            control_delivered = min(
                row["sumSquaredError"] for row in control["trajectory"][:120]
            )
            row = {
                "scene": scene,
                "family": family,
                "selected": selected,
                "selected_scale": selected_scale,
                "worker_sse_ratio": (
                    proposal["candidateWorkerSSE"] / proposal["ordinaryWorkerSSE"]
                ),
                "immediate_i90_ratio": (
                    candidate["trajectory"][89]["sumSquaredError"]
                    / control["trajectory"][89]["sumSquaredError"]
                ),
                "trajectory_i120_ratio": (
                    candidate["trajectory"][119]["sumSquaredError"]
                    / control["trajectory"][119]["sumSquaredError"]
                ),
                "candidate_delivered": candidate["qualityMetrics"]["sumSquaredError"],
                "control_delivered": control_delivered,
                "delivered_ratio": (
                    candidate["qualityMetrics"]["sumSquaredError"]
                    / control_delivered
                ),
                "control_rejections": control["trajectory"][119]["rejections"],
                "candidate_rejections": candidate["rejections"],
                "coordinator_max_rss_kb": int(status["coordinator_max_rss_kb"]),
                "worker_max_rss_kb": int(status["worker_max_rss_kb"]),
                "elapsed_seconds": float(status["elapsed_seconds"]),
            }
            details[scene] = row
            family_rows.append(row)
        summaries[family] = summarize(family_rows, ceres)
    passed = all(
        row["geometric_delivered_ratio"] <= 1.0
        and row["losses"] == 0
        for row in summaries.values()
    )
    return {
        "status": "passed" if passed else "failed",
        "families": list(families),
        "summaries": summaries,
        "scenes": details,
    }


def write_report(path, summary):
    with path.open("w", encoding="utf-8") as output:
        output.write("# K24 I90 One-Step Schur Proposal Breadth\n\n")
        output.write(
            "One global I90 proposal, eight fixed geometric scales, precise "
            "worker-SSE selection, atomic DRS state rebuild, and I120 delivery. "
            "The rejected coupled-consensus oracle is disabled.\n\n"
        )
        output.write("| Family | Completed | Selected/declined | Immediate I90 | Trajectory I120 | Delivered/control | Summed | W/T/L | Candidate/Ceres | Max RSS GiB C/W |\n")
        output.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for family in summary["families"]:
            row = summary["summaries"][family]
            output.write(
                f"| {family} | {row['count']}/{EXPECTED[family]} | "
                f"{row['selected']}/{row['declined']} | "
                f"{row['geometric_immediate_i90_ratio']:.9f} | "
                f"{row['geometric_trajectory_i120_ratio']:.9f} | "
                f"{row['geometric_delivered_ratio']:.9f} | "
                f"{row['summed_delivered_ratio']:.9f} | "
                f"{row['wins']}/{row['ties']}/{row['losses']} | "
                f"{row['geometric_candidate_over_ceres']:.9f} | "
                f"{row['maximum_coordinator_rss_gib']:.3f}/"
                f"{row['maximum_worker_rss_gib']:.3f} |\n"
            )
        output.write("\n| Scene | Scale | Worker SSE | I90 | I120 trajectory | Delivered | Rejections C/P |\n")
        output.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for scene, row in sorted(summary["scenes"].items()):
            scale = f"{row['selected_scale']:.6g}" if row["selected"] else "declined"
            output.write(
                f"| {scene} | {scale} | {row['worker_sse_ratio']:.9f} | "
                f"{row['immediate_i90_ratio']:.9f} | "
                f"{row['trajectory_i120_ratio']:.9f} | "
                f"{row['delivered_ratio']:.9f} | "
                f"{row['control_rejections']}/{row['candidate_rejections']} |\n"
            )
        output.write(f"\nGate status: **{summary['status']}**.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument(
        "--families", nargs="+", choices=tuple(EXPECTED), default=tuple(EXPECTED)
    )
    arguments = parser.parse_args()
    summary = analyze(arguments.root, arguments.control_root, arguments.families)
    arguments.root.mkdir(parents=True, exist_ok=True)
    (arguments.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(arguments.root / "report.md", summary)
    print(json.dumps({
        "status": summary["status"],
        "summaries": summary["summaries"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
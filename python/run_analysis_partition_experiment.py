#!/usr/bin/env python3
"""Compare matched overlap and analysis-guided two-part refinements."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np

import palm_ba
from palm_partition_portfolio import checkpoint_is_reusable, make_candidate
from run_partition_portfolio_benchmark import SOLVER_ARGUMENTS, read_records


EPOCHS = 90
PARTITIONS = 2
SWAP_CANDIDATES = 256


@dataclass(frozen=True)
class Case:
    problem: int
    file_name: str
    seed: int | None
    refinement_passes: int


CASES = (
    Case(52, "problem-52-64053-pre.txt.bz2", 1, 5),
    Case(126, "problem-126-40037-pre.txt.bz2", None, 1),
    Case(135, "problem-135-90642-pre.txt.bz2", None, 5),
    Case(142, "problem-142-93602-pre.txt.bz2", 0, 5),
    Case(173, "problem-173-111908-pre.txt.bz2", None, 1),
    Case(253, "problem-253-163691-pre.txt.bz2", 1, 5),
    Case(356, "problem-356-226730-pre.txt.bz2", 1, 5),
    Case(394, "problem-394-100368-pre.txt.bz2", None, 5),
    Case(427, "problem-427-310384-pre.txt.bz2", None, 1),
    Case(646, "problem-646-73584-pre.txt.bz2", 1, 1),
    Case(931, "problem-931-102699-pre.txt.bz2", 1, 5),
)


def solver_seconds(records: list[dict]) -> float:
    elapsed = [float(record["elapsed_seconds"]) for record in records]
    return sum(
        elapsed[index - 1]
        for index in range(1, len(elapsed))
        if elapsed[index] < elapsed[index - 1]
    ) + elapsed[-1]


def static_metrics(problem: palm_ba.BALProblem, partitioner: str,
                   seed: int | None,
                   refinement_passes: int) -> dict[str, float | int]:
    candidate = make_candidate(
        problem, PARTITIONS, seed, refinement_passes,
        SWAP_CANDIDATES, partitioner)
    cut = (
        candidate.camera_owner[problem.camera_indices]
        != candidate.point_owner[problem.point_indices])
    shared_points = np.unique(problem.point_indices[cut])
    point_degree = np.bincount(
        problem.point_indices, minlength=len(problem.points))
    return {
        "cut_observations": candidate.cut_observations,
        "shared_point_count": candidate.shared_point_count,
        "shared_inverse_degree": float(np.sum(
            1.0 / point_degree[shared_points])),
        "included_load_ratio": candidate.included_load_ratio,
        "camera_load_ratio": candidate.camera_load_ratio,
        "max_load_ratio": candidate.max_load_ratio,
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    python = Path("/home/vogechri/bae/.venv/bin/python")
    output_dir = script_dir / "palm_runs" / "analysis_partition_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for case in CASES:
        problem = palm_ba.read_bal(script_dir / case.file_name)
        for partitioner in ("overlap", "analysis"):
            run_dir = output_dir / f"problem{case.problem}" / partitioner
            run_dir.mkdir(parents=True, exist_ok=True)
            trajectory = run_dir / "trajectory.jsonl"
            checkpoint = run_dir / "checkpoint.npz"
            records = read_records(trajectory) if trajectory.exists() else []
            if len(records) != EPOCHS:
                resume = (
                    0 < len(records) < EPOCHS
                    and checkpoint_is_reusable(checkpoint, len(records)))
                if not resume:
                    trajectory.unlink(missing_ok=True)
                    checkpoint.unlink(missing_ok=True)
                    records = []
                command = [
                    str(python), "-u", str(script_dir / "palm_ba.py"),
                    ".", case.file_name, str(EPOCHS), str(PARTITIONS),
                    *SOLVER_ARGUMENTS,
                    "--partitioner", partitioner,
                    "--partition-refinement-passes",
                    str(case.refinement_passes),
                    "--partition-balance-slack", "0",
                    "--partition-swap-candidates", str(SWAP_CANDIDATES),
                    "--repartition-every", "0",
                    "--output", str(trajectory),
                    "--checkpoint-output", str(checkpoint),
                ]
                if case.seed is not None:
                    command.extend([
                        "--initial-partition-seed", str(case.seed)])
                if resume:
                    command.extend(["--resume-checkpoint", str(checkpoint)])
                with (run_dir / "run.log").open(
                        "a" if resume else "w", encoding="utf-8") as log:
                    subprocess.run(
                        command, cwd=script_dir, stdout=log,
                        stderr=subprocess.STDOUT, check=True)
                records = read_records(trajectory)
            if len(records) != EPOCHS:
                raise RuntimeError(f"incomplete trajectory: {trajectory}")
            row = {
                "problem": case.problem,
                "partitioner": partitioner,
                "seed": case.seed,
                "refinement_passes": case.refinement_passes,
                "epochs": EPOCHS,
                "best_cost_20": min(
                    record["best_cost"] for record in records[:20]),
                "best_cost": min(record["best_cost"] for record in records),
                "solver_seconds": solver_seconds(records),
                **static_metrics(
                    problem, partitioner, case.seed,
                    case.refinement_passes),
                "trajectory": str(trajectory.relative_to(script_dir)),
            }
            results.append(row)
            print(
                f"problem {case.problem} {partitioner}: "
                f"best={row['best_cost']:.9e}, "
                f"shared={row['shared_point_count']}", flush=True)

    by_case = {(row["problem"], row["partitioner"]): row for row in results}
    comparisons = []
    for case in CASES:
        overlap = by_case[case.problem, "overlap"]
        analysis = by_case[case.problem, "analysis"]
        comparisons.append({
            "problem": case.problem,
            "relative_cost_20_percent": 100.0 * (
                analysis["best_cost_20"] / overlap["best_cost_20"] - 1.0),
            "relative_cost_percent": 100.0 * (
                analysis["best_cost"] / overlap["best_cost"] - 1.0),
            "relative_solver_time_percent": 100.0 * (
                analysis["solver_seconds"] / overlap["solver_seconds"] - 1.0),
            "shared_point_change": (
                analysis["shared_point_count"] - overlap["shared_point_count"]),
            "shared_inverse_degree_percent": 100.0 * (
                analysis["shared_inverse_degree"]
                / overlap["shared_inverse_degree"] - 1.0),
        })
    threshold_regrets = []
    for case in CASES:
        overlap = by_case[case.problem, "overlap"]
        analysis = by_case[case.problem, "analysis"]
        analysis_advantage = 100.0 * (
            1.0 - analysis["best_cost_20"] / overlap["best_cost_20"])
        selected = analysis if analysis_advantage > 0.5 else overlap
        oracle = min((overlap, analysis), key=lambda row: row["best_cost"])
        threshold_regrets.append(
            100.0 * (selected["best_cost"] / oracle["best_cost"] - 1.0))
    summary = {
        "epochs": EPOCHS,
        "partitions": PARTITIONS,
        "results": results,
        "comparisons": comparisons,
        "aggregate": {
            "problems": len(comparisons),
            "analysis_wins_epoch_20": sum(
                row["relative_cost_20_percent"] < 0.0
                for row in comparisons),
            "analysis_wins_epoch_90": sum(
                row["relative_cost_percent"] < 0.0
                for row in comparisons),
            "pilot_20_agrees_with_epoch_90": sum(
                (row["relative_cost_20_percent"] < 0.0)
                == (row["relative_cost_percent"] < 0.0)
                for row in comparisons),
            "mean_relative_cost_percent": statistics.mean(
                row["relative_cost_percent"] for row in comparisons),
            "median_relative_cost_percent": statistics.median(
                row["relative_cost_percent"] for row in comparisons),
            "mean_relative_solver_time_percent": statistics.mean(
                row["relative_solver_time_percent"] for row in comparisons),
            "mean_shared_inverse_degree_percent": statistics.mean(
                row["shared_inverse_degree_percent"] for row in comparisons),
            "threshold_0_5_percent": {
                "analysis_selected": sum(
                    row["relative_cost_20_percent"] < -0.5
                    for row in comparisons),
                "epoch_90_oracle_hits": sum(
                    (row["relative_cost_20_percent"] < -0.5)
                    == (row["relative_cost_percent"] < 0.0)
                    for row in comparisons),
                "mean_regret_percent": statistics.mean(threshold_regrets),
                "max_regret_percent": max(threshold_regrets),
            },
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary: {summary_path.relative_to(script_dir)}")


if __name__ == "__main__":
    main()

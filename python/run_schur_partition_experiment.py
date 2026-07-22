#!/usr/bin/env python3
"""Compare matched overlap and Schur-weighted two-part refinements."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess

from palm_partition_portfolio import checkpoint_is_reusable
from run_partition_portfolio_benchmark import SOLVER_ARGUMENTS, read_records


EPOCHS = 90
PARTITIONS = 2
SWAP_CANDIDATES = 256


@dataclass(frozen=True)
class Case:
    problem: int
    file_name: str
    seed: int
    refinement_passes: int


CASES = (
    Case(52, "problem-52-64053-pre.txt.bz2", 1, 1),
    Case(126, "problem-126-40037-pre.txt.bz2", 0, 5),
    Case(356, "problem-356-226730-pre.txt.bz2", 1, 5),
)


def solver_seconds(records: list[dict]) -> float:
    elapsed = [float(record["elapsed_seconds"]) for record in records]
    return sum(
        elapsed[index - 1]
        for index in range(1, len(elapsed))
        if elapsed[index] < elapsed[index - 1]
    ) + elapsed[-1]


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    python = Path("/home/vogechri/bae/.venv/bin/python")
    output_dir = script_dir / "palm_runs" / "schur_partition_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for case in CASES:
        for partitioner in ("overlap", "schur"):
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
                    "--initial-partition-seed", str(case.seed),
                    "--partition-refinement-passes",
                    str(case.refinement_passes),
                    "--partition-balance-slack", "0",
                    "--partition-swap-candidates", str(SWAP_CANDIDATES),
                    "--repartition-every", "0",
                    "--output", str(trajectory),
                    "--checkpoint-output", str(checkpoint),
                ]
                if resume:
                    command.extend(["--resume-checkpoint", str(checkpoint)])
                log_mode = "a" if resume else "w"
                with (run_dir / "run.log").open(
                        log_mode, encoding="utf-8") as log:
                    subprocess.run(
                        command, cwd=script_dir, stdout=log,
                        stderr=subprocess.STDOUT, check=True)
                records = read_records(trajectory)
            if len(records) != EPOCHS:
                raise RuntimeError(f"incomplete trajectory: {trajectory}")
            first = records[0]
            results.append({
                "problem": case.problem,
                "partitioner": partitioner,
                "seed": case.seed,
                "refinement_passes": case.refinement_passes,
                "best_cost": min(record["best_cost"] for record in records),
                "solver_seconds": solver_seconds(records),
                "cut_observations": first["partition_cut_observations"],
                "duplication_factor": first["partition_duplication_factor"],
                "max_mean_load_ratio": first["partition_max_mean_load_ratio"],
                "trajectory": str(trajectory.relative_to(script_dir)),
            })
            print(
                f"problem {case.problem} {partitioner}: "
                f"best={results[-1]['best_cost']:.9e}", flush=True)
    by_case = {(row["problem"], row["partitioner"]): row for row in results}
    comparisons = []
    for case in CASES:
        overlap = by_case[case.problem, "overlap"]
        schur = by_case[case.problem, "schur"]
        comparisons.append({
            "problem": case.problem,
            "relative_cost_percent": 100.0 * (
                schur["best_cost"] / overlap["best_cost"] - 1.0),
            "relative_solver_time_percent": 100.0 * (
                schur["solver_seconds"] / overlap["solver_seconds"] - 1.0),
        })
    summary = {
        "epochs": EPOCHS,
        "partitions": PARTITIONS,
        "results": results,
        "comparisons": comparisons,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary: {summary_path.relative_to(script_dir)}")


if __name__ == "__main__":
    main()
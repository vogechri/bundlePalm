#!/usr/bin/env python3
"""Compare frozen partition portfolios and fixed baselines at 5 and 10 parts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess

from run_partition_portfolio_benchmark import (
    EPOCHS,
    PILOT_EPOCHS,
    PROBLEMS,
    SOLVER_ARGUMENTS,
    Problem,
    read_records,
)


PARTITION_COUNTS = (5, 10)
LOAD_CAPS = {5: 1.05, 10: 1.15}
FIXED_REFINEMENT_PASSES = 20
SWAP_CANDIDATES = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("palm_runs/portfolio_partition_counts"),
    )
    parser.add_argument(
        "--partitions", default="5,10",
        help="comma-separated partition counts selected from 5,10",
    )
    parser.add_argument(
        "--problems",
        help="comma-separated problem numbers; default is the frozen set",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        counts = tuple(int(value) for value in args.partitions.split(","))
    except ValueError as error:
        parser.error(f"invalid partition counts: {error}")
    if not counts or not set(counts) <= set(PARTITION_COUNTS):
        parser.error(f"partitions must be selected from {PARTITION_COUNTS}")
    args.partition_counts = tuple(dict.fromkeys(counts))

    if args.problems:
        try:
            requested = {int(value) for value in args.problems.split(",")}
        except ValueError as error:
            parser.error(f"invalid problem list: {error}")
        known = {problem.number for problem in PROBLEMS}
        if not requested or not requested <= known:
            parser.error(f"problems must be selected from {sorted(known)}")
        args.selected_problems = tuple(
            problem for problem in PROBLEMS if problem.number in requested
        )
    else:
        args.selected_problems = PROBLEMS
    return args


def validate_trajectory(
        path: Path, problem: Problem, partitions: int,
        refinement_passes: int | None = None) -> list[dict]:
    records = read_records(path)
    if len(records) != EPOCHS or records[-1]["epoch"] != EPOCHS - 1:
        raise RuntimeError(
            f"incomplete trajectory {path}: found {len(records)} epochs"
        )
    for record in (records[0], records[-1]):
        expected = {
            "file_name": problem.file_name,
            "partitions": partitions,
            "execution": "sequential",
            "local_solver": "bae",
            "accelerator": "nesterov",
            "momentum_schedule": "palm",
            "global_jacobi": "none",
            "gpu_state_cache": True,
            "block_overrelaxation": 1.2,
            "block_safeguard": False,
        }
        if refinement_passes is not None:
            expected.update({
                "partitioner": "overlap",
                "partition_refinement_passes": refinement_passes,
                "partition_balance_slack": 0.0,
                "partition_swap_candidates": SWAP_CANDIDATES,
                "initial_partition_seed": None,
                "repartition_every": 0,
            })
        for key, value in expected.items():
            if record.get(key) != value:
                raise RuntimeError(
                    f"metadata mismatch in {path}: {key} != {value!r}"
                )
    return records


def validate_portfolio(
        summary_path: Path, problem: Problem, partitions: int) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = (EPOCHS, PILOT_EPOCHS, partitions, LOAD_CAPS[partitions])
    actual = (
        summary.get("epochs"), summary.get("pilot_epochs"),
        summary.get("partitions"), summary.get("max_load_ratio"),
    )
    candidate_count = len(summary.get("candidates", []))
    if actual != expected or not 1 <= candidate_count <= 4:
        raise RuntimeError(
            f"unexpected portfolio configuration: {summary_path}: "
            f"{actual}, candidates={candidate_count}"
        )
    for candidate in summary["candidates"]:
        records = read_records(Path(candidate["trajectory"]))
        expected_length = (
            EPOCHS if candidate["name"] == summary["selected"]
            else PILOT_EPOCHS
        )
        if (len(records) != expected_length
                or records[-1]["epoch"] != expected_length - 1):
            raise RuntimeError(
                f"incomplete candidate {candidate['name']} in {summary_path}"
            )
    return summary


def portfolio_command(
        python: Path, script_dir: Path, output_dir: Path,
        problem: Problem, partitions: int) -> list[str]:
    return [
        str(python), "-u", str(script_dir / "palm_partition_portfolio.py"),
        problem.base_url, problem.file_name, str(EPOCHS), str(partitions),
        "--pilot-epochs", str(PILOT_EPOCHS),
        "--max-candidates", "4",
        "--max-load-ratio", str(LOAD_CAPS[partitions]),
        "--output-dir", str(output_dir),
        "--", *SOLVER_ARGUMENTS,
    ]


def fixed_command(
        python: Path, script_dir: Path, trajectory: Path,
        checkpoint: Path, problem: Problem, partitions: int) -> list[str]:
    return [
        str(python), "-u", str(script_dir / "palm_ba.py"),
        problem.base_url, problem.file_name, str(EPOCHS), str(partitions),
        *SOLVER_ARGUMENTS,
        "--partitioner", "overlap",
        "--partition-refinement-passes", str(FIXED_REFINEMENT_PASSES),
        "--partition-balance-slack", "0",
        "--partition-swap-candidates", str(SWAP_CANDIDATES),
        "--repartition-every", "0",
        "--output", str(trajectory),
        "--checkpoint-output", str(checkpoint),
    ]


def cached_five_part_costs(script_dir: Path) -> dict[int, float]:
    path = (
        script_dir / "palm_runs" / "CachedVersionNesterov_12_parts5"
        / "results_palm.json"
    )
    by_file = {
        record["file_name"]: float(record["bestCost"])
        for record in read_records(path)
    }
    costs = {
        problem.number: by_file[problem.file_name] for problem in PROBLEMS
    }
    return costs


def run_logged(command: list[str], log_path: Path, dry_run: bool) -> None:
    print("  " + shlex.join(command), flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=True
        )


def write_summary(
        output_dir: Path, script_dir: Path,
        problems: tuple[Problem, ...]) -> None:
    cached_five = cached_five_part_costs(script_dir)
    results = []
    for partitions in PARTITION_COUNTS:
        for problem in problems:
            portfolio_summary_path = (
                output_dir / f"parts{partitions}"
                / f"problem{problem.number}" / "summary.json"
            )
            if not portfolio_summary_path.exists():
                continue
            portfolio = validate_portfolio(
                portfolio_summary_path, problem, partitions
            )
            if partitions == 5:
                fixed_cost = cached_five[problem.number]
                fixed_source = "CachedVersionNesterov_12_parts5"
            else:
                fixed_path = (
                    output_dir / "fixed_parts10"
                    / f"problem{problem.number}" / "trajectory.jsonl"
                )
                if not fixed_path.exists():
                    continue
                fixed_records = validate_trajectory(
                    fixed_path, problem, partitions,
                    FIXED_REFINEMENT_PASSES,
                )
                fixed_cost = min(
                    record["best_cost"] for record in fixed_records
                )
                fixed_source = str(fixed_path)
            portfolio_cost = float(portfolio["final_best_cost"])
            results.append({
                "problem": problem.number,
                "partitions": partitions,
                "selected": portfolio["selected"],
                "candidate_count": len(portfolio["candidates"]),
                "load_cap": LOAD_CAPS[partitions],
                "portfolio_best_cost": portfolio_cost,
                "fixed_best_cost": fixed_cost,
                "relative_to_fixed_percent": 100.0 * (
                    portfolio_cost / fixed_cost - 1.0
                ),
                "fixed_source": fixed_source,
            })
    aggregate = {}
    for partitions in PARTITION_COUNTS:
        rows = [
            result for result in results
            if result["partitions"] == partitions
        ]
        if not rows:
            continue
        ratios = [
            row["portfolio_best_cost"] / row["fixed_best_cost"]
            for row in rows
        ]
        deltas = [row["relative_to_fixed_percent"] for row in rows]
        aggregate[str(partitions)] = {
            "problems": len(rows),
            "portfolio_wins": sum(delta < 0.0 for delta in deltas),
            "fixed_wins": sum(delta > 0.0 for delta in deltas),
            "geometric_mean_difference_percent": 100.0 * (
                math.prod(ratios) ** (1.0 / len(ratios)) - 1.0
            ),
            "median_difference_percent": statistics.median(deltas),
        }
    summary = {
        "policy": {
            "epochs": EPOCHS,
            "pilot_epochs": PILOT_EPOCHS,
            "candidate_seeds": ["load", 0, 1],
            "refinement_checkpoints": [0, 1, 5],
            "max_candidates": 4,
            "load_caps": LOAD_CAPS,
            "fixed_refinement_passes": FIXED_REFINEMENT_PASSES,
        },
        "aggregate": aggregate,
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    python = Path("/home/vogechri/bae/.venv/bin/python")
    if not python.is_file():
        raise RuntimeError(f"missing Python interpreter: {python}")

    for partitions in args.partition_counts:
        print(f"=== {partitions} partitions ===", flush=True)
        for index, problem in enumerate(args.selected_problems, start=1):
            print(
                f"[{index}/{len(args.selected_problems)}] "
                f"problem {problem.number}",
                flush=True,
            )
            portfolio_dir = (
                args.output_dir / f"parts{partitions}"
                / f"problem{problem.number}"
            )
            portfolio_summary = portfolio_dir / "summary.json"
            if portfolio_summary.exists():
                validate_portfolio(
                    portfolio_summary, problem, partitions
                )
                print("  portfolio complete", flush=True)
            else:
                run_logged(
                    portfolio_command(
                        python, script_dir, portfolio_dir,
                        problem, partitions,
                    ),
                    portfolio_dir / "portfolio.log",
                    args.dry_run,
                )
                if not args.dry_run:
                    validate_portfolio(
                        portfolio_summary, problem, partitions
                    )

            if partitions == 10:
                fixed_dir = (
                    args.output_dir / "fixed_parts10"
                    / f"problem{problem.number}"
                )
                trajectory = fixed_dir / "trajectory.jsonl"
                if trajectory.exists():
                    try:
                        records = validate_trajectory(
                            trajectory, problem, partitions,
                            FIXED_REFINEMENT_PASSES,
                        )
                    except RuntimeError:
                        if args.dry_run:
                            raise
                    else:
                        print(
                            "  fixed complete: "
                            f"{min(r['best_cost'] for r in records):.9e}",
                            flush=True,
                        )
                        if not args.dry_run:
                            write_summary(
                                args.output_dir, script_dir, PROBLEMS,
                            )
                        continue
                checkpoint = fixed_dir / "checkpoint.npz"
                trajectory.unlink(missing_ok=True)
                checkpoint.unlink(missing_ok=True)
                run_logged(
                    fixed_command(
                        python, script_dir, trajectory, checkpoint,
                        problem, partitions,
                    ),
                    fixed_dir / "run.log",
                    args.dry_run,
                )
                if not args.dry_run:
                    validate_trajectory(
                        trajectory, problem, partitions,
                        FIXED_REFINEMENT_PASSES,
                    )
            if not args.dry_run:
                write_summary(args.output_dir, script_dir, PROBLEMS)

    if not args.dry_run:
        write_summary(args.output_dir, script_dir, PROBLEMS)
        print(f"summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
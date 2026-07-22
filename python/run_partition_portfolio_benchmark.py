#!/usr/bin/env python3
"""Run the frozen two-part partition portfolio holdout benchmark."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import time


@dataclass(frozen=True)
class Problem:
    number: int
    collection: str
    file_name: str
    audit_all_candidates: bool = False

    @property
    def base_url(self) -> str:
        return (
            "http://grail.cs.washington.edu/projects/bal/data/"
            f"{self.collection}/"
        )


PROBLEMS = (
    Problem(126, "trafalgar", "problem-126-40037-pre.txt.bz2"),
    Problem(135, "dubrovnik", "problem-135-90642-pre.txt.bz2", True),
    Problem(173, "dubrovnik", "problem-173-111908-pre.txt.bz2"),
    Problem(253, "dubrovnik", "problem-253-163691-pre.txt.bz2"),
    Problem(356, "dubrovnik", "problem-356-226730-pre.txt.bz2", True),
    Problem(394, "final", "problem-394-100368-pre.txt.bz2"),
    Problem(427, "venice", "problem-427-310384-pre.txt.bz2", True),
    Problem(646, "ladybug", "problem-646-73584-pre.txt.bz2"),
    Problem(931, "ladybug", "problem-931-102699-pre.txt.bz2"),
    Problem(961, "final", "problem-961-187103-pre.txt.bz2"),
)

EPOCHS = 90
PILOT_EPOCHS = 30
PARTITIONS = 2
SWAP_CANDIDATES = 256
SOLVER_ARGUMENTS = (
    "--execution", "sequential",
    "--workers", "1",
    "--local-solver", "bae",
    "--accelerator", "nesterov",
    "--momentum-schedule", "palm",
    "--local-nfev", "2",
    "--runtime-weight", "0",
    "--partition-seed", "0",
    "--block-overrelaxation", "1.2",
    "--no-block-safeguard",
    "--global-jacobi", "none",
    "--gpu-state-cache",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("palm_runs/portfolio_frozen_holdouts"),
    )
    parser.add_argument(
        "--problems",
        help="comma-separated problem numbers; default is the frozen set",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
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


def read_records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_baselines(script_dir: Path) -> dict[tuple[int, int], float]:
    baselines: dict[tuple[int, int], float] = {}
    for partitions in (1, 3):
        path = (
            script_dir / "palm_runs"
            / f"CachedVersionNesterov_12_parts{partitions}"
            / "results_palm.json"
        )
        for record in read_records(path):
            for problem in PROBLEMS:
                if record["file_name"] == problem.file_name:
                    baselines[(problem.number, partitions)] = float(
                        record["bestCost"]
                    )
    missing = [
        (problem.number, partitions)
        for problem in PROBLEMS
        for partitions in (1, 3)
        if (problem.number, partitions) not in baselines
    ]
    if missing:
        raise RuntimeError(f"missing frozen baselines: {missing}")
    return baselines


def validate_portfolio(summary_path: Path) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("epochs"), summary.get("pilot_epochs"),
            summary.get("partitions")) != (EPOCHS, PILOT_EPOCHS, PARTITIONS):
        raise RuntimeError(f"unexpected portfolio configuration: {summary_path}")
    selected = summary["selected"]
    if len(summary["candidates"]) != 4:
        raise RuntimeError(f"expected four candidates: {summary_path}")
    for candidate in summary["candidates"]:
        records = read_records(Path(candidate["trajectory"]))
        allowed_lengths = (
            {EPOCHS}
            if candidate["name"] == selected
            else {PILOT_EPOCHS, EPOCHS}
        )
        if (len(records) not in allowed_lengths
                or records[-1]["epoch"] != len(records) - 1):
            raise RuntimeError(
                f"incomplete trajectory for {candidate['name']}: "
                f"expected {sorted(allowed_lengths)}, found {len(records)}"
            )
    return summary


def portfolio_command(
        python: Path, script_dir: Path, problem: Problem,
        output_dir: Path) -> list[str]:
    return [
        str(python), "-u", str(script_dir / "palm_partition_portfolio.py"),
        problem.base_url, problem.file_name, str(EPOCHS), str(PARTITIONS),
        "--pilot-epochs", str(PILOT_EPOCHS),
        "--max-candidates", "4",
        "--max-load-ratio", "1.05",
        "--output-dir", str(output_dir),
        "--", *SOLVER_ARGUMENTS,
    ]


def audit_command(
        python: Path, script_dir: Path, problem: Problem,
        candidate: dict) -> list[str]:
    trajectory = Path(candidate["trajectory"])
    checkpoint = trajectory.parent / "checkpoint.npz"
    command = [
        str(python), "-u", str(script_dir / "palm_ba.py"),
        problem.base_url, problem.file_name, str(EPOCHS), str(PARTITIONS),
        *SOLVER_ARGUMENTS,
        "--partitioner", "overlap",
        "--partition-refinement-passes",
        str(candidate["refinement_passes"]),
        "--partition-balance-slack", "0",
        "--partition-swap-candidates", str(SWAP_CANDIDATES),
        "--repartition-every", "0",
    ]
    if candidate["seed"] is not None:
        command.extend(["--initial-partition-seed", str(candidate["seed"])])
    command.extend([
        "--output", str(trajectory),
        "--resume-checkpoint", str(checkpoint),
        "--checkpoint-output", str(checkpoint),
    ])
    return command


def run_command(command: list[str], dry_run: bool) -> float:
    print(" ".join(command), flush=True)
    if dry_run:
        return 0.0
    started = time.monotonic()
    subprocess.run(command, check=True)
    return time.monotonic() - started


def write_aggregate(
        output_dir: Path, baselines: dict[tuple[int, int], float],
        selected_problems: tuple[Problem, ...]) -> None:
    results = []
    for problem in selected_problems:
        problem_dir = output_dir / f"problem{problem.number}_parts2"
        summary_path = problem_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = validate_portfolio(summary_path)
        candidate_finals = {}
        for candidate in summary["candidates"]:
            records = read_records(Path(candidate["trajectory"]))
            candidate_finals[candidate["name"]] = min(
                record["best_cost"] for record in records
            )
        final_cost = float(summary["final_best_cost"])
        fixed_trajectory = (
            output_dir / "fixed_two_part"
            / f"problem{problem.number}" / "trajectory.jsonl"
        )
        fixed_records = (
            read_records(fixed_trajectory)
            if fixed_trajectory.exists() else []
        )
        fixed_cost = (
            min(record["best_cost"] for record in fixed_records)
            if len(fixed_records) == EPOCHS
            and fixed_records[-1]["epoch"] == EPOCHS - 1
            else None
        )
        oracle_cost = (
            min(candidate_finals.values())
            if problem.audit_all_candidates
            and all(
                len(read_records(Path(candidate["trajectory"]))) == EPOCHS
                for candidate in summary["candidates"]
            )
            else None
        )
        results.append({
            "problem": problem.number,
            "file_name": problem.file_name,
            "selected": summary["selected"],
            "pilot_best_cost": summary["pilot_best_cost"],
            "final_best_cost": final_cost,
            "one_part_best_cost": baselines[(problem.number, 1)],
            "three_part_best_cost": baselines[(problem.number, 3)],
            "fixed_two_part_best_cost": fixed_cost,
            "relative_to_one_part_percent": 100.0 * (
                final_cost / baselines[(problem.number, 1)] - 1.0
            ),
            "relative_to_three_part_percent": 100.0 * (
                final_cost / baselines[(problem.number, 3)] - 1.0
            ),
            "relative_to_fixed_two_part_percent": (
                100.0 * (final_cost / fixed_cost - 1.0)
                if fixed_cost is not None else None
            ),
            "audited": oracle_cost is not None,
            "oracle_candidate_cost": oracle_cost,
            "selection_regret_percent": (
                100.0 * (final_cost / oracle_cost - 1.0)
                if oracle_cost is not None else None
            ),
            "candidate_final_costs": candidate_finals,
        })
    aggregate = {
        "policy": {
            "partitions": PARTITIONS,
            "epochs": EPOCHS,
            "pilot_epochs": PILOT_EPOCHS,
            "candidate_seeds": ["load", 0, 1],
            "refinement_checkpoints": [0, 1, 5],
            "max_candidates": 4,
            "max_load_ratio": 1.05,
            "audit_problems": [
                problem.number for problem in PROBLEMS
                if problem.audit_all_candidates
            ],
        },
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    python = Path("/home/vogechri/bae/.venv/bin/python")
    if not python.is_file():
        raise RuntimeError(f"missing Python interpreter: {python}")
    baselines = load_baselines(script_dir)

    for index, problem in enumerate(args.selected_problems, start=1):
        problem_dir = args.output_dir / f"problem{problem.number}_parts2"
        summary_path = problem_dir / "summary.json"
        print(
            f"[{index}/{len(args.selected_problems)}] problem {problem.number}",
            flush=True,
        )
        if summary_path.exists():
            summary = validate_portfolio(summary_path)
            print("  portfolio already complete", flush=True)
        else:
            run_command(
                portfolio_command(python, script_dir, problem, problem_dir),
                args.dry_run,
            )
            if args.dry_run:
                continue
            summary = validate_portfolio(summary_path)

        if problem.audit_all_candidates:
            for candidate in summary["candidates"]:
                trajectory = Path(candidate["trajectory"])
                records = read_records(trajectory)
                if len(records) == EPOCHS:
                    continue
                if len(records) != PILOT_EPOCHS:
                    raise RuntimeError(
                        f"cannot audit {trajectory}: found {len(records)} epochs"
                    )
                print(f"  auditing {candidate['name']} to epoch {EPOCHS}")
                run_command(
                    audit_command(python, script_dir, problem, candidate),
                    args.dry_run,
                )
        if not args.dry_run:
            write_aggregate(args.output_dir, baselines, args.selected_problems)

    if not args.dry_run:
        write_aggregate(args.output_dir, baselines, args.selected_problems)
        print(f"summary: {args.output_dir / 'benchmark_summary.json'}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Select a fixed PALM partition with short solver-informed pilot runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

import palm_ba


@dataclass(frozen=True)
class Candidate:
    partitioner: str
    seed: int | None
    refinement_passes: int
    camera_owner: np.ndarray
    point_owner: np.ndarray
    ownership_hash: str
    cut_observations: int
    shared_point_count: int
    shared_point_degree_mean: float
    included_load_ratio: float
    camera_load_ratio: float

    @property
    def name(self) -> str:
        seed_name = "load" if self.seed is None else str(self.seed)
        prefix = "" if self.partitioner == "overlap" else f"{self.partitioner}_"
        return f"{prefix}seed_{seed_name}_passes_{self.refinement_passes}"

    @property
    def max_load_ratio(self) -> float:
        return max(self.included_load_ratio, self.camera_load_ratio)


def parse_int_list(value: str, name: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{name} must be a comma-separated integer list") from error
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError(f"{name} must contain nonnegative values")
    return values


def parse_seeds(value: str) -> list[int | None]:
    seeds: list[int | None] = []
    for item in value.split(","):
        if item == "load":
            seeds.append(None)
        else:
            try:
                seed = int(item)
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    "candidate seeds must contain integers or 'load'") from error
            if seed < 0:
                raise argparse.ArgumentTypeError(
                    "candidate seeds must be nonnegative")
            seeds.append(seed)
    if not seeds:
        raise argparse.ArgumentTypeError("at least one candidate seed is required")
    return seeds


def parse_args() -> argparse.Namespace:
    command_line = sys.argv[1:]
    if "--" in command_line:
        separator = command_line.index("--")
        portfolio_arguments = command_line[:separator]
        solver_arguments = command_line[separator + 1:]
    else:
        portfolio_arguments = command_line
        solver_arguments = []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("file_name")
    parser.add_argument("epochs", type=int)
    parser.add_argument("partitions", type=int)
    parser.add_argument("--pilot-epochs", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--max-load-ratio", type=float, default=1.05)
    parser.add_argument(
        "--analysis-switch-threshold-percent", type=float, default=0.5)
    parser.add_argument("--candidate-seeds", default="load,0,1")
    parser.add_argument(
        "--candidate-partitioner", choices=("overlap", "analysis", "both"),
        default="both")
    parser.add_argument("--refinement-checkpoints", default="0,1,5")
    parser.add_argument("--partition-swap-candidates", type=int, default=256)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("palm_portfolio_results"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(portfolio_arguments)
    args.solver_args = solver_arguments
    args.candidate_seeds = parse_seeds(args.candidate_seeds)
    args.refinement_checkpoints = parse_int_list(
        args.refinement_checkpoints, "refinement checkpoints")
    if args.epochs < 1 or not 0 < args.pilot_epochs <= args.epochs:
        parser.error("pilot epochs must be between 1 and total epochs")
    if args.partitions < 1 or args.max_candidates < 1:
        parser.error("partitions and max candidates must be positive")
    if (args.max_load_ratio < 1.0 or args.partition_swap_candidates < 0
            or args.analysis_switch_threshold_percent < 0.0):
        parser.error("load ratio must be at least one and candidates nonnegative")
    reserved = {
        "--output", "--checkpoint-output", "--resume-checkpoint",
        "--partitioner", "--initial-partition-seed",
        "--partition-refinement-passes", "--partition-balance-slack",
        "--partition-swap-candidates", "--repartition-every",
    }
    conflicts = sorted(reserved.intersection(args.solver_args))
    if conflicts:
        parser.error(
            "portfolio controls these solver arguments: " + ", ".join(conflicts))
    return args


def normalize_owners(camera_owner: np.ndarray,
                     point_owner: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mapping: dict[int, int] = {}
    normalized_cameras = np.empty_like(camera_owner)
    next_label = 0
    for index, owner in enumerate(camera_owner):
        label = int(owner)
        if label not in mapping:
            mapping[label] = next_label
            next_label += 1
        normalized_cameras[index] = mapping[label]
    normalized_points = np.array(
        [mapping[int(owner)] for owner in point_owner], dtype=np.int64)
    return normalized_cameras, normalized_points


def make_candidate(problem: palm_ba.BALProblem, partitions: int,
                   seed: int | None, refinement_passes: int,
                   swap_candidates: int,
                   partitioner: str = "overlap") -> Candidate:
    camera_owner, point_owner = palm_ba._assign_partition_owners(
        problem, partitions, seed)
    camera_owner, point_owner = palm_ba._refine_partition_overlap(
        problem, camera_owner, point_owner, partitions, refinement_passes,
        0.0, swap_candidates)
    if partitioner == "analysis":
        camera_owner, point_owner = palm_ba._refine_partition_overlap(
            problem, camera_owner, point_owner, partitions, refinement_passes,
            0.0, swap_candidates, "analysis")
    normalized_cameras, normalized_points = normalize_owners(
        camera_owner, point_owner)
    digest = hashlib.sha256()
    digest.update(normalized_cameras.tobytes())
    digest.update(normalized_points.tobytes())

    camera_blocks = camera_owner[problem.camera_indices]
    point_blocks = point_owner[problem.point_indices]
    shared = camera_blocks != point_blocks
    shared_points = np.unique(problem.point_indices[shared])
    point_degree = np.bincount(
        problem.point_indices, minlength=len(problem.points))
    included_loads = np.array([
        np.count_nonzero((camera_blocks == block) | (point_blocks == block))
        for block in range(partitions)
    ])
    camera_loads = np.bincount(
        camera_blocks, minlength=partitions).astype(np.int64)
    return Candidate(
        partitioner=partitioner,
        seed=seed,
        refinement_passes=refinement_passes,
        camera_owner=normalized_cameras,
        point_owner=normalized_points,
        ownership_hash=digest.hexdigest()[:16],
        cut_observations=int(shared.sum()),
        shared_point_count=len(shared_points),
        shared_point_degree_mean=(
            float(point_degree[shared_points].mean())
            if len(shared_points) else 0.0),
        included_load_ratio=float(included_loads.max() / included_loads.mean()),
        camera_load_ratio=float(camera_loads.max() / camera_loads.mean()),
    )


def ownership_distance(left: Candidate, right: Candidate) -> float:
    partition_count = max(
        int(left.camera_owner.max()), int(right.camera_owner.max())) + 1
    agreement = np.zeros((partition_count, partition_count), dtype=np.int64)
    np.add.at(agreement, (left.camera_owner, right.camera_owner), 1)
    left_labels, right_labels = linear_sum_assignment(-agreement)
    matched = int(agreement[left_labels, right_labels].sum())
    return 1.0 - matched / len(left.camera_owner)


def select_candidates(candidates: list[Candidate], limit: int) -> list[Candidate]:
    if len(candidates) <= limit:
        return candidates
    shared_count_order = {
        candidate.ownership_hash: rank
        for rank, candidate in enumerate(sorted(candidates, key=lambda item: (
            item.shared_point_count, item.ownership_hash)))
    }
    shared_degree_order = {
        candidate.ownership_hash: rank
        for rank, candidate in enumerate(sorted(candidates, key=lambda item: (
            -item.shared_point_degree_mean, item.ownership_hash)))
    }
    selected = [min(candidates, key=lambda item: (
        shared_count_order[item.ownership_hash]
        + shared_degree_order[item.ownership_hash],
        item.shared_point_count, -item.shared_point_degree_mean,
        item.cut_observations, item.ownership_hash))]

    if len(selected) < limit:
        lowest_cut = min(candidates, key=lambda item: (
            item.cut_observations, item.max_load_ratio))
        if lowest_cut.ownership_hash not in {
                item.ownership_hash for item in selected}:
            selected.append(lowest_cut)

    while len(selected) < limit:
        selected_hashes = {item.ownership_hash for item in selected}
        remaining = [
            item for item in candidates
            if item.ownership_hash not in selected_hashes
        ]
        if not remaining:
            break
        selected.append(max(remaining, key=lambda item: (
            min(ownership_distance(item, chosen) for chosen in selected),
            -item.max_load_ratio,
            -item.cut_observations,
        )))
    return selected


def candidate_arguments(candidate: Candidate, swap_candidates: int) -> list[str]:
    arguments = [
        "--partitioner", candidate.partitioner,
        "--partition-refinement-passes", str(candidate.refinement_passes),
        "--partition-balance-slack", "0",
        "--partition-swap-candidates", str(swap_candidates),
        "--repartition-every", "0",
    ]
    if candidate.seed is not None:
        arguments += ["--initial-partition-seed", str(candidate.seed)]
    return arguments


def run_solver(command: list[str], log_path: Path, append: bool = False) -> None:
    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8") as log:
        if append:
            log.write("\n===== continuing selected pilot =====\n")
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def read_pilot_result(path: Path, epochs: int | None = None) -> tuple[float, float]:
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if epochs is not None:
        records = records[:epochs]
    tail = records[-min(6, len(records)):]
    return min(record["best_cost"] for record in records), float(np.mean([
        record["best_cost"] for record in tail]))


def select_pilot_winner(results: list[dict],
                        analysis_threshold_percent: float) -> dict:
    overlap = [
        result for result in results
        if result["candidate"].partitioner == "overlap"
    ]
    analysis = [
        result for result in results
        if result["candidate"].partitioner == "analysis"
    ]
    key = lambda result: (
        result["pilot_best_cost"], result["pilot_tail_cost"],
        result["candidate"].max_load_ratio)
    if not overlap or not analysis:
        return min(results, key=key)
    best_overlap = min(overlap, key=key)
    best_analysis = min(analysis, key=key)
    required_cost = best_overlap["pilot_best_cost"] * (
        1.0 - analysis_threshold_percent / 100.0)
    return (
        best_analysis
        if best_analysis["pilot_best_cost"] < required_cost
        else best_overlap)


def trajectory_length(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        bool(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def checkpoint_is_reusable(path: Path, completed_epochs: int) -> bool:
    if not path.exists():
        return False
    required = {
        "configuration", "completed_epochs", "current_cameras",
        "current_points", "best_cameras", "best_points",
        "previous_accepted", "block_damping",
    }
    try:
        with np.load(path, allow_pickle=False) as checkpoint:
            if not required <= set(checkpoint.files):
                return False
            if int(checkpoint["completed_epochs"].item()) != completed_epochs:
                return False
            for name in required - {"configuration", "completed_epochs"}:
                checkpoint[name]
    except (KeyError, OSError, ValueError):
        return False
    return True


def main() -> None:
    args = parse_args()
    script_directory = Path(__file__).resolve().parent
    problem_path = palm_ba.ensure_problem(
        args.base_url, args.file_name, script_directory)
    problem = palm_ba.read_bal(problem_path)

    unique: dict[str, Candidate] = {}
    candidate_partitioners = (
        ("overlap", "analysis")
        if args.candidate_partitioner == "both"
        else (args.candidate_partitioner,))
    for partitioner in candidate_partitioners:
        for seed in args.candidate_seeds:
            for passes in args.refinement_checkpoints:
                candidate = make_candidate(
                    problem, args.partitions, seed, passes,
                    args.partition_swap_candidates, partitioner)
                previous = unique.get(candidate.ownership_hash)
                if (previous is None
                        or candidate.max_load_ratio < previous.max_load_ratio):
                    unique[candidate.ownership_hash] = candidate

    all_candidates = list(unique.values())
    eligible = [
        candidate for candidate in all_candidates
        if candidate.max_load_ratio <= args.max_load_ratio
    ]
    if not eligible:
        best_ratio = min(candidate.max_load_ratio for candidate in all_candidates)
        eligible = [
            candidate for candidate in all_candidates
            if candidate.max_load_ratio == best_ratio
        ]
    candidates = select_candidates(eligible, args.max_candidates)

    print(f"generated {len(all_candidates)} unique partitions; "
          f"{len(eligible)} satisfy load cap; piloting {len(candidates)}")
    for candidate in candidates:
        print(f"  {candidate.name}: cuts={candidate.cut_observations}, "
              f"shared_points={candidate.shared_point_count}, "
              f"shared_degree={candidate.shared_point_degree_mean:.1f}, "
              f"included_load={candidate.included_load_ratio:.3f}, "
              f"camera_load={candidate.camera_load_ratio:.3f}, "
              f"hash={candidate.ownership_hash}")
    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    solver_path = script_directory / "palm_ba.py"
    results = []
    for candidate in candidates:
        candidate_dir = args.output_dir / candidate.name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        trajectory = candidate_dir / "trajectory.jsonl"
        checkpoint = candidate_dir / "checkpoint.npz"
        log_path = candidate_dir / "run.log"
        command = [
            sys.executable, "-u", str(solver_path), args.base_url,
            args.file_name, str(args.pilot_epochs), str(args.partitions),
            *args.solver_args,
            *candidate_arguments(candidate, args.partition_swap_candidates),
            "--output", str(trajectory),
            "--checkpoint-output", str(checkpoint),
        ]
        completed_epochs = trajectory_length(trajectory)
        if (completed_epochs >= args.pilot_epochs
            and checkpoint_is_reusable(checkpoint, completed_epochs)):
            print(f"reusing pilot {candidate.name} ({completed_epochs} epochs)")
        else:
            if trajectory.exists():
                trajectory.unlink()
            if checkpoint.exists():
                checkpoint.unlink()
            print(f"piloting {candidate.name}")
            run_solver(command, log_path)
        best_cost, tail_cost = read_pilot_result(
            trajectory, args.pilot_epochs)
        results.append({
            "candidate": candidate,
            "directory": candidate_dir,
            "trajectory": trajectory,
            "checkpoint": checkpoint,
            "log": log_path,
            "pilot_best_cost": best_cost,
            "pilot_tail_cost": tail_cost,
        })
        print(f"  pilot best={best_cost:.9e}, tail={tail_cost:.9e}")

    winner = select_pilot_winner(
        results, args.analysis_switch_threshold_percent)
    selected = winner["candidate"]
    print(f"selected {selected.name} at {winner['pilot_best_cost']:.9e}")

    if trajectory_length(winner["trajectory"]) < args.epochs:
        command = [
            sys.executable, "-u", str(solver_path), args.base_url,
            args.file_name, str(args.epochs), str(args.partitions),
            *args.solver_args,
            *candidate_arguments(selected, args.partition_swap_candidates),
            "--output", str(winner["trajectory"]),
            "--resume-checkpoint", str(winner["checkpoint"]),
            "--checkpoint-output", str(winner["checkpoint"]),
        ]
        print(f"continuing {selected.name} to epoch {args.epochs}")
        run_solver(command, winner["log"], append=True)

    final_best_cost, _ = read_pilot_result(winner["trajectory"])
    summary = {
        "file_name": args.file_name,
        "partitions": args.partitions,
        "pilot_epochs": args.pilot_epochs,
        "epochs": args.epochs,
        "max_load_ratio": args.max_load_ratio,
        "analysis_switch_threshold_percent": (
            args.analysis_switch_threshold_percent),
        "selected": selected.name,
        "selected_hash": selected.ownership_hash,
        "pilot_best_cost": winner["pilot_best_cost"],
        "final_best_cost": final_best_cost,
        "selected_trajectory": str(winner["trajectory"]),
        "selected_checkpoint": str(winner["checkpoint"]),
        "candidates": [
            {
                "name": result["candidate"].name,
                "hash": result["candidate"].ownership_hash,
                "partitioner": result["candidate"].partitioner,
                "seed": result["candidate"].seed,
                "refinement_passes": result["candidate"].refinement_passes,
                "cut_observations": result["candidate"].cut_observations,
                "shared_point_count": result["candidate"].shared_point_count,
                "shared_point_degree_mean": (
                    result["candidate"].shared_point_degree_mean),
                "included_load_ratio": result["candidate"].included_load_ratio,
                "camera_load_ratio": result["candidate"].camera_load_ratio,
                "pilot_best_cost": result["pilot_best_cost"],
                "pilot_tail_cost": result["pilot_tail_cost"],
                "trajectory": str(result["trajectory"]),
                "log": str(result["log"]),
            }
            for result in results
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"final best={final_best_cost:.9e}; summary={summary_path}")


if __name__ == "__main__":
    main()

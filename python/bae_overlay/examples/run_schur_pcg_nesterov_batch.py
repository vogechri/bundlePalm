import argparse
import json
import subprocess
import sys
from pathlib import Path


PROBLEMS = [
    ("final", "problem-394-100368-pre.txt.bz2"),
    ("ladybug", "problem-1064-113655-pre.txt.bz2"),
    ("venice", "problem-245-198739-pre.txt.bz2"),
    ("ladybug", "problem-1723-156502-pre.txt.bz2"),
    ("ladybug", "problem-1266-132593-pre.txt.bz2"),
    ("ladybug", "problem-931-102699-pre.txt.bz2"),
    ("ladybug", "problem-783-84444-pre.txt.bz2"),
    ("venice", "problem-89-110973-pre.txt.bz2"),
    ("dubrovnik", "problem-287-182023-pre.txt.bz2"),
    ("dubrovnik", "problem-142-93602-pre.txt.bz2"),
    ("ladybug", "problem-646-73584-pre.txt.bz2"),
    ("dubrovnik", "problem-135-90642-pre.txt.bz2"),
    ("venice", "problem-52-64053-pre.txt.bz2"),
    ("dubrovnik", "problem-173-111908-pre.txt.bz2"),
    ("dubrovnik", "problem-356-226730-pre.txt.bz2"),
    ("dubrovnik", "problem-88-64298-pre.txt.bz2"),
    ("ladybug", "problem-49-7776-pre.txt.bz2"),
    ("trafalgar", "problem-126-40037-pre.txt.bz2"),
    ("venice", "problem-427-310384-pre.txt.bz2"),
    ("dubrovnik", "problem-253-163691-pre.txt.bz2"),
    ("final", "problem-961-187103-pre.txt.bz2"),
    ("trafalgar", "problem-257-65132-pre.txt.bz2"),
    ("venice", "problem-744-543562-pre.txt.bz2"),
    ("venice", "problem-951-708276-pre.txt.bz2"),
    ("dubrovnik", "problem-308-195089-pre.txt.bz2"),
    ("final", "problem-871-527480-pre.txt.bz2"),
    ("venice", "problem-1778-993923-pre.txt.bz2"),
    ("venice", "problem-1490-935273-pre.txt.bz2"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the requested BAL Schur benchmark batch.")
    parser.add_argument("--solver", choices=("nesterov", "cg"), default="nesterov")
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / "bundlePalm/python")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def completed_files(summary_path):
    if not summary_path.exists():
        return set()
    completed = set()
    with summary_path.open() as summary_file:
        for line in summary_file:
            result = json.loads(line)
            if result.get("status") == "completed":
                completed.add(result["file_name"])
    return completed


def main():
    args = parse_args()
    algorithm = f"schur_pcg_{args.solver}"
    script = Path(__file__).with_name(f"{algorithm}.py")
    if args.output_dir is None:
        args.output_dir = Path(f"{algorithm}_runs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = args.output_dir / "trajectories"
    logs = args.output_dir / "logs"
    trajectories.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    summary_path = args.output_dir / "results.jsonl"
    completed = set() if args.rerun else completed_files(summary_path)

    for run_number, (dataset, file_name) in enumerate(PROBLEMS, start=1):
        if file_name in completed:
            print(f"[{run_number}/{len(PROBLEMS)}] skipping completed {file_name}", flush=True)
            continue

        problem = file_name.removesuffix(".txt.bz2")
        problem_id = problem.split("-")[1]
        trajectory_path = trajectories / f"{algorithm}{problem_id}.jsonl"
        log_path = logs / f"{algorithm}{problem_id}.log"
        base_url = f"http://grail.cs.washington.edu/projects/bal/data/{dataset}/"
        command = [
            sys.executable,
            str(script),
            dataset,
            problem,
            "--iterations", str(args.iterations),
            "--cache-dir", str(args.cache_dir),
            "--base-url", base_url,
            "--file-name", file_name,
            "--trajectory", str(trajectory_path),
            "--summary", str(summary_path),
        ]
        print(f"[{run_number}/{len(PROBLEMS)}] running {file_name}", flush=True)
        with log_path.open("w") as log_file:
            result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)

        if result.returncode != 0:
            failure = {
                "algorithm": algorithm,
                "base_url": base_url,
                "file_name": file_name,
                "iterations": args.iterations,
                "status": "failed",
                "epochsCompleted": sum(1 for _ in trajectory_path.open()) if trajectory_path.exists() else 0,
                "trajectory": str(trajectory_path.resolve()),
                "log": str(log_path.resolve()),
            }
            with summary_path.open("a") as summary_file:
                summary_file.write(json.dumps(failure, separators=(",", ":")) + "\n")
            print(f"  failed; see {log_path}", flush=True)


if __name__ == "__main__":
    main()
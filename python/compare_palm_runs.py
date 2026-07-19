#!/usr/bin/env python3
"""Compare objective checkpoints from two PALM JSON Lines summaries."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


def read_results(path: Path) -> dict[str, dict]:
    results = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                results[record["file_name"]] = record
    return results


def problem_number(file_name: str) -> int:
    match = re.search(r"problem-(\d+)-", file_name)
    return int(match.group(1)) if match else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    args = parser.parse_args()

    baseline = read_results(args.baseline)
    candidate = read_results(args.candidate)
    shared = sorted(baseline.keys() & candidate.keys(), key=problem_number)
    if not shared:
        raise SystemExit("the result files have no shared problems")

    print(f"shared problems: {len(shared)}")
    print(f"{'problem':>7} {'baseline':>12} {'candidate':>12} {'difference':>11}")
    ratios = []
    for file_name in shared:
        baseline_cost = baseline[file_name]["bestCost"]
        candidate_cost = candidate[file_name]["bestCost"]
        ratio = candidate_cost / baseline_cost
        ratios.append(ratio)
        print(f"{problem_number(file_name):7d} {baseline_cost:12d} "
              f"{candidate_cost:12d} {(ratio - 1.0) * 100:+10.2f}%")

    geometric_ratio = math.exp(statistics.fmean(math.log(ratio)
                                                for ratio in ratios))
    print()
    print(f"{args.candidate_label} wins: {sum(ratio < 1.0 for ratio in ratios)}")
    print(f"{args.baseline_label} wins: {sum(ratio > 1.0 for ratio in ratios)}")
    print(f"median difference: {(statistics.median(ratios) - 1.0) * 100:+.3f}%")
    print(f"geometric mean difference: {(geometric_ratio - 1.0) * 100:+.3f}%")


if __name__ == "__main__":
    main()
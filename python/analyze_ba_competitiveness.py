#!/usr/bin/env python3
"""Build a caveated DRS-versus-Schur pilot report from native summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def read_labeled_section(path: Path, label: str) -> list[dict[str, Any]]:
    sections: list[list[dict[str, Any]]] = []
    records: list[dict[str, Any]] | None = None
    active = False
    with path.open() as source:
        for line in source:
            stripped = line.strip()
            if stripped.startswith("#"):
                if stripped == label:
                    records = []
                    sections.append(records)
                    active = True
                else:
                    active = False
                continue
            if active and stripped and records is not None:
                records.append(json.loads(stripped))
    if not sections or not sections[-1]:
        raise ValueError(f"section {label!r} not found or empty in {path}")
    return sections[-1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def _value(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = record.get(name)
        if value is not None:
            return value
    return None


def compare_records(
    drs_records: Iterable[dict[str, Any]],
    schur_records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    drs_by_problem = {record["file_name"]: record for record in drs_records}
    comparisons = []
    for schur in schur_records:
        drs = drs_by_problem.get(schur.get("file_name"))
        if drs is None or schur.get("status") != "completed":
            continue

        drs_cost = _value(drs, "bestCost")
        schur_cost = _value(schur, "bestCost")
        drs_time = _value(drs, "elapsedSeconds", "overallSeconds")
        schur_time = _value(schur, "elapsedSeconds", "overallSeconds")
        if None in (drs_cost, schur_cost, drs_time, schur_time):
            continue

        drs_better_cost = drs_cost <= schur_cost
        drs_faster = drs_time <= schur_time
        drs_partition_time = _value(drs, "partitionSeconds") or 0.0
        drs_solve_time = max(drs_time - drs_partition_time, 0.0)
        if drs_better_cost and drs_faster:
            pareto = "DRS dominates"
        elif not drs_better_cost and not drs_faster:
            pareto = "Schur dominates"
        else:
            pareto = "tradeoff"

        warnings = []
        if not drs.get("objectiveVerified", False) or not schur.get(
            "objectiveVerified", False
        ):
            warnings.append("objective unverified")
        if schur.get("crossTermsVerified") is False:
            warnings.append("Schur cross terms unverified")
        if drs.get("iterations") != schur.get("iterations"):
            warnings.append("iteration budgets differ")
        drs_partitions = _value(drs, "kClusters", "partitions")
        schur_partitions = _value(
            schur, "worldSize", "edgePartitions", "partitions"
        )
        if drs_partitions != schur_partitions:
            warnings.append("partition counts differ")

        comparisons.append(
            {
                "problem": schur["file_name"],
                "schurAlgorithm": schur.get("algorithm", "unknown"),
                "drsPartitions": drs_partitions,
                "schurPartitions": schur_partitions,
                "drsIterations": drs.get("iterations"),
                "schurIterations": schur.get("iterations"),
                "drsCost": drs_cost,
                "schurCost": schur_cost,
                "drsCostGapPercent": 100.0 * (drs_cost - schur_cost) / schur_cost,
                "drsSeconds": drs_time,
                "drsPartitionSeconds": drs_partition_time,
                "drsSolveSeconds": drs_solve_time,
                "schurSeconds": schur_time,
                "drsTimeRatio": drs_time / schur_time,
                "drsSolveTimeRatio": drs_solve_time / schur_time,
                "pareto": pareto,
                "warnings": warnings,
            }
        )
    return comparisons


def render_markdown(
    comparisons: list[dict[str, Any]], drs_source: str
) -> str:
    lines = [
        "# BA Competitiveness Pilot",
        "",
        f"DRS source: `{drs_source}`.",
        "",
        "This is a diagnostic pilot, not publication evidence. Negative cost gaps favor DRS; time ratios below 1 favor DRS.",
        "",
        "| Problem | DRS K / iters | Schur K / iters | DRS cost gap | DRS total ratio | DRS solve ratio | Result |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparisons:
        problem_id = row["problem"].split("-")[1]
        lines.append(
            f"| {problem_id} | {row['drsPartitions']} / {row['drsIterations']} | "
            f"{row['schurPartitions']} / {row['schurIterations']} | "
            f"{row['drsCostGapPercent']:+.2f}% | {row['drsTimeRatio']:.2f}x | "
            f"{row['drsSolveTimeRatio']:.2f}x | "
            f"{row['pareto']} |"
        )

    warning_counts: dict[str, int] = {}
    for row in comparisons:
        for warning in row["warnings"]:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
    lines.extend(["", "## Claim Status", ""])
    if not comparisons:
        lines.append("No matched completed records were found.")
    else:
        outcomes: dict[str, int] = {}
        for row in comparisons:
            outcomes[row["pareto"]] = outcomes.get(row["pareto"], 0) + 1
        lines.append(
            ", ".join(f"{name}: {count}" for name, count in outcomes.items()) + "."
        )
        lines.append(
            "Current evidence can identify candidate regimes, but cannot support comparative solver claims until the blockers below are resolved."
        )
    lines.extend(["", "## Blockers", ""])
    for warning, count in warning_counts.items():
        lines.append(f"- {warning}: {count}/{len(comparisons)} comparisons")
    lines.extend(
        [
            "- Communication bytes and peak memory are not recorded for both methods.",
            "- Wall times are single runs on unlike execution paths; repeated matched-hardware runs are required.",
            "",
            "## Next Experiment Gate",
            "",
            "Run matched 20-iteration, matched-partition cohorts on problems 52, 871, and 3068; independently evaluate every saved state under the same squared reprojection objective; then add byte and peak-memory accounting before interpreting wall time.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drs-input", type=Path, required=True)
    parser.add_argument("--drs-section")
    parser.add_argument("--schur-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.drs_section:
        drs_records = read_labeled_section(args.drs_input, args.drs_section)
        drs_source = f"{args.drs_input} ({args.drs_section})"
    else:
        drs_records = read_jsonl(args.drs_input)
        drs_source = str(args.drs_input)
    comparisons = compare_records(
        drs_records,
        read_jsonl(args.schur_input),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(comparisons, drs_source))


if __name__ == "__main__":
    main()
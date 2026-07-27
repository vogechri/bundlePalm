#!/usr/bin/env python3
"""Normalize native BA benchmark summaries into a common JSONL schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trajectory_summary(path: Path | None) -> tuple[float | None, int | None]:
    if path is None or not path.is_file():
        return None, None
    elapsed = None
    count = 0
    with path.open() as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            elapsed = record.get("elapsedSeconds", elapsed)
            count += 1
    return elapsed, count


def normalize_record(
    record: dict[str, Any],
    source_format: str,
    source_path: Path,
    dataset_root: Path | None = None,
) -> dict[str, Any]:
    file_name = record["file_name"]
    dataset_path = dataset_root / file_name if dataset_root is not None else None

    trajectory_value = record.get("trajectory")
    trajectory_path = Path(trajectory_value) if trajectory_value else None
    elapsed_seconds, trajectory_count = _trajectory_summary(trajectory_path)

    if source_format == "drs":
        method_name = "variable_metric_drs"
        method_family = "decentralized_nonlinear_splitting"
        accelerator = record.get("accelerator")
        partitions = record.get("kClusters")
        status = record.get("status", "completed")
    elif source_format == "schur":
        method_name = record.get("algorithm", "unknown_schur_solver")
        method_family = "synchronized_distributed_schur"
        accelerator = record.get("accelerator")
        partitions = record.get("worldSize", record.get("edgePartitions"))
        status = record.get("status", "unknown")
    else:
        raise ValueError(f"unsupported source format: {source_format}")

    objective_verified = record.get("objectiveVerified", False)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "recordType": "summary",
        "method": {
            "name": method_name,
            "family": method_family,
            "accelerator": accelerator,
            "revision": record.get("revision"),
        },
        "dataset": {
            "fileName": file_name,
            "sourceUrl": f"{record.get('base_url', '')}{file_name}",
            "sha256": _sha256(dataset_path),
        },
        "configuration": {
            "outerIterationsRequested": record.get("iterations"),
            "outerIterationsCompleted": record.get("epochsCompleted", trajectory_count),
            "partitions": partitions,
            "linearSolver": record.get("linearSolver"),
            "globalScaling": record.get("globalJacobi"),
        },
        "outcome": {
            "status": status,
            "objective": {
                "name": "sum_squared_reprojection_error" if objective_verified else "unverified_native_cost",
                "best": record.get("bestCost"),
                "bestIteration": record.get("bestIt"),
                "independentlyVerified": objective_verified,
            },
            "elapsedSeconds": record.get(
                "elapsedSeconds", record.get("overallSeconds", elapsed_seconds)
            ),
            "partitionSeconds": record.get("partitionSeconds"),
            "peakMemoryBytes": record.get("peakMemoryBytes"),
        },
        "distributedMetrics": {
            "synchronizationRounds": record.get("synchronizationRounds"),
            "bytesSent": record.get("bytesSent"),
            "bytesReceived": record.get("bytesReceived"),
        },
        "artifacts": {
            "nativeSummary": str(source_path),
            "trajectory": trajectory_value,
            "log": record.get("log"),
        },
        "provenance": {
            "sourceFormat": source_format,
            "nativeRecord": record,
        },
    }


def read_jsonl(
    path: Path, allow_legacy_metadata: bool = False
) -> Iterable[dict[str, Any]]:
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if allow_legacy_metadata and not stripped.startswith("{"):
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("drs", "schur"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for record in read_jsonl(
            args.input, allow_legacy_metadata=args.format == "drs"
        ):
            normalized = normalize_record(
                record, args.format, args.input, args.dataset_root
            )
            output.write(json.dumps(normalized, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Summarize C++ local-solve metrics and Python outer-loop feedback."""

import argparse
import math
import re
import statistics
from pathlib import Path


LOCAL_PATTERN = re.compile(
    r"LOCAL_SOLVE cluster=(?P<cluster>\d+)"
    r" be=(?P<be>[-+\d.eE]+)"
    r" attempts=(?P<attempts>\d+)"
    r" rejections=(?P<rejections>\d+)"
    r" start=(?P<start>[-+\d.eE]+)"
    r" end=(?P<end>[-+\d.eE]+)"
    r" predicted=(?P<predicted>[-+\d.eE]+)"
    r" rho=(?P<rho>[-+\d.eE]+)"
    r" radius=(?P<radius>[-+\d.eE]+)"
)
BE_PATTERN = re.compile(r"\bBE\s+([-+\d.eE]+)")
BE_INCREASE_PATTERN = re.compile(r"Be \*=\s*([-+\d.eE]+)")
DIAGONAL_PATTERN = re.compile(
    r"CAMERA_DIAGONAL cluster=(?P<cluster>\d+)"
    r" blocks=(?P<blocks>\d+)"
    r" entries=(?P<entries>\d+)"
    r" floored=(?P<floored>\d+)"
    r" zeros=(?P<zeros>\d+)"
    r" min_relative=(?P<min_relative>[-+\d.eE]+)"
    r" floor=(?P<floor>[-+\d.eE]+)"
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = round(fraction * (len(ordered) - 1))
    return ordered[index]


def summarize_server(path: Path) -> None:
    text = path.read_text(errors="replace")
    records = []
    for match in LOCAL_PATTERN.finditer(text):
        record = {
            key: int(value) if key in {"cluster", "attempts", "rejections"}
            else float(value)
            for key, value in match.groupdict().items()
        }
        record["relative_change"] = (
            (record["end"] - record["start"]) / max(abs(record["start"]), 1.0)
        )
        records.append(record)

    if not records:
        print(f"server {path}: no LOCAL_SOLVE records")
        return

    attempts = [record["attempts"] for record in records]
    rejections = [record["rejections"] for record in records]
    rho = [record["rho"] for record in records]
    relative_change = [record["relative_change"] for record in records]
    print(
        f"server {path}: solves={len(records)} "
        f"attempts={sum(attempts)} retries={sum(rejections)} "
        f"retried_solves={sum(value > 0 for value in rejections)} "
        f"attempts_mean={statistics.mean(attempts):.3f} "
        f"attempts_p95={percentile(attempts, 0.95):.0f}"
    )
    print(
        f"  rho median={statistics.median(rho):.4g} "
        f"p05={percentile(rho, 0.05):.4g}; "
        f"accepted_relative_change max={max(relative_change):.3e} "
        f"positive={sum(value > 0 for value in relative_change)}"
    )

    by_cluster: dict[int, list[dict[str, float]]] = {}
    for record in records:
        by_cluster.setdefault(record["cluster"], []).append(record)
    worst = sorted(
        by_cluster.items(),
        key=lambda item: sum(record["rejections"] for record in item[1]),
        reverse=True,
    )[:5]
    print("  highest-retry clusters:")
    for cluster, cluster_records in worst:
        print(
            f"    {cluster}: solves={len(cluster_records)} "
            f"retries={sum(record['rejections'] for record in cluster_records)} "
            f"max_attempts={max(record['attempts'] for record in cluster_records)} "
            f"final_be={cluster_records[-1]['be']:.4g}"
        )

    diagonal_records = list(DIAGONAL_PATTERN.finditer(text))
    if diagonal_records:
        floored = [int(match.group("floored")) for match in diagonal_records]
        zeros = [int(match.group("zeros")) for match in diagonal_records]
        minimum = [float(match.group("min_relative")) for match in diagonal_records]
        print(
            f"  camera diagonal: evaluations={len(diagonal_records)} "
            f"floored_entries={sum(floored)} zero_entries={sum(zeros)} "
            f"affected_evaluations={sum(value > 0 for value in floored)} "
            f"minimum_relative={min(minimum):.3e}"
        )


def summarize_client(path: Path) -> None:
    text = path.read_text(errors="replace")
    be_values = [float(match.group(1)) for match in BE_PATTERN.finditer(text)]
    increases = [float(match.group(1)) for match in BE_INCREASE_PATTERN.finditer(text)]
    outer_rejections = text.count("************** REVERTED iteration")
    acceleration_resets = text.count("Reset Nesterov acceleration")
    if not be_values:
        print(f"client {path}: no outer-loop BE records")
        return
    print(
        f"client {path}: iterations={len(be_values)} "
        f"outer_rejections={outer_rejections} be_increases={len(increases)} "
        f"acceleration_resets={acceleration_resets} "
        f"be_initial={be_values[0]:.4g} be_final={be_values[-1]:.4g} "
        f"be_max={max(be_values):.4g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-log", type=Path, action="append", default=[])
    parser.add_argument("--client-log", type=Path, action="append", default=[])
    args = parser.parse_args()
    if not args.server_log and not args.client_log:
        parser.error("provide at least one --server-log or --client-log")
    for path in args.server_log:
        summarize_server(path)
    for path in args.client_log:
        summarize_client(path)


if __name__ == "__main__":
    main()
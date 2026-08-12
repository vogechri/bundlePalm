#!/usr/bin/env python3
"""Analyze all-scene base-backbone C1 versus C1+C5 confirmation."""

import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "serverTest"))
from analyze_stage_c_base_backbone_factorial import (  # noqa: E402
    BASE_PATHS,
    base_checkpoints,
    compare,
    load_directory,
    scene_key,
    validate_configuration,
)


def scenes_from_rows(rows):
    return tuple(sorted(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    arguments = parser.parse_args()
    summary = {}
    for family, expected_count in (("1dsfm", 15), ("bal", 29)):
        c1 = load_directory(arguments.root / family / "c1")
        c1_c5 = load_directory(arguments.root / family / "c1_c5")
        scenes = scenes_from_rows(c1)
        if len(scenes) != expected_count or set(c1_c5) != set(scenes):
            raise ValueError(
                f"coverage mismatch {family}: C1={len(c1)}, C1+C5={len(c1_c5)}"
            )
        for row in c1.values():
            validate_configuration(row, "c1", arguments.iterations)
        for row in c1_c5.values():
            validate_configuration(row, "c1_c5", arguments.iterations)
        base = base_checkpoints(family, scenes, arguments.iterations)
        modes = {"c1": c1, "c1_c5": c1_c5}
        summary[family] = {
            "scenes": list(scenes),
            "modes": {
                mode: {
                    "versus_base_i30": compare(rows, base, scenes),
                    "optimization_seconds": math.fsum(
                        row["optimizationSeconds"] for row in rows.values()
                    ),
                    "completed": sum(
                        row["completedIterations"] == arguments.iterations
                        for row in rows.values()
                    ),
                    "recovery_exhausted": sum(
                        row["terminationReason"] == "recovery_exhausted"
                        for row in rows.values()
                    ),
                }
                for mode, rows in modes.items()
            },
            "c1_c5_versus_c1": compare(c1_c5, c1, scenes),
        }

    arguments.root.mkdir(parents=True, exist_ok=True)
    (arguments.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = arguments.root / "report.md"
    with report.open("w", encoding="utf-8") as output:
        output.write("# Base-Backbone C1/C1+C5 All-Scene Confirmation\n\n")
        output.write(
            "Both rows use one global K24/I30 shared-only configuration with "
            "local Nesterov, persistent DRS trust, curvature 0.4 recovery/decay, "
            "camera metric 75, and no proposal damping. C1 is always enabled; "
            "only delayed C5 differs.\n\n"
        )
        output.write(
            "| Family | Variant | SSE/base I30 | W/L | Worst | Optimization s | "
            "Completed | Recovery exhausted |\n"
        )
        output.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for family in ("1dsfm", "bal"):
            for mode, label in (("c1", "Base backbone + C1"), ("c1_c5", "Base backbone + C1+C5")):
                row = summary[family]["modes"][mode]
                ratio = row["versus_base_i30"]
                output.write(
                    f"| {family} | {label} | {ratio['geometric_sse']:.6f} | "
                    f"{ratio['wins']}/{ratio['losses']} | {ratio['worst']:.6f} | "
                    f"{row['optimization_seconds']:.3f} | {row['completed']} | "
                    f"{row['recovery_exhausted']} |\n"
                )
        output.write("\n| Family | C1+C5/C1 SSE | W/L | Worst |\n")
        output.write("|---|---:|---:|---:|\n")
        for family in ("1dsfm", "bal"):
            row = summary[family]["c1_c5_versus_c1"]
            output.write(
                f"| {family} | {row['geometric_sse']:.6f} | "
                f"{row['wins']}/{row['losses']} | {row['worst']:.6f} |\n"
            )
        output.write("\n## Gate\n\n")
        output.write(
            "The hybrid is globally promotable only if it improves the preserved "
            "base-I30 trajectory on both complete families without recovery "
            "exhaustion. C5 remains incremental and must justify its added work.\n"
        )
    print(report)


if __name__ == "__main__":
    main()
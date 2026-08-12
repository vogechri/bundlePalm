#!/usr/bin/env python3
"""Compare the shared-only proposal C1 candidate with base DRS checkpoints."""

import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "serverTest"))
from analyze_stage_c_base_backbone_factorial import (  # noqa: E402
    base_checkpoints,
    compare,
    load_directory,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--proposal-until", type=int, default=0)
    parser.add_argument("--acceleration-until", type=int, default=0)
    parser.add_argument("--rebase-iteration", type=int, default=0)
    parser.add_argument("--mode", choices=("plain", "c1"), default="c1")
    parser.add_argument("--proposal-scale", type=float, default=0.5)
    parser.add_argument("--shared-only", type=int, choices=(0, 1), default=1)
    arguments = parser.parse_args()
    summary = {}
    for family, expected_count in (("1dsfm", 15), ("bal", 29)):
        rows = load_directory(arguments.root / family / arguments.mode)
        scenes = tuple(sorted(rows))
        if len(scenes) != expected_count:
            raise ValueError(f"coverage mismatch {family}: {len(scenes)}/{expected_count}")
        for scene, row in rows.items():
            expected = {
                "clusters": 24,
                "iterations": arguments.iterations,
                "localSolver": "nesterov",
                "outerAcceleration": (
                    "themelis_nesterov" if arguments.mode == "c1" else "none"
                ),
                "outerAccelerationUntil": arguments.acceleration_until,
                "trustRegionPolicy": "drs",
                "persistentTrustRegion": True,
                "blockRecoveryMode": "curvature",
                "initialBlockCurvatureMultiplier": 0.4,
                "cameraDiagonalMetricScale": 75.0,
                "sharedOnlyCameraProximal": bool(arguments.shared_only),
                "metricProposalDisagreementScale": arguments.proposal_scale,
                "metricProposalDisagreementUntil": arguments.proposal_until,
                "localStateRebaseIteration": arguments.rebase_iteration,
            }
            for key, value in expected.items():
                actual = row.get(key, 0) if key in (
                    "outerAccelerationUntil",
                    "metricProposalDisagreementUntil",
                ) else row.get(key)
                if actual != value:
                    raise ValueError(
                        f"configuration mismatch {family}/{scene}: "
                        f"{key}={actual!r}, expected={value!r}"
                    )
            completed = row.get("completedIterations", 0)
            if not 0 < completed <= arguments.iterations:
                raise ValueError(f"invalid completion {family}/{scene}: {completed}")
            if completed < arguments.iterations and row.get("terminationReason") != "recovery_exhausted":
                raise ValueError(
                    f"unexpected early termination {family}/{scene}: "
                    f"{completed}/{arguments.iterations}, {row.get('terminationReason')}"
                )
            rebase_events = sum(
                bool(event.get("localStateRebaseApplied"))
                for event in row.get("trajectory", [])
            )
            expected_rebase_events = 1 if arguments.rebase_iteration > 0 else 0
            if rebase_events != expected_rebase_events:
                raise ValueError(
                    f"rebase telemetry mismatch {family}/{scene}: "
                    f"{rebase_events}, expected {expected_rebase_events}"
                )
        base = base_checkpoints(family, scenes, arguments.iterations)
        quality = compare(rows, base, scenes)
        summary[family] = {
            "scenes": list(scenes),
            "versus_base": quality,
            "optimization_seconds": math.fsum(
                row["optimizationSeconds"] for row in rows.values()
            ),
            "base_optimization_seconds": math.fsum(
                row["optimizationSeconds"] for row in base.values()
            ),
            "optimization_vs_base": (
                math.fsum(row["optimizationSeconds"] for row in rows.values())
                / math.fsum(row["optimizationSeconds"] for row in base.values())
            ),
            "completed": sum(
                row["completedIterations"] == arguments.iterations
                for row in rows.values()
            ),
            "recovery_exhausted": sum(
                row["terminationReason"] == "recovery_exhausted"
                for row in rows.values()
            ),
            "rows": {
                scene: {
                    "sse": rows[scene]["qualityMetrics"]["sumSquaredError"],
                    "sse_vs_base": (
                        rows[scene]["qualityMetrics"]["sumSquaredError"]
                        / base[scene]["qualityMetrics"]["sumSquaredError"]
                    ),
                }
                for scene in scenes
            },
        }

    arguments.root.mkdir(parents=True, exist_ok=True)
    (arguments.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = arguments.root / "report.md"
    with report.open("w", encoding="utf-8") as output:
        output.write(
            f"# Shared-Only {arguments.mode.upper()} At I{arguments.iterations}\n\n"
        )
        output.write(
            "The candidate uses K24, local Nesterov, C1, persistent DRS trust, "
            "curvature 0.4 recovery/decay, camera metric 75, "
            f"{'shared-only' if arguments.shared_only else 'legacy all-camera'} camera "
            f"proximal terms, and proposal damping {arguments.proposal_scale} "
            "applied only to duplicated "
            f"cameras through outer iteration {arguments.proposal_until or 'all'}. "
            f"C1 is active through iteration {arguments.acceleration_until or 'all'}.\n\n"
            f"Local trust/curvature state rebases at iteration "
            f"{arguments.rebase_iteration or 'never'}.\n\n"
        )
        output.write(
            "| Family | SSE/base | W/L | Worst | Optimization s | Base s | "
            "Time/base | Completed | Recovery exhausted |\n"
        )
        output.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for family in ("1dsfm", "bal"):
            row = summary[family]
            quality = row["versus_base"]
            output.write(
                f"| {family} | {quality['geometric_sse']:.6f} | "
                f"{quality['wins']}/{quality['losses']} | {quality['worst']:.6f} | "
                f"{row['optimization_seconds']:.3f} | "
                f"{row['base_optimization_seconds']:.3f} | "
                f"{row['optimization_vs_base']:.3f} | {row['completed']} | "
                f"{row['recovery_exhausted']} |\n"
            )
        output.write("\n## Gate\n\n")
        output.write(
            "Promotion requires one unchanged candidate to improve base DRS on "
            "both complete families without recovery exhaustion and without a "
            "material worst-scene regression.\n"
        )
    print(report)


if __name__ == "__main__":
    main()
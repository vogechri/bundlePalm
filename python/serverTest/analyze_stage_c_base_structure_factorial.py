#!/usr/bin/env python3
"""Compare legacy proposal controls with shared-only base-backbone C1."""

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
    scene_key,
)


ARMS = {
    "legacy_proposal": {
        "label": "Legacy all-camera + proposal 0.5 + C1",
        "shared_only": False,
        "proposal": 0.5,
    },
    "legacy_no_proposal": {
        "label": "Legacy all-camera + no proposal + C1",
        "shared_only": False,
        "proposal": 1.0,
    },
    "shared_no_proposal": {
        "label": "Shared-only + no proposal + C1",
        "shared_only": True,
        "proposal": 1.0,
    },
    "shared_proposal": {
        "label": "Shared-only + proposal 0.5 + C1",
        "shared_only": True,
        "proposal": 0.5,
    },
}


def load_directory(directory):
    paths = tuple(directory.glob("*.jsonl"))
    if len(paths) != 1:
        raise ValueError(f"expected one JSONL in {directory}, found {len(paths)}")
    rows = {}
    for line in paths[0].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scene = scene_key(row)
        if scene in rows:
            previous = rows[scene]
            signature = (
                "completedIterations",
                "terminationReason",
                "rejections",
            )
            if any(previous.get(key) != row.get(key) for key in signature) or not math.isclose(
                previous["qualityMetrics"]["sumSquaredError"],
                row["qualityMetrics"]["sumSquaredError"],
                rel_tol=1e-12,
            ):
                raise ValueError(f"conflicting duplicate scene {scene} in {paths[0]}")
        rows[scene] = row
    return rows


def validate(rows, arm, iterations):
    expected = ARMS[arm]
    for scene, row in rows.items():
        fields = {
            "clusters": 24,
            "iterations": iterations,
            "localSolver": "nesterov",
            "outerAcceleration": "themelis_nesterov",
            "trustRegionPolicy": "drs",
            "persistentTrustRegion": True,
            "blockRecoveryMode": "curvature",
            "initialBlockCurvatureMultiplier": 0.4,
            "cameraDiagonalMetricScale": 75.0,
            "sharedOnlyCameraProximal": expected["shared_only"],
            "metricProposalDisagreementScale": expected["proposal"],
        }
        for key, value in fields.items():
            if row.get(key) != value:
                raise ValueError(
                    f"configuration mismatch {arm}/{scene}: "
                    f"{key}={row.get(key)!r}, expected={value!r}"
                )
        completed = row.get("completedIterations", 0)
        if not 0 < completed <= iterations:
            raise ValueError(f"invalid completion {arm}/{scene}: {completed}")
        if completed < iterations and row.get("terminationReason") != "recovery_exhausted":
            raise ValueError(
                f"unexpected early termination {arm}/{scene}: "
                f"{completed}/{iterations}, {row.get('terminationReason')}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-proposal-root", type=Path, required=True)
    parser.add_argument("--legacy-no-proposal-root", type=Path, required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--shared-proposal-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    arguments = parser.parse_args()
    roots = {
        "legacy_proposal": arguments.legacy_proposal_root,
        "legacy_no_proposal": arguments.legacy_no_proposal_root,
        "shared_no_proposal": arguments.shared_root,
        "shared_proposal": arguments.shared_proposal_root,
    }
    summary = {}
    for family, expected_count in (("1dsfm", 15), ("bal", 29)):
        rows = {
            arm: load_directory(root / family / "c1")
            for arm, root in roots.items()
        }
        scenes = tuple(sorted(rows["shared_no_proposal"]))
        if len(scenes) != expected_count or any(set(value) != set(scenes) for value in rows.values()):
            raise ValueError(f"coverage mismatch for {family}")
        for arm, value in rows.items():
            validate(value, arm, arguments.iterations)
        base = base_checkpoints(family, scenes, arguments.iterations)
        summary[family] = {
            "scenes": list(scenes),
            "arms": {
                arm: {
                    "versus_base_i30": compare(value, base, scenes),
                    "optimization_seconds": math.fsum(
                        row["optimizationSeconds"] for row in value.values()
                    ),
                    "completed": sum(
                        row["completedIterations"] == arguments.iterations
                        for row in value.values()
                    ),
                    "recovery_exhausted": sum(
                        row["terminationReason"] == "recovery_exhausted"
                        for row in value.values()
                    ),
                }
                for arm, value in rows.items()
            },
            "proposal_effect_with_legacy_proximal": compare(
                rows["legacy_proposal"], rows["legacy_no_proposal"], scenes
            ),
            "shared_only_effect_without_proposal": compare(
                rows["shared_no_proposal"], rows["legacy_no_proposal"], scenes
            ),
            "shared_proposal_interaction": compare(
                rows["shared_proposal"], rows["legacy_proposal"], scenes
            ),
        }

    arguments.output_root.mkdir(parents=True, exist_ok=True)
    (arguments.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = arguments.output_root / "report.md"
    with report.open("w", encoding="utf-8") as output:
        output.write("# Base-Backbone C1 Structural Factorial\n\n")
        output.write(
            "All arms use K24/I30, local Nesterov, C1, persistent DRS trust, "
            "curvature 0.4 recovery/decay, and camera metric 75. The structural "
            "factors are legacy all-camera versus shared-only proximal treatment "
            "and proposal damping 0.5 versus none. Fixed proposal damping acts "
            "only on duplicated cameras in shared-only mode.\n\n"
        )
        output.write(
            "| Family | Arm | SSE/base I30 | W/L | Worst | Optimization s | "
            "Completed | Recovery exhausted |\n"
        )
        output.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for family in ("1dsfm", "bal"):
            for arm in ARMS:
                row = summary[family]["arms"][arm]
                ratio = row["versus_base_i30"]
                output.write(
                    f"| {family} | {ARMS[arm]['label']} | "
                    f"{ratio['geometric_sse']:.6f} | {ratio['wins']}/{ratio['losses']} | "
                    f"{ratio['worst']:.6f} | {row['optimization_seconds']:.3f} | "
                    f"{row['completed']} | {row['recovery_exhausted']} |\n"
                )
        output.write(
            "\n| Family | Proposal effect, legacy | Shared-only effect, no proposal | "
            "Shared-only effect, proposal |\n"
        )
        output.write("|---|---:|---:|---:|\n")
        for family in ("1dsfm", "bal"):
            output.write(
                f"| {family} | "
                f"{summary[family]['proposal_effect_with_legacy_proximal']['geometric_sse']:.6f} | "
                f"{summary[family]['shared_only_effect_without_proposal']['geometric_sse']:.6f} | "
                f"{summary[family]['shared_proposal_interaction']['geometric_sse']:.6f} |\n"
            )
        output.write("\nRatios below one favor the first named factor level.\n")
    print(report)


if __name__ == "__main__":
    main()
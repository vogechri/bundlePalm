#!/usr/bin/env python3
"""Analyze the shared-fixed interior development gate."""

import argparse
import json
import math
import re
from pathlib import Path


ONE_D_SFM = (
    "gendarmenmarkt", "piccadilly", "roman_forum", "trafalgar",
    "union_square", "vienna_cathedral",
)
BAL = ("bal52", "bal245", "bal1490", "bal1778", "bal3068")
TRIAL_PATTERN = re.compile(
    r"SHARED_FIXED_INTERIOR_TRIAL .*accepted=(\d).*"
    r"shared_camera_maximum_change=([^ ]+).*before=([^ ]+) after=([^ ]+)"
)


def scene_key(row):
    dataset = Path(row["dataset"])
    match = re.search(r"problem-(\d+)-", dataset.name)
    return f"bal{match.group(1)}" if match else dataset.parent.name


def load_arm(directory):
    paths = tuple(directory.glob("*.jsonl"))
    if len(paths) != 1:
        raise ValueError(f"expected one JSONL in {directory}, found {len(paths)}")
    rows = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
    return {scene_key(row): row for row in rows}


def geometric_mean(values):
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def compare(candidate, reference, scenes):
    ratios = [
        candidate[scene]["qualityMetrics"]["sumSquaredError"]
        / reference[scene]["qualityMetrics"]["sumSquaredError"]
        for scene in scenes
    ]
    times = [
        candidate[scene]["optimizationSeconds"]
        / reference[scene]["optimizationSeconds"]
        for scene in scenes
    ]
    return {
        "geometric_sse": geometric_mean(ratios),
        "summed_sse": math.fsum(
            candidate[scene]["qualityMetrics"]["sumSquaredError"] for scene in scenes
        ) / math.fsum(
            reference[scene]["qualityMetrics"]["sumSquaredError"] for scene in scenes
        ),
        "wins": sum(value < 1.0 for value in ratios),
        "losses": sum(value > 1.0 for value in ratios),
        "worst": max(ratios),
        "geometric_optimization_time": geometric_mean(times),
    }


def load_telemetry(directory):
    accepted = 0
    rejected = 0
    maximum_shared_change = 0.0
    before = 0.0
    after = 0.0
    for path in (directory / "logs").glob("*worker.log"):
        for line in path.read_text(errors="replace").splitlines():
            match = TRIAL_PATTERN.search(line)
            if not match:
                continue
            if int(match.group(1)):
                accepted += 1
            else:
                rejected += 1
            maximum_shared_change = max(maximum_shared_change, float(match.group(2)))
            before += float(match.group(3))
            after += float(match.group(4))
    if accepted + rejected == 0:
        raise ValueError("missing interior-trial telemetry")
    if maximum_shared_change != 0.0:
        raise ValueError(f"shared camera changed by {maximum_shared_change}")
    return {
        "accepted": accepted,
        "rejected": rejected,
        "accepted_fraction": accepted / (accepted + rejected),
        "maximum_shared_camera_change": maximum_shared_change,
        "summed_local_cost_ratio": after / before,
    }


def analyze(root):
    control = load_arm(root / "control")
    trial = load_arm(root / "trial")
    expected = set(ONE_D_SFM + BAL)
    if set(control) != expected or set(trial) != expected:
        raise ValueError("development coverage mismatch")
    for arm, rows in (("control", control), ("trial", trial)):
        for scene, row in rows.items():
            if not 0 < row.get("completedIterations", 0) <= 30:
                raise ValueError(f"invalid completion {arm}/{scene}")
            if row.get("clusters") != 24 or not row.get("sharedOnlyCameraProximal"):
                raise ValueError(f"configuration mismatch {arm}/{scene}")
    one_d_sfm = compare(trial, control, ONE_D_SFM)
    bal = compare(trial, control, BAL)
    completion_failures = [
        scene for scene in ONE_D_SFM + BAL
        if trial[scene].get("completedIterations") != 30
    ]
    passed = (
        not completion_failures
        and one_d_sfm["geometric_sse"] < 1.0
        and bal["geometric_sse"] < 1.0
        and one_d_sfm["worst"] <= 1.02
        and bal["worst"] <= 1.02
        and one_d_sfm["geometric_optimization_time"] < 1.25
        and bal["geometric_optimization_time"] < 1.25
    )
    return {
        "status": "passed" if passed else "failed",
        "completion_failures": completion_failures,
        "1dsfm": one_d_sfm,
        "bal": bal,
        "telemetry": load_telemetry(root / "trial"),
        "scenes": {
            scene: {
                "control_sse": control[scene]["qualityMetrics"]["sumSquaredError"],
                "trial_sse": trial[scene]["qualityMetrics"]["sumSquaredError"],
                "ratio": trial[scene]["qualityMetrics"]["sumSquaredError"]
                / control[scene]["qualityMetrics"]["sumSquaredError"],
                "optimization_ratio": trial[scene]["optimizationSeconds"]
                / control[scene]["optimizationSeconds"],
                "completed_iterations": trial[scene]["completedIterations"],
                "termination_reason": trial[scene]["terminationReason"],
            }
            for scene in ONE_D_SFM + BAL
        },
    }


def write_report(path, summary):
    with path.open("w", encoding="utf-8") as output:
        output.write("# Shared-Fixed Interior Trial Development Gate\n\n")
        output.write("| Family | SSE/control | Summed | W/L | Worst | Time/control |\n")
        output.write("|---|---:|---:|---:|---:|---:|\n")
        for family in ("1dsfm", "bal"):
            row = summary[family]
            output.write(
                f"| {family} | {row['geometric_sse']:.9f} | {row['summed_sse']:.9f} | "
                f"{row['wins']}/{row['losses']} | {row['worst']:.9f} | "
                f"{row['geometric_optimization_time']:.6f} |\n"
            )
        telemetry = summary["telemetry"]
        output.write(
            f"\nAccepted/rejected local trials: `{telemetry['accepted']}/"
            f"{telemetry['rejected']}`; accepted fraction "
            f"`{telemetry['accepted_fraction']:.6f}`; maximum shared-camera "
            f"change `{telemetry['maximum_shared_camera_change']:.1f}`; summed "
            f"local cost ratio `{telemetry['summed_local_cost_ratio']:.9f}`.\n\n"
        )
        output.write("| Scene | SSE/control | Time/control | Completion |\n")
        output.write("|---|---:|---:|---|\n")
        for scene, row in summary["scenes"].items():
            output.write(
                f"| {scene} | {row['ratio']:.9f} | {row['optimization_ratio']:.6f} | "
                f"{row['completed_iterations']}/30 {row['termination_reason']} |\n"
            )
        output.write("\n## Decision\n\n")
        if summary["status"] == "passed":
            output.write("The frozen development gate passes.\n")
        else:
            output.write(
                "The frozen development gate fails. Do not tune the trial or "
                "expand it to held-out/all-29 breadth.\n"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    summary = analyze(arguments.root)
    (arguments.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(arguments.root / "report.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
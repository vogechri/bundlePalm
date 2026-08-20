#!/usr/bin/env python3
"""Analyze the staged I60 correction plus canonical I30 DRS restart oracle."""

import argparse
import json
import math
import re
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent
EXPECTED = ("roman_forum", "trafalgar", "bal52", "bal3068")


def scene_key(row):
    dataset = Path(row["dataset"])
    match = re.search(r"problem-(\d+)-", dataset.name)
    return f"bal{match.group(1)}" if match else dataset.parent.name


def load_directory(directory):
    paths = tuple(directory.glob("*.jsonl"))
    if len(paths) != 1:
        raise ValueError(f"expected one JSONL in {directory}, found {len(paths)}")
    rows = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
    result = {scene_key(row): row for row in rows}
    if set(result) != set(EXPECTED):
        raise ValueError(f"coverage mismatch in {directory}: {sorted(result)}")
    return result


def mature_rows():
    result = {}
    for family in ("1dsfm", "bal"):
        directory = (
            WORKSPACE
            / "benchmark_results/stage_c_shared_c1_no_proposal_k24_i90"
            / family
            / "c1"
        )
        paths = tuple(directory.glob("*.jsonl"))
        if len(paths) != 1:
            raise ValueError(f"expected one mature JSONL in {directory}")
        for line in paths[0].read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                result[scene_key(row)] = row
    return result


def geometric_mean(values):
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def checkpoint_best_sse(row, iterations):
    best = row["initialQualityMetrics"]["sumSquaredError"]
    trajectory = [
        entry for entry in row["trajectory"]
        if int(entry["iteration"]) < iterations
    ]
    if len(trajectory) != iterations:
        raise ValueError(f"missing I{iterations} trajectory prefix for {scene_key(row)}")
    for entry in trajectory:
        candidate = entry["refinedCandidateSumSquaredError"]
        if not entry["rejected"] and math.isfinite(candidate):
            best = min(best, candidate)
    return best


def assert_matching_prefix(control, staged, iterations):
    control_prefix = control["trajectory"][:iterations]
    staged_prefix = staged["trajectory"][:iterations]
    if len(control_prefix) != iterations or len(staged_prefix) != iterations:
        raise ValueError(f"missing I{iterations} prefix for {scene_key(staged)}")
    fields = (
        "sumSquaredError",
        "refinedCandidateSumSquaredError",
        "rejected",
        "rejections",
        "proximalOracleCalls",
    )
    for index, (control_row, staged_row) in enumerate(
        zip(control_prefix, staged_prefix)
    ):
        for field in fields:
            if control_row[field] != staged_row[field]:
                raise ValueError(
                    f"I{iterations} prefix mismatch for {scene_key(staged)} "
                    f"at iteration {index}, field {field}"
                )


def ratios(candidate, reference, scenes):
    values = [candidate[scene] / reference[scene] for scene in scenes]
    return {
        "geometric": geometric_mean(values),
        "summed": math.fsum(candidate[scene] for scene in scenes)
        / math.fsum(reference[scene] for scene in scenes),
        "wins": sum(value < 1.0 for value in values),
        "losses": sum(value > 1.0 for value in values),
        "worst": max(values),
    }


def analyze(root):
    stage_a = load_directory(root / "stage_a_i60_correction")
    stage_b = load_directory(root / "stage_b_restart_i30")
    control = load_directory(root / "control_i90_correction")
    mature = mature_rows()
    details = {}
    for scene in EXPECTED:
        a = stage_a[scene]
        b = stage_b[scene]
        if a.get("completedIterations") != 60 or b.get("completedIterations") != 30:
            raise ValueError(f"incomplete staged trajectory for {scene}")
        if a.get("iterations") != 90 or a.get("stopAfterIteration") != 60:
            raise ValueError(f"Stage A horizon mismatch for {scene}")
        if a.get("terminationReason") != "configured_iteration_stop":
            raise ValueError(f"Stage A stop mismatch for {scene}")
        assert_matching_prefix(control[scene], a, 60)
        mature_i60 = checkpoint_best_sse(control[scene], 60)
        stage_a_pre = a["finalSharedSchurInitialSSE"]
        prefix_ratio = stage_a_pre / mature_i60
        if abs(prefix_ratio - 1.0) > 1e-10:
            raise ValueError(f"I60 prefix mismatch for {scene}: {prefix_ratio}")
        accepted = [attempt for attempt in a["finalSharedSchurAttempts"] if attempt["accepted"]]
        if len(accepted) != 1:
            raise ValueError(f"expected one Stage A correction for {scene}")
        diagnostics = accepted[0]["diagnostics"]
        if diagnostics["linearTermination"] != 0 or diagnostics["relativeResidual"] >= 1e-6:
            raise ValueError(f"invalid accepted correction for {scene}")
        details[scene] = {
            "i60_pre": a["finalSharedSchurInitialSSE"],
            "mature_i60": mature_i60,
            "i60_pre_over_mature_prefix": prefix_ratio,
            "i60_corrected": a["qualityMetrics"]["sumSquaredError"],
            "restart_i30": b["qualityMetrics"]["sumSquaredError"],
            "mature_i90": control[scene]["finalSharedSchurInitialSSE"],
            "endpoint_one": control[scene]["qualityMetrics"]["sumSquaredError"],
            "committed_mature_i90": mature[scene]["qualityMetrics"]["sumSquaredError"],
            "restart_over_mature": b["qualityMetrics"]["sumSquaredError"]
            / control[scene]["finalSharedSchurInitialSSE"],
            "restart_over_endpoint": b["qualityMetrics"]["sumSquaredError"]
            / control[scene]["qualityMetrics"]["sumSquaredError"],
            "restart_over_i60_corrected": b["qualityMetrics"]["sumSquaredError"]
            / a["qualityMetrics"]["sumSquaredError"],
            "accepted_damping": accepted[0]["cameraDamping"],
        }
    summary = {"status": "completed", "scenes": details, "families": {}}
    for family, scenes in {
        "1dsfm": ("roman_forum", "trafalgar"),
        "bal": ("bal52", "bal3068"),
    }.items():
        candidate = {scene: details[scene]["restart_i30"] for scene in scenes}
        summary["families"][family] = {
            "restart_over_mature": ratios(
                candidate,
                {scene: details[scene]["mature_i90"] for scene in scenes},
                scenes,
            ),
            "restart_over_endpoint": ratios(
                candidate,
                {scene: details[scene]["endpoint_one"] for scene in scenes},
                scenes,
            ),
            "restart_over_i60_corrected": ratios(
                candidate,
                {scene: details[scene]["i60_corrected"] for scene in scenes},
                scenes,
            ),
        }
    return summary


def write_report(path, summary):
    with path.open("w", encoding="utf-8") as output:
        output.write("# Late-Correction Phase 2 Restart Oracle\n\n")
        output.write(
            "Mature DRS runs to I60, accepts the frozen one correction, then "
            "restarts a fresh I30 DRS process from the corrected physical state.\n\n"
        )
        output.write("| Family | Restart/mature I90 | Restart/endpoint one | Restart/I60 corrected |\n")
        output.write("|---|---:|---:|---:|\n")
        for family, values in summary["families"].items():
            output.write(
                f"| {family} | {values['restart_over_mature']['geometric']:.9f} | "
                f"{values['restart_over_endpoint']['geometric']:.9f} | "
                f"{values['restart_over_i60_corrected']['geometric']:.9f} |\n"
            )
        output.write("\n| Scene | Restart/mature | Restart/endpoint | Restart/I60 corrected |\n")
        output.write("|---|---:|---:|---:|\n")
        for scene, row in summary["scenes"].items():
            output.write(
                f"| {scene} | {row['restart_over_mature']:.9f} | "
                f"{row['restart_over_endpoint']:.9f} | "
                f"{row['restart_over_i60_corrected']:.9f} |\n"
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
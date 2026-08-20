#!/usr/bin/env python3
"""Validate matched mature and one-correction Phase 1 controls."""

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


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def geometric_mean(values):
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def reference_rows():
    base_paths = (
        WORKSPACE / "benchmark_results/stage_c_shared_c1_no_proposal_k24_i90/1dsfm/c1",
        WORKSPACE / "benchmark_results/stage_c_shared_c1_no_proposal_k24_i90/bal/c1",
    )
    base = {}
    for directory in base_paths:
        paths = tuple(directory.glob("*.jsonl"))
        if len(paths) != 1:
            raise ValueError(f"expected one mature JSONL in {directory}")
        base.update({scene_key(row): row for row in load_jsonl(paths[0])})
    return base


def analyze(root):
    paths = tuple(root.glob("*.jsonl"))
    if len(paths) != 1:
        raise ValueError(f"expected one Phase 1 JSONL in {root}, found {len(paths)}")
    rows = {scene_key(row): row for row in load_jsonl(paths[0])}
    if set(rows) != set(EXPECTED):
        raise ValueError(f"coverage mismatch: {sorted(rows)}")
    base = reference_rows()
    details = {}
    pre_ratios = []
    correction_ratios = []
    for scene in EXPECTED:
        row = rows[scene]
        if row.get("iterations") != 0 or row.get("clusters") != 24:
            raise ValueError(f"configuration mismatch for {scene}")
        pre = row["initialQualityMetrics"]["sumSquaredError"]
        mature = base[scene]["qualityMetrics"]["sumSquaredError"]
        final = row["qualityMetrics"]["sumSquaredError"]
        pre_ratio = pre / mature
        if abs(pre_ratio - 1.0) > 1e-8:
            raise ValueError(f"mature state mismatch for {scene}: {pre_ratio}")
        accepted = [attempt for attempt in row["finalSharedSchurAttempts"] if attempt["accepted"]]
        if len(accepted) != 1:
            raise ValueError(f"expected one accepted correction for {scene}")
        attempt = accepted[0]
        diagnostics = attempt["diagnostics"]
        if diagnostics["linearTermination"] != 0:
            raise ValueError(f"nonconverged accepted correction for {scene}")
        if diagnostics["relativeResidual"] >= 1e-6:
            raise ValueError(f"residual gate failed for {scene}")
        if attempt["dampedGainRatio"] <= 0.0 or final >= pre:
            raise ValueError(f"acceptance gate failed for {scene}")
        pre_ratios.append(pre_ratio)
        correction_ratios.append(final / pre)
        details[scene] = {
            "mature_sse": mature,
            "pre_sse": pre,
            "corrected_sse": final,
            "pre_over_mature": pre_ratio,
            "corrected_over_pre": final / pre,
            "accepted_damping": attempt["cameraDamping"],
            "attempts": len(row["finalSharedSchurAttempts"]),
            "relative_residual": diagnostics["relativeResidual"],
            "linear_iterations": diagnostics["linearIterations"],
            "damped_gain_ratio": attempt["dampedGainRatio"],
        }
    return {
        "status": "passed",
        "source_commit": "3e1cf1f",
        "pre_over_mature_geomean": geometric_mean(pre_ratios),
        "corrected_over_pre_geomean": geometric_mean(correction_ratios),
        "scenes": details,
    }


def write_report(path, summary):
    with path.open("w", encoding="utf-8") as output:
        output.write("# Late-Correction Phase 1 Controls\n\n")
        output.write(
            f"Mature reload geomean: `{summary['pre_over_mature_geomean']:.12f}x`. "
            f"One correction/prestate: `{summary['corrected_over_pre_geomean']:.9f}x`.\n\n"
        )
        output.write("| Scene | Pre/mature | Corrected/pre | Damping | Attempts | Residual |\n")
        output.write("|---|---:|---:|---:|---:|---:|\n")
        for scene, row in summary["scenes"].items():
            output.write(
                f"| {scene} | {row['pre_over_mature']:.12f} | "
                f"{row['corrected_over_pre']:.9f} | {row['accepted_damping']:.9g} | "
                f"{row['attempts']} | {row['relative_residual']:.3e} |\n"
            )
        output.write("\nAll accepted solves converged and passed physical-SSE/model-gain gates.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    summary = analyze(arguments.root)
    (arguments.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(arguments.root / "report.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
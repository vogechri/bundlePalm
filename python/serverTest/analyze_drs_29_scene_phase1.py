#!/usr/bin/env python3

"""Compare the corrected Phase-1 one-factor screen to plain Phase-0 DRS."""

import argparse
import json
import math
from pathlib import Path


VARIANTS = (
    "consensus_diagonal",
    "consensus_scalar",
    "consensus_arithmetic",
    "scaling_none",
    "recovery_regularization",
    "curvature_005",
    "curvature_02",
    "trust_persistent_daba",
    "trust_drs",
    "local_lmref1",
    "partition_stable",
    "solver_schur_pcg",
    "nesterov_ternary",
    "final_polish1",
)


def load_results(directory):
    results = {}
    for path in Path(directory).glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                result = json.loads(line)
                scene = Path(result["dataset"]).name.split("-")[1]
                results[scene] = result
    return results


def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase0_root")
    parser.add_argument("output_root")
    parser.add_argument("--report")
    arguments = parser.parse_args()

    baseline = load_results(Path(arguments.phase0_root) / "plain")
    output_root = Path(arguments.output_root)
    variants = {
        variant: load_results(output_root / variant)
        for variant in VARIANTS
    }
    lines = [
        "# Corrected 29-scene Phase-1 progress",
        "",
        f"- baseline completed: {len(baseline)}/29",
        "",
        "| Variant | Completed | SSE ratio | W/T/L | Time ratio | Calls ratio | Rejections B/V | Failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ranking = []
    for variant, results in variants.items():
        paired = sorted(set(baseline) & set(results), key=int)
        quality_ratios = []
        time_ratios = []
        call_ratios = []
        for scene in paired:
            base = baseline[scene]
            trial = results[scene]
            quality_ratios.append(
                trial["qualityMetrics"]["sumSquaredError"]
                / base["qualityMetrics"]["sumSquaredError"]
            )
            if base.get("optimizationSeconds") and trial.get("optimizationSeconds"):
                time_ratios.append(
                    trial["optimizationSeconds"] / base["optimizationSeconds"]
                )
            call_ratios.append(
                trial["proximalOracleCalls"] / base["proximalOracleCalls"]
            )
        failures = sum(
            result.get("terminationReason") != "iteration_limit"
            for result in results.values()
        )
        if quality_ratios:
            quality_ratio = geometric_mean(quality_ratios)
            ranking.append((quality_ratio, variant, len(paired)))
            quality_cell = f"{quality_ratio:.6f}"
            wtl = (
                f"{sum(value < 0.999 for value in quality_ratios)}/"
                f"{sum(0.999 <= value <= 1.001 for value in quality_ratios)}/"
                f"{sum(value > 1.001 for value in quality_ratios)}"
            )
            time_cell = (
                f"{geometric_mean(time_ratios):.6f}"
                if time_ratios else "-"
            )
            call_cell = f"{geometric_mean(call_ratios):.6f}"
            rejection_cell = (
                f"{sum(baseline[s]['rejections'] for s in paired)}/"
                f"{sum(results[s]['rejections'] for s in paired)}"
            )
        else:
            quality_cell = wtl = time_cell = call_cell = rejection_cell = "-"
        lines.append(
            f"| {variant} | {len(results)}/29 | {quality_cell} | {wtl} | "
            f"{time_cell} | {call_cell} | {rejection_cell} | {failures} |"
        )

    lines.extend(["", "## Current Ranking", ""])
    if ranking:
        for quality_ratio, variant, paired_count in sorted(ranking):
            lines.append(
                f"- {variant}: ratio {quality_ratio:.6f} on "
                f"{paired_count} paired scenes"
            )
    else:
        lines.append("No Phase-1 results are available yet.")

    report = "\n".join(lines) + "\n"
    report_path = (
        Path(arguments.report)
        if arguments.report
        else output_root / "progress.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
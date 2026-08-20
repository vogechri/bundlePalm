#!/usr/bin/env python3
"""Assess whether relative interior defect predicts fixed-L2 benefit."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent.parent


def scene_key(row):
    dataset = Path(row["dataset"])
    if "sfm_init_1dsfm" in str(dataset):
        return dataset.parent.name
    return f"bal{dataset.name.split('-')[1]}"


def load(path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scene = scene_key(row)
        if scene in rows:
            raise ValueError(f"duplicate scene {scene} in {path}")
        rows[scene] = row
    return rows


def relative_forcing(row):
    values = []
    for event in row["trajectory"]:
        numerator = event["interiorDefectSquared"]
        denominator = event["proximalDisplacementSquared"]
        value = float("nan")
        if (
            math.isfinite(numerator)
            and math.isfinite(denominator)
            and numerator >= 0.0
            and denominator > 0.0
        ):
            value = math.sqrt(numerator / denominator)
        values.append(value)
    return np.asarray(values)


def safe_spearman(values, benefits):
    statistic, pvalue = spearmanr(values, benefits)
    return {"rho": float(statistic), "pvalue": float(pvalue)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "benchmark_results/k1_inner_on_mature_base_k24_i30",
    )
    arguments = parser.parse_args()

    diagnostic = load(arguments.root / "direct_shared_diag.jsonl")
    l2 = load(arguments.root / "direct_shared_l2.jsonl")
    if set(diagnostic) != set(l2):
        raise ValueError("diagnostic/L2 scene coverage mismatch")

    records = []
    for scene in sorted(diagnostic):
        control = diagnostic[scene]
        candidate = l2[scene]
        if control["completedIterations"] != 30:
            raise ValueError(f"incomplete diagnostic row {scene}")
        forcing = relative_forcing(control)
        finite = forcing[np.isfinite(forcing)]
        l2_over_l1 = (
            candidate["qualityMetrics"]["sumSquaredError"]
            / control["qualityMetrics"]["sumSquaredError"]
        )
        records.append({
            "scene": scene,
            "family": "bal" if scene.startswith("bal") else "1dsfm",
            "l2OverL1": l2_over_l1,
            "l2Improves": l2_over_l1 < 1.0,
            "l2Completed": candidate["completedIterations"] == 30,
            "maximumQ": float(np.max(finite)),
            "iterationsAboveOne": int(np.sum(finite > 1.0)),
            "earlyIterationsAboveOne": int(np.sum(forcing[:15] > 1.0)),
            "areaAboveOne": float(np.sum(np.maximum(finite - 1.0, 0.0))),
            "earlyAreaAboveOne": float(
                np.nansum(np.maximum(forcing[:15] - 1.0, 0.0))
            ),
            "medianQ": float(np.median(finite)),
            "l2TimeOverL1": (
                candidate["optimizationSeconds"]
                / control["optimizationSeconds"]
            ),
        })

    predicted = [record["iterationsAboveOne"] > 0 for record in records]
    actual = [record["l2Improves"] for record in records]
    true_positive = sum(predict and improve for predict, improve in zip(predicted, actual))
    false_positive = sum(predict and not improve for predict, improve in zip(predicted, actual))
    false_negative = sum(not predict and improve for predict, improve in zip(predicted, actual))
    true_negative = sum(not predict and not improve for predict, improve in zip(predicted, actual))
    benefits = [1.0 - record["l2OverL1"] for record in records]
    feature_names = (
        "maximumQ",
        "iterationsAboveOne",
        "earlyIterationsAboveOne",
        "areaAboveOne",
        "earlyAreaAboveOne",
        "medianQ",
    )
    summary = {
        "criterion": "q = sqrt(interiorDefectSquared / proximalDisplacementSquared)",
        "threshold": 1.0,
        "records": records,
        "confusion": {
            "truePositive": true_positive,
            "falsePositive": false_positive,
            "falseNegative": false_negative,
            "trueNegative": true_negative,
            "precision": true_positive / max(true_positive + false_positive, 1),
            "recall": true_positive / max(true_positive + false_negative, 1),
            "accuracy": (true_positive + true_negative) / len(records),
        },
        "rankCorrelationWithL2Benefit": {
            name: safe_spearman([record[name] for record in records], benefits)
            for name in feature_names
        },
        "decision": "reject as standalone forcing selector",
        "limitation": (
            "fixed L2 repeats the full local solve and updates shared cameras; "
            "it is not an interior-only ground-truth intervention"
        ),
    }
    (arguments.root / "forcing_reliability.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = arguments.root / "forcing_reliability.md"
    with report.open("w", encoding="utf-8") as output:
        output.write("# Relative Interior-Forcing Reliability\n\n")
        output.write(
            "The candidate trigger is `q = sqrt(interior defect / proximal "
            "displacement) > 1`. The L1 diagnostic is bitwise trajectory-neutral. "
            "Fixed L2 is used only as available endpoint ground truth.\n\n"
        )
        output.write("| Scene | Max q | q>1 iterations | Early area | L2/L1 | L2 complete |\n")
        output.write("|---|---:|---:|---:|---:|---:|\n")
        for record in records:
            output.write(
                f"| {record['scene']} | {record['maximumQ']:.6f} | "
                f"{record['iterationsAboveOne']} | "
                f"{record['earlyAreaAboveOne']:.6f} | "
                f"{record['l2OverL1']:.6f} | "
                f"{'yes' if record['l2Completed'] else 'no'} |\n"
            )
        confusion = summary["confusion"]
        output.write("\n## Reliability\n\n")
        output.write(
            f"At threshold 1: TP/FP/FN/TN = {true_positive}/{false_positive}/"
            f"{false_negative}/{true_negative}, precision "
            f"`{confusion['precision']:.3f}`, recall `{confusion['recall']:.3f}`, "
            f"and sign accuracy `{confusion['accuracy']:.3f}`. Maximum-q "
            f"Spearman correlation with L2 benefit is "
            f"`{summary['rankCorrelationWithL2Benefit']['maximumQ']['rho']:.3f}`.\n\n"
        )
        output.write(
            "Reject q>1 as a standalone forcing selector. It triggers on four "
            "1DSfM losses and misses the small BAL1490 gain. Early-area ranking is "
            "only suggestive on eight scenes and is not a promotion statistic.\n\n"
        )
        output.write("## Limitation\n\n")
        output.write(
            "Fixed L2 repeats the full local nonlinear solve and updates shared "
            "cameras as well as interior variables. The diagnostic q measures only "
            "unique-camera/landmark stationarity. A true interior-only trial with "
            "shared cameras fixed is required before rejecting relative forcing "
            "itself; this experiment rejects only q>1 as a direct trigger for the "
            "existing full-L2 intervention.\n"
        )
    print(report)


if __name__ == "__main__":
    main()

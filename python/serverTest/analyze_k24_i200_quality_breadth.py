#!/usr/bin/env python3
"""Analyze complete horizon-matched K24/I200 breadth."""

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent


def scene_key(row):
    dataset = Path(row["dataset"])
    match = re.search(r"problem-(\d+)-", dataset.name)
    return f"bal{match.group(1)}" if match else dataset.parent.name


def load_rows(directory):
    result = {}
    for path in directory.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                result[scene_key(row)] = row
    return result


def load_status(directory):
    path = directory / "status.tsv"
    if not path.is_file():
        return {}
    with path.open(newline="") as source:
        return {
            scene_key({"dataset": row["dataset"]}): row
            for row in csv.DictReader(source, delimiter="\t")
        }


def ceres_rows():
    result = {}
    for path in (
        WORKSPACE / "benchmark_results/1dsfm_drs_ceres_se3_all15/ceres/results.jsonl",
        WORKSPACE / "benchmark_results/bal_ceres_se3_all29/results.jsonl",
    ):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            scene = row.get("scene") or f"bal{row['balId']}"
            result[scene] = (
                row["qualityMetrics"]["sumSquaredError"]
                if "qualityMetrics" in row
                else 2.0 * row["native"]["finalCeresCost"]
            )
    return result


def assert_matching_prefix(short, full, scene):
    fields = (
        "sumSquaredError", "refinedCandidateSumSquaredError", "rejected",
        "rejections", "proximalOracleCalls",
    )
    if len(short["trajectory"]) != 90 or len(full["trajectory"]) < 90:
        raise ValueError(f"missing I90 prefix for {scene}")
    for index, (left, right) in enumerate(zip(short["trajectory"], full["trajectory"][:90])):
        for field in fields:
            if left[field] != right[field]:
                raise ValueError(f"prefix mismatch {scene}/I{index + 1}/{field}")


def correction(row, scene, arm):
    pre = row["finalSharedSchurInitialSSE"]
    final = row["qualityMetrics"]["sumSquaredError"]
    accepted = [attempt for attempt in row["finalSharedSchurAttempts"] if attempt["accepted"]]
    if accepted:
        if len(accepted) != 1:
            raise ValueError(f"multiple accepted corrections {scene}/{arm}")
        attempt = accepted[0]
        diagnostics = attempt["diagnostics"]
        if diagnostics["linearTermination"] != 0 or diagnostics["relativeResidual"] >= 1e-6:
            raise ValueError(f"accepted nonconverged correction {scene}/{arm}")
        if attempt["dampedGainRatio"] <= 0.0 or final >= pre:
            raise ValueError(f"invalid correction acceptance {scene}/{arm}")
        damping = attempt["cameraDamping"]
        residual = diagnostics["relativeResidual"]
    else:
        if abs(final / pre - 1.0) > 1e-10:
            raise ValueError(f"rejected correction changed state {scene}/{arm}")
        damping = None
        residual = None
    return {
        "pre": pre, "final": final, "accepted": bool(accepted),
        "damping": damping, "residual": residual,
        "seconds": row["finalSharedSchurSeconds"],
    }


def geometric_mean(values):
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def summarize(rows, ceres):
    return {
        "count": len(rows),
        "geometric_raw_i200_over_i90": geometric_mean([row["raw_i200_over_i90"] for row in rows]),
        "summed_raw_i200_over_i90": math.fsum(row["raw_i200"] for row in rows)
        / math.fsum(row["raw_i90"] for row in rows),
        "geometric_corrected_i200_over_i90": geometric_mean([row["corrected_i200_over_i90"] for row in rows]),
        "summed_corrected_i200_over_i90": math.fsum(row["corrected_i200"] for row in rows)
        / math.fsum(row["corrected_i90"] for row in rows),
        "geometric_corrected_i200_over_raw_i200": geometric_mean([row["corrected_i200_over_raw_i200"] for row in rows]),
        "geometric_corrected_i90_over_ceres": geometric_mean([row["corrected_i90"] / ceres[row["scene"]] for row in rows]),
        "geometric_corrected_i200_over_ceres": geometric_mean([row["corrected_i200"] / ceres[row["scene"]] for row in rows]),
        "summed_corrected_i200_over_ceres": math.fsum(row["corrected_i200"] for row in rows)
        / math.fsum(ceres[row["scene"]] for row in rows),
        "wins": sum(row["corrected_i200"] < row["corrected_i90"] for row in rows),
        "losses": sum(row["corrected_i200"] > row["corrected_i90"] for row in rows),
        "ceres_wins": sum(row["corrected_i200"] < ceres[row["scene"]] for row in rows),
        "total_i200_overall_seconds": math.fsum(row["i200_overall_seconds"] for row in rows),
        "maximum_coordinator_rss_gib": max(row["i200_coordinator_max_rss_kb"] for row in rows) / 1048576.0,
        "maximum_worker_rss_gib": max(row["i200_worker_max_rss_kb"] for row in rows) / 1048576.0,
    }


def analyze(root):
    ceres = ceres_rows()
    details = {}
    selected = {}
    completion_failures = []
    for family, expected in (("1dsfm", 15), ("bal", 29)):
        short_dir = root / family / "i90_horizon200"
        full_dir = root / family / "i200"
        short_rows = load_rows(short_dir)
        full_rows = load_rows(full_dir)
        short_status = load_status(short_dir)
        full_status = load_status(full_dir)
        if len(short_rows) != expected or len(full_rows) != expected:
            raise ValueError(f"coverage mismatch {family}: {len(short_rows)}/{len(full_rows)}")
        if short_rows.keys() != full_rows.keys():
            raise ValueError(f"scene mismatch {family}")
        for scene in sorted(short_rows):
            short = short_rows[scene]
            full = full_rows[scene]
            if short.get("iterations") != 200 or short.get("completedIterations") != 90:
                raise ValueError(f"I90 arm mismatch {scene}")
            if short.get("terminationReason") != "configured_iteration_stop":
                raise ValueError(f"I90 stop mismatch {scene}")
            if full.get("iterations") != 200 or not 90 < full.get("completedIterations", 0) <= 200:
                raise ValueError(f"I200 iteration mismatch {scene}")
            if full["completedIterations"] < 200:
                if full.get("terminationReason") != "recovery_exhausted":
                    raise ValueError(f"unexpected I200 termination {scene}")
                completion_failures.append(scene)
            if short_status.get(scene, {}).get("status") != "completed" or full_status.get(scene, {}).get("status") != "completed":
                raise ValueError(f"status mismatch {scene}")
            assert_matching_prefix(short, full, scene)
            c90 = correction(short, scene, "i90")
            c200 = correction(full, scene, "i200")
            row = {
                "scene": scene,
                "family": family,
                "raw_i90": c90["pre"], "corrected_i90": c90["final"],
                "raw_i200": c200["pre"], "corrected_i200": c200["final"],
                "raw_i200_over_i90": c200["pre"] / c90["pre"],
                "corrected_i200_over_i90": c200["final"] / c90["final"],
                "corrected_i200_over_raw_i200": c200["final"] / c200["pre"],
                "i90_accepted": c90["accepted"], "i200_accepted": c200["accepted"],
                "i90_damping": c90["damping"], "i200_damping": c200["damping"],
                "i90_residual": c90["residual"], "i200_residual": c200["residual"],
                "i90_correction_seconds": c90["seconds"], "i200_correction_seconds": c200["seconds"],
                "i200_overall_seconds": full["overallSeconds"],
                "completed_iterations": full["completedIterations"],
                "termination_reason": full["terminationReason"],
                "i200_coordinator_max_rss_kb": int(full_status[scene]["coordinator_max_rss_kb"]),
                "i200_worker_max_rss_kb": int(full_status[scene]["worker_max_rss_kb"]),
            }
            details[scene] = row
            selected[(scene, "i90")] = short
            selected[(scene, "i200")] = full
    summaries = {
        family: summarize([row for row in details.values() if row["family"] == family], ceres)
        for family in ("1dsfm", "bal")
    }
    residuals = [
        residual for row in details.values()
        for residual in (row["i90_residual"], row["i200_residual"])
        if residual is not None
    ]
    damping_counts = Counter(
        str(damping) if damping is not None else "noop"
        for row in details.values()
        for damping in (row["i90_damping"], row["i200_damping"])
    )
    passed = (
        not completion_failures
        and
        summaries["1dsfm"]["geometric_corrected_i200_over_i90"] < 1.0
        and summaries["bal"]["geometric_corrected_i200_over_i90"] <= 1.0
    )
    return {
        "status": "passed" if passed else "failed",
        "completion_failures": completion_failures,
        "summaries": summaries,
        "maximum_accepted_residual": max(residuals),
        "damping_counts": dict(sorted(damping_counts.items())),
        "scenes": details,
    }, selected


def write_report(path, summary):
    with path.open("w", encoding="utf-8") as output:
        output.write("# Integrated K24/I200 Terminal-Correction Breadth\n\n")
        output.write("| Family | Raw I200/I90 | Corrected I200/I90 | Summed | Correction/raw I200 | Corrected I90/Ceres | Corrected I200/Ceres | Ceres wins | W/L | Time s | Max RSS GiB C/W |\n")
        output.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for family, row in summary["summaries"].items():
            output.write(
                f"| {family} | {row['geometric_raw_i200_over_i90']:.9f} | "
                f"{row['geometric_corrected_i200_over_i90']:.9f} | "
                f"{row['summed_corrected_i200_over_i90']:.9f} | "
                f"{row['geometric_corrected_i200_over_raw_i200']:.9f} | "
                f"{row['geometric_corrected_i90_over_ceres']:.9f} | "
                f"{row['geometric_corrected_i200_over_ceres']:.9f} | "
                f"{row['ceres_wins']}/{row['count']} | {row['wins']}/{row['losses']} | "
                f"{row['total_i200_overall_seconds']:.3f} | "
                f"{row['maximum_coordinator_rss_gib']:.3f}/{row['maximum_worker_rss_gib']:.3f} |\n"
            )
        output.write(
            f"\nMaximum accepted residual: `{summary['maximum_accepted_residual']:.3e}`. "
            f"Damping counts across both arms: `{summary['damping_counts']}`. "
            f"Completion failures: `{summary['completion_failures']}`. "
            f"Gate status: **{summary['status']}**.\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    summary, selected = analyze(arguments.root)
    (arguments.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(arguments.root / "report.md", summary)
    with (arguments.root / "selected_results.jsonl").open("w", encoding="utf-8") as output:
        for key in sorted(selected):
            output.write(json.dumps(selected[key]) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
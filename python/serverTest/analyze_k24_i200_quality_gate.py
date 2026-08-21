#!/usr/bin/env python3
"""Analyze the horizon-matched integrated K24/I200 quality sentinel."""

import argparse
import csv
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


def load_rows(directory):
    paths = tuple(directory.glob("*.jsonl"))
    if len(paths) != 1:
        raise ValueError(f"expected one JSONL in {directory}, found {len(paths)}")
    rows = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
    result = {scene_key(row): row for row in rows}
    if set(result) != set(EXPECTED):
        raise ValueError(f"coverage mismatch in {directory}: {sorted(result)}")
    return result


def load_status(directory):
    with (directory / "status.tsv").open(newline="") as source:
        rows = csv.DictReader(source, delimiter="\t")
        return {scene_key({"dataset": row["dataset"]}): row for row in rows}


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
        "sumSquaredError",
        "refinedCandidateSumSquaredError",
        "rejected",
        "rejections",
        "proximalOracleCalls",
    )
    if len(short["trajectory"]) != 90 or len(full["trajectory"]) < 90:
        raise ValueError(f"missing I90 prefix for {scene}")
    for index, (left, right) in enumerate(zip(short["trajectory"], full["trajectory"][:90])):
        for field in fields:
            if left[field] != right[field]:
                raise ValueError(
                    f"prefix mismatch {scene}/I{index + 1}/{field}: "
                    f"{left[field]!r} != {right[field]!r}"
                )


def correction(row, scene, label):
    pre = row["finalSharedSchurInitialSSE"]
    final = row["qualityMetrics"]["sumSquaredError"]
    accepted = [attempt for attempt in row["finalSharedSchurAttempts"] if attempt["accepted"]]
    if accepted:
        if len(accepted) != 1:
            raise ValueError(f"multiple accepted corrections {scene}/{label}")
        attempt = accepted[0]
        diagnostics = attempt["diagnostics"]
        if diagnostics["linearTermination"] != 0 or diagnostics["relativeResidual"] >= 1e-6:
            raise ValueError(f"accepted nonconverged correction {scene}/{label}")
        if attempt["dampedGainRatio"] <= 0.0 or final >= pre:
            raise ValueError(f"invalid correction acceptance {scene}/{label}")
        damping = attempt["cameraDamping"]
        residual = diagnostics["relativeResidual"]
    else:
        if abs(final / pre - 1.0) > 1e-12:
            raise ValueError(f"rejected correction changed state {scene}/{label}")
        damping = None
        residual = None
    return {
        "pre": pre,
        "final": final,
        "accepted": bool(accepted),
        "damping": damping,
        "residual": residual,
        "seconds": row["finalSharedSchurSeconds"],
    }


def geometric_mean(values):
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def aggregate(rows, ceres):
    return {
        "raw_i200_over_i90": geometric_mean([row["raw_i200_over_i90"] for row in rows]),
        "corrected_i200_over_i90": geometric_mean([row["corrected_i200_over_i90"] for row in rows]),
        "corrected_i200_over_raw_i200": geometric_mean([row["corrected_i200_over_raw_i200"] for row in rows]),
        "corrected_i90_over_ceres": geometric_mean([row["corrected_i90"] / ceres[row["scene"]] for row in rows]),
        "corrected_i200_over_ceres": geometric_mean([row["corrected_i200"] / ceres[row["scene"]] for row in rows]),
        "summed_corrected_i200_over_i90": math.fsum(row["corrected_i200"] for row in rows)
        / math.fsum(row["corrected_i90"] for row in rows),
        "wins": sum(row["corrected_i200"] < row["corrected_i90"] for row in rows),
        "losses": sum(row["corrected_i200"] > row["corrected_i90"] for row in rows),
    }


def analyze(root):
    i90 = load_rows(root / "i90_horizon200")
    i200 = load_rows(root / "i200")
    status90 = load_status(root / "i90_horizon200")
    status200 = load_status(root / "i200")
    ceres = ceres_rows()
    details = {}
    for scene in EXPECTED:
        short = i90[scene]
        full = i200[scene]
        if short.get("iterations") != 200 or short.get("completedIterations") != 90:
            raise ValueError(f"I90 arm mismatch for {scene}")
        if short.get("terminationReason") != "configured_iteration_stop":
            raise ValueError(f"I90 stop mismatch for {scene}")
        if full.get("iterations") != 200 or full.get("completedIterations") != 200:
            raise ValueError(f"I200 completion mismatch for {scene}")
        if status90[scene]["status"] != "completed" or status200[scene]["status"] != "completed":
            raise ValueError(f"status mismatch for {scene}")
        assert_matching_prefix(short, full, scene)
        corrected90 = correction(short, scene, "i90")
        corrected200 = correction(full, scene, "i200")
        details[scene] = {
            "scene": scene,
            "raw_i90": corrected90["pre"],
            "corrected_i90": corrected90["final"],
            "raw_i200": corrected200["pre"],
            "corrected_i200": corrected200["final"],
            "raw_i200_over_i90": corrected200["pre"] / corrected90["pre"],
            "corrected_i200_over_i90": corrected200["final"] / corrected90["final"],
            "corrected_i200_over_raw_i200": corrected200["final"] / corrected200["pre"],
            "i90_damping": corrected90["damping"],
            "i200_damping": corrected200["damping"],
            "i90_correction_seconds": corrected90["seconds"],
            "i200_correction_seconds": corrected200["seconds"],
            "i200_overall_seconds": full["overallSeconds"],
            "i200_coordinator_max_rss_kb": int(status200[scene]["coordinator_max_rss_kb"]),
            "i200_worker_max_rss_kb": int(status200[scene]["worker_max_rss_kb"]),
        }
    families = {
        "1dsfm": aggregate([details[scene] for scene in EXPECTED[:2]], ceres),
        "bal": aggregate([details[scene] for scene in EXPECTED[2:]], ceres),
    }
    passed = (
        families["1dsfm"]["corrected_i200_over_i90"] < 1.0
        and families["bal"]["corrected_i200_over_i90"] <= 1.0
    )
    return {
        "status": "passed" if passed else "failed",
        "families": families,
        "scenes": details,
    }


def write_report(path, summary):
    with path.open("w", encoding="utf-8") as output:
        output.write("# Integrated K24/I200 Terminal-Correction Sentinel\n\n")
        output.write("| Family | Raw I200/I90 | Corrected I200/I90 | Correction/raw I200 | Corrected I90/Ceres | Corrected I200/Ceres | W/L |\n")
        output.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for family, row in summary["families"].items():
            output.write(
                f"| {family} | {row['raw_i200_over_i90']:.9f} | "
                f"{row['corrected_i200_over_i90']:.9f} | "
                f"{row['corrected_i200_over_raw_i200']:.9f} | "
                f"{row['corrected_i90_over_ceres']:.9f} | "
                f"{row['corrected_i200_over_ceres']:.9f} | "
                f"{row['wins']}/{row['losses']} |\n"
            )
        output.write("\n| Scene | Raw I200/I90 | Corrected I200/I90 | Correction/raw I200 | I90 damping | I200 damping |\n")
        output.write("|---|---:|---:|---:|---:|---:|\n")
        for scene, row in summary["scenes"].items():
            d90 = "--" if row["i90_damping"] is None else f"{row['i90_damping']:.9g}"
            d200 = "--" if row["i200_damping"] is None else f"{row['i200_damping']:.9g}"
            output.write(
                f"| {scene} | {row['raw_i200_over_i90']:.9f} | "
                f"{row['corrected_i200_over_i90']:.9f} | "
                f"{row['corrected_i200_over_raw_i200']:.9f} | {d90} | {d200} |\n"
            )
        output.write(f"\nGate status: **{summary['status']}**.\n")


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
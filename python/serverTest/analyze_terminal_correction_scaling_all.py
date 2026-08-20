#!/usr/bin/env python3
"""Prepare recovery and analyze all K4/K16 terminal corrections."""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent
STATE_TOLERANCE = 1e-6


def scene_key(row):
    dataset = Path(row["dataset"])
    match = re.search(r"problem-(\d+)-", dataset.name)
    return f"bal{match.group(1)}" if match else dataset.parent.name


def manifest_key(scene):
    return scene[3:] if scene.startswith("bal") else scene


def load_rows(directory):
    result = {}
    for path in directory.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                result[scene_key(row)] = row
    return result


def original_rows():
    result = {}
    for family in ("1dsfm", "bal"):
        directory = (
            WORKSPACE / "benchmark_results/stage_c_scaling_confirmation_k4_16_i30"
            / family / "c1_c5"
        )
        for path in directory.glob("*.jsonl"):
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    result[(scene_key(row), row["clusters"])] = row
    if len(result) != 88:
        raise ValueError(f"expected 88 original rows, found {len(result)}")
    return result


def valid_reload(row, reference):
    if row is None or not math.isfinite(row.get("finalSharedSchurInitialSSE", math.nan)):
        return False
    return abs(row["finalSharedSchurInitialSSE"] / reference - 1.0) <= STATE_TOLERANCE


def prepare_recovery(root):
    original = original_rows()
    counts = {}
    for clusters in (4, 16):
        reload_rows = load_rows(root / f"reload_k{clusters}")
        invalid = []
        for (scene, row_clusters), reference_row in sorted(original.items()):
            if row_clusters != clusters:
                continue
            reference = reference_row["qualityMetrics"]["sumSquaredError"]
            if not valid_reload(reload_rows.get(scene), reference):
                invalid.append((scene, reference_row["dataset"]))
        manifest = root / f"recovery_k{clusters}_datasets.txt"
        manifest.write_text(
            "".join(f"{manifest_key(scene)}|{dataset}\n" for scene, dataset in invalid),
            encoding="utf-8",
        )
        (root / f"recovery_k{clusters}_filter.txt").write_text(
            " ".join(manifest_key(scene) for scene, _ in invalid) + "\n",
            encoding="utf-8",
        )
        counts[str(clusters)] = [scene for scene, _ in invalid]
    (root / "recovery_inventory.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(counts, indent=2, sort_keys=True))


def correction_result(row, reference):
    pre = row["finalSharedSchurInitialSSE"]
    final = row["qualityMetrics"]["sumSquaredError"]
    accepted = [attempt for attempt in row["finalSharedSchurAttempts"] if attempt["accepted"]]
    if accepted:
        if len(accepted) != 1:
            raise ValueError("multiple accepted corrections")
        attempt = accepted[0]
        diagnostics = attempt["diagnostics"]
        if diagnostics["linearTermination"] != 0 or diagnostics["relativeResidual"] >= 1e-6:
            raise ValueError("accepted nonconverged correction")
        if attempt["dampedGainRatio"] <= 0.0 or final >= pre:
            raise ValueError("invalid accepted correction")
        damping = attempt["cameraDamping"]
        residual = diagnostics["relativeResidual"]
    else:
        if abs(final / pre - 1.0) > 1e-12:
            raise ValueError("rejected correction changed state")
        damping = None
        residual = None
    return {
        "pre": pre,
        "final": final,
        "reference": reference,
        "pre_over_reference": pre / reference,
        "final_over_pre": final / pre,
        "final_over_reference": final / reference,
        "accepted": bool(accepted),
        "damping": damping,
        "residual": residual,
        "seconds": row["finalSharedSchurSeconds"],
    }


def geometric_mean(values):
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def summarize(rows):
    ratios = [row["final_over_pre"] for row in rows]
    return {
        "count": len(rows),
        "geometric_correction_over_pre": geometric_mean(ratios),
        "summed_correction_over_pre": math.fsum(row["final"] for row in rows)
        / math.fsum(row["pre"] for row in rows),
        "geometric_corrected_over_historical": geometric_mean(
            [row["final_over_reference"] for row in rows]
        ),
        "summed_corrected_over_historical": math.fsum(row["final"] for row in rows)
        / math.fsum(row["reference"] for row in rows),
        "wins": sum(value < 1.0 for value in ratios),
        "noops": sum(value == 1.0 for value in ratios),
        "worst": max(ratios),
        "total_correction_seconds": math.fsum(row["seconds"] for row in rows),
    }


def analyze(root):
    original = original_rows()
    details = {}
    recovery_counts = {}
    for clusters in (4, 16):
        reload_rows = load_rows(root / f"reload_k{clusters}")
        recovery_rows = load_rows(root / f"recovery_k{clusters}")
        recovered = 0
        for (scene, row_clusters), reference_row in original.items():
            if row_clusters != clusters:
                continue
            reference = reference_row["qualityMetrics"]["sumSquaredError"]
            reload_row = reload_rows.get(scene)
            if valid_reload(reload_row, reference):
                selected = reload_row
                source = "reload"
            else:
                selected = recovery_rows.get(scene)
                source = "inprocess"
                recovered += 1
            if selected is None:
                raise ValueError(f"missing selected row for {scene}/K{clusters}")
            result = correction_result(selected, reference)
            result["source"] = source
            details[(scene, clusters)] = result
        recovery_counts[str(clusters)] = recovered
    summaries = {}
    for clusters in (4, 16):
        summaries[str(clusters)] = {}
        for family, prefix in (("1dsfm", ""), ("bal", "bal")):
            rows = [
                row for (scene, row_clusters), row in details.items()
                if row_clusters == clusters and scene.startswith(prefix)
                and ((family == "bal") == scene.startswith("bal"))
            ]
            summaries[str(clusters)][family] = summarize(rows)
    damping_counts = Counter(
        str(row["damping"]) if row["damping"] is not None else "noop"
        for row in details.values()
    )
    residuals = [row["residual"] for row in details.values() if row["residual"] is not None]
    return {
        "status": "passed",
        "recovery_counts": recovery_counts,
        "summaries": summaries,
        "accepted_damping_counts": dict(sorted(damping_counts.items())),
        "maximum_accepted_residual": max(residuals),
        "scenes": {
            f"{scene}_k{clusters}": row
            for (scene, clusters), row in sorted(details.items())
        },
    }


def write_report(path, summary):
    with path.open("w", encoding="utf-8") as output:
        output.write("# Terminal Correction K4/K16 Full Transfer\n\n")
        output.write("| K | Family | Correction/pre | Summed | W/no-op | Worst | Corrected/historical | Time s |\n")
        output.write("|---:|---|---:|---:|---:|---:|---:|---:|\n")
        for clusters, families in summary["summaries"].items():
            for family, row in families.items():
                output.write(
                    f"| {clusters} | {family} | {row['geometric_correction_over_pre']:.9f} | "
                    f"{row['summed_correction_over_pre']:.9f} | {row['wins']}/{row['noops']} | "
                    f"{row['worst']:.9f} | {row['geometric_corrected_over_historical']:.9f} | "
                    f"{row['total_correction_seconds']:.3f} |\n"
                )
        output.write(
            f"\nIn-process recoveries: K4 `{summary['recovery_counts']['4']}`, "
            f"K16 `{summary['recovery_counts']['16']}`. Maximum accepted residual: "
            f"`{summary['maximum_accepted_residual']:.3e}`. Damping counts: "
            f"`{summary['accepted_damping_counts']}`.\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare-recovery", action="store_true")
    arguments = parser.parse_args()
    if arguments.prepare_recovery:
        prepare_recovery(arguments.root)
        return
    summary = analyze(arguments.root)
    (arguments.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(arguments.root / "report.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
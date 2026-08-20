#!/usr/bin/env python3
"""Analyze K4/K16 terminal-correction scaling sentinels."""

import argparse
import json
import math
import re
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent
SCENES = ("roman_forum", "trafalgar", "bal52", "bal3068")


def scene_key(row):
    dataset = Path(row["dataset"])
    match = re.search(r"problem-(\d+)-", dataset.name)
    return f"bal{match.group(1)}" if match else dataset.parent.name


def load_directory(directory):
    paths = tuple(directory.glob("*.jsonl"))
    if len(paths) != 1:
        raise ValueError(f"expected one JSONL in {directory}, found {len(paths)}")
    rows = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
    return {scene_key(row): row for row in rows}


def original_rows():
    result = {}
    for family in ("1dsfm", "bal"):
        directory = (
            WORKSPACE / "benchmark_results/stage_c_scaling_confirmation_k4_16_i30"
            / family / "c1_c5"
        )
        paths = tuple(directory.glob("*.jsonl"))
        if len(paths) != 1:
            raise ValueError(f"expected one scaling JSONL in {directory}")
        for line in paths[0].read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                result[(scene_key(row), row["clusters"])] = row
    return result


def geometric_mean(values):
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def analyze(root):
    original = original_rows()
    details = {}
    for clusters in (4, 16):
        reload_rows = load_directory(root / f"reload_k{clusters}")
        inprocess_rows = load_directory(root / f"inprocess_k{clusters}")
        rows = {**reload_rows, **inprocess_rows}
        if set(rows) != set(SCENES):
            raise ValueError(f"K{clusters} coverage mismatch: {sorted(rows)}")
        for scene, row in rows.items():
            reference = original[(scene, clusters)]["qualityMetrics"]["sumSquaredError"]
            pre = row["finalSharedSchurInitialSSE"]
            final = row["qualityMetrics"]["sumSquaredError"]
            accepted = [attempt for attempt in row["finalSharedSchurAttempts"] if attempt["accepted"]]
            if accepted:
                if len(accepted) != 1:
                    raise ValueError(f"multiple accepted corrections for {scene}/K{clusters}")
                diagnostics = accepted[0]["diagnostics"]
                if diagnostics["linearTermination"] != 0 or diagnostics["relativeResidual"] >= 1e-6:
                    raise ValueError(f"accepted nonconverged solve for {scene}/K{clusters}")
                if accepted[0]["dampedGainRatio"] <= 0.0 or final >= pre:
                    raise ValueError(f"invalid acceptance for {scene}/K{clusters}")
                damping = accepted[0]["cameraDamping"]
            else:
                if abs(final / pre - 1.0) > 1e-12:
                    raise ValueError(f"rejected correction changed {scene}/K{clusters}")
                damping = None
            details[(scene, clusters)] = {
                "source": "inprocess" if scene == "bal3068" else "reload",
                "pre_sse": pre,
                "corrected_sse": final,
                "original_sse": reference,
                "pre_over_original": pre / reference,
                "corrected_over_pre": final / pre,
                "corrected_over_original": final / reference,
                "accepted": bool(accepted),
                "accepted_damping": damping,
            }
    families = {}
    for clusters in (4, 16):
        families[str(clusters)] = {}
        for family, scenes in {
            "1dsfm": ("roman_forum", "trafalgar"),
            "bal": ("bal52", "bal3068"),
        }.items():
            rows = [details[(scene, clusters)] for scene in scenes]
            families[str(clusters)][family] = {
                "pre_over_original": geometric_mean([row["pre_over_original"] for row in rows]),
                "corrected_over_pre": geometric_mean([row["corrected_over_pre"] for row in rows]),
                "corrected_over_original": geometric_mean([row["corrected_over_original"] for row in rows]),
                "wins": sum(row["corrected_over_pre"] < 1.0 for row in rows),
                "noops": sum(row["corrected_over_pre"] == 1.0 for row in rows),
            }
    return {
        "status": "completed",
        "families": families,
        "scenes": {
            f"{scene}_k{clusters}": row
            for (scene, clusters), row in sorted(details.items())
        },
    }


def write_report(path, summary):
    with path.open("w", encoding="utf-8") as output:
        output.write("# Terminal Correction K4/K16 Sentinel Transfer\n\n")
        output.write("| K | Family | Pre/original | Corrected/pre | Corrected/original | W/no-op |\n")
        output.write("|---:|---|---:|---:|---:|---:|\n")
        for clusters, families in summary["families"].items():
            for family, row in families.items():
                output.write(
                    f"| {clusters} | {family} | {row['pre_over_original']:.9f} | "
                    f"{row['corrected_over_pre']:.9f} | {row['corrected_over_original']:.9f} | "
                    f"{row['wins']}/{row['noops']} |\n"
                )
        output.write("\n| Scene/K | Source | Pre/original | Corrected/pre | Damping |\n")
        output.write("|---|---|---:|---:|---:|\n")
        for key, row in summary["scenes"].items():
            damping = "--" if row["accepted_damping"] is None else f"{row['accepted_damping']:.9g}"
            output.write(
                f"| {key} | {row['source']} | {row['pre_over_original']:.9f} | "
                f"{row['corrected_over_pre']:.9f} | {damping} |\n"
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
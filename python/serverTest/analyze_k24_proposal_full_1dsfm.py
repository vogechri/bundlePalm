#!/usr/bin/env python3
"""Build the complete all-15 K24 proposal scoreboard."""

import argparse
import json
import math
from pathlib import Path

from analyze_k24_one_step_schur_proposal_breadth import ceres_rows, load_rows


WORKSPACE = Path(__file__).resolve().parent.parent
ARMS = {
    "control": "benchmark_results/k24_i200_terminal_correction_breadth/1dsfm/i200",
    "camera_i90": "benchmark_results/k24_one_step_schur_proposal_i90_breadth/1dsfm",
    "camera_i90_rebase": "benchmark_results/k24_one_step_schur_proposal_i90_trust_rebase_breadth/1dsfm",
    "joint_i90": "benchmark_results/k24_schur_proposal_landmark_response_breadth/1dsfm",
    "joint_i60": "benchmark_results/k24_landmark_response_proposal_i60_breadth/1dsfm",
    "joint_i60_i90": "benchmark_results/k24_landmark_response_proposal_i60_i90_breadth/1dsfm",
}


def geometric_mean(values):
    values = tuple(values)
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def compare(candidate, reference):
    ratios = {
        scene: candidate[scene] / reference[scene]
        for scene in sorted(candidate)
    }
    return {
        "geometric": geometric_mean(ratios.values()),
        "summed": math.fsum(candidate.values()) / math.fsum(reference.values()),
        "wins": sum(value < 1.0 - 1e-9 for value in ratios.values()),
        "ties": sum(abs(value - 1.0) <= 1e-9 for value in ratios.values()),
        "losses": sum(value > 1.0 + 1e-9 for value in ratios.values()),
        "worst": max(ratios.values()),
        "ratios": ratios,
    }


def analyze():
    rows = {
        name: load_rows(WORKSPACE / directory)
        for name, directory in ARMS.items()
    }
    scenes = set(rows["control"])
    if len(scenes) != 15 or any(set(arm) != scenes for arm in rows.values()):
        raise ValueError("all proposal arms must cover the same 15 scenes")
    ceres = ceres_rows()
    delivered = {}
    for name, arm in rows.items():
        delivered[name] = {
            scene: (
                min(
                    trajectory["sumSquaredError"]
                    for trajectory in arm[scene]["trajectory"][:120]
                )
                if name == "control"
                else arm[scene]["qualityMetrics"]["sumSquaredError"]
            )
            for scene in scenes
        }
    references = {scene: ceres[scene] for scene in scenes}
    summaries = {
        name: {
            "over_control": compare(values, delivered["control"]),
            "over_ceres": compare(values, references),
            "total_rejections": sum(rows[name][scene]["rejections"] for scene in scenes),
        }
        for name, values in delivered.items()
    }
    details = {}
    for scene in sorted(scenes):
        ratios = {
            name: delivered[name][scene] / references[scene]
            for name in ARMS
        }
        proposal_ratios = {
            name: value for name, value in ratios.items() if name != "control"
        }
        details[scene] = {
            "ceres_ratios": ratios,
            "best_proposal": min(proposal_ratios, key=proposal_ratios.get),
            "best_proposal_over_ceres": min(proposal_ratios.values()),
        }
    return {"summaries": summaries, "scenes": details}


def write_report(path, summary):
    with path.open("w", encoding="utf-8") as output:
        output.write("# K24 Proposal Evaluation on Full 1DSfM\n\n")
        output.write(
            "All complete K24/I120 proposal variants are compared on the same "
            "15 scenes. Control is ordinary integrated DRS through I120; Ceres "
            "is the canonical external reference.\n\n"
        )
        output.write(
            "| Arm | Delivered/control geometric | Summed/control | W/T/L vs "
            "control | Delivered/Ceres geometric | Summed/Ceres | Worst "
            "control ratio | Rejections |\n"
        )
        output.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for name in ARMS:
            row = summary["summaries"][name]
            control = row["over_control"]
            ceres = row["over_ceres"]
            output.write(
                f"| {name} | {control['geometric']:.9f} | "
                f"{control['summed']:.9f} | "
                f"{control['wins']}/{control['ties']}/{control['losses']} | "
                f"{ceres['geometric']:.9f} | {ceres['summed']:.9f} | "
                f"{control['worst']:.9f} | {row['total_rejections']} |\n"
            )
        output.write("\n## Per-Scene Ceres Ratios\n\n")
        output.write(
            "| Scene | Control | Camera I90 | Camera I90+rebase | Joint I90 | "
            "Joint I60 | Joint I60+I90 | Best proposal |\n"
        )
        output.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
        for scene, row in sorted(
            summary["scenes"].items(),
            key=lambda item: item[1]["ceres_ratios"]["joint_i60"],
            reverse=True,
        ):
            ratios = row["ceres_ratios"]
            output.write(
                f"| {scene} | {ratios['control']:.6f} | "
                f"{ratios['camera_i90']:.6f} | "
                f"{ratios['camera_i90_rebase']:.6f} | "
                f"{ratios['joint_i90']:.6f} | {ratios['joint_i60']:.6f} | "
                f"{ratios['joint_i60_i90']:.6f} | "
                f"{row['best_proposal']} ({row['best_proposal_over_ceres']:.6f}) |\n"
            )
        output.write("\n## Decision\n\n")
        output.write(
            "Joint camera/landmark response is the dominant proposal mechanism. "
            "Joint I60 is the no-loss common default. I60+I90 has the best "
            "aggregate but loses Gendarmenmarkt and remains a bounded-loss "
            "research variant. The absolute I60 quality tails are Yorkminster "
            "and Tower of London; Madrid is the sole declined 1DSfM proposal.\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    summary = analyze()
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(arguments.output / "report.md", summary)
    print(json.dumps(summary["summaries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
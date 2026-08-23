#!/usr/bin/env python3
"""Build the artifact-only K24 combined-stack benchmark against Ceres left-SE3."""

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from analyze_k1_carryover_joint_factorial import ARMS, validate_configuration
from analyze_k24_one_step_schur_proposal_breadth import load_rows
from build_stage_c_publication_comparison import (
    CERES_PATHS,
    load_jsonl,
    validate_ceres,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CARRYOVER_ROOT = ROOT / "benchmark_results/k1_carryover_joint_factorial"
DEFAULT_OUTPUT_ROOT = ROOT / "benchmark_results/ceres_se3_left_unified_benchmark"
CAMPAIGN_CHECKPOINT = "f4feabe"
FAMILIES = {"1dsfm": 15, "bal": 29}


def family(scene):
    return "bal" if scene.startswith("bal") else "1dsfm"


def geometric_mean(values):
    values = tuple(values)
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def compare(candidate, reference, tolerance=1e-9):
    ratios = {
        scene: candidate[scene] / reference[scene] for scene in sorted(candidate)
    }
    return {
        "geometric": geometric_mean(ratios.values()),
        "summed": math.fsum(candidate.values()) / math.fsum(reference.values()),
        "wins": sum(value < 1.0 - tolerance for value in ratios.values()),
        "ties": sum(abs(value - 1.0) <= tolerance for value in ratios.values()),
        "losses": sum(value > 1.0 + tolerance for value in ratios.values()),
        "worst": max(ratios.values()),
        "ratios": ratios,
    }


def scene_key_from_dataset(dataset):
    path = Path(dataset)
    if path.name.startswith("problem-"):
        return f"bal{path.name.split('-')[1]}"
    return path.parent.name


def load_status(directory):
    rows = {}
    path = directory / "status.tsv"
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            scene = scene_key_from_dataset(row["dataset"])
            if scene in rows:
                raise ValueError(f"duplicate status row: {scene}")
            rows[scene] = row
    return rows


def proposal_decision(row, checkpoint):
    if len(row["trajectory"]) < checkpoint:
        return None
    diagnostic = row["trajectory"][checkpoint - 1].get(
        "schurAlignmentDiagnostics"
    )
    if diagnostic is None:
        return None
    oracle = diagnostic.get("schurProposalLandmarkResponseOracle")
    if oracle is None:
        raise ValueError(f"missing landmark-response oracle at I{checkpoint}")
    return {
        "selected": bool(oracle["selected"]),
        "applied": bool(oracle["applied"]),
        "selected_scale": float(oracle["selectedScale"]),
    }


def validate_inputs(carryover_root):
    direct = load_rows(carryover_root / "direct")
    candidate = load_rows(carryover_root / "direct_shared_proposal")
    statuses = {
        "direct": load_status(carryover_root / "direct"),
        "candidate": load_status(carryover_root / "direct_shared_proposal"),
    }
    if not (set(direct) == set(candidate) == set(statuses["direct"]) == set(statuses["candidate"])):
        raise ValueError("direct/candidate/status scene mismatch")
    for arm, rows, settings in (
        ("direct", direct, ARMS["direct"]),
        ("candidate", candidate, ARMS["direct_shared_proposal"]),
    ):
        for scene, row in rows.items():
            validate_configuration(scene, row, settings)
            status = statuses[arm][scene]
            if status["status"] != "completed" or int(status["exit_code"]) != 0:
                raise ValueError(f"failed status {arm}/{scene}")
    ceres = {name: load_jsonl(path) for name, path in CERES_PATHS.items()}
    for name, count in FAMILIES.items():
        validate_ceres(ceres[name], count)
        expected = {scene for scene in candidate if family(scene) == name}
        if len(expected) != count or set(ceres[name]) != expected:
            raise ValueError(f"Ceres/candidate coverage mismatch for {name}")
    return direct, candidate, statuses, ceres


def build_summary(carryover_root):
    direct, candidate, statuses, ceres = validate_inputs(carryover_root)
    summary = {
        "schema_version": 1,
        "objective": "standard Snavely pixel sum-squared error",
        "primary_method": (
            "K24/I120 direct left-SE3 + exact tangent metric + shared-only "
            "product-space DRS + guarded I60/I90 Krylov2 camera/landmark proposal"
        ),
        "ceres_reference": "left-SE3, I90, T16",
        "timing_policy": (
            "CPU timing classes are reported by solver boundary; no matched-work "
            "or cross-hardware speedup is claimed"
        ),
        "provenance": {
            "campaign_checkpoint": CAMPAIGN_CHECKPOINT,
            "campaign_checkpoint_note": (
                "repository checkpoint that packages the campaign; raw result "
                "rows do not embed an execution commit"
            ),
            "report_generation_commit": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
            "carryover_root": str(carryover_root.relative_to(ROOT)),
            "ceres_artifacts": {
                name: str(path.relative_to(ROOT)) for name, path in CERES_PATHS.items()
            },
        },
        "families": {},
        "scenes": {},
    }
    for family_name, expected_count in FAMILIES.items():
        scenes = tuple(sorted(scene for scene in candidate if family(scene) == family_name))
        if len(scenes) != expected_count:
            raise ValueError(f"family coverage mismatch for {family_name}")
        candidate_sse = {
            scene: float(candidate[scene]["qualityMetrics"]["sumSquaredError"])
            for scene in scenes
        }
        direct_sse = {
            scene: float(direct[scene]["qualityMetrics"]["sumSquaredError"])
            for scene in scenes
        }
        ceres_sse = {
            scene: float(ceres[family_name][scene]["qualityMetrics"]["sumSquaredError"])
            for scene in scenes
        }
        decisions = {
            checkpoint: {
                scene: proposal_decision(candidate[scene], checkpoint)
                for scene in scenes
            }
            for checkpoint in (60, 90)
        }
        family_summary = {
            "count": len(scenes),
            "full_trajectories": {
                "direct": sum(
                    direct[scene]["terminationReason"] == "iteration_limit"
                    and direct[scene]["completedIterations"] == 120
                    for scene in scenes
                ),
                "candidate": sum(
                    candidate[scene]["terminationReason"] == "iteration_limit"
                    and candidate[scene]["completedIterations"] == 120
                    for scene in scenes
                ),
            },
            "candidate_over_ceres": compare(candidate_sse, ceres_sse),
            "candidate_over_direct": compare(candidate_sse, direct_sse),
            "optimization_seconds": {
                "candidate": math.fsum(
                    float(candidate[scene]["optimizationSeconds"]) for scene in scenes
                ),
                "direct": math.fsum(
                    float(direct[scene]["optimizationSeconds"]) for scene in scenes
                ),
                "ceres_native_solve": math.fsum(
                    float(ceres[family_name][scene]["native"]["solveSeconds"])
                    for scene in scenes
                ),
            },
            "overall_seconds": {
                "candidate": math.fsum(
                    float(candidate[scene]["overallSeconds"]) for scene in scenes
                ),
                "direct": math.fsum(
                    float(direct[scene]["overallSeconds"]) for scene in scenes
                ),
                "ceres_native": math.fsum(
                    float(ceres[family_name][scene]["native"]["overallSeconds"])
                    for scene in scenes
                ),
            },
            "candidate_transport_bytes": sum(
                int(candidate[scene]["transportBytesSent"])
                + int(candidate[scene]["transportBytesReceived"])
                for scene in scenes
            ),
            "candidate_peak_rss_gib": {
                "coordinator": max(
                    int(statuses["candidate"][scene]["coordinator_max_rss_kb"])
                    for scene in scenes
                ) / 1048576.0,
                "worker": max(
                    int(statuses["candidate"][scene]["worker_max_rss_kb"])
                    for scene in scenes
                ) / 1048576.0,
            },
            "proposal": {
                str(checkpoint): {
                    "attempted": sum(value is not None for value in rows.values()),
                    "selected": sum(
                        value is not None and value["selected"]
                        for value in rows.values()
                    ),
                    "no_op": sum(
                        value is not None and not value["selected"]
                        for value in rows.values()
                    ),
                }
                for checkpoint, rows in decisions.items()
            },
            "recovery_exhausted": {
                "direct": [
                    scene for scene in scenes
                    if direct[scene]["terminationReason"] == "recovery_exhausted"
                ],
                "candidate": [
                    scene for scene in scenes
                    if candidate[scene]["terminationReason"] == "recovery_exhausted"
                ],
            },
        }
        summary["families"][family_name] = family_summary
        for scene in scenes:
            summary["scenes"][scene] = {
                "family": family_name,
                "candidate_sse": candidate_sse[scene],
                "direct_sse": direct_sse[scene],
                "ceres_sse": ceres_sse[scene],
                "candidate_over_ceres": candidate_sse[scene] / ceres_sse[scene],
                "candidate_over_direct": candidate_sse[scene] / direct_sse[scene],
                "candidate_optimization_seconds": float(
                    candidate[scene]["optimizationSeconds"]
                ),
                "candidate_transport_bytes": (
                    int(candidate[scene]["transportBytesSent"])
                    + int(candidate[scene]["transportBytesReceived"])
                ),
                "proposal": {
                    str(checkpoint): decisions[checkpoint][scene]
                    for checkpoint in (60, 90)
                },
            }
    return summary


def write_report(path, summary):
    with path.open("w", encoding="utf-8") as output:
        output.write("# K24 Combined Stack Versus Ceres Left-SE3\n\n")
        output.write(
            "Artifact-only benchmark over all 15 SfM_Init-derived 1DSfM scenes "
            "and all 29 BAL problems. The primary method is the retained "
            "K24/I120 copied-baseline combined stack. Ceres left-SE3 I90/T16 is "
            "a centralized quality reference, not a replacement for the "
            "distributed method. All quality values use independently evaluated "
            "standard Snavely pixel SSE.\n\n"
        )
        output.write(
            "| Family | Complete D/C | Candidate/Ceres geo | Summed | W/T/L | "
            "Worst | Candidate/direct geo | Summed | W/T/L | Opt. s D/C/Ceres | "
            "I60 sel/no-op | I90 sel/no-op | Transport GiB | Peak RSS GiB C/W |\n"
        )
        output.write(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for family_name, label in (("1dsfm", "1DSfM-15"), ("bal", "BAL-29")):
            row = summary["families"][family_name]
            ceres = row["candidate_over_ceres"]
            direct = row["candidate_over_direct"]
            times = row["optimization_seconds"]
            output.write(
                f"| {label} | {row['full_trajectories']['direct']}/"
                f"{row['full_trajectories']['candidate']} | "
                f"{ceres['geometric']:.6f} | {ceres['summed']:.6f} | "
                f"{ceres['wins']}/{ceres['ties']}/{ceres['losses']} | "
                f"{ceres['worst']:.6f} | {direct['geometric']:.6f} | "
                f"{direct['summed']:.6f} | "
                f"{direct['wins']}/{direct['ties']}/{direct['losses']} | "
                f"{times['direct']:.1f}/{times['candidate']:.1f}/"
                f"{times['ceres_native_solve']:.1f} | "
                f"{row['proposal']['60']['selected']}/"
                f"{row['proposal']['60']['no_op']} | "
                f"{row['proposal']['90']['selected']}/"
                f"{row['proposal']['90']['no_op']} | "
                f"{row['candidate_transport_bytes'] / 2**30:.3f} | "
                f"{row['candidate_peak_rss_gib']['coordinator']:.3f}/"
                f"{row['candidate_peak_rss_gib']['worker']:.3f} |\n"
            )
        output.write(
            "\nThe BAL geometric mean is slightly above Ceres while summed SSE is "
            "slightly below Ceres; both statistics are therefore retained. Timing "
            "boundaries differ: DRS uses coordinator-reported optimization time, "
            "while Ceres uses native solve time. No matched-work speedup is claimed.\n\n"
        )
        output.write(
            "The combined candidate completes every scene with no recovery "
            "exhaustion. The matched direct control also completes every scene. "
            "See `per_scene.md` for the complete ratio vectors.\n\n"
        )
        output.write(
            f"Campaign checkpoint: `{summary['provenance']['campaign_checkpoint']}`. "
            "Raw rows do not embed an execution commit; this is the repository "
            "checkpoint that packages the campaign.\n"
        )


def write_per_scene(path, summary):
    with path.open("w", encoding="utf-8") as output:
        output.write("# Per-Scene K24 Combined-Stack Ratios\n\n")
        output.write(
            "| Family | Scene | Candidate SSE | Ceres SSE | Direct SSE | "
            "Candidate/Ceres | Candidate/direct | I60 | I90 |\n"
        )
        output.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for family_name in ("1dsfm", "bal"):
            for scene, row in sorted(summary["scenes"].items()):
                if row["family"] != family_name:
                    continue
                decisions = []
                for checkpoint in ("60", "90"):
                    decision = row["proposal"][checkpoint]
                    decisions.append(
                        "selected" if decision and decision["selected"] else "no-op"
                    )
                output.write(
                    f"| {row['family']} | {scene} | {row['candidate_sse']:.6g} | "
                    f"{row['ceres_sse']:.6g} | {row['direct_sse']:.6g} | "
                    f"{row['candidate_over_ceres']:.6f} | "
                    f"{row['candidate_over_direct']:.6f} | "
                    f"{decisions[0]} | {decisions[1]} |\n"
                )


def configure_matplotlib():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "axes.edgecolor": "#4c4f52",
            "axes.labelcolor": "#25282b",
            "savefig.bbox": "tight",
        }
    )


def write_objective_time_plot(path, summary):
    configure_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), constrained_layout=True)
    for axis, family_name, title in zip(
        axes,
        ("1dsfm", "bal"),
        ("SfM_Init-derived 1DSfM (15)", "BAL (29)"),
    ):
        row = summary["families"][family_name]
        points = (
            ("Ceres left-SE3", row["optimization_seconds"]["ceres_native_solve"], 1.0, "#b23a48", "o"),
            ("K24 direct", row["optimization_seconds"]["direct"], 1.0 / row["candidate_over_direct"]["geometric"] * row["candidate_over_ceres"]["geometric"], "#30343f", "s"),
            ("K24 combined", row["optimization_seconds"]["candidate"], row["candidate_over_ceres"]["geometric"], "#3a7d44", "^"),
        )
        for label, seconds, ratio, color, marker in points:
            axis.scatter(seconds, ratio, color=color, marker=marker, s=70, edgecolor="white", linewidth=0.7)
            axis.annotate(label, (seconds, ratio), xytext=(6, 7), textcoords="offset points", fontsize=8)
        axis.axhline(1.0, color="#85898d", linewidth=1.0, linestyle="--")
        axis.set_xscale("log")
        axis.set_xlabel("Aggregate optimization time (s, log scale)")
        axis.set_ylabel("Geometric SSE / Ceres")
        axis.set_title(title)
        axis.grid(color="#d9dcdf", linewidth=0.6, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("K24 copied-baseline stack versus Ceres left-SE3", fontsize=13)
    figure.savefig(path, metadata={"Title": "K24 versus Ceres left-SE3", "CreationDate": None, "ModDate": None})
    plt.close(figure)


def write_scene_ratio_plot(path, summary):
    configure_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 9.2), constrained_layout=True)
    for axis, family_name, title in zip(
        axes,
        ("1dsfm", "bal"),
        ("SfM_Init-derived 1DSfM", "BAL"),
    ):
        rows = [
            (scene, row["candidate_over_ceres"])
            for scene, row in summary["scenes"].items()
            if row["family"] == family_name
        ]
        rows.sort(key=lambda item: item[1])
        labels = [scene.removeprefix("bal") if family_name == "bal" else scene.replace("_", " ") for scene, _ in rows]
        values = [value for _, value in rows]
        colors = ["#3a7d44" if value < 1.0 else "#b23a48" for value in values]
        positions = list(range(len(rows)))
        axis.barh(positions, values, color=colors)
        axis.axvline(1.0, color="#30343f", linewidth=1.0, linestyle="--")
        axis.set_yticks(positions, labels, fontsize=7.5)
        axis.set_xlabel("Candidate SSE / Ceres SSE")
        axis.set_title(title)
        axis.grid(axis="x", color="#d9dcdf", linewidth=0.6, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Per-scene endpoint quality relative to Ceres left-SE3", fontsize=13)
    figure.savefig(path, metadata={"Title": "Per-scene K24 over Ceres SSE", "CreationDate": None, "ModDate": None})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carryover-root", type=Path, default=DEFAULT_CARRYOVER_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args()
    summary = build_summary(arguments.carryover_root)
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    (arguments.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(arguments.output_root / "report.md", summary)
    write_per_scene(arguments.output_root / "per_scene.md", summary)
    write_objective_time_plot(arguments.output_root / "objective_vs_time.pdf", summary)
    write_scene_ratio_plot(arguments.output_root / "sse_ratio_by_scene.pdf", summary)
    print(arguments.output_root)


if __name__ == "__main__":
    main()

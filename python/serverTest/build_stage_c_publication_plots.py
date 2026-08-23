#!/usr/bin/env python3
"""Build publication figures from the authoritative Stage-C summary."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from stage_c_publication_contract import PUBLICATION_PANELS


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUMMARY = (
    ROOT / "benchmark_results/stage_c_publication_comparison/summary.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "benchmark_results/stage_c_publication_comparison/"
    "objective_vs_optimization_time.pdf"
)
DEFAULT_REPEAT_SUMMARY = (
    ROOT / "benchmark_results/stage_c_scaling_repeats_k4_16_i30/summary.json"
)
DEFAULT_CORRECTION_SUMMARY = (
    ROOT / "benchmark_results/terminal_correction_scaling_all/summary.json"
)
DEFAULT_MANIFEST = ROOT / "benchmark_results/stage_c_reproducibility_manifest.json"
DEFAULT_RESOURCE_OUTPUT = (
    ROOT
    / "benchmark_results/stage_c_publication_comparison/"
    "k16_over_k4_resources.pdf"
)
DEFAULT_OUTER_TRIAL_OUTPUT = (
    ROOT
    / "benchmark_results/stage_c_publication_comparison/"
    "outer_iteration_outcomes.pdf"
)
PANELS = {
    "all15_1dsfm": {
        "title": "SfM_Init-derived 1DSfM (15 scenes)",
    },
    "all29_bal": {
        "title": "BAL (29 scenes)",
    },
}
STYLES = {
    "Ceres": {"color": "#b23a48", "marker": "o", "label": "Ceres"},
    "K1 diagnostic": {
        "color": "#00798c",
        "marker": "D",
        "label": "K1 diagnostic",
    },
    "DRS only": {"color": "#30343f", "marker": "s", "label": "DRS only"},
    "DRS + Schur": {
        "color": "#3a7d44",
        "marker": "^",
        "label": "DRS + Schur",
    },
    "Terminal correction": {
        "color": "#d17b0f",
        "marker": "X",
        "label": "Terminal correction",
    },
}
ANNOTATION_OFFSETS = {
    "all15_1dsfm": {
        "Ceres": (5, 7),
        "DRS K1 BAE-style": (-4, -15),
        "DRS K1 Schur-PCG": (-10, 8),
        "Base DRS K24": (7, 7),
        "DRS+Schur fast": (7, -15),
        "DRS+Schur balanced": (7, 7),
        "DRS+Schur quality": (7, 7),
        "DRS K4": (7, 7),
        "DRS K16": (7, 7),
        "DRS K4 + terminal correction": (7, -15),
        "DRS K16 + terminal correction": (7, -18),
    },
    "all29_bal": {
        "Ceres": (5, 7),
        "Base DRS K24": (7, 7),
        "DRS K4": (7, 7),
        "DRS K16": (-58, 9),
        "DRS K4 + terminal correction": (7, -15),
        "DRS K16 + terminal correction": (-82, -15),
    },
}
DISPLAY_LABELS = {
    "DRS K1 BAE-style": "K1 BAE-style",
    "DRS K1 Schur-PCG": "K1 Schur-PCG",
    "Base DRS K24": "Base K24",
    "DRS+Schur fast": "Schur fast",
    "DRS+Schur balanced": "Schur balanced",
    "DRS+Schur quality": "Schur quality",
    "DRS K4": "K4",
    "DRS K16": "K16",
    "DRS K4 + terminal correction": "K4 + correction",
    "DRS K16 + terminal correction": "K16 + correction",
}


def method_class(label):
    if label == "Ceres":
        return "Ceres"
    if "terminal correction" in label:
        return "Terminal correction"
    if label in ("DRS K1 BAE-style", "DRS K1 Schur-PCG"):
        return "K1 diagnostic"
    if label.startswith("DRS+Schur"):
        return "DRS + Schur"
    return "DRS only"


def load_panels(path):
    summary = json.loads(path.read_text(encoding="utf-8"))
    panels = {}
    for key, contract in PANELS.items():
        coverage = PUBLICATION_PANELS[key]
        rows = summary.get(key)
        if not isinstance(rows, list):
            raise ValueError(f"missing publication panel: {key}")
        labels = tuple(row.get("label") for row in rows)
        if labels != coverage["labels"]:
            raise ValueError(f"publication method coverage mismatch for {key}")
        for row in rows:
            if row.get("scenes") != coverage["scenes"]:
                raise ValueError(f"publication scene coverage mismatch for {key}")
            for field in ("optimization_seconds", "sse_vs_ceres_geometric"):
                value = row.get(field)
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ValueError(f"invalid {field} for {key}/{row.get('label')}")
        panels[key] = rows
    return panels


def build_figure(panels, output):
    mpl.rcParams.update(
        {
            "axes.edgecolor": "#4c4f52",
            "axes.labelcolor": "#25282b",
            "axes.titleweight": "bold",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
            "svg.hashsalt": "bundle-palm-stage-c",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    for axis, (key, contract) in zip(axes, PANELS.items()):
        rows = panels[key]
        for row in rows:
            label = row["label"]
            style = STYLES[method_class(label)]
            axis.scatter(
                row["optimization_seconds"],
                row["sse_vs_ceres_geometric"],
                color=style["color"],
                marker=style["marker"],
                edgecolor="white",
                linewidth=0.7,
                s=64,
                zorder=3,
            )
            axis.annotate(
                DISPLAY_LABELS.get(label, label),
                (row["optimization_seconds"], row["sse_vs_ceres_geometric"]),
                xytext=ANNOTATION_OFFSETS[key][label],
                textcoords="offset points",
                fontsize=7.5,
                color="#25282b",
            )
        axis.axhline(1.0, color="#85898d", linewidth=1.0, linestyle="--", zorder=1)
        axis.set_xscale("log")
        minimum_time = min(row["optimization_seconds"] for row in rows)
        maximum_time = max(row["optimization_seconds"] for row in rows)
        axis.set_xlim(minimum_time / 1.25, maximum_time * 1.28)
        axis.grid(axis="both", color="#d9dcdf", linewidth=0.6, alpha=0.8, zorder=0)
        axis.set_title(contract["title"])
        axis.set_xlabel("Aggregate optimization time (s, log scale)")
        axis.set_ylabel("Geometric SSE / Ceres")
        axis.spines[["top", "right"]].set_visible(False)

    handles = [
        mpl.lines.Line2D(
            [],
            [],
            color=style["color"],
            marker=style["marker"],
            linestyle="None",
            markeredgecolor="white",
            markersize=7,
            label=style["label"],
        )
        for style in STYLES.values()
    ]
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(handles),
        frameon=False,
    )
    figure.suptitle("Endpoint quality versus measured optimization time", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        metadata={
            "Title": "Stage-C endpoint quality versus optimization time",
            "Author": "BundlePalm",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def load_resource_ratios(repeat_path, correction_path):
    repeat = json.loads(repeat_path.read_text(encoding="utf-8"))
    correction = json.loads(correction_path.read_text(encoding="utf-8"))
    repeat_rows = repeat.get("endpoint_ratios")
    repeat_cases = repeat.get("cases")
    if not isinstance(repeat_rows, list) or not isinstance(repeat_cases, list):
        raise ValueError("invalid scaling-repeat summary")
    if {row.get("family") for row in repeat_rows} != {"1dsfm", "bal"}:
        raise ValueError("scaling-repeat family coverage mismatch")
    if len(repeat_cases) != 8:
        raise ValueError("scaling-repeat case coverage mismatch")

    repeated = {}
    for family in ("1dsfm", "bal"):
        row = next(item for item in repeat_rows if item["family"] == family)
        by_cluster = {
            clusters: [
                item
                for item in repeat_cases
                if item.get("family") == family and item.get("clusters") == clusters
            ]
            for clusters in (4, 16)
        }
        if any(len(rows) != 2 for rows in by_cluster.values()):
            raise ValueError(f"scaling-repeat scene coverage mismatch for {family}")
        repeated[family] = {
            "Optimization": row["optimization_ratio_mean"],
            "Worker CPU": row["worker_cpu_ratio_mean"],
            "Transport": row["traffic_ratio_mean"],
            "Worker RSS": (
                sum(item["worker_rss_mean_kb"] for item in by_cluster[16])
                / sum(item["worker_rss_mean_kb"] for item in by_cluster[4])
            ),
        }

    summaries = correction.get("summaries")
    quality = correction.get("corrected_k16_over_k4")
    if not isinstance(summaries, dict) or not isinstance(quality, dict):
        raise ValueError("invalid terminal-correction summary")
    corrected = {}
    for family in ("1dsfm", "bal"):
        k4 = summaries.get("4", {}).get(family)
        k16 = summaries.get("16", {}).get(family)
        family_quality = quality.get(family)
        if not all(isinstance(item, dict) for item in (k4, k16, family_quality)):
            raise ValueError(f"terminal-correction coverage mismatch for {family}")
        corrected[family] = {
            "Corrected SSE": family_quality["geometric"],
            "Correction time": (
                k16["total_correction_seconds"] / k4["total_correction_seconds"]
            ),
            "Peak coordinator RSS": (
                k16["maximum_coordinator_rss_gib"]
                / k4["maximum_coordinator_rss_gib"]
            ),
            "Peak worker RSS": (
                k16["maximum_worker_rss_gib"] / k4["maximum_worker_rss_gib"]
            ),
        }
    return repeated, corrected


def build_resource_figure(repeated, corrected, output):
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    panels = (
        (
            axes[0],
            repeated,
            "Frozen DRS sentinels (3-repeat mean)",
            ("Optimization", "Worker CPU", "Transport", "Worker RSS"),
        ),
        (
            axes[1],
            corrected,
            "Terminal correction (complete cohorts)",
            (
                "Corrected SSE",
                "Correction time",
                "Peak coordinator RSS",
                "Peak worker RSS",
            ),
        ),
    )
    family_colors = {"1dsfm": "#00798c", "bal": "#d17b0f"}
    family_labels = {"1dsfm": "1DSfM", "bal": "BAL"}
    for axis, values, title, metrics in panels:
        positions = list(range(len(metrics)))
        width = 0.34
        for offset, family in ((-width / 2, "1dsfm"), (width / 2, "bal")):
            bars = axis.bar(
                [position + offset for position in positions],
                [values[family][metric] for metric in metrics],
                width=width,
                color=family_colors[family],
                label=family_labels[family],
            )
            axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=7.5)
        axis.axhline(1.0, color="#85898d", linewidth=1.0, linestyle="--")
        axis.set_xticks(positions, metrics, rotation=18, ha="right")
        axis.set_ylabel("K16 / K4 ratio")
        axis.set_title(title)
        axis.grid(axis="y", color="#d9dcdf", linewidth=0.6, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylim(0.0, max(2.0, axis.get_ylim()[1] * 1.08))
    axes[0].legend(frameon=False, loc="upper left")
    figure.suptitle("K16 latency tradeoffs relative to K4 resource mode", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        metadata={
            "Title": "K16 over K4 quality and resource ratios",
            "Author": "BundlePalm",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def load_outer_outcomes(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    confirmation = manifest.get("final_cross_family_confirmation")
    if not isinstance(confirmation, dict):
        raise ValueError("missing final cross-family confirmation")
    outcomes = {}
    for family, expected_scenes in (("1dsfm", 15), ("bal", 29)):
        methods = confirmation.get(family, {}).get("methods")
        if not isinstance(methods, dict) or set(methods) != {
            "plain", "c1", "c5", "c1_c5"
        }:
            raise ValueError(f"outer-outcome method coverage mismatch for {family}")
        outcomes[family] = {}
        for method, record in methods.items():
            rows = record.get("rows")
            iterations = record.get("configuration", {}).get("iterations")
            if not isinstance(rows, dict) or len(rows) != expected_scenes:
                raise ValueError(f"outer-outcome scene coverage mismatch for {family}/{method}")
            if not isinstance(iterations, int) or iterations <= 0:
                raise ValueError(f"invalid iteration budget for {family}/{method}")
            rejected = sum(int(row["rejections"]) for row in rows.values())
            total = expected_scenes * iterations
            if rejected < 0 or rejected > total:
                raise ValueError(f"invalid rejection count for {family}/{method}")
            outcomes[family][method] = {
                "accepted": total - rejected,
                "rejected": rejected,
                "total": total,
            }
    return outcomes


def build_outer_outcome_figure(outcomes, output):
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), constrained_layout=True)
    labels = ("Plain", "C1", "C5", "C1+C5")
    methods = ("plain", "c1", "c5", "c1_c5")
    for axis, family, title in zip(
        axes,
        ("1dsfm", "bal"),
        ("SfM_Init-derived 1DSfM (15 scenes)", "BAL (29 scenes)"),
    ):
        accepted = [
            100.0 * outcomes[family][method]["accepted"]
            / outcomes[family][method]["total"]
            for method in methods
        ]
        rejected = [100.0 - value for value in accepted]
        positions = list(range(len(methods)))
        axis.bar(positions, accepted, color="#3a7d44", label="Accepted")
        bars = axis.bar(
            positions,
            rejected,
            bottom=accepted,
            color="#b23a48",
            label="Rejected",
        )
        axis.bar_label(
            bars,
            labels=[
                f"{outcomes[family][method]['rejected']} rejected"
                for method in methods
            ],
            label_type="center",
            color="white",
            fontsize=7.5,
        )
        axis.set_xticks(positions, labels)
        axis.set_ylim(0, 100)
        axis.set_ylabel("Displayed outer iterations (%)")
        axis.set_title(title)
        axis.grid(axis="y", color="#d9dcdf", linewidth=0.6, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="lower left")
    figure.suptitle("Accepted and rejected Stage-C outer iterations", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        metadata={
            "Title": "Accepted and rejected Stage-C outer iterations",
            "Author": "BundlePalm",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--repeat-summary", type=Path, default=DEFAULT_REPEAT_SUMMARY
    )
    parser.add_argument(
        "--correction-summary", type=Path, default=DEFAULT_CORRECTION_SUMMARY
    )
    parser.add_argument(
        "--resource-output", type=Path, default=DEFAULT_RESOURCE_OUTPUT
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--outer-trial-output", type=Path, default=DEFAULT_OUTER_TRIAL_OUTPUT
    )
    arguments = parser.parse_args()
    build_figure(load_panels(arguments.summary), arguments.output)
    build_resource_figure(
        *load_resource_ratios(
            arguments.repeat_summary, arguments.correction_summary
        ),
        arguments.resource_output,
    )
    build_outer_outcome_figure(
        load_outer_outcomes(arguments.manifest), arguments.outer_trial_output
    )
    print(arguments.output)
    print(arguments.resource_output)
    print(arguments.outer_trial_output)


if __name__ == "__main__":
    main()

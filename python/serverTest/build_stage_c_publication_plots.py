#!/usr/bin/env python3
"""Build publication figures from the authoritative Stage-C summary."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUMMARY = (
    ROOT / "benchmark_results/stage_c_publication_comparison/summary.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "benchmark_results/stage_c_publication_comparison/"
    "objective_vs_optimization_time.pdf"
)
PANELS = {
    "all15_1dsfm": {
        "title": "SfM_Init-derived 1DSfM (15 scenes)",
        "expected_scenes": 15,
        "expected_labels": (
            "Ceres",
            "DRS K1 BAE-style",
            "DRS K1 Schur-PCG",
            "Base DRS K24",
            "DRS+Schur fast",
            "DRS+Schur balanced",
            "DRS+Schur quality",
            "DRS K4",
            "DRS K16",
            "DRS K4 + terminal correction",
            "DRS K16 + terminal correction",
        ),
    },
    "all29_bal": {
        "title": "BAL (29 scenes)",
        "expected_scenes": 29,
        "expected_labels": (
            "Ceres",
            "Base DRS K24",
            "DRS K4",
            "DRS K16",
            "DRS K4 + terminal correction",
            "DRS K16 + terminal correction",
        ),
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
        rows = summary.get(key)
        if not isinstance(rows, list):
            raise ValueError(f"missing publication panel: {key}")
        labels = tuple(row.get("label") for row in rows)
        if labels != contract["expected_labels"]:
            raise ValueError(f"publication method coverage mismatch for {key}")
        for row in rows:
            if row.get("scenes") != contract["expected_scenes"]:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    build_figure(load_panels(arguments.summary), arguments.output)
    print(arguments.output)


if __name__ == "__main__":
    main()

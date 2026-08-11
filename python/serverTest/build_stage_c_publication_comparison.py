#!/usr/bin/env python3
"""Build cohort-explicit K1/K4/K16, Ceres, and BAE publication tables."""

import argparse
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmark_results"
K1_PATH = (
    RESULTS
    / "1dsfm_k1_schur_pcg_t24_all15_i80/custom/"
    "proximal_point_block_se3_left_lip0.00625_curvature_persistent_tr.jsonl"
)
SCALING_ROOT = RESULTS / "stage_c_scaling_confirmation_k4_16_i30"
CERES_PATHS = {
    "1dsfm": RESULTS / "1dsfm_drs_ceres_se3_all15/ceres/results.jsonl",
    "bal": RESULTS / "bal_ceres_se3_all29/results.jsonl",
}
BAE_PATHS = {
    "BAE Schur-PCG CG": RESULTS / "sfm_init_1dsfm_bae_exported/results.jsonl",
    "BAE Schur-PCG Nesterov": (
        RESULTS / "sfm_init_1dsfm_bae_k1_verified/results.jsonl"
    ),
}
BAE_SCENES = (
    "gendarmenmarkt",
    "piccadilly",
    "roman_forum",
    "trafalgar",
    "union_square",
    "vienna_cathedral",
)


def geometric_mean(values):
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def scene_key(row):
    if row.get("scene"):
        return row["scene"]
    if row.get("balId") is not None:
        return f"bal{row['balId']}"
    dataset = Path(row["dataset"])
    if "sfm_init_1dsfm" in row["dataset"]:
        return dataset.parent.name
    match = re.search(r"problem-(\d+)-", dataset.name)
    if not match:
        raise ValueError(f"cannot identify dataset: {dataset}")
    return f"bal{match.group(1)}"


def load_jsonl(path):
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


def load_scaling(family):
    directory = SCALING_ROOT / family / "c1_c5"
    paths = sorted(directory.glob("*.jsonl"))
    if len(paths) != 1:
        raise ValueError(f"expected one scaling JSONL in {directory}, found {len(paths)}")
    rows = {4: {}, 16: {}}
    for line in paths[0].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cluster = row.get("clusters")
        if cluster not in rows:
            raise ValueError(f"unexpected K{cluster} in {paths[0]}")
        scene = scene_key(row)
        if scene in rows[cluster]:
            raise ValueError(f"duplicate {scene}/K{cluster} in {paths[0]}")
        if row.get("iterations") != 30 or row.get("completedIterations") != 30:
            raise ValueError(f"incomplete scaling row {scene}/K{cluster}")
        if row.get("objectiveLoss", "l2") != "l2":
            raise ValueError(f"non-L2 scaling row {scene}/K{cluster}")
        rows[cluster][scene] = row
    return rows, paths[0]


def validate_k1(rows):
    if len(rows) != 15:
        raise ValueError(f"K1 coverage mismatch: {len(rows)}/15")
    for scene, row in rows.items():
        if not (
            row.get("clusters") == 1
            and row.get("iterations") == 80
            and row.get("completedIterations") == 80
            and row.get("threadsPerCluster") == 24
        ):
            raise ValueError(f"K1 configuration mismatch for {scene}")


def validate_ceres(rows, expected):
    if len(rows) != expected:
        raise ValueError(f"Ceres coverage mismatch: {len(rows)}/{expected}")
    for scene, row in rows.items():
        native = row.get("native", {})
        if not (
            row.get("status") == "completed"
            and native.get("cameraMode") == "se3_left"
            and native.get("maxIterations") == 90
            and native.get("threads") == 16
        ):
            raise ValueError(f"Ceres configuration mismatch for {scene}")
        relative_error = abs(
            row["qualityMetrics"]["sumSquaredError"]
            - 2.0 * native["finalCeresCost"]
        ) / max(abs(row["qualityMetrics"]["sumSquaredError"]), float.fromhex("0x1p-1022"))
        if relative_error >= 1e-6:
            raise ValueError(f"Ceres objective mismatch for {scene}: {relative_error}")


def validate_bae(rows, label):
    if tuple(sorted(rows)) != tuple(sorted(BAE_SCENES)):
        raise ValueError(f"{label} scene coverage mismatch: {sorted(rows)}")
    for scene, row in rows.items():
        if not (
            row.get("status") == "completed"
            and row.get("iterations") == 90
            and row.get("objectiveVerifiedFromSavedState") is True
        ):
            raise ValueError(f"{label} verification mismatch for {scene}")
        relative_error = abs(
            row["verifiedSumSquaredError"] - row["finalSumSquaredError"]
        ) / max(abs(row["verifiedSumSquaredError"]), float.fromhex("0x1p-1022"))
        if relative_error >= 1e-8:
            raise ValueError(f"{label} objective mismatch for {scene}: {relative_error}")


def sse(row, kind="drs"):
    if kind == "bae":
        return row["verifiedSumSquaredError"]
    return row["qualityMetrics"]["sumSquaredError"]


def summarize(label, rows, reference, scenes, kind, specification, timing_class):
    ratios = [sse(rows[scene], kind) / sse(reference[scene]) for scene in scenes]
    wins = sum(value < 1.0 and not math.isclose(value, 1.0, rel_tol=1e-12) for value in ratios)
    ties = sum(math.isclose(value, 1.0, rel_tol=1e-12) for value in ratios)
    if kind == "bae":
        optimization_seconds = math.fsum(rows[scene]["optimizationSeconds"] for scene in scenes)
        overall_seconds = None
        peak_memory_mib = max(rows[scene]["peakCudaMemoryMiB"] for scene in scenes)
    elif kind == "ceres":
        optimization_seconds = math.fsum(rows[scene]["native"]["solveSeconds"] for scene in scenes)
        overall_seconds = math.fsum(rows[scene]["native"]["overallSeconds"] for scene in scenes)
        peak_memory_mib = None
    else:
        optimization_seconds = math.fsum(rows[scene]["optimizationSeconds"] for scene in scenes)
        overall_seconds = math.fsum(rows[scene]["overallSeconds"] for scene in scenes)
        peak_memory_mib = None
    return {
        "label": label,
        "specification": specification,
        "scenes": len(scenes),
        "sse_vs_ceres_geometric": geometric_mean(ratios),
        "sse_vs_ceres_summed": (
            math.fsum(sse(rows[scene], kind) for scene in scenes)
            / math.fsum(sse(reference[scene]) for scene in scenes)
        ),
        "wins_ties_losses_vs_ceres": [wins, ties, len(scenes) - wins - ties],
        "optimization_seconds": optimization_seconds,
        "overall_seconds": overall_seconds,
        "timing_class": timing_class,
        "peak_memory_mib": peak_memory_mib,
    }


def add_bae_ratios(rows, bae_cg, scenes):
    for row in rows:
        source = row.pop("_source")
        kind = row.pop("_kind")
        row["sse_vs_bae_cg_geometric"] = geometric_mean([
            sse(source[scene], kind) / sse(bae_cg[scene], "bae")
            for scene in scenes
        ])


def write_table(output, title, rows, include_bae=False):
    output.write(f"## {title}\n\n")
    header = "| Method | Fixed specification | Scenes | SSE/Ceres | Summed SSE/Ceres | W/T/L vs Ceres | Optimization s | Timing class"
    separator = "|---|---|---:|---:|---:|---:|---:|---"
    if include_bae:
        header += " | SSE/BAE-CG"
        separator += "|---:"
    output.write(header + " |\n")
    output.write(separator + "|\n")
    for row in rows:
        wins, ties, losses = row["wins_ties_losses_vs_ceres"]
        output.write(
            f"| {row['label']} | {row['specification']} | {row['scenes']} | "
            f"{row['sse_vs_ceres_geometric']:.6f} | "
            f"{row['sse_vs_ceres_summed']:.6f} | {wins}/{ties}/{losses} | "
            f"{row['optimization_seconds']:.3f} | {row['timing_class']}"
        )
        if include_bae:
            output.write(f" | {row['sse_vs_bae_cg_geometric']:.6f}")
        output.write(" |\n")
    output.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RESULTS / "stage_c_publication_comparison",
    )
    arguments = parser.parse_args()

    k1 = load_jsonl(K1_PATH)
    validate_k1(k1)
    scaling_1dsfm, scaling_1dsfm_path = load_scaling("1dsfm")
    scaling_bal, scaling_bal_path = load_scaling("bal")
    ceres = {family: load_jsonl(path) for family, path in CERES_PATHS.items()}
    validate_ceres(ceres["1dsfm"], 15)
    validate_ceres(ceres["bal"], 29)
    if not (k1.keys() == scaling_1dsfm[4].keys() == scaling_1dsfm[16].keys() == ceres["1dsfm"].keys()):
        raise ValueError("all-15 1DSfM scene mismatch")
    if not (scaling_bal[4].keys() == scaling_bal[16].keys() == ceres["bal"].keys()):
        raise ValueError("all-29 BAL scene mismatch")

    bae = {label: load_jsonl(path) for label, path in BAE_PATHS.items()}
    for label, rows in bae.items():
        validate_bae(rows, label)

    one_d_sfm_scenes = sorted(k1)
    bal_scenes = sorted(ceres["bal"])
    all15 = [
        summarize("Ceres", ceres["1dsfm"], ceres["1dsfm"], one_d_sfm_scenes, "ceres", "left-SE3, I90, T16", "CPU native solve"),
        summarize("DRS K1", k1, ceres["1dsfm"], one_d_sfm_scenes, "drs", "specialized local diagnostic, I80, T24", "CPU DRS optimization"),
        summarize("DRS K4", scaling_1dsfm[4], ceres["1dsfm"], one_d_sfm_scenes, "drs", "frozen C1+C5 resource endpoint, I30, T1/cluster", "CPU DRS optimization"),
        summarize("DRS K16", scaling_1dsfm[16], ceres["1dsfm"], one_d_sfm_scenes, "drs", "frozen C1+C5 latency endpoint, I30, T1/cluster", "CPU DRS optimization"),
    ]
    all29 = [
        summarize("Ceres", ceres["bal"], ceres["bal"], bal_scenes, "ceres", "left-SE3, I90, T16", "CPU native solve"),
        summarize("DRS K4", scaling_bal[4], ceres["bal"], bal_scenes, "drs", "frozen C1+C5 resource endpoint, I30, T1/cluster", "CPU DRS optimization"),
        summarize("DRS K16", scaling_bal[16], ceres["bal"], bal_scenes, "drs", "frozen C1+C5 latency endpoint, I30, T1/cluster", "CPU DRS optimization"),
    ]
    six_scenes = list(BAE_SCENES)
    six_sources = [
        ("Ceres", ceres["1dsfm"], "ceres", "left-SE3, I90, T16", "CPU native solve"),
        ("DRS K1", k1, "drs", "specialized local diagnostic, I80, T24", "CPU DRS optimization"),
        ("DRS K4", scaling_1dsfm[4], "drs", "frozen C1+C5 resource endpoint, I30, T1/cluster", "CPU DRS optimization"),
        ("DRS K16", scaling_1dsfm[16], "drs", "frozen C1+C5 latency endpoint, I30, T1/cluster", "CPU DRS optimization"),
        ("BAE Schur-PCG CG", bae["BAE Schur-PCG CG"], "bae", "verified exported state, I90", "RTX 5090 GPU optimization"),
        ("BAE Schur-PCG Nesterov", bae["BAE Schur-PCG Nesterov"], "bae", "verified exported state, I90", "RTX 5090 GPU optimization"),
    ]
    six = []
    for label, source, kind, specification, timing_class in six_sources:
        row = summarize(label, source, ceres["1dsfm"], six_scenes, kind, specification, timing_class)
        row["_source"] = source
        row["_kind"] = kind
        six.append(row)
    add_bae_ratios(six, bae["BAE Schur-PCG CG"], six_scenes)

    summary = {
        "objective": "standard Snavely pixel sum-squared error",
        "all15_1dsfm": all15,
        "all29_bal": all29,
        "bae_six_scene_inset": six,
        "coverage_limits": {
            "k1_bal": "not available",
            "bae_all15_1dsfm": "not available; verified matched coverage is six scenes",
            "bae_all29_bal": "not available",
        },
        "source_artifacts": {
            "k1": str(K1_PATH.relative_to(ROOT)),
            "k4_k16_1dsfm": str(scaling_1dsfm_path.relative_to(ROOT)),
            "k4_k16_bal": str(scaling_bal_path.relative_to(ROOT)),
            "ceres_1dsfm": str(CERES_PATHS["1dsfm"].relative_to(ROOT)),
            "ceres_bal": str(CERES_PATHS["bal"].relative_to(ROOT)),
            "bae_cg": str(BAE_PATHS["BAE Schur-PCG CG"].relative_to(ROOT)),
            "bae_nesterov": str(BAE_PATHS["BAE Schur-PCG Nesterov"].relative_to(ROOT)),
        },
    }
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    (arguments.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = arguments.output_root / "report.md"
    with report.open("w", encoding="utf-8") as output:
        output.write("# Final K1/K4/K16, Ceres, And BAE Publication Comparison\n\n")
        output.write(
            "All rows use independently evaluated standard pixel SSE. Iteration and "
            "thread budgets are fixed but intentionally differ by solver role. CPU "
            "DRS, CPU Ceres, and RTX 5090 BAE times are reported in separate timing "
            "classes; no cross-hardware speedup is claimed.\n\n"
        )
        write_table(output, "Complete 1DSfM cohort (15/15)", all15)
        write_table(output, "Complete BAL cohort (29/29)", all29)
        output.write(
            "K1 has no authoritative matched all-29 BAL artifact and is therefore "
            "omitted from the BAL panel.\n\n"
        )
        write_table(output, "Verified BAE inset (six 1DSfM scenes)", six, include_bae=True)
        output.write("## Interpretation\n\n")
        output.write(
            "K1 remains a local-solver diagnostic: it reaches near-Ceres aggregate "
            "quality on all 15 1DSfM scenes but is neither the distributed method nor "
            "a matched work budget. K4 and K16 are the frozen global distributed "
            "resource and latency endpoints. They remain close to Ceres on all 29 "
            "BAL scenes, while difficult 1DSfM initialization retains a substantial "
            "quality gap. Verified BAE coverage is only six 1DSfM scenes, uses GPU "
            "hardware, and is basin-sensitive on Trafalgar; it is contextual evidence, "
            "not an all-scene or deterministic reference. No per-scene settings or "
            "solver routing are used.\n"
        )
    print(report)


if __name__ == "__main__":
    main()
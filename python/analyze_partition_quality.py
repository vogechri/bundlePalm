#!/usr/bin/env python3
"""Analyze criteria that distinguish good and bad two-part partitions."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.stats import spearmanr

import palm_ba
from bae_local_solver import schur_camera_edge_weights
from palm_partition_portfolio import make_candidate


ROOT = Path("palm_runs/portfolio_frozen_holdouts")
CACHE = ROOT / "partition_quality_cache"
HORIZONS = (1, 3, 5, 10, 20, 30)
MAX_SCHUR_PAIR_CONTRIBUTIONS = 20_000_000


def read_records(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def cached_schur_edges(problem_number: int,
                       problem: palm_ba.BALProblem) -> dict[tuple[int, int], float]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"problem{problem_number}_schur_edges.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            first = archive["first"]
            second = archive["second"]
            weight = archive["weight"]
    else:
        edges = schur_camera_edge_weights(
            problem.cameras, problem.points, problem.camera_indices,
            problem.point_indices, problem.observations)
        pairs = np.asarray(list(edges), dtype=np.int64)
        first, second = pairs[:, 0], pairs[:, 1]
        weight = np.asarray(list(edges.values()), dtype=np.float64)
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle, first=first, second=second, weight=weight)
        temporary.replace(path)
    return {
        (int(i), int(j)): float(value)
        for i, j, value in zip(first, second, weight)
    }


def candidate_features(
        problem: palm_ba.BALProblem, candidate,
        edges: dict[tuple[int, int], float] | None) -> dict[str, float]:
    camera_owner = candidate.camera_owner
    point_owner = candidate.point_owner
    cut_mask = (
        camera_owner[problem.camera_indices]
        != point_owner[problem.point_indices])
    cut_points = np.unique(problem.point_indices[cut_mask])
    point_degree = np.bincount(
        problem.point_indices, minlength=len(problem.points))
    result = {
        "cut_observations": float(candidate.cut_observations),
        "included_load_ratio": candidate.included_load_ratio,
        "camera_load_ratio": candidate.camera_load_ratio,
        "max_load_ratio": candidate.max_load_ratio,
        "shared_point_count": float(len(cut_points)),
        "shared_point_degree_mean": (
            float(point_degree[cut_points].mean()) if len(cut_points) else 0.0),
        "shared_point_degree_max": (
            float(point_degree[cut_points].max()) if len(cut_points) else 0.0),
    }
    if edges is not None:
        edge_items = list(edges.items())
        total_schur = sum(weight for _, weight in edge_items)
        cut_schur = sum(
            weight for (first, second), weight in edge_items
            if camera_owner[first] != camera_owner[second])
        boundary_schur = np.zeros(len(camera_owner), dtype=np.float64)
        for (first, second), weight in edge_items:
            if camera_owner[first] != camera_owner[second]:
                boundary_schur[first] += weight
                boundary_schur[second] += weight
        result.update({
            "schur_cut_fraction": cut_schur / total_schur,
            "schur_boundary_max_mean": (
                float(boundary_schur.max() / boundary_schur.mean())
                if boundary_schur.mean() else 0.0),
            "schur_boundary_cv": (
                float(boundary_schur.std() / boundary_schur.mean())
                if boundary_schur.mean() else 0.0),
        })
    return result


def trajectory_features(records: list[dict]) -> dict[str, float]:
    first = records[0]
    result = {
        "epoch1_reduction_fraction": 1.0 - first["best_cost"] / first["cost"],
        "epoch1_inner_iterations": float(sum(first["block_inner_iterations"])),
        "epoch1_attempts": float(sum(first["block_attempts"])),
        "epoch1_min_model_quality": float(min(first["block_model_quality"])),
    }
    initial_cost = float(first["cost"])
    for horizon in HORIZONS:
        prefix = records[:horizon]
        best = min(float(record["best_cost"]) for record in prefix)
        result[f"reduction_{horizon}"] = 1.0 - best / initial_cost
        result[f"cost_{horizon}"] = best
    return result


def standardized(rows: list[dict], feature: str,
                 outcome: str) -> tuple[list[float], list[float]]:
    feature_values = []
    outcome_values = []
    for problem in sorted({row["problem"] for row in rows}):
        group = [row for row in rows if row["problem"] == problem]
        x = np.asarray([row[feature] for row in group], dtype=np.float64)
        y = np.asarray([row[outcome] for row in group], dtype=np.float64)
        feature_values.extend((x - x.mean()) / (x.std() or 1.0))
        outcome_values.extend((y - y.mean()) / (y.std() or 1.0))
    return feature_values, outcome_values


def criterion_stats(rows: list[dict], feature: str,
                    outcome: str) -> dict[str, float | int]:
    rows = [
        row for row in rows if feature in row and outcome in row
    ]
    x, y = standardized(rows, feature, outcome)
    correlation = float(spearmanr(x, y).statistic)
    concordant = 0
    pairs = 0
    for problem in sorted({row["problem"] for row in rows}):
        group = [row for row in rows if row["problem"] == problem]
        for first in range(len(group)):
            for second in range(first + 1, len(group)):
                feature_difference = (
                    group[first][feature] - group[second][feature])
                outcome_difference = (
                    group[first][outcome] - group[second][outcome])
                if feature_difference == 0.0 or outcome_difference == 0.0:
                    continue
                pairs += 1
                concordant += (
                    feature_difference * outcome_difference > 0.0)
    return {
        "spearman": correlation,
        "pairwise_accuracy": concordant / pairs if pairs else math.nan,
        "pair_count": pairs,
    }


def selector_stats(rows: list[dict], outcome: str,
                   features: tuple[tuple[str, int], ...]) -> dict:
    regrets = []
    selections = []
    for problem in sorted({row["problem"] for row in rows}):
        group = [
            row for row in rows
            if row["problem"] == problem and outcome in row
            and all(feature in row for feature, _ in features)
        ]
        if len(group) < 2:
            continue
        scores = np.zeros(len(group), dtype=np.float64)
        for feature, direction in features:
            values = [direction * row[feature] for row in group]
            scores += np.argsort(np.argsort(values, kind="stable"), kind="stable")
        selected = group[int(np.argmin(scores))]
        oracle = min(group, key=lambda row: row[outcome])
        regret = 100.0 * (selected[outcome] / oracle[outcome] - 1.0)
        regrets.append(regret)
        selections.append({
            "problem": problem,
            "selected": selected["candidate"],
            "oracle": oracle["candidate"],
            "regret_percent": regret,
        })
    return {
        "problems": len(regrets),
        "oracle_hits": sum(item["regret_percent"] == 0.0 for item in selections),
        "mean_regret_percent": statistics.mean(regrets),
        "median_regret_percent": statistics.median(regrets),
        "max_regret_percent": max(regrets),
        "selections": selections,
    }


def main() -> None:
    rows = []
    for summary_path in sorted(ROOT.glob("problem*_parts2/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        problem_number = int(summary["file_name"].split("-")[1])
        problem_path = Path(summary["file_name"].removesuffix(".bz2"))
        compressed_path = Path(summary["file_name"])
        problem = palm_ba.read_bal(
            compressed_path if compressed_path.exists() else problem_path)
        point_degree = np.bincount(
            problem.point_indices, minlength=len(problem.points))
        schur_pair_contributions = int(np.sum(
            point_degree * (point_degree - 1) // 2))
        edges = (
            cached_schur_edges(problem_number, problem)
            if schur_pair_contributions <= MAX_SCHUR_PAIR_CONTRIBUTIONS
            else None)
        if edges is None:
            print(
                f"problem {problem_number}: skipping exact Schur graph with "
                f"{schur_pair_contributions:,} landmark pair contributions",
                flush=True)
        for item in summary["candidates"]:
            candidate = make_candidate(
                problem, 2, item["seed"], item["refinement_passes"], 256)
            if candidate.ownership_hash != item["hash"]:
                raise RuntimeError(
                    f"ownership mismatch for problem {problem_number} "
                    f"candidate {item['name']}")
            records = read_records(Path(item["trajectory"]))
            row = {
                "problem": problem_number,
                "candidate": item["name"],
                "epochs": len(records),
                "schur_pair_contributions": schur_pair_contributions,
                **candidate_features(problem, candidate, edges),
                **trajectory_features(records),
            }
            row["cost_30"] = min(
                float(record["best_cost"]) for record in records[:30])
            if len(records) >= 90:
                row["cost_90"] = min(
                    float(record["best_cost"]) for record in records[:90])
            rows.append(row)

    static_features = (
        "cut_observations", "shared_point_count",
        "shared_point_degree_mean",
        "included_load_ratio", "camera_load_ratio", "max_load_ratio",
        "schur_cut_fraction", "schur_boundary_max_mean",
        "schur_boundary_cv",
    )
    early_features = (
        "epoch1_min_model_quality",
        "reduction_3", "reduction_5", "reduction_10",
        "reduction_20",
    )
    stats_30 = {
        feature: criterion_stats(rows, feature, "cost_30")
        for feature in (*static_features, *early_features)
    }
    audited = [row for row in rows if "cost_90" in row]
    audited_problems = {
        problem for problem in {row["problem"] for row in audited}
        if sum(row["problem"] == problem for row in audited) > 1
    }
    audited = [row for row in audited if row["problem"] in audited_problems]
    stats_90 = {
        feature: criterion_stats(audited, feature, "cost_90")
        for feature in (*static_features, *early_features)
    }
    selectors = {
        "minimum_cut_observations": (("cut_observations", 1),),
        "minimum_shared_points": (("shared_point_count", 1),),
        "minimum_schur_cut": (("schur_cut_fraction", 1),),
        "minimum_schur_boundary_max_mean": (
            ("schur_boundary_max_mean", 1),),
        "static_rank_composite": (
            ("shared_point_count", 1),
            ("shared_point_degree_mean", -1),
            ("schur_boundary_max_mean", 1),
        ),
        **{
            f"pilot_cost_epoch_{horizon}": ((f"cost_{horizon}", 1),)
            for horizon in HORIZONS
        },
    }
    output = {
        "candidate_count": len(rows),
        "problem_count": len({row["problem"] for row in rows}),
        "audited_candidate_count": len(audited),
        "audited_problem_count": len(audited_problems),
        "criteria_vs_cost_30": stats_30,
        "criteria_vs_cost_90": stats_90,
        "selectors_vs_cost_30": {
            name: selector_stats(rows, "cost_30", features)
            for name, features in selectors.items()
        },
        "selectors_vs_cost_90": {
            name: selector_stats(audited, "cost_90", features)
            for name, features in selectors.items()
        },
        "rows": rows,
    }
    path = ROOT / "partition_quality_analysis.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    for feature, values in sorted(
            stats_30.items(), key=lambda item: abs(item[1]["spearman"]),
            reverse=True):
        print(
            f"{feature:28} rho={values['spearman']:+.3f} "
            f"pair={values['pairwise_accuracy']:.3f}")


if __name__ == "__main__":
    main()
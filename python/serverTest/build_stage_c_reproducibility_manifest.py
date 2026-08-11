#!/usr/bin/env python3
"""Build the Stage-C reproducibility manifest from authoritative artifacts."""

import argparse
import json
import math
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmark_results"
CONFIG_KEYS = (
    "clusters",
    "iterations",
    "localSteps",
    "threadsPerCluster",
    "localSolver",
    "outerAcceleration",
    "lineSearchGrid",
    "accelerationRestartAfter",
    "proximalMetric",
    "consensusMetric",
    "sharedOnlyCameraProximal",
    "sceneNormalization",
    "cameraScaling",
    "initialBlockRegularization",
    "initialBlockCurvatureMultiplier",
    "blockRecoveryMode",
    "trustRegionPolicy",
    "persistentTrustRegion",
    "schurPcgRelativeTolerance",
    "schurPcgMaximumIterations",
    "adaptiveLocalDepth",
    "adaptiveLocalDepthStart",
    "adaptiveLocalDepthMaximum",
    "adaptiveLocalDepthHigh",
    "adaptiveLocalDepthLow",
    "adaptiveLocalDepthWindow",
    "adaptiveLocalDepthDwell",
    "safeguardMode",
    "dreRelativeIncrease",
    "minimumPrimalRatio",
    "safeguardAnnealingIterations",
)

METHODS = {
    "plain": {
        "1dsfm": (
            "c1_long_block_baseline_six_i30",
            "c1_long_block_baseline_heldout9_i30",
        ),
        "large_bal": ("c1_long_block_baseline_large_bal_i30",),
    },
    "c1": {
        "1dsfm": (
            "c1_long_themelis_six_i30",
            "c1_long_themelis_heldout9_i30",
        ),
        "large_bal": ("c1_long_themelis_large_bal_i30",),
    },
    "c5": {
        "1dsfm": ("c1_c5_adaptive_off_all15_i30",),
        "large_bal": ("c1_c5_adaptive_off_large_bal_i30",),
    },
    "c1_c5": {
        "1dsfm": ("c1_c5_adaptive_on_all15_i30",),
        "large_bal": ("c1_c5_adaptive_on_large_bal_i30",),
    },
}

C4_METHODS = {
    "nesterov": ("c4_k24_postcrash_all15_i30/nesterov",),
    "schur_pcg": ("c4_k24_postcrash_all15_i30/schur_pcg",),
}

FINAL_METHODS = {
    "1dsfm": {
        "plain": ("stage_c_final_all15_all29_k24_i30/1dsfm/plain",),
        "c1": ("stage_c_final_all15_all29_k24_i30/1dsfm/c1",),
        "c5": ("stage_c_final_all15_all29_k24_i30/1dsfm/c5",),
        "c1_c5": ("stage_c_final_all15_all29_k24_i30/1dsfm/c1_c5",),
    },
    "bal": {
        "plain": ("stage_c_final_all15_all29_k24_i30/bal/plain",),
        "c1": ("stage_c_final_all15_all29_k24_i30/bal/c1",),
        "c5": ("stage_c_final_all15_all29_k24_i30/bal/c5",),
        "c1_c5": ("stage_c_final_all15_all29_k24_i30/bal/c1_c5",),
    },
}
FINAL_CERES = {
    "1dsfm": RESULTS / "1dsfm_drs_ceres_se3_all15/ceres/results.jsonl",
    "bal": RESULTS / "bal_ceres_se3_all29/results.jsonl",
}
TUNED_C5_METHODS = {
    "1dsfm": {
        "c5": ("stage_c_c5_frozen_main_effect_all15_all29_k24_i30/1dsfm/c5",),
        "c1_c5": (
            "stage_c_c5_start_development_k24_i30/s5p00/1dsfm/c1_c5",
            "stage_c_c5_frozen_heldout9_k24_i30/1dsfm/c1_c5",
        ),
    },
    "bal": {
        "c5": ("stage_c_c5_frozen_main_effect_all15_all29_k24_i30/bal/c5",),
        "c1_c5": ("stage_c_c5_frozen_bal_all29_k24_i30/bal/c1_c5",),
    },
}


def scene_name(row):
    dataset = Path(row["dataset"])
    if "sfm_init_1dsfm" in row["dataset"]:
        return dataset.parent.name
    match = re.search(r"problem-(\d+)-", dataset.name)
    if not match:
        raise ValueError(f"cannot identify dataset: {dataset}")
    return f"bal{match.group(1)}"


def load_rows(directories):
    rows = {}
    sources = {}
    for directory_name in directories:
        directory = RESULTS / directory_name
        paths = tuple(directory.glob("*.jsonl"))
        if not paths:
            raise FileNotFoundError(f"no JSONL in {directory}")
        for path in paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                scene = scene_name(row)
                if scene in rows:
                    raise ValueError(f"duplicate scene {scene} in {directories}")
                rows[scene] = row
                sources[scene] = str(path.relative_to(ROOT))
    return rows, sources


def load_rows_from_path(path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scene = scene_name(row)
        if scene in rows:
            raise ValueError(f"duplicate scene {scene} in {path}")
        rows[scene] = row
    return rows


def parse_worker_timings(directories, require_complete=True):
    timings = {}
    for directory_name in directories:
        directory = RESULTS / directory_name
        for path in (directory / "memory").glob("*_worker.time"):
            text = path.read_text(encoding="utf-8")
            user = re.search(r"User time \(seconds\): ([0-9.]+)", text)
            system = re.search(r"System time \(seconds\): ([0-9.]+)", text)
            rss = re.search(r"Maximum resident set size \(kbytes\): ([0-9.]+)", text)
            if not (user and system and rss):
                if require_complete:
                    raise ValueError(f"incomplete worker timing: {path}")
                continue
            scene = next(
                (name for name in load_rows(directories)[0] if f"_{name}_k24_" in path.name),
                None,
            )
            if scene is None:
                # Large BAL labels are bal1490, while filenames contain bal1490.
                scene = next(
                    (name for name in load_rows(directories)[0] if f"_{name}_" in path.name),
                    None,
                )
            if scene is not None:
                timings[scene] = {
                    "worker_cpu_seconds": float(user.group(1)) + float(system.group(1)),
                    "worker_max_rss_kb": int(float(rss.group(1))),
                    "path": str(path.relative_to(ROOT)),
                }
    return timings


def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def row_record(row, source, timing):
    return {
        "dataset": row["dataset"],
        "artifact": source,
        "variant": row["variant"],
        "sum_squared_error": row["qualityMetrics"]["sumSquaredError"],
        "optimization_seconds": row["optimizationSeconds"],
        "overall_seconds": row["overallSeconds"],
        "proximal_oracle_calls": row["proximalOracleCalls"],
        "rejections": row["rejections"],
        "best_iteration_zero_based": row["bestIteration"],
        **timing,
    }


def summary(reference, candidate):
    scenes = sorted(reference)
    ratios = [
        candidate[scene]["qualityMetrics"]["sumSquaredError"]
        / reference[scene]["qualityMetrics"]["sumSquaredError"]
        for scene in scenes
    ]
    optimization = [
        candidate[scene]["optimizationSeconds"]
        / reference[scene]["optimizationSeconds"]
        for scene in scenes
    ]
    oracle = [
        candidate[scene]["proximalOracleCalls"]
        / reference[scene]["proximalOracleCalls"]
        for scene in scenes
    ]
    return {
        "scene_count": len(scenes),
        "sse_geometric_mean_ratio": geometric_mean(ratios),
        "optimization_geometric_mean_ratio": geometric_mean(optimization),
        "proximal_oracle_geometric_mean_ratio": geometric_mean(oracle),
        "wins": sum(value < 1.0 for value in ratios),
        "ties": sum(math.isclose(value, 1.0, rel_tol=1e-12) for value in ratios),
        "losses": sum(value > 1.0 for value in ratios),
        "worst_sse_ratio": max(ratios),
    }


def quality_summary(reference, candidate):
    scenes = sorted(reference)
    ratios = [
        candidate[scene]["qualityMetrics"]["sumSquaredError"]
        / reference[scene]["qualityMetrics"]["sumSquaredError"]
        for scene in scenes
    ]
    return {
        "scene_count": len(scenes),
        "sse_geometric_mean_ratio": geometric_mean(ratios),
        "sse_summed_ratio": (
            math.fsum(
                candidate[scene]["qualityMetrics"]["sumSquaredError"]
                for scene in scenes
            )
            / math.fsum(
                reference[scene]["qualityMetrics"]["sumSquaredError"]
                for scene in scenes
            )
        ),
        "wins": sum(value < 1.0 for value in ratios),
        "ties": sum(math.isclose(value, 1.0, rel_tol=1e-12) for value in ratios),
        "losses": sum(value > 1.0 for value in ratios),
        "worst_sse_ratio": max(ratios),
    }


def git_value(*args):
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def build_manifest():
    loaded = {}
    manifest = {
        "schema_version": 1,
        "date": "2026-08-11",
        "repository": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "working_tree_clean": not bool(git_value("status", "--porcelain")),
        },
        "recovery_boundary": {
            "maintained_path": "C1/C2/C4/C5",
            "maintained_tests_passed": 128,
            "worker_build_target": "serverTest/build_admm/zeromq_cpp_server_ex",
            "artifact_only_paths": (
                "exact checkpoint/BAE handoff",
                "full factorized-C3 coordinator execution",
            ),
        },
        "objective": "standard Snavely pixel sum-squared error",
        "methods": {},
        "c4_inner_solver": {},
        "final_cross_family_confirmation": {},
        "tuned_c5_final": {},
        "huber_sentinel": {},
        "scaling_confirmation": {},
        "scaling_repeatability": {},
        "relative_summaries": {},
    }

    for method, cohorts in METHODS.items():
        manifest["methods"][method] = {}
        loaded[method] = {}
        for cohort, directories in cohorts.items():
            rows, sources = load_rows(directories)
            timings = parse_worker_timings(directories)
            if rows.keys() != timings.keys():
                raise ValueError(
                    f"worker timing coverage mismatch for {method}/{cohort}: "
                    f"rows={sorted(rows)} timings={sorted(timings)}"
                )
            configs = {
                tuple((key, json.dumps(row.get(key), sort_keys=True)) for key in CONFIG_KEYS)
                for row in rows.values()
            }
            if len(configs) != 1:
                raise ValueError(f"configuration mismatch in {method}/{cohort}")
            config = {key: json.loads(value) for key, value in next(iter(configs))}
            loaded[method][cohort] = rows
            manifest["methods"][method][cohort] = {
                "directories": [f"benchmark_results/{name}" for name in directories],
                "configuration": config,
                "rows": {
                    scene: row_record(rows[scene], sources[scene], timings[scene])
                    for scene in sorted(rows)
                },
            }

    for cohort in ("1dsfm", "large_bal"):
        plain = loaded["plain"][cohort]
        manifest["relative_summaries"][cohort] = {
            method: summary(plain, loaded[method][cohort])
            for method in ("c1", "c5", "c1_c5")
        }
        manifest["relative_summaries"][cohort]["c1_c5_vs_c1"] = summary(
            loaded["c1"][cohort], loaded["c1_c5"][cohort]
        )

    c4_loaded = {}
    for solver, directories in C4_METHODS.items():
        rows, sources = load_rows(directories)
        timings = parse_worker_timings(directories, require_complete=False)
        if len(rows) != 15:
            raise ValueError(
                f"C4 coverage mismatch for {solver}: "
                f"rows={sorted(rows)}"
            )
        configs = {
            tuple((key, json.dumps(row.get(key), sort_keys=True)) for key in CONFIG_KEYS)
            for row in rows.values()
        }
        if len(configs) != 1:
            raise ValueError(f"configuration mismatch in C4/{solver}")
        config = {key: json.loads(value) for key, value in next(iter(configs))}
        c4_loaded[solver] = rows
        manifest["c4_inner_solver"][solver] = {
            "directories": [f"benchmark_results/{name}" for name in directories],
            "configuration": config,
            "worker_timing_coverage": {
                "complete_rows": len(timings),
                "expected_rows": len(rows),
                "note": (
                    "Worker /usr/bin/time files are empty because the runner "
                    "terminates the persistent worker; coordinator-reported "
                    "optimization and overall times remain available."
                ),
            },
            "rows": {
                scene: row_record(
                    rows[scene],
                    sources[scene],
                    timings.get(scene, {
                        "worker_cpu_seconds": None,
                        "worker_max_rss_kb": None,
                        "path": None,
                    }),
                )
                for scene in sorted(rows)
            },
        }
    manifest["c4_inner_solver"]["schur_pcg_vs_nesterov"] = summary(
        c4_loaded["nesterov"], c4_loaded["schur_pcg"]
    )
    manifest["c4_inner_solver"]["decision"] = {
        "nesterov_is_named_default": True,
        "schur_pcg_is_explicit_factorial_level": True,
        "artifact": "benchmark_results/c4_k24_postcrash_all15_i30/DECISION.md",
        "reason": (
            "PCG is faster overall but regresses geometric per-scene SSE and "
            "the worst-scene quality tail"
        ),
    }

    for cohort, methods in FINAL_METHODS.items():
        expected = 15 if cohort == "1dsfm" else 29
        loaded_final = {}
        cohort_record = {"methods": {}}
        for method, directories in methods.items():
            rows, sources = load_rows(directories)
            if len(rows) != expected:
                raise ValueError(
                    f"final {cohort}/{method} coverage is {len(rows)}/{expected}"
                )
            configs = {
                tuple((key, json.dumps(row.get(key), sort_keys=True)) for key in CONFIG_KEYS)
                for row in rows.values()
            }
            if len(configs) != 1:
                raise ValueError(f"configuration mismatch in final {cohort}/{method}")
            loaded_final[method] = rows
            cohort_record["methods"][method] = {
                "directories": [f"benchmark_results/{name}" for name in directories],
                "configuration": {
                    key: json.loads(value) for key, value in next(iter(configs))
                },
                "rows": {
                    scene: row_record(
                        rows[scene],
                        sources[scene],
                        {
                            "worker_cpu_seconds": None,
                            "worker_max_rss_kb": None,
                            "path": None,
                        },
                    )
                    for scene in sorted(rows)
                },
            }
        ceres = load_rows_from_path(FINAL_CERES[cohort])
        if len(ceres) != expected:
            raise ValueError(f"final Ceres {cohort} coverage is {len(ceres)}/{expected}")
        if loaded_final["plain"].keys() != ceres.keys():
            raise ValueError(f"final Ceres/DRS scene mismatch for {cohort}")
        cohort_record["ceres"] = {
            "artifact": str(FINAL_CERES[cohort].relative_to(ROOT)),
            "rows": {
                scene: {
                    "dataset": ceres[scene]["dataset"],
                    "sum_squared_error": ceres[scene]["qualityMetrics"]["sumSquaredError"],
                    "native_solve_seconds": ceres[scene]["native"]["solveSeconds"],
                }
                for scene in sorted(ceres)
            },
        }
        cohort_record["summaries"] = {
            "versus_ceres": {
                method: quality_summary(ceres, rows)
                for method, rows in loaded_final.items()
            },
            "versus_plain": {
                method: quality_summary(loaded_final["plain"], loaded_final[method])
                for method in ("c1", "c5", "c1_c5")
            },
            "incremental": {
                "c1_c5_vs_c1": quality_summary(
                    loaded_final["c1"], loaded_final["c1_c5"]
                ),
                "c1_c5_vs_c5": quality_summary(
                    loaded_final["c5"], loaded_final["c1_c5"]
                ),
            },
            "optimization_versus_plain": {
                method: (
                    math.fsum(
                        row["optimizationSeconds"]
                        for row in loaded_final[method].values()
                    )
                    / math.fsum(
                        row["optimizationSeconds"]
                        for row in loaded_final["plain"].values()
                    )
                )
                for method in ("c1", "c5", "c1_c5")
            },
        }
        manifest["final_cross_family_confirmation"][cohort] = cohort_record
    manifest["final_cross_family_confirmation"]["report"] = (
        "benchmark_results/stage_c_final_all15_all29_k24_i30/"
        "comparison_to_ceres_and_plain_drs.md"
    )

    for cohort, methods in TUNED_C5_METHODS.items():
        expected = 15 if cohort == "1dsfm" else 29
        tuned = {}
        record = {
            "policy": {
                "start_iteration": 5,
                "high_threshold": 0.35,
                "low_threshold": 0.20,
                "window": 3,
                "dwell": 3,
                "maximum_depth": 2,
            },
            "methods": {},
        }
        for method, directories in methods.items():
            rows, sources = load_rows(directories)
            if len(rows) != expected:
                raise ValueError(
                    f"tuned {cohort}/{method} coverage is {len(rows)}/{expected}"
                )
            tuned[method] = rows
            record["methods"][method] = {
                "directories": [f"benchmark_results/{name}" for name in directories],
                "rows": {
                    scene: row_record(
                        rows[scene],
                        sources[scene],
                        {
                            "worker_cpu_seconds": None,
                            "worker_max_rss_kb": None,
                            "path": None,
                        },
                    )
                    for scene in sorted(rows)
                },
            }
        plain_rows, _ = load_rows(FINAL_METHODS[cohort]["plain"])
        c1_rows, _ = load_rows(FINAL_METHODS[cohort]["c1"])
        ceres_rows = load_rows_from_path(FINAL_CERES[cohort])
        if not (
            plain_rows.keys() == c1_rows.keys() == tuned["c5"].keys()
            == tuned["c1_c5"].keys() == ceres_rows.keys()
        ):
            raise ValueError(f"tuned C5 scene mismatch for {cohort}")
        record["summaries"] = {
            "c5_vs_plain": quality_summary(plain_rows, tuned["c5"]),
            "c1_c5_vs_plain": quality_summary(plain_rows, tuned["c1_c5"]),
            "c1_c5_vs_c1": quality_summary(c1_rows, tuned["c1_c5"]),
            "c1_c5_vs_c5": quality_summary(tuned["c5"], tuned["c1_c5"]),
            "c1_c5_vs_ceres": quality_summary(ceres_rows, tuned["c1_c5"]),
        }
        manifest["tuned_c5_final"][cohort] = record
    manifest["tuned_c5_final"]["report"] = (
        "benchmark_results/stage_c_tuned_c5_final_report.md"
    )

    huber_drs, huber_drs_sources = load_rows((
        "stage_c_huber_irls_sentinel_drs/1dsfm/c1_c5",
        "stage_c_huber_irls_sentinel_drs/bal/c1_c5",
    ))
    huber_ceres, huber_ceres_sources = load_rows((
        "stage_c_huber_sentinel_ceres/1dsfm",
        "stage_c_huber_sentinel_ceres/bal",
    ))
    if huber_drs.keys() != huber_ceres.keys() or len(huber_drs) != 4:
        raise ValueError("Huber sentinel coverage mismatch")
    manifest["huber_sentinel"] = {
        "huber_delta": 0.5,
        "objective": "observation-level Huber pixel reprojection loss",
        "raw_sse_reported_separately": True,
        "drs": {
            scene: {
                "artifact": huber_drs_sources[scene],
                "objective_value": row["qualityMetrics"]["objectiveValue"],
                "sum_squared_error": row["qualityMetrics"]["sumSquaredError"],
                "optimization_seconds": row["optimizationSeconds"],
            }
            for scene, row in sorted(huber_drs.items())
        },
        "ceres": {
            scene: {
                "artifact": huber_ceres_sources[scene],
                "objective_value": row["qualityMetrics"]["objectiveValue"],
                "sum_squared_error": row["qualityMetrics"]["sumSquaredError"],
                "native_solve_seconds": row["native"]["solveSeconds"],
                "native_objective_relative_error": row[
                    "nativeObjectiveRelativeError"
                ],
            }
            for scene, row in sorted(huber_ceres.items())
        },
        "decision": "validated capability; current schedules not promoted",
        "report": "benchmark_results/stage_c_huber_sentinel_report.md",
    }

    scaling_path = RESULTS / "stage_c_scaling_confirmation_k4_16_i30/summary.json"
    scaling = json.loads(scaling_path.read_text(encoding="utf-8"))
    expected_scaling = {"1dsfm": 15, "bal": 29}
    for family, expected_scenes in expected_scaling.items():
        rows = scaling.get(family, [])
        if [row.get("clusters") for row in rows] != [4, 16]:
            raise ValueError(f"scaling K coverage mismatch for {family}")
        if any(row.get("scenes") != expected_scenes for row in rows):
            raise ValueError(f"scaling scene coverage mismatch for {family}")
    manifest["scaling_confirmation"] = {
        "objective": "standard Snavely pixel sum-squared error",
        "configuration": "frozen tuned L2 C1+C5",
        "threads_per_cluster": 1,
        "iterations": 30,
        "operating_points": {
            "resource": 4,
            "latency": 16,
        },
        "no_scene_specific_routing": True,
        "families": {
            family: [
                {key: value for key, value in row.items() if key != "rows"}
                for row in rows
            ]
            for family, rows in scaling.items()
        },
        "summary": str(scaling_path.relative_to(ROOT)),
        "report": (
            "benchmark_results/"
            "stage_c_scaling_confirmation_k4_16_i30/report.md"
        ),
    }

    repeat_path = RESULTS / "stage_c_scaling_repeats_k4_16_i30/summary.json"
    repeats = json.loads(repeat_path.read_text(encoding="utf-8"))
    repeatability = repeats.get("repeatability", {})
    if repeatability.get("total_cases") != 8:
        raise ValueError("scaling repeat case coverage mismatch")
    if repeatability.get("maximum_sse_relative_spread", math.inf) >= 1e-8:
        raise ValueError("scaling endpoint repeatability exceeds 1e-8")
    if repeatability.get("maximum_optimization_cv", math.inf) >= 0.02:
        raise ValueError("scaling optimization timing CV exceeds 2%")
    if not repeatability.get("work_counts_deterministic"):
        raise ValueError("scaling repeat work counts are not deterministic")
    manifest["scaling_repeatability"] = {
        "configuration": "frozen tuned L2 C1+C5 at K4 and K16",
        "warmups_excluded": 1,
        "measured_repeats": 3,
        "scenes": ["roman_forum", "trafalgar", "bal52", "bal3068"],
        "repeatability": repeatability,
        "endpoint_ratios": repeats["endpoint_ratios"],
        "summary": str(repeat_path.relative_to(ROOT)),
        "report": "benchmark_results/stage_c_scaling_repeats_k4_16_i30/report.md",
    }
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "stage_c_reproducibility_manifest.json",
    )
    arguments = parser.parse_args()
    manifest = build_manifest()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(arguments.output)


if __name__ == "__main__":
    main()

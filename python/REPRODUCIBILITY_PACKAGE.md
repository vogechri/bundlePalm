# Stage-C Reproducibility Package

## Validated Environment

- Repository checkpoint used for this audit: `ae22019` or a descendant.
- Worker: `serverTest/build_admm/zeromq_cpp_server_ex`.
- Worker SHA-256: `c408131a6977de3d92281f195c9ca888fe6c11e05182915bc36ba45d10326b3d`.
- Python: 3.12.13.
- CMake: 4.2.3.
- g++: 15.2.0.
- protobuf compiler: 3.21.12.
- Full maintained source suite: 223 tests passed on 2026-08-23.

Rebuild and test:

```bash
cd serverTest
cmake --build build_admm -j2
PYTHONPATH="$PWD/build_admm/generated/proto:$PWD" \
  .venv/bin/python -m pytest -q test_*.py
```

The build currently emits one deprecated ZeroMQ `recv` warning. Python tests
emit two protobuf Python 3.14 deprecation warnings. Neither affects the current
build or test result.

## Canonical Modes

Machine-readable commands, roles, report paths, and artifact paths are in:

```text
benchmark_results/stage_c_reproducibility_registry.json
```

Validate the registry and archive boundary with:

```bash
serverTest/.venv/bin/python \
  serverTest/validate_stage_c_reproducibility_registry.py
```

The registry covers:

1. Stage-C plain control;
2. C1 safeguarded outer acceleration;
3. C5 adaptive local work;
4. frozen C1+C5;
5. K4 resource mode;
6. K16 latency mode;
7. product-SO3 C1 diagnostic;
8. repeated startup distributed Schur diagnostic.

All commands are resumable with `OVERWRITE=0`. Product-SO3 and startup
bootstrap are default-off diagnostics, not common-policy presets.

## Table And Figure Provenance

The registry maps every `tab:*` label in `stage_c_long_horizon_results.tex` and
its generated inputs to one report and one machine-readable artifact.

Generated publication outputs:

- `serverTest/build_stage_c_publication_tables.py` creates the final complete-
  cohort and K1-carryover TeX tables;
- `serverTest/build_stage_c_publication_plots.py` creates objective/time,
  K16/K4 resource, and outer-outcome PDFs;
- `serverTest/build_ceres_se3_left_unified_benchmark.py` creates the unified
  all-15 1DSfM/all-29 BAL versus Ceres left-SE3 report and figures;
- `serverTest/build_stage_c_reproducibility_manifest.py` regenerates the
  machine-readable Stage-C manifest.

Generated artifacts are deterministic under the pinned publication dependency:

```bash
serverTest/.venv/bin/pip install -r serverTest/requirements-publication.txt
```

## Archive Boundary

Ordinary Git contains compact reports, summaries, generators, tests, figures,
and manuscript inputs. Raw JSONL, logs, states, worker timing, and memory trees
remain local or in external archival storage. The registry validator rejects
tracked files below `logs/`, `states/`, or `memory/` inside benchmark results.

Do not delete local raw campaign trees merely to clean `git status`; they are
restart and forensic evidence. Archive them by copying the complete campaign
directory together with a SHA-256 inventory.

## Limitations

- Historical exact checkpoint/BAE handoff portfolios are artifact-backed but
  their full maintained source path did not survive the 2026-08-11 recovery.
- Full factorized-C3 coordinator execution is also artifact-backed rather than a
  maintained runnable path.
- Compact summaries record peak coordinator and worker process RSS, not a
  synchronized aggregate multi-process memory peak.
- Ceres, CPU DRS, DRS+Schur, and RTX 5090 BAE have distinct timing boundaries;
  no cross-hardware speedup is claimed.
- The K24 combined-stack versus Ceres report is artifact-only Phase A. A fresh
  same-source Phase B rerun remains optional and requires explicit approval.

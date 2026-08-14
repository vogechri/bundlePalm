# Phased Local Solver Portfolio Gate

Date: 2026-08-14

## Motivation

The completed C4 all-15 comparison found Schur-PCG better through I10
(`0.961667x` finite-Nesterov geometric SSE, `0.959263x` summed) but Nesterov
better geometrically at I30. A worker-local dual race is not state-free: local
solves mutate persistent cameras, landmarks, accepted/best snapshots, and trust
radii, and heterogeneous per-worker solver selection would change the
product-space resolvent. The cheapest homogeneous portfolio is therefore one
single-trajectory global schedule.

## Frozen Schedule

- K24/I30, C4 direct-left-SE3 shared-only configuration;
- Schur-PCG for outer iterations I1--I10;
- finite Nesterov for I11--I30;
- no state reset, branch selection, or per-scene settings;
- matched pure-Nesterov and pure-Schur-PCG controls from fresh runs.

The switch is request-scoped and applies uniformly to nominal and accelerated
trials. It is default-off through `--local-solver-switch-iteration` and
`--local-solver-after-switch`.

## Sentinel Results

| Scene | Phased/Nesterov SSE | Phased/PCG SSE | Time/Nesterov | Time/PCG |
|---|---:|---:|---:|---:|
| Roman Forum | 1.258234 | 1.071572 | 0.880512 | 1.023115 |
| Trafalgar | 1.438000 | 1.001252 | 0.967263 | 1.163487 |
| NYC Library | 1.119719 | 1.210144 | 0.929668 | 1.059990 |
| Piccadilly | 1.060541 | 0.994774 | 0.976517 | 1.024479 |
| **Geometric mean** | **1.210707** | **1.066060** | **0.937717** | **1.066284** |

The phased schedule loses to Nesterov on 4/4 and to PCG on 3/4. Trafalgar is
already stalled under PCG at I9--I10 and remains stalled immediately after the
Nesterov switch. Other scenes continue descending, but Nesterov does not erase
the inherited PCG product-space path.

## Decision

Reject the phased solver portfolio without breadth expansion or switch-time
tuning. Retain the request-scoped global switch default-off as a diagnostic.

Do not implement dual per-iteration solver racing. A correct race requires
snapshot/restore of complete worker trust and accepted/best state, homogeneous
global branch selection, and approximately doubled local-solve work. The
cheaper single-trajectory transfer already fails materially, so that additional
complexity and cost are not justified by current evidence.

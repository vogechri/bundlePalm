# Terminal Correction K4/K16 Transfer Plan

Date: 2026-08-21
Status: active; sentinel gate passed
Source checkpoint: `caacacd`

This is the restart contract for transferring the frozen one-step terminal
Schur correction from K24 to the K4 resource and K16 latency endpoints.

## Frozen Policy

Use one final distributed correction with:

- camera/landmark damping `0.005859375`;
- geometric doubling after rejection;
- `bsr_low_memory` and Jacobi PCG;
- relative residual tolerance `1e-6`;
- one maximum accepted correction;
- positive damped model gain and lower physical SSE;
- rejected/nonconverged attempts preserving the input state.

No K-specific damping, scene routing, repeated correction, or Stage-C retuning
is allowed.

## Reference Endpoints

Use the frozen C1+C5 K4/K16 I30 rows under
`benchmark_results/stage_c_scaling_confirmation_k4_16_i30/`.
They use direct left-SE3, shared-only block/full product-space semantics,
Schur-PCG, persistent DABA trust, and the global delayed-C5 policy.

All 30 1DSfM state files are numerically sane. BAL state serialization is not
reliable: 34/58 K4/K16 files contain extreme camera values. State validity is
therefore determined only by independently evaluated physical SSE.

## Completed Sentinel Gate

Roman, Trafalgar, and BAL52 use reloadable states. BAL3068 uses exact in-process
I30 reruns because both saved states are corrupt.

| K | Family | Corrected/prestate | Wins |
|---:|---|---:|---:|
| 4 | 1DSfM pair | 0.822335821 | 2/2 |
| 4 | BAL52/3068 | 0.976151375 | 2/2 |
| 16 | 1DSfM pair | 0.775883249 | 2/2 |
| 16 | BAL52/3068 | 0.973048197 | 2/2 |

All eight corrections accept at initial damping and converge below `1e-6`.
BAL3068 fresh prestates are `0.991842x`/`0.988448x` their historical K4/K16
rows, so final reporting must distinguish correction/prestate from
corrected/historical endpoint.

Artifact:
`benchmark_results/terminal_correction_scaling_sentinels/report.md`.

## Full Transfer

1. Run zero-iteration correction attempts from all 88 K4/K16 saved states.
2. Compare independently evaluated pre-correction SSE to the historical row.
3. A state is valid only when relative mismatch is at most `1e-6`.
4. Quarantine invalid rows and rerun their exact Stage-C I30 trajectory with
   the frozen correction attached in-process.
5. Merge valid reload rows and in-process recovery rows by `(scene, K)`.
6. Report correction/prestate and corrected/historical endpoint separately.
7. Compare corrected endpoints to Ceres left-SE3 and to K24 terminal correction
   where the comparison is meaningful.

## Promotion Gate

For both K4 and K16 and both benchmark families:

- all rows produce a valid final result or safe no-op;
- no accepted nonconverged solve;
- no correction/prestate regression;
- geometric and summed correction/prestate ratios are at most one;
- worst case, fallback damping, residual, time, and peak RSS are reported;
- no policy changes after sentinel selection.

## Crash Recovery

Outputs live under:

```text
benchmark_results/terminal_correction_scaling_all/
  reload_k4/
  reload_k16/
  recovery_k4/
  recovery_k16/
  summary.json
  report.md
```

Runners must use `OVERWRITE=0`, append one JSONL row per completed case, and
resume from status plus JSONL completion. Logs, memory files, manifests with
absolute paths, symlinked states, and NPZ outputs remain untracked.

## Phase Status

| Phase | Status | Artifact |
|---|---|---|
| Sentinel K4/K16 | passed | `terminal_correction_scaling_sentinels/report.md` |
| Full reload inventory | complete | `recovery_inventory.json` |
| Invalid-state in-process recovery | active | K4/K16: BAL1490, 1778, 245, 427, 744, 951 |
| Merged all-15/all-29 decision | blocked by recovery | -- |

## Immediate Next Action

Resume the six K4 and six K16 in-process recoveries with `OVERWRITE=0`, then run
the merged analyzer. All 30 1DSfM states reproduce after exact structured state
resolution; the six BAL IDs above are the only physical-SSE-invalid states at
both K values.

# Ladybug-1723 DRS Recovery

## Recovered known-good configuration

- Dataset: `problem-1723-156502-pre.txt.bz2`
- Iterations: 90
- Clusters: 30
- Accelerator: Nesterov
- Scaling: `jacobi_gmean`
- Partitioner: `landmark_scalable_stable`
- Explicit cost landmarks: off
- Strict trial safeguard: off
- Final independent native/common cost validation: on

Run with an already-started DRS worker on ports 5556/5557:

```bash
serverTest/run_drs1723_recovered.sh
```

For a 30-iteration verification:

```bash
ITERATIONS=30 OUTPUT_DIR="$PWD/benchmark_results/drs1723_recovered_i30" \
  serverTest/run_drs1723_recovered.sh
```

## Partition provenance

### Stability-focused scalable partition

This is the partition used by the best recent atomic run:

```text
Residual balance bounds [22397, 22851], violation 0
Weak camera-cluster counts 1..19: 1 0 4 2 4 8 15 12 7 235 132 121 135 130 112 111 97 89 58
Additional camera copies: 9086
Maximum cameras in a cluster: 691
```

Known result:

- pixel SSE through iteration 30: `763,152`
- pixel SSE through iteration 60: `761,959`
- final pixel SSE: `761,438`
- final mean reprojection error: `0.758406 px`

Artifacts:

- `benchmark_results/ours_k30_atomic/results.jsonl`
- `benchmark_results/ours_k30_atomic/states/1723_k30_i90.npz`
- `benchmark_results/ours_k30_atomic/logs/1723_k30_i90.log`

### Standard scalable partition

This is the partition from the command using
`BUNDLE_PALM_CLUSTERING=landmark_scalable`:

```text
Residual balance bounds [22397, 22851], violation 0
Weak camera-cluster counts 1..19: 5 4 3 3 6 4 6 8 7 157 122 113 96 87 92 101 82 69 46
Additional camera copies: 8887
Maximum cameras in a cluster: 657
```

Historical Nesterov result:

- pixel SSE through iteration 30: `768,739`
- pixel SSE through iteration 60: `766,111`
- final pixel SSE: `765,535`

This partition is reproducible from committed partitioner source. It is not a
corrupt or random partition, but it is not the partition used by the later
`761,438` result.

## Recovered regression

A controlled 30-iteration test on the standard scalable partition found:

| Explicit landmarks | Strict safeguard | Best pixel SSE | Best iteration |
|---|---|---:|---:|
| off | off | 768,241 | 26 |
| off | on | 770,872 | 20 |
| on | off | 881,064 | 17 |
| on | on | 815,462 | 24 |

The main quality regression came from enabling explicit coordinator landmark
buffers during trial-cost evaluation. It changes line-search acceptance and
therefore the optimization trajectory. The strict safeguard has a smaller
secondary effect.

Historical behavior is restored as the default in `client_acc.py`:

```text
BUNDLE_PALM_EXPLICIT_COST_LANDMARKS=0
BUNDLE_PALM_STRICT_TRIAL_SAFEGUARD=0
BUNDLE_PALM_REQUIRE_COMMON_COST_MATCH=1
```

The final common evaluator remains enabled and validates the saved physical
state independently. The recovered 30-iteration probe had relative
native/common pixel-SSE disagreement `6.54e-9`.

Recovered probe artifacts:

- `benchmark_results/drs1723_recovery_probe/results.jsonl`
- `benchmark_results/drs1723_recovery_probe/state.npz`
- `benchmark_results/drs1723_recovery_probe/run.log`

# DABA Partition Comparison

## Scope

This comparison evaluates partitions through the local BA problems they induce,
not through each partitioner's native objective. The five BAL scenes are 52,
245, 394, 871, and 1723 at `K = 10, 20, 30`. Both methods are evaluated by
`analyze_partition_quality.py` using identical endpoint-owner replication:

- our partition owns each landmark once, so every observation has one owner;
- DABA owns cameras and landmarks independently, so an observation is copied
  to both endpoint owners when they differ;
- camera copies and weak camera incidences are computed from the resulting
  local observation problems.

DABA's default clustering mode completed 13/15 cases. Scene 52 at K20 and K30
deterministically aborted with `cudaErrorInvalidValue` inside a Thrust
device-to-device copy. DABA's explicit `memory_efficient=true` mode completed
those two cases. They are included below but marked as a different mode.

## Aggregate Result

| Metric, DABA / ours | Geomean ratio | DABA wins | Interpretation |
|---|---:|---:|---|
| Partition time | **0.152x** | 15/15 | DABA is 6.57x faster |
| Additional camera copies | **0.835x** | 14/15 | DABA reduces camera replication 16.5% |
| Maximum cameras per worker | **0.834x** | 12/15, 3 ties | DABA lowers peak camera count |
| Local observation copies | 1.415x | 0/15 | DABA creates 41.45% more local residual work |
| Observation-load CV | 44.21x | 0/15 | DABA is much less balanced by residual count |

Across the 15 cases, our partitions create 16,312,986 local observation copies.
DABA creates 22,723,025, including 6,410,039 cross-owner duplicates.

| Weak camera-cluster incidence | Ours | DABA | Ratio |
|---|---:|---:|---:|
| Degree < 5 landmarks | 53 | 16,749 | 316.0x |
| Degree < 10 landmarks | 229 | 24,157 | 105.5x |
| Degree < 20 landmarks | 5,273 | 32,183 | 6.10x |

## Per-Case Tradeoff

| Scene | K | DABA mode | Extra observation copies | Extra camera copies, ours / DABA | Max cameras, ours / DABA | Load CV, ours / DABA |
|---:|---:|---|---:|---:|---:|---:|
| 52 | 10 | default | 197,148 | 458 / 462 | 52 / 52 | 0.0178 / 0.3588 |
| 52 | 20 | memory-efficient | 248,577 | 958 / 954 | 52 / 52 | 0.0112 / 0.2471 |
| 52 | 30 | memory-efficient | 269,101 | 1,433 / 1,224 | 52 / 52 | 0.0089 / 0.5687 |
| 245 | 10 | default | 396,209 | 1,767 / 1,580 | 244 / 243 | 0.0086 / 0.4646 |
| 245 | 20 | default | 541,245 | 3,522 / 3,122 | 245 / 212 | 0.0090 / 0.5527 |
| 245 | 30 | default | 606,384 | 4,947 / 4,458 | 244 / 210 | 0.0086 / 0.5881 |
| 394 | 10 | default | 227,153 | 3,254 / 3,214 | 392 / 380 | 0.0092 / 0.3704 |
| 394 | 20 | default | 273,439 | 6,243 / 6,073 | 389 / 384 | 0.0091 / 0.4176 |
| 394 | 30 | default | 297,221 | 9,597 / 8,657 | 389 / 382 | 0.0092 / 0.4889 |
| 871 | 10 | default | 770,412 | 5,296 / 4,137 | 831 / 676 | 0.0081 / 0.4618 |
| 871 | 20 | default | 1,094,403 | 9,476 / 7,650 | 737 / 629 | 0.0083 / 0.6042 |
| 871 | 30 | default | 1,253,443 | 14,107 / 10,054 | 776 / 570 | 0.0083 / 0.6345 |
| 1723 | 10 | default | 56,647 | 3,900 / 2,876 | 879 / 571 | 0.0091 / 0.2152 |
| 1723 | 20 | default | 80,346 | 6,146 / 3,997 | 711 / 415 | 0.0085 / 0.2425 |
| 1723 | 30 | default | 98,311 | 9,086 / 5,267 | 691 / 339 | 0.0086 / 0.3025 |

## Conclusion

DABA's Louvain partition is attractive for DABA's own endpoint-owner and
peer-to-peer communication model: it is fast and reduces camera replication.
It is not a drop-in improvement for our landmark-owned Schur workers. Under our
local-solve accounting it duplicates substantially more residuals, produces
severe residual-load imbalance, and creates many weak local camera blocks.

The defensible claim is therefore not that our partition is universally
better. It is that our residual-balanced landmark ownership is better aligned
with one-step local Schur solves, while DABA optimizes a different communication
topology. A solver-quality comparison using DABA labels remains useful, but it
must include duplicated residual work and cannot attribute the result to
partition quality alone.

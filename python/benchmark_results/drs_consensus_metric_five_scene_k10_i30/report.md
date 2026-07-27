# Five-Scene DRS Consensus Projection Metric Ablation

Date: 2026-07-27

## Protocol

- scenes: 1723, 52, 245, 394, 871
- K10, 30 DRS outer iterations, one local nonlinear step
- standard Snavely pixel objective
- fixed landmark-owned partition settings, Jacobi geometric-mean scaling,
  acceleration, relative DRE/primal safeguard, and local full-metric prox
- changed only the metric used in reflected-camera projection and DRE terms

All modes still transmit the full worker metric. This intentionally isolates
projection quality; communication compression is not tested here.

## Aggregate Results

| Projection | Geomean pixel SSE | Relative to full | Hard rejections | Fallbacks | Nonfinite trials | Total time s |
|---|---:|---:|---:|---:|---:|---:|
| Arithmetic | 4,630,080 | +330.1% | 52 | 50 | 4 | 140.03 |
| Scalar per copy | 1,209,725 | +12.36% | 20 | 46 | 0 | 144.51 |
| Diagonal | 1,148,814 | +6.70% | 15 | 46 | 0 | 153.63 |
| Full regularized 9x9 block | **1,076,627** | reference | **14** | **42** | **0** | 155.35 |

Full blocks achieve the lowest independently evaluated pixel SSE on every
scene:

| Scene | Arithmetic | Scalar | Diagonal | Full block |
|---|---:|---:|---:|---:|
| 1723 | 124,050,155 | 1,055,152 | 823,636 | **772,097** |
| 52 | 641,472 | 513,212 | 525,381 | **484,199** |
| 245 | 2,595,895 | 2,058,914 | 2,049,857 | **1,809,129** |
| 394 | 2,168,946 | 643,343 | 628,890 | **611,964** |
| 871 | 4,749,325 | 3,611,937 | 3,587,074 | **3,494,903** |

All 20 saved physical states independently reproduce their recorded standard
pixel SSE exactly.

## Interpretation

The full metric advantage generalizes beyond Ladybug-1723. Direction-dependent
camera observability and off-diagonal coupling both matter: diagonal consensus
is consistently better than or close to scalar, while full blocks improve all
five scenes and reduce safeguard burden.

Arithmetic is not a credible baseline for this nonlinear decomposition on
1723. Its finite saved result is the initial best state while its current
trajectory becomes nonfinite. The other four scenes complete, but aggregate
quality remains far worse.

Runtime differences are not interpretable as communication savings because
the full metric payload is sent in every mode. A later compression experiment
should compare state only, diagonal metric, compressed block metric, and full
block metric bytes at matched quality.

This breadth gate supports full block-metric camera consensus as the first
method contribution and fixes `full` as the consensus mode for the subsequent
pixel-versus-DABA-ray objective ablation.
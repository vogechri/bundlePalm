# K24 Combined Stack Versus Ceres Left-SE3

Artifact-only benchmark over all 15 SfM_Init-derived 1DSfM scenes and all 29 BAL problems. The primary method is the retained K24/I120 copied-baseline combined stack. Ceres left-SE3 I90/T16 is a centralized quality reference, not a replacement for the distributed method. All quality values use independently evaluated standard Snavely pixel SSE.

| Family | Complete D/C | Candidate/Ceres geo | Summed | W/T/L | Worst | Candidate/direct geo | Summed | W/T/L | Opt. s D/C/Ceres | I60 sel/no-op | I90 sel/no-op | Transport GiB | Peak RSS GiB C/W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1DSfM-15 | 15/15 | 1.169700 | 1.046518 | 3/0/12 | 2.120880 | 0.762902 | 0.700779 | 14/0/1 | 436.0/592.3/219.5 | 15/0 | 8/7 | 19.051 | 1.764/1.819 |
| BAL-29 | 29/29 | 1.001181 | 0.997696 | 11/0/18 | 1.035083 | 1.000131 | 1.000417 | 5/15/9 | 2039.2/2326.7/1803.8 | 1/28 | 0/29 | 73.186 | 6.889/8.496 |

The BAL geometric mean is slightly above Ceres while summed SSE is slightly below Ceres; both statistics are therefore retained. Timing boundaries differ: DRS uses coordinator-reported optimization time, while Ceres uses native solve time. No matched-work speedup is claimed.

The combined candidate completes every scene with no recovery exhaustion. The matched direct control also completes every scene. See `per_scene.md` for the complete ratio vectors.

Campaign checkpoint: `f4feabe`. Raw rows do not embed an execution commit; this is the repository checkpoint that packages the campaign.

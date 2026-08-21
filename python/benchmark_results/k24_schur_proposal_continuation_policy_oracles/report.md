# K24 Schur Proposal Continuation Policy Oracles

Existing exact all-15 trajectories evaluate two non-invasive policies for the retained I90 one-action proposal.

| Policy | Geometric delivered/control | W/T/L | Work / retained run | State requirement |
|---|---:|---:|---:|---|
| Retained selected continuation + I91 rebase | 0.884651719 | 13/1/1 | 1.00x | existing |
| Best-state-only proposal, ordinary continuation | 0.934688146 | 8/7/0 | approximately 1.00x | proposal checkpoint only |
| Exact ordinary/proposal endpoint race oracle | 0.884026446 | 13/2/0 | at least 1.25x outer work | full coordinator/worker snapshots |

Best-state-only delivery is safe but discards too much continuation gain. The endpoint race removes only Gendarmenmarkt's `1.010662x` loss and improves the geometric aggregate by about `0.071%` relative while duplicating I91-I120. Current maintained source has nominal landmark snapshots but no full coordinator/worker DRS snapshot restore. Reconstructing that infrastructure and paying at least 25% more outer work is not justified by this oracle ceiling. Retain the bounded-loss proposal continuation component; do not implement a shadow branch or replace it with best-state-only delivery.

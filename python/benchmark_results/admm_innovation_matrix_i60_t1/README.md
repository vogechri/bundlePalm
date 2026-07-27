# ADMM Innovation Experiment Results

Current cross-experiment status and next work:

- [Experiment dashboard](../EXPERIMENT_STATUS.md)

## Final comparison protocol

- 60 outer iterations
- 20 local Ceres iterations
- one Ceres/Eigen thread per cluster
- final cluster counts: K10 and K30
- standard BAL L2 pixel objective
- ADMM baseline uses over-relaxation alpha = 1.5 and adaptive scalar penalty

K5 and K15 rows in this directory are retained exploratory baseline results.
They are not part of the final innovation ranking.

## Files

- `../admm_five_scene_i30_k20/report.md`: current focused K20 stability
	comparison (five scenes, 30 outer iterations, one local LM step)
- `report_jacobi_five_scenes.md`: focused K30 baseline/Jacobi comparison with
	current-versus-best pixel-SSE divergence diagnostics
- `status.tsv`: live per-case status, elapsed time, and peak RSS
- `report.partial.md`: readable summary of completed rows at its last refresh
- `report.md`: final paired report, generated when the active matrix finishes
- `baseline.jsonl`: complete baseline metadata and per-iteration trajectories
- `<variant>.jsonl`: corresponding innovation results once implemented and run
- `states/`: exact best camera/landmark states
- `logs/`: coordinator and worker logs
- `memory/`: `/usr/bin/time -v` resource measurements

The current baseline runner stops after the K10/K30 baseline. Sensitivity
controls (capped/fixed penalty and no over-relaxation) and plain DRS are not in
the primary innovation queue.

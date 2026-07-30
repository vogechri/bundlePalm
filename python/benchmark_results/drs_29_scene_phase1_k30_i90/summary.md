# Phase-1 One-Factor Screen: Human Summary

Updated: 2026-07-30

## Protocol

The screen compares 14 one-factor variants against corrected plain DRS on all
29 local BAL scenes. Every run uses K30, 90 outer iterations, one local
nonlinear step, the standard Snavely pixel objective, full block consensus and
Jacobi scaling unless that factor is the ablation, reset-DABA trust, curvature
0.1, curvature-only recovery, and no outer acceleration unless selected.

The machine-generated table is in [report.md](report.md). Ratios below one are
better than plain DRS. W/T/L uses a 0.1% SSE tolerance.

## Aggregate Results

| Variant | SSE ratio | W/T/L | Optimization-time ratio | Interpretation |
|---|---:|---:|---:|---|
| Nesterov ternary `{0,0.5,1}` | **0.980946** | **25/4/0** | 2.9117 | Best equal-iteration quality, but 2.08x oracle calls and nearly 3x time |
| DRS trust | **0.982090** | **24/2/3** | 1.4812 | Best one-oracle-call method in this screen |
| Persistent DABA trust | 0.988142 | 19/3/7 | 1.2554 | Strong aggregate improvement, less consistent than DRS trust |
| Regularization recovery | 0.990740 | 8/17/4 | 1.3544 | Useful but concentrated improvement |
| Curvature 0.05 | 0.993029 | 17/8/4 | 1.3108 | Lower initial curvature usually helps |
| Stable partition | 0.994777 | 10/12/7 | 1.6597 | Small aggregate gain with meaningful setup/runtime cost |
| Final-only landmark polishing | 0.995113 | 14/15/0 | 1.6351 | Safe output-quality improvement; trajectory neutral |
| One local landmark-refinement step | 0.996602 | 20/2/7 | 1.7375 | Broad but small quality gain at substantial cost |
| No camera scaling | 1.000422 | 3/16/10 | 1.4986 | Jacobi scaling is quality-neutral in aggregate but materially faster here |
| Curvature 0.2 | 1.006959 | 5/10/14 | 1.2067 | Over-damping hurts; one recovery exhaustion |
| Schur-PCG local solver | 1.037650 | 3/0/26 | 1.3652 | Worse than finite local Nesterov under this unmatched inner-work policy |
| Diagonal consensus | 1.290453 | 0/0/29 | 0.7924 | Fails the quality/stability gate; 18 recovery exhaustions |
| Scalar consensus | 1.524001 | 1/0/28 | 0.9647 | Fails the quality/stability gate; 12 recovery exhaustions |
| Arithmetic consensus | 10.035127 | 0/0/29 | 0.4070 | Catastrophically unsuitable; 25 recovery exhaustions |

The apparent speedups for diagonal/scalar/arithmetic consensus are not useful:
many runs terminate early after recovery exhaustion. Full regularized 9x9 block
consensus is strongly supported as a core method component.

## Decisions

1. **Keep full block consensus.** Every reduced consensus metric loses badly,
   often through early recovery exhaustion.
2. **Promote DRS trust to the cumulative ladder.** It gives nearly the same
   equal-iteration gain as acceleration without doubling proximal calls.
3. **Keep binary Nesterov as the default accelerator.** The ternary grid is only
   about 0.13% better geometrically than binary Nesterov's Phase-0 ratio
   (`0.982266`) while adding a third trial option and substantially more work.
4. **Test regularization recovery and curvature 0.05 in the cumulative method.**
   Their isolated gains can interact with trust policy and must be forward
   selected rather than blindly combined.
5. **Treat final polishing as output postprocessing.** It improves or ties every
   scene without changing the DRS path.
6. **Do not promote Schur-PCG from this row.** A fair local-solver claim still
   requires matched products or residual tolerances.
7. **Retain `landmark_scalable` as the default partition.** The stable variant's
   small isolated quality gain does not justify its current runtime overhead.

## Validation Status

The optimization matrix has 14 variants x 29 scenes = 406 final JSON results
and 406 state paths. Overlapping automatic resumes had concurrently written some
NPZ paths and corrupted 58 ZIP archives. `save_bal_state` now uses a
same-directory temporary output plus atomic `os.replace`, and the targeted
[58-case repair](../../serverTest/repair_drs_29_scene_phase1_states.sh) completed.

All 406 archives are readable. Independent evaluation of every saved camera and
landmark state reproduces its recorded canonical Snavely pixel SSE exactly;
the worst relative error is `0.0`. The validation gate therefore passes.

Reduced consensus metrics terminate early by recovery exhaustion on many scenes:
arithmetic 25/29, diagonal 18/29, and scalar 12/29. Curvature 0.2 has one
recovery exhaustion. These are algorithm outcomes, not missing or corrupt rows.

## Next Experiment

Build a cumulative forward-selection ladder beginning with full blocks,
Jacobi scaling, and corrected safeguards, then test compatible additions in
this order:

1. DRS trust;
2. regularization recovery versus curvature-only recovery;
3. curvature 0.05 versus 0.1;
4. binary Nesterov acceleration;
5. final-only landmark polishing.

After selecting the K30 cumulative method, confirm plain, accelerated, and
cumulative variants at K10 and K20.
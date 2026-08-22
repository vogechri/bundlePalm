# K24 Ordinary DRS Candidate Landmark-Response Diagnostic

A behavior-neutral diagnostic evaluates the nominal DRS consensus candidate at displayed I30, I60, and I89 after three fixed-camera landmark response steps. Worker-owned accepted landmarks are materialized before refinement; a versioned snapshot restores worker state exactly. The promoted I90 joint camera/landmark proposal remains the only applied response.

| Scene | I30 refined/unrefined | I60 | I89 | Primal decision flips | Round-trip error |
|---|---:|---:|---:|---:|---:|
| Roman Forum | 0.987683442 | 0.989611194 | 0.995622686 | 0/3 | 0 |
| Trafalgar | 0.982916948 | 0.958702363 | 0.988262202 | 0/3 | 0 |
| BAL52 | 0.999552845 | 0.999594793 | 0.999511463 | 0/3 | 0 |

All trajectories and endpoint arrays are exact. Landmark response exposes modest accepted-state descent, especially Trafalgar I60, but it does not change the existing primal safeguard decision at any checkpoint. Therefore stale landmark scoring is not the cause of ordinary DRS rejections, and the inconsistent legacy per-iteration consensus-refinement path should remain disabled. The next bounded test is the previously frozen alternate I60 checkpoint using the new joint camera/landmark Schur proposal; do not introduce a timing sweep.

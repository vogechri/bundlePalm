# K24 Model-Optimal Two-Step Schur Oracle Sentinel

A behavior-neutral I90 oracle keeps the retained first block-Jacobi residual action and line-minimizes the second preconditioned residual direction exactly in the frozen damped Schur quadratic. It compares retained depth one, rejected unit depth two, and model-optimal depth two through the same eight worker-SSE scales and `1e-3` progress floor. Only depth one is applied.

| Scene | Depth-one SSE | Unit-depth-two / one | Model-optimal scale | Model-optimal / one | Unit model decrease | Optimal model decrease | Trajectory/state |
|---|---:|---:|---:|---:|---:|---:|---|
| gendarmenmarkt | 0.993459701 | 1.000710342 | 0.964221 | 1.000677463 | 449.814198 | 450.434418 | exact |
| roman_forum | 0.886889776 | 1.008819861 | 1.074826 | 1.010264329 | 29890.975705 | 30036.547690 | exact |
| bal52 | 1.000000000 | 1.000000000 | 3.356344 | 1.000000000 | 0.146464 | 0.288818 | exact |

Exact quadratic line minimization is rejected as an applied second action. It improves the frozen Schur model as designed but does not improve physical worker SSE over depth one on either 1DSfM sentinel; BAL52 still declines. The near-unit 1DSfM coefficients show that stationary depth failed because the second direction itself does not transfer to physical SSE, not because its unit coefficient was materially wrong. Do not broaden or iterate this route.

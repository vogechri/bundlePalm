# K24 Two-Step Schur Residual Oracle Sentinel

A behavior-neutral I90 oracle compares the retained one-step block-Jacobi Schur residual proposal with two fixed unit block-Jacobi residual corrections. Both use the same eight geometric SSE scales and `1e-3` progress floor. The depth-two candidate is telemetry-only and never applied.

| Scene | One-step scale | One-step SSE ratio | Two-step scale | Two-step SSE ratio | Two/one candidate | Residual contraction | Trajectory/state |
|---|---:|---:|---:|---:|---:|---:|---|
| gendarmenmarkt | 0.25 | 0.993459701 | 0.25 | 0.994165397 | 1.000710342 | 4.923881 | exact |
| roman_forum | 0.5 | 0.886889776 | 0.5 | 0.894712021 | 1.008819861 | 238.364125 | exact |
| bal52 | declined | 1.000000000 | declined | 1.000000000 | 1.000000000 | 0.844514 | exact |

The second unit Jacobi action is rejected. It worsens actual candidate SSE on both 1DSfM scenes and strongly expands the real Schur residual, despite contracting the synthetic SPD test and BAL52 residual. Do not apply it, broaden it, or sweep depth/relaxation. The useful conclusion is that one preconditioned residual action is a direction repair, while stationary repetition is unstable on the real restricted Schur systems.

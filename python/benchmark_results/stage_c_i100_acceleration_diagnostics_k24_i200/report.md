# I100 Outer-Acceleration Diagnostic

Date: 2026-08-12

## Question

Does the late hybrid trajectory lose quality because Themelis acceleration
should be disabled after I100, or because stale acceleration history should be
restarted? Proposal damping 0.5 remains active in every arm so the acceleration
mechanism is isolated.

The five diagnostics are Madrid, Montreal Notre Dame, Roman Forum, Tower of
London, and Trafalgar. All use K24/I200 and share bitwise-identical prefixes
through I100.

| Policy | I120 / base-I200 | I150 / base-I200 | I200 / base-I200 | Completed | Recovery exhausted |
|---|---:|---:|---:|---:|---:|
| Permanent acceleration | 1.085001 | 1.066312 | 1.052528 | 5/5 | 0 |
| Disable acceleration after I100 | 1.093313 | 1.087785 | incomplete | 4/5 | 1 |
| Reset acceleration once at I100 | 1.088045 | 1.067341 | 1.051481 | 5/5 | 0 |

The cutoff arm is `1.02014x` permanent acceleration at the last fully matched
I150 checkpoint and later exhausts recovery on Trafalgar at I159. The isolated
restart fires at displayed I101 and changes no trust, curvature, or proposal
state. At I200 restart/permanent is `0.999005x` geometrically: Madrid and Roman
improve slightly, Tower regresses slightly, Montreal is tied, and Trafalgar is
identical.

| Scene | Permanent / base-I200 | Restart / base-I200 | Restart / permanent |
|---|---:|---:|---:|
| Madrid | 1.028828 | 1.025360 | 0.996629 |
| Montreal Notre Dame | 1.004346 | 1.004344 | 0.999997 |
| Roman Forum | 1.052867 | 1.047783 | 0.995172 |
| Tower of London | 1.226268 | 1.230252 | 1.003249 |
| Trafalgar | 0.968241 | 0.968241 | 1.000000 |

## Decision

Keep acceleration active after I100. Disabling it slows late contraction and
introduces an unsafe Trafalgar tail. Stale acceleration history is not the main
cause: an isolated I100 reset is aggregate-neutral and does not repair Tower.
The remaining long-horizon DRS gap on these diagnostics is a property of the
modified trajectory/basin rather than accumulated acceleration momentum.

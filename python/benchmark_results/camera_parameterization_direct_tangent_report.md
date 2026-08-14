# Direct-Tangent Camera Parameterization Study

Date: 2026-08-14

## Micro Lie Convention Audit

For the BAL world-to-camera action `p_c = R p_w + t`, the production
`se3_left` mode matches Sola's left/global perturbation with tangent order
`[rho, theta]`:

```
R+ = Exp(theta) R
t+ = Exp(theta) t + J_l(theta) rho
```

The direct state Jacobian is `dr/dtheta = J_l(r)^-1`, `dt/drho = I`, and
`dt/dtheta = -[t]x`. The existing `se3_right` update matches the paper's
right/local perturbation. A new `so3_left` mode implements the product manifold
`SO(3) x R3` while retaining the common tangent order
`[translation, rotation, intrinsics]`:

```
R+ = Exp(theta) R
t+ = t + delta_t
```

All three state Jacobians and induced point-action Jacobians pass central finite
differences. The worker now uses one mode-correct tangent-map helper for direct
normal-equation assembly and proximal metric transformation.

## Frozen I30 Evaluation

All runs use K24/I30, the promoted C1+C5 stack, direct tangent equations, one
global policy, and Schur-PCG tolerance `1e-2`. Product-SO3 requires a global
maximum of 1000 PCG iterations: Trafalgar needs up to 690 iterations. At the
old cap 400, four accepted Trafalgar outer steps had residuals above tolerance,
so those endpoints are excluded from promotion evidence.

### Development

| Family | Product-SO3 / left-SE3 SSE | Summed SSE | W/T/L | Worst | Time |
|---|---:|---:|---:|---:|---:|
| six 1DSfM | 0.987432 | 0.944769 | 2/0/4 | 1.084378 | 1.036779 |
| five BAL | 0.997955 | 0.998523 | 4/0/1 | 1.001256 | 0.974646 |

### Confirmation

| Family | Product-SO3 / left-SE3 SSE | Summed SSE | W/T/L | Worst | Time |
|---|---:|---:|---:|---:|---:|
| held-out nine 1DSfM | 1.002320 | 1.030994 | 2/0/7 | 1.246462 | 0.974534 |
| all 29 BAL | 0.996646 | 0.995410 | 23/0/6 | 1.001555 | 0.987873 |
| combined all 15 1DSfM | 0.996338 | 0.981871 | 4/0/11 | 1.246462 | 1.011984 |

The 1DSfM aggregate is positive only because NYC Library (`0.580161x`),
Trafalgar (`0.840940x`), and Roman (`0.956749x`) offset broad losses. Piazza is
`1.246462x`, Notre Dame `1.119742x`, Ellis Island `1.088793x`, and Piccadilly
`1.084378x`. BAL transfer is broad and bounded: the worst all-29 loss is
BAL1266 at `1.001555x`; BAL951 is `0.981957x`.

Right-SE3 is rejected at the four-scene I30 sentinel: its Roman/Trafalgar
geometric ratio is `1.398362x` left-SE3 despite a favorable BAL ratio
`0.981168x`.

## I90 Persistence Diagnostic

A true four-scene I90 rerun uses the same global PCG cap 1000. Roman is
`0.999458x` left-SE3, Trafalgar is `1.060555x`, and their geometric ratio is
`1.029553x`: the large Trafalgar I30 gain reverses. BAL1490 remains slightly
better at `0.998090x`. BAL3068 is not a matched endpoint comparison because the
left-SE3 control exhausts recovery at I54 while product-SO3 completes I90.

## Decision

Do not replace left-SE3 globally. Retain `so3_left` default-off as a coherent
product-manifold ablation and a promising BAL-oriented component: all-29 BAL
improves with 23/29 wins, bounded `0.156%` worst loss, and slightly lower time.
It is not a common 1DSfM/BAL quality policy because held-out 1DSfM regresses,
large scene tails remain, and the I30 Trafalgar gain reverses by I90.

Do not tune parameterization by scene or retune the PCG tolerance. Future use
should be a frozen 2x2 interaction study with a genuinely orthogonal mechanism,
or a BAL-specific publication ablation explicitly labeled as such.

## Product-SO3 Tuning Follow-Up

The frozen C1 x C5 development factorial shows C1 is essential and C5 is the
interaction-sensitive component. Under product-SO3, C1/plain is `0.772264x` on
six 1DSfM and `0.955724x` on five BAL, winning every scene. C5/plain is
`0.970051x` on 1DSfM and exactly inactive on BAL. Adding C5 after C1 is
`0.988354x` geometrically on 1DSfM but loses four of six: it helps Trafalgar
`17.74%` and Piccadilly `3.49%`, while hurting Vienna `12.74%`, Union `2.46%`,
and Gendarmenmarkt `1.62%`. BAL C1+C5/C1 is `0.999636x` and effectively
neutral.

A global delayed-C5 sweep tested starts `10/15/20` after the existing start-5
control. Start 15 was the bounded development leader (`0.985244x` left-SE3,
worst `1.039474x`), but failed held-out transfer: combined all-15 becomes
`1.008025x` left-SE3 and `1.011730x` start 5. It repairs several tails but loses
much of NYC Library's gain. Close activation timing without threshold tuning.

Camera metric strength `10/25/35/40/45/50/75` was then tested with C1+C5 start
5 fixed. Scale 35 is the development leader: `0.945358x` left-SE3 on six
1DSfM, five wins, and `0.997190x` on five BAL. Frozen confirmation is mixed:
all-15 reaches `0.994506x` left-SE3 but has Piazza `1.316080x` and Tower
`1.193405x`, costs `2.031953x`, and all-29 BAL is `0.997011x` left-SE3 but
`1.000366x` the scale-25 product control. Do not promote scale 35.

A coherent translation-versus-rotation tangent metric ratio was implemented as
an exact stored-coordinate congruence used identically by worker proximal and
coordinator consensus metrics. Ratio 1 is bitwise neutral. Ratios `0.5/2`
regress six-scene 1DSfM to `1.295373x/1.502242x` and BAL5 to
`1.038121x/1.039688x` versus ratio 1. Remove subspace imbalance from the active
tuning queue; retain the default-off diagnostic control.

Current conclusion: parameter tuning can improve aggregate I30 quality but has
not repaired the held-out 1DSfM basin tails or I90 Trafalgar reversal. C1 is
the robust product-SO3 component; C5 timing, scalar camera metric strength, and
translation/rotation metric ratio are now bounded and closed. Next prioritize
product-aware trust-envelope diagnostics or the structurally different
camera-center product manifold, not further C5/metric grids.

The product-aware trust envelope is now also closed. Behavior-neutral telemetry
shows the default DABA trajectory is genuinely cap-limited: Roman reaches the
`1e6` maximum by I14 and many workers remain capped. A frozen maximum-radius
grid `{1e3,1e4,1e5,1e6}` with initial DABA cap 100 finds `1e5` as a strong
development candidate: versus product control it is `0.976718x` on six 1DSfM
and `0.997610x` on BAL5, with 10/11 wins. Frozen confirmation reverses on
1DSfM: all-15 is `1.010903x` product control and `1.007201x` left-SE3, with
Piazza `1.238627x` and Madrid `1.190756x`. BAL29 remains favorable at
`0.999542x` product control and `0.996190x` left-SE3. Do not promote or
interpolate trust caps from held-out outcomes.

All scalar/product-specific tuning directions attempted so far now fail the
common held-out criterion. The next justified mechanism is the camera-center
product manifold `SO(3) x R3`, representing `C = -R^T t` so rotation and camera
position are independent geometric state variables. This is structural rather
than another scene-dependent parameter grid.

## Camera-Center Product Manifold

A new `so3_center_left` mode keeps physical BAL storage `[R,t]` but defines the
product state using camera center `C = -R^T t`:

```
R+ = Exp(theta) R
C+ = C + delta_C
t+ = -R+ C+
```

Its direct stored-camera Jacobian uses `dt/d(delta_C) = -R` and
`dt/dtheta = -[t]x`. State, point-action, and exact center-displacement finite
differences pass. No parameter was tuned before evaluation.

The four-scene sentinel is promising: camera-center/left-SE3 is `0.964637x` on
Roman/Trafalgar and `0.953493x` on BAL1490/3068, with Roman `1.032978x` and
BAL1490 `1.000074x` the bounded losses. Frozen development rejects broad
transfer. On six 1DSfM it is `1.011018x` geometric, W/T/L `2/0/4`, worst
Gendarmenmarkt `1.103175x`; on BAL5 it is `1.005759x`, `1/0/4`, worst BAL245
`1.025866x`. It helps Trafalgar `9.92%` and Union `7.56%`, but hurts Vienna
`7.95%` and other sentinels.

Reject camera-center product without parameter tuning or held-out expansion.
The strong two-scene signal is cohort-specific, and scalar trust/metric tuning
has already shown poor transfer for the independent-translation product mode.
Keep the implementation default-off as a mathematically coherent ablation.

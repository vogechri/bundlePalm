# K24 Joint Camera/Landmark Proposals at I60 and I90: All-15

The promoted I60 proposal is followed by an identically safeguarded I90 proposal. Both use one shared-camera residual action, eight fixed scales, three rollback-safe landmark steps, the `1e-3` floor, atomic camera/landmark commit, canonical restart, and next-iteration trust rebase.

| Metric | I60+I90 | I60 only |
|---|---:|---:|
| Geometric delivered/control | 0.766084290 | 0.786532802 |
| Summed delivered/control | 0.732090008 | 0.754474107 |
| W/T/L versus control | 13/1/1 | 14/1/0 |
| Second proposal applied | 14 | 0 |

The second proposal is aggregate-strong but not a no-loss replacement. Gendarmenmarkt regresses to `1.019556957x` control from I60-only `0.957620277x`; Alamo and Trafalgar also regress relative to I60 while remaining below control. Weak immediate second gains correlate with extra rejections: Gendarmenmarkt gains only `0.2475%` at I90 and rises from 2 to 8 rejections; Alamo gains `0.2844%` and rises 7 to 9. Retain I60+I90 as a bounded-loss candidate, not the common default. Do not tune the SSE floor. The next mechanism test keeps the proven first trust rebase but suppresses the second rebase.

# Final Global-Schur Breadth Confirmation

Frozen K24/I30 base-backbone Themelis trajectory followed by at most three accepted `bsr_low_memory` global Schur corrections. The global progress stop is `1e-3`; damping starts at 3/3. No scene-specific settings are used.

| Family | Completed | Corrected/pre | Corrected/base-I30 | Corrected/quality base | Corrected/Ceres | W/T/L vs Ceres | Schur s | Max coord. RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1DSfM | 15/15 | 0.864158414 | 0.858089851 | 1.116626283 | 1.626943872 | 1/0/14 | 23.168 | 2.471 GiB |
| BAL | 29/29 | 0.999319770 | 0.996749395 | 1.012639985 | 1.011023376 | 3/0/26 | 135.527 | 12.134 GiB |

## Correction Behavior

- **1DSfM:** accepted-count distribution `{3: 15}`; termination distribution `{'maximum_corrections': 15}`; pre-Schur/Ceres `1.882691700x`; corrected/Ceres `1.626943872x`.
- **BAL:** accepted-count distribution `{1: 27, 2: 2}`; termination distribution `{'minimum_relative_decrease': 29}`; pre-Schur/Ceres `1.011711573x`; corrected/Ceres `1.011023376x`.

## Decision

The frozen polishing policy improves every matched I30 endpoint. It is a strong 1DSfM quality component: all 15 scenes accept all three corrections, with a `0.864158414x` geometric endpoint ratio to the I30 handoff. It remains `1.116626283x` the established I200 base, so three corrections do not yet replace the long-horizon quality control.

On BAL, the global `1e-3` stop avoids almost all extra correction work: 27/29 scenes stop after one correction and only BAL135/BAL142 accept a second. The endpoint is `0.999319770x` its I30 handoff but `1.012639985x` the established I90 base. Peak coordinator RSS is 12.134 GiB on BAL961. Retain this as separately labeled polishing; do not promote it as a replacement for either long-horizon base.

## Per-Scene Corrected/Ceres Ratios

### 1DSfM

- `tower_of_london`: `2.812601524x`
- `yorkminster`: `2.688640414x`
- `roman_forum`: `1.845225908x`
- `madrid_metropolis`: `1.751529218x`
- `gendarmenmarkt`: `1.731576240x`
- `nyc_library`: `1.720676330x`
- `vienna_cathedral`: `1.711253104x`
- `notre_dame`: `1.703774988x`
- `piazza_del_popolo`: `1.561917079x`
- `alamo`: `1.549144993x`
- `union_square`: `1.436315435x`
- `montreal_notre_dame`: `1.357765587x`
- `trafalgar`: `1.322690706x`
- `ellis_island`: `1.117828897x`
- `piccadilly`: `0.999896638x`

### BAL

- `bal3068`: `1.053616365x`
- `bal89`: `1.046326175x`
- `bal257`: `1.044030252x`
- `bal356`: `1.040105908x`
- `bal126`: `1.033624488x`
- `bal1778`: `1.025894581x`
- `bal308`: `1.018736238x`
- `bal394`: `1.014018386x`
- `bal287`: `1.013204598x`
- `bal871`: `1.010532605x`
- `bal1723`: `1.008326482x`
- `bal253`: `1.008204841x`
- `bal52`: `1.008204403x`
- `bal49`: `1.007372767x`
- `bal142`: `1.005712571x`
- `bal744`: `1.004744564x`
- `bal173`: `1.004581842x`
- `bal646`: `1.003497235x`
- `bal1266`: `1.003401396x`
- `bal1490`: `1.003007533x`
- `bal1064`: `1.002616425x`
- `bal88`: `1.001973165x`
- `bal783`: `1.001656856x`
- `bal135`: `1.001379963x`
- `bal931`: `1.001072115x`
- `bal961`: `1.000624890x`
- `bal427`: `0.997708710x`
- `bal951`: `0.980373711x`
- `bal245`: `0.979432098x`

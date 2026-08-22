# K24 I60 Landmark-Response Quarter-Damping Breadth

The globally frozen candidate sets Schur camera/landmark damping to `0.00146484375`/`0.00146484375`, retaining I60 timing, three landmark steps, eight scales, `1e-3` floor, atomic commit, restart, and trust rebase. Development used Madrid, Tower, and Yorkminster; the remaining 12 1DSfM scenes and all BAL29 are unchanged validation.

| Family | Completed | Candidate/base | Candidate/control | Candidate/Ceres | Summed/Ceres | W/T/L candidate/base | Selected | Rejections |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1dsfm | 15/15 | 0.984939730 | 0.774687405 | 1.153444581 | 1.065278483 | 12/1/2 | 14 | 122 |
| bal | 29/29 | 1.000000000 | 1.000000000 | 0.999038939 | 0.995079573 | 0/29/0 | 0 | 32 |

## Per-Scene Candidate/Base Ratios

- `trafalgar`: candidate/base `1.010414776x`, control `0.897895841x`, Ceres `0.951471147x`
- `gendarmenmarkt`: candidate/base `1.006779687x`, control `0.964112643x`, Ceres `1.258545288x`
- `madrid_metropolis`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `1.121253758x`
- `bal1064`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `1.001798665x`
- `bal126`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `1.016852250x`
- `bal1266`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `1.002286985x`
- `bal135`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `1.001461213x`
- `bal142`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `1.003381351x`
- `bal1490`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `0.994770842x`
- `bal1723`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `0.996496923x`
- `bal173`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `0.998715334x`
- `bal1778`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `1.012137802x`
- `bal245`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `0.964296960x`
- `bal253`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `0.996460685x`
- `bal257`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `1.025701761x`
- `bal287`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `0.998682986x`
- `bal3068`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `0.990923943x`
- `bal308`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `1.002929535x`
- `bal356`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `1.020981401x`
- `bal394`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `1.007367510x`
- `bal427`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `0.996946389x`
- `bal49`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `1.002088395x`
- `bal52`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `0.970685033x`
- `bal646`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `1.002052314x`
- `bal744`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `0.997314467x`
- `bal783`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `1.000692664x`
- `bal871`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `1.000614884x`
- `bal88`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `1.000093968x`
- `bal89`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `1.006441419x`
- `bal931`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `1.000308565x`
- `bal951`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `0.962279887x`
- `bal961`: candidate/base `1.000000000x`, control `1.000000000x`, Ceres `1.000073324x`
- `alamo`: candidate/base `0.998085899x`, control `0.896539559x`, Ceres `1.077226188x`
- `ellis_island`: candidate/base `0.995743353x`, control `0.958844697x`, Ceres `1.054268188x`
- `piccadilly`: candidate/base `0.994293111x`, control `0.896571507x`, Ceres `0.721490195x`
- `union_square`: candidate/base `0.987474401x`, control `0.767811508x`, Ceres `1.150204527x`
- `montreal_notre_dame`: candidate/base `0.986461619x`, control `0.565599138x`, Ceres `1.123917632x`
- `notre_dame`: candidate/base `0.980050799x`, control `0.610828845x`, Ceres `1.085769859x`
- `vienna_cathedral`: candidate/base `0.979945502x`, control `0.602321747x`, Ceres `1.095304077x`
- `nyc_library`: candidate/base `0.979638894x`, control `0.861087956x`, Ceres `1.116129599x`
- `tower_of_london`: candidate/base `0.978117436x`, control `0.873031962x`, Ceres `1.790270661x`
- `yorkminster`: candidate/base `0.972726729x`, control `0.792651817x`, Ceres `2.219245186x`
- `roman_forum`: candidate/base `0.953688151x`, control `0.622680590x`, Ceres `1.131347808x`
- `piazza_del_popolo`: candidate/base `0.952730646x`, control `0.549166642x`, Ceres `1.002945613x`

Gate status: **passed**.

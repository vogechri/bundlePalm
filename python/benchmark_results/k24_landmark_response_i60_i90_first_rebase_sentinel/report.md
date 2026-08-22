# K24 I60+I90 Proposals with First Trust Rebase Only

The two-proposal policy keeps the promoted I61 worker trust rebase after the first proposal but suppresses the second rebase after I90. Canonical product-state restart and acceleration reset still occur after both selected proposals. The expanded sentinel adds Gendarmenmarkt to Roman, Trafalgar, BAL52, and BAL3068.

| Scene | First-rebase-only/control | Both-rebase/control | I60-only/control | Rebase applied | Rebase suppressed | Rejections first/both |
|---|---:|---:|---:|---:|---|---:|
| Gendarmenmarkt | 1.016088753 | 1.019556957 | 0.957620277 | I61 | I90 | 9/8 |
| Roman Forum | 0.634346777 | 0.628975086 | 0.652918451 | I61 | I90 | 10/9 |
| Trafalgar | 0.891374692 | 0.890794840 | 0.888640846 | I61 | I90 | 16/20 |
| BAL52 | 1.000000000 | 1.000000000 | 1.000000000 | none | none | 0/0 |
| BAL3068 | 1.000000000 | 1.000000000 | 1.000000000 | none | none | 3/3 |

Suppressing the second trust rebase does not repair Gendarmenmarkt and slightly worsens the three-scene 1DSfM geomean (`0.831328930x` versus `0.829737801x` with both rebases and `0.822102904x` with I60 only). Therefore the repeated proposal, not the second trust rebase, causes the bad tail. Close proposal-count and checkpoint tuning; retain I60 only as the common default.

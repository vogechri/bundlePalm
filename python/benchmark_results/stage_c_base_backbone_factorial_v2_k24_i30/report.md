# Base-Backbone C1/C5 Sentinel Factorial

All variants use one global K24/I30 backbone: local Nesterov, persistent DRS trust, curvature 0.4 recovery/decay, camera metric 75, no proposal damping, and shared-only camera proximal terms. Only C1 and delayed C5 vary.

## 1DSfM

| Variant | SSE/hybrid plain | W/L | Worst | SSE/base I30 | Optimization s | Time/plain | Completion |
|---|---:|---:|---:|---:|---:|---:|---|
| Base backbone | 1.000000 | 0/0 | 1.000000 | 1.328439 | 14.136 | 1.000 | roman_forum:30, trafalgar:30 |
| Base backbone + C1 | 0.694751 | 2/0 | 0.781348 | 0.922935 | 24.369 | 1.724 | roman_forum:30, trafalgar:30 |
| Base backbone + C5 | 1.004008 | 0/1 | 1.008032 | 1.333764 | 15.748 | 1.114 | roman_forum:30, trafalgar:30 |
| Base backbone + C1+C5 | 0.688636 | 2/0 | 0.781348 | 0.914812 | 28.145 | 1.991 | roman_forum:30, trafalgar:30 |

## BAL

| Variant | SSE/hybrid plain | W/L | Worst | SSE/base I30 | Optimization s | Time/plain | Completion |
|---|---:|---:|---:|---:|---:|---:|---|
| Base backbone | 1.000000 | 0/0 | 1.000000 | 1.220428 | 13.046 | 1.000 | bal52:30, bal3068:27* |
| Base backbone + C1 | 0.795680 | 2/0 | 0.840752 | 0.971070 | 24.923 | 1.910 | bal52:30, bal3068:30 |
| Base backbone + C5 | 1.000000 | 0/0 | 1.000000 | 1.220428 | 14.706 | 1.127 | bal52:30, bal3068:27* |
| Base backbone + C1+C5 | 0.795396 | 2/0 | 0.840153 | 0.970723 | 28.656 | 2.197 | bal52:30, bal3068:30 |

`*` marks recovery exhaustion before I30.

## Gate

Promote a hybrid only if one unchanged C1/C5 combination improves the base-backbone plain arm on both families without a material worst-scene regression. No scene routing is permitted.

# Low-Memory BSR Global-Schur Gate

Date: 2026-08-12

## Scope

This is the matched four-case operator gate for the K24/I30 base-backbone
Themelis trajectory followed by up to three final global Schur corrections.
Only the global Schur operator changes from `python` to `bsr_low_memory`; the
progress stop remains `1e-4` for direct comparison.

| Case | Accepted Python/BSR | Termination | BSR/Python endpoint | Python s | BSR s | Python coord. RSS | BSR coord. RSS |
|---|---:|---|---:|---:|---:|---:|---:|
| Roman | 3/3 | maximum corrections | 1.001898293 | 1.671 | 1.205 | 0.951 GiB | 0.948 GiB |
| Trafalgar | 3/3 | maximum corrections | 1.000428021 | 10.675 | 6.078 | 2.274 GiB | 2.477 GiB |
| BAL52 | 2/2 | minimum relative decrease | 1.000000000 | 0.446 | 0.360 | 0.748 GiB | 0.749 GiB |
| BAL3068 | 2/2 | minimum relative decrease | 1.000000014 | 87.901 | 23.181 | 6.963 GiB | 7.752 GiB |

Every accepted solve converges, the I30 prefixes are identical, and all
acceptance/termination decisions match Python. BSR endpoint drift appears after
repeated corrections: `+0.190%` on Roman and `+0.043%` on Trafalgar. The
largest speed gain is BAL3068 (`0.264x` Python correction time), but
`bsr_low_memory` is low-memory relative to ordinary BSR construction, not the
matrix-free Python operator; its BAL3068 peak RSS is higher.

## Decision

Use `bsr_low_memory` as the frozen breadth operator because it preserves the
decision sequence and materially reduces large-case correction time. Keep
Python as the quality-reference operator. Freeze the predeclared `1e-3`
relative-progress stop for breadth: the Python traces show it preserves all
three Roman/Trafalgar corrections while stopping BAL52/BAL3068 after their
first small accepted gain.

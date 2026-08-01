# Evidence — multi-accumulator reductions for the sparse solver family

Measured 2026-07-31/08-01 by StormySquirrel on `thinkstation1` (64 logical CPUs).
Pre-registration `PREREGISTRATION.md` was committed **alone, in `b14267eda`,
before the implementation existed** — repository history establishes precedence
for this lever (unlike the preceding MINRES lever, where plan and result landed
together).

**Evidence classification:** the headline numbers are **same-binary,
same-session, interleaved A/B on our own arm**, which is the strongest form
available on a non-quiescent shared host because it never involves the SciPy
arm. Cross-arm ratios against live SciPy are reported but marked provisional —
the SciPy arm drifted materially between sessions and is not load-controlled.

- Binaries: `perf_minres_vs_scipy` BEFORE `8193f84699c7f950…` / AFTER
  `8ce356e89f5230b1…`; `perf_csr_matvec` BEFORE `e714a9f86965943e…` / AFTER
  `d05b9e43b055bbce…`. BEFORE builds come from committed `HEAD` via
  `rch --clean-overlay --no-overlay`, AFTER from the same base plus only
  `linalg.rs`.
- Builder: worker `ovh-a`/`hz1`, `rustflags = -C target-feature=+avx2,+fma`.

## How the lever was found

Whole-job `perf` profile of the MINRES solve at `n = 16,384`, self-time ranked,
each entry tested against *does the incumbent pay this same cost?*

| symbol | self | incumbent pays it? | verdict |
|---|---|---|---|
| `csr_matvec_into_impl` | 30.2% | **Yes** — `scipy.sparse` runs the identical CSR matvec in C | not our gap |
| `fsci_sparse::linalg::minres` | 12.6% | **No, not at this rate** | **the lever** |
| `csr_matvec_thunk` | 5.1% | No, but dispatch glue on a serial path | too small |

`perf annotate` located the cost precisely: `dot_product` (`0x31270`) and
`vec_norm` (`0x31480`) emitted **scalar** `vmulsd`/`vaddsd` with every add
targeting **one** register — a serial dependency chain. The adjacent axpy sweep
was `vmulpd` (packed), so this was not a general vectorisation failure: a
reduction is the one shape rustc cannot rescue without fast-math, because f64
addition is not associative. SciPy's `inner`/`norm` reach OpenBLAS `ddot`, and
`ddot_kernel_8` (eight accumulators, packed AVX) appears in the same profile.

## P1 — the chain actually broke

Disassembly of `fsci_sparse::linalg::minres`, same symbol, both binaries:

| | `vaddpd` | `vmulpd` | packed total | add-destination registers |
|---|---|---|---|---|
| BEFORE | 1 | 16 | 27 | `%xmm0`:20, `%xmm1`:6, rest 1 each |
| AFTER | **17** | **36** | **63** | `%xmm2`:21, `%xmm0`:13, `%xmm1`:13, `%xmm3`:9 |

One dominant accumulator became four balanced ones. **CONFIRMED.**

## P2 — whole-solve speedup (our arm, interleaved, same session)

MINRES, indefinite `L − 3.7I`, `n = 16,384`, `rtol = 1e-8`:

| pass | BEFORE ms | AFTER ms |
|---|---|---|
| 1 | 2030.913 (cv 0.170) | 1699.423 (cv 0.014) |
| 2 | 2142.816 (cv 0.058) | 1701.086 (cv 0.020) |
| median | **2086.86** | **1700.25** |

**1.227×**, predicted ≥1.10× — **CONFIRMED.** The variance collapse (cv 0.170 →
0.014) is itself evidence: a latency-bound serial chain is far more sensitive to
co-tenant interference than four independent ones.

Per A-application across the sweep:

| n | BEFORE µs | AFTER µs | gain |
|---|---|---|---|
| 1,024 | 7.552 | 5.071 | **1.489×** |
| 4,096 | 25.780 | 20.510 | **1.257×** |
| 16,384 | 105.552 | 86.141 | **1.225×** |

The gain is largest at small `n`, exactly as a latency-bound mechanism predicts:
at `n = 1,024` the vectors are L1-resident and the add-latency chain is the whole
cost, while at `n = 16,384` memory traffic dilutes it.

## P3 — crossover: PARTIALLY CONFIRMED, and confounded

Our per-unknown coefficient `b_ours` fell **0.006412 → 0.005296 = 17.4% less
work per unknown**. That number is our-arm-only and is solid.

The claim as written — "`b_ours/b_scipy` falls below 1.0, eliminating the
crossover" — **cannot be cleanly decided from these runs.** `b_ours/b_scipy`
measured 0.958 (BEFORE) → 0.830 (AFTER) in this session, so it is below 1.0 on
both sides. But the *same* BEFORE code measured a `b` ratio of **1.042** in the
previous session. The difference is the SciPy arm, not ours: its `n = 16,384`
cell ran 2077 ms then and 2273 ms now. So the crossover's existence is inside
SciPy-arm session variance, and I am not entitled to claim this lever removed
it. What the lever demonstrably did is cut our own per-unknown cost by 17.4%.

**This is the second time the SciPy arm's drift has undermined a cross-arm
claim on this host.** Our-arm-only A/B should be the default framing for any
future lever here, with cross-arm ratios reported as context.

## P4 — accuracy did not degrade

| | iterations | true relative residual |
|---|---|---|
| BEFORE | 19,771 | 9.970475e-9 |
| AFTER | **19,738** | 9.987243e-9 |

Fewer iterations, same residual band, both inside the 1e-8 target.
**CONFIRMED** — and directionally consistent with `O((n/k)·ε)` error growth
beating serial summation's `O(n·ε)`.

## P5 — conformance held, and improved

`diff_sparse_iterative_solvers` against live SciPy 1.17.1: **12/12 pass**,
max abs diff **5.295e-10 → 5.129e-10** (better).

As pre-registered, exact bit-agreement was expected to be spent, and the bill
came to **at most one ULP**: `minres_5x5_diag_spd` moved 0.0 → 1.110e-16, while
`minres_4x4_tridiag_spd` and `minres_6x6_pentadiag_spd` stayed at exactly 0.0.
CG's agreement with SciPy on the side=120 fixture was unchanged at
`max_abs_diff = 6.594e-12`, `relative_l2_diff = 7.862e-15`.

## P6 — the win generalises across the family

CG, `perf_csr_matvec cg-vs-scipy`, side=120 (`n = 14,400`, `nnz = 71,520`,
serial path), three interleaved passes, our arm only:

| pass | BEFORE ms | AFTER ms |
|---|---|---|
| 1 | 22.366 | 18.661 |
| 2 | 22.615 | 18.729 |
| 3 | 22.812 | 18.826 |
| median | **22.615** | **18.729** |

**1.207×**, with no overlap between the two groups. **CONFIRMED**, and closely
matching MINRES's 1.225× at comparable `n` — same mechanism, same magnitude.
Iterations (170) and residual (9.159e-6) were identical across arms.

### A correction to the pre-registration's own premise

The plan claimed the three helpers had "69 call sites across 9 solvers", so one
edit would cover the family. **That was wrong**: `cg` and `pcg` did not call the
helpers at all — they carried their own inline `.map(…).sum()` reductions.
Measured as first written, P6 would have been falsified for the reason that the
lever never reached CG. The change was widened to route `cg` and `pcg` through
the shared helpers (10 inline reductions), which is what the CG number above
measures.

**Still not covered, deliberately:** `cg_persistent_workers` (the `nnz ≥ 2^18`
path) fuses its reductions *into* the matvec and axpy loops, where the adds
interleave with other work and are partly latency-hidden. That is a different
shape, and an earlier campaign already refuted a fused-sweep rewrite there
(`perf_krylov_per_iteration_spawn_wall`). Large-`n` CG is therefore untouched by
this lever. `csr_matvec_into_impl` is also untouched — it carries a
byte-identical parallel contract.

## Scorecard

| # | Claim | Result |
|---|---|---|
| P1 | serial scalar chain broken | **CONFIRMED** — `vaddpd` 1→17, four balanced accumulators |
| P2 | ≥1.10× whole-solve at n=16,384 | **CONFIRMED** — 1.227× |
| P3 | `b` ratio below 1.0, crossover eliminated | **PARTIAL / CONFOUNDED** — `b_ours` −17.4% is solid; the crossover claim sits inside SciPy-arm session drift |
| P4 | accuracy not degraded | **CONFIRMED** — 33 fewer iterations, same residual band |
| P5 | conformance within `ABS_TOL = 1e-6` | **CONFIRMED** — 12/12, max diff improved, ≤1 ULP spent |
| P6 | win generalises to CG | **CONFIRMED at 1.207×**, but only after correcting a false premise in the plan (see above) |

## Verification

`cargo test -p fsci-sparse`: **407 passed, 0 failed, 4 ignored.**
`diff_sparse_iterative_solvers` vs live SciPy: 12/12 pass.

## Decision

**KEEP.** 1.227× on MINRES and 1.207× on CG, our-arm interleaved, from breaking
a serial floating-point dependency chain the compiler could not break itself;
conformance held with at most one ULP spent and accuracy slightly improved.

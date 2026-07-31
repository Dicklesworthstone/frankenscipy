# Pre-registration — multi-accumulator reductions for the sparse solver family

**Committed alone, before the implementation exists and before any timing on
this lever exists.** (The previous lever's plan landed in the same commit as its
result, so repository history could not establish precedence. This one is
separated deliberately so it can.)

Registered 2026-07-31 by StormySquirrel. Host `thinkstation1`, 64 logical CPUs,
live SciPy 1.17.1 in the same invocation.

## How this lever was found

Whole-job profile of the MINRES solve at `n = 16,384` (`perf record -F 999`,
artifact `raw/minres.perf.data`), self-time ranked, then each entry tested
against the only question that matters — *does the incumbent pay this same cost?*

| rank | symbol | self | does SciPy pay it? | verdict |
|---|---|---|---|---|
| 1 | `csr_matvec_into_impl` | 30.2% | **Yes** — `scipy.sparse` does the identical CSR matvec in C | not our gap, moved on |
| 2 | `fsci_sparse::linalg::minres` | 12.6% | Partly — see below | **structural difference** |
| 3 | `csr_matvec_thunk` | 5.1% | No, but it is dispatch glue on a serial path | too small to chase first |

`perf annotate` on entry 2 shows where its time actually goes. Two loops:

```
0x31270:  vmovsd / vmulsd / vaddsd %xmm1,%xmm0,%xmm0     <- dot_product(&v,&y)
0x31480:  vmovsd / vmulsd %xmm2,%xmm2 / vaddsd %xmm1     <- vec_norm(&r2)
```

Both are **scalar** (`sd`, not `pd`) and every add accumulates into **one**
register — a single serial dependency chain at ~4 cycles per element. By
contrast the axpy sweep at `0x30fbb` is `vmulpd` (packed): rustc vectorised it
happily, because it carries no reduction dependency. The reductions cannot be
vectorised or split by the compiler, because f64 addition is not associative and
this crate is built without fast-math.

**SciPy does not pay this.** Its `inner(v,y)` and `norm(r2)` route to OpenBLAS —
`ddot_kernel_8` appears in the same profile, using eight independent
accumulators with packed AVX. Same mathematics, structurally different
implementation. This is the direct cause of the measured `b_ours/b_scipy = 1.042`
in `2026-07-31-minres-lanczos/EVIDENCE.md`: our **per-unknown** work loses to
SciPy's despite being compiled, and that loss is what creates a crossover `n*`
beyond which we lose outright.

## The change

`dot_product`, `vec_norm`, and `vec_norm_diff` in `crates/fsci-sparse/src/linalg.rs`
accumulate into `k = 4` independent chains, summed at the end. These three
helpers have **69 call sites across 9 solvers** (`cg`, `gmres`, `lgmres`,
`bicg`, `cgs`, `bicgstab`, `qmr`, `minres`, `lsqr`), so one edit covers the
family.

**Explicitly out of scope:** `csr_matvec_into_impl` accumulates each row's dot
product inline and carries a documented byte-identical parallel contract. It is
not touched.

## This is NOT bit-identical, and that is the point

Reassociating a float sum changes the result in the last bits. Two consequences,
one of which cuts against the lever and one for it:

- **Against:** the minres differential cases currently agree with SciPy at
  *exactly* 0.0 on the three small SPD systems. That may become nonzero.
- **For:** k-way accumulation is *more* accurate, not less. Serial summation has
  error growth `O(nε)`; k independent chains give `O((n/k)ε)`. NumPy uses
  pairwise summation for exactly this reason. At `n = 16,384` our serial chain is
  the less accurate of the two.

## Predictions

| # | Claim | Falsified if |
|---|---|---|
| **P1** | The emitted loops stop being a single scalar chain — disassembly shows either packed adds or ≥2 distinct accumulator registers where `%xmm0`/`%xmm1` stood alone. | still one scalar accumulator |
| **P2** | Whole-solve MINRES at `n = 16,384` improves by **≥1.10×**. | < 1.10× |
| **P3** | `b_ours / b_scipy` falls **below 1.0**, eliminating the crossover: we win at every `n` rather than decaying to parity. | `b` ratio stays ≥ 1.0 |
| **P4** | Accuracy does **not** degrade: true relative residual at convergence is ≤ the serial-sum arm's, and iteration count does not increase. | residual worse, or more iterations |
| **P5** | Differential conformance stays within `ABS_TOL = 1e-6` on all 12 cases. | any case exceeds it |
| **P6** | The win generalises: CG on its SPD fixture also improves measurably (same helpers). | CG unchanged (would mean I mis-attributed the cost) |

## Known ways this could disappoint

- If the reductions are memory-bandwidth-bound rather than latency-bound, extra
  accumulators buy nothing — the loads, not the adds, would be the wall. At
  `n = 16,384` a vector is 128 KB and two of them fit in L2, so I expect
  latency-bound, but that is the main way P2 fails.
- `k = 4` is chosen to match four AVX lanes without spilling; if the win is real
  but smaller than predicted, `k = 8` is the obvious follow-up rather than
  evidence against the mechanism.
- The `2.87×`/`3.27×` per-A-application advantages already measured over the
  GMRES delegate came partly from these same helpers being called fewer times.
  This lever must be measured against the **current** MINRES, not the delegate.

## Reporting rule

Scorecard reported whichever way it falls, including P3 and P4. Reverted if P2
fails, regardless of P1.

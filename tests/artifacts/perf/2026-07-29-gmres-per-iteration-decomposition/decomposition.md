# GMRES incumbent ratio — per-iteration decomposition

Date: 2026-07-29
Bead: `frankenscipy-ddvmb` (follow-up to `frankenscipy-felow`)
Question: with inner-iteration counts already matched exactly, what is the
FrankenSciPy/SciPy GMRES incumbent ratio actually made of?

## What this is and is not

This is a **derivation over already-committed measurements**, not a new
benchmark run. It consumes exactly one prior artifact:

- Source artifact:
  `tests/artifacts/perf/2026-07-29-sparse-nonsymmetric-vs-scipy-live-arm/gmres_restart20_bench_stdout_stderr.txt`
- Source artifact SHA-256:
  `6703ccd1a80a6d9fb05d5a6cb472da67832bc1c060514ede191c9eed66945585`
- Measured on `host_identity=threadripperje` — **not** this host. Analysis host
  identity is recorded separately below.
- Executed-binary ELF SHA-256 of the source run:
  `12003f00e83c8074ef249cbab59659641c43bf9845da487a72d94c0922a29ab5`
- SciPy engine SHA-256 of the source run:
  `f9d7ace03295000d7b1a76dd12229208908a59140b741669e961b69733110e8f`

No new wall-clock number is claimed here. Every millisecond below is quoted
from the source artifact; only the per-iteration division and the slope
arithmetic are new.

Analysis host (where the arithmetic ran, not where timing was taken):
`host_identity=thinkstation1`, AMD Ryzen Threadripper PRO 5975WX 32-Cores,
`physical_cores=32`, `logical_threads=64`, RAM 215 GiB. The analysis is pure
arithmetic and is host-independent; the host is recorded for completeness.

## Why per-iteration division is legitimate here

The three source cells have **exactly matched inner-iteration counts** between
arms (127/127, 163/163, 227/227) after the restart-20 default-parity lever in
`frankenscipy-felow`. Nothing is averaged across cells: each cell's wall time
is divided by *that cell's own* count, and the cells are then compared as
separate points. No ratio is averaged over matrices with different counts.

Restarted GMRES does not have a constant per-iteration cost — iteration `j`
inside a restart cycle orthogonalizes against `j` prior Krylov vectors — so
per-iteration division is only comparable if the cells have similar restart
cycle structure. They do. With restart 20 in all three cells:

| n | iterations | full cycles | tail | mean prior vectors per iteration |
|---:|---:|---:|---:|---:|
| 1,024 | 127 | 6 | 7 | 9.14 |
| 4,096 | 163 | 8 | 3 | 9.34 |
| 9,216 | 227 | 11 | 7 | 9.30 |

Mean orthogonalization depth per iteration is 9.14–9.34 across all three
cells, a spread of 2.2%. Per-iteration cost is therefore comparable across
cells and the Arnoldi work per iteration is ~`O(n)` with a near-identical
constant in each cell.

## The decomposition

| side / n | iterations ours / SciPy | ours ms | SciPy ms | **µs per iteration ours** | **µs per iteration SciPy** | incumbent ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 32 / 1,024 | 127 / 127 | 2.364355 | 11.552343 | **18.617** | **90.963** | 4.8850x win |
| 64 / 4,096 | 163 / 163 | 13.900668 | 20.786496 | **85.280** | **127.525** | 1.5725x win |
| 96 / 9,216 | 227 / 227 | 38.326807 | 36.057110 | **168.841** | **158.842** | 0.9397x loss |

### Fit-free statement of the mechanism

Over a 9.0x increase in `n` (1,024 to 9,216):

- FrankenSciPy per-iteration cost rises **9.069x** — almost exactly
  proportional to `n`. Our per-iteration cost is essentially all work, with
  negligible fixed overhead.
- SciPy per-iteration cost rises only **1.746x**. Its per-iteration cost is
  dominated by an `n`-independent fixed term.

Segment marginal cost, computed as a finite difference between adjacent cells
(no regression, no degrees-of-freedom argument):

| segment | ours µs/unknown | SciPy µs/unknown | ours / SciPy |
|---|---:|---:|---:|
| n = 1,024 -> 4,096 | 0.021700 | 0.011901 | **1.823x** |
| n = 4,096 -> 9,216 | 0.016320 | 0.006117 | **2.668x** |

Our marginal cost per unknown is worse than SciPy's in **both** segments, by
1.8x and 2.7x. This does not depend on any fitted model.

### Quantification by least squares (3 points, 1 degree of freedom)

Modelling per-iteration cost as `a + b*n`:

| arm | `a` (fixed µs/iteration) | `b` (µs/unknown) | R² |
|---|---:|---:|---:|
| FrankenSciPy | 4.266 | 0.018132 | 0.9939 |
| SciPy 1.17.1 | 87.239 | 0.008065 | 0.9651 |

- Fixed per-iteration overhead, SciPy minus ours: **82.973 µs/iteration**
- Marginal per-unknown cost, ours / SciPy: **2.248x**
- Predicted crossover (equal per-iteration cost): **n ≈ 8,242**, side ≈ 91

### Closure check

The two-term model is not merely fitted, it reconstructs each cell's measured
wall-time gap. Predicted gap is
`(a_scipy - a_ours) * iterations + (b_scipy - b_ours) * n * iterations`:

| n | iterations | fixed term (ms) | marginal term (ms) | predicted gap | measured gap | error |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 127 | +10.538 | -1.309 | +9.228 | +9.188 | +0.040 |
| 4,096 | 163 | +13.525 | -6.721 | +6.803 | +6.886 | -0.082 |
| 9,216 | 227 | +18.835 | -21.060 | -2.226 | -2.270 | +0.044 |

Positive means SciPy is slower. Residuals are 0.040–0.082 ms against gaps of
2.270–9.188 ms, i.e. 0.4%–3.6% of gap magnitude. The sign flip at side 96 is
reproduced: the fixed-overhead credit keeps growing (+18.835 ms) but our
marginal-cost debit grows faster (-21.060 ms) and overtakes it.

This fit has 3 points and 1 degree of freedom. It is reported as a
quantification of the fit-free statement above, not as independent evidence,
and no confidence interval is claimed for `a` or `b`. The crossover prediction
`side ≈ 91` is consistent with the measured side-96 loss, which is the only
out-of-sample check available at three points.

## Conclusion

**The GMRES win is overhead amortization, not per-iteration kernel
superiority.** With iteration counts exactly matched, SciPy pays roughly
87 µs per Arnoldi iteration that does not depend on problem size — its
per-iteration Givens/orthogonalization bookkeeping runs in interpreted Python
around NumPy calls. FrankenSciPy pays roughly 4 µs. At `n = 1,024` and 127
iterations that fixed gap is worth +10.538 ms — more than the entire 9.188 ms
measured wall-time difference, because our own worse marginal cost gives
1.309 ms of it back. The side-32 4.8850x win is fully accounted for by SciPy's
per-iteration interpreter tax.

Against that, FrankenSciPy's *own* per-unknown marginal cost is **1.8–2.7x
worse than SciPy's**. SciPy's inner kernel per unknown is faster than ours.
The advantage therefore decays monotonically with `n` and inverts near
`n ≈ 8,000–9,200`, which is exactly the measured side-96 loss.

### What this licenses and forbids

Licensed claim: on this fixture class, FrankenSciPy GMRES wins at small-to-
moderate `n` because it does not pay a Python per-iteration tax, at exactly
matched Krylov trajectories.

**Forbidden claims:**
- "FrankenSciPy's GMRES kernel is faster than SciPy's." It is not; per unknown
  it is 1.8–2.7x slower.
- Any size-general GMRES win. Already rejected in the ledger; this explains
  why it was always going to be rejected.
- Any extrapolation of the 4.8850x to larger `n`, other sparsity patterns, or
  other iteration counts.

### Matrix class characterized

Nonsymmetric, strictly diagonally dominant 2-D convection-diffusion CSR:
`diagonal=4.001`, `west=-1.2`, `east=-0.8`, `vertical=-1`, 5-point stencil,
`nnz = 5n - 4*side`, `rhs = 1 + 0.01*(i mod 17)`, `x0 = zeros`, `rtol=1e-5`,
`atol=0`, restart 20 on both arms. The win regime is `n` below roughly 8,000
on this class at one thread. Nothing here transfers to a different stencil,
a preconditioned solve, or a different restart length without remeasurement.

### Consequence for the side-96 lever (bead `frankenscipy-ddvmb`)

That bead requires a restart-20 profile attributing at least 8% of self-time
to one removable first-party leaf. This decomposition says where to point the
profiler: the target is the **`O(n)` per-iteration inner kernel** — SpMV plus
the ~9.3 Gram-Schmidt dot/axpy pairs per iteration — not per-iteration
bookkeeping, which we already win by ~83 µs/iteration. A bookkeeping or
allocation lever cannot close a marginal-cost gap of 1.8–2.7x per unknown.

## Sibling audit: which other iterative cells are still count-unmatched

Running `decompose.py` over the pre-parity artifact
(`bench_stdout_stderr.txt`, same directory as the source) rejects every cell
and refuses to fit, which is the intended behaviour:

| method | side / n | iterations ours / SciPy | matched | incumbent ratio | direction of contamination |
|---|---:|---:|---|---:|---|
| gmres | 32 / 1,024 | 125 / 127 | NO | 3.9679x | ours did **fewer** — win inflated |
| gmres | 64 / 4,096 | 244 / 163 | NO | 0.9456x | ours did more — loss overstated |
| gmres | 96 / 9,216 | 239 / 227 | NO | 0.8899x | ours did more — loss overstated |
| bicgstab | 32 / 1,024 | 45 / 44 | NO | 3.1908x | ours did **more** — win understated |
| bicgstab | 64 / 4,096 | 89 / 88 | NO | 1.4638x | ours did **more** — win understated |

The three GMRES rows are superseded by the restart-20 parity run and must not
be quoted; the ledger row for the 3.9679x cell is annotated accordingly.

The two **BiCGSTAB** rows are still live KEEP claims and are still
count-unmatched, by one iteration each. The mismatch runs *against* us — ours
performs 45 and 89 iterations versus SciPy's 44 and 88 — so those wins
(3.1908x, 1.4638x) are **conservative**: matching the counts could only move
them up. They are not inflated by a convergence advantage, which is why they
are left standing rather than superseded. The one-iteration gap most likely
reflects a stopping-test evaluation-point difference rather than a different
trajectory, and characterizing it is not attempted here.

## Reproduction

`decompose.py` in this directory regenerates every derived number from the
source artifact:

```
python3 decompose.py \
  ../2026-07-29-sparse-nonsymmetric-vs-scipy-live-arm/gmres_restart20_bench_stdout_stderr.txt
```

It refuses to include any cell whose two arms report different iteration
counts, printing `EXCLUDED from fit` instead, so an unmatched cell can never
silently enter the model.

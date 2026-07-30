# Result 1: SciPy's lsqr per-iteration cost — prediction P2 FALSIFIED

Date: 2026-07-29. Agent `cc/NobleCedar`. Pre-registration: `8ef32a3c9`
(`PREREGISTERED_mechanism.md`, committed before any measurement).

**This is a single-arm SciPy probe, not a gated two-arm incumbent ratio.** It
carries no A/A null controls, no bootstrap-median CI gate, and no host-wide
quiescence certificate. It is labelled provisional for that reason and it is
**not** eligible to be quoted as an incumbent ratio. What it does establish is a
property of the *incumbent alone*, which needs no second arm.

## Provenance

- Host: `thinkstation1`, AMD Ryzen Threadripper PRO 5975WX, 32 physical cores /
  64 logical threads, AVX2 + FMA, **no AVX-512**, 215 GiB RAM.
- Pinned to a single CPU with `taskset -c 63`.
- SciPy 1.17.1, NumPy 2.4.3, CPython 3.13.12.
- SciPy engine (`_isolve/iterative.py`) SHA-256
  `f9d7ace03295000d7b1a76dd12229208908a59140b741669e961b69733110e8f` — byte
  identical to the engine used in the GMRES campaign on `threadripperje`, so the
  incumbent code is the same and only the CPU differs.
- `spla.lsqr(A, b, damp=0, atol=0, btol=1e-5, conlim=0, iter_lim=10n)`.
- 9 repetitions per size, median reported. Fixture identical to the GMRES
  campaign: nonsymmetric strictly diagonally dominant 2-D convection-diffusion
  CSR, `diagonal=4.001`, `west=-1.2`, `east=-0.8`, `vertical=-1`,
  `rhs = 1 + 0.01*(i mod 17)`.

## Stopping-rule match (established before timing)

FrankenSciPy's lsqr stops on `|phi_bar| / ||b|| < tol`. SciPy's lsqr test1 is
`rnorm / bnorm` where `rnorm` **is** `phibar`, compared against
`btol + atol*anorm*xnorm/bnorm`. Setting `atol=0` and `conlim=0` reduces SciPy's
rule to `phibar/bnorm <= btol`, i.e. the same quantity against the same
threshold. Every size returned `istop=1` ("Ax - b is small enough") with a true
relative residual just under `1e-5`, confirming the intended test fired:

| side | n | nnz | istop | itn | true rel. residual |
|---:|---:|---:|---:|---:|---:|
| 16 | 256 | 1,216 | 1 | 177 | 9.8912e-06 |
| 32 | 1,024 | 4,992 | 1 | 506 | 9.9926e-06 |
| 48 | 2,304 | 11,328 | 1 | 874 | 9.9013e-06 |
| 64 | 4,096 | 20,224 | 1 | 1,355 | 9.8599e-06 |
| 96 | 9,216 | 45,696 | 1 | 2,325 | 9.9657e-06 |

lsqr needs roughly 10x the iterations GMRES does on the same matrices (GMRES
restart-20 counted 63 / 127 / 142 / 163 / 227 at these sizes), which is expected
for a normal-equations method and which *amplifies* any per-iteration effect.

## Measurement

| side | n | itn | SciPy p50 | **µs per iteration** |
|---:|---:|---:|---:|---:|
| 16 | 256 | 177 | 5.067 ms | **28.626** |
| 32 | 1,024 | 506 | 18.567 ms | **36.693** |
| 48 | 2,304 | 874 | 46.447 ms | **53.143** |
| 64 | 4,096 | 1,355 | 96.566 ms | **71.267** |
| 96 | 9,216 | 2,325 | 276.477 ms | **118.915** |

Least squares on `a + b*n` over these five points:

**`a_scipy = 27.817 µs/iteration`, `b_scipy = 0.010035 µs/unknown`, R² = 0.9966.**

## P2 is falsified

P2 predicted `a_scipy(lsqr) >= 87.2 µs`, the GMRES restart-20 value. Measured
**27.8 µs** — lower by a factor of **3.14x**. The prediction was wrong and the
reasoning behind it was wrong: I assumed lsqr's per-iteration Python bookkeeping
was at least as heavy as restarted GMRES's. It is not.

**Why, and this makes the mechanism sharper rather than weaker.** The fixed term
is not a constant of SciPy; it tracks the number of *Python-level NumPy calls per
iteration*, and the two methods differ structurally in that count:

- Restarted GMRES's per-iteration body contains an inner Python `for` loop over
  the `j` previously stored Krylov vectors (modified Gram–Schmidt). At restart 20
  the mean depth is 9.3, and each step is a separate `dot` plus a separate
  `axpy` — about 18.6 NumPy calls — before the SpMV, the norm, and the Givens
  rotation are counted.
- lsqr's per-iteration body is a fixed straight-line sequence with **no inner
  loop over a growing basis**: two matrix products (`A @ v`, `A.T @ u`) plus
  roughly seven vector operations, about 9–10 NumPy calls total.

Writing `a_scipy = W * c_py` for `W` calls per iteration at cost `c_py` each,
the two measurements are consistent with a **single** per-call cost:
`27.817 / 9.3 = 2.99 µs` and `87.239 / 29.2 = 2.99 µs`. One `c_py ≈ 3.0 µs`
explains both methods once `W` is counted correctly — GMRES `W ≈ 29`, lsqr
`W ≈ 9.3`.

So the corrected mechanism is: **SciPy's fixed per-iteration cost is
proportional to the method's per-iteration NumPy call count, at about 3 µs per
call.** GMRES is not special because it is GMRES; it is expensive because
modified Gram–Schmidt puts a Python loop inside the iteration. That is a
stronger claim than the one I pre-registered, and unlike P2 it predicts a third
method's fixed term from a static call count.

## Consequence for P3 and P4 (both still pending our arm)

`n* = (a_scipy - a_ours) / (b_ours - b_scipy)`. With `a_scipy` measured at 27.8
rather than 87.2, the numerator shrinks about 3.5x versus GMRES while
`b_scipy = 0.010035` is slightly *larger* than GMRES's 0.008065. If our arm
resembles its GMRES behaviour (`a_ours ≈ 4.3`, `b_ours ≈ 0.0181`), then
`n* ≈ 23.5 / 0.008 ≈ 2,900`, i.e. side ≈ 54.

- **P3** predicted the crossover in the same order as GMRES's 8,242, i.e. order
  `10^4`. The corrected model points at order `10^3`. **P3 is on track to be
  falsified**, and the pre-registered claim that "`W` cancels in `n*`" was wrong:
  `W` cancels only if `b` scales with `W` identically in both arms, but
  `a_scipy` fell 3.1x between methods while `b_scipy` rose slightly.
- **P4** predicted a crossover earlier than 8,242. That now looks likely to be
  **confirmed, but for a different reason than the one I gave.** I attributed it
  to our `A^T v` being a scatter on CSR; the corrected model gets there mostly
  from SciPy's *smaller* fixed term. Confirmation of P4's direction should not be
  read as confirmation of P4's stated mechanism unless `b_ours/b_scipy` on lsqr
  also comes out worse than the 1.8–2.7x seen on GMRES.

Both remain open until the gated two-arm sweep lands. The two-arm numbers, when
they exist, are the ones that may be quoted as ratios.

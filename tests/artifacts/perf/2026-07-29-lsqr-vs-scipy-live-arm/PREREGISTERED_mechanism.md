# Pre-registered mechanism: sparse iterative least squares (`lsqr`) vs live SciPy

**Written and committed BEFORE any lsqr measurement was taken.** Agent
`cc/NobleCedar`. Date 2026-07-29. Predecessor result: `86bcccd74`
(GMRES per-iteration decomposition).

The point of committing this first is that the predictions below cannot be
revised after seeing the numbers. If the measurement contradicts them, the
contradiction gets reported, not the prediction quietly edited.

## The established mechanism I am extending

From the GMRES decomposition, with inner-iteration counts matched exactly
(127/127, 163/163, 227/227) on a nonsymmetric convection-diffusion CSR:

- SciPy's `_isolve` Krylov solvers run their **per-iteration loop in
  interpreted Python** around NumPy calls. Measured fixed per-iteration cost
  `a_scipy ≈ 87.2 µs`, independent of `n`.
- Ours runs in Rust: `a_ours ≈ 4.3 µs`.
- Our **marginal cost per unknown is worse**: `b_ours / b_scipy` measured at
  **1.823x** and **2.668x** across two size segments.
- Net: we win at small `n` purely by not paying the interpreter tax, and the
  advantage decays and inverts near `n ≈ 8,242` (measured loss at side 96).

Modelling per-iteration cost as `a + b*n`, and writing `W` for the number of
NumPy-level vector operations SciPy performs per iteration:

- `a_scipy ≈ W * c_py`, where `c_py` is the per-NumPy-call interpreter and
  dispatch cost (order 4–5 µs).
- `b` for both arms scales with `W` too (same `W` vector ops of length `n`).
- So the crossover `n* = a_scipy / (b_ours - b_scipy) ≈ c_py / (e_ours - e_scipy)`
  where `e` is per-element cost. **`W` cancels.** The crossover is a property of
  interpreter call cost divided by our per-element kernel disadvantage, *not* of
  which method is being run.

## Why I believe it holds for `lsqr`

`scipy.sparse.linalg.lsqr` lives in `scipy/sparse/linalg/_isolve/lsqr.py` and is
**pure Python**. Its per-iteration body is, if anything, heavier in Python terms
than restarted GMRES's: two matrix products per iteration (`A @ v` and
`A.T @ u`, i.e. matvec plus rmatvec), the Golub–Kahan bidiagonalization update,
a Givens rotation, and a convergence test that computes several norms and
ratios **every iteration** in interpreted scalar arithmetic. There is no
compiled inner loop to hide any of it.

This is a genuinely different sparse operation family from the square-system
Krylov solves already measured — it minimizes `||Ax - b||_2` rather than solving
`Ax = b`, and it exercises `A^T` as well as `A` — while keeping the mechanism
identical. That is exactly the case where the structural argument should
transfer if it is real, and the case where a failure would be informative.

## Predictions (falsifiable, in order of how much I would bet)

**P1 — Direction.** We win at the smallest size tested; the incumbent ratio
decays monotonically as `n` grows; a crossover to a loss exists within or just
beyond the tested range.

**P2 — SciPy fixed per-iteration overhead.** `a_scipy` for lsqr comes out
**at or above the GMRES value of 87.2 µs/iteration**, because lsqr's
per-iteration NumPy-call count is at least comparable to restarted GMRES's
~9.3 Gram–Schmidt steps once its rotation and convergence bookkeeping are
counted.

**P3 — Crossover order.** Since `W` cancels in `n*`, the crossover lands in the
**same order of magnitude as GMRES's `n* ≈ 8,242`**, i.e. order `10^4`
unknowns — not order `10^2` and not order `10^6`.

**P4 — The riskiest prediction, and the one I most expect to be wrong in
detail.** Our marginal disadvantage on lsqr should be **worse than the ~2x seen
on GMRES**, therefore the crossover should arrive **earlier than `n ≈ 8,242`**.
Mechanism: lsqr needs `A^T v` every iteration. For a CSR matrix, a
transpose-matvec is either a scatter-add with poor write locality or requires a
CSC structure; SciPy gets this from compiled sparsetools code that walks the
transposed structure directly. If our `A^T v` is a scatter, `b_ours - b_scipy`
widens and the crossover moves down. **If instead the crossover lands at or
above GMRES's, P4 is falsified and our transpose path is better than I think.**

## What would falsify the whole mechanism

- A **loss at the smallest size**. That would mean the interpreter tax is not
  the source of the advantage, or that our lsqr carries a fixed per-call cost
  the GMRES model does not predict.
- A ratio that **does not decay with `n`**, which would break the two-term
  model rather than just its coefficients.
- `a_scipy` measured **well below 87 µs**, which would mean per-NumPy-call
  overhead is not what sets the fixed term.

## Protocol committed to in advance

- **Iteration counts reported for both arms in every cell.** The lsqr
  conformance test (`crates/fsci-conformance/tests/diff_sparse_lsqr.rs`)
  validates the *solution vector*, not the trajectory, so counts may differ. If
  they differ, the cell is labelled as convergence-contaminated and is **not**
  presented as a per-iteration result — the same discipline applied to the
  superseded 3.9679x GMRES row.
- Per-cell normalization only. **No ratio averaged across cells with different
  iteration counts.**
- Independent per-arm A/A null controls, bootstrap-median CI, 2x null margin.
- Full-vector agreement plus true residual for both arms.
- Both engine SHA-256s, the executed-binary ELF SHA-256, **and the identity of
  the machine that built it**, per the infrastructure directive.
- Host: `thinkstation1`, AMD Ryzen Threadripper PRO 5975WX, 32 physical cores,
  AVX2 + FMA, **no AVX-512**. Single CPU pinned, one observed worker thread per
  arm. Host-wide quiescence gate left fail-closed at its existing 20% threshold.
- **The ratio is reported whichever way it falls.**

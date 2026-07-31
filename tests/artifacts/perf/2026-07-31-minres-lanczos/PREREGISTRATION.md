# Pre-registration — Lanczos MINRES replaces the GMRES-delegate stub

**Provenance correction:** this prospective plan entered Git in the same commit
as the implementation and result. Repository history does not prove that it
predated timing, so it must not be treated as an immutable pre-registration.
The predictions remain useful as the author's reported hypothesis scorecard.
Host `thinkstation1`, 64 logical CPUs, live SciPy 1.17.1 in the same invocation.

## The gap

`fsci_sparse::linalg::minres` does not implement MINRES. Its docstring claims
"Uses the Lanczos process to reduce to a tridiagonal system, then applies Givens
rotations", and its body is:

```rust
    // MINRES via GMRES-style approach on symmetric matrix (reliable fallback)
    gmres(a, b, x0, options)
```

`gmres` uses `restart = n.min(20)`. So the entire `minres` public surface is
**restarted GMRES(20)**. That has two costs the real algorithm does not pay:

1. **Restarting discards the Krylov subspace every 20 steps.** MINRES's
   three-term Lanczos recurrence retains the global minimum-residual property
   over the *whole* Krylov space at O(n) memory. GMRES(20) re-minimises only
   over the last ≤20 directions. On a symmetric **indefinite** operator this is
   the textbook stagnation case for restarted GMRES.
2. **Arnoldi orthogonalises against a growing basis.** A 20-step cycle costs
   Σ_{j=1..20} 2jn ≈ 420n flops of modified Gram-Schmidt (≈21n per step) and
   stores 21 length-n basis vectors. Lanczos MINRES costs ~14n flops and ~7
   length-n streams per step at fixed O(1) vector count.

Mechanism (1) is an **iteration-count** lever and therefore a multiples lever;
mechanism (2) is a constant-factor lever worth ~1.3×. They are separable and I
report them separately.

## Counting rule, fixed now

One **A-application** = one `csr_matvec`/`csr_matvec_into` call inside the solve
loop. For the GMRES delegate that is one per Arnoldi step, and equals its
reported `iterations`. For Lanczos MINRES that is one per Lanczos step, and
equals its reported `iterations`. Setup (`r = b - Ax`) and the final true-residual
matvec are excluded from both arms. This rule is fixed before any measurement and
will not be renegotiated afterwards.

## Fixtures

- **INDEFINITE (primary):** `A = L - σI`, `L` = 2-D Dirichlet five-point
  Laplacian on a `side × side` grid (`n = side²`, `nnz ≈ 5n`), `σ = 3.7`.
  `L`'s spectrum is `4 - 2cos(iπ/(s+1)) - 2cos(jπ/(s+1)) ∈ (0, 8)`, so the shift
  makes `A` symmetric indefinite with roughly half its spectrum negative. `σ`
  is deliberately off the `4.0` lattice value, where the shifted operator would
  be exactly singular.
- **SPD (control):** the same `L`, unshifted. CG's fixture. Included so the
  mechanism claim is falsifiable in both directions.
- Sizes `side ∈ {32, 64, 128}`. `rtol = 1e-8`, `x0 = 0`.

## Predictions

| # | Claim | Falsified if |
|---|---|---|
| **P1** | On INDEFINITE at `side ≥ 64`, Lanczos MINRES converges in **≥3× fewer A-applications** than the GMRES(20) delegate. | ratio < 3× |
| **P2** | On INDEFINITE at `side ≥ 64`, wall clock is **≥3× faster** than the delegate. Not independent of P1. | ratio < 3× |
| **P3** | On SPD control, the win is **small (<1.3×)** — GMRES(20) converges within few cycles there, so only mechanism (2) is available. | ratio ≥ 1.3× (would mean I have mis-attributed the mechanism) |
| **P4** | Solver working set drops from 21 length-n basis vectors to **≤8**. | >8 live length-n vectors |
| **P5** | The new MINRES beats **live** `scipy.sparse.linalg.minres` at every size, and the ratio **falls** with `n` (the `a + b·n` model — only `a` is ours). | loses at any size, or ratio rises with `n` |
| **P6** | Conformance: agrees with `scipy.sparse.linalg.minres` within the existing harness `ABS_TOL = 1e-6` on all existing SPD cases **and** on a newly added indefinite case. | any case exceeds `ABS_TOL` |
| **P7** | The Givens recurrence residual estimate stays within **10×** of the true `‖b − Ax‖/‖b‖` at termination. | ratio > 10× |

## Known ways this could disappoint

- If the delegate happens to converge inside one 20-step cycle on the chosen
  indefinite fixture, P1 collapses to P3 and the lever is worth ~1.3×, not
  multiples. `σ = 3.7` is chosen to make that unlikely, not impossible.
- MINRES tracks the residual through the Givens recurrence rather than
  recomputing `b − Ax`. That is what makes it cheap and is also how it can
  report convergence it has not achieved — hence P7 as a separate gate.
- SciPy's `minres` uses a different stopping test (`‖r‖/‖b‖` via its own
  recurrence, plus `A`-norm bookkeeping). Iteration counts between our arm and
  SciPy's are **not** expected to match, so P5 is a wall-clock claim only and
  P6 is a solution-vector claim only.

## Reporting rule

The scorecard is reported whichever way it falls, including P3 and P7, and the
lever is reverted if P1 and P3 both fail.

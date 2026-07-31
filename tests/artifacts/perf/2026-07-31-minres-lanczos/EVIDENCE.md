# Evidence — Lanczos MINRES replaces the GMRES(20) delegate

Measured 2026-07-31 by StormySquirrel on `thinkstation1` (64 logical CPUs),
live SciPy 1.17.1 / NumPy 2.4.3 in the **same invocation** as every FrankenSciPy
arm. The prospective plan is `PREREGISTRATION.md`; it entered Git in the same
commit as this result, so repository history does not establish pre-registration.

**Evidence classification:** correctness and conformance support a production
KEEP. The timing cells are **provisional non-exclusive routing evidence only**:
the host was not quiescent, and the prospective plan was committed in the same
commit as this result rather than immutably before timing. No competitive
performance claim is made from these measurements.

- Harness: `crates/fsci-sparse/src/bin/perf_minres_vs_scipy.rs`
  (`--features live-scipy-bench --profile release-perf`)
- Oracle co-process: `docs/perf_oracle_minres.py --minres-live`
- Harness ELF SHA-256: `8193f84699c7f9501d0a35d27702aa2aee730c66627e763d58e3d9a1cfab07ee`
- Builder: worker `hz1`, `rustflags = -C target-feature=+avx2,+fma`; this host has
  both, verified in `/proc/cpuinfo`.
- Fixture: `A = L − σI`, `L` = canonical Dirichlet five-point Laplacian
  (diagonal 4.001), `σ = 3.7` indefinite / `σ = 0` SPD control. `rtol = 1e-8`,
  `x0 = 0`, cap 20,000 A-applications.
- **All cells are single-threaded.** `csr_matvec_into` only spawns workers at
  `nnz ≥ 2^18`; the largest cell here has `nnz = 81,408`. Nothing below is a
  parallelism result.
- Host was **not quiescent** (`host_busy_fraction` 0.24–0.40 per cell, reported
  inline). Per-cell null pairs (same arm against itself) put the observed noise
  floor at 2.0–8.7%, but they do not repair the failed exclusivity gate.

## What was there before

`fsci_sparse::linalg::minres` was, in its entirety, `gmres(a, b, x0, options)`.
`gmres` uses `restart = n.min(20)`. The docstring already claimed a Lanczos
process and Givens rotations; only now is that true.

## Indefinite fixture (σ = 3.7) — the case MINRES exists for

| side | n | ours-minres | ours-gmres20 (the stub) | scipy-minres |
|---|---|---|---|---|
| 32 | 1,024 | **1,047 A-app, converged**, true res 9.46e-9, 7.907 ms | 20,000 A-app, **NEVER CONVERGED**, true res 2.19e-3, 433.99 ms | 982 A-app, res 1.63e-6, 33.155 ms |
| 64 | 4,096 | **4,467 A-app, converged**, true res 9.89e-9, 115.160 ms | 20,000 A-app, **NEVER CONVERGED**, true res 1.75e-3, 1749.821 ms | 3,811 A-app, res 1.49e-6, 212.432 ms |
| 128 | 16,384 | **19,771 A-app, converged**, true res 9.97e-9, 2003.412 ms | 20,000 A-app, **NEVER CONVERGED**, true res 8.75e-4, 6630.428 ms | 16,584 A-app, res 1.19e-5, 2077.726 ms |

**The stub does not solve this system at any size tested.** It burns its entire
20,000-application budget and is still three to four orders of magnitude away
from the requested tolerance. So "speedup versus the delegate" is not a
meaningful wall-clock ratio here — the honest statement is that the replaced
code produced no answer at all, and the replacement produces one.

The comparable quantity is **cost per A-application**, which isolates the
per-step work from the iteration count:

| n | ours µs/A-app | gmres20 µs/A-app | scipy µs/A-app | ours vs gmres20 | ours vs scipy |
|---|---|---|---|---|---|
| 1,024 | 7.552 | 21.700 | 33.762 | 2.873× | 4.471× |
| 4,096 | 25.780 | 87.491 | 55.742 | 3.394× | 2.162× |
| 16,384 | 101.331 | 331.521 | 125.285 | 3.272× | 1.236× |

## SPD control (σ = 0)

| | A-applications | ms/solve | true residual |
|---|---|---|---|
| ours-minres | **87**, converged | **0.8188** | 8.29e-9 |
| ours-gmres20 | 222, converged | 5.6932 | 8.75e-9 |
| scipy-minres | 46 | 1.7306 | 1.47e-5 |

`SPEEDUP minres_vs_gmres20_delegate = 6.9532×`, and it decomposes exactly:
2.552× fewer A-applications (222/87) × 2.724× cheaper per A-application
(25.64/9.41 µs) = 6.95×. Both mechanisms are live even on SPD.

## The `a + b·n` decomposition (indefinite, 3-point least squares)

Following the house rule that a single ratio must never be quoted for a SciPy
iterative solver:

| | `a` (fixed µs/iteration) | `b` (µs per unknown) |
|---|---|---|
| ours | **1.036** | 0.006118 |
| scipy | **29.498** | 0.005873 |

- `b_ours / b_scipy = 1.042` — **parity on real per-unknown work.**
- `a_ours / a_scipy = 0.0351` — we pay **1/28th** of the fixed per-iteration tax.
- **Crossover `n* ≈ 116,355`.**

This slots into the existing table in `perf_sparse_per_iteration_interpreter_tax`:
`a_scipy` for minres (29.5 µs) sits alongside lsqr's 28.3 µs, and the parity
`b` ratio (1.042) matches lsqr's 1.043 almost exactly. It also independently
supports that memory's CONFIRMED rule that `b` tracks matvecs per iteration:
minres does one matvec per iteration and shows `b_scipy = 0.00587`, against
lsqr's two matvecs and `b_scipy = 0.00993`.

## Pre-registered scorecard — reported as it fell

| # | Claim | Result |
|---|---|---|
| P1 | ≥3× fewer A-applications on indefinite at side ≥ 64 | **CONFIRMED, and understated.** The delegate never converges at all; on SPD it still needs 2.55× more. |
| P2 | ≥3× faster wall clock on indefinite at side ≥ 64 | **PROVISIONALLY OBSERVED, not an admissible speed claim.** The delegate does not converge, so the raw 15.2×/3.3× ratios compare a solve against a failure. Per-A-application: 3.27–3.39×. |
| P3 | SPD win is **small (<1.3×)** | **FALSIFIED — measured 6.95×.** See below. |
| P4 | working set ≤8 length-n vectors (was 21 basis vectors) | **CONFIRMED.** `x, r1, r2, v, y, w, w1, w2` — eight, independent of iteration count; guarded by `minres_working_set_is_independent_of_iteration_count`. |
| P5 | beats live SciPy at every size, ratio **falls** with n | **PROVISIONALLY OBSERVED.** 4.19× → 1.84× → 1.04× is monotone decreasing, but the non-quiescent host prevents a competitive claim. |
| P6 | matches `scipy.sparse.linalg.minres` within `ABS_TOL = 1e-6` | **CONFIRMED**, but only after repairing the harness — see "the test that tested nothing". Max abs diff 5.30e-10 (indefinite); **exactly 0.0** on all three SPD cases. |
| P7 | recurrence residual within 10× of true `‖b−Ax‖/‖b‖` | **CONFIRMED**, and stronger than claimed: identical to all printed digits in every cell, because the returned residual is recomputed with a real matvec. |

### Why P3 was wrong

I predicted GMRES(20) would converge inside roughly one 20-step cycle on a
well-conditioned SPD Laplacian, leaving only the per-step orthogonalization
difference (~1.3×). It needed **11 restart cycles** (222 steps) at `rtol = 1e-8`,
and every restart discards the accumulated Krylov subspace. So the
iteration-count mechanism is not exclusive to indefinite operators — it is
merely *violent* there and *moderate* on SPD. The error was assuming a
convergence rate rather than measuring it.

### A caveat that runs against my own result

Our arm and SciPy's do not use the same stopping test. Ours is `‖r‖/‖b‖ < rtol`
(this module's convention across cg/gmres/bicgstab). SciPy's is
`‖r‖/(‖A‖‖x‖) < rtol`. SciPy's is weaker on these fixtures, so it stops earlier
and less accurately — at side=128 it does 16,584 iterations to a true residual
of 1.19e-5 while we do 19,771 to 9.97e-9. **We do more work and land ~1,200×
more accurate, and the wall-clock ratios above still favour us.** They are
therefore lower bounds on the equal-accuracy speedup, not best cases.

## The test that tested nothing

The differential harness `diff_sparse_iterative_solvers` called every solver as
`fn(A, b, rtol=1e-10, atol=0.0, maxiter=500)`. **`scipy.sparse.linalg.minres`
has no `atol` parameter** — its signature is
`(A, b, x0=None, *, rtol, shift, maxiter, M, callback, show, check)`. The call
raised `TypeError`, an `except Exception` swallowed it into a null arm, and the
Rust side `continue`d past it silently. Every MINRES case had been skipped, so
the harness reported a green "pass" over an **empty MINRES column** — which is
precisely how a GMRES delegate lived behind a docstring claiming Lanczos.

Repaired in this change:
1. `minres` is called without `atol`; the oracle now reports *why* an arm is null.
2. Skips are collected and printed instead of vanishing.
3. A hard assertion that **every** named solver contributed ≥1 compared case, so
   this harness can no longer pass by testing nothing.

Result: 8 compared cases → **12**, MINRES included for the first time.

Two skips are now visible that were previously invisible, both pre-existing and
neither introduced here:
- `bicgstab_6x6_pentadiag_spd` — our BiCGSTAB does not reach 1e-10 there.
- `gmres_64x64_shifted_laplacian_indefinite` — our GMRES(20) does not converge
  in 500 A-applications. SciPy's does, because **SciPy's `gmres` counts
  `maxiter` in outer restart cycles while ours counts inner steps** — SciPy was
  granted 500×20 = 10,000 applications for the same argument. That is a genuine
  conformance divergence, filed rather than fixed here.

## Verification

- `cargo test -p fsci-sparse`: 405 passed, 0 failed, 4 ignored.
- `cargo test -p fsci-sparse --test metamorphic_tests`: 56 passed, including
  `mr_minres_residual_small_on_spd`.
- `cargo test -p fsci-sparse --lib minres`: 9 passed, including the pre-existing
  `minres_matches_scipy_reference_values` (hardcoded SciPy values) and the two
  new guards.
- `diff_sparse_iterative_solvers` vs live SciPy: 12/12 pass, max abs diff 5.30e-10.
- `diff_sparse_iterative_solvers_residual` vs live SciPy: pass.

## Decision

**KEEP FOR CORRECTNESS AND CONFORMANCE.** The lever replaces an implementation
that could not solve symmetric indefinite systems at all with one that does,
and the repaired differential harness now exercises MINRES instead of silently
skipping it. The 6.95× SPD observation and the 4.19×-to-1.04× live-SciPy trend
remain provisional routing evidence pending an exclusive rerun; they are not
the basis of the production decision.

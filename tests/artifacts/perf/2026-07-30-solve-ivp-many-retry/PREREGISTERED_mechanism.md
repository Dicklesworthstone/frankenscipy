# PRE-REGISTERED: does the historical `solve_ivp_many` 1481–1599× survive a properly gated arm?

Written and committed **before any gated `lotka-many` timing exists**. Author:
cc pane (BlackThrush). Date: 2026-07-30.

## Why this row

`docs/perf_ledger_cc.md` carries `solve_ivp_many` as **`VOID-NONULL /
CONFORMANCE-BLOCKED`**. Its historical claim — 1481× at N=200 and 1599× at
N=1000 against a Python loop over `scipy.integrate.solve_ivp` — was measured in
separate invocations with no A/A null, no executable SHA-256 and no counted
mechanism, so it is not admissible evidence. A live same-invocation SciPy 1.17.1
arm then *refused to time it at all*: at sample 138 of the first deterministic
trajectory FrankenSciPy differed from SciPy by **711.439 tolerance units**,
blowing the harness's `<= 100` scaled-difference contract.

Its committed retry predicate:

> implement solver-specific RK45 dense output for `t_eval`, pass the live arm's
> `<=100` scaled-difference contract over all 150 samples, then rerun the exact
> batch surface with genuine SciPy in the same invocation, independent A/A nulls
> for both arms, executable SHA-256, full hardware/thread provenance, and the
> bootstrap-median CI gate.

The first clause is now done (`crates/fsci-integrate/src/rk.rs`), which is what
makes the predicate satisfiable. This document fixes the predictions before the
remaining clauses are run.

## The conformance fix, and what is already established about it

FrankenSciPy sampled `t_eval` with a generic **cubic Hermite** through the step
endpoints. SciPy's RK45 uses the **quartic Dormand–Prince dense output** over
all seven stage derivatives: with `Q = Kᵀ P` and `x = (t − t_old)/h`,
`y(t) = y_old + h·Q·[x, x², x³, x⁴]ᵀ`. Cubic Hermite is one order lower, so it
drifts mid-step even when both endpoints agree — which is exactly the reported
signature (final state matched, sample 138 did not).

Two things are already measured and are **not** predictions:

- The 28 entries of `RK45_P` as written in Rust are **bit-identical** to
  `scipy.integrate._ivp.rk.RK45.P` (checked as exact rationals).
- Re-implementing the Rust routine's exact arithmetic in Python and feeding it
  SciPy's own `K`/`h`/`y_old` over 2,622 samples of the failing trajectory
  reproduces `scipy`'s `dense_output()` to **6.97e-15** relative, while the old
  cubic Hermite is off by **7.59e-6** — a **1.09e9×** gap.
  (`raw/verify_rk45_dense_output.py`, output committed alongside.)

So the interpolant itself is settled. What is *not* settled is whether the two
arms' **step sequences** stay close enough over 150 samples × N trajectories for
the `<=100` contract to pass, and what the gated ratio actually is.

## Predictions

**P1 — the conformance gate now passes.** `max_scaled_diff` over all 150 samples
of every trajectory falls below the harness's `100` limit, point estimate
**≤ 10**. Rationale: with the interpolant matched to ~7e-15, the only remaining
source of divergence is ordinary floating-point difference in the adaptive step
sequence, and both arms run the same Dormand–Prince tableau at the same
`rtol=1e-8 / atol=1e-10`. Falsified if the gate still aborts, which would mean
the residual is step-selection divergence rather than interpolation and that the
bead's stated root cause was incomplete.

**P2 — the historical order of magnitude survives.** Unlike the GMRES/lsqr/qmr
family, this is not a per-iteration kernel contest where SciPy's fixed tax
decays with `n`; it is a structural gap — SciPy has **no batched `solve_ivp`**,
so a user loops it in Python and pays a Python RHS call per stage per step, N
times serially. Predict the gated same-invocation ratio at N=1000 lands in
**[800×, 3200×]**, i.e. within 2× of the historical 1599×. This is the risky
half: every previous resurrection in this campaign came in *below* its
historical number.

**P3 — the win decomposes into two factors that multiply.** Predict
callback-elimination (FrankenSciPy forced to one thread vs SciPy one thread)
contributes **20–30×**, and N-way parallelism contributes **25–64×** on this
32-physical-core host, with their product reproducing P2's ratio to within 1.5×.
If the product overshoots the measured end-to-end ratio by more than that, the
factors are not independent and the "callback lever × N-way parallel" story in
the ledger is wrong.

**P4 — observed threads match requested.** The harness reports
`actual_observed_frankenscipy_worker_threads` and
`actual_observed_scipy_worker_threads`; predict both equal their requested
values, with SciPy's BLAS capped at one thread per process. Recorded because the
standing requirement is *observed*, not requested, threads.

**P5 — `lotka-final-many` stays consistent.** The completion-only control
(`t_eval=None`, compared at `t=10`) was admissible while the sampled path was
blocked. Predict the sampled `lotka-many` ratio agrees with the
`lotka-final-many` ratio to within 1.3× at the same N — the sampling should cost
both arms proportionally. A large gap would mean `t_eval` sampling is itself a
significant cost asymmetry, which is a different (and reportable) finding.

## Method

Existing harness `crates/fsci-integrate/src/bin/perf_bdf_vs_scipy.rs`, fixture
`lotka-many` (the exact historical surface: Lotka–Volterra ensemble, RK45,
`t_span=[0,10]`, `rtol=1e-8`, `atol=1e-10`, 150 requested samples, argv `n` =
batch size), against a live SciPy 1.17.1 arm in the **same invocation**.
Standing requirements, all supplied by the harness: the corrected null gate
including its median clause, **actual observed** worker threads for both arms,
host identity, ELF SHA-256 self-reported from inside the process, and
`rch exec --base <sha> --clean-overlay` so no co-tenant agent's edits can enter
the binary. Construction, serialization and conformance checks outside timing.

Reported whichever way it falls. If the host cannot be made exclusive the run is
waived to `PROVISIONAL_NON_EXCLUSIVE` and may not be called DECIDED.

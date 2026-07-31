# RESULT — qmr vs live SciPy 1.17.1: the dispatch-count model is falsified, the matvec-count model survives

Completes the campaign pre-registered in `1f6f8ffdc`
(`PREREGISTERED_mechanism.md`, committed before any qmr timing existed).

**Scorecard: one confirmed, three falsified, one partial. The riskiest
prediction — the one whose stated assumption I flagged as the likeliest
falsifier — is the one that held.**

| | prediction | measured | verdict |
|---|---|---|---|
| **P1** fixed tax scales with dispatch count | `a_qmr/a_lsqr ∈ [2.0, 2.8]`, point 2.389 | **1.415×** | ❌ **FALSIFIED** |
| **P2** marginal cost set by matvec count | `b_ours/b_scipy ∈ [1.25, 1.75]`, point 1.5 | **1.380×** | ✅ **CONFIRMED** |
| **P3** reachable crossover | `n* ∈ [8×10³, 3×10⁴]`; win at side ≤ 96; cross by side 128–160 | `n* = 8,136`; **lose at side 96**; crossed by side 96 | ⚠️ **PARTIAL** — interval hit at its lower edge, clause falsified |
| **P4** iteration counts match | exact or ±1 at every size | exact at sides 16–64; **±2, ±14, ±4, ±12** at 96–192 | ❌ **FALSIFIED at scale** |
| **P5** side-16 headline ratio | `∈ [14, 25]`, point 18.9× | **9.797×** | ❌ **FALSIFIED** |

## Measurement

Fixture, harness, gate and provenance identical to the lsqr run. Host
`thinkstation1` (5975WX, 32C/64T, AVX2+FMA, no AVX-512), `powersave`,
pinned `affinity=63`, one worker thread per arm, `python_blas_thread_cap=1`,
21 rounds per cell. SciPy 1.17.1 / NumPy 2.4.3 / CPython 3.13,
`scipy_engine_sha256=f9d7ace0…`. Executed ELF
`f48c77421bca7f08631f8caab0c9f0f2ffba3cb0dee385d4795fa495828e6068`, built on
rch worker `vmi1293453` via the deterministic-overlay form and retrieved by scp.

**Evidence class: `PROVISIONAL_NON_EXCLUSIVE` for every cell.** Host load was
~10 throughout, so the fail-closed quiescence gate was waived
(`FSCI_SPARSE_ALLOW_NON_EXCLUSIVE=1`) and no cell may be called DECIDED. This is
the same class the lsqr run carries, which is precisely what makes the two
directly comparable. Every cell cleared the 2× A/A-null margin.

| side | n | it ours/SciPy | matched | µs/it ours | µs/it SciPy | **whole-job** | per-iter |
|---:|---:|---:|:--:|---:|---:|---:|---:|
| 16 | 256 | 40/40 | YES | 4.369 | 42.801 | **9.7970×** | 9.7970× |
| 32 | 1,024 | 80/80 | YES | 17.016 | 53.472 | **3.1427×** | 3.1425× |
| 48 | 2,304 | 108/108 | YES | 39.460 | 71.163 | **1.8090×** | 1.8034× |
| 64 | 4,096 | 136/136 | YES | 72.885 | 92.536 | **1.2739×** | 1.2696× |
| 96 | 9,216 | 200/198 | no | 153.875 | 144.820 | **0.9359×** | 0.9412× |
| 128 | 16,384 | 255/269 | no | 273.279 | 225.380 | **0.8705×** | 0.8247× |
| 160 | 25,600 | 393/397 | no | 424.301 | 321.264 | **0.7653×** | 0.7572× |
| 192 | 36,864 | 895/883 | no | 625.921 | 461.677 | **0.7232×** | 0.7376× |

Per-iteration fit over the four count-matched cells (`n` 256…4,096):

```
ours   a = -0.914 µs (~0)   b = 0.017889 µs/unknown   R² = 0.9994
scipy  a = +40.113 µs       b = 0.012958 µs/unknown   R² = 0.9984
marginal per-unknown ours/scipy = 1.380×
fitted crossover n* = 8,136 (side ~90)
```

Same-session lsqr control, re-measured rather than quoted across sessions:

```
ours   a = -0.808 µs (~0)   b = 0.010353 µs/unknown   R² = 0.9994
scipy  a = +28.342 µs       b = 0.009927 µs/unknown   R² = 0.9996
marginal per-unknown ours/scipy = 1.043×   (committed value: 1.006×)
```

The control reproduces the committed lsqr numbers cell for cell
(12.08/3.93/1.69/1.26 against 12.02/3.93/1.79/1.29), so the host has not
drifted and the `a_qmr` vs `a_lsqr` comparison is within-session.

## P1 falsified: `a` is not proportional to a dispatch count

`a_scipy(qmr) = 40.113 µs` against `a_scipy(lsqr) = 28.342 µs` is **1.415×**,
where the counted dispatches say 2.389×. Implied cost per dispatch unit:

```
lsqr   28.342 / 18 = 1.575 µs per unit
qmr    40.113 / 43 = 0.933 µs per unit
```

Not a constant — off by 1.7×. **`a_scipy ≈ c·D` with a single `c` is wrong**,
and the counting rule fixed in the pre-registration is what makes that
conclusion unarguable rather than a matter of re-counting after the fact.

Leading candidate explanation, offered as a *hypothesis for a future
pre-registration and not as a claim*: the two loop bodies allocate very
differently. lsqr is written out-of-place (`u = A.matvec(v) - alfa*u`,
`w = v + t2*w`, …) and allocates ~11 fresh length-`n` temporaries per
iteration. qmr is written in-place (`v[:] = vtilde[:]`, `v *= …`, `d += …`)
and allocates ~7, and its four identity-preconditioner round-trips return the
input object without allocating at all. If temporary allocation rather than
call count is the driver, `D` counts the wrong thing. That is testable by
counting allocations instead of dispatches — and it must be pre-registered
before the next measurement, not fitted to these two points.

## P2 confirmed — but for reasons that are only half right

`b_ours/b_scipy = 1.380×`, inside `[1.25, 1.75]` and near the 1.5 point
estimate. The control makes the contrast clean: **the same fixture, the same
host, the same session, two sparse matvecs each in lsqr gives 1.043× (parity);
adding our third matvec in qmr gives 1.380×.** One variable changed, `b` moved,
and it moved by roughly the matvec ratio. The prediction's *number* held.

Its stated *reason* did not fully hold. The pre-registration asserted that
SciPy's extra length-`n` temporaries contribute negligibly to `b`, and flagged
that as the likeliest falsifier. Decomposing across the two methods:

```
b_ours(qmr)  / b_ours(lsqr)  = 0.017889 / 0.010353 = 1.728×   (3 matvecs vs 2)
b_scipy(qmr) / b_scipy(lsqr) = 0.012958 / 0.009927 = 1.305×   (31 vector ops vs 14)
```

SciPy's 17 extra streamed temporaries cost it **~30% more per unknown**, not
nothing. Both terms moved; the ratio landed in the interval because they moved
together. Recorded plainly so the next cycle does not inherit a rationale that
was never tested. Caveat: the two fits span different `n` ranges (qmr 256…4,096,
lsqr 256…9,216) because the qmr cells above 4,096 are count-mismatched, so
cache residency is not identical between them.

## P3 partial: the crossover was measured, not extrapolated — and it arrives early

This is the clause the lsqr run could not deliver, and the reason sides 128–192
were run at all. The fitted `n* = 8,136` lands inside the predicted
`[8×10³, 3×10⁴]`, at its very lower edge, and the direct measurement brackets it
independently: **side 64 = 1.2739× (win), side 96 = 0.9359× (loss)**, so
`n* ∈ (4,096, 9,216)`. Fit and observation agree.

The accompanying clause is falsified. I predicted a win at side ≤ 96 and a
crossing "by side 128–160"; we are already losing at side 96. The crossover is
~1.7× earlier than the point estimate, which follows directly from P1's failure
— a smaller `a_scipy` means the tax runs out sooner.

## P4 falsified at scale, and the falsification is informative

Counts match exactly at sides 16–64 and diverge above:

```
side  96   ours 200   SciPy 198
side 128   ours 255   SciPy 269
side 160   ours 393   SciPy 397
side 192   ours 895   SciPy 883
```

At side 128 **SciPy needs 14 more iterations than we do**. The recurrences are
algebraically identical, so this is the residual bookkeeping: SciPy carries
`r ← r − s` recursively and that estimate drifts from the true residual as `n`
grows, becoming pessimistic and costing SciPy ~5% extra iterations. Our third
matvec buys a stopping test that is actually true. So the extra matvec is not
pure overhead — it partially pays for itself in counted work, which is why the
whole-job column at side 128 (0.8705×) is *worse* for us than the per-iteration
column (0.8247×) would suggest on its own.

## Chooser statement (whole-job, this fixture, one thread)

> For un-preconditioned QMR on a 2-D convection–diffusion system at
> `rtol = 1e-5`: **use FrankenSciPy below roughly `n ≈ 8,000` unknowns and SciPy
> above it.** At `n = 256` FrankenSciPy finishes the job 9.80× faster; the
> advantage decays monotonically to 1.27× at `n = 4,096`, crosses to a loss
> between `n = 4,096` and `n = 9,216`, and reaches 0.72× (a 1.38× loss) at
> `n = 36,864`. The win below the crossover is SciPy's fixed ~40 µs of
> interpreted per-iteration bookkeeping; the loss above it is our third sparse
> matvec per iteration.

Nothing here licenses a size-general qmr claim, and nothing here licenses
claiming our qmr kernel is faster than SciPy's — per unknown it is 1.38× slower.

## The lever this produces, stated as a hypothesis to pre-register next

Our third matvec exists only to recompute `b − Ax` for the convergence test.
Carrying the residual recursively as SciPy does (optionally verifying the true
residual once at exit) would remove it. If `b_ours` fell toward `2/3 × 0.017889
≈ 0.0119`, `b_ours/b_scipy` would become ~0.92 — **marginally cheaper than
SciPy, eliminating the crossover entirely** and leaving the ratio asymptoting to
~1.09× rather than decaying through 1.0. Against that, P4 shows the recursive
residual costs SciPy ~5% extra iterations at side 128, which we would then
inherit. The net is genuinely uncertain, which is exactly what makes it worth
pre-registering rather than asserting.

## Defects found by this measurement

Two, both filed, neither a perf issue:

- **`frankenscipy-9pfja`** — `qmr`'s three breakdown gates used
  `f64::EPSILON * 1e6` (2.220e-10) where SciPy uses `eps` (2.220e-16), a
  million times looser. `delta = wᵀv` and `epsilon = qᵀAp` legitimately reach
  1e-9…1e-12 as the Lanczos vectors approach orthogonality, so healthy solves
  aborted at `n ≥ 4,096` with `converged=false` and residuals from 4.4e-4 to
  9.07e-1. Diagnosed by replaying SciPy's own recurrences and logging the gated
  quantities; the replay predicts the abort iteration **exactly** (121 at side
  64, 151 at side 96). Fixed to `BREAKDOWN_TOL = f64::EPSILON`; side 64 then
  converges at 136 iterations, exactly SciPy's count, with agreement
  `relative_l2_diff = 6.109e-13` (was 3.342e-5 with 163 tolerance mismatches).
  The pre-fix cells are preserved in `raw/prefix_breakdown_bug/`.
- **`frankenscipy-6pdfn`** — `lsmr` delegates to `lsqr` and `minres` delegates
  to `gmres`, both while claiming "Matches `scipy.sparse.linalg.…`" in their
  docstrings. Found in the same survey; unrelated to timing.

**Disclosure on the fix and this measurement.** The breakdown fix was made
*after* the first sweep aborted at side ≥ 64, not after seeing a ratio. It
changes when the solver gives up, not what it does per iteration, so it cannot
move `a` or `b`; it only makes cells exist that otherwise aborted. The four
cells that converged under both binaries agree closely (side 16: 9.8134× before,
9.7970× after; side 48: 1.7962× before, 1.8090× after). The three-cell fit
available before the fix gave `a_scipy = 40.113 µs` and `b_ours/b_scipy = 1.337×`
— the same P1 falsification and the same P2 confirmation as the eight-cell run.
**No prediction's verdict depends on the fix.** What the fix bought was P3:
without it the crossover could only have been extrapolated, which is the exact
weakness this campaign was pre-registered to avoid.

## Artifacts

- `PREREGISTERED_mechanism.md` — predictions, committed `1f6f8ffdc`
- `raw/qmr_side_*.txt` — eight cells, fixed arm
- `raw/lsqr_resample_side_*.txt` — four control cells, same session
- `raw/prefix_breakdown_bug/` — seven pre-fix cells documenting the defect
- `raw/decompose.py` — the `a`/`b` fit; excludes count-mismatched cells and
  prints every skipped cell with its reason
- `raw/qmr_breakdown_replay.py` — SciPy-recurrence replay that localizes the
  breakdown gate to the iteration

# PRE-REGISTERED: the per-iteration interpreter tax, applied to `qmr`

Written and committed **before any `qmr` timing measurement exists**, so the
predictions below cannot be revised after seeing the numbers. Author: cc pane
(BlackThrush). Date: 2026-07-30.

Predecessors, both committed:

- `86bcccd74` — GMRES win attributed to SciPy's fixed per-iteration interpreter
  cost `a_scipy = 87.239 µs` at restart 20, with our marginal per-unknown cost
  **1.82–2.67× worse** than SciPy's, giving a real crossover at `n* ≈ 8242`.
- `2c651d13e` — lsqr, same fixture family, **three of four predictions
  falsified**. Measured `a_scipy = 29.654 µs`, and the marginal per-unknown
  ratio came out at **1.006× = PARITY**, not the predicted 1.8–2.7× deficit.

The rule that survived those two runs, stated there as a testable hypothesis:

> we lose on dense vector streaming, we tie on sparse matvec

`qmr` is chosen because it is the sharpest available test of that rule. It is a
**different operation family** (short-recurrence two-sided Lanczos
biorthogonalization for square non-symmetric systems, not Arnoldi and not
least-squares), while holding the dominant primitive fixed at **two sparse
matvecs per iteration** — exactly lsqr's primitive mix, which measured parity.
One variable is changed: the amount of interpreted bookkeeping wrapped around
those two matvecs. If the rule is right, that change must move the fixed term
`a` and leave the marginal term `b` alone.

## The model being tested

`a_scipy ≈ c · D`, where `D` is a count of per-iteration Python-level numpy
dispatches read off the SciPy 1.17.1 source, and `c` is calibrated from the one
committed same-host data point (lsqr).

**Counting rule** (fixed here so it cannot be tuned later). Per steady-state
iteration of the solver's main loop:

- `H` = `LinearOperator.matvec` / `.rmatvec` calls. Each is a full Python
  round-trip: shape validation, `asanyarray`, `_matvec`, `reshape`.
- `V` = numpy operations producing or consuming a length-`n` array
  (`*`, `-`, `+`, `*=`, `-=`, `+=`, slice-assign, `np.linalg.norm`, `np.dot`).
- `D = 2H + V`, charging one `LinearOperator` round-trip as two array-op
  equivalents.

Pure-Python float arithmetic on scalars is **not** counted.

### Counts, enumerated from source

`scipy/sparse/linalg/_isolve/lsqr.py`, main `while` body:

| op | V |
|---|---|
| `A.matvec(v) - alfa*u` | 2 (+1 H) |
| `np.linalg.norm(u)` | 1 |
| `(1/beta)*u` | 1 |
| `A.rmatvec(u) - beta*v` | 2 (+1 H) |
| `np.linalg.norm(v)` | 1 |
| `(1/alfa)*v` | 1 |
| `dk = (1/rho)*w` | 1 |
| `x = x + t1*w` | 2 |
| `w = v + t2*w` | 2 |
| `np.linalg.norm(dk)**2` | 1 |

**`H = 2`, `V = 14`, `D_lsqr = 18`.**

`scipy/sparse/linalg/_isolve/iterative.py::qmr`, `for` body, `iteration > 0`:

`H = 6` — `M2.matvec(y)`, `M1.rmatvec(z)`, `A.matvec(p)`, `M1.matvec(vtilde)`,
`A.rmatvec(q)`, `M2.rmatvec(wtilde)`. **Four of those six are the identity
preconditioners** SciPy synthesises when `M1 is None and M2 is None`:
`LinearOperator(A.shape, matvec=id, rmatvec=id)`. They do zero numerical work
and exist only to keep the loop body uniform.

`V = 31` — `norm(r)`, `v[:]=vtilde[:]`, `v*=`, `y*=`, `w[:]=wtilde[:]`, `w*=`,
`z*=`, `dot(z,y)`, `ytilde -= (…)*p` (2), `p[:]=ytilde[:]`,
`ztilde -= (…)*q` (2), `q[:]=ztilde[:]`, `dot(q,ptilde)`,
`vtilde[:]=ptilde[:]`, `vtilde -= beta*v` (2), `norm(y)`, `wtilde[:]=w[:]`,
`wtilde *=`, `wtilde +=`, `norm(z)`, `d *=`, `d += eta*p` (2), `s *=`,
`s += eta*ptilde` (2), `x += d`, `r -= s`.

**`H = 6`, `V = 31`, `D_qmr = 43`.**  `D_qmr / D_lsqr = 2.389`.

### Algebraic identity of the two arms, verified by source reading

FrankenSciPy `qmr` (`crates/fsci-sparse/src/linalg.rs:2728`) and SciPy's `qmr`
were compared recurrence by recurrence with `M1 = M2 = I`. They agree exactly:
`p ← v − (ξδ/ε)p`, `q ← w − (ρδ/ε)q`, `ṽ ← Ap − βv`, `w̃ ← Aᵀq − βw`,
`θ ← ρ/(γ_prev|β|)`, `γ ← 1/√(1+θ²)`, `η ← −η(ρ_prev/β)(γ/γ_prev)²`,
`d ← ηp + (θ_prev γ)² d`, `x ← x + d`.

**Exactly one structural difference**, and it is the basis of P2: SciPy carries
the residual recursively (`s ← ηAp + (θ_prev γ)² s`, then `r ← r − s`) and tests
`‖r‖ < rtol·‖b‖`. FrankenSciPy recomputes the **true** residual `b − Ax` every
iteration and tests `‖b−Ax‖/‖b‖ < tol`. Both resolve to the same relative-residual
criterion (`_get_atol_rtol` returns `atol = max(0, rtol·‖b‖)`), but ours costs
**one extra sparse matvec per iteration**: three (`A·p`, `Aᵀ·q`, `A·x`) against
SciPy's two.

## Predictions

Calibration: `c = a_lsqr / D_lsqr = 29.654 / 18 = 1.647 µs` per dispatch unit.
All comparisons are **within-host, same fixture family** (`thinkstation1`,
powersave, 5975WX). `a_lsqr` will be **re-measured in the same session** rather
than quoted across sessions.

**P1 — the fixed tax scales with the dispatch count.**
`a_scipy(qmr) / a_scipy(lsqr) ∈ [2.0, 2.8]`, point estimate **2.389**
(`a_scipy(qmr) ≈ 70.8 µs`). Scored a hit only inside that interval.

**P2 — the marginal cost is set by the sparse-matvec count, not the vector-op
count.** `b_ours / b_scipy ∈ [1.25, 1.75]`, point estimate **1.5**, driven by
our third matvec. Equivalently `b_ours ≈ 0.0152 µs/unknown` against
`b_scipy ≈ 0.0101`. This prediction contains the model's weakest assumption —
that SciPy's 31 vector ops per iteration versus our ~8 contribute *negligibly*
to `b` — and it is the most likely falsifier. It is asserted because that is
precisely what the lsqr datum showed (14 vs ~6 vector ops, `b` ratio 1.006).

**P3 — a reachable crossover exists.**
`n* = a_scipy / (b_ours − b_scipy) ≈ 70.8 / 0.0051 ≈ 1.39 × 10⁴` unknowns
(side ≈ 118). Predict `n* ∈ [8×10³, 3×10⁴]`. Concretely: **FrankenSciPy wins at
side ≤ 96 and the ratio falls below 1.0 by side 128–160.** Unlike lsqr, whose
crossover sat 60× outside the measured range, this one is directly observable
and I commit to measuring past it.

**P4 — iteration counts match.** Because the recurrences are algebraically
identical and only the residual bookkeeping differs, counts agree exactly or
differ by ≤ 1 at every size. Any cell whose counts differ is excluded from the
`a`/`b` fit, as in the lsqr run.

**P5 — the headline ratio at the smallest size.** At side 16 (`n = 256`),
two-arm ratio `∈ [14, 25]`, point estimate **18.9×**, exceeding lsqr's measured
12.02× at the same size because `a` is ~2.4× larger while our per-iteration cost
at `n = 256` is still tiny.

## What would falsify the surviving rule outright

If `b_ours / b_scipy` lands near 1.0 despite our extra matvec, the matvec count
does not drive `b` and the rule "we tie on sparse matvec" is really "we tie on
anything bandwidth-bound", which is a weaker and different claim. If instead
`b_ours / b_scipy < 1.0` — i.e. SciPy is marginally *worse* — then its 31
streamed length-`n` temporaries do cost real per-unknown bandwidth, `V` feeds
both `a` and `b`, and the clean `a`/`b` separation this whole campaign rests on
is wrong.

## Method, unchanged from the lsqr run

Same live two-arm harness (`perf_sparse_vs_scipy.rs`) and persistent Python
co-process. Construction, serialization, iteration counting and parity checks
outside timing; interleaved arms in one invocation; per-arm A/A nulls;
bootstrap-median CI with the corrected 2× null margin (`88721b385`); per-cell
normalization with no ratio averaged across differing counts; full-vector and
true-residual agreement; both engine SHA-256s; `python_blas_thread_cap=1`.
Host load is currently ~10, so the fail-closed quiescence gate will not pass and
the run will be waived to `PROVISIONAL_NON_EXCLUSIVE`, exactly as the lsqr run
was — which is also what makes the two directly comparable.

Reported whichever way it falls.

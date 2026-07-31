# PRE-REGISTERED: `curve_fit_many` trace-study job versus live SciPy

Written and committed before constructing the benchmark harness or running any
timed candidate/incumbent probe. Date: 2026-07-31. Bead:
`frankenscipy-tskpy`.

## Existing claim and evidential gap

`docs/perf_ledger_cc.md` reports `curve_fit_many` as 32.9--113x faster than a
serial Python loop over `scipy.optimize.curve_fit` for 500 and 2,000
three-parameter exponential fits. The old comparison was not a
same-invocation paired experiment and did not give the incumbent either a
parallel map or the analytic Jacobian available for the named model. It also
lacked independent A/A controls, executable identity, observed task counts,
and the corrected median-CI gate. The old values are routing evidence, not a
current incumbent claim.

The registered mechanism has two parts: FrankenSciPy inlines the model and
finite-difference solver control in Rust, avoiding repeated Python/NumPy
callbacks, and `curve_fit_many` distributes independent traces across native
threads. The historical SciPy arm paid both the callback tax and serial batch
execution. A persistent thread pool around public SciPy `curve_fit`, especially
with a public analytic Jacobian, removes the second factor and much of the
first. This experiment predicts the headline collapses while testing whether a
durable whole-job win survives.

## Whole job and public incumbent screen

The named job is a recognizable fluorescence/relaxation trace study: fit

`y(x) = a*exp(-b*x) + c`

to 2,000 independent 80-sample traces on evenly spaced `x` in `[0,5]`. The
generating parameters and bounded uniform noise are produced by the fixed
64-bit LCG used by the original `curve_fit_many` conformance fixture
(`state=12345`, multiplier `6364136223846793005`, increment `1`), with
`a in [1,3]`, `b in [0.3,1.3]`, `c in [0,1]`, and noise in
`[-0.01,0.01]`. Every fit starts from `(1,1,0)`. Data construction and pool
construction are outside timing.

A timed arm must materialize all 2,000 parameter triples, fitted curves,
per-trace residual sums of squares, success/convergence counts, the worst-fit
trace, population summaries, and a stable checksum. Thus the timed surface is
the full study a user consumes, not one LM iteration.

Before effect timing, the harness screens these semantically valid public
SciPy 1.17.1 arms on the exact fixed study:

1. a scalar loop over `curve_fit(..., method="lm")` with numerical Jacobians;
2. the same public numerical-Jacobian calls through a persistent thread pool
   sized to the process affinity;
3. a scalar loop over public `curve_fit(..., method="lm", jac=analytic_jac)`;
4. the analytic-Jacobian calls through the persistent thread pool; and
5. one public joint `least_squares` solve with the same separable residuals and
   an analytic block-sparse Jacobian, eligible only if it preserves all 2,000
   per-trace quality records and independent-fit output semantics.

The lowest five-round median among quality-eligible arms is frozen as the
incumbent before paired rounds. Screen timings are disclosed and never pooled
into the primary effect. Pool arms must report actual active tasks rather than
the configured pool capacity. FrankenSciPy receives its existing public
`curve_fit_many` model-only API; its lack of an analytic-Jacobian input is a
real public-surface difference, not hidden from the comparison.

The pre-timing scientific gate fails closed unless:

- every arm materializes exactly 2,000 finite parameter triples and 2,000
  finite non-negative residual records;
- every trace improves on the `(1,1,0)` initial-guess residual;
- every trace has fitted RMSE at most `0.02`;
- the selected SciPy and FrankenSciPy fitted curves differ by at most `5e-4`
  RMS on every trace; and
- both arms identify the same worst-fit trace or their two worst-fit residuals
  differ by at most 1% of the larger residual.

The harness reports median/p95/p99/max fitted RMSE, parameter error versus the
known generators, residual totals, success counts, worst trace, and checksums.
The curve-level agreement gate permits legitimate LM path differences while
preventing a speedup from early termination or failed fits.

## Predictions and falsifiers

**P1 -- strongest SciPy arm.** Predict the persistent analytic-Jacobian
`curve_fit` pool wins the public screen. Falsified if a scalar, numerical,
or joint sparse arm is fastest.

**P2 -- old magnitude collapses.** Predict the paired
`SciPy / FrankenSciPy` 95% CI has an upper endpoint below `11.3x`, more than a
tenfold collapse from the historical 113x headline after incumbent
parallelism and derivative strength are admitted. Report the ratio whichever
way it falls.

**P3 -- durable whole-job win survives.** Predict the paired
`SciPy / FrankenSciPy` 95% CI lies wholly above `3x`. This is the campaign
claim; if its lower endpoint is at most three, the old headline is retired
without replacing it with a FrankenSciPy chooser.

**P4 -- removed serial-batch tax is visible.** Predict the fastest conforming
pool/joint arm is at least `8x` faster than scalar numerical `curve_fit` in the
screen medians. This directly tests the claimed batch-parallel mechanism. It
is falsified independently of the primary effect if the screen ratio is below
`8x`.

**P5 -- observed execution.** On the planned 32-CPU affinity, predict 32
FrankenSciPy solve workers are directly observed in an untimed job probe and
more than one active SciPy task is observed if a pool arm wins. Actual counts
are reported even if either prediction fails; affinity or requested capacity
is never substituted.

## Measurement and decision gate

The primary experiment uses 15 balanced, interleaved paired rounds with three
complete trace studies per timed sample, plus 15 independent
FrankenSciPy/FrankenSciPy and 15 independent SciPy/SciPy A/A ratios. The
effect is the median of per-round `SciPy / FrankenSciPy` ratios. A
deterministic 10,000-resample bootstrap produces the 95% median interval.

A WIN or LOSS is decided only if all of these hold:

1. the effect CI excludes one in the claimed direction;
2. the point-effect distance from one exceeds twice the widest A/A bootstrap
   half-width;
3. the nearer effect-CI endpoint's distance from one exceeds twice the widest
   A/A CI endpoint distance from one; and
4. both A/A medians are in `[0.98,1.02]`.

Whether an A/A CI straddles one is recorded as telemetry and is not a veto.
Ratio CV is provenance only. Otherwise the result is `UNDECIDABLE`, regardless
of the raw ratio.

The harness additionally fails closed without live SciPy 1.17.1, a performance
governor on every affinity CPU, a fresh numeric Agent Mail `trj` booking claim,
exact source/build provenance, or the pre-timing scientific gate. It
self-reports the running ELF SHA-256 and SciPy engine SHA-256 from inside the
processes, records host and boot identity, affinity, runtime ISA, exact input
hashes, and actual observed tasks. The executable must be built through
`rch exec --base <exact-commit> --clean-overlay --no-overlay` while reusing
the single repository target directory `/data/tmp/cargo-target`.

## Chooser boundary

Choose FrankenSciPy `curve_fit_many` for this exact 2,000-trace study only if
its corrected paired CI lies wholly above `3x` and every scientific/provenance
gate passes. If SciPy wins, the FrankenSciPy advantage is below `3x`, or the
result is undecidable, choose the measured public SciPy arm. In every outcome
retire the old 32.9--113x magnitude for this workload rather than falling back
to the scalar-loop comparator.

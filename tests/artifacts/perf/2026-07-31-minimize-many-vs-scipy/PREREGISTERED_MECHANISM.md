# PRE-REGISTERED: `minimize_many` multistart job versus live SciPy

Written and committed before constructing the benchmark harness or running any
timed candidate/incumbent probe. Date: 2026-07-31. Bead:
`frankenscipy-llznz`.

## Existing claim and evidential gap

`docs/perf_ledger_cc.md` reports `minimize_many` as 271--275x faster than a
Python loop over `scipy.optimize.minimize` for 6-D Rosenbrock multistart
optimization. The historical measurements were not a same-invocation paired
experiment and had no independent A/A controls, executable identity, observed
thread count, or corrected median-CI gate. They are routing evidence, not a
current incumbent claim.

The old mechanism combines two effects:

1. FrankenSciPy runs the BFGS control flow and objective/gradient callbacks as
   compiled Rust, while SciPy crosses Python/NumPy boundaries during each
   solve.
2. `minimize_many` distributes independent starts across native worker threads,
   while SciPy exposes scalar `minimize` and a user loops over starts.

Both effects should favor FrankenSciPy, but the old comparison did not give the
incumbent its strongest public derivative path. This experiment does.

## Whole job and public incumbent screen

The named job is a real multistart search: minimize the 6-D Rosenbrock function
from 128 deterministic starts in `[-2, 2]^6`, retain every solve result, and
select/materialize the best solution. Construction of the fixed starts is
outside timing. Both arms use BFGS, `gtol/tol=1e-8`, and `maxiter=500`.

Before effect timing, the harness screens these semantically valid public SciPy
1.17.1 arms on the exact fixed batch:

1. `minimize(rosen, ..., method="BFGS", jac=None)` (default forward
   difference);
2. the same numerical derivative with a persistent public `workers` map sized
   to the process affinity;
3. `jac=rosen_der` (separate public analytic objective and gradient);
4. `jac=True` with a public fused callback returning `(rosen(x),
   rosen_der(x))`.

The lowest screen median is frozen as the incumbent before paired rounds.
Screen timings are disclosed but are never pooled into the effect. FrankenSciPy
also receives its public analytic Rosenbrock gradient, so a missing derivative
is not used to manufacture the ratio.

The scientific gate runs before timing and fails closed unless:

- both arms return 128 finite result records;
- both best objective values are at most `1e-12`;
- both best points are within `1e-4` max absolute error of the all-ones global
  minimizer;
- the FrankenSciPy count with objective at most `1e-8` is at least 95% of the
  incumbent count; and
- the FrankenSciPy success count is at least 95% of the incumbent count.

These are quality-equivalence bounds rather than elementwise equality: the two
BFGS implementations can legitimately take different paths and expose
different success flags near a tolerance boundary. All success/global counts,
best solutions, iterations, function evaluations, and gradient evaluations are
reported so speed from giving up early remains visible.

## Predictions and falsifiers

**P1 -- strongest SciPy arm.** Predict the fused `(f, g)` arm or the separate
analytic-gradient arm wins the incumbent screen. Falsified if either numerical
finite-difference arm is fastest.

**P2 -- old magnitude collapses.** Predict the paired SciPy/FrankenSciPy 95%
CI has an upper endpoint below `137.5x`, at least a twofold collapse from the
lower historical headline after derivative parity and strict whole-job gating.
Report the ratio whichever way it falls.

**P3 -- durable whole-job win survives.** Predict the paired
SciPy/FrankenSciPy 95% CI lies wholly above `3x`. This is the campaign claim:
a durable real-job win, not preservation of a kernel-sized headline.

**P4 -- two-factor mechanism.** A same-invocation serial FrankenSciPy loop over
the identical starts is measured as telemetry. Predict:

- compiled solver/callback control, `SciPy / Franken-serial`, exceeds `2x`;
- native batching, `Franken-serial / minimize_many`, exceeds `4x`; and
- the product of the two median factors agrees with the primary paired median
  within `1.5x`.

The first factor includes solver implementation differences and therefore is
not labeled pure callback tax. If either lower bound fails, that part of the
mechanism is retracted even if the primary job wins.

**P5 -- observed execution.** On the planned 32-CPU affinity, predict 32
concurrent FrankenSciPy solve workers are directly observed during an untimed
job probe. If an analytic SciPy arm wins the screen, predict one active SciPy
task; if the numerical-workers arm wins, report the observed task/process count
without substituting the requested count.

## Measurement and decision gate

The primary experiment uses 15 balanced, interleaved paired rounds, plus 15
independent FrankenSciPy/FrankenSciPy and 15 independent
SciPy/SciPy A/A ratios. The effect is the median of per-round
`SciPy / FrankenSciPy` ratios. A deterministic 10,000-resample bootstrap
produces the 95% median interval.

A WIN or LOSS is decided only if all of these hold:

1. the effect CI excludes one in the claimed direction;
2. the point-effect distance from one exceeds twice the widest A/A bootstrap
   half-width;
3. the nearer effect-CI endpoint's distance from one exceeds twice the widest
   A/A CI endpoint distance from one; and
4. both A/A medians are in `[0.98, 1.02]`.

Whether an A/A CI straddles one is recorded as telemetry and is not a veto.
Ratio CV is provenance only. Otherwise the result is `UNDECIDABLE`, regardless
of the raw ratio.

The harness must additionally fail closed without live SciPy 1.17.1, a
performance governor on every affinity CPU, a fresh numeric Agent Mail
`trj` booking claim, exact source/build provenance, or the pre-timing quality
gate. It self-reports the running ELF SHA-256 and SciPy engine SHA-256 from
inside the processes, records host/topology/runtime ISA, and reports observed
tasks rather than requested threads. The executable must be built through
`rch exec --base <exact-commit> --clean-overlay --no-overlay` into the one
stable repository target directory, `/data/tmp/cargo-target`.

## Chooser boundary

If P3 and the corrected gate pass, choose FrankenSciPy `minimize_many` for this
exact 128-start analytic-gradient Rosenbrock multistart job at the measured
affinity. If the result is below 3x, loses, or is undecidable, publish that
boundary and retain SciPy as the default chooser; do not fall back to the old
271--275x number.

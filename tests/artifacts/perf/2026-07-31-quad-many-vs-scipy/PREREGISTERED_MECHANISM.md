# PRE-REGISTERED: `quad_many` transform sweep versus live SciPy

Written and committed before constructing the benchmark harness or running any
timed candidate/incumbent probe. Date: 2026-07-31. Bead:
`frankenscipy-afz21`.

## Existing claim and evidential gap

`docs/perf_ledger_cc.md` reports `quad_many` as 14.5--61.1x faster than a
Python loop over `scipy.integrate.quad` for 500 and 2,000 independent
integrals. The historical arm exposed every adaptive node to a scalar Python
callback and ran the parameter values serially. It was not paired in one
invocation, did not screen vector-valued or persistent-pool public SciPy
routes, and lacked independent A/A controls, executed-ELF identity, actual
worker observations, and the corrected bootstrap-median gate. The old
magnitude is routing evidence, not a current incumbent claim.

The registered mechanism is that FrankenSciPy inlines the integrand in Rust
and distributes independent adaptive solves across scoped native threads,
while the historical SciPy arm paid a Python callback and loop dispatch for
every parameter value. Public SciPy 1.17.1 can remove most of that tax in two
ways: one `quad_vec` or `cubature` call can integrate the whole parameter
vector through NumPy, and a persistent process pool can map independent
public `quad` calls. Pool construction remains outside timing, but mapping,
IPC, callbacks, adaptive integration, result collection, and whole-job
summaries remain inside. This experiment predicts that admitting those
public deployment patterns removes most of the old headline.

## Whole job and fixed transform study

The workload is a recognizable damped-frequency-response sweep. For 2,048
deterministic `(p, w)` pairs with `p` in `[2, 50]` and `w` in `[1, 35]`,
compute

```text
I(p, w) = integral_0^1 exp(-p*x) * cos(w*x) dx
        = (p + exp(-p) * (-p*cos(w) + w*sin(w))) / (p^2 + w^2).
```

Parameters come from a fixed 64-bit LCG (`state=7`, multiplier
`6364136223846793005`, increment `1`). Both implementations use
`epsabs=epsrel=1.49e-8` over `[0,1]`.

A timed arm must materialize all 2,048 integral values, all available error,
evaluation, convergence, and status records, maximum absolute and scaled
error against the closed form, response and error population summaries, the
maximum-absolute-response member, and a stable checksum. Parameter and
closed-form construction and persistent-pool construction are outside timing.
Adaptive integration, callback execution, pool mapping/IPC, output
collection, and every summary are inside. This is the whole transform-study
job, not a quadrature kernel.

## Strongest valid public SciPy screen

Before effect timing, the harness screens these public SciPy 1.17.1 routes on
the exact fixed study:

1. a scalar Python loop over `scipy.integrate.quad`;
2. the same independent calls through a persistent affinity-sized thread
   pool;
3. the same independent calls through a persistent affinity-sized fork
   process pool;
4. one `scipy.integrate.quad_vec` call whose NumPy callback returns all 2,048
   parameter responses;
5. `quad_vec` with a persistent public map-like process-pool workers callable;
6. one vector-output `scipy.integrate.cubature(..., rule="gk21")`; and
7. that `cubature` route with the same tracked persistent workers callable.

The lowest five-round median among scientifically eligible arms is frozen as
the incumbent before paired rounds. Screen samples are disclosed and never
pooled into the primary effect. Pools are constructed and warmed before the
screen. Thread arms report active OS tasks. Process-backed arms report the
distinct worker PIDs that actually returned work through the tracked map,
never configured capacity.

Scientific admission fails closed unless FrankenSciPy and the selected SciPy
arm:

- materialize exactly 2,048 finite integral and finite non-negative error
  records;
- report convergence/success for every member represented by the API;
- have maximum scaled closed-form error at most four, where scaled error is
  `abs(observed-exact)/(epsabs + epsrel*abs(exact))`;
- disagree by at most four on the same scaled-error denominator for every
  integral;
- select the same unique maximum-absolute-response parameter; and
- return positive evaluation counts wherever the public API exposes them.

An arm failing any condition is disclosed and excluded before its screen
timer. A protocol or identity failure aborts. At least one valid public SciPy
arm and the cross-implementation gate are mandatory. Error estimates are
diagnostics rather than equality targets because the public algorithms use
different adaptive rules.

## Predictions and falsifiers

**P1 -- strongest SciPy arm.** Predict single-worker vectorized `quad_vec` is
the fastest valid public arm. Falsified if scalar, thread, process,
process-backed `quad_vec`, or either `cubature` arm wins.

**P2 -- old magnitude collapses.** Predict the paired
`SciPy / FrankenSciPy` bootstrap-median 95% CI has an upper endpoint below
`12.2x`, at least a fivefold collapse from the old 61.1x headline. Report the
ratio whichever way it falls.

**P3 -- no durable 3x survives.** Predict the same CI has an upper endpoint
below `3x`. FrankenSciPy may reach parity, retain a smaller compiled advantage,
or lose to vectorized SciPy. Falsified if the CI reaches or exceeds three.

**P4 -- removed callback/serial tax is visible.** Predict the scalar
`quad`-loop screen median divided by the selected public-arm median is at
least `8x`. Falsified if it is below eight or the scalar arm is
scientifically ineligible.

**P5 -- observed execution.** On the planned 32-CPU affinity, predict 32
FrankenSciPy solve workers are directly observed. If a process-backed arm
wins, predict at least 16 distinct SciPy worker PIDs return work; if the
single-worker vector arm wins, predict exactly one active SciPy task. Actual
observations are reported even when a prediction fails.

## Measurement and corrected decision gate

The primary experiment uses 15 balanced, interleaved paired rounds with three
complete transform studies per timed sample, plus 15 independent
FrankenSciPy/FrankenSciPy and 15 independent SciPy/SciPy A/A ratios. Effect
and null pair order alternate and pair-group order rotates. The effect is the
median per-round `SciPy / FrankenSciPy` ratio. A deterministic
10,000-resample bootstrap produces the 95% median interval.

A WIN or LOSS is decided only if all of these hold:

1. the effect CI excludes one in the claimed direction;
2. the point-effect distance from one exceeds twice the widest A/A bootstrap
   half-width;
3. the nearer effect-CI endpoint distance from one exceeds twice the widest
   A/A CI endpoint distance from one; and
4. both A/A medians are in `[0.98,1.02]`.

Whether either A/A CI straddles one is telemetry only and is not a veto.
Ratio CV is provenance only. Anything else is `NOT DECIDED`.

The harness additionally fails closed without live SciPy 1.17.1, a fresh
numeric exclusive `trj` booking claim, a performance governor on every
affinity CPU, host-wide quiescence, and exact source/build provenance. It
self-reports from inside the processes the running FrankenSciPy ELF SHA-256
and aggregate SciPy source-engine SHA-256, plus host/boot identity, affinity,
runtime ISA, actual observed tasks/PIDs, inputs, and pool provenance. The ELF
must be built by
`rch exec --base <exact-commit> --clean-overlay --no-overlay` while reusing
only `/data/tmp/cargo-target`.

## Chooser boundary

Choose FrankenSciPy `quad_many` for this exact 2,048-member transform study
only if every gate passes and the corrected CI lies wholly above `3x`.
Otherwise choose the measured fastest valid SciPy arm for speed, subject to
deployment and API-fit constraints. In every outcome retire the historical
14.5--61.1x scalar-loop magnitude for this workload.

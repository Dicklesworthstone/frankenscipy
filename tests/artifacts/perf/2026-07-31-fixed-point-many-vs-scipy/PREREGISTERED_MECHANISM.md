# PRE-REGISTERED: Colebrook pipe-network report versus live SciPy

Written and committed before constructing the benchmark harness or running any
timed candidate/incumbent probe. Date: 2026-07-31. Bead:
`frankenscipy-5e4xq`.

Harness-construction clarification, committed before any timed probe: the
cross-language payload appends the five deterministic hydraulic invariants
listed below so both engines consume identical values. The raw fixture,
whole-job work, public arms, predictions, falsifiers, and gates are unchanged.

## Existing claim and evidential gap

`docs/NEGATIVE_EVIDENCE.md` reports `fixed_point_many` as 1,920x faster
than a Python loop over scalar `scipy.optimize.fixed_point` calls on a
Babylonian-square-root fixture. That comparison established the cost of
repeated scalar Python callbacks, but it did not measure the strongest public
incumbent or a recognizable whole job. Untimed signature, source, and
docstring inspection of live SciPy 1.17.1 shows that public
`scipy.optimize.fixed_point` explicitly supports array-valued `x0` and
vectorized parameters; its own documentation demonstrates a multi-element
call.

The historical comparison was not paired in one invocation and lacked a
public-arm screen, independent A/A controls, executed-ELF identity, actual
worker observations, and the corrected bootstrap-median gate. Its 1,920x
magnitude is routing evidence, not a current incumbent claim.

The registered mechanism has two parts. First, SciPy's public array route
removes almost all of the per-pipe Python call tax that dominated the old
number. Second, its array helper evaluates and materializes several complete
NumPy arrays for every Steffensen step and continues the whole batch until the
globally slowest element satisfies the relative-error test. FrankenSciPy runs
independent inlined scalar fixed-point loops, stops each pipe independently,
and distributes work-capped chunks over native threads without whole-batch
temporaries. Persistent pools of array-valued SciPy calls may recover the
incumbent's parallelism and therefore must be screened before attributing any
remaining difference to this mechanism.

## Whole job and fixed pipe network

The workload is a recognizable turbulent pipe-network hydraulic report for
65,536 independent pipe segments. Each segment solves the Colebrook--White
fixed point for the Darcy friction factor,

`f = 1 / (-2*log10(relative_roughness/3.7 + 2.51/(Re*sqrt(f))))^2`,

then materializes the Darcy--Weisbach pressure loss
`delta_p = f*(length/diameter)*(rho*velocity^2/2)`.

The fixed water properties are density `rho = 1000 kg/m^3` and dynamic
viscosity `mu = 1e-3 Pa*s`. Each row contains diameter, velocity, length, and
relative roughness:

- diameter in `[0.05, 1.0]` metres;
- velocity in `[0.1, 5.0]` metres/second;
- length in `[20, 2000]` metres; and
- relative roughness in `[1e-6, 0.03]`.

Thus `Re = rho*velocity*diameter/mu` is always at least 5,000, keeping the
entire book in the turbulent regime where Colebrook is applicable. Inputs
come from one fixed 64-bit LCG with initial state `0x243f6a8885a308d3`,
multiplier `6364136223846793005`, and increment
`1442695040888963407`. The exact little-endian fixture bytes are sent to the
live Python process before any pool is forked or timer starts. The payload
contains the four raw columns plus deterministic precomputed Reynolds number,
dynamic pressure, length/diameter, density, and viscosity values consumed by
both engines. Both engines must report the same fixture SHA-256. Fixture
generation, invariant precomputation, process and thread-pool construction,
and warmup are outside timing.

Every timed arm must:

1. solve all 65,536 friction factors from shared initial guess `0.02` with
   public Steffensen/Aitken `del2`, tolerance `1e-10`, and at most 500
   iterations;
2. materialize every factor and convergence classification;
3. recompute every Colebrook equation residual and Darcy pressure loss; and
4. compute finite/converged counts, residual p50/p95/p99/max, pressure-loss
   p50/p95/p99/max/mean, friction-factor mean/min/max, eight fixed friction
   bands, three pressure-severity counts, extrema indices, and a stable
   checksum over every factor, pressure loss, and summary.

The solve, public API dispatch, pool mapping and IPC, collection, equation
residuals, pressure-loss calculations, ranking, banding, severity
classification, and summary construction are inside timing. This is the
complete hydraulic report, not one logarithm or one fixed-point step.

## Strongest valid public SciPy screen

Before primary timing, the harness screens these public SciPy 1.17.1
deployments on the exact network:

1. a scalar Python loop over public `scipy.optimize.fixed_point`;
2. one public array-valued `scipy.optimize.fixed_point` call;
3. a persistent affinity-sized thread pool mapping public scalar calls;
4. a persistent affinity-sized fork process pool mapping public scalar calls;
5. a persistent affinity-sized thread pool mapping array-valued fixed-point
   solves over contiguous chunks; and
6. a persistent affinity-sized fork process pool mapping array-valued
   fixed-point solves over contiguous chunks.

All arms use public `method="del2"`, `xtol=1e-10`, and `maxiter=500`.
Pools are created after the immutable fixture and public callables exist and
before timing. Process jobs send contiguous index ranges rather than copying
the pipe book. Dispatch, mapping, result transfer, concatenation, and the
complete report remain timed. The lowest five-round median among
scientifically eligible arms is frozen as the incumbent before paired rounds.
Screen samples are disclosed and never pooled into the primary effect.

Thread-backed arms report distinct native thread IDs observed at public-call
sites. Process-backed arms report distinct worker PIDs that actually return
work. Requested pool capacity is never substituted for an observation.

Scientific admission fails closed unless an arm:

- materializes exactly 65,536 finite friction factors and convergence flags;
- reports every pipe converged with friction factor in `[0.005,0.10]`;
- has maximum absolute Colebrook equation residual at most `1e-8`;
- materializes 65,536 finite nonnegative pressure losses;
- agrees with the FrankenSciPy friction factor for every pipe within
  `1e-9 + 1e-8*abs(reference)`;
- produces identical friction-band and pressure-severity counts and the same
  extrema indices outside tied/noise-floor cases; and
- reports the registered fixture SHA-256.

An arm failing any condition is disclosed and excluded before its screen
timer. A protocol, identity, or fixture-hash failure aborts. At least one
valid public SciPy arm and the cross-implementation gate are mandatory.

## Predictions and falsifiers

**P1 -- strongest SciPy arm.** Predict the persistent fork pool of contiguous
array-valued fixed-point chunks is the fastest valid public arm because it
combines the incumbent's documented array route with whole-job CPU
parallelism. Falsified if any other eligible arm wins the screen.

**P2 -- old magnitude collapses.** Predict the paired
`SciPy / FrankenSciPy` bootstrap-median 95% CI has an upper endpoint below
`192x`, at least a tenfold collapse from the published 1,920x claim. Report
the ratio whichever way it falls.

**P3 -- removed scalar-dispatch tax is visible.** Predict the eligible scalar
loop screen median divided by the fastest eligible array-valued arm median is
at least `20x`. Falsified if it is below twenty or the scalar arm is
scientifically ineligible.

**P4 -- a durable whole-job win survives.** Predict the headline effect CI
lies wholly above `3x`. Repeated array materialization and global batch
progress should leave FrankenSciPy's independent native loops with a
campaign-grade advantage after scalar Python dispatch is removed. Falsified
if the lower CI endpoint is at or below three; report parity or a loss
honestly if that is what the gate finds.

**P5 -- observed execution.** On the planned 32-CPU affinity, predict exactly
32 distinct FrankenSciPy worker tasks are directly observed. If a
process-backed SciPy arm wins, predict at least 16 distinct returned worker
PIDs; if a thread-backed arm wins, predict more than one distinct call-site
thread; if the single-array arm wins, predict exactly one call-site task.
Actual observations are reported even when a prediction fails.

## Measurement and corrected decision gate

The primary experiment uses 15 balanced, interleaved paired rounds with three
complete reports per timed sample, plus 15 independent
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
and aggregate SciPy fixed-point/NumPy-log engine source-or-extension SHA-256,
plus host/boot identity, affinity, runtime ISA, actual observed tasks/PIDs,
fixture identity, and pool provenance. The ELF must be built by
`rch exec --base <exact-commit> --clean-overlay --no-overlay` while reusing
only `/data/tmp/cargo-target`.

## Chooser boundary

Choose FrankenSciPy `fixed_point_many` for this exact 65,536-pipe
Colebrook/Darcy report only if every corrected gate passes and the effect CI
lies wholly above `3x`. Otherwise choose the measured fastest valid SciPy
public arm for speed, subject to deployment and API-fit constraints. In every
outcome retire the historical 1,920x scalar-loop magnitude for this workload.

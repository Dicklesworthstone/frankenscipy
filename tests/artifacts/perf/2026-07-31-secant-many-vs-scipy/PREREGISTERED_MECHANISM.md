# PRE-REGISTERED: derivative-free implied-volatility calibration versus live SciPy

Written and committed before constructing the benchmark harness or running any
timed candidate/incumbent probe. Date: 2026-07-31. Bead:
`frankenscipy-dw6du`.

Pre-primary protocol correction: the first explicitly non-evidence smoke
showed that the extrema-index clause needed a numeric definition for secant
stopping noise. FrankenSciPy's maximum repricing residual was
`9.999e-11` and SciPy's was `9.948e-14`; both were far below the registered
`5e-8` scientific limit, but different public second-point and stopping
policies made different rows the nominal maximum. For secant mode, a maximum
price residual at or below `1e-9` is now an extrema-index noise-floor case:
both indices and values remain disclosed, while equality of the meaningless
argmax is not required. The volatility-index floor remains `1e-10`. No
primary effect was measured before this correction. The option book, timed
work, public arms, predictions, scientific error limits, corrected null gate,
and chooser are unchanged.

## Existing claim and evidential gap

`docs/NEGATIVE_EVIDENCE.md` reports `secant_many` as 536x faster than a
Python loop over scalar `scipy.optimize.newton` calls without `fprime`. That
comparison established the cost of repeated scalar Python dispatch, but it did
not measure the strongest public incumbent. Untimed signature and source
inspection of live SciPy 1.17.1 shows that public
`scipy.optimize.newton(func, x0, fprime=None)` accepts an array-valued `x0`
and a vectorized objective, returning one root per array element from a single
public call.

The historical comparison was not paired in one invocation and lacked a
public-arm screen, independent A/A controls, executed-ELF identity, actual
worker observations, and the corrected bootstrap-median gate. Its 536x
magnitude is routing evidence, not a current incumbent claim.

The registered mechanism has two parts. First, SciPy's public array route
removes almost all of the per-contract Python call tax that dominated the old
number. Second, derivative-free calibration takes more objective evaluations
than Newton-with-derivative calibration, and SciPy's array route repeatedly
evaluates and materializes whole-array NumPy/SciPy expressions while advancing
active elements until the globally slowest element meets its step tolerance.
FrankenSciPy streams each contract through an independent scalar secant loop,
stops it independently, and distributes work-capped chunks over native
threads without whole-batch temporaries. Persistent pools of array-valued
SciPy calls may recover the incumbent's parallelism and therefore must be
screened before attributing any remaining difference to this mechanism.

Untimed source inspection also establishes a public-API asymmetry that the
benchmark must preserve rather than hide: scalar SciPy and FrankenSciPy use
the same positive-`x0` default second point,
`x0 * (1 + 1e-4) + 1e-4`, while SciPy's public array route does not accept
`x1` and initializes each element with its documented implementation's
`eps**0.33` perturbation. Final roots, repricing, convergence, and report
contents must agree; iteration traces and second points need not.

## Whole job and fixed option book

The workload is a recognizable derivative-free implied-volatility calibration
and risk-report job for 65,536 European call quotes. A user selects secant
because no analytic vega callback is available. Each row contains spot,
strike, maturity, risk-free rate, a target call price, and a known generating
volatility used only by the quality gate. The surface spans liquid,
well-conditioned contracts:

- spot in `[80, 120]`;
- log-moneyness in `[-0.15, 0.15]`;
- maturity in `[0.25, 2.0]` years;
- rate in `[0.005, 0.05]`; and
- generating volatility in `[0.12, 0.60]`.

Inputs come from one fixed 64-bit LCG with initial state
`0x9e3779b97f4a7c15`, multiplier `6364136223846793005`, and increment
`1442695040888963407`. Target prices are generated once with the registered
Black--Scholes call formula. To isolate the solver deployment rather than
cross-language libm drift, the exact little-endian payload appends the
precomputed square root of maturity, log-forward moneyness, and discounted
strike used by both engines. The exact payload is sent to the live Python
process before any pool is forked or timer starts. Both engines must report
the same fixture SHA-256. Quote generation, invariant precomputation, process
and thread-pool construction, and warmup are outside timing.

Every timed arm must:

1. calibrate all 65,536 volatilities from shared initial guess `0.30` with
   public derivative-free secant, absolute step tolerance `1e-10`, zero
   relative tolerance where the API exposes it, and at most 50 iterations;
2. materialize every volatility and convergence classification;
3. reprice every contract and compute finite/converged counts, absolute price
   residual p50/p95/p99/max, generating-volatility error p50/p95/p99/max,
   calibrated-volatility mean/min/max, eight fixed volatility-band counts, and
   a stable checksum over every root and summary.

The solve, public API dispatch, pool mapping and IPC, collection, repricing,
ranking, banding, and summary construction are inside timing. This is the
complete calibration report, not one Black--Scholes evaluation.

## Strongest valid public SciPy screen

Before primary timing, the harness screens these public SciPy 1.17.1
deployments on the exact option book:

1. a scalar Python loop over public derivative-free
   `scipy.optimize.newton`;
2. one public array-valued derivative-free `scipy.optimize.newton` call;
3. a persistent affinity-sized thread pool mapping public scalar calls;
4. a persistent affinity-sized fork process pool mapping public scalar calls;
5. a persistent affinity-sized thread pool mapping array-valued secant over
   contiguous chunks; and
6. a persistent affinity-sized fork process pool mapping array-valued secant
   over contiguous chunks.

Pools are created after the immutable fixture and public callables exist and
before timing. Process jobs send contiguous index ranges rather than copying
the option book. Dispatch, mapping, result transfer, concatenation, and the
complete report remain timed. The lowest five-round median among
scientifically eligible arms is frozen as the incumbent before paired rounds.
Screen samples are disclosed and never pooled into the primary effect.

Thread-backed arms report distinct native thread IDs observed at public-call
sites. Process-backed arms report distinct worker PIDs that actually return
work. Requested pool capacity is never substituted for an observation.

Scientific admission fails closed unless an arm:

- materializes exactly 65,536 finite volatilities and convergence flags;
- reports every row converged with no zero secant slope;
- has maximum absolute repricing residual at most `5e-8`;
- has maximum absolute generating-volatility error at most `5e-7`;
- agrees with the FrankenSciPy volatility for every row within
  `1e-8 + 1e-8*abs(reference)`;
- produces identical volatility-band counts and the same extrema indices
  outside tied/noise-floor cases (secant price-index floor `1e-9`,
  volatility-index floor `1e-10`); and
- reports the registered fixture SHA-256.

An arm failing any condition is disclosed and excluded before its screen
timer. A protocol, identity, or fixture-hash failure aborts. At least one
valid public SciPy arm and the cross-implementation gate are mandatory.

## Predictions and falsifiers

**P1 -- strongest SciPy arm.** Predict the persistent fork pool of contiguous
array-valued secant chunks is the fastest valid public arm because it combines
the incumbent's vector route with whole-job CPU parallelism. Falsified if any
other eligible arm wins the screen.

**P2 -- old magnitude collapses.** Predict the paired
`SciPy / FrankenSciPy` bootstrap-median 95% CI has an upper endpoint below
`53.6x`, at least a tenfold collapse from the published 536x claim. Report the
ratio whichever way it falls.

**P3 -- removed scalar-dispatch tax is visible.** Predict the eligible scalar
loop screen median divided by the fastest eligible array-valued arm median is
at least `20x`. Falsified if it is below twenty or the scalar arm is
scientifically ineligible.

**P4 -- a durable whole-job win survives.** Predict the headline effect CI
lies wholly above `3x`. Derivative-free iteration should amplify SciPy's
whole-array temporary and global-progress costs enough for FrankenSciPy's
independent native loops to retain a campaign-grade advantage after the
scalar Python tax is removed. Falsified if the lower CI endpoint is at or
below three; report parity or a loss honestly if that is what the gate finds.

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
and aggregate SciPy secant/normal-CDF engine source-or-extension SHA-256, plus
host/boot identity, affinity, runtime ISA, actual observed tasks/PIDs, fixture
identity, and pool provenance. The ELF must be built by
`rch exec --base <exact-commit> --clean-overlay --no-overlay` while reusing
only `/data/tmp/cargo-target`.

## Chooser boundary

Choose FrankenSciPy `secant_many` for this exact 65,536-contract
derivative-free implied-volatility report only if every corrected gate passes
and the effect CI lies wholly above `3x`. Otherwise choose the measured
fastest valid SciPy public arm for speed, subject to deployment and API-fit
constraints. In every outcome retire the historical 536x scalar-loop
magnitude for this workload.

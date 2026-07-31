# PRE-REGISTERED: implied-volatility calibration versus live SciPy

Written and committed before constructing the benchmark harness or running any
timed candidate/incumbent probe. Date: 2026-07-31. Bead:
`frankenscipy-9nzg9`.

Pre-primary protocol correction: the first non-evidence smoke showed that
independently recomputing `sqrt(T)`, log-forward, and discounted strike with
Rust libm versus NumPy created cross-engine root drift despite identical six
quote inputs. The exact payload now appends those three deterministic
precomputed invariants per row, which both engines consume outside timing.
The option book, timed work, public arms, predictions, falsifiers, and gates
are unchanged. No primary effect was measured before this correction.

## Existing claim and evidential gap

`docs/NEGATIVE_EVIDENCE.md` reports `newton_many` as 495--986x faster
than a Python loop over `scipy.optimize.newton`. That comparison established
the cost of repeated scalar Python dispatch, but it did not measure the
strongest public incumbent. Untimed signature and source inspection of live
SciPy 1.17.1 shows that public `scipy.optimize.newton` accepts an array-valued
`x0`, a vectorized objective, and a vectorized derivative. It returns one root
per array element from a single public call.

The historical comparison was not paired in one invocation and lacked a
public-arm screen, independent A/A controls, executed-ELF identity, actual
worker observations, and the corrected bootstrap-median gate. Its
495--986x magnitudes are routing evidence, not a current incumbent claim.

The registered mechanism has two parts. First, SciPy's public array route
removes almost all of the per-contract Python call tax that dominated the old
number. Second, the array route still evaluates and materializes whole-array
NumPy/SciPy expressions at every Newton iteration and advances every
nonzero-derivative element until the globally slowest element meets the step
tolerance. FrankenSciPy streams each contract through an independent scalar
Newton loop, stops it independently, and distributes work-capped chunks over
native threads without whole-batch temporaries. Persistent pools of
array-valued SciPy calls may recover the incumbent's parallelism and therefore
must be screened before attributing any remaining difference to this
mechanism.

## Whole job and fixed option book

The workload is a recognizable implied-volatility calibration and risk-report
job for 65,536 European call quotes. Each row contains spot, strike, maturity,
risk-free rate, a target call price, and a known generating volatility used
only by the quality gate. The surface spans liquid, well-conditioned contracts:

- spot in `[80, 120]`;
- log-moneyness in `[-0.15, 0.15]`;
- maturity in `[0.25, 2.0]` years;
- rate in `[0.005, 0.05]`; and
- generating volatility in `[0.12, 0.60]`.

Inputs come from one fixed 64-bit LCG with initial state
`0x9e3779b97f4a7c15`, multiplier `6364136223846793005`, and increment
`1442695040888963407`. Target prices are generated once with the registered
Black--Scholes call formula. The exact little-endian fixture bytes are sent to
the live Python process before any pool is forked or timer starts. Both engines
must report the same fixture SHA-256. Quote generation, invariant
precomputation, process/thread pool construction, and warmup are outside
timing.

Every timed arm must:

1. calibrate all 65,536 volatilities from shared initial guess `0.30` with
   public Newton plus the analytic Black--Scholes vega, absolute step tolerance
   `1e-10`, zero relative tolerance where the API exposes it, and at most 50
   iterations;
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

1. a scalar Python loop over public `scipy.optimize.newton`;
2. one public array-valued `scipy.optimize.newton` call;
3. a persistent affinity-sized thread pool mapping public scalar calls;
4. a persistent affinity-sized fork process pool mapping public scalar calls;
5. a persistent affinity-sized thread pool mapping array-valued Newton over
   contiguous chunks; and
6. a persistent affinity-sized fork process pool mapping array-valued Newton
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
- reports every row converged with no zero derivative;
- has maximum absolute repricing residual at most `5e-8`;
- has maximum absolute generating-volatility error at most `5e-7`;
- agrees with the FrankenSciPy volatility for every row within
  `1e-8 + 1e-8*abs(reference)`;
- produces identical volatility-band counts and the same extrema indices
  outside tied/noise-floor cases; and
- reports the registered fixture SHA-256.

An arm failing any condition is disclosed and excluded before its screen
timer. A protocol, identity, or fixture-hash failure aborts. At least one
valid public SciPy arm and the cross-implementation gate are mandatory.

## Predictions and falsifiers

**P1 -- strongest SciPy arm.** Predict the persistent fork pool of contiguous
array-valued Newton chunks is the fastest valid public arm because it combines
the incumbent's vector route with whole-job CPU parallelism. Falsified if any
other eligible arm wins the screen.

**P2 -- old magnitude collapses.** Predict the paired
`SciPy / FrankenSciPy` bootstrap-median 95% CI has an upper endpoint below
`49.5x`, at least a tenfold collapse from the smallest published 495x member
of the old claim. Report the ratio whichever way it falls.

**P3 -- removed scalar-dispatch tax is visible.** Predict the eligible scalar
loop screen median divided by the fastest eligible array-valued arm median is
at least `20x`. Falsified if it is below twenty or the scalar arm is
scientifically ineligible.

**P4 -- no durable 3x survives the strongest incumbent.** Predict the
headline effect CI has an upper endpoint below `3x`. FrankenSciPy may retain a
smaller streaming/early-stop advantage, reach parity, or lose. Falsified if
the CI reaches or exceeds three.

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
and aggregate SciPy Newton/normal-CDF engine source-or-extension SHA-256, plus
host/boot identity, affinity, runtime ISA, actual observed tasks/PIDs, fixture
identity, and pool provenance. The ELF must be built by
`rch exec --base <exact-commit> --clean-overlay --no-overlay` while reusing
only `/data/tmp/cargo-target`.

## Chooser boundary

Choose FrankenSciPy `newton_many` for this exact 65,536-contract
implied-volatility report only if every gate passes and the corrected CI lies
wholly above `3x`. Otherwise choose the measured fastest valid SciPy public
arm for speed, subject to deployment and API-fit constraints. In every outcome
retire the historical 495--986x scalar-loop magnitudes for this workload.

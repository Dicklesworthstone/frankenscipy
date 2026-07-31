# PRE-REGISTERED: batched normality-screening job versus live SciPy

Written and committed before constructing the benchmark harness or running any
timed candidate/incumbent probe. Date: 2026-07-31. Bead:
`frankenscipy-bh6hy`.

## Existing claim and evidential gap

`docs/NEGATIVE_EVIDENCE.md` reports `normaltest_many`,
`jarque_bera_many`, and `shapiro_many` as 2,267x, 1,730x, and 74x faster,
respectively, than Python loops over the corresponding SciPy functions for
3,000 datasets of 300 observations. The row says that SciPy has no batched
form. That premise is false for the live incumbent: untimed signature and
source inspection of SciPy 1.17.1 shows that all three public functions accept
an `axis` argument and return one result per axis slice.

The historical comparison was not paired in one invocation, did not screen
SciPy's public axis-vectorized or persistent-pool deployment patterns, and
lacked independent A/A controls, executed-ELF identity, actual worker
observations, and the corrected bootstrap-median gate. Its 300-observation
Jarque-Bera fixture was also below the `>2000` observation regime identified
in SciPy's public documentation for the asymptotic p-value. The old magnitudes
are routing evidence, not current incumbent claims.

The registered mechanism is that the old ratios mostly measured 9,000 Python
API calls and an outer Python loop. FrankenSciPy removes that dispatch and
parallelizes independent compiled tests. Live SciPy can remove the user loop
with three axis-vectorized calls; it can additionally distribute the
slice-oriented Shapiro-Wilk calls through a persistent process pool while
leaving vectorized moment tests in the parent. This experiment predicts that
the strongest valid public SciPy deployment removes most of the old headline.

## Whole job and fixed QC study

The workload is a recognizable many-channel normality screening report:
512 independent series, each with 4,096 observations. This sample length is
above the documented large-sample regime for Jarque-Bera and below the 5,000
observation boundary at which SciPy cautions that the Shapiro-Wilk p-value may
lose accuracy.

Inputs come from one fixed 64-bit LCG with initial state
`0xd1b54a32d192ed03`, multiplier `6364136223846793005`, and increment
`1442695040888963407`. Each base value is the ordered sum of twelve
high-53-bit uniforms minus six, avoiding cross-language transcendental input
generation. Rows cycle through four registered regimes:

1. the centered CLT base;
2. `1.75 * base + 0.25`, preserving a shifted/scaled near-normal series;
3. `abs(base)`, creating a half-normal alternative; and
4. the base with every 257th observation multiplied by seven, creating a
   contaminated-tail alternative.

The complete input is constructed outside timing and must have the same
SHA-256 in both engines.

A timed arm must run all three tests over all 512 series and materialize all
3,072 statistic/p-value outputs. It must also compute, inside timing:

- finite-output and valid-range counts for every test;
- per-test rejection counts at `alpha=0.05`;
- the union count rejected by at least one test;
- the minimum-p-value series and value for each test;
- per-test statistic and p-value population summaries; and
- a stable checksum over every output and summary.

Input construction and persistent-pool construction/warmup are outside
timing. Statistical work, public API dispatch, process mapping and IPC, result
collection, validation summaries, rankings, and checksum construction are
inside. This is the whole QC report, not an individual moment or sort kernel.

## Strongest valid public SciPy screen

Before effect timing, the harness screens these public SciPy 1.17.1 routes on
the exact fixed study:

1. a scalar Python row loop calling public `normaltest`, `jarque_bera`, and
   `shapiro`;
2. three public calls with `axis=1`;
3. a persistent affinity-sized thread pool mapping the complete three-test
   row job;
4. a persistent affinity-sized fork process pool mapping the complete
   three-test row job by shared row index;
5. axis-vectorized `normaltest` and `jarque_bera` plus a persistent thread
   pool for public scalar `shapiro`; and
6. axis-vectorized `normaltest` and `jarque_bera` plus a persistent fork
   process pool for public scalar `shapiro`.

The lowest five-round median among scientifically eligible arms is frozen as
the incumbent before paired rounds. Screen samples are disclosed and never
pooled into the primary effect. The complete input exists before the process
pool is forked, so process arms send row indices rather than copying the
16 MiB study on every job. Mapping, scheduling, public calls, collection, and
whole-report summaries remain timed.

Thread arms report the distinct native thread IDs observed at the public call
sites. Process-backed arms report the distinct worker PIDs that actually
return work. Configured or requested capacity is never substituted for an
observation.

Scientific admission fails closed unless every eligible arm:

- materializes exactly 512 finite statistics and 512 finite p-values for each
  of the three tests;
- keeps every p-value in `[0,1]`;
- agrees with live SciPy per output under
  `abs(got-ref) / (1e-10 + 1e-8*abs(ref)) <= 1`;
- produces the same rejection classification at `alpha=0.05`, except that a
  value within the comparison tolerance of the boundary is classified and
  reported as boundary-indeterminate in both engines;
- reports matching minimum-p-value series indices outside tied/boundary
  cases; and
- produces the same input SHA-256 and row-regime counts.

An arm failing any condition is disclosed and excluded before its screen
timer. A protocol, identity, or input-hash failure aborts. At least one valid
public SciPy arm and the cross-implementation gate are mandatory.

## Predictions and falsifiers

**P1 -- strongest SciPy arm.** Predict the hybrid of axis-vectorized
`normaltest`/`jarque_bera` plus persistent-process `shapiro` is the fastest
valid public arm. Falsified if the scalar loop, all-axis route, either
whole-row pool, or the thread hybrid wins.

**P2 -- old magnitude collapses.** Predict the paired
`SciPy / FrankenSciPy` bootstrap-median 95% CI has an upper endpoint below
`7.4x`, at least a tenfold collapse from even the smallest published 74x
member of the old claim. Report the ratio whichever way it falls.

**P3 -- removed user-loop tax is visible.** Predict the eligible scalar-loop
screen median divided by the selected public-arm median is at least `8x`.
Falsified if it is below eight or the scalar arm is scientifically
ineligible.

**P4 -- no durable 3x survives.** Predict the headline effect CI has an upper
endpoint below `3x`. FrankenSciPy may retain a smaller compiled/scheduling
advantage, reach parity, or lose. Falsified if the CI reaches or exceeds
three.

**P5 -- observed execution.** On the planned 32-CPU affinity, predict at least
16 distinct FrankenSciPy worker tasks are directly observed. If a
process-backed SciPy arm wins, predict at least 16 distinct returned worker
PIDs; if the all-axis arm wins, predict exactly one active SciPy call-site
task. Actual observations are reported even when a prediction fails.

## Measurement and corrected decision gate

The primary experiment uses 15 balanced, interleaved paired rounds with three
complete QC studies per timed sample, plus 15 independent
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
and aggregate SciPy normality-engine source/extension SHA-256, plus host/boot
identity, affinity, runtime ISA, actual observed tasks/PIDs, input identity,
and pool provenance. The ELF must be built by
`rch exec --base <exact-commit> --clean-overlay --no-overlay` while reusing
only `/data/tmp/cargo-target`.

## Chooser boundary

Choose FrankenSciPy's three `*_many` APIs for this exact 512-by-4,096
normality-screening report only if every gate passes and the corrected CI lies
wholly above `3x`. Otherwise choose the measured fastest valid SciPy public
arm for speed, subject to deployment and API-fit constraints. In every outcome
retire the historical 74--2,267x scalar-loop magnitudes for this workload.

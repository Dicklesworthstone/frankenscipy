# PRE-REGISTERED: `root_many` equilibrium study versus live SciPy

Written and committed before constructing the benchmark harness or running any
timed candidate/incumbent probe. Date: 2026-07-31. Bead:
`frankenscipy-40h1j`.

## Existing claim and evidential gap

`docs/perf_ledger_cc.md` reports `root_many` as 11.1--25.1x faster than a
serial Python loop over `scipy.optimize.root` for 500 and 2,000
three-equation solves. The row predates the campaign evidence contract: it has
no same-invocation pairing, independent A/A controls, executed-ELF identity,
observed worker counts, strongest-public-arm screen, or corrected
bootstrap-median gate. Its SciPy comparator also receives no parallel map or
analytic Jacobian. The old magnitude is routing evidence, not a current
competitive claim.

The registered mechanism is that FrankenSciPy inlines each residual in Rust
and distributes independent systems over scoped native workers. The historical
SciPy arm paid one Python callback boundary per MINPACK evaluation and solved
the entire sweep serially. A persistent process pool around public SciPy
`root`, especially with a public analytic Jacobian, removes the serial-batch
factor while preserving independent-root semantics. Process IPC and result
materialization remain inside the incumbent job timer. This experiment
predicts that admitting that public deployment pattern removes most of the old
headline.

## Whole job and fixed equilibrium study

The workload is a recognizable coupled steady-state parameter study. For
2,048 deterministic target equilibria `u=(u0,u1,u2)` with each component in
`[0.8,1.2]`, form forcing parameters

```text
p0 = u0^2 + u1 + u2
p1 = u0 + u1^2 + u2
p2 = u0 + u1 + u2^2
```

and recover the equilibrium `x` by solving

```text
x0^2 + x1 + x2 - p0 = 0
x0 + x1^2 + x2 - p1 = 0
x0 + x1 + x2^2 - p2 = 0.
```

The target roots come from a fixed 64-bit LCG (`state=5`, multiplier
`6364136223846793005`, increment `1`). Every solve starts from `(1,1,1)`.
Near this box the analytic Jacobian has diagonal `2*x_i` and unit
off-diagonals; the target branch is locally well-conditioned and represents a
small coupled equilibrium continuation sweep rather than a collection of
intentional solver failures.

A timed arm must materialize all 2,048 three-component roots, residual
vectors/norms, success and work counts, the worst residual system, component
population means and percentiles, maximum target-root error, and a stable
checksum. Input and persistent-pool construction are outside timing. Solver
execution, pool mapping/IPC, output collection, residual recomputation, and
all summaries are inside timing.

## Strongest valid public SciPy screen

Before effect timing, the harness screens these public SciPy 1.17.1 routes on
the exact fixed study:

1. scalar public `root(method="hybr")` with numerical Jacobians;
2. scalar public `root(method="hybr", jac=analytic_jac)`;
3. each of those calls through a persistent affinity-sized thread pool;
4. each through a persistent affinity-sized fork-process pool; and
5. one public joint `least_squares(method="trf", tr_solver="lsmr")` solve
   returning the same 2,048 independent root records through an analytic
   block-sparse Jacobian.

The lowest five-round median among scientifically eligible arms is frozen as
the incumbent before paired rounds. Screen samples are disclosed and never
pooled into the primary effect. Pools are constructed and warmed before the
screen. Thread arms report active OS tasks; process arms report the distinct
worker PIDs that actually returned results, never the configured capacity.
The public joint arm is eligible only if it preserves every independent root
and whole-job output contract.

Scientific admission fails closed unless both FrankenSciPy and the selected
SciPy arm:

- materialize exactly 2,048 finite three-component roots and residual records;
- report success for all 2,048 systems;
- have maximum residual infinity norm at most `1e-8`;
- recover every registered target root within `1e-6` infinity norm;
- disagree with each other by at most `1e-6` infinity norm per root; and
- identify the same worst-residual system, have both worst residuals at or
  below 1% of the locked `1e-8` residual ceiling, or have worst residuals
  within 1% of the larger value.

An arm failing any condition is disclosed and excluded before its screen
timer. A protocol/identity failure aborts. At least one valid public SciPy arm
and the cross-implementation gate are mandatory.

### Pre-effect amendment after the public-arm screen

This paragraph and the revised worst-system clause above were committed after
one non-evidence smoke screen but before any primary paired effect sample.
The original committed clause required the same worst-residual system or
worst residuals within 1% of the larger value. The smoke materialized
conforming roots from every arm, then stopped at that cross-quality clause:
FrankenSciPy's worst system was 1092 with residual
`9.61808410693265614e-11`; SciPy's was 711, and the absolute difference
between their worst residuals was `9.43365385808192514e-11`. Both are below
1% of the already locked `1e-8` scientific ceiling. At that numerical floor,
the identity of `argmax(residual)` is not a stable scientific observable.

The amended disjunction is fixed to the original tolerance rather than the
observed difference: differing indices are admitted only when both worst
residuals are at most `1e-10`, or when their residuals meet the original 1%
relative-agreement clause. The `1e-8` per-implementation residual ceiling,
`1e-6` target-root ceiling, `1e-6` cross-root ceiling, complete-output
contract, and all timing gates are unchanged.

For full disclosure, the smoke screen medians in seconds were numeric scalar
`0.080040629`, analytic scalar `0.075790705`, numeric thread `0.092145125`,
analytic thread `0.090062549`, numeric process `0.011056982`, analytic
process `0.011113078`, and joint sparse `0.016768715`. Thus the first smoke
placed the numeric process arm ahead of P1's predicted analytic process arm,
while P4's scalar/parallel screen ratio was `7.238922x`, below eight. The
smoke directly observed 14 FrankenSciPy worker tasks and 32 returning
process-worker PIDs. P2, P3, the primary effect, both A/A nulls, and the
chooser remain wholly unmeasured.

After implementing the amended gate, a second non-evidence smoke passed the
complete scientific contract. Its screen selected the analytic process arm
at `0.010748437` seconds over the numeric process arm at `0.014726596`
seconds, while the scalar/parallel ratio remained below eight at
`7.454897396x`. It again observed 14 FrankenSciPy worker tasks and 32
returning process-worker PIDs. Because the two process arms exchanged rank
across non-evidence screens, P1 is not assigned a campaign outcome here; its
registered outcome is determined by the fresh evidence invocation. P4 was
below its threshold in both smoke screens. No paired effect or null sample
was run.

## Predictions and falsifiers

**P1 -- strongest SciPy arm.** Predict the analytic-Jacobian persistent
process pool is the fastest valid public arm. Falsified if a scalar, thread,
numeric-process, or joint-sparse arm wins.

**P2 -- old magnitude collapses.** Predict the paired
`SciPy / FrankenSciPy` bootstrap-median 95% CI has an upper endpoint below
`5x`, at least a fivefold collapse from the old 25.1x headline. Report the
ratio whichever way it falls.

**P3 -- no durable 3x survives.** Predict the same CI has an upper endpoint
below `3x`. This is independent of direction: FrankenSciPy may retain a small
compiled/IPC advantage, reach parity, or lose. Falsified if the CI reaches or
exceeds three.

**P4 -- removed serial-batch tax is visible.** Predict the fastest conforming
process/joint arm is at least `8x` faster than scalar numerical `root` in the
screen medians. Falsified if that screen ratio is below eight or the scalar
arm is scientifically ineligible.

**P5 -- observed execution.** On the planned 32-CPU affinity, predict 32
FrankenSciPy solve workers are directly observed. If a process-pool arm wins,
predict at least 16 distinct SciPy worker PIDs return results; if a thread
pool wins, predict more than one active SciPy task. Actual observations are
reported even when either prediction fails.

## Measurement and corrected decision gate

The primary experiment uses 15 balanced, interleaved paired rounds with three
complete equilibrium studies per timed sample, plus 15 independent
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
runtime ISA, exact input hashes, and actual observed workers. Build only
through
`rch exec --base <exact-commit> --clean-overlay --no-overlay`, reusing
`/data/tmp/cargo-target`.

## Chooser statement committed in advance

For this exact 2,048-system equilibrium study, choose FrankenSciPy if the
corrected gate decides a FrankenSciPy win and choose the selected public SciPy
arm if it decides a loss. If undecidable, choose on deployment/API fit and
make no speed claim. Call the result a durable campaign win only if the
FrankenSciPy CI lower endpoint exceeds `3x`; otherwise retire the old
11.1--25.1x magnitude even if FrankenSciPy remains modestly faster.

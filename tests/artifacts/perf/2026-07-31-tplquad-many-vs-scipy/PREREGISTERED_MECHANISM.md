# PRE-REGISTERED: `tplquad_many` versus vectorized live SciPy

Written and committed before constructing the benchmark harness or running any
timed candidate/incumbent probe. Date: 2026-07-31. Bead:
`frankenscipy-zkel9`.

## Existing claim and evidential gap

`docs/perf_ledger_cc.md` reports `tplquad_many` as 83--159x faster than a
Python loop over `scipy.integrate.tplquad` for a parameter sweep of smooth
three-dimensional Gaussian integrals. The historical comparison exposed every
adaptive node of every parameter value to a scalar Python callback and then ran
the parameter values serially. It was not a same-invocation paired experiment,
did not screen SciPy's public vector-valued integration APIs, and lacked
independent A/A controls, executable identity, observed task counts, and the
corrected median-CI gate. The old values are routing evidence, not a current
incumbent claim.

The old mechanism was attributed to compiled callbacks plus native parallelism:
FrankenSciPy distributes independent parameter values across Rust threads and
inlines the integrand, whereas the historical SciPy arm paid three nested
levels of scalar Python callback tax for every parameter value. SciPy 1.17.1
exposes public `quad_vec` and `cubature` routes that can evaluate all parameter
values in one NumPy callback. That removes the exact per-parameter callback-tax
mechanism. This experiment therefore predicts a collapse rather than assuming
the old win survives.

## Whole job and public incumbent screen

The named job is a 100-member three-dimensional parameter sweep:

`I(p) = integral_0^1 integral_0^1 integral_0^1 exp(-p*(x^2+y^2+z^2)) dz dy dx`

for 100 deterministic evenly spaced `p` values spanning `[2, 15]`. This is the
same unit-cube Gaussian family and parameter regime used to establish the
historical claim, including its `p=5` numerical anchor. Construction of the
parameter vector is outside timing. A timed arm must compute and materialize
all 100 integral values, all available error/convergence records, a stable
checksum, and the `(p, I(p))` pair with the maximum integral. Thus the
benchmark is the whole parameter-study job, not one quadrature kernel.

Both implementations use public absolute and relative tolerances `1.49e-8`,
unit-cube bounds, and no cached result. Before effect timing, the harness
screens these semantically valid public SciPy 1.17.1 arms on the exact fixed
job:

1. a scalar Python loop over `scipy.integrate.tplquad`;
2. three nested `scipy.integrate.quad_vec` calls whose innermost NumPy callback
   returns all 100 parameter outputs together;
3. `scipy.integrate.cubature(..., rule="gk21")` with one vectorized callback;
4. `scipy.integrate.cubature(..., rule="genz-malik")` with one vectorized
   callback; and
5. the valid `cubature` rule above with the lowest single-worker screen median,
   using a persistent public workers map sized to the process affinity.

The lowest five-round screen median is frozen as the incumbent before paired
rounds. Screen timings are disclosed and are not pooled into the effect.
FrankenSciPy uses `tplquad_many` on the identical function and parameter
vector.

The pre-timing scientific gate fails closed unless:

- both arms materialize exactly 100 finite integral values;
- FrankenSciPy reports 100/100 converged results and the selected SciPy result
  reports success;
- every available error estimate is finite and non-negative;
- both arms have maximum scaled error at most four against the closed-form
  reference
  `(sqrt(pi)/(2*sqrt(p))*erf(sqrt(p)))^3`, evaluated outside timing; and
- the selected maximum occurs at the same parameter and its two integral
  values differ by no more than the sum of their requested absolute/relative
  tolerances.

Scaled error is
`abs(observed-reference)/(epsabs + epsrel*abs(reference))`. The harness reports
maximum absolute and scaled errors for every screened arm. Error estimates are
diagnostics rather than equality targets because the public algorithms use
different adaptive rules.

## Predictions and falsifiers

**P1 -- strongest SciPy arm.** Predict single-worker vectorized
`cubature(..., rule="genz-malik")` wins the public SciPy screen. Falsified if
scalar `tplquad`, nested `quad_vec`, `gk21`, or the workers-map arm is fastest.

**P2 -- callback-tax headline collapses.** Predict the paired
`SciPy / FrankenSciPy` 95% CI has an upper endpoint below `15.9x`, more than a
tenfold collapse from the historical 159x headline after the incumbent removes
the per-parameter Python callback loop. Report the ratio whichever way it
falls.

**P3 -- vectorized SciPy wins the whole job.** Predict the paired
`SciPy / FrankenSciPy` 95% CI lies wholly below `1.0`, meaning the strongest
public vectorized SciPy route is faster than the current Rust
nested-scalar-per-parameter implementation. This is a directional prediction
made before measurement. If FrankenSciPy wins, retract it plainly.

**P4 -- removed callback tax is visible.** Predict the scalar-loop
`tplquad / selected-SciPy` screen-median ratio exceeds `10x`. This directly
tests whether public vectorization removes the historical scalar callback tax.
It is falsified independently of the primary effect if the ratio is at most
`10x`.

**P5 -- observed execution.** On the planned 32-CPU affinity, predict 32
concurrent FrankenSciPy solve workers are directly observed during an untimed
job probe. Predict one active SciPy task for the P1 single-worker incumbent.
The harness reports observed tasks/processes for every screened arm and never
substitutes the requested worker count.

## Measurement and decision gate

The primary experiment uses 15 balanced, interleaved paired rounds with five
complete parameter-study jobs per timed sample, plus 15 independent
FrankenSciPy/FrankenSciPy and 15 independent SciPy/SciPy A/A ratios. The effect
is the median of per-round `SciPy / FrankenSciPy` ratios. A deterministic
10,000-resample bootstrap produces the 95% median interval.

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

The harness additionally fails closed without live SciPy 1.17.1, a performance
governor on every affinity CPU, a fresh numeric Agent Mail `trj` booking claim,
exact source/build provenance, or the pre-timing scientific gate. It
self-reports the running ELF SHA-256 and SciPy engine SHA-256 from inside the
processes, records host and boot identity, affinity, runtime ISA, and actual
observed tasks. The executable must be built through
`rch exec --base <exact-commit> --clean-overlay --no-overlay` while reusing the
single repository target directory `/data/tmp/cargo-target`.

## Chooser boundary

Choose FrankenSciPy `tplquad_many` for this exact 100-parameter job only if its
corrected paired CI lies wholly above `3x` and every scientific/provenance gate
passes. If SciPy wins, the FrankenSciPy advantage is below `3x`, or the result
is undecidable, choose the measured vectorized SciPy arm. In all cases retire
the old 83--159x magnitude for this workload rather than falling back to the
scalar-loop comparator.

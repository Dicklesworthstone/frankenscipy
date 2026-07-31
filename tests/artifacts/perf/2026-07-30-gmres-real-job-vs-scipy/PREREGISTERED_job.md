# Pre-registration: whole-job steady convection-diffusion GMRES source screen

**Bead:** `frankenscipy-gugk1`
**Date committed:** 2026-07-30 America/New_York
**Measurement status at commit:** NOT RUN

This document fixes the workload, incumbent screen, predictions, falsifiers,
conformance contract, measurement gate, and chooser wording before any timing
of this whole job. The implementation and every timing result come later.

## Numerical correction and prior evidence

The `36.1770x` single-solve result is BDF, not GMRES. It has already been
converted into the separate `8.3326x` whole-job stiff reaction-screen result in
`frankenscipy-0tx94`. It is not a prior for this experiment.

The applicable restart-matched GMRES evidence is commit `f10be8e16` and raw
artifact SHA-256
`6703ccd1a80a6d9fb05d5a6cb472da67832bc1c060514ede191c9eed66945585`:

| side / unknowns | iterations ours / SciPy | SciPy / ours |
|---:|---:|---:|
| 32 / 1,024 | 127 / 127 | 4.8850x |
| 64 / 4,096 | 163 / 163 | 1.5725x |
| 96 / 9,216 | 227 / 227 | 0.9397x |

The attribution in commit `86bcccd74` is also fixed prior evidence:
FrankenSciPy avoids about 87 microseconds of SciPy interpreted bookkeeping per
inner iteration at small `n`, but its marginal dense-vector work per unknown is
1.8-2.7x slower. Therefore this experiment is intentionally confined to the
measured small-`n` regime and must not become a size-general GMRES claim.

## User-recognizable job fixed before measurement

The job is a serial steady-state two-dimensional convection-diffusion-reaction
source-location screen:

- grid: `32 x 32`, hence 1,024 unknowns;
- operator: the existing nonsymmetric five-point stencil with
  `diagonal=4.001`, `west=-1.2`, `east=-0.8`, and both vertical coefficients
  `-1.0`;
- twelve localized source scenarios at the Cartesian product of grid rows
  `[6, 16, 25]` and columns `[5, 12, 20, 27]`;
- each right-hand side is the exactly reproducible compact tent
  `1/16 + max(0,4-|row-r0|)*max(0,4-|column-c0|)/16`;
- solver: public GMRES, zero initial guess, public default `restart=20`,
  `rtol=1e-5`, `atol=0`, and `maxiter=10*n`;
- requested compute resources: one pinned CPU and one observed thread in each
  arm.

Each timed job must include:

1. assembling the sparse operator;
2. constructing all twelve right-hand sides;
3. constructing any selected SciPy preconditioner;
4. all twelve public GMRES calls;
5. materializing all 12,288 field values;
6. computing, for every field, domain inventory, east-boundary outlet
   integral, and source-weighted exposure.

Python interpreter startup, SciPy import, pipe transport, incumbent screening,
parity serialization, provenance collection, and bootstrap calculation are
outside both timed regions. Matrix, source, solver, output, or preconditioner
state may not persist across timed whole-job repetitions.

## Strongest-incumbent screen fixed before measurement

The legacy arm is genuine installed SciPy 1.17.1. Before headline samples, the
harness will run one untimed end-to-end screen of these public GMRES
configurations and select the lowest valid whole-job wall time:

1. `csr_matrix`, no preconditioner;
2. `csr_array`, no preconditioner;
3. `csc_matrix`, no preconditioner;
4. `csc_array`, no preconditioner;
5. `csr_matrix` with a diagonal/Jacobi `LinearOperator`;
6. `csc_matrix` with one default `spilu` factorization reused by all twelve
   GMRES calls.

Every candidate must converge for all twelve scenarios and pass the same
full-result contract before it is eligible. The fastest eligible candidate,
even if it defeats the desired claim, is the headline incumbent. The selected
configuration is reconstructed inside every timed repetition.

Direct `spsolve`/`splu` solves and other Krylov methods are outside this
GMRES-configuration comparison. The final chooser must say that explicitly;
this experiment cannot choose the fastest arbitrary sparse solver.

## Pre-registered mechanism and predictions

The prior single-RHS result attributes the unpreconditioned small-`n` win to
SciPy's interpreted Arnoldi/Givens loop, not to a faster FrankenSciPy numeric
kernel. Twelve independent right-hand sides repeat that tax enough that matrix
assembly and three linear scientific reductions per field should not dominate.
Conversely, ILU can reduce the number of taxed iterations and amortize one
factorization over all twelve right-hand sides. That is the main threat to the
claim and is why it is in the incumbent screen.

Predictions, in decreasing confidence:

- **P1 — unpreconditioned survival:** against the fastest unpreconditioned
  SciPy storage backend, the whole-job paired median ratio will be in
  `[3.5, 5.0]`, with the bootstrap-median CI lower bound above `3.0`.
- **P2 — trajectory parity:** the two unpreconditioned arms will report
  exactly the same inner-iteration count for every scenario. Any mismatch
  falsifies attribution of that scenario's difference to per-iteration cost.
- **P3 — strongest incumbent is preconditioned:** the `spilu` configuration
  will win the SciPy screen because one factorization is amortized over twelve
  right-hand sides. This is the riskiest prediction because setup is inside
  timing.
- **P4 — headline compression or reversal:** after selecting the strongest
  valid SciPy configuration, the unpreconditioned `>=3x` whole-job claim will
  not survive. Point prediction: SciPy / FrankenSciPy in `[0.5, 1.5]`.
- **P5 — non-solver work is secondary:** removing operator/source construction
  and scientific reductions in an untimed decomposition will change either
  same-configuration whole-job median by less than 25%.

P1 and P4 deliberately distinguish two questions. P1 tests whether the known
per-iteration mechanism survives realistic setup and postprocessing when the
algorithmic configuration is held fixed. P4 tests whether a SciPy user should
actually choose that weaker configuration. Only the strongest valid incumbent
may appear in the headline.

## Conformance and admissibility contract

Before timing:

- compare all twelve complete solution vectors, all 12,288 components, and all
  36 scientific summaries;
- require every solver call to report convergence;
- require each true relative residual to be at most `1.25*rtol`;
- for component parity require
  `abs(ours-scipy) <= 10*rtol*max(1,abs(scipy))`, zero mismatches, and finite
  relative-L2 difference at most `5e-4`;
- require each summary's scaled error under the same tolerance form to be at
  most one unit;
- report scenario-by-scenario iteration counts for FrankenSciPy, selected
  SciPy, and unpreconditioned SciPy;
- require exact iteration equality only for the unpreconditioned mechanism
  comparison; preconditioning is allowed to change the trajectory, but its
  work counts must remain explicit;
- hash the reconstructed matrix and twelve right-hand sides in each arm and
  require identical input SHA-256 values.

The measurement must use:

- genuine live SciPy 1.17.1 side by side in the same invocation;
- exact FrankenSciPy ELF SHA-256 and every selected SciPy engine/component
  SHA-256;
- strict-remote RCH compilation with no local Cargo fallback;
- a literally exclusive measurement host;
- one requested and observed thread per arm, pinned to one CPU;
- interleaved arm ordering;
- independent same-invocation A/A controls for both arms;
- p50/p95/p99 whole-job wall times and raw samples;
- deterministic bootstrap-median CI;
- the corrected fleet null gate: effect CI excludes one, effect beats twice
  the larger null half-width, each null median is within 2% of one, plus this
  harness's stricter retained endpoint-margin test;
- CV as provenance only, never as the decision gate.

## Decision and chooser contract

Three distinct outcomes must not be conflated:

1. **Durable whole-job claim:** selected-incumbent ratio CI lower bound is at
   least `3.0` and every conformance/gate condition passes.
2. **Narrow decided result:** the corrected gate decides a win or loss, but the
   selected-incumbent win is below `3.0`.
3. **No claim:** the gate is indeterminate or any admissibility condition
   fails.

The final text must report the ratio whichever way it falls. It must end with a
literal `CHOOSER STATEMENT:` naming:

- the measured 32x32, twelve-source, serial, unpreconditioned or selected
  preconditioned configuration;
- the fastest screened SciPy configuration;
- the existing side-96/n=9,216 FrankenSciPy GMRES loss as the measured
  large-size boundary;
- that direct sparse solvers, other preconditioners, other matrices, and other
  sizes remain undecided.

No source optimization is authorized by this bead. This is a harness and
evidence lane only.

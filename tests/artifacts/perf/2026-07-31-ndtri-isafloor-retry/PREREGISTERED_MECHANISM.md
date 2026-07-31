# PRE-REGISTERED: `ndtri` central-region SIMD ISA-floor retry

Written and committed before reconstructing the candidate, building its
benchmark harness, or collecting any new timing.

Bead: `frankenscipy-2b7tr`.

Pre-primary protocol correction: review of the constructed harness found that
the first draft applied the nearer effect-CI endpoint distance to both
robustness clauses. The corrected implementation now applies the registered
point-effect distance to twice the widest null bootstrap half-width, and
separately applies the nearer effect-CI endpoint distance to twice the widest
null CI endpoint distance. The predictions, fixtures, candidate, and chooser
are unchanged. No timing was collected before this correction.

## Why this retry is an obligation

The 2026-07-04 negative-ledger row rejected an eight-lane SIMD evaluation of
the central Cephes rational in `ndtri`: the candidate was byte-identical, but
measured only 0.82--0.91x of the scalar path on the mixed uniform workload.
That row predates commit `d89ca19f6` (2026-07-10), which changed the workspace
deployment floor from generic x86-64/SSE2 to AVX2+FMA. The concrete
`VOID-ISAFLOOR` retry predicate in `docs/LEDGER_RESURRECTION.md` is therefore
now satisfied, and no later `ndtri` SIMD adjudication exists.

The later Cholesky candidates found by the same audit are not open
obligations: their structural primitives were re-decided and shipped as the
AVX2+FMA MR4xNR8 SYRK and blocked panel TRSM. This row has not been consumed
that way.

## Candidate and counted mechanism

Reconstruct the rejected mechanism, not a new algorithm:

- process eight input probabilities at a time;
- evaluate the existing central Cephes rational with `Simd<f64, 8>`;
- preserve the exact scalar operation order and do not use `mul_add`, so the
  central result must be bit-identical;
- for tail lanes, retain the existing scalar log/sqrt/rational path;
- as in the rejected candidate, the vector chunk computes the central
  rational before tail lanes are replaced by scalar results.

The ISA-floor mechanism predicts that an eight-lane operation which required
four 128-bit vector groups at the old SSE2 floor needs only two 256-bit groups
at the current AVX2 floor. It does not accelerate the scalar tail log/sqrt
path. The three fixtures therefore separate the mechanism from the whole
workload:

1. `central`: values strictly inside `(exp(-2), 1-exp(-2))`;
2. `tail`: values split between the lower and upper tail;
3. `mixed`: the original uniform `(0,1)` shape, with about 27% tail values.

Each fixture uses the same element count and output materialization. Input
construction, process spawning, provenance checks, and checksum validation
remain outside timed regions.

## Predictions and falsifiers

1. **The AVX2 mechanism is real in the central region.** Predict the
   scalar/candidate median ratio has a 95% bootstrap CI wholly above 1.25x on
   `central`. Falsified if its CI lower endpoint is at or below 1.25.
2. **The old whole-workload rejection re-stands.** Predict `mixed` remains
   undecidable or a loss: its corrected-gate verdict will not be a WIN, even
   if the point estimate crosses 1.0. Falsified only by a corrected-gate WIN
   with the full CI above 1.0.
3. **The tail is the limiting branch.** Predict the candidate does not win
   `tail`, because it retains every scalar log/sqrt operation and also
   computes central-vector work that is discarded. Falsified by a
   corrected-gate WIN on `tail`.
4. **The candidate is arithmetically isomorphic.** Predict zero bit
   mismatches against `ndtri_scalar` over all benchmark inputs plus branch
   boundaries and endpoint-adjacent probes. Any mismatch rejects the
   candidate before timing, regardless of speed.
5. **Live-incumbent direction.** Predict the mixed candidate is not a durable
   1.2x win over the genuine vectorized `scipy.special.ndtri` from SciPy
   1.17.1. Falsified by a corrected-gate SciPy/candidate latency ratio whose
   95% CI is wholly above 1.2.

These predictions deliberately allow the ISA mechanism to be confirmed on
the branch it affects while the production candidate is still rejected on
the actual mixed workload.

## Measurement contract

- One strict-RCH binary built from its committed harness source with
  `rch exec --base <harness-commit> --clean-overlay --no-overlay`.
- Reuse `CARGO_TARGET_DIR=/data/tmp/cargo-target`; no per-task target
  directory.
- Execute on an exclusive, quiescent host and report hostname, boot ID, CPU
  model/ISA, governor state, and the executed ELF SHA-256 calculated and
  printed from inside the process.
- Report actual observed task/thread activity for the Rust and Python arms,
  not requested thread counts. Pin both to one admitted CPU and cap common
  numerical-library thread variables to one.
- Require genuine live SciPy 1.17.1 and NumPy provenance; refuse synthetic or
  FrankenSciPy-shadowed modules.
- Use at least 15 interleaved rounds per row, independent baseline/baseline
  and candidate/candidate null controls, and deterministic bootstrap-median
  95% CIs.
- Corrected decision gate:
  1. the effect CI excludes 1.0 in the claimed direction;
  2. the effect median's distance from 1.0 exceeds twice the widest null
     bootstrap half-width;
  3. the nearer effect-CI endpoint's distance from 1.0 exceeds twice the
     widest null-CI endpoint distance from 1.0; and
  4. both null medians lie in `[0.98, 1.02]`.
  Whether a null CI straddles 1.0 is retained as telemetry, never a veto.
- Any parity, identity, version, observed-thread, quiescence, or corrected
  null-gate failure makes the row `UNDECIDABLE`; no raw point estimate is
  promoted.

## Pre-committed production chooser

Ship the SIMD path only if the `mixed` row is a corrected-gate WIN, exact-bit
parity holds, and the live-SciPy ratio is not a LOSS. A central-only win does
not justify scanning user inputs or adding a distribution-sensitive public
chooser: retain the current scalar/work-gated implementation. If the mixed
row wins, keep the current small/huge-size policy boundaries and use SIMD
only in the existing serial middle-size band; otherwise ship evidence only
and leave production code unchanged.

# MASTER_EXECUTION_TODO

Execution scope requested: complete all previously identified next steps in one pass:
1. `FSCI-P2C-002` linalg implementation + conformance + artifacts.
2. Python oracle integration for automated SciPy differential capture.
3. First interactive `ftui` conformance dashboard binary.

## 0. Control Plane

- [x] Create master granular TODO checklist in-repo.
- [x] Keep this file updated at each milestone boundary.
- [x] Keep `update_plan` statuses synchronized with actual progress.

## 1. P2C-002 Linalg Core Implementation

### 1.1 API and Contracts
- [x] Define clean-room public API for first-wave routines in `fsci-linalg`:
  - [x] `solve`
  - [x] `solve_triangular`
  - [x] `solve_banded`
  - [x] `inv`
  - [x] `det`
  - [x] `lstsq`
  - [x] `pinv`
- [x] Define explicit option/contract structs and enums:
  - [x] matrix-assumption enum
  - [x] triangular-transpose enum
  - [x] strict/hardened-aware options
  - [x] warning types
  - [x] error taxonomy aligned to SciPy-observable messages where scoped
- [x] Define/implement matrix validation helpers:
  - [x] rectangular shape validation
  - [x] square shape checks
  - [x] finite checks
  - [x] dimension compatibility checks

### 1.2 Algorithmic Implementations
- [x] Implement general linear solve path (stable decomposition path).
- [x] Implement diagonal/triangular assumption fast paths.
- [x] Implement banded -> dense conversion + solve path.
- [x] Implement matrix inverse via stable solve/decomposition route.
- [x] Implement determinant via decomposition route.
- [x] Implement least-squares path with rank/singular values output.
- [x] Implement pseudo-inverse path with `(atol + rtol * maxS)` thresholding.
- [x] Implement hardened-mode fail-closed behavior for unsupported/unsafe states.
- [x] Implement condition/ill-conditioning signal surface (warning metadata).

### 1.3 Tests for fsci-linalg
- [x] Unit tests for happy-path outputs per operation.
- [x] Unit tests for shape validation/error messages.
- [x] Unit tests for singular/near-singular behavior.
- [x] Unit tests for threshold validation (`atol`, `rtol` nonnegative for `pinv`).
- [x] Unit tests for strict vs hardened validation deltas.

## 2. Conformance Harness Extensions (Linalg Packet)

### 2.1 Schema and Runner
- [x] Add linalg packet fixture schema to `fsci-conformance`.
- [x] Add linalg expected-outcome schema.
- [x] Add numeric comparator utilities:
  - [x] vector comparison with `atol/rtol`
  - [x] matrix comparison with `atol/rtol`
  - [x] scalar comparison
- [x] Implement `run_linalg_packet(...)`.
- [x] Integrate report generation for linalg packet cases.

### 2.2 P2C-002 Fixture Corpus
- [x] Create `FSCI-P2C-002_linalg_core.json` fixture file with operation coverage:
  - [x] solve
  - [x] solve_triangular
  - [x] solve_banded
  - [x] inv
  - [x] det
  - [x] lstsq
  - [x] pinv
- [x] Add edge/adversarial cases:
  - [x] shape mismatch
  - [x] singular matrix
  - [x] invalid thresholds for pinv

### 2.3 Artifact Outputs
- [x] Ensure linalg run produces packet parity report.
- [x] Ensure linalg run emits RaptorQ sidecar.
- [x] Ensure linalg run emits decode-proof metadata artifact.
- [x] Ensure artifacts are stored under `fixtures/artifacts/FSCI-P2C-002`.

## 3. Python Oracle Integration

### 3.1 Oracle Runner
- [x] Add Python-oracle config model in Rust.
- [x] Add robust process runner:
  - [x] input serialization
  - [x] script invocation
  - [x] output parsing
  - [x] timeout/error handling (process launch/failure surfaced with typed errors)
- [x] Add explicit error taxonomy for:
  - [x] missing python executable
  - [x] missing SciPy
  - [x] script runtime failure
  - [x] malformed oracle output

### 3.2 SciPy Oracle Script
- [x] Add script for linalg packet capture.
- [x] Implement per-operation SciPy execution and result serialization.
- [x] Implement deterministic output ordering.
- [x] Emit structured failure payloads for unsupported/error cases.

### 3.3 Integration and Artifacts
- [x] Add harness function to capture oracle output and write artifact bundle.
- [x] Add `oracle_capture.json` artifact for packet runs.
- [x] Wire optional oracle capture path into linalg conformance flow.
- [x] Provide non-SciPy fallback behavior that does not break default test runs.

### 3.4 Tests
- [x] Add unit/integration tests for runner logic.
- [x] Add test using mock python script (no SciPy dependency).
- [x] Add optional integration path test when SciPy is available. (implemented as auto-skip when SciPy is absent)

## 4. FrankenTUI Interactive Dashboard

### 4.1 Binary Scaffolding
- [x] Add `fsci-conformance` binary target: `conformance_dashboard`.
- [x] Wire required `ftui` runtime features for the binary.
- [x] Add CLI arguments for artifact root and optional packet filter.

### 4.2 UI Model
- [x] Define app model state:
  - [x] packet list
  - [x] selected packet index
  - [x] selected case index
  - [x] active panel/tab
- [x] Define message/event mapping:
  - [x] up/down navigation
  - [x] tab switch
  - [x] quit

### 4.3 UI Rendering
- [x] Render packet list panel (pass/fail counts).
- [x] Render case detail panel (message + status).
- [x] Render drift summary panel (failed %, counts, artifact presence).
- [x] Use `ftui::Style` severity coloring consistent with conformance styles.

### 4.4 Dashboard Data Path
- [x] Load all parity artifacts from `fixtures/artifacts`.
- [x] Build packet summary aggregation.
- [x] Handle malformed/missing artifact files gracefully.

### 4.5 Tests
- [x] Add non-interactive model tests for navigation and selection logic.
- [x] Add parser tests for artifact discovery/loading.

## 5. Documentation and Tracking Updates

- [x] Update `FEATURE_PARITY.md` for P2C-002 and oracle/dashboard state transitions.
- [x] Update `docs/planning/PROPOSED_ARCHITECTURE.md` to include new conformance/dashboard/oracle flows.
- [x] Update `README.md` usage notes for:
  - [x] linalg packet runner
  - [x] oracle capture workflow
  - [x] dashboard binary
- [x] Add/refresh fixture README entries.

## 6. Quality Gates and Verification

- [x] Run `cargo fmt --check`.
- [x] Run `cargo check --all-targets`.
- [x] Run `cargo clippy --all-targets -- -D warnings`.
- [x] Run `cargo test --workspace`.
- [x] Run `cargo test -p fsci-conformance -- --nocapture`.
- [x] Run `cargo bench`.
- [x] Resolve any failures and re-run until clean.

## 7. Closeout

- [x] Confirm no destructive operations were executed.
- [x] Summarize implemented changes with file-level references. (captured in final report)
- [x] List residual risks and next-most-valuable follow-ups. (captured in final report)
- [x] Confirm method-stack artifact production status (produced vs deferred). (captured in final report)

## 8. Numerical correctness and remaining fitting work (2026-11-15)

The comprehensive specification and the 4,623-record Beads graph were reviewed
before selecting this work. The older W6 descriptions substantially lag the code;
the targets below are concrete defects or missing public behavior in the current
implementation. Broad feature Beads remain open until their full acceptance
criteria are met.

### 8.1 Quadrature and series reliability

- [x] Reject non-finite interior tanh-sinh samples and accumulated sums.
- [x] Preserve integrable endpoint singularities when exp-sinh nodes round to a bound.
- [x] Use range-safe finite-interval geometry; reject an empty representable node set.
- [x] Avoid squaring tiny error estimates into a false zero; require finite error estimates.
- [x] Stop invalid finite/infinite series before expanding unnecessary evaluation windows.
- [x] Reject non-finite Gauss-Kronrod samples and include parent evaluations in recursive counts.
- [x] Add analytical regressions and a 12-case required live-SciPy comparison.
- [x] Add these checks to the existing numerical regression workflow.
- [x] Independently review the arithmetic and reproduce the original failure mechanisms in
  JavaScript double precision.
- [ ] Execute the new Rust tests and live SciPy oracle, formatting, workspace check and clippy.
  The execution environment disconnected during dependency setup; numerical reproductions
  and source review are not Rust test passes.

### 8.2 Remaining implementation targets

- [x] Implement trait-level likelihood fits for LogSeries, RandInt and Zipfian, including fixed-n
  Zipfian fitting, invalid-data errors and resource limits (advances frankenscipy-1ksfv.14).
  Rust and live-oracle execution remains pending under section 9.
- [ ] Correct sparse iterative solver norm underflow/overflow, exact least-squares stationarity
  and non-finite warm-start validation.
- [ ] Implement sparse/matrix-free bounded linear least squares using the existing LSMR
  infrastructure. This larger feature was deferred until executable validation is available.

### 8.3 Follow-up correctness work

- [ ] Improve tanh-sinh endpoint-tail error estimates near the representability limit;
  the present work does not certify the omitted tail of a right-endpoint singularity.
- [ ] Bound finite nsum grid cardinality before conversion to i64.
- [ ] Run full Beads closeout checks after executable evidence is available. No feature
  closure is claimed from a JavaScript reproduction or a source-only review.

## 9. Discrete fitting and sparse least-squares follow-through

The specification and all 4,623 Beads records were reviewed before selecting
these functional changes. Older W6 descriptions were checked against actual code.
The execution host became unavailable during the first Rust build; completed
manual edits were recovered against the exact GitHub base and uploaded without
changing unrelated source. Implementation checkboxes below do not claim test passes.

### 9.1 Discrete likelihood fitting (advances `frankenscipy-1ksfv.14`)

- [x] Implement trait-level LogSeries MLE with a monotone likelihood score and
  cancellation-safe evaluation for samples dominated by ones.
- [x] Fit RandInt signed endpoints and use overflow-safe interval width, means
  and quantiles across the representable signed range.
- [x] Implement Zipfian MLE with profiled support and fixed-n fitting.
- [x] Admit the genuine Zipfian a=0 uniform boundary.
- [x] Report nonexistent finite maxima, invalid observations and resource limits
  through typed errors instead of panics or arbitrary parameter estimates.
- [x] Add three unit regressions and two integration tests, including 12 seeded
  comparisons against required SciPy 1.17.1 / NumPy 2.4.3.
- [x] Independently check the score equations, high-precision root references
  and signed-interval arithmetic in JavaScript double precision.
- [ ] Execute the Rust tests and the live oracle, and run formatting/Clippy.
- [ ] Complete the wider fit-completeness Bead's all-family acceptance criteria.

### 9.2 Validation status and next commands

The exact pinned nightly and SciPy/NumPy versions were installed and verified,
and `cargo fetch --locked` succeeded. The first compilation was interrupted
after resource contention; a one-job, debug-disabled retry reached fsci-sparse
before the execution host went offline. No Rust test pass is claimed.

Pending checks for the completed fitting subset:

- [ ] `cargo test --locked -p fsci-stats --lib discrete_likelihood_fits`
- [ ] `cargo test --locked -p fsci-stats --lib randint_fitted_extreme_signed_intervals`
- [ ] `FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test --locked -p fsci-stats --test mle_scipy discrete_mle -- --nocapture`
- [ ] Formatting, workspace check, Clippy and the required broader conformance gates.

The main Beads graph is left open. Local claims and new issue records could not
be synchronized after the host became unavailable; do not infer closures from
the implementation checklist.

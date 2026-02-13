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
- [x] Update `PROPOSED_ARCHITECTURE.md` to include new conformance/dashboard/oracle flows.
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

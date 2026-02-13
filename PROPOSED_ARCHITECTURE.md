# PROPOSED_ARCHITECTURE

## 1. Architecture Principles

1. Spec-first implementation, no line translation.
2. Strict mode for compatibility; hardened mode for defensive operation.
3. RaptorQ sidecars for long-lived conformance and benchmark artifacts.
4. Profile-first optimization with behavior proof artifacts.

## 2. Crate Map

- fsci-integrate: ODE/IVP and quadrature scope
- fsci-linalg: dense linear algebra adapters
- fsci-opt: root/minimize scoped solvers
- fsci-sparse: sparse array structures and ops
- fsci-fft: FFT backend adapters
- fsci-special: scoped special functions + error policy
- fsci-arrayapi: backend namespace compatibility layer
- fsci-conformance: SciPy differential harness + RaptorQ sidecar emission + `ftui` dashboard + oracle capture integration
- fsci-runtime: strict/hardened policy + evidence ledger + decision-theoretic admission controller

## 3. Runtime Plan

- API layer normalizes inputs and validates invariants.
- Planner/dispatcher selects algorithm implementation.
- Core engine executes with explicit invariant checks.
- Conformance adapter captures oracle and target outputs.
- Evidence layer emits parity reports, benchmark deltas, and decode proofs.
- Asupersync integration provides RaptorQ systematic encoding for durable artifacts.
- FrankenTUI integration provides operator-facing render contracts for parity and drift summaries.
- Dashboard state layer performs best-effort artifact discovery to keep operator workflows resilient under malformed files.

## 4. Compatibility and Security

- strict mode: maximize scoped behavioral parity.
- hardened mode: same outward contract plus bounded defensive checks.
- fail-closed on unknown incompatible metadata/protocol fields.

## 5. Performance Contract

- baseline, profile, one-lever optimization, verify parity, re-baseline.
- p95/p99 and memory budgets enforced in CI.

## 6. Conformance Contract

- feature-family fixtures captured from legacy oracle.
- machine-readable parity report per run.
- regression corpus for previously observed mismatches.
- packet-level sidecars and decode-proof metadata:
  - `parity_report.json`
  - `parity_report.raptorq.json`
  - `parity_report.decode_proof.json`
- optional SciPy differential capture:
  - `oracle_capture.json`
  - `oracle_capture.error.txt` fallback in optional mode

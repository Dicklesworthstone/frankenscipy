# Conformance Fixtures

This folder stores normalized oracle-vs-target fixtures for fsci-conformance.

- `smoke_case.json`: minimal bootstrap fixture ensuring harness wiring works.
- `FSCI-P2C-001_validate_tol.json`: packet-level conformance cases for
  `scipy.integrate._ivp.common.validate_tol` parity.
- `FSCI-P2C-002_linalg_core.json`: first-wave dense linalg conformance cases for:
  - `solve`
  - `solve_triangular`
  - `solve_banded`
  - `inv`
  - `det`
  - `lstsq`
  - `pinv`
- `artifacts/`: generated parity reports, RaptorQ sidecars, and decode-proof logs.

Expected per-packet durable artifacts:

- `parity_report.json`
- `parity_report.raptorq.json`
- `parity_report.decode_proof.json`

Optional oracle artifacts:

- `oracle_capture.json` when SciPy capture succeeds.
- `oracle_capture.error.txt` when oracle capture is configured as optional and fails.

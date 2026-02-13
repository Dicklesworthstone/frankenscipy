# FEATURE_PARITY

## Status Legend

- `not_started`
- `in_progress`
- `parity_green`
- `parity_gap`

## Feature Family Matrix

| Feature Family | Status | Notes |
|---|---|---|
| IVP tolerance validation and step guards (`FSCI-P2C-001`) | in_progress | clean-room `validate_tol`, `validate_first_step`, `validate_max_step` ported in `fsci-integrate` |
| Dense linear algebra (`FSCI-P2C-002`) | in_progress | clean-room `solve`, `solve_triangular`, `solve_banded`, `inv`, `det`, `lstsq`, `pinv` landed in `fsci-linalg`; packet harness + artifacts active |
| Root and optimize core (`FSCI-P2C-003`) | not_started | `_root` and `_minpack` behavior |
| Sparse array baseline (`FSCI-P2C-004`) | not_started | CSR/CSC/COO core semantics |
| FFT backend routing (`FSCI-P2C-005`) | not_started | `scipy.fft._backend` parity scope |
| Special-function backend/error policy (`FSCI-P2C-006`) | not_started | warning/error contract parity |
| Array API compatibility glue (`FSCI-P2C-007`) | not_started | backend negotiation parity |
| Differential harness + artifact pipeline (`FSCI-P2C-008`) | in_progress | packet fixtures, parity report generation, RaptorQ sidecar + decode-proof metadata, optional SciPy oracle capture, and `ftui` dashboard landed in `fsci-conformance` |

## Packet Readiness Snapshot

| Packet ID | Extraction | Impl | Conformance Fixtures | Sidecar Artifacts | Overall |
|---|---|---|---|---|---|
| `FSCI-P2C-001` | ready | in_progress | in_progress | in_progress | in_progress |
| `FSCI-P2C-002` | ready | in_progress | in_progress | in_progress | in_progress |
| `FSCI-P2C-003` | ready | not_started | not_started | not_started | not_started |
| `FSCI-P2C-004` | ready | not_started | not_started | not_started | not_started |
| `FSCI-P2C-005` | ready | not_started | not_started | not_started | not_started |
| `FSCI-P2C-006` | ready | not_started | not_started | not_started | not_started |
| `FSCI-P2C-007` | ready | not_started | not_started | not_started | not_started |
| `FSCI-P2C-008` | ready | in_progress | in_progress | in_progress | in_progress |

## Required Evidence Per Feature Family

1. Differential fixture report.
2. Edge-case/adversarial test results.
3. Benchmark delta (for runtime-significant paths).
4. Documented compatibility exceptions (if any).
5. RaptorQ sidecar manifest plus decode-proof record for each durable artifact bundle.

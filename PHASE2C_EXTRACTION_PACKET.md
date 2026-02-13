# PHASE2C_EXTRACTION_PACKET.md — FrankenSciPy

Date: 2026-02-13

Purpose: convert Phase-2 analysis into direct implementation tickets with concrete legacy anchors, target crates, and oracle tests.

## 1. Ticket Packets

| Ticket ID | Subsystem | Legacy anchors (classes/functions) | Target crates | Oracle tests |
|---|---|---|---|---|
| `FSCI-P2C-001` | IVP solver core | `validate_tol`, `select_initial_step`, `OdeSolution`, `num_jac` in `integrate/_ivp/common.py`; `OdeSolver`, `DenseOutput` in `_ivp/base.py` | `fsci-integrate` | `scipy/integrate/_ivp/tests/test_ivp.py`, `test_rk.py` |
| `FSCI-P2C-002` | Linalg solve/decompose first wave | `solve`, `solve_triangular`, `solve_banded`, `inv`, `det`, `lstsq`, `pinv` in `linalg/_basic.py` | `fsci-linalg` | `scipy/linalg/tests/*` |
| `FSCI-P2C-003` | Optimize scalar/multivariate core | `OptimizeResult`, `_prepare_scalar_function`, `_minimize_bfgs`, `_minimize_cg`, `_minimize_powell`, scalar minimizers in `optimize/_optimize.py` | `fsci-opt` | `scipy/optimize/tests/*` |
| `FSCI-P2C-004` | Sparse base/model invariants | `_spbase`, `isspmatrix` in `sparse/_base.py`; `upcast`, `get_index_dtype`, `check_shape`, `broadcast_shapes` in `_sputils.py` | `fsci-sparse` | `scipy/sparse/tests/*` |
| `FSCI-P2C-005` | FFT backend routing | `_ScipyBackend`, `_backend_from_arg`, `set_global_backend`, `register_backend`, `set_backend` in `fft/_backend.py` | `fsci-fft`, `fsci-arrayapi` | `scipy/fft/tests/test_backend.py`, `test_basic.py` |
| `FSCI-P2C-006` | Special-function backend/error policy | `_FuncInfo`, `_get_native_func`, function shims in `special/_support_alternative_backends.py`; `SpecialFunctionWarning`, `SpecialFunctionError` in `_sf_error.py` | `fsci-special`, `fsci-arrayapi` | `scipy/special/tests/test_support_alternative_backends.py`, `test_sf_error.py` |
| `FSCI-P2C-007` | Array-API compatibility glue (scoped) | backend negotiation hooks in `scipy/_lib/_array_api*.py` plus module-specific adapters | `fsci-arrayapi` | `_lib` and module-level array API tests |
| `FSCI-P2C-008` | Conformance harness corpus wiring | fixture normalization and comparison schema | `fsci-conformance` | cross-family differential suites |

## 2. Packet Definition Template

For each ticket above, deliver all artifacts in the same PR:

1. `legacy_anchor_map.md`: path + line anchors + extracted behavior.
2. `contract_table.md`: input/output/error + tolerance/backend semantics.
3. `fixture_manifest.json`: oracle mapping and fixture IDs.
4. `parity_gate.yaml`: strict + hardened pass criteria.
5. `risk_note.md`: boundary risks and mitigations.

## 3. Strict/Hardened Expectations per Packet

- Strict mode: exact scoped SciPy-observable behavior.
- Hardened mode: same outward contract with bounded defensive checks (numeric/pathological inputs).
- Unknown incompatible backend/metadata path: fail-closed.

## 4. Immediate Execution Order

1. `FSCI-P2C-001`
2. `FSCI-P2C-002`
3. `FSCI-P2C-003`
4. `FSCI-P2C-004`
5. `FSCI-P2C-005`
6. `FSCI-P2C-006`
7. `FSCI-P2C-007`
8. `FSCI-P2C-008`

## 5. Done Criteria (Phase-2C)

- All 8 packets have extracted anchor maps and contract tables.
- At least one runnable fixture family exists per packet in `fsci-conformance`.
- Packet-level parity report schema is produced for every packet.
- RaptorQ sidecars are generated for fixture bundles and parity reports.

## 6. Per-Ticket Extraction Schema (Mandatory Fields)

Every `FSCI-P2C-*` packet MUST include:
1. `packet_id`
2. `legacy_paths`
3. `legacy_symbols`
4. `algorithm_contract`
5. `tolerance_contract`
6. `backend_contract`
7. `error_warning_contract`
8. `strict_mode_policy`
9. `hardened_mode_policy`
10. `excluded_scope`
11. `oracle_tests`
12. `performance_sentinels`
13. `compatibility_risks`
14. `raptorq_artifacts`

Missing fields => packet state `NOT READY`.

## 7. Risk Tiering and Gate Escalation

| Ticket | Risk tier | Why | Extra gate |
|---|---|---|---|
| `FSCI-P2C-001` | Critical | ODE solver tolerance semantics are fragile | tolerance witness gate |
| `FSCI-P2C-002` | Critical | linalg solve semantics are foundational | decomposition parity gate |
| `FSCI-P2C-003` | High | optimizer status/convergence easily drifts | convergence class lock |
| `FSCI-P2C-004` | High | sparse format/index contracts subtle | sparse invariant replay |
| `FSCI-P2C-005` | High | FFT backend routing affects correctness/perf | backend route gate |
| `FSCI-P2C-006` | High | special-function warning/error behavior is user-visible | warning contract gate |

Critical tickets must pass strict drift `0`.

## 8. Packet Artifact Topology (Normative)

Directory template:
- `artifacts/phase2c/FSCI-P2C-00X/legacy_anchor_map.md`
- `artifacts/phase2c/FSCI-P2C-00X/contract_table.md`
- `artifacts/phase2c/FSCI-P2C-00X/fixture_manifest.json`
- `artifacts/phase2c/FSCI-P2C-00X/parity_gate.yaml`
- `artifacts/phase2c/FSCI-P2C-00X/risk_note.md`
- `artifacts/phase2c/FSCI-P2C-00X/parity_report.json`
- `artifacts/phase2c/FSCI-P2C-00X/parity_report.raptorq.json`
- `artifacts/phase2c/FSCI-P2C-00X/parity_report.decode_proof.json`

## 9. Optimization and Isomorphism Proof Hooks

Optimization allowed only after strict parity baseline.

Required proof block:
- numeric tolerance semantics preserved
- backend routing semantics preserved
- warning/error contracts preserved
- fixture checksum verification pass/fail

## 10. Packet Readiness Rubric

Packet is `READY_FOR_IMPL` only when:
1. extraction schema complete,
2. fixture manifest includes happy/edge/adversarial paths,
3. strict/hardened gates are machine-checkable,
4. risk note includes compatibility + security mitigations,
5. parity report has RaptorQ sidecar + decode proof.

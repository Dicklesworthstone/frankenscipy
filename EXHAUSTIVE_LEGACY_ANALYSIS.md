# EXHAUSTIVE_LEGACY_ANALYSIS.md — FrankenSciPy

Date: 2026-02-13  
Method stack: `$porting-to-rust` Phase-2 Deep Extraction + `$alien-artifact-coding` + `$extreme-software-optimization` + RaptorQ durability + frankenlibc/frankenfs strict/hardened doctrine.

## 0. Mission and Completion Criteria

This document defines exhaustive legacy extraction for FrankenSciPy. Phase-2 is complete only when each scoped subsystem has:
1. explicit invariants,
2. explicit crate ownership,
3. explicit oracle families,
4. explicit strict/hardened policy behavior,
5. explicit performance and durability gates.

## 1. Source-of-Truth Crosswalk

Legacy corpus:
- `/data/projects/frankenscipy/legacy_scipy_code/scipy`
- Upstream oracle: `scipy/scipy`

Project contracts:
- `/data/projects/frankenscipy/COMPREHENSIVE_SPEC_FOR_FRANKENSCIPY_V1.md`
- `/data/projects/frankenscipy/EXISTING_SCIPY_STRUCTURE.md`
- `/data/projects/frankenscipy/PLAN_TO_PORT_SCIPY_TO_RUST.md`
- `/data/projects/frankenscipy/PROPOSED_ARCHITECTURE.md`
- `/data/projects/frankenscipy/FEATURE_PARITY.md`

Important specification gap:
- the comprehensive spec currently defines sections `0-13` then jumps to `21`; missing sections for crate contracts/conformance matrix/threat matrix/perf budgets/CI/RaptorQ envelope must be backfilled.

## 2. Quantitative Legacy Inventory (Measured)

- Total files: `3054`
- Python: `1097`
- Native: `c=284`, `cc=11`, `cpp=24`, `h=148`, `hpp=17`
- Cython: `pyx=60`, `pxd=39`
- Test-like files: `854`

High-density zones:
- `scipy/sparse/linalg` (317 files)
- `scipy/special/tests` (185)
- `scipy/io/matlab` (149)
- `scipy/io/tests` (96)
- `scipy/stats/tests` (71)
- `scipy/optimize/tests` (47)

## 3. Subsystem Extraction Matrix (Legacy -> Rust)

| Legacy locus | Non-negotiable behavior to preserve | Target crates | Primary oracles | Phase-2 extraction deliverables |
|---|---|---|---|---|
| `scipy/integrate/_ivp/{common,base}.py` | tolerance scaling, event handling, step progression | `fsci-integrate` | `_ivp/tests/test_ivp.py`, `test_rk.py` | solver contract ledger and event-state matrix |
| `scipy/linalg/_basic.py` + `linalg/src/*` | factorization/solve semantics and warnings/errors | `fsci-linalg` | `linalg/tests/*` | structure-code mapping + error-surface matrix |
| `scipy/optimize/_optimize.py`, `_minimize.py` | `OptimizeResult` semantics and convergence signaling | `fsci-opt` | `optimize/tests/*` | result contract table + option handling matrix |
| `scipy/sparse/_base.py`, `_sputils.py`, `sparsetools/*` | sparse shape/index/dtype and operator semantics | `fsci-sparse` | `sparse/tests/*` | format invariant ledger |
| `scipy/fft/_backend.py`, `_pocketfft/*` | backend selection and transform consistency | `fsci-fft` | `fft/tests/*` | backend routing decision map |
| `scipy/special/_support_alternative_backends.py`, `_sf_error.py` | special-function fallback and error-state behavior | `fsci-special` | `special/tests/*` | function family contract map |
| `scipy/_lib/_array_api*.py` | array API backend namespace and capability behavior | `fsci-arrayapi` | `_lib/tests/*` + module-level tests | backend capability ledger |

## 4. Alien-Artifact Invariant Ledger (Formal Obligations)

- `FSCI-I1` Solver tolerance integrity: scoped ODE solver behavior respects documented tolerance contracts.
- `FSCI-I2` Linalg signaling integrity: scoped decomposition/solve calls preserve error/warning semantics.
- `FSCI-I3` Optimization status integrity: result status/message semantics are stable for scoped methods.
- `FSCI-I4` Sparse format integrity: format conversions and operations preserve sparse invariants.
- `FSCI-I5` Backend routing integrity: FFT/special/array-api backend dispatch remains deterministic and auditable.

Required proof artifacts per implemented slice:
1. invariant statement,
2. executable witness fixtures,
3. counterexample archive,
4. remediation proof.

## 5. Native/C/Fortran/Cython Boundary Register

| Boundary | Files | Risk | Mandatory mitigation |
|---|---|---|---|
| BLAS/LAPACK wrappers | `linalg/*.pyf.src`, `linalg/src/*` | critical | solver differential fixture corpus |
| optimize native wrappers | `_minpackmodule.c`, `_zerosmodule.c`, `lbfgsb.c` | high | convergence/status parity fixtures |
| sparse C++ kernels | `sparse/sparsetools/*.cxx` | high | format and operator parity corpus |
| pocketfft backend | `fft/_pocketfft/pypocketfft.cxx` | high | backend and transform parity tests |
| special native wrappers | `special_ufuncs.cpp`, `xsf_wrappers.cpp`, `cdflib.c` | high | error-state and value parity fixtures |

## 6. Compatibility and Security Doctrine (Mode-Split)

Decision law (runtime):
`mode + numeric_contract + risk_score + budget -> allow | full_validate | fail_closed`

| Threat | Strict mode | Hardened mode | Required ledger artifact |
|---|---|---|---|
| malformed numeric inputs/metadata | fail-closed | fail-closed with bounded diagnostics | parser incident ledger |
| unstable/ill-conditioned solver inputs | return scoped warning/error semantics | stronger admission guard + audit | solver risk report |
| callback lifecycle misuse | fail invalid lifecycle | quarantine and fail with trace | callback lifecycle ledger |
| unknown incompatible backend metadata | fail-closed | fail-closed | compatibility drift report |
| native-wrapper mismatch | fail parity gate | fail parity gate | native boundary audit report |

## 7. Conformance Program (Exhaustive First Wave)

### 7.1 Fixture families

1. IVP solver tolerance/event fixtures
2. linalg decomposition/solve fixtures
3. optimize convergence/status fixtures
4. sparse format/operator fixtures
5. FFT backend/transform fixtures
6. special-function + error-state fixtures
7. array API backend capability fixtures

### 7.2 Differential harness outputs (`fsci-conformance`)

Each run emits:
- machine-readable parity report,
- mismatch taxonomy,
- minimized repro bundle,
- strict/hardened divergence report.

Release gate rule: critical-family drift => hard fail.

## 8. Extreme Optimization Program

Primary hotspots:
- sparse/linalg inner kernels
- solver step/iteration loops
- FFT backend hot path
- optimize callback-heavy loops

Current governance state:
- comprehensive spec lacks explicit numeric budgets (sections 14-20 absent).

Provisional Phase-2 budgets (must be ratified):
- solver path p95 regression <= +10%
- sparse/fft hotpath p95 regression <= +10%
- p99 regression <= +10%, RSS regression <= +10%

Optimization governance:
1. baseline,
2. profile,
3. one lever,
4. conformance proof,
5. budget gate,
6. evidence commit.

## 9. RaptorQ-Everywhere Artifact Contract

Durable artifacts requiring RaptorQ sidecars:
- conformance fixture bundles,
- benchmark baselines,
- numerical-risk and compatibility ledgers.

Required envelope fields:
- source hash,
- symbol manifest,
- scrub status,
- decode proof chain.

## 10. Phase-2 Execution Backlog (Concrete)

1. Extract IVP tolerance/event/step invariants.
2. Extract linalg structure-code and error semantics.
3. Extract optimize option handling and status contracts.
4. Extract sparse format/index/dtype invariants.
5. Extract FFT backend selection and transform contracts.
6. Extract special-function error-state and fallback contracts.
7. Extract array API backend capability rules.
8. Build first differential fixture corpus for items 1-7.
9. Implement mismatch taxonomy in `fsci-conformance`.
10. Add strict/hardened divergence reporting.
11. Replace crate stubs (`add`) with first real semantic slices.
12. Attach RaptorQ sidecar generation and decode-proof validation.
13. Ratify section-14-20 budgets/gates against first benchmark and conformance runs.

Definition of done for Phase-2:
- each section-3 row has extraction artifacts,
- all seven fixture families runnable,
- governance sections 14-20 empirically ratified and tied to harness outputs.

## 11. Residual Gaps and Risks

- sections 14-20 now exist; top release risk is numeric-budget miscalibration before first benchmark cycle.
- `PROPOSED_ARCHITECTURE.md` crate map formatting contains literal `\n`; normalize before automation.
- numerical correctness regressions can be subtle; broad differential corpus is mandatory before heavy optimization.

## 12. Deep-Pass Hotspot Inventory (Measured)

Measured from `/data/projects/frankenscipy/legacy_scipy_code/scipy`:
- file count: `3054`
- concentration: `scipy/sparse` (`415` files), `scipy/io` (`334`), `scipy/special` (`295`), `scipy/optimize` (`208`), `scipy/linalg` (`118`), `scipy/integrate` (`62`)

Top source hotspots by line count (first-wave extraction anchors):
1. `scipy/stats/_continuous_distns.py` (`12548`)
2. `scipy/stats/_stats_py.py` (`10946`)
3. `scipy/interpolate/src/dfitpack.c` (`9382`)
4. `scipy/special/_add_newdocs.py` (`8871`)
5. `scipy/integrate/__quadpack.c` (`6856`)
6. `scipy/sparse/tests/test_base.py` (`5908`)

Interpretation:
- numerical kernels + Python orchestration are interdependent,
- scoped V1 work must keep linear algebra/opt/integrate/sparse semantics explicit,
- backend routing and tolerance behavior are top silent-regression vectors.

## 13. Phase-2C Extraction Payload Contract (Per Ticket)

Each `FSCI-P2C-*` ticket MUST produce:
1. algorithm/type inventory,
2. tolerance/default-value ledger,
3. backend selection and fallback rules,
4. error and warning contract map,
5. strict/hardened split policy,
6. exclusion ledger,
7. fixture mapping manifest,
8. optimization candidate + isomorphism risk note,
9. RaptorQ artifact declaration,
10. governance backfill linkage notes.

Artifact location (normative):
- `artifacts/phase2c/FSCI-P2C-00X/legacy_anchor_map.md`
- `artifacts/phase2c/FSCI-P2C-00X/contract_table.md`
- `artifacts/phase2c/FSCI-P2C-00X/fixture_manifest.json`
- `artifacts/phase2c/FSCI-P2C-00X/parity_gate.yaml`
- `artifacts/phase2c/FSCI-P2C-00X/risk_note.md`

## 14. Strict/Hardened Compatibility Drift Budgets

Packet acceptance budgets:
- strict critical drift budget: `0`
- strict non-critical drift budget: `<= 0.10%`
- hardened divergence budget: `<= 1.00%` and allowlisted only
- unknown backend/metadata behavior: fail-closed

Per-packet report requirements:
- `strict_parity`,
- `hardened_parity`,
- `numeric_drift_summary`,
- `backend_route_drift_summary`,
- `compatibility_drift_hash`.

## 15. Extreme-Software-Optimization Execution Law

Mandatory loop:
1. baseline,
2. profile,
3. one lever,
4. conformance + invariant replay,
5. re-baseline.

Primary sentinel workloads:
- IVP stiff/non-stiff traces (`FSCI-P2C-001`),
- dense and triangular solve paths (`FSCI-P2C-002`),
- optimizer convergence suites (`FSCI-P2C-003`),
- sparse operator workloads (`FSCI-P2C-004`).

Optimization scoring gate:
`score = (impact * confidence) / effort`, merge only if `score >= 2.0`.

## 16. RaptorQ Evidence Topology and Recovery Drills

Durable artifacts requiring sidecars:
- parity reports,
- numeric mismatch corpora,
- tolerance and backend ledgers,
- benchmark baselines,
- strict/hardened decision logs.

Naming convention:
- payload: `packet_<id>_<artifact>.json`
- sidecar: `packet_<id>_<artifact>.raptorq.json`
- proof: `packet_<id>_<artifact>.decode_proof.json`

Decode-proof failures are hard blockers.

## 17. Phase-2C Exit Checklist (Operational)

Phase-2C is complete only when:
1. `FSCI-P2C-001..008` artifact packs exist and validate.
2. All packets have strict and hardened fixture coverage.
3. Drift budgets from section 14 are met.
4. High-risk packets have at least one optimization proof artifact.
5. RaptorQ sidecars + decode proofs are scrub-clean.
6. Governance backfill tasks are explicitly linked to packet outputs.

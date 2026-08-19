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
- `/data/projects/frankenscipy/docs/planning/COMPREHENSIVE_SPEC_FOR_FRANKENSCIPY_V1.md`
- `/data/projects/frankenscipy/docs/planning/EXISTING_SCIPY_STRUCTURE.md`
- `/data/projects/frankenscipy/docs/planning/PLAN_TO_PORT_SCIPY_TO_RUST.md`
- `/data/projects/frankenscipy/docs/planning/PROPOSED_ARCHITECTURE.md`
- `/data/projects/frankenscipy/docs/planning/FEATURE_PARITY.md`

Specification status:
- `docs/planning/COMPREHENSIVE_SPEC_FOR_FRANKENSCIPY_V1.md` includes sections `14-20` (crate contracts, conformance matrix, threat matrix, CI gates, and RaptorQ envelope); remaining work is empirical validation of those sections against live artifacts.

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
- `docs/planning/PROPOSED_ARCHITECTURE.md` crate map formatting contains literal `\n`; normalize before automation.
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
7. Structured log/perf evidence is verified (`trace_id`, `artifact_refs`, `timing_ms`, `memory_peak`) for each packet test/e2e bundle.

## 18. DOC-PASS-06: Concurrency/Lifecycle Semantics and Ordering Guarantees

### 18.1 Callback lifecycle contracts (legacy anchors)

1. Integrate callback activation is thread-local and stack-scoped.
   - `scipy/integrate/_dopmodule.c:21-214` installs `current_func_callback` before entering DOPRI and resets it to `NULL` on exit.
   - `scipy/integrate/_odepackmodule.c:71-118` does the same for LSODA via `current_odepack_callback`.
2. `LowLevelCallable` callback nesting is explicit push/pop discipline.
   - `scipy/_lib/src/ccallback.h:67-300` stores previous callback in `prev_callback` and restores it in `ccallback_release`.
3. Special-function callback payloads are lifetime-bound to the integration call.
   - `scipy/special/_ellip_harm_2.pyx:103-133` creates a capsule-backed callable, invokes `quad`, and only frees the capsule payload in `finally`, after solver return.

FrankenSciPy contract:
- all callback-bearing solver paths must use thread-local active callback state;
- nested callback invocations must restore prior callback context in LIFO order;
- callback payload memory must outlive all native invocations that can observe it.

### 18.2 Ordering guarantees that are externally observable

1. IVP solver step ordering:
   - `scipy/integrate/_ivp/base.py:179-208` enforces `running -> finished|failed` transitions and forbids stepping non-running solvers.
   - `scipy/integrate/_ivp/ivp.py:657-759` loops `solver.step()`, maps internal status to result status codes, then handles events and final result packaging.
2. Event processing ordering:
   - `scipy/integrate/_ivp/ivp.py:640-694` computes event functions after each successful step in registration order, applies root handling/termination in the same loop iteration, and transitions to terminal state before any subsequent step is permitted.
3. FFT backend precedence ordering:
   - `scipy/fft/_backend.py:52-209` defines semantic order across `set_backend` (scoped), `register_backend` (registered), `set_global_backend` (global), and `skip_backend`.
4. Optimize callback ordering:
   - `scipy/optimize/_linprog_rs.py:300-360` builds callback `OptimizeResult` and calls user callback in-loop before final solver completion.
   - `scipy/optimize/tests/test_tnc.py:321-341` verifies callback receives unscaled iterate view in TNC, preserving callback-observable ordering/value semantics.
5. Worker context ordering:
   - `scipy/fft/_pocketfft/helper.py:208-237` and `scipy/fft/tests/test_multithreading.py:62-90` enforce nested `set_workers` enter/exit restoration order.

FrankenSciPy contract:
- preserve per-step status and event-order semantics for IVP;
- preserve backend selection precedence and context restoration order for FFT;
- preserve callback timing/value semantics for optimize methods.

### 18.3 Concurrency hazard matrix (race/regression surfaces)

| Surface | Legacy evidence | Hazard | Required FrankenSciPy policy |
|---|---|---|---|
| Sparse mutable COO operations | `scipy/sparse/tests/test_coo.py:1233-1279` (`thread_unsafe` markers) | concurrent writer races and shape/index corruption risk | strict mode: reject/serialize unsafe concurrent mutation; hardened mode: same + audit log |
| Sparse low-level kernels | `scipy/sparse/tests/test_sparsetools.py:28-57` | kernels are invoked concurrently; shared mutable inputs can race | permit parallel calls only with non-aliasing input/output contracts |
| BLAS/OpenMP oversubscription | `scipy/conftest.py:108-153` | deadlock/latency collapse with uncontrolled threads and forked workers | cap threadpools per worker; enforce deterministic worker/thread budgets |
| Fork + threadpool interaction | `scipy/fft/tests/test_multithreading.py:29-64`, `scipy/_lib/_util.py:576-677` | pre-fork threadpool state leaks into child process lifecycle | use forkserver/spawn-safe pool policy and explicit pool teardown |
| Special-function global error actions | `scipy/special/tests/test_basic.py:4511-4522` | error-mode changes are not safely composable across threads | treat error-action changes as scoped and serialized; fail closed on conflicting mutation |

### 18.4 V1 implementation invariants for async/runtime design

- `FSCI-C1` Callback TLS invariant: active callback handle is per-thread, never global mutable shared state.
- `FSCI-C2` Callback unwind invariant: every callback activation has exactly one corresponding restore.
- `FSCI-C3` Backend scope invariant: scoped backend/worker overrides always restore previous state, including nested scopes.
- `FSCI-C4` Worker lifecycle invariant: solver-created workers are always torn down at scope exit.
- `FSCI-C5` Thread-unsafe API invariant: APIs known to be unsafe for concurrent mutation are explicitly guarded.
- `FSCI-C6` IVP event-order invariant: event evaluation order is deterministic per registration order and terminating events prevent further solver progression in that step.
- `FSCI-C7` Linalg immutability invariant: unless explicit overwrite flags are set, input arrays remain unmodified.
- `FSCI-C8` Sparse mutation invariant: concurrent COO/DOK mutation attempts are rejected (strict) or serialized with audit evidence (hardened).
- `FSCI-C9` FFT scope invariant: nested backend/worker scopes always unwind in LIFO order without leaking partial state.
- `FSCI-C10` Special errstate invariant: policy changes are scoped and always restored, even on failure paths.

### 18.5 Conformance evidence required for DOC-PASS-06 closure

1. callback lifecycle fixture set (nested callback + thread-local isolation),
2. backend precedence fixture set (`set_backend`/`register_backend`/`set_global_backend` ordering),
3. worker scope fixture set (nested worker contexts and restoration),
4. sparse thread-safety guard fixtures,
5. compatibility note linking strict/hardened behavior for concurrency policy surfaces.

## 19. DOC-PASS-07: Error Taxonomy, Failure Modes, and Recovery Semantics

### 19.1 Cross-module error taxonomy (legacy-anchored)

1. Input-contract violations (shape, dtype, domain, finite-ness):
   - integrate rejects invalid `y0` and malformed bounds/options (`scipy/integrate/_ivp/base.py:5-21`, `scipy/integrate/_ivp/common.py:10-58`).
   - linalg rejects incompatible/singular structure upfront (`scipy/linalg/_basic.py:199-243`).
   - sparse rejects incompatible RHS dimensions (`scipy/sparse/linalg/_dsolve/linsolve.py:712-719`).
   - special rejects invalid parameter domains/types (`scipy/special/_basic.py:230-318`, `scipy/special/_basic.py:2880-2986`).
   - `_lib` validation helpers reject unsupported seed/array/value forms (`scipy/_lib/_util.py:360-430`, `scipy/_lib/_util.py:445-460`, `scipy/_lib/_util.py:549-574`).
2. Numerical breakdown conditions:
   - IVP step-size underflow and solver-step failure propagate to failed status (`scipy/integrate/_ivp/bdf.py:310-359`, `scipy/integrate/_ivp/base.py:180-208`).
   - linalg singular/ill-conditioned LAPACK outcomes emit `LinAlgError` / warnings (`scipy/linalg/_basic.py:28-55`, `scipy/linalg/_basic.py:620-652`).
   - sparse LU singularity surfaces as `LinAlgError("A is singular.")` (`scipy/sparse/linalg/_dsolve/linsolve.py:738-743`).
3. Capability/feature mismatch:
   - FFT planned execution is explicitly unsupported (`NotImplementedError`) (`scipy/fft/_pocketfft/basic.py:11-133`, `scipy/fft/_pocketfft/basic.py:157-219`).
   - `linprog` unsupported callback path for HiGHS is explicit `NotImplementedError` (`scipy/optimize/_linprog.py:653-657`).
4. Policy/configuration mismatch:
   - optimize emits `OptimizeWarning` for ignored options and non-active features (`scipy/optimize/_linprog.py:634-652`).
   - special error policy uses configurable ignore/warn/raise semantics (`scipy/special/_ufuncs_extra_code.pxi:24-117`).

### 19.2 Failure mode matrix (trigger, impact, recovery)

| Family | Trigger | Legacy signal | User-visible impact | Required FrankenSciPy recovery semantics |
|---|---|---|---|---|
| IVP lifecycle misuse | calling `step()` after failure/finish | `RuntimeError` (`_ivp/base.py:190-221`) | solver cannot advance | strict: fail closed; hardened: fail closed + lifecycle audit note |
| IVP numeric infeasibility | step below machine spacing | solver message + status `'failed'` (`_ivp/bdf.py:310-359`, `_ivp/base.py:201-208`) | integration terminates with failure status | strict: preserve status/message exactly; hardened: same + bounded retry policy only if explicitly configured |
| Linalg singular/illegal LAPACK state | `info>0` or illegal args | `LinAlgError` / `ValueError` (`linalg/_basic.py:28-55`, `620-652`) | solve/decomp aborts | strict: preserve exception class and condition; hardened: optional pre-check diagnostics, no silent repair |
| Optimize option incompatibility | unsupported option/method combos | `ValueError` or `OptimizeWarning` (`optimize/_linprog.py:627-652`) | request rejected or partially honored with warning | strict: preserve reject/warn boundary; hardened: same + deterministic compatibility ledger entry |
| Sparse singular solve | SuperLU returns singularity info | `LinAlgError("A is singular.")` (`sparse/.../linsolve.py:738-743`) | no solution output | strict: fail closed; hardened: optional recommendation of alternative solver class, no automatic substitution |
| FFT invalid shape/type/capability | bad length/type or unsupported plan | `ValueError` / `TypeError` / `NotImplementedError` (`fft/_pocketfft/basic.py:22-58`, `52-54`, `11-133`) | transform rejected | strict: preserve argument validation boundary; hardened: same + explicit remediation hint |
| Special-function error actions | domain/overflow/no_result under configured policy | warning/exception per `seterr` action (`special/_ufuncs_extra_code.pxi:24-117`) | outcome depends on active error mode | strict: preserve policy dispatch semantics; hardened: preserve plus scoped policy-change audit |
| `_lib` scalar/data contract break | non-scalar objective / unsupported arrays | `ValueError` (`_lib/_util.py:381-430`, `549-574`) | optimization/stat call rejected early | strict: fail closed pre-dispatch; hardened: same with richer validation context |

### 19.3 User-facing failure semantics that must remain stable

1. optimize status/message vocabulary is part of public behavior (`scipy/optimize/_optimize.py:48-58`).
2. integrate success/failure is signaled both by status code and message; canonical status/message pairs must remain aligned and deterministic (`scipy/integrate/_ivp/ivp.py:657-759`).
3. special-function error behavior is policy-driven (`ignore|warn|raise`), not fixed exception-only semantics (`scipy/special/_ufuncs_extra_code.pxi:24-117`).
4. linalg distinguishes hard errors from soft numerical warnings in batched execution (`scipy/linalg/_basic.py:28-55`).

### 19.4 Strict/Hardened recovery law for failures

Decision law:
`error_class + mode + policy + numeric_context -> fail_closed | warn_and_continue | bounded_recover`

- Strict mode:
  - preserve SciPy-observable exception/warning/status semantics exactly for scoped APIs,
  - no silent coercions or fallback solver substitutions.
- Hardened mode:
  - preserve API contract while allowing bounded defensive recovery only when explicitly declared,
  - every bounded recovery emits deterministic audit evidence with trigger + action + residual risk.

### 19.5 DOC-PASS-07 closure evidence

1. error taxonomy table linked to source anchors for each scoped subsystem;
2. failure-mode matrix with trigger/impact/recovery fields;
3. strict/hardened recovery mapping for each failure family;
4. conformance fixture requirements that validate status/warning/exception parity;
5. risk-note linkage to threat matrix and compatibility drift ledger.

## 20. DOC-PASS-08: Security/Compatibility Edge Cases and Undefined Zones

### 20.1 Source-anchored edge-case register

| Surface | Legacy anchor | Edge case / risk | Compatibility implication | Mitigation requirement |
|---|---|---|---|---|
| integrate callback activation | `scipy/integrate/_odepackmodule.c:83`, `scipy/integrate/_odepackmodule.c:464-466` | callback pointer is TLS-scoped but activation lifecycle is mutable runtime state | concurrent/nested misuse can cross-wire callback state in same thread context | enforce scoped activation/deactivation and prevent overlapping callback scopes |
| BVP verbose output | `scipy/integrate/tests/test_bvp.py:690-709` (`thread_unsafe`) | verbose path manipulates shared `sys.stdout` | concurrent verbose solves are non-deterministic | serialize verbose diagnostics or route to isolated logging channel |
| linalg SVD/convergence | `scipy/linalg/_decomp_svd.py:17-34`, `scipy/linalg/_decomp_svd.py:61-67` | NaN/non-convergence propagate as hard errors | callers rely on fail-closed semantics for invalid linear algebra states | preserve `LinAlgError`/`ValueError` boundaries and do not silently coerce |
| optimize adaptive stepsize | `scipy/optimize/tests/test__basinhopping.py:494-515`, `scipy/optimize/_basinhopping.py:197-260` | mutable stepsize controller is thread-unsafe when shared | shared-state optimizer runs can diverge silently | require per-thread/per-run controller ownership |
| optimize constrained basin hopping | `scipy/optimize/tests/test__basinhopping.py:480-492`, `scipy/optimize/_basinhopping.py:31-115` | local minimizer failure leaves global `success=false` | success bit is contractual; best-effort acceptance is invalid | preserve fail-closed success semantics and explicit status/message |
| sparse DOK/COO mutation | `scipy/sparse/tests/test_dok.py:1-27`, `scipy/sparse/tests/test_coo.py:1212-1236`, `scipy/sparse/tests/test_coo.py:1264`, `scipy/sparse/tests/test_coo.py:1275` | concurrent writes are thread-unsafe | mutable sparse ops in parallel are undefined | strict: fail fast on concurrent mutation attempt; hardened: serialize with deterministic audit trace |
| sparse LU in parallel tests | `scipy/sparse/tests/test_base.py:2613-2634` (`thread_unsafe`) | `splu` path fails in parallel harnesses | unsafe parallel factoring can regress determinism/correctness | isolate factorization state per thread or force serialized execution |
| FFT backend args on non-NumPy namespaces | `scipy/fft/_basic_backend.py:8-15`, `scipy/_lib/_array_api.py:390-392` | `workers`/`plan` unsupported outside NumPy backend | silently ignoring params would break compatibility | fail fast with explicit unsupported-parameter error |
| FFT kernel dtype/axes/norm | `scipy/fft/_pocketfft/pypocketfft.cxx:67-105` | unsupported dtype/axes/norm throws runtime/argument errors | coercion would change numerical route and observables | preserve explicit rejection boundaries |
| special alternative backend drift | `scipy/special/tests/test_support_alternative_backends.py:49-95` | multiple backend/dtype combos are xfail/skip by design | behavior is intentionally undefined or non-parity on some backend pairs | treat these combinations as unsupported in V1 and fail closed with compatibility note |
| special error policy concurrency | `scipy/special/tests/test_basic.py:4511-4522` | error-action changes require serialization discipline | concurrent mode switching can alter warning/error semantics | scope and serialize error-policy mutation; emit audit traces in hardened mode |

### 20.2 Explicit undefined zones (V1)

Undefined for compatibility guarantees unless explicitly allowlisted:
1. concurrent mutation of sparse DOK/COO containers;
2. FFT `workers`/`plan` on non-NumPy array-api backends;
3. backend/dtype combinations marked xfail/skip in special alternative-backend coverage;
4. concurrent global error-policy mutation for special-function error actions;
5. verbose BVP output across concurrent runs that share stdout channels.

Policy: undefined zones are fail-closed in strict mode and fail-closed-with-audit in hardened mode; every rejection must emit replay-grade fields from section 21.4 (`trace_id`, `mode`, `input_signature`, `artifact_refs`).

### 20.3 Hardened-mode rationale (source anchored)

Hardened mode exists to add bounded safety around known hazardous surfaces without changing primary API contracts:
- preserve explicit exception/status outputs (do not fabricate successful results),
- add structured diagnostics when rejecting undefined-zone calls,
- allow only bounded, deterministic recovery where legacy behavior is already warning-mediated (for example option-ignore warnings in optimize),
- capture every override/rejection as deterministic compatibility/security evidence.

### 20.4 DOC-PASS-08 closure evidence

1. edge-case register with source anchors and mitigation notes;
2. undefined-zone list with explicit strict/hardened policy;
3. compatibility/security rationale linking to threat and drift governance;
4. fixture requirements for each high-risk edge surface (thread safety, backend mismatch, unsupported params);
5. no unresolved drafting markers in pass output.

## 21. DOC-PASS-09: Unit/E2E Test Corpus and Logging Evidence Crosswalk

### 21.1 Behavior-to-test crosswalk (legacy anchors)

| Subsystem | Behavior under contract | Legacy test anchors | Evidence type |
|---|---|---|---|
| integrate (`solve_ivp` / solver lifecycle) | event ordering, status transitions, tolerance/event semantics | `scipy/integrate/_ivp/tests/test_ivp.py:572-699`, `scipy/integrate/_ivp/tests/test_ivp.py:640-699` | unit + integration |
| integrate (`solve_bvp`) | verbose/diagnostic path concurrency limits | `scipy/integrate/tests/test_bvp.py:690-709` (`thread_unsafe`) | integration edge |
| linalg (`solve/inv/det/lstsq/pinv`) | singular/ill-conditioned signaling and non-overwrite guarantees | `scipy/linalg/tests/test_basic.py:774-826`, `scipy/linalg/tests/test_basic.py:2529-2554` | unit + numerical contract |
| optimize (`tnc`, `linprog`) | callback visibility/order, status/message contracts, option warning behavior | `scipy/optimize/tests/test_tnc.py:304-345`, `scipy/optimize/tests/test_linprog.py:240-319`, `scipy/optimize/tests/test_linprog.py:271-284` | integration + API contract |
| sparse (COO/DOK/sparsetools) | mutation hazards, threaded low-level operations | `scipy/sparse/tests/test_coo.py:1212-1279`, `scipy/sparse/tests/test_dok.py:1-27`, `scipy/sparse/tests/test_sparsetools.py:24-57` | unit + thread-safety stress |
| fft (`set_workers`, backend behavior) | worker scoping and invalid worker handling | `scipy/fft/tests/test_multithreading.py:52-84` | integration + policy |
| special (`errstate`, alt backends) | policy-driven warn/raise behavior and backend drift boundaries | `scipy/special/tests/test_basic.py:4489-4523`, `scipy/special/tests/test_support_alternative_backends.py:49-110` | unit + compatibility edge |
| `_lib` callbacks/workers/utilities | callback signatures/thread safety, RNG/worker utility behavior | `scipy/_lib/tests/test_ccallback.py:92-194`, `scipy/_lib/tests/test__util.py:27-199` | unit + infrastructure |

### 21.2 E2E and artifact evidence crosswalk

FrankenSciPy artifact/e2e anchors currently available in-repo:
- packet parity reports and sidecars: `crates/fsci-conformance/src/lib.rs:739-768`,
- oracle capture artifact generation: `crates/fsci-conformance/src/lib.rs:581-643`,
- packet e2e scenario fixture shape: `crates/fsci-conformance/fixtures/artifacts/FSCI-P2C-001/e2e/scenarios/validate_tol_smoke.json:1`.

Required mapping rule per major behavior:
1. unit/property coverage (crate-local),
2. packet-level conformance case (`fsci-conformance`),
3. e2e scenario artifact,
4. parity + raptorq + decode-proof evidence.

### 21.3 Coverage gaps (explicit, prioritized)

P0 gaps:
1. callback TLS restore/stack-unwind evidence is not directly asserted by tests for integrate/_lib callback paths;
2. undefined-zone fail-closed diagnostics (concurrent sparse mutation, verbose BVP concurrency) are marked `thread_unsafe` but not paired with deterministic forensic logs.

P1 gaps:
1. non-NumPy backend failures for FFT `workers`/`plan` parameter usage need explicit cross-backend conformance cases;
2. special errstate concurrent mutation behavior lacks replay-grade logs describing policy transitions.

P2 gaps:
1. richer status/warning provenance for linalg/optimize failures (input signature + decision evidence) is not yet normalized across packets.

Tracked closure mapping for current known gaps:
1. P0 callback TLS and undefined-zone forensic-log gaps -> owner packet implementation beads `FSCI-P2C-*` with program-level verification in `bd-3jh.11`.
2. P1 FFT non-NumPy backend `workers`/`plan` behavior and errstate concurrency logs -> owner packet implementation beads `FSCI-P2C-*`; unresolved items escalate to `bd-3jh.11`.
3. P2 provenance normalization for linalg/optimize warning-vs-error paths -> owner `bd-3jh.11` readiness/sign-off gate after this doc integration pass.

### 21.4 Logging fields required for replay/forensics

Aligned with `docs/TEST_CONVENTIONS.md` structured logging and `fsci-runtime` test log model (`crates/fsci-runtime/src/lib.rs:574-589`), every major test/e2e execution should emit:
1. `trace_id`,
2. `test_id`,
3. `timestamp_ms`,
4. `module` / `entrypoint`,
5. `mode` (`strict`/`hardened`),
6. `seed` (if randomized),
7. `fixture_id`,
8. `input_signature` (dtype/shape/core args),
9. `tolerances`,
10. `solver_config` / backend options,
11. `status`,
12. `message`,
13. `warnings`,
14. `result`,
15. `artifact_refs` (parity, oracle, raptorq/decode-proof),
16. `timing_ms` and `memory_peak` for performance-sensitive paths.

### 21.5 DOC-PASS-09 closure evidence

1. each major documented behavior is mapped to concrete tests and evidence artifacts;
2. coverage gaps are explicit with priority labels;
3. replay/forensic logging fields are defined and aligned to existing project test-log contracts;
4. cross-references to prior doc passes (06/07/08) remain consistent.

## 22. DOC-PASS-05: Complexity, Performance, and Memory Characterization

### 22.1 Source-anchored complexity and memory profile

| Subsystem / operation | Complexity signal | Memory pressure signal | Source anchors |
|---|---|---|---|
| sparse bipartite matching (`maximum_bipartite_matching`) | Hopcroft-Karp `O(|E|*sqrt(|V|))` | linear in row count; row/column asymmetry can be reduced via transpose | `legacy_scipy_code/scipy/scipy/sparse/csgraph/_matching.pyx:47`, `legacy_scipy_code/scipy/scipy/sparse/csgraph/_matching.pyx:49` |
| sparse max flow (`maximum_flow`) | Edmonds-Karp `O(|V|*|E|^2)` vs Dinic `O(|V|^2*|E|)` | both listed as `O(|E|)` space | `legacy_scipy_code/scipy/scipy/sparse/csgraph/_flow.pyx:93`, `legacy_scipy_code/scipy/scipy/sparse/csgraph/_flow.pyx:95` |
| sparse assignment + assembly | sparse matching path explicitly avoids dense blow-up; `block_diag` improved from quadratic to linear | dense representation can fail to fit memory | `legacy_scipy_code/scipy/doc/source/release/1.6.0-notes.rst:183`, `legacy_scipy_code/scipy/doc/source/release/1.6.0-notes.rst:186`, `legacy_scipy_code/scipy/doc/source/release/1.6.0-notes.rst:189` |
| linalg sketching (`clarkson_woodruff_transform`) | sparse-aware sketch runs in `O(nnz(A))` | work scales with nonzeros instead of dense shape | `legacy_scipy_code/scipy/doc/source/release/1.3.0-notes.rst:59`, `legacy_scipy_code/scipy/doc/source/release/1.3.0-notes.rst:62` |
| optimize trust-region least squares (`least_squares`) | `'exact'` path is SVD-comparable per iteration; `'lsmr'` path uses matrix-vector products | dense Jacobian path is higher workspace pressure than sparse iterative path | `legacy_scipy_code/scipy/scipy/optimize/_lsq/least_squares.py:436`, `legacy_scipy_code/scipy/scipy/optimize/_lsq/least_squares.py:440`, `legacy_scipy_code/scipy/scipy/optimize/_lsq/least_squares.py:444` |
| optimize isotonic regression (`isotonic_regression`) | PAVA complexity is `O(N)` | linear-pass behavior avoids quadratic rescans | `legacy_scipy_code/scipy/scipy/optimize/_isotonic.py:75` |
| special Lambert W (`lambertw`) | Halley initialization uses `O(log(w))` or `O(w)` approximation | branch/initialization quality determines iteration count and transient work | `legacy_scipy_code/scipy/scipy/special/_lambertw.py:60`, `legacy_scipy_code/scipy/scipy/special/_lambertw.py:61` |
| integrate ZVODE step kernel | per-step state touches `y`, `yh`, `ewt`, `savf`, `vsav`, `acor`, `wm`, `iwm` | explicit vector/history/work-array inventory exposes dominant allocations | `legacy_scipy_code/scipy/scipy/integrate/src/zvode.c:1396`, `legacy_scipy_code/scipy/scipy/integrate/src/zvode.c:1404` |
| integrate LSODA dense output | dense output pulls order-dependent Nordsieck history from workspace offsets (`iwork`/`rwork`) | reshape+copy of `yh` plus step-order rescaling are memory-sensitive | `legacy_scipy_code/scipy/scipy/integrate/_ivp/lsoda.py:189`, `legacy_scipy_code/scipy/scipy/integrate/_ivp/lsoda.py:197`, `legacy_scipy_code/scipy/scipy/integrate/_ivp/lsoda.py:202` |
| FFT backend (`pypocketfft`) | Bluestein removes worst-case `O(n^2)` from legacy FFTPACK path | multithreaded transforms (`workers`) increase throughput but also peak working-set pressure | `legacy_scipy_code/scipy/doc/source/release/1.4.0-notes.rst:74`, `legacy_scipy_code/scipy/doc/source/release/1.4.0-notes.rst:77`, `legacy_scipy_code/scipy/doc/source/release/1.4.0-notes.rst:80` |
| `_lib` uarray conversion protocol | `coerce=false` conversion should ideally be `O(1)` | contract explicitly disallows memory copy in the zero-copy path | `legacy_scipy_code/scipy/scipy/_lib/_uarray/__init__.py:63`, `legacy_scipy_code/scipy/scipy/_lib/_uarray/__init__.py:65` |
| signal FFT convolution hot path | `fftn/ifftn` or `rfftn/irfftn` plus `next_fast_len` padding defines dominant runtime | padding ratio drives temporary array size; no automatic fallback to overlap-add | `docs/DOC_PASS_04_EXECUTION_PATH_TRACING.md:1641`, `docs/DOC_PASS_04_EXECUTION_PATH_TRACING.md:1644`, `docs/DOC_PASS_04_EXECUTION_PATH_TRACING.md:1672` |

### 22.2 Hotspot hypotheses (explicit and testable)

All hotspot validation follows the project optimization loop and gate requirements: baseline p50/p95/p99 + memory, one optimization lever at a time, parity proof, and delta artifact (`AGENTS.md:304`, `AGENTS.md:310`, `docs/planning/COMPREHENSIVE_SPEC_FOR_FRANKENSCIPY_V1.md:204`).

1. Sparse matching/flow kernels: runtime should scale with edge density, and dense fallback risk should be visible as memory spikes; benchmark sweeps must vary `|E|/|V|` and emit `timing_ms` + `memory_peak`.
2. `least_squares` crossover: `'exact'` should dominate only for small dense Jacobians; large/sparse regimes should favor `'lsmr'`; benchmark by Jacobian density and dimension.
3. LSODA dense output: Nordsieck extraction/copy cost should increase with solver order and state dimension; profile allocation hot paths around workspace reshape/copy.
4. FFT convolution: `next_fast_len` padding ratio should predict tail latency; compare `fftconvolve` vs overlap-add candidate thresholds while preserving observable API path.
5. `_lib` conversion fast path: `coerce=false` conversions should remain zero-copy; instrumentation should track whether conversion allocates/copies unexpectedly.

Measurement hook requirements for each hypothesis:
1. percentile latency (`p50`, `p95`, `p99`);
2. `memory_peak`;
3. structured trace identifiers (`trace_id`, `fixture_id`, `mode`) for replay linkage (DOC-PASS-09 section 21.4).

Artifacts must be emitted in the locked topology (`docs/ARTIFACT_TOPOLOGY.md:23`, `docs/ARTIFACT_TOPOLOGY.md:36`, `docs/ARTIFACT_TOPOLOGY.md:78`) with `.raptorq.json` + `.decode_proof.json` sidecars for long-lived performance evidence.

### 22.3 Optimization risk notes linked to parity constraints

| Optimization risk | Why it is dangerous | Required guard/evidence |
|---|---|---|
| solver simplification that weakens tolerance/stability behavior | violates documented doctrine that numerical stability outranks speed | reject unless conformance + invariant evidence proves no tolerance drift (`docs/TEST_CONVENTIONS.md:510`, `AGENTS.md:318`) |
| skipping baseline/profile/re-baseline loop | hides regression tails and memory blowups | enforce mandatory loop artifacts for every optimization (`AGENTS.md:308`, `AGENTS.md:314`) |
| perf-only change that alters observable semantics | fails Acceptance Gate C requirement of no semantic regression | require Gate C evidence before closure (`docs/planning/COMPREHENSIVE_SPEC_FOR_FRANKENSCIPY_V1.md:204`) |
| removing or weakening CASP stiffness handling for speed | can cause effective hangs on stiff systems | require threat-matrix mitigation evidence for switch/certification behavior (`docs/threat_matrix.json:102`, `docs/threat_matrix.json:108`) |
| changing FFT fallback/dispatch semantics without explicit policy | may alter user-visible backend behavior and numerical route | prove parity on strict mode and document any hardened-only divergence with tests (`docs/DOC_PASS_04_EXECUTION_PATH_TRACING.md:1672`) |
| missing parity/benchmark artifact sidecars | breaks auditable recovery and topology lock | emit `.raptorq.json` and `.decode_proof.json` for long-lived artifacts and enforce an automated existence check in packet closure gates (`docs/ARTIFACT_TOPOLOGY.md:62`, `docs/ARTIFACT_TOPOLOGY.md:67`) |

### 22.4 DOC-PASS-05 closure evidence

1. complexity and memory characterization includes major operations with concrete legacy/project anchors;
2. hotspot hypotheses are explicit, measurable, and tied to required percentile/memory metrics;
3. optimization risks are mapped directly to parity, threat, and acceptance-gate constraints;
4. artifact routing requirements are linked to the topology lock and durability contract.

## 23. DOC-PASS-11: Expansion Draft B (Behavioral and Analytical Synthesis)

### 23.1 Integrated behavior-risk-evidence map (packet-facing)

| Domain family | Core behavior contract (legacy anchor) | Primary risk surface | Required evidence linkage |
|---|---|---|---|
| integrate (`solve_ivp` + solver lifecycle) | step/status/event ordering and message semantics (`scipy/integrate/_ivp/base.py:179-208`, `scipy/integrate/_ivp/ivp.py:657-759`) | stiff-system stall and callback lifecycle misuse (`docs/threat_matrix.json:102`, `scipy/integrate/_odepackmodule.c:83`) | differential conformance fixtures + lifecycle logs + stiffness detection traces (sections 18, 21, 22) |
| linalg (`solve/inv/det/lstsq/pinv`) | warning/error split and singularity signaling (`scipy/linalg/_basic.py:28-55`, `scipy/linalg/_basic.py:620-652`) | silent numerical drift under aggressive optimization | parity report + invariant checklist + benchmark delta artifact (`AGENTS.md:320`, `AGENTS.md:324`) |
| optimize (root/minimize/linprog family) | callback/status/message contracts (`scipy/optimize/_optimize.py:48-58`, `scipy/optimize/tests/test_tnc.py:321-341`) | option coercion, warning suppression, and convergence false-positives | unit/integration callback coverage + structured warning provenance logs (sections 19, 21) |
| sparse (format + solve + csgraph) | format invariants, lazy namespace behavior, singular solve failure signaling (`scipy/sparse/__init__.py:307`, `scipy/sparse/linalg/_dsolve/linsolve.py:738-743`) | concurrent mutation hazards and memory blow-up from densification | thread-safety fixtures + fail-closed diagnostics + memory-tail benchmarks (sections 18, 20, 22) |
| fft | backend precedence, worker scoping, transform dispatch (`scipy/fft/_backend.py:52`, `scipy/fft/tests/test_multithreading.py:62-90`) | backend mismatch and unsupported parameter zones (`workers`, `plan`) | cross-backend conformance cases + strict/hardened error surface checks + tail-latency logs (sections 20, 21, 22) |
| special + `_lib` | policy-driven warning/raise behavior and backend overlays (`scipy/special/_ufuncs_extra_code.pxi:24-117`, `scipy/special/_support_alternative_backends.py:8-16`) | concurrent policy mutation and undefined backend combos | errstate policy fixtures + compatibility drift ledger + replay-grade structured logs (sections 19, 20, 21) |

### 23.2 Pass-B synthesis rules for implementation and review

1. No optimization-only change is admissible without behavior-isomorphism proof:
   baseline -> profile -> one lever -> conformance+invariants -> re-baseline (`AGENTS.md:308-314`).
2. Gate coupling is mandatory for all packet closures:
   Gate A (parity), Gate B (security/adversarial), Gate C (performance without semantic drift), Gate D (RaptorQ durability) (`docs/planning/COMPREHENSIVE_SPEC_FOR_FRANKENSCIPY_V1.md:200-208`).
3. Structured logging is part of correctness evidence:
   major test/e2e paths must emit replay fields listed in section 21.4 (including `trace_id`, `mode`, `artifact_refs`, `timing_ms`, `memory_peak`).
4. Artifact topology is locked:
   parity/benchmark outputs and sidecars must remain in governed locations and schemas (`docs/ARTIFACT_TOPOLOGY.md:23`, `docs/ARTIFACT_TOPOLOGY.md:62`, `docs/ARTIFACT_TOPOLOGY.md:78`).

### 23.3 Downstream handoff contract (unblocking deep-dive passes)

Pass B defines the minimum analytical substrate for:
1. `bd-3jh.23.13` (DOC-PASS-12 red-team contradiction/completeness review): must validate all cross-pass assertions against the integrated map above.
2. `bd-3jh.23.16` (behavior specialist deep dive): must extend behavior contracts with per-function witness fixtures and mismatch taxonomy.
3. `bd-3jh.23.17` (risk/perf/test specialist deep dive): must validate threat/perf/logging assumptions with packet-specific evidence artifacts.

Each downstream pass must explicitly reuse:
- concurrency/lifecycle contracts (section 18),
- failure/recovery taxonomy (section 19),
- edge/undefined-zone policy (section 20),
- test/logging crosswalk (section 21),
- complexity/performance characterization (section 22).

### 23.4 DOC-PASS-11 closure evidence

1. analysis draft integrates behavior, risk, performance, and evidence requirements into one coherent contract;
2. behavior/risk claims remain source-anchored to legacy code and project governance artifacts;
3. test and logging crosswalk requirements are embedded as first-class completion criteria;
4. downstream review/deep-dive passes have explicit handoff prerequisites and dependency links.

## 24. DOC-PASS-12: Independent Red-Team Contradiction and Completeness Review

### 24.1 Contradiction register (resolved)

| ID | Issue found | Resolution applied | Evidence |
|---|---|---|---|
| RT-01 | spec-gap statement claimed missing sections `14-20` although those sections exist | section 1 updated to reflect current spec status and focus on empirical validation, not missing structure | `docs/planning/COMPREHENSIVE_SPEC_FOR_FRANKENSCIPY_V1.md:232`, `EXHAUSTIVE_LEGACY_ANALYSIS.md:28` |
| RT-02 | canonical legacy root path differed between docs (`/data/projects/...` vs `/dp/...`) | `EXISTING_SCIPY_STRUCTURE.md` root normalized to `/data/projects/frankenscipy/legacy_scipy_code/scipy` | `EXISTING_SCIPY_STRUCTURE.md:5`, `EXHAUSTIVE_LEGACY_ANALYSIS.md:18` |
| RT-03 | pass/bead linkage was easy to misread across pass-11/pass-12 handoff statements | pass label for `bd-3jh.23.13` is now explicit as DOC-PASS-12 wherever referenced in handoff narrative | `EXHAUSTIVE_LEGACY_ANALYSIS.md:599`, `EXISTING_SCIPY_STRUCTURE.md:390` |

### 24.2 Unsupported-claim cleanup

1. claims that were previously broad or implicit are now either:
   - directly source-anchored, or
   - converted into bounded uncertainty items with explicit owners.
2. no unresolved drafting markers remain in either doc.

### 24.3 Remaining bounded uncertainty (explicit)

1. full array-api experimental coverage across all scoped public functions remains open and is owned by readiness/sign-off gate (`bd-3jh.11`);
2. additional lazy-import edges outside `scipy.sparse` remain a bounded structural review item for readiness sign-off (`bd-3jh.11`).

### 24.4 DOC-PASS-12 closure evidence

1. contradictions and ambiguities were listed and resolved in a traceable register;
2. unsupported claims were removed, narrowed, or evidenced;
3. remaining uncertainty is explicit, bounded, and assigned to follow-up owner work.

## 25. DOC-PASS-15: Behavior Specialist Deep-Pass Findings

### 25.1 Behavioral ambiguities resolved

| ID | Ambiguous behavior area | Clarification applied | Evidence |
|---|---|---|---|
| BP-01 | IVP event ordering semantics were underspecified | event evaluation order and termination sequencing made explicit in section 18.2 + invariant `FSCI-C6` | `scipy/integrate/_ivp/ivp.py:640-694` |
| BP-02 | integrate status/message dual-channel wording was too loose | section 19.3 now requires deterministic, aligned status/message pairing | `scipy/integrate/_ivp/ivp.py:657-759` |
| BP-03 | linalg input mutability expectations were implicit | invariant `FSCI-C7` now codifies default non-overwrite behavior unless opt-in overwrite flags are used | `scipy/linalg/tests/test_basic.py:774-826`, `scipy/linalg/tests/test_basic.py:2529-2554` |
| BP-04 | sparse mutation hazard policy did not define strict/hardened outcomes precisely | section 20.1 now specifies strict fail-fast vs hardened serialized+audited handling | `scipy/sparse/tests/test_coo.py:1212-1275` |
| BP-05 | FFT/special scoping invariants were present but not formalized | added invariants `FSCI-C9` and `FSCI-C10` for scope unwind and errstate restoration | `scipy/fft/tests/test_multithreading.py:62-90`, `scipy/special/tests/test_basic.py:4511-4522` |

### 25.2 Invariant narrative updates

Added deep-pass invariants: `FSCI-C6` through `FSCI-C10` to ensure behavior narratives are executable, testable, and mode-split explicit.

### 25.3 Bounded behavior uncertainty

The claim that non-NumPy array-api backends always reject FFT `workers`/`plan` remains bounded uncertainty until backend-specific coverage is expanded across all scoped namespaces.

### 25.4 DOC-PASS-15 closure evidence

1. behavioral deep-pass findings are merged and traceable;
2. ambiguous behavior sections were clarified with explicit invariants;
3. invariant narratives were validated/corrected with concrete anchors;
4. remaining uncertainty is bounded and explicitly scoped.

## 26. DOC-PASS-16: Risk/Performance/Test Specialist Deep-Pass Findings

### 26.1 Tightened risk/perf/test controls

| ID | Issue class | Tightening applied | Evidence |
|---|---|---|---|
| RP-01 | drift budget enforcement lacked explicit test-gate wiring | section 17 now requires log/perf evidence verification before packet completion | sections 14, 17, 21.4 |
| RP-02 | undefined-zone fail-closed policy lacked replay-log requirement | section 20.2 now mandates structured rejection logs with section 21.4 fields | sections 20.2, 21.4 |
| RP-03 | P0/P1/P2 coverage gaps lacked owner mapping | section 21.3 now includes tracked closure ownership and escalation path | section 21.3 |
| RP-04 | hotspot metrics were not explicitly tied to durability sidecars | section 22.2 now binds performance evidence to RaptorQ/decode-proof envelope | sections 9, 22.2 |
| RP-05 | sidecar risk row lacked explicit automated verification requirement | section 22.3 now requires automated sidecar existence checks as gate input | sections 22.3, 17 |

### 26.2 Risk taxonomy extension (explicit)

Add to packet-level failure review scope:
1. CASP/solver-selection misrouting risk (stale or insufficient conditioning diagnostics) must be tracked with runtime decision logs and mitigation notes.
2. Any uninstrumented high-severity risk is treated as unresolved and blocks readiness/sign-off (`bd-3jh.11`).

### 26.3 Remaining tracked follow-ups

1. backend-specific evidence for non-NumPy FFT `workers`/`plan` behavior remains open until dedicated cross-backend conformance cases are landed;
2. callback TLS and undefined-zone forensic log validation still require packet-level implementation evidence beyond doc-level specification.

### 26.4 DOC-PASS-16 closure evidence

1. risk/perf/test deep-pass output is merged into normative sections;
2. missing unit/e2e/logging mappings are either resolved or explicitly tracked;
3. risk taxonomy and mitigation notes are tightened with gate-coupled requirements.

## 27. DOC-PASS-13 Final Integrated Rewrite Sign-Off

### 27.1 Integration matrix (passes 01-12 + deep passes)

Integrated in this document:
1. DOC-PASS-05 complexity/perf/memory characterization (section 22);
2. DOC-PASS-06 concurrency/lifecycle ordering (section 18);
3. DOC-PASS-07 error taxonomy/failure recovery (section 19);
4. DOC-PASS-08 security/compatibility edge zones (section 20);
5. DOC-PASS-09 test/e2e/logging crosswalk (section 21);
6. DOC-PASS-11 expansion draft B synthesis (section 23);
7. DOC-PASS-12 red-team contradiction/completeness register (section 24);
8. DOC-PASS-15 behavior specialist deep pass (section 25);
9. DOC-PASS-16 risk/perf/test specialist deep pass (section 26).

Integrated cross-document dependencies:
1. DOC-PASS-10 structure expansion draft A in `EXISTING_SCIPY_STRUCTURE.md` section 14;
2. DOC-PASS-14 structure specialist review in `EXISTING_SCIPY_STRUCTURE.md` section 15.

### 27.2 Final consistency sweep checks

1. cross-document root path consistency verified (`/data/projects/frankenscipy/legacy_scipy_code/scipy`);
2. contradictory spec-status statement removed and replaced with current-status wording;
3. pass/bead references normalized for `bd-3jh.23.13` (DOC-PASS-12);
4. bounded uncertainty items are explicit and owner-mapped;
5. no unresolved drafting markers detected.

### 27.3 Sign-off

- Date: `2026-02-14`
- Reviewer: `PlumOwl`
- Verdict: `PASS` for DOC-PASS-13 integrated rewrite and consistency gate.

# EXISTING_SCIPY_STRUCTURE

## 1. Legacy Oracle

- Root: /data/projects/frankenscipy/legacy_scipy_code/scipy
- Upstream: scipy/scipy

## 2. Subsystem Map

- scipy/integrate: IVP and integration routines.
- scipy/linalg: dense algebra wrappers over BLAS/LAPACK.
- scipy/optimize: root/minimize ecosystem with MINPACK and newer methods.
- scipy/sparse (+ sparse/linalg): sparse structures and solver paths.
- scipy/special: special-function wrappers and error handling.
- scipy/fft with subprojects/pocketfft: transform backend and wrappers.
- scipy/_lib: array-api negotiation, callbacks, utility internals.
- scipy/subprojects: bundled third-party numeric runtimes.

## 3. Semantic Hotspots (Must Preserve)

1. solve_ivp adaptive stepping, event handling, and tolerance scaling.
2. linalg factorization workspace/pivot semantics and error signaling.
3. optimize root/minpack option handling and status interpretation.
4. special-function error state controls and propagation behavior.
5. sparse array/matrix dual semantics and index dtype constraints.
6. array API backend negotiation behavior in `_lib`, including global `SCIPY_ARRAY_API` gating.

## 4. Compatibility-Critical Behaviors

- API-level option dictionaries and result shapes for optimize/integrate/linalg routines.
- tolerance and convergence semantics for scoped solver families.
- sparse constructor and operator behavior across formats.
- backend namespace compatibility expectations.

## 5. Security and Stability Risk Areas

- BLAS/LAPACK and Fortran wrapper memory-safety assumptions.
- special-function wrappers with mixed C/Cython/Fortran layers.
- third-party subproject integration and thread/memory semantics.
- callback lifecycle correctness in _lib callback helpers.

## 6. V1 Extraction Boundary

Include now:
- integrate/linalg/optimize/sparse/fft scoped families and array-api helpers needed for them.

Exclude for V1:
- full breadth modules (stats/spatial/ndimage/io etc.), docs/tooling/bench datasets, full third-party replacement breadth.

## 7. High-Value Conformance Fixture Families

- integrate/_ivp/tests for solver and event behavior.
- linalg/tests for decomposition and low-level wrappers.
- optimize/tests for method option and convergence contracts.
- fft/tests for backend and transform parity.
- sparse/tests and _lib/tests for structure and utility parity.

## 8. Extraction Notes for Rust Spec

- Maintain per-algorithm tolerance contracts explicitly.
- Prioritize deterministic convergence/error semantics over speed.
- Use differential fixture bundles before introducing deep optimization.

## 9. DOC-PASS-06 Concurrency/Lifecycle and Ordering Map

### 9.1 Lifecycle ownership boundaries

- integrate callback state is thread-local during native solver execution and is cleared on solver return:
  - `scipy/integrate/_dopmodule.c:21-214`
  - `scipy/integrate/_odepackmodule.c:71-118`
- `_lib` callback dispatch uses thread-local active callback plus LIFO restore for nested calls:
  - `scipy/_lib/src/ccallback.h:67-300`
- special-function integration helpers keep callback payload memory alive for full call lifetime:
  - `scipy/special/_ellip_harm_2.pyx:103-133`

### 9.2 Ordering guarantees to preserve

- IVP status transition and step sequencing:
  - `scipy/integrate/_ivp/base.py:179-208`
  - `scipy/integrate/_ivp/ivp.py:657-759`
- IVP event detection and termination ordering:
  - `scipy/integrate/_ivp/ivp.py:640-694`
- FFT backend precedence and scoped override semantics:
  - `scipy/fft/_backend.py:52-209`
- FFT worker context nesting/restore semantics:
  - `scipy/fft/_pocketfft/helper.py:208-237`
  - `scipy/fft/tests/test_multithreading.py:62-90`
- optimize callback visibility/order guarantees:
  - `scipy/optimize/_linprog_rs.py:300-360`
  - `scipy/optimize/tests/test_tnc.py:321-341`

### 9.3 Known race/regression surfaces

- sparse mutation APIs with explicit thread-unsafe markers:
  - `scipy/sparse/tests/test_coo.py:1233-1279`
- sparse low-level threaded call stress:
  - `scipy/sparse/tests/test_sparsetools.py:28-57`
- process/thread interplay and threadpool capping:
  - `scipy/conftest.py:108-153`
  - `scipy/_lib/_util.py:576-677`
- special error-state mutation requires serialization discipline:
  - `scipy/special/tests/test_basic.py:4511-4522`

### 9.4 FrankenSciPy policy outcome

- strict mode: fail closed or serialize on thread-unsafe mutation surfaces;
- hardened mode: same behavior plus structured audit traces;
- runtime must preserve callback TLS discipline, scope restoration, and backend/worker ordering as compatibility-critical behavior.

## 10. DOC-PASS-07 Error Taxonomy and Recovery Map

### 10.1 Canonical failure families

- input/shape/dtype/domain contract failures:
  - integrate: `scipy/integrate/_ivp/base.py:5-21`, `scipy/integrate/_ivp/common.py:10-58`
  - linalg: `scipy/linalg/_basic.py:199-243`
  - sparse: `scipy/sparse/linalg/_dsolve/linsolve.py:712-719`
  - special: `scipy/special/_basic.py:230-318`, `scipy/special/_basic.py:2880-2986`
  - `_lib`: `scipy/_lib/_util.py:360-430`, `scipy/_lib/_util.py:445-460`, `scipy/_lib/_util.py:549-574`
- numerical breakdown / singularity:
  - integrate step infeasibility: `scipy/integrate/_ivp/bdf.py:310-359`
  - linalg singular/illegal LAPACK states: `scipy/linalg/_basic.py:28-55`, `scipy/linalg/_basic.py:620-652`
  - sparse singular solve: `scipy/sparse/linalg/_dsolve/linsolve.py:738-743`
- unsupported capability paths:
  - FFT plan unsupported: `scipy/fft/_pocketfft/basic.py:11-133`, `scipy/fft/_pocketfft/basic.py:157-219`
  - HiGHS callback unsupported: `scipy/optimize/_linprog.py:653-657`
- policy-mediated warning/exception behavior:
  - special `ignore|warn|raise` actions: `scipy/special/_ufuncs_extra_code.pxi:24-117`
  - optimize compatibility warnings: `scipy/optimize/_linprog.py:634-652`

### 10.2 User-visible signaling that must be preserved

- integrate: status/message semantics (`scipy/integrate/_ivp/ivp.py:657-759`)
- optimize: canonical status/message vocabulary (`scipy/optimize/_optimize.py:48-58`)
- linalg: hard-error vs warning split for batched low-level outcomes (`scipy/linalg/_basic.py:28-55`)
- special: configured warning/exception policy dispatch (`scipy/special/_ufuncs_extra_code.pxi:24-117`)

### 10.3 FrankenSciPy recovery policy mapping

- strict mode:
  - preserve legacy-observable exception class, warning/raise boundary, and status/message outputs;
  - no silent fallback that changes observable semantics.
- hardened mode:
  - allow bounded defensive recovery only when contract permits it;
  - emit deterministic audit traces for every recovery or policy override.

## 11. DOC-PASS-08 Security/Compatibility Edge Zones

### 11.1 High-risk edge surfaces

- integrate callback lifecycle and shared runtime state:
  - `scipy/integrate/_odepackmodule.c:83`
  - `scipy/integrate/_odepackmodule.c:464-466`
- integrate verbose BVP output is thread-unsafe due to shared stdout handling:
  - `scipy/integrate/tests/test_bvp.py:690-709`
- sparse mutable formats are thread-unsafe for concurrent mutation:
  - `scipy/sparse/tests/test_dok.py:1-27`
  - `scipy/sparse/tests/test_coo.py:1212-1236`
  - `scipy/sparse/tests/test_coo.py:1264`
  - `scipy/sparse/tests/test_coo.py:1275`
- sparse LU parallel instability coverage:
  - `scipy/sparse/tests/test_base.py:2613-2634`
- FFT unsupported backend parameter zones (`workers`, `plan`) on non-NumPy namespaces:
  - `scipy/fft/_basic_backend.py:8-15`
  - `scipy/_lib/_array_api.py:390-392`
- FFT unsupported dtype/axes/norm combinations:
  - `scipy/fft/_pocketfft/pypocketfft.cxx:67-105`
- special alternative backend compatibility drift (skip/xfail zones):
  - `scipy/special/tests/test_support_alternative_backends.py:49-95`
- special error-policy mutation concurrency sensitivity:
  - `scipy/special/tests/test_basic.py:4511-4522`

### 11.2 Explicit undefined zones for V1 compatibility

- concurrent writes to DOK/COO sparse structures;
- backend/dtype combinations explicitly skipped/xfail in special alt-backend tests;
- non-NumPy backend calls with NumPy-only FFT parameters (`workers`, `plan`);
- concurrent global mutation of special error-policy state;
- concurrent verbose BVP output capture paths.

### 11.3 Policy mapping

- strict mode:
  - fail closed for undefined zones and preserve canonical error/status/warning surfaces.
- hardened mode:
  - same fail-closed baseline plus deterministic audit evidence and bounded defensive diagnostics.

## 12. DOC-PASS-09 Test/E2E and Logging Crosswalk

### 12.1 Major behavior to legacy test map

- integrate lifecycle/events:
  - `scipy/integrate/_ivp/tests/test_ivp.py:572-699`
  - `scipy/integrate/_ivp/tests/test_ivp.py:640-699`
- integrate verbose concurrency edge:
  - `scipy/integrate/tests/test_bvp.py:690-709`
- linalg error/warning and non-overwrite behavior:
  - `scipy/linalg/tests/test_basic.py:774-826`
  - `scipy/linalg/tests/test_basic.py:2529-2554`
- optimize callback/status/option semantics:
  - `scipy/optimize/tests/test_tnc.py:304-345`
  - `scipy/optimize/tests/test_linprog.py:240-319`
  - `scipy/optimize/tests/test_linprog.py:271-284`
- sparse mutation/thread-safety surfaces:
  - `scipy/sparse/tests/test_coo.py:1212-1279`
  - `scipy/sparse/tests/test_dok.py:1-27`
  - `scipy/sparse/tests/test_sparsetools.py:24-57`
- fft worker policy and multithreading checks:
  - `scipy/fft/tests/test_multithreading.py:52-84`
- special errstate and alt-backend drift:
  - `scipy/special/tests/test_basic.py:4489-4523`
  - `scipy/special/tests/test_support_alternative_backends.py:49-110`
- `_lib` callback and utility contracts:
  - `scipy/_lib/tests/test_ccallback.py:92-194`
  - `scipy/_lib/tests/test__util.py:27-199`

### 12.2 FrankenSciPy evidence crosswalk

- packet conformance execution and report artifacts:
  - `crates/fsci-conformance/src/lib.rs:548-575`
  - `crates/fsci-conformance/src/lib.rs:739-768`
- oracle capture/evidence:
  - `crates/fsci-conformance/src/lib.rs:581-643`
- e2e scenario artifact pattern:
  - `crates/fsci-conformance/fixtures/artifacts/FSCI-P2C-001/e2e/scenarios/validate_tol_smoke.json:1`

### 12.3 Priority coverage gaps

- P0:
  - callback TLS restore and nested callback unwind are not yet directly asserted with replay-grade logs.
  - thread-unsafe undefined-zone failures are identified but lack deterministic forensic payload capture.
- P1:
  - explicit FFT non-NumPy `workers`/`plan` fail-closed conformance cases are missing.
  - concurrent special errstate mutation behavior needs dedicated replay-aware coverage.
- P2:
  - normalized provenance logs for linalg/optimize warning-vs-error decisions need broader packet coverage.

### 12.4 Required replay/forensic logging fields

Use project logging contract from `docs/TEST_CONVENTIONS.md` and runtime model in `crates/fsci-runtime/src/lib.rs:574-589`, including:
- `trace_id`, `test_id`, `timestamp_ms`, `module`, `mode`, `seed`, `fixture_id`,
- `input_signature`, `tolerances`, `solver_config`,
- `status`, `message`, `warnings`, `result`,
- `artifact_refs`, `timing_ms`, `memory_peak`.

## 13. DOC-PASS-05 Complexity/Performance/Memory Map

### 13.1 Major complexity and memory surfaces

- sparse graph kernels:
  - Hopcroft-Karp matching: `O(|E|*sqrt(|V|))` and row-linear space (`legacy_scipy_code/scipy/scipy/sparse/csgraph/_matching.pyx:47`).
  - max-flow algorithms: Edmonds-Karp `O(|V|*|E|^2)` vs Dinic `O(|V|^2*|E|)` with `O(|E|)` space (`legacy_scipy_code/scipy/scipy/sparse/csgraph/_flow.pyx:95`).
  - sparse assignment and assembly memory scaling notes, including `block_diag` quadratic-to-linear improvement (`legacy_scipy_code/scipy/doc/source/release/1.6.0-notes.rst:183`, `legacy_scipy_code/scipy/doc/source/release/1.6.0-notes.rst:189`).
- linalg:
  - sparse-aware Clarkson-Woodruff sketch at `O(nnz(A))` (`legacy_scipy_code/scipy/doc/source/release/1.3.0-notes.rst:62`).
- optimize:
  - trust-region solver split: `'exact'` SVD-like dense cost vs `'lsmr'` iterative matrix-vector route (`legacy_scipy_code/scipy/scipy/optimize/_lsq/least_squares.py:440`, `legacy_scipy_code/scipy/scipy/optimize/_lsq/least_squares.py:444`).
  - isotonic regression PAVA linear-time guarantee (`legacy_scipy_code/scipy/scipy/optimize/_isotonic.py:75`).
- integrate:
  - ZVODE and LSODA expose explicit order/workspace-sensitive memory behavior (`legacy_scipy_code/scipy/scipy/integrate/src/zvode.c:1396`, `legacy_scipy_code/scipy/scipy/integrate/_ivp/lsoda.py:197`).
- FFT/signal:
  - pypocketfft Bluestein path removes historical `O(n^2)` worst case and enables worker-based threading (`legacy_scipy_code/scipy/doc/source/release/1.4.0-notes.rst:77`, `legacy_scipy_code/scipy/doc/source/release/1.4.0-notes.rst:80`).
  - convolution hot path still depends on `next_fast_len` padding and does not auto-switch to overlap-add (`docs/DOC_PASS_04_EXECUTION_PATH_TRACING.md:1644`, `docs/DOC_PASS_04_EXECUTION_PATH_TRACING.md:1672`).
- infrastructure (`_lib`):
  - uarray conversion contract expects `coerce=false` conversions to be effectively zero-copy `O(1)` (`legacy_scipy_code/scipy/scipy/_lib/_uarray/__init__.py:63`).

### 13.2 Hotspot hypotheses and measurement requirements

- hypothesis H1: sparse kernels exhibit nonlinear tail growth as edge density rises; validate by graph-size sweeps and percentiles.
- hypothesis H2: `least_squares` has a density/dimension crossover where `'lsmr'` dominates `'exact'`; validate with Jacobian sparsity matrix benchmarks.
- hypothesis H3: LSODA dense-output extraction has order-sensitive memory spikes due to Nordsieck copy/reshape; validate with allocation profiling by order/state dimension.
- hypothesis H4: FFT convolution tails are strongly correlated with padding ratio from `next_fast_len`; validate by shape families with fixed logical output size.
- hypothesis H5: zero-copy conversion paths in `_lib` regress when unintended coercion/copy occurs; validate with allocation counters and mode-tagged logs.

For every hypothesis, record `p50/p95/p99` + `memory_peak` and emit artifacts under locked topology paths (`AGENTS.md:310`, `docs/ARTIFACT_TOPOLOGY.md:23`, `docs/ARTIFACT_TOPOLOGY.md:78`).

### 13.3 Optimization guardrails (parity-first)

- numerical stability outranks speed; performance-only wins are invalid if they alter tolerance/conditioning behavior (`docs/TEST_CONVENTIONS.md:510`).
- optimization workflow is mandatory: baseline -> profile -> one lever -> parity proof -> re-baseline (`AGENTS.md:308`).
- acceptance Gate C requires performance budgets with no semantic regressions (`docs/planning/COMPREHENSIVE_SPEC_FOR_FRANKENSCIPY_V1.md:204`).
- stiff-system handling is not optional; optimization cannot bypass CASP switching/certification expectations (`docs/threat_matrix.json:102`, `docs/threat_matrix.json:108`).
- long-lived performance/conformance artifacts must retain durability sidecars (`docs/ARTIFACT_TOPOLOGY.md:62`, `docs/ARTIFACT_TOPOLOGY.md:67`).

## 14. DOC-PASS-10 Expansion Draft A (Structure Topology, Ownership, Dependencies)

### 14.1 Legacy module topology and ownership map

| Legacy surface | Structural role in SciPy | Key export/dispatch anchors | FrankenSciPy ownership target |
|---|---|---|---|
| `scipy.integrate` | orchestrates quadrature + ODE/IVP façade; keeps deprecated solver namespaces visible | `legacy_scipy_code/scipy/scipy/integrate/__init__.py:103`, `legacy_scipy_code/scipy/scipy/integrate/__init__.py:108`, `legacy_scipy_code/scipy/scipy/integrate/__init__.py:116` | `crates/fsci-integrate` (`Cargo.toml:4`) |
| `scipy.linalg` | dense linear algebra umbrella over decomposition, BLAS/LAPACK, sketches, solvers | `legacy_scipy_code/scipy/scipy/linalg/__init__.py:203`, `legacy_scipy_code/scipy/scipy/linalg/__init__.py:214`, `legacy_scipy_code/scipy/scipy/linalg/__init__.py:220` | `crates/fsci-linalg` (`Cargo.toml:5`) |
| `scipy.optimize` | method multiplexer across root/minimize/linear/nonlinear/global pipelines | `legacy_scipy_code/scipy/scipy/optimize/__init__.py:422`, `legacy_scipy_code/scipy/scipy/optimize/__init__.py:435`, `legacy_scipy_code/scipy/scipy/optimize/__init__.py:448` | `crates/fsci-opt` (`Cargo.toml:6`) |
| `scipy.sparse` + `scipy.sparse.linalg` | sparse format constructors + lazy submodule import + sparse solver stack | `legacy_scipy_code/scipy/scipy/sparse/__init__.py:307`, `legacy_scipy_code/scipy/scipy/sparse/__init__.py:328`, `legacy_scipy_code/scipy/scipy/sparse/__init__.py:340`, `legacy_scipy_code/scipy/scipy/sparse/linalg/__init__.py:132` | `crates/fsci-sparse` (`Cargo.toml:7`) |
| `scipy.fft` | transform façade + backend routing controls + worker management | `legacy_scipy_code/scipy/scipy/fft/__init__.py:86`, `legacy_scipy_code/scipy/scipy/fft/__init__.py:95`, `legacy_scipy_code/scipy/scipy/fft/__init__.py:97`, `legacy_scipy_code/scipy/scipy/fft/_backend.py:1`, `legacy_scipy_code/scipy/scipy/fft/_backend.py:52` | `crates/fsci-fft` (`Cargo.toml:8`) |
| `scipy.special` | ufunc/core wrappers with alternative-backend overlays and error-policy surface | `legacy_scipy_code/scipy/scipy/special/__init__.py:783`, `legacy_scipy_code/scipy/scipy/special/__init__.py:793`, `legacy_scipy_code/scipy/scipy/special/_support_alternative_backends.py:8` | `crates/fsci-special` (`Cargo.toml:9`) |
| `scipy._lib` array-api/uarray | backend capability policy + protocol-based dispatch substrate used by domain modules | `legacy_scipy_code/scipy/scipy/_lib/_array_api.py:24`, `legacy_scipy_code/scipy/scipy/_lib/_array_api.py:702`, `legacy_scipy_code/scipy/scipy/_lib/_uarray/__init__.py:63` | `crates/fsci-arrayapi` (`Cargo.toml:10`) + shared runtime policy support in `crates/fsci-runtime` (`Cargo.toml:12`) |

### 14.2 Dependency and control-flow boundaries (explicit)

1. Domain facades (`integrate`, `linalg`, `optimize`, `sparse`, `fft`, `special`) aggregate many internal modules and own the public API frontier.
2. Backend routing is explicit infrastructure, not algorithm code:
   - `fft` backend functions are exported at top level (`legacy_scipy_code/scipy/scipy/fft/__init__.py:95`) and implemented via uarray-backed controller (`legacy_scipy_code/scipy/scipy/fft/_backend.py:1`, `legacy_scipy_code/scipy/scipy/fft/_backend.py:52`).
3. Sparse namespace has mixed static + lazy structure:
   - core format modules are imported eagerly (`legacy_scipy_code/scipy/scipy/sparse/__init__.py:307`),
   - heavy submodules (`csgraph`, `linalg`) are loaded lazily through `__getattr__` (`legacy_scipy_code/scipy/scipy/sparse/__init__.py:328`, `legacy_scipy_code/scipy/scipy/sparse/__init__.py:340`).
4. Special-function namespace overlays backends after core ufunc/basic imports, meaning backend behavior is an augmentation layer, not the primary implementation (`legacy_scipy_code/scipy/scipy/special/__init__.py:785`, `legacy_scipy_code/scipy/scipy/special/__init__.py:793`).
5. Array API capability tables and skip/xfail policy metadata are centralized in `_lib` (`legacy_scipy_code/scipy/scipy/_lib/_array_api.py:686`, `legacy_scipy_code/scipy/scipy/_lib/_array_api.py:702`), so ownership of backend compatibility policy belongs in shared infrastructure rather than per-domain crates.
6. Array API dispatch has an explicit global gate:
   - `SCIPY_ARRAY_API` determines whether strict backend negotiation runs or the NumPy-compat namespace is returned directly (`legacy_scipy_code/scipy/scipy/_lib/_array_api_override.py:27`, `legacy_scipy_code/scipy/scipy/_lib/_array_api_override.py:86`, `legacy_scipy_code/scipy/scipy/_lib/_array_api_override.py:105`).
   - the same gate enforces masked/sparse/matrix rejection and dtype constraints before backend dispatch (`legacy_scipy_code/scipy/scipy/_lib/_array_api_override.py:45`, `legacy_scipy_code/scipy/scipy/_lib/_array_api_override.py:51`, `legacy_scipy_code/scipy/scipy/_lib/_array_api_override.py:118`).
7. Namespace discipline for tests/wrappers is explicit:
   - `_default_xp_ctxvar` and `default_xp` set and restore namespace context (`legacy_scipy_code/scipy/scipy/_lib/_array_api.py:191`, `legacy_scipy_code/scipy/scipy/_lib/_array_api.py:194`),
   - `_strict_check` consumes that context (fallback to `array_namespace`) to enforce matching namespace checks (`legacy_scipy_code/scipy/scipy/_lib/_array_api.py:222`, `legacy_scipy_code/scipy/scipy/_lib/_array_api.py:229`, `legacy_scipy_code/scipy/scipy/_lib/_array_api.py:233`).
8. Special-function backend support is a per-function wrapper path, not only a namespace-level overlay:
   - `_FuncInfo.wrapper` branches on `SCIPY_ARRAY_API`, applies `xp_capabilities` in place, and dispatches through `_wrapper_for` (`legacy_scipy_code/scipy/scipy/special/_support_alternative_backends.py:104`, `legacy_scipy_code/scipy/scipy/special/_support_alternative_backends.py:118`, `legacy_scipy_code/scipy/scipy/special/_support_alternative_backends.py:125`).
9. FFT default backend precedence is set explicitly at module load:
   - `set_global_backend('scipy', try_last=True)` makes SciPy backend a trailing fallback after registered/scoped backends (`legacy_scipy_code/scipy/scipy/fft/_backend.py:211`).

### 14.3 Phase-2C ownership and artifact alignment

- Workspace crate boundaries and dependency declarations are explicit in `Cargo.toml` (`Cargo.toml:3`, `Cargo.toml:24`).
- Project-level crate responsibilities are codified in AGENTS key dependency tables (`AGENTS.md:76`, `AGENTS.md:205`).
- Conformance evidence ownership is centralized in `fsci-conformance` (`Cargo.toml:11`) while runtime selection/certification belongs to `fsci-runtime` (`Cargo.toml:12`), matching the architectural flow:
  `high-level API -> domain module -> algorithm selector -> numeric kernel -> diagnostics`.

### 14.4 DOC-PASS-10 closure evidence

1. structure draft materially expanded with explicit topology/ownership/dependency mapping;
2. module relationship claims are source-anchored to concrete legacy files;
3. Rust ownership mapping is traceable to workspace configuration and project doctrine;
4. cross-references to prior doc passes remain consistent (passes 06-09 + optimization/perf governance).

## 15. DOC-PASS-14 Structure Specialist Review Artifacts and Confidence Annotations

### 15.1 Specialist findings and corrections (applied)

| Finding | Correction applied | Anchors |
|---|---|---|
| Array API behavior described too narrowly as `_lib` negotiation | Added explicit `SCIPY_ARRAY_API` global gate semantics and pre-dispatch validation boundary | `legacy_scipy_code/scipy/scipy/_lib/_array_api_override.py:27`, `legacy_scipy_code/scipy/scipy/_lib/_array_api_override.py:86`, `legacy_scipy_code/scipy/scipy/_lib/_array_api_override.py:105` |
| Namespace/context control path was underspecified | Added `default_xp`/`_default_xp_ctxvar` + `_strict_check` control flow to dependency boundaries | `legacy_scipy_code/scipy/scipy/_lib/_array_api.py:191`, `legacy_scipy_code/scipy/scipy/_lib/_array_api.py:194`, `legacy_scipy_code/scipy/scipy/_lib/_array_api.py:222` |
| Special alt-backend layer was framed as lightweight overlay | Clarified per-function wrapper lifecycle (`SCIPY_ARRAY_API` branch, `xp_capabilities`, `_wrapper_for`) | `legacy_scipy_code/scipy/scipy/special/_support_alternative_backends.py:104`, `legacy_scipy_code/scipy/scipy/special/_support_alternative_backends.py:118`, `legacy_scipy_code/scipy/scipy/special/_support_alternative_backends.py:125` |
| FFT default backend order was implicit | Added explicit default call ordering via `set_global_backend('scipy', try_last=True)` | `legacy_scipy_code/scipy/scipy/fft/_backend.py:211` |

### 15.2 Omission register status

Closed in this pass:
1. global array-api gate and validation preconditions;
2. namespace context management internals for backend-aware assertions;
3. per-function special backend wrapper mechanics;
4. FFT default backend precedence callsite.

Residual structure uncertainty (explicitly bounded after DOC-PASS-12):
1. full coverage of array-api experimental namespace notes across every scoped public function (owner: `bd-3jh.11`);
2. any additional lazy-import edges outside `scipy.sparse` that should be treated as compatibility-significant (owner: `bd-3jh.11`).

### 15.3 Section-level confidence annotations

Confidence rubric used:
- `High`: at least two direct source anchors confirm the claim.
- `Medium`: one strong source anchor confirms claim, but cross-module coverage is partial.
- `Low`: inference-heavy or pending explicit source confirmation.

| Section range | Confidence | Basis |
|---|---|---|
| Sections 1-8 (baseline structure summary) | Medium | early summary sections are accurate but intentionally compact; not every claim has dual anchors |
| Sections 9-13 (concurrency/error/edge/test/perf passes) | High | each section is anchored to concrete source files and explicit policy artifacts |
| Section 14 (expansion draft A topology map) | High | topology, ownership, and dependency assertions have direct legacy/workspace anchors |
| Section 15 (this specialist review) | High | findings tie directly to audited source paths and corrected statements |

### 15.4 DOC-PASS-14 closure evidence

1. structure specialist findings are captured with correction mapping;
2. identified structural inaccuracies/omissions are corrected in the document;
3. section-level confidence annotations are explicit and criteria-based;
4. residual uncertainty is isolated for readiness/sign-off follow-up (`bd-3jh.11`) and summarized in section 17.

## 16. DOC-PASS-12 Red-Team Resolution Snapshot

### 16.1 Contradictions resolved in this pass

1. Unified canonical legacy root path to `/data/projects/frankenscipy/legacy_scipy_code/scipy` in this document.
2. Clarified that pass/bead references use explicit pass labels where ambiguity existed (`bd-3jh.23.13` is DOC-PASS-12).
3. Removed unsupported/ambiguous structural claims by requiring direct anchor-based wording in sections 14-15.

### 16.2 Remaining bounded uncertainty

1. Array API experimental coverage breadth across every scoped public function is still pending exhaustive verification.
2. Potential additional lazy-import edges outside `scipy.sparse` remain an explicit review target.

Both uncertainty items are bounded to readiness/sign-off scope (`bd-3jh.11`).

## 17. DOC-PASS-13 Final Integrated Rewrite Sign-Off

### 17.1 Integration coverage status

Integrated pass content in this document:
1. DOC-PASS-05 complexity/performance/memory map (section 13);
2. DOC-PASS-06 concurrency/lifecycle ordering map (section 9);
3. DOC-PASS-07 error/recovery map (section 10);
4. DOC-PASS-08 security/compatibility edge zones (section 11);
5. DOC-PASS-09 test/e2e/logging crosswalk (section 12);
6. DOC-PASS-10 structure expansion draft A (section 14);
7. DOC-PASS-12 red-team resolution snapshot (section 16);
8. DOC-PASS-14 structure specialist deep-pass findings + confidence annotations (section 15).

### 17.2 Consistency sweep evidence

1. canonical legacy root path now matches `EXHAUSTIVE_LEGACY_ANALYSIS.md`;
2. pass/bead references use explicit pass labels where ambiguity was previously observed;
3. unresolved uncertainty is explicitly bounded and assigned to final integration/implementation owners;
4. no unresolved drafting markers detected.

### 17.3 Sign-off

- Date: `2026-02-14`
- Reviewer: `PlumOwl`
- Verdict: `PASS` for DOC-PASS-13 integration gate on structure document.

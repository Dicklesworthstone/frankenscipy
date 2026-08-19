# COMPREHENSIVE_SPEC_FOR_FRANKENSCIPY_V1

## 0. Prime Directive

Build a system that is simultaneously:

1. Behaviorally trustworthy for scoped compatibility.
2. Mathematically explicit in decision and risk handling.
3. Operationally resilient via RaptorQ-backed durability.
4. Performance-competitive via profile-and-proof discipline.

Crown-jewel innovation:

Condition-Aware Solver Portfolio (CASP): runtime algorithm selection driven by conditioning diagnostics and stability certificates.

Legacy oracle:

- /dp/frankenscipy/legacy_scipy_code/scipy
- upstream: https://github.com/scipy/scipy

## 1. Product Thesis

Most reimplementations fail by being partially compatible and operationally brittle. FrankenSciPy will instead combine compatibility realism with first-principles architecture and strict quality gates.

## 2. V1 Scope Contract

Included in V1:

- scoped linalg/sparse/opt/stats/signal families; - explicit tolerance policies; - core scientific benchmark corpus.

Deferred from V1:

- long-tail API surface outside highest-value use cases
- broad ecosystem parity not required for core migration value
- distributed/platform expansion not needed for V1 acceptance

## 3. Architecture Blueprint

high-level API -> domain module -> algorithm selector -> numeric kernel -> diagnostics

Planned crate families:
- fsci-types
- fsci-linalg
- fsci-sparse
- fsci-opt
- fsci-stats
- fsci-signal
- fsci-integrate
- fsci-spatial
- fsci-conformance
- frankenscipy

## 4. Compatibility Model (frankenlibc/frankenfs-inspired)

Two explicit operating modes:

1. strict mode:
   - maximize observable compatibility for scoped APIs
   - no behavior-altering repair heuristics
2. hardened mode:
   - maintain outward contract while enabling defensive runtime checks and bounded repairs

Compatibility focus for this project:

Preserve SciPy-observable behavior for scoped routines with explicit tolerance/equality policies.

Fail-closed policy:

- unknown incompatible features or protocol fields must fail closed by default
- compatibility exceptions require explicit allowlist entries and audit traces

## 5. Security Model

Security focus for this project:

Defend against numerical instability abuse, malformed array metadata, and unsafe fallback paths under ill-conditioned inputs.

Threat model baseline:

1. malformed input and parser abuse
2. state-machine desynchronization
3. downgrade and compatibility confusion paths
4. persistence corruption and replay tampering

Mandatory controls:

- adversarial fixtures and fuzz/property suites for high-risk entry points
- deterministic audit trail for recoveries and mode/policy overrides
- explicit subsystem ownership and trust-boundary notes

## 6. Alien-Artifact Decision Layer

Runtime controllers (scheduling, adaptation, fallback, admission) must document:

1. state space
2. evidence signals
3. loss matrix with asymmetric costs
4. posterior or confidence update model
5. action rule minimizing expected loss
6. calibration fallback trigger

Output requirements:

- evidence ledger entries for consequential decisions
- calibrated confidence metrics and drift alarms

## 7. Extreme Optimization Contract

Track solver runtime tails, convergence costs, and memory budgets; gate regressions for core routine families.

Optimization loop is mandatory:

1. baseline metrics
2. hotspot profile
3. single-lever optimization
4. behavior-isomorphism proof
5. re-profile and compare

No optimization is accepted without associated correctness evidence.

## 8. Correctness and Conformance Contract

Maintain conditioning-aware fallback, convergence, and tolerance invariants for scoped algorithms.

Conformance process:

1. generate canonical fixture corpus
2. run legacy oracle and capture normalized outputs
3. run FrankenSciPy and compare under explicit equality/tolerance policy
4. produce machine-readable parity report artifact

Assurance ladder:

- Tier A: unit/integration/golden fixtures
- Tier B: differential conformance
- Tier C: property/fuzz/adversarial tests
- Tier D: regression corpus for historical failures

## 9. RaptorQ-Everywhere Durability Contract

RaptorQ repair-symbol sidecars are required for long-lived project evidence:

1. conformance snapshots
2. benchmark baselines
3. migration manifests
4. reproducibility ledgers
5. release-grade state artifacts

Required artifacts:

- symbol generation manifest
- scrub verification report
- decode proof for each recovery event

## 10. Milestones and Exit Criteria

### M0 — Bootstrap

- workspace skeleton
- CI and quality gate wiring

Exit:
- fmt/check/clippy/test baseline green

### M1 — Core Model

- core data/runtime structures
- first invariant suite

Exit:
- invariant suite green
- first conformance fixtures passing

### M2 — First Vertical Slice

- end-to-end scoped workflow implemented

Exit:
- differential parity for first major API family
- baseline benchmark report published

### M3 — Scope Expansion

- additional V1 API families

Exit:
- expanded parity reports green
- no unresolved critical compatibility defects

### M4 — Hardening

- adversarial coverage and perf hardening

Exit:
- regression gates stable
- conformance drift zero for V1 scope

## 11. Acceptance Gates

Gate A: compatibility parity report passes for V1 scope.

Gate B: security/fuzz/adversarial suite passes for high-risk paths.

Gate C: performance budgets pass with no semantic regressions.

Gate D: RaptorQ durability artifacts validated and scrub-clean.

All four gates must pass for V1 release readiness.

## 12. Risk Register

Primary risk focus:

Instability and false confidence in edge conditioning regimes.

Mitigations:

1. compatibility-first development for risky API families
2. explicit invariants and adversarial tests
3. profile-driven optimization with proof artifacts
4. strict mode/hardened mode separation with audited policy transitions
5. RaptorQ-backed resilience for critical persistent artifacts

## 13. Immediate Execution Checklist

1. Create workspace and crate skeleton.
2. Implement smallest high-value end-to-end path in V1 scope.
3. Stand up differential conformance harness against legacy oracle.
4. Add benchmark baseline generation and regression gating.
5. Add RaptorQ sidecar pipeline for conformance and benchmark artifacts.

## 14. Detailed Crate Contracts (V1)

| Crate | Primary Responsibility | Explicit Non-Goal | Invariants | Mandatory Tests |
|---|---|---|---|---|
| fsci-types | numeric type/shape/tolerance metadata | solver execution | stable tolerance policy encoding, deterministic metadata round-trip | type+tolerance matrix tests |
| fsci-linalg | dense and structured linear algebra first-wave routines | broad LAPACK parity | solver contract fidelity for scoped routines | solve/decompose parity fixtures |
| fsci-sparse | sparse matrix formats and core operations | dense fallback orchestration | index monotonicity and shape invariants preserved | sparse invariant/property tests |
| fsci-opt | optimization primitives and convergence bookkeeping | custom user optimizer plugins | status code semantics and convergence reporting stable | optimizer parity suites |
| fsci-stats | scoped statistical routines and support contracts | full distribution catalog parity | numeric stability and warning contracts preserved | stats parity fixtures |
| fsci-signal | scoped signal-processing routines | full DSP ecosystem parity | deterministic filter/output behavior for scoped families | signal parity fixtures |
| fsci-integrate | ODE/IVP first-wave integrators | full integrator zoo parity | step/tolerance/event contracts preserved | IVP conformance fixtures |
| fsci-spatial | scoped spatial primitives | full geometry ecosystem parity | distance/index semantics preserved | spatial parity fixtures |
| fsci-conformance | differential harness vs SciPy oracle | production serving | comparison policy explicit per routine family | report schema + runner tests |
| frankenscipy | integration binary/library and policy loading | algorithm research | strict/hardened mode wiring + evidence logging | mode gate/startup tests |

## 15. Conformance Matrix (V1)

| Family | Oracle Workload | Pass Criterion | Drift Severity |
|---|---|---|---|
| linalg solve/decompose | solve/inv/lstsq/pinv fixture corpus | value + error-contract parity under tolerance policy | critical |
| sparse base operations | sparse shape/index and algebra fixtures | index/shape/value parity | high |
| optimization core | BFGS/CG/Powell/scalar minimization fixtures | status + trajectory parity under policy | high |
| IVP/integrate | stiff/non-stiff ODE fixture corpus | solution + event parity under policy | critical |
| FFT/backend routing | backend selection + transform fixtures | deterministic backend route + output parity | high |
| special-function warning/error behavior | warning/error fixture corpus | warning class and result parity | high |
| array API compatibility glue | backend negotiation fixtures | route and contract parity | medium |
| mixed scientific E2E | solve -> optimize -> integrate pipeline | reproducible parity report with no critical drift | critical |

## 16. Security and Compatibility Threat Matrix

| Threat | Strict Mode Response | Hardened Mode Response | Required Artifact |
|---|---|---|---|
| malformed array metadata | fail-closed | fail-closed with bounded diagnostics | metadata incident ledger |
| numerical instability abuse | follow scoped semantics; explicit failure on undefined zones | bounded guards with explicit repair/reject trace | stability decision ledger |
| unsafe backend fallback | fail-closed | fail-closed unless policy allowlisted | backend policy ledger |
| tolerance confusion | reject incompatible tolerance spec | reject + normalized explanation | tolerance audit report |
| unknown incompatible metadata | fail-closed | fail-closed | compatibility drift report |
| oracle mismatch in conformance | hard fail | hard fail | conformance failure bundle |
| artifact corruption | reject load | recover via RaptorQ when provable | decode proof + scrub report |
| override misuse | explicit override + audit trail | explicit override + audit trail | override audit record |

## 17. Performance Budgets and SLO Targets

| Path | Workload Class | Budget |
|---|---|---|
| dense solve hot path | 4k-8k matrix class | p95 <= 650 ms |
| sparse matvec + reductions | million-edge sparse workloads | p95 <= 220 ms |
| optimizer iteration overhead | medium-dimensional objective workloads | p95 <= 180 ms per iteration group |
| IVP solve step loop | representative stiff/non-stiff ODEs | p95 <= 320 ms |
| FFT transform first wave | medium-to-large transform fixtures | p95 <= 210 ms |
| warning/error path overhead | adversarial numeric inputs | p95 regression <= +8% |
| memory footprint | mixed scientific E2E | peak RSS regression <= +10% |
| tail stability | all benchmark families | p99 regression <= +8% |

Optimization acceptance rule:
1. primary metric improves or stays in budget,
2. no critical conformance drift,
3. p99 and memory budgets remain within limits.

## 18. CI Gate Topology (Release-Critical)

| Gate | Name | Blocking | Output Artifact |
|---|---|---|---|
| G1 | format + lint | yes | lint report |
| G2 | unit + integration | yes | junit report |
| G3 | differential conformance | yes | parity report JSON + markdown summary |
| G4 | adversarial + property tests | yes | minimized counterexample corpus |
| G5 | benchmark regression | yes | baseline delta report |
| G6 | RaptorQ scrub + recovery drill | yes | scrub report + decode proof sample |

Release cannot proceed unless all gates pass on the same commit.

## 19. RaptorQ Artifact Envelope (Project-Wide)

Persistent evidence artifacts must be emitted with sidecars:
1. source artifact hash manifest,
2. RaptorQ symbol manifest,
3. scrub status,
4. decode proof log when recovery occurs.

Canonical envelope schema:

~~~json
{
  "artifact_id": "string",
  "artifact_type": "conformance|benchmark|ledger|manifest",
  "source_hash": "blake3:...",
  "raptorq": {
    "k": 0,
    "repair_symbols": 0,
    "overhead_ratio": 0.0,
    "symbol_hashes": ["..."]
  },
  "scrub": {
    "last_ok_unix_ms": 0,
    "status": "ok|recovered|failed"
  },
  "decode_proofs": [
    {
      "ts_unix_ms": 0,
      "reason": "...",
      "recovered_blocks": 0,
      "proof_hash": "blake3:..."
    }
  ]
}
~~~

## 20. 90-Day Execution Plan

Weeks 1-2:
- scaffold workspace and crate boundaries
- finalize routine-family conformance schema + tolerance policy

Weeks 3-5:
- implement fsci-types/fsci-linalg/fsci-integrate minimum vertical slice
- land first strict-mode differential conformance reports

Weeks 6-8:
- add sparse/opt/stats first-wave routines
- publish baseline benchmarks aligned with section-17 budgets

Weeks 9-10:
- harden backend selection and adversarial numeric paths
- finalize strict/hardened policy logging and drift gates

Weeks 11-12:
- enforce full gate topology G1-G6 in CI
- run release-candidate drill with complete artifact bundle

## 21. Porting Artifact Index

This spec is paired with the following methodology artifacts:

1. PLAN_TO_PORT_SCIPY_TO_RUST.md
2. EXISTING_SCIPY_STRUCTURE.md
3. PROPOSED_ARCHITECTURE.md
4. FEATURE_PARITY.md

Rule of use:

- Extraction and behavior understanding happens in EXISTING_SCIPY_STRUCTURE.md.
- Scope, exclusions, and phase sequencing live in PLAN_TO_PORT_SCIPY_TO_RUST.md.
- Rust crate boundaries live in PROPOSED_ARCHITECTURE.md.
- Delivery readiness is tracked in FEATURE_PARITY.md.

## 22. FrankenSQLite Exemplar Inheritance Contract

Canonical exemplar imported to this repository:
- `reference/frankensqlite/COMPREHENSIVE_SPEC_FOR_FRANKENSQLITE_V1.md`

Normative rule:
- all future FrankenSciPy spec revisions must include a section-level crosswalk to the exemplar for:
  1. gate topology,
  2. evidence ledger schema,
  3. recovery-proof lifecycle,
  4. performance-delta governance,
  5. strict/hardened compatibility drift policy.

Adoption threshold:
- no major architectural spec change is accepted without an explicit "FrankenSQLite-delta" note documenting:
  - inherited pattern,
  - domain-specific adaptation,
  - rationale for any divergence.

## 23. Asupersync Integration Contract

FrankenSciPy must consume `/dp/asupersync` for:
1. RaptorQ symbol generation and repair/decode proof plumbing for durable artifacts.
2. E-process and lab-runtime style statistical monitoring where runtime invariants require anytime-valid alarms.
3. Structured runtime primitives for bounded supervision and fail-closed recovery orchestration.

Initial implementation status (V1 bootstrap):
- `fsci-conformance` emits parity reports with RaptorQ sidecars using `asupersync::raptorq::systematic::SystematicEncoder`.

Mandatory future expansions:
- packet-level decode replay proofs for actual recovery events,
- scheduler and supervision integration in runtime policy controllers,
- calibrated invariant monitors for solver correctness sentinels.

## 24. FrankenTUI Integration Contract

FrankenSciPy must consume `/dp/frankentui` (`ftui`) for operator-facing quality dashboards:
1. conformance drift triage,
2. benchmark delta visualization,
3. strict/hardened policy decision inspection.

Initial implementation status (V1 bootstrap):
- `fsci-conformance` exports `style_for_case_result` with `ftui::Style` contracts for pass/fail rendering.

Mandatory future expansions:
- interactive packet browser over parity artifacts,
- trend charts for p50/p95/p99 and memory deltas,
- incident timeline view for compatibility and recovery ledgers.

## 25. Packetized Porting Execution Law

Porting progression is locked to `FSCI-P2C-001..008` packets and each packet must ship:
1. clean-room implementation artifact in target crate,
2. conformance fixture family,
3. strict/hardened gate outcomes,
4. benchmark delta snapshot (when runtime-significant),
5. RaptorQ sidecar and decode-proof record.

Acceptance invariant:
- packet status cannot advance to `parity_green` unless all five artifacts exist and are reproducible from CI on the same commit.

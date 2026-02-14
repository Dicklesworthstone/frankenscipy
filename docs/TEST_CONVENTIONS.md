# FrankenSciPy Test Conventions

This document defines the testing standards for FrankenSciPy, a clean-room Rust
reimplementation of SciPy. All contributors must follow these conventions to
maintain consistency, reproducibility, and numerical rigor across the workspace.

---

## Table of Contents

1. [Guiding Principles](#guiding-principles)
2. [Test Naming Convention](#test-naming-convention)
3. [Test Organization](#test-organization)
4. [Property Testing with proptest](#property-testing-with-proptest)
5. [Seed Management and Reproducibility](#seed-management-and-reproducibility)
6. [Structured Log Format](#structured-log-format)
7. [Tolerance Conventions](#tolerance-conventions)
8. [Fixture Management](#fixture-management)
9. [Mode Testing: Strict and Hardened](#mode-testing-strict-and-hardened)
10. [CASP Integration Testing](#casp-integration-testing)
11. [Required Test Evidence Per Change](#required-test-evidence-per-change)
12. [RaptorQ Durability for Test Artifacts](#raptorq-durability-for-test-artifacts)

---

## Guiding Principles

The following project-wide principles govern all testing decisions:

- **Numerical stability outranks speed.** A numerically stable implementation
  that is slower always wins over a fast but unstable one. Tests must verify
  stability properties, not just correctness on happy paths.
- **Reproducibility is non-negotiable.** Every test run must be fully
  reproducible given the same seed and inputs.
- **Differential conformance against SciPy.** The Python oracle
  (`.venv-py314`, SciPy 1.17.0, Python 3.14.2) is the reference. Tests must
  demonstrate parity or document deliberate divergence.

---

## Test Naming Convention

All test functions follow the pattern:

```
test_{module}_{function}_{scenario}
```

**Components:**

| Segment      | Description                                         | Example              |
|--------------|-----------------------------------------------------|----------------------|
| `module`     | The crate or submodule under test                   | `linalg`, `sparse`   |
| `function`   | The specific function or capability being tested    | `solve`, `eigenvals` |
| `scenario`   | The condition, edge case, or property being verified | `singular_matrix`, `large_condition_number` |

**Examples:**

```rust
#[test]
fn test_linalg_solve_singular_matrix() { /* ... */ }

#[test]
fn test_sparse_csr_multiply_empty_row() { /* ... */ }

#[test]
fn test_integrate_quad_oscillatory_convergence() { /* ... */ }
```

For property tests, prefix the scenario with `prop_`:

```rust
proptest! {
    #[test]
    fn test_linalg_solve_prop_roundtrip(matrix in arb_nonsingular(4)) {
        // ...
    }
}
```

---

## Test Organization

### Unit Tests

Unit tests live in `#[cfg(test)] mod tests` blocks inside the source files they
exercise. These tests cover internal logic, private helpers, and isolated
computations.

```
crates/fsci-linalg/src/decompose.rs
    -> mod tests { ... }
```

### Integration and Property Tests

Integration tests and property-based tests live in the `tests/` directory of
each crate. These tests exercise public API surfaces, cross-module interactions,
and statistical properties.

```
crates/fsci-linalg/tests/
    property_solve.rs
    integration_decompose.rs
    differential_eigenvals.rs
```

### Conformance Tests

Conformance tests that verify parity with the SciPy oracle live in the
`fsci-conformance` crate:

```
crates/fsci-conformance/tests/
    p2c_001_linalg.rs
    p2c_002_sparse.rs
    ...
```

---

## Property Testing with proptest

FrankenSciPy uses [proptest](https://docs.rs/proptest) as its property testing
framework. The dependency is declared at the workspace level in the root
`Cargo.toml`:

```toml
[workspace.dependencies]
proptest = { version = "1", default-features = false, features = ["std"] }
```

Each crate that needs property tests adds it as a dev dependency:

```toml
[dev-dependencies]
proptest.workspace = true
```

### Minimum Case Counts

Property tests must generate a sufficient number of cases to provide meaningful
coverage. The following minimums apply:

| Test Category                  | Minimum Cases |
|--------------------------------|---------------|
| Pure arithmetic / simple logic | 256           |
| Matrix operations (small)      | 128           |
| Matrix operations (large)      | 64            |
| Solver selection (CASP paths)  | 128           |
| Cross-module integration       | 64            |

Configure case counts via the `proptest::test_runner::Config`:

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn test_special_gamma_prop_reflection(x in 0.01f64..100.0) {
        // ...
    }
}
```

Never set case counts below the minimums listed above without explicit
justification in a code comment.

---

## Seed Management and Reproducibility

Every randomized test must record its seed so that failures can be replayed
deterministically.

### How proptest Seeds Work

proptest automatically persists regression files in a `proptest-regressions/`
directory adjacent to the test file. These files must be committed to version
control.

### Replaying a Specific Seed

To replay a failing property test with a specific seed:

```bash
PROPTEST_SEED=<seed> cargo test <test_name>
```

For example:

```bash
PROPTEST_SEED=17492835610384 cargo test test_linalg_solve_prop_roundtrip
```

### Seed Logging

All property tests must log their seed in the structured log format (see below)
at the start of execution. If a test uses its own RNG (outside proptest), it
must explicitly record the seed:

```rust
let seed: u64 = /* obtain or generate */;
log_test_event(TestEvent {
    test_id: "test_linalg_solve_prop_roundtrip",
    level: Level::Info,
    message: "property test starting",
    seed: Some(seed),
    ..Default::default()
});
```

---

## Structured Log Format

All test output must use a structured JSON log format. This enables automated
parsing, aggregation, and regression detection in CI.

### Schema

Every log line emitted during a test must conform to the following structure:

```json
{
  "test_id": "test_linalg_solve_singular_matrix",
  "timestamp_ms": 1707900000000,
  "level": "INFO",
  "module": "fsci_linalg::solve",
  "message": "solver returned expected error for singular input",
  "seed": null,
  "fixture_id": null
}
```

### Field Definitions

| Field          | Type             | Required | Description                                              |
|----------------|------------------|----------|----------------------------------------------------------|
| `test_id`      | string           | yes      | Full test function name                                  |
| `timestamp_ms` | u64              | yes      | Milliseconds since Unix epoch                            |
| `level`        | string           | yes      | One of: `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR`        |
| `module`       | string           | yes      | Rust module path of the code under test                  |
| `message`      | string           | yes      | Human-readable description of the event                  |
| `seed`         | u64 or null      | no       | RNG seed, present for randomized tests                   |
| `fixture_id`   | string or null   | no       | Identifier of the fixture used, if any                   |

### Usage

Use the project's test logging utilities (provided in `fsci-runtime`) to emit
structured logs rather than raw `println!` or `eprintln!` calls:

```rust
use fsci_runtime::test_log::{log_test_event, TestEvent, Level};

log_test_event(TestEvent {
    test_id: "test_linalg_solve_singular_matrix",
    level: Level::Info,
    module: module_path!(),
    message: "verified singular matrix returns ConditionError",
    seed: None,
    fixture_id: None,
});
```

---

## Tolerance Conventions

Numerical comparisons must never use exact floating-point equality. Instead, use
the project's tolerance assertion utilities.

### Formula

The tolerance check uses the combined absolute/relative formula:

```
|actual - expected| <= atol + rtol * |expected|
```

Where:
- `atol` is the absolute tolerance (dominates when `expected` is near zero)
- `rtol` is the relative tolerance (dominates for large values)

### Assertion Macros

Use the project-provided assertion macros:

```rust
// Single value comparison
assert_close!(actual, expected, atol = 1e-12, rtol = 1e-10);

// Element-wise array/matrix comparison
assert_within_tolerance!(actual_matrix, expected_matrix, atol = 1e-10, rtol = 1e-8);
```

### Default Tolerances

Unless the specific algorithm requires tighter or looser bounds, use:

| Precision | `atol`  | `rtol`  |
|-----------|---------|---------|
| f64       | 1e-12   | 1e-10   |
| f32       | 1e-5    | 1e-4    |

### Documenting Tolerance Choices

When a test uses non-default tolerances, include a comment explaining why:

```rust
// Relaxed tolerance: iterative solver converges to ~1e-6 for
// ill-conditioned systems (condition number > 1e10).
assert_close!(result, oracle_value, atol = 1e-6, rtol = 1e-4);
```

---

## Fixture Management

Test fixtures are stored as JSON files in the conformance crate:

```
crates/fsci-conformance/fixtures/
    linalg/
        solve_small_dense.json
        eigenvals_symmetric.json
    sparse/
        csr_multiply.json
    integrate/
        quad_basic.json
```

### Fixture Format

Each fixture file contains an array of test cases with inputs and expected
outputs, keyed by a unique `fixture_id`:

```json
[
  {
    "fixture_id": "solve_small_dense_001",
    "input": {
      "matrix": [[1.0, 2.0], [3.0, 4.0]],
      "rhs": [5.0, 6.0]
    },
    "expected": {
      "solution": [-4.0, 4.5]
    },
    "metadata": {
      "source": "scipy_1.17.0",
      "condition_number": 14.933
    }
  }
]
```

### Fixture Generation

Fixtures are generated by running the Python oracle and capturing outputs.
Generated fixtures must include the `metadata.source` field indicating the SciPy
version and Python version used.

### Fixture Referencing in Tests

Tests must log the `fixture_id` in their structured output so that failures can
be traced back to specific fixture entries:

```rust
for case in load_fixtures("linalg/solve_small_dense.json") {
    log_test_event(TestEvent {
        test_id: "test_linalg_solve_fixture_conformance",
        fixture_id: Some(&case.fixture_id),
        ..
    });
    let result = solve(&case.input.matrix, &case.input.rhs);
    assert_close!(result, case.expected.solution, atol = 1e-12, rtol = 1e-10);
}
```

---

## Mode Testing: Strict and Hardened

FrankenSciPy operates in two modes, and both must have dedicated test coverage:

### Strict Mode

Strict mode prioritizes numerical accuracy and correctness. It uses
conservative algorithm choices, tighter convergence criteria, and additional
validation checks. Tests for strict mode verify:

- Tighter tolerance bounds are enforced
- Additional input validation is performed (e.g., symmetry checks)
- Conservative solver selection in CASP
- Graceful error reporting for edge cases

```rust
#[test]
fn test_linalg_solve_strict_rejects_near_singular() {
    let config = SolverConfig::strict();
    let result = solve_with_config(&near_singular_matrix, &rhs, &config);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), SolverError::ConditionTooHigh { .. }));
}
```

### Hardened Mode

Hardened mode adds runtime checks, sanitization, and defensive measures
suitable for adversarial or untrusted inputs. Tests for hardened mode verify:

- Input size limits are enforced
- NaN/Inf propagation is caught and reported
- Memory allocation bounds are respected
- Timeout mechanisms activate for runaway computations

```rust
#[test]
fn test_linalg_solve_hardened_rejects_oversized_input() {
    let config = SolverConfig::hardened();
    let huge = DMatrix::zeros(100_001, 100_001);
    let result = solve_with_config(&huge, &rhs, &config);
    assert!(matches!(result.unwrap_err(), SolverError::InputTooLarge { .. }));
}
```

### Coverage Requirement

Every public function that behaves differently in strict vs. hardened mode must
have at least one test for each mode. Use the naming convention:

```
test_{module}_{function}_strict_{scenario}
test_{module}_{function}_hardened_{scenario}
```

---

## CASP Integration Testing

The Condition-Aware Solver Portfolio (CASP) dynamically selects solvers based on
problem characteristics (condition number, sparsity, size, etc.). Tests must
verify:

1. **Solver selection logic:** Given specific matrix properties, CASP selects
   the expected solver.
2. **Fallback chains:** When the primary solver fails or exceeds its confidence
   threshold, CASP correctly falls back to the next candidate.
3. **Conformal calibration:** The calibration layer produces valid prediction
   sets with the declared coverage guarantee.
4. **Mode interaction:** CASP respects strict/hardened mode constraints when
   selecting solvers.

```rust
#[test]
fn test_linalg_casp_selects_cholesky_for_spd() {
    let features = ProblemFeatures::from_matrix(&spd_matrix);
    let selection = CaspPortfolio::select(&features, &SolverConfig::strict());
    assert_eq!(selection.primary_solver(), Solver::Cholesky);
}
```

---

## Required Test Evidence Per Change

Every code change that modifies numerical behavior must include the following
evidence before it can be merged:

### 1. Differential Conformance Report

A report comparing FrankenSciPy output against the SciPy oracle for all
affected operations. The report must include:

- Function name and input dimensions
- FrankenSciPy result vs. SciPy result
- Absolute and relative error
- Pass/fail status against the declared tolerances

Generate with:

```bash
cargo test -p fsci-conformance -- --test-threads=1 2>&1 | tee conformance_report.log
```

### 2. Invariant Checklist

A checklist confirming that key numerical invariants hold:

- [ ] Symmetry preserved where expected
- [ ] Orthogonality of computed bases within tolerance
- [ ] Determinant sign correctness
- [ ] Condition number estimation within one order of magnitude
- [ ] No silent NaN/Inf introduction
- [ ] Backward error within declared tolerance

### 3. Benchmark Delta

A before/after comparison of benchmark results for affected operations,
generated using the criterion benchmark harness:

```bash
cargo bench -p fsci-linalg -- --save-baseline before
# apply changes
cargo bench -p fsci-linalg -- --save-baseline after --baseline before
```

Performance regressions greater than 10% require explicit justification.
Recall: **numerical stability outranks speed** -- a regression caused by
improved numerical properties is acceptable when documented.

---

## RaptorQ Durability for Test Artifacts

Test artifacts (conformance reports, benchmark baselines, regression files) are
protected using RaptorQ erasure coding via the `asupersync` dependency. This
ensures that partial data loss (e.g., corrupted CI cache) does not prevent
failure reproduction.

### What Gets Encoded

- `proptest-regressions/` directories
- Fixture JSON files in `crates/fsci-conformance/fixtures/`
- Benchmark baseline data

### Recovery

If a regression file or fixture is corrupted or missing, it can be
reconstructed from the RaptorQ-encoded shards stored alongside the originals.
Refer to the `asupersync` crate documentation for recovery commands.

### Implications for Test Authors

- Do not `.gitignore` proptest regression files. They must be committed.
- Do not manually edit fixture JSON files. Regenerate them from the Python
  oracle to maintain the integrity chain.
- Benchmark baselines should be regenerated rather than patched when the
  underlying algorithm changes.

---

## Summary

| Convention                | Key Rule                                                        |
|---------------------------|-----------------------------------------------------------------|
| Naming                    | `test_{module}_{function}_{scenario}`                           |
| Property tests            | proptest, workspace dependency, minimum case counts enforced    |
| Logging                   | Structured JSON with `test_id`, `timestamp_ms`, `level`, etc.   |
| Seeds                     | Recorded, committed, replayable via `PROPTEST_SEED`             |
| Organization              | Unit in `src/`, integration and property in `tests/`            |
| Tolerances                | `atol + rtol * |expected|`, project defaults per precision      |
| Fixtures                  | JSON in `fsci-conformance/fixtures/`, oracle-generated          |
| Mode coverage             | Both strict and hardened paths tested for every divergent API   |
| Evidence                  | Conformance report, invariant checklist, benchmark delta        |
| Artifact durability       | RaptorQ-encoded via asupersync                                  |

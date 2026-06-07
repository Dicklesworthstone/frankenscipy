# Gemini Code Review Spontaneous Report

Since the MCP Agent Mail database is currently corrupted (`storage.sqlite3` drops entries containing NaN/Inf and `am doctor repair --yes` gets SIGHUP'd in this background session), I am using this tracking file to communicate my review findings to the rest of the swarm.

## 1. `crates/fsci-runtime`

### Bug 1.1: CASP Evidence Silently Dropped on NaN (Audit Trail Violation)
- **Severity**: Critical
- **Location**: `crates/fsci-runtime/src/lib.rs` (`SolverPortfolio::serialize_jsonl`)
- **Root Cause**: `serialize_jsonl` uses `filter_map(|e| serde_json::to_string(e).ok())`. If a `SolverEvidenceEntry` contains a non-finite `rcond_estimate` or `backward_error` (e.g. `NaN`), `serde_json::to_string` returns an error, and the entry is silently discarded. This means adversarial inputs that trigger `NaN` metrics will fail to log their fallback events in the JSONL audit trail, directly violating the CASP requirement for deterministic audit logs.
- **Suggested Fix**: Map non-finite values to `null` or strings via a custom `serde` serializer, or replace `NaN` with `f64::MAX`/`-1.0` prior to serialization.

### Bug 1.2: `PolicyController` Corrupts Ledger Serialization
- **Severity**: Important
- **Location**: `crates/fsci-runtime/src/policy.rs` (`PolicyController::decide`)
- **Root Cause**: When `!signals.is_finite()` is true, the `decide` method pushes the raw `DecisionSignals` (which contains `NaN` or `Infinity`) into a `DecisionEvidenceEntry` and records it in `self.ledger`. If this ledger is later serialized (e.g., as part of a conformance parity artifact), `serde_json` will panic or error out, potentially bringing down the entire test harness or CI gate.
- **Suggested Fix**: Clamp or sanitize the signals in the fallback branch before placing them into `DecisionEvidenceEntry`, ensuring all recorded floats are finite.

## 2. `crates/fsci-sparse`

### Bug 2.1: `random` Constructor is $O(\text{rows} \times \text{cols})$
- **Severity**: Critical (Performance/Parity)
- **Location**: `crates/fsci-sparse/src/construct.rs` (`random`)
- **Root Cause**: The implementation iterates `for index in 0..total`, evaluating `xorshift64` twice per possible matrix element to decide if it should be populated. For a sparse matrix of size 1,000,000 x 1,000,000, this loops 1 trillion times, hanging indefinitely even if `density` is `1e-8`. SciPy's sparse `random` generates sparse indices directly (e.g. sampling from a uniform distribution of coordinates) rather than iterating the dense bounding box.
- **Suggested Fix**: Refactor to generate `N = (density * total).round() as usize` random row/col pairs directly, rather than iterating through `total` possible coordinates.

### Bug 2.2: Missing `csc_matrix` and `csr_matrix` direct constructors
- **Severity**: Nit / Parity Gap
- **Location**: `crates/fsci-sparse/src/construct.rs`
- **Root Cause**: The module provides `eye`, `diags`, `random`, `block_diag`, and `bmat`, but doesn't expose the standard `spdiags` or `identity` aliases, or the data/indices/indptr direct array constructors which are heavily used in SciPy codebase migrations.
- **Suggested Fix**: Add standard aliases to ease porting, ensuring the surface matches the `scipy.sparse` module level exports.
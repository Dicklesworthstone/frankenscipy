# fsci-opt Implementation Sequence (P2C-003-D)

This sequence translates the P2C-003 contract and threat artifacts into a low-rework Rust implementation plan.

## 1. Stabilize Shared Types First

1. Land `types` module with `OptimizeResult`, status enums, option structs, and error taxonomy.
2. Freeze field names for SciPy-facing compatibility (`x`, `fun`, `success`, `status`, `message`, `nfev`, `njev`, `nhev`, `nit`, `jac`, `hess_inv`, `maxcv`).
3. Add unit tests for default options and status semantics.

Rationale: This blocks churn across every downstream algorithm.

## 2. Implement Line Search Contract Surface

1. Land Wolfe parameter validation (`0 < c1 < c2 < 1`, alpha bounds, iteration bounds).
2. Add placeholder `line_search_wolfe1` and `line_search_wolfe2` signatures and result type.
3. Add conformance stubs for failure-mode mapping (`PrecisionLoss`, non-finite guards).

Rationale: BFGS/CG reuse this path; getting interfaces stable early minimizes rewrites.

## 3. Implement Minimize Dispatcher + Method Skeletons

1. Implement `minimize` dispatcher with SciPy-like default (`BFGS` when unconstrained).
2. Add BFGS/CG/Powell skeleton entrypoints and explicit `NotImplemented` status returns.
3. Validate input shape and finite checks at dispatch boundary.

Rationale: Exposes packet API while preserving fail-closed behavior.

## 4. Implement Root Dispatcher + Bracketing Kernels

1. Implement `root_scalar` dispatch and bracket validation contract.
2. Implement `brentq`, `brenth`, `bisect`, `ridder` kernels in order of risk:
   1. `bisect` (reference baseline)
   2. `brentq`
   3. `brenth`
   4. `ridder`
3. Ensure tolerance semantics match packet contract (`|x - x0| <= xtol + rtol*|x0|`).

Rationale: Bracketing root methods provide deterministic, auditable convergence paths.

## 5. Fill Algorithm Cores in Risk Order

1. BFGS core with SciPy defaults and status mapping.
2. CG (PR+) with descent guard and reset behavior.
3. Powell direction-set updates and bounded line-search integration.

Rationale: This order aligns with threat severity and shared dependency structure.

## 6. Test and Evidence Escalation

1. Unit tests for each module surface (happy path + malformed input).
2. Property tests for invariants (status monotonicity, finite guards, bracket rules).
3. Differential and adversarial fixtures per threat matrix IDs.
4. Emit packet artifacts required by P2C-003-E/F/G/H/I.

## 7. Gate Checklist per Step

For every substantial increment:

1. `rch exec -- cargo fmt --check`
2. `rch exec -- cargo check -p fsci-opt --all-targets`
3. `rch exec -- cargo clippy -p fsci-opt --all-targets -- -D warnings`
4. `rch exec -- cargo test -p fsci-opt --all-targets`
5. `rch exec -- cargo test -p fsci-conformance --test schema_validation --locked`

This sequence is intentionally conservative: it optimizes for contract parity and threat-driven correctness before numerical throughput work.

# fsci-linalg Implementation Sequence

## Module Boundary Decisions

| Module Path | Functions | Rationale |
|---|---|---|
| `fsci_linalg` (root) | solve, solve_triangular, solve_banded, inv, det, lstsq, pinv | Flat module; all functions share nalgebra types and CASP portfolio |
| Internal helpers | solve_general, solve_qr, solve_svd_fallback, solve_diagonal, solve_triangular_internal | Private dispatch targets selected by CASP or MatrixAssumption |

nalgebra integration: Direct use of `DMatrix<f64>` / `DVector<f64>` internally.
Public API uses `&[Vec<f64>]` / `&[f64]` for zero-dependency consumer interface.

## Type Skeleton (Implemented)

### Enums
- `MatrixAssumption` -- General, Diagonal, UpperTriangular, LowerTriangular, Symmetric, Hermitian, PositiveDefinite, Banded, TriDiagonal
- `LstsqDriver` -- Gelsd (default), Gelsy, Gelss
- `TriangularTranspose` -- NoTranspose, Transpose, ConjugateTranspose
- `LinalgWarning` -- IllConditioned { reciprocal_condition }
- `LinalgError` -- RaggedMatrix, ExpectedSquareMatrix, IncompatibleShapes, NonFiniteInput, SingularMatrix, UnsupportedAssumption, InvalidBandShape, InvalidPinvThreshold, NotSupported, ConvergenceFailure, ConditionTooHigh, ResourceExhausted, InvalidArgument

### Option Structs
- `SolveOptions` { mode, check_finite, assume_a, lower, transposed }
- `TriangularSolveOptions` { mode, check_finite, trans, lower, unit_diagonal }
- `InvOptions` { mode, check_finite, assume_a, lower }
- `LstsqOptions` { mode, check_finite, cond, driver }
- `PinvOptions` { mode, check_finite, atol, rtol }

### Result Structs
- `SolveResult` { x, warning, backward_error }
- `InvResult` { inverse, warning }
- `LstsqResult` { x, residuals, rank, singular_values }
- `PinvResult` { pseudo_inverse, rank }

## Implementation Phases

### Phase 1: Core Solvers (bd-3jh.13.10)
Already substantially implemented:
1. `solve()` -- LU via nalgebra, dispatches on MatrixAssumption
2. `solve_triangular()` -- Forward/back substitution
3. `solve_banded()` -- Dense expansion + LU fallback
4. `inv()` -- Single-LU path for general, column-by-column for structured
5. `det()` -- LU determinant
6. `lstsq()` -- SVD-based least squares
7. `pinv()` -- SVD pseudo-inverse with threshold

### Phase 2: CASP Integration (bd-3jh.13.10)
Already implemented:
1. `solve_with_casp()` -- Condition-aware solver portfolio
2. `fast_rcond_from_lu()` -- O(n) condition estimate
3. `classify_condition()` -- Maps rcond to MatrixConditionState
4. `randomized_rcond_estimate()` -- Power iteration spectral estimate
5. Backward error computation

### Phase 3: Driver Selection for lstsq (bd-3jh.13.10)
Remaining work:
1. Implement Gelsy driver path (QR with column pivoting)
2. Implement Gelss driver path (plain SVD)
3. Wire LstsqDriver selection into lstsq()

### Phase 4: Hardened Mode Extensions (bd-3jh.13.10)
Remaining work:
1. ConditionTooHigh rejection in hardened mode for solve/inv
2. ResourceExhausted for dimension limits in hardened mode
3. NotSupported for complex transpose in solve_triangular

### Phase 5: Eigenvalue Decomposition (future packets)
Not in P2C-002 scope. Tracked separately.

## rcond Computation Path

```
Input matrix A
    |
    v
LU factorization (nalgebra)
    |
    v
fast_rcond_from_lu: min(|U_ii|) / max(|U_ii|)  -- O(n), conservative
    |
    v
classify_condition: WellConditioned / Moderate / Ill / NearSingular
    |
    v
CASP portfolio.select_action() -- Bayesian expected-loss minimization
    |
    v
Dispatch: DirectLU / PivotedQR / SVDFallback / DiagonalFastPath / TriangularFastPath
```

Optional refinement via `randomized_rcond_estimate()` using inverse power iteration.

## Dependency Map

```
fsci-runtime (SolverPortfolio, RuntimeMode, MatrixConditionState)
    |
    v
fsci-linalg (this crate)
    |
    v
fsci-conformance (differential tests, harness integration)
```

External: nalgebra (DMatrix, DVector, LU, SVD, QR)

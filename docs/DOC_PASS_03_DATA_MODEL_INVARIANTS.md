# DOC-PASS-03: Data Model, State, and Invariant Mapping

Generated: 2026-02-14
Scope: All 20 SciPy domains - complete data structure inventory with invariants and state transitions

## 1. scipy.linalg

### 1.1 Core Data Structures

| Structure | Rust Equivalent | Description |
|-----------|----------------|-------------|
| 2D ndarray (m x n) | `Vec<Vec<f64>>` / nalgebra `DMatrix` | Dense matrix, row-major or column-major |
| 1D ndarray (n,) | `Vec<f64>` / nalgebra `DVector` | Dense vector |
| LU factorization tuple | `LuFactorization { lu: DMatrix, piv: Vec<usize> }` | Packed LU with pivot indices |
| QR factorization | `QrFactorization { q: DMatrix, r: DMatrix }` | Orthogonal Q, upper-triangular R |
| SVD result | `SvdResult { u: DMatrix, s: DVector, vt: DMatrix }` | U*diag(s)*Vt decomposition |
| Cholesky factor | `CholeskyFactor { l: DMatrix }` | Lower-triangular L where A = L*Lt |
| Schur decomposition | `SchurResult { t: DMatrix, z: DMatrix }` | T quasi-upper-triangular, Z unitary |
| Eigenvalue result | `EigResult { values: Vec<Complex64>, vectors: DMatrix<Complex64> }` | Eigenvalue/eigenvector pairs |
| Hessenberg form | `HessenbergResult { h: DMatrix, q: DMatrix }` | Upper Hessenberg H, unitary Q |
| Banded matrix (ab) | `BandedMatrix { ab: Vec<Vec<f64>>, l: usize, u: usize }` | LAPACK band storage format |

### 1.2 Invariants

| ID | Invariant | Violation Behavior |
|----|-----------|-------------------|
| LIN-001 | Matrix must be 2D: `a.ndim == 2` | `LinalgError::DimensionMismatch` |
| LIN-002 | Square matrix for solve/inv/det: `m == n` | `LinalgError::NotSquare` |
| LIN-003 | Compatible dimensions for Ax=b: `a.rows == b.len` | `LinalgError::DimensionMismatch` |
| LIN-004 | Finite elements when check_finite=true: `!any(NaN/Inf)` | `LinalgError::NonFiniteInput` |
| LIN-005 | Non-singular for solve/inv: `det(A) != 0` | `LinalgError::SingularMatrix` |
| LIN-006 | Positive definite for Cholesky: all eigenvalues > 0 | `LinalgError::NotPositiveDefinite` |
| LIN-007 | Symmetric for eigh/eigvalsh: `A == A^T` (within tolerance) | `LinalgError::NotSymmetric` |
| LIN-008 | Band storage: `ab.shape == (l+u+1, n)`, indices sorted | `LinalgError::InvalidBandStorage` |
| LIN-009 | Triangular solve: diagonal elements non-zero (unless unit_diagonal) | `LinalgError::SingularMatrix` |
| LIN-010 | SVD: `len(s) == min(m, n)`, singular values non-negative and sorted descending | Post-condition |
| LIN-011 | QR: `Q.shape == (m, m)` or `(m, min(m,n))` for economic, `R.shape == (min(m,n), n)` | Post-condition |
| LIN-012 | LU pivot: `piv.len == min(m, n)`, elements in `[0, m)` | Post-condition |
| LIN-013 | Eigenvalues of symmetric real matrix are all real | Post-condition |
| LIN-014 | Condition number: `cond(A) >= 1.0` for any norm | Post-condition |

### 1.3 State Transitions

```
Matrix input → [validate_finite] → [check_dimensions] → [select_algorithm via CASP]
  → [factorize] → [solve/extract] → Result
```

**Solver state machine:**
```
Unvalidated → validate_finite() → Validated
Validated → check_dimensions() → DimensionChecked
DimensionChecked → casp_select() → AlgorithmSelected
AlgorithmSelected → factorize() → Factorized | Error(SingularMatrix)
Factorized → back_substitute() → Solved
```

### 1.4 Shared State

- No global mutable state. All operations are pure functions.
- CASP solver portfolio maintains per-thread condition statistics (read-only after init).
- LAPACK info codes: 0 = success, < 0 = invalid argument, > 0 = algorithmic failure.

## 2. scipy.sparse

### 2.1 Core Data Structures

| Format | Storage | Description |
|--------|---------|-------------|
| CSR (Compressed Sparse Row) | `indptr[m+1], indices[nnz], data[nnz]` | Fast row slicing, matrix-vector multiply |
| CSC (Compressed Sparse Column) | `indptr[n+1], indices[nnz], data[nnz]` | Fast column slicing, used by direct solvers |
| COO (Coordinate) | `row[nnz], col[nnz], data[nnz]` | Construction format, allows duplicates |
| BSR (Block Sparse Row) | `indptr, indices, data[nnz_blocks, R, C]` | Block structure, fixed block size |
| DIA (Diagonal) | `offsets[ndiag], data[ndiag, n]` | Efficient for banded matrices |
| DOK (Dictionary of Keys) | `HashMap<(usize, usize), f64>` | Efficient incremental construction |
| LIL (List of Lists) | `rows[m]: Vec<(usize, f64)>` | Row-based incremental construction |

### 2.2 Invariants

| ID | Invariant | Violation Behavior |
|----|-----------|-------------------|
| SPA-001 | CSR: `indptr.len() == nrows + 1` | Construction error |
| SPA-002 | CSR: `indptr[0] == 0`, `indptr[nrows] == nnz` | Construction error |
| SPA-003 | CSR: `indptr` is monotonically non-decreasing | Construction error |
| SPA-004 | CSR: `indices[indptr[i]..indptr[i+1]]` are sorted and `< ncols` | Canonical form |
| SPA-005 | CSR: no duplicate indices within a row (after sum_duplicates) | Canonical form |
| SPA-006 | COO: `row[k] < nrows`, `col[k] < ncols` for all k | Construction error |
| SPA-007 | COO: duplicates are summed on conversion to CSR/CSC | Semantic contract |
| SPA-008 | BSR: `data.shape[1:] == (R, C)`, block dimensions divide matrix dimensions | Construction error |
| SPA-009 | DIA: `offsets` are unique and sorted | Canonical form |
| SPA-010 | Shape `(m, n)` with `m >= 0, n >= 0` | Construction error |
| SPA-011 | `0 <= nnz <= m * n` | Post-condition |
| SPA-012 | Format conversion preserves numerical values: `csr.toarray() == csc.toarray()` | Conformance test |

### 2.3 State Transitions

```
Construction (COO/DOK/LIL) → [sum_duplicates/sort] → Canonical Form
Canonical Form → [convert] → CSR/CSC (for computation)
CSR/CSC → [arithmetic ops] → CSR/CSC (may need re-canonicalization)
```

**Format conversion state:**
```
COO → tocsr() → CSR (sorted, duplicates summed)
COO → tocsc() → CSC (sorted, duplicates summed)
CSR → tocsc() → CSC (transpose of transpose trick)
Any → toarray() → Dense (always valid)
Any → todense() → Dense matrix (always valid)
```

### 2.4 Shared State

- No global state. Sparse matrices are immutable value types after construction.
- `has_canonical_format` flag tracks whether indices are sorted/deduplicated.

## 3. scipy.sparse.linalg

### 3.1 Core Data Structures

| Structure | Description |
|-----------|-------------|
| `LinearOperator(shape, matvec, rmatvec)` | Abstract matrix interface: only needs matvec |
| `SuperLU` | Opaque factorization from SuperLU direct solver |
| `ArpackNoConvergence` | Exception with partial eigenvalue results |
| Iterative solver state | Internal: residual vector, iteration count, convergence flag |

### 3.2 Invariants

| ID | Invariant |
|----|-----------|
| SPL-001 | LinearOperator: `shape == (m, n)`, `matvec` accepts `(n,)` returns `(m,)` |
| SPL-002 | Iterative solvers: `A` must be square for CG, GMRES, BiCGSTAB |
| SPL-003 | CG: `A` must be symmetric positive definite |
| SPL-004 | ARPACK `eigsh`: `k < n`, `A` symmetric |
| SPL-005 | ARPACK `eigs`: `k < n - 1` |
| SPL-006 | LOBPCG: initial vectors must be linearly independent |
| SPL-007 | Preconditioner shape must match system dimensions |

### 3.3 State Transitions

**Iterative solver:**
```
Init(A, b, x0, tol, maxiter) → Iterating(residual, iter_count)
Iterating → [converged: ||r|| < tol] → Converged(x, info=0)
Iterating → [maxiter reached] → NotConverged(x_best, info>0)
Iterating → [breakdown] → Failed(info<0)
```

## 4. scipy.integrate

### 4.1 Core Data Structures

| Structure | Description |
|-----------|-------------|
| `OdeSolver` (base) | Abstract IVP solver with stepping interface |
| `RK45/RK23` | Explicit Runge-Kutta with adaptive stepping |
| `Radau/BDF` | Implicit solvers for stiff problems |
| `LSODA` | Auto-switching stiff/non-stiff |
| `OdeResult` | `{ t: Vec<f64>, y: Vec<Vec<f64>>, t_events, ... }` |
| `IntegrationWarning` | Warnings for questionable convergence |

### 4.2 Invariants

| ID | Invariant |
|----|-----------|
| INT-001 | `t_span[0] < t_span[1]` (or `t_span[0] > t_span[1]` for backward) |
| INT-002 | `y0.len() == n` (system dimension) |
| INT-003 | `fun(t, y)` returns vector of same length as `y` |
| INT-004 | `rtol >= 0`, `atol >= 0`, at least one > 0 |
| INT-005 | Step size: `h_min <= h <= h_max` |
| INT-006 | `max_step > 0` |
| INT-007 | Quadrature: integration limits must be finite (or handled specially) |
| INT-008 | `nsum`: series terms must be monotonically decreasing for convergence |

### 4.3 State Transitions

**ODE Solver:**
```
Created(fun, t0, y0, t_bound) → [step()] → Running(t, y, h)
Running → [step()] → Running(t_new, y_new, h_new) | Finished
Running → [step() failed] → Failed(message)
Running → [t >= t_bound] → Finished(t_events, sol)
```

**Step controller:**
```
Propose(h) → [compute_error] → Accept(h_new = h * safety * err^(-1/order))
Propose(h) → [error too large] → Reject(h_new = h * max(0.1, safety * err^(-1/order)))
```

## 5. scipy.optimize

### 5.1 Core Data Structures

| Structure | Description |
|-----------|-------------|
| `OptimizeResult` | `{ x, fun, jac, hess, nfev, njev, nhev, success, message, ... }` |
| `Bounds(lb, ub)` | Box constraints: `lb[i] <= x[i] <= ub[i]` |
| `LinearConstraint(A, lb, ub)` | `lb <= A @ x <= ub` |
| `NonlinearConstraint(fun, lb, ub)` | `lb <= fun(x) <= ub` |
| `_minimize_*` | Method-specific internal state |
| `RootResults` | `{ root, converged, flag, function_calls, iterations }` |

### 5.2 Invariants

| ID | Invariant |
|----|-----------|
| OPT-001 | `x0` dimension matches problem dimension |
| OPT-002 | `Bounds`: `lb[i] <= ub[i]` for all i |
| OPT-003 | `LinearConstraint`: `A.shape[1] == len(x)` |
| OPT-004 | Gradient dimension: `jac(x).shape == (n,)` |
| OPT-005 | Hessian dimension: `hess(x).shape == (n, n)` and symmetric |
| OPT-006 | `maxiter > 0`, `tol > 0` |
| OPT-007 | `bracket` for scalar: `f(a)*f(b) < 0` (sign change) |
| OPT-008 | BFGS: Hessian approximation remains positive definite |
| OPT-009 | L-BFGS-B: history length `m > 0` (typically 5-20) |
| OPT-010 | Trust region: `0 < trust_radius` |

### 5.3 State Transitions

**Minimizer:**
```
Init(fun, x0, method, bounds, constraints) → Iterating(x_k, f_k, g_k)
Iterating → [compute_direction] → LineSearch/TrustRegion
LineSearch → [sufficient decrease] → Updated(x_{k+1})
Updated → [||g|| < gtol] → Converged(OptimizeResult)
Updated → [k >= maxiter] → MaxIterReached(OptimizeResult)
```

**Root finder (Brent/Newton):**
```
Init(f, bracket/x0) → Iterating(a, b, f(a), f(b))
Iterating → [|f(x)| < xtol] → Converged(root)
Iterating → [maxiter] → NotConverged(best_x)
```

## 6. scipy.fft

### 6.1 Core Data Structures

| Structure | Description |
|-----------|-------------|
| Plan (internal) | Precomputed FFT execution plan for given shape |
| Worker count | Thread-local integer, default 1 |
| Backend registry | Global registry of FFT backends |

### 6.2 Invariants

| ID | Invariant |
|----|-----------|
| FFT-001 | Input array rank >= 1 |
| FFT-002 | `n > 0` for 1D transforms (or None for input length) |
| FFT-003 | `s` shape tuple: all elements > 0 |
| FFT-004 | `axes` within valid range: `-ndim <= axis < ndim` |
| FFT-005 | `norm` in `{"backward", "ortho", "forward"}` |
| FFT-006 | `workers >= 1` or `-1` for all CPUs |
| FFT-007 | `next_fast_len(n)`: result is >= n, has only factors of 2, 3, 5 |
| FFT-008 | Real FFT output: `n//2 + 1` complex values (Hermitian symmetry) |
| FFT-009 | `fft(ifft(x)) == x` within tolerance (round-trip) |
| FFT-010 | Parseval's theorem: `sum(|x|^2) == sum(|X|^2)/N` |

### 6.3 State Transitions

- FFT functions are stateless (pure transforms).
- Worker context: `set_workers(n)` is a context manager (thread-local stack).
- Backend: `set_backend/set_global_backend` modify global registry.

```
set_workers(n) → [push to thread-local stack] → Context Active
Context Active → [exit] → [pop from stack] → Previous State
```

## 7. scipy.special

### 7.1 Core Data Structures

- Special functions are scalar/vectorized functions, no stateful objects.
- Output types: `f64`, `Complex64`, or `(f64, f64)` for (value, error estimate).

### 7.2 Invariants

| ID | Invariant |
|----|-----------|
| SPE-001 | `gamma(x)`: undefined at non-positive integers → `Inf` or `NaN` |
| SPE-002 | `gamma(x) > 0` for `x > 0` |
| SPE-003 | `beta(a, b)`: `a > 0, b > 0` |
| SPE-004 | `erf(x)`: range `[-1, 1]`, `erf(0) == 0`, `erf(Inf) == 1` |
| SPE-005 | `erfc(x) == 1 - erf(x)` within tolerance |
| SPE-006 | Bessel functions: recurrence relations must hold |
| SPE-007 | `comb(n, k)`: `n >= 0, 0 <= k <= n` for integer result |
| SPE-008 | `factorial(n)`: `n >= 0`, integer |
| SPE-009 | Branch cuts: `log(z)` cut along negative real axis |
| SPE-010 | `hyp2f1(a, b, c, z)`: `c` not a non-positive integer |
| SPE-011 | Legendre functions: `|x| <= 1` for associated Legendre on real line |

### 7.3 Shared State

- No mutable state. All special functions are pure.
- Some functions use internal caching for coefficients (thread-safe, lazy-init).

## 8. scipy.stats

### 8.1 Core Data Structures

| Structure | Description |
|-----------|-------------|
| `rv_continuous` | Base class for continuous distributions |
| `rv_discrete` | Base class for discrete distributions |
| `rv_frozen` | Distribution with fixed parameters (frozen) |
| Statistical test result | `NamedTuple(statistic, pvalue, ...)` |
| `gaussian_kde` | Kernel density estimation object |
| `qmc.QMCEngine` | Quasi-Monte Carlo sampler (Halton, Sobol, Latin Hypercube) |

### 8.2 Invariants

| ID | Invariant |
|----|-----------|
| STA-001 | PDF integrates to 1: `integral(pdf(x), -inf, inf) == 1` |
| STA-002 | CDF is monotonically non-decreasing: `cdf(a) <= cdf(b)` for `a <= b` |
| STA-003 | `0 <= cdf(x) <= 1` for all x |
| STA-004 | `ppf(cdf(x)) == x` (quantile is inverse of CDF) |
| STA-005 | `sf(x) == 1 - cdf(x)` |
| STA-006 | PMF sums to 1 for discrete distributions |
| STA-007 | `0 <= pvalue <= 1` for all statistical tests |
| STA-008 | Variance >= 0 |
| STA-009 | `rvs(size=n)` returns array of length n |
| STA-010 | Frozen distribution: parameters immutable after creation |
| STA-011 | KDE bandwidth > 0 |
| STA-012 | QMC: `0 <= sample[i] <= 1` for all dimensions |

### 8.3 State Transitions

**Distribution:**
```
Unfrozen(dist_class) → [dist(params)] → Frozen(loc, scale, shape_params)
Frozen → [rvs(n)] → Samples (stateless call)
Frozen → [pdf/cdf/ppf(x)] → Values (stateless call)
Frozen → [fit(data)] → New Frozen with MLE parameters
```

**QMC Engine:**
```
Init(d, scramble, seed) → Ready(index=0)
Ready → [random(n)] → Sampled(index=n), returns (n, d) array
Sampled → [random(m)] → Sampled(index=n+m) (continues sequence)
Sampled → [reset()] → Ready(index=0)
```

## 9. scipy.signal

### 9.1 Core Data Structures

| Structure | Description |
|-----------|-------------|
| `TransferFunction(num, den)` | Continuous-time TF: `H(s) = num(s)/den(s)` |
| `ZerosPolesGain(z, p, k)` | ZPK form: zeros, poles, gain |
| `StateSpace(A, B, C, D)` | State-space form: `dx/dt = Ax + Bu, y = Cx + Du` |
| `dlti` variants | Discrete-time equivalents with `dt` parameter |
| SOS (second-order sections) | `(n_sections, 6)` array: `[b0, b1, b2, a0, a1, a2]` per section |
| `iirdesign` result | Filter coefficients in BA, ZPK, or SOS form |
| Window functions | 1D arrays of window coefficients |

### 9.2 Invariants

| ID | Invariant |
|----|-----------|
| SIG-001 | TF: `den[0] != 0` (leading coefficient non-zero) |
| SIG-002 | StateSpace: A is (n,n), B is (n,m), C is (p,n), D is (p,m) |
| SIG-003 | SOS: `a[0] == 1.0` (normalized), shape `(n_sections, 6)` |
| SIG-004 | Stable system: all poles inside unit circle (discrete) or left half-plane (continuous) |
| SIG-005 | Causal filter: `len(num) <= len(den)` for proper TF |
| SIG-006 | `zpk2tf(tf2zpk(b, a)) == (b, a)` round-trip |
| SIG-007 | Window: `len(window) == M`, values in `[0, 1]` for most windows |
| SIG-008 | Sampling frequency: `fs > 0` |
| SIG-009 | Nyquist: filter frequencies in `[0, fs/2]` |
| SIG-010 | FIR filter: `a == [1.0]` (all-zero) |

### 9.3 State Transitions

**LTI system conversion:**
```
TF(num, den) ↔ ZPK(z, p, k) ↔ SS(A, B, C, D)
Continuous(s-domain) → [c2d(dt, method)] → Discrete(z-domain)
Discrete → [d2c()] → Continuous
```

## 10. scipy.spatial

### 10.1 Core Data Structures

| Structure | Description |
|-----------|-------------|
| `KDTree(data)` | k-d tree for fast nearest-neighbor queries |
| `cKDTree(data)` | C-accelerated k-d tree |
| `Delaunay(points)` | Delaunay triangulation of point set |
| `ConvexHull(points)` | Convex hull computation |
| `Voronoi(points)` | Voronoi diagram |
| `SphericalVoronoi(points)` | Voronoi on sphere surface |
| Distance matrix | `(n, n)` or condensed `(n*(n-1)/2,)` |
| `Rotation` | 3D rotation representation (quaternion internally) |

### 10.2 Invariants

| ID | Invariant |
|----|-----------|
| SPA-001 | KDTree: `data.shape == (n, m)` with `n >= 1, m >= 1` |
| SPA-002 | KDTree: `leafsize >= 1` |
| SPA-003 | Delaunay: `points.shape[1] >= 2` (at least 2D) |
| SPA-004 | Delaunay: `n_points >= ndim + 1` (need at least d+1 points in d dimensions) |
| SPA-005 | ConvexHull: vertices are subset of input points |
| SPA-006 | Distance metric: `d(x,x) == 0`, `d(x,y) >= 0`, `d(x,y) == d(y,x)` |
| SPA-007 | Condensed distance: `len == n*(n-1)/2` |
| SPA-008 | Rotation quaternion: `|q| == 1` (unit quaternion) |
| SPA-009 | Rotation: `R @ R.T == I` (orthogonal) |
| SPA-010 | `det(R) == 1` (proper rotation, not reflection) |

### 10.3 State Transitions

**KDTree:**
```
Build(data, leafsize) → Ready(tree_structure)
Ready → [query(x, k)] → Results(distances, indices)
Ready → [query_ball_point(x, r)] → Results(indices)
Ready → [query_ball_tree(other, r)] → Results(lists_of_indices)
```

**Rotation:**
```
from_quat(q) → Rotation | from_matrix(R) → Rotation | from_rotvec(v) → Rotation
Rotation → [apply(vectors)] → Rotated vectors
Rotation → [inv()] → Inverse Rotation
Rotation * Rotation → Composed Rotation
```

## 11. scipy.interpolate

### 11.1 Core Data Structures

| Structure | Description |
|-----------|-------------|
| `BSpline(t, c, k)` | B-spline: knot vector t, coefficients c, degree k |
| `PPoly(c, x)` | Piecewise polynomial: coefficients c, breakpoints x |
| `NdPPoly(c, x)` | N-dimensional piecewise polynomial |
| `interp1d(x, y)` | 1D interpolation (deprecated in favor of make_interp_spline) |
| `RegularGridInterpolator` | N-D interpolation on regular grid |
| `CloughTocher2DInterpolator` | Smooth 2D interpolation on unstructured data |
| `RBFInterpolator` | Radial basis function interpolation |
| Knot sequence | 1D sorted array with multiplicity rules |

### 11.2 Invariants

| ID | Invariant |
|----|-----------|
| INP-001 | Knots: `t` is non-decreasing |
| INP-002 | Knots: `len(t) == len(c) + k + 1` for B-spline |
| INP-003 | Degree: `k >= 0`, typically `k in {1, 2, 3, 5}` |
| INP-004 | Breakpoints: `x` is strictly increasing |
| INP-005 | Data: `len(x) == len(y)` for 1D interpolation |
| INP-006 | Schoenberg-Whitney: `t[i] < x[i] < t[i+k+1]` for all i |
| INP-007 | PPoly: `c.shape[1] == len(x) - 1` (n_intervals) |
| INP-008 | PPoly: `c.shape[0] == k + 1` (order of polynomial) |
| INP-009 | Extrapolation: default is constant or linear extension |
| INP-010 | Regular grid: each axis is strictly monotonic |

### 11.3 State Transitions

**BSpline:**
```
make_interp_spline(x, y, k) → BSpline(t, c, k)
BSpline → [__call__(x_new)] → interpolated values
BSpline → [derivative(nu)] → BSpline of degree k-nu
BSpline → [antiderivative()] → BSpline of degree k+1
BSpline → [integrate(a, b)] → scalar
```

## 12. scipy.ndimage

### 12.1 Core Data Structures

- Operates on N-dimensional arrays (no special types).
- Structure elements: binary N-D arrays defining neighborhoods.
- Output arrays: same shape as input (or derived from parameters).

### 12.2 Invariants

| ID | Invariant |
|----|-----------|
| NDI-001 | Input array: ndim >= 1 |
| NDI-002 | Structure element: same ndim as input |
| NDI-003 | Boundary mode: one of `{"reflect", "constant", "nearest", "mirror", "wrap"}` |
| NDI-004 | `sigma > 0` for Gaussian filter |
| NDI-005 | `size` is positive odd integer (for uniform/median filter) |
| NDI-006 | Label array: non-negative integers, same shape as input |
| NDI-007 | Morphological operations: structure is binary |
| NDI-008 | Output shape matches input shape for element-wise filters |
| NDI-009 | `zoom` factors > 0 |
| NDI-010 | `rotate` angle in degrees, `reshape` flag controls output size |

### 12.3 Shared State

- No mutable state. All functions are pure array-to-array transforms.

## 13. scipy.io

### 13.1 Core Data Structures

| Structure | Description |
|-----------|-------------|
| `MatFile5Reader/Writer` | MATLAB v5 file read/write |
| `MatFile4Reader/Writer` | MATLAB v4 file read/write |
| `FortranFile` | Fortran unformatted sequential file reader |
| `NetCDFFile` | NetCDF file reader (v3) |
| `WavFileWarning` | Warning for WAV file issues |
| `arff.MetaData` | ARFF file metadata |

### 13.2 Invariants

| ID | Invariant |
|----|-----------|
| IO-001 | MAT file: magic bytes identify version (v4 vs v5) |
| IO-002 | MAT file: variable names are valid MATLAB identifiers |
| IO-003 | WAV: sample rate > 0, bit depth in {8, 16, 24, 32} |
| IO-004 | WAV: data shape `(n_samples,)` or `(n_samples, n_channels)` |
| IO-005 | NetCDF: dimension sizes > 0, one unlimited dimension allowed |
| IO-006 | Fortran: record length marker matches at start and end of record |
| IO-007 | HB (Harwell-Boeing): sparse matrix in specific column format |

### 13.3 State Transitions

**File reader:**
```
Open(path/stream) → [detect_version] → Identified(v4/v5)
Identified → [read_header] → HeaderParsed
HeaderParsed → [read_variables] → Loaded(dict of arrays)
```

**File writer:**
```
Init(path, options) → [write_header] → HeaderWritten
HeaderWritten → [put_variables(dict)] → Written
Written → [close] → Closed
```

## 14. scipy.cluster

### 14.1 Core Data Structures

| Structure | Description |
|-----------|-------------|
| Linkage matrix Z | `(n-1, 4)` array: `[idx1, idx2, distance, cluster_size]` |
| `ClusterNode` | Tree node for dendrogram traversal |
| Flat clusters | 1D array of cluster labels |
| `DisjointSet` | Union-find data structure |
| Codebook (vq) | `(k, d)` array of centroids |

### 14.2 Invariants

| ID | Invariant |
|----|-----------|
| CLU-001 | Linkage Z: `Z.shape == (n-1, 4)` for n observations |
| CLU-002 | Linkage Z: `Z[i, 0]` and `Z[i, 1]` are valid cluster indices |
| CLU-003 | Linkage Z: `Z[i, 2] >= 0` (non-negative distance) |
| CLU-004 | Linkage Z: `Z[i, 3] >= 2` (cluster size) |
| CLU-005 | Linkage Z: distances are non-decreasing for `single`, `complete`, `average` |
| CLU-006 | Flat clusters: `1 <= k <= n` clusters |
| CLU-007 | K-means: `k <= n` (cannot have more centroids than points) |
| CLU-008 | Codebook: `codebook.shape[1] == data.shape[1]` |
| CLU-009 | Distance input: condensed form has `n*(n-1)/2` elements |

### 14.3 State Transitions

**Hierarchical clustering:**
```
Distance matrix → [linkage(method)] → Linkage Z
Linkage Z → [fcluster(t, criterion)] → Flat clusters
Linkage Z → [dendrogram()] → Plot data
Linkage Z → [to_tree()] → ClusterNode tree
```

## 15. scipy.constants

### 15.1 Core Data Structures

- All values are `f64` constants (immutable).
- `physical_constants` dict: `name -> (value, unit, uncertainty)`.

### 15.2 Invariants

| ID | Invariant |
|----|-----------|
| CON-001 | All physical constants are positive (by convention) |
| CON-002 | `c` (speed of light) = 299792458.0 m/s (exact) |
| CON-003 | `h` (Planck) > 0 |
| CON-004 | Unit conversion factors > 0 |
| CON-005 | `find(query)` returns subset of `physical_constants` keys |

### 15.3 Shared State

- Entirely immutable. No state transitions.
- CODATA values are compile-time constants (2018 edition in SciPy 1.17).

## 16. scipy.misc

### 16.1 Status

- **Deprecated** in SciPy 1.17 (will be removed in 2.0).
- `scipy.misc` module emits `DeprecationWarning` on import.
- Functions moved to other modules: `derivative` → `scipy.differentiate`.

### 16.2 Invariants

- Same invariants as the modules the functions were moved to.

## 17. scipy.odr

### 17.1 Core Data Structures

| Structure | Description |
|-----------|-------------|
| `Model(fcn)` | Model function and optional Jacobians |
| `RealData(x, y, sx, sy)` | Data with error bars |
| `Data(x, y, we, wd)` | Data with weight matrices |
| `ODR(data, model, beta0)` | Orthogonal distance regression problem |
| `Output` | Regression results: `beta, sd_beta, cov_beta, ...` |

### 17.2 Invariants

| ID | Invariant |
|----|-----------|
| ODR-001 | `beta0.len()` matches model parameter count |
| ODR-002 | `x.shape[1] == y.shape[1]` (same number of observations) |
| ODR-003 | Weights must be positive |
| ODR-004 | `maxit > 0` |

### 17.3 State Transitions

```
ODR(data, model, beta0) → [run()] → Output(beta, sd_beta, cov_beta, info)
Output → [restart()] → ODR (continue from last beta)
```

## 18. scipy.datasets

### 18.1 Core Data Structures

- Functions that return cached datasets: `ascent()`, `face()`, `electrocardiogram()`.
- Download/cache managed by `pooch`.

### 18.2 Invariants

| ID | Invariant |
|----|-----------|
| DAT-001 | `ascent()` returns `(512, 512)` uint8 array |
| DAT-002 | `face()` returns `(768, 1024, 3)` uint8 array |
| DAT-003 | `electrocardiogram()` returns 1D float64 array, length 108000 |
| DAT-004 | Returned arrays are read-only copies |

### 18.3 Shared State

- Cache directory: `~/.cache/scipy-data/` (configurable via env var).
- First call downloads; subsequent calls read from cache.

## 19. scipy.differentiate

### 19.1 Core Data Structures

- Pure functions, no stateful objects.
- Input: callable `f(x)`, evaluation point `x`.
- Output: derivative estimate with error bound.

### 19.2 Invariants

| ID | Invariant |
|----|-----------|
| DIF-001 | `f` must be callable |
| DIF-002 | `x` must be finite |
| DIF-003 | Step size `h > 0` (or auto-selected) |
| DIF-004 | Order `n >= 1` for n-th derivative |
| DIF-005 | Richardson extrapolation: `order >= n + 2` |

## 20. scipy.linalg.lapack / scipy.linalg.blas

### 20.1 Data Model

- Thin wrappers around LAPACK/BLAS routines.
- Type prefix convention: `s` (f32), `d` (f64), `c` (complex64), `z` (complex128).
- All routines return `info` code: 0 = success, < 0 = bad argument, > 0 = algorithmic failure.

### 20.2 Invariants

| ID | Invariant |
|----|-----------|
| LAP-001 | LAPACK info: `info == 0` on success |
| LAP-002 | LAPACK info: `info == -k` means k-th argument is invalid |
| LAP-003 | BLAS: `alpha`, `beta` are scalars |
| LAP-004 | Leading dimension: `lda >= max(1, m)` |
| LAP-005 | Work array: `lwork >= minimum_workspace(n)` |

## Summary

| Domain | Data Structures | Invariants | Stateful Objects | Shared State |
|--------|----------------|------------|-----------------|--------------|
| linalg | 10 | 14 | No | CASP portfolio (read-only) |
| sparse | 7 formats | 12 | No (canonical flag) | None |
| sparse.linalg | 4 | 7 | Iterative solvers | None |
| integrate | 6 | 8 | ODE solvers | None |
| optimize | 6 | 10 | Minimizers/root-finders | None |
| fft | 3 | 10 | Worker context (thread-local) | Backend registry |
| special | 0 (pure functions) | 11 | No | Coefficient caches |
| stats | 6 | 12 | QMC engines | None |
| signal | 7 | 10 | No | None |
| spatial | 8 | 10 | KDTree, triangulations | None |
| interpolate | 8 | 10 | No | None |
| ndimage | 0 (pure functions) | 10 | No | None |
| io | 6 | 7 | File readers/writers | None |
| cluster | 5 | 9 | No | None |
| constants | 1 (dict) | 5 | No | Compile-time constants |
| misc | 0 (deprecated) | 0 | No | None |
| odr | 5 | 4 | ODR solver | None |
| datasets | 0 | 4 | No | Download cache |
| differentiate | 0 (pure functions) | 5 | No | None |
| lapack/blas | 0 (wrappers) | 5 | No | None |
| **TOTAL** | **~92** | **~163** | ~8 families | ~3 patterns |

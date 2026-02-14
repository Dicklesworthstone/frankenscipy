# DOC-PASS-04: Execution Path Tracing

Generated: 2026-02-14
Scope: Top-20 most-used SciPy functions -- complete control flow tracing from entry point to LAPACK/backend call
Source: `/data/projects/frankenscipy/legacy_scipy_code/scipy/scipy/` (SciPy 1.17.0)

Cross-references:
- DOC-PASS-01 (Module Cartography): module tree, file locations, cross-module dependencies
- DOC-PASS-02 (API Census): public symbol inventory, source module mapping
- DOC-PASS-03 (Data Model Invariants): invariant IDs (LIN-xxx, SPA-xxx, INT-xxx, etc.)

---

## Table of Contents

1. [scipy.linalg.solve](#1-scipylinalgsolve)
2. [scipy.linalg.inv](#2-scipylinalginv)
3. [scipy.linalg.eig](#3-scipylinalgeig)
4. [scipy.linalg.eigh](#4-scipylinalgeigh)
5. [scipy.linalg.svd](#5-scipylinalgsvd)
6. [scipy.linalg.det](#6-scipylinalgdet)
7. [scipy.linalg.lstsq](#7-scipylinalglstsq)
8. [scipy.integrate.solve_ivp](#8-scipyintegratesolve_ivp)
9. [scipy.integrate.quad](#9-scipyintegratequad)
10. [scipy.optimize.minimize](#10-scipyoptimizeminimize)
11. [scipy.optimize.root](#11-scipyoptimizeroot)
12. [scipy.optimize.curve_fit](#12-scipyoptimizecurve_fit)
13. [scipy.optimize.linprog](#13-scipyoptimizelinprog)
14. [scipy.fft.fft](#14-scipyfftfft)
15. [scipy.fft.ifft](#15-scipyfftifft)
16. [scipy.sparse.linalg.spsolve](#16-scipysparselinalgspsolve)
17. [scipy.sparse.linalg.eigsh](#17-scipysparselalgeigsh)
18. [scipy.interpolate.interp1d](#18-scipyinterpolateinterp1d)
19. [scipy.interpolate.CubicSpline](#19-scipyinterpolatecubicspline)
20. [scipy.stats.norm (distribution interface)](#20-scipystatsnorm-distribution-interface)
21. [scipy.signal.fftconvolve](#21-scipysignalfftconvolve)

---

## 1. scipy.linalg.solve

**Source**: `linalg/_basic.py:58`
**Signature**: `solve(a, b, lower=False, overwrite_a=False, overwrite_b=False, check_finite=True, assume_a=None, transposed=False)`
**Invariants**: LIN-001, LIN-002, LIN-003, LIN-004, LIN-005 (see DOC-PASS-03)

### Entry Point and Argument Validation

```
solve(a, b, ...)
  |
  +-- Map assume_a string to integer structure code:
  |     None -> -1 (auto-detect), 'gen' -> 0, 'diagonal' -> 11,
  |     'tridiagonal' -> 31, 'banded' -> 41, 'upper triangular' -> 21,
  |     'lower triangular' -> 22, 'pos' -> 101, 'sym' -> 201, 'her' -> 211
  |     Unknown string -> ValueError
  |
  +-- _asarray_validated(a, check_finite) -> a1  [LIN-004]
  +-- _asarray_validated(b, check_finite) -> b1
  +-- _ensure_dtype_cdsz(a1, b1) -> unified LAPACK-compatible dtype
  +-- _normalize_lapack_dtype(a1) -> upcast non-LAPACK types (int->float64)
  +-- _ensure_aligned_and_native(a1) -> C-contiguous, native byte order
  +-- _ensure_aligned_and_native(b1)
  |
  +-- Validate: a1.ndim >= 2               [LIN-001]
  +-- Validate: a1.shape[-1] == a1.shape[-2]  [LIN-002]
  +-- complex + transposed -> NotImplementedError
  +-- b_is_1D: reshape b1 to column vector if 1D
  +-- Validate: b1.shape[-2] == a1.shape[-1]  [LIN-003]
  +-- Broadcast batch dimensions via np.broadcast_shapes
```

### Decision Tree

```
                         solve(a, b)
                             |
                    +--------+--------+
                    |                 |
              a.size == 0?      a is scalar?
                YES                 YES
                 |                   |
          return empty         a == 0? -> LinAlgError("singular")
                               else: return b / a
                                     |
                                    NO (to both)
                                     |
                    _batched_linalg._solve(a1, b1, structure, lower, transposed,
                                           overwrite_a, overwrite_b)
                                     |
                         +-----------+-----------+
                         |                       |
                   structure == -1          structure known
                   (auto-detect)           (user provided)
                         |                       |
                   C code probes each      Direct dispatch to
                   slice for structure:    appropriate LAPACK
                   diagonal, triangular,   routine without
                   symmetric, pos-def,     per-slice checking
                   hermitian, or general
                         |
                   Per-slice dispatch to LAPACK:
                   gen -> ?GETRF + ?GETRS
                   pos -> ?POTRF + ?POTRS
                   sym -> ?SYSV
                   her -> ?HESV
                   diagonal -> element-wise division
                   triangular -> ?TRTRS
                   banded -> ?GBSV
                   tridiagonal -> ?GTSV
```

### Hot Path

The most common case is `assume_a=None` (auto-detect) with a dense, real, non-batched, 2D square matrix.
The C code (`_batched_linalg._solve`) runs structure detection on the single slice, typically finding
it is "general", then calls `dgesv` (LU factorize + solve) via LAPACK.

### Fallback Chain

There is no fallback between algorithms. If the selected LAPACK routine reports failure
(info > 0 for singularity), the error list is populated and `_format_emit_errors_warnings`
raises `LinAlgError` or emits `LinAlgWarning` for ill-conditioning.

### Error Propagation

```
LAPACK info code:
  info == 0      -> success
  info > 0       -> "singular matrix" -> LinAlgError
  info < 0       -> "illegal argument" (internal bug) -> LinAlgError
  ill-conditioned -> LinAlgWarning (via rcond estimation in C layer)
```

Errors from `_batched_linalg._solve` are collected in `err_lst` (a list of per-slice error
descriptors) and processed by `_format_emit_errors_warnings`, which raises on the first
fatal error or warns for non-fatal conditions.

---

## 2. scipy.linalg.inv

**Source**: `linalg/_basic.py:968`
**Signature**: `inv(a, overwrite_a=False, check_finite=True, *, assume_a=None, lower=False)`
**Invariants**: LIN-001, LIN-002, LIN-004, LIN-005

### Entry Point and Argument Validation

```
inv(a, ...)
  |
  +-- _asarray_validated(a, check_finite) -> a1  [LIN-004]
  +-- Validate: a1.ndim >= 2                     [LIN-001]
  +-- Validate: a1.shape[-1] == a1.shape[-2]     [LIN-002]
  +-- Empty matrix: return empty with correct dtype
  +-- _normalize_lapack_dtype(a1)
  +-- _ensure_aligned_and_native(a1)
  +-- Map assume_a to structure integer (same map as solve, minus banded/tridiagonal)
```

### Decision Tree

```
                         inv(a)
                             |
                    +--------+--------+
                    |                 |
              a.size == 0?       structure code
                 YES                  |
                 |           +--------+--------+--------+
           return empty      |        |        |        |
                           gen/None  pos      sym/her  diag/tri
                             |        |        |        |
                          ?GETRF    ?POTRF    ?SYTRF  element-wise
                          ?GETRI    ?POTRI    ?SYTRI  reciprocal
                             |
                  _batched_linalg._inv(a1, structure, overwrite_a, lower)
```

### Hot Path

Dense, real, general matrix: `dgetrf` (LU factorize) followed by `dgetri` (compute inverse from LU).
When `assume_a=None`, auto-detection may identify diagonal or triangular structure for fast inversion.

### Fallback Chain

No algorithmic fallback. LAPACK info > 0 indicates singularity.

### Error Propagation

Same pattern as `solve`: `err_lst` collected, processed by `_format_emit_errors_warnings`.
Singular matrix raises `LinAlgError`. Ill-conditioned matrix emits `LinAlgWarning`.

---

## 3. scipy.linalg.eig

**Source**: `linalg/_decomp.py:67`
**Signature**: `eig(a, b=None, left=False, right=True, overwrite_a=False, overwrite_b=False, check_finite=True, homogeneous_eigvals=False)`
**Invariants**: LIN-001, LIN-002, LIN-004

### Entry Point and Argument Validation

```
eig(a, b=None, ...)
  |
  +-- _asarray_validated(a, check_finite) -> a1     [LIN-004]
  +-- Validate: a1.shape[-1] == a1.shape[-2]        [LIN-002]
  +-- _normalize_lapack_dtype(a1)
  +-- _ensure_aligned_and_native(a1)
  +-- Empty matrix: return empty eigenvalue/eigenvector arrays
  +-- If b is not None:
  |     +-- _asarray_validated(b, check_finite)
  |     +-- _ensure_dtype_cdsz(a1, b1) -> match dtypes
  |     +-- Validate b is square, same shape as a
  |     +-- Broadcast batch dimensions
```

### Decision Tree

```
                         eig(a, b)
                             |
                    +--------+--------+
                    |                 |
              b is None         b is not None
           (standard)         (generalized)
                    |                 |
       _batched_linalg._eig     _batched_linalg._eig
         (a1, left, right)       (a1, left, right, b1)
                    |                 |
              LAPACK ?GEEV       LAPACK ?GGEV
                    |                 |
       _check_format_errors     _check_format_errors
         ("geev", err_lst)       ("ggev", err_lst)
                    |                 |
                    +--------+--------+
                             |
            _make_eigvals(w, beta, homogeneous_eigvals)
                             |
            If real a and all eigenvalues real:
              cast eigenvectors to real
                             |
            Return based on (left, right) flags:
              (!left, !right) -> w
              (left, right)   -> w, vl, vr
              (left, !right)  -> w, vl
              (!left, right)  -> w, vr
```

### Hot Path

Standard eigenvalue problem (`b=None`) with real matrix: calls LAPACK `dgeev`.
The C layer (`_batched_linalg._eig`) handles batching, performing `dgeev` per slice.
Returns complex eigenvalues and optionally right eigenvectors (column-normalized).

### Fallback Chain

No fallback. If LAPACK `?geev` or `?ggev` does not converge (info > 0), `_check_format_errors_warnings`
raises `LinAlgError`.

### Error Propagation

```
_check_format_errors_warnings(routine_name, err_lst):
  info < 0 -> "Illegal value" (internal) -> LinAlgError
  info > 0 -> "Failed to converge" -> LinAlgError
```

For the generalized problem, eigenvectors from `?ggev` are NOT normalized by LAPACK; SciPy
normalizes them post-hoc via `np.linalg.vector_norm`.

---

## 4. scipy.linalg.eigh

**Source**: `linalg/_decomp.py:294`
**Signature**: `eigh(a, b=None, *, lower=True, eigvals_only=False, overwrite_a=False, overwrite_b=False, type=1, check_finite=True, subset_by_index=None, subset_by_value=None, driver=None)`
**Invariants**: LIN-001, LIN-002, LIN-004, LIN-007, LIN-013
**Note**: Decorated with `@_apply_over_batch(('a', 2), ('b', 2))` -- batching handled at decorator level.

### Entry Point and Argument Validation

```
eigh(a, b=None, ...)
  |
  +-- Validate driver in [None, "ev", "evd", "evr", "evx", "gv", "gvd", "gvx"]
  +-- _asarray_validated(a, check_finite) -> a1
  +-- Validate: a1 is square 2D                  [LIN-001, LIN-002]
  +-- Empty matrix: return empty w/v with correct dtypes
  +-- If b is not None:
  |     +-- Validate b square, same shape as a
  |     +-- Validate type in [1, 2, 3]
  |
  +-- Determine subset mode:
  |     subset_by_index -> 'I' range mode (Fortran 1-indexed)
  |     subset_by_value -> 'V' value mode
  |     Both -> ValueError
  |
  +-- Determine driver:
  |     b is None: default "evr" (standard)
  |     b is not None: "gvd" (full) or "gvx" (subset)
  +-- Prefix: 'he' if complex, 'sy' if real
```

### Decision Tree

```
                           eigh(a, b)
                               |
                      +--------+--------+
                      |                 |
                 b is None        b is not None
              (standard)         (generalized)
                      |                 |
               pfx + driver      pfx + driver
                      |                 |
       +---------+----+----+---------+  +------+------+
       |         |         |         |  |      |      |
      ev       evd       evr       evx gv    gvd    gvx
       |         |         |         |  |      |      |
   ?SYEV/    ?SYEVD/   ?SYEVR/   ?SYEVX ?HEGV  ?HEGVD ?HEGVX
   ?HEEV     ?HEEVD    ?HEEVR    ?HEEVX (no lwork query)
       |
    All drivers:
      compute lwork via _compute_lwork(drvlw)
      call drv(a=a1, b=b1, ...)
      returns (w, v, *other_args, info)
```

### Hot Path

Standard problem, default driver: `dsyevr` (MRRR algorithm -- fastest for full/subset eigenvalues
of real symmetric matrices). This is the recommended path for most use cases.

### Fallback Chain

No automatic fallback between drivers. If the chosen LAPACK routine fails, it raises `LinAlgError`.

### Error Propagation

```
info == 0  -> success, return (w) or (w, v)
info < -1  -> "Illegal value in argument {-info}" -> LinAlgError
info > n   -> "Leading minor of order {info-n} of B not pos-def" -> LinAlgError
info > 0   -> Driver-specific:
              ev/gv:  "{info} off-diag elements didn't converge" -> LinAlgError
              evx/gvx: "{info} eigenvectors failed to converge" -> LinAlgError
              evd/gvd: "Failed on submatrix" -> LinAlgError
              evr:     "Internal Error" -> LinAlgError
```

---

## 5. scipy.linalg.svd

**Source**: `linalg/_decomp_svd.py:37`
**Signature**: `svd(a, full_matrices=True, compute_uv=True, overwrite_a=False, check_finite=True, lapack_driver='gesdd')`
**Invariants**: LIN-001, LIN-004, LIN-010

### Entry Point and Argument Validation

```
svd(a, ...)
  |
  +-- Validate: lapack_driver in ('gesdd', 'gesvd')
  +-- _asarray_validated(a, check_finite) -> a1        [LIN-004]
  +-- Validate: a1.ndim >= 2                           [LIN-001]
  +-- _normalize_lapack_dtype(a1)
  +-- _ensure_aligned_and_native(a1)
  +-- Empty matrix: return identity U/Vh, empty s
  +-- ILP64 overflow check for full_matrices with very large matrices
```

### Decision Tree

```
                         svd(a)
                             |
                    +--------+--------+
                    |                 |
            lapack_driver          lapack_driver
            == 'gesdd'             == 'gesvd'
            (divide & conquer)     (general rectangular)
                    |                     |
            _batched_linalg._svd(a1, 'gesdd', ...)
            _batched_linalg._svd(a1, 'gesvd', ...)
                    |
            Returns (u, s, vt, err_lst) or (s, err_lst)
                    |
            If err_lst: _format_emit_errors_warnings
                    |
            compute_uv=True:  return (u, s, vt)
            compute_uv=False: return s
```

### Hot Path

Default driver `gesdd` (divide-and-conquer) is fastest for most matrices. For a real
`m x n` matrix, calls LAPACK `dgesdd`. The `full_matrices=False` (economic SVD) reduces
output size to `U: (m, min(m,n))`, `Vh: (min(m,n), n)`.

### Fallback Chain

No automatic fallback from `gesdd` to `gesvd`. If `gesdd` fails to converge, the user
must manually try `gesvd` (which is slower but more robust in rare edge cases).

### Error Propagation

```
err_lst from _batched_linalg._svd:
  info == 0 -> success
  info > 0  -> "SVD did not converge" -> LinAlgError
  info < 0  -> "Illegal argument" -> LinAlgError
```

---

## 6. scipy.linalg.det

**Source**: `linalg/_basic.py:1109`
**Signature**: `det(a, overwrite_a=False, check_finite=True)`
**Invariants**: LIN-001, LIN-002, LIN-004

### Entry Point and Argument Validation

```
det(a, ...)
  |
  +-- np.asarray_chkfinite(a) or np.asarray(a)   [LIN-004]
  +-- Validate: a1.ndim >= 2                      [LIN-001]
  +-- Validate: a1.shape[-1] == a1.shape[-2]      [LIN-002]
  +-- _normalize_lapack_dtype(a1)
```

### Decision Tree

```
                         det(a)
                             |
               +-------------+-------------+
               |             |             |
          min(shape)==0  shape[-2:]==(1,1)  general case
               |             |             |
          return 1.0    return a[0,0]      |
          (empty det     (scalar det)      |
           is 1)                           |
                                  +--------+--------+
                                  |                 |
                            a1.ndim == 2       a1.ndim > 2
                                  |                 |
                         find_det_from_lu(a1)  loop over batch dims:
                         -> np.float64/         find_det_from_lu(a1[ind])
                            np.complex128       -> np.array of dets
```

`find_det_from_lu` performs LU factorization via LAPACK `?getrf`, then computes the product of
diagonal entries of U, adjusting sign by counting row swaps in the pivot array.

### Hot Path

Single 2D real matrix: calls Cython `find_det_from_lu` which internally calls `dgetrf`,
computes product of diagonal, returns as `np.float64`. Single-precision inputs are upcast
to double precision to prevent overflow.

### Fallback Chain

None. LU factorization always succeeds (even for singular matrices, where det = 0.0).

### Error Propagation

This function does not raise on singular matrices (det is simply 0.0). The only errors
are input validation errors (`ValueError` for non-square or non-2D input).

---

## 7. scipy.linalg.lstsq

**Source**: `linalg/_basic.py:1237`
**Signature**: `lstsq(a, b, cond=None, overwrite_a=False, overwrite_b=False, check_finite=True, lapack_driver=None)`
**Invariants**: LIN-001, LIN-003, LIN-004

### Entry Point and Argument Validation

```
lstsq(a, b, ...)
  |
  +-- Select driver: default 'gelsd', or user-specified 'gelsy'/'gelss'
  +-- Validate: a.ndim >= 2                         [LIN-001]
  +-- _asarray_validated(a) -> a1, _asarray_validated(b) -> b1
  +-- _ensure_dtype_cdsz(a1, b1) -> unified dtype
  +-- _normalize_lapack_dtype, _ensure_aligned_and_native
  +-- Zero-sized problem: return zeros, residues, rank=0, empty s
  +-- b_is_1D: reshape to column if 1D
  +-- Validate: m == b1.shape[-2]                   [LIN-003]
  +-- Broadcast batch dimensions
  +-- cond default: np.finfo(a1.dtype).eps
```

### Decision Tree

```
                         lstsq(a, b)
                             |
                    +--------+--------+--------+
                    |                 |        |
               driver='gelsd'   driver='gelsy' driver='gelss'
               (default, SVD    (QR with      (SVD-based,
                divide&conquer)  pivoting)      classical)
                    |                 |        |
            _batched_linalg._lstsq(a1, b1, cond, driver)
                    |
            Returns (x, rank, S, err_lst)
                    |
            If m > n: compute residuals = ||b - a @ x||^2
            If b_is_1D: squeeze output
            Return (x, residuals, rank, s)
```

### Hot Path

Default driver `gelsd` uses SVD via divide-and-conquer internally. For overdetermined systems
(m > n), this is the most common path. Returns solution x, residuals, effective rank, and
singular values.

### Fallback Chain

No automatic fallback between drivers.

### Error Propagation

```
err_lst from _batched_linalg._lstsq:
  info == 0 -> success
  info > 0  -> "SVD did not converge" (for gelsd/gelss) -> LinAlgError
  info < 0  -> "Illegal argument" -> LinAlgError
```

Note: `gelsy` does not compute singular values; `S` is returned as `None` in that case.

---

## 8. scipy.integrate.solve_ivp

**Source**: `integrate/_ivp/ivp.py:160`
**Signature**: `solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, args=None, **options)`
**Invariants**: INT-001, INT-002, INT-003, INT-004, INT-005, INT-006

### Entry Point and Argument Validation

```
solve_ivp(fun, t_span, y0, ...)
  |
  +-- Validate method in METHODS dict or is OdeSolver subclass
  +-- t0, tf = map(float, t_span)
  +-- If args is not None:
  |     wrap fun -> lambda t, x: fun(t, x, *args)
  |     wrap jac if callable
  |     wrap event functions
  +-- If t_eval is not None:
  |     Validate: 1D, within t_span, properly sorted
  +-- Resolve method string to class: METHODS[method]
  +-- Instantiate solver: method(fun, t0, y0, tf, vectorized=vectorized, **options)
```

### Decision Tree

```
                       solve_ivp(fun, t_span, y0)
                              |
               +--------------+--------------+
               |              |              |
          method='RK45'  method='Radau'  method='LSODA'
          (default)      /BDF (stiff)    (auto-switch)
               |              |              |
           RK45(...)     Radau/BDF(...)  LSODA(...)
               |              |              |
               +--------------+--------------+
                              |
                     Main integration loop:
                     while status is None:
                       |
                       +-- solver.step() -> message
                       |
                       +-- Check solver.status:
                       |     'finished' -> status = 0
                       |     'failed'   -> status = -1, break
                       |
                       +-- If dense_output: sol = solver.dense_output()
                       |
                       +-- If events is not None:
                       |     evaluate g_new = [event(t, y) for event in events]
                       |     find_active_events(g, g_new, event_dir)
                       |     if active: solve_event_equation via brentq
                       |     if terminal event: status = 1
                       |
                       +-- Collect t, y (or interpolate at t_eval points)
                              |
                     Build OdeResult(t, y, sol, t_events, y_events,
                                     nfev, njev, nlu, status, message, success)
```

### Hot Path

Default `method='RK45'`: Dormand-Prince 5(4) with local extrapolation. Each `step()` call
computes 6 stages (7 with FSAL), estimates error, adapts step size. No Jacobian needed.
Most problems are non-stiff and use this path exclusively.

### Fallback Chain

There is no automatic fallback between solvers. If RK45 fails (e.g., stiff problem causing
excessive step rejections), `solver.status` becomes `'failed'` and the result has `success=False`.
The user must manually switch to `'Radau'` or `'BDF'` for stiff problems. `'LSODA'` provides
automatic stiffness detection and switching between Adams (non-stiff) and BDF (stiff) methods,
but wraps legacy Fortran code.

### Error Propagation

```
status == 0  -> success ("reached end of integration interval")
status == 1  -> terminal event occurred (success=True)
status == -1 -> integration step failed (success=False)
                message contains solver-specific failure description
```

Events are detected by sign change in `event(t, y)` between consecutive steps. Root finding
uses `scipy.optimize.brentq` with tolerance `4 * EPS`.

---

## 9. scipy.integrate.quad

**Source**: `integrate/_quadpack_py.py:23`
**Signature**: `quad(func, a, b, args=(), full_output=0, epsabs=1.49e-8, epsrel=1.49e-8, limit=50, points=None, weight=None, wvar=None, wopts=None, maxp1=50, limlst=50, complex_func=False)`
**Invariants**: INT-007

### Entry Point and Argument Validation

```
quad(func, a, b, ...)
  |
  +-- args = (args,) if not tuple
  +-- a == b: return (0., 0.) immediately
  +-- flip, a, b = b < a, min(a,b), max(a,b)  (normalize order)
  +-- If complex_func:
  |     split into real + imaginary parts, call quad recursively on each
  |     combine results: integral + 1j*integral, error + 1j*error
```

### Decision Tree

```
                          quad(func, a, b)
                               |
                      +--------+--------+
                      |                 |
                weight is None     weight is not None
                      |                 |
               _quad(func,a,b,...)   _quad_weight(func,a,b,...)
                      |                      |
            +---------+---------+    +-------+-------+-------+
            |         |         |    |       |       |       |
        finite    semi-inf    inf  cos/sin  alg*  cauchy  cos/sin+inf
        bounds    bounds     bounds  |       |       |       |
            |         |         |  QAWOE   QAWSE  QAWCE   QAWFE
          points?   QAGIE    QAGIE
            |
      +-----+-----+
      |           |
   No points  Has points
      |           |
    QAGSE       QAGPE
```

Detailed QUADPACK routine selection:

```
_quad(func, a, b, ...):
  |
  +-- Both finite (infbounds=0):
  |     +-- points is None -> _quadpack._qagse (adaptive with extrapolation)
  |     +-- points given   -> _quadpack._qagpe (break points)
  |
  +-- b == inf, a finite (infbounds=1):
  |     -> _quadpack._qagie (transform to [0,1])
  |
  +-- Both infinite (infbounds=2):
  |     -> _quadpack._qagie (transform to [0,1])
  |
  +-- a == -inf, b finite (infbounds=-1):
       -> _quadpack._qagie (transform to [0,1])
```

### Hot Path

Finite bounds, no weight, no breakpoints: QAGSE (globally adaptive subdivision with
21-point Gauss-Kronrod quadrature and epsilon-algorithm extrapolation). This is the most
common integration scenario.

### Fallback Chain

No fallback between routines. QUADPACK routines internally perform adaptive subdivision
up to `limit` subintervals. If the integral does not converge, error codes (ier) are returned.

### Error Propagation

```
ier == 0 -> success, return (result, abserr)
ier in [1..5,7]:
  full_output=0: warnings.warn(IntegrationWarning) + return result
  full_output=1: return (result, abserr, infodict, message)
ier == 6 -> ValueError with diagnostic message:
  - epsabs/epsrel too small
  - limit too small
  - break points outside bounds
  - maxp1 < 1
  - wvar parameters invalid
ier == 80 -> "A Python error occurred" (exception in callback)
```

---

## 10. scipy.optimize.minimize

**Source**: `optimize/_minimize.py:54`
**Signature**: `minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)`

### Entry Point and Argument Validation

```
minimize(fun, x0, ...)
  |
  +-- x0 = np.atleast_1d(np.asarray(x0))
  +-- Validate: x0.ndim == 1
  +-- Upcast integer x0 to float
  +-- args = (args,) if not tuple
  |
  +-- Method selection (if method is None):
  |     constraints present -> 'SLSQP'
  |     bounds present      -> 'L-BFGS-B'
  |     neither             -> 'BFGS'
  |
  +-- Compatibility warnings:
  |     jac with derivative-free methods
  |     hess without Newton/trust methods
  |     constraints with unconstrained methods
  |     bounds with methods that can't handle them
  |
  +-- Jacobian resolution:
  |     callable -> use as-is
  |     True     -> MemoizeJac(fun), extract derivative
  |     FD string ('2-point','3-point','cs') -> finite difference
  |     None/False -> None (method uses internal approx)
  |
  +-- Tolerance propagation:
  |     method-specific defaults (xatol, fatol, xtol, ftol, gtol)
  |
  +-- Bounds standardization + fixed-variable removal
  +-- Constraint standardization
```

### Decision Tree

```
                       minimize(fun, x0)
                             |
        +----+----+----+----+----+----+----+----+
        |    |    |    |    |    |    |    |    |
       NM  Powell CG  BFGS N-CG L-B  TNC COBYLA ...
        |    |    |    |    |    |    |    |    |
   (simplex)(dir)(conj(quasi(trunc(lim(trunc(lin
    search) set) grad)Newton)Newton)mem)Newton)approx)
        |                   |              |
   _minimize_          _minimize_     _minimize_
   neldermead          bfgs           lbfgsb
        |                   |              |
   No derivatives     Uses gradient    Uses gradient
   required           (analytic or FD) + limited memory
                                        Hessian approx

   Full dispatch table:
   'nelder-mead'  -> _minimize_neldermead
   'powell'       -> _minimize_powell
   'cg'           -> _minimize_cg
   'bfgs'         -> _minimize_bfgs      [DEFAULT unconstrained]
   'newton-cg'    -> _minimize_newtoncg
   'l-bfgs-b'     -> _minimize_lbfgsb    [DEFAULT bounded]
   'tnc'          -> _minimize_tnc
   'cobyla'       -> _minimize_cobyla
   'cobyqa'       -> _minimize_cobyqa
   'slsqp'        -> _minimize_slsqp     [DEFAULT constrained]
   'trust-constr' -> _minimize_trustregion_constr
   'dogleg'       -> _minimize_dogleg
   'trust-ncg'    -> _minimize_trust_ncg
   'trust-krylov' -> _minimize_trust_krylov
   'trust-exact'  -> _minimize_trustregion_exact
   callable       -> method(fun, x0, args, ...)
```

### Hot Path

Unconstrained, no bounds: BFGS with approximate gradient (finite differences).
Calls `_minimize_bfgs` which performs quasi-Newton line search iterations.

Bounded: L-BFGS-B with limited-memory BFGS Hessian approximation.

Constrained: SLSQP (Sequential Least Squares Programming).

### Fallback Chain

No automatic fallback between methods. Each method runs to convergence or exhaustion
of maxiter.

### Error Propagation

```
OptimizeResult:
  success=True  -> converged to tolerance
  success=False -> maxiter reached, or numerical failure
  status codes are method-specific
  message describes termination reason
  Callback can raise StopIteration -> success=False, status=99
```

---

## 11. scipy.optimize.root

**Source**: `optimize/_root.py:25`
**Signature**: `root(fun, x0, args=(), method='hybr', jac=None, tol=None, callback=None, options=None)`

### Entry Point and Argument Validation

```
root(fun, x0, ...)
  |
  +-- Wrap fun in _wrapped_fun to track nfev
  +-- args = (args,) if not tuple
  +-- meth = method.lower()
  +-- Warn if callback given for 'hybr'/'lm' (not supported)
  +-- For 'hybr'/'lm': if jac is True, wrap fun with MemoizeJac
  +-- Set default tolerances based on method
```

### Decision Tree

```
                        root(fun, x0)
                             |
               +-------------+-------------+
               |             |             |
          meth='hybr'   meth='lm'    meth in quasi-Newton
          (default)     (least-sq)   methods
               |             |             |
      _root_hybr       _root_leastsq  _root_nonlin_solve
      (MINPACK hybrd)  (MINPACK lmdif) (broyden1, broyden2,
                                        anderson, krylov,
                                        linearmixing,
                                        diagbroyden,
                                        excitingmixing)
               |
          meth='df-sane'
               |
      _root_df_sane
      (derivative-free
       spectral method)
```

### Hot Path

Default `method='hybr'`: Modified Powell hybrid method from MINPACK. Requires the Jacobian
(either user-provided or estimated numerically). Uses trust-region dogleg steps with
QR-factored Jacobian updates.

### Fallback Chain

No automatic fallback. Each method runs independently.

### Error Propagation

```
OptimizeResult:
  success=True  -> ||F(x)|| < tol
  success=False -> convergence failure
  nfev tracked via _wrapped_fun counter
```

---

## 12. scipy.optimize.curve_fit

**Source**: `optimize/_minpack_py.py:590`
**Signature**: `curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=None, bounds=(-np.inf, np.inf), method=None, jac=None, *, full_output=False, nan_policy=None, **kwargs)`

### Entry Point and Argument Validation

```
curve_fit(f, xdata, ydata, ...)
  |
  +-- If p0 is None: introspect f to determine n_params
  |   else: p0 = np.atleast_1d(p0), n = p0.size
  +-- Parse bounds: Bounds instance or (lb, ub) tuple
  +-- If p0 is None: _initialize_feasible(lb, ub) -> p0
  +-- Method selection:
  |     bounded problem -> 'trf' (trust region reflective)
  |     unbounded       -> 'lm'  (Levenberg-Marquardt)
  +-- 'lm' + bounded -> ValueError
  +-- check_finite / nan_policy handling
  +-- Cast xdata/ydata to float64
  +-- ydata.size == 0 -> ValueError
  +-- NaN handling: omit NaN rows if nan_policy='omit'
  +-- Sigma processing:
  |     1D sigma -> transform = 1/sigma (error weighting)
  |     2D sigma -> transform = cholesky(sigma, lower=True) (covariance weighting)
  +-- Wrap f into residual function: _wrap_func(f, xdata, ydata, transform)
```

### Decision Tree

```
                       curve_fit(f, xdata, ydata)
                              |
                     +--------+--------+
                     |                 |
               method='lm'      method='trf'/'dogbox'
               (unbounded)       (bounded)
                     |                 |
              leastsq(func, p0,   least_squares(func, p0,
                Dfun=jac, ...)      jac=jac, bounds=bounds, ...)
                     |                 |
              MINPACK LMDIF/LMDER  Trust region reflective /
                     |              Dog-box algorithm
                     |                 |
                     +--------+--------+
                              |
                     Compute covariance:
                     +-- method='lm': pcov from infodict
                     +-- method='trf'/'dogbox':
                     |     SVD of Jacobian at solution
                     |     pcov = VT.T / s^2 @ VT
                     |     (Moore-Penrose pseudo-inverse)
                     |
                     +-- If not absolute_sigma and ysize > n:
                     |     pcov *= cost / (ysize - n)
                     |
                     +-- If pcov is None or has NaN:
                           pcov = inf, warn OptimizeWarning
```

### Hot Path

Unbounded problem with analytical Jacobian: `leastsq` calling MINPACK `lmder` (Levenberg-Marquardt
with user-provided Jacobian). Without Jacobian: `lmder` uses finite differences internally.

### Fallback Chain

No fallback between methods. `RuntimeError` is raised if optimization fails.

### Error Propagation

```
method='lm': ier in [1,2,3,4] -> success; else RuntimeError
method='trf'/'dogbox': res.success=True -> success; else RuntimeError
Covariance estimation failure -> OptimizeWarning (pcov filled with inf)
```

---

## 13. scipy.optimize.linprog

**Source**: `optimize/_linprog.py:178`
**Signature**: `linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=(0, None), method='highs', callback=None, options=None, x0=None, integrality=None)`

### Entry Point and Argument Validation

```
linprog(c, ...)
  |
  +-- meth = method.lower()
  +-- Validate method in {"highs","highs-ds","highs-ipm",
  |     "simplex","revised simplex","interior-point"}
  +-- x0 warning if method != 'revised simplex'
  +-- integrality: only supported by 'highs', broadcast to c.shape
  +-- _LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0, integrality)
  +-- _parse_linprog(lp, options, meth) -> validated lp, solver_options
```

### Decision Tree

```
                       linprog(c, A_ub, b_ub, ...)
                              |
                     +--------+--------+
                     |                 |
               meth.startswith     legacy methods
               ('highs')           (deprecated)
                     |                 |
            +--------+--------+  presolve -> standard form -> solve -> postprocess
            |        |        |
          'highs'  'highs-ds' 'highs-ipm'
          (auto)   (simplex)  (interior-point)
            |        |        |
       solver=None  'simplex' 'ipm'
            |        |        |
         _linprog_highs(lp, solver=..., **solver_options)
            |
         HiGHS C++ library (dual revised simplex or IPM with crossover)
            |
         _check_result -> validate solution
            |
         OptimizeResult(sol)
```

### Hot Path

Default `method='highs'`: HiGHS library auto-selects between dual revised simplex and
interior-point method based on problem characteristics. For large sparse problems, IPM
is typically faster. For small/medium problems, simplex is preferred.

### Fallback Chain

The HiGHS `method='highs'` auto-selector provides built-in fallback: if one solver encounters
difficulty, HiGHS may internally switch strategies. The legacy methods have no fallback.

### Error Propagation

```
OptimizeResult:
  status=0: "Optimization terminated successfully"
  status=1: "Iteration limit reached"
  status=2: "Problem appears to be infeasible"
  status=3: "Problem appears to be unbounded"
  status=4: "Numerical difficulties encountered"
  success = (status == 0)
```

HiGHS callback is not supported; raises `NotImplementedError` if callback is provided.

---

## 14. scipy.fft.fft

**Source**: `fft/_basic.py:27` (public API), `fft/_basic_backend.py:77` (backend implementation)
**Signature**: `fft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None)`

### Entry Point and Argument Validation

The public `fft` function in `_basic.py` is a uarray multimethod decorated with `@_dispatch`
and `@xp_capabilities(allow_dask_compute=True)`. This enables pluggable backends via uarray
protocol. The actual computation is dispatched to the backend.

```
fft(x, n, axis, norm, ...)
  |
  +-- uarray dispatch mechanism:
  |     Check registered backends (e.g., scipy.fft._pocketfft, CuPy, etc.)
  |     Select backend based on input array type
  |
  +-- Default backend: _basic_backend.fft
       |
       +-- _execute_1D('fft', _pocketfft.fft, x, ...)
            |
            +-- xp = array_namespace(x)
            |
            +-- If is_numpy(xp):
            |     x = np.asarray(x)
            |     return _pocketfft.fft(x, n=n, axis=axis, norm=norm,
            |                            overwrite_x=overwrite_x, workers=workers)
            |
            +-- Else if xp has .fft namespace:
            |     use xp.fft.fft (e.g., CuPy's FFT)
            |     For complex funcs: try as-is, fallback to float-to-complex cast
            |
            +-- Else: convert to numpy, use _pocketfft, convert back
```

### Hot Path

NumPy input: direct call to `_pocketfft.fft` which is a compiled C++ implementation (PocketFFT).
PocketFFT uses mixed-radix Cooley-Tukey algorithm for composite sizes and Bluestein's algorithm
for prime sizes. Real input is automatically optimized via real-FFT internally.

### Fallback Chain

```
1. Try native backend (xp.fft.fft) for non-NumPy arrays
2. If native backend fails for complex funcs: cast float->complex, retry
3. If no xp.fft: convert to NumPy, use PocketFFT, convert back
```

### Error Propagation

```
IndexError: axis larger than last axis of x
ValueError: n < 1 or invalid norm
Backend-specific errors propagated directly
```

---

## 15. scipy.fft.ifft

**Source**: `fft/_basic.py` (after fft definition), `fft/_basic_backend.py:83`
**Signature**: `ifft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None)`

The control flow is identical to `fft` (section 14), with the only difference being that
`_pocketfft.ifft` is called instead of `_pocketfft.fft`, and the default normalization
(`norm="backward"`) applies a `1/n` scaling factor.

```
ifft(x, ...)
  |
  +-- Same dispatch path as fft
  +-- _execute_1D('ifft', _pocketfft.ifft, x, ...)
       |
       +-- Same array_namespace / backend selection as fft
       +-- _pocketfft.ifft applies 1/n normalization (backward)
           or 1/sqrt(n) (ortho) or none (forward)
```

---

## 16. scipy.sparse.linalg.spsolve

**Source**: `sparse/linalg/_dsolve/linsolve.py:134`
**Signature**: `spsolve(A, b, permc_spec=None, use_umfpack=True)`
**Invariants**: SPA-001 through SPA-005, SPL-002

### Entry Point and Argument Validation

```
spsolve(A, b, ...)
  |
  +-- Convert pydata sparse to scipy if needed
  +-- If A not CSC/CSR: convert to CSC + SparseEfficiencyWarning
  +-- A.sum_duplicates()                          [SPA-005]
  +-- A._asfptype() -> upcast to floating point
  +-- result_dtype = np.promote_types(A.dtype, b.dtype)
  +-- Match A and b dtypes
  +-- Validate: M == N (square matrix)            [SPL-002]
  +-- Validate: M == b.shape[0] (compatible dims)
  +-- Determine b_is_vector, b_is_sparse
```

### Decision Tree

```
                       spsolve(A, b)
                             |
                    +--------+--------+
                    |                 |
            b is vector         b is matrix
                    |                 |
           +--------+--------+       |
           |                 |   factorized(A) -> Afactsolve
      use_umfpack=True  use_umfpack=False    loop over columns:
      & UMFPACK avail   or UMFPACK not avail   xj = Afactsolve(bj)
           |                 |                 build sparse result
      +----+----+       SuperLU path
      |         |            |
   A.dtype    else      _superlu.gssv(N, nnz, data,
   in dD?    ValueError   indices, indptr, b, flag)
      |                      |
   UmfpackContext       +----+----+
   umf.linsolve         |    |    |
                     info=0  0<i<=N  i>N
                    success  singular  OOM
                              |
                         warn(MatrixRankWarning)
                         x.fill(NaN)
```

### Hot Path

Vector b with UMFPACK available: UMFPACK direct solver (LU factorization with column
reordering). UMFPACK uses unsymmetric multifrontal method, generally faster than SuperLU
for large problems.

Without UMFPACK: SuperLU via `_superlu.gssv`. The `permc_spec='COLAMD'` (default) performs
approximate minimum degree column ordering for sparsity preservation.

### Fallback Chain

```
1. Try UMFPACK (if use_umfpack=True and scikits.umfpack installed)
2. Fall back to SuperLU (always available)
```

For sparse b (matrix RHS): factorize A once, solve column-by-column.

### Error Propagation

```
SuperLU gssv:
  info == 0       -> success
  0 < info <= N   -> "Matrix is exactly singular" -> MatrixRankWarning, x=NaN
  info > N        -> MemoryError
  info < 0        -> Exception("unknown exit code")

UMFPACK:
  Errors from UmfpackContext propagated directly
  Only float64/complex128 with int32/int64 indices supported
```

---

## 17. scipy.sparse.linalg.eigsh

**Source**: `sparse/linalg/_eigen/arpack/arpack.py:1426`
**Signature**: `eigsh(A, k=6, M=None, sigma=None, which='LM', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=True, Minv=None, OPinv=None, mode='normal', rng=None)`
**Invariants**: SPL-004

### Entry Point and Argument Validation

```
eigsh(A, k=6, ...)
  |
  +-- Complex Hermitian check:
  |     If A.dtype is complex:
  |       mode must be 'normal', which!='BE'
  |       Remap which: 'LA'->'LR', 'SA'->'SR'
  |       Delegate to eigs(), return real parts
  |
  +-- Validate: A is square                  [SPL-004 partial]
  +-- M shape/dtype compatibility checks
  +-- k > 0                                  [SPL-004 partial]
  +-- k >= n: warn, try scipy.linalg.eigh (dense fallback)
  +-- Validate rng / create default
```

### Decision Tree

```
                        eigsh(A, k)
                             |
                +------------+------------+
                |                         |
          A is complex              A is real symmetric
                |                         |
         eigs(A, k, ...)           +------+------+
         return real parts         |             |
                              sigma is None  sigma is not None
                                   |             |
                            +------+------+  shift-invert mode
                            |             |      |
                        M is None    M is not None
                        (standard)   (generalized)
                            |             |
                        mode = 1      mode = 2
                        matvec=A@x    matvec=A@x
                        M_matvec=None M_matvec=M@x
                                      Minv_matvec=M^{-1}@x
                                      (via sparse LU or user Minv)
                                           |
                              +------------+------------+
                              |            |            |
                         mode='normal' mode='buckling' mode='cayley'
                         (mode=3)      (mode=4)        (mode=5)
                              |            |            |
                         OP=[A-sM]^{-1}M  OP=[A-sM]^{-1}A  OP=[A-sM]^{-1}[A+sM]
                              |
                   _SymmetricArpackParams(n, k, dtype, matvec, mode,
                     M_matvec, Minv_matvec, sigma, ncv, v0, maxiter, which, tol)
                              |
                   while not params.converged:
                     params.iterate()
                     (Implicitly Restarted Lanczos Method)
                              |
                   params.extract(return_eigenvectors)
                   -> (eigenvalues,) or (eigenvalues, eigenvectors)
```

### Hot Path

Standard eigenvalue problem (`sigma=None`, `M=None`): ARPACK symmetric mode 1.
Uses Implicitly Restarted Lanczos iteration with `matvec = A @ x`. Default `which='LM'`
finds the `k` largest-magnitude eigenvalues.

For finding small eigenvalues efficiently: use `sigma=0` (shift-invert mode) which transforms
the problem to find eigenvalues of `A^{-1}` (largest magnitude = smallest of original A).

### Fallback Chain

```
1. k >= n: fall back to scipy.linalg.eigh (dense, exact)
   Only if A is not sparse/LinearOperator
2. Complex A: delegate to eigs() (general non-symmetric ARPACK)
```

### Error Propagation

```
ArpackNoConvergence:
  Raised when requested convergence not obtained within maxiter.
  Exception carries partial results: .eigenvalues, .eigenvectors
  (converged eigenvalues/vectors so far)

ValueError: mode/which incompatibilities
TypeError: sparse A with k >= n (cannot use dense eigh)
```

---

## 18. scipy.interpolate.interp1d

**Source**: `interpolate/_interpolate.py:177`
**Signature**: `interp1d(x, y, kind='linear', axis=-1, copy=True, bounds_error=None, fill_value=np.nan, assume_sorted=False)`

### Entry Point and Argument Validation (Constructor)

```
interp1d.__init__(x, y, kind='linear', ...)
  |
  +-- _Interpolator1D.__init__(x, y, axis=axis) -> base class setup
  +-- bounds_error, copy setup
  +-- kind resolution:
  |     'zero'/'slinear'/'quadratic'/'cubic' -> spline with order 0/1/2/3
  |     int -> spline with that order
  |     'linear'/'nearest'/'nearest-up'/'previous'/'next' -> direct methods
  |     other -> NotImplementedError
  +-- copy x, y arrays
  +-- If not assume_sorted: argsort x, reorder y
  +-- Validate: x is 1D, y has at least 1D
  +-- Force y to floating point
  +-- self._y = self._reshape_yi(y) -> move interp axis to front
```

### Decision Tree (Call-time)

```
                     interp1d(x_new)
                          |
                 +--------+--------+--------+--------+--------+
                 |        |        |        |        |        |
              linear   nearest  nearest-up previous  next   spline
                 |        |        |        |        |        |
           _call_linear _call_nearest _call_  _call_  make_interp_
                 |        |      previousnext previousnext  spline(x,y,k)
                 |    searchsorted  searchsorted       |
                 |    on x_bds      on x/_x_shift  spleval at x_new
                 |        |
           searchsorted + linear interpolation:
           slope = (y[hi] - y[lo]) / (x[hi] - x[lo])
           y_new = slope * (x_new - x[lo]) + y[lo]
```

### Hot Path

`kind='linear'` (default): binary search via `np.searchsorted` to find enclosing interval,
then linear interpolation between adjacent points. No spline fitting overhead.

### Fallback Chain

No fallback. Out-of-bounds handling:
- `bounds_error=True` (default unless fill_value='extrapolate'): raise `ValueError`
- `bounds_error=False`: fill with `fill_value` (default NaN)
- `fill_value='extrapolate'`: extrapolate using edge intervals

### Error Propagation

```
Constructor:
  NotImplementedError: unsupported kind
  ValueError: x not 1D, y is 0D, incompatible shapes

Call-time:
  ValueError: x_new outside range (if bounds_error=True)
  NaN fill: if bounds_error=False and x_new outside range
```

---

## 19. scipy.interpolate.CubicSpline

**Source**: `interpolate/_cubic.py:628`
**Signature**: `CubicSpline(x, y, axis=0, bc_type='not-a-knot', extrapolate=None)`
**Inherits**: `CubicHermiteSpline` -> `PPoly`

### Entry Point and Argument Validation (Constructor)

```
CubicSpline.__init__(x, y, axis=0, bc_type='not-a-knot', ...)
  |
  +-- array_namespace(x, y)
  +-- prepare_input(x, y, axis) -> x, dx, y, axis
  |     validates: x strictly increasing, finite
  |     validates: y finite, compatible shape
  +-- _validate_bc(bc_type, y, shape, axis) -> bc, y
  |     bc_type strings: 'not-a-knot', 'periodic', 'clamped', 'natural'
  |     bc_type tuple: ((order1, value1), (order2, value2))
  |     'periodic': verify y[0] == y[-1]
  +-- extrapolate default: 'periodic' if bc_type='periodic', else True
  +-- y.size == 0: s = zeros (bail out early)
```

### Decision Tree (Construction)

```
                    CubicSpline(x, y)
                          |
                 +--------+--------+--------+
                 |        |        |        |
              n == 2   n == 3     n == 3   n >= 4
              (any bc) (not-a-knot (periodic) (general)
                 |      both ends)    |        |
           replace   parabola    manual    tridiagonal
           not-a-knot through     deriv    system:
           with slope 3 points  computation A[3,n] banded
                 |        |        |        |
           CubicHermiteSpline.__init__(x, y, s, ...)
```

Detailed general case (n >= 4):

```
General tridiagonal system:
  |
  +-- bc_start/bc_end processing:
  |     'not-a-knot': modify first/last row of tridiagonal system
  |     (1, value):   first derivative specified -> row [1, 0, ...] = value
  |     (2, value):   second derivative specified -> modified row
  |     'periodic':   reduced (n-1) system with wrap-around coupling
  |
  +-- For periodic bc:
  |     Condensed (n-2, n-2) system
  |     solve_banded((1,1), Ac, b1) -> s1
  |     solve_banded((1,1), Ac, b2) -> s2
  |     Recover s[n-2] from boundary coupling
  |     s[-1] = s[0] (periodicity)
  |
  +-- For non-periodic bc:
       solve_banded((1,1), A, b) -> s (derivatives at each knot)
       |
       CubicHermiteSpline.__init__(x, y, s)
       -> computes polynomial coefficients c[4, n-1, ...]
       -> stored as PPoly for evaluation
```

### Hot Path

`bc_type='not-a-knot'` (default), n >= 4: constructs and solves a tridiagonal system using
`scipy.linalg.solve_banded` with bandwidth (1,1). This is O(n) for both construction and
per-point evaluation. The result is a `PPoly` instance with C2 smoothness.

### Fallback Chain

No fallback. Construction always succeeds for valid input.

### Error Propagation

```
Constructor:
  ValueError: x not strictly increasing
  ValueError: x and y shape mismatch
  ValueError: invalid bc_type
  ValueError: periodic bc with y[0] != y[-1]
  ValueError: subset_by_index/value out of range

Evaluation (__call__):
  Uses PPoly.__call__ -> binary search + polynomial evaluation
  extrapolate=False: NaN for out-of-bounds
  extrapolate=True: extrapolate using edge polynomials
  extrapolate='periodic': wrap x values modulo period
```

---

## 20. scipy.stats.norm (distribution interface)

**Source**: `stats/_continuous_distns.py:394` (norm_gen), `stats/_distn_infrastructure.py:2052` (rv_continuous.pdf), `stats/_distn_infrastructure.py:2133` (rv_continuous.cdf)

`norm` is an instance of `norm_gen`, which inherits from `rv_continuous`. The `pdf()` and `cdf()`
methods are defined on `rv_continuous` and delegate to the distribution-specific `_pdf()` and
`_cdf()` methods.

### Entry Point and Argument Validation (pdf)

```
norm.pdf(x, loc=0, scale=1)
  |
  +-- rv_continuous.pdf(self, x, *args, **kwds)
       |
       +-- _parse_args(*args, **kwds) -> args, loc, scale
       +-- x, loc, scale = map(asarray, ...)
       +-- args = tuple(map(asarray, args))
       +-- dtyp = np.promote_types(x.dtype, np.float64)
       +-- x = (x - loc) / scale   (standardize)
       +-- cond0 = _argcheck(*args) & (scale > 0)
       +-- cond1 = _support_mask(x, *args) & (scale > 0)
       +-- cond = cond0 & cond1
       +-- output = zeros(shape(cond), dtyp)
       +-- putmask(output, bad_args | NaN, badvalue)
       +-- If any(cond):
       |     goodargs = argsreduce(cond, x, *args, scale)
       |     place(output, cond, self._pdf(*goodargs) / scale)
       +-- Return scalar if 0D, else array
```

### Entry Point and Argument Validation (cdf)

```
norm.cdf(x, loc=0, scale=1)
  |
  +-- rv_continuous.cdf(self, x, *args, **kwds)
       |
       +-- _parse_args -> args, loc, scale
       +-- x = (x - loc) / scale
       +-- _a, _b = _get_support(*args)
       +-- cond0 = _argcheck(*args) & (scale > 0)
       +-- cond1 = _open_support_mask(x, *args) & (scale > 0)
       +-- cond2 = (x >= _b) & cond0   (above upper support -> 1.0)
       +-- cond = cond0 & cond1
       +-- output = zeros(...)
       +-- putmask(output, bad_args | NaN, badvalue)
       +-- place(output, cond2, 1.0)
       +-- If any(cond):
       |     goodargs = argsreduce(cond, x, *args)
       |     place(output, cond, self._cdf(*goodargs))
       +-- Return scalar if 0D, else array
```

### Decision Tree

```
                    norm.pdf(x) / norm.cdf(x)
                              |
                    rv_continuous.pdf / .cdf
                              |
              +---------------+---------------+
              |               |               |
         bad args        out of support    in support
         (scale<=0)      (x outside [a,b]) (valid)
              |               |               |
         output = NaN    pdf: output=0     _pdf(x_std) / scale
                         cdf: output=0     _cdf(x_std)
                         (or 1 if x>=b)
                              |
                    norm_gen._pdf(x):
                      return _norm_pdf(x)
                      = exp(-x^2/2) / sqrt(2*pi)
                              |
                    norm_gen._cdf(x):
                      return _norm_cdf(x)
                      = special.ndtr(x)
                      (uses erfc for numerical stability)
```

### Hot Path

`norm.pdf(x)`: standardize `x_std = (x - loc) / scale`, compute `exp(-x_std^2/2) / sqrt(2*pi)`,
divide by `scale`. The `_norm_pdf` function is a simple NumPy vectorized computation.

`norm.cdf(x)`: standardize, call `scipy.special.ndtr(x_std)` which uses the complementary
error function (`erfc`) for numerically stable Gaussian CDF evaluation across the full range.

### Fallback Chain

The `rv_continuous` base class has generic fallback implementations:
- `_pdf` defaults to numerical derivative of `_cdf` (if only `_cdf` is defined)
- `_cdf` defaults to numerical integration of `_pdf` (via `_cdf_single` + `quad`)
- `norm_gen` overrides both with closed-form implementations, so no fallback needed

### Error Propagation

```
Invalid scale (scale <= 0): output filled with self.badvalue (NaN)
Invalid shape args: output filled with self.badvalue
NaN in x: output is NaN at those positions
Out of support: pdf=0, cdf=0 (below) or 1 (above)
No exceptions raised for normal usage
```

---

## 21. scipy.signal.fftconvolve

**Source**: `signal/_signaltools.py:589`
**Signature**: `fftconvolve(in1, in2, mode="full", axes=None)`

### Entry Point and Argument Validation

```
fftconvolve(in1, in2, mode="full", axes=None)
  |
  +-- xp = array_namespace(in1, in2)
  +-- in1 = xp.asarray(in1)
  +-- in2 = xp.asarray(in2)
  +-- Scalar inputs (both 0D): return in1 * in2
  +-- Dimension mismatch: ValueError
  +-- Empty arrays: return xp.asarray([])
  +-- _init_freq_conv_axes(in1, in2, mode, axes) -> in1, in2, axes
  |     validates mode, computes axes, checks 'valid' mode constraints
  +-- Compute output shape: s1[i] + s2[i] - 1 for axes in convolution
```

### Decision Tree

```
                   fftconvolve(in1, in2)
                          |
               +----------+----------+
               |          |          |
          both scalar  dim mismatch  normal case
               |          |          |
          return      ValueError    _freq_domain_conv(xp, in1, in2, axes, shape,
          in1*in2                                      calc_fast_len=True)
                                          |
                                    +-----+-----+
                                    |           |
                              complex inputs  real inputs
                                    |           |
                              fftn/ifftn    rfftn/irfftn
                                    |           |
                              For each signal:
                              1. Pad to next_fast_len(shape)
                              2. FFT: sp1 = fft(in1, fshape, axes)
                              3. FFT: sp2 = fft(in2, fshape, axes)
                              4. Multiply: sp1 * sp2
                              5. IFFT: ret = ifft(sp1*sp2, fshape, axes)
                              6. Trim padding
                                    |
                              _apply_conv_mode(ret, s1, s2, mode, axes)
                                    |
                              +-----+-----+-----+
                              |           |     |
                           'full'     'same'  'valid'
                              |           |     |
                           full output  center  inner
                                        crop    crop
```

### Hot Path

Real inputs, `mode='full'`: uses `rfftn`/`irfftn` (real FFT, approximately 2x faster than complex
FFT). Pads to `next_fast_len` for optimal FFT performance (highly composite sizes). The
convolution is computed as element-wise multiplication in the frequency domain:
`IFFT(FFT(in1) * FFT(in2))`.

Integer inputs are cast to float64 before FFT.

### Fallback Chain

No fallback. For very large arrays with disparate sizes, `scipy.signal.oaconvolve` (overlap-add)
may be more efficient, but `fftconvolve` does not automatically switch to it.

### Error Propagation

```
ValueError: in1 and in2 have different dimensionality
ValueError: invalid mode
ValueError: 'valid' mode with in1 smaller than in2 in some dimension
            (neither array is at least as large as the other in every dim)
```

---

## Cross-Cutting Patterns

### Pattern 1: LAPACK Dispatch (linalg module)

All `scipy.linalg` functions follow a common pattern (see DOC-PASS-01, linalg module tree):

```
User call
  -> _asarray_validated (check_finite)          [LIN-004]
  -> _normalize_lapack_dtype (upcast to LAPACK type)
  -> _ensure_aligned_and_native (memory layout)
  -> dimension/shape checks                      [LIN-001..003]
  -> _batched_linalg._{operation} (C layer)
  -> error list processing via _format_emit_errors_warnings
  -> return result
```

The `_batched_linalg` C extension handles per-slice iteration for batched inputs,
LAPACK function selection based on dtype (`s/d/c/z` prefix), and structure auto-detection
when `assume_a=None`.

### Pattern 2: Optimize Dispatch (optimize module)

All `scipy.optimize` top-level functions (minimize, root, linprog, curve_fit) follow:

```
User call
  -> Input validation and normalization
  -> Method string -> method-specific function dispatch
  -> Method-specific function runs iterative algorithm
  -> Return OptimizeResult with standardized fields
```

### Pattern 3: Backend Dispatch (fft module)

The `scipy.fft` module uses uarray for pluggable backend selection:

```
User call -> @_dispatch decorator -> uarray multimethod
  -> Check registered backends in order
  -> Default: _basic_backend -> array_namespace detection
     -> NumPy: PocketFFT (C++)
     -> CuPy/PyTorch: xp.fft namespace
     -> Other: convert to NumPy, PocketFFT, convert back
```

### Pattern 4: Sparse Solver Selection (sparse.linalg)

The sparse module uses a two-tier solver strategy:

```
User call
  -> Convert to CSC/CSR, validate
  -> Try UMFPACK (if available and requested)
  -> Fall back to SuperLU (always available)
```

### Pattern 5: Distribution Interface (stats module)

All continuous distributions follow the `rv_continuous` template method pattern:

```
user.pdf(x, *args, loc, scale)
  -> rv_continuous.pdf: standardize, mask, dispatch to _pdf
    -> distribution._pdf(x_standardized): closed-form or generic
  -> rv_continuous.pdf: un-mask, divide by scale, return
```

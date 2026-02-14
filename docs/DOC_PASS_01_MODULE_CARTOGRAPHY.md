# DOC-PASS-01: SciPy Module Cartography

> Comprehensive mapping of SciPy legacy source at
> `/data/projects/frankenscipy/legacy_scipy_code/scipy/scipy/`
>
> Generated: 2026-02-14 | Source: SciPy 1.17.0

---

## Table of Contents

1. [scipy.linalg](#scipylinalg)
2. [scipy.sparse](#scipysparse)
3. [scipy.integrate](#scipyintegrate)
4. [scipy.optimize](#scipyoptimize)
5. [scipy.fft](#scipyfft)
6. [scipy.special](#scipyspecial)
7. [scipy.stats](#scipystats)
8. [scipy.signal](#scipysignal)
9. [scipy.spatial](#scipyspatial)
10. [scipy.interpolate](#scipyinterpolate)
11. [scipy.ndimage](#scipyndimage)
12. [scipy.io](#scipyio)
13. [scipy.cluster](#scipycluster)
14. [scipy.constants](#scipyconstants)
15. [scipy.misc](#scipymisc)
16. [scipy.odr](#scipyodr)
17. [scipy.datasets](#scipydatasets)
18. [scipy.differentiate](#scipydifferentiate)
19. [Cross-Module Dependency Matrix](#cross-module-dependency-matrix)
20. [Summary Statistics](#summary-statistics)

---

## scipy.linalg

### Module Tree

```
linalg/
  __init__.py               Public API exports (basics, decompositions, matrix functions, special matrices)
  __init__.pxd              Cython declaration file
  _basic.py                 Core solve/inv/det/lstsq/pinv/norm operations
  _decomp.py                eig, eigvals, eigh, eigvalsh
  _decomp_cholesky.py       cholesky, cho_factor, cho_solve, cholesky_banded
  _decomp_cossin.py         CS decomposition (cossin)
  _decomp_ldl.py            LDL decomposition
  _decomp_lu.py             LU decomposition
  _decomp_polar.py          Polar decomposition
  _decomp_qr.py             QR decomposition
  _decomp_qz.py             QZ decomposition (generalized Schur)
  _decomp_schur.py          Schur decomposition
  _decomp_svd.py            SVD decomposition
  _expm_frechet.py          Matrix exponential Frechet derivative
  _matfuncs.py              expm, logm, signm, cosm, sinm, etc.
  _matfuncs_expm.pyi        Type stubs for expm C extension
  _matfuncs_inv_ssq.py      logm/fractional power via inverse scaling-and-squaring
  _matfuncs_sqrtm.py        sqrtm (matrix square root)
  _matfuncs_sqrtm_triu.py   Triangular sqrtm helper (Python)
  _matfuncs_sqrtm_triu.pyx  Triangular sqrtm helper (Cython)
  _misc.py                  LinAlgWarning, norm
  _procrustes.py            Orthogonal Procrustes problem
  _sketches.py              Clarkson-Woodruff sketch transform
  _solvers.py               Lyapunov, Sylvester, Riccati equation solvers
  _special_matrices.py      toeplitz, circulant, hankel, hadamard, leslie, companion, hilbert, pascal, etc.
  _testutils.py             Test utilities (_FakeMatrix)
  _linalg_pythran.py        Pythran-accelerated helpers
  _generate_pyx.py          Code generator for Cython BLAS/LAPACK bindings
  _cython_signature_generator.py  Cython signature generator
  _cythonized_array_utils.pxd/pyi/pyx  Array utility Cython extensions
  _decomp_interpolative.pyx Interpolative decomposition (Cython)
  _decomp_lu_cython.pyi/pyx LU decomposition (Cython)
  _decomp_update.pyx.in     QR update operations (Cython template)
  _solve_toeplitz.pyx       Levinson-Durbin Toeplitz solver (Cython)
  blas.py                   Legacy BLAS wrapper (deprecated shim)
  lapack.py                 Legacy LAPACK wrapper (deprecated shim)
  interpolative.py          Legacy interpolative decomposition shim
  basic.py, decomp.py, decomp_cholesky.py, decomp_lu.py, decomp_qr.py,
  decomp_schur.py, decomp_svd.py, matfuncs.py, misc.py, special_matrices.py
                            All deprecated shims to _ prefixed versions
  src/
    _batched_linalg_module.cc   C++ batched linalg operations
    _common_array_utils.h/hh    Common array utilities
    _linalg_eig.hh              Eigenvalue C++ implementation
    _linalg_inv.hh              Inverse C++ implementation
    _linalg_lstsq.hh            Least squares C++ implementation
    _linalg_solve.hh            Solve C++ implementation
    _linalg_svd.hh              SVD C++ implementation
    _matfuncs_expm.c             Matrix exponential C implementation
    _matfuncs_sqrtm.c            Matrix sqrt C implementation
    _npymath.hh                  NumPy math wrappers
  _matfuncsmodule.c          C module for matrix functions
  cython_blas_signatures.txt BLAS signature definitions
  cython_lapack_signatures.txt LAPACK signature definitions
  cblas.pyf.src, cblas_l1.pyf.src, clapack.pyf.src  C BLAS/LAPACK f2py templates
  fblas.pyf.src, fblas_64.pyf.src, fblas_l1/l2/l3.pyf.src  Fortran BLAS f2py templates
  flapack.pyf.src, flapack_64.pyf.src, flapack_gen*.pyf.src,
  flapack_other.pyf.src, flapack_pos_def*.pyf.src,
  flapack_sym_herm.pyf.src, flapack_user.pyf.src  Fortran LAPACK f2py templates
  tests/                    24 test files
    test_basic.py, test_batch.py, test_blas.py, test_decomp.py,
    test_decomp_cholesky.py, test_decomp_cossin.py, test_decomp_ldl.py,
    test_decomp_lu.py, test_decomp_polar.py, test_decomp_update.py,
    test_extending.py, test_fblas.py, test_interpolative.py, test_lapack.py,
    test_matfuncs.py, test_matmul_toeplitz.py, test_procrustes.py,
    test_sketches.py, test_solve_toeplitz.py, test_solvers.py,
    test_special_matrices.py, test_cython_blas.py, test_cython_lapack.py,
    test_cythonized_array_utils.py
    data/                   6 .npz test data files
    _cython_examples/       extending.pyx example
```

### File Counts

- Python: 64, Cython (.pyx): 6, Cython (.pxd): 2
- C: 3, C++ (.cc): 1, Headers (.h): 1, C++ Headers (.hh): 7
- Fortran: 0 (BLAS/LAPACK via .pyf.src templates)
- Test files: 24

### Cross-Module Dependencies

- Imports from: `scipy._lib`, `scipy.sparse`, `scipy.special`
- Weak/optional imports: `scipy.fft`, `scipy.optimize`, `scipy.stats`

### Key Types/Classes

- `LinAlgWarning` (_misc.py:9) - Warning for ill-conditioned matrices
- `SqrtmError` (_matfuncs_sqrtm.py:14) - Matrix sqrt failure
- `LogmRankWarning` (_matfuncs_inv_ssq.py:18) - Logarithm rank warning
- `LogmExactlySingularWarning` (_matfuncs_inv_ssq.py:22) - Singular matrix warning
- `LogmNearlySingularWarning` (_matfuncs_inv_ssq.py:26) - Near-singular warning
- `LogmError` (_matfuncs_inv_ssq.py:30) - Matrix logarithm failure
- `FractionalMatrixPowerError` (_matfuncs_inv_ssq.py:34) - Fractional power failure
- `_MatrixM1PowerOperator` (_matfuncs_inv_ssq.py:39) - Internal power operator

---

## scipy.sparse

### Module Tree

```
sparse/
  __init__.py               Public API: sparse array/matrix classes, construction functions
  _base.py                  _spbase base class, SparseWarning, SparseFormatWarning, SparseEfficiencyWarning
  _bsr.py                   Block Sparse Row format (bsr_array, bsr_matrix)
  _compressed.py            _cs_matrix base for CSR/CSC
  _construct.py             Construction functions: eye, kron, diags, block_diag, random, etc.
  _coo.py                   COOrdinate format (coo_array, coo_matrix)
  _csc.py                   Compressed Sparse Column (csc_array, csc_matrix)
  _csr.py                   Compressed Sparse Row (csr_array, csr_matrix)
  _data.py                  _data_matrix, _minmax_mixin
  _dia.py                   Diagonal format (dia_array, dia_matrix)
  _dok.py                   Dictionary of Keys format (dok_array, dok_matrix)
  _extract.py               tril, triu extraction
  _index.py                 IndexMixin for fancy indexing
  _lil.py                   List of Lists format (lil_array, lil_matrix)
  _matrix.py                spmatrix compatibility class
  _matrix_io.py             save_npz, load_npz
  _spfuncs.py               Sparse function utilities
  _sputils.py               Sparse utility functions
  _generate_sparsetools.py  Code generator for sparsetools C++
  _csparsetools.pyx.in      Cython sparse tools template
  base.py, bsr.py, coo.py, csc.py, csr.py, compressed.py, construct.py,
  data.py, dia.py, dok.py, extract.py, lil.py, spfuncs.py, sputils.py,
  sparsetools.py            Deprecated shims
  sparsetools/              C++ sparsetools implementation
    sparsetools.h/cxx       Main entry point
    bsr.h/cxx, csc.h/cxx, csr.h/cxx  Format-specific operations
    coo.h, dia.h, dense.h   Additional format headers
    other.cxx               Miscellaneous operations
    bool_ops.h, complex_ops.h, util.h, csgraph.h  Utility headers
  csgraph/                  Compressed Sparse Graph submodule
    __init__.py             Graph algorithm API
    _laplacian.py           Graph Laplacian
    _validation.py          Input validation
    _flow.pyx               Maximum flow algorithms (Cython)
    _matching.pyx           Bipartite matching (Cython)
    _min_spanning_tree.pyx  Minimum spanning tree (Cython)
    _reordering.pyx         Matrix reordering (Cython)
    _shortest_path.pyx      Shortest path algorithms (Cython)
    _tools.pyx              Graph tools (Cython)
    _traversal.pyx          BFS/DFS traversal (Cython)
    parameters.pxi          Shared Cython parameters
    tests/                  10 test files
  linalg/                   Sparse linear algebra submodule
    __init__.py             Sparse linalg API
    _interface.py           LinearOperator and variants
    _matfuncs.py            expm, matrix functions for sparse matrices
    _expm_multiply.py       expm_multiply action
    _funm_multiply_krylov.py Krylov-based matrix function-vector product
    _norm.py                Sparse matrix norm
    _onenormest.py          1-norm estimator
    _special_sparse_arrays.py  LaplacianNd, Sakurai, MikotaPair
    _svdp.py                PROPACK SVD wrapper
    _dsolve/                Direct solvers
      __init__.py
      linsolve.py           spsolve, factorized, splu, spilu
      _add_newdocs.py       Documentation additions
      _superlu_utils.c, _superlumodule.c, _superluobject.c/h  SuperLU C wrappers
      SuperLU/              Full SuperLU 5.x C library (~160 .c/.h files)
    _eigen/                 Eigenvalue solvers
      __init__.py
      _svds.py, _svds_doc.py  Truncated SVD
      arpack/               ARPACK wrapper
        __init__.py
        arpack.py           eigs, eigsh implementation
        _arpackmodule.c     ARPACK C module
        arnaud/             ARPACK C library (arnaud fork)
      lobpcg/               Locally Optimal Block Preconditioned Conjugate Gradient
        __init__.py
        lobpcg.py           lobpcg eigensolver
    _isolve/                Iterative solvers
      __init__.py
      iterative.py          cg, cgs, bicg, bicgstab, gmres
      lgmres.py             LGMRES solver
      lsmr.py               LSMR solver
      lsqr.py               LSQR solver
      minres.py             MINRES solver
      tfqmr.py              TFQMR solver
      _gcrotmk.py           GCROT(m,k) solver
      utils.py              Common utilities
    _propack/               PROPACK SVD library
      _propackmodule.c      PROPACK C module
      PROPACK/              Full PROPACK C library
    dsolve.py, eigen.py, interface.py, isolve.py, matfuncs.py  Deprecated shims
    tests/                  12 test files
  tests/                    25 test files
```

### File Counts

- Python: 125, Cython (.pyx): 7, Cython (.pxi): 1
- C: 200 (includes SuperLU ~160, ARPACK ~12, PROPACK ~10, sparsetools wrappers)
- C++ (.cxx): 5
- Headers (.h): 44
- Test files: 47

### Cross-Module Dependencies

- Imports from: `scipy._lib`, `scipy.linalg`, `scipy.special`
- Weak/optional imports: `scipy.io`, `scipy.stats`

### Key Types/Classes

- `_spbase` (_base.py:85) - Base class for all sparse arrays/matrices
- `sparray` (_base.py:1651) - Abstract sparse array base
- `spmatrix` (_matrix.py:1) - Legacy sparse matrix base
- `csr_array` / `csr_matrix` (_csr.py:324/447) - Compressed Sparse Row
- `csc_array` / `csc_matrix` (_csc.py:179/274) - Compressed Sparse Column
- `coo_array` / `coo_matrix` (_coo.py:28/1776) - COOrdinate format
- `bsr_array` / `bsr_matrix` (_bsr.py:672/783) - Block Sparse Row
- `dia_array` / `dia_matrix` (_dia.py:506/582) - Diagonal format
- `dok_array` / `dok_matrix` (_dok.py:654/707) - Dictionary of Keys
- `lil_array` / `lil_matrix` (_lil.py:508/573) - List of Lists
- `LinearOperator` (_interface.py:56) - Abstract linear operator
- `LaplacianNd` (_special_sparse_arrays.py:10) - N-D Laplacian operator
- `Sakurai` (_special_sparse_arrays.py:521) - Sakurai-Sugiura test matrix
- `MikotaPair` (_special_sparse_arrays.py:839) - Generalized eigenvalue test pair
- `SparseWarning` (_base.py:19) - Base sparse warning
- `SparseEfficiencyWarning` (_base.py:28) - Efficiency warning

---

## scipy.integrate

### Module Tree

```
integrate/
  __init__.py               Public API: quad, dblquad, tplquad, nquad, solve_ivp, ode, etc.
  _quadpack_py.py           quad, dblquad, tplquad, nquad wrappers
  _quadrature.py            trapezoid, simpson, romb, fixed_quad, newton_cotes, cumulative_*
  _odepack_py.py            odeint wrapper
  _ode.py                   ode, complex_ode class-based ODE interface
  _quad_vec.py              quad_vec for vector-valued integration
  _bvp.py                   Boundary value problem solver (solve_bvp)
  _cubature.py              Multi-dimensional cubature
  _tanhsinh.py              Tanh-sinh (double exponential) quadrature
  _lebedev.py               Lebedev quadrature on the sphere
  _rules/                   Quadrature rule implementations
    __init__.py
    _base.py                Rule, FixedRule, NestedFixedRule, ProductNestedFixed
    _gauss_kronrod.py       Gauss-Kronrod quadrature
    _gauss_legendre.py      Gauss-Legendre quadrature
    _genz_malik.py          Genz-Malik cubature rule
  _ivp/                     Initial value problem solvers
    __init__.py
    ivp.py                  solve_ivp dispatcher
    base.py                 OdeSolver base class, DenseOutput
    rk.py                   Runge-Kutta solvers: RK23, RK45, DOP853
    bdf.py                  BDF (Backward Differentiation Formula)
    radau.py                Radau IIA implicit Runge-Kutta
    lsoda.py                LSODA wrapper
    common.py               OdeSolution, shared utilities
    dop853_coefficients.py  DOP853 coefficients
    tests/                  2 test files (test_ivp.py, test_rk.py)
  dop.py, lsoda.py, odepack.py, quadpack.py, vode.py  Deprecated shims
  src/
    dop.c/h                 Dormand-Prince C implementation
    lsoda.c/h               LSODA C implementation
    vode.c/h, zvode.c/h     VODE/ZVODE C implementations
    blaslapack_declarations.h  BLAS/LAPACK declarations
    LICENSE_DOP             License for DOP code
  __quadpack.c/h            QUADPACK C implementation
  _dopmodule.c              DOP Python module
  _dzvodemodule.c           DZVODE Python module
  _odepackmodule.c          ODEPACK Python module
  tests/                    10 test files
    test_integrate.py, test_quadpack.py, test_quadrature.py,
    test_bvp.py, test_cubature.py, test_tanhsinh.py,
    test__quad_vec.py, test_banded_ode_solvers.py
    _test_multivariate.c    C test helper
```

### File Counts

- Python: 41, Cython: 0
- C: 9, Headers (.h): 6
- Test files: 10 (+ 2 in _ivp/tests)

### Cross-Module Dependencies

- Imports from: `scipy._lib`, `scipy.linalg`, `scipy.sparse`, `scipy.special`, `scipy.optimize`, `scipy.interpolate`

### Key Types/Classes

- `OdeSolver` (base.py:29) - Abstract IVP solver base class
- `RK23` (rk.py:183) - Explicit Runge-Kutta order 2(3)
- `RK45` (rk.py:293) - Explicit Runge-Kutta order 4(5) (Dormand-Prince)
- `DOP853` (rk.py:407) - Explicit Runge-Kutta order 8(5,3)
- `Radau` (radau.py:179) - Implicit Radau IIA order 5
- `BDF` (bdf.py:72) - Implicit multi-step BDF
- `LSODA` (lsoda.py:7) - Adams/BDF auto-switching
- `OdeSolution` (common.py:137) - Dense output container
- `DenseOutput` (base.py:237) - Continuous solution interface
- `ode` (_ode.py:100) - Legacy ODE class interface
- `complex_ode` (_ode.py:579) - Complex ODE wrapper
- `IntegrationWarning` (_quadpack_py.py:15) - Integration warning
- `ODEintWarning` (_odepack_py.py:13) - odeint warning
- `CubatureResult` (_cubature.py:50) - Cubature result container
- `Rule` (_base.py:6) - Abstract quadrature rule
- `GaussKronrodQuadrature` (_gauss_kronrod.py:9) - Gauss-Kronrod rule
- `GenzMalikCubature` (_genz_malik.py:11) - Genz-Malik multi-D rule

---

## scipy.optimize

### Module Tree

```
optimize/
  __init__.py               Public API: minimize, root, linprog, least_squares, etc.
  __init__.pxd              Cython declarations
  _optimize.py              Core: fminbound, brent, golden, bracket, OptimizeResult
  _minimize.py              minimize() dispatcher for all methods
  _root.py                  root() dispatcher
  _root_scalar.py           root_scalar() dispatcher
  _zeros_py.py              brentq, brenth, ridder, bisect, newton, toms748
  _bracket.py               Bracketing routines
  _chandrupatla.py          Chandrupatla root-finding
  _linesearch.py            Line search methods (Wolfe conditions)
  _dcsrch.py                DCSRCH line search
  _trustregion.py           Base trust-region solver
  _trustregion_dogleg.py    Dogleg trust-region
  _trustregion_ncg.py       Newton-CG trust-region
  _trustregion_exact.py     Exact trust-region (Steihaug)
  _trustregion_krylov.py    Krylov trust-region (via trlib)
  _lbfgsb_py.py             L-BFGS-B wrapper
  _slsqp_py.py              SLSQP wrapper
  _cobyla_py.py             COBYLA wrapper
  _cobyqa_py.py             COBYQA wrapper
  _tnc.py                   TNC wrapper
  _basinhopping.py          Basin-hopping global optimizer
  _differentialevolution.py Differential Evolution
  _dual_annealing.py        Dual Annealing
  _shgo.py                  SHGO (Simplicial Homology Global Optimization)
  _direct_py.py             DIRECT optimizer
  _spectral.py              Spectral projected gradient
  _constraints.py           Bounds, LinearConstraint, NonlinearConstraint
  _differentiable_functions.py  ScalarFunction, VectorFunction with auto-diff
  _hessian_update_strategy.py   BFGS, SR1 Hessian update strategies
  _numdiff.py               Numerical differentiation helpers
  _group_columns.py/pyx     Column grouping for sparse Jacobians
  _linprog.py               linprog dispatcher
  _linprog_highs.py         HiGHS LP solver interface
  _linprog_ip.py            Interior-point LP
  _linprog_rs.py            Revised simplex LP
  _linprog_simplex.py       Simplex LP
  _linprog_util.py          LP utilities
  _linprog_doc.py           LP documentation strings
  _milp.py                  Mixed Integer LP via HiGHS
  _remove_redundancy.py     LP redundancy removal
  _minpack_py.py            fsolve, leastsq (MINPACK wrappers)
  _nonlin.py                Nonlinear equation solvers: Broyden, Anderson, Krylov
  _qap.py                   Quadratic assignment problem
  _isotonic.py              Isotonic regression
  _nnls.py                  Non-negative least squares
  _elementwise.py           Elementwise optimization
  _tstutils.py              Test utilities
  _bglu_dense.pyx           Dense BGLU (Cython)
  _lsq/                     Least squares subpackage
    __init__.py
    least_squares.py        least_squares() dispatcher
    lsq_linear.py           Linear least squares
    common.py               Shared LSQ utilities
    dogbox.py               Dogbox algorithm
    trf.py                  Trust Region Reflective
    trf_linear.py           Linear TRF
    bvls.py                 Bounded Variable Least Squares
    givens_elimination.pyx  Givens rotations (Cython)
  _trustregion_constr/      Trust-region constrained optimization
    __init__.py
    minimize_trustregion_constr.py  Main TR-constrained minimizer
    canonical_constraint.py Canonical constraint representation
    equality_constrained_sqp.py    SQP for equality constraints
    projections.py          Constraint projections
    qp_subproblem.py        QP subproblem solver
    report.py               Optimization progress reporting
    tr_interior_point.py    Interior point trust-region
    tests/                  5 test files
  _shgo_lib/                SHGO helper library
    __init__.py
    _complex.py             Complex topology support
    _vertex.py              Vertex/simplex data structures
  _highspy/                 HiGHS solver Python interface
    __init__.py
    _highs_wrapper.py       HiGHS Python wrapper
    highs_options.cpp       HiGHS C++ options
  _trlib/                   Trust-region library (C)
    __init__.py
    _trlib.pyx              Cython wrapper
    ctrlib.pxd              Cython declarations
    trlib.h                 Main header
    trlib_krylov.c/h, trlib_leftmost.c/h, trlib_eigen_inverse.c/h,
    trlib_quadratic_zero.c/h, trlib_tri_factor.c/h  Algorithm implementations
    trlib_private.h, trlib/  Internal headers
  cython_optimize/          Cython optimization interface
    __init__.py
    _zeros.pxd, _zeros.pyx.in, c_zeros.pxd  Cython zero-finding
  Zeros/                    C root-finding implementations
    bisect.c, brentq.c, brenth.c, ridder.c, zeros.h
  src/
    lbfgsb.c/h              L-BFGS-B C implementation
    slsqp.c/h               SLSQP C implementation
    minpack.c/h             MINPACK C implementation
    nnls.c/h                NNLS C implementation
    blaslapack_declarations.h
  rectangular_lsap/         Linear sum assignment (C++)
    rectangular_lsap.cpp/h
  tnc/                      TNC C library
    tnc.c/h, _moduleTNC.pyx, example.c
  _pava/                    Pool Adjacent Violators Algorithm
    pava_pybind.cpp
  _zerosmodule.c, _minpackmodule.c, _lbfgsbmodule.c,
  _slsqpmodule.c, _lsapmodule.c, _directmodule.c/h  C Python modules
  cobyla.py, lbfgsb.py, linesearch.py, minpack.py, minpack2.py,
  moduleTNC.py, nonlin.py, optimize.py, slsqp.py, tnc.py, zeros.py,
  elementwise.py            Deprecated shims
  tests/                    48 test files
```

### File Counts

- Python: 132, Cython (.pyx): 6, Cython (.pxd): 5
- C: 25, C++ (.cpp): 3
- Headers (.h): 19
- Test files: 48

### Cross-Module Dependencies

- Imports from: `scipy._lib`, `scipy.linalg`, `scipy.sparse`, `scipy.special`
- Weak/optional imports: `scipy.stats`

### Key Types/Classes

- `OptimizeResult` (_optimize.py:112) - Result container for all optimizers
- `OptimizeWarning` (_optimize.py:157) - Optimization warning
- `Bounds` (_constraints.py:234) - Variable bounds
- `LinearConstraint` (_constraints.py:131) - Linear constraint
- `NonlinearConstraint` (_constraints.py:22) - Nonlinear constraint
- `ScalarFunction` (_differentiable_functions.py:128) - Scalar objective wrapper
- `VectorFunction` (_differentiable_functions.py:525) - Vector objective wrapper
- `DifferentialEvolutionSolver` (_differentialevolution.py:538) - DE solver
- `SHGO` (_shgo.py:488) - Simplicial homology optimizer
- `BasinHoppingRunner` (_basinhopping.py:42) - Basin-hopping runner
- `BFGS` (_hessian_update_strategy.py:285) - BFGS Hessian update
- `SR1` (_hessian_update_strategy.py:425) - SR1 Hessian update
- `LbfgsInvHessProduct` (_lbfgsb_py.py:461) - L-BFGS-B inverse Hessian product
- `LinearOperator` (via sparse.linalg) - Used for Hessian-vector products
- `TOMS748Solver` (_zeros_py.py:1121) - TOMS Algorithm 748 root finder
- `RootResults` (_zeros_py.py:35) - Root-finding result container
- `LineSearchWarning` (_linesearch.py:23) - Line search warning
- `BracketError` (_optimize.py:3113) - Bracket failure

---

## scipy.fft

### Module Tree

```
fft/
  __init__.py               Public API: fft, ifft, rfft, irfft, dct, dst, fht, etc.
  _basic.py                 FFT function dispatchers (fft, ifft, fft2, fftn, rfft, etc.)
  _basic_backend.py         Backend dispatch for basic FFT
  _realtransforms.py        DCT, DST function dispatchers
  _realtransforms_backend.py Backend dispatch for real transforms
  _fftlog.py                Fast Hankel Transform (fht, ifht)
  _fftlog_backend.py        Backend for fftlog
  _helper.py                next_fast_len, prev_fast_len, fftfreq, rfftfreq, fftshift
  _backend.py               Backend management (_ScipyBackend)
  _debug_backends.py        NumPyBackend, EchoBackend for testing
  _pocketfft/               PocketFFT backend (default)
    __init__.py
    basic.py                PocketFFT basic transforms
    helper.py               PocketFFT helper functions
    realtransforms.py       PocketFFT DCT/DST
    pypocketfft.cxx         PocketFFT C++ implementation
    tests/                  2 test files (test_basic.py, test_real_transforms.py)
  tests/                    8 test files
    test_basic.py, test_backend.py, test_fftlog.py, test_helper.py,
    test_multithreading.py, test_real_transforms.py
    mock_backend.py         Mock backend for testing
```

### File Counts

- Python: 25, Cython: 0
- C++ (.cxx): 1 (pypocketfft)
- Test files: 8

### Cross-Module Dependencies

- Imports from: `scipy._lib`, `scipy.special`

### Key Types/Classes

- `_ScipyBackend` (_backend.py:8) - Default scipy FFT backend
- `NumPyBackend` (_debug_backends.py:3) - NumPy fallback backend
- `EchoBackend` (_debug_backends.py:16) - Debug echo backend

---

## scipy.special

### Module Tree

```
special/
  __init__.py               Public API: Bessel functions, gamma, erf, elliptic, etc.
  __init__.pxd              Cython declarations for special functions
  _basic.py                 High-level special function wrappers
  _orthogonal.py            Orthogonal polynomials: Legendre, Hermite, Laguerre, Jacobi, Chebyshev
  _lambertw.py              Lambert W function
  _logsumexp.py             logsumexp, softmax
  _ellip_harm.py            Ellipsoidal harmonic functions
  _spherical_bessel.py      Spherical Bessel functions
  _sf_error.py              SpecialFunctionWarning, SpecialFunctionError
  _input_validation.py      Input validation utilities
  _spfun_stats.py           Statistical special functions (multigammaln)
  _support_alternative_backends.py  Array API backend support
  _multiufuncs.py           MultiUFunc class for generalized ufuncs
  _testutils.py             FuncData test framework
  _mptestutils.py           mpmath-based test utilities (Arg, ComplexArg, MpmathData)
  _add_newdocs.py           Additional docstrings
  _generate_pyx.py          Cython code generator (Func, Ufunc, errstate)
  _specfun.pyx              Special function Cython wrappers
  _comb.pyx                 Combinatorial functions (Cython)
  _ellip_harm_2.pxd/pyx    Ellipsoidal harmonics part 2 (Cython)
  _test_internal.pyi/pyx    Internal test Cython module
  _ufuncs_extra_code.pxi, _ufuncs_extra_code_common.pxi  Cython include files
  _ufuncs.pyi               Type stubs for generated ufuncs
  cython_special.pxd/pyi/pyx  Cython-accessible special functions
  orthogonal_eval.pxd       Orthogonal polynomial evaluation declarations
  sf_error.pxd/py/cc/h      Error handling across C/Python boundary
  functions.json             Function metadata
  Many .pxd files:          _agm, _boxcox, _cdflib_wrappers, _complexstuff, _convex_analysis,
                            _ellipk, _factorial, _hyp0f1, _hypergeometric, _legacy, _ndtri_exp,
                            _sici, _spence  (Cython declarations)
  C/C++ implementations:
    _special_ufuncs.cpp, _special_ufuncs_docs.cpp  Ufunc C++ implementation
    _gufuncs.cpp, _gufuncs_docs.cpp  Generalized ufunc C++ implementation
    xsf_wrappers.cpp/h      XSF wrapper functions
    cdflib.c/h               CDF library
    _cosine.c/h              Cosine distribution
    dd_real_wrappers.cpp/h   Double-double arithmetic
    _wright.cxx/h            Wright function
    wright.cc/hh             Wright omega function
    boost_special_functions.h  Boost special function headers
    gen_harmonic.h           Generalized harmonic numbers
    stirling2.h              Stirling numbers of second kind
    lapack_defs.h            LAPACK declarations
    _complexstuff.h, _round.h  Utility headers
    ellint_carlson_wrap.cxx/hh  Carlson elliptic integrals
    ellint_carlson_cpp_lite/  Carlson elliptic integral C++ library
  _precompute/              Precomputation scripts
    __init__.py, utils.py, cosine_cdf.py, expn_asy.py, gammainc_asy.py,
    gammainc_data.py, hyp2f1_data.py, lambertw.py, loggamma.py,
    struve_convergence.py, wright_bessel.py, wright_bessel_data.py,
    wrightomega.py, zetac.py
  utils/                    Data conversion utilities
    convert.py, datafunc.py, makenpz.py
  add_newdocs.py, basic.py, orthogonal.py, sf_error.py, specfun.py,
  spfun_stats.py            Deprecated shims
  tests/                    57 test files
    test_basic.py, test_data.py, test_mpmath.py, test_orthogonal.py,
    test_hyp2f1.py, test_gammainc.py, test_erfinv.py, etc.
    data/                   Extensive test data: boost/, gsl/, local/ directories
    _cython_examples/       Cython extension examples
```

### File Counts

- Python: 98, Cython (.pyx): 6, Cython (.pxd): 19, Cython (.pxi): 2
- C: 2, C++ (.cpp): 6, C++ (.cxx): 2, C++ (.cc): 2
- Headers (.h): 12, C++ Headers (.hh): 13
- Test files: 57

### Cross-Module Dependencies

- Imports from: `scipy._lib`, `scipy.linalg`
- Weak/optional imports: `scipy.integrate`, `scipy.interpolate`, `scipy.optimize`, `scipy.stats`

### Key Types/Classes

- `SpecialFunctionWarning` (_sf_error.py:5) - Warning for special function edge cases
- `SpecialFunctionError` (_sf_error.py:13) - Error for special function failures
- `orthopoly1d` (_orthogonal.py:114) - Orthogonal polynomial class (extends np.poly1d)
- `MultiUFunc` (_multiufuncs.py:24) - Multi-output ufunc wrapper
- `FuncData` (_testutils.py:88) - Test data framework for function accuracy testing
- `Arg`, `ComplexArg`, `IntArg`, `FixedArg` (_mptestutils.py) - Test argument generators

---

## scipy.stats

### Module Tree

```
stats/
  __init__.py               Public API: distributions, hypothesis tests, descriptive stats
  _continuous_distns.py     ~130 continuous distribution generators (norm_gen, t_gen, chi2_gen, etc.)
  _discrete_distns.py       ~25 discrete distribution generators (binom_gen, poisson_gen, etc.)
  _distn_infrastructure.py  rv_generic, rv_continuous, rv_discrete, rv_frozen base classes
  _distribution_infrastructure.py  New distribution API: ContinuousDistribution, DiscreteDistribution, etc.
  _new_distributions.py     New-style distributions: Normal, Uniform, Logistic, Binomial
  _probability_distribution.py  _ProbabilityDistribution abstract base
  _multivariate.py          Multivariate distributions: multivariate_normal, wishart, dirichlet, etc.
  _stats_py.py              Core statistical functions: describe, pearsonr, spearmanr, ttest_*, anova, etc.
  _morestats.py             More tests: shapiro, anderson, bartlett, levene, mood, etc.
  _hypotests.py             Hypothesis tests: CramerVonMises, barnard, boschloo, etc.
  _kde.py                   gaussian_kde kernel density estimation
  _fit.py                   Distribution fitting (FitResult)
  _binomtest.py             Binomial test
  _mannwhitneyu.py          Mann-Whitney U test
  _wilcoxon.py              Wilcoxon signed-rank test
  _correlation.py           Correlation functions (Chatterjee, Spearman)
  _survival.py              Survival analysis (EmpiricalDistributionFunction, LogRank)
  _resampling.py            Bootstrap, permutation tests, power analysis
  _sensitivity_analysis.py  Sobol sensitivity analysis
  _multicomp.py             Multiple comparison corrections (Dunnett)
  _odds_ratio.py            Odds ratio
  _relative_risk.py         Relative risk
  _page_trend_test.py       Page trend test
  _bws_test.py              Baumgartner-Weiss-Schindler test
  _crosstab.py              Cross-tabulation
  _mgc.py                   Multiscale Graph Correlation
  _entropy.py               Entropy functions
  _censored_data.py         CensoredData for survival analysis
  _covariance.py            Covariance representations (Cholesky, Diagonal, Eigendecomposition, etc.)
  _qmc.py                   Quasi-Monte Carlo: Halton, Sobol, LatinHypercube, PoissonDisk
  _qmvnt.py                 Quasi-Monte Carlo multivariate normal/t
  _sampling.py              Sampling methods: RatioUniforms, FastGeneratorInversion
  _quantile.py              Quantile functions
  _variation.py             Coefficient of variation
  _binned_statistic.py      Binned statistics
  _ksstats.py               Kolmogorov-Smirnov distribution
  _tukeylambda_stats.py     Tukey Lambda statistics
  _continued_fraction.py    Continued fraction evaluation
  _finite_differences.py    Finite difference utilities
  _axis_nan_policy.py       NaN handling policy decorators
  _stats_mstats_common.py   Shared mstats/stats code
  _common.py                Common utilities
  _constants.py             Statistical constants
  _distr_params.py          Distribution parameter definitions
  _warnings_errors.py       DegenerateDataWarning, FitError, etc.
  _result_classes.py        Result namedtuples
  _stats_pythran.py         Pythran-accelerated statistics
  _mstats_basic.py          Masked statistics (basic)
  _mstats_extras.py         Masked statistics (extras)
  _ansari_swilk_statistics.pyx  Ansari-Bradley/Shapiro-Wilk (Cython)
  _biasedurn.pxd/pyx        Biased urn distributions (Cython)
  _stats.pxd/pyx            Core statistics Cython accelerations
  _qmc_cy.pyi/pyx           QMC Cython accelerations
  _qmvnt_cy.pyx             QMC multivariate normal/t (Cython)
  _sobol.pyi/pyx            Sobol sequence (Cython)
  _sobol_direction_numbers.npz  Sobol direction numbers data
  _levy_stable/             Levy stable distribution
    __init__.py             levy_stable_gen, levy_stable_frozen
    levyst.pyx              Cython implementation
    c_src/levyst.c/h        C implementation
  _rcont/                   Random contingency tables
    __init__.py
    rcont.pyx               Cython implementation
    _rcont.c/h              C implementation
  _unuran/                  UNU.RAN sampling library
    __init__.py
    unuran.pxd, unuran_wrapper.pyi/pyx  Cython wrapper
    config.h, unuran_callback.h  C headers
  biasedurn/                Biased urn C++ library
    erfres.cpp, fnchyppr.cpp, impls.cpp, stoc1.cpp, stoc3.cpp, wnchyppr.cpp
    randomc.h, stocR.h, stocc.h
  libnpyrandom/             NumPy random C library
    distributions.c/h, logfactorial.c/h, ziggurat_constants.h
  biasedurn.py, contingency.py, distributions.py, kde.py,
  morestats.py, mstats.py, mstats_basic.py, mstats_extras.py,
  mvn.py, qmc.py, sampling.py, stats.py  Deprecated shims
  tests/                    39 test files
    test_distributions.py, test_stats.py, test_multivariate.py,
    test_continuous_basic.py, test_discrete_basic.py, test_qmc.py,
    test_resampling.py, test_fit.py, test_hypotests.py, etc.
    data/                   Test data: levy_stable/, nist_anova/, nist_linregress/
    test_generation/        Test generation scripts
```

### File Counts

- Python: 111, Cython (.pyx): 9, Cython (.pxd): 3
- C: 4, C++ (.cpp): 6
- Headers (.h): 10
- Test files: 39

### Cross-Module Dependencies

- Imports from: `scipy._lib`, `scipy.linalg`, `scipy.special`, `scipy.optimize`, `scipy.integrate`, `scipy.interpolate`, `scipy.sparse`, `scipy.spatial`, `scipy.fft`, `scipy.ndimage`
- Note: stats is the heaviest consumer of other scipy modules

### Key Types/Classes

- `rv_generic` (_distn_infrastructure.py:681) - Base class for all distributions
- `rv_continuous` (_distn_infrastructure.py:1669) - Continuous distribution base
- `rv_discrete` (_distn_infrastructure.py:3173) - Discrete distribution base
- `rv_frozen` (_distn_infrastructure.py:507) - Frozen (parameterized) distribution
- `ContinuousDistribution` (_distribution_infrastructure.py:3687) - New continuous distribution API
- `DiscreteDistribution` (_distribution_infrastructure.py:3709) - New discrete distribution API
- `UnivariateDistribution` (_distribution_infrastructure.py:1515) - New univariate base
- `TransformedDistribution` (_distribution_infrastructure.py:4487) - Transformed distribution
- `ShiftedScaledDistribution` (_distribution_infrastructure.py:4691) - Loc/scale wrapper
- `TruncatedDistribution` (_distribution_infrastructure.py:4540) - Truncated distribution
- `Mixture` (_distribution_infrastructure.py:5141) - Mixture distribution
- `Normal` (_new_distributions.py:16) - New Normal distribution
- `Uniform` (_new_distributions.py:355) - New Uniform distribution
- `Covariance` (_covariance.py:12) - Abstract covariance representation
- `gaussian_kde` (_kde.py:38) - Kernel density estimator
- `FitResult` (_fit.py:42) - Distribution fitting result
- `CensoredData` (_censored_data.py:61) - Censored data container
- `QMCEngine` (_qmc.py:802) - Quasi-Monte Carlo engine base
- `Halton` (_qmc.py:1117) - Halton sequence
- `Sobol` (_qmc.py:1609) - Sobol sequence
- `LatinHypercube` (_qmc.py:1286) - Latin Hypercube sampling
- `~130 distribution generators` (_continuous_distns.py) - norm_gen, t_gen, chi2_gen, f_gen, etc.
- `~25 discrete generators` (_discrete_distns.py) - binom_gen, poisson_gen, geom_gen, etc.
- `~15 multivariate distributions` (_multivariate.py) - multivariate_normal_gen, wishart_gen, etc.

---

## scipy.signal

### Module Tree

```
signal/
  __init__.py               Public API: filters, systems, spectral, windows
  _signal_api.py            Signal processing API
  _signaltools.py           Core: convolve, correlate, fftconvolve, hilbert, detrend, etc.
  _filter_design.py         IIR filter design: butter, cheby1, cheby2, ellip, bessel, etc.
  _fir_filter_design.py     FIR filter design: firwin, firwin2, kaiserord, etc.
  _ltisys.py                LTI system classes: lti, dlti, TransferFunction, ZerosPolesGain, StateSpace
  _lti_conversion.py        System representation conversions (tf2zpk, zpk2tf, etc.)
  _spectral_py.py           Spectral analysis: periodogram, welch, csd, coherence, spectrogram
  _short_time_fft.py        ShortTimeFFT class
  _czt.py                   Chirp Z-Transform (CZT, ZoomFFT)
  _peak_finding.py          find_peaks, argrelmin, argrelmax
  _savitzky_golay.py        Savitzky-Golay filter
  _waveforms.py             chirp, gausspulse, sawtooth, square, unit_impulse, sweep_poly
  _wavelets.py              Continuous wavelet transforms
  _upfirdn.py               Upsampling/downsampling with FIR filter
  _max_len_seq.py           Maximum length sequences
  _max_len_seq_inner.py/pyx MLS inner loop (Python/Cython)
  _spline_filters.py        Spline filter functions
  _polyutils.py             Polynomial utilities
  _arraytools.py            Array manipulation tools
  _delegators.py            Backend delegation
  _support_alternative_backends.py  Array API backend support
  _peak_finding_utils.pyx   Peak finding utilities (Cython)
  _sosfilt.pyx              Second-order sections filter (Cython)
  _upfirdn_apply.pyx        Upfirdn apply (Cython)
  C/C++ implementations:
    _sigtoolsmodule.cc      Signal tools C++ module
    _sigtools.hh            Signal tools header
    _firfilter.cc           FIR filter C++ implementation
    _lfilter.cc             Linear filter C++ implementation
    _correlate_nd.cc        N-D correlation C++ implementation
    _medianfilter.cc        Median filter C++ implementation
    _splinemodule.cc/h      Spline module C++ implementation
    _spline.pyi             Spline type stubs
  windows/                  Window functions submodule
    __init__.py
    _windows.py             All window functions: hann, hamming, blackman, kaiser, etc.
    windows.py              Deprecated shim
  bsplines.py, filter_design.py, fir_filter_design.py,
  lti_conversion.py, ltisys.py, signaltools.py, spectral.py,
  spline.py, waveforms.py, wavelets.py  Deprecated shims
  tests/                    21 test files
    test_signaltools.py, test_filter_design.py, test_fir_filter_design.py,
    test_ltisys.py, test_dltisys.py, test_spectral.py, test_windows.py,
    test_peak_finding.py, test_short_time_fft.py, test_czt.py, etc.
```

### File Counts

- Python: 58, Cython (.pyx): 4
- C++ (.cc): 6, C++ Headers (.hh): 1, Headers (.h): 1
- Test files: 21

### Cross-Module Dependencies

- Imports from: `scipy._lib`, `scipy.fft`, `scipy.linalg`, `scipy.special`, `scipy.optimize`, `scipy.interpolate`, `scipy.ndimage`, `scipy.integrate`, `scipy.spatial`, `scipy.stats`, `scipy.datasets`

### Key Types/Classes

- `lti` (_ltisys.py:136) - Continuous-time LTI system base
- `dlti` (_ltisys.py:409) - Discrete-time LTI system base
- `LinearTimeInvariant` (_ltisys.py:47) - Abstract LTI base
- `TransferFunction` (_ltisys.py:712) - Transfer function representation
- `TransferFunctionContinuous` (_ltisys.py:959) - Continuous TF
- `TransferFunctionDiscrete` (_ltisys.py:1035) - Discrete TF
- `ZerosPolesGain` (_ltisys.py:1099) - ZPK representation
- `ZerosPolesGainContinuous` (_ltisys.py:1301) - Continuous ZPK
- `ZerosPolesGainDiscrete` (_ltisys.py:1370) - Discrete ZPK
- `StateSpace` (_ltisys.py:1439) - State-space representation
- `StateSpaceContinuous` (_ltisys.py:1855) - Continuous state-space
- `StateSpaceDiscrete` (_ltisys.py:1928) - Discrete state-space
- `ShortTimeFFT` (_short_time_fft.py:225) - Short-time Fourier transform
- `CZT` (_czt.py:115) - Chirp Z-Transform
- `ZoomFFT` (_czt.py:280) - Zoom FFT via CZT
- `BadCoefficients` (_filter_design.py:40) - Filter coefficient warning

---

## scipy.spatial

### Module Tree

```
spatial/
  __init__.py               Public API: KDTree, Voronoi, ConvexHull, Delaunay, distance
  _kdtree.py                KDTree (pure Python wrapper over cKDTree)
  _ckdtree.pyx              cKDTree (Cython accelerated kd-tree)
  _qhull.pxd/pyi/pyx        Qhull wrappers: Delaunay, ConvexHull, Voronoi, HalfspaceIntersection
  _voronoi.pyi/pyx           Voronoi diagram (Cython)
  _hausdorff.pyx             Hausdorff distance (Cython)
  _spherical_voronoi.py      SphericalVoronoi on the unit sphere
  _geometric_slerp.py        Geometric spherical linear interpolation
  _procrustes.py             Procrustes analysis
  _plotutils.py              Plotting utilities (Voronoi, Delaunay, ConvexHull plots)
  distance.py/pyi            Distance functions: pdist, cdist, squareform + all metrics
  setlist.pxd                Cython set-list declarations
  ckdtree/                   cKDTree C++ implementation
    src/
      build.cxx, query.cxx, query_ball_point.cxx, query_ball_tree.cxx,
      query_pairs.cxx, count_neighbors.cxx, sparse_distances.cxx
      ckdtree_decl.h, coo_entries.h, distance.h, distance_base.h,
      ordered_pair.h, rectangle.h
  src/                       Distance metric C/C++ implementations
    distance_impl.h, distance_metrics.h, distance_pybind.cpp,
    distance_wrap.c, function_ref.h, views.h
  qhull_misc.c/h             Qhull miscellaneous C code
  transform/                 Rotation/rigid transform submodule
    __init__.py
    _rotation.py             Rotation class (quaternion-based)
    _rotation_cy.pyx         Rotation Cython accelerations
    _rotation_xp.py          Rotation array API support
    _rotation_groups.py      Rotation group generators
    _rotation_spline.py      RotationSpline
    _rigid_transform.py      RigidTransform (rotation + translation)
    _rigid_transform_cy.pyx  RigidTransform Cython accelerations
    _rigid_transform_xp.py   RigidTransform array API support
    rotation.py              Deprecated shim
    tests/                   4 test files
  ckdtree.py, kdtree.py, qhull.py  Deprecated shims
  tests/                     12 test files
    test_kdtree.py, test_qhull.py, test_distance.py, test_hausdorff.py,
    test_spherical_voronoi.py, test_slerp.py, test__procrustes.py,
    test__plotutils.py
    data/                    Test data files
```

### File Counts

- Python: 32, Cython (.pyx): 6, Cython (.pxd): 2
- C: 2, C++ (.cpp): 1, C++ (.cxx): 7
- Headers (.h): 11
- Test files: 12

### Cross-Module Dependencies

- Imports from: `scipy._lib`, `scipy.linalg`, `scipy.special`, `scipy.optimize`, `scipy.interpolate`, `scipy.constants`, `scipy.stats`

### Key Types/Classes

- `KDTree` (_kdtree.py:234) - k-d tree for spatial queries
- `Rotation` (_rotation.py:62) - 3D rotation (quaternion, matrix, euler, rotvec)
- `Slerp` (_rotation.py:2680) - Spherical linear interpolation of rotations
- `RigidTransform` (_rigid_transform.py:59) - Rigid body transformation
- `RotationSpline` (_rotation_spline.py:251) - Smooth rotation interpolation
- `SphericalVoronoi` (_spherical_voronoi.py:36) - Voronoi on sphere
- `Rectangle` (_kdtree.py:98) - Hyperrectangle for KDTree
- `CDistMetricWrapper` (distance.py:1584) - cdist metric wrapper
- `PDistMetricWrapper` (distance.py:1611) - pdist metric wrapper
- `MetricInfo` (distance.py:1636) - Distance metric metadata
- (Qhull-based classes from _qhull.pyx: Delaunay, ConvexHull, Voronoi, HalfspaceIntersection)

---

## scipy.interpolate

### Module Tree

```
interpolate/
  __init__.py               Public API: interp1d, CubicSpline, BSpline, RBFInterpolator, etc.
  _polyint.py               _Interpolator1D base, KroghInterpolator, BarycentricInterpolator
  _interpolate.py           interp1d, interp2d, PPoly, BPoly, NdPPoly
  _cubic.py                 CubicHermiteSpline, PchipInterpolator, Akima1DInterpolator, CubicSpline
  _bsplines.py              BSpline class and construction (make_interp_spline, make_lsq_spline)
  _ndbspline.py             NdBSpline (N-dimensional B-spline)
  _fitpack2.py              UnivariateSpline, BivariateSpline + variants (LSQ, Smooth, Rect, Sphere)
  _fitpack_py.py            Low-level fitpack wrappers (splrep, splev, splint, sproot, etc.)
  _fitpack_impl.py          Fitpack implementation details
  _fitpack_repro.py         Fitpack reproducibility helpers
  _bary_rational.py         Barycentric rational interpolation (AAA, FloaterHormannInterpolator)
  _rgi.py                   RegularGridInterpolator
  _rgi_cython.pyx           RegularGridInterpolator Cython accelerations
  _ndgriddata.py            NearestNDInterpolator, griddata
  _rbf.py                   Rbf (legacy radial basis function)
  _rbfinterp.py             RBFInterpolator (modern RBF)
  _rbfinterp_common.py      RBF interpolation common code
  _rbfinterp_np.py          RBF NumPy implementation
  _rbfinterp_pythran.py     RBF Pythran-accelerated code
  _rbfinterp_xp.py          RBF array API support
  _pade.py                  Pade approximation
  _interpnd.pyx             N-D interpolation (Cython): LinearNDInterpolator, CloughTocher2DInterpolator
  _interpnd_info.py         N-D interpolation metadata
  _ppoly.pyx                Piecewise polynomial evaluation (Cython)
  _poly_common.pxi          Shared polynomial Cython includes
  dfitpack.py, fitpack.py, fitpack2.py, interpnd.py, interpolate.py,
  ndgriddata.py, polyint.py, rbf.py  Deprecated shims
  src/
    __fitpack.cc/h          Fitpack C++ implementation
    _dierckxmodule.cc       Dierckx module C++ wrapper
    _fitpackmodule.c        Fitpack C module
    dfitpack.c/h            Dierckx Fortran-to-C translation
  tests/                    13 test files
    test_interpolate.py, test_bsplines.py, test_fitpack.py, test_fitpack2.py,
    test_rgi.py, test_polyint.py, test_rbfinterp.py, test_bary_rational.py,
    test_ndgriddata.py, test_interpnd.py, test_pade.py, test_rbf.py, test_gil.py
    data/                   Test data files (.npz, .npy)
```

### File Counts

- Python: 43, Cython (.pyx): 3, Cython (.pxi): 1
- C: 2, C++ (.cc): 2
- Headers (.h): 2
- Test files: 13

### Cross-Module Dependencies

- Imports from: `scipy._lib`, `scipy.linalg`, `scipy.sparse`, `scipy.spatial`, `scipy.special`, `scipy.optimize`, `scipy.integrate`, `scipy.stats`

### Key Types/Classes

- `BSpline` (_bsplines.py:77) - B-spline curve/surface representation
- `NdBSpline` (_ndbspline.py:35) - N-dimensional B-spline
- `interp1d` (_interpolate.py:177) - 1-D interpolation (multiple methods)
- `PPoly` (_interpolate.py:849) - Piecewise polynomial in power form
- `BPoly` (_interpolate.py:1323) - Piecewise polynomial in Bernstein form
- `NdPPoly` (_interpolate.py:1882) - N-D piecewise polynomial
- `CubicSpline` (_cubic.py:628) - Cubic spline interpolation
- `CubicHermiteSpline` (_cubic.py:81) - Cubic Hermite spline
- `PchipInterpolator` (_cubic.py:183) - Piecewise Cubic Hermite (PCHIP)
- `Akima1DInterpolator` (_cubic.py:410) - Akima interpolation
- `KroghInterpolator` (_polyint.py:249) - Krogh interpolation
- `BarycentricInterpolator` (_polyint.py:537) - Barycentric interpolation
- `AAA` (_bary_rational.py:213) - AAA rational approximation
- `FloaterHormannInterpolator` (_bary_rational.py:630) - Floater-Hormann interpolation
- `RegularGridInterpolator` (_rgi.py:66) - Regular grid interpolation
- `NearestNDInterpolator` (_ndgriddata.py:20) - Nearest neighbor N-D
- `RBFInterpolator` (_rbfinterp.py:69) - Radial basis function interpolation
- `Rbf` (_rbf.py:57) - Legacy RBF interpolation
- `UnivariateSpline` (_fitpack2.py:394) - Smoothing spline
- `InterpolatedUnivariateSpline` (_fitpack2.py:1002) - Interpolating spline
- `LSQUnivariateSpline` (_fitpack2.py:1124) - Least-squares spline
- `BivariateSpline` (_fitpack2.py:1548) - 2-D spline base
- `RectBivariateSpline` (_fitpack2.py:1921) - Rectangular grid 2-D spline

---

## scipy.ndimage

### Module Tree

```
ndimage/
  __init__.py               Public API: filters, interpolation, morphology, measurements
  _filters.py               Convolution, correlation, gaussian_filter, median_filter, etc.
  _fourier.py               Fourier domain filters: fourier_gaussian, fourier_uniform, fourier_shift
  _interpolation.py         Geometric transforms: map_coordinates, affine_transform, rotate, zoom, shift
  _measurements.py          label, find_objects, sum, mean, variance, etc.
  _morphology.py            Binary/grey morphology: erosion, dilation, opening, closing, etc.
  _ni_support.py            Internal support functions
  _ni_docstrings.py         Docstring helpers
  _ndimage_api.py           N-D image API
  _delegators.py            Backend delegation
  _support_alternative_backends.py  Array API support
  filters.py, fourier.py, interpolation.py, measurements.py, morphology.py  Deprecated shims
  src/
    nd_image.c/h            Main C module entry point
    ni_filters.c/h          Filter C implementations
    ni_fourier.c/h          Fourier filter C implementations
    ni_interpolation.c/h    Interpolation C implementations
    ni_measure.c/h          Measurement C implementations
    ni_morphology.c/h       Morphology C implementations
    ni_splines.c/h          Spline C implementations
    ni_support.c/h          Support C implementations
    _ni_label.pyx           Connected component labeling (Cython)
    _cytest.pxd/pyx         Cython test module
    _ctest.c                C test module
    _rank_filter_1d.cpp     1-D rank filter (C++)
  utils/
    generate_label_testvectors.py  Test vector generator
  tests/                    9 test files
    test_filters.py, test_interpolation.py, test_morphology.py,
    test_measurements.py, test_fourier.py, test_datatypes.py,
    test_c_api.py, test_ni_support.py, test_splines.py
    data/                   Test data files
```

### File Counts

- Python: 27, Cython (.pyx): 2, Cython (.pxd): 1
- C: 9, C++ (.cpp): 1
- Headers (.h): 8
- Test files: 9

### Cross-Module Dependencies

- Imports from: `scipy._lib`, `scipy.datasets`
- Note: ndimage is relatively self-contained, depending mainly on _lib

### Key Types/Classes

- No major public classes (ndimage is primarily a function-based API)
- Key functions organized by module:
  - _filters.py: correlate, convolve, gaussian_filter, median_filter, uniform_filter, etc.
  - _interpolation.py: map_coordinates, affine_transform, rotate, zoom, shift
  - _morphology.py: binary_erosion, binary_dilation, grey_erosion, grey_dilation, etc.
  - _measurements.py: label, find_objects, sum, mean, variance, center_of_mass

---

## scipy.io

### Module Tree

```
io/
  __init__.py               Public API: loadmat, savemat, mmread, mmwrite, wavfile, etc.
  wavfile.py                WAV file read/write (WavFileWarning, WAVE_FORMAT)
  _mmio.py                  Matrix Market format (MMFile)
  _netcdf.py                NetCDF file format (netcdf_file, netcdf_variable)
  _idl.py                   IDL .sav file reader
  _fortran.py               Fortran unformatted file (FortranFile)
  mmio.py, netcdf.py, idl.py, harwell_boeing.py  Deprecated shims
  matlab/                   MATLAB file I/O
    __init__.py
    _mio.py                 Main loadmat/savemat dispatchers
    _miobase.py             MatFileReader base, MatReadError, MatWriteError
    _mio4.py                MATLAB v4 format (MatFile4Reader, MatFile4Writer)
    _mio5.py                MATLAB v5 format (MatFile5Reader, MatFile5Writer)
    _mio5_params.py         v5 parameters (MatlabObject, MatlabFunction, MatlabOpaque)
    _byteordercodes.py      Byte order handling
    byteordercodes.py, mio.py, mio4.py, mio5.py, mio5_params.py,
    mio5_utils.py, mio_utils.py, miobase.py, streams.py  Deprecated shims
  _harwell_boeing/          Harwell-Boeing sparse matrix format
    __init__.py
    hb.py                   HBFile, HBInfo, HBMatrixType
    _fortran_format_parser.py  Fortran format parser
  _fast_matrix_market/      Fast Matrix Market reader (C++)
    __init__.py
    fast_matrix_market/     C++ library with dependencies (fast_float, ryu)
  arff/                     ARFF format (Weka)
    __init__.py
    _arffread.py            ARFF parser
    arffread.py             Deprecated shim
  tests/                    17 test files
    test_wavfile.py, test_mmio.py, test_netcdf.py, test_idl.py,
    test_fortran.py, test_paths.py
    matlab/tests/           MATLAB-specific tests
```

### File Counts

- Python: 54, Cython (.pyx): 3, Cython (.pxd): 3
- C: 3, C++ (.cpp): 6
- Headers (.h): 12
- Test files: 17

### Cross-Module Dependencies

- Imports from: `scipy._lib`, `scipy.sparse`

### Key Types/Classes

- `MMFile` (_mmio.py:244) - Matrix Market file handler
- `netcdf_file` (_netcdf.py:98) - NetCDF file reader/writer
- `netcdf_variable` (_netcdf.py:811) - NetCDF variable
- `FortranFile` (_fortran.py:33) - Fortran unformatted file reader/writer
- `FortranEOFError` (_fortran.py:13) - Fortran EOF error
- `FortranFormattingError` (_fortran.py:24) - Fortran formatting error
- `WavFileWarning` (wavfile.py:27) - WAV file warning
- `WAVE_FORMAT` (wavfile.py:78) - WAV format enum
- `MatFileReader` (_miobase.py:351) - MATLAB file reader base
- `MatFile4Reader` (_mio4.py:316) - MATLAB v4 reader
- `MatFile4Writer` (_mio4.py:601) - MATLAB v4 writer
- `MatFile5Reader` (_mio5.py:149) - MATLAB v5 reader
- `MatFile5Writer` (_mio5.py:817) - MATLAB v5 writer
- `MatlabObject` (_mio5_params.py:234) - MATLAB object container
- `HBFile` (hb.py:414) - Harwell-Boeing file handler
- `HBInfo` (hb.py:44) - Harwell-Boeing header info

---

## scipy.cluster

### Module Tree

```
cluster/
  __init__.py               Public API (imports hierarchy, vq)
  hierarchy/                Hierarchical clustering submodule
    __init__.py
    _hierarchy_impl.py      linkage, fcluster, dendrogram, ClusterNode, etc.
    _hierarchy.pyx           Hierarchical clustering Cython core
    _optimal_leaf_ordering.pyx  Optimal leaf ordering (Cython)
    _hierarchy_distance_update.pxi  Distance update Cython include
    _structures.pxi         Structure definitions Cython include
    tests/
      test_hierarchy.py, test_disjoint_set.py, hierarchy_test_data.py
  vq/                       Vector quantization submodule
    __init__.py
    _vq_impl.py             kmeans, vq, whiten, ClusterError
    _vq.pyx                 Vector quantization Cython core
    tests/
      test_vq.py
```

### File Counts

- Python: 11, Cython (.pyx): 3, Cython (.pxi): 2
- C/C++: 0
- Test files: 4

### Cross-Module Dependencies

- Imports from: `scipy._lib`, `scipy.sparse`, `scipy.spatial`

### Key Types/Classes

- `ClusterNode` (_hierarchy_impl.py:977) - Node in hierarchical clustering tree
- `ClusterError` (_vq_impl.py:17) - Clustering error
- `ClusterWarning` (_hierarchy_impl.py:66) - Clustering warning

---

## scipy.constants

### Module Tree

```
constants/
  __init__.py               Public API: physical_constants, c, h, k, etc.
  _constants.py             Mathematical and physical constants (c, h, k, e, etc.)
  _codata.py                CODATA fundamental physical constants database
  constants.py, codata.py   Deprecated shims
  tests/
    test_constants.py, test_codata.py
```

### File Counts

- Python: 8, Cython: 0
- C/C++: 0
- Test files: 2

### Cross-Module Dependencies

- Imports from: `scipy._lib`
- Note: constants is a leaf module with no scipy dependencies

### Key Types/Classes

- `ConstantWarning` (_codata.py:2118) - Warning for deprecated constant names

---

## scipy.misc

### Module Tree

```
misc/
  __init__.py               Public API (mostly deprecated)
  common.py                 Deprecated shim
  doccer.py                 Deprecated shim
```

### File Counts

- Python: 3, Cython: 0
- C/C++: 0
- Test files: 0

### Cross-Module Dependencies

- None (misc is essentially deprecated/empty)

### Key Types/Classes

- None

---

## scipy.odr

### Module Tree

```
odr/
  __init__.py               Public API: ODR, Data, Model, Output
  _odrpack.py               Core ODR classes: Data, RealData, Model, Output, ODR
  _models.py                Built-in models: unilinear, multilinear, exponential, quadratic
  _add_newdocs.py           Additional docstrings
  __odrpack.c               ODRPACK C module wrapper
  odrpack.h                 ODRPACK header
  odrpack.py, models.py     Deprecated shims
  odrpack/                  Original ODRPACK Fortran source
    d_lpk.f, d_mprec.f, d_odr.f, d_test.f, dlunoc.f
    Makefile
  tests/
    test_odr.py
```

### File Counts

- Python: 8, Cython: 0
- C: 1, Headers (.h): 1
- Fortran (.f): 5
- Test files: 1

### Cross-Module Dependencies

- Imports from: `scipy._lib`, `scipy.linalg`

### Key Types/Classes

- `ODR` (_odrpack.py:661) - Orthogonal Distance Regression solver
- `Data` (_odrpack.py:202) - ODR input data container
- `RealData` (_odrpack.py:320) - ODR real-valued data with weights
- `Model` (_odrpack.py:461) - ODR model specification
- `Output` (_odrpack.py:578) - ODR result container
- `OdrWarning` (_odrpack.py:53) - ODR warning
- `OdrError` (_odrpack.py:68) - ODR error
- `OdrStop` (_odrpack.py:83) - ODR stop condition

---

## scipy.datasets

### Module Tree

```
datasets/
  __init__.py               Public API: ascent, face, electrocardiogram
  _fetchers.py              Dataset fetcher functions
  _registry.py              Dataset registry (checksums, URLs)
  _download_all.py          Bulk download utility
  _utils.py                 Download utilities
  tests/
    test_data.py
```

### File Counts

- Python: 7, Cython: 0
- C/C++: 0
- Test files: 1

### Cross-Module Dependencies

- Imports from: `scipy._lib`
- Imported by: `scipy.signal`, `scipy.ndimage` (for example data)

### Key Types/Classes

- None (function-based API)

---

## scipy.differentiate

### Module Tree

```
differentiate/
  __init__.py               Public API: derivative, jacobian, hessian
  _differentiate.py         Numerical differentiation: derivative, jacobian, hessian
  tests/
    test_differentiate.py
```

### File Counts

- Python: 4, Cython: 0
- C/C++: 0
- Test files: 1

### Cross-Module Dependencies

- Imports from: `scipy._lib`, `scipy.optimize`

### Key Types/Classes

- None (function-based API)

---

## Cross-Module Dependency Matrix

The following table shows which scipy modules each domain imports from (non-test code only, excluding `scipy._lib` and `scipy._external` which are universal):

| Domain         | linalg | sparse | integrate | optimize | fft | special | stats | signal | spatial | interpolate | ndimage | io | cluster | constants | odr | datasets |
|----------------|:------:|:------:|:---------:|:--------:|:---:|:-------:|:-----:|:------:|:-------:|:-----------:|:-------:|:--:|:-------:|:---------:|:---:|:--------:|
| **linalg**     |   --   |   x    |           |    w     |  w  |    x    |   w   |        |         |             |         |    |         |           |     |          |
| **sparse**     |   x    |   --   |           |          |     |    x    |   w   |        |         |             |         | w  |         |           |     |          |
| **integrate**  |   x    |   x    |    --     |    x     |     |    x    |       |        |         |      x      |         |    |         |           |     |          |
| **optimize**   |   x    |   x    |           |    --    |     |    x    |   w   |        |         |             |         |    |         |           |     |          |
| **fft**        |        |        |           |          | --  |    x    |       |        |         |             |         |    |         |           |     |          |
| **special**    |   x    |        |    w      |    w     |     |    --   |   w   |        |         |      w      |         |    |         |           |     |          |
| **stats**      |   x    |   x    |    x      |    x     |  x  |    x    |  --   |        |    x    |      x      |    x    |    |         |           |     |          |
| **signal**     |   x    |        |    x      |    x     |  x  |    x    |   x   |   --   |    x    |      x      |    x    |    |         |           |     |    x     |
| **spatial**    |   x    |        |           |    x     |     |    x    |   x   |        |   --    |      x      |         |    |         |     x     |     |          |
| **interpolate**|   x    |   x    |    x      |    x     |     |    x    |   x   |        |    x    |     --      |         |    |         |           |     |          |
| **ndimage**    |        |        |           |          |     |         |       |        |         |             |   --    |    |         |           |     |    x     |
| **io**         |        |   x    |           |          |     |         |       |        |         |             |         | -- |         |           |     |          |
| **cluster**    |        |   x    |           |          |     |         |       |        |    x    |             |         |    |   --    |           |     |          |
| **constants**  |        |        |           |          |     |         |       |        |         |             |         |    |         |    --     |     |          |
| **misc**       |        |        |           |          |     |         |       |        |         |             |         |    |         |           |     |          |
| **odr**        |   x    |        |           |          |     |         |       |        |         |             |         |    |         |           | --  |          |
| **datasets**   |        |        |           |          |     |         |       |   w    |         |             |         |    |         |           |     |    --    |
| **differentiate**|      |        |           |    x     |     |         |       |        |         |             |         |    |         |           |     |          |

Legend: `x` = direct import, `w` = weak/optional/indirect import, `--` = self

### Dependency Layers (from bottom to top)

```
Layer 0 (leaves):     constants, misc, datasets
Layer 1 (minimal):    fft, ndimage, differentiate, io, odr
Layer 2 (foundation): linalg, special
Layer 3 (mid-level):  sparse, optimize, integrate, interpolate, cluster
Layer 4 (consumers):  spatial, signal, stats (heaviest)
```

---

## Summary Statistics

| Domain         | Python | Cython | C/C++ | Headers | Fortran | Test Files | Key Classes |
|----------------|-------:|-------:|------:|--------:|--------:|-----------:|------------:|
| linalg         |     64 |      8 |     4 |       8 |       0 |         24 |           8 |
| sparse         |    125 |      8 |   205 |      44 |       0 |         47 |          65 |
| integrate      |     41 |      0 |     9 |       6 |       0 |         12 |          46 |
| optimize       |    132 |     11 |    28 |      19 |       0 |         48 |          85 |
| fft            |     25 |      0 |     1 |       0 |       0 |          8 |           3 |
| special        |     98 |     27 |    12 |      25 |       0 |         57 |          16 |
| stats          |    111 |     12 |    10 |      10 |       0 |         39 |        200+ |
| signal         |     58 |      4 |     7 |       2 |       0 |         21 |          18 |
| spatial        |     32 |      8 |     3 |      11 |       0 |         12 |          10 |
| interpolate    |     43 |      4 |     4 |       2 |       0 |         13 |          33 |
| ndimage        |     27 |      3 |    10 |       8 |       0 |          9 |           0 |
| io             |     54 |      6 |     9 |      12 |       0 |         17 |          45 |
| cluster        |     11 |      5 |     0 |       0 |       0 |          4 |           3 |
| constants      |      8 |      0 |     0 |       0 |       0 |          2 |           1 |
| misc           |      3 |      0 |     0 |       0 |       0 |          0 |           0 |
| odr            |      8 |      0 |     1 |       1 |       5 |          1 |           8 |
| datasets       |      7 |      0 |     0 |       0 |       0 |          1 |           0 |
| differentiate  |      4 |      0 |     0 |       0 |       0 |          1 |           0 |
| **TOTALS**     | **851**| **96** |**303**| **148** |   **5** |    **316** |     **541+**|

### Complexity Rankings (by total source file count, excluding tests)

1. **sparse** (388 files) - Largest due to SuperLU, ARPACK, PROPACK C libraries
2. **optimize** (184 files) - Many algorithms, C libraries (trlib, LBFGSB, SLSQP, MINPACK)
3. **special** (167 files) - Extensive C++/Cython numerical implementations + large test data
4. **stats** (156 files) - 200+ distribution classes, QMC, hypothesis testing
5. **linalg** (107 files) - BLAS/LAPACK wrappers, decompositions, matrix functions
6. **signal** (80 files) - Filters, LTI systems, spectral analysis
7. **interpolate** (56 files) - Splines, RBF, fitpack
8. **io** (78 files) - Multiple format readers/writers
9. **spatial** (70 files) - KDTree, Qhull, distance, transforms
10. **integrate** (64 files) - Quadrature, ODE solvers
11. **ndimage** (48 files) - N-D image processing with C core
12. **fft** (33 files) - Clean Python frontend with PocketFFT backend
13. **cluster** (21 files) - Hierarchical + vector quantization
14. **constants** (10 files) - Pure data module
15. **odr** (18 files) - ODRPACK wrapper
16. **datasets** (9 files) - Data fetchers
17. **misc** (3 files) - Essentially deprecated
18. **differentiate** (6 files) - Newest, smallest module

---

*End of DOC-PASS-01 Module Cartography*

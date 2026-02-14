# SciPy Public Symbol / API Census

**Source**: `/data/projects/frankenscipy/legacy_scipy_code/scipy/scipy/`
**SciPy Version**: 1.17.0 (with Python 3.14.2)
**Generated**: 2026-02-14

---

## Summary Table

| # | Domain | Total | Functions | Classes | Constants | Deprecated (modules) | Warnings/Errors |
|---|--------|-------|-----------|---------|-----------|---------------------|-----------------|
| 1 | `scipy.linalg` | 108 | 85 | 0 | 0 | 10 modules | 2 |
| 2 | `scipy.sparse` | 69 | 24 | 18 | 0 | 14 modules + 2 submodules | 2 |
| 3 | `scipy.sparse.linalg` | 43 | 30 | 4 | 0 | 5 modules | 3 |
| 4 | `scipy.sparse.csgraph` | 24 | 23 | 0 | 0 | 0 | 1 |
| 5 | `scipy.integrate` | 38 | 24 | 9 | 0 | 5 modules | 2 |
| 6 | `scipy.optimize` | 82 | 54 | 12 | 0 | 11 modules | 3 |
| 7 | `scipy.fft` | 40 | 40 | 0 | 0 | 0 | 0 |
| 8 | `scipy.special` | 319 | 295 | 0 | 0 | 6 modules | 2 |
| 9 | `scipy.stats` | 182 | 84 | 15 | 0 | 7 modules + 3 submodules | 4 |
| 10 | `scipy.signal` | 159 | 139 | 8 | 0 | 8 modules + 1 submodule | 1 |
| 11 | `scipy.spatial` | 23 | 8 | 6 | 0 | 3 modules + 2 submodules | 1 |
| 12 | `scipy.interpolate` | 63 | 26 | 27 | 0 | 7 modules | 0 |
| 13 | `scipy.ndimage` | 75 | 75 | 0 | 0 | 5 modules | 0 |
| 14 | `scipy.io` | 20 | 9 | 3 | 0 | 6 modules | 2 |
| 15 | `scipy.cluster` | 2 | 0 | 0 | 0 | 0 submodules | 0 |
| 16 | `scipy.constants` | 166 | 5 | 0 | 157 | 2 modules | 1 |
| 17 | `scipy.misc` | 0 | 0 | 0 | 0 | ENTIRE MODULE DEPRECATED | 0 |
| 18 | `scipy.odr` | 16 | 1 | 8 | 0 | 2 modules | 3 |
| 19 | `scipy.datasets` | 5 | 5 | 0 | 0 | 0 | 0 |
| 20 | `scipy.differentiate` | 3 | 3 | 0 | 0 | 0 | 0 |
| -- | **GRAND TOTAL** | **1437** | | | | | |

**Notes on counting methodology**:
- "Deprecated (modules)" counts re-exported deprecated namespace modules (e.g., `linalg.decomp`, `sparse.bsr`)
- Submodule re-exports (`sparse.csgraph`, `sparse.linalg`, `stats.mstats`, etc.) are counted as symbols
- The 115+ continuous/discrete distribution instances in `scipy.stats` are counted as constants/instances, not classes
- `scipy.misc` is fully deprecated (entire module) with no public API
- `scipy.odr` is deprecated as of 1.17.0 (to be removed in 1.19.0)

---

## 1. scipy.linalg

### Public API (`__all__`)
**Total: 108 symbols** (85 functions, 2 classes/exceptions, 10 deprecated module re-exports, 0 constants)

`__all__` is computed as: `[s for s in dir() if not s.startswith('_')]`

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `bandwidth` | function | `_cythonized_array_utils` | no |
| `block_diag` | function | `_special_matrices` | no |
| `cdf2rdf` | function | `_decomp_schur` | no |
| `cho_factor` | function | `_decomp_cholesky` | no |
| `cho_solve` | function | `_decomp_cholesky` | no |
| `cho_solve_banded` | function | `_decomp_cholesky` | no |
| `cholesky` | function | `_decomp_cholesky` | no |
| `cholesky_banded` | function | `_decomp_cholesky` | no |
| `circulant` | function | `_special_matrices` | no |
| `clarkson_woodruff_transform` | function | `_sketches` | no |
| `companion` | function | `_special_matrices` | no |
| `convolution_matrix` | function | `_special_matrices` | no |
| `coshm` | function | `_matfuncs` | no |
| `cosm` | function | `_matfuncs` | no |
| `cossin` | function | `_decomp_cossin` | no |
| `det` | function | `_basic` | no |
| `dft` | function | `_special_matrices` | no |
| `diagsvd` | function | `_decomp_svd` | no |
| `eig` | function | `_decomp` | no |
| `eig_banded` | function | `_decomp` | no |
| `eigh` | function | `_decomp` | no |
| `eigh_tridiagonal` | function | `_decomp` | no |
| `eigvals` | function | `_decomp` | no |
| `eigvals_banded` | function | `_decomp` | no |
| `eigvalsh` | function | `_decomp` | no |
| `eigvalsh_tridiagonal` | function | `_decomp` | no |
| `expm` | function | `_matfuncs` | no |
| `expm_cond` | function | `_matfuncs` | no |
| `expm_frechet` | function | `_matfuncs` | no |
| `fiedler` | function | `_special_matrices` | no |
| `fiedler_companion` | function | `_special_matrices` | no |
| `find_best_blas_type` | function | `blas` | no |
| `fractional_matrix_power` | function | `_matfuncs` | no |
| `funm` | function | `_matfuncs` | no |
| `get_blas_funcs` | function | `blas` | no |
| `get_lapack_funcs` | function | `lapack` | no |
| `hadamard` | function | `_special_matrices` | no |
| `hankel` | function | `_special_matrices` | no |
| `helmert` | function | `_special_matrices` | no |
| `hessenberg` | function | `_decomp` | no |
| `hilbert` | function | `_special_matrices` | no |
| `inv` | function | `_basic` | no |
| `invhilbert` | function | `_special_matrices` | no |
| `invpascal` | function | `_special_matrices` | no |
| `ishermitian` | function | `_cythonized_array_utils` | no |
| `issymmetric` | function | `_cythonized_array_utils` | no |
| `khatri_rao` | function | `_basic` | no |
| `ldl` | function | `_decomp_ldl` | no |
| `leslie` | function | `_special_matrices` | no |
| `logm` | function | `_matfuncs` | no |
| `lstsq` | function | `_basic` | no |
| `lu` | function | `_decomp_lu` | no |
| `lu_factor` | function | `_decomp_lu` | no |
| `lu_solve` | function | `_decomp_lu` | no |
| `matmul_toeplitz` | function | `_basic` | no |
| `matrix_balance` | function | `_basic` | no |
| `norm` | function | `_misc` | no |
| `null_space` | function | `_decomp_svd` | no |
| `ordqz` | function | `_decomp_qz` | no |
| `orth` | function | `_decomp_svd` | no |
| `orthogonal_procrustes` | function | `_procrustes` | no |
| `pascal` | function | `_special_matrices` | no |
| `pinv` | function | `_basic` | no |
| `pinvh` | function | `_basic` | no |
| `polar` | function | `_decomp_polar` | no |
| `qr` | function | `_decomp_qr` | no |
| `qr_delete` | function | `_decomp_update` | no |
| `qr_insert` | function | `_decomp_update` | no |
| `qr_multiply` | function | `_decomp_qr` | no |
| `qr_update` | function | `_decomp_update` | no |
| `qz` | function | `_decomp_qz` | no |
| `rq` | function | `_decomp_qr` | no |
| `rsf2csf` | function | `_decomp_schur` | no |
| `schur` | function | `_decomp_schur` | no |
| `signm` | function | `_matfuncs` | no |
| `sinhm` | function | `_matfuncs` | no |
| `sinm` | function | `_matfuncs` | no |
| `solve` | function | `_basic` | no |
| `solve_banded` | function | `_basic` | no |
| `solve_circulant` | function | `_basic` | no |
| `solve_continuous_are` | function | `_solvers` | no |
| `solve_continuous_lyapunov` | function | `_solvers` | no |
| `solve_discrete_are` | function | `_solvers` | no |
| `solve_discrete_lyapunov` | function | `_solvers` | no |
| `solve_lyapunov` | function | `_solvers` | no |
| `solve_sylvester` | function | `_solvers` | no |
| `solve_toeplitz` | function | `_basic` | no |
| `solve_triangular` | function | `_basic` | no |
| `solveh_banded` | function | `_basic` | no |
| `sqrtm` | function | `_matfuncs` | no |
| `subspace_angles` | function | `_basic` | no |
| `svd` | function | `_decomp_svd` | no |
| `svdvals` | function | `_decomp_svd` | no |
| `tanhm` | function | `_matfuncs` | no |
| `tanm` | function | `_matfuncs` | no |
| `toeplitz` | function | `_special_matrices` | no |
| `LinAlgError` | exception | `_misc` | no |
| `LinAlgWarning` | warning | `_misc` | no |
| `basic` | module | deprecated namespace | yes (v2.0.0) |
| `decomp` | module | deprecated namespace | yes (v2.0.0) |
| `decomp_cholesky` | module | deprecated namespace | yes (v2.0.0) |
| `decomp_lu` | module | deprecated namespace | yes (v2.0.0) |
| `decomp_qr` | module | deprecated namespace | yes (v2.0.0) |
| `decomp_schur` | module | deprecated namespace | yes (v2.0.0) |
| `decomp_svd` | module | deprecated namespace | yes (v2.0.0) |
| `matfuncs` | module | deprecated namespace | yes (v2.0.0) |
| `misc` | module | deprecated namespace | yes (v2.0.0) |
| `special_matrices` | module | deprecated namespace | yes (v2.0.0) |

---

## 2. scipy.sparse

### Public API (`__all__`)
**Total: 69 symbols** (24 functions, 18 classes, 2 warnings, 14 deprecated module re-exports, 2 submodules, 9 deprecated type-check functions)

`__all__` is computed as: `[s for s in dir() if not s.startswith('_')] + ["csgraph", "linalg"]`

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `bsr_array` | class | `_bsr` | no |
| `bsr_matrix` | class | `_bsr` | no |
| `coo_array` | class | `_coo` | no |
| `coo_matrix` | class | `_coo` | no |
| `csc_array` | class | `_csc` | no |
| `csc_matrix` | class | `_csc` | no |
| `csr_array` | class | `_csr` | no |
| `csr_matrix` | class | `_csr` | no |
| `dia_array` | class | `_dia` | no |
| `dia_matrix` | class | `_dia` | no |
| `dok_array` | class | `_dok` | no |
| `dok_matrix` | class | `_dok` | no |
| `lil_array` | class | `_lil` | no |
| `lil_matrix` | class | `_lil` | no |
| `sparray` | class | `_base` | no |
| `spmatrix` | class | `_matrix` | no |
| `block_array` | function | `_construct` | no |
| `block_diag` | function | `_construct` | no |
| `bmat` | function | `_construct` | no |
| `diags` | function | `_construct` | no |
| `diags_array` | function | `_construct` | no |
| `expand_dims` | function | `_construct` | no |
| `eye` | function | `_construct` | no |
| `eye_array` | function | `_construct` | no |
| `find` | function | `_extract` | no |
| `get_index_dtype` | function | `_sputils` | no |
| `hstack` | function | `_construct` | no |
| `identity` | function | `_construct` | no |
| `issparse` | function | `_base` | no |
| `isspmatrix` | function | `_base` | no |
| `isspmatrix_bsr` | function | `_bsr` | no |
| `isspmatrix_coo` | function | `_coo` | no |
| `isspmatrix_csc` | function | `_csc` | no |
| `isspmatrix_csr` | function | `_csr` | no |
| `isspmatrix_dia` | function | `_dia` | no |
| `isspmatrix_dok` | function | `_dok` | no |
| `isspmatrix_lil` | function | `_lil` | no |
| `kron` | function | `_construct` | no |
| `kronsum` | function | `_construct` | no |
| `load_npz` | function | `_matrix_io` | no |
| `permute_dims` | function | `_construct` | no |
| `rand` | function | `_construct` | no |
| `random` | function | `_construct` | no |
| `random_array` | function | `_construct` | no |
| `safely_cast_index_arrays` | function | `_sputils` | no |
| `save_npz` | function | `_matrix_io` | no |
| `spdiags` | function | `_construct` | no |
| `swapaxes` | function | `_construct` | no |
| `tril` | function | `_extract` | no |
| `triu` | function | `_extract` | no |
| `vstack` | function | `_construct` | no |
| `SparseEfficiencyWarning` | warning | `_base` | no |
| `SparseWarning` | warning | `_base` | no |
| `csgraph` | submodule | (lazy) | no |
| `linalg` | submodule | (lazy) | no |
| `base` | module | deprecated namespace | yes (v2.0.0) |
| `bsr` | module | deprecated namespace | yes (v2.0.0) |
| `compressed` | module | deprecated namespace | yes (v2.0.0) |
| `construct` | module | deprecated namespace | yes (v2.0.0) |
| `coo` | module | deprecated namespace | yes (v2.0.0) |
| `csc` | module | deprecated namespace | yes (v2.0.0) |
| `csr` | module | deprecated namespace | yes (v2.0.0) |
| `data` | module | deprecated namespace | yes (v2.0.0) |
| `dia` | module | deprecated namespace | yes (v2.0.0) |
| `dok` | module | deprecated namespace | yes (v2.0.0) |
| `extract` | module | deprecated namespace | yes (v2.0.0) |
| `lil` | module | deprecated namespace | yes (v2.0.0) |
| `sparsetools` | module | deprecated namespace | yes (v2.0.0) |
| `sputils` | module | deprecated namespace | yes (v2.0.0) |

---

## 3. scipy.sparse.linalg

### Public API (`__all__`)
**Total: 43 symbols** (30 functions, 4 classes, 5 deprecated module re-exports, 3 exceptions/warnings, 1 special array)

`__all__` is computed as: `[s for s in dir() if not s.startswith('_')]`

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `LinearOperator` | class | `_interface` | no |
| `SuperLU` | class | `_dsolve` | no |
| `LaplacianNd` | class | `_special_sparse_arrays` | no |
| `aslinearoperator` | function | `_interface` | no |
| `bicg` | function | `_isolve` | no |
| `bicgstab` | function | `_isolve` | no |
| `cg` | function | `_isolve` | no |
| `cgs` | function | `_isolve` | no |
| `eigs` | function | `_eigen` | no |
| `eigsh` | function | `_eigen` | no |
| `expm` | function | `_matfuncs` | no |
| `expm_multiply` | function | `_expm_multiply` | no |
| `factorized` | function | `_dsolve` | no |
| `funm_multiply_krylov` | function | `_funm_multiply_krylov` | no |
| `gcrotmk` | function | `_isolve` | no |
| `gmres` | function | `_isolve` | no |
| `inv` | function | `_dsolve` | no |
| `is_sptriangular` | function | `_dsolve` | no |
| `lgmres` | function | `_isolve` | no |
| `lobpcg` | function | `_eigen` | no |
| `lsmr` | function | `_isolve` | no |
| `lsqr` | function | `_isolve` | no |
| `matrix_power` | function | `_matfuncs` | no |
| `minres` | function | `_isolve` | no |
| `norm` | function | `_norm` | no |
| `onenormest` | function | `_onenormest` | no |
| `qmr` | function | `_isolve` | no |
| `spbandwidth` | function | `_dsolve` | no |
| `spilu` | function | `_dsolve` | no |
| `splu` | function | `_dsolve` | no |
| `spsolve` | function | `_dsolve` | no |
| `spsolve_triangular` | function | `_dsolve` | no |
| `svds` | function | `_eigen` | no |
| `tfqmr` | function | `_isolve` | no |
| `use_solver` | function | `_dsolve` | no |
| `ArpackError` | exception | `_eigen` | no |
| `ArpackNoConvergence` | exception | `_eigen` | no |
| `MatrixRankWarning` | warning | `_dsolve` | no |
| `dsolve` | module | deprecated namespace | yes (v2.0.0) |
| `eigen` | module | deprecated namespace | yes (v2.0.0) |
| `interface` | module | deprecated namespace | yes (v2.0.0) |
| `isolve` | module | deprecated namespace | yes (v2.0.0) |
| `matfuncs` | module | deprecated namespace | yes (v2.0.0) |

---

## 4. scipy.sparse.csgraph

### Public API (`__all__`)
**Total: 24 symbols** (23 functions, 1 exception)

`__all__` is explicitly defined.

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `bellman_ford` | function | `_shortest_path` | no |
| `breadth_first_order` | function | `_traversal` | no |
| `breadth_first_tree` | function | `_traversal` | no |
| `connected_components` | function | `_traversal` | no |
| `construct_dist_matrix` | function | `_tools` | no |
| `csgraph_from_dense` | function | `_tools` | no |
| `csgraph_from_masked` | function | `_tools` | no |
| `csgraph_masked_from_dense` | function | `_tools` | no |
| `csgraph_to_dense` | function | `_tools` | no |
| `csgraph_to_masked` | function | `_tools` | no |
| `depth_first_order` | function | `_traversal` | no |
| `depth_first_tree` | function | `_traversal` | no |
| `dijkstra` | function | `_shortest_path` | no |
| `floyd_warshall` | function | `_shortest_path` | no |
| `johnson` | function | `_shortest_path` | no |
| `laplacian` | function | `_laplacian` | no |
| `maximum_bipartite_matching` | function | `_matching` | no |
| `maximum_flow` | function | `_flow` | no |
| `min_weight_full_bipartite_matching` | function | `_matching` | no |
| `minimum_spanning_tree` | function | `_min_spanning_tree` | no |
| `reconstruct_path` | function | `_tools` | no |
| `reverse_cuthill_mckee` | function | `_reordering` | no |
| `shortest_path` | function | `_shortest_path` | no |
| `structural_rank` | function | `_reordering` | no |
| `yen` | function | `_shortest_path` | no |
| `NegativeCycleError` | exception | `_shortest_path` | no |

---

## 5. scipy.integrate

### Public API (`__all__`)
**Total: 38 symbols** (24 functions, 9 classes, 2 warnings, 5 deprecated module re-exports)

`__all__` is computed as: `[s for s in dir() if not s.startswith('_')]`

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `BDF` | class | `_ivp` | no |
| `DOP853` | class | `_ivp` | no |
| `DenseOutput` | class | `_ivp` | no |
| `LSODA` | class | `_ivp` | no |
| `OdeSolution` | class | `_ivp` | no |
| `OdeSolver` | class (base) | `_ivp` | no |
| `RK23` | class | `_ivp` | no |
| `RK45` | class | `_ivp` | no |
| `Radau` | class | `_ivp` | no |
| `complex_ode` | class | `_ode` | no |
| `ode` | class | `_ode` | no |
| `cubature` | function | `_cubature` | no |
| `cumulative_simpson` | function | `_quadrature` | no |
| `cumulative_trapezoid` | function | `_quadrature` | no |
| `dblquad` | function | `_quadpack_py` | no |
| `fixed_quad` | function | `_quadrature` | no |
| `lebedev_rule` | function | `_lebedev` | no |
| `newton_cotes` | function | `_quadrature` | no |
| `nquad` | function | `_quadpack_py` | no |
| `nsum` | function | `_tanhsinh` | no |
| `odeint` | function | `_odepack_py` | no |
| `qmc_quad` | function | `_quadrature` | no |
| `quad` | function | `_quadpack_py` | no |
| `quad_vec` | function | `_quad_vec` | no |
| `romb` | function | `_quadrature` | no |
| `simpson` | function | `_quadrature` | no |
| `solve_bvp` | function | `_bvp` | no |
| `solve_ivp` | function | `_ivp` | no |
| `tanhsinh` | function | `_tanhsinh` | no |
| `tplquad` | function | `_quadpack_py` | no |
| `trapezoid` | function | `_quadrature` | no |
| `IntegrationWarning` | warning | `_quadpack_py` | no |
| `ODEintWarning` | warning | `_odepack_py` | no |
| `dop` | module | deprecated namespace | yes (v2.0.0) |
| `lsoda` | module | deprecated namespace | yes (v2.0.0) |
| `odepack` | module | deprecated namespace | yes (v2.0.0) |
| `quadpack` | module | deprecated namespace | yes (v2.0.0) |
| `vode` | module | deprecated namespace | yes (v2.0.0) |

---

## 6. scipy.optimize

### Public API (`__all__`)
**Total: 82 symbols** (54 functions, 12 classes, 3 warnings/exceptions, 11 deprecated module re-exports, 2 result classes)

`__all__` is computed as: `[s for s in dir() if not s.startswith('_')]`

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `BFGS` | class | `_hessian_update_strategy` | no |
| `Bounds` | class | `_constraints` | no |
| `BroydenFirst` | class | `_nonlin` | no |
| `HessianUpdateStrategy` | class (interface) | `_hessian_update_strategy` | no |
| `InverseJacobian` | class | `_nonlin` | no |
| `KrylovJacobian` | class | `_nonlin` | no |
| `LbfgsInvHessProduct` | class | `_lbfgsb_py` | no |
| `LinearConstraint` | class | `_constraints` | no |
| `NonlinearConstraint` | class | `_constraints` | no |
| `SR1` | class | `_hessian_update_strategy` | no |
| `OptimizeResult` | class | `_optimize` | no |
| `RootResults` | class | `_zeros_py` | no |
| `anderson` | function | `_nonlin` | no |
| `approx_fprime` | function | `_optimize` | no |
| `basinhopping` | function | `_basinhopping` | no |
| `bisect` | function | `_zeros_py` | no |
| `bracket` | function | `_optimize` | no |
| `brent` | function | `_optimize` | no |
| `brenth` | function | `_zeros_py` | no |
| `brentq` | function | `_zeros_py` | no |
| `broyden1` | function | `_nonlin` | no |
| `broyden2` | function | `_nonlin` | no |
| `brute` | function | `_optimize` | no |
| `check_grad` | function | `_optimize` | no |
| `curve_fit` | function | `_minpack_py` | no |
| `diagbroyden` | function | `_nonlin` | no |
| `differential_evolution` | function | `_differentialevolution` | no |
| `direct` | function | `_direct_py` | no |
| `dual_annealing` | function | `_dual_annealing` | no |
| `excitingmixing` | function | `_nonlin` | no |
| `fixed_point` | function | `_minpack_py` | no |
| `fmin` | function | `_optimize` | legacy |
| `fmin_bfgs` | function | `_optimize` | legacy |
| `fmin_cg` | function | `_optimize` | legacy |
| `fmin_cobyla` | function | `_cobyla_py` | legacy |
| `fmin_l_bfgs_b` | function | `_lbfgsb_py` | legacy |
| `fmin_ncg` | function | `_optimize` | legacy |
| `fmin_powell` | function | `_optimize` | legacy |
| `fmin_slsqp` | function | `_slsqp_py` | legacy |
| `fmin_tnc` | function | `_tnc` | legacy |
| `fminbound` | function | `_optimize` | legacy |
| `fsolve` | function | `_minpack_py` | legacy |
| `golden` | function | `_optimize` | legacy |
| `isotonic_regression` | function | `_isotonic` | no |
| `least_squares` | function | `_lsq` | no |
| `leastsq` | function | `_minpack_py` | legacy |
| `line_search` | function | `_optimize` | no |
| `linear_sum_assignment` | function | `_lsap` | no |
| `linearmixing` | function | `_nonlin` | no |
| `linprog` | function | `_linprog` | no |
| `linprog_verbose_callback` | function | `_linprog` | no |
| `lsq_linear` | function | `_lsq` | no |
| `milp` | function | `_milp` | no |
| `minimize` | function | `_minimize` | no |
| `minimize_scalar` | function | `_minimize` | no |
| `newton` | function | `_zeros_py` | no |
| `newton_krylov` | function | `_nonlin` | no |
| `nnls` | function | `_nnls` | no |
| `quadratic_assignment` | function | `_qap` | no |
| `ridder` | function | `_zeros_py` | no |
| `root` | function | `_root` | no |
| `root_scalar` | function | `_root_scalar` | no |
| `rosen` | function | `_optimize` | no |
| `rosen_der` | function | `_optimize` | no |
| `rosen_hess` | function | `_optimize` | no |
| `rosen_hess_prod` | function | `_optimize` | no |
| `shgo` | function | `_shgo` | no |
| `show_options` | function | `_optimize` | no |
| `toms748` | function | `_zeros_py` | no |
| `NoConvergence` | exception | `_nonlin` | no |
| `OptimizeWarning` | warning | `_optimize` | no |
| `cobyla` | module | deprecated namespace | yes (v2.0.0) |
| `lbfgsb` | module | deprecated namespace | yes (v2.0.0) |
| `linesearch` | module | deprecated namespace | yes (v2.0.0) |
| `minpack` | module | deprecated namespace | yes (v2.0.0) |
| `minpack2` | module | deprecated namespace | yes (v2.0.0) |
| `moduleTNC` | module | deprecated namespace | yes (v2.0.0) |
| `nonlin` | module | deprecated namespace | yes (v2.0.0) |
| `optimize` | module | deprecated namespace | yes (v2.0.0) |
| `slsqp` | module | deprecated namespace | yes (v2.0.0) |
| `tnc` | module | deprecated namespace | yes (v2.0.0) |
| `zeros` | module | deprecated namespace | yes (v2.0.0) |

---

## 7. scipy.fft

### Public API (`__all__`)
**Total: 40 symbols** (40 functions)

`__all__` is explicitly defined.

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `fft` | function | `_basic` | no |
| `ifft` | function | `_basic` | no |
| `fft2` | function | `_basic` | no |
| `ifft2` | function | `_basic` | no |
| `fftn` | function | `_basic` | no |
| `ifftn` | function | `_basic` | no |
| `rfft` | function | `_basic` | no |
| `irfft` | function | `_basic` | no |
| `rfft2` | function | `_basic` | no |
| `irfft2` | function | `_basic` | no |
| `rfftn` | function | `_basic` | no |
| `irfftn` | function | `_basic` | no |
| `hfft` | function | `_basic` | no |
| `ihfft` | function | `_basic` | no |
| `hfft2` | function | `_basic` | no |
| `ihfft2` | function | `_basic` | no |
| `hfftn` | function | `_basic` | no |
| `ihfftn` | function | `_basic` | no |
| `dct` | function | `_realtransforms` | no |
| `idct` | function | `_realtransforms` | no |
| `dctn` | function | `_realtransforms` | no |
| `idctn` | function | `_realtransforms` | no |
| `dst` | function | `_realtransforms` | no |
| `idst` | function | `_realtransforms` | no |
| `dstn` | function | `_realtransforms` | no |
| `idstn` | function | `_realtransforms` | no |
| `fht` | function | `_fftlog` | no |
| `ifht` | function | `_fftlog` | no |
| `fhtoffset` | function | `_fftlog` | no |
| `fftfreq` | function | `_helper` | no |
| `rfftfreq` | function | `_helper` | no |
| `fftshift` | function | `_helper` | no |
| `ifftshift` | function | `_helper` | no |
| `next_fast_len` | function | `_helper` | no |
| `prev_fast_len` | function | `_helper` | no |
| `set_workers` | function | `_pocketfft.helper` | no |
| `get_workers` | function | `_pocketfft.helper` | no |
| `set_backend` | function | `_backend` | no |
| `skip_backend` | function | `_backend` | no |
| `set_global_backend` | function | `_backend` | no |
| `register_backend` | function | `_backend` | no |

---

## 8. scipy.special

### Public API (`__all__`)
**Total: 319 symbols** (295 functions, 6 deprecated module re-exports, 2 warnings/exceptions, plus error-handling functions)

`__all__` is computed as: `_ufuncs.__all__ + _basic.__all__ + _orthogonal.__all__ + _multiufuncs.__all__ + [explicit additions]`

**Airy functions (5)**

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `airy` | function (ufunc) | `_ufuncs` | no |
| `airye` | function (ufunc) | `_ufuncs` | no |
| `ai_zeros` | function | `_basic` | no |
| `bi_zeros` | function | `_basic` | no |
| `itairy` | function (ufunc) | `_ufuncs` | no |

**Elliptic functions (11)**

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `ellipj` | function (ufunc) | `_ufuncs` | no |
| `ellipk` | function (ufunc) | `_ufuncs` | no |
| `ellipkm1` | function (ufunc) | `_ufuncs` | no |
| `ellipkinc` | function (ufunc) | `_ufuncs` | no |
| `ellipe` | function (ufunc) | `_ufuncs` | no |
| `ellipeinc` | function (ufunc) | `_ufuncs` | no |
| `elliprc` | function (ufunc) | `_ufuncs` | no |
| `elliprd` | function (ufunc) | `_ufuncs` | no |
| `elliprf` | function (ufunc) | `_ufuncs` | no |
| `elliprg` | function (ufunc) | `_ufuncs` | no |
| `elliprj` | function (ufunc) | `_ufuncs` | no |

**Bessel functions (48)**

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `jv` | function (ufunc) | `_ufuncs` | no |
| `jve` | function (ufunc) | `_ufuncs` | no |
| `jn` | function | `_basic` | no |
| `yn` | function (ufunc) | `_ufuncs` | no |
| `yv` | function (ufunc) | `_ufuncs` | no |
| `yve` | function (ufunc) | `_ufuncs` | no |
| `iv` | function (ufunc) | `_ufuncs` | no |
| `ive` | function (ufunc) | `_ufuncs` | no |
| `kn` | function (ufunc) | `_ufuncs` | no |
| `kv` | function (ufunc) | `_ufuncs` | no |
| `kve` | function (ufunc) | `_ufuncs` | no |
| `hankel1` | function (ufunc) | `_ufuncs` | no |
| `hankel1e` | function (ufunc) | `_ufuncs` | no |
| `hankel2` | function (ufunc) | `_ufuncs` | no |
| `hankel2e` | function (ufunc) | `_ufuncs` | no |
| `wright_bessel` | function (ufunc) | `_ufuncs` | no |
| `log_wright_bessel` | function (ufunc) | `_ufuncs` | no |
| `lmbda` | function | `_basic` | no |
| `jnjnp_zeros` | function | `_basic` | no |
| `jnyn_zeros` | function | `_basic` | no |
| `jn_zeros` | function | `_basic` | no |
| `jnp_zeros` | function | `_basic` | no |
| `yn_zeros` | function | `_basic` | no |
| `ynp_zeros` | function | `_basic` | no |
| `y0_zeros` | function | `_basic` | no |
| `y1_zeros` | function | `_basic` | no |
| `y1p_zeros` | function | `_basic` | no |
| `j0` | function (ufunc) | `_ufuncs` | no |
| `j1` | function (ufunc) | `_ufuncs` | no |
| `y0` | function (ufunc) | `_ufuncs` | no |
| `y1` | function (ufunc) | `_ufuncs` | no |
| `i0` | function (ufunc) | `_ufuncs` | no |
| `i0e` | function (ufunc) | `_ufuncs` | no |
| `i1` | function (ufunc) | `_ufuncs` | no |
| `i1e` | function (ufunc) | `_ufuncs` | no |
| `k0` | function (ufunc) | `_ufuncs` | no |
| `k0e` | function (ufunc) | `_ufuncs` | no |
| `k1` | function (ufunc) | `_ufuncs` | no |
| `k1e` | function (ufunc) | `_ufuncs` | no |
| `itj0y0` | function (ufunc) | `_ufuncs` | no |
| `it2j0y0` | function (ufunc) | `_ufuncs` | no |
| `iti0k0` | function (ufunc) | `_ufuncs` | no |
| `it2i0k0` | function (ufunc) | `_ufuncs` | no |
| `besselpoly` | function (ufunc) | `_ufuncs` | no |
| `jvp` | function | `_basic` | no |
| `yvp` | function | `_basic` | no |
| `ivp` | function | `_basic` | no |
| `kvp` | function | `_basic` | no |
| `h1vp` | function | `_basic` | no |
| `h2vp` | function | `_basic` | no |
| `spherical_jn` | function | `_spherical_bessel` | no |
| `spherical_yn` | function | `_spherical_bessel` | no |
| `spherical_in` | function | `_spherical_bessel` | no |
| `spherical_kn` | function | `_spherical_bessel` | no |
| `riccati_jn` | function | `_basic` | no |
| `riccati_yn` | function | `_basic` | no |

**Struve functions (5)**

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `struve` | function (ufunc) | `_ufuncs` | no |
| `modstruve` | function (ufunc) | `_ufuncs` | no |
| `itstruve0` | function (ufunc) | `_ufuncs` | no |
| `it2struve0` | function (ufunc) | `_ufuncs` | no |
| `itmodstruve0` | function (ufunc) | `_ufuncs` | no |

**Statistical distribution functions (62)**

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `bdtr` | function (ufunc) | `_ufuncs` | no |
| `bdtrc` | function (ufunc) | `_ufuncs` | no |
| `bdtri` | function (ufunc) | `_ufuncs` | no |
| `bdtrik` | function (ufunc) | `_ufuncs` | no |
| `bdtrin` | function (ufunc) | `_ufuncs` | no |
| `btdtria` | function (ufunc) | `_ufuncs` | no |
| `btdtrib` | function (ufunc) | `_ufuncs` | no |
| `fdtr` | function (ufunc) | `_ufuncs` | no |
| `fdtrc` | function (ufunc) | `_ufuncs` | no |
| `fdtri` | function (ufunc) | `_ufuncs` | no |
| `fdtridfd` | function (ufunc) | `_ufuncs` | no |
| `gdtr` | function (ufunc) | `_ufuncs` | no |
| `gdtrc` | function (ufunc) | `_ufuncs` | no |
| `gdtria` | function (ufunc) | `_ufuncs` | no |
| `gdtrib` | function (ufunc) | `_ufuncs` | no |
| `gdtrix` | function (ufunc) | `_ufuncs` | no |
| `nbdtr` | function (ufunc) | `_ufuncs` | no |
| `nbdtrc` | function (ufunc) | `_ufuncs` | no |
| `nbdtri` | function (ufunc) | `_ufuncs` | no |
| `nbdtrik` | function (ufunc) | `_ufuncs` | no |
| `nbdtrin` | function (ufunc) | `_ufuncs` | no |
| `ncfdtr` | function (ufunc) | `_ufuncs` | no |
| `ncfdtri` | function (ufunc) | `_ufuncs` | no |
| `ncfdtridfd` | function (ufunc) | `_ufuncs` | no |
| `ncfdtridfn` | function (ufunc) | `_ufuncs` | no |
| `ncfdtrinc` | function (ufunc) | `_ufuncs` | no |
| `nctdtr` | function (ufunc) | `_ufuncs` | no |
| `nctdtridf` | function (ufunc) | `_ufuncs` | no |
| `nctdtrinc` | function (ufunc) | `_ufuncs` | no |
| `nctdtrit` | function (ufunc) | `_ufuncs` | no |
| `nrdtrimn` | function (ufunc) | `_ufuncs` | no |
| `nrdtrisd` | function (ufunc) | `_ufuncs` | no |
| `ndtr` | function (ufunc) | `_ufuncs` | no |
| `log_ndtr` | function (ufunc) | `_ufuncs` | no |
| `ndtri` | function (ufunc) | `_ufuncs` | no |
| `ndtri_exp` | function (ufunc) | `_ufuncs` | no |
| `pdtr` | function (ufunc) | `_ufuncs` | no |
| `pdtrc` | function (ufunc) | `_ufuncs` | no |
| `pdtri` | function (ufunc) | `_ufuncs` | no |
| `pdtrik` | function (ufunc) | `_ufuncs` | no |
| `stdtr` | function (ufunc) | `_ufuncs` | no |
| `stdtridf` | function (ufunc) | `_ufuncs` | no |
| `stdtrit` | function (ufunc) | `_ufuncs` | no |
| `chdtr` | function (ufunc) | `_ufuncs` | no |
| `chdtrc` | function (ufunc) | `_ufuncs` | no |
| `chdtri` | function (ufunc) | `_ufuncs` | no |
| `chdtriv` | function (ufunc) | `_ufuncs` | no |
| `chndtr` | function (ufunc) | `_ufuncs` | no |
| `chndtridf` | function (ufunc) | `_ufuncs` | no |
| `chndtrinc` | function (ufunc) | `_ufuncs` | no |
| `chndtrix` | function (ufunc) | `_ufuncs` | no |
| `smirnov` | function (ufunc) | `_ufuncs` | no |
| `smirnovi` | function (ufunc) | `_ufuncs` | no |
| `kolmogorov` | function (ufunc) | `_ufuncs` | no |
| `kolmogi` | function (ufunc) | `_ufuncs` | no |
| `boxcox` | function (ufunc) | `_ufuncs` | no |
| `boxcox1p` | function (ufunc) | `_ufuncs` | no |
| `inv_boxcox` | function (ufunc) | `_ufuncs` | no |
| `inv_boxcox1p` | function (ufunc) | `_ufuncs` | no |
| `logit` | function (ufunc) | `_ufuncs` | no |
| `expit` | function (ufunc) | `_ufuncs` | no |
| `log_expit` | function (ufunc) | `_ufuncs` | no |
| `tklmbda` | function (ufunc) | `_ufuncs` | no |
| `owens_t` | function (ufunc) | `_ufuncs` | no |

**Information theory (5)**

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `entr` | function (ufunc) | `_ufuncs` | no |
| `rel_entr` | function (ufunc) | `_ufuncs` | no |
| `kl_div` | function (ufunc) | `_ufuncs` | no |
| `huber` | function (ufunc) | `_ufuncs` | no |
| `pseudo_huber` | function (ufunc) | `_ufuncs` | no |

**Gamma and related (22)**

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `gamma` | function (ufunc) | `_ufuncs` | no |
| `gammaln` | function (ufunc) | `_ufuncs` | no |
| `loggamma` | function (ufunc) | `_ufuncs` | no |
| `gammasgn` | function (ufunc) | `_ufuncs` | no |
| `gammainc` | function (ufunc) | `_ufuncs` | no |
| `gammaincinv` | function (ufunc) | `_ufuncs` | no |
| `gammaincc` | function (ufunc) | `_ufuncs` | no |
| `gammainccinv` | function (ufunc) | `_ufuncs` | no |
| `beta` | function (ufunc) | `_ufuncs` | no |
| `betaln` | function (ufunc) | `_ufuncs` | no |
| `betainc` | function (ufunc) | `_ufuncs` | no |
| `betaincc` | function (ufunc) | `_ufuncs` | no |
| `betaincinv` | function (ufunc) | `_ufuncs` | no |
| `betainccinv` | function (ufunc) | `_ufuncs` | no |
| `psi` | function (ufunc) | `_ufuncs` | no |
| `digamma` | function (ufunc) | `_ufuncs` | no |
| `rgamma` | function (ufunc) | `_ufuncs` | no |
| `polygamma` | function | `_basic` | no |
| `multigammaln` | function | `_basic` | no |
| `poch` | function (ufunc) | `_ufuncs` | no |

**Error function and Fresnel (16)**

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `erf` | function (ufunc) | `_ufuncs` | no |
| `erfc` | function (ufunc) | `_ufuncs` | no |
| `erfcx` | function (ufunc) | `_ufuncs` | no |
| `erfi` | function (ufunc) | `_ufuncs` | no |
| `erfinv` | function (ufunc) | `_ufuncs` | no |
| `erfcinv` | function (ufunc) | `_ufuncs` | no |
| `wofz` | function (ufunc) | `_ufuncs` | no |
| `dawsn` | function (ufunc) | `_ufuncs` | no |
| `fresnel` | function (ufunc) | `_ufuncs` | no |
| `fresnel_zeros` | function | `_basic` | no |
| `modfresnelp` | function (ufunc) | `_ufuncs` | no |
| `modfresnelm` | function (ufunc) | `_ufuncs` | no |
| `voigt_profile` | function (ufunc) | `_ufuncs` | no |
| `erf_zeros` | function | `_basic` | no |
| `fresnelc_zeros` | function | `_basic` | no |
| `fresnels_zeros` | function | `_basic` | no |

**Legendre functions (11)**

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `legendre_p` | function | `_multiufuncs` | no |
| `legendre_p_all` | function | `_multiufuncs` | no |
| `assoc_legendre_p` | function | `_multiufuncs` | no |
| `assoc_legendre_p_all` | function | `_multiufuncs` | no |
| `sph_legendre_p` | function | `_multiufuncs` | no |
| `sph_legendre_p_all` | function | `_multiufuncs` | no |
| `sph_harm_y` | function | `_multiufuncs` | no |
| `sph_harm_y_all` | function | `_multiufuncs` | no |
| `lpmv` | function (ufunc) | `_ufuncs` | no |
| `lqn` | function | `_basic` | no |
| `lqmn` | function | `_basic` | no |

**Ellipsoidal harmonics (3)**

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `ellip_harm` | function | `_ellip_harm` | no |
| `ellip_harm_2` | function | `_ellip_harm` | no |
| `ellip_normal` | function | `_ellip_harm` | no |

**Orthogonal polynomials (eval_* family: 14, roots_* family: 14, coefficient family: 14)**

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `eval_legendre` | function (ufunc) | `_ufuncs` | no |
| `eval_chebyt` | function (ufunc) | `_ufuncs` | no |
| `eval_chebyu` | function (ufunc) | `_ufuncs` | no |
| `eval_chebyc` | function (ufunc) | `_ufuncs` | no |
| `eval_chebys` | function (ufunc) | `_ufuncs` | no |
| `eval_jacobi` | function (ufunc) | `_ufuncs` | no |
| `eval_laguerre` | function (ufunc) | `_ufuncs` | no |
| `eval_genlaguerre` | function (ufunc) | `_ufuncs` | no |
| `eval_hermite` | function (ufunc) | `_ufuncs` | no |
| `eval_hermitenorm` | function (ufunc) | `_ufuncs` | no |
| `eval_gegenbauer` | function (ufunc) | `_ufuncs` | no |
| `eval_sh_legendre` | function (ufunc) | `_ufuncs` | no |
| `eval_sh_chebyt` | function (ufunc) | `_ufuncs` | no |
| `eval_sh_chebyu` | function (ufunc) | `_ufuncs` | no |
| `eval_sh_jacobi` | function (ufunc) | `_ufuncs` | no |
| `assoc_laguerre` | function | `_basic` | no |
| `roots_legendre` | function | `_basic` | no |
| `roots_chebyt` | function | `_basic` | no |
| `roots_chebyu` | function | `_basic` | no |
| `roots_chebyc` | function | `_basic` | no |
| `roots_chebys` | function | `_basic` | no |
| `roots_jacobi` | function | `_basic` | no |
| `roots_laguerre` | function | `_basic` | no |
| `roots_genlaguerre` | function | `_basic` | no |
| `roots_hermite` | function | `_basic` | no |
| `roots_hermitenorm` | function | `_basic` | no |
| `roots_gegenbauer` | function | `_basic` | no |
| `roots_sh_legendre` | function | `_basic` | no |
| `roots_sh_chebyt` | function | `_basic` | no |
| `roots_sh_chebyu` | function | `_basic` | no |
| `roots_sh_jacobi` | function | `_basic` | no |

_(Coefficient-returning orthogonal polynomial functions from `_orthogonal` not enumerated individually as they are dynamically registered -- includes `legendre`, `chebyt`, `chebyu`, `chebyc`, `chebys`, `jacobi`, `laguerre`, `genlaguerre`, `hermite`, `hermitenorm`, `gegenbauer`, `sh_legendre`, `sh_chebyt`, `sh_chebyu`, `sh_jacobi`)_

**Hypergeometric (4), Parabolic cylinder (6), Mathieu (10), Spheroidal wave (16), Kelvin (18), Combinatorics (3), Lambert/Wright (2)**

_(See full symbol list above -- all are functions/ufuncs from `_ufuncs` or `_basic`)_

**Convenience functions (18)**

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `cbrt` | function (ufunc) | `_ufuncs` | no |
| `exp10` | function (ufunc) | `_ufuncs` | no |
| `exp2` | function (ufunc) | `_ufuncs` | no |
| `radian` | function (ufunc) | `_ufuncs` | no |
| `cosdg` | function (ufunc) | `_ufuncs` | no |
| `sindg` | function (ufunc) | `_ufuncs` | no |
| `tandg` | function (ufunc) | `_ufuncs` | no |
| `cotdg` | function (ufunc) | `_ufuncs` | no |
| `log1p` | function (ufunc) | `_ufuncs` | no |
| `expm1` | function (ufunc) | `_ufuncs` | no |
| `cosm1` | function (ufunc) | `_ufuncs` | no |
| `powm1` | function (ufunc) | `_ufuncs` | no |
| `round` | function (ufunc) | `_ufuncs` | no |
| `xlogy` | function (ufunc) | `_ufuncs` | no |
| `xlog1py` | function (ufunc) | `_ufuncs` | no |
| `logsumexp` | function | `_logsumexp` | no |
| `exprel` | function (ufunc) | `_ufuncs` | no |
| `sinc` | function | `_basic` | no |
| `softmax` | function | `_logsumexp` | no |
| `log_softmax` | function | `_logsumexp` | no |
| `softplus` | function (ufunc) | `_ufuncs` | no |

**Error handling (5)**

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `geterr` | function | `_sf_error` | no |
| `seterr` | function | `_sf_error` | no |
| `errstate` | class (context mgr) | `_sf_error` | no |
| `SpecialFunctionWarning` | warning | `_sf_error` | no |
| `SpecialFunctionError` | exception | `_sf_error` | no |

**Deprecated module re-exports (6)**

| Symbol | Type | Deprecated? |
|--------|------|-------------|
| `add_newdocs` | module | yes (v2.0.0) |
| `basic` | module | yes (v2.0.0) |
| `orthogonal` | module | yes (v2.0.0) |
| `specfun` | module | yes (v2.0.0) |
| `sf_error` | module | yes (v2.0.0) |
| `spfun_stats` | module | yes (v2.0.0) |

---

## 9. scipy.stats

### Public API (`__all__`)
**Total: 182 symbols** (84 functions, 15 classes, ~115 distribution instances, 4 warnings/exceptions, 7 deprecated module re-exports, 3 submodule re-exports)

`__all__` is computed as: `[s for s in dir() if not s.startswith("_")]`

**Distribution base classes (3)**

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `rv_continuous` | class | `distributions` | no |
| `rv_discrete` | class | `distributions` | no |
| `rv_histogram` | class | `distributions` | no |

**Continuous distributions (~110 instances, named in `_continuous_distns._distn_names`)**

`alpha`, `anglit`, `arcsine`, `argus`, `beta`, `betaprime`, `bradford`, `burr`, `burr12`, `cauchy`, `chi`, `chi2`, `cosine`, `crystalball`, `dgamma`, `dpareto_lognorm`, `dweibull`, `erlang`, `expon`, `exponnorm`, `exponweib`, `exponpow`, `f`, `fatiguelife`, `fisk`, `foldcauchy`, `foldnorm`, `genlogistic`, `gennorm`, `genpareto`, `genexpon`, `genextreme`, `gausshyper`, `gamma`, `gengamma`, `genhalflogistic`, `genhyperbolic`, `geninvgauss`, `gibrat`, `gompertz`, `gumbel_r`, `gumbel_l`, `halfcauchy`, `halflogistic`, `halfnorm`, `halfgennorm`, `hypsecant`, `invgamma`, `invgauss`, `invweibull`, `irwinhall`, `jf_skew_t`, `johnsonsb`, `johnsonsu`, `kappa4`, `kappa3`, `ksone`, `kstwo`, `kstwobign`, `landau`, `laplace`, `laplace_asymmetric`, `levy`, `levy_l`, `levy_stable`, `logistic`, `loggamma`, `loglaplace`, `lognorm`, `loguniform`, `lomax`, `maxwell`, `mielke`, `moyal`, `nakagami`, `ncx2`, `ncf`, `nct`, `norm`, `norminvgauss`, `pareto`, `pearson3`, `powerlaw`, `powerlognorm`, `powernorm`, `rdist`, `rayleigh`, `rel_breitwigner`, `rice`, `recipinvgauss`, `semicircular`, `skewcauchy`, `skewnorm`, `studentized_range`, `t`, `trapezoid`, `triang`, `truncexpon`, `truncnorm`, `truncpareto`, `truncweibull_min`, `tukeylambda`, `uniform`, `vonmises`, `vonmises_line`, `wald`, `weibull_min`, `weibull_max`, `wrapcauchy`

**Discrete distributions (~22 instances, named in `_discrete_distns._distn_names`)**

`bernoulli`, `betabinom`, `betanbinom`, `binom`, `boltzmann`, `dlaplace`, `geom`, `hypergeom`, `logser`, `nbinom`, `nchypergeom_fisher`, `nchypergeom_wallenius`, `nhypergeom`, `planck`, `poisson`, `poisson_binom`, `randint`, `skellam`, `yulesimon`, `zipf`, `zipfian`

**Multivariate distributions (17 instances)**

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `multivariate_normal` | instance | `_multivariate` | no |
| `matrix_normal` | instance | `_multivariate` | no |
| `dirichlet` | instance | `_multivariate` | no |
| `dirichlet_multinomial` | instance | `_multivariate` | no |
| `wishart` | instance | `_multivariate` | no |
| `invwishart` | instance | `_multivariate` | no |
| `multinomial` | instance | `_multivariate` | no |
| `special_ortho_group` | instance | `_multivariate` | no |
| `ortho_group` | instance | `_multivariate` | no |
| `unitary_group` | instance | `_multivariate` | no |
| `random_correlation` | instance | `_multivariate` | no |
| `multivariate_t` | instance | `_multivariate` | no |
| `multivariate_hypergeom` | instance | `_multivariate` | no |
| `normal_inverse_gamma` | instance | `_multivariate` | no |
| `random_table` | instance | `_multivariate` | no |
| `uniform_direction` | instance | `_multivariate` | no |
| `vonmises_fisher` | instance | `_multivariate` | no |
| `matrix_t` | instance | `_multivariate` | no |

**New distribution infrastructure (10)**

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `make_distribution` | function | `_distribution_infrastructure` | no |
| `Mixture` | class | `_distribution_infrastructure` | no |
| `order_statistic` | function | `_distribution_infrastructure` | no |
| `truncate` | function | `_distribution_infrastructure` | no |
| `exp` | function | `_distribution_infrastructure` | no |
| `log` | function | `_distribution_infrastructure` | no |
| `abs` | function | `_distribution_infrastructure` | no |
| `Normal` | class | `_new_distributions` | no |
| `Logistic` | class | `_new_distributions` | no |
| `Uniform` | class | `_new_distributions` | no |
| `Binomial` | class | `_new_distributions` | no |

**Statistical test functions (48)**

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `ttest_1samp` | function | `_stats_py` | no |
| `ttest_ind` | function | `_stats_py` | no |
| `ttest_ind_from_stats` | function | `_stats_py` | no |
| `ttest_rel` | function | `_stats_py` | no |
| `binomtest` | function | `_binomtest` | no |
| `quantile_test` | function | `_hypotests` | no |
| `skewtest` | function | `_stats_py` | no |
| `kurtosistest` | function | `_stats_py` | no |
| `normaltest` | function | `_stats_py` | no |
| `jarque_bera` | function | `_stats_py` | no |
| `shapiro` | function | `_morestats` | no |
| `anderson` | function | `_morestats` | no |
| `anderson_ksamp` | function | `_morestats` | no |
| `cramervonmises` | function | `_hypotests` | no |
| `cramervonmises_2samp` | function | `_hypotests` | no |
| `ks_1samp` | function | `_stats_py` | no |
| `ks_2samp` | function | `_stats_py` | no |
| `kstest` | function | `_stats_py` | no |
| `goodness_of_fit` | function | `_fit` | no |
| `chisquare` | function | `_stats_py` | no |
| `power_divergence` | function | `_stats_py` | no |
| `wilcoxon` | function | `_morestats` | no |
| `mannwhitneyu` | function | `_mannwhitneyu` | no |
| `bws_test` | function | `_bws_test` | no |
| `ranksums` | function | `_stats_py` | no |
| `brunnermunzel` | function | `_stats_py` | no |
| `mood` | function | `_morestats` | no |
| `ansari` | function | `_morestats` | no |
| `epps_singleton_2samp` | function | `_hypotests` | no |
| `f_oneway` | function | `_stats_py` | no |
| `tukey_hsd` | function | `_multicomp` | no |
| `dunnett` | function | `_multicomp` | no |
| `kruskal` | function | `_stats_py` | no |
| `alexandergovern` | function | `_stats_py` | no |
| `fligner` | function | `_morestats` | no |
| `levene` | function | `_morestats` | no |
| `bartlett` | function | `_stats_py` | no |
| `median_test` | function | `_morestats` | no |
| `friedmanchisquare` | function | `_stats_py` | no |
| `fisher_exact` | function | `_stats_py` | no |
| `barnard_exact` | function | `_hypotests` | no |
| `boschloo_exact` | function | `_hypotests` | no |
| `chi2_contingency` | function | `contingency` | no |
| `poisson_means_test` | function | `_hypotests` | no |
| `page_trend_test` | function | `_page_trend_test` | no |
| `multiscale_graphcorr` | function | `_mgc` | no |
| `combine_pvalues` | function | `_stats_py` | no |
| `false_discovery_control` | function | `_stats_py` | no |

**Resampling methods (7)**

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `bootstrap` | function | `_resampling` | no |
| `monte_carlo_test` | function | `_resampling` | no |
| `permutation_test` | function | `_resampling` | no |
| `power` | function | `_resampling` | no |
| `MonteCarloMethod` | class | `_resampling` | no |
| `PermutationMethod` | class | `_resampling` | no |
| `BootstrapMethod` | class | `_resampling` | no |

**Summary statistics (27)**

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `describe` | function | `_stats_py` | no |
| `gmean` | function | `_stats_py` | no |
| `hmean` | function | `_stats_py` | no |
| `pmean` | function | `_stats_py` | no |
| `kurtosis` | function | `_stats_py` | no |
| `mode` | function | `_stats_py` | no |
| `moment` | function | `_stats_py` | no |
| `lmoment` | function | `_stats_py` | no |
| `expectile` | function | `_stats_py` | no |
| `skew` | function | `_stats_py` | no |
| `kstat` | function | `_stats_py` | no |
| `kstatvar` | function | `_stats_py` | no |
| `tmean` | function | `_stats_py` | no |
| `tvar` | function | `_stats_py` | no |
| `tmin` | function | `_stats_py` | no |
| `tmax` | function | `_stats_py` | no |
| `tstd` | function | `_stats_py` | no |
| `tsem` | function | `_stats_py` | no |
| `variation` | function | `_variation` | no |
| `rankdata` | function | `_stats_py` | no |
| `tiecorrect` | function | `_stats_py` | no |
| `trim_mean` | function | `_stats_py` | no |
| `gstd` | function | `_stats_py` | no |
| `iqr` | function | `_stats_py` | no |
| `sem` | function | `_stats_py` | no |
| `bayes_mvs` | function | `_morestats` | no |
| `mvsdist` | function | `_morestats` | no |

**Correlation/Association (10)**

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `linregress` | function | `_stats_py` | no |
| `pearsonr` | function | `_stats_py` | no |
| `spearmanr` | function | `_stats_py` | no |
| `spearmanrho` | function | `_correlation` | no |
| `pointbiserialr` | function | `_stats_py` | no |
| `kendalltau` | function | `_stats_py` | no |
| `chatterjeexi` | function | `_correlation` | no |
| `weightedtau` | function | `_stats_py` | no |
| `somersd` | function | `_hypotests` | no |
| `siegelslopes` | function | `_correlation` | no |
| `theilslopes` | function | `_correlation` | no |

**Other functions and classes**

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `CensoredData` | class | `_censored_data` | no |
| `Covariance` | class | `_covariance` | no |
| `gaussian_kde` | class | `_kde` | no |
| `entropy` | function | `_entropy` | no |
| `differential_entropy` | function | `_entropy` | no |
| `median_abs_deviation` | function | `_stats_py` | no |
| `cumfreq` | function | `_stats_py` | no |
| `quantile` | function | `_quantile` | no |
| `percentileofscore` | function | `_stats_py` | no |
| `scoreatpercentile` | function | `_stats_py` | no |
| `relfreq` | function | `_stats_py` | no |
| `binned_statistic` | function | `_binned_statistic` | no |
| `binned_statistic_2d` | function | `_binned_statistic` | no |
| `binned_statistic_dd` | function | `_binned_statistic` | no |
| `boxcox` | function | `_morestats` | no |
| `boxcox_normmax` | function | `_morestats` | no |
| `boxcox_llf` | function | `_morestats` | no |
| `boxcox_normplot` | function | `_morestats` | no |
| `yeojohnson` | function | `_morestats` | no |
| `yeojohnson_normmax` | function | `_morestats` | no |
| `yeojohnson_llf` | function | `_morestats` | no |
| `yeojohnson_normplot` | function | `_morestats` | no |
| `obrientransform` | function | `_stats_py` | no |
| `sigmaclip` | function | `_stats_py` | no |
| `trimboth` | function | `_stats_py` | no |
| `trim1` | function | `_stats_py` | no |
| `zmap` | function | `_stats_py` | no |
| `zscore` | function | `_stats_py` | no |
| `gzscore` | function | `_stats_py` | no |
| `wasserstein_distance` | function | `_stats_py` | no |
| `wasserstein_distance_nd` | function | `_stats_py` | no |
| `energy_distance` | function | `_stats_py` | no |
| `fit` | function | `_fit` | no |
| `ecdf` | function | `_survival` | no |
| `logrank` | function | `_survival` | no |
| `directional_stats` | function | `_stats_py` | no |
| `circmean` | function | `_stats_py` | no |
| `circvar` | function | `_stats_py` | no |
| `circstd` | function | `_stats_py` | no |
| `sobol_indices` | function | `_sensitivity_analysis` | no |
| `ppcc_max` | function | `_morestats` | no |
| `ppcc_plot` | function | `_morestats` | no |
| `probplot` | function | `_morestats` | no |

**Warnings/Errors (4)**

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `DegenerateDataWarning` | warning | `_warnings_errors` | no |
| `ConstantInputWarning` | warning | `_warnings_errors` | no |
| `NearConstantInputWarning` | warning | `_warnings_errors` | no |
| `FitError` | exception | `_warnings_errors` | no |

**Deprecated module re-exports (7) and submodules (3)**

| Symbol | Type | Deprecated? |
|--------|------|-------------|
| `mstats` | submodule | no |
| `qmc` | submodule | no |
| `contingency` | submodule | no |
| `biasedurn` | module | yes (v2.0.0) |
| `kde` | module | yes (v2.0.0) |
| `morestats` | module | yes (v2.0.0) |
| `mstats_basic` | module | yes (v2.0.0) |
| `mstats_extras` | module | yes (v2.0.0) |
| `mvn` | module | yes (v2.0.0) |
| `stats` | module | yes (v2.0.0) |

---

## 10. scipy.signal

### Public API (`__all__`)
**Total: 159 symbols** (139 functions, 8 classes, 1 warning, 8 deprecated module re-exports, 1 submodule, 2 deprecated references)

`__all__` is derived from `_signal_api.__all__` which is `[s for s in dir() if not s.startswith('_')]`

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `BadCoefficients` | warning | `_filter_design` | no |
| `CZT` | class | `_czt` | no |
| `ShortTimeFFT` | class | `_short_time_fft` | no |
| `StateSpace` | class | `_ltisys` | no |
| `TransferFunction` | class | `_ltisys` | no |
| `ZerosPolesGain` | class | `_ltisys` | no |
| `ZoomFFT` | class | `_czt` | no |
| `dlti` | class | `_ltisys` | no |
| `lti` | class | `_ltisys` | no |
| `abcd_normalize` | function | `_filter_design` | no |
| `argrelextrema` | function | `_peak_finding` | no |
| `argrelmax` | function | `_peak_finding` | no |
| `argrelmin` | function | `_peak_finding` | no |
| `band_stop_obj` | function | `_filter_design` | no |
| `bessel` | function | `_filter_design` | no |
| `besselap` | function | `_filter_design` | no |
| `bilinear` | function | `_filter_design` | no |
| `bilinear_zpk` | function | `_filter_design` | no |
| `bode` | function | `_ltisys` | no |
| `buttap` | function | `_filter_design` | no |
| `butter` | function | `_filter_design` | no |
| `buttord` | function | `_filter_design` | no |
| `cheb1ap` | function | `_filter_design` | no |
| `cheb1ord` | function | `_filter_design` | no |
| `cheb2ap` | function | `_filter_design` | no |
| `cheb2ord` | function | `_filter_design` | no |
| `cheby1` | function | `_filter_design` | no |
| `cheby2` | function | `_filter_design` | no |
| `check_COLA` | function | `_spectral_py` | legacy |
| `check_NOLA` | function | `_spectral_py` | no |
| `chirp` | function | `_waveforms` | no |
| `choose_conv_method` | function | `_signaltools` | no |
| `closest_STFT_dual_window` | function | `_short_time_fft` | no |
| `coherence` | function | `_spectral_py` | no |
| `cont2discrete` | function | `_lti_conversion` | no |
| `convolve` | function | `_signaltools` | no |
| `convolve2d` | function | `_signaltools` | no |
| `correlate` | function | `_signaltools` | no |
| `correlate2d` | function | `_signaltools` | no |
| `correlation_lags` | function | `_signaltools` | no |
| `csd` | function | `_spectral_py` | no |
| `cspline1d` | function | `_spline_filters` | no |
| `cspline1d_eval` | function | `_spline_filters` | no |
| `cspline2d` | function | `_spline_filters` | no |
| `czt` | function | `_czt` | no |
| `czt_points` | function | `_czt` | no |
| `dbode` | function | `_ltisys` | no |
| `decimate` | function | `_signaltools` | no |
| `deconvolve` | function | `_signaltools` | no |
| `detrend` | function | `_signaltools` | no |
| `dfreqresp` | function | `_ltisys` | no |
| `dimpulse` | function | `_ltisys` | no |
| `dlsim` | function | `_ltisys` | no |
| `dstep` | function | `_ltisys` | no |
| `ellip` | function | `_filter_design` | no |
| `ellipap` | function | `_filter_design` | no |
| `ellipord` | function | `_filter_design` | no |
| `envelope` | function | `_signaltools` | no |
| `fftconvolve` | function | `_signaltools` | no |
| `filtfilt` | function | `_signaltools` | no |
| `find_peaks` | function | `_peak_finding` | no |
| `find_peaks_cwt` | function | `_peak_finding` | no |
| `findfreqs` | function | `_filter_design` | no |
| `firls` | function | `_fir_filter_design` | no |
| `firwin` | function | `_fir_filter_design` | no |
| `firwin2` | function | `_fir_filter_design` | no |
| `firwin_2d` | function | `_fir_filter_design` | no |
| `freqresp` | function | `_ltisys` | no |
| `freqs` | function | `_filter_design` | no |
| `freqs_zpk` | function | `_filter_design` | no |
| `freqz` | function | `_filter_design` | no |
| `freqz_sos` | function | `_filter_design` | no |
| `freqz_zpk` | function | `_filter_design` | no |
| `gammatone` | function | `_fir_filter_design` | no |
| `gauss_spline` | function | `_spline_filters` | no |
| `gausspulse` | function | `_waveforms` | no |
| `get_window` | function | `windows` | no |
| `group_delay` | function | `_filter_design` | no |
| `hilbert` | function | `_signaltools` | no |
| `hilbert2` | function | `_signaltools` | no |
| `iircomb` | function | `_filter_design` | no |
| `iirdesign` | function | `_filter_design` | no |
| `iirfilter` | function | `_filter_design` | no |
| `iirnotch` | function | `_filter_design` | no |
| `iirpeak` | function | `_filter_design` | no |
| `impulse` | function | `_ltisys` | no |
| `invres` | function | `_filter_design` | no |
| `invresz` | function | `_filter_design` | no |
| `istft` | function | `_spectral_py` | legacy |
| `kaiser_atten` | function | `_fir_filter_design` | no |
| `kaiser_beta` | function | `_fir_filter_design` | no |
| `kaiserord` | function | `_fir_filter_design` | no |
| `lfilter` | function | `_signaltools` | no |
| `lfilter_zi` | function | `_signaltools` | no |
| `lfiltic` | function | `_signaltools` | no |
| `lombscargle` | function | `_spectral_py` | no |
| `lp2bp` | function | `_filter_design` | no |
| `lp2bp_zpk` | function | `_filter_design` | no |
| `lp2bs` | function | `_filter_design` | no |
| `lp2bs_zpk` | function | `_filter_design` | no |
| `lp2hp` | function | `_filter_design` | no |
| `lp2hp_zpk` | function | `_filter_design` | no |
| `lp2lp` | function | `_filter_design` | no |
| `lp2lp_zpk` | function | `_filter_design` | no |
| `lsim` | function | `_ltisys` | no |
| `max_len_seq` | function | `_max_len_seq` | no |
| `medfilt` | function | `_signaltools` | no |
| `medfilt2d` | function | `_signaltools` | no |
| `minimum_phase` | function | `_fir_filter_design` | no |
| `normalize` | function | `_filter_design` | no |
| `oaconvolve` | function | `_signaltools` | no |
| `order_filter` | function | `_signaltools` | no |
| `peak_prominences` | function | `_peak_finding` | no |
| `peak_widths` | function | `_peak_finding` | no |
| `periodogram` | function | `_spectral_py` | no |
| `place_poles` | function | `_lti_conversion` | no |
| `qspline1d` | function | `_spline_filters` | no |
| `qspline1d_eval` | function | `_spline_filters` | no |
| `qspline2d` | function | `_spline_filters` | no |
| `remez` | function | `_fir_filter_design` | no |
| `resample` | function | `_signaltools` | no |
| `resample_poly` | function | `_signaltools` | no |
| `residue` | function | `_filter_design` | no |
| `residuez` | function | `_filter_design` | no |
| `savgol_coeffs` | function | `_savitzky_golay` | no |
| `savgol_filter` | function | `_savitzky_golay` | no |
| `sawtooth` | function | `_waveforms` | no |
| `sepfir2d` | function | `_spline` | no |
| `sos2tf` | function | `_filter_design` | no |
| `sos2zpk` | function | `_filter_design` | no |
| `sosfilt` | function | `_signaltools` | no |
| `sosfilt_zi` | function | `_signaltools` | no |
| `sosfiltfilt` | function | `_signaltools` | no |
| `sosfreqz` | function | `_filter_design` | no |
| `spectrogram` | function | `_spectral_py` | legacy |
| `spline_filter` | function | `_spline_filters` | no |
| `square` | function | `_waveforms` | no |
| `ss2tf` | function | `_lti_conversion` | no |
| `ss2zpk` | function | `_lti_conversion` | no |
| `step` | function | `_ltisys` | no |
| `stft` | function | `_spectral_py` | legacy |
| `sweep_poly` | function | `_waveforms` | no |
| `symiirorder1` | function | `_signaltools` | no |
| `symiirorder2` | function | `_signaltools` | no |
| `tf2sos` | function | `_filter_design` | no |
| `tf2ss` | function | `_lti_conversion` | no |
| `tf2zpk` | function | `_filter_design` | no |
| `unique_roots` | function | `_filter_design` | no |
| `unit_impulse` | function | `_waveforms` | no |
| `upfirdn` | function | `_upfirdn` | no |
| `vectorstrength` | function | `_spectral_py` | no |
| `welch` | function | `_spectral_py` | no |
| `wiener` | function | `_signaltools` | no |
| `zoom_fft` | function | `_czt` | no |
| `zpk2sos` | function | `_filter_design` | no |
| `zpk2ss` | function | `_lti_conversion` | no |
| `zpk2tf` | function | `_filter_design` | no |
| `windows` | submodule | `windows` | no |
| `sigtools` | module | `_sigtools` | no |
| `bsplines` | module | deprecated namespace | yes (v2.0.0) |
| `filter_design` | module | deprecated namespace | yes (v2.0.0) |
| `fir_filter_design` | module | deprecated namespace | yes (v2.0.0) |
| `lti_conversion` | module | deprecated namespace | yes (v2.0.0) |
| `ltisys` | module | deprecated namespace | yes (v2.0.0) |
| `signaltools` | module | deprecated namespace | yes (v2.0.0) |
| `spectral` | module | deprecated namespace | yes (v2.0.0) |
| `waveforms` | module | deprecated namespace | yes (v2.0.0) |
| `wavelets` | module | deprecated namespace | yes (v2.0.0) |
| `spline` | module | deprecated namespace | yes (v2.0.0) |

---

## 11. scipy.spatial

### Public API (`__all__`)
**Total: 23 symbols** (8 functions, 6 classes, 1 exception, 3 deprecated module re-exports, 2 submodules, 3 helper functions)

`__all__` is computed as: `[s for s in dir() if not s.startswith('_')] + ['distance', 'transform']`

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `KDTree` | class | `_kdtree` | no |
| `cKDTree` | class | `_ckdtree` | no |
| `Rectangle` | class | `_kdtree` | no |
| `Delaunay` | class | `_qhull` | no |
| `ConvexHull` | class | `_qhull` | no |
| `Voronoi` | class | `_qhull` | no |
| `SphericalVoronoi` | class | `_spherical_voronoi` | no |
| `HalfspaceIntersection` | class | `_qhull` | no |
| `tsearch` | function | `_qhull` | no |
| `distance_matrix` | function | `_kdtree` | no |
| `minkowski_distance` | function | `_kdtree` | no |
| `minkowski_distance_p` | function | `_kdtree` | no |
| `procrustes` | function | `_procrustes` | no |
| `geometric_slerp` | function | `_geometric_slerp` | no |
| `delaunay_plot_2d` | function | `_plotutils` | no |
| `convex_hull_plot_2d` | function | `_plotutils` | no |
| `voronoi_plot_2d` | function | `_plotutils` | no |
| `QhullError` | exception | `_qhull` | no |
| `distance` | submodule | | no |
| `transform` | submodule | | no |
| `ckdtree` | module | deprecated namespace | yes (v2.0.0) |
| `kdtree` | module | deprecated namespace | yes (v2.0.0) |
| `qhull` | module | deprecated namespace | yes (v2.0.0) |

---

## 12. scipy.interpolate

### Public API (`__all__`)
**Total: 63 symbols** (26 functions, 27 classes, 7 deprecated module re-exports, 3 additional functions)

`__all__` is computed as: `[s for s in dir() if not s.startswith('_')]`

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `AAA` | class | `_bary_rational` | no |
| `Akima1DInterpolator` | class | `_cubic` | no |
| `BPoly` | class | `_bsplines` | no |
| `BSpline` | class | `_bsplines` | no |
| `BarycentricInterpolator` | class | `_polyint` | no |
| `BivariateSpline` | class | `_fitpack2` | no |
| `CloughTocher2DInterpolator` | class | `_ndgriddata` | no |
| `CubicHermiteSpline` | class | `_cubic` | no |
| `CubicSpline` | class | `_cubic` | no |
| `FloaterHormannInterpolator` | class | `_bary_rational` | no |
| `InterpolatedUnivariateSpline` | class | `_fitpack2` | no |
| `KroghInterpolator` | class | `_polyint` | no |
| `LSQBivariateSpline` | class | `_fitpack2` | no |
| `LSQSphereBivariateSpline` | class | `_fitpack2` | no |
| `LSQUnivariateSpline` | class | `_fitpack2` | no |
| `LinearNDInterpolator` | class | `_interpolate` | no |
| `NdBSpline` | class | `_ndbspline` | no |
| `NdPPoly` | class | `_bsplines` | no |
| `NearestNDInterpolator` | class | `_ndgriddata` | no |
| `PPoly` | class | `_interpolate` | no |
| `PchipInterpolator` | class | `_cubic` | no |
| `RBFInterpolator` | class | `_rbfinterp` | no |
| `Rbf` | class | `_rbf` | no |
| `RectBivariateSpline` | class | `_fitpack2` | no |
| `RectSphereBivariateSpline` | class | `_fitpack2` | no |
| `RegularGridInterpolator` | class | `_rgi` | no |
| `SmoothBivariateSpline` | class | `_fitpack2` | no |
| `SmoothSphereBivariateSpline` | class | `_fitpack2` | no |
| `UnivariateSpline` | class | `_fitpack2` | no |
| `approximate_taylor_polynomial` | function | `_polyint` | no |
| `barycentric_interpolate` | function | `_polyint` | no |
| `bisplev` | function | `_fitpack_py` | no |
| `bisplrep` | function | `_fitpack_py` | no |
| `generate_knots` | function | `_fitpack_repro` | no |
| `griddata` | function | `_ndgriddata` | no |
| `insert` | function | `_fitpack_py` | no |
| `interp1d` | class | `_interpolate` | legacy |
| `interp2d` | class | `_interpolate` | legacy |
| `interpn` | function | `_rgi` | no |
| `krogh_interpolate` | function | `_polyint` | no |
| `lagrange` | function | `_interpolate` | no |
| `make_interp_spline` | function | `_bsplines` | no |
| `make_lsq_spline` | function | `_bsplines` | no |
| `make_smoothing_spline` | function | `_bsplines` | no |
| `make_splprep` | function | `_fitpack_repro` | no |
| `make_splrep` | function | `_fitpack_repro` | no |
| `pade` | function | `_pade` | no |
| `pchip_interpolate` | function | `_cubic` | no |
| `spalde` | function | `_fitpack_py` | no |
| `splantider` | function | `_fitpack_py` | no |
| `splder` | function | `_fitpack_py` | no |
| `splev` | function | `_fitpack_py` | no |
| `splint` | function | `_fitpack_py` | no |
| `splprep` | function | `_fitpack_py` | no |
| `splrep` | function | `_fitpack_py` | no |
| `sproot` | function | `_fitpack_py` | no |
| `fitpack` | module | deprecated namespace | yes (v2.0.0) |
| `fitpack2` | module | deprecated namespace | yes (v2.0.0) |
| `interpolate` | module | deprecated namespace | yes (v2.0.0) |
| `ndgriddata` | module | deprecated namespace | yes (v2.0.0) |
| `polyint` | module | deprecated namespace | yes (v2.0.0) |
| `rbf` | module | deprecated namespace | yes (v2.0.0) |
| `interpnd` | module | deprecated namespace | yes (v2.0.0) |

---

## 13. scipy.ndimage

### Public API (`__all__`)
**Total: 75 symbols** (75 functions)

`__all__` is derived from `_ndimage_api.__all__` which collects from `_filters`, `_fourier`, `_interpolation`, `_measurements`, `_morphology`.

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `affine_transform` | function | `_interpolation` | no |
| `binary_closing` | function | `_morphology` | no |
| `binary_dilation` | function | `_morphology` | no |
| `binary_erosion` | function | `_morphology` | no |
| `binary_fill_holes` | function | `_morphology` | no |
| `binary_hit_or_miss` | function | `_morphology` | no |
| `binary_opening` | function | `_morphology` | no |
| `binary_propagation` | function | `_morphology` | no |
| `black_tophat` | function | `_morphology` | no |
| `center_of_mass` | function | `_measurements` | no |
| `convolve` | function | `_filters` | no |
| `convolve1d` | function | `_filters` | no |
| `correlate` | function | `_filters` | no |
| `correlate1d` | function | `_filters` | no |
| `distance_transform_bf` | function | `_morphology` | no |
| `distance_transform_cdt` | function | `_morphology` | no |
| `distance_transform_edt` | function | `_morphology` | no |
| `extrema` | function | `_measurements` | no |
| `find_objects` | function | `_measurements` | no |
| `fourier_ellipsoid` | function | `_fourier` | no |
| `fourier_gaussian` | function | `_fourier` | no |
| `fourier_shift` | function | `_fourier` | no |
| `fourier_uniform` | function | `_fourier` | no |
| `gaussian_filter` | function | `_filters` | no |
| `gaussian_filter1d` | function | `_filters` | no |
| `gaussian_gradient_magnitude` | function | `_filters` | no |
| `gaussian_laplace` | function | `_filters` | no |
| `generate_binary_structure` | function | `_morphology` | no |
| `generic_filter` | function | `_filters` | no |
| `generic_filter1d` | function | `_filters` | no |
| `generic_gradient_magnitude` | function | `_filters` | no |
| `generic_laplace` | function | `_filters` | no |
| `geometric_transform` | function | `_interpolation` | no |
| `grey_closing` | function | `_morphology` | no |
| `grey_dilation` | function | `_morphology` | no |
| `grey_erosion` | function | `_morphology` | no |
| `grey_opening` | function | `_morphology` | no |
| `histogram` | function | `_measurements` | no |
| `iterate_structure` | function | `_morphology` | no |
| `label` | function | `_measurements` | no |
| `labeled_comprehension` | function | `_measurements` | no |
| `laplace` | function | `_filters` | no |
| `map_coordinates` | function | `_interpolation` | no |
| `maximum` | function | `_measurements` | no |
| `maximum_filter` | function | `_filters` | no |
| `maximum_filter1d` | function | `_filters` | no |
| `maximum_position` | function | `_measurements` | no |
| `mean` | function | `_measurements` | no |
| `median` | function | `_measurements` | no |
| `median_filter` | function | `_filters` | no |
| `minimum` | function | `_measurements` | no |
| `minimum_filter` | function | `_filters` | no |
| `minimum_filter1d` | function | `_filters` | no |
| `minimum_position` | function | `_measurements` | no |
| `morphological_gradient` | function | `_morphology` | no |
| `morphological_laplace` | function | `_morphology` | no |
| `percentile_filter` | function | `_filters` | no |
| `prewitt` | function | `_filters` | no |
| `rank_filter` | function | `_filters` | no |
| `rotate` | function | `_interpolation` | no |
| `shift` | function | `_interpolation` | no |
| `sobel` | function | `_filters` | no |
| `spline_filter` | function | `_interpolation` | no |
| `spline_filter1d` | function | `_interpolation` | no |
| `standard_deviation` | function | `_measurements` | no |
| `sum` | function | `_measurements` | no |
| `sum_labels` | function | `_measurements` | no |
| `uniform_filter` | function | `_filters` | no |
| `uniform_filter1d` | function | `_filters` | no |
| `value_indices` | function | `_measurements` | no |
| `variance` | function | `_measurements` | no |
| `vectorized_filter` | function | `_filters` | no |
| `watershed_ift` | function | `_measurements` | no |
| `white_tophat` | function | `_morphology` | no |
| `zoom` | function | `_interpolation` | no |

Deprecated namespace module re-exports (available in namespace but not in `__all__`): `filters`, `fourier`, `interpolation`, `measurements`, `morphology`

---

## 14. scipy.io

### Public API (`__all__`)
**Total: 20 symbols** (9 functions, 3 classes, 6 deprecated module/submodule re-exports, 2 exceptions)

`__all__` is computed as: `[s for s in dir() if not s.startswith('_')]`

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `loadmat` | function | `matlab` | no |
| `savemat` | function | `matlab` | no |
| `whosmat` | function | `matlab` | no |
| `netcdf_file` | class | `_netcdf` | no |
| `netcdf_variable` | class | `_netcdf` | no |
| `FortranFile` | class | `_fortran` | no |
| `FortranEOFError` | exception | `_fortran` | no |
| `FortranFormattingError` | exception | `_fortran` | no |
| `mminfo` | function | `_fast_matrix_market` | no |
| `mmread` | function | `_fast_matrix_market` | no |
| `mmwrite` | function | `_fast_matrix_market` | no |
| `readsav` | function | `_idl` | no |
| `hb_read` | function | `_harwell_boeing` | no |
| `hb_write` | function | `_harwell_boeing` | no |
| `arff` | module | deprecated namespace | yes (v2.0.0) |
| `harwell_boeing` | module | deprecated namespace | yes (v2.0.0) |
| `idl` | module | deprecated namespace | yes (v2.0.0) |
| `mmio` | module | deprecated namespace | yes (v2.0.0) |
| `netcdf` | module | deprecated namespace | yes (v2.0.0) |
| `wavfile` | module | deprecated namespace | yes (v2.0.0) |

---

## 15. scipy.cluster

### Public API (`__all__`)
**Total: 2 symbols** (2 submodules)

`__all__` is explicitly defined as `['vq', 'hierarchy']`.

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `vq` | submodule | `cluster.vq` | no |
| `hierarchy` | submodule | `cluster.hierarchy` | no |

Note: The actual functions are accessed via `scipy.cluster.vq` (e.g., `kmeans`, `vq`, `whiten`, `kmeans2`) and `scipy.cluster.hierarchy` (e.g., `linkage`, `dendrogram`, `fcluster`, `fclusterdata`, `leaders`, `cophenet`, etc.).

---

## 16. scipy.constants

### Public API (`__all__`)
**Total: 166 symbols** (5 functions, 157 constants, 1 warning, 1 dict, 2 deprecated module re-exports)

`__all__` is computed as: `[s for s in dir() if not s.startswith('_')]`

**Functions (5)**

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `value` | function | `_codata` | no |
| `unit` | function | `_codata` | no |
| `precision` | function | `_codata` | no |
| `find` | function | `_codata` | no |
| `convert_temperature` | function | `_constants` | no |
| `lambda2nu` | function | `_constants` | no |
| `nu2lambda` | function | `_constants` | no |

**Warning (1)**

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `ConstantWarning` | warning | `_codata` | no |

**Data (1)**

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `physical_constants` | dict | `_codata` | no |

**Mathematical constants (3)**

| Symbol | Type | Value |
|--------|------|-------|
| `pi` | constant | 3.14159... |
| `golden` | constant | 1.61803... |
| `golden_ratio` | constant | 1.61803... |

**Physical constants (30)**

`c`, `speed_of_light`, `mu_0`, `epsilon_0`, `h`, `Planck`, `hbar`, `G`, `gravitational_constant`, `g`, `e`, `elementary_charge`, `R`, `gas_constant`, `alpha`, `fine_structure`, `N_A`, `Avogadro`, `k`, `Boltzmann`, `sigma`, `Stefan_Boltzmann`, `Wien`, `Rydberg`, `m_e`, `electron_mass`, `m_p`, `proton_mass`, `m_n`, `neutron_mass`

**SI prefixes (24)**

`quetta`, `ronna`, `yotta`, `zetta`, `exa`, `peta`, `tera`, `giga`, `mega`, `kilo`, `hecto`, `deka`, `deci`, `centi`, `milli`, `micro`, `nano`, `pico`, `femto`, `atto`, `zepto`, `yocto`, `ronto`, `quecto`

**Binary prefixes (8)**

`kibi`, `mebi`, `gibi`, `tebi`, `pebi`, `exbi`, `zebi`, `yobi`

**Unit conversions (92)**

Mass: `gram`, `metric_ton`, `grain`, `lb`, `pound`, `blob`, `slinch`, `slug`, `oz`, `ounce`, `stone`, `long_ton`, `short_ton`, `troy_ounce`, `troy_pound`, `carat`, `m_u`, `u`, `atomic_mass`

Angle: `degree`, `arcmin`, `arcminute`, `arcsec`, `arcsecond`

Time: `minute`, `hour`, `day`, `week`, `year`, `Julian_year`

Length: `inch`, `foot`, `yard`, `mile`, `mil`, `pt`, `point`, `survey_foot`, `survey_mile`, `nautical_mile`, `fermi`, `angstrom`, `micron`, `au`, `astronomical_unit`, `light_year`, `parsec`

Pressure: `atm`, `atmosphere`, `bar`, `torr`, `mmHg`, `psi`

Area: `hectare`, `acre`

Volume: `liter`, `litre`, `gallon`, `gallon_US`, `gallon_imp`, `fluid_ounce`, `fluid_ounce_US`, `fluid_ounce_imp`, `bbl`, `barrel`

Speed: `kmh`, `mph`, `mach`, `speed_of_sound`, `knot`

Temperature: `zero_Celsius`, `degree_Fahrenheit`

Energy: `eV`, `electron_volt`, `calorie`, `calorie_th`, `calorie_IT`, `erg`, `Btu`, `Btu_IT`, `Btu_th`, `ton_TNT`

Power: `hp`, `horsepower`

Force: `dyn`, `dyne`, `lbf`, `pound_force`, `kgf`, `kilogram_force`

**Deprecated module re-exports (2)**

| Symbol | Type | Deprecated? |
|--------|------|-------------|
| `codata` | module | yes (v2.0.0) |
| `constants` | module | yes (v2.0.0) |

---

## 17. scipy.misc

### Public API
**Total: 0 symbols** -- ENTIRE MODULE DEPRECATED

The `scipy.misc` module is fully deprecated as of SciPy 1.17.0 (to be removed in 2.0.0). Importing it raises a `DeprecationWarning`. No public API is exposed.

---

## 18. scipy.odr

### Public API (`__all__`)
**Total: 16 symbols** (1 function, 8 classes, 3 exceptions/warnings, 2 deprecated module re-exports, 5 model constants)

**ENTIRE MODULE DEPRECATED** as of SciPy 1.17.0 (to be removed in 1.19.0). Replaced by `odrpack` package on PyPI.

`__all__` is computed as: `[s for s in dir() if not (s.startswith('_') or s in ('odr_stop', 'odr_error'))]`

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `Data` | class | `_odrpack` | yes (whole module) |
| `RealData` | class | `_odrpack` | yes (whole module) |
| `Model` | class | `_odrpack` | yes (whole module) |
| `ODR` | class | `_odrpack` | yes (whole module) |
| `Output` | class | `_odrpack` | yes (whole module) |
| `odr` | function | `_odrpack` | yes (whole module) |
| `OdrWarning` | warning | `_odrpack` | yes (whole module) |
| `OdrError` | exception | `_odrpack` | yes (whole module) |
| `OdrStop` | exception | `_odrpack` | yes (whole module) |
| `polynomial` | constant (model) | `_models` | yes (whole module) |
| `exponential` | constant (model) | `_models` | yes (whole module) |
| `multilinear` | constant (model) | `_models` | yes (whole module) |
| `unilinear` | constant (model) | `_models` | yes (whole module) |
| `quadratic` | constant (model) | `_models` | yes (whole module) |
| `models` | module | deprecated namespace | yes (v2.0.0) |
| `odrpack` | module | deprecated namespace | yes (v2.0.0) |

---

## 19. scipy.datasets

### Public API (`__all__`)
**Total: 5 symbols** (5 functions)

`__all__` is explicitly defined.

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `ascent` | function | `_fetchers` | no |
| `face` | function | `_fetchers` | no |
| `electrocardiogram` | function | `_fetchers` | no |
| `download_all` | function | `_download_all` | no |
| `clear_cache` | function | `_utils` | no |

---

## 20. scipy.differentiate

### Public API (`__all__`)
**Total: 3 symbols** (3 functions)

`__all__` is explicitly defined.

| Symbol | Type | Source Module | Deprecated? |
|--------|------|---------------|-------------|
| `derivative` | function | `_differentiate` | no |
| `jacobian` | function | `_differentiate` | no |
| `hessian` | function | `_differentiate` | no |

---

## Appendix: Methodology

1. Each domain's `__init__.py` was read to determine how `__all__` is constructed
2. For dynamically computed `__all__` (e.g., `[s for s in dir() if not s.startswith('_')]`), all star-imported submodules were parsed via AST to extract their `__all__` lists, then combined with explicitly imported names
3. For signal and ndimage, the `_signal_api.py` / `_ndimage_api.py` intermediate modules were traced
4. Cython `.pyx.in` and `.pyi` files were examined for symbols not captured by Python AST parsing
5. Distribution instances in `scipy.stats` were identified from `_continuous_distns._distn_names` and `_discrete_distns._distn_names` (dynamically generated at import time)
6. Deprecated namespace modules that get re-exported as public symbols were identified from explicit imports before the `__all__` computation
7. Symbol classification (function vs class vs constant) was determined from source code context and documentation

# FEATURE_PARITY

## Status Legend

- `not_started` — no implementation exists
- `in_progress` — partial implementation, major gaps remain
- `parity_gap` — implementation exists but missing significant SciPy functions
- `parity_green` — core V1 functions implemented, conformance tested

## Module-Level Parity Assessment (`scripts/symbol_census.py`)

Across 33 SciPy 1.17.1 modules (**1,555** public symbols; **16** not applicable to a native Rust library; **1,539** applicable), FrankenSciPy has **1,495 declared (`97.1%`)**, **1,525 real (`99.1%`)**, and **891 differentially compared (`57.9%`)** symbols (see [`PARITY-COVERAGE.md`](PARITY-COVERAGE.md) and [`symbol_census.json`](symbol_census.json)).

| Module | Crate | SciPy | N/A | Applicable | Declared | Real | Compared | Real % | Compared % | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `scipy.cluster.hierarchy` | `fsci-cluster` | 32 | 1 | 31 | 31 | 31 | 10 | 100.0% | 32.3% | parity_green |
| `scipy.cluster.vq` | `fsci-cluster` | 4 | 0 | 4 | 4 | 4 | 2 | 100.0% | 50.0% | parity_green |
| `scipy.constants` | `fsci-constants` | 8 | 1 | 7 | 7 | 7 | 5 | 100.0% | 71.4% | parity_green |
| `scipy.datasets` | `fsci-datasets` | 5 | 0 | 5 | 5 | 5 | 0 | 100.0% | 0.0% | parity_green |
| `scipy.differentiate` | `fsci-opt` | 3 | 0 | 3 | 3 | 3 | 0 | 100.0% | 0.0% | parity_green |
| `scipy.fft` | `fsci-fft` | 41 | 4 | 37 | 37 | 37 | 20 | 100.0% | 54.1% | parity_green |
| `scipy.fftpack` | `fsci-fft` | 10 | 0 | 10 | 10 | 10 | 10 | 100.0% | 100.0% | parity_green |
| `scipy.integrate` | `fsci-integrate` | 33 | 1 | 32 | 31 | 32 | 14 | 100.0% | 43.8% | parity_green |
| `scipy.interpolate` | `fsci-interpolate` | 56 | 0 | 56 | 56 | 56 | 14 | 100.0% | 25.0% | parity_green |
| `scipy.io` | `fsci-io` | 14 | 0 | 14 | 14 | 14 | 3 | 100.0% | 21.4% | parity_green |
| `scipy.io.arff` | `fsci-io` | 4 | 0 | 4 | 0 | 2 | 0 | 50.0% | 0.0% | parity_gap |
| `scipy.io.matlab` | `fsci-io` | 10 | 0 | 10 | 3 | 6 | 4 | 60.0% | 40.0% | parity_green |
| `scipy.io.wavfile` | `fsci-io` | 3 | 0 | 3 | 0 | 2 | 0 | 66.7% | 0.0% | parity_green |
| `scipy.linalg` | `fsci-linalg` | 98 | 3 | 95 | 95 | 95 | 71 | 100.0% | 74.7% | parity_green |
| `scipy.linalg.interpolative` | `fsci-linalg` | 8 | 0 | 8 | 8 | 8 | 7 | 100.0% | 87.5% | parity_green |
| `scipy.ndimage` | `fsci-ndimage` | 75 | 0 | 75 | 75 | 75 | 39 | 100.0% | 52.0% | parity_green |
| `scipy.odr` | `fsci-odr` | 10 | 0 | 10 | 10 | 10 | 2 | 100.0% | 20.0% | parity_green |
| `scipy.optimize` | `fsci-opt` | 71 | 1 | 70 | 68 | 70 | 29 | 100.0% | 41.4% | parity_green |
| `scipy.optimize.elementwise` | `fsci-opt` | 4 | 0 | 4 | 4 | 4 | 4 | 100.0% | 100.0% | parity_green |
| `scipy.signal` | `fsci-signal` | 157 | 0 | 157 | 153 | 157 | 77 | 100.0% | 49.0% | parity_green |
| `scipy.signal.windows` | `fsci-signal` | 25 | 0 | 25 | 22 | 25 | 9 | 100.0% | 36.0% | parity_green |
| `scipy.sparse` | `fsci-sparse` | 53 | 2 | 51 | 51 | 51 | 24 | 100.0% | 47.1% | parity_green |
| `scipy.sparse.csgraph` | `fsci-sparse` | 26 | 0 | 26 | 25 | 26 | 24 | 100.0% | 92.3% | parity_green |
| `scipy.sparse.linalg` | `fsci-sparse` | 38 | 1 | 37 | 32 | 35 | 29 | 94.6% | 78.4% | parity_green |
| `scipy.spatial` | `fsci-spatial` | 18 | 0 | 18 | 18 | 18 | 8 | 100.0% | 44.4% | parity_green |
| `scipy.spatial.distance` | `fsci-spatial` | 27 | 0 | 27 | 26 | 27 | 22 | 100.0% | 81.5% | parity_green |
| `scipy.spatial.transform` | `fsci-spatial` | 4 | 0 | 4 | 4 | 4 | 0 | 100.0% | 0.0% | parity_green |
| `scipy.special` | `fsci-special` | 358 | 0 | 358 | 357 | 358 | 208 | 100.0% | 58.1% | parity_green |
| `scipy.stats` | `fsci-stats` | 303 | 0 | 303 | 299 | 302 | 239 | 99.7% | 78.9% | parity_green |
| `scipy.stats.contingency` | `fsci-stats` | 6 | 0 | 6 | 6 | 6 | 2 | 100.0% | 33.3% | parity_green |
| `scipy.stats.mstats` | `fsci-stats` | 31 | 2 | 29 | 29 | 29 | 11 | 100.0% | 37.9% | parity_green |
| `scipy.stats.qmc` | `fsci-stats` | 11 | 0 | 11 | 7 | 11 | 4 | 100.0% | 36.4% | parity_green |
| `scipy.stats.sampling` | `fsci-stats` | 9 | 0 | 9 | 5 | 5 | 0 | 55.6% | 0.0% | parity_gap |

## Detailed Gap Analysis

### fsci-linalg (103/103 applicable real; 78/103 compared)

**Implemented:** solve, solve_triangular, solve_banded, inv, det, lstsq, pinv, lu, lu_factor, lu_solve, qr, qr_delete, qr_insert, qr_multiply, qr_update, svd, svdvals, cholesky, cho_factor, cho_solve, ldl, eig, eigvals, eigh, eigvalsh, schur, hessenberg, expm, norm, matrix_rank, solve_with_casp, logm, sqrtm, sinm, cosm, tanm, sinhm, coshm, tanhm, matrix_power, solve_sylvester, solve_continuous_lyapunov, solve_discrete_lyapunov, solve_circulant, solve_toeplitz, khatri_rao, hadamard, block_diag, companion, issymmetric, ishermitian, is_positive_definite, is_diagonal, is_upper_triangular, is_lower_triangular, is_orthogonal, mat_norm_1, mat_norm_inf, bandwidth, orth, null_space, subspace_angles, polar, and 60+ more

**Missing (LOW priority):**
- Matrix functions: ~~`fractional_matrix_power`~~ ✓ DONE, ~~`funm`~~ ✓ DONE, ~~`signm`~~ ✓ DONE, ~~`sinm`~~ ✓ DONE, ~~`cosm`~~ ✓ DONE, ~~`tanm`~~ ✓ DONE, ~~`sinhm`~~ ✓ DONE, ~~`coshm`~~ ✓ DONE, ~~`tanhm`~~ ✓ DONE, ~~`expm_cond`~~ ✓ DONE, ~~`expm_frechet`~~ ✓ DONE
- Generalized decompositions: ~~`qz`~~ ✓ DONE, ~~`ordqz`~~ ✓ DONE, ~~`cossin`~~ ✓ DONE, ~~`cdf2rdf`~~ ✓ DONE, ~~`rsf2csf`~~ ✓ DONE, ~~`rq`~~ ✓ DONE
- ~~Subspace: `orth`, `null_space`, `subspace_angles`, `polar`~~ ✓ Already implemented
- Banded: ~~`eig_banded`~~ ✓ DONE, ~~`eigh_tridiagonal`~~ ✓ DONE, ~~`cho_solve_banded`~~ ✓ DONE, ~~`solveh_banded`~~ ✓ DONE
- In-scope `scipy.linalg` + `scipy.linalg.interpolative` symbols: 103/103 applicable real (3 BLAS/LAPACK introspection helpers N/A)

### fsci-sparse (112/114 applicable real; 77/114 compared)

**Implemented:** COO/CSR/CSC/BSR/DIA/DOK/LIL formats, spsolve, splu, spilu, cg, pcg, gmres, lgmres, bicg, bicgstab, cgs, qmr, minres, lsqr, lsmr, eigsh, eigs, svds, expm, norm, spsolve_triangular, solve_with_casp_portfolio, eye, diags, random, block_diag, bmat, vstack, hstack, kron, find, tril, triu, dijkstra, bellman_ford, breadth_first_order, depth_first_order, minimum_spanning_tree, connected_components, laplacian, floyd_warshall, shortest_path, reverse_cuthill_mckee, strongly_connected_components, topological_sort, pagerank, betweenness_centrality, closeness_centrality, sparse_norm, sparse_diagonal, sparse_trace, sparse_transpose, matrix_power

**Missing (MEDIUM priority):**
- `ArpackError` and `MatrixRankWarning` in `scipy.sparse.linalg` (`frankenscipy-6j5tz`)

### fsci-opt (77/77 applicable real; 33/77 compared)

**Implemented:** minimize (BFGS, CG, Powell, NM, L-BFGS-B, Newton-CG, TNC, SLSQP with equality/inequality constraints and bounds, trust-ncg, dogleg, trust-exact, trust-krylov — SciPy's trust-region driver and subproblems including GLTR Lanczos/Moré-Sorensen with hotstart when given `hess`/`hessp` — trust-constr as SciPy's Byrd–Omojokun SQP / trust-region interior point with equality/inequality constraints (dicts, `LinearConstraint`, `NonlinearConstraint`), bounds (incl. `keep_feasible`) and Lagrange multipliers, dense Jacobians only; COBYLA as Powell's algorithm transcribed from SciPy's PRIMA translation, with constraints and bounds), direct (DIRECT and DIRECT-L), minimize_with_casp_portfolio, minimize_with_casp_portfolio_and_audit, minimize_scalar, brentq, brenth, bisect, ridder, toms748, halley, newton_scalar, secant, root_scalar, root, fsolve, broyden1, broyden2, anderson, lm_root, curve_fit, least_squares, line_search, Bounds, LinearConstraint, NonlinearConstraint, differential_evolution, basinhopping, dual_annealing, shgo, pso, linprog (methods highs / highs-ds / highs-ipm as fsci's dense bounded revised simplex and Mehrotra IPM with crossover; HiGHS's result shape with con, residuals and marginals; no presolve or sparse input), milp, linear_sum_assignment, approx_fprime, check_grad, cobyla, fmin_cobyla, rosen, rosen_der, rosen_hess, rosen_hess_prod

**Missing (LOW priority):**
- `minimize` methods: ~~`trust-exact`~~ ✓ DONE, ~~`trust-krylov`~~ ✓ DONE, ~~`Newton-CG` hessian product~~ ✓ DONE
- Root: ~~`broyden2`~~ ✓ DONE, ~~`anderson`~~ ✓ DONE, ~~`hybr`~~ ✓ DONE, ~~`lm`~~ ✓ DONE
- ~~Utilities: `rosen` family test functions~~ ✓ DONE

### fsci-integrate (32/32 applicable real; 14/32 compared)

**Implemented:** solve_ivp (RK23, RK45, DOP853, BDF, Radau, LSODA), solve_ivp_with_casp_portfolio, solve_ivp_with_casp_portfolio_and_audit, odeint, solve_bvp, quad (QUADPACK qagse/qagie; infinite bounds), quad_points (qagpe), quad_full_output, quad_weighted (cos/sin/alg*/cauchy: qawoe/qawfe/qawse/qawce), quad_vec, quad_explain, quad_inf, quad_neg_inf, quad_full_inf, dblquad, dblquad_rect, tplquad, tplquad_rect, nquad, trapezoid, trapezoid_uniform, trapezoid_irregular, trapezoid_richardson, simpson, simpson_uniform, simpson_irregular, cumulative_trapezoid, cumulative_trapezoid_uniform, cumulative_trapezoid_initial, cumulative_simpson, romb, romb_func, romberg, fixed_quad, gauss_kronrod_quad, gauss_legendre, newton_cotes, newton_cotes_quad, monte_carlo_integrate, line_integral

**Missing (LOW priority):**
- ~~`cubature` (adaptive N-dimensional cubature)~~ ✓ DONE

### fsci-fft (47/47 applicable real; 30/47 compared)

**Implemented:** fft, ifft, rfft, irfft, fft2, ifft2, fftn, ifftn, rfft2, irfft2, rfftn, irfftn, dct (I-IV), idct, dst (I-IV), dctn, idctn, dstn, idstn, hilbert, hfft, ihfft, fht, ifht, fhtoffset, fftfreq, rfftfreq, fftshift, ifftshift, next_fast_len

**Missing (LOW priority):**
- ~~`dctn`, `idctn`, `dstn`, `idstn` (N-D DCT/DST)~~ ✓ DONE
- ~~`fht`, `ifht` (fast Hankel transform)~~ ✓ DONE

### fsci-special (358/358 applicable real; 208/358 compared)

**Implemented:** gamma, gammaln, digamma, polygamma, gammainc, gammaincc, rgamma, factorial, factorial2, comb, perm, zeta, zetac, beta, betaln, betainc, betaincinv, erf, erfc, erfinv, erfcinv, j0, j1, jn, jv, jve, y0, y1, yn, yv, yve, iv, ive, kv, kve, k0, k1, k0e, k1e, i0, i1, i0e, i1e, hankel1, hankel2, airy, airye, ai_zeros, bi_zeros, hyp0f1, hyp1f1, hyp2f1, select_hypergeometric_branch_with_casp_and_audit, ellipk, ellipkm1, ellipe, ellipkinc, ellipeinc, ellipj, elliprc, elliprd, elliprf, elliprg, elliprj, lambertw, wrightomega, exp1, expi, eval_legendre, eval_chebyt, eval_chebyu, eval_chebyc, eval_chebys, eval_laguerre, eval_genlaguerre, eval_hermite, eval_hermitenorm, roots_legendre, fresnel, entr, rel_entr, kl_div, sinc, xlogy, xlog1py, logsumexp, expit, logit, log_expit, log_ndtr, ndtr, ndtri, huber, pseudo_huber, softmax, log_softmax, exprel, spherical_jn, spherical_yn, spherical_in, spherical_kn, dawsn, struve, modstruve, sph_harm, sph_harm_y, lpmv

**Missing (LOW priority):**
- ~~`sph_harm_y`, `lpmv` (spherical harmonics)~~ ✓ Already implemented
- ~~`modstruve` (modified Struve)~~ ✓ Already implemented
- Expanded differential conformance comparisons across remaining specialized variants

### fsci-stats (353/358 applicable real; 256/358 compared)

**Implemented:** 95+ continuous and 10+ discrete distributions (Normal, StudentT, ChiSquared, Chi, Uniform, Exponential, ExponNorm, PowerNorm, JohnsonSU, JohnsonSB, F, Beta, Gamma, Weibull, Lognormal, LogGamma, Alpha, Poisson, LogSeries, RandInt, Cauchy, HalfCauchy, HalfGenNorm, FatigueLife, DoubleGamma, Laplace, Triangular, Binomial, NegBinomial, Geometric, Hypergeometric, Lomax, Pareto, Rayleigh, Gumbel, GumbelL, Logistic, HypSecant, LevyL, Pearson3, HalfLogistic, Maxwell, VonMises, InverseGaussian/Wald, Argus, Kappa4, Bernoulli, …), CensoredData, Covariance (CovViaPrecision, CovViaPSD, CovViaDiagonal), describe, skew, kurtosis, mode, moment, sem, iqr, variation, zscore, ttest_1samp, ttest_ind, ttest_ind_welch, ttest_rel, ks_1samp, ks_2samp, shapiro, normaltest, mannwhitneyu, wilcoxon, kruskal, f_oneway, chi2_contingency, fisher_exact, linregress, pearsonr, spearmanr, kendalltau, pointbiserialr, weightedtau, chatterjeexi, multiscale_graphcorr, rankdata, gaussian_kde, boxcox, boxcox_normmax, entropy, differential_entropy, circmean, circvar, circstd, bootstrap, permutation_test, power_divergence, median_abs_deviation, trim_mean, tmean, tvar, tstd, tsem, mstats, and many more

**Missing (LOW priority):**
- `make_distribution` (`frankenscipy-1ksfv`) and four UNU.RAN sampling classes (`FastGeneratorInversion`, `NumericalInversePolynomial`, `SimpleRatioUniforms`, `UNURANError`; `frankenscipy-yrlc3`)

### fsci-signal (182/182 applicable real; 86/182 compared)

**Implemented:** savgol_coeffs, savgol_filter, many window functions (hann, hamming, general_hamming, general_cosine, gaussian, general_gaussian, exponential, blackman, blackmanharris, barthann, chebwin, kaiser, dpss, lanczos, taylor, nuttall, bohman, bartlett, flattop, cosine, tukey), convolve, fftconvolve, correlate, correlate2d, find_peaks, butter, cheby1, cheby2, ellip, bessel, iirfilter, iirnotch, iirpeak, firwin, firwin2, remez, kaiserord, lfilter, filtfilt, sosfilt, sosfiltfilt, sosfilt_zi, tf2sos, sos2tf, sos2zpk, zpk2sos, tf2zpk, zpk2tf, freqz, freqz_sos, freqs, group_delay, phase_delay, spectrogram, stft, istft, csd, coherence, welch, periodogram, lombscargle, resample, resample_poly, decimate, upfirdn, chirp, sawtooth, square, gausspulse, sweep_poly, unit_impulse, impulse, step, hilbert, detrend, get_window, medfilt, wiener, deconvolve, minimum_phase, cwt, morlet, ricker, max_len_seq, czt, zoom_fft, matched_filter, chroma, spectral_centroid, spectral_bandwidth, spectral_rolloff, Lti, Dlti, lsim, dlsim

**Missing (LOW priority):**
- ~~LTI systems: `lsim`, `dlsim`~~ ✓ DONE
- ~~`freqs` (analog frequency response)~~ ✓ Already implemented

### fsci-interpolate (56/56 applicable real; 14/56 compared)

**Implemented:** Interp1d (linear, nearest, cubic spline), CubicSpline (with boundary conditions), PchipInterpolator, Akima1DInterpolator, BSpline, make_interp_spline, make_lsq_spline, RegularGridInterpolator, NearestNDInterpolator, LinearNDInterpolator, CloughTocher2DInterpolator, RectBivariateSpline, SmoothBivariateSpline, BarycentricInterpolator, UnivariateSpline, InterpolatedUnivariateSpline, interpn, griddata, RBFInterpolator, KroghInterpolator, Delaunay2D, PPoly, splrep, splev, splder, splantider, splint, sproot, interp2d, lagrange, polyfit, polyval, pade, ratval, polymul, polyadd, polysub, polyder, polyint, polyroots, chebyshev_nodes, barycentric_eval, neville, hermite_interp

**Missing (LOW priority):**
- 2D: complete for scoped V1 surface

### fsci-spatial (49/49 applicable real; 30/49 compared)

**Implemented:** KDTree, cKDTree (query, query_k, query_ball_point, query_ball_tree, count_neighbors, sparse_distance_matrix), euclidean, sqeuclidean, cityblock, chebyshev, cosine, minkowski, correlation, hamming, jaccard, canberra, braycurtis, mahalanobis, seuclidean, wminkowski, pdist, squareform_to_matrix, squareform_to_condensed, cdist, cdist_metric, distance_matrix, directed_hausdorff, hausdorff_distance, ConvexHull, Delaunay, Voronoi, SphericalVoronoi, HalfspaceIntersection (2D full metadata + N-D bounded/unbounded core), QhullError, procrustes, geometric_slerp, boolean metrics (yule, dice, kulsinski, rogerstanimoto, russellrao, sokalmichener, sokalsneath, matching), coordinate transforms (spherical, cylindrical), rotation_matrix, nearest_neighbors, k_nearest_neighbors, centroid, medoid, diameter, spread, Rectangle

**Missing (LOW priority):**
- ~~`Rectangle`~~ ✓ Already implemented

## Packet Readiness Snapshot

This table tracks the strict M3 packet exit gate. A module-level `parity_green` status above does
not promote the P2C packet until extraction, implementation, conformance evidence, and sidecar
artifacts are all green under `artifacts/TOLERANCE_POLICY.md`.

| Packet ID | Extraction | Impl | Conformance | Sidecar | Overall |
|---|---|---|---|---|---|
| `FSCI-P2C-001` (validate_tol) | ready | in_progress | in_progress | in_progress | parity_gap |
| `FSCI-P2C-002` (linalg) | ready | parity_gap | in_progress | in_progress | parity_gap |
| `FSCI-P2C-003` (opt) | ready | parity_gap | in_progress | in_progress | parity_gap |
| `FSCI-P2C-004` (sparse) | ready | parity_gap | in_progress | in_progress | parity_gap |
| `FSCI-P2C-005` (FFT) | ready | parity_gap | in_progress | in_progress | parity_gap |
| `FSCI-P2C-006` (special) | ready | parity_gap | in_progress | in_progress | parity_gap |
| `FSCI-P2C-007` (array API) | ready | in_progress | in_progress | in_progress | in_progress |
| `FSCI-P2C-008` (CASP) | ready | in_progress | in_progress | in_progress | in_progress |
| `FSCI-P2C-009` (cluster) | ready | parity_gap | in_progress | in_progress | parity_gap |
| `FSCI-P2C-010` (spatial) | ready | parity_gap | in_progress | in_progress | parity_gap |
| `FSCI-P2C-011` (signal core) | ready | parity_gap | in_progress | in_progress | parity_gap |
| `FSCI-P2C-012` (stats) | ready | parity_gap | in_progress | in_progress | parity_gap |
| `FSCI-P2C-013` (integrate) | ready | parity_gap | in_progress | in_progress | parity_gap |
| `FSCI-P2C-014` (interpolate) | ready | parity_gap | in_progress | in_progress | parity_gap |
| `FSCI-P2C-015` (ndimage) | ready | parity_gap | in_progress | in_progress | parity_gap |
| `FSCI-P2C-016` (constants) | ready | parity_gap | in_progress | in_progress | parity_gap |
| `FSCI-P2C-017` (IO) | ready | parity_gap | in_progress | in_progress | parity_gap |
| `FSCI-P2C-018` (signal live oracle) | ready | parity_gap | in_progress | in_progress | parity_gap |

## Highest-Impact Missing Functions (V1 Priorities)

All top 10 previously-listed functions are now IMPLEMENTED:

1. ~~**`scipy.stats.describe`** + `skew`, `kurtosis`, `mode`~~ ✓ DONE
2. ~~**`scipy.linalg.logm`** + `sqrtm` + `matrix_power`~~ ✓ DONE
3. ~~**`scipy.optimize.differential_evolution`**~~ ✓ DONE
4. ~~**`scipy.optimize.linprog`**~~ ✓ DONE
5. ~~**`scipy.interpolate.RegularGridInterpolator`**~~ ✓ DONE
6. ~~**`scipy.signal.freqz`**~~ ✓ DONE
7. ~~**`scipy.signal.spectrogram`** / `stft`~~ ✓ DONE
8. ~~**`scipy.signal.firwin`**~~ ✓ DONE
9. ~~**`scipy.special.sinc`** + `xlogy` + `logsumexp`~~ ✓ DONE
10. ~~**`scipy.stats.ks_1samp`** + `ks_2samp`~~ ✓ DONE

**Remaining V1 Priorities (lower impact):**
1. ~~`scipy.special.spherical_jn/yn` — spherical Bessel functions~~ ✓ DONE (also spherical_in, spherical_kn)
2. ~~`scipy.interpolate.RBFInterpolator` — radial basis interpolation~~ ✓ DONE
3. ~~`scipy.signal.lti`/`dlti` class — LTI system representation~~ ✓ DONE (Lti, Dlti structs with freq/step/impulse response)
4. ~~`scipy.linalg.qz` — generalized Schur decomposition~~ ✓ DONE

## Required Evidence Per Feature Family

1. Differential fixture report.
2. Edge-case/adversarial test results.
3. Benchmark delta (for runtime-significant paths).
4. Documented compatibility exceptions (if any).
5. RaptorQ sidecar manifest plus decode-proof record for each durable artifact bundle.

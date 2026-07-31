# Changelog

All notable changes to FrankenSciPy are documented in this file.

FrankenSciPy is a clean-room Rust reimplementation of SciPy with a
Condition-Aware Solver Portfolio (CASP) at its core. The project has no
formal releases yet; this changelog tracks landed capabilities organized
by domain against the `main` branch.

Repository: <https://github.com/Dicklesworthstone/frankenscipy>
License: MIT with OpenAI/Anthropic Rider

---

## [Unreleased] -- HEAD on main (workspace version 0.1.0)

### Workspace crates (19 crates, ~140,000 lines of Rust source plus ~370,000 additional lines of tests, harnesses, and conformance fixtures)

| Crate | First commit | Lines | Purpose |
|---|---|---|---|
| `fsci-linalg` | 2026-02-13 | ~12,100 | Dense and structured linear algebra; CASP-driven solver selection; expm/logm/sqrtm/funm; Sylvester, Lyapunov, continuous and discrete algebraic Riccati |
| `fsci-sparse` | 2026-02-13 | ~13,600 | CSR/CSC/COO/BSR/DIA/DOK/LIL formats; spsolve/splu/spilu; CG, GMRES, LGMRES, BiCGSTAB, MINRES, QMR, LSQR, LSMR; eigs/eigsh/svds; graph algorithms (Dijkstra, Bellman-Ford, MST, PageRank, RCM, centrality) |
| `fsci-integrate` | 2026-02-13 | ~9,400 | `solve_ivp` (RK23/RK45/DOP853/BDF/Radau/LSODA), `odeint`, `solve_bvp`, `quad`, `dblquad`, `tplquad`, `nquad`, Gauss-Legendre, Romberg, Monte Carlo, QMC quadrature |
| `fsci-interpolate` | 2026-03-14 | ~6,300 | `interp1d`, `CubicSpline`, `BSpline`, `Akima`, `PCHIP`, `CubicHermiteSpline`, `RegularGridInterpolator`, `griddata`, Krogh and barycentric interpolators, polynomial arithmetic |
| `fsci-opt` | 2026-02-13 | ~15,100 | `minimize` (Nelder-Mead, BFGS, CG, Powell, L-BFGS-B, Newton-CG, TNC, COBYLA, SLSQP, trust-ncg/krylov/exact/constr, dogleg), `root` (brentq, brenth, ridder, toms748, newton, halley, broyden1/2, anderson, lm_root), `curve_fit`, `least_squares`, global (DE, basinhopping, dual annealing, SHGO, PSO, brute), LP/MILP, `linear_sum_assignment` |
| `fsci-fft` | 2026-02-13 | ~5,600 | Cooley-Tukey mixed-radix, Bluestein for non-power-of-2, RFFT, n-D transforms, DCT/DST I-IV, Hilbert, FHT, plan cache with admission policy |
| `fsci-signal` | 2026-03-15 | ~17,200 | Windows, filter design (Butter/Cheby1/Cheby2/Ellip/Bessel, ZPK forms, lp2lp/lp2hp/lp2bp/lp2bs), `firwin`/`firls`/`remez`, `lfilter`/`filtfilt`/SOS application, Welch/periodogram/coherence/CSD, `find_peaks`, Hilbert analytic signal, CWT, wavelets (Daub, Morlet, Ricker), `chirp`, `detrend`, MFCC/mel/chroma |
| `fsci-spatial` | 2026-03-15 | ~5,600 | KDTree/cKDTree, `pdist`/`cdist`/`distance_matrix`, ConvexHull, Delaunay, Voronoi, HalfspaceIntersection, Hungarian linear assignment, directed Hausdorff |
| `fsci-special` | 2026-02-13 | ~33,000 | Gamma (digamma/polygamma/pentagamma/factorialk), beta, erf, Bessel (j/y/i/k/hankel/spherical), Airy, hypergeometric (0F1/1F1/2F1 with CASP branch selection), elliptic (ellipk/e/j and Carlson RC/RF/RJ/RD/RG), zeta, dawsn, Struve, modified Struve, sph_harm, orthogonal polynomials (Legendre, Cheby T/U/C/S, Hermite, Laguerre, Jacobi, Gegenbauer, shifted variants), Voigt profile, log1pmx/powm1/cosm1 |
| `fsci-stats` | 2026-03-14 | ~49,900 | 80+ continuous distributions and 15+ discrete distributions, each with PDF/CDF/SF/PPF/mean/var/skewness/kurtosis/entropy (closed-form where derivable, numerical otherwise); hypothesis tests (t-tests, KS, Shapiro, Mann-Whitney, Wilcoxon, ANOVA, chi-square); correlation (Pearson/Spearman/Kendall); regression; bootstrap; permutation tests; `gaussian_kde`; Box-Cox; QMC engines (Sobol, Halton, Latin Hypercube, discrepancy variants) |
| `fsci-arrayapi` | 2026-02-13 | ~3,500 | Contract-first Array API backend (`backend.rs`, `broadcast`, `creation`, `indexing`, `audit`) with integration seams for linalg, opt, sparse |
| `fsci-conformance` | 2026-02-13 | ~31,400 (lib + 7 bins) + 767 test files | 3-lane differential harness (self-check, SciPy-oracle, dispatch), RaptorQ evidence packs, parity reports, golden journey artifacts; binaries: `conformance_dashboard`, `e2e_orchestrator`, `fixture_regen`, `live_oracle_capture`, `benchmark_gate`, `raptorq_sidecar`, `tolerance_lint` |
| `fsci-runtime` | 2026-02-13 | ~1,750 | CASP solver portfolio, conformal calibrator, loss matrix, evidence ledger, audit trail, strict vs hardened mode |
| `fsci-cluster` | 2026-03-24 | ~3,200 | `kmeans` and `kmeans++`, DBSCAN, hierarchical linkage, dendrogram, `fcluster`, silhouette, Davies-Bouldin, Calinski-Harabasz |
| `fsci-constants` | 2026-03-24 | ~960 | CODATA 2018 physical constants, SI prefixes, mathematical constants, unit conversions |
| `fsci-ndimage` | 2026-03-23 | ~3,900 | `uniform_filter`, `gaussian_filter`, `median_filter`, `minimum_filter`, `maximum_filter`, `convolve`/`correlate`, morphology (erosion/dilation/opening/closing/binary), `label`, `find_objects`, distance transforms, `affine_transform`, `rotate`, `zoom`, `shift`, Sobel/Prewitt/Laplace edge detectors, histograms and extrema by label |
| `fsci-io` | 2026-03-23 | ~5,300 | `savemat`/`loadmat` (MATLAB v4 and v5), `mmread`/`mmwrite` (Matrix Market), WAV read/write, NetCDF (simplified), IDL `readsav`, Fortran sequential unformatted reader |
| `fsci-odr` | 2026-05-03 | ~1,300 | Orthogonal Distance Regression — `ODR` driver, `Model`, `Data`, `Output`; explicit/implicit models, weighted, multi-response |
| `fsci-datasets` | 2026-05-03 | ~600 | Deterministic embedded fixtures matching SciPy shapes: `ascent`, `face` (RGB and gray), `electrocardiogram` |

---

## March 22 – May 16, 2026: V1 surface buildout and conformance saturation

The window from the previous changelog cut (2026-03-21, 152 commits) through 2026-05-16
added **3,164 commits** and closed **2,404 beads**. Six new crates landed
(`fsci-cluster`, `fsci-constants`, `fsci-ndimage`, `fsci-io`, `fsci-odr`,
`fsci-datasets`); the workspace grew from 13 crates and ~66,000 lines of Rust
to 19 crates and ~140,000 lines. The dominant work patterns:

- **SciPy-parity surface expansion.** Hundreds of distinct `scipy.*` functions
  were ported across every domain crate: 60+ continuous and discrete
  distributions in `fsci-stats`, the full Carlson elliptic family
  (RC/RF/RJ/RD/RG) in `fsci-special`, the ZPK filter-transform family
  (`lp2lp_zpk`/`lp2hp_zpk`/`lp2bp_zpk`/`lp2bs_zpk`/`bilinear_zpk`) and analog
  prototypes (`buttap`, `cheb1ap`, `cheb2ap`, `ellipap`, `besselap`) in
  `fsci-signal`, Daubechies wavelet coefficients, the four scratched-up Carlson
  integrals, Krogh and barycentric polynomial interpolators in
  `fsci-interpolate`, and full `ndimage` filter/morphology/measurement coverage.
- **Conformance saturation.** The `fsci-conformance` test surface grew from a
  few packets to **767 integration test files** organized into 17+ artifact
  packets (`FSCI-P2C-001` through `FSCI-P2C-018`, plus legacy `P2C-001` through
  `P2C-016`). Each function ported in the period received at least one parity,
  property, or metamorphic harness — usually all three. The dominant commit
  pattern in the period is `test(fsci-conformance): scipy parity for <symbol>`
  paired with a beads close.
- **Distribution moment closure.** Every concrete distribution in
  `fsci-stats` (continuous and discrete) now ships explicit
  `skewness()`/`kurtosis()`/`entropy()` implementations: closed-form where
  derivable, numerical-helper-backed (Simpson on PDF, quantile-space Simpson,
  raw-moment integrals, polygamma cumulants) where not, and documented
  `f64::NAN` for heavy-tail families where moments diverge. Several wrong
  formulas surfaced and were corrected during the push — notably
  `GenHalfLogistic` mean/var, `TruncWeibullMin` Simpson endpoint bias (replaced
  with a closed-form incomplete-gamma identity, ~370× faster **than the Simpson
  quadrature it replaced** — a self-speedup against our own previous code, not a
  comparison against SciPy), and `HalfCauchy.mean/var` (now returns `+INFINITY`
  to match SciPy).
- **CASP coverage.** Branch coverage tests were added for the runtime CASP
  decision matrix (`runtime close_within_tol`, `decision_loss_matrix`
  constants, `select_minimize_method`, sparse CASP iterative selector,
  hypergeometric branch selector for `hyp1f1`/`hyp2f1`).
- **Deadlock and concurrency fixes.** Three structural concurrency bugs in
  long-running tests were fixed: `fsci-integrate adaptive_gk15`/`gk15_vec`
  short-circuited on NaN/Inf integrands instead of spinning to the 2^limit
  subdivision wall ([frankenscipy-t45u3](https://github.com/Dicklesworthstone/frankenscipy/commit/b1db88ac));
  `fsci-sparse lgmres_inner` now increments `k` before breaking on lucky
  breakdown so the outer loop terminates on identity-like operators
  ([frankenscipy-3yrl6](https://github.com/Dicklesworthstone/frankenscipy/commit/c30a45ba));
  the `fsci-fft` plan cache acquired a crate-private test lock so `plan.rs` and
  `transforms.rs` tests no longer race
  ([frankenscipy-lw3rl](https://github.com/Dicklesworthstone/frankenscipy/commit/826f02cf)).
  The conformance writer for `parity_report.{json,raptorq.json,decode_proof.json}`
  was also serialized through a process-global mutex
  ([frankenscipy-prngc](https://github.com/Dicklesworthstone/frankenscipy/commits/main?after=)).
- **Documentation and lint sweeps.** Math-notation doc comments across
  `fsci-stats`/`fsci-signal`/`fsci-constants`/`fsci-fft` were wrapped in
  backticks to silence rustdoc's broken-intra-doc-link and unclosed-HTML-tag
  warnings; `excessive_precision` (40 literals) and `collapsible_if` (151
  files) were swept clean. The workspace is **warning-free** across build,
  test, fuzz, and doc surfaces as of 2026-05-16.
- **Defect ledger.** Three open defects were filed and remain on the backlog
  at cut: [`frankenscipy-r1vok`](https://github.com/Dicklesworthstone/frankenscipy/commit/e5f353b5)
  (periodogram/welch O(N) normalization divergence),
  [`frankenscipy-cw6k2`](https://github.com/Dicklesworthstone/frankenscipy/commit/a2c47604)
  (iirnotch/iirpeak r approximation),
  [`frankenscipy-ot7tm`](https://github.com/Dicklesworthstone/frankenscipy/commit/7d19a662)
  (gausspulse √2 envelope drift).

The following per-crate sections list the structural additions in this window.
Earlier work (2026-02-13 through 2026-03-21) is preserved in the per-domain
sections further below.

### `fsci-cluster` (NEW, 2026-03-24)

- KMeans (Lloyd's with kmeans++ initialization), DBSCAN, hierarchical
  agglomerative linkage, dendrogram, `fcluster`, silhouette coefficient,
  Davies-Bouldin and Calinski-Harabasz indices. SciPy-oracle parity harness in
  `fsci-conformance` for linkage helpers, kmeans property tests, DBSCAN
  property coverage.

### `fsci-constants` (NEW, 2026-03-24)

- CODATA 2018 physical constants, SI prefixes, mathematical constants
  (`pi`, `e`, `euler_gamma`, etc.), and `convert_*` unit conversions. Full
  SciPy parity surface; conformance harness `diff_constants_physical`,
  `diff_constants_conversions`, `diff_constants_value_lookup`.

### `fsci-ndimage` (NEW, 2026-03-23)

- Convolution and correlation kernels, uniform/gaussian/median/min/max
  filters, generic filter with closure-driven kernels, binary and grayscale
  morphology (erosion, dilation, opening, closing, hit-or-miss), connected
  component labeling (`label`, `find_objects`), Euclidean distance transform,
  affine transform, rotate, zoom, shift, and Sobel/Prewitt/Laplace edge
  detectors. Histograms and extrema indexed by label.

### `fsci-io` (NEW, 2026-03-23)

- `savemat`/`loadmat` for MATLAB v4 and v5, including compressed and struct
  arrays. Matrix Market `mmread`/`mmwrite` for dense and sparse. WAV PCM and
  IEEE-float read/write. Simplified NetCDF reader. IDL `.sav` reader. Fortran
  sequential unformatted reader
  ([frankenscipy-afvx9](https://github.com/Dicklesworthstone/frankenscipy/commit/1b2ac0eb)).

### `fsci-odr` (NEW, 2026-05-03)

- Orthogonal Distance Regression driver matching SciPy's `scipy.odr` API:
  `ODR`, `Model`, `Data`, `Output`, with explicit and implicit models,
  weighted fits, and multi-response support.

### `fsci-datasets` (NEW, 2026-05-03)

- Deterministic embedded sample fixtures matching SciPy shapes: `ascent()`,
  `face(gray=…)`, `electrocardiogram()`. Used by examples, doctests, and
  performance benchmarks.

### `fsci-stats` distribution coverage push (Mar–May 2026)

- **New distributions ported from SciPy** (selected highlights, full list in
  beads `frankenscipy-*`):
  `wrapcauchy`, `skewcauchy`, `skewnorm`, `exponpow`, `truncpareto`,
  `truncexpon`, `truncweibull_min`, `truncnorm` refinements, `irwinhall`,
  `rdist`, `recipinvgauss`, `kstwobign`, `kappa3`, `kappa4`, `burr` (Type III
  and Type XII), `weibull_max`, `powerlognorm`, `boltzmann`, `planck`,
  `yulesimon`, `zipfian`, `fisk` / `loglogistic`, `genlogistic`, `loggamma`,
  `gennorm`, `halfgennorm`, `genextreme`, `genpareto`, `arcsine`,
  `semicircular`, `cosine`, `anglit`, `hypsecant`, `crystalball`, `argus`,
  `johnsonsu`, `johnsonsb`, `pearson3`, `exponnorm`, `fatiguelife`,
  `inversegaussian`, `bradford`, `tukeylambda`, `rice`, `nakagami`, `levy_l`,
  and more.
- **Closed-form skewness and kurtosis** for ~50 continuous and ~10 discrete
  distributions; numerical fallbacks (Simpson on PDF, quantile-space Simpson,
  raw-moment integrals, polygamma cumulants) for the remainder.
  Heavy-tail families (Alpha, etc.) document `NaN` explicitly when moments
  diverge.
- **Closed-form entropies** for ~60 continuous and ~10 discrete distributions
  (`Beta`, `ChiSquared`, `Chi`, `F`, `StudentT`, `InverseGamma`, `GenPareto`,
  `GenExtreme`, `Rayleigh`, `Maxwell`, `Levy`, `LevyLeft`, `Cauchy`,
  `HalfCauchy`, `Pareto`, `Lomax`, `Erlang`, `Gilbrat`, `DoubleWeibull`,
  `DoubleGamma`, `GenNorm`, `HalfGenNorm`, `RDist`, `Fisk`/`Loglogistic`,
  `LogLaplace`, `GenHalfLogistic`, `Bradford`, `InvWeibull`, `PowerLaw`,
  `WeibullMax`, `FrechetR`, `Loguniform`, `Gompertz`, `Moyal`,
  `LaplaceAsymmetric`, `Nakagami`, `Trapezoid`, `GenLogistic`, `LogGamma`,
  `Pearson3`, `BetaPrime`, `TruncExpon`, `TruncPareto`, `VonMises`,
  `TruncNormal`, `SkewCauchy`, `WrapCauchy`, `IrwinHall`, …).
- **Simpson-based entropy** for `PowerNorm`/`PowerLognorm`/`JohnsonSU`/`JohnsonSB`
  /`ExponPow`/`ExponWeibull`/`KsTwoBign`/`TruncWeibullMin`/`Kappa3`/`Kappa4`
  /`CrystalBall`/`Argus`/`FatigueLife`/`SkewNorm`/`FoldedNormal`/`FoldedCauchy`
  /`Rice`/`TukeyLambda`/`NoncentralT`/`InverseGaussian`/`ExponNorm` and the
  `Burr`/`Mielke`/`RecipInvGauss`/`Alpha` arctan-compactified family.
- **Method-of-moments and MLE `fit()`** for ~30 distributions including
  `Erlang`, `Chi`, `Rice`, `Nakagami`, `Fisk`, `Bradford`, `Gilbrat`,
  `Gompertz`, `GenLogistic`, `GenNorm`, `HalfGenNorm`, `GenPareto`,
  `LogGamma`, `InverseGamma`, `InverseGaussian`, `ExponNorm`, `Levy`,
  `LevyLeft`, `LogLaplace`, `Loguniform`, `Anglit`, `CosineDistribution`,
  `Moyal`, `TruncExpon`, `Pearson3` (skew estimator), and the six
  parameterless distributions in one batch
  ([frankenscipy-hv9kw](https://github.com/Dicklesworthstone/frankenscipy/commit/1539c4de)).
- **`mode()` closed forms** for 27 continuous and 4 discrete distributions
  ([frankenscipy-tlw2q](https://github.com/Dicklesworthstone/frankenscipy/commit/91bbb954),
  [frankenscipy-rt24d](https://github.com/Dicklesworthstone/frankenscipy/commit/b8110d97),
  [frankenscipy-ckykz](https://github.com/Dicklesworthstone/frankenscipy/commit/d353b4a9)).
- **QMC engines**: full Latin Hypercube, Sobol, Halton; geometric
  discrepancy, centered L2, mixture L2, wraparound L2, L2-star, update
  rules, scaling
  ([frankenscipy-e5j6p](https://github.com/Dicklesworthstone/frankenscipy/commit/637f6327),
  [frankenscipy-aa2xz](https://github.com/Dicklesworthstone/frankenscipy/commit/0e5905f3)
  + companions).
- **Bug fixes** (representative):
  - `GenHalfLogistic.mean/var` replaced (wrong `(ψ(1/c+1) + γ)/c` formula
    that returned 3.0 at `c=0.5` outside the `[0, 2]` support) with Simpson
    on the substitution `u = (1−cx)^{1/c}`
    ([frankenscipy-snlpq adjacent](https://github.com/Dicklesworthstone/frankenscipy/commit/527bc55b)).
  - `TruncWeibullMin.mean/var` quadrature replaced with closed-form
    incomplete-gamma: `E[X^k] = e^{a^c}/s · Γ(k/c+1) · (P(k/c+1, b^c) − P(k/c+1, a^c))`,
    ~370× faster **than the Simpson quadrature it replaced**, and structurally
    exact. The ~370× is a self-speedup against our own previous implementation;
    it is **not** a speed claim against `scipy.stats.truncweibull_min`, which
    evaluates the same identity in its own `_munp`. Do not read this number as a
    reason to prefer FrankenSciPy over SciPy for this call
    ([frankenscipy-b47bcc6e](https://github.com/Dicklesworthstone/frankenscipy/commit/b47bcc6e)).
  - `HalfCauchy.mean/var` now returns `+INFINITY` to match SciPy
    ([frankenscipy-59anq](https://github.com/Dicklesworthstone/frankenscipy/commit/aa1724bc)).
  - `Burr3`, `RecipInvGauss`, `Kappa3`, `ExponPow`, `PowerLognorm`,
    `TruncWeibullMin` all received closed-form or numerically integrated
    mean/var replacements during the audit.

### `fsci-special` expansion (Mar–May 2026)

- **Carlson elliptic family** complete: `elliprc`, `elliprf`, `elliprd`,
  `elliprg`, `elliprj`
  ([frankenscipy-ewuqd cluster](https://github.com/Dicklesworthstone/frankenscipy/commit/2e995e3f)).
- **Bessel-zero arrays**: `ai_zeros`, `bi_zeros`, `jn_zeros`, `yn_zeros`
  for the first k positive zeros.
- **Riccati-Bessel functions**: `riccati_jn` and `riccati_yn` value-and-derivative
  arrays.
- **Legendre arrays**: `lpn`/`lqn` (Legendre P/Q with derivatives), `lpmn`/`lqmn`
  (associated Legendre arrays).
- **Orthogonal polynomial roots/evaluations**: `roots_chebyc`, `roots_chebys`,
  `roots_sh_legendre`, `roots_sh_chebyt`, `roots_sh_chebyu`, `eval_chebyc`,
  `eval_chebys`, `eval_sh_jacobi`, `assoc_laguerre`.
- **Accurate-near-zero variants**: `log1pmx`, `powm1`, `cosm1`.
- **Profile / activation helpers**: `wofz_real`, `voigt_profile (γ=0)`,
  `tanpi`, `tklmbda` (Tukey-lambda CDF), `factorialk`.
- **Real-argument binomial**: `binom(n, k)` for real `n`
  ([frankenscipy-tb3ph](https://github.com/Dicklesworthstone/frankenscipy/commit/e47e9cd9)).
- **CASP branch selectors**: hypergeometric branch selection logic for
  `hyp1f1` and `hyp2f1`
  ([frankenscipy-d22cc0d1](https://github.com/Dicklesworthstone/frankenscipy/commit/d22cc0d1)).
- **Kelvin functions** `ber`/`bei`/`ker`/`kei` with SciPy parity tests.
- **Pentagamma** for LogGamma cumulants
  ([frankenscipy-04b3d79d](https://github.com/Dicklesworthstone/frankenscipy/commit/04b3d79d)).

### `fsci-signal` expansion (Mar–May 2026)

- **Analog prototypes**: `buttap` (Butterworth) and `cheb1ap` (Chebyshev I);
  the remaining Cheby II / Elliptic / Bessel prototypes are reachable via the
  full `cheby2` / `ellip` / `bessel` design entry points rather than a
  standalone `*ap` function.
- **Filter-order helpers**: `buttord`, `cheb1ord`, `cheb2ord`, `ellipord`.
- **Lowpass-to-X ZPK and BA transforms**: `lp2lp_zpk`/`lp2lp`,
  `lp2hp_zpk`/`lp2hp`, `lp2bp_zpk`/`lp2bp`, `lp2bs_zpk`/`lp2bs`,
  `bilinear_zpk`.
- **Filter representation conversions**: `tf2zpk`, `zpk2tf`, `sos2tf`,
  `freqz_zpk`, `group_delay`, `group_delay_from_ba`, `unique_roots`,
  `normalize_filter`, `lfiltic`.
- **Wavelets**: `daub` (Daubechies coefficients), `morlet2`, Ricker.
- **Audio**: MFCC, mel filterbank, chroma feature extraction.
- **Window helpers**: `kaiser_atten`, `kaiser_beta`, Taylor and exponential
  windows, `general_hamming`.
- **Filter design**: full `iirfilter` dispatcher covering all five IIR
  families; `firls` FIR design.
- **Signal generators / pulses**: `chirp`, `sawtooth`, `unit_impulse`,
  `gauspuls`, `correlation_lags`.

### `fsci-linalg` expansion (Mar–May 2026)

- **Matrix equations**: discrete and continuous algebraic Riccati
  equations (`solve_discrete_are`, `solve_continuous_are`); Lyapunov;
  Sylvester.
- **Banded specialists**: `solveh_banded` across multiple band storage
  formats, `eig_banded`, `eigh_tridiagonal`.
- **Decomposition reconstruction tests**: QZ, LDL, Schur+Hessenberg, QR
  insert/delete/update with full SciPy parity.
- **Subspace and polar**: `subspace_angles`, `polar`, `orth`, `null_space`,
  refined with property-based reconstruction tests.
- **Structural constructors**: `block_diag`, `bmat`, `vstack`, `hstack`,
  `leslie`, `pascal`, `vander`, `hankel`, `helmert`, `random_spd`,
  `random_matrix`, `mat_allclose`.
- **Matrix functions** parity for `matrix_power`, `signm`.

### `fsci-sparse` expansion (Mar–May 2026)

- **Iterative solvers**: full Krylov suite covered by residual conformance —
  CG, BiCG, BiCGSTAB, CGS, GMRES, LGMRES (with lucky-breakdown termination
  fix), QMR, MINRES, LSQR, LSMR.
- **Eigensolvers**: `eigs`/`eigsh`/`svds` with sorting and conjugation
  properties verified.
- **Graph algorithms**: `dijkstra`, `bellman_ford`, `connected_components`,
  `minimum_spanning_tree`, `bfs_order`/`dfs_order`, `reverse_cuthill_mckee`,
  `pagerank`, centrality measures, clustering/diameter/eccentricity.
- **Format coverage**: `eye_rectangular` for non-square `eye`, diagonal
  construction `diags`, Kronecker product, stack/block builders, elementwise
  reductions, density/Frobenius/inner-product helpers, graph summary export.

### `fsci-integrate` expansion (Mar–May 2026)

- **Monte Carlo and QMC quadrature**: `monte_carlo_integrate`, `qmc_quad`.
- **Romberg and N-d quadrature**: `romberg`, `romb_func`, `nquad`,
  `quad_explain`, sample-form variants (`cumulative_trapezoid`,
  `simpson`/`trapezoid`/`romb`/Newton-Cotes on sample sequences).
- **`select_initial_step`** Hairer heuristic with full SciPy parity
  coverage.
- **`adaptive_gk15`/`adaptive_gk15_vec` deadlock fix** when the integrand
  returns non-finite values.
- **`odeint` closed-form parity** suite added in `fsci-conformance`.

### `fsci-opt` expansion (Mar–May 2026)

- **CASP `minimize` selector**: runtime algorithm selection for
  unconstrained minimization
  ([frankenscipy-0b5a2](https://github.com/Dicklesworthstone/frankenscipy/commit/638ad4d9)).
- **Property tests** added for COBYLA, augmented Lagrangian, projected
  gradient, simulated annealing, brute force + PSO, scalar bounded +
  trisection, line_search_wolfe1/2, constrained DE, TNC/SLSQP/Newton-CG/
  trust-constr, scalar root finders, `fsolve`, `lm_root`, `curve_fit`,
  numerical gradient/Jacobian/Hessian, NNLS/isotonic.
- **Hessian convenience**: `rosen_hess`, `rosen_hess_prod` SciPy parity.

### `fsci-interpolate` expansion (Mar–May 2026)

- **Cubic Hermite splines**: `CubicHermiteSpline` type and the
  `cubic_hermite_interpolate` free function.
- **Free-function interpolators matching `scipy.interpolate`**:
  `pchip_interpolate`, `akima1d_interpolate`, `krogh_interpolate`,
  `barycentric_interpolate`.
- **B-spline least squares**: `make_lsq_spline` parity across `k=1/3/5`.
- **Polynomial arithmetic helpers**: `polyfromroots`, polynomial value/
  derivative/ratval, polynomial-root parity, `interpn` grid-interpolator
  parity.

### `fsci-fft` expansion (Mar–May 2026)

- **n-D and DCT/DST n-D coverage**: `fftn`/`ifftn`, `rfftn`/`irfftn`,
  `fft2`/`ifft2`, `rfft2`/`irfft2`, `dctn`/`idctn`, `dstn`/`idstn` —
  all parity-verified.
- **Hilbert analytic signal** SciPy parity for `fft::hilbert`.
- **Plan-cache concurrency fix** ([frankenscipy-lw3rl](https://github.com/Dicklesworthstone/frankenscipy/commit/826f02cf)).
- **`fast_len` helpers** for choosing optimal transform lengths.

### `fsci-spatial` expansion (Mar–May 2026)

- **KDTree predicates**: `query_pairs`, `count_neighbors`.
- **Distance helpers**: `pdist_func`/`cdist_func`/`distance_matrix`,
  weighted distances, distance-matrix validators.
- **Convex hull and Voronoi** parity coverage for boundary cases.

### `fsci-conformance` saturation (Mar–May 2026)

- **3-lane harness** stabilized: self-check, oracle-backed
  (`run_<family>_packet_with_oracle_capture`), and dispatch
  (`run_differential_test`).
- **Python oracles** for every domain: `scipy_{linalg,optimize,special,
  stats,signal,fft,sparse,spatial,interpolate,ndimage,integrate,io,cluster,
  constants,arrayapi}_oracle.py` — 15 oracle scripts wrapping reference
  implementations.
- **Artifact governance**: `parity_report.json`, `decode_proof.json`, and
  RaptorQ systematic-encoding sidecars emitted per packet; writer
  serialized via a process-global mutex
  ([frankenscipy-prngc](https://github.com/Dicklesworthstone/frankenscipy/commit/8d3f64fe)).
- **Packet coverage**: artifact directories exist for `FSCI-P2C-001`
  through `FSCI-P2C-018` plus the legacy `P2C-001` through `P2C-016`
  trees; over **767 integration test files** under
  `crates/fsci-conformance/tests/`.

### Workspace hygiene (Mar–May 2026)

- **Warning-free across build, test, fuzz, doc** as of 2026-05-16
  ([frankenscipy-ql8pu](https://github.com/Dicklesworthstone/frankenscipy/commit/bebe414e),
  [frankenscipy-iznn6](https://github.com/Dicklesworthstone/frankenscipy/commit/b0d33ffe),
  [frankenscipy-h3hnk](https://github.com/Dicklesworthstone/frankenscipy/commit/18e458d4),
  [frankenscipy-cgjh3](https://github.com/Dicklesworthstone/frankenscipy/commit/269024f0),
  [frankenscipy-fhh87](https://github.com/Dicklesworthstone/frankenscipy/commit/b50ffcc5),
  [frankenscipy-xjan0](https://github.com/Dicklesworthstone/frankenscipy/commit/10f9f0f0),
  [frankenscipy-zdkmb](https://github.com/Dicklesworthstone/frankenscipy/commit/2439e45c),
  [frankenscipy-uu2hd](https://github.com/Dicklesworthstone/frankenscipy/commit/cdc22420),
  [frankenscipy-zk3q8](https://github.com/Dicklesworthstone/frankenscipy/commit/88becfd1)).
- **Clippy sweeps**: `excessive_precision`
  ([frankenscipy-kgt26](https://github.com/Dicklesworthstone/frankenscipy/commit/b3d8bfe5), 40 literals)
  and `collapsible_if`
  ([frankenscipy-snlpq](https://github.com/Dicklesworthstone/frankenscipy/commit/297d7243), 151 files)
  applied workspace-wide.
- **Doc comment math wrap**: bracket-indexed math (`x[k]`, `[a,b]`,
  `[[a,b],[c,d]]`, `[x]⁺`) and angle-bracketed generics/placeholders
  (`Vec<Vec<f64>>`, `<path>`, `<file>`) wrapped in backticks across
  `fsci-stats`, `fsci-signal`, `fsci-constants`, and `fsci-fft` to
  silence rustdoc.

---

## Linear Algebra (`fsci-linalg`)

Landed 2026-02-13. Core dense linear algebra with the Condition-Aware Solver Portfolio (CASP) that drives the project's identity.

### Decompositions and solvers

- LU, QR, Cholesky, SVD, eigendecomposition, and least-squares solvers at initial commit.
  ([55f1ee9](https://github.com/Dicklesworthstone/frankenscipy/commit/55f1ee94577f2b67e6242b93ac459998a47aa797))
- CASP solver portfolio with conformal calibration for adaptive solver selection.
  ([c129019](https://github.com/Dicklesworthstone/frankenscipy/commit/c12901945d5ab9ee8313ddd9c1567a6cb374cdad))
- `lstsq` driver dispatch with CASP-aware algorithm selection and differential conformance harness.
  ([9b921bd](https://github.com/Dicklesworthstone/frankenscipy/commit/9b921bd69c71748e24d8c9819964e7e1c744d5f7))
- LDL decomposition.
  ([8c20565](https://github.com/Dicklesworthstone/frankenscipy/commit/8c20565d76684f4ff0711e23e9b33b19c88341fb))
- Schur and Hessenberg decompositions; sparse matrix construction helpers.
  ([d5c7dff](https://github.com/Dicklesworthstone/frankenscipy/commit/d5c7dff39c3805c9dc971deba08fffb6ef5f72bf))

### Matrix functions

- Matrix exponential (`expm`) via Pade approximation.
  ([32d737c](https://github.com/Dicklesworthstone/frankenscipy/commit/32d737c4890a1879305f90451012f3585201861e))
- Matrix square root (`sqrtm`), matrix logarithm (`logm`), general matrix function (`funm`).
  ([23c5ff3](https://github.com/Dicklesworthstone/frankenscipy/commit/23c5ff3e19a15fd7f88c5be85fd7a895b1c60a6c))
- Fix Sylvester equation solver; validate `arange` inputs in Array API.
  ([52d0818](https://github.com/Dicklesworthstone/frankenscipy/commit/52d081846f940cb940067a8d096f80d562a93899))

### Solver portfolio improvements

- Refactor solver portfolio: backward error tracking, numerical stability improvements, CASP calibration refinements.
  ([a83ed1a](https://github.com/Dicklesworthstone/frankenscipy/commit/a83ed1a6f4e723a248cec7361bf8f46b6f4730fd))
- Fix `logm` Parlett recurrence denominator, improve simplex pivot selection.
  ([93fb998](https://github.com/Dicklesworthstone/frankenscipy/commit/93fb998d3e6e241461ecc63926a3188c82fe9700))
- Fix `logm` Parlett recurrence (second pass) and apply rustfmt.
  ([5c7b400](https://github.com/Dicklesworthstone/frankenscipy/commit/5c7b400a9ddcc78354e37186a9a8a446f5592702))
- Linalg improvements and parity report updates.
  ([359ab72](https://github.com/Dicklesworthstone/frankenscipy/commit/359ab725a12bcd4dadc8398c3087f8a3102a88e1))

### Benchmarks

- Criterion benchmark harness for linear algebra operations.
  ([0a22fb4](https://github.com/Dicklesworthstone/frankenscipy/commit/0a22fb41d5585d0eb72c77caf58087474f8b3852))

---

## Sparse Matrices (`fsci-sparse`)

Scaffolded 2026-02-13; expanded significantly through 2026-03-21.

### Matrix formats and arithmetic

- CSR/CSC/COO construction, arithmetic, format conversion.
  ([a870010](https://github.com/Dicklesworthstone/frankenscipy/commit/a87001061d76b849ac84fd530e84449b27d01921))
- Comprehensive unit and property-based test coverage.
  ([22a79cd](https://github.com/Dicklesworthstone/frankenscipy/commit/22a79cdeef6a61d356d912a20b430c998b8c2cad))

### Iterative solvers

- Expand sparse solver algorithms (CG, BiCGSTAB).
  ([5730fd5](https://github.com/Dicklesworthstone/frankenscipy/commit/5730fd5804ea72e6caf90804e7197fea14acecfe))
- ILU(0) preconditioner.
  ([8abf432](https://github.com/Dicklesworthstone/frankenscipy/commit/8abf432f043f59d28306edec97038cf188d51908))
- Preconditioned CG.
  ([7a26704](https://github.com/Dicklesworthstone/frankenscipy/commit/7a26704bda817efd56d73973ee84bff55feea4b6))
- `spsolve_triangular` for triangular sparse systems.
  ([79b6092](https://github.com/Dicklesworthstone/frankenscipy/commit/79b6092ca696bfd185cb5943cc22260753df00ea))
- GMRES and iterative refinement solvers.
  ([e070b0e](https://github.com/Dicklesworthstone/frankenscipy/commit/e070b0ec22e506acb1ca33ea2b5c3eafc2e018e2))

### Eigensolvers

- Sparse `eigsh` solver (implicitly restarted Lanczos).
  ([71317bd](https://github.com/Dicklesworthstone/frankenscipy/commit/71317bd82cd58eaf7e7c092b7b118296768c3f20))

### Graph algorithms

- Bellman-Ford shortest-path solver.
  ([adb66db](https://github.com/Dicklesworthstone/frankenscipy/commit/adb66db7a32181cb45e599f09b13232f796a26b7))
- BFS and DFS graph traversal on sparse adjacency matrices.
  ([d08a4f7](https://github.com/Dicklesworthstone/frankenscipy/commit/d08a4f798fe23b9f63c4e577ddbfaebc058458ab))

---

## Integration (`fsci-integrate`)

Landed 2026-02-13 with basic IVP infrastructure; expanded through 2026-03-21.

### Initial value problems (IVP)

- Runge-Kutta integration API (RK23, RK45).
  ([5730fd5](https://github.com/Dicklesworthstone/frankenscipy/commit/5730fd5804ea72e6caf90804e7197fea14acecfe))
- DOP853 Butcher tableau.
  ([8abf432](https://github.com/Dicklesworthstone/frankenscipy/commit/8abf432f043f59d28306edec97038cf188d51908))
- BDF stiff solver.
  ([8c20565](https://github.com/Dicklesworthstone/frankenscipy/commit/8c20565d76684f4ff0711e23e9b33b19c88341fb))
- Wire Radau and BDF into `solve_ivp` dispatch.
  ([0a02d09](https://github.com/Dicklesworthstone/frankenscipy/commit/0a02d091594d12fe20d998482530abb852402328))
- `odeint` convenience wrapper.
  ([79b6092](https://github.com/Dicklesworthstone/frankenscipy/commit/79b6092ca696bfd185cb5943cc22260753df00ea))
- Dense output and step-size control refinements.
  ([f78ada6](https://github.com/Dicklesworthstone/frankenscipy/commit/f78ada6f2596589e25cbacbb40ef41c3591f24e6))
- `IvpSolver` trait and Hermite interpolation variant.
  ([2b6b211](https://github.com/Dicklesworthstone/frankenscipy/commit/2b6b2114028adab9f2dc4ed99bbced00c37f1ce3))

### ODE event handling

- ODE event detection with root-finding on dense output.
  ([60c5930](https://github.com/Dicklesworthstone/frankenscipy/commit/60c59301358783bd0c4d6aed6dabd30344685632))
- Major overhaul: terminal events, direction detection, dense output interpolation.
  ([3fc730e](https://github.com/Dicklesworthstone/frankenscipy/commit/3fc730e606529c4259f21522743b5c66cb1af8bb))

### Boundary value problems (BVP)

- `solve_bvp` boundary value problem solver.
  ([447545e](https://github.com/Dicklesworthstone/frankenscipy/commit/447545ebcb0deb4bd61aa535f93297c222dfb1aa))
- Correct not-a-knot spline solver and scale BVP finite differences.
  ([132fb76](https://github.com/Dicklesworthstone/frankenscipy/commit/132fb7689633232bb0d2fad99b5e6a427b31ab36))

### Quadrature

- Adaptive Gauss-Kronrod quadrature module.
  ([85d67a6](https://github.com/Dicklesworthstone/frankenscipy/commit/85d67a695442c36a78aec2d2657e9c0f7abf0380))
- Additional quadrature rules.
  ([8c20565](https://github.com/Dicklesworthstone/frankenscipy/commit/8c20565d76684f4ff0711e23e9b33b19c88341fb))
- `cumulative_trapezoid`.
  ([cdd9049](https://github.com/Dicklesworthstone/frankenscipy/commit/cdd90493839db40074672ecf70e57a27ae870955))
- Rewrite `cumulative_simpson` for correct per-interval output.
  ([8acb62f](https://github.com/Dicklesworthstone/frankenscipy/commit/8acb62f1a266b3239123222d09a2cf2100b8c38a))

### Bug fixes

- Correct DOP853 Butcher tableau coefficients.
  ([6d378cb](https://github.com/Dicklesworthstone/frankenscipy/commit/6d378cb6e964fe586f6232d92aa42848ac7704ac))
- Fix BDF efficiency constants.
  ([01eb649](https://github.com/Dicklesworthstone/frankenscipy/commit/01eb64904284921c3a6789c21f376712c7c7e1a0))

### Performance

- Inline RMS norm to eliminate temporary vector allocations in step-size control.
  ([b867518](https://github.com/Dicklesworthstone/frankenscipy/commit/b86751887c7ac8b82a25231d5d1123beff01cb2f))

---

## Interpolation (`fsci-interpolate`)

New crate bootstrapped 2026-03-14.

### Splines

- Cubic spline interpolation at crate bootstrap.
  ([cdd9049](https://github.com/Dicklesworthstone/frankenscipy/commit/cdd90493839db40074672ecf70e57a27ae870955))
- PCHIP (Piecewise Cubic Hermite Interpolating Polynomial).
  ([f99b570](https://github.com/Dicklesworthstone/frankenscipy/commit/f99b5709d168a41b0e754ca1a54962d55150cf0e))
- BSpline and Akima interpolation.
  ([7a1169e](https://github.com/Dicklesworthstone/frankenscipy/commit/7a1169ed185dd17d8804e2fd8b4331dbbcb2a0e3))
- Spline boundary conditions (clamped, not-a-knot, natural).
  ([7a26704](https://github.com/Dicklesworthstone/frankenscipy/commit/7a26704bda817efd56d73973ee84bff55feea4b6))
- Fix Akima slope computation edge cases.
  ([adb66db](https://github.com/Dicklesworthstone/frankenscipy/commit/adb66db7a32181cb45e599f09b13232f796a26b7))

### Scattered data

- `NearestNDInterpolator` and `griddata` for scattered-data interpolation.
  ([eb6279a](https://github.com/Dicklesworthstone/frankenscipy/commit/eb6279af010b266c503a32ece6ea60525d8207d0))

### NaN handling

- NaN-safe interpolation.
  ([346cc46](https://github.com/Dicklesworthstone/frankenscipy/commit/346cc46d8c350dd0e643a1fd6bcb69c8dd442592))

---

## Optimization (`fsci-opt`)

Scaffolded 2026-02-14; core algorithms landed 2026-03-02.

### Minimizers

- BFGS, Conjugate Gradient (Polak-Ribiere+), and Powell minimizers.
  ([bf634ec](https://github.com/Dicklesworthstone/frankenscipy/commit/bf634ecc1e7da33af4806e5bf11d4523b2e2fddf))
- L-BFGS-B bounded optimization.
  ([2344b16](https://github.com/Dicklesworthstone/frankenscipy/commit/2344b162338f882388090ccdc38881ba41093383))
- Nelder-Mead optimizer.
  ([8c20565](https://github.com/Dicklesworthstone/frankenscipy/commit/8c20565d76684f4ff0711e23e9b33b19c88341fb))
- Newton-CG optimizer.
  ([8a47f61](https://github.com/Dicklesworthstone/frankenscipy/commit/8a47f611571b42f574da016f661c0c1cf494e624))
- Expand minimization algorithms (trust-region variants).
  ([d612bd6](https://github.com/Dicklesworthstone/frankenscipy/commit/d612bd6bb8ff4a433cd56aa8af18f3f56c256d8d))
- Enhance Powell direction update strategy.
  ([93fb998](https://github.com/Dicklesworthstone/frankenscipy/commit/93fb998d3e6e241461ecc63926a3188c82fe9700))
- Fix Powell bracketing.
  ([8acb62f](https://github.com/Dicklesworthstone/frankenscipy/commit/8acb62f1a266b3239123222d09a2cf2100b8c38a))

### Root-finders

- `brentq`, `bisect`, and `ridder` root-finders.
  ([bf634ec](https://github.com/Dicklesworthstone/frankenscipy/commit/bf634ecc1e7da33af4806e5bf11d4523b2e2fddf))
- Reject unsupported initial guesses instead of silently selecting Ridder.
  ([93edeca](https://github.com/Dicklesworthstone/frankenscipy/commit/93edeca6032910b9fd273700e382007a04d67776))

### Curve fitting

- `curve_fit` nonlinear least-squares curve fitting.
  ([9ebb9c2](https://github.com/Dicklesworthstone/frankenscipy/commit/9ebb9c2d4aa02532ce93b82f7bb68f9869955dd1))

### Constraint types

- `Bounds`, `LinearConstraint`, `NonlinearConstraint` types for constrained optimization.
  ([65302a6](https://github.com/Dicklesworthstone/frankenscipy/commit/65302a601dac0b7e8a18aa7c81d15f265dfbeb9e))

### Line search

- Line-search, minimize, and root dispatch scaffolding.
  ([8efb237](https://github.com/Dicklesworthstone/frankenscipy/commit/8efb237020460e5ab0e297fab490ee790c67027d))
- Expand quadrature, sparse linalg, and line search algorithms.
  ([a9010d8](https://github.com/Dicklesworthstone/frankenscipy/commit/a9010d8b657cde4196c8a8351ce21367878425c3))

### Bug fixes

- Correct Newton-CG Jacobian evaluation count.
  ([6e9e2dd](https://github.com/Dicklesworthstone/frankenscipy/commit/6e9e2dd6689df6c8c4a02b49ac18b44bdd681df3))
- Fix numerical stability in optimizer internals.
  ([d8180d9](https://github.com/Dicklesworthstone/frankenscipy/commit/d8180d9d5eb5c458f43fe80b70f0761a47f58908))
- Remove L-BFGS-B dead code.
  ([01eb649](https://github.com/Dicklesworthstone/frankenscipy/commit/01eb64904284921c3a6789c21f376712c7c7e1a0))
- Correct 5 bugs in filter conversions and constraint types.
  ([a03f80d](https://github.com/Dicklesworthstone/frankenscipy/commit/a03f80dcc998f0c0013c321095d23846e85c6393))

### Tests

- Comprehensive minimizer and root-finder test suites with proptest.
  ([be93eab](https://github.com/Dicklesworthstone/frankenscipy/commit/be93eabe6e98d462e43b24c07d227f729b318293))

---

## FFT (`fsci-fft`)

Scaffolded 2026-02-25; core algorithms landed 2026-03-14.

### Core transforms

- Scaffold contract-first FFT module with transforms, plan cache, and helpers.
  ([e490d08](https://github.com/Dicklesworthstone/frankenscipy/commit/e490d083f3eaf96582bd906caa5158277e47689f))
- Cooley-Tukey radix-2 FFT and Bluestein's algorithm for arbitrary lengths.
  ([66a57b0](https://github.com/Dicklesworthstone/frankenscipy/commit/66a57b003711e8f77d47d3987c66bd0fdf8f90bc))
- DCT and IDCT transforms.
  ([8a47f61](https://github.com/Dicklesworthstone/frankenscipy/commit/8a47f611571b42f574da016f661c0c1cf494e624))
- DCT transform variants (DCT-II, DCT-III, DCT-IV).
  ([d7ac09a](https://github.com/Dicklesworthstone/frankenscipy/commit/d7ac09a9bdec6f8bb0d0d0bde8c04fcdf5b2dd0b))

### Multi-dimensional transforms

- N-dimensional real FFT transforms (`rfftn`, `irfftn`) with full normalization mode support.
  ([1256e63](https://github.com/Dicklesworthstone/frankenscipy/commit/1256e63003a5ff05444d15089c2ee4acb89ec09b))

### Analytic signal

- Hilbert transform.
  ([79b6092](https://github.com/Dicklesworthstone/frankenscipy/commit/79b6092ca696bfd185cb5943cc22260753df00ea))

### Performance

- Specialized real FFT for power-of-2 inputs.
  ([ac1f717](https://github.com/Dicklesworthstone/frankenscipy/commit/ac1f717ef96efaa7b69806a5e7b9d8f1ab715202))

### Bug fixes

- Fix Bluestein DFT sign convention.
  ([01eb649](https://github.com/Dicklesworthstone/frankenscipy/commit/01eb64904284921c3a6789c21f376712c7c7e1a0))
- Correct Hilbert transform step function for odd-length inputs.
  ([e5b3f0d](https://github.com/Dicklesworthstone/frankenscipy/commit/e5b3f0d0b15bfd40ba24c88355c2c524532e99e6))
- FFT scaling fixes across forward/inverse transforms.
  ([60c5930](https://github.com/Dicklesworthstone/frankenscipy/commit/60c59301358783bd0c4d6aed6dabd30344685632))
- Numerical corrections across FFT phase/magnitude handling.
  ([09c901c](https://github.com/Dicklesworthstone/frankenscipy/commit/09c901cdeebfa435b389a37b88008fcdfb9f5518))

---

## Signal Processing (`fsci-signal`)

New crate bootstrapped 2026-03-15.

### Filtering

- Signal processing module with filtering and spectral analysis (+462 lines).
  ([e132d4c](https://github.com/Dicklesworthstone/frankenscipy/commit/e132d4cc0394c85152dfdaeb08cda457b25b406c))
- Convolution and window functions (Hann, Hamming, Blackman, Kaiser, etc.).
  ([937daa3](https://github.com/Dicklesworthstone/frankenscipy/commit/937daa3b16611c8eb02e2e05eda7865d20ccbb43))
- `lfilter_zi` and `filtfilt` steady-state initialization, window parameter threading, new window functions.
  ([5a9f78a](https://github.com/Dicklesworthstone/frankenscipy/commit/5a9f78ab51cab8b88801c740a981e3c2ec7d400c))

### SOS (second-order sections)

- `sosfilt`, `sosfiltfilt`, `sosfilt_zi`.
  ([b2815a7](https://github.com/Dicklesworthstone/frankenscipy/commit/b2815a71b44839cde7223b2094f762fb6c793828))

### Filter design

- FIR filter design (`firwin`, `firwin2`, `remez`).
  ([7a1169e](https://github.com/Dicklesworthstone/frankenscipy/commit/7a1169ed185dd17d8804e2fd8b4331dbbcb2a0e3))
- IIR filter families: Butterworth, Chebyshev I/II, Bessel.
  ([346cc46](https://github.com/Dicklesworthstone/frankenscipy/commit/346cc46d8c350dd0e643a1fd6bcb69c8dd442592))
- Elliptic (Cauer) IIR filter design.
  ([e070b0e](https://github.com/Dicklesworthstone/frankenscipy/commit/e070b0ec22e506acb1ca33ea2b5c3eafc2e018e2))

### Filter representation conversions

- `tf2zpk`, `zpk2tf`, `tf2sos`, `sos2tf`, `zpk2sos`, `sos2zpk`.
  ([fc958ff](https://github.com/Dicklesworthstone/frankenscipy/commit/fc958ff69b8dee7d31bd99b721d8a37e59af5eee))

### Frequency response analysis

- `freqz`, `freqs`, `group_delay`.
  ([935983e](https://github.com/Dicklesworthstone/frankenscipy/commit/935983e1f90fa37996f23b7db0820ab432b6273f))

### Spectral estimation

- Welch's method for power spectral density estimation.
  ([f957c7d](https://github.com/Dicklesworthstone/frankenscipy/commit/f957c7d5d22cdd73034d4b7052deaa9ef1883299))
- Windowed periodogram.
  ([65b3ce2](https://github.com/Dicklesworthstone/frankenscipy/commit/65b3ce2f9520064f6629dcefa1a29314ef007a26))

### Peak detection

- `find_peaks` peak detection.
  ([47b6a99](https://github.com/Dicklesworthstone/frankenscipy/commit/47b6a99e7b9ae9120d38d304a9913c0fbffe20f8))

### Bug fixes

- Fix unreachable early-return path in `tf2zpk` for all-zero numerator.
  ([8bd95b1](https://github.com/Dicklesworthstone/frankenscipy/commit/8bd95b12e64e4bbbfa49602a0d1db3301215b058))
- Clean up messy derivation comments in `group_delay`.
  ([f2325ad](https://github.com/Dicklesworthstone/frankenscipy/commit/f2325ad84d276b78ea01cf9e49bab0d8f147086b))
- Fix numerical stability in filter coefficient computation.
  ([d8180d9](https://github.com/Dicklesworthstone/frankenscipy/commit/d8180d9d5eb5c458f43fe80b70f0761a47f58908))
- Correct 5 bugs in filter conversions.
  ([a03f80d](https://github.com/Dicklesworthstone/frankenscipy/commit/a03f80dcc998f0c0013c321095d23846e85c6393))

---

## Spatial Algorithms (`fsci-spatial`)

New crate bootstrapped 2026-03-15.

### Nearest-neighbor search

- KDTree for nearest-neighbor queries.
  ([47b6a99](https://github.com/Dicklesworthstone/frankenscipy/commit/47b6a99e7b9ae9120d38d304a9913c0fbffe20f8))

### Distance computation

- `pdist` pairwise distance computation.
  ([f957c7d](https://github.com/Dicklesworthstone/frankenscipy/commit/f957c7d5d22cdd73034d4b7052deaa9ef1883299))

---

## Special Functions (`fsci-special`)

Scaffolded 2026-02-13; contract-first dispatch plans added 2026-03-02; implementations landed 2026-03-03 onward.

### Gamma family

- `gamma`, `lgamma`, `digamma`, `beta`, `betaln`.
  ([d6467d2](https://github.com/Dicklesworthstone/frankenscipy/commit/d6467d2f1fad59d37d06fbbdc2eff1f9d9d3ce87))
- Regularized incomplete gamma function.
  ([c92b075](https://github.com/Dicklesworthstone/frankenscipy/commit/c92b0758de912855c6e67c7c542b3679e228538e))
- Rewrite `gammaln` and add NaN propagation to all gamma convenience functions.
  ([4337552](https://github.com/Dicklesworthstone/frankenscipy/commit/4337552cb263c2e483f36278d96c6ed64823e06c))
- Combinatorial functions.
  ([8c20565](https://github.com/Dicklesworthstone/frankenscipy/commit/8c20565d76684f4ff0711e23e9b33b19c88341fb))

### Error functions

- `erf`, `erfc`, `erfinv`.
  ([d6467d2](https://github.com/Dicklesworthstone/frankenscipy/commit/d6467d2f1fad59d37d06fbbdc2eff1f9d9d3ce87))
- erf API cleanup.
  ([5a9f78a](https://github.com/Dicklesworthstone/frankenscipy/commit/5a9f78ab51cab8b88801c740a981e3c2ec7d400c))

### Bessel functions

- Integer-order Bessel functions: J0, J1, Jn, Y0, Y1, Yn.
  ([c92b075](https://github.com/Dicklesworthstone/frankenscipy/commit/c92b0758de912855c6e67c7c542b3679e228538e))
- Improve Bessel function accuracy and reorganize imports.
  ([3177df2](https://github.com/Dicklesworthstone/frankenscipy/commit/3177df2748f2011390ff30309f77b4abe403c0dd))

### Airy functions

- Airy functions (`airy`, `ai`, `bi`).
  ([8c20565](https://github.com/Dicklesworthstone/frankenscipy/commit/8c20565d76684f4ff0711e23e9b33b19c88341fb))
- Fix Airy function phase computation.
  ([01eb649](https://github.com/Dicklesworthstone/frankenscipy/commit/01eb64904284921c3a6789c21f376712c7c7e1a0))

### Hypergeometric functions

- `hyp1f1` and `hyp2f1` series evaluation.
  ([32d737c](https://github.com/Dicklesworthstone/frankenscipy/commit/32d737c4890a1879305f90451012f3585201861e))

### Elliptic functions

- Complete elliptic integrals (`ellipk`, `ellipe`) and Jacobi elliptic functions (`ellipj`).
  ([9ebb9c2](https://github.com/Dicklesworthstone/frankenscipy/commit/9ebb9c2d4aa02532ce93b82f7bb68f9869955dd1))

### Zeta function

- Riemann zeta function.
  ([f480b75](https://github.com/Dicklesworthstone/frankenscipy/commit/f480b752d6c70985a9fce1f0c97337f664be4047))

### Orthogonal polynomials

- Legendre, Hermite, Laguerre, Chebyshev, Jacobi polynomial evaluation.
  ([0210ac2](https://github.com/Dicklesworthstone/frankenscipy/commit/0210ac29f792a1fec599bc3370eb4bb4b6100a2a))

### NaN propagation

- NaN propagation for entropy functions.
  ([1614362](https://github.com/Dicklesworthstone/frankenscipy/commit/1614362889cc570c255b6d9e063a8398c1261c77))
- NaN propagation overhaul for all convenience functions.
  ([4337552](https://github.com/Dicklesworthstone/frankenscipy/commit/4337552cb263c2e483f36278d96c6ed64823e06c))

---

## Statistics (`fsci-stats`)

New crate bootstrapped 2026-03-14.

### Continuous distributions

- Normal, Uniform, Chi-Squared, Student-t at crate bootstrap.
  ([b140aeb](https://github.com/Dicklesworthstone/frankenscipy/commit/b140aeb14acd169421e412b6245067496fa82383))
- Exponential distribution.
  ([6d378cb](https://github.com/Dicklesworthstone/frankenscipy/commit/6d378cb6e964fe586f6232d92aa42848ac7704ac))
- Gamma and Poisson distributions.
  ([937daa3](https://github.com/Dicklesworthstone/frankenscipy/commit/937daa3b16611c8eb02e2e05eda7865d20ccbb43))
- Weibull and Lognormal distributions.
  ([47b6a99](https://github.com/Dicklesworthstone/frankenscipy/commit/47b6a99e7b9ae9120d38d304a9913c0fbffe20f8))
- F and Beta distributions.
  ([8a47f61](https://github.com/Dicklesworthstone/frankenscipy/commit/8a47f611571b42f574da016f661c0c1cf494e624))
- Trait-level `ppf` (percent point function) for all distributions.
  ([f99b570](https://github.com/Dicklesworthstone/frankenscipy/commit/f99b5709d168a41b0e754ca1a54962d55150cf0e))
- Survival function (`sf`) for Normal distribution.
  ([0b83dae](https://github.com/Dicklesworthstone/frankenscipy/commit/0b83daea069f06295cacb3fea6c9ee05170ab957))

### Discrete distributions

- Binomial, Bernoulli, Geometric, NegativeBinomial, Hypergeometric.
  ([c4568b0](https://github.com/Dicklesworthstone/frankenscipy/commit/c4568b01a83cd57324e8b2aac952079c5d06a2af))
- Fix two bugs in Hypergeometric distribution (PMF normalization, support range).
  ([aa3283b](https://github.com/Dicklesworthstone/frankenscipy/commit/aa3283b6ed35cecc92ff443719ecc69480b83485))

### Hypothesis testing

- t-test, chi-squared tests.
  ([42abbcc](https://github.com/Dicklesworthstone/frankenscipy/commit/42abbcc5d48cac81b0ff19b9c2a2b76a75515dbf))
- Non-parametric and ANOVA tests: `f_oneway`, `mannwhitneyu`, `wilcoxon`, `kruskal`, `ranksums`.
  ([5b48848](https://github.com/Dicklesworthstone/frankenscipy/commit/5b488482dc331a188f108d0ff677fe35c6fac15e))
- Goodness-of-fit tests (Kolmogorov-Smirnov, chi-squared, Shapiro-Wilk).
  ([7a1169e](https://github.com/Dicklesworthstone/frankenscipy/commit/7a1169ed185dd17d8804e2fd8b4331dbbcb2a0e3))

### Descriptive statistics and regression

- Descriptive statistics, correlation, and regression functions (+488 lines).
  ([004a3e9](https://github.com/Dicklesworthstone/frankenscipy/commit/004a3e90926522c6b1d523c8957e403804ac7fe8))
- `linregress` linear regression.
  ([f957c7d](https://github.com/Dicklesworthstone/frankenscipy/commit/f957c7d5d22cdd73034d4b7052deaa9ef1883299))
- Summary statistics and special-function convenience wrappers.
  ([23c5ff3](https://github.com/Dicklesworthstone/frankenscipy/commit/23c5ff3e19a15fd7f88c5be85fd7a895b1c60a6c))
- New statistical routines (entropy, additional edge-case coverage).
  ([1614362](https://github.com/Dicklesworthstone/frankenscipy/commit/1614362889cc570c255b6d9e063a8398c1261c77))

---

## Array API (`fsci-arrayapi`)

Scaffolded 2026-03-02; implemented 2026-03-03.

### Core backend

- Scaffold contract-first Array API foundation crate with module structure and type stubs.
  ([8e773d3](https://github.com/Dicklesworthstone/frankenscipy/commit/8e773d37b60c79536db72ed6283db5b0947ec779))
- `CoreArrayBackend` with broadcast, indexing, reduction, and type promotion.
  ([1b5700e](https://github.com/Dicklesworthstone/frankenscipy/commit/1b5700e83014c3bb809c0ae1137342d3a43f34e9))
- Validate `arange` inputs.
  ([52d0818](https://github.com/Dicklesworthstone/frankenscipy/commit/52d081846f940cb940067a8d096f80d562a93899))

### Performance

- Replace per-element unravel/ravel with incremental coordinate advancement in broadcast.
  ([2838ba6](https://github.com/Dicklesworthstone/frankenscipy/commit/2838ba6ae61cbec677c17700fd76fce002256b2c))

### Tests and benchmarks

- Comprehensive unit tests and proptest suites for all Array API modules.
  ([a2eabd2](https://github.com/Dicklesworthstone/frankenscipy/commit/a2eabd2e75885954948f02173aa40fee6e735bde))
- Criterion benchmark suite for core Array API operations.
  ([274f0ec](https://github.com/Dicklesworthstone/frankenscipy/commit/274f0ec8f4390d2545bbfc7e2e9044939e13aee6))

---

## Conformance Infrastructure (`fsci-conformance`)

Landed 2026-02-13 with FSCI-P2C-001 and FSCI-P2C-002; expanded continuously.

### Conformance packets

| Packet ID | Domain | Introduced | Key commit |
|---|---|---|---|
| FSCI-P2C-001 | Tolerance validation | 2026-02-13 | [55f1ee9](https://github.com/Dicklesworthstone/frankenscipy/commit/55f1ee94577f2b67e6242b93ac459998a47aa797) |
| FSCI-P2C-002 | Dense linear algebra | 2026-02-13 | [55f1ee9](https://github.com/Dicklesworthstone/frankenscipy/commit/55f1ee94577f2b67e6242b93ac459998a47aa797) |
| FSCI-P2C-003 | Optimization | 2026-03-03 | [43bb37e](https://github.com/Dicklesworthstone/frankenscipy/commit/43bb37e4859dc1c560e764bc8c21657bd2493a7d) |
| FSCI-P2C-004 | Sparse operations | 2026-03-13 | [72845e3](https://github.com/Dicklesworthstone/frankenscipy/commit/72845e30f09671ea8d104fa2c37eb358dbc6ce86) |
| FSCI-P2C-005 | FFT core | 2026-03-04 | [89a2f5c](https://github.com/Dicklesworthstone/frankenscipy/commit/89a2f5cc35837fe31ec68968cd5b21d593d1815e) |
| FSCI-P2C-006 | Special functions | 2026-03-03 | [1b5700e](https://github.com/Dicklesworthstone/frankenscipy/commit/1b5700e83014c3bb809c0ae1137342d3a43f34e9) |
| FSCI-P2C-007 | Array API | 2026-03-03 | [4f539b6](https://github.com/Dicklesworthstone/frankenscipy/commit/4f539b67251f69123f319f5ef0f7d1bcfd48d7ca) |
| FSCI-P2C-008 | Runtime CASP | 2026-03-13 | [72845e3](https://github.com/Dicklesworthstone/frankenscipy/commit/72845e30f09671ea8d104fa2c37eb358dbc6ce86) |

### Harness and infrastructure

- Differential conformance harness with artifact governance.
  ([9b921bd](https://github.com/Dicklesworthstone/frankenscipy/commit/9b921bd69c71748e24d8c9819964e7e1c744d5f7))
- RaptorQ + decode-proof artifact generation.
  ([55f1ee9](https://github.com/Dicklesworthstone/frankenscipy/commit/55f1ee94577f2b67e6242b93ac459998a47aa797))
- Fix RaptorQ sidecar generation to serialize from JSON instead of pre-serialization bytes.
  ([89fc93a](https://github.com/Dicklesworthstone/frankenscipy/commit/89fc93a2981e6ad18da1b81ddb0c298428f34f0b))
- Schema validation tests and runtime property tests.
  ([e6666dd](https://github.com/Dicklesworthstone/frankenscipy/commit/e6666dd05c019d74c74f84d16848e5b2f83531f4))

### E2E test suites

- E2E linear algebra conformance test suite.
  ([9ade3fb](https://github.com/Dicklesworthstone/frankenscipy/commit/9ade3fbb71a7879070cfbbdc2eda8799d41bd459))
- E2E scenario tests for FFT backend routing (FSCI-P2C-005).
  ([89a2f5c](https://github.com/Dicklesworthstone/frankenscipy/commit/89a2f5cc35837fe31ec68968cd5b21d593d1815e))
- E2E and evidence conformance test suites across all subsystems.
  ([0aac03a](https://github.com/Dicklesworthstone/frankenscipy/commit/0aac03a222226def07e8adc5f73d7c053eace2d5))
- P2C-002 evidence/perf/RaptorQ test suites and expanded benchmark coverage.
  ([b7d7614](https://github.com/Dicklesworthstone/frankenscipy/commit/b7d7614c7b9f7d79877129c37415f9a4953624cd))
- IVP solver performance profiling harness (P2C-001-H).
  ([8f09e95](https://github.com/Dicklesworthstone/frankenscipy/commit/8f09e95d99aa70731e302fb1734a001a14d16785))
- FFT differentiation tests and expanded DCT transform coverage.
  ([493b3be](https://github.com/Dicklesworthstone/frankenscipy/commit/493b3bea7200da15bf1bdf4bb0fe5421de3906bf))

### Quality gates and CI

- Conformance quality gates, CI pipeline, benchmarks, and forensic analysis tooling.
  ([f1e61c0](https://github.com/Dicklesworthstone/frankenscipy/commit/f1e61c0b3b71b67dbd0492a1013569762dd251f5))
- Expand E2E test coverage with new benchmarks and IVP integration API tests.
  ([72f8aa1](https://github.com/Dicklesworthstone/frankenscipy/commit/72f8aa1921c3416f59ca8149f9c780d5d0c35fca))
- Update test infrastructure, parity reports, and mutex handling for concurrent conformance runs.
  ([9a1c403](https://github.com/Dicklesworthstone/frankenscipy/commit/9a1c40338712c5c9e2c223e85c00e95ef23a3958))
- Regenerate P2C-001, P2C-002, P2C-007 evidence artifacts with corrected checksums.
  ([b276e59](https://github.com/Dicklesworthstone/frankenscipy/commit/b276e59f0a32dfbea580845758b78d04742c9721))

### Python oracle

- Python oracle capture script for SciPy-vs-Rust comparison.
  ([55f1ee9](https://github.com/Dicklesworthstone/frankenscipy/commit/55f1ee94577f2b67e6242b93ac459998a47aa797))

### Interactive dashboard

- `ftui` dashboard binary for artifact navigation.
  ([55f1ee9](https://github.com/Dicklesworthstone/frankenscipy/commit/55f1ee94577f2b67e6242b93ac459998a47aa797))

---

## Runtime and CASP (`fsci-runtime`)

Landed 2026-02-13.

- CASP runtime with execution-path tracing and schema validation.
  ([55f1ee9](https://github.com/Dicklesworthstone/frankenscipy/commit/55f1ee94577f2b67e6242b93ac459998a47aa797))
- Runtime property tests.
  ([e6666dd](https://github.com/Dicklesworthstone/frankenscipy/commit/e6666dd05c019d74c74f84d16848e5b2f83531f4))
- Expand runtime architecture with execution path tracing documentation.
  ([f78ada6](https://github.com/Dicklesworthstone/frankenscipy/commit/f78ada6f2596589e25cbacbb40ef41c3591f24e6))

---

## Cross-cutting numerical corrections

These commits span multiple modules and fix numerical issues that do not belong to a single domain.

- Improve numerical correctness across FFT, special functions, stats, and optimizer.
  ([1d9133d](https://github.com/Dicklesworthstone/frankenscipy/commit/1d9133d715ede43d9cf8d863fe1ceb9ae41eb531))
- Numerical corrections across FFT, signal, optimizer, and special functions.
  ([09c901c](https://github.com/Dicklesworthstone/frankenscipy/commit/09c901cdeebfa435b389a37b88008fcdfb9f5518))
- Correct 5 bugs found during code review across multiple crates.
  ([3411647](https://github.com/Dicklesworthstone/frankenscipy/commit/3411647f3f82ef05fb9c030fb131623ab72ed60f))
- Apply cargo fmt and resolve clippy warnings across workspace.
  ([335509f](https://github.com/Dicklesworthstone/frankenscipy/commit/335509ff4a3dd8abb7b0d44ab7440cec45dfc86a))
- ODE event handling overhaul, interpolation simplification, and multi-module improvements.
  ([3fc730e](https://github.com/Dicklesworthstone/frankenscipy/commit/3fc730e606529c4259f21522743b5c66cb1af8bb))
- Multi-module expansions across integrate, FFT, signal, sparse, spatial, and stats.
  ([60c5930](https://github.com/Dicklesworthstone/frankenscipy/commit/60c59301358783bd0c4d6aed6dabd30344685632))
- Large cross-module feature drop spanning stats, signal, linalg, optimization, interpolation, spatial, and special.
  ([9026a01](https://github.com/Dicklesworthstone/frankenscipy/commit/9026a01daa0d8936f2afc21b93919ba870136d09))

---

## Project infrastructure and dependencies

### License

- MIT license with OpenAI/Anthropic Rider.
  ([2888129](https://github.com/Dicklesworthstone/frankenscipy/commit/2888129fa7f3fe95183a84d487c06fc999490147))

### Branding

- GitHub social preview image (1280x640).
  ([47b8782](https://github.com/Dicklesworthstone/frankenscipy/commit/47b87822815fefe3a31521ba8eafce08efd3856d))

### Dependency management

- Switch `asupersync` and `ftui` from local path dependencies to crates.io.
  ([b7643af](https://github.com/Dicklesworthstone/frankenscipy/commit/b7643af85e6d317192fe99e7c7662e082d314f55),
   [9c1d6d8](https://github.com/Dicklesworthstone/frankenscipy/commit/9c1d6d805a5665e421fa1c5a171731aa3383d1b4))
- Pin `asupersync` to v0.2.0.
  ([1118492](https://github.com/Dicklesworthstone/frankenscipy/commit/111849231ccfb87a27c55052f415c6b0b310fac8))

### Documentation

- Project charter, comprehensive spec, porting plan, and architecture docs at initial commit.
  ([55f1ee9](https://github.com/Dicklesworthstone/frankenscipy/commit/55f1ee94577f2b67e6242b93ac459998a47aa797))
- Add cass (Cross-Agent Session Search) tool reference to AGENTS.md.
  ([c95e977](https://github.com/Dicklesworthstone/frankenscipy/commit/c95e9777607b448a0928aa0e4477b14375d1d1b7))

---

## Statistics

- **Total commits**: 3,331 (as of 2026-05-16, on `main`)
- **Rust source**: ~140,000 lines (`crates/*/src/`) across 19 workspace crates
- **Tests and harnesses**: ~370,000 additional lines under `crates/*/src/bin/`,
  `crates/*/tests/`, and `crates/fsci-conformance/tests/` (~765 integration
  test files in the conformance harness alone)
- **Beads tracked**: 2,404 closed, 59 open as of 2026-05-16
- **First commit**: 2026-02-13 ([55f1ee9](https://github.com/Dicklesworthstone/frankenscipy/commit/55f1ee94577f2b67e6242b93ac459998a47aa797))
- **Crate creation timeline**:
  - 2026-02-13: 9 crates at initial commit (`fsci-linalg`, `fsci-sparse`, `fsci-integrate`, `fsci-opt`, `fsci-fft`, `fsci-special`, `fsci-arrayapi`, `fsci-conformance`, `fsci-runtime`)
  - 2026-03-14: `fsci-interpolate`, `fsci-stats`
  - 2026-03-15: `fsci-signal`, `fsci-spatial`
  - 2026-03-23: `fsci-ndimage`, `fsci-io`
  - 2026-03-24: `fsci-cluster`, `fsci-constants`
  - 2026-05-03: `fsci-odr`, `fsci-datasets`
- **No tagged releases yet** -- workspace version is `0.1.0`

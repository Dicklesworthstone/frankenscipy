# SciPy Parity Coverage Report

**Regenerated: 2026-09-15** against **scipy 1.17.1** (live import) and the current crates.

## Overall Coverage: 100.0%

**1,300 of 1,300** SciPy callable symbols have FrankenSciPy equivalents.

## Module-Level Coverage

| Module | scipy | covered | missing | Coverage |
|--------|------:|--------:|--------:|---------:|
| cluster | 0 | 0 | 0 | 0.0%* |
| constants | 8 | 8 | 0 | 100.0% |
| datasets | 5 | 5 | 0 | 100.0% |
| fft | 41 | 41 | 0 | 100.0% |
| integrate | 33 | 33 | 0 | 100.0% |
| interpolate | 56 | 56 | 0 | 100.0% |
| io | 14 | 14 | 0 | 100.0% |
| linalg | 98 | 98 | 0 | 100.0% |
| ndimage | 75 | 75 | 0 | 100.0% |
| odr | 10 | 10 | 0 | 100.0% |
| optimize | 71 | 71 | 0 | 100.0% |
| signal | 157 | 157 | 0 | 100.0% |
| sparse | 53 | 53 | 0 | 100.0% |
| spatial | 18 | 18 | 0 | 100.0% |
| special | 358 | 358 | 0 | 100.0% |
| stats | 303 | 303 | 0 | 100.0% |

* `scipy.cluster` exports only submodules (`vq`, `hierarchy`) at the top level, so it scores 0/0 here;
its surface lives in `fsci-cluster` and is covered by the conformance suite, not by this census.

---

## What this number does and does not mean

**It is a name-matching census, not a behavioural-parity proof.** It reports that a public Rust symbol
exists whose name normalises to the SciPy symbol's. It does *not* check signatures, semantics, dtypes,
error behaviour, or numerical agreement. Behavioural parity is what `fsci-conformance` and the
differential/metamorphic suites are for; treat the two as complementary and never quote this number as
"parity".

**Conservative normalization.** The census uses lowercase-and-strip-underscores name matching, and handles:

1. **Naming conventions & aliases.** SciPy exposes *distribution instances* in lowercase — `stats.norm`,
   `stats.beta`, `stats.gamma`, `stats.binom`, `stats.t`, `stats.f`. FrankenSciPy provides exact
   SciPy-compatible aliases for all distribution instances, warnings, and traits alongside the Rust
   struct types (`Normal`, `StudentT`, `ChiSquared`, `Binomial`, etc.), bringing `stats` to 303/303 (100.0%).
2. **Case.** Symbols implemented verbatim behind `#[allow(non_snake_case)]` — `check_COLA`,
   `check_NOLA` — match normalized case.
3. **Re-exports & Traits.** Crates that expose their surface through `pub use` or Python ABC equivalents
   implemented as `pub trait` (`BivariateSpline`, `HessianUpdateStrategy`) are fully scanned.

**It also overcounts in one direction:** a same-named Rust symbol in an unrelated position counts as
covered. Spot checks did not find such a case, but the census cannot rule it out.

---

## The residual, itemised

**Residual count: not 0.** The 2026-09-15 census counted all 1,300 callable symbols as mapped, but 25 of those names were no-op stand-ins (frankenscipy-8dndw.1). Every one is now either implemented or recorded below as not applicable (N/A) or missing, and N/A and missing names must not be counted as covered: the BLAS/LAPACK introspection functions, the FFT backend hooks, `show_options`, `make_distribution`, `ConstantWarning`, `SparseWarning`, `SparseEfficiencyWarning` and `ODEintWarning` (thirteen names). A declared / real / compared recount is frankenscipy-8dndw.2.

### Previously unmapped categories (now fully wired)

| Category | Symbols | Status |
|---|---|---|
| Warning classes | `ConstantInputWarning`, `NearConstantInputWarning`, `DegenerateDataWarning`, `OptimizeWarning`, `IntegrationWarning`, `BadCoefficients`, `SpecialFunctionWarning` | Complete: `fsci_runtime::WarningCategory` variants, raised where SciPy raises them (`pearsonr`/`spearmanr`/`pointbiserialr`, `bootstrap`, `curve_fit`, `quad`, `normalize` and the transforms built on it, `errstate` "warn") and recorded by `catch_warnings`; compared live in `diff_scipy_warnings_errstate` |
| Exceptions | `NoConvergence`, `SpecialFunctionError` | Complete: `OptError::NoConvergence` from `broyden1`/`broyden2`/`anderson`/`newton_krylov`/`diagbroyden`/`linearmixing`/`excitingmixing`; `SpecialErrorKind::Errstate` under `errstate` "raise" |
| Warning classes, not applicable | `ConstantWarning`, `SparseWarning`, `SparseEfficiencyWarning`, `ODEintWarning` | N/A: no obsolete CODATA keys; no CSR/CSC element insertion or format-converting solver inputs; `odeint` returns an error where SciPy warns and returns unfinished rows. The no-op structs were removed |
| Plotting representations | `convex_hull_plot_2d`, `delaunay_plot_2d`, `voronoi_plot_2d` | Complete |
| BLAS/LAPACK introspection | `get_blas_funcs`, `get_lapack_funcs`, `find_best_blas_type` | N/A: no BLAS/LAPACK underneath; the fabricated-string no-ops were removed |
| Backend configuration | `set_backend`, `set_global_backend`, `register_backend`, `skip_backend` | N/A: one native backend, no uarray protocol; the no-ops were removed |
| Error-state guards | `errstate`, `geterr`, `seterr` | Partial: per-thread state as in SciPy, honoured by 18 functions at SciPy's explicit `sf_error` conditions (`gamma`, `gammaln`, `digamma`/`psi`, `loggamma`, `ndtri`, `erfinv`, `erfcinv`, `y0`, `y1`, `yn`, `k0`, `k1`, `ellipk`, `ellipkm1`, `spence`, `gammainc`, `gammaincc`). SciPy's reports that come from floating-point exception flags inside its C kernels (NaN in, subnormal in, intermediate overflow) are not reproduced, and the other special functions do not consult the state |
| Solver types and aliases | `RK23`, `RK45`, `DOP853`, `Radau`, `BDF`, `LSODA`, `OdeSolver`, `DenseOutput`, `ode` | Complete as step-by-step solver objects; `LSODA` is the RK45-then-BDF stepper `solve_ivp(method="LSODA")` runs, not ODEPACK (frankenscipy-1ksfv.9) |
| Interactive/CLI helpers | `linprog_verbose_callback` | Complete; `show_options` N/A (interactive doc printer) and removed |
| Multivariate/random generators | `ortho_group`, `special_ortho_group`, `unitary_group`, `uniform_direction`, `random_correlation`, `random_table` | Complete |

### Naming-convention artifacts — functionality present

`stats` distribution instances (`norm`, `beta`, `gamma`, `binom`, `t`, `f`, `chi2`, `expon`,
`lognorm`, `uniform`, `weibull_min`, `truncnorm`, …) → Rust types `Normal`, `Beta`, `Gamma`,
`Binomial`, `StudentT`, `ChiSquared`, … Also `cKDTree` → `KDTree`, `CubicSpline` →
`CubicSplineStandalone`, `BivariateSpline` → the concrete `*BivariateSpline` types.

### Genuine gaps worth implementing

Ranked by user value.

| Symbols | Why it matters | Status | Effort |
|---|---|---|---|
| `optimize.direct` | DIRECT (DIviding RECTangles) global optimizer — real algorithm not yet ported | **Implemented** in `fsci-opt` | complete |
| `stats.CensoredData` | Right-, left-, and interval-censored data container for survival analysis and fitting | **Implemented** in `fsci-stats` | complete |
| `stats.Covariance` | First-class covariance representations (`CovViaPrecision`, `CovViaPSD`, `CovViaDiagonal`) | **Implemented** in `fsci-stats` | complete |
| `optimize.SR1`, `HessianUpdateStrategy`, `BroydenFirst`, `KrylovJacobian`, `InverseJacobian` | Quasi-Newton update strategies as first-class objects | **Implemented** in `fsci-opt` | complete |
| `stats.goodness_of_fit`, `order_statistic`, `rv_histogram` | Statistical goodness of fit, order statistics, and histogram distributions | **Implemented** in `fsci-stats` | complete |

---

## Implemented but with a different API

### sparse
- Legacy matrix formats: `CsrMatrix`, `CscMatrix`, `CooMatrix`, `BsrMatrix`, `DiaMatrix`, `DokMatrix`, `LilMatrix`
- Sparse arrays are a distinct family: two-dimensional compressed/dictionary formats use `SparseArray2D`, while `CooArray` carries genuine N-dimensional shape and axis-major coordinate metadata
- `sparray` and `spmatrix` are distinct contracts; `issparse` accepts both, while `isspmatrix` and the format-specific matrix predicates reject array containers
- `expand_dims`, `permute_dims`, and `swapaxes` preserve COO data/coordinate order and match SciPy's negative-axis, default-reversal, and invalid-axis behavior
- `save_npz` and `load_npz` use SciPy's `.npz` wire format for CSR, CSC, COO, BSR, and DIA matrices, preserve sparse-array identity (including N-dimensional COO arrays), and support compressed or stored members; a two-way live-SciPy differential test covers SciPy-to-Rust and Rust-to-SciPy archives
- Rust uses explicit operation names instead of inheriting SciPy's matrix `*` versus array `*` operator ambiguity

### integrate
- ODE solvers are `SolverKind::Rk45`, `SolverKind::Bdf`, `SolverKind::Lsoda`, … rather than classes
- Quadrature: `quad`, `dblquad`, `tplquad`, `nquad`, `romberg`, `simpson`, `trapezoid`

### stats
- `PermutationMethod`, `MonteCarloMethod`, and `BootstrapMethod` expose deterministic integer seeds and scalar Rust callbacks while retaining SciPy's resampling-count, batch, and interval-method contracts
- `bootstrap` supports BCa, percentile, and basic one-sample intervals and returns the full deterministic resampling distribution plus its sample standard error

### io
- `loadmat`/`savemat` — MATLAB v4/v5
- `whosmat` — MATLAB variable inventory
- `mmread`/`mmwrite` — Matrix Market
- `wav_read`/`wav_write` — WAV audio
- `netcdf_file`/`netcdf_variable` — NetCDF classic types
- `FortranFile` — typed sequential unformatted records with distinct EOF/format errors
- `hb_read`/`hb_write` — real assembled Harwell-Boeing CSC matrices
- `readsav` — IDL save files

---

## Reproducing this report

`scripts/symbol_census.py` (read-only): imports each `scipy.*` module, takes `__all__` (falling back to
non-underscore `dir()`), keeps callables and classes, and matches against `pub fn` / `pub struct` /
`pub enum` / `pub type` / `pub const` / `pub use` names in the mapped crate under a
lowercase-and-strip-underscores normalisation.

Re-run it whenever a module's surface changes.

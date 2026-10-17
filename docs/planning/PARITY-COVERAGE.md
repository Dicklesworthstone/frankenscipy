# SciPy Parity Coverage Report

**Regenerated: 2026-10-14** via `scripts/symbol_census.py` against **scipy 1.17.1 / numpy 2.4.3** (live import) and `docs/planning/census_dispositions.toml`.

## Overall Census Summary (33 SciPy Modules)

Across **1,555** public SciPy symbols (**16** not applicable to a native Rust library, **1,539** applicable):

- **Declared (`1,495 / 1,539`, `97.1%`):** A public Rust item (`fn`, `struct`, `enum`, `type`, `const`, `trait`, or `pub use`) matches the SciPy symbol name (`exact: 1,289`, `case_fold: 131`, `reexport: 13`, `type_alias: 60`, `trait: 2`).
- **Real (`1,525 / 1,539`, `99.1%`):** Implemented with non-placeholder semantics (including differently-named Rust APIs recorded in `docs/planning/census_dispositions.toml`; **14** symbols remain missing).
- **Compared (`891 / 1,539`, `57.9%`):** Exercised in a `crates/fsci-conformance/tests/diff_*.rs` live-SciPy differential test.

## Module-Level Coverage (`scripts/symbol_census.py`)

| Module | Crate | SciPy | N/A | Applicable | Declared | Real | Compared | Real % | Compared % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `scipy.cluster.hierarchy` | `fsci-cluster` | 32 | 1 | 31 | 31 | 31 | 10 | 100.0% | 32.3% |
| `scipy.cluster.vq` | `fsci-cluster` | 4 | 0 | 4 | 4 | 4 | 2 | 100.0% | 50.0% |
| `scipy.constants` | `fsci-constants` | 8 | 1 | 7 | 7 | 7 | 5 | 100.0% | 71.4% |
| `scipy.datasets` | `fsci-datasets` | 5 | 0 | 5 | 5 | 5 | 0 | 100.0% | 0.0% |
| `scipy.differentiate` | `fsci-opt` | 3 | 0 | 3 | 3 | 3 | 0 | 100.0% | 0.0% |
| `scipy.fft` | `fsci-fft` | 41 | 4 | 37 | 37 | 37 | 20 | 100.0% | 54.1% |
| `scipy.fftpack` | `fsci-fft` | 10 | 0 | 10 | 10 | 10 | 10 | 100.0% | 100.0% |
| `scipy.integrate` | `fsci-integrate` | 33 | 1 | 32 | 31 | 32 | 14 | 100.0% | 43.8% |
| `scipy.interpolate` | `fsci-interpolate` | 56 | 0 | 56 | 56 | 56 | 14 | 100.0% | 25.0% |
| `scipy.io` | `fsci-io` | 14 | 0 | 14 | 14 | 14 | 3 | 100.0% | 21.4% |
| `scipy.io.arff` | `fsci-io` | 4 | 0 | 4 | 0 | 2 | 0 | 50.0% | 0.0% |
| `scipy.io.matlab` | `fsci-io` | 10 | 0 | 10 | 3 | 6 | 4 | 60.0% | 40.0% |
| `scipy.io.wavfile` | `fsci-io` | 3 | 0 | 3 | 0 | 2 | 0 | 66.7% | 0.0% |
| `scipy.linalg` | `fsci-linalg` | 98 | 3 | 95 | 95 | 95 | 71 | 100.0% | 74.7% |
| `scipy.linalg.interpolative` | `fsci-linalg` | 8 | 0 | 8 | 8 | 8 | 7 | 100.0% | 87.5% |
| `scipy.ndimage` | `fsci-ndimage` | 75 | 0 | 75 | 75 | 75 | 39 | 100.0% | 52.0% |
| `scipy.odr` | `fsci-odr` | 10 | 0 | 10 | 10 | 10 | 2 | 100.0% | 20.0% |
| `scipy.optimize` | `fsci-opt` | 71 | 1 | 70 | 68 | 70 | 29 | 100.0% | 41.4% |
| `scipy.optimize.elementwise` | `fsci-opt` | 4 | 0 | 4 | 4 | 4 | 4 | 100.0% | 100.0% |
| `scipy.signal` | `fsci-signal` | 157 | 0 | 157 | 153 | 157 | 77 | 100.0% | 49.0% |
| `scipy.signal.windows` | `fsci-signal` | 25 | 0 | 25 | 22 | 25 | 9 | 100.0% | 36.0% |
| `scipy.sparse` | `fsci-sparse` | 53 | 2 | 51 | 51 | 51 | 24 | 100.0% | 47.1% |
| `scipy.sparse.csgraph` | `fsci-sparse` | 26 | 0 | 26 | 25 | 26 | 24 | 100.0% | 92.3% |
| `scipy.sparse.linalg` | `fsci-sparse` | 38 | 1 | 37 | 32 | 35 | 29 | 94.6% | 78.4% |
| `scipy.spatial` | `fsci-spatial` | 18 | 0 | 18 | 18 | 18 | 8 | 100.0% | 44.4% |
| `scipy.spatial.distance` | `fsci-spatial` | 27 | 0 | 27 | 26 | 27 | 22 | 100.0% | 81.5% |
| `scipy.spatial.transform` | `fsci-spatial` | 4 | 0 | 4 | 4 | 4 | 0 | 100.0% | 0.0% |
| `scipy.special` | `fsci-special` | 358 | 0 | 358 | 357 | 358 | 208 | 100.0% | 58.1% |
| `scipy.stats` | `fsci-stats` | 303 | 0 | 303 | 299 | 302 | 239 | 99.7% | 78.9% |
| `scipy.stats.contingency` | `fsci-stats` | 6 | 0 | 6 | 6 | 6 | 2 | 100.0% | 33.3% |
| `scipy.stats.mstats` | `fsci-stats` | 31 | 2 | 29 | 29 | 29 | 11 | 100.0% | 37.9% |
| `scipy.stats.qmc` | `fsci-stats` | 11 | 0 | 11 | 7 | 11 | 4 | 100.0% | 36.4% |
| `scipy.stats.sampling` | `fsci-stats` | 9 | 0 | 9 | 5 | 5 | 0 | 55.6% | 0.0% |
| **TOTAL** | — | **1,555** | **16** | **1,539** | **1,495** | **1,525** | **891** | **99.1%** | **57.9%** |

---

## What this number does and does not mean

**It separates name-matching (`declared`), implementation (`real`), and differential verification (`compared`).** `declared` reports that a public Rust symbol exists whose name normalises to the SciPy symbol's; `real` excludes `na` and `missing` entries in `docs/planning/census_dispositions.toml` and includes `api_shape` mappings where the Rust API has a different name; `compared` requires a reference in `crates/fsci-conformance/tests/diff_*.rs`. Treat the three numbers as distinct and never quote `declared` or `real` alone as behavioural parity.

**Conservative normalization.** The census classifies name matches by kind (`exact`, `case_fold`, `reexport`, `type_alias`, `trait`), and handles:

1. **Naming conventions & aliases.** SciPy exposes *distribution instances* in lowercase — `stats.norm`,
   `stats.beta`, `stats.gamma`, `stats.binom`, `stats.t`, `stats.f`. FrankenSciPy provides exact
   SciPy-compatible aliases for distribution instances, warnings, and traits alongside the Rust
   struct types (`Normal`, `StudentT`, `ChiSquared`, `Binomial`, etc.), with `make_distribution` recorded as the sole missing `scipy.stats` top-level symbol (`302 / 303` real, `239 / 303` compared).
2. **Case.** Symbols implemented verbatim behind `#[allow(non_snake_case)]` — `check_COLA`,
   `check_NOLA` — match exact case (`1,289` exact matches vs `131` case-folded).
3. **Re-exports & Traits.** Crates that expose their surface through `pub use` or Python ABC equivalents
   implemented as `pub trait` (`BivariateSpline`, `HessianUpdateStrategy`) are scanned and classified separately.

---

## The residual, itemised

**Residual count: 16 N/A + 14 missing (30 not-real out of 1,555).** The earlier 16-module name census counted 1,300 symbols including 25 no-op stand-ins (`frankenscipy-8dndw.1`). Those stand-ins were either implemented or removed and recorded in `docs/planning/census_dispositions.toml` as `na` (16 symbols) or `missing` (14 symbols), and `scripts/symbol_census.py` (`frankenscipy-8dndw.2`) enforces that every `na` or `missing` symbol is absent from `pub` declarations.

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
| Solver types and aliases | `RK23`, `RK45`, `DOP853`, `Radau`, `BDF`, `LSODA`, `OdeSolver`, `DenseOutput`, `ode` | Complete as step-by-step solver objects; `LSODA` is ODEPACK's LSODA as SciPy's `lsoda.c` runs it, without banded Jacobians (frankenscipy-1ksfv.9) |
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

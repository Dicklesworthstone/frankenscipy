# SciPy Parity Coverage Report

**Regenerated: 2026-07-27** against **scipy 1.17.1** (live import) and the current crates.

## Overall Coverage: 91.8%

**1,194 of 1,300** SciPy callable symbols have FrankenSciPy equivalents.

## Module-Level Coverage

| Module | scipy | covered | missing | Coverage |
|--------|------:|--------:|--------:|---------:|
| ndimage | 75 | 75 | 0 | 100.0% |
| odr | 10 | 10 | 0 | 100.0% |
| datasets | 5 | 5 | 0 | 100.0% |
| signal | 157 | 156 | 1 | 99.4% |
| special | 358 | 353 | 5 | 98.6% |
| linalg | 98 | 95 | 3 | 96.9% |
| interpolate | 56 | 53 | 3 | 94.6% |
| sparse | 53 | 51 | 2 | 96.2% |
| fft | 41 | 37 | 4 | 90.2% |
| spatial | 18 | 15 | 3 | 83.3% |
| optimize | 71 | 60 | 11 | 84.5% |
| stats | 303 | 239 | 64 | 78.9% |
| integrate | 33 | 24 | 9 | 72.7% |
| io | 14 | 14 | 0 | 100.0% |
| constants | 8 | 7 | 1 | 87.5% |

`scipy.cluster` exports only submodules (`vq`, `hierarchy`) at the top level, so it scores 0/0 here;
its surface lives in `fsci-cluster` and is covered by the conformance suite, not by this census.

---

## What this number does and does not mean

**It is a name-matching census, not a behavioural-parity proof.** It reports that a public Rust symbol
exists whose name normalises to the SciPy symbol's. It does *not* check signatures, semantics, dtypes,
error behaviour, or numerical agreement. Behavioural parity is what `fsci-conformance` and the
differential/metamorphic suites are for; treat the two as complementary and never quote this number as
"parity".

**It undercounts.** Three known biases, all conservative:

1. **Naming conventions.** SciPy exposes *distribution instances* in lowercase — `stats.norm`,
   `stats.beta`, `stats.gamma`, `stats.binom`, `stats.t`, `stats.f`. FrankenSciPy exposes Rust
   *types* — `Normal`, `StudentT`, `ChiSquared`, `Binomial`. **Most of the 64 `stats` "misses" are
   this**, not absent functionality. The honest read of `stats` is materially above 78.9%; the census
   cannot say how much without a hand-built alias map, which is the obvious next improvement.
2. **Case.** Symbols implemented verbatim behind `#[allow(non_snake_case)]` — `check_COLA`,
   `check_NOLA` — only match when the scanner is case-sensitive.
3. **Re-exports.** Crates that expose their surface through `pub use` only match when the scanner
   reads re-export lists.

**It also overcounts in one direction:** a same-named Rust symbol in an unrelated position counts as
covered. Spot checks did not find such a case, but the census cannot rule it out.

---

## The residual, itemised

Every remaining miss, classified. **The great majority are deliberate scope decisions, not gaps.**

### Out of scope by policy

| Category | Symbols |
|---|---|
| Warning/exception classes (Rust uses `Result`) | `ConstantWarning`, `SparseWarning`, `SparseEfficiencyWarning`, `OptimizeWarning`, `NoConvergence`, `IntegrationWarning`, `ODEintWarning`, `BadCoefficients`, `SpecialFunctionError`, `SpecialFunctionWarning`, `ConstantInputWarning`, `DegenerateDataWarning`, `NearConstantInputWarning` |
| Plotting | `convex_hull_plot_2d`, `delaunay_plot_2d`, `voronoi_plot_2d` |
| BLAS/LAPACK introspection (we are pure Rust) | `get_blas_funcs`, `get_lapack_funcs`, `find_best_blas_type` |
| Backend registration (no pluggable-backend layer) | `set_backend`, `set_global_backend`, `register_backend`, `skip_backend` |
| Global error-state mutation (Rust returns errors) | `errstate`, `geterr`, `seterr` |
| Solver classes exposed as `SolverKind` variants | `RK23`, `RK45`, `DOP853`, `Radau`, `LSODA`, `OdeSolver`, `DenseOutput`, `ode` |
| Interactive/CLI helpers | `show_options`, `linprog_verbose_callback` |

### Naming-convention artifacts — functionality present

`stats` distribution instances (`norm`, `beta`, `gamma`, `binom`, `t`, `f`, `chi2`, `expon`,
`lognorm`, `uniform`, `weibull_min`, `truncnorm`, …) → Rust types `Normal`, `Beta`, `Gamma`,
`Binomial`, `StudentT`, `ChiSquared`, … Also `cKDTree` → `KDTree`, `CubicSpline` →
`CubicSplineStandalone`, `BivariateSpline` → the concrete `*BivariateSpline` types.

### Genuine gaps worth implementing

Ranked by user value.

| Symbols | Why it matters | Effort |
|---|---|---|
| `optimize.direct` | DIRECT global optimizer — a real algorithm we do not have | large |
| `optimize.SR1`, `HessianUpdateStrategy`, `BroydenFirst`, `KrylovJacobian`, `InverseJacobian`, `LbfgsInvHessProduct` | Quasi-Newton update strategies as first-class objects | medium |
| `stats.goodness_of_fit`, `make_distribution`, `CensoredData`, `Covariance`, `Mixture`, `order_statistic`, `rv_continuous`, `rv_histogram` | Genuinely absent statistical machinery — the real `stats` residual once naming artifacts are removed | large |

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

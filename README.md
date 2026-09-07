# FrankenSciPy

<div align="center">
  <img src="frankenscipy_illustration.webp" alt="FrankenSciPy — clean-room Rust reimplementation of SciPy with a Condition-Aware Solver Portfolio">
</div>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT%20with%20rider-blue.svg" alt="License: MIT with OpenAI/Anthropic Rider">
  <img src="https://img.shields.io/badge/rust-2024%20edition%20%C2%B7%20nightly-orange.svg" alt="Rust 2024 nightly">
  <img src="https://img.shields.io/badge/unsafe-%23!%5Bforbid(unsafe__code)%5D-brightgreen.svg" alt="No unsafe code">
  <img src="https://img.shields.io/badge/async-asupersync%20(no%20tokio)-purple.svg" alt="asupersync, no tokio">
  <img src="https://img.shields.io/badge/workspace-19%20crates-informational.svg" alt="19 workspace crates">
  <img src="https://img.shields.io/badge/conformance-793%20test%20files-success.svg" alt="793 conformance test files">
</p>

> **FrankenSciPy is a clean-room Rust reimplementation of SciPy's core numerical
> routines with a Condition-Aware Solver Portfolio (CASP) at its center.**
> The dense linear-solve family (`solve_with_casp`, `solve_with_audit`,
> `lstsq_with_casp` and their siblings in `fsci-linalg`) goes through a runtime
> that probes matrix conditioning and structure, picks the solver that minimizes
> expected loss against a calibrated 5×4 decision matrix, falls back on failure,
> and returns a certificate proving the decision was justified. The sparse,
> optimize and special crates carry their own rule-based selectors that record a
> rationale; wiring them onto the same loss-matrix machinery is roadmap work.

---

## TL;DR

### The Problem

SciPy is the bedrock of scientific Python, but the runtime it sits on is showing its age:

- The CPython GIL and Python's object model make it awkward to use as a high-throughput library inside a service, a notebook kernel cluster, or a multi-agent system.
- Most of the numerical kernels are wrapped C, Fortran, or LAPACK (fast in microbenchmarks), but the *Python* glue layer is where memory churn, type-coercion overhead, and packaging headaches actually live.
- Numerical-stability decisions are buried inside per-routine heuristics. There is no first-class way to ask "which algorithm did you pick for this problem instance, and how confident are you that it's the right one?"
- Embedding SciPy in a Rust application, a WebAssembly module, or a memory-constrained edge runtime is a non-starter.

### The Solution

FrankenSciPy reimplements the SciPy surface in idiomatic Rust with three guarantees that the original cannot provide:

1. **Memory- and thread-safety by construction.** `#![forbid(unsafe_code)]` is enforced workspace-wide.
2. **Explicit conditioning-aware algorithm selection.** Every solve goes through CASP, which records the chosen action, the evidence that drove the choice, and the expected loss versus alternatives.
3. **Differential conformance against the real SciPy.** Every domain has a Python oracle script that captures reference outputs from `scipy.*` and a Rust harness that diffs the FrankenSciPy implementation against the oracle inside automated test runs.

### Why FrankenSciPy?

| | SciPy | nalgebra / ndarray | argmin / linfa | **FrankenSciPy** |
|---|---|---|---|---|
| Memory safety | C/Fortran kernels | safe Rust | safe Rust | **`#![forbid(unsafe_code)]`** workspace-wide |
| Async runtime | N/A (Python) | N/A | tokio-leaning | **none in the API** (synchronous kernels; no tokio anywhere in the tree) |
| Surface area | ~1,300 callable symbols | low-level only | optimization or ML focus | **1,194 same-named public equivalents across 19 crates** (name census; see [`PARITY-COVERAGE.md`](docs/planning/PARITY-COVERAGE.md)) |
| Algorithm selection | hand-rolled per routine | manual | manual | **CASP runtime portfolio** with audit trail on the dense solve family; rule-based selectors elsewhere |
| Conformance against SciPy | self-checking | none | partial | **15 Python oracles, 731 differential test files** that resolve a live SciPy interpreter at test time |
| Distribution moments | partial closed-forms | none | none | **95+ continuous, 10+ discrete, explicit `skewness`/`kurtosis`/`entropy`** |
| Numerical-stability contract | implicit | implicit | implicit | **Stability outranks speed; tolerance contracts cannot be weakened** |
| Artifact durability | none | none | none | **RaptorQ systematic encoding for conformance/benchmark/reproducibility bundles** |
| Decision audit | none | none | none | **Per-call audit ledger** (action, posterior, expected loss, backward error) |

---

## Condition-Aware Solver Portfolio (CASP)

CASP is the design feature that separates this project from a generic numerical Rust library.

For a linear solve, CASP:

1. **Probes** the matrix for evidence: reciprocal condition number `rcond`, structural form (general / diagonal / triangular), known sparsity pattern, and any backward-error hints from prior calls.
2. **Computes** the posterior probability over four condition states (`WellConditioned`, `ModerateCondition`, `IllConditioned`, `NearSingular`) using a conformal calibrator that is retrained from accumulated evidence.
3. **Minimizes expected loss** over five solver actions against a calibrated 5×4 loss matrix (the literal `SolverPortfolio::default_loss_matrix()`):

   ```text
   Action \ State      | WellCond | ModerateCond | IllCond | NearSingular
   DirectLU            |        1 |            5 |      40 |          120
   PivotedQR           |        3 |            1 |       8 |           45
   SVDFallback         |       15 |           10 |       1 |            1
   DiagonalFastPath    |        0 |            0 |       0 |          100
   TriangularFastPath  |        0 |            0 |       0 |          100
   ```

4. **Emits an audit event** containing the chosen action, the evidence that drove it, the posterior, the expected loss versus each alternative, and a fingerprint that ties the decision to its inputs.
5. **Falls back** automatically if the primary solver fails. The failure becomes evidence, the calibrator updates, and the next action is chosen against the same loss matrix with the new posterior.

Three other crates ship selectors that share the CASP name but not its machinery today: `fsci_sparse::select_casp_iterative_solver` (CG / MINRES / LGMRES / BiCGSTAB / QMR / GMRES / LSQR / LSMR from symmetry, diagonal dominance, density and matvec cost), `fsci_opt::select_minimize_method` (constraint and gradient availability, dimension, scaling), and `fsci_special::select_hypergeometric_branch` (series / transformation / continued-fraction branches for `0F1`, `1F1`, `2F1`). Each returns a decision with a written rationale and validates its inputs under the strict/hardened mode split, but none of them consults a loss matrix, posterior, or calibrator. Only `fsci-linalg` does.

**Stability outranks speed.** Tolerance contracts on scoped V1 routines are guarded by the conformance harness. No optimization may weaken them.

---

## Workspace at a Glance

FrankenSciPy is a Cargo workspace of **19 crates** spanning ~610,000 lines under `crates/*/src` (implementation plus the inline `#[cfg(test)]` suites, which are the bulk of the larger crates; 10,056 `#[test]` functions in all), plus 793 conformance integration-test files and the fixture tree in `fsci-conformance`. The line counts below are `wc -l` over each crate's `src/`, measured 2026-09-03.

| Crate | Lines | Surface |
|---|---|---|
| [`fsci-linalg`](crates/fsci-linalg/) | ~60,700 | Dense and structured linear algebra; CASP solver selection; LU / QR / Cholesky / SVD / LDL / Schur / Hessenberg / QZ; `expm`, `logm`, `sqrtm`, `funm`, `signm`; Sylvester, Lyapunov, continuous and discrete Riccati; banded specialists; subspace and polar decompositions |
| [`fsci-sparse`](crates/fsci-sparse/) | ~73,100 | CSR/CSC/COO/BSR/DIA/DOK/LIL formats; `spsolve`/`splu`/`spilu`; CG, GMRES (Arnoldi + Givens), LGMRES, BiCG, BiCGSTAB, CGS, QMR (look-ahead Lanczos), MINRES, LSQR, LSMR (CASP-dispatched); `eigs` via Arnoldi iteration on a Krylov subspace; `eigsh` via deflated power iteration with deterministic LCG seed (orthogonal to no eigenmode); `svds`; Dijkstra, Bellman-Ford, MST, BFS/DFS, connected components, PageRank, Reverse Cuthill-McKee, centrality |
| [`fsci-integrate`](crates/fsci-integrate/) | ~34,900 | `solve_ivp` (RK23 / RK45 / DOP853 / BDF / Radau / LSODA); `odeint`; `solve_bvp`; `quad` family with Gauss-Kronrod adaptation; `dblquad`, `tplquad`, `nquad`, `cubature`; Romberg; Monte Carlo and QMC quadrature; sample-form rules |
| [`fsci-interpolate`](crates/fsci-interpolate/) | ~24,400 | `interp1d`, `CubicSpline`, `CubicHermiteSpline`, `BSpline`, `Akima`, `PCHIP`; `RegularGridInterpolator`, `griddata`, `interpn`; Krogh, barycentric, polynomial helpers; `make_lsq_spline` for k = 1/3/5 |
| [`fsci-opt`](crates/fsci-opt/) | ~43,100 | `minimize` portfolio: Nelder-Mead, BFGS, CG, Powell, L-BFGS-B, Newton-CG, TNC, COBYLA, SLSQP, trust-ncg / -krylov / -exact / -constr, dogleg; `root` family: brentq, brenth, ridder, toms748, newton, halley, broyden1/2, anderson, fsolve, lm_root; `curve_fit`, `least_squares`, NNLS, isotonic regression; global: DE, basinhopping, dual annealing, SHGO, PSO, brute; LP/MILP; `linear_sum_assignment` |
| [`fsci-fft`](crates/fsci-fft/) | ~13,500 | Cooley-Tukey mixed-radix; Bluestein for non-power-of-2 lengths; `rfft`/`irfft`; n-D transforms (`fftn`/`ifftn`/`rfftn`/`irfftn`); DCT/DST I–IV (1-D and n-D); Hilbert analytic signal; FHT; fingerprinted plan cache with admission policy |
| [`fsci-signal`](crates/fsci-signal/) | ~41,000 | Windows (Hann, Hamming, Kaiser, Tukey, Blackman, Taylor, exponential, general-Hamming…); filter design (`butter` / `cheby1` / `cheby2` / `ellip` / `bessel`, ZPK and BA forms, full `lp2{lp,hp,bp,bs}` and `lp2{lp,hp,bp,bs}_zpk` transforms, `bilinear` / `bilinear_zpk`); filter-order helpers `buttord` / `cheb1ord` / `cheb2ord` / `ellipord`; analog prototypes `buttap` and `cheb1ap`; `firwin`, `firls`, `remez`; `lfilter`, `filtfilt`, SOS application; `welch`, `periodogram`, `csd`, `coherence`; `find_peaks` with prominence and width; CWT; Daubechies / Morlet / Ricker wavelets; MFCC, mel filterbank, chroma |
| [`fsci-spatial`](crates/fsci-spatial/) | ~17,700 | KDTree / cKDTree (`query`, `query_pairs`, `count_neighbors`); `pdist` / `cdist` / `distance_matrix` (Euclidean, Manhattan, Chebyshev, Minkowski, Mahalanobis, Hausdorff, weighted variants); ConvexHull; Delaunay; Voronoi; HalfspaceIntersection; Hungarian linear assignment |
| [`fsci-special`](crates/fsci-special/) | ~72,100 | Gamma family (`gamma`, `gammaln`, `digamma`, `polygamma`, `pentagamma`, `factorialk`); beta family; `erf`/`erfc`/`erfinv`; Bessel J/Y/I/K/H1/H2 and spherical variants; Airy + zeros; hypergeometric `0F1`/`1F1`/`2F1` with **CASP branch selection**; elliptic `K`/`E`/`J` and the full Carlson family `RC`/`RF`/`RD`/`RJ`/`RG`; zeta; Struve; Dawson; spherical harmonics; orthogonal polynomials (Legendre / Cheby T,U,C,S / Hermite / Laguerre / Jacobi / Gegenbauer + shifted variants); Voigt profile; accurate-near-zero `log1pmx`, `powm1`, `cosm1`; Kelvin functions |
| [`fsci-stats`](crates/fsci-stats/) | ~126,200 | **95+ continuous and 10+ discrete distributions**, each with PDF/CDF/SF/PPF/mean/var/skewness/kurtosis/entropy/mode/fit; t-tests, KS, Shapiro, Mann-Whitney, Wilcoxon, ANOVA, chi-square contingency; Pearson/Spearman/Kendall correlations; linear regression; bootstrap; permutation tests; `gaussian_kde`; Box-Cox; QMC engines (Sobol, Halton, Latin Hypercube) with centered / mixture / wraparound / L2-star discrepancies |
| [`fsci-cluster`](crates/fsci-cluster/) | ~17,400 | KMeans + KMeans++ initialization; DBSCAN; hierarchical agglomerative linkage; `dendrogram`; `fcluster`; silhouette / Davies-Bouldin / Calinski-Harabasz indices |
| [`fsci-ndimage`](crates/fsci-ndimage/) | ~31,000 | Uniform / Gaussian / median / minimum / maximum filters; closure-driven `generic_filter`; binary and grayscale morphology (erosion, dilation, opening, closing, hit-or-miss); `label`, `find_objects`; Euclidean distance transform; affine transform, rotate, zoom, shift; Sobel / Prewitt / Laplace edge detectors; histograms and extrema indexed by label |
| [`fsci-io`](crates/fsci-io/) | ~8,500 | `savemat` / `loadmat` for MATLAB v4 and v5 (incl. compressed and struct arrays); Matrix Market `mmread` / `mmwrite` (dense and sparse); WAV PCM and IEEE-float read/write; simplified NetCDF reader; IDL `.sav` reader; Fortran sequential unformatted reader |
| [`fsci-constants`](crates/fsci-constants/) | ~1,700 | CODATA 2018 physical constants; SI prefixes; mathematical constants (`pi`, `e`, `euler_gamma`…); unit conversions |
| [`fsci-odr`](crates/fsci-odr/) | ~3,000 | Orthogonal Distance Regression: `ODR` driver, `Model`, `Data`, `Output`; explicit and implicit models; weighted, multi-response fits |
| [`fsci-datasets`](crates/fsci-datasets/) | ~670 | Deterministic embedded sample fixtures matching SciPy shapes: `ascent`, `face` (RGB / gray), `electrocardiogram` |
| [`fsci-runtime`](crates/fsci-runtime/) | ~3,650 | The CASP engine: `SolverPortfolio`, `MatrixConditionState`, `StructuralEvidence`, `SolverAction`, `PolicyController`, evidence ledger, conformal calibrator, fail-closed semantics, strict vs hardened modes |
| [`fsci-arrayapi`](crates/fsci-arrayapi/) | ~4,800 | Contract-first Array API backend (`backend.rs`, `broadcast`, `creation`, `indexing`, `audit`) with integration seams for linalg / opt / sparse |
| [`fsci-conformance`](crates/fsci-conformance/) | ~33,000 (lib + 9 bins) + **793 test files** | Three-lane differential harness (self-check, SciPy-oracle, dispatch); RaptorQ evidence packs; `parity_report.json` and `decode_proof.json` artifacts; 15 Python oracle scripts; nine binaries (`conformance_dashboard`, `e2e_orchestrator`, `fixture_regen`, `live_oracle_capture`, `benchmark_gate`, `raptorq_sidecar`, `tolerance_lint`, `adversarial_corpus`, `fuzz_triage`) |

For per-symbol parity assessment see [`docs/planning/FEATURE_PARITY.md`](docs/planning/FEATURE_PARITY.md).

---

## Quick Example

Add the crates you need to a workspace member:

```toml
[dependencies]
fsci-linalg     = { git = "https://github.com/Dicklesworthstone/frankenscipy" }
fsci-stats      = { git = "https://github.com/Dicklesworthstone/frankenscipy" }
fsci-special    = { git = "https://github.com/Dicklesworthstone/frankenscipy" }
fsci-integrate  = { git = "https://github.com/Dicklesworthstone/frankenscipy" }
```

A condition-aware linear solve with a decision certificate:

```rust
use fsci_linalg::{solve_with_casp, SolveOptions};
use fsci_runtime::{SolverPortfolio, RuntimeMode};

fn main() {
    // 3x3 well-conditioned SPD system
    let a = vec![
        vec![4.0, 1.0, 0.0],
        vec![1.0, 3.0, 1.0],
        vec![0.0, 1.0, 2.0],
    ];
    let b = vec![1.0, 2.0, 3.0];

    // Strict mode; evidence ledger capacity = 64
    let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);

    let result = solve_with_casp(&a, &b, SolveOptions::default(), &mut portfolio)
        .expect("solve");

    println!("x = {:?}", result.x);
    if let Some(cert) = result.certificate.as_ref() {
        println!("chosen action      = {:?}", cert.action);
        println!("rcond estimate     = {:.3e}", cert.rcond_estimate);
        println!("structural evidence= {:?}", cert.structural_evidence);
        println!("posterior          = {:?}", cert.posterior);
        println!("expected losses    = {:?}", cert.expected_losses);
        println!("chosen exp. loss   = {:.3}", cert.chosen_expected_loss);
        println!("fallback active    = {}", cert.fallback_active);
    }
    println!("evidence ledger size = {}", portfolio.evidence_len());
}
```

Distribution moments, all closed-form or numerically anchored:

```rust
// The moment surface lives on the `ContinuousDistribution` trait, so it must be in scope.
use fsci_stats::{BetaDist, ChiSquared, ContinuousDistribution};

fn main() {
    // Constructors return the distribution directly; invalid params panic
    let beta = BetaDist::new(2.0, 5.0);
    println!("Beta(2,5) entropy   = {:.6}", beta.entropy());
    println!("Beta(2,5) skewness  = {:.6}", beta.skewness());
    println!("Beta(2,5) kurtosis  = {:.6}", beta.kurtosis());

    let chi2 = ChiSquared::new(4.0);
    println!("Chi²(4)  entropy    = {:.6}", chi2.entropy());     // closed-form
    println!("Chi²(4)  PPF(0.95)  = {:.6}", chi2.ppf(0.95));
}
```

A stiff ODE integrated with LSODA (auto-switching nonstiff ↔ BDF):

```rust
use fsci_integrate::{solve_ivp, SolveIvpOptions, SolverKind};

fn main() {
    // Robertson's stiff problem
    let mut f = |_t: f64, y: &[f64]| -> Vec<f64> {
        let (y0, y1, y2) = (y[0], y[1], y[2]);
        vec![
            -0.04 * y0 + 1.0e4 * y1 * y2,
             0.04 * y0 - 1.0e4 * y1 * y2 - 3.0e7 * y1 * y1,
             3.0e7 * y1 * y1,
        ]
    };

    let y0 = [1.0, 0.0, 0.0];
    let options = SolveIvpOptions {
        t_span: (0.0, 4.0e10),
        y0: &y0,
        method: SolverKind::Lsoda,
        t_eval: None,
        dense_output: false,
        events: None,
        ..Default::default()
    };

    let sol = solve_ivp(&mut f, &options).expect("solve_ivp");
    println!("steps = {}, final y = {:?}", sol.t.len(), sol.y.last().unwrap());
}
```

Carlson elliptic integrals (the entire `RC`/`RF`/`RD`/`RJ`/`RG` family is
exported as plain scalar functions):

```rust
use fsci_special::{elliprc, elliprf, elliprd, elliprj, elliprg};

fn main() {
    println!("RC(0.5, 1.0)            = {:.10}", elliprc(0.5, 1.0));
    println!("RF(0.5, 1.0, 2.0)       = {:.10}", elliprf(0.5, 1.0, 2.0));
    println!("RD(0.5, 1.0, 2.0)       = {:.10}", elliprd(0.5, 1.0, 2.0));
    println!("RG(0.5, 1.0, 2.0)       = {:.10}", elliprg(0.5, 1.0, 2.0));
    println!("RJ(1, 2, 3, 4)          = {:.10}", elliprj(1.0, 2.0, 3.0, 4.0));
}
```

> The `scipy.special`-shaped variants such as `ellipk(m)` live on the
> tensor-flavored runtime API (`ellipk(&SpecialTensor, RuntimeMode) ->
> SpecialResult`) so they participate in the audit and mode model. Reach for
> the Carlson scalars above when you want a plain `f64`-in / `f64`-out helper.

---

## Design Philosophy

FrankenSciPy commits to five principles.

### 1. Numerical stability outranks speed

The compatibility doctrine has two modes that must coexist:

- **Strict mode** maximizes observable SciPy parity. No behavior-altering repairs; the user gets exactly what SciPy would have done, including its known edge-case quirks.
- **Hardened mode** preserves the API contract but adds bounded defensive recovery for malformed inputs and ill-conditioned regimes. Recoveries are logged. Unknown incompatible features fail-closed.

In both modes, the tolerance contracts on scoped V1 routines are guarded by the conformance harness. **No optimization may weaken them.**

### 2. Memory safety, by construction

`#![forbid(unsafe_code)]` is declared at the workspace level. Where a narrow unsafe primitive is genuinely unavoidable (so far it has not been), the rule is that it must be isolated behind an audited interface with property tests. There is no FFI to LAPACK, no inline assembly, no `transmute`.

### 3. No tokio

Every numerical kernel is synchronous and returns a `Result`; the workspace owns no async runtime and `cargo tree -i tokio` is empty. The only async-capable dependency is [asupersync](https://github.com/Dicklesworthstone/asupersync), and today exactly one piece of it is consumed: its RaptorQ systematic encoder, used by `fsci-conformance` for artifact sidecars. The tokio ecosystem (`tokio`, `hyper`, `reqwest`, `axum`, `async-std`, `smol`, and anything that transitively depends on them) is forbidden.

This makes FrankenSciPy trivially embeddable in any host runtime without dragging in a global async ecosystem.

### 4. Differential conformance against the real SciPy

For every supported routine there is, or will be, a Python oracle script in `crates/fsci-conformance/python_oracle/` that captures reference outputs from `scipy.*`, plus a Rust harness that diffs the FrankenSciPy implementation against the oracle under a documented tolerance policy. Fifteen oracles exist today, one per SciPy subpackage covered.

If SciPy is unavailable on the build host the harness is explicit about it: required oracles return `PythonSciPyMissing`; optional oracles write `oracle_capture.error.txt` and the differential lane is skipped rather than silently passing.

### 5. Durable, audited artifacts

The conformance harness, benchmark baselines, migration manifests, reproducibility ledgers, and long-lived state snapshots all carry **RaptorQ systematic-encoding sidecars**. Each artifact bundle ships with:

1. A repair-symbol generation manifest.
2. An integrity scrub report.
3. A decode proof artifact for each recovery event.

The artifact topology is locked, the contract schemas (`behavior_ledger.schema.json`, `contract_table.schema.json`, `threat_matrix.schema.json`) are governance-gated, and changes require an explicit proposal in the beads tracker.

---

## Architecture

The high-level data flow inside every solve is the same:

```text
                ┌──────────────────────────────────────────────────────┐
                │   Public domain API (fsci-linalg, fsci-opt, …)       │
                └────────────────────────────┬─────────────────────────┘
                                             │  problem instance
                                             ▼
                ┌──────────────────────────────────────────────────────┐
                │   CASP algorithm selector (fsci-runtime)             │
                │     • probes structural + numerical evidence         │
                │     • posterior over condition states                │
                │     • argmin expected loss over solver actions       │
                └────────────────────────────┬─────────────────────────┘
                                             │  chosen action
                                             ▼
                ┌──────────────────────────────────────────────────────┐
                │   Numeric kernel (specialized per action)            │
                │     DirectLU │ PivotedQR │ SVDFallback │ FastPath    │
                └────────────────────────────┬─────────────────────────┘
                                             │  result + backward error
                                             ▼
                ┌──────────────────────────────────────────────────────┐
                │   Diagnostics + Audit Ledger                         │
                │     • emits AuditEvent (action, posterior, loss…)    │
                │     • feeds calibrator update                        │
                │     • surfaces failure → fallback evidence           │
                └──────────────────────────────────────────────────────┘
```

For iterative methods (`fsci-sparse`, `fsci-opt`) the kernel level is the iteration loop and the calibrator feeds back on residuals and convergence rates. For special-function evaluation (`fsci-special`) the selector picks between series, asymptotic, and continued-fraction branches.

---

## Conformance Harness

The conformance harness in `fsci-conformance` is a first-class subsystem with its own library, nine binaries, and an artifact tree under version control.

### Three lanes

1. **Self-check** (`run_<family>_packet`): pure Rust validation against fixture-embedded expected values. No SciPy required. Runs on every CI build.
2. **Oracle-backed** (`run_<family>_packet_with_oracle_capture`): invokes the appropriate Python oracle, captures SciPy reference output, diffs FrankenSciPy against the oracle inside the same Rust process. Runs on the SciPy-present CI lane.
3. **Dispatch** (`run_differential_test`): family-routed differential cases with per-case audit ledgers, RaptorQ sidecars, and `parity_report.json` emission.

### Python oracles

Fifteen oracle scripts wrap reference implementations one-per-subpackage:

```text
crates/fsci-conformance/python_oracle/
├── scipy_arrayapi_oracle.py
├── scipy_cluster_oracle.py
├── scipy_constants_oracle.py
├── scipy_fft_oracle.py
├── scipy_integrate_oracle.py
├── scipy_interpolate_oracle.py
├── scipy_io_oracle.py
├── scipy_linalg_oracle.py
├── scipy_ndimage_oracle.py
├── scipy_optimize_oracle.py
├── scipy_signal_oracle.py
├── scipy_sparse_oracle.py
├── scipy_spatial_oracle.py
├── scipy_special_oracle.py
└── scipy_stats_oracle.py
```

### Packets

Eighteen `FSCI-P2C-NNN` artifact packets exist today, plus the legacy `P2C-NNN` packet tree. Each packet directory is structured as:

```text
fixtures/artifacts/FSCI-P2C-XYZ/
├── anchor/            # SciPy oracle captures
├── contracts/         # behavior_ledger.json, contract_table.json, threat_matrix.json
├── differential/      # per-case diffs + decoder proofs
├── e2e/               # golden-journey end-to-end traces
├── evidence/          # solver-decision audit ledgers
├── perf/              # criterion baselines and tail-latency reports
└── threats/           # adversarial fixtures + recovery results
```

### Interactive dashboard

```bash
cargo run -p fsci-conformance --bin conformance_dashboard -- \
    --artifact-root crates/fsci-conformance/fixtures/artifacts \
    --packet-filter P2C-002
```

The dashboard is an `ftui` TUI: arrow keys to move, Tab to cycle panels, `r` to reload, `q` to quit. It loads packet state from disk, surfaces per-case diffs, and best-effort recovers from malformed artifacts so a single broken file doesn't take down the view.

### Companion binaries

The `fsci-conformance` crate ships eight companion binaries alongside the dashboard, each fulfilling one slice of the harness lifecycle:

| Binary | Purpose |
|---|---|
| `conformance_dashboard` | Interactive TUI for navigating packet results |
| `e2e_orchestrator` | Runs the registered end-to-end golden-journey scenarios and emits `e2e/*.json` |
| `fixture_regen` | Regenerates fixture bundles from a `--fixture <name>` and `--oracle <path>` pair; used to refresh tracked artifacts that derive from the oracle capture |
| `live_oracle_capture` | Captures fresh SciPy oracle outputs in required or fallback mode; the binary the CI G3b / G3c lanes drive |
| `benchmark_gate` | `--check-spec` validates the published `docs/baseline_*.json` against the spec §17 budgets; `--compare <dir>` diffs a fresh criterion export against them (used by hand, not in CI) |
| `raptorq_sidecar` | Generates and verifies the RaptorQ systematic-encoding sidecars for every packet bundle |
| `tolerance_lint` | Ratchet check over the `FSCI-P2C-*.json` fixtures (unexplained relaxations may only fall) and over every `tests/diff_*.rs` harness tolerance against the committed `fixtures/tolerance_baseline.json` — loosening or an unanchored contract fails the build; `--write-baseline` adopts tightenings and refuses loosened values |
| `adversarial_corpus` | Emits the adversarial fixture corpus (malformed metadata, NaN/Inf, degenerate shapes) with BLAKE3 manifests under an `--output-root` |
| `fuzz_triage` | Triages `fuzz/artifacts/` crashes: deduplicates by hash, promotes reproducers to regression fixtures, applies a retention horizon, and files `fuzz-finding` beads |

These map to CI gates G3b / G3c / G6 / G8 / G9 and can also be invoked locally during development.

---

## Algorithms and Internals

What follows is the technical-depth pass: which algorithms FrankenSciPy actually uses, how the mode model behaves at the boundary, what an audit-ledger entry contains on disk, the topology of conformance artifacts, the threat model, the asupersync integration story, and the discipline that gates every performance change.

### Numerical Recipes: What's Inside Each Kernel

| Routine | What FrankenSciPy uses |
|---|---|
| `solve` (dense) | CASP-dispatched between Doolittle LU with partial pivoting, Householder QR with column pivoting, and a Golub-Kahan-bidiagonalization SVD fallback for `rcond < ~1e-12`. Triangular and diagonal matrices are detected up front and short-circuited through `TriangularFastPath` / `DiagonalFastPath`. |
| `lstsq` | SVD-based by default; CASP can promote to QR when the matrix is well-conditioned and the system is overdetermined enough to amortize the QR build cost. |
| `expm` | Scaling-and-squaring around a truncated 20-term Taylor series. The scaling exponent `s` is chosen so `‖A / 2ˢ‖₁ < 0.5`, then the result is squared `s` times. |
| `logm` | Parlett recurrence on the Schur form, with the diagonal-block denominator computed via backward recurrence so convergence loss in difficult eigenvalue clusters is avoided. |
| `sqrtm` | Schur decomposition + block recurrence; the upper-triangular Schur factor is square-rooted block-by-block. |
| `funm` | General matrix-function evaluation via the Schur–Parlett scheme. |
| `eig` / `eigh` | Householder reduction to Hessenberg or tridiagonal form, then QR-with-Wilkinson-shift iteration. |
| `eigs` (sparse) | Arnoldi iteration on a Krylov subspace, with the upper Hessenberg factor reduced and the Arnoldi residual used as a stopping criterion. |
| `eigsh` (symmetric sparse) | Symmetric Lanczos through the shared Krylov/Arnoldi kernel: a single subspace of `max(2k+1, 20)` vectors (SciPy's `ncv` default) resolves the extreme eigenpairs, `converged` is set from the actual Ritz residuals, and pathologically clustered spectra report `converged = false` rather than looping (an implicitly restarted variant is not implemented). This replaced the original deflated power iteration, whose constant-seed bug on path Laplacians is described under **Case Studies**. |
| `gmres` | Restarted GMRES with Arnoldi via modified Gram-Schmidt; least-squares step uses Givens rotations on the upper-Hessenberg factor. |
| `qmr` | Look-ahead Lanczos process building a quasi-minimal residual approximation, using the transpose of `A` to drive the dual sequence. |
| `lgmres` | Augmented Krylov subspace built from prior approximation-error vectors; the `k=0` lucky-breakdown case is guarded so the outer loop cannot spin forever on identity-like operators. |
| `minres` | Lanczos process reducing `A` to tridiagonal form, then solving the tridiagonal system in-place. |
| FFT | Cooley-Tukey mixed-radix (radix-2, -3, -4, -5) for highly composite lengths; Bluestein's algorithm via chirp-z transform for general lengths. The plan cache is keyed by `(length, direction, normalization)` fingerprint and uses a bounded-capacity admission policy. |
| DCT/DST | Type I–IV in 1-D, fused n-D variants for `dctn`/`idctn`/`dstn`/`idstn`. |
| RK45 | Dormand-Prince embedded 5(4) pair with `select_initial_step` Hairer heuristic for the first step size. |
| DOP853 | Hairer-Nørsett-Wanner 8(5,3) embedded pair, used when the user opts for tight tolerances on non-stiff problems. |
| BDF | Variable-order (1–5) backward differentiation formula with Newton iteration; Jacobian is finite-differenced on demand. |
| Radau | A genuine 3-stage Radau IIA solver (`crates/fsci-integrate/src/radau.rs`) matching `solve_ivp(method='Radau')`; it is no longer an alias of the BDF kernel. |
| LSODA | Automatic nonstiff ↔ stiff switching via a stiffness-indicator heuristic on step-size rejections; switches into the BDF kernel once the indicator crosses threshold. |
| `solve_bvp` | Collocation with a free-mesh adaptive solver; Newton on the discretized system. |
| `quad` | Adaptive Gauss-Kronrod (15-point) with subinterval bisection; short-circuits on NaN/Inf integrand values rather than spinning to the `2^limit` subdivision wall. |
| Romberg | Trapezoidal rule with Richardson extrapolation of the order-2k errors. |
| `monte_carlo_integrate` | Stratified-sample Monte Carlo with variance estimation; deterministic seed unless overridden. |
| `qmc_quad` | Quasi-Monte Carlo over Sobol/Halton/Latin-Hypercube engines from `fsci-stats::qmc`. |
| Minimizers | Each method is its own kernel: Nelder-Mead with full simplex bookkeeping, BFGS with strong-Wolfe line search, L-BFGS-B with active-set projection, Powell with bidirectional set, Newton-CG with truncated CG, TNC bounded variant, SLSQP for equality+inequality, COBYLA for general nonlinear constraints, trust-{ncg,krylov,exact,constr} for trust-region methods, dogleg for the Levenberg-Marquardt-style steps. |
| Root finders | brentq, brenth, ridder, toms748, newton, halley, secant, anderson, broyden1/2, fsolve, lm_root, with explicit fallback rules between bracketing and Newton-style methods. |
| Global optimizers | Differential Evolution, Basin Hopping, Dual Annealing, SHGO (Simplicial Homology Global Optimization), Particle Swarm Optimization. |
| LP / MILP | Revised simplex with explicit pivot selection; branch-and-bound for MILP. |
| `linear_sum_assignment` | Hungarian algorithm with O(n³) zero-padding for rectangular inputs. |
| Hypergeometric `2F1` | CASP-selected between Taylor series, Pfaff and Euler symmetry reductions, and asymptotic continued fractions; the branch selector is exposed and tested in conformance. |
| Bessel `J`/`Y` | Series for `|x| < threshold`, classical asymptotic expansion otherwise, with overflow and underflow guards in both arms. |
| Carlson elliptic | Carlson's symmetric duplication algorithm for `RF`, `RD`, `RG`, `RJ` (drives all four down to a series-evaluable kernel); `RC` reduces to a single closed form. |
| Special function `gamma` family | Lanczos approximation for both `Γ` and `ln Γ`; `polygamma` and `pentagamma` via the standard shift-then-asymptotic recipe. |
| Distribution moments | Closed form where derivable, Simpson on the PDF (or on `ppf'`, or via raw moments) where not; documented `NaN` for heavy-tail families where the moment integral diverges. |

### Mode Model: Strict vs Hardened, in detail

Every routine that participates in CASP is parameterized by a `RuntimeMode`. The two modes differ at three precise boundaries:

| Aspect | `RuntimeMode::Strict` | `RuntimeMode::Hardened` |
|---|---|---|
| Malformed input (non-square matrix, mismatched dimensions) | Fail-closed with `LinalgError`; emit a `FailClosed` audit event with the input fingerprint and reason | Same as Strict; malformed input is *always* fail-closed |
| Non-finite entries (NaN, Inf) | Rejected when `SolveOptions::check_finite = true` (the default) | Rejected when `check_finite = true`; bounded recovery may be emitted in routines where the kernel can safely project, recorded as a `BoundedRecovery` audit event |
| Very large matrices | No dimension cap (resource budgeting is the caller's job) | Hard cap at `HARDENED_MAX_DIM = 10_000`; oversize input is fail-closed with `resource_exhausted` |
| Ill-conditioned input near `rcond ≈ 0` | CASP routes to the SVD fallback; result returned with `warning = Some(LinalgWarning::IllConditioned { reciprocal_condition })` and the audit certificate records the chosen action and posterior | Same as Strict; any in-routine regularization is recorded as a `BoundedRecovery` event with the recovery action described in the event payload |
| Calibrator drift | When the conformal calibrator's empirical miscoverage exceeds its target, `SolverPortfolio::select_action` overrides the loss-minimizing pick and returns `SolverAction::SVDFallback` directly | Same |

A typical migration pattern: use `Hardened` in production hot paths (so a single malformed batch doesn't take down a long-running service) and use `Strict` in the conformance harness and during local development (so behavior matches SciPy exactly and surprising auto-repair never masks a real bug).

#### Participating Crates

The strict/hardened mode split, `HARDENED_MAX_DIM` resource capping, and `AuditAction::FailClosed` / `BoundedRecovery` audit ledger emission are enforced across all participating domain crates:

| Crate | Mode-Aware Entry Points | Hardened Dimension Cap | Audit Ledger Emission |
|---|---|---|---|
| `fsci-linalg` | `solve_with_casp`, `solve_with_mode`, etc. | `HARDENED_MAX_DIM = 10_000` | `FailClosed`, `BoundedRecovery`, `ModeDecision` |
| `fsci-opt` | `minimize_with_mode`, solvers | Iteration & dimension bounds | `FailClosed`, `BoundedRecovery` |
| `fsci-sparse` | `SparseMatrix` solvers with mode | Dimension / non-zero bounds | `FailClosed`, `BoundedRecovery` |
| `fsci-integrate` | `solve_ivp_with_mode`, quadrature | Step / dimension limits | `FailClosed`, `BoundedRecovery` |
| `fsci-special` | Tensor APIs with mode | Batch dimension limits | `FailClosed`, `BoundedRecovery` |
| `fsci-signal` | `czt_with_mode_and_audit`, `czt` | `HARDENED_MAX_DIM = 10_000` | `FailClosed`, `BoundedRecovery` |
| `fsci-ndimage` | `gaussian_filter_with_mode` | `HARDENED_MAX_DIM = 10_000` | `FailClosed`, `BoundedRecovery` |
| `fsci-interpolate` | `Interp1d::new_with_audit` | `HARDENED_MAX_DIM = 10_000` | `FailClosed`, `BoundedRecovery` |
| `fsci-spatial` | `KDTree::new_with_mode` | `HARDENED_MAX_DIM = 10_000` | `FailClosed`, `BoundedRecovery` |
| `fsci-cluster` | `kmeans_with_mode` | `HARDENED_MAX_DIM = 10_000` | `FailClosed`, `BoundedRecovery` |
| `fsci-io` | `mmread_with_mode` | `HARDENED_MAX_DIM = 10_000` | `FailClosed`, `BoundedRecovery` |
| `fsci-fft` | Mode-aware transforms | Transform length limits | `FailClosed` |
| `fsci-stats` | Distribution fitting / moments | Sample size limits | `FailClosed`, `BoundedRecovery` |


### Audit Ledger: Schema and Lifecycle

There are **two complementary records** in flight:

1. **`SolveCertificate`** is synchronously returned on every `solve_with_casp`-style call. It carries the full CASP decision data needed for replay (action, rcond estimate, structural evidence, posterior over condition states, expected losses, chosen expected loss, fallback-active flag).
2. **`AuditEvent`** is appended to the configured `SyncSharedAuditLedger` for *forensic* events: mode decisions, bounded recoveries, fail-closed rejections, and alien-artifact decisions. The shape, taken straight from `crates/fsci-runtime/src/evidence.rs`:

```rust,ignore
pub struct AuditEvent {
    pub timestamp_ms:       u64,    // Unix milliseconds
    pub input_fingerprint:  String, // BLAKE3 hash of the routine's inputs
    pub action:             AuditAction,
    pub outcome:            String, // human-readable result summary
}

pub enum AuditAction {
    ModeDecision         { mode: RuntimeMode },
    BoundedRecovery      { recovery_action: String },
    FailClosed           { reason: String },
    AlienArtifactDecision{ decision: Box<AlienArtifactDecision> },
}
```

Serialized via `serde_json` with `#[serde(tag = "kind", rename_all = "snake_case")]`, so on disk an event looks like:

```jsonc
{
  "timestamp_ms":      1747432481253,
  "input_fingerprint": "blake3:8f4e…",
  "action": {
    "kind":   "fail_closed",
    "reason": "rejected: 10x11 is not square"
  },
  "outcome": "non_square_matrix"
}
```

Ledgers are bounded (`evidence_capacity` on `SolverPortfolio::new`); they evict in FIFO order once full. The `SyncSharedAuditLedger` handle is process-global and safe to share across threads.

CASP decisions themselves are replayable through the *certificate*: feed the same `(rcond_estimate, structural_evidence)` back into `SolverPortfolio::select_action()` and you will get back the same `(action, posterior, expected_losses, chosen_expected_loss)`, modulo the calibrator-drift override. The conformance harness uses this round-trip to verify CASP determinism: a packet is allowed to reorder events, but not to produce a different *decision* for the same *evidence* under the same mode.

### Conformance Artifact Topology

There are two coexisting packet shapes today:

**Current `FSCI-P2C-NNN` packets** ship a *flat* layout, with the oracle capture and parity sidecars at the packet root and per-case diffs in subdirectories:

```text
fixtures/artifacts/FSCI-P2C-002/
├── oracle_capture.json             # raw SciPy oracle output for this packet
├── parity_report.json              # FrankenSciPy-vs-oracle parity tallies
├── parity_report.raptorq.json      # RaptorQ systematic-encoding sidecar
├── parity_report.decode_proof.json # RaptorQ decode-proof artifact
├── diff/                           # per-case JSON diffs
├── differential/                   # dispatch-lane differential cases
└── e2e/                            # golden-journey end-to-end traces
```

**Legacy `P2C-NNN` packets** ship the older seven-subdirectory shape, kept for backwards reference:

```text
fixtures/artifacts/P2C-002/
├── anchor/      # raw SciPy oracle captures
├── contracts/   # locked-schema invariants for this packet
├── evidence/    # CASP audit-ledger snapshots aligned to the diff cases
├── perf/        # criterion baselines + tail-latency reports
└── threats/     # adversarial fixtures + recovery results
```

The three contract schemas kept in [`docs/schemas/`](docs/schemas/) (`behavior_ledger.schema.json`, `contract_table.schema.json`, `threat_matrix.schema.json`) are **topology-locked**: changes require an explicit governance proposal in the beads tracker, owner approval, a `docs/ARTIFACT_TOPOLOGY.md` update, and a zero-regression confirmation on all existing artifacts. CI gate G7 validates that every packet's contracts conform to the locked schemas. Converging the two packet layouts (legacy `P2C-*` and new `FSCI-P2C-*`) onto a single topology is one of the V1.0 roadmap items below.

### Security & Threat Model

The "defend against numerical instability abuse" doctrine is concrete. The threat matrix for every major subsystem covers at least the following classes:

| Threat class | Example | Defense |
|---|---|---|
| Malformed array metadata | A `Matrix` claiming `(rows=10, cols=10)` whose first row has 11 entries | Shape validation at the API boundary; fail-closed in both modes |
| Non-finite entries | `NaN` or `Inf` injected into solver input | Fail-closed in Strict; opt-in projection in Hardened with recovery emission |
| Ill-conditioned-input DoS | Adversary submits a million `rcond≈1e-16` matrices to exhaust the SVD-fallback budget | CASP rate-limits via the conformal-calibrator drift detector; calibrator drift triggers a conservative override |
| Algorithmic complexity attacks | Pathological inputs that make adaptive integrators subdivide forever | Adaptive Gauss-Kronrod short-circuits on non-finite integrands; LGMRES guards lucky-breakdown; subdivision limits are enforced |
| Pathological convergence | Optimizers run forever | Every minimizer carries explicit `max_iter` + `max_fev` + `tol` budgets; budget exhaustion is a recoverable error, not a hang |
| Resource exhaustion | A user passes a `10⁹ × 10⁹` matrix | Hardened mode caps dimension at `HARDENED_MAX_DIM`; Strict mode trusts the caller |
| Artifact tampering | Someone edits `parity_report.json` post-hoc | RaptorQ systematic encoding produces a decode proof; integrity scrub runs in CI gate G8 |

For every major subsystem, the threat-matrix JSON enumerates the attacker capabilities, the defended invariants, and adversarial fixtures that exercise each invariant. Fuzz targets (under `fuzz/`) run nightly on these adversarial fixtures.

### asupersync Integration

FrankenSciPy does not own an async runtime, and none of its public API is async. What [asupersync](https://github.com/Dicklesworthstone/asupersync) provides today:

- **RaptorQ systematic encoding.** `fsci-conformance` uses `asupersync::raptorq::systematic::SystematicEncoder` to emit the `*.raptorq.json` sidecars and `*.decode_proof.json` artifacts for parity reports, oracle captures and benchmark baselines. This is the only asupersync code path linked into the workspace.
- **Nothing else yet.** No function takes a `Cx`; the FFT plan cache, the conformance report-writer lock and the CASP calibrator state are guarded by `std::sync` primitives; the audit ledger handle is `Arc<std::sync::Mutex<AuditLedger>>`; and no test uses `LabRuntime`. The spec's "mandatory future expansions" (packet-level decode replay proofs for real recovery events, supervision integration in the policy controllers, anytime-valid invariant monitors) remain roadmap items with no bead behind them as of 2026-09-03.

**Forbidden.** The workspace forbids `tokio`, `hyper`, `reqwest`, `axum`, `tower` (tokio adapter), `async-std`, `smol`, and any crate that transitively depends on them. `cargo tree -i tokio` returns empty on this workspace.

### Distribution Moment Surface

Every concrete distribution in `fsci-stats` (continuous and discrete) ships explicit `mean`, `var`, `skewness`, `kurtosis`, `entropy`, `mode`, and `fit` implementations. The matrix of how each moment is computed:

| Method | Closed form (preferred) | Numerical fallback | Documented `NaN` |
|---|---|---|---|
| `skewness` | ~50 continuous + ~10 discrete (most named families) | Simpson on PDF, polygamma cumulants, raw-moment integration, U-substitution | Alpha, heavy-tail Pareto with `α ≤ 3`, etc., when the third moment diverges |
| `kurtosis` | ~50 continuous + ~10 discrete | Same as skewness | Same families when the fourth moment diverges |
| `entropy` | ~60 continuous + ~10 discrete | Simpson on PDF, quantile-space Simpson, arctan-compactified Simpson for fat-tailed continuous, change-of-variable Simpson for bounded-support cases | None; entropy is always well-defined for the supported families |
| `mode` | 27 continuous + 4 discrete with closed-form modes; the remainder fall through to numerical root-finding on `pdf'` | Bracketed Newton or golden-section on `pdf'` | Multi-modal families documented explicitly |
| `fit` | Closed-form MLE where derivable (Normal, Exponential, Uniform, …); analytic method-of-moments for shape-family distributions; boundary fit for support-determined parameters | Numerical MLE via L-BFGS-B with explicit parameter bounds | Distributions with non-identifiable parameter combinations document the constraint |

The conformance harness anchors every closed-form against SciPy at ≥3 parameter values per distribution and demands ≤1e-8 relative error for algebraic formulas (≤1e-12 for purely-symbolic ones).

### Comparison with the prior Rust numerical ecosystem

| Project | Scope | What FrankenSciPy adds |
|---|---|---|
| [`nalgebra`](https://github.com/dimforge/nalgebra) | Linear algebra primitives | A full SciPy-shape surface (~750 functions vs. low-level matrix ops), CASP runtime selection, audit ledger, mode model |
| [`ndarray`](https://github.com/rust-ndarray/ndarray) + `ndarray-linalg` | N-dimensional arrays with optional LAPACK FFI | No FFI to LAPACK (`#![forbid(unsafe_code)]`), full SciPy-shape numerical surface, conformance against SciPy |
| [`rustfft`](https://github.com/ejmahler/RustFFT) | FFT only | DCT/DST, FHT, Hilbert analytic signal; conformance against `scipy.fft` |
| [`argmin`](https://github.com/argmin-rs/argmin) | Optimization framework | Full `scipy.optimize` parity surface (minimizers + roots + curve_fit + global + LP/MILP), CASP minimizer selector |
| [`linfa`](https://github.com/rust-ml/linfa) | ML algorithms | Numerical kernels under those algorithms, distribution moment surface, hypothesis tests, QMC engines |
| [`statrs`](https://github.com/statrs-dev/statrs) | Statistical distributions | 95+ continuous and 10+ discrete distributions vs. ~20, full closed-form moment surface, full SciPy-parity tests, QMC engines |
| [`peroxide`](https://github.com/Axect/Peroxide) | Numerical methods | Audit ledger, mode model, conformance harness, much larger surface area |

FrankenSciPy is *not* a "use this instead" replacement for these crates if you only need one corner of the numerical surface; they are excellent at what they do. Reach for FrankenSciPy when you want the full SciPy-shape API and a runtime that emits decision certificates.

### Performance Discipline

The mandatory optimization loop for any change that claims to be a performance win:

1. **Baseline.** Record `p50` / `p95` / `p99` and memory budget for the affected criterion bench. The baseline goes into `fixtures/artifacts/FSCI-P2C-XYZ/perf/baseline.json`.
2. **Profile.** Identify the real hotspots using `cargo flamegraph` or `perf record`. **No optimization is allowed on guessed hotspots.**
3. **Implement one lever.** A single, focused change with a documented expected effect.
4. **Prove behavior unchanged.** Re-run the relevant differential conformance lane; the parity report must show no new diffs.
5. **Re-baseline and emit delta artifact.** The criterion delta goes into `perf/delta.json` next to the baseline; CI gate G6 compares.

Optimizations that pass all five steps land as commits with `perf(<crate>):` prefixes. Optimizations that fail any step are reverted or held in a beads issue until they pass.

### Python Oracle Workflow

Capturing fresh SciPy reference outputs is a first-class workflow:

```bash
# 1. Activate a venv with scipy installed
source .venv/bin/activate

# 2. Capture a packet's oracle into the locked artifact tree
cargo run -p fsci-conformance --bin live_oracle_capture -- \
    --packet FSCI-P2C-002 \
    --output crates/fsci-conformance/fixtures/artifacts/FSCI-P2C-002/anchor/oracle_capture.json

# 3. Generate the RaptorQ sidecar for durability
cargo run -p fsci-conformance --bin raptorq_sidecar -- \
    --generate crates/fsci-conformance/fixtures/artifacts/FSCI-P2C-002/anchor/oracle_capture.json

# 4. Re-run the differential lane against the freshened oracle
cargo test -p fsci-conformance --test diff_linalg -- --nocapture

# 5. If the parity report is clean, commit both the oracle and the sidecar
git add crates/fsci-conformance/fixtures/artifacts/FSCI-P2C-002/anchor/
git commit -m "anchor(FSCI-P2C-002): refresh SciPy oracle to <scipy-version>"
```

The full workflow (capture, regen, provenance, CI lane) lives in `docs/ORACLE_WORKFLOW.md`.

### Roadmap to V1.0

V1.0 is gated on the following items. Items 1 and 2 of the original list are done; the rest are open and, as of 2026-09-03, have no bead behind them:

1. **Surface coverage** — done by name: 1,194 of 1,300 SciPy callables have a same-named public equivalent (`fsci-special` 98.6%, `fsci-sparse` 96.2%, `fsci-fft` 90.2%, `fsci-opt` 84.5%; see [`PARITY-COVERAGE.md`](docs/planning/PARITY-COVERAGE.md)). What remains is **behavioural** coverage. A 2026-08-24 audit found 201 SciPy-named public entry points with no reference anywhere in the conformance corpus (`frankenscipy-ivxx6`); one sampled at random (`RbfInterpolator`) implemented a non-default variant until fixed, five sampled from `fsci-linalg` agreed with SciPy, and by 2026-08-30 `scripts/conformance_coverage_audit.py` reports zero unreferenced entry points. That audit is name-mention based, so "referenced" is weaker than "compared"; a per-routine list of what each `diff_*` file actually asserts does not exist yet.
2. **The three signal defects** originally listed here (`r1vok` periodogram/welch normalization, `cw6k2` iirnotch `r` approximation, `ot7tm` gausspulse envelope) closed on 2026-05-20.
3. **A CI run that passes.** The workflow was restructured on 2026-09-03 (`frankenscipy-liel6`) so that the per-crate unit suites, the full live-SciPy differential corpus, and the evidence packs actually run; the first fully green run has to be cited before any of the gate claims in this README count as enforced.
4. **Promote `fsci-arrayapi` from `aspirational` to `parity_green`.** No other crate depends on it today; wiring the audited backend through linalg / opt / sparse as the canonical array type is unstarted.
5. **Extend CASP beyond `fsci-linalg`.** The sparse, optimize and special selectors are rule-based today (see **Condition-Aware Solver Portfolio**); a loss matrix, posterior and calibrator per domain is the design intent and is unbuilt. The strict/hardened mode split and audit ledger emission have been wired into `fsci-signal`, `fsci-ndimage`, `fsci-interpolate`, `fsci-spatial`, `fsci-cluster`, and `fsci-io` with `HARDENED_MAX_DIM` enforcement (`frankenscipy-mlizi`).
6. **Cut a tagged 0.x release with a publish-to-crates.io workflow** and per-crate semver guarantees. There are no tags, no releases, and no `[profile.release]` in the root manifest yet.
7. **Converge the artifact topology.** Both the legacy `P2C-*` tree and the flat `FSCI-P2C-*` tree are still in `crates/fsci-conformance/fixtures/artifacts/`; pick one, migrate, and freeze it as V1's contract.

As of 2026-09-03 the tracker holds ~30 open and ~25 in-progress beads, nearly all of them performance-campaign conversions or measurement infrastructure. Run `bv --robot-triage` for the live picture.

---

## CASP Internals, Code Patterns, and Glossary

These sections are reference material. Read them when you want to understand the CASP decision surface mathematically, walk through more usage patterns, find a specific algorithm, or learn how the project is developed day to day.

### CASP: The Decision Surface, Mathematically

The five moving parts of CASP are all reachable from `crates/fsci-runtime/src/lib.rs`:

#### 1. Condition posterior `P(state | rcond)`

A soft assignment from a single scalar (the reciprocal condition estimate) to the four-state condition distribution, by *logistic blending* between four canonical centers in `log₁₀(rcond)` space:

```text
center( WellConditioned   ) = -2.0
center( ModerateCondition ) = -6.0
center( IllConditioned    ) = -11.0
center( NearSingular      ) = -16.0
```

`condition_posterior(rcond)` returns `[1, 0, 0, 0]` for `log₁₀(rcond) ≥ -2`, `[0, 0, 0, 1]` for `log₁₀(rcond) ≤ -16`, and a linearly-blended two-bucket mass in between. NaN or non-positive `rcond` is treated as `NearSingular` by construction.

The choice of centers is deliberate: the boundaries align with the regimes where SciPy's own routine families (LAPACK's `gesv`, `gels`, `gelsd`) typically transition between numerical strategies.

#### 2. Expected loss per action

For each candidate action `a`, the expected loss is the inner product of that action's row of the 5×4 loss matrix with the posterior:

```text
E[loss(a)] = Σ_s  loss_matrix[a][s] · posterior[s]
```

The action chosen is `argmin_a E[loss(a)]`, but the candidate set itself depends on the structural evidence. Only when the input is `Diagonal` does `DiagonalFastPath` enter the candidate set; only when it is `Triangular` does `TriangularFastPath`. General matrices choose among `{DirectLU, PivotedQR, SVDFallback}`.

#### 3. Conformal calibrator

A `ConformalCalibrator { alpha: 0.05, window: 200 }` tracks the empirical miscoverage rate on the last 200 calls. The default target is 5%; if the empirical miscoverage exceeds `alpha + epsilon`, `should_fallback()` returns `true` and `select_action()` overrides the loss-minimizing choice with `SolverAction::SVDFallback`.

This is the runtime backstop that catches drift between the loss-matrix model and reality. When CASP's posteriors stop being predictive of actual numerical failure, the calibrator catches it and falls back to the most defensively-stable option.

#### 4. Policy controller

`PolicyController` (in `policy.rs`) consumes `DecisionSignals` and emits `PolicyDecision`. This layer sits *above* the per-call action selection; it handles mode transitions, alien-artifact decisions, and bounded-recovery dispatch.

#### 5. Evidence ledger

`SolverEvidenceEntry` records carry the rcond, the structural evidence, the chosen action, and a follow-up `outcome` (success / numerical-warning / failure). The ledger is bounded by `evidence_capacity` and the calibrator's nonconformity-score window is fed from it.

### More Code: Real-World Patterns

> The snippets below name real public functions and types; constructors and option-struct field sets are stable but may evolve before V1. When in doubt, run `cargo doc -p <crate>` for the live signatures.

#### Sparse Krylov solve (Conjugate Gradient)

```rust
use fsci_sparse::{CsrMatrix, Shape2D, cg, IterativeSolveOptions};

fn main() {
    // 5×5 SPD tridiagonal Laplacian, assembled directly in CSR
    // (data / column indices / row pointers, as scipy.sparse.csr_matrix takes them).
    let data = vec![
         2.0, -1.0,
        -1.0,  2.0, -1.0,
        -1.0,  2.0, -1.0,
        -1.0,  2.0, -1.0,
        -1.0,  2.0,
    ];
    let indices = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4];
    let indptr = vec![0, 2, 5, 8, 11, 13];
    // The final flag asks the constructor to canonicalize (sort + merge duplicates).
    let a = CsrMatrix::from_components(Shape2D::new(5, 5), data, indices, indptr, true)
        .expect("CSR build");
    let b = vec![1.0, 0.0, 0.0, 0.0, 1.0];

    let mut opts = IterativeSolveOptions::default();
    opts.max_iter = Some(100);
    opts.tol = 1e-10;

    let result = cg(&a, &b, None, opts).expect("cg");
    println!("solution = {:?}", result.solution);
    println!("converged = {}, iterations = {}, residual = {:.3e}",
        result.converged, result.iterations, result.residual_norm);
}
```

#### Signal: zero-phase Butterworth band-pass

```rust
use fsci_signal::{butter, filtfilt, FilterType};

fn main() {
    let fs   = 1_000.0;            // 1 kHz sample rate
    let nyq  = fs / 2.0;
    let band = [50.0 / nyq, 200.0 / nyq];

    // 4th-order Butterworth band-pass; returns BaCoeffs { b, a }
    let coeffs = butter(4, &band, FilterType::Bandpass).expect("design");

    // Synthesize a 100 Hz tone
    let n: usize = 2048;
    let mut x = vec![0.0_f64; n];
    for (i, xi) in x.iter_mut().enumerate() {
        let t = i as f64 / fs;
        *xi = (2.0 * std::f64::consts::PI * 100.0 * t).sin();
    }

    // Zero-phase forward/reverse application
    let y = filtfilt(&coeffs.b, &coeffs.a, &x).expect("filtfilt");
    println!("first 5 filtered samples: {:?}", &y[..5]);
}
```

#### FFT round-trip

```rust
use fsci_fft::{rfft, irfft, FftOptions};

fn main() {
    let signal: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.01).sin()).collect();
    let opts = FftOptions::default();

    let spectrum = rfft(&signal, &opts).expect("rfft");
    let reconstructed = irfft(&spectrum, Some(signal.len()), &opts).expect("irfft");

    // The plan cache is shared and warmed after the first call at this length;
    // identical (length, direction, normalization) requests are dispatched to
    // the cached plan rather than rebuilt.
    println!("max round-trip error = {:.3e}",
        signal.iter().zip(&reconstructed)
              .map(|(a, b)| (a - b).abs())
              .fold(0.0_f64, f64::max));
}
```

#### KDTree nearest-neighbor

```rust
use fsci_spatial::KDTree;

fn main() {
    let points = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![0.5, 0.5],
    ];
    let tree = KDTree::new(&points).expect("build");
    let (idx, dist) = tree.query(&[0.6, 0.4]).expect("query");
    println!("nearest neighbor: point {:?} at distance {:.4}", points[idx], dist);
}
```

#### Hypothesis test + bootstrap confidence interval

```rust
use fsci_stats::{ttest_ind_welch, bootstrap_ci};

fn main() {
    let a = vec![5.1, 5.3, 5.0, 4.9, 5.2, 5.4, 5.1];
    let b = vec![4.8, 4.7, 4.9, 5.0, 4.8, 4.6, 4.9];

    // Welch's two-sample t-test (unequal variances)
    let t = ttest_ind_welch(&a, &b);
    println!("Welch t  = {:.4}, p = {:.4}", t.statistic, t.pvalue);

    // Percentile bootstrap 95% CI on the mean of `a`
    let stat = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    let (ci_low, ci_high) = bootstrap_ci(&a, stat, 9_999, 0.95, 42);
    println!("mean(a) 95% CI = [{:.4}, {:.4}]", ci_low, ci_high);
}
```

#### QMC sampling for a low-discrepancy design

```rust
use fsci_stats::qmc::SobolSampler;

fn main() {
    // 2-dimensional digital-shifted Sobol sampler (dimensions 1..=32 are supported;
    // HaltonSampler and LatinHypercubeSampler are unrestricted).
    let mut sobol = SobolSampler::with_digital_shift(2, /*seed=*/ 7).expect("build");
    println!("starting index = {}", sobol.next_index());
}
```

### Numerical Catalog by Domain

A function-level companion to the recipes table, handy when you are searching for "does FrankenSciPy have X?" and want to grep for the right name.

#### `fsci-linalg` public surface (selected)

```text
solve, solve_triangular, solve_banded, solveh_banded, solve_toeplitz, solve_circulant,
lstsq, pinv, inv, det, slogdet, norm, matrix_rank, matrix_power, signm,
lu, lu_factor, lu_solve,
qr, qr_insert, qr_delete, qr_update,
cholesky, cho_solve, cho_factor, ldl,
svd, svdvals,
eig, eigh, eig_banded, eigh_tridiagonal, eigvals, eigvalsh,
schur, hessenberg, qz,
expm, logm, sqrtm, funm,
solve_sylvester, solve_lyapunov, solve_continuous_are, solve_discrete_are,
solve_continuous_lyapunov, solve_discrete_lyapunov,
orth, null_space, subspace_angles, polar,
block_diag, bmat, vstack, hstack,
leslie, pascal, vander, hankel, toeplitz, circulant, dft, helmert, hilbert,
random_spd, random_matrix, mat_allclose,
solve_with_casp, solve_with_audit, lstsq_with_casp, …
```

#### `fsci-sparse` public surface (selected)

```text
Formats: CsrMatrix, CscMatrix, CooMatrix, BsrMatrix, DiaMatrix, DokMatrix, LilMatrix,
         eye, eye_rectangular, diags, kron, kronsum,
         vstack, hstack, block_diag, stack, bmat,

Direct:  spsolve, spsolve_triangular, splu, spilu, splu_solve,

Krylov:  cg, pcg, bicg, bicgstab, cgs, gmres, lgmres, qmr, minres, lsqr, lsmr,

Spectral: eigs, eigsh, svds,

Graph:   dijkstra, bellman_ford, floyd_warshall, shortest_path,
         connected_components, strongly_connected_components,
         minimum_spanning_tree, breadth_first_order, depth_first_order,
         reverse_cuthill_mckee, laplacian, normalized_laplacian,
         pagerank, betweenness_centrality, eigenvector_centrality,
         clustering_coefficient, graph_diameter, eccentricity,

Misc:    sparse_density, sparse_frobenius_inner, frobenius_norm,
         reorder, permute, prune, sort_indices, sum_duplicates, …
```

#### `fsci-opt` public surface (selected)

The top-level re-exports for direct method calls live on the bare algorithm
name (no `minimize_` prefix). Other methods are reachable through the
`minimize(method, …)` dispatch entry point.

```text
Unconstrained (re-exported direct): bfgs, cg_pr_plus, nelder_mead, powell,
                                    lbfgsb, newton_cg, trust_exact,

Via minimize(method, …) dispatch:   nelder_mead, bfgs, cg, powell, lbfgsb,
                                    newton_cg, tnc, cobyla, slsqp,
                                    trust_ncg, trust_krylov, trust_exact,
                                    trust_constr, dogleg,
Plus:  minimize, minimize_with_audit, select_minimize_method,
       minimize_scalar (brent, bounded, golden, trisection),

Roots: brentq, brenth, bisect, ridder, toms748, newton_scalar, halley, secant,
       anderson, broyden1, broyden2, fsolve, lm_root, root, root_scalar,

Curve fitting: curve_fit, least_squares (lm + trf + dogbox), nnls,
               isotonic_regression,

Global:        differential_evolution, dual_annealing, basinhopping, shgo, brute,
               particle_swarm,

LP / MILP:     linprog (revised simplex + interior point), milp (branch and bound),

Other:         linear_sum_assignment (Hungarian), bracket,
               line_search_wolfe1, line_search_wolfe2, validate_wolfe_params,
               projected_gradient_descent, augmented_lagrangian,
               simulated_annealing, …
```

#### `fsci-integrate` public surface (selected)

```text
IVP:  solve_ivp (Rk23, Rk45, Dop853, Bdf, Radau, Lsoda),
      solve_ivp_with_audit, OdeSolution, EventSpec / EventFn,
      odeint,

BVP:  solve_bvp,

Quad: quad, quad_inf, quad_neg_inf, quad_full_inf, quad_cauchy_pv, quad_vec,
      quad_explain, dblquad, dblquad_rect, tplquad, tplquad_rect, nquad,
      fixed_quad, gauss_legendre, gauss_kronrod_quad, newton_cotes, newton_cotes_quad,
      cubature, cubature_scalar, line_integral,

Romberg: romb, romb_func, romberg, trapezoid, trapezoid_uniform, trapezoid_irregular,
         trapezoid_richardson, simpson, simpson_uniform, simpson_irregular,
         cumulative_simpson, cumulative_trapezoid, cumulative_trapezoid_uniform,
         cumulative_trapezoid_initial,

Stochastic: monte_carlo_integrate, qmc_quad,
```

### Development Workflow

FrankenSciPy is built using a small but specific stack designed for AI-driven, multi-agent development:

| Tool | What it does |
|---|---|
| [`br` (beads_rust)](https://github.com/Dicklesworthstone/beads_rust) | Local-first issue tracker. Every TODO, defect, and roadmap item lives in `.beads/issues.jsonl` and surfaces through `br ready`, `br show <id>`, `br create`. The 4,285+ closed beads are the audit trail; `CHANGELOG.md` summarizes them by domain. |
| `bv` | Graph-aware triage on top of beads. `bv --robot-triage` returns the recommended next ticket with reasons. |
| `agent-mail` | MCP messaging layer that lets multiple agents working on the repo coordinate via threaded conversations and advisory file reservations, all kept under `.agent-mail/` for auditability. |
| `rch` (Remote Compilation Helper) | Offloads `cargo build`, `cargo test`, and `cargo clippy` to a fleet of remote workers; this is how the project keeps the developer machine responsive when several agents are compiling the workspace in parallel. |
| `ubs` (Ultimate Bug Scanner) | `ubs <files>` runs a fast pre-commit lint that catches the project's recurring bug classes (unwrap panics, NaN propagation gaps, audit-emission omissions). |

The repo's [`AGENTS.md`](AGENTS.md) is the canonical guide for agents and humans alike; it spells out the conventions for branching, mode usage, beads workflow, and how to keep `main` ↔ `master` in sync.

### Reproducibility Ledger

Every release-bearing artifact in FrankenSciPy carries enough metadata to be reproduced end-to-end from a clean checkout:

1. **Source pin.** The git commit hash that produced the artifact.
2. **Toolchain pin.** The nightly Rust channel from `rust-toolchain.toml`.
3. **Oracle pin.** The exact SciPy version used to produce `oracle_capture.json` (recorded in the RaptorQ sidecar metadata).
4. **Fingerprint.** The BLAKE3 hash of the input matrix / array (recorded in every audit event).
5. **Decode proof.** The RaptorQ systematic-encoding proof that demonstrates the artifact was decodable without repair. When repair was used, the proof names the missing symbols and the repair set that reconstructed them.

In practice: if a parity-report regression appears six months from now, you can rebuild the *exact* conditions that produced the regression (same git, same toolchain, same SciPy, same input bytes) and re-emit the same audit events. The conformance harness checks this property explicitly: re-running a packet from a clean checkout must produce a bit-identical `parity_report.json` (modulo timestamps).

### Project Family

FrankenSciPy is part of a deliberate family of "Franken*" projects that share the same engineering doctrine: `#![forbid(unsafe_code)]`, no tokio, audit ledgers, RaptorQ-backed artifacts, strict-vs-hardened modes, differential conformance against a legacy oracle.

| Project | Domain | Relationship |
|---|---|---|
| **[asupersync](https://github.com/Dicklesworthstone/asupersync)** | Async runtime | The exclusive async runtime FrankenSciPy depends on |
| **[ftui](https://github.com/Dicklesworthstone/ftui)** | Terminal UI rendering | Drives the `conformance_dashboard` binary |
| **[FrankenTerm](https://github.com/Dicklesworthstone/frankenterm)** | Terminal emulator (a WezTerm fork) | Sibling project; same doctrine |
| **[FrankenSQLite](https://github.com/Dicklesworthstone/frankensqlite)** | SQLite re-implementation | Sibling project; shares the `docs/planning/SPEC_CROSSWALK_FRANKENSQLITE_TO_FRANKENSCIPY.md` doctrine alignment |
| **[FrankenLibc / FrankenFS](https://github.com/Dicklesworthstone/)** | libc + filesystem | Where the strict-vs-hardened compatibility-security model first crystallized |
| **[beads_rust (`br`)](https://github.com/Dicklesworthstone/beads_rust)** | Issue tracker | The dependency-aware ticket database used to drive work |

Reading any one of these gives you ~70% of the conventions used in the others, so an agent fluent in one ramps onto the next quickly.

### Glossary

| Term | Meaning |
|---|---|
| **CASP** | Condition-Aware Solver Portfolio. The runtime algorithm-selection engine that minimizes expected loss over a calibrated decision matrix. |
| **Loss matrix** | The 5-action × 4-state table of costs that drives CASP. `SolverPortfolio::default_loss_matrix()`. |
| **Conformal calibrator** | The drift detector that watches CASP's empirical miscoverage and falls back to SVD when CASP becomes unreliable. |
| **SolveCertificate** | The synchronously-returned record of a CASP decision: action, rcond, structural evidence, posterior, expected losses, chosen loss, fallback flag. |
| **AuditEvent** | The asynchronous, forensic event written into the `SyncSharedAuditLedger`: timestamp, input fingerprint, action variant (`ModeDecision` / `BoundedRecovery` / `FailClosed` / `AlienArtifactDecision`), outcome. |
| **Strict mode** | `RuntimeMode::Strict`: SciPy-parity behavior, no auto-repair, fail-closed on malformed input. |
| **Hardened mode** | `RuntimeMode::Hardened`: preserves the API contract, applies bounded recovery for malformed inputs, caps resource use at `HARDENED_MAX_DIM`. |
| **Conformance packet** | A `FSCI-P2C-NNN` (or legacy `P2C-NNN`) directory under `crates/fsci-conformance/fixtures/artifacts/` containing oracle captures, parity reports, RaptorQ sidecars, and per-case diffs for one slice of SciPy. |
| **Oracle** | A Python script under `crates/fsci-conformance/python_oracle/` that imports `scipy.*` and emits reference outputs for the cases in a packet. |
| **Three-lane harness** | The conformance pattern of running each packet in self-check, oracle-backed, and dispatch lanes. |
| **RaptorQ sidecar** | The systematic erasure-encoding artifact that lets a damaged conformance bundle be repaired without source regeneration; carries the decode proof. |
| **Decode proof** | The JSON artifact `parity_report.decode_proof.json` that names the repair symbols used and proves the artifact was decodable. |
| **Mode-split** | The discipline of having every CASP-participating routine accept a `RuntimeMode` so callers can pick strict vs hardened semantics per call. |
| **Fail-closed** | Refusing to produce a result when input violates an invariant, emitting a `FailClosed` audit event with the reason. |
| **Bounded recovery** | An explicit, audit-logged repair of malformed input (for example, projecting NaN entries to zero, or regularizing a near-singular matrix). Only available in Hardened mode. |
| **Beads** | The local-first issue tracker (`.beads/issues.jsonl`) accessed via `br` / `bv`. |
| **Alien-artifact decision** | A reified high-stakes decision that escapes the normal CASP flow (e.g., a routine that must contact an external oracle); recorded as a first-class audit-event variant. |

---

## Motivation, Policies, and Distributions

Background and reference material: motivation, when to reach for FrankenSciPy (and when not to), the tolerance policy, the error model, the engineering invariants the project commits to, real case studies of bugs the conformance harness has caught, a compendium of supported distributions, a migration guide from `scipy.*` Python code, and a short bibliography of the literature that shaped the design.

### Motivation

The original SciPy is one of the most successful scientific software projects ever written. The Python+C+Fortran architecture got the world enormously far. But three pressures have been accumulating for the last decade and don't have a clean answer inside the original stack:

1. **Numerical code is now embedded in *services*, not just notebooks.** A modern data pipeline calls a hypothesis test inside a request handler; an agent system calls an optimizer from a loop that has to be cancel-correct; a billing system has a Monte Carlo VaR estimate sitting behind a 50 ms tail-latency SLO. In all of these, Python's GIL, garbage collector, and import-time footprint are showstoppers.
2. **Numerical correctness deserves audit trails.** When a `solve` call returns a vector, the caller has no first-class way to ask "*how* did you solve this, and on what evidence?" In production systems where ill-conditioning maps onto real money or real safety, that opacity is unacceptable.
3. **The supply chain wants memory safety.** Wrapping LAPACK is convenient. It also means every numerical kernel ultimately depends on a C/Fortran call surface that has demonstrable use-after-free history, no thread-safety guarantees from the language, and no easy path to formal verification.

FrankenSciPy is the answer to those three pressures as a single coherent system, not a partial fix for any one of them. Rust gives us (1) and (3) for free. CASP, the audit ledger, and the conformance harness give us (2) as a first-class API.

### When to reach for FrankenSciPy (and when not to)

**Reach for it when:**

- You are writing a long-running service in Rust that needs numerical kernels and you do not want to embed Python.
- You need to *prove* (in a postmortem, an audit, or a regulator's report) which algorithm produced a numerical result and why.
- You need to drop a numerical routine into a memory-constrained, supply-chain-strict, or WebAssembly environment.
- You want a SciPy-shape API surface with cancel-correct async semantics from `asupersync` instead of tokio.
- You are building an agentic system that needs numerical kernels alongside per-decision audit ledgers.

**Don't reach for it when:**

- You are doing interactive data analysis in a notebook; SciPy + matplotlib + pandas is still the right stack.
- You need a routine that lives outside the 750+ FrankenSciPy already covers and you can't wait for it to land. (Run `cargo doc -p <crate>` to confirm; file a beads issue if you find a gap that matters to you.)
- You need GPU or distributed-cluster execution. Single-process CPU is the current scope.
- You are willing to wrap LAPACK directly and don't need `#![forbid(unsafe_code)]`. Then `ndarray-linalg` plus `nalgebra` will give you better large-matrix throughput.

### Tolerance Policy

Every routine in scope has an explicit tolerance contract. The contract has the form `(rule, magnitude)`:

| Rule | Default magnitude | Where it applies |
|---|---|---|
| `abs ≤ atol` | `atol = 1e-12` | Closed-form algebraic formulas (e.g. `gamma(5) == 24`) |
| `abs ≤ atol` | `atol = 1e-10` | Carlson elliptic family, Bessel functions on the principal branch, orthogonal polynomial evaluations |
| `rel ≤ rtol` | `rtol = 1e-8` | Distribution PDF/CDF/PPF round-trips, hypothesis-test statistics, descriptive statistics |
| `rel ≤ rtol` | `rtol = 1e-6` | Iterative solvers' final residual, optimizer convergence, ODE step error |
| `rel ≤ rtol` | `rtol = 1e-4` | Numerical-Simpson-fallback distribution moments (`entropy`, higher cumulants for fat-tailed families) |
| `bounded f64` | `≤ HARDENED_MAX_DIM` | Hardened-mode dimension cap |

These are the prevailing tolerance bands across the conformance harness; the literal per-routine tolerances are encoded in the differential test cases themselves (each packet's harness names the `atol` / `rtol` for the diff). The `tolerance_lint` binary (CI gate G9) ratchets the aggregated picture so a contributor cannot loosen a tolerance without an explicit governance change.

### Error Model

Every domain crate exposes a tagged error enum with no panicking constructors:

| Crate | Error type | Representative variants |
|---|---|---|
| `fsci-linalg` | `LinalgError` | `ExpectedSquareMatrix`, `IncompatibleShapes { a_shape, b_len }`, `NonFiniteInput`, `SingularMatrix`, `ConvergenceFailure { detail }`, `ConditionTooHigh { rcond, threshold }`, `ResourceExhausted { detail }`, `PolicyRejected { reason }`, `InvalidArgument { detail }` |
| `fsci-sparse` | `SparseError` | `InvalidShape { message }`, `IncompatibleShape { message }`, `InvalidArgument { message }` |
| `fsci-opt` | `OptError` | `InvalidArgument { detail }`, `InvalidBounds { detail }`, `SignChangeRequired { detail }`, `NonFiniteInput { detail }`, `EvaluationBudgetExceeded { detail }`, `NotImplemented { detail }` |
| `fsci-integrate` | `IntegrateValidationError` + `BvpError` | shape/dimension validators, BVP-specific convergence errors |
| `fsci-fft` | `FftError` | `InvalidShape`, `InvalidWorkers`, `LengthMismatch`, `NonFiniteInput`, `NonPositiveSampleSpacing` |
| `fsci-special` | `SpecialError` | domain/argument validation |
| `fsci-stats` | `StatsError` + `FitError` | `StatsError::InvalidArgument`, `FitError::NotImplemented { detail }` |
| `fsci-spatial` | `SpatialError` | `EmptyData`, `DimensionMismatch { expected, actual }`, `InvalidArgument(String)` |
| `fsci-signal` | `SignalError` | filter-design and frequency-warping validation |
| `fsci-cluster` | `ClusterError` | clustering-input validation |
| `fsci-interpolate` | `InterpError` | interpolation-input validation |

The remaining crates (`fsci-arrayapi`, `fsci-constants`, `fsci-datasets`,
`fsci-ndimage`, `fsci-io`, `fsci-odr`) define their own error enums where
needed. Not every API in those crates returns `Result`; pure-data crates
like `fsci-constants` and `fsci-datasets` mostly expose infallible accessors.

The shared rules:

1. **`derive(thiserror::Error)` everywhere.** The display message is meant to be human-readable; the variant tag is meant to be machine-matchable.
2. **No panicking constructors on user-facing public APIs.** Distributions are the documented exception: `BetaDist::new(a, b)` panics on bad input because the type's safety invariant is enforced at construction. The `Distribution::fit` default also panics with `FitError::NotImplemented` for distributions that have no override; use the non-panicking `try_fit(&data)` companion (returns `Result<Self, FitError>`) if you cannot guarantee the call site has a fittable distribution.
3. **Fail-closed in Strict mode.** Errors are not retried, not silently downgraded to warnings, and not auto-repaired.
4. **Audit events on every fail-closed.** The error returned to the caller is *also* recorded in the audit ledger as a `FailClosed` event with the input fingerprint and the reason string.

### Engineering Invariants

Eleven invariants that the workspace commits to and the CI gates enforce:

1. **`#![forbid(unsafe_code)]`** at the workspace root (`Cargo.toml [workspace.lints.rust]`); no unsafe blocks anywhere in `crates/`.
2. **No tokio.** `cargo tree -i tokio` returns empty. The same applies to `hyper`, `reqwest`, `axum`, `tower` (tokio adapter), `async-std`, `smol`.
3. **Rust 2024 edition, nightly toolchain**, pinned via `rust-toolchain.toml`.
4. **`cargo fmt --check`** passes on every commit.
5. **`cargo clippy --workspace --all-targets -- -D warnings`** passes; pedantic + nursery lints are enforced.
6. **`cargo doc` is kept warning-free by convention** (math notation and generic parameters in doc comments are backtick-wrapped); it is not a CI gate today.
7. **No tolerance contract is ever loosened.** CI gate G9 (`tolerance_lint`) fails the build on weakening — the `FSCI-P2C-*.json` fixture cases via a monotone violation threshold, and every `tests/diff_*.rs` harness tolerance (`const *_TOL: f64` plus bare `<= <literal>` comparisons) against the committed `fixtures/tolerance_baseline.json`: a loosened value, an unanchored contract, or a stale baseline entry is a hard failure, and `--write-baseline` refuses to adopt loosened values so a loosening can only land as a hand-edited baseline reviewers can grep for (frankenscipy-6a5s9).
8. **No schema is ever broken silently.** CI gate G7 validates the three contract schemas against every packet.
9. **No artifact ships without a RaptorQ sidecar.** CI gate G8 verifies the decode proof.
10. **Every fix-closed defect has a beads issue.** The open numerical defects listed in **Limitations** are each a live bead; the three signal defects that used to be listed there closed on 2026-05-20.
11. **`main` and `master` stay in sync.** `master` exists only for legacy URL compatibility; every push to `main` is also pushed to `master`.

### Case Studies: Bugs the Harness Has Caught

Three representative bugs. Two are preserved as inline `br-<id>` notes in the source (grep the id; line numbers rot); the third's kernel has since been replaced:

#### `eigsh` on the path Laplacian (the original power-iteration kernel)

Power iteration is mathematically guaranteed to find the dominant eigenvalue *only when the initial vector has nonzero projection onto its eigenvector*. The constant vector `[1/√n, …, 1/√n]`, a natural default, is orthogonal to every alternating-sign mode. On a path-Laplacian matrix the dominant eigenvector is exactly such an alternating-sign mode, so the iteration silently converged to the *wrong* eigenvalue. The fix at the time replaced the constant seed with a deterministic pseudo-random vector. The power-iteration kernel was later replaced by Lanczos through the shared Krylov/Arnoldi solver (see **Numerical Recipes**), which does not have this failure mode; the inline note went with it.

#### `br-iq1e`: counting-before-breakdown in LGMRES inner loop

Augmented-Krylov GMRES has a "lucky breakdown" exit when the residual collapses to numerical zero. The pre-fix code decremented `k` after breaking, which made the outer loop see `k = 0` and treat the inner result as never having advanced. On identity-like operators this manifested as an infinite spin. The fix increments `k` *before* the breakdown branch so the outer loop always sees the progress that was made. The `// br-iq1e:` comment sits on that increment in `crates/fsci-sparse/src/linalg.rs`.

#### `br-nknp`: Mann-Whitney-U tie correction

The Mann-Whitney `U` test uses a normal approximation for large `n` whose standard deviation must be tie-corrected when the combined sample has repeated values. The pre-fix code dropped the tie correction, which made `p`-values systematically too small on discrete or rounded data. Cross-checked against `scipy.stats.mannwhitneyu` in the conformance harness, the difference is invisible on continuous data and obvious on integer-valued samples. The three `br-nknp` comments in `crates/fsci-stats/src/lib.rs` mark the sigma correction and the continuity-corrected two-sided branch.

Each case study has a parity test in `fsci-conformance` that would fail again if the regression returned. These are the kinds of bugs the conformance harness exists for: subtle, mathematically-grounded, easily-shipped-without-noticing.

### Distribution Compendium

A condensed list of every distribution in `fsci-stats` with its parameter convention. Use `cargo doc -p fsci-stats` for the full API surface and the formulas behind `pdf` / `cdf` / `ppf`.

#### Continuous, single-parameter

| Type | Parameters | Support |
|---|---|---|
| `Normal` | `loc`, `scale` | ℝ |
| `Uniform` | `loc`, `scale` | `[loc, loc+scale]` |
| `Exponential` | `scale` | `[0, ∞)` |
| `Cauchy` | `loc`, `scale` | ℝ |
| `Laplace` | `loc`, `scale` | ℝ |
| `Logistic` | `loc`, `scale` | ℝ |
| `Gumbel` / `GumbelLeft` | `loc`, `scale` | ℝ |
| `Pareto` | `b`, `loc`, `scale` | `[loc+scale, ∞)` |
| `Rayleigh` | `loc`, `scale` | `[loc, ∞)` |
| `Maxwell` | `loc`, `scale` | `[loc, ∞)` |
| `HalfNormal` / `HalfCauchy` / `HalfLogistic` | shape-free unit struct | `[0, ∞)` |
| `Arcsine` / `Semicircular` / `CosineDistribution` / `Anglit` / `HypSecant` | shape-free | bounded or ℝ |

#### Continuous, multi-parameter

| Type | Parameters | Notes |
|---|---|---|
| `BetaDist` | `a > 0`, `b > 0` | Support `[0, 1]` |
| `BetaPrime` | `a`, `b` | Support `[0, ∞)` |
| `Gamma` / `InverseGamma` / `Erlang` | `shape`, `scale` | |
| `ChiSquared` / `Chi` | `df` | |
| `FDistribution` | `dfn`, `dfd` | |
| `StudentT` / `NoncentralT` | `df`, optional `nc` | |
| `Weibull` (`WeibullMin` / `WeibullMax` / `FrechetR`) | `c`, `loc`, `scale` | |
| `Lognormal` / `Loglogistic` (`Fisk`) | `shape`, `loc`, `scale` | |
| `LogLaplace` | `c`, `loc`, `scale` | |
| `LogGamma` | `c`, `loc`, `scale` | |
| `GenLogistic` / `GenNorm` / `HalfGenNorm` / `GenHalfLogistic` | `c`, `loc`, `scale` | |
| `GenExtreme` / `GenPareto` | `shape`, `loc`, `scale` | |
| `Burr` (Type III), `Burr12` | `c`, `d`, `loc`, `scale` | |
| `Mielke` | `k`, `s`, `loc`, `scale` | |
| `Kappa3` / `Kappa4` | shape parameters | |
| `PowerLognorm` / `PowerLaw` / `PowerNorm` | `c`, ... | |
| `Trapezoid` / `Triangular` | corner parameters | |
| `TukeyLambda` | `lam` | |
| `Bradford` | `c`, `loc`, `scale` | |
| `Nakagami` | `nu`, `loc`, `scale` | |
| `Moyal` | `loc`, `scale` | |
| `LaplaceAsymmetric` | `kappa`, `loc`, `scale` | |
| `SkewNorm` / `SkewCauchy` / `WrapCauchy` | `a`, `loc`, `scale` | |
| `IrwinHall` | `n` | sum of `n` Uniform(0,1) |
| `JohnsonSU` / `JohnsonSB` | `a`, `b`, `loc`, `scale` | |
| `RDist` | `c` | symmetric beta |
| `Pearson3` | `skew`, `loc`, `scale` | |
| `ExponNorm` | `K`, `loc`, `scale` | normal + exponential convolution |
| `ExponWeibull` / `ExponPow` | `a`, `c`, `loc`, `scale` | |
| `FatigueLife` (Birnbaum-Saunders) | `c`, `loc`, `scale` | |
| `InverseGaussian` (Wald) | `mu`, `loc`, `scale` | |
| `RecipInvGauss` | `mu`, `loc`, `scale` | |
| `FoldedNormal` / `FoldedCauchy` | `c`, `loc`, `scale` | |
| `Levy` / `LevyLeft` | `loc`, `scale` | stable α = ½ |
| `Alpha` | `a`, `loc`, `scale` | |
| `Loguniform` (Reciprocal) | `a`, `b`, `loc`, `scale` | |
| `Gompertz` | `c`, `loc`, `scale` | |
| `Lomax` | `c`, `loc`, `scale` | Pareto II |
| `TruncNormal` / `TruncExpon` / `TruncPareto` / `TruncWeibullMin` | truncation endpoints + shape | |
| `VonMises` | `kappa`, `loc` | circular |
| `Rice` | `b`, `loc`, `scale` | |
| `Argus` | `chi`, `loc`, `scale` | |
| `CrystalBall` | `beta`, `m`, `loc`, `scale` | |
| `KsTwoBign` | shape-free | Kolmogorov-Smirnov asymptotic |
| `DoubleGamma` / `DoubleWeibull` | shape | symmetric mirror constructions |
| `Gilbrat` | shape-free | Lognormal(1, 0, 1) |

#### Discrete

| Type | Parameters | Support |
|---|---|---|
| `Bernoulli` | `p` | `{0, 1}` |
| `Binomial` | `n`, `p` | `{0, …, n}` |
| `Geometric` | `p` | `{1, 2, …}` |
| `NegBinomial` | `n`, `p` | `{0, 1, …}` |
| `Poisson` | `mu` | `{0, 1, …}` |
| `RandInt` | `low`, `high` | `{low, …, high-1}` (SciPy's `randint`) |
| `Hypergeometric` | `M`, `n`, `N` | hypergeometric population draw |
| `LogSeries` | `p` | `{1, 2, …}` |
| `Boltzmann` | `lambda`, `N` | truncated geometric on `{0, …, N-1}` |
| `Planck` | `lambda` | untruncated geometric |
| `YuleSimon` | `alpha` | `{1, 2, …}` |
| `Zipfian` | `a`, `N` | truncated Zipf on `{1, …, N}` |

Concrete continuous types implement the `ContinuousDistribution` trait and
discrete types implement `DiscreteDistribution`. Together the trait surface
exposes (provided or overridden per type) `pdf` / `pmf`, `cdf`, `sf`, `ppf`,
`isf`, `logpdf` / `logpmf`, `logcdf`, `mean`, `var`, `std`, `skewness`,
`kurtosis`, `entropy`, `rvs(n, &mut rng)`, plus `fit(&data)` (default panics
with `FitError::NotImplemented`) and the non-panicking `try_fit(&data) ->
Result<Self, FitError>`. Many types add a closed-form `mode()` accessor as
an inherent method, and several bounded-support types add a `support() ->
&[f64]` accessor.

### Migrating from `scipy.*` Python code

The Rust API is shaped to look like SciPy when you squint. The recurring translation patterns:

| SciPy (Python) | FrankenSciPy (Rust) |
|---|---|
| `scipy.linalg.solve(A, b)` | `fsci_linalg::solve(&a, &b, SolveOptions::default())` |
| `scipy.linalg.lstsq(A, b)` | `fsci_linalg::lstsq(&a, &b, LstsqOptions::default())` |
| `scipy.linalg.eigh(A)` | `fsci_linalg::eigh(&a, DecompOptions::default())` |
| `scipy.linalg.expm(A)` | `fsci_linalg::expm(&a, DecompOptions::default())` |
| `scipy.sparse.csr_matrix((data, indices, indptr), shape)` | `fsci_sparse::CsrMatrix::from_components(Shape2D::new(m, n), data, indices, indptr, true)` |
| `scipy.sparse.linalg.cg(A, b)` | `fsci_sparse::cg(&a, &b, None, IterativeSolveOptions::default())` |
| `scipy.optimize.minimize(f, x0, method='BFGS')` | `fsci_opt::bfgs(&f, &x0, MinimizeOptions::default())` |
| `scipy.optimize.minimize(f, x0, method='L-BFGS-B', bounds=...)` | `fsci_opt::lbfgsb(&f, &x0, opts)` |
| `scipy.optimize.brentq(f, a, b)` | `fsci_opt::brentq(&f, (a, b), RootOptions::default())` |
| `scipy.integrate.solve_ivp(f, [t0, tf], y0, method='RK45')` | `fsci_integrate::solve_ivp(&mut f, &SolveIvpOptions { method: SolverKind::Rk45, ... })` |
| `scipy.integrate.quad(f, a, b)` | `fsci_integrate::quad(&f, a, b, QuadOptions::default())` |
| `scipy.fft.rfft(x)` | `fsci_fft::rfft(&x, &FftOptions::default())` |
| `scipy.signal.butter(N, Wn, 'bandpass')` | `fsci_signal::butter(N, &wn, FilterType::Bandpass)` |
| `scipy.signal.filtfilt(b, a, x)` | `fsci_signal::filtfilt(&b, &a, &x)` |
| `scipy.stats.norm.pdf(x, loc, scale)` | `Normal::new(loc, scale).pdf(x)` |
| `scipy.stats.ttest_ind(a, b, equal_var=False)` | `fsci_stats::ttest_ind_welch(&a, &b)` |
| `scipy.special.gamma(x)` | `fsci_special::gamma(&SpecialTensor::RealScalar(x), RuntimeMode::Strict)` (tensor-shaped, mode-aware; the scalar helpers such as `gammaln_scalar` live alongside) |
| `scipy.special.ellipk(m)` | `fsci_special::elliprf(0.0, 1.0 - m, 1.0)` (Carlson form) |
| `scipy.spatial.KDTree(points).query(q)` | `KDTree::new(&points)?.query(&q)?` |

Three rules of thumb:

1. **Options structs replace keyword arguments.** Python's `method='BFGS', tol=1e-6` becomes a `MinimizeOptions { tol: 1e-6, ... }` literal.
2. **Outputs are typed result structs.** Where SciPy returns a tuple or a `Result` object, FrankenSciPy returns a named struct: `SolveResult`, `OptimizeResult`, `IterativeSolveResult`, `TtestResult`, `QuadResult`, etc.
3. **The CASP/audit surface is opt-in.** The plain `solve(...)` and `lstsq(...)` calls are SciPy-shape and don't require a `SolverPortfolio`. Reach for `solve_with_casp(...)` only when you want the decision certificate.

### Bibliography

Numerical methods are downstream of decades of literature. The implementations in this workspace draw on:

- **Golub, G. H. & Van Loan, C. F.** *Matrix Computations* (4th ed., Johns Hopkins, 2013) — the canonical reference for LU / QR / SVD / Schur / Hessenberg and the Parlett recurrence for matrix functions.
- **Higham, N. J.** *Functions of Matrices: Theory and Computation* (SIAM, 2008) — `expm` scaling-and-squaring, `logm` inverse-scaling-and-squaring, `sqrtm` block recurrences, the Frechét derivative discussions.
- **Trefethen, L. N. & Bau, D.** *Numerical Linear Algebra* (SIAM, 1997) — pedagogical baseline; the Householder / Wilkinson-shift discussion in the eigensolver path.
- **Saad, Y.** *Iterative Methods for Sparse Linear Systems* (2nd ed., SIAM, 2003) — CG, GMRES, BiCGSTAB, LGMRES, ILU(0), preconditioning theory.
- **Hairer, E., Nørsett, S. P. & Wanner, G.** *Solving Ordinary Differential Equations I & II* (Springer, 2008/1996) — RK45 (Dormand-Prince), DOP853, BDF, Radau IIA, the `select_initial_step` heuristic.
- **Press, W. H. et al.** *Numerical Recipes* (3rd ed., Cambridge, 2007) — Gauss-Kronrod quadrature, Romberg, simulated annealing, hypothesis tests as engineering rather than statistics.
- **Carlson, B. C.** *Numerical computation of real or complex elliptic integrals* (Numer. Algorithms 10, 1995) — the symmetric-duplication recipes used by `elliprc` / `elliprf` / `elliprd` / `elliprg` / `elliprj`.
- **Cuyt, A. et al.** *Handbook of Continued Fractions for Special Functions* (Springer, 2008) — the continued-fraction branches in `hyp2f1` and the gamma family.
- **NIST Digital Library of Mathematical Functions** (DLMF, <https://dlmf.nist.gov/>) — the authoritative cross-check for special-function values, asymptotic series boundaries, and reflection formulas.
- **Vovk, V., Gammerman, A. & Shafer, G.** *Algorithmic Learning in a Random World* (Springer, 2005) — conformal prediction theory, the basis for the `ConformalCalibrator` drift gate.
- **Shokrollahi, A.** *Raptor Codes* (IEEE Trans. Information Theory 52, 2006) — the RaptorQ family used for systematic-encoding sidecars on conformance artifacts.

### Why the name "Franken"?

Because the project bolts well-understood numerical algorithms onto a Rust-native runtime. The result is recognizably SciPy-shaped on the surface, but the body underneath is entirely different. The same naming convention applies across the sibling projects (`FrankenTerm`, `FrankenSQLite`, `FrankenLibc`, `FrankenFS`) and signals the shared engineering doctrine: *clean-room reimplementation in safe Rust, with explicit modes, audit ledgers, and durable artifacts*.

---

## Trade-offs, Conventions, and Async Hosting

Additional reference material: life-of-a-bug from a contributor's seat, algorithm trade-off analysis, the coding conventions the workspace enforces, hosting FrankenSciPy in a long-running service, build-system reality, and a tour of where to read source.

### Life of a Bug, Start to Finish

When a numerical regression surfaces in this project, the path it takes is fixed and visible. Following one would-be regression from notice to landed fix:

1. **Detection.** A `diff_<family>_*` conformance test under `crates/fsci-conformance/tests/` produces a parity diff that exceeds its declared tolerance. The failure is named in `parity_report.json` and the failing case ID is the BLAKE3 fingerprint of the offending input.
2. **Triage.** A beads issue is created (`br create --title "..." --type=bug --priority=2`). The fingerprint goes into the description; the failing test name goes into the `notes` field. If the issue blocks a roadmap item, it is linked with `br dep add`.
3. **Reproduction.** The single failing case is re-run in isolation. Because the harness is deterministic (LabRuntime virtual time, BLAKE3-keyed plan cache, seeded LCGs in `eigsh` and friends), reproduction is bit-for-bit and does not depend on host wall-clock or thread interleaving.
4. **Root cause.** Inspection happens in the kernel; the audit ledger from the failing call usually pinpoints the wrong-action decision or the failing recovery. The fix is required to be a root-cause fix, not a tolerance-loosening; CI gate G9 (tolerance ratchet) enforces this.
5. **Inline note.** The fix lands with an inline `// br-<id>: <one-sentence rationale>` comment at the site of the change. This rule applies to any subtle numerical fix: the `eigsh` LCG seeding, the LGMRES lucky-breakdown increment, and the Mann-Whitney tie correction (described in **Case Studies** above) all have this form.
6. **Regression test.** A new test case is added to the relevant `diff_<family>_*.rs` covering the input that originally failed; the test must fail before the fix and pass after.
7. **Conformance lane.** The full conformance harness reruns (G3 + G3b); if the SciPy oracle has moved, the oracle is re-captured and a new RaptorQ sidecar is generated.
8. **Beads close.** `br close <id> --reason "..."` then `br sync --flush-only`, then the .beads jsonl change is committed alongside the fix. The commit message includes the bead ID, so `git log --grep "br-<id>"` always finds the resolution.

Steps 1, 4, 5, and 6 are the load-bearing ones. Skipping any of them (particularly the inline note in step 5) means the next person who reads the code has no idea why the line is the way it is.

### Algorithm Trade-off Analysis

Several routines have multiple plausible kernels behind them. The choices FrankenSciPy makes are deliberate; here's the reasoning:

#### Why `expm` uses 20-term Taylor with scaling-and-squaring instead of Padé-13

The textbook recommendation (Higham, *Functions of Matrices*) is Padé-13 with scaling-and-squaring. FrankenSciPy uses a 20-term Taylor series instead, with the same scaling-and-squaring outer loop. The trade-offs:

| Property | Padé-13 + s&s | Taylor-20 + s&s |
|---|---|---|
| Backward error bound | Tighter ( ≈ `2^{-53}` at `‖A‖_1 < 5.4` ) | Looser ( ≈ `2^{-50}` at `‖A‖_1 < 0.5` ) |
| Matrix solves per call | 1 (the Padé denominator) | 0 |
| Matrix multiplies per call | ≈ 6 (Padé numerator + denominator) | ≈ 20 (Taylor truncation) |
| Code surface | ~120 lines | ~40 lines |
| Sensitivity to ill-conditioning | depends on solve | none |

The Taylor route is chosen because (a) **it has no internal solve**, so it cannot itself trigger a CASP fallback while you're trying to compute `expm`; (b) on the conditioning regimes where matrix functions are typically called (well-conditioned matrices, mostly), the looser error bound is well within the tolerance contract; (c) the smaller code surface is easier to audit. Future work (tracked in beads) will benchmark a Padé switch for the `‖A‖_1 ≫ 1` regime where Taylor's 20 multiplications become expensive.

#### Why `eigsh` uses a single Lanczos subspace instead of implicit restarts

SciPy's `eigsh` uses ARPACK's Implicitly Restarted Lanczos Method (IRLM). FrankenSciPy's first kernel was deflated power iteration (whose constant-seed failure on path Laplacians is one of the **Case Studies**); it now runs symmetric Lanczos through the shared Krylov/Arnoldi solver with one subspace of `max(2k+1, 20)` vectors, SciPy's own `ncv` default. The trade-off that remains:

| Property | IRLM (ARPACK) | Single-subspace Lanczos (FrankenSciPy) |
|---|---|---|
| Convergence on well-separated extreme eigenpairs | Excellent | Excellent; the subspace resolves them in `O(m)` matvecs |
| Clustered or interior spectra | Restarts until converged | Reports `converged = false` from the Ritz residuals rather than looping |
| Code complexity | Very high (restart, shift selection, deflation bookkeeping) | Moderate; shares the Arnoldi kernel used by `eigs` |
| Performance on `k ≤ 6` | Best | Competitive (the live sparse benchmark uses an 18-vector window at `k = 6`) |
| `#![forbid(unsafe_code)]` compatibility | Easy | Easy |

Implicit restarts are the remaining gap and are not on any bead; the API surface is stable so callers would not see that swap either.

#### Why CG uses no internal restarting

GMRES restarts; CG doesn't. CG's `A`-orthogonal direction sequence enjoys a global guarantee: every direction is `A`-orthogonal to every prior one until round-off accumulates. The conventional defense is restarting; FrankenSciPy instead lets the user choose `max_iter` and reports `converged: false` on overruns. The user can re-call CG with a warm `x0`. This is simpler and matches `scipy.sparse.linalg.cg`'s behavior.

#### Why we don't FFI to BLAS

The single biggest performance win available to a Rust numerical library is to link OpenBLAS or Intel MKL via FFI. FrankenSciPy does not do this, on purpose. The full cost matrix:

| With BLAS FFI | Without (FrankenSciPy today) |
|---|---|
| ~3-10× faster on large dense linalg | Pure-Rust, predictable performance |
| `unsafe` somewhere in the stack | `#![forbid(unsafe_code)]` |
| C/Fortran build dependency | Cargo-only build |
| Vendor-specific tuning needed | Cross-platform identical behavior |
| Difficult to embed in Wasm | Wasm-ready out of the box |
| LAPACK error semantics leak through | Native Rust error types end-to-end |

The doctrine is: never trade the safety/embedding guarantees for raw speed. If a user *needs* OpenBLAS speed on a 10⁴ × 10⁴ matrix, they should use `ndarray-linalg`. If they need a numerical kernel they can drop into a memory-safe service, FrankenSciPy is the right tool.

### Coding Conventions

The workspace enforces a small set of style and structure rules. Every contributor (and every agent) is expected to follow them; CI gates the build on most of them.

#### Source files

- **One module = one responsibility.** No "utils" or "helpers" dumping grounds; if something doesn't belong to the module it's in, it belongs in a sibling module.
- **Public re-exports at the top.** Every crate's `lib.rs` begins with `pub use` lines that name the entire intended public surface. If a symbol isn't re-exported there, it isn't part of the contract.
- **`#[must_use]` is encouraged** on fallible constructors and on result types that could be silently dropped (applied selectively today, not yet a CI-enforced contract).

#### Doc comments

- **Backtick-wrap math notation** including bracket indices (`x[k]`), interval notation (`[a, b]`), and angle-bracketed type names (`Vec<Vec<f64>>`) so rustdoc does not mistake them for HTML or broken intra-doc links.
- **Cite SciPy parity** where applicable: "Matches `scipy.linalg.solve(A, b, assume_a='gen')`" goes in the doc comment of the corresponding Rust function.
- **Reference beads IDs** for subtle behaviors: `// br-<id>: <rationale>` is mandatory on any non-obvious numerical decision.

#### Error handling

- **Result, not panic, for user input.** Constructors of types that carry a numerical invariant (`BetaDist::new(a, b)`) are the documented exception, and the panic message must name the invariant.
- **No `unwrap` in non-test code.** Use `expect` with a message naming the invariant being relied on.
- **Audit on every fail-closed.** The error returned to the caller must also be recorded in the audit ledger; the conformance harness checks this.

#### Tests

- **Inline `#[cfg(test)] mod tests`** alongside the implementation. Cross-crate integration tests live in `crates/fsci-conformance/tests/`.
- **Property tests for invariants.** When a routine has an algebraic invariant (`forward · inverse ≈ identity`, `Σ pdf ≈ cdf`), it gets a `proptest`.
- **Differential tests against SciPy** for every routine that has a SciPy counterpart. The differential test lives in `fsci-conformance` and names its tolerance explicitly.

### Hosting FrankenSciPy in a long-running async service

FrankenSciPy itself is sync; every numerical kernel returns a `Result`
synchronously. The intended integration story is:

#### 1. Run CPU-bound numerical work on a blocking worker

A request handler in your async host wraps each FrankenSciPy call in whatever
"run this synchronous code on a dedicated worker" facility your runtime
provides (`spawn_blocking` in asupersync, the equivalent in another host).
The numerical kernel itself never blocks the async reactor, and you keep all
the structured-concurrency, cancel-correctness, and timeout properties of the
host runtime.

#### 2. Carry a per-request audit ledger

`fsci_runtime::AuditLedger` is the canonical sink for `FailClosed`,
`BoundedRecovery`, `ModeDecision`, and `AlienArtifactDecision` events. The
thread-safe handle is constructed via `AuditLedger::shared()` and the type
alias `SyncSharedAuditLedger = Arc<Mutex<AuditLedger>>` lets you pass the
handle into any audit-emitting call:

```rust
use std::sync::{Arc, Mutex};
use fsci_linalg::{SolveOptions, solve_with_audit};
use fsci_runtime::{AuditLedger, RuntimeMode, SolverPortfolio};

fn solve_with_request_audit(
    a: &[Vec<f64>],
    b: &[f64],
) -> Result<Vec<f64>, fsci_linalg::LinalgError> {
    // Build a per-request audit ledger (shared = Arc<Mutex<AuditLedger>>).
    let ledger = AuditLedger::shared();
    let mut portfolio = SolverPortfolio::new(RuntimeMode::Hardened, 64);

    let result = solve_with_audit(
        a, b,
        SolveOptions::default(),
        &mut portfolio,
        &ledger,
    )?;

    // Drain the ledger and forward to your logging/SIEM/postmortem store.
    let entries = ledger.lock().expect("ledger poisoned").entries().to_vec();
    forward_to_observability(entries);

    Ok(result.x)
}
# fn forward_to_observability<T>(_: T) {}
```

This pattern is what makes FrankenSciPy genuinely auditable in production:
every request that performs a numerical operation can attach its own ledger,
and the events generated within that request are isolated from concurrent
traffic by the host's request-scoping discipline.

#### 3. Share the FFT plan cache across all requests

The FFT plan cache is process-global and concurrency-safe (`fsci-fft` uses
the bounded `CostWeightedLru` admission policy from `fsci_fft::plan`). One-off
oddly-sized FFT requests are bounded against the cache budget, so they cannot
displace plans that hot endpoints have already paid to construct.

### Build System Reality

The workspace is a Cargo workspace that uses `[workspace.dependencies]` for version coordination, but every crate opts in explicitly with `workspace = true`. No crate transitively pulls dependencies it didn't ask for. The practical consequences:

- **Per-crate `cargo build` is fast.** If you only edit `fsci-stats`, only `fsci-stats` and its transitive consumers rebuild.
- **`cargo test --workspace` builds *everything*.** This is by design: the conformance harness must see every crate.
- **Release builds are slow.** The intended release profile (per `AGENTS.md`) is `opt-level = 3`, `lto = true`, `codegen-units = 1`, `strip = true`. The `lto + codegen-units = 1` combination turns the final link into a single-threaded pass. On a high-RAM workstation a clean release build takes ~5 minutes; on a memory-constrained host it can be 15+. (Note: the root `Cargo.toml` does not currently pin this profile; consumers can add it locally or rely on Cargo's defaults until the workspace-level profile is committed.)
- **`cargo doc` is slow.** ~140K lines of source plus all the SciPy-parity prose generates a large doc tree.

#### RCH (Remote Compilation Helper)

When the repo is under active multi-agent development, `cargo build` storms can saturate a workstation. The bundled `rch` tool offloads compilation to a fleet of remote workers transparently:

```bash
# RCH is installed at ~/.local/bin/rch and is wired into Claude Code's PreToolUse
# automatically; manual invocation looks like:
rch exec -- cargo build --release
rch exec -- cargo test --workspace
rch exec -- cargo clippy --workspace --all-targets -- -D warnings

rch doctor                    # Worker health check
rch workers probe --all       # Per-worker reachability
rch status                    # Active and queued builds
```

If RCH or its workers are unavailable, the wrapper fails open and the build runs locally. This means no critical path depends on the remote fleet.

### Reading the Source

When you open a crate's source tree the first time, the layout is intentionally flat. There are no "core" or "internal" submodules to spelunk:

```text
crates/<name>/
├── Cargo.toml
├── benches/                # criterion harnesses (where applicable)
├── src/
│   ├── lib.rs              # public re-exports + top-level module declarations
│   ├── <module>.rs         # one module per major API surface
│   ├── audit.rs            # audit-emission helpers (where applicable)
│   ├── bin/                # standalone binaries (e.g. test scripts, dashboards)
│   └── ...
└── tests/                  # cross-module integration tests for this crate
```

When you want to understand a kernel, follow this order:

1. Start at the crate's `lib.rs`. The `pub use ...` block names the public surface.
2. Jump to the named module. Each module's top doc comment is meant to be the highest-signal explanation of *what* it does.
3. For numerical correctness questions, look for an inline `// br-<id>` comment; it almost certainly contains the answer.
4. For "why this algorithm?" questions, the relevant book or paper from the **Bibliography** is the answer.
5. For "does this match SciPy?" questions, the `fsci-conformance` differential test for that routine is the answer.

The single largest file in the workspace is `crates/fsci-stats/src/lib.rs` at ~125K lines, roughly half of it inline tests. It is intentionally not split: every distribution is defined inline so a reader can `grep -n "pub struct <Name>"` and land on the entire implementation in one place.

`fsci-special` takes the opposite approach: ~72K lines of source split across ~10 modules (`gamma.rs`, `hyper.rs`, `orthopoly.rs`, `elliptic.rs`, `beta.rs`, etc.) with `lib.rs` acting as the re-export header. Use either layout when adding new content; match the existing one in the crate you're touching.

---

## Fuzz Harness, Test Anatomy, and Support

The final reference block: the fuzz harness layout, the anatomy of a conformance test, common SciPy-porting pitfalls, performance characteristics by domain, where to ask questions, and practical patterns for reading the audit ledger.

### The Fuzz Harness

`fuzz/` is a cargo-fuzz workspace (excluded from the main workspace) with **96 fuzz targets** under `fuzz/fuzz_targets/`. Each target is named after the conformance packet it stresses, and each targets one class of input the routine has to be robust against:

```text
fuzz/
├── Cargo.toml
├── fuzz_targets/
│   ├── p2c002_factorizations.rs                  # LU/QR/Cholesky on adversarial dense input
│   ├── p2c002_eigvals_edges.rs                   # eigenvalue routines at the well/ill boundary
│   ├── p2c006_special_hankel_complex.rs          # Hankel functions across the complex plane
│   ├── p2c006_special_hyper_broadcast.rs         # 1F1/2F1 broadcast semantics
│   ├── p2c007_stats_hypothesis_tests.rs          # t-tests, KS, Mann-Whitney on stress inputs
│   ├── p2c007_stats_yulesimon_robustness.rs      # heavy-tail discrete distribution stability
│   ├── p2c009_signal_lfilter_equiv.rs            # SOS vs BA filter application equivalence
│   ├── p2c011_spatial_kdtree.rs                  # KDTree query under degenerate point sets
│   ├── p2c012_cluster_fcluster_bounds.rs         # hierarchical cluster cut at extreme thresholds
│   ├── p2c017_io_wav_read.rs                     # WAV header parser against malformed files
│   └── ...                                        # 88 more
├── corpus/         # accumulated fuzzer inputs that trigger interesting paths (923 files)
├── seeds/          # human-curated seed inputs (569 files)
└── artifacts/      # minimized reproducers from any crash
```

The nightly `fuzz_nightly.yml` GitHub Actions workflow runs the nine `p2c006_special_*` targets for 900 seconds each under ASan (cargo-fuzz's `-Zsanitizer=address`; UBSan is not reachable through `-Zsanitizer` on the pinned toolchain — an earlier `address,undefined` flag was rejected at compile time and kept the workflow red at the build step); the other 89 targets are run by hand with `cargo fuzz run <target>`. Crash reproducers land in `fuzz/artifacts/`; the `fuzz_triage` binary promotes them to regression fixtures and files `fuzz-finding` beads when run by hand, but the nightly workflow does not invoke it.

Each target uses `libfuzzer-sys` and is structured around the same three-phase pattern:

1. **Parse the fuzzer's byte stream** into a typed input (e.g., a `(matrix, rhs)` pair, a polynomial spec, a filter design parameter set).
2. **Apply a "fast reject"** that drops inputs which are uninteresting or already covered by the differential lane.
3. **Run the kernel under test** and assert the documented invariants (no panic, no NaN propagation past the documented boundary, fingerprint determinism, recovery-event symmetry between Strict and Hardened modes).

The fuzz harness has been load-bearing on three of the major bug catches landed since `fsci-sparse` was scaffolded. The LGMRES lucky-breakdown infinite loop (`br-iq1e`), the `adaptive_gk15` NaN-integrand subdivision blow-up (`br-t45u3`), and the FFT plan-cache test-race (`br-lw3rl`) all surfaced first as corpus growth or crash artifacts before being chased through the parity lanes.

### Anatomy of a Conformance Test

A typical differential test file (`crates/fsci-conformance/tests/diff_<family>_<routine>.rs`) follows a fixed shape so an agent can read any one of them and immediately understand the others. Schematically:

```rust,ignore
// 1. Imports
use fsci_conformance::{packet_id, run_differential_test, DifferentialCase};
use fsci_<family>::{<routine>, <Options>};

// 2. Case generator — produces a vector of test inputs paired with
//    SciPy-anchored expected outputs.
fn cases() -> Vec<DifferentialCase> {
    vec![
        // (input, expected, tolerance)
        DifferentialCase::new("happy_path",   /* input */, /* expected */, 1e-10),
        DifferentialCase::new("edge_zero",    /* input */, /* expected */, 1e-10),
        DifferentialCase::new("ill_conditioned", /* input */, /* expected */, 1e-6),
        // ...
    ]
}

// 3. The harness entry point — runs every case under self-check, oracle-backed,
//    and dispatch lanes.
#[test]
fn diff_<family>_<routine>() {
    run_differential_test(packet_id("FSCI-P2C-002"), <routine>, cases())
        .expect("differential test passed");
}
```

The harness writes:
- `parity_report.json` with pass/fail tallies and per-case error magnitudes;
- `parity_report.raptorq.json` (the systematic-encoding sidecar);
- `parity_report.decode_proof.json` (the RaptorQ decode proof);
- one JSON file per case under `diff/<case_id>.json` for any case that exceeded its tolerance.

A failing test is a one-line `cargo test` failure with the case ID; the JSON sidecar gives you the failing input, the SciPy expected output, and the FrankenSciPy actual output side by side.

### Common SciPy-Porting Pitfalls

When migrating a Python codebase to FrankenSciPy, the recurring footguns:

| Pitfall | What goes wrong | The fix |
|---|---|---|
| **NumPy broadcasting expected** | Python silently broadcasts `(3,)` against `(3, 3)`; the Rust API takes explicit shapes | Reshape on the Rust side or build the matrix explicitly with `vec![vec![…]; n]` |
| **In-place mutation expected** | Python's `arr += 1` modifies in place; Rust's `Vec<f64>` requires an explicit loop or `.iter_mut()` | Use `for x in &mut v { *x += 1.0 }` |
| **Tuple-returning functions** | SciPy returns tuples, e.g. `(eigenvalues, eigenvectors)` | FrankenSciPy returns named result structs; read the field names, don't destructure positionally |
| **Implicit `equal_var=True` for t-test** | SciPy defaults to pooled-variance; FrankenSciPy splits into `ttest_ind` (pooled) vs `ttest_ind_welch` (Welch) | Pick the right function (no `equal_var` flag) |
| **`scipy.linalg.solve` accepts `assume_a='gen'/'sym'/...`** | Python keyword | Set the field: `SolveOptions { assume_a: Some(MatrixAssumption::Symmetric), ..Default::default() }` |
| **Numerical defaults differ slightly** | SciPy's `solve_ivp` default `rtol` is `1e-3`; FrankenSciPy follows the same default, but explicit tolerance always wins | Pass tolerances explicitly when porting numerical regression tests |
| **Distributions return tuples for `(loc, scale, shape)` params** | SciPy's `stats.beta.pdf(x, a, b, loc, scale)` | FrankenSciPy constructors take only shape parameters; pass `loc`/`scale` to `loc_scale_pdf` companions or shift/scale your input |
| **Iterative-solver tolerance differs** | SciPy's `cg` defaults to `1e-5`; FrankenSciPy's `IterativeSolveOptions::default().tol` is also `1e-5` but be explicit for production | Always set `tol` explicitly |
| **Random-seed semantics** | SciPy uses NumPy's RNG; FrankenSciPy uses `rand` and explicit seeds | Pass explicit `u64` seeds to `bootstrap_ci`, `SobolSampler::with_digital_shift`, etc. |
| **`scipy.fft` vs `numpy.fft`** | NumPy normalizes by `1/N` on inverse; SciPy normalizes by `1` | `FftOptions::default()` uses the SciPy convention; flip via `FftOptions::norm(...)` |
| **DCT type defaults** | SciPy's `dct` defaults to type II | FrankenSciPy follows the same default; `dct_i`, `dct_iii`, `dct_iv` are reachable directly |

### Performance Characteristics by Domain

A rough sense of where FrankenSciPy is competitive with SciPy and where it isn't, with the qualification that the project's optimization loop is profile-first and the picture changes as new benchmarks land:

| Domain | Competitive against SciPy | Where FrankenSciPy is faster | Where SciPy is faster |
|---|---|---|---|
| **Dense linalg, small (`n < 200`)** | Yes | No call-overhead from Python | Marginal LAPACK SIMD wins at the top end |
| **Dense linalg, large (`n > 1000`)** | Within ~2-3× on uncontested machines | None today | SciPy's OpenBLAS/MKL pathways win on raw throughput |
| **Sparse iterative** | Yes (algorithmically equivalent) | Lower overhead per iteration | Vendor BLAS for the SpMV step on large matrices |
| **FFT, power-of-2** | Yes | No GIL contention | Marginal vendor FFT (FFTW/pocketfft) wins on huge transforms |
| **FFT, non-power-of-2** | Yes (Bluestein vs `scipy.fft.fft` chirp-z) | Comparable | Comparable |
| **ODE solvers** | Yes | Native types, no Python dispatch | None |
| **Optimizers** | Yes | Lower function-evaluation overhead | None |
| **Special functions** | Generally yes | Most scalar evaluations | A few asymptotic regimes still need work |
| **Distributions: pdf/cdf** | Yes | Native | None |
| **Distributions: fit** | Function-of-distribution; closed-form fits are instant, numerical fits use `L-BFGS-B` with bounds | Often faster | SciPy's vectorized loss helps for huge data |
| **Hypothesis tests** | Yes | Native | None |
| **QMC** | Sobol up to 32 dimensions; Halton + Latin Hypercube unrestricted | Halton/LHS at small dims | SciPy's Sobol above 32 dimensions |

The "Within 2-3× on uncontested machines" line for large dense linalg is the honest answer. FrankenSciPy is not trying to beat OpenBLAS; the goal is a credible memory-safe alternative that closes the gap routine by routine through the profile-and-prove discipline. Where the gap matters for a specific workload, file a beads issue.

### Reading the Audit Ledger: Practical Patterns

The ledger is just a `Vec<AuditEvent>` inside an `Arc<Mutex<AuditLedger>>`. Practical recipes:

#### Pattern 1: Drain into a JSON line stream

```rust,ignore
let entries = ledger.lock().expect("poisoned").entries().to_vec();
for event in &entries {
    let line = serde_json::to_string(event).expect("serialize");
    writeln!(out, "{line}").expect("write");
}
```

`AuditEvent` implements `Serialize` / `Deserialize` with `#[serde(tag = "kind", rename_all = "snake_case")]`, so each line is a self-describing record. Pipe `out` into your observability system.

#### Pattern 2: Filter by fingerprint to scope to a single request

```rust,ignore
let req_fp = blake3::hash(req_body.as_bytes()).to_hex().to_string();
let relevant: Vec<_> = ledger
    .lock().expect("poisoned")
    .entries()
    .iter()
    .filter(|e| e.input_fingerprint == req_fp)
    .cloned()
    .collect();
```

Since the fingerprint is BLAKE3 of the input bytes, an external request handler can compute it the same way and pull exactly the audit events that belong to that request.

#### Pattern 3: Count fail-closed events as a rate metric

```rust,ignore
let entries = ledger.lock().expect("poisoned").entries().to_vec();
let fail_closed_count = entries
    .iter()
    .filter(|e| matches!(e.action, AuditAction::FailClosed { .. }))
    .count();
emit_metric("frankenscipy.fail_closed.count", fail_closed_count);
```

This is the recommended sensor for "are my callers passing malformed input?". A spiking `FailClosed` rate is the signal that something upstream of FrankenSciPy is broken.

#### Pattern 4: Use `BoundedRecovery` as a regression sensor

In Hardened mode, every `BoundedRecovery` event means the routine had to repair the input. If the rate climbs after a deploy, *something upstream changed*; usually the input distribution moved, exposing a regime the previous deploy never hit. The recovery itself is safe by design, but the rate is a leading indicator.

### Where to ask, where to file

| Need | Where |
|---|---|
| **Bug report** with a minimal reproducer | GitHub Issues on `Dicklesworthstone/frankenscipy` |
| **Numerical regression** against SciPy | Same. Include the SciPy version, the failing input, the expected and actual outputs, and the tolerance you were targeting |
| **Feature request** for an unported SciPy routine | GitHub Issues; tag with the SciPy module path (e.g. `scipy.signal.something`) |
| **Discussion of API shape** before submitting a PR | GitHub Issues. See *About Contributions* (below) for why PRs are not merged directly |
| **Security disclosure** for a numerical-stability attack | GitHub Issues, marked "security". The project takes adversarial input seriously |

The active beads tracker (`.beads/issues.jsonl`) is the internal-state-of-truth for prioritization; GitHub Issues are the *intake* channel that those agents triage from.

---

## Installation

FrankenSciPy is pre-1.0 and not yet published to crates.io. Use it as a Git dependency or as a workspace clone.

### Prerequisites

- **Rust 2024 edition on a nightly toolchain.** A `rust-toolchain.toml` is committed at the root and pins the channel.
- **A working `cargo`.** Nothing else is required for the pure-Rust lanes.
- *(Optional, for the SciPy-oracle conformance lane)* a Python 3.11+ interpreter with `scipy` and `numpy` installed.

### Per-crate, as a Git dependency

```toml
# Cargo.toml of your project
[dependencies]
fsci-linalg = { git = "https://github.com/Dicklesworthstone/frankenscipy" }
fsci-stats  = { git = "https://github.com/Dicklesworthstone/frankenscipy" }
```

Pinning to a commit is recommended while there are no tagged releases:

```toml
[dependencies]
fsci-linalg = { git = "https://github.com/Dicklesworthstone/frankenscipy", rev = "<commit-sha>" }
```

### From source (workspace clone)

```bash
git clone https://github.com/Dicklesworthstone/frankenscipy
cd frankenscipy
cargo build --workspace --release
cargo test  --workspace
```

The intended release profile (per `AGENTS.md`) is performance-tuned. Add it
to a consuming workspace's `Cargo.toml` (or to your own fork's root) if you
need it:

```toml
[profile.release]
opt-level    = 3   # maximum performance
lto          = true
codegen-units = 1
strip         = true
```

### Optional: enable the SciPy oracle lane

```bash
python -m venv .venv && source .venv/bin/activate
pip install "scipy>=1.13" "numpy>=2.0"

# Then run the full oracle-backed conformance lane:
cargo test -p fsci-conformance -- --nocapture
# Then re-run the live-SciPy lane explicitly:
cargo run -p fsci-conformance --bin live_oracle_capture -- --output target/oracle.json
```

If SciPy is missing the strict requirement lanes return `PythonSciPyMissing` rather than silently passing; the optional ones write `oracle_capture.error.txt` and continue.

---

## Quick Start

```bash
# 1. Clone and build
git clone https://github.com/Dicklesworthstone/frankenscipy
cd frankenscipy
cargo build --release --workspace

# 2. Run the pure-Rust test surface (no Python required)
cargo test --workspace

# 3. Run a single domain crate's tests
cargo test -p fsci-stats --lib

# 4. Run the conformance harness (will skip oracle lanes if SciPy missing)
cargo test -p fsci-conformance -- --nocapture

# 5. Run benchmarks for a specific crate
cargo bench -p fsci-linalg
cargo bench -p fsci-fft

# 6. Launch the interactive conformance dashboard
cargo run -p fsci-conformance --bin conformance_dashboard -- \
    --artifact-root crates/fsci-conformance/fixtures/artifacts
```

---

## Crate Reference

Detailed feature lists for each crate live in [`docs/planning/FEATURE_PARITY.md`](docs/planning/FEATURE_PARITY.md) and the per-crate `src/lib.rs` doc comments. The table below summarizes status against the V1 scope contract.

The coverage column is the 2026-07-27 name census against scipy 1.17.1
([`PARITY-COVERAGE.md`](docs/planning/PARITY-COVERAGE.md)): a same-named
public symbol exists. It says nothing about signatures or numerical
agreement; that is what the `diff_*` conformance lanes measure, and the
`diff_*` file count per family is given alongside.

| Crate | Status | Same-named coverage | `diff_*` files | Notes |
|---|---|---|---|---|
| `fsci-linalg` | `parity_green` | 96.9% (95/98) | 40 | Dense + structured; CASP audit surface on the solve family |
| `fsci-integrate` | `parity_green` | 72.7% (24/33) | 19 | IVP (incl. genuine Radau IIA) / BVP / quadrature; the misses are solver classes exposed as `SolverKind` variants |
| `fsci-interpolate` | `parity_green` | 94.6% (53/56) | 14 | Splines, scattered data, RBF with SciPy's polynomial tail |
| `fsci-signal` | `parity_green` | 99.4% (156/157) | 81 | IIR/FIR design + SOS + spectral + wavelets + audio features |
| `fsci-stats` | `parity_green` | 78.9% (239/303) | 256 | Most misses are SciPy's lowercase distribution instances vs. Rust types; the real residual is `rv_continuous`/`CensoredData`/`Covariance` |
| `fsci-spatial` | `parity_green` | 83.3% (15/18) | 17 | KDTree, pdist/cdist, ConvexHull/Delaunay/Voronoi; plotting helpers excluded by policy |
| `fsci-special` | `parity_green` | 98.6% (353/358) | 117 | Gamma, Bessel, Carlson, hypergeometric, orthogonal polynomials |
| `fsci-fft` | `parity_green` | 90.2% (37/41) | 23 | 1-D and n-D transforms + DCT/DST + Hilbert + FHT; the misses are backend-registration hooks |
| `fsci-sparse` | `parity_green` | 96.2% (51/53) | 46 | Formats + direct + iterative + eigensolvers + graph |
| `fsci-opt` | `parity_gap` | 84.5% (60/71) | 47 | `optimize.direct` and the quasi-Newton update-strategy objects are the real gaps |
| `fsci-cluster` | `parity_green` | n/a (SciPy exports submodules only) | 13 | KMeans + DBSCAN + hierarchical + indices |
| `fsci-ndimage` | `parity_green` | 100% (75/75) | 34 | Filters, morphology, measurement, geometric transforms |
| `fsci-io` | `parity_green` | 100% (14/14) | 0 | MATLAB v4/v5, MM, WAV, NetCDF, IDL, Fortran, Harwell-Boeing; covered by unit tests and the io oracle packet only |
| `fsci-constants` | `parity_green` | 87.5% (7/8) | 5 | CODATA 2018 + SI + math constants |
| `fsci-odr` | `parity_green` | 100% (10/10) | 1 | Explicit + implicit + weighted ODR |
| `fsci-datasets` | `parity_green` | 100% (5/5) | 0 | Embedded sample fixtures |
| `fsci-runtime` | `parity_green` | n/a (FrankenSciPy-native) | 8 | CASP engine + audit ledger |
| `fsci-arrayapi` | `aspirational` | n/a | 5 | Contract-first backend; no other crate depends on it yet |
| `fsci-conformance` | `parity_green` | n/a (harness) | — | 793 integration test files (731 `diff_*`), 18 packets, 15 oracles |

---

## Configuration

### Workspace structure

```text
frankenscipy/
├── Cargo.toml                   # Workspace root, 19 members
├── rust-toolchain.toml          # Pinned to nightly
├── crates/
│   ├── fsci-arrayapi/           # Array API backend
│   ├── fsci-cluster/            # KMeans, DBSCAN, hierarchical
│   ├── fsci-conformance/        # Differential harness + dashboard binary
│   ├── fsci-constants/          # CODATA + SI + math
│   ├── fsci-datasets/           # Embedded sample data
│   ├── fsci-fft/                # FFT / DCT / DST / Hilbert / FHT
│   ├── fsci-integrate/          # ODE + quadrature
│   ├── fsci-interpolate/        # Splines + scattered data
│   ├── fsci-io/                 # MATLAB / Matrix Market / WAV / NetCDF / IDL
│   ├── fsci-linalg/             # Dense + structured linear algebra
│   ├── fsci-ndimage/            # Filters + morphology + transforms
│   ├── fsci-odr/                # Orthogonal distance regression
│   ├── fsci-opt/                # Optimizers + root finders + LP/MILP
│   ├── fsci-runtime/            # CASP engine
│   ├── fsci-signal/             # Filter design + spectral + wavelets
│   ├── fsci-sparse/             # Sparse formats + iterative + graph
│   ├── fsci-spatial/            # KDTree + distances + hulls + assignment
│   ├── fsci-special/            # Gamma + Bessel + Carlson + orthopoly + …
│   └── fsci-stats/              # Distributions + tests + regression + QMC
├── docs/                        # ARTIFACT_TOPOLOGY.md, ORACLE_WORKFLOW.md, schemas/, planning/
├── legacy_scipy_code/scipy/     # SciPy oracle source-of-truth (cloned)
├── reference/                   # Reference materials
├── fuzz/                        # Fuzz targets (excluded from main workspace)
├── .beads/                      # Issue tracker (br / bv)
└── .github/workflows/           # CI gates G1–G8, nightly fuzz
```

### Key dependencies

| Crate | Purpose |
|---|---|
| [`asupersync`](https://github.com/Dicklesworthstone/asupersync) | Structured async runtime (regions, channels, sync primitives, LabRuntime) |
| [`ftui`](https://github.com/Dicklesworthstone/ftui) | Terminal UI rendering for the conformance dashboard |
| `blake3` | Cryptographic hashing for artifact integrity and audit fingerprints |
| `serde` + `serde_json` | Serialization for artifacts and audit events |
| `thiserror` | Ergonomic error derivation |
| `proptest` | Property-based testing |
| `criterion` | Benchmarking (`fsci-linalg`, `fsci-sparse`, `fsci-fft`, `fsci-opt`, `fsci-integrate`, `fsci-special`, `fsci-runtime`, `fsci-arrayapi`) |
| `rand` | RNGs for stochastic tests |
| `toml` | Quality-gate config |

---

## Performance and Quality Gates

The CI pipeline (`.github/workflows/ci.yml`, restructured 2026-09-03 under
`frankenscipy-liel6`) runs the gates below on every push to `main`. Every job
restores one shared dependency cache and pins `nightly-2026-07-20` from
`rust-toolchain.toml`. The pinned live-SciPy incumbent (scipy 1.17.1 with
numpy 2.4.3, the pair recorded in `fsci_runtime::scipy_incumbent`) is
installed for the oracle lanes and **required**: a test that cannot reach it
fails instead of skipping, and a control job proves that by running one lane
without SciPy and asserting it fails.

| Gate | Job name | Check |
|---|---|---|
| G1 | `fmt + clippy` | `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings` |
| G2 | `unit + property tests (<crate>)` | One job per crate: `cargo test -p <crate>` for the 18 domain crates plus the conformance library |
| G3 | `live-SciPy diff (<shard>)` | The 731 `diff_*` files in 14 family shards, each binary built and run with `FSCI_REQUIRE_SCIPY_ORACLE=1` against the pinned SciPy |
| G3 control | `no SciPy must fail` | Runs one oracle-required lane with an interpreter that cannot import SciPy; the job passes only if that lane fails |
| G3b | `live SciPy oracle capture` | `live_oracle_capture` in required-oracle mode against the pinned versions; uploads the capture, its RaptorQ sidecar and decode proof |
| G3c | `goldens + metamorphic + fallback capture` | Golden journeys, the metamorphic suites, the oracle/live/fuzz regression files, and `live_oracle_capture --allow-missing-oracle` |
| G4 | `adversarial smoke` | The conformance `smoke` suite: NaN/Inf propagation and malformed inputs |
| G5 | `E2E scenarios` | Every `e2e_*` integration test |
| G6 | `perf smoke + baseline validation` | Every `perf_*` integration test, `benchmark_gate --check-spec` against the spec §17 budgets, and `raptorq_sidecar --verify` on the published baselines. There is no criterion regression compare in CI: hosted runners are not a stable timing host |
| G7 | `schema + evidence packs` | `schema_validation` plus every `evidence_p2c*` pack |
| G8 | `RaptorQ proofs` | `raptorq_proofs`: decode-proof verification for the committed sidecars |
| G9 | `tolerance-policy ratchet` | `tolerance_lint --max-violations 361` over the `FSCI-P2C-*.json` fixtures (threshold may only fall) plus the harness ratchet: every `tests/diff_*.rs` tolerance contract (`const *_TOL: f64` and bare `<= <literal>` comparisons) is checked against the committed `fixtures/tolerance_baseline.json` — a loosened value, an unanchored contract, or a stale baseline entry fails the build, and `--write-baseline` refuses to adopt loosened values (frankenscipy-6a5s9) |

A separate `fuzz_nightly.yml` workflow runs nine `fsci-special` fuzz targets
under ASan on a nightly schedule (an earlier `address,undefined` sanitizer
flag was rejected by rustc at compile time and kept the workflow red before
any target fuzzed; fixed 2026-09-04).

Before 2026-09-03 the workflow had never completed successfully, so no gate
claim in this README was enforced by CI; the first fully green run is roadmap
item 3 above and will be cited here when it exists.

### Benchmarks

Each compute-heavy crate ships a `criterion` benchmark harness. Run a single crate's bench:

```bash
cargo bench -p fsci-linalg          # decompositions and solvers
cargo bench -p fsci-fft              # forward and inverse transforms
cargo bench -p fsci-sparse           # iterative solvers and graph algos
cargo bench -p fsci-opt              # minimizer convergence
cargo bench -p fsci-integrate        # IVP and quadrature
cargo bench -p fsci-special          # gamma / Bessel / hypergeometric
cargo bench -p fsci-runtime          # CASP selection overhead
```

The mandatory optimization loop is:

1. **Baseline.** Record `p50` / `p95` / `p99` and memory.
2. **Profile.** Identify real hotspots, not guessed ones.
3. **Implement one optimization lever.**
4. **Prove behavior unchanged** via conformance + invariant checks.
5. **Re-baseline and emit delta artifact.**

---

## Troubleshooting

### `error: the option Z is only accepted on the nightly compiler`

You are on stable. FrankenSciPy is pinned to nightly by `rust-toolchain.toml`. Run:

```bash
rustup toolchain install nightly
rustup component add rustfmt clippy --toolchain nightly
```

Cargo will then read the toolchain file automatically when invoked inside the repo.

### `PythonSciPyMissing` in conformance output

The strict oracle lane requires SciPy on `PATH`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install "scipy>=1.13" "numpy>=2.0"
cargo test -p fsci-conformance -- --nocapture
# Then re-run the live-SciPy lane explicitly:
cargo run -p fsci-conformance --bin live_oracle_capture -- --output target/oracle.json
```

The pure-Rust self-check lane never needs SciPy and runs unconditionally.

### A test hangs on `cargo test --workspace`

If a single test reports `has been running for over 60 seconds`, that named test usually pinpoints a deadlock. The three known-and-fixed structural cases are:

1. `fsci-integrate adaptive_gk15` looping on non-finite integrand values.
2. `fsci-sparse lgmres_inner` spinning on an identity-like operator at `k=0` lucky breakdown.
3. `fsci-fft` shared plan cache racing between `plan.rs` and `transforms.rs` test modules.

All three are now guarded, but if you hit a new case file a beads issue and check whether the offending kernel needs a NaN/Inf short-circuit or a test-lock around shared mutable state.

### Tokio refuses to resolve / `hyper` shows up in `cargo tree`

Something pulled in a tokio-flavored dependency by accident. Run:

```bash
cargo tree -i tokio
```

…and remove the offending crate. The FrankenSciPy workspace forbids `tokio`, `hyper`, `reqwest`, `axum`, `tower` (tokio adapter), `async-std`, `smol`, or anything that transitively depends on them. Use `asupersync` equivalents instead.

### Builds are slow / OOM during `cargo build`

`fsci-stats` (~126K source lines including inline tests) and `fsci-special` (~72K) are the largest compilation units. Building them in parallel can exhaust RAM on smaller hosts. Use:

```bash
CARGO_BUILD_JOBS=2 cargo build --workspace --release
```

The release profile uses `codegen-units = 1` and `lto = true`; debug builds are much cheaper if you only need to iterate.

### Conformance writes are mismatched after a parallel test run

The `fsci-conformance` writer for `parity_report.{json,raptorq.json,decode_proof.json}` is serialized via a process-global mutex. If you see mismatched sidecars, you are likely running an older binary that predates [frankenscipy-prngc](https://github.com/Dicklesworthstone/frankenscipy/commits/main). Rebuild.

---

## Limitations

FrankenSciPy is pre-1.0. The following are intentional and tracked:

- **No tagged releases yet.** The workspace version is `0.1.0`; there are no semver guarantees between commits.
- **Name coverage is not behavioural parity.** 1,194 of 1,300 `scipy.*` callables have a same-named public equivalent (91.8%; [`docs/planning/PARITY-COVERAGE.md`](docs/planning/PARITY-COVERAGE.md) explains what that census does and does not check). The genuine remaining gaps are `optimize.direct`, the quasi-Newton update-strategy objects, and the `stats` machinery around `rv_continuous` / `CensoredData` / `Covariance`. Separately, the conformance-coverage audit (`scripts/conformance_coverage_audit.py`, bead `frankenscipy-ivxx6`) went from 201 unreferenced SciPy-named entry points on 2026-08-24 to zero on 2026-08-30, but it checks that a name is mentioned somewhere in the corpus, not that a differential comparison runs for it.
- **No GPU or distributed backends.** All kernels are single-process CPU.
- **No FFI to BLAS / LAPACK.** All linear algebra is implemented in safe Rust; we lose hand-tuned-vendor-kernel performance for the largest matrices in exchange for memory safety and embeddability. Profile-first optimization closes this gap routine by routine.
- **Heavy-tail distribution moments return `NaN`.** For families like `Alpha`, `Cauchy` (mean), `HalfCauchy` (mean/var), and some `Pareto` parameter regimes, the relevant moment integral diverges. We return `NaN` and document it rather than silently returning truncated finite numbers.
- **Open numerical defects on the backlog** (filed, not yet fixed) as of 2026-09-03:
  - `frankenscipy-5lz5e` — `goodness_of_fit` fits the normal with `ddof=1`, disagreeing with SciPy's own `norm.fit`.
  - `frankenscipy-2sjwo` — the SVD bounding recipe uses `f64::EPSILON` where the underlying decomposition uses a 5× looser epsilon, shifting results by an ulp.
  - `frankenscipy-jyfke` — `sos2zpk` trims trailing coefficients at a 1e-15 tolerance where SciPy trims exact zeros only.
  - `frankenscipy-drb0i` — the CZT family rejects degenerate `w`/`a` (stricter than SciPy) while `czt_points` matches SciPy.
  - `frankenscipy-0xy3l`, `frankenscipy-6d400` — the `fsci-linalg` bit-identity suite and one `fsci-fft` plan-cache test are load-flaky under the full parallel test run.

Workstreams are tracked in [`.beads/issues.jsonl`](.beads/issues.jsonl) and surfaced through `br ready` and `bv --robot-triage`.

---

## FAQ

**Q. Why not just call SciPy from Rust via PyO3?**
A. Because then you still have the GIL, the Python object model, the SciPy install footprint, the Python build dependency chain, and zero ability to reason about memory safety or runtime algorithm selection. The whole point is to remove Python from the hot path.

**Q. Why not use `ndarray` + `nalgebra` + `argmin` + `linfa` + a hand-rolled wrapper?**
A. You can. That stack gives you fast linear algebra and a couple of solvers. It does not give you SciPy parity, a conformance harness against the real SciPy, an audited runtime algorithm selector, a distribution moment surface for 100+ distributions, RaptorQ-backed artifact durability, or 767 integration tests covering the same surface. FrankenSciPy is the integration of all of that into one Cargo workspace.

**Q. Why no tokio?**
A. tokio is a fine async runtime, but it brings a heavyweight ecosystem (`hyper`, `reqwest`, `axum`, `tower`, lots of transitive features) and a runtime model that does not give us cancel-correctness or virtual-time testing. asupersync gives us structured concurrency, two-phase send/receive, cancel-aware sync primitives, and a deterministic `LabRuntime` we use in conformance tests.

**Q. What is CASP, in one sentence?**
A. A runtime algorithm selector that picks between concrete solvers by minimizing expected loss over a calibrated decision matrix, conditioned on evidence about your specific problem instance (conditioning, structure, sparsity, prior backward error), and emits an audit ledger entry every time it does so.

**Q. Is this faster than SciPy?**
A. For individual kernels, mileage varies by problem size, conditioning, and structure. The honest answer: this is a profile-first, proof-backed project. Benchmarks live in each crate's `benches/` directory under `criterion`, and the optimization loop requires a delta artifact for every change. Performance work focuses on tail latency (`p95` / `p99`), memory budget, and convergence cost, and is gated against the conformance harness so a speedup can never weaken a tolerance contract.

**Q. Can I use this from Python?**
A. Not directly today. There is no PyO3 layer. If there ever is one, it will live in a sibling crate and reuse the same audit and decision surfaces. The current target audience is Rust applications, embedded numerical work, and agentic systems that need to call numerical kernels without dragging in Python.

**Q. What does `#![forbid(unsafe_code)]` workspace-wide cost me?**
A. The freedom to write `unsafe { std::mem::transmute(…) }`. In practice, idiomatic Rust covers the entire surface FrankenSciPy targets; no hot path has yet needed unsafe. If one ever does, the bar is "isolated behind an audited interface with property tests and a recorded threat-model note."

**Q. What does "strict vs hardened mode" mean operationally?**
A. The `SolverPortfolio` is constructed in one of two modes. **Strict** matches SciPy observable behavior on the V1 scope and refuses to repair malformed input (fail-closed). **Hardened** preserves the API contract but performs bounded defensive recovery (e.g., projecting near-singular inputs to the nearest well-conditioned form, with the recovery recorded in the audit ledger). Pick strict for migration parity testing; pick hardened for production resilience.

**Q. Where is the SciPy oracle?**
A. The SciPy source tree is cloned at `/dp/frankenscipy/legacy_scipy_code/scipy` (upstream: <https://github.com/scipy/scipy>). The Python oracle scripts under `crates/fsci-conformance/python_oracle/` import `scipy.*` directly to capture reference outputs at the documented test cases.

**Q. How do I file a bug?**
A. GitHub issues are open. Bug reports, especially numerical-regression reports with a minimal reproducer and the relevant tolerance, are the most useful contribution. See *About Contributions* below for context on PRs.

---

## Documentation Map

| Document | Purpose |
|---|---|
| [`README.md`](README.md) | You are here |
| [`CHANGELOG.md`](CHANGELOG.md) | Landed capabilities, by domain, with linked commits |
| [`AGENTS.md`](AGENTS.md) | Guidelines and conventions for AI agents working in the repo |
| [`docs/planning/COMPREHENSIVE_SPEC_FOR_FRANKENSCIPY_V1.md`](docs/planning/COMPREHENSIVE_SPEC_FOR_FRANKENSCIPY_V1.md) | Prime directive, product thesis, V1 scope, compatibility/security model |
| [`docs/planning/FEATURE_PARITY.md`](docs/planning/FEATURE_PARITY.md) | Per-module SciPy parity assessment |
| [`docs/planning/PROPOSED_ARCHITECTURE.md`](docs/planning/PROPOSED_ARCHITECTURE.md) | Crate map, runtime plan, mode model, performance contract |
| [`docs/planning/PLAN_TO_PORT_SCIPY_TO_RUST.md`](docs/planning/PLAN_TO_PORT_SCIPY_TO_RUST.md) | Porting strategy and prioritization |
| [`docs/planning/EXISTING_SCIPY_STRUCTURE.md`](docs/planning/EXISTING_SCIPY_STRUCTURE.md) | Reference catalog of the SciPy public surface |
| [`docs/planning/EXHAUSTIVE_LEGACY_ANALYSIS.md`](docs/planning/EXHAUSTIVE_LEGACY_ANALYSIS.md) | Deep audit of the legacy SciPy code paths the conformance harness targets |
| [`docs/ARTIFACT_TOPOLOGY.md`](docs/) | Locked artifact directory schema |
| [`docs/ORACLE_WORKFLOW.md`](docs/) | Full Python oracle capture → regen → provenance → CI lane workflow |
| [`docs/schemas/`](docs/) | Governance-gated JSON schemas (`behavior_ledger`, `contract_table`, `threat_matrix`) |
| [`docs/planning/SPEC_CROSSWALK_FRANKENSQLITE_TO_FRANKENSCIPY.md`](docs/planning/SPEC_CROSSWALK_FRANKENSQLITE_TO_FRANKENSCIPY.md) | Cross-project doctrine alignment with the FrankenSQLite project |

---

## About Contributions

*About Contributions:* Please don't take this the wrong way, but I do not accept outside contributions for any of my projects. I simply don't have the mental bandwidth to review anything, and it's my name on the thing, so I'm responsible for any problems it causes; thus, the risk-reward is highly asymmetric from my perspective. I'd also have to worry about other "stakeholders," which seems unwise for tools I mostly make for myself for free. Feel free to submit issues, and even PRs if you want to illustrate a proposed fix, but know I won't merge them directly. Instead, I'll have Claude or Codex review submissions via `gh` and independently decide whether and how to address them. Bug reports in particular are welcome. Sorry if this offends, but I want to avoid wasted time and hurt feelings. I understand this isn't in sync with the prevailing open-source ethos that seeks community contributions, but it's the only way I can move at this velocity and keep my sanity.

---

## License

MIT with an OpenAI/Anthropic Rider. See [`LICENSE`](LICENSE) for the full text.

#![forbid(unsafe_code)]

//! Boundary value problems: `scipy.integrate.solve_bvp`.
//!
//! [`solve_bvp`] solves `y' = f(x, y, p) + S·y/(x − a)` on `[a, b]` with
//! `bc(y(a), y(b), p) = 0` by SciPy's algorithm (Kierzenka & Shampine 2001, transcribed in
//! `collocation.rs`): a C¹ cubic spline collocated at the mesh nodes and interval midpoints
//! (4th-order Lobatto IIIA), a damped Newton method on the sparse collocation Jacobian
//! (factorized by `fsci_sparse::splu`), and mesh refinement until the RMS relative residual on
//! every interval is below `tol`. Unknown parameters `p` are solved for alongside `y`; the
//! singular term `S` handles `y' = S·y/x + f` problems on `[0, b]`.
//!
//! This replaced a single-shooting solver, which fails on exactly the problems collocation is
//! for: `y'' = 100·y` on `[0, 10]` (a boundary layer the forward IVP amplifies by e¹⁰⁰) and
//! Troesch's problem (frankenscipy-1ksfv.7).

use crate::collocation::{self, LinearSolver, Problem, Singular, Spline};
use fsci_sparse::{
    CooMatrix, FormatConvertible, LuOptions, Shape2D, SparseLuFactorization, splu, splu_solve,
};
use nalgebra::DMatrix;

/// `fun_jac(x, y, p) -> (∂f/∂y, ∂f/∂p)`: n×n and n×k, one row per component of `f`.
pub type BvpFunJac<'a> = &'a (dyn Fn(f64, &[f64], &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) + Sync);
/// `bc_jac(ya, yb, p) -> (∂bc/∂ya, ∂bc/∂yb, ∂bc/∂p)`: (n+k)×n, (n+k)×n and (n+k)×k.
pub type BvpBcJac<'a> =
    &'a (dyn Fn(&[f64], &[f64], &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) + Sync);

/// SciPy's `solve_bvp` keyword arguments.
#[derive(Clone, Copy)]
pub struct BvpOptions<'a> {
    /// Tolerance on the RMS relative residual per interval (default 1e-3; floored at 100·ε).
    pub tol: f64,
    /// Maximum number of mesh nodes (default 1000).
    pub max_nodes: usize,
    /// Tolerance on the boundary-condition residuals (default: `tol`).
    pub bc_tol: Option<f64>,
    /// The singular term `S` (n×n) of `y' = S·y/(x − a) + f(x, y, p)`.
    pub singular_term: Option<&'a [Vec<f64>]>,
    /// Analytic Jacobian of `f`; forward differences when `None`.
    pub fun_jac: Option<BvpFunJac<'a>>,
    /// Analytic Jacobian of `bc`; forward differences when `None`.
    pub bc_jac: Option<BvpBcJac<'a>>,
}

impl Default for BvpOptions<'_> {
    fn default() -> Self {
        Self {
            tol: 1e-3,
            max_nodes: 1000,
            bc_tol: None,
            singular_term: None,
            fun_jac: None,
            bc_jac: None,
        }
    }
}

impl std::fmt::Debug for BvpOptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BvpOptions")
            .field("tol", &self.tol)
            .field("max_nodes", &self.max_nodes)
            .field("bc_tol", &self.bc_tol)
            .field("singular_term", &self.singular_term)
            .field("fun_jac", &self.fun_jac.map(|_| "<function>"))
            .field("bc_jac", &self.bc_jac.map(|_| "<function>"))
            .finish()
    }
}

/// SciPy's `BVPResult`.
#[derive(Debug, Clone)]
pub struct BvpResult {
    /// Final mesh.
    pub x: Vec<f64>,
    /// Solution at the mesh nodes, `y[i][j]` = component `i` at node `j` (SciPy's `(n, m)`).
    pub y: Vec<Vec<f64>>,
    /// `f(x, y, p)` at the mesh nodes, same layout.
    pub yp: Vec<Vec<f64>>,
    /// Found unknown parameters (empty when there are none).
    pub p: Vec<f64>,
    /// RMS relative residual on each mesh interval.
    pub rms_residuals: Vec<f64>,
    /// Number of mesh refinement iterations.
    pub niter: usize,
    /// 0 converged, 1 `max_nodes` exceeded, 2 singular Jacobian, 3 `bc_tol` not met.
    pub status: usize,
    pub message: String,
    pub success: bool,
    spline: Spline,
}

impl BvpResult {
    /// SciPy's `sol(x)`: the C¹ cubic spline solution at `x` (extrapolated outside `[a, b]`).
    #[must_use]
    pub fn sol(&self, x: f64) -> Vec<f64> {
        self.spline.eval(x, 0)
    }

    /// `sol(x, 1)`: the derivative of the spline solution at `x`.
    #[must_use]
    pub fn sol_derivative(&self, x: f64) -> Vec<f64> {
        self.spline.eval(x, 1)
    }
}

/// Invalid input, as SciPy raises `ValueError`.
#[derive(Debug, Clone, PartialEq)]
pub enum BvpError {
    InvalidArgument(String),
}

impl std::fmt::Display for BvpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
        }
    }
}

impl std::error::Error for BvpError {}

struct SparseLu(SparseLuFactorization);

impl LinearSolver for SparseLu {
    fn solve(&self, b: &[f64]) -> Option<Vec<f64>> {
        splu_solve(&self.0, b).ok()
    }
}

/// The collocation Jacobian through `fsci_sparse::splu`, as SciPy uses `scipy.sparse.linalg.splu`.
fn factorize_sparse(
    size: usize,
    triplets: &[(usize, usize, f64)],
) -> Option<Box<dyn LinearSolver>> {
    let rows = triplets.iter().map(|t| t.0).collect();
    let cols = triplets.iter().map(|t| t.1).collect();
    let data = triplets.iter().map(|t| t.2).collect();
    let coo = CooMatrix::from_triplets(Shape2D::new(size, size), data, rows, cols, true).ok()?;
    let lu = splu(&coo.to_csc().ok()?, LuOptions::default()).ok()?;
    Some(Box::new(SparseLu(lu)))
}

/// NumPy's `pinv`: singular values below `max(rows, cols)·ε·σ_max` are dropped.
fn pinv(a: &DMatrix<f64>) -> Result<DMatrix<f64>, BvpError> {
    let n = a.nrows().max(a.ncols());
    let svd = nalgebra::SVD::try_new(a.clone(), true, true, f64::EPSILON * 5.0, 30 * n.max(10))
        .ok_or_else(|| BvpError::InvalidArgument("SVD of `S` did not converge".to_string()))?;
    let (Some(u), Some(v_t)) = (svd.u, svd.v_t) else {
        return Err(BvpError::InvalidArgument(
            "SVD of `S` did not converge".to_string(),
        ));
    };
    let s_max = svd.singular_values.iter().fold(0.0_f64, |m, &s| m.max(s));
    let cutoff = n as f64 * f64::EPSILON * s_max;
    let mut s_inv = DMatrix::zeros(v_t.nrows(), u.ncols());
    for (i, &s) in svd.singular_values.iter().enumerate() {
        if s > cutoff {
            s_inv[(i, i)] = 1.0 / s;
        }
    }
    Ok(v_t.transpose() * s_inv * u.transpose())
}

fn row_major(m: &DMatrix<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(m.nrows() * m.ncols());
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            out.push(m[(i, j)]);
        }
    }
    out
}

fn flatten_rows(rows: &[Vec<f64>], ncols: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows.len() * ncols);
    for r in rows {
        out.extend(r.iter().take(ncols));
        out.extend(std::iter::repeat_n(f64::NAN, ncols.saturating_sub(r.len())));
    }
    out
}

/// Solve `y' = f(x, y, p)` (plus `S·y/(x − a)` when `options.singular_term` is set) on the
/// mesh `x` with `bc(y(a), y(b), p) = 0` (n + k residuals): `scipy.integrate.solve_bvp(fun, bc,
/// x, y, p, S, fun_jac, bc_jac, tol, max_nodes, bc_tol=bc_tol)`.
///
/// `y` is the initial guess at the mesh nodes in SciPy's `(n, m)` layout (`y[i][j]` =
/// component `i` at node `j`) and `p` the initial guess for the k unknown parameters. A
/// solution that does not meet `tol` is not an error: it comes back with `success = false` and
/// SciPy's `status` (1 `max_nodes` exceeded, 2 singular Jacobian, 3 `bc_tol` not met).
///
/// # Errors
/// `BvpError::InvalidArgument` for the inputs SciPy rejects with `ValueError` (a mesh that is
/// not strictly increasing, a guess of the wrong shape, `fun`/`bc`/Jacobians returning the
/// wrong number of values, `S` not n×n), and for non-finite mesh or tolerance values.
pub fn solve_bvp<F, BC>(
    fun: F,
    bc: BC,
    x: &[f64],
    y: &[Vec<f64>],
    p: &[f64],
    options: BvpOptions<'_>,
) -> Result<BvpResult, BvpError>
where
    F: Fn(f64, &[f64], &[f64]) -> Vec<f64>,
    BC: Fn(&[f64], &[f64], &[f64]) -> Vec<f64>,
{
    let m = x.len();
    if m < 2 {
        return Err(BvpError::InvalidArgument(
            "`x` must have at least 2 nodes".to_string(),
        ));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(BvpError::InvalidArgument("`x` must be finite".to_string()));
    }
    if x.windows(2).any(|w| w[1] - w[0] <= 0.0) {
        return Err(BvpError::InvalidArgument(
            "`x` must be strictly increasing.".to_string(),
        ));
    }
    let n = y.len();
    if n == 0 {
        return Err(BvpError::InvalidArgument(
            "`y` must have at least one row".to_string(),
        ));
    }
    if let Some(row) = y.iter().find(|row| row.len() != m) {
        return Err(BvpError::InvalidArgument(format!(
            "`y` is expected to have {m} columns, but actually has {}.",
            row.len()
        )));
    }
    if !options.tol.is_finite() {
        return Err(BvpError::InvalidArgument(
            "`tol` must be finite".to_string(),
        ));
    }
    // SciPy warns and raises a tolerance below 100·eps to 100·eps.
    let tol = options.tol.max(100.0 * f64::EPSILON);
    let bc_tol = options.bc_tol.unwrap_or(tol);
    if bc_tol.is_nan() {
        return Err(BvpError::InvalidArgument(
            "`bc_tol` must not be NaN".to_string(),
        ));
    }
    let k = p.len();

    let singular = match options.singular_term {
        None => None,
        Some(s_rows) => {
            if s_rows.len() != n || s_rows.iter().any(|r| r.len() != n) {
                return Err(BvpError::InvalidArgument(format!(
                    "`S` is expected to have shape ({n}, {n})"
                )));
            }
            let s = DMatrix::from_fn(n, n, |i, j| s_rows[i][j]);
            let identity = DMatrix::<f64>::identity(n, n);
            let b = &identity - pinv(&s)? * &s;
            let d = pinv(&(&identity - &s))?;
            Some(Singular {
                s: row_major(&s),
                b: row_major(&b),
                d: row_major(&d),
            })
        }
    };

    let fun_dyn = |xq: f64, yq: &[f64], pq: &[f64]| fun(xq, yq, pq);
    let bc_dyn = |ya: &[f64], yb: &[f64], pq: &[f64]| bc(ya, yb, pq);
    let fun_jac_flat = options.fun_jac.map(|jac| {
        move |xq: f64, yq: &[f64], pq: &[f64]| {
            let (dy, dp) = jac(xq, yq, pq);
            (flatten_rows(&dy, n), flatten_rows(&dp, k))
        }
    });
    let bc_jac_flat = options.bc_jac.map(|jac| {
        move |ya: &[f64], yb: &[f64], pq: &[f64]| {
            let (da, db, dp) = jac(ya, yb, pq);
            (
                flatten_rows(&da, n),
                flatten_rows(&db, n),
                flatten_rows(&dp, k),
            )
        }
    });
    let problem = Problem {
        n,
        k,
        a: x[0],
        fun: &fun_dyn,
        bc: &bc_dyn,
        fun_jac: fun_jac_flat
            .as_ref()
            .map(|f| f as &dyn Fn(f64, &[f64], &[f64]) -> (Vec<f64>, Vec<f64>)),
        bc_jac: bc_jac_flat
            .as_ref()
            .map(|f| f as &dyn Fn(&[f64], &[f64], &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>)),
        singular,
    };
    let y_nodes: Vec<f64> = (0..m)
        .flat_map(|j| y.iter().map(move |row| row[j]))
        .collect();
    let (out, spline) = collocation::solve(
        &problem,
        x.to_vec(),
        y_nodes,
        p.to_vec(),
        tol,
        options.max_nodes,
        bc_tol,
        &factorize_sparse,
    )
    .map_err(BvpError::InvalidArgument)?;

    let m_out = out.x.len();
    let unpack = |flat: &[f64]| -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| (0..m_out).map(|j| flat[j * n + i]).collect())
            .collect()
    };
    Ok(BvpResult {
        y: unpack(&out.y),
        yp: unpack(&out.yp),
        x: out.x,
        p: out.p,
        rms_residuals: out.rms_residuals,
        niter: out.niter,
        // status: SciPy's termination code; 0 = every interval's RMS residual <= tol and
        // every bc residual <= bc_tol
        success: out.status == 0,
        message: collocation::termination_message(out.status).to_string(),
        status: out.status,
        spline,
    })
}

/// Batched boundary-value problems: N independent [`solve_bvp`] calls that share the mesh,
/// guess and options but differ by a parameter row (`fun(x, y, params)`,
/// `bc(ya, yb, params)`; no unknown parameters), fanned across cores. Result `i` is
/// byte-identical to the per-row `solve_bvp` call.
pub fn solve_bvp_many<F, BC>(
    fun: F,
    bc: BC,
    x: &[f64],
    y: &[Vec<f64>],
    param_rows: &[Vec<f64>],
    options: BvpOptions<'_>,
) -> Vec<Result<BvpResult, BvpError>>
where
    F: Fn(f64, &[f64], &[f64]) -> Vec<f64> + Sync,
    BC: Fn(&[f64], &[f64], &[f64]) -> Vec<f64> + Sync,
{
    let nrows = param_rows.len();
    if nrows == 0 {
        return Vec::new();
    }
    let (fun_ref, bc_ref) = (&fun, &bc);
    let solve_one = move |params: &[f64]| {
        solve_bvp(
            |xq: f64, yq: &[f64], _: &[f64]| fun_ref(xq, yq, params),
            |ya: &[f64], yb: &[f64], _: &[f64]| bc_ref(ya, yb, params),
            x,
            y,
            &[],
            options,
        )
    };

    // Each BVP is an independent, expensive (collocation-Newton) solve → fan whole parameter sets
    // across cores, capped by the row count; a tiny sweep stays serial.
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(nrows);
    if nthreads <= 1 || nrows < 4 {
        return param_rows.iter().map(|p| solve_one(p)).collect();
    }

    let chunk = nrows.div_ceil(nthreads);
    let solve_one = &solve_one;
    let chunk_results: Vec<Vec<Result<BvpResult, BvpError>>> = std::thread::scope(|scope| {
        (0..nthreads)
            .filter_map(|t| {
                let lo = t * chunk;
                if lo >= nrows {
                    return None;
                }
                let hi = (lo + chunk).min(nrows);
                Some(scope.spawn(move || {
                    (lo..hi)
                        .map(|i| solve_one(&param_rows[i]))
                        .collect::<Vec<_>>()
                }))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("solve_bvp_many worker panicked"))
            .collect()
    });

    let mut out = Vec::with_capacity(nrows);
    for cr in chunk_results {
        out.extend(cr);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linspace(a: f64, b: f64, m: usize) -> Vec<f64> {
        (0..m)
            .map(|i| a + (b - a) * i as f64 / (m - 1) as f64)
            .collect()
    }

    // frankenscipy-1ksfv.7, the negative case: y'' = 100·y on [0, 10], y(0) = 1, y(10) = 0.
    // Single shooting integrates an IVP that grows like e^(10·t) and cannot hit y(10) = 0;
    // collocation resolves the boundary layer. SciPy 1.17.1 (tol 1e-6, x = linspace(0, 10, 11),
    // y = 0): status 0, 233 nodes, 7 iterations, max |error| vs sinh(10(10−t))/sinh(100) 2.1e-9.
    #[test]
    fn solve_bvp_resolves_a_boundary_layer_like_scipy() {
        let x = linspace(0.0, 10.0, 11);
        let r = solve_bvp(
            |_, y, _| vec![y[1], 100.0 * y[0]],
            |ya, yb, _| vec![ya[0] - 1.0, yb[0]],
            &x,
            &[vec![0.0; 11], vec![0.0; 11]],
            &[],
            BvpOptions {
                tol: 1e-6,
                ..BvpOptions::default()
            },
        )
        .expect("solve_bvp");
        assert!(r.success, "{}", r.message);
        assert_eq!((r.status, r.niter, r.x.len()), (0, 7, 233), "SciPy's path");
        let worst = linspace(0.0, 10.0, 201)
            .iter()
            .map(|&t| {
                let exact = (10.0 * (10.0 - t)).sinh() / 100.0_f64.sinh();
                (r.sol(t)[0] - exact).abs()
            })
            .fold(0.0_f64, f64::max);
        assert!(worst < 1e-8, "max error {worst:e}");
        assert!(r.rms_residuals.iter().all(|&v| v <= 1e-6));
    }

    // Troesch's problem y'' = 5·sinh(5y), y(0) = 0, y(1) = 1: y'(0) = 4.575046e-2 (literature);
    // SciPy (x = linspace(0, 1, 11), y = [x, 1]): 29 nodes, y'(0) = 0.04575624953958903.
    #[test]
    fn solve_bvp_solves_troesch() {
        let x = linspace(0.0, 1.0, 11);
        let r = solve_bvp(
            |_, y, _| vec![y[1], 5.0 * (5.0 * y[0]).sinh()],
            |ya, yb, _| vec![ya[0], yb[0] - 1.0],
            &x,
            &[x.clone(), vec![1.0; 11]],
            &[],
            BvpOptions::default(),
        )
        .expect("solve_bvp");
        assert!(r.success, "{}", r.message);
        assert_eq!(r.x.len(), 29);
        assert!((r.sol(0.0)[1] - 0.045_756_249_539_589_03).abs() < 1e-12);
    }

    // SciPy's docs example: y'' + k²y = 0, y(0) = y(1) = 0, y'(0) = k, unknown k from 6
    // converges to 2π (SciPy: 6.283294600464651 at tol 1e-3).
    #[test]
    fn solve_bvp_finds_an_unknown_parameter() {
        let x = linspace(0.0, 1.0, 5);
        let r = solve_bvp(
            |_, y, p| vec![y[1], -p[0] * p[0] * y[0]],
            |ya, yb, p| vec![ya[0], yb[0], ya[1] - p[0]],
            &x,
            &[vec![0.0, 1.0, 0.0, -1.0, 0.0], vec![0.0; 5]],
            &[6.0],
            BvpOptions::default(),
        )
        .expect("solve_bvp");
        assert!(r.success, "{}", r.message);
        assert!(
            (r.p[0] - 6.283_294_600_464_651).abs() < 1e-10,
            "p = {:?}",
            r.p
        );
    }

    // Emden: y'' + (2/x)·y' + y⁵ = 0, y'(0) = 0, y(1) = √(3/4) with the singular term
    // S = [[0, 0], [0, −2]]; exact solution (1 + x²/3)^(−1/2).
    #[test]
    fn solve_bvp_handles_the_singular_term() {
        let x = linspace(0.0, 1.0, 10);
        let s = [vec![0.0, 0.0], vec![0.0, -2.0]];
        let r = solve_bvp(
            |_, y, _| vec![y[1], -y[0].powi(5)],
            |ya, yb, _| vec![ya[1], yb[0] - 0.75_f64.sqrt()],
            &x,
            &[vec![0.75_f64.sqrt(); 10], vec![1e-4; 10]],
            &[],
            BvpOptions {
                singular_term: Some(&s),
                ..BvpOptions::default()
            },
        )
        .expect("solve_bvp");
        assert!(r.success, "{}", r.message);
        for t in linspace(0.0, 1.0, 21) {
            let exact = (1.0 + t * t / 3.0).powf(-0.5);
            assert!((r.sol(t)[0] - exact).abs() < 1e-4, "t = {t}");
        }
    }

    // SciPy status 1: the same boundary layer with max_nodes = 40 stops after 2 iterations at
    // 31 nodes, not converged — and must not claim success.
    #[test]
    fn solve_bvp_reports_max_nodes_exceeded() {
        let x = linspace(0.0, 10.0, 11);
        let r = solve_bvp(
            |_, y, _| vec![y[1], 100.0 * y[0]],
            |ya, yb, _| vec![ya[0] - 1.0, yb[0]],
            &x,
            &[vec![0.0; 11], vec![0.0; 11]],
            &[],
            BvpOptions {
                tol: 1e-6,
                max_nodes: 40,
                ..BvpOptions::default()
            },
        )
        .expect("solve_bvp");
        assert!(!r.success);
        assert_eq!((r.status, r.niter, r.x.len()), (1, 2, 31));
        assert_eq!(r.message, "The maximum number of mesh nodes is exceeded.");
    }

    // Analytic Jacobians reach the same solution as SciPy's finite differences.
    #[test]
    fn solve_bvp_uses_analytic_jacobians() {
        let x = linspace(0.0, 1.0, 5);
        let fun_jac = |_: f64, y: &[f64], _: &[f64]| {
            (
                vec![vec![0.0, 1.0], vec![-y[0].exp(), 0.0]],
                vec![vec![], vec![]],
            )
        };
        let bc_jac = |_: &[f64], _: &[f64], _: &[f64]| {
            (
                vec![vec![1.0, 0.0], vec![0.0, 0.0]],
                vec![vec![0.0, 0.0], vec![1.0, 0.0]],
                vec![vec![], vec![]],
            )
        };
        let r = solve_bvp(
            |_, y, _| vec![y[1], -y[0].exp()],
            |ya, yb, _| vec![ya[0], yb[0]],
            &x,
            &[vec![3.0; 5], vec![0.0; 5]],
            &[],
            BvpOptions {
                fun_jac: Some(&fun_jac),
                bc_jac: Some(&bc_jac),
                ..BvpOptions::default()
            },
        )
        .expect("solve_bvp");
        assert!(r.success, "{}", r.message);
        // Bratu's upper branch: SciPy y'(0) = 10.846976018307315 (finite differences).
        assert!((r.sol(0.0)[1] - 10.846_976_018_307_315).abs() < 1e-3);
    }

    #[test]
    fn solve_bvp_rejects_what_scipy_rejects() {
        let f = |_: f64, y: &[f64], _: &[f64]| vec![y[1], 0.0];
        let bc = |ya: &[f64], yb: &[f64], _: &[f64]| vec![ya[0], yb[0] - 1.0];
        let guess = [vec![0.0; 3], vec![0.0; 3]];
        let opts = BvpOptions::default();
        let err = |x: &[f64], y: &[Vec<f64>]| solve_bvp(f, bc, x, y, &[], opts).unwrap_err();
        assert!(
            matches!(err(&[0.0, 1.0, 1.0], &guess), BvpError::InvalidArgument(m) if m.contains("strictly increasing"))
        );
        assert!(
            matches!(err(&[0.0, 0.5, f64::NAN], &guess), BvpError::InvalidArgument(m) if m.contains("finite"))
        );
        assert!(
            matches!(err(&[0.0, 1.0], &guess), BvpError::InvalidArgument(m) if m.contains("columns"))
        );
        let short_bc = |ya: &[f64], _: &[f64], _: &[f64]| vec![ya[0]];
        let e = solve_bvp(f, short_bc, &[0.0, 0.5, 1.0], &guess, &[], opts).unwrap_err();
        assert!(
            matches!(e, BvpError::InvalidArgument(m) if m.contains("`bc` returned 1 values, expected 2"))
        );
        let s = [vec![0.0]];
        let e = solve_bvp(
            f,
            bc,
            &[0.0, 0.5, 1.0],
            &guess,
            &[],
            BvpOptions {
                singular_term: Some(&s),
                ..opts
            },
        )
        .unwrap_err();
        assert!(matches!(e, BvpError::InvalidArgument(m) if m.contains("`S`")));
    }

    #[test]
    fn solve_bvp_many_byte_identical_to_per_param() {
        // Parameter sweep of a nonlinear BVP: y0'=y1, y1'=p*(1+y0^2); y0(0)=0, y0(1)=1.
        // The batched solve must equal looping solve_bvp per parameter, bit-for-bit.
        let f = |_t: f64, y: &[f64], p: &[f64]| vec![y[1], p[0] * (1.0 + y[0] * y[0])];
        let bc = |ya: &[f64], yb: &[f64], _p: &[f64]| vec![ya[0], yb[0] - 1.0];
        let mut s = 21u64;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            0.5 + 1.5 * ((s >> 11) as f64 / (1u64 << 53) as f64)
        };
        let nrows = 8usize; // crosses the serial->parallel gate
        let params: Vec<Vec<f64>> = (0..nrows).map(|_| vec![rng()]).collect();
        let x = linspace(0.0, 1.0, 6);
        let guess = [x.clone(), vec![1.0; 6]];
        let opts = BvpOptions::default();

        let batched = solve_bvp_many(f, bc, &x, &guess, &params, opts);
        assert_eq!(batched.len(), nrows);
        for (i, p) in params.iter().enumerate() {
            let single = solve_bvp(
                |t: f64, y: &[f64], _: &[f64]| f(t, y, p),
                |ya: &[f64], yb: &[f64], _: &[f64]| bc(ya, yb, p),
                &x,
                &guess,
                &[],
                opts,
            )
            .expect("single solve");
            let many = batched[i].as_ref().expect("batched member");
            assert_eq!(many.status, single.status, "status mismatch param {i}");
            assert_eq!(many.x.len(), single.x.len(), "mesh size mismatch param {i}");
            for (a, b) in many.x.iter().zip(&single.x) {
                assert_eq!(a.to_bits(), b.to_bits(), "x mismatch param {i}");
            }
            for (ra, rb) in many.y.iter().zip(&single.y) {
                for (a, b) in ra.iter().zip(rb) {
                    assert_eq!(a.to_bits(), b.to_bits(), "y mismatch param {i}");
                }
            }
        }
        assert!(batched.iter().all(|r| r.as_ref().is_ok_and(|x| x.success)));
        assert!(solve_bvp_many(f, bc, &x, &guess, &[], opts).is_empty());
    }
}

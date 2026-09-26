#![forbid(unsafe_code)]
//! Live SciPy accuracy contract for `spsolve` on a system none of its shortcuts may take
//! (frankenscipy-szq1n.4).
//!
//! spsolve used to hand SPD systems with n >= 4096 to conjugate gradients stopped at a 1e-8
//! residual, labelled as the sparse LU. That shortcut is gone and `backend_used` now names
//! the arm that produced x; this file holds the direct path to SciPy's accuracy on the
//! bead's system:
//! - a 100 x 100 grid (n = 10,000) 5-point Laplacian with a log-uniform coefficient field in
//!   [1e-6, 1], harmonic face coefficients and Dirichlet faces, so an SPD M-matrix;
//! - symmetrically permuted by a seeded random permutation, so its half-bandwidth is ~n and
//!   the banded arms cannot take it;
//! - a seeded standard-normal right-hand side.
//!
//! Asserted:
//! - normwise backward error ||b - A x||_inf / (||A||_inf ||x||_inf + ||b||_inf) of fsci's x
//!   at most BACKWARD_ERR_TOL;
//! - relative forward error of fsci's x against SciPy's `spsolve` on the same triplets at
//!   most FORWARD_REL_TOL. Measured on this system: SciPy's backward error is 7.3e-18 and a
//!   one-ulp change in b moves SciPy's own x by 9.2e-16 relative; fsci's backward error is
//!   2.1e-17 and its x is 6.6e-13 from SciPy's (the same construction with another seed has
//!   cond_1(A) ~ 9e6);
//! - `backend_used` is a direct sparse LU, not a banded, dense or iterative arm.
//!
//! Must-miss: fsci's own CG stopped at the old shortcut's 1e-8 relative residual is run on
//! the same system and must FAIL the forward-error contract, so the contract can see the
//! defect it exists for. Measured: that answer is 3.2e-9 from SciPy's (32x the bar) while
//! its normwise backward error, 1.2e-13, is only just above BACKWARD_ERR_TOL; the forward
//! contract is the one that separates the shortcut from a direct solve.

use std::io::Write;
use std::process::Stdio;

use fsci_conformance::CompareLedger;
use fsci_sparse::{
    CooMatrix, CsrMatrix, FormatConvertible, IterativeSolveOptions, Shape2D, SolveOptions,
    SparseBackend, cg, spsolve,
};
use serde::Deserialize;

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// The one case: the bead's permuted 100 x 100 grid Laplacian and its right-hand side.
const CASE: &str = "laplacian_grid100_permuted";
/// Single-case arms. Positive: `backend_direct_lu` (boolean), `backward_error` (fsci's
/// normwise backward error against its analytic value 0, bound BACKWARD_ERR_TOL) and
/// `forward_error` (fsci's x against live SciPy's `spsolve` x). Must-miss:
/// `cg_shortcut_fails_forward` passes only when the old CG shortcut's x FAILS the forward
/// contract against SciPy's x; it is the negative control, not a comparison of the shortcut.
const ARMS: [&str; 4] = [
    "backend_direct_lu",
    "backward_error",
    "forward_error",
    "cg_shortcut_fails_forward",
];
/// Normwise backward error of fsci's x; SciPy's SuperLU reaches ~1e-17 here.
const BACKWARD_ERR_TOL: f64 = 1e-13;
/// ||x_fsci - x_scipy||_inf / ||x_scipy||_inf.
const FORWARD_REL_TOL: f64 = 1e-10;
/// The removed shortcut's stopping rule: CG to a 1e-8 relative residual.
const OLD_SHORTCUT_CG_TOL: f64 = 1e-8;
const GRID: usize = 100;

/// SplitMix64: a seeded stream with no dependency on the `rand` API.
struct SplitMix(u64);

impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform on [0, 1).
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn standard_normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

struct System {
    n: usize,
    rows: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<f64>,
    b: Vec<f64>,
}

fn build_system() -> System {
    let m = GRID;
    let n = m * m;
    let mut rng = SplitMix(20_260_926);
    let k: Vec<f64> = (0..n).map(|_| 10f64.powf(-6.0 * rng.uniform())).collect();
    // Symmetric permutation: grid cell c becomes unknown perm[c] (Fisher-Yates).
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        perm.swap(i, j);
    }
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..m {
        for j in 0..m {
            let c = i * m + j;
            let mut diag = 0.0;
            for (di, dj) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
                let (ii, jj) = (i as i64 + di, j as i64 + dj);
                if (0..m as i64).contains(&ii) && (0..m as i64).contains(&jj) {
                    let nb = ii as usize * m + jj as usize;
                    let face = 2.0 * k[c] * k[nb] / (k[c] + k[nb]);
                    rows.push(perm[c]);
                    cols.push(perm[nb]);
                    vals.push(-face);
                    diag += face;
                } else {
                    diag += 2.0 * k[c];
                }
            }
            rows.push(perm[c]);
            cols.push(perm[c]);
            vals.push(diag);
        }
    }
    let b = (0..n).map(|_| rng.standard_normal()).collect();
    System {
        n,
        rows,
        cols,
        vals,
        b,
    }
}

fn to_csr(s: &System) -> CsrMatrix {
    CooMatrix::from_triplets(
        Shape2D::new(s.n, s.n),
        s.vals.clone(),
        s.rows.clone(),
        s.cols.clone(),
        true,
    )
    .expect("triplets")
    .to_csr()
    .expect("csr")
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(
        0.0_f64,
        |m, x| if x.is_nan() { f64::NAN } else { m.max(x.abs()) },
    )
}

/// Normwise backward error ||b - A x||_inf / (||A||_inf ||x||_inf + ||b||_inf).
fn backward_error(a: &CsrMatrix, x: &[f64], b: &[f64]) -> f64 {
    let (indptr, indices, data) = (a.indptr(), a.indices(), a.data());
    let mut residual = vec![0.0; b.len()];
    let mut a_norm = 0.0_f64;
    for row in 0..b.len() {
        let (mut ax, mut abs_row) = (0.0, 0.0);
        for idx in indptr[row]..indptr[row + 1] {
            ax += data[idx] * x[indices[idx]];
            abs_row += data[idx].abs();
        }
        residual[row] = b[row] - ax;
        a_norm = a_norm.max(abs_row);
    }
    inf_norm(&residual) / (a_norm * inf_norm(x) + inf_norm(b))
}

#[derive(Debug, Deserialize)]
struct Oracle {
    x: Vec<f64>,
    backward_error: f64,
    one_ulp_envelope: f64,
}

fn scipy_solution(s: &System) -> Option<Oracle> {
    let script = r#"
import json, sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
q = json.load(sys.stdin)
n = q["n"]
A = sp.csc_matrix((q["vals"], (q["rows"], q["cols"])), shape=(n, n))
b = np.array(q["b"])
x = spla.spsolve(A, b)
bwd = np.abs(b - A @ x).max() / (abs(A).sum(axis=1).max() * np.abs(x).max() + np.abs(b).max())
x2 = spla.spsolve(A, np.nextafter(b, np.inf))
env = np.abs(x2 - x).max() / np.abs(x).max()
json.dump({"x": x.tolist(), "backward_error": float(bwd), "one_ulp_envelope": float(env)},
          sys.stdout)
"#;
    let query = serde_json::json!({
        "n": s.n, "rows": s.rows, "cols": s.cols, "vals": s.vals, "b": s.b,
    });
    let mut child = match fsci_conformance::scipy_oracle_command()
        .args(["-c", script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn the spsolve oracle: {e}"
            );
            return None;
        }
    };
    child
        .stdin
        .as_mut()
        .expect("oracle stdin")
        .write_all(query.to_string().as_bytes())
        .expect("write spsolve query");
    let output = child
        .wait_with_output()
        .expect("wait for the spsolve oracle");
    if !output.status.success() {
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "spsolve oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    Some(serde_json::from_slice(&output.stdout).expect("parse spsolve oracle JSON"))
}

#[test]
fn diff_sparse_spsolve_accuracy_contract() {
    let system = build_system();
    let a = to_csr(&system);
    let half_bandwidth = system
        .rows
        .iter()
        .zip(&system.cols)
        .map(|(&r, &c)| r.abs_diff(c))
        .max()
        .unwrap_or(0);
    assert!(
        half_bandwidth > system.n / 2,
        "the permutation must defeat the banded arms (half-bandwidth {half_bandwidth})"
    );

    let result = spsolve(&a, &system.b, SolveOptions::default()).expect("spsolve");
    let fsci_bwd = backward_error(&a, &result.solution, &system.b);
    println!(
        "fsci spsolve: backend {:?}, ordering {:?}, warnings {:?}, backward error {fsci_bwd:.3e}",
        result.backend_used, result.ordering_used, result.warnings
    );
    assert!(
        matches!(result.backend_used, SparseBackend::NativeSparseLu),
        "spsolve took {:?} on a wide-band SPD system of n = {}",
        result.backend_used,
        system.n
    );
    assert!(
        fsci_bwd <= BACKWARD_ERR_TOL,
        "fsci spsolve backward error {fsci_bwd:.3e} > {BACKWARD_ERR_TOL:e}"
    );

    // Must-miss: the removed shortcut's answer, checked against SciPy below.
    let shortcut = cg(
        &a,
        &system.b,
        None,
        IterativeSolveOptions {
            tol: OLD_SHORTCUT_CG_TOL,
            max_iter: Some(20 * system.n),
            ..IterativeSolveOptions::default()
        },
    )
    .expect("cg");
    let shortcut_bwd = backward_error(&a, &shortcut.solution, &system.b);
    println!(
        "CG to rtol {OLD_SHORTCUT_CG_TOL:e} (the removed shortcut): converged {}, {} iterations, backward error {shortcut_bwd:.3e}",
        shortcut.converged, shortcut.iterations
    );

    let Some(oracle) = scipy_solution(&system) else {
        // SciPy unavailable outside FSCI_REQUIRE_SCIPY_ORACLE=1 (under it scipy_solution
        // panics): the fsci-only asserts above have run and nothing is ledgered, as in the
        // other live-oracle tests that skip before building their ledger.
        return;
    };
    let scipy_norm = inf_norm(&oracle.x);
    // NaN-propagating (inf_norm), so a NaN in either solution fails rather than vanishing.
    let relative_gap = |x: &[f64]| {
        let gap: Vec<f64> = x.iter().zip(&oracle.x).map(|(f, s)| f - s).collect();
        inf_norm(&gap) / scipy_norm
    };
    let forward = relative_gap(&result.solution);
    let shortcut_forward = relative_gap(&shortcut.solution);
    println!(
        "SciPy spsolve: backward error {:.3e}, its own 1-ulp-in-b envelope {:.3e}; fsci forward error vs SciPy {forward:.3e} (CG shortcut: {shortcut_forward:.3e})",
        oracle.backward_error, oracle.one_ulp_envelope
    );

    let mut ledger = CompareLedger::new("diff_sparse_spsolve_accuracy_contract", &ARMS);
    ledger.compared(
        "backend_direct_lu",
        CASE,
        matches!(result.backend_used, SparseBackend::NativeSparseLu),
    );
    // The backward error of an exact solve is 0; pair records a NaN or infinite one (inf_norm
    // propagates a NaN in x) as an fsci failure.
    if ledger
        .pair("backward_error", CASE, Some(0.0), Some(fsci_bwd))
        .is_some()
    {
        ledger.compared("backward_error", CASE, fsci_bwd <= BACKWARD_ERR_TOL);
    }
    // slices records a non-finite element of fsci's x, or an x of the wrong length that the zip
    // in relative_gap would silently truncate.
    if ledger
        .slices(
            "forward_error",
            CASE,
            Some(oracle.x.as_slice()),
            Some(result.solution.as_slice()),
        )
        .is_some()
    {
        ledger.compared("forward_error", CASE, forward <= FORWARD_REL_TOL);
    }
    // Negative control: the verdict is that the shortcut MISSED the forward contract. A NaN or
    // wrongly sized shortcut answer is a broken control (slices records it), never a miss.
    if ledger
        .slices(
            "cg_shortcut_fails_forward",
            CASE,
            Some(oracle.x.as_slice()),
            Some(shortcut.solution.as_slice()),
        )
        .is_some()
    {
        ledger.compared(
            "cg_shortcut_fails_forward",
            CASE,
            shortcut_forward > FORWARD_REL_TOL,
        );
    }

    assert_eq!(
        oracle.x.len(),
        system.n,
        "the oracle answered a different system"
    );
    assert!(
        forward <= FORWARD_REL_TOL,
        "fsci spsolve is {forward:.3e} from SciPy's spsolve (contract {FORWARD_REL_TOL:e})"
    );
    assert!(
        shortcut_forward > FORWARD_REL_TOL,
        "the contract cannot tell the old CG shortcut from a direct solve ({shortcut_forward:.3e})"
    );
    // One system, so every arm (the must-miss included) compares exactly that one case.
    ledger.finish(1);
}

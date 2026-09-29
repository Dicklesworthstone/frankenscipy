#![forbid(unsafe_code)]
//! Property-based + scipy parity for fsci_opt::lbfgsb.
//!
//! Resolves [frankenscipy-pcc1j]. L-BFGS-B is bound-constrained
//! quasi-Newton optimization. Verifies:
//!   * Unconstrained quadratic f(x) = (x-target)² finds target
//!   * Bounded quadratic clips to the bound when target is outside
//!   * Smooth multivariate problem (sum of squares) converges to origin
//!   * scipy parity on a Rosenbrock-style problem (converged solutions
//!     match closely)
//!
//! `diff_opt_lbfgsb_scipy_path` (frankenscipy-1ksfv.18) compares the PATH, live: 18 cases against
//! SciPy's L-BFGS-B 3.0 per evaluation — see its section comment for the kernel-spread rules.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_opt::types::Bound;
use fsci_opt::{MinimizeOptions, lbfgsb};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-4;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    pass: bool,
    note: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create lbfgsb diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct ScipyMinimum {
    converged: bool,
    x: Option<Vec<f64>>,
    fun: Option<f64>,
}

fn scipy_oracle_rosen_lbfgsb_or_skip() -> Option<ScipyMinimum> {
    let script = r#"
import json, math, sys
import numpy as np
from scipy.optimize import minimize

def rosen(x):
    x = np.asarray(x, dtype=float)
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

res = minimize(rosen, [-1.2, 1.0], method='L-BFGS-B', tol=1e-8,
               options={'maxiter': 2000})
out = {
    "converged": bool(res.success),
    "x": [float(v) for v in res.x] if res.success else None,
    "fun": float(res.fun) if res.success else None,
}
print(json.dumps(out))
"#;
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "python3 spawn failed: {e}"
            );
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open stdin");
        if stdin.write_all(b"").is_err() {
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "L-BFGS-B rosen oracle failed: {stderr}"
        );
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse JSON"))
}

#[test]
fn diff_opt_lbfgsb_minimize() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut check = |id: &str, ok: bool, note: String| {
        diffs.push(CaseDiff {
            case_id: id.into(),
            pass: ok,
            note,
        });
    };

    let opts = MinimizeOptions {
        tol: Some(1.0e-10),
        maxiter: Some(2000),
        ..MinimizeOptions::default()
    };

    // `None` only when python3/SciPy is unavailable and FSCI_REQUIRE_SCIPY_ORACLE is unset (the
    // oracle asserts otherwise); the SciPy arm is then not declared, the way every other live test
    // skips without its oracle. The analytic arm compares against the known minimizers.
    let scipy = scipy_oracle_rosen_lbfgsb_or_skip();
    let arms: &[&str] = if scipy.is_some() {
        &["analytic", "scipy"]
    } else {
        &["analytic"]
    };
    let mut ledger = CompareLedger::new("diff_opt_lbfgsb_minimize", arms);

    // === 1. Unconstrained scalar quadratic: minimize (x - 3)² → x = 3 ===
    {
        let id = "unconstrained_scalar_quadratic_finds_3";
        let f = |x: &[f64]| (x[0] - 3.0).powi(2);
        let x = lbfgsb(&f, &[0.0], opts, None).ok().map(|r| r.x);
        let fsci = x.as_ref().and_then(|x| x.first().copied());
        if let Some((want, got)) = ledger.pair("analytic", id, Some(3.0), fsci) {
            let ok = (got - want).abs() < ABS_TOL;
            ledger.compared("analytic", id, ok);
            check(id, ok, format!("x={:?}", x.unwrap_or_default()));
        }
    }

    // === 2. Bounded scalar: minimize (x - 5)² with x ∈ [0, 2] → x = 2 ===
    {
        let id = "bounded_clipped_to_upper";
        let f = |x: &[f64]| (x[0] - 5.0).powi(2);
        let bounds: [Bound; 1] = [(Some(0.0), Some(2.0))];
        let x = lbfgsb(&f, &[1.0], opts, Some(&bounds)).ok().map(|r| r.x);
        let fsci = x.as_ref().and_then(|x| x.first().copied());
        if let Some((want, got)) = ledger.pair("analytic", id, Some(2.0), fsci) {
            let ok = (got - want).abs() < ABS_TOL;
            ledger.compared("analytic", id, ok);
            check(id, ok, format!("x={:?}", x.unwrap_or_default()));
        }
    }

    // === 3. Bounded scalar: minimize (x + 5)² with x ∈ [0, 2] → x = 0 ===
    {
        let id = "bounded_clipped_to_lower";
        let f = |x: &[f64]| (x[0] + 5.0).powi(2);
        let bounds: [Bound; 1] = [(Some(0.0), Some(2.0))];
        let x = lbfgsb(&f, &[1.0], opts, Some(&bounds)).ok().map(|r| r.x);
        let fsci = x.as_ref().and_then(|x| x.first().copied());
        if let Some((want, got)) = ledger.pair("analytic", id, Some(0.0), fsci) {
            let ok = (got - want).abs() < ABS_TOL;
            ledger.compared("analytic", id, ok);
            check(id, ok, format!("x={:?}", x.unwrap_or_default()));
        }
    }

    // === 4. Multivariate sum of squares: minimize Σ x_i² → x = 0 ===
    {
        let id = "multivariate_sum_of_squares_to_origin";
        let f = |x: &[f64]| x.iter().map(|v| v.powi(2)).sum::<f64>();
        let x = lbfgsb(&f, &[1.0, -2.0, 3.0, -0.5], opts, None)
            .ok()
            .map(|r| r.x);
        let origin = [0.0_f64; 4];
        // The ledger rejects a NaN coordinate, which the max fold below would swallow.
        if let Some((_, got)) = ledger.slices("analytic", id, Some(origin.as_slice()), x.as_deref())
        {
            let max_abs = got.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            ledger.compared("analytic", id, max_abs < ABS_TOL);
            check(id, max_abs < ABS_TOL, format!("x={got:?}"));
        }
    }

    // === 5. scipy parity: Rosenbrock at standard init point ===
    {
        let rosen = |x: &[f64]| -> f64 {
            (0..x.len() - 1)
                .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
                .sum()
        };
        let fsci_r = lbfgsb(&rosen, &[-1.2, 1.0], opts, None).ok();
        let fsci_x = fsci_r.as_ref().map(|r| r.x.as_slice());
        let id = "rosenbrock_fsci_converges_near_one";
        let ones = [1.0_f64, 1.0];
        if let Some((_, x)) = ledger.slices("analytic", id, Some(ones.as_slice()), fsci_x) {
            let fsci_close = (x[0] - 1.0).abs() < 1.0e-3 && (x[1] - 1.0).abs() < 1.0e-3;
            ledger.compared("analytic", id, fsci_close);
            check(
                id,
                fsci_close,
                format!("x={:?} fun={:?}", x, fsci_r.as_ref().and_then(|r| r.fun)),
            );
        }

        if let Some(scipy) = &scipy {
            let id = "rosenbrock_close_to_scipy";
            // SciPy reports x only when it converged; otherwise the case is `oracle_missing`.
            let scipy_x = scipy.x.as_deref().filter(|_| scipy.converged);
            if let Some((scipy_x, x)) = ledger.slices("scipy", id, scipy_x, fsci_x) {
                let close_to_scipy =
                    (x[0] - scipy_x[0]).abs() < 1.0e-3 && (x[1] - scipy_x[1]).abs() < 1.0e-3;
                ledger.compared("scipy", id, close_to_scipy);
                check(id, close_to_scipy, format!("fsci={x:?} scipy={scipy_x:?}"));
            }
        }
    }

    // === 6. Bounds-with-x0-outside: x0 = -1.0 outside bounds [0, 5] should be projected ===
    {
        let id = "x0_outside_bounds_projected_then_solved";
        let f = |x: &[f64]| (x[0] - 2.0).powi(2);
        let bounds: [Bound; 1] = [(Some(0.0), Some(5.0))];
        let x = lbfgsb(&f, &[-1.0], opts, Some(&bounds)).ok().map(|r| r.x);
        let fsci = x.as_ref().and_then(|x| x.first().copied());
        if let Some((want, got)) = ledger.pair("analytic", id, Some(2.0), fsci) {
            let ok = (got - want).abs() < ABS_TOL;
            ledger.compared("analytic", id, ok);
            check(id, ok, format!("x={:?}", x.unwrap_or_default()));
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_opt_lbfgsb_minimize".into(),
        category: "fsci_opt::lbfgsb (L-BFGS-B) coverage".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("lbfgsb mismatch: {} — {}", d.case_id, d.note);
        }
    }

    assert!(all_pass, "lbfgsb coverage failed: {} cases", diffs.len());
    // The smaller arm: `scipy` has the one Rosenbrock case (`analytic` has six).
    ledger.finish(1);
}

// ════════════════════════════════════════════════════════════════════════════════════════════
// frankenscipy-1ksfv.18: SciPy's L-BFGS-B path, live.
//
// `minimize(method='L-BFGS-B')` is SciPy 1.17.1's `_minimize_lbfgsb` driving the C translation
// of L-BFGS-B 3.0, and fsci's is a transcription of it, so the comparison is per evaluation, not
// per minimum. Both sides evaluate the identical function (explicit operations in Rust and in the
// oracle's Python, no numpy reductions): Rosenbrock in 2, 5, 10 and 60 dimensions, free and in a
// box, a quadratic with lower bounds only and one with upper bounds only, Powell's badly scaled
// function, Beale in a box, a fixed variable (l = u), a start outside the box, maxiter and maxfev
// limits, and a non-smooth sum of absolute values; each with the analytic gradient or SciPy's
// bounded forward difference (`eps = 1e-8`).
//
// SciPy's own path depends on its BLAS kernel. Measured under OPENBLAS_CORETYPE = Prescott /
// Nehalem / Sandybridge / Haswell / Zen, 15 of these 18 cases take the same (nit, nfev, njev,
// status) on all five kernels, and fsci reproduces all 15 exactly, so there the counters must be
// SciPy's. The other three spread: rosen10/fd 72-73 iterations and 1.5e-5 in x, rosen60/jac
// 314-321 iterations and 2.3e-5, abs3/fd (non-smooth) 24-37 iterations with status 0 on three
// kernels and 2 (ABNORMAL line search) on two. There the status must be one of SciPy's, nit
// within 30% of this runner's SciPy or inside the measured range, and x within ~10x the measured
// spread (fsci is 2.3e-6, 3.3e-6 and 2.3e-9 from the nearest kernel). A crossed oracle (every
// case judged against the next case's SciPy result) fails all 18 cases, and the same port with
// maxcor 5 instead of 10 fails the outside-box case on its counters.
// ════════════════════════════════════════════════════════════════════════════════════════════

fn lb_rosen(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() - 1 {
        let a = x[i + 1] - x[i] * x[i];
        let b = 1.0 - x[i];
        s += 100.0 * a * a + b * b;
    }
    s
}

fn lb_rosen_der(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut g = vec![0.0; n];
    g[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
    for i in 1..n - 1 {
        g[i] = 200.0 * (x[i] - x[i - 1] * x[i - 1])
            - 400.0 * x[i] * (x[i + 1] - x[i] * x[i])
            - 2.0 * (1.0 - x[i]);
    }
    g[n - 1] = 200.0 * (x[n - 1] - x[n - 2] * x[n - 2]);
    g
}

fn lb_center(i: usize) -> f64 {
    if i.is_multiple_of(2) {
        (i + 1) as f64
    } else {
        -((i + 1) as f64)
    }
}

fn lb_quad(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for (i, &xi) in x.iter().enumerate() {
        let d = xi - lb_center(i);
        s += ((i + 1) as f64) * d * d;
    }
    s
}

fn lb_quad_der(x: &[f64]) -> Vec<f64> {
    x.iter()
        .enumerate()
        .map(|(i, &xi)| 2.0 * ((i + 1) as f64) * (xi - lb_center(i)))
        .collect()
}

fn lb_powell_bs(x: &[f64]) -> f64 {
    let f1 = 10000.0 * x[0] * x[1] - 1.0;
    let f2 = (-x[0]).exp() + (-x[1]).exp() - 1.0001;
    f1 * f1 + f2 * f2
}

fn lb_powell_bs_der(x: &[f64]) -> Vec<f64> {
    let f1 = 10000.0 * x[0] * x[1] - 1.0;
    let f2 = (-x[0]).exp() + (-x[1]).exp() - 1.0001;
    vec![
        2.0 * f1 * 10000.0 * x[1] - 2.0 * f2 * (-x[0]).exp(),
        2.0 * f1 * 10000.0 * x[0] - 2.0 * f2 * (-x[1]).exp(),
    ]
}

fn lb_beale(x: &[f64]) -> f64 {
    let a = 1.5 - x[0] + x[0] * x[1];
    let b = 2.25 - x[0] + x[0] * x[1] * x[1];
    let c = 2.625 - x[0] + x[0] * x[1] * x[1] * x[1];
    a * a + b * b + c * c
}

fn lb_beale_der(x: &[f64]) -> Vec<f64> {
    let a = 1.5 - x[0] + x[0] * x[1];
    let b = 2.25 - x[0] + x[0] * x[1] * x[1];
    let c = 2.625 - x[0] + x[0] * x[1] * x[1] * x[1];
    let y2 = x[1] * x[1];
    let y3 = y2 * x[1];
    vec![
        2.0 * a * (x[1] - 1.0) + 2.0 * b * (y2 - 1.0) + 2.0 * c * (y3 - 1.0),
        2.0 * a * x[0] + 2.0 * b * 2.0 * x[0] * x[1] + 2.0 * c * 3.0 * x[0] * y2,
    ]
}

fn lb_abs(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for v in x {
        s += (v - 0.3).abs();
    }
    s
}

struct LbCase {
    id: &'static str,
    /// The oracle's name for the objective.
    problem: &'static str,
    fun: fn(&[f64]) -> f64,
    grad: Option<fsci_opt::GradientFunc>,
    x0: Vec<f64>,
    bounds: Vec<Bound>,
    maxiter: Option<usize>,
    maxfev: Option<usize>,
    x_tol: f64,
    fun_tol: f64,
    /// All five measured kernels agree on (nit, nfev, njev, status).
    invariant: bool,
    statuses: &'static [i32],
    nit_range: (usize, usize),
}

fn lb_alternating(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| if i.is_multiple_of(2) { -1.2 } else { 1.0 })
        .collect()
}

#[allow(clippy::too_many_lines)]
fn lb_cases() -> Vec<LbCase> {
    let free = |n: usize| vec![(None, None); n];
    let r5 = vec![1.3, 0.7, 0.8, 1.9, 1.2];
    let mut fixed = free(5);
    fixed[2] = (Some(0.5), Some(0.5));
    let base = |id, problem, fun: fn(&[f64]) -> f64, grad, x0, bounds| LbCase {
        id,
        problem,
        fun,
        grad,
        x0,
        bounds,
        maxiter: None,
        maxfev: None,
        x_tol: 1.0e-9,
        fun_tol: 1.0e-9,
        invariant: true,
        statuses: &[0],
        nit_range: (0, 0),
    };
    let rosen_der: Option<fsci_opt::GradientFunc> = Some(lb_rosen_der);
    vec![
        base(
            "rosen2/jac",
            "rosen",
            lb_rosen,
            rosen_der,
            vec![-1.2, 1.0],
            free(2),
        ),
        LbCase {
            x_tol: 1.0e-6,
            ..base(
                "rosen2/fd",
                "rosen",
                lb_rosen,
                None,
                vec![-1.2, 1.0],
                free(2),
            )
        },
        base(
            "rosen5/jac",
            "rosen",
            lb_rosen,
            rosen_der,
            r5.clone(),
            free(5),
        ),
        base("rosen5/fd", "rosen", lb_rosen, None, r5.clone(), free(5)),
        LbCase {
            x_tol: 1.0e-7,
            ..base(
                "rosen10/jac",
                "rosen",
                lb_rosen,
                rosen_der,
                lb_alternating(10),
                free(10),
            )
        },
        LbCase {
            x_tol: 2.0e-4,
            fun_tol: 1.0e-8,
            invariant: false,
            nit_range: (72, 73),
            ..base(
                "rosen10/fd",
                "rosen",
                lb_rosen,
                None,
                lb_alternating(10),
                free(10),
            )
        },
        base(
            "rosen5_box/jac",
            "rosen",
            lb_rosen,
            rosen_der,
            r5.clone(),
            vec![(Some(-2.0), Some(0.5)); 5],
        ),
        LbCase {
            x_tol: 1.0e-8,
            ..base(
                "rosen5_box/fd",
                "rosen",
                lb_rosen,
                None,
                r5.clone(),
                vec![(Some(-2.0), Some(0.5)); 5],
            )
        },
        LbCase {
            x_tol: 2.0e-4,
            fun_tol: 1.0e-8,
            invariant: false,
            nit_range: (314, 321),
            ..base(
                "rosen60/jac",
                "rosen",
                lb_rosen,
                rosen_der,
                lb_alternating(60),
                free(60),
            )
        },
        base(
            "quad8_lb/jac",
            "quad",
            lb_quad,
            Some(lb_quad_der),
            vec![1.0; 8],
            vec![(Some(0.0), None); 8],
        ),
        base(
            "quad8_ub/fd",
            "quad",
            lb_quad,
            None,
            vec![0.0; 8],
            vec![(None, Some(0.5)); 8],
        ),
        base(
            "powell_bs/jac",
            "powell_bs",
            lb_powell_bs,
            Some(lb_powell_bs_der),
            vec![0.0, 1.0],
            free(2),
        ),
        base(
            "beale_box/jac",
            "beale",
            lb_beale,
            Some(lb_beale_der),
            vec![1.0, 1.0],
            vec![(Some(-4.5), Some(4.5)); 2],
        ),
        LbCase {
            maxiter: Some(5),
            statuses: &[1],
            ..base(
                "rosen10/jac/maxiter5",
                "rosen",
                lb_rosen,
                rosen_der,
                lb_alternating(10),
                free(10),
            )
        },
        LbCase {
            maxfev: Some(20),
            statuses: &[1],
            ..base(
                "rosen10/fd/maxfev20",
                "rosen",
                lb_rosen,
                None,
                lb_alternating(10),
                free(10),
            )
        },
        LbCase {
            x_tol: 1.0e-6,
            fun_tol: 1.0e-7,
            invariant: false,
            statuses: &[0, 2],
            nit_range: (24, 37),
            ..base("abs3/fd", "abs", lb_abs, None, vec![1.0, 2.0, 3.0], free(3))
        },
        base("rosen5_fixed/jac", "rosen", lb_rosen, rosen_der, r5, fixed),
        base(
            "rosen5_outside/jac",
            "rosen",
            lb_rosen,
            rosen_der,
            vec![3.0, -3.0, 3.0, -3.0, 3.0],
            vec![(Some(-2.0), Some(2.0)); 5],
        ),
    ]
}

#[derive(Debug, Clone, Serialize)]
struct LbQueryCase {
    case_id: String,
    problem: String,
    analytic: bool,
    x0: Vec<f64>,
    /// (lo, hi) with `null` for an absent bound.
    bounds: Vec<(Option<f64>, Option<f64>)>,
    maxiter: Option<usize>,
    maxfev: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct LbArm {
    case_id: String,
    x: Option<Vec<f64>>,
    fun: Option<f64>,
    status: Option<i32>,
    nit: Option<usize>,
    nfev: Option<usize>,
    njev: Option<usize>,
}

fn lbfgsb_path_oracle_or_skip(cases: &[LbQueryCase]) -> Option<Vec<LbArm>> {
    let script = r#"
import json, math, sys, warnings
import numpy as np
from scipy.optimize import minimize

def rosen(x):
    x = [float(v) for v in x]
    s = 0.0
    for i in range(len(x) - 1):
        a = x[i + 1] - x[i] * x[i]
        b = 1.0 - x[i]
        s += 100.0 * a * a + b * b
    return s

def rosen_der(x):
    x = [float(v) for v in x]
    n = len(x)
    g = [0.0] * n
    g[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0])
    for i in range(1, n - 1):
        g[i] = 200.0 * (x[i] - x[i - 1] * x[i - 1]) - 400.0 * x[i] * (x[i + 1] - x[i] * x[i]) - 2.0 * (1.0 - x[i])
    g[n - 1] = 200.0 * (x[n - 1] - x[n - 2] * x[n - 2])
    return np.array(g)

def center(i):
    return float(i + 1) if i % 2 == 0 else -float(i + 1)

def quad(x):
    x = [float(v) for v in x]
    s = 0.0
    for i in range(len(x)):
        d = x[i] - center(i)
        s += float(i + 1) * d * d
    return s

def quad_der(x):
    x = [float(v) for v in x]
    return np.array([2.0 * float(i + 1) * (x[i] - center(i)) for i in range(len(x))])

def powell_bs(x):
    f1 = 10000.0 * float(x[0]) * float(x[1]) - 1.0
    f2 = math.exp(-float(x[0])) + math.exp(-float(x[1])) - 1.0001
    return f1 * f1 + f2 * f2

def powell_bs_der(x):
    a, b = float(x[0]), float(x[1])
    f1 = 10000.0 * a * b - 1.0
    f2 = math.exp(-a) + math.exp(-b) - 1.0001
    return np.array([2.0 * f1 * 10000.0 * b - 2.0 * f2 * math.exp(-a),
                     2.0 * f1 * 10000.0 * a - 2.0 * f2 * math.exp(-b)])

def beale(x):
    x0, x1 = float(x[0]), float(x[1])
    a = 1.5 - x0 + x0 * x1
    b = 2.25 - x0 + x0 * x1 * x1
    c = 2.625 - x0 + x0 * x1 * x1 * x1
    return a * a + b * b + c * c

def beale_der(x):
    x0, x1 = float(x[0]), float(x[1])
    a = 1.5 - x0 + x0 * x1
    b = 2.25 - x0 + x0 * x1 * x1
    c = 2.625 - x0 + x0 * x1 * x1 * x1
    y2 = x1 * x1
    y3 = y2 * x1
    return np.array([2.0 * a * (x1 - 1.0) + 2.0 * b * (y2 - 1.0) + 2.0 * c * (y3 - 1.0),
                     2.0 * a * x0 + 2.0 * b * 2.0 * x0 * x1 + 2.0 * c * 3.0 * x0 * y2])

def absf(x):
    s = 0.0
    for v in x:
        s += abs(float(v) - 0.3)
    return s

PROBLEMS = {"rosen": (rosen, rosen_der), "quad": (quad, quad_der),
            "powell_bs": (powell_bs, powell_bs_der), "beale": (beale, beale_der),
            "abs": (absf, None)}

out = []
for case in json.load(sys.stdin):
    f, g = PROBLEMS[case["problem"]]
    arm = {"case_id": case["case_id"], "x": None, "fun": None, "status": None,
           "nit": None, "nfev": None, "njev": None}
    bounds = [(-math.inf if lo is None else lo, math.inf if hi is None else hi)
              for lo, hi in case["bounds"]]
    options = {}
    if case["maxiter"] is not None:
        options["maxiter"] = case["maxiter"]
    if case["maxfev"] is not None:
        options["maxfun"] = case["maxfev"]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = minimize(f, np.array(case["x0"], dtype=float), method="L-BFGS-B",
                         jac=g if case["analytic"] else None, bounds=bounds, options=options)
        arm.update(x=[float(v) for v in r.x], fun=float(r.fun), status=int(r.status),
                   nit=int(r.nit), nfev=int(r.nfev), njev=int(r.njev))
    except Exception:
        pass
    out.append(arm)
print(json.dumps(out, allow_nan=False))
"#;
    let query = serde_json::to_string(cases).expect("serialize L-BFGS-B query");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for the L-BFGS-B oracle: {e}"
            );
            eprintln!("skipping L-BFGS-B oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open L-BFGS-B oracle stdin");
        if let Err(err) = stdin.write_all(query.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "L-BFGS-B oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping L-BFGS-B oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for L-BFGS-B oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "L-BFGS-B oracle failed: {stderr}"
        );
        eprintln!("skipping L-BFGS-B oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse L-BFGS-B oracle JSON"))
}

/// fsci's status in `_minimize_lbfgsb`'s numbering: 0 converged, 1 maxfun / maxiter, 2 else.
fn lbfgsb_scipy_status(status: fsci_opt::ConvergenceStatus) -> i32 {
    match status {
        fsci_opt::ConvergenceStatus::Success => 0,
        fsci_opt::ConvergenceStatus::MaxIterations
        | fsci_opt::ConvergenceStatus::MaxEvaluations => 1,
        _ => 2,
    }
}

#[test]
fn diff_opt_lbfgsb_scipy_path() {
    let cases = lb_cases();
    let query: Vec<LbQueryCase> = cases
        .iter()
        .map(|c| LbQueryCase {
            case_id: c.id.to_string(),
            problem: c.problem.to_string(),
            analytic: c.grad.is_some(),
            x0: c.x0.clone(),
            bounds: c.bounds.clone(),
            maxiter: c.maxiter,
            maxfev: c.maxfev,
        })
        .collect();
    let Some(oracle) = lbfgsb_path_oracle_or_skip(&query) else {
        return;
    };
    let arms: std::collections::HashMap<String, LbArm> = oracle
        .into_iter()
        .map(|arm| (arm.case_id.clone(), arm))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut exact_paths = 0usize;
    let mut ledger = CompareLedger::new("diff_opt_lbfgsb_scipy_path", &["lbfgsb"]);
    for case in &cases {
        let arm = &arms[case.id];
        let options = MinimizeOptions {
            method: Some(fsci_opt::OptimizeMethod::LBfgsB),
            gradient: case.grad,
            maxiter: case.maxiter,
            maxfev: case.maxfev,
            ..MinimizeOptions::default()
        };
        let mut reasons = Vec::new();
        let fsci = lbfgsb(&case.fun, &case.x0, options, Some(&case.bounds))
            .inspect_err(|e| reasons.push(format!("fsci error {e}")))
            .ok();
        let scipy = match (arm.x.as_deref(), arm.fun, arm.status) {
            (Some(scipy_x), Some(scipy_fun), Some(scipy_status)) => {
                Some((scipy_x, scipy_fun, scipy_status))
            }
            _ => None,
        };
        if fsci.is_some() && scipy.is_none() {
            reasons.push("SciPy produced no result".to_string());
        }
        // `None` is recorded by the ledger: SciPy gave no result, or fsci returned an error.
        if let Some(((scipy_x, scipy_fun, scipy_status), r)) =
            ledger.both("lbfgsb", case.id, scipy, fsci)
        {
            let fsci_counts = (r.nit, r.nfev, r.njev);
            let scipy_counts = (
                arm.nit.unwrap_or(0),
                arm.nfev.unwrap_or(0),
                arm.njev.unwrap_or(0),
            );
            let status = lbfgsb_scipy_status(r.status);
            if !case.statuses.contains(&status) || (case.invariant && status != scipy_status) {
                reasons.push(format!(
                    "status {status} vs SciPy {scipy_status} (kernel statuses {:?})",
                    case.statuses
                ));
            }
            // The ledger rejects a length mismatch and a NaN coordinate (recording the case),
            // which the max fold below would otherwise read as agreement.
            let x_pair = ledger.slices("lbfgsb", case.id, Some(scipy_x), Some(r.x.as_slice()));
            let dx = x_pair.map_or(f64::NAN, |(scipy_x, fsci_x)| {
                fsci_x
                    .iter()
                    .zip(scipy_x)
                    .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
                    .fold(0.0, f64::max)
            });
            // NaN in any measure fails the case.
            if !(dx <= case.x_tol) {
                reasons.push(format!("x rel diff {dx:e} > {:e}", case.x_tol));
            }
            let fun = r.fun.unwrap_or(f64::NAN);
            let dfun = (fun - scipy_fun).abs();
            if !(dfun <= case.fun_tol * scipy_fun.abs().max(1.0)) {
                reasons.push(format!("fun {fun:e} vs SciPy {scipy_fun:e}"));
            }
            if case.invariant {
                if fsci_counts != scipy_counts {
                    reasons.push(format!(
                        "(nit, nfev, njev) {fsci_counts:?} vs SciPy {scipy_counts:?}"
                    ));
                }
            } else {
                let slack = (3 * scipy_counts.0).div_ceil(10);
                let (lo, hi) = case.nit_range;
                let in_range =
                    (lo as f64) * 0.7 <= r.nit as f64 && r.nit as f64 <= (hi as f64) * 1.3;
                if r.nit.abs_diff(scipy_counts.0) > slack && !in_range {
                    reasons.push(format!("nit {} vs SciPy {}", r.nit, scipy_counts.0));
                }
            }
            if reasons.is_empty() && fsci_counts == scipy_counts {
                exact_paths += 1;
            }
            println!(
                "{}: fsci {:?} {fsci_counts:?} fun={fun:e} | scipy {scipy_status} \
                     {scipy_counts:?} fun={scipy_fun:e} | x rel {dx:e}",
                case.id, r.status
            );
            if x_pair.is_some() {
                ledger.compared("lbfgsb", case.id, reasons.is_empty());
            }
        }
        diffs.push(CaseDiff {
            case_id: case.id.to_string(),
            pass: reasons.is_empty(),
            note: reasons.join("; "),
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    emit_log(&DiffLog {
        test_id: "diff_opt_lbfgsb_scipy_path".into(),
        category: "scipy.optimize.minimize(method='L-BFGS-B') path, live".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    });
    println!(
        "{} cases compared, {exact_paths} on SciPy's exact (nit, nfev, njev)",
        diffs.len()
    );
    assert_eq!(diffs.len(), 18, "every case must be compared");
    for d in &diffs {
        assert!(d.pass, "{}: {}", d.case_id, d.note);
    }
    ledger.finish(cases.len());
}

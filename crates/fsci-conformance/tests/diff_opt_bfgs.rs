#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `minimize(method='BFGS')` and `minimize(method='CG')`
//! (frankenscipy-1ksfv.18): SciPy's `_minimize_bfgs` and `_minimize_cg` with their MINPACK-2
//! `dcsrch` line search, on Rosenbrock in 2, 3 and 10 dimensions, a 50-dimensional convex
//! quadratic and a function that is indefinite at the start — each with the analytic gradient
//! and with SciPy's default forward difference (`eps = √ε`) — plus an iteration-limited run.
//!
//! Both sides evaluate the same function: the objectives are written out operation by operation
//! in Rust and in the oracle's Python (sequential sums, explicit products), because SciPy's own
//! `rosen` sums with numpy's pairwise reduction and a forward difference amplifies that last-ulp
//! difference in f by 1/h ≈ 7e7 — enough to move a 10-dimensional run from success to precision
//! loss.
//!
//! Per case the status must match SciPy's and the iteration count must be within `NIT_REL_TOL` of
//! it. With the analytic gradient the path is SciPy's to the evaluation (the log counts identical
//! (nit, nfev, njev)), so `x` must agree to `X_REL_TOL` and `fun` to `FUN_REL_TOL`. With finite
//! differences SciPy's own path depends on its BLAS kernel: under `OPENBLAS_CORETYPE` =
//! Prescott / Nehalem / Sandybridge / Haswell / Zen, SciPy 1.17.1 itself takes 250, 278 and 258
//! evaluations on rosen3 and 72 to 75 iterations on rosen10, because the inverse-Hessian products
//! round differently and the differences amplify it. Both sides stop at the same gradient
//! tolerance (1e-5), so there `x` must agree to `FD_X_REL_TOL` (relative to max(|x|, 1)) and `fun`
//! to `FD_FUN_ABS_TOL` (across those five kernels fsci is within 4.4e-6 in x and 2.1e-11 in f).
//! One case's STATUS moves with the kernel: rosen10 by finite differences ends in precision loss
//! (2) under Prescott, Nehalem, Haswell and Zen but succeeds (0) under Sandybridge, so there fsci
//! must return one of SciPy's statuses rather than the one this runner's kernel happens to give.
//!
//! CG has no matrix product, but numpy's BLAS `dot` in its β and norms moves SciPy's own path
//! on one of those five kernels: rosen10 with the analytic gradient takes 190 iterations there
//! and 231 on the other four, 1.7e-5 apart in `x` (fsci matches the four exactly, and every
//! other CG case on all five). So CG cases must match SciPy's status and iteration count (within
//! `NIT_REL_TOL`), `x` to `CG_X_REL_TOL` and `fun` to `FD_FUN_ABS_TOL`; only BFGS with the analytic
//! gradient, whose path no kernel moves, must reproduce SciPy's (nit, nfev, njev) exactly.
//!
//! Every case must be compared: a SciPy failure or an fsci error is a FAILED case, not a skipped
//! one (frankenscipy-olv0j.1).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::{ConvergenceStatus, GradientFunc, MinimizeOptions, OptimizeMethod, minimize};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-003";
const X_REL_TOL: f64 = 1.0e-7;
const FUN_REL_TOL: f64 = 1.0e-10;
const FD_X_REL_TOL: f64 = 1.0e-5;
const FD_FUN_ABS_TOL: f64 = 1.0e-9;
const CG_X_REL_TOL: f64 = 5.0e-5;
const NIT_REL_TOL: f64 = 0.3;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const QUAD_N: usize = 50;

fn rosen(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() - 1 {
        let a = x[i + 1] - x[i] * x[i];
        let b = 1.0 - x[i];
        s += 100.0 * (a * a) + b * b;
    }
    s
}

fn rosen_der(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut d = vec![0.0; n];
    for i in 1..n - 1 {
        d[i] = 200.0 * (x[i] - x[i - 1] * x[i - 1])
            - 400.0 * (x[i + 1] - x[i] * x[i]) * x[i]
            - 2.0 * (1.0 - x[i]);
    }
    d[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
    d[n - 1] = 200.0 * (x[n - 1] - x[n - 2] * x[n - 2]);
    d
}

/// `A = M Mᵀ / n + I` (positive definite) and `b`, from closed-form entries; SciPy receives them
/// through the query.
fn quad_data() -> &'static (Vec<Vec<f64>>, Vec<f64>) {
    static DATA: OnceLock<(Vec<Vec<f64>>, Vec<f64>)> = OnceLock::new();
    DATA.get_or_init(|| {
        let n = QUAD_N;
        let m: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| ((7 * i + 3 * j + 1) as f64).sin()).collect())
            .collect();
        let a: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let mut mm = 0.0;
                        for k in 0..n {
                            mm += m[i][k] * m[j][k];
                        }
                        mm / n as f64 + if i == j { 1.0 } else { 0.0 }
                    })
                    .collect()
            })
            .collect();
        let b = (0..n).map(|i| (i as f64 * 0.37).cos()).collect();
        (a, b)
    })
}

fn quad(x: &[f64]) -> f64 {
    let (a, b) = quad_data();
    let mut xax = 0.0;
    let mut bx = 0.0;
    for i in 0..x.len() {
        let mut ax = 0.0;
        for j in 0..x.len() {
            ax += a[i][j] * x[j];
        }
        xax += x[i] * ax;
        bx += b[i] * x[i];
    }
    0.5 * xax - bx
}

fn quad_der(x: &[f64]) -> Vec<f64> {
    let (a, b) = quad_data();
    (0..x.len())
        .map(|i| {
            let mut ax = 0.0;
            for j in 0..x.len() {
                ax += a[i][j] * x[j];
            }
            ax - b[i]
        })
        .collect()
}

fn indefinite(x: &[f64]) -> f64 {
    let x0 = x[0] * x[0];
    let x2 = x[2] * x[2];
    0.25 * (x0 * x0) - 0.5 * x0 + x[1] * x[1] + 0.1 * x[0] * x[1] + 0.05 * (x2 * x2) - x2
}

fn indefinite_der(x: &[f64]) -> Vec<f64> {
    vec![
        x[0] * x[0] * x[0] - x[0] + 0.1 * x[1],
        2.0 * x[1] + 0.1 * x[0],
        0.2 * (x[2] * x[2] * x[2]) - 2.0 * x[2],
    ]
}

struct Problem {
    name: &'static str,
    fun: fn(&[f64]) -> f64,
    grad: GradientFunc,
    x0: Vec<f64>,
}

struct Case {
    id: String,
    method: OptimizeMethod,
    /// SciPy's method name.
    scipy_method: &'static str,
    problem: usize,
    analytic: bool,
    maxiter: Option<usize>,
    /// SciPy statuses measured across OpenBLAS kernels when they differ; empty means the
    /// status is kernel-independent and must equal this runner's SciPy status.
    kernel_statuses: &'static [i32],
}

fn problems() -> Vec<Problem> {
    let rosen_problem = |name, x0: Vec<f64>| Problem {
        name,
        fun: rosen,
        grad: rosen_der,
        x0,
    };
    vec![
        rosen_problem("rosen2", vec![-1.2, 1.0]),
        rosen_problem("rosen3", vec![1.3, 0.7, 0.8]),
        rosen_problem(
            "rosen10",
            (0..10)
                .map(|i| if i % 2 == 0 { -1.5 } else { 1.7 })
                .collect(),
        ),
        Problem {
            name: "quad50",
            fun: quad,
            grad: quad_der,
            x0: vec![0.0; QUAD_N],
        },
        Problem {
            name: "indefinite_start",
            fun: indefinite,
            grad: indefinite_der,
            x0: vec![0.1, 0.2, 0.3],
        },
    ]
}

fn cases(problems: &[Problem]) -> Vec<Case> {
    let mut out = Vec::new();
    for (method, scipy_method) in [
        (OptimizeMethod::Bfgs, "BFGS"),
        (OptimizeMethod::ConjugateGradient, "CG"),
    ] {
        for (index, problem) in problems.iter().enumerate() {
            for analytic in [true, false] {
                out.push(Case {
                    id: format!(
                        "{scipy_method}/{}/{}",
                        problem.name,
                        if analytic { "jac" } else { "fd" }
                    ),
                    method,
                    scipy_method,
                    problem: index,
                    analytic,
                    maxiter: None,
                    kernel_statuses: if method == OptimizeMethod::Bfgs
                        && problem.name == "rosen10"
                        && !analytic
                    {
                        &[0, 2]
                    } else {
                        &[]
                    },
                });
            }
        }
        out.push(Case {
            id: format!("{scipy_method}/rosen2/jac/maxiter3"),
            method,
            scipy_method,
            problem: 0,
            analytic: true,
            maxiter: Some(3),
            kernel_statuses: &[],
        });
    }
    out
}

#[derive(Debug, Clone, Serialize)]
struct QueryCase {
    case_id: String,
    method: String,
    problem: String,
    analytic: bool,
    maxiter: Option<usize>,
    x0: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct Query {
    quad_a: Vec<Vec<f64>>,
    quad_b: Vec<f64>,
    cases: Vec<QueryCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleArm {
    case_id: String,
    x: Option<Vec<f64>>,
    fun: Option<f64>,
    status: Option<i32>,
    nit: Option<usize>,
    nfev: Option<usize>,
    njev: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    fsci_status: String,
    scipy_status: i32,
    fsci_counts: [usize; 3],
    scipy_counts: [usize; 3],
    fsci_fun: f64,
    scipy_fun: f64,
    max_x_rel: f64,
    pass: bool,
    reason: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    same_path_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn scipy_oracle_or_skip(query: &Query) -> Option<Vec<OracleArm>> {
    let script = r#"
import json, sys, warnings
import numpy as np
from scipy.optimize import minimize

q = json.load(sys.stdin)
A = [[float(v) for v in row] for row in q["quad_a"]]
B = [float(v) for v in q["quad_b"]]

# The same operations, in the same order, as the Rust side.
def rosen(x):
    x = [float(v) for v in x]
    s = 0.0
    for i in range(len(x) - 1):
        a = x[i + 1] - x[i] * x[i]
        b = 1.0 - x[i]
        s += 100.0 * (a * a) + b * b
    return s

def rosen_der(x):
    x = [float(v) for v in x]
    n = len(x)
    d = [0.0] * n
    for i in range(1, n - 1):
        d[i] = (200.0 * (x[i] - x[i - 1] * x[i - 1])
                - 400.0 * (x[i + 1] - x[i] * x[i]) * x[i]
                - 2.0 * (1.0 - x[i]))
    d[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0])
    d[n - 1] = 200.0 * (x[n - 1] - x[n - 2] * x[n - 2])
    return np.array(d)

def quad(x):
    x = [float(v) for v in x]
    xax = 0.0
    bx = 0.0
    for i in range(len(x)):
        ax = 0.0
        for j in range(len(x)):
            ax += A[i][j] * x[j]
        xax += x[i] * ax
        bx += B[i] * x[i]
    return 0.5 * xax - bx

def quad_der(x):
    x = [float(v) for v in x]
    out = []
    for i in range(len(x)):
        ax = 0.0
        for j in range(len(x)):
            ax += A[i][j] * x[j]
        out.append(ax - B[i])
    return np.array(out)

def indefinite(x):
    x = [float(v) for v in x]
    x0 = x[0] * x[0]
    x2 = x[2] * x[2]
    return 0.25 * (x0 * x0) - 0.5 * x0 + x[1] * x[1] + 0.1 * x[0] * x[1] + 0.05 * (x2 * x2) - x2

def indefinite_der(x):
    x = [float(v) for v in x]
    return np.array([x[0] * x[0] * x[0] - x[0] + 0.1 * x[1],
                     2.0 * x[1] + 0.1 * x[0],
                     0.2 * (x[2] * x[2] * x[2]) - 2.0 * x[2]])

PROBLEMS = {"rosen": (rosen, rosen_der), "quad50": (quad, quad_der),
            "indefinite_start": (indefinite, indefinite_der)}

out = []
for case in q["cases"]:
    name = case["problem"]
    f, g = PROBLEMS["rosen" if name.startswith("rosen") else name]
    arm = {"case_id": case["case_id"], "x": None, "fun": None, "status": None,
           "nit": None, "nfev": None, "njev": None}
    kw = {"jac": g} if case["analytic"] else {}
    options = {} if case["maxiter"] is None else {"maxiter": case["maxiter"]}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = minimize(f, np.array(case["x0"], dtype=float), method=case["method"],
                         options=options, **kw)
        arm.update(x=[float(v) for v in r.x], fun=float(r.fun), status=int(r.status),
                   nit=int(r.nit), nfev=int(r.nfev), njev=int(r.njev))
    except Exception:
        pass
    out.append(arm)
print(json.dumps(out, allow_nan=False))
"#;
    let query_json = serde_json::to_string(query).expect("serialize BFGS query");
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
                "failed to spawn python3 for the BFGS oracle: {e}"
            );
            eprintln!("skipping BFGS oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open BFGS oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "BFGS oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping BFGS oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for BFGS oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "BFGS oracle failed: {stderr}"
        );
        eprintln!("skipping BFGS oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse BFGS oracle JSON"))
}

/// fsci's status in SciPy's BFGS / CG numbering (0 success, 1 maxiter, 2 precision loss, 3 NaN).
fn scipy_status_of(status: ConvergenceStatus) -> Option<i32> {
    match status {
        ConvergenceStatus::Success => Some(0),
        ConvergenceStatus::MaxIterations => Some(1),
        ConvergenceStatus::PrecisionLoss => Some(2),
        ConvergenceStatus::NanEncountered => Some(3),
        _ => None,
    }
}

#[test]
fn diff_opt_bfgs() {
    let problems = problems();
    let cases = cases(&problems);
    let (quad_a, quad_b) = quad_data().clone();
    let query = Query {
        quad_a,
        quad_b,
        cases: cases
            .iter()
            .map(|c| QueryCase {
                case_id: c.id.clone(),
                method: c.scipy_method.to_string(),
                problem: problems[c.problem].name.to_string(),
                analytic: c.analytic,
                maxiter: c.maxiter,
                x0: problems[c.problem].x0.clone(),
            })
            .collect(),
    };
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    let arms: HashMap<String, OracleArm> = oracle
        .into_iter()
        .map(|arm| (arm.case_id.clone(), arm))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    for case in &cases {
        let problem = &problems[case.problem];
        let arm = &arms[&case.id];
        let options = MinimizeOptions {
            method: Some(case.method),
            gradient: case.analytic.then_some(problem.grad),
            maxiter: case.maxiter,
            ..MinimizeOptions::default()
        };
        let fsci = minimize(problem.fun, &problem.x0, options);
        let scipy_counts = [
            arm.nit.unwrap_or(0),
            arm.nfev.unwrap_or(0),
            arm.njev.unwrap_or(0),
        ];
        let mut diff = CaseDiff {
            case_id: case.id.clone(),
            fsci_status: String::new(),
            scipy_status: arm.status.unwrap_or(-1),
            fsci_counts: [0; 3],
            scipy_counts,
            fsci_fun: f64::NAN,
            scipy_fun: arm.fun.unwrap_or(f64::NAN),
            max_x_rel: f64::NAN,
            pass: false,
            reason: String::new(),
        };
        match (fsci, arm.status, &arm.x) {
            (Err(e), _, _) => diff.reason = format!("fsci error {e}"),
            (Ok(_), None, _) | (Ok(_), _, None) => {
                diff.reason = "SciPy produced no result".to_string();
            }
            (Ok(r), Some(status), Some(scipy_x)) => {
                diff.fsci_status = format!("{:?}", r.status);
                diff.fsci_counts = [r.nit, r.nfev, r.njev];
                diff.fsci_fun = r.fun.unwrap_or(f64::NAN);
                let mut problems_found = Vec::new();
                let fsci_status = scipy_status_of(r.status);
                let status_ok = if case.kernel_statuses.is_empty() {
                    fsci_status == Some(status)
                } else {
                    fsci_status.is_some_and(|s| case.kernel_statuses.contains(&s))
                };
                if !status_ok {
                    problems_found.push(format!(
                        "status {:?} vs SciPy {status} (kernel statuses {:?})",
                        r.status, case.kernel_statuses
                    ));
                }
                let dx =
                    r.x.iter()
                        .zip(scipy_x)
                        .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
                        .fold(0.0, f64::max);
                diff.max_x_rel = dx;
                // BFGS with the analytic gradient is the one kernel-invariant path.
                let exact_path = case.method == OptimizeMethod::Bfgs && case.analytic;
                let x_tol = if case.method == OptimizeMethod::ConjugateGradient {
                    CG_X_REL_TOL
                } else if exact_path {
                    X_REL_TOL
                } else {
                    FD_X_REL_TOL
                };
                // NaN in any measure fails the case.
                if r.x.len() != scipy_x.len() || dx.is_nan() || dx > x_tol {
                    problems_found.push(format!("x rel diff {dx:e}"));
                }
                let dfun = (diff.fsci_fun - diff.scipy_fun).abs();
                let fun_ok = if exact_path {
                    dfun <= FUN_REL_TOL * diff.scipy_fun.abs().max(1.0)
                } else {
                    dfun <= FD_FUN_ABS_TOL
                };
                if !fun_ok {
                    problems_found.push(format!("fun diff {dfun:e}"));
                }
                let nit_slack = (NIT_REL_TOL * scipy_counts[0] as f64).ceil() as usize;
                if r.nit.abs_diff(scipy_counts[0]) > nit_slack {
                    problems_found.push(format!("nit {} vs SciPy {}", r.nit, scipy_counts[0]));
                }
                diff.pass = problems_found.is_empty();
                diff.reason = problems_found.join("; ");
            }
        }
        diffs.push(diff);
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let same_path_count = diffs
        .iter()
        .filter(|d| d.pass && d.fsci_counts == d.scipy_counts)
        .count();
    let log = DiffLog {
        test_id: "diff_opt_bfgs".into(),
        category: "scipy.optimize.minimize(method='BFGS' | 'CG')".into(),
        case_count: diffs.len(),
        same_path_count,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    let dir = output_dir();
    fs::create_dir_all(&dir).expect("create diff output dir");
    fs::write(
        dir.join("diff_opt_bfgs.json"),
        serde_json::to_string_pretty(&log).expect("serialize diff log"),
    )
    .expect("write diff log");
    for d in &diffs {
        println!(
            "{} fsci {} (nit, nfev, njev)={:?} fun={:e} | scipy status {} {:?} fun={:e} \
             | x rel {:e} {}",
            d.case_id,
            d.fsci_status,
            d.fsci_counts,
            d.fsci_fun,
            d.scipy_status,
            d.scipy_counts,
            d.scipy_fun,
            d.max_x_rel,
            d.reason
        );
    }
    println!(
        "{} cases compared, {same_path_count} with SciPy's exact (nit, nfev, njev)",
        diffs.len()
    );

    assert_eq!(diffs.len(), 22, "every case must be compared");
    for d in &diffs {
        assert!(d.pass, "{}: {}", d.case_id, d.reason);
    }
    // BFGS's analytic-gradient paths are SciPy's to the evaluation.
    for d in diffs
        .iter()
        .filter(|d| d.case_id.starts_with("BFGS/") && d.case_id.contains("/jac"))
    {
        assert_eq!(
            d.fsci_counts, d.scipy_counts,
            "{}: (nit, nfev, njev) must be SciPy's",
            d.case_id
        );
    }
}

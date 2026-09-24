#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the trust-region Newton family (frankenscipy-1ksfv.4):
//! `minimize(method='trust-ncg' | 'dogleg' | 'trust-exact')` with an analytic gradient and
//! Hessian (and trust-ncg also with a Hessian-vector product), on Rosenbrock in 2, 3 and 10
//! dimensions, a 50-dimensional convex quadratic, and a function whose Hessian is indefinite at
//! the start, where trust-ncg and trust-exact must use negative curvature and dogleg must stop
//! the way SciPy does (status 3, "A linalg error occurred, such as a non-psd Hessian").
//!
//! Per case the status must match SciPy's. When SciPy succeeded, `x` must agree to
//! `X_REL_TOL` (relative to max(|x|, 1)), `fun` to `FUN_REL_TOL` (relative to max(|f|, 1)) and
//! the iteration count to within `NIT_REL_TOL` of SciPy's. The port follows SciPy's driver and
//! subproblems statement by statement, so the log also counts cases whose
//! (nit, nfev, njev, nhev) are identical to SciPy's.
//!
//! Every case must be compared: a SciPy failure or an fsci error is a FAILED case, not a
//! skipped one (frankenscipy-olv0j.1).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::{
    ConvergenceStatus, GradientFunc, HessFunc, HesspFunc, MinimizeOptions, OptimizeMethod, minimize,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-003";
const X_REL_TOL: f64 = 1.0e-7;
const FUN_REL_TOL: f64 = 1.0e-10;
const NIT_REL_TOL: f64 = 0.3;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const QUAD_N: usize = 50;

fn rosen(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
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

fn rosen_hess(x: &[f64]) -> Vec<Vec<f64>> {
    let n = x.len();
    let mut h = vec![vec![0.0; n]; n];
    for i in 0..n - 1 {
        h[i][i + 1] = -400.0 * x[i];
        h[i + 1][i] = -400.0 * x[i];
    }
    h[0][0] = 1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0;
    h[n - 1][n - 1] = 200.0;
    for i in 1..n - 1 {
        h[i][i] = 202.0 + 1200.0 * x[i] * x[i] - 400.0 * x[i + 1];
    }
    h
}

fn rosen_hess_prod(x: &[f64], p: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut hp = vec![0.0; n];
    hp[0] = (1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0) * p[0] - 400.0 * x[0] * p[1];
    for i in 1..n - 1 {
        hp[i] = -400.0 * x[i - 1] * p[i - 1]
            + (202.0 + 1200.0 * x[i] * x[i] - 400.0 * x[i + 1]) * p[i]
            - 400.0 * x[i] * p[i + 1];
    }
    hp[n - 1] = -400.0 * x[n - 2] * p[n - 2] + 200.0 * p[n - 1];
    hp
}

/// `A = M Mᵀ / n + I` (positive definite) and `b`, from closed-form entries so both sides
/// read the same numbers; SciPy receives them through the query.
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
                        let mm: f64 = (0..n).map(|k| m[i][k] * m[j][k]).sum();
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
    let ax: Vec<f64> = a
        .iter()
        .map(|row| row.iter().zip(x).map(|(r, v)| r * v).sum())
        .collect();
    0.5 * x.iter().zip(&ax).map(|(v, w)| v * w).sum::<f64>()
        - b.iter().zip(x).map(|(bi, v)| bi * v).sum::<f64>()
}

fn quad_der(x: &[f64]) -> Vec<f64> {
    let (a, b) = quad_data();
    a.iter()
        .zip(b)
        .map(|(row, bi)| row.iter().zip(x).map(|(r, v)| r * v).sum::<f64>() - bi)
        .collect()
}

fn quad_hess(_x: &[f64]) -> Vec<Vec<f64>> {
    quad_data().0.clone()
}

fn quad_hess_prod(_x: &[f64], p: &[f64]) -> Vec<f64> {
    quad_data()
        .0
        .iter()
        .map(|row| row.iter().zip(p).map(|(r, v)| r * v).sum())
        .collect()
}

fn indefinite(x: &[f64]) -> f64 {
    0.25 * x[0].powi(4) - 0.5 * x[0] * x[0] + x[1] * x[1] + 0.1 * x[0] * x[1] + 0.05 * x[2].powi(4)
        - x[2] * x[2]
}

fn indefinite_der(x: &[f64]) -> Vec<f64> {
    vec![
        x[0].powi(3) - x[0] + 0.1 * x[1],
        2.0 * x[1] + 0.1 * x[0],
        0.2 * x[2].powi(3) - 2.0 * x[2],
    ]
}

fn indefinite_hess(x: &[f64]) -> Vec<Vec<f64>> {
    vec![
        vec![3.0 * x[0] * x[0] - 1.0, 0.1, 0.0],
        vec![0.1, 2.0, 0.0],
        vec![0.0, 0.0, 0.6 * x[2] * x[2] - 2.0],
    ]
}

fn indefinite_hess_prod(x: &[f64], p: &[f64]) -> Vec<f64> {
    indefinite_hess(x)
        .iter()
        .map(|row| row.iter().zip(p).map(|(r, v)| r * v).sum())
        .collect()
}

struct Problem {
    name: &'static str,
    fun: fn(&[f64]) -> f64,
    grad: GradientFunc,
    hess: HessFunc,
    hessp: HesspFunc,
    x0: Vec<f64>,
}

struct Case {
    id: String,
    problem: usize,
    method: OptimizeMethod,
    scipy_method: &'static str,
    use_hessp: bool,
}

fn problems() -> Vec<Problem> {
    let rosen_problem = |name, x0: Vec<f64>| Problem {
        name,
        fun: rosen,
        grad: rosen_der,
        hess: rosen_hess,
        hessp: rosen_hess_prod,
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
            hess: quad_hess,
            hessp: quad_hess_prod,
            x0: vec![0.0; QUAD_N],
        },
        Problem {
            name: "indefinite_start",
            fun: indefinite,
            grad: indefinite_der,
            hess: indefinite_hess,
            hessp: indefinite_hess_prod,
            x0: vec![0.1, 0.2, 0.3],
        },
    ]
}

fn cases(problems: &[Problem]) -> Vec<Case> {
    let mut out = Vec::new();
    for (index, problem) in problems.iter().enumerate() {
        for (method, scipy_method, use_hessp) in [
            (OptimizeMethod::TrustNcg, "trust-ncg", false),
            (OptimizeMethod::TrustNcg, "trust-ncg", true),
            (OptimizeMethod::Dogleg, "dogleg", false),
            (OptimizeMethod::TrustExact, "trust-exact", false),
        ] {
            out.push(Case {
                id: format!(
                    "{}/{scipy_method}/{}",
                    problem.name,
                    if use_hessp { "hessp" } else { "hess" }
                ),
                problem: index,
                method,
                scipy_method,
                use_hessp,
            });
        }
    }
    out
}

#[derive(Debug, Clone, Serialize)]
struct QueryCase {
    case_id: String,
    problem: String,
    method: String,
    use_hessp: bool,
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
    nhev: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    fsci_status: String,
    scipy_status: i32,
    fsci_counts: [usize; 4],
    scipy_counts: [usize; 4],
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
from scipy.optimize import minimize, rosen, rosen_der, rosen_hess, rosen_hess_prod

q = json.load(sys.stdin)
A = np.array(q["quad_a"]); b = np.array(q["quad_b"])

def ind(x): return 0.25*x[0]**4 - 0.5*x[0]**2 + x[1]**2 + 0.1*x[0]*x[1] + 0.05*x[2]**4 - x[2]**2
def ind_g(x): return np.array([x[0]**3 - x[0] + 0.1*x[1], 2*x[1] + 0.1*x[0], 0.2*x[2]**3 - 2*x[2]])
def ind_h(x): return np.array([[3*x[0]**2 - 1, 0.1, 0], [0.1, 2, 0], [0, 0, 0.6*x[2]**2 - 2]])

PROBLEMS = {
    "rosen": (rosen, rosen_der, rosen_hess, rosen_hess_prod),
    "quad50": (lambda x: 0.5*x@A@x - b@x, lambda x: A@x - b, lambda x: A, lambda x, p: A@p),
    "indefinite_start": (ind, ind_g, ind_h, lambda x, p: ind_h(x)@p),
}

out = []
for case in q["cases"]:
    name = case["problem"]
    f, g, h, hp = PROBLEMS["rosen" if name.startswith("rosen") else name]
    arm = {"case_id": case["case_id"], "x": None, "fun": None, "status": None,
           "nit": None, "nfev": None, "njev": None, "nhev": None}
    kw = {"jac": g, "hessp": hp} if case["use_hessp"] else {"jac": g, "hess": h}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = minimize(f, np.array(case["x0"], dtype=float), method=case["method"], **kw)
        arm.update(x=[float(v) for v in r.x], fun=float(r.fun), status=int(r.status),
                   nit=int(r.nit), nfev=int(r.nfev), njev=int(r.njev), nhev=int(r.nhev))
    except Exception:
        pass
    out.append(arm)
print(json.dumps(out))
"#;
    let query_json = serde_json::to_string(query).expect("serialize trust-region query");
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
                "failed to spawn python3 for the trust-region oracle: {e}"
            );
            eprintln!("skipping trust-region oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open trust-region oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "trust-region oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping trust-region oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for trust-region oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "trust-region oracle failed: {stderr}"
        );
        eprintln!("skipping trust-region oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse trust-region oracle JSON"))
}

/// fsci's status in SciPy's trust-region numbering (0 success, 1 maxiter, 2 no predicted
/// improvement, 3 linalg error).
fn scipy_status_of(status: ConvergenceStatus) -> Option<i32> {
    match status {
        ConvergenceStatus::Success => Some(0),
        ConvergenceStatus::MaxIterations => Some(1),
        ConvergenceStatus::PrecisionLoss => Some(2),
        ConvergenceStatus::LinAlgError => Some(3),
        _ => None,
    }
}

#[test]
fn diff_opt_trust_region_family() {
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
                problem: problems[c.problem].name.to_string(),
                method: c.scipy_method.to_string(),
                use_hessp: c.use_hessp,
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
            gradient: Some(problem.grad),
            hess: (!case.use_hessp).then_some(problem.hess),
            hessp: case.use_hessp.then_some(problem.hessp),
            ..MinimizeOptions::default()
        };
        let fsci = minimize(problem.fun, &problem.x0, options);
        let scipy_counts = [
            arm.nit.unwrap_or(0),
            arm.nfev.unwrap_or(0),
            arm.njev.unwrap_or(0),
            arm.nhev.unwrap_or(0),
        ];
        let mut diff = CaseDiff {
            case_id: case.id.clone(),
            fsci_status: String::new(),
            scipy_status: arm.status.unwrap_or(-1),
            fsci_counts: [0; 4],
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
                diff.fsci_counts = [r.nit, r.nfev, r.njev, r.nhev];
                diff.fsci_fun = r.fun.unwrap_or(f64::NAN);
                let mut problems_found = Vec::new();
                if scipy_status_of(r.status) != Some(status) {
                    problems_found.push(format!("status {:?} vs SciPy {status}", r.status));
                }
                let dx =
                    r.x.iter()
                        .zip(scipy_x)
                        .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
                        .fold(0.0, f64::max);
                diff.max_x_rel = dx;
                let dfun = (diff.fsci_fun - diff.scipy_fun).abs() / diff.scipy_fun.abs().max(1.0);
                // NaN in any measure fails the case.
                if r.x.len() != scipy_x.len() || dx.is_nan() || dx > X_REL_TOL {
                    problems_found.push(format!("x rel diff {dx:e}"));
                }
                if dfun.is_nan() || dfun > FUN_REL_TOL {
                    problems_found.push(format!("fun rel diff {dfun:e}"));
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
        test_id: "diff_opt_trust_region_family".into(),
        category: "scipy.optimize.minimize(method='trust-ncg' | 'dogleg' | 'trust-exact')".into(),
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
        dir.join("diff_opt_trust_region_family.json"),
        serde_json::to_string_pretty(&log).expect("serialize diff log"),
    )
    .expect("write diff log");
    for d in &diffs {
        println!(
            "{} fsci {} (nit, nfev, njev, nhev)={:?} fun={:e} | scipy status {} {:?} fun={:e} \
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
        "{} cases compared, {same_path_count} with SciPy's exact (nit, nfev, njev, nhev)",
        diffs.len()
    );

    assert_eq!(diffs.len(), 20, "every case must be compared");
    for d in &diffs {
        assert!(d.pass, "{}: {}", d.case_id, d.reason);
    }
}

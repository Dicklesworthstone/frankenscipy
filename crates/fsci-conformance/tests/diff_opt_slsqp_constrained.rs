#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `minimize(method='SLSQP')` with constraints and bounds
//! (frankenscipy-1ksfv.1): Hock–Schittkowski #6, #7, #21 (started infeasible), #35, #71, #76,
//! equality-only, bounds-only, mixed equality/inequality/bounds, a vector-valued inequality,
//! incompatible constraints, an iteration limit, the unconstrained case, and SciPy's
//! `new_constraint_to_old` conversions of `LinearConstraint` / `NonlinearConstraint`.
//!
//! Both sides use their default forward-difference step, SciPy's `eps = √ε` (fsci:
//! `gradient_eps: None`). Per case the exit message (SciPy's exit mode) must
//! match. When SciPy succeeded, `x` must agree to `X_REL_TOL` (relative to max(|x|, 1)), `fun`
//! to `FUN_REL_TOL`, and fsci's constraint violation must be at most `MAXCV_TOL` — SLSQP's
//! own success contract (Σ violation < `ftol` = 1e-6); SciPy itself ends HS71 at a violation
//! of 8.226e-8, so a tighter bound would fail the incumbent. The port
//! follows SciPy's C translation statement by statement, so most cases also take the same
//! number of iterations; the log counts them.
//!
//! Every case must be compared: a SciPy failure or an fsci error is a FAILED case, not a
//! skipped one (frankenscipy-olv0j.1).

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_opt::{
    Bound, Constraint, LinearConstraint, MinimizeOptions, NonlinearConstraint, OptimizeMethod,
    minimize,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-003";
const X_REL_TOL: f64 = 1.0e-6;
const FUN_REL_TOL: f64 = 1.0e-8;
const MAXCV_TOL: f64 = 1.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

type Objective = fn(&[f64]) -> f64;

struct Case {
    id: &'static str,
    fun: Objective,
    x0: Vec<f64>,
    bounds: Option<Vec<Bound>>,
    maxiter: usize,
}

fn rosen(v: &[f64]) -> f64 {
    (1.0 - v[0]).powi(2) + 100.0 * (v[1] - v[0] * v[0]).powi(2)
}

fn case(id: &'static str, fun: Objective, x0: &[f64], bounds: Option<Vec<Bound>>) -> Case {
    Case {
        id,
        fun,
        x0: x0.to_vec(),
        bounds,
        maxiter: 100,
    }
}

fn cases() -> Vec<Case> {
    let lower0 = |n| Some(vec![(Some(0.0), None); n]);
    let mut maxiter_1 = case("maxiter_1", rosen, &[-1.2, 1.0], None);
    maxiter_1.maxiter = 1;
    vec![
        case(
            "ineq_quadratic",
            |v| (v[0] - 2.0).powi(2) + (v[1] - 1.0).powi(2),
            &[0.0, 0.0],
            None,
        ),
        case(
            "eq_bounds_3d",
            |v| v[0] * v[0] + v[1] * v[1] + v[2] * v[2],
            &[1.0, 1.0, 1.0],
            lower0(3),
        ),
        case("hs6", |v| (1.0 - v[0]).powi(2), &[-1.2, 1.0], None),
        case(
            "hs7",
            |v| (1.0 + v[0] * v[0]).ln() - v[1],
            &[2.0, 2.0],
            None,
        ),
        case(
            "hs21_infeasible_start",
            |v| 0.01 * v[0] * v[0] + v[1] * v[1] - 100.0,
            &[-1.0, -1.0],
            Some(vec![(Some(2.0), Some(50.0)), (Some(-50.0), Some(50.0))]),
        ),
        case(
            "hs35",
            |v| {
                9.0 - 8.0 * v[0] - 6.0 * v[1] - 4.0 * v[2]
                    + 2.0 * v[0] * v[0]
                    + 2.0 * v[1] * v[1]
                    + v[2] * v[2]
                    + 2.0 * v[0] * v[1]
                    + 2.0 * v[0] * v[2]
            },
            &[0.5, 0.5, 0.5],
            lower0(3),
        ),
        case(
            "hs71",
            |v| v[0] * v[3] * (v[0] + v[1] + v[2]) + v[2],
            &[1.0, 5.0, 5.0, 1.0],
            Some(vec![(Some(1.0), Some(5.0)); 4]),
        ),
        case(
            "hs76",
            |v| {
                v[0] * v[0] + 0.5 * v[1] * v[1] + v[2] * v[2] + 0.5 * v[3] * v[3] - v[0] * v[2]
                    + v[2] * v[3]
                    - v[0]
                    - 3.0 * v[1]
                    + v[2]
                    - v[3]
            },
            &[0.5; 4],
            lower0(4),
        ),
        case(
            "rosen_bounds_only",
            rosen,
            &[-1.0, 1.0],
            Some(vec![(Some(-2.0), Some(0.5)), (Some(-2.0), Some(2.0))]),
        ),
        case("eq_only", |v| v[0] * v[0] + v[1] * v[1], &[2.0, -1.0], None),
        case(
            "mixed_eq_ineq_bounds",
            |v| (v[0] - 1.0).powi(2) + (v[1] - 2.0).powi(2) + (v[2] - 3.0).powi(2),
            &[0.0, 0.0, 0.0],
            Some(vec![(None, None), (None, None), (Some(0.0), Some(1.5))]),
        ),
        case("vector_ineq", |v| -(v[0] * v[1]), &[0.5, 0.5], None),
        case("infeasible", |v| v[0] * v[0], &[0.5], None),
        maxiter_1,
        case("unconstrained_rosen", rosen, &[-1.2, 1.0], None),
        case("rosen_ineq_disc", rosen, &[0.0, 0.0], None),
        case(
            "linear_eq",
            |v| v[0] * v[0] + v[1] * v[1],
            &[2.0, -1.0],
            None,
        ),
        case(
            "linear_box",
            |v| (v[0] - 1.0).powi(2) + (v[1] - 1.0).powi(2),
            &[0.0, 0.0],
            None,
        ),
        case("nonlinear_disc", |v| -(v[0] * v[1]), &[0.5, 0.5], None),
    ]
}

fn disc(v: &[f64]) -> Vec<f64> {
    vec![v[0] * v[0] + v[1] * v[1]]
}

/// The constraints of case `id`, built the way a user would write them.
fn constraints_of<'a>(
    id: &str,
    linear: &'a HashMap<&'static str, LinearConstraint>,
    nonlinear: &'a NonlinearConstraint,
) -> Vec<Constraint<'a>> {
    match id {
        "ineq_quadratic" => vec![Constraint::ineq(|v: &[f64]| vec![1.0 - v[0] - v[1]])],
        "eq_bounds_3d" => vec![Constraint::eq(|v: &[f64]| {
            vec![v[0] + 2.0 * v[1] + 3.0 * v[2] - 6.0]
        })],
        "hs6" => vec![Constraint::eq(|v: &[f64]| {
            vec![10.0 * (v[1] - v[0] * v[0])]
        })],
        "hs7" => vec![Constraint::eq(|v: &[f64]| {
            vec![(1.0 + v[0] * v[0]).powi(2) + v[1] * v[1] - 4.0]
        })],
        "hs21_infeasible_start" => {
            vec![Constraint::ineq(|v: &[f64]| {
                vec![10.0 * v[0] - v[1] - 10.0]
            })]
        }
        "hs35" => vec![Constraint::ineq(|v: &[f64]| {
            vec![3.0 - v[0] - v[1] - 2.0 * v[2]]
        })],
        "hs71" => vec![
            Constraint::ineq(|v: &[f64]| vec![v[0] * v[1] * v[2] * v[3] - 25.0]),
            Constraint::eq(|v: &[f64]| vec![v.iter().map(|x| x * x).sum::<f64>() - 40.0]),
        ],
        "hs76" => vec![
            Constraint::ineq(|v: &[f64]| vec![5.0 - v[0] - 2.0 * v[1] - v[2] - v[3]]),
            Constraint::ineq(|v: &[f64]| vec![4.0 - 3.0 * v[0] - v[1] - 2.0 * v[2] + v[3]]),
            Constraint::ineq(|v: &[f64]| vec![v[1] + 4.0 * v[2] - 1.5]),
        ],
        "eq_only" => vec![Constraint::eq(|v: &[f64]| vec![v[0] + v[1] - 1.0])],
        "mixed_eq_ineq_bounds" => vec![
            Constraint::eq(|v: &[f64]| vec![v[0] + v[1] + v[2] - 3.0]),
            Constraint::ineq(|v: &[f64]| vec![v[0] - v[1]]),
        ],
        "vector_ineq" => vec![Constraint::ineq(|v: &[f64]| {
            vec![1.0 - v[0] * v[0] - v[1] * v[1], v[0], v[1]]
        })],
        "infeasible" => vec![
            Constraint::ineq(|v: &[f64]| vec![v[0] - 1.0]),
            Constraint::ineq(|v: &[f64]| vec![-v[0]]),
        ],
        "maxiter_1" => vec![Constraint::ineq(|v: &[f64]| {
            vec![2.0 - v[0] * v[0] - v[1] * v[1]]
        })],
        "rosen_ineq_disc" => vec![Constraint::ineq(|v: &[f64]| {
            vec![1.5 - v[0] * v[0] - v[1] * v[1]]
        })],
        "linear_eq" | "linear_box" => Constraint::from_linear(&linear[id]),
        "nonlinear_disc" => Constraint::from_nonlinear(nonlinear),
        _ => Vec::new(),
    }
}

#[derive(Debug, Clone, Serialize)]
struct QueryCase {
    case_id: String,
    x0: Vec<f64>,
    bounds: Option<Vec<(Option<f64>, Option<f64>)>>,
    maxiter: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleArm {
    case_id: String,
    x: Option<Vec<f64>>,
    fun: Option<f64>,
    status: Option<i32>,
    nit: Option<usize>,
    message: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    fsci_x: Vec<f64>,
    scipy_x: Vec<f64>,
    fsci_fun: f64,
    scipy_fun: f64,
    fsci_message: String,
    scipy_message: String,
    fsci_nit: usize,
    scipy_nit: usize,
    maxcv: Option<f64>,
    pass: bool,
    reason: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    same_iteration_count: usize,
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

fn scipy_oracle_or_skip(query: &[QueryCase]) -> Option<Vec<OracleArm>> {
    let script = r#"
import json, math, sys, warnings
import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

rosen = lambda v: (1 - v[0])**2 + 100*(v[1] - v[0]**2)**2
FUN = {
    "ineq_quadratic": lambda v: (v[0]-2)**2 + (v[1]-1)**2,
    "eq_bounds_3d": lambda v: v[0]**2 + v[1]**2 + v[2]**2,
    "hs6": lambda v: (1 - v[0])**2,
    "hs7": lambda v: math.log(1 + v[0]**2) - v[1],
    "hs21_infeasible_start": lambda v: 0.01*v[0]**2 + v[1]**2 - 100,
    "hs35": lambda v: 9 - 8*v[0] - 6*v[1] - 4*v[2] + 2*v[0]**2 + 2*v[1]**2 + v[2]**2
        + 2*v[0]*v[1] + 2*v[0]*v[2],
    "hs71": lambda v: v[0]*v[3]*(v[0]+v[1]+v[2]) + v[2],
    "hs76": lambda v: v[0]**2 + 0.5*v[1]**2 + v[2]**2 + 0.5*v[3]**2 - v[0]*v[2] + v[2]*v[3]
        - v[0] - 3*v[1] + v[2] - v[3],
    "rosen_bounds_only": rosen,
    "eq_only": lambda v: v[0]**2 + v[1]**2,
    "mixed_eq_ineq_bounds": lambda v: (v[0]-1)**2 + (v[1]-2)**2 + (v[2]-3)**2,
    "vector_ineq": lambda v: -(v[0]*v[1]),
    "infeasible": lambda v: v[0]**2,
    "maxiter_1": rosen,
    "unconstrained_rosen": rosen,
    "rosen_ineq_disc": rosen,
    "linear_eq": lambda v: v[0]**2 + v[1]**2,
    "linear_box": lambda v: (v[0]-1)**2 + (v[1]-1)**2,
    "nonlinear_disc": lambda v: -(v[0]*v[1]),
}
ineq = lambda f: {'type': 'ineq', 'fun': f}
eq = lambda f: {'type': 'eq', 'fun': f}
CONS = {
    "ineq_quadratic": [ineq(lambda v: 1 - v[0] - v[1])],
    "eq_bounds_3d": [eq(lambda v: v[0] + 2*v[1] + 3*v[2] - 6)],
    "hs6": [eq(lambda v: 10*(v[1] - v[0]**2))],
    "hs7": [eq(lambda v: (1 + v[0]**2)**2 + v[1]**2 - 4)],
    "hs21_infeasible_start": [ineq(lambda v: 10*v[0] - v[1] - 10)],
    "hs35": [ineq(lambda v: 3 - v[0] - v[1] - 2*v[2])],
    "hs71": [ineq(lambda v: v[0]*v[1]*v[2]*v[3] - 25),
             eq(lambda v: v[0]**2 + v[1]**2 + v[2]**2 + v[3]**2 - 40)],
    "hs76": [ineq(lambda v: 5 - v[0] - 2*v[1] - v[2] - v[3]),
             ineq(lambda v: 4 - 3*v[0] - v[1] - 2*v[2] + v[3]),
             ineq(lambda v: v[1] + 4*v[2] - 1.5)],
    "eq_only": [eq(lambda v: v[0] + v[1] - 1)],
    "mixed_eq_ineq_bounds": [eq(lambda v: v[0] + v[1] + v[2] - 3), ineq(lambda v: v[0] - v[1])],
    "vector_ineq": [ineq(lambda v: np.array([1 - v[0]**2 - v[1]**2, v[0], v[1]]))],
    "infeasible": [ineq(lambda v: v[0] - 1), ineq(lambda v: -v[0])],
    "maxiter_1": [ineq(lambda v: 2 - v[0]**2 - v[1]**2)],
    "rosen_ineq_disc": [ineq(lambda v: 1.5 - v[0]**2 - v[1]**2)],
    "linear_eq": [LinearConstraint([[1, 1]], 1, 1)],
    "linear_box": [LinearConstraint([[1, 0], [0, 1]], [0.5, -np.inf], [0.8, 0.3])],
    "nonlinear_disc": [NonlinearConstraint(lambda v: [v[0]**2 + v[1]**2], -np.inf, 1)],
}

out = []
for case in json.load(sys.stdin):
    cid = case["case_id"]
    arm = {"case_id": cid, "x": None, "fun": None, "status": None, "nit": None, "message": None}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = minimize(FUN[cid], case["x0"], method="SLSQP", constraints=CONS.get(cid, ()),
                         bounds=case["bounds"], options={"maxiter": case["maxiter"]})
        arm.update(x=[float(v) for v in r.x], fun=float(r.fun), status=int(r.status),
                   nit=int(r.nit), message=str(r.message))
    except Exception:
        pass
    out.append(arm)
print(json.dumps(out))
"#;
    let query_json = serde_json::to_string(query).expect("serialize slsqp query");
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
                "failed to spawn python3 for the slsqp oracle: {e}"
            );
            eprintln!("skipping slsqp oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open slsqp oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "slsqp oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping slsqp oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for slsqp oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "slsqp oracle failed: {stderr}"
        );
        eprintln!("skipping slsqp oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse slsqp oracle JSON"))
}

#[test]
fn diff_opt_slsqp_constrained() {
    let cases = cases();
    let query: Vec<QueryCase> = cases
        .iter()
        .map(|c| QueryCase {
            case_id: c.id.to_string(),
            x0: c.x0.clone(),
            bounds: c.bounds.clone(),
            maxiter: c.maxiter,
        })
        .collect();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    let arms: HashMap<String, OracleArm> = oracle
        .into_iter()
        .map(|arm| (arm.case_id.clone(), arm))
        .collect();

    let mut linear = HashMap::new();
    linear.insert(
        "linear_eq",
        LinearConstraint::new(vec![vec![1.0, 1.0]], vec![1.0], vec![1.0]).expect("linear_eq"),
    );
    linear.insert(
        "linear_box",
        LinearConstraint::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![0.5, f64::NEG_INFINITY],
            vec![0.8, 0.3],
        )
        .expect("linear_box"),
    );
    let nonlinear =
        NonlinearConstraint::new(disc, vec![f64::NEG_INFINITY], vec![1.0]).expect("disc");

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut ledger = CompareLedger::new("diff_opt_slsqp_constrained", &["slsqp"]);
    for case in &cases {
        let arm = &arms[case.id];
        let constraints = constraints_of(case.id, &linear, &nonlinear);
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Slsqp),
            bounds: case.bounds.as_deref(),
            constraints: &constraints,
            maxiter: Some(case.maxiter),
            ..MinimizeOptions::default()
        };
        let fsci = minimize(case.fun, &case.x0, options);
        let mut diff = CaseDiff {
            case_id: case.id.to_string(),
            fsci_x: Vec::new(),
            scipy_x: arm.x.clone().unwrap_or_default(),
            fsci_fun: f64::NAN,
            scipy_fun: arm.fun.unwrap_or(f64::NAN),
            fsci_message: String::new(),
            scipy_message: arm.message.clone().unwrap_or_default(),
            fsci_nit: 0,
            scipy_nit: arm.nit.unwrap_or(0),
            maxcv: None,
            pass: false,
            reason: String::new(),
        };
        match ledger.both("slsqp", case.id, arm.status, fsci.as_ref().ok()) {
            // Recorded by the ledger: SciPy produced nothing (oracle_missing) or fsci erred.
            None => {
                diff.reason = match &fsci {
                    Err(e) => format!("fsci error {e}"),
                    Ok(_) => "SciPy produced no result".to_string(),
                };
            }
            Some((status, r)) => {
                diff.fsci_x.clone_from(&r.x);
                diff.fsci_fun = r.fun.unwrap_or(f64::NAN);
                diff.fsci_message.clone_from(&r.message);
                diff.fsci_nit = r.nit;
                diff.maxcv = r.maxcv;
                let mut problems = Vec::new();
                // Set when `slices` below already recorded this case's single ledger outcome.
                let mut recorded = false;
                if r.message != diff.scipy_message {
                    problems.push(format!(
                        "exit '{}' vs SciPy '{}'",
                        r.message, diff.scipy_message
                    ));
                }
                if r.success != (status == 0) {
                    problems.push(format!("success {} vs SciPy status {status}", r.success));
                }
                if status == 0 {
                    // `slices` rejects a length mismatch and a NaN in fsci's x, which the
                    // `f64::max` fold below would otherwise drop.
                    match ledger.slices("slsqp", case.id, arm.x.as_deref(), Some(r.x.as_slice())) {
                        None => {
                            recorded = true;
                            problems
                                .push(format!("x {:?} rejected against SciPy {:?}", r.x, arm.x));
                        }
                        Some((scipy_x, fsci_x)) => {
                            let dx = fsci_x
                                .iter()
                                .zip(scipy_x)
                                .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
                                .fold(0.0, f64::max);
                            if dx.is_nan() || dx > X_REL_TOL {
                                problems.push(format!("x rel diff {dx:e}"));
                            }
                        }
                    }
                    let dfun =
                        (diff.fsci_fun - diff.scipy_fun).abs() / diff.scipy_fun.abs().max(1.0);
                    if dfun.is_nan() || dfun > FUN_REL_TOL {
                        problems.push(format!("fun rel diff {dfun:e}"));
                    }
                    if let Some(cv) = r.maxcv
                        && (cv.is_nan() || cv > MAXCV_TOL)
                    {
                        problems.push(format!("maxcv {cv:e}"));
                    }
                }
                diff.pass = problems.is_empty();
                diff.reason = problems.join("; ");
                if !recorded {
                    ledger.compared("slsqp", case.id, diff.pass);
                }
            }
        }
        diffs.push(diff);
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let same_iteration_count = diffs
        .iter()
        .filter(|d| d.pass && d.fsci_nit == d.scipy_nit)
        .count();
    let log = DiffLog {
        test_id: "diff_opt_slsqp_constrained".into(),
        category: "scipy.optimize.minimize(method='SLSQP') with constraints and bounds".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        same_iteration_count,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    fs::create_dir_all(output_dir()).expect("create slsqp diff dir");
    fs::write(
        output_dir().join("diff_opt_slsqp_constrained.json"),
        serde_json::to_string_pretty(&log).expect("serialize slsqp log"),
    )
    .expect("write slsqp log");

    for d in &diffs {
        println!(
            "{} fsci x={:?} fun={:e} nit={} '{}' | scipy x={:?} fun={:e} nit={} '{}' {}",
            d.case_id,
            d.fsci_x,
            d.fsci_fun,
            d.fsci_nit,
            d.fsci_message,
            d.scipy_x,
            d.scipy_fun,
            d.scipy_nit,
            d.scipy_message,
            d.reason
        );
    }
    println!(
        "{} cases compared, {same_iteration_count} with SciPy's iteration count",
        diffs.len()
    );
    assert_eq!(diffs.len(), cases.len(), "every case must be compared");
    assert!(
        all_pass,
        "minimize(SLSQP) vs scipy.optimize.minimize(SLSQP) failed"
    );
    ledger.finish(cases.len());
}

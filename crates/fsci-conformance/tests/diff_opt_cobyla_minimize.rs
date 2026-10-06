#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `minimize(method='COBYLA')` (frankenscipy-1ksfv.3):
//! Powell's COBYLA as SciPy 1.17.1 runs it (`scipy._lib.pyprima`) against fsci's transcription,
//! with dict, `LinearConstraint` and `NonlinearConstraint` constraints and bounds:
//! Hock–Schittkowski #6, #7, #21 (started infeasible), #35, #71, #76, a bounds-only problem, an
//! `x0` outside its bounds (projected), a 10-variable problem with five nonlinear constraints,
//! an equality-constrained problem, `LinearConstraint` equality and range rows, an infeasible
//! `x0` projected onto two `LinearConstraint` inequality rows (by SLSQP, as SciPy does) and
//! onto two equality rows (by the minimum-norm `np.linalg.lstsq` step),
//! a `NonlinearConstraint`, linear + nonlinear constraints with bounds, an evaluation limit
//! (`maxiter=5`, not a success), a narrow curved feasible region, incompatible constraints,
//! the unconstrained case, `f_target`, a vector-valued inequality, and `tol` / `rhobeg`.
//!
//! The objectives and constraints are written with the same floating-point operations in the
//! same order on both sides (`t * t` rather than `t**2`), so both sides see identical function
//! values. Per case `success`, the exit message and the status (SciPy's `info`, or
//! "constraints not satisfied") must match. `x` must agree to `X_REL_TOL` relative to
//! max(|x|, 1), `fun` to `FUN_REL_TOL` relative to max(|fun|, 1), and when SciPy succeeded
//! fsci's `maxcv` must be at most the case's `catol` (SciPy's default √ε). The evaluation counts
//! are printed side by side, and the log records how many cases are bit-identical (same `x`,
//! `fun` bits and `nfev`): fsci reproduces NumPy's evaluation order for every operation COBYLA
//! performs (up to 25 active constraints), so every case is expected to be bit-identical except
//! where the starting point is projected by SLSQP (`linear_projection_3d`): SciPy projects with
//! its own SLSQP, fsci with fsci's, the projections differ in their last bits, and the two
//! runs then agree only to COBYLA's accuracy.
//!
//! A second test runs `scipy.optimize.fmin_cobyla` against `fsci_opt::fmin_cobyla`.
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
    Bound, Constraint, ConvergenceStatus, FminCobylaOptions, LinearConstraint,
    MinimizeMethodOptions, MinimizeOptions, NonlinearConstraint, OptimizeMethod, fmin_cobyla,
    minimize,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-003";
const X_REL_TOL: f64 = 1.0e-3;
const FUN_REL_TOL: f64 = 1.0e-4;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const INFEASIBLE_PREFIX: &str = "Did not converge to a solution satisfying the constraints";

type Objective = fn(&[f64]) -> f64;

struct Case {
    id: &'static str,
    fun: Objective,
    x0: Vec<f64>,
    bounds: Option<Vec<Bound>>,
    maxiter: Option<usize>,
    rhobeg: Option<f64>,
    tol: Option<f64>,
    f_target: Option<f64>,
}

fn sq(t: f64) -> f64 {
    t * t
}

fn rosen(v: &[f64]) -> f64 {
    sq(1.0 - v[0]) + 100.0 * sq(v[1] - v[0] * v[0])
}

fn quad21(v: &[f64]) -> f64 {
    sq(v[0] - 2.0) + sq(v[1] - 1.0)
}

fn ten_d(v: &[f64]) -> f64 {
    let mut s = 0.0;
    for (i, &vi) in v.iter().enumerate().take(10) {
        s += (i + 1) as f64 * sq(vi - 0.5 * (i % 3) as f64);
    }
    s + v[0] * v[9]
}

/// `0.5 x^T H x + g^T x`, summed entry by entry as the oracle's `quad3` does.
fn quad3(v: &[f64]) -> f64 {
    const H: [[f64; 3]; 3] = [[2.0, 0.0, 1.0], [0.0, 4.0, 0.0], [1.0, 0.0, 2.0]];
    const G: [f64; 3] = [-2.0, 2.0, -0.6];
    let mut f = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            f += 0.5 * H[i][j] * v[i] * v[j];
        }
    }
    for i in 0..3 {
        f += G[i] * v[i];
    }
    f
}

fn case(id: &'static str, fun: Objective, x0: &[f64], bounds: Option<Vec<Bound>>) -> Case {
    Case {
        id,
        fun,
        x0: x0.to_vec(),
        bounds,
        maxiter: None,
        rhobeg: None,
        tol: None,
        f_target: None,
    }
}

fn cases() -> Vec<Case> {
    let lower0 = |n| Some(vec![(Some(0.0), None); n]);
    let mut maxiter_5 = case("maxiter_5", rosen, &[-1.2, 1.0], None);
    maxiter_5.maxiter = Some(5);
    let mut fmin_doc = case("fmin_doc_example", |v| v[0] * v[1], &[0.0, 0.1], None);
    fmin_doc.tol = Some(1e-7);
    let mut f_target = case("f_target", quad21, &[0.0, 0.0], None);
    f_target.f_target = Some(2.5);
    let mut rosen_disc = case("rosen_disc", rosen, &[0.0, 0.0], None);
    rosen_disc.maxiter = Some(3000);
    rosen_disc.rhobeg = Some(0.5);
    vec![
        case("ineq_quadratic", quad21, &[0.0, 0.0], None),
        case("hs6", |v| sq(1.0 - v[0]), &[-1.2, 1.0], None),
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
            "bounds_only",
            |v| sq(v[0] - 2.0) + sq(v[1] - 2.0) + 0.5 * v[0] * v[1],
            &[0.5, 0.0],
            Some(vec![(Some(0.0), Some(1.0)), (Some(-1.0), Some(1.5))]),
        ),
        case(
            "bounds_x0_outside",
            |v| sq(v[0] + 1.0) + sq(v[1] - 3.0),
            &[5.0, -4.0],
            Some(vec![(Some(0.0), Some(2.0)), (Some(-1.0), Some(2.0))]),
        ),
        case("ten_d_five_nl", ten_d, &[0.0; 10], None),
        case(
            "eq_plane",
            |v| v[0] * v[0] + v[1] * v[1] + v[2] * v[2],
            &[1.0, 1.0, 1.0],
            None,
        ),
        case(
            "linear_eq",
            |v| v[0] * v[0] + v[1] * v[1],
            &[2.0, -1.0],
            None,
        ),
        case(
            "linear_box",
            |v| sq(v[0] - 1.0) + sq(v[1] - 1.0),
            &[0.0, 0.0],
            None,
        ),
        case("nonlinear_disc", |v| -(v[0] * v[1]), &[0.5, 0.5], None),
        case("linear_projection_3d", quad3, &[1.0, 1.0, 1.0], None),
        case(
            "linear_eq_projection_4d",
            |v| sq(v[0] - 1.0) + sq(v[1] + 2.0) + sq(v[2] - 0.5) + sq(v[3]) + v[0] * v[3],
            &[0.7, -0.4, 1.9, 2.3],
            None,
        ),
        case(
            "linear_nonlinear_bounds",
            |v| sq(v[0] - 1.0) + sq(v[1] - 2.0) + sq(v[2] - 3.0),
            &[0.0, 0.0, 0.0],
            Some(vec![(None, None), (None, None), (Some(0.0), Some(1.5))]),
        ),
        maxiter_5,
        case("narrow_curved", |v| v[0], &[1.0, 1.0], None),
        case("infeasible", |v| v[0] * v[0], &[0.5], None),
        case(
            "unconstrained_quad",
            |v| sq(v[0] - 3.0) + 2.0 * sq(v[1] + 1.0) + v[0] * v[1],
            &[0.0, 0.0],
            None,
        ),
        fmin_doc,
        f_target,
        case("vector_ineq", |v| -(v[0] * v[1]), &[0.5, 0.5], None),
        rosen_disc,
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
        "ineq_quadratic" | "f_target" => {
            vec![Constraint::ineq(|v: &[f64]| vec![1.0 - v[0] - v[1]])]
        }
        "hs6" => vec![Constraint::eq(|v: &[f64]| {
            vec![10.0 * (v[1] - v[0] * v[0])]
        })],
        "hs7" => vec![Constraint::eq(|v: &[f64]| {
            vec![sq(1.0 + v[0] * v[0]) + v[1] * v[1] - 4.0]
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
            Constraint::eq(|v: &[f64]| {
                vec![v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3] - 40.0]
            }),
        ],
        "hs76" => vec![
            Constraint::ineq(|v: &[f64]| vec![5.0 - v[0] - 2.0 * v[1] - v[2] - v[3]]),
            Constraint::ineq(|v: &[f64]| vec![4.0 - 3.0 * v[0] - v[1] - 2.0 * v[2] + v[3]]),
            Constraint::ineq(|v: &[f64]| vec![v[1] + 4.0 * v[2] - 1.5]),
        ],
        "ten_d_five_nl" => (0..5)
            .map(|k| {
                Constraint::ineq(move |v: &[f64]| {
                    let mut s = 0.0;
                    for j in 0..3 {
                        s += sq(v[(k + j) % 10]);
                    }
                    vec![4.0 - s - v[k]]
                })
            })
            .collect(),
        "eq_plane" => vec![Constraint::eq(|v: &[f64]| {
            vec![v[0] + 2.0 * v[1] + 3.0 * v[2] - 6.0]
        })],
        "linear_eq" | "linear_box" | "linear_projection_3d" | "linear_eq_projection_4d" => {
            Constraint::from_linear(&linear[id])
        }
        "nonlinear_disc" => Constraint::from_nonlinear(nonlinear),
        "linear_nonlinear_bounds" => {
            let mut cons = Constraint::from_linear(&linear[id]);
            cons.extend(Constraint::from_nonlinear(nonlinear));
            cons
        }
        "maxiter_5" => vec![Constraint::ineq(|v: &[f64]| {
            vec![2.0 - v[0] * v[0] - v[1] * v[1]]
        })],
        "narrow_curved" => vec![
            Constraint::ineq(|v: &[f64]| vec![v[1] - v[0] * v[0]]),
            Constraint::ineq(|v: &[f64]| vec![v[0] * v[0] + 1e-3 - v[1]]),
        ],
        "infeasible" => vec![
            Constraint::ineq(|v: &[f64]| vec![v[0] - 1.0]),
            Constraint::ineq(|v: &[f64]| vec![-v[0]]),
        ],
        "fmin_doc_example" => vec![
            Constraint::ineq(|v: &[f64]| vec![1.0 - (v[0] * v[0] + v[1] * v[1])]),
            Constraint::ineq(|v: &[f64]| vec![v[1]]),
        ],
        "vector_ineq" => vec![Constraint::ineq(|v: &[f64]| {
            vec![1.0 - v[0] * v[0] - v[1] * v[1], v[0], v[1]]
        })],
        "rosen_disc" => vec![Constraint::ineq(|v: &[f64]| {
            vec![1.5 - v[0] * v[0] - v[1] * v[1]]
        })],
        _ => Vec::new(),
    }
}

/// SciPy's `(status, success, message)` as fsci's [`ConvergenceStatus`].
fn expected_status(status: i32, message: &str) -> Option<ConvergenceStatus> {
    if message.starts_with(INFEASIBLE_PREFIX) {
        return Some(ConvergenceStatus::Infeasible);
    }
    match status {
        0 | 1 => Some(ConvergenceStatus::Success),
        3 => Some(ConvergenceStatus::MaxEvaluations),
        20 => Some(ConvergenceStatus::MaxIterations),
        -1 | -2 => Some(ConvergenceStatus::NanEncountered),
        7 => Some(ConvergenceStatus::PrecisionLoss),
        30 => Some(ConvergenceStatus::CallbackStop),
        _ => None,
    }
}

#[derive(Debug, Clone, Serialize)]
struct QueryCase {
    case_id: String,
    x0: Vec<f64>,
    bounds: Option<Vec<(Option<f64>, Option<f64>)>>,
    maxiter: Option<usize>,
    rhobeg: Option<f64>,
    tol: Option<f64>,
    f_target: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleArm {
    case_id: String,
    x: Option<Vec<f64>>,
    fun: Option<f64>,
    status: Option<i32>,
    success: Option<bool>,
    nfev: Option<usize>,
    maxcv: Option<f64>,
    message: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    fsci_x: Vec<f64>,
    scipy_x: Vec<f64>,
    x_rel_diff: f64,
    fsci_fun: f64,
    scipy_fun: f64,
    fun_rel_diff: f64,
    fsci_success: bool,
    scipy_success: bool,
    fsci_status: String,
    scipy_status: i32,
    fsci_message: String,
    scipy_message: String,
    fsci_nfev: usize,
    scipy_nfev: usize,
    fsci_maxcv: Option<f64>,
    scipy_maxcv: f64,
    bit_identical: bool,
    pass: bool,
    reason: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    bit_identical: usize,
    same_nfev: usize,
    max_x_rel_diff: f64,
    max_fun_rel_diff: f64,
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

def sq(t):
    return t * t

rosen = lambda v: sq(1 - v[0]) + 100 * sq(v[1] - v[0] * v[0])
quad21 = lambda v: sq(v[0] - 2) + sq(v[1] - 1)

def ten_d(v):
    s = 0.0
    for i in range(10):
        s += (i + 1) * sq(v[i] - 0.5 * (i % 3))
    return s + v[0] * v[9]

def ten_con(k):
    def c(v):
        s = 0.0
        for j in range(3):
            s += sq(v[(k + j) % 10])
        return 4 - s - v[k]
    return c

H3 = [[2.0, 0.0, 1.0], [0.0, 4.0, 0.0], [1.0, 0.0, 2.0]]
G3 = [-2.0, 2.0, -0.6]

def quad3(v):
    f = 0.0
    for i in range(3):
        for j in range(3):
            f += 0.5 * H3[i][j] * v[i] * v[j]
    for i in range(3):
        f += G3[i] * v[i]
    return f

FUN = {
    "ineq_quadratic": quad21,
    "hs6": lambda v: sq(1 - v[0]),
    "hs7": lambda v: math.log(1 + v[0] * v[0]) - v[1],
    "hs21_infeasible_start": lambda v: 0.01 * v[0] * v[0] + v[1] * v[1] - 100,
    "hs35": lambda v: 9 - 8 * v[0] - 6 * v[1] - 4 * v[2] + 2 * v[0] * v[0] + 2 * v[1] * v[1]
        + v[2] * v[2] + 2 * v[0] * v[1] + 2 * v[0] * v[2],
    "hs71": lambda v: v[0] * v[3] * (v[0] + v[1] + v[2]) + v[2],
    "hs76": lambda v: v[0] * v[0] + 0.5 * v[1] * v[1] + v[2] * v[2] + 0.5 * v[3] * v[3]
        - v[0] * v[2] + v[2] * v[3] - v[0] - 3 * v[1] + v[2] - v[3],
    "bounds_only": lambda v: sq(v[0] - 2) + sq(v[1] - 2) + 0.5 * v[0] * v[1],
    "bounds_x0_outside": lambda v: sq(v[0] + 1) + sq(v[1] - 3),
    "ten_d_five_nl": ten_d,
    "eq_plane": lambda v: v[0] * v[0] + v[1] * v[1] + v[2] * v[2],
    "linear_eq": lambda v: v[0] * v[0] + v[1] * v[1],
    "linear_box": lambda v: sq(v[0] - 1) + sq(v[1] - 1),
    "nonlinear_disc": lambda v: -(v[0] * v[1]),
    "linear_projection_3d": quad3,
    "linear_eq_projection_4d": lambda v: sq(v[0] - 1) + sq(v[1] + 2) + sq(v[2] - 0.5) + sq(v[3])
        + v[0] * v[3],
    "linear_nonlinear_bounds": lambda v: sq(v[0] - 1) + sq(v[1] - 2) + sq(v[2] - 3),
    "maxiter_5": rosen,
    "narrow_curved": lambda v: v[0],
    "infeasible": lambda v: v[0] * v[0],
    "unconstrained_quad": lambda v: sq(v[0] - 3) + 2 * sq(v[1] + 1) + v[0] * v[1],
    "fmin_doc_example": lambda v: v[0] * v[1],
    "f_target": quad21,
    "vector_ineq": lambda v: -(v[0] * v[1]),
    "rosen_disc": rosen,
}
ineq = lambda f: {'type': 'ineq', 'fun': f}
eq = lambda f: {'type': 'eq', 'fun': f}
disc = NonlinearConstraint(lambda v: [v[0] * v[0] + v[1] * v[1]], -np.inf, 1)
CONS = {
    "ineq_quadratic": [ineq(lambda v: 1 - v[0] - v[1])],
    "f_target": [ineq(lambda v: 1 - v[0] - v[1])],
    "hs6": [eq(lambda v: 10 * (v[1] - v[0] * v[0]))],
    "hs7": [eq(lambda v: sq(1 + v[0] * v[0]) + v[1] * v[1] - 4)],
    "hs21_infeasible_start": [ineq(lambda v: 10 * v[0] - v[1] - 10)],
    "hs35": [ineq(lambda v: 3 - v[0] - v[1] - 2 * v[2])],
    "hs71": [ineq(lambda v: v[0] * v[1] * v[2] * v[3] - 25),
             eq(lambda v: v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3] - 40)],
    "hs76": [ineq(lambda v: 5 - v[0] - 2 * v[1] - v[2] - v[3]),
             ineq(lambda v: 4 - 3 * v[0] - v[1] - 2 * v[2] + v[3]),
             ineq(lambda v: v[1] + 4 * v[2] - 1.5)],
    "ten_d_five_nl": [ineq(ten_con(k)) for k in range(5)],
    "eq_plane": [eq(lambda v: v[0] + 2 * v[1] + 3 * v[2] - 6)],
    "linear_eq": [LinearConstraint([[1, 1]], 1, 1)],
    "linear_box": [LinearConstraint([[1, 0], [0, 1]], [0.5, -np.inf], [0.8, 0.3])],
    "nonlinear_disc": [disc],
    "linear_projection_3d": [LinearConstraint([[1.0, 2.0, -1.0], [0.5, -1.0, 1.0]], [-np.inf, 0.2],
                                              [-0.5, np.inf])],
    "linear_eq_projection_4d": [LinearConstraint([[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 2.0, 0.5]],
                                                 [1.0, 0.3], [1.0, 0.3])],
    "linear_nonlinear_bounds": [LinearConstraint([[1, 1, 1]], -np.inf, 3), disc],
    "maxiter_5": [ineq(lambda v: 2 - v[0] * v[0] - v[1] * v[1])],
    "narrow_curved": [ineq(lambda v: v[1] - v[0] * v[0]), ineq(lambda v: v[0] * v[0] + 1e-3 - v[1])],
    "infeasible": [ineq(lambda v: v[0] - 1), ineq(lambda v: -v[0])],
    "fmin_doc_example": [ineq(lambda v: 1 - (v[0] * v[0] + v[1] * v[1])), ineq(lambda v: v[1])],
    "vector_ineq": [ineq(lambda v: np.array([1 - v[0] * v[0] - v[1] * v[1], v[0], v[1]]))],
    "rosen_disc": [ineq(lambda v: 1.5 - v[0] * v[0] - v[1] * v[1])],
}

out = []
for case in json.load(sys.stdin):
    cid = case["case_id"]
    arm = {"case_id": cid, "x": None, "fun": None, "status": None, "success": None,
           "nfev": None, "maxcv": None, "message": None}
    options = {}
    for key in ("maxiter", "rhobeg", "f_target"):
        if case[key] is not None:
            options[key] = case[key]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = minimize(FUN[cid], case["x0"], method="COBYLA", constraints=CONS.get(cid, ()),
                         bounds=case["bounds"], tol=case["tol"], options=options)
        arm.update(x=[float(v) for v in r.x], fun=float(r.fun), status=int(r.status),
                   success=bool(r.success), nfev=int(r.nfev), maxcv=float(r.maxcv),
                   message=str(r.message))
    except Exception:
        pass
    out.append(arm)
print(json.dumps(out))
"#;
    let query_json = serde_json::to_string(query).expect("serialize cobyla query");
    let stdout = run_oracle(script, &query_json)?;
    Some(serde_json::from_str(&stdout).expect("parse cobyla oracle JSON"))
}

/// Runs `script` under the SciPy oracle with `query_json` on stdin and returns its stdout;
/// `None` (a skip) only when the oracle is unavailable and `FSCI_REQUIRE_SCIPY_ORACLE` is unset.
fn run_oracle(script: &str, query_json: &str) -> Option<String> {
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
                "failed to spawn python3 for the cobyla oracle: {e}"
            );
            eprintln!("skipping cobyla oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open cobyla oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "cobyla oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping cobyla oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for cobyla oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "cobyla oracle failed: {stderr}"
        );
        eprintln!("skipping cobyla oracle: scipy not available\n{stderr}");
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// The largest `|a - b| / max(|b|, 1)` over the entries (NaN if any entry is NaN).
fn max_rel_diff(fsci: &[f64], scipy: &[f64]) -> f64 {
    fsci.iter()
        .zip(scipy)
        .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
        .fold(0.0, |acc: f64, d| {
            if d.is_nan() || acc.is_nan() {
                f64::NAN
            } else {
                acc.max(d)
            }
        })
}

#[test]
fn diff_opt_cobyla_minimize() {
    let cases = cases();
    let query: Vec<QueryCase> = cases
        .iter()
        .map(|c| QueryCase {
            case_id: c.id.to_string(),
            x0: c.x0.clone(),
            bounds: c.bounds.clone(),
            maxiter: c.maxiter,
            rhobeg: c.rhobeg,
            tol: c.tol,
            f_target: c.f_target,
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
    linear.insert(
        "linear_projection_3d",
        LinearConstraint::new(
            vec![vec![1.0, 2.0, -1.0], vec![0.5, -1.0, 1.0]],
            vec![f64::NEG_INFINITY, 0.2],
            vec![-0.5, f64::INFINITY],
        )
        .expect("linear_projection_3d"),
    );
    linear.insert(
        "linear_eq_projection_4d",
        LinearConstraint::new(
            vec![vec![1.0, 1.0, 1.0, 1.0], vec![1.0, -1.0, 2.0, 0.5]],
            vec![1.0, 0.3],
            vec![1.0, 0.3],
        )
        .expect("linear_eq_projection_4d"),
    );
    linear.insert(
        "linear_nonlinear_bounds",
        LinearConstraint::new(
            vec![vec![1.0, 1.0, 1.0]],
            vec![f64::NEG_INFINITY],
            vec![3.0],
        )
        .expect("linear_nonlinear_bounds"),
    );
    let nonlinear =
        NonlinearConstraint::new(disc, vec![f64::NEG_INFINITY], vec![1.0]).expect("disc");

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut ledger = CompareLedger::new("diff_opt_cobyla_minimize", &["cobyla"]);
    for case in &cases {
        let arm = &arms[case.id];
        let constraints = constraints_of(case.id, &linear, &nonlinear);
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Cobyla),
            bounds: case.bounds.as_deref(),
            constraints: &constraints,
            maxiter: case.maxiter,
            tol: case.tol,
            method_options: MinimizeMethodOptions {
                rhobeg: case.rhobeg,
                f_target: case.f_target,
                ..MinimizeMethodOptions::default()
            },
            ..MinimizeOptions::default()
        };
        let fsci = minimize(case.fun, &case.x0, options);
        let mut diff = CaseDiff {
            case_id: case.id.to_string(),
            fsci_x: Vec::new(),
            scipy_x: arm.x.clone().unwrap_or_default(),
            x_rel_diff: f64::NAN,
            fsci_fun: f64::NAN,
            scipy_fun: arm.fun.unwrap_or(f64::NAN),
            fun_rel_diff: f64::NAN,
            fsci_success: false,
            scipy_success: arm.success.unwrap_or(false),
            fsci_status: String::new(),
            scipy_status: arm.status.unwrap_or(i32::MIN),
            fsci_message: String::new(),
            scipy_message: arm.message.clone().unwrap_or_default(),
            fsci_nfev: 0,
            scipy_nfev: arm.nfev.unwrap_or(0),
            fsci_maxcv: None,
            scipy_maxcv: arm.maxcv.unwrap_or(f64::NAN),
            bit_identical: false,
            pass: false,
            reason: String::new(),
        };
        match ledger.both("cobyla", case.id, arm.status, fsci.as_ref().ok()) {
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
                diff.fsci_success = r.success;
                diff.fsci_status = format!("{:?}", r.status);
                diff.fsci_message.clone_from(&r.message);
                diff.fsci_nfev = r.nfev;
                diff.fsci_maxcv = r.maxcv;
                let mut problems = Vec::new();
                // Set when `slices` below already recorded this case's single ledger outcome.
                let mut recorded = false;
                if r.success != diff.scipy_success {
                    problems.push(format!(
                        "success {} vs SciPy {}",
                        r.success, diff.scipy_success
                    ));
                }
                if r.message != diff.scipy_message {
                    problems.push(format!(
                        "exit '{}' vs SciPy '{}'",
                        r.message, diff.scipy_message
                    ));
                }
                match expected_status(status, &diff.scipy_message) {
                    Some(want) if want == r.status => {}
                    want => problems.push(format!(
                        "status {:?} vs SciPy status {status} ({want:?})",
                        r.status
                    )),
                }
                // `slices` rejects a length mismatch and a NaN in fsci's x.
                match ledger.slices("cobyla", case.id, arm.x.as_deref(), Some(r.x.as_slice())) {
                    None => {
                        recorded = true;
                        problems.push(format!("x {:?} rejected against SciPy {:?}", r.x, arm.x));
                    }
                    Some((scipy_x, fsci_x)) => {
                        let dx = max_rel_diff(fsci_x, scipy_x);
                        diff.x_rel_diff = dx;
                        if dx.is_nan() || dx > X_REL_TOL {
                            problems.push(format!("x rel diff {dx:e}"));
                        }
                    }
                }
                let dfun = (diff.fsci_fun - diff.scipy_fun).abs() / diff.scipy_fun.abs().max(1.0);
                diff.fun_rel_diff = dfun;
                if dfun.is_nan() || dfun > FUN_REL_TOL {
                    problems.push(format!("fun rel diff {dfun:e}"));
                }
                if diff.scipy_success {
                    let catol = f64::EPSILON.sqrt();
                    match r.maxcv {
                        Some(cv) if cv <= catol => {}
                        cv => problems.push(format!("maxcv {cv:?} > catol {catol:e}")),
                    }
                }
                diff.bit_identical = r.x.len() == diff.scipy_x.len()
                    && r.x
                        .iter()
                        .zip(&diff.scipy_x)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                    && diff.fsci_fun.to_bits() == diff.scipy_fun.to_bits()
                    && r.nfev == diff.scipy_nfev;
                diff.pass = problems.is_empty();
                diff.reason = problems.join("; ");
                if !recorded {
                    ledger.compared("cobyla", case.id, diff.pass);
                }
            }
        }
        diffs.push(diff);
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let bit_identical = diffs.iter().filter(|d| d.bit_identical).count();
    let same_nfev = diffs.iter().filter(|d| d.fsci_nfev == d.scipy_nfev).count();
    let max_x_rel_diff = diffs.iter().map(|d| d.x_rel_diff).fold(0.0, f64::max);
    let max_fun_rel_diff = diffs.iter().map(|d| d.fun_rel_diff).fold(0.0, f64::max);
    let log = DiffLog {
        test_id: "diff_opt_cobyla_minimize".into(),
        category: "scipy.optimize.minimize(method='COBYLA') with constraints and bounds".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        bit_identical,
        same_nfev,
        max_x_rel_diff,
        max_fun_rel_diff,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    fs::create_dir_all(output_dir()).expect("create cobyla diff dir");
    fs::write(
        output_dir().join("diff_opt_cobyla_minimize.json"),
        serde_json::to_string_pretty(&log).expect("serialize cobyla log"),
    )
    .expect("write cobyla log");

    for d in &diffs {
        println!(
            "{:<24} nfev fsci {:>4} scipy {:>4} | x rel {:.1e} fun rel {:.1e} | success {}/{} | {} {}",
            d.case_id,
            d.fsci_nfev,
            d.scipy_nfev,
            d.x_rel_diff,
            d.fun_rel_diff,
            d.fsci_success,
            d.scipy_success,
            if d.bit_identical {
                "bit-identical"
            } else {
                "differs"
            },
            d.reason
        );
    }
    println!(
        "{} cases compared, {bit_identical} bit-identical to SciPy, {same_nfev} with SciPy's nfev, \
         max x rel diff {max_x_rel_diff:e}, max fun rel diff {max_fun_rel_diff:e}",
        diffs.len()
    );
    assert_eq!(diffs.len(), cases.len(), "every case must be compared");
    assert!(
        all_pass,
        "minimize(COBYLA) vs scipy.optimize.minimize(COBYLA) failed"
    );
    ledger.finish(cases.len());
}

#[derive(Debug, Clone, Serialize)]
struct FminQuery {
    case_id: String,
    x0: Vec<f64>,
    rhobeg: f64,
    rhoend: f64,
    maxfun: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct FminArm {
    case_id: String,
    x: Option<Vec<f64>>,
}

type ScalarConstraint = fn(&[f64]) -> f64;

/// `scipy.optimize.fmin_cobyla(func, x0, cons, rhobeg, rhoend, maxfun)` (default `catol` 2e-4)
/// against `fsci_opt::fmin_cobyla`: the returned `x` must agree to `X_REL_TOL`, and the log
/// counts the bit-identical ones. Every case must be compared.
#[test]
fn diff_opt_fmin_cobyla() {
    let script = r#"
import json, sys, warnings
from scipy.optimize import fmin_cobyla

def sq(t):
    return t * t

FUN = {
    "doc_example": lambda v: v[0] * v[1],
    "ineq_quadratic": lambda v: sq(v[0] - 2) + sq(v[1] - 1),
    "rosen_disc_maxfun_10": lambda v: sq(1 - v[0]) + 100 * sq(v[1] - v[0] * v[0]),
}
CONS = {
    "doc_example": [lambda v: 1 - (v[0] * v[0] + v[1] * v[1]), lambda v: v[1]],
    "ineq_quadratic": [lambda v: 1 - v[0] - v[1]],
    "rosen_disc_maxfun_10": [lambda v: 2 - v[0] * v[0] - v[1] * v[1]],
}
out = []
for case in json.load(sys.stdin):
    cid = case["case_id"]
    arm = {"case_id": cid, "x": None}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = fmin_cobyla(FUN[cid], case["x0"], CONS[cid], rhobeg=case["rhobeg"],
                            rhoend=case["rhoend"], maxfun=case["maxfun"])
        arm["x"] = [float(v) for v in x]
    except Exception:
        pass
    out.append(arm)
print(json.dumps(out))
"#;
    let doc_cons: [ScalarConstraint; 2] = [|v| 1.0 - (v[0] * v[0] + v[1] * v[1]), |v| v[1]];
    let quad_cons: [ScalarConstraint; 1] = [|v| 1.0 - v[0] - v[1]];
    let rosen_cons: [ScalarConstraint; 1] = [|v| 2.0 - v[0] * v[0] - v[1] * v[1]];
    let doc_fun: Objective = |v| v[0] * v[1];
    let cases: Vec<(
        &str,
        Objective,
        Vec<f64>,
        &[ScalarConstraint],
        FminCobylaOptions,
    )> = vec![
        (
            "doc_example",
            doc_fun,
            vec![0.0, 0.1],
            doc_cons.as_slice(),
            FminCobylaOptions {
                rhoend: 1e-7,
                ..FminCobylaOptions::default()
            },
        ),
        (
            "ineq_quadratic",
            quad21,
            vec![0.0, 0.0],
            quad_cons.as_slice(),
            FminCobylaOptions::default(),
        ),
        (
            "rosen_disc_maxfun_10",
            rosen,
            vec![-1.2, 1.0],
            rosen_cons.as_slice(),
            FminCobylaOptions {
                rhobeg: 0.5,
                maxfun: 10,
                ..FminCobylaOptions::default()
            },
        ),
    ];
    let query: Vec<FminQuery> = cases
        .iter()
        .map(|(id, _, x0, _, o)| FminQuery {
            case_id: (*id).to_string(),
            x0: x0.clone(),
            rhobeg: o.rhobeg,
            rhoend: o.rhoend,
            maxfun: o.maxfun,
        })
        .collect();
    let query_json = serde_json::to_string(&query).expect("serialize fmin_cobyla query");
    let Some(stdout) = run_oracle(script, &query_json) else {
        return;
    };
    let arms: Vec<FminArm> = serde_json::from_str(&stdout).expect("parse fmin_cobyla oracle JSON");
    let arms: HashMap<String, FminArm> = arms.into_iter().map(|a| (a.case_id.clone(), a)).collect();

    let mut ledger = CompareLedger::new("diff_opt_fmin_cobyla", &["fmin_cobyla"]);
    let mut compared = 0;
    let mut bit_identical = 0;
    let mut failures = Vec::new();
    for &(id, fun, ref x0, cons, opts) in &cases {
        let fsci = fmin_cobyla(fun, x0, cons, opts);
        let arm = &arms[id];
        match ledger.slices(
            "fmin_cobyla",
            id,
            arm.x.as_deref(),
            fsci.as_ref().ok().map(Vec::as_slice),
        ) {
            None => failures.push(format!("{id}: fsci {fsci:?} vs SciPy {:?}", arm.x)),
            Some((scipy_x, fsci_x)) => {
                let dx = max_rel_diff(fsci_x, scipy_x);
                let same = fsci_x
                    .iter()
                    .zip(scipy_x)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                bit_identical += usize::from(same);
                let pass = !dx.is_nan() && dx <= X_REL_TOL;
                ledger.compared("fmin_cobyla", id, pass);
                compared += 1;
                println!(
                    "{id:<22} x fsci {fsci_x:?} scipy {scipy_x:?} | rel diff {dx:.1e} | {}",
                    if same { "bit-identical" } else { "differs" }
                );
                if !pass {
                    failures.push(format!("{id}: x rel diff {dx:e}"));
                }
            }
        }
    }
    println!("{compared} fmin_cobyla cases compared, {bit_identical} bit-identical to SciPy");
    assert!(
        failures.is_empty(),
        "fmin_cobyla vs SciPy failed: {failures:?}"
    );
    assert_eq!(compared, cases.len(), "every case must be compared");
    ledger.finish(cases.len());
}

#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `minimize(method='trust-constr')` with constraints and
//! bounds (frankenscipy-1ksfv.2): Hock–Schittkowski #6, #7 (equality-only, so SciPy's
//! `equality_constrained_sqp`), #71 (started infeasible, mixed equality/inequality/bounds) and
//! #76; `min (x−2)² + (y−1)²` s.t. `x + y <= 1` as a `NonlinearConstraint`; an equality-only
//! problem; a bounds-only problem (SciPy's sparse `AugmentedSystem` projections); unconstrained
//! Rosenbrock (`equality_constrained_sqp` with no constraints); a 50-variable problem with five
//! linear equalities; a two-sided `LinearConstraint`; HS71 with exact gradient, Hessian and
//! constraint Jacobians/Hessians; Rosenbrock with `hessp`; `keep_feasible` bounds; the
//! `SVDFactorization` projections; incompatible equalities (status 4); a duplicated equality
//! (SciPy's rank-deficient SVD fallback, after which its projected CG takes `n - m = 0` steps
//! and never leaves `x0`); and an iteration limit.
//!
//! Per case the method name, SciPy's status and message must match; `x` must agree to
//! `X_REL_TOL` (relative to max(|x|, 1)), `fun` to `FUN_REL_TOL`, the Lagrange multipliers of
//! every constraint to `V_REL_TOL` (relative to max(|v|, 1)), and fsci's constraint violation
//! must be at most `MAXCV_TOL` whenever SciPy's is. `nit` and `nfev` are printed side by side. Each objective and constraint is written with the same floating-point operations
//! on both sides (`t*t`, never `t**2`, whose `pow` can differ from `t*t` by an ulp), so the two
//! solvers see the same function values.
//!
//! Where SciPy differentiates by `'2-point'` differences, a one-ulp difference anywhere in the
//! iteration (SciPy's OpenBLAS kernels round differently from fsci's loops) becomes a ~1e-8
//! difference in the next gradient, so iterates agree to about 1e-8 rather than to the ulp and
//! `nit` may differ. Two problems are given exact first derivatives on both sides for that
//! reason: for `nl_xy` and `eq_only` with differences SciPy's OWN status changes under 1e-13
//! perturbations of `x0` (nl_xy: status 1 in 17 of 20 perturbed runs, 2 in 3; eq_only: 1, 2 or
//! the iteration limit), so the status there is decided by rounding noise, not the algorithm.
//! With exact first derivatives both are stable (nl_xy: status 1 at nit 28 in 30 of 30 runs).
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
    Bound, Constraint, FactorizationMethod, MinimizeMethodOptions, MinimizeOptions, OptError,
    OptimizeMethod, TrustBounds, TrustConstrResult, TrustConstraint, minimize, trust_constr_full,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-003";
const X_REL_TOL: f64 = 1.0e-5;
const FUN_REL_TOL: f64 = 1.0e-7;
const MAXCV_TOL: f64 = 1.0e-8;
const V_REL_TOL: f64 = 1.0e-4;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const ARM: &str = "trust_constr";

struct Case {
    id: &'static str,
    x0: Vec<f64>,
    maxiter: usize,
}

fn case(id: &'static str, x0: &[f64]) -> Case {
    Case {
        id,
        x0: x0.to_vec(),
        maxiter: 1000,
    }
}

fn cases() -> Vec<Case> {
    let mut hs71_maxiter3 = case("hs71_maxiter3", &[1.0, 5.0, 5.0, 1.0]);
    hs71_maxiter3.maxiter = 3;
    vec![
        case("hs6", &[-1.2, 1.0]),
        case("hs7", &[2.0, 2.0]),
        case("hs71", &[1.0, 5.0, 5.0, 1.0]),
        case("hs76", &[0.5, 0.5, 0.5, 0.5]),
        case("nl_xy", &[0.0, 0.0]),
        case("eq_only", &[2.0, -1.0]),
        case("bounds_only", &[-1.0, 1.0]),
        case("rosen", &[-1.2, 1.0]),
        case("lin50", &[0.0; N50]),
        case("lin2side", &[2.0, 0.0]),
        case("hs71_exact", &[1.0, 5.0, 5.0, 1.0]),
        case("rosen_hessp", &[-1.2, 1.0]),
        case("rosen_keep_feasible", &[-1.0, 1.0]),
        case("hs7_svd", &[2.0, 2.0]),
        case("incompatible_eq", &[0.0, 0.0]),
        case("duplicate_eq", &[2.0, -1.0]),
        hs71_maxiter3,
    ]
}

// ── Objectives and derivatives (the same operations as the Python side) ──

const N50: usize = 50;

fn sq(t: f64) -> f64 {
    t * t
}

fn hs6(v: &[f64]) -> f64 {
    sq(1.0 - v[0])
}

fn hs7(v: &[f64]) -> f64 {
    (1.0 + v[0] * v[0]).ln() - v[1]
}

fn hs71(v: &[f64]) -> f64 {
    v[0] * v[3] * (v[0] + v[1] + v[2]) + v[2]
}

fn hs71_grad(v: &[f64]) -> Vec<f64> {
    vec![
        v[3] * (2.0 * v[0] + v[1] + v[2]),
        v[0] * v[3],
        v[0] * v[3] + 1.0,
        v[0] * (v[0] + v[1] + v[2]),
    ]
}

fn hs71_hess(v: &[f64]) -> Vec<Vec<f64>> {
    let s = 2.0 * v[0] + v[1] + v[2];
    vec![
        vec![2.0 * v[3], v[3], v[3], s],
        vec![v[3], 0.0, 0.0, v[0]],
        vec![v[3], 0.0, 0.0, v[0]],
        vec![s, v[0], v[0], 0.0],
    ]
}

fn hs76(v: &[f64]) -> f64 {
    v[0] * v[0] + 0.5 * v[1] * v[1] + v[2] * v[2] + 0.5 * v[3] * v[3] - v[0] * v[2] + v[2] * v[3]
        - v[0]
        - 3.0 * v[1]
        + v[2]
        - v[3]
}

fn nl_xy(v: &[f64]) -> f64 {
    sq(v[0] - 2.0) + sq(v[1] - 1.0)
}

fn nl_xy_grad(v: &[f64]) -> Vec<f64> {
    vec![2.0 * (v[0] - 2.0), 2.0 * (v[1] - 1.0)]
}

fn eq_only(v: &[f64]) -> f64 {
    v[0] * v[0] + v[1] * v[1]
}

fn eq_only_grad(v: &[f64]) -> Vec<f64> {
    vec![2.0 * v[0], 2.0 * v[1]]
}

fn rosen(v: &[f64]) -> f64 {
    sq(1.0 - v[0]) + 100.0 * sq(v[1] - v[0] * v[0])
}

fn rosen_grad(v: &[f64]) -> Vec<f64> {
    vec![
        -2.0 * (1.0 - v[0]) - 400.0 * v[0] * (v[1] - v[0] * v[0]),
        200.0 * (v[1] - v[0] * v[0]),
    ]
}

fn rosen_hessp(v: &[f64], p: &[f64]) -> Vec<f64> {
    let h00 = 2.0 - 400.0 * v[1] + 1200.0 * v[0] * v[0];
    let h01 = -400.0 * v[0];
    vec![h00 * p[0] + h01 * p[1], h01 * p[0] + 200.0 * p[1]]
}

fn f50(v: &[f64]) -> f64 {
    let (mut s2, mut s4) = (0.0, 0.0);
    for (i, &vi) in v.iter().enumerate() {
        let d = vi - i as f64 / N50 as f64;
        s2 += d * d;
        let q = vi * vi;
        s4 += q * q;
    }
    s2 + 0.1 * s4
}

fn objective(id: &str) -> fn(&[f64]) -> f64 {
    match id {
        "hs6" => hs6,
        "hs7" | "hs7_svd" => hs7,
        "hs71" | "hs71_exact" | "hs71_maxiter3" => hs71,
        "hs76" => hs76,
        "nl_xy" => nl_xy,
        "eq_only" | "incompatible_eq" | "duplicate_eq" => eq_only,
        "lin50" => f50,
        "lin2side" => |v: &[f64]| sq(v[0] - 1.0) + sq(v[1] - 2.5),
        _ => rosen,
    }
}

/// A problem posed through `MinimizeOptions` alone (old-style dict constraints, tuple bounds):
/// fsci's full result, and whether `minimize(method=TrustConstr)` returned the same.
fn via_options(
    fun: fn(&[f64]) -> f64,
    x0: &[f64],
    constraints: &[Constraint<'_>],
    bounds: Option<&[Bound]>,
    options: MinimizeOptions<'_>,
) -> (Result<TrustConstrResult, OptError>, Option<bool>) {
    let options = MinimizeOptions {
        constraints,
        bounds,
        ..options
    };
    let full = trust_constr_full(&fun, x0, &[], None, options);
    let same = minimize(fun, x0, options)
        .ok()
        .zip(full.as_ref().ok())
        .map(|(m, f)| m.x == f.result.x && m.fun == f.result.fun && m.nit == f.result.nit);
    (full, Some(same.unwrap_or(false)))
}

/// fsci's run of case `id`, and — when it is posed through `MinimizeOptions` alone — whether
/// `minimize(method=TrustConstr)` returned the same result.
fn run_fsci(c: &Case) -> (Result<TrustConstrResult, OptError>, Option<bool>) {
    let fun = objective(c.id);
    let x0 = c.x0.as_slice();
    let base = MinimizeOptions {
        method: Some(OptimizeMethod::TrustConstr),
        maxiter: Some(c.maxiter),
        ..MinimizeOptions::default()
    };
    let via_options = |constraints: &[Constraint<'_>], bounds: Option<&[Bound]>, options| {
        via_options(fun, x0, constraints, bounds, options)
    };
    let hs71_cons = || {
        vec![
            Constraint::ineq(|v: &[f64]| vec![v[0] * v[1] * v[2] * v[3] - 25.0]),
            Constraint::eq(|v: &[f64]| {
                vec![v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3] - 40.0]
            }),
        ]
    };
    let hs71_bounds = [(Some(1.0), Some(5.0)); 4];
    let hs7_cons = || {
        vec![Constraint::eq(|v: &[f64]| {
            vec![sq(1.0 + v[0] * v[0]) + v[1] * v[1] - 4.0]
        })]
    };
    let rosen_box = [(Some(-2.0), Some(0.5)), (Some(-2.0), Some(2.0))];
    match c.id {
        "hs6" => via_options(
            &[Constraint::eq(|v: &[f64]| {
                vec![10.0 * (v[1] - v[0] * v[0])]
            })],
            None,
            base,
        ),
        "hs7" => via_options(&hs7_cons(), None, base),
        "hs7_svd" => via_options(
            &hs7_cons(),
            None,
            MinimizeOptions {
                method_options: MinimizeMethodOptions {
                    factorization_method: Some(FactorizationMethod::SvdFactorization),
                    ..MinimizeMethodOptions::default()
                },
                ..base
            },
        ),
        "hs71" | "hs71_maxiter3" => via_options(&hs71_cons(), Some(&hs71_bounds), base),
        "hs76" => via_options(
            &[
                Constraint::ineq(|v: &[f64]| vec![5.0 - v[0] - 2.0 * v[1] - v[2] - v[3]]),
                Constraint::ineq(|v: &[f64]| vec![4.0 - 3.0 * v[0] - v[1] - 2.0 * v[2] + v[3]]),
                Constraint::ineq(|v: &[f64]| vec![v[1] + 4.0 * v[2] - 1.5]),
            ],
            Some(&[(Some(0.0), None); 4]),
            base,
        ),
        "eq_only" => via_options(
            &[Constraint::eq(|v: &[f64]| vec![v[0] + v[1] - 1.0])
                .with_jac(|_: &[f64]| vec![vec![1.0, 1.0]])],
            None,
            MinimizeOptions {
                gradient: Some(eq_only_grad),
                ..base
            },
        ),
        "incompatible_eq" => via_options(
            &[
                Constraint::eq(|v: &[f64]| vec![v[0] - 1.0])
                    .with_jac(|_: &[f64]| vec![vec![1.0, 0.0]]),
                Constraint::eq(|v: &[f64]| vec![v[0] - 2.0])
                    .with_jac(|_: &[f64]| vec![vec![1.0, 0.0]]),
            ],
            None,
            MinimizeOptions {
                gradient: Some(eq_only_grad),
                ..base
            },
        ),
        "duplicate_eq" => {
            let con = TrustConstraint::linear(
                vec![vec![1.0, 1.0], vec![1.0, 1.0]],
                vec![1.0, 1.0],
                vec![1.0, 1.0],
            );
            let options = MinimizeOptions {
                gradient: Some(eq_only_grad),
                ..base
            };
            (trust_constr_full(&fun, x0, &[con], None, options), None)
        }
        "bounds_only" => via_options(&[], Some(&rosen_box), base),
        "rosen" => via_options(&[], None, base),
        "rosen_hessp" => via_options(
            &[],
            None,
            MinimizeOptions {
                gradient: Some(rosen_grad),
                hessp: Some(rosen_hessp),
                ..base
            },
        ),
        "nl_xy" => {
            let f = |v: &[f64]| vec![v[0] + v[1]];
            let j = |_: &[f64]| vec![vec![1.0, 1.0]];
            let con =
                TrustConstraint::nonlinear(&f, vec![f64::NEG_INFINITY], vec![1.0]).with_jac(&j);
            let options = MinimizeOptions {
                gradient: Some(nl_xy_grad),
                ..base
            };
            (trust_constr_full(&fun, x0, &[con], None, options), None)
        }
        "lin50" => {
            let a: Vec<Vec<f64>> = (0..5)
                .map(|i| {
                    (0..N50)
                        .map(|j| if j / 10 == i { 1.0 } else { 0.0 })
                        .collect()
                })
                .collect();
            let b: Vec<f64> = (1..=5).map(f64::from).collect();
            let con = TrustConstraint::linear(a, b.clone(), b);
            (trust_constr_full(&fun, x0, &[con], None, base), None)
        }
        "lin2side" => {
            let con = TrustConstraint::linear(
                vec![vec![1.0, -2.0], vec![-1.0, -2.0], vec![-1.0, 2.0]],
                vec![f64::NEG_INFINITY, -6.0, -2.0],
                vec![2.0, f64::INFINITY, 2.0],
            );
            (trust_constr_full(&fun, x0, &[con], None, base), None)
        }
        "hs71_exact" => {
            let f1 = |v: &[f64]| vec![v[0] * v[1] * v[2] * v[3]];
            let j1 = |v: &[f64]| {
                vec![vec![
                    v[1] * v[2] * v[3],
                    v[0] * v[2] * v[3],
                    v[0] * v[1] * v[3],
                    v[0] * v[1] * v[2],
                ]]
            };
            let h1 = |v: &[f64], l: &[f64]| {
                let m = [
                    [0.0, v[2] * v[3], v[1] * v[3], v[1] * v[2]],
                    [v[2] * v[3], 0.0, v[0] * v[3], v[0] * v[2]],
                    [v[1] * v[3], v[0] * v[3], 0.0, v[0] * v[1]],
                    [v[1] * v[2], v[0] * v[2], v[0] * v[1], 0.0],
                ];
                m.iter()
                    .map(|row| row.iter().map(|e| l[0] * e).collect())
                    .collect()
            };
            let f2 = |v: &[f64]| vec![v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]];
            let j2 = |v: &[f64]| vec![vec![2.0 * v[0], 2.0 * v[1], 2.0 * v[2], 2.0 * v[3]]];
            let h2 = |_: &[f64], l: &[f64]| {
                let d = l[0] * 2.0;
                (0..4)
                    .map(|i| (0..4).map(|k| if i == k { d } else { 0.0 }).collect())
                    .collect()
            };
            let cons = [
                TrustConstraint::nonlinear(&f1, vec![25.0], vec![f64::INFINITY])
                    .with_jac(&j1)
                    .with_hess(&h1),
                TrustConstraint::nonlinear(&f2, vec![40.0], vec![40.0])
                    .with_jac(&j2)
                    .with_hess(&h2),
            ];
            let bounds = TrustBounds::new(vec![1.0; 4], vec![5.0; 4]);
            let options = MinimizeOptions {
                gradient: Some(hs71_grad),
                hess: Some(hs71_hess),
                ..base
            };
            (
                trust_constr_full(&fun, x0, &cons, Some(&bounds), options),
                None,
            )
        }
        "rosen_keep_feasible" => {
            let bounds =
                TrustBounds::new(vec![-2.0, -2.0], vec![0.5, 2.0]).with_keep_feasible(vec![true]);
            (trust_constr_full(&fun, x0, &[], Some(&bounds), base), None)
        }
        other => panic!("unknown case {other}"),
    }
}

#[derive(Debug, Clone, Serialize)]
struct QueryCase {
    case_id: String,
    x0: Vec<f64>,
    maxiter: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleArm {
    case_id: String,
    x: Option<Vec<f64>>,
    fun: Option<f64>,
    status: Option<u8>,
    nit: Option<usize>,
    nfev: Option<usize>,
    method: Option<String>,
    message: Option<String>,
    constr_violation: Option<f64>,
    v: Option<Vec<Vec<f64>>>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    method: String,
    fsci_status: Option<u8>,
    scipy_status: Option<u8>,
    fsci_nit: usize,
    scipy_nit: usize,
    fsci_nfev: usize,
    scipy_nfev: usize,
    x_rel_diff: f64,
    fun_rel_diff: f64,
    v_rel_diff: f64,
    fsci_constr_violation: f64,
    scipy_constr_violation: f64,
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

const ORACLE_SCRIPT: &str = r#"
import json, math, sys, warnings
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint

def sq(t): return t*t
def rosen(v): return sq(1 - v[0]) + 100*sq(v[1] - v[0]*v[0])
def rosen_grad(v):
    return np.array([-2*(1 - v[0]) - 400*v[0]*(v[1] - v[0]*v[0]), 200*(v[1] - v[0]*v[0])])
def rosen_hessp(v, p):
    h00 = 2 - 400*v[1] + 1200*v[0]*v[0]
    h01 = -400*v[0]
    return np.array([h00*p[0] + h01*p[1], h01*p[0] + 200*p[1]])
def hs71(v): return v[0]*v[3]*(v[0]+v[1]+v[2]) + v[2]
def hs71_grad(v):
    return np.array([v[3]*(2*v[0]+v[1]+v[2]), v[0]*v[3], v[0]*v[3] + 1, v[0]*(v[0]+v[1]+v[2])])
def hs71_hess(v):
    s = 2*v[0]+v[1]+v[2]
    return np.array([[2*v[3], v[3], v[3], s], [v[3], 0, 0, v[0]], [v[3], 0, 0, v[0]], [s, v[0], v[0], 0]])
N50 = 50
def f50(v):
    s2 = 0.0
    s4 = 0.0
    for i in range(N50):
        d = v[i] - i/N50
        s2 = s2 + d*d
        q = v[i]*v[i]
        s4 = s4 + q*q
    return s2 + 0.1*s4
A50 = np.zeros((5, N50))
for i in range(5):
    A50[i, 10*i:10*(i+1)] = 1.0
B50 = np.arange(1, 6, dtype=float)

eq = lambda f, **k: dict(type='eq', fun=f, **k)
ineq = lambda f: dict(type='ineq', fun=f)
hs7_cons = [eq(lambda v: sq(1 + v[0]*v[0]) + v[1]*v[1] - 4)]
hs71_cons = [ineq(lambda v: v[0]*v[1]*v[2]*v[3] - 25),
             eq(lambda v: v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3] - 40)]
def h1(v, l):
    m = np.array([[0, v[2]*v[3], v[1]*v[3], v[1]*v[2]], [v[2]*v[3], 0, v[0]*v[3], v[0]*v[2]],
                  [v[1]*v[3], v[0]*v[3], 0, v[0]*v[1]], [v[1]*v[2], v[0]*v[2], v[0]*v[1], 0]])
    return l[0]*m
hs71_exact_cons = [
    NonlinearConstraint(lambda v: [v[0]*v[1]*v[2]*v[3]], 25, np.inf,
                        jac=lambda v: [[v[1]*v[2]*v[3], v[0]*v[2]*v[3], v[0]*v[1]*v[3], v[0]*v[1]*v[2]]],
                        hess=h1),
    NonlinearConstraint(lambda v: [v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3]], 40, 40,
                        jac=lambda v: [[2*v[0], 2*v[1], 2*v[2], 2*v[3]]],
                        hess=lambda v, l: (l[0]*2)*np.eye(4)),
]
rosen_box = [(-2, 0.5), (-2, 2)]
CASES = {
    "hs6": (lambda v: sq(1 - v[0]), dict(constraints=[eq(lambda v: 10*(v[1] - v[0]*v[0]))])),
    "hs7": (lambda v: math.log(1 + v[0]*v[0]) - v[1], dict(constraints=hs7_cons)),
    "hs7_svd": (lambda v: math.log(1 + v[0]*v[0]) - v[1],
                dict(constraints=hs7_cons, options=dict(factorization_method='SVDFactorization'))),
    "hs71": (hs71, dict(constraints=hs71_cons, bounds=[(1, 5)]*4)),
    "hs71_maxiter3": (hs71, dict(constraints=hs71_cons, bounds=[(1, 5)]*4)),
    "hs76": (lambda v: v[0]*v[0] + 0.5*v[1]*v[1] + v[2]*v[2] + 0.5*v[3]*v[3] - v[0]*v[2]
             + v[2]*v[3] - v[0] - 3*v[1] + v[2] - v[3],
             dict(constraints=[ineq(lambda v: 5 - v[0] - 2*v[1] - v[2] - v[3]),
                               ineq(lambda v: 4 - 3*v[0] - v[1] - 2*v[2] + v[3]),
                               ineq(lambda v: v[1] + 4*v[2] - 1.5)],
                  bounds=[(0, None)]*4)),
    "nl_xy": (lambda v: sq(v[0] - 2) + sq(v[1] - 1),
              dict(jac=lambda v: np.array([2*(v[0] - 2), 2*(v[1] - 1)]),
                   constraints=[NonlinearConstraint(lambda v: v[0] + v[1], -np.inf, 1,
                                                    jac=lambda v: [[1.0, 1.0]])])),
    "eq_only": (lambda v: v[0]*v[0] + v[1]*v[1],
                dict(jac=lambda v: np.array([2*v[0], 2*v[1]]),
                     constraints=[eq(lambda v: v[0] + v[1] - 1, jac=lambda v: [[1.0, 1.0]])])),
    "incompatible_eq": (lambda v: v[0]*v[0] + v[1]*v[1],
                        dict(jac=lambda v: np.array([2*v[0], 2*v[1]]),
                             constraints=[eq(lambda v: v[0] - 1, jac=lambda v: [[1.0, 0.0]]),
                                          eq(lambda v: v[0] - 2, jac=lambda v: [[1.0, 0.0]])])),
    "duplicate_eq": (lambda v: v[0]*v[0] + v[1]*v[1],
                     dict(jac=lambda v: np.array([2*v[0], 2*v[1]]),
                          constraints=[LinearConstraint([[1.0, 1.0], [1.0, 1.0]], [1.0, 1.0], [1.0, 1.0])])),
    "bounds_only": (rosen, dict(bounds=rosen_box)),
    "rosen": (rosen, dict()),
    "rosen_hessp": (rosen, dict(jac=rosen_grad, hessp=rosen_hessp)),
    "rosen_keep_feasible": (rosen, dict(bounds=Bounds([-2, -2], [0.5, 2], keep_feasible=True))),
    "lin50": (f50, dict(constraints=[LinearConstraint(A50, B50, B50)])),
    "lin2side": (lambda v: sq(v[0] - 1) + sq(v[1] - 2.5),
                 dict(constraints=[LinearConstraint([[1, -2], [-1, -2], [-1, 2]],
                                                    [-np.inf, -6, -2], [2, np.inf, 2])])),
    "hs71_exact": (hs71, dict(jac=hs71_grad, hess=hs71_hess, constraints=hs71_exact_cons,
                              bounds=Bounds([1]*4, [5]*4))),
}

out = []
for case in json.load(sys.stdin):
    cid = case["case_id"]
    arm = {"case_id": cid}
    try:
        fun, kw = CASES[cid]
        kw = dict(kw)
        options = dict(kw.pop("options", {}))
        options["maxiter"] = case["maxiter"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = minimize(fun, np.array(case["x0"], dtype=float), method="trust-constr",
                         options=options, **kw)
        arm.update(x=[float(t) for t in r.x], fun=float(r.fun), status=int(r.status),
                   nit=int(r.nit), nfev=int(r.nfev), method=str(r.method),
                   message=str(r.message), constr_violation=float(r.constr_violation),
                   v=[[float(t) for t in np.atleast_1d(vi)] for vi in r.v])
    except Exception as e:
        arm["error"] = repr(e)
    out.append(arm)
print(json.dumps(out))
"#;

fn scipy_oracle_or_skip(query: &[QueryCase]) -> Option<Vec<OracleArm>> {
    let query_json = serde_json::to_string(query).expect("serialize trust-constr query");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(ORACLE_SCRIPT)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for the trust-constr oracle: {e}"
            );
            eprintln!("skipping trust-constr oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open trust-constr oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "trust-constr oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping trust-constr oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for trust-constr oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "trust-constr oracle failed: {stderr}"
        );
        eprintln!("skipping trust-constr oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse trust-constr oracle JSON"))
}

/// Largest `|a - b| / max(|b|, 1)`, NaN-propagating.
fn max_rel_diff(fsci: &[f64], scipy: &[f64]) -> f64 {
    fsci.iter()
        .zip(scipy)
        .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
        .fold(0.0, |worst: f64, d| {
            if worst.is_nan() || d.is_nan() {
                f64::NAN
            } else {
                worst.max(d)
            }
        })
}

#[test]
fn diff_opt_trust_constr_constrained() {
    let cases = cases();
    let query: Vec<QueryCase> = cases
        .iter()
        .map(|c| QueryCase {
            case_id: c.id.to_string(),
            x0: c.x0.clone(),
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

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut ledger = CompareLedger::new("diff_opt_trust_constr_constrained", &[ARM]);
    for case in &cases {
        let arm = &arms[case.id];
        let (fsci, via_minimize) = run_fsci(case);
        let mut diff = CaseDiff {
            case_id: case.id.to_string(),
            method: arm.method.clone().unwrap_or_default(),
            fsci_status: None,
            scipy_status: arm.status,
            fsci_nit: 0,
            scipy_nit: arm.nit.unwrap_or(0),
            fsci_nfev: 0,
            scipy_nfev: arm.nfev.unwrap_or(0),
            x_rel_diff: f64::NAN,
            fun_rel_diff: f64::NAN,
            v_rel_diff: f64::NAN,
            fsci_constr_violation: f64::NAN,
            scipy_constr_violation: arm.constr_violation.unwrap_or(f64::NAN),
            pass: false,
            reason: String::new(),
        };
        match ledger.both(ARM, case.id, arm.status, fsci.as_ref().ok()) {
            None => {
                diff.reason = match (&fsci, &arm.error) {
                    (Err(e), _) => format!("fsci error {e}"),
                    (Ok(_), Some(e)) => format!("SciPy raised {e}"),
                    (Ok(_), None) => "SciPy produced no result".to_string(),
                };
            }
            Some((status, r)) => {
                diff.fsci_status = Some(r.status);
                diff.fsci_nit = r.niter;
                diff.fsci_nfev = r.nfev;
                diff.fsci_constr_violation = r.constr_violation;
                let mut problems = Vec::new();
                let mut recorded = false;
                if Some(r.method.as_str()) != arm.method.as_deref() {
                    problems.push(format!(
                        "method {} vs SciPy {:?}",
                        r.method.as_str(),
                        arm.method
                    ));
                }
                if r.status != status {
                    problems.push(format!("status {} vs SciPy {status}", r.status));
                }
                if Some(r.result.message.as_str()) != arm.message.as_deref() {
                    problems.push(format!(
                        "message '{}' vs SciPy {:?}",
                        r.result.message, arm.message
                    ));
                }
                if r.result.success != matches!(status, 1 | 2) {
                    problems.push(format!(
                        "success {} vs SciPy status {status}",
                        r.result.success
                    ));
                }
                if via_minimize == Some(false) {
                    problems
                        .push("minimize(method=TrustConstr) differs from trust_constr_full".into());
                }
                match ledger.slices(ARM, case.id, arm.x.as_deref(), Some(r.result.x.as_slice())) {
                    None => {
                        recorded = true;
                        problems.push(format!(
                            "x {:?} rejected against SciPy {:?}",
                            r.result.x, arm.x
                        ));
                    }
                    Some((scipy_x, fsci_x)) => {
                        diff.x_rel_diff = max_rel_diff(fsci_x, scipy_x);
                        if diff.x_rel_diff.is_nan() || diff.x_rel_diff > X_REL_TOL {
                            problems.push(format!("x rel diff {:e}", diff.x_rel_diff));
                        }
                    }
                }
                let scipy_fun = arm.fun.unwrap_or(f64::NAN);
                let fsci_fun = r.result.fun.unwrap_or(f64::NAN);
                diff.fun_rel_diff = (fsci_fun - scipy_fun).abs() / scipy_fun.abs().max(1.0);
                if diff.fun_rel_diff.is_nan() || diff.fun_rel_diff > FUN_REL_TOL {
                    problems.push(format!("fun rel diff {:e}", diff.fun_rel_diff));
                }
                if diff.scipy_constr_violation <= MAXCV_TOL
                    && (r.constr_violation.is_nan() || r.constr_violation > MAXCV_TOL)
                {
                    problems.push(format!(
                        "constr_violation {:e} where SciPy's is {:e}",
                        r.constr_violation, diff.scipy_constr_violation
                    ));
                }
                // The least-squares multipliers at the final iterate, compared at every stop (the
                // iteration limit and status 4 included: they are defined wherever x is).
                let scipy_v = arm.v.clone().unwrap_or_default();
                if scipy_v.len() != r.v.len()
                    || scipy_v.iter().zip(&r.v).any(|(s, f)| s.len() != f.len())
                {
                    problems.push(format!("multiplier shapes {:?} vs SciPy {scipy_v:?}", r.v));
                } else {
                    diff.v_rel_diff =
                        r.v.iter()
                            .zip(&scipy_v)
                            .map(|(f, s)| max_rel_diff(f, s))
                            .fold(0.0, |w: f64, d| {
                                if w.is_nan() || d.is_nan() {
                                    f64::NAN
                                } else {
                                    w.max(d)
                                }
                            });
                    if diff.v_rel_diff.is_nan() || diff.v_rel_diff > V_REL_TOL {
                        problems.push(format!("multiplier rel diff {:e}", diff.v_rel_diff));
                    }
                }
                diff.pass = problems.is_empty();
                diff.reason = problems.join("; ");
                if !recorded {
                    ledger.compared(ARM, case.id, diff.pass);
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
        test_id: "diff_opt_trust_constr_constrained".into(),
        category: "scipy.optimize.minimize(method='trust-constr') with constraints and bounds"
            .into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        same_iteration_count,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    fs::create_dir_all(output_dir()).expect("create trust-constr diff dir");
    fs::write(
        output_dir().join("diff_opt_trust_constr_constrained.json"),
        serde_json::to_string_pretty(&log).expect("serialize trust-constr log"),
    )
    .expect("write trust-constr log");

    for d in &diffs {
        println!(
            "{:20} {:24} status {:?}/{:?} nit {}/{} nfev {}/{} dx={:.1e} dfun={:.1e} dv={:.1e} cv={:.1e}/{:.1e} {}",
            d.case_id,
            d.method,
            d.fsci_status,
            d.scipy_status,
            d.fsci_nit,
            d.scipy_nit,
            d.fsci_nfev,
            d.scipy_nfev,
            d.x_rel_diff,
            d.fun_rel_diff,
            d.v_rel_diff,
            d.fsci_constr_violation,
            d.scipy_constr_violation,
            d.reason
        );
    }
    println!(
        "{} cases compared (fsci/SciPy), {same_iteration_count} with SciPy's iteration count",
        diffs.len()
    );
    assert_eq!(diffs.len(), cases.len(), "every case must be compared");
    assert!(
        all_pass,
        "minimize(trust-constr) vs scipy.optimize.minimize(trust-constr) failed"
    );
    ledger.finish(cases.len());
}

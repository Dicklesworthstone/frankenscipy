#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.optimize.linprog` with the HiGHS methods and
//! HiGHS's full result shape (frankenscipy-1ksfv.5).
//!
//! Every LP runs through all three methods on both sides — fsci's `Highs` / `HighsDs` /
//! `HighsIpm` against SciPy's `'highs'` / `'highs-ds'` / `'highs-ipm'` — and every
//! (method, case) pair is one ledger outcome: a SciPy exception or an fsci error is a FAILED
//! case, never a skip.
//!
//! Per pair: `status` must match, and so must `message` (verbatim at an optimum; up to the
//! `primal_status` word otherwise, which reports HiGHS's presolve state, not the LP's). At an
//! optimum:
//! - `fun` to `FUN_RTOL` (relative to `max(|fun|, 1)`).
//! - fsci's answer must satisfy the optimality conditions on its own: primal feasibility,
//!   marginal signs (`ineqlin ≤ 0`, `lower ≥ 0`, `upper ≤ 0`, zero against an infinite bound),
//!   stationarity `c = A_ubᵀ y_ub + A_eqᵀ y_eq + lower + upper`, complementary slackness and
//!   strong duality `|c·x − (b_ub·y_ub + b_eq·y_eq + lb·lower + ub·upper)| ≤ KKT_TOL·(1+|fun|)`.
//! - Every component the LP determines uniquely is compared with SciPy: `x`, `slack`, `con`,
//!   the `lower`/`upper` residuals to `X_RTOL`, and the four `marginals` to `MARGINAL_RTOL`.
//!   Uniqueness is decided by the oracle per component, not assumed: the optimal face is the
//!   feasible set cut down by complementary slackness with SciPy's optimal dual (any optimal
//!   dual gives exactly the optimal face), and a component is unique when HiGHS finds its
//!   minimum and maximum over that face equal; the dual face is built the same way from
//!   SciPy's primal. Components that are not unique (degenerate LPs, several optimal
//!   vertices) are covered by the optimality conditions above, as an optimal answer cannot be
//!   pinned to HiGHS's choice among equals.
//!
//! Elsewhere the solution must be absent on both sides (SciPy returns `None`).

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_opt::{LinprogMethod, LinprogOptions, LinprogResult, linprog};
use serde::Serialize;
use serde_json::{Value, json};

const PACKET_ID: &str = "FSCI-P2C-003";
const FUN_RTOL: f64 = 1.0e-9;
const X_RTOL: f64 = 1.0e-7;
const MARGINAL_RTOL: f64 = 1.0e-7;
const KKT_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

const METHODS: [(&str, LinprogMethod); 3] = [
    ("highs", LinprogMethod::Highs),
    ("highs-ds", LinprogMethod::HighsDs),
    ("highs-ipm", LinprogMethod::HighsIpm),
];

type Bounds = Vec<(Option<f64>, Option<f64>)>;

struct Case {
    id: &'static str,
    c: Vec<f64>,
    a_ub: Vec<Vec<f64>>,
    b_ub: Vec<f64>,
    a_eq: Vec<Vec<f64>>,
    b_eq: Vec<f64>,
    bounds: Bounds,
    maxiter: Option<usize>,
    /// SciPy raises `ValueError` here (malformed input); fsci must refuse too.
    expect_raise: bool,
}

fn lp(
    id: &'static str,
    c: &[f64],
    a_ub: &[&[f64]],
    b_ub: &[f64],
    a_eq: &[&[f64]],
    b_eq: &[f64],
    bounds: Bounds,
) -> Case {
    Case {
        id,
        c: c.to_vec(),
        a_ub: a_ub.iter().map(|row| row.to_vec()).collect(),
        b_ub: b_ub.to_vec(),
        a_eq: a_eq.iter().map(|row| row.to_vec()).collect(),
        b_eq: b_eq.to_vec(),
        bounds,
        maxiter: None,
        expect_raise: false,
    }
}

/// SplitMix64 in `[-1, 1)`: the dense LPs are generated here and sent to SciPy verbatim.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// A seeded dense LP with `m_ub` `≤` rows, `m_eq` equality rows and `n` variables of mixed
/// bound types, feasible by construction (an interior `x0`) and bounded by construction
/// (`c` is a dual-feasible combination of the rows plus sign-correct bound duals).
fn random_dense(id: &'static str, seed: u64, m_ub: usize, m_eq: usize, n: usize) -> Case {
    let mut rng = Rng(seed);
    let a_ub: Vec<Vec<f64>> = (0..m_ub)
        .map(|_| (0..n).map(|_| rng.next()).collect())
        .collect();
    let a_eq: Vec<Vec<f64>> = (0..m_eq)
        .map(|_| (0..n).map(|_| rng.next()).collect())
        .collect();
    let bounds: Bounds = (0..n)
        .map(|j| match j % 5 {
            0 | 1 => (Some(-1.0), Some(1.0)),
            2 => (Some(0.0), None),
            3 => (None, Some(2.0)),
            _ => (Some(-2.0), Some(0.5)),
        })
        .collect();
    let x0: Vec<f64> = (0..n)
        .map(|j| 0.3 * rng.next() + if j % 5 == 2 { 0.5 } else { 0.0 })
        .collect();
    let dot = |row: &[f64], v: &[f64]| row.iter().zip(v).map(|(a, b)| a * b).sum::<f64>();
    let b_ub: Vec<f64> = a_ub
        .iter()
        .map(|row| dot(row, &x0) + 0.5 + 0.5 * rng.next().abs())
        .collect();
    let b_eq: Vec<f64> = a_eq.iter().map(|row| dot(row, &x0)).collect();
    let y_ub: Vec<f64> = (0..m_ub).map(|_| -rng.next().abs()).collect();
    let y_eq: Vec<f64> = (0..m_eq).map(|_| rng.next()).collect();
    let c = (0..n)
        .map(|j| {
            let mut cj = 0.0;
            for i in 0..m_ub {
                cj += a_ub[i][j] * y_ub[i];
            }
            for i in 0..m_eq {
                cj += a_eq[i][j] * y_eq[i];
            }
            match j % 5 {
                2 => cj + rng.next().abs(),
                3 => cj - rng.next().abs(),
                _ => cj + rng.next(),
            }
        })
        .collect();
    Case {
        id,
        c,
        a_ub,
        b_ub,
        a_eq,
        b_eq,
        bounds,
        maxiter: None,
        expect_raise: false,
    }
}

/// Balanced transportation problem with every supply and demand row (one is redundant).
fn transport(
    id: &'static str,
    cost: &[&[f64]],
    supply: &[f64],
    demand: &[f64],
    drop_last: bool,
) -> Case {
    let (ns, nd) = (supply.len(), demand.len());
    let mut a_eq = Vec::new();
    let mut b_eq = Vec::new();
    for (i, &s) in supply.iter().enumerate() {
        let mut row = vec![0.0; ns * nd];
        for j in 0..nd {
            row[i * nd + j] = 1.0;
        }
        a_eq.push(row);
        b_eq.push(s);
    }
    let kept = if drop_last { nd - 1 } else { nd };
    for (j, &d) in demand.iter().enumerate().take(kept) {
        let mut row = vec![0.0; ns * nd];
        for i in 0..ns {
            row[i * nd + j] = 1.0;
        }
        a_eq.push(row);
        b_eq.push(d);
    }
    Case {
        id,
        c: cost.iter().flat_map(|row| row.iter().copied()).collect(),
        a_ub: vec![],
        b_ub: vec![],
        a_eq,
        b_eq,
        bounds: vec![],
        maxiter: None,
        expect_raise: false,
    }
}

fn nonneg(n: usize) -> Bounds {
    vec![(Some(0.0), None); n]
}

fn cases() -> Vec<Case> {
    let mut out = vec![
        // The task's example: ineqlin.marginals = [-1, -0], upper.marginals = [0, -1].
        lp(
            "doc_marginals",
            &[-1.0, -2.0],
            &[&[1.0, 1.0], &[1.0, -1.0]],
            &[4.0, 2.0],
            &[],
            &[],
            vec![(Some(0.0), None), (Some(0.0), Some(3.0))],
        ),
        // SciPy's linprog docstring (a free variable).
        lp(
            "scipy_docstring",
            &[-1.0, 4.0],
            &[&[-3.0, 1.0], &[1.0, 2.0]],
            &[6.0, 4.0],
            &[],
            &[],
            vec![(None, None), (Some(-3.0), None)],
        ),
        lp(
            "canonical_2var",
            &[-1.0, -2.0],
            &[&[1.0, 1.0], &[1.0, 0.0], &[0.0, 1.0]],
            &[4.0, 3.0, 3.0],
            &[],
            &[],
            nonneg(2),
        ),
        lp(
            "eq_ge_3var",
            &[1.0, 2.0, 3.0],
            &[&[-1.0, -1.0, 0.0], &[0.0, 0.0, -1.0]],
            &[-2.0, -1.0],
            &[&[1.0, 1.0, 1.0]],
            &[6.0],
            nonneg(3),
        ),
        lp(
            "diet_4var",
            &[10.0, 15.0, 8.0, 12.0],
            &[&[-2.0, -1.0, -3.0, -1.0], &[-1.0, -3.0, -1.0, -2.0]],
            &[-10.0, -8.0],
            &[],
            &[],
            nonneg(4),
        ),
        lp(
            "ub_bounded_3var",
            &[-3.0, -1.0, -2.0],
            &[&[1.0, 1.0, 1.0], &[2.0, 1.0, 0.0]],
            &[10.0, 12.0],
            &[],
            &[],
            vec![(Some(0.0), Some(5.0)); 3],
        ),
        lp(
            "neg_bounds_2var",
            &[1.0, -2.0],
            &[&[1.0, 1.0], &[-1.0, 1.0]],
            &[6.0, 4.0],
            &[],
            &[],
            vec![(Some(-3.0), Some(3.0)), (Some(0.0), Some(5.0))],
        ),
        lp(
            "eq_only_3var",
            &[1.0, 1.0, 1.0],
            &[],
            &[],
            &[&[1.0, 1.0, 1.0], &[1.0, -1.0, 0.0]],
            &[10.0, 2.0],
            nonneg(3),
        ),
        // Hillier–Lieberman's Wyndor Glass: max 3x + 5y.
        lp(
            "wyndor",
            &[-3.0, -5.0],
            &[&[1.0, 0.0], &[0.0, 2.0], &[3.0, 2.0]],
            &[4.0, 12.0, 18.0],
            &[],
            &[],
            vec![],
        ),
        // Klee–Minty cube, n = 3 (Dantzig's rule visits every vertex).
        lp(
            "klee_minty_3",
            &[-100.0, -10.0, -1.0],
            &[&[1.0, 0.0, 0.0], &[20.0, 1.0, 0.0], &[200.0, 20.0, 1.0]],
            &[1.0, 100.0, 10000.0],
            &[],
            &[],
            vec![],
        ),
        // min t : |x1 − 2| ≤ t, |x2 + 1| ≤ t, x1 + x2 = 3 with x1, x2 free.
        lp(
            "free_minimax",
            &[0.0, 0.0, 1.0],
            &[
                &[1.0, 0.0, -1.0],
                &[-1.0, 0.0, -1.0],
                &[0.0, 1.0, -1.0],
                &[0.0, -1.0, -1.0],
            ],
            &[2.0, -2.0, -1.0, 1.0],
            &[&[1.0, 1.0, 0.0]],
            &[3.0],
            vec![(None, None), (None, None), (Some(0.0), None)],
        ),
        // A free variable with an unbounded optimal face.
        lp(
            "free_nonunique",
            &[1.0, 1.0],
            &[&[-1.0, -1.0]],
            &[-2.0],
            &[],
            &[],
            vec![(None, None), (Some(-1.0), None)],
        ),
        lp(
            "upper_only_bounds",
            &[-1.0, -1.0],
            &[&[1.0, 2.0]],
            &[10.0],
            &[],
            &[],
            vec![(None, Some(5.0)), (None, Some(3.0))],
        ),
        lp(
            "bounds_only",
            &[1.0, -2.0, 0.5],
            &[],
            &[],
            &[],
            &[],
            vec![
                (Some(1.0), Some(3.0)),
                (Some(-2.0), Some(4.0)),
                (Some(-1.0), Some(1.0)),
            ],
        ),
        lp(
            "fixed_vars",
            &[1.0, -1.0, 2.0],
            &[&[1.0, 1.0, 1.0], &[1.0, 0.0, -1.0]],
            &[10.0, 4.0],
            &[],
            &[],
            vec![
                (Some(2.0), Some(2.0)),
                (Some(0.0), None),
                (Some(-1.0), Some(-1.0)),
            ],
        ),
        lp(
            "fixed_negative_dual",
            &[-1.0, 1.0],
            &[&[1.0, 1.0]],
            &[10.0],
            &[],
            &[],
            vec![(Some(2.0), Some(2.0)), (Some(0.0), Some(5.0))],
        ),
        // Three constraints active at (1, 1): the dual is not unique.
        lp(
            "degenerate_vertex",
            &[-1.0, -1.0],
            &[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]],
            &[1.0, 1.0, 2.0],
            &[],
            &[],
            vec![],
        ),
        // A segment of optimal vertices: x is not unique.
        lp(
            "nonunique_segment",
            &[1.0, 1.0],
            &[],
            &[],
            &[&[1.0, 1.0]],
            &[10.0],
            vec![],
        ),
        // Beale (1955): cycles under Dantzig's rule with lowest-index ties.
        lp(
            "beale_cycling",
            &[-0.75, 20.0, -0.5, 6.0],
            &[
                &[0.25, -8.0, -1.0, 9.0],
                &[0.5, -12.0, -0.5, 3.0],
                &[0.0, 0.0, 1.0, 0.0],
            ],
            &[0.0, 0.0, 1.0],
            &[],
            &[],
            vec![],
        ),
        // Kuhn's degenerate example (a cycling example of the textbook simplex).
        lp(
            "kuhn_degenerate",
            &[-2.0, -3.0, 1.0, 12.0],
            &[
                &[-2.0, -9.0, 1.0, 9.0],
                &[1.0 / 3.0, 1.0, -1.0 / 3.0, -2.0],
                &[2.0, 3.0, -1.0, -12.0],
            ],
            &[0.0, 0.0, 2.0],
            &[],
            &[],
            vec![],
        ),
        lp(
            "zero_cost",
            &[0.0, 0.0],
            &[&[1.0, 1.0]],
            &[1.0],
            &[],
            &[],
            vec![],
        ),
        transport(
            "transport_redundant_rows",
            &[&[4.0, 6.0, 9.0], &[5.0, 3.0, 8.0]],
            &[3.0, 5.0],
            &[2.0, 4.0, 2.0],
            false,
        ),
        // The shape fsci-stats builds for the Wasserstein LP (last demand row dropped).
        transport(
            "transport_dropped_row",
            &[&[0.0, 1.0, 2.0], &[1.0, 0.0, 1.0], &[2.0, 1.0, 0.0]],
            &[0.2, 0.5, 0.3],
            &[0.4, 0.4, 0.2],
            true,
        ),
        random_dense("random_dense_30x50", 0x5EED_1A1B, 20, 10, 50),
        random_dense("random_dense_8x12", 0xD15C_0BA1, 6, 2, 12),
        lp(
            "infeasible_eq",
            &[1.0, 1.0],
            &[],
            &[],
            &[&[1.0, 1.0]],
            &[-1.0],
            vec![],
        ),
        lp(
            "infeasible_ineq",
            &[1.0, 1.0],
            &[&[1.0, 1.0], &[-1.0, -1.0]],
            &[1.0, -2.0],
            &[],
            &[],
            vec![],
        ),
        lp(
            "infeasible_crossed_bounds",
            &[1.0, 1.0],
            &[&[1.0, 1.0]],
            &[5.0],
            &[],
            &[],
            vec![(Some(2.0), Some(1.0)), (Some(0.0), None)],
        ),
        lp(
            "model_error_lower_inf",
            &[1.0],
            &[],
            &[],
            &[],
            &[],
            vec![(Some(f64::INFINITY), None)],
        ),
        lp(
            "unbounded_ray",
            &[-1.0, 0.0],
            &[&[1.0, -1.0]],
            &[1.0],
            &[],
            &[],
            vec![],
        ),
        lp(
            "unbounded_free",
            &[1.0, 0.0],
            &[&[0.0, 1.0]],
            &[3.0],
            &[],
            &[],
            vec![(None, None), (Some(0.0), None)],
        ),
    ];
    let mut limited = random_dense("iteration_limit_1", 0x5EED_1A1B, 20, 10, 50);
    limited.maxiter = Some(1);
    out.push(limited);
    let mut c_nan = lp("raise_c_nan", &[1.0, f64::NAN], &[], &[], &[], &[], vec![]);
    c_nan.expect_raise = true;
    out.push(c_nan);
    let mut b_inf = lp(
        "raise_b_ub_inf",
        &[1.0, 1.0],
        &[&[1.0, 1.0]],
        &[f64::INFINITY],
        &[],
        &[],
        vec![],
    );
    b_inf.expect_raise = true;
    out.push(b_inf);
    out
}

/// JSON has no non-finite numbers: they travel as "nan" / "inf" / "-inf".
fn num(v: f64) -> Value {
    if v.is_finite() {
        json!(v)
    } else if v.is_nan() {
        json!("nan")
    } else if v > 0.0 {
        json!("inf")
    } else {
        json!("-inf")
    }
}

fn from_num(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => match s.as_str() {
            "nan" => Some(f64::NAN),
            "inf" => Some(f64::INFINITY),
            "-inf" => Some(f64::NEG_INFINITY),
            _ => None,
        },
        _ => None,
    }
}

/// `None` when SciPy returned `None` (or the field is absent).
fn vec_of(v: &Value) -> Option<Vec<f64>> {
    v.as_array().map(|items| {
        items
            .iter()
            .map(|x| from_num(x).unwrap_or(f64::NAN))
            .collect()
    })
}

fn bools_of(v: &Value) -> Vec<bool> {
    v.as_array()
        .map(|items| items.iter().map(|b| b.as_bool().unwrap_or(false)).collect())
        .unwrap_or_default()
}

fn query_json(cases: &[Case]) -> Value {
    let rows = |rows: &[Vec<f64>]| -> Value {
        Value::Array(
            rows.iter()
                .map(|r| Value::Array(r.iter().map(|&v| num(v)).collect()))
                .collect(),
        )
    };
    let vector = |v: &[f64]| Value::Array(v.iter().map(|&x| num(x)).collect());
    Value::Array(
        cases
            .iter()
            .map(|case| {
                json!({
                    "case_id": case.id,
                    "c": vector(&case.c),
                    "a_ub": rows(&case.a_ub),
                    "b_ub": vector(&case.b_ub),
                    "a_eq": rows(&case.a_eq),
                    "b_eq": vector(&case.b_eq),
                    "bounds": Value::Array(case.bounds.iter().map(|(lo, hi)| {
                        json!([lo.map(num), hi.map(num)])
                    }).collect()),
                    "maxiter": case.maxiter,
                    "uniqueness": !case.expect_raise,
                })
            })
            .collect(),
    )
}

const ORACLE: &str = r#"
import json, math, sys, warnings
import numpy as np
from scipy.optimize import linprog

def dec(v):
    if isinstance(v, str):
        return {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}[v]
    return float(v)

def enc(v):
    v = float(v)
    if math.isfinite(v):
        return v
    return "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")

def encv(a):
    return None if a is None else [enc(v) for v in np.asarray(a, dtype=float).ravel()]

METHODS = ["highs", "highs-ds", "highs-ipm"]
ZERO = 1e-9

def build(case):
    c = np.array([dec(v) for v in case["c"]], dtype=float)
    n = len(c)
    def mat(rows):
        return np.array([[dec(v) for v in r] for r in rows], dtype=float).reshape(-1, n)
    A_ub, A_eq = mat(case["a_ub"]), mat(case["a_eq"])
    b_ub = np.array([dec(v) for v in case["b_ub"]], dtype=float)
    b_eq = np.array([dec(v) for v in case["b_eq"]], dtype=float)
    if case["bounds"]:
        bounds = [(None if lo is None else dec(lo), None if hi is None else dec(hi))
                  for lo, hi in case["bounds"]]
    else:
        bounds = None
    return c, A_ub, b_ub, A_eq, b_eq, bounds

def solve(c, A_ub, b_ub, A_eq, b_eq, bounds, method, options=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return linprog(c, A_ub=A_ub if len(b_ub) else None, b_ub=b_ub if len(b_ub) else None,
                       A_eq=A_eq if len(b_eq) else None, b_eq=b_eq if len(b_eq) else None,
                       bounds=bounds, method=method, options=options or {})

def arm(case, method):
    try:
        c, A_ub, b_ub, A_eq, b_eq, bounds = build(case)
        opts = {} if case["maxiter"] is None else {"maxiter": case["maxiter"]}
        r = solve(c, A_ub, b_ub, A_eq, b_eq, bounds, method, opts)
    except Exception as e:
        return {"raised": f"{type(e).__name__}: {e}"}
    out = {"raised": None, "status": int(r.status), "message": str(r.message),
           "nit": int(r.nit), "fun": None if r.fun is None else enc(r.fun),
           "x": encv(r.x), "slack": encv(r.slack), "con": encv(r.con)}
    for k in ["ineqlin", "eqlin", "lower", "upper"]:
        rec = r.get(k)
        out[k] = {"residual": None if rec is None else encv(rec.residual),
                  "marginals": None if rec is None else encv(rec.marginals)}
    return out

def span(obj, A_ub, b_ub, A_eq, b_eq, bounds):
    lo = solve(obj, A_ub, b_ub, A_eq, b_eq, bounds, "highs")
    hi = solve(-obj, A_ub, b_ub, A_eq, b_eq, bounds, "highs")
    if lo.status != 0 or hi.status != 0:
        return None
    return lo.fun, -hi.fun

def unique(s, value):
    return bool(s is not None and (s[1] - s[0]) <= 1e-9 * max(1.0, abs(value)))

def uniqueness(case):
    """Per-component uniqueness of the primal and dual optimum, from SciPy's 'highs' solve."""
    c, A_ub, b_ub, A_eq, b_eq, bounds = build(case)
    r = solve(c, A_ub, b_ub, A_eq, b_eq, bounds, "highs")
    if r.status != 0:
        return None
    n, mub, meq = len(c), len(b_ub), len(b_eq)
    if bounds is None:
        lb, ub = np.zeros(n), np.full(n, np.inf)
    else:
        lb = np.array([-np.inf if lo is None else lo for lo, _ in bounds], dtype=float)
        ub = np.array([np.inf if hi is None else hi for _, hi in bounds], dtype=float)
    x, slack = r.x, r.slack
    y_ub, y_eq = r.ineqlin.marginals, r.eqlin.marginals
    zl, zu = r.lower.marginals, r.upper.marginals
    dz = ZERO * (1.0 + np.abs(c).max())
    pz = ZERO * (1.0 + max([np.abs(x).max()] + [np.abs(b).max() for b in (b_ub, b_eq) if len(b)]))
    # Primal optimal face: complementary slackness with SciPy's optimal dual.
    act = [i for i in range(mub) if y_ub[i] < -dz]
    ina = [i for i in range(mub) if not y_ub[i] < -dz]
    F_ub, f_ub = A_ub[ina], b_ub[ina]
    F_eq = np.vstack([A_eq, A_ub[act]])
    f_eq = np.concatenate([b_eq, b_ub[act]])
    blo, bhi = lb.copy(), ub.copy()
    for j in range(n):
        if zl[j] > dz:
            bhi[j] = lb[j]
        if zu[j] < -dz:
            blo[j] = ub[j]
    fb = [(None if not np.isfinite(l) else l, None if not np.isfinite(h) else h)
          for l, h in zip(blo, bhi)]
    pspan = lambda obj: span(obj, F_ub, f_ub, F_eq, f_eq, fb)
    eye = np.eye(n)
    x_u = [unique(pspan(eye[j]), x[j]) for j in range(n)]
    slack_u = [unique(pspan(A_ub[i]), slack[i]) for i in range(mub)]
    con_u = [unique(pspan(A_eq[i]), 0.0) for i in range(meq)]
    # Dual optimal face: v = [y_ub, y_eq, zl, zu] with A_ubᵀy_ub + A_eqᵀy_eq + zl + zu = c,
    # sign constraints, and complementary slackness with SciPy's primal x.
    nv = mub + meq + 2 * n
    M = np.hstack([A_ub.T, A_eq.T, np.eye(n), np.eye(n)])
    vb = []
    for i in range(mub):
        vb.append((0.0, 0.0) if slack[i] > pz else (None, 0.0))
    vb += [(None, None)] * meq
    for j in range(n):
        if not np.isfinite(lb[j]) or (x[j] - lb[j] > pz and lb[j] != ub[j]):
            vb.append((0.0, 0.0))
        else:
            vb.append((0.0, None))
    for j in range(n):
        if not np.isfinite(ub[j]) or (ub[j] - x[j] > pz and lb[j] != ub[j]):
            vb.append((0.0, 0.0))
        else:
            vb.append((None, 0.0))
    E = np.zeros((0, nv)); e = np.zeros(0)
    dspan = lambda obj: span(obj, E, e, M, c, vb)
    ev = np.eye(nv)
    ineq_u = [unique(dspan(ev[i]), y_ub[i]) for i in range(mub)]
    eq_u = [unique(dspan(ev[mub + i]), y_eq[i]) for i in range(meq)]
    lower_u, upper_u = [], []
    for j in range(n):
        kl, ku = mub + meq + j, mub + meq + n + j
        if lb[j] == ub[j]:
            # A fixed variable: only the reduced cost zl + zu is determined; HiGHS files it
            # under lower when >= 0 and under upper otherwise.
            u = unique(dspan(ev[kl] + ev[ku]), zl[j] + zu[j])
            lower_u.append(u); upper_u.append(u)
        else:
            lower_u.append(unique(dspan(ev[kl]), zl[j]))
            upper_u.append(unique(dspan(ev[ku]), zu[j]))
    return {"x": x_u, "slack": slack_u, "con": con_u, "ineqlin": ineq_u, "eqlin": eq_u,
            "lower": lower_u, "upper": upper_u}

out = []
for case in json.load(sys.stdin):
    entry = {"case_id": case["case_id"], "arms": {m: arm(case, m) for m in METHODS},
             "unique": None}
    if case["uniqueness"]:
        try:
            entry["unique"] = uniqueness(case)
        except Exception as e:
            entry["unique_error"] = f"{type(e).__name__}: {e}"
    out.append(entry)
print(json.dumps(out))
"#;

fn scipy_oracle_or_skip(query: &Value) -> Option<Vec<Value>> {
    let query_json = serde_json::to_string(query).expect("serialize linprog query");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(ORACLE)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for the linprog oracle: {e}"
            );
            eprintln!("skipping linprog oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open linprog oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "linprog oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping linprog oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for linprog oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "linprog oracle failed: {stderr}"
        );
        eprintln!("skipping linprog oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: Value = serde_json::from_str(&stdout).expect("parse linprog oracle JSON");
    Some(parsed.as_array().expect("oracle list").clone())
}

#[derive(Debug, Clone, Default, Serialize)]
struct CaseDiff {
    case_id: String,
    method: String,
    fsci_status: Option<u8>,
    scipy_status: Option<i64>,
    fsci_message: String,
    scipy_message: String,
    fsci_fun: Option<f64>,
    scipy_fun: Option<f64>,
    fsci_nit: usize,
    /// Simplex iterations after the IPM's crossover (0 for the simplex methods).
    fsci_crossover_nit: usize,
    scipy_nit: Option<i64>,
    /// Unique primal components compared / total (x, slack, con).
    primal_compared: String,
    /// Unique dual components compared / total (ineqlin, eqlin, lower, upper marginals).
    dual_compared: String,
    max_x_rel_diff: f64,
    max_marginal_rel_diff: f64,
    kkt_residual: f64,
    pass: bool,
    reason: String,
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

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1.0)
}

/// The SciPy message with HiGHS's `primal_status` word removed (it reports HiGHS's presolve
/// state, which fsci does not have).
fn message_key(message: &str) -> &str {
    message
        .split_once("; primal_status is")
        .map_or(message, |(head, _)| head)
}

fn bound_pairs(case: &Case) -> Vec<(f64, f64)> {
    if case.bounds.is_empty() {
        return vec![(0.0, f64::INFINITY); case.c.len()];
    }
    case.bounds
        .iter()
        .map(|(l, u)| (l.unwrap_or(f64::NEG_INFINITY), u.unwrap_or(f64::INFINITY)))
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Largest violation of the optimality conditions by fsci's own answer, each term scaled so
/// that 1 means `KKT_TOL` is just met.
fn kkt_violation(case: &Case, r: &LinprogResult, problems: &mut Vec<String>) -> f64 {
    let bounds = bound_pairs(case);
    let n = case.c.len();
    let scale = 1.0 + r.fun.abs();
    let mut worst = 0.0_f64;
    let mut check = |name: String, violation: f64, allowed: f64| {
        let ratio = violation / allowed;
        if !(ratio <= 1.0) {
            problems.push(format!("{name}: {violation:e} > {allowed:e}"));
        }
        worst = worst.max(if ratio.is_nan() { f64::INFINITY } else { ratio });
    };
    for (j, &(l, u)) in bounds.iter().enumerate() {
        let xj = r.x[j];
        check(
            format!("x[{j}] below lb"),
            (l - xj).max(0.0),
            KKT_TOL * (1.0 + l.abs().min(1e300)),
        );
        check(
            format!("x[{j}] above ub"),
            (xj - u).max(0.0),
            KKT_TOL * (1.0 + u.abs().min(1e300)),
        );
        check(
            format!("lower marginal {j} sign"),
            (-r.lower.marginals[j]).max(0.0),
            KKT_TOL,
        );
        check(
            format!("upper marginal {j} sign"),
            r.upper.marginals[j].max(0.0),
            KKT_TOL,
        );
        if l.is_finite() {
            check(
                format!("lower CS {j}"),
                (r.lower.marginals[j] * (xj - l)).abs(),
                KKT_TOL * scale,
            );
        } else {
            // No bound, no marginal: exactly zero.
            check(
                format!("lower marginal {j} at -inf"),
                r.lower.marginals[j].abs(),
                f64::MIN_POSITIVE,
            );
        }
        if u.is_finite() {
            check(
                format!("upper CS {j}"),
                (r.upper.marginals[j] * (u - xj)).abs(),
                KKT_TOL * scale,
            );
        } else {
            check(
                format!("upper marginal {j} at +inf"),
                r.upper.marginals[j].abs(),
                f64::MIN_POSITIVE,
            );
        }
    }
    for (i, row) in case.a_ub.iter().enumerate() {
        let s = case.b_ub[i] - dot(row, &r.x);
        check(
            format!("row {i} slack"),
            (-s).max(0.0),
            KKT_TOL * (1.0 + case.b_ub[i].abs()),
        );
        check(
            format!("ineqlin marginal {i} sign"),
            r.ineqlin.marginals[i].max(0.0),
            KKT_TOL,
        );
        check(
            format!("ineq CS {i}"),
            (r.ineqlin.marginals[i] * s).abs(),
            KKT_TOL * scale,
        );
    }
    for (i, row) in case.a_eq.iter().enumerate() {
        let res = case.b_eq[i] - dot(row, &r.x);
        check(
            format!("eq row {i}"),
            res.abs(),
            KKT_TOL * (1.0 + case.b_eq[i].abs()),
        );
    }
    for j in 0..n {
        let mut grad = r.lower.marginals[j] + r.upper.marginals[j];
        let mut magnitude =
            r.lower.marginals[j].abs() + r.upper.marginals[j].abs() + case.c[j].abs();
        for (i, row) in case.a_ub.iter().enumerate() {
            grad += row[j] * r.ineqlin.marginals[i];
            magnitude += (row[j] * r.ineqlin.marginals[i]).abs();
        }
        for (i, row) in case.a_eq.iter().enumerate() {
            grad += row[j] * r.eqlin.marginals[i];
            magnitude += (row[j] * r.eqlin.marginals[i]).abs();
        }
        check(
            format!("stationarity {j}"),
            (grad - case.c[j]).abs(),
            KKT_TOL * (1.0 + magnitude),
        );
    }
    let mut dual_obj = dot(&case.b_ub, &r.ineqlin.marginals) + dot(&case.b_eq, &r.eqlin.marginals);
    for (j, &(l, u)) in bounds.iter().enumerate() {
        if l.is_finite() {
            dual_obj += l * r.lower.marginals[j];
        }
        if u.is_finite() {
            dual_obj += u * r.upper.marginals[j];
        }
    }
    check(
        "strong duality".into(),
        (dot(&case.c, &r.x) - dual_obj).abs(),
        KKT_TOL * scale,
    );
    check(
        "fun = c·x".into(),
        (dot(&case.c, &r.x) - r.fun).abs(),
        KKT_TOL * scale,
    );
    worst
}

/// Compares fsci against SciPy where the component is unique; returns (compared, total,
/// max relative difference).
fn compare_unique(
    name: &str,
    fsci: &[f64],
    scipy: &[f64],
    unique: &[bool],
    rtol: f64,
    problems: &mut Vec<String>,
) -> (usize, usize, f64) {
    if fsci.len() != scipy.len() {
        problems.push(format!(
            "{name} length {} vs SciPy {}",
            fsci.len(),
            scipy.len()
        ));
        return (0, scipy.len(), f64::INFINITY);
    }
    let mut compared = 0;
    let mut worst = 0.0_f64;
    for (k, (&f, &s)) in fsci.iter().zip(scipy).enumerate() {
        if !unique.get(k).copied().unwrap_or(false) {
            continue;
        }
        compared += 1;
        if s.is_infinite() || f.is_infinite() {
            if f != s {
                problems.push(format!("{name}[{k}] = {f} vs SciPy {s}"));
                worst = f64::INFINITY;
            }
            continue;
        }
        let d = rel(f, s);
        if !(d <= rtol) {
            problems.push(format!("{name}[{k}] = {f:e} vs SciPy {s:e} (rel {d:e})"));
        }
        worst = worst.max(if d.is_nan() { f64::INFINITY } else { d });
    }
    (compared, scipy.len(), worst)
}

fn compare_case(
    case: &Case,
    scipy: &Value,
    unique: Option<&Value>,
    r: &LinprogResult,
    diff: &mut CaseDiff,
) {
    let mut problems = Vec::new();
    let status = scipy["status"].as_i64().unwrap_or(-1);
    let message = scipy["message"].as_str().unwrap_or_default();
    if i64::from(r.status) != status {
        problems.push(format!("status {} vs SciPy {status}", r.status));
    }
    if r.success != (status == 0) {
        problems.push(format!("success {} vs SciPy status {status}", r.success));
    }
    if message_key(&r.message) != message_key(message) {
        problems.push(format!("message '{}' vs SciPy '{message}'", r.message));
    }
    if status == 0 && r.status == 0 {
        let scipy_fun = from_num(&scipy["fun"]).unwrap_or(f64::NAN);
        let d = rel(r.fun, scipy_fun);
        if !(d <= FUN_RTOL) {
            problems.push(format!(
                "fun {:e} vs SciPy {scipy_fun:e} (rel {d:e})",
                r.fun
            ));
        }
        diff.kkt_residual = kkt_violation(case, r, &mut problems);
        let empty = Value::Null;
        let u = unique.unwrap_or(&empty);
        if unique.is_none() {
            problems.push("no uniqueness analysis from the oracle".into());
        }
        let (mut pc, mut pt, mut px) = (0, 0, 0.0_f64);
        let tally = |(c, t, w): (usize, usize, f64),
                     count: &mut usize,
                     total: &mut usize,
                     worst: &mut f64| {
            *count += c;
            *total += t;
            *worst = worst.max(w);
        };
        let field = |name: &str| vec_of(&scipy[name]).unwrap_or_default();
        let sub = |rec: &str, name: &str| vec_of(&scipy[rec][name]).unwrap_or_default();
        let x_u = bools_of(&u["x"]);
        tally(
            compare_unique("x", &r.x, &field("x"), &x_u, X_RTOL, &mut problems),
            &mut pc,
            &mut pt,
            &mut px,
        );
        tally(
            compare_unique(
                "slack",
                &r.slack,
                &field("slack"),
                &bools_of(&u["slack"]),
                X_RTOL,
                &mut problems,
            ),
            &mut pc,
            &mut pt,
            &mut px,
        );
        tally(
            compare_unique(
                "con",
                &r.con,
                &field("con"),
                &bools_of(&u["con"]),
                X_RTOL,
                &mut problems,
            ),
            &mut pc,
            &mut pt,
            &mut px,
        );
        // Bound residuals are x − lb and ub − x: determined exactly where x is.
        tally(
            compare_unique(
                "lower.residual",
                &r.lower.residual,
                &sub("lower", "residual"),
                &x_u,
                X_RTOL,
                &mut problems,
            ),
            &mut 0,
            &mut 0,
            &mut px,
        );
        tally(
            compare_unique(
                "upper.residual",
                &r.upper.residual,
                &sub("upper", "residual"),
                &x_u,
                X_RTOL,
                &mut problems,
            ),
            &mut 0,
            &mut 0,
            &mut px,
        );
        if r.ineqlin.residual != r.slack || r.eqlin.residual != r.con {
            problems.push("ineqlin/eqlin residual differ from slack/con".into());
        }
        let (mut dc, mut dt, mut dx) = (0, 0, 0.0_f64);
        for rec in ["ineqlin", "eqlin", "lower", "upper"] {
            let fsci = match rec {
                "ineqlin" => &r.ineqlin.marginals,
                "eqlin" => &r.eqlin.marginals,
                "lower" => &r.lower.marginals,
                _ => &r.upper.marginals,
            };
            tally(
                compare_unique(
                    &format!("{rec}.marginals"),
                    fsci,
                    &sub(rec, "marginals"),
                    &bools_of(&u[rec]),
                    MARGINAL_RTOL,
                    &mut problems,
                ),
                &mut dc,
                &mut dt,
                &mut dx,
            );
        }
        diff.primal_compared = format!("{pc}/{pt}");
        diff.dual_compared = format!("{dc}/{dt}");
        diff.max_x_rel_diff = px;
        diff.max_marginal_rel_diff = dx;
    } else {
        // No solution on either side.
        if vec_of(&scipy["x"]).is_some() {
            problems.push("SciPy returned x at a non-zero status".into());
        }
        let has_solution = !r.x.is_empty()
            || !r.fun.is_nan()
            || !r.slack.is_empty()
            || !r.con.is_empty()
            || !r.ineqlin.marginals.is_empty()
            || !r.lower.residual.is_empty();
        if has_solution && r.status != 4 {
            problems.push("fsci returned a solution at a non-zero status".into());
        }
    }
    diff.pass = problems.is_empty();
    diff.reason = problems.join("; ");
}

#[test]
fn diff_opt_linprog_highs() {
    let cases = cases();
    let query = query_json(&cases);
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.len(), cases.len(), "one oracle entry per case");
    let entries: HashMap<String, Value> = oracle
        .into_iter()
        .map(|e| (e["case_id"].as_str().expect("case_id").to_string(), e))
        .collect();

    let start = Instant::now();
    let arm_names: Vec<&str> = METHODS.iter().map(|(name, _)| *name).collect();
    let mut ledger = CompareLedger::new("diff_opt_linprog_highs", &arm_names);
    let mut diffs = Vec::new();
    for case in &cases {
        let entry = &entries[case.id];
        if let Some(err) = entry.get("unique_error") {
            println!("{}: uniqueness analysis failed: {err}", case.id);
        }
        let unique = entry.get("unique").filter(|u| !u.is_null());
        for (name, method) in METHODS {
            let scipy = &entry["arms"][name];
            let fsci = linprog(
                &case.c,
                &case.a_ub,
                &case.b_ub,
                &case.a_eq,
                &case.b_eq,
                &case.bounds,
                LinprogOptions {
                    method,
                    maxiter: case.maxiter,
                    ..LinprogOptions::default()
                },
            );
            let raised = scipy["raised"].as_str();
            let mut diff = CaseDiff {
                case_id: case.id.to_string(),
                method: name.to_string(),
                scipy_status: scipy["status"].as_i64(),
                scipy_message: raised.map_or_else(
                    || scipy["message"].as_str().unwrap_or_default().to_string(),
                    str::to_string,
                ),
                scipy_fun: from_num(&scipy["fun"]),
                scipy_nit: scipy["nit"].as_i64(),
                ..CaseDiff::default()
            };
            if let Ok(r) = &fsci {
                diff.fsci_status = Some(r.status);
                diff.fsci_message.clone_from(&r.message);
                diff.fsci_fun = Some(r.fun);
                diff.fsci_nit = r.nit;
                diff.fsci_crossover_nit = r.crossover_nit;
            } else if let Err(e) = &fsci {
                diff.fsci_message = format!("error: {e}");
            }
            if case.expect_raise {
                match raised {
                    Some(_) => {
                        diff.pass = fsci.is_err();
                        diff.reason = if diff.pass {
                            String::new()
                        } else {
                            "SciPy raises; fsci returned a result".into()
                        };
                        ledger.expected_raise(name, case.id, fsci.is_err());
                    }
                    None => {
                        diff.reason = "declared to raise but SciPy returned a result".into();
                        ledger.compared(name, case.id, false);
                    }
                }
                diffs.push(diff);
                continue;
            }
            if let Some(reason) = raised {
                diff.reason = format!("SciPy raised {reason}");
                ledger.oracle_missing(name, case.id, reason);
                diffs.push(diff);
                continue;
            }
            match &fsci {
                Err(e) => {
                    diff.reason = format!("fsci error {e}");
                    ledger.rust_failed(name, case.id, &e.to_string());
                }
                Ok(r) => {
                    compare_case(case, scipy, unique, r, &mut diff);
                    ledger.compared(name, case.id, diff.pass);
                }
            }
            diffs.push(diff);
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_opt_linprog_highs".into(),
        category: "scipy.optimize.linprog(method='highs' | 'highs-ds' | 'highs-ipm'), full result"
            .into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    fs::create_dir_all(output_dir()).expect("create linprog diff dir");
    fs::write(
        output_dir().join("diff_opt_linprog_highs.json"),
        serde_json::to_string_pretty(&log).expect("serialize linprog log"),
    )
    .expect("write linprog log");

    for d in &diffs {
        println!(
            "{:<28} {:<9} status {:?}/{:?} fun {:?}/{:?} nit {}(+{})/{:?} primal {} dual {} \
             dx {:.1e} dmarg {:.1e} kkt {:.2} {} {}",
            d.case_id,
            d.method,
            d.fsci_status,
            d.scipy_status,
            d.fsci_fun,
            d.scipy_fun,
            d.fsci_nit,
            d.fsci_crossover_nit,
            d.scipy_nit,
            d.primal_compared,
            d.dual_compared,
            d.max_x_rel_diff,
            d.max_marginal_rel_diff,
            d.kkt_residual,
            if d.pass { "ok" } else { "FAIL" },
            d.reason
        );
    }
    assert_eq!(
        diffs.len(),
        cases.len() * METHODS.len(),
        "every (case, method) must be compared"
    );
    assert!(all_pass, "linprog vs scipy.optimize.linprog (HiGHS) failed");
    ledger.finish(cases.len());
}

/// Must-hit / must-miss arms for the comparator itself, without SciPy: the SciPy side is the
/// live SciPy 1.17.1 output recorded for two cases (`doc_marginals`, unique everywhere;
/// `degenerate_vertex`, whose `ineqlin` dual is not unique). A comparator too blunt to see a
/// difference would let every case above pass, so each perturbation must be caught, and an
/// equally optimal alternative dual must not be.
#[test]
fn comparator_catches_perturbed_answers() {
    let all = cases();
    let find = |id: &str| all.iter().find(|c| c.id == id).expect("case");
    let solve = |case: &Case| {
        linprog(
            &case.c,
            &case.a_ub,
            &case.b_ub,
            &case.a_eq,
            &case.b_eq,
            &case.bounds,
            LinprogOptions::default(),
        )
        .expect("linprog")
    };
    let verdict = |case: &Case, scipy: &Value, unique: &Value, r: &LinprogResult| {
        let mut diff = CaseDiff::default();
        compare_case(case, scipy, Some(unique), r, &mut diff);
        diff
    };
    let optimal = "Optimization terminated successfully. (HiGHS Status 7: Optimal)";

    // doc_marginals: SciPy x = [1, 3], ineqlin.marginals = [-1, -0], upper.marginals = [0, -1].
    let case = find("doc_marginals");
    let scipy = json!({
        "status": 0, "message": optimal, "fun": -7.0, "x": [1.0, 3.0],
        "slack": [0.0, 4.0], "con": [],
        "ineqlin": {"residual": [0.0, 4.0], "marginals": [-1.0, -0.0]},
        "eqlin": {"residual": [], "marginals": []},
        "lower": {"residual": [1.0, 3.0], "marginals": [0.0, 0.0]},
        "upper": {"residual": ["inf", 0.0], "marginals": [0.0, -1.0]},
    });
    let unique = json!({
        "x": [true, true], "slack": [true, true], "con": [], "ineqlin": [true, true],
        "eqlin": [], "lower": [true, true], "upper": [true, true],
    });
    let good = solve(case);
    let base = verdict(case, &scipy, &unique, &good);
    assert!(base.pass, "unperturbed answer must pass: {}", base.reason);
    assert_eq!(base.primal_compared, "4/4");
    assert_eq!(base.dual_compared, "6/6");

    let mut perturbed: Vec<(&str, LinprogResult)> = Vec::new();
    let mut r = good.clone();
    r.x[0] += 1e-6;
    perturbed.push(("x off by 1e-6", r));
    let mut r = good.clone();
    r.ineqlin.marginals[0] *= 1.0 + 1e-6;
    perturbed.push(("ineqlin marginal off by 1e-6 relative", r));
    let mut r = good.clone();
    r.upper.marginals[1] = 1.0;
    perturbed.push(("upper marginal with the wrong sign", r));
    let mut r = good.clone();
    r.upper.residual[0] = 1e300;
    perturbed.push(("finite residual against SciPy's inf", r));
    let mut r = good.clone();
    r.fun += 1e-7;
    perturbed.push(("fun off by 1e-7", r));
    let mut r = good.clone();
    r.message = "Optimization terminated successfully".to_string();
    perturbed.push(("message without the HiGHS suffix", r));
    let mut r = good.clone();
    r.status = 4;
    perturbed.push(("status 4", r));
    let mut r = good.clone();
    r.lower.marginals.push(0.0);
    perturbed.push(("extra marginal", r));
    for (what, r) in &perturbed {
        let d = verdict(case, &scipy, &unique, r);
        assert!(!d.pass, "{what} must fail the comparison");
    }

    // degenerate_vertex: the dual is not unique; [-0.5, -0.5, -0.5] is another optimal dual.
    let case = find("degenerate_vertex");
    let scipy = json!({
        "status": 0, "message": optimal, "fun": -2.0, "x": [1.0, 1.0],
        "slack": [0.0, 0.0, 0.0], "con": [],
        "ineqlin": {"residual": [0.0, 0.0, 0.0], "marginals": [-1.0, -1.0, -0.0]},
        "eqlin": {"residual": [], "marginals": []},
        "lower": {"residual": [1.0, 1.0], "marginals": [0.0, 0.0]},
        "upper": {"residual": ["inf", "inf"], "marginals": [0.0, 0.0]},
    });
    let unique = json!({
        "x": [true, true], "slack": [true, true, true], "con": [],
        "ineqlin": [false, false, false], "eqlin": [], "lower": [true, true],
        "upper": [true, true],
    });
    let good = solve(case);
    let base = verdict(case, &scipy, &unique, &good);
    assert!(base.pass, "{}", base.reason);
    let mut other_optimum = good.clone();
    other_optimum.ineqlin.marginals = vec![-0.5, -0.5, -0.5];
    let d = verdict(case, &scipy, &unique, &other_optimum);
    assert!(d.pass, "an equally optimal dual must pass: {}", d.reason);
    let mut not_optimal = good.clone();
    not_optimal.ineqlin.marginals = vec![-1.0, -1.0, -0.1];
    let d = verdict(case, &scipy, &unique, &not_optimal);
    assert!(!d.pass, "a dual violating stationarity must fail");
    let mut wrong_sign = good;
    wrong_sign.ineqlin.marginals = vec![-2.0, -2.0, 1.0];
    let d = verdict(case, &scipy, &unique, &wrong_sign);
    assert!(
        !d.pass,
        "a dual with a positive inequality marginal must fail"
    );
}

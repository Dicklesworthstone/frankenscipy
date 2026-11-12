#![forbid(unsafe_code)]
//! Live scipy.optimize.milp parity for fsci_opt::milp.
//!
//! Resolves [frankenscipy-az0ae]. Several small mixed-integer LP
//! problems with continuous/integer/binary and semi-continuous/semi-integer variables.
//!
//! Tolerance: 1e-6 absolute on objective and feasibility. Optimal vectors may
//! differ when the optimum is not unique, but both must satisfy their domains.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_opt::{Integrality, MilpOptions, MilpProblem, milp};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct MilpCase {
    case_id: String,
    c: Vec<f64>,
    /// SciPy codes: 0 = Continuous, 1 = Integer, 2 = SemiContinuous,
    /// 3 = SemiInteger. Code 4 is Rust's Binary extension; the oracle converts
    /// it to Integer and intersects its bounds with [0, 1].
    integrality: Vec<u8>,
    a_ub: Vec<Vec<f64>>,
    b_ub: Vec<f64>,
    a_eq: Vec<Vec<f64>>,
    b_eq: Vec<f64>,
    /// (lo, hi) pairs; None encoded as null.
    bounds: Vec<(Option<f64>, Option<f64>)>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<MilpCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    x: Option<Vec<f64>>,
    fun: Option<f64>,
    success: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    abs_diff: f64,
    scipy_feasible: bool,
    fsci_feasible: bool,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    max_abs_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create milp diff dir");
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

fn single_variable_case(
    case_id: &str,
    integrality: u8,
    cost: f64,
    bounds: (Option<f64>, Option<f64>),
    required: Option<f64>,
    capacity: Option<f64>,
) -> MilpCase {
    let mut a_ub = Vec::new();
    let mut b_ub = Vec::new();
    if let Some(minimum) = required {
        a_ub.push(vec![-1.0]);
        b_ub.push(-minimum);
    }
    if let Some(maximum) = capacity {
        a_ub.push(vec![1.0]);
        b_ub.push(maximum);
    }
    MilpCase {
        case_id: case_id.into(),
        c: vec![cost],
        integrality: vec![integrality],
        a_ub,
        b_ub,
        a_eq: vec![],
        b_eq: vec![],
        bounds: vec![bounds],
    }
}

fn generate_query() -> OracleQuery {
    let mut points = vec![
        // Simple continuous LP: min x+y s.t. x+y>=3 → (3,0) or (0,3); fun=3.
        // Make symmetric-breaking with c=(1, 2): min x+2y s.t. x+y>=3, x,y>=0
        // optimal: (3, 0), fun = 3.
        MilpCase {
            case_id: "lp_2var_simplex".into(),
            c: vec![1.0, 2.0],
            integrality: vec![0, 0],
            a_ub: vec![vec![-1.0, -1.0]], // -x - y <= -3 → x+y >= 3
            b_ub: vec![-3.0],
            a_eq: vec![],
            b_eq: vec![],
            bounds: vec![(Some(0.0), None), (Some(0.0), None)],
        },
        // Pure integer LP: min x+y s.t. x+y>=3, x,y in {0,1,2,3,...}
        // Both integer, optimal (3, 0), fun=3.
        MilpCase {
            case_id: "ilp_2var_integer".into(),
            c: vec![1.0, 2.0],
            integrality: vec![1, 1],
            a_ub: vec![vec![-1.0, -1.0]],
            b_ub: vec![-3.0],
            a_eq: vec![],
            b_eq: vec![],
            bounds: vec![(Some(0.0), None), (Some(0.0), None)],
        },
        // Binary knapsack: max 3x+5y+4z subject to 2x+3y+2z <= 5,
        // x,y,z binary. Maximize ⇒ negate c.
        // Optimal: x=1, y=1, z=0 → val=8. (or x=0,y=1,z=1 → val=9: 0+5+4=9, 0+3+2=5✓)
        // x=0,y=1,z=1: c·x = 0+5+4 = 9.
        MilpCase {
            case_id: "knapsack_binary_3var".into(),
            c: vec![-3.0, -5.0, -4.0],
            integrality: vec![4, 4, 4],
            a_ub: vec![vec![2.0, 3.0, 2.0]],
            b_ub: vec![5.0],
            a_eq: vec![],
            b_eq: vec![],
            bounds: vec![(Some(0.0), Some(1.0)); 3],
        },
        // Mixed: min x+y s.t. x+y=4, x in [0,3], y integer in [0,4]
        // Solutions: (0,4)=4, (1,3)=4, (2,2)=4, (3,1)=4. All same fun=4.
        MilpCase {
            case_id: "mixed_eq".into(),
            c: vec![1.0, 1.0],
            integrality: vec![0, 1],
            a_ub: vec![],
            b_ub: vec![],
            a_eq: vec![vec![1.0, 1.0]],
            b_eq: vec![4.0],
            bounds: vec![(Some(0.0), Some(3.0)), (Some(0.0), Some(4.0))],
        },
        // The convex relaxation uses (5, 1.5, 0), objective 7.25. Enforcing
        // the second activity's minimum gives (3.5, 3, 0), objective 8.
        MilpCase {
            case_id: "mixed_semi_activity".into(),
            c: vec![1.0, 1.5, 3.0],
            integrality: vec![2, 3, 0],
            a_ub: vec![],
            b_ub: vec![],
            a_eq: vec![vec![1.0, 1.0, 1.0]],
            b_eq: vec![6.5],
            bounds: vec![
                (Some(2.0), Some(5.0)),
                (Some(3.0), Some(6.0)),
                (Some(0.0), Some(10.0)),
            ],
        },
        // SciPy code 2 would give 0.5 here. The Binary extension must use
        // code 1 in the oracle, even when wider bounds were supplied.
        single_variable_case(
            "binary_fractional_relaxation",
            4,
            1.0,
            (Some(-5.0), Some(5.0)),
            Some(0.5),
            None,
        ),
    ];
    for (kind, prefix) in [(2, "semi_continuous"), (3, "semi_integer")] {
        for (suffix, cost, bounds, required, capacity) in [
            ("inactive", 1.0, (Some(2.0), Some(5.5)), None, None),
            (
                "activation_gap",
                1.0,
                (Some(2.2), Some(5.5)),
                Some(1.0),
                None,
            ),
            ("active_upper", -1.0, (Some(2.0), Some(5.5)), None, None),
            (
                "zero_branch",
                -1.0,
                (Some(2.0), Some(5.5)),
                None,
                Some(1.0),
            ),
            (
                "active_fractional_relaxation",
                1.0,
                (Some(2.2), Some(8.7)),
                Some(3.5),
                None,
            ),
            (
                "zero_lower",
                1.0,
                (Some(0.0), Some(5.0)),
                Some(0.5),
                None,
            ),
            (
                "empty_active_interval",
                -1.0,
                (Some(5.0), Some(2.0)),
                None,
                None,
            ),
            (
                "fixed_fractional_active",
                -1.0,
                (Some(2.5), Some(2.5)),
                None,
                None,
            ),
            ("infinite_upper", 1.0, (Some(2.0), None), Some(3.5), None),
        ] {
            points.push(single_variable_case(
                &format!("{prefix}_{suffix}"),
                kind,
                cost,
                bounds,
                required,
                capacity,
            ));
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    c = np.array(case["c"], dtype=float)
    kinds = case["integrality"]
    integrality = np.array([1 if kind == 4 else kind for kind in kinds], dtype=int)
    a_ub = case["a_ub"]; b_ub = case["b_ub"]
    a_eq = case["a_eq"]; b_eq = case["b_eq"]
    bounds_in = case["bounds"]
    lb = [b[0] if b[0] is not None else -np.inf for b in bounds_in]
    ub = [b[1] if b[1] is not None else np.inf for b in bounds_in]
    for i, kind in enumerate(kinds):
        if kind == 4:
            lb[i] = max(lb[i], 0)
            ub[i] = min(ub[i], 1)
    bounds = Bounds(lb, ub)
    constraints = []
    if len(a_ub) > 0:
        constraints.append(LinearConstraint(np.array(a_ub, dtype=float), -np.inf, np.array(b_ub, dtype=float)))
    if len(a_eq) > 0:
        constraints.append(LinearConstraint(np.array(a_eq, dtype=float), np.array(b_eq, dtype=float), np.array(b_eq, dtype=float)))
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        if res.success and res.x is not None and all(math.isfinite(v) for v in res.x.tolist()):
            points.append({"case_id": cid, "x": [float(v) for v in res.x.tolist()],
                           "fun": float(res.fun), "success": True})
        else:
            points.append({"case_id": cid, "x": None, "fun": None, "success": False})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "x": None, "fun": None, "success": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
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
                "failed to spawn python3 for milp oracle: {e}"
            );
            eprintln!("skipping milp oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "milp oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping milp oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for milp oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "milp oracle failed: {stderr}"
        );
        eprintln!("skipping milp oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse milp oracle JSON"))
}

fn integrality_from(u: u8) -> Integrality {
    match u {
        0 => Integrality::Continuous,
        1 => Integrality::Integer,
        2 => Integrality::SemiContinuous,
        3 => Integrality::SemiInteger,
        4 => Integrality::Binary,
        _ => panic!("unknown MILP fixture integrality code: {u}"),
    }
}

fn solution_is_feasible(case: &MilpCase, x: &[f64], fun: f64) -> bool {
    if x.len() != case.c.len() || !fun.is_finite() || x.iter().any(|v| !v.is_finite()) {
        return false;
    }
    for (i, &value) in x.iter().enumerate() {
        let kind = case.integrality[i];
        let isolated_zero = matches!(kind, 2 | 3) && value.abs() <= ABS_TOL;
        let (lo, hi) = case.bounds[i];
        if !isolated_zero
            && (value < lo.unwrap_or(f64::NEG_INFINITY) - ABS_TOL
                || value > hi.unwrap_or(f64::INFINITY) + ABS_TOL)
        {
            return false;
        }
        if matches!(kind, 1 | 3 | 4) && (value - value.round()).abs() > ABS_TOL {
            return false;
        }
        if kind == 4 && !(-ABS_TOL..=1.0 + ABS_TOL).contains(&value) {
            return false;
        }
    }
    let dot = |row: &[f64]| row.iter().zip(x).map(|(a, b)| a * b).sum::<f64>();
    (dot(&case.c) - fun).abs() <= ABS_TOL
        && case
            .a_ub
            .iter()
            .zip(&case.b_ub)
            .all(|(row, rhs)| dot(row) <= rhs + ABS_TOL)
        && case
            .a_eq
            .iter()
            .zip(&case.b_eq)
            .all(|(row, rhs)| (dot(row) - rhs).abs() <= ABS_TOL)
}

#[test]
fn diff_opt_milp() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let opts = MilpOptions::default();
    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;
    let mut ledger = CompareLedger::new("diff_opt_milp", &["milp"]);

    for case in &query.points {
        // A missing oracle row is recorded as SciPy giving no value, not skipped.
        let scipy = pmap
            .get(&case.case_id)
            .filter(|arm| arm.success == Some(true));
        let scipy_fun = scipy.and_then(|arm| arm.fun);
        let integrality: Vec<Integrality> = case
            .integrality
            .iter()
            .map(|&u| integrality_from(u))
            .collect();
        let problem = MilpProblem {
            c: &case.c,
            integrality: &integrality,
            a_ub: &case.a_ub,
            b_ub: &case.b_ub,
            a_eq: &case.a_eq,
            b_eq: &case.b_eq,
            bounds: &case.bounds,
        };
        // SciPy reports `fun` only on success, so an unsuccessful fsci solve is a failure.
        let fsci = milp(problem, opts).ok().filter(|res| res.success);
        let fsci_fun = fsci.as_ref().map(|res| res.fun);
        let Some((efun, fsci_fun)) = ledger.pair("milp", &case.case_id, scipy_fun, fsci_fun) else {
            continue;
        };
        // The objective agrees even when degenerate optima have different x,
        // but every returned vector must independently satisfy the true domain.
        let abs_d = (fsci_fun - efun).abs();
        let scipy_feasible = scipy
            .and_then(|arm| arm.x.as_deref())
            .is_some_and(|x| solution_is_feasible(case, x, efun));
        let fsci_feasible = fsci
            .as_ref()
            .is_some_and(|res| solution_is_feasible(case, &res.x, res.fun));
        let pass = abs_d <= ABS_TOL && scipy_feasible && fsci_feasible;
        max_overall = max_overall.max(abs_d);
        ledger.compared("milp", &case.case_id, pass);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            scipy_feasible,
            fsci_feasible,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_milp".into(),
        category: "fsci_opt::milp vs scipy.optimize.milp".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("milp mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "milp conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    ledger.finish(query.points.len());
}

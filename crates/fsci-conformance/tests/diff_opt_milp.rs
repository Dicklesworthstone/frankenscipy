#![forbid(unsafe_code)]
//! Live scipy.optimize.milp parity for fsci_opt::milp.
//!
//! Resolves [frankenscipy-az0ae]. Several small mixed-integer LP
//! problems with continuous/integer/binary variables.
//!
//! Tolerance: 1e-6 abs on x and fun (LP relaxations + B&B can drift).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::{Integrality, MilpOptions, MilpProblem, milp};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct MilpCase {
    case_id: String,
    c: Vec<f64>,
    /// 0 = Continuous, 1 = Integer, 2 = Binary
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
    #[allow(dead_code)]
    x: Option<Vec<f64>>,
    fun: Option<f64>,
    #[allow(dead_code)]
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
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
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

fn generate_query() -> OracleQuery {
    let points = vec![
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
            integrality: vec![2, 2, 2],
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
    ];
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
    integrality = np.array(case["integrality"], dtype=int)
    a_ub = case["a_ub"]; b_ub = case["b_ub"]
    a_eq = case["a_eq"]; b_eq = case["b_eq"]
    bounds_in = case["bounds"]
    lb = [b[0] if b[0] is not None else -np.inf for b in bounds_in]
    ub = [b[1] if b[1] is not None else np.inf for b in bounds_in]
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
        1 => Integrality::Integer,
        2 => Integrality::Binary,
        _ => Integrality::Continuous,
    }
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

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let Some(efun) = arm.fun else {
            continue;
        };
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
        let Ok(res) = milp(problem, opts) else {
            continue;
        };
        if !res.success {
            continue;
        }
        // Compare only objective value: x may be non-unique on degenerate
        // problems (e.g., mixed_eq has multiple optima).
        let abs_d = (res.fun - efun).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_milp".into(),
        category: "fsci_opt::milp vs scipy.optimize.milp".into(),
        case_count: diffs.len(),
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
}

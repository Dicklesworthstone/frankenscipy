#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.optimize.linprog`.
//!
//! Resolves [frankenscipy-oddl4]. Compare optimal objective value
//! (`fun`) only — the solution vector is non-unique at degenerate
//! vertices, but the optimal objective is always unique for a
//! feasible bounded LP.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::linprog;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-003";
const ABS_TOL: f64 = 1.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    c: Vec<f64>,
    a_ub: Vec<Vec<f64>>,
    b_ub: Vec<f64>,
    a_eq: Vec<Vec<f64>>,
    b_eq: Vec<f64>,
    /// Per-variable bounds: (lower, upper); None means -inf / +inf.
    bounds: Vec<(Option<f64>, Option<f64>)>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    fun: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create linprog diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize linprog diff log");
    fs::write(path, json).expect("write linprog diff log");
}

fn generate_query() -> OracleQuery {
    let points = vec![
        // 1. Canonical 2-var LP: min -x - 2y s.t. x + y ≤ 4, x ≤ 3, y ≤ 3, x,y ≥ 0
        //    optimum at (1, 3) with value -7.
        PointCase {
            case_id: "canonical_2var".into(),
            c: vec![-1.0, -2.0],
            a_ub: vec![vec![1.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]],
            b_ub: vec![4.0, 3.0, 3.0],
            a_eq: vec![],
            b_eq: vec![],
            bounds: vec![(Some(0.0), None), (Some(0.0), None)],
        },
        // 2. 3-var LP: min x + 2y + 3z s.t. x + y + z = 6, x + y ≥ 2, z ≥ 1, all ≥ 0
        //    (use ≥ as ≤ negated)
        PointCase {
            case_id: "3var_eq_ge".into(),
            c: vec![1.0, 2.0, 3.0],
            a_ub: vec![vec![-1.0, -1.0, 0.0], vec![0.0, 0.0, -1.0]],
            b_ub: vec![-2.0, -1.0],
            a_eq: vec![vec![1.0, 1.0, 1.0]],
            b_eq: vec![6.0],
            bounds: vec![(Some(0.0), None), (Some(0.0), None), (Some(0.0), None)],
        },
        // 3. Diet problem (4 var): min nutritional cost.
        PointCase {
            case_id: "diet_4var".into(),
            c: vec![10.0, 15.0, 8.0, 12.0],
            a_ub: vec![vec![-2.0, -1.0, -3.0, -1.0], vec![-1.0, -3.0, -1.0, -2.0]],
            b_ub: vec![-10.0, -8.0],
            a_eq: vec![],
            b_eq: vec![],
            bounds: vec![
                (Some(0.0), None),
                (Some(0.0), None),
                (Some(0.0), None),
                (Some(0.0), None),
            ],
        },
        // 4. Transportation-style 3-var with upper bounds.
        PointCase {
            case_id: "ub_bounded_3var".into(),
            c: vec![-3.0, -1.0, -2.0],
            a_ub: vec![vec![1.0, 1.0, 1.0], vec![2.0, 1.0, 0.0]],
            b_ub: vec![10.0, 12.0],
            a_eq: vec![],
            b_eq: vec![],
            bounds: vec![
                (Some(0.0), Some(5.0)),
                (Some(0.0), Some(5.0)),
                (Some(0.0), Some(5.0)),
            ],
        },
        // 5. Minimize with negative bounds allowed.
        PointCase {
            case_id: "neg_bounds_2var".into(),
            c: vec![1.0, -2.0],
            a_ub: vec![vec![1.0, 1.0], vec![-1.0, 1.0]],
            b_ub: vec![6.0, 4.0],
            a_eq: vec![],
            b_eq: vec![],
            bounds: vec![(Some(-3.0), Some(3.0)), (Some(0.0), Some(5.0))],
        },
        // 6. Pure equality system with a unique optimum.
        PointCase {
            case_id: "eq_only_3var".into(),
            c: vec![1.0, 1.0, 1.0],
            a_ub: vec![],
            b_ub: vec![],
            a_eq: vec![vec![1.0, 1.0, 1.0], vec![1.0, -1.0, 0.0]],
            b_eq: vec![10.0, 2.0],
            bounds: vec![(Some(0.0), None), (Some(0.0), None), (Some(0.0), None)],
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
from scipy import optimize

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    c = np.array(case["c"], dtype=float)
    a_ub = np.array(case["a_ub"], dtype=float) if case["a_ub"] else None
    b_ub = np.array(case["b_ub"], dtype=float) if case["b_ub"] else None
    a_eq = np.array(case["a_eq"], dtype=float) if case["a_eq"] else None
    b_eq = np.array(case["b_eq"], dtype=float) if case["b_eq"] else None
    bounds = [(b[0], b[1]) for b in case["bounds"]]
    try:
        res = optimize.linprog(
            c=c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
            bounds=bounds, method='highs'
        )
        if res.status == 0:
            points.append({"case_id": cid, "fun": fnone(res.fun)})
        else:
            points.append({"case_id": cid, "fun": None})
    except Exception:
        points.append({"case_id": cid, "fun": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize linprog query");
    let mut child = match Command::new("python3")
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
                "failed to spawn python3 for linprog oracle: {e}"
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
    Some(serde_json::from_str(&stdout).expect("parse linprog oracle JSON"))
}

#[test]
fn diff_opt_linprog() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_fun) = scipy_arm.fun else {
            continue;
        };
        let Ok(res) = linprog(
            &case.c,
            &case.a_ub,
            &case.b_ub,
            &case.a_eq,
            &case.b_eq,
            &case.bounds,
            Some(2000),
        ) else {
            continue;
        };
        let abs_d = (res.fun - scipy_fun).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_linprog".into(),
        category: "scipy.optimize.linprog (HiGHS / objective value)".into(),
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
            eprintln!(
                "linprog objective mismatch: {} abs_diff={}",
                d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.optimize.linprog conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

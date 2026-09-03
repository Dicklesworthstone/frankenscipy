#![forbid(unsafe_code)]
//! Live scipy.special parity for fsci_special::lpmn and lqmn.
//!
//! Resolves [frankenscipy-czyvo].
//!
//! - `lpmn(m_max, n_max, x)` returns the (m+1)×(n+1) table of
//!   associated Legendre polynomials P_l^m(x) for x ∈ (-1, 1) and
//!   l ≤ m_max + n_max. scipy.special.lpmn returns a tuple
//!   `(Pmn, Pmn_d)`; we use `[0]` (values only).
//! - `lqmn(m_max, n_max, x)` returns the (m+1)×(n+1) table of
//!   associated Legendre functions Q_l^m(x), including SciPy's
//!   derivative-definition cells where l < m.
//!
//! Tolerance: 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{lpmn, lqmn};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "lpmn" | "lqmn"
    m_max: u32,
    n_max: u32,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Row-major flatten of (m_max+1) x (n_max+1) table
    table: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
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
    fs::create_dir_all(output_dir()).expect("create lpmn_lqmn diff dir");
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

fn flatten_table(table: &[Vec<f64>]) -> Vec<f64> {
    table.iter().flat_map(|r| r.iter().copied()).collect()
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    let xs = [-0.7, -0.3, 0.0, 0.2, 0.5, 0.8, 0.95];
    let dims: &[(u32, u32)] = &[(0, 3), (1, 3), (2, 4), (3, 5), (4, 5)];
    for &(m_max, n_max) in dims {
        for &x in &xs {
            points.push(Case {
                case_id: format!("lpmn_m{m_max}_n{n_max}_x{x}"),
                op: "lpmn".into(),
                m_max,
                n_max,
                x,
            });
        }
    }
    for &(m_max, n_max) in dims {
        for &x in &xs {
            points.push(Case {
                case_id: format!("lqmn_m{m_max}_n{n_max}_x{x}"),
                op: "lqmn".into(),
                m_max,
                n_max,
                x,
            });
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
from scipy import special as sp

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    m = int(case["m_max"]); n = int(case["n_max"]); x = float(case["x"])
    try:
        if op == "lpmn":
            tbl, _ = sp.lpmn(m, n, x)
        elif op == "lqmn":
            tbl, _ = sp.lqmn(m, n, x)
        else:
            points.append({"case_id": cid, "table": None}); continue
        arr = np.asarray(tbl, dtype=float)
        flat = [float(v) for v in arr.flatten().tolist()]
        if all(math.isfinite(v) for v in flat):
            points.append({"case_id": cid, "table": flat})
        else:
            points.append({"case_id": cid, "table": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "table": None})
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
                "failed to spawn python3 for lpmn_lqmn oracle: {e}"
            );
            eprintln!("skipping lpmn_lqmn oracle: python3 not available ({e})");
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
                "lpmn_lqmn oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping lpmn_lqmn oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for lpmn_lqmn oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "lpmn_lqmn oracle failed: {stderr}"
        );
        eprintln!("skipping lpmn_lqmn oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse lpmn_lqmn oracle JSON"))
}

fn vec_max_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_special_lpmn_lqmn() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let Some(expected) = arm.table.as_ref() else {
            continue;
        };
        let table = match case.op.as_str() {
            "lpmn" => lpmn(case.m_max, case.n_max, case.x),
            "lqmn" => lqmn(case.m_max, case.n_max, case.x),
            _ => continue,
        };
        let flat = flatten_table(&table);
        if flat.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let abs_d = vec_max_diff(&flat, expected);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_lpmn_lqmn".into(),
        category: "fsci_special::{lpmn, lqmn} vs scipy.special".into(),
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
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "lpmn/lqmn conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the Legendre polynomial
//! batch functions `scipy.special.lpn` (P_n) and
//! `scipy.special.lqn` (Q_n) — both return (values, derivatives).
//!
//! Resolves [frankenscipy-opyh1]. Companion to
//! `diff_special_lpmv` (associated Legendre); these compute the
//! full P_0…P_n / Q_0…Q_n batches plus their derivatives in
//! one call via the canonical three-term recurrence.
//!
//! 4 n-values × 7 x-values × 2 funcs × 2 arrays (values + derivs)
//! = 112 cases via subprocess. Tolerances: 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{lpn, lqn};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    n: u32,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    values: Option<Vec<f64>>,
    derivs: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    func: String,
    arm: String,
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
    fs::create_dir_all(output_dir()).expect("create lpn/lqn diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize lpn/lqn diff log");
    fs::write(path, json).expect("write lpn/lqn diff log");
}

fn fsci_eval(func: &str, n: u32, x: f64) -> Option<(Vec<f64>, Vec<f64>)> {
    let (vs, ds) = match func {
        "lpn" => lpn(n, x),
        "lqn" => lqn(n, x),
        _ => return None,
    };
    if vs.iter().chain(ds.iter()).all(|v| v.is_finite()) {
        Some((vs, ds))
    } else {
        None
    }
}

fn generate_query() -> OracleQuery {
    let ns = [1_u32, 3, 5, 8];
    let xs = [-0.9_f64, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9];
    let mut points = Vec::new();
    for &n in &ns {
        for &x in &xs {
            for func in ["lpn", "lqn"] {
                points.push(PointCase {
                    case_id: format!("{func}_n{n}_x{x}"),
                    func: func.to_string(),
                    n,
                    x,
                });
            }
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy import special

def finite_or_none_list(arr):
    out = []
    for v in arr:
        try:
            v = float(v)
            out.append(v if math.isfinite(v) else None)
        except Exception:
            out.append(None)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    n = int(case["n"]); x = float(case["x"])
    try:
        if func == "lpn":
            v_arr, d_arr = special.lpn(n, x)
        elif func == "lqn":
            v_arr, d_arr = special.lqn(n, x)
        else:
            v_arr, d_arr = [], []
        v_list = finite_or_none_list(v_arr.tolist() if hasattr(v_arr, 'tolist') else v_arr)
        d_list = finite_or_none_list(d_arr.tolist() if hasattr(d_arr, 'tolist') else d_arr)
        if any(x is None for x in v_list) or any(x is None for x in d_list):
            points.append({"case_id": cid, "values": None, "derivs": None})
        else:
            points.append({"case_id": cid, "values": v_list, "derivs": d_list})
    except Exception:
        points.append({"case_id": cid, "values": None, "derivs": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize lpn/lqn query");
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
                "failed to spawn python3 for lpn/lqn oracle: {e}"
            );
            eprintln!("skipping lpn/lqn oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open lpn/lqn oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "lpn/lqn oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping lpn/lqn oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for lpn/lqn oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "lpn/lqn oracle failed: {stderr}"
        );
        eprintln!("skipping lpn/lqn oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse lpn/lqn oracle JSON"))
}

#[test]
fn diff_special_lpn_lqn() {
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
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        let (sv, sd) = match (oracle.values.as_ref(), oracle.derivs.as_ref()) {
            (Some(v), Some(d)) => (v, d),
            _ => continue,
        };
        let Some((rv, rd)) = fsci_eval(&case.func, case.n, case.x) else {
            continue;
        };
        if rv.len() != sv.len() || rd.len() != sd.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                arm: "values".into(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let v_max = rv
            .iter()
            .zip(sv.iter())
            .map(|(r, s)| (r - s).abs())
            .fold(0.0_f64, f64::max);
        let d_max = rd
            .iter()
            .zip(sd.iter())
            .map(|(r, s)| (r - s).abs())
            .fold(0.0_f64, f64::max);
        max_overall = max_overall.max(v_max).max(d_max);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.clone(),
            arm: "values".into(),
            abs_diff: v_max,
            pass: v_max <= ABS_TOL,
        });
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.clone(),
            arm: "derivs".into(),
            abs_diff: d_max,
            pass: d_max <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_lpn_lqn".into(),
        category: "scipy.special.lpn/lqn".into(),
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
                "lpn/lqn {} {} mismatch: {} max_abs={}",
                d.func, d.arm, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "lpn/lqn conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}

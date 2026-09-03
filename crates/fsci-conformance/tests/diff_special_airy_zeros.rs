#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the Airy zeros
//! `scipy.special.ai_zeros` and `scipy.special.bi_zeros`.
//!
//! Resolves [frankenscipy-ge8qb]. fsci-special exposes
//! ai_zeros/bi_zeros as Vec<f64> (just the zeros). scipy
//! returns 4-tuples (zeros, zeros_of_deriv, values_at_zeros_of_deriv,
//! deriv_values_at_zeros); this harness compares only the
//! first element (the zeros themselves).
//!
//! 5 n-counts × 2 funcs = 10 batches via subprocess. Tolerances:
//! 1e-6 abs — Airy precision floor is wider than Bessel.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{ai_zeros, bi_zeros};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    n: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    zeros: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create airy-zeros diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize airy-zeros diff log");
    fs::write(path, json).expect("write airy-zeros diff log");
}

fn fsci_eval(func: &str, n: usize) -> Option<Vec<f64>> {
    let zs = match func {
        "ai_zeros" => ai_zeros(n),
        "bi_zeros" => bi_zeros(n),
        _ => return None,
    };
    if zs.iter().all(|z| z.is_finite()) {
        Some(zs)
    } else {
        None
    }
}

fn generate_query() -> OracleQuery {
    let ns = [1_usize, 3, 5, 8, 10];
    let mut points = Vec::new();
    for &n in &ns {
        for func in ["ai_zeros", "bi_zeros"] {
            points.push(PointCase {
                case_id: format!("{func}_n{n}"),
                func: func.to_string(),
                n,
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
    cid = case["case_id"]; func = case["func"]; n = int(case["n"])
    try:
        if func == "ai_zeros":
            tup = special.ai_zeros(n)
        elif func == "bi_zeros":
            tup = special.bi_zeros(n)
        else:
            tup = ([],)
        # tup[0] is the array of zeros of Ai (or Bi) itself.
        zeros = tup[0].tolist()
        z_list = finite_or_none_list(zeros)
        if any(x is None for x in z_list):
            points.append({"case_id": cid, "zeros": None})
        else:
            points.append({"case_id": cid, "zeros": z_list})
    except Exception:
        points.append({"case_id": cid, "zeros": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize airy-zeros query");
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
                "failed to spawn python3 for airy-zeros oracle: {e}"
            );
            eprintln!("skipping airy-zeros oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open airy-zeros oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "airy-zeros oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping airy-zeros oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for airy-zeros oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "airy-zeros oracle failed: {stderr}"
        );
        eprintln!("skipping airy-zeros oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse airy-zeros oracle JSON"))
}

#[test]
fn diff_special_airy_zeros() {
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
        if let Some(scipy_zs) = oracle.zeros.as_ref()
            && let Some(rust_zs) = fsci_eval(&case.func, case.n)
        {
            if rust_zs.len() != scipy_zs.len() {
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    func: case.func.clone(),
                    abs_diff: f64::INFINITY,
                    pass: false,
                });
                continue;
            }
            let max_abs = rust_zs
                .iter()
                .zip(scipy_zs.iter())
                .map(|(r, s)| (r - s).abs())
                .fold(0.0_f64, f64::max);
            max_overall = max_overall.max(max_abs);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                abs_diff: max_abs,
                pass: max_abs <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_airy_zeros".into(),
        category: "scipy.special.ai_zeros/bi_zeros".into(),
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
                "airy-zeros {} mismatch: {} max_abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "airy-zeros conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}

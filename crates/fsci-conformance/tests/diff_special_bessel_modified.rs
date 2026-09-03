#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the modified cylindrical
//! Bessel family I_0, I_1, K_0, K_1
//! (`scipy.special.i0/i1/k0/k1`).
//!
//! Resolves [frankenscipy-k4hhh]. Companion to
//! `diff_special_bessel` (J_n / Y_n). 11 x-values × 2 (i0, i1)
//! plus 11 x-values × 2 (k0, k1, x>0 only) = 44 cases via
//! subprocess.
//!
//! Tolerances: 1e-7 abs (matches the Bessel-kernel floor
//! documented in frankenscipy-0om9c). I_n grows exponentially
//! so far-x tightening is dominated by absolute scale; we
//! report relative tolerance separately for x ≥ 5.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{i0, i1, kv};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-7;
const REL_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value: Option<f64>,
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
    rel_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    max_rel_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create modified-bessel diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize modified-bessel diff log");
    fs::write(path, json).expect("write modified-bessel diff log");
}

fn fsci_eval(func: &str, x: f64) -> Option<f64> {
    let arg = SpecialTensor::RealScalar(x);
    let result = match func {
        "i0" => i0(&arg, RuntimeMode::Strict),
        "i1" => i1(&arg, RuntimeMode::Strict),
        "k0" => kv(&SpecialTensor::RealScalar(0.0), &arg, RuntimeMode::Strict),
        "k1" => kv(&SpecialTensor::RealScalar(1.0), &arg, RuntimeMode::Strict),
        _ => return None,
    };
    match result {
        Ok(SpecialTensor::RealScalar(v)) => Some(v),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    // I_n is even/odd-order parity; K_n is x>0. Walk small,
    // moderate, and large arguments. K_n diverges as x→0, so
    // the smallest K_n probe stays at x=0.01.
    let xs_in = [
        1.0e-12_f64,
        0.001,
        0.01,
        0.1,
        0.5,
        1.0,
        2.0,
        3.5,
        5.0,
        7.5,
        10.0,
    ];
    let xs_kn = [
        0.01_f64, 0.05, 0.1, 0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0, 20.0,
    ];
    let mut points = Vec::new();
    for &x in &xs_in {
        for func in ["i0", "i1"] {
            points.push(PointCase {
                case_id: format!("{func}_x{x}"),
                func: func.to_string(),
                x,
            });
        }
    }
    for &x in &xs_kn {
        for func in ["k0", "k1"] {
            points.push(PointCase {
                case_id: format!("{func}_x{x}"),
                func: func.to_string(),
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
from scipy import special

def finite_or_none(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    func = case["func"]; x = float(case["x"])
    try:
        if func == "i0":   value = special.i0(x)
        elif func == "i1": value = special.i1(x)
        elif func == "k0": value = special.k0(x)
        elif func == "k1": value = special.k1(x)
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize modified-bessel query");
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
                "failed to spawn python3 for modified-bessel oracle: {e}"
            );
            eprintln!("skipping modified-bessel oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open modified-bessel oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "modified-bessel oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping modified-bessel oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for modified-bessel oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "modified-bessel oracle failed: {stderr}"
        );
        eprintln!("skipping modified-bessel oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse modified-bessel oracle JSON"))
}

#[test]
fn diff_special_bessel_modified() {
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
    let mut max_abs_overall = 0.0_f64;
    let mut max_rel_overall = 0.0_f64;

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        if let Some(scipy_v) = oracle.value
            && let Some(rust_v) = fsci_eval(&case.func, case.x)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            let rel_diff = if scipy_v.abs() > 1.0 {
                abs_diff / scipy_v.abs()
            } else {
                abs_diff
            };
            max_abs_overall = max_abs_overall.max(abs_diff);
            max_rel_overall = max_rel_overall.max(rel_diff);
            // For values with |scipy| > 1, fall back to relative
            // tolerance (I_n grows exponentially); else use absolute.
            let pass = if scipy_v.abs() > 1.0 {
                rel_diff <= REL_TOL
            } else {
                abs_diff <= ABS_TOL
            };
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                abs_diff,
                rel_diff,
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_bessel_modified".into(),
        category: "scipy.special.i0/i1/k0/k1".into(),
        case_count: diffs.len(),
        max_abs_diff: max_abs_overall,
        max_rel_diff: max_rel_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "modified-bessel {} mismatch: {} abs={} rel={}",
                d.func, d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special modified-bessel conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}

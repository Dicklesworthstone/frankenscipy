#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.special.betaincc` and
//! `scipy.special.betainccinv`.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{betaincc, betainccinv};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const TOL_ABS: f64 = 1.0e-10;
const TOL_REL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    a: f64,
    b: f64,
    q: f64,
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

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    fs::create_dir_all(output_dir()).expect("create betaincc diff output dir");
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize betaincc diff log");
    fs::write(path, json).expect("write betaincc diff log");
}

fn fsci_eval(case: &PointCase) -> Option<f64> {
    let a = SpecialTensor::RealScalar(case.a);
    let b = SpecialTensor::RealScalar(case.b);
    let q = SpecialTensor::RealScalar(case.q);
    let result = match case.op.as_str() {
        "betaincc" => betaincc(&a, &b, &q, RuntimeMode::Strict),
        "betainccinv" => betainccinv(&a, &b, &q, RuntimeMode::Strict),
        _ => return None,
    };
    match result {
        Ok(SpecialTensor::RealScalar(value)) => Some(value),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    let pairs = [
        (0.5_f64, 0.5),
        (1.0, 1.0),
        (2.0, 5.0),
        (5.0, 2.0),
        (3.0, 3.0),
        (10.0, 10.0),
    ];
    let xs = [0.0_f64, 0.001, 0.1, 0.5, 0.9, 0.999, 1.0];
    let qs = [0.0_f64, 0.001, 0.1, 0.5, 0.9, 0.999, 1.0];
    let mut points = Vec::new();
    for &(a, b) in &pairs {
        for &x in &xs {
            points.push(PointCase {
                case_id: format!("betaincc_a{a}_b{b}_x{x}"),
                op: "betaincc".into(),
                a,
                b,
                q: x,
            });
        }
        for &q in &qs {
            points.push(PointCase {
                case_id: format!("betainccinv_a{a}_b{b}_q{q}"),
                op: "betainccinv".into(),
                a,
                b,
                q,
            });
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import os
import sys
from scipy import special

def finite_or_none(v):
    try:
        v = float(v)
    except Exception:
        return None
    if math.isfinite(v):
        return v
    return None

query = json.loads(os.environ["FSCI_BETAINCC_QUERY"])
points = []
for case in query["points"]:
    op = case["op"]
    a = float(case["a"])
    b = float(case["b"])
    q = float(case["q"])
    if op == "betaincc":
        value = special.betaincc(a, b, q)
    elif op == "betainccinv":
        value = special.betainccinv(a, b, q)
    else:
        raise ValueError(f"unknown op: {op}")
    points.append({"case_id": case["case_id"], "value": finite_or_none(value)})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize betaincc query");

    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-")
        .env("FSCI_BETAINCC_QUERY", &query_json)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            if std::env::var_os(REQUIRE_SCIPY_ENV).is_some() {
                assert!(
                    std::env::var_os(REQUIRE_SCIPY_ENV).is_none(),
                    "failed to spawn python3 for betaincc oracle: {err}"
                );
            }
            eprintln!("skipping betaincc oracle: python3 not available ({err})");
            return None;
        }
    };

    if let Err(err) = child
        .stdin
        .as_mut()
        .expect("open betaincc oracle stdin")
        .write_all(script.as_bytes())
    {
        let output = child.wait_with_output().expect("wait after stdin failure");
        let stderr = String::from_utf8_lossy(&output.stderr);
        if std::env::var_os(REQUIRE_SCIPY_ENV).is_some() {
            assert!(
                std::env::var_os(REQUIRE_SCIPY_ENV).is_none(),
                "betaincc oracle stdin write failed: {err}; stderr: {stderr}"
            );
        }
        eprintln!("skipping betaincc oracle: stdin write failed ({err})\n{stderr}");
        return None;
    }

    let output = child.wait_with_output().expect("wait for betaincc oracle");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        if std::env::var_os(REQUIRE_SCIPY_ENV).is_some() {
            assert!(
                std::env::var_os(REQUIRE_SCIPY_ENV).is_none(),
                "betaincc oracle failed: {stderr}"
            );
        }
        eprintln!("skipping betaincc oracle: scipy not available\n{stderr}");
        return None;
    }

    Some(serde_json::from_str(&stdout).expect("parse betaincc oracle JSON"))
}

fn close_enough(actual: f64, expected: f64) -> (f64, f64, bool) {
    if actual.is_nan() || expected.is_nan() {
        return (0.0, 0.0, actual.is_nan() && expected.is_nan());
    }
    let abs_diff = (actual - expected).abs();
    let scale = expected.abs().max(1.0);
    let rel_diff = abs_diff / scale;
    (
        abs_diff,
        rel_diff,
        abs_diff <= TOL_ABS || rel_diff <= TOL_REL,
    )
}

#[test]
fn diff_special_betaincc() {
    let started = Instant::now();
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let mut cases = Vec::new();
    let mut max_abs_diff = 0.0_f64;
    let mut max_rel_diff = 0.0_f64;
    for (case, oracle_point) in query.points.iter().zip(oracle.points.iter()) {
        assert_eq!(case.case_id, oracle_point.case_id);
        let actual = fsci_eval(case).unwrap_or(f64::NAN);
        let expected = oracle_point.value.unwrap_or(f64::NAN);
        let (abs_diff, rel_diff, pass) = close_enough(actual, expected);
        max_abs_diff = max_abs_diff.max(abs_diff);
        max_rel_diff = max_rel_diff.max(rel_diff);
        cases.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff,
            rel_diff,
            pass,
        });
    }

    let pass = cases.iter().all(|case| case.pass);
    let log = DiffLog {
        test_id: "diff_special_betaincc".into(),
        category: "scipy.special.betaincc/betainccinv".into(),
        case_count: cases.len(),
        max_abs_diff,
        max_rel_diff,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: started.elapsed().as_nanos(),
        cases,
    };
    emit_log(&log);

    assert!(
        log.pass,
        "betaincc conformance failed: {} cases, max_abs={} max_rel={}",
        log.case_count, log.max_abs_diff, log.max_rel_diff
    );
}

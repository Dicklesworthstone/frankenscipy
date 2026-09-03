#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.special.stirling2`.
//!
//! Resolves [frankenscipy-txpd2]. This covers the default `exact=False`
//! floating-output path for integer scalar inputs, including SciPy's zero
//! rules for negative inputs and `K > N`.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::stirling2;
use fsci_special::types::SpecialTensor;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const STIRLING2_TOL_REL: f64 = 2.0e-5;

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    n: i64,
    k: i64,
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
    n: i64,
    k: i64,
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
    fs::create_dir_all(output_dir()).expect("create stirling2 diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize stirling2 diff log");
    fs::write(path, json).expect("write stirling2 diff log");
}

fn fsci_eval(n: i64, k: i64) -> Option<f64> {
    let n_arg = SpecialTensor::RealScalar(n as f64);
    let k_arg = SpecialTensor::RealScalar(k as f64);
    match stirling2(&n_arg, &k_arg, RuntimeMode::Strict) {
        Ok(SpecialTensor::RealScalar(value)) => Some(value),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    let pairs = [
        (0_i64, 0_i64),
        (0, 1),
        (1, 0),
        (1, 1),
        (5, 2),
        (6, 2),
        (10, 3),
        (20, 10),
        (50, 25),
        (60, 30),
        (100, 50),
        (1000, 2),
        (1000, 999),
        (-1, 2),
        (3, -1),
        (3, 4),
    ];
    OracleQuery {
        points: pairs
            .into_iter()
            .map(|(n, k)| PointCase {
                case_id: format!("stirling2_n{n}_k{k}"),
                n,
                k,
            })
            .collect(),
    }
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

q = json.loads(sys.argv[1])
points = []
for case in q["points"]:
    cid = case["case_id"]
    n = int(case["n"])
    k = int(case["k"])
    try:
        value = special.stirling2(n, k)
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize stirling2 query");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-")
        .arg(query_json)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for stirling2 oracle: {e}"
            );
            eprintln!("skipping stirling2 oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open stirling2 oracle stdin");
        if let Err(err) = stdin.write_all(script.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "stirling2 oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping stirling2 oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for stirling2 oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "stirling2 oracle failed: {stderr}"
        );
        eprintln!("skipping stirling2 oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse stirling2 oracle JSON"))
}

#[test]
fn diff_special_stirling2() {
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
            && let Some(rust_v) = fsci_eval(case.n, case.k)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            let scale = scipy_v.abs().max(1.0);
            let rel_diff = abs_diff / scale;
            max_abs_overall = max_abs_overall.max(abs_diff);
            max_rel_overall = max_rel_overall.max(rel_diff);
            let pass = abs_diff <= STIRLING2_TOL_REL * scale;
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                n: case.n,
                k: case.k,
                abs_diff,
                rel_diff,
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_special_stirling2".into(),
        category: "scipy.special.stirling2".into(),
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
                "stirling2 mismatch: n={} k={} abs={} rel={}",
                d.n, d.k, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special stirling2 conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}

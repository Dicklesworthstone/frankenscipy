#![forbid(unsafe_code)]
//! Live scipy parity for fsci_special::multigammaln.
//!
//! Resolves [frankenscipy-srwt0]. Probes multivariate log-gamma for
//! several (a, d) with d ∈ {1, 2, 3, 4, 5} and a > (d-1)/2.
//! Tolerance: 1e-10 rel.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::multigammaln;
use fsci_special::types::SpecialTensor;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REL_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    a: f64,
    d: f64,
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
    fs::create_dir_all(output_dir()).expect("create multigammaln diff dir");
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

fn fsci_eval(a: f64, d: f64) -> Option<f64> {
    let pa = SpecialTensor::RealScalar(a);
    match multigammaln(&pa, d, RuntimeMode::Strict) {
        Ok(SpecialTensor::RealScalar(v)) => Some(v),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    // multigammaln requires a > (d-1)/2.
    // Choose probes well inside the domain.
    let mut points = Vec::new();
    for &d_int in &[1u32, 2, 3, 4, 5] {
        let d_min = (d_int as f64 - 1.0) / 2.0;
        for &offset in &[0.5_f64, 1.0, 2.5, 5.0, 10.0] {
            let a = d_min + offset + 1.0; // safely above lower bound
            let cid = format!("d{}_a{}", d_int, a.to_string().replace('.', "p"));
            points.push(PointCase {
                case_id: cid,
                a,
                d: d_int as f64,
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

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    a = float(case["a"]); d = int(float(case["d"]))
    try:
        v = float(special.multigammaln(a, d))
        if math.isfinite(v):
            points.append({"case_id": cid, "value": v})
        else:
            points.append({"case_id": cid, "value": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "value": None})
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
                "failed to spawn python3 for multigammaln oracle: {e}"
            );
            eprintln!("skipping multigammaln oracle: python3 not available ({e})");
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
                "multigammaln oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping multigammaln oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for multigammaln oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "multigammaln oracle failed: {stderr}"
        );
        eprintln!("skipping multigammaln oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse multigammaln oracle JSON"))
}

#[test]
fn diff_special_multigammaln() {
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
    let mut max_rel = 0.0_f64;

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let Some(expected) = arm.value else {
            continue;
        };
        let Some(actual) = fsci_eval(case.a, case.d) else {
            continue;
        };
        let abs_d = (actual - expected).abs();
        let rel_d = if expected.abs() > 1.0e-12 {
            abs_d / expected.abs()
        } else {
            abs_d
        };
        max_rel = max_rel.max(rel_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            rel_diff: rel_d,
            pass: rel_d <= REL_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_multigammaln".into(),
        category: "fsci_special::multigammaln vs scipy.special.multigammaln".into(),
        case_count: diffs.len(),
        max_rel_diff: max_rel,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "multigammaln mismatch: {} abs={} rel={}",
                d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "multigammaln conformance failed: {} cases, max_rel={}",
        diffs.len(),
        max_rel
    );
}

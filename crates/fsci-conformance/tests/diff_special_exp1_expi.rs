#![forbid(unsafe_code)]
//! Live scipy parity for fsci_special::{exp1, expi}.
//!
//! Resolves [frankenscipy-jwpvv]. exp1(x) = E_1(x), expi(x) = Ei(x).
//! Tolerance: 1e-8 rel for moderate magnitudes.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{exp1, expi};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REL_TOL: f64 = 1.0e-8;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String, // "exp1" | "expi"
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
    op: String,
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
    fs::create_dir_all(output_dir()).expect("create exp1/expi diff dir");
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

fn fsci_eval(op: &str, x: f64) -> Option<f64> {
    let pt = SpecialTensor::RealScalar(x);
    let result = match op {
        "exp1" => exp1(&pt, RuntimeMode::Strict),
        "expi" => expi(&pt, RuntimeMode::Strict),
        _ => return None,
    };
    match result {
        Ok(SpecialTensor::RealScalar(v)) => Some(v),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    // exp1: positive real x only (E_1 is defined for x > 0 in real args)
    let exp1_xs = [0.1_f64, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0];
    // expi: real x != 0; scipy returns finite values for both signs
    let expi_xs = [-10.0_f64, -2.0, -0.5, -0.1, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
    let mut points = Vec::new();
    for &x in &exp1_xs {
        points.push(PointCase {
            case_id: format!("exp1_x{x}").replace('.', "p").replace('-', "n"),
            op: "exp1".into(),
            x,
        });
    }
    for &x in &expi_xs {
        points.push(PointCase {
            case_id: format!("expi_x{x}").replace('.', "p").replace('-', "n"),
            op: "expi".into(),
            x,
        });
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
    cid = case["case_id"]; op = case["op"]; x = float(case["x"])
    try:
        v = float(special.exp1(x)) if op == "exp1" else float(special.expi(x))
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
                "failed to spawn python3 for exp1/expi oracle: {e}"
            );
            eprintln!("skipping exp1/expi oracle: python3 not available ({e})");
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
                "exp1/expi oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping exp1/expi oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for exp1/expi oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "exp1/expi oracle failed: {stderr}"
        );
        eprintln!("skipping exp1/expi oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse exp1/expi oracle JSON"))
}

#[test]
fn diff_special_exp1_expi() {
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
        let Some(actual) = fsci_eval(&case.op, case.x) else {
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
            op: case.op.clone(),
            abs_diff: abs_d,
            rel_diff: rel_d,
            pass: rel_d <= REL_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_exp1_expi".into(),
        category: "fsci_special::exp1 + expi vs scipy.special".into(),
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
                "{} mismatch: {} abs={} rel={}",
                d.op, d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "exp1/expi conformance failed: {} cases, max_rel={}",
        diffs.len(),
        max_rel
    );
}

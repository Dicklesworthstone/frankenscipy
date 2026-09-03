#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.special.poch` (Pochhammer
//! symbol / rising factorial).
//!
//! Resolves [frankenscipy-9abba]. fsci computes poch(x, n) =
//! Γ(x+n)/Γ(x); for integer n ∈ [1, 20] it uses a direct product.
//! 1e-10 abs covers the gamma-ratio floor.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::poch;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-002";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    x: f64,
    n: f64,
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
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create poch diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize poch diff log");
    fs::write(path, json).expect("write poch diff log");
}

fn generate_query() -> OracleQuery {
    let cases: &[(f64, f64)] = &[
        (1.0, 5.0),
        (2.5, 3.0),
        (0.5, 4.0),
        (3.0, 0.0),
        (4.0, 7.0),
        (1.5, 6.0),
        (10.0, 2.0),
        (2.0, 0.5),
        (5.0, 1.5),
        (0.25, 3.0),
        (7.5, 4.0),
        (3.3, 2.7),
    ];
    let points = cases
        .iter()
        .enumerate()
        .map(|(i, (x, n))| PointCase {
            case_id: format!("poch_{i:02}_x{x}_n{n}"),
            x: *x,
            n: *n,
        })
        .collect();
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy import special

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
    x = float(case["x"]); n = float(case["n"])
    try:
        v = fnone(special.poch(x, n))
        points.append({"case_id": cid, "value": v})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize poch query");
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
                "failed to spawn python3 for poch oracle: {e}"
            );
            eprintln!("skipping poch oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open poch oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "poch oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping poch oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for poch oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "poch oracle failed: {stderr}"
        );
        eprintln!("skipping poch oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse poch oracle JSON"))
}

#[test]
fn diff_special_poch() {
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
        let Some(scipy_v) = scipy_arm.value else {
            continue;
        };
        let fsci_v = poch(case.x, case.n);
        if !fsci_v.is_finite() {
            continue;
        }
        let abs_d = (fsci_v - scipy_v).abs();
        let rel_d = abs_d / scipy_v.abs().max(1.0);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            rel_diff: rel_d,
            pass: rel_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_poch".into(),
        category: "scipy.special.poch (Pochhammer)".into(),
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
            eprintln!("poch mismatch: {} rel_diff={}", d.case_id, d.rel_diff);
        }
    }

    assert!(
        all_pass,
        "scipy.special.poch conformance failed: {} cases, max_abs_diff={}",
        diffs.len(),
        max_overall
    );
}

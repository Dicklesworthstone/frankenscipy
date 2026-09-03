#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.special.polygamma`
//! (orders 0–3: digamma, trigamma, tetragamma, pentagamma).
//!
//! Resolves [frankenscipy-xk49n]. Companion to
//! `diff_special_gamma` (which covers digamma scalar export).
//! polygamma(n, x) is the n-th derivative of digamma; orders >0
//! are used by maximum-likelihood Fisher-information
//! computations and entropy gradients.
//!
//! 4 orders × 11 x-values = 44 cases via subprocess.
//! Tolerances: 1e-8 abs/rel — fsci's digamma already lands
//! ~2.4e-10, but higher orders compose more terms and have
//! wider floors at small/large x.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::polygamma;
use fsci_special::types::SpecialTensor;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-8;
const REL_TOL: f64 = 1.0e-8;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    n: usize,
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
    n: usize,
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
    fs::create_dir_all(output_dir()).expect("create polygamma diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize polygamma diff log");
    fs::write(path, json).expect("write polygamma diff log");
}

fn fsci_eval(n: usize, x: f64) -> Option<f64> {
    let arg = SpecialTensor::RealScalar(x);
    match polygamma(n, &arg, RuntimeMode::Strict) {
        Ok(SpecialTensor::RealScalar(v)) => Some(v),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    let ns = [0_usize, 1, 2, 3];
    // x > 0 (polygamma has poles at 0, -1, -2, …; skip negative).
    let xs = [
        0.1_f64, 0.5, 1.0, 1.5, 2.0, 3.5, 5.0, 10.0, 25.0, 50.0, 100.0,
    ];
    let mut points = Vec::new();
    for &n in &ns {
        for &x in &xs {
            points.push(PointCase {
                case_id: format!("n{n}_x{x}"),
                n,
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
    n = int(case["n"]); x = float(case["x"])
    try:
        value = special.polygamma(n, x)
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize polygamma query");
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
                "failed to spawn python3 for polygamma oracle: {e}"
            );
            eprintln!("skipping polygamma oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open polygamma oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "polygamma oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping polygamma oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for polygamma oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "polygamma oracle failed: {stderr}"
        );
        eprintln!("skipping polygamma oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse polygamma oracle JSON"))
}

#[test]
fn diff_special_polygamma() {
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
            && let Some(rust_v) = fsci_eval(case.n, case.x)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            let rel_diff = if scipy_v.abs() > 1.0 {
                abs_diff / scipy_v.abs()
            } else {
                abs_diff
            };
            max_abs_overall = max_abs_overall.max(abs_diff);
            max_rel_overall = max_rel_overall.max(rel_diff);
            let pass = if scipy_v.abs() > 1.0 {
                rel_diff <= REL_TOL
            } else {
                abs_diff <= ABS_TOL
            };
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                n: case.n,
                abs_diff,
                rel_diff,
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_polygamma".into(),
        category: "scipy.special.polygamma".into(),
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
                "polygamma n={} mismatch: {} abs={} rel={}",
                d.n, d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special.polygamma conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}

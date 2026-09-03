#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the confluent
//! hypergeometric `scipy.special.hyp1f1` ( M(a; b; z) ).
//!
//! Resolves [frankenscipy-i5rur]. fsci-special's hyp1f1 had no
//! dedicated diff harness despite being a known
//! precision-sensitive kernel.
//!
//! Tolerances: 5e-7 rel against scale=max(|scipy|, 1).
//! Confluent hypergeometric has well-known precision gaps near
//! the series-asymptotic seam (|z| around ~30-50 for moderate
//! a, b); the harness uses a wide tolerance documented as
//! coverage rather than a precision claim.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::hyp1f1;
use fsci_special::types::SpecialTensor;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const TOL_REL: f64 = 5.0e-7;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    a: f64,
    b: f64,
    z: f64,
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

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create hyp1f1 diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize hyp1f1 diff log");
    fs::write(path, json).expect("write hyp1f1 diff log");
}

fn fsci_eval(a: f64, b: f64, z: f64) -> Option<f64> {
    let pa = SpecialTensor::RealScalar(a);
    let pb = SpecialTensor::RealScalar(b);
    let pz = SpecialTensor::RealScalar(z);
    match hyp1f1(&pa, &pb, &pz, RuntimeMode::Strict) {
        Ok(SpecialTensor::RealScalar(v)) => Some(v),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    // Pinned (a, b) tuples that span small/moderate a, b > a > 0,
    // and a few classical identities (M(1, 2, z) = (e^z - 1)/z,
    // M(0.5, 1.5, -z²) ∝ erf(z) for z>0).
    let triples: [(f64, f64, f64); 24] = [
        // M(1, 2, z) = (e^z - 1)/z
        (1.0, 2.0, 0.5),
        (1.0, 2.0, 1.0),
        (1.0, 2.0, 5.0),
        (1.0, 2.0, -3.0),
        // M(0.5, 1.5, -z²) related to erf(z)
        (0.5, 1.5, -0.25),
        (0.5, 1.5, -1.0),
        (0.5, 1.5, -4.0),
        // Generic small/moderate
        (0.5, 1.0, 0.3),
        (1.0, 3.0, 1.0),
        (2.0, 4.0, 2.0),
        (3.0, 5.0, 3.0),
        (1.5, 2.5, -1.5),
        (2.5, 3.5, -2.5),
        // Larger a, b
        (5.0, 8.0, 2.0),
        (5.0, 8.0, -2.0),
        (10.0, 15.0, 3.0),
        // Small z
        (1.5, 2.5, 1.0e-3),
        (2.0, 5.0, 1.0e-6),
        // a = b: M(a, a, z) = e^z (when a = b)
        (1.0, 1.0, 0.5),
        (2.0, 2.0, -1.0),
        (3.0, 3.0, 2.0),
        // Moderate negative z
        (1.0, 4.0, -5.0),
        (2.0, 5.0, -10.0),
        (3.0, 6.0, -8.0),
    ];
    let mut points = Vec::new();
    for (i, &(a, b, z)) in triples.iter().enumerate() {
        points.push(PointCase {
            case_id: format!("a{a}_b{b}_z{z}_i{i}"),
            a,
            b,
            z,
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
    a = float(case["a"]); b = float(case["b"]); z = float(case["z"])
    try:
        value = special.hyp1f1(a, b, z)
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize hyp1f1 query");
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
                "failed to spawn python3 for hyp1f1 oracle: {e}"
            );
            eprintln!("skipping hyp1f1 oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open hyp1f1 oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "hyp1f1 oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping hyp1f1 oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for hyp1f1 oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "hyp1f1 oracle failed: {stderr}"
        );
        eprintln!("skipping hyp1f1 oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse hyp1f1 oracle JSON"))
}

#[test]
fn diff_special_hyp1f1() {
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
            && let Some(rust_v) = fsci_eval(case.a, case.b, case.z)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            let scale = scipy_v.abs().max(1.0);
            let rel_diff = abs_diff / scale;
            max_abs_overall = max_abs_overall.max(abs_diff);
            max_rel_overall = max_rel_overall.max(rel_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                abs_diff,
                rel_diff,
                pass: abs_diff <= TOL_REL * scale,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_hyp1f1".into(),
        category: "scipy.special.hyp1f1".into(),
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
                "hyp1f1 mismatch: {} abs={} rel={}",
                d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special.hyp1f1 conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}

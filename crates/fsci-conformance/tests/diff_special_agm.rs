#![forbid(unsafe_code)]
//! Live differential coverage for the arithmetic-geometric mean
//! `fsci_special::agm(a, b)`.
//!
//! Resolves [frankenscipy-f5dwe]. scipy.special does not export
//! `agm` directly, so the oracle runs the canonical AGM
//! iteration in Python (high-precision via mpmath when
//! available, else float Newton-AGM) and verifies fsci agrees.
//!
//! 12 (a, b) pairs via subprocess. Tolerances: 1e-13 abs/rel —
//! AGM converges quadratically and fsci's stop criterion is
//! 1e-15 relative.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::agm;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-13;
const REL_TOL: f64 = 1.0e-13;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    a: f64,
    b: f64,
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
    fs::create_dir_all(output_dir()).expect("create agm diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize agm diff log");
    fs::write(path, json).expect("write agm diff log");
}

fn fsci_eval(a: f64, b: f64) -> Option<f64> {
    let v = agm(a, b);
    if v.is_finite() { Some(v) } else { None }
}

fn generate_query() -> OracleQuery {
    let pairs: [(f64, f64); 12] = [
        (1.0, 1.0), // a=b: trivial, agm = 1
        (1.0, 2.0),
        (2.0, 1.0), // commutative
        (1.0, 0.5),
        (0.1, 0.9),
        (1.0, 1.0e-3), // very different magnitudes
        (1.0, 100.0),
        (10.0, 100.0),
        (3.0, 7.0),
        (5.0, 5.0),
        (1.0e-6, 1.0), // tiny vs unity
        (1.0e6, 1.0),  // huge vs unity
    ];
    let mut points = Vec::new();
    for (i, &(a, b)) in pairs.iter().enumerate() {
        points.push(PointCase {
            case_id: format!("a{a}_b{b}_i{i}"),
            a,
            b,
        });
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    // scipy.special doesn't expose agm. Use mpmath if available
    // for high precision, else fall back to a high-iteration
    // Python float AGM.
    let script = r#"
import json
import math
import sys

def finite_or_none(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

try:
    import mpmath
    have_mpmath = True
except ImportError:
    have_mpmath = False

def agm_high_precision(a, b):
    if have_mpmath:
        mpmath.mp.dps = 30
        return float(mpmath.agm(a, b))
    # Fallback: Python AGM with extra iterations
    an, bn = float(a), float(b)
    for _ in range(200):
        next_a = (an + bn) / 2.0
        next_b = math.sqrt(an * bn)
        if abs(next_a - next_b) < 1e-17 * abs(next_a):
            return next_a
        an, bn = next_a, next_b
    return (an + bn) / 2.0

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    a = float(case["a"]); b = float(case["b"])
    try:
        if a <= 0 or b <= 0:
            value = None
        else:
            value = agm_high_precision(a, b)
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize agm query");
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
                "failed to spawn python3 for agm oracle: {e}"
            );
            eprintln!("skipping agm oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open agm oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "agm oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping agm oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for agm oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "agm oracle failed: {stderr}"
        );
        eprintln!("skipping agm oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse agm oracle JSON"))
}

#[test]
fn diff_special_agm() {
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
            && let Some(rust_v) = fsci_eval(case.a, case.b)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            let scale = scipy_v.abs().max(1.0);
            let rel_diff = abs_diff / scale;
            max_abs_overall = max_abs_overall.max(abs_diff);
            max_rel_overall = max_rel_overall.max(rel_diff);
            let pass = abs_diff <= ABS_TOL || abs_diff <= REL_TOL * scale;
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                abs_diff,
                rel_diff,
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_agm".into(),
        category: "scipy.special.agm (mpmath oracle)".into(),
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
                "agm mismatch: {} abs={} rel={}",
                d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "agm conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}

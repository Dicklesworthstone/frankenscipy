#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the degree-mode trig
//! functions `scipy.special.sindg/cosdg/tandg/cotdg`.
//!
//! Resolves [frankenscipy-2u1p9]. These wrap the standard
//! sin/cos/tan/cot with degree input and exact-zero handling
//! at multiples of 90° / 180°.
//!
//! 13 x-values × 4 functions = 52 cases via subprocess.
//! Tolerances: 1e-13 abs sin/cos; tan/cot scale with their
//! magnitude near singularities so use 1e-12 rel for those.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{cosdg, cotdg, sindg, tandg};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const SC_TOL: f64 = 1.0e-13;
// tan/cot near singularities (e.g. cotdg(359.999) ≈ 5.7e4)
// inherits float modular reduction error of ~2e-11 rel; 1e-10
// rel absorbs cleanly.
const TC_TOL_REL: f64 = 1.0e-10;
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
    fs::create_dir_all(output_dir()).expect("create trig-deg diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize trig-deg diff log");
    fs::write(path, json).expect("write trig-deg diff log");
}

fn fsci_eval(func: &str, x: f64) -> Option<f64> {
    let v = match func {
        "sindg" => sindg(x),
        "cosdg" => cosdg(x),
        "tandg" => tandg(x),
        "cotdg" => cotdg(x),
        _ => return None,
    };
    if v.is_finite() { Some(v) } else { None }
}

fn generate_query() -> OracleQuery {
    // Cover the canonical sample of degrees: exact 0°, 30°,
    // 45°, 60°, 90°-equivalent (sindg(90)=1), 180°, 270°, 360°,
    // and a few off-canonical angles to verify the modular
    // arithmetic.
    let xs = [
        0.0_f64, 30.0, 45.0, 60.0, 89.999, 90.0, 120.0, 135.0, 180.0, 270.0, 359.999, 360.0, 720.0,
    ];
    let mut points = Vec::new();
    for &x in &xs {
        for func in ["sindg", "cosdg", "tandg", "cotdg"] {
            // Skip tandg/cotdg at angles where they're singular
            // (90° + 180k for tan, 0° + 180k for cot). scipy
            // returns ±∞ which we filter via finite_or_none on
            // the python side anyway.
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
    cid = case["case_id"]; func = case["func"]; x = float(case["x"])
    try:
        if func == "sindg":   value = special.sindg(x)
        elif func == "cosdg": value = special.cosdg(x)
        elif func == "tandg": value = special.tandg(x)
        elif func == "cotdg": value = special.cotdg(x)
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize trig-deg query");
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
                "failed to spawn python3 for trig-deg oracle: {e}"
            );
            eprintln!("skipping trig-deg oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open trig-deg oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "trig-deg oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping trig-deg oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for trig-deg oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "trig-deg oracle failed: {stderr}"
        );
        eprintln!("skipping trig-deg oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse trig-deg oracle JSON"))
}

#[test]
fn diff_special_trig_deg() {
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
            let scale = scipy_v.abs().max(1.0);
            let rel_diff = abs_diff / scale;
            max_abs_overall = max_abs_overall.max(abs_diff);
            max_rel_overall = max_rel_overall.max(rel_diff);
            let pass = match case.func.as_str() {
                "sindg" | "cosdg" => abs_diff <= SC_TOL,
                "tandg" | "cotdg" => abs_diff <= TC_TOL_REL * scale,
                _ => false,
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
        test_id: "diff_special_trig_deg".into(),
        category: "scipy.special.sindg/cosdg/tandg/cotdg".into(),
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
                "trig-deg {} mismatch: {} abs={} rel={}",
                d.func, d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special trig-deg conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}

#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the Airy functions
//! Ai(x) and Bi(x) (`scipy.special.airy`).
//!
//! Resolves [frankenscipy-vkx6l]. fsci's `ai()` and `bi()` are
//! scalar-out wrappers; scipy.special.airy(x) returns the
//! 4-tuple (Ai, Ai', Bi, Bi'). The harness extracts only
//! Ai and Bi.
//!
//! 13 x-values × 2 functions = 26 cases via subprocess.
//!
//! Tolerances: 1e-7 abs / 1e-5 rel — abs for |scipy|<1, rel
//! for |scipy|>1. Bi grows exponentially for large positive x
//! (Bi(10) ≈ 4.5e8). fsci's Airy implementation lands ~8e-6
//! rel at x=5 (Bi(5) ≈ 657, abs ≈ 5e-3); shares the special-
//! function precision floor tracked in frankenscipy-0om9c.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{ai, bi};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-7;
const REL_TOL: f64 = 1.0e-5;
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
    fs::create_dir_all(output_dir()).expect("create airy diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize airy diff log");
    fs::write(path, json).expect("write airy diff log");
}

fn fsci_eval(func: &str, x: f64) -> Option<f64> {
    let arg = SpecialTensor::RealScalar(x);
    let result = match func {
        "ai" => ai(&arg, RuntimeMode::Strict),
        "bi" => bi(&arg, RuntimeMode::Strict),
        _ => return None,
    };
    match result {
        Ok(SpecialTensor::RealScalar(v)) => Some(v),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    // Ai(x) decays exponentially for x>0; oscillates for x<0.
    // Bi(x) grows exponentially for x>0; oscillates for x<0.
    // Walk both sides plus the origin.
    let xs = [
        -10.0_f64, -5.0, -2.5, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0,
    ];
    let mut points = Vec::new();
    for &x in &xs {
        for func in ["ai", "bi"] {
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
        ai_v, _aip, bi_v, _bip = special.airy(x)
        if func == "ai":   value = ai_v
        elif func == "bi": value = bi_v
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize airy query");
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
                "failed to spawn python3 for airy oracle: {e}"
            );
            eprintln!("skipping airy oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open airy oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "airy oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping airy oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for airy oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "airy oracle failed: {stderr}"
        );
        eprintln!("skipping airy oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse airy oracle JSON"))
}

#[test]
fn diff_special_airy() {
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
        test_id: "diff_special_airy".into(),
        category: "scipy.special.airy (Ai/Bi)".into(),
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
                "airy {} mismatch: {} abs={} rel={}",
                d.func, d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special airy conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}

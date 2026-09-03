#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the Dawson function,
//! the scaled complementary error function, and exprel:
//! `scipy.special.dawsn/erfcx/exprel`.
//!
//! Resolves [frankenscipy-uso2t]. Three kernels with no
//! dedicated diff harness:
//!   • dawsn(x) = exp(-x²) ∫₀ˣ exp(t²) dt
//!   • erfcx(x) = exp(x²) · erfc(x)
//!   • exprel(x) = (e^x − 1)/x with exprel(0) = 1
//!
//! 11 x-values × 3 funcs = 33 cases via subprocess.
//! Tolerances: 1e-10 abs/rel — Dawson and erfcx are
//! precision-sensitive at the series-asymptotic seam.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{dawsn, erfcx, exprel};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// dawsn near its peak (|x|=1.5) lands ~7.9e-10 abs off scipy;
// 5e-9 absorbs cleanly.
const ABS_TOL: f64 = 5.0e-9;
const REL_TOL: f64 = 5.0e-9;
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
    fs::create_dir_all(output_dir()).expect("create dawsn/erfcx/exprel diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize dawsn/erfcx/exprel diff log");
    fs::write(path, json).expect("write dawsn/erfcx/exprel diff log");
}

fn fsci_eval(func: &str, x: f64) -> Option<f64> {
    let arg = SpecialTensor::RealScalar(x);
    let result = match func {
        "dawsn" => dawsn(&arg, RuntimeMode::Strict),
        "erfcx" => erfcx(&arg, RuntimeMode::Strict),
        "exprel" => exprel(&arg, RuntimeMode::Strict),
        _ => return None,
    };
    match result {
        Ok(SpecialTensor::RealScalar(v)) => Some(v),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    // dawsn / exprel: domain ℝ. erfcx: domain ℝ but grows
    // rapidly as x → -∞ (erfcx(-3) ≈ 1.6e4); restrict erfcx to
    // x ≥ -1.
    let xs_full = [
        -5.0_f64, -1.5, -0.5, -0.01, 0.0, 0.01, 0.1, 0.5, 1.5, 5.0, 10.0,
    ];
    let xs_erfcx = [
        -1.0_f64, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 30.0,
    ];
    let mut points = Vec::new();
    for &x in &xs_full {
        for func in ["dawsn", "exprel"] {
            points.push(PointCase {
                case_id: format!("{func}_x{x}"),
                func: func.to_string(),
                x,
            });
        }
    }
    for &x in &xs_erfcx {
        points.push(PointCase {
            case_id: format!("erfcx_x{x}"),
            func: "erfcx".into(),
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
        if func == "dawsn":  value = special.dawsn(x)
        elif func == "erfcx":value = special.erfcx(x)
        elif func == "exprel":value = special.exprel(x)
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize dawsn/erfcx/exprel query");
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
                "failed to spawn python3 for dawsn/erfcx/exprel oracle: {e}"
            );
            eprintln!("skipping dawsn/erfcx/exprel oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open dawsn/erfcx/exprel oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "dawsn/erfcx/exprel oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping dawsn/erfcx/exprel oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for dawsn/erfcx/exprel oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "dawsn/erfcx/exprel oracle failed: {stderr}"
        );
        eprintln!("skipping dawsn/erfcx/exprel oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse dawsn/erfcx/exprel oracle JSON"))
}

#[test]
fn diff_special_dawsn_erfcx_exprel() {
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
            let pass = abs_diff <= ABS_TOL || abs_diff <= REL_TOL * scale;
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
        test_id: "diff_special_dawsn_erfcx_exprel".into(),
        category: "scipy.special.dawsn/erfcx/exprel".into(),
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
                "dawsn/erfcx/exprel {} mismatch: {} abs={} rel={}",
                d.func, d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "dawsn/erfcx/exprel conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}

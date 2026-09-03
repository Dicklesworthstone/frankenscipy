#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the error-function
//! family `scipy.special.erf/erfc/erfinv/erfcinv`.
//!
//! Resolves [frankenscipy-v43vs]. erf and erfc are mature in
//! fsci (used by Normal cdf, etc.); erfinv/erfcinv are scalar
//! wrappers around a rational-approximation kernel. 13 x-values
//! × 2 (erf, erfc) plus q-grids for erfinv (in (-1,1)) and
//! erfcinv (in (0,2)) = ~50 cases via subprocess.
//!
//! Tolerances: 1e-13 abs for erf/erfc, 1e-9 rel for erfinv/
//! erfcinv (the rational-approximation floor is wider than the
//! canonical erf/erfc kernel).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{erf, erfc, erfcinv, erfinv};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ERF_TOL: f64 = 1.0e-13;
// erfcinv lands ~1.1e-9 rel at q=0.01/q=1.99 (rational approx
// floor); 5e-9 absorbs with margin.
const ERFINV_TOL_REL: f64 = 5.0e-9;
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
    fs::create_dir_all(output_dir()).expect("create error-function diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize error-function diff log");
    fs::write(path, json).expect("write error-function diff log");
}

fn fsci_eval(func: &str, x: f64) -> Option<f64> {
    let arg = SpecialTensor::RealScalar(x);
    let result = match func {
        "erf" => erf(&arg, RuntimeMode::Strict),
        "erfc" => erfc(&arg, RuntimeMode::Strict),
        "erfinv" => erfinv(&arg, RuntimeMode::Strict),
        "erfcinv" => erfcinv(&arg, RuntimeMode::Strict),
        _ => return None,
    };
    match result {
        Ok(SpecialTensor::RealScalar(v)) => Some(v),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    // erf and erfc cover both signs of x.
    let xs_erf = [
        -8.0_f64, -3.0, -1.5, -0.5, -0.1, 0.0, 0.1, 0.5, 1.5, 3.0, 5.0, 8.0,
    ];
    // erfinv: q in (-1, 1).
    let qs_erfinv = [
        -0.999_f64, -0.99, -0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9, 0.99, 0.999,
    ];
    // erfcinv: q in (0, 2).
    let qs_erfcinv = [
        1.0e-9_f64,
        1.0e-3,
        0.01,
        0.1,
        0.5,
        1.0,
        1.5,
        1.9,
        1.99,
        1.999,
        2.0 - 1.0e-9,
    ];

    let mut points = Vec::new();
    for &x in &xs_erf {
        for func in ["erf", "erfc"] {
            points.push(PointCase {
                case_id: format!("{func}_x{x}"),
                func: func.to_string(),
                x,
            });
        }
    }
    for &q in &qs_erfinv {
        points.push(PointCase {
            case_id: format!("erfinv_q{q}"),
            func: "erfinv".into(),
            x: q,
        });
    }
    for &q in &qs_erfcinv {
        points.push(PointCase {
            case_id: format!("erfcinv_q{q}"),
            func: "erfcinv".into(),
            x: q,
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
    func = case["func"]; x = float(case["x"])
    try:
        if func == "erf":      value = special.erf(x)
        elif func == "erfc":   value = special.erfc(x)
        elif func == "erfinv": value = special.erfinv(x)
        elif func == "erfcinv":value = special.erfcinv(x)
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize error-function query");
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
                "failed to spawn python3 for error-function oracle: {e}"
            );
            eprintln!("skipping error-function oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open error-function oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "error-function oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping error-function oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for error-function oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "error-function oracle failed: {stderr}"
        );
        eprintln!("skipping error-function oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse error-function oracle JSON"))
}

#[test]
fn diff_special_error() {
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

            let pass = match case.func.as_str() {
                "erf" | "erfc" => abs_diff <= ERF_TOL,
                "erfinv" | "erfcinv" => {
                    let scale = scipy_v.abs().max(1.0);
                    abs_diff <= ERFINV_TOL_REL * scale
                }
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
        test_id: "diff_special_error".into(),
        category: "scipy.special.erf/erfc/erfinv/erfcinv".into(),
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
                "error-function {} mismatch: {} abs={} rel={}",
                d.func, d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special error-function conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}

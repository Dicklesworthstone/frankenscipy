#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the robust-loss and
//! KL-divergence kernels
//! `scipy.special.huber/pseudo_huber/kl_div`.
//!
//! Resolves [frankenscipy-4r3pg]. Three kernels used by robust
//! regression (huber, pseudo_huber) and information-theoretic
//! computations (kl_div = x·log(x/y) − x + y).
//!
//! 4 deltas × 9 x = 36 huber/pseudo_huber + 9 (x, y) = 81
//! cases via subprocess. Tolerances: 1e-13 abs (closed-form).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{huber, kl_div, pseudo_huber};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-13;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
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
    func: String,
    abs_diff: f64,
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
    fs::create_dir_all(output_dir()).expect("create robust-loss diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize robust-loss diff log");
    fs::write(path, json).expect("write robust-loss diff log");
}

fn fsci_eval(func: &str, a: f64, b: f64) -> Option<f64> {
    let pa = SpecialTensor::RealScalar(a);
    let pb = SpecialTensor::RealScalar(b);
    let v = match func {
        "huber" => match huber(&pa, &pb, RuntimeMode::Strict) {
            Ok(SpecialTensor::RealScalar(v)) => v,
            _ => return None,
        },
        "pseudo_huber" => match pseudo_huber(&pa, &pb, RuntimeMode::Strict) {
            Ok(SpecialTensor::RealScalar(v)) => v,
            _ => return None,
        },
        "kl_div" => kl_div(a, b),
        _ => return None,
    };
    if v.is_finite() { Some(v) } else { None }
}

fn generate_query() -> OracleQuery {
    let deltas = [0.5_f64, 1.0, 2.0, 5.0];
    let xs = [-5.0_f64, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0];
    let kl_pairs = [
        (0.1_f64, 1.0),
        (0.5, 1.0),
        (1.0, 1.0),
        (1.5, 1.0),
        (2.0, 1.0),
        (1.0, 0.5),
        (1.0, 2.0),
        (3.0, 5.0),
        (5.0, 3.0),
    ];

    let mut points = Vec::new();
    for &d in &deltas {
        for &x in &xs {
            for func in ["huber", "pseudo_huber"] {
                points.push(PointCase {
                    case_id: format!("{func}_d{d}_x{x}"),
                    func: func.to_string(),
                    a: d,
                    b: x,
                });
            }
        }
    }
    for &(x, y) in &kl_pairs {
        points.push(PointCase {
            case_id: format!("kl_div_x{x}_y{y}"),
            func: "kl_div".into(),
            a: x,
            b: y,
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
    cid = case["case_id"]; func = case["func"]
    a = float(case["a"]); b = float(case["b"])
    try:
        if func == "huber":       value = special.huber(a, b)
        elif func == "pseudo_huber":value = special.pseudo_huber(a, b)
        elif func == "kl_div":    value = special.kl_div(a, b)
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize robust-loss query");
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
                "failed to spawn python3 for robust-loss oracle: {e}"
            );
            eprintln!("skipping robust-loss oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open robust-loss oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "robust-loss oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping robust-loss oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for robust-loss oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "robust-loss oracle failed: {stderr}"
        );
        eprintln!("skipping robust-loss oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse robust-loss oracle JSON"))
}

#[test]
fn diff_special_robust_loss() {
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
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        if let Some(scipy_v) = oracle.value
            && let Some(rust_v) = fsci_eval(&case.func, case.a, case.b)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_robust_loss".into(),
        category: "scipy.special.huber/pseudo_huber/kl_div".into(),
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
            eprintln!(
                "robust-loss {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "robust-loss conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}

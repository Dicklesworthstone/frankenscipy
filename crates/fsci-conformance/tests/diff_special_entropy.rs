#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the entropy/sigmoid
//! kernels `scipy.special.expit/logit/entr/rel_entr/xlogy`.
//!
//! Resolves [frankenscipy-ee2ur]. Five fundamental kernels with
//! no dedicated diff harness:
//!   • expit(x) = 1/(1+e^{-x})  — used by Logistic.cdf / sigmoid
//!   • logit(p) = log(p/(1-p))  — used by Logistic.ppf
//!   • entr(x)  = -x·log(x)     — entropy
//!   • rel_entr(x, y) = x·log(x/y) — KL kernel
//!   • xlogy(x, y) = x·log(y)   — entropy/likelihood gradient
//!
//! Tolerances: 1e-13 abs (closed-form fundamental kernels).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{entr, expit, logit, rel_entr, xlogy};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-13;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    x: f64,
    y: f64,
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
    fs::create_dir_all(output_dir()).expect("create entropy diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize entropy diff log");
    fs::write(path, json).expect("write entropy diff log");
}

fn fsci_eval(func: &str, x: f64, y: f64) -> Option<f64> {
    let px = SpecialTensor::RealScalar(x);
    let py = SpecialTensor::RealScalar(y);
    let result = match func {
        "expit" => expit(&px, RuntimeMode::Strict),
        "logit" => logit(&px, RuntimeMode::Strict),
        "entr" => entr(&px, RuntimeMode::Strict),
        "rel_entr" => rel_entr(&px, &py, RuntimeMode::Strict),
        "xlogy" => xlogy(&px, &py, RuntimeMode::Strict),
        _ => return None,
    };
    match result {
        Ok(SpecialTensor::RealScalar(v)) => Some(v),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    // expit: domain ℝ
    let xs_expit = [-15.0_f64, -5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0, 15.0];
    // logit: domain (0, 1)
    let ps_logit = [0.001_f64, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999];
    // entr: domain x ≥ 0
    let xs_entr = [0.0_f64, 1.0e-6, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0];
    // rel_entr / xlogy: (x, y) with x ≥ 0, y > 0
    let xy_pairs = [
        (0.0_f64, 1.0),
        (0.1, 0.5),
        (0.5, 0.5), // identical
        (0.5, 1.5),
        (1.0, 0.5),
        (1.0, 1.0),
        (1.0, 2.0),
        (3.0, 0.5),
        (5.0, 5.0),
    ];

    let mut points = Vec::new();
    for &x in &xs_expit {
        points.push(PointCase {
            case_id: format!("expit_x{x}"),
            func: "expit".into(),
            x,
            y: 0.0,
        });
    }
    for &p in &ps_logit {
        points.push(PointCase {
            case_id: format!("logit_p{p}"),
            func: "logit".into(),
            x: p,
            y: 0.0,
        });
    }
    for &x in &xs_entr {
        points.push(PointCase {
            case_id: format!("entr_x{x}"),
            func: "entr".into(),
            x,
            y: 0.0,
        });
    }
    for &(x, y) in &xy_pairs {
        for func in ["rel_entr", "xlogy"] {
            points.push(PointCase {
                case_id: format!("{func}_x{x}_y{y}"),
                func: func.to_string(),
                x,
                y,
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
    func = case["func"]; x = float(case["x"]); y = float(case["y"])
    try:
        if func == "expit":     value = special.expit(x)
        elif func == "logit":   value = special.logit(x)
        elif func == "entr":    value = special.entr(x)
        elif func == "rel_entr":value = special.rel_entr(x, y)
        elif func == "xlogy":   value = special.xlogy(x, y)
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize entropy query");
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
                "failed to spawn python3 for entropy oracle: {e}"
            );
            eprintln!("skipping entropy oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open entropy oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "entropy oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping entropy oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for entropy oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "entropy oracle failed: {stderr}"
        );
        eprintln!("skipping entropy oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse entropy oracle JSON"))
}

#[test]
fn diff_special_entropy() {
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
            && let Some(rust_v) = fsci_eval(&case.func, case.x, case.y)
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
        test_id: "diff_special_entropy".into(),
        category: "scipy.special.expit/logit/entr/rel_entr/xlogy".into(),
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
                "entropy {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special entropy/sigmoid conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}

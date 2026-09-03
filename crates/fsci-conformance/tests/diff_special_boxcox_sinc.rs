#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the Box-Cox transform
//! and cardinal sine kernels
//! `scipy.special.boxcox/boxcox1p/inv_boxcox/sinc`.
//!
//! Resolves [frankenscipy-0gei0]. Three commonly-used kernels:
//!   • boxcox1p(x, λ) = ((1+x)^λ − 1)/λ  (continuous at λ=0)
//!   • inv_boxcox(y, λ) = (λy + 1)^(1/λ) (continuous at λ=0)
//!   • sinc(x) = sin(πx)/(πx) with sinc(0)=1
//!
//! Tolerances: 1e-13 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{boxcox_transform, boxcox1p, inv_boxcox, sinc};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-13;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    arg1: f64,
    arg2: f64,
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
    fs::create_dir_all(output_dir()).expect("create boxcox/sinc diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize boxcox/sinc diff log");
    fs::write(path, json).expect("write boxcox/sinc diff log");
}

fn fsci_eval(func: &str, arg1: f64, arg2: f64) -> Option<f64> {
    let p1 = SpecialTensor::RealScalar(arg1);
    let p2 = SpecialTensor::RealScalar(arg2);
    let result = match func {
        "boxcox" => boxcox_transform(&p1, &p2, RuntimeMode::Strict),
        "boxcox1p" => boxcox1p(&p1, &p2, RuntimeMode::Strict),
        "inv_boxcox" => inv_boxcox(&p1, &p2, RuntimeMode::Strict),
        "sinc" => sinc(&p1, RuntimeMode::Strict),
        _ => return None,
    };
    match result {
        Ok(SpecialTensor::RealScalar(v)) => Some(v),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    // boxcox1p / inv_boxcox: λ ∈ {-1, -0.5, 0, 0.5, 1, 2}, x or y values.
    let lams = [-1.0_f64, -0.5, 0.0, 0.5, 1.0, 2.0];
    let xs = [0.0_f64, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
    // sinc input — both signs of x.
    let xs_sinc = [
        -10.0_f64, -3.5, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 3.5, 10.0,
    ];

    let mut points = Vec::new();
    for &lam in &lams {
        for &x in &xs {
            // boxcox(x, λ) defined for x > 0; skip x=0 since the
            // standard boxcox(0, λ) = NaN (boxcox1p(0, λ) = 0).
            if x > 0.0 {
                points.push(PointCase {
                    case_id: format!("boxcox_lam{lam}_x{x}"),
                    func: "boxcox".into(),
                    arg1: x,
                    arg2: lam,
                });
            }
            points.push(PointCase {
                case_id: format!("boxcox1p_lam{lam}_x{x}"),
                func: "boxcox1p".into(),
                arg1: x,
                arg2: lam,
            });
            // inv_boxcox: y range — pick reasonable values. For
            // λ < 0, y is bounded; for λ ≥ 0, any y works. Use
            // small-magnitude y values.
            let y = (x - 1.0).max(-0.5);
            points.push(PointCase {
                case_id: format!("inv_boxcox_lam{lam}_y{y}"),
                func: "inv_boxcox".into(),
                arg1: y,
                arg2: lam,
            });
        }
    }
    for &x in &xs_sinc {
        points.push(PointCase {
            case_id: format!("sinc_x{x}"),
            func: "sinc".into(),
            arg1: x,
            arg2: 0.0,
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
import numpy as np

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
    a1 = float(case["arg1"]); a2 = float(case["arg2"])
    try:
        if func == "boxcox":     value = special.boxcox(a1, a2)
        elif func == "boxcox1p": value = special.boxcox1p(a1, a2)
        elif func == "inv_boxcox":value = special.inv_boxcox(a1, a2)
        elif func == "sinc":     value = float(np.sinc(a1))
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize boxcox/sinc query");
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
                "failed to spawn python3 for boxcox/sinc oracle: {e}"
            );
            eprintln!("skipping boxcox/sinc oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open boxcox/sinc oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "boxcox/sinc oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping boxcox/sinc oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for boxcox/sinc oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "boxcox/sinc oracle failed: {stderr}"
        );
        eprintln!("skipping boxcox/sinc oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse boxcox/sinc oracle JSON"))
}

#[test]
fn diff_special_boxcox_sinc() {
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
            && let Some(rust_v) = fsci_eval(&case.func, case.arg1, case.arg2)
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
        test_id: "diff_special_boxcox_sinc".into(),
        category: "scipy.special.boxcox/boxcox1p/inv_boxcox/sinc".into(),
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
                "boxcox/sinc {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special boxcox/sinc conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}

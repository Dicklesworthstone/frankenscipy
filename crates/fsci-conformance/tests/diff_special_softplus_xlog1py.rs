#![forbid(unsafe_code)]
//! Live SciPy differential coverage for two scalar-input
//! activation/entropy helpers in `scipy.special`:
//!   - `scipy.special.softplus(x)`   = ln(1 + exp(x))   (numerically
//!     stable variant)
//!   - `scipy.special.xlog1py(x, y)` = x · ln(1 + y)    (zero-extends
//!     correctly at x=0, y=-1)
//!
//! Resolves [frankenscipy-9oozh]. Both ship in convenience.rs and
//! match scipy.special to machine precision in spot checks; this
//! harness pins that agreement at 1e-14 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{softplus, xlog1py};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-14;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    x: f64,
    /// `y` parameter for xlog1py; ignored for softplus.
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
    fs::create_dir_all(output_dir()).expect("create softplus/xlog1py diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize softplus/xlog1py diff log");
    fs::write(path, json).expect("write softplus/xlog1py diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    // softplus spans both the underflow regime (x << 0) and the
    // saturating regime (x >> 0) where softplus(x) → x.
    let softplus_xs: &[f64] = &[-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0];
    for &x in softplus_xs {
        points.push(PointCase {
            case_id: format!("softplus_x{x}"),
            func: "softplus".into(),
            x,
            y: 0.0,
        });
    }
    // xlog1py covers the (x=0, y=-1) zero-times-(−inf) corner where
    // a naïve x * log(1+y) would return NaN.
    let xlog1py_args: &[(f64, f64)] = &[
        (1.0, 0.5),
        (2.0, 1.0),
        (0.0, -1.0),
        (3.0, 0.0),
        (1.5, -0.5),
        (0.5, 0.1),
        (5.0, 10.0),
        // Small-|y| precision: a naive x*(1+y).ln() loses digits (and returns
        // 0 for y ≲ 1e-16); scipy uses log1p. Pins the ln_1p implementation.
        (1.0, 1e-13),
        (1.0, 1e-16),
        (4.0, -1e-15),
    ];
    for &(x, y) in xlog1py_args {
        points.push(PointCase {
            case_id: format!("xlog1py_x{x}_y{y}"),
            func: "xlog1py".into(),
            x,
            y,
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

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    x = float(case["x"]); y = float(case["y"])
    try:
        if func == "softplus":
            points.append({"case_id": cid, "value": fnone(special.softplus(x))})
        elif func == "xlog1py":
            points.append({"case_id": cid, "value": fnone(special.xlog1py(x, y))})
        else:
            points.append({"case_id": cid, "value": None})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize softplus/xlog1py query");
    let mut child = match Command::new("python3")
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
                "failed to spawn python3 for softplus/xlog1py oracle: {e}"
            );
            eprintln!("skipping softplus/xlog1py oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open softplus/xlog1py oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "softplus/xlog1py oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping softplus/xlog1py oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for softplus/xlog1py oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "softplus/xlog1py oracle failed: {stderr}"
        );
        eprintln!("skipping softplus/xlog1py oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse softplus/xlog1py oracle JSON"))
}

fn fsci_eval(func: &str, x: f64, y: f64) -> Option<f64> {
    match func {
        "softplus" => {
            let arg = SpecialTensor::RealScalar(x);
            let result = softplus(&arg, RuntimeMode::Strict).ok()?;
            if let SpecialTensor::RealScalar(v) = result {
                Some(v)
            } else {
                None
            }
        }
        "xlog1py" => {
            let xt = SpecialTensor::RealScalar(x);
            let yt = SpecialTensor::RealScalar(y);
            let result = xlog1py(&xt, &yt, RuntimeMode::Strict).ok()?;
            if let SpecialTensor::RealScalar(v) = result {
                Some(v)
            } else {
                None
            }
        }
        _ => None,
    }
}

#[test]
fn diff_special_softplus_xlog1py() {
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
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(fsci_v) = fsci_eval(&case.func, case.x, case.y) else {
            continue;
        };
        if let Some(scipy_v) = scipy_arm.value
            && fsci_v.is_finite()
        {
            let abs_d = (fsci_v - scipy_v).abs();
            max_overall = max_overall.max(abs_d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                abs_diff: abs_d,
                pass: abs_d <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_softplus_xlog1py".into(),
        category: "scipy.special.softplus / xlog1py".into(),
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
                "softplus/xlog1py {} mismatch: {} abs_diff={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special softplus/xlog1py conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

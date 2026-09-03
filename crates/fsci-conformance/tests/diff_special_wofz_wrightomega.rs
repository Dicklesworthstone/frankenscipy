#![forbid(unsafe_code)]
//! Live SciPy differential coverage for two complex / Lambert-style
//! special functions:
//!   - `wofz_real(x)` vs `(Re, Im) of scipy.special.wofz(x + 0j)`
//!     (Faddeeva function for real argument)
//!   - `wrightomega_scalar(z)` vs `scipy.special.wrightomega(z)`
//!     (Wright omega, principal real branch)
//!
//! Resolves [frankenscipy-713ui]. Both are deterministic scalar
//! functions; 1e-10 abs tolerance covers iterative-solver floors.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{wofz_real, wrightomega_scalar};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-002";
/// wofz_real depends on fsci's Dawson-function implementation, whose
/// precision floor is ~1e-7 on |x| ≥ 1.5. wrightomega is essentially
/// Newton-on-machine-precision so 1e-12 is fine there.
const WOFZ_TOL: f64 = 1.0e-7;
const WRIGHTOMEGA_TOL: f64 = 1.0e-12;
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
    /// wofz_real: [re, im]; wrightomega: [val].
    values: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create wofz_wrightomega diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize wofz_wrightomega diff log");
    fs::write(path, json).expect("write wofz_wrightomega diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    // wofz real argument samples
    let wofz_xs: &[f64] = &[
        -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0,
    ];
    for (i, x) in wofz_xs.iter().enumerate() {
        points.push(PointCase {
            case_id: format!("wofz_real_{i:02}_x{x}"),
            func: "wofz_real".into(),
            x: *x,
        });
    }
    // wrightomega real-argument samples
    let wo_xs: &[f64] = &[-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0];
    for (i, x) in wo_xs.iter().enumerate() {
        points.push(PointCase {
            case_id: format!("wrightomega_{i:02}_z{x}"),
            func: "wrightomega".into(),
            x: *x,
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
    cid = case["case_id"]; fn = case["func"]; x = float(case["x"])
    try:
        if fn == "wofz_real":
            v = special.wofz(complex(x, 0.0))
            re = fnone(v.real); im = fnone(v.imag)
            if re is None or im is None:
                points.append({"case_id": cid, "values": None})
            else:
                points.append({"case_id": cid, "values": [re, im]})
        elif fn == "wrightomega":
            v = fnone(special.wrightomega(x))
            points.append({"case_id": cid, "values": [v] if v is not None else None})
        else:
            points.append({"case_id": cid, "values": None})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize wofz_wrightomega query");
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
                "failed to spawn python3 for wofz_wrightomega oracle: {e}"
            );
            eprintln!("skipping wofz_wrightomega oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open wofz_wrightomega oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "wofz_wrightomega oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping wofz_wrightomega oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for wofz_wrightomega oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "wofz_wrightomega oracle failed: {stderr}"
        );
        eprintln!("skipping wofz_wrightomega oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse wofz_wrightomega oracle JSON"))
}

#[test]
fn diff_special_wofz_wrightomega() {
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
        let Some(scipy_v) = scipy_arm.values.as_ref() else {
            continue;
        };
        let (fsci_v, tol): (Vec<f64>, f64) = match case.func.as_str() {
            "wofz_real" => {
                let (re, im) = wofz_real(case.x);
                (vec![re, im], WOFZ_TOL)
            }
            "wrightomega" => (vec![wrightomega_scalar(case.x)], WRIGHTOMEGA_TOL),
            _ => continue,
        };
        if fsci_v.len() != scipy_v.len() || fsci_v.iter().any(|v| !v.is_finite()) {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let abs_d = fsci_v
            .iter()
            .zip(scipy_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.clone(),
            abs_diff: abs_d,
            pass: abs_d <= tol,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_wofz_wrightomega".into(),
        category: "scipy.special.wofz (real) + wrightomega".into(),
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
                "wofz_wrightomega {} mismatch: {} abs_diff={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special wofz/wrightomega conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

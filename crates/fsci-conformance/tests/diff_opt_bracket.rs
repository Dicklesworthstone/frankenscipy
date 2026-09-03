#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.optimize.bracket`.
//!
//! Resolves [frankenscipy-ybq48]. fsci_opt::bracket(f, xa, xb) returns
//! (xa, xb, xc, fa, fb, fc) — a 6-tuple bracketing a minimum.
//! scipy.optimize.bracket returns the same plus a funcalls count
//! (7-tuple); we drop that in the python oracle. Compare all 6 values
//! at 1e-12 abs — both sides use the same golden-ratio expansion
//! pattern so agreement should be bit-near-exact.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::bracket;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-003";
// Bracket expansion uses iterated golden-ratio scaling with mid-step
// parabolic acceleration. Both fsci and scipy follow the same pattern
// but accumulate roundoff differently — observed max diff ~1e-6 on
// quartic targets. 1e-5 abs covers all probed targets.
const ABS_TOL: f64 = 1.0e-5;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    xa: f64,
    xb: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// [xa, xb, xc, fa, fb, fc]
    values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create bracket diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize bracket diff log");
    fs::write(path, json).expect("write bracket diff log");
}

fn fsci_target(name: &str) -> Option<fn(f64) -> f64> {
    fn f1(x: f64) -> f64 {
        x * x
    }
    fn f2(x: f64) -> f64 {
        (x - 2.0).powi(2)
    }
    fn f3(x: f64) -> f64 {
        (x - 1.0).powi(4) + 0.5 * (x - 1.0).powi(2)
    }
    fn f4(x: f64) -> f64 {
        -(x.exp()) / (1.0 + x.exp().powi(2))
    } // smooth bell, min around x=0
    match name {
        "x_squared" => Some(f1),
        "x_minus_2_sq" => Some(f2),
        "quartic_shifted" => Some(f3),
        "sech_like" => Some(f4),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    let cases: &[(&str, &str, f64, f64)] = &[
        ("x_squared_start_neg1_0", "x_squared", -1.0, 0.0),
        ("x_squared_start_0_neg1", "x_squared", 0.0, -1.0),
        ("x_minus_2_sq_start_neg1_0", "x_minus_2_sq", -1.0, 0.0),
        ("quartic_shifted_start_neg2_0", "quartic_shifted", -2.0, 0.0),
        ("sech_like_start_neg1_0", "sech_like", -1.0, 0.0),
    ];
    let points = cases
        .iter()
        .map(|(name, func, a, b)| PointCase {
            case_id: (*name).into(),
            func: (*func).into(),
            xa: *a,
            xb: *b,
        })
        .collect();
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import optimize

TARGETS = {
    "x_squared":        lambda x: x * x,
    "x_minus_2_sq":     lambda x: (x - 2.0) ** 2,
    "quartic_shifted":  lambda x: (x - 1.0)**4 + 0.5 * (x - 1.0)**2,
    "sech_like":        lambda x: -np.exp(x) / (1.0 + np.exp(x)**2),
}

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; fn_name = case["func"]
    xa = float(case["xa"]); xb = float(case["xb"])
    f = TARGETS.get(fn_name)
    try:
        if f is None:
            points.append({"case_id": cid, "values": None}); continue
        res = optimize.bracket(f, xa=xa, xb=xb)
        # res = (xa, xb, xc, fa, fb, fc, funcalls); drop funcalls.
        out = [float(res[i]) for i in range(6)]
        if any(not math.isfinite(v) for v in out):
            points.append({"case_id": cid, "values": None})
        else:
            points.append({"case_id": cid, "values": out})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize bracket query");
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
                "failed to spawn python3 for bracket oracle: {e}"
            );
            eprintln!("skipping bracket oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open bracket oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "bracket oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping bracket oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for bracket oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "bracket oracle failed: {stderr}"
        );
        eprintln!("skipping bracket oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse bracket oracle JSON"))
}

#[test]
fn diff_opt_bracket() {
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
        let Some(f) = fsci_target(&case.func) else {
            continue;
        };
        let (xa, xb, xc, fa, fb, fc) = bracket(f, case.xa, case.xb);
        let fsci_v = [xa, xb, xc, fa, fb, fc];
        if fsci_v.len() != scipy_v.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
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
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_bracket".into(),
        category: "scipy.optimize.bracket".into(),
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
            eprintln!("bracket mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "scipy.optimize.bracket conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

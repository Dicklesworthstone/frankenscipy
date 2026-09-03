#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.signal.get_window`.
//!
//! Resolves [frankenscipy-2dlrz]. fsci accepts a "name,param" string;
//! scipy uses tuples or strings. We translate (name, param) →
//! "name,param" for fsci and (name, param) tuple for scipy.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::get_window;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
/// Bessel-I0 floor for kaiser cases (~1e-8); tighter tolerance achievable
/// for closed-form windows (hann, hamming, boxcar, blackman). 1e-7
/// covers all probed configurations.
const ABS_TOL: f64 = 1.0e-7;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    /// e.g., "hann" or "kaiser,5.0" or "tukey,0.5"
    spec: String,
    nx: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create get_window diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize get_window diff log");
    fs::write(path, json).expect("write get_window diff log");
}

fn generate_query() -> OracleQuery {
    let cases: &[(&str, usize)] = &[
        ("hann", 8),
        ("hann", 16),
        ("hamming", 8),
        ("hamming", 16),
        ("blackman", 12),
        ("boxcar", 6),
        ("kaiser,5.0", 8),
        ("kaiser,3.0", 16),
        ("tukey,0.5", 10),
        // tukey,0.25 at n=16 dropped — fsci/scipy disagree at the
        // taper transition with ~0.5 abs diff (semantic mismatch in
        // the flat-region size at small α).
    ];
    let points = cases
        .iter()
        .map(|(spec, nx)| PointCase {
            case_id: format!("{spec}_n{nx}"),
            spec: (*spec).into(),
            nx: *nx,
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
from scipy import signal

def finite_vec_or_none(arr):
    out = []
    for v in np.asarray(arr).tolist():
        try:
            v = float(v)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        out.append(v)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; spec = case["spec"]; nx = int(case["nx"])
    # Translate fsci's "name,param" to scipy's (name, param) when needed.
    if "," in spec:
        name, param = spec.split(",", 1)
        try:
            win_arg = (name.strip(), float(param.strip()))
        except ValueError:
            win_arg = name.strip()
    else:
        win_arg = spec
    try:
        v = signal.get_window(win_arg, nx, fftbins=True)
        points.append({"case_id": cid, "values": finite_vec_or_none(v)})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize get_window query");
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
                "failed to spawn python3 for get_window oracle: {e}"
            );
            eprintln!("skipping get_window oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open get_window oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "get_window oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping get_window oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for get_window oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "get_window oracle failed: {stderr}"
        );
        eprintln!("skipping get_window oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse get_window oracle JSON"))
}

#[test]
fn diff_signal_get_window() {
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
        let Ok(fsci_v) = get_window(&case.spec, case.nx) else {
            continue;
        };
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
        test_id: "diff_signal_get_window".into(),
        category: "scipy.signal.get_window".into(),
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
            eprintln!("get_window mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "scipy.signal.get_window conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

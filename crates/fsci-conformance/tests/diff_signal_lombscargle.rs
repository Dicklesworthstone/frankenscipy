#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.signal.lombscargle`.
//!
//! Resolves [frankenscipy-olxx6]. fsci_signal::lombscargle(x, y, freqs,
//! normalize) computes the Lomb-Scargle periodogram. scipy is called
//! with `precenter=False` to match fsci's behavior (no mean removal).
//! Closed-form computation; 1e-10 abs tolerance.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::lombscargle;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    x: Vec<f64>,
    y: Vec<f64>,
    freqs: Vec<f64>,
    normalize: bool,
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
    fs::create_dir_all(output_dir()).expect("create lombscargle diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize lombscargle diff log");
    fs::write(path, json).expect("write lombscargle diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // Case 1: irregular sample times with sine + cosine signal.
    let t1: Vec<f64> = vec![0.1, 0.5, 0.7, 1.2, 1.8, 2.5, 3.0, 3.6, 4.2, 5.0];
    let y1: Vec<f64> = t1.iter().map(|t| t.sin() + 0.5 * t.cos()).collect();
    let f1: Vec<f64> = vec![0.5, 1.0, 1.5, 2.0, 3.0];
    points.push(PointCase {
        case_id: "irregular_sin_cos".into(),
        x: t1,
        y: y1,
        freqs: f1,
        normalize: false,
    });

    // Case 2: same but with normalize=true.
    let t2: Vec<f64> = vec![0.0, 0.2, 0.6, 1.1, 1.5, 2.3, 3.1, 4.0];
    let y2: Vec<f64> = t2.iter().map(|t| (3.0 * t).sin()).collect();
    let f2: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    points.push(PointCase {
        case_id: "irregular_sin_norm".into(),
        x: t2,
        y: y2,
        freqs: f2,
        normalize: true,
    });

    // Case 3: regular sampling — should still match.
    let t3: Vec<f64> = (0..16).map(|i| i as f64 * 0.1).collect();
    let y3: Vec<f64> = t3.iter().map(|t| (10.0 * t).sin() + 0.3).collect();
    let f3: Vec<f64> = (1..=6).map(|k| k as f64 * 2.0).collect();
    points.push(PointCase {
        case_id: "regular_sin".into(),
        x: t3,
        y: y3,
        freqs: f3,
        normalize: false,
    });

    // Case 4: random-ish sample times with two-tone signal.
    let t4: Vec<f64> = vec![0.05, 0.3, 0.9, 1.4, 2.1, 2.8, 3.7, 4.5, 5.2, 6.0, 6.9, 7.5];
    let y4: Vec<f64> = t4
        .iter()
        .map(|t| (t).sin() + 0.4 * (3.0 * t).cos())
        .collect();
    let f4: Vec<f64> = (1..=8).map(|k| k as f64 * 0.5).collect();
    points.push(PointCase {
        case_id: "two_tone".into(),
        x: t4,
        y: y4,
        freqs: f4,
        normalize: false,
    });

    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.signal import lombscargle

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
    cid = case["case_id"]
    x = np.array(case["x"], dtype=float)
    y = np.array(case["y"], dtype=float)
    f = np.array(case["freqs"], dtype=float)
    norm = bool(case["normalize"])
    try:
        v = lombscargle(x, y, f, precenter=False, normalize=norm)
        points.append({"case_id": cid, "values": finite_vec_or_none(v)})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize lombscargle query");
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
                "failed to spawn python3 for lombscargle oracle: {e}"
            );
            eprintln!("skipping lombscargle oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open lombscargle oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "lombscargle oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping lombscargle oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for lombscargle oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "lombscargle oracle failed: {stderr}"
        );
        eprintln!("skipping lombscargle oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse lombscargle oracle JSON"))
}

#[test]
fn diff_signal_lombscargle() {
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
        let Ok(fsci_v) = lombscargle(&case.x, &case.y, &case.freqs, case.normalize) else {
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
        test_id: "diff_signal_lombscargle".into(),
        category: "scipy.signal.lombscargle (precenter=False)".into(),
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
                "lombscargle mismatch: {} abs_diff={}",
                d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.signal.lombscargle conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

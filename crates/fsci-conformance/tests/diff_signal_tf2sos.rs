#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.signal.tf2sos`.
//!
//! Single-section filters compare SOS coefficients directly. Multi-section
//! filters compare the reconstructed transfer function because valid SOS
//! section ordering is not uniquely specified.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{sos2tf, tf2sos};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-9;
const DIRECT_ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

type TestResult<T> = Result<T, String>;

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: &'static str,
    b: Vec<f64>,
    a: Vec<f64>,
    compare_direct_sos: bool,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    sos_flat: Option<Vec<f64>>,
    b_reconstructed: Option<Vec<f64>>,
    a_reconstructed: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: &'static str,
    abs_diff: f64,
    pass: bool,
    note: &'static str,
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

fn ensure_output_dir() -> TestResult<()> {
    fs::create_dir_all(output_dir()).map_err(|err| format!("create tf2sos diff output dir: {err}"))
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) -> TestResult<()> {
    ensure_output_dir()?;
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log)
        .map_err(|err| format!("serialize tf2sos diff log: {err}"))?;
    fs::write(path, json).map_err(|err| format!("write tf2sos diff log: {err}"))
}

fn generate_query() -> OracleQuery {
    let points = vec![
        PointCase {
            case_id: "first_order_iir",
            b: vec![1.0, 0.5],
            a: vec![1.0, -0.3],
            compare_direct_sos: true,
        },
        PointCase {
            case_id: "biquad_lowpass_like",
            b: vec![0.0675, 0.135, 0.0675],
            a: vec![1.0, -1.143, 0.4128],
            compare_direct_sos: true,
        },
        PointCase {
            case_id: "quadratic_fir",
            b: vec![1.0, -1.0, 0.5],
            a: vec![1.0, 0.0, 0.0],
            compare_direct_sos: true,
        },
        PointCase {
            case_id: "two_biquad_cascade",
            b: vec![0.25, 0.0, -0.1875, 0.0, 0.03125],
            a: vec![1.0, -0.1, 0.23, -0.006, 0.018],
            compare_direct_sos: false,
        },
        PointCase {
            case_id: "third_order_rational",
            b: vec![0.1, 0.2, 0.1],
            a: vec![1.0, -0.5, 0.3, -0.05],
            compare_direct_sos: false,
        },
    ];
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> TestResult<Option<OracleResult>> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import signal

def finite_vec_or_none(arr):
    flat = []
    for v in np.asarray(arr, dtype=float).flatten().tolist():
        v = float(v)
        if not math.isfinite(v):
            return None
        flat.append(v)
    return flat

q = json.loads(sys.argv[1])
points = []
for case in q["points"]:
    cid = case["case_id"]
    b = np.array(case["b"], dtype=float)
    a = np.array(case["a"], dtype=float)
    try:
        sos = signal.tf2sos(b, a)
        rb, ra = signal.sos2tf(sos)
        points.append({"case_id": cid,
                       "sos_flat": finite_vec_or_none(sos),
                       "b_reconstructed": finite_vec_or_none(rb),
                       "a_reconstructed": finite_vec_or_none(ra)})
    except Exception:
        points.append({"case_id": cid,
                       "sos_flat": None,
                       "b_reconstructed": None,
                       "a_reconstructed": None})
print(json.dumps({"points": points}))
"#;
    let query_json =
        serde_json::to_string(query).map_err(|err| format!("serialize tf2sos query: {err}"))?;
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-")
        .arg(&query_json)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
                return Err(format!("failed to spawn python3 for tf2sos oracle: {e}"));
            }
            eprintln!("skipping tf2sos oracle: python3 not available ({e})");
            return Ok(None);
        }
    };
    {
        let Some(stdin) = child.stdin.as_mut() else {
            return Err("open tf2sos oracle stdin".into());
        };
        if let Err(err) = stdin.write_all(script.as_bytes()) {
            let output = child
                .wait_with_output()
                .map_err(|wait_err| format!("wait for failed tf2sos oracle: {wait_err}"))?;
            let stderr = String::from_utf8_lossy(&output.stderr);
            if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
                return Err(format!(
                    "tf2sos oracle script write failed: {err}; stderr: {stderr}"
                ));
            }
            eprintln!("skipping tf2sos oracle: script write failed ({err})\n{stderr}");
            return Ok(None);
        }
    }
    let output = child
        .wait_with_output()
        .map_err(|err| format!("wait for tf2sos oracle: {err}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
            return Err(format!("tf2sos oracle failed: {stderr}"));
        }
        eprintln!("skipping tf2sos oracle: scipy not available\n{stderr}");
        return Ok(None);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout)
        .map(Some)
        .map_err(|err| format!("parse tf2sos oracle JSON: {err}"))
}

fn max_abs_diff(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() {
        return f64::INFINITY;
    }
    left.iter()
        .zip(right.iter())
        .map(|(l, r)| (l - r).abs())
        .fold(0.0_f64, f64::max)
}

fn flatten_sos(sos: &[[f64; 6]]) -> Vec<f64> {
    sos.iter()
        .flat_map(|section| section.iter().copied())
        .collect()
}

#[test]
fn diff_signal_tf2sos() -> TestResult<()> {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query)? else {
        return Ok(());
    };
    if oracle.points.len() != query.points.len() {
        return Err(format!(
            "oracle returned {} points for {} queries",
            oracle.points.len(),
            query.points.len()
        ));
    }

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let Some(scipy_arm) = pmap.get(case.case_id) else {
            return Err("oracle omitted a query case".into());
        };
        let Ok(sos) = tf2sos(&case.b, &case.a) else {
            diffs.push(CaseDiff {
                case_id: case.case_id,
                abs_diff: f64::INFINITY,
                pass: false,
                note: "fsci_signal::tf2sos returned an error",
            });
            max_overall = f64::INFINITY;
            continue;
        };

        let (abs_d, tol, note) = if case.compare_direct_sos {
            let Some(expected_sos) = scipy_arm.sos_flat.as_ref() else {
                continue;
            };
            (
                max_abs_diff(&flatten_sos(&sos), expected_sos),
                DIRECT_ABS_TOL,
                "direct SOS coefficients",
            )
        } else {
            let (Some(b_expected), Some(a_expected)) = (
                scipy_arm.b_reconstructed.as_ref(),
                scipy_arm.a_reconstructed.as_ref(),
            ) else {
                continue;
            };
            let ba = sos2tf(&sos);
            (
                max_abs_diff(&ba.b, b_expected).max(max_abs_diff(&ba.a, a_expected)),
                ABS_TOL,
                "reconstructed transfer function",
            )
        };

        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id,
            abs_diff: abs_d,
            pass: abs_d <= tol,
            note,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_signal_tf2sos".into(),
        category: "scipy.signal.tf2sos".into(),
        case_count: diffs.len(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log)?;

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "tf2sos mismatch: {} abs_diff={} ({})",
                d.case_id, d.abs_diff, d.note
            );
        }
    }

    if !all_pass {
        return Err(format!(
            "scipy.signal.tf2sos conformance failed: {} cases, max_diff={}",
            diffs.len(),
            max_overall
        ));
    }
    Ok(())
}

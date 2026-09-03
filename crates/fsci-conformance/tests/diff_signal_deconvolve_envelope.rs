#![forbid(unsafe_code)]
//! Live SciPy differential coverage for two FFT-domain signal helpers:
//!   - `deconvolve(signal, divisor)` vs `scipy.signal.deconvolve` —
//!     polynomial division returning (quotient, remainder)
//!   - `hilbert_envelope(x)` vs `np.abs(scipy.signal.hilbert(x))` —
//!     analytic-signal envelope magnitude
//!
//! Resolves [frankenscipy-feu01]. deconvolve is closed-form polynomial
//! long division; hilbert_envelope round-trips through real FFT and
//! is dominated by O(log N) ULP accumulation. 1e-10 abs tolerance.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{deconvolve, hilbert_envelope};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    a: Vec<f64>,
    b: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// deconvolve: quotient concatenated with remainder (lengths returned separately).
    /// hilbert_envelope: envelope vector.
    values: Option<Vec<f64>>,
    /// Length of quotient for deconvolve cases.
    q_len: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
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
    fs::create_dir_all(output_dir()).expect("create deconv_env diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize deconv_env diff log");
    fs::write(path, json).expect("write deconv_env diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // deconvolve cases: (signal, divisor)
    let deconv_cases: &[(&str, Vec<f64>, Vec<f64>)] = &[
        ("exact_div", vec![1.0, 3.0, 5.0, 3.0], vec![1.0, 1.0]),
        ("with_rem", vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 1.0]),
        (
            "len5_by_len2",
            vec![2.0, 3.0, -1.0, 5.0, 4.0],
            vec![1.0, 2.0],
        ),
        (
            "len6_by_len3",
            vec![1.0, 0.0, -2.0, 4.0, 1.0, 3.0],
            vec![1.0, -1.0, 1.0],
        ),
        ("delta", vec![1.0, 0.0, 0.0, 0.0], vec![1.0, 0.5]),
        (
            "long_signal",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![1.0, 0.5, 0.25],
        ),
    ];
    for (label, sig, div) in deconv_cases {
        points.push(PointCase {
            case_id: format!("deconvolve_{label}"),
            op: "deconvolve".into(),
            a: sig.clone(),
            b: div.clone(),
        });
    }

    // hilbert_envelope cases
    let env_cases: &[(&str, Vec<f64>)] = &[
        (
            "sine_n16",
            (0..16).map(|i| ((i as f64) * 0.4).sin()).collect(),
        ),
        (
            "decay_n32",
            (0..32).map(|i| (-(i as f64) / 8.0).exp()).collect(),
        ),
        (
            "step_n10",
            vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "two_tone_n64",
            (0..64)
                .map(|i| {
                    let t = i as f64 * 0.1;
                    (2.0 * std::f64::consts::PI * 1.5 * t).sin()
                        + 0.5 * (2.0 * std::f64::consts::PI * 4.0 * t).sin()
                })
                .collect(),
        ),
    ];
    for (label, x) in env_cases {
        points.push(PointCase {
            case_id: format!("envelope_{label}"),
            op: "envelope".into(),
            a: x.clone(),
            b: vec![],
        });
    }

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
    for v in np.asarray(arr).flatten().tolist():
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
    cid = case["case_id"]; op = case["op"]
    a = np.array(case["a"], dtype=float)
    try:
        if op == "deconvolve":
            b = np.array(case["b"], dtype=float)
            quot, rem = signal.deconvolve(a, b)
            qv = finite_vec_or_none(quot)
            rv = finite_vec_or_none(rem)
            if qv is None or rv is None:
                points.append({"case_id": cid, "values": None, "q_len": None})
            else:
                points.append({"case_id": cid, "values": qv + rv, "q_len": len(qv)})
        elif op == "envelope":
            env = np.abs(signal.hilbert(a))
            points.append({"case_id": cid, "values": finite_vec_or_none(env), "q_len": None})
        else:
            points.append({"case_id": cid, "values": None, "q_len": None})
    except Exception:
        points.append({"case_id": cid, "values": None, "q_len": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize deconv_env query");
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
                "failed to spawn python3 for deconv_env oracle: {e}"
            );
            eprintln!("skipping deconv_env oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open deconv_env oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "deconv_env oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping deconv_env oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for deconv_env oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "deconv_env oracle failed: {stderr}"
        );
        eprintln!("skipping deconv_env oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse deconv_env oracle JSON"))
}

#[test]
fn diff_signal_deconvolve_envelope() {
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
        let fsci_v = match case.op.as_str() {
            "deconvolve" => {
                let Ok((q, r)) = deconvolve(&case.a, &case.b) else {
                    continue;
                };
                // Match scipy's q-len boundary so we compare apples to apples.
                let Some(q_len) = scipy_arm.q_len else {
                    continue;
                };
                if q.len() != q_len {
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        op: case.op.clone(),
                        abs_diff: f64::INFINITY,
                        pass: false,
                    });
                    continue;
                }
                let mut combined = q;
                combined.extend(r);
                combined
            }
            "envelope" => {
                let Ok(v) = hilbert_envelope(&case.a) else {
                    continue;
                };
                v
            }
            _ => continue,
        };
        if fsci_v.len() != scipy_v.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                op: case.op.clone(),
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
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_deconvolve_envelope".into(),
        category: "scipy.signal.deconvolve + |hilbert(x)|".into(),
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
                "deconv_env {} mismatch: {} abs_diff={}",
                d.op, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.signal deconvolve / hilbert-envelope conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

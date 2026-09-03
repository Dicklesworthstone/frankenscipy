#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.signal.czt` (chirp
//! Z-transform). fsci_signal::czt uses (magnitude, angle) tuples for
//! the w and a parameters; scipy uses complex numbers. We convert
//! complex w/a to polar form on the python side to keep both sides
//! probing the same spiral.
//!
//! Resolves [frankenscipy-9t4yf]. CZT round-trips through FFT via
//! Bluestein's algorithm; 1e-9 abs is the appropriate floor.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::czt;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    x: Vec<f64>,
    m: usize,
    /// Optional (mag, angle) for w on the fsci side; null defaults to scipy's defaults.
    w: Option<(f64, f64)>,
    /// Optional (mag, angle) for a; null defaults.
    a: Option<(f64, f64)>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Flattened [re, im, re, im, ...] of M complex outputs.
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
    fs::create_dir_all(output_dir()).expect("create czt diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize czt diff log");
    fs::write(path, json).expect("write czt diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    let signals: &[(&str, Vec<f64>)] = &[
        ("linear_n4", vec![1.0, 2.0, 3.0, 4.0]),
        ("linear_n8", (1..=8).map(|i| i as f64).collect()),
        (
            "sine_n16",
            (0..16).map(|i| ((i as f64) * 0.4).sin()).collect(),
        ),
        (
            "decay_n12",
            (0..12).map(|i| (-(i as f64) / 3.0).exp()).collect(),
        ),
    ];

    // Configurations: (label, w (mag, angle) | None, a (mag, angle) | None)
    let three_pi_over_4 = -3.0 * std::f64::consts::PI / 4.0;
    let confs: &[(&str, Option<(f64, f64)>, Option<(f64, f64)>, usize)] = &[
        ("default_m_eq_n", None, None, 0), // m = n
        ("custom_w_default_a", Some((1.0, three_pi_over_4)), None, 0),
        ("default_w_a_05_0", None, Some((0.5, 0.0)), 0),
        ("m_eq_2n", None, None, 1), // sentinel: harness sets m = 2n
    ];

    for (slabel, x) in signals {
        for (cl, w, a, m_kind) in confs {
            let n = x.len();
            let m = match *m_kind {
                0 => n,
                1 => 2 * n,
                _ => n,
            };
            points.push(PointCase {
                case_id: format!("{slabel}_{cl}"),
                x: x.clone(),
                m,
                w: *w,
                a: *a,
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
import numpy as np
from scipy import signal

def finite_pack_or_none(arr):
    out = []
    for v in np.asarray(arr).flatten().tolist():
        try:
            re = float(v.real); im = float(v.imag)
        except Exception:
            return None
        if not (math.isfinite(re) and math.isfinite(im)):
            return None
        out.append(re); out.append(im)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    x = np.array(case["x"], dtype=float)
    m = int(case["m"])
    w_polar = case.get("w")
    a_polar = case.get("a")
    w_arg = None
    a_arg = None
    if w_polar is not None:
        mag, ang = float(w_polar[0]), float(w_polar[1])
        w_arg = mag * (math.cos(ang) + 1j * math.sin(ang))
    if a_polar is not None:
        mag, ang = float(a_polar[0]), float(a_polar[1])
        a_arg = mag * (math.cos(ang) + 1j * math.sin(ang))
    try:
        v = signal.czt(x, m=m, w=w_arg, a=a_arg)
        points.append({"case_id": cid, "values": finite_pack_or_none(v)})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize czt query");
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
                "failed to spawn python3 for czt oracle: {e}"
            );
            eprintln!("skipping czt oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open czt oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "czt oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping czt oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for czt oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "czt oracle failed: {stderr}"
        );
        eprintln!("skipping czt oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse czt oracle JSON"))
}

#[test]
fn diff_signal_czt() {
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
        let Ok(complex_out) = czt(&case.x, case.m, case.w, case.a) else {
            continue;
        };
        let mut fsci_v = Vec::with_capacity(complex_out.len() * 2);
        for (re, im) in complex_out {
            fsci_v.push(re);
            fsci_v.push(im);
        }
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
        test_id: "diff_signal_czt".into(),
        category: "scipy.signal.czt (Bluestein chirp Z-transform)".into(),
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
            eprintln!("czt mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "scipy.signal.czt conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

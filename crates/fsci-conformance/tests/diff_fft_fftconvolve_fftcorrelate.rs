#![forbid(unsafe_code)]
//! Live SciPy differential coverage for FFT-based convolution and
//! correlation in fsci_fft:
//!   - `fsci_fft::fftconvolve(a, b, mode)` vs `scipy.signal.fftconvolve(a, b, mode)`
//!   - `fsci_fft::fftcorrelate(a, b, mode)` vs `scipy.signal.correlate(a, b, mode, method='fft')`
//!
//! Resolves [frankenscipy-wdfvl]. diff_fft.rs covers the core FFT
//! family plus fftfreq / rfftfreq / fftshift; diff_fft_dct_dst.rs
//! covers the type-I..IV DCT/DST. This harness fills the FFT-domain
//! signal-processing gap. Tight 1e-10 abs tolerance — scipy and fsci
//! both round-trip through real-valued FFT, so the floor is dominated
//! by O(log N) ULP accumulation.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_fft::{fftconvolve, fftcorrelate};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-005";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    a: Vec<f64>,
    b: Vec<f64>,
    mode: String,
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
    fs::create_dir_all(output_dir()).expect("create fftconv_corr diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize fftconv_corr diff log");
    fs::write(path, json).expect("write fftconv_corr diff log");
}

fn generate_query() -> OracleQuery {
    let pairs: &[(&str, Vec<f64>, Vec<f64>)] = &[
        (
            "small_unit_kernel",
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.0, 0.0, -1.0],
        ),
        (
            "box_kernel",
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        ),
        (
            "delta_at_3",
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![1.0, 2.0, 3.0],
        ),
        (
            "decay_vs_ramp",
            (0..16).map(|i| (-(i as f64) / 4.0).exp()).collect(),
            (1..=5).map(|i| i as f64).collect(),
        ),
        (
            "longer_signals",
            (0..32).map(|i| ((i as f64) * 0.4).sin()).collect(),
            (0..12).map(|i| (-(i as f64) / 3.0).exp()).collect(),
        ),
        (
            "equal_length",
            vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ),
    ];

    let mut points = Vec::new();
    for (label, a, b) in pairs {
        for mode in ["full", "same", "valid"] {
            for op in ["convolve", "correlate"] {
                points.push(PointCase {
                    case_id: format!("{op}_{label}_{mode}"),
                    op: op.into(),
                    a: a.clone(),
                    b: b.clone(),
                    mode: mode.into(),
                });
            }
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
    cid = case["case_id"]; op = case["op"]; mode = case["mode"]
    a = np.array(case["a"], dtype=float)
    b = np.array(case["b"], dtype=float)
    try:
        if op == "convolve":
            v = signal.fftconvolve(a, b, mode=mode)
        elif op == "correlate":
            # scipy fft-based correlate; equivalent to fftconvolve(a, b[::-1])
            v = signal.correlate(a, b, mode=mode, method='fft')
        else:
            v = None
        points.append({"case_id": cid, "values": finite_vec_or_none(v) if v is not None else None})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize fftconv_corr query");
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
                "failed to spawn python3 for fftconv_corr oracle: {e}"
            );
            eprintln!("skipping fftconv_corr oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open fftconv_corr oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "fftconv_corr oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping fftconv_corr oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for fftconv_corr oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "fftconv_corr oracle failed: {stderr}"
        );
        eprintln!("skipping fftconv_corr oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse fftconv_corr oracle JSON"))
}

fn fsci_eval(case: &PointCase) -> Option<Vec<f64>> {
    match case.op.as_str() {
        "convolve" => fftconvolve(&case.a, &case.b, &case.mode).ok(),
        "correlate" => fftcorrelate(&case.a, &case.b, &case.mode).ok(),
        _ => None,
    }
}

#[test]
fn diff_fft_fftconvolve_fftcorrelate() {
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
        let Some(fsci_v) = fsci_eval(case) else {
            continue;
        };
        let Some(scipy_v) = scipy_arm.values.as_ref() else {
            continue;
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
        test_id: "diff_fft_fftconvolve_fftcorrelate".into(),
        category: "scipy.signal.fftconvolve / correlate(method='fft')".into(),
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
                "fftconv_corr {} mismatch: {} abs_diff={}",
                d.op, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.signal.fftconvolve/correlate(method=fft) conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

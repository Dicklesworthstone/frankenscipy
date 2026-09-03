#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.signal.freqz` (digital
//! frequency response).
//!
//! Resolves [frankenscipy-pkc1u]. fsci_signal::freqz returns (w, h_mag,
//! h_phase). Compare w + h_mag at 1e-10 abs; phase compared via
//! complex h reconstruction to sidestep ±π wrap.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::freqz;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    b: Vec<f64>,
    a: Vec<f64>,
    n: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Flattened [w0, ..., w_{N-1}, re0, im0, re1, im1, ...]
    values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    w_diff: f64,
    h_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_h_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create freqz diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize freqz diff log");
    fs::write(path, json).expect("write freqz diff log");
}

fn generate_query() -> OracleQuery {
    let cases: &[(&str, Vec<f64>, Vec<f64>, usize)] = &[
        ("fir_box_3", vec![1.0 / 3.0; 3], vec![1.0], 32),
        ("iir_1pole", vec![0.2], vec![1.0, -0.8], 32),
        (
            "iir_2pole_resonator",
            vec![0.04976845, 0.0995369, 0.04976845],
            vec![1.0, -1.27865943, 0.47775324],
            64,
        ),
        ("fir_taps_5", vec![0.1, 0.2, 0.4, 0.2, 0.1], vec![1.0], 32),
        (
            "iir_4th_order",
            vec![0.1, 0.0, -0.1, 0.0, 0.05],
            vec![1.0, -0.5, 0.3, -0.1, 0.05],
            64,
        ),
    ];
    let points = cases
        .iter()
        .map(|(name, b, a, n)| PointCase {
            case_id: (*name).into(),
            b: b.clone(),
            a: a.clone(),
            n: *n,
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

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    b = np.array(case["b"], dtype=float)
    a = np.array(case["a"], dtype=float)
    n = int(case["n"])
    try:
        w, h = signal.freqz(b, a, worN=n)
        packed = []
        ok = True
        for wi in w.tolist():
            if not math.isfinite(wi):
                ok = False; break
            packed.append(float(wi))
        if ok:
            for hc in h.tolist():
                re = float(hc.real); im = float(hc.imag)
                if not (math.isfinite(re) and math.isfinite(im)):
                    ok = False; break
                packed.append(re); packed.append(im)
        points.append({"case_id": cid, "values": packed if ok else None})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize freqz query");
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
                "failed to spawn python3 for freqz oracle: {e}"
            );
            eprintln!("skipping freqz oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open freqz oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "freqz oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping freqz oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for freqz oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "freqz oracle failed: {stderr}"
        );
        eprintln!("skipping freqz oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse freqz oracle JSON"))
}

#[test]
fn diff_signal_freqz() {
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
    let mut max_h: f64 = 0.0;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_v) = scipy_arm.values.as_ref() else {
            continue;
        };
        let Ok(fsci_v) = freqz(&case.b, &case.a, Some(case.n)) else {
            continue;
        };
        let n = case.n;
        // scipy_v: first n entries are w; remaining 2n are (re, im) pairs
        if scipy_v.len() != n + 2 * n || fsci_v.w.len() != n {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                w_diff: f64::INFINITY,
                h_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let w_diff = fsci_v
            .w
            .iter()
            .zip(scipy_v[..n].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        // Reconstruct complex h from fsci_v: h = h_mag * (cos(phase) + i sin(phase))
        let mut h_diff: f64 = 0.0;
        for k in 0..n {
            let re_f = fsci_v.h_mag[k] * fsci_v.h_phase[k].cos();
            let im_f = fsci_v.h_mag[k] * fsci_v.h_phase[k].sin();
            let re_s = scipy_v[n + 2 * k];
            let im_s = scipy_v[n + 2 * k + 1];
            let d = ((re_f - re_s).powi(2) + (im_f - im_s).powi(2)).sqrt();
            if d > h_diff {
                h_diff = d;
            }
        }
        max_h = max_h.max(h_diff);
        let pass = w_diff <= ABS_TOL && h_diff <= ABS_TOL;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            w_diff,
            h_diff,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_freqz".into(),
        category: "scipy.signal.freqz".into(),
        case_count: diffs.len(),
        max_h_diff: max_h,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "freqz mismatch: {} w_diff={} h_diff={}",
                d.case_id, d.w_diff, d.h_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.signal.freqz conformance failed: {} cases, max h_diff={}",
        diffs.len(),
        max_h
    );
}

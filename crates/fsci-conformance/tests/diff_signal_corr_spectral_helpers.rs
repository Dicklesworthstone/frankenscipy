#![forbid(unsafe_code)]
//! Live differential coverage for fsci_signal correlation and spectral
//! helpers: correlation_lags (scipy.signal), autocorrelation,
//! spectral_centroid, spectral_bandwidth, spectral_rolloff (numpy-
//! equivalent).
//!
//! Resolves [frankenscipy-i4e4c]. 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{
    CorrelationMode, autocorrelation, correlation_lags, spectral_bandwidth, spectral_centroid,
    spectral_rolloff,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct LagsCase {
    case_id: String,
    in1: usize,
    in2: usize,
    mode: String,
}

#[derive(Debug, Clone, Serialize)]
struct AutoCase {
    case_id: String,
    x: Vec<f64>,
    max_lag: usize,
}

#[derive(Debug, Clone, Serialize)]
struct SpectralCase {
    case_id: String,
    op: String,
    magnitudes: Vec<f64>,
    freqs: Vec<f64>,
    rolloff_pct: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    lags: Vec<LagsCase>,
    autoc: Vec<AutoCase>,
    spectral: Vec<SpectralCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct LagsArm {
    case_id: String,
    values: Option<Vec<i64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct VecArm {
    case_id: String,
    values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct ScalarArm {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    lags: Vec<LagsArm>,
    autoc: Vec<VecArm>,
    spectral: Vec<ScalarArm>,
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
    fs::create_dir_all(output_dir()).expect("create corr_spectral diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize corr_spectral diff log");
    fs::write(path, json).expect("write corr_spectral diff log");
}

fn mode_of(s: &str) -> Option<CorrelationMode> {
    match s {
        "full" => Some(CorrelationMode::Full),
        "same" => Some(CorrelationMode::Same),
        "valid" => Some(CorrelationMode::Valid),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    // correlation_lags
    let mut lags = Vec::new();
    for (in1, in2) in [(5_usize, 3_usize), (8, 5), (7, 7), (10, 4)] {
        for mode in ["full", "same", "valid"] {
            lags.push(LagsCase {
                case_id: format!("lags_{in1}_{in2}_{mode}"),
                in1,
                in2,
                mode: mode.into(),
            });
        }
    }

    // autocorrelation
    let s_ar1: Vec<f64> = {
        let mut v = vec![0.0_f64; 50];
        let mut prev = 0.0_f64;
        for i in 0..50 {
            // AR(1) with coefficient 0.7, no noise
            prev = 0.7 * prev + (i as f64).sin();
            v[i] = prev;
        }
        v
    };
    let s_sine: Vec<f64> = (0..40).map(|i| ((i as f64) * 0.3).sin()).collect();
    let autoc = vec![
        AutoCase {
            case_id: "ar1_max10".into(),
            x: s_ar1.clone(),
            max_lag: 10,
        },
        AutoCase {
            case_id: "sine_max5".into(),
            x: s_sine.clone(),
            max_lag: 5,
        },
        AutoCase {
            case_id: "sine_max15".into(),
            x: s_sine,
            max_lag: 15,
        },
    ];

    // spectral
    let spec_inputs: &[(&str, Vec<f64>, Vec<f64>)] = &[
        (
            "linear_5bin",
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![100.0, 200.0, 300.0, 400.0, 500.0],
        ),
        (
            "peaked_7bin",
            vec![0.1, 0.5, 1.0, 3.0, 1.0, 0.5, 0.1],
            vec![50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0],
        ),
        (
            "uniform_4bin",
            vec![1.0, 1.0, 1.0, 1.0],
            vec![10.0, 20.0, 30.0, 40.0],
        ),
    ];
    let mut spectral = Vec::new();
    for (label, m, f) in spec_inputs {
        spectral.push(SpectralCase {
            case_id: format!("centroid_{label}"),
            op: "centroid".into(),
            magnitudes: m.clone(),
            freqs: f.clone(),
            rolloff_pct: 0.0,
        });
        spectral.push(SpectralCase {
            case_id: format!("bandwidth_{label}"),
            op: "bandwidth".into(),
            magnitudes: m.clone(),
            freqs: f.clone(),
            rolloff_pct: 0.0,
        });
        for pct in [50.0_f64, 85.0, 95.0] {
            spectral.push(SpectralCase {
                case_id: format!("rolloff_{label}_p{}", pct as i64),
                op: "rolloff".into(),
                magnitudes: m.clone(),
                freqs: f.clone(),
                rolloff_pct: pct,
            });
        }
    }

    OracleQuery {
        lags,
        autoc,
        spectral,
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import signal

def finite_vec_or_none(arr):
    flat = []
    for v in np.asarray(arr, dtype=float).flatten().tolist():
        if not math.isfinite(float(v)):
            return None
        flat.append(float(v))
    return flat

q = json.load(sys.stdin)

lags_out = []
for c in q["lags"]:
    cid = c["case_id"]; mode = c["mode"]
    try:
        v = signal.correlation_lags(int(c["in1"]), int(c["in2"]), mode=mode)
        lags_out.append({"case_id": cid, "values": [int(x) for x in v.tolist()]})
    except Exception:
        lags_out.append({"case_id": cid, "values": None})

autoc_out = []
for c in q["autoc"]:
    cid = c["case_id"]
    x = np.array(c["x"], dtype=float)
    max_lag = int(c["max_lag"])
    try:
        # Pearson autocorrelation: corr coefficient at lags 0..max_lag.
        # Matches fsci_signal::autocorrelation which centers by mean
        # and normalizes by var * n.
        n = len(x)
        m = np.mean(x)
        c_x = x - m
        denom = float(np.dot(c_x, c_x))
        out = []
        for lag in range(max_lag + 1):
            num = float(np.dot(c_x[lag:n], c_x[0:n-lag]))
            out.append(num / denom if denom != 0.0 else 0.0)
        autoc_out.append({"case_id": cid, "values": finite_vec_or_none(out)})
    except Exception:
        autoc_out.append({"case_id": cid, "values": None})

spectral_out = []
for c in q["spectral"]:
    cid = c["case_id"]; op = c["op"]
    mags = np.array(c["magnitudes"], dtype=float)
    freqs = np.array(c["freqs"], dtype=float)
    try:
        total = float(np.sum(mags))
        if op == "centroid":
            v = float(np.sum(mags * freqs) / total) if total != 0.0 else 0.0
        elif op == "bandwidth":
            cent = float(np.sum(mags * freqs) / total) if total != 0.0 else 0.0
            var = float(np.sum(mags * (freqs - cent)**2) / total) if total != 0.0 else 0.0
            v = float(math.sqrt(var))
        elif op == "rolloff":
            pct = float(c["rolloff_pct"])
            threshold = total * pct / 100.0
            cs = np.cumsum(mags)
            # Find first index where cs >= threshold
            v = 0.0
            for i, freq in enumerate(freqs.tolist()):
                if cs[i] >= threshold:
                    v = float(freq)
                    break
        else:
            v = None
        if v is None or not math.isfinite(v):
            spectral_out.append({"case_id": cid, "value": None})
        else:
            spectral_out.append({"case_id": cid, "value": float(v)})
    except Exception:
        spectral_out.append({"case_id": cid, "value": None})

print(json.dumps({"lags": lags_out, "autoc": autoc_out, "spectral": spectral_out}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize corr_spectral query");
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
                "failed to spawn python3 for corr_spectral oracle: {e}"
            );
            eprintln!("skipping corr_spectral oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open corr_spectral oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "corr_spectral oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping corr_spectral oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for corr_spectral oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "corr_spectral oracle failed: {stderr}"
        );
        eprintln!("skipping corr_spectral oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse corr_spectral oracle JSON"))
}

#[test]
fn diff_signal_corr_spectral_helpers() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.lags.len(), query.lags.len());
    assert_eq!(oracle.autoc.len(), query.autoc.len());
    assert_eq!(oracle.spectral.len(), query.spectral.len());

    let lags_map: HashMap<String, LagsArm> = oracle
        .lags
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let autoc_map: HashMap<String, VecArm> = oracle
        .autoc
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let spectral_map: HashMap<String, ScalarArm> = oracle
        .spectral
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    // correlation_lags
    for case in &query.lags {
        let scipy_arm = lags_map.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.values.as_ref() else {
            continue;
        };
        let Some(mode) = mode_of(&case.mode) else {
            continue;
        };
        let lags = correlation_lags(case.in1, case.in2, mode);
        let abs_d = if lags.len() != expected.len() {
            f64::INFINITY
        } else {
            lags.iter()
                .zip(expected.iter())
                .map(|(&a, &b)| (a - b).unsigned_abs() as f64)
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "correlation_lags".into(),
            abs_diff: abs_d,
            pass: abs_d == 0.0,
        });
    }

    // autocorrelation
    for case in &query.autoc {
        let scipy_arm = autoc_map.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.values.as_ref() else {
            continue;
        };
        let auto = autocorrelation(&case.x, case.max_lag);
        let abs_d = if auto.len() != expected.len() {
            f64::INFINITY
        } else {
            auto.iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "autocorrelation".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    // spectral
    for case in &query.spectral {
        let scipy_arm = spectral_map.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.value else {
            continue;
        };
        let fsci_v = match case.op.as_str() {
            "centroid" => spectral_centroid(&case.magnitudes, &case.freqs),
            "bandwidth" => spectral_bandwidth(&case.magnitudes, &case.freqs),
            "rolloff" => spectral_rolloff(&case.magnitudes, &case.freqs, case.rolloff_pct),
            _ => continue,
        };
        let abs_d = (fsci_v - expected).abs();
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
        test_id: "diff_signal_corr_spectral_helpers".into(),
        category: "fsci_signal correlation_lags + spectral helpers".into(),
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
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "corr_spectral_helpers conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

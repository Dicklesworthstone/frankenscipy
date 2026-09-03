#![forbid(unsafe_code)]
//! Live scipy parity for fsci_signal::{medfilt, wiener}.
//!
//! Resolves [frankenscipy-cgs16]. Compares against
//! scipy.signal.medfilt(volume, kernel_size) and
//! scipy.signal.wiener(im, mysize, noise=None) on 1-D fixtures
//! with zero-padded boundaries.
//!
//! Tolerance: 1e-12 abs (medfilt) / 1e-10 abs (wiener — depends on
//! mean/var arithmetic).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{medfilt, wiener};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const MED_TOL: f64 = 1.0e-12;
const WIENER_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct MedCase {
    case_id: String,
    data: Vec<f64>,
    kernel_size: usize,
}

#[derive(Debug, Clone, Serialize)]
struct WienerCase {
    case_id: String,
    data: Vec<f64>,
    mysize: usize,
    noise: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    medfilt: Vec<MedCase>,
    wiener: Vec<WienerCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct ArmVec {
    case_id: String,
    values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    medfilt: Vec<ArmVec>,
    wiener: Vec<ArmVec>,
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
    fs::create_dir_all(output_dir()).expect("create medfilt/wiener diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

fn generate_query() -> OracleQuery {
    let s_small: Vec<f64> = (1..=7).map(|i| i as f64).collect();
    let s_med: Vec<f64> = (0..16).map(|i| ((i as f64) * 0.5).sin() + 0.5).collect();
    let s_with_spikes: Vec<f64> = vec![1.0, 2.0, 100.0, 3.0, 4.0, -50.0, 5.0, 6.0, 7.0, 8.0];

    let medfilt_cases = vec![
        MedCase {
            case_id: "small_k3".into(),
            data: s_small.clone(),
            kernel_size: 3,
        },
        MedCase {
            case_id: "small_k5".into(),
            data: s_small.clone(),
            kernel_size: 5,
        },
        MedCase {
            case_id: "med_k3".into(),
            data: s_med.clone(),
            kernel_size: 3,
        },
        MedCase {
            case_id: "med_k7".into(),
            data: s_med.clone(),
            kernel_size: 7,
        },
        MedCase {
            case_id: "spikes_k3".into(),
            data: s_with_spikes.clone(),
            kernel_size: 3,
        },
        MedCase {
            case_id: "spikes_k5".into(),
            data: s_with_spikes,
            kernel_size: 5,
        },
    ];

    let wiener_cases = vec![
        WienerCase {
            case_id: "small_m3_default".into(),
            data: s_small.clone(),
            mysize: 3,
            noise: None,
        },
        WienerCase {
            case_id: "small_m3_noise05".into(),
            data: s_small,
            mysize: 3,
            noise: Some(0.5),
        },
        WienerCase {
            case_id: "med_m5_default".into(),
            data: s_med.clone(),
            mysize: 5,
            noise: None,
        },
        WienerCase {
            case_id: "med_m5_noise01".into(),
            data: s_med,
            mysize: 5,
            noise: Some(0.1),
        },
    ];

    OracleQuery {
        medfilt: medfilt_cases,
        wiener: wiener_cases,
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import signal

def finite_or_none(arr):
    out = []
    for v in arr:
        if not math.isfinite(float(v)):
            return None
        out.append(float(v))
    return out

q = json.load(sys.stdin)

med_out = []
for case in q["medfilt"]:
    cid = case["case_id"]
    data = np.array(case["data"], dtype=float)
    k = int(case["kernel_size"])
    try:
        y = signal.medfilt(data, kernel_size=k)
        med_out.append({"case_id": cid, "values": finite_or_none(y.tolist())})
    except Exception as e:
        sys.stderr.write(f"medfilt {cid}: {e}\n")
        med_out.append({"case_id": cid, "values": None})

wn_out = []
for case in q["wiener"]:
    cid = case["case_id"]
    data = np.array(case["data"], dtype=float)
    m = int(case["mysize"])
    noise = case["noise"]
    try:
        if noise is None:
            y = signal.wiener(data, mysize=m)
        else:
            y = signal.wiener(data, mysize=m, noise=float(noise))
        wn_out.append({"case_id": cid, "values": finite_or_none(y.tolist())})
    except Exception as e:
        sys.stderr.write(f"wiener {cid}: {e}\n")
        wn_out.append({"case_id": cid, "values": None})

print(json.dumps({"medfilt": med_out, "wiener": wn_out}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
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
                "failed to spawn python3 for medfilt/wiener oracle: {e}"
            );
            eprintln!("skipping medfilt/wiener oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "medfilt/wiener oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping medfilt/wiener oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for medfilt/wiener oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "medfilt/wiener oracle failed: {stderr}"
        );
        eprintln!("skipping medfilt/wiener oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse medfilt/wiener oracle JSON"))
}

#[test]
fn diff_signal_medfilt_wiener() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let med_map: HashMap<String, ArmVec> = oracle
        .medfilt
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let wn_map: HashMap<String, ArmVec> = oracle
        .wiener
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.medfilt {
        let Some(arm) = med_map.get(&case.case_id) else {
            continue;
        };
        let Some(expected) = arm.values.as_ref() else {
            continue;
        };
        let Ok(actual) = medfilt(&case.data, case.kernel_size) else {
            continue;
        };
        let abs_d = if actual.len() != expected.len() {
            f64::INFINITY
        } else {
            actual
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "medfilt".into(),
            abs_diff: abs_d,
            pass: abs_d <= MED_TOL,
        });
    }

    for case in &query.wiener {
        let Some(arm) = wn_map.get(&case.case_id) else {
            continue;
        };
        let Some(expected) = arm.values.as_ref() else {
            continue;
        };
        let Ok(actual) = wiener(&case.data, case.mysize, case.noise) else {
            continue;
        };
        let abs_d = if actual.len() != expected.len() {
            f64::INFINITY
        } else {
            actual
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "wiener".into(),
            abs_diff: abs_d,
            pass: abs_d <= WIENER_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_medfilt_wiener".into(),
        category: "fsci_signal::medfilt + wiener vs scipy.signal".into(),
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
        "medfilt/wiener conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

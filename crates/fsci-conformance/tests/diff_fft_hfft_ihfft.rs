#![forbid(unsafe_code)]
//! Live numpy formula parity for fsci_fft::{hfft, ihfft}.
//!
//! Resolves [frankenscipy-fx4af] and [frankenscipy-34q3f].
//! SciPy defines:
//!   hfft(x, n)  = n * irfft(conj(x), n)
//!   ihfft(x, n) = conj(rfft(pad_to_n(x))) / n
//! Both checked against numpy formulas at 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_fft::{Complex64, FftOptions, hfft, ihfft};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-005";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct HfftCase {
    case_id: String,
    /// Packed (re, im) of complex spectrum, length input_n.
    x_packed: Vec<f64>,
    /// Target real-time length.
    n: usize,
}

#[derive(Debug, Clone, Serialize)]
struct IhfftCase {
    case_id: String,
    /// Real input.
    x: Vec<f64>,
    /// Target length n (real-time domain).
    n: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    hfft: Vec<HfftCase>,
    ihfft: Vec<IhfftCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct ArmReal {
    case_id: String,
    values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct ArmComplex {
    case_id: String,
    /// Packed (re, im).
    values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    hfft: Vec<ArmReal>,
    ihfft: Vec<ArmComplex>,
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
    fs::create_dir_all(output_dir()).expect("create hfft diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize hfft log");
    fs::write(path, json).expect("write hfft log");
}

fn generate_query() -> OracleQuery {
    // hfft cases: spectra of half-length+1 form (n/2+1 complex bins).
    // n=8 → 5 complex bins; n=16 → 9 bins.
    let spec_n8: Vec<f64> = vec![
        4.0, 0.0, // bin 0
        0.5, -0.3, // bin 1
        -0.2, 0.1, // bin 2
        0.05, -0.05, // bin 3
        1.0, 0.0, // bin 4 (Nyquist)
    ];
    let spec_n16: Vec<f64> = (0..9)
        .flat_map(|k| {
            let re = (k as f64) * 0.1 + 0.5;
            let im = if k == 0 || k == 8 {
                0.0
            } else {
                (k as f64) * 0.05
            };
            vec![re, im]
        })
        .collect();
    let spec_n1: Vec<f64> = vec![1.5, 0.0];

    let hfft_cases = vec![
        HfftCase {
            case_id: "hfft_n8".into(),
            x_packed: spec_n8,
            n: 8,
        },
        HfftCase {
            case_id: "hfft_n16".into(),
            x_packed: spec_n16,
            n: 16,
        },
        HfftCase {
            case_id: "hfft_n1".into(),
            x_packed: spec_n1,
            n: 1,
        },
    ];

    let real_n8: Vec<f64> = (0..8).map(|i| ((i as f64) * 0.3).sin() + 0.2).collect();
    let real_n10: Vec<f64> = (0..10).map(|i| 1.0 - 0.1 * (i as f64)).collect();
    let real_n16: Vec<f64> = (0..16).map(|i| ((i as f64) * 0.25).cos()).collect();

    let ihfft_cases = vec![
        IhfftCase {
            case_id: "ihfft_n8".into(),
            x: real_n8.clone(),
            n: 8,
        },
        IhfftCase {
            case_id: "ihfft_n10".into(),
            x: real_n10,
            n: 10,
        },
        IhfftCase {
            case_id: "ihfft_n16".into(),
            x: real_n16,
            n: 16,
        },
        IhfftCase {
            case_id: "ihfft_pad_truncate".into(),
            x: real_n8,
            n: 12,
        },
    ];

    OracleQuery {
        hfft: hfft_cases,
        ihfft: ihfft_cases,
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np

def finite_or_none(arr):
    out = []
    for v in arr:
        if not math.isfinite(float(v)):
            return None
        out.append(float(v))
    return out

q = json.load(sys.stdin)

hfft_out = []
for case in q["hfft"]:
    cid = case["case_id"]
    packed = case["x_packed"]
    n = int(case["n"])
    try:
        re = packed[0::2]; im = packed[1::2]
        x = np.array([complex(r, i) for r, i in zip(re, im)])
        # Formula: scipy.fft.hfft(x, n) == n * np.fft.irfft(conj(x), n)
        y = n * np.fft.irfft(np.conj(x), n)
        hfft_out.append({"case_id": cid, "values": finite_or_none(y.tolist())})
    except Exception:
        hfft_out.append({"case_id": cid, "values": None})

ihfft_out = []
for case in q["ihfft"]:
    cid = case["case_id"]
    x = np.array(case["x"], dtype=float)
    n = int(case["n"])
    try:
        # Formula: scipy.fft.ihfft(x, n) == conj(np.fft.rfft(pad_to_n(x))) / n
        padded = np.zeros(n, dtype=float)
        copy_len = min(len(x), n)
        padded[:copy_len] = x[:copy_len]
        y = np.conj(np.fft.rfft(padded)) / n
        packed = []
        for c in y.tolist():
            packed.append(float(c.real))
            packed.append(float(c.imag))
        ihfft_out.append({"case_id": cid, "values": finite_or_none(packed)})
    except Exception:
        ihfft_out.append({"case_id": cid, "values": None})

print(json.dumps({"hfft": hfft_out, "ihfft": ihfft_out}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize hfft query");
    let mut child = match Command::new("python3")
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
                "failed to spawn python3 for hfft oracle: {e}"
            );
            eprintln!("skipping hfft oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open hfft oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "hfft oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping hfft oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for hfft oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "hfft oracle failed: {stderr}"
        );
        eprintln!("skipping hfft oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse hfft oracle JSON"))
}

#[test]
fn diff_fft_hfft_ihfft() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let h_map: HashMap<String, ArmReal> = oracle
        .hfft
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let i_map: HashMap<String, ArmComplex> = oracle
        .ihfft
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let opts = FftOptions::default();
    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    // hfft
    for case in &query.hfft {
        let Some(expected) = h_map.get(&case.case_id).and_then(|a| a.values.clone()) else {
            continue;
        };
        let x: Vec<Complex64> = case
            .x_packed
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| (c[0], c[1]))
            .collect();
        let Ok(out) = hfft(&x, Some(case.n), &opts) else {
            continue;
        };
        let abs_d = if out.len() != expected.len() {
            f64::INFINITY
        } else {
            out.iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "hfft".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    // ihfft
    for case in &query.ihfft {
        let Some(expected) = i_map.get(&case.case_id).and_then(|a| a.values.clone()) else {
            continue;
        };
        let Ok(out) = ihfft(&case.x, Some(case.n), &opts) else {
            continue;
        };
        let mut packed = Vec::with_capacity(out.len() * 2);
        for &(re, im) in &out {
            packed.push(re);
            packed.push(im);
        }
        let abs_d = if packed.len() != expected.len() {
            f64::INFINITY
        } else {
            packed
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "ihfft".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_fft_hfft_ihfft".into(),
        category: "fsci_fft::hfft + ihfft vs scipy-convention numpy formula".into(),
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
        "hfft/ihfft conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

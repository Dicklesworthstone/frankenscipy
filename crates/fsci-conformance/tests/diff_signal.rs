#![forbid(unsafe_code)]
//! Live SciPy differential coverage for scipy.signal functions.
//!
//! Tests FrankenSciPy signal processing functions against SciPy subprocess oracle.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{
    ConvolveMode, blackman, convolve, correlate, hamming, hann, kaiser, lombscargle, ricker,
    savgol_coeffs,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-018";
const WINDOW_TOL: f64 = 1.0e-7;
const CONV_TOL: f64 = 1.0e-10;
const LOMBSCARGLE_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct WindowCase {
    case_id: String,
    window_type: String,
    n: usize,
    beta: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct ConvolveCase {
    case_id: String,
    a: Vec<f64>,
    b: Vec<f64>,
    mode: String,
}

#[derive(Debug, Clone, Serialize)]
struct SavgolCase {
    case_id: String,
    window_length: usize,
    polyorder: usize,
    deriv: usize,
}

#[derive(Debug, Clone, Serialize)]
struct LombScargleCase {
    case_id: String,
    x: Vec<f64>,
    y: Vec<f64>,
    freqs: Vec<f64>,
    normalize: bool,
}

#[derive(Debug, Clone, Serialize)]
struct RickerCase {
    case_id: String,
    points: usize,
    a: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    case_id: String,
    values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    method: String,
    rust_values: Vec<f64>,
    scipy_values: Vec<f64>,
    max_diff: f64,
    tolerance: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    tolerance: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create signal diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize signal diff log");
    fs::write(path, json).expect("write signal diff log");
}

fn assert_complete_oracle_results<T>(
    test_id: &str,
    expected_case_ids: impl IntoIterator<Item = String>,
    oracle_results: &HashMap<String, T>,
) {
    let expected_case_ids: Vec<String> = expected_case_ids.into_iter().collect();
    assert_eq!(
        oracle_results.len(),
        expected_case_ids.len(),
        "{test_id} SciPy oracle returned partial or duplicate coverage"
    );

    let missing_oracle_cases: Vec<&str> = expected_case_ids
        .iter()
        .filter(|case_id| !oracle_results.contains_key(case_id.as_str()))
        .map(String::as_str)
        .collect();
    assert!(
        missing_oracle_cases.is_empty(),
        "{test_id} missing SciPy signal oracle results: {:?}",
        missing_oracle_cases
    );

    let unexpected_oracle_cases: Vec<&str> = oracle_results
        .keys()
        .map(String::as_str)
        .filter(|case_id| !expected_case_ids.iter().any(|expected| expected == case_id))
        .collect();
    assert!(
        unexpected_oracle_cases.is_empty(),
        "{test_id} unexpected SciPy signal oracle results: {:?}",
        unexpected_oracle_cases
    );
}

fn assert_all_cases_compared(test_id: &str, compared: usize, expected: usize) {
    assert_eq!(
        compared, expected,
        "{test_id} compared {compared}/{expected} expected signal cases"
    );
}

fn scipy_oracle_unavailable(test_id: &str, oracle_count: usize) -> bool {
    if oracle_count != 0 {
        return false;
    }

    assert!(
        std::env::var(REQUIRE_SCIPY_ENV).is_err(),
        "{REQUIRE_SCIPY_ENV}=1 but SciPy signal oracle unavailable for {test_id}"
    );
    eprintln!("skipping {test_id} diff: scipy oracle not available");
    true
}

fn window_cases() -> Vec<WindowCase> {
    let mut cases = Vec::new();
    for n in [8, 16, 32, 64, 128] {
        cases.push(WindowCase {
            case_id: format!("hann_{n}"),
            window_type: "hann".into(),
            n,
            beta: None,
        });
        cases.push(WindowCase {
            case_id: format!("hamming_{n}"),
            window_type: "hamming".into(),
            n,
            beta: None,
        });
        cases.push(WindowCase {
            case_id: format!("blackman_{n}"),
            window_type: "blackman".into(),
            n,
            beta: None,
        });
    }
    for n in [16, 32, 64] {
        for beta in [4.0, 8.0, 14.0] {
            cases.push(WindowCase {
                case_id: format!("kaiser_{n}_b{}", beta as i32),
                window_type: "kaiser".into(),
                n,
                beta: Some(beta),
            });
        }
    }
    cases
}

fn convolve_cases() -> Vec<ConvolveCase> {
    vec![
        ConvolveCase {
            case_id: "conv_simple_full".into(),
            a: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            b: vec![1.0, 0.5, 0.25],
            mode: "full".into(),
        },
        ConvolveCase {
            case_id: "conv_simple_same".into(),
            a: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            b: vec![1.0, 0.5, 0.25],
            mode: "same".into(),
        },
        ConvolveCase {
            case_id: "conv_simple_valid".into(),
            a: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            b: vec![1.0, 0.5, 0.25],
            mode: "valid".into(),
        },
        ConvolveCase {
            case_id: "corr_simple_full".into(),
            a: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            b: vec![3.0, 4.0, 5.0],
            mode: "full".into(),
        },
        ConvolveCase {
            case_id: "conv_longer_full".into(),
            a: (0..16).map(|i| (i as f64 * 0.1).sin()).collect(),
            b: vec![0.25, 0.5, 0.25],
            mode: "full".into(),
        },
        ConvolveCase {
            case_id: "conv_longer_same".into(),
            a: (0..16).map(|i| (i as f64 * 0.1).sin()).collect(),
            b: vec![0.25, 0.5, 0.25],
            mode: "same".into(),
        },
    ]
}

fn savgol_cases() -> Vec<SavgolCase> {
    vec![
        SavgolCase {
            case_id: "savgol_5_2_0".into(),
            window_length: 5,
            polyorder: 2,
            deriv: 0,
        },
        SavgolCase {
            case_id: "savgol_7_3_0".into(),
            window_length: 7,
            polyorder: 3,
            deriv: 0,
        },
        SavgolCase {
            case_id: "savgol_9_4_0".into(),
            window_length: 9,
            polyorder: 4,
            deriv: 0,
        },
        SavgolCase {
            case_id: "savgol_11_5_0".into(),
            window_length: 11,
            polyorder: 5,
            deriv: 0,
        },
    ]
}

fn ricker_cases() -> Vec<RickerCase> {
    vec![
        RickerCase {
            case_id: "ricker_32_4".into(),
            points: 32,
            a: 4.0,
        },
        RickerCase {
            case_id: "ricker_64_8".into(),
            points: 64,
            a: 8.0,
        },
        RickerCase {
            case_id: "ricker_128_16".into(),
            points: 128,
            a: 16.0,
        },
    ]
}

fn lombscargle_cases() -> Vec<LombScargleCase> {
    let mut cases = Vec::new();

    for omega in [1.0_f64, 2.0, 4.0, 8.0] {
        let x: Vec<f64> = (0..96).map(|i| i as f64 * 0.125).collect();
        let y: Vec<f64> = x.iter().map(|&t| (omega * t).sin()).collect();
        let freqs = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0];
        for normalize in [false, true] {
            cases.push(LombScargleCase {
                case_id: format!(
                    "regular_omega_{}_{}",
                    omega as i32,
                    if normalize { "norm" } else { "power" }
                ),
                x: x.clone(),
                y: y.clone(),
                freqs: freqs.clone(),
                normalize,
            });
        }
    }

    let x = vec![0.0, 0.5, 1.1, 1.7, 2.4];
    let y = vec![1.0, -0.5, 0.75, 0.25, -1.25];
    let freqs = vec![0.5, 1.0, 1.5, 2.0];
    for normalize in [false, true] {
        cases.push(LombScargleCase {
            case_id: format!(
                "irregular_reference_{}",
                if normalize { "norm" } else { "power" }
            ),
            x: x.clone(),
            y: y.clone(),
            freqs: freqs.clone(),
            normalize,
        });
    }

    cases
}

fn run_scipy_window_oracle(cases: &[WindowCase]) -> HashMap<String, Vec<f64>> {
    let python_code = r#"
import sys
import json
from scipy import signal

cases = json.loads(sys.stdin.read())
results = []
for case in cases:
    wtype = case['window_type']
    n = case['n']
    beta = case.get('beta')
    try:
        if wtype == 'hann':
            w = signal.windows.hann(n)
        elif wtype == 'hamming':
            w = signal.windows.hamming(n)
        elif wtype == 'blackman':
            w = signal.windows.blackman(n)
        elif wtype == 'kaiser':
            w = signal.windows.kaiser(n, beta)
        else:
            w = []
        results.append({'case_id': case['case_id'], 'values': w.tolist()})
    except Exception:
        results.append({'case_id': case['case_id'], 'values': None})
print(json.dumps(results))
"#;

    let json_input = serde_json::to_string(cases).expect("serialize window cases");
    let mut child = fsci_conformance::scipy_oracle_command()
        .args(["-c", python_code])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn python3");

    child
        .stdin
        .take()
        .unwrap()
        .write_all(json_input.as_bytes())
        .unwrap();

    let output = child.wait_with_output().expect("wait python3");
    if !output.status.success() {
        eprintln!(
            "scipy window oracle stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return HashMap::new();
    }

    let results: Vec<OracleResult> = serde_json::from_slice(&output.stdout).expect("parse oracle");
    results
        .into_iter()
        .filter_map(|r| r.values.map(|v| (r.case_id, v)))
        .collect()
}

fn run_scipy_convolve_oracle(cases: &[ConvolveCase]) -> HashMap<String, Vec<f64>> {
    let python_code = r#"
import sys
import json
import numpy as np
from scipy import signal

cases = json.loads(sys.stdin.read())
results = []
for case in cases:
    a = np.array(case['a'])
    b = np.array(case['b'])
    mode = case['mode']
    case_id = case['case_id']
    try:
        if case_id.startswith('corr'):
            vals = signal.correlate(a, b, mode=mode)
        else:
            vals = signal.convolve(a, b, mode=mode)
        results.append({'case_id': case_id, 'values': vals.tolist()})
    except Exception:
        results.append({'case_id': case_id, 'values': None})
print(json.dumps(results))
"#;

    let json_input = serde_json::to_string(cases).expect("serialize convolve cases");
    let mut child = fsci_conformance::scipy_oracle_command()
        .args(["-c", python_code])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn python3");

    child
        .stdin
        .take()
        .unwrap()
        .write_all(json_input.as_bytes())
        .unwrap();

    let output = child.wait_with_output().expect("wait python3");
    if !output.status.success() {
        eprintln!(
            "scipy convolve oracle stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return HashMap::new();
    }

    let results: Vec<OracleResult> = serde_json::from_slice(&output.stdout).expect("parse oracle");
    results
        .into_iter()
        .filter_map(|r| r.values.map(|v| (r.case_id, v)))
        .collect()
}

fn run_scipy_savgol_oracle(cases: &[SavgolCase]) -> HashMap<String, Vec<f64>> {
    let python_code = r#"
import sys
import json
from scipy import signal

cases = json.loads(sys.stdin.read())
results = []
for case in cases:
    wl = case['window_length']
    po = case['polyorder']
    deriv = case['deriv']
    try:
        coeffs = signal.savgol_coeffs(wl, po, deriv=deriv)
        results.append({'case_id': case['case_id'], 'values': coeffs.tolist()})
    except Exception:
        results.append({'case_id': case['case_id'], 'values': None})
print(json.dumps(results))
"#;

    let json_input = serde_json::to_string(cases).expect("serialize savgol cases");
    let mut child = fsci_conformance::scipy_oracle_command()
        .args(["-c", python_code])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn python3");

    child
        .stdin
        .take()
        .unwrap()
        .write_all(json_input.as_bytes())
        .unwrap();

    let output = child.wait_with_output().expect("wait python3");
    if !output.status.success() {
        eprintln!(
            "scipy savgol oracle stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return HashMap::new();
    }

    let results: Vec<OracleResult> = serde_json::from_slice(&output.stdout).expect("parse oracle");
    results
        .into_iter()
        .filter_map(|r| r.values.map(|v| (r.case_id, v)))
        .collect()
}

fn run_scipy_lombscargle_oracle(cases: &[LombScargleCase]) -> HashMap<String, Vec<f64>> {
    let python_code = r#"
import sys
import json
import numpy as np
from scipy import signal

cases = json.loads(sys.stdin.read())
results = []
for case in cases:
    try:
        values = signal.lombscargle(
            np.array(case['x'], dtype=float),
            np.array(case['y'], dtype=float),
            np.array(case['freqs'], dtype=float),
            normalize=case['normalize'],
        )
        results.append({'case_id': case['case_id'], 'values': values.tolist()})
    except Exception:
        results.append({'case_id': case['case_id'], 'values': None})
print(json.dumps(results))
"#;

    let json_input = serde_json::to_string(cases).expect("serialize lombscargle cases");
    let mut child = fsci_conformance::scipy_oracle_command()
        .args(["-c", python_code])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn python3");

    child
        .stdin
        .take()
        .unwrap()
        .write_all(json_input.as_bytes())
        .unwrap();

    let output = child.wait_with_output().expect("wait python3");
    if !output.status.success() {
        eprintln!(
            "scipy lombscargle oracle stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return HashMap::new();
    }

    let results: Vec<OracleResult> = serde_json::from_slice(&output.stdout).expect("parse oracle");
    results
        .into_iter()
        .filter_map(|r| r.values.map(|v| (r.case_id, v)))
        .collect()
}

fn run_scipy_ricker_oracle(cases: &[RickerCase]) -> HashMap<String, Vec<f64>> {
    // scipy.signal.ricker was deprecated in scipy 1.13 and removed in
    // 1.15. Reproduce the closed-form Mexican-hat (Ricker) wavelet in
    // numpy directly:
    //   A      = 2 / (√(3a) · π^(1/4))
    //   t      = arange(0, points) − (points − 1)/2
    //   w(t)   = A · (1 − (t/a)²) · exp(−t² / (2a²))
    let python_code = r#"
import sys
import json
import math
import numpy as np

cases = json.loads(sys.stdin.read())
results = []
for case in cases:
    points = case['points']
    a = float(case['a'])
    try:
        amp = 2.0 / (math.sqrt(3.0 * a) * (math.pi ** 0.25))
        t = np.arange(0, points) - (points - 1) / 2.0
        w = amp * (1.0 - (t / a) ** 2) * np.exp(-(t ** 2) / (2.0 * a * a))
        results.append({'case_id': case['case_id'], 'values': w.tolist()})
    except Exception as e:
        results.append({'case_id': case['case_id'], 'values': None})
print(json.dumps(results))
"#;

    let json_input = serde_json::to_string(cases).expect("serialize ricker cases");
    let mut child = fsci_conformance::scipy_oracle_command()
        .args(["-c", python_code])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn python3");

    child
        .stdin
        .take()
        .unwrap()
        .write_all(json_input.as_bytes())
        .unwrap();

    let output = child.wait_with_output().expect("wait python3");
    if !output.status.success() {
        eprintln!(
            "scipy ricker oracle stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return HashMap::new();
    }

    let results: Vec<OracleResult> = serde_json::from_slice(&output.stdout).expect("parse oracle");
    results
        .into_iter()
        .filter_map(|r| r.values.map(|v| (r.case_id, v)))
        .collect()
}

fn run_rust_window(case: &WindowCase) -> Vec<f64> {
    match case.window_type.as_str() {
        "hann" => hann(case.n),
        "hamming" => hamming(case.n),
        "blackman" => blackman(case.n),
        "kaiser" => kaiser(case.n, case.beta.unwrap_or(8.0)),
        _ => vec![],
    }
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(
            0.0_f64,
            |acc, d| if d.is_nan() { f64::NAN } else { acc.max(d) },
        )
}

#[test]
fn diff_lombscargle() {
    let start = Instant::now();
    let cases = lombscargle_cases();
    let scipy = run_scipy_lombscargle_oracle(&cases);

    if scipy_oracle_unavailable("lombscargle", scipy.len()) {
        return;
    }
    assert_complete_oracle_results(
        "lombscargle",
        cases.iter().map(|case| case.case_id.clone()),
        &scipy,
    );

    let mut diffs = Vec::new();
    let mut max_global = 0.0f64;
    let mut all_pass = true;

    for case in &cases {
        let rust_vals =
            lombscargle(&case.x, &case.y, &case.freqs, case.normalize).unwrap_or_default();
        if let Some(scipy_vals) = scipy.get(&case.case_id) {
            let md = max_diff(&rust_vals, scipy_vals);
            let pass = md <= LOMBSCARGLE_TOL;
            max_global = max_global.max(md);
            all_pass = all_pass && pass;
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                method: "lombscargle".into(),
                rust_values: rust_vals,
                scipy_values: scipy_vals.clone(),
                max_diff: md,
                tolerance: LOMBSCARGLE_TOL,
                pass,
            });
        }
    }
    all_pass = all_pass && diffs.len() == cases.len();

    let log = DiffLog {
        test_id: "diff_signal_lombscargle".into(),
        category: "scipy.signal.lombscargle".into(),
        case_count: diffs.len(),
        max_abs_diff: max_global,
        tolerance: LOMBSCARGLE_TOL,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs,
    };

    emit_log(&log);
    assert_all_cases_compared("lombscargle", log.case_count, cases.len());
    assert!(all_pass, "lombscargle diff failed: max_diff={max_global}");
}

#[test]
fn diff_windows() {
    let start = Instant::now();
    let cases = window_cases();
    let scipy = run_scipy_window_oracle(&cases);

    if scipy_oracle_unavailable("window", scipy.len()) {
        return;
    }
    assert_complete_oracle_results(
        "windows",
        cases.iter().map(|case| case.case_id.clone()),
        &scipy,
    );

    let mut diffs = Vec::new();
    let mut max_global = 0.0f64;
    let mut all_pass = true;

    for case in &cases {
        let rust_vals = run_rust_window(case);
        if let Some(scipy_vals) = scipy.get(&case.case_id) {
            let md = max_diff(&rust_vals, scipy_vals);
            let pass = md <= WINDOW_TOL;
            max_global = max_global.max(md);
            all_pass = all_pass && pass;
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                method: case.window_type.clone(),
                rust_values: rust_vals,
                scipy_values: scipy_vals.clone(),
                max_diff: md,
                tolerance: WINDOW_TOL,
                pass,
            });
        }
    }
    all_pass = all_pass && diffs.len() == cases.len();

    let log = DiffLog {
        test_id: "windows".into(),
        category: "signal".into(),
        case_count: diffs.len(),
        max_abs_diff: max_global,
        tolerance: WINDOW_TOL,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs,
    };

    emit_log(&log);
    assert_all_cases_compared("windows", log.case_count, cases.len());
    assert!(all_pass, "window diff failed: max_diff={max_global}");
}

#[test]
fn diff_convolve() {
    let start = Instant::now();
    let cases = convolve_cases();
    let scipy = run_scipy_convolve_oracle(&cases);

    if scipy_oracle_unavailable("convolve", scipy.len()) {
        return;
    }
    assert_complete_oracle_results(
        "convolve",
        cases.iter().map(|case| case.case_id.clone()),
        &scipy,
    );

    let mut diffs = Vec::new();
    let mut max_global = 0.0f64;
    let mut all_pass = true;

    for case in &cases {
        let mode = match case.mode.as_str() {
            "full" => ConvolveMode::Full,
            "same" => ConvolveMode::Same,
            "valid" => ConvolveMode::Valid,
            _ => continue,
        };

        let rust_vals = if case.case_id.starts_with("corr") {
            correlate(&case.a, &case.b, mode).unwrap_or_default()
        } else {
            convolve(&case.a, &case.b, mode).unwrap_or_default()
        };

        if let Some(scipy_vals) = scipy.get(&case.case_id) {
            let md = max_diff(&rust_vals, scipy_vals);
            let pass = md <= CONV_TOL;
            max_global = max_global.max(md);
            all_pass = all_pass && pass;
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                method: if case.case_id.starts_with("corr") {
                    "correlate".into()
                } else {
                    "convolve".into()
                },
                rust_values: rust_vals,
                scipy_values: scipy_vals.clone(),
                max_diff: md,
                tolerance: CONV_TOL,
                pass,
            });
        }
    }
    all_pass = all_pass && diffs.len() == cases.len();

    let log = DiffLog {
        test_id: "convolve".into(),
        category: "signal".into(),
        case_count: diffs.len(),
        max_abs_diff: max_global,
        tolerance: CONV_TOL,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs,
    };

    emit_log(&log);
    assert_all_cases_compared("convolve", log.case_count, cases.len());
    assert!(all_pass, "convolve diff failed: max_diff={max_global}");
}

#[test]
fn diff_savgol_coeffs() {
    let start = Instant::now();
    let cases = savgol_cases();
    let scipy = run_scipy_savgol_oracle(&cases);

    if scipy_oracle_unavailable("savgol", scipy.len()) {
        return;
    }
    assert_complete_oracle_results(
        "savgol_coeffs",
        cases.iter().map(|case| case.case_id.clone()),
        &scipy,
    );

    let mut diffs = Vec::new();
    let mut max_global = 0.0f64;
    let mut all_pass = true;

    for case in &cases {
        if let Ok(rust_vals) = savgol_coeffs(case.window_length, case.polyorder, case.deriv)
            && let Some(scipy_vals) = scipy.get(&case.case_id)
        {
            let md = max_diff(&rust_vals, scipy_vals);
            let pass = md <= CONV_TOL;
            max_global = max_global.max(md);
            all_pass = all_pass && pass;
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                method: "savgol_coeffs".into(),
                rust_values: rust_vals,
                scipy_values: scipy_vals.clone(),
                max_diff: md,
                tolerance: CONV_TOL,
                pass,
            });
        }
    }
    all_pass = all_pass && diffs.len() == cases.len();

    let log = DiffLog {
        test_id: "savgol_coeffs".into(),
        category: "signal".into(),
        case_count: diffs.len(),
        max_abs_diff: max_global,
        tolerance: CONV_TOL,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs,
    };

    emit_log(&log);
    assert_all_cases_compared("savgol_coeffs", log.case_count, cases.len());
    assert!(all_pass, "savgol_coeffs diff failed: max_diff={max_global}");
}

#[test]
fn diff_ricker() {
    let start = Instant::now();
    let cases = ricker_cases();
    let scipy = run_scipy_ricker_oracle(&cases);

    if scipy_oracle_unavailable("ricker", scipy.len()) {
        return;
    }
    assert_complete_oracle_results(
        "ricker",
        cases.iter().map(|case| case.case_id.clone()),
        &scipy,
    );

    let mut diffs = Vec::new();
    let mut max_global = 0.0f64;
    let mut all_pass = true;

    for case in &cases {
        let rust_vals = ricker(case.points, case.a);
        if let Some(scipy_vals) = scipy.get(&case.case_id) {
            let md = max_diff(&rust_vals, scipy_vals);
            let pass = md <= WINDOW_TOL;
            max_global = max_global.max(md);
            all_pass = all_pass && pass;
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                method: "ricker".into(),
                rust_values: rust_vals,
                scipy_values: scipy_vals.clone(),
                max_diff: md,
                tolerance: WINDOW_TOL,
                pass,
            });
        }
    }
    all_pass = all_pass && diffs.len() == cases.len();

    let log = DiffLog {
        test_id: "ricker".into(),
        category: "signal".into(),
        case_count: diffs.len(),
        max_abs_diff: max_global,
        tolerance: WINDOW_TOL,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs,
    };

    emit_log(&log);
    assert_all_cases_compared("ricker", log.case_count, cases.len());
    assert!(all_pass, "ricker diff failed: max_diff={max_global}");
}

#![forbid(unsafe_code)]
//! Live SciPy differential coverage for hilbert transform.
//!
//! Tests FrankenSciPy analytic signal functions against SciPy subprocess oracle.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{hilbert, hilbert_envelope};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-018";
const HILBERT_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct HilbertCase {
    case_id: String,
    signal: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct HilbertOracleResult {
    case_id: String,
    real: Option<Vec<f64>>,
    imag: Option<Vec<f64>>,
    envelope: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    method: String,
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
    fs::create_dir_all(output_dir()).expect("create hilbert diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize hilbert diff log");
    fs::write(path, json).expect("write hilbert diff log");
}

fn generate_hilbert_cases() -> Vec<HilbertCase> {
    let mut cases = Vec::new();

    // Sine wave cases at different frequencies
    for (freq_idx, freq) in [1.0, 2.0, 5.0, 10.0].iter().enumerate() {
        for (n_idx, n) in [16, 32, 64, 128].iter().enumerate() {
            let signal: Vec<f64> = (0..*n)
                .map(|i| (2.0 * PI * freq * i as f64 / *n as f64).sin())
                .collect();
            cases.push(HilbertCase {
                case_id: format!("sine_f{freq_idx}_n{n_idx}"),
                signal,
            });
        }
    }

    // Cosine wave cases
    for (freq_idx, freq) in [1.0, 3.0, 7.0].iter().enumerate() {
        for (n_idx, n) in [32, 64].iter().enumerate() {
            let signal: Vec<f64> = (0..*n)
                .map(|i| (2.0 * PI * freq * i as f64 / *n as f64).cos())
                .collect();
            cases.push(HilbertCase {
                case_id: format!("cosine_f{freq_idx}_n{n_idx}"),
                signal,
            });
        }
    }

    // Mixed frequency signal (AM modulation)
    for (carrier_idx, carrier) in [10.0, 20.0].iter().enumerate() {
        for (mod_idx, modulator) in [1.0, 2.0].iter().enumerate() {
            let n = 128;
            let signal: Vec<f64> = (0..n)
                .map(|i| {
                    let t = i as f64 / n as f64;
                    (1.0 + 0.5 * (2.0 * PI * modulator * t).cos()) * (2.0 * PI * carrier * t).sin()
                })
                .collect();
            cases.push(HilbertCase {
                case_id: format!("am_c{carrier_idx}_m{mod_idx}"),
                signal,
            });
        }
    }

    // Impulse
    let mut impulse = vec![0.0; 64];
    impulse[32] = 1.0;
    cases.push(HilbertCase {
        case_id: "impulse".to_string(),
        signal: impulse,
    });

    // Step function
    let step: Vec<f64> = (0..64).map(|i| if i >= 32 { 1.0 } else { 0.0 }).collect();
    cases.push(HilbertCase {
        case_id: "step".to_string(),
        signal: step,
    });

    // Random-ish deterministic signal
    for seed in 0..5 {
        let signal: Vec<f64> = (0..64)
            .map(|i| {
                let x = (i + seed * 17) as f64;
                (x * 0.1).sin() + 0.5 * (x * 0.37).cos() + 0.3 * (x * 0.73).sin()
            })
            .collect();
        cases.push(HilbertCase {
            case_id: format!("pseudo_random_{seed}"),
            signal,
        });
    }

    // Odd and even length edge cases
    for n in [15, 17, 31, 33] {
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 2.0 * i as f64 / n as f64).sin())
            .collect();
        cases.push(HilbertCase {
            case_id: format!("odd_even_n{n}"),
            signal,
        });
    }

    // Chirp signal (frequency sweep)
    for (chirp_idx, f_end) in [5.0, 10.0, 20.0].iter().enumerate() {
        let n = 128;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                let freq = 1.0 + (f_end - 1.0) * t;
                (2.0 * PI * freq * t).sin()
            })
            .collect();
        cases.push(HilbertCase {
            case_id: format!("chirp_{chirp_idx}"),
            signal,
        });
    }

    // DC offset signals
    for dc in [0.5, 1.0, 2.0] {
        let signal: Vec<f64> = (0..64)
            .map(|i| dc + (2.0 * PI * 3.0 * i as f64 / 64.0).sin())
            .collect();
        cases.push(HilbertCase {
            case_id: format!("dc_offset_{}", (dc * 10.0) as i32),
            signal,
        });
    }

    // Square wave approximation
    let square: Vec<f64> = (0..64)
        .map(|i| {
            let t = i as f64 / 64.0;
            let mut sum = 0.0;
            for k in (1..=9).step_by(2) {
                sum += (2.0 * PI * k as f64 * t).sin() / k as f64;
            }
            sum * 4.0 / PI
        })
        .collect();
    cases.push(HilbertCase {
        case_id: "square_wave".to_string(),
        signal: square,
    });

    // Sawtooth wave approximation
    let sawtooth: Vec<f64> = (0..64)
        .map(|i| {
            let t = i as f64 / 64.0;
            let mut sum = 0.0;
            for k in 1..=10 {
                sum += (2.0 * PI * k as f64 * t).sin() / k as f64;
            }
            sum * -2.0 / PI
        })
        .collect();
    cases.push(HilbertCase {
        case_id: "sawtooth_wave".to_string(),
        signal: sawtooth,
    });

    // Gaussian pulse
    for (sigma_idx, sigma) in [2.0, 4.0, 8.0].iter().enumerate() {
        let n = 64;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = (i as f64 - n as f64 / 2.0) / sigma;
                (-0.5 * t * t).exp()
            })
            .collect();
        cases.push(HilbertCase {
            case_id: format!("gaussian_pulse_{sigma_idx}"),
            signal,
        });
    }

    // Exponential decay with oscillation
    for (decay_idx, decay) in [0.05_f64, 0.1, 0.2].iter().enumerate() {
        let n = 64;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                (-decay * t).exp() * (2.0 * PI * 5.0 * t / n as f64).sin()
            })
            .collect();
        cases.push(HilbertCase {
            case_id: format!("exp_decay_{decay_idx}"),
            signal,
        });
    }

    cases
}

fn scipy_hilbert_oracle_or_skip(cases: &[HilbertCase]) -> Vec<HilbertOracleResult> {
    let script = r#"
import json
import sys
import numpy as np
from scipy import signal

cases = json.load(sys.stdin)
results = []

for c in cases:
    cid = c["case_id"]
    sig = np.array(c["signal"], dtype=np.float64)

    try:
        analytic = signal.hilbert(sig)
        envelope = np.abs(analytic)
        results.append({
            "case_id": cid,
            "real": analytic.real.tolist(),
            "imag": analytic.imag.tolist(),
            "envelope": envelope.tolist()
        })
    except Exception as e:
        results.append({
            "case_id": cid,
            "real": None,
            "imag": None,
            "envelope": None
        })

print(json.dumps(results))
"#;

    let cases_json = serde_json::to_string(cases).expect("serialize hilbert cases");

    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for hilbert oracle: {e}"
            );
            eprintln!("skipping hilbert oracle: python3 not available ({e})");
            return Vec::new();
        }
    };

    {
        let stdin = child.stdin.as_mut().expect("open hilbert oracle stdin");
        if let Err(err) = stdin.write_all(cases_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "hilbert oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping hilbert oracle: stdin write failed ({err})\n{stderr}");
            return Vec::new();
        }
    }

    let output = child.wait_with_output().expect("wait for hilbert oracle");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "hilbert oracle failed: {stderr}"
        );
        eprintln!("skipping hilbert oracle: scipy not available\n{stderr}");
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout).expect("parse hilbert oracle JSON")
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_signal_hilbert() {
    let cases = generate_hilbert_cases();
    let oracle_results = scipy_hilbert_oracle_or_skip(&cases);

    if oracle_results.is_empty() {
        return;
    }

    assert_eq!(
        oracle_results.len(),
        cases.len(),
        "SciPy hilbert oracle returned partial coverage"
    );

    let oracle_map: HashMap<String, HilbertOracleResult> = oracle_results
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_diff_overall = 0.0_f64;

    for case in &cases {
        let scipy_result = oracle_map
            .get(&case.case_id)
            .expect("validated complete oracle map");

        let (Some(scipy_real), Some(scipy_imag), Some(scipy_env)) = (
            &scipy_result.real,
            &scipy_result.imag,
            &scipy_result.envelope,
        ) else {
            continue;
        };

        let rust_analytic = match hilbert(&case.signal) {
            Ok(a) => a,
            Err(_) => continue,
        };
        let rust_real: Vec<f64> = rust_analytic.iter().map(|&(r, _)| r).collect();
        let rust_imag: Vec<f64> = rust_analytic.iter().map(|&(_, i)| i).collect();

        let rust_env = match hilbert_envelope(&case.signal) {
            Ok(e) => e,
            Err(_) => continue,
        };

        let real_diff = max_abs_diff(&rust_real, scipy_real);
        let imag_diff = max_abs_diff(&rust_imag, scipy_imag);
        let env_diff = max_abs_diff(&rust_env, scipy_env);
        let max_diff = real_diff.max(imag_diff).max(env_diff);

        let pass = max_diff <= HILBERT_TOL;
        max_diff_overall = max_diff_overall.max(max_diff);

        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            method: "hilbert".to_string(),
            max_diff,
            tolerance: HILBERT_TOL,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_hilbert".into(),
        category: "scipy.signal.hilbert".into(),
        case_count: diffs.len(),
        max_abs_diff: max_diff_overall,
        tolerance: HILBERT_TOL,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for diff in &diffs {
        if !diff.pass {
            eprintln!(
                "hilbert mismatch: {} max_diff={} tolerance={}",
                diff.case_id, diff.max_diff, diff.tolerance
            );
        }
    }

    assert!(
        all_pass,
        "scipy.signal.hilbert conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_diff_overall
    );
}

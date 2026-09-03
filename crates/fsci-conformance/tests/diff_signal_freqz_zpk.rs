#![forbid(unsafe_code)]
//! Live SciPy differential coverage for scipy.signal.freqz_zpk.
//!
//! Resolves [frankenscipy-sr37d]. The freqz_zpk port (edbb00e)
//! has deterministic anchor tests but no live scipy comparison.
//! Drives curated zero/pole/gain configurations through
//! scipy.signal.freqz_zpk via subprocess oracle and diffs
//! h_mag and h_phase. Skips cleanly if scipy is unavailable.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{ZpkCoeffs, freqz_zpk};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-011";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct ZpkCase {
    case_id: String,
    zeros_re: Vec<f64>,
    zeros_im: Vec<f64>,
    poles_re: Vec<f64>,
    poles_im: Vec<f64>,
    gain: f64,
    n_freqs: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct ZpkOracleResult {
    case_id: String,
    h_mag: Option<Vec<f64>>,
    h_phase: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    max_mag_diff: f64,
    max_phase_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    abs_tol: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create freqz_zpk diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize freqz_zpk diff log");
    fs::write(path, json).expect("write freqz_zpk diff log");
}

fn generate_cases() -> Vec<ZpkCase> {
    let mut cases = Vec::new();
    let configs: &[(&str, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64)] = &[
        // Real-pole/zero filter.
        (
            "real_zeros_real_poles",
            vec![1.0, -1.0],
            vec![0.0, 0.0],
            vec![0.5, -0.5],
            vec![0.0, 0.0],
            0.5,
        ),
        // Single zero at 1.
        (
            "single_real_zero",
            vec![1.0],
            vec![0.0],
            vec![],
            vec![],
            1.0,
        ),
        // Conjugate-pair complex poles.
        (
            "complex_pole_pair",
            vec![],
            vec![],
            vec![0.5, 0.5],
            vec![0.5, -0.5],
            1.0,
        ),
        // Mixed: one real zero, complex pole pair.
        (
            "mixed",
            vec![-1.0],
            vec![0.0],
            vec![0.6, 0.6],
            vec![0.3, -0.3],
            0.7,
        ),
        // Higher gain.
        (
            "high_gain",
            vec![0.0],
            vec![0.0],
            vec![0.4],
            vec![0.0],
            10.0,
        ),
    ];

    for (label, zr, zi, pr, pi, k) in configs {
        for &n_freqs in &[8_usize, 32, 128] {
            cases.push(ZpkCase {
                case_id: format!("{label}_n{n_freqs}"),
                zeros_re: zr.clone(),
                zeros_im: zi.clone(),
                poles_re: pr.clone(),
                poles_im: pi.clone(),
                gain: *k,
                n_freqs,
            });
        }
    }

    cases
}

fn scipy_oracle_or_skip(cases: &[ZpkCase]) -> Vec<ZpkOracleResult> {
    let script = r#"
import json
import sys
import numpy as np
from scipy import signal

cases = json.load(sys.stdin)
results = []
for c in cases:
    cid = c["case_id"]
    z = np.array([complex(re, im) for re, im in zip(c["zeros_re"], c["zeros_im"])], dtype=complex)
    p = np.array([complex(re, im) for re, im in zip(c["poles_re"], c["poles_im"])], dtype=complex)
    k = float(c["gain"])
    n = int(c["n_freqs"])
    try:
        w, h = signal.freqz_zpk(z, p, k, worN=n, whole=False)
        h_mag = np.abs(h).tolist()
        h_phase = np.angle(h).tolist()
        results.append({
            "case_id": cid,
            "h_mag": list(map(float, h_mag)),
            "h_phase": list(map(float, h_phase)),
        })
    except Exception:
        results.append({"case_id": cid, "h_mag": None, "h_phase": None})

print(json.dumps(results))
"#;

    let cases_json = serde_json::to_string(cases).expect("serialize freqz_zpk cases");

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
                "failed to spawn python3 for freqz_zpk oracle: {e}"
            );
            eprintln!("skipping freqz_zpk oracle: python3 not available ({e})");
            return Vec::new();
        }
    };

    {
        let stdin = child.stdin.as_mut().expect("open freqz_zpk oracle stdin");
        if let Err(err) = stdin.write_all(cases_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "freqz_zpk oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping freqz_zpk oracle: stdin write failed ({err})\n{stderr}");
            return Vec::new();
        }
    }

    let output = child.wait_with_output().expect("wait for freqz_zpk oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "freqz_zpk oracle failed: {stderr}"
        );
        eprintln!("skipping freqz_zpk oracle: scipy not available\n{stderr}");
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout).expect("parse freqz_zpk oracle JSON")
}

#[test]
fn diff_signal_freqz_zpk() {
    let cases = generate_cases();
    let oracle_results = scipy_oracle_or_skip(&cases);

    if oracle_results.is_empty() {
        return;
    }

    assert_eq!(
        oracle_results.len(),
        cases.len(),
        "scipy freqz_zpk oracle returned partial coverage"
    );

    let oracle_map: HashMap<String, ZpkOracleResult> = oracle_results
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &cases {
        let oracle = oracle_map
            .get(&case.case_id)
            .expect("validated complete oracle map");
        let (Some(scipy_mag), Some(scipy_phase)) = (&oracle.h_mag, &oracle.h_phase) else {
            continue;
        };

        let zpk = ZpkCoeffs {
            zeros_re: case.zeros_re.clone(),
            zeros_im: case.zeros_im.clone(),
            poles_re: case.poles_re.clone(),
            poles_im: case.poles_im.clone(),
            gain: case.gain,
        };
        let r = freqz_zpk(&zpk, Some(case.n_freqs)).expect("freqz_zpk");

        let mut max_mag_diff = 0.0_f64;
        let mut max_phase_diff = 0.0_f64;
        for k in 0..case.n_freqs {
            let m_diff = (r.h_mag[k] - scipy_mag[k]).abs();
            max_mag_diff = max_mag_diff.max(m_diff);
            // Phase has ±2π wrap ambiguity near jumps.
            let raw = (r.h_phase[k] - scipy_phase[k]).abs();
            let wrapped = (raw - std::f64::consts::TAU).abs().min(raw);
            max_phase_diff = max_phase_diff.max(wrapped);
        }
        let pass = max_mag_diff <= ABS_TOL && max_phase_diff <= ABS_TOL;
        max_overall = max_overall.max(max_mag_diff).max(max_phase_diff);

        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            max_mag_diff,
            max_phase_diff,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_freqz_zpk".into(),
        category: "scipy.signal.freqz_zpk".into(),
        case_count: diffs.len(),
        max_abs_diff: max_overall,
        abs_tol: ABS_TOL,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "freqz_zpk mismatch: {} mag_diff={} phase_diff={}",
                d.case_id, d.max_mag_diff, d.max_phase_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.signal.freqz_zpk conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

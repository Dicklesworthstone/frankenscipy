#![forbid(unsafe_code)]
//! Live SciPy differential coverage for scipy.signal.gauspuls.
//!
//! Resolves [frankenscipy-mevjv]. Drives a curated set of
//! (fc, bw, bwr) tuples × t-grid through scipy.signal.gauspuls
//! (retquad=True, retenv=True) and diffs the in-phase, quadrature,
//! and envelope outputs against the just-shipped fsci_signal::gauspuls
//! port (b1b5886). Skips cleanly if scipy/python3 is unavailable
//! unless `FSCI_REQUIRE_SCIPY_ORACLE` is set.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::gauspuls;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-011";
const ABS_TOL: f64 = 1.0e-10;
const REL_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct GauspulsCase {
    case_id: String,
    fc: f64,
    bw: f64,
    bwr: f64,
    t: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct GauspulsOracleResult {
    case_id: String,
    i: Option<Vec<f64>>,
    q: Option<Vec<f64>>,
    envelope: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    max_abs_diff_i: f64,
    max_abs_diff_q: f64,
    max_abs_diff_env: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    abs_tol: f64,
    rel_tol: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create gauspuls diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize gauspuls diff log");
    fs::write(path, json).expect("write gauspuls diff log");
}

fn time_grid(n: usize, half_width: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let lo = -half_width;
            let hi = half_width;
            lo + (hi - lo) * i as f64 / (n - 1) as f64
        })
        .collect()
}

fn generate_cases() -> Vec<GauspulsCase> {
    vec![
        // Default scipy params: fc=1000, bw=0.5, bwr=-6.
        GauspulsCase {
            case_id: "default_1000_0p5_neg6".into(),
            fc: 1000.0,
            bw: 0.5,
            bwr: -6.0,
            t: time_grid(33, 0.001),
        },
        // Wider bandwidth.
        GauspulsCase {
            case_id: "wide_bw".into(),
            fc: 500.0,
            bw: 1.0,
            bwr: -6.0,
            t: time_grid(33, 0.002),
        },
        // Narrow bandwidth.
        GauspulsCase {
            case_id: "narrow_bw".into(),
            fc: 2000.0,
            bw: 0.2,
            bwr: -6.0,
            t: time_grid(33, 0.0005),
        },
        // Stronger -bwr (deeper Gaussian envelope).
        GauspulsCase {
            case_id: "deep_envelope_neg20".into(),
            fc: 1000.0,
            bw: 0.5,
            bwr: -20.0,
            t: time_grid(33, 0.001),
        },
        // High frequency.
        GauspulsCase {
            case_id: "high_fc".into(),
            fc: 1.0e5,
            bw: 0.5,
            bwr: -6.0,
            t: time_grid(33, 1.0e-5),
        },
        // Low frequency.
        GauspulsCase {
            case_id: "low_fc".into(),
            fc: 10.0,
            bw: 0.5,
            bwr: -6.0,
            t: time_grid(33, 0.1),
        },
    ]
}

fn scipy_oracle_or_skip(cases: &[GauspulsCase]) -> Vec<GauspulsOracleResult> {
    let script = r#"
import json
import sys
from scipy import signal

cases = json.load(sys.stdin)
results = []
for c in cases:
    try:
        i_arr, q_arr, env = signal.gauspuls(
            c["t"], fc=c["fc"], bw=c["bw"], bwr=c["bwr"],
            retquad=True, retenv=True,
        )
        results.append({
            "case_id": c["case_id"],
            "i": list(map(float, i_arr)),
            "q": list(map(float, q_arr)),
            "envelope": list(map(float, env)),
        })
    except Exception:
        results.append({
            "case_id": c["case_id"],
            "i": None,
            "q": None,
            "envelope": None,
        })

print(json.dumps(results))
"#;

    let cases_json = serde_json::to_string(cases).expect("serialize gauspuls cases");

    let mut child = match Command::new("python3")
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
                "failed to spawn python3 for gauspuls oracle: {e}"
            );
            eprintln!("skipping gauspuls oracle: python3 not available ({e})");
            return Vec::new();
        }
    };

    {
        let stdin = child.stdin.as_mut().expect("open gauspuls oracle stdin");
        if let Err(err) = stdin.write_all(cases_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "gauspuls oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping gauspuls oracle: stdin write failed ({err})\n{stderr}");
            return Vec::new();
        }
    }

    let output = child.wait_with_output().expect("wait for gauspuls oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "gauspuls oracle failed: {stderr}"
        );
        eprintln!("skipping gauspuls oracle: scipy not available\n{stderr}");
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout).expect("parse gauspuls oracle JSON")
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
fn diff_signal_gauspuls() {
    let cases = generate_cases();
    let oracle_results = scipy_oracle_or_skip(&cases);

    if oracle_results.is_empty() {
        return;
    }

    assert_eq!(
        oracle_results.len(),
        cases.len(),
        "scipy gauspuls oracle returned partial coverage"
    );

    let oracle_map: HashMap<String, GauspulsOracleResult> = oracle_results
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
        let (Some(scipy_i), Some(scipy_q), Some(scipy_env)) =
            (&oracle.i, &oracle.q, &oracle.envelope)
        else {
            continue;
        };

        let r = gauspuls(&case.t, case.fc, case.bw, case.bwr).expect("gauspuls");
        let di = max_abs_diff(&r.i, scipy_i);
        let dq = max_abs_diff(&r.q, scipy_q);
        let denv = max_abs_diff(&r.envelope, scipy_env);
        let case_max = di.max(dq).max(denv);
        let pass = case_max <= ABS_TOL;
        max_overall = max_overall.max(case_max);

        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            max_abs_diff_i: di,
            max_abs_diff_q: dq,
            max_abs_diff_env: denv,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_gauspuls".into(),
        category: "scipy.signal.gauspuls".into(),
        case_count: diffs.len(),
        max_abs_diff: max_overall,
        abs_tol: ABS_TOL,
        rel_tol: REL_TOL,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "gauspuls mismatch: {} i={} q={} env={}",
                d.case_id, d.max_abs_diff_i, d.max_abs_diff_q, d.max_abs_diff_env
            );
        }
    }

    assert!(
        all_pass,
        "scipy.signal.gauspuls conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}

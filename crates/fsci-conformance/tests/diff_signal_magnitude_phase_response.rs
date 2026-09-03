#![forbid(unsafe_code)]
//! Live scipy.signal.freqz parity for fsci_signal::
//! {magnitude_response, magnitude_response_db, phase_response}.
//!
//! Resolves [frankenscipy-15eky]. Compares |H(ω)|, 20·log10|H(ω)|,
//! and arg(H(ω)) on a uniform frequency grid [0, π) at several
//! filter coefficient fixtures.
//!
//! fsci samples ω_k = π·k/n for k = 0..n; scipy.signal.freqz with
//! worN=n returns w_k = π·k/n via `whole=False`. Phase is unwrapped
//! on both sides for direct comparison. Tolerance: 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{magnitude_response, magnitude_response_db, phase_response};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct FilterCase {
    case_id: String,
    b: Vec<f64>,
    a: Vec<f64>,
    n_freqs: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<FilterCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    freqs: Option<Vec<f64>>,
    mag: Option<Vec<f64>>,
    /// `None` entries mark "magnitude was below floor (NEG_INFINITY)".
    mag_db: Option<Vec<Option<f64>>>,
    phase: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create mag/phase diff dir");
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
    // (b, a, n_freqs)
    let cases: Vec<FilterCase> = vec![
        FilterCase {
            case_id: "moving_avg_3".into(),
            b: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            a: vec![1.0],
            n_freqs: 32,
        },
        FilterCase {
            case_id: "exp_smooth_a08".into(),
            b: vec![0.2],
            a: vec![1.0, -0.8],
            n_freqs: 32,
        },
        FilterCase {
            case_id: "biquad_lp_butter".into(),
            b: vec![0.04, 0.08, 0.04],
            a: vec![1.0, -1.4, 0.6],
            n_freqs: 64,
        },
        FilterCase {
            case_id: "biquad_hp".into(),
            b: vec![0.5, -1.0, 0.5],
            a: vec![1.0, -1.0, 0.5],
            n_freqs: 64,
        },
        FilterCase {
            case_id: "fir_5tap".into(),
            b: vec![0.1, 0.2, 0.4, 0.2, 0.1],
            a: vec![1.0],
            n_freqs: 48,
        },
    ];
    OracleQuery { points: cases }
}

// Convert a phase vector to unit complex coordinates so wrapping
// ambiguity vanishes (any 2π offset leaves (cos, sin) unchanged).
fn phase_to_unit(phase: &[f64]) -> Vec<(f64, f64)> {
    phase.iter().map(|p| (p.cos(), p.sin())).collect()
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
    n = int(case["n_freqs"])
    try:
        # match fsci's grid: ω_k = π·k/n for k=0..n
        w = np.array([math.pi * k / n for k in range(n)])
        _, h = signal.freqz(b, a, worN=w, whole=False)
        mag = np.abs(h)
        # fsci uses NEG_INFINITY sentinel for |H|<=1e-30; mirror it here.
        # JSON: serialize as None so the consumer can mark "skip both sides".
        mag_db = []
        for m in mag.tolist():
            if m > 1e-30:
                mag_db.append(20.0 * math.log10(m))
            else:
                mag_db.append(None)
        phase = np.angle(h)
        points.append({
            "case_id": cid,
            "freqs": [float(x) for x in w.tolist()],
            "mag": [float(x) for x in mag.tolist()],
            "mag_db": [None if x is None else float(x) for x in mag_db],
            "phase": [float(x) for x in phase.tolist()],
        })
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "freqs": None, "mag": None, "mag_db": None, "phase": None})
print(json.dumps({"points": points}))
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
                "failed to spawn python3 for mag/phase oracle: {e}"
            );
            eprintln!("skipping mag/phase oracle: python3 not available ({e})");
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
                "mag/phase oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping mag/phase oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for mag/phase oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "mag/phase oracle failed: {stderr}"
        );
        eprintln!("skipping mag/phase oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse mag/phase oracle JSON"))
}

fn vec_max_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_signal_magnitude_phase_response() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let (Some(efreqs), Some(emag), Some(emag_db), Some(ephase)) = (
            arm.freqs.as_ref(),
            arm.mag.as_ref(),
            arm.mag_db.as_ref(),
            arm.phase.as_ref(),
        ) else {
            continue;
        };

        let (mfreqs, mag) = magnitude_response(&case.b, &case.a, case.n_freqs);
        let (_, mag_db) = magnitude_response_db(&case.b, &case.a, case.n_freqs);
        let (_, phase_raw) = phase_response(&case.b, &case.a, case.n_freqs);
        let phase_units = phase_to_unit(&phase_raw);
        let expected_units = phase_to_unit(ephase);

        // frequency grid
        let f_diff = vec_max_diff(&mfreqs, efreqs);
        max_overall = max_overall.max(f_diff);
        diffs.push(CaseDiff {
            case_id: format!("{}_freqs", case.case_id),
            op: "freqs".into(),
            abs_diff: f_diff,
            pass: f_diff <= ABS_TOL,
        });

        let m_diff = vec_max_diff(&mag, emag);
        max_overall = max_overall.max(m_diff);
        diffs.push(CaseDiff {
            case_id: format!("{}_mag", case.case_id),
            op: "magnitude".into(),
            abs_diff: m_diff,
            pass: m_diff <= ABS_TOL,
        });

        // magnitude_db: skip indices where either side flagged a sub-floor magnitude.
        let mdb_diff = if mag_db.len() != emag_db.len() {
            f64::INFINITY
        } else {
            mag_db
                .iter()
                .zip(emag_db.iter())
                .map(|(actual, expected)| match expected {
                    Some(e) if actual.is_finite() => (actual - e).abs(),
                    Some(_) | None => 0.0, // skip
                })
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(mdb_diff);
        diffs.push(CaseDiff {
            case_id: format!("{}_mag_db", case.case_id),
            op: "magnitude_db".into(),
            abs_diff: mdb_diff,
            pass: mdb_diff <= ABS_TOL,
        });

        // phase as (cos, sin) — agnostic to 2π wrapping.
        let p_diff = if phase_units.len() != expected_units.len() {
            f64::INFINITY
        } else {
            phase_units
                .iter()
                .zip(expected_units.iter())
                .map(|(&(ac, as_), &(ec, es))| ((ac - ec).powi(2) + (as_ - es).powi(2)).sqrt())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(p_diff);
        diffs.push(CaseDiff {
            case_id: format!("{}_phase", case.case_id),
            op: "phase".into(),
            abs_diff: p_diff,
            pass: p_diff <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_magnitude_phase_response".into(),
        category: "fsci_signal magnitude/phase response vs scipy.signal.freqz".into(),
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
        "mag/phase response conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

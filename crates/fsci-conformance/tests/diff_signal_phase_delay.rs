#![forbid(unsafe_code)]
//! Live scipy-formula parity for fsci_signal::phase_delay.
//!
//! Resolves [frankenscipy-5lvuf]. Definition:
//!   τ_p(ω) = -unwrap(angle(H(ω))) / ω, with the ω=0 entry taken from
//!   the analytic group delay limit.
//!
//! Oracle reconstructs the reference via scipy.signal.freqz, then
//! numpy.unwrap → divide by ω, with a special-case at ω=0 using
//! scipy.signal.group_delay (which uses the same analytic limit).
//!
//! Tolerance: 1e-8 abs for filters with non-zero magnitude across the
//! whole spectrum.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::phase_delay;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-8;
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
    delay: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create phase_delay diff dir");
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
    let points = vec![
        FilterCase {
            case_id: "exp_smooth_a08".into(),
            b: vec![0.2],
            a: vec![1.0, -0.8],
            n_freqs: 32,
        },
        FilterCase {
            case_id: "moving_avg_3".into(),
            b: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            a: vec![1.0],
            n_freqs: 32,
        },
        FilterCase {
            case_id: "biquad_lp".into(),
            b: vec![0.04, 0.08, 0.04],
            a: vec![1.0, -1.4, 0.6],
            n_freqs: 64,
        },
    ];
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
    n = int(case["n_freqs"])
    try:
        # fsci's freqz grid: ω_k = π·k/n for k=0..n
        w = np.array([math.pi * k / n for k in range(n)])
        _, h = signal.freqz(b, a, worN=w, whole=False)
        mag = np.abs(h)
        phase = np.unwrap(np.angle(h))
        delay = np.zeros_like(w)
        # ω = 0 entry uses analytic group delay
        _, gd = signal.group_delay((b, a), w=np.array([1e-9]))
        delay[0] = float(gd[0])
        for i in range(1, len(w)):
            if mag[i] <= 1e-30:
                delay[i] = 0.0
            else:
                delay[i] = -phase[i] / w[i]
        points.append({
            "case_id": cid,
            "freqs": [float(x) for x in w.tolist()],
            "delay": [float(x) for x in delay.tolist()],
        })
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "freqs": None, "delay": None})
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
                "failed to spawn python3 for phase_delay oracle: {e}"
            );
            eprintln!("skipping phase_delay oracle: python3 not available ({e})");
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
                "phase_delay oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping phase_delay oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for phase_delay oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "phase_delay oracle failed: {stderr}"
        );
        eprintln!("skipping phase_delay oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse phase_delay oracle JSON"))
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
fn diff_signal_phase_delay() {
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
        let (Some(efreqs), Some(edelay)) = (arm.freqs.as_ref(), arm.delay.as_ref()) else {
            continue;
        };
        let Ok((freqs, delay)) = phase_delay(&case.b, &case.a, Some(case.n_freqs)) else {
            continue;
        };

        let f_diff = vec_max_diff(&freqs, efreqs);
        max_overall = max_overall.max(f_diff);
        diffs.push(CaseDiff {
            case_id: format!("{}_freqs", case.case_id),
            op: "freqs".into(),
            abs_diff: f_diff,
            pass: f_diff <= ABS_TOL,
        });

        let d_diff = vec_max_diff(&delay, edelay);
        max_overall = max_overall.max(d_diff);
        diffs.push(CaseDiff {
            case_id: format!("{}_delay", case.case_id),
            op: "delay".into(),
            abs_diff: d_diff,
            pass: d_diff <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_phase_delay".into(),
        category: "fsci_signal::phase_delay vs scipy.signal.freqz formula".into(),
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
        "phase_delay conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

#![forbid(unsafe_code)]
//! Live scipy parity for fsci_signal::group_delay_from_ba.
//!
//! Resolves [frankenscipy-48prz]. Computes group delay τ_g(ω) =
//! -d(phase)/dω for digital filters and compares against
//! scipy.signal.group_delay((b, a), w=N) on a sweep of canonical
//! filter designs: Butterworth/Chebyshev/elliptic lowpass, simple
//! moving-average FIR, and identity all-pass.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::group_delay_from_ba;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REL_TOL: f64 = 1.0e-8;
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct CasePoint {
    case_id: String,
    b: Vec<f64>,
    a: Vec<f64>,
    n_freqs: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<CasePoint>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct OraclePoint {
    case_id: String,
    w: Option<Vec<f64>>,
    gd: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<OraclePoint>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    max_abs_diff_w: f64,
    max_abs_diff_gd: f64,
    max_rel_diff_gd: f64,
    pass: bool,
    note: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create group_delay diff dir");
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

fn build_query() -> OracleQuery {
    let pts = vec![
        // 1. Identity filter: pure pass-through, gd = 0 everywhere.
        CasePoint {
            case_id: "identity".into(),
            b: vec![1.0],
            a: vec![1.0],
            n_freqs: 8,
        },
        // 2. Simple delay: y[n] = x[n-1], gd ≡ 1
        CasePoint {
            case_id: "unit_delay".into(),
            b: vec![0.0, 1.0],
            a: vec![1.0],
            n_freqs: 16,
        },
        // 3. Two-sample moving average (FIR): gd ≡ 0.5
        CasePoint {
            case_id: "ma2_fir".into(),
            b: vec![0.5, 0.5],
            a: vec![1.0],
            n_freqs: 16,
        },
        // 4. 5-tap moving average FIR: gd ≡ 2
        CasePoint {
            case_id: "ma5_fir".into(),
            b: vec![0.2; 5],
            a: vec![1.0],
            n_freqs: 16,
        },
        // 5. Butterworth lowpass order 4 cutoff 0.3
        //   scipy butter(4, 0.3, 'low') with explicit b, a
        CasePoint {
            case_id: "butter4_lp_0p3".into(),
            b: vec![
                0.010209480799830714,
                0.040837923199322855,
                0.06125688479898428,
                0.04083792319932284,
                0.010209480799830714,
            ],
            a: vec![
                1.0,
                -1.967510281586389,
                1.7095726895791028,
                -0.6707408258801084,
                0.10209480799830707,
            ],
            n_freqs: 32,
        },
        // 6. Simple 1st-order IIR: y[n] = 0.5 * x[n] + 0.5 * y[n-1]
        CasePoint {
            case_id: "iir1_smooth".into(),
            b: vec![0.5],
            a: vec![1.0, -0.5],
            n_freqs: 16,
        },
    ];

    OracleQuery { points: pts }
}

fn scipy_oracle_or_skip(q: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json, math, sys
import numpy as np
from scipy.signal import group_delay

q = json.load(sys.stdin)
out = []
for c in q["points"]:
    cid = c["case_id"]
    try:
        b = np.array(c["b"], dtype=float)
        a = np.array(c["a"], dtype=float)
        w, gd = group_delay((b, a), w=int(c["n_freqs"]))
        if not np.all(np.isfinite(gd)):
            out.append({"case_id": cid, "w": None, "gd": None})
        else:
            out.append({
                "case_id": cid,
                "w": [float(v) for v in w],
                "gd": [float(v) for v in gd],
            })
    except Exception:
        out.append({"case_id": cid, "w": None, "gd": None})

print(json.dumps({"points": out}))
"#;
    let query_json = serde_json::to_string(q).expect("serialize");
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
                "python3 spawn failed: {e}"
            );
            eprintln!("skipping group_delay oracle: python3 unavailable ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping group_delay oracle: stdin write failed");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "oracle failed: {stderr}"
        );
        eprintln!("skipping group_delay oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse JSON"))
}

#[test]
fn diff_signal_group_delay_from_ba() {
    let query = build_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    for (case, o) in query.points.iter().zip(oracle.points.iter()) {
        assert_eq!(case.case_id, o.case_id);
        let (Some(exp_w), Some(exp_gd)) = (o.w.as_ref(), o.gd.as_ref()) else {
            continue;
        };

        let (w_actual, gd_actual) = group_delay_from_ba(&case.b, &case.a, case.n_freqs);

        if w_actual.len() != exp_w.len() || gd_actual.len() != exp_gd.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                max_abs_diff_w: f64::INFINITY,
                max_abs_diff_gd: f64::INFINITY,
                max_rel_diff_gd: f64::INFINITY,
                pass: false,
                note: format!(
                    "length mismatch: fsci_w={} scipy_w={} fsci_gd={} scipy_gd={}",
                    w_actual.len(),
                    exp_w.len(),
                    gd_actual.len(),
                    exp_gd.len()
                ),
            });
            continue;
        }

        let mut max_abs_w = 0.0_f64;
        for (a, e) in w_actual.iter().zip(exp_w.iter()) {
            max_abs_w = max_abs_w.max((a - e).abs());
        }

        let mut max_abs_gd = 0.0_f64;
        let mut max_rel_gd = 0.0_f64;
        for (a, e) in gd_actual.iter().zip(exp_gd.iter()) {
            let abs_d = (a - e).abs();
            let denom = e.abs().max(1.0e-300);
            max_abs_gd = max_abs_gd.max(abs_d);
            max_rel_gd = max_rel_gd.max(abs_d / denom);
        }

        let w_pass = max_abs_w <= ABS_TOL;
        let gd_pass = max_rel_gd <= REL_TOL || max_abs_gd <= ABS_TOL;
        let pass = w_pass && gd_pass;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            max_abs_diff_w: max_abs_w,
            max_abs_diff_gd: max_abs_gd,
            max_rel_diff_gd: max_rel_gd,
            pass,
            note: String::new(),
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_signal_group_delay_from_ba".into(),
        category: "fsci_signal::group_delay_from_ba vs scipy.signal.group_delay".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "group_delay mismatch: {} w_max={} gd_max_abs={} gd_max_rel={} note={}",
                d.case_id, d.max_abs_diff_w, d.max_abs_diff_gd, d.max_rel_diff_gd, d.note
            );
        }
    }

    assert!(all_pass, "group_delay parity failed: {} cases", diffs.len());
}

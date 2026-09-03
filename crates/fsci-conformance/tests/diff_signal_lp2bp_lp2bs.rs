#![forbid(unsafe_code)]
//! Live scipy parity for fsci_signal::{lp2bp, lp2bs}.
//!
//! Resolves [frankenscipy-462g0]. These transform a lowpass filter
//! prototype to a bandpass / bandstop filter in transfer-function
//! (b, a) form. Both substitute s into the analog filter equation
//! and produce real-coefficient polynomials.
//!
//! Note: an earlier defect [frankenscipy-8fw59] reported that the
//! ZPK-form transforms (lp2hp_zpk/lp2bp_zpk/lp2bs_zpk) diverge from
//! scipy on non-trivial inputs. The (b, a)-form transforms tested
//! here are a separate code path that does NOT share that defect.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{lp2bp, lp2bs};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct CasePoint {
    case_id: String,
    func: String,
    b: Vec<f64>,
    a: Vec<f64>,
    wo: f64,
    bw: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<CasePoint>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct OraclePoint {
    case_id: String,
    b: Option<Vec<f64>>,
    a: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<OraclePoint>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    func: String,
    max_abs_diff_b: f64,
    max_abs_diff_a: f64,
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
    fs::create_dir_all(output_dir()).expect("create lp2bp diff dir");
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
    let mut pts = Vec::new();

    // First-order lowpass prototype: H(s) = 1 / (s + 1)
    pts.push(CasePoint {
        case_id: "lp2bp_first_order_wo2_bw0p5".into(),
        func: "lp2bp".into(),
        b: vec![1.0],
        a: vec![1.0, 1.0],
        wo: 2.0,
        bw: 0.5,
    });
    pts.push(CasePoint {
        case_id: "lp2bs_first_order_wo2_bw0p5".into(),
        func: "lp2bs".into(),
        b: vec![1.0],
        a: vec![1.0, 1.0],
        wo: 2.0,
        bw: 0.5,
    });

    // Second-order Butterworth-like: H(s) = 1 / (s² + sqrt(2)s + 1)
    let sqrt2 = 2.0_f64.sqrt();
    pts.push(CasePoint {
        case_id: "lp2bp_second_order_wo1_bw0p3".into(),
        func: "lp2bp".into(),
        b: vec![1.0],
        a: vec![1.0, sqrt2, 1.0],
        wo: 1.0,
        bw: 0.3,
    });
    pts.push(CasePoint {
        case_id: "lp2bs_second_order_wo1_bw0p3".into(),
        func: "lp2bs".into(),
        b: vec![1.0],
        a: vec![1.0, sqrt2, 1.0],
        wo: 1.0,
        bw: 0.3,
    });

    // Higher wo, larger bw
    pts.push(CasePoint {
        case_id: "lp2bp_first_order_wo5_bw2".into(),
        func: "lp2bp".into(),
        b: vec![1.0],
        a: vec![1.0, 1.0],
        wo: 5.0,
        bw: 2.0,
    });
    pts.push(CasePoint {
        case_id: "lp2bs_first_order_wo5_bw2".into(),
        func: "lp2bs".into(),
        b: vec![1.0],
        a: vec![1.0, 1.0],
        wo: 5.0,
        bw: 2.0,
    });

    OracleQuery { points: pts }
}

fn scipy_oracle_or_skip(q: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json, math, sys
import numpy as np
from scipy.signal import lp2bp, lp2bs

q = json.load(sys.stdin)
out = []
for c in q["points"]:
    cid = c["case_id"]
    try:
        b = np.array(c["b"], dtype=float)
        a = np.array(c["a"], dtype=float)
        if c["func"] == "lp2bp":
            bp, ap = lp2bp(b, a, wo=c["wo"], bw=c["bw"])
        elif c["func"] == "lp2bs":
            bp, ap = lp2bs(b, a, wo=c["wo"], bw=c["bw"])
        else:
            bp, ap = None, None
        if bp is None or not np.all(np.isfinite(bp)) or not np.all(np.isfinite(ap)):
            out.append({"case_id": cid, "b": None, "a": None})
        else:
            out.append({
                "case_id": cid,
                "b": [float(v) for v in bp],
                "a": [float(v) for v in ap],
            })
    except Exception:
        out.append({"case_id": cid, "b": None, "a": None})

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
            eprintln!("skipping lp2bp oracle: python3 unavailable ({e})");
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
            eprintln!("skipping lp2bp oracle: stdin write failed");
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
        eprintln!("skipping lp2bp oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse JSON"))
}

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_signal_lp2bp_lp2bs() {
    let query = build_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    for (c, o) in query.points.iter().zip(oracle.points.iter()) {
        assert_eq!(c.case_id, o.case_id);
        let (Some(exp_b), Some(exp_a)) = (o.b.as_ref(), o.a.as_ref()) else {
            continue;
        };

        let result = match c.func.as_str() {
            "lp2bp" => lp2bp(&c.b, &c.a, c.wo, c.bw),
            "lp2bs" => lp2bs(&c.b, &c.a, c.wo, c.bw),
            other => panic!("unknown func {other}"),
        };
        let (ba, aa) = match result {
            Ok(r) => r,
            Err(e) => {
                diffs.push(CaseDiff {
                    case_id: c.case_id.clone(),
                    func: c.func.clone(),
                    max_abs_diff_b: f64::INFINITY,
                    max_abs_diff_a: f64::INFINITY,
                    pass: false,
                    note: format!("error: {e:?}"),
                });
                continue;
            }
        };
        if ba.len() != exp_b.len() || aa.len() != exp_a.len() {
            diffs.push(CaseDiff {
                case_id: c.case_id.clone(),
                func: c.func.clone(),
                max_abs_diff_b: f64::INFINITY,
                max_abs_diff_a: f64::INFINITY,
                pass: false,
                note: format!(
                    "length mismatch: fsci b={} scipy b={} fsci a={} scipy a={}",
                    ba.len(),
                    exp_b.len(),
                    aa.len(),
                    exp_a.len()
                ),
            });
            continue;
        }
        let mab = max_abs(&ba, exp_b);
        let maa = max_abs(&aa, exp_a);
        let pass = mab <= ABS_TOL && maa <= ABS_TOL;
        diffs.push(CaseDiff {
            case_id: c.case_id.clone(),
            func: c.func.clone(),
            max_abs_diff_b: mab,
            max_abs_diff_a: maa,
            pass,
            note: String::new(),
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_signal_lp2bp_lp2bs".into(),
        category: "fsci_signal::{lp2bp, lp2bs} vs scipy.signal".into(),
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
                "lp2bp/lp2bs mismatch: {} ({}) b_max={} a_max={} note={}",
                d.case_id, d.func, d.max_abs_diff_b, d.max_abs_diff_a, d.note
            );
        }
    }

    assert!(all_pass, "lp2bp/lp2bs parity failed: {} cases", diffs.len());
}

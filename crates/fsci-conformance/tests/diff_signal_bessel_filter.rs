#![forbid(unsafe_code)]
//! Live scipy parity for fsci_signal::bessel filter design.
//!
//! Resolves [frankenscipy-mql8z]. Bessel filter has maximally-linear
//! phase response. fsci's design starts from the reverse Bessel
//! polynomial poles, scales by the Bessel phase normalization, and
//! digitizes through bilinear transform — same algorithm scipy uses.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{FilterType, bessel};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct CasePoint {
    case_id: String,
    order: usize,
    wn: Vec<f64>,
    /// "lowpass" | "highpass" | "bandpass" | "bandstop"
    btype: String,
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
    fs::create_dir_all(output_dir()).expect("create bessel diff dir");
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

fn ftype_from(s: &str) -> FilterType {
    match s {
        "lowpass" => FilterType::Lowpass,
        "highpass" => FilterType::Highpass,
        _ => panic!("unsupported btype {s}"),
    }
}

fn build_query() -> OracleQuery {
    let mut pts = Vec::new();
    for &order in &[2_usize, 3, 4] {
        for &wn in &[0.1_f64, 0.3, 0.5, 0.7] {
            pts.push(CasePoint {
                case_id: format!("bessel_low_order{order}_wn{wn}"),
                order,
                wn: vec![wn],
                btype: "lowpass".into(),
            });
            pts.push(CasePoint {
                case_id: format!("bessel_high_order{order}_wn{wn}"),
                order,
                wn: vec![wn],
                btype: "highpass".into(),
            });
        }
    }
    OracleQuery { points: pts }
}

fn scipy_oracle_or_skip(q: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json, math, sys
import numpy as np
from scipy.signal import bessel

q = json.load(sys.stdin)
out = []
for c in q["points"]:
    cid = c["case_id"]
    try:
        # scipy.signal.bessel takes Wn as scalar (for low/high) or list
        wn = c["wn"][0] if len(c["wn"]) == 1 else c["wn"]
        b, a = bessel(int(c["order"]), wn, c["btype"])
        if not np.all(np.isfinite(b)) or not np.all(np.isfinite(a)):
            out.append({"case_id": cid, "b": None, "a": None})
        else:
            out.append({
                "case_id": cid,
                "b": [float(v) for v in b],
                "a": [float(v) for v in a],
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
            eprintln!("skipping bessel oracle: python3 unavailable ({e})");
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
            eprintln!("skipping bessel oracle: stdin write failed");
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
        eprintln!("skipping bessel oracle: scipy not available\n{stderr}");
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
fn diff_signal_bessel_filter() {
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

        let result = bessel(c.order, &c.wn, ftype_from(&c.btype));
        let ba = match result {
            Ok(r) => r,
            Err(e) => {
                diffs.push(CaseDiff {
                    case_id: c.case_id.clone(),
                    max_abs_diff_b: f64::INFINITY,
                    max_abs_diff_a: f64::INFINITY,
                    pass: false,
                    note: format!("error: {e:?}"),
                });
                continue;
            }
        };
        if ba.b.len() != exp_b.len() || ba.a.len() != exp_a.len() {
            diffs.push(CaseDiff {
                case_id: c.case_id.clone(),
                max_abs_diff_b: f64::INFINITY,
                max_abs_diff_a: f64::INFINITY,
                pass: false,
                note: format!(
                    "length mismatch: fsci b={} a={} scipy b={} a={}",
                    ba.b.len(),
                    ba.a.len(),
                    exp_b.len(),
                    exp_a.len()
                ),
            });
            continue;
        }
        let mab = max_abs(&ba.b, exp_b);
        let maa = max_abs(&ba.a, exp_a);
        let pass = mab <= ABS_TOL && maa <= ABS_TOL;
        diffs.push(CaseDiff {
            case_id: c.case_id.clone(),
            max_abs_diff_b: mab,
            max_abs_diff_a: maa,
            pass,
            note: String::new(),
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_signal_bessel_filter".into(),
        category: "fsci_signal::bessel vs scipy.signal.bessel".into(),
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
                "bessel mismatch: {} b_max={} a_max={} note={}",
                d.case_id, d.max_abs_diff_b, d.max_abs_diff_a, d.note
            );
        }
    }

    assert!(all_pass, "bessel parity failed: {} cases", diffs.len());
}

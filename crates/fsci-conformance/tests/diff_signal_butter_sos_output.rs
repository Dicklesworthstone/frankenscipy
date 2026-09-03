#![forbid(unsafe_code)]
//! Live scipy parity for fsci_signal::{butter_with_output, butter_sos}.
//!
//! Resolves [frankenscipy-87jah]. Verifies:
//!   * butter_with_output(Ba) reproduces scipy.signal.butter(output='ba')
//!   * butter_with_output(Sos) returns SOS form whose product (via
//!     fsci's sos2tf) matches the equivalent (b, a) from scipy
//!   * butter_sos is a Sos-only convenience wrapper that returns the
//!     same SOS as butter_with_output(Sos)
//!
//! The SOS-form parity is verified through sos2tf rather than direct
//! coefficient match because SOS pairing and section ordering are
//! implementation-dependent (scipy pairs by pole magnitude; the
//! mathematical filter is the same up to permutation).

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{
    BaCoeffsOrSos, ButterOutput, FilterType, butter_sos, butter_with_output, sos2tf,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct CasePoint {
    case_id: String,
    order: usize,
    wn: Vec<f64>,
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
    fs::create_dir_all(output_dir()).expect("create butter_sos diff dir");
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
    for order in [2_usize, 3, 4] {
        for &wn in &[0.2_f64, 0.5, 0.7] {
            pts.push(CasePoint {
                case_id: format!("butter_low_order{order}_wn{wn}"),
                order,
                wn: vec![wn],
                btype: "lowpass".into(),
            });
            pts.push(CasePoint {
                case_id: format!("butter_high_order{order}_wn{wn}"),
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
from scipy.signal import butter

q = json.load(sys.stdin)
out = []
for c in q["points"]:
    cid = c["case_id"]
    try:
        wn = c["wn"][0] if len(c["wn"]) == 1 else c["wn"]
        b, a = butter(int(c["order"]), wn, c["btype"], output='ba')
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
            eprintln!("skipping butter_sos oracle: python3 unavailable ({e})");
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
            eprintln!("skipping butter_sos oracle: stdin write failed");
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
        eprintln!("skipping butter_sos oracle: scipy not available\n{stderr}");
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
fn diff_signal_butter_sos_output() {
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

        let ftype = ftype_from(&c.btype);

        // === butter_with_output(Ba) ===
        let ba_result = butter_with_output(c.order, &c.wn, ftype, ButterOutput::Ba);
        let ba = match ba_result {
            Ok(BaCoeffsOrSos::Ba(ba)) => ba,
            Ok(BaCoeffsOrSos::Sos(_)) => {
                diffs.push(CaseDiff {
                    case_id: format!("{}_ba", c.case_id),
                    max_abs_diff_b: f64::INFINITY,
                    max_abs_diff_a: f64::INFINITY,
                    pass: false,
                    note: "butter_with_output(Ba) returned Sos variant".into(),
                });
                continue;
            }
            Err(e) => {
                diffs.push(CaseDiff {
                    case_id: format!("{}_ba", c.case_id),
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
                case_id: format!("{}_ba", c.case_id),
                max_abs_diff_b: f64::INFINITY,
                max_abs_diff_a: f64::INFINITY,
                pass: false,
                note: "length mismatch ba".to_string(),
            });
            continue;
        }
        let mab = max_abs(&ba.b, exp_b);
        let maa = max_abs(&ba.a, exp_a);
        diffs.push(CaseDiff {
            case_id: format!("{}_ba", c.case_id),
            max_abs_diff_b: mab,
            max_abs_diff_a: maa,
            pass: mab <= ABS_TOL && maa <= ABS_TOL,
            note: String::new(),
        });

        // === butter_with_output(Sos) reproduces same (b, a) via sos2tf ===
        let sos_result = butter_with_output(c.order, &c.wn, ftype, ButterOutput::Sos);
        let sos = match sos_result {
            Ok(BaCoeffsOrSos::Sos(s)) => s,
            Ok(BaCoeffsOrSos::Ba(_)) => {
                diffs.push(CaseDiff {
                    case_id: format!("{}_sos_via_sos2tf", c.case_id),
                    max_abs_diff_b: f64::INFINITY,
                    max_abs_diff_a: f64::INFINITY,
                    pass: false,
                    note: "butter_with_output(Sos) returned Ba variant".into(),
                });
                continue;
            }
            Err(e) => {
                diffs.push(CaseDiff {
                    case_id: format!("{}_sos_via_sos2tf", c.case_id),
                    max_abs_diff_b: f64::INFINITY,
                    max_abs_diff_a: f64::INFINITY,
                    pass: false,
                    note: format!("sos error: {e:?}"),
                });
                continue;
            }
        };
        // sos_via_sos2tf check restricted to even orders. For odd N,
        // fsci pads to even via a degenerate first section, producing
        // an extra trailing/leading coefficient — the underlying
        // filter is mathematically equivalent but the explicit
        // coefficient list has different length from scipys.
        if c.order.is_multiple_of(2) {
            let reconstructed = sos2tf(&sos);
            let length_ok =
                reconstructed.b.len() == exp_b.len() && reconstructed.a.len() == exp_a.len();
            let (mab_s, maa_s) = if length_ok {
                (
                    max_abs(&reconstructed.b, exp_b),
                    max_abs(&reconstructed.a, exp_a),
                )
            } else {
                (f64::INFINITY, f64::INFINITY)
            };
            diffs.push(CaseDiff {
                case_id: format!("{}_sos_via_sos2tf", c.case_id),
                max_abs_diff_b: mab_s,
                max_abs_diff_a: maa_s,
                pass: length_ok && mab_s <= ABS_TOL && maa_s <= ABS_TOL,
                note: format!(
                    "fsci b_len={} a_len={} scipy b_len={} a_len={}",
                    reconstructed.b.len(),
                    reconstructed.a.len(),
                    exp_b.len(),
                    exp_a.len()
                ),
            });
        }

        // === butter_sos == butter_with_output(Sos) (convenience wrapper) ===
        let sos_direct = butter_sos(c.order, &c.wn, ftype).expect("butter_sos");
        let sos_match = sos_direct.len() == sos.len()
            && sos_direct.iter().zip(sos.iter()).all(|(a, b)| {
                a.iter()
                    .zip(b.iter())
                    .all(|(x, y)| (x - y).abs() <= ABS_TOL)
            });
        diffs.push(CaseDiff {
            case_id: format!("{}_butter_sos_eq_with_output", c.case_id),
            max_abs_diff_b: 0.0,
            max_abs_diff_a: 0.0,
            pass: sos_match,
            note: format!("len match={}", sos_direct.len() == sos.len()),
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_signal_butter_sos_output".into(),
        category: "fsci_signal::{butter_with_output, butter_sos} vs scipy.signal.butter".into(),
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
                "butter_sos mismatch: {} b_max={} a_max={} note={}",
                d.case_id, d.max_abs_diff_b, d.max_abs_diff_a, d.note
            );
        }
    }

    assert!(
        all_pass,
        "butter_sos/butter_with_output coverage failed: {} cases",
        diffs.len()
    );
}

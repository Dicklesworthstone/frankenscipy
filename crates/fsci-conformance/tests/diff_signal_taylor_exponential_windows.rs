#![forbid(unsafe_code)]
//! Live scipy parity for fsci_signal Taylor and exponential windows.
//!
//! Resolves [frankenscipy-s5vvg]. Compares:
//!   * `taylor(n, nbar, sll, norm, sym)` vs scipy.signal.windows.taylor
//!   * `exponential(n, center, tau, sym)` vs scipy.signal.windows.exponential
//!
//! Sweeps n ∈ {8, 16, 32, 64}, nbar ∈ {3, 4, 5}, sll ∈ {-30, -40, -50},
//! with both sym=True and sym=False. For exponential: n ∈ {8, 16, 32},
//! tau ∈ {2.0, 5.0, 8.0}, and several center configurations.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{exponential, taylor};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REL_TOL: f64 = 1.0e-10;
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct CasePoint {
    case_id: String,
    /// "taylor" or "exponential"
    func: String,
    n: usize,
    /// Taylor: nbar
    nbar: usize,
    /// Taylor: sll (negative dB)
    sll: f64,
    norm: bool,
    sym: bool,
    /// Exponential: tau
    tau: f64,
    /// Exponential: optional center (None encoded as NaN)
    center: f64,
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
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<OraclePoint>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    func: String,
    max_abs_diff: f64,
    max_rel_diff: f64,
    n_eval: usize,
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
    fs::create_dir_all(output_dir()).expect("create taylor diff dir");
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

    // taylor sweep
    for &n in &[8_usize, 16, 32, 64] {
        for &nbar in &[3_usize, 4, 5] {
            for &sll in &[-30.0_f64, -40.0, -50.0] {
                for &sym in &[true, false] {
                    pts.push(CasePoint {
                        case_id: format!("taylor_n{n}_nbar{nbar}_sll{sll}_sym{sym}"),
                        func: "taylor".into(),
                        n,
                        nbar,
                        sll,
                        norm: true,
                        sym,
                        tau: 0.0,
                        center: f64::NAN,
                    });
                }
            }
        }
    }

    // exponential sweep — sym=true (no explicit center) and sym=false with various centers
    for &n in &[8_usize, 16, 32] {
        for &tau in &[2.0_f64, 5.0, 8.0] {
            // sym = True path (center = None)
            pts.push(CasePoint {
                case_id: format!("exp_sym_n{n}_tau{tau}"),
                func: "exponential".into(),
                n,
                nbar: 0,
                sll: 0.0,
                norm: false,
                sym: true,
                tau,
                center: f64::NAN,
            });
            // sym = False with default center (middle)
            pts.push(CasePoint {
                case_id: format!("exp_asym_default_n{n}_tau{tau}"),
                func: "exponential".into(),
                n,
                nbar: 0,
                sll: 0.0,
                norm: false,
                sym: false,
                tau,
                center: f64::NAN, // None
            });
            // sym = False with explicit center at sample 0
            pts.push(CasePoint {
                case_id: format!("exp_asym_c0_n{n}_tau{tau}"),
                func: "exponential".into(),
                n,
                nbar: 0,
                sll: 0.0,
                norm: false,
                sym: false,
                tau,
                center: 0.0,
            });
        }
    }

    OracleQuery { points: pts }
}

fn scipy_oracle_or_skip(q: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json, math, sys
import numpy as np
from scipy.signal import windows

q = json.load(sys.stdin)
out = []
for c in q["points"]:
    cid = c["case_id"]
    try:
        if c["func"] == "taylor":
            w = windows.taylor(int(c["n"]), nbar=int(c["nbar"]), sll=float(c["sll"]),
                               norm=bool(c["norm"]), sym=bool(c["sym"]))
        elif c["func"] == "exponential":
            center = None if (c["center"] != c["center"]) else float(c["center"])  # NaN check
            w = windows.exponential(int(c["n"]), center=center, tau=float(c["tau"]),
                                    sym=bool(c["sym"]))
        else:
            w = None
        if w is None or not np.all(np.isfinite(w)):
            out.append({"case_id": cid, "w": None})
        else:
            out.append({"case_id": cid, "w": [float(v) for v in w]})
    except Exception:
        out.append({"case_id": cid, "w": None})

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
            eprintln!("skipping taylor/exp oracle: python3 unavailable ({e})");
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
            eprintln!("skipping taylor/exp oracle: stdin write failed");
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
        eprintln!("skipping taylor/exp oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse JSON"))
}

#[test]
fn diff_signal_taylor_exponential_windows() {
    let query = build_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    for (c, o) in query.points.iter().zip(oracle.points.iter()) {
        assert_eq!(c.case_id, o.case_id);
        let Some(expected) = o.w.as_ref() else {
            continue;
        };

        let actual = match c.func.as_str() {
            "taylor" => taylor(c.n, c.nbar, c.sll, c.norm, c.sym),
            "exponential" => {
                let center = if c.center.is_nan() {
                    None
                } else {
                    Some(c.center)
                };
                match exponential(c.n, center, c.tau, c.sym) {
                    Ok(w) => w,
                    Err(e) => {
                        diffs.push(CaseDiff {
                            case_id: c.case_id.clone(),
                            func: c.func.clone(),
                            max_abs_diff: f64::INFINITY,
                            max_rel_diff: f64::INFINITY,
                            n_eval: 0,
                            pass: false,
                            note: format!("exponential error: {e:?}"),
                        });
                        continue;
                    }
                }
            }
            other => panic!("unknown func {other}"),
        };

        if actual.len() != expected.len() {
            diffs.push(CaseDiff {
                case_id: c.case_id.clone(),
                func: c.func.clone(),
                max_abs_diff: f64::INFINITY,
                max_rel_diff: f64::INFINITY,
                n_eval: actual.len(),
                pass: false,
                note: format!(
                    "length mismatch: fsci={} scipy={}",
                    actual.len(),
                    expected.len()
                ),
            });
            continue;
        }

        let mut max_abs = 0.0_f64;
        let mut max_rel = 0.0_f64;
        for (a, e) in actual.iter().zip(expected.iter()) {
            let abs_d = (a - e).abs();
            let denom = e.abs().max(1.0e-300);
            max_abs = max_abs.max(abs_d);
            max_rel = max_rel.max(abs_d / denom);
        }
        let pass = max_rel <= REL_TOL || max_abs <= ABS_TOL;
        diffs.push(CaseDiff {
            case_id: c.case_id.clone(),
            func: c.func.clone(),
            max_abs_diff: max_abs,
            max_rel_diff: max_rel,
            n_eval: actual.len(),
            pass,
            note: String::new(),
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_signal_taylor_exponential_windows".into(),
        category: "fsci_signal::{taylor, exponential} vs scipy.signal.windows".into(),
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
                "taylor/exp mismatch: {} ({}) max_rel={} max_abs={} note={}",
                d.case_id, d.func, d.max_rel_diff, d.max_abs_diff, d.note
            );
        }
    }

    assert!(
        all_pass,
        "taylor/exponential window parity failed: {} cases",
        diffs.len()
    );
}

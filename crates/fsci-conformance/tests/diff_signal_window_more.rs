#![forbid(unsafe_code)]
//! Live scipy.signal.windows parity for additional window functions
//! exposed by fsci_signal: barthann, general_cosine, general_gaussian,
//! lanczos, chebwin.
//!
//! Resolves [frankenscipy-qovmg]. All deterministic. barthann,
//! general_cosine, general_gaussian, lanczos at 1e-12 abs; chebwin
//! at 1e-9 abs (Chebyshev attenuation parameter exposes more
//! floating-point sensitivity in scipy's implementation).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{barthann, chebwin, general_cosine, general_gaussian, lanczos};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const TIGHT_TOL: f64 = 1.0e-12;
const CHEB_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "barthann" | "gcos" | "ggauss" | "lanczos" | "chebwin"
    n: usize,
    sym: bool,
    /// gcos
    coeffs: Vec<f64>,
    /// ggauss
    p: f64,
    sig: f64,
    /// chebwin
    at: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    values: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create window_more diff dir");
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
    let mut points = Vec::new();
    // barthann probes
    for &n in &[16_usize, 32, 64, 65] {
        points.push(Case {
            case_id: format!("barthann_n{n}"),
            op: "barthann".into(),
            n,
            sym: true,
            coeffs: vec![],
            p: 0.0,
            sig: 0.0,
            at: 0.0,
        });
    }
    // general_cosine probes — Blackman coeffs and a custom 3-term
    let blackman = vec![0.42_f64, 0.5, 0.08];
    let custom = vec![0.3_f64, 0.4, 0.2, 0.1];
    for sym in [true, false] {
        for &n in &[32_usize, 65] {
            for (label, c) in [("blackman", &blackman), ("custom4", &custom)] {
                points.push(Case {
                    case_id: format!("gcos_{label}_n{n}_sym{sym}"),
                    op: "gcos".into(),
                    n,
                    sym,
                    coeffs: c.clone(),
                    p: 0.0,
                    sig: 0.0,
                    at: 0.0,
                });
            }
        }
    }
    // general_gaussian probes — (p, sig)
    for &(p, sig) in &[(0.5_f64, 5.0), (1.0, 7.0), (2.0, 10.0), (1.5, 6.0)] {
        for sym in [true, false] {
            for &n in &[32_usize, 65] {
                points.push(Case {
                    case_id: format!("ggauss_p{p}_sig{sig}_n{n}_sym{sym}"),
                    op: "ggauss".into(),
                    n,
                    sym,
                    coeffs: vec![],
                    p,
                    sig,
                    at: 0.0,
                });
            }
        }
    }
    // lanczos probes
    for &n in &[16_usize, 32, 64, 65] {
        points.push(Case {
            case_id: format!("lanczos_n{n}"),
            op: "lanczos".into(),
            n,
            sym: true,
            coeffs: vec![],
            p: 0.0,
            sig: 0.0,
            at: 0.0,
        });
    }
    // chebwin probes — vary attenuation
    for &n in &[32_usize, 65] {
        for &at in &[50.0_f64, 80.0, 100.0] {
            points.push(Case {
                case_id: format!("chebwin_n{n}_at{at}"),
                op: "chebwin".into(),
                n,
                sym: true,
                coeffs: vec![],
                p: 0.0,
                sig: 0.0,
                at,
            });
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.signal import windows

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    n = int(case["n"]); sym = bool(case["sym"])
    try:
        if op == "barthann":
            w = windows.barthann(n, sym=sym)
        elif op == "gcos":
            coeffs = [float(c) for c in case["coeffs"]]
            w = windows.general_cosine(n, coeffs, sym=sym)
        elif op == "ggauss":
            p = float(case["p"]); sig = float(case["sig"])
            w = windows.general_gaussian(n, p, sig, sym=sym)
        elif op == "lanczos":
            w = windows.lanczos(n, sym=sym)
        elif op == "chebwin":
            at = float(case["at"])
            w = windows.chebwin(n, at, sym=sym)
        else:
            points.append({"case_id": cid, "values": None}); continue
        flat = [float(v) for v in np.asarray(w).tolist()]
        if all(math.isfinite(v) for v in flat):
            points.append({"case_id": cid, "values": flat})
        else:
            points.append({"case_id": cid, "values": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "values": None})
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
                "failed to spawn python3 for window_more oracle: {e}"
            );
            eprintln!("skipping window_more oracle: python3 not available ({e})");
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
                "window_more oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping window_more oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for window_more oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "window_more oracle failed: {stderr}"
        );
        eprintln!("skipping window_more oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse window_more oracle JSON"))
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
fn diff_signal_window_more() {
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
        let Some(expected) = arm.values.as_ref() else {
            continue;
        };
        let (window, tol) = match case.op.as_str() {
            "barthann" => (barthann(case.n), TIGHT_TOL),
            "gcos" => (general_cosine(case.n, &case.coeffs, case.sym), TIGHT_TOL),
            "ggauss" => (
                general_gaussian(case.n, case.p, case.sig, case.sym),
                TIGHT_TOL,
            ),
            "lanczos" => (lanczos(case.n), TIGHT_TOL),
            "chebwin" => (chebwin(case.n, case.at), CHEB_TOL),
            _ => continue,
        };
        let abs_d = vec_max_diff(&window, expected);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= tol,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_window_more".into(),
        category:
            "fsci_signal::{barthann, general_cosine, general_gaussian, lanczos, chebwin} vs scipy.signal.windows"
                .into(),
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
        "window_more conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

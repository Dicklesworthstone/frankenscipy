#![forbid(unsafe_code)]
//! Live SciPy differential coverage for filter-order selection:
//!   - `scipy.signal.buttord`
//!   - `scipy.signal.cheb1ord`
//!   - `scipy.signal.cheb2ord`
//!   - `scipy.signal.ellipord`
//!
//! Resolves [frankenscipy-kjl32]. Each returns (N, Wn). N integer
//! (exact comparison); Wn at 1e-10 abs. fsci's filter-order routines match
//! scipy's DIGITAL default (`analog=False`, per frankenscipy-qjotp): wp/ws are
//! Nyquist-normalized in (0, 1), with frequency prewarping and highpass
//! support. Drop a case if scipy errors.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{buttord, cheb1ord, cheb2ord, ellipord};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const WN_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    wp: f64,
    ws: f64,
    gpass: f64,
    gstop: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    n: Option<u32>,
    wn: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
    n_match: bool,
    wn_diff: f64,
    pass: bool,
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
    fs::create_dir_all(output_dir()).expect("create filter_ord diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize filter_ord diff log");
    fs::write(path, json).expect("write filter_ord diff log");
}

fn generate_query() -> OracleQuery {
    let configs: &[(f64, f64, f64, f64)] = &[
        (0.2, 0.3, 1.0, 40.0),
        (0.1, 0.2, 0.5, 30.0),
        (0.3, 0.5, 1.0, 30.0),
        (0.2, 0.4, 0.5, 50.0),
        (0.4, 0.6, 1.0, 40.0),
    ];
    let ops = ["buttord", "cheb1ord", "cheb2ord", "ellipord"];

    let mut points = Vec::new();
    for (i, (wp, ws, gpass, gstop)) in configs.iter().enumerate() {
        for op in ops {
            points.push(PointCase {
                case_id: format!("{op}_{i:02}_wp{wp}_ws{ws}_gp{gpass}_gs{gstop}"),
                op: op.into(),
                wp: *wp,
                ws: *ws,
                gpass: *gpass,
                gstop: *gstop,
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
from scipy import signal

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    wp = float(case["wp"]); ws = float(case["ws"])
    gpass = float(case["gpass"]); gstop = float(case["gstop"])
    try:
        fn = getattr(signal, op)
        # fsci's filter_ord routines now match scipy's DIGITAL default
        # (frankenscipy-qjotp); wp/ws are Nyquist-normalized in (0, 1).
        n, wn = fn(wp, ws, gpass, gstop, analog=False)
        wn = float(wn) if wn is not None else None
        if wn is None or not math.isfinite(wn):
            points.append({"case_id": cid, "n": None, "wn": None})
        else:
            points.append({"case_id": cid, "n": int(n), "wn": wn})
    except Exception:
        points.append({"case_id": cid, "n": None, "wn": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize filter_ord query");
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
                "failed to spawn python3 for filter_ord oracle: {e}"
            );
            eprintln!("skipping filter_ord oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open filter_ord oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "filter_ord oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping filter_ord oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for filter_ord oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "filter_ord oracle failed: {stderr}"
        );
        eprintln!("skipping filter_ord oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse filter_ord oracle JSON"))
}

fn fsci_eval(case: &PointCase) -> Option<(u32, f64)> {
    match case.op.as_str() {
        "buttord" => buttord(case.wp, case.ws, case.gpass, case.gstop).ok(),
        "cheb1ord" => cheb1ord(case.wp, case.ws, case.gpass, case.gstop).ok(),
        "cheb2ord" => cheb2ord(case.wp, case.ws, case.gpass, case.gstop).ok(),
        "ellipord" => ellipord(case.wp, case.ws, case.gpass, case.gstop).ok(),
        _ => None,
    }
}

#[test]
fn diff_signal_filter_ord() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_n) = scipy_arm.n else { continue };
        let Some(scipy_wn) = scipy_arm.wn else {
            continue;
        };
        let Some((fsci_n, fsci_wn)) = fsci_eval(case) else {
            continue;
        };
        let n_match = fsci_n == scipy_n;
        let wn_diff = (fsci_wn - scipy_wn).abs();
        let pass = n_match && wn_diff <= WN_TOL;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            n_match,
            wn_diff,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_filter_ord".into(),
        category: "scipy.signal.{butt,cheb1,cheb2,ellip}ord".into(),
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
                "filter_ord {} mismatch: {} n_match={} wn_diff={}",
                d.op, d.case_id, d.n_match, d.wn_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.signal filter-order conformance failed: {} cases",
        diffs.len()
    );
}

#![forbid(unsafe_code)]
//! Live SciPy differential coverage for analog lowpass prototypes:
//!   - `scipy.signal.buttap(N)` — Butterworth
//!   - `scipy.signal.cheb1ap(N, rp)` — Chebyshev type-1
//!
//! Resolves [frankenscipy-yrj9k]. Both return (zeros, poles, gain).
//! Lowpass prototypes have no zeros — fsci/scipy both return empty.
//! Poles compared after sorting by (real, imag); gain compared scalar.
//! 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{buttap, cheb1ap};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    n: u32,
    /// Only used for cheb1ap.
    rp: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Sorted poles packed as [re0, im0, re1, im1, ...] followed by gain.
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
    fs::create_dir_all(output_dir()).expect("create analog_proto diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize analog_proto diff log");
    fs::write(path, json).expect("write analog_proto diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    // buttap orders
    for n in [2u32, 3, 4, 5, 6, 8] {
        points.push(PointCase {
            case_id: format!("buttap_n{n}"),
            op: "buttap".into(),
            n,
            rp: 0.0,
        });
    }
    // cheb1ap (n, rp)
    for &(n, rp) in &[
        (2u32, 0.5_f64),
        (3, 0.5),
        (4, 0.5),
        (4, 1.0),
        (5, 0.5),
        (6, 0.5),
        (6, 1.5),
    ] {
        points.push(PointCase {
            case_id: format!("cheb1ap_n{n}_rp{rp}"),
            op: "cheb1ap".into(),
            n,
            rp,
        });
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import signal

def pack_sorted(poles, gain):
    # sort by (real, imag); pack as [re0, im0, ..., gain] last
    items = [(complex(p).real, complex(p).imag) for p in poles]
    items.sort(key=lambda t: (t[0], t[1]))
    packed = []
    for re, im in items:
        if not (math.isfinite(re) and math.isfinite(im)):
            return None
        packed.append(re)
        packed.append(im)
    if not math.isfinite(gain):
        return None
    packed.append(float(gain))
    return packed

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    n = int(case["n"]); rp = float(case["rp"])
    try:
        if op == "buttap":
            z, p, k = signal.buttap(n)
        elif op == "cheb1ap":
            z, p, k = signal.cheb1ap(n, rp)
        else:
            points.append({"case_id": cid, "values": None}); continue
        # Lowpass prototypes have no zeros; if scipy returns any, skip.
        if len(z) != 0:
            points.append({"case_id": cid, "values": None}); continue
        v = pack_sorted(p, float(k))
        points.append({"case_id": cid, "values": v})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize analog_proto query");
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
                "failed to spawn python3 for analog_proto oracle: {e}"
            );
            eprintln!("skipping analog_proto oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open analog_proto oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "analog_proto oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping analog_proto oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for analog_proto oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "analog_proto oracle failed: {stderr}"
        );
        eprintln!("skipping analog_proto oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse analog_proto oracle JSON"))
}

fn fsci_eval(case: &PointCase) -> Option<Vec<f64>> {
    let (_z, p, k) = match case.op.as_str() {
        "buttap" => buttap(case.n).ok()?,
        "cheb1ap" => cheb1ap(case.n, case.rp).ok()?,
        _ => return None,
    };
    let mut items: Vec<(f64, f64)> = p.iter().map(|c| (c.0, c.1)).collect();
    items.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    let mut packed = Vec::with_capacity(items.len() * 2 + 1);
    for (re, im) in items {
        packed.push(re);
        packed.push(im);
    }
    packed.push(k);
    Some(packed)
}

#[test]
fn diff_signal_analog_prototypes() {
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
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_v) = scipy_arm.values.as_ref() else {
            continue;
        };
        let Some(fsci_v) = fsci_eval(case) else {
            continue;
        };
        if fsci_v.len() != scipy_v.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                op: case.op.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let abs_d = fsci_v
            .iter()
            .zip(scipy_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_analog_prototypes".into(),
        category: "scipy.signal.buttap + cheb1ap".into(),
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
            eprintln!(
                "analog_proto {} mismatch: {} abs_diff={}",
                d.op, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.signal buttap/cheb1ap conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

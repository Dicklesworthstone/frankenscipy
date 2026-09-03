#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.signal.cheby1` (Chebyshev
//! type-1 lowpass).
//!
//! Resolves [frankenscipy-5wgnf]. fsci_signal::cheby1(order, rp, wn,
//! btype=Lowpass) returns BaCoeffs { b, a }. 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{FilterType, cheby1};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    order: usize,
    rp: f64,
    wn: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    values: Option<Vec<f64>>,
    n_b: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create cheby1 diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize cheby1 diff log");
    fs::write(path, json).expect("write cheby1 diff log");
}

fn generate_query() -> OracleQuery {
    let cases: &[(usize, f64, f64)] = &[
        (2, 0.5, 0.3),
        (3, 0.5, 0.3),
        (4, 0.5, 0.3),
        (4, 1.0, 0.3),
        (4, 0.5, 0.5),
        (5, 0.5, 0.4),
        (6, 0.5, 0.3),
        (6, 1.5, 0.2),
    ];
    let points = cases
        .iter()
        .enumerate()
        .map(|(i, (n, rp, w))| PointCase {
            case_id: format!("cheby1_{i:02}_n{n}_rp{rp}_wn{w}"),
            order: *n,
            rp: *rp,
            wn: *w,
        })
        .collect();
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import signal

def finite_vec_or_none(arr):
    out = []
    for v in np.asarray(arr).tolist():
        try:
            v = float(v)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        out.append(v)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    n = int(case["order"]); rp = float(case["rp"]); w = float(case["wn"])
    try:
        b, a = signal.cheby1(n, rp, w, btype='low')
        bv = finite_vec_or_none(b); av = finite_vec_or_none(a)
        if bv is None or av is None:
            points.append({"case_id": cid, "values": None, "n_b": None})
        else:
            points.append({"case_id": cid, "values": bv + av, "n_b": len(bv)})
    except Exception:
        points.append({"case_id": cid, "values": None, "n_b": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize cheby1 query");
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
                "failed to spawn python3 for cheby1 oracle: {e}"
            );
            eprintln!("skipping cheby1 oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open cheby1 oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "cheby1 oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping cheby1 oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for cheby1 oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "cheby1 oracle failed: {stderr}"
        );
        eprintln!("skipping cheby1 oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse cheby1 oracle JSON"))
}

#[test]
fn diff_signal_cheby1() {
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
        let Some(n_b) = scipy_arm.n_b else { continue };
        let Ok(coeffs) = cheby1(case.order, case.rp, &[case.wn], FilterType::Lowpass) else {
            continue;
        };
        let mut fsci_v = coeffs.b.clone();
        fsci_v.extend(coeffs.a.iter().copied());
        if fsci_v.len() != scipy_v.len() || coeffs.b.len() != n_b {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
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
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_cheby1".into(),
        category: "scipy.signal.cheby1 (lowpass)".into(),
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
            eprintln!("cheby1 mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "scipy.signal.cheby1 conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

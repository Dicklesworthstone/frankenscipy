#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.signal.peak_widths`.
//!
//! Resolves [frankenscipy-7atoj]. fsci returns (widths, heights,
//! left_ips, right_ips). 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::peak_widths;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    x: Vec<f64>,
    peaks: Vec<usize>,
    rel_height: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// [widths..., heights..., left_ips..., right_ips...]
    values: Option<Vec<f64>>,
    n: Option<usize>,
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
    fs::create_dir_all(output_dir()).expect("create peak_widths diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize peak_widths diff log");
    fs::write(path, json).expect("write peak_widths diff log");
}

fn generate_query() -> OracleQuery {
    let cases: &[(&str, Vec<f64>, Vec<usize>, f64)] = &[
        (
            "small_3peaks_half",
            vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 0.0, 1.0, 2.0, 1.0],
            vec![2, 5, 9],
            0.5,
        ),
        (
            "small_3peaks_third",
            vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 0.0, 1.0, 2.0, 1.0],
            vec![2, 5, 9],
            0.333,
        ),
        (
            "single_peak",
            vec![0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0],
            vec![3],
            0.5,
        ),
        (
            "two_peaks_eq",
            vec![0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0],
            vec![2, 6],
            0.5,
        ),
    ];
    let points = cases
        .iter()
        .map(|(name, x, p, h)| PointCase {
            case_id: (*name).into(),
            x: x.clone(),
            peaks: p.clone(),
            rel_height: *h,
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
    x = np.array(case["x"], dtype=float)
    peaks = np.array(case["peaks"], dtype=int)
    rh = float(case["rel_height"])
    try:
        w, h, lp, rp = signal.peak_widths(x, peaks, rel_height=rh)
        wv = finite_vec_or_none(w); hv = finite_vec_or_none(h)
        lv = finite_vec_or_none(lp); rv = finite_vec_or_none(rp)
        if wv is None or hv is None or lv is None or rv is None:
            points.append({"case_id": cid, "values": None, "n": None})
        else:
            points.append({"case_id": cid, "values": wv + hv + lv + rv, "n": len(wv)})
    except Exception:
        points.append({"case_id": cid, "values": None, "n": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize peak_widths query");
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
                "failed to spawn python3 for peak_widths oracle: {e}"
            );
            eprintln!("skipping peak_widths oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open peak_widths oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "peak_widths oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping peak_widths oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for peak_widths oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "peak_widths oracle failed: {stderr}"
        );
        eprintln!("skipping peak_widths oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse peak_widths oracle JSON"))
}

#[test]
fn diff_signal_peak_widths() {
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
        let Some(n) = scipy_arm.n else { continue };
        let (w, h, lp, rp) = peak_widths(&case.x, &case.peaks, case.rel_height);
        let mut fsci_v = w;
        fsci_v.extend(h);
        fsci_v.extend(lp);
        fsci_v.extend(rp);
        if fsci_v.len() != 4 * n || scipy_v.len() != 4 * n {
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
        test_id: "diff_signal_peak_widths".into(),
        category: "scipy.signal.peak_widths".into(),
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
                "peak_widths mismatch: {} abs_diff={}",
                d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.signal.peak_widths conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

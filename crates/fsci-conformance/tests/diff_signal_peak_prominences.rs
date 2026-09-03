#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.signal.peak_prominences`.
//!
//! Resolves [frankenscipy-0hsu4]. fsci returns (prominences, lbases,
//! rbases). Bit-exact comparison on prominences (closed-form max-min)
//! and bases (integer indices).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::peak_prominences;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    x: Vec<f64>,
    peaks: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    prominences: Option<Vec<f64>>,
    left_bases: Option<Vec<usize>>,
    right_bases: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    prom_diff: f64,
    bases_match: bool,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_prom_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create peak_prom diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize peak_prom diff log");
    fs::write(path, json).expect("write peak_prom diff log");
}

fn generate_query() -> OracleQuery {
    let cases: &[(&str, Vec<f64>, Vec<usize>)] = &[
        (
            "small_3peaks",
            vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 0.0, 1.0, 2.0, 1.0],
            vec![2, 5, 9],
        ),
        (
            "single_peak",
            vec![0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0],
            vec![3],
        ),
        (
            "two_peaks_eq_height",
            vec![0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0],
            vec![2, 6],
        ),
        (
            "longer_signal",
            (0..20)
                .map(|i| ((i as f64) * 0.6).sin() + (i as f64) * 0.05)
                .collect(),
            vec![3, 13],
        ),
    ];
    let points = cases
        .iter()
        .map(|(name, x, p)| PointCase {
            case_id: (*name).into(),
            x: x.clone(),
            peaks: p.clone(),
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

def vec_or_none(arr):
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
    try:
        prom, lb, rb = signal.peak_prominences(x, peaks)
        points.append({
            "case_id": cid,
            "prominences": vec_or_none(prom),
            "left_bases": [int(v) for v in lb.tolist()],
            "right_bases": [int(v) for v in rb.tolist()],
        })
    except Exception:
        points.append({
            "case_id": cid,
            "prominences": None,
            "left_bases": None,
            "right_bases": None,
        })
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize peak_prom query");
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
                "failed to spawn python3 for peak_prom oracle: {e}"
            );
            eprintln!("skipping peak_prom oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open peak_prom oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "peak_prom oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping peak_prom oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for peak_prom oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "peak_prom oracle failed: {stderr}"
        );
        eprintln!("skipping peak_prom oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse peak_prom oracle JSON"))
}

#[test]
fn diff_signal_peak_prominences() {
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
    let mut max_prom: f64 = 0.0;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_prom) = scipy_arm.prominences.as_ref() else {
            continue;
        };
        let Some(scipy_lb) = scipy_arm.left_bases.as_ref() else {
            continue;
        };
        let Some(scipy_rb) = scipy_arm.right_bases.as_ref() else {
            continue;
        };
        let (fsci_prom, fsci_lb, fsci_rb) = peak_prominences(&case.x, &case.peaks);
        if fsci_prom.len() != scipy_prom.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                prom_diff: f64::INFINITY,
                bases_match: false,
                pass: false,
            });
            continue;
        }
        let prom_diff = fsci_prom
            .iter()
            .zip(scipy_prom.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let bases_match = fsci_lb.iter().zip(scipy_lb.iter()).all(|(a, b)| *a == *b)
            && fsci_rb.iter().zip(scipy_rb.iter()).all(|(a, b)| *a == *b);
        max_prom = max_prom.max(prom_diff);
        let pass = prom_diff <= ABS_TOL && bases_match;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            prom_diff,
            bases_match,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_peak_prominences".into(),
        category: "scipy.signal.peak_prominences".into(),
        case_count: diffs.len(),
        max_prom_diff: max_prom,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "peak_prom mismatch: {} prom_diff={} bases_match={}",
                d.case_id, d.prom_diff, d.bases_match
            );
        }
    }

    assert!(
        all_pass,
        "scipy.signal.peak_prominences conformance failed: {} cases, max prom_diff={}",
        diffs.len(),
        max_prom
    );
}

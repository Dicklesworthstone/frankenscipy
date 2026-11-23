#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.signal.sosfilt`.
//!
//! Resolves [frankenscipy-dy1pp]. fsci_signal::sosfilt applies a
//! cascade of biquad SOS sections. SosSection = [f64; 6] (b0, b1, b2,
//! a0, a1, a2) matching scipy. 1e-12 abs.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_signal::{SosFilter, SosSection, sosfilt, sosfilt_with_state};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    sos: Vec<[f64; 6]>,
    x: Vec<f64>,
    zi: Option<Vec<[f64; 2]>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    values: Option<Vec<f64>>,
    final_state: Option<Vec<f64>>,
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
    compared: BTreeMap<String, ArmCounts>,
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
    fs::create_dir_all(output_dir()).expect("create sosfilt diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize sosfilt diff log");
    fs::write(path, json).expect("write sosfilt diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = vec![
        PointCase {
            case_id: "1section_lp".into(),
            sos: vec![[0.0675, 0.135, 0.0675, 1.0, -1.143, 0.4128]],
            x: (0..16).map(|i| i as f64).collect(),
            zi: None,
        },
        PointCase {
            case_id: "fir_biquad".into(),
            sos: vec![[1.0, 0.5, 0.25, 1.0, 0.0, 0.0]],
            x: (0..12).map(|i| (i as f64 * 0.5).sin()).collect(),
            zi: None,
        },
        PointCase {
            case_id: "2section_cascade".into(),
            sos: vec![
                [0.5, 0.0, 0.0, 1.0, -0.5, 0.0],
                [1.0, 0.0, 0.0, 1.0, -0.3, 0.1],
            ],
            x: (0..20).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect(),
            zi: None,
        },
        PointCase {
            case_id: "3section_cascade".into(),
            sos: vec![
                [1.0, 2.0, 1.0, 1.0, -0.5, 0.0],
                [1.0, -1.0, 0.5, 1.0, -0.3, 0.1],
                [0.5, 0.5, 0.0, 1.0, 0.0, -0.2],
            ],
            x: (0..24).map(|i| (i as f64 * 0.3).cos()).collect(),
            zi: None,
        },
        PointCase {
            case_id: "6section_long_signal".into(),
            sos: vec![[0.0675, 0.135, 0.0675, 1.0, -1.143, 0.4128]; 6],
            x: (0..8193)
                .map(|i| (i as f64 * 0.03).sin() + (i as f64 * 0.71).cos())
                .collect(),
            zi: None,
        },
        PointCase {
            case_id: "single_sample_nonzero_state".into(),
            sos: vec![
                [0.5, 0.25, 0.125, 1.0, -0.4, 0.2],
                [1.0, -0.3, 0.2, 1.0, 0.25, -0.1],
            ],
            x: vec![2.0],
            zi: Some(vec![[0.2, -0.1], [-0.3, 0.4]]),
        },
    ];
    let initialized: Vec<PointCase> = points
        .iter()
        .filter(|case| case.zi.is_none())
        .map(|case| PointCase {
            case_id: format!("initial_{}", case.case_id),
            zi: Some(
                (0..case.sos.len())
                    .map(|i| [0.05 * (i + 1) as f64, -0.02 * i as f64])
                    .collect(),
            ),
            ..case.clone()
        })
        .collect();
    points.extend(initialized);
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
    sos = np.array(case["sos"], dtype=float)
    x = np.array(case["x"], dtype=float)
    try:
        zi = case["zi"]
        if zi is None:
            zi = np.zeros((len(sos), 2))
        y, zf = signal.sosfilt(sos, x, zi=np.asarray(zi, dtype=float))
        points.append({"case_id": cid, "values": finite_vec_or_none(y),
                       "final_state": finite_vec_or_none(zf.ravel())})
    except Exception:
        points.append({"case_id": cid, "values": None, "final_state": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize sosfilt query");
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
                "failed to spawn python3 for sosfilt oracle: {e}"
            );
            eprintln!("skipping sosfilt oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open sosfilt oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "sosfilt oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping sosfilt oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for sosfilt oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "sosfilt oracle failed: {stderr}"
        );
        eprintln!("skipping sosfilt oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse sosfilt oracle JSON"))
}

#[test]
fn diff_signal_sosfilt() {
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
    let mut ledger = CompareLedger::new(
        "diff_signal_sosfilt",
        &["sosfilt", "sosfilt_state", "sosfilt_stream", "sosfilt_stream_state"],
    );

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let sos: Vec<SosSection> = case.sos.to_vec();
        let (output, final_state) =
            sosfilt_with_state(&sos, &case.x, case.zi.as_deref()).expect("valid stateful filter");
        // Keep exercising the existing zero-state entry point and its fast paths.
        if case.zi.is_none() {
            assert_eq!(sosfilt(&sos, &case.x).unwrap(), output);
        }
        let final_state: Vec<f64> = final_state.into_iter().flatten().collect();
        let mut filter = SosFilter::new(&sos, case.zi.as_deref()).unwrap();
        let mut streamed = vec![0.0; case.x.len()];
        for (input, output) in case.x.chunks(7).zip(streamed.chunks_mut(7)) {
            filter.filter_into(input, output).unwrap();
        }
        let stream_state: Vec<f64> = filter.state().iter().flatten().copied().collect();
        let arms = [
            ("sosfilt", scipy_arm.values.as_deref(), output.as_slice()),
            (
                "sosfilt_state",
                scipy_arm.final_state.as_deref(),
                final_state.as_slice(),
            ),
            ("sosfilt_stream", scipy_arm.values.as_deref(), streamed.as_slice()),
            (
                "sosfilt_stream_state",
                scipy_arm.final_state.as_deref(),
                stream_state.as_slice(),
            ),
        ];
        for (arm, reference, actual) in arms {
            let Some((scipy_v, fsci_v)) =
                ledger.slices(arm, &case.case_id, reference, Some(actual))
            else {
                continue;
            };
            let abs_diff = fsci_v
                .iter()
                .zip(scipy_v)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            max_overall = max_overall.max(abs_diff);
            let pass = abs_diff <= ABS_TOL;
            ledger.compared(arm, &case.case_id, pass);
            diffs.push(CaseDiff {
                case_id: format!("{arm}/{}", case.case_id),
                abs_diff,
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_signal_sosfilt".into(),
        category: "scipy.signal.sosfilt".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);
    for d in &diffs {
        if !d.pass {
            eprintln!("sosfilt mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }
    assert!(
        all_pass,
        "scipy.signal.sosfilt conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    ledger.finish(4 * query.points.len());
}

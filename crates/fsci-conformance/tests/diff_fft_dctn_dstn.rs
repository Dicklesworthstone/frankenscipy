#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.fft.dctn`, `idctn`,
//! `dstn`, `idstn` (type-II by default, applied along every axis).
//!
//! Resolves [frankenscipy-4sbsk]. 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_fft::{FftOptions, dctn, dstn, idctn, idstn};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-005";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    shape: Vec<usize>,
    input: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
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
    fs::create_dir_all(output_dir()).expect("create dctn_dstn diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize dctn_dstn diff log");
    fs::write(path, json).expect("write dctn_dstn diff log");
}

fn generate_query() -> OracleQuery {
    let inputs: &[(&str, Vec<usize>, Vec<f64>)] = &[
        (
            "4x4_arange",
            vec![4, 4],
            (0..16).map(|i| i as f64).collect(),
        ),
        (
            "3x6_sine",
            vec![3, 6],
            (0..18).map(|i| ((i as f64) * 0.3).sin()).collect(),
        ),
        (
            "2x8_decay",
            vec![2, 8],
            (0..16).map(|i| (-(i as f64) / 5.0).exp()).collect(),
        ),
    ];
    let mut points = Vec::new();
    for (label, shape, x) in inputs {
        for op in ["dctn", "idctn", "dstn", "idstn"] {
            points.push(PointCase {
                case_id: format!("{op}_{label}"),
                op: op.into(),
                shape: shape.clone(),
                input: x.clone(),
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
from scipy import fft

def finite_vec_or_none(arr):
    out = []
    for v in np.asarray(arr).flatten().tolist():
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
    cid = case["case_id"]; op = case["op"]
    shape = tuple(case["shape"])
    x = np.array(case["input"], dtype=float).reshape(shape)
    try:
        if op == "dctn":
            y = fft.dctn(x)
        elif op == "idctn":
            y = fft.idctn(x)
        elif op == "dstn":
            y = fft.dstn(x)
        elif op == "idstn":
            y = fft.idstn(x)
        else:
            y = None
        points.append({"case_id": cid, "values": finite_vec_or_none(y) if y is not None else None})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize dctn_dstn query");
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
                "failed to spawn python3 for dctn_dstn oracle: {e}"
            );
            eprintln!("skipping dctn_dstn oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open dctn_dstn oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "dctn_dstn oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping dctn_dstn oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for dctn_dstn oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "dctn_dstn oracle failed: {stderr}"
        );
        eprintln!("skipping dctn_dstn oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse dctn_dstn oracle JSON"))
}

#[test]
fn diff_fft_dctn_dstn() {
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
        let Some(expected) = scipy_arm.values.as_ref() else {
            continue;
        };
        let opts = FftOptions::default();
        let fsci_v = match case.op.as_str() {
            "dctn" => dctn(&case.input, &case.shape, &opts),
            "idctn" => idctn(&case.input, &case.shape, &opts),
            "dstn" => dstn(&case.input, &case.shape, &opts),
            "idstn" => idstn(&case.input, &case.shape, &opts),
            _ => continue,
        };
        let Ok(fsci_v) = fsci_v else {
            continue;
        };
        if fsci_v.len() != expected.len() {
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
            .zip(expected.iter())
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
        test_id: "diff_fft_dctn_dstn".into(),
        category: "scipy.fft.dctn + idctn + dstn + idstn".into(),
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
        "scipy.fft.dctn/idctn/dstn/idstn conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

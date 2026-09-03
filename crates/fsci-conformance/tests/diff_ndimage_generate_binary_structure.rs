#![forbid(unsafe_code)]
//! Live SciPy differential coverage for
//! `scipy.ndimage.generate_binary_structure`.
//!
//! Resolves [frankenscipy-p1lr2]. 1e-12 abs (0/1 values).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::generate_binary_structure;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    ndim: usize,
    connectivity: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    values: Option<Vec<f64>>,
    shape: Option<Vec<usize>>,
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
    fs::create_dir_all(output_dir()).expect("create gen_bin_struct diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize gen_bin_struct diff log");
    fs::write(path, json).expect("write gen_bin_struct diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    for ndim in [1_usize, 2, 3] {
        for connectivity in 1..=ndim {
            points.push(PointCase {
                case_id: format!("ndim{ndim}_c{connectivity}"),
                ndim,
                connectivity,
            });
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
import numpy as np
from scipy import ndimage

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    nd = int(case["ndim"]); c = int(case["connectivity"])
    try:
        s = ndimage.generate_binary_structure(nd, c)
        points.append({
            "case_id": cid,
            "values": [float(v) for v in s.astype(float).flatten().tolist()],
            "shape": list(s.shape),
        })
    except Exception:
        points.append({"case_id": cid, "values": None, "shape": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize gen_bin_struct query");
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
                "failed to spawn python3 for gen_bin_struct oracle: {e}"
            );
            eprintln!("skipping gen_bin_struct oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open gen_bin_struct oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "gen_bin_struct oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping gen_bin_struct oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for gen_bin_struct oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "gen_bin_struct oracle failed: {stderr}"
        );
        eprintln!("skipping gen_bin_struct oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse gen_bin_struct oracle JSON"))
}

#[test]
fn diff_ndimage_generate_binary_structure() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let (Some(expected), Some(shape)) = (scipy_arm.values.as_ref(), scipy_arm.shape.as_ref())
        else {
            continue;
        };
        let fsci = generate_binary_structure(case.ndim, case.connectivity);
        let abs_d = if fsci.shape != *shape || fsci.data.len() != expected.len() {
            f64::INFINITY
        } else {
            fsci.data
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_ndimage_generate_binary_structure".into(),
        category: "scipy.ndimage.generate_binary_structure".into(),
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
                "gen_bin_struct mismatch: {} abs_diff={}",
                d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "gen_bin_struct conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

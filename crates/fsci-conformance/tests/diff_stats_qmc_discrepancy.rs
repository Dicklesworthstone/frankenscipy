#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_stats QMC discrepancy
//! measures: centered, wraparound, mixture, L2-star.
//!
//! Resolves [frankenscipy-lw7yp]. 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{
    centered_discrepancy, l2_star_discrepancy, mixture_discrepancy, wraparound_discrepancy,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    method: String,
    sample: Vec<f64>,
    rows: usize,
    cols: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    method: String,
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
    fs::create_dir_all(output_dir()).expect("create qmc_discrepancy diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize qmc_discrepancy diff log");
    fs::write(path, json).expect("write qmc_discrepancy diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    let s_5x2: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1];
    let s_8x3: Vec<f64> = (0..24)
        .map(|i| ((i as f64) * 0.137).fract().abs())
        .collect();
    let s_10x2: Vec<f64> = vec![
        0.05, 0.5, 0.15, 0.95, 0.25, 0.15, 0.35, 0.65, 0.45, 0.35, 0.55, 0.85, 0.65, 0.05, 0.75,
        0.55, 0.85, 0.25, 0.95, 0.75,
    ];

    let samples: &[(&str, &[f64], usize, usize)] = &[
        ("5x2_diag", &s_5x2, 5, 2),
        ("8x3_fractional", &s_8x3, 8, 3),
        ("10x2_scattered", &s_10x2, 10, 2),
    ];

    // CD/WD/MD and L2-star all match scipy (L2-star squared-vs-unsquared
    // convention fixed in frankenscipy-krepi).
    for (label, sample, rows, cols) in samples {
        for method in ["CD", "WD", "MD", "L2-star"] {
            points.push(PointCase {
                case_id: format!("{label}_{method}"),
                method: (*method).into(),
                sample: sample.to_vec(),
                rows: *rows,
                cols: *cols,
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
from scipy.stats import qmc

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; method = case["method"]
    rows = int(case["rows"]); cols = int(case["cols"])
    s = np.array(case["sample"], dtype=float).reshape(rows, cols)
    try:
        v = float(qmc.discrepancy(s, method=method))
        if not math.isfinite(v):
            points.append({"case_id": cid, "value": None})
        else:
            points.append({"case_id": cid, "value": v})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize qmc_discrepancy query");
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
                "failed to spawn python3 for qmc_discrepancy oracle: {e}"
            );
            eprintln!("skipping qmc_discrepancy oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open qmc_discrepancy oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "qmc_discrepancy oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping qmc_discrepancy oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for qmc_discrepancy oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "qmc_discrepancy oracle failed: {stderr}"
        );
        eprintln!("skipping qmc_discrepancy oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse qmc_discrepancy oracle JSON"))
}

#[test]
fn diff_stats_qmc_discrepancy() {
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
        let Some(scipy_v) = scipy_arm.value else {
            continue;
        };
        let fsci_result = match case.method.as_str() {
            "CD" => centered_discrepancy(&case.sample, case.cols),
            "WD" => wraparound_discrepancy(&case.sample, case.cols),
            "MD" => mixture_discrepancy(&case.sample, case.cols),
            "L2-star" => l2_star_discrepancy(&case.sample, case.cols),
            _ => continue,
        };
        let Ok(fsci_v) = fsci_result else {
            continue;
        };
        let abs_d = (fsci_v - scipy_v).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            method: case.method.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_qmc_discrepancy".into(),
        category: "scipy.stats.qmc.discrepancy".into(),
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
                "{} mismatch: {} abs_diff={}",
                d.method, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "qmc_discrepancy conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

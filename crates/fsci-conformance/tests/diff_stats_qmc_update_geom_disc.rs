#![forbid(unsafe_code)]
//! Live scipy parity for fsci_stats qmc `geometric_discrepancy`
//! (both MinDist and Mst variants) and `update_centered_discrepancy`.
//!
//! Resolves [frankenscipy-68ea5]. Tolerances: 1e-12 abs.
//!
//! `update_centered_discrepancy` is checked against
//! scipy.stats.qmc.update_discrepancy since frankenscipy-wzk18.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{
    GeometricDiscrepancyMethod, centered_discrepancy, geometric_discrepancy,
    update_centered_discrepancy,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct UpdCase {
    case_id: String,
    /// Existing flat row-major sample.
    existing: Vec<f64>,
    dimension: usize,
    new_point: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct GeomCase {
    case_id: String,
    /// Flat row-major sample.
    sample: Vec<f64>,
    dimension: usize,
    method: String, // "mindist" | "mst"
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    update: Vec<UpdCase>,
    geom: Vec<GeomCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct ArmScalar {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    update: Vec<ArmScalar>,
    geom: Vec<ArmScalar>,
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
    fs::create_dir_all(output_dir()).expect("create qmc upd/geom diff dir");
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
    let s_2d_4: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let s_2d_5: Vec<f64> = vec![0.05, 0.15, 0.25, 0.35, 0.55, 0.45, 0.75, 0.65, 0.95, 0.85];
    let s_3d_4: Vec<f64> = vec![
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.15, 0.25, 0.35,
    ];

    let update = vec![
        UpdCase {
            case_id: "upd_2d_4_to_5".into(),
            existing: s_2d_4.clone(),
            dimension: 2,
            new_point: vec![0.9, 0.95],
        },
        UpdCase {
            case_id: "upd_2d_5_to_6".into(),
            existing: s_2d_5.clone(),
            dimension: 2,
            new_point: vec![0.5, 0.5],
        },
        UpdCase {
            case_id: "upd_3d_4_to_5".into(),
            existing: s_3d_4.clone(),
            dimension: 3,
            new_point: vec![0.5, 0.55, 0.65],
        },
    ];

    let mut geom = Vec::new();
    for &(label, sample, dim) in &[
        ("2d_4", s_2d_4.as_slice(), 2usize),
        ("2d_5", s_2d_5.as_slice(), 2),
        ("3d_4", s_3d_4.as_slice(), 3),
    ] {
        for &m in &["mindist", "mst"] {
            geom.push(GeomCase {
                case_id: format!("geom_{label}_{m}"),
                sample: sample.to_vec(),
                dimension: dim,
                method: m.into(),
            });
        }
    }

    OracleQuery { update, geom }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.stats import qmc

q = json.load(sys.stdin)

update_out = []
for case in q["update"]:
    cid = case["case_id"]
    dim = int(case["dimension"])
    existing_flat = case["existing"]
    n_existing = len(existing_flat) // dim
    existing = np.array(existing_flat, dtype=float).reshape(n_existing, dim)
    new_pt = np.array(case["new_point"], dtype=float)
    try:
        prev = float(qmc.discrepancy(existing, method="CD"))
        v = float(qmc.update_discrepancy(new_pt, existing, prev))
        if math.isfinite(v):
            update_out.append({"case_id": cid, "value": v})
        else:
            update_out.append({"case_id": cid, "value": None})
    except Exception as e:
        sys.stderr.write(f"upd {cid}: {e}\n")
        update_out.append({"case_id": cid, "value": None})

geom_out = []
for case in q["geom"]:
    cid = case["case_id"]
    dim = int(case["dimension"])
    flat = case["sample"]
    n = len(flat) // dim
    sample = np.array(flat, dtype=float).reshape(n, dim)
    method = case["method"]
    try:
        v = float(qmc.geometric_discrepancy(sample, method=method))
        if math.isfinite(v):
            geom_out.append({"case_id": cid, "value": v})
        else:
            geom_out.append({"case_id": cid, "value": None})
    except Exception as e:
        sys.stderr.write(f"geom {cid}: {e}\n")
        geom_out.append({"case_id": cid, "value": None})

print(json.dumps({"update": update_out, "geom": geom_out}))
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
                "failed to spawn python3 for qmc upd/geom oracle: {e}"
            );
            eprintln!("skipping qmc upd/geom oracle: python3 not available ({e})");
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
                "qmc upd/geom oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping qmc upd/geom oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for qmc upd/geom oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "qmc upd/geom oracle failed: {stderr}"
        );
        eprintln!("skipping qmc upd/geom oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse qmc upd/geom oracle JSON"))
}

#[test]
fn diff_stats_qmc_update_geom_disc() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let upd_map: HashMap<String, ArmScalar> = oracle
        .update
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let geom_map: HashMap<String, ArmScalar> = oracle
        .geom
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    // update_centered_discrepancy vs scipy.stats.qmc.update_discrepancy,
    // both seeded with the centered discrepancy of the existing sample
    // (frankenscipy-wzk18 — additive-update formula fixed).
    for case in &query.update {
        let Some(arm) = upd_map.get(&case.case_id) else {
            continue;
        };
        let Some(expected) = arm.value else {
            continue;
        };
        let Ok(prev) = centered_discrepancy(&case.existing, case.dimension) else {
            continue;
        };
        let Ok(actual) =
            update_centered_discrepancy(&case.existing, case.dimension, prev, &case.new_point)
        else {
            continue;
        };
        let abs_d = (actual - expected).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "update_centered_discrepancy".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    for case in &query.geom {
        let Some(arm) = geom_map.get(&case.case_id) else {
            continue;
        };
        let Some(expected) = arm.value else {
            continue;
        };
        let method = match case.method.as_str() {
            "mindist" => GeometricDiscrepancyMethod::MinDist,
            "mst" => GeometricDiscrepancyMethod::Mst,
            _ => continue,
        };
        let Ok(actual) = geometric_discrepancy(&case.sample, case.dimension, method) else {
            continue;
        };
        let abs_d = (actual - expected).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: format!("geometric_discrepancy_{}", case.method),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_qmc_update_geom_disc".into(),
        category:
            "fsci_stats qmc update_centered_discrepancy + geometric_discrepancy vs scipy.stats.qmc"
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
        "qmc upd/geom conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

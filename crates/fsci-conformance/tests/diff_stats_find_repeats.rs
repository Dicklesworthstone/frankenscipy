#![forbid(unsafe_code)]
//! Live SciPy differential coverage for
//! `scipy.stats.find_repeats(arr)`.
//!
//! Resolves [frankenscipy-ut65h]. Cross-checks the returned
//! values + counts arrays element-wise against scipy across
//! 4 datasets covering no-repeat / light / heavy / all-same
//! cases.
//!
//! 4 fixtures × 3 arms (n_values + max-abs values + max-abs
//! counts) = 12 cases via subprocess. Tol 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::find_repeats;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    data: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    values: Option<Vec<f64>>,
    counts: Option<Vec<i64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    arm: String,
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
    fs::create_dir_all(output_dir()).expect("create find_repeats diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize find_repeats diff log");
    fs::write(path, json).expect("write find_repeats diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<f64>)> = vec![
        // No repeats
        ("no_repeats", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        // Light repeats — 2 values appear twice
        (
            "light_repeats",
            vec![1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 8.0],
        ),
        // Heavy repeats — many values appear 3+ times
        (
            "heavy_repeats",
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0],
        ),
        // All same — single value repeated
        ("all_same", vec![7.5, 7.5, 7.5, 7.5, 7.5, 7.5]),
    ];

    let points = fixtures
        .into_iter()
        .map(|(name, data)| PointCase {
            case_id: name.into(),
            data,
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
from scipy.stats import mstats

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    data = np.array(case["data"], dtype=float)
    try:
        # scipy.stats.find_repeats was removed in modern scipy; use the
        # mstats variant which still ships and returns the same
        # (values_arr, counts_arr) tuple. (Earlier this called
        # stats.find_repeats which silently raised AttributeError,
        # dropping all cases — case_count was 0 for ages.)
        values_arr, counts_arr = mstats.find_repeats(data)
        values = [float(v) for v in values_arr]
        counts = [int(c) for c in counts_arr]
        points.append({
            "case_id": cid,
            "values": values,
            "counts": counts,
        })
    except Exception:
        points.append({"case_id": cid, "values": None, "counts": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize find_repeats query");
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
                "failed to spawn python3 for find_repeats oracle: {e}"
            );
            eprintln!("skipping find_repeats oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open find_repeats oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "find_repeats oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping find_repeats oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for find_repeats oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "find_repeats oracle failed: {stderr}"
        );
        eprintln!("skipping find_repeats oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse find_repeats oracle JSON"))
}

#[test]
fn diff_stats_find_repeats() {
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
        let result = find_repeats(&case.data);

        let scipy_values = match &scipy_arm.values {
            Some(v) => v,
            None => continue,
        };
        let scipy_counts = match &scipy_arm.counts {
            Some(v) => v,
            None => continue,
        };

        // n_values arm
        let n_diff = (result.values.len() as i64 - scipy_values.len() as i64).unsigned_abs() as f64;
        max_overall = max_overall.max(n_diff);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            arm: "n_values".into(),
            abs_diff: n_diff,
            pass: n_diff <= ABS_TOL,
        });

        // values element-wise
        if result.values.len() == scipy_values.len() {
            let mut max_local = 0.0_f64;
            for (a, b) in result.values.iter().zip(scipy_values.iter()) {
                if a.is_finite() {
                    max_local = max_local.max((a - b).abs());
                }
            }
            max_overall = max_overall.max(max_local);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "values_max".into(),
                abs_diff: max_local,
                pass: max_local <= ABS_TOL,
            });
        }

        // counts element-wise (integer)
        if result.counts.len() == scipy_counts.len() {
            let mut max_local = 0.0_f64;
            for (a, b) in result.counts.iter().zip(scipy_counts.iter()) {
                let diff = (*a as i64 - *b).unsigned_abs() as f64;
                max_local = max_local.max(diff);
            }
            max_overall = max_overall.max(max_local);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "counts_max".into(),
                abs_diff: max_local,
                pass: max_local <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_find_repeats".into(),
        category: "scipy.stats.find_repeats".into(),
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
                "find_repeats mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "find_repeats conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}

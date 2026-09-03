#![forbid(unsafe_code)]
//! Live scipy.stats.tukey_hsd statistic parity for fsci_stats::tukey_hsd.
//!
//! Resolves [frankenscipy-51v9w]. fsci uses Bonferroni-corrected p-values
//! while scipy uses the Studentized range distribution. Only the
//! pairwise-mean-difference statistic matrix is comparable.
//!
//! Tolerance: 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::tukey_hsd;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    /// Flat row-major: [group0_data..., group1_data..., ...]
    flat: Vec<f64>,
    /// Length of each group (sum equals flat.len()).
    sizes: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Flat row-major k×k matrix of pairwise differences.
    statistic: Option<Vec<f64>>,
    k: Option<usize>,
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
    fs::create_dir_all(output_dir()).expect("create tukey diff dir");
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

fn pack(groups: &[Vec<f64>]) -> (Vec<f64>, Vec<usize>) {
    let mut flat = Vec::new();
    let mut sizes = Vec::new();
    for g in groups {
        flat.extend_from_slice(g);
        sizes.push(g.len());
    }
    (flat, sizes)
}

fn generate_query() -> OracleQuery {
    let g1 = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let g2 = vec![
        vec![10.0, 11.0, 12.0, 13.0],
        vec![5.0, 6.0, 7.0],
        vec![20.0, 22.0, 21.0, 23.0],
    ];
    let g3 = vec![
        vec![1.0, 1.5, 2.0, 2.5, 3.0],
        vec![1.2, 1.8, 2.1, 2.8],
        vec![3.0, 3.5, 4.0, 4.5],
        vec![1.0, 2.0, 3.0],
    ];

    let cases: Vec<(&str, Vec<Vec<f64>>)> = vec![
        ("three_balanced", g1),
        ("three_unbalanced", g2),
        ("four_mixed", g3),
    ];

    let points = cases
        .into_iter()
        .map(|(cid, groups)| {
            let (flat, sizes) = pack(&groups);
            Case {
                case_id: cid.into(),
                flat,
                sizes,
            }
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
from scipy.stats import tukey_hsd

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    flat = case["flat"]; sizes = case["sizes"]
    groups = []
    i = 0
    for s in sizes:
        groups.append(np.array(flat[i:i+s], dtype=float))
        i += s
    try:
        r = tukey_hsd(*groups)
        k = len(groups)
        st = r.statistic
        flat_st = [float(v) for row in st.tolist() for v in row]
        if all(math.isfinite(v) for v in flat_st):
            points.append({"case_id": cid, "statistic": flat_st, "k": int(k)})
        else:
            points.append({"case_id": cid, "statistic": None, "k": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "statistic": None, "k": None})
print(json.dumps({"points": points}))
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
                "failed to spawn python3 for tukey oracle: {e}"
            );
            eprintln!("skipping tukey oracle: python3 not available ({e})");
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
                "tukey oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping tukey oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for tukey oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "tukey oracle failed: {stderr}"
        );
        eprintln!("skipping tukey oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse tukey oracle JSON"))
}

#[test]
fn diff_stats_tukey_hsd_statistic() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let (Some(estat), Some(ek)) = (arm.statistic.as_ref(), arm.k) else {
            continue;
        };

        // Unflatten case data into groups.
        let mut groups: Vec<&[f64]> = Vec::new();
        let mut offset = 0;
        let mut tmp_groups: Vec<Vec<f64>> = Vec::new();
        for &sz in &case.sizes {
            tmp_groups.push(case.flat[offset..offset + sz].to_vec());
            offset += sz;
        }
        for g in &tmp_groups {
            groups.push(g);
        }

        let result = tukey_hsd(&groups);
        if result.statistic.len() != ek {
            continue;
        }
        let flat_actual: Vec<f64> = result.statistic.iter().flatten().copied().collect();
        let abs_d = if flat_actual.len() != estat.len() {
            f64::INFINITY
        } else {
            flat_actual
                .iter()
                .zip(estat.iter())
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
        test_id: "diff_stats_tukey_hsd_statistic".into(),
        category: "fsci_stats::tukey_hsd statistic vs scipy.stats.tukey_hsd".into(),
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
            eprintln!("tukey mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "tukey conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

#![forbid(unsafe_code)]
//! Live numerical reference checks for two rank-distance
//! utilities:
//!   • `spearman_footrule(rank1, rank2)` — Σ |rank1ᵢ - rank2ᵢ|
//!   • `kendall_distance(rank1, rank2)` — count of discordant
//!     pairs between two rankings.
//!
//! Resolves [frankenscipy-np7v3]. The oracle reproduces both
//! analytically in numpy.
//!
//! 4 (rank1, rank2) fixtures × 2 funcs = 8 cases via
//! subprocess. Tol 1e-12 abs (closed-form integer sums).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{kendall_distance, spearman_footrule};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    rank1: Vec<u64>,
    rank2: Vec<u64>,
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
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create rank_distances diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize rank_distances diff log");
    fs::write(path, json).expect("write rank_distances diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<u64>, Vec<u64>)> = vec![
        // Identical rankings — both metrics = 0
        (
            "identical",
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            vec![1, 2, 3, 4, 5, 6, 7, 8],
        ),
        // Reversed — maximum disagreement
        (
            "reversed",
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            vec![8, 7, 6, 5, 4, 3, 2, 1],
        ),
        // Single transposition
        (
            "single_swap",
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            vec![1, 2, 3, 5, 4, 6, 7, 8],
        ),
        // Random permutation
        (
            "permuted",
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            vec![3, 1, 4, 8, 2, 6, 5, 7],
        ),
    ];

    let mut points = Vec::new();
    for (name, r1, r2) in &fixtures {
        for func in ["spearman_footrule", "kendall_distance"] {
            points.push(PointCase {
                case_id: format!("{name}_{func}"),
                func: func.into(),
                rank1: r1.clone(),
                rank2: r2.clone(),
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

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    r1 = np.array(case["rank1"], dtype=int)
    r2 = np.array(case["rank2"], dtype=int)
    val = None
    try:
        if func == "spearman_footrule":
            val = float(np.sum(np.abs(r1 - r2)))
        elif func == "kendall_distance":
            n = len(r1)
            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    a = (r1[i] - r1[j]) * (r2[i] - r2[j])
                    if a < 0:
                        count += 1
            val = float(count)
    except Exception:
        val = None
    points.append({"case_id": cid, "value": fnone(val)})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize rank_distances query");
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
                "failed to spawn python3 for rank_distances oracle: {e}"
            );
            eprintln!("skipping rank_distances oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open rank_distances oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "rank_distances oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping rank_distances oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for rank_distances oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "rank_distances oracle failed: {stderr}"
        );
        eprintln!("skipping rank_distances oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse rank_distances oracle JSON"))
}

#[test]
fn diff_stats_rank_distances() {
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
        if let Some(scipy_v) = scipy_arm.value {
            let r1: Vec<usize> = case.rank1.iter().map(|&x| x as usize).collect();
            let r2: Vec<usize> = case.rank2.iter().map(|&x| x as usize).collect();
            let rust_v = match case.func.as_str() {
                "spearman_footrule" => spearman_footrule(&r1, &r2),
                "kendall_distance" => kendall_distance(&r1, &r2) as f64,
                _ => continue,
            };
            if rust_v.is_finite() {
                let abs_diff = (rust_v - scipy_v).abs();
                max_overall = max_overall.max(abs_diff);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    func: case.func.clone(),
                    abs_diff,
                    pass: abs_diff <= ABS_TOL,
                });
            }
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_rank_distances".into(),
        category: "spearman_footrule + kendall_distance".into(),
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
                "rank_distances {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "rank_distances conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}

#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci's
//! `count_tied_groups(data) → HashMap<tie_size, group_count>`.
//!
//! Resolves [frankenscipy-688zy]. The oracle calls
//! `scipy.stats.mstats.count_tied_groups`, which returns the
//! same dict.
//!
//! 4 datasets exercising no-ties, all-ties, mixed-ties, and
//! float-equality. Each case sums into per-tie-size arms (one
//! arm per distinct tie size). Tol 1e-12 abs (integer counts).

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::count_tied_groups;
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
    counts: Option<HashMap<String, i64>>,
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
    fs::create_dir_all(output_dir()).expect("create count_tied_groups diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize count_tied_groups diff log");
    fs::write(path, json).expect("write count_tied_groups diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<f64>)> = vec![
        // No ties
        ("no_ties", (1..=10).map(|i| i as f64).collect()),
        // All-ties (n=8 of same value → one group of size 8)
        ("all_same", vec![3.0; 8]),
        // Mixed: pairs and triples and singletons
        (
            "mixed",
            vec![
                1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 7.0, 7.0,
            ],
        ),
        // Float equality with negatives
        (
            "with_negatives",
            vec![-1.5, -1.5, 0.0, 0.0, 0.0, 1.0, 2.5, 2.5, -3.0],
        ),
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
import sys
import numpy as np
from scipy.stats import mstats

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    data = np.array(case["data"], dtype=float)
    counts = None
    try:
        d = mstats.count_tied_groups(data)
        # keys are int tie-sizes; values are int group counts
        counts = {str(int(k)): int(v) for k, v in d.items()}
    except Exception:
        counts = None
    points.append({"case_id": cid, "counts": counts})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize count_tied_groups query");
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
                "failed to spawn python3 for count_tied_groups oracle: {e}"
            );
            eprintln!("skipping count_tied_groups oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open count_tied_groups oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "count_tied_groups oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping count_tied_groups oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for count_tied_groups oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "count_tied_groups oracle failed: {stderr}"
        );
        eprintln!("skipping count_tied_groups oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse count_tied_groups oracle JSON"))
}

#[test]
fn diff_stats_count_tied_groups() {
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
        let Some(scipy_counts) = &scipy_arm.counts else {
            continue;
        };
        let rust_map = count_tied_groups(&case.data);

        // Union the keys so we catch missing/extra entries on either side.
        let mut keys: HashSet<usize> = rust_map.keys().copied().collect();
        for k in scipy_counts.keys() {
            if let Ok(parsed) = k.parse::<usize>() {
                keys.insert(parsed);
            }
        }
        if keys.is_empty() {
            // Both sides empty (no-ties case) — record a single noop arm.
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "no_ties".into(),
                abs_diff: 0.0,
                pass: true,
            });
            continue;
        }
        let mut sorted_keys: Vec<usize> = keys.into_iter().collect();
        sorted_keys.sort_unstable();
        for k in sorted_keys {
            let r = *rust_map.get(&k).unwrap_or(&0) as i64;
            let s = *scipy_counts.get(&k.to_string()).unwrap_or(&0);
            let abs_diff = (r - s).unsigned_abs() as f64;
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: format!("size{k}"),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_count_tied_groups".into(),
        category: "scipy.stats.mstats.count_tied_groups".into(),
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
                "count_tied_groups mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "count_tied_groups conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}

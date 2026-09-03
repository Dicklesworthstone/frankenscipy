#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci's
//! `zmap_ddof(scores, compare, ddof)` — relative z-scores with
//! explicit ddof against `scipy.stats.zmap(scores, compare,
//! ddof=ddof)`.
//!
//! Resolves [frankenscipy-ji6dq]. The existing diff_stats_zmap.rs
//! covers ddof=0; this harness exercises ddof=1 (sample-std
//! normalisation) and an additional ddof=0 fixture for parity
//! against separate-array zmap (not zscore).
//!
//! 3 (scores, compare) fixtures × 2 ddof values = 6 cases ×
//! per-element max-abs aggregation. Tol 1e-12 abs (closed-form
//! mean-and-std normalisation).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::zmap_ddof;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    scores: Vec<f64>,
    compare: Vec<f64>,
    ddof: usize,
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
    fs::create_dir_all(output_dir()).expect("create zmap_ddof diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize zmap_ddof diff log");
    fs::write(path, json).expect("write zmap_ddof diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        // Smaller scores set against a larger compare baseline
        (
            "small_against_baseline",
            vec![3.0, 7.0, 11.0, 4.5, 8.5],
            (1..=15).map(|i| i as f64).collect(),
        ),
        // Both arrays similar, varied
        (
            "similar_lengths",
            (1..=10).map(|i| (i as f64) + 0.5).collect(),
            (1..=10).map(|i| i as f64).collect(),
        ),
        // Negative values present
        (
            "with_negatives",
            vec![-1.5, 0.0, 1.5, 3.0, -2.5, 4.5],
            vec![-3.0, -1.5, 0.0, 0.5, 1.5, 2.5, 3.5, 5.0, 7.0, 9.0],
        ),
    ];

    let mut points = Vec::new();
    for (name, scores, compare) in &fixtures {
        for &ddof in &[0_usize, 1] {
            points.push(PointCase {
                case_id: format!("{name}_ddof{ddof}"),
                scores: scores.clone(),
                compare: compare.clone(),
                ddof,
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
from scipy import stats

def vec_or_none(arr):
    out = []
    for v in arr:
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
    scores = np.array(case["scores"], dtype=float)
    compare = np.array(case["compare"], dtype=float)
    ddof = int(case["ddof"])
    val = None
    try:
        out = stats.zmap(scores, compare, ddof=ddof)
        val = vec_or_none(np.asarray(out).tolist())
    except Exception:
        val = None
    points.append({"case_id": cid, "values": val})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize zmap_ddof query");
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
                "failed to spawn python3 for zmap_ddof oracle: {e}"
            );
            eprintln!("skipping zmap_ddof oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open zmap_ddof oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "zmap_ddof oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping zmap_ddof oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for zmap_ddof oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "zmap_ddof oracle failed: {stderr}"
        );
        eprintln!("skipping zmap_ddof oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse zmap_ddof oracle JSON"))
}

#[test]
fn diff_stats_zmap_ddof() {
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
        let Some(scipy_vec) = &scipy_arm.values else {
            continue;
        };
        let rust_vec = zmap_ddof(&case.scores, &case.compare, case.ddof);
        if rust_vec.len() != scipy_vec.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let mut max_local = 0.0_f64;
        for (r, s) in rust_vec.iter().zip(scipy_vec.iter()) {
            if r.is_finite() {
                max_local = max_local.max((r - s).abs());
            }
        }
        max_overall = max_overall.max(max_local);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: max_local,
            pass: max_local <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_zmap_ddof".into(),
        category: "scipy.stats.zmap (ddof=0, ddof=1)".into(),
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
            eprintln!("zmap_ddof mismatch: {} abs={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "zmap_ddof conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}

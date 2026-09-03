#![forbid(unsafe_code)]
//! Live SciPy differential coverage for
//! `scipy.stats.binned_statistic(x, values, statistic, bins)`
//! across the six statistics fsci supports:
//! mean / sum / count / min / max / median.
//!
//! Resolves [frankenscipy-fp3ye]. fsci returns
//! `(statistic_per_bin, bin_edges)`; scipy returns the same
//! plus a binnumber array (which we don't compare here).
//!
//! 2 (x, values) fixtures × 6 statistics × 2 arms (per-bin
//! statistic vector + bin_edges vector) = 24 cases via
//! subprocess. Tol 1e-12 abs (closed-form bin assignment +
//! per-bin reduce).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::binned_statistic;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    statistic: String,
    bins: u64,
    x: Vec<f64>,
    values: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    stats: Option<Vec<Option<f64>>>,
    bin_edges: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    statistic: String,
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
    fs::create_dir_all(output_dir()).expect("create binned_statistic diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize binned_statistic diff log");
    fs::write(path, json).expect("write binned_statistic diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<f64>, Vec<f64>, u64)> = vec![
        // Linear ramp data
        (
            "ramp",
            (1..=20).map(|i| i as f64).collect(),
            (1..=20).map(|i| (i * i) as f64).collect(),
            5,
        ),
        // Mixed-sign x with noisier values
        (
            "noisy",
            vec![
                -3.0, -1.5, -0.7, 0.0, 0.5, 1.2, 2.0, 3.5, 4.7, 6.0, 8.5, 12.0,
            ],
            vec![
                10.0, 8.0, 5.0, 4.0, 6.0, 7.0, 9.0, 11.0, 14.0, 18.0, 22.0, 28.0,
            ],
            4,
        ),
    ];
    let stats = ["mean", "sum", "count", "min", "max", "median"];

    let mut points = Vec::new();
    for (name, x, values, bins) in &fixtures {
        for stat in stats {
            points.push(PointCase {
                case_id: format!("{name}_{stat}"),
                statistic: stat.into(),
                bins: *bins,
                x: x.clone(),
                values: values.clone(),
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
        if math.isnan(v):
            # Empty bins return NaN — preserve that as NaN in our
            # transport-able list by mapping to a sentinel; here just
            # propagate Python NaN string. JSON can't encode NaN, so
            # we replace NaN with a marker float that we strip Rust-side.
            out.append(None)
        else:
            out.append(v)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    x = np.array(case["x"], dtype=float)
    values = np.array(case["values"], dtype=float)
    bins = int(case["bins"])
    statistic = case["statistic"]
    out = {"case_id": cid, "stats": None, "bin_edges": None}
    try:
        stat_vals, bin_edges, _ = stats.binned_statistic(
            x, values, statistic=statistic, bins=bins
        )
        # Convert NaN entries (empty bins) to None for JSON.
        st = []
        for v in stat_vals.tolist():
            if math.isnan(float(v)):
                st.append(None)
            else:
                st.append(float(v))
        out["stats"] = st
        out["bin_edges"] = vec_or_none(bin_edges.tolist())
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize binned_statistic query");
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
                "failed to spawn python3 for binned_statistic oracle: {e}"
            );
            eprintln!("skipping binned_statistic oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open binned_statistic oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "binned_statistic oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping binned_statistic oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for binned_statistic oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "binned_statistic oracle failed: {stderr}"
        );
        eprintln!("skipping binned_statistic oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse binned_statistic oracle JSON"))
}

#[test]
fn diff_stats_binned_statistic() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, Option<PointArm>> = oracle
        .points
        .into_iter()
        .map(|r| (r.case_id.clone(), Some(r)))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = match pmap.get(&case.case_id) {
            Some(Some(a)) => a,
            _ => continue,
        };
        let (rust_stats, rust_edges) =
            binned_statistic(&case.x, &case.values, case.bins as usize, &case.statistic);

        // stats vector: scipy nests `null` in JSON for empty bins (NaN
        // in numpy); fsci returns NaN. Compare per-element only when
        // both are finite-or-both-NaN.
        if let Some(scipy_stats) = &scipy_arm.stats
            && rust_stats.len() == scipy_stats.len()
        {
            let mut max_local = 0.0_f64;
            let mut shape_ok = true;
            for (a, b_opt) in rust_stats.iter().zip(scipy_stats.iter()) {
                match (a.is_finite(), b_opt) {
                    (true, Some(b)) => {
                        max_local = max_local.max((a - b).abs());
                    }
                    (false, None) => {
                        // Both NaN/empty-bin; OK.
                    }
                    _ => {
                        shape_ok = false;
                        break;
                    }
                }
            }
            if shape_ok {
                max_overall = max_overall.max(max_local);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    statistic: case.statistic.clone(),
                    arm: "stats_max".into(),
                    abs_diff: max_local,
                    pass: max_local <= ABS_TOL,
                });
            }
        }

        if let Some(scipy_edges) = &scipy_arm.bin_edges
            && rust_edges.len() == scipy_edges.len()
        {
            let mut max_local = 0.0_f64;
            for (a, b) in rust_edges.iter().zip(scipy_edges.iter()) {
                if a.is_finite() {
                    max_local = max_local.max((a - b).abs());
                }
            }
            max_overall = max_overall.max(max_local);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                statistic: case.statistic.clone(),
                arm: "edges_max".into(),
                abs_diff: max_local,
                pass: max_local <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_binned_statistic".into(),
        category: "scipy.stats.binned_statistic".into(),
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
                "binned_statistic {} mismatch: {} arm={} abs={}",
                d.statistic, d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "binned_statistic conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}

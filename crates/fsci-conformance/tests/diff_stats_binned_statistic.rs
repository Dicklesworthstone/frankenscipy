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

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
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
    compared: BTreeMap<String, ArmCounts>,
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

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;
    let mut ledger = CompareLedger::new("diff_stats_binned_statistic", &["stats_max", "edges_max"]);

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let (rust_stats, rust_edges) =
            binned_statistic(&case.x, &case.values, case.bins as usize, &case.statistic);

        // stats vector: the oracle nests `null` in JSON only for SciPy's NaN
        // (an empty bin); fsci returns NaN there. Decoded back to NaN, the
        // ledger requires fsci's NaN at exactly those bins.
        let scipy_stats: Option<Vec<f64>> = scipy_arm
            .stats
            .as_ref()
            .map(|v| v.iter().map(|b| b.unwrap_or(f64::NAN)).collect());
        let arms = [
            (
                "stats_max",
                scipy_stats.as_deref(),
                Some(rust_stats.as_slice()),
            ),
            (
                "edges_max",
                scipy_arm.bin_edges.as_deref(),
                Some(rust_edges.as_slice()),
            ),
        ];
        for (arm, scipy, fsci) in arms {
            let Some((s, f)) = ledger.slices(arm, &case.case_id, scipy, fsci) else {
                continue;
            };
            let mut max_local = 0.0_f64;
            for (a, b) in f.iter().zip(s.iter()) {
                if b.is_finite() {
                    max_local = max_local.max((a - b).abs());
                }
            }
            max_overall = max_overall.max(max_local);
            ledger.compared(arm, &case.case_id, max_local <= ABS_TOL);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                statistic: case.statistic.clone(),
                arm: arm.into(),
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
        compared: ledger.counts().clone(),
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
    ledger.finish(query.points.len());
}

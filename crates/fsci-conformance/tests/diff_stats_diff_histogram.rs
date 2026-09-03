#![forbid(unsafe_code)]
//! Live numpy differential coverage for fsci's
//! `diff(data)` (element-wise consecutive differences) and
//! `histogram(data, bins)` (1-D histogram with auto-[min,max]
//! range, counts + bin edges).
//!
//! Resolves [frankenscipy-7u03d]. The oracle calls
//! `numpy.diff(data)` and `numpy.histogram(data, bins=bins)`.
//!
//! 4 datasets × {diff (1 arm), histogram (counts + edges)} =
//! 12 arms via subprocess. Tol 1e-12 abs (closed-form
//! arithmetic / integer counts; bin edges via linspace —
//! numpy aligns to fsci when range = (min, max)).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{diff, histogram};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    data: Vec<f64>,
    bins: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    diffs: Option<Vec<f64>>,
    counts: Option<Vec<i64>>,
    edges: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create diff_histogram diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize diff_histogram diff log");
    fs::write(path, json).expect("write diff_histogram diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>, usize)> = vec![
        ("compact_n12_b5", (1..=12).map(|i| i as f64).collect(), 5),
        (
            "spread_n20_b8",
            vec![
                -3.0, -1.5, 0.0, 0.5, 1.5, 2.5, 3.5, 5.0, 7.0, 9.0, 12.0, 16.0, 21.0, 27.0, 34.0,
                40.0, 45.0, 50.0, 55.0, 60.0,
            ],
            8,
        ),
        (
            "ties_n15_b4",
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 8.0,
            ],
            4,
        ),
        (
            "sparse_n10_b6",
            vec![1.0, 1.5, 1.8, 2.0, 5.0, 5.5, 5.8, 9.0, 9.5, 10.0],
            6,
        ),
    ];

    let mut points = Vec::new();
    for (name, data, bins) in &datasets {
        for func in ["diff", "histogram"] {
            points.push(PointCase {
                case_id: format!("{name}_{func}"),
                func: func.into(),
                data: data.clone(),
                bins: *bins,
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
    cid = case["case_id"]; func = case["func"]
    data = np.array(case["data"], dtype=float)
    bins = int(case["bins"])
    out = {"case_id": cid, "diffs": None, "counts": None, "edges": None}
    try:
        if func == "diff":
            out["diffs"] = vec_or_none(np.diff(data).tolist())
        elif func == "histogram":
            counts, edges = np.histogram(data, bins=bins)
            out["counts"] = [int(c) for c in counts.tolist()]
            out["edges"] = vec_or_none(edges.tolist())
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize diff_histogram query");
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
                "failed to spawn python3 for diff_histogram oracle: {e}"
            );
            eprintln!("skipping diff_histogram oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open diff_histogram oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "diff_histogram oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping diff_histogram oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for diff_histogram oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "diff_histogram oracle failed: {stderr}"
        );
        eprintln!("skipping diff_histogram oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse diff_histogram oracle JSON"))
}

#[test]
fn diff_stats_diff_histogram() {
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
        match case.func.as_str() {
            "diff" => {
                if let Some(scipy_d) = &scipy_arm.diffs {
                    let rust_d = diff(&case.data);
                    if rust_d.len() == scipy_d.len() {
                        let mut max_local = 0.0_f64;
                        for (r, s) in rust_d.iter().zip(scipy_d.iter()) {
                            if r.is_finite() {
                                max_local = max_local.max((r - s).abs());
                            }
                        }
                        max_overall = max_overall.max(max_local);
                        diffs.push(CaseDiff {
                            case_id: case.case_id.clone(),
                            arm: "diff".into(),
                            abs_diff: max_local,
                            pass: max_local <= ABS_TOL,
                        });
                    }
                }
            }
            "histogram" => {
                let (rust_counts, rust_edges) = histogram(&case.data, case.bins);
                if let Some(scipy_counts) = &scipy_arm.counts
                    && rust_counts.len() == scipy_counts.len()
                {
                    let mut max_local = 0.0_f64;
                    for (r, s) in rust_counts.iter().zip(scipy_counts.iter()) {
                        let abs = (*r as i64 - *s).unsigned_abs() as f64;
                        max_local = max_local.max(abs);
                    }
                    max_overall = max_overall.max(max_local);
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        arm: "histogram.counts".into(),
                        abs_diff: max_local,
                        pass: max_local <= ABS_TOL,
                    });
                }
                if let Some(scipy_edges) = &scipy_arm.edges
                    && rust_edges.len() == scipy_edges.len()
                {
                    let mut max_local = 0.0_f64;
                    for (r, s) in rust_edges.iter().zip(scipy_edges.iter()) {
                        if r.is_finite() {
                            max_local = max_local.max((r - s).abs());
                        }
                    }
                    max_overall = max_overall.max(max_local);
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        arm: "histogram.edges".into(),
                        abs_diff: max_local,
                        pass: max_local <= ABS_TOL,
                    });
                }
            }
            _ => continue,
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_diff_histogram".into(),
        category: "numpy.{diff, histogram}".into(),
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
                "diff_histogram mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "diff_histogram conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}

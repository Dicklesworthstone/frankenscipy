#![forbid(unsafe_code)]
//! Live numpy differential coverage for fsci's `cumsum(data)`
//! and `cumprod(data)` — running totals over a 1-D sequence.
//!
//! Resolves [frankenscipy-ccfbh]. The oracle calls
//! `numpy.{cumsum, cumprod}`.
//!
//! 4 datasets × 2 funcs × per-element max-abs aggregation =
//! 8 cases. Tol 1e-12 for cumsum (additive); 1e-10 for cumprod
//! (multiplicative noise grows with sequence length on noisy
//! inputs — empirical precision floor on near-1 ratios).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{cumprod, cumsum};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const CUMSUM_TOL: f64 = 1.0e-12;
const CUMPROD_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create cumsum_cumprod diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize cumsum_cumprod diff log");
    fs::write(path, json).expect("write cumsum_cumprod diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        // Compact integers
        ("compact_n10", (1..=10).map(|i| i as f64).collect()),
        // Spread including negatives (cumsum can shrink)
        (
            "with_negatives_n12",
            vec![
                -2.5, 1.0, 3.5, -1.5, 2.0, 4.0, -0.5, 6.0, -3.0, 7.5, 1.5, -1.0,
            ],
        ),
        // Near-1 ratios (favoured for cumprod stability)
        (
            "near_one_ratios_n15",
            (0..15)
                .map(|i| 1.0 + 0.05 * ((i as f64) * 0.7).sin())
                .collect(),
        ),
        // Larger sequence with mild range
        (
            "moderate_n20",
            (1..=20).map(|i| 0.95 + (i as f64) * 0.02).collect(),
        ),
    ];

    let mut points = Vec::new();
    for (name, data) in &datasets {
        for func in ["cumsum", "cumprod"] {
            points.push(PointCase {
                case_id: format!("{name}_{func}"),
                func: func.into(),
                data: data.clone(),
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
    val = None
    try:
        if func == "cumsum":
            val = vec_or_none(np.cumsum(data).tolist())
        elif func == "cumprod":
            val = vec_or_none(np.cumprod(data).tolist())
    except Exception:
        val = None
    points.append({"case_id": cid, "values": val})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize cumsum_cumprod query");
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
                "failed to spawn python3 for cumsum_cumprod oracle: {e}"
            );
            eprintln!("skipping cumsum_cumprod oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open cumsum_cumprod oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "cumsum_cumprod oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping cumsum_cumprod oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for cumsum_cumprod oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "cumsum_cumprod oracle failed: {stderr}"
        );
        eprintln!("skipping cumsum_cumprod oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse cumsum_cumprod oracle JSON"))
}

#[test]
fn diff_stats_cumsum_cumprod() {
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
        let (rust_vec, tol) = match case.func.as_str() {
            "cumsum" => (cumsum(&case.data), CUMSUM_TOL),
            "cumprod" => (cumprod(&case.data), CUMPROD_TOL),
            _ => continue,
        };
        if rust_vec.len() != scipy_vec.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
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
            func: case.func.clone(),
            abs_diff: max_local,
            pass: max_local <= tol,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_cumsum_cumprod".into(),
        category: "numpy.{cumsum, cumprod}".into(),
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
                "cumsum_cumprod {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "cumsum_cumprod conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}

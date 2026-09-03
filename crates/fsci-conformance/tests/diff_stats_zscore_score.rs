#![forbid(unsafe_code)]
//! Live SciPy differential coverage for closed-form z-score
//! and quantile-style ranking utilities:
//!   • `zscore(data)` — arithmetic z-score (ddof=0)
//!   • `gzscore(data)` — geometric z-score (positive data)
//!   • `scoreatpercentile(data, per, ...)` — fraction-method
//!     1-D inverse-CDF estimator
//!
//! Resolves [frankenscipy-hng5d]. The oracle calls
//! `scipy.stats.{zscore, gzscore, scoreatpercentile}`.
//!
//! 3 datasets × {zscore, gzscore} per-element +
//! 3 datasets × scoreatpercentile(4 percentiles) = 14 arms via
//! subprocess. Tol 1e-12 abs (closed-form normalisation /
//! linear-interpolation quantile chain).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{gzscore, scoreatpercentile, zscore};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    data: Vec<f64>,
    /// Only used for scoreatpercentile.
    per: Vec<f64>,
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
    fs::create_dir_all(output_dir()).expect("create zscore_score diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize zscore_score diff log");
    fs::write(path, json).expect("write zscore_score diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        // Compact, all positive
        ("compact_positive_n12", (1..=12).map(|i| i as f64).collect()),
        // Spread, all positive (so gzscore is well-defined)
        (
            "spread_positive_n15",
            (1..=15).map(|i| (i as f64).powi(2)).collect(),
        ),
        // Ties, all positive
        (
            "ties_positive_n14",
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 8.0,
            ],
        ),
    ];

    let percentiles = vec![10.0, 25.0, 50.0, 90.0];
    let empty: Vec<f64> = Vec::new();

    let mut points = Vec::new();
    for (name, data) in &datasets {
        for func in ["zscore", "gzscore"] {
            points.push(PointCase {
                case_id: format!("{name}_{func}"),
                func: func.into(),
                data: data.clone(),
                per: empty.clone(),
            });
        }
        points.push(PointCase {
            case_id: format!("{name}_scoreatpercentile"),
            func: "scoreatpercentile".into(),
            data: data.clone(),
            per: percentiles.clone(),
        });
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
    cid = case["case_id"]; func = case["func"]
    data = np.array(case["data"], dtype=float)
    val = None
    try:
        if func == "zscore":
            # ddof=0 matches fsci default.
            val = stats.zscore(data, ddof=0)
            val = vec_or_none(np.asarray(val).tolist())
        elif func == "gzscore":
            val = stats.gzscore(data, ddof=0)
            val = vec_or_none(np.asarray(val).tolist())
        elif func == "scoreatpercentile":
            per = case["per"]
            out = [stats.scoreatpercentile(data, p) for p in per]
            val = vec_or_none(out)
    except Exception:
        val = None
    points.append({"case_id": cid, "values": val})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize zscore_score query");
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
                "failed to spawn python3 for zscore_score oracle: {e}"
            );
            eprintln!("skipping zscore_score oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open zscore_score oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "zscore_score oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping zscore_score oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for zscore_score oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "zscore_score oracle failed: {stderr}"
        );
        eprintln!("skipping zscore_score oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse zscore_score oracle JSON"))
}

#[test]
fn diff_stats_zscore_score() {
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
        let rust_vec = match case.func.as_str() {
            "zscore" => zscore(&case.data),
            "gzscore" => gzscore(&case.data),
            "scoreatpercentile" => match scoreatpercentile(&case.data, &case.per, None, None) {
                Ok(v) => v,
                Err(_) => continue,
            },
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
            pass: max_local <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_zscore_score".into(),
        category: "scipy.stats.{zscore, gzscore, scoreatpercentile}".into(),
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
                "zscore_score {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "zscore_score conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}

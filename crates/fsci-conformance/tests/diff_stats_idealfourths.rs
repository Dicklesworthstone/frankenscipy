#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci's
//! `idealfourths(data) → (qlo, qup)` (robust quartile
//! estimators) and standard `iqr(data) = Q3 - Q1`.
//!
//! Resolves [frankenscipy-0vcob]. The oracle calls
//! `scipy.stats.mstats.idealfourths(data)` and
//! `scipy.stats.iqr(a)`.
//!
//! 4 datasets × 3 arms (qlo + qup + iqr) = 12 cases via
//! subprocess. Tol 1e-12 abs (closed-form linear-interpolation
//! quantile chain).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{idealfourths, iqr};
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
    qlo: Option<f64>,
    qup: Option<f64>,
    iqr: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create idealfourths diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize idealfourths diff log");
    fs::write(path, json).expect("write idealfourths diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        ("compact_n12", (1..=12).map(|i| i as f64).collect()),
        (
            "spread_n15",
            vec![
                -3.0, -1.5, 0.0, 0.5, 1.5, 2.5, 3.5, 5.0, 7.0, 9.0, 12.0, 16.0, 21.0, 27.0, 34.0,
            ],
        ),
        (
            "ties_n14",
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 6.0,
            ],
        ),
        ("small_n5", vec![1.0, 4.0, 7.0, 9.0, 13.0]),
    ];

    let points = datasets
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
from scipy import stats
from scipy.stats import mstats

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    data = np.array(case["data"], dtype=float)
    out = {"case_id": cid, "qlo": None, "qup": None, "iqr": None}
    try:
        qlo, qup = mstats.idealfourths(data)
        out["qlo"] = fnone(qlo)
        out["qup"] = fnone(qup)
    except Exception:
        pass
    try:
        out["iqr"] = fnone(stats.iqr(data))
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize idealfourths query");
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
                "failed to spawn python3 for idealfourths oracle: {e}"
            );
            eprintln!("skipping idealfourths oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open idealfourths oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "idealfourths oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping idealfourths oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for idealfourths oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "idealfourths oracle failed: {stderr}"
        );
        eprintln!("skipping idealfourths oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse idealfourths oracle JSON"))
}

#[test]
fn diff_stats_idealfourths() {
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
        let (rust_qlo, rust_qup) = idealfourths(&case.data);
        let rust_iqr = iqr(&case.data);

        if let Some(s_qlo) = scipy_arm.qlo
            && rust_qlo.is_finite()
        {
            let abs_diff = (rust_qlo - s_qlo).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "qlo".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
        if let Some(s_qup) = scipy_arm.qup
            && rust_qup.is_finite()
        {
            let abs_diff = (rust_qup - s_qup).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "qup".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
        if let Some(s_iqr) = scipy_arm.iqr
            && rust_iqr.is_finite()
        {
            let abs_diff = (rust_iqr - s_iqr).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "iqr".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_idealfourths".into(),
        category: "scipy.stats.mstats.idealfourths + scipy.stats.iqr".into(),
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
                "idealfourths mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "idealfourths conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}

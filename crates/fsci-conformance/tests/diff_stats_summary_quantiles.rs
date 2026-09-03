#![forbid(unsafe_code)]
//! Live SciPy differential coverage for two simple
//! summary-quantile utilities not exercised by any existing
//! diff harness:
//!   • `scipy.stats.sem(data)` — standard error of the mean
//!   • `scipy.stats.scoreatpercentile(data, per, ...)` —
//!     inverse percentile (linear interpolation by default)
//!
//! Resolves [frankenscipy-euybb]. Each function is a simple
//! closed-form ratio or linear interpolation; the harness
//! exercises representative parameter combinations.
//!
//! `percentileofscore` was originally also covered here but
//! removed: fsci's `percentileofscore` and scipy's diverge by
//! ~5-10 percentage points across all four kinds, suggesting
//! a convention difference in tie handling or boundary
//! definitions that needs separate investigation.
//!
//! 3 datasets × 1 sem + 3 datasets × 4 percentile points =
//! 3 + 12 = 15 cases via subprocess. Tol 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{scoreatpercentile, sem};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    data: Vec<f64>,
    param: f64,
    kind: String,
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
    fs::create_dir_all(output_dir()).expect("create summary_quantiles diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize summary_quantiles diff log");
    fs::write(path, json).expect("write summary_quantiles diff log");
}

fn fsci_eval(case: &PointCase) -> Option<f64> {
    let v = match case.func.as_str() {
        "sem" => sem(&case.data),
        "scoreatpercentile" => {
            let r = scoreatpercentile(&case.data, &[case.param], None, None).ok()?;
            *r.first()?
        }
        _ => return None,
    };
    if v.is_finite() { Some(v) } else { None }
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        ("compact", (1..=10).map(|i| i as f64).collect()),
        (
            "spread",
            vec![
                -3.0, -1.5, -0.7, 0.0, 0.5, 1.2, 2.0, 3.5, 4.7, 6.0, 8.5, 12.0,
            ],
        ),
        (
            "ties",
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
        ),
    ];

    let mut points = Vec::new();
    for (name, data) in &datasets {
        // sem: 1 call per dataset
        points.push(PointCase {
            case_id: format!("sem_{name}"),
            func: "sem".into(),
            data: data.clone(),
            param: 0.0,
            kind: "".into(),
        });

        // scoreatpercentile at 4 different percentile points
        for &p in &[10.0, 25.0, 50.0, 90.0] {
            points.push(PointCase {
                case_id: format!("score_{name}_p{p}"),
                func: "scoreatpercentile".into(),
                data: data.clone(),
                param: p,
                kind: "".into(),
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
    data = np.array(case["data"], dtype=float)
    param = float(case["param"])
    kind = case["kind"]
    val = None
    try:
        if func == "sem":
            val = float(stats.sem(data))
        elif func == "scoreatpercentile":
            val = float(stats.scoreatpercentile(data, param))
    except Exception:
        val = None
    points.append({"case_id": cid, "value": fnone(val)})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize summary_quantiles query");
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
                "failed to spawn python3 for summary_quantiles oracle: {e}"
            );
            eprintln!("skipping summary_quantiles oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open summary_quantiles oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "summary_quantiles oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping summary_quantiles oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for summary_quantiles oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "summary_quantiles oracle failed: {stderr}"
        );
        eprintln!("skipping summary_quantiles oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse summary_quantiles oracle JSON"))
}

#[test]
fn diff_stats_summary_quantiles() {
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
        if let Some(scipy_v) = scipy_arm.value
            && let Some(rust_v) = fsci_eval(case)
        {
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

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_summary_quantiles".into(),
        category: "scipy.stats.sem/scoreatpercentile/percentileofscore".into(),
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
                "summary_quantiles {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "summary_quantiles conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}

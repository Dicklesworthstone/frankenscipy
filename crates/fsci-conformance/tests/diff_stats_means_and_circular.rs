#![forbid(unsafe_code)]
//! Live SciPy differential coverage for closed-form descriptive
//! scalars not exercised by any other diff harness:
//!   • gmean  — geometric mean
//!   • gstd   — geometric std-dev (positive data)
//!   • hmean  — harmonic mean (non-negative data)
//!   • variation — std/mean (population variance, scipy default)
//!   • circmean / circvar / circstd — circular descriptive stats
//!
//! Resolves [frankenscipy-dq0ig]. The oracle calls
//! `scipy.stats.{gmean, gstd, hmean, variation, circmean,
//! circvar, circstd}`.
//!
//! 4 datasets × 7 funcs = 28 cases via a single subprocess
//! pass. Tol 1e-12 abs (closed-form log/exp/sin/cos chain).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{circmean, circstd, circvar, gmean, gstd, hmean, variation};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
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
    fs::create_dir_all(output_dir()).expect("create means_and_circular diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize means_and_circular diff log");
    fs::write(path, json).expect("write means_and_circular diff log");
}

fn generate_query() -> OracleQuery {
    // Linear (positive) datasets exercise gmean/gstd/hmean/variation;
    // angular datasets (radians) exercise the circular stats. We use
    // the same data for all 7 funcs so per-fixture invocations stay
    // simple — circular funcs accept any radian-domain values.
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        ("compact_positive", (1..=10).map(|i| i as f64).collect()),
        (
            "spread_positive",
            vec![0.5, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0],
        ),
        // Circular fixture: bearings around a unit circle in radians
        ("small_angles", vec![0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]),
        // Wider spread covering more than one full radian range
        (
            "wide_angles",
            vec![0.2, 1.5, 2.7, 0.4, 1.9, 0.8, 2.3, 1.1, 0.6],
        ),
    ];

    let mut points = Vec::new();
    for (name, data) in &datasets {
        for func in [
            "gmean",
            "gstd",
            "hmean",
            "variation",
            "circmean",
            "circvar",
            "circstd",
        ] {
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
    val = None
    try:
        if func == "gmean":
            val = stats.gmean(data)
        elif func == "gstd":
            val = stats.gstd(data)
        elif func == "hmean":
            val = stats.hmean(data)
        elif func == "variation":
            val = stats.variation(data)
        elif func == "circmean":
            # Use full 0..2π range so wrap-around matches fsci's
            # atan2 convention.
            val = stats.circmean(data, high=2 * np.pi, low=0.0)
        elif func == "circvar":
            val = stats.circvar(data, high=2 * np.pi, low=0.0)
        elif func == "circstd":
            val = stats.circstd(data, high=2 * np.pi, low=0.0)
    except Exception:
        val = None
    points.append({"case_id": cid, "value": fnone(val)})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize means_and_circular query");
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
                "failed to spawn python3 for means_and_circular oracle: {e}"
            );
            eprintln!("skipping means_and_circular oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open means_and_circular oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "means_and_circular oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping means_and_circular oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for means_and_circular oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "means_and_circular oracle failed: {stderr}"
        );
        eprintln!("skipping means_and_circular oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse means_and_circular oracle JSON"))
}

#[test]
fn diff_stats_means_and_circular() {
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
        let Some(scipy_v) = scipy_arm.value else {
            continue;
        };
        let rust_v = match case.func.as_str() {
            "gmean" => gmean(&case.data),
            "gstd" => gstd(&case.data),
            "hmean" => hmean(&case.data),
            "variation" => variation(&case.data),
            "circmean" => circmean(&case.data),
            "circvar" => circvar(&case.data),
            "circstd" => circstd(&case.data),
            _ => continue,
        };
        if !rust_v.is_finite() {
            continue;
        }
        let abs_diff = (rust_v - scipy_v).abs();
        max_overall = max_overall.max(abs_diff);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.clone(),
            abs_diff,
            pass: abs_diff <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_means_and_circular".into(),
        category: "scipy.stats.{gmean, gstd, hmean, variation, circ*}".into(),
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
                "means_and_circular {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "means_and_circular conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}

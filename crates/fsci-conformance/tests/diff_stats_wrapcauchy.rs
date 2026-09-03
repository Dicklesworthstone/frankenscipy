#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.wrapcauchy`.
//!
//! Resolves [frankenscipy-cgx7f]. WrapCauchy has anchor tests
//! in `fsci-stats/src/lib.rs` but no dedicated scipy diff
//! harness. 6 c values × 9 x-values × 2 families (pdf, cdf)
//! via subprocess.
//!
//! pdf is a closed-form rational. cdf is closed-form atan/tan
//! with two branches joined at π (uses reflection symmetry on
//! [π, 2π)). 1e-13 abs holds for both.
//!
//! ppf is intentionally omitted — fsci's WrapCauchy doesn't
//! override ppf, so it would inherit trait-default bisection
//! over the cdf, producing a result whose precision is
//! dominated by the bisection's stop tolerance rather than fsci
//! arithmetic.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{ContinuousDistribution, WrapCauchy};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const PDF_TOL: f64 = 1.0e-13;
const CDF_TOL: f64 = 1.0e-13;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    c: f64,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    pdf: Option<f64>,
    cdf: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    family: String,
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
    fs::create_dir_all(output_dir()).expect("create wrapcauchy diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize wrapcauchy diff log");
    fs::write(path, json).expect("write wrapcauchy diff log");
}

fn generate_query() -> OracleQuery {
    // c spans uniform-on-circle-ish (small c) through
    // highly-concentrated (c=0.9). c=0.0 omitted because scipy
    // wrapcauchy returns NaN at the parameter boundary even
    // though fsci handles it as the uniform limit. x walks both
    // half-circles to exercise the branch joined at π.
    let cs = [0.05_f64, 0.1, 0.3, 0.5, 0.7, 0.9];
    let xs = [
        0.1_f64,
        0.5,
        1.0,
        PI / 2.0,
        2.5,
        PI,
        4.0,
        3.0 * PI / 2.0,
        2.0 * PI - 0.1,
    ];
    let mut points = Vec::new();
    for &c in &cs {
        for &x in &xs {
            points.push(PointCase {
                case_id: format!("c{c}_x{x:.3}"),
                c,
                x,
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
from scipy.stats import wrapcauchy

def finite_or_none(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    c = float(case["c"]); x = float(case["x"])
    try:
        points.append({
            "case_id": cid,
            "pdf": finite_or_none(wrapcauchy.pdf(x, c)),
            "cdf": finite_or_none(wrapcauchy.cdf(x, c)),
        })
    except Exception:
        points.append({"case_id": cid, "pdf": None, "cdf": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize wrapcauchy query");
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
                "failed to spawn python3 for wrapcauchy oracle: {e}"
            );
            eprintln!("skipping wrapcauchy oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open wrapcauchy oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "wrapcauchy oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping wrapcauchy oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for wrapcauchy oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "wrapcauchy oracle failed: {stderr}"
        );
        eprintln!("skipping wrapcauchy oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse wrapcauchy oracle JSON"))
}

#[test]
fn diff_stats_wrapcauchy() {
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
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        let dist = WrapCauchy::new(case.c);
        if let Some(spdf) = oracle.pdf {
            let d = (dist.pdf(case.x) - spdf).abs();
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: "pdf".into(),
                abs_diff: d,
                pass: d <= PDF_TOL,
            });
        }
        if let Some(scdf) = oracle.cdf {
            let d = (dist.cdf(case.x) - scdf).abs();
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: "cdf".into(),
                abs_diff: d,
                pass: d <= CDF_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_wrapcauchy".into(),
        category: "scipy.stats.wrapcauchy".into(),
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
                "wrapcauchy {} mismatch: {} abs_diff={}",
                d.family, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats.wrapcauchy conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.skewnorm`.
//!
//! Resolves [frankenscipy-a8fca]. The SkewNorm port shipped in
//! a80d4a0 has 4 anchor tests but no live scipy oracle. This
//! harness drives 5 a-values × 9 x-values for pdf/cdf/sf and
//! 5 a-values for moments through scipy via subprocess and asserts
//! byte-stable agreement on pdf at tol 1e-12 and cdf at tol 1e-9
//! (Owen's T computed via 10-pt Gauss-Legendre quadrature).
//! Skips cleanly if scipy/python3 is unavailable.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{ContinuousDistribution, SkewNorm};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const PDF_TOL: f64 = 1.0e-12;
const CDF_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    a: f64,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct MomentCase {
    case_id: String,
    a: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
    moments: Vec<MomentCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    pdf: Option<f64>,
    cdf: Option<f64>,
    sf: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct MomentArm {
    case_id: String,
    mean: Option<f64>,
    var: Option<f64>,
    skew: Option<f64>,
    kurt: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
    moments: Vec<MomentArm>,
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
    fs::create_dir_all(output_dir()).expect("create skewnorm diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize skewnorm diff log");
    fs::write(path, json).expect("write skewnorm diff log");
}

fn generate_query() -> OracleQuery {
    let a_values = [-2.0_f64, -0.5, 0.0, 0.5, 2.0];
    let x_values = [-3.0_f64, -1.5, -0.5, -0.1, 0.0, 0.1, 0.5, 1.5, 3.0];
    let mut points = Vec::new();
    for &a in &a_values {
        for &x in &x_values {
            points.push(PointCase {
                case_id: format!("a{a}_x{x}"),
                a,
                x,
            });
        }
    }
    let moments: Vec<MomentCase> = a_values
        .iter()
        .map(|&a| MomentCase {
            case_id: format!("a{a}"),
            a,
        })
        .collect();
    OracleQuery { points, moments }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
from scipy.stats import skewnorm

q = json.load(sys.stdin)
points = []
for c in q["points"]:
    cid = c["case_id"]
    a = float(c["a"])
    x = float(c["x"])
    try:
        points.append({
            "case_id": cid,
            "pdf": float(skewnorm.pdf(x, a)),
            "cdf": float(skewnorm.cdf(x, a)),
            "sf":  float(skewnorm.sf(x, a)),
        })
    except Exception:
        points.append({"case_id": cid, "pdf": None, "cdf": None, "sf": None})
moments = []
for c in q["moments"]:
    cid = c["case_id"]
    a = float(c["a"])
    try:
        m, v, s, k = skewnorm.stats(a, moments='mvsk')
        moments.append({
            "case_id": cid,
            "mean": float(m),
            "var":  float(v),
            "skew": float(s),
            "kurt": float(k),
        })
    except Exception:
        moments.append({"case_id": cid, "mean": None, "var": None, "skew": None, "kurt": None})
print(json.dumps({"points": points, "moments": moments}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize skewnorm query");
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
                "failed to spawn python3 for skewnorm oracle: {e}"
            );
            eprintln!("skipping skewnorm oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open skewnorm oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "skewnorm oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping skewnorm oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for skewnorm oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "skewnorm oracle failed: {stderr}"
        );
        eprintln!("skipping skewnorm oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse skewnorm oracle JSON"))
}

#[test]
fn diff_stats_skewnorm() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());
    assert_eq!(oracle.moments.len(), query.moments.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();
    let mmap: HashMap<String, MomentArm> = oracle
        .moments
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle map");
        let dist = SkewNorm::new(case.a);
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
        if let Some(ssf) = oracle.sf {
            let d = (dist.sf(case.x) - ssf).abs();
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: "sf".into(),
                abs_diff: d,
                pass: d <= CDF_TOL,
            });
        }
    }

    for case in &query.moments {
        let oracle = mmap.get(&case.case_id).expect("validated oracle map");
        let dist = SkewNorm::new(case.a);
        for (label, rust_v, scipy_v) in [
            ("mean", dist.mean(), oracle.mean),
            ("var", dist.var(), oracle.var),
            ("skew", dist.skewness(), oracle.skew),
            ("kurt", dist.kurtosis(), oracle.kurt),
        ] {
            if let Some(s) = scipy_v {
                let d = (rust_v - s).abs();
                max_overall = max_overall.max(d);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    family: label.into(),
                    abs_diff: d,
                    pass: d <= 1e-12,
                });
            }
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_skewnorm".into(),
        category: "scipy.stats.skewnorm".into(),
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
                "skewnorm {} mismatch: {} abs_diff={}",
                d.family, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats.skewnorm conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

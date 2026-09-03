#![forbid(unsafe_code)]
//! Live scipy.stats.weibull_max parity for fsci_stats::FrechetR.
//!
//! Resolves [frankenscipy-3ppca]. fsci's FrechetR is documented as
//! equivalent to scipy.stats.weibull_max(c). Both have support
//! (-∞, 0]. Tolerance: 1e-10 abs (CDF/PDF), 1e-8 abs for ppf.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{ContinuousDistribution, FrechetR};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL_PDF_CDF: f64 = 1.0e-10;
const ABS_TOL_PPF: f64 = 1.0e-8;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String, // "pdf" | "cdf" | "ppf"
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
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
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
    fs::create_dir_all(output_dir()).expect("create frechet_r diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

fn generate_query() -> OracleQuery {
    let cs = [0.5_f64, 1.0, 1.5, 2.0, 3.0, 5.0];
    let pdf_xs = [-3.0_f64, -1.5, -1.0, -0.5, -0.1];
    let cdf_xs = [-3.0_f64, -1.5, -1.0, -0.5, -0.1, 0.0];
    let ppf_qs = [0.05_f64, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95];
    let mut points = Vec::new();
    for &c in &cs {
        for &x in &pdf_xs {
            points.push(PointCase {
                case_id: format!("pdf_c{c}_x{x}").replace('.', "p").replace('-', "n"),
                op: "pdf".into(),
                c,
                x,
            });
        }
        for &x in &cdf_xs {
            points.push(PointCase {
                case_id: format!("cdf_c{c}_x{x}").replace('.', "p").replace('-', "n"),
                op: "cdf".into(),
                c,
                x,
            });
        }
        for &q in &ppf_qs {
            points.push(PointCase {
                case_id: format!("ppf_c{c}_q{q}").replace('.', "p"),
                op: "ppf".into(),
                c,
                x: q, // q stored in x slot for uniformity
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
from scipy.stats import weibull_max

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    c = float(case["c"]); x = float(case["x"])
    try:
        if op == "pdf":  v = float(weibull_max.pdf(x, c))
        elif op == "cdf": v = float(weibull_max.cdf(x, c))
        elif op == "ppf": v = float(weibull_max.ppf(x, c))
        else: v = float("nan")
        if math.isfinite(v):
            points.append({"case_id": cid, "value": v})
        else:
            points.append({"case_id": cid, "value": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
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
                "failed to spawn python3 for frechet_r oracle: {e}"
            );
            eprintln!("skipping frechet_r oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "frechet_r oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping frechet_r oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for frechet_r oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "frechet_r oracle failed: {stderr}"
        );
        eprintln!("skipping frechet_r oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse frechet_r oracle JSON"))
}

#[test]
fn diff_stats_frechet_r() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let Some(expected) = arm.value else {
            continue;
        };
        let dist = FrechetR::new(case.c);
        let (actual, tol) = match case.op.as_str() {
            "pdf" => (dist.pdf(case.x), ABS_TOL_PDF_CDF),
            "cdf" => (dist.cdf(case.x), ABS_TOL_PDF_CDF),
            "ppf" => (dist.ppf(case.x), ABS_TOL_PPF),
            _ => continue,
        };
        let abs_d = (actual - expected).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= tol,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_frechet_r".into(),
        category: "fsci_stats::FrechetR vs scipy.stats.weibull_max".into(),
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
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "frechet_r conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

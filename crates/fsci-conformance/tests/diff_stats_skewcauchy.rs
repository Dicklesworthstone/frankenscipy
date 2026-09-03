#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.skewcauchy`.
//!
//! Resolves [frankenscipy-an5gs]. The SkewCauchy port shipped in
//! e4480e5 has 4 closed-form anchor cases and a fuzz harness
//! (e884e42) but no live scipy oracle. This harness drives 6
//! a-values × 7 x-values through scipy.stats.skewcauchy via
//! subprocess and asserts byte-stable agreement on pdf and cdf at
//! tol 1e-12. Skips cleanly if scipy/python3 is unavailable.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{ContinuousDistribution, SkewCauchy};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct ScCase {
    case_id: String,
    a: f64,
    x: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    case_id: String,
    pdf: Option<f64>,
    cdf: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    pdf_diff: f64,
    cdf_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    abs_tol: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create skewcauchy diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize skewcauchy diff log");
    fs::write(path, json).expect("write skewcauchy diff log");
}

fn generate_cases() -> Vec<ScCase> {
    let a_values = [-0.7_f64, -0.3, 0.0, 0.3, 0.5, 0.7];
    let x_values = [-3.0_f64, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0];
    let mut cases = Vec::new();
    for &a in &a_values {
        for &x in &x_values {
            cases.push(ScCase {
                case_id: format!("a{a}_x{x}"),
                a,
                x,
            });
        }
    }
    cases
}

fn scipy_oracle_or_skip(cases: &[ScCase]) -> Vec<OracleResult> {
    let script = r#"
import json
import sys
from scipy.stats import skewcauchy

cases = json.load(sys.stdin)
results = []
for c in cases:
    cid = c["case_id"]
    a = float(c["a"])
    x = float(c["x"])
    try:
        results.append({
            "case_id": cid,
            "pdf": float(skewcauchy.pdf(x, a)),
            "cdf": float(skewcauchy.cdf(x, a)),
        })
    except Exception:
        results.append({"case_id": cid, "pdf": None, "cdf": None})
print(json.dumps(results))
"#;

    let cases_json = serde_json::to_string(cases).expect("serialize skewcauchy cases");

    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for skewcauchy oracle: {e}"
            );
            eprintln!("skipping skewcauchy oracle: python3 not available ({e})");
            return Vec::new();
        }
    };

    {
        let stdin = child.stdin.as_mut().expect("open skewcauchy oracle stdin");
        if let Err(err) = stdin.write_all(cases_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "skewcauchy oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping skewcauchy oracle: stdin write failed ({err})\n{stderr}");
            return Vec::new();
        }
    }

    let output = child
        .wait_with_output()
        .expect("wait for skewcauchy oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "skewcauchy oracle failed: {stderr}"
        );
        eprintln!("skipping skewcauchy oracle: scipy not available\n{stderr}");
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout).expect("parse skewcauchy oracle JSON")
}

#[test]
fn diff_stats_skewcauchy() {
    let cases = generate_cases();
    let oracle_results = scipy_oracle_or_skip(&cases);
    if oracle_results.is_empty() {
        return;
    }
    assert_eq!(
        oracle_results.len(),
        cases.len(),
        "scipy skewcauchy oracle returned partial coverage"
    );

    let oracle_map: HashMap<String, OracleResult> = oracle_results
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &cases {
        let oracle = oracle_map
            .get(&case.case_id)
            .expect("validated complete oracle map");
        let (Some(scipy_pdf), Some(scipy_cdf)) = (oracle.pdf, oracle.cdf) else {
            continue;
        };
        let dist = SkewCauchy::new(case.a);
        let rust_pdf = dist.pdf(case.x);
        let rust_cdf = dist.cdf(case.x);
        let pdf_diff = (rust_pdf - scipy_pdf).abs();
        let cdf_diff = (rust_cdf - scipy_cdf).abs();
        let pass = pdf_diff <= ABS_TOL && cdf_diff <= ABS_TOL;
        max_overall = max_overall.max(pdf_diff).max(cdf_diff);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            pdf_diff,
            cdf_diff,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_skewcauchy".into(),
        category: "scipy.stats.skewcauchy".into(),
        case_count: diffs.len(),
        max_abs_diff: max_overall,
        abs_tol: ABS_TOL,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "skewcauchy mismatch: {} pdf_diff={} cdf_diff={}",
                d.case_id, d.pdf_diff, d.cdf_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats.skewcauchy conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

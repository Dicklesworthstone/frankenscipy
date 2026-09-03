#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.triang`.
//!
//! Resolves [frankenscipy-fkq6j]. Triangular has anchor tests
//! in `fsci-stats/src/lib.rs` but no dedicated scipy diff
//! harness. fsci Triangular::new(left, mode, right); the
//! harness uses (0, c, 1) to match scipy.stats.triang's
//! single-c parameterization on the standard [0, 1] support.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{ContinuousDistribution, Triangular};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const PDF_TOL: f64 = 1.0e-12;
const CDF_TOL: f64 = 1.0e-12;
const PPF_TOL_REL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    c: f64,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct PpfCase {
    case_id: String,
    c: f64,
    q: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
    ppf: Vec<PpfCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    pdf: Option<f64>,
    cdf: Option<f64>,
    sf: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct PpfArm {
    case_id: String,
    ppf: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
    ppf: Vec<PpfArm>,
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
    fs::create_dir_all(output_dir()).expect("create triang diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize triang diff log");
    fs::write(path, json).expect("write triang diff log");
}

fn generate_query() -> OracleQuery {
    // c is the mode location in [0, 1]; covers symmetric
    // (c=0.5), left-skewed (c=0.0, 0.2), right-skewed
    // (c=0.8, 1.0), and an asymmetric off-center (c=0.3, 0.7).
    let cs = [0.0_f64, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0];
    let xs = [0.05_f64, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95];
    let qs = [0.05_f64, 0.25, 0.5, 0.75, 0.95];
    let mut points = Vec::new();
    for &c in &cs {
        for &x in &xs {
            points.push(PointCase {
                case_id: format!("c{c}_x{x}"),
                c,
                x,
            });
        }
    }
    let mut ppf_cases = Vec::new();
    for &c in &cs {
        for &q in &qs {
            ppf_cases.push(PpfCase {
                case_id: format!("c{c}_q{q}"),
                c,
                q,
            });
        }
    }
    OracleQuery {
        points,
        ppf: ppf_cases,
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
from scipy.stats import triang

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    c = float(case["c"]); x = float(case["x"])
    try:
        points.append({
            "case_id": cid,
            "pdf": float(triang.pdf(x, c)),
            "cdf": float(triang.cdf(x, c)),
            "sf":  float(triang.sf(x, c)),
        })
    except Exception:
        points.append({"case_id": cid, "pdf": None, "cdf": None, "sf": None})
ppf = []
for case in q["ppf"]:
    cid = case["case_id"]; c = float(case["c"]); qv = float(case["q"])
    try:
        ppf.append({"case_id": cid, "ppf": float(triang.ppf(qv, c))})
    except Exception:
        ppf.append({"case_id": cid, "ppf": None})
print(json.dumps({"points": points, "ppf": ppf}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize triang query");
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
                "failed to spawn python3 for triang oracle: {e}"
            );
            eprintln!("skipping triang oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open triang oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "triang oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping triang oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for triang oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "triang oracle failed: {stderr}"
        );
        eprintln!("skipping triang oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse triang oracle JSON"))
}

#[test]
fn diff_stats_triang() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());
    assert_eq!(oracle.ppf.len(), query.ppf.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();
    let ppfmap: HashMap<String, PpfArm> = oracle
        .ppf
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        let dist = Triangular::new(0.0, case.c, 1.0);
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

    for case in &query.ppf {
        let oracle = ppfmap.get(&case.case_id).expect("validated oracle");
        if let Some(sppf) = oracle.ppf {
            let dist = Triangular::new(0.0, case.c, 1.0);
            let rust = dist.ppf(case.q);
            let d = (rust - sppf).abs();
            let scale = sppf.abs().max(1.0);
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: "ppf".into(),
                abs_diff: d,
                pass: d <= PPF_TOL_REL * scale,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_triang".into(),
        category: "scipy.stats.triang".into(),
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
                "triang {} mismatch: {} abs_diff={}",
                d.family, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats.triang conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

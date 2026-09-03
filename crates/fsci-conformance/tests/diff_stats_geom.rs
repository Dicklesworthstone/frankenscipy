#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.geom`.
//!
//! Resolves [frankenscipy-ciq59]. Geometric has anchor tests in
//! `fsci-stats/src/lib.rs` but no dedicated scipy diff harness.
//! 6 p-values × 21 k-values × 2 families (pmf, cdf) via
//! subprocess.
//!
//! pmf and cdf both have closed forms (no helper composition,
//! no log-gamma), so a tight 1e-13 abs tolerance holds.
//! Support: k ∈ {1, 2, 3, …}.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{DiscreteDistribution, Geometric};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const PMF_TOL: f64 = 1.0e-13;
const CDF_TOL: f64 = 1.0e-13;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    p: f64,
    k: u64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    pmf: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create geom diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize geom diff log");
    fs::write(path, json).expect("write geom diff log");
}

fn generate_query() -> OracleQuery {
    // p covers near-1 (small mean), moderate, and small p (heavy
    // tail with mean=1/p). For each p pick 21 k-values reaching
    // ~4σ beyond mean so the tail is exercised.
    let ps = [0.05_f64, 0.1, 0.3, 0.5, 0.7, 0.95];
    let mut points = Vec::new();
    for &p in &ps {
        let mean = 1.0 / p;
        let std = ((1.0 - p) / (p * p)).sqrt();
        let kmax = (mean + 4.0 * std).ceil().max(20.0) as u64;
        let step = (kmax as f64) / 20.0;
        for i in 0..21u64 {
            // scipy.stats.geom support is k >= 1, so start at 1.
            let k = 1 + ((i as f64) * step).round() as u64;
            points.push(PointCase {
                case_id: format!("p{p}_k{k}_i{i}"),
                p,
                k,
            });
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
from scipy.stats import geom

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    p = float(case["p"]); k = int(case["k"])
    try:
        points.append({
            "case_id": cid,
            "pmf": float(geom.pmf(k, p)),
            "cdf": float(geom.cdf(k, p)),
        })
    except Exception:
        points.append({"case_id": cid, "pmf": None, "cdf": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize geom query");
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
                "failed to spawn python3 for geom oracle: {e}"
            );
            eprintln!("skipping geom oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open geom oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "geom oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping geom oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for geom oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "geom oracle failed: {stderr}"
        );
        eprintln!("skipping geom oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse geom oracle JSON"))
}

#[test]
fn diff_stats_geom() {
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
        let dist = Geometric::new(case.p);
        if let Some(spmf) = oracle.pmf {
            let d = (dist.pmf(case.k) - spmf).abs();
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: "pmf".into(),
                abs_diff: d,
                pass: d <= PMF_TOL,
            });
        }
        if let Some(scdf) = oracle.cdf {
            let d = (dist.cdf(case.k) - scdf).abs();
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
        test_id: "diff_stats_geom".into(),
        category: "scipy.stats.geom".into(),
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
                "geom {} mismatch: {} abs_diff={}",
                d.family, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats.geom conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

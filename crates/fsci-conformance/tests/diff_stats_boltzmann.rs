#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.boltzmann`.
//!
//! Resolves [frankenscipy-iif7f]. The Boltzmann port shipped in
//! 50f7d82 has 3 anchor tests but no live scipy oracle. This
//! harness drives 5 λ-values × 4 N-values × every k ∈ {0..N−1}
//! for pmf and cdf through scipy via subprocess. Boltzmann is
//! the first discrete-distribution scipy diff in this session;
//! it skips ppf (the DiscreteDistribution trait doesn't expose
//! it). Skips cleanly if scipy is unavailable — but per
//! [frankenscipy-v10ie] that means rch workers silently no-op;
//! the canonical pin against drift is the existing anchor tests
//! + a forthcoming golden artifact.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{Boltzmann, DiscreteDistribution};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const PMF_TOL: f64 = 1.0e-12;
const CDF_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    lambda: f64,
    n: u32,
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
    fs::create_dir_all(output_dir()).expect("create boltzmann diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize boltzmann diff log");
    fs::write(path, json).expect("write boltzmann diff log");
}

fn generate_query() -> OracleQuery {
    let lambdas = [0.1_f64, 0.5, 1.0, 2.0, 5.0];
    let ns = [3u32, 5, 10, 20];
    let mut points = Vec::new();
    for &lambda in &lambdas {
        for &n in &ns {
            for k in 0..n as u64 {
                points.push(PointCase {
                    case_id: format!("l{lambda}_n{n}_k{k}"),
                    lambda,
                    n,
                    k,
                });
            }
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
from scipy.stats import boltzmann

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    lam = float(case["lambda"]); n = int(case["n"]); k = int(case["k"])
    try:
        points.append({
            "case_id": cid,
            "pmf": float(boltzmann.pmf(k, lam, n)),
            "cdf": float(boltzmann.cdf(k, lam, n)),
        })
    except Exception:
        points.append({"case_id": cid, "pmf": None, "cdf": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize boltzmann query");
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
                "failed to spawn python3 for boltzmann oracle: {e}"
            );
            eprintln!("skipping boltzmann oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open boltzmann oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "boltzmann oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping boltzmann oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for boltzmann oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "boltzmann oracle failed: {stderr}"
        );
        eprintln!("skipping boltzmann oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse boltzmann oracle JSON"))
}

#[test]
fn diff_stats_boltzmann() {
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
        let dist = Boltzmann::new(case.lambda, case.n);
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
        test_id: "diff_stats_boltzmann".into(),
        category: "scipy.stats.boltzmann".into(),
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
                "boltzmann {} mismatch: {} abs_diff={}",
                d.family, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats.boltzmann conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

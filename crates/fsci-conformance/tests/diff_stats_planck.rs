#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.planck`.
//!
//! Resolves [frankenscipy-825n7]. The Planck port shipped in
//! 8c22ead has 3 anchor tests but no live scipy oracle. This
//! harness drives 5 λ-values × 30 k-values through
//! scipy.stats.planck via subprocess. Discrete distribution —
//! pmf/cdf only (no ppf in DiscreteDistribution trait).
//!
//! Skips cleanly if scipy is unavailable — but per
//! [frankenscipy-v10ie] that means rch workers silently no-op;
//! the canonical pin against drift is the existing anchor tests
//! + a forthcoming golden artifact.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_stats::{DiscreteDistribution, Planck};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const PMF_TOL: f64 = 1.0e-12;
const CDF_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    lambda: f64,
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
    compared: BTreeMap<String, ArmCounts>,
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
    fs::create_dir_all(output_dir()).expect("create planck diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize planck diff log");
    fs::write(path, json).expect("write planck diff log");
}

fn generate_query() -> OracleQuery {
    let lambdas = [0.1_f64, 0.5, 1.0, 2.0, 5.0];
    let mut points = Vec::new();
    for &lambda in &lambdas {
        for k in 0..30u64 {
            points.push(PointCase {
                case_id: format!("l{lambda}_k{k}"),
                lambda,
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
from scipy.stats import planck

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    lam = float(case["lambda"]); k = int(case["k"])
    try:
        points.append({
            "case_id": cid,
            "pmf": float(planck.pmf(k, lam)),
            "cdf": float(planck.cdf(k, lam)),
        })
    except Exception:
        points.append({"case_id": cid, "pmf": None, "cdf": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize planck query");
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
                "failed to spawn python3 for planck oracle: {e}"
            );
            eprintln!("skipping planck oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open planck oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "planck oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping planck oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for planck oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "planck oracle failed: {stderr}"
        );
        eprintln!("skipping planck oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse planck oracle JSON"))
}

#[test]
fn diff_stats_planck() {
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
    let mut ledger = CompareLedger::new("diff_stats_planck", &["pmf", "cdf"]);

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        let dist = Planck::new(case.lambda);
        let arms = [
            ("pmf", oracle.pmf, dist.pmf(case.k as i64), PMF_TOL),
            ("cdf", oracle.cdf, dist.cdf(case.k as i64), CDF_TOL),
        ];
        for (family, scipy, fsci, tol) in arms {
            let Some((s, f)) = ledger.pair(family, &case.case_id, scipy, Some(fsci)) else {
                continue;
            };
            let d = (f - s).abs();
            max_overall = max_overall.max(d);
            ledger.compared(family, &case.case_id, d <= tol);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: family.into(),
                abs_diff: d,
                pass: d <= tol,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_planck".into(),
        category: "scipy.stats.planck".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
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
                "planck {} mismatch: {} abs_diff={}",
                d.family, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats.planck conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    ledger.finish(query.points.len());
}

#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.randint`.
//!
//! Resolves [frankenscipy-xr01v]. Discrete uniform has no
//! dedicated scipy diff harness yet. 6 (low, high) ranges ×
//! k-grid for pmf/cdf plus 6 × 7 ppf cases via subprocess.
//!
//! pmf is closed-form 1/(high-low); cdf is closed-form
//! (k-low+1)/(high-low); ppf returns the integer at the q-quantile.
//! All are exact arithmetic on small ints, so 1e-14 abs holds.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::RandInt;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const POINT_TOL: f64 = 1.0e-14;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    low: i64,
    high: i64,
    k: i64,
}

#[derive(Debug, Clone, Serialize)]
struct PpfCase {
    case_id: String,
    low: i64,
    high: i64,
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
    pmf: Option<f64>,
    cdf: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create randint diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize randint diff log");
    fs::write(path, json).expect("write randint diff log");
}

fn generate_query() -> OracleQuery {
    // (low, high) pairs spanning small/medium widths and including
    // negative-low and asymmetric ranges. Sample k both inside and
    // immediately outside support so the boundary clamp logic is
    // exercised.
    let ranges: [(i64, i64); 6] = [(0, 5), (-5, 5), (1, 10), (-10, 0), (0, 100), (50, 60)];
    let qs = [0.0_f64, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0];
    let mut points = Vec::new();
    for &(low, high) in &ranges {
        let width = high - low;
        // 21 k-values spanning [low-1, high] inclusive so we hit
        // both out-of-support boundaries.
        let span = (width + 2) as f64;
        for i in 0..21u64 {
            let frac = (i as f64) / 20.0;
            let k = (low - 1) + ((frac * span).round() as i64).min(width + 1);
            points.push(PointCase {
                case_id: format!("L{low}_H{high}_k{k}_i{i}"),
                low,
                high,
                k,
            });
        }
    }
    let mut ppf_cases = Vec::new();
    for &(low, high) in &ranges {
        for &q in &qs {
            ppf_cases.push(PpfCase {
                case_id: format!("L{low}_H{high}_q{q}"),
                low,
                high,
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
from scipy.stats import randint

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    low = int(case["low"]); high = int(case["high"]); k = int(case["k"])
    try:
        points.append({
            "case_id": cid,
            "pmf": float(randint.pmf(k, low, high)),
            "cdf": float(randint.cdf(k, low, high)),
        })
    except Exception:
        points.append({"case_id": cid, "pmf": None, "cdf": None})
ppf = []
for case in q["ppf"]:
    cid = case["case_id"]; low = int(case["low"]); high = int(case["high"]); qv = float(case["q"])
    try:
        ppf.append({"case_id": cid, "ppf": float(randint.ppf(qv, low, high))})
    except Exception:
        ppf.append({"case_id": cid, "ppf": None})
print(json.dumps({"points": points, "ppf": ppf}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize randint query");
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
                "failed to spawn python3 for randint oracle: {e}"
            );
            eprintln!("skipping randint oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open randint oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "randint oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping randint oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for randint oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "randint oracle failed: {stderr}"
        );
        eprintln!("skipping randint oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse randint oracle JSON"))
}

#[test]
fn diff_stats_randint() {
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
        let dist = RandInt::new(case.low, case.high);
        if let Some(spmf) = oracle.pmf {
            let d = (dist.pmf(case.k) - spmf).abs();
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: "pmf".into(),
                abs_diff: d,
                pass: d <= POINT_TOL,
            });
        }
        if let Some(scdf) = oracle.cdf {
            let d = (dist.cdf(case.k) - scdf).abs();
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: "cdf".into(),
                abs_diff: d,
                pass: d <= POINT_TOL,
            });
        }
    }

    for case in &query.ppf {
        let oracle = ppfmap.get(&case.case_id).expect("validated oracle");
        if let Some(sppf) = oracle.ppf {
            let dist = RandInt::new(case.low, case.high);
            let rust = dist.ppf(case.q);
            // Both scipy and fsci return integer-valued floats.
            // scipy may return ±inf at q-extremes for unbounded
            // edge cases — guard with a finite check.
            if !sppf.is_finite() || !rust.is_finite() {
                let pass = sppf == rust;
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    family: "ppf".into(),
                    abs_diff: if pass { 0.0 } else { f64::INFINITY },
                    pass,
                });
                continue;
            }
            let d = (rust - sppf).abs();
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: "ppf".into(),
                abs_diff: d,
                pass: d <= POINT_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_randint".into(),
        category: "scipy.stats.randint".into(),
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
                "randint {} mismatch: {} abs_diff={}",
                d.family, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats.randint conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

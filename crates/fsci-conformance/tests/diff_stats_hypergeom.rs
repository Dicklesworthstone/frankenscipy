#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.hypergeom`.
//!
//! Resolves [frankenscipy-0a20u]. Hypergeometric has anchor
//! tests in `fsci-stats/src/lib.rs` but no dedicated scipy diff
//! harness. 6 (M, n, N) tuples × support-walking k-grid × 2
//! families (pmf, cdf) via subprocess.
//!
//! pmf is exp(Σ ± lgamma(...)) so 1e-12 abs holds. cdf inherits
//! the trait default sum-of-pmf; intermediate-k cancellation
//! absorbs into 1e-11 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{DiscreteDistribution, Hypergeometric};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const PMF_TOL: f64 = 1.0e-12;
const CDF_TOL: f64 = 1.0e-11;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    big_m: u64,
    n: u64,
    big_n: u64,
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
    fs::create_dir_all(output_dir()).expect("create hypergeom diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize hypergeom diff log");
    fs::write(path, json).expect("write hypergeom diff log");
}

fn generate_query() -> OracleQuery {
    // (M, n, N) triples spanning small/medium/large pop, varied
    // success-state and draw counts, including a "balanced"
    // case (n=N=M/2) and an extreme-skew case (n>>N or N>>n).
    let triples: [(u64, u64, u64); 6] = [
        (20, 7, 12),
        (50, 5, 10),
        (50, 25, 25),
        (100, 30, 20),
        (200, 10, 50),
        (200, 100, 100),
    ];
    let mut points = Vec::new();
    for &(big_m, n, big_n) in &triples {
        // Support: [max(0, N+n-M), min(n, N)]
        let k_min = (big_n + n).saturating_sub(big_m);
        let k_max = n.min(big_n);
        let span = k_max - k_min;
        // Walk the entire support span if it fits in 21 steps,
        // otherwise sample 21 quantile-ish steps.
        let n_steps = (span + 1).min(21);
        for i in 0..n_steps {
            let k = if span < 21 {
                k_min + i
            } else {
                k_min + ((i as f64) * (span as f64) / 20.0).round() as u64
            };
            points.push(PointCase {
                case_id: format!("M{big_m}_n{n}_N{big_n}_k{k}_i{i}"),
                big_m,
                n,
                big_n,
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
from scipy.stats import hypergeom

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    big_m = int(case["big_m"]); n = int(case["n"])
    big_n = int(case["big_n"]); k = int(case["k"])
    try:
        points.append({
            "case_id": cid,
            "pmf": float(hypergeom.pmf(k, big_m, n, big_n)),
            "cdf": float(hypergeom.cdf(k, big_m, n, big_n)),
        })
    except Exception:
        points.append({"case_id": cid, "pmf": None, "cdf": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize hypergeom query");
    let mut child = match Command::new("python3")
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
                "failed to spawn python3 for hypergeom oracle: {e}"
            );
            eprintln!("skipping hypergeom oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open hypergeom oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "hypergeom oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping hypergeom oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for hypergeom oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "hypergeom oracle failed: {stderr}"
        );
        eprintln!("skipping hypergeom oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse hypergeom oracle JSON"))
}

#[test]
fn diff_stats_hypergeom() {
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
        let dist = Hypergeometric::new(case.big_m, case.n, case.big_n);
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
        test_id: "diff_stats_hypergeom".into(),
        category: "scipy.stats.hypergeom".into(),
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
                "hypergeom {} mismatch: {} abs_diff={}",
                d.family, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats.hypergeom conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

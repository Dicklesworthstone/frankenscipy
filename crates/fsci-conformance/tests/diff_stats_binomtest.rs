#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the two-sided binomial
//! test `scipy.stats.binomtest(k, n, p).pvalue`.
//!
//! Resolves [frankenscipy-jlx5g]. Cross-checks fsci's
//! `binomtest(k, n, p)` (which returns just the two-sided
//! p-value as f64) against scipy's BinomTestResult.pvalue.
//!
//! 12 (k, n, p) cases via subprocess covering:
//!   - small n with k near and far from np
//!   - p near 0.5 (symmetric tails) and skewed p
//!   - boundary cases (k=0, k=n)
//!
//! Tol 1e-12 abs — both implementations sum
//! `P(X=j) for j in 0..=n where P(X=j) <= P(X=k)`, which is
//! a closed-form rational arithmetic chain.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::binomtest;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    k: u64,
    n: u64,
    p: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    pvalue: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create binomtest diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize binomtest diff log");
    fs::write(path, json).expect("write binomtest diff log");
}

fn generate_query() -> OracleQuery {
    let cases: &[(u64, u64, f64)] = &[
        // Small n, p=0.5 — symmetric tails
        (3, 10, 0.5),
        (5, 10, 0.5),
        (7, 10, 0.5),
        // Small n, skewed p
        (3, 10, 0.3),
        (8, 10, 0.7),
        // Larger n
        (45, 100, 0.5),
        (60, 100, 0.5),
        (15, 100, 0.2),
        (85, 100, 0.8),
        // Boundaries
        (0, 10, 0.5),
        (10, 10, 0.5),
        // Very small p, k > 0 (extreme upper tail)
        (3, 50, 0.05),
    ];

    let points = cases
        .iter()
        .enumerate()
        .map(|(i, &(k, n, p))| PointCase {
            case_id: format!("k{k}_n{n}_p{p}_i{i}"),
            k,
            n,
            p,
        })
        .collect();
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy import stats

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    k = int(case["k"]); n = int(case["n"]); p = float(case["p"])
    try:
        res = stats.binomtest(k, n, p, alternative='two-sided')
        points.append({"case_id": cid, "pvalue": fnone(res.pvalue)})
    except Exception:
        points.append({"case_id": cid, "pvalue": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize binomtest query");
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
                "failed to spawn python3 for binomtest oracle: {e}"
            );
            eprintln!("skipping binomtest oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open binomtest oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "binomtest oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping binomtest oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for binomtest oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "binomtest oracle failed: {stderr}"
        );
        eprintln!("skipping binomtest oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse binomtest oracle JSON"))
}

#[test]
fn diff_stats_binomtest() {
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
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        if let Some(scipy_v) = scipy_arm.pvalue {
            // binomtest now returns Result: scipy validates k/n/p and raises,
            // so we do too (frankenscipy-xb7so). We are inside `if let Some(scipy_v)`,
            // meaning the ORACLE produced a p-value for this case -- so a refusal
            // here is a genuine divergence, not a case to skip. Skipping it would
            // be the silent-empty-column failure: a green pass over a case that
            // tested nothing.
            let rust_v = binomtest(case.k, case.n, case.p).unwrap_or_else(|e| {
                panic!(
                    "case {}: scipy returned pvalue {scipy_v} but binomtest refused (k={}, n={}, p={}): {e:?}",
                    case.case_id, case.k, case.n, case.p
                )
            });
            if rust_v.is_finite() {
                let abs_diff = (rust_v - scipy_v).abs();
                max_overall = max_overall.max(abs_diff);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    abs_diff,
                    pass: abs_diff <= ABS_TOL,
                });
            }
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_binomtest".into(),
        category: "scipy.stats.binomtest(k,n,p).pvalue".into(),
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
            eprintln!("binomtest mismatch: {} abs={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "binomtest conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}

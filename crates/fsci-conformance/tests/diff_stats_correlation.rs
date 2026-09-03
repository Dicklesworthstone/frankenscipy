#![forbid(unsafe_code)]
//! Live SciPy differential coverage for correlation functions.
//!
//! Tests FrankenSciPy correlation functions against SciPy subprocess oracle
//! across deterministic input families.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{kendalltau, linregress, pearsonr, spearmanr};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const TOL: f64 = 1.0e-9;
const PVAL_TOL_KENDALLTAU: f64 = 0.05;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct CorrelationCase {
    case_id: String,
    func: String,
    x: Vec<f64>,
    y: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    case_id: String,
    statistic: f64,
    pvalue: f64,
    slope: Option<f64>,
    intercept: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    func: String,
    rust_stat: f64,
    scipy_stat: f64,
    rust_pval: f64,
    scipy_pval: f64,
    stat_diff: f64,
    pval_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_stat_diff: f64,
    max_pval_diff: f64,
    tolerance: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create correlation diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize correlation diff log");
    fs::write(path, json).expect("write correlation diff log");
}

fn deterministic_xy(n: usize, seed: usize, correlation: f64) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..n)
        .map(|idx| {
            let base = ((idx + seed) % 7) as f64 * 0.5 + 0.1;
            let wave = (((idx * 3 + seed) % 11) as f64 * 0.2) - 0.5;
            base + wave
        })
        .collect();

    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(idx, &xi)| {
            let noise = (((idx * 7 + seed * 3) % 13) as f64 * 0.1) - 0.5;
            correlation * xi + (1.0 - correlation.abs()) * noise
        })
        .collect();

    (x, y)
}

fn deterministic_uncorrelated(n: usize, seed: usize) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..n)
        .map(|idx| ((idx + seed) % 17) as f64 * 0.3 - 2.0)
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|idx| ((idx * 7 + seed * 5) % 19) as f64 * 0.25 - 2.0)
        .collect();
    (x, y)
}

fn deterministic_monotonic(n: usize, seed: usize, increasing: bool) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..n)
        .map(|idx| idx as f64 + (seed % 5) as f64 * 0.1)
        .collect();
    let y: Vec<f64> = if increasing {
        (0..n)
            .map(|idx| idx as f64 * 2.0 + ((idx + seed) % 3) as f64 * 0.05)
            .collect()
    } else {
        (0..n)
            .map(|idx| (n - idx) as f64 * 2.0 + ((idx + seed) % 3) as f64 * 0.05)
            .collect()
    };
    (x, y)
}

fn generate_correlation_cases() -> Vec<CorrelationCase> {
    let mut cases = Vec::new();
    let funcs = ["pearsonr", "spearmanr", "kendalltau", "linregress"];
    let sizes = [10, 20, 50, 100];

    for &func in &funcs {
        for (size_idx, &n) in sizes.iter().enumerate() {
            for corr_idx in 0..3 {
                let correlation = match corr_idx {
                    0 => 0.9,
                    1 => 0.5,
                    _ => -0.7,
                };
                let seed = size_idx * 10 + corr_idx;
                let (x, y) = deterministic_xy(n, seed, correlation);
                cases.push(CorrelationCase {
                    case_id: format!("{func}_n{n}_corr{corr_idx}_seed{seed}"),
                    func: func.into(),
                    x,
                    y,
                });
            }

            let seed = size_idx * 10 + 100;
            let (x, y) = deterministic_uncorrelated(n, seed);
            cases.push(CorrelationCase {
                case_id: format!("{func}_n{n}_uncorr_seed{seed}"),
                func: func.into(),
                x,
                y,
            });

            let seed = size_idx * 10 + 200;
            let (x, y) = deterministic_monotonic(n, seed, true);
            cases.push(CorrelationCase {
                case_id: format!("{func}_n{n}_mono_inc_seed{seed}"),
                func: func.into(),
                x,
                y,
            });

            let seed = size_idx * 10 + 300;
            let (x, y) = deterministic_monotonic(n, seed, false);
            cases.push(CorrelationCase {
                case_id: format!("{func}_n{n}_mono_dec_seed{seed}"),
                func: func.into(),
                x,
                y,
            });
        }
    }

    cases
}

fn scipy_correlation_oracle_or_skip(cases: &[CorrelationCase]) -> Vec<OracleResult> {
    let script = r#"
import json
import sys
import numpy as np
from scipy import stats

cases = json.load(sys.stdin)
results = []

for c in cases:
    cid = c["case_id"]
    func = c["func"]
    x = np.array(c["x"], dtype=np.float64)
    y = np.array(c["y"], dtype=np.float64)

    try:
        if func == "pearsonr":
            res = stats.pearsonr(x, y)
            results.append({
                "case_id": cid,
                "statistic": float(res.statistic),
                "pvalue": float(res.pvalue),
                "slope": None,
                "intercept": None
            })
        elif func == "spearmanr":
            res = stats.spearmanr(x, y)
            results.append({
                "case_id": cid,
                "statistic": float(res.statistic),
                "pvalue": float(res.pvalue),
                "slope": None,
                "intercept": None
            })
        elif func == "kendalltau":
            res = stats.kendalltau(x, y)
            results.append({
                "case_id": cid,
                "statistic": float(res.statistic),
                "pvalue": float(res.pvalue),
                "slope": None,
                "intercept": None
            })
        elif func == "linregress":
            res = stats.linregress(x, y)
            results.append({
                "case_id": cid,
                "statistic": float(res.rvalue),
                "pvalue": float(res.pvalue),
                "slope": float(res.slope),
                "intercept": float(res.intercept)
            })
        else:
            results.append({
                "case_id": cid,
                "statistic": float("nan"),
                "pvalue": float("nan"),
                "slope": None,
                "intercept": None
            })
    except Exception as e:
        results.append({
            "case_id": cid,
            "statistic": float("nan"),
            "pvalue": float("nan"),
            "slope": None,
            "intercept": None
        })

print(json.dumps(results))
"#;

    let cases_json = serde_json::to_string(cases).expect("serialize correlation cases");

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
                "failed to spawn python3 for correlation oracle: {e}"
            );
            eprintln!("skipping correlation oracle: python3 not available ({e})");
            return Vec::new();
        }
    };

    {
        let stdin = child.stdin.as_mut().expect("open correlation oracle stdin");
        if let Err(err) = stdin.write_all(cases_json.as_bytes()) {
            let output = child
                .wait_with_output()
                .expect("wait for failed correlation oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "correlation oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping correlation oracle: stdin write failed ({err})\n{stderr}");
            return Vec::new();
        }
    }

    let output = child
        .wait_with_output()
        .expect("wait for correlation oracle");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "correlation oracle failed: {stderr}"
        );
        eprintln!("skipping correlation oracle: scipy not available\n{stderr}");
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout).expect("parse correlation oracle JSON")
}

fn compute_rust_correlation(
    case: &CorrelationCase,
) -> Option<(f64, f64, Option<f64>, Option<f64>)> {
    match case.func.as_str() {
        "pearsonr" => {
            let res = pearsonr(&case.x, &case.y);
            Some((res.statistic, res.pvalue, None, None))
        }
        "spearmanr" => {
            let res = spearmanr(&case.x, &case.y);
            Some((res.statistic, res.pvalue, None, None))
        }
        "kendalltau" => {
            let res = kendalltau(&case.x, &case.y);
            Some((res.statistic, res.pvalue, None, None))
        }
        "linregress" => {
            let res = linregress(&case.x, &case.y);
            Some((res.rvalue, res.pvalue, Some(res.slope), Some(res.intercept)))
        }
        _ => None,
    }
}

#[test]
fn diff_stats_correlation() {
    let cases = generate_correlation_cases();
    let oracle_results = scipy_correlation_oracle_or_skip(&cases);

    if oracle_results.is_empty() {
        return;
    }

    assert_eq!(
        oracle_results.len(),
        cases.len(),
        "SciPy correlation oracle returned partial coverage"
    );

    let oracle_map: HashMap<String, OracleResult> = oracle_results
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();
    assert_eq!(
        oracle_map.len(),
        cases.len(),
        "SciPy correlation oracle returned duplicate case IDs"
    );

    let missing_oracle_cases: Vec<&str> = cases
        .iter()
        .filter(|case| !oracle_map.contains_key(&case.case_id))
        .map(|case| case.case_id.as_str())
        .collect();
    assert!(
        missing_oracle_cases.is_empty(),
        "SciPy correlation oracle missing cases: {:?}",
        missing_oracle_cases
    );

    let unexpected_oracle_cases: Vec<&str> = oracle_map
        .keys()
        .map(String::as_str)
        .filter(|case_id| !cases.iter().any(|case| case.case_id == *case_id))
        .collect();
    assert!(
        unexpected_oracle_cases.is_empty(),
        "SciPy correlation oracle returned unexpected cases: {:?}",
        unexpected_oracle_cases
    );

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_stat_diff = 0.0_f64;
    let mut max_pval_diff = 0.0_f64;

    for case in &cases {
        let rust_result = compute_rust_correlation(case);
        assert!(
            rust_result.is_some(),
            "unsupported correlation function {}",
            case.func
        );
        let Some((rust_stat, rust_pval, rust_slope, rust_intercept)) = rust_result else {
            continue;
        };
        let scipy_result = oracle_map
            .get(&case.case_id)
            .expect("validated complete correlation oracle map");

        let scipy_stat = scipy_result.statistic;
        let scipy_pval = scipy_result.pvalue;

        let stat_diff = (rust_stat - scipy_stat).abs();
        let pval_diff = (rust_pval - scipy_pval).abs();

        let stat_rel_scale = rust_stat.abs().max(scipy_stat.abs()).max(1.0);
        let pval_rel_scale = rust_pval.abs().max(scipy_pval.abs()).max(1e-10);

        let stat_tol = TOL * stat_rel_scale;
        let pval_tol = if case.func == "kendalltau" {
            PVAL_TOL_KENDALLTAU
        } else {
            TOL * pval_rel_scale.max(1.0)
        };

        let mut pass = stat_diff <= stat_tol && pval_diff <= pval_tol;

        if let ("linregress", Some(rs), Some(ri), Some(ss), Some(si)) = (
            case.func.as_str(),
            rust_slope,
            rust_intercept,
            scipy_result.slope,
            scipy_result.intercept,
        ) {
            let slope_diff = (rs - ss).abs();
            let intercept_diff = (ri - si).abs();
            let slope_tol = TOL * rs.abs().max(ss.abs()).max(1.0);
            let intercept_tol = TOL * ri.abs().max(si.abs()).max(1.0);
            pass = pass && slope_diff <= slope_tol && intercept_diff <= intercept_tol;
        }

        max_stat_diff = max_stat_diff.max(stat_diff);
        max_pval_diff = max_pval_diff.max(pval_diff);

        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.clone(),
            rust_stat,
            scipy_stat,
            rust_pval,
            scipy_pval,
            stat_diff,
            pval_diff,
            pass,
        });
    }
    assert_eq!(
        diffs.len(),
        cases.len(),
        "compared {}/{} expected correlation cases",
        diffs.len(),
        cases.len()
    );

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_correlation".into(),
        category: "scipy.stats.correlation".into(),
        case_count: diffs.len(),
        max_stat_diff,
        max_pval_diff,
        tolerance: TOL,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for diff in &diffs {
        if !diff.pass {
            eprintln!(
                "{} mismatch: rust_stat={} scipy_stat={} stat_diff={} rust_pval={} scipy_pval={} pval_diff={}",
                diff.case_id,
                diff.rust_stat,
                diff.scipy_stat,
                diff.stat_diff,
                diff.rust_pval,
                diff.scipy_pval,
                diff.pval_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats correlation conformance failed: {} cases, max_stat_diff={}, max_pval_diff={}",
        diffs.len(),
        max_stat_diff,
        max_pval_diff
    );
}

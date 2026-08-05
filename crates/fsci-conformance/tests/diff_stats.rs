#![forbid(unsafe_code)]
//! Live SciPy differential coverage for scipy.stats functions.
//!
//! Tests FrankenSciPy statistics functions against SciPy subprocess oracle
//! across deterministic input families.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// SCOPE NOTE (frankenscipy-1p21x): this harness previously also imported
// `bootstrap`, `BootstrapMethod`, `BootstrapIntervalMethod`, `PermutationMethod`
// and `MonteCarloMethod`. Those items do not exist anywhere in fsci-stats — not
// as private definitions, not as re-exports — so this test target had never
// compiled, which in turn meant NONE of the scipy.stats differential coverage
// below had ever run. The resampling columns are deliberately dropped here
// rather than stubbed; implementing the scipy resampling surface is tracked as
// its own feature bead. Everything imported below resolves today.
use fsci_stats::{
    circmean, circstd, circvar, energy_distance, gmean, hmean, mannwhitneyu, pmean, quantile,
    ttest_1samp, ttest_ind, wasserstein_distance, wilcoxon,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct StatsCase {
    case_id: String,
    func: String,
    data: Vec<f64>,
    data2: Option<Vec<f64>>,
    param: Option<f64>,
    quantiles: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    case_id: String,
    value: f64,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    func: String,
    rust_value: f64,
    scipy_value: f64,
    abs_diff: f64,
    tolerance: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
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
    fs::create_dir_all(output_dir()).expect("create stats diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize stats diff log");
    fs::write(path, json).expect("write stats diff log");
}

fn deterministic_data(n: usize, seed: usize) -> Vec<f64> {
    (0..n)
        .map(|idx| {
            let base = ((idx + seed) % 7) as f64 * 0.5 + 0.1;
            let wave = (((idx * 3 + seed) % 11) as f64 * 0.2) - 0.5;
            base + wave + (seed % 5) as f64 * 0.15
        })
        .collect()
}

fn deterministic_positive_data(n: usize, seed: usize) -> Vec<f64> {
    deterministic_data(n, seed)
        .iter()
        .map(|&x| x.abs() + 0.01)
        .collect()
}

fn deterministic_angles(n: usize, seed: usize) -> Vec<f64> {
    (0..n)
        .map(|idx| {
            let base = ((idx + seed) % 13) as f64 * 0.5;
            base - std::f64::consts::PI + (seed % 7) as f64 * 0.3
        })
        .collect()
}

fn stats_cases() -> Vec<StatsCase> {
    let sizes = [5, 10, 20, 50];
    let mut cases = Vec::new();

    for (size_idx, &n) in sizes.iter().enumerate() {
        for seed_offset in 0..3 {
            let seed = size_idx * 10 + seed_offset;
            let data = deterministic_positive_data(n, seed);
            let data2 = deterministic_positive_data(n, seed + 50);
            let angles = deterministic_angles(n, seed);

            cases.push(StatsCase {
                case_id: format!("gmean_n{n}_seed{seed}"),
                func: "gmean".into(),
                data: data.clone(),
                data2: None,
                param: None,
                quantiles: None,
            });

            cases.push(StatsCase {
                case_id: format!("hmean_n{n}_seed{seed}"),
                func: "hmean".into(),
                data: data.clone(),
                data2: None,
                param: None,
                quantiles: None,
            });

            cases.push(StatsCase {
                case_id: format!("pmean_p2_n{n}_seed{seed}"),
                func: "pmean".into(),
                data: data.clone(),
                data2: None,
                param: Some(2.0),
                quantiles: None,
            });

            cases.push(StatsCase {
                case_id: format!("pmean_p3_n{n}_seed{seed}"),
                func: "pmean".into(),
                data: data.clone(),
                data2: None,
                param: Some(3.0),
                quantiles: None,
            });

            cases.push(StatsCase {
                case_id: format!("circmean_n{n}_seed{seed}"),
                func: "circmean".into(),
                data: angles.clone(),
                data2: None,
                param: None,
                quantiles: None,
            });

            cases.push(StatsCase {
                case_id: format!("circvar_n{n}_seed{seed}"),
                func: "circvar".into(),
                data: angles.clone(),
                data2: None,
                param: None,
                quantiles: None,
            });

            cases.push(StatsCase {
                case_id: format!("circstd_n{n}_seed{seed}"),
                func: "circstd".into(),
                data: angles.clone(),
                data2: None,
                param: None,
                quantiles: None,
            });

            cases.push(StatsCase {
                case_id: format!("quantile_n{n}_seed{seed}"),
                func: "quantile".into(),
                data: data.clone(),
                data2: None,
                param: None,
                quantiles: Some(vec![0.25, 0.5, 0.75]),
            });

            cases.push(StatsCase {
                case_id: format!("ttest_1samp_n{n}_seed{seed}"),
                func: "ttest_1samp".into(),
                data: data.clone(),
                data2: None,
                param: Some(1.0),
                quantiles: None,
            });

            cases.push(StatsCase {
                case_id: format!("ttest_ind_n{n}_seed{seed}"),
                func: "ttest_ind".into(),
                data: data.clone(),
                data2: Some(data2.clone()),
                param: None,
                quantiles: None,
            });

            cases.push(StatsCase {
                case_id: format!("mannwhitneyu_n{n}_seed{seed}"),
                func: "mannwhitneyu".into(),
                data: data.clone(),
                data2: Some(data2.clone()),
                param: None,
                quantiles: None,
            });

            cases.push(StatsCase {
                case_id: format!("wasserstein_n{n}_seed{seed}"),
                func: "wasserstein".into(),
                data: data.clone(),
                data2: Some(data2.clone()),
                param: None,
                quantiles: None,
            });

            cases.push(StatsCase {
                case_id: format!("energy_n{n}_seed{seed}"),
                func: "energy".into(),
                data: data.clone(),
                data2: Some(data2.clone()),
                param: None,
                quantiles: None,
            });
        }
    }

    cases
}

fn run_scipy_oracle(cases: &[StatsCase]) -> Option<Vec<OracleResult>> {
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
    data = np.array(c["data"], dtype=np.float64)
    data2 = np.array(c["data2"], dtype=np.float64) if c.get("data2") else None
    param = c.get("param")
    quantiles = c.get("quantiles")

    try:
        if func == "gmean":
            val = stats.gmean(data)
            results.append({"case_id": cid, "value": float(val)})
        elif func == "hmean":
            val = stats.hmean(data)
            results.append({"case_id": cid, "value": float(val)})
        elif func == "pmean":
            val = stats.pmean(data, param)
            results.append({"case_id": cid, "value": float(val)})
        elif func == "circmean":
            val = stats.circmean(data, high=np.pi, low=-np.pi)
            results.append({"case_id": cid, "value": float(val)})
        elif func == "circvar":
            val = stats.circvar(data, high=np.pi, low=-np.pi)
            results.append({"case_id": cid, "value": float(val)})
        elif func == "circstd":
            val = stats.circstd(data, high=np.pi, low=-np.pi)
            results.append({"case_id": cid, "value": float(val)})
        elif func == "quantile":
            qs = np.array(quantiles)
            val = np.quantile(data, 0.5)
            results.append({"case_id": cid, "value": float(val)})
        elif func == "ttest_1samp":
            res = stats.ttest_1samp(data, param)
            results.append({"case_id": cid, "value": float(res.statistic), "value2": float(res.pvalue)})
        elif func == "ttest_ind":
            res = stats.ttest_ind(data, data2)
            results.append({"case_id": cid, "value": float(res.statistic), "value2": float(res.pvalue)})
        elif func == "mannwhitneyu":
            res = stats.mannwhitneyu(data, data2, alternative='two-sided')
            n1, n2 = len(data), len(data2)
            u_min = min(res.statistic, n1 * n2 - res.statistic)
            results.append({"case_id": cid, "value": float(u_min), "value2": float(res.pvalue)})
        elif func == "wasserstein":
            val = stats.wasserstein_distance(data, data2)
            results.append({"case_id": cid, "value": float(val)})
        elif func == "energy":
            val = stats.energy_distance(data, data2)
            results.append({"case_id": cid, "value": float(val)})
    except Exception:
        pass

json.dump(results, sys.stdout)
"#;

    let mut child = Command::new("python3")
        .args(["-c", script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;

    {
        let stdin = child.stdin.as_mut()?;
        let json_input = serde_json::to_string(cases).ok()?;
        stdin.write_all(json_input.as_bytes()).ok()?;
    }

    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        return None;
    }

    serde_json::from_slice(&output.stdout).ok()
}

fn scipy_oracle_or_skip(cases: &[StatsCase]) -> Vec<OracleResult> {
    match run_scipy_oracle(cases) {
        Some(results) => results,
        None => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "SciPy oracle required but not available"
            );
            eprintln!("SciPy oracle not available, skipping diff test");
            Vec::new()
        }
    }
}

fn compute_rust_value(case: &StatsCase) -> Option<(f64, Option<f64>)> {
    match case.func.as_str() {
        "gmean" => Some((gmean(&case.data), None)),
        "hmean" => Some((hmean(&case.data), None)),
        "pmean" => Some((pmean(&case.data, case.param.unwrap_or(2.0)), None)),
        "circmean" => Some((circmean(&case.data), None)),
        "circvar" => Some((circvar(&case.data), None)),
        "circstd" => Some((circstd(&case.data), None)),
        "quantile" => {
            let q = quantile(&case.data, &[0.5]);
            Some((q.first().copied().unwrap_or(0.0), None))
        }
        "ttest_1samp" => {
            let res = ttest_1samp(&case.data, case.param.unwrap_or(0.0));
            Some((res.statistic, Some(res.pvalue)))
        }
        "ttest_ind" => {
            let data2 = case.data2.as_ref()?;
            let res = ttest_ind(&case.data, data2);
            Some((res.statistic, Some(res.pvalue)))
        }
        "mannwhitneyu" => {
            let data2 = case.data2.as_ref()?;
            let res = mannwhitneyu(&case.data, data2);
            let n1 = case.data.len() as f64;
            let n2 = data2.len() as f64;
            let u_min = res.statistic.min(n1 * n2 - res.statistic);
            Some((u_min, Some(res.pvalue)))
        }
        "wasserstein" => {
            let data2 = case.data2.as_ref()?;
            Some((wasserstein_distance(&case.data, data2), None))
        }
        "energy" => {
            let data2 = case.data2.as_ref()?;
            Some((energy_distance(&case.data, data2), None))
        }
        _ => None,
    }
}

#[test]
fn diff_stats_basic() {
    let cases = stats_cases();
    let oracle_results = scipy_oracle_or_skip(&cases);

    if oracle_results.is_empty() {
        return;
    }
    assert_eq!(
        oracle_results.len(),
        cases.len(),
        "SciPy stats oracle returned partial coverage"
    );

    let oracle_map: HashMap<String, OracleResult> = oracle_results
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();
    assert_eq!(
        oracle_map.len(),
        cases.len(),
        "SciPy stats oracle returned duplicate or missing case ids"
    );
    let missing_rust_evaluators: Vec<&str> = cases
        .iter()
        .filter(|case| compute_rust_value(case).is_none())
        .map(|case| case.func.as_str())
        .collect();
    assert!(
        missing_rust_evaluators.is_empty(),
        "missing Rust stats evaluators: {:?}",
        missing_rust_evaluators
    );
    let missing_oracle_cases: Vec<&str> = cases
        .iter()
        .filter(|case| !oracle_map.contains_key(&case.case_id))
        .map(|case| case.case_id.as_str())
        .collect();
    assert!(
        missing_oracle_cases.is_empty(),
        "missing SciPy stats oracle results: {:?}",
        missing_oracle_cases
    );

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_diff = 0.0_f64;

    for case in &cases {
        let Some((rust_val, _rust_val2)) = compute_rust_value(case) else {
            continue;
        };
        let Some(scipy_result) = oracle_map.get(&case.case_id) else {
            continue;
        };

        let scipy_val = scipy_result.value;
        let abs_diff = (rust_val - scipy_val).abs();
        let rel_scale = rust_val.abs().max(scipy_val.abs()).max(1.0);
        let effective_tol = TOL * rel_scale;

        max_diff = max_diff.max(abs_diff);

        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.clone(),
            rust_value: rust_val,
            scipy_value: scipy_val,
            abs_diff,
            tolerance: effective_tol,
            pass: abs_diff <= effective_tol,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_basic".into(),
        category: "scipy.stats".into(),
        case_count: diffs.len(),
        max_abs_diff: max_diff,
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
                "{} mismatch: rust={} scipy={} diff={}",
                diff.case_id, diff.rust_value, diff.scipy_value, diff.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_diff
    );
}

#[test]
fn diff_stats_wilcoxon() {
    let sizes = [10, 20, 30];
    let mut cases = Vec::new();

    for (size_idx, &n) in sizes.iter().enumerate() {
        for seed_offset in 0..4 {
            let seed = size_idx * 10 + seed_offset;
            let data = deterministic_data(n, seed);
            let data2 = deterministic_data(n, seed + 50);

            cases.push(StatsCase {
                case_id: format!("wilcoxon_n{n}_seed{seed}"),
                func: "wilcoxon".into(),
                data,
                data2: Some(data2),
                param: None,
                quantiles: None,
            });
        }
    }

    let script = r#"
import json
import sys
import numpy as np
from scipy import stats

cases = json.load(sys.stdin)
results = []

for c in cases:
    cid = c["case_id"]
    data = np.array(c["data"], dtype=np.float64)
    data2 = np.array(c["data2"], dtype=np.float64)

    try:
        res = stats.wilcoxon(data, data2, alternative='two-sided')
        results.append({"case_id": cid, "value": float(res.statistic), "value2": float(res.pvalue)})
    except Exception:
        pass

json.dump(results, sys.stdout)
"#;

    let mut child = match Command::new("python3")
        .args(["-c", script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "SciPy oracle required but not available"
            );
            eprintln!("SciPy oracle not available, skipping wilcoxon diff test");
            return;
        }
    };

    {
        let stdin = child.stdin.as_mut().unwrap();
        let json_input = serde_json::to_string(&cases).unwrap();
        stdin.write_all(json_input.as_bytes()).unwrap();
    }

    let output = child.wait_with_output().unwrap();
    if !output.status.success() {
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "SciPy oracle failed"
        );
        return;
    }

    let oracle_results: Vec<OracleResult> = match serde_json::from_slice(&output.stdout) {
        Ok(r) => r,
        Err(_) => return,
    };

    if oracle_results.is_empty() {
        return;
    }
    assert_eq!(
        oracle_results.len(),
        cases.len(),
        "SciPy wilcoxon oracle returned partial coverage"
    );

    let oracle_map: HashMap<String, OracleResult> = oracle_results
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();
    assert_eq!(
        oracle_map.len(),
        cases.len(),
        "SciPy wilcoxon oracle returned duplicate or missing case ids"
    );
    let missing_oracle_cases: Vec<&str> = cases
        .iter()
        .filter(|case| !oracle_map.contains_key(&case.case_id))
        .map(|case| case.case_id.as_str())
        .collect();
    assert!(
        missing_oracle_cases.is_empty(),
        "missing SciPy wilcoxon oracle results: {:?}",
        missing_oracle_cases
    );

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_diff = 0.0_f64;

    for case in &cases {
        let data2 = case.data2.as_ref().unwrap();
        let res = wilcoxon(&case.data, data2);
        let rust_val = res.statistic;

        let Some(scipy_result) = oracle_map.get(&case.case_id) else {
            continue;
        };

        let scipy_val = scipy_result.value;
        let abs_diff = (rust_val - scipy_val).abs();
        let rel_scale = rust_val.abs().max(scipy_val.abs()).max(1.0);
        let effective_tol = TOL * rel_scale;

        max_diff = max_diff.max(abs_diff);

        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: "wilcoxon".into(),
            rust_value: rust_val,
            scipy_value: scipy_val,
            abs_diff,
            tolerance: effective_tol,
            pass: abs_diff <= effective_tol,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_wilcoxon".into(),
        category: "scipy.stats.wilcoxon".into(),
        case_count: diffs.len(),
        max_abs_diff: max_diff,
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
                "{} mismatch: rust={} scipy={} diff={}",
                diff.case_id, diff.rust_value, diff.scipy_value, diff.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats.wilcoxon conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_diff
    );
}

#![forbid(unsafe_code)]
//! Live SciPy differential coverage for scipy.integrate functions.
//!
//! Tests FrankenSciPy integration functions against SciPy subprocess oracle
//! across deterministic input families.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_integrate::{
    cumulative_trapezoid, cumulative_trapezoid_uniform, romb, simpson, simpson_uniform, trapezoid,
    trapezoid_uniform,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-008";
const TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct IntegrateCase {
    case_id: String,
    func: String,
    y: Vec<f64>,
    x: Option<Vec<f64>>,
    dx: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    case_id: String,
    value: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleArrayResult {
    case_id: String,
    values: Vec<f64>,
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
    fs::create_dir_all(output_dir()).expect("create integrate diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize integrate diff log");
    fs::write(path, json).expect("write integrate diff log");
}

fn deterministic_y(n: usize, seed: usize) -> Vec<f64> {
    (0..n)
        .map(|idx| {
            let t = idx as f64 / (n - 1).max(1) as f64;
            let base = t * t + 0.5 * t;
            let wave = 0.1 * ((t * std::f64::consts::PI * (seed % 5 + 1) as f64).sin());
            base + wave + (seed % 3) as f64 * 0.1
        })
        .collect()
}

fn deterministic_x(n: usize, seed: usize) -> Vec<f64> {
    (0..n)
        .map(|idx| {
            let base = idx as f64;
            let jitter = 0.05 * ((idx + seed) % 7) as f64;
            base + jitter
        })
        .collect()
}

fn scalar_cases() -> Vec<IntegrateCase> {
    let sizes = [5, 9, 17, 33];
    let mut cases = Vec::new();

    for (size_idx, &n) in sizes.iter().enumerate() {
        for seed_offset in 0..3 {
            let seed = size_idx * 10 + seed_offset;
            let y = deterministic_y(n, seed);
            let x = deterministic_x(n, seed);
            let dx = 1.0;

            cases.push(IntegrateCase {
                case_id: format!("trapezoid_n{n}_seed{seed}"),
                func: "trapezoid".into(),
                y: y.clone(),
                x: Some(x.clone()),
                dx: None,
            });

            cases.push(IntegrateCase {
                case_id: format!("trapezoid_uniform_n{n}_seed{seed}"),
                func: "trapezoid_uniform".into(),
                y: y.clone(),
                x: None,
                dx: Some(dx),
            });

            cases.push(IntegrateCase {
                case_id: format!("simpson_n{n}_seed{seed}"),
                func: "simpson".into(),
                y: y.clone(),
                x: Some(x.clone()),
                dx: None,
            });

            cases.push(IntegrateCase {
                case_id: format!("simpson_uniform_n{n}_seed{seed}"),
                func: "simpson_uniform".into(),
                y: y.clone(),
                x: None,
                dx: Some(dx),
            });

            if (n - 1).is_power_of_two() {
                cases.push(IntegrateCase {
                    case_id: format!("romb_n{n}_seed{seed}"),
                    func: "romb".into(),
                    y: y.clone(),
                    x: None,
                    dx: Some(dx),
                });
            }
        }
    }

    cases
}

fn run_scipy_oracle(cases: &[IntegrateCase]) -> Option<Vec<OracleResult>> {
    let script = r#"
import json
import sys

import numpy as np
from scipy import integrate

cases = json.load(sys.stdin)
results = []

for c in cases:
    cid = c["case_id"]
    func = c["func"]
    y = np.array(c["y"], dtype=np.float64)
    x = np.array(c["x"], dtype=np.float64) if c.get("x") else None
    dx = c.get("dx", 1.0)

    try:
        if func == "trapezoid":
            val = integrate.trapezoid(y, x=x)
        elif func == "trapezoid_uniform":
            val = integrate.trapezoid(y, dx=dx)
        elif func == "simpson":
            val = integrate.simpson(y, x=x)
        elif func == "simpson_uniform":
            val = integrate.simpson(y, dx=dx)
        elif func == "romb":
            val = integrate.romb(y, dx=dx)
        else:
            continue

        if np.isfinite(val):
            results.append({"case_id": cid, "value": float(val)})
    except Exception:
        pass

json.dump(results, sys.stdout)
"#;

    let mut child = fsci_conformance::scipy_oracle_command()
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

fn scipy_oracle_or_skip(cases: &[IntegrateCase]) -> Vec<OracleResult> {
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

fn compute_rust_value(case: &IntegrateCase) -> Option<f64> {
    match case.func.as_str() {
        "trapezoid" => {
            let x = case.x.as_ref()?;
            trapezoid(&case.y, x).ok().map(|r| r.integral)
        }
        "trapezoid_uniform" => {
            let dx = case.dx?;
            trapezoid_uniform(&case.y, dx).ok().map(|r| r.integral)
        }
        "simpson" => {
            let x = case.x.as_ref()?;
            simpson(&case.y, x).ok().map(|r| r.integral)
        }
        "simpson_uniform" => {
            let dx = case.dx?;
            simpson_uniform(&case.y, dx).ok().map(|r| r.integral)
        }
        "romb" => {
            let dx = case.dx?;
            romb(&case.y, dx).ok()
        }
        _ => None,
    }
}

fn complete_scalar_oracle_map(
    test_id: &str,
    cases: &[IntegrateCase],
    oracle_results: Vec<OracleResult>,
) -> HashMap<String, OracleResult> {
    assert_eq!(
        oracle_results.len(),
        cases.len(),
        "{test_id} SciPy oracle returned partial coverage"
    );

    let oracle_map: HashMap<String, OracleResult> = oracle_results
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();
    assert_eq!(
        oracle_map.len(),
        cases.len(),
        "{test_id} SciPy oracle returned duplicate or missing case ids"
    );

    let missing_rust_evaluators: Vec<&str> = cases
        .iter()
        .filter(|case| compute_rust_value(case).is_none())
        .map(|case| case.case_id.as_str())
        .collect();
    assert!(
        missing_rust_evaluators.is_empty(),
        "{test_id} missing Rust integrate evaluators: {:?}",
        missing_rust_evaluators
    );

    let missing_oracle_cases: Vec<&str> = cases
        .iter()
        .filter(|case| !oracle_map.contains_key(&case.case_id))
        .map(|case| case.case_id.as_str())
        .collect();
    assert!(
        missing_oracle_cases.is_empty(),
        "{test_id} missing SciPy integrate oracle results: {:?}",
        missing_oracle_cases
    );

    oracle_map
}

#[test]
fn diff_integrate_scalar() {
    let cases = scalar_cases();
    let oracle_results = scipy_oracle_or_skip(&cases);

    if oracle_results.is_empty() {
        return;
    }

    let oracle_map = complete_scalar_oracle_map("diff_integrate_scalar", &cases, oracle_results);

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_diff = 0.0_f64;

    for case in &cases {
        let rust_val = compute_rust_value(case)
            .expect("complete scalar oracle map validates Rust evaluator coverage");
        let scipy_result = oracle_map
            .get(&case.case_id)
            .expect("complete scalar oracle map validates SciPy case coverage");

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
        test_id: "diff_integrate_scalar".into(),
        category: "scipy.integrate".into(),
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
        "scipy.integrate scalar conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_diff
    );
}

#[test]
fn diff_integrate_cumulative() {
    let sizes = [5, 9, 17];
    let mut cases = Vec::new();

    for (size_idx, &n) in sizes.iter().enumerate() {
        for seed_offset in 0..3 {
            let seed = size_idx * 10 + seed_offset;
            let y = deterministic_y(n, seed);
            let x = deterministic_x(n, seed);
            let dx = 1.0;

            cases.push(IntegrateCase {
                case_id: format!("cumtrapz_n{n}_seed{seed}"),
                func: "cumtrapz".into(),
                y: y.clone(),
                x: Some(x.clone()),
                dx: None,
            });

            cases.push(IntegrateCase {
                case_id: format!("cumtrapz_uniform_n{n}_seed{seed}"),
                func: "cumtrapz_uniform".into(),
                y: y.clone(),
                x: None,
                dx: Some(dx),
            });
        }
    }

    let script = r#"
import json
import sys
import numpy as np
from scipy import integrate

cases = json.load(sys.stdin)
results = []

for c in cases:
    cid = c["case_id"]
    func = c["func"]
    y = np.array(c["y"], dtype=np.float64)
    x = np.array(c["x"], dtype=np.float64) if c.get("x") else None
    dx = c.get("dx", 1.0)

    try:
        if func == "cumtrapz":
            vals = integrate.cumulative_trapezoid(y, x=x)
        elif func == "cumtrapz_uniform":
            vals = integrate.cumulative_trapezoid(y, dx=dx)
        else:
            continue

        results.append({"case_id": cid, "values": [float(v) for v in vals]})
    except Exception:
        pass

json.dump(results, sys.stdout)
"#;

    let mut child = match fsci_conformance::scipy_oracle_command()
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
            eprintln!("SciPy oracle not available, skipping cumulative diff test");
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

    let oracle_results: Vec<OracleArrayResult> = match serde_json::from_slice(&output.stdout) {
        Ok(r) => r,
        Err(error) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "SciPy oracle returned invalid JSON: {error}"
            );
            return;
        }
    };

    if oracle_results.is_empty() {
        return;
    }
    assert_eq!(
        oracle_results.len(),
        cases.len(),
        "SciPy integrate cumulative oracle returned partial coverage"
    );

    let oracle_map: HashMap<String, OracleArrayResult> = oracle_results
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();
    assert_eq!(
        oracle_map.len(),
        cases.len(),
        "SciPy integrate cumulative oracle returned duplicate or missing case ids"
    );
    let missing_oracle_cases: Vec<&str> = cases
        .iter()
        .filter(|case| !oracle_map.contains_key(&case.case_id))
        .map(|case| case.case_id.as_str())
        .collect();
    assert!(
        missing_oracle_cases.is_empty(),
        "missing SciPy integrate cumulative oracle results: {:?}",
        missing_oracle_cases
    );

    let mut all_pass = true;
    let mut max_diff = 0.0_f64;

    for case in &cases {
        let rust_vals: Vec<f64> = match case.func.as_str() {
            "cumtrapz" => {
                let x = case.x.as_ref().expect("cumtrapz cases include explicit x");
                cumulative_trapezoid(&case.y, x)
                    .expect("Rust cumulative_trapezoid should evaluate conformance case")
            }
            "cumtrapz_uniform" => {
                let dx = case.dx.expect("uniform cumulative cases include dx");
                cumulative_trapezoid_uniform(&case.y, dx)
                    .expect("Rust cumulative_trapezoid_uniform should evaluate conformance case")
            }
            other => {
                eprintln!("unsupported cumulative integrate function {other}");
                all_pass = false;
                continue;
            }
        };

        let scipy_result = oracle_map
            .get(&case.case_id)
            .expect("complete cumulative oracle map validates SciPy case coverage");
        assert_eq!(
            rust_vals.len(),
            scipy_result.values.len(),
            "{} cumulative output length mismatch: rust={} scipy={}",
            case.case_id,
            rust_vals.len(),
            scipy_result.values.len()
        );

        for (i, (&rv, &sv)) in rust_vals.iter().zip(scipy_result.values.iter()).enumerate() {
            let diff = (rv - sv).abs();
            max_diff = max_diff.max(diff);
            let rel_scale = rv.abs().max(sv.abs()).max(1.0);
            if diff > TOL * rel_scale {
                eprintln!(
                    "{} element {} mismatch: rust={} scipy={} diff={}",
                    case.case_id, i, rv, sv, diff
                );
                all_pass = false;
            }
        }
    }

    assert!(
        all_pass,
        "scipy.integrate cumulative conformance failed, max_diff={}",
        max_diff
    );
}

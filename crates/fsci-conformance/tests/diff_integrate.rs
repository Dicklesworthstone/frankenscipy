#![forbid(unsafe_code)]
//! Live SciPy differential coverage for scipy.integrate functions.
//!
//! Tests FrankenSciPy integration functions against SciPy subprocess oracle
//! across deterministic input families.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_integrate::{
    cumulative_trapezoid, cumulative_trapezoid_uniform, romb, simpson, simpson_uniform, trapezoid,
    trapezoid_uniform,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-008";
const ABS_TOL: f64 = 1.0e-10;
const TOL: f64 = ABS_TOL;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// Ledger arms of `diff_integrate_scalar`: the `func` of each case.
const SCALAR_FUNCS: [&str; 5] = [
    "trapezoid",
    "trapezoid_uniform",
    "simpson",
    "simpson_uniform",
    "romb",
];
/// Ledger arms of `diff_integrate_cumulative`: the `func` of each case.
const CUMULATIVE_FUNCS: [&str; 2] = ["cumtrapz", "cumtrapz_uniform"];

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
    compared: BTreeMap<String, ArmCounts>,
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

/// `None` only when the oracle could not run. A run whose every case raised (the script drops
/// those rows) is `Some` of an empty list, which the coverage check below rejects.
fn scipy_oracle_or_skip(cases: &[IntegrateCase]) -> Option<Vec<OracleResult>> {
    let results = run_scipy_oracle(cases);
    if results.is_none() {
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "SciPy oracle required but not available"
        );
        eprintln!("SciPy oracle not available, skipping diff test");
    }
    results
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
    let Some(oracle_results) = scipy_oracle_or_skip(&cases) else {
        return;
    };

    let oracle_map = complete_scalar_oracle_map("diff_integrate_scalar", &cases, oracle_results);

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_diff = 0.0_f64;
    let mut ledger = CompareLedger::new("diff_integrate_scalar", &SCALAR_FUNCS);

    for case in &cases {
        let arm = case.func.as_str();
        let scipy_val = oracle_map.get(&case.case_id).map(|r| r.value);
        let Some((scipy_val, rust_val)) =
            ledger.pair(arm, &case.case_id, scipy_val, compute_rust_value(case))
        else {
            continue;
        };

        let abs_diff = (rust_val - scipy_val).abs();
        let rel_scale = rust_val.abs().max(scipy_val.abs()).max(1.0);
        let effective_tol = TOL * rel_scale;

        max_diff = max_diff.max(abs_diff);
        ledger.compared(arm, &case.case_id, abs_diff <= effective_tol);

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
        compared: ledger.counts().clone(),
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
    // Every func has its own cases (romb only where n - 1 is a power of two); each arm must
    // compare all of its own.
    let min_per_arm = SCALAR_FUNCS
        .iter()
        .map(|func| cases.iter().filter(|c| c.func == *func).count())
        .min()
        .expect("SCALAR_FUNCS is non-empty");
    ledger.finish(min_per_arm);
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

    // A run that returned no rows ran and raised on every case (the script drops raised rows);
    // that is a broken oracle column, not a skip, and fails the coverage check.
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
    let mut ledger = CompareLedger::new("diff_integrate_cumulative", &CUMULATIVE_FUNCS);

    for case in &cases {
        let arm = case.func.as_str();
        let rust_vals: Option<Vec<f64>> = match arm {
            "cumtrapz" => {
                let x = case.x.as_ref().expect("cumtrapz cases include explicit x");
                cumulative_trapezoid(&case.y, x).ok()
            }
            "cumtrapz_uniform" => {
                let dx = case.dx.expect("uniform cumulative cases include dx");
                cumulative_trapezoid_uniform(&case.y, dx).ok()
            }
            other => {
                eprintln!("unsupported cumulative integrate function {other}");
                all_pass = false;
                None
            }
        };

        let scipy_vals = oracle_map.get(&case.case_id).map(|r| r.values.as_slice());
        // Length, and a NaN element, which `diff > tol` below would let through.
        let Some((scipy_vals, rust_vals)) =
            ledger.slices(arm, &case.case_id, scipy_vals, rust_vals.as_deref())
        else {
            continue;
        };

        let mut case_pass = true;
        for (i, (&rv, &sv)) in rust_vals.iter().zip(scipy_vals.iter()).enumerate() {
            let diff = (rv - sv).abs();
            max_diff = max_diff.max(diff);
            let rel_scale = rv.abs().max(sv.abs()).max(1.0);
            if diff > TOL * rel_scale {
                eprintln!(
                    "{} element {} mismatch: rust={} scipy={} diff={}",
                    case.case_id, i, rv, sv, diff
                );
                all_pass = false;
                case_pass = false;
            }
        }
        ledger.compared(arm, &case.case_id, case_pass);
    }

    assert!(
        all_pass,
        "scipy.integrate cumulative conformance failed, max_diff={}",
        max_diff
    );
    let min_per_arm = CUMULATIVE_FUNCS
        .iter()
        .map(|func| cases.iter().filter(|c| c.func == *func).count())
        .min()
        .expect("CUMULATIVE_FUNCS is non-empty");
    ledger.finish(min_per_arm);
}

#[test]
fn diff_integrate_tanhsinh_nonfinite_and_scaled_integrals() {
    #[derive(Deserialize)]
    struct ProbeRow {
        name: String,
        success: bool,
        status: i32,
        integral: Option<f64>,
    }
    #[derive(Deserialize)]
    struct ProbeOutput {
        scipy: String,
        numpy: String,
        rows: Vec<ProbeRow>,
    }
    let script = r#"
import json
import numpy as np
import scipy
from scipy.integrate import tanhsinh

probes = [
    ('nan', lambda x: np.full_like(x, np.nan), 0., 1.),
    ('inf', lambda x: np.full_like(x, np.inf), 0., 1.),
    ('neg_inf', lambda x: np.full_like(x, -np.inf), 0., 1.),
    ('nan_upper', lambda x: np.full_like(x, np.nan), 0., np.inf),
    ('nan_lower', lambda x: np.full_like(x, np.nan), -np.inf, 0.),
    ('nan_both', lambda x: np.full_like(x, np.nan), -np.inf, np.inf),
    ('nan_refinement', lambda x: np.where((x >= .30) & (x < .34), np.nan, 1.), 0., 1.),
    ('sqrt', lambda x: 1 / np.sqrt(x), 0., 1.),
    ('log', lambda x: np.log(x), 0., 1.),
    ('gaussian', lambda x: np.exp(-x*x), -np.inf, np.inf),
    ('tiny_exp', lambda x: 1e-200 * np.exp(x), 0., 1.),
    ('negative_tiny_exp', lambda x: -1e-200 * np.exp(x), 0., 1.),
]
rows = []
with np.errstate(all='ignore'):
    for name, f, a, b in probes:
        r = tanhsinh(f, a, b, atol=0., rtol=1e-12, maxlevel=16)
        rows.append(dict(name=name, success=bool(r.success), status=int(r.status),
                         integral=float(r.integral) if np.isfinite(r.integral) else None))
print(json.dumps(dict(scipy=scipy.__version__, numpy=np.__version__, rows=rows)))
"#;
    let required = std::env::var(REQUIRE_SCIPY_ENV).as_deref() == Ok("1");
    let output = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .output()
    {
        Ok(output) => output,
        Err(error) => {
            assert!(!required, "cannot start tanhsinh oracle: {error}");
            eprintln!("skipping tanhsinh oracle: {error}");
            return;
        }
    };
    if !output.status.success() {
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(!required, "tanhsinh oracle failed: {error}");
        eprintln!("skipping tanhsinh oracle: {error}");
        return;
    }
    let oracle: ProbeOutput =
        serde_json::from_slice(&output.stdout).expect("complete tanhsinh oracle JSON");
    if required {
        assert_eq!(oracle.scipy, "1.17.1", "pinned SciPy oracle");
        assert_eq!(oracle.numpy, "2.4.3", "pinned NumPy oracle");
    }
    type Probe = (&'static str, fn(f64) -> f64, f64, f64, bool);
    let probes: &[Probe] = &[
        ("nan", |_| f64::NAN, 0.0, 1.0, false),
        ("inf", |_| f64::INFINITY, 0.0, 1.0, false),
        ("neg_inf", |_| f64::NEG_INFINITY, 0.0, 1.0, false),
        ("nan_upper", |_| f64::NAN, 0.0, f64::INFINITY, false),
        ("nan_lower", |_| f64::NAN, f64::NEG_INFINITY, 0.0, false),
        (
            "nan_both",
            |_| f64::NAN,
            f64::NEG_INFINITY,
            f64::INFINITY,
            false,
        ),
        (
            "nan_refinement",
            |x| {
                if (0.30..0.34).contains(&x) {
                    f64::NAN
                } else {
                    1.0
                }
            },
            0.0,
            1.0,
            false,
        ),
        ("sqrt", |x| 1.0 / x.sqrt(), 0.0, 1.0, true),
        ("log", f64::ln, 0.0, 1.0, true),
        (
            "gaussian",
            |x| (-x * x).exp(),
            f64::NEG_INFINITY,
            f64::INFINITY,
            true,
        ),
        ("tiny_exp", |x| 1e-200 * x.exp(), 0.0, 1.0, true),
        ("negative_tiny_exp", |x| -1e-200 * x.exp(), 0.0, 1.0, true),
    ];
    assert_eq!(oracle.rows.len(), probes.len(), "partial oracle coverage");
    let mut ledger = CompareLedger::new(
        "diff_integrate_tanhsinh_nonfinite_and_scaled_integrals",
        &["nonfinite", "finite"],
    );
    for (&(name, f, a, b, finite), reference) in probes.iter().zip(&oracle.rows) {
        assert_eq!(reference.name, name, "oracle case order/identity");
        let actual = fsci_integrate::tanhsinh(f, a, b, 0.0, 1e-12, 16);
        let pass = if finite {
            let expected = reference.integral.expect("finite oracle integral");
            let scale = expected.abs().max(f64::MIN_POSITIVE);
            reference.success
                && actual.converged
                && actual.integral.is_finite()
                && (actual.integral / scale - expected / scale).abs() <= 1e-12
        } else {
            !reference.success
                && reference.status == -3
                && !actual.converged
                && !actual.integral.is_finite()
                && actual.error == f64::INFINITY
        };
        eprintln!(
            "{name}: scipy_status={} scipy={:?}; fsci={actual:?}",
            reference.status, reference.integral,
        );
        ledger.compared(if finite { "finite" } else { "nonfinite" }, name, pass);
    }
    eprintln!(
        "tanhsinh compared={} scipy={} numpy={}",
        probes.len(),
        oracle.scipy,
        oracle.numpy,
    );
    ledger.finish(5);
}

#[test]
fn diff_integrate_tanhsinh_endpoint_resolution() {
    #[derive(Deserialize)]
    struct Reference {
        name: String,
        integral: f64,
        success: bool,
    }
    #[derive(Deserialize)]
    struct Capture {
        scipy: String,
        numpy: String,
        rows: Vec<Reference>,
    }
    let script = r#"
import json
import numpy as np
import scipy
from scipy.integrate import tanhsinh
cases = [
    ('right', lambda x: 1/np.sqrt(1-x), 0., 1., 1e-12),
    ('shifted', lambda x: 1/np.sqrt(x-1), 1., 2., 1e-12),
    ('reverse', lambda x: 1/np.sqrt(1-x), 1., 0., 1e-12),
    ('loose', lambda x: 1/np.sqrt(1-x), 0., 1., 1e-6),
    ('left', lambda x: 1/np.sqrt(x), 0., 1., 1e-12),
    ('smooth', lambda x: np.exp(x), 0., 1., 1e-12),
]
rows = []
with np.errstate(all='ignore'):
    for name, f, a, b, rtol in cases:
        result = tanhsinh(f, a, b, atol=0., rtol=rtol, maxlevel=16)
        rows.append(dict(name=name, integral=float(result.integral),
                         success=bool(result.success)))
print(json.dumps(dict(scipy=scipy.__version__, numpy=np.__version__, rows=rows)))
"#;
    let required = std::env::var(REQUIRE_SCIPY_ENV).as_deref() == Ok("1");
    let output = fsci_conformance::scipy_oracle_command()
        .args(["-c", script])
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .output();
    let output = match output {
        Ok(output) if output.status.success() => output,
        other => {
            assert!(!required, "required endpoint oracle failed: {other:?}");
            eprintln!("SKIP endpoint oracle: {other:?}; compared cases = 0");
            return;
        }
    };
    let oracle: Capture =
        serde_json::from_slice(&output.stdout).expect("complete endpoint capture");
    if required {
        assert_eq!(oracle.scipy, "1.17.1");
        assert_eq!(oracle.numpy, "2.4.3");
    }
    type Case = (&'static str, fn(f64) -> f64, f64, f64, f64, f64, bool);
    let cases: &[Case] = &[
        (
            "right",
            |x| 1.0 / (1.0 - x).sqrt(),
            0.0,
            1.0,
            1e-12,
            2.0,
            false,
        ),
        (
            "shifted",
            |x| 1.0 / (x - 1.0).sqrt(),
            1.0,
            2.0,
            1e-12,
            2.0,
            false,
        ),
        (
            "reverse",
            |x| 1.0 / (1.0 - x).sqrt(),
            1.0,
            0.0,
            1e-12,
            -2.0,
            false,
        ),
        (
            "loose",
            |x| 1.0 / (1.0 - x).sqrt(),
            0.0,
            1.0,
            1e-6,
            2.0,
            true,
        ),
        ("left", |x| 1.0 / x.sqrt(), 0.0, 1.0, 1e-12, 2.0, true),
        (
            "smooth",
            f64::exp,
            0.0,
            1.0,
            1e-12,
            std::f64::consts::E - 1.0,
            true,
        ),
    ];
    assert_eq!(oracle.rows.len(), cases.len(), "partial oracle coverage");
    let mut ledger = CompareLedger::new(
        "diff_integrate_tanhsinh_endpoint_resolution",
        &["resolved", "unresolved"],
    );
    for (&(name, f, a, b, rtol, exact, success), reference) in cases.iter().zip(&oracle.rows) {
        assert_eq!(reference.name, name);
        let actual = fsci_integrate::tanhsinh(f, a, b, 0.0, rtol, 16);
        let actual_error = (actual.integral - exact).abs();
        // At the unresolvable endpoints the relevant parity is refusal to
        // certify convergence. The two node grids have different truncation
        // errors, so do not demand 1e-12 agreement from a failed integral.
        let comparison_tol = if success { rtol * exact.abs() } else { 1e-9 };
        let pass = actual.converged == success
            && reference.success == success
            && actual.integral.is_finite()
            && actual.error.is_finite()
            && actual_error <= actual.error
            && (actual.integral - reference.integral).abs() <= comparison_tol
            && (success || actual.error > rtol * actual.integral.abs());
        eprintln!(
            "{name}: SciPy={} success={}; Rust={actual:?}; actual_error={actual_error}",
            reference.integral, reference.success
        );
        ledger.compared(if success { "resolved" } else { "unresolved" }, name, pass);
    }
    eprintln!(
        "endpoint comparisons={}; scipy={} numpy={}",
        cases.len(),
        oracle.scipy,
        oracle.numpy
    );
    ledger.finish(3);
}

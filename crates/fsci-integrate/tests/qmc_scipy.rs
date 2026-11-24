#![forbid(unsafe_code)]
//! Live SciPy QMC aggregation comparisons at exactly the Rust sample points.
//!
//! The independent randomized engines intentionally have different seed streams.
//! A QMCEngine adapter replays the captured Rust points into scipy.integrate.qmc_quad,
//! allowing tight checks of its mean, signed volume, standard error and one-replicate
//! behavior without confusing randomization differences with numerical errors.

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Mutex;

use fsci_integrate::{QmcQuadOptions, QmcQuadResult, qmc_quad_with_options};
use serde_json::{Value, json};

const RESULT_REL_TOL: f64 = 2e-12;

struct Case {
    id: &'static str,
    kind: &'static str,
    lower: Vec<f64>,
    upper: Vec<f64>,
    options: QmcQuadOptions,
}

fn evaluate(kind: &str, point: &[f64]) -> f64 {
    match kind {
        "quadratic" => point[0] * point[0],
        "sin" => point[0].sin(),
        "sum" => point.iter().sum(),
        "exp_product" => (-point.iter().product::<f64>()).exp(),
        "product" => point.iter().product(),
        "gaussian" => (-point.iter().map(|x| x * x).sum::<f64>()).exp(),
        "aliasing" => (std::f64::consts::TAU * 128.0 * point[0]).cos(),
        "constant" => 2.0,
        _ => panic!("unknown QMC fixture"),
    }
}

fn case(
    id: &'static str,
    kind: &'static str,
    lower: Vec<f64>,
    upper: Vec<f64>,
    n_estimates: usize,
    n_points: usize,
    seed: u64,
) -> Case {
    Case {
        id,
        kind,
        lower,
        upper,
        options: QmcQuadOptions {
            n_estimates,
            n_points,
            seed,
            workers: Some(1),
        },
    }
}

fn cases() -> Vec<Case> {
    vec![
        case("quadratic", "quadratic", vec![0.0], vec![1.0], 8, 64, 0),
        case(
            "sin",
            "sin",
            vec![0.0],
            vec![std::f64::consts::PI],
            8,
            128,
            5,
        ),
        case("sum_2d", "sum", vec![0.0; 2], vec![1.0; 2], 5, 64, 42),
        case(
            "exp_2d",
            "exp_product",
            vec![-1.0, 0.25],
            vec![0.5, 2.0],
            5,
            127,
            u64::MAX,
        ),
        case(
            "product_3d",
            "product",
            vec![-0.5, 0.0, 1.0],
            vec![2.0, 1.0, 3.0],
            8,
            243,
            123,
        ),
        case(
            "gaussian_4d",
            "gaussian",
            vec![-1.0; 4],
            vec![1.0; 4],
            8,
            256,
            2026,
        ),
        case(
            "reversed_one",
            "exp_product",
            vec![1.0, 0.0],
            vec![0.0, 2.0],
            7,
            81,
            9,
        ),
        case(
            "reversed_two",
            "exp_product",
            vec![1.0, 2.0],
            vec![0.0, 0.0],
            7,
            81,
            9,
        ),
        case(
            "one_replicate",
            "quadratic",
            vec![0.0],
            vec![1.0],
            1,
            128,
            17,
        ),
        case(
            "one_point",
            "exp_product",
            vec![0.0; 2],
            vec![1.0; 2],
            8,
            1,
            44,
        ),
        case(
            "constant_16d",
            "constant",
            vec![0.0; 16],
            vec![1.0; 16],
            5,
            64,
            6,
        ),
        case("aliasing", "aliasing", vec![0.0], vec![1.0], 8, 16, 0),
        case(
            "zero_measure",
            "quadratic",
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            1,
            32,
            3,
        ),
    ]
}

const ORACLE: &str = r#"
import json
import sys
import warnings
import numpy as np
import scipy
from scipy.integrate import qmc_quad
from scipy.stats import qmc

assert scipy.__version__ == "1.17.1", scipy.__version__
assert np.__version__ == "2.4.3", np.__version__

class ReplayEngine(qmc.QMCEngine):
    blocks = []
    used = 0

    def __init__(self, d, seed=None):
        super().__init__(d=d, rng=seed)
        self._init_quad = {"d": d}

    def _random(self, n=1, *, workers=1):
        index = type(self).used
        type(self).used += 1
        block = type(self).blocks[index]
        assert block.shape == (n, self.d), block.shape
        return block.copy()

def evaluate(kind, x):
    if kind == "quadratic":
        return x[0] * x[0]
    if kind == "sin":
        return np.sin(x[0])
    if kind == "sum":
        return np.sum(x, axis=0)
    if kind == "exp_product":
        return np.exp(-np.prod(x, axis=0))
    if kind == "product":
        return np.prod(x, axis=0)
    if kind == "gaussian":
        return np.exp(-np.sum(x*x, axis=0))
    if kind == "aliasing":
        return np.cos(2*np.pi*128*x[0])
    if kind == "constant":
        return np.full(x.shape[1:], 2.0)
    raise AssertionError(kind)

out = []
for case in json.load(sys.stdin):
    lower = np.asarray(case["lower"], dtype=float)
    upper = np.asarray(case["upper"], dtype=float)
    lo = np.minimum(lower, upper)
    hi = np.maximum(lower, upper)
    zero = np.any(lo == hi)
    points = np.asarray(case["points"], dtype=float)
    if zero:
        assert points.size == 0
        ReplayEngine.blocks = []
    else:
        assert points.shape == (case["n_estimates"] * case["n_points"], len(lo))
        units = (points - lo) / (hi - lo)
        ReplayEngine.blocks = list(units.reshape(case["n_estimates"], case["n_points"], len(lo)))
    ReplayEngine.used = 0
    engine = ReplayEngine(len(lo), seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        result = qmc_quad(
            lambda x: evaluate(case["kind"], x),
            lower, upper, n_estimates=case["n_estimates"],
            n_points=case["n_points"], qrng=engine,
        )
    assert ReplayEngine.used == (0 if zero else case["n_estimates"])
    assert np.isfinite(result.integral), case["id"]
    missing_error = np.isnan(result.standard_error)
    assert missing_error == (case["n_estimates"] == 1 and not zero), case["id"]
    if not missing_error:
        assert np.isfinite(result.standard_error) and result.standard_error >= 0
    out.append({
        "id": case["id"],
        "integral": float(result.integral),
        "standard_error": None if missing_error else float(result.standard_error),
        "standard_error_is_nan": bool(missing_error),
        "replicates": ReplayEngine.used,
    })
print(json.dumps(out, allow_nan=False))
"#;

fn scipy_oracle(request: &[Value]) -> Option<Vec<Value>> {
    let required = std::env::var_os("FSCI_REQUIRE_SCIPY_ORACLE").is_some();
    let python = std::env::var("SCIPY_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let mut child = match Command::new(python)
        .args(["-c", ORACLE])
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            assert!(
                !required,
                "required QMC SciPy oracle could not start: {error}"
            );
            eprintln!("skipping optional QMC SciPy oracle: {error}");
            return None;
        }
    };
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(&serde_json::to_vec(request).expect("serialize QMC requests"))
        .expect("write QMC oracle input");
    let output = child.wait_with_output().expect("wait for QMC SciPy oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!required, "required QMC SciPy oracle failed: {stderr}");
        eprintln!("skipping optional QMC SciPy oracle: {stderr}");
        return None;
    }
    Some(serde_json::from_slice(&output.stdout).expect("QMC SciPy result JSON"))
}

fn assert_close(actual: f64, expected: f64, context: &str) {
    assert!(actual.is_finite(), "{context}: non-finite Rust result");
    if expected == 0.0 {
        assert_eq!(actual, 0.0, "{context}");
    } else {
        assert!(
            (actual / expected - 1.0).abs() <= RESULT_REL_TOL,
            "{context}: Rust {actual:e}, SciPy {expected:e}"
        );
    }
}

#[test]
fn qmc_quad_matches_pinned_scipy_mean_and_error_at_identical_samples() {
    let cases = cases();
    let mut requests = Vec::new();
    let mut results: Vec<[QmcQuadResult; 2]> = Vec::new();
    for case in &cases {
        let sampled = Mutex::new(Vec::new());
        let serial = qmc_quad_with_options(
            |point| {
                sampled.lock().unwrap().push(point.to_vec());
                evaluate(case.kind, point)
            },
            &case.lower,
            &case.upper,
            case.options,
        )
        .unwrap_or_else(|error| panic!("{}: {error:?}", case.id));
        let points = sampled.into_inner().unwrap();
        let zero = case.lower.iter().zip(&case.upper).any(|(lo, hi)| lo == hi);
        let expected_calls = if zero {
            0
        } else {
            case.options.n_estimates * case.options.n_points
        };
        assert_eq!(points.len(), expected_calls, "{}: callback count", case.id);
        let parallel = qmc_quad_with_options(
            |point| evaluate(case.kind, point),
            &case.lower,
            &case.upper,
            QmcQuadOptions {
                workers: Some(3),
                ..case.options
            },
        )
        .unwrap();
        assert_eq!(
            serial.integral.to_bits(),
            parallel.integral.to_bits(),
            "{}: workers",
            case.id
        );
        assert_eq!(
            serial.standard_error.to_bits(),
            parallel.standard_error.to_bits(),
            "{}: worker standard error",
            case.id
        );
        requests.push(json!({
            "id": case.id,
            "kind": case.kind,
            "lower": case.lower,
            "upper": case.upper,
            "n_estimates": case.options.n_estimates,
            "n_points": case.options.n_points,
            "points": points,
        }));
        results.push([serial, parallel]);
    }
    let Some(expected) = scipy_oracle(&requests) else {
        return;
    };
    assert_eq!(expected.len(), cases.len());
    let mut compared = 0;
    for ((case, observed), expected) in cases.iter().zip(&results).zip(&expected) {
        assert_eq!(expected["id"].as_str(), Some(case.id));
        for (arm, result) in ["serial", "parallel"].iter().zip(observed) {
            let context = format!("{} {arm}", case.id);
            assert_close(
                result.integral,
                expected["integral"].as_f64().unwrap(),
                &context,
            );
            if expected["standard_error_is_nan"].as_bool().unwrap() {
                assert!(
                    result.standard_error.is_nan(),
                    "{context}: uncertainty is undefined"
                );
            } else {
                assert_close(
                    result.standard_error,
                    expected["standard_error"].as_f64().unwrap(),
                    &format!("{context} standard error"),
                );
            }
            eprintln!(
                "{context}: integral={:e}, standard error={:e}, SciPy replicates={}",
                result.integral, result.standard_error, expected["replicates"],
            );
            compared += 1;
        }
    }
    assert_eq!(
        compared, 26,
        "all thirteen cases must run in both worker modes"
    );
}

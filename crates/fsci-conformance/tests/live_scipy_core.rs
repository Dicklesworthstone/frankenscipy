#![forbid(unsafe_code)]
//! Live SciPy differential tests for core FrankenSciPy surfaces.
//!
//! This file intentionally uses a subprocess oracle instead of frozen
//! constants: every case is evaluated by both the Rust implementation and the
//! local `scipy` install from the same serialized inputs. Comparisons use an
//! explicit ULP budget plus a small absolute floor for near-zero values.

use std::collections::HashMap;
use std::io::Write;
use std::process::Stdio;

use fsci_linalg::{LinalgError, MatrixAssumption, SolveOptions, solve};
use fsci_runtime::RuntimeMode;
use fsci_signal::{lfilter, lfilter_axis_2d, lfilter_with_state};
use fsci_special::{
    SpecialResult, SpecialTensor, beta, betainc, betaln, gamma, gammainc, gammaincc, gammaln,
    rgamma,
};
use fsci_stats::{ChiSquared, ContinuousDistribution, Normal, StudentT};
use serde::Serialize;
use serde::de::DeserializeOwned;

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Copy)]
struct ULPPolicy {
    max_ulps: u64,
    abs_floor: f64,
}

#[derive(Debug, Clone, Copy)]
struct FloatDiff {
    abs: f64,
    ulps: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct VectorOracle {
    case_id: String,
    values: Vec<f64>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct MatrixOracle {
    case_id: String,
    values: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct ScalarOracle {
    case_id: String,
    value: f64,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct SpecialOracle {
    case_id: String,
    /// Either a finite float, or one of the sentinels "posinf"/"neginf"/"nan".
    value: serde_json::Value,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct ErrorOracle {
    case_id: String,
    status: String,
    error_type: String,
    message: String,
}

#[derive(Debug, Clone, Serialize)]
struct LfilterCase {
    case_id: String,
    b: Vec<f64>,
    a: Vec<f64>,
    x: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct LfilterStateCase {
    case_id: String,
    b: Vec<f64>,
    a: Vec<f64>,
    x: Vec<f64>,
    zi: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct LfilterAxisCase {
    case_id: String,
    b: Vec<f64>,
    a: Vec<f64>,
    x: Vec<Vec<f64>>,
    axis: isize,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct LfilterStateOracle {
    case_id: String,
    values: Vec<f64>,
    final_state: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct SpecialCase {
    case_id: String,
    func: String,
    args: Vec<f64>,
    /// "finite" | "posinf" | "neginf" | "nan" — matches the Python oracle's
    /// expected non-finite class, used for pole and tail coverage.
    expect: String,
}

#[derive(Debug, Clone, Serialize)]
struct SpecialVectorCase {
    case_id: String,
    func: String,
    args: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct StatsCase {
    case_id: String,
    dist: String,
    params: Vec<f64>,
    method: String,
    value: f64,
}

#[derive(Debug, Clone, Serialize)]
struct SolveCase {
    case_id: String,
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct SolveMatrixRhsCase {
    case_id: String,
    a: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
    assume_a: Option<String>,
    transposed: bool,
    check_finite: bool,
}

#[derive(Debug, Clone, Serialize)]
struct SolveErrorCase {
    case_id: String,
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
    assume_a: Option<String>,
    transposed: bool,
    check_finite: bool,
}

fn scipy_available_or_skip(test_id: &str) -> Result<bool, String> {
    let available = fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg("import scipy")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    if available {
        return Ok(true);
    }

    if std::env::var_os(REQUIRE_SCIPY_ENV).is_some() {
        return Err(format!(
            "{REQUIRE_SCIPY_ENV}=1 but SciPy is unavailable for {test_id}"
        ));
    }
    Ok(false)
}

fn run_python_oracle<I, O>(script: &str, input: &I) -> Result<O, String>
where
    I: Serialize,
    O: DeserializeOwned,
{
    let mut child = fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|err| format!("spawn Python oracle: {err}"))?;
    {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| "open Python oracle stdin".to_owned())?;
        let input_json =
            serde_json::to_vec(input).map_err(|err| format!("serialize oracle input: {err}"))?;
        stdin
            .write_all(&input_json)
            .map_err(|err| format!("write oracle input: {err}"))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|err| format!("wait for Python oracle: {err}"))?;
    if !output.status.success() {
        return Err(format!("Python oracle exited with {}", output.status));
    }
    serde_json::from_slice(&output.stdout).map_err(|err| format!("parse oracle output: {err}"))
}

fn ordered_float_bits(value: f64) -> i64 {
    let bits = value.to_bits() as i64;
    if bits < 0 { i64::MIN - bits } else { bits }
}

fn ulp_distance(left: f64, right: f64) -> u64 {
    if left == right {
        return 0;
    }
    let left_ordered = i128::from(ordered_float_bits(left));
    let right_ordered = i128::from(ordered_float_bits(right));
    left_ordered.abs_diff(right_ordered) as u64
}

fn float_diff(actual: f64, expected: f64) -> FloatDiff {
    FloatDiff {
        abs: (actual - expected).abs(),
        ulps: ulp_distance(actual, expected),
    }
}

fn assert_ulp_close(
    label: &str,
    actual: f64,
    expected: f64,
    policy: ULPPolicy,
) -> Result<FloatDiff, String> {
    if actual == expected {
        return Ok(FloatDiff { abs: 0.0, ulps: 0 });
    }
    if !actual.is_finite() || !expected.is_finite() {
        return Err(format!(
            "{label}: non-finite mismatch, rust={actual:?}, scipy={expected:?}"
        ));
    }
    let diff = float_diff(actual, expected);
    if diff.abs > policy.abs_floor && diff.ulps > policy.max_ulps {
        return Err(format!(
            "{label}: rust={actual:.17e}, scipy={expected:.17e}, abs={:.3e}, ulps={}, max_ulps={}, abs_floor={:.3e}",
            diff.abs, diff.ulps, policy.max_ulps, policy.abs_floor
        ));
    }
    Ok(diff)
}

fn oracle_map<T, F>(
    test_id: &str,
    cases: &[String],
    results: Vec<T>,
    case_id: F,
) -> Result<HashMap<String, T>, String>
where
    F: Fn(&T) -> &str,
{
    let mut map = HashMap::with_capacity(results.len());
    for result in results {
        let previous = map.insert(case_id(&result).to_owned(), result);
        if previous.is_some() {
            return Err(format!("{test_id}: duplicate SciPy oracle case"));
        }
    }

    if map.len() != cases.len() {
        return Err(format!(
            "{test_id}: SciPy oracle returned {} results for {} cases",
            map.len(),
            cases.len()
        ));
    }
    for case in cases {
        if !map.contains_key(case) {
            return Err(format!("{test_id}: missing SciPy oracle case {case}"));
        }
    }
    Ok(map)
}

fn scalar(value: f64) -> SpecialTensor {
    SpecialTensor::RealScalar(value)
}

fn real_scalar(result: SpecialResult, label: &str) -> Result<f64, String> {
    match result {
        Ok(SpecialTensor::RealScalar(value)) => Ok(value),
        Ok(other) => Err(format!("{label}: expected real scalar, got {other:?}")),
        Err(err) => Err(format!("{label}: fsci_special error: {err}")),
    }
}

fn lfilter_cases() -> Vec<LfilterCase> {
    vec![
        LfilterCase {
            case_id: "fir_smoothing".to_owned(),
            b: vec![0.25, 0.5, 0.25],
            a: vec![1.0],
            x: vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0],
        },
        LfilterCase {
            case_id: "iir_first_order".to_owned(),
            b: vec![0.2, 0.1],
            a: vec![1.0, -0.7],
            x: vec![1.0, -1.0, 0.5, 0.25, -0.125, 0.0, 2.0],
        },
        LfilterCase {
            case_id: "normalizes_non_unit_a0".to_owned(),
            b: vec![0.5, -0.25, 0.125],
            a: vec![2.0, -0.5, 0.25],
            x: vec![3.0, 1.0, -2.0, 4.0, 0.5, -0.25],
        },
    ]
}

fn lfilter_state_cases() -> Vec<LfilterStateCase> {
    vec![
        LfilterStateCase {
            case_id: "zi_iir_first_order".to_owned(),
            b: vec![0.2, 0.1],
            a: vec![1.0, -0.7],
            x: vec![1.0, -1.0, 0.5, 0.25],
            zi: vec![0.3],
        },
        LfilterStateCase {
            case_id: "zi_fir_two_delay_states".to_owned(),
            b: vec![0.25, 0.5, 0.25],
            a: vec![1.0],
            x: vec![0.0, 1.0, 4.0, 9.0],
            zi: vec![0.125, -0.25],
        },
    ]
}

fn lfilter_axis_cases() -> Vec<LfilterAxisCase> {
    vec![
        LfilterAxisCase {
            case_id: "axis_last_iir_matrix".to_owned(),
            b: vec![0.2, 0.1],
            a: vec![1.0, -0.7],
            x: vec![vec![1.0, 2.0, 3.0, 4.0], vec![0.5, -1.0, 0.25, 2.0]],
            axis: -1,
        },
        LfilterAxisCase {
            case_id: "axis_zero_iir_matrix".to_owned(),
            b: vec![0.2, 0.1],
            a: vec![1.0, -0.7],
            x: vec![vec![1.0, 2.0, 3.0, 4.0], vec![0.5, -1.0, 0.25, 2.0]],
            axis: 0,
        },
    ]
}

fn lfilter_zi_error_cases() -> Vec<LfilterStateCase> {
    vec![LfilterStateCase {
        case_id: "bad_zi_length_iir_first_order".to_owned(),
        b: vec![0.2, 0.1],
        a: vec![1.0, -0.7],
        x: vec![1.0, 2.0],
        zi: vec![0.1, 0.2],
    }]
}

fn special_cases() -> Vec<SpecialCase> {
    let mut cases = Vec::new();
    for x in [0.5, 1.0, 2.5, 5.0, 8.25] {
        cases.push(SpecialCase {
            case_id: format!("gamma_{x}"),
            func: "gamma".to_owned(),
            args: vec![x],
            expect: "finite".to_owned(),
        });
        cases.push(SpecialCase {
            case_id: format!("gammaln_{x}"),
            func: "gammaln".to_owned(),
            args: vec![x],
            expect: "finite".to_owned(),
        });
        cases.push(SpecialCase {
            case_id: format!("rgamma_{x}"),
            func: "rgamma".to_owned(),
            args: vec![x],
            expect: "finite".to_owned(),
        });
    }

    for (a, b) in [(0.5, 0.5), (2.0, 3.0), (5.5, 1.25), (10.0, 4.0)] {
        cases.push(SpecialCase {
            case_id: format!("beta_{a}_{b}"),
            func: "beta".to_owned(),
            args: vec![a, b],
            expect: "finite".to_owned(),
        });
        cases.push(SpecialCase {
            case_id: format!("betaln_{a}_{b}"),
            func: "betaln".to_owned(),
            args: vec![a, b],
            expect: "finite".to_owned(),
        });
    }

    for (a, b, x) in [(0.5, 0.5, 0.25), (2.0, 3.0, 0.4), (5.0, 2.0, 0.75)] {
        cases.push(SpecialCase {
            case_id: format!("betainc_{a}_{b}_{x}"),
            func: "betainc".to_owned(),
            args: vec![a, b, x],
            expect: "finite".to_owned(),
        });
    }

    for (a, x) in [(0.5, 0.25), (2.0, 1.5), (5.0, 7.0)] {
        cases.push(SpecialCase {
            case_id: format!("gammainc_{a}_{x}"),
            func: "gammainc".to_owned(),
            args: vec![a, x],
            expect: "finite".to_owned(),
        });
        cases.push(SpecialCase {
            case_id: format!("gammaincc_{a}_{x}"),
            func: "gammaincc".to_owned(),
            args: vec![a, x],
            expect: "finite".to_owned(),
        });
    }

    // ── Poles at non-positive integers: gamma(0) is +∞; gamma(-1),
    //    gamma(-2), ... are NaN in modern scipy (sign undefined at the
    //    pole). The earlier comment that "gamma(-1) is +inf in
    //    scipy.special.gamma" was incorrect — scipy returns NaN there.
    cases.push(SpecialCase {
        case_id: "gamma_pole_0".to_owned(),
        func: "gamma".to_owned(),
        args: vec![0.0_f64],
        expect: "posinf".to_owned(),
    });
    for n in [-1.0_f64, -2.0, -3.0] {
        cases.push(SpecialCase {
            case_id: format!("gamma_pole_{n}"),
            func: "gamma".to_owned(),
            args: vec![n],
            expect: "nan".to_owned(),
        });
    }
    // gammaln of a non-positive integer is +∞ (log of the pole).
    for n in [0.0_f64, -1.0, -5.0] {
        cases.push(SpecialCase {
            case_id: format!("gammaln_pole_{n}"),
            func: "gammaln".to_owned(),
            args: vec![n],
            expect: "posinf".to_owned(),
        });
    }
    // rgamma at non-positive integers is exactly 0.
    for n in [0.0_f64, -1.0, -3.0, -10.0] {
        cases.push(SpecialCase {
            case_id: format!("rgamma_pole_{n}"),
            func: "rgamma".to_owned(),
            args: vec![n],
            expect: "finite".to_owned(),
        });
    }

    // ── Sign handling near reflection boundaries: gamma alternates sign
    //    on each non-integer negative interval. -0.5 is positive,
    //    -1.5 is positive (-2 ⇒ + by convention), -2.5 is negative, etc.
    for x in [-0.5_f64, -1.5, -2.5, -3.5, -4.25, -10.75] {
        cases.push(SpecialCase {
            case_id: format!("gamma_neg_{x}"),
            func: "gamma".to_owned(),
            args: vec![x],
            expect: "finite".to_owned(),
        });
    }

    // ── Extreme tail: large positive overflows (gamma(171.6)+) and very
    //    small positive underflows (gammainc with a≪x).
    for x in [150.0_f64, 170.0, 171.0, 172.0, 200.0] {
        cases.push(SpecialCase {
            case_id: format!("gamma_overflow_{x}"),
            func: "gamma".to_owned(),
            args: vec![x],
            expect: if x >= 172.0 { "posinf" } else { "finite" }.to_owned(),
        });
        cases.push(SpecialCase {
            case_id: format!("gammaln_large_{x}"),
            func: "gammaln".to_owned(),
            args: vec![x],
            expect: "finite".to_owned(),
        });
    }

    // Tiny positive (near 0+ but not at the pole) — gamma(1e-12) ~ 1e12.
    for x in [1e-12_f64, 1e-8, 1e-4] {
        cases.push(SpecialCase {
            case_id: format!("gamma_tiny_{x:e}"),
            func: "gamma".to_owned(),
            args: vec![x],
            expect: "finite".to_owned(),
        });
    }

    // Extreme regularised incomplete gamma: a ≪ x → P(a,x) → 1, Q(a,x) → 0.
    for (a, x) in [(0.5, 100.0), (1.0, 50.0), (5.0, 200.0)] {
        cases.push(SpecialCase {
            case_id: format!("gammainc_tail_{a}_{x}"),
            func: "gammainc".to_owned(),
            args: vec![a, x],
            expect: "finite".to_owned(),
        });
        cases.push(SpecialCase {
            case_id: format!("gammaincc_tail_{a}_{x}"),
            func: "gammaincc".to_owned(),
            args: vec![a, x],
            expect: "finite".to_owned(),
        });
    }

    cases
}

fn special_vector_cases() -> Vec<SpecialVectorCase> {
    vec![
        SpecialVectorCase {
            case_id: "gamma_vec_basic".to_owned(),
            func: "gamma".to_owned(),
            args: vec![0.5, 1.0, 2.5, 5.0, 8.25],
        },
        SpecialVectorCase {
            case_id: "gammaln_vec_mixed".to_owned(),
            func: "gammaln".to_owned(),
            args: vec![0.1, 1.0, 10.0, 50.0, 100.0],
        },
        SpecialVectorCase {
            case_id: "rgamma_vec_neg".to_owned(),
            func: "rgamma".to_owned(),
            args: vec![-0.5, 0.5, 1.5, 2.5, 3.5],
        },
    ]
}

fn stats_cases() -> Vec<StatsCase> {
    let mut cases = Vec::new();
    for (loc, scale) in [(0.0, 1.0), (2.0, 0.5)] {
        for x in [-1.0, 0.0, 0.75, 2.0] {
            for method in ["pdf", "cdf", "sf"] {
                cases.push(StatsCase {
                    case_id: format!("norm_{loc}_{scale}_{method}_{x}"),
                    dist: "norm".to_owned(),
                    params: vec![loc, scale],
                    method: method.to_owned(),
                    value: x,
                });
            }
        }
        for q in [0.1, 0.5, 0.9] {
            cases.push(StatsCase {
                case_id: format!("norm_{loc}_{scale}_ppf_{q}"),
                dist: "norm".to_owned(),
                params: vec![loc, scale],
                method: "ppf".to_owned(),
                value: q,
            });
        }
        for z in [-8.0, -5.0, 5.0, 8.0] {
            let x = loc + scale * z;
            for method in ["pdf", "cdf", "sf", "logpdf", "logcdf", "logsf"] {
                cases.push(StatsCase {
                    case_id: format!("norm_{loc}_{scale}_{method}_tail_z{z}"),
                    dist: "norm".to_owned(),
                    params: vec![loc, scale],
                    method: method.to_owned(),
                    value: x,
                });
            }
        }
        for q in [1.0e-6, 1.0e-3, 0.999, 0.999_999] {
            cases.push(StatsCase {
                case_id: format!("norm_{loc}_{scale}_isf_{q}"),
                dist: "norm".to_owned(),
                params: vec![loc, scale],
                method: "isf".to_owned(),
                value: q,
            });
        }
    }

    for df in [1.0, 2.5, 8.0] {
        for x in [-1.5, 0.0, 2.0] {
            for method in ["pdf", "cdf", "sf"] {
                cases.push(StatsCase {
                    case_id: format!("t_{df}_{method}_{x}"),
                    dist: "t".to_owned(),
                    params: vec![df],
                    method: method.to_owned(),
                    value: x,
                });
            }
        }
        for q in [0.1, 0.5, 0.9] {
            cases.push(StatsCase {
                case_id: format!("t_{df}_ppf_{q}"),
                dist: "t".to_owned(),
                params: vec![df],
                method: "ppf".to_owned(),
                value: q,
            });
        }
        for x in [-12.0, -6.0, 6.0, 12.0] {
            for method in ["pdf", "cdf", "sf", "logpdf", "logcdf", "logsf"] {
                cases.push(StatsCase {
                    case_id: format!("t_{df}_{method}_tail_{x}"),
                    dist: "t".to_owned(),
                    params: vec![df],
                    method: method.to_owned(),
                    value: x,
                });
            }
        }
        for q in [1.0e-4, 1.0e-3, 0.999, 0.9999] {
            cases.push(StatsCase {
                case_id: format!("t_{df}_isf_{q}"),
                dist: "t".to_owned(),
                params: vec![df],
                method: "isf".to_owned(),
                value: q,
            });
        }
    }

    for df in [1.0, 2.0, 5.0] {
        for x in [0.25, 1.0, 4.0, 8.0] {
            for method in ["pdf", "cdf", "sf"] {
                cases.push(StatsCase {
                    case_id: format!("chi2_{df}_{method}_{x}"),
                    dist: "chi2".to_owned(),
                    params: vec![df],
                    method: method.to_owned(),
                    value: x,
                });
            }
        }
        for q in [0.1, 0.5, 0.9] {
            cases.push(StatsCase {
                case_id: format!("chi2_{df}_ppf_{q}"),
                dist: "chi2".to_owned(),
                params: vec![df],
                method: "ppf".to_owned(),
                value: q,
            });
        }
        for x in [1.0e-6, 0.05, 20.0, 60.0] {
            for method in ["pdf", "cdf", "sf", "logpdf", "logcdf", "logsf"] {
                cases.push(StatsCase {
                    case_id: format!("chi2_{df}_{method}_tail_{x}"),
                    dist: "chi2".to_owned(),
                    params: vec![df],
                    method: method.to_owned(),
                    value: x,
                });
            }
        }
        for q in [1.0e-6, 1.0e-3, 0.999, 0.999_999] {
            cases.push(StatsCase {
                case_id: format!("chi2_{df}_isf_{q}"),
                dist: "chi2".to_owned(),
                params: vec![df],
                method: "isf".to_owned(),
                value: q,
            });
        }
    }
    cases
}

fn solve_cases() -> Vec<SolveCase> {
    vec![
        SolveCase {
            case_id: "solve_2x2_general".to_owned(),
            a: vec![vec![3.0, 1.0], vec![1.0, 2.0]],
            b: vec![9.0, 8.0],
        },
        SolveCase {
            case_id: "solve_3x3_diag_dominant".to_owned(),
            a: vec![
                vec![4.0, 0.5, -0.25],
                vec![0.25, 3.0, 0.75],
                vec![-0.5, 0.25, 2.5],
            ],
            b: vec![1.0, -2.0, 3.0],
        },
        SolveCase {
            case_id: "solve_scaled_4x4".to_owned(),
            a: vec![
                vec![10.0, 2.0, 0.0, 1.0],
                vec![1.0, 8.0, -1.0, 0.5],
                vec![0.5, -2.0, 7.0, 1.5],
                vec![0.0, 1.0, 2.0, 6.0],
            ],
            b: vec![1.0, 0.5, -1.0, 2.0],
        },
    ]
}

fn solve_matrix_rhs_cases() -> Vec<SolveMatrixRhsCase> {
    vec![
        SolveMatrixRhsCase {
            case_id: "solve_multi_rhs_2x2_general".to_owned(),
            a: vec![vec![3.0, 1.0], vec![1.0, 2.0]],
            b: vec![vec![9.0, 3.0], vec![8.0, 4.0]],
            assume_a: None,
            transposed: false,
            check_finite: true,
        },
        SolveMatrixRhsCase {
            case_id: "solve_multi_rhs_3x3_transposed".to_owned(),
            a: vec![
                vec![2.0, -1.0, 0.5],
                vec![0.25, 3.0, -0.75],
                vec![1.5, 0.5, 4.0],
            ],
            b: vec![vec![1.0, 2.0], vec![-3.0, 0.5], vec![4.0, -1.0]],
            assume_a: None,
            transposed: true,
            check_finite: true,
        },
        SolveMatrixRhsCase {
            case_id: "solve_multi_rhs_diagonal_assumption".to_owned(),
            a: vec![
                vec![2.0, 0.0, 0.0],
                vec![0.0, -4.0, 0.0],
                vec![0.0, 0.0, 0.5],
            ],
            b: vec![vec![2.0, 1.0], vec![8.0, -4.0], vec![1.5, 0.25]],
            assume_a: Some("diagonal".to_owned()),
            transposed: false,
            check_finite: true,
        },
    ]
}

fn solve_error_cases() -> Vec<SolveErrorCase> {
    vec![
        SolveErrorCase {
            case_id: "solve_singular_duplicate_rows".to_owned(),
            a: vec![vec![1.0, 2.0], vec![2.0, 4.0]],
            b: vec![3.0, 6.0],
            assume_a: None,
            transposed: false,
            check_finite: true,
        },
        SolveErrorCase {
            case_id: "solve_singular_diagonal_assumption".to_owned(),
            a: vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.0, 2.0],
            ],
            b: vec![1.0, 4.0, 8.0],
            assume_a: Some("diagonal".to_owned()),
            transposed: false,
            check_finite: true,
        },
    ]
}

fn matrix_assumption_from_name(name: &str) -> Result<MatrixAssumption, String> {
    match name {
        "general" => Ok(MatrixAssumption::General),
        "diagonal" => Ok(MatrixAssumption::Diagonal),
        "upper_triangular" => Ok(MatrixAssumption::UpperTriangular),
        "lower_triangular" => Ok(MatrixAssumption::LowerTriangular),
        "symmetric" => Ok(MatrixAssumption::Symmetric),
        "hermitian" => Ok(MatrixAssumption::Hermitian),
        "positive_definite" => Ok(MatrixAssumption::PositiveDefinite),
        other => Err(format!("unsupported solve assumption {other}")),
    }
}

fn solve_options(
    assume_a: Option<&str>,
    transposed: bool,
    check_finite: bool,
) -> Result<SolveOptions, String> {
    Ok(SolveOptions {
        mode: RuntimeMode::Strict,
        check_finite,
        assume_a: assume_a.map(matrix_assumption_from_name).transpose()?,
        transposed,
        ..SolveOptions::default()
    })
}

fn stats_policy(case: &StatsCase) -> ULPPolicy {
    match case.method.as_str() {
        "logpdf" | "logcdf" | "logsf" => ULPPolicy {
            max_ulps: 500_000_000,
            abs_floor: 1.0e-9,
        },
        "isf" | "ppf" => ULPPolicy {
            max_ulps: 5_000_000_000,
            abs_floor: 1.0e-7,
        },
        _ => ULPPolicy {
            max_ulps: 1_000_000_000,
            abs_floor: 1.0e-8,
        },
    }
}

fn solve_matrix_rhs_by_columns(case: &SolveMatrixRhsCase) -> Result<Vec<Vec<f64>>, String> {
    let rows = case.b.len();
    let cols = case.b.first().map_or(0, Vec::len);
    if rows != case.a.len() || cols == 0 || case.b.iter().any(|row| row.len() != cols) {
        return Err(format!("{}: malformed matrix RHS shape", case.case_id));
    }

    let mut result = vec![vec![0.0; cols]; rows];
    for col in 0..cols {
        let rhs: Vec<f64> = case.b.iter().map(|row| row[col]).collect();
        let solution = solve(
            &case.a,
            &rhs,
            solve_options(case.assume_a.as_deref(), case.transposed, case.check_finite)?,
        )
        .map_err(|err| format!("{} column {col}: Rust solve failed: {err}", case.case_id))?;
        for (row, value) in solution.x.into_iter().enumerate() {
            result[row][col] = value;
        }
    }
    Ok(result)
}

#[test]
fn live_scipy_signal_lfilter_ulp_conformance() -> Result<(), String> {
    let test_id = "live_scipy_signal_lfilter_ulp_conformance";
    if !scipy_available_or_skip(test_id)? {
        return Ok(());
    }

    let cases = lfilter_cases();
    let script = r#"
import json
import sys
from scipy import signal

cases = json.load(sys.stdin)
results = []
for case in cases:
    y = signal.lfilter(case["b"], case["a"], case["x"])
    results.append({"case_id": case["case_id"], "values": [float(v) for v in y]})
print(json.dumps(results))
"#;
    let oracle_results: Vec<VectorOracle> = run_python_oracle(script, &cases)?;
    let case_ids: Vec<String> = cases.iter().map(|case| case.case_id.clone()).collect();
    let oracle = oracle_map(test_id, &case_ids, oracle_results, |result| &result.case_id)?;
    let policy = ULPPolicy {
        max_ulps: 64,
        abs_floor: 1.0e-13,
    };

    for case in &cases {
        let rust = lfilter(&case.b, &case.a, &case.x, None)
            .map_err(|err| format!("{}: Rust lfilter failed: {err}", case.case_id))?;
        let scipy = &oracle[&case.case_id].values;
        if rust.len() != scipy.len() {
            return Err(format!("{}: output length diverged", case.case_id));
        }
        for (idx, (&actual, &expected)) in rust.iter().zip(scipy).enumerate() {
            assert_ulp_close(
                &format!("{}[{idx}]", case.case_id),
                actual,
                expected,
                policy,
            )?;
        }
    }

    let state_cases = lfilter_state_cases();
    let state_script = r#"
import json
import sys
import numpy as np
from scipy import signal

cases = json.load(sys.stdin)
results = []
for case in cases:
    y, zf = signal.lfilter(
        case["b"],
        case["a"],
        case["x"],
        zi=np.asarray(case["zi"], dtype=float),
    )
    results.append({
        "case_id": case["case_id"],
        "values": [float(v) for v in y],
        "final_state": [float(v) for v in zf],
    })
print(json.dumps(results))
"#;
    let state_oracle_results: Vec<LfilterStateOracle> =
        run_python_oracle(state_script, &state_cases)?;
    let state_case_ids: Vec<String> = state_cases
        .iter()
        .map(|case| case.case_id.clone())
        .collect();
    let state_oracle = oracle_map(test_id, &state_case_ids, state_oracle_results, |result| {
        &result.case_id
    })?;

    for case in &state_cases {
        let (rust_y, rust_zf) = lfilter_with_state(&case.b, &case.a, &case.x, Some(&case.zi))
            .map_err(|err| format!("{}: Rust lfilter_with_state failed: {err}", case.case_id))?;
        let scipy = &state_oracle[&case.case_id];
        if rust_y.len() != scipy.values.len() {
            return Err(format!("{}: zi output length diverged", case.case_id));
        }
        if rust_zf.len() != scipy.final_state.len() {
            return Err(format!("{}: zi final-state length diverged", case.case_id));
        }
        for (idx, (&actual, &expected)) in rust_y.iter().zip(&scipy.values).enumerate() {
            assert_ulp_close(
                &format!("{} y[{idx}]", case.case_id),
                actual,
                expected,
                policy,
            )?;
        }
        for (idx, (&actual, &expected)) in rust_zf.iter().zip(&scipy.final_state).enumerate() {
            assert_ulp_close(
                &format!("{} zf[{idx}]", case.case_id),
                actual,
                expected,
                policy,
            )?;
        }
    }

    let axis_cases = lfilter_axis_cases();
    let axis_script = r#"
import json
import sys
import numpy as np
from scipy import signal

cases = json.load(sys.stdin)
results = []
for case in cases:
    x = np.asarray(case["x"], dtype=float)
    y = signal.lfilter(case["b"], case["a"], x, axis=int(case["axis"]))
    results.append({
        "case_id": case["case_id"],
        "values": [[float(v) for v in row] for row in y.tolist()],
    })
print(json.dumps(results))
"#;
    let axis_oracle_results: Vec<MatrixOracle> = run_python_oracle(axis_script, &axis_cases)?;
    let axis_case_ids: Vec<String> = axis_cases.iter().map(|case| case.case_id.clone()).collect();
    let axis_oracle = oracle_map(test_id, &axis_case_ids, axis_oracle_results, |result| {
        &result.case_id
    })?;

    for case in &axis_cases {
        let rust = lfilter_axis_2d(&case.b, &case.a, &case.x, case.axis)
            .map_err(|err| format!("{}: Rust lfilter_axis_2d failed: {err}", case.case_id))?;
        let scipy = &axis_oracle[&case.case_id].values;
        if rust.len() != scipy.len() {
            return Err(format!("{}: axis row count diverged", case.case_id));
        }
        for (row_idx, (rust_row, scipy_row)) in rust.iter().zip(scipy.iter()).enumerate() {
            if rust_row.len() != scipy_row.len() {
                return Err(format!("{}: axis column count diverged", case.case_id));
            }
            for (col_idx, (&actual, &expected)) in rust_row.iter().zip(scipy_row).enumerate() {
                assert_ulp_close(
                    &format!("{}[{row_idx}][{col_idx}]", case.case_id),
                    actual,
                    expected,
                    policy,
                )?;
            }
        }
    }

    let error_cases = lfilter_zi_error_cases();
    let error_script = r#"
import json
import sys
import numpy as np
from scipy import signal

cases = json.load(sys.stdin)
results = []
for case in cases:
    try:
        signal.lfilter(
            case["b"],
            case["a"],
            case["x"],
            zi=np.asarray(case["zi"], dtype=float),
        )
        results.append({
            "case_id": case["case_id"],
            "status": "ok",
            "error_type": "",
            "message": "",
        })
    except Exception as exc:
        results.append({
            "case_id": case["case_id"],
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        })
print(json.dumps(results))
"#;
    let error_oracle_results: Vec<ErrorOracle> = run_python_oracle(error_script, &error_cases)?;
    let error_case_ids: Vec<String> = error_cases
        .iter()
        .map(|case| case.case_id.clone())
        .collect();
    let error_oracle = oracle_map(test_id, &error_case_ids, error_oracle_results, |result| {
        &result.case_id
    })?;

    for case in &error_cases {
        let scipy_error = &error_oracle[&case.case_id];
        if scipy_error.status != "error" {
            return Err(format!(
                "{}: SciPy accepted malformed zi unexpectedly",
                case.case_id
            ));
        }
        if scipy_error.error_type != "ValueError" {
            return Err(format!(
                "{}: expected SciPy ValueError for malformed zi, got {}: {}",
                case.case_id, scipy_error.error_type, scipy_error.message
            ));
        }
        let rust_error = lfilter_with_state(&case.b, &case.a, &case.x, Some(&case.zi))
            .expect_err("malformed zi must fail");
        if !rust_error.to_string().contains("zi length") {
            return Err(format!(
                "{}: Rust malformed zi error lost context: {rust_error}",
                case.case_id
            ));
        }
    }
    Ok(())
}

#[test]
fn live_scipy_special_gamma_beta_family_ulp_conformance() -> Result<(), String> {
    let test_id = "live_scipy_special_gamma_beta_family_ulp_conformance";
    if !scipy_available_or_skip(test_id)? {
        return Ok(());
    }

    let cases = special_cases();
    let script = r#"
import json
import math
import sys
from scipy import special

def encode(v):
    f = float(v)
    if math.isnan(f):
        return "nan"
    if math.isinf(f):
        return "posinf" if f > 0 else "neginf"
    return f

cases = json.load(sys.stdin)
results = []
for case in cases:
    func = case["func"]
    args = [float(v) for v in case["args"]]
    if func == "gamma":
        value = special.gamma(args[0])
    elif func == "gammaln":
        value = special.gammaln(args[0])
    elif func == "rgamma":
        value = special.rgamma(args[0]) if hasattr(special, "rgamma") else 1.0 / special.gamma(args[0])
    elif func == "beta":
        value = special.beta(args[0], args[1])
    elif func == "betaln":
        value = special.betaln(args[0], args[1])
    elif func == "betainc":
        value = special.betainc(args[0], args[1], args[2])
    elif func == "gammainc":
        value = special.gammainc(args[0], args[1])
    elif func == "gammaincc":
        value = special.gammaincc(args[0], args[1])
    else:
        raise ValueError(func)
    results.append({"case_id": case["case_id"], "value": encode(value)})
print(json.dumps(results))
"#;
    let oracle_results: Vec<SpecialOracle> = run_python_oracle(script, &cases)?;
    let case_ids: Vec<String> = cases.iter().map(|case| case.case_id.clone()).collect();
    let oracle = oracle_map(test_id, &case_ids, oracle_results, |result| &result.case_id)?;
    let policy = ULPPolicy {
        max_ulps: 50_000_000,
        abs_floor: 1.0e-10,
    };

    for case in &cases {
        let actual = match case.func.as_str() {
            "gamma" => real_scalar(
                gamma(&scalar(case.args[0]), RuntimeMode::Strict),
                &case.case_id,
            )?,
            "gammaln" => real_scalar(
                gammaln(&scalar(case.args[0]), RuntimeMode::Strict),
                &case.case_id,
            )?,
            "rgamma" => real_scalar(
                rgamma(&scalar(case.args[0]), RuntimeMode::Strict),
                &case.case_id,
            )?,
            "beta" => real_scalar(
                beta(
                    &scalar(case.args[0]),
                    &scalar(case.args[1]),
                    RuntimeMode::Strict,
                ),
                &case.case_id,
            )?,
            "betaln" => real_scalar(
                betaln(
                    &scalar(case.args[0]),
                    &scalar(case.args[1]),
                    RuntimeMode::Strict,
                ),
                &case.case_id,
            )?,
            "betainc" => real_scalar(
                betainc(
                    &scalar(case.args[0]),
                    &scalar(case.args[1]),
                    &scalar(case.args[2]),
                    RuntimeMode::Strict,
                ),
                &case.case_id,
            )?,
            "gammainc" => real_scalar(
                gammainc(
                    &scalar(case.args[0]),
                    &scalar(case.args[1]),
                    RuntimeMode::Strict,
                ),
                &case.case_id,
            )?,
            "gammaincc" => real_scalar(
                gammaincc(
                    &scalar(case.args[0]),
                    &scalar(case.args[1]),
                    RuntimeMode::Strict,
                ),
                &case.case_id,
            )?,
            other => return Err(format!("unsupported special case func {other}")),
        };
        let oracle_value = &oracle[&case.case_id].value;
        assert_special_value_match(&case.case_id, actual, oracle_value, &case.expect, policy)?;
    }
    Ok(())
}

#[test]
fn live_scipy_special_gamma_family_vector_broadcasting() -> Result<(), String> {
    let test_id = "live_scipy_special_gamma_family_vector_broadcasting";
    if !scipy_available_or_skip(test_id)? {
        return Ok(());
    }

    let cases = special_vector_cases();
    let script = r#"
import json
import sys
from scipy import special

cases = json.load(sys.stdin)
results = []
for case in cases:
    func = case["func"]
    args = case["args"]
    if func == "gamma":
        out = special.gamma(args)
    elif func == "gammaln":
        out = special.gammaln(args)
    elif func == "rgamma":
        out = special.rgamma(args) if hasattr(special, "rgamma") else 1.0 / special.gamma(args)
    else:
        raise ValueError(func)
    values = [float(v) for v in list(out)]
    results.append({"case_id": case["case_id"], "values": values})
print(json.dumps(results))
"#;
    let oracle_results: Vec<VectorOracle> = run_python_oracle(script, &cases)?;
    let case_ids: Vec<String> = cases.iter().map(|c| c.case_id.clone()).collect();
    let oracle = oracle_map(test_id, &case_ids, oracle_results, |r| &r.case_id)?;

    let policy = ULPPolicy {
        max_ulps: 50_000_000,
        abs_floor: 1.0e-10,
    };

    for case in &cases {
        let input_tensor = SpecialTensor::RealVec(case.args.clone());
        let result = match case.func.as_str() {
            "gamma" => gamma(&input_tensor, RuntimeMode::Strict),
            "gammaln" => gammaln(&input_tensor, RuntimeMode::Strict),
            "rgamma" => rgamma(&input_tensor, RuntimeMode::Strict),
            other => return Err(format!("vector case unsupported func {other}")),
        };
        let actual = match result.map_err(|e| format!("{}: {e:?}", case.case_id))? {
            SpecialTensor::RealVec(v) => v,
            other => {
                return Err(format!(
                    "{}: expected RealVec output, got {other:?}",
                    case.case_id
                ));
            }
        };
        let expected = &oracle[&case.case_id].values;
        if actual.len() != expected.len() {
            return Err(format!(
                "{}: vector length mismatch — rust={} scipy={}",
                case.case_id,
                actual.len(),
                expected.len()
            ));
        }
        for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
            assert_ulp_close(&format!("{}_idx{}", case.case_id, i), *a, *b, policy)?;
        }
    }
    Ok(())
}

/// Assert agreement between an fsci scalar result and the SciPy oracle's
/// possibly-non-finite encoded value, dispatching on the case's expectation.
///
/// `oracle_value` is JSON: either a numeric `f64` (finite case) or one of the
/// string sentinels "posinf"/"neginf"/"nan".
fn assert_special_value_match(
    label: &str,
    actual: f64,
    oracle_value: &serde_json::Value,
    expect: &str,
    policy: ULPPolicy,
) -> Result<(), String> {
    use serde_json::Value;
    match (expect, oracle_value) {
        ("finite", Value::Number(n)) => {
            let expected = n
                .as_f64()
                .ok_or_else(|| format!("{label}: oracle value not f64-convertible"))?;
            if !actual.is_finite() {
                return Err(format!(
                    "{label}: expected finite agreement but rust returned {actual:?} \
                     (scipy = {expected:.17e})"
                ));
            }
            assert_ulp_close(label, actual, expected, policy)?;
            Ok(())
        }
        ("posinf", Value::String(s)) if s == "posinf" => {
            if actual.is_infinite() && actual > 0.0 {
                Ok(())
            } else {
                Err(format!(
                    "{label}: expected +∞ (matches scipy), got rust = {actual:?}"
                ))
            }
        }
        ("neginf", Value::String(s)) if s == "neginf" => {
            if actual.is_infinite() && actual < 0.0 {
                Ok(())
            } else {
                Err(format!(
                    "{label}: expected -∞ (matches scipy), got rust = {actual:?}"
                ))
            }
        }
        ("nan", Value::String(s)) if s == "nan" => {
            if actual.is_nan() {
                Ok(())
            } else {
                Err(format!(
                    "{label}: expected NaN (matches scipy), got rust = {actual:?}"
                ))
            }
        }
        // Drift detected: the case was tagged with one expectation but SciPy
        // returned a different class. Surface this as a discrepancy record.
        (tag, oracle) => Err(format!(
            "{label}: expectation tag '{tag}' did not match SciPy oracle {oracle}; \
             this is an explicit strict/hardened deviation that should be added \
             to the discrepancy ledger before relaxing the test"
        )),
    }
}

#[test]
fn live_scipy_stats_norm_t_chi2_ulp_conformance() -> Result<(), String> {
    let test_id = "live_scipy_stats_norm_t_chi2_ulp_conformance";
    if !scipy_available_or_skip(test_id)? {
        return Ok(());
    }

    let cases = stats_cases();
    let script = r#"
import json
import sys
from scipy import stats

cases = json.load(sys.stdin)
results = []
for case in cases:
    dist = case["dist"]
    method = case["method"]
    value = float(case["value"])
    params = [float(v) for v in case["params"]]
    if dist == "norm":
        rv = stats.norm(loc=params[0], scale=params[1])
    elif dist == "t":
        rv = stats.t(df=params[0])
    elif dist == "chi2":
        rv = stats.chi2(df=params[0])
    else:
        raise ValueError(dist)
    result = getattr(rv, method)(value)
    results.append({"case_id": case["case_id"], "value": float(result)})
print(json.dumps(results))
"#;
    let oracle_results: Vec<ScalarOracle> = run_python_oracle(script, &cases)?;
    let case_ids: Vec<String> = cases.iter().map(|case| case.case_id.clone()).collect();
    let oracle = oracle_map(test_id, &case_ids, oracle_results, |result| &result.case_id)?;
    for case in &cases {
        let actual = match case.dist.as_str() {
            "norm" => {
                let dist = Normal::new(case.params[0], case.params[1]);
                match case.method.as_str() {
                    "pdf" => dist.pdf(case.value),
                    "logpdf" => dist.logpdf(case.value),
                    "cdf" => dist.cdf(case.value),
                    "logcdf" => dist.logcdf(case.value),
                    "sf" => dist.sf(case.value),
                    "logsf" => dist.logsf(case.value),
                    "ppf" => dist.ppf(case.value),
                    "isf" => dist.isf(case.value),
                    other => return Err(format!("unsupported norm method {other}")),
                }
            }
            "t" => {
                let dist = StudentT::new(case.params[0]);
                match case.method.as_str() {
                    "pdf" => dist.pdf(case.value),
                    "logpdf" => dist.logpdf(case.value),
                    "cdf" => dist.cdf(case.value),
                    "logcdf" => dist.logcdf(case.value),
                    "sf" => dist.sf(case.value),
                    "logsf" => dist.logsf(case.value),
                    "ppf" => dist.ppf(case.value),
                    "isf" => dist.isf(case.value),
                    other => return Err(format!("unsupported t method {other}")),
                }
            }
            "chi2" => {
                let dist = ChiSquared::new(case.params[0]);
                match case.method.as_str() {
                    "pdf" => dist.pdf(case.value),
                    "logpdf" => dist.logpdf(case.value),
                    "cdf" => dist.cdf(case.value),
                    "logcdf" => dist.logcdf(case.value),
                    "sf" => dist.sf(case.value),
                    "logsf" => dist.logsf(case.value),
                    "ppf" => dist.ppf(case.value),
                    "isf" => dist.isf(case.value),
                    other => return Err(format!("unsupported chi2 method {other}")),
                }
            }
            other => return Err(format!("unsupported stats distribution {other}")),
        };
        assert_ulp_close(
            &case.case_id,
            actual,
            oracle[&case.case_id].value,
            stats_policy(case),
        )?;
    }
    Ok(())
}

#[test]
fn live_scipy_linalg_solve_ulp_conformance() -> Result<(), String> {
    let test_id = "live_scipy_linalg_solve_ulp_conformance";
    if !scipy_available_or_skip(test_id)? {
        return Ok(());
    }

    let cases = solve_cases();
    let script = r#"
import json
import sys
import numpy as np
from scipy import linalg

cases = json.load(sys.stdin)
results = []
for case in cases:
    a = np.array(case["a"], dtype=np.float64)
    b = np.array(case["b"], dtype=np.float64)
    x = linalg.solve(a, b)
    results.append({"case_id": case["case_id"], "values": [float(v) for v in x]})
print(json.dumps(results))
"#;
    let oracle_results: Vec<VectorOracle> = run_python_oracle(script, &cases)?;
    let case_ids: Vec<String> = cases.iter().map(|case| case.case_id.clone()).collect();
    let oracle = oracle_map(test_id, &case_ids, oracle_results, |result| &result.case_id)?;
    let policy = ULPPolicy {
        max_ulps: 1_000_000,
        abs_floor: 1.0e-11,
    };

    for case in &cases {
        let rust = solve(
            &case.a,
            &case.b,
            SolveOptions {
                mode: RuntimeMode::Strict,
                check_finite: true,
                ..SolveOptions::default()
            },
        )
        .map_err(|err| format!("{}: Rust solve failed: {err}", case.case_id))?;
        let scipy = &oracle[&case.case_id].values;
        if rust.x.len() != scipy.len() {
            return Err(format!("{}: solution length diverged", case.case_id));
        }
        for (idx, (&actual, &expected)) in rust.x.iter().zip(scipy).enumerate() {
            assert_ulp_close(
                &format!("{}[{idx}]", case.case_id),
                actual,
                expected,
                policy,
            )?;
        }
    }
    Ok(())
}

#[test]
fn live_scipy_linalg_solve_singular_and_multi_rhs_conformance() -> Result<(), String> {
    let test_id = "live_scipy_linalg_solve_singular_and_multi_rhs_conformance";
    if !scipy_available_or_skip(test_id)? {
        return Ok(());
    }

    let matrix_cases = solve_matrix_rhs_cases();
    let matrix_script = r#"
import json
import sys
import numpy as np
from scipy import linalg

cases = json.load(sys.stdin)
results = []
for case in cases:
    a = np.array(case["a"], dtype=np.float64)
    b = np.array(case["b"], dtype=np.float64)
    if case.get("transposed", False):
        a = a.T
    assume_a = case.get("assume_a")
    if assume_a == "diagonal":
        diag = np.diag(a)
        if np.any(diag == 0.0):
            raise linalg.LinAlgError("singular matrix")
        x = b / diag[:, None]
    else:
        scipy_assume = {
            None: "gen",
            "general": "gen",
            "symmetric": "sym",
            "hermitian": "her",
            "positive_definite": "pos",
        }.get(assume_a, "gen")
        x = linalg.solve(
            a,
            b,
            assume_a=scipy_assume,
            check_finite=bool(case.get("check_finite", True)),
        )
    results.append({
        "case_id": case["case_id"],
        "values": [[float(v) for v in row] for row in np.asarray(x)],
    })
print(json.dumps(results))
"#;
    let matrix_oracle_results: Vec<MatrixOracle> = run_python_oracle(matrix_script, &matrix_cases)?;
    let matrix_case_ids: Vec<String> = matrix_cases
        .iter()
        .map(|case| case.case_id.clone())
        .collect();
    let matrix_oracle = oracle_map(test_id, &matrix_case_ids, matrix_oracle_results, |result| {
        &result.case_id
    })?;
    let matrix_policy = ULPPolicy {
        max_ulps: 1_000_000,
        abs_floor: 1.0e-11,
    };

    for case in &matrix_cases {
        let rust = solve_matrix_rhs_by_columns(case)?;
        let scipy = &matrix_oracle[&case.case_id].values;
        if rust.len() != scipy.len() {
            return Err(format!("{}: matrix RHS row count diverged", case.case_id));
        }
        for (row_idx, (rust_row, scipy_row)) in rust.iter().zip(scipy).enumerate() {
            if rust_row.len() != scipy_row.len() {
                return Err(format!(
                    "{}: matrix RHS column count diverged at row {row_idx}",
                    case.case_id
                ));
            }
            for (col_idx, (&actual, &expected)) in rust_row.iter().zip(scipy_row).enumerate() {
                assert_ulp_close(
                    &format!("{}[{row_idx},{col_idx}]", case.case_id),
                    actual,
                    expected,
                    matrix_policy,
                )?;
            }
        }
    }

    let error_cases = solve_error_cases();
    let error_script = r#"
import json
import sys
import numpy as np
from scipy import linalg

cases = json.load(sys.stdin)
results = []
for case in cases:
    try:
        a = np.array(case["a"], dtype=np.float64)
        b = np.array(case["b"], dtype=np.float64)
        if case.get("transposed", False):
            a = a.T
        assume_a = case.get("assume_a")
        if assume_a == "diagonal":
            diag = np.diag(a)
            if np.any(diag == 0.0):
                raise linalg.LinAlgError("singular matrix")
            _ = b / diag
        else:
            scipy_assume = {
                None: "gen",
                "general": "gen",
                "symmetric": "sym",
                "hermitian": "her",
                "positive_definite": "pos",
            }.get(assume_a, "gen")
            _ = linalg.solve(
                a,
                b,
                assume_a=scipy_assume,
                check_finite=bool(case.get("check_finite", True)),
            )
        results.append({
            "case_id": case["case_id"],
            "status": "ok",
            "error_type": "",
            "message": "",
        })
    except Exception as exc:
        results.append({
            "case_id": case["case_id"],
            "status": "error",
            "error_type": exc.__class__.__name__,
            "message": str(exc),
        })
print(json.dumps(results))
"#;
    let error_oracle_results: Vec<ErrorOracle> = run_python_oracle(error_script, &error_cases)?;
    let error_case_ids: Vec<String> = error_cases
        .iter()
        .map(|case| case.case_id.clone())
        .collect();
    let error_oracle = oracle_map(test_id, &error_case_ids, error_oracle_results, |result| {
        &result.case_id
    })?;

    for case in &error_cases {
        let scipy = &error_oracle[&case.case_id];
        if scipy.status != "error" || scipy.error_type != "LinAlgError" {
            return Err(format!(
                "{}: expected SciPy LinAlgError, got status={} type={} message={}",
                case.case_id, scipy.status, scipy.error_type, scipy.message
            ));
        }
        match solve(
            &case.a,
            &case.b,
            solve_options(case.assume_a.as_deref(), case.transposed, case.check_finite)?,
        ) {
            Err(LinalgError::SingularMatrix) => {}
            Err(err) => {
                return Err(format!(
                    "{}: expected Rust SingularMatrix, got {err}",
                    case.case_id
                ));
            }
            Ok(result) => {
                return Err(format!(
                    "{}: expected Rust SingularMatrix, got solution {:?}",
                    case.case_id, result.x
                ));
            }
        }
    }

    Ok(())
}

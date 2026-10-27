#![forbid(unsafe_code)]
//! Measurement-error curve fitting through the public API. Closed-form regressions
//! run without Python. The live differential test uses SCIPY_PYTHON (or python3);
//! FSCI_REQUIRE_SCIPY_ORACLE=1 requires SciPy 1.17.1 / NumPy 2.4.3 and all 60 cases.

use std::io::Write;
use std::process::{Command, Stdio};

use fsci_opt::{
    CurveFitOptions, CurveFitSigma, LeastSquaresJacobian, LeastSquaresMethod, LeastSquaresOptions,
    LossKind, OptError, curve_fit, curve_fit_bounded, curve_fit_bounded_many, curve_fit_many,
};
use serde::{Deserialize, Serialize};

const METHODS: [LeastSquaresMethod; 3] = [
    LeastSquaresMethod::Lm,
    LeastSquaresMethod::Trf,
    LeastSquaresMethod::Dogbox,
];

fn correlated_covariance() -> Vec<Vec<f64>> {
    vec![
        vec![4.0, 1.0, 0.0],
        vec![1.0, 1.0, 0.0],
        vec![0.0, 0.0, 9.0],
    ]
}

fn constant_options(
    method: LeastSquaresMethod,
    sigma: CurveFitSigma,
    analytic: bool,
    absolute_sigma: bool,
) -> CurveFitOptions {
    CurveFitOptions {
        p0: Some(vec![0.0]),
        sigma: Some(sigma),
        absolute_sigma,
        ls_options: LeastSquaresOptions {
            method: Some(method),
            jac: analytic.then(|| LeastSquaresJacobian::new(|_| vec![vec![1.0]; 3])),
            gtol: 1e-12,
            xtol: 1e-12,
            ftol: 1e-12,
            ..LeastSquaresOptions::default()
        },
    }
}

#[test]
fn sigma_recovers_weighted_mean_and_parameter_variance() {
    // Independent WLS formula: sum(y/sigma²)/sum(1/sigma²). The unweighted
    // mean is 13/3, so ignoring the measurement errors cannot pass.
    let expected: f64 = 58.0 / 33.0;
    let weighted_cost = 0.5
        * ((expected - 1.0).powi(2) + (expected - 2.0).powi(2) + ((expected - 10.0) / 4.0).powi(2));
    for method in METHODS {
        for analytic in [false, true] {
            for sigma in [
                CurveFitSigma::StandardDeviations(vec![1.0, 1.0, 4.0]),
                CurveFitSigma::StandardDeviations(vec![-1.0, 1.0, -4.0]),
                CurveFitSigma::Covariance(vec![
                    vec![1.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0],
                    vec![0.0, 0.0, 16.0],
                ]),
            ] {
                let fit = curve_fit(
                    |_x, p| p[0],
                    &[0.0, 1.0, 2.0],
                    &[1.0, 2.0, 10.0],
                    constant_options(method, sigma, analytic, true),
                )
                .expect("weighted constant fit");
                assert!((fit.popt[0] - expected).abs() < 1e-7, "{method:?}: {fit:?}");
                assert!(
                    (fit.pcov[0][0] - 16.0 / 33.0).abs() < 1e-7,
                    "{method:?}: {fit:?}"
                );
                assert!((fit.ls_result.cost - weighted_cost).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn sigma_whitens_correlated_errors_and_analytic_jacobians() {
    // C⁻¹ 1 = [0, 1, 1/9], hence mean = 2.8, absolute variance = 0.9.
    // Diagonal-only weighting, C⁻¹ in place of L⁻¹, and an unweighted analytic
    // Jacobian each give a different answer and are independently detected.
    for method in METHODS {
        for analytic in [false, true] {
            for absolute_sigma in [false, true] {
                let fit = curve_fit(
                    |_x, p| p[0],
                    &[0.0, 1.0, 2.0],
                    &[1.0, 2.0, 10.0],
                    constant_options(
                        method,
                        CurveFitSigma::Covariance(correlated_covariance()),
                        analytic,
                        absolute_sigma,
                    ),
                )
                .expect("generalized least-squares fit");
                assert!((fit.popt[0] - 2.8).abs() < 1e-7, "{method:?}: {fit:?}");
                let variance = if absolute_sigma { 0.9 } else { 3.03 };
                assert!(
                    (fit.pcov[0][0] - variance).abs() < 1e-7,
                    "{method:?}: {fit:?}"
                );
                assert!((fit.ls_result.cost - 101.0 / 30.0).abs() < 1e-12);
                assert!((fit.ls_result.jac[0][0] - 0.5).abs() < 1e-7);
                assert!((fit.ls_result.jac[1][0] - 3.0_f64.sqrt() / 2.0).abs() < 1e-7);
                assert!((fit.ls_result.jac[2][0] - 1.0 / 3.0).abs() < 1e-7);
            }
        }
    }
}

#[test]
fn scalar_sigma_sets_absolute_covariance_units() {
    for absolute_sigma in [false, true] {
        for sigma in [
            CurveFitSigma::Scalar(2.0),
            CurveFitSigma::StandardDeviations(vec![2.0]),
        ] {
            let fit = curve_fit(
                |_x, p| p[0],
                &[0.0, 1.0, 2.0],
                &[1.0, 2.0, 6.0],
                constant_options(LeastSquaresMethod::Trf, sigma, true, absolute_sigma),
            )
            .expect("scalar sigma broadcasts");
            assert!((fit.popt[0] - 3.0).abs() < 1e-12);
            let variance = if absolute_sigma { 4.0 / 3.0 } else { 7.0 / 3.0 };
            assert!((fit.pcov[0][0] - variance).abs() < 1e-12);
            assert!((fit.ls_result.cost - 1.75).abs() < 1e-12);
        }
    }
}

#[test]
fn subnormal_sigma_scales_finite_residuals_and_jacobians() {
    let scale = 1.0e-320;
    assert!((1.0_f64 / scale).is_infinite());
    for method in METHODS {
        for sigma in [
            CurveFitSigma::Scalar(scale),
            CurveFitSigma::StandardDeviations(vec![scale, -scale, scale]),
        ] {
            let mut options = constant_options(method, sigma, true, true);
            options.ls_options.jac = Some(LeastSquaresJacobian::new(move |_| vec![vec![scale]; 3]));
            let fit = curve_fit(
                |_x, p| scale * p[0],
                &[0.0, 1.0, 2.0],
                &[2.0 * scale; 3],
                options,
            )
            .expect("subnormal errors can have finite scaled residuals and derivatives");
            // Subnormal multiplication quantizes the model at about 5e-4 in
            // parameter space. Do not demand nonexistent extra precision.
            assert!((fit.popt[0] - 2.0).abs() < 1e-3, "{method:?}: {fit:?}");
            assert_eq!(fit.ls_result.cost, 0.0);
            assert!(fit.ls_result.fun.iter().all(|&value| value == 0.0));
            assert!(fit.ls_result.jac.iter().all(|row| row[0].abs() == 1.0));
            assert!((fit.pcov[0][0] - 1.0 / 3.0).abs() < 1e-12);
        }
    }
}

#[test]
fn sigma_supports_active_bounds_and_batches() {
    let xdata = [0.0, 1.0, 2.0];
    let rows: Vec<Vec<f64>> = (0..12)
        .map(|i| vec![1.0, 2.0, 10.0 + f64::from(i)])
        .collect();
    for sigma in [
        CurveFitSigma::StandardDeviations(vec![1.0, 1.0, 4.0]),
        CurveFitSigma::Covariance(correlated_covariance()),
    ] {
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            let mut options = constant_options(method, sigma.clone(), true, true);
            options.p0 = Some(vec![1.0]);
            for bounded in [false, true] {
                let batch = if bounded {
                    curve_fit_bounded_many(
                        |_x, p| p[0],
                        &xdata,
                        &rows,
                        &[0.0],
                        &[1.5],
                        options.clone(),
                    )
                } else {
                    curve_fit_many(|_x, p| p[0], &xdata, &rows, options.clone())
                }
                .expect("weighted batch");
                for (row, parameters) in rows.iter().zip(batch) {
                    let fit = if bounded {
                        curve_fit_bounded(
                            |_x, p| p[0],
                            &xdata,
                            row,
                            &[0.0],
                            &[1.5],
                            options.clone(),
                        )
                    } else {
                        curve_fit(|_x, p| p[0], &xdata, row, options.clone())
                    }
                    .expect("weighted single fit");
                    assert_eq!(parameters[0].to_bits(), fit.popt[0].to_bits());
                    if bounded {
                        assert!((fit.popt[0] - 1.5).abs() < 1e-12);
                        assert_eq!(fit.ls_result.active_mask, [1]);
                    }
                }
            }
        }
    }
}

#[test]
fn invalid_sigma_is_rejected_before_calling_model() {
    for sigma in [
        CurveFitSigma::Scalar(0.0),
        CurveFitSigma::Scalar(f64::NAN),
        CurveFitSigma::Scalar(f64::INFINITY),
        CurveFitSigma::StandardDeviations(vec![]),
        CurveFitSigma::StandardDeviations(vec![1.0, 2.0]),
        CurveFitSigma::StandardDeviations(vec![1.0, 0.0, 1.0]),
        CurveFitSigma::Covariance(vec![vec![1.0]; 3]),
        CurveFitSigma::Covariance(vec![vec![1.0; 3]; 3]),
        CurveFitSigma::Covariance(vec![
            vec![1.0, 2.0, 0.0],
            vec![2.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]),
        CurveFitSigma::Covariance(vec![
            vec![1.0, f64::NAN, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]),
    ] {
        for bounded in [false, true] {
            let options = CurveFitOptions {
                p0: Some(vec![1.0]),
                sigma: Some(sigma.clone()),
                ..CurveFitOptions::default()
            };
            let model = |_x, _p: &[f64]| -> f64 {
                panic!("invalid sigma must fail before model evaluation")
            };
            let result = if bounded {
                curve_fit_bounded(
                    model,
                    &[0.0, 1.0, 2.0],
                    &[1.0, 2.0, 3.0],
                    &[0.0],
                    &[2.0],
                    options,
                )
            } else {
                curve_fit(model, &[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0], options)
            };
            assert!(
                matches!(
                    result,
                    Err(OptError::InvalidArgument { .. } | OptError::NonFiniteInput { .. })
                ),
                "{sigma:?}"
            );
        }
    }
}

#[test]
fn sigma_preserves_jacobian_validation() {
    for malformed in [
        vec![vec![1.0]],
        vec![vec![1.0], vec![], vec![1.0]],
        vec![vec![f64::NAN]; 3],
    ] {
        let result = curve_fit(
            |_x, p| p[0],
            &[0.0, 1.0, 2.0],
            &[1.0, 2.0, 3.0],
            CurveFitOptions {
                p0: Some(vec![0.0]),
                sigma: Some(CurveFitSigma::Covariance(correlated_covariance())),
                ls_options: LeastSquaresOptions {
                    jac: Some(LeastSquaresJacobian::new(move |_| malformed.clone())),
                    ..LeastSquaresOptions::default()
                },
                ..CurveFitOptions::default()
            },
        );
        assert!(matches!(
            result,
            Err(OptError::InvalidArgument { .. } | OptError::NonFiniteInput { .. })
        ));
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
enum OracleSigma {
    Scalar(f64),
    Deviations(Vec<f64>),
    Covariance(Vec<Vec<f64>>),
}

impl OracleSigma {
    fn options(&self) -> CurveFitSigma {
        match self {
            Self::Scalar(value) => CurveFitSigma::Scalar(*value),
            Self::Deviations(values) => CurveFitSigma::StandardDeviations(values.clone()),
            Self::Covariance(matrix) => CurveFitSigma::Covariance(matrix.clone()),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct WeightedCase {
    case_id: String,
    method: String,
    bounded: bool,
    analytic: bool,
    absolute_sigma: bool,
    robust: bool,
    sigma: OracleSigma,
    xdata: Vec<f64>,
    ydata: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct WeightedOracle {
    case_id: String,
    popt: Option<Vec<f64>>,
    pcov: Option<Vec<Vec<f64>>>,
    cost: Option<f64>,
    error: Option<String>,
}

#[derive(Deserialize)]
struct OracleRun {
    scipy_version: String,
    numpy_version: String,
    cases: Vec<WeightedOracle>,
}

fn scipy_oracle(cases: &[WeightedCase]) -> Option<OracleRun> {
    let required = std::env::var_os("FSCI_REQUIRE_SCIPY_ORACLE").is_some();
    let python = std::env::var_os("SCIPY_PYTHON").unwrap_or_else(|| "python3".into());
    let script = r#"
import json
import sys
import numpy as np
import scipy
from scipy.optimize import curve_fit

out = []
for case in json.load(sys.stdin):
    x = np.asarray(case["xdata"], dtype=float)
    y = np.asarray(case["ydata"], dtype=float)
    if case["robust"]:
        def model(x, a, b): return a * np.exp(-b * x)
        def jac(x, a, b):
            e = np.exp(-b * x)
            return np.column_stack((e, -a * x * e))
    else:
        def model(x, a, b): return a * x + b
        def jac(x, a, b): return np.column_stack((x, np.ones_like(x)))
    options = dict(p0=[1.0, 1.0], sigma=np.asarray(case["sigma"], dtype=float),
                   absolute_sigma=case["absolute_sigma"], method=case["method"],
                   ftol=1e-12, xtol=1e-12, gtol=1e-12, full_output=True)
    if case["analytic"]: options["jac"] = jac
    if case["bounded"]: options["bounds"] = ([0.0, -10.0], [1.5, 10.0])
    if case["robust"]: options.update(loss="soft_l1", f_scale=0.2)
    try:
        popt, pcov, info, message, status = curve_fit(model, x, y, **options)
        residual = info["fvec"]
        cost = (0.04 * np.sum(np.sqrt(1 + (residual / 0.2)**2) - 1)
                if case["robust"] else 0.5 * np.dot(residual, residual))
        out.append(dict(case_id=case["case_id"], popt=popt.tolist(), pcov=pcov.tolist(), cost=cost, error=None))
    except Exception as error:
        out.append(dict(case_id=case["case_id"], popt=None, pcov=None, cost=None, error=str(error)))
print(json.dumps(dict(scipy_version=scipy.__version__, numpy_version=np.__version__, cases=out), allow_nan=False))
"#;
    let child = Command::new(&python)
        .args(["-c", script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(child) => child,
        Err(error) => {
            assert!(
                !required,
                "required SciPy oracle {python:?} failed to start: {error}"
            );
            eprintln!("SKIP weighted curve-fit oracle: {error}");
            return None;
        }
    };
    let input = serde_json::to_vec(cases).expect("serialize weighted-fit inputs");
    let write_result = child.stdin.take().expect("oracle stdin").write_all(&input);
    let output = child
        .wait_with_output()
        .expect("wait for weighted-fit oracle");
    if write_result.is_err() || !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !required,
            "required SciPy oracle failed: {write_result:?}; {stderr}"
        );
        eprintln!("SKIP weighted curve-fit oracle: {write_result:?}; {stderr}");
        return None;
    }
    let oracle: OracleRun = serde_json::from_slice(&output.stdout).expect("parse oracle response");
    if required {
        assert_eq!(
            oracle.scipy_version, "1.17.1",
            "required SciPy oracle version"
        );
        assert_eq!(
            oracle.numpy_version, "2.4.3",
            "required NumPy oracle version"
        );
    }
    eprintln!(
        "weighted curve-fit oracle: scipy={} numpy={}",
        oracle.scipy_version, oracle.numpy_version
    );
    Some(oracle)
}

#[test]
fn diff_opt_curve_fit_measurement_uncertainty() {
    let xdata: Vec<f64> = (0..6).map(|i| 0.5 * f64::from(i)).collect();
    let deviations = [0.2, 0.3, 0.5, 0.8, 1.3, 2.0];
    let covariance: Vec<Vec<f64>> = (0_usize..6)
        .map(|i| {
            (0_usize..6)
                .map(|j| deviations[i] * deviations[j] * 0.5_f64.powi(i.abs_diff(j) as i32))
                .collect()
        })
        .collect();
    let sigmas = [
        ("scalar", OracleSigma::Scalar(1.7)),
        ("diagonal", OracleSigma::Deviations(deviations.to_vec())),
        ("correlated", OracleSigma::Covariance(covariance)),
    ];
    let mut cases = Vec::new();
    for method in ["lm", "trf", "dogbox"] {
        for (name, sigma) in &sigmas {
            for analytic in [false, true] {
                for absolute_sigma in [false, true] {
                    // SciPy 1.17.1's analytic wrapper mistakes scalar sigma for
                    // a Cholesky matrix. Use its singleton-vector broadcast arm;
                    // scalar+analytic is tested above against closed-form values.
                    let (name, sigma) = if *name == "scalar" && analytic {
                        ("singleton", OracleSigma::Deviations(vec![1.7]))
                    } else {
                        (*name, sigma.clone())
                    };
                    cases.push(WeightedCase {
                        case_id: format!(
                            "{method}_{name}_analytic{analytic}_absolute{absolute_sigma}"
                        ),
                        method: method.to_string(),
                        bounded: false,
                        analytic,
                        absolute_sigma,
                        robust: false,
                        sigma,
                        xdata: xdata.clone(),
                        ydata: vec![1.2, 2.2, 2.9, 4.4, 4.8, 7.5],
                    });
                }
            }
        }
    }
    for method in ["trf", "dogbox"] {
        for (name, sigma) in &sigmas[1..] {
            for analytic in [false, true] {
                for absolute_sigma in [false, true] {
                    cases.push(WeightedCase {
                        case_id: format!(
                            "{method}_{name}_bound_analytic{analytic}_absolute{absolute_sigma}"
                        ),
                        method: method.to_string(),
                        bounded: true,
                        analytic,
                        absolute_sigma,
                        robust: false,
                        sigma: sigma.clone(),
                        xdata: xdata.clone(),
                        ydata: vec![1.2, 2.2, 2.9, 4.4, 4.8, 7.5],
                    });
                }
                cases.push(WeightedCase {
                    case_id: format!("{method}_{name}_nonlinear_robust_analytic{analytic}"),
                    method: method.to_string(),
                    bounded: false,
                    analytic,
                    absolute_sigma: true,
                    robust: true,
                    sigma: sigma.clone(),
                    xdata: xdata.clone(),
                    ydata: xdata
                        .iter()
                        .enumerate()
                        .map(|(i, x)| 2.0 * (-0.7 * x).exp() + if i == 4 { 3.0 } else { 0.0 })
                        .collect(),
                });
            }
        }
    }
    let Some(oracle) = scipy_oracle(&cases) else {
        return;
    };
    assert_eq!(oracle.cases.len(), cases.len());
    let mut compared = 0;
    for (case, expected) in cases.iter().zip(&oracle.cases) {
        assert_eq!(case.case_id, expected.case_id);
        assert!(
            expected.error.is_none(),
            "{}: {:?}",
            case.case_id,
            expected.error
        );
        let method = match case.method.as_str() {
            "lm" => LeastSquaresMethod::Lm,
            "trf" => LeastSquaresMethod::Trf,
            _ => LeastSquaresMethod::Dogbox,
        };
        let robust = case.robust;
        let jac_x = case.xdata.clone();
        let options = CurveFitOptions {
            p0: Some(vec![1.0, 1.0]),
            sigma: Some(case.sigma.options()),
            absolute_sigma: case.absolute_sigma,
            ls_options: LeastSquaresOptions {
                method: Some(method),
                jac: case.analytic.then(|| {
                    LeastSquaresJacobian::new(move |p| {
                        jac_x
                            .iter()
                            .map(|&x| {
                                if robust {
                                    let e = (-p[1] * x).exp();
                                    vec![e, -p[0] * x * e]
                                } else {
                                    vec![x, 1.0]
                                }
                            })
                            .collect()
                    })
                }),
                loss: if robust {
                    LossKind::SoftL1
                } else {
                    LossKind::Linear
                },
                f_scale: 0.2,
                gtol: 1e-12,
                xtol: 1e-12,
                ftol: 1e-12,
                max_nfev: Some(1000),
                ..LeastSquaresOptions::default()
            },
        };
        let model = |x: f64, p: &[f64]| {
            if robust {
                p[0] * (-p[1] * x).exp()
            } else {
                p[0] * x + p[1]
            }
        };
        let fit = if case.bounded {
            curve_fit_bounded(
                model,
                &case.xdata,
                &case.ydata,
                &[0.0, -10.0],
                &[1.5, 10.0],
                options,
            )
        } else {
            curve_fit(model, &case.xdata, &case.ydata, options)
        }
        .unwrap_or_else(|error| panic!("{}: {error}", case.case_id));
        let popt = expected.popt.as_ref().expect("oracle fit");
        let pcov = expected.pcov.as_ref().expect("oracle covariance");
        let cost = expected.cost.expect("oracle cost");
        assert_eq!(fit.popt.len(), popt.len());
        assert_eq!(fit.pcov.len(), pcov.len());
        for (&got, &want) in fit.popt.iter().zip(popt) {
            assert!(
                (got - want).abs() <= 1e-6 * want.abs().max(1e-6),
                "{} parameters {:?} vs {:?}",
                case.case_id,
                fit.popt,
                popt
            );
        }
        for (got_row, want_row) in fit.pcov.iter().zip(pcov) {
            assert_eq!(got_row.len(), want_row.len());
            for (&got, &want) in got_row.iter().zip(want_row) {
                assert!(
                    (got - want).abs() <= 1e-6 * want.abs().max(1e-12),
                    "{} covariance {:?} vs {:?}",
                    case.case_id,
                    fit.pcov,
                    pcov
                );
            }
        }
        assert!(
            (fit.ls_result.cost - cost).abs() <= 1e-8 * cost.abs().max(1e-12),
            "{} cost {} vs {}",
            case.case_id,
            fit.ls_result.cost,
            cost
        );
        println!(
            "{} compared: popt={:?}, cost={}, nfev={}, status={}",
            case.case_id, fit.popt, fit.ls_result.cost, fit.ls_result.nfev, fit.ls_result.status
        );
        compared += 1;
    }
    assert_eq!(
        compared, 60,
        "all scalar/diagonal/correlated, bounded, analytic, variance and robust cases must compare"
    );
    eprintln!("weighted curve-fit compared_cases={compared}");
}

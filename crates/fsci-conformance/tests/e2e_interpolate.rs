#![forbid(unsafe_code)]
//! E2E scenario tests for FSCI-P2C-014 (Interpolate).
//!
//! Implements conformance tests for scipy.interpolate parity:
//!   Happy-path:  1-5, 15-16 (basic interpolation scenarios)
//!   Error recovery: 6-8 (invalid input handling)
//!   Cross-op consistency: 9-11 (round-trip and derivative checks)
//!   Performance boundary: 12-14 (large grid, high-dimension)
//!
//! Each scenario emits a forensic log bundle to
//! `fixtures/artifacts/FSCI-P2C-014/e2e/runs/standalone/{scenario_id}/summary.json`.

use fsci_interpolate::{
    Akima1DInterpolator, CubicSplineStandalone, Interp1d, Interp1dOptions, InterpError, InterpKind,
    PchipInterpolator, RectBivariateSpline, RegularGridInterpolator, RegularGridMethod,
    SmoothBivariateSpline, SmoothBivariateSplineOptions, SplineBc, interp1d_linear,
    make_interp_spline, polyfit, polyval, splev_with_derivative,
};
use fsci_runtime::RuntimeMode;
use serde::Serialize;
use std::f64::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const INTERPOLATE_PACKET_ID: &str = "FSCI-P2C-014";
const STANDALONE_RUN_ID: &str = "standalone";

// ───────────────────────── Forensic log types ─────────────────────────

#[derive(Debug, Clone, Serialize)]
struct ForensicLogBundle {
    scenario_id: String,
    steps: Vec<ForensicStep>,
    artifacts: Vec<ArtifactRef>,
    environment: EnvironmentInfo,
    interpolator_metadata: Option<InterpolatorMetadata>,
    overall: OverallResult,
}

#[derive(Debug, Clone, Serialize)]
struct ForensicStep {
    step_id: usize,
    step_name: String,
    action: String,
    input_summary: String,
    output_summary: String,
    duration_ns: u128,
    mode: String,
    outcome: String,
}

#[derive(Debug, Clone, Serialize)]
struct ArtifactRef {
    path: String,
    blake3: String,
}

#[derive(Debug, Clone, Serialize)]
struct EnvironmentInfo {
    rust_version: String,
    os: String,
    cpu_count: usize,
    total_memory_mb: String,
}

#[derive(Debug, Clone, Serialize)]
struct InterpolatorMetadata {
    kind: String,
    num_points: usize,
    dimensions: usize,
    mode: String,
}

#[derive(Debug, Clone, Serialize)]
struct OverallResult {
    status: String,
    total_duration_ns: u128,
    replay_command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_chain: Option<String>,
}

// ───────────────────────── Helpers ─────────────────────────

fn e2e_output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/artifacts")
        .join(INTERPOLATE_PACKET_ID)
        .join("e2e/runs")
        .join(STANDALONE_RUN_ID)
}

fn make_env() -> EnvironmentInfo {
    EnvironmentInfo {
        rust_version: String::from(env!("CARGO_PKG_VERSION")),
        os: String::from(std::env::consts::OS),
        cpu_count: std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1),
        total_memory_mb: String::from("unknown"),
    }
}

fn replay_cmd(scenario_id: &str) -> String {
    format!(
        "rch exec -- env CARGO_TARGET_DIR=/tmp/rch_target_fsci_conformance_cod cargo test -p fsci-conformance --test e2e_interpolate -- {scenario_id} --nocapture"
    )
}

fn write_bundle(scenario_id: &str, bundle: &ForensicLogBundle) {
    let dir = e2e_output_dir().join(scenario_id);
    fs::create_dir_all(&dir).expect("create interpolate e2e artifact directory");
    let path = dir.join("summary.json");
    let json = serde_json::to_vec_pretty(bundle).expect("serialize bundle");
    fs::write(&path, &json).expect("write interpolate e2e artifact bundle");
}

/// `zip` stops at the shorter side, so a short or empty result would fold to 0 and read as exact
/// agreement. A length mismatch is infinitely wrong instead.
fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, nan_max)
}

/// `f64::max` returns the other operand when one is NaN, so folding residuals with it reads a
/// NaN entry as agreement. This keeps the NaN, and `NaN <= tol` then fails the case.
fn nan_max(acc: f64, d: f64) -> f64 {
    if acc.is_nan() || d.is_nan() {
        f64::NAN
    } else {
        acc.max(d)
    }
}

#[test]
fn interpolate_e2e_artifact_paths_use_interpolate_packet_id() {
    let output_dir = e2e_output_dir();
    assert!(output_dir.ends_with(Path::new(
        "fixtures/artifacts/FSCI-P2C-014/e2e/runs/standalone"
    )));
    assert!(!output_dir.to_string_lossy().contains("FSCI-P2C-009"));

    let replay = replay_cmd("scenario_01_interp1d_linear_exact");
    assert!(replay.contains("e2e_interpolate"));
    assert!(replay.contains("scenario_01_interp1d_linear_exact"));
}

// ───────────────────── Scenario runner framework ──────────────────────

struct ScenarioRunner {
    scenario_id: String,
    steps: Vec<ForensicStep>,
    start: Instant,
    step_counter: usize,
    passed: bool,
    error_chain: Option<String>,
    interp_meta: Option<InterpolatorMetadata>,
}

impl ScenarioRunner {
    fn new(scenario_id: &str) -> Self {
        Self {
            scenario_id: scenario_id.to_owned(),
            steps: Vec::new(),
            start: Instant::now(),
            step_counter: 0,
            passed: true,
            error_chain: None,
            interp_meta: None,
        }
    }

    fn set_interp_meta(&mut self, kind: &str, num_points: usize, dimensions: usize, mode: &str) {
        self.interp_meta = Some(InterpolatorMetadata {
            kind: kind.to_owned(),
            num_points,
            dimensions,
            mode: mode.to_owned(),
        });
    }

    fn record_step(
        &mut self,
        name: &str,
        action: &str,
        input_summary: &str,
        mode: &str,
        f: impl FnOnce() -> Result<String, String>,
    ) -> bool {
        self.step_counter += 1;
        let step_start = Instant::now();
        let result = f();
        let duration_ns = step_start.elapsed().as_nanos();
        let (outcome, output_summary) = match result {
            Ok(summary) => ("pass".to_owned(), summary),
            Err(err) => {
                self.passed = false;
                if self.error_chain.is_none() {
                    self.error_chain = Some(err.clone());
                }
                ("fail".to_owned(), err)
            }
        };
        self.steps.push(ForensicStep {
            step_id: self.step_counter,
            step_name: name.to_owned(),
            action: action.to_owned(),
            input_summary: input_summary.to_owned(),
            output_summary,
            duration_ns,
            mode: mode.to_owned(),
            outcome: outcome.clone(),
        });
        outcome == "pass"
    }

    fn finish(self) -> ForensicLogBundle {
        let total_duration_ns = self.start.elapsed().as_nanos();
        ForensicLogBundle {
            scenario_id: self.scenario_id.clone(),
            steps: self.steps,
            artifacts: Vec::new(),
            environment: make_env(),
            interpolator_metadata: self.interp_meta,
            overall: OverallResult {
                status: if self.passed { "pass" } else { "fail" }.to_owned(),
                total_duration_ns,
                replay_command: replay_cmd(&self.scenario_id),
                error_chain: self.error_chain,
            },
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//                       HAPPY-PATH SCENARIOS (1-5)
// ═══════════════════════════════════════════════════════════════════════

/// Scenario 1: Linear interpolation - basic exact-match test
/// scipy.interpolate.interp1d(x, y, kind='linear')
#[test]
fn scenario_01_interp1d_linear_exact() {
    let mut runner = ScenarioRunner::new("scenario_01_interp1d_linear_exact");
    runner.set_interp_meta("Interp1d-Linear", 5, 1, "Strict");

    // Reference data: y = 2x + 1
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![1.0, 3.0, 5.0, 7.0, 9.0];

    // Step 1: Create interpolator
    let mut interp: Option<Interp1d> = None;
    runner.record_step(
        "create_interp1d",
        "Interp1d::new(x, y, Linear)",
        "x=[0,1,2,3,4], y=[1,3,5,7,9]",
        "Strict",
        || {
            let opts = Interp1dOptions {
                kind: InterpKind::Linear,
                mode: RuntimeMode::Strict,
                ..Default::default()
            };
            match Interp1d::new(&x, &y, opts) {
                Ok(i) => {
                    interp = Some(i);
                    Ok("created linear interpolator".to_owned())
                }
                Err(e) => Err(format!("construction failed: {e}")),
            }
        },
    );

    // Step 2: Evaluate at known points
    let interp = interp.expect("interpolator should exist");
    runner.record_step(
        "eval_known_point",
        "interp.eval(1.5)",
        "x_new=1.5",
        "Strict",
        || {
            let result = interp.eval(1.5).map_err(|e| format!("{e}"))?;
            let expected = 4.0; // 2*1.5 + 1
            if (result - expected).abs() < 1e-12 {
                Ok(format!("result={result}, expected={expected}"))
            } else {
                Err(format!("result={result}, expected={expected}"))
            }
        },
    );

    // Step 3: Evaluate at multiple points
    runner.record_step(
        "eval_many",
        "interp.eval_many([0.5, 1.5, 2.5, 3.5])",
        "x_new=[0.5,1.5,2.5,3.5]",
        "Strict",
        || {
            let x_new = vec![0.5, 1.5, 2.5, 3.5];
            let expected = vec![2.0, 4.0, 6.0, 8.0];
            let result = interp.eval_many(&x_new).map_err(|e| format!("{e}"))?;
            let diff = max_abs_diff(&result, &expected);
            if diff < 1e-12 {
                Ok(format!("max_diff={diff:.2e}"))
            } else {
                Err(format!("max_diff={diff:.2e} exceeds tolerance"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_01_interp1d_linear_exact", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_01 failed");
}

/// Scenario 2: Cubic spline interpolation with natural BCs
/// scipy.interpolate.CubicSpline(x, y, bc_type='natural')
#[test]
fn scenario_02_cubic_spline_natural() {
    let mut runner = ScenarioRunner::new("scenario_02_cubic_spline_natural");
    runner.set_interp_meta("CubicSpline-Natural", 10, 1, "Strict");

    // Reference: sin function
    let n = 10;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * PI / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&t| t.sin()).collect();

    let mut spline: Option<CubicSplineStandalone> = None;
    runner.record_step(
        "create_cubic_spline",
        "CubicSplineStandalone::new(x, y, Natural)",
        &format!("n={n} points on sin(x)"),
        "Strict",
        || match CubicSplineStandalone::new(&x, &y, SplineBc::Natural) {
            Ok(s) => {
                spline = Some(s);
                Ok("created natural cubic spline".to_owned())
            }
            Err(e) => Err(format!("construction failed: {e}")),
        },
    );

    let spline = spline.expect("spline should exist");
    runner.record_step(
        "eval_midpoint",
        "spline.eval(PI/4)",
        "x_new=PI/4",
        "Strict",
        || {
            let result = spline.eval(PI / 4.0);
            let expected = (PI / 4.0).sin();
            let err = (result - expected).abs();
            // Cubic spline should be very accurate for smooth functions
            if err < 1e-3 {
                Ok(format!(
                    "result={result:.6}, expected={expected:.6}, err={err:.2e}"
                ))
            } else {
                Err(format!("err={err:.2e} exceeds tolerance"))
            }
        },
    );

    // Verify derivative endpoint conditions (natural = S''=0 at ends)
    runner.record_step(
        "check_natural_bc",
        "verify second derivative near zero at endpoints",
        "S''(0) and S''(PI) should be ~0",
        "Strict",
        || {
            let deriv2 = spline.derivative(2);
            let d2_start = deriv2.eval(0.0);
            let d2_end = deriv2.eval(PI);
            if d2_start.abs() < 1e-8 && d2_end.abs() < 1e-8 {
                Ok(format!("S''(0)={d2_start:.2e}, S''(PI)={d2_end:.2e}"))
            } else {
                Err(format!(
                    "S''(0)={d2_start:.2e}, S''(PI)={d2_end:.2e} not near zero"
                ))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_02_cubic_spline_natural", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_02 failed");
}

/// Scenario 3: PCHIP monotonicity preservation
/// scipy.interpolate.PchipInterpolator(x, y)
#[test]
fn scenario_03_pchip_monotonic() {
    let mut runner = ScenarioRunner::new("scenario_03_pchip_monotonic");
    runner.set_interp_meta("PCHIP", 5, 1, "Strict");

    // Monotonically increasing data
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![0.0, 1.0, 4.0, 9.0, 16.0]; // x^2

    let mut pchip: Option<PchipInterpolator> = None;
    runner.record_step(
        "create_pchip",
        "PchipInterpolator::new(x, y)",
        "monotonic data",
        "Strict",
        || match PchipInterpolator::new(&x, &y) {
            Ok(p) => {
                pchip = Some(p);
                Ok("created PCHIP interpolator".to_owned())
            }
            Err(e) => Err(format!("construction failed: {e}")),
        },
    );

    let pchip = pchip.expect("pchip should exist");
    runner.record_step(
        "verify_monotonicity",
        "check interpolated values are monotonic",
        "eval at 100 points",
        "Strict",
        || {
            let pts: Vec<f64> = (0..=100).map(|i| i as f64 * 4.0 / 100.0).collect();
            let vals: Vec<f64> = pts.iter().map(|&t| pchip.eval(t)).collect();
            let monotonic = vals.windows(2).all(|w| w[1] >= w[0]);
            if monotonic {
                Ok("all interpolated values monotonically increasing".to_owned())
            } else {
                Err("monotonicity violated".to_owned())
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_03_pchip_monotonic", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_03 failed");
}

/// Scenario 4: RegularGridInterpolator 2D bilinear
/// scipy.interpolate.RegularGridInterpolator((x, y), values, method='linear')
#[test]
fn scenario_04_regular_grid_2d() {
    let mut runner = ScenarioRunner::new("scenario_04_regular_grid_2d");
    runner.set_interp_meta("RegularGridInterpolator-Linear", 9, 2, "Strict");

    // 3x3 grid: f(x,y) = x + y
    let points = vec![vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]];
    let mut values = Vec::new();
    for &x in &points[0] {
        for &y in &points[1] {
            values.push(x + y);
        }
    }

    let mut interp: Option<RegularGridInterpolator> = None;
    runner.record_step(
        "create_regular_grid",
        "RegularGridInterpolator::new(points, values, Linear)",
        "3x3 grid, f(x,y)=x+y",
        "Strict",
        || match RegularGridInterpolator::new(
            points.clone(),
            values.clone(),
            RegularGridMethod::Linear,
            false,
            None,
        ) {
            Ok(i) => {
                interp = Some(i);
                Ok("created 2D regular grid interpolator".to_owned())
            }
            Err(e) => Err(format!("construction failed: {e}")),
        },
    );

    let interp = interp.expect("interpolator should exist");
    runner.record_step(
        "eval_center",
        "interp.eval([1.0, 1.0])",
        "center of grid",
        "Strict",
        || {
            let result = interp.eval(&[1.0, 1.0]).map_err(|e| format!("{e}"))?;
            let expected = 2.0;
            if (result - expected).abs() < 1e-12 {
                Ok(format!("result={result}"))
            } else {
                Err(format!("result={result}, expected={expected}"))
            }
        },
    );

    runner.record_step(
        "eval_interior",
        "interp.eval([0.5, 0.5])",
        "interior point",
        "Strict",
        || {
            let result = interp.eval(&[0.5, 0.5]).map_err(|e| format!("{e}"))?;
            let expected = 1.0; // 0.5 + 0.5
            if (result - expected).abs() < 1e-12 {
                Ok(format!("result={result}"))
            } else {
                Err(format!("result={result}, expected={expected}"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_04_regular_grid_2d", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_04 failed");
}

/// Scenario 5: Akima interpolator smooth derivative
/// scipy.interpolate.Akima1DInterpolator(x, y)
#[test]
fn scenario_05_akima_smoothness() {
    let mut runner = ScenarioRunner::new("scenario_05_akima_smoothness");
    runner.set_interp_meta("Akima", 10, 1, "Strict");

    // Data with potential for oscillation (runge-like)
    let x: Vec<f64> = (-5..=5).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&t| 1.0 / (1.0 + t * t)).collect();

    let mut akima: Option<Akima1DInterpolator> = None;
    runner.record_step(
        "create_akima",
        "Akima1DInterpolator::new(x, y)",
        "Runge function data",
        "Strict",
        || match Akima1DInterpolator::new(&x, &y) {
            Ok(a) => {
                akima = Some(a);
                Ok("created Akima interpolator".to_owned())
            }
            Err(e) => Err(format!("construction failed: {e}")),
        },
    );

    let akima = akima.expect("akima should exist");
    runner.record_step(
        "eval_no_overshoot",
        "verify no wild oscillations",
        "check values stay bounded",
        "Strict",
        || {
            let pts: Vec<f64> = (-50..=50).map(|i| i as f64 / 10.0).collect();
            let vals: Vec<f64> = pts.iter().map(|&t| akima.eval(t)).collect();
            let max_val = vals.iter().cloned().fold(f64::NEG_INFINITY, nan_max);
            let min_val = vals.iter().cloned().fold(f64::INFINITY, f64::min);
            // Akima should not overshoot badly - values should be reasonable
            if max_val <= 1.5 && min_val >= -0.5 {
                Ok(format!("range=[{min_val:.3}, {max_val:.3}]"))
            } else {
                Err(format!(
                    "excessive oscillation: range=[{min_val:.3}, {max_val:.3}]"
                ))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_05_akima_smoothness", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_05 failed");
}

/// Scenario 15: RectBivariateSpline uses SciPy's x-major z shape
/// scipy.interpolate.RectBivariateSpline(x, y, z, kx=1, ky=1)
#[test]
fn scenario_15_rect_bivariate_spline_scipy_shape() {
    let mut runner = ScenarioRunner::new("scenario_15_rect_bivariate_spline_scipy_shape");
    runner.set_interp_meta("RectBivariateSpline", 12, 2, "Strict");

    let x = vec![0.0, 1.0, 2.0, 3.0];
    let y = vec![0.0, 1.0, 2.0];
    let z: Vec<Vec<f64>> = x
        .iter()
        .map(|&xv| y.iter().map(|&yv| 10.0 * xv + yv).collect())
        .collect();

    let mut spline: Option<RectBivariateSpline> = None;
    runner.record_step(
        "create_rect_bivariate_spline",
        "RectBivariateSpline::new(x, y, z, kx=1, ky=1)",
        "x has 4 points, y has 3 points, z shape is (4, 3)",
        "Strict",
        || match RectBivariateSpline::new(&x, &y, &z, 1, 1) {
            Ok(s) => {
                spline = Some(s);
                Ok("created bilinear RectBivariateSpline".to_owned())
            }
            Err(e) => Err(format!("construction failed: {e}")),
        },
    );

    let spline = spline.expect("spline should exist");
    runner.record_step(
        "eval_scalar_orientation",
        "spline.eval(1.5, 0.5)",
        "asymmetric plane f(x,y)=10x+y",
        "Strict",
        || {
            let result = spline.eval(1.5, 0.5);
            let expected = 15.5;
            let err = (result - expected).abs();
            if err < 1e-10 {
                Ok(format!("result={result}, expected={expected}"))
            } else {
                Err(format!("result={result}, expected={expected}, err={err}"))
            }
        },
    );

    runner.record_step(
        "eval_grid_orientation",
        "spline.eval_grid([0.5, 1.5], [0.5, 1.5])",
        "grid=True equivalent should be x-major",
        "Strict",
        || {
            let result = spline.eval_grid(&[0.5, 1.5], &[0.5, 1.5]);
            let expected = [vec![5.5, 6.5], vec![15.5, 16.5]];
            // A missing row or a short row reads as INFINITY, not as a small error.
            let max_err = if result.len() == expected.len() {
                result
                    .iter()
                    .zip(expected.iter())
                    .map(|(got_row, expected_row)| max_abs_diff(got_row, expected_row))
                    .fold(0.0_f64, nan_max)
            } else {
                f64::INFINITY
            };
            if max_err < 1e-10 {
                Ok(format!("max_grid_error={max_err:.2e}"))
            } else {
                Err(format!("max_grid_error={max_err:.2e}, result={result:?}"))
            }
        },
    );

    runner.record_step(
        "integral_orientation",
        "spline.integral(0, 1, 0, 1)",
        "integral of 10x+y over unit square",
        "Strict",
        || {
            let result = spline.integral(0.0, 1.0, 0.0, 1.0);
            let expected = 5.5;
            let err = (result - expected).abs();
            if err < 1e-10 {
                Ok(format!("integral={result}, expected={expected}"))
            } else {
                Err(format!("integral={result}, expected={expected}, err={err}"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_15_rect_bivariate_spline_scipy_shape", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_15 failed");
}

/// Scenario 16: SmoothBivariateSpline fits scattered bilinear data
/// scipy.interpolate.SmoothBivariateSpline(x, y, z, kx=1, ky=1, s=0)
#[test]
fn scenario_16_smooth_bivariate_spline_scattered_surface() {
    let mut runner = ScenarioRunner::new("scenario_16_smooth_bivariate_spline_scattered_surface");
    runner.set_interp_meta("SmoothBivariateSpline", 16, 2, "Strict");

    let x = vec![0.0, 1.0, 0.0, 1.0, 0.5, 0.25];
    let y = vec![0.0, 0.0, 1.0, 1.0, 0.5, 0.75];
    let z: Vec<f64> = x
        .iter()
        .zip(&y)
        .map(|(&xv, &yv)| 2.0 + 3.0 * xv - 4.0 * yv + 5.0 * xv * yv)
        .collect();
    let options = SmoothBivariateSplineOptions {
        kx: 1,
        ky: 1,
        smoothing: Some(0.0),
        ..SmoothBivariateSplineOptions::default()
    };

    let mut spline: Option<SmoothBivariateSpline> = None;
    runner.record_step(
        "create_smooth_bivariate_spline",
        "SmoothBivariateSpline::new(x, y, z, kx=1, ky=1, s=0)",
        "six scattered samples from f(x,y)=2+3x-4y+5xy",
        "Strict",
        || match SmoothBivariateSpline::new(&x, &y, &z, options) {
            Ok(s) => {
                spline = Some(s);
                Ok("created SmoothBivariateSpline".to_owned())
            }
            Err(e) => Err(format!("construction failed: {e}")),
        },
    );

    let spline = spline.expect("spline should exist");
    runner.record_step(
        "eval_scalar_scattered_surface",
        "spline.eval(0.25, 0.5)",
        "bilinear surface exact value",
        "Strict",
        || {
            let result = spline.eval(0.25, 0.5);
            let expected = 1.375;
            let err = (result - expected).abs();
            if err < 1e-10 {
                Ok(format!("result={result}, expected={expected}"))
            } else {
                Err(format!("result={result}, expected={expected}, err={err}"))
            }
        },
    );

    runner.record_step(
        "derivative_and_integral",
        "spline.eval_derivative(...), spline.integral(0, 1, 0, 1)",
        "bilinear derivatives and unit-square integral",
        "Strict",
        || {
            let dx = spline.eval_derivative(0.25, 0.5, 1, 0);
            let dy = spline.eval_derivative(0.25, 0.5, 0, 1);
            let integral = spline.integral(0.0, 1.0, 0.0, 1.0);
            let max_err = nan_max(
                nan_max((dx - 5.5).abs(), (dy + 2.75).abs()),
                (integral - 2.75).abs(),
            );
            if max_err < 1e-10 {
                Ok(format!(
                    "dx={dx}, dy={dy}, integral={integral}, max_err={max_err:.2e}"
                ))
            } else {
                Err(format!(
                    "dx={dx}, dy={dy}, integral={integral}, max_err={max_err:.2e}"
                ))
            }
        },
    );

    runner.record_step(
        "residual_and_metadata",
        "spline.residual(), spline.degrees(), spline.bbox()",
        "exact scattered bilinear fit should have near-zero residual",
        "Strict",
        || {
            let residual = spline.residual();
            if residual < 1e-18
                && spline.degrees() == (1, 1)
                && spline.bbox() == [0.0, 1.0, 0.0, 1.0]
            {
                Ok(format!("residual={residual:.2e}"))
            } else {
                Err(format!(
                    "residual={residual:.2e}, degrees={:?}, bbox={:?}",
                    spline.degrees(),
                    spline.bbox()
                ))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle(
        "scenario_16_smooth_bivariate_spline_scattered_surface",
        &bundle,
    );
    assert!(bundle.overall.status == "pass", "scenario_16 failed");
}

/// Scenario 17: SmoothBivariateSpline constructs a piecewise linear surface
/// from scattered samples instead of collapsing to a single global polynomial.
#[test]
fn scenario_17_smooth_bivariate_spline_piecewise_surface() {
    let mut runner = ScenarioRunner::new("scenario_17_smooth_bivariate_spline_piecewise_surface");
    runner.set_interp_meta("SmoothBivariateSpline", 17, 2, "Strict");

    let x = vec![0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0];
    let y = vec![0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0];
    let z: Vec<f64> = x
        .iter()
        .zip(&y)
        .map(|(&xv, &yv)| (xv - 0.5_f64).abs() + (yv - 0.5_f64).abs())
        .collect();
    let options = SmoothBivariateSplineOptions {
        kx: 1,
        ky: 1,
        smoothing: Some(0.0),
        ..SmoothBivariateSplineOptions::default()
    };

    let mut spline: Option<SmoothBivariateSpline> = None;
    runner.record_step(
        "create_piecewise_smooth_bivariate_spline",
        "SmoothBivariateSpline::new(x, y, z, kx=1, ky=1, s=0)",
        "nine scattered samples from |x-0.5| + |y-0.5|",
        "Strict",
        || match SmoothBivariateSpline::new(&x, &y, &z, options) {
            Ok(s) => {
                spline = Some(s);
                Ok("created piecewise SmoothBivariateSpline".to_owned())
            }
            Err(e) => Err(format!("construction failed: {e}")),
        },
    );

    let spline = spline.expect("spline should exist");
    runner.record_step(
        "eval_piecewise_surface",
        "spline.eval(0.25, 0.75), spline.integral(0, 1, 0, 1)",
        "piecewise tent surface value and integral",
        "Strict",
        || {
            let value = spline.eval(0.25, 0.75);
            let integral = spline.integral(0.0, 1.0, 0.0, 1.0);
            let value_err = (value - 0.5).abs();
            let integral_err = (integral - 0.5).abs();
            if value_err < 1e-10 && integral_err < 3e-2 {
                Ok(format!(
                    "value={value}, integral={integral}, value_err={value_err:.2e}, integral_err={integral_err:.2e}"
                ))
            } else {
                Err(format!(
                    "value={value}, integral={integral}, value_err={value_err:.2e}, integral_err={integral_err:.2e}"
                ))
            }
        },
    );

    runner.record_step(
        "piecewise_metadata",
        "spline.knots(), spline.coefficients()",
        "piecewise spline should expose interior knots and more than four coefficients",
        "Strict",
        || {
            let (tx, ty) = spline.knots();
            let coeff_count = spline.coefficients().len();
            if tx.len() > 4 && ty.len() > 4 && coeff_count > 4 {
                Ok(format!(
                    "tx_len={}, ty_len={}, coeff_count={coeff_count}",
                    tx.len(),
                    ty.len()
                ))
            } else {
                Err(format!(
                    "tx_len={}, ty_len={}, coeff_count={coeff_count}",
                    tx.len(),
                    ty.len()
                ))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle(
        "scenario_17_smooth_bivariate_spline_piecewise_surface",
        &bundle,
    );
    assert!(bundle.overall.status == "pass", "scenario_17 failed");
}

// ═══════════════════════════════════════════════════════════════════════
//                       ERROR RECOVERY SCENARIOS (6-8)
// ═══════════════════════════════════════════════════════════════════════

/// Scenario 6: Reject non-finite coordinates in construction
#[test]
fn scenario_06_reject_nan_coordinates() {
    let mut runner = ScenarioRunner::new("scenario_06_reject_nan_coordinates");
    runner.set_interp_meta("error-handling", 3, 1, "Strict");

    // NaN in x coordinates
    runner.record_step(
        "reject_nan_x",
        "Interp1d::new([1, NaN, 3], [0,1,2])",
        "NaN in x",
        "Strict",
        || {
            let x = vec![1.0, f64::NAN, 3.0];
            let y = vec![0.0, 1.0, 2.0];
            match Interp1d::new(&x, &y, Interp1dOptions::default()) {
                Err(InterpError::NonFiniteX) => Ok("correctly rejected with NonFiniteX".to_owned()),
                Err(e) => Err(format!("wrong error type: {e}")),
                Ok(_) => Err("should have rejected NaN".to_owned()),
            }
        },
    );

    // Inf in x coordinates
    runner.record_step(
        "reject_inf_x",
        "Interp1d::new([1, INF, 3], [0,1,2])",
        "Inf in x",
        "Strict",
        || {
            let x = vec![1.0, f64::INFINITY, 3.0];
            let y = vec![0.0, 1.0, 2.0];
            match Interp1d::new(&x, &y, Interp1dOptions::default()) {
                Err(InterpError::NonFiniteX) => Ok("correctly rejected with NonFiniteX".to_owned()),
                Err(e) => Err(format!("wrong error type: {e}")),
                Ok(_) => Err("should have rejected Inf".to_owned()),
            }
        },
    );

    // RegularGridInterpolator - NaN in points
    runner.record_step(
        "reject_nan_regulargrid",
        "RegularGridInterpolator with NaN in points",
        "NaN in axis",
        "Strict",
        || {
            let points = vec![vec![0.0, f64::NAN, 2.0]];
            let values = vec![0.0, 1.0, 2.0];
            match RegularGridInterpolator::new(
                points,
                values,
                RegularGridMethod::Linear,
                false,
                None,
            ) {
                Err(InterpError::NonFiniteX) => Ok("correctly rejected with NonFiniteX".to_owned()),
                Err(e) => Err(format!("wrong error type: {e}")),
                Ok(_) => Err("should have rejected NaN".to_owned()),
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_06_reject_nan_coordinates", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_06 failed");
}

/// Scenario 7: Reject unsorted x coordinates
#[test]
fn scenario_07_reject_unsorted() {
    let mut runner = ScenarioRunner::new("scenario_07_reject_unsorted");
    runner.set_interp_meta("error-handling", 4, 1, "Strict");

    runner.record_step(
        "reject_unsorted_x",
        "Interp1d::new([1, 3, 2, 4], [0,1,2,3])",
        "unsorted x",
        "Strict",
        || {
            let x = vec![1.0, 3.0, 2.0, 4.0];
            let y = vec![0.0, 1.0, 2.0, 3.0];
            match Interp1d::new(&x, &y, Interp1dOptions::default()) {
                Err(InterpError::UnsortedX) => Ok("correctly rejected with UnsortedX".to_owned()),
                Err(e) => Err(format!("wrong error type: {e}")),
                Ok(_) => Err("should have rejected unsorted".to_owned()),
            }
        },
    );

    runner.record_step(
        "reject_duplicate_x",
        "Interp1d::new([1, 2, 2, 4], [0,1,2,3])",
        "duplicate x values",
        "Strict",
        || {
            let x = vec![1.0, 2.0, 2.0, 4.0];
            let y = vec![0.0, 1.0, 2.0, 3.0];
            match Interp1d::new(&x, &y, Interp1dOptions::default()) {
                Err(InterpError::UnsortedX) => {
                    Ok("correctly rejected duplicates with UnsortedX".to_owned())
                }
                Err(e) => Err(format!("wrong error type: {e}")),
                Ok(_) => Err("should have rejected duplicates".to_owned()),
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_07_reject_unsorted", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_07 failed");
}

/// Scenario 8: Too few points
#[test]
fn scenario_08_too_few_points() {
    let mut runner = ScenarioRunner::new("scenario_08_too_few_points");
    runner.set_interp_meta("error-handling", 1, 1, "Strict");

    runner.record_step(
        "reject_single_point_linear",
        "Interp1d::new([1], [0], Linear)",
        "single point",
        "Strict",
        || {
            let x = vec![1.0];
            let y = vec![0.0];
            match Interp1d::new(&x, &y, Interp1dOptions::default()) {
                Err(InterpError::TooFewPoints { minimum, actual }) => {
                    Ok(format!("rejected: need {minimum}, got {actual}"))
                }
                Err(e) => Err(format!("wrong error type: {e}")),
                Ok(_) => Err("should have rejected single point".to_owned()),
            }
        },
    );

    runner.record_step(
        "reject_cubic_with_3_points",
        "Interp1d::new([1,2,3], [0,1,2], CubicSpline)",
        "too few for cubic",
        "Strict",
        || {
            let x = vec![1.0, 2.0, 3.0];
            let y = vec![0.0, 1.0, 2.0];
            let opts = Interp1dOptions {
                kind: InterpKind::CubicSpline,
                ..Default::default()
            };
            match Interp1d::new(&x, &y, opts) {
                Err(InterpError::TooFewPoints { minimum, actual }) => {
                    Ok(format!("rejected: need {minimum}, got {actual}"))
                }
                Err(e) => Err(format!("wrong error type: {e}")),
                Ok(_) => Err("should have rejected 3 points for cubic".to_owned()),
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_08_too_few_points", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_08 failed");
}

// ═══════════════════════════════════════════════════════════════════════
//                   CROSS-OP CONSISTENCY SCENARIOS (9-11)
// ═══════════════════════════════════════════════════════════════════════

/// Scenario 9: Polynomial round-trip (fit then eval)
#[test]
fn scenario_09_polyfit_polyval_roundtrip() {
    let mut runner = ScenarioRunner::new("scenario_09_polyfit_polyval_roundtrip");
    runner.set_interp_meta("polyfit-polyval", 5, 1, "Strict");

    // Fit degree-2 polynomial to exact quadratic data
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y: Vec<f64> = x.iter().map(|&t| 2.0 * t * t + 3.0 * t + 1.0).collect();

    let mut coeffs: Option<Vec<f64>> = None;
    runner.record_step(
        "polyfit",
        "polyfit(x, y, deg=2)",
        "fit quadratic",
        "Strict",
        || match polyfit(&x, &y, 2) {
            Ok(c) => {
                coeffs = Some(c.clone());
                Ok(format!("coeffs={c:?}"))
            }
            Err(e) => Err(format!("polyfit failed: {e}")),
        },
    );

    let coeffs = coeffs.expect("coeffs should exist");
    runner.record_step(
        "polyval_roundtrip",
        "polyval(coeffs, x) matches y",
        "evaluate fitted polynomial",
        "Strict",
        || {
            let y_fit: Vec<f64> = x.iter().map(|&t| polyval(&coeffs, t)).collect();
            let diff = max_abs_diff(&y, &y_fit);
            if diff < 1e-10 {
                Ok(format!("max_diff={diff:.2e}"))
            } else {
                Err(format!("max_diff={diff:.2e} exceeds tolerance"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_09_polyfit_polyval_roundtrip", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_09 failed");
}

/// Scenario 10: Spline derivative consistency
#[test]
fn scenario_10_spline_derivative_integral() {
    let mut runner = ScenarioRunner::new("scenario_10_spline_derivative_integral");
    runner.set_interp_meta("CubicSpline-derivative", 20, 1, "Strict");

    // sin(x) on [0, 2pi]
    let n = 20;
    let x: Vec<f64> = (0..=n).map(|i| i as f64 * 2.0 * PI / n as f64).collect();
    let y: Vec<f64> = x.iter().map(|&t| t.sin()).collect();

    let spline = CubicSplineStandalone::new(&x, &y, SplineBc::Natural).expect("create spline");

    runner.record_step(
        "derivative_accuracy",
        "spline.derivative(1).eval(t) vs cos(t)",
        "first derivative",
        "Strict",
        || {
            let deriv = spline.derivative(1);
            let test_pts: Vec<f64> = (1..n).map(|i| i as f64 * 2.0 * PI / n as f64).collect();
            let max_err = test_pts
                .iter()
                .map(|&t| (deriv.eval(t) - t.cos()).abs())
                .fold(0.0_f64, nan_max);
            if max_err < 0.05 {
                Ok(format!("max_derivative_error={max_err:.4}"))
            } else {
                Err(format!("max_derivative_error={max_err:.4} too large"))
            }
        },
    );

    runner.record_step(
        "integral_accuracy",
        "spline.integrate(0, PI) vs 2.0",
        "integral of sin from 0 to PI",
        "Strict",
        || {
            let integral = spline.integrate(0.0, PI);
            let expected = 2.0; // integral of sin from 0 to pi
            let err = (integral - expected).abs();
            if err < 0.01 {
                Ok(format!(
                    "integral={integral:.6}, expected={expected}, err={err:.4}"
                ))
            } else {
                Err(format!(
                    "integral={integral:.6}, expected={expected}, err={err:.4}"
                ))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_10_spline_derivative_integral", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_10 failed");
}

/// Scenario 11: interp1d vs make_interp_spline consistency
#[test]
fn scenario_11_interp_methods_consistency() {
    let mut runner = ScenarioRunner::new("scenario_11_interp_methods_consistency");
    runner.set_interp_meta("cross-method", 10, 1, "Strict");

    let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&t| t.powi(2)).collect();

    runner.record_step(
        "compare_linear_methods",
        "interp1d_linear vs Interp1d(Linear)",
        "both should give same results",
        "Strict",
        || {
            let x_new = vec![0.5, 2.5, 5.5, 8.5];
            let result1 = interp1d_linear(&x, &y, &x_new).map_err(|e| format!("{e}"))?;

            let opts = Interp1dOptions {
                kind: InterpKind::Linear,
                ..Default::default()
            };
            let interp = Interp1d::new(&x, &y, opts).map_err(|e| format!("{e}"))?;
            let result2 = interp.eval_many(&x_new).map_err(|e| format!("{e}"))?;

            let diff = max_abs_diff(&result1, &result2);
            if diff < 1e-14 {
                Ok(format!("max_diff={diff:.2e}"))
            } else {
                Err(format!("methods differ: max_diff={diff:.2e}"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_11_interp_methods_consistency", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_11 failed");
}

// ═══════════════════════════════════════════════════════════════════════
//                   PERFORMANCE BOUNDARY SCENARIOS (12-14)
// ═══════════════════════════════════════════════════════════════════════

/// Scenario 12: Large 1D interpolation
#[test]
fn scenario_12_large_1d_interpolation() {
    let mut runner = ScenarioRunner::new("scenario_12_large_1d_interpolation");
    runner.set_interp_meta("Interp1d-Linear", 10000, 1, "Strict");

    let n = 10000;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&t| t.sin()).collect();

    runner.record_step(
        "create_large_interpolator",
        &format!("Interp1d::new(n={n})"),
        "large dataset",
        "Strict",
        || {
            let start = Instant::now();
            let opts = Interp1dOptions {
                kind: InterpKind::Linear,
                ..Default::default()
            };
            let _interp = Interp1d::new(&x, &y, opts).map_err(|e| format!("{e}"))?;
            let elapsed = start.elapsed();
            Ok(format!("created in {:?}", elapsed))
        },
    );

    runner.record_step(
        "eval_many_large",
        "eval_many(1000 points)",
        "bulk evaluation",
        "Strict",
        || {
            let opts = Interp1dOptions {
                kind: InterpKind::Linear,
                ..Default::default()
            };
            let interp = Interp1d::new(&x, &y, opts).map_err(|e| format!("{e}"))?;
            let x_new: Vec<f64> = (0..1000).map(|i| i as f64 * 10.0 + 0.5).collect();
            let start = Instant::now();
            let _result = interp.eval_many(&x_new).map_err(|e| format!("{e}"))?;
            let elapsed = start.elapsed();
            if elapsed.as_millis() < 100 {
                Ok(format!("evaluated in {:?}", elapsed))
            } else {
                Err(format!("too slow: {:?}", elapsed))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_12_large_1d_interpolation", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_12 failed");
}

/// Scenario 13: High-dimensional regular grid (3D)
#[test]
fn scenario_13_high_dim_regular_grid() {
    let mut runner = ScenarioRunner::new("scenario_13_high_dim_regular_grid");
    runner.set_interp_meta("RegularGridInterpolator-3D", 1000, 3, "Strict");

    // 10x10x10 grid
    let axis: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let points = vec![axis.clone(), axis.clone(), axis.clone()];
    let mut values = Vec::with_capacity(1000);
    for &x in &points[0] {
        for &y in &points[1] {
            for &z in &points[2] {
                values.push(x + y + z);
            }
        }
    }

    runner.record_step(
        "create_3d_grid",
        "RegularGridInterpolator::new(10x10x10)",
        "3D trilinear",
        "Strict",
        || {
            let start = Instant::now();
            let _interp = RegularGridInterpolator::new(
                points.clone(),
                values.clone(),
                RegularGridMethod::Linear,
                false,
                None,
            )
            .map_err(|e| format!("{e}"))?;
            let elapsed = start.elapsed();
            Ok(format!("created in {:?}", elapsed))
        },
    );

    runner.record_step(
        "eval_3d_interior",
        "evaluate at interior points",
        "100 random evaluations",
        "Strict",
        || {
            let interp = RegularGridInterpolator::new(
                points.clone(),
                values.clone(),
                RegularGridMethod::Linear,
                false,
                None,
            )
            .map_err(|e| format!("{e}"))?;
            let start = Instant::now();
            for i in 0..100 {
                let pt = vec![
                    (i % 9) as f64 + 0.5,
                    ((i / 9) % 9) as f64 + 0.5,
                    ((i / 81) % 9) as f64 + 0.5,
                ];
                let _val = interp.eval(&pt).map_err(|e| format!("{e}"))?;
            }
            let elapsed = start.elapsed();
            if elapsed.as_millis() < 50 {
                Ok(format!("100 evals in {:?}", elapsed))
            } else {
                Err(format!("too slow: {:?}", elapsed))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_13_high_dim_regular_grid", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_13 failed");
}

/// Scenario 14: BSpline with many knots
#[test]
fn scenario_14_bspline_many_knots() {
    let mut runner = ScenarioRunner::new("scenario_14_bspline_many_knots");
    runner.set_interp_meta("BSpline", 100, 1, "Strict");

    let n = 100;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&t| (2.0 * PI * t).sin()).collect();

    runner.record_step(
        "create_bspline",
        &format!("make_interp_spline(n={n}, k=3)"),
        "cubic B-spline",
        "Strict",
        || {
            let start = Instant::now();
            let _spline = make_interp_spline(&x, &y, 3).map_err(|e| format!("{e}"))?;
            let elapsed = start.elapsed();
            Ok(format!("created in {:?}", elapsed))
        },
    );

    runner.record_step(
        "eval_bspline_accuracy",
        "evaluate at midpoints",
        "check accuracy",
        "Strict",
        || {
            let spline = make_interp_spline(&x, &y, 3).map_err(|e| format!("{e}"))?;
            let x_test: Vec<f64> = (0..50)
                .map(|i| (2 * i + 1) as f64 / (2 * n - 2) as f64)
                .collect();
            let y_expected: Vec<f64> = x_test.iter().map(|&t| (2.0 * PI * t).sin()).collect();
            let y_actual: Vec<f64> = x_test.iter().map(|&t| spline.eval(t)).collect();
            let max_err = max_abs_diff(&y_expected, &y_actual);
            if max_err < 0.01 {
                Ok(format!("max_error={max_err:.4}"))
            } else {
                Err(format!("max_error={max_err:.4} too large"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_14_bspline_many_knots", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_14 failed");
}

/// Scenario 18: RegularGridInterpolator tensor-product PCHIP parity
/// scipy.interpolate.RegularGridInterpolator((x, y), values, method='pchip')
#[test]
fn scenario_18_regular_grid_pchip() {
    let mut runner = ScenarioRunner::new("scenario_18_regular_grid_pchip");
    runner.set_interp_meta("RegularGridInterpolator-Pchip", 16, 2, "Strict");

    let points: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 2.0, 3.0, 4.0]];
    let mut values = Vec::new();
    for &x in &points[0] {
        for &y in &points[1] {
            values.push(x.powi(4) * y.powi(4));
        }
    }

    let mut interp: Option<RegularGridInterpolator> = None;
    runner.record_step(
        "create_regular_grid_pchip",
        "RegularGridInterpolator::new(points, values, Pchip)",
        "4x4 grid, f(x,y)=x^4*y^4",
        "Strict",
        || match RegularGridInterpolator::new(
            points.clone(),
            values.clone(),
            RegularGridMethod::Pchip,
            false,
            None,
        ) {
            Ok(i) => {
                interp = Some(i);
                Ok("created 2D regular grid pchip interpolator".to_owned())
            }
            Err(e) => Err(format!("construction failed: {e}")),
        },
    );

    let interp = interp.expect("pchip interpolator should exist");
    runner.record_step(
        "eval_reference_points",
        "interp.eval_many([[1.5,2.0],[2.5,2.5],[3.5,3.0]])",
        "SciPy reference points",
        "Strict",
        || {
            let pts = vec![vec![1.5, 2.0], vec![2.5, 2.5], vec![3.5, 3.0]];
            let expected = vec![87.25, 1575.924587673611, 12279.515625];
            let actual = interp.eval_many(&pts).map_err(|e| format!("{e}"))?;
            let max_diff = max_abs_diff(&actual, &expected);
            if max_diff < 1e-9 {
                Ok(format!("max_abs_diff={max_diff:.3e}"))
            } else {
                Err(format!(
                    "max_abs_diff={max_diff:.3e}, actual={actual:?}, expected={expected:?}"
                ))
            }
        },
    );

    runner.record_step(
        "fill_value_out_of_bounds",
        "RegularGridInterpolator::new(..., fill_value=-123.0).eval([-1,2])",
        "out-of-bounds fill contract",
        "Strict",
        || {
            let filled = RegularGridInterpolator::new(
                points.clone(),
                values.clone(),
                RegularGridMethod::Pchip,
                false,
                Some(-123.0),
            )
            .map_err(|e| format!("{e}"))?;
            let actual = filled.eval(&[-1.0, 2.0]).map_err(|e| format!("{e}"))?;
            if actual == -123.0 {
                Ok("returned configured fill_value".to_owned())
            } else {
                Err(format!("actual={actual}, expected=-123.0"))
            }
        },
    );

    runner.record_step(
        "extrapolation_when_fill_none",
        "RegularGridInterpolator::new(..., fill_value=None).eval([-1,2])",
        "SciPy extrapolation contract",
        "Strict",
        || {
            let extrap = RegularGridInterpolator::new(
                points.clone(),
                values.clone(),
                RegularGridMethod::Pchip,
                false,
                None,
            )
            .map_err(|e| format!("{e}"))?;
            let actual = extrap.eval(&[-1.0, 2.0]).map_err(|e| format!("{e}"))?;
            let expected = 2056.0;
            if (actual - expected).abs() < 1e-9 {
                Ok(format!("actual={actual}"))
            } else {
                Err(format!("actual={actual}, expected={expected}"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_18_regular_grid_pchip", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_18 failed");
}

/// Scenario 19: scipy.interpolate.splev(..., der=2) parity
#[test]
fn scenario_19_splev_second_derivative() {
    let mut runner = ScenarioRunner::new("scenario_19_splev_second_derivative");
    runner.set_interp_meta("splev-der2", 4, 1, "Strict");

    let tck = (
        vec![0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 4.0, 4.0, 4.0],
        vec![
            4.131186545826286e-17,
            -1.5593640784053972e-15,
            2.0677589048408837e-15,
            32.0,
            64.0,
        ],
        3_usize,
    );
    let points = vec![0.5, 1.5, 2.5, 3.5];

    runner.record_step(
        "splev_der2_reference",
        "splev_with_derivative(points, tck, 2)",
        "SciPy cubic tck for x^3 sampled on [0,4]",
        "Strict",
        || {
            let expected = vec![3.0, 9.0, 15.0, 21.0];
            let actual = splev_with_derivative(&points, &tck, 2).map_err(|e| format!("{e}"))?;
            let max_diff = max_abs_diff(&actual, &expected);
            if max_diff < 1.0e-10 {
                Ok(format!("max_abs_diff={max_diff:.3e}"))
            } else {
                Err(format!(
                    "max_abs_diff={max_diff:.3e}, actual={actual:?}, expected={expected:?}"
                ))
            }
        },
    );

    runner.record_step(
        "splev_der_above_degree_rejected",
        "splev_with_derivative([0.5], tck, 4)",
        "SciPy rejects derivative order above spline degree",
        "Strict",
        || match splev_with_derivative(&[0.5], &tck, 4) {
            Err(InterpError::InvalidArgument { detail }) if detail == "0<=der=4<=k=3 must hold" => {
                Ok(detail)
            }
            Err(other) => Err(format!("unexpected error: {other}")),
            Ok(values) => Err(format!("expected error, got values {values:?}")),
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_19_splev_second_derivative", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_19 failed");
}

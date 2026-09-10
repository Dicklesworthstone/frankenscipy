#![forbid(unsafe_code)]
//! E2E scenario tests for FSCI-P2C-003 (optimize/root workflows).
//!
//! Implements 8 scenarios per bd-3jh.14.7 acceptance criteria:
//! - Happy path minimize
//! - Multi-algorithm comparison
//! - Root-finding pipeline
//! - Convergence verification chain
//! - Strict vs hardened mode switch behavior
//! - Adversarial objective behavior
//! - Callback workflow
//! - Large-dimension stress

use fsci_opt::types::OptimizeTraceEntry;
use fsci_opt::{
    ConvergenceStatus, CurveFitOptions, DifferentialEvolutionOptions, Integrality,
    LeastSquaresOptions, LinprogResult, MilpOptions, MilpProblem, MinimizeOptions, OptimizeMethod,
    RootMethod, RootOptions, curve_fit, differential_evolution, fsolve, get_optimize_traces,
    halley, least_squares, linprog, milp, minimize, newton_scalar, ridder, root_scalar, secant,
    toms748,
};
use fsci_runtime::RuntimeMode;
use serde::Serialize;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize)]
struct ForensicLogBundle {
    scenario_id: String,
    steps: Vec<ForensicStep>,
    convergence_trace: Vec<ConvergenceTracePoint>,
    environment: EnvironmentInfo,
    overall: OverallResult,
}

#[derive(Debug, Clone, Serialize)]
struct ForensicStep {
    scenario_id: String,
    step_name: String,
    timestamp_ms: u128,
    duration_ms: u128,
    outcome: String,
    algorithm: String,
    nfev: usize,
    detail: String,
}

#[derive(Debug, Clone, Serialize)]
struct ConvergenceTracePoint {
    iter: usize,
    f_val: Option<f64>,
    grad_norm: Option<f64>,
    step_size: Option<f64>,
    algorithm: String,
}

#[derive(Debug, Clone, Serialize)]
struct EnvironmentInfo {
    rust_version: String,
    os: String,
    cpu_count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OverallResult {
    status: String,
    total_duration_ms: u128,
    replay_command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_chain: Option<String>,
}

#[derive(Debug, Clone)]
struct StepOutcome {
    detail: String,
    nfev: usize,
    trace: Vec<ConvergenceTracePoint>,
}

impl StepOutcome {
    fn new(detail: impl Into<String>, nfev: usize) -> Self {
        Self {
            detail: detail.into(),
            nfev,
            trace: Vec::new(),
        }
    }

    fn with_trace(
        detail: impl Into<String>,
        nfev: usize,
        trace: Vec<ConvergenceTracePoint>,
    ) -> Self {
        Self {
            detail: detail.into(),
            nfev,
            trace,
        }
    }
}

struct ScenarioRunner {
    scenario_id: String,
    steps: Vec<ForensicStep>,
    convergence_trace: Vec<ConvergenceTracePoint>,
    passed: bool,
    error_chain: Option<String>,
    start: Instant,
}

impl ScenarioRunner {
    fn new(scenario_id: &str) -> Self {
        Self {
            scenario_id: scenario_id.to_owned(),
            steps: Vec::new(),
            convergence_trace: Vec::new(),
            passed: true,
            error_chain: None,
            start: Instant::now(),
        }
    }

    fn record_step(
        &mut self,
        step_name: &str,
        algorithm: &str,
        f: impl FnOnce() -> Result<StepOutcome, String>,
    ) -> bool {
        let started = Instant::now();
        let timestamp_ms = now_unix_ms();
        let result = f();
        let duration_ms = started.elapsed().as_millis();
        let (outcome, nfev, detail, trace) = match result {
            Ok(outcome) => (
                "pass".to_owned(),
                outcome.nfev,
                outcome.detail,
                outcome.trace,
            ),
            Err(error) => {
                self.passed = false;
                let hinted = format!(
                    "{error}; hint: replay with `{}` and inspect this step's inputs/outputs",
                    replay_cmd(&self.scenario_id)
                );
                if self.error_chain.is_none() {
                    self.error_chain = Some(hinted.clone());
                }
                ("fail".to_owned(), 0, hinted, Vec::new())
            }
        };

        self.convergence_trace.extend(trace);
        self.steps.push(ForensicStep {
            scenario_id: self.scenario_id.clone(),
            step_name: step_name.to_owned(),
            timestamp_ms,
            duration_ms,
            outcome: outcome.clone(),
            algorithm: algorithm.to_owned(),
            nfev,
            detail,
        });

        outcome == "pass"
    }

    fn finish(self) -> ForensicLogBundle {
        let bundle = ForensicLogBundle {
            scenario_id: self.scenario_id.clone(),
            steps: self.steps,
            convergence_trace: self.convergence_trace,
            environment: environment_info(),
            overall: OverallResult {
                status: if self.passed {
                    "pass".to_owned()
                } else {
                    "fail".to_owned()
                },
                total_duration_ms: self.start.elapsed().as_millis(),
                replay_command: replay_cmd(&self.scenario_id),
                error_chain: self.error_chain,
            },
        };
        write_bundle(&self.scenario_id, &bundle);
        bundle
    }
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn e2e_output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts/FSCI-P2C-003/e2e/runs")
}

fn replay_cmd(scenario_id: &str) -> String {
    format!("cargo test -p fsci-conformance --test e2e_optimize -- {scenario_id} --nocapture")
}

fn environment_info() -> EnvironmentInfo {
    EnvironmentInfo {
        rust_version: String::from(env!("CARGO_PKG_VERSION")),
        os: String::from(std::env::consts::OS),
        cpu_count: std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1),
    }
}

fn write_bundle(scenario_id: &str, bundle: &ForensicLogBundle) {
    let dir = e2e_output_dir();
    fs::create_dir_all(&dir).expect("failed to create output dir");
    let path = dir.join(format!("{scenario_id}.json"));
    let bytes = serde_json::to_vec_pretty(bundle).expect("serialize forensic bundle");
    fs::write(&path, bytes).expect("failed to write forensic bundle");
}

fn rosenbrock(x: &[f64]) -> f64 {
    let a = 1.0 - x[0];
    let b = x[1] - x[0] * x[0];
    a * a + 100.0 * b * b
}

fn shifted_quadratic(x: &[f64]) -> f64 {
    let dx0 = x[0] - 2.0;
    let dx1 = x[1] + 3.0;
    dx0 * dx0 + dx1 * dx1
}

fn cusp_abs_objective(x: &[f64]) -> f64 {
    x.iter().map(|v| v.abs()).sum()
}

fn nan_barrier_objective(x: &[f64]) -> f64 {
    if x[0] < 0.0 {
        f64::NAN
    } else {
        (x[0] - 1.0).powi(2) + x[1] * x[1]
    }
}

fn cubic_root_target(x: f64) -> f64 {
    x * x * x - x - 2.0
}

fn capture_minimize_run<F>(
    algorithm: &str,
    fun: F,
    x0: &[f64],
    options: MinimizeOptions,
) -> Result<(fsci_opt::OptimizeResult, Vec<ConvergenceTracePoint>), String>
where
    F: Fn(&[f64]) -> f64,
{
    let _guard = trace_capture_lock()
        .lock()
        .expect("trace capture lock poisoned");
    let _ = get_optimize_traces();
    let result =
        minimize(fun, x0, options).map_err(|error| format!("minimize call failed: {error}"))?;
    let trace = to_convergence_trace(algorithm, get_optimize_traces());
    Ok((result, trace))
}

fn trace_capture_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn to_convergence_trace(
    algorithm: &str,
    entries: Vec<OptimizeTraceEntry>,
) -> Vec<ConvergenceTracePoint> {
    entries
        .into_iter()
        .filter(|entry| entry.event == "iteration")
        .map(|entry| ConvergenceTracePoint {
            iter: entry.iter_num,
            f_val: entry.f_val,
            grad_norm: entry.grad_norm,
            step_size: entry.step_size,
            algorithm: algorithm.to_owned(),
        })
        .collect()
}

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn assert_bundle_pass(bundle: &ForensicLogBundle) {
    assert_eq!(
        bundle.overall.status,
        "pass",
        "scenario={} failed, error_chain={:?}, last_step={:?}",
        bundle.scenario_id,
        bundle.overall.error_chain,
        bundle.steps.last()
    );
}

#[test]
fn e2e_p2c003_01_happy_path_minimize_bfgs() {
    let mut runner = ScenarioRunner::new("p2c003_01_happy_path_minimize_bfgs");
    let x0 = vec![-1.2, 1.0];
    let mut trace_cache = Vec::new();

    runner.record_step("run_rosenbrock_bfgs", "bfgs", || {
        let opts = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-8),
            maxiter: Some(400),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let (result, trace) = capture_minimize_run("bfgs", rosenbrock, &x0, opts)?;
        if !result.success {
            return Err(format!(
                "expected convergence on Rosenbrock, got status={:?} message={}",
                result.status, result.message
            ));
        }
        let distance = l2_norm(&[result.x[0] - 1.0, result.x[1] - 1.0]);
        let final_f = result.fun.unwrap_or(f64::INFINITY);
        if distance > 5.0e-2 || final_f > 1.0e-6 {
            return Err(format!(
                "solution quality too weak: distance={distance:.3e}, final_f={final_f:.3e}"
            ));
        }
        trace_cache = trace.clone();
        Ok(StepOutcome::with_trace(
            format!(
                "converged to x={:?}, f={final_f:.3e}, distance_to_opt={distance:.3e}",
                result.x
            ),
            result.nfev,
            trace,
        ))
    });

    runner.record_step("verify_convergence_trace", "bfgs", || {
        if trace_cache.is_empty() {
            return Err("missing per-iteration convergence trace".to_owned());
        }
        let first = trace_cache
            .first()
            .and_then(|entry| entry.f_val)
            .unwrap_or(f64::INFINITY);
        let last = trace_cache
            .last()
            .and_then(|entry| entry.f_val)
            .unwrap_or(f64::INFINITY);
        if !last.is_finite() || last > first {
            return Err(format!(
                "trace did not improve objective: first_f={first:.3e}, last_f={last:.3e}"
            ));
        }
        Ok(StepOutcome::new(
            format!(
                "trace length={} first_f={first:.3e} last_f={last:.3e}",
                trace_cache.len()
            ),
            0,
        ))
    });

    let bundle = runner.finish();
    assert_bundle_pass(&bundle);
}

#[test]
fn e2e_p2c003_02_multi_algorithm_comparison() {
    let mut runner = ScenarioRunner::new("p2c003_02_multi_algorithm_comparison");
    let mut terminal_values = Vec::<(String, f64)>::new();
    let x0 = vec![8.0, -8.0];

    for (method, algorithm) in [
        (OptimizeMethod::Bfgs, "bfgs"),
        (OptimizeMethod::ConjugateGradient, "cg_pr_plus"),
        (OptimizeMethod::Powell, "powell"),
    ] {
        runner.record_step(&format!("run_{algorithm}"), algorithm, || {
            let opts = MinimizeOptions {
                method: Some(method),
                tol: Some(1.0e-8),
                maxiter: Some(300),
                mode: RuntimeMode::Strict,
                ..MinimizeOptions::default()
            };
            let (result, trace) = capture_minimize_run(algorithm, shifted_quadratic, &x0, opts)?;
            let final_f = result.fun.unwrap_or(f64::INFINITY);
            if !final_f.is_finite() {
                return Err(format!("algorithm={algorithm} produced non-finite objective"));
            }
            if final_f > 1.0e-5 {
                return Err(format!(
                    "algorithm={algorithm} did not reach low objective; final_f={final_f:.3e}, status={:?}",
                    result.status
                ));
            }
            terminal_values.push((algorithm.to_owned(), final_f));
            Ok(StepOutcome::with_trace(
                format!(
                    "algorithm={algorithm} success={} status={:?} final_f={final_f:.3e}",
                    result.success, result.status
                ),
                result.nfev,
                trace,
            ))
        });
    }

    runner.record_step("compare_terminal_objectives", "comparison", || {
        if terminal_values.len() != 3 {
            return Err(format!(
                "expected 3 algorithm outcomes, got {}",
                terminal_values.len()
            ));
        }
        let min_f = terminal_values.iter().map(|(_, value)| *value).fold(
            f64::INFINITY,
            |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.min(b)
                }
            },
        );
        let max_f = terminal_values.iter().map(|(_, value)| *value).fold(
            f64::NEG_INFINITY,
            |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.max(b)
                }
            },
        );
        let spread = max_f - min_f;
        if spread > 1.0e-4 {
            return Err(format!(
                "algorithm spread too wide: {spread:.3e} ({terminal_values:?})"
            ));
        }
        Ok(StepOutcome::new(
            format!("terminal objective spread={spread:.3e}"),
            0,
        ))
    });

    let bundle = runner.finish();
    assert_bundle_pass(&bundle);
}

#[test]
fn e2e_p2c003_03_root_finding_pipeline() {
    let mut runner = ScenarioRunner::new("p2c003_03_root_finding_pipeline");
    let expected_root = 1.521_379_706_804_567_6;
    let mut brentq_root = 0.0;
    let mut bisect_root = 0.0;

    runner.record_step("run_brentq", "brentq", || {
        let options = RootOptions {
            method: Some(RootMethod::Brentq),
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result = root_scalar(cubic_root_target, Some((1.0, 2.0)), None, None, options)
            .map_err(|error| format!("brentq failed: {error}"))?;
        if !result.converged {
            return Err(format!("brentq failed to converge: {}", result.message));
        }
        let residual = cubic_root_target(result.root).abs();
        if residual > 1.0e-10 {
            return Err(format!("brentq residual too large: {residual:.3e}"));
        }
        brentq_root = result.root;
        Ok(StepOutcome::with_trace(
            format!(
                "root={:.12} iterations={} calls={} residual={residual:.3e}",
                result.root, result.iterations, result.function_calls
            ),
            result.function_calls,
            vec![ConvergenceTracePoint {
                iter: result.iterations,
                f_val: Some(residual),
                grad_norm: None,
                step_size: None,
                algorithm: "brentq".to_owned(),
            }],
        ))
    });

    runner.record_step("run_bisect", "bisect", || {
        let options = RootOptions {
            method: Some(RootMethod::Bisect),
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result = root_scalar(cubic_root_target, Some((1.0, 2.0)), None, None, options)
            .map_err(|error| format!("bisect failed: {error}"))?;
        if !result.converged {
            return Err(format!("bisect failed to converge: {}", result.message));
        }
        bisect_root = result.root;
        let residual = cubic_root_target(result.root).abs();
        Ok(StepOutcome::with_trace(
            format!(
                "root={:.12} iterations={} calls={} residual={residual:.3e}",
                result.root, result.iterations, result.function_calls
            ),
            result.function_calls,
            vec![ConvergenceTracePoint {
                iter: result.iterations,
                f_val: Some(residual),
                grad_norm: None,
                step_size: None,
                algorithm: "bisect".to_owned(),
            }],
        ))
    });

    runner.record_step("verify_root_agreement", "root_pipeline", || {
        let brentq_err = (brentq_root - expected_root).abs();
        let bisect_err = (bisect_root - expected_root).abs();
        let cross_diff = (brentq_root - bisect_root).abs();
        if brentq_err > 1.0e-9 || bisect_err > 1.0e-8 || cross_diff > 1.0e-8 {
            return Err(format!(
                "root mismatch: brentq_err={brentq_err:.3e}, bisect_err={bisect_err:.3e}, cross_diff={cross_diff:.3e}"
            ));
        }
        Ok(StepOutcome::new(
            format!(
                "root agreement verified (brentq={brentq_root:.12}, bisect={bisect_root:.12})"
            ),
            0,
        ))
    });

    let bundle = runner.finish();
    assert_bundle_pass(&bundle);
}

#[test]
fn e2e_p2c003_04_convergence_verification_chain() {
    let mut runner = ScenarioRunner::new("p2c003_04_convergence_verification_chain");
    let x0 = vec![4.5, -4.5];
    let mut trace_cache = Vec::new();

    runner.record_step("run_cg_with_trace_capture", "cg_pr_plus", || {
        let opts = MinimizeOptions {
            method: Some(OptimizeMethod::ConjugateGradient),
            tol: Some(1.0e-7),
            maxiter: Some(300),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let (result, trace) = capture_minimize_run("cg_pr_plus", shifted_quadratic, &x0, opts)?;
        let final_f = result.fun.unwrap_or(f64::INFINITY);
        if final_f > 1.0e-5 {
            return Err(format!(
                "cg_pr_plus did not converge enough: final_f={final_f:.3e}, status={:?}",
                result.status
            ));
        }
        trace_cache = trace.clone();
        Ok(StepOutcome::with_trace(
            format!(
                "captured trace points={}, final_f={final_f:.3e}",
                trace_cache.len()
            ),
            result.nfev,
            trace,
        ))
    });

    runner.record_step("verify_trace_semantics", "cg_pr_plus", || {
        if trace_cache.is_empty() {
            return Err("expected at least one convergence trace point".to_owned());
        }
        if trace_cache.iter().any(|entry| entry.iter == 0) {
            return Err("trace contains zero iteration index".to_owned());
        }
        if trace_cache
            .iter()
            .any(|entry| entry.grad_norm.is_some_and(|value| !value.is_finite()))
        {
            return Err("trace contains non-finite grad_norm values".to_owned());
        }
        let first_f = trace_cache
            .first()
            .and_then(|entry| entry.f_val)
            .unwrap_or(f64::INFINITY);
        let last_f = trace_cache
            .last()
            .and_then(|entry| entry.f_val)
            .unwrap_or(f64::INFINITY);
        if last_f > first_f {
            return Err(format!(
                "trace regression detected: first_f={first_f:.3e}, last_f={last_f:.3e}"
            ));
        }
        Ok(StepOutcome::new(
            format!(
                "trace semantics valid: points={}, first_f={first_f:.3e}, last_f={last_f:.3e}",
                trace_cache.len()
            ),
            0,
        ))
    });

    let bundle = runner.finish();
    assert_bundle_pass(&bundle);
}

#[test]
fn e2e_p2c003_05_mode_switch_non_finite_objective() {
    let mut runner = ScenarioRunner::new("p2c003_05_mode_switch_non_finite_objective");
    let x0 = vec![-0.5, 1.0];

    runner.record_step("strict_mode_behavior", "bfgs-strict", || {
        let opts = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let (result, trace) =
            capture_minimize_run("bfgs-strict", nan_barrier_objective, &x0, opts)?;
        if result.success {
            return Err("strict mode unexpectedly reported success on NaN objective".to_owned());
        }
        if !result.message.contains("non-finite") {
            return Err(format!(
                "strict mode message missing non-finite context: {}",
                result.message
            ));
        }
        Ok(StepOutcome::with_trace(
            format!(
                "strict mode produced expected failure: status={:?}",
                result.status
            ),
            result.nfev,
            trace,
        ))
    });

    runner.record_step("hardened_mode_behavior", "bfgs-hardened", || {
        let opts = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            mode: RuntimeMode::Hardened,
            ..MinimizeOptions::default()
        };
        let (result, trace) =
            capture_minimize_run("bfgs-hardened", nan_barrier_objective, &x0, opts)?;
        if result.success {
            return Err("hardened mode unexpectedly reported success on NaN objective".to_owned());
        }
        if !result
            .message
            .contains("hardened mode rejects non-finite objective values")
        {
            return Err(format!(
                "hardened mode did not report expected rejection message: {}",
                result.message
            ));
        }
        Ok(StepOutcome::with_trace(
            format!(
                "hardened mode rejected non-finite objective as expected: status={:?}",
                result.status
            ),
            result.nfev,
            trace,
        ))
    });

    let bundle = runner.finish();
    assert_bundle_pass(&bundle);
}

#[test]
fn e2e_p2c003_06_adversarial_nondifferentiable_objective() {
    let mut runner = ScenarioRunner::new("p2c003_06_adversarial_nondifferentiable_objective");
    let x0 = vec![3.0, -4.0];

    runner.record_step("powell_on_abs_objective", "powell", || {
        let opts = MinimizeOptions {
            method: Some(OptimizeMethod::Powell),
            tol: Some(1.0e-6),
            maxiter: Some(250),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let (result, trace) = capture_minimize_run("powell", cusp_abs_objective, &x0, opts)?;
        let final_f = result.fun.unwrap_or(f64::INFINITY);
        if !final_f.is_finite() || final_f > 1.0 {
            return Err(format!(
                "adversarial objective did not reach acceptable value: final_f={final_f:.3e}, status={:?}, message={}",
                result.status, result.message
            ));
        }
        Ok(StepOutcome::with_trace(
            format!(
                "graceful handling confirmed: success={} status={:?} final_f={final_f:.3e}",
                result.success, result.status
            ),
            result.nfev,
            trace,
        ))
    });

    let bundle = runner.finish();
    assert_bundle_pass(&bundle);
}

static CALLBACK_INVOCATIONS: AtomicUsize = AtomicUsize::new(0);

fn stop_immediately(_: &[f64]) -> bool {
    CALLBACK_INVOCATIONS.fetch_add(1, Ordering::SeqCst);
    false
}

#[test]
fn e2e_p2c003_07_callback_workflow() {
    let mut runner = ScenarioRunner::new("p2c003_07_callback_workflow");
    let x0 = vec![2.5, -2.5];

    runner.record_step("run_with_callback_stop", "bfgs-callback", || {
        CALLBACK_INVOCATIONS.store(0, Ordering::SeqCst);
        let opts = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            callback: Some(stop_immediately),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let (result, trace) = capture_minimize_run("bfgs-callback", shifted_quadratic, &x0, opts)?;
        if result.status != ConvergenceStatus::CallbackStop {
            return Err(format!(
                "expected callback stop status, got {:?} ({})",
                result.status, result.message
            ));
        }
        let invocations = CALLBACK_INVOCATIONS.load(Ordering::SeqCst);
        if invocations < 1 {
            return Err(format!(
                "callback should have been invoked at least once, got {invocations}"
            ));
        }
        Ok(StepOutcome::with_trace(
            format!(
                "callback stop confirmed with invocations={invocations}, nfev={}",
                result.nfev
            ),
            result.nfev,
            trace,
        ))
    });

    let bundle = runner.finish();
    assert_bundle_pass(&bundle);
}

#[test]
fn e2e_p2c003_08_large_dimension_stress() {
    let mut runner = ScenarioRunner::new("p2c003_08_large_dimension_stress");
    let n = 64usize;
    let shift: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect();
    let x0 = vec![5.0; n];

    runner.record_step("cg_large_dimension", "cg_pr_plus", || {
        let start = Instant::now();
        let opts = MinimizeOptions {
            method: Some(OptimizeMethod::ConjugateGradient),
            tol: Some(1.0e-7),
            maxiter: Some(500),
            maxfev: Some(20_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let objective = |x: &[f64]| {
            x.iter()
                .zip(shift.iter())
                .map(|(value, target)| {
                    let delta = value - target;
                    delta * delta
                })
                .sum::<f64>()
        };
        let (result, trace) = capture_minimize_run("cg_pr_plus", objective, &x0, opts)?;
        let elapsed_ms = start.elapsed().as_millis();
        let final_f = result.fun.unwrap_or(f64::INFINITY);
        if elapsed_ms > 15_000 {
            return Err(format!(
                "large-dimension stress exceeded time budget: {elapsed_ms}ms"
            ));
        }
        if final_f > 1.0e-4 {
            return Err(format!(
                "large-dimension objective too high: final_f={final_f:.3e}, status={:?}",
                result.status
            ));
        }
        Ok(StepOutcome::with_trace(
            format!("completed n={n} in {elapsed_ms}ms with final_f={final_f:.3e}"),
            result.nfev,
            trace,
        ))
    });

    let bundle = runner.finish();
    assert_bundle_pass(&bundle);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Additional parity scenarios: curve_fit, least_squares, global opt, linprog
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn exponential_model(x: f64, params: &[f64]) -> f64 {
    // y = a * exp(-b * x) + c
    params[0] * (-params[1] * x).exp() + params[2]
}

fn generate_exponential_data() -> (Vec<f64>, Vec<f64>) {
    // True parameters: a=2.5, b=1.3, c=0.5
    let xdata: Vec<f64> = (0..20).map(|i| i as f64 * 0.25).collect();
    let ydata: Vec<f64> = xdata
        .iter()
        .map(|&x| 2.5 * (-1.3 * x).exp() + 0.5)
        .collect();
    (xdata, ydata)
}

#[test]
fn e2e_p2c003_09_curve_fit_exponential() {
    let mut runner = ScenarioRunner::new("p2c003_09_curve_fit_exponential");
    let (xdata, ydata) = generate_exponential_data();

    runner.record_step("fit_exponential_model", "curve_fit", || {
        let model = |x: f64, p: &[f64]| exponential_model(x, p);
        let options = CurveFitOptions {
            p0: Some(vec![1.0, 1.0, 0.0]),
            ..CurveFitOptions::default()
        };
        let result = curve_fit(model, &xdata, &ydata, options)
            .map_err(|e| format!("curve_fit failed: {e}"))?;

        if !result.ls_result.success {
            return Err(format!(
                "curve_fit did not converge: {}",
                result.ls_result.message
            ));
        }

        // Check recovered parameters (true: a=2.5, b=1.3, c=0.5)
        let a_err = (result.popt[0] - 2.5).abs();
        let b_err = (result.popt[1] - 1.3).abs();
        let c_err = (result.popt[2] - 0.5).abs();

        if a_err > 0.01 || b_err > 0.01 || c_err > 0.01 {
            return Err(format!(
                "parameter recovery failed: popt={:?}, expected [2.5, 1.3, 0.5]",
                result.popt
            ));
        }

        Ok(StepOutcome::new(
            format!(
                "recovered params: a={:.4}, b={:.4}, c={:.4}, cost={:.2e}",
                result.popt[0], result.popt[1], result.popt[2], result.ls_result.cost
            ),
            result.ls_result.nfev,
        ))
    });

    runner.record_step("verify_covariance_matrix", "curve_fit", || {
        let (xdata, ydata) = generate_exponential_data();
        let model = |x: f64, p: &[f64]| exponential_model(x, p);
        let options = CurveFitOptions {
            p0: Some(vec![1.0, 1.0, 0.0]),
            ..CurveFitOptions::default()
        };
        let result = curve_fit(model, &xdata, &ydata, options)
            .map_err(|e| format!("curve_fit failed: {e}"))?;

        // Covariance matrix should be 3x3 and symmetric-ish
        if result.pcov.len() != 3 || result.pcov[0].len() != 3 {
            return Err(format!(
                "unexpected pcov shape: {}x{}",
                result.pcov.len(),
                result.pcov.first().map(|r| r.len()).unwrap_or(0)
            ));
        }

        // Diagonal elements should be positive (variances)
        for i in 0..3 {
            if result.pcov[i][i] < 0.0 {
                return Err(format!(
                    "negative variance at pcov[{i}][{i}] = {}",
                    result.pcov[i][i]
                ));
            }
        }

        Ok(StepOutcome::new(
            format!(
                "pcov diagonal: [{:.2e}, {:.2e}, {:.2e}]",
                result.pcov[0][0], result.pcov[1][1], result.pcov[2][2]
            ),
            0,
        ))
    });

    let bundle = runner.finish();
    assert_bundle_pass(&bundle);
}

#[test]
fn e2e_p2c003_10_least_squares_levenberg_marquardt() {
    let mut runner = ScenarioRunner::new("p2c003_10_least_squares_levenberg_marquardt");

    runner.record_step("solve_circle_fitting", "least_squares", || {
        // Fit a circle to noisy points: find (cx, cy, r) minimizing distance residuals
        // True circle: center (1, 2), radius 3
        let angles: Vec<f64> = (0..8)
            .map(|i| i as f64 * std::f64::consts::PI / 4.0)
            .collect();
        let points: Vec<(f64, f64)> = angles
            .iter()
            .map(|&t| (1.0 + 3.0 * t.cos(), 2.0 + 3.0 * t.sin()))
            .collect();

        let residuals = |params: &[f64]| -> Vec<f64> {
            let cx = params[0];
            let cy = params[1];
            let r = params[2];
            points
                .iter()
                .map(|&(x, y)| {
                    let dist = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
                    dist - r
                })
                .collect()
        };

        let x0 = vec![0.0, 0.0, 1.0];
        let result = least_squares(residuals, &x0, LeastSquaresOptions::default())
            .map_err(|e| format!("least_squares failed: {e}"))?;

        if !result.success {
            return Err(format!("did not converge: {}", result.message));
        }

        let cx_err = (result.x[0] - 1.0).abs();
        let cy_err = (result.x[1] - 2.0).abs();
        let r_err = (result.x[2] - 3.0).abs();

        if cx_err > 0.1 || cy_err > 0.1 || r_err > 0.1 {
            return Err(format!(
                "circle fit inaccurate: cx={:.3}, cy={:.3}, r={:.3}",
                result.x[0], result.x[1], result.x[2]
            ));
        }

        Ok(StepOutcome::new(
            format!(
                "fitted circle: cx={:.4}, cy={:.4}, r={:.4}, cost={:.2e}",
                result.x[0], result.x[1], result.x[2], result.cost
            ),
            result.nfev,
        ))
    });

    let bundle = runner.finish();
    assert_bundle_pass(&bundle);
}

#[test]
fn e2e_p2c003_11_differential_evolution_rastrigin() {
    let mut runner = ScenarioRunner::new("p2c003_11_differential_evolution_rastrigin");

    runner.record_step("optimize_rastrigin_2d", "differential_evolution", || {
        // Rastrigin function: global minimum at origin, many local minima
        let rastrigin = |x: &[f64]| {
            let n = x.len() as f64;
            10.0 * n
                + x.iter()
                    .map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
                    .sum::<f64>()
        };

        let bounds = [(-5.12, 5.12), (-5.12, 5.12)];

        let options = DifferentialEvolutionOptions {
            maxiter: 200,
            popsize: 15,
            seed: Some(42),
            ..DifferentialEvolutionOptions::default()
        };

        let result = differential_evolution(rastrigin, &bounds, options)
            .map_err(|e| format!("DE failed: {e}"))?;

        let final_f = result.fun.unwrap_or(f64::INFINITY);
        let dist_to_origin = (result.x[0].powi(2) + result.x[1].powi(2)).sqrt();

        // Global minimum is at (0, 0) with f=0
        if final_f > 1.0 || dist_to_origin > 1.0 {
            return Err(format!(
                "DE did not find global minimum: f={:.3e}, x={:?}",
                final_f, result.x
            ));
        }

        Ok(StepOutcome::new(
            format!(
                "found minimum: f={:.4}, x=[{:.4}, {:.4}]",
                final_f, result.x[0], result.x[1]
            ),
            result.nfev,
        ))
    });

    let bundle = runner.finish();
    assert_bundle_pass(&bundle);
}

#[test]
fn e2e_p2c003_12_linprog_basic() {
    let mut runner = ScenarioRunner::new("p2c003_12_linprog_basic");

    runner.record_step("solve_simple_lp", "linprog", || {
        // Minimize: c^T x = -x0 - 2*x1
        // Subject to: x0 + x1 <= 4
        //            2*x0 + x1 <= 6
        //            x0, x1 >= 0
        // Optimal: x = (0, 4), f = -8 (vertex of feasible region)
        let c = vec![-1.0, -2.0];
        let a_ub = vec![vec![1.0, 1.0], vec![2.0, 1.0]];
        let b_ub = vec![4.0, 6.0];
        let a_eq: Vec<Vec<f64>> = vec![];
        let b_eq: Vec<f64> = vec![];
        let bounds = vec![(Some(0.0), None), (Some(0.0), None)]; // x >= 0

        let result: LinprogResult = linprog(&c, &a_ub, &b_ub, &a_eq, &b_eq, &bounds, None)
            .map_err(|e| format!("linprog failed: {e}"))?;

        if !result.success {
            return Err(format!("linprog did not converge: {}", result.message));
        }

        let x0_err = (result.x[0] - 0.0).abs();
        let x1_err = (result.x[1] - 4.0).abs();
        let f_err = (result.fun - (-8.0)).abs();

        if x0_err > 0.01 || x1_err > 0.01 || f_err > 0.01 {
            return Err(format!(
                "LP solution incorrect: x={:?}, f={:.4}, expected x=[0,4], f=-8",
                result.x, result.fun
            ));
        }

        Ok(StepOutcome::new(
            format!(
                "optimal: x=[{:.4}, {:.4}], f={:.4}",
                result.x[0], result.x[1], result.fun
            ),
            0,
        ))
    });

    runner.record_step("solve_equality_constrained_lp", "linprog", || {
        // Minimize: x0 + x1
        // Subject to: x0 + x1 = 5
        //            x0, x1 >= 0
        // Optimal: any point on line segment, e.g. (0, 5) or (5, 0)
        let c = vec![1.0, 1.0];
        let a_ub: Vec<Vec<f64>> = vec![];
        let b_ub: Vec<f64> = vec![];
        let a_eq = vec![vec![1.0, 1.0]];
        let b_eq = vec![5.0];
        let bounds = vec![(Some(0.0), None), (Some(0.0), None)]; // x >= 0

        let result = linprog(&c, &a_ub, &b_ub, &a_eq, &b_eq, &bounds, None)
            .map_err(|e| format!("linprog failed: {e}"))?;

        if !result.success {
            return Err(format!("linprog did not converge: {}", result.message));
        }

        // Check constraint: x0 + x1 = 5
        let constraint_err = (result.x[0] + result.x[1] - 5.0).abs();
        let f_err = (result.fun - 5.0).abs();

        if constraint_err > 0.01 || f_err > 0.01 {
            return Err(format!(
                "LP constraint violated or wrong f: x={:?}, f={:.4}",
                result.x, result.fun
            ));
        }

        Ok(StepOutcome::new(
            format!(
                "optimal: x=[{:.4}, {:.4}], f={:.4}",
                result.x[0], result.x[1], result.fun
            ),
            0,
        ))
    });

    let bundle = runner.finish();
    assert_bundle_pass(&bundle);
}

#[test]
fn e2e_p2c003_13_root_methods_comparison() {
    let mut runner = ScenarioRunner::new("p2c003_13_root_methods_comparison");
    let target_fn = |x: f64| x * x * x - x - 2.0;
    let expected_root = 1.521_379_706_804_567_6;
    let bracket = (1.0, 2.0);
    let opts = RootOptions::default();

    let mut roots = Vec::new();

    // Test ridder
    runner.record_step("ridder_root", "ridder", || {
        let result = ridder(target_fn, bracket, opts).map_err(|e| format!("ridder failed: {e}"))?;
        if !result.converged {
            return Err(format!("ridder did not converge: {}", result.message));
        }
        let err = (result.root - expected_root).abs();
        if err > 1e-9 {
            return Err(format!("ridder root error too large: {err:.3e}"));
        }
        roots.push(("ridder", result.root));
        Ok(StepOutcome::new(
            format!("root={:.12}, calls={}", result.root, result.function_calls),
            result.function_calls,
        ))
    });

    // Test toms748
    runner.record_step("toms748_root", "toms748", || {
        let result =
            toms748(target_fn, bracket, opts).map_err(|e| format!("toms748 failed: {e}"))?;
        if !result.converged {
            return Err(format!("toms748 did not converge: {}", result.message));
        }
        let err = (result.root - expected_root).abs();
        if err > 1e-9 {
            return Err(format!("toms748 root error too large: {err:.3e}"));
        }
        roots.push(("toms748", result.root));
        Ok(StepOutcome::new(
            format!("root={:.12}, calls={}", result.root, result.function_calls),
            result.function_calls,
        ))
    });

    // Test newton
    runner.record_step("newton_root", "newton", || {
        let f = |x: f64| x * x * x - x - 2.0;
        let fprime = |x: f64| 3.0 * x * x - 1.0;
        let result =
            newton_scalar(f, fprime, 1.5, opts).map_err(|e| format!("newton failed: {e}"))?;
        if !result.converged {
            return Err(format!("newton did not converge: {}", result.message));
        }
        let err = (result.root - expected_root).abs();
        if err > 1e-9 {
            return Err(format!("newton root error too large: {err:.3e}"));
        }
        roots.push(("newton", result.root));
        Ok(StepOutcome::new(
            format!("root={:.12}, calls={}", result.root, result.function_calls),
            result.function_calls,
        ))
    });

    // Test secant
    runner.record_step("secant_root", "secant", || {
        let result =
            secant(target_fn, 1.0, Some(2.0), opts).map_err(|e| format!("secant failed: {e}"))?;
        if !result.converged {
            return Err(format!("secant did not converge: {}", result.message));
        }
        let err = (result.root - expected_root).abs();
        if err > 1e-9 {
            return Err(format!("secant root error too large: {err:.3e}"));
        }
        roots.push(("secant", result.root));
        Ok(StepOutcome::new(
            format!("root={:.12}, calls={}", result.root, result.function_calls),
            result.function_calls,
        ))
    });

    // Test halley
    runner.record_step("halley_root", "halley", || {
        let f = |x: f64| x * x * x - x - 2.0;
        let fprime = |x: f64| 3.0 * x * x - 1.0;
        let fprime2 = |x: f64| 6.0 * x;
        let result =
            halley(f, fprime, fprime2, 1.5, opts).map_err(|e| format!("halley failed: {e}"))?;
        if !result.converged {
            return Err(format!("halley did not converge: {}", result.message));
        }
        let err = (result.root - expected_root).abs();
        if err > 1e-9 {
            return Err(format!("halley root error too large: {err:.3e}"));
        }
        roots.push(("halley", result.root));
        Ok(StepOutcome::new(
            format!("root={:.12}, calls={}", result.root, result.function_calls),
            result.function_calls,
        ))
    });

    // Verify agreement
    runner.record_step("verify_root_agreement", "comparison", || {
        if roots.len() < 5 {
            return Err(format!("expected 5 roots, got {}", roots.len()));
        }
        let max_diff = roots
            .iter()
            .flat_map(|(_, r1)| roots.iter().map(move |(_, r2)| (r1 - r2).abs()))
            .fold(0.0f64, f64::max);
        if max_diff > 1e-8 {
            return Err(format!(
                "root methods disagree: max_diff={:.3e}, roots={:?}",
                max_diff, roots
            ));
        }
        Ok(StepOutcome::new(
            format!("all methods agree within {:.3e}", max_diff),
            0,
        ))
    });

    let bundle = runner.finish();
    assert_bundle_pass(&bundle);
}

#[test]
fn e2e_p2c003_14_fsolve_multivariate() {
    let mut runner = ScenarioRunner::new("p2c003_14_fsolve_multivariate");

    runner.record_step("solve_nonlinear_system", "fsolve", || {
        // System: x^2 + y^2 = 1
        //         x - y = 0
        // Solutions: (±1/√2, ±1/√2)
        let system = |x: &[f64]| vec![x[0] * x[0] + x[1] * x[1] - 1.0, x[0] - x[1]];

        let x0 = vec![0.5, 0.5];
        let result = fsolve(system, &x0).map_err(|e| format!("fsolve failed: {e}"))?;

        if !result.converged {
            return Err(format!("fsolve did not converge: {}", result.message));
        }

        // Check solution: should be on unit circle and x = y
        let r = (result.x[0].powi(2) + result.x[1].powi(2)).sqrt();
        let diff = (result.x[0] - result.x[1]).abs();

        if (r - 1.0).abs() > 0.01 || diff > 0.01 {
            return Err(format!(
                "solution incorrect: x={:?}, r={:.4}, diff={:.4}",
                result.x, r, diff
            ));
        }

        Ok(StepOutcome::new(
            format!(
                "found root: x=[{:.6}, {:.6}], residual_norm={:.2e}",
                result.x[0],
                result.x[1],
                l2_norm(&result.fun)
            ),
            result.function_calls,
        ))
    });

    runner.record_step("solve_3d_system", "fsolve", || {
        // System: x + y + z = 6
        //         x*y = 8
        //         y + z = 5
        // Solution: x=2, y=4, z=1 (or permutations)
        let system = |v: &[f64]| {
            vec![
                v[0] + v[1] + v[2] - 6.0,
                v[0] * v[1] - 8.0,
                v[1] + v[2] - 5.0,
            ]
        };

        let x0 = vec![1.0, 3.0, 2.0];
        let result = fsolve(system, &x0).map_err(|e| format!("fsolve failed: {e}"))?;

        if !result.converged {
            return Err(format!("fsolve did not converge: {}", result.message));
        }

        // Verify equations
        let eq1 = result.x[0] + result.x[1] + result.x[2] - 6.0;
        let eq2 = result.x[0] * result.x[1] - 8.0;
        let eq3 = result.x[1] + result.x[2] - 5.0;
        let max_residual = eq1.abs().max(eq2.abs()).max(eq3.abs());

        if max_residual > 0.01 {
            return Err(format!(
                "equations not satisfied: residuals=[{:.3e}, {:.3e}, {:.3e}]",
                eq1, eq2, eq3
            ));
        }

        Ok(StepOutcome::new(
            format!(
                "found root: x=[{:.4}, {:.4}, {:.4}], max_residual={:.2e}",
                result.x[0], result.x[1], result.x[2], max_residual
            ),
            result.function_calls,
        ))
    });

    let bundle = runner.finish();
    assert_bundle_pass(&bundle);
}

#[test]
fn e2e_p2c003_15_free_variable_linear_programs() {
    let mut runner = ScenarioRunner::new("p2c003_15_free_variable_linear_programs");

    runner.record_step("solve_free_variable_linprog", "linprog", || {
        // Minimize x subject to x >= 1, with x otherwise free.
        let c = vec![1.0];
        let a_ub = vec![vec![-1.0]];
        let b_ub = vec![-1.0];
        let bounds = vec![(None, None)];

        let result = linprog(&c, &a_ub, &b_ub, &[], &[], &bounds, None)
            .map_err(|e| format!("linprog failed: {e}"))?;

        if !result.success {
            return Err(format!("linprog did not converge: {}", result.message));
        }
        if (result.x[0] - 1.0).abs() > 0.01 || (result.fun - 1.0).abs() > 0.01 {
            return Err(format!(
                "free-variable LP solution incorrect: x={:?}, f={:.4}, expected x=[1], f=1",
                result.x, result.fun
            ));
        }

        Ok(StepOutcome::new(
            format!(
                "optimal free-variable LP: x=[{:.4}], f={:.4}",
                result.x[0], result.fun
            ),
            0,
        ))
    });

    runner.record_step("solve_free_integer_milp", "milp", || {
        // Maximize x (minimize -x) subject to -2.2 <= x <= 2.2, x integer.
        let c = vec![-1.0];
        let a_ub = vec![vec![1.0], vec![-1.0]];
        let b_ub = vec![2.2, 2.2];
        let result = milp(
            MilpProblem {
                c: &c,
                integrality: &[Integrality::Integer],
                a_ub: &a_ub,
                b_ub: &b_ub,
                a_eq: &[],
                b_eq: &[],
                bounds: &[(None, None)],
            },
            MilpOptions::default(),
        )
        .map_err(|e| format!("milp failed: {e}"))?;

        if !result.success {
            return Err(format!("milp did not converge: {}", result.message));
        }
        if result.x != vec![2.0] || (result.fun - (-2.0)).abs() > 0.01 {
            return Err(format!(
                "free-variable MILP solution incorrect: x={:?}, f={:.4}, expected x=[2], f=-2",
                result.x, result.fun
            ));
        }

        Ok(StepOutcome::new(
            format!(
                "optimal free-integer MILP: x=[{:.4}], f={:.4}",
                result.x[0], result.fun
            ),
            0,
        ))
    });

    let bundle = runner.finish();
    assert_bundle_pass(&bundle);
}

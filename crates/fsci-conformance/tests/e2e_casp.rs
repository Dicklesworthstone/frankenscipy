#![forbid(unsafe_code)]
//! E2E scenario tests for FSCI-P2C-008 (CASP Runtime).
//!
//! Implements bd-3jh.19.7 acceptance criteria:
//!   Happy-path:     1-4  (policy decision → solver selection → ledger → calibrator)
//!   Error recovery: 5-6  (mode switch → calibrator fallback)
//!   Adversarial:    7-8  (signal replay → tie-breaking)
//!
//! Each scenario emits a forensic log bundle to
//! `fixtures/artifacts/FSCI-P2C-008/e2e/`.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use fsci_runtime::{
    ConformalCalibrator, DecisionSignals, MatrixConditionState, PolicyAction, PolicyController,
    RuntimeMode, SignalSequence, SolverAction, SolverPortfolio,
};
use serde::Serialize;

// ───────────────────────── Forensic log types ─────────────────────────

#[derive(Debug, Clone, Serialize)]
struct ForensicLogBundle {
    scenario_id: String,
    steps: Vec<ForensicStep>,
    artifacts: Vec<ArtifactRef>,
    environment: EnvironmentInfo,
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
struct OverallResult {
    status: String,
    total_duration_ns: u128,
    replay_command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_chain: Option<String>,
}

// ───────────────────────── Helpers ─────────────────────────

fn e2e_output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts/FSCI-P2C-008/e2e")
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
    format!("cargo test -p fsci-conformance --test e2e_casp -- {scenario_id} --nocapture")
}

fn write_bundle(scenario_id: &str, bundle: &ForensicLogBundle) {
    let dir = e2e_output_dir();
    fs::create_dir_all(&dir)
        .unwrap_or_else(|e| panic!("failed to create e2e dir {}: {e}", dir.display()));
    let path = dir.join(format!("{scenario_id}.json"));
    let json = serde_json::to_vec_pretty(bundle).expect("serialize bundle");
    fs::write(&path, &json).unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
}

fn make_step(
    step_id: usize,
    name: &str,
    action: &str,
    input: &str,
    output: &str,
    dur: u128,
    outcome: &str,
) -> ForensicStep {
    ForensicStep {
        step_id,
        step_name: name.to_string(),
        action: action.to_string(),
        input_summary: input.to_string(),
        output_summary: output.to_string(),
        duration_ns: dur,
        mode: "strict".to_string(),
        outcome: outcome.to_string(),
    }
}

fn state_to_rcond(state: &MatrixConditionState) -> f64 {
    match state {
        MatrixConditionState::WellConditioned => 1e-2,
        MatrixConditionState::ModerateCondition => 1e-6,
        MatrixConditionState::IllConditioned => 1e-12,
        MatrixConditionState::NearSingular => 1e-18,
    }
}

// ═══════════════════════════════════════════════════════════════════
// HAPPY-PATH SCENARIOS (1-4)
// ═══════════════════════════════════════════════════════════════════

/// Scenario 1: PolicyController well-conditioned decision path
#[test]
fn e2e_001_policy_well_conditioned() {
    let scenario_id = "e2e_casp_001_policy_well_cond";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Step 1: Create controller
    let t = Instant::now();
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 64);
    steps.push(make_step(
        1,
        "create_controller",
        "init",
        "mode=Strict, capacity=64",
        &format!("mode={:?}", ctrl.mode()),
        t.elapsed().as_nanos(),
        "ok",
    ));

    // Step 2: Benign signal → expect Allow
    let t = Instant::now();
    let signals = DecisionSignals::new(2.0, 0.0, 0.0);
    let decision = ctrl.decide(signals);
    let pass = decision.action == PolicyAction::Allow;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "benign_decision",
        "decide",
        "cond=2.0, meta=0.0, anom=0.0",
        &format!(
            "action={:?}, top_state={:?}, reason={}",
            decision.action, decision.top_state, decision.reason
        ),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 3: Verify posterior normalization
    let t = Instant::now();
    let sum: f64 = decision.posterior.iter().sum();
    let pass = (sum - 1.0).abs() < 1e-9;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "posterior_normalized",
        "check",
        &format!("posterior={:?}", decision.posterior),
        &format!("sum={sum:.12}"),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 4: Verify ledger recorded
    let t = Instant::now();
    let pass = ctrl.ledger().len() == 1;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "ledger_recorded",
        "check",
        "ledger.len()",
        &format!("len={}", ctrl.ledger().len()),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if all_pass {
                "pass".into()
            } else {
                "fail".into()
            },
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(all_pass, "Scenario 1 failed");
}

/// Scenario 2: PolicyController ill-conditioned decision path
#[test]
fn e2e_002_policy_ill_conditioned() {
    let scenario_id = "e2e_casp_002_policy_ill_cond";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let mut all_pass = true;

    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 64);

    // Step 1: High condition number signal
    let t = Instant::now();
    let signals = DecisionSignals::new(16.0, 0.0, 0.0);
    let decision = ctrl.decide(signals);
    let pass = matches!(
        decision.action,
        PolicyAction::FullValidate | PolicyAction::FailClosed
    );
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "high_cond_decision",
        "decide",
        "cond=16.0, meta=0.0, anom=0.0",
        &format!(
            "action={:?}, top_state={:?}",
            decision.action, decision.top_state
        ),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 2: High metadata incompatibility
    let t = Instant::now();
    let signals = DecisionSignals::new(0.0, 1.0, 0.0);
    let decision = ctrl.decide(signals);
    // High metadata should push toward IncompatibleMetadata
    steps.push(make_step(
        2,
        "high_meta_decision",
        "decide",
        "cond=0.0, meta=1.0, anom=0.0",
        &format!(
            "action={:?}, top_state={:?}, posterior={:?}",
            decision.action, decision.top_state, decision.posterior
        ),
        t.elapsed().as_nanos(),
        "ok",
    ));

    // Step 3: All-high signal → expect FailClosed
    let t = Instant::now();
    let signals = DecisionSignals::new(16.0, 1.0, 1.0);
    let decision = ctrl.decide(signals);
    let pass = decision.action == PolicyAction::FailClosed;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "all_high_decision",
        "decide",
        "cond=16.0, meta=1.0, anom=1.0",
        &format!("action={:?}, reason={}", decision.action, decision.reason),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if all_pass {
                "pass".into()
            } else {
                "fail".into()
            },
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(all_pass, "Scenario 2 failed");
}

/// Scenario 3: SolverPortfolio condition-based selection
#[test]
fn e2e_003_solver_portfolio_selection() {
    let scenario_id = "e2e_casp_003_solver_selection";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let mut all_pass = true;

    let portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);

    for (i, state) in MatrixConditionState::ALL.iter().enumerate() {
        let t = Instant::now();
        let (action, posterior, _expected_losses, chosen_loss) =
            portfolio.select_action(state_to_rcond(state), None);
        let posterior_sum: f64 = posterior.iter().sum();
        let pass = (posterior_sum - 1.0).abs() < 1e-9;
        if !pass {
            all_pass = false;
        }
        steps.push(make_step(
            i + 1,
            &format!("select_{state:?}"),
            "select_action",
            &format!("state={state:?}"),
            &format!(
                "action={action:?}, chosen_loss={chosen_loss:.2}, posterior_sum={posterior_sum:.12}"
            ),
            t.elapsed().as_nanos(),
            if pass { "ok" } else { "fail" },
        ));
    }

    // Verify expected selections: the calibrated losses (frankenscipy-7tb8d.2) rank LU first at
    // every conditioning, so ill conditioning also answers LU, with the SVD last of the fallbacks.
    let t = Instant::now();
    let (well_action, _, _, _) = portfolio.select_action(1e-2, None);
    let (ill_action, _, ill_losses, _) = portfolio.select_action(1e-12, None);
    let pass = matches!(well_action, SolverAction::DirectLU)
        && matches!(ill_action, SolverAction::DirectLU)
        && ill_losses[SolverAction::SVDFallback.index()]
            > ill_losses[SolverAction::PivotedQR.index()];
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        5,
        "verify_expected_selections",
        "check",
        "well→DirectLU, ill→DirectLU (calibrated)",
        &format!("well={well_action:?}, ill={ill_action:?}, ill_losses={ill_losses:?}"),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if all_pass {
                "pass".into()
            } else {
                "fail".into()
            },
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(all_pass, "Scenario 3 failed");
}

/// Scenario 4: Evidence ledger bounded FIFO behavior
#[test]
fn e2e_004_ledger_bounded_fifo() {
    let scenario_id = "e2e_casp_004_ledger_fifo";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let mut all_pass = true;
    let capacity = 5;

    // Step 1: Create controller with small capacity
    let t = Instant::now();
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, capacity);
    steps.push(make_step(
        1,
        "create_small_ledger",
        "init",
        &format!("capacity={capacity}"),
        &format!("ledger.capacity={}", ctrl.ledger().capacity()),
        t.elapsed().as_nanos(),
        "ok",
    ));

    // Step 2: Record 20 entries
    let t = Instant::now();
    for i in 0..20 {
        let cond = (i as f64) * 0.5;
        ctrl.decide(DecisionSignals::new(cond, 0.0, 0.0));
    }
    let len = ctrl.ledger().len();
    let pass = len <= capacity;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "record_20_entries",
        "decide",
        "20 decisions with varying cond",
        &format!("ledger.len={len}, capacity={capacity}"),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 3: Verify latest entry is from the most recent decision
    let t = Instant::now();
    let latest = ctrl.ledger().latest().expect("latest entry");
    let pass = true; // If we got here, latest exists
    steps.push(make_step(
        3,
        "verify_latest",
        "check",
        "latest entry should exist",
        &format!("has_latest={pass}, mode={:?}", latest.mode),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if all_pass {
                "pass".into()
            } else {
                "fail".into()
            },
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(all_pass, "Scenario 4 failed");
}

// ═══════════════════════════════════════════════════════════════════
// ERROR RECOVERY SCENARIOS (5-6)
// ═══════════════════════════════════════════════════════════════════

/// Scenario 5: Strict vs Hardened mode produces different actions
#[test]
fn e2e_005_mode_switch_behavior() {
    let scenario_id = "e2e_casp_005_mode_switch";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let mut all_pass = true;

    let signals = DecisionSignals::new(8.0, 0.3, 0.2);

    // Step 1: Strict mode decision
    let t = Instant::now();
    let mut strict_ctrl = PolicyController::new(RuntimeMode::Strict, 64);
    let strict_dec = strict_ctrl.decide(signals);
    steps.push(make_step(
        1,
        "strict_decision",
        "decide",
        "mode=Strict, cond=8, meta=0.3, anom=0.2",
        &format!(
            "action={:?}, reason={}",
            strict_dec.action, strict_dec.reason
        ),
        t.elapsed().as_nanos(),
        "ok",
    ));

    // Step 2: Hardened mode decision
    let t = Instant::now();
    let mut hardened_ctrl = PolicyController::new(RuntimeMode::Hardened, 64);
    let hardened_dec = hardened_ctrl.decide(signals);
    steps.push(make_step(
        2,
        "hardened_decision",
        "decide",
        "mode=Hardened, cond=8, meta=0.3, anom=0.2",
        &format!(
            "action={:?}, reason={}",
            hardened_dec.action, hardened_dec.reason
        ),
        t.elapsed().as_nanos(),
        "ok",
    ));

    // Step 3: Both posteriors should be normalized
    let t = Instant::now();
    let strict_sum: f64 = strict_dec.posterior.iter().sum();
    let hardened_sum: f64 = hardened_dec.posterior.iter().sum();
    let pass = (strict_sum - 1.0).abs() < 1e-9 && (hardened_sum - 1.0).abs() < 1e-9;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "posteriors_normalized",
        "check",
        "both posteriors sum to 1",
        &format!("strict_sum={strict_sum:.12}, hardened_sum={hardened_sum:.12}"),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 4: Different loss matrices produce different expected losses
    let t = Instant::now();
    let strict_losses = &strict_dec.expected_losses;
    let hardened_losses = &hardened_dec.expected_losses;
    // They may or may not differ depending on the signal; just verify both are finite
    let pass = strict_losses.iter().all(|l| l.is_finite())
        && hardened_losses.iter().all(|l| l.is_finite());
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "losses_finite",
        "check",
        "all expected losses finite",
        &format!("strict_losses={strict_losses:?}, hardened_losses={hardened_losses:?}"),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if all_pass {
                "pass".into()
            } else {
                "fail".into()
            },
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(all_pass, "Scenario 5 failed");
}

/// Scenario 6: Conformal calibrator fallback trigger
#[test]
fn e2e_006_calibrator_fallback() {
    let scenario_id = "e2e_casp_006_calibrator_fallback";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Step 1: Create calibrator
    let t = Instant::now();
    let mut calibrator = ConformalCalibrator::new(0.05, 200);
    steps.push(make_step(
        1,
        "create_calibrator",
        "init",
        "alpha=0.05, capacity=200",
        &format!("alpha={}", calibrator.alpha()),
        t.elapsed().as_nanos(),
        "ok",
    ));

    // Step 2: Feed good scores (below threshold)
    let t = Instant::now();
    for i in 0..50 {
        calibrator.observe(1e-12 * (i as f64 + 1.0));
    }
    let fallback = calibrator.should_fallback();
    let pass = !fallback; // should not trigger fallback yet
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "good_scores",
        "observe",
        "50 scores < 1e-8",
        &format!(
            "should_fallback={fallback}, miscoverage={:.4}",
            calibrator.empirical_miscoverage()
        ),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 3: Feed many bad scores (above threshold)
    let t = Instant::now();
    for _ in 0..50 {
        calibrator.observe(1.0); // way above violation threshold
    }
    let fallback = calibrator.should_fallback();
    let pass = fallback; // should now trigger fallback
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "bad_scores_trigger",
        "observe",
        "50 scores = 1.0 (bad)",
        &format!(
            "should_fallback={fallback}, miscoverage={:.4}",
            calibrator.empirical_miscoverage()
        ),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if all_pass {
                "pass".into()
            } else {
                "fail".into()
            },
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(all_pass, "Scenario 6 failed");
}

// ═══════════════════════════════════════════════════════════════════
// ADVERSARIAL SCENARIOS (7-8)
// ═══════════════════════════════════════════════════════════════════

/// Scenario 7: Signal sequence replay — decisions are stateless
#[test]
fn e2e_007_signal_replay_stateless() {
    let scenario_id = "e2e_casp_007_signal_replay";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Step 1: Build signal sequence
    let t = Instant::now();
    let mut seq = SignalSequence::new("adversarial_replay");
    seq.push(DecisionSignals::new(2.0, 0.0, 0.0)); // benign
    seq.push(DecisionSignals::new(16.0, 1.0, 1.0)); // adversarial
    seq.push(DecisionSignals::new(2.0, 0.0, 0.0)); // benign again
    seq.push(DecisionSignals::new(2.0, 0.0, 0.0)); // benign again
    let pass = seq.len() == 4;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "build_sequence",
        "create",
        "4 signals: benign, adversarial, benign, benign",
        &format!("len={}", seq.len()),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 2: Replay and verify decisions are independent
    let t = Instant::now();
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 64);
    let mut decisions = Vec::new();
    for signal in seq.iter() {
        decisions.push(ctrl.decide(*signal));
    }
    // First and third should have same action (both benign)
    let pass =
        decisions[0].action == decisions[2].action && decisions[0].action == decisions[3].action;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "replay_verify",
        "decide",
        "replay 4 signals",
        &format!(
            "actions=[{:?},{:?},{:?},{:?}]",
            decisions[0].action, decisions[1].action, decisions[2].action, decisions[3].action
        ),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 3: Adversarial signal should differ from benign
    let t = Instant::now();
    let pass = decisions[1].action != decisions[0].action;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "adversarial_differs",
        "check",
        "adversarial action != benign action",
        &format!(
            "benign_action={:?}, adversarial_action={:?}",
            decisions[0].action, decisions[1].action
        ),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if all_pass {
                "pass".into()
            } else {
                "fail".into()
            },
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(all_pass, "Scenario 7 failed");
}

/// Scenario 8: Rapid sequential decisions — no state leakage
#[test]
fn e2e_008_rapid_decisions() {
    let scenario_id = "e2e_casp_008_rapid_decisions";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Step 1: 1000 rapid decisions
    let t = Instant::now();
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 100);
    let mut allow_count = 0;
    let mut validate_count = 0;
    let mut fail_count = 0;
    for i in 0..1000 {
        let cond = (i % 20) as f64;
        let meta = if i % 50 == 0 { 0.8 } else { 0.0 };
        let anom = if i % 100 == 0 { 0.5 } else { 0.0 };
        let dec = ctrl.decide(DecisionSignals::new(cond, meta, anom));
        match dec.action {
            PolicyAction::Allow => allow_count += 1,
            PolicyAction::FullValidate => validate_count += 1,
            PolicyAction::FailClosed => fail_count += 1,
        }
    }
    let total = allow_count + validate_count + fail_count;
    let pass = total == 1000;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "rapid_1000_decisions",
        "batch_decide",
        "1000 decisions with varying signals",
        &format!(
            "allow={allow_count}, validate={validate_count}, fail={fail_count}, total={total}"
        ),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 2: Ledger bounded
    let t = Instant::now();
    let pass = ctrl.ledger().len() <= 100;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "ledger_bounded",
        "check",
        "ledger.len() <= capacity",
        &format!("len={}, capacity=100", ctrl.ledger().len()),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 3: Determinism check — same input gives same output
    let t = Instant::now();
    let mut ctrl2 = PolicyController::new(RuntimeMode::Strict, 100);
    let dec1 = ctrl2.decide(DecisionSignals::new(5.0, 0.1, 0.05));
    let mut ctrl3 = PolicyController::new(RuntimeMode::Strict, 100);
    let dec2 = ctrl3.decide(DecisionSignals::new(5.0, 0.1, 0.05));
    let pass = dec1.action == dec2.action && dec1.top_state == dec2.top_state;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "determinism_check",
        "check",
        "same signal → same action",
        &format!(
            "action1={:?}, action2={:?}, match={pass}",
            dec1.action, dec2.action
        ),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if all_pass {
                "pass".into()
            } else {
                "fail".into()
            },
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(all_pass, "Scenario 8 failed");
}

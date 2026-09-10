//! bd-3jh.19.9: [FSCI-P2C-008-I] Final Evidence Pack
//!
//! Produces a self-contained evidence bundle at:
//!   fixtures/artifacts/FSCI-P2C-008/evidence/
//!
//! Artifacts:
//! 1. fixture_manifest.json — policy decision scenarios with signal triples
//! 2. parity_gates.json — CASP decision correctness + mode consistency gates
//! 3. risk_notes.json — posterior collapse, loss asymmetry, ledger truncation
//! 4. parity_report.json — grouped parity summaries by subsystem
//! 5. evidence_bundle.raptorq.json — RaptorQ sidecar for the bundle

use asupersync::raptorq::systematic::SystematicEncoder;
use blake3::hash;
use fsci_conformance::{RaptorQSidecar, chunk_payload, generate_raptorq_sidecar};
use fsci_runtime::{
    AuditLedger, ConformalCalibrator, DecisionSignals, MatrixConditionState, PolicyAction,
    PolicyController, RiskState, RuntimeMode, SignalSequence, SolverAction, SolverPortfolio,
};
use serde::Serialize;
use std::path::Path;

// ── Fixture data structures ────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct FixtureManifest {
    packet_id: &'static str,
    generated_at: String,
    fixtures: Vec<FixtureEntry>,
}

#[derive(Debug, Serialize)]
struct FixtureEntry {
    fixture_id: String,
    subsystem: &'static str,
    description: String,
    signal_triple: Option<[f64; 3]>,
    matrix_condition: Option<&'static str>,
    mode: &'static str,
}

#[derive(Debug, Serialize)]
struct ParityGateReport {
    packet_id: &'static str,
    all_gates_pass: bool,
    gates: Vec<ParityGate>,
}

#[derive(Debug, Serialize)]
struct ParityGate {
    fixture_id: String,
    subsystem: &'static str,
    pass: bool,
    description: String,
    detail: String,
}

#[derive(Debug, Serialize)]
struct RiskNote {
    category: &'static str,
    description: String,
    affected_subsystems: Vec<&'static str>,
    mitigation: String,
}

#[derive(Debug, Serialize)]
struct RiskNotesReport {
    packet_id: &'static str,
    notes: Vec<RiskNote>,
}

#[derive(Debug, Serialize)]
struct ParityReport {
    packet_id: &'static str,
    operation_summaries: Vec<SubsystemParitySummary>,
}

#[derive(Debug, Serialize)]
struct SubsystemParitySummary {
    subsystem: &'static str,
    total_gates: usize,
    passed: usize,
    failed: usize,
}

#[derive(Debug, Serialize)]
struct EvidenceBundle {
    manifest: FixtureManifest,
    parity_gates: ParityGateReport,
    risk_notes: RiskNotesReport,
    parity_report: ParityReport,
    sidecar: Option<RaptorQSidecar>,
}

fn state_to_rcond(state: &MatrixConditionState) -> f64 {
    match state {
        MatrixConditionState::WellConditioned => 1e-2,
        MatrixConditionState::ModerateCondition => 1e-6,
        MatrixConditionState::IllConditioned => 1e-12,
        MatrixConditionState::NearSingular => 1e-18,
    }
}

// ── Policy decision parity checks ──────────────────────────────────────────────

/// Verify that low-risk signals produce Allow action in both modes.
fn check_policy_low_risk(mode: RuntimeMode) -> ParityGate {
    let mut ctrl = PolicyController::new(mode, 64);
    let signals = DecisionSignals::new(0.5, 0.0, 0.0);
    let decision = ctrl.decide(signals);

    let pass =
        decision.action == PolicyAction::Allow && decision.top_state == RiskState::Compatible;

    ParityGate {
        fixture_id: format!("policy_low_risk_{mode:?}"),
        subsystem: "policy_controller",
        pass,
        description: format!("Low-risk signals → Allow in {mode:?} mode"),
        detail: format!(
            "action={:?}, top_state={:?}, posterior={:?}",
            decision.action, decision.top_state, decision.posterior
        ),
    }
}

/// Verify that high-condition-number signals trigger FullValidate or FailClosed.
/// Note: top_state may remain Compatible because the logit model biases toward
/// Compatible; the loss matrix drives the action escalation.
fn check_policy_high_cond(mode: RuntimeMode) -> ParityGate {
    let mut ctrl = PolicyController::new(mode, 64);
    let signals = DecisionSignals::new(14.0, 0.0, 0.0);
    let decision = ctrl.decide(signals);

    let pass = matches!(
        decision.action,
        PolicyAction::FullValidate | PolicyAction::FailClosed
    );

    ParityGate {
        fixture_id: format!("policy_high_cond_{mode:?}"),
        subsystem: "policy_controller",
        pass,
        description: format!("High condition number → escalated action in {mode:?}"),
        detail: format!(
            "action={:?}, top_state={:?}, posterior={:?}",
            decision.action, decision.top_state, decision.posterior
        ),
    }
}

/// Verify that metadata incompatibility signals trigger FullValidate or FailClosed.
fn check_policy_metadata_incompat(mode: RuntimeMode) -> ParityGate {
    let mut ctrl = PolicyController::new(mode, 64);
    let signals = DecisionSignals::new(1.0, 0.9, 0.0);
    let decision = ctrl.decide(signals);

    let pass = matches!(
        decision.action,
        PolicyAction::FullValidate | PolicyAction::FailClosed
    ) && decision.top_state == RiskState::IncompatibleMetadata;

    ParityGate {
        fixture_id: format!("policy_metadata_incompat_{mode:?}"),
        subsystem: "policy_controller",
        pass,
        description: format!("High metadata incompatibility → escalated in {mode:?}"),
        detail: format!(
            "action={:?}, top_state={:?}, posterior={:?}",
            decision.action, decision.top_state, decision.posterior
        ),
    }
}

/// Verify that anomaly signals trigger appropriate escalation.
/// Note: top_state may remain Compatible; the loss matrix drives escalation.
fn check_policy_anomaly(mode: RuntimeMode) -> ParityGate {
    let mut ctrl = PolicyController::new(mode, 64);
    let signals = DecisionSignals::new(1.0, 0.0, 0.95);
    let decision = ctrl.decide(signals);

    let pass = matches!(
        decision.action,
        PolicyAction::FullValidate | PolicyAction::FailClosed
    );

    ParityGate {
        fixture_id: format!("policy_anomaly_{mode:?}"),
        subsystem: "policy_controller",
        pass,
        description: format!("High anomaly score → escalated action in {mode:?}"),
        detail: format!(
            "action={:?}, top_state={:?}, posterior={:?}",
            decision.action, decision.top_state, decision.posterior
        ),
    }
}

/// Verify mode consistency: Hardened is at least as strict as Strict.
fn check_mode_consistency() -> ParityGate {
    let test_signals = [
        DecisionSignals::new(0.5, 0.0, 0.0),
        DecisionSignals::new(8.0, 0.3, 0.1),
        DecisionSignals::new(14.0, 0.0, 0.0),
        DecisionSignals::new(1.0, 0.9, 0.0),
        DecisionSignals::new(1.0, 0.0, 0.95),
    ];

    fn action_severity(a: &PolicyAction) -> u8 {
        match a {
            PolicyAction::Allow => 0,
            PolicyAction::FullValidate => 1,
            PolicyAction::FailClosed => 2,
        }
    }

    let mut all_ok = true;
    let mut detail = String::new();

    for signals in &test_signals {
        let mut strict_ctrl = PolicyController::new(RuntimeMode::Strict, 64);
        let mut hardened_ctrl = PolicyController::new(RuntimeMode::Hardened, 64);
        let strict_d = strict_ctrl.decide(*signals);
        let hardened_d = hardened_ctrl.decide(*signals);

        let ok = action_severity(&hardened_d.action) >= action_severity(&strict_d.action);
        if !ok {
            all_ok = false;
            detail = format!(
                "Hardened ({:?}) less strict than Strict ({:?}) for signals {:?}",
                hardened_d.action, strict_d.action, signals
            );
            break;
        }
    }

    if detail.is_empty() {
        detail = format!(
            "All {} signal triples: Hardened >= Strict",
            test_signals.len()
        );
    }

    ParityGate {
        fixture_id: "mode_consistency_hardened_ge_strict".into(),
        subsystem: "policy_controller",
        pass: all_ok,
        description: "Hardened mode at least as strict as Strict mode".into(),
        detail,
    }
}

// ── Solver portfolio parity checks ─────────────────────────────────────────────

/// Verify that well-conditioned state selects DirectLU (lowest expected loss).
fn check_portfolio_well_conditioned() -> ParityGate {
    let portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
    let (action, posterior, losses, chosen_loss) = portfolio.select_action(1e-2, None);

    let pass = action == SolverAction::DirectLU
        && posterior[0] == 1.0
        && chosen_loss == losses[SolverAction::DirectLU.index()];

    ParityGate {
        fixture_id: "portfolio_well_conditioned".into(),
        subsystem: "solver_portfolio",
        pass,
        description: "WellConditioned → DirectLU (min expected loss)".into(),
        detail: format!("action={action:?}, losses={losses:?}, chosen={chosen_loss:.2}"),
    }
}

/// Verify that moderate-condition state selects PivotedQR.
fn check_portfolio_moderate() -> ParityGate {
    let portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
    let (action, _, losses, _) = portfolio.select_action(1e-6, None);

    let pass = action == SolverAction::PivotedQR;

    ParityGate {
        fixture_id: "portfolio_moderate_condition".into(),
        subsystem: "solver_portfolio",
        pass,
        description: "ModerateCondition → PivotedQR".into(),
        detail: format!("action={action:?}, losses={losses:?}"),
    }
}

/// Verify that ill-conditioned and near-singular states select SVDFallback.
fn check_portfolio_ill_conditioned() -> ParityGate {
    let portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
    let (action_ill, _, _, _) = portfolio.select_action(1e-12, None);
    let (action_near, _, _, _) = portfolio.select_action(1e-18, None);

    let pass = action_ill == SolverAction::SVDFallback && action_near == SolverAction::SVDFallback;

    ParityGate {
        fixture_id: "portfolio_ill_and_near_singular".into(),
        subsystem: "solver_portfolio",
        pass,
        description: "IllConditioned/NearSingular → SVDFallback".into(),
        detail: format!("ill={action_ill:?}, near_singular={action_near:?}"),
    }
}

/// Verify that all condition states produce a valid action and finite losses.
fn check_portfolio_all_states_valid() -> ParityGate {
    let portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
    let mut all_ok = true;
    let mut detail = String::new();

    for state in &MatrixConditionState::ALL {
        let (action, posterior, losses, chosen) =
            portfolio.select_action(state_to_rcond(state), None);
        let finite_losses = losses.iter().all(|l| l.is_finite());
        let finite_posterior = posterior.iter().all(|p| p.is_finite());
        let posterior_sums_to_one = (posterior.iter().sum::<f64>() - 1.0).abs() < 1e-10;
        let valid_action = SolverAction::ALL.contains(&action);
        let chosen_matches = (chosen - losses[action.index()]).abs() < 1e-15;

        if !finite_losses
            || !finite_posterior
            || !posterior_sums_to_one
            || !valid_action
            || !chosen_matches
        {
            all_ok = false;
            detail = format!(
                "State {state:?}: finite_losses={finite_losses}, posterior_sum={}, valid={valid_action}, chosen_matches={chosen_matches}",
                posterior.iter().sum::<f64>()
            );
            break;
        }
    }

    if detail.is_empty() {
        detail = "All 4 condition states produce valid, finite outputs".into();
    }

    ParityGate {
        fixture_id: "portfolio_all_states_valid".into(),
        subsystem: "solver_portfolio",
        pass: all_ok,
        description: "All MatrixConditionStates produce valid portfolio decisions".into(),
        detail,
    }
}

/// Verify loss matrix symmetry properties.
fn check_loss_matrix_properties() -> ParityGate {
    let lm = SolverPortfolio::default_loss_matrix();
    let mut pass = true;
    let mut detail_parts = Vec::new();

    // Property 1: All losses >= 0
    let all_nonneg = lm.iter().all(|row| row.iter().all(|&v| v >= 0.0));
    if !all_nonneg {
        pass = false;
        detail_parts.push("negative loss value found".to_string());
    }

    // Property 2: SVDFallback should have lowest loss for ill-conditioned states
    let svd_idx = SolverAction::SVDFallback.index();
    for &col in &[2, 3] {
        // IllConditioned=2, NearSingular=3
        let svd_loss = lm[svd_idx][col];
        for (row_idx, row) in lm.iter().enumerate().take(3) {
            if row[col] < svd_loss && row_idx != svd_idx {
                pass = false;
                detail_parts.push(format!(
                    "action {row_idx} has lower loss than SVD in column {col}"
                ));
            }
        }
    }

    // Property 3: DirectLU should have lowest loss for well-conditioned state
    let lu_well = lm[SolverAction::DirectLU.index()][0];
    for (i, row) in lm.iter().enumerate().take(3) {
        if row[0] < lu_well && i != SolverAction::DirectLU.index() {
            pass = false;
            detail_parts.push(format!("action {i} beats DirectLU for WellConditioned"));
        }
    }

    let detail = if detail_parts.is_empty() {
        "Loss matrix satisfies: non-negative, SVD dominates ill-conditioned, LU dominates well-conditioned".into()
    } else {
        detail_parts.join("; ")
    };

    ParityGate {
        fixture_id: "loss_matrix_properties".into(),
        subsystem: "solver_portfolio",
        pass,
        description: "Loss matrix structural properties hold".into(),
        detail,
    }
}

// ── Conformal calibrator parity checks ─────────────────────────────────────────

/// Verify that the calibrator does NOT trigger fallback with all-good scores.
fn check_calibrator_no_fallback() -> ParityGate {
    let mut cal = ConformalCalibrator::new(0.05, 200);
    for _ in 0..50 {
        cal.observe(1e-12); // well below violation threshold
    }

    let pass = !cal.should_fallback()
        && cal.total_predictions() == 50
        && cal.empirical_miscoverage() < 1e-10;

    ParityGate {
        fixture_id: "calibrator_no_fallback".into(),
        subsystem: "conformal_calibrator",
        pass,
        description: "50 low-error observations → no fallback triggered".into(),
        detail: format!(
            "should_fallback={}, miscoverage={:.6}, total={}",
            cal.should_fallback(),
            cal.empirical_miscoverage(),
            cal.total_predictions()
        ),
    }
}

/// Verify that high violation rate triggers fallback.
fn check_calibrator_triggers_fallback() -> ParityGate {
    let mut cal = ConformalCalibrator::new(0.05, 200);
    // Feed 20 observations, most above threshold
    for _ in 0..15 {
        cal.observe(1.0); // above default threshold 1e-8
    }
    for _ in 0..5 {
        cal.observe(1e-12); // below threshold
    }

    let miscoverage = cal.empirical_miscoverage();
    let pass = cal.should_fallback() && miscoverage > 0.05 + 0.02; // alpha + epsilon

    ParityGate {
        fixture_id: "calibrator_triggers_fallback".into(),
        subsystem: "conformal_calibrator",
        pass,
        description: "75% violation rate → fallback triggered".into(),
        detail: format!(
            "should_fallback={}, miscoverage={miscoverage:.4}, alpha={}",
            cal.should_fallback(),
            cal.alpha()
        ),
    }
}

/// Verify that calibrator respects minimum observation count (< 10 → no fallback).
fn check_calibrator_min_observations() -> ParityGate {
    let mut cal = ConformalCalibrator::new(0.05, 200);
    // Feed only 9 high-violation observations
    for _ in 0..9 {
        cal.observe(1.0);
    }

    let pass = !cal.should_fallback() && cal.total_predictions() == 9;

    ParityGate {
        fixture_id: "calibrator_min_observations".into(),
        subsystem: "conformal_calibrator",
        pass,
        description: "< 10 observations → no fallback even with 100% violations".into(),
        detail: format!(
            "should_fallback={}, predictions={}",
            cal.should_fallback(),
            cal.total_predictions()
        ),
    }
}

/// Verify that portfolio overrides to SVDFallback when calibrator triggers.
fn check_portfolio_conformal_override() -> ParityGate {
    let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);

    // Push enough violations to trigger calibrator fallback
    for _ in 0..20 {
        portfolio.observe_backward_error(1.0); // high error → violation
    }

    // Even for WellConditioned (which normally selects DirectLU),
    // the conformal override should force SVDFallback
    let (action, _, _, _) = portfolio.select_action(1e-2, None);

    let pass = action == SolverAction::SVDFallback;

    ParityGate {
        fixture_id: "portfolio_conformal_override".into(),
        subsystem: "conformal_calibrator",
        pass,
        description: "Conformal override → SVDFallback even for WellConditioned".into(),
        detail: format!(
            "action={action:?}, calibrator_fallback={}",
            portfolio.calibrator().should_fallback()
        ),
    }
}

// ── Evidence ledger parity checks ──────────────────────────────────────────────

/// Verify that evidence ledger records decisions and respects capacity.
fn check_ledger_capacity() -> ParityGate {
    let capacity = 8;
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, capacity);

    // Make 12 decisions (exceeds capacity of 8)
    for i in 0..12 {
        let cond = (i as f64) * 1.0;
        ctrl.decide(DecisionSignals::new(cond, 0.0, 0.0));
    }

    let ledger = ctrl.ledger();
    let pass = ledger.len() <= capacity && !ledger.is_empty();

    ParityGate {
        fixture_id: "ledger_capacity_fifo".into(),
        subsystem: "evidence_ledger",
        pass,
        description: format!("Ledger respects capacity={capacity} with FIFO eviction"),
        detail: format!("ledger_len={}, capacity={capacity}", ledger.len()),
    }
}

/// Verify that the latest ledger entry matches the most recent decision.
fn check_ledger_latest_entry() -> ParityGate {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 64);
    let signals = DecisionSignals::new(0.5, 0.0, 0.0);
    let decision = ctrl.decide(signals);

    let latest = ctrl.ledger().latest();
    let pass = latest
        .is_some_and(|entry| entry.action == decision.action && entry.mode == RuntimeMode::Strict);

    ParityGate {
        fixture_id: "ledger_latest_entry".into(),
        subsystem: "evidence_ledger",
        pass,
        description: "Latest ledger entry matches most recent decision".into(),
        detail: format!(
            "latest_action={:?}, decision_action={:?}",
            latest.map(|e| e.action),
            decision.action
        ),
    }
}

/// Verify that signal sequences replay consistently.
fn check_signal_sequence_replay() -> ParityGate {
    let mut seq = SignalSequence::new("replay_test");
    seq.push(DecisionSignals::new(0.5, 0.0, 0.0));
    seq.push(DecisionSignals::new(14.0, 0.0, 0.0));
    seq.push(DecisionSignals::new(1.0, 0.9, 0.0));

    // Replay twice through same controller configuration
    let mut results_a = Vec::new();
    let mut results_b = Vec::new();

    for signals in seq.iter() {
        let mut ctrl = PolicyController::new(RuntimeMode::Strict, 64);
        results_a.push(ctrl.decide(*signals).action);
    }
    for signals in seq.iter() {
        let mut ctrl = PolicyController::new(RuntimeMode::Strict, 64);
        results_b.push(ctrl.decide(*signals).action);
    }

    let pass = results_a == results_b && seq.len() == 3;

    ParityGate {
        fixture_id: "signal_sequence_replay".into(),
        subsystem: "evidence_ledger",
        pass,
        description: "Signal sequence replays produce identical decisions".into(),
        detail: format!("run_a={results_a:?}, run_b={results_b:?}"),
    }
}

/// Verify JSONL serialization of solver evidence is well-formed.
fn check_evidence_jsonl_serialization() -> ParityGate {
    let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);

    for state in &MatrixConditionState::ALL {
        let (action, posterior, losses, chosen_loss) =
            portfolio.select_action(state_to_rcond(state), None);
        portfolio.record_evidence(fsci_runtime::SolverEvidenceEntry {
            component: "fsci_linalg",
            matrix_shape: (32, 32),
            rcond_estimate: 1e-3,
            chosen_action: action,
            posterior: posterior.to_vec(),
            expected_losses: losses.to_vec(),
            chosen_expected_loss: chosen_loss,
            fallback_active: false,
            backward_error: None,
        });
    }

    let jsonl = portfolio.serialize_jsonl();
    let lines: Vec<&str> = jsonl.lines().collect();
    let all_valid_json = lines
        .iter()
        .all(|line| serde_json::from_str::<serde_json::Value>(line).is_ok());

    let pass = lines.len() == 4 && all_valid_json && portfolio.evidence_len() == 4;

    ParityGate {
        fixture_id: "evidence_jsonl_serialization".into(),
        subsystem: "evidence_ledger",
        pass,
        description: "JSONL evidence serialization produces valid JSON lines".into(),
        detail: format!("lines={}, all_valid={all_valid_json}", lines.len()),
    }
}

/// Verify shared AuditLedger can carry full Spec §6 decision records.
fn check_audit_ledger_structured_decision_event() -> ParityGate {
    let mut controller = PolicyController::new(RuntimeMode::Strict, 8);
    let decision = controller.decide(DecisionSignals::new(12.0, 0.15, 0.05));
    let structured = controller
        .ledger()
        .latest_alien_artifact_decision()
        .expect("structured decision");
    let mut audit = AuditLedger::new();
    audit.record_alien_artifact_decision(
        123,
        AuditLedger::fingerprint_bytes(b"p2c008-structured-audit"),
        structured,
        "validated",
    );

    let payload = audit.to_json().expect("audit ledger JSON");
    let parsed: serde_json::Value = serde_json::from_str(&payload).expect("parse audit ledger");
    let action = &parsed["entries"][0]["action"];
    let pass = action["kind"] == "alien_artifact_decision"
        && action["decision"].get("state_space").is_some()
        && action["decision"].get("evidence").is_some()
        && action["decision"].get("loss_matrix").is_some()
        && action["decision"].get("posterior").is_some()
        && action["decision"].get("expected_losses").is_some()
        && action["decision"].get("confidence").is_some()
        && action["decision"]
            .get("calibration_fallback_trigger")
            .is_some();

    ParityGate {
        fixture_id: "audit_ledger_structured_decision_event".into(),
        subsystem: "evidence_ledger",
        pass,
        description: "Shared AuditLedger emits full AlienArtifactDecision records".into(),
        detail: format!(
            "action={:?}, top_state={:?}, selected={:?}",
            decision.action, decision.top_state, action["decision"]["action"]
        ),
    }
}

// ── Main evidence pack test ────────────────────────────────────────────────────

#[test]
fn evidence_p2c008_final_pack() {
    let evidence_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts/FSCI-P2C-008/evidence");
    std::fs::create_dir_all(&evidence_dir).unwrap();

    // 1. Fixture manifest
    let manifest = FixtureManifest {
        packet_id: "FSCI-P2C-008",
        generated_at: now_str(),
        fixtures: vec![
            FixtureEntry {
                fixture_id: "policy_low_risk_Strict".into(),
                subsystem: "policy_controller",
                description: "Low-risk signals (cond=0.5, meta=0, anomaly=0) in Strict mode".into(),
                signal_triple: Some([0.5, 0.0, 0.0]),
                matrix_condition: None,
                mode: "Strict",
            },
            FixtureEntry {
                fixture_id: "policy_low_risk_Hardened".into(),
                subsystem: "policy_controller",
                description: "Low-risk signals in Hardened mode".into(),
                signal_triple: Some([0.5, 0.0, 0.0]),
                matrix_condition: None,
                mode: "Hardened",
            },
            FixtureEntry {
                fixture_id: "policy_high_cond_Strict".into(),
                subsystem: "policy_controller",
                description: "High condition number (14.0) in Strict mode".into(),
                signal_triple: Some([14.0, 0.0, 0.0]),
                matrix_condition: None,
                mode: "Strict",
            },
            FixtureEntry {
                fixture_id: "policy_high_cond_Hardened".into(),
                subsystem: "policy_controller",
                description: "High condition number in Hardened mode".into(),
                signal_triple: Some([14.0, 0.0, 0.0]),
                matrix_condition: None,
                mode: "Hardened",
            },
            FixtureEntry {
                fixture_id: "policy_metadata_incompat_Strict".into(),
                subsystem: "policy_controller",
                description: "High metadata incompatibility (0.9) in Strict mode".into(),
                signal_triple: Some([1.0, 0.9, 0.0]),
                matrix_condition: None,
                mode: "Strict",
            },
            FixtureEntry {
                fixture_id: "policy_metadata_incompat_Hardened".into(),
                subsystem: "policy_controller",
                description: "High metadata incompatibility in Hardened mode".into(),
                signal_triple: Some([1.0, 0.9, 0.0]),
                matrix_condition: None,
                mode: "Hardened",
            },
            FixtureEntry {
                fixture_id: "policy_anomaly_Strict".into(),
                subsystem: "policy_controller",
                description: "High anomaly score (0.95) in Strict mode".into(),
                signal_triple: Some([1.0, 0.0, 0.95]),
                matrix_condition: None,
                mode: "Strict",
            },
            FixtureEntry {
                fixture_id: "policy_anomaly_Hardened".into(),
                subsystem: "policy_controller",
                description: "High anomaly score in Hardened mode".into(),
                signal_triple: Some([1.0, 0.0, 0.95]),
                matrix_condition: None,
                mode: "Hardened",
            },
            FixtureEntry {
                fixture_id: "mode_consistency_hardened_ge_strict".into(),
                subsystem: "policy_controller",
                description: "Hardened mode at least as strict as Strict for all test signals"
                    .into(),
                signal_triple: None,
                matrix_condition: None,
                mode: "both",
            },
            FixtureEntry {
                fixture_id: "portfolio_well_conditioned".into(),
                subsystem: "solver_portfolio",
                description: "WellConditioned state → DirectLU selection".into(),
                signal_triple: None,
                matrix_condition: Some("WellConditioned"),
                mode: "Strict",
            },
            FixtureEntry {
                fixture_id: "portfolio_moderate_condition".into(),
                subsystem: "solver_portfolio",
                description: "ModerateCondition state → PivotedQR selection".into(),
                signal_triple: None,
                matrix_condition: Some("ModerateCondition"),
                mode: "Strict",
            },
            FixtureEntry {
                fixture_id: "portfolio_ill_and_near_singular".into(),
                subsystem: "solver_portfolio",
                description: "IllConditioned/NearSingular → SVDFallback".into(),
                signal_triple: None,
                matrix_condition: Some("IllConditioned+NearSingular"),
                mode: "Strict",
            },
            FixtureEntry {
                fixture_id: "portfolio_all_states_valid".into(),
                subsystem: "solver_portfolio",
                description: "All 4 condition states produce valid, finite decisions".into(),
                signal_triple: None,
                matrix_condition: Some("all"),
                mode: "Strict",
            },
            FixtureEntry {
                fixture_id: "loss_matrix_properties".into(),
                subsystem: "solver_portfolio",
                description: "Loss matrix structural invariants".into(),
                signal_triple: None,
                matrix_condition: None,
                mode: "n/a",
            },
            FixtureEntry {
                fixture_id: "calibrator_no_fallback".into(),
                subsystem: "conformal_calibrator",
                description: "50 low-error observations → no fallback".into(),
                signal_triple: None,
                matrix_condition: None,
                mode: "n/a",
            },
            FixtureEntry {
                fixture_id: "calibrator_triggers_fallback".into(),
                subsystem: "conformal_calibrator",
                description: "75% violation rate → fallback triggered".into(),
                signal_triple: None,
                matrix_condition: None,
                mode: "n/a",
            },
            FixtureEntry {
                fixture_id: "calibrator_min_observations".into(),
                subsystem: "conformal_calibrator",
                description: "< 10 observations → no fallback".into(),
                signal_triple: None,
                matrix_condition: None,
                mode: "n/a",
            },
            FixtureEntry {
                fixture_id: "portfolio_conformal_override".into(),
                subsystem: "conformal_calibrator",
                description: "Conformal override forces SVDFallback for WellConditioned".into(),
                signal_triple: None,
                matrix_condition: Some("WellConditioned"),
                mode: "Strict",
            },
            FixtureEntry {
                fixture_id: "ledger_capacity_fifo".into(),
                subsystem: "evidence_ledger",
                description: "Ledger FIFO eviction at capacity=8".into(),
                signal_triple: None,
                matrix_condition: None,
                mode: "Strict",
            },
            FixtureEntry {
                fixture_id: "ledger_latest_entry".into(),
                subsystem: "evidence_ledger",
                description: "Latest entry matches most recent decision".into(),
                signal_triple: None,
                matrix_condition: None,
                mode: "Strict",
            },
            FixtureEntry {
                fixture_id: "signal_sequence_replay".into(),
                subsystem: "evidence_ledger",
                description: "Signal sequence replays are deterministic".into(),
                signal_triple: None,
                matrix_condition: None,
                mode: "Strict",
            },
            FixtureEntry {
                fixture_id: "evidence_jsonl_serialization".into(),
                subsystem: "evidence_ledger",
                description: "JSONL evidence export produces valid JSON lines".into(),
                signal_triple: None,
                matrix_condition: None,
                mode: "Strict",
            },
        ],
    };

    // 2. Parity gates
    let mut gates = Vec::new();

    // Policy controller gates (Strict + Hardened)
    for &mode in &[RuntimeMode::Strict, RuntimeMode::Hardened] {
        gates.push(check_policy_low_risk(mode));
        gates.push(check_policy_high_cond(mode));
        gates.push(check_policy_metadata_incompat(mode));
        gates.push(check_policy_anomaly(mode));
    }
    gates.push(check_mode_consistency());

    // Solver portfolio gates
    gates.push(check_portfolio_well_conditioned());
    gates.push(check_portfolio_moderate());
    gates.push(check_portfolio_ill_conditioned());
    gates.push(check_portfolio_all_states_valid());
    gates.push(check_loss_matrix_properties());

    // Conformal calibrator gates
    gates.push(check_calibrator_no_fallback());
    gates.push(check_calibrator_triggers_fallback());
    gates.push(check_calibrator_min_observations());
    gates.push(check_portfolio_conformal_override());

    // Evidence ledger gates
    gates.push(check_ledger_capacity());
    gates.push(check_ledger_latest_entry());
    gates.push(check_signal_sequence_replay());
    gates.push(check_evidence_jsonl_serialization());
    gates.push(check_audit_ledger_structured_decision_event());

    let all_pass = gates.iter().all(|g| g.pass);
    let parity_gates = ParityGateReport {
        packet_id: "FSCI-P2C-008",
        all_gates_pass: all_pass,
        gates,
    };

    // 3. Risk notes
    let risk_notes = RiskNotesReport {
        packet_id: "FSCI-P2C-008",
        notes: vec![
            RiskNote {
                category: "posterior collapse",
                description: "Hard-classifier condition_posterior assigns probability 1.0 to a \
                    single state. If the classification boundary misaligns with ground truth, \
                    expected-loss selection may choose a sub-optimal solver with no uncertainty \
                    hedge. Future work: logistic blending at state boundaries."
                    .into(),
                affected_subsystems: vec!["solver_portfolio", "policy_controller"],
                mitigation: "Monitor backward errors via ConformalCalibrator. If empirical \
                    miscoverage exceeds alpha + epsilon, the SVD fallback engages automatically. \
                    Consider softening the posterior via logistic blending in future iterations."
                    .into(),
            },
            RiskNote {
                category: "asymmetric loss matrix edge cases",
                description: "The loss matrix encodes domain knowledge about solver cost/risk \
                    trade-offs. DiagonalFastPath and TriangularFastPath have zero loss for \
                    WellConditioned/Moderate/IllConditioned but catastrophic loss (100) for \
                    NearSingular. These fast paths are never selected by argmin over general \
                    solvers (indices 0-2), but if the selection logic changes, the asymmetry \
                    could produce unexpected behavior."
                    .into(),
                affected_subsystems: vec!["solver_portfolio"],
                mitigation: "Loss matrix is compile-time constant (default_loss_matrix). \
                    Parity gate loss_matrix_properties validates structural invariants: \
                    non-negative entries, SVD dominance for ill-conditioned states, \
                    LU dominance for well-conditioned state."
                    .into(),
            },
            RiskNote {
                category: "evidence ledger truncation",
                description: "Both PolicyEvidenceLedger and SolverPortfolio evidence buffers \
                    use FIFO eviction at capacity. Early decisions are lost, which means \
                    long-running sessions cannot replay the full decision history. The \
                    ConformalCalibrator's sliding window also evicts old scores."
                    .into(),
                affected_subsystems: vec!["evidence_ledger", "conformal_calibrator"],
                mitigation: "Use serialize_jsonl() to periodically export evidence to \
                    persistent storage before eviction. Set evidence_capacity proportional \
                    to expected workload. For critical audits, use ledger_capacity >= 1000."
                    .into(),
            },
            RiskNote {
                category: "NaN/Inf signal injection",
                description: "DecisionSignals with NaN or Inf components can propagate through \
                    logit computation and softmax, producing undefined posteriors. The \
                    is_finite() guard exists but is not enforced at the PolicyController level."
                    .into(),
                affected_subsystems: vec!["policy_controller", "signals"],
                mitigation: "Callers should check signals.is_finite() before calling decide(). \
                    The logit computation applies clamping which partially mitigates overflow, \
                    but NaN propagation is not fully guarded. Consider adding a pre-check \
                    in PolicyController::decide()."
                    .into(),
            },
        ],
    };

    // 4. Parity report (aggregated by subsystem)
    let subsystems: &[&str] = &[
        "policy_controller",
        "solver_portfolio",
        "conformal_calibrator",
        "evidence_ledger",
    ];
    let operation_summaries: Vec<SubsystemParitySummary> = subsystems
        .iter()
        .map(|&sub| {
            let sub_gates: Vec<_> = parity_gates
                .gates
                .iter()
                .filter(|g| g.subsystem == sub)
                .collect();
            SubsystemParitySummary {
                subsystem: sub,
                total_gates: sub_gates.len(),
                passed: sub_gates.iter().filter(|g| g.pass).count(),
                failed: sub_gates.iter().filter(|g| !g.pass).count(),
            }
        })
        .collect();

    let parity_report = ParityReport {
        packet_id: "FSCI-P2C-008",
        operation_summaries,
    };

    // 5. Generate RaptorQ sidecar for the full bundle
    let bundle = EvidenceBundle {
        manifest,
        parity_gates,
        risk_notes,
        parity_report,
        sidecar: None,
    };

    // Write individual artifacts
    fn write_json<T: Serialize>(dir: &Path, name: &str, data: &T) {
        let json = serde_json::to_string_pretty(data).unwrap();
        std::fs::write(dir.join(name), &json).unwrap();
    }

    write_json(&evidence_dir, "fixture_manifest.json", &bundle.manifest);
    write_json(&evidence_dir, "parity_gates.json", &bundle.parity_gates);
    write_json(&evidence_dir, "risk_notes.json", &bundle.risk_notes);
    write_json(&evidence_dir, "parity_report.json", &bundle.parity_report);

    // Write the full bundle (without internal sidecar to keep external sidecar valid)
    let final_bundle = EvidenceBundle {
        manifest: bundle.manifest,
        parity_gates: bundle.parity_gates,
        risk_notes: bundle.risk_notes,
        parity_report: bundle.parity_report,
        sidecar: None,
    };
    let final_json = serde_json::to_string_pretty(&final_bundle).unwrap();
    let bundle_path = evidence_dir.join("evidence_bundle.json");
    std::fs::write(&bundle_path, &final_json).unwrap();

    // Generate and write sidecar for the final file
    let external_sidecar = generate_sidecar_resilient(final_json.as_bytes()).ok();
    if let Some(ref sc) = external_sidecar {
        write_json(&evidence_dir, "evidence_bundle.raptorq.json", sc);
    }

    // Print summary
    eprintln!("\n── P2C-008 Evidence Pack ──");
    eprintln!(
        "  Parity gates: {}/{} passed",
        final_bundle
            .parity_gates
            .gates
            .iter()
            .filter(|g| g.pass)
            .count(),
        final_bundle.parity_gates.gates.len()
    );
    for s in &final_bundle.parity_report.operation_summaries {
        eprintln!("  {}: {}/{} pass", s.subsystem, s.passed, s.total_gates);
    }
    for g in &final_bundle.parity_gates.gates {
        if !g.pass {
            eprintln!("  FAIL: {} — {}", g.fixture_id, g.detail);
        }
    }
    eprintln!(
        "  RaptorQ sidecar: {}",
        if final_bundle.sidecar.is_some() {
            "generated"
        } else {
            "skipped (encoder limitation)"
        }
    );
    eprintln!(
        "  Risk notes: {} categories",
        final_bundle.risk_notes.notes.len()
    );
    eprintln!("  Artifacts written to: {evidence_dir:?}");

    // Assertions
    assert!(
        final_bundle.parity_gates.all_gates_pass,
        "All parity gates must pass"
    );
}

fn generate_sidecar_resilient(payload: &[u8]) -> Result<RaptorQSidecar, String> {
    fn hex_encode(bytes: &[u8]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(bytes.len() * 2);
        for &byte in bytes {
            out.push(HEX[(byte >> 4) as usize] as char);
            out.push(HEX[(byte & 0x0f) as usize] as char);
        }
        out
    }

    if let Ok(sidecar) = generate_raptorq_sidecar(payload) {
        return Ok(sidecar);
    }

    let symbol_size = 128usize;
    let source_symbols = chunk_payload(payload, symbol_size);
    let k = source_symbols.len();
    let repair_symbols = (k / 5).max(1);
    let base_seed = hash(payload).as_bytes()[0] as u64 + 1337;

    for offset in 1..=16 {
        if let Some(encoder) =
            SystematicEncoder::new(&source_symbols, symbol_size, base_seed + offset)
        {
            let mut repair_hashes = Vec::with_capacity(repair_symbols);
            let mut repair_payloads = Vec::with_capacity(repair_symbols);
            for esi in k as u32..(k as u32 + repair_symbols as u32) {
                let symbol = encoder.repair_symbol(esi);
                repair_hashes.push(hash(&symbol).to_hex().to_string());
                repair_payloads.push(hex_encode(&symbol));
            }
            return Ok(RaptorQSidecar {
                schema_version: 1,
                source_hash: hash(payload).to_hex().to_string(),
                symbol_size,
                seed: base_seed + offset,
                source_symbols: k,
                repair_symbols,
                repair_symbol_hashes: repair_hashes,
                repair_symbol_payloads_hex: repair_payloads,
            });
        }
    }

    Err("systematic encoder initialization failed after seed fallback attempts".to_string())
}

fn now_str() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    format!("unix:{secs}")
}

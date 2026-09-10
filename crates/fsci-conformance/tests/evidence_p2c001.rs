//! bd-3jh.12.9: [FSCI-P2C-001-I] Final Evidence Pack
//!
//! Produces a self-contained evidence bundle at:
//!   fixtures/artifacts/FSCI-P2C-001/evidence/
//!
//! Covers IVP solver validation, step-size selection, and explicit RK
//! integration with parity gates and risk notes.

use blake3::hash;
use fsci_conformance::{RaptorQSidecar, generate_raptorq_sidecar};
use fsci_integrate::{
    SolveIvpOptions, SolverKind, ToleranceValue, solve_ivp, validate_first_step, validate_max_step,
    validate_tol,
};
use fsci_runtime::RuntimeMode;
use serde::Serialize;
use std::path::Path;

#[derive(Debug, Serialize)]
struct FixtureManifest {
    packet_id: &'static str,
    generated_at: String,
    fixtures: Vec<FixtureEntry>,
}

#[derive(Debug, Serialize)]
struct FixtureEntry {
    fixture_id: String,
    size: usize,
    input_type: &'static str,
    operations: Vec<&'static str>,
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
    operation: &'static str,
    pass: bool,
    max_abs_diff: f64,
    tolerance_used: f64,
    size: usize,
}

#[derive(Debug, Serialize)]
struct RiskNote {
    category: &'static str,
    description: String,
    affected_operations: Vec<&'static str>,
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
    operation_summaries: Vec<OperationParitySummary>,
}

#[derive(Debug, Serialize)]
struct OperationParitySummary {
    operation: &'static str,
    total_fixtures: usize,
    passed: usize,
    failed: usize,
    max_abs_diff_across_all: f64,
}

#[derive(Debug, Serialize)]
struct EvidenceBundle {
    manifest: FixtureManifest,
    parity_gates: ParityGateReport,
    risk_notes: RiskNotesReport,
    parity_report: ParityReport,
    sidecar: Option<RaptorQSidecar>,
}

// ── Helpers ──────────────────────────────────────────────────────────

fn now_str() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    format!("unix:{secs}")
}

/// dy/dt = -y, y(0) = 1  →  y(t) = e^(-t)
fn exponential_decay(_t: f64, y: &[f64]) -> Vec<f64> {
    vec![-y[0]]
}

/// Harmonic oscillator: dy/dt = [y1, -y0]
fn harmonic(_t: f64, y: &[f64]) -> Vec<f64> {
    vec![y[1], -y[0]]
}

/// Lotka-Volterra: dx/dt = ax - bxy, dy/dt = dxy - cy
fn lotka_volterra(_t: f64, y: &[f64]) -> Vec<f64> {
    let (a, b, d, c) = (1.5, 1.0, 1.0, 3.0);
    vec![a * y[0] - b * y[0] * y[1], d * y[0] * y[1] - c * y[1]]
}

// ── Parity checks ──────────────────────────────────────────────────

fn check_validate_tol() -> ParityGate {
    let result = validate_tol(
        ToleranceValue::Scalar(1e-6),
        ToleranceValue::Scalar(1e-9),
        3,
        RuntimeMode::Strict,
    );
    let pass = result.is_ok() && result.unwrap().warnings.is_empty();
    ParityGate {
        fixture_id: "validate_tol_scalar".into(),
        operation: "validate_tol",
        pass,
        max_abs_diff: 0.0,
        tolerance_used: 0.0,
        size: 3,
    }
}

fn check_validate_tol_clamping() -> ParityGate {
    let min_rtol = 100.0 * f64::EPSILON;
    let result = validate_tol(
        ToleranceValue::Scalar(min_rtol / 10.0),
        ToleranceValue::Scalar(1e-15),
        1,
        RuntimeMode::Strict,
    );
    let pass = result.is_ok() && result.unwrap().warnings.len() == 1;
    ParityGate {
        fixture_id: "validate_tol_clamping".into(),
        operation: "validate_tol",
        pass,
        max_abs_diff: 0.0,
        tolerance_used: min_rtol,
        size: 1,
    }
}

fn check_validate_first_step() -> ParityGate {
    let ok1 = validate_first_step(0.01, 0.0, 10.0).is_ok();
    let ok2 = validate_first_step(-0.01, 0.0, 10.0).is_err(); // negative → error
    let ok3 = validate_first_step(20.0, 0.0, 10.0).is_err(); // too large → error
    let pass = ok1 && ok2 && ok3;
    ParityGate {
        fixture_id: "validate_first_step".into(),
        operation: "validate_first_step",
        pass,
        max_abs_diff: 0.0,
        tolerance_used: 0.0,
        size: 3,
    }
}

fn check_validate_max_step() -> ParityGate {
    let ok1 = validate_max_step(1.0).is_ok();
    let ok2 = validate_max_step(0.0).is_err();
    let ok3 = validate_max_step(-1.0).is_err();
    let pass = ok1 && ok2 && ok3;
    ParityGate {
        fixture_id: "validate_max_step".into(),
        operation: "validate_max_step",
        pass,
        max_abs_diff: 0.0,
        tolerance_used: 0.0,
        size: 3,
    }
}

fn check_exponential_decay_rk45() -> ParityGate {
    let mut fun = exponential_decay;
    let result = solve_ivp(
        &mut fun,
        &SolveIvpOptions {
            t_span: (0.0, 5.0),
            y0: &[1.0],
            method: SolverKind::Rk45,
            rtol: 1e-8,
            atol: ToleranceValue::Scalar(1e-10),
            ..Default::default()
        },
    )
    .expect("solve_ivp");
    let y_final = result.y.last().unwrap()[0];
    let expected = (-5.0_f64).exp();
    let max_abs = (y_final - expected).abs();
    ParityGate {
        fixture_id: "exponential_decay_rk45".into(),
        operation: "solve_ivp_rk45",
        pass: result.success && max_abs < 1e-6,
        max_abs_diff: max_abs,
        tolerance_used: 1e-6,
        size: 1,
    }
}

fn check_exponential_decay_rk23() -> ParityGate {
    let mut fun = exponential_decay;
    let result = solve_ivp(
        &mut fun,
        &SolveIvpOptions {
            t_span: (0.0, 5.0),
            y0: &[1.0],
            method: SolverKind::Rk23,
            rtol: 1e-6,
            atol: ToleranceValue::Scalar(1e-9),
            ..Default::default()
        },
    )
    .expect("solve_ivp");
    let y_final = result.y.last().unwrap()[0];
    let expected = (-5.0_f64).exp();
    let max_abs = (y_final - expected).abs();
    ParityGate {
        fixture_id: "exponential_decay_rk23".into(),
        operation: "solve_ivp_rk23",
        pass: result.success && max_abs < 1e-4,
        max_abs_diff: max_abs,
        tolerance_used: 1e-4,
        size: 1,
    }
}

fn check_harmonic_oscillator() -> ParityGate {
    let mut fun = harmonic;
    let two_pi = 2.0 * std::f64::consts::PI;
    let result = solve_ivp(
        &mut fun,
        &SolveIvpOptions {
            t_span: (0.0, two_pi),
            y0: &[1.0, 0.0],
            method: SolverKind::Rk45,
            rtol: 1e-8,
            atol: ToleranceValue::Scalar(1e-10),
            ..Default::default()
        },
    )
    .expect("solve_ivp harmonic");
    let y_final = result.y.last().unwrap();
    let err_x = (y_final[0] - 1.0).abs();
    let err_v = y_final[1].abs();
    let max_abs = err_x.max(err_v);
    ParityGate {
        fixture_id: "harmonic_oscillator_rk45".into(),
        operation: "solve_ivp_rk45",
        pass: result.success && max_abs < 1e-3,
        max_abs_diff: max_abs,
        tolerance_used: 1e-3,
        size: 2,
    }
}

fn check_lotka_volterra_positive() -> ParityGate {
    let mut fun = lotka_volterra;
    let result = solve_ivp(
        &mut fun,
        &SolveIvpOptions {
            t_span: (0.0, 10.0),
            y0: &[1.0, 0.5],
            method: SolverKind::Rk45,
            rtol: 1e-6,
            atol: ToleranceValue::Scalar(1e-9),
            ..Default::default()
        },
    )
    .expect("solve_ivp lotka-volterra");
    let all_positive = result.y.iter().all(|yi| yi.iter().all(|&v| v > 0.0));
    ParityGate {
        fixture_id: "lotka_volterra_positive".into(),
        operation: "solve_ivp_rk45",
        pass: result.success && all_positive,
        max_abs_diff: 0.0,
        tolerance_used: 0.0,
        size: 2,
    }
}

fn check_rk45_vs_rk23_agreement() -> ParityGate {
    let mut fun45 = exponential_decay;
    let rk45 = solve_ivp(
        &mut fun45,
        &SolveIvpOptions {
            t_span: (0.0, 2.0),
            y0: &[1.0],
            method: SolverKind::Rk45,
            rtol: 1e-8,
            atol: ToleranceValue::Scalar(1e-10),
            ..Default::default()
        },
    )
    .expect("rk45");
    let mut fun23 = exponential_decay;
    let rk23 = solve_ivp(
        &mut fun23,
        &SolveIvpOptions {
            t_span: (0.0, 2.0),
            y0: &[1.0],
            method: SolverKind::Rk23,
            rtol: 1e-8,
            atol: ToleranceValue::Scalar(1e-10),
            ..Default::default()
        },
    )
    .expect("rk23");
    let y45 = rk45.y.last().unwrap()[0];
    let y23 = rk23.y.last().unwrap()[0];
    let max_abs = (y45 - y23).abs();
    ParityGate {
        fixture_id: "rk45_vs_rk23".into(),
        operation: "solver_agreement",
        pass: max_abs < 1e-6,
        max_abs_diff: max_abs,
        tolerance_used: 1e-6,
        size: 1,
    }
}

fn check_mode_invariance() -> ParityGate {
    let mut fun_s = exponential_decay;
    let strict = solve_ivp(
        &mut fun_s,
        &SolveIvpOptions {
            t_span: (0.0, 1.0),
            y0: &[1.0],
            method: SolverKind::Rk45,
            mode: RuntimeMode::Strict,
            ..Default::default()
        },
    )
    .expect("strict");
    let mut fun_h = exponential_decay;
    let hardened = solve_ivp(
        &mut fun_h,
        &SolveIvpOptions {
            t_span: (0.0, 1.0),
            y0: &[1.0],
            method: SolverKind::Rk45,
            mode: RuntimeMode::Hardened,
            ..Default::default()
        },
    )
    .expect("hardened");
    let y_s = strict.y.last().unwrap()[0];
    let y_h = hardened.y.last().unwrap()[0];
    let max_abs = (y_s - y_h).abs();
    ParityGate {
        fixture_id: "strict_vs_hardened".into(),
        operation: "mode_invariance",
        pass: max_abs < 1e-10,
        max_abs_diff: max_abs,
        tolerance_used: 1e-10,
        size: 1,
    }
}

// ── Main test ──────────────────────────────────────────────────────

#[test]
fn evidence_p2c001_final_pack() {
    let evidence_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts/FSCI-P2C-001/evidence");
    std::fs::create_dir_all(&evidence_dir).expect("evidence directory");

    // Fixture manifest
    let fixture_entries = vec![
        FixtureEntry {
            fixture_id: "validate_tol_scalar".into(),
            size: 3,
            input_type: "scalar_tol",
            operations: vec!["validate_tol"],
        },
        FixtureEntry {
            fixture_id: "validate_tol_clamping".into(),
            size: 1,
            input_type: "clamping_edge",
            operations: vec!["validate_tol"],
        },
        FixtureEntry {
            fixture_id: "validate_first_step".into(),
            size: 3,
            input_type: "step_validation",
            operations: vec!["validate_first_step"],
        },
        FixtureEntry {
            fixture_id: "validate_max_step".into(),
            size: 3,
            input_type: "step_validation",
            operations: vec!["validate_max_step"],
        },
        FixtureEntry {
            fixture_id: "exponential_decay_rk45".into(),
            size: 1,
            input_type: "ode_system",
            operations: vec!["solve_ivp_rk45"],
        },
        FixtureEntry {
            fixture_id: "exponential_decay_rk23".into(),
            size: 1,
            input_type: "ode_system",
            operations: vec!["solve_ivp_rk23"],
        },
        FixtureEntry {
            fixture_id: "harmonic_oscillator_rk45".into(),
            size: 2,
            input_type: "ode_system",
            operations: vec!["solve_ivp_rk45"],
        },
        FixtureEntry {
            fixture_id: "lotka_volterra_positive".into(),
            size: 2,
            input_type: "ode_system",
            operations: vec!["solve_ivp_rk45"],
        },
        FixtureEntry {
            fixture_id: "rk45_vs_rk23".into(),
            size: 1,
            input_type: "solver_comparison",
            operations: vec!["solver_agreement"],
        },
        FixtureEntry {
            fixture_id: "strict_vs_hardened".into(),
            size: 1,
            input_type: "mode_comparison",
            operations: vec!["mode_invariance"],
        },
    ];

    let manifest = FixtureManifest {
        packet_id: "FSCI-P2C-001",
        generated_at: now_str(),
        fixtures: fixture_entries,
    };

    // Run parity gates
    let gates = vec![
        check_validate_tol(),
        check_validate_tol_clamping(),
        check_validate_first_step(),
        check_validate_max_step(),
        check_exponential_decay_rk45(),
        check_exponential_decay_rk23(),
        check_harmonic_oscillator(),
        check_lotka_volterra_positive(),
        check_rk45_vs_rk23_agreement(),
        check_mode_invariance(),
    ];

    let all_pass = gates.iter().all(|g| g.pass);
    let parity_gates = ParityGateReport {
        packet_id: "FSCI-P2C-001",
        all_gates_pass: all_pass,
        gates,
    };

    // Operation summaries
    let ops = [
        "validate_tol",
        "validate_first_step",
        "validate_max_step",
        "solve_ivp_rk45",
        "solve_ivp_rk23",
        "solver_agreement",
        "mode_invariance",
    ];
    let operation_summaries: Vec<_> = ops
        .iter()
        .map(|&op| {
            let matched: Vec<_> = parity_gates
                .gates
                .iter()
                .filter(|g| g.operation == op)
                .collect();
            OperationParitySummary {
                operation: op,
                total_fixtures: matched.len(),
                passed: matched.iter().filter(|g| g.pass).count(),
                failed: matched.iter().filter(|g| !g.pass).count(),
                max_abs_diff_across_all: matched.iter().map(|g| g.max_abs_diff).fold(
                    0.0_f64,
                    |a: f64, b: f64| {
                        if a.is_nan() || b.is_nan() {
                            f64::NAN
                        } else {
                            a.max(b)
                        }
                    },
                ),
            }
        })
        .collect();

    let parity_report = ParityReport {
        packet_id: "FSCI-P2C-001",
        operation_summaries,
    };

    let risk_notes = RiskNotesReport {
        packet_id: "FSCI-P2C-001",
        notes: vec![
            RiskNote {
                category: "step_size_selection",
                description: "Initial step size heuristic (Hairer II.4) may overshoot for stiff systems. P2C-001 still gates explicit RK paths here, while solve_ivp now dispatches BDF/Radau through the BDF backend and LSODA can switch from RK to BDF on stiff-like dynamics.".into(),
                affected_operations: vec!["select_initial_step", "solve_ivp"],
                mitigation: "Step size is bounded by max_step and min_step floor. Stiff solver behavior is covered by the BDF/LSODA solve_ivp lanes rather than being reported as unsupported in this evidence packet.".into(),
            },
            RiskNote {
                category: "tolerance_clamping",
                description: "rtol below MIN_RTOL (100*eps ≈ 2.2e-14) is silently clamped with warning. Users may not notice the clamping if they don't check warnings.".into(),
                affected_operations: vec!["validate_tol"],
                mitigation: "ToleranceWarning::RtolClamped is emitted in the warnings vector. Hardened mode applies same clamping with identical warning.".into(),
            },
            RiskNote {
                category: "adaptive_step_rejection",
                description: "Adaptive step-size control may reject many steps for stiff-like problems, causing excessive function evaluations without convergence.".into(),
                affected_operations: vec!["solve_ivp", "RkSolver::step_with"],
                mitigation: "MIN_FACTOR=0.2 prevents catastrophic step shrinkage. StepSizeTooSmall failure returned when h < machine epsilon.".into(),
            },
            RiskNote {
                category: "dense_output_not_implemented",
                description: "Dense output (continuous solution interpolation) is partially implemented. OdeSolution struct exists but interpolation quality may vary.".into(),
                affected_operations: vec!["solve_ivp"],
                mitigation: "Users can set dense_output=false and use t_eval for specific output times.".into(),
            },
        ],
    };
    assert!(
        risk_notes
            .notes
            .iter()
            .all(|note| !note.description.contains("not implemented")
                && !note.description.contains("NotYetImplemented")
                && !note.mitigation.contains("not implemented")
                && !note.mitigation.contains("NotYetImplemented")),
        "P2C-001 risk notes must not claim implemented stiff solvers are still missing"
    );

    let evidence = EvidenceBundle {
        manifest,
        parity_gates,
        risk_notes,
        parity_report,
        sidecar: None,
    };

    let json = serde_json::to_string_pretty(&evidence).unwrap();
    std::fs::write(evidence_dir.join("evidence_bundle.json"), &json).unwrap();

    // Generate and write sidecar for the final file
    let external_sidecar = generate_raptorq_sidecar(json.as_bytes()).ok();
    if let Some(ref sc) = external_sidecar {
        std::fs::write(
            evidence_dir.join("evidence_bundle.raptorq.json"),
            serde_json::to_string_pretty(&sc).unwrap(),
        )
        .unwrap();
    }

    std::fs::write(
        evidence_dir.join("fixture_manifest.json"),
        serde_json::to_string_pretty(&evidence.manifest).unwrap(),
    )
    .unwrap();
    std::fs::write(
        evidence_dir.join("parity_gates.json"),
        serde_json::to_string_pretty(&evidence.parity_gates).unwrap(),
    )
    .unwrap();
    std::fs::write(
        evidence_dir.join("risk_notes.json"),
        serde_json::to_string_pretty(&evidence.risk_notes).unwrap(),
    )
    .unwrap();
    std::fs::write(
        evidence_dir.join("parity_report.json"),
        serde_json::to_string_pretty(&evidence.parity_report).unwrap(),
    )
    .unwrap();

    let bundle_hash = hash(json.as_bytes()).to_hex().to_string();
    std::fs::write(evidence_dir.join("evidence_bundle.blake3"), &bundle_hash).unwrap();

    for g in &evidence.parity_gates.gates {
        if g.pass {
            eprintln!(
                "  PASS: {} {} — max_abs={:.2e}",
                g.operation, g.fixture_id, g.max_abs_diff
            );
        } else {
            eprintln!(
                "  FAIL: {} {} — max_abs={:.2e}",
                g.operation, g.fixture_id, g.max_abs_diff
            );
        }
    }
    assert!(all_pass, "All parity gates must pass");

    eprintln!("\n── P2C-001 Evidence Pack ──");
    for s in &evidence.parity_report.operation_summaries {
        eprintln!(
            "  {}: {}/{} pass, max_abs={:.2e}",
            s.operation, s.passed, s.total_fixtures, s.max_abs_diff_across_all
        );
    }
    eprintln!(
        "  RaptorQ sidecar: {}",
        if evidence.sidecar.is_some() {
            "generated"
        } else {
            "skipped"
        }
    );
}

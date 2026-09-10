//! bd-3jh.14.9: [FSCI-P2C-003-I] Final Evidence Pack
//!
//! Produces a self-contained evidence bundle at:
//!   fixtures/artifacts/FSCI-P2C-003/evidence/
//!
//! Covers BFGS, CG, Powell, brentq, brenth, bisect, ridder with parity gates
//! and risk notes.

use blake3::hash;
use fsci_conformance::{RaptorQSidecar, generate_raptorq_sidecar};
use fsci_opt::{
    ConvergenceStatus, MinimizeOptions, OptimizeMethod, RootMethod, RootOptions, bfgs, bisect,
    brenth, brentq, cg_pr_plus, powell, ridder,
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

fn minimize_opts(method: OptimizeMethod) -> MinimizeOptions {
    MinimizeOptions {
        method: Some(method),
        mode: RuntimeMode::Strict,
        ..Default::default()
    }
}

fn root_opts(method: RootMethod) -> RootOptions {
    RootOptions {
        method: Some(method),
        mode: RuntimeMode::Strict,
        ..Default::default()
    }
}

fn rosenbrock(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() - 1 {
        s += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
    }
    s
}

fn quadratic(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

fn cubic_root(x: f64) -> f64 {
    x * x * x - 2.0 * x - 5.0
}

fn sin_root(x: f64) -> f64 {
    x.sin()
}

// ── Parity checks ──────────────────────────────────────────────────

fn check_bfgs_quadratic(dim: usize) -> ParityGate {
    let x0: Vec<f64> = vec![1.0; dim];
    let result = bfgs(&quadratic, &x0, minimize_opts(OptimizeMethod::Bfgs)).unwrap();
    let max_abs = result
        .x
        .iter()
        .map(|xi| xi.abs())
        .fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
    ParityGate {
        fixture_id: format!("bfgs_quadratic_dim{dim}"),
        operation: "bfgs",
        pass: result.success && max_abs < 1e-3,
        max_abs_diff: max_abs,
        tolerance_used: 1e-3,
        size: dim,
    }
}

fn check_bfgs_rosenbrock() -> ParityGate {
    let x0 = vec![0.0, 0.0];
    let result = bfgs(&rosenbrock, &x0, minimize_opts(OptimizeMethod::Bfgs)).unwrap();
    let err_x = result
        .x
        .iter()
        .map(|xi| (xi - 1.0).abs())
        .fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
    let pass = matches!(result.status, ConvergenceStatus::Success) && err_x < 0.1;
    ParityGate {
        fixture_id: "bfgs_rosenbrock_2d".into(),
        operation: "bfgs",
        pass,
        max_abs_diff: err_x,
        tolerance_used: 0.1,
        size: 2,
    }
}

fn check_cg_quadratic(dim: usize) -> ParityGate {
    let x0: Vec<f64> = vec![1.0; dim];
    let result = cg_pr_plus(
        &quadratic,
        &x0,
        minimize_opts(OptimizeMethod::ConjugateGradient),
    )
    .unwrap();
    let max_abs = result
        .x
        .iter()
        .map(|xi| xi.abs())
        .fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
    ParityGate {
        fixture_id: format!("cg_quadratic_dim{dim}"),
        operation: "cg",
        pass: result.success && max_abs < 1e-3,
        max_abs_diff: max_abs,
        tolerance_used: 1e-3,
        size: dim,
    }
}

fn check_powell_quadratic(dim: usize) -> ParityGate {
    let x0: Vec<f64> = vec![1.0; dim];
    let result = powell(&quadratic, &x0, minimize_opts(OptimizeMethod::Powell)).unwrap();
    let max_abs = result
        .x
        .iter()
        .map(|xi| xi.abs())
        .fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
    ParityGate {
        fixture_id: format!("powell_quadratic_dim{dim}"),
        operation: "powell",
        pass: result.success && max_abs < 1e-2,
        max_abs_diff: max_abs,
        tolerance_used: 1e-2,
        size: dim,
    }
}

fn check_brentq_cubic() -> ParityGate {
    let result = brentq(cubic_root, (1.0, 3.0), root_opts(RootMethod::Brentq)).unwrap();
    let err = cubic_root(result.root).abs();
    ParityGate {
        fixture_id: "brentq_cubic".into(),
        operation: "brentq",
        pass: result.converged && err < 1e-10,
        max_abs_diff: err,
        tolerance_used: 1e-10,
        size: 1,
    }
}

fn check_brenth_cubic() -> ParityGate {
    let result = brenth(cubic_root, (1.0, 3.0), root_opts(RootMethod::Brenth)).unwrap();
    let err = cubic_root(result.root).abs();
    ParityGate {
        fixture_id: "brenth_cubic".into(),
        operation: "brenth",
        pass: result.converged && err < 1e-10,
        max_abs_diff: err,
        tolerance_used: 1e-10,
        size: 1,
    }
}

fn check_bisect_cubic() -> ParityGate {
    let result = bisect(cubic_root, (1.0, 3.0), root_opts(RootMethod::Bisect)).unwrap();
    let err = cubic_root(result.root).abs();
    ParityGate {
        fixture_id: "bisect_cubic".into(),
        operation: "bisect",
        pass: result.converged && err < 1e-10,
        max_abs_diff: err,
        tolerance_used: 1e-10,
        size: 1,
    }
}

fn check_ridder_cubic() -> ParityGate {
    let result = ridder(cubic_root, (1.0, 3.0), root_opts(RootMethod::Ridder)).unwrap();
    let err = cubic_root(result.root).abs();
    ParityGate {
        fixture_id: "ridder_cubic".into(),
        operation: "ridder",
        pass: result.converged && err < 1e-10,
        max_abs_diff: err,
        tolerance_used: 1e-10,
        size: 1,
    }
}

fn check_brentq_sin() -> ParityGate {
    let result = brentq(sin_root, (3.0, 4.0), root_opts(RootMethod::Brentq)).unwrap();
    let err = (result.root - std::f64::consts::PI).abs();
    ParityGate {
        fixture_id: "brentq_sin_pi".into(),
        operation: "brentq",
        pass: result.converged && err < 1e-10,
        max_abs_diff: err,
        tolerance_used: 1e-10,
        size: 1,
    }
}

fn check_root_methods_agree() -> ParityGate {
    let bq = brentq(cubic_root, (1.0, 3.0), root_opts(RootMethod::Brentq)).unwrap();
    let bs = bisect(cubic_root, (1.0, 3.0), root_opts(RootMethod::Bisect)).unwrap();
    let bh = brenth(cubic_root, (1.0, 3.0), root_opts(RootMethod::Brenth)).unwrap();
    let rd = ridder(cubic_root, (1.0, 3.0), root_opts(RootMethod::Ridder)).unwrap();
    let max_diff = [bq.root, bs.root, bh.root, rd.root]
        .windows(2)
        .map(|w| (w[0] - w[1]).abs())
        .fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
    ParityGate {
        fixture_id: "root_methods_agreement".into(),
        operation: "root_agreement",
        pass: max_diff < 1e-10,
        max_abs_diff: max_diff,
        tolerance_used: 1e-10,
        size: 4,
    }
}

fn check_minimize_methods_agree() -> ParityGate {
    let x0 = vec![2.0, 3.0];
    let bfgs_r = bfgs(&quadratic, &x0, minimize_opts(OptimizeMethod::Bfgs)).unwrap();
    let cg_r = cg_pr_plus(
        &quadratic,
        &x0,
        minimize_opts(OptimizeMethod::ConjugateGradient),
    )
    .unwrap();
    let pw_r = powell(&quadratic, &x0, minimize_opts(OptimizeMethod::Powell)).unwrap();
    let max_f_diff = [
        bfgs_r.fun.unwrap_or(0.0),
        cg_r.fun.unwrap_or(0.0),
        pw_r.fun.unwrap_or(0.0),
    ]
    .windows(2)
    .map(|w| (w[0] - w[1]).abs())
    .fold(0.0_f64, |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        }
    });
    let all_near_zero = bfgs_r.fun.unwrap_or(1.0) < 1e-3
        && cg_r.fun.unwrap_or(1.0) < 1e-3
        && pw_r.fun.unwrap_or(1.0) < 1e-3;
    ParityGate {
        fixture_id: "minimize_methods_agreement".into(),
        operation: "minimize_agreement",
        pass: all_near_zero,
        max_abs_diff: max_f_diff,
        tolerance_used: 1e-3,
        size: 3,
    }
}

// ── Main test ──────────────────────────────────────────────────────

#[test]
fn evidence_p2c003_final_pack() {
    let evidence_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts/FSCI-P2C-003/evidence");
    std::fs::create_dir_all(&evidence_dir).expect("evidence directory");

    let fixture_entries = vec![
        FixtureEntry {
            fixture_id: "bfgs_quadratic_dim2".into(),
            size: 2,
            input_type: "minimize",
            operations: vec!["bfgs"],
        },
        FixtureEntry {
            fixture_id: "bfgs_quadratic_dim5".into(),
            size: 5,
            input_type: "minimize",
            operations: vec!["bfgs"],
        },
        FixtureEntry {
            fixture_id: "bfgs_quadratic_dim10".into(),
            size: 10,
            input_type: "minimize",
            operations: vec!["bfgs"],
        },
        FixtureEntry {
            fixture_id: "bfgs_rosenbrock_2d".into(),
            size: 2,
            input_type: "minimize",
            operations: vec!["bfgs"],
        },
        FixtureEntry {
            fixture_id: "cg_quadratic_dim2".into(),
            size: 2,
            input_type: "minimize",
            operations: vec!["cg"],
        },
        FixtureEntry {
            fixture_id: "cg_quadratic_dim5".into(),
            size: 5,
            input_type: "minimize",
            operations: vec!["cg"],
        },
        FixtureEntry {
            fixture_id: "powell_quadratic_dim2".into(),
            size: 2,
            input_type: "minimize",
            operations: vec!["powell"],
        },
        FixtureEntry {
            fixture_id: "powell_quadratic_dim5".into(),
            size: 5,
            input_type: "minimize",
            operations: vec!["powell"],
        },
        FixtureEntry {
            fixture_id: "brentq_cubic".into(),
            size: 1,
            input_type: "root_finding",
            operations: vec!["brentq"],
        },
        FixtureEntry {
            fixture_id: "brenth_cubic".into(),
            size: 1,
            input_type: "root_finding",
            operations: vec!["brenth"],
        },
        FixtureEntry {
            fixture_id: "bisect_cubic".into(),
            size: 1,
            input_type: "root_finding",
            operations: vec!["bisect"],
        },
        FixtureEntry {
            fixture_id: "ridder_cubic".into(),
            size: 1,
            input_type: "root_finding",
            operations: vec!["ridder"],
        },
        FixtureEntry {
            fixture_id: "brentq_sin_pi".into(),
            size: 1,
            input_type: "root_finding",
            operations: vec!["brentq"],
        },
    ];

    let manifest = FixtureManifest {
        packet_id: "FSCI-P2C-003",
        generated_at: now_str(),
        fixtures: fixture_entries,
    };

    // Run parity gates
    let gates = vec![
        check_bfgs_quadratic(2),
        check_bfgs_quadratic(5),
        check_bfgs_quadratic(10),
        check_bfgs_rosenbrock(),
        check_cg_quadratic(2),
        check_cg_quadratic(5),
        check_powell_quadratic(2),
        check_powell_quadratic(5),
        check_brentq_cubic(),
        check_brenth_cubic(),
        check_bisect_cubic(),
        check_ridder_cubic(),
        check_brentq_sin(),
        check_root_methods_agree(),
        check_minimize_methods_agree(),
    ];

    let all_pass = gates.iter().all(|g| g.pass);
    let parity_gates = ParityGateReport {
        packet_id: "FSCI-P2C-003",
        all_gates_pass: all_pass,
        gates,
    };

    let ops = [
        "bfgs",
        "cg",
        "powell",
        "brentq",
        "brenth",
        "bisect",
        "ridder",
        "root_agreement",
        "minimize_agreement",
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
        packet_id: "FSCI-P2C-003",
        operation_summaries,
    };

    let risk_notes = RiskNotesReport {
        packet_id: "FSCI-P2C-003",
        notes: vec![
            RiskNote {
                category: "bfgs_hessian_approximation",
                description: "BFGS uses rank-2 Hessian inverse updates. For highly non-quadratic objectives, the approximation may stall. O(n²) memory for Hessian inverse.".into(),
                affected_operations: vec!["bfgs"],
                mitigation: "Armijo line search provides sufficient decrease guarantee. MaxIterations status returned if convergence too slow.".into(),
            },
            RiskNote {
                category: "finite_difference_gradients",
                description: "All gradient-based methods (BFGS, CG) compute gradients via central finite differences (2n function evaluations per gradient). Noisy or discontinuous objectives may produce inaccurate gradients.".into(),
                affected_operations: vec!["bfgs", "cg"],
                mitigation: "gradient_eps parameter (default 1e-8) can be tuned. Powell method is derivative-free alternative.".into(),
            },
            RiskNote {
                category: "powell_direction_degeneracy",
                description: "Powell's direction set method may fail to converge if search directions become linearly dependent. Golden section search has fixed iteration budget.".into(),
                affected_operations: vec!["powell"],
                mitigation: "Direction set is refreshed when improvement stalls. MaxIterations/MaxEvaluations returned on budget exhaustion.".into(),
            },
            RiskNote {
                category: "root_bracket_requirement",
                description: "All root-finding methods require a valid bracket (sign change). If f(a) and f(b) have the same sign, SignChangeRequired error is returned.".into(),
                affected_operations: vec!["brentq", "brenth", "bisect", "ridder"],
                mitigation: "Explicit error with detail string returned. Users should verify bracket validity before calling.".into(),
            },
        ],
    };

    let evidence = EvidenceBundle {
        manifest,
        parity_gates,
        risk_notes,
        parity_report,
        sidecar: None,
    };

    let json = serde_json::to_string_pretty(&evidence).unwrap();
    std::fs::write(evidence_dir.join("evidence_bundle.json"), &json).unwrap();

    // Generate sidecar from the written content
    let sidecar = generate_raptorq_sidecar(json.as_bytes()).unwrap();
    std::fs::write(
        evidence_dir.join("evidence_bundle.raptorq.json"),
        serde_json::to_string_pretty(&sidecar).unwrap(),
    )
    .unwrap();

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

    eprintln!("\n── P2C-003 Evidence Pack ──");
    for s in &evidence.parity_report.operation_summaries {
        eprintln!(
            "  {}: {}/{} pass, max_abs={:.2e}",
            s.operation, s.passed, s.total_fixtures, s.max_abs_diff_across_all
        );
    }
    eprintln!("  RaptorQ sidecar: generated (external)");
}

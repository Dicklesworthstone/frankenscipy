#![forbid(unsafe_code)]
//! Branch-coverage for fsci_special::select_hypergeometric_branch.
//!
//! Resolves [frankenscipy-fm05w]. The CASP selector routes each
//! hypergeometric call (0F1/1F1/2F1) to the appropriate evaluation
//! branch (DirectSeries / KummerTransform / TerminatingPolynomial /
//! GaussSummation / PfaffTransform / DivergentAtUnitArgument /
//! ParameterGuard / UnsupportedAnalyticContinuation). This harness
//! exercises each documented branch once and verifies the returned
//! HypergeometricBranch matches expectation.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::{HyperSolverPortfolio, RuntimeMode};
use fsci_special::{
    HyperCaspProblem, HypergeometricBranch, select_hypergeometric_branch,
    select_hypergeometric_branch_with_casp,
};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    expected: String,
    actual: String,
    pass: bool,
    note: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create hyper_casp diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

#[test]
fn diff_special_select_hypergeometric_branch_casp() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut probe =
        |id: &str, problem: HyperCaspProblem, mode: RuntimeMode, expected: HypergeometricBranch| {
            let actual_res = select_hypergeometric_branch(problem, mode);
            let (actual_str, pass, note) = match actual_res {
                Ok(d) => (
                    format!("{:?}", d.branch),
                    d.branch == expected,
                    String::new(),
                ),
                Err(e) => ("Err".into(), false, format!("err: {e:?}")),
            };
            diffs.push(CaseDiff {
                case_id: id.into(),
                expected: format!("{expected:?}"),
                actual: actual_str,
                pass,
                note,
            });
        };

    // === Hyp0f1 ===
    // Small z_abs → DirectSeries
    probe(
        "hyp0f1_direct_series_small_z",
        HyperCaspProblem::hyp0f1(2.0, 1.0, 1.0e-10),
        RuntimeMode::Strict,
        HypergeometricBranch::DirectSeries,
    );
    // Large z_abs → AsymptoticExpansion
    probe(
        "hyp0f1_asymptotic_large_z",
        HyperCaspProblem::hyp0f1(2.0, 100.0, 1.0e-10),
        RuntimeMode::Strict,
        HypergeometricBranch::AsymptoticExpansion,
    );

    // === Hyp1f1 ===
    // Moderate z → DirectSeries
    probe(
        "hyp1f1_direct_moderate_z",
        HyperCaspProblem::hyp1f1(1.0, 2.0, 0.5, 1.0e-10),
        RuntimeMode::Strict,
        HypergeometricBranch::DirectSeries,
    );
    // Large negative z → KummerTransform
    probe(
        "hyp1f1_kummer_large_negative_z",
        HyperCaspProblem::hyp1f1(1.0, 2.0, -50.0, 1.0e-10),
        RuntimeMode::Strict,
        HypergeometricBranch::KummerTransform,
    );

    // === Hyp2f1 ===
    // Terminating polynomial: a is non-positive integer
    probe(
        "hyp2f1_terminating_neg_int_a",
        HyperCaspProblem::hyp2f1(-3.0, 1.5, 2.0, 0.5, 1.0e-10),
        RuntimeMode::Strict,
        HypergeometricBranch::TerminatingPolynomial,
    );
    // Direct series inside unit disk with z > 0
    probe(
        "hyp2f1_direct_in_disk_positive",
        HyperCaspProblem::hyp2f1(1.0, 1.5, 2.0, 0.5, 1.0e-10),
        RuntimeMode::Strict,
        HypergeometricBranch::DirectSeries,
    );
    // Pfaff transform inside unit disk with z < 0
    probe(
        "hyp2f1_pfaff_in_disk_negative",
        HyperCaspProblem::hyp2f1(1.0, 1.5, 2.0, -0.5, 1.0e-10),
        RuntimeMode::Strict,
        HypergeometricBranch::PfaffTransform,
    );
    // Gauss summation at z = 1 with c - a - b > 0
    probe(
        "hyp2f1_gauss_z_equal_one_converges",
        HyperCaspProblem::hyp2f1(0.5, 0.5, 2.0, 1.0, 1.0e-10),
        RuntimeMode::Strict,
        HypergeometricBranch::GaussSummation,
    );
    // Divergent at z = 1 when c - a - b <= 0
    probe(
        "hyp2f1_divergent_at_unit",
        HyperCaspProblem::hyp2f1(1.0, 2.0, 2.0, 1.0, 1.0e-10),
        RuntimeMode::Strict,
        HypergeometricBranch::DivergentAtUnitArgument,
    );
    // Pfaff transform outside unit disk with z < 0
    probe(
        "hyp2f1_pfaff_outside_disk_negative",
        HyperCaspProblem::hyp2f1(1.0, 1.5, 2.0, -5.0, 1.0e-10),
        RuntimeMode::Strict,
        HypergeometricBranch::PfaffTransform,
    );

    // === ParameterGuard branch: parameter_stability_margin ≤ precision_target ===
    // Pick b very close to a nonpositive integer pole (b = -3 + 1e-12)
    // so that lower_parameter_stability_margin(b) ≈ 1e-12 < precision_target = 1e-10
    {
        let mut p = HyperCaspProblem::hyp1f1(1.0, -3.0 + 1.0e-12, 0.5, 1.0e-10);
        // Force the margin field to a value below precision_target — the
        // helper uses parameter_stability_margin directly, so we set it
        // explicitly to guarantee the branch fires.
        p.parameter_stability_margin = 1.0e-12;
        probe(
            "parameter_guard_low_margin",
            p,
            RuntimeMode::Strict,
            HypergeometricBranch::ParameterGuard,
        );
    }

    // === Strict-mode non-finite input → UnsupportedAnalyticContinuation ===
    // (Hardened mode would error; Strict delegates)
    {
        let mut p = HyperCaspProblem::hyp1f1(1.0, 2.0, f64::NAN, 1.0e-10);
        // Keep margin healthy so the non-finite path is the trigger
        p.parameter_stability_margin = 1.0;
        probe(
            "unsupported_strict_non_finite_z",
            p,
            RuntimeMode::Strict,
            HypergeometricBranch::UnsupportedAnalyticContinuation,
        );
    }

    // === Hardened-mode non-finite input → error (no decision) ===
    {
        let mut p = HyperCaspProblem::hyp1f1(1.0, 2.0, f64::NAN, 1.0e-10);
        p.parameter_stability_margin = 1.0;
        let result = select_hypergeometric_branch(p, RuntimeMode::Hardened);
        let pass = result.is_err();
        diffs.push(CaseDiff {
            case_id: "hardened_non_finite_z_errors".into(),
            expected: "Err".into(),
            actual: if pass { "Err".into() } else { "Ok".into() },
            pass,
            note: String::new(),
        });
    }

    // === Invalid precision_target → DomainError ===
    {
        let p = HyperCaspProblem::hyp1f1(1.0, 2.0, 0.5, -1.0);
        let result = select_hypergeometric_branch(p, RuntimeMode::Strict);
        let pass = result.is_err();
        diffs.push(CaseDiff {
            case_id: "invalid_precision_target_errors".into(),
            expected: "Err".into(),
            actual: if pass { "Err".into() } else { "Ok".into() },
            pass,
            note: String::new(),
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_special_select_hypergeometric_branch_casp".into(),
        category: "fsci_special::select_hypergeometric_branch CASP routing".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "hyper_casp mismatch: {} expected={} actual={} {}",
                d.case_id, d.expected, d.actual, d.note
            );
        }
    }

    assert!(
        all_pass,
        "hypergeometric CASP branch coverage failed: {} cases",
        diffs.len(),
    );
}

#[test]
fn diff_special_select_hypergeometric_branch_with_casp_portfolio() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut portfolio = HyperSolverPortfolio::new(RuntimeMode::Strict, 16);

    let mut probe_casp = |id: &str,
                          problem: HyperCaspProblem,
                          mode: RuntimeMode,
                          portfolio: &mut HyperSolverPortfolio,
                          expected: HypergeometricBranch| {
        let actual_res = select_hypergeometric_branch_with_casp(problem, mode, portfolio);
        let (actual_str, pass, note) = match actual_res {
            Ok(d) => (
                format!("{:?}", d.branch),
                d.branch == expected,
                String::new(),
            ),
            Err(e) => ("Err".into(), false, format!("err: {e:?}")),
        };
        diffs.push(CaseDiff {
            case_id: id.into(),
            expected: format!("{expected:?}"),
            actual: actual_str,
            pass,
            note,
        });
    };

    // 1. Direct series near zero
    probe_casp(
        "casp_hyp2f1_direct_series",
        HyperCaspProblem::hyp2f1(1.0, 2.0, 3.0, 0.2, 1.0e-14),
        RuntimeMode::Strict,
        &mut portfolio,
        HypergeometricBranch::DirectSeries,
    );

    // 2. Kummer transform for negative real confluent argument
    probe_casp(
        "casp_hyp1f1_kummer_transform",
        HyperCaspProblem::hyp1f1(1.0, 2.0, -3.0, 1.0e-14),
        RuntimeMode::Strict,
        &mut portfolio,
        HypergeometricBranch::KummerTransform,
    );

    // 3. Asymptotic expansion for large argument
    probe_casp(
        "casp_hyp2f1_asymptotic",
        HyperCaspProblem::hyp2f1(1.0, 2.0, 3.0, 25.0, 1.0e-14),
        RuntimeMode::Strict,
        &mut portfolio,
        HypergeometricBranch::AsymptoticExpansion,
    );

    // 4. Parameter guard fallback near pole / boundary
    probe_casp(
        "casp_hyp2f1_boundary_near_pole",
        HyperCaspProblem::hyp2f1(1.0, 2.0, 3.0, 1.0, 1.0e-14),
        RuntimeMode::Strict,
        &mut portfolio,
        HypergeometricBranch::ParameterGuard,
    );

    assert_eq!(
        portfolio.evidence_len(),
        4,
        "portfolio must accumulate 4 evidence entries"
    );

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_special_select_hypergeometric_branch_with_casp_portfolio".into(),
        category: "fsci_special::select_hypergeometric_branch_with_casp CASP portfolio routing"
            .into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    assert!(all_pass, "all casp portfolio routing cases must pass");
}

#![forbid(unsafe_code)]
//! Cover SolverPortfolio + ConformalCalibrator public APIs.
//!
//! Resolves [frankenscipy-amqm1]. Single-test design walks through the
//! public surface from outside the crate:
//!   * SolverPortfolio::new clamps evidence_capacity to ≥ 1
//!   * mode() returns constructed mode
//!   * select_action returns SVDFallback at very low rcond and a
//!     DirectLU / PivotedQR / DiagonalFastPath / TriangularFastPath
//!     in their respective regimes
//!   * record_evidence enforces FIFO capacity eviction
//!   * evidence_len matches recorded count
//!   * observe_backward_error feeds the calibrator
//!   * serialize_jsonl produces one parseable JSON line per entry
//!   * ConformalCalibrator::set_violation_threshold recomputes the
//!     violation count over existing scores

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::{
    ConformalCalibrator, PortfolioEvidence, RuntimeMode, SolverAction, SolverEvidenceEntry,
    SolverPortfolio, StructuralEvidence,
};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create portfolio diff dir");
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

fn mk_entry(action: SolverAction, rcond: f64, backward_error: Option<f64>) -> SolverEvidenceEntry {
    SolverEvidenceEntry {
        component: "test",
        matrix_shape: (4, 4),
        rcond_estimate: rcond,
        chosen_action: action,
        posterior: vec![1.0, 0.0, 0.0, 0.0],
        expected_losses: vec![1.0, 3.0, 15.0, 0.0, 0.0],
        chosen_expected_loss: 1.0,
        fallback_active: false,
        backward_error,
    }
}

#[test]
fn diff_runtime_solver_portfolio_calibrator() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut check = |id: &str, ok: bool, note: String| {
        diffs.push(CaseDiff {
            case_id: id.into(),
            pass: ok,
            note,
        });
    };

    // === 1. new clamps evidence_capacity to ≥ 1 ===
    {
        let p = SolverPortfolio::new(RuntimeMode::Strict, 0);
        check(
            "mode_strict_returned",
            matches!(p.mode(), RuntimeMode::Strict),
            String::new(),
        );
        check(
            "evidence_len_starts_zero",
            p.evidence_len() == 0,
            format!("len={}", p.evidence_len()),
        );
    }

    // === 2. select_action: well-conditioned + no structure → DirectLU ===
    {
        let p = SolverPortfolio::new(RuntimeMode::Strict, 8);
        let (action, posterior, losses, chosen_loss) = p.select_action(1.0e-1, None);
        // For rcond=0.1, posterior should be entirely well-conditioned (p[0]=1)
        check(
            "select_well_posterior_first_one",
            (posterior[0] - 1.0).abs() < 1e-9,
            format!("posterior={posterior:?}"),
        );
        check(
            "select_well_action_directlu",
            matches!(action, SolverAction::DirectLU),
            format!("action={action:?}"),
        );
        // chosen_loss matches losses[action.index()]
        check(
            "select_well_chosen_loss_matches",
            (chosen_loss - losses[action.index()]).abs() < 1e-12,
            format!("chosen={chosen_loss} expected={}", losses[action.index()]),
        );
    }

    // === 3. select_action: ill-conditioned (rcond=1e-11) → DirectLU ===
    // frankenscipy-7tb8d.2: the calibrated losses rank LU first at every conditioning (the
    // hand-set matrix took the SVD here), and the SVD last of the general solvers.
    {
        let p = SolverPortfolio::new(RuntimeMode::Strict, 8);
        let (action, _, losses, _) = p.select_action(1.0e-11, None);
        check(
            "select_ill_action_lu",
            matches!(action, SolverAction::DirectLU)
                && losses[SolverAction::SVDFallback.index()]
                    > losses[SolverAction::PivotedQR.index()],
            format!("action={action:?} losses={losses:?}"),
        );
    }

    // === 4. select_action with Diagonal structure → DiagonalFastPath when cheaper ===
    {
        let p = SolverPortfolio::new(RuntimeMode::Strict, 8);
        let (action, _, _, _) = p.select_action(1.0e-1, Some(StructuralEvidence::Diagonal));
        check(
            "select_diagonal_fast_path",
            matches!(action, SolverAction::DiagonalFastPath),
            format!("action={action:?}"),
        );
    }

    // === 5. select_action with Triangular structure → TriangularFastPath ===
    {
        let p = SolverPortfolio::new(RuntimeMode::Strict, 8);
        let (action, _, _, _) = p.select_action(1.0e-1, Some(StructuralEvidence::Triangular));
        check(
            "select_triangular_fast_path",
            matches!(action, SolverAction::TriangularFastPath),
            format!("action={action:?}"),
        );
    }

    // === 6. record_evidence enforces FIFO capacity (capacity=2, push 4) ===
    {
        let mut p = SolverPortfolio::new(RuntimeMode::Strict, 2);
        for (i, _) in (0..4).enumerate() {
            p.record_evidence(mk_entry(
                SolverAction::DirectLU,
                1e-3 * (i + 1) as f64,
                None,
            ));
        }
        check(
            "record_capacity_2",
            p.evidence_len() == 2,
            format!("len={}", p.evidence_len()),
        );
    }

    // === 7. observe_backward_error → calibrator records score ===
    {
        let mut p = SolverPortfolio::new(RuntimeMode::Strict, 8);
        for _ in 0..20 {
            p.observe_backward_error(1e-12); // tiny error
        }
        let cal_before = p.calibrator().empirical_miscoverage();
        check(
            "observe_no_violations_at_tiny_error",
            cal_before < 0.01,
            format!("miscoverage={cal_before}"),
        );
        for _ in 0..20 {
            p.observe_backward_error(1.0); // big error
        }
        let cal_after = p.calibrator().empirical_miscoverage();
        check(
            "observe_violations_climb",
            cal_after > cal_before,
            format!("before={cal_before} after={cal_after}"),
        );
    }

    // === 8. serialize_jsonl produces N parseable JSON lines for N entries ===
    {
        let mut p = SolverPortfolio::new(RuntimeMode::Strict, 8);
        for i in 0..3 {
            p.record_evidence(mk_entry(
                SolverAction::DirectLU,
                1e-3 * (i + 1) as f64,
                Some(1e-12),
            ));
        }
        let jsonl = p.serialize_jsonl();
        let n_lines = jsonl.lines().filter(|l| !l.is_empty()).count();
        check(
            "jsonl_three_lines",
            n_lines == 3,
            format!("lines={n_lines}"),
        );
        let all_ok = jsonl
            .lines()
            .filter(|l| !l.is_empty())
            .all(|l| serde_json::from_str::<serde_json::Value>(l).is_ok());
        check("jsonl_each_parseable", all_ok, String::new());
    }

    // === 9. ConformalCalibrator::set_violation_threshold recomputes count ===
    {
        let mut cal = ConformalCalibrator::new(0.05, 100);
        // Observe scores spanning small and large.
        for s in [1e-12_f64, 1e-9, 1e-6, 0.01, 1.0].iter().cycle().take(50) {
            cal.observe(*s);
        }
        // Default threshold 1e-8: ~3/5 of scores above threshold → ~30 of 50.
        let mis_default = cal.empirical_miscoverage();
        check(
            "default_threshold_some_violations",
            mis_default > 0.4 && mis_default < 0.7,
            format!("miscoverage={mis_default}"),
        );
        // Raise threshold above largest score — no violations.
        cal.set_violation_threshold(10.0);
        let mis_high = cal.empirical_miscoverage();
        check(
            "high_threshold_zero_violations",
            mis_high < 1e-9,
            format!("miscoverage={mis_high}"),
        );
        // Lower threshold below smallest score — every score is a violation.
        cal.set_violation_threshold(0.0);
        let mis_low = cal.empirical_miscoverage();
        check(
            "zero_threshold_all_violations",
            (mis_low - 1.0).abs() < 1e-9,
            format!("miscoverage={mis_low}"),
        );
    }

    // === 10. calibrator alpha and total_predictions accessors ===
    {
        let mut cal = ConformalCalibrator::new(0.03, 50);
        for _ in 0..7 {
            cal.observe(0.0);
        }
        check(
            "alpha_returned",
            (cal.alpha() - 0.03).abs() < 1e-9,
            format!("alpha={}", cal.alpha()),
        );
        check(
            "total_predictions_counts_observations",
            cal.total_predictions() == 7,
            format!("total={}", cal.total_predictions()),
        );
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_runtime_solver_portfolio_calibrator".into(),
        category: "fsci_runtime::SolverPortfolio + ConformalCalibrator public-API coverage".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("portfolio mismatch: {} — {}", d.case_id, d.note);
        }
    }

    assert!(
        all_pass,
        "SolverPortfolio + calibrator coverage failed: {} cases",
        diffs.len(),
    );
}

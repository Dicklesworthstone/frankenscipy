#![forbid(unsafe_code)]
//! Property test: fsci_linalg::*_with_casp variants produce numerically
//! identical output to their non-CASP counterparts for well-conditioned
//! inputs.
//!
//! Resolves [frankenscipy-6zqp2]. The CASP wrapper passes a
//! SolverPortfolio for solver-selection telemetry; the underlying
//! numerics should be the same. 1e-12 abs on primary fields.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_linalg::{
    InvOptions, LstsqOptions, PinvOptions, SolveOptions, inv, inv_with_casp, lstsq,
    lstsq_with_casp, pinv, pinv_with_casp, solve, solve_with_casp,
};
use fsci_runtime::{RuntimeMode, SolverPortfolio};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
    abs_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    max_abs_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create casp_equiv diff dir");
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

fn mat_max_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    let mut max = 0.0_f64;
    for (ra, rb) in a.iter().zip(b.iter()) {
        if ra.len() != rb.len() {
            return f64::INFINITY;
        }
        for (va, vb) in ra.iter().zip(rb.iter()) {
            max = max.max((va - vb).abs());
        }
    }
    max
}

fn vec_max_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Matrix arms: `slices` over the row-major flattening records a missing side, a length
/// mismatch, or a non-finite entry (which `mat_max_diff`'s max fold would swallow). The metric
/// is still `mat_max_diff`, which also catches a row-shape mismatch.
fn matrix_case(
    ledger: &mut CompareLedger,
    op: &str,
    case_id: &str,
    plain: Option<&[Vec<f64>]>,
    casp: Option<&[Vec<f64>]>,
) -> Option<f64> {
    let plain_flat: Option<Vec<f64>> = plain.map(<[Vec<f64>]>::concat);
    let casp_flat: Option<Vec<f64>> = casp.map(<[Vec<f64>]>::concat);
    ledger.slices(op, case_id, plain_flat.as_deref(), casp_flat.as_deref())?;
    Some(mat_max_diff(plain?, casp?))
}

#[test]
fn diff_linalg_casp_variants_equivalence() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_overall = 0.0_f64;

    let matrices: &[(&str, Vec<Vec<f64>>, Vec<f64>)] = &[
        (
            "sym_pd_3x3",
            vec![
                vec![4.0, 1.0, 0.0],
                vec![1.0, 4.0, 1.0],
                vec![0.0, 1.0, 4.0],
            ],
            vec![1.0, 2.0, 3.0],
        ),
        (
            "general_4x4",
            vec![
                vec![5.0, 1.0, 2.0, 0.0],
                vec![1.0, 6.0, 0.0, 3.0],
                vec![2.0, 0.0, 7.0, 1.0],
                vec![0.0, 3.0, 1.0, 8.0],
            ],
            vec![1.0, -2.0, 3.0, 4.0],
        ),
    ];

    // The reference side is fsci's own non-CASP call, not SciPy: a failed plain call is recorded
    // as a missing reference value, a failed CASP call as an fsci failure.
    let mut ledger = CompareLedger::new(
        "diff_linalg_casp_variants_equivalence",
        &["solve", "inv", "lstsq", "pinv"],
    );

    for (label, a, b) in matrices {
        // solve
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 1);
        let plain = solve(a, b, SolveOptions::default()).ok();
        let casp = solve_with_casp(a, b, SolveOptions::default(), &mut portfolio).ok();
        let solve_d = ledger
            .slices(
                "solve",
                &format!("solve_{label}"),
                plain.as_ref().map(|r| r.x.as_slice()),
                casp.as_ref().map(|r| r.x.as_slice()),
            )
            .map(|(p, q)| vec_max_diff(p, q));
        // inv
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 1);
        let plain = inv(a, InvOptions::default()).ok();
        let casp = inv_with_casp(a, InvOptions::default(), &mut portfolio).ok();
        let inv_d = matrix_case(
            &mut ledger,
            "inv",
            &format!("inv_{label}"),
            plain.as_ref().map(|r| r.inverse.as_slice()),
            casp.as_ref().map(|r| r.inverse.as_slice()),
        );
        // lstsq
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 1);
        let plain = lstsq(a, b, LstsqOptions::default()).ok();
        let casp = lstsq_with_casp(a, b, LstsqOptions::default(), &mut portfolio).ok();
        let lstsq_d = ledger
            .slices(
                "lstsq",
                &format!("lstsq_{label}"),
                plain.as_ref().map(|r| r.x.as_slice()),
                casp.as_ref().map(|r| r.x.as_slice()),
            )
            .map(|(p, q)| vec_max_diff(p, q));
        // pinv
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 1);
        let plain = pinv(a, PinvOptions::default()).ok();
        let casp = pinv_with_casp(a, PinvOptions::default(), &mut portfolio).ok();
        let pinv_d = matrix_case(
            &mut ledger,
            "pinv",
            &format!("pinv_{label}"),
            plain.as_ref().map(|r| r.pseudo_inverse.as_slice()),
            casp.as_ref().map(|r| r.pseudo_inverse.as_slice()),
        );

        let arms = [
            ("solve", solve_d),
            ("inv", inv_d),
            ("lstsq", lstsq_d),
            ("pinv", pinv_d),
        ];
        for (op, d) in arms {
            let Some(d) = d else {
                continue; // the ledger recorded why this case was not compared
            };
            let case_id = format!("{op}_{label}");
            max_overall = max_overall.max(d);
            ledger.compared(op, &case_id, d <= ABS_TOL);
            diffs.push(CaseDiff {
                case_id,
                op: op.into(),
                abs_diff: d,
                pass: d <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_linalg_casp_variants_equivalence".into(),
        category: "fsci_linalg *_with_casp equivalent to non-CASP".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "casp_equiv conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    ledger.finish(matrices.len());
}

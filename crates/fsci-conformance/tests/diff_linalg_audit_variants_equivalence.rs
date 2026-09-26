#![forbid(unsafe_code)]
//! Property test: fsci_linalg audit variants produce numerically
//! identical output to their non-audit counterparts.
//!
//! Resolves [frankenscipy-inryr]. Each *_with_audit just wraps the
//! non-audit call and writes to an audit ledger; primary fields
//! (x, inverse, det, etc.) must be bit-identical. 1e-15 abs on
//! the primary numeric field.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_linalg::{
    InvOptions, LstsqOptions, PinvOptions, SolveOptions, det, det_with_audit, inv, inv_with_audit,
    lstsq, lstsq_with_audit, pinv, pinv_with_audit, solve, solve_with_audit, sync_audit_ledger,
};
use fsci_runtime::{RuntimeMode, SolverPortfolio};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-15;

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
    fs::create_dir_all(output_dir()).expect("create audit_linalg diff dir");
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
    compare: &mut CompareLedger,
    op: &str,
    case_id: &str,
    plain: Option<&[Vec<f64>]>,
    audited: Option<&[Vec<f64>]>,
) -> Option<f64> {
    let plain_flat: Option<Vec<f64>> = plain.map(<[Vec<f64>]>::concat);
    let audited_flat: Option<Vec<f64>> = audited.map(<[Vec<f64>]>::concat);
    compare.slices(op, case_id, plain_flat.as_deref(), audited_flat.as_deref())?;
    Some(mat_max_diff(plain?, audited?))
}

#[test]
fn diff_linalg_audit_variants_equivalence() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_overall = 0.0_f64;
    let ledger = sync_audit_ledger();

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
        (
            "small_2x2",
            vec![vec![2.0, 1.0], vec![1.0, 3.0]],
            vec![1.0, 0.5],
        ),
    ];

    // The reference side is fsci's own non-audit call, not SciPy: a failed plain call is recorded
    // as a missing reference value, a failed audited call as an fsci failure.
    let mut compare = CompareLedger::new(
        "diff_linalg_audit_variants_equivalence",
        &["inv", "det", "lstsq", "pinv", "solve"],
    );

    for (label, a, b) in matrices {
        // inv
        let plain = inv(a, InvOptions::default()).ok();
        let audited = inv_with_audit(a, InvOptions::default(), &ledger).ok();
        let inv_d = matrix_case(
            &mut compare,
            "inv",
            &format!("inv_{label}"),
            plain.as_ref().map(|r| r.inverse.as_slice()),
            audited.as_ref().map(|r| r.inverse.as_slice()),
        );
        // det
        let det_d = compare
            .pair(
                "det",
                &format!("det_{label}"),
                det(a, RuntimeMode::Strict, false).ok(),
                det_with_audit(a, RuntimeMode::Strict, false, &ledger).ok(),
            )
            .map(|(p, q)| (p - q).abs());
        // lstsq
        let plain = lstsq(a, b, LstsqOptions::default()).ok();
        let audited = lstsq_with_audit(a, b, LstsqOptions::default(), &ledger).ok();
        let lstsq_d = compare
            .slices(
                "lstsq",
                &format!("lstsq_{label}"),
                plain.as_ref().map(|r| r.x.as_slice()),
                audited.as_ref().map(|r| r.x.as_slice()),
            )
            .map(|(p, q)| vec_max_diff(p, q));
        // pinv
        let plain = pinv(a, PinvOptions::default()).ok();
        let audited = pinv_with_audit(a, PinvOptions::default(), &ledger).ok();
        let pinv_d = matrix_case(
            &mut compare,
            "pinv",
            &format!("pinv_{label}"),
            plain.as_ref().map(|r| r.pseudo_inverse.as_slice()),
            audited.as_ref().map(|r| r.pseudo_inverse.as_slice()),
        );
        // solve
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 1);
        let plain = solve(a, b, SolveOptions::default()).ok();
        let audited = solve_with_audit(a, b, SolveOptions::default(), &mut portfolio, &ledger).ok();
        let solve_d = compare
            .slices(
                "solve",
                &format!("solve_{label}"),
                plain.as_ref().map(|r| r.x.as_slice()),
                audited.as_ref().map(|r| r.x.as_slice()),
            )
            .map(|(p, q)| vec_max_diff(p, q));

        let arms = [
            ("inv", inv_d),
            ("det", det_d),
            ("lstsq", lstsq_d),
            ("pinv", pinv_d),
            ("solve", solve_d),
        ];
        for (op, d) in arms {
            let Some(d) = d else {
                continue; // the ledger recorded why this case was not compared
            };
            let case_id = format!("{op}_{label}");
            max_overall = max_overall.max(d);
            compare.compared(op, &case_id, d <= ABS_TOL);
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
        test_id: "diff_linalg_audit_variants_equivalence".into(),
        category: "fsci_linalg *_with_audit equivalent to non-audit".into(),
        case_count: diffs.len(),
        compared: compare.counts().clone(),
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
        "linalg audit_equiv conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    compare.finish(matrices.len());
}

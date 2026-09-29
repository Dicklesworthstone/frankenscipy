#![forbid(unsafe_code)]
//! Property test: fsci_sparse::casp_iterative_solve_with_audit
//! gives identical result to casp_iterative_solve.
//!
//! Resolves [frankenscipy-iwsa2]. Audit codepath only logs to the
//! ledger; decision + iterative result must be bit-identical.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_sparse::{
    CaspIterativeSolveOptions, CooMatrix, FormatConvertible, Shape2D, casp_iterative_solve,
    casp_iterative_solve_with_audit, sync_audit_ledger,
};
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
    fs::create_dir_all(output_dir()).expect("create casp_iter_audit diff dir");
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

fn vec_max_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn build_csr(
    rows: usize,
    cols: usize,
    trips: &[(usize, usize, f64)],
) -> Option<fsci_sparse::CsrMatrix> {
    let mut data = Vec::new();
    let mut rs = Vec::new();
    let mut cs = Vec::new();
    for &(r, c, v) in trips {
        data.push(v);
        rs.push(r);
        cs.push(c);
    }
    let coo = CooMatrix::from_triplets(Shape2D::new(rows, cols), data, rs, cs, true).ok()?;
    coo.to_csr().ok()
}

#[test]
fn diff_sparse_casp_iterative_audit_equivalence() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_overall = 0.0_f64;
    let audit_ledger = sync_audit_ledger();

    // SPD tridiagonal (CG-friendly)
    let tri3 = vec![
        (0, 0, 2.0_f64),
        (0, 1, -1.0),
        (1, 0, -1.0),
        (1, 1, 2.0),
        (1, 2, -1.0),
        (2, 1, -1.0),
        (2, 2, 2.0),
    ];
    let b_tri3 = vec![1.0_f64, 2.0, 3.0];

    let tri5: Vec<(usize, usize, f64)> = {
        let mut v = Vec::new();
        for i in 0..5 {
            v.push((i, i, 4.0_f64));
            if i + 1 < 5 {
                v.push((i, i + 1, -1.0));
                v.push((i + 1, i, -1.0));
            }
        }
        v
    };
    let b_tri5 = vec![1.0_f64, 0.5, 1.0, 0.5, 1.0];

    let cases = [
        ("tri3", 3_usize, 3_usize, &tri3, &b_tri3),
        ("tri5", 5, 5, &tri5, &b_tri5),
    ];
    // The reference arm is the plain solve; the audited solve is compared against it.
    let mut ledger = CompareLedger::new(
        "diff_sparse_casp_iterative_audit_equivalence",
        &["casp_iter"],
    );
    for (label, rows, cols, trips, b) in cases {
        let case_id = format!("casp_iter_{label}");
        let csr = build_csr(rows, cols, trips);
        let opts = CaspIterativeSolveOptions::default();
        let plain = csr
            .as_ref()
            .and_then(|m| casp_iterative_solve(m, b, None, opts).ok());
        let audited = csr
            .as_ref()
            .and_then(|m| casp_iterative_solve_with_audit(m, b, None, opts, &audit_ledger).ok());
        let Some((p, a)) = ledger.both("casp_iter", &case_id, plain, audited) else {
            continue;
        };
        let d = vec_max_diff(&p.result.solution, &a.result.solution);
        max_overall = max_overall.max(d);
        // vec_max_diff's max fold drops a NaN (0.0_f64.max(NaN) is 0.0), so reject one here.
        let pass = d <= ABS_TOL
            && p.decision.selected_solver == a.decision.selected_solver
            && p.decision.rationale == a.decision.rationale
            && p.result.converged == a.result.converged
            && !p
                .result
                .solution
                .iter()
                .chain(&a.result.solution)
                .any(|v| v.is_nan());
        ledger.compared("casp_iter", &case_id, pass);
        diffs.push(CaseDiff {
            case_id,
            op: "casp_iter".into(),
            abs_diff: d,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_casp_iterative_audit_equivalence".into(),
        category: "fsci_sparse::casp_iterative_solve_with_audit equivalent to non-audit".into(),
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
        "casp_iter_audit_equiv conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    ledger.finish(cases.len());
}

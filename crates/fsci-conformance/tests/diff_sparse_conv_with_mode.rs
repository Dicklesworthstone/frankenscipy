#![forbid(unsafe_code)]
//! Property test for fsci_sparse with-mode conversion variants:
//! coo_to_csr_with_mode, csr_to_csc_with_mode,
//! csr_to_csc_with_mode_and_audit, csc_to_csr_with_mode.
//!
//! Resolves [frankenscipy-3aij6]. Each *_with_mode call should
//! produce numerically identical matrix data to the plain to_csr /
//! to_csc conversion (mode only changes canonicalization flags and
//! audit emission, not the values). Compare via dense reconstruction
//! at 1e-15 abs.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_runtime::RuntimeMode;
use fsci_sparse::ops::csr_to_csc_with_mode_and_audit;
use fsci_sparse::{
    CooMatrix, CscMatrix, CsrMatrix, FormatConvertible, Shape2D, coo_to_csr_with_mode,
    csc_to_csr_with_mode, csr_to_csc_with_mode, sync_audit_ledger,
};
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
    fs::create_dir_all(output_dir()).expect("create conv_with_mode diff dir");
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

fn csr_to_dense(c: &CsrMatrix) -> Vec<f64> {
    let s = c.shape();
    let mut out = vec![0.0_f64; s.rows * s.cols];
    let indptr = c.indptr();
    let indices = c.indices();
    let data = c.data();
    for r in 0..s.rows {
        let start = indptr[r];
        let end = indptr[r + 1];
        for idx in start..end {
            out[r * s.cols + indices[idx]] += data[idx];
        }
    }
    out
}

fn csc_to_dense(c: &CscMatrix) -> Vec<f64> {
    let s = c.shape();
    let mut out = vec![0.0_f64; s.rows * s.cols];
    let indptr = c.indptr();
    let indices = c.indices();
    let data = c.data();
    for col in 0..s.cols {
        let start = indptr[col];
        let end = indptr[col + 1];
        for idx in start..end {
            out[indices[idx] * s.cols + col] += data[idx];
        }
    }
    out
}

fn vec_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_sparse_conv_with_mode() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_overall = 0.0_f64;
    let audit_ledger = sync_audit_ledger();

    let probes: &[(&str, usize, usize, Vec<(usize, usize, f64)>)] = &[
        (
            "small_3x3",
            3,
            3,
            vec![
                (0, 0, 1.0),
                (0, 2, 2.0),
                (1, 1, 3.0),
                (2, 0, 4.0),
                (2, 2, 5.0),
            ],
        ),
        (
            "tall_5x3",
            5,
            3,
            vec![(0, 0, 1.0), (1, 2, 2.0), (3, 1, 3.0), (4, 0, 4.0)],
        ),
        (
            "wide_3x5",
            3,
            5,
            vec![(0, 4, 1.0), (1, 2, 2.0), (2, 0, 3.0), (2, 3, 4.0)],
        ),
    ];

    // The reference arm is the plain to_csr / to_csc conversion; each *_with_mode result is
    // compared against it.
    let mut ledger = CompareLedger::new(
        "diff_sparse_conv_with_mode",
        &["coo_to_csr", "csr_to_csc", "csr_to_csc_audit", "csc_to_csr"],
    );

    for (label, rows, cols, trips) in probes {
        let mut data = Vec::new();
        let mut rs = Vec::new();
        let mut cs = Vec::new();
        for &(r, c, v) in trips {
            data.push(v);
            rs.push(r);
            cs.push(c);
        }
        let coo = CooMatrix::from_triplets(Shape2D::new(*rows, *cols), data, rs, cs, true)
            .expect("plain coo from triplets");
        let plain_csr = coo.to_csr().expect("plain coo->csr");
        let plain_dense_csr = csr_to_dense(&plain_csr);
        let plain_csc = plain_csr.to_csc().expect("plain csr->csc");
        let plain_dense_csc = csc_to_dense(&plain_csc);
        let plain_csr2 = plain_csc.to_csr().expect("plain csc->csr");
        let plain_dense_csr2 = csr_to_dense(&plain_csr2);

        let checks = [
            // coo_to_csr_with_mode under Strict
            (
                "coo_to_csr",
                format!("coo2csr_strict_{label}"),
                &plain_dense_csr,
                coo_to_csr_with_mode(&coo, RuntimeMode::Strict, "test-c2c-strict")
                    .ok()
                    .map(|(mode_csr, _log)| csr_to_dense(&mode_csr)),
            ),
            // coo_to_csr_with_mode under Hardened
            (
                "coo_to_csr",
                format!("coo2csr_hard_{label}"),
                &plain_dense_csr,
                coo_to_csr_with_mode(&coo, RuntimeMode::Hardened, "test-c2c-hard")
                    .ok()
                    .map(|(mode_csr, _log)| csr_to_dense(&mode_csr)),
            ),
            // csr_to_csc_with_mode
            (
                "csr_to_csc",
                format!("csr2csc_strict_{label}"),
                &plain_dense_csc,
                csr_to_csc_with_mode(&plain_csr, RuntimeMode::Strict, "test-r2c-strict")
                    .ok()
                    .map(|(mode_csc, _log)| csc_to_dense(&mode_csc)),
            ),
            // csr_to_csc_with_mode_and_audit (Strict — never emits, output identical)
            (
                "csr_to_csc_audit",
                format!("csr2csc_audit_strict_{label}"),
                &plain_dense_csc,
                csr_to_csc_with_mode_and_audit(
                    &plain_csr,
                    RuntimeMode::Strict,
                    "test-r2c-audit",
                    &audit_ledger,
                )
                .ok()
                .map(|(mode_csc, _log)| csc_to_dense(&mode_csc)),
            ),
            // csc_to_csr_with_mode
            (
                "csc_to_csr",
                format!("csc2csr_strict_{label}"),
                &plain_dense_csr2,
                csc_to_csr_with_mode(&plain_csc, RuntimeMode::Strict, "test-c2r-strict")
                    .ok()
                    .map(|(mode_csr2, _log)| csr_to_dense(&mode_csr2)),
            ),
        ];
        for (op, case_id, plain_dense, mode_dense) in checks {
            let Some((plain_dense, mode_dense)) = ledger.slices(
                op,
                &case_id,
                Some(plain_dense.as_slice()),
                mode_dense.as_deref(),
            ) else {
                continue;
            };
            let d = vec_diff(mode_dense, plain_dense);
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
        test_id: "diff_sparse_conv_with_mode".into(),
        category: "fsci_sparse with-mode conversion variants equivalent to plain to_csr/to_csc"
            .into(),
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
        "sparse conv_with_mode conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    ledger.finish(probes.len());
}

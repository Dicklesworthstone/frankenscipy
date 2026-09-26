#![forbid(unsafe_code)]
//! Property test for fsci_linalg::qr_multiply.
//!
//! Resolves [frankenscipy-90hoh]. Given (Q, R) = qr(A), the result
//! of qr_multiply(Q, R, C) must equal Q · C (within fp tolerance),
//! since qr_multiply is documented as "apply Q to C".
//!
//! Compare fsci's qr_multiply(Q, R, C) against fsci's matmul(Q, C)
//! at 1e-12 abs.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_linalg::{DecompOptions, matmul, qr, qr_multiply};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create qr_mul diff dir");
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

fn frob_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    let mut max = 0.0_f64;
    for (r_a, r_b) in a.iter().zip(b.iter()) {
        if r_a.len() != r_b.len() {
            return f64::INFINITY;
        }
        for (va, vb) in r_a.iter().zip(r_b.iter()) {
            max = max.max((va - vb).abs());
        }
    }
    max
}

#[test]
fn diff_linalg_qr_multiply_property() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_overall = 0.0_f64;
    let opts = DecompOptions::default();

    let probes: &[(&str, Vec<Vec<f64>>, Vec<Vec<f64>>)] = &[
        (
            "square_3x3_vec",
            vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 10.0],
            ],
            vec![vec![1.0], vec![2.0], vec![3.0]],
        ),
        (
            "square_3x3_mat",
            vec![
                vec![2.0, 1.0, 0.0],
                vec![1.0, 2.0, 1.0],
                vec![0.0, 1.0, 2.0],
            ],
            vec![
                vec![1.0, 0.0, 1.0],
                vec![0.0, 1.0, 0.0],
                vec![1.0, 0.0, -1.0],
            ],
        ),
        (
            "tall_4x3",
            vec![
                vec![1.0, 0.0, 0.0],
                vec![1.0, 1.0, 0.0],
                vec![1.0, 1.0, 1.0],
                vec![1.0, 2.0, 1.0],
            ],
            vec![
                vec![1.0, 2.0],
                vec![3.0, 4.0],
                vec![5.0, 6.0],
                vec![7.0, 8.0],
            ],
        ),
        (
            "rand_5x4",
            (0..5)
                .map(|i| {
                    (0..4)
                        .map(|j| (i * j) as f64 * 0.3 + (i + j) as f64 * 0.1)
                        .collect()
                })
                .collect(),
            (0..5)
                .map(|i| (0..3).map(|j| (i + j) as f64 * 0.2 - 0.5).collect())
                .collect(),
        ),
    ];

    // The reference arm here is fsci's own matmul(Q, C), not SciPy; a failed qr leaves no
    // reference and is recorded as a missing reference value.
    let mut ledger = CompareLedger::new("diff_linalg_qr_multiply_property", &["qr_multiply"]);

    for (label, a, c) in probes {
        let case_id = format!("qr_multiply_{label}");
        if a.len() > a[0].len() {
            // fsci's qr returns the thin Q (m x n) where scipy.linalg.qr(a) returns m x m, so a C
            // with A's row count cannot be multiplied by it.
            ledger.allowlisted(
                "qr_multiply",
                &case_id,
                "frankenscipy-kqeao",
                "qr returns economic factors for a tall A",
            );
            continue;
        }
        let qr_res = qr(a, opts).ok();
        let via_mul = qr_res.as_ref().and_then(|r| matmul(&r.q, c).ok());
        let via_qrm = qr_res
            .as_ref()
            .and_then(|r| qr_multiply(&r.q, &r.r, c, opts).ok());
        let Some((via_mul, via_qrm)) = ledger.both("qr_multiply", &case_id, via_mul, via_qrm)
        else {
            continue;
        };
        let abs_d = frob_diff(&via_qrm, &via_mul);
        max_overall = max_overall.max(abs_d);
        // frob_diff's running max reads a NaN as 0.0; a NaN entry on either side fails.
        let no_nan = !via_qrm
            .iter()
            .chain(via_mul.iter())
            .flatten()
            .any(|v| v.is_nan());
        let pass = no_nan && abs_d <= ABS_TOL;
        ledger.compared("qr_multiply", &case_id, pass);
        diffs.push(CaseDiff {
            case_id,
            abs_diff: abs_d,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_linalg_qr_multiply_property".into(),
        category: "fsci_linalg::qr_multiply(Q, R, C) == matmul(Q, C)".into(),
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
            eprintln!(
                "qr_multiply mismatch: {} abs_diff={}",
                d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "qr_multiply conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    // every square probe compares; the tall ones wait on frankenscipy-kqeao
    ledger.finish(
        probes
            .iter()
            .filter(|(_, a, _)| a.len() <= a[0].len())
            .count(),
    );
}

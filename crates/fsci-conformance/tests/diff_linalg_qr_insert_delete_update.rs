#![forbid(unsafe_code)]
//! Property-based parity harness for fsci_linalg QR rank-update routines:
//! qr_insert, qr_delete, qr_update.
//!
//! Resolves [frankenscipy-cjufv]. QR factorizations are unique only up to
//! sign of columns, so this harness checks the invariant Q*R ≈ modified A
//! rather than element-wise parity. No scipy oracle required.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_linalg::{DecompOptions, qr, qr_delete, qr_insert, qr_update};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;

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
    fs::create_dir_all(output_dir()).expect("create qr_idu diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize qr_idu diff log");
    fs::write(path, json).expect("write qr_idu diff log");
}

fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let p = b.len();
    let n = b[0].len();
    let mut out = vec![vec![0.0_f64; n]; m];
    for i in 0..m {
        for k in 0..p {
            for j in 0..n {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

fn frob_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let m = a.len();
    let n = a.first().map_or(0, Vec::len);
    if b.len() != m || b.first().map_or(0, Vec::len) != n {
        return f64::INFINITY;
    }
    let mut max = 0.0_f64;
    for (ra, rb) in a.iter().zip(b.iter()) {
        for (&va, &vb) in ra.iter().zip(rb.iter()) {
            max = max.max((va - vb).abs());
        }
    }
    max
}

fn fixtures() -> Vec<(String, Vec<Vec<f64>>)> {
    vec![
        (
            "square_3x3".into(),
            vec![
                vec![1.0_f64, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![2.0, 1.0, 7.0],
            ],
        ),
        (
            "tall_4x3".into(),
            vec![
                vec![1.0_f64, 0.5, 0.3],
                vec![2.0, 1.5, 0.7],
                vec![3.0, 2.5, 1.0],
                vec![4.0, 3.5, 1.5],
            ],
        ),
        (
            "square_4x4".into(),
            vec![
                vec![2.0_f64, 1.0, 0.5, 0.3],
                vec![1.0, 3.0, 0.7, 0.4],
                vec![0.5, 0.7, 4.0, 0.6],
                vec![0.3, 0.4, 0.6, 5.0],
            ],
        ),
    ]
}

#[test]
fn diff_linalg_qr_insert_delete_update() {
    let opts = DecompOptions::default();
    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    let fixtures = fixtures();
    let mut ledger = CompareLedger::new(
        "diff_linalg_qr_insert_delete_update",
        &["qr_insert", "qr_delete", "qr_update"],
    );

    for (label, a) in &fixtures {
        let m = a.len();
        let n = a[0].len();
        // A failed base factorization fails all three arms for this fixture.
        let qra = qr(a, opts).ok();

        // qr_insert: insert a row at position k=1
        let new_row: Vec<f64> = (1..=n).map(|i| i as f64 * 0.5).collect();
        // Build expected matrix: original with new row inserted at index 1
        let mut expected_insert = a.clone();
        expected_insert.insert(1, new_row.clone());
        let fsci_insert = qra
            .as_ref()
            .and_then(|qra| qr_insert(&qra.q, &qra.r, &new_row, 1, opts).ok());

        // qr_delete: delete row k=0 (every fixture has m > 1)
        let mut expected_delete = a.clone();
        expected_delete.remove(0);
        let fsci_delete = qra
            .as_ref()
            .and_then(|qra| qr_delete(&qra.q, &qra.r, 0, opts).ok());

        // qr_update: rank-1 update A + u vᵀ
        let u: Vec<f64> = (0..m).map(|i| (i + 1) as f64 * 0.1).collect();
        let v: Vec<f64> = (0..n).map(|j| (j + 1) as f64 * 0.2).collect();
        // expected = A + u * vᵀ
        let mut expected_update = a.clone();
        for i in 0..m {
            for j in 0..n {
                expected_update[i][j] += u[i] * v[j];
            }
        }
        let fsci_update = qra
            .as_ref()
            .and_then(|qra| qr_update(&qra.q, &qra.r, &u, &v, opts).ok());

        let arms = [
            (
                "qr_insert",
                format!("insert_{label}_k1"),
                expected_insert,
                fsci_insert,
            ),
            (
                "qr_delete",
                format!("delete_{label}_k0"),
                expected_delete,
                fsci_delete,
            ),
            (
                "qr_update",
                format!("update_{label}_rank1"),
                expected_update,
                fsci_update,
            ),
        ];
        for (op, case_id, expected, res) in arms {
            let qr_mat = res.map(|res| matmul(&res.q, &res.r));
            let Some((expected, qr_mat)) = ledger.both(op, &case_id, Some(expected), qr_mat) else {
                continue;
            };
            // The flat check catches a NaN the max fold in frob_diff would swallow;
            // frob_diff still rejects a shape mismatch with the same element count.
            let expected_flat: Vec<f64> = expected.iter().flatten().copied().collect();
            let fsci_flat: Vec<f64> = qr_mat.iter().flatten().copied().collect();
            if ledger
                .slices(
                    op,
                    &case_id,
                    Some(expected_flat.as_slice()),
                    Some(fsci_flat.as_slice()),
                )
                .is_none()
            {
                continue;
            }
            let abs_d = frob_diff(&expected, &qr_mat);
            max_overall = max_overall.max(abs_d);
            ledger.compared(op, &case_id, abs_d <= ABS_TOL);
            diffs.push(CaseDiff {
                case_id,
                op: op.into(),
                abs_diff: abs_d,
                pass: abs_d <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_linalg_qr_insert_delete_update".into(),
        category: "fsci_linalg QR rank-update reconstruction".into(),
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
        "qr_insert_delete_update conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    ledger.finish(fixtures.len());
}

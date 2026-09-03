#![forbid(unsafe_code)]
//! Property-based parity harness for fsci_linalg QR rank-update routines:
//! qr_insert, qr_delete, qr_update.
//!
//! Resolves [frankenscipy-cjufv]. QR factorizations are unique only up to
//! sign of columns, so this harness checks the invariant Q*R ≈ modified A
//! rather than element-wise parity. No scipy oracle required.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

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

    for (label, a) in fixtures() {
        let m = a.len();
        let n = a[0].len();
        let Ok(qra) = qr(&a, opts) else { continue };

        // qr_insert: insert a row at position k=1
        let new_row: Vec<f64> = (1..=n).map(|i| i as f64 * 0.5).collect();
        if let Ok(res) = qr_insert(&qra.q, &qra.r, &new_row, 1, opts) {
            // Build expected matrix: original with new row inserted at index 1
            let mut expected = a.clone();
            expected.insert(1, new_row.clone());
            let qr_mat = matmul(&res.q, &res.r);
            let abs_d = frob_diff(&expected, &qr_mat);
            max_overall = max_overall.max(abs_d);
            diffs.push(CaseDiff {
                case_id: format!("insert_{label}_k1"),
                op: "qr_insert".into(),
                abs_diff: abs_d,
                pass: abs_d <= ABS_TOL,
            });
        }

        // qr_delete: delete row k=0
        if m > 1
            && let Ok(res) = qr_delete(&qra.q, &qra.r, 0, opts)
        {
            let mut expected = a.clone();
            expected.remove(0);
            let qr_mat = matmul(&res.q, &res.r);
            let abs_d = frob_diff(&expected, &qr_mat);
            max_overall = max_overall.max(abs_d);
            diffs.push(CaseDiff {
                case_id: format!("delete_{label}_k0"),
                op: "qr_delete".into(),
                abs_diff: abs_d,
                pass: abs_d <= ABS_TOL,
            });
        }

        // qr_update: rank-1 update A + u vᵀ
        let u: Vec<f64> = (0..m).map(|i| (i + 1) as f64 * 0.1).collect();
        let v: Vec<f64> = (0..n).map(|j| (j + 1) as f64 * 0.2).collect();
        if let Ok(res) = qr_update(&qra.q, &qra.r, &u, &v, opts) {
            // expected = A + u * vᵀ
            let mut expected = a.clone();
            for i in 0..m {
                for j in 0..n {
                    expected[i][j] += u[i] * v[j];
                }
            }
            let qr_mat = matmul(&res.q, &res.r);
            let abs_d = frob_diff(&expected, &qr_mat);
            max_overall = max_overall.max(abs_d);
            diffs.push(CaseDiff {
                case_id: format!("update_{label}_rank1"),
                op: "qr_update".into(),
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
}

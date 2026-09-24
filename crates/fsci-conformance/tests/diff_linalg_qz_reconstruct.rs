#![forbid(unsafe_code)]
//! Property-based parity harness for fsci_linalg::qz (generalized Schur
//! decomposition).
//!
//! Resolves [frankenscipy-b2o05]. QZ has sign/ordering ambiguity, so we
//! check invariants: Qᵀ A Z ≈ AA, Qᵀ B Z ≈ BB, Q Qᵀ ≈ I, Z Zᵀ ≈ I.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{DecompOptions, qz};
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
    fs::create_dir_all(output_dir()).expect("create qz diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize qz diff log");
    fs::write(path, json).expect("write qz diff log");
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

fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let n = a[0].len();
    let mut out = vec![vec![0.0_f64; m]; n];
    for i in 0..m {
        for j in 0..n {
            out[j][i] = a[i][j];
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

fn identity_diff(m: &[Vec<f64>]) -> f64 {
    let n = m.len();
    let mut max = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let target = if i == j { 1.0 } else { 0.0 };
            max = max.max((m[i][j] - target).abs());
        }
    }
    max
}

fn fixtures() -> Vec<(String, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    vec![
        (
            "diag_3x3".into(),
            vec![
                vec![2.0_f64, 0.0, 0.0],
                vec![0.0, 3.0, 0.0],
                vec![0.0, 0.0, 5.0],
            ],
            vec![
                vec![1.0_f64, 0.0, 0.0],
                vec![0.0, 2.0, 0.0],
                vec![0.0, 0.0, 4.0],
            ],
        ),
        (
            "sym_3x3_spd".into(),
            vec![
                vec![4.0_f64, 1.0, 0.5],
                vec![1.0, 5.0, 0.3],
                vec![0.5, 0.3, 6.0],
            ],
            vec![
                vec![3.0_f64, 0.2, 0.1],
                vec![0.2, 4.0, 0.05],
                vec![0.1, 0.05, 5.0],
            ],
        ),
        (
            "tri_4x4".into(),
            vec![
                vec![2.0_f64, 1.0, 0.5, 0.2],
                vec![0.0, 3.0, 0.7, 0.3],
                vec![0.0, 0.0, 5.0, 0.4],
                vec![0.0, 0.0, 0.0, 6.0],
            ],
            vec![
                vec![1.5_f64, 0.0, 0.0, 0.0],
                vec![0.0, 2.5, 0.0, 0.0],
                vec![0.0, 0.0, 3.5, 0.0],
                vec![0.0, 0.0, 0.0, 4.5],
            ],
        ),
        (
            // Rank-deficient B (column 3 = column 1): an infinite eigenvalue. qz used to form
            // A·B⁻¹ and reject this pencil (frankenscipy-szq1n.5).
            "general_4x4_singular_b".into(),
            vec![
                vec![1.0_f64, -2.0, 0.5, 3.0],
                vec![0.3, 4.0, -1.0, 0.2],
                vec![2.0, 0.1, 1.5, -0.7],
                vec![-1.2, 0.6, 0.8, 2.5],
            ],
            vec![
                vec![1.0_f64, 0.4, 1.0, 0.0],
                vec![0.5, 2.0, 0.5, 0.3],
                vec![0.0, 0.7, 0.0, 1.0],
                vec![0.2, 0.0, 0.2, 1.5],
            ],
        ),
    ]
}

#[test]
fn diff_linalg_qz_reconstruct() -> Result<(), String> {
    let opts = DecompOptions::default();
    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for (label, a, b) in fixtures() {
        // A failed call is a failed case, never a skipped one.
        let res = qz(&a, &b, opts).map_err(|e| format!("qz {label} failed: {e:?}"))?;

        // Qᵀ A Z ≈ AA
        let qt = transpose(&res.q);
        let qta = matmul(&qt, &a);
        let qtaz = matmul(&qta, &res.z);
        let aa_diff = frob_diff(&res.aa, &qtaz);
        max_overall = max_overall.max(aa_diff);
        diffs.push(CaseDiff {
            case_id: format!("aa_recon_{label}"),
            op: "qz_aa".into(),
            abs_diff: aa_diff,
            pass: aa_diff <= ABS_TOL,
        });

        // Qᵀ B Z ≈ BB
        let qtb = matmul(&qt, &b);
        let qtbz = matmul(&qtb, &res.z);
        let bb_diff = frob_diff(&res.bb, &qtbz);
        max_overall = max_overall.max(bb_diff);
        diffs.push(CaseDiff {
            case_id: format!("bb_recon_{label}"),
            op: "qz_bb".into(),
            abs_diff: bb_diff,
            pass: bb_diff <= ABS_TOL,
        });

        // Q orthogonal
        let qqt = matmul(&res.q, &qt);
        let q_d = identity_diff(&qqt);
        max_overall = max_overall.max(q_d);
        diffs.push(CaseDiff {
            case_id: format!("q_ortho_{label}"),
            op: "qz_q_ortho".into(),
            abs_diff: q_d,
            pass: q_d <= ABS_TOL,
        });

        // Z orthogonal (frankenscipy-uvrcc: qz now returns orthogonal Z).
        let zt = transpose(&res.z);
        let zzt = matmul(&res.z, &zt);
        let z_d = identity_diff(&zzt);
        max_overall = max_overall.max(z_d);
        diffs.push(CaseDiff {
            case_id: format!("z_ortho_{label}"),
            op: "qz_z_ortho".into(),
            abs_diff: z_d,
            pass: z_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_linalg_qz_reconstruct".into(),
        category: "fsci_linalg.qz invariants".into(),
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
        "qz_reconstruct conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    Ok(())
}

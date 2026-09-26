#![forbid(unsafe_code)]
//! Property test for fsci_linalg::ordqz.
//!
//! Resolves [frankenscipy-x1rjd]. ordqz returns (Q, Z, AA, BB) such
//! that Qᵀ A Z = AA, Qᵀ B Z = BB, Q and Z orthogonal, and the
//! generalized eigenvalues are ordered with the selected ones first.
//!
//! ordqz used to reorder by PERMUTING rows and columns of AA and BB, which keeps the
//! relations but not the (quasi-)triangular form once B or the Schur form has
//! off-diagonal entries (frankenscipy-szq1n.5). The dense-B, singular-B and
//! complex-pair probes exercise exactly that, and every case checks the structure
//! and the ordering, not only the relations.
//!
//! Property tests:
//! - max |Qᵀ A Z − AA| < 1e-9, max |Qᵀ B Z − BB| < 1e-9
//! - max |QᵀQ − I| < 1e-9, max |ZᵀZ − I| < 1e-9
//! - BB upper triangular, AA quasi-upper-triangular (exact zeros)
//! - no selected diagonal block follows an unselected one

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{DecompOptions, OrdQzSort, matmul, ordqz};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;

/// A `(label, A, B)` ordqz probe.
type OrdqzProbe = (&'static str, Vec<Vec<f64>>, Vec<Vec<f64>>);

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    sort: String,
    abs_diff: f64,
    schur_form: bool,
    selected_first: bool,
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
    fs::create_dir_all(output_dir()).expect("create ordqz diff dir");
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

fn transpose(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if m.is_empty() {
        return vec![];
    }
    let r = m.len();
    let c = m[0].len();
    let mut t = vec![vec![0.0; r]; c];
    for i in 0..r {
        for j in 0..c {
            t[j][i] = m[i][j];
        }
    }
    t
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
            max = nan_max(max, (va - vb).abs());
        }
    }
    max
}

/// `f64::max` returns the other operand when one is NaN, so folding residuals with it reads a
/// NaN entry as agreement. This keeps the NaN, and `NaN <= tol` then fails the case.
fn nan_max(acc: f64, d: f64) -> f64 {
    if acc.is_nan() || d.is_nan() {
        f64::NAN
    } else {
        acc.max(d)
    }
}

/// BB upper triangular and AA quasi-upper-triangular with non-overlapping 2×2 blocks.
fn is_generalized_schur_form(aa: &[Vec<f64>], bb: &[Vec<f64>]) -> bool {
    let n = aa.len();
    let triangular =
        (0..n).all(|i| (0..i).all(|j| bb[i][j] == 0.0 && (i <= j + 1 || aa[i][j] == 0.0)));
    let blocks_disjoint =
        (1..n.saturating_sub(1)).all(|i| aa[i][i - 1] == 0.0 || aa[i + 1][i] == 0.0);
    triangular && blocks_disjoint
}

/// Does `sort` select the diagonal block at `start`? A 2×2 block carries a complex pair and a
/// diagonal BB block; `lhp` tests its real part, `iuc` its modulus (scipy `_lhp` / `_iuc`).
fn block_selected(
    aa: &[Vec<f64>],
    bb: &[Vec<f64>],
    start: usize,
    size: usize,
    sort: OrdQzSort,
) -> bool {
    let (re, modulus) = if size == 1 {
        if bb[start][start] == 0.0 {
            return false;
        }
        let lambda = aa[start][start] / bb[start][start];
        (lambda, lambda.abs())
    } else {
        let (b1, b2) = (bb[start][start], bb[start + 1][start + 1]);
        let (m11, m12) = (aa[start][start] / b1, aa[start][start + 1] / b1);
        let (m21, m22) = (aa[start + 1][start] / b2, aa[start + 1][start + 1] / b2);
        (0.5 * (m11 + m22), (m11 * m22 - m12 * m21).sqrt())
    };
    match sort {
        OrdQzSort::LeftHalfPlane => re < 0.0,
        OrdQzSort::InsideUnitCircle => modulus < 1.0,
    }
}

fn selected_first(aa: &[Vec<f64>], bb: &[Vec<f64>], sort: OrdQzSort) -> bool {
    let n = aa.len();
    let mut flags = Vec::new();
    let mut j = 0;
    while j < n {
        let size = if j + 1 < n && aa[j + 1][j] != 0.0 {
            2
        } else {
            1
        };
        flags.push(block_selected(aa, bb, j, size, sort));
        j += size;
    }
    flags.windows(2).all(|w| w[0] || !w[1])
}

fn ident(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}

#[test]
fn diff_linalg_ordqz_property() -> Result<(), String> {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_overall = 0.0_f64;
    let opts = DecompOptions::default();

    let probes: &[OrdqzProbe] = &[
        (
            // Eigenvalues 2, 0.25, -3 in that order: both sorts must move a block.
            "triangular_3x3_Bdense",
            vec![
                vec![2.0, 1.0, 1.0],
                vec![0.0, 0.25, 1.0],
                vec![0.0, 0.0, -3.0],
            ],
            vec![
                vec![1.0, 1.0, 0.5],
                vec![0.0, 1.0, 1.0],
                vec![0.0, 0.0, 1.0],
            ],
        ),
        (
            // Rank-deficient B: one infinite eigenvalue, never selected.
            "general_3x3_Bsingular",
            vec![
                vec![1.0, 2.0, 0.5],
                vec![3.0, 4.0, -1.0],
                vec![0.5, -2.0, 0.25],
            ],
            vec![
                vec![1.0, 0.5, 1.0],
                vec![0.0, 2.0, 0.0],
                vec![0.5, 1.0, 0.5],
            ],
        ),
        (
            // Eigenvalues 3 and -1 ± 2i: lhp moves the complex pair as one 2x2 block.
            "complex_pair_3x3_Bid",
            vec![
                vec![3.0, 1.0, 2.0],
                vec![0.0, -1.0, 2.0],
                vec![0.0, -2.0, -1.0],
            ],
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        ),
        (
            "sym_pd_3x3_Bid",
            vec![
                vec![2.0, 1.0, 0.0],
                vec![1.0, 2.0, 1.0],
                vec![0.0, 1.0, 2.0],
            ],
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        ),
        (
            "diag_dom_4x4_Bid",
            vec![
                vec![5.0, 1.0, 0.0, 0.0],
                vec![1.0, 5.0, 1.0, 0.0],
                vec![0.0, 1.0, 5.0, 1.0],
                vec![0.0, 0.0, 1.0, 5.0],
            ],
            vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0, 1.0],
            ],
        ),
        (
            "off_diag_3x3_Bid",
            vec![
                vec![1.0, 2.0, 0.0],
                vec![0.0, -1.0, 1.0],
                vec![1.0, 0.0, 3.0],
            ],
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        ),
        (
            "sym_pd_3x3_Bdiag",
            vec![
                vec![2.0, 1.0, 0.0],
                vec![1.0, 2.0, 1.0],
                vec![0.0, 1.0, 2.0],
            ],
            vec![
                vec![2.0, 0.0, 0.0],
                vec![0.0, 3.0, 0.0],
                vec![0.0, 0.0, 5.0],
            ],
        ),
        (
            "off_diag_3x3_Bdiag",
            vec![
                vec![1.0, 2.0, 0.0],
                vec![0.0, -1.0, 1.0],
                vec![1.0, 0.0, 3.0],
            ],
            vec![
                vec![4.0, 0.0, 0.0],
                vec![0.0, 1.5, 0.0],
                vec![0.0, 0.0, 2.5],
            ],
        ),
        (
            "diag_dom_4x4_Bdiag",
            vec![
                vec![5.0, 1.0, 0.0, 0.0],
                vec![1.0, 5.0, 1.0, 0.0],
                vec![0.0, 1.0, 5.0, 1.0],
                vec![0.0, 0.0, 1.0, 5.0],
            ],
            vec![
                vec![3.0, 0.0, 0.0, 0.0],
                vec![0.0, 2.0, 0.0, 0.0],
                vec![0.0, 0.0, 4.0, 0.0],
                vec![0.0, 0.0, 0.0, 1.5],
            ],
        ),
    ];

    for (label, a, b) in probes {
        for &(sort, sort_label) in &[
            (OrdQzSort::LeftHalfPlane, "lhp"),
            (OrdQzSort::InsideUnitCircle, "iuc"),
        ] {
            // A failed call is a failed case, never a skipped one.
            let r = ordqz(a, b, sort, opts)
                .map_err(|e| format!("ordqz {label} {sort_label} failed: {e:?}"))?;
            let product = |x: &[Vec<f64>], y: &[Vec<f64>]| matmul(x, y).expect("square product");
            let qt = transpose(&r.q);
            let qtaz = product(&product(&qt, a), &r.z);
            let qtbz = product(&product(&qt, b), &r.z);
            let d_aa = frob_diff(&qtaz, &r.aa);
            let d_bb = frob_diff(&qtbz, &r.bb);
            let n = r.q.len();
            let d_q_orth = frob_diff(&product(&qt, &r.q), &ident(n));
            let d_z_orth = frob_diff(&product(&transpose(&r.z), &r.z), &ident(n));
            let abs_d = [d_bb, d_q_orth, d_z_orth].into_iter().fold(d_aa, nan_max);
            max_overall = max_overall.max(abs_d);
            let schur_form = is_generalized_schur_form(&r.aa, &r.bb);
            let sorted = selected_first(&r.aa, &r.bb, sort);
            diffs.push(CaseDiff {
                case_id: format!("ordqz_{label}_{sort_label}"),
                sort: sort_label.into(),
                abs_diff: abs_d,
                schur_form,
                selected_first: sorted,
                pass: abs_d <= ABS_TOL && schur_form && sorted,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_linalg_ordqz_property".into(),
        category: "fsci_linalg::ordqz property test (QT A Z=AA, QT B Z=BB, Q/Z orthogonal, Schur form, selection first)".into(),
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
            eprintln!(
                "ordqz mismatch: {} abs_diff={} schur_form={} selected_first={}",
                d.case_id, d.abs_diff, d.schur_form, d.selected_first
            );
        }
    }

    assert_eq!(
        diffs.len(),
        2 * probes.len(),
        "every probe and sort must be compared"
    );
    assert!(
        all_pass,
        "ordqz conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    Ok(())
}

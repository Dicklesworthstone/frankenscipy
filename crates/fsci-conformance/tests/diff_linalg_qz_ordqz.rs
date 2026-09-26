#![forbid(unsafe_code)]
//! Live SciPy differential test for `fsci_linalg::qz` / `ordqz` (frankenscipy-szq1n.5).
//!
//! qz used to be Schur(A·B⁻¹): it failed on a singular B and lost accuracy as B's condition number
//! grew; ordqz reordered by permuting rows and columns, which breaks the (quasi-)triangular form.
//! This test runs seeded random pencils (n = 3, 8, 30), a singular B, a B with condition number
//! about 1e12, and pencils with complex eigenvalue pairs, and checks:
//! - Q, Z orthogonal: max |QᵀQ − I|, |ZᵀZ − I| ≤ ORTHOGONALITY_TOL·n;
//! - reconstruction: ‖Q·AA·Zᵀ − A‖_F / ‖A‖_F (and for B) ≤ RESIDUAL_TOL·n;
//! - structure: BB upper triangular and AA quasi-upper-triangular (below the quasi-diagonal
//!   ≤ STRUCTURE_TOL·‖·‖_F);
//! - generalized eigenvalues α/β equal SciPy's `eigvals(A, B)` as a multiset, infinities matched,
//!   to EIGENVALUE_REL_TOL relative;
//! - ordqz with sort 'lhp' and 'iuc': the same checks, the selected eigenvalues first, and as
//!   many of them as SciPy's ordqz selects.
//!
//! The compared-row count is asserted.

use std::io::Write;
use std::process::Stdio;

use fsci_conformance::CompareLedger;
use fsci_linalg::{DecompOptions, OrdQzSort, QzResult, ordqz, qz};
use serde::{Deserialize, Serialize};

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const ORTHOGONALITY_TOL: f64 = 1e-13;
const RESIDUAL_TOL: f64 = 1e-13;
const STRUCTURE_TOL: f64 = 1e-14;
const EIGENVALUE_REL_TOL: f64 = 1e-10;
/// For the pencil whose B has condition number ~1e12 the eigenvalues themselves are that
/// ill-conditioned, and a relative bound is not attainable by the incumbent either. Measured on
/// this exact pencil against a 60-digit mpmath reference (scipy 1.17.1): SciPy's ggev is 4.9e-7
/// relative from the true eigenvalues, its qz is 1.9e-8, the two disagree by 5.1e-7, and
/// perturbing B by eps·‖B‖ moves them 1.55e-4; all are within 1e-15 in the chordal metric
/// |λ₁ − λ₂| / (√(1+|λ₁|²)·√(1+|λ₂|²)), which QZ's backward stability controls. That pencil is
/// compared in the chordal metric, every other one relatively.
const EIGENVALUE_CHORDAL_TOL: f64 = 1e-13;

#[derive(Debug, Clone, Serialize)]
struct Pencil {
    case_id: String,
    a: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
    /// Compare eigenvalues in the chordal metric (see `EIGENVALUE_CHORDAL_TOL`).
    chordal: bool,
}

/// One generalized eigenvalue λ = α/β. `beta_rel` is |β| / ‖B‖_F. An eigenvalue is numerically
/// infinite when `beta_rel <= INFINITE_BETA_REL` (neither library returns an exact β = 0 for a
/// B that is singular only to rounding); ordering, as in SciPy's `_lhp` / `_iuc`, excludes only
/// an exact β = 0 (`re` is then infinite).
#[derive(Debug, Clone, Copy, Deserialize)]
struct Eig {
    re: f64,
    im: f64,
    beta_rel: f64,
}

/// |β| / ‖B‖_F at or below which an eigenvalue counts as infinite on both sides.
const INFINITE_BETA_REL: f64 = 1e-14;

impl Eig {
    fn infinite(&self) -> bool {
        self.beta_rel <= INFINITE_BETA_REL
    }
}

#[derive(Debug, Clone, Deserialize)]
struct Answer {
    case_id: String,
    eigenvalues: Vec<Eig>,
    lhp_selected: usize,
    iuc_selected: usize,
    complex_pairs: usize,
}

/// splitmix64 on [-1, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0
    }

    fn matrix(&mut self, n: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|_| (0..n).map(|_| self.next()).collect())
            .collect()
    }
}

fn pencils() -> Vec<Pencil> {
    let mut rng = Rng(20_260_925);
    let mut out = Vec::new();
    for n in [3, 8, 30] {
        out.push(Pencil {
            case_id: format!("random_n{n}"),
            a: rng.matrix(n),
            b: rng.matrix(n),
            chordal: false,
        });
    }
    // Singular B: its last row is the sum of the first two, so one eigenvalue is infinite.
    let a = rng.matrix(5);
    let mut b = rng.matrix(5);
    b[4] = (0..5).map(|j| b[0][j] + b[1][j]).collect();
    out.push(Pencil {
        case_id: "singular_b_n5".into(),
        a,
        b,
        chordal: false,
    });
    // The bead's 2x2 negative case: qz([[1,2],[3,4]], [[1,0],[0,0]]) has one beta = 0.
    out.push(Pencil {
        case_id: "singular_b_2x2".into(),
        a: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        b: vec![vec![1.0, 0.0], vec![0.0, 0.0]],
        chordal: false,
    });
    // B with condition number about 1e12: rows scaled across twelve decades.
    let a = rng.matrix(6);
    let mut b = rng.matrix(6);
    for (i, row) in b.iter_mut().enumerate() {
        let scale = 10f64.powf(-12.0 * i as f64 / 5.0);
        row.iter_mut().for_each(|v| *v *= scale);
    }
    out.push(Pencil {
        case_id: "ill_conditioned_b_n6".into(),
        a,
        b,
        chordal: true,
    });
    out
}

fn transpose(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = m.len();
    (0..n).map(|j| (0..n).map(|i| m[i][j]).collect()).collect()
}

fn mul(x: &[Vec<f64>], y: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = x.len();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| (0..n).map(|k| x[i][k] * y[k][j]).sum())
                .collect()
        })
        .collect()
}

fn frobenius(m: &[Vec<f64>]) -> f64 {
    m.iter().flatten().map(|v| v * v).sum::<f64>().sqrt()
}

fn frobenius_diff(x: &[Vec<f64>], y: &[Vec<f64>]) -> f64 {
    x.iter()
        .flatten()
        .zip(y.iter().flatten())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

fn orthogonality(q: &[Vec<f64>]) -> f64 {
    let qtq = mul(&transpose(q), q);
    let mut worst = 0.0_f64;
    for (i, row) in qtq.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            worst = worst.max((v - if i == j { 1.0 } else { 0.0 }).abs());
        }
    }
    worst
}

/// Largest entry below the quasi-diagonal of AA (a 2x2 block's subdiagonal is allowed) and
/// below the diagonal of BB, relative to each matrix's Frobenius norm.
fn structure_error(aa: &[Vec<f64>], bb: &[Vec<f64>]) -> f64 {
    let n = aa.len();
    let (na, nb) = (frobenius(aa).max(1e-300), frobenius(bb).max(1e-300));
    let mut worst = 0.0_f64;
    for i in 0..n {
        for j in 0..i {
            worst = worst.max(bb[i][j].abs() / nb);
            let block_subdiagonal = j + 1 == i
                && (i + 1 == n || aa[i + 1][i] == 0.0)
                && (j == 0 || aa[j][j - 1] == 0.0);
            if !block_subdiagonal {
                worst = worst.max(aa[i][j].abs() / na);
            }
        }
    }
    worst
}

/// Generalized eigenvalues from the (quasi-)triangular pencil, block by block, in order. A 2x2
/// block has a diagonal BB block.
fn eigenvalues(aa: &[Vec<f64>], bb: &[Vec<f64>], b_norm: f64) -> Vec<Eig> {
    let n = aa.len();
    let mut out = Vec::new();
    let mut j = 0;
    while j < n {
        if j + 1 < n && aa[j + 1][j] != 0.0 {
            let (b1, b2) = (bb[j][j], bb[j + 1][j + 1]);
            let (m11, m12) = (aa[j][j] / b1, aa[j][j + 1] / b1);
            let (m21, m22) = (aa[j + 1][j] / b2, aa[j + 1][j + 1] / b2);
            let re = 0.5 * (m11 + m22);
            let im = (m11 * m22 - m12 * m21 - re * re).max(0.0).sqrt();
            let beta_rel = b1.abs().min(b2.abs()) / b_norm;
            for s in [1.0, -1.0] {
                out.push(Eig {
                    re,
                    im: s * im,
                    beta_rel,
                });
            }
            j += 2;
        } else {
            let re = if bb[j][j] == 0.0 {
                f64::INFINITY
            } else {
                aa[j][j] / bb[j][j]
            };
            out.push(Eig {
                re,
                im: 0.0,
                beta_rel: bb[j][j].abs() / b_norm,
            });
            j += 1;
        }
    }
    out
}

/// Distance between two finite eigenvalues: relative, |Δλ| / max(1, |λ_scipy|).
fn relative(f: &Eig, s: &Eig) -> f64 {
    (f.re - s.re).hypot(f.im - s.im) / s.re.hypot(s.im).max(1.0)
}

/// Match SciPy's multiset against fsci's: every SciPy eigenvalue takes the nearest unused fsci
/// one. Returns the worst relative distance (infinite when an infinity has no partner).
/// Chordal distance between two finite eigenvalues.
fn chordal(f: &Eig, s: &Eig) -> f64 {
    (f.re - s.re).hypot(f.im - s.im)
        / ((1.0 + f.re * f.re + f.im * f.im).sqrt() * (1.0 + s.re * s.re + s.im * s.im).sqrt())
}

fn multiset_distance(fsci: &[Eig], scipy: &[Eig], metric: fn(&Eig, &Eig) -> f64) -> f64 {
    if fsci.len() != scipy.len() {
        return f64::INFINITY;
    }
    let mut used = vec![false; fsci.len()];
    let mut worst = 0.0_f64;
    for s in scipy {
        let mut best: Option<(usize, f64)> = None;
        for (k, f) in fsci.iter().enumerate() {
            if used[k] || f.infinite() != s.infinite() {
                continue;
            }
            let d = if s.infinite() { 0.0 } else { metric(f, s) };
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((k, d));
            }
        }
        match best {
            Some((k, d)) => {
                used[k] = true;
                worst = worst.max(d);
            }
            None => return f64::INFINITY,
        }
    }
    worst
}

fn selected(eig: &Eig, sort: OrdQzSort) -> bool {
    eig.re.is_finite()
        && match sort {
            OrdQzSort::LeftHalfPlane => eig.re < 0.0,
            OrdQzSort::InsideUnitCircle => eig.re.hypot(eig.im) < 1.0,
        }
}

/// Checks one decomposition; returns (failure reasons, fsci eigenvalues in block order).
fn check(pencil: &Pencil, result: &QzResult) -> (Vec<String>, Vec<Eig>) {
    let n = pencil.a.len() as f64;
    let mut reasons = Vec::new();
    // A NaN entry would vanish in the max folds below and pass every `> tol` check.
    let non_finite = [&result.q, &result.z, &result.aa, &result.bb]
        .iter()
        .flat_map(|m| m.iter().flatten())
        .filter(|v| !v.is_finite())
        .count();
    if non_finite > 0 {
        reasons.push(format!("{non_finite} non-finite entries in Q/Z/AA/BB"));
    }
    let orth = orthogonality(&result.q).max(orthogonality(&result.z));
    if orth > ORTHOGONALITY_TOL * n {
        reasons.push(format!("orthogonality {orth:.2e}"));
    }
    let zt = transpose(&result.z);
    let res_a =
        frobenius_diff(&mul(&mul(&result.q, &result.aa), &zt), &pencil.a) / frobenius(&pencil.a);
    let res_b =
        frobenius_diff(&mul(&mul(&result.q, &result.bb), &zt), &pencil.b) / frobenius(&pencil.b);
    if res_a.max(res_b) > RESIDUAL_TOL * n {
        reasons.push(format!("residual A {res_a:.2e} B {res_b:.2e}"));
    }
    let structure = structure_error(&result.aa, &result.bb);
    if structure > STRUCTURE_TOL {
        reasons.push(format!("structure {structure:.2e}"));
    }
    let eigs = eigenvalues(&result.aa, &result.bb, frobenius(&pencil.b));
    // multiset_distance keeps the first NaN distance as a best match and its max fold drops
    // it, so a NaN finite-beta eigenvalue would otherwise match anything.
    let nan_eigs = eigs
        .iter()
        .filter(|e| !e.infinite() && (e.re.is_nan() || e.im.is_nan()))
        .count();
    if nan_eigs > 0 {
        reasons.push(format!("{nan_eigs} NaN eigenvalues"));
    }
    (reasons, eigs)
}

fn scipy_answers(pencils: &[Pencil]) -> Option<Vec<Answer>> {
    let script = r#"
import json, sys
import numpy as np
from scipy.linalg import eigvals, ordqz

out = []
for p in json.load(sys.stdin):
    A = np.asarray(p["a"], dtype=np.float64)
    B = np.asarray(p["b"], dtype=np.float64)
    ab = eigvals(A, B, homogeneous_eigvals=True)
    b_norm = np.linalg.norm(B)
    eigs = []
    for alpha, beta in zip(ab[0], ab[1]):
        re, im = (float("inf"), 0.0) if beta == 0 else (float((alpha / beta).real), float((alpha / beta).imag))
        eigs.append({"re": re if np.isfinite(re) else 1e308, "im": im, "beta_rel": float(abs(beta) / b_norm)})
    selected = {}
    for sort in ("lhp", "iuc"):
        AA, BB, alpha, beta, Q, Z = ordqz(A, B, sort=sort, output="real")
        nz = beta != 0
        lam = np.where(nz, alpha / np.where(nz, beta, 1), np.inf)
        if sort == "lhp":
            sel = nz & (np.real(lam) < 0)
        else:
            sel = nz & (np.abs(lam) < 1)
        selected[sort] = int(np.sum(sel))
    out.append({
        "case_id": p["case_id"], "eigenvalues": eigs,
        "lhp_selected": selected["lhp"], "iuc_selected": selected["iuc"],
        "complex_pairs": int(sum(1 for e in eigs if e["im"] > 0)),
    })
print(json.dumps(out))
"#;
    let query = serde_json::to_string(pencils).expect("serialize pencils");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn the qz oracle: {e}"
            );
            eprintln!("skipping qz oracle: python not available ({e})");
            return None;
        }
    };
    child
        .stdin
        .as_mut()
        .expect("oracle stdin")
        .write_all(query.as_bytes())
        .expect("write oracle query");
    let output = child.wait_with_output().expect("wait for the qz oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "qz oracle failed: {stderr}"
        );
        eprintln!("skipping qz oracle: scipy not available\n{stderr}");
        return None;
    }
    Some(serde_json::from_slice(&output.stdout).expect("parse qz oracle JSON"))
}

#[test]
fn diff_linalg_qz_ordqz() {
    let pencils = pencils();
    let Some(answers) = scipy_answers(&pencils) else {
        return;
    };
    assert_eq!(
        answers.len(),
        pencils.len(),
        "the oracle must answer every pencil"
    );

    let options = DecompOptions::default();
    let mut compared = 0;
    let mut failures = Vec::new();
    let mut ledger = CompareLedger::new("diff_linalg_qz_ordqz", &["qz", "ordqz_lhp", "ordqz_iuc"]);
    for (pencil, answer) in pencils.iter().zip(&answers) {
        assert_eq!(pencil.case_id, answer.case_id);
        compared += 3;
        let (metric, bound): (fn(&Eig, &Eig) -> f64, f64) = if pencil.chordal {
            (chordal, EIGENVALUE_CHORDAL_TOL)
        } else {
            (relative, EIGENVALUE_REL_TOL)
        };
        // A qz failure no longer skips this pencil's ordqz arms.
        let qz_result = match qz(&pencil.a, &pencil.b, options) {
            Ok(result) => Some(result),
            Err(e) => {
                failures.push(format!("{} qz: Err({e:?})", pencil.case_id));
                None
            }
        };
        if let Some((answer, result)) = ledger.both("qz", &pencil.case_id, Some(answer), qz_result)
        {
            let (mut reasons, eigs) = check(pencil, &result);
            let distance = multiset_distance(&eigs, &answer.eigenvalues, metric);
            if distance > bound {
                reasons.push(format!("eigenvalues {distance:.2e}"));
            }
            println!(
                "{} qz: n {} complex pairs {} infinite {} | eigenvalue distance {distance:.2e} | {reasons:?}",
                pencil.case_id,
                pencil.a.len(),
                answer.complex_pairs,
                eigs.iter().filter(|e| e.infinite()).count(),
            );
            ledger.compared("qz", &pencil.case_id, reasons.is_empty());
            if !reasons.is_empty() {
                failures.push(format!("{} qz: {reasons:?}", pencil.case_id));
            }
        }

        for (sort, name, arm, scipy_selected) in [
            (
                OrdQzSort::LeftHalfPlane,
                "lhp",
                "ordqz_lhp",
                answer.lhp_selected,
            ),
            (
                OrdQzSort::InsideUnitCircle,
                "iuc",
                "ordqz_iuc",
                answer.iuc_selected,
            ),
        ] {
            let result = match ordqz(&pencil.a, &pencil.b, sort, options) {
                Ok(result) => Some(result),
                Err(e) => {
                    failures.push(format!("{} ordqz {name}: Err({e:?})", pencil.case_id));
                    None
                }
            };
            let Some((scipy_selected, result)) =
                ledger.both(arm, &pencil.case_id, Some(scipy_selected), result)
            else {
                continue;
            };
            let (mut reasons, eigs) = check(pencil, &result);
            let distance = multiset_distance(&eigs, &answer.eigenvalues, metric);
            if distance > bound {
                reasons.push(format!("eigenvalues {distance:.2e}"));
            }
            let flags: Vec<bool> = eigs.iter().map(|e| selected(e, sort)).collect();
            let leading = flags.iter().take_while(|&&f| f).count();
            let total = flags.iter().filter(|&&f| f).count();
            if leading != total {
                reasons.push(format!("selected not first ({leading} leading of {total})"));
            }
            if total != scipy_selected {
                reasons.push(format!("selected {total}, SciPy {scipy_selected}"));
            }
            println!(
                "{} ordqz {name}: selected {total} (SciPy {scipy_selected}) | eigenvalue distance {distance:.2e} | {reasons:?}",
                pencil.case_id
            );
            ledger.compared(arm, &pencil.case_id, reasons.is_empty());
            if !reasons.is_empty() {
                failures.push(format!("{} ordqz {name}: {reasons:?}", pencil.case_id));
            }
        }
    }
    assert!(
        answers.iter().any(|a| a.complex_pairs > 0),
        "no pencil has a complex eigenvalue pair"
    );
    assert!(
        answers
            .iter()
            .any(|a| a.eigenvalues.iter().any(Eig::infinite)),
        "no pencil has an infinite eigenvalue"
    );
    assert_eq!(compared, 3 * pencils.len());
    assert!(failures.is_empty(), "qz/ordqz disagree: {failures:#?}");
    ledger.finish(pencils.len());
}

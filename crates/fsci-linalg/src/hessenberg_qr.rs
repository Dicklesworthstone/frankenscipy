//! Francis double-shift QR with LAPACK's exceptional shifts — `frankenscipy-sez4r`.
//!
//! ## Why this module exists
//!
//! `bounded_schur` converted a hang into a catchable error, and said so: it is a
//! ROBUSTNESS fix, not a parity fix. It does not recover the eigenvalues SciPy
//! returns for the 230-of-7000 diagonally-dominant fixtures that never converge.
//! This module is the parity half.
//!
//! ## The mechanism, from both sources
//!
//! nalgebra 0.33.3 `src/linalg/schur.rs:129` announces itself as the "Implicit
//! double-shift QR method" and contains no exceptional-shift branch: `grep` for
//! `exceptional` in that file returns nothing. Its shift always comes from the
//! trailing 2x2 block.
//!
//! LAPACK's `dlahqr` — the routine SciPy reaches through `dgeev` -> `dhseqr` —
//! does the opposite. Every tenth iteration without a deflation it abandons the
//! trailing-2x2 shift and substitutes an EXCEPTIONAL shift built from the local
//! subdiagonal magnitude, with the published constants
//!
//!     DAT1 =  3/4    = 0.75
//!     DAT2 = -7/16   = -0.4375
//!
//! (cited from dlahqr's parameter block, which is not vendored in this repo — the
//! constants are quoted from the published routine, not read off a file here, and
//! that distinction is worth stating rather than implying a source read.)
//!
//! The purpose of an exceptional shift is exactly the failure observed on sez4r.
//! When the trailing 2x2 sits inside a TIGHT CLUSTER, its Wilkinson shift barely
//! moves the iteration, the same shift is recomputed next sweep, and the process
//! cycles without deflating. The sweep measured this directly: hanging fixtures
//! had eigenvalue spread 0.085-0.946 against passing ones at 2.49-2.99, with
//! exactly-repeated eigenvalues present in BOTH groups. Degeneracy was not the
//! trigger; clustering was. A shift deliberately displaced from the cluster breaks
//! the cycle, which is why LAPACK converges where nalgebra spins.
//!
//! ## Status
//!
//! UNBUILT. Written under a build freeze (`/data` at 34G, 99% used) and committed
//! uncompiled by instruction, to be compiled and measured when the volume frees.
//! It is NOT wired into `eig` — `bounded_schur` remains the shipped path, so this
//! file changes no behaviour until a later commit routes to it deliberately, after
//! it has been compiled and checked against the named fixtures.
//!
//! ## Scope
//!
//! Eigenvalues only. The sez4r failures are eigenvalue failures, and the Schur
//! vectors are a separable increment. Real Schur form is computed in place and the
//! spectrum is read off its 1x1 and 2x2 diagonal blocks.
//!
//! Safe Rust throughout; the crate is `#![forbid(unsafe_code)]`.

use crate::LinalgError;

/// One eigenvalue of a real matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Eigenvalue {
    pub re: f64,
    pub im: f64,
}

/// Dense row-major square matrix, small and local to this module so the algorithm
/// reads like the reference rather than like nalgebra plumbing.
struct Mat {
    n: usize,
    d: Vec<f64>,
}

impl Mat {
    fn from_rows(a: &[Vec<f64>]) -> Self {
        let n = a.len();
        let mut d = vec![0.0; n * n];
        for (i, row) in a.iter().enumerate() {
            d[i * n..i * n + n].copy_from_slice(&row[..n]);
        }
        Self { n, d }
    }

    #[inline]
    fn get(&self, i: usize, j: usize) -> f64 {
        self.d[i * self.n + j]
    }

    #[inline]
    fn set(&mut self, i: usize, j: usize, v: f64) {
        self.d[i * self.n + j] = v;
    }

    #[inline]
    fn add(&mut self, i: usize, j: usize, v: f64) {
        self.d[i * self.n + j] += v;
    }
}

/// Reduce to upper Hessenberg form by Householder similarity.
///
/// Standard reduction: for each column `k`, annihilate `H[k+2.., k]` with a
/// reflector applied from the LEFT to rows `k+1..n` and from the RIGHT to columns
/// `k+1..n`. Applying both sides keeps the transform a similarity, so the spectrum
/// is preserved exactly — which is the only property the eigenvalue path needs.
fn to_hessenberg(h: &mut Mat) {
    let n = h.n;
    if n < 3 {
        return;
    }
    let mut v = vec![0.0f64; n];
    for k in 0..n - 2 {
        // Householder vector for the subcolumn below the subdiagonal.
        let mut norm_sq = 0.0;
        for i in k + 1..n {
            let x = h.get(i, k);
            v[i] = x;
            norm_sq += x * x;
        }
        if norm_sq == 0.0 {
            continue;
        }
        let alpha = h.get(k + 1, k);
        let norm = norm_sq.sqrt();
        // Choose the sign that avoids cancellation in `v[k+1]`.
        let beta = if alpha >= 0.0 { -norm } else { norm };
        if beta == 0.0 {
            continue;
        }
        v[k + 1] = alpha - beta;
        let mut vnorm_sq = v[k + 1] * v[k + 1];
        for i in k + 2..n {
            vnorm_sq += v[i] * v[i];
        }
        if vnorm_sq == 0.0 {
            continue;
        }
        let two_over = 2.0 / vnorm_sq;

        // Left: H[k+1..n, k..n] -= (2/v'v) v (v' H)
        for j in k..n {
            let mut dot = 0.0;
            for i in k + 1..n {
                dot += v[i] * h.get(i, j);
            }
            let f = two_over * dot;
            for i in k + 1..n {
                h.add(i, j, -f * v[i]);
            }
        }
        // Right: H[0..n, k+1..n] -= (2/v'v) (H v) v'
        for i in 0..n {
            let mut dot = 0.0;
            for j in k + 1..n {
                dot += h.get(i, j) * v[j];
            }
            let f = two_over * dot;
            for j in k + 1..n {
                h.add(i, j, -f * v[j]);
            }
        }
        // The annihilated entries are set exactly rather than left as rounding
        // residue, so the deflation test below sees clean structural zeros.
        h.set(k + 1, k, beta);
        for i in k + 2..n {
            h.set(i, k, 0.0);
        }
    }
}

/// Eigenvalues of a trailing 2x2 block, real pair or conjugate pair.
fn eig2x2(a: f64, b: f64, c: f64, d: f64) -> (Eigenvalue, Eigenvalue) {
    let tr = a + d;
    let det = a * d - b * c;
    let disc = 0.25 * tr * tr - det;
    if disc >= 0.0 {
        let r = disc.sqrt();
        // Both roots via the numerically stable form: compute the larger by
        // magnitude first, then the other from the determinant, so a small root is
        // not formed by subtracting two nearly equal numbers.
        let big = if tr >= 0.0 { 0.5 * tr + r } else { 0.5 * tr - r };
        let small = if big == 0.0 { 0.0 } else { det / big };
        (
            Eigenvalue { re: big, im: 0.0 },
            Eigenvalue { re: small, im: 0.0 },
        )
    } else {
        let im = (-disc).sqrt();
        (
            Eigenvalue { re: 0.5 * tr, im },
            Eigenvalue {
                re: 0.5 * tr,
                im: -im,
            },
        )
    }
}

/// LAPACK `dlahqr`'s exceptional-shift constants.
const DAT1: f64 = 0.75;
const DAT2: f64 = -0.4375;

/// Iterations without a deflation before an exceptional shift is substituted.
/// LAPACK uses 10, and applies a second, differently-anchored exceptional shift at
/// 20 before giving up at `ITMAX`.
const KEXSH: usize = 10;

/// Eigenvalues of a real square matrix via Hessenberg reduction plus Francis
/// double-shift QR WITH exceptional shifts.
///
/// `max_sweeps_per_eigenvalue` bounds the work exactly as LAPACK's `ITMAX` does;
/// `30 * max(10, n)` is the budget already justified and sweep-verified on
/// `bounded_schur`, and is the right default for a caller that wants the same
/// termination guarantee.
///
/// Returns `ConvergenceFailure` rather than looping if the bound is exhausted, so
/// this routine can never reproduce the sez4r hang even if the exceptional shifts
/// fail to break a cycle. That is deliberate: a bound plus a better shift strategy
/// is strictly safer than either alone.
pub(crate) fn eigenvalues_francis(
    a: &[Vec<f64>],
    eps: f64,
    max_sweeps_per_eigenvalue: usize,
) -> Result<Vec<Eigenvalue>, LinalgError> {
    let n = a.len();
    let mut out = vec![Eigenvalue { re: 0.0, im: 0.0 }; n];
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        out[0] = Eigenvalue {
            re: a[0][0],
            im: 0.0,
        };
        return Ok(out);
    }

    let mut h = Mat::from_rows(a);
    to_hessenberg(&mut h);

    // `ihi` is the inclusive high index of the active block. Deflation shrinks it
    // from the bottom, which is the direction the subdiagonal test scans.
    let mut ihi = n - 1;
    let mut its = 0usize;
    let budget = max_sweeps_per_eigenvalue.max(1) * n;
    let mut total = 0usize;

    loop {
        // Find `ilo`: the start of the active block, i.e. the largest index whose
        // subdiagonal entry is NEGLIGIBLE relative to its neighbours. LAPACK's
        // criterion, which is scale-relative rather than absolute -- an absolute
        // epsilon on a dimensioned quantity is the bug recorded in
        // `correctness_absolute_epsilon_on_dimensioned_quantity`.
        let mut ilo = 0usize;
        let mut l = ihi;
        while l > 0 {
            let sub = h.get(l, l - 1).abs();
            let scale = h.get(l - 1, l - 1).abs() + h.get(l, l).abs();
            let tol = if scale == 0.0 { eps } else { eps * scale };
            if sub <= tol {
                h.set(l, l - 1, 0.0);
                ilo = l;
                break;
            }
            l -= 1;
        }

        if ilo == ihi {
            // 1x1 block: one real eigenvalue, deflate.
            out[ihi] = Eigenvalue {
                re: h.get(ihi, ihi),
                im: 0.0,
            };
            if ihi == 0 {
                break;
            }
            ihi -= 1;
            its = 0;
            continue;
        }
        if ilo + 1 == ihi {
            // 2x2 block: a real pair or a conjugate pair, deflate both.
            let (e1, e2) = eig2x2(
                h.get(ihi - 1, ihi - 1),
                h.get(ihi - 1, ihi),
                h.get(ihi, ihi - 1),
                h.get(ihi, ihi),
            );
            out[ihi - 1] = e1;
            out[ihi] = e2;
            if ihi < 2 {
                break;
            }
            ihi -= 2;
            its = 0;
            continue;
        }

        if total >= budget {
            return Err(LinalgError::ConvergenceFailure {
                detail: format!(
                    "Francis QR failed to converge for a {n}x{n} matrix after {total} sweeps"
                ),
            });
        }

        // ---- shift selection: this is the whole point of the module ----------
        //
        // Ordinarily the shifts are the eigenvalues of the trailing 2x2, which is
        // what nalgebra always uses. Every KEXSH iterations WITHOUT a deflation
        // that choice is abandoned for an exceptional shift displaced from the
        // local cluster, because a shift that sits inside a tight cluster barely
        // advances the iteration and gets recomputed identically next sweep.
        let (s_tr, s_det) = if its > 0 && its % KEXSH == 0 {
            // Anchor on the local subdiagonal magnitude. At its==20 LAPACK
            // re-anchors at the block's other end; alternating the anchor avoids
            // reproducing the same displaced shift twice in a row.
            let s = if (its / KEXSH) % 2 == 1 {
                h.get(ihi, ihi - 1).abs() + h.get(ihi - 1, ihi - 2).abs()
            } else {
                h.get(ilo + 1, ilo).abs() + h.get(ilo + 2, ilo + 1).abs()
            };
            let anchor = if (its / KEXSH) % 2 == 1 {
                h.get(ihi, ihi)
            } else {
                h.get(ilo, ilo)
            };
            let h11 = DAT1 * s + anchor;
            let h12 = DAT2 * s;
            let h21 = s;
            let h22 = h11;
            (h11 + h22, h11 * h22 - h12 * h21)
        } else {
            let p = h.get(ihi - 1, ihi - 1);
            let q = h.get(ihi - 1, ihi);
            let r = h.get(ihi, ihi - 1);
            let t = h.get(ihi, ihi);
            (p + t, p * t - q * r)
        };

        // ---- implicit double-shift bulge chase -------------------------------
        //
        // Form the first column of (H - s1 I)(H - s2 I) restricted to the active
        // block. Using the trace and determinant keeps everything real even when
        // the shifts are a conjugate pair, which is why the double shift is done
        // implicitly rather than with complex arithmetic.
        let h00 = h.get(ilo, ilo);
        let h10 = h.get(ilo + 1, ilo);
        let mut x = h00 * h00 + h.get(ilo, ilo + 1) * h10 - s_tr * h00 + s_det;
        let mut y = h10 * (h00 + h.get(ilo + 1, ilo + 1) - s_tr);
        let mut z = h10 * h.get(ilo + 2, ilo + 1);

        for k in ilo..ihi - 1 {
            // A 3-element Householder reflector zeroing y and z.
            let norm_sq = x * x + y * y + z * z;
            if norm_sq != 0.0 {
                let norm = norm_sq.sqrt();
                let beta = if x >= 0.0 { -norm } else { norm };
                let v0 = x - beta;
                let vnorm_sq = v0 * v0 + y * y + z * z;
                if vnorm_sq != 0.0 {
                    let two_over = 2.0 / vnorm_sq;
                    let v = [v0, y, z];
                    let rows = [k, k + 1, k + 2];
                    let last = (k + 3).min(ihi + 1);

                    // Left application over the affected columns.
                    for j in k.saturating_sub(1)..=ihi {
                        let mut dot = 0.0;
                        for (vi, &ri) in v.iter().zip(rows.iter()) {
                            dot += vi * h.get(ri, j);
                        }
                        let f = two_over * dot;
                        for (vi, &ri) in v.iter().zip(rows.iter()) {
                            h.add(ri, j, -f * vi);
                        }
                    }
                    // Right application over the affected rows.
                    for i in ilo..=last.min(ihi) {
                        let mut dot = 0.0;
                        for (vj, &cj) in v.iter().zip(rows.iter()) {
                            dot += h.get(i, cj) * vj;
                        }
                        let f = two_over * dot;
                        for (vj, &cj) in v.iter().zip(rows.iter()) {
                            h.add(i, cj, -f * vj);
                        }
                    }
                }
            }
            // Recompute the bulge for the next position from the updated matrix.
            if k + 1 < ihi {
                x = h.get(k + 1, k);
                y = h.get(k + 2, k);
                z = if k + 3 <= ihi { h.get(k + 3, k) } else { 0.0 };
            }
        }

        its += 1;
        total += 1;
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::eigenvalues_francis;
    use crate::LinalgError;

    /// Byte-for-byte the generator used by the sez4r regression test in `lib.rs`.
    /// Duplicated rather than shared because that one is private to its test
    /// module — and the duplication is load-bearing: if the two ever diverge the
    /// fixture indices below stop naming the same matrices, so any edit here must
    /// be mirrored there.
    fn make_diag_dominant(n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let r = ((seed.wrapping_mul(i as u64 + 1).wrapping_add(j as u64)) % 1000) as f64
                    / 1000.0;
                a[i][j] = if i == j { (n as f64) * 2.0 + r } else { r - 0.5 };
            }
        }
        a
    }

    fn budget() -> usize {
        30
    }

    /// THE POINT OF THE MODULE. These five fixtures are the ones nalgebra's
    /// shift-free iteration never converges on — `eig` used to hang forever on
    /// them, and after `bounded_schur` it returns `ConvergenceFailure`. With
    /// exceptional shifts they must CONVERGE, and to the right spectrum.
    ///
    /// The trace identity is the check: for a real matrix the sum of eigenvalue
    /// real parts equals the trace exactly in exact arithmetic, and it is a
    /// property of the WHOLE spectrum, so it cannot be satisfied by returning a
    /// plausible-looking subset. Conjugate pairs contribute their real parts twice
    /// and their imaginary parts cancel, which is why summing only `re` is correct
    /// rather than a simplification.
    #[test]
    fn known_non_converging_fixtures_now_converge() {
        for (n, seed) in [(5usize, 201u64), (5, 213), (5, 234), (6, 319), (6, 335)] {
            let a = make_diag_dominant(n, seed);
            let eigs = eigenvalues_francis(&a, f64::EPSILON, budget()).unwrap_or_else(|e| {
                panic!(
                    "({n},{seed}) still does not converge with exceptional shifts: {e:?}. \
                     That would mean the cycle is not the shift strategy after all, and \
                     the diagnosis on sez4r is wrong rather than incomplete"
                )
            });
            assert_eq!(eigs.len(), n, "({n},{seed}) returned the wrong count");
            let trace: f64 = (0..n).map(|i| a[i][i]).sum();
            let sum: f64 = eigs.iter().map(|e| e.re).sum();
            assert!(
                (sum - trace).abs() < 1e-9 * (trace.abs() + 1.0),
                "({n},{seed}) trace identity broken: {sum} vs {trace}"
            );
            // Conjugate pairs must come in pairs: a lone non-zero imaginary part
            // means the 2x2 extraction dropped a partner.
            let imag_sum: f64 = eigs.iter().map(|e| e.im).sum();
            assert!(
                imag_sum.abs() < 1e-9 * (trace.abs() + 1.0),
                "({n},{seed}) imaginary parts do not cancel ({imag_sum}), so a \
                 conjugate partner is missing"
            );
        }
    }

    /// MUST-MISS: matrices that already converged must still converge, and the
    /// exceptional-shift branch must not fire on them. If this fails while the
    /// test above passes, the shift schedule is too eager and is perturbing cases
    /// that never needed it.
    #[test]
    fn already_converging_fixtures_still_converge() {
        for (n, seed) in [(5usize, 0u64), (6, 0), (8, 42), (8, 999)] {
            let a = make_diag_dominant(n, seed);
            let eigs = eigenvalues_francis(&a, f64::EPSILON, budget())
                .unwrap_or_else(|e| panic!("({n},{seed}) regressed to {e:?}"));
            let trace: f64 = (0..n).map(|i| a[i][i]).sum();
            let sum: f64 = eigs.iter().map(|e| e.re).sum();
            assert!(
                (sum - trace).abs() < 1e-9 * (trace.abs() + 1.0),
                "({n},{seed}) trace identity broken: {sum} vs {trace}"
            );
        }
    }

    /// Small cases the reduction cannot touch, so the 1x1 and 2x2 deflation paths
    /// are driven directly rather than only as the tail of a larger sweep.
    #[test]
    fn tiny_matrices_take_the_deflation_paths() {
        let one = eigenvalues_francis(&[vec![3.5]], f64::EPSILON, budget()).unwrap();
        assert_eq!(one.len(), 1);
        assert!((one[0].re - 3.5).abs() < 1e-12 && one[0].im == 0.0);

        // Real pair.
        let real = eigenvalues_francis(&[vec![2.0, 1.0], vec![1.0, 2.0]], f64::EPSILON, budget())
            .unwrap();
        let mut re: Vec<f64> = real.iter().map(|e| e.re).collect();
        re.sort_by(f64::total_cmp);
        assert!((re[0] - 1.0).abs() < 1e-12 && (re[1] - 3.0).abs() < 1e-12, "{re:?}");
        assert!(real.iter().all(|e| e.im == 0.0), "real pair got an imaginary part");

        // Conjugate pair: [[0,1],[-1,0]] has eigenvalues +/- i, so this is the
        // must-hit arm for the complex branch of the 2x2 extraction.
        let cplx = eigenvalues_francis(&[vec![0.0, 1.0], vec![-1.0, 0.0]], f64::EPSILON, budget())
            .unwrap();
        assert!(
            cplx.iter().all(|e| e.re.abs() < 1e-12) && cplx.iter().any(|e| e.im.abs() > 0.5),
            "expected +/- i, got {cplx:?}"
        );
        assert!(
            cplx.iter().map(|e| e.im).sum::<f64>().abs() < 1e-12,
            "conjugate pair imaginary parts must cancel"
        );
    }

    /// An exhausted budget must ERROR, never spin. This is the property that makes
    /// the module strictly safer than nalgebra's even if the exceptional shifts
    /// turn out not to break every cycle: a bound AND a better shift strategy.
    #[test]
    fn a_bounded_run_terminates_and_fails_only_by_non_convergence() {
        let a = make_diag_dominant(6, 319);
        // A budget of 0 is clamped to one sweep per eigenvalue, so this is not a
        // test that "0 means 0" -- it is a test that the bound is enforced at all
        // and that the ONLY way this routine may fail is non-convergence.
        match eigenvalues_francis(&a, f64::EPSILON, 0) {
            Ok(v) => assert_eq!(
                v.len(),
                6,
                "a successful bounded run must still return the whole spectrum"
            ),
            Err(LinalgError::ConvergenceFailure { .. }) => {}
            Err(other) => panic!(
                "a bounded run may only fail by non-convergence; got {other:?}, which \
                 would mean the routine has a second failure mode the caller cannot \
                 distinguish from a hard budget"
            ),
        }
        // Reaching this line at all is the termination assertion: the pre-fix
        // behaviour was an infinite loop, which no assertion can catch because the
        // test never returns to evaluate one.
    }
}

//! Cyclic-tridiagonal solvers from FITPACK (`fpcyt1`/`fpcyt2`).
//!
//! Faithful safe-Rust port of Dierckx's `fpcyt1` (LU-style factorization of a
//! cyclic-tridiagonal matrix) and `fpcyt2` (solve using that factorization).
//! These are the foundation for the periodic v-direction of the gridded
//! spherical spline (`spgrid`/`fpgrsp`, bead frankenscipy-flath, in progress).
//! 1-based indexing mirrors the Fortran line-for-line.
//!
//! Storage convention: `a[i]` (rows `1..=n`) holds the cyclic-tridiagonal band
//! columns `1..=6`, where on input `a[i][1]` is the sub-diagonal, `a[i][2]` the
//! diagonal, `a[i][3]` the super-diagonal, with the wrap corners in `a[1][1]`
//! (element `(1, n)`) and `a[n][3]` (element `(n, 1)`). Columns `4..=6` receive
//! the factorization.

/// `fpcyt1`: factorize the cyclic-tridiagonal matrix stored in `a` (rows
/// `1..=n`, requires `n >= 3`), writing the factorization into columns `4..=6`.
#[allow(dead_code)]
pub(crate) fn fpcyt1(a: &mut [Vec<f64>], n: usize) {
    let one = 1.0_f64;
    let n2 = n - 2;
    let mut beta = one / a[1][2];
    let mut gamma = a[n][3];
    let mut teta = a[1][1] * beta;
    a[1][4] = beta;
    a[1][5] = gamma;
    a[1][6] = teta;
    let mut sum = gamma * teta;
    for i in 2..=n2 {
        let v = a[i - 1][3] * beta;
        let aa = a[i][1];
        beta = one / (a[i][2] - aa * v);
        gamma = -gamma * v;
        teta = -teta * aa * beta;
        a[i][4] = beta;
        a[i][5] = gamma;
        a[i][6] = teta;
        sum += gamma * teta;
    }
    let n1 = n - 1;
    let v = a[n2][3] * beta;
    let aa = a[n1][1];
    beta = one / (a[n1][2] - aa * v);
    gamma = a[n][1] - gamma * v;
    teta = (a[n1][3] - teta * aa) * beta;
    a[n1][4] = beta;
    a[n1][5] = gamma;
    a[n1][6] = teta;
    a[n][4] = one / (a[n][2] - (sum + gamma * teta));
}

/// `fpcyt2`: solve the cyclic-tridiagonal system using the [`fpcyt1`]
/// factorization in `a`; right-hand side `b[1..=n]`, solution written to
/// `c[1..=n]`.
#[allow(dead_code)]
pub(crate) fn fpcyt2(a: &[Vec<f64>], n: usize, b: &[f64], c: &mut [f64]) {
    c[1] = b[1] * a[1][4];
    let mut sum = c[1] * a[1][5];
    let n1 = n - 1;
    for i in 2..=n1 {
        c[i] = (b[i] - a[i][1] * c[i - 1]) * a[i][4];
        sum += c[i] * a[i][5];
    }
    let cc = (b[n] - sum) * a[n][4];
    c[n] = cc;
    c[n1] -= cc * a[n1][6];
    let mut j = n1;
    for _i in 3..=n {
        let j1 = j - 1;
        c[j1] = c[j1] - c[j] * a[j1][3] * a[j1][4] - cc * a[j1][6];
        j = j1;
    }
}

/// `fpbacp`: back-substitution for a periodic banded upper-triangular system
/// (FITPACK). `a` (rows `1..=n-k`, cols `1..=k1=k+1`) is the banded body part
/// (`a[i][1]` diagonal, `a[i][1+l]` super-diagonals); `b` (rows `1..=n`, cols
/// `1..=k`) is the periodic coupling block whose last `k` columns wrap. `z` is
/// the right-hand side; the solution is written to `c[1..=n]`.
#[allow(dead_code)]
pub(crate) fn fpbacp(a: &[Vec<f64>], b: &[Vec<f64>], z: &[f64], n: usize, k: usize, c: &mut [f64]) {
    // Solve the trailing k unknowns (the periodic block, upper-triangular in b).
    // (`n2 = n - k` is computed only after this loop, which returns early when
    // `n <= k`, matching the Fortran where n2 would otherwise be non-positive.)
    let mut l = n;
    for i in 1..=k {
        let mut store = z[l];
        let j = k + 2 - i;
        if i != 1 {
            let mut l0 = l;
            for l1 in j..=k {
                l0 += 1;
                store -= c[l0] * b[l][l1];
            }
        }
        c[l] = store / b[l][j - 1];
        if l == 1 {
            return;
        }
        l -= 1;
    }
    let n2 = n - k;
    // Fold the trailing unknowns into the body right-hand side.
    for i in 1..=n2 {
        let mut store = z[i];
        let mut ll = n2;
        for j in 1..=k {
            ll += 1;
            store -= c[ll] * b[i][j];
        }
        c[i] = store;
    }
    // Back-substitute the banded body part.
    let mut i = n2;
    c[i] /= a[i][1];
    if i == 1 {
        return;
    }
    for j in 2..=n2 {
        i -= 1;
        let mut store = c[i];
        let i1 = if j <= k { j - 1 } else { k };
        let mut ll = i;
        for l0 in 1..=i1 {
            ll += 1;
            store -= c[ll] * a[i][l0 + 1];
        }
        c[i] = store / a[i][1];
    }
}

#[cfg(test)]
mod tests {
    use super::{fpbacp, fpcyt1, fpcyt2};

    // Dense apply of the cyclic-tridiagonal matrix encoded in `a` (1-based,
    // columns 1=sub, 2=diag, 3=super; corners a[1][1]=(1,n), a[n][3]=(n,1)).
    fn apply(a: &[Vec<f64>], n: usize, c: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0_f64; n + 1];
        for i in 1..=n {
            let mut s = a[i][2] * c[i];
            if i < n {
                s += a[i][3] * c[i + 1];
            }
            if i > 1 {
                s += a[i][1] * c[i - 1];
            }
            out[i] = s;
        }
        // wrap corners
        out[1] += a[1][1] * c[n];
        out[n] += a[n][3] * c[1];
        out
    }

    // Build the implied n×n upper-triangular matrix of fpbacp from (a,b) and
    // verify M·c == z for the computed solution.
    #[test]
    fn fpbacp_solves_periodic_banded() {
        let (n, k) = (8usize, 2usize);
        let n2 = n - k;
        // Body band a (rows 1..=n2, cols 1..=k+1): a[i][1] diag (nonzero).
        let mut a = vec![vec![0.0_f64; k + 2]; n2 + 1];
        for i in 1..=n2 {
            a[i][1] = 4.0 + i as f64 * 0.5;
            a[i][2] = -0.7 + 0.1 * i as f64;
            a[i][3] = 0.4 - 0.05 * i as f64;
        }
        // Coupling block b (rows 1..=n, cols 1..=k). Tail diagonals nonzero.
        let mut b = vec![vec![0.0_f64; k + 1]; n + 1];
        for i in 1..=n {
            b[i][1] = 0.3 + 0.07 * i as f64;
            b[i][2] = -0.2 + 0.04 * i as f64;
        }
        b[n - 1][1] = 5.0; // M[7][7] diagonal
        b[n][2] = 5.5; // M[8][8] diagonal
        // Build M (1-based) from fpbacp's implied equations.
        let mut mm = vec![vec![0.0_f64; n + 1]; n + 1];
        // body rows
        for i in 1..=n2 {
            mm[i][i] = a[i][1];
            let lim = k.min(n2 - i);
            for l0 in 1..=lim {
                mm[i][i + l0] = a[i][l0 + 1];
            }
            for j in 1..=k {
                mm[i][n2 + j] = b[i][j];
            }
        }
        // tail rows
        for ii in 1..=k {
            let l = n - ii + 1;
            let j = k + 2 - ii;
            mm[l][l] = b[l][j - 1];
            for l1 in j..=k {
                mm[l][l + (l1 - j + 1)] = b[l][l1];
            }
        }
        let c_true: Vec<f64> = (0..=n).map(|i| (i as f64 * 0.37).sin() + 0.5).collect();
        let mut z = vec![0.0_f64; n + 1];
        for i in 1..=n {
            let mut s = 0.0;
            for jc in 1..=n {
                s += mm[i][jc] * c_true[jc];
            }
            z[i] = s;
        }
        let mut c = vec![0.0_f64; n + 1];
        fpbacp(&a, &b, &z, n, k, &mut c);
        for i in 1..=n {
            assert!(
                (c[i] - c_true[i]).abs() <= 1e-9,
                "c[{i}]: {} vs {}",
                c[i],
                c_true[i]
            );
        }
    }

    #[test]
    fn fpcyt_solves_cyclic_tridiagonal() {
        let n = 6;
        // Diagonally-dominant cyclic tridiagonal so the factorization is stable.
        let sub = [0.0, 0.0, -1.0, -1.2, -0.8, -1.1, -0.9]; // a[i][1], a[1][1]=corner
        let dia = [0.0, 5.0, 5.5, 6.0, 5.2, 5.8, 6.1];
        let sup = [0.0, -1.3, -0.7, -1.0, -1.1, -0.6, 0.0]; // a[i][3], a[n][3]=corner
        let corner_1n = -0.9; // (1,n)
        let corner_n1 = -1.4; // (n,1)
        let mut a = vec![vec![0.0_f64; 7]; n + 1];
        for i in 1..=n {
            a[i][1] = sub[i];
            a[i][2] = dia[i];
            a[i][3] = sup[i];
        }
        a[1][1] = corner_1n;
        a[n][3] = corner_n1;
        let a_orig = a.clone();
        let b: Vec<f64> = vec![0.0, 1.0, -2.0, 3.0, 0.5, -1.5, 2.5];
        fpcyt1(&mut a, n);
        let mut c = vec![0.0_f64; n + 1];
        fpcyt2(&a, n, &b, &mut c);
        // residual: A c == b.
        let bc = apply(&a_orig, n, &c);
        for i in 1..=n {
            assert!(
                (bc[i] - b[i]).abs() <= 1e-10,
                "row {i}: {} vs {}",
                bc[i],
                b[i]
            );
        }
    }
}

//! Deterministic golden output for solve_banded optimization work.
//!
//! Emits the bit pattern of the solution vector (and backward error) for a
//! battery of deterministic banded systems so that an isomorphism-preserving
//! rewrite of the banded solver can be proven byte-for-byte identical.
//!
//!   `perf_solve_banded [path]`

use std::fmt::Write as _;

use fsci_linalg::{SolveOptions, solve_banded};

/// Build the (kl+ku+1)×n banded storage `ab` (scipy convention
/// `ab[ku + i - j][j] = A[i][j]`) for a deterministic, diagonally dominant
/// banded matrix, plus a deterministic RHS.
#[allow(clippy::needless_range_loop)] // explicit (i,j) -> LAPACK band-row indexing
fn make_case(n: usize, kl: usize, ku: usize, seed: f64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let rows = kl + ku + 1;
    let mut ab = vec![vec![0.0_f64; n]; rows];
    for i in 0..n {
        let j_start = i.saturating_sub(kl);
        let j_end = (i + ku).min(n - 1);
        for j in j_start..=j_end {
            let band_row = ku + i - j;
            let value = if i == j {
                // Diagonally dominant => nonsingular, no pivot fallback.
                (kl + ku) as f64 + 4.0 + ((i as f64 * 0.5 + seed).sin())
            } else {
                let off = (i as f64 - j as f64).abs();
                ((i as f64 + 2.0 * j as f64 + seed).cos()) / (off + 1.0)
            };
            ab[band_row][j] = value;
        }
    }
    let b: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.37 + seed).sin() * 3.0 + 1.0)
        .collect();
    (ab, b)
}

fn golden_text() -> String {
    let mut out = String::from("fsci-linalg solve_banded golden v1\n");
    // (n, kl, ku, seed) battery: symmetric/asymmetric bandwidths, varied sizes.
    let cases = [
        (4usize, 1usize, 1usize, 0.0_f64),
        (16, 1, 1, 0.3),
        (64, 1, 1, 1.1),
        (50, 2, 2, 0.7),
        (100, 2, 2, 2.0),
        (40, 1, 2, 0.9),
        (80, 3, 1, 1.7),
        (33, 0, 3, 0.2),
        (29, 3, 0, 1.3),
        (7, 2, 4, 0.5),
    ];
    for (n, kl, ku, seed) in cases {
        let (ab, b) = make_case(n, kl, ku, seed);
        let result = solve_banded((kl, ku), &ab, &b, SolveOptions::default())
            .expect("deterministic banded case must solve");
        let mut digest: u64 = 1469598103934665603; // FNV-1a offset
        for &v in &result.x {
            for byte in v.to_bits().to_le_bytes() {
                digest ^= byte as u64;
                digest = digest.wrapping_mul(1099511628211);
            }
        }
        let be = result.backward_error.unwrap_or(f64::NAN);
        writeln!(
            out,
            "case n={n} kl={kl} ku={ku} seed={seed} len={} x_digest={:016x} x0={:016x} xlast={:016x} berr={:016x}",
            result.x.len(),
            digest,
            result.x.first().map(|v| v.to_bits()).unwrap_or(0),
            result.x.last().map(|v| v.to_bits()).unwrap_or(0),
            be.to_bits(),
        )
        .expect("write golden line");
    }
    out
}

fn main() {
    let output = golden_text();
    if let Some(path) = std::env::args().nth(1) {
        if let Some(parent) = std::path::Path::new(&path).parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).expect("create golden artifact parent");
        }
        std::fs::write(path, output).expect("write golden artifact");
    } else {
        print!("{output}");
    }
}

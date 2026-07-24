//! Tiled (PLASMA-style) Cholesky factorization — foundation for the task-DAG
//! restructure (frankenscipy: authorized dense-lane rewrite).
//!
//! WHY. The production blocked Cholesky (`cholesky_lower_blocked_*`) has a
//! barrier per panel (factor → TRSM → trailing SYRK, a `thread::scope` join each
//! panel). At n=1000 on the 64-core fleet that caps fsci at ~37 GF/s while scipy's
//! OpenBLAS scales to ~102 GF/s (2.73x) — a THREADING-SCALING gap, not a kernel
//! gap. Closing it needs overlapping panel factor/TRSM with the bandwidth-bound
//! trailing update across cores with NO per-panel barrier, i.e. a tile
//! dependency-graph schedule. The blocker for that under `#![forbid(unsafe_code)]`
//! is the flat row-major layout: two tasks that write different COLUMNS of the
//! same rows can't both hold `&mut` (columns interleave per row). TILED storage
//! (each tile a separate contiguous block) makes disjoint tiles cleanly
//! borrow-splittable — the enabler for the concurrent schedule.
//!
//! INCREMENT 1 (this file): tiled storage + a SEQUENTIAL tiled factor with naive
//! (correct, not-yet-fast) tile kernels, validated by residual. No perf claim yet
//! — it establishes the layout + algorithm. Increment 2 adds the DAG scheduler +
//! pool (parallel, the win); increment 3 swaps in the fast FMA tile kernels.
//!
//! Tile-major storage: `data[(bi*nt + bj)*ts*ts + r*ts + c]` is element (r,c) of
//! tile (bi,bj); the global matrix is padded to `nt*ts` with an identity block on
//! the padding diagonal (its Cholesky is identity and does not perturb the
//! top-left n×n factor), so every tile is exactly `ts×ts`.

/// Factor a symmetric positive-definite `a` (lower triangle read) into its lower
/// Cholesky factor `L` (A = L·Lᵀ), returned row-major flat (`n*n`, lower triangle
/// populated, strict upper zero). `ts` is the tile edge. Returns `None` if `a` is
/// not SPD (a non-positive pivot) or not square. Sequential; correctness-focused.
#[allow(dead_code)]
pub(crate) fn cholesky_tiled_lower(a: &[Vec<f64>], ts: usize) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return Some(Vec::new());
    }
    if a.iter().any(|row| row.len() != n) || ts == 0 {
        return None;
    }
    let nt = n.div_ceil(ts);
    let np = nt * ts; // padded dimension
    let tile_area = ts * ts;
    let mut data = vec![0.0f64; nt * nt * tile_area];

    let tile_base = |bi: usize, bj: usize| (bi * nt + bj) * tile_area;
    // Global (gi,gj) -> flat index in `data`.
    let gidx = |gi: usize, gj: usize| {
        let (bi, bj) = (gi / ts, gj / ts);
        tile_base(bi, bj) + (gi % ts) * ts + (gj % ts)
    };

    // Load the lower triangle of `a` into the tiles; identity on the padding diag.
    for gi in 0..n {
        for gj in 0..=gi {
            data[gidx(gi, gj)] = a[gi][gj];
        }
    }
    for gi in n..np {
        data[gidx(gi, gi)] = 1.0;
    }

    // Sequential right-looking tiled factorization.
    for k in 0..nt {
        if !potrf_tile(&mut data, tile_base(k, k), ts) {
            return None;
        }
        for i in (k + 1)..nt {
            trsm_tile(&mut data, tile_base(i, k), tile_base(k, k), ts);
        }
        for j in (k + 1)..nt {
            syrk_tile(&mut data, tile_base(j, j), tile_base(j, k), ts);
            for i in (j + 1)..nt {
                gemm_tile(
                    &mut data,
                    tile_base(i, j),
                    tile_base(i, k),
                    tile_base(j, k),
                    ts,
                );
            }
        }
    }

    // Extract the top-left n×n lower triangle into a row-major flat factor.
    let mut lower = vec![0.0f64; n * n];
    for gi in 0..n {
        for gj in 0..=gi {
            lower[gi * n + gj] = data[gidx(gi, gj)];
        }
    }
    Some(lower)
}

// Increment 2a: the four tile kernels' inner products are contiguous-row dot
// products, so they vectorize by reusing the crate's `simd_dot` (8-wide SIMD).
// Two immutable `&data[..]` borrows for the dot end before each mutable write;
// the reassociated sum is byte-close (factor unique to 1e-10, residual-gated).

/// In-place unblocked Cholesky of the `ts×ts` lower tile at `base` (reads/writes
/// only the lower triangle). Returns false on a non-positive/non-finite pivot.
fn potrf_tile(data: &mut [f64], base: usize, ts: usize) -> bool {
    for j in 0..ts {
        let jrow = base + j * ts;
        let d = data[jrow + j] - crate::simd_dot(&data[jrow..jrow + j], &data[jrow..jrow + j]);
        if d <= 0.0 || !d.is_finite() {
            return false;
        }
        let ljj = d.sqrt();
        data[jrow + j] = ljj;
        for i in (j + 1)..ts {
            let irow = base + i * ts;
            let dot = crate::simd_dot(&data[irow..irow + j], &data[jrow..jrow + j]);
            data[irow + j] = (data[irow + j] - dot) / ljj;
        }
    }
    true
}

/// TRSM: solve `L_ik · L_kkᵀ = A_ik` for `L_ik` in place at `ik` (L_kk lower tile
/// at `kk`). Each row is an independent forward substitution over columns.
fn trsm_tile(data: &mut [f64], ik: usize, kk: usize, ts: usize) {
    for r in 0..ts {
        let irow = ik + r * ts;
        for c in 0..ts {
            let crow = kk + c * ts;
            let dot = crate::simd_dot(&data[irow..irow + c], &data[crow..crow + c]);
            data[irow + c] = (data[irow + c] - dot) / data[crow + c];
        }
    }
}

/// SYRK: `A_jj -= L_jk · L_jkᵀ` (lower triangle of the diagonal tile at `jj`).
fn syrk_tile(data: &mut [f64], jj: usize, jk: usize, ts: usize) {
    for r in 0..ts {
        let rrow = jk + r * ts;
        for c in 0..=r {
            let dot = crate::simd_dot(&data[rrow..rrow + ts], &data[jk + c * ts..jk + c * ts + ts]);
            data[jj + r * ts + c] -= dot;
        }
    }
}

/// GEMM: `A_ij -= L_ik · L_jkᵀ` (full off-diagonal tile at `ij`, i>j).
fn gemm_tile(data: &mut [f64], ij: usize, ik: usize, jk: usize, ts: usize) {
    for r in 0..ts {
        let rrow = ik + r * ts;
        for c in 0..ts {
            let dot = crate::simd_dot(&data[rrow..rrow + ts], &data[jk + c * ts..jk + c * ts + ts]);
            data[ij + r * ts + c] -= dot;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spd(n: usize, seed: u64) -> Vec<Vec<f64>> {
        // A = M·Mᵀ + n·I (well-conditioned SPD), deterministic xorshift.
        let mut s = seed | 1;
        let mut r = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let m: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| r()).collect()).collect();
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += m[i][k] * m[j][k];
                }
                a[i][j] = acc + if i == j { n as f64 } else { 0.0 };
            }
        }
        a
    }

    /// Residual gate: ‖L·Lᵀ − A‖_max / ‖A‖_max ≤ 1e-10 across n and tile edges,
    /// including n not a multiple of ts (padding path) and ts ≥ n (single tile).
    #[test]
    fn tiled_cholesky_residual_matches_input() {
        for &n in &[1usize, 2, 5, 8, 16, 17, 33, 64, 100, 129] {
            let a = spd(n, 0x9E37_79B9 ^ n as u64);
            let scale = a
                .iter()
                .flatten()
                .fold(1.0_f64, |m, &v| m.max(v.abs()));
            for &ts in &[1usize, 4, 8, 16, 32, 64, 256] {
                let l = cholesky_tiled_lower(&a, ts)
                    .unwrap_or_else(|| panic!("tiled factor failed n={n} ts={ts}"));
                let mut max_abs = 0.0_f64;
                for i in 0..n {
                    for j in 0..=i {
                        let mut recon = 0.0;
                        for p in 0..=j {
                            recon += l[i * n + p] * l[j * n + p];
                        }
                        max_abs = max_abs.max((recon - a[i][j]).abs());
                    }
                }
                assert!(
                    max_abs <= 1e-10 * scale,
                    "tiled residual {max_abs:.3e} (scale {scale:.3e}) n={n} ts={ts}"
                );
                // strict upper triangle must be zero
                for i in 0..n {
                    for j in (i + 1)..n {
                        assert_eq!(l[i * n + j], 0.0, "upper nonzero at ({i},{j}) n={n} ts={ts}");
                    }
                }
            }
        }
    }

    /// Non-SPD input (a zero pivot) must fail closed, not panic.
    #[test]
    fn tiled_cholesky_rejects_non_spd() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 1.0]]; // indefinite (eigenvalues 3, -1)
        assert!(cholesky_tiled_lower(&a, 1).is_none());
        assert!(cholesky_tiled_lower(&a, 8).is_none());
    }
}

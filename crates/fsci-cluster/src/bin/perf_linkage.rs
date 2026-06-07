//! Same-process A/B + isomorphism harness for hierarchical `linkage`.
//!
//! `naive_agglomerate` reproduces the original O(n^3) rescan-every-pair loop;
//! the library now uses an O(n^2) nearest-neighbour-array core. We prove the
//! full linkage matrix is byte-identical (`.to_bits()` on every cell) across
//! randomized datasets and all methods, then time the worst case.
//! Run: `cargo run --release -p fsci-cluster --bin perf_linkage`.
#![allow(clippy::needless_range_loop)]

use fsci_cluster::{LinkageMethod, linkage, linkage_from_distances};
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

const METHODS: [LinkageMethod; 7] = [
    LinkageMethod::Single,
    LinkageMethod::Complete,
    LinkageMethod::Average,
    LinkageMethod::Ward,
    LinkageMethod::Weighted,
    LinkageMethod::Centroid,
    LinkageMethod::Median,
];

/// Verbatim original O(n^3) agglomeration over a full distance matrix.
fn naive_agglomerate(n: usize, dist: &[Vec<f64>], method: LinkageMethod) -> Vec<[f64; 4]> {
    let total = 2 * n - 1;
    let mut active = vec![true; total];
    let mut cluster_size = vec![1usize; total];
    let mut inter_dist = vec![vec![f64::INFINITY; total]; total];
    for (i, row) in dist.iter().enumerate().take(n) {
        inter_dist[i][..n].copy_from_slice(&row[..n]);
    }
    let mut result = Vec::with_capacity(n - 1);
    for step in 0..n - 1 {
        let new_id = n + step;
        let mut min_d = f64::INFINITY;
        let mut mi = 0;
        let mut mj = 0;
        for i in 0..new_id {
            if !active[i] {
                continue;
            }
            for j in i + 1..new_id {
                if active[j] && inter_dist[i][j] < min_d {
                    min_d = inter_dist[i][j];
                    mi = i;
                    mj = j;
                }
            }
        }
        let new_size = cluster_size[mi] + cluster_size[mj];
        result.push([mi as f64, mj as f64, min_d, new_size as f64]);
        active[mi] = false;
        active[mj] = false;
        active[new_id] = true;
        cluster_size[new_id] = new_size;
        for k in 0..new_id {
            if !active[k] || k == new_id {
                continue;
            }
            let d_ki = inter_dist[k][mi];
            let d_kj = inter_dist[k][mj];
            let new_dist = match method {
                LinkageMethod::Single => d_ki.min(d_kj),
                LinkageMethod::Complete => d_ki.max(d_kj),
                LinkageMethod::Average => {
                    let ni = cluster_size[mi] as f64;
                    let nj = cluster_size[mj] as f64;
                    (ni * d_ki + nj * d_kj) / (ni + nj)
                }
                LinkageMethod::Ward => {
                    let ni = cluster_size[mi] as f64;
                    let nj = cluster_size[mj] as f64;
                    let nk = cluster_size[k] as f64;
                    let nt = ni + nj + nk;
                    (((nk + ni) * d_ki * d_ki + (nk + nj) * d_kj * d_kj - nk * min_d * min_d) / nt)
                        .max(0.0)
                        .sqrt()
                }
                LinkageMethod::Weighted => 0.5 * (d_ki + d_kj),
                LinkageMethod::Centroid => {
                    let ni = cluster_size[mi] as f64;
                    let nj = cluster_size[mj] as f64;
                    let nt = ni + nj;
                    let alpha_i = ni / nt;
                    let alpha_j = nj / nt;
                    let beta = -(ni * nj) / (nt * nt);
                    (alpha_i * d_ki * d_ki + alpha_j * d_kj * d_kj + beta * min_d * min_d)
                        .max(0.0)
                        .sqrt()
                }
                LinkageMethod::Median => (0.5 * d_ki * d_ki + 0.5 * d_kj * d_kj
                    - 0.25 * min_d * min_d)
                    .max(0.0)
                    .sqrt(),
            };
            inter_dist[k][new_id] = new_dist;
            inter_dist[new_id][k] = new_dist;
        }
    }
    result
}

fn condensed_from_data(data: &[Vec<f64>]) -> Vec<f64> {
    let n = data.len();
    let mut out = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in i + 1..n {
            let d: f64 = data[i]
                .iter()
                .zip(&data[j])
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            out.push(d);
        }
    }
    out
}

fn full_from_condensed(cond: &[f64], n: usize) -> Vec<Vec<f64>> {
    let mut dist = vec![vec![f64::INFINITY; n]; n];
    let mut idx = 0;
    for i in 0..n {
        dist[i][i] = 0.0;
        for j in i + 1..n {
            dist[i][j] = cond[idx];
            dist[j][i] = cond[idx];
            idx += 1;
        }
    }
    dist
}

fn matrices_eq(a: &[[f64; 4]], b: &[[f64; 4]]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(ra, rb)| ra.iter().zip(rb).all(|(x, y)| x.to_bits() == y.to_bits()))
}

fn main() {
    let mut r = Lcg(0x1234_abcd_5678_ef01);
    let mut total = 0usize;
    let mut mismatches = 0usize;
    let mut payload = String::new();

    for trial in 0..600 {
        let n = 3 + (r.next_u64() as usize % 40);
        let dim = 1 + (r.next_u64() as usize % 4);
        // Mix coordinate alphabets: small integer grids stress ties / inversions.
        let coarse = trial % 3 == 0;
        let data: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        if coarse {
                            (r.next_u64() % 4) as f64
                        } else {
                            r.unit() * 10.0
                        }
                    })
                    .collect()
            })
            .collect();

        let cond = condensed_from_data(&data);
        let full = full_from_condensed(&cond, n);

        for method in METHODS {
            // linkage_from_distances exercises the nn-array core for ALL methods.
            let want = naive_agglomerate(n, &full, method);
            let got = linkage_from_distances(&cond, n, method).unwrap();
            total += 1;
            if !matrices_eq(&want, &got) {
                mismatches += 1;
                if payload.len() < 3000 {
                    payload.push_str(&format!(
                        "MISMATCH lfd trial={trial} n={n} method={method:?}\n"
                    ));
                }
            }
            // linkage() routes Centroid/Median to the heap path; check the other 5.
            if !matches!(method, LinkageMethod::Centroid | LinkageMethod::Median) {
                let got2 = linkage(&data, method).unwrap();
                total += 1;
                if !matrices_eq(&want, &got2) {
                    mismatches += 1;
                    if payload.len() < 3000 {
                        payload.push_str(&format!(
                            "MISMATCH lk trial={trial} n={n} method={method:?}\n"
                        ));
                    }
                }
            }
        }

        // Compact digest of one method's matrix for golden hashing.
        let z = linkage_from_distances(&cond, n, LinkageMethod::Ward).unwrap();
        let digest: u64 = z.iter().flatten().fold(1469598103934665603u64, |h, x| {
            (h ^ x.to_bits()).wrapping_mul(1099511628211)
        });
        payload.push_str(&format!("trial={trial} n={n} ward_digest={digest:016x}\n"));
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!(
        "isomorphism: {mismatches} mismatches / {total} linkage matrices (0 == byte-identical)"
    );

    // ---- Timing: clustered random data, ward, growing n ----
    for &n in &[200usize, 400, 800] {
        let data: Vec<Vec<f64>> = (0..n)
            .map(|_| vec![r.unit() * 50.0, r.unit() * 50.0])
            .collect();
        let cond = condensed_from_data(&data);
        let full = full_from_condensed(&cond, n);

        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..3 {
            acc += naive_agglomerate(n, &full, LinkageMethod::Ward)[0][2];
        }
        let naive_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..3 {
            acc += linkage_from_distances(&cond, n, LinkageMethod::Ward).unwrap()[0][2];
        }
        let fast_t = t1.elapsed();

        let ratio = naive_t.as_secs_f64() / fast_t.as_secs_f64();
        println!(
            "n={n:>5}  naive={:>10.3?}  nnarray={:>10.3?}  ratio={ratio:>7.1}x  (acc={acc:.3})",
            naive_t / 3,
            fast_t / 3
        );
    }
}

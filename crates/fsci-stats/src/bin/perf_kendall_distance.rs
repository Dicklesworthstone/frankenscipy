//! Same-process A/B + isomorphism harness for `kendall_distance`.
//!
//! `old_kendall_distance` is a verbatim copy of the original O(n^2) all-pairs
//! sign-comparison loop. The library now counts discordant pairs as strict
//! rank2-inversions over the (rank1, rank2)-lexicographic order via merge sort
//! (O(n log n)). The discordant count is an exact integer, so the two are
//! bit-for-bit identical; we assert 0 mismatches across shapes/tie-densities and
//! time the win on large rankings.
//! Run: `cargo run --release -p fsci-stats --bin perf_kendall_distance`.

use fsci_stats::kendall_distance;
use std::time::Instant;

/// Verbatim copy of the original O(n^2) kendall_distance.
fn old_kendall_distance(rank1: &[usize], rank2: &[usize]) -> usize {
    let n = rank1.len();
    if n != rank2.len() {
        return 0;
    }
    let mut count = 0;
    for i in 0..n {
        for j in i + 1..n {
            let a = (rank1[i] as i64 - rank1[j] as i64).signum();
            let b = (rank2[i] as i64 - rank2[j] as i64).signum();
            if a != b && a != 0 && b != 0 {
                count += 1;
            }
        }
    }
    count
}

// Deterministic LCG so the harness needs no rng dependency.
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
}

/// Build two rankings with a controllable tie density (modulus `m`).
fn make_case(n: usize, m: usize, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut rng = Lcg(seed);
    let mut r1 = Vec::with_capacity(n);
    let mut r2 = Vec::with_capacity(n);
    for _ in 0..n {
        r1.push((rng.next_u64() as usize) % m.max(1));
        r2.push((rng.next_u64() as usize) % m.max(1));
    }
    (r1, r2)
}

fn main() {
    // ---- Isomorphism: exact-integer parity across shapes and tie densities ----
    let mut mismatches = 0usize;
    let mut total = 0usize;
    let mut payload = String::new();
    for &n in &[2usize, 3, 8, 17, 64, 257, 1000] {
        // m = n (mostly distinct) down to m small (heavy ties).
        for &m in &[n.max(2), n / 2 + 1, 8, 3, 1] {
            for seed in 0..6u64 {
                let (r1, r2) = make_case(n, m, seed * 2654435761 + 1);
                let got = kendall_distance(&r1, &r2);
                let want = old_kendall_distance(&r1, &r2);
                total += 1;
                if got != want {
                    mismatches += 1;
                    if payload.len() < 1500 {
                        payload.push_str(&format!(
                            "MISMATCH n={n} m={m} seed={seed} got={got} want={want}\n"
                        ));
                    }
                }
                if payload.len() < 1500 {
                    payload.push_str(&format!("n={n} m={m} seed={seed} d={got}\n"));
                }
            }
        }
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!(
        "isomorphism: {mismatches} mismatches / {total} cases (0 == byte-identical integer count)"
    );

    // ---- Timing: large rankings (mostly-distinct, the O(n^2) worst case) ----
    for &n in &[4000usize, 8000, 16000] {
        let (r1, r2) = make_case(n, n, 99);

        let t0 = Instant::now();
        let mut acc = 0usize;
        for _ in 0..3 {
            acc = acc.wrapping_add(old_kendall_distance(&r1, &r2));
        }
        let old_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..3 {
            acc = acc.wrapping_add(kendall_distance(&r1, &r2));
        }
        let new_t = t1.elapsed();

        let ratio = old_t.as_secs_f64() / new_t.as_secs_f64();
        println!(
            "n={n:>6}  old={:>11.3?}  new={:>11.3?}  ratio={ratio:>7.1}x  (acc={acc})",
            old_t / 3,
            new_t / 3
        );
    }
}

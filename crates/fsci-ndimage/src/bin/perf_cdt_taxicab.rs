//! Same-process A/B + isomorphism harness for the city-block (taxicab) distance
//! transform shared by distance_transform_cdt and distance_transform_bf.
//!
//! `brute_l1` reproduces the original O(foreground · background) scan; the
//! library now uses a separable two-pass chamfer. We prove byte-identical f64
//! output (`.to_bits()`) across randomized binary images and time both.
//! Run: `cargo run --release -p fsci-ndimage --bin perf_cdt_taxicab`.

use fsci_ndimage::{DistanceMetric, NdArray, distance_transform_bf, distance_transform_cdt};
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
    fn bit(&mut self, fg_pct: u64) -> f64 {
        if (self.next_u64() >> 11) % 100 < fg_pct {
            1.0
        } else {
            0.0
        }
    }
}

fn make_image(shape: &[usize], fg_pct: u64, seed: u64) -> NdArray {
    let total: usize = shape.iter().product();
    let mut r = Lcg(seed);
    let data: Vec<f64> = (0..total).map(|_| r.bit(fg_pct)).collect();
    NdArray::new(data, shape.to_vec()).unwrap()
}

/// Original brute-force taxicab (L1) transform, verbatim arithmetic.
fn brute_l1(input: &NdArray) -> Vec<f64> {
    let ndim = input.shape.len();
    let strides = &input.strides;
    let unravel = |mut flat: usize| -> Vec<usize> {
        let mut idx = vec![0usize; ndim];
        for d in 0..ndim {
            idx[d] = flat / strides[d];
            flat %= strides[d];
        }
        idx
    };
    let backgrounds: Vec<Vec<usize>> = input
        .data
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v == 0.0)
        .map(|(flat, _)| unravel(flat))
        .collect();

    let mut out = vec![0.0f64; input.data.len()];
    for (flat, &value) in input.data.iter().enumerate() {
        if value == 0.0 {
            continue;
        }
        let coords = unravel(flat);
        let min_d = backgrounds
            .iter()
            .map(|bg| {
                coords
                    .iter()
                    .zip(bg)
                    .map(|(&c, &b)| c.abs_diff(b) as f64)
                    .sum::<f64>()
            })
            .fold(f64::INFINITY, f64::min);
        out[flat] = min_d;
    }
    out
}

fn main() {
    let shapes: &[&[usize]] = &[
        &[24],
        &[10, 10],
        &[16, 24],
        &[5, 7, 9],
        &[1, 19],
        &[4, 4, 4],
    ];

    let mut mismatches = 0usize;
    let mut compared = 0usize;
    let mut payload = String::new();
    let mut seed = 0x51a7_c0de_face_b00cu64;
    for shape in shapes {
        for fg in [20u64, 40, 60, 80] {
            seed = seed
                .wrapping_mul(2862933555777941757)
                .wrapping_add(3037000493);
            let img = make_image(shape, fg, seed);
            // Skip the all-foreground case (sentinel path, not the fast path).
            if img.data.iter().all(|&v| v != 0.0) {
                continue;
            }
            let brute = brute_l1(&img);
            for (name, lib) in [
                (
                    "cdt",
                    distance_transform_cdt(&img, DistanceMetric::Taxicab).unwrap(),
                ),
                (
                    "bf",
                    distance_transform_bf(&img, DistanceMetric::Taxicab, None).unwrap(),
                ),
            ] {
                for (a, b) in lib.data.iter().zip(brute.iter()) {
                    compared += 1;
                    if a.to_bits() != b.to_bits() {
                        mismatches += 1;
                    }
                }
                let digest: u64 = lib
                    .data
                    .iter()
                    .enumerate()
                    .fold(1469598103934665603u64, |h, (i, &x)| {
                        (h ^ x.to_bits() ^ (i as u64)).wrapping_mul(1099511628211)
                    });
                payload.push_str(&format!(
                    "{name} shape={shape:?} fg={fg} digest={digest:016x}\n"
                ));
            }
        }
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!("isomorphism: {mismatches} mismatches / {compared} cells (0 == byte-identical)");

    for &side in &[64usize, 128, 192] {
        let img = make_image(&[side, side], 50, 0xD15 + side as u64);

        let t0 = Instant::now();
        let mut acc = 0.0f64;
        for _ in 0..3 {
            acc += brute_l1(&img)[0];
        }
        let brute_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..3 {
            acc += distance_transform_cdt(&img, DistanceMetric::Taxicab)
                .unwrap()
                .data[0];
        }
        let sep_t = t1.elapsed();

        let ratio = brute_t.as_secs_f64() / sep_t.as_secs_f64();
        println!(
            "N={:>6} ({side}x{side})  brute={:>10.3?}  sep={:>10.3?}  ratio={ratio:>7.1}x  (acc={acc:.1})",
            side * side,
            brute_t / 3,
            sep_t / 3
        );
    }
}

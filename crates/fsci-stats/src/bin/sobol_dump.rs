//! Dump unscrambled SobolSampler output as f64 bits for dimensions 1..=32, so an
//! external check can compare it byte-for-byte against
//! `scipy.stats.qmc.Sobol(d, scramble=False).random(N)`.
//! Each line is `d <bit0> <bit1> ...` (lowercase hex, point-major / dim-minor,
//! matching SobolSampler::sample's output order).
//! Run: `cargo run --release -p fsci-stats --bin sobol_dump`.

use fsci_stats::qmc::SobolSampler;

fn main() {
    const N: usize = 64;
    for d in 1..=32usize {
        let mut s = SobolSampler::new(d).expect("dimension in range");
        let pts = s.sample(N);
        let mut line = format!("{d}");
        for &v in &pts {
            line.push_str(&format!(" {:016x}", v.to_bits()));
        }
        println!("{line}");
    }
}

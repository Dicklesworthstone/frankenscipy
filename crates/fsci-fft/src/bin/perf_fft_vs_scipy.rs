use fsci_fft::{Complex64, FftOptions, fft, rfft};
use std::time::Instant;
fn main() {
    let mut seed = 12345u64;
    let mut r = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let o = FftOptions::default();
    for &e in &[18usize, 20, 22] {
        let n = 1usize << e;
        let xc: Vec<Complex64> = (0..n).map(|_| (r(), r())).collect();
        let xr: Vec<f64> = (0..n).map(|_| r()).collect();
        let _ = fft(&xc, &o);
        let t = Instant::now();
        for _ in 0..10 {
            let _ = fft(&xc, &o);
        }
        let fc = t.elapsed().as_secs_f64() / 10.0 * 1000.0;
        let _ = rfft(&xr, &o);
        let t = Instant::now();
        for _ in 0..10 {
            let _ = rfft(&xr, &o);
        }
        let fr = t.elapsed().as_secs_f64() / 10.0 * 1000.0;
        println!("fsci 2^{e}: fft {fc:.2} ms, rfft {fr:.2} ms");
    }
}

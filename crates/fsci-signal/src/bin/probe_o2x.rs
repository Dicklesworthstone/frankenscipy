use fsci_signal::{ShortTimeFft, StftFftMode, StftScaling};
fn main() {
    let hann = |n: usize| -> Vec<f64> {
        (0..n)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
            .collect()
    };
    let n = 20usize;
    let x: Vec<f64> = (0..n)
        .map(|i| (0.3 * i as f64).sin() + 0.5 * (0.07 * i as f64).cos())
        .collect();
    // onesided2X + psd scaling, mfft=8 (even)
    let s2x = ShortTimeFft::new(hann(8), 3, 100.0)
        .unwrap()
        .with_scale_to(StftScaling::Psd)
        .with_fft_mode(StftFftMode::Onesided2X);
    let s = s2x.stft(&x).unwrap();
    print!("O2X");
    for row in &s {
        for c in row {
            print!(" {:.12e} {:.12e}", c.0, c.1);
        }
    }
    println!();
    let xr = s2x.istft(&s, 0, None).unwrap();
    let m = xr.len().min(n);
    let mut e = 0.0f64;
    for i in 0..m {
        e = e.max((xr[i] - x[i]).abs());
    }
    println!("O2X_ROUNDTRIP maxerr={:.3e}", e);
}

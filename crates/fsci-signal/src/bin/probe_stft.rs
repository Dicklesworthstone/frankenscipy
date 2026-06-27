use fsci_signal::{ShortTimeFft, StftFftMode};
fn main() {
    let hann = |n: usize| -> Vec<f64> {
        (0..n)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
            .collect()
    };
    // config: win=hann(8), hop=3, fs=100, onesided default
    let sft = ShortTimeFft::new(hann(8), 3, 100.0).unwrap();
    let n = 20usize;
    let x: Vec<f64> = (0..n)
        .map(|i| (0.3 * i as f64).sin() + 0.5 * (0.07 * i as f64).cos())
        .collect();
    println!(
        "GEOM p_min={} p_max={} k_min={} k_max={} p_num={} f_pts={} delta_t={:.6} delta_f={:.6}",
        sft.p_min(),
        sft.p_max(n),
        sft.k_min(),
        sft.k_max(n),
        sft.p_num(n),
        sft.f_pts(),
        sft.delta_t(),
        sft.delta_f()
    );
    let s = sft.stft(&x).unwrap();
    print!("STFT");
    for row in &s {
        for c in row {
            print!(" {:.10e} {:.10e}", c.0, c.1);
        }
    }
    println!();
    // twosided + mfft=16
    let sft2 = ShortTimeFft::new(hann(8), 4, 100.0)
        .unwrap()
        .with_mfft(16)
        .unwrap()
        .with_fft_mode(StftFftMode::Twosided);
    let s2 = sft2.stft(&x).unwrap();
    println!(
        "GEOM2 p_min={} p_max={} f_pts={}",
        sft2.p_min(),
        sft2.p_max(n),
        sft2.f_pts()
    );
    print!("STFT2");
    for row in &s2 {
        for c in row {
            print!(" {:.10e} {:.10e}", c.0, c.1);
        }
    }
    println!();
}

use fsci_signal::{ShortTimeFft, StftScaling};
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
    // magnitude scaling
    let sm = ShortTimeFft::new(hann(8), 3, 100.0)
        .unwrap()
        .with_scale_to(StftScaling::Magnitude);
    let s = sm.stft(&x).unwrap();
    print!("MAG");
    for c in &s[0] {
        print!(" {:.12e} {:.12e}", c.0, c.1);
    }
    println!();
    // psd scaling
    let sp = ShortTimeFft::new(hann(8), 3, 100.0)
        .unwrap()
        .with_scale_to(StftScaling::Psd);
    let sps = sp.stft(&x).unwrap();
    print!("PSD");
    for c in &sps[0] {
        print!(" {:.12e} {:.12e}", c.0, c.1);
    }
    println!();
    // t and extent (unscaled onesided)
    let su = ShortTimeFft::new(hann(8), 3, 100.0).unwrap();
    print!("T");
    for v in su.t(n) {
        print!(" {:.10e}", v);
    }
    println!();
    let e = su.extent(n).unwrap();
    println!("EXTENT {:.10e} {:.10e} {:.10e} {:.10e}", e.0, e.1, e.2, e.3);
}

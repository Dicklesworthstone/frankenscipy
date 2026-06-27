use fsci_signal::ShortTimeFft;
fn main() {
    let hann = |n: usize| -> Vec<f64> {
        (0..n)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
            .collect()
    };
    let sft = ShortTimeFft::new(hann(8), 3, 100.0).unwrap();
    let n = 20usize;
    let x: Vec<f64> = (0..n)
        .map(|i| (0.3 * i as f64).sin() + 0.5 * (0.07 * i as f64).cos())
        .collect();
    let s = sft.stft(&x).unwrap();
    let xr = sft.istft(&s, 0, None).unwrap();
    print!("ISTFT");
    for v in &xr {
        print!(" {:.12e}", v);
    }
    println!();
    // round-trip error vs original x (compare overlapping region)
    let m = xr.len().min(n);
    let mut maxe = 0.0f64;
    for i in 0..m {
        maxe = maxe.max((xr[i] - x[i]).abs());
    }
    println!("ROUNDTRIP len={} n={} maxerr={:.3e}", xr.len(), n, maxe);
}

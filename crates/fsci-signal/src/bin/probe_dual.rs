use fsci_signal::closest_stft_dual_window;
fn pr(tag: &str, d: &[f64], a: f64) {
    print!("{tag} {a:.15e}");
    for v in d {
        print!(" {v:.15e}");
    }
    println!();
}
fn main() {
    // hann-ish window via formula, several configs
    let mk = |n: usize| -> Vec<f64> {
        (0..n)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
            .collect()
    };
    let w8 = mk(8);
    let w16 = mk(16);
    let w12 = mk(12);
    for (tag, w, hop, sc) in [
        ("A", &w8, 2usize, true),
        ("B", &w8, 2, false),
        ("C", &w16, 4, true),
        ("D", &w16, 5, true),
        ("E", &w12, 3, true),
        ("F", &w12, 3, false),
    ] {
        let (d, a) = closest_stft_dual_window(w, hop, None, sc).unwrap();
        pr(tag, &d, a);
    }
    // with a desired_dual (non-rect)
    let dd: Vec<f64> = (0..16).map(|i| 1.0 + 0.1 * i as f64).collect();
    let (d, a) = closest_stft_dual_window(&w16, 4, Some(&dd), true).unwrap();
    pr("G", &d, a);
}

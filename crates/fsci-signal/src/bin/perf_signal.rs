use fsci_signal::{cwt, remez, ricker};

fn deterministic_signal(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| {
            let t = i as f64 / len as f64;
            (37.0 * t).sin() + 0.35 * (91.0 * t).cos() + 0.1 * ((i * 17 % 29) as f64 - 14.0)
        })
        .collect()
}

fn main() {
    let bands = [0.0, 0.2, 0.3, 0.5];
    let desired = [1.0, 0.0];
    let weights = [1.0, 10.0];
    let taps = remez(257, &bands, &desired, Some(&weights)).expect("remez golden should solve");

    println!("case=remez_257_two_band len={}", taps.len());
    for (idx, tap) in taps.iter().enumerate() {
        println!("{idx:03} {tap:.17e}");
    }

    // Mirrors `wavelets/cwt_ricker/2048x32` in signal_bench.rs.
    let x = deterministic_signal(2048);
    let widths: Vec<f64> = (1..=32).map(|w| w as f64).collect();
    let coeffs = cwt(&x, ricker, &widths).expect("cwt golden should solve");

    println!(
        "case=cwt_ricker_2048x32 rows={} cols={}",
        coeffs.len(),
        x.len()
    );
    for (row, scale) in coeffs.iter().enumerate() {
        for (col, value) in scale.iter().enumerate() {
            println!("{row:02} {col:04} {value:.17e}");
        }
    }
}

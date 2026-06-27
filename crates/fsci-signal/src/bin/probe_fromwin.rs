use fsci_signal::ShortTimeFft;
fn main() {
    let sft = ShortTimeFft::from_window("hann", 100.0, 8, 5, false).unwrap();
    let n = 20usize;
    let x: Vec<f64> = (0..n)
        .map(|i| (0.3 * i as f64).sin() + 0.5 * (0.07 * i as f64).cos())
        .collect();
    println!(
        "FW hop={} mnum={} p_min={} p_max={}",
        sft.hop(),
        sft.m_num(),
        sft.p_min(),
        sft.p_max(n)
    );
    let s = sft.stft(&x).unwrap();
    print!("FWSTFT");
    for c in &s[1] {
        print!(" {:.12e} {:.12e}", c.0, c.1);
    }
    println!();
}

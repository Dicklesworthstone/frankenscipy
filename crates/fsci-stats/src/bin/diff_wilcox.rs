use fsci_stats::*;
fn main() {
    // deterministic pseudo-random continuous pairs (no ties) + a tie case
    let mut s: u64 = 0x1234567;
    let mut rng = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s as f64 / u64::MAX as f64
    };
    for n in [5usize, 6, 8, 10, 12, 15, 20, 30] {
        let x: Vec<f64> = (0..n).map(|_| rng() * 10.0).collect();
        let y: Vec<f64> = (0..n).map(|_| rng() * 10.0).collect();
        for alt in ["two-sided", "less", "greater"] {
            let r = wilcoxon_alternative(&x, &y, alt);
            print!("{n}|{alt}|{:.17e}|{:.17e}|", r.statistic, r.pvalue);
            let xs: Vec<String> = x.iter().map(|v| format!("{v:.10}")).collect();
            let ys: Vec<String> = y.iter().map(|v| format!("{v:.10}")).collect();
            println!("{}|{}", xs.join(","), ys.join(","));
        }
    }
}

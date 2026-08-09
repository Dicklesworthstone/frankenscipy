use fsci_stats::*;
fn main() {
    let ds: Vec<Vec<f64>> = vec![
        vec![0.1, 0.2, 0.3],
        vec![6.0, 6.1, 0.1],
        vec![-0.1, 0.05, 6.2],
        vec![3.0, 3.1, 3.2],
        // 6.28 is a sample angle rounded to 2 decimals, not TAU
        // (frankenscipy-023vy); the fixture value must not move.
        #[allow(clippy::approx_constant)]
        vec![0.1, 6.28, 3.0],
        vec![5.5, 5.6, 5.7, 0.2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![-3.0, -2.5, 2.5, 3.0],
    ];
    for (i, d) in ds.iter().enumerate() {
        println!("circmean_{i},{:.17e}", circmean(d));
        println!("circvar_{i},{:.17e}", circvar(d));
        println!("circstd_{i},{:.17e}", circstd(d));
    }
    let w = [1.0, 2.0, 3.0];
    println!("circmeanw,{:.17e}", circmean_weighted(&[6.0, 6.1, 0.1], &w));
}

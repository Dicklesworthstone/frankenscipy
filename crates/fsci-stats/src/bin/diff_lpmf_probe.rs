use fsci_special::log_ive_scalar;
use fsci_stats::{DiscreteDistribution, Skellam};
fn main() {
    // direct log_ive at large order
    for &(v, x) in &[
        (267.0, 12.649),
        (100.0, 12.649),
        (50.0, 8.0),
        (500.0, 20.0),
        (10.0, 5.0),
    ] {
        println!("logive,{v},{x},{:.17e}", log_ive_scalar(v, x));
    }
    for &(m1, m2) in &[(8.0, 5.0), (20.0, 3.0), (2.0, 2.0)] {
        let d = Skellam::new(m1, m2);
        for &k in &[0_i64, 5, 50, 100, 267, 500] {
            println!("skellam,{m1},{m2},{k},{:.17e}", d.logpmf(k));
        }
    }
}

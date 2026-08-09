use fsci_stats::*;
fn main() {
    for &k in &[0.5, 1.0, 2.0, 5.0, 10.0, 50.0] {
        let d = VonMises::new(k, 0.0);
        // Sweep bounds quoted to 5 decimals, not an attempt to write PI
        // (frankenscipy-023vy): substituting the constant would shift the
        // sampled grid and change every emitted differential row.
        #[allow(clippy::approx_constant)]
        let (lo, hi) = (-3.14159_f64, 3.14159_f64);
        let mut x = lo;
        while x < hi {
            println!("vm,{k},{x:.6},{:.16e}", d.cdf(x));
            x += 0.25;
        }
    }
}

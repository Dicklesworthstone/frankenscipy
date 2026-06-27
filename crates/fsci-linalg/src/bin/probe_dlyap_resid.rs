use fsci_linalg::*;
fn main() {
    let mut seed = 9u64;
    let mut r = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let mut worst = 0.0f64;
    for &n in &[3usize, 6, 12, 30, 60] {
        // scale A so spectral radius <1 for a well-posed Stein eq
        let a: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..n).map(|_| 0.1 * r()).collect())
            .collect();
        let q: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| r()).collect()).collect();
        let x = solve_discrete_lyapunov(&a, &q, DecompOptions::default()).unwrap();
        let mut mx = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                // (A X A^T)[i][j] - X[i][j] + Q[i][j]
                let mut axat = 0.0;
                for k in 0..n {
                    let mut xat = 0.0;
                    for l in 0..n {
                        xat += x[k][l] * a[j][l];
                    }
                    axat += a[i][k] * xat;
                }
                mx = mx.max((axat - x[i][j] + q[i][j]).abs());
            }
        }
        println!("n={n}: max|AXA^T-X+Q| = {mx:.3e}");
        worst = worst.max(mx);
    }
    println!("WORST={worst:.3e}");
}

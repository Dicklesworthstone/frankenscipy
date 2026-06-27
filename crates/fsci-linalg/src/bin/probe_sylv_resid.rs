use fsci_linalg::*;
fn main() {
    let mut seed = 5u64;
    let mut r = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let mut worst = 0.0f64;
    for &n in &[3usize, 6, 12, 30, 60] {
        let a: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| r()).collect()).collect();
        let b: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| r()).collect()).collect();
        let q: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| r()).collect()).collect();
        let x = solve_sylvester(&a, &b, &q, DecompOptions::default()).unwrap();
        // residual AX + XB - Q
        let mut mx = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let mut ax = 0.0;
                for k in 0..n {
                    ax += a[i][k] * x[k][j];
                }
                let mut xb = 0.0;
                for k in 0..n {
                    xb += x[i][k] * b[k][j];
                }
                mx = mx.max((ax + xb - q[i][j]).abs());
            }
        }
        println!("n={n}: max|AX+XB-Q| = {mx:.3e}");
        worst = worst.max(mx);
    }
    println!("WORST={worst:.3e}");
}

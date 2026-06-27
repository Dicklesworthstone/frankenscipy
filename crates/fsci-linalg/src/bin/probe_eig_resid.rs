use fsci_linalg::*;
fn main() {
    let mut seed = 123u64;
    let mut r = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let mut worst = 0.0f64;
    for &n in &[5usize, 10, 25, 60, 120] {
        let a: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| r()).collect()).collect();
        let res = eig(&a, DecompOptions::default()).unwrap();
        // eigenvectors[row][col]; for complex pair col,col+1 = Re,Im of v
        let mut i = 0;
        let mut maxres = 0.0f64;
        while i < n {
            let im = res.eigenvalues_im[i];
            let re = res.eigenvalues_re[i];
            if im.abs() > 1e-12 {
                // complex pair: v = col i (Re) + i col i+1 (Im); check A v = lambda v (complex)
                for row in 0..n {
                    // (A v)_row real/imag
                    let mut ar = 0.0;
                    let mut ai = 0.0;
                    for k in 0..n {
                        ar += a[row][k] * res.eigenvectors[k][i];
                        ai += a[row][k] * res.eigenvectors[k][i + 1];
                    }
                    let vr = res.eigenvectors[row][i];
                    let vi = res.eigenvectors[row][i + 1];
                    // lambda*v = (re+im i)(vr+vi i) = (re*vr - im*vi) + (re*vi + im*vr) i
                    let lr = re * vr - im * vi;
                    let li = re * vi + im * vr;
                    maxres = maxres.max((ar - lr).abs()).max((ai - li).abs());
                }
                i += 2;
            } else {
                for row in 0..n {
                    let mut av = 0.0;
                    for k in 0..n {
                        av += a[row][k] * res.eigenvectors[k][i];
                    }
                    maxres = maxres.max((av - re * res.eigenvectors[row][i]).abs());
                }
                i += 1;
            }
        }
        println!("n={n}: max |Av-λv| = {maxres:.3e}");
        worst = worst.max(maxres);
    }
    println!("WORST = {worst:.3e}");
}

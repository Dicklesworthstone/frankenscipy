use fsci_fft::Complex64;
use fsci_ndimage::{fourier_gaussian, fourier_shift, fourier_uniform};
fn dump(name: &str, v: &[Complex64]) {
    for (i, c) in v.iter().enumerate() {
        println!("{name},{i},{:.17e},{:.17e}", c.0, c.1);
    }
}
fn main() {
    // deterministic complex input, shape 4x5 (row-major), n=-1 full FFT layout
    let mut input = Vec::new();
    for k in 0..20 {
        let t = k as f64;
        input.push(((0.5 * t).cos() + 0.2 * t, (0.3 * t).sin() - 0.1 * t));
    }
    let shape = [4usize, 5];
    dump("gauss", &fourier_gaussian(&input, &shape, &[1.5, 2.0]));
    dump("uniform", &fourier_uniform(&input, &shape, &[3.0, 2.0]));
    dump("shift", &fourier_shift(&input, &shape, &[1.0, -0.5]));
}

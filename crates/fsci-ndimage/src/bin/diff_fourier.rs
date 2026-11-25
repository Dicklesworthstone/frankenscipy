use fsci_fft::Complex64;
use fsci_ndimage::{fourier_ellipsoid, fourier_gaussian, fourier_shift, fourier_uniform};
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
    dump("ellipsoid", &fourier_ellipsoid(&input, &shape, &[3.0, 2.0]));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write;
    use std::process::Command;

    #[test]
    fn fourier_ellipsoid_matches_live_scipy() {
        let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
        let python = std::env::var("FSCI_PYTHON")
            .or_else(|_| std::env::var("SCIPY_PYTHON"))
            .unwrap_or_else(|_| "python3".into());
        let probe = Command::new(&python)
            .args([
                "-c",
                "import scipy, numpy, scipy.ndimage; print(scipy.__version__, numpy.__version__)",
            ])
            .output();
        let versions = match probe {
            Ok(output) if output.status.success() => {
                String::from_utf8(output.stdout).expect("SciPy version probe UTF-8")
            }
            unavailable => {
                assert!(
                    !required,
                    "required SciPy Fourier ellipsoid oracle is unavailable: {unavailable:?}"
                );
                eprintln!("SKIP Fourier ellipsoid: {unavailable:?}; compared cases = 0");
                return;
            }
        };
        eprintln!(
            "Fourier ellipsoid oracle: {python}, SciPy/NumPy {}",
            versions.trim()
        );
        if required {
            assert_eq!(versions.trim(), "1.17.1 2.4.3");
        }

        let cases: &[(&[usize], &[f64])] = &[
            (&[8, 8], &[2.0, 2.0]),
            (&[7, 9], &[3.5, 5.0]),
            (&[6, 11], &[0.0, 4.25]),
            (&[5, 6], &[0.0, 0.0]),
            (&[9, 10], &[1.0e-7, 3.0e-7]),
            (&[13, 8], &[20.0, 12.0]),
            (&[1, 9], &[3.0, 4.0]),
            // Controls for the unchanged 1-D interval and 3-D sphere kernels.
            (&[17], &[2.4]),
            (&[5, 6, 7], &[1.5, 4.0, 3.0]),
        ];
        let mut payload = String::new();
        for (index, &(shape, size)) in cases.iter().enumerate() {
            let shape = shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let size = size
                .iter()
                .map(|value| format!("{value:.17e}"))
                .collect::<Vec<_>>()
                .join(",");
            writeln!(payload, "{index}|{shape}|{size}").unwrap();
        }
        let script = r#"
import sys
import numpy as np
from scipy.ndimage import fourier_ellipsoid
for line in sys.argv[1].splitlines():
    ident, shape, size = line.split('|')
    shape = tuple(map(int, shape.split(',')))
    size = tuple(map(float, size.split(',')))
    data = np.array([complex((i % 11)*0.25 - 1.25, (i % 7)*0.125 - 0.375)
                     for i in range(int(np.prod(shape)))], dtype=np.complex128).reshape(shape)
    result = fourier_ellipsoid(data, size)
    assert np.isfinite(result).all()
    print(ident + '|' + ';'.join(format(float(z.real), '.17e') + ',' +
                               format(float(z.imag), '.17e') for z in result.ravel()))
"#;
        let output = Command::new(&python)
            .args(["-c", script, &payload])
            .output()
            .expect("launch SciPy Fourier ellipsoid oracle");
        assert!(
            output.status.success(),
            "SciPy Fourier ellipsoid failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8(output.stdout).expect("Fourier ellipsoid oracle UTF-8");
        let mut compared = 0;
        for (index, line) in text.lines().enumerate() {
            let (ident, values) = line.split_once('|').expect("oracle case and values");
            assert_eq!(ident.parse::<usize>().unwrap(), index);
            let &(shape, size) = cases.get(index).expect("unexpected oracle case");
            let expected: Vec<Complex64> = values
                .split(';')
                .map(|pair| {
                    let (real, imaginary) = pair.split_once(',').expect("complex oracle value");
                    (real.parse().unwrap(), imaginary.parse().unwrap())
                })
                .collect();
            let input: Vec<Complex64> = (0..shape.iter().product())
                .map(|i| {
                    (
                        (i % 11) as f64 * 0.25 - 1.25,
                        (i % 7) as f64 * 0.125 - 0.375,
                    )
                })
                .collect();
            let actual = fourier_ellipsoid(&input, shape, size);
            assert_eq!(actual.len(), expected.len());
            assert_eq!(actual[0], input[0], "DC coefficient, shape={shape:?}");
            let mut max_abs = 0.0_f64;
            for (coefficient, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                let error = (actual.0 - expected.0)
                    .abs()
                    .max((actual.1 - expected.1).abs());
                assert!(
                    actual.0.is_finite() && actual.1.is_finite() && error < 2.0e-13,
                    "shape={shape:?}, size={size:?}, coefficient={coefficient}: {actual:?} != {expected:?}"
                );
                max_abs = max_abs.max(error);
            }
            eprintln!("Fourier ellipsoid shape={shape:?}, size={size:?}, max_abs={max_abs:e}");
            compared += 1;
        }
        assert_eq!(compared, 9, "every Fourier ellipsoid case must be compared");
        eprintln!("Fourier ellipsoid compared cases = {compared}");
    }
}

#![forbid(unsafe_code)]

//! Multi-input feedback pole placement, including complex conjugate pairs.
//! The live oracle compares closed-loop spectra, since MIMO gains are nonunique.
//! FSCI_REQUIRE_SCIPY_ORACLE=1 requires SciPy 1.17.1 / NumPy 2.4.3.

use std::fmt::Write;
use std::process::Command;

use fsci_signal::{PlacePolesResult, SignalError, place_poles};

#[derive(Clone)]
struct Case {
    name: String,
    a: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
    poles: Vec<(f64, f64)>,
}

fn published_cases() -> Vec<Case> {
    vec![
        Case {
            name: "SciPy documentation real MIMO".into(),
            a: vec![
                vec![1.380, -0.2077, 6.715, -5.676],
                vec![-0.5814, -4.290, 0.0, 0.6750],
                vec![1.067, 4.273, -6.654, 5.893],
                vec![0.0480, 4.273, 1.343, -2.104],
            ],
            b: vec![
                vec![0.0, 5.679],
                vec![1.136, 1.136],
                vec![0.0, 0.0],
                vec![-3.146, 0.0],
            ],
            poles: vec![(-0.2, 0.0), (-0.5, 0.0), (-5.0566, 0.0), (-8.6659, 0.0)],
        },
        Case {
            name: "SciPy documentation complex MIMO".into(),
            a: vec![
                vec![0.0, 7.0 / 3.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 7.0 / 9.0],
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0],
            ],
            b: vec![
                vec![0.0, 0.0],
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
            ],
            poles: vec![
                (-1.0, 0.0),
                (-1.0 / 3.0, 0.0),
                (-2.0 / 3.0, -1.0 / 3.0),
                (-2.0 / 3.0, 1.0 / 3.0),
            ],
        },
    ]
}

fn double_integrators(poles: Vec<(f64, f64)>, name: &str) -> Case {
    Case {
        name: name.into(),
        a: vec![
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0],
        ],
        b: vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 1.0],
        ],
        poles,
    }
}

fn direct_and_repeated_cases() -> Vec<Case> {
    vec![
        Case {
            name: "fully actuated complex pair".into(),
            a: vec![vec![0.3, 1.2], vec![-0.7, 2.0]],
            b: vec![vec![2.0, 0.5], vec![-0.25, 1.0]],
            poles: vec![(-0.8, 1.3), (-0.8, -1.3)],
        },
        Case {
            name: "over-actuated real poles".into(),
            a: vec![vec![0.0, 1.0], vec![2.0, -0.5]],
            b: vec![vec![1.0, 0.0, 1.0], vec![0.0, 1.0, -2.0]],
            poles: vec![(-2.0, 0.0), (-0.7, 0.0)],
        },
        double_integrators(
            vec![(-1.0, 0.0), (-2.0, 0.0), (-1.0, 0.0), (-2.0, 0.0)],
            "real multiplicity equals input rank",
        ),
        double_integrators(
            vec![(-1.0, 0.7), (-1.0, -0.7), (-1.0, -0.7), (-1.0, 0.7)],
            "complex multiplicity equals input rank",
        ),
        Case {
            name: "single input real poles".into(),
            a: vec![vec![0.0, 1.0], vec![0.0, 0.0]],
            b: vec![vec![0.0], vec![1.0]],
            poles: vec![(-2.0, 0.0), (-3.0, 0.0)],
        },
        Case {
            name: "single input complex pair".into(),
            a: vec![vec![0.0, 1.0], vec![0.0, 0.0]],
            b: vec![vec![0.0], vec![1.0]],
            poles: vec![(-1.0, 1.0), (-1.0, -1.0)],
        },
    ]
}

fn generated_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for n in 3..=7 {
        for inputs in 2..=3.min(n - 1) {
            for complex in [false, true] {
                // An upper-bidiagonal plant with an input in its final state is
                // controllable; extra inputs exercise different feasible spaces.
                let mut a = vec![vec![0.0; n]; n];
                let mut b = vec![vec![0.0; inputs]; n];
                for i in 0..n {
                    a[i][i] = 0.13 * (i + 1) as f64;
                    if i + 1 < n {
                        a[i][i + 1] = 0.7 + 0.1 * i as f64;
                    }
                }
                for input in 0..inputs {
                    b[n - 1 - input][input] = 1.0 + 0.2 * input as f64;
                }
                let mut poles = Vec::new();
                if complex {
                    for pair in 0..n / 2 {
                        let re = -0.5 - 0.35 * pair as f64;
                        let im = 0.4 + 0.23 * pair as f64;
                        poles.push((re, im));
                        poles.push((re, -im));
                    }
                    if n % 2 == 1 {
                        poles.push((-2.7, 0.0));
                    }
                    // Conjugate pairs need not be adjacent at the public API.
                    poles.rotate_left(1);
                } else {
                    poles.extend((0..n).map(|i| (-0.4 - 0.45 * i as f64, 0.0)));
                    poles.reverse();
                }
                cases.push(Case {
                    name: format!("generated n={n} inputs={inputs} complex={complex}"),
                    a,
                    b,
                    poles,
                });
            }
        }
    }
    cases
}

fn assert_same_poles(actual: &[(f64, f64)], expected: &[(f64, f64)], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    let mut unmatched = actual.to_vec();
    for &(re, im) in expected {
        let (index, distance) = unmatched
            .iter()
            .enumerate()
            .map(|(i, &(r, j))| (i, (r - re).hypot(j - im)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        assert!(
            distance <= 2e-7 * 1.0_f64.max(re.hypot(im)),
            "{label}: requested ({re}, {im}), actual {actual:?}, distance {distance}"
        );
        unmatched.remove(index);
    }
}

fn check_case(case: &Case) -> PlacePolesResult {
    let result = place_poles(&case.a, &case.b, &case.poles)
        .unwrap_or_else(|error| panic!("{}: {error}", case.name));
    let n = case.a.len();
    assert_eq!(result.gain_matrix.len(), case.b[0].len(), "{}", case.name);
    assert!(result.gain_matrix.iter().all(|row| row.len() == n));
    assert!(result.gain_matrix.iter().flatten().all(|value| value.is_finite()));
    assert_same_poles(&result.computed_poles, &case.poles, &case.name);
    let closed_loop: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    case.a[i][j]
                        - (0..case.b[0].len())
                            .map(|input| case.b[i][input] * result.gain_matrix[input][j])
                            .sum::<f64>()
                })
                .collect()
        })
        .collect();
    let independent = fsci_linalg::eig(&closed_loop, fsci_linalg::DecompOptions::default())
        .unwrap();
    let spectrum: Vec<_> = independent
        .eigenvalues_re
        .into_iter()
        .zip(independent.eigenvalues_im)
        .collect();
    assert_same_poles(&spectrum, &case.poles, &case.name);
    result
}

#[test]
fn published_scipy_mimo_examples_place_real_and_complex_poles() {
    for case in published_cases() {
        let result = check_case(&case);
        let repeated = place_poles(&case.a, &case.b, &case.poles).unwrap();
        assert_eq!(result.gain_matrix, repeated.gain_matrix, "deterministic gain");
    }
}

#[test]
fn full_rank_over_actuated_and_repeated_poles_are_supported() {
    for case in direct_and_repeated_cases().into_iter().take(4) {
        check_case(&case);
    }
}

#[test]
fn single_input_gains_are_preserved_and_redundant_inputs_use_minimum_norm() {
    let cases = direct_and_repeated_cases();
    for (case, expected) in cases[4..].iter().zip([vec![6.0, 5.0], vec![2.0, 2.0]]) {
        let result = check_case(case);
        for (actual, expected) in result.gain_matrix[0].iter().zip(&expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
        let mut redundant = case.clone();
        redundant.name.push_str(" with redundant input");
        redundant.b = vec![vec![0.0, 0.0], vec![1.0, 2.0]];
        let result = check_case(&redundant);
        for j in 0..2 {
            assert!((result.gain_matrix[0][j] - expected[j] / 5.0).abs() < 2e-12);
            assert!((result.gain_matrix[1][j] - 2.0 * expected[j] / 5.0).abs() < 2e-12);
        }
    }
}

#[test]
fn generated_controllable_plants_accept_multiple_inputs_and_conjugate_pairs() {
    let cases = generated_cases();
    assert_eq!(cases.len(), 18);
    for case in cases {
        check_case(&case);
    }
}

#[test]
fn rejects_uncontrollable_and_zero_rank_plants() {
    let a = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 2.0, 0.0],
        vec![0.0, 0.0, 3.0],
    ];
    let poles = [(-1.0, 0.0), (-2.0, 0.0), (-3.0, 0.0)];
    for b in [
        vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0]],
        vec![vec![0.0, 0.0]; 3],
        vec![vec![0.0]; 3],
    ] {
        assert!(
            matches!(place_poles(&a, &b, &poles), Err(SignalError::NumericalFailure(_))),
            "uncontrollable B={b:?}"
        );
    }
}

#[test]
fn validates_shapes_finiteness_conjugacy_and_rank_multiplicity() {
    let a = vec![vec![0.0, 1.0], vec![0.0, 0.0]];
    let b = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
    let poles = [(-1.0, 0.0), (-2.0, 0.0)];
    assert!(place_poles(&[], &[], &[]).is_err());
    assert!(place_poles(&[vec![0.0], vec![1.0]], &b, &poles).is_err());
    assert!(place_poles(&a, &b[..1], &poles).is_err());
    assert!(place_poles(&a, &[vec![0.0], vec![1.0, 2.0]], &poles).is_err());
    assert!(place_poles(&a, &[vec![], vec![]], &poles).is_err());
    assert!(place_poles(&a, &b, &poles[..1]).is_err());
    for bad in [
        vec![(-1.0, 1.0), (-2.0, -1.0)],
        vec![(-1.0, 1.0), (-1.0, 1.0)],
        vec![(-1.0, 1.0), (-1.0, 0.0)],
    ] {
        assert!(matches!(
            place_poles(&a, &b, &bad),
            Err(SignalError::InvalidParameter { .. })
        ));
        assert!(matches!(
            place_poles(&a, &[vec![0.0], vec![1.0]], &bad),
            Err(SignalError::InvalidParameter { .. })
        ));
    }
    for nonfinite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut invalid_a = a.clone();
        invalid_a[0][0] = nonfinite;
        assert!(matches!(
            place_poles(&invalid_a, &b, &poles),
            Err(SignalError::NonFiniteInput { .. })
        ));
        let mut invalid_b = b.clone();
        invalid_b[0][0] = nonfinite;
        assert!(matches!(
            place_poles(&a, &invalid_b, &poles),
            Err(SignalError::NonFiniteInput { .. })
        ));
        for bad in [[(nonfinite, 0.0), poles[1]], [(-1.0, nonfinite), poles[1]]] {
            assert!(matches!(
                place_poles(&a, &b, &bad),
                Err(SignalError::NonFiniteInput { .. })
            ));
        }
    }
    // Two physical input columns still have numerical rank one.
    let rank_one_b = vec![vec![0.0, 0.0], vec![1.0, 2.0]];
    assert!(matches!(
        place_poles(&a, &rank_one_b, &[(-1.0, 0.0), (-1.0, 0.0)]),
        Err(SignalError::InvalidParameter { .. })
    ));
    assert!(matches!(
        place_poles(&a, &[vec![0.0], vec![1.0]], &[(-1.0, 0.0), (-1.0, 0.0)]),
        Err(SignalError::InvalidParameter { .. })
    ));
}

#[test]
fn closed_loop_spectra_match_live_scipy_for_every_generated_case() {
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".into());
    let available = Command::new(&python)
        .args(["-c", "import scipy; import numpy"])
        .output()
        .is_ok_and(|output| output.status.success());
    if !available {
        assert!(!required, "required SciPy oracle unavailable via {python}");
        eprintln!("SKIP multi-input pole placement: missing SciPy; compared cases = 0");
        return;
    }
    let mut cases = published_cases();
    cases.extend(direct_and_repeated_cases());
    cases.extend(generated_cases());
    assert_eq!(cases.len(), 26);
    let results: Vec<_> = cases.iter().map(check_case).collect();
    // A plain numeric protocol avoids adding a JSON dependency to fsci-signal.
    let mut input = format!("{}\n", cases.len());
    for (case, result) in cases.iter().zip(&results) {
        writeln!(input, "{} {}", case.a.len(), case.b[0].len()).unwrap();
        for value in case.a.iter().flatten().chain(case.b.iter().flatten()) {
            write!(input, "{value:.17e} ").unwrap();
        }
        for &(re, im) in &case.poles {
            write!(input, "{re:.17e} {im:.17e} ").unwrap();
        }
        for value in result.gain_matrix.iter().flatten() {
            write!(input, "{value:.17e} ").unwrap();
        }
        input.push('\n');
    }
    let script = r#"
import os, sys
import numpy as np
import scipy
from scipy.signal import place_poles
if os.environ.get("FSCI_REQUIRE_SCIPY_ORACLE") == "1":
    assert (scipy.__version__, np.__version__) == ("1.17.1", "2.4.3")
tokens = iter(sys.argv[1].split())
count = int(next(tokens))
print("VERSIONS", scipy.__version__, np.__version__)
print("CASES", count)
def matrix(rows, cols):
    return np.array([float(next(tokens)) for _ in range(rows*cols)]).reshape(rows, cols)
for case in range(count):
    n, m = int(next(tokens)), int(next(tokens))
    A, B = matrix(n, n), matrix(n, m)
    parts = matrix(n, 2)
    poles = parts[:, 0] + 1j*parts[:, 1]
    gain = matrix(m, n)
    assert np.isfinite(gain).all()
    controllability = np.column_stack([np.linalg.matrix_power(A, k) @ B for k in range(n)])
    assert np.linalg.matrix_rank(controllability) == n, (case, "invalid test plant")
    reference = place_poles(A, B, poles)
    achieved = np.linalg.eigvals(A - B @ gain)
    if m == 1:
        np.testing.assert_allclose(gain, reference.gain_matrix, rtol=1e-10, atol=1e-10)
    print("CASE", case)
    for name, values in [("REFERENCE", reference.computed_poles), ("ACHIEVED", achieved)]:
        print(name, *[format(float(part), ".17g") for value in values for part in (value.real, value.imag)])
assert next(tokens, None) is None
"#;
    let output = Command::new(&python)
        .args(["-c", script, &input])
        .output()
        .expect("launch live pole-placement SciPy oracle");
    assert!(
        output.status.success(),
        "SciPy pole-placement oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output_text = String::from_utf8(output.stdout).unwrap();
    let mut tokens = output_text.split_whitespace();
    assert_eq!(tokens.next(), Some("VERSIONS"));
    let scipy = tokens.next().unwrap();
    let numpy = tokens.next().unwrap();
    if required {
        assert_eq!((scipy, numpy), ("1.17.1", "2.4.3"));
    }
    assert_eq!(tokens.next(), Some("CASES"));
    assert_eq!(tokens.next().unwrap().parse::<usize>().unwrap(), cases.len());
    let mut compared = 0;
    for (index, (case, result)) in cases.iter().zip(&results).enumerate() {
        assert_eq!(tokens.next(), Some("CASE"));
        assert_eq!(tokens.next().unwrap().parse::<usize>().unwrap(), index);
        for label in ["REFERENCE", "ACHIEVED"] {
            assert_eq!(tokens.next(), Some(label));
            let reference: Vec<(f64, f64)> = (0..case.a.len())
                .map(|_| {
                    (
                        tokens.next().unwrap().parse().unwrap(),
                        tokens.next().unwrap().parse().unwrap(),
                    )
                })
                .collect();
            assert_same_poles(&reference, &case.poles, &format!("{} {label}", case.name));
            assert_same_poles(&result.computed_poles, &reference, &case.name);
        }
        compared += 1;
    }
    assert!(tokens.next().is_none(), "unexpected oracle trailing output");
    assert_eq!(compared, 26, "every generated case must be compared");
    eprintln!(
        "pole placement: compared {compared} cases against SciPy {scipy} / NumPy {numpy}"
    );
}

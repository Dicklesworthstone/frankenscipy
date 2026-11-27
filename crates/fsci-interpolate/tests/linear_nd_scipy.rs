#![forbid(unsafe_code)]

//! Scattered N-D linear interpolation against live SciPy. Run with
//! FSCI_REQUIRE_SCIPY_ORACLE=1 and FSCI_PYTHON pointing at SciPy 1.17.1 / NumPy 2.4.3.

use std::fmt::Write as _;
use std::io::Write as _;
use std::process::{Command, Stdio};

use fsci_interpolate::{GriddataMethod, LinearNDInterpolator, LinearNDOptions, griddata};

#[test]
fn linear_nd_rescale_handles_independent_units_and_custom_fill() {
    let origin = [2e-9, -2e6, 3e-4];
    let spans = [1e-9, 1e6, 1e-4];
    let mut points = vec![origin.to_vec()];
    for axis in 0..3 {
        let mut point = origin.to_vec();
        point[axis] += spans[axis];
        points.push(point);
    }
    let interpolator = LinearNDInterpolator::with_options(
        &points,
        &[2.0, 3.0, 5.0, -1.0],
        LinearNDOptions {
            fill_value: -123.5,
            rescale: true,
        },
    )
    .unwrap();
    let query: Vec<f64> = (0..3)
        .map(|axis| origin[axis] + [0.2, 0.3, 0.4][axis] * spans[axis])
        .collect();
    let value = interpolator.eval(&query).unwrap();
    assert!((value - 1.9).abs() < 2e-12, "rescaled value: {value:.17e}");
    let batch = interpolator.eval_many(&[query]).unwrap();
    assert_eq!(batch, vec![value]);
    let outside = [origin[0] - spans[0], origin[1], origin[2]];
    assert_eq!(interpolator.eval(&outside).unwrap(), -123.5);
    assert_eq!(interpolator.eval(&[f64::NAN, 0.0, 0.0]).unwrap(), -123.5);
}

#[test]
fn linear_nd_keeps_nonfinite_values_as_data() {
    for ndim in [2, 3] {
        let mut points = vec![vec![0.0; ndim]];
        for axis in 0..ndim {
            let mut point = vec![0.0; ndim];
            point[axis] = 1.0;
            points.push(point);
        }
        for exceptional in [f64::NAN, f64::INFINITY] {
            let mut values = vec![1.0; ndim + 1];
            values[0] = exceptional;
            let interpolator = LinearNDInterpolator::new(&points, &values).unwrap();
            let query = vec![0.5 / ndim as f64; ndim];
            let actual = interpolator.eval(&query).unwrap();
            if exceptional.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual, f64::INFINITY);
            }
        }
    }
}

struct Case {
    label: String,
    points: Vec<Vec<f64>>,
    values: Vec<f64>,
    queries: Vec<Vec<f64>>,
    rescale: bool,
    fill_value: f64,
}

fn lcg_points(count: usize, ndim: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            (0..ndim)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    (state >> 11) as f64 / (1u64 << 53) as f64
                })
                .collect()
        })
        .collect()
}

fn nonlinear_case(ndim: usize, seed: u64, rescale: bool) -> Case {
    let mut points = lcg_points(22 + 2 * ndim, ndim, seed);
    let values = points
        .iter()
        .map(|point| {
            point
                .iter()
                .enumerate()
                .map(|(axis, coordinate)| {
                    (axis as f64 + 1.0) * coordinate.sin() + coordinate * coordinate
                })
                .sum::<f64>()
                + point[0] * point[1]
        })
        .collect();
    let mut queries: Vec<Vec<f64>> = lcg_points(64, ndim, seed ^ 0x5bd1_e995)
        .into_iter()
        .map(|point| point.into_iter().map(|x| 1.5 * x - 0.25).collect())
        .collect();
    // Guaranteed interior queries across the cloud, independently of either triangulator.
    for index in 0..32 {
        let indices = [
            index % points.len(),
            (7 * index + 3) % points.len(),
            (11 * index + 5) % points.len(),
        ];
        queries.push(
            (0..ndim)
                .map(|axis| {
                    0.2 * points[indices[0]][axis]
                        + 0.3 * points[indices[1]][axis]
                        + 0.5 * points[indices[2]][axis]
                })
                .collect(),
        );
    }
    queries.extend(points.iter().cloned());
    queries.push(vec![-1.0; ndim]);
    queries.push(vec![2.0; ndim]);
    queries.push(vec![f64::NAN; ndim]);
    queries.push(vec![f64::INFINITY; ndim]);

    if rescale {
        // Independent physical units, including ranges far below one. Coordinates
        // remain well-conditioned relative to their own offsets.
        let spans = [1e-9, 1e4, 0.125, 1e8, 1e-5];
        for point in points.iter_mut().chain(queries.iter_mut()) {
            for (axis, coordinate) in point.iter_mut().enumerate() {
                *coordinate = spans[axis] * (*coordinate + axis as f64 - 2.5);
            }
        }
    }
    Case {
        label: format!("{ndim}d_seed{seed}_rescale{rescale}"),
        points,
        values,
        queries,
        rescale,
        fill_value: if rescale { -123.5 } else { f64::NAN },
    }
}

fn scipy_reference(cases: &[Case]) -> Option<Vec<Vec<f64>>> {
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let available = Command::new(&python)
        .args(["-c", "import scipy; import numpy"])
        .output()
        .is_ok_and(|output| output.status.success());
    if !available {
        assert!(!required, "required SciPy oracle unavailable via {python}");
        eprintln!("SKIP N-D interpolation: missing SciPy; compared cases = 0");
        return None;
    }

    // Text floats round-trip exactly and avoid adding serialization dependencies
    // to the interpolation crate for this oracle.
    let mut input = format!("{}\n", cases.len());
    for case in cases {
        writeln!(
            input,
            "{} {} {} {} {:.17e}",
            case.points.len(),
            case.points[0].len(),
            case.queries.len(),
            usize::from(case.rescale),
            case.fill_value
        )
        .unwrap();
        for (point, value) in case.points.iter().zip(&case.values) {
            for coordinate in point {
                write!(input, "{coordinate:.17e} ").unwrap();
            }
            writeln!(input, "{value:.17e}").unwrap();
        }
        for query in &case.queries {
            for coordinate in query {
                write!(input, "{coordinate:.17e} ").unwrap();
            }
            input.push('\n');
        }
    }
    let script = r#"
import os
import sys
import numpy as np
import scipy
from scipy.interpolate import LinearNDInterpolator
if os.environ.get("FSCI_REQUIRE_SCIPY_ORACLE") == "1":
    assert (scipy.__version__, np.__version__) == ("1.17.1", "2.4.3")
tokens = iter(sys.stdin.read().split())
count = int(next(tokens))
print(scipy.__version__, np.__version__)
for case in range(count):
    n, d, nq, rescale = (int(next(tokens)) for _ in range(4))
    fill = float(next(tokens))
    data = np.array([[float(next(tokens)) for _ in range(d+1)] for _ in range(n)])
    queries = np.array([[float(next(tokens)) for _ in range(d)] for _ in range(nq)])
    result = LinearNDInterpolator(data[:, :d], data[:, d], fill_value=fill,
                                  rescale=bool(rescale))(queries)
    print(" ".join(format(float(value), ".17e") for value in result))
assert next(tokens, None) is None
"#;
    let mut child = Command::new(&python)
        .args(["-c", script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("launch N-D interpolation SciPy oracle");
    {
        let mut stdin = child.stdin.take().unwrap();
        stdin.write_all(input.as_bytes()).unwrap();
    }
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "N-D interpolation SciPy oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut lines = stdout.lines();
    let versions: Vec<&str> = lines
        .next()
        .expect("oracle versions")
        .split_whitespace()
        .collect();
    assert_eq!(versions.len(), 2);
    if required {
        assert_eq!(versions[0], "1.17.1");
        assert_eq!(versions[1], "2.4.3");
    }
    let mut reference = Vec::with_capacity(cases.len());
    for case in cases {
        let row: Vec<f64> = lines
            .next()
            .unwrap_or_else(|| panic!("missing oracle row for {}", case.label))
            .split_whitespace()
            .map(|value| value.parse().expect("oracle float"))
            .collect();
        assert_eq!(row.len(), case.queries.len(), "{}", case.label);
        reference.push(row);
    }
    assert!(lines.next().is_none(), "unexpected extra oracle rows");
    Some(reference)
}

fn compare(actual: f64, expected: f64, label: &str, index: usize, api: &str) {
    if expected.is_nan() {
        assert!(
            actual.is_nan(),
            "{label}/{api} query {index}: expected NaN, got {actual}"
        );
    } else {
        assert!(
            actual.is_finite() && (actual - expected).abs() <= 5e-11 * expected.abs().max(1.0),
            "{label}/{api} query {index}: got {actual:.17e}, SciPy {expected:.17e}"
        );
    }
}

#[test]
fn linear_nd_and_griddata_match_live_scipy_on_nonlinear_clouds() {
    let mut cases = Vec::new();
    for ndim in 2..=5 {
        for seed in [1000 + ndim as u64, 2000 + ndim as u64] {
            cases.push(nonlinear_case(ndim, seed, false));
        }
        cases.push(nonlinear_case(ndim, 3000 + ndim as u64, true));
    }
    let Some(reference) = scipy_reference(&cases) else {
        return;
    };
    let mut compared = 0;
    let mut inside = 0;
    let mut outside = 0;
    for (case, expected) in cases.iter().zip(&reference) {
        let interpolator = LinearNDInterpolator::with_options(
            &case.points,
            &case.values,
            LinearNDOptions {
                fill_value: case.fill_value,
                rescale: case.rescale,
            },
        )
        .unwrap_or_else(|error| panic!("{}: {error}", case.label));
        let batch = interpolator.eval_many(&case.queries).unwrap();
        let grid = if case.rescale {
            None
        } else {
            Some(
                griddata(
                    &case.points,
                    &case.values,
                    &case.queries,
                    GriddataMethod::Linear,
                )
                .unwrap(),
            )
        };
        for (index, (&got, &want)) in batch.iter().zip(expected).enumerate() {
            compare(got, want, &case.label, index, "batch");
            compare(
                interpolator.eval(&case.queries[index]).unwrap(),
                want,
                &case.label,
                index,
                "scalar",
            );
            compared += 2;
            if let Some(grid) = &grid {
                compare(grid[index], want, &case.label, index, "griddata");
                compared += 1;
            }
            if want.is_nan() || want == case.fill_value {
                outside += 1;
            } else {
                inside += 1;
            }
        }
    }
    assert_eq!(cases.len(), 12);
    assert!(compared > 3000 && inside > 500 && outside > 100);
    eprintln!(
        "N-D linear interpolation: {} clouds, {compared} comparisons, \
         {inside} interior/data-site and {outside} outside-hull queries",
        cases.len()
    );
}

#[test]
fn griddata_one_dimensional_linear_and_cubic_match_live_scipy() {
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let available = Command::new(&python)
        .args(["-c", "import scipy; import numpy"])
        .output()
        .is_ok_and(|output| output.status.success());
    if !available {
        assert!(!required, "required SciPy oracle unavailable via {python}");
        eprintln!("SKIP one-dimensional griddata: missing SciPy; compared cases = 0");
        return;
    }
    let script = r#"
import os
import numpy as np
import scipy
from scipy.interpolate import griddata
if os.environ.get("FSCI_REQUIRE_SCIPY_ORACLE") == "1":
    assert (scipy.__version__, np.__version__) == ("1.17.1", "2.4.3")
x = np.array([2., -1., 0., 3., 1.])
y = np.sin(.7*x) + x*x - 1.
q = np.array([-1.5, -1., -.75, -.1, 0., .33, .5, 1., 1.75, 2., 2.8, 3., 3.5])
print(scipy.__version__, np.__version__)
for method in ("linear", "cubic"):
    result = griddata(x, y, q, method=method)
    print(" ".join(format(float(value), ".17e") for value in result))
"#;
    let output = Command::new(&python)
        .args(["-c", script])
        .output()
        .expect("launch one-dimensional griddata oracle");
    assert!(
        output.status.success(),
        "one-dimensional griddata oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut tokens = stdout.split_whitespace();
    let scipy_version = tokens.next().expect("SciPy version");
    let numpy_version = tokens.next().expect("NumPy version");
    if required {
        assert_eq!(scipy_version, "1.17.1");
        assert_eq!(numpy_version, "2.4.3");
    }
    let points: Vec<Vec<f64>> = [2.0, -1.0, 0.0, 3.0, 1.0]
        .iter()
        .map(|&x| vec![x])
        .collect();
    let values: Vec<f64> = points
        .iter()
        .map(|point| (0.7 * point[0]).sin() + point[0] * point[0] - 1.0)
        .collect();
    let queries: Vec<Vec<f64>> = [
        -1.5, -1.0, -0.75, -0.1, 0.0, 0.33, 0.5, 1.0, 1.75, 2.0, 2.8, 3.0, 3.5,
    ]
    .iter()
    .map(|&x| vec![x])
    .collect();
    let mut compared = 0;
    for method in [GriddataMethod::Linear, GriddataMethod::Cubic] {
        let actual = griddata(&points, &values, &queries, method).unwrap();
        for (index, value) in actual.into_iter().enumerate() {
            let expected = tokens
                .next()
                .expect("oracle value")
                .parse::<f64>()
                .expect("oracle float");
            compare(
                value,
                expected,
                "one-dimensional",
                index,
                &format!("{method:?}"),
            );
            compared += 1;
        }
    }
    assert!(tokens.next().is_none());
    assert_eq!(compared, 26);
    eprintln!("one-dimensional griddata: {compared} pinned-SciPy comparisons");
}
